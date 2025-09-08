# Standard library imports
import os
import shutil
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# Environment and API configuration
from dotenv import load_dotenv

# LangChain core components
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from model import gpt4all_embeddings

load_dotenv()

CHROMA_PATH = "database"

# Global variables that will be initialized lazily
_db = None
_retriever = None

def initialize_database():
    """Initialize the database only when needed"""
    global _db, _retriever
    
    if _db is not None:
        return _db, _retriever
    
    try:
        # Check if database exists and has files
        if (os.path.exists(CHROMA_PATH) and 
            any(fname.endswith('.faiss') or fname.endswith('.pkl') 
                for fname in os.listdir(CHROMA_PATH))):
            
            _db = FAISS.load_local(
                CHROMA_PATH,
                gpt4all_embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Loaded existing FAISS database")
        else:
            # Create empty database if none exists
            print("⚠️ No FAISS database found. Creating empty database...")
            os.makedirs(CHROMA_PATH, exist_ok=True)
            empty_docs = [Document(page_content="No documents processed yet.", metadata={})]
            _db = FAISS.from_documents(empty_docs, gpt4all_embeddings)
            _db.save_local(CHROMA_PATH)
            print("✅ Created empty FAISS database")
        
        # Create retriever
        _retriever = _db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.8
            }
        )
        
        return _db, _retriever
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        # Fallback: create empty database in memory
        empty_docs = [Document(page_content="No documents processed yet.", metadata={})]
        _db = FAISS.from_documents(empty_docs, gpt4all_embeddings)
        _retriever = _db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        )
        return _db, _retriever

def get_retriever():
    """Get the retriever, initializing if necessary"""
    global _retriever
    if _retriever is None:
        _, _retriever = initialize_database()
    return _retriever

llm = OllamaLLM(
    model="mistral",
    temperature=0.2,
    base_url="http://localhost:11434"
)

def contextualize_question():
    """Create a history-aware retriever"""
    question_reformulation_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    question_reformulation_template = ChatPromptTemplate.from_messages(
        [
            ("system", question_reformulation_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = get_retriever()  # Use the lazy-loaded retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, question_reformulation_template 
    )
    
    return history_aware_retriever

def answer_question():
    """Creates a RAG chain"""
    answer_question_prompt = """ 
    You are a helpful assistant. Use ONLY the context provided below to answer.

    Rules:
    - If the context is empty, say: "I couldn't find that information based on the course materials."
    - DO NOT guess or fabricate anything.
    - DO NOT rely on prior memory or general knowledge.
    - Your answer must be based 100% on the retrieved context only.

    Context: {context}
    """
    
    answer_question_template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    answer_question_chain = create_stuff_documents_chain(llm, answer_question_template)
    history_aware_retriever = contextualize_question()
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)
    
    return rag_chain

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    rag_chain = answer_question()
    response = rag_chain.invoke(state)

    retrieved_docs = response.get("context")
    answer = response.get("answer")

    if not retrieved_docs or all(doc.page_content.strip() == "" for doc in retrieved_docs):
        answer = "I couldn't find that information based on the course materials."

    print("\n🟡 USER QUESTION:", state["input"])
    print("\n🟩 CONTEXT RETRIEVED FROM CHROMA:\n", retrieved_docs)
    print("\n🟦 MODEL RESPONSE:\n", answer)

    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(answer),
        ],
        "context": retrieved_docs,
        "answer": answer,
    }

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def execute_user_query(query_text):
    """Execute the QA workflow"""
    config = {"configurable": {"thread_id": "abc123"}}
    
    result = app.invoke(
        {"input": query_text},
        config=config,
    )
    
    return result["answer"]

if __name__ == "__main__":
    get_retriever()
    execute_user_query("test query")