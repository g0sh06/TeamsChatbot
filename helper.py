# Standard library imports
import os
import shutil
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# Environment and API configuration
from dotenv import load_dotenv

# LangChain core components
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaLLM
# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from model import gpt4all_embeddings

load_dotenv()

CHROMA_PATH = "database"

db = Chroma(persist_directory=CHROMA_PATH,
            embedding_function=gpt4all_embeddings)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 25,
        "lambda_mult": 0.6
    }
)

llm = OllamaLLM(
    model="mistral",
    temperature=0.2,
    base_url="http://localhost:11434"  # Point to local Ollama
)
    
def contextualize_question():
    """
    Create a history-aware retriever to reformulate user queries into standalone questions.

    This function generates a retriever that uses a language model (LLM) to reformulate a user's 
    latest question into a standalone question, ensuring it is understandable without requiring 
    the context of prior chat history. The retriever is configured using a specified prompt 
    template and associated components.

    Returns
    -------
    history_aware_retriever : object
        A retriever object capable of reformulating user queries into standalone questions 
        by leveraging the provided language model, retriever, and prompt.
    """
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

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, question_reformulation_template 
    )
    
    return history_aware_retriever

def answer_question():
    """
    Creates a Retrieval-Augmented Generation (RAG) chain to answer user questions 
    by leveraging a history-aware retriever and a question-answering chain.

    This function combines context retrieval and answer generation:
    - Reformulates user queries into standalone questions if necessary.
    - Retrieves relevant context from a knowledge base or vector database.
    - Generates concise, depthful answers based on the retrieved context.

    Returns
    -------
    rag_chain : RetrievalAugmentedGenerationChain
        A chain that reformulates questions, retrieves relevant context, and generates answers.
    """
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
    """
    Represents the application state for a conversational workflow.

    Attributes
    ----------
    input : str
        The latest user query or input.
    chat_history : Annotated[Sequence[BaseMessage], add_messages]
        A sequence of messages representing the chat history, including user and AI messages.
    context : str
        The retrieved context relevant to the current query.
    answer : str
        The generated response to the user's query.
    """
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


# Defines and compiles a stateful workflow for managing a conversational application.

# Attributes
# ----------
# workflow : StateGraph
#     A directed graph that defines the flow of tasks (nodes) and their connections (edges).
# memory : MemorySaver
#     A persistence mechanism that saves and restores the workflow's state in memory.
# app : CompiledStateGraph
#     The final compiled workflow, ready to execute with state management.

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def execute_user_query(query_text):
    """
    Executes the question-answering (QA) workflow with the provided query.

    Parameters
    ----------
    query_text : str
        The input query or question to be processed by the QA system.

    Returns
    -------
    str
        The generated answer to the input query.

    Notes
    -----
    - The function uses a precompiled `app` to execute the workflow.
    - A configuration dictionary is passed, which includes a `thread_id` for tracking.
    - The `app.invoke` method processes the query, retrieves relevant context, and generates a response.
    """
    config = {"configurable": {"thread_id": "abc123"}}
    
    result = app.invoke(
        {"input": query_text},
        config=config,
    )
    
    return result["answer"]


if __name__ == "__main__":
    execute_user_query(query_text)