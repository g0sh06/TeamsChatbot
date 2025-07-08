from rag import RAGSystem
from extractText import get_all_texts
import os

from rag import RAGSystem
from extractText import get_all_texts
import os
from dotenv import load_dotenv

def main():
    try:
        # Load environment variables
        load_dotenv()
        documents_folder = os.getenv('FOLDER', 'documents')  # Default to 'documents' if not set
        
        rag = RAGSystem()
        
        # Load and index documents from the folder specified in .env
        if os.path.exists(documents_folder):
            documents = get_all_texts(documents_folder)
            if documents:
                rag.index_documents(documents)
                print(f"Documents successfully indexed from '{documents_folder}'.")
            else:
                print(f"Warning: No documents found in '{documents_folder}' folder")
        else:
            print(f"Error: Folder '{documents_folder}' does not exist")
            print("Please create the folder and add documents, or update FOLDER in .env")
            return
        
        print("RAG System Initialized. Type 'exit' to quit.\n")
        
        while True:
            user_input = input("Question: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = rag.answer_question(user_input)
            print("\nResponse:")
            print(response)
            print("\n" + "="*80 + "\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()