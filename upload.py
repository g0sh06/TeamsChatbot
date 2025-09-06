import streamlit as st
import os
import shutil

from extractText import create_vector_db_from_files

def main():
    st.set_page_config(page_title="Document Management")
    st.title("Document Management")
    
    # Document upload section
    uploaded_files = st.file_uploader(
        "Upload course materials (PDF)", 
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("uploaded_docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join("uploaded_docs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

        if st.button("Process Documents"):
            with st.spinner("Creating knowledge base..."):
                try:
                    create_vector_db_from_files([os.path.join("uploaded_docs", f.name) for f in uploaded_files])
                    st.success("Knowledge base updated successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()