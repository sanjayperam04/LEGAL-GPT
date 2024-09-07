import os
import logging
import streamlit as st
import gdown
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
os.environ["GROQ_API_KEY"] = "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"

# Google Drive file ID and destination path
file_id = 'your_google_drive_file_id'
destination = './chroma_langchain_db.zip'

def download_from_drive(file_id, destination):
    try:
        gdown.download(f"https://drive.google.com/drive/folders/1nE0a34uTKnMefvOuqLbBWlb-R5t_LDam?usp=sharing", destination, quiet=False)
        os.system(f'unzip {destination} -d ./')
        logging.info("Downloaded and extracted the vector store from Google Drive.")
    except Exception as e:
        logging.error(f"Failed to download from Google Drive: {e}")

def load_vector_store():
    try:
        persist_directory = "./chroma_langchain_db"
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        return vectorstore
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}")
        return None

def create_rag_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    prompt_template = """You are an AI assistant tasked with providing accurate information based on the given context. Your goal is to extract relevant information and present it clearly.

    Context:
    {context}

    Question: {question}

    Instructions:
    1. Provide an answer based on the information in the given context.
    2. If the context contains relevant information, use it to construct a clear and informative answer.
    3. If the exact answer is not in the context but related information is present, provide that information and explain its relevance.
    4. Quote relevant parts of the context when appropriate, using quotation marks.
    5. If multiple sources provide relevant information, synthesize them into a coherent answer.

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for i, doc in enumerate(docs))

    rag_chain = (
        RunnableParallel(
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    st.title("Legal GPT Assistant")
    st.write("You can ask questions about the legal documents in the specified directory.")
    
    download_from_drive(file_id, destination)
    
    vectorstore = load_vector_store()
    if vectorstore is None:
        st.error("Failed to load vector store. Exiting.")
        return

    rag_chain = create_rag_chain(vectorstore)

    user_question = st.text_input("Please enter your question:")
    
    if user_question:
        try:
            # Use the RAG chain to get the answer
            answer = rag_chain.invoke(user_question)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please try asking your question again.")

if __name__ == "__main__":
    main()
