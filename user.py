import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document, HumanMessage, AIMessage
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables
load_dotenv()

class CustomRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        print("\nRetrieving relevant documents...")
        documents = self.vectorstore.similarity_search(query, k=10)
        print(f"Retrieved {len(documents)} documents.")
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

def setup_vectorstore(index_name: str) -> PineconeVectorStore:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return PineconeVectorStore.from_existing_index(index_name, embedding_model)

def setup_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(vectorstore: PineconeVectorStore, llm: ChatGroq):
    retriever = CustomRetriever(vectorstore=vectorstore)
    
    template = """You are a knowledgeable legal assistant. Answer the following query based on the provided context:
    
    Context: {context}
    
    Query: {question}
    
    Ensure your response is accurate and based on legal statutes, precedents, or procedures."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    print("\nInitializing Legal Assistant...")
    
    try:
        index_name = "legal-rag"
        vectorstore = setup_vectorstore(index_name)
        llm = setup_llm()
        rag_chain = setup_rag_chain(vectorstore, llm)

        print("\nLegal Assistant Ready. Enter your queries (type 'quit' to exit).\n")
        
        chat_history = []
        while True:
            query = input("\nEnter your legal query: ").strip()
            if query.lower() == 'quit':
                break
                
            try:
                response = rag_chain.invoke(query)
                
                print("\nResponse:", response)
                print("\n" + "="*50)
                
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response))
                
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try again with a different query.")
                
    except Exception as e:
        print(f"\nError initializing the Legal Assistant: {str(e)}")
        print("Please check your environment variables and dependencies.")

if __name__ == "__main__":
    main()
