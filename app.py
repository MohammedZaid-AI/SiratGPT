import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
import streamlit as st
import os



# LangChain (modern modular imports)
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")




        # client = pymongo.MongoClient("mongodb://localhost:27017/")
        # db = client["SiratGPT"]
        # collection = db["Hadiths"]

        # def get_hadith(query, limit=3):
        #     logging.info(f"Searching for hadith with query: {query}")
            
        #     exact_match_query = {"reference": {"$regex": f"{query.strip()}", "$options": "i"}}
            
        #     broader_match_query = {
        #         "$or": [
        #             {"reference": {"$regex": f".*{query.strip()}.*", "$options": "i"}},
        #             {"english": {"$regex": f".*{query.strip()}.*", "$options": "i"}},
        #             {"book": {"$regex": f".*{query.strip()}.*", "$options": "i"}}
        #         ]
        #     }


        #     results = list(collection.find(exact_match_query).limit(limit))
        #     logging.info(f"Exact match results: {len(results)}")


        #     if not results:
        #         results = list(collection.find(broader_match_query).limit(limit))
        #         logging.info(f"Broader match results: {len(results)}")

        #     if results:
        #         dataset = ""
        #         for i, hadith in enumerate(results):
        #             dataset += f"""
        #             Book: {hadith.get('book', 'Unknown')}
        #             Narrated by: {hadith.get('narrated', 'Unknown')}
        #             Reference: {hadith.get('reference', 'Unknown')}
        #             Hadith: {hadith.get('english', 'No text available')}
        #             """
        #         logging.info(f"Returning {len(results)} hadiths")
        #         return dataset
            
        #     logging.info("No hadith results found")
        #     return None 

loader=DirectoryLoader(
        path="pdfs/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False
        )

docs=loader.load()

#Quran RAG
class ChunkManager:
    def __init__(self, chunks_size:int=2000, chunk_overlap:int=300):
        self.chunks_size = chunks_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[Document]):
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=self.chunks_size,
                chunk_overlap=self.chunk_overlap     
            )

            all_chunks=[]
            for doc in documents:
                chunks=text_splitter.split_documents([doc])
                all_chunks+=chunks
            return all_chunks


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model=None
        self.model_name=model_name
        self.load_model()

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model=SentenceTransformer(self.model_name)

    def get_embeddings(self, chunks: list[Document]) -> np.ndarray:
        texts=[chunk.page_content for chunk in chunks]
        embeddings=self.model.encode(texts)
        return embeddings

class VectorStoreManager:
    def __init__(self,collection_name:str="Quran",persist_directory:str="./vectorstore"):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self.initialize_client()

    def initialize_client(self):
        self.client=chromadb.PersistentClient(path=self.persist_directory)
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description":"Quran embeddings"}
            )
    
    def add_documents(self,documents:list[Document],embeddings:np.ndarray):
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        ids=[]
        metadatas=[]
        document_texts=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata=dict(doc.metadata)
            metadata["doc_length"]=len(doc.page_content)
            metadata["context"]=doc.page_content
            metadatas.append(metadata)

            document_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

            self.collection.add(
                ids=[doc_id],
                metadatas=[metadata],
                documents=[doc.page_content],
                embeddings=[embedding.tolist()]
            )
        print(f"successfully added {len(documents)} to the vector store")
        print(f"Current collection size: {self.collection.count()}")


#calling

chunk_manager=ChunkManager()
chunks=chunk_manager.chunk_documents(docs)

vector_store_manager=VectorStoreManager()

embedding_manager=EmbeddingManager()
embeddings=embedding_manager.get_embeddings(chunks)


vector_store_manager.add_documents(chunks,embeddings)



class RAGRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager, embedding_manager: EmbeddingManager, top_k: int = 3):
        self.vector_store_manager = vector_store_manager
        self.embedding_manager = embedding_manager
        self.top_k = top_k

    def retrieve(self, query: str):
        # Get embedding for the query
        query_embedding = self.embedding_manager.model.encode([query])[0]

        # Search in ChromaDB
        results = self.vector_store_manager.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k
        )

        # Format the results
        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            retrieved_docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved_docs

    def pretty_print(self, results):
        for i, res in enumerate(results, 1):
            print(f"\n🔹 Result {i}")
            print(f"Distance: {res['distance']:.4f}")
            print(f"Metadata: {res['metadata']}")
            print(f"Content:\n{res['text'][:400]}...")  # show first 400 chars


st.title("Sirat GPT")
st.header("LETS KNOW MORE ABOUT ISLAM")
input_text=st.text_input("ENTER YOUR CHAT HERE")

retriever=RAGRetriever(vector_store_manager,embedding_manager)
# retriever.pretty_print(results)
            
llm =ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_tokens=300
        )




system_prompt = """
Database and PDF Content: {context}
User Query: {input_text}

Response:

"""

prompt=PromptTemplate(
            input_variables=["context","input_text"],
            template=system_prompt
        )




chain = prompt | llm

submit=st.button("GENERATE")


if submit:
            if input_text =="Who created sirat gpt?":
            
                output="SiratGPT was created by Zaid, a visionary AI engineer and entrepreneur who is passionate about fusing technology with knowledge. As the Founder of HatchUp.ai, Zaid built SiratGPT to bring deep Islamic insights to the digital world, combining modern AI techniques with timeless wisdom. His expertise in AI, app development, and automation drives this project, making SiratGPT a unique and intelligent guide for seekers of knowledge."
                st.write(output)
                
            else:   
                # hadith = get_hadith(input_text)
                quran = retriever.retrieve(input_text)
                        
                combined_data = ""

                # if hadith:
                #     combined_data += hadith + "\n"
                for r in quran:
                    combined_data += r["text"] + "\n"
                        
                output = chain.invoke({"context": combined_data, "input_text": input_text})

                            
                    
                clean_response = output.content.strip()
                        
                            
                st.write(clean_response) 
                        
            

# button=st.toggle("Deep Search")
# if button:
#                 st.write("Deep Search mode is on")
#                 output2=get_response(input_text)   
#                 st.write(output2)