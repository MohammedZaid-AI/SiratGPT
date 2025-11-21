from flask import Flask, request, jsonify, render_template

import logging
import sys
import os
import traceback

# -------------------------
# IMPORT YOUR STREAMLIT RAG COMPONENTS
# -------------------------

from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -------------------------
# LOGGING
# -------------------------

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# -------------------------
# CHUNK MANAGER
# -------------------------

class ChunkManager:
    def __init__(self, chunk_size=2000, chunk_overlap=300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                  chunk_overlap=self.chunk_overlap)
        chunks = []
        for doc in docs:
            pieces = splitter.split_documents([doc])
            chunks += pieces
        return chunks


# -------------------------
# EMBEDDING MANAGER
# -------------------------

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, chunks):
        texts = [c.page_content for c in chunks]
        return self.model.encode(texts)


# -------------------------
# VECTORSTORE MANAGER
# -------------------------

class VectorStoreManager:
    def __init__(self, name="Quran", persist="./vectorstore"):
        self.client = chromadb.PersistentClient(path=persist)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": "Quran embeddings"}
        )

    def add_documents(self, chunks, embeddings):
        for i, (doc, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            metadata = dict(doc.metadata)
            metadata["context"] = doc.page_content
            self.collection.add(
                ids=[doc_id],
                documents=[doc.page_content],
                metadatas=[metadata],
                embeddings=[emb.tolist()]
            )
        print(f"Added {len(chunks)} new chunks.")

    def count(self):
        return self.collection.count()


# -------------------------
# RAG RETRIEVER
# -------------------------

class RAGRetriever:
    def __init__(self, vs, em, top_k=3):
        self.vs = vs
        self.em = em
        self.top_k = top_k

    def retrieve(self, query):
        q_emb = self.em.model.encode([query])[0]

        results = self.vs.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=self.top_k
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return output


# -------------------------
# LOAD OR BUILD VECTORSTORE (ONLY ONCE)
# -------------------------

def load_index():
    vs = VectorStoreManager()
    em = EmbeddingManager()

    # If vectorstore already built → don't rebuild
    if vs.count() > 0:
        return vs, em

    # Otherwise build index the first time
    loader = DirectoryLoader(
        path="pdfs/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

    chunker = ChunkManager()
    chunks = chunker.chunk_documents(docs)

    embeddings = em.get_embeddings(chunks)
    vs.add_documents(chunks, embeddings)

    return vs, em


vectorstore, embedding_manager = load_index()
retriever = RAGRetriever(vectorstore, embedding_manager)


# -------------------------
# LLM SETUP
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9,
    max_output_tokens=1000
)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are SiratGPT, an Islamic knowledge assistant.
Use only the given context from the Quran data.

Context:
{context}

User Question:
{query}

Answer respectfully:
"""
)

chain = prompt | llm


# -------------------------
# FLASK SETUP
# -------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query_api():
    try:
        user_input = request.form.get("input_text", "")

        # Special Case
        if "created" in user_input.lower() and "sirat" in user_input.lower():
            return jsonify({"response": "SiratGPT was created by Zaid, a visionary AI engineer and entrepreneur who is passionate about fusing technology with knowledge. As the Founder of HatchUp.ai, Zaid built SiratGPT to bring deep Islamic insights to the digital world, combining modern AI techniques with timeless wisdom. His expertise in AI, app development, and automation drives this project, making SiratGPT a unique and intelligent guide for seekers of knowledge"})

        # Retrieve context
        results = retriever.retrieve(user_input)
        context = "\n\n".join([r["text"] for r in results])

        # If no context found
        if not context.strip():
            return jsonify({"response": "I couldn't find relevant Quran context for your question."})

        # Run LLM
        response = chain.invoke({"context": context, "query": user_input})
        return jsonify({"response": response.content})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🔥 SiratGPT Flask server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
