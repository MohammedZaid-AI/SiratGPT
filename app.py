import os
import pymongo
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGING_FACE_TOKEN")



client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["SiratGPT"]
collection = db["Hadiths"]

def get_hadith(query):
    words=query.lower().split()
    og_query = {"$or": [{"english": {"$regex": f".*{word}.*", "$options": "i"}} for word in words]}
    hadith = collection.find_one(og_query)

    if hadith:
        return f"""
            Book: {hadith.get('book', 'Unknown')}
            Narrated by: {hadith.get('narrated', 'Unknown')}
            Hadith: {hadith.get('english', 'No text available')}
        """
    return None

def get_quran():
    
    text=""
    pdf_reader=PdfReader("Quran-English.pdf")
    
    for i,page in enumerate(pdf_reader.pages):
        content=page.extract_text()
        if content:
            text+=content

    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
    )
    chunks=text_splitter.split_text(text)
    return chunks

quran_chunks=get_quran()
def search_info(query):
    words=query.lower().split()
    matching_chunks=[]

    for chunk in quran_chunks:
        chunk_lower=chunk.lower()
        if any(word in chunk_lower for word in words):
            matching_chunks.append(chunk)

    if matching_chunks:
        return "\n\n".join(matching_chunks[:3])  # Return the first 3 relevant chunks
    return None

llm=HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature":1}
)


st.title("ISLAM GPT")
st.header("LETS KNOW MORE ABOUT ISLAM")
input_text=st.text_input("ENTER YOUR CHAT HERE")

prompt=PromptTemplate(
    input_variables=["input"],
    template="You are an Islamic scholar. Answer {input} with authentic Islamic teachings concisely, avoiding disclaimers."
)

chain=LLMChain(
    llm=llm,
    prompt=prompt
)

submit=st.button("GENERATE")


if submit:

    hadith=get_hadith(input_text)

    if hadith:
        hadith_summary=chain.run(f"summarize this hadith :{hadith}")
        st.write(hadith)
        st.write()
        st.write(hadith_summary)

    else:
        quran=search_info(input_text)

        if quran:
            explanation = chain.run(f"Provide an Islamic explanation for: {quran}")
            st.write(f"**Explanation:** {explanation}")


        else:    
            response=chain.run(input_text)
            st.write(response)      