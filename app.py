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

def get_hadith(query, limit=3):
    if any(char.isdigit() for char in query):
        search_query = {"reference": {"$regex": query.strip(), "$options": "i"}}
    else:
        words = query.lower().split()
        search_query = {
            "$or": [
                {"english": {"$regex": f".*{word}.*", "$options": "i"}} for word in words
            ] + [{"book": {"$regex": query.strip(), "$options": "i"}}]
        }

    results = list(collection.find(search_query).limit(limit))

    if results:
        dataset = ""
        for i, hadith in enumerate(results):
            dataset += f"""
    Hadith {i+1}
            Book: {hadith.get('book', 'Unknown')}
            Narrated by: {hadith.get('narrated', 'Unknown')}
            Reference: {hadith.get('reference', 'Unknown')}
            Hadith: {hadith.get('english', 'No text available')}
            """
        return dataset
    return None 

def get_quran(query):
    pdf_reader = PdfReader("Quran-English.pdf")
    text=""
    for pagenum,page in enumerate(pdf_reader.pages):  #.pages is property of pdf_reader that holds all the pages of the pdf"
        content=page.extract_text()                   #page holds the content of the  each page of the pdf
        if content:
            text+=content

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks=text_splitter.split_text(text)
    for i,chunk in enumerate(chunks):
        if query in chunk:
             return chunk
    return None

    
def get_response(query):
    response=llm.run(query)
    return response



llm=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={"temperature":0.5}
)


st.title("ISLAM GPT")
st.header("LETS KNOW MORE ABOUT ISLAM")
input_text=st.text_input("ENTER YOUR CHAT HERE")

system_prompt = """
You are an expert Islamic scholar and a knowledgeable assistant skilled in retrieving authentic Islamic information. You can fetch Hadith details from a MongoDB database and Quranic references from a PDF document to provide accurate, concise, and clear explanations.

When a user asks for a Hadith, search the MongoDB database for relevant content. When a Quranic reference is requested, extract the required text from the provided PDF document. If no relevant content is found in either source, generate an insightful response based on your Islamic knowledge.

Example 1:
**User Query:** "Tell me about Sunan Ibn Majah 1446"
**Response:** Hadith #1446 from Sunan Ibn Majah states... (include details here).

Example 2:
**User Query:** "I need a Hadith about patience."
**Response:** Here’s a Hadith on patience: (include relevant details).

Example 3:
**User Query:** "What does Surah Ar-Rahman say?"
**Response:** Surah Ar-Rahman emphasizes Allah's blessings and repeatedly asks: "Then which of the favors of your Lord will you deny?" (Include relevant verse details).

Database (Hadith/Quran): {dataset}
User Query: {input}

Response:
"""



prompt=PromptTemplate(
    input_variables=["dataset","input"],
    template=system_prompt
)




chain=LLMChain(
    llm=llm,
    prompt=prompt
)

submit=st.button("GENERATE")


if submit:
    hadith = get_hadith(input_text)
    quran=get_quran(input_text)
    
    
    if hadith:
        output = chain.run(dataset=hadith, input=input_text)
    elif quran:
        output = chain.run(dataset=quran, input=input_text)
    else:
        output=get_response(input=input_text)
        
    if "Response:" in output:
        clean_response = output.split("Response:")[-1].strip()
    else:
        clean_response = output.strip()

    st.write(clean_response)
     