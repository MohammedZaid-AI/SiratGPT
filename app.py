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
    logging.info(f"Searching for hadith with query: {query}")
    
    # Create queries for different search strategies
    exact_match_query = {"reference": {"$regex": f"{query.strip()}", "$options": "i"}}
    
    broader_match_query = {
        "$or": [
            {"reference": {"$regex": f".*{query.strip()}.*", "$options": "i"}},
            {"english": {"$regex": f".*{query.strip()}.*", "$options": "i"}},
            {"book": {"$regex": f".*{query.strip()}.*", "$options": "i"}}
        ]
    }

    # Try exact match first
    results = list(collection.find(exact_match_query).limit(limit))
    logging.info(f"Exact match results: {len(results)}")

    # If no exact matches, try broader match
    if not results:
        results = list(collection.find(broader_match_query).limit(limit))
        logging.info(f"Broader match results: {len(results)}")

    # Format results if any found
    if results:
        dataset = ""
        for i, hadith in enumerate(results):
            dataset += f"""
            Book: {hadith.get('book', 'Unknown')}
            Narrated by: {hadith.get('narrated', 'Unknown')}
            Reference: {hadith.get('reference', 'Unknown')}
            Hadith: {hadith.get('english', 'No text available')}
            """
        logging.info(f"Returning {len(results)} hadiths")
        return dataset
    
    logging.info("No hadith results found")
    return None 

def get_quran(query, limit=5):
    pdf_reader = PdfReader("Quran-English.pdf")
    text = ""
    
    for pagenum, page in enumerate(pdf_reader.pages):           #.pages is property of pdf_reader that holds all the pages of the pdf
        content = page.extract_text()                            #page holds the content of the  each page of the pdf
        if content:
            text += content
    print(text[:500]) 
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300     
    )
    
    chunks = text_splitter.split_text(text)
    matched_chunks = []

    for chunk in chunks:
        if query.lower() in chunk.lower():
            matched_chunks.append(chunk)
            if len(matched_chunks) == limit: 
                break  

    return "\n\n".join(matched_chunks) if matched_chunks else None


def get_response(query):
    logging.info(f"Generating AI response for: {query}")
    response = llm.predict(query)
    return response



llm=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={
        "temperature":0.5,
        "max_new_tokens": 300 
        }
)


st.title("ISLAM GPT")
st.header("LETS KNOW MORE ABOUT ISLAM")
input_text=st.text_input("ENTER YOUR CHAT HERE")


system_prompt = """
You are an expert Islamic scholar and MongoDB expert with access to both a Hadith database and Quran PDF. 
Your role is to provide **detailed information** from both sources without summarizing or altering the content. 

If the user asks for information from the Quran, **fetch content directly from the PDF** before responding. 
If no matching data is found in the Quran PDF or Hadith database, respond with "**No information found in the available sources.**"

Example 1:
**User Query:** "Tell me about Ramadan."
**Response:** 
Here’s what I found from Hadith: (Hadith details)
Here’s what I found from the Quran PDF: (Quran details)

Example 2:
**User Query:** "Tell me about patience."
**Response:** 
Here’s a Hadith on patience: (Hadith details)
Here’s a relevant Quran verse from the PDF on patience: (Quran details)

Database and PDF Content: {dataset}
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
    quran = get_quran(input_text)
    #ai_response = get_response(input_text)
    
    combined_data = ""

    if hadith:
        combined_data += hadith + "\n"
    if quran:
        combined_data += quran + "\n"
    
    
    if combined_data.strip():  
        output = chain.run(dataset=combined_data, input=input_text)
        st.write(output.strip())