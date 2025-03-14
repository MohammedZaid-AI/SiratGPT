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
        # General keyword search
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
            Hadith #{i+1}
            Book: {hadith.get('book', 'Unknown')}
            Narrated by: {hadith.get('narrated', 'Unknown')}
            Reference: {hadith.get('reference', 'Unknown')}
            Hadith: {hadith.get('english', 'No text available')}
            """
        return dataset
    return "No matching Hadith found in the database."  


llm=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature":0.7}
)


st.title("ISLAM GPT")
st.header("LETS KNOW MORE ABOUT ISLAM")
input_text=st.text_input("ENTER YOUR CHAT HERE")

system_prompt = """
You are an Islamic scholar and MongoDB expert. Your role is to retrieve Hadith information from the database and provide concise, clear answers.

Example 1:
**User Query:** "Tell me about Sunan Ibn Majah 1446"
**Response:** Hadith #1446 from Sunan Ibn Majah states... (include details here).

Example 2:
**User Query:** "I need a Hadith about patience."
**Response:** Here’s a Hadith on patience: (include relevant details).

Database: {dataset}
User Query: {input}

Response:
"""



prompt=PromptTemplate(
    input_variables=["dataset","input"],
    template=system_prompt
)


# hadith_prompt=PromptTemplate(
#     input_variables=["dataset","query"],
#     template="You are analyzing Islamic texts. Review this dataset: {dataset}\n\nAnswer the following query: {query}\nProvide a concise and informative response based on the dataset."
# )

# hadith_answer=LLMChain(
#     llm=llm,
#     prompt=hadith_prompt
# )



chain=LLMChain(
    llm=llm,
    prompt=prompt
)

submit=st.button("GENERATE")


if submit:

    hadith=get_hadith(input_text)

    if hadith:
        output=chain.run(dataset=hadith,input=input_text)
        st.write(output)

    # else:    
    #     response=chain.run(collection="Hadiths", input=input_text)
    #     st.write(response)      