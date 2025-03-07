import pymongo
from datasets import load_dataset

# Connect to MongoDB (Replace <your_connection_string>)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["SiratGPT"]
collection = db["Hadiths"]

# Load Hadith dataset


ds = load_dataset("cibfaye/hadiths_dataset")

# Insert Hadiths into MongoDB
hadith_list = []
for hadith in ds["train"]:
    hadith_data = {
        "book": hadith["Book"],
        "reference": hadith["Reference"],
        "narrated": hadith["Narrated"],
        "english": hadith["English"],
        "arabic": hadith["Arabic"]
    }
    hadith_list.append(hadith_data)

collection.insert_many(hadith_list)
print("✅ Hadiths stored in MongoDB!")
