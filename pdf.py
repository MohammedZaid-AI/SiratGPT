from datasets import load_dataset
from PyPDF2 import PdfWriter, PageObject

# Load the dataset
dataset = load_dataset("cibfaye/hadiths_dataset")["train"]  # Select the "train" split

# Select specific indices
selected_hadiths = [dataset[i] for i in [2, 5, 12]]

# Create a new PDF document
pdf_writer = PdfWriter()

# Add data to the PDF
for hadith in selected_hadiths:
    text_to_add = f"Hadith: {hadith.get('text', 'No text found')}"

    # Create a blank page
    page = PageObject.create_blank_page(width=500, height=700)

    # Add the page to the PDF
    pdf_writer.add_page(page)

# Save the PDF
with open("hadiths.pdf", "wb") as f:
    pdf_writer.write(f)

print("PDF created successfully!")
from datasets import load_dataset
from PyPDF2 import PdfWriter, PageObject

# Load the dataset
dataset = load_dataset("cibfaye/hadiths_dataset")["train"]  # Select the "train" split

# Select specific indices
selected_hadiths = [dataset[i] for i in [2, 5, 12]]

# Create a new PDF document
pdf_writer = PdfWriter()

# Add data to the PDF
for hadith in selected_hadiths:
    text_to_add = f"Hadith: {hadith.get('text', 'No text found')}"

    # Create a blank page
    page = PageObject.create_blank_page(width=500, height=700)

    # Add the page to the PDF
    pdf_writer.add_page(page)

# Save the PDF
with open("hadiths.pdf", "wb") as f:
    pdf_writer.write(f)

print("PDF created successfully!")
