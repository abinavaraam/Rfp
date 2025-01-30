import fitz  # PyMuPDF for PDFs
import spacy
import pandas as pd
from transformers import pipeline

# Load NLP model for header classification
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to extract text blocks from PDF
def extract_headers_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
        sections = []
            current_header = None
                current_value = ""

                    for page in doc:
                            blocks = page.get_text("dict")["blocks"]

                                    for block in blocks:
                                                for line in block.get("lines", []):
                                                                for span in line.get("spans", []):
                                                                                    text = span["text"].strip()
                                                                                                        font_size = span["size"]
                                                                                                                            is_bold = "bold" in span["font"].lower()

                                                                                                                                                # Use ML-based classification to detect headers
                                                                                                                                                                    result = classifier(text, candidate_labels=["header", "content"])
                                                                                                                                                                                        if result["labels"][0] == "header" and result["scores"][0] > 0.8:
                                                                                                                                                                                                                if current_header:
                                                                                                                                                                                                                                            sections.append([current_header, current_value.strip()])
                                                                                                                                                                                                                                                                    current_header = text
                                                                                                                                                                                                                                                                                            current_value = ""
                                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                                                        current_value += text + " "

                                                                                                                                                                                                                                                                                                                                            if current_header:
                                                                                                                                                                                                                                                                                                                                                    sections.append([current_header, current_value.strip()])

                                                                                                                                                                                                                                                                                                                                                        return sections

                                                                                                                                                                                                                                                                                                                                                        # Example Usage
                                                                                                                                                                                                                                                                                                                                                        pdf_sections = extract_headers_from_pdf("sample.pdf")

                                                                                                                                                                                                                                                                                                                                                        # Convert to DataFrame
                                                                                                                                                                                                                                                                                                                                                        df_pdf = pd.DataFrame(pdf_sections, columns=["Header", "Value"])
                                                                                                                                                                                                                                                                                                                                                        df_pdf.to_csv("pdf_extracted_data.csv", index=False)

                                                                                                                                                                                                                                                                                                                                                        print("Fully automated extraction complete!")
                                                                                                                                                                                                                                                                                                                                                        