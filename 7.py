from transformers import pipeline
import fitz  # PyMuPDF for PDFs
from docx import Document
import pandas as pd

# Load a different NLP model for text classification
model_name = "allenai/longformer-large-4096-finetuned"
classifier = pipeline("zero-shot-classification", model=model_name)

# Function to classify text as Header or Content
def classify_text(text):
    labels = ["Header", "Content"]
        classification = classifier(text, candidate_labels=labels)
            return classification["labels"][0]  # Returns "Header" or "Content"

            # Function to extract headers & values
            def extract_headers_with_model(text):
                lines = text.split("\n")
                    sections = []
                        current_section = None

                            for line in lines:
                                    line = line.strip()
                                            if not line:
                                                        continue

                                                                label = classify_text(line)

                                                                        if label == "Header":
                                                                                    if current_section:
                                                                                                    sections.append(current_section)
                                                                                                                current_section = {"Header": line, "Content": ""}
                                                                                                                        elif current_section:
                                                                                                                                    current_section["Content"] += line + " "

                                                                                                                                        if current_section:
                                                                                                                                                sections.append(current_section)

                                                                                                                                                    return sections

                                                                                                                                                    # Function to extract text from PDFs
                                                                                                                                                    def extract_text_from_pdf(pdf_path):
                                                                                                                                                        doc = fitz.open(pdf_path)
                                                                                                                                                            text = "\n".join([page.get_text("text") for page in doc])
                                                                                                                                                                return text

                                                                                                                                                                # Function to extract text from DOCX
                                                                                                                                                                def extract_text_from_docx(docx_path):
                                                                                                                                                                    doc = Document(docx_path)
                                                                                                                                                                        text = "\n".join([para.text for para in doc.paragraphs])
                                                                                                                                                                            return text

                                                                                                                                                                            # Function to handle any document type
                                                                                                                                                                            def extract_from_document(file_path, file_type):
                                                                                                                                                                                if file_type.lower() == "pdf":
                                                                                                                                                                                        text = extract_text_from_pdf(file_path)
                                                                                                                                                                                            elif file_type.lower() == "docx":
                                                                                                                                                                                                    text = extract_text_from_docx(file_path)
                                                                                                                                                                                                        else:
                                                                                                                                                                                                                print("Unsupported file type.")
                                                                                                                                                                                                                        return []

                                                                                                                                                                                                                            return extract_headers_with_model(text)

                                                                                                                                                                                                                            # Example Usage
                                                                                                                                                                                                                            pdf_results = extract_from_document("sample.pdf", "pdf")
                                                                                                                                                                                                                            docx_results = extract_from_document("sample.docx", "docx")

                                                                                                                                                                                                                            # Convert to DataFrame and save
                                                                                                                                                                                                                            df_pdf = pd.DataFrame(pdf_results, columns=["Header", "Content"])
                                                                                                                                                                                                                            df_docx = pd.DataFrame(docx_results, columns=["Header", "Content"])

                                                                                                                                                                                                                            df_pdf.to_csv("pdf_extracted_data.csv", index=False)
                                                                                                                                                                                                                            df_docx.to_csv("docx_extracted_data.csv", index=False)

                                                                                                                                                                                                                            print("Extraction complete! Data saved as structured CSV.")
                                                                                                                                                                                                                            