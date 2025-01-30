from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import fitz  # PyMuPDF
from docx import Document
import pandas as pd

# Load LayoutLM model for document understanding
model_name = "microsoft/layoutlmv3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to process text through LayoutLM
def extract_headers_with_model(text):
    results = nlp_pipeline(text)
        headers = []
            current_header = None
                current_content = ""

                    for entity in results:
                            label = entity['entity_group']
                                    word = entity['word']

                                            if "HEADER" in label:
                                                        if current_header:
                                                                        headers.append([current_header, current_content.strip()])
                                                                                    current_header = word
                                                                                                current_content = ""
                                                                                                        else:
                                                                                                                    current_content += word + " "

                                                                                                                        if current_header:
                                                                                                                                headers.append([current_header, current_content.strip()])

                                                                                                                                    return headers

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
                                                                                                                                                                                                            