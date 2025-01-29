import re
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Pretrained BERT Model & Tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight BERT model for embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to extract section-wise content
def extract_sections(text):
    sections = []
        section_pattern = re.compile(r"(\d+\.\d+(?:\.\d+)*)\s+(.+)")  # Matches "3.1 Requirement Voice"
            
                lines = text.split("\n")
                    current_section = None

                        for line in lines:
                                match = section_pattern.match(line.strip())
                                        if match:
                                                    section_num, section_title = match.groups()
                                                                current_section = {"Section": section_num, "Title": section_title, "Content": ""}
                                                                            sections.append(current_section)
                                                                                    elif current_section:
                                                                                                current_section["Content"] += line.strip() + " "  # Append content

                                                                                                    return sections

                                                                                                    # Function to get BERT embeddings
                                                                                                    def get_embedding(text):
                                                                                                        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                                                                                                            with torch.no_grad():
                                                                                                                    output = model(**tokens)
                                                                                                                        return output.last_hidden_state.mean(dim=1).numpy()  # Get sentence embedding

                                                                                                                        # Load Document (Replace with file reading)
                                                                                                                        document_text = """ 
                                                                                                                        3.1 Requirement Voice 
                                                                                                                        This section describes the voice service requirements. The solution should support VoIP and SIP protocols.

                                                                                                                        3.1.1 Call Quality 
                                                                                                                        The solution must provide high-definition voice quality and echo cancellation.

                                                                                                                        3.2 General Terms 
                                                                                                                        All vendors must comply with security policies.

                                                                                                                        3.2.1 Compliance 
                                                                                                                        Vendors should follow GDPR and data security regulations.

                                                                                                                        4.0 Payment Conditions
                                                                                                                        Vendors must agree to a 30-day payment cycle.
                                                                                                                        """

                                                                                                                        # Define Matching Keywords
                                                                                                                        matching_keywords = ["Requirement", "Terms", "Conditions"]

                                                                                                                        # Extract Sections
                                                                                                                        sections = extract_sections(document_text)

                                                                                                                        # Compute Keyword Embeddings
                                                                                                                        keyword_embeddings = [get_embedding(k) for k in matching_keywords]

                                                                                                                        # Match Sections Based on Similarity
                                                                                                                        matched_sections = []
                                                                                                                        for section in sections:
                                                                                                                            section_text = section["Title"] + " " + section["Content"]
                                                                                                                                section_embedding = get_embedding(section_text)
                                                                                                                                    
                                                                                                                                        # Compute similarity with each keyword
                                                                                                                                            similarities = [cosine_similarity(section_embedding, kw_emb)[0][0] for kw_emb in keyword_embeddings]
                                                                                                                                                
                                                                                                                                                    # If similarity above threshold (e.g., 0.6), consider it relevant
                                                                                                                                                        if max(similarities) > 0.6:
                                                                                                                                                                matched_sections.append(section)

                                                                                                                                                                # Convert to Pandas DataFrame
                                                                                                                                                                df = pd.DataFrame(matched_sections)

                                                                                                                                                                # Display Structured Output
                                                                                                                                                                print(df)
                                                                                                                                                                