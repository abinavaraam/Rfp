import re
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Pretrained Transformer Model (Optimized for Similarity)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to Extract Sections
def extract_sections(text):
    sections = []
        lines = text.split("\n")
            current_section = None

                # Heading Detection: Look for Capitalized Lines or Keywords
                    for i, line in enumerate(lines):
                            line = line.strip()

                                    # Consider as a heading if:
                                            # - It has keywords ("requirement", "terms", "conditions")
                                                    # - It is ALL CAPS or Title Case (indicating a heading)
                                                            if re.match(r"^\d+(\.\d+)*\s+", line) or any(kw in line.lower() for kw in ["requirement", "terms", "conditions"]):
                                                                        current_section = {"Title": line, "Content": ""}
                                                                                    sections.append(current_section)

                                                                                            elif current_section:
                                                                                                        # Append non-empty lines as content
                                                                                                                    if line:
                                                                                                                                    current_section["Content"] += line + " "

                                                                                                                                        return sections

                                                                                                                                        # Function to Get Normalized Embeddings
                                                                                                                                        def get_embedding(text):
                                                                                                                                            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                                                                                                                                                with torch.no_grad():
                                                                                                                                                        output = model(**tokens)
                                                                                                                                                            embedding = output.last_hidden_state.mean(dim=1).numpy()
                                                                                                                                                                
                                                                                                                                                                    # Normalize Embeddings to Improve Similarity Scores
                                                                                                                                                                        return embedding / np.linalg.norm(embedding)

                                                                                                                                                                        # Load Document (Replace with File Reading)
                                                                                                                                                                        document_text = """ 
                                                                                                                                                                        3.1 REQUIREMENT VOICE 
                                                                                                                                                                        This section describes the voice service requirements. The solution should support VoIP and SIP protocols.

                                                                                                                                                                        3.1.1 CALL QUALITY 
                                                                                                                                                                        The solution must provide high-definition voice quality and echo cancellation.

                                                                                                                                                                        3.2 GENERAL TERMS 
                                                                                                                                                                        All vendors must comply with security policies.

                                                                                                                                                                        3.2.1 COMPLIANCE 
                                                                                                                                                                        Vendors should follow GDPR and data security regulations.

                                                                                                                                                                        PAYMENT CONDITIONS
                                                                                                                                                                        Vendors must agree to a 30-day payment cycle.
                                                                                                                                                                        """

                                                                                                                                                                        # Define Matching Keywords
                                                                                                                                                                        matching_keywords = ["Requirement", "Terms", "Conditions"]

                                                                                                                                                                        # Extract Sections
                                                                                                                                                                        sections = extract_sections(document_text)

                                                                                                                                                                        # Compute Keyword Embeddings (Normalized)
                                                                                                                                                                        keyword_embeddings = [get_embedding(k) for k in matching_keywords]

                                                                                                                                                                        # Match Sections Based on Cosine Similarity
                                                                                                                                                                        matched_sections = []
                                                                                                                                                                        for section in sections:
                                                                                                                                                                            section_text = section["Title"] + " " + section["Content"]
                                                                                                                                                                                section_embedding = get_embedding(section_text)
                                                                                                                                                                                    
                                                                                                                                                                                        # Compute Cosine Similarity with Each Keyword
                                                                                                                                                                                            similarities = [cosine_similarity(section_embedding, kw_emb)[0][0] for kw_emb in keyword_embeddings]
                                                                                                                                                                                                
                                                                                                                                                                                                    # If similarity is above 0.6, consider it relevant
                                                                                                                                                                                                        if max(similarities) > 0.6:
                                                                                                                                                                                                                section["Similarity Score"] = max(similarities)
                                                                                                                                                                                                                        matched_sections.append(section)

                                                                                                                                                                                                                        # Convert to Pandas DataFrame
                                                                                                                                                                                                                        df = pd.DataFrame(matched_sections)

                                                                                                                                                                                                                        # Display Structured Output
                                                                                                                                                                                                                        print(df)
                                                                                                                                                                                                                        