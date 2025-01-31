import zipfile
import lxml.etree as ET
import re

def extract_text_from_docx(docx_path):
    """Extracts structured text (headers and contents) from a DOCX file using lxml"""
        
            with zipfile.ZipFile(docx_path, 'r') as docx:
                    xml_content = docx.read('word/document.xml')
                        
                            tree = ET.fromstring(xml_content)
                                namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                                    
                                        extracted_data = []
                                            current_header = None
                                                current_content = []

                                                    for para in tree.findall('.//w:p', namespace):
                                                            text = ''.join(node.text or '' for node in para.findall('.//w:t', namespace)).strip()
                                                                    if not text:
                                                                                continue  # Skip empty lines

                                                                                        # Check for bold formatting
                                                                                                is_bold = any(node.find('.//w:b', namespace) is not None for node in para.findall('.//w:r', namespace))

                                                                                                        # Check if paragraph starts with a valid numbered format (e.g., 1., 1), 1.1), 3.1, 3.1.1)
                                                                                                                is_numbered = bool(re.match(r'^\d+(\.\d+)*\)?\s+', text))  # Matches "1.", "1)", "1.1)", "3.1"
                                                                                                                        is_bullet = bool(re.match(r'^[-*•]\s+', text))  # Matches bullets like "- ", "* ", "• "

                                                                                                                                # If this paragraph is a header
                                                                                                                                        if is_numbered or (is_bold and len(text) < 80):
                                                                                                                                                    if current_header:
                                                                                                                                                                    extracted_data.append((current_header, "\n".join(current_content)))  # Save previous section
                                                                                                                                                                                
                                                                                                                                                                                            current_header = text  # Store new header
                                                                                                                                                                                                        current_content = []  # Reset content storage
                                                                                                                                                                                                                else:
                                                                                                                                                                                                                            current_content.append(text)  # Append text under the last detected header

                                                                                                                                                                                                                                # Save the last detected section
                                                                                                                                                                                                                                    if current_header:
                                                                                                                                                                                                                                            extracted_data.append((current_header, "\n".join(current_content)))

                                                                                                                                                                                                                                                return extracted_data

                                                                                                                                                                                                                                                # Example Usage
                                                                                                                                                                                                                                                docx_path = "sample.docx"
                                                                                                                                                                                                                                                headers_content = extract_text_from_docx(docx_path)

                                                                                                                                                                                                                                                # Print results
                                                                                                                                                                                                                                                for header, content in headers_content:
                                                                                                                                                                                                                                                    print(f"HEADER: {header}\nCONTENT: {content}\n{'='*50}")
                                                                                                                                                                                                                                                    