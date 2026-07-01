import zipfile
import xml.etree.ElementTree as ET
import os

def extract_docx_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespace map for wordprocessingml
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error extracting {docx_path}: {e}"

def main():
    base_dir = r"C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
    files = ["job_description.docx", "redrob_signals_doc.docx", "submission_spec.docx", "README.docx"]
    
    for f in files:
        path = os.path.join(base_dir, f)
        text = extract_docx_text(path)
        out_path = os.path.join(base_dir, f.replace(".docx", ".txt"))
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(text)
        print(f"Extracted {f} to {out_path} ({len(text)} chars)")

if __name__ == "__main__":
    main()
