import sys

def install_and_import():
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
            from pypdf import PdfReader
            return PdfReader
        except Exception as e:
            with open("extracted_text.txt", "w") as f:
                f.write(f"Failed to install/import pypdf: {e}")
            sys.exit(1)

def extract_text_from_pdf(pdf_path):
    try:
        PdfReader = install_and_import()
        reader = PdfReader(pdf_path)
        text = ""
        # Read first 15 pages to be safe
        num_pages = len(reader.pages)
        
        for i in range(min(num_pages, 15)): 
            page = reader.pages[i]
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

if __name__ == "__main__":
    pdf_path = "Pothole_paper.pdf"
    content = extract_text_from_pdf(pdf_path)
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(content)
