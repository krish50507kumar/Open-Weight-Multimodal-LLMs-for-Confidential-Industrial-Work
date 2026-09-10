from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, skip_pages=0):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i < skip_pages:
            continue
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text