import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# Step 2: Chunk text into smaller parts
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Example Usage
pdf_path = "invoice.pdf"  # Replace with actual PDF file path
pdf_text = extract_text_from_pdf(pdf_path)

# Check if text extraction worked
if pdf_text:
    chunks = split_text_into_chunks(pdf_text)
    print(f"✅ Total Chunks Created: {len(chunks)}")
    
    # Print first 3 chunks for reference
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n🔹 Chunk {i+1}:\n{chunk}\n{'-'*50}")
else:
    print("⚠️ No text extracted. Check the PDF file.")
