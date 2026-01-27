from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

def load_pdf_resume(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    resume_content=""
    for page in docs:
        resume_content+=page.page_content
    metadata = {
        "source": pdf_path,
        "page_count": len(docs)
    }
    return resume_content, metadata

resume_path="C:\\Users\\haris\\.git\\genAI_clg\\resume-004.pdf"
resume_text, resume_info = load_pdf_resume(resume_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=["\n\n", "\n", " "])
text = [resume_text]

docs = text_splitter.create_documents(text)
print("type of docs:",type(docs))
print("number of docs:",len(docs))
print(resume_info)
for i, doc in enumerate(docs):
    print(f"\n--- Document Chunk {i+1} ---")
    print(doc.page_content)
