import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory: str):
    """遍历目录,加载所有支持的文档"""
    all_docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.lower().split(".")[-1]
            if ext == "txt" or ext == "md":
                loader = TextLoader(file_path, encoding="utf-8")
                all_docs.extend(loader.load())
            if ext == "pdf":
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())

    print(f"共加载 {len(all_docs)} 个文档")
    return all_docs


def split_documents(docs, chunk_size=300, chunk_overlap=30):
    """把文档切成小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"共切分成 {len(chunks)} 个片段")
    return chunks

def get_chunks(directory:str,chunk_size=300, chunk_overlap=30):
    """获取指定目录下的所有文档"""
    docs = load_documents(directory)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    return chunks

