from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

from data_init import get_documents

load_dotenv()
collection_name = "cleansed_greenhousegas"
raw_docs, documents, metadatas = get_documents(collection_name)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents(documents, metadatas=metadatas)
print(texts[0])

