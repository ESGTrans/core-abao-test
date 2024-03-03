from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

from data_init import get_documents

load_dotenv()
collection_name = "cleansed_greenhousegas"
documents, metadatas = get_documents(collection_name)

print("Normal: ")
text_splitter = SemanticChunker(OpenAIEmbeddings())
texts = text_splitter.create_documents(documents, metadatas=metadatas)
print(texts[0])
print(len(texts))

methdos = ["percentile", "standard_deviation", "interquartile"]
for method in methdos:
    print(f"Using {method}")
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type=method
    )
    texts = text_splitter.create_documents(documents, metadatas=metadatas)
    print(texts[0])
    print(len(texts))
    print("=============\n\n")

