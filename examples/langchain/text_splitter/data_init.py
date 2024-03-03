import requests
import pandas as pd
from dotenv import load_dotenv
from esg_toolkit.qdrant_utils import QdrantDBWrapper
from esg_toolkit.curation_utils import to_embeddings

load_dotenv()
collection_name = "cleansed_greenhousegas"
qdrant_wrapper = QdrantDBWrapper()


def get_documents(collection_name):
    doc_count = qdrant_wrapper.count(collection_name=collection_name, count_filter=None).count
    raw_docs = qdrant_wrapper.scroll(
        collection_name=collection_name, limit=doc_count, scroll_filter=None
    )[0]
    documents = [cleansing(d.payload["content"]) for d in raw_docs]
    metadatas = []
    for d in raw_docs:
        metadata = {}
        for k, v in d.payload.items():
            if k not in  ["content", "full_text"]:
                metadata[k] = v
        metadatas.append(metadata)
    raw_docs = [item.payload for item in raw_docs]
    return raw_docs, documents, metadatas
    
def get_company_name(content: str):
    return content.split("\u3000上市公司\r\n")[-1].split(" ")[0]

def cleansing(text: str):
    text = "\n".join(text.split("資料年度")[-1].split("\n")[1:])
    text = text.replace("\r", "\n")
    text = "\n".join([t.strip() for t in text.split("\n") if t.strip()])
    return text

def document_store_qdrant(data, url_name: str, collection_name: str = None):
    if collection_name is None:
        collection_name = url_name
    data["full_text"] = data.apply(
        lambda row: f'{row["year"]}\n{row["co_id"]}\n{row["content"]}', axis=1
    )
    embeddings = to_embeddings(data["full_text"].tolist())
    data.drop(columns=["full_text"], inplace=True)
    qdrant = QdrantDBWrapper()
    qdrant.upsert(
        collection_name=collection_name,
        data=data,
        embeddings=embeddings,
    )

