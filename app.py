from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings


load_dotenv()
host = "34.80.177.36"
port = 6333
client = QdrantClient(f'{host}:{port}')
embeddings = OpenAIEmbeddings()
collection_name = "greenhousegas_openaiembeddings_abao"
qdrant = Qdrant(client, collection_name, embeddings)

