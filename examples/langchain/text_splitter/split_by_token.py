from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from transformers import BertTokenizer

from data_init import get_documents

load_dotenv()
collection_name = "cleansed_greenhousegas"
documents, metadatas = get_documents(collection_name)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=0,
)
texts = text_splitter.create_documents(documents, metadatas=metadatas)
print(texts[0])
text_splitter.split_text(documents[0])

# 亂碼
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
text_splitter.split_text(documents[0])

# 要先下載 spacy 模型
text_splitter = SpacyTextSplitter(chunk_size=1000)
text_splitter.split_text(documents[0])



text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
text_splitter.split_text(documents[0])

model_name = "shibing624/text2vec-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
text_splitter.split_text(documents[0])
texts = text_splitter.create_documents(documents, metadatas=metadatas)
print(len(texts))
