import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd
import openai
import os
import fitz
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)

from table_transformer import detect_and_crop_save_table

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def pdf_page_to_image(pdf_path):
    pdf_document = fitz.open(pdf_path)
    output_directory_path, _ = os.path.splitext(pdf_path)
    # Iterate through each page and convert to an image
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]
        # Convert the page to an image
        pix = page.get_pixmap()
        # Create a Pillow Image object from the pixmap
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Save the image
        image.save(f"{output_directory_path}/page_{page_number + 1}.png")
    pdf_document.close()

data_dir = "/home/abaoyang/app/core-abao-test/data"
pdf_path = f"{data_dir}/llama2.pdf"
pdf_page_to_image(pdf_path)

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 9:
                break

image_paths = []
for img_path in os.listdir(f"{data_dir}/llama2"):
    image_paths.append(str(os.path.join(f"{data_dir}/llama2", img_path)))

# plot_images(image_paths[9:12])


openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_KEY, max_new_tokens=1500
)

# Read the images
documents_images = SimpleDirectoryReader(f"{data_dir}/llama2/").load_data()
image_prompt = """
    Please load the table data and output in the json format from the image.
    Please try your best to extract the table data from the image.
    If you can't extract the table data, please summarize image and return the summary.
"""
response = openai_mm_llm.complete(
    prompt=image_prompt,
    image_documents=[documents_images[15]],
)
print("===== Directly from the image =====")
print(response)
print("===================================\n\n")


# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_index")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents_images,
    storage_context=storage_context,
)
retriever_engine = index.as_retriever(image_similarity_top_k=2)
query = "Compare llama2 with llama1?"
assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
# retrieve for the query using text to image retrieval
retrieval_results = retriever_engine.text_to_image_retrieve(query)
# ['/home/abaoyang/app/core-abao-test/data/llama2/page_50.png', '/home/abaoyang/app/core-abao-test/data/llama2/page_8.png']

retrieved_images = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_images.append(res_node.node.metadata["file_path"])
    else:
        # (res_node, source_length=200)

# plot_images(retrieved_images)

for file_path in retrieved_images:
    detect_and_crop_save_table(file_path, cropped_table_directory=f"{data_dir}/table_images")


# Read the cropped tables
image_documents = SimpleDirectoryReader(f"{data_dir}/table_images/").load_data()

response = openai_mm_llm.complete(
    prompt="Compare llama2 with llama1?",
    image_documents=image_documents,
)
print("===== From the cropped tables =====")
print(response)
print("===================================\n\n")
