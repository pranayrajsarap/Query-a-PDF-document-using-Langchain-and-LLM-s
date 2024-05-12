from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
import os


os.environ["OPENAI_API_KEY"] = #api key
os.environ["SERPAPI_API_KEY"] = ""

#Read pdf file
pdfreader = PdfReader('bs2023_24.pdf')

from typing_extensions import Concatenate

#read text from pdf
raw_text = ''
for i,page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


len(texts)
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts,embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "How much the agriculture target will be increased by"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs,question=query)

query1 = "What is the vision for amrit kaal"
docs = document_search.similarity_search(query1)
chain.run(input_documents=docs,question=query1)
