from langchain.embeddings import OpenAIEmbeddings # langchain openai based vector embedding model
from langchain.vectorstores import FAISS # vector database
from langchain.document_loaders import PyPDFLoader # extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter # for create chunks

from dotenv import load_dotenv
import os

load_dotenv() # load API key

api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError('OPENAI_API_KEY is not found')

# defines the OpenAI API key in the system environment variables
os.environ['OPENAI_API_KEY'] = api_key

# load Faqs file
loader = PyPDFLoader('musteri_destek_faq.pdf')

# creates langchain documents object
documents = loader.load()

# to divide the text into chunks
# splitter tries to preserve sentence or paragraph integrity while dividing the text into meaningful chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # each chunk include max 500 character
    chunk_overlap = 50  # each chunk can get max 50 character from previous one
)

# create chunks
docs = text_splitter.split_documents(documents)

# openai embedding model
embedding = OpenAIEmbeddings(model= 'text-embedding-3-large') # bu model kaliteli ve turkce destegi cok iyi

# faiss vector database, transforms the chunked text into vectors using embeddings and creates index
vectordb = FAISS.from_documents(docs, embedding)

# save the created Faiss cvector db to the local disk
vectordb.save_local('faq_vectorstore')

print('The embedding and vector database were created successfully.')