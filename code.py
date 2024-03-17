import os
import openai
import sys
import tempfile
import time

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
from langchain_community.document_loaders import TextLoader

from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec, PodSpec

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import PineconeVectorStore

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FILE_NAME = "book.txt"


model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

parser = StrOutputParser()

template = """
Answer the question based on the context below. If you cant 
answer the question, reply "I dont know"

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model | parser
translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)

translation_chain = (
    {"answer" : chain, "language" : itemgetter("language")} | translation_prompt | model | parser
)

translation_chain.invoke (
    {
    "context" : "Mary's sister is Susana and her brother name is raul",
    "question": "How many siblings does mary have?",
    "language" : "Hindi",
}
)

# Load the book
loader = PyPDFLoader('AtomicHabits.pdf')
book_pages = loader.load()
len(book_pages)

if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'w') as file:
            # Write content into the file
            file.write(book_pages)




index_name = 'langchain-book-rag-fast'

def call_pinecode():
    use_serverless = True
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')

    # configure client
    pc = Pinecone(api_key=pinecone_api_key)

    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-west-2')
    else:
        # if not using a starter index, you should specify a pod_type too
        spec = PodSpec()

    # check for and delete index if already exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # create a new index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    index = pc.Index(index_name)
    index.describe_index_stats()

call_pinecode()



def load_document_pinecone():
    loader = TextLoader(FILE_NAME)
    text_documents = loader.load()

    # Split the book
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1050,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    embeddings = OpenAIEmbeddings()
    documents = text_splitter.split_documents(text_documents)
    #vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)
    pinecone = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=index_name
    )

    return pinecone

pinecone = load_document_pinecone()

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

chain.invoke("Can you summarize the details?")
chain.invoke("Tell me about the chapter 1 of the book?")