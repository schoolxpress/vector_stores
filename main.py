from langchain.vectorstores import Pinecone as langpine
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import  ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec
import os
import pypdf
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
pc =Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))



def main():
    #step 1: LOAD THE PDF FILE
    loader = PyPDFLoader(file_path='pdf/samplepdf.pdf')
    pdf_file = loader.load()

    #step2: split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    documents = text_splitter.split_documents(pdf_file)

    #step 3: create your vector embedding
    embedding = OpenAIEmbeddings()

    for index_name in pc.list_indexes():
        print(index_name)

    docsearch = langpine.from_documents(documents=documents, embedding=embedding, index_name=os.environ.get('PINECONE_INDEX'))
    print("THIS WORKS")

    query = "What was the outcome of the judgement?"
    query_results = docsearch.similarity_search(query=query, k=2)
    for result in query_results:
        print(result.page_content)



if __name__ == "__main__":
    main()