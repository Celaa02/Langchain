

#initially import the libraries to be used 
#author: Darly Marcela Vergara Alvarez
#created: 20/02/2024
#update: 21/02/2024
#target: The following code allows the reading of a PDF file 
#and makes it easier to obtain information from it, 
#by asking questions through a chat (Chatgpt)

import streamlit as st
import logging
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

#The components that the interface 
#will have are defined through the steamlit library.
#A decorator is defined to optimize function sharing and loading.
#The function for loading and reading the file is defined.

st.set_page_config(page_title='QuestionPDF')
st.header(body="Performs the query")
OPENAI_API_KEY = st.text_input(label='OpenAI API Key', type='password')
file = st.file_uploader(label="Load your file", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource 
def create_embeddings(file=file):
    """
        This function takes a file type path and returns a list of numeric vetores.

            Args:
                file (str): /cultura_colombiana.pdf

            Returns:
            vectors
    """
    logging.info("there is a file")
    pdf_reader = PdfReader(stream=file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        logging.info("extracts the pages and text of the document")

        
        #The creation of the chunks is performed again.
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
            )        
        chunks = text_splitter.split_text(text=text)
        logging.info("chunks are generated")

        
        #Information is stored locally
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        logging.info("embedding is created according to the model")

        knowledge_base = FAISS.from_texts(chunks, embeddings)
        logging.info("you get the embedding of the created chunks")
        return knowledge_base

        

#The user is prompted for the question, 
#all defined variables are passed to the 
#OpenAI installation.

try: 
    if file:
        knowledge_base = create_embeddings(file)
        logging.info("the existence of the file is validated in order to call the function")
        user_question = st.text_input("Ask a question about your PDF:")
        logging.info("question component")


        if user_question:
            logging.info("the existence of the question is validated")
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            logging.info("load key")
            docs = knowledge_base.similarity_search(user_question, 3)
            logging.info("you get the 3 embedding similar to the question")
            llm = ChatOpenAI(model='gpt-3.5-turbo')
            logging.info("load model OpenAI")
            chain = load_qa_chain(llm, chain_type="stuff")
            logging.info("load type chain")
            reponse = chain.run(input_documents=docs, question=user_question)
            logging.info("the chain is executed with the models, embedding and the question")

            st.write(reponse)
            logging.info("the response is loaded to the user's view")
except Exception as e:
    logging.debug(e)


