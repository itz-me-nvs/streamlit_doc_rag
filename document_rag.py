from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # type: ignore
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.title("Workmate AI Document Chatbot")

# File uploader for user to upload PDF
upload_files = st.file_uploader(
    "Upload your PDF documents", accept_multiple_files=True, type=["pdf"]
)

# Form to handle the input and submission
with st.form("question_form"):
    input_text = st.text_input("Enter your question", value=st.session_state.input_text)
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Define your logic here
    if input_text and upload_files:
        # Load PDF and process data
        docs = []
        for upload_file in upload_files:

          # save the file in temporary directory
          file_path = os.path.join("temps", upload_file.name)
          os.makedirs("temps", exist_ok=True)

          with open(file_path, "wb") as f:
               f.write(upload_file.getbuffer())

          # Use PyPDFLoader to load the saved PDF file
          loader = PyPDFLoader(file_path)
          docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        # HuggingFace Embedding
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = "docs/chroma/"

        vectordb = Chroma.from_documents(
            documents=splits, embedding=embedding, persist_directory=persist_directory
        )

        # QA Chain
        llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.7)
        compressor = LLMChainExtractor.from_llm(llm)

        template = """Based on the provided context, answer the question below. Only respond to questions directly related to the uploaded documents. If the answer is not found in the context, simply state that you don't know and avoid making up information. Be concise and limit your response to three sentences. End your answer with "Thanks for asking!"
        Context:
        {context}
        Question:
        {question}
        Answer:"""


        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        qa_chain_mr = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        # Generate response
        response = qa_chain_mr({"query": input_text})
        st.write(response["result"])
        st.session_state.input_text = ""  # Clear the input text after processing

        # clear the form after use
        # st.form_submit_button("Submit")

        # clear temporary directory file after use
        for file in upload_files:
            temp_path = os.path.join("temps", file.name)
            os.remove(temp_path)

    elif not upload_files:
       st.write("Please upload at least one PDF document.")
    else:
        st.write("Please enter a question.")
