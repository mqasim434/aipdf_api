from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def handler(event, context):
    # Check if a PDF file URL was provided
    if 'pdf_url' not in event:
        return {'error': 'No PDF file URL provided'}, 400

    pdf_url = event['pdf_url']

    # Download the PDF file and extract text (you'll need to implement this)
    text = download_and_extract_text(pdf_url)

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Get the user's question from the request
    user_question = event.get('question')

    # Perform similarity search
    docs = knowledge_base.similarity_search(user_question)

    # Initialize OpenAI language model
    llm = OpenAI(api_key=OPENAI_API_KEY)

    # Load question-answering model
    chain = load_qa_chain(llm, chain_type="stuff")

    # Get the answer to the user's question
    response = chain.run(input_documents=docs, question=user_question)

    return {'answer': response}, 200
