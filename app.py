import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from load_docs import load_docs
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn


class ChatRequest(BaseModel):
    prompt: str


app = FastAPI()


@app.get("/")
async def health_check():
    return "The health check is successful!"


@app.get("/api/greet")
def greet():
    return {"message": "Is that a Uvicorn or are my requests magically fast?"}


@app.get("/hello")
def hello_world():
    return {"message": "Hello from FastAPI!"}


@app.post("/chat")
async def chat(
    request: ChatRequest,  # Receive the user's prompt as a Pydantic model
):
    user_prompt = request.prompt
    # Replace this with your actual language model interaction logic
    response = generate_response_from_model(user_prompt)
    return {"response": response}

def generate_response_from_model(prompt):
    # Logic to interact with your chosen language model (e.g., OpenAI, Hugging Face)
    result = qa_chain({"question": prompt, "chat_history": chat_history})
    return result["answer"]


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# load files
documents = load_docs()
chat_history = []

# Now we split the data into chunks
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# create our vector db chromadb
vectordb = Chroma.from_documents(
    documents=docs, embedding=OpenAIEmbeddings(), persist_directory="./data"
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,
)


# ==== Streamlit front-end ====
st.title("VoteGPT")
st.header("Ask about US Voting and Elections... 🤖")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_query():
    input_text = st.chat_input("Ask a question about US Voting and Elections...")
    return input_text


# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain({"question": user_input, "chat_history": chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result["answer"])


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
