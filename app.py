import streamlit as st
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

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

# Streamlit app
def main():
    st.title("Streamlit with REST Endpoint")
    st.write("Check out the endpoint at `/hello`")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    main()