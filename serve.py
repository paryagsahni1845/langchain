from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
import uvicorn

# Load environment variables
load_dotenv()

# Get GROQ API key and handle missing key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator. Translate the following text into {language}."),
    ("user", "{text}")
])

# Define the parser
parser = StrOutputParser()

# Create the LangChain expression language chain
chain = prompt | model | parser

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Translation Server",
    version="1.0",
    description="A simple API for translating text using LangChain and Groq."
)

# Add routes for the chain
add_routes(
    app,
    chain,
    path="/translate",
    playground_type="default"
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)