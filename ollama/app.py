import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#langsmith tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond."),
    ("user", "question: {question}")
])

# streamlit framework
st.title("Langchain demo with gemma")
input_text=st.text_input("what question u have in mind?")

##ollama model
llm=Ollama(model= "gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
