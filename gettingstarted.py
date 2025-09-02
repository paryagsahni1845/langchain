'''
## In this quickstart we’ll see how to:
Get setup with LangChain, LangSmith, and LangServe

Use the most basic and common components of LangChain: prompt templates, models, and output parsers

Build a simple application with LangChain

Trace your application with LangSmith

Serve your application with langserve'''

import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPEN_API_KEY"]=os.getenv("OPEN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  #langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

from langchain_groq import ChatO
llm = ChatOpenAI
print(llm)

