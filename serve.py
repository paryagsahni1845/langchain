from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes   #create apis
load_dotenv()

#model
# groq_api_key=os.getenv("GROQ_API_KEY")
model = ChatGroq(api_key=groq_api_key,model="llama-3.3-70b-versatile", temperature=0.7)

#prompt template - convert into list of message 
from langchain_core.prompts import ChatPromptTemplate

template = "translate the following into {language}"

prompt = ChatPromptTemplate.from_messages(
    [("system",template),("user","{text}")]
)

#parser
parser = StrOutputParser()

#chain
chain = prompt|model|parser

#app
app = FastAPI(title="langchain server",
              version="1.0",
              description="simple app interface")

#add route
add_routes(
    app,chain,path="/chain",playground_type="default"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)