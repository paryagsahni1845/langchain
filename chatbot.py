'''this will bild foundations for 2 major concepts: 
    RAG - enable a chatbot experience over external source of data
    agents - build a chatbot that can take actions '''

''' how to design and implement a llm powered chatbot (this one will only use language model for coversation)'''

from ast import Store
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

from langchain_groq import ChatGroq
model = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

from langchain_core.messages import HumanMessage
response = model.invoke([HumanMessage(content="hi,my name is bruce wayne and i am batman")])

from langchain_core.messages import AIMessage
response= model.invoke([
        HumanMessage(content="hi,my name is bruce wayne and i am batman"),
        AIMessage(content="Greetings, Mr. Wayne. Or should I say, Batman? It's not every day that I get to meet a billionaire philanthropist with a secret life as a crime-fighting vigilante."),
        HumanMessage(content="what is my name?")
    ]
)

'''its also able to remember the previouse context
    lets discuss about message history(to make it stateful) '''

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(model,get_session_history)

config = {"configurable":{"session_id":"chat1"}}

response=with_message_history.invoke(
    [ HumanMessage(content="hi,my name is bruce wayne and i am batman")],
    config=config
)
print("Chat1 - Bot:", response.content)

'''changing the config will change things '''
config1 = {"configurable":{"session_id":"chat2"}}
response=with_message_history.invoke(
    [ HumanMessage(content="what is my name?")],
    config=config1
)
print("Chat2 - Bot:", response.content)



