'''this will bild foundations for 2 major concepts: 
    RAG - enable a chatbot experience over external source of data
    agents - build a chatbot that can take actions '''

''' how to design and implement a llm powered chatbot (this one will only use language model for coversation)'''

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

'''changing the config will change things '''
config1 = {"configurable":{"session_id":"chat2"}}
response=with_message_history.invoke(
    [ HumanMessage(content="what is my name?")],
    config=config1
)

'''### Prompt templates
Prompt Templates help to turn raw user information into a format that the LLM can work with.In this case, the raw user input is just a message,
 which we are passing to the LLM. Let's now make that a bit more complicated. 
 First, let's add in a system message with some custom instructions (but still taking messages as input). 
 Next, we'll add in more input besides just the messages.'''

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Amnswer all the question to the nest of your ability"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain=prompt|model
chain.invoke({"messages":[HumanMessage(content="Hi My name is bruce wayne")]})
with_message_history=RunnableWithMessageHistory(chain,get_session_history)

config = {"configurable": {"session_id": "chat3"}}
response=with_message_history.invoke(
    [HumanMessage(content="Hi My name is bruce wayne")],
    config=config
)

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)


## Add more complexity

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response=chain.invoke({"messages":[HumanMessage(content="Hi My name is bruce")],"language":"Hindi"})
print(response.content)

'''Let's now wrap this more complicated chain in a Message History class. This time, because there are multiple keys in the input,
    we need to specify the correct key to use to save the chat history.'''

with_message_history=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable": {"session_id": "chat4"}}
repsonse=with_message_history.invoke(
    {'messages': [HumanMessage(content="Hi,I am bruce wayne")],"language":"Hindi"},
    config=config
)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Hindi"},
    config=config,
)


from langchain_core.messages import SystemMessage,trim_messages
trimmer=trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)