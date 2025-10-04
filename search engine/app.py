import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper 
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults 
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

# Initialize API wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(name="search")

# Streamlit UI
st.title("LangChain Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search."}
    ]

# Display past messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Save user input
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
    tools = [search, arxiv, wiki]

    # Create agent
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(prompt, callbacks=[st_cb])

        # Save assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
