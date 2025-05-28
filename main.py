import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

langchain_key = os.getenv('LANGCHAIN_API_KEY')

if langchain_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_key
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'true')
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'Groq ChatBot')
else:
    st.error("LANGCHAIN_API_KEY not found. Please ensure it's set correctly in the .env file.")
    st.stop()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Please response to the user queries'),
        ('user', 'question :{question}')
    ]
)

def generate_response(question, api_key, llm, temperature, max_token):
    ChatGroq.groq_api_key = api_key
    llm_instance = ChatGroq(model=llm, temperature=temperature, max_tokens=max_token, api_key=api_key)
    parser = StrOutputParser()
    chain = prompt | llm_instance | parser
    answer = chain.invoke({'question': question})
    return answer

# --- Streamlit UI Enhancements ---

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE; font-weight: bold;'>
        🤖 Groq Chatbot Application
    </h1>
    <p style='text-align: center; color: #555; font-size: 18px;'>
        Ask me anything. Powered by Groq & LangChain.
    </p>
    <hr style='margin-bottom: 30px;'>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings ⚙️")
    api_key = st.text_input('Enter your Groq API Key', type='password', help="Your Groq API key for authentication")

    model_options = {
        "LLaMA 8b": "llama3-8b-8192",
        "Gemma2 9b": "gemma2-9b-it",
    }
    model_name = st.selectbox('Select Model', list(model_options.keys()))
    llm = model_options[model_name]

    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, help="Higher value = more creative responses")
    max_token = st.slider('Max Tokens', min_value=50, max_value=300, value=150, help="Max length of generated answer")

st.markdown("### 💬 Chat with Groq")

user_input = st.text_input('You:', placeholder="Type your message here and press Enter")

if user_input:
    with st.spinner("Generating response..."):
        response = generate_response(question=user_input, api_key=api_key, llm=llm, temperature=temperature, max_token=max_token)
    st.markdown(f"**Groq:** {response}")
elif user_input == "":
    st.info("Please enter a question to get started!")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>Made by <a href='https://www.linkedin.com/in/aadarsh-vani-a60a641a0/' target='_blank' style='text-decoration:none; color:#4B8BBE;'>Aadarsh Vani</a></p>",
    unsafe_allow_html=True,
)
