import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import pandas as pd

load_dotenv()


# Safe environment variable assignment
langchain_key = os.getenv('LANGCHAIN_API_KEY')

if langchain_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_key
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'true')
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'Groq ChatBot')
else:
    st.error("LANGCHAIN_API_KEY not found. Please ensure it's set correctly in the .env file.")
    st.stop()  # Stop app execution if API key is missing

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant. Please response to the user queries'),
        ('user','question :{question}')
    ]
)

def generate_response(question, api_key, llm, temperature, max_token):
    ChatGroq.groq_api_key= api_key
    llm = ChatGroq(model = llm, temperature=temperature, max_tokens=max_token, api_key=api_key)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    answer = chain.invoke({'question':question})
    return answer

#Streamlit application 
st.title('Groq-Chatbot Applicaiton')


api_key = st.sidebar.text_input('Enter your Groq API Key', type='password')

## Dropdown for various models
model_options = {
    "LLaMA 8b": "llama3-8b-8192",
    "Gemma2 9b": "gemma2-9b-it",
}

model_name = st.sidebar.selectbox('Select model',list(model_options.keys()))
llm = model_options[model_name]

# Response params

temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value = 0.7)
max_token = st.sidebar.slider('Max Tokens', min_value=50, max_value=300, value=150)

# Main interface for the user 

st.write('Hello I am Groq, what is in your mind?')
user_input = st.text_input('You:')

if user_input:
    response = generate_response(question=user_input, api_key=api_key, llm=llm, temperature=temperature, max_token=max_token)
    st.write(response)
else:
    st.write('Please provide the query!')