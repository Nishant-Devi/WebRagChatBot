import streamlit as st
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
exa_api_key = os.getenv("EXA_API_KEY")

# Define the retriever
retriever = ExaSearchRetriever(api_key=exa_api_key, k=3, highlights=True)

# Define the document prompt
document_prompt = PromptTemplate.from_template("""
<source>
    <url>{url}</url>
    <highlights>{highlights}</highlights>
</source>
""")

# Define the document chain
document_chain = RunnableLambda(
    lambda document: {
        "highlights": document.metadata["highlights"],
        "url": document.metadata["url"]
    }
) | document_prompt

# Define the retriever chain
retriever_chain = retriever | document_chain.map() | (lambda docs: "\n".join([i.text for i in docs]))

# Define the generation prompt
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research assistant. You use xml-formatted context to research people's questions"),
    ("human", """
Please answer the following query based on the provided context. Please cite your sources at the end of the response:
     
"Query: {query}
---
<context>
{context}
</context>
""")
])

# Define the LLM
llm = ChatOpenAI(api_key=openai_api_key)

# Define the main chain
chain = RunnableParallel({
    "query": RunnablePassthrough(),
    "context": retriever_chain,
}) | generation_prompt | llm

# Streamlit UI
st.title("Web Rag Chatbot")

# Input form for the user query
user_query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Fetching the answer..."):
            try:
                # Invoke the chain with the user query
                result = chain.invoke(user_query)

                # The result is an instance of AIMessage, extract the content correctly
                if hasattr(result, 'content'):
                    answer = result.content

                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to get an answer.")