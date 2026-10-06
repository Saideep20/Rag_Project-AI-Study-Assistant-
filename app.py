import streamlit as st
from day26_rag import documents


st.title("AI Study Assistant")

question = st.text_input("Ask a question")
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")


if uploaded_file and question:
    st.write("PDF and question received!")

    answer = documents(uploaded_file, question)

    st.write(answer)