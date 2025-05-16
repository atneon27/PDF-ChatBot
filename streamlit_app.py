import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

load_dotenv()

models = {
    "llama-3": {
        "llm": lambda: ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.3, model_name="llama-3.3-70b-versatile"),
        "embedding": lambda: OpenAIEmbeddings(model="text-embedding-3-large")
    },
    "llama-4": {
        "llm": lambda: ChatGroq(temperature=0.3, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"),
        "embedding": lambda: GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    },
    "gpt-4o": {
        "llm": lambda: ChatOpenAI(model="gpt-4o"),
        "embedding": lambda: OpenAIEmbeddings(model="text-embedding-3-small")
    },
    "gemini-2.5-flash": {
        "llm": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temprature=0.3),
        "embedding": lambda: GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    },
    "gemma-3": {
        "llm": lambda: ChatGoogleGenerativeAI(model="gemma-3-27b-it", temprature=0.3),
        "embedding": lambda: OpenAIEmbeddings(model="text-embedding-ada-002")
    },
    # "gpt-4.1": {
    #     "llm": lambda: ChatOpenAI(model="gpt-4.1"),
    #     "embedding": lambda: OpenAIEmbeddings(model="text-embedding-3-small")
    # },
    # "gpt-o1": {
    #     "llm": lambda: ChatOpenAI(model="o1"),
    #     "embedding": lambda: OpenAIEmbeddings(model="text-embedding-3-small")
    # }, 
    # "gemini-1.5-pro": {
    # },
    #     "llm": lambda: ChatGoogleGenerativeAI(model="gemini-1.5-pro", temprature=0.3),
    #     "embedding": lambda: GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # "mistral-medium-3": {
    #     "llm": lambda: ChatMistralAI(model="mistral-medium-latest", temperature=0.3),
    #     "embedding": lambda: MistralAIEmbeddings(model="mistral-embed")
    # },
    # "ministral": {
    #     "llm": lambda: ChatMistralAI(model="ministral-8b-latest", temperature=0.3),
    #     "embedding": lambda: MistralAIEmbeddings(model="mistral-embed")
    # }
}

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, model_name=None):
    embeddings = models[model_name]["embedding"]()
    vector_store=FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(f"faiss_index/{model_name}")

def get_conversational_chain(retriever, model_name=None):
    prompt_template = '''
    You are an intelligent assistant designed to answer user questions based solely on the provided context.

    Instructions:
        - Do not respond to any query with more then 600 words at max.
        - Only use information explicitly stated in the provided context to answer the question.
        - If the question cannot be answered based on the context, respond clearly and politely that the necessary information is not available in the document.
        - Do not assume, infer, or fabricate any information that is not directly present in the context.
        - Structure your response for clarity and readability, using bullet points, headings, or code blocks as needed.
        - Keep the answer short and concise, giving only the required information quickly

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=models[model_name]["llm"](),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )
    # model = models[model_name]["llm"]()
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name, chat_history):
    embeddings = models[model_name]["embedding"]()
    index_name = f"faiss_index/{model_name}"

    new_db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()

    # docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(retriever, model_name)

    # response = chain(
    #     {"input_documents": docs, "question": user_question},
    #     return_only_outputs=True
    # )

    for q, a in chat_history:
        chain.memory.chat_memory.add_user_message(q)
        chain.memory.chat_memory.add_ai_message(a)
    
    response = chain(
        {
            "question": user_question,
        },
        return_only_outputs=True
    )
    
    st.write(response["answer"])
    chat_history.append((user_question, response["answer"]))
    
    return chat_history


def main():
    st.title("PDF Chat")

    with st.sidebar:     
        st.subheader("Choose Model")
        selected_model = st.selectbox(
            "Select a language model",
            options=list(models.keys()),
            index=0
        )

        st.divider()

        st.subheader("Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload your PDF document",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    for model_name, _ in models.items():
                        get_vector_store(text_chunks, model_name)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF before processing.")

    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.subheader("Chat History")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        message = st.chat_message(f"user{i}", avatar="🧑‍💻")
        message.write(question)
        
        message = st.chat_message(f"assistant{i}", avatar="🤖")
        message.write(answer)

    user_question = st.chat_input("Type your question here") if hasattr(st, "chat_input") else st.text_input("Your question:")

    if user_question:
        message = st.chat_message("user")
        message.write(user_question)

        with st.chat_message("assistant"):
            st.session_state.chat_history = user_input(user_question, selected_model, st.session_state.chat_history)

if __name__ == "__main__":
    main()
