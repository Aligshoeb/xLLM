from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI



def get_converstion_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_pdf_text(pdf_docs):
    text ="The PDF texts starts here: "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size= 1000, chunk_overlap= 250)
    chunks = text_splitter.split_text(raw_text)
    return chunks

user_template = ":nerd_face: HUMAN <h6>{{MSG}}</h5>"
bot_template = ":robot_face: AI <h6>{{MSG}}</h5>"


def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)
    st.session_state.chat_history=response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    st.title("Welcome to Local Data AI")

    user_question = st.text_input("What you want to ask regarding uploded doc/docs?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload all docs here")
        pdf_doc = st.file_uploader("Support multiple files", accept_multiple_files=True)
        if st.button("Read Docs"):
            with st.spinner("Hold On"):

                raw_text = get_pdf_text(pdf_doc)
                #st.write(raw_text)

                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                vectorstore = get_vectorstore(text_chunks)
                #st.write(vectorstore)


                st.session_state.conversation = get_converstion_chain(vectorstore)
                #st.write(st.session_state.conversation)



if __name__ == '__main__':
    main()