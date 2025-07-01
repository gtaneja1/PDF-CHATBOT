
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
import streamlit as st
import tempfile


from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS    # Facebook AI Similarity Search it used like vector storage? and finidng the vector?
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from langchain.text_splitter  import RecursiveCharacterTextSplitter


st.title("PDF Chatbot")

if "memory" not in st.session_state:
    st.session_state['previous_chats'] = []
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded")
  
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 200)   #from the 1000th token we have to backtrace till 200 tokens
# which is chunk overlap and the second chunk(1000) starts from the word we stop at after backtracing == overlapping??
    text_chunks = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

  
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)


        
    llm = ChatOllama(model="tinyllama", base_url='http://localhost:11434')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key = "answer",
    )


    question = st.text_input("Ask a question about the PDF")
    if question:
        with st.spinner("Processing..."):
            answer = qa_chain({"question": question})
            st.session_state['previous_chats'] += [answer['chat_history']]
    
            with st.expander('Chat History'):
                for i in range(len(st.session_state['previous_chats'])-1):
                    with st.expander(st.session_state['previous_chats'][i][0].content):
                        st.write(st.session_state['previous_chats'][i][1].content)

            st.write(answer['answer'])
           