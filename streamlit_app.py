import streamlit as st # a python library to create a web interface

from langchain_openai import ChatOpenAI, OpenAIEmbeddings # langchain OpenAI moduls: chat llm and embedding model

from langchain_community.vectorstores import FAISS # faiss vector db
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os
import tempfile # for temporary file operations

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# set the streamlit page title and icon
st.set_page_config(
    page_title='Customer Support Bot',
    page_icon='💬'
)
st.title('PDF Support Bot (RAG + Memory)')
st.write('Upload a pdf file, ask questions about its content.')

# pdf upload component
uploaded_file = st.file_uploader('Upload Your PDF File', type='pdf', key= 'pdf_uploader')

# if the user uploaded a new pdf and it is not the same as the one previously uploaded
if uploaded_file is not None:
    if 'last_uploaded_name' not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        # let's send a 'processing' message to the user
        with st.spinner('PDF is processing'):
            # write the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name # path of the temporary file

            # load PDF content with PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # divide text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
            docs = splitter.split_documents(documents)

            # vectorize text with openai embedding
            embedding = OpenAIEmbeddings(model='text-embedding-3-large')

            # vector db with Faiss
            vectordb = FAISS.from_documents(docs, embedding)

            # create memory and llm with gpt 4
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages= True)
            llm = ChatOpenAI(model_name = 'gpt-4', temperature=0)

            # rag + memory chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever= vectordb.as_retriever(search_kwargs = {'k' : 3}),
                memory=memory
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.last_uploaded_name = uploaded_file.name # to prevent the same file from being processed again
        
        st.success('the PDF file was processed successfully')

if 'qa_chain' in st.session_state: # if pdf has been processed
    # take user question
    user_question = st.text_input('👥 Write your question: ')

    if user_question:
        response = st.session_state.qa_chain.invoke(user_question) # send the question to the langchain chain
        st.session_state.chat_history.append(('👥', user_question)) # add the user message to the history
        st.session_state.chat_history.append(('🤖'), response['answer']) # add the model answer to the history

    if st.session_state.chat_history:
        st.subheader('📂 Chat History: ')
        for sender, msg in st.session_state.chat_history:
            st.markdown(f'**{sender}**: {msg}')