
from langchain.chat_models import ChatOpenAI # fpr openai powered LLM models
from langchain.chains import ConversationalRetrievalChain # RAG + chat chain
from langchain.vectorstores import FAISS # faiss vector database
from langchain.embeddings import OpenAIEmbeddings # to vectorize the text
from langchain.memory import ConversationBufferMemory # memory that holds the chat history

from dotenv import load_dotenv
import os

load_dotenv() # load environment variables from the .env file
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY is not found')

os.environ['OPENAI_API_KEY'] = api_key

# start embedding model (text -> vector)
embedding = OpenAIEmbeddings( model='text-embedding-3-large')

# load the created vector db
vectordb = FAISS.load_local(
    'faq_vectorstore', # saved Faiss database folder
    embedding, # it should be the same embedding model
    allow_dangerous_deserialization=True # show the pickle security warning
)

# create memory for chat history
memory = ConversationBufferMemory(
    memory_key='chat_history', # store chat history with this key
    return_messages= True, # the past messages are returned in their entirety; since it was a short chat, we can return all of them
)

# language model operates with zero randomness, providing fixed answers
llm = ChatOpenAI(
    model_name = 'gpt-4', # the language model used
    temperature = 0, # deterministic (same input gives same output)
)

# create RAG + memory chain
# - llm
# - faiss retriever: most similar 3 files (k=3)
# - memory

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(search_kwargs = {'k': 3}),
    memory = memory,
    verbose = True
)

print('Welcome to the Customer Support Bot')

while True:
    user_input = input('You: ')
    if user_input.lower() == 'quit':
        break

    # The user's question is fed into the LLM + RAG + memory chain.
    response = qa_chain.run(user_input)
    print('Customer Support Bot: ', response)