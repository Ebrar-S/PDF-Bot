# 💬 PDF Support Bot with RAG and Memory

This project is an intelligent chatbot designed to answer questions about the content of a PDF document. Users can upload a PDF, and the bot, powered by a Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) system, will provide relevant answers. The chatbot also maintains a conversation history for a more contextual and interactive experience.

## ✨ Overview

The core functionality of this application is to transform a static PDF document into an interactive conversational partner. Instead of manually searching through a potentially long document, you can simply ask questions in natural language and receive concise, relevant answers sourced directly from the document's content.

This is achieved by combining the power of modern LLMs (like OpenAI's GPT-4) with a vector database (FAISS) to create a powerful question-answering system. The application is presented through a user-friendly web interface created with Streamlit.

## ⚙️ How It Works

The project follows the **Retrieval-Augmented Generation (RAG)** architecture. Here's a step-by-step breakdown of the process:

1.  **PDF Loading & Text Splitting:** When a user uploads a PDF, the application first extracts all the text. This text is then broken down into smaller, meaningful chunks. This is crucial for the embedding process, as LLMs have a context limit.

2.  **Embedding & Vector Storage:** Each text chunk is converted into a numerical representation called an "embedding" using an OpenAI embedding model. These embeddings capture the semantic meaning of the text. All these vector embeddings are then stored in a **FAISS** vector database, which allows for efficient similarity searches.

3.  **User Query:** The user asks a question through the Streamlit web interface.

4.  **Retrieval:** The user's question is also converted into an embedding. The FAISS database is then queried to find the text chunks with embeddings that are most similar to the question's embedding. These relevant chunks are "retrieved" from the document.

5.  **Generation:** The original question, the retrieved text chunks, and the conversation history are all passed to the LLM (e.g., GPT-4). The LLM uses this context to generate a comprehensive and accurate answer.

6.  **Memory:** The system uses a `ConversationBufferMemory` to keep track of the conversation history. This allows the chatbot to understand follow-up questions and provide more context-aware responses.

## 🛠️ Key Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web user interface.
* **LangChain:** A framework for developing applications powered by language models. It's used to orchestrate the entire RAG pipeline, from text splitting to chaining the retriever and LLM.
* **OpenAI:** For providing the powerful LLM (`gpt-4`) and the text embedding model (`text-embedding-3-large`).
* **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search and clustering of dense vectors. It serves as our vector database.
* **PyPDFLoader:** For loading and extracting text from PDF documents.

## 📂 Project Structure

The project is composed of three main Python scripts:

* `load_pdf_and_embedding.py`: A script to perform the initial processing of a PDF file. It loads the document, splits it into chunks, creates embeddings, and saves them to a local FAISS vector store (`faq_vectorstore`). This is a one-time setup step for a given document.
* `chatbot_rag_memory.py`: A command-line version of the chatbot that loads the pre-processed FAISS database and allows for a conversational Q&A session in the terminal.
* `streamlit_app.py`: The main file that creates the web application. It includes the logic for file uploading, on-the-fly processing of the PDF, and the interactive chat interface with conversation history.

## 🚀 Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Ensure your `requirements.txt` file is up-to-date. Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up OpenAI API Key:**
    Add your OpenAI API key in the `.env` file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## ▶️ How to Use

### Running the Web Application

This is the main way to interact with the chatbot. The Streamlit app handles PDF processing on the fly.

* Run the Streamlit application from your terminal:
    ```bash
    streamlit run streamlit_app.py
    ```
* Open your web browser and navigate to the local URL provided by Streamlit.
* Upload a PDF file using the file uploader.
* Wait for the processing to complete.
* Start asking questions about your document!