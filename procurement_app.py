import streamlit as st
import os
from Pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="My Smart AI", layout="wide")
st.title("🤖 Chat with Context & Memory")

# --- 2. SIDEBAR & STATE MANAGEMENT ---
# Initialize Session State variables if they don't exist
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Google API Key:", type="password")
    
    st.header("2. Data")
    uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True, type=["pdf"])
    
    # We use a callback to handle the processing so the message sticks
    if st.button("Submit & Process"):
        if not api_key:
            st.error("⚠️ Please enter API Key first.")
        elif not uploaded_files:
            st.error("⚠️ Please upload a file.")
        else:
            with st.spinner("Building the Brain..."):
                try:
                    # A. Read PDF
                    text = ""
                    for pdf in uploaded_files:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            text += page.extract_text() or ""
                    
                    # B. Split Text
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    chunks = text_splitter.split_text(text)

                    # C. Create Embeddings & Save
                    # Note: We use the 'text-embedding-004' model you fixed earlier
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    vector_store.save_local("faiss_index")
                    
                    # D. Set Flag to True (This remembers the PDF is ready)
                    st.session_state.db_ready = True
                    st.success("✅ Brain successfully built!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Persistent Status Indicator
    if st.session_state.db_ready:
        st.success("🟢 System Ready: PDF Loaded")
    else:
        st.info("⚪ System Idle: Waiting for PDF")

# --- 3. HELPER FUNCTION: GET CHAT HISTORY ---
def get_chat_history_string():
    """Converts the list of message objects into a simple string for the AI to read."""
    history_str = ""
    # We take the last 4 messages to save costs/tokens
    for msg in st.session_state.messages[-4:]: 
        role = "User" if msg["role"] == "user" else "AI"
        history_str += f"{role}: {msg['content']}\n"
    return history_str

# --- 4. CHAT INTERFACE ---
# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask me anything...")

if user_question:
    # 1. Show User Question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Generate Answer
    if api_key:
        try:
            # Prepare Memory
            chat_history = get_chat_history_string()

            # --- MODE 1: RAG (Use PDF + Memory) ---
            if st.session_state.db_ready and os.path.exists("faiss_index"):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question)
                
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
                
                # UPDATED PROMPT: Now includes {chat_history}
                prompt_template = """
                You are a helpful AI assistant.
                
                HISTORY OF CONVERSATION:
                {chat_history}
                
                CONTEXT FROM DOCUMENT:
                {context}
                
                USER QUESTION:
                {question}
                
                INSTRUCTIONS:
                1. Use the "History" to understand the flow (e.g., if user says "it", know what "it" refers to).
                2. Use the "Context" to answer specific questions.
                3. If the answer is not in the context, use your general knowledge (mark it as General Knowledge).
                
                Answer:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response = chain(
                        {"input_documents": docs, "question": user_question, "chat_history": chat_history}, 
                        return_only_outputs=True
                    )
                    answer = response["output_text"]
                    message_placeholder.markdown(answer)

            # --- MODE 2: GENERAL CHAT (Memory Only) ---
            else:
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                
                # We manually combine History + Question for General Chat
                full_prompt = f"""
                Here is the conversation history:
                {chat_history}
                
                User's new question: {user_question}
                
                Answer the user naturally.
                """
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response = model.invoke(full_prompt)
                    answer = response.content
                    message_placeholder.markdown(answer)

            # Save Answer
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Enter API Key in sidebar.")
