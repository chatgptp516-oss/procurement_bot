import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="My AI Knowledge Base", layout="wide")
st.title("🤖 Chat with my PDF")

# --- 2. SIDEBAR: CREDENTIALS & DATA ---
with st.sidebar:
    st.header("1. Security")
    # This captures the API key from the UI (Password mode)
    api_key = st.text_input("Enter Google API Key:", type="password")
    
    st.header("2. Upload Document")
    uploaded_files = st.file_uploader("Upload your PDF", accept_multiple_files=True, type=["pdf"])
    
    process_button = st.button("Submit & Build Brain")

# --- 3. MAIN LOGIC (Building the Brain) ---
if process_button:
    if not api_key:
        st.error("⚠️ Stop! You must enter the API Key first.")
    elif not uploaded_files:
        st.error("⚠️ Stop! You must upload a PDF file.")
    else:
        with st.spinner("Reading PDF and building knowledge base..."):
            try:
                # A. Extract Text from PDF
                text = ""
                for pdf in uploaded_files:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                
                # B. Split Text into small chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                chunks = text_splitter.split_text(text)

                # C. Turn chunks into Math (Embeddings) using your Key
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                
                # D. Save to FAISS (The Vector Database)
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                
                st.success("✅ Done! The AI has read your PDF. You can now chat.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Question
user_question = st.chat_input("Ask a question about the file...")

if user_question:
    # 1. Show User Question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Generate Answer (Only if Key and Database exist)
    if api_key and os.path.exists("faiss_index"):
        try:
            # Load the Vector Database
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Find relevant snippets
            docs = new_db.similarity_search(user_question)

            # Setup Gemini (using the Cheap "Flash" model)
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

            # The Instructions (System Prompt)
            prompt_template = """
            Answer the question accurately using ONLY the context provided below.
            If the answer is not in the context, say "I cannot find the answer in the provided PDF."
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

            # Run the AI
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("Thinking...")
                
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                
                message_placeholder.markdown(response["output_text"])
                st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        if not api_key:
            st.warning("⚠️ Please enter your API Key in the sidebar.")
        else:
            st.warning("⚠️ Please upload a PDF and click 'Submit' first.")
