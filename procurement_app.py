import streamlit as st
import google.generativeai as genai
import time

# 1. Configuration
st.set_page_config(page_title="My Gemini RAG", page_icon="🤖")
st.title("🤖 Ask My Documents")

# 2. Sidebar for API Key and Uploads
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])

# 3. The Logic (Brain)
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your API Key in the sidebar to start.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat Interface (The "Gemini UI" look)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ask a question about your file..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the context (Simple RAG)
    context = ""
    if uploaded_file is not None:
        # Read the file content
        context = uploaded_file.getvalue().decode("utf-8")
        # Create a "System Prompt" that forces Gemini to use the file
        full_prompt = f"""
        You are a helpful assistant. Use the following document to answer the question. 
        If the answer is not in the document, say so.
        
        DOCUMENT:
        {context}
        
        QUESTION:
        {prompt}
        """
    else:
        full_prompt = prompt # Normal chat if no file

    # Ask Gemini
    try:
        if api_key:
            model = genai.GenerativeModel('gemini-1.5-flash')
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("Thinking...")
                
                response = model.generate_content(full_prompt)
                answer = response.text
                
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {e}")
