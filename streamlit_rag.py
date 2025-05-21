# import streamlit as st
# from rag_pipeline import final_rag_chain
# from PIL import Image

# st.set_page_config(page_title="UNITApedia AI Assistant", layout="wide")

# st.image("logos.jpg",width=300)
# # Page title
# st.markdown("<h1 style='text-align: center;'>🤖 UNITApedia AI Assistant</h1>", unsafe_allow_html=True)


# # Custom CSS for message styling
# st.markdown("""
#     <style>
#     .message-container {
#         background-color: #f1f1f1;
#         border-radius: 10px;
#         padding: 10px;
#         margin-bottom: 15px;
#     }
#     .user-message {
#         background-color: #d1e7dd;
#         border-left: 5px solid #0f5132;
#     }
#     .bot-message {
#         background-color: #fff3cd;
#         border-left: 5px solid #664d03;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Session history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # User input
# query = st.text_input("❓ Ask a question about the UNITA Alliance:")

# # Ask button only
# ask = st.button("Ask")

# # Handle ask button
# if ask and query:
#     with st.spinner("Thinking..."):
#         response = final_rag_chain.invoke({"question": query})
#         st.session_state.history.append((query, response))

# # Display chat history
# if st.session_state.history:
#     st.markdown("### 🗂️ Conversation History")
#     for q, a in reversed(st.session_state.history):
#         st.markdown(f"<div class='message-container user-message'><strong>You:</strong><br>{q}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div class='message-container bot-message'><strong>UNITA Assistant:</strong><br>{a}</div>", unsafe_allow_html=True)

# # Clear button below history
# if st.session_state.history:
#     if st.button("Clear Conversation"):
#         st.session_state.history = []
#         query = ""

import streamlit as st
from rag_pipeline import final_rag_chain
from PIL import Image
import logging
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)

st.set_page_config(page_title="UNITApedia AI Assistant", layout="wide")

st.image("logos.jpg", width=300)
st.markdown("<h1 style='text-align: center;'>🤖 UNITApedia AI Assistant</h1>", unsafe_allow_html=True)

# Estilo CSS
st.markdown("""
    <style>
    .message-container {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .user-message {
        background-color: #d1e7dd;
        border-left: 5px solid #0f5132;
    }
    .bot-message {
        background-color: #fff3cd;
        border-left: 5px solid #664d03;
    }
    </style>
""", unsafe_allow_html=True)

# Historial
if "history" not in st.session_state:
    st.session_state.history = []

# Entrada del usuario
query = st.text_input("❓ Ask a question about the UNITA Alliance:")

# Botón
ask = st.button("Ask")

# Procesar pregunta
if ask and query:
    logging.info(f"User question: {query}")
    with st.spinner("Thinking..."):
        try:
            start_time = time.time()
            response = final_rag_chain.invoke({"question": query})
            end_time = time.time()
            
            duration = round(end_time - start_time, 2)
            token_count = len(response.split())

            st.session_state.history.append((query, response, duration, token_count))

            logging.info(f"Response OK | Duration: {duration}s | Tokens: {token_count}")

        except Exception as e:
            logging.error(f"RAG error: {e}")
            st.error("⚠️ Something went wrong while processing your question.")

# Mostrar historial
if st.session_state.history:
    st.markdown("### 🗂️ Conversation History")
    for q, a, t, tok in reversed(st.session_state.history):
        st.markdown(f"<div class='message-container user-message'><strong>You:</strong><br>{q}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='message-container bot-message'><strong>UNITA Assistant:</strong><br>{a}<br><br>"
            f"<em>⏱️ Time: {t}s | 🔢 Tokens: {tok}</em></div>",
            unsafe_allow_html=True
        )

# Botón de limpiar
if st.session_state.history:
    if st.button("Clear Conversation"):
        logging.info("Conversation cleared by user.")
        st.session_state.history = []
