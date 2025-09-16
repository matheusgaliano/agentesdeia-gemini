import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

st.set_page_config(page_title="🤖 Meu Chatbot com Gemini", page_icon="🤖", layout="centered")

load_dotenv()

try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except AttributeError:
    st.error("Chave de API do Google não encontrada! Por favor, configure os Secrets.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao configurar a API: {e}")
    st.stop()

def load_model():
    generation_config = {
        "candidate_count": 1,
        "temperature": 0.7,
    }
    safety_settings = {
        "HARASSMENT": "BLOCK_NONE",
        "HATE": "BLOCK_NONE",
        "SEXUAL": "BLOCK_NONE",
        "DANGEROUS": "BLOCK_NONE",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

model = load_model()

st.title("Meu Agente de IA com Memória")
st.caption("Desenvolvido com Gemini, Python e Streamlit")

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

for message in st.session_state.chat.history:
    role = "Você" if message.role == "user" else "IA"
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("Você"):
        st.markdown(prompt)

    try:
        response = st.session_state.chat.send_message(prompt)
        with st.chat_message("IA"):
            st.markdown(response.text)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")