import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

st.set_page_config(page_title="🤖 Meu Agente de IA com Memória", page_icon="🤖", layout="centered")

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Chave de API do Google não encontrada! Por favor, configure os Secrets.")
    st.stop()
else:
    genai.configure(api_key=api_key)

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

user_avatar = "👤"
ai_avatar = "🤖"

with st.sidebar:
    st.title("Opções")
    if st.button("Limpar Histórico da Conversa"):
        st.session_state.chat = model.start_chat(history=[])
        st.rerun()

st.title("Meu Agente de IA com Memória")
st.caption("Desenvolvido com Gemini, Python e Streamlit")

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if not st.session_state.chat.history:
    with st.chat_message("assistant", avatar=ai_avatar):
        st.write("Olá! Sou seu agente de IA. Como posso te ajudar hoje?")

for message in st.session_state.chat.history:
    role = "user" if message.role == "user" else "assistant"
    with st.chat_message(role, avatar=(user_avatar if role == "user" else ai_avatar)):
        st.markdown(message.parts[0].text)

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant", avatar=ai_avatar):
            response_container = st.empty()
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            response_container.write_stream(response_stream)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")