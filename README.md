# 🤖 Agente de Service Desk com Gemini e LangGraph

**Um agente de IA especialista em políticas internas, capaz de responder perguntas, solicitar mais informações ou abrir chamados automaticamente.**

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-orange?style=for-the-badge&logo=streamlit" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/LangChain-0.2%2B-green?style=for-the-badge&logo=langchain" alt="LangChain Version">
</p>

---

## 🎞️ Demonstração em Ação

![Demonstração do Chatbot](caminho/para/seu/demo.gif)

---

## 📝 Sobre o Projeto

Este projeto foi desenvolvido como parte da Imersão Dev com o objetivo de criar um agente de IA robusto, indo do protótipo à produção. O resultado é um assistente de Service Desk que utiliza um fluxo de raciocínio (construído com LangGraph) para lidar com as solicitações dos usuários sobre as políticas internas da empresa.

O agente primeiro realiza uma triagem para classificar a intenção do usuário. Em seguida, ele pode:
1.  **Auto-resolver:** Usar RAG (Retrieval-Augmented Generation) para buscar informações em documentos PDF e responder à pergunta.
2.  **Pedir mais informações:** Caso a pergunta seja muito vaga.
3.  **Abrir um chamado:** Para solicitações que exigem ação humana ou aprovação.

---

## ✨ Principais Funcionalidades

-   **Triagem Inteligente:** Classifica as perguntas em `AUTO_RESOLVER`, `PEDIR_INFO` ou `ABRIR_CHAMADO`.
-   **RAG (Busca Aumentada por Geração):** Responde a perguntas com base em uma base de conhecimento de documentos PDF.
-   **Agente com LangGraph:** Util
