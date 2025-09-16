import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Literal

# Imports do LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Imports do LangGraph
from langgraph.graph import StateGraph, START, END

# --- CONFIGURAÇÃO INICIAL E CHAVE DE API ---
st.set_page_config(page_title="🤖 Agente de Service Desk", page_icon="🤖", layout="wide")
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Chave de API do Google não encontrada! Por favor, configure os Secrets.")
    st.stop()

# --- LÓGICA DO AGENTE (AULAS 01, 02, 03) ---

# Usamos cache_resource para inicializar tudo apenas uma vez
@st.cache_resource
def setup_agent():
    # AULA 01: LÓGICA DE TRIAGEM
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=api_key)

    TRIAGEM_PROMPT = (
        "Você é um triador de Service Desk para políticas internas da empresa Pretor Concursos. "
        "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
        "{\n"
        '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
        '  "urgencia": "BAIXA" | "MEDIA" | "ALTA"\n'
        "}\n"
        "Regras:\n"
        '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar uma compra?", "Como funciona a política de home office?").\n'
        '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema (Ex: "Preciso de ajuda com uma política").\n'
        '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial (Ex: "Quero um cupom de desconto diferenciado.", "Solicito liberação para acesso.").'
    )
    class TriagemOut(BaseModel):
        decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
        urgencia: Literal["BAIXA", "MEDIA", "ALTA"]

    triagem_chain = llm_gemini.with_structured_output(TriagemOut)

    def triagem(mensagem: str) -> Dict:
        saida: TriagemOut = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT),
            HumanMessage(content=mensagem)
        ])
        return saida.model_dump()

    # AULA 02: LÓGICA DE RAG
    docs = []
    for f in Path("./policies/").glob("*.pdf"):
        loader = PyMuPDFLoader(str(f))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_rag = ChatPromptTemplate.from_messages([
        ("system", "Você é um Assistente de Políticas Internas da empresa Pretor Concursos. Responda SOMENTE com base no contexto fornecido. Se não houver base suficiente, responda apenas 'Não encontrei informações sobre isso nas políticas da empresa.'."),
        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])
    document_chain = create_stuff_documents_chain(llm_gemini, prompt_rag)

    def perguntar_politica_RAG(pergunta: str) -> Dict:
        docs_relacionados = retriever.invoke(pergunta)
        if not docs_relacionados:
            return {"answer": "Não encontrei informações sobre isso nas políticas da empresa.", "rag_sucesso": False}
        
        answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
        if "não sei" in answer.lower() or "não encontrei" in answer.lower():
             return {"answer": "Não encontrei informações sobre isso nas políticas da empresa.", "rag_sucesso": False}
        
        return {"answer": answer, "rag_sucesso": True}

    # AULA 03: LÓGICA DO LANGGRAPH
    class AgentState(TypedDict, total=False):
        pergunta: str
        triagem: Dict
        resposta: Optional[str]
        acao_final: str

    def node_triagem(state: AgentState) -> AgentState:
        return {"triagem": triagem(state["pergunta"])}

    def node_auto_resolver(state: AgentState) -> AgentState:
        resultado_rag = perguntar_politica_RAG(state["pergunta"])
        return {"resposta": resultado_rag["answer"], "rag_sucesso": resultado_rag["rag_sucesso"]}

    def node_pedir_info(state: AgentState) -> AgentState:
        return {"resposta": "Não entendi sua pergunta. Por favor, forneça mais detalhes sobre a política que você tem dúvida.", "acao_final": "PEDIR_INFO"}

    def node_abrir_chamado(state: AgentState) -> AgentState:
        urgencia = state["triagem"]["urgencia"]
        return {"resposta": f"Entendido. Para resolver sua solicitação, abri um chamado para a equipe responsável com urgência {urgencia}.", "acao_final": "ABRIR_CHAMADO"}
    
    def decidir_pos_triagem(state: AgentState) -> str:
        return state["triagem"]["decisao"]

    def decidir_pos_auto_resolver(state: AgentState) -> str:
        return "ok" if state.get("rag_sucesso") else "pedir_info"

    workflow = StateGraph(AgentState)
    workflow.add_node("triagem", node_triagem)
    workflow.add_node("auto_resolver", node_auto_resolver)
    workflow.add_node("pedir_info", node_pedir_info)
    workflow.add_node("abrir_chamado", node_abrir_chamado)
    
    workflow.add_edge(START, "triagem")
    workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
        "AUTO_RESOLVER": "auto_resolver",
        "PEDIR_INFO": "pedir_info",
        "ABRIR_CHAMADO": "abrir_chamado"
    })
    workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
        "ok": END,
        "pedir_info": "pedir_info"
    })
    workflow.add_edge("pedir_info", END)
    workflow.add_edge("abrir_chamado", END)
    
    return workflow.compile()

# Inicializa o agente
try:
    agent_graph = setup_agent()
except Exception as e:
    st.error(f"Erro ao inicializar o agente. Verifique seus PDFs e configurações. Erro: {e}")
    st.stop()

# --- INTERFACE DO STREAMLIT ---

st.title("🤖 Agente de Service Desk - Políticas Internas")
st.caption("Este agente inteligente usa RAG e LangGraph para te ajudar com as políticas da empresa.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
     with st.chat_message("assistant"):
        st.write("Olá! Sou o assistente de políticas internas da Pretor Concursos. Como posso ajudar?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Qual sua dúvida sobre as políticas?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Analisando sua pergunta..."):
                final_state = agent_graph.invoke({"pergunta": prompt})
                response = final_state["resposta"]
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar sua solicitação: {e}")