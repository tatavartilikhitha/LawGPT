import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
st.set_page_config(
    page_title="LawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------
# Load Environment
# -----------------------

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------
# Streamlit
# -----------------------

# Paste here 👇

if not st.session_state.start_chat:

    st.image("banner.png.png", use_container_width=True)

    st.title("⚖️ LawGPT")
    st.subheader("Your AI Legal Assistant")
    st.write("""
             Ask questions related to Indian Penal Code (IPC).
             
             📄Upload legal PDFs

            🎤Voice input support
             
             🌐Supports 4 languages
             """)
             

    col1,col2,col3,col4=st.columns(4)

    with col1:
        st.info("⚖️\n\nIPC Laws")

    with col2:
        st.info("📄\n\nUpload PDF")

    with col3:
        st.info("🎤\n\nVoice Input")

    with col4:
        st.info("🌐\n\n4 Languages")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start Chat",use_container_width=True):
        st.session_state.start_chat=True
        st.rerun()

    st.stop()
    if st.sidebar.button("Home"):
        st.session_state.start_chat=False
        st.rerun()
with st.sidebar:

    language = st.selectbox(
        "Select response language",
        ["English", "Hindi", "Tamil", "Telugu"]
    )

    st.header("⚖️ LawGPT")

    st.success("✅ IPC Document Loaded")

    st.write("🤖 AI Model")
    st.info("Google Gemini 2.5 Flash")

    st.write("🧠 Vector Database")
    st.info("FAISS")

    st.write("📚 Framework")
    st.info("LangChain")

    st.write("💻 Frontend")
    st.info("Streamlit")

    st.write("📄 Dataset")
    st.info("Indian Penal Code (IPC)")

    col1, col2, col3 = st.columns(3)

    uploaded_file=st.file_uploader("Upload a legal PDF",type=["pdf"])


# -----------------------
# Chat Memory
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"]) 

if st.button("🗑 Reset Chat"):
    st.session_state.messages=[]
    st.rerun()

# -----------------------
# Load FAISS
# -----------------------

@st.cache_resource

def load_db():

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db=FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db=load_db()

if uploaded_file:

    reader = PdfReader(uploaded_file)

    pdf_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.create_documents([pdf_text])
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    pdf_db=FAISS.from_documents(chunks, embeddings)
    st.session_state["pdf_db"] = pdf_db

    st.success(f"✅ PDF Loaded Successfully!({len(chunks)})Chunks Created")
# -----------------------
# Show Old Chats
# -----------------------

# -----------------------
# User Question
# -----------------------
voice_question=speech_to_text(
    language="en",
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑Stop Recording",
    use_container_width=True
)
st.markdown("###Try Sample Questions")
sample_questions = [
    "what is the punishment for theft under IPC?",
    "Explain section 420 of IPC",
    "What are the rights of an accused person under IPC?"
]
selected=st.selectbox("Choose a sample question",[""]+sample_questions)
question=selected if selected else st.chat_input("Ask any question related to Indian Penal Code (IPC)",key="law_chat")

if voice_question:
    question=voice_question

st.write("Voice Output:",question)

if question:
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if "pdf_db" in st.session_state:
            docs = st.session_state["pdf_db"].similarity_search(question, k=4)
        else:
            docs = db.similarity_search(question, k=4)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )
        history="\n".join(
            [
                f'{m["role"]}: {m["content"]}'
                for m in st.session_state.messages[-6:]
            ]
        )
        prompt = f"""
You are LawGPT.

Answer ONLY using the legal context below.

Respond in **{language}**.

If the answer is not found in the context, reply:

"I couldn't find this information in the provided legal document."

Conversation History:
{history}

Context:

{context}

Question:

{question}

Answer:
"""
    with st.spinner("🔍Searching legal documents..."):
        print("Context Lenght:", len(context))
        print("Prompt Lenght:", len(prompt))
        print("Question:", question)
        start=time.time()
        
        response = model.generate_content(prompt)
        end=time.time()
        st.success(f"✅ Answer generated in{end-start:.2f} seconds")

        full = ""

        placeholder = st.empty()

        for ch in response.text:
            full += ch
            time.sleep(0.01)
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full
        })
        tts=gTTS(full, lang="en")
        tts.save("answer.mp3")
        with open("answer.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        st.code(full,language=None)
        st.download_button(
            label="💾 Download Answer",
            data=full,
            file_name="lawGPT_answer.txt",
            mime="text/plain"
        )
        st.markdown("---")
        st.subheader("📚 Sources Used")

        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "IPC")

            with st.expander(f"📄 Source {i}"):
                st.write(f"**Source:** {source} ")
                st.write(f"**Page:** {page}")
                st.write(doc.page_content[:500]+"...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text
    })
    st.markdown("---")
    st.caption(
        "⚠️This Chatbot is educational purpose only.It is not a substitute for legal advice. "
    )
st.markdown("---")

st.markdown(
    """
    <center>

    ⚖️ <b>LawGPT</b><br>

    AI Powered Legal Assistant using
    <b>RAG • FAISS • LangChain • Gemini 2.5 Flash</b>

    <br><br>

    <sub>
    This project is for educational purposes only and should not be considered legal advice.
    </sub>

    </center>
    """,
    unsafe_allow_html=True
)