import streamlit as st
from google import genai

from dotenv import load_dotenv
import os
import time

from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from streamlit_mic_recorder import speech_to_text
from gtts import gTTS


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_db" not in st.session_state:
    st.session_state.pdf_db = None


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY is missing.")
    st.info("Add GEMINI_API_KEY to your Streamlit secrets or .env file.")
    st.stop()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"


# =========================================================
# LANGUAGE SETTINGS
# =========================================================

LANGUAGES = {
    "English": {
        "code": "en",
        "voice": "en"
    },
    "Hindi": {
        "code": "hi",
        "voice": "hi"
    },
    "Tamil": {
        "code": "ta",
        "voice": "ta"
    },
    "Telugu": {
        "code": "te",
        "voice": "te"
    }
}


# =========================================================
# HOME PAGE
# =========================================================

if not st.session_state.start_chat:

    st.image(
        "banner.png.png",
        width="stretch"
    )

    st.title("⚖️ LawGPT")
    st.subheader("Your AI Legal Assistant")

    st.write(
        """
        Ask questions related to Indian Penal Code (IPC).

        📄 Upload legal PDFs  
        🎤 Voice input support  
        🌐 Supports 4 languages
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("⚖️\n\nIPC Laws")

    with col2:
        st.info("📄\n\nUpload PDF")

    with col3:
        st.info("🎤\n\nVoice Input")

    with col4:
        st.info("🌐\n\n4 Languages")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🚀 Start Chat",
        use_container_width=True
    ):
        st.session_state.start_chat = True
        st.rerun()

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚖️ LawGPT")

    language = st.selectbox(
        "Select response language",
        list(LANGUAGES.keys())
    )

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

    st.markdown("---")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.start_chat = False
        st.rerun()

    if st.button("🗑 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# LOAD FAISS DATABASE
# =========================================================

@st.cache_resource
def load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


try:
    db = load_db()
except Exception as e:
    st.error("❌ Could not load the FAISS vector database.")
    st.exception(e)
    st.stop()


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload a legal PDF",
    type=["pdf"]
)


if uploaded_file:

    try:

        reader = PdfReader(uploaded_file)

        pdf_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pdf_text += text + "\n"

        if not pdf_text.strip():

            st.error(
                "❌ Could not extract text from this PDF."
            )

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.create_documents(
                [pdf_text]
            )

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            pdf_db = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.pdf_db = pdf_db

            st.success(
                f"✅ PDF Loaded Successfully! "
                f"{len(chunks)} chunks created."
            )

    except Exception as e:

        st.error("❌ Error processing PDF.")
        st.exception(e)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# SAMPLE QUESTIONS
# =========================================================

st.markdown("### 💡 Try Sample Questions")

sample_questions = [
    "What is the punishment for theft under IPC?",
    "Explain section 420 of IPC",
    "What are the rights of an accused person under IPC?"
]

selected = st.selectbox(
    "Choose a sample question",
    [""] + sample_questions
)


# =========================================================
# VOICE INPUT
# =========================================================

voice_question = speech_to_text(
    language=LANGUAGES[language]["code"],
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop Recording",
    use_container_width=True
)


# =========================================================
# USER QUESTION
# =========================================================

question = None

if selected:
    question = selected
else:

    question = st.chat_input(
        "Ask any question related to Indian Penal Code (IPC)",
        key="law_chat"
    )


if voice_question:
    question = voice_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    question = question.strip()

    if not question:

        st.warning("⚠️ Please enter a valid question.")
        st.stop()


    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    with st.chat_message("user"):
        st.markdown(question)


    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # -------------------------------------------------
            # SEARCH DOCUMENTS
            # -------------------------------------------------

            if st.session_state.pdf_db is not None:

                docs = st.session_state.pdf_db.similarity_search(
                    question,
                    k=4
                )

            else:

                docs = db.similarity_search(
                    question,
                    k=4
                )


            # -------------------------------------------------
            # CREATE CONTEXT
            # -------------------------------------------------

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )


            # -------------------------------------------------
            # CHAT HISTORY
            # -------------------------------------------------

            history = "\n".join(
                f'{m["role"]}: {m["content"]}'
                for m in st.session_state.messages[-6:]
            )


            # -------------------------------------------------
            # PROMPT
            # -------------------------------------------------

            prompt = f"""
You are LawGPT, an AI legal information assistant.

Your task is to answer the user's question ONLY using
the legal context provided below.

Response language:
{language}

IMPORTANT RULES:

1. Use ONLY the provided legal context.
2. Do not invent legal sections, punishments or facts.
3. If the answer cannot be found in the context, say:

"I couldn't find this information in the provided legal document."

4. Give a clear and concise answer.
5. Mention relevant IPC sections when they are present
   in the provided context.
6. This is an educational legal assistant and not a substitute
   for professional legal advice.

Conversation History:
{history}

Legal Context:
{context}

User Question:
{question}

Answer:
"""


            # -------------------------------------------------
            # GEMINI
            # -------------------------------------------------

            with st.spinner(
                "🔍 Searching legal documents..."
            ):

                start = time.time()

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )

                end = time.time()


            # -------------------------------------------------
            # RESPONSE
            # -------------------------------------------------

            full = response.text

            st.success(
                f"✅ Answer generated in {end - start:.2f} seconds"
            )


            # -------------------------------------------------
            # STREAMING EFFECT
            # -------------------------------------------------

            placeholder = st.empty()

            displayed = ""

            for ch in full:

                displayed += ch

                placeholder.markdown(
                    displayed + "▌"
                )

                time.sleep(0.005)

            placeholder.markdown(full)


            # -------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # -------------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": full
            })


            # -------------------------------------------------
            # TEXT TO SPEECH
            # -------------------------------------------------

            try:

                voice_code = LANGUAGES[language]["voice"]

                tts = gTTS(
                    text=full,
                    lang=voice_code
                )

                audio_file = "answer.mp3"

                tts.save(audio_file)

                with open(audio_file, "rb") as audio:

                    st.audio(
                        audio.read(),
                        format="audio/mp3"
                    )

            except Exception as tts_error:

                st.warning(
                    f"⚠️ Voice output unavailable: {tts_error}"
                )


            # -------------------------------------------------
            # DOWNLOAD ANSWER
            # -------------------------------------------------

            st.download_button(
                label="💾 Download Answer",
                data=full,
                file_name="lawGPT_answer.txt",
                mime="text/plain"
            )


            # -------------------------------------------------
            # SOURCES
            # -------------------------------------------------

            st.markdown("---")

            st.subheader("📚 Sources Used")

            for i, doc in enumerate(
                docs,
                start=1
            ):

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                source = doc.metadata.get(
                    "source",
                    "IPC"
                )

                with st.expander(
                    f"📄 Source {i}"
                ):

                    st.write(
                        f"**Source:** {source}"
                    )

                    st.write(
                        f"**Page:** {page}"
                    )

                    st.write(
                        doc.page_content[:500] + "..."
                    )


        except Exception as e:

            st.error(
                "❌ Something went wrong while generating the answer."
            )

            st.exception(e)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.caption(
    "⚠️ This chatbot is for educational purposes only. "
    "It is not a substitute for professional legal advice."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>

    ⚖️ <b>LawGPT</b><br>

    AI Powered Legal Assistant using
    <b>RAG • FAISS • LangChain • Gemini 2.5 Flash</b>

    <br><br>

    <sub>
    This project is for educational purposes only and
    should not be considered legal advice.
    </sub>

    </center>
    """,
    unsafe_allow_html=True
    ) 
