import streamlit as st
import google.generativeai as genai
import PyPDF2
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("AI Resume Analyzer")
st.write("Upload your resume and get AI feedback using gemini")
api_key = st.text_input("Enter your Google API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_role = st.text_input("Target Job Role")

if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume.")
        st.stop()

    if api_key == "":
        st.warning("Please enter Gemini API Key.")
        st.stop()

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    prompt = f"""
You are an expert HR recruiter.

Analyze this resume.

Target Job:
{job_role}

Resume:
{resume_text}

Give:

1. Resume Score out of 100
2. Skills Found
3. Missing Skills
4. Strengths
5. Weaknesses
6. Suggestions to improve
7. Final Verdict
"""
    model=genai.GenerativeModel("gemini-2.5-flash")
    responses=model.generate_content(prompt)
    st.success("Analysis Complete!")
    st.markdown(responses.text)