import streamlit as st
import pandas as pd
import logging
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""

def rank_resumes(job_description, resumes):
    """Ranks resumes based on similarity to job description."""
    try:
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()
        
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        
        return cosine_similarity([job_vector], resume_vectors).flatten()
    except Exception as e:
        logging.error(f"Error in ranking resumes: {e}")
        return []

# Streamlit UI
st.set_page_config(page_title="AI Resume Scanner", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f4f4;
        }
        .title-text {
            font-size: 36px;
            font-weight: bold;
            color: #2E3B55;
        }
        .header-text {
            font-size: 24px;
            color: #444;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">📄 AI Resume Scanning & Candidate Ranking System</p>', unsafe_allow_html=True)

# Job description input
st.markdown('<p class="header-text">🔍 Job Description</p>', unsafe_allow_html=True)
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.markdown('<p class="header-text">📂 Upload Resumes</p>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.markdown('<p class="header-text">📊 Ranking Resumes</p>', unsafe_allow_html=True)
    
    with st.spinner("Processing resumes..."):
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)
        scores_percentage = [round(score * 100, 2) for score in scores]
        
        results = pd.DataFrame({
            "Candidate Name": [file.name for file in uploaded_files],
            "Match Score (%)": scores_percentage
        }).sort_values(by="Match Score (%)", ascending=False)
    
    # Display results
    st.success("✅ Ranking Complete!")
    st.dataframe(results.style.set_properties(**{'background-color': '#ffffff', 'border': '1px solid #ddd', 'padding': '10px'}), use_container_width=True)
    
    # Show top candidate
    if not results.empty:
        top_candidate = results.iloc[0]
        st.markdown(f"### 🏆 Top Candidate: **{top_candidate['Candidate Name']}** - {top_candidate['Match Score (%)']}% Match")
        st.progress(top_candidate['Match Score (%)'] / 100)
else:
    st.info("Please provide a job description and upload resumes to proceed.")
