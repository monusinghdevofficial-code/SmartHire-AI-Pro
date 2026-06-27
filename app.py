import streamlit as st
import numpy as np
import pickle
import re
from PyPDF2 import PdfReader
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SmartHire AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>

* {
    margin: 0;
    padding: 0;
}

html, body {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%) !important;
}

[data-testid="stSidebarNav"], [data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%) !important;
}

.main {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%) !important;
    color: #ffffff;
    padding-top: 20px;
}

.big-title {
    font-size: 72px;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 50%, #64b5f6 100%);
    padding: 40px 50px;
    border-radius: 20px;
    display: block;
    box-shadow: 0 25px 50px rgba(30, 136, 229, 0.3);
    margin: 20px auto;
    width: 90%;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 2px solid rgba(255, 255, 255, 0.1);
}

.sub-title {
    text-align: center;
    color: #b3e5fc;
    font-size: 24px;
    margin-top: 16px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: capitalize;
}

h3 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #42a5f5 !important;
    text-align: center !important;
    margin-top: 24px !important;
    margin-bottom: 16px !important;
    padding: 16px !important;
    background: rgba(66, 165, 245, 0.1) !important;
    border-radius: 12px !important;
    border-left: 4px solid #42a5f5 !important;
}

.metric-card {
    background: linear-gradient(135deg, rgba(30, 136, 229, 0.1), rgba(66, 165, 245, 0.1));
    border: 2px solid rgba(66, 165, 245, 0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 12px 32px rgba(30, 136, 229, 0.2);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(30, 136, 229, 0.3);
}

.metric-label {
    color: #b3e5fc;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.metric-value {
    color: #42a5f5;
    font-size: 36px;
    font-weight: 800;
    margin: 12px 0;
}

.skill-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    color: #ffffff;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
}

.recommended-skill {
    display: inline-block;
    background: linear-gradient(135deg, #f57c00, #fb8c00);
    color: #ffffff;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(245, 124, 0, 0.3);
}

.strength-bar {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
}

.strength-fill {
    height: 100%;
    background: linear-gradient(90deg, #42a5f5, #64b5f6);
    border-radius: 12px;
}

.footer-text {
    text-align: center;
    color: #90caf9;
    font-size: 14px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(66, 165, 245, 0.3);
}

label {
    color: #e0e0e0 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

p {
    color: #d0d0d0 !important;
    font-size: 15px !important;
}

.stFileUploader {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px dashed #42a5f5 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.stButton button {
    background: linear-gradient(135deg, #1e88e5, #42a5f5) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    box-shadow: 0 8px 16px rgba(30, 136, 229, 0.4) !important;
}

.sidebar-section {
    background: rgba(66, 165, 245, 0.1);
    border-left: 4px solid #42a5f5;
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
}

.sidebar-title {
    color: #42a5f5;
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.sidebar-text {
    color: #d0d0d0;
    font-size: 13px;
    line-height: 1.6;
}

.missing-keyword {
    display: inline-block;
    background: linear-gradient(135deg, #c62828, #e53935);
    color: #ffffff;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(230, 57, 57, 0.3);
}

.match-score-excellent {
    display: inline-block;
    background: linear-gradient(135deg, #2e7d32, #4caf50);
    color: #ffffff;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}

.match-score-good {
    display: inline-block;
    background: linear-gradient(135deg, #f57c00, #fb8c00);
    color: #ffffff;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(245, 124, 0, 0.3);
}

.match-score-poor {
    display: inline-block;
    background: linear-gradient(135deg, #c62828, #e53935);
    color: #ffffff;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(230, 57, 57, 0.3);
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# RESOURCE LOADING (Cached)
# =============================================================================
@st.cache_resource
def load_resources():
    """Load ML model and preprocessors from disk."""
    model = load_model("resume_cnn_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder


# Load resources
model, tokenizer, label_encoder = load_resources()
MAX_LEN = 500

# =============================================================================
# SKILL DATABASE & RECOMMENDATIONS
# =============================================================================
SKILLS_DB = [
    "python", "java", "c++", "c", "sql", "mysql",
    "machine learning", "deep learning", "tensorflow",
    "keras", "pandas", "numpy", "power bi",
    "excel", "data analysis", "nlp",
    "streamlit", "flask", "django",
    "html", "css", "javascript", "react",
    "aws", "azure", "git", "github"
]

# Skill recommendations mapping - suggests complementary skills
SKILL_RECOMMENDATIONS = {
    "python": ["sql", "django", "flask"],
    "machine learning": ["power bi", "sql", "aws"],
    "tensorflow": ["aws", "azure", "python"],
    "deep learning": ["aws", "tensorflow", "python"],
    "data analysis": ["power bi", "sql", "python"],
    "nlp": ["tensorflow", "keras", "python"],
    "react": ["javascript", "html", "css"],
    "java": ["sql", "aws", "git"],
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def extract_text_from_pdf(pdf_file):
    """
    Extract text content from uploaded PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object
    
    Returns:
        str: Extracted text from all pages
    """
    text = ""
    pdf_reader = PdfReader(pdf_file)
    
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    
    return text


def extract_skills_from_text(text):
    """
    Extract detected skills from resume text by matching against skill database.
    
    Args:
        text: Resume text content
    
    Returns:
        list: List of detected skills
    """
    found_skills = []
    text_lower = text.lower()
    
    for skill in SKILLS_DB:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))


def calculate_resume_score(skills):
    """
    Calculate ATS resume score based on number of detected skills.
    
    Args:
        skills: List of detected skills
    
    Returns:
        int: Resume score (0-100)
    """
    score = min(len(skills) * 10, 100)
    return score


def get_resume_strength(score):
    """
    Classify resume strength level based on score.
    
    Args:
        score: Resume score (0-100)
    
    Returns:
        tuple: (strength_level, color_code)
    """
    if score >= 80:
        return "Advanced", "#4caf50"
    elif score >= 50:
        return "Intermediate", "#ff9800"
    else:
        return "Beginner", "#f44336"


def predict_resume_category(text):
    """
    Predict resume category using trained CNN model.
    
    Args:
        text: Resume text content
    
    Returns:
        tuple: (predicted_category, confidence_percentage)
    """
    seq = tokenizer.texts_to_sequences([text])
    
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    
    pred = model.predict(padded, verbose=0)
    category_index = np.argmax(pred)
    category = label_encoder.inverse_transform([category_index])[0]
    confidence = float(np.max(pred) * 100)
    
    return category, confidence


def get_recommended_skills(detected_skills):
    """
    Get recommended skills based on detected skills using knowledge rules.
    
    Args:
        detected_skills: List of detected skills
    
    Returns:
        list: List of recommended skills to add
    """
    recommended = set()
    
    for skill in detected_skills:
        skill_lower = skill.lower()
        if skill_lower in SKILL_RECOMMENDATIONS:
            recommended.update(SKILL_RECOMMENDATIONS[skill_lower])
    
    # Filter out already detected skills
    recommended = [s for s in recommended if s not in [sk.lower() for sk in detected_skills]]
    
    return list(set(recommended))[:5]  # Return top 5 recommendations


def calculate_job_match(resume_text, job_description_text):
    """
    Calculate similarity score between resume and job description using TF-IDF and cosine similarity.
    
    Args:
        resume_text: Resume content
        job_description_text: Job description content
    
    Returns:
        float: Similarity score (0-100)
    """
    if not resume_text or not job_description_text:
        return 0.0
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage (0-100)
    match_score = round(similarity * 100, 2)
    
    return match_score


def extract_keywords_from_text(text):
    """
    Extract important keywords from text (words with 4+ characters, excluding common words).
    
    Args:
        text: Text to extract keywords from
    
    Returns:
        list: List of extracted keywords
    """
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'that', 'have', 'this', 'will',
        'from', 'your', 'their', 'which', 'about', 'more', 'also', 'only',
        'other', 'some', 'can', 'been', 'being', 'have', 'has', 'had',
        'work', 'must', 'requirements'
    }
    
    # Convert to lowercase and extract words
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{4,}\b', text_lower)
    
    # Filter out stop words and get unique keywords
    keywords = [word for word in set(words) if word not in stop_words]
    
    return sorted(keywords)[:15]  # Return top 15 keywords


def find_missing_keywords(resume_text, jd_text):
    """
    Find keywords from job description that are missing in resume.
    
    Args:
        resume_text: Resume content
        jd_text: Job description content
    
    Returns:
        list: List of missing keywords
    """
    jd_keywords = set(extract_keywords_from_text(jd_text))
    resume_keywords = set(extract_keywords_from_text(resume_text))
    
    missing = list(jd_keywords - resume_keywords)
    
    return sorted(missing)[:10]  # Return top 10 missing keywords


# =============================================================================
# SIDEBAR INFORMATION
# =============================================================================
def render_sidebar():
    """Render professional sidebar with project information and metadata."""
    with st.sidebar:
        # Title
        st.markdown(
            '<div style="text-align: center; margin: 20px 0;"><h1 style="color: #42a5f5; font-size: 24px;">🚀 SmartHire AI Pro</h1></div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Project Information
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📌 Project Information</div><div class="sidebar-text">Deep Learning Resume Screening System</div></div>',
            unsafe_allow_html=True
        )
        
        # Tech Stack
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">🛠 Tech Stack</div><div class="sidebar-text">• Python<br>• TensorFlow<br>• Keras<br>• CNN<br>• Streamlit<br>• PyPDF2<br>• NumPy<br>• Pandas</div></div>',
            unsafe_allow_html=True
        )
        
        # Model Performance
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📊 Model Performance</div><div class="sidebar-text">• CNN Accuracy: 80.68%</div></div>',
            unsafe_allow_html=True
        )
        
        # Dataset Information
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📂 Dataset Information</div><div class="sidebar-text">• 2484 Resume Samples<br>• 24 Categories</div></div>',
            unsafe_allow_html=True
        )
        
        # AI Pipeline
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">🧠 AI Pipeline</div><div class="sidebar-text">1. Resume Upload<br>2. PDF Text Extraction<br>3. Text Cleaning<br>4. Tokenization<br>5. CNN Prediction<br>6. Skill Detection<br>7. ATS Analysis<br>8. Final Recommendation</div></div>',
            unsafe_allow_html=True
        )
        
        # Developer Information
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">👨‍💻 Developer</div><div class="sidebar-text">Monu Singh<br>B.Tech CSE (AI & ML)</div></div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point with resume analysis workflow."""
    
    # Render sidebar
    render_sidebar()
    
    # Main title
    st.markdown(
        '<div class="big-title">🚀 SmartHire AI Pro</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p class="sub-title">Deep Learning Resume Screening System</p>',
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )
    
    # Job Description input section
    st.markdown("### 🎯 Job Description Matching")
    job_description = st.text_area(
        "Paste Job Description Here",
        height=200,
        placeholder="Enter the job description to match against your resume..."
    )
    
    if uploaded_file is not None:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Resume Uploaded Successfully")
        
        # Show extracted text (expandable)
        with st.expander("📄 View Extracted Resume Text"):
            st.write(resume_text[:5000])
        
        # Predict category
        predicted_category, confidence_score = predict_resume_category(resume_text)
        
        # Low confidence warning
        if confidence_score < 40:
            st.warning(
                "⚠️ Low confidence prediction. Resume may not clearly match any trained category."
            )
        
        # Extract skills
        detected_skills = extract_skills_from_text(resume_text)
        resume_score = calculate_resume_score(detected_skills)
        strength_level, strength_color = get_resume_strength(resume_score)
        
        # =================================================================
        # DASHBOARD SECTION - Professional Metric Cards
        # =================================================================
        st.markdown("### 📊 Resume Analysis Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Resume Score</div>
                    <div class="metric-value">{resume_score}/100</div>
                    <div class="strength-bar">
                        <div class="strength-fill" style="width: {resume_score}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Category</div>
                    <div class="metric-value" style="font-size: 24px;">{predicted_category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{confidence_score:.2f}%</div>
                    <div class="strength-bar">
                        <div class="strength-fill" style="width: {confidence_score}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # =================================================================
        # ATS ANALYSIS SECTION
        # =================================================================
        st.markdown("### ✨ ATS Resume Analysis")
        
        if resume_score >= 80:
            st.success("✅ ATS Friendly Resume - Excellent Quality")
        elif resume_score >= 50:
            st.warning("⚠️ Average ATS Score - Needs Improvement")
        else:
            st.error("❌ ATS Score Needs Significant Improvement")
        
        # =================================================================
        # RESUME STRENGTH INDICATOR
        # =================================================================
        st.markdown("### 💪 Resume Strength Indicator")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            strength_color_code = '#4caf50' if resume_score >= 80 else '#ff9800' if resume_score >= 50 else '#f44336'
            st.markdown(
                f"""
                <div class="strength-bar">
                    <div class="strength-fill" style="width: {resume_score}%; background: {strength_color_code};"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f"**{strength_level}**", unsafe_allow_html=True)
        
        # =================================================================
        # DETECTED SKILLS SECTION - Display as Badges
        # =================================================================
        st.markdown("### 🎯 Detected Skills")
        
        if len(detected_skills) > 0:
            skills_html = ""
            for skill in sorted(detected_skills):
                skills_html += f'<span class="skill-badge">{skill.title()}</span>'
            
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No predefined skills detected. Consider adding relevant skills to your resume.")
        
        # =================================================================
        # RECOMMENDED SKILLS SECTION
        # =================================================================
        st.markdown("### 🚀 Recommended Skills to Add")
        
        recommended_skills = get_recommended_skills(detected_skills)
        
        if len(recommended_skills) > 0:
            recommendations_html = ""
            for skill in recommended_skills:
                recommendations_html += f'<span class="recommended-skill">{skill.title()}</span>'
            
            st.markdown(recommendations_html, unsafe_allow_html=True)
            st.info("💡 Adding these skills could significantly improve your resume's ATS score and market value.")
        else:
            st.success("✅ Great job! Your resume covers most recommended skills.")
        
        # =================================================================
        # AI RECOMMENDATION SECTION
        # =================================================================
        st.markdown("### 🤖 AI Recommendation")
        
        if resume_score >= 80:
            st.success(
                "🌟 Excellent Resume! Strong candidate profile with diverse technical skills. "
                "Your resume demonstrates advanced expertise and is highly likely to pass ATS screening."
            )
        elif resume_score >= 50:
            st.info(
                "📚 Good Resume! Your profile shows solid technical foundation. "
                "Consider adding more relevant skills and certifications to strengthen your candidacy."
            )
        else:
            st.error(
                "⚙️ Resume Needs Improvement! Focus on adding more technical skills and relevant experience. "
                "Review the recommended skills section above to enhance your resume."
            )
        
        # =================================================================
        # PREDICTION CONFIDENCE PROGRESS BAR
        # =================================================================
        st.markdown("### 📈 Prediction Confidence Level")
        st.progress(min(int(confidence_score), 100) / 100)
        st.caption(f"Model Confidence: {confidence_score:.2f}%")
        
        # =================================================================
        # JOB DESCRIPTION MATCHING SECTION
        # =================================================================
        if job_description and len(job_description.strip()) > 10:
            st.markdown("### 🎯 Job Match Analysis")
            
            # Calculate match score
            match_score = calculate_job_match(resume_text, job_description)
            
            # Display match score with color coding
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Job Match Score</div>
                        <div class="metric-value">{match_score:.1f}%</div>
                        <div class="strength-bar">
                            <div class="strength-fill" style="width: {match_score}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                if match_score >= 80:
                    match_status = "Excellent Match"
                    st.markdown(
                        '<div class="match-score-excellent">✅ Excellent Match</div>',
                        unsafe_allow_html=True
                    )
                elif match_score >= 60:
                    match_status = "Good Match"
                    st.markdown(
                        '<div class="match-score-good">✓ Good Match</div>',
                        unsafe_allow_html=True
                    )
                else:
                    match_status = "Poor Match"
                    st.markdown(
                        '<div class="match-score-poor">✗ Poor Match</div>',
                        unsafe_allow_html=True
                    )
            
            with col3:
                st.metric("Match Status", f"{match_score:.1f}/100")
            
            # Display intelligent messages based on score
            st.markdown("---")
            if match_score >= 80:
                st.success(
                    "🌟 Excellent Match for this role! Your resume aligns very well with the job description. "
                    "You have most of the required skills and keywords."
                )
            elif match_score >= 60:
                st.warning(
                    "📊 Good Match! Your resume matches the job description well. "
                    "Consider adding the missing keywords to improve your chances."
                )
            else:
                st.error(
                    "⚠️ Low Match. Your resume needs more relevant keywords and skills matching the job description. "
                    "Review the missing keywords section below to enhance your resume."
                )
            
            # =================================================================
            # MISSING KEYWORDS SECTION
            # =================================================================
            st.markdown("### 📌 Missing Keywords")
            
            missing_keywords = find_missing_keywords(resume_text, job_description)
            
            if missing_keywords:
                keywords_html = ""
                for keyword in missing_keywords:
                    keywords_html += f'<span class="missing-keyword">{keyword.upper()}</span>'
                
                st.markdown(keywords_html, unsafe_allow_html=True)
                st.info(
                    "💡 Tip: Adding these keywords to your resume can significantly improve your job match score "
                    "and increase the chances of getting selected by ATS systems."
                )
            else:
                st.success("✅ Great! Your resume contains most of the keywords from the job description.")
    
    # =================================================================
    # FOOTER SECTION - Professional Footer
    # =================================================================
    st.markdown("---")
    st.markdown(
        '<div class="footer-text">Built with ❤️ using TensorFlow, CNN, NLP and Streamlit</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="footer-text">© 2024 SmartHire AI - Intelligent Resume Screening System</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
