import streamlit as st
import sqlite3
import pandas as pd
import json
import uuid
import re
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# NLP/AI Libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Download NLTK resources if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURATION & INITIALIZATION ---

APP_TITLE = "IntelliHire: AI Candidate Screener"
DB_FILE = "resumes_db.sqlite"
STOPWORDS = set(stopwords.words('english'))
SKILL_MAPPER = 'skill_mapper.json' # Placeholder for skill matching library
nlp = None # To be loaded by load_spacy_model

# Load spaCy model once
@st.cache_resource
def load_spacy_model():
    """Load the spacy model efficiently."""
    try:
        # Load the small English model
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        return None

# Load the model
nlp = load_spacy_model()

# --- DATABASE FUNCTIONS ---

def init_db():
    """Initializes the SQLite database and creates the resumes table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            total_experience REAL,
            skills TEXT,
            education TEXT,
            experience_raw TEXT,
            score REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_resume_data(data):
    """Saves a single resume record to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO resumes (
            id, name, email, phone, total_experience, 
            skills, education, experience_raw, score, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['id'], data['name'], data['email'], data['phone'], data['total_experience'],
        data['skills'], data['education'], data['experience_raw'], data['score'], data['timestamp']
    ))
    conn.commit()
    conn.close()

def get_all_resumes():
    """Retrieves all resume data from the database."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM resumes", conn)
    conn.close()
    return df

def delete_resume_by_id(resume_id):
    """Deletes a single resume record by its ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
    conn.commit()
    conn.close()

# --- UTILITY & NLP FUNCTIONS ---

def clean_text(text):
    """Cleans text by converting to lowercase, removing non-alphanumeric, and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def extract_data_from_pdf(file):
    """Placeholder for PyResParser or similar tool. Returns structured data."""
    # Since PyResParser cannot be used directly, we simulate the structure
    # This simulation is based on a successful run of pyresparser on a sample PDF.
    time.sleep(1.5) # Simulate processing time

    # Mocked data for demonstration
    mock_data = {
        "Name, Skills, Experience": {
            "name": "Mohamed Faroz S",
            "email": "smohamedfaroz@example.com",
            "phone": "9998887770",
            "total_experience": 4.5,
            "skills": ["Python", "SQL", "Pandas", "Scikit-learn", "Streamlit", "Data Science", "Machine Learning", "NLP"],
            "education": ["Master of Science in Data Analytics", "Bachelor of Engineering"],
            "experience_raw": (
                "Lead Data Scientist at TechCorp (3 years): Developed and deployed ML models "
                "using Python and Scikit-learn. Expertise in NLP for text classification. "
                "Data Analyst at FinTech Solutions (1.5 years): Managed large SQL databases "
                "and created data visualization dashboards."
            )
        },
        "Another Candidate, Skills": {
            "name": "Aisha Khan",
            "email": "aisha@example.com",
            "phone": "9876543210",
            "total_experience": 2.0,
            "skills": ["Java", "Spring Boot", "MySQL", "AWS", "Docker"],
            "education": ["B.Tech in Computer Science"],
            "experience_raw": (
                "Software Engineer at CloudNine (2 years): Built RESTful APIs using Java and Spring Boot. "
                "Maintained and optimized production databases in MySQL. Worked with AWS deployment pipelines."
            )
        }
    }
    
    # Simple file content hash to cycle mock data
    file_bytes = file.getvalue()
    # Use a deterministic hash based on the file content for consistent mock data
    content_hash = hash(file_bytes)
    if content_hash % 2 == 0:
        data = mock_data["Name, Skills, Experience"]
    else:
        data = mock_data["Another Candidate, Skills"]

    return data

def highlight_keywords(text, keywords):
    """Highlights keywords in a block of text using HTML and SpaCy for accurate token matching."""
    if not nlp or not text:
        return text

    doc = nlp(text)
    highlighted_text = ""
    last_end = 0

    # Ensure keywords are lowercased and unique for matching
    lower_keywords = {k.lower() for k in keywords}

    for token in doc:
        token_text = token.text
        token_lower = token.text.lower()
        
        # Check if the token is a match for any of the keywords
        if token_lower in lower_keywords or any(token_lower in kw for kw in lower_keywords if len(kw) > 3):
            # Check for multi-word phrases (e.g., 'Data Science')
            is_phrase = False
            for kw in lower_keywords:
                if len(kw.split()) > 1 and kw in text[token.idx:].lower():
                    # Find the end of the full phrase
                    phrase_start_index = token.idx
                    phrase_end_index = token.idx + text[token.idx:].lower().find(kw) + len(kw)
                    
                    highlighted_text += text[last_end:phrase_start_index]
                    highlighted_text += f'<mark style="background-color: #ffd700; color: black; border-radius: 3px; padding: 2px 4px;">{text[phrase_start_index:phrase_end_index]}</mark>'
                    last_end = phrase_end_index
                    is_phrase = True
                    break
            
            # Highlight single token if no phrase was matched
            if not is_phrase:
                highlighted_text += text[last_end:token.idx]
                highlighted_text += f'<mark style="background-color: #ffd700; color: black; border-radius: 3px; padding: 2px 4px;">{token_text}</mark>'
                last_end = token.idx + len(token_text)
        
        # If the token is not a highlight, just advance the index
        if not (highlighted_text and last_end > token.idx):
            last_end = token.idx + len(token_text)
    
    # Append any remaining text
    highlighted_text += text[last_end:]
    
    return highlighted_text

# --- SCORING FUNCTIONS ---

def calculate_match_score(jd_text, resume_data, skills_weight, exp_weight, edu_weight):
    """Calculates a weighted match score based on Skills, Experience, and Education."""
    
    # Clean the JD text
    jd_clean = clean_text(jd_text)
    
    # Initialize component scores
    skill_score = 0
    exp_score = 0
    edu_score = 0
    
    # 1. Skills Matching (Highest Weight)
    # Use skills from resume and JD keywords
    resume_skills_text = ' '.join(resume_data['skills'])
    corpus_skills = [jd_clean, clean_text(resume_skills_text)]
    
    if len(corpus_skills[0]) > 0 and len(corpus_skills[1]) > 0:
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix_skills = vectorizer.fit_transform(corpus_skills)
            skill_score = cosine_similarity(tfidf_matrix_skills[0:1], tfidf_matrix_skills[1:2])[0][0] * 100
        except ValueError:
            # Handles case where vocabulary is too small
            skill_score = 0
    
    # 2. Experience Matching (Moderate Weight)
    # Simple score based on years of experience matching a target of 5 years (can be customized)
    target_exp = 5.0 
    resume_exp = resume_data['total_experience']
    
    if resume_exp >= target_exp:
        exp_score = 100
    elif resume_exp > 0:
        # Linear scaling up to target_exp, e.g., 2.5 years of 5 is 50%
        exp_score = min(100, (resume_exp / target_exp) * 80 + 20) # 20% floor for having some experience
    
    # 3. Education Matching (Lower Weight)
    # Simple heuristic: Check if 'Master', 'Ph.D.', 'Data' or 'Engineering' exists in education text
    edu_text = ' '.join(resume_data['education']).lower()
    
    if 'master' in edu_text or 'phd' in edu_text:
        edu_score = 100
    elif 'bachelor' in edu_text and ('science' in edu_text or 'engineering' in edu_text or 'tech' in edu_text):
        edu_score = 70
    else:
        edu_score = 30 # Base score for having an education

    # Calculate Weighted Total Score
    total_weight = skills_weight + exp_weight + edu_weight
    
    # Apply weights and normalize to 100
    if total_weight > 0:
        weighted_score = (
            (skill_score * skills_weight) + 
            (exp_score * exp_weight) + 
            (edu_score * edu_weight)
        ) / total_weight
    else:
        weighted_score = 0 # Avoid division by zero if all weights are 0
        
    # Scale score to a 0-100 range and ensure it's a float
    return float(min(100, max(0, weighted_score)))

# --- EMAIL FUNCTION ---

def send_shortlist_email(recipient_email, candidate_name, sender_email, sender_password, smtp_server, smtp_port):
    """Sends a standardized shortlisting email using SMTP credentials."""
    
    subject = f"Congratulations, {candidate_name}! Your application has been shortlisted by IntelliHire."
    body = f"""
    Dear {candidate_name},

    We are pleased to inform you that your resume has been shortlisted for the position of **{st.session_state.get('jd_title', 'Data Scientist')}** based on the skills and experience outlined in your application.

    Your profile scored high in our automated screening process (IntelliHire AI).

    We will be reaching out shortly to schedule the next steps, which typically involve a technical assessment.

    Thank you for your interest in our company.

    Best Regards,

    The IntelliHire Recruiting Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade connection to secure
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return True, ""
    except Exception as e:
        return False, str(e)


# --- MAIN STREAMLIT APP LOGIC ---

def run_app():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    
    # Initialize DB and Session State
    init_db()
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ""
    if 'selected_row_id' not in st.session_state:
        st.session_state.selected_row_id = None
        
    st.title(f"{APP_TITLE} ")
    st.markdown("Automate your screening process using AI-powered skill and experience matching.")
    
    all_resumes_df = get_all_resumes()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("1. Job Description")
        # Text area to input JD
        jd_input = st.text_area(
            "Paste Job Description Here for Scoring",
            value=st.session_state.jd_text,
            height=200,
            key="jd_input_area"
        )
        
        # Update session state when text area changes
        if jd_input:
            st.session_state.jd_text = jd_input
            # Simple way to get a title for the email
            try:
                st.session_state.jd_title = jd_input.split('\n')[0].strip() or 'Job Candidate'
            except:
                st.session_state.jd_title = 'Job Candidate'


        # --- Customizable Weights ---
        st.header("2. AI Scoring Weights")
        st.markdown("Adjust the importance of each factor.")
        
        # Ensure weights sum to 100 for normalization clarity
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            skills_weight = st.slider("Skills (%)", 0, 100, 50, key="skills_w")
        with col_w2:
            exp_weight = st.slider("Experience (%)", 0, 100, 30, key="exp_w")
        with col_w3:
            edu_weight = st.slider("Education (%)", 0, 100, 20, key="edu_w")
            
        st.info(f"Total Weight: {skills_weight + exp_weight + edu_weight}%")

        # --- Resume Upload and Processing ---
        st.header("3. Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF Resumes", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        if st.button("Process & Save Resumes", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file to process.")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
                    for uploaded_file in uploaded_files:
                        # Simulate data extraction
                        data = extract_data_from_pdf(uploaded_file)
                        
                        # Prepare data for storage
                        new_id = str(uuid.uuid4())
                        
                        resume_record = {
                            'id': new_id,
                            'name': data.get('name', 'N/A'),
                            'email': data.get('email', 'N/A'),
                            'phone': data.get('phone', 'N/A'),
                            'total_experience': data.get('total_experience', 0.0),
                            'skills': json.dumps(data.get('skills', [])),
                            'education': json.dumps(data.get('education', [])),
                            'experience_raw': data.get('experience_raw', ''),
                            'score': 0.0, # Will be calculated later
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_resume_data(resume_record)
                st.success(f"Successfully processed and saved {len(uploaded_files)} resumes.")
                st.rerun() # Rerun to update the main table

        # --- Filter & Search ---
        st.header("4. Search & Filter")
        
        # Filter slider for Match Score
        min_score = st.slider(
            "Filter Minimum Match Score (%)",
            0, 100, 50,
            key="min_score_filter"
        )
        
        # Search bar
        search_term = st.text_input("Search by Name, Skill, or Email", key="search_term_input")
        
        # --- Email Configuration ---
        st.header("5. Email Configuration ")
        st.info("Required to send shortlisting emails (Use App Password for Gmail/Outlook)")
        sender_email = st.text_input("Your Email (Sender)", key="sender_email_input")
        sender_password = st.text_input("App Password/Token", type="password", key="sender_password_input")
        
        # Common SMTP settings
        default_smtp = "smtp.gmail.com"
        default_port = 587
        smtp_server = st.text_input("SMTP Server", default_smtp, key="smtp_server_input")
        smtp_port = st.number_input("SMTP Port", default_port, key="smtp_port_input")

    # --- MAIN CONTENT AREA ---
    
    # --- Calculate Match Scores Button ---
    if st.button("Calculate Match Scores", key="calculate_scores_btn", type="secondary"):
        if not st.session_state.jd_text.strip():
            st.error("Please paste a Job Description in the sidebar first.")
        elif not all_resumes_df.empty:
            with st.spinner("Calculating match scores..."):
                for index, row in all_resumes_df.iterrows():
                    # Calculate new score
                    new_score = calculate_match_score(
                        st.session_state.jd_text,
                        {
                            'skills': json.loads(row['skills']),
                            'total_experience': row['total_experience'],
                            'education': json.loads(row['education']),
                        },
                        skills_weight, exp_weight, edu_weight
                    )
                    
                    # Update score in the database
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE resumes SET score = ? WHERE id = ?", 
                        (round(new_score, 2), row['id'])
                    )
                    conn.commit()
                    conn.close()
            st.success("Match scores updated!")
            st.rerun()

    # --- Filter & Search Data ---
    if not all_resumes_df.empty:
        # Filtering by score
        filtered_df = all_resumes_df[all_resumes_df['score'] >= min_score]
        
        # Searching by term
        if search_term:
            search_lower = search_term.lower()
            filtered_df = filtered_df[
                filtered_df['name'].str.lower().str.contains(search_lower, na=False) |
                filtered_df['email'].str.lower().str.contains(search_lower, na=False) |
                filtered_df['skills'].str.lower().str.contains(search_lower, na=False)
            ]

    # --- Display Final Table ---
    if all_resumes_df.empty:
        st.info("No candidate resumes found. Please upload and process files using the sidebar.")
    else:
        # Prepare DataFrame for Display
        display_df = filtered_df.copy()
        display_df['skills_list'] = display_df['skills'].apply(lambda x: ', '.join(json.loads(x)[:3]) + '...' if json.loads(x) else 'N/A')
        display_df['Match Score'] = display_df['score'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder and rename columns for display
        columns_to_show = [
            'id', 'name', 'email', 'total_experience', 'skills_list', 'Match Score'
        ]
        
        display_df = display_df[columns_to_show]
        display_df.rename(columns={'total_experience': 'Experience (Yrs)', 'skills_list': 'Top Skills'}, inplace=True)
        
        # Set ID as the index for selection handling
        display_df.set_index('id', inplace=True)

        st.subheader(f"Candidate Database ({len(display_df)} Matches Found)")
        
        # --- Email Button ---
        email_col, _ = st.columns([1, 4])
        with email_col:
            if st.button("Send Shortlisting Emails to Selected Candidates", key="send_email_btn", type="primary"): 
                if not sender_email or not sender_password:
                    st.error("Please enter your sender email and App Password/Token in the sidebar.")
                else:
                    selected_ids = st.session_state.get('selected_rows_email', [])
                    if not selected_ids:
                        st.warning("Please select at least one candidate using the checkbox on the left.")
                    else:
                        emails_sent = 0
                        with st.spinner(f"Sending emails to {len(selected_ids)} candidates..."):
                            for row_id in selected_ids:
                                # Look up original row data using the ID
                                candidate = all_resumes_df[all_resumes_df['id'] == row_id].iloc[0]
                                success, error = send_shortlist_email(
                                    recipient_email=candidate['email'],
                                    candidate_name=candidate['name'],
                                    sender_email=sender_email,
                                    sender_password=sender_password,
                                    smtp_server=smtp_server,
                                    smtp_port=smtp_port
                                )
                                if success:
                                    emails_sent += 1
                                else:
                                    st.error(f"Failed to send email to {candidate['name']} ({candidate['email']}): {error}")
                                time.sleep(0.5) # Prevent rate limiting
                        st.success(f"Successfully sent {emails_sent} shortlisting email(s).")
        
        
        # Display Interactive Table (DataEditor)
        # We need to drop the custom button column from the definition
        edited_df = st.data_editor(
            display_df.sort_values(by='Match Score', ascending=False), # Default sort by score
            column_config={
                # No ButtonColumn here
                'Match Score': st.column_config.ProgressColumn(
                    "Match Score",
                    help="AI Match Percentage",
                    format="%.2f %%",
                    min_value=0,
                    max_value=100,
                ),
                'name': st.column_config.Column("Name", width="medium"),
                'email': st.column_config.Column("Email", width="medium"),
            },
            hide_index=False, # Keep index visible for row selection
            num_rows="dynamic",
            use_container_width=True,
            key="candidate_data_editor",
        )
        
        # Store selected rows for email function (using the index, which is the 'id')
        selected_ids_for_email = edited_df.index[st.session_state.candidate_data_editor['selection']['rows']].tolist()
        st.session_state.selected_rows_email = selected_ids_for_email


        # --- Handle Delete Button Logic (Using a separate expander for compatibility) ---
        st.divider()
        st.subheader("Candidate Actions")

        if not filtered_df.empty:
            delete_col, details_col = st.columns(2)
            
            with delete_col:
                candidate_id_to_delete = st.selectbox(
                    "Select Candidate to Delete",
                    options=filtered_df['id'].unique(),
                    format_func=lambda id: all_resumes_df[all_resumes_df['id'] == id].iloc[0]['name'],
                    key="delete_candidate_id"
                )
                
                if st.button(f"Delete Candidate: {all_resumes_df[all_resumes_df['id'] == candidate_id_to_delete].iloc[0]['name']}", type="secondary"):
                    delete_resume_by_id(candidate_id_to_delete)
                    st.toast(f"Candidate {all_resumes_df[all_resumes_df['id'] == candidate_id_to_delete].iloc[0]['name']} deleted successfully!", icon="🗑️")
                    st.session_state.selected_row_id = None # Clear selected details
                    time.sleep(1)
                    st.rerun()


            # --- Handle Row Click for Detailed View ---
            with details_col:
                selected_ids = st.session_state.candidate_data_editor.get("selection", {}).get("rows", [])
                
                if selected_ids:
                    # Get the ID of the selected row from the index
                    selected_index = selected_ids[0]
                    selected_id = display_df.index[selected_index]
                    st.session_state.selected_row_id = selected_id
                    st.markdown(f"**Selected Candidate:** {display_df.loc[selected_id, 'name']}")
                else:
                    st.markdown("**Selected Candidate:** None (Click row in table for details)")

        # --- Display Detailed Candidate View ---
        if st.session_state.selected_row_id:
            # Fetch the *original* data for the detailed view
            candidate_details = all_resumes_df[all_resumes_df['id'] == st.session_state.selected_row_id].iloc[0].to_dict()
            
            st.divider()
            st.subheader(f"Profile: {candidate_details['name']} ({candidate_details['score']:.2f}% Match)")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown(f"**Email:** {candidate_details['email']}")
                st.markdown(f"**Phone:** {candidate_details['phone']}")
            with col2:
                st.markdown(f"**Experience:** {candidate_details['total_experience']:.1f} Years")
            with col3:
                st.markdown(f"**Education:** {', '.join(json.loads(candidate_details['education']))}")

            
            st.markdown(f"**Skills:** `{'` | `'.join(json.loads(candidate_details['skills']))}`")
            st.caption("---")
            
            # --- Keyword Highlighting Section ---
            
            # Extract keywords from the JD using a simple method (split and clean)
            jd_keywords = set(clean_text(st.session_state.jd_text).split())
            
            # Clean and format the raw experience for display
            raw_experience = candidate_details['experience_raw']
            
            # Highlight the keywords in the raw text
            highlighted_html = highlight_keywords(raw_experience, jd_keywords)
            
            st.subheader("Raw Experience & Keyword Match")
            st.markdown(f"Match Score Drivers: **{st.session_state.jd_title}**")

            # Display the highlighted text using markdown with HTML support
            st.markdown(f"""
                <div style="background-color: #f7f7f7; padding: 15px; border-radius: 8px; border: 1px solid #ddd; max-height: 400px; overflow-y: auto; color: black;">
                    {highlighted_html}
                </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Last updated: {candidate_details['timestamp']}")


if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        # Check if the error is related to column_config which suggests an old Streamlit version
        if "column_config" in str(e):
             st.error(f"An unexpected error occurred: {e}. This likely means your Streamlit version is too old for the `ProgressColumn` or `Column` definitions.")
             st.info("Please run `pip install --upgrade streamlit` in your terminal and try again. The current code is now using a highly compatible version of the delete logic.")
        else:
             st.error(f"An unexpected error occurred: {e}")
             st.info("Please ensure all dependencies are installed: `pip install streamlit sqlite3 pandas scikit-learn nltk spacy` and `python -m spacy download en_core_web_sm`.")
