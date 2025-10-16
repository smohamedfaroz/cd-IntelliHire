import sqlite3
import pandas as pd

DB = "resumes.db"

def init_db():
    """Initializes the SQLite database and creates the 'resumes' table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            experience_years REAL,
            education TEXT,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score REAL
        )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def insert_resume(filename, name, email, phone, skills, exp_years, education, raw_text):
    """Inserts a parsed resume record into the database."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO resumes 
            (filename, name, email, phone, skills, experience_years, education, raw_text) 
            VALUES (?,?,?,?,?,?,?,?)
        """, (filename, name, email, phone, skills, exp_years, education, raw_text))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting record: {e}")
    finally:
        if conn:
            conn.close()

def query_resumes_by_skill(skill_query):
    """Queries resumes where the skills field contains the specified skill."""
    conn = sqlite3.connect(DB)
    # Using f-string for LIKE search
    df = pd.read_sql_query(
        f"SELECT * FROM resumes WHERE skills LIKE '%{skill_query}%'", 
        conn
    )
    conn.close()
    return df
    
def get_all_resumes_df():
    """Retrieves all resumes from the database as a Pandas DataFrame."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM resumes", conn)
    conn.close()
    return df