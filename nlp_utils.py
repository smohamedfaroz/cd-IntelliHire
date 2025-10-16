from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity_score(resume_texts: list, job_description: str) -> list:
    """
    Computes TF-IDF Cosine Similarity between a list of resume texts and a job description.
    """
    if not resume_texts or not job_description:
        return [0.0] * len(resume_texts)

    # The corpus for fitting the vectorizer includes all resumes and the job description
    corpus = resume_texts + [job_description]
    
    # Initialize the Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Fit and transform the corpus
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError as e:
        print(f"TF-IDF Error: {e}")
        return [0.0] * len(resume_texts)

    # Separate the resume vectors from the job description vector
    resume_vecs = X[:-1]
    job_vec = X[-1]
    
    # Compute the cosine similarity between all resume vectors and the JD vector
    sims = cosine_similarity(resume_vecs, job_vec.reshape(1, -1))
    
    # sims is an Nx1 array, convert to a flat list
    return sims.flatten().tolist()