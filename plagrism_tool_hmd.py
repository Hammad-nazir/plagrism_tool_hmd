import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Specify the directory containing text files
directory = '/home/hammad/Documents/CODEXCUE_PROJECTS'  # Update this path

# List text files in the specified directory
student_files = [doc for doc in os.listdir(directory) if doc.endswith('.txt')]
print(f"Files found: {student_files}")

# Check if no files are found
if not student_files:
    raise FileNotFoundError("No text files found in the specified directory.")

# Read and preprocess the text documents
student_notes = []
for file in student_files:
    file_path = os.path.join(directory, file)
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
        student_notes.append(content)

# Preprocess the text documents
preprocessed_notes = [preprocess_text(note) for note in student_notes]

# Print out preprocessed notes to debug
for idx, note in enumerate(preprocessed_notes):
    print(f"Document {student_files[idx]}: {note}")

# Check if any documents are empty after preprocessing
if all(note == '' for note in preprocessed_notes):
    raise ValueError("All documents are empty after preprocessing.")

# Vectorize text using TF-IDF
def vectorize(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

# Calculate similarity between two vectors
def similarity(vec1, vec2):
    return cosine_similarity([vec1, vec2])[0][1]

# Vectorize student notes
vectors = vectorize(preprocessed_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

def check_plagiarism():
    global s_vectors
    for i, (student_a, text_vector_a) in enumerate(s_vectors):
        for j, (student_b, text_vector_b) in enumerate(s_vectors):
            if i < j:  # Avoid comparing the same pair twice
                sim_score = similarity(text_vector_a, text_vector_b)
                if sim_score > 0.5:  # Threshold for similarity score
                    plagiarism_results.add((student_a, student_b, sim_score))
    return plagiarism_results

# Print plagiarism results
for result in check_plagiarism():
    print(f"Plagiarism detected between {result[0]} and {result[1]} with similarity score: {result[2]:.2f}")
