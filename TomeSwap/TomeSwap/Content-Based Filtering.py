# Content-Based Filtering Algorithm (Example: TfidfVectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample book titles
book_titles = ["The Great Gatsby", "To Kill a Mockingbird", "Pride and Prejudice", ...]

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit-transform book titles to TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(book_titles)

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations for a given book index
def get_recommendations(book_index, cosine_sim=cosine_sim):
    sim_scores = list(enumerate(cosine_sim[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar books
    book_indices = [i[0] for i in sim_scores]
    return book_indices

# Get recommendations for a book
book_index = 0  # Replace with the index of the book for which you want recommendations
recommendations = get_recommendations(book_index)
print("Recommended Book Indices:", recommendations)
