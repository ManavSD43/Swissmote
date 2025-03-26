import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def recommend_properties(user_input, property_data):
    """
    Recommend properties based on user preferences using text similarity.
    :param user_input: A dictionary with user preferences (e.g., {"location": "New York", "budget": "500000", "bedrooms": "3"})
    :param property_data: A Pandas DataFrame containing property listings.
    :return: Recommended properties sorted by relevance.
    """
    
    # Combine all text-based property details into a single description column
    property_data['description'] = property_data.apply(lambda x: f"{x['location']} {x['budget']} {x['bedrooms']} {x['features']}", axis=1)
    
    # Convert user input into a query string
    user_query = f"{user_input['location']} {user_input['budget']} {user_input['bedrooms']}"
    
    # Use TF-IDF Vectorizer to transform text data
    vectorizer = TfidfVectorizer()
    property_vectors = vectorizer.fit_transform(property_data['description'])
    user_vector = vectorizer.transform([user_query])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(user_vector, property_vectors).flatten()
    
    # Attach scores to properties and sort
    property_data['score'] = similarity_scores
    recommendations = property_data.sort_values(by='score', ascending=False)
    
    return recommendations[['location', 'budget', 'bedrooms', 'features', 'score']].head(5)

# Sample property data
data = pd.DataFrame({
    'location': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Miami'],
    'budget': [500000, 700000, 450000, 800000, 600000],
    'bedrooms': [3, 2, 3, 4, 2],
    'features': ['Pool, Garage', 'Balcony, Garden', 'Near Park, Garage', 'Ocean View, Modern', 'Pool, Balcony']
})

# Streamlit UI
def main():
    st.title("AI-Powered Real Estate Recommendation System")
    
    # User input fields
    location = st.text_input("Enter preferred location:")
    budget = st.number_input("Enter your budget:", min_value=0, step=50000)
    bedrooms = st.number_input("Number of bedrooms:", min_value=1, step=1)
    
    if st.button("Find Properties"): 
        user_preferences = {'location': location, 'budget': str(budget), 'bedrooms': str(bedrooms)}
        recommended_properties = recommend_properties(user_preferences, data)
        
        st.subheader("Recommended Properties")
        st.dataframe(recommended_properties)

if __name__ == "__main__":
    main()
