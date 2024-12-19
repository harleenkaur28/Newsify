import streamlit as st
import joblib
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import streamlit.components.v1 as components
from datetime import datetime
st.set_page_config(page_title="", layout="wide")

lemmatizer = WordNetLemmatizer()
vectors= joblib.load('vectors.pkl')
vectorizer=joblib.load('vectorizer.pkl')

data = []
with open('News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))   

def lemmatize(text):
    y=[]
    for i in text.split():
        y.append(lemmatizer.lemmatize(i,  pos='v'))
    return " ".join(y)

def fetch_headline(index):
    headline= data[index]['headline']
    return headline

def fetch_links(index):
    link=data[index]['link']
    return link

def fetch_category(index):
    category=data[index]['category']
    return category

def classify_similarity(similarity_score):
    if similarity_score >= 0.5:
        return "Good"
    elif similarity_score >= 0.2:
        return "Moderate"
    else:
        return "Bad"

def recommend(user_input):
    user_input = lemmatize(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, vectors)
    total_similarity=0
    top_indices = similarity_scores.argsort()[0][-10:][::-1]  # Top 10 results
    
    recommendations = []
    
    for index in top_indices:
        similarity_score = similarity_scores[0][index]  
        classification = classify_similarity(similarity_score)
        
        recommendation = {
            "category": fetch_category(index),
            "headline": fetch_headline(index),
            "link": fetch_links(index),
            "similarity_score": similarity_score,
            "classification": classification  
        }
        recommendations.append(recommendation)
        total_similarity+= similarity_score

    avg_similarity= total_similarity/10
    avg_classification = classify_similarity(avg_similarity)
    
    return recommendations, avg_classification
 
def main():
    st.title('Newsify - News Recommendation System')
    
    # Initialize session state for tracking submission
    if 'rating_submitted' not in st.session_state:
        st.session_state.rating_submitted = False
    if 'show_rating_input' not in st.session_state:
        st.session_state.show_rating_input = False
    
    user_input = st.text_input(
        label="Search for politics, tech, entertainment and more",
        value=""
    )
    
    if st.button('Find Recommendations'):
        if user_input.strip():
            recommendations, avg_classification = recommend(user_input)
            st.write("Here are the top 10 recommended news articles based on your input:")
            
            for rec in recommendations:
                st.write(f"Category: {rec['category']}")
                st.subheader(f"{rec['headline']}")
                st.link_button("Read More", rec['link'])
                st.write(rec['classification'])
                st.subheader(" ")
            
            # Store the query and classification in session state
            st.session_state.last_query = user_input
            st.session_state.last_classification = avg_classification
            # Show rating input after recommendations
            st.session_state.show_rating_input = True
        else:
            st.error("Please enter a valid query to get recommendations.")
    
    # Only show rating input after recommendations are displayed
    if st.session_state.show_rating_input and not st.session_state.rating_submitted:
        rate = st.text_input(
            label="How relevant were recommendations? (Enter 1-10)",
            key="rating_input"
        )
        
        # Submit button for rating
        if st.button("Submit Rating"):
            try:
                if rate.strip():
                    rating = int(rate)
                    
                    if 1 <= rating <= 10:
                        feedback_data = {
                            'query': st.session_state.get('last_query', ''),
                            'user_rating': rating,
                            'avg_classification': st.session_state.get('last_classification', ''),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        filename = 'feedback.json'
                        
                        try:
                            # Read existing data
                            if os.path.exists(filename):
                                with open(filename, 'r') as f:
                                    try:
                                        existing_data = json.load(f)
                                        if not isinstance(existing_data, list):
                                            existing_data = []
                                    except json.JSONDecodeError:
                                        existing_data = []
                            else:
                                existing_data = []

                            existing_data.append(feedback_data)

                            with open(filename, 'w') as f:
                                json.dump(existing_data, f, indent=4)

                            if feedback_data['avg_classification'] == "Good" and rating >= 7:
                                st.success("Thank you! The recommendations were highly relevant.")
                            elif feedback_data['avg_classification'] == "Moderate" and rating >= 4:
                                st.warning("Thanks! We will improve based on your feedback.")
                            else:
                                st.error("Sorry the recommendations weren't helpful enough.")

                            st.session_state.rating_submitted = True
                            
                        except Exception as e:
                            st.error(f"Error saving feedback: {str(e)}")
                    else:
                        st.error("Please enter a rating between 1 and 10")
                else:
                    st.error("Please enter a rating")
                    
            except ValueError:
                st.error("Please enter a valid number")

if __name__ == "__main__":
    main()