import streamlit as st
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import streamlit.components.v1 as components
import time
# from streamlit_star_rating import st_star_rating

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

def recommend(user_input):
    user_input = lemmatize(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, vectors)
    top_indices = similarity_scores.argsort()[0][-10:][::-1]  # Top 10 results
    
    recommendations = []
    for index in top_indices:
        recommendation = {
        "category": fetch_category(index),
        "headline": fetch_headline(index),
        "link": fetch_links(index)
   
    }
        recommendations.append(recommendation)
    
    return recommendations
    
matrix_js = """
<script>
consol.log("Hello")
let startTime = Date.now();
let lastScrollPos = window.pageYOffset;
let scrollDistance = 0;
let scrollCount = 0;
let clickCount = 0;

// Track scrolling
document.addEventListener('scroll', function() {
    let currentPos = window.pageYOffset;
    scrollDistance += Math.abs(currentPos - lastScrollPos);
    lastScrollPos = currentPos;
    scrollCount++;
});

// Track clicks
document.addEventListener('click', function() {
    clickCount++;
});

// Send metrics every 2 seconds
setInterval(function() {
    let currentTime = Date.now();
    let timeElapsed = (currentTime - startTime) / 1000;
    let avgScrollSpeed = scrollCount > 0 ? scrollDistance / timeElapsed : 0;
    
    // Send to Streamlit
    window.parent.postMessage({
        type: 'metrics',
        scrollSpeed: avgScrollSpeed,
        clicks: clickCount
    }, '*');
}, 2000);
</script>
"""
# Dictionary to store stats per search
search_stats = {}


# Function to display JS in Streamlit
def metrix():
    components.html(matrix_js, height=0)

def temp_func():
    print("This is a Rating Check")

def main():
    metrix()

    st.title('Newsify - News Recommendation System')
    user_input = st.text_input(label="Search for politics, tech, entertainment and more", value="")
    if st.button('Find Recommendations'):
        if user_input:
            recommendations = recommend(user_input)
            st.write("Here are the top 10 recommended news articles based on your input:")
            for rec in recommendations:
                 st.write(f"Category: {rec['category']}")
                 st.subheader(f"{rec['headline']}")
                 st.link_button("Read More", rec['link']) 
                 st.subheader(" ")

            # stars = st_star_rating("Please rate you experience", maxValue=5, defaultValue=3, key="rating", on_click=temp_func)
        else:
            st.error("Please enter a valid query to get recommendations.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Clicks", value="0")
    with col2:
        st.metric(label="Average Scroll Speed (px/s)", value="0.00")

if __name__ == "__main__":
    main()

