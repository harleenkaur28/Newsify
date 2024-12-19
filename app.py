import streamlit as st
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import streamlit.components.v1 as components
from streamlit_star_rating import st_star_rating

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
    
scroll_js = """
<script>
let lastScrollTop = 0; 
let lastTime = 0;

function sendScrollSpeed(speed) {
    console.log("Scroll speed: " + speed + " px/s"); // Log the scroll speed to the console
    const streamlitContainer = window.parent.document.querySelector('iframe');
    if (streamlitContainer) {
        streamlitContainer.dispatchEvent(
            new CustomEvent("streamlit:customMessage", { detail: { scrollSpeed: speed } })
        );
    }
}

window.addEventListener("scroll", () => {
    const currentTime = Date.now(); 
    const currentScrollTop = window.scrollY;

    if (lastTime !== 0) {
        const distance = Math.abs(currentScrollTop - lastScrollTop);
        const timeDiff = (currentTime - lastTime) / 1000;
        const scrollSpeed = distance / timeDiff;

        // Log the scroll speed to the console
        sendScrollSpeed(scrollSpeed.toFixed(2));
    }

    lastScrollTop = currentScrollTop;
    lastTime = currentTime;
});
</script>
"""


# Function to display JS in Streamlit
def show_scroll_js():
    components.html(scroll_js, height=0)

def temp_func():
    print("This is a Rating Check")

def main():
    show_scroll_js()

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

            stars = st_star_rating("Please rate you experience", maxValue=5, defaultValue=3, key="rating", on_click=temp_func)
        else:
            st.error("Please enter a valid query to get recommendations.")

if __name__ == "__main__":
    main()

