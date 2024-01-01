import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import logging
logging.set_verbosity_error()
# Load sentiment analysis model
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_id)
# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]["label"], result[0]["score"]

# Function to create a word cloud
def generate_wordcloud(comments):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(" ".join(comments))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

# Streamlit GUI
def main():
    # Set overall page style including the background color
    st.set_page_config(
        page_title="Facebook Post Sentiment Analyzer",
        page_icon=":smiley:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Use a container with a specific style to set the background color
    container_style = """
        <style>
        .stApp {
            background-color: #CBEEF3;
        }
        </style>
    """
    st.markdown(container_style, unsafe_allow_html=True)

    st.title("Facebook Post Sentiment Analyzer")

    # Input for the Facebook post
    post_text = st.text_area("Enter Facebook Post:", "")

    # Input for comments
    comments_text = st.text_area("Enter Comments (one per line):", "")

    # Analyze sentiment when the button is clicked
    if st.button("Analyze Sentiment"):
        if not post_text:
            st.warning("Please enter a Facebook post.")
        else:
            st.success("Sentiment Analysis Results:")
            sentiment, score = analyze_sentiment(post_text)
            st.write(f"Sentiment for Post: {sentiment}")
            st.write(f"Confidence: {score:.2%}")

            if comments_text:
                st.write("\n---\n")
                comments_list = comments_text.split("\n")
                for i, comment in enumerate(comments_list, 1):
                    sentiment, score = analyze_sentiment(comment)
                    st.write(f"Comment {i}: {comment}")
                    st.write(f"Sentiment: {sentiment}")
                    st.write(f"Confidence: {score:.2%}")
                    st.write("\n---\n")

                # Generate and display word cloud
                generate_wordcloud(comments_list)


# Run the Streamlit app
if __name__ == "__main__":
    main()