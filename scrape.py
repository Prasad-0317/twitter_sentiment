import pandas as pd
from ntscraper import Nitter
import streamlit as st

def get_tweets(name,modes,no):
  scraper = Nitter()
  tweets = scraper.get_tweets(name,mode=modes,number=no)
  final_tweets =[]
  for tweet in tweets['tweets']:
    data = [tweet['text'] , tweet['date'], tweet['stats']['likes'], tweet['stats']['retweets']]
    final_tweets.append(data)
  data = pd.DataFrame(final_tweets, columns=['text','date','No_of_likes','No_of_retweets'])
  return data

class SessionState:
    def __init__(self):
        self.fetch_clicked = False

session_state = SessionState()

with st.form(key='tweet_fetch_form'):
    user_name = st.text_input("Enter Twitter term, or hashtag", key='user_name')
    user_mode = st.selectbox("Select Mode", ['term', 'hashtag'], key='user_mode')
    user_number = st.number_input("Enter Number of Tweets", min_value=1, max_value=1000, step=1, key='user_number')
    
    fetch_button = st.form_submit_button("Fetch Tweets")

    if fetch_button:
        session_state.fetch_clicked = True

if session_state.fetch_clicked:
        if user_name and user_mode and user_number:
            # Fetch tweets based on user input
            data = get_tweets(user_name, user_mode, user_number)
            st.write("### Fetched Tweets")
            st.write(data)  # Display the fetched tweets in a DataFrame format
            df = data
            print(df.shape)
            df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y · %I:%M %p UTC')
            df['date'] = df['date'].dt.strftime('%d-%m-%Y')
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('ggplot')
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            from tqdm.notebook import tqdm
            nltk.download('vader_lexicon')
            sia = SentimentIntensityAnalyzer()

            sia.polarity_scores('I am so happy!')

            sia.polarity_scores('This is the worst thing ever.')

    

            df['id'] = range(1, len(df) + 1)

            res = {}
            for i, row in tqdm(df.iterrows(), total=len(df)):
                text = row['text']
                myid = row['id']
                res[myid] = sia.polarity_scores(text)

            vaders = pd.DataFrame(res).T
            vaders = vaders.reset_index().rename(columns={'index': 'id'})
            vaders = vaders.merge(df, how='left')

            st.write("### DataFrame with scores")
            st.write(vaders.head(10))

            from transformers import AutoTokenizer
            from transformers import AutoModelForSequenceClassification
            from scipy.special import softmax

            MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL)

            import nltk
            nltk.downloader.download('vader_lexicon')
            def label_text(text):
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sid = SentimentIntensityAnalyzer()
                score = sid.polarity_scores(text)

                if score['compound'] >= 0.05:
                    label = 'Positive'
                elif score['compound'] <= -0.05:
                    label = 'Negative'
                else:
                    label = 'Neutral'

                return label

            df['labelSentiment'] = df['text'].apply(label_text)
            st.write("### DataFrame")
            st.write(df)

            sentiment_column = df['labelSentiment']
            positive_count = sentiment_column.value_counts()['Positive']
            negative_count = sentiment_column.value_counts()['Negative']
            neutral_count = sentiment_column.value_counts()['Neutral']

            colors = ['#99ff99', '#ff9999', '#ffcc99']
            plt.pie([positive_count, negative_count,neutral_count], labels=['Positive', 'Negative','Neutral'],colors=colors, autopct="%1.1f%%",radius=0.5)

            st.write('### Sentiment Distribution')
            plt.shadow = True
            # plt.show()
            st.pyplot(plt)

            from nltk.corpus import stopwords
            nltk.download('vader_lexicon')
            nltk.download('stopwords')
            stemmer = nltk.SnowballStemmer("english")
            import string
            stopword = set(stopwords.words('english'))

            def clean(text):
                text = str(text).lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w*\d\w*', '', text)
                text = [word for word in text.split(' ') if word not in stopword]
                text = " ".join(text)
                text = [stemmer.stem(word) for word in text.split(' ')]
                text = " ".join(text)
                return text

            import re
            re.compile('<title>(.*)</title>')
            df['text'] = df['text'].apply(clean)
            st.write("### Count of Reviews by label")
            ax = df['labelSentiment'].value_counts().sort_index().to_frame().plot(kind='bar',
                                                                            figsize=(10, 5))

            st.pyplot(plt) 

            all_words = " ".join([sentence for sentence in df['text']])

            from wordcloud import WordCloud
            wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100).generate(all_words)

            plt.figure(figsize=(10,6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.write("## Word Cloud (frequent words)")
            st.pyplot(plt) 
            len(set(all_words)) 

            df = df.assign(label=df['labelSentiment'].map({'Positive': 1, 'Negative': -1,'Neutral':0}))
            vaders = pd.DataFrame(res).T
            vaders = vaders.reset_index().rename(columns={'index': 'id'})

            pos_words = " ".join([sentence for sentence in df['text'][df['label']==1]])

            wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100).generate(pos_words)

            plt.figure(figsize=(2,4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.write("## Positive (frequent words)")
            st.pyplot(plt) 
            neg_words = " ".join([sentence for sentence in df['text'][df['label']==-1]])

            wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100).generate(neg_words)

            plt.figure(figsize=(10,6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.write("## Negative (frequent words)")
            st.pyplot(plt) 


            file_path = 'D:/revanth.xlsx'
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)  

        else:
            st.warning("Please fill in all the fields")


# Add custom CSS
container_style = """
        <style>
        .stApp {
            background-color: #CBEEF3;
        }
        </style>
    """
st.markdown(container_style, unsafe_allow_html=True)