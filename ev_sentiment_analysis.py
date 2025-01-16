import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import kagglehub

# Custom Vader Lexicon for Sentiment Analysis
vader_lexicon = {
    "good": 2.0,
    "excellent": 3.0,
    "great": 2.5,
    "awesome": 3.0,
    "bad": -2.0,
    "terrible": -3.0,
    "horrible": -3.0,
    "poor": -2.5
}

# Custom implementation of Sentiment Analyzer
class CustomSentimentAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def polarity_scores(self, text):
        words = text.split()
        score = sum([self.lexicon.get(word, 0) for word in words])
        return {"compound": score}

# Step 1: Download Dataset from Kaggle
path = kagglehub.dataset_download("kazanova/sentiment140")
print("Path to dataset files:", path)

# Load the dataset in chunks to handle memory issues
data_path = f"{path}/training.1600000.processed.noemoticon.csv"  # Adjust based on the dataset structure
try:
    chunk_size = 1000  # Process in chunks
    chunks = []
    for chunk in pd.read_csv(data_path, encoding='latin-1', header=None, chunksize=chunk_size):
        chunk.columns = ['target', 'id', 'date', 'flag', 'user', 'text']  # Rename columns
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
except FileNotFoundError:
    print("Dataset not found. Please ensure the file path is correct.")

# Filter for Electric Car Mentions
data = data[data['text'].str.contains("electric car|EV|Tesla|electric vehicle", case=False, na=False)]
print(f"Filtered dataset contains {data.shape[0]} rows related to electric cars.")

# Step 2: Display basic information and statistics
print("\nDataset Overview:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe(include='all'))

# Step 3: Data Cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#[\w-]+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_text'] = data['text'].apply(clean_text)

# Step 4: Tokenization and Stopword Removal
# Define a custom list of common stopwords
stop_words = [
    "english", "though", "maybe", "wont", "twitter", "marketers", "book", "sold", "woodstock", "stuff", "stocking", "lunch", 
    "people", "money", "sorry", "order", "food", "found", "seems", "tonight", "guess", "sad", "soon", "keep", "hours", 
    "never", "say", "internet", "check", "store", "little", "house", "yet", "enough", "stockton", "always", "bit", "fresh", 
    "late", "ready", "phone", "miss", "around", "supermarkets", "best", "big", "ours", "ourselves", "hair", "yourself", 
    "yourselves", "himself", "herself", "itself", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", 
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not"
]
def remove_stopwords(text):
    tokens = text.split()  # Use simple string split instead of word_tokenize
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

# Step 5: Sentiment Analysis
sia = CustomSentimentAnalyzer(vader_lexicon)
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

data['sentiment_score'] = data['cleaned_text'].apply(analyze_sentiment)
data['sentiment'] = data['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Step 6: Data Visualization
# Downsample for visualization to avoid memory issues
sampled_data = data.sample(n=min(5000, data.shape[0]), random_state=42)

# Sentiment Distribution
sns.countplot(sampled_data['sentiment'], order=['Positive', 'Neutral', 'Negative'])
plt.title('Electric Car Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Word Cloud for Positive Sentiment (Hidden)
# positive_text = ' '.join(sampled_data[sampled_data['sentiment'] == 'Positive']['cleaned_text'])
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud for Positive Sentiments about Electric Cars')
# plt.show()

# Most Conclusive Visualization: Sentiment Distribution by Manufacturer and Overall Comparison
manufacturers = ["Tesla", "Nissan", "Chevrolet", "Ford", "Volkswagen"]
manufacturer_sentiments = []

for manufacturer in manufacturers:
    manufacturer_data = data[data['cleaned_text'].str.contains(manufacturer.lower(), case=False, na=False)]
    sentiment_counts = manufacturer_data['sentiment'].value_counts()
    sentiment_counts = sentiment_counts.reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
    manufacturer_sentiments.append({"Manufacturer": manufacturer, "Positive": sentiment_counts['Positive'], "Neutral": sentiment_counts['Neutral'], "Negative": sentiment_counts['Negative']})

# Convert to DataFrame for visualization
manufacturer_sentiments_df = pd.DataFrame(manufacturer_sentiments)
manufacturer_sentiments_df.set_index("Manufacturer", inplace=True)

# Plot Combined Sentiment Distribution
manufacturer_sentiments_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Combined Sentiment Distribution by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Sentiment Count')
plt.legend(title='Sentiment')
plt.show()

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF Vectorizer with EV-related keywords
ev_keywords = ["battery", "charging", "range", "sustainability", "eco-friendly", "autonomous", "electric"]
vectorizer = TfidfVectorizer(vocabulary=ev_keywords, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['cleaned_text'])

# Convert TF-IDF Matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Display Top TF-IDF Terms for Positive Sentiment
positive_data = data[data['sentiment'] == 'Positive']
positive_tfidf_matrix = vectorizer.fit_transform(positive_data['cleaned_text'])
positive_tfidf_df = pd.DataFrame(positive_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Summarize Top Words for Positive Sentiment
positive_tfidf_summary = positive_tfidf_df.sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
positive_tfidf_summary.plot(kind='bar', title='Top TF-IDF Terms for Positive Sentiment (EV-Specific)', color='green')
plt.ylabel('TF-IDF Score')
plt.xlabel('Terms')
plt.show()

# Display Top TF-IDF Terms for Negative Sentiment
negative_data = data[data['sentiment'] == 'Negative']
negative_tfidf_matrix = vectorizer.fit_transform(negative_data['cleaned_text'])
negative_tfidf_df = pd.DataFrame(negative_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Summarize Top Words for Negative Sentiment
negative_tfidf_summary = negative_tfidf_df.sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
negative_tfidf_summary.plot(kind='bar', title='Top TF-IDF Terms for Negative Sentiment (EV-Specific)', color='red')
plt.ylabel('TF-IDF Score')
plt.xlabel('Terms')
plt.show()

# Comparative Visualization of Positive and Negative TF-IDF Scores
combined_tfidf = pd.concat([positive_tfidf_summary, negative_tfidf_summary], axis=1, keys=['Positive', 'Negative'])
combined_tfidf.plot(kind='bar', figsize=(12, 6), title='Comparison of Top Terms in Positive and Negative Sentiment (EV-Specific)', colormap='coolwarm')
plt.xlabel('Terms')
plt.ylabel('TF-IDF Score')
plt.legend(title='Sentiment')
plt.show()

# Sentiment Over Time (Trend Analysis)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
daily_sentiment = data.groupby([data['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
daily_sentiment.plot(kind='line', figsize=(12, 6), title='Daily Sentiment Trends')
plt.xlabel('Date')
plt.ylabel('Number of Mentions')
plt.legend(title='Sentiment')
plt.show()

# Regional Sentiment Analysis
if 'region' in data.columns:
    regional_sentiment = data.groupby(['region', 'sentiment']).size().unstack(fill_value=0)
    regional_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6), title='Regional Sentiment Distribution')
    plt.xlabel('Region')
    plt.ylabel('Mentions')
    plt.show()

# Topic Modeling (LDA)
from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)
print("Top Words for Each Topic:")
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic {topic_idx + 1}: {[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}")

# Add a Summary Table for Manufacturers
summary_table = manufacturer_sentiments_df.copy()
summary_table['Total Mentions'] = summary_table.sum(axis=1)
summary_table['Positive %'] = (summary_table['Positive'] / summary_table['Total Mentions']) * 100
summary_table['Negative %'] = (summary_table['Negative'] / summary_table['Total Mentions']) * 100
print(summary_table)

# Step 7: Save Cleaned Data
cleaned_data_path = 'cleaned_electric_car_data.csv'
data.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to {cleaned_data_path}")

# Step 8: Summary and Insights
positive_percentage = (data[data['sentiment'] == 'Positive'].shape[0] / data.shape[0]) * 100
negative_percentage = (data[data['sentiment'] == 'Negative'].shape[0] / data.shape[0]) * 100
neutral_percentage = 100 - (positive_percentage + negative_percentage)

print(f"Positive Sentiments: {positive_percentage:.2f}%")
print(f"Negative Sentiments: {negative_percentage:.2f}%")
print(f"Neutral Sentiments: {neutral_percentage:.2f}%")


# Iteration 5: Incorporate Trigrams and Refine Visualization

# Update TF-IDF Vectorizer to include trigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=300, stop_words= stop_words)
tfidf_matrix = vectorizer.fit_transform(sampled_data[sampled_data['sentiment'] == 'Positive']['cleaned_text'])

# Extract terms and their TF-IDF scores
tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1))
top_tfidf_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:100]

# Generate a combined text of top TF-IDF terms (including trigrams)
tfidf_text = ' '.join([term for term, score in top_tfidf_terms])

if tfidf_text.strip():
    # Generate Word Cloud with refined visualization
    wordcloud = WordCloud(
        width=1800,
        height=900,
        background_color='white',
        colormap='inferno',
        max_words=200,
        contour_width=2,
        contour_color='black'
    ).generate(tfidf_text)
    plt.figure(figsize=(18, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Enhanced Word Cloud for Positive Sentiments', fontsize=20)
    plt.show()
else:
    print("No significant terms or trigrams found in positive sentiment text.")
