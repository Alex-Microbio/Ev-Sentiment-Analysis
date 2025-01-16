# Electric Vehicle (EV) Sentiment Analysis Project

This project analyzes public sentiment toward electric vehicles (EVs) using a dataset of tweets from the Sentiment140 dataset. It includes the following features:

- **Data Cleaning**: Preprocessing text data by removing URLs, mentions, hashtags, and stop words.
- **Sentiment Analysis**: Using a custom sentiment analyzer based on a lexicon of positive and negative terms.
- **Data Visualization**: Generating insights through bar charts, time-series analysis, and word clouds.
- **TF-IDF Analysis**: Extracting important terms associated with positive and negative sentiments.
- **Topic Modeling**: Identifying themes in EV-related discussions using Latent Dirichlet Allocation (LDA).

## Project Setup

### 1. Installation
Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn kagglehub
```

### 2. Dataset
Download the Sentiment140 dataset automatically using KaggleHub:
```python
path = kagglehub.dataset_download("kazanova/sentiment140")
```

### 3. Usage
Run the main Python script (`ev_sentiment_analysis.py`) to:

1. **Download and clean the dataset**.
2. **Analyze sentiment by manufacturers**.
3. **Generate visualizations** for trends, distributions, and important terms.

## Key Scripts

### 1. `data_cleaning.py`
Contains functions for:
- Removing URLs, hashtags, and special characters.
- Filtering tweets for EV-related mentions.

### 2. `sentiment_analysis.py`
Implements sentiment analysis using a custom lexicon and calculates:
- Positive, negative, and neutral sentiment distributions.

### 3. `visualization.py`
Generates visualizations, including:
- Word clouds.
- Sentiment distributions by manufacturer.
- Time-series trends for sentiment.

### 4. `tfidf_analysis.py`
Calculates and visualizes important terms for positive and negative sentiments using TF-IDF.

## Results
- **Overall Sentiment Distribution**:
    - Positive: `XX%`
    - Neutral: `XX%`
    - Negative: `XX%`

- **Key Terms**:
    - Positive Sentiments: `battery, charging, range`
    - Negative Sentiments: `expensive, slow, limited`

- **Visualizations**:
    - Sentiment distribution by manufacturer.
    - Word cloud for EV-related terms.

## Insights
- High positive sentiment toward Tesla due to innovation and range.
- Common concerns include limited charging infrastructure and high costs.

## Contributing
Feel free to fork and submit pull requests to improve sentiment analysis or visualizations.

## License
This project is licensed under the MIT License.

