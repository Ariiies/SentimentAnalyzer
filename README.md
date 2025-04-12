# Sentiment Analysis with Python

A Python library for analyzing sentiment in text corpora using a lexicon-based approach, enhanced with TF-IDF weighting. This project provides three core classes: Corpy for corpus analysis, TFIDF for term weighting, and SentimentAnalyzer for sentiment classification.
## Features
- Corpus Analysis: Extract vocabulary, word frequencies, and document statistics with the Corpy class.
- TF-IDF Weighting: Compute term frequency-inverse document frequency (TF-IDF) scores using the TFIDF class.
- Sentiment Analysis: Classify text as positive, negative, or neutral with the SentimentAnalyzer class, leveraging a user-provided sentiment lexicon.
- Customizable: Easily adjust sentiment thresholds and lexicons to suit your needs.
- Lightweight: Built with minimal dependencies (numpy for matrix operations).

## Installation
Clone the repository:
```bash
git clone https://github.com/Ariiies/SentimentAnalyzer.git
cd sentiment-analysis-python
```
Install the required dependency:
```bash
pip install numpy
```
Ensure Python 3.6+ is installed.
## Usage
The project consists of three main classes: Corpy, TFIDF, and SentimentAnalyzer. Below is an example demonstrating how to use the SentimentAnalyzer to classify sentiments in a sample corpus.

Example:
```python
# Import the SentimentAnalyzer class (renamed as SA for convenience)
from sentimentanalyzer import SentimentAnalyzer as SA

# Define a text corpus (collection of documents) to analyze
corpus = [
    "I love programming in Python! It's so much fun and rewarding.",
    "I hate bugs in my code. They are so annoying!",
    "The weather is nice today, but I miss the sun.",
    "This is a neutral statement without strong feelings.",
]

# Create a sentiment lexicon dictionary where:
# - Positive words have scores > 0
# - Negative words have scores < 0
# - Neutral words have score = 0
lexicon = {
    "love": 2.0,
    "fun": 1.5,
    "rewarding": 1.0,
    "bugs": -1.5,
    "annoying": -1.0,
    "miss": -0.5,
    "weather": 0.0,
    "sun": 0.5,
    "strong": 1.0,
    "feelings": 0.5,
    "programming": 1.0,
    "Python": 1.0,
    "hate": -2.0,
    "nice": 0.5,
    "miss": -0.5,
    "neutral": 0.0
}

# Initialize the sentiment analyzer with corpus and lexicon
analyzer = SA(corpus, lexicon)

# Perform sentiment analysis on all documents
sentiments = analyzer.analyze()

# Print results showing document ID, sentiment score and classification
for doc, result in sentiments.items():
    print(f"Document {doc}: Score {result['score']} - Sentiment: {result['sentiment']}")

# Print summary
print("\nSummary:")
print(analyzer)
```
Output:
```
doc1: 0.2836 - positive
doc2: -0.2426 - negative
doc3: 0.0 - neutral
doc4: 0.0866 - neutral

Summary:
<SentimentAnalyzer>
- Total Documents: 4
- Positive: 1
- Negative: 1
- Neutral: 2
```
## Notes
- Lexicon: The sentiment lexicon maps words to sentiment scores (positive or negative floats). You can customize it or use external resources like VADER or SentiWordNet.
- Thresholds: Adjust threshold_positive and threshold_negative in analyzer.analyze() to fine-tune sentiment classification.
- Corpus: The Corpy class expects a list of strings, where each string is a document.

