import numpy as np
from requirements.corpy import Corpy
from requirements.TFIDF import TF_IDF


class SentimentAnalyzer:
    """Lexicon-based sentiment analysis with TF-IDF."""
    def __init__(self, corpus: dict, lexicon: dict):
        """Initialize with a Corpy object and a {word: score} sentiment lexicon."""
        self.corpy = Corpy(corpus)  # Initialize Corpy with the corpus
        self.lexicon = lexicon
        self.tfidf = TF_IDF(self.corpy)  # Initialize TFIDF with the Corpy object

    def analyze(self, threshold_positive=0.1, threshold_negative=-0.1) -> dict:
        """Sentiment analysis per doc. Returns scores and labels based on thresholds."""  
        results = {}
        tfidf_matrix = self.tfidf.matrix  # TF-IDF matrix: [docs x words]

        for i in range(len(self.corpy.corpus)):
            score = 0.0
            # Calculate sentiment score for document
            for j, word in enumerate(self.corpy.vocabulary):
                if word in self.lexicon:
                    score += self.lexicon[word] * tfidf_matrix[i, j]
            
            # Classify sentiment based on score
            if score > threshold_positive:
                sentiment = "positive"
            elif score < threshold_negative:
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            results[f"doc{i+1}"] = {"score": round(score, 4), "sentiment": sentiment}
        
        return results

    def summary(self) -> dict:
        """Returns a summary of the sentiment analysis."""
        analysis = self.analyze()
        positive = sum(1 for doc in analysis.values() if doc["sentiment"] == "positive")
        negative = sum(1 for doc in analysis.values() if doc["sentiment"] == "negative")
        neutral = sum(1 for doc in analysis.values() if doc["sentiment"] == "neutral")
        
        return {
            "total_documents": len(self.corpy.corpus),
            "positive_documents": positive,
            "negative_documents": negative,
            "neutral_documents": neutral,
            "average_score": round(sum(doc["score"] for doc in analysis.values()) / len(analysis), 4)
        }

    def __str__(self) -> str:
        """Return a readable string summary of the SentimentAnalyzer object."""
        stats = self.summary()
        return (
            f"<SentimentAnalyzer>\n"
            f"- Total Documents: {stats['total_documents']}\n"
            f"- Positive: {stats['positive_documents']}\n"
            f"- Negative: {stats['negative_documents']}\n"
            f"- Neutral: {stats['neutral_documents']}\n"
            f"- Average Sentiment Score: {stats['average_score']:.4f}"
        )

    __repr__ = __str__