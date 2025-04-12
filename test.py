from sentimentanalyzer import SentimentAnalyzer as SA

corpsus = [
    "I love programming in Python! It's so much fun and rewarding.",
    "I hate bugs in my code. They are so annoying!",
    "The weather is nice today, but I miss the sun.",
    "This is a neutral statement without strong feelings.",
]

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

analyzer = SA(corpsus, lexicon)
print("----------------")
sentiments = analyzer.analyze()
for doc, result in sentiments.items():
    print(f"{doc}: {result['score']} - {result['sentiment']}")