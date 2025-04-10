from SentimentAnalyser import analyze_text 

test_comments = [
    "I love this! It's amazing! ❤️",
    "This is the worst thing ever. I hate it! 😡",
    "It's okay, not too bad, not too good.",
    "What a terrible experience. Never again!",
    "Absolutely fantastic, I’m so happy!",
]

print("Starting sentiment analysis test...\n")  

for comment in test_comments:
    print(f"Analyzing comment: {comment}")  
    result = analyze_text(comment)
    print(f"Result: {result}\n")  

print("Sentiment analysis test completed.")
