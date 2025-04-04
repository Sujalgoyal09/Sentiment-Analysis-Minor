from SentimentAnalyser import analyze_text  # Importing function

# Test different comments
test_comments = [
    "I love this! It's amazing! ❤️",
    "This is the worst thing ever. I hate it! 😡",
    "It's okay, not too bad, not too good.",
    "What a terrible experience. Never again!",
    "Absolutely fantastic, I’m so happy!",
]

print("Starting sentiment analysis test...\n")  # Debugging print

# Analyze each comment
for comment in test_comments:
    print(f"Analyzing comment: {comment}")  # Debugging print
    result = analyze_text(comment)
    print(f"Result: {result}\n")  # Debugging print

print("Sentiment analysis test completed.")
