import os
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from SentimentAnalyser import analyze_text  # Import your NLP model function

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Twitter API Authentication
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def fetch_twitter_comments(tweet_url):
    try:
        tweet_id = tweet_url.split("/")[-1]
        url = f"https://api.twitter.com/2/tweets/search/recent?query=conversation_id:{tweet_id}&tweet.fields=author_id,text"
        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        response = requests.get(url, headers=headers)

        remaining_limit = response.headers.get("x-rate-limit-remaining")
        reset_time = response.headers.get("x-rate-limit-reset")
        print(f"Twitter API Remaining Requests: {remaining_limit}, Reset Time: {reset_time}")

        if response.status_code == 200:
            tweets = response.json().get("data", [])
            return [tweet["text"] for tweet in tweets]
        elif response.status_code == 429:
            print("Rate limit exceeded. Retrying after 15 seconds...")
            time.sleep(15)
            return {"error": "Twitter API rate limit exceeded. Try again later."}
        else:
            return {"error": f"Failed to fetch comments, status: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    tweet_url = data.get("url")

    if not tweet_url:
        return jsonify({"error": "Missing tweet URL"}), 400

    comments = fetch_twitter_comments(tweet_url)
    
    if isinstance(comments, dict) and "error" in comments:
        return jsonify(comments)
    
    # Analyze each comment using your model
    analyzed_results = [analyze_text(comment) for comment in comments]

    # Count sentiment categories
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in analyzed_results:
        if result in sentiment_counts:
            sentiment_counts[result] += 1
        else:
            sentiment_counts["neutral"] += 1  # fallback
    
    total = len(analyzed_results)
    sentiment_percentages = {
        "positive": round((sentiment_counts["positive"] / total) * 100, 2) if total else 0,
        "negative": round((sentiment_counts["negative"] / total) * 100, 2) if total else 0,
        "neutral": round((sentiment_counts["neutral"] / total) * 100, 2) if total else 0,
        "comments": comments,
        "classified": analyzed_results
    }

    return jsonify(sentiment_percentages)

if __name__ == '__main__':
    app.run(debug=True)
