const API_BASE_URL = "http://127.0.0.1:5000"; // Flask backend URL

// Analyze sentiment for a given tweet URL
export const analyzeSentiment = async (twitterPostUrl) => {
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: twitterPostUrl })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            console.error("Backend Error:", data.error || response.statusText);
            return { error: data.error || "Failed to analyze sentiment." };
        }

        console.log("✅ Sentiment Analysis API Response:", data);
        return {
            positive: data.positive,
            negative: data.negative,
            neutral: data.neutral,
            comments: data.comments,
            classified: data.classified
        };
    } catch (error) {
        console.error("❌ Error analyzing sentiment:", error);
        return { error: "Failed to connect to sentiment analysis service." };
    }
};

// Fetch tweets by username (if this route exists in your Flask backend)
export const fetchUserTweets = async (twitterUsername) => {
    try {
        const response = await fetch(`${API_BASE_URL}/fetch_tweets/${twitterUsername}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            console.error("Backend Error:", data.error || response.statusText);
            return { error: data.error || "Failed to fetch tweets." };
        }

        console.log("✅ Fetch Tweets API Response:", data);
        return data;
    } catch (error) {
        console.error("❌ Error fetching tweets:", error);
        return { error: "Failed to connect to tweet service." };
    }
};
