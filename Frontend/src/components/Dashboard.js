import React, { useState } from "react";
import { analyzeSentiment, fetchUserTweets } from "../api"; // Correct API import

const Dashboard = () => {
    const [twitterUrl, setTwitterUrl] = useState("");
    const [twitterUserId, setTwitterUserId] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);
    const [userTweets, setUserTweets] = useState([]);
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [loadingTweets, setLoadingTweets] = useState(false);
    const [error, setError] = useState("");

    // Function to analyze sentiment of a tweet
    const handleAnalyze = async () => {
        setLoadingAnalysis(true);
        setError("");
        setAnalysisResult(null);

        try {
            const result = await analyzeSentiment(twitterUrl);
            if (result.error) {
                setError(result.error);
            } else {
                setAnalysisResult(result);
            }
        } catch (err) {
            setError("Failed to analyze sentiment.");
        }

        setLoadingAnalysis(false);
    };

    // Function to fetch a user's tweets
    const handleFetchTweets = async () => {
        setLoadingTweets(true);
        setError("");
        setUserTweets([]);

        try {
            const result = await fetchUserTweets(twitterUserId);
            if (result.error) {
                setError(result.error);
            } else {
                setUserTweets(result.tweets || []);
            }
        } catch (err) {
            setError("Failed to fetch tweets.");
        }

        setLoadingTweets(false);
    };

    return (
        <div className="container">
            <h2>Twitter Sentiment Analysis</h2>

            {/* Input for Tweet URL */}
            <input
                type="text"
                placeholder="Enter Twitter Post URL"
                value={twitterUrl}
                onChange={(e) => setTwitterUrl(e.target.value)}
            />
            <button onClick={handleAnalyze} disabled={loadingAnalysis}>
                {loadingAnalysis ? "Analyzing..." : "Analyze Sentiment"}
            </button>

            {error && <p style={{ color: "red" }}>{error}</p>}

            {/* Display Sentiment Analysis Results */}
            {analysisResult && (
                <div className="results">
                    <h3>Sentiment Analysis</h3>
                    <p>👍 Positive: {analysisResult.positive}%</p>
                    <p>😡 Negative: {analysisResult.negative}%</p>
                    <p>😐 Neutral: {analysisResult.neutral}%</p>

                    {/* Display extracted comments */}
                    <h3>Extracted Comments:</h3>
                    {Array.isArray(analysisResult.comments) ? (
                        <ul>
                            {analysisResult.comments.map((comment, index) => (
                                <li key={index}>{comment.text}</li>
                            ))}
                        </ul>
                    ) : (
                        <p>No comments found or failed to fetch.</p>
                    )}
                </div>
            )}

            <h2>Fetch User Tweets</h2>

            {/* Input for Twitter User ID */}
            <input
                type="text"
                placeholder="Enter Twitter User ID"
                value={twitterUserId}
                onChange={(e) => setTwitterUserId(e.target.value)}
            />
            <button onClick={handleFetchTweets} disabled={loadingTweets}>
                {loadingTweets ? "Fetching..." : "Fetch Tweets"}
            </button>

            {/* Display Fetched Tweets */}
            {userTweets.length > 0 && (
                <div className="tweets">
                    <h3>User Tweets:</h3>
                    <ul>
                        {userTweets.map((tweet, index) => (
                            <li key={index}>{tweet.text}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Dashboard;
