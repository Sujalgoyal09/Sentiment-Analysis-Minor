import React, { useState } from "react";
import { analyzeSentiment, fetchUserTweets } from "../api";

const Dashboard = () => {
    const [twitterUrl, setTwitterUrl] = useState("");
    const [twitterUserId, setTwitterUserId] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);
    const [userTweets, setUserTweets] = useState([]);
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [loadingTweets, setLoadingTweets] = useState(false);
    const [error, setError] = useState("");

    const handleAnalyze = async () => {
        setLoadingAnalysis(true);
        setError("");
        setAnalysisResult(null);

        try {
            const result = await analyzeSentiment(twitterUrl);
            console.log("Analysis Result:", result); // Debug
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

    const handleFetchTweets = async () => {
        setLoadingTweets(true);
        setError("");
        setUserTweets([]);

        try {
            const result = await fetchUserTweets(twitterUserId);
            console.log("User Tweets:", result); // Debug
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
        <div className="max-w-4xl mx-auto px-4 py-10">
            <h2 className="text-3xl font-bold text-center mb-8 text-blue-700">Twitter Sentiment Analysis</h2>

            <div className="bg-white shadow-md rounded-xl p-6 mb-8 space-y-4">
                <input
                    type="text"
                    placeholder="Enter Twitter Post URL"
                    value={twitterUrl}
                    onChange={(e) => setTwitterUrl(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                    onClick={handleAnalyze}
                    disabled={loadingAnalysis}
                    className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-200 disabled:opacity-50"
                >
                    {loadingAnalysis ? "Analyzing..." : "Analyze Sentiment"}
                </button>
            </div>

            {error && (
                <div className="bg-red-100 text-red-700 p-4 rounded-lg mb-6 text-center font-medium">
                    {error}
                </div>
            )}

            {analysisResult && (
                <div className="bg-green-50 border border-green-200 rounded-xl p-6 mb-10">
                    <h3 className="text-2xl font-semibold text-green-700 mb-4">Sentiment Analysis Result</h3>
                    <div className="space-y-2">
                        <p>👍 <strong>Positive:</strong> {analysisResult.positive}%</p>
                        <p>😡 <strong>Negative:</strong> {analysisResult.negative}%</p>
                        <p>😐 <strong>Neutral:</strong> {analysisResult.neutral}%</p>
                    </div>

                    <div className="mt-6">
                        <h4 className="text-xl font-semibold mb-2">Extracted Comments</h4>
                        {Array.isArray(analysisResult.comments) && analysisResult.comments.length > 0 ? (
                            <ul className="list-disc list-inside space-y-1">
                                {analysisResult.comments.map((comment, index) => (
                                    <li key={index}>
                                        {typeof comment === "string" ? comment : comment?.text || "Invalid comment format"}
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-gray-600">No comments found or failed to fetch.</p>
                        )}
                    </div>
                </div>
            )}

            <h2 className="text-2xl font-bold text-blue-700 mb-4">Fetch User Tweets</h2>

            <div className="bg-white shadow-md rounded-xl p-6 space-y-4">
                <input
                    type="text"
                    placeholder="Enter Twitter User ID"
                    value={twitterUserId}
                    onChange={(e) => setTwitterUserId(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                    onClick={handleFetchTweets}
                    disabled={loadingTweets}
                    className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-200 disabled:opacity-50"
                >
                    {loadingTweets ? "Fetching..." : "Fetch Tweets"}
                </button>
            </div>

            {userTweets.length > 0 && (
                <div className="mt-10 bg-gray-50 border border-gray-200 rounded-xl p-6">
                    <h3 className="text-xl font-semibold mb-4 text-gray-800">User Tweets</h3>
                    <ul className="list-disc list-inside space-y-2">
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
