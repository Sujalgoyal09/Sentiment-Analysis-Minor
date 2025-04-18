import React from "react";
import Carousel from "./Carousel";

const Home = () => {
  return (
    <div className="text-center py-20 bg-gradient-to-r from-blue-500 to-purple-600 text-white min-h-screen">
      <h1 className="text-4xl font-bold">Welcome to Sentiment Analysis</h1>
      <p className="text-lg mt-4">Analyze text sentiments with AI-powered insights.</p>
      <div className="p-4">
      <Carousel/>
    </div>
    </div>
  );
};

export default Home;
