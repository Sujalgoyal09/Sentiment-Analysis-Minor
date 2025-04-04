import React from "react";

const About = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-200">
      <div className="bg-white p-8 shadow-lg rounded-lg w-2/3">
        <h2 className="text-3xl font-bold text-center">About Sentiment Analysis</h2>
        <p className="text-gray-700 mt-4">
          This platform helps users analyze text data and determine the sentiment behind it—whether it's positive, negative, or neutral.
        </p>
      </div>
    </div>
  );
};

export default About;
