import React from "react";

const ContactUs = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      
      {/* Contact Card */}
      <div className="bg-white shadow-lg rounded-lg p-8 max-w-lg w-full">
        <h2 className="text-3xl font-bold text-center text-blue-600 mb-6">Contact Us</h2>

        <p className="text-gray-700 text-center mb-6">
          Have questions or feedback? Reach out to us!
        </p>

        {/* Contact Form */}
        <form className="flex flex-col space-y-4">
          <input
            type="text"
            placeholder="Your Name"
            className="p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          <input
            type="email"
            placeholder="Your Email"
            className="p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          <textarea
            placeholder="Your Message"
            rows="4"
            className="p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          ></textarea>

          <button className="bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-all duration-300">
            Send Message
          </button>
        </form>

        {/* Contact Details */}
        <div className="mt-6 text-center text-gray-600">
          <p>Email: <a href="mailto:support@sentimentanalysis.com" className="text-blue-500 hover:underline">support@sentimentanalysis.com</a></p>
          <p>Phone: +1 (234) 567-890</p>
          <p>Location: New York, USA</p>
        </div>
      </div>
    </div>
  );
};

export default ContactUs;
