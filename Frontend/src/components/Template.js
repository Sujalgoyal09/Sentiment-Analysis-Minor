import React from 'react'
import SignupForm from './SignupForm'
import LoginForm from './LoginForm'

const Template = ({title, des1, des2, formtype, setIsLoggedIn}) => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-md rounded-lg p-8 w-full max-w-md">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-4">
            {title}
        </h1>

        <p className="text-gray-600 text-center mb-6">
          <span className="font-semibold">{des1}</span> <br />
          <span className="italic">{des2}</span>
        </p>

        {formtype == "signup" ? (<SignupForm/>):(<LoginForm/>)}

        <div className="flex items-center justify-center my-4">
          <div className="w-1/3 border-t border-gray-300"></div>
          <p className="mx-2 text-gray-500">OR</p>
          <div className="w-1/3 border-t border-gray-300"></div>
        </div>
        
        <button className="w-full bg-red-500 text-white py-2 rounded-lg hover:bg-red-600 transition duration-300">
          <p className="font-semibold">Signup with Google</p>
        </button>

        </div>
    </div>
  )
}

export default Template