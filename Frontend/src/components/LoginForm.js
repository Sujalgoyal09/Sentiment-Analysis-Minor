import React, { useState } from 'react'
import { Link } from 'react-router-dom';
// import {AiOutLineEye, AiOutLineEyeInvisible} from "react-icons/ai";


 function LoginForm({setIsLoggedIn}) {

    const[formData, setFormData] = useState({ email:"", password:""});

    function changeHandler(event){
    
        setFormData((prevData) => ( 
            {
                ...{prevData},
                [event.target.name] : event.target.value
            }
        ))
    }
    
     const[showPassword, setShowPassword] = useState(false);
  
    
  return (
 <div className="flex justify-center items-center min-h-screen bg-gray-100">
        <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-md">

        <form className="space-y-4">

            <label htmlFor="email" className="block text-gray-700 font-semibold">
                <p>Email Address<sup className="text-red-500">*</sup></p>
            </label>

            <input 
            className="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
            type = "email"
            name = "email"
            value = {formData.email}
            onchange = {changeHandler}
            placeholder = "enter email id"
            />

            <label className="block text-gray-700 font-semibold">
                <p>Password<sup className="text-red-500">*</sup></p>
            </label>

            <div className="relative">
                <input className="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
                type = {showPassword ? ("text") : ("password")}
                name = "password"
                value = {formData.showPassword}
                onchange = {changeHandler}
                placeholder = "enter password"
                />

                <span onClick={() => setShowPassword((prev) => !prev)}
                    className="absolute right-3 top-3 cursor-pointer text-gray-500"
                > 
                {showPassword ? "👁️" : "🙈"}
                </span>
            </div>

            <Link to="#" className="text-blue-500 text-sm hover:underline">
                    Forgot Password?
            </Link>
            
            <button className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition duration-300">
                   Sign In
            </button>

        </form>

      </div>
    </div>
  )
}

export default LoginForm