import React from "react"
import { Link , NavLink} from "react-router-dom";
import { useState } from "react";
import { toast } from "react-hot-toast";

const Navbar = (props) => {

    let isLoggedIn = props.isLoggedIn;
    let setIsLoggedIn = props.setIsLoggedIn;

    const handleLogout = () => {
        setIsLoggedIn(false); 
        toast.success("Logged Out");
      };

   return ( 
   
   <nav className="bg-blue-600 text-white p-4 shadow-md flex justify-between items-center">

        {/* <nav> */}
        <h1 className="text-2xl font-bold">Sentiment Analysis</h1>

        <ul className="flex gap-6 text-lg">
        <li>
          <NavLink to="/" className="hover:text-gray-300">Home</NavLink>
        </li>
        <li>
          <NavLink to="/about" className="hover:text-gray-300">About</NavLink>
        </li>
        <li>
          <NavLink to="/contacts" className="hover:text-gray-300">Contacts</NavLink>
        </li>
      </ul>

        <div className='flex gap-4'>
            { !isLoggedIn && <Link to = "/login">
             <button className="bg-green-500 px-4 py-2 rounded-lg hover:bg-green-700">
                Login </button> </Link> }

            { !isLoggedIn && <Link to = "/signup"> 
            <button className="bg-yellow-500 px-4 py-2 rounded-lg hover:bg-yellow-700">
                Signup</button> </Link> }

            { isLoggedIn && <Link to = "/logout"> 
            <button onClick={handleLogout} className="bg-red-500 px-4 py-2 rounded-lg hover:bg-red-700">
                Logout</button> </Link> }

            { isLoggedIn && <Link to = "/dashboard"> 
            <button className="bg-purple-500 px-4 py-2 rounded-lg hover:bg-purple-700">Dashboard</button> </Link> }
        </div>
        
    </nav>
   ) 
};

export default Navbar;