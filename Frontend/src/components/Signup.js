import React from 'react'
import Template from './Template'

const Signup = ({SetIsLoggedIn}) => {
  return (
    <Template
   title = "Welcome Back"
    des1 = "Securely log in to access sentiment analysis tools and gain insights from text-based data."
    des2 = "Analysis people emotions "
    formtype = "signup"
    setIsLoggedIn = {SetIsLoggedIn}

   />
  )
}

export default Signup