import React from 'react'
import Template from './Template'

const Login = ({SetIsLoggedIn}) => {
  return (
   <Template
   title = "Welcome Back"
    des1 = "Securely log in to access sentiment analysis tools and gain insights from text-based data."
    des2 = "Analysis people emotions "
    formtype = "login"
    setISLoggedIn = {SetIsLoggedIn}

   />
    
  )
}

export default Login