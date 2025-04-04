import React from 'react';
import Navbar from './components/Navbar';
import Home from './components/Home';
import About from './components/About';
import Contacts from './components/Contacts';
import NotFound from './components/NotFound';
import Login from './components/Login';
import Signup from './components/Signup';
import Dashboard from './components/Dashboard';
import { Routes, Route } from 'react-router-dom';
import MainHeader from './components/MainHeader';
import { useState } from 'react';

function App() {

  const[isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <div>
      <Navbar isLoggedIn = {isLoggedIn} setIsLoggedIn = {setIsLoggedIn} />
      <MainHeader />
      <Routes>
        {/* <Route  path="/" element = {<MainHeader/>}> */}
        <Route path="/" element={<Home />} />
        <Route  index element = {<Home/>}/>
        <Route  path="/about" element = {<About/>}/>
        <Route  path="/contacts" element = {<Contacts/>}/>
        <Route path="*" element = {<NotFound/>} />
        
        <Route  path="/login" element = {<Login/>}/>
        <Route  path="/signup" element = {<Signup/>}/>
        <Route  path="/dashboard" element = {<Dashboard/>}/>
        {/* <Route path="/dashboard" element={isLoggedIn ? <Dashboard /> : <Login setIsLoggedIn={setIsLoggedIn} />} /> */}
        {/* </Route> */}
      </Routes>
    </div>
  );
}

export default App;
