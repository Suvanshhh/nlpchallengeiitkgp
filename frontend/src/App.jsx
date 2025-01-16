import React from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import ChatInterface from "./components/ChatInterface";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/chat" element={<ChatInterface />} />
    </Routes>
  );
}

export default App;
