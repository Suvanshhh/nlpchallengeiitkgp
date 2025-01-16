import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuthState } from "react-firebase-hooks/auth";
import { auth } from "./config/firebase"; // Import Firebase config
import Home from "./components/Home";
import Login from "./components/Login";
import ChatInterface from "./components/ChatInterface";

// Create a logout function
const logout = () => {
  auth.signOut().then(() => {
    window.location.reload(); // Optionally reload the page after logout
  });
};

function App() {
  const [user, loading, error] = useAuthState(auth); // State to check user status

  // Protect routes that require login
  const ProtectedRoute = ({ element }) => {
    return user ? element : <Navigate to="/auth" />; // Redirect to login if not authenticated
  };

  return (
    <div>
      <nav>
        {/* Add logout button if user is authenticated */}
        {user ? (
          <button onClick={logout} className="logout-button">
            Logout
          </button>
        ) : (
          <button className="login-button" onClick={() => window.location.href = "/auth"}>
            Login
          </button>
        )}
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/auth" element={<Login />} />
        <Route
          path="/chat"
          element={<ProtectedRoute element={<ChatInterface />} />}
        />
      </Routes>
    </div>
  );
}

export default App;
