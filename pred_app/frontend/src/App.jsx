import { useState, useEffect } from "react";
import "./App.css";
import Navbar from "./components/pages/Navbar/Navbar";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./components/pages/HomePage/Home";
import Footer from "./components/pages/Footer/Footer";
import GameCards from "./components/pages/GameCards/CardDisplay";
import ScrollToTop from "./components/other/scrollToTop";
import Table from "./components/pages/History/History";
import Payout from "./components/pages/Calculators/Payout";
import Hedge from "./components/pages/Calculators/Hedge";
import Hold from "./components/pages/Calculators/Hold";
import RegisterForm from "./components/pages/Users/RegisterForm";
import LoginForm from "./components/pages/Users/Login";
import WIP from "./components/pages/WIP/WIP";
import Future from "./components/pages/Future/Future";

function App() {
  const [user, setUser] = useState("");

  useEffect(() => {
    (async () => {
      const response = await fetch("http://54.161.55.120:8000/api/user/", {
        headers: { "Content-Type": "application/json" },
        credentials: "include",
      });
      const content = await response.json();

      if (content.detail === "User not authenticated.") {
        setUser("");
      } else {
        setUser(content.name);
      }

      console.log(user);
    })();
  });

  return (
    <Router>
      <ScrollToTop />
      <Navbar user={user} setUser={setUser} />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predictions" element={<GameCards />} />
        <Route path="/history" element={<Table />} />
        <Route path="/future" element={<Future />} />
        <Route path="/compare" element={<WIP />} />
        <Route path="/calc/payout" element={<Payout />} />
        <Route path="/calc/hedge" element={<Hedge />} />
        <Route path="/calc/hold" element={<Hold />} />
        <Route path="/calc/parlay" element={<Payout />} />
        <Route path="/register" element={<RegisterForm />} />
        <Route path="/login" element={<LoginForm setUser={setUser} />} />
        <Route path="/WIP" element={<WIP />} />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
