import { HashRouter as Router, Route, Routes } from "react-router-dom";
import Home from "@/pages/Home";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/zh" element={<Home initialLanguage="zh" />} />
        <Route path="/en" element={<Home initialLanguage="en" />} />
        <Route path="/cn" element={<Home initialMarket="cn" />} />
        <Route path="/us" element={<Home initialMarket="us" />} />
        <Route path="/other" element={<div className="text-center text-xl">Other Page - Coming Soon</div>} />
      </Routes>
    </Router>
  );
}
