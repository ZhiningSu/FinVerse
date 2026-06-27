import { HashRouter as Router, Navigate, Route, Routes } from "react-router-dom";
import Home from "@/pages/Home";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/zh" replace />} />
        <Route path="/zh" element={<Home initialMarket="cn" language="zh" />} />
        <Route path="/en" element={<Home initialMarket="us" language="en" />} />
        <Route path="/other" element={<div className="text-center text-xl">Other Page - Coming Soon</div>} />
      </Routes>
    </Router>
  );
}
