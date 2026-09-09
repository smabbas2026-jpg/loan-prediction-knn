"use client";

import React, { useEffect, useState } from "react";
import { ShieldCheck, Moon, Sun, Activity } from "lucide-react";

export default function Header() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    // Check saved theme or default to dark
    const saved = localStorage.getItem("crediflux-theme") as "dark" | "light" | null;
    const preferred = saved || "dark";
    setTheme(preferred);
    document.documentElement.setAttribute("data-theme", preferred);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("crediflux-theme", next);
    document.documentElement.setAttribute("data-theme", next);
  };

  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon">
          <ShieldCheck size={26} strokeWidth={2.4} />
        </div>
        <div>
          <div className="brand-title">CrediFlux</div>
          <div className="brand-tagline">
            KNN Loan Decision Intelligence & Explainability Engine
          </div>
        </div>
      </div>

      <div className="header-actions">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            fontSize: "0.82rem",
            fontWeight: 600,
            padding: "0.4rem 0.8rem",
            borderRadius: "var(--radius-full)",
            background: "var(--card)",
            border: "1px solid var(--border)",
            color: "var(--green)",
          }}
        >
          <Activity size={14} className="animate-pulse" />
          <span>KNN Service Live</span>
        </div>

        <button
          onClick={toggleTheme}
          className="theme-btn"
          title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
        </button>
      </div>
    </header>
  );
}
