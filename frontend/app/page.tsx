"use client";

import React, { useState } from "react";
import Header from "@/components/Header";
import GlobalMetrics from "@/components/GlobalMetrics";
import SimulatorTab from "@/components/SimulatorTab";
import EdaTab from "@/components/EdaTab";
import DiagnosticsTab from "@/components/DiagnosticsTab";
import BatchTab from "@/components/BatchTab";
import { Sparkles, BarChart3, Binary, Layers } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"simulator" | "eda" | "diagnostics" | "batch">(
    "simulator"
  );

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header />

      {/* Global Performance Strip */}
      <GlobalMetrics />

      {/* Tab Navigation Pill Controller */}
      <nav className="tabs-nav">
        <button
          onClick={() => setActiveTab("simulator")}
          className={`tab-btn ${activeTab === "simulator" ? "active" : ""}`}
        >
          <Sparkles size={18} />
          <span>Loan Simulator & Neighbors</span>
        </button>

        <button
          onClick={() => setActiveTab("eda")}
          className={`tab-btn ${activeTab === "eda" ? "active" : ""}`}
        >
          <BarChart3 size={18} />
          <span>Dataset Intelligence & EDA</span>
        </button>

        <button
          onClick={() => setActiveTab("diagnostics")}
          className={`tab-btn ${activeTab === "diagnostics" ? "active" : ""}`}
        >
          <Binary size={18} />
          <span>Model Diagnostics & Tuning</span>
        </button>

        <button
          onClick={() => setActiveTab("batch")}
          className={`tab-btn ${activeTab === "batch" ? "active" : ""}`}
        >
          <Layers size={18} />
          <span>Batch Loan Underwriting</span>
        </button>
      </nav>

      {/* Tab Contents */}
      <main>
        {activeTab === "simulator" && <SimulatorTab />}
        {activeTab === "eda" && <EdaTab />}
        {activeTab === "diagnostics" && <DiagnosticsTab />}
        {activeTab === "batch" && <BatchTab />}
      </main>

      {/* Footer */}
      <footer
        style={{
          marginTop: "4rem",
          paddingTop: "2rem",
          borderTop: "1px solid var(--border)",
          textAlign: "center",
          fontSize: "0.82rem",
          color: "var(--text-muted)",
        }}
      >
        <div>
          CrediFlux Decision Engine &bull; Next.js 15 &bull; Scikit-Learn KNN Classifier ($K=25$) &bull; Kaggle Loan Prediction Dataset
        </div>
      </footer>
    </div>
  );
}
