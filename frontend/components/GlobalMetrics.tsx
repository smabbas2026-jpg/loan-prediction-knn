"use client";

import React from "react";
import { Award, Target, Zap, BarChart2, Cpu } from "lucide-react";

export default function GlobalMetrics() {
  return (
    <div className="metrics-grid">
      <div className="metric-box">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="metric-box-label">Test Accuracy</span>
          <Award size={18} color="var(--primary-light)" />
        </div>
        <div className="metric-box-value">84.55%</div>
        <div className="metric-box-sub">Holdout test set (N=123 applicants)</div>
      </div>

      <div className="metric-box">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="metric-box-label">Approval Recall</span>
          <Target size={18} color="var(--green)" />
        </div>
        <div className="metric-box-value" style={{ color: "var(--green)" }}>
          98.82%
        </div>
        <div className="metric-box-sub">84 of 85 eligible loans detected</div>
      </div>

      <div className="metric-box">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="metric-box-label">Approval Precision</span>
          <Zap size={18} color="var(--amber)" />
        </div>
        <div className="metric-box-value">82.35%</div>
        <div className="metric-box-sub">True approvals out of positive alerts</div>
      </div>

      <div className="metric-box">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="metric-box-label">ROC - AUC</span>
          <BarChart2 size={18} color="var(--purple)" />
        </div>
        <div className="metric-box-value">81.35%</div>
        <div className="metric-box-sub">Discriminative separation power</div>
      </div>

      <div className="metric-box">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="metric-box-label">Model Tuning</span>
          <Cpu size={18} color="var(--primary)" />
        </div>
        <div className="metric-box-value" style={{ fontSize: "1.2rem", paddingTop: "0.25rem" }}>
          K=25 Euclidean
        </div>
        <div className="metric-box-sub">5-Fold Stratified CV Optimized</div>
      </div>
    </div>
  );
}
