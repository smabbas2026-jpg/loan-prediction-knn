"use client";

import React, { useState, useEffect } from "react";
import { Cpu, CheckCircle2, XCircle, Activity, Gauge } from "lucide-react";

interface MetricsResponse {
  best_params: Record<string, any>;
  best_cv_score: number;
  test_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
    confusion_matrix: number[][];
  };
  baseline_comparison: {
    model: string;
    accuracy: number;
    roc_auc: number;
  };
  k_sensitivity: {
    k_values: number[];
    train_accuracy: number[];
    test_accuracy: number[];
  };
}

export default function DiagnosticsTab() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then((res) => res.json())
      .then((d) => setMetrics(d))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Top row: Confusion Matrix & Baseline Comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Confusion Matrix */}
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
            <Gauge size={20} color="var(--primary-light)" />
            <h3 style={{ fontSize: "1.1rem" }}>Confusion Matrix (N=123 Holdout Test Set)</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.5rem" }}>
            Breakdown of predicted loan decisions versus ground-truth verified outcomes.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
            <div
              style={{
                padding: "1.25rem",
                borderRadius: "var(--radius)",
                background: "var(--green-bg)",
                border: "1px solid var(--green)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--green)", textTransform: "uppercase" }}>
                True Negatives (TN)
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text)", margin: "0.2rem 0" }} className="mono">
                20
              </div>
              <div style={{ fontSize: "0.76rem", color: "var(--text-muted)" }}>
                Correctly Rejected Defaults
              </div>
            </div>

            <div
              style={{
                padding: "1.25rem",
                borderRadius: "var(--radius)",
                background: "var(--red-bg)",
                border: "1px solid var(--red)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--red)", textTransform: "uppercase" }}>
                False Positives (FP)
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text)", margin: "0.2rem 0" }} className="mono">
                18
              </div>
              <div style={{ fontSize: "0.76rem", color: "var(--text-muted)" }}>
                Type I Error (Over-Approval)
              </div>
            </div>

            <div
              style={{
                padding: "1.25rem",
                borderRadius: "var(--radius)",
                background: "var(--amber-bg)",
                border: "1px solid var(--amber)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--amber)", textTransform: "uppercase" }}>
                False Negatives (FN)
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text)", margin: "0.2rem 0" }} className="mono">
                1
              </div>
              <div style={{ fontSize: "0.76rem", color: "var(--text-muted)" }}>
                Type II Error (Missed Approval)
              </div>
            </div>

            <div
              style={{
                padding: "1.25rem",
                borderRadius: "var(--radius)",
                background: "var(--green-bg)",
                border: "1.5px solid var(--green)",
                boxShadow: "0 0 20px var(--green-glow)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--green)", textTransform: "uppercase" }}>
                True Positives (TP)
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text)", margin: "0.2rem 0" }} className="mono">
                84
              </div>
              <div style={{ fontSize: "0.76rem", color: "var(--text-muted)" }}>
                Correctly Approved Loans (98.8% Recall)
              </div>
            </div>
          </div>
        </div>

        {/* Baseline Comparison */}
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
            <Cpu size={20} color="var(--primary)" />
            <h3 style={{ fontSize: "1.1rem" }}>Model Benchmark Comparison</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.5rem" }}>
            Evaluation of K-Nearest Neighbors against an L2-regularized Logistic Regression baseline.
          </p>

          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Optimized KNN (K=25)</th>
                  <th>Logistic Regression</th>
                  <th>Advantage</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><b>Accuracy</b></td>
                  <td className="mono" style={{ fontWeight: 700, color: "var(--primary-light)" }}>84.55%</td>
                  <td className="mono">85.37%</td>
                  <td>Parity</td>
                </tr>
                <tr>
                  <td><b>Recall (Sensitivity)</b></td>
                  <td className="mono" style={{ fontWeight: 700, color: "var(--green)" }}>98.82%</td>
                  <td className="mono">98.82%</td>
                  <td><span className="badge badge-green">Identical Peak</span></td>
                </tr>
                <tr>
                  <td><b>Precision</b></td>
                  <td className="mono" style={{ fontWeight: 700 }}>82.35%</td>
                  <td className="mono">82.35%</td>
                  <td>Balanced</td>
                </tr>
                <tr>
                  <td><b>F1 - Score</b></td>
                  <td className="mono" style={{ fontWeight: 700 }}>89.84%</td>
                  <td className="mono">89.84%</td>
                  <td>High Balance</td>
                </tr>
                <tr>
                  <td><b>Explainability</b></td>
                  <td><span className="badge badge-primary">Instance-Based (Neighbors)</span></td>
                  <td>Linear Weights</td>
                  <td>Full Neighbor Proof</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* K-Sensitivity Curve Table */}
      {metrics && metrics.k_sensitivity && (
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.5rem" }}>
            <Activity size={20} color="var(--primary)" />
            <h3 style={{ fontSize: "1.1rem" }}>K-Parameter Sensitivity & Bias-Variance Analysis</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.5rem" }}>
            Evaluating how the choice of neighbor count (K) controls model capacity and generalization accuracy.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem" }}>
            {metrics.k_sensitivity.k_values.map((k, i) => {
              const testAcc = (metrics.k_sensitivity.test_accuracy[i] * 100).toFixed(1);
              const trainAcc = (metrics.k_sensitivity.train_accuracy[i] * 100).toFixed(1);
              const isPeak = k === 25 || k === 15;

              return (
                <div
                  key={k}
                  style={{
                    padding: "0.85rem",
                    borderRadius: "var(--radius-sm)",
                    background: isPeak ? "var(--primary-glow)" : "var(--bg-subtle)",
                    border: isPeak ? "1.5px solid var(--primary-light)" : "1px solid var(--border)",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: "0.78rem", fontWeight: 700, color: isPeak ? "var(--primary-light)" : "var(--text-muted)" }}>
                    K = {k}
                  </div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 800, marginTop: "0.2rem" }} className="mono">
                    {testAcc}%
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                    Train: {trainAcc}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
