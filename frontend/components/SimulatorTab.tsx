"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  CheckCircle2,
  XCircle,
  TrendingUp,
  UserCheck,
  AlertTriangle,
  Flame,
  Search,
  Users,
  Percent,
} from "lucide-react";

interface Neighbor {
  index: number;
  distance: number;
  similarity_pct: number;
  loan_status: number;
  applicant_income: number;
  coapplicant_income: number;
  loan_amount: number | null;
  term: number;
  credit_history: number | null;
  property_area: string;
  education: string;
  married: string;
}

interface PredictionResponse {
  prediction: number;
  verdict: "APPROVED" | "REJECTED";
  approval_probability: number;
  rejection_probability: number;
  approved_neighbors: number;
  total_neighbors: number;
  financials: {
    total_household_income: number;
    monthly_emi: number;
    debt_to_income_pct: number;
    dti_category: string;
  };
  neighbors: Neighbor[];
}

export default function SimulatorTab() {
  // Form State
  const [gender, setGender] = useState("Male");
  const [married, setMarried] = useState("Yes");
  const [dependents, setDependents] = useState("0");
  const [education, setEducation] = useState("Graduate");
  const [selfEmployed, setSelfEmployed] = useState("No");
  const [applicantIncome, setApplicantIncome] = useState(6500);
  const [coapplicantIncome, setCoapplicantIncome] = useState(2500);
  const [loanAmount, setLoanAmount] = useState(140);
  const [loanTerm, setLoanTerm] = useState(360);
  const [creditHistory, setCreditHistory] = useState(1.0);
  const [propertyArea, setPropertyArea] = useState("Semiurban");
  const [kNeighbors, setKNeighbors] = useState(15);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  // Fetch prediction from API
  const runPrediction = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Gender: gender,
          Married: married,
          Dependents: dependents,
          Education: education,
          Self_Employed: selfEmployed,
          ApplicantIncome: Number(applicantIncome),
          CoapplicantIncome: Number(coapplicantIncome),
          LoanAmount: Number(loanAmount),
          Loan_Amount_Term: Number(loanTerm),
          Credit_History: Number(creditHistory),
          Property_Area: propertyArea,
          k_neighbors: Number(kNeighbors),
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to score application");
      }

      const data = (await res.json()) as PredictionResponse;
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [
    gender,
    married,
    dependents,
    education,
    selfEmployed,
    applicantIncome,
    coapplicantIncome,
    loanAmount,
    loanTerm,
    creditHistory,
    propertyArea,
    kNeighbors,
  ]);

  // Debounced run whenever inputs change
  useEffect(() => {
    const timer = setTimeout(() => {
      runPrediction();
    }, 250);
    return () => clearTimeout(timer);
  }, [runPrediction]);

  // Presets Handlers
  const applyStrongPreset = () => {
    setGender("Male");
    setMarried("Yes");
    setDependents("0");
    setEducation("Graduate");
    setSelfEmployed("No");
    setApplicantIncome(6500);
    setCoapplicantIncome(2500);
    setLoanAmount(140);
    setLoanTerm(360);
    setCreditHistory(1.0);
    setPropertyArea("Semiurban");
    setKNeighbors(15);
  };

  const applyBorderlinePreset = () => {
    setGender("Female");
    setMarried("No");
    setDependents("2");
    setEducation("Not Graduate");
    setSelfEmployed("Yes");
    setApplicantIncome(2600);
    setCoapplicantIncome(0);
    setLoanAmount(150);
    setLoanTerm(360);
    setCreditHistory(1.0);
    setPropertyArea("Rural");
    setKNeighbors(15);
  };

  const applyHighRiskPreset = () => {
    setGender("Male");
    setMarried("No");
    setDependents("2");
    setEducation("Not Graduate");
    setSelfEmployed("No");
    setApplicantIncome(2200);
    setCoapplicantIncome(0);
    setLoanAmount(180);
    setLoanTerm(180);
    setCreditHistory(0.0);
    setPropertyArea("Urban");
    setKNeighbors(15);
  };

  return (
    <div>
      {/* Presets Strip */}
      <div className="presets-strip">
        <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
          Quick Scenarios:
        </span>
        <button onClick={applyStrongPreset} className="preset-chip strong">
          <UserCheck size={16} />
          <span>🟢 Strong Applicant (High Income, Low DTI)</span>
        </button>
        <button onClick={applyBorderlinePreset} className="preset-chip borderline">
          <AlertTriangle size={16} />
          <span>🟡 Borderline Applicant (Tight Cashflow, High Debt)</span>
        </button>
        <button onClick={applyHighRiskPreset} className="preset-chip highrisk">
          <Flame size={16} />
          <span>🔴 High-Risk Applicant (Credit Delinquency)</span>
        </button>
      </div>

      <div className="form-grid">
        {/* Input Form Column */}
        <div className="glass-card form-section">
          <div className="section-title">
            <TrendingUp size={20} color="var(--primary)" />
            <span>Applicant Parameters</span>
          </div>

          <div className="row-3">
            <div className="field-group">
              <label className="field-label">Gender</label>
              <select className="field-select" value={gender} onChange={(e) => setGender(e.target.value)}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Married</label>
              <select className="field-select" value={married} onChange={(e) => setMarried(e.target.value)}>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Dependents</label>
              <select className="field-select" value={dependents} onChange={(e) => setDependents(e.target.value)}>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3+">3+</option>
              </select>
            </div>
          </div>

          <div className="row-2">
            <div className="field-group">
              <label className="field-label">Education</label>
              <select className="field-select" value={education} onChange={(e) => setEducation(e.target.value)}>
                <option value="Graduate">Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Self Employed</label>
              <select className="field-select" value={selfEmployed} onChange={(e) => setSelfEmployed(e.target.value)}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          <div className="row-2">
            <div className="field-group">
              <label className="field-label">
                <span>Applicant Income</span>
                <span className="mono" style={{ color: "var(--primary-light)" }}>${applicantIncome.toLocaleString()} / mo</span>
              </label>
              <input
                type="range"
                className="range-slider"
                min={150}
                max={45000}
                step={250}
                value={applicantIncome}
                onChange={(e) => setApplicantIncome(Number(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label className="field-label">
                <span>Coapplicant Income</span>
                <span className="mono" style={{ color: "var(--primary-light)" }}>${coapplicantIncome.toLocaleString()} / mo</span>
              </label>
              <input
                type="range"
                className="range-slider"
                min={0}
                max={30000}
                step={250}
                value={coapplicantIncome}
                onChange={(e) => setCoapplicantIncome(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row-2">
            <div className="field-group">
              <label className="field-label">
                <span>Loan Amount</span>
                <span className="mono" style={{ color: "var(--primary-light)" }}>${loanAmount.toLocaleString()}k (${(loanAmount * 1000).toLocaleString()})</span>
              </label>
              <input
                type="range"
                className="range-slider"
                min={9}
                max={700}
                step={5}
                value={loanAmount}
                onChange={(e) => setLoanAmount(Number(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label className="field-label">
                <span>Loan Term</span>
                <span className="mono" style={{ color: "var(--primary-light)" }}>{loanTerm} Months ({loanTerm / 12} Years)</span>
              </label>
              <select className="field-select" value={loanTerm} onChange={(e) => setLoanTerm(Number(e.target.value))}>
                <option value={12}>12 Months (1 Year)</option>
                <option value={36}>36 Months (3 Years)</option>
                <option value={60}>60 Months (5 Years)</option>
                <option value={84}>84 Months (7 Years)</option>
                <option value={120}>120 Months (10 Years)</option>
                <option value={180}>180 Months (15 Years)</option>
                <option value={240}>240 Months (20 Years)</option>
                <option value={300}>300 Months (25 Years)</option>
                <option value={360}>360 Months (30 Years)</option>
                <option value={480}>480 Months (40 Years)</option>
              </select>
            </div>
          </div>

          <div className="row-2">
            <div className="field-group">
              <label className="field-label">Credit History Guidelines</label>
              <select
                className="field-select"
                value={creditHistory}
                onChange={(e) => setCreditHistory(Number(e.target.value))}
              >
                <option value={1.0}>Yes (1.0) - High Creditworthiness</option>
                <option value={0.0}>No (0.0) - Past Delinquency / Default</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Property Area</label>
              <select className="field-select" value={propertyArea} onChange={(e) => setPropertyArea(e.target.value)}>
                <option value="Semiurban">Semiurban (Highest Success Rate)</option>
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>
          </div>

          <div className="field-group" style={{ marginTop: "0.5rem" }}>
            <label className="field-label">
              <span>KNN Neighbors (K Value for Decision)</span>
              <span className="mono" style={{ color: "var(--primary-light)" }}>K = {kNeighbors}</span>
            </label>
            <input
              type="range"
              className="range-slider"
              min={3}
              max={25}
              step={2}
              value={kNeighbors}
              onChange={(e) => setKNeighbors(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Prediction Results Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {result && (
            <>
              {/* Verdict Banner */}
              <div className={`verdict-banner ${result.verdict === "APPROVED" ? "approved" : "rejected"}`}>
                <div style={{ fontSize: "0.78rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
                  Assessment Result
                </div>
                <div className={`verdict-status ${result.verdict === "APPROVED" ? "approved" : "rejected"}`}>
                  {result.verdict === "APPROVED" ? (
                    <>
                      <CheckCircle2 size={32} />
                      <span>LOAN APPROVED</span>
                    </>
                  ) : (
                    <>
                      <XCircle size={32} />
                      <span>LOAN REJECTED</span>
                    </>
                  )}
                </div>
                <div className="verdict-subtitle">
                  {result.verdict === "APPROVED"
                    ? "Low default risk profile detected by K-Nearest Neighbors"
                    : "High default risk profile detected by K-Nearest Neighbors"}
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="confidence-container">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text)" }}>
                    Approval Confidence Score
                  </span>
                  <span className={`badge ${result.verdict === "APPROVED" ? "badge-green" : "badge-red"}`}>
                    {result.approval_probability}% Probability
                  </span>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-bar-approved"
                    style={{ width: `${result.approval_probability}%` }}
                  />
                  <div
                    className="progress-bar-rejected"
                    style={{ width: `${result.rejection_probability}%` }}
                  />
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.76rem", color: "var(--text-muted)" }}>
                  <span>
                    Approved: <b>{result.approval_probability}%</b> ({result.approved_neighbors}/{result.total_neighbors} neighbors)
                  </span>
                  <span>
                    Rejected: <b>{result.rejection_probability}%</b> ({result.total_neighbors - result.approved_neighbors}/{result.total_neighbors} neighbors)
                  </span>
                </div>
              </div>

              {/* Financial Health Indicators */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.85rem" }}>
                <div className="glass-card" style={{ padding: "0.9rem 1.1rem" }}>
                  <div style={{ fontSize: "0.74rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" }}>
                    Total Household Income
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--text)", marginTop: "0.2rem" }} className="mono">
                    ${result.financials.total_household_income.toLocaleString()}
                  </div>
                </div>

                <div className="glass-card" style={{ padding: "0.9rem 1.1rem" }}>
                  <div style={{ fontSize: "0.74rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" }}>
                    Debt-to-Income (DTI)
                  </div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "0.4rem", marginTop: "0.2rem" }}>
                    <span style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--text)" }} className="mono">
                      {result.financials.debt_to_income_pct}%
                    </span>
                    <span className={`badge ${result.financials.debt_to_income_pct < 15 ? "badge-green" : result.financials.debt_to_income_pct < 35 ? "badge-amber" : "badge-red"}`}>
                      {result.financials.dti_category}
                    </span>
                  </div>
                </div>

                <div className="glass-card" style={{ padding: "0.9rem 1.1rem" }}>
                  <div style={{ fontSize: "0.74rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" }}>
                    Est. Monthly EMI
                  </div>
                  <div style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--text)", marginTop: "0.2rem" }} className="mono">
                    ${result.financials.monthly_emi.toLocaleString()}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Nearest Neighbors Table */}
      {result && result.neighbors && (
        <div style={{ marginTop: "2.5rem" }} className="glass-card">
          <div style={{ padding: "1.5rem", borderBottom: "1px solid var(--border)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                <Search size={20} color="var(--primary-light)" />
                <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>
                  KNN Neighbor Inspector (Top {result.total_neighbors} Closest Historical Applicants)
                </h3>
              </div>
              <span className="badge badge-primary">
                {result.approved_neighbors} Approved / {result.total_neighbors - result.approved_neighbors} Rejected
              </span>
            </div>
            <div style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
              These {result.total_neighbors} applicants have the lowest Euclidean distance in normalized feature space and cast the votes that determined this decision.
            </div>
          </div>

          <div className="data-table-wrapper" style={{ border: "none", borderRadius: "0 0 var(--radius) var(--radius)" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Outcome</th>
                  <th>Distance</th>
                  <th>Similarity</th>
                  <th>Applicant Income</th>
                  <th>Coapplicant Income</th>
                  <th>Loan Amount</th>
                  <th>Term</th>
                  <th>Credit History</th>
                  <th>Property Area</th>
                  <th>Education</th>
                </tr>
              </thead>
              <tbody>
                {result.neighbors.map((n, i) => (
                  <tr key={n.index}>
                    <td className="mono" style={{ fontWeight: 700 }}>#{i + 1}</td>
                    <td>
                      <span className={`badge ${n.loan_status === 1 ? "badge-green" : "badge-red"}`}>
                        {n.loan_status === 1 ? "Approved (Y)" : "Rejected (N)"}
                      </span>
                    </td>
                    <td className="mono">{n.distance.toFixed(3)}</td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ width: "40px", height: "6px", background: "var(--bg-subtle)", borderRadius: "3px", overflow: "hidden" }}>
                          <div style={{ width: `${n.similarity_pct}%`, height: "100%", background: "var(--primary)" }} />
                        </div>
                        <span className="mono" style={{ fontSize: "0.78rem" }}>{n.similarity_pct}%</span>
                      </div>
                    </td>
                    <td className="mono">${n.applicant_income.toLocaleString()}</td>
                    <td className="mono">${n.coapplicant_income.toLocaleString()}</td>
                    <td className="mono">{n.loan_amount ? `$${n.loan_amount}k` : "N/A"}</td>
                    <td className="mono">{n.term}m</td>
                    <td className="mono">
                      {n.credit_history !== null ? (
                        n.credit_history === 1.0 ? (
                          <span style={{ color: "var(--green)" }}>1.0 (Good)</span>
                        ) : (
                          <span style={{ color: "var(--red)" }}>0.0 (Bad)</span>
                        )
                      ) : (
                        "N/A"
                      )}
                    </td>
                    <td>{n.property_area}</td>
                    <td>{n.education}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
