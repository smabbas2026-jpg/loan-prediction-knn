"use client";

import React, { useState, useEffect } from "react";
import { Database, Search, CheckCircle, XCircle, MapPin, CreditCard } from "lucide-react";

interface DatasetResponse {
  total_records: number;
  approved_total: number;
  rejected_total: number;
  overall_approval_rate: number;
  records: Record<string, any>[];
  aggregates: {
    credit_history: Record<string, Record<string, number>>;
    property_area: Record<string, Record<string, number>>;
  };
}

export default function EdaTab() {
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dataset?limit=100")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const filteredRecords = (data?.records || []).filter((r) => {
    if (!searchTerm) return true;
    const term = searchTerm.toLowerCase();
    return (
      String(r.Loan_ID || "").toLowerCase().includes(term) ||
      String(r.Property_Area || "").toLowerCase().includes(term) ||
      String(r.Education || "").toLowerCase().includes(term) ||
      String(r.Loan_Status || "").toLowerCase().includes(term)
    );
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* High level stats strip */}
      {data && (
        <div className="metrics-grid">
          <div className="glass-card" style={{ padding: "1.25rem" }}>
            <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Total Historical Applications
            </div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, marginTop: "0.25rem" }} className="mono">
              {data.total_records}
            </div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              Kaggle Loan Problem Dataset
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.25rem" }}>
            <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Approved Loans (Y)
            </div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--green)", marginTop: "0.25rem" }} className="mono">
              {data.approved_total} ({data.overall_approval_rate}%)
            </div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              Eligible borrower decisions
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.25rem" }}>
            <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Rejected Loans (N)
            </div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--red)", marginTop: "0.25rem" }} className="mono">
              {data.rejected_total} ({(100 - data.overall_approval_rate).toFixed(1)}%)
            </div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              High-risk or guideline disqualifications
            </div>
          </div>
        </div>
      )}

      {/* Visual EDA Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Credit History Card */}
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
            <CreditCard size={20} color="var(--primary-light)" />
            <h3 style={{ fontSize: "1.1rem" }}>Credit History Impact</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.5rem" }}>
            Credit history is by far the single most decisive factor for loan approval in both real-world banking and this dataset.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
                <span><b>Meets Guidelines (1.0)</b></span>
                <span className="mono" style={{ color: "var(--green)", fontWeight: 700 }}>79.6% Approval Rate</span>
              </div>
              <div style={{ height: "12px", background: "var(--bg-subtle)", borderRadius: "6px", overflow: "hidden" }}>
                <div style={{ width: "79.6%", height: "100%", background: "linear-gradient(90deg, #10b981, #059669)" }} />
              </div>
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
                <span><b>Past Delinquency (0.0)</b></span>
                <span className="mono" style={{ color: "var(--red)", fontWeight: 700 }}>8.3% Approval Rate</span>
              </div>
              <div style={{ height: "12px", background: "var(--bg-subtle)", borderRadius: "6px", overflow: "hidden" }}>
                <div style={{ width: "8.3%", height: "100%", background: "linear-gradient(90deg, #ef4444, #b91c1c)" }} />
              </div>
            </div>
          </div>
        </div>

        {/* Property Area Card */}
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
            <MapPin size={20} color="var(--purple)" />
            <h3 style={{ fontSize: "1.1rem" }}>Property Area Success Rates</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.5rem" }}>
            Distribution of loan approval probabilities across residential property locations.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
                <span><b>Semiurban</b></span>
                <span className="mono" style={{ color: "var(--green)", fontWeight: 700 }}>76.8% Approval Rate</span>
              </div>
              <div style={{ height: "12px", background: "var(--bg-subtle)", borderRadius: "6px", overflow: "hidden" }}>
                <div style={{ width: "76.8%", height: "100%", background: "linear-gradient(90deg, #2563eb, #3b82f6)" }} />
              </div>
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
                <span><b>Urban</b></span>
                <span className="mono" style={{ color: "var(--text-muted)", fontWeight: 700 }}>65.8% Approval Rate</span>
              </div>
              <div style={{ height: "12px", background: "var(--bg-subtle)", borderRadius: "6px", overflow: "hidden" }}>
                <div style={{ width: "65.8%", height: "100%", background: "linear-gradient(90deg, #8b5cf6, #a78bfa)" }} />
              </div>
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
                <span><b>Rural</b></span>
                <span className="mono" style={{ color: "var(--amber)", fontWeight: 700 }}>61.5% Approval Rate</span>
              </div>
              <div style={{ height: "12px", background: "var(--bg-subtle)", borderRadius: "6px", overflow: "hidden" }}>
                <div style={{ width: "61.5%", height: "100%", background: "linear-gradient(90deg, #f59e0b, #fbbf24)" }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Explorer Table */}
      <div className="glass-card" style={{ padding: "1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <Database size={20} color="var(--primary)" />
            <h3 style={{ fontSize: "1.1rem" }}>Dataset Records Explorer</h3>
          </div>

          <div style={{ position: "relative", minWidth: "260px" }}>
            <input
              type="text"
              placeholder="Search by ID, Area, Education..."
              className="field-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ paddingLeft: "2.2rem" }}
            />
            <Search size={16} style={{ position: "absolute", left: "0.8rem", top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)" }} />
          </div>
        </div>

        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Loan ID</th>
                <th>Gender</th>
                <th>Married</th>
                <th>Dependents</th>
                <th>Education</th>
                <th>Self Employed</th>
                <th>Applicant Income</th>
                <th>Coapplicant Income</th>
                <th>Loan Amount</th>
                <th>Term</th>
                <th>Credit History</th>
                <th>Property Area</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.slice(0, 50).map((r, i) => (
                <tr key={i}>
                  <td className="mono" style={{ fontWeight: 600 }}>{r.Loan_ID}</td>
                  <td>{r.Gender}</td>
                  <td>{r.Married}</td>
                  <td>{r.Dependents}</td>
                  <td>{r.Education}</td>
                  <td>{r.Self_Employed}</td>
                  <td className="mono">${Number(r.ApplicantIncome || 0).toLocaleString()}</td>
                  <td className="mono">${Number(r.CoapplicantIncome || 0).toLocaleString()}</td>
                  <td className="mono">{r.LoanAmount !== "N/A" ? `$${r.LoanAmount}k` : "N/A"}</td>
                  <td className="mono">{r.Loan_Amount_Term}m</td>
                  <td className="mono">
                    {r.Credit_History === 1 ? (
                      <span style={{ color: "var(--green)" }}>1.0</span>
                    ) : r.Credit_History === 0 ? (
                      <span style={{ color: "var(--red)" }}>0.0</span>
                    ) : (
                      "N/A"
                    )}
                  </td>
                  <td>{r.Property_Area}</td>
                  <td>
                    <span className={`badge ${r.Loan_Status === "Y" ? "badge-green" : "badge-red"}`}>
                      {r.Loan_Status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
