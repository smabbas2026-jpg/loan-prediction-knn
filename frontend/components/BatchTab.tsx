"use client";

import React, { useState } from "react";
import { UploadCloud, Download, CheckCircle, AlertCircle, FileText } from "lucide-react";

interface BatchResultItem {
  applicant: Record<string, any>;
  verdict: "APPROVED" | "REJECTED";
  probability: number;
  dti_pct: number;
}

export default function BatchTab() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BatchResultItem[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);

  // Template CSV generator
  const downloadTemplate = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area\n" +
      "Male,Yes,0,Graduate,No,6500,2500,140,360,1.0,Semiurban\n" +
      "Female,No,2,Not Graduate,Yes,2600,0,150,360,1.0,Rural\n" +
      "Male,No,1,Not Graduate,No,2200,0,180,180,0.0,Urban\n" +
      "Female,Yes,1,Graduate,No,4500,1800,120,360,1.0,Urban\n" +
      "Male,Yes,3+,Graduate,Yes,8000,3200,220,360,1.0,Semiurban\n";

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "loan_applicant_batch_template.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);

    try {
      const text = await file.text();
      const lines = text.trim().split("\n");
      if (lines.length < 2) {
        alert("CSV file must have a header row and at least 1 applicant row.");
        setLoading(false);
        return;
      }

      const headers = lines[0].split(",").map((h) => h.trim());
      const applicants = [];

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const vals = line.split(",").map((v) => v.trim());
        const row: Record<string, any> = {};
        headers.forEach((h, idx) => {
          row[h] = vals[idx] || "";
        });

        applicants.push({
          Gender: row.Gender || "Male",
          Married: row.Married || "Yes",
          Dependents: row.Dependents || "0",
          Education: row.Education || "Graduate",
          Self_Employed: row.Self_Employed || "No",
          ApplicantIncome: parseFloat(row.ApplicantIncome) || 5000,
          CoapplicantIncome: parseFloat(row.CoapplicantIncome) || 0,
          LoanAmount: parseFloat(row.LoanAmount) || 120,
          Loan_Amount_Term: parseFloat(row.Loan_Amount_Term) || 360,
          Credit_History: parseFloat(row.Credit_History ?? 1.0),
          Property_Area: row.Property_Area || "Semiurban",
          k_neighbors: 15,
        });
      }

      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ applicants }),
      });

      if (!res.ok) throw new Error("Batch scoring failed");
      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      console.error(err);
      alert("Failed to process batch CSV. Check file formatting.");
    } finally {
      setLoading(false);
    }
  };

  const exportResults = () => {
    if (!results.length) return;

    const headers = [
      "Gender",
      "Married",
      "Dependents",
      "Education",
      "ApplicantIncome",
      "CoapplicantIncome",
      "LoanAmount",
      "Term",
      "Credit_History",
      "Property_Area",
      "Verdict",
      "Confidence_Probability",
      "DTI_Percent",
    ];

    const rows = results.map((r) => [
      r.applicant.Gender,
      r.applicant.Married,
      r.applicant.Dependents,
      r.applicant.Education,
      r.applicant.ApplicantIncome,
      r.applicant.CoapplicantIncome,
      r.applicant.LoanAmount,
      r.applicant.Loan_Amount_Term,
      r.applicant.Credit_History,
      r.applicant.Property_Area,
      r.verdict,
      `${r.probability}%`,
      `${r.dti_pct}%`,
    ]);

    const csvString = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `loan_predictions_evaluated_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const approvedCount = results.filter((r) => r.verdict === "APPROVED").length;
  const rejectedCount = results.length - approvedCount;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Upload Banner Card */}
      <div className="glass-card" style={{ padding: "2rem", textAlign: "center" }}>
        <div style={{ maxWidth: "540px", margin: "0 auto" }}>
          <div
            style={{
              width: "60px",
              height: "60px",
              borderRadius: "16px",
              background: "var(--primary-glow)",
              color: "var(--primary-light)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 1.25rem auto",
            }}
          >
            <UploadCloud size={32} />
          </div>
          <h3 style={{ fontSize: "1.3rem", fontWeight: 700, marginBottom: "0.5rem" }}>
            Batch Loan Underwriting & Evaluation
          </h3>
          <p style={{ fontSize: "0.86rem", color: "var(--text-muted)", marginBottom: "1.75rem" }}>
            Upload applicant lists in CSV format for automated KNN assessment, confidence scoring, and debt ratio calculation.
          </p>

          <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
            <label className="btn btn-primary" style={{ cursor: "pointer" }}>
              <FileText size={18} />
              <span>{loading ? "Processing..." : "Select CSV File to Score"}</span>
              <input type="file" accept=".csv" onChange={handleFileUpload} style={{ display: "none" }} />
            </label>

            <button onClick={downloadTemplate} className="btn btn-outline">
              <Download size={18} />
              <span>Download CSV Template</span>
            </button>
          </div>

          {fileName && (
            <div style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "1rem" }}>
              Uploaded: <span className="mono" style={{ color: "var(--text)" }}>{fileName}</span>
            </div>
          )}
        </div>
      </div>

      {/* Batch Results Summary */}
      {results.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="metrics-grid">
            <div className="glass-card" style={{ padding: "1.25rem" }}>
              <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                Total Applicants Scored
              </div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, marginTop: "0.25rem" }} className="mono">
                {results.length}
              </div>
            </div>

            <div className="glass-card" style={{ padding: "1.25rem" }}>
              <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                Approved
              </div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--green)", marginTop: "0.25rem" }} className="mono">
                {approvedCount} ({((approvedCount / results.length) * 100).toFixed(1)}%)
              </div>
            </div>

            <div className="glass-card" style={{ padding: "1.25rem" }}>
              <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                Rejected
              </div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--red)", marginTop: "0.25rem" }} className="mono">
                {rejectedCount} ({((rejectedCount / results.length) * 100).toFixed(1)}%)
              </div>
            </div>
          </div>

          {/* Results Table */}
          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Evaluated Batch Decisions</h3>
              <button onClick={exportResults} className="btn btn-primary" style={{ padding: "0.5rem 1rem", fontSize: "0.82rem" }}>
                <Download size={16} />
                <span>Export Enriched Results (.csv)</span>
              </button>
            </div>

            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                    <th>Household Income</th>
                    <th>Loan Amount</th>
                    <th>DTI Ratio</th>
                    <th>Credit History</th>
                    <th>Property Area</th>
                    <th>Education</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i}>
                      <td className="mono" style={{ fontWeight: 600 }}>#{i + 1}</td>
                      <td>
                        <span className={`badge ${r.verdict === "APPROVED" ? "badge-green" : "badge-red"}`}>
                          {r.verdict}
                        </span>
                      </td>
                      <td className="mono" style={{ fontWeight: 700 }}>{r.probability}%</td>
                      <td className="mono">${(r.applicant.ApplicantIncome + r.applicant.CoapplicantIncome).toLocaleString()}</td>
                      <td className="mono">${r.applicant.LoanAmount}k</td>
                      <td className="mono">{r.dti_pct}%</td>
                      <td className="mono">
                        {r.applicant.Credit_History === 1 ? (
                          <span style={{ color: "var(--green)" }}>1.0</span>
                        ) : (
                          <span style={{ color: "var(--red)" }}>0.0</span>
                        )}
                      </td>
                      <td>{r.applicant.Property_Area}</td>
                      <td>{r.applicant.Education}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
