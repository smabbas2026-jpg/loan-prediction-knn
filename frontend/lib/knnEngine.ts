import modelData from './modelData.json';

export interface ApplicantInput {
  Gender?: string;
  Married?: string;
  Dependents?: string;
  Education?: string;
  Self_Employed?: string;
  ApplicantIncome?: number;
  CoapplicantIncome?: number;
  LoanAmount?: number;
  Loan_Amount_Term?: number;
  Credit_History?: number;
  Property_Area?: string;
  k_neighbors?: number;
}

export interface NeighborRecord {
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

export interface PredictionResult {
  prediction: number;
  verdict: 'APPROVED' | 'REJECTED';
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
  neighbors: NeighborRecord[];
}

function transformApplicant(data: ApplicantInput): number[] {
  const depStr = String(data.Dependents || '0').replace('+', '');
  const dep = parseFloat(depStr) || 0.0;

  const appInc = typeof data.ApplicantIncome === 'number' ? data.ApplicantIncome : 5000.0 * 0.7;
  const coappInc = typeof data.CoapplicantIncome === 'number' ? data.CoapplicantIncome : 0.0;
  const totalInc = Math.max(appInc + coappInc, 500.0);
  const totalIncLog = Math.log(totalInc);

  const loanAmt = typeof data.LoanAmount === 'number' ? data.LoanAmount : 128.0;
  const loanTerm = Math.max(typeof data.Loan_Amount_Term === 'number' ? data.Loan_Amount_Term : 360.0, 12.0);
  const loanRisk = Math.max(loanAmt, 60.0);

  const monthlyEmi = (loanAmt * 1000.0) / loanTerm;
  const dti = (monthlyEmi / totalInc) * 100.0;
  const debtBurden = Math.max(dti, 10.0);

  const cred = typeof data.Credit_History === 'number' ? data.Credit_History : 1.0;
  const creditScore = cred * modelData.credit_weight;

  const rawNum = [creditScore, totalIncLog, loanRisk, debtBurden, dep];
  const scaledNum = rawNum.map((val, i) => (val - modelData.scaler_means[i]) / modelData.scaler_scales[i]);

  // Categoricals
  const catFeats: number[] = [];

  // Gender: Female, Male
  const g = data.Gender || 'Male';
  catFeats.push(g === 'Female' ? 1.0 : 0.0, g === 'Male' ? 1.0 : 0.0);

  // Married: No, Yes
  const m = data.Married || 'Yes';
  catFeats.push(m === 'No' ? 1.0 : 0.0, m === 'Yes' ? 1.0 : 0.0);

  // Education: Graduate, Not Graduate
  const e = data.Education || 'Graduate';
  catFeats.push(e === 'Graduate' ? 1.0 : 0.0, e === 'Not Graduate' ? 1.0 : 0.0);

  // Self_Employed: No, Yes
  const se = data.Self_Employed || 'No';
  catFeats.push(se === 'No' ? 1.0 : 0.0, se === 'Yes' ? 1.0 : 0.0);

  // Property_Area: Rural, Semiurban, Urban
  const pa = data.Property_Area || 'Semiurban';
  catFeats.push(
    pa === 'Rural' ? 1.0 : 0.0,
    pa === 'Semiurban' ? 1.0 : 0.0,
    pa === 'Urban' ? 1.0 : 0.0
  );

  return [...scaledNum, ...catFeats];
}

export function predictSingle(data: ApplicantInput): PredictionResult {
  const k = Math.max(3, Math.min(data.k_neighbors || 15, 25));
  const vector = transformApplicant(data);

  // Euclidean distances to all training points
  const distancesWithIndex: { dist: number; idx: number; target: number }[] = [];
  const trainFeatures = modelData.train_features as number[][];
  const trainTargets = modelData.train_targets as number[];

  for (let i = 0; i < trainFeatures.length; i++) {
    const feat16 = trainFeatures[i].slice(0, 16);
    let sumSq = 0;
    for (let j = 0; j < 16; j++) {
      const diff = vector[j] - feat16[j];
      sumSq += diff * diff;
    }
    const dist = Math.sqrt(sumSq);
    distancesWithIndex.push({ dist, idx: i, target: trainTargets[i] });
  }

  distancesWithIndex.sort((a, b) => a.dist - b.dist);
  const topK = distancesWithIndex.slice(0, k);

  let approvedCount = 0;
  const neighborsList: NeighborRecord[] = [];

  const trainRaw = modelData.train_raw as Record<string, any>[];

  for (const item of topK) {
    if (item.target === 1) approvedCount++;
    const row = trainRaw[item.idx] || {};
    const simPct = Math.max(0.0, Math.round((1.0 / (1.0 + item.dist)) * 1000) / 10);

    neighborsList.push({
      index: item.idx,
      distance: Math.round(item.dist * 1000) / 1000,
      similarity_pct: simPct,
      loan_status: item.target,
      applicant_income: Number(row.ApplicantIncome || 0),
      coapplicant_income: Number(row.CoapplicantIncome || 0),
      loan_amount: row.LoanAmount != null ? Number(row.LoanAmount) : null,
      term: row.Loan_Amount_Term != null ? Number(row.Loan_Amount_Term) : 360,
      credit_history: row.Credit_History != null ? Number(row.Credit_History) : null,
      property_area: String(row.Property_Area || ''),
      education: String(row.Education || ''),
      married: String(row.Married || ''),
    });
  }

  const probApproved = Math.round((approvedCount / k) * 1000) / 10;
  const probRejected = Math.round((100.0 - probApproved) * 10) / 10;
  const prediction = probApproved >= 50.0 ? 1 : 0;

  const appInc = data.ApplicantIncome || 0;
  const coappInc = data.CoapplicantIncome || 0;
  const totalIncome = appInc + coappInc;
  const term = Math.max(data.Loan_Amount_Term || 360.0, 12.0);
  const loanAmt = data.LoanAmount || 0;
  const monthlyEmi = Math.round(((loanAmt * 1000.0) / term) * 100) / 100;
  const dti = Math.round((monthlyEmi / (totalIncome + 1e-5)) * 1000) / 10;
  const dtiCat = dti < 15 ? 'Prime' : dti < 35 ? 'Standard' : 'Elevated Leverage';

  return {
    prediction,
    verdict: prediction === 1 ? 'APPROVED' : 'REJECTED',
    approval_probability: probApproved,
    rejection_probability: probRejected,
    approved_neighbors: approvedCount,
    total_neighbors: k,
    financials: {
      total_household_income: Math.round(totalIncome * 100) / 100,
      monthly_emi: monthlyEmi,
      debt_to_income_pct: dti,
      dti_category: dtiCat,
    },
    neighbors: neighborsList,
  };
}

export function getMetrics() {
  return modelData.metrics;
}

export function getDatasetSummary(limit = 50, offset = 0) {
  const ds = modelData.dataset as Record<string, any>[];
  const total = ds.length;
  const records = ds.slice(offset, offset + limit);

  let approvedTotal = 0;
  let rejectedTotal = 0;

  const creditCounts: Record<string, { Y: number; N: number }> = {};
  const areaCounts: Record<string, { Y: number; N: number }> = {};

  for (const row of ds) {
    const status = row.Loan_Status === 'Y' ? 'Y' : 'N';
    if (status === 'Y') approvedTotal++;
    else rejectedTotal++;

    const cred = String(row.Credit_History);
    if (!creditCounts[cred]) creditCounts[cred] = { Y: 0, N: 0 };
    creditCounts[cred][status]++;

    const area = String(row.Property_Area);
    if (!areaCounts[area]) areaCounts[area] = { Y: 0, N: 0 };
    areaCounts[area][status]++;
  }

  // Normalize credit history
  const creditAgg: Record<string, Record<string, number>> = { Y: {}, N: {} };
  for (const [k, v] of Object.entries(creditCounts)) {
    const sum = v.Y + v.N || 1;
    creditAgg.Y[k] = Math.round((v.Y / sum) * 1000) / 1000;
    creditAgg.N[k] = Math.round((v.N / sum) * 1000) / 1000;
  }

  // Area counts
  const areaAgg: Record<string, Record<string, number>> = { Y: {}, N: {} };
  for (const [k, v] of Object.entries(areaCounts)) {
    areaAgg.Y[k] = v.Y;
    areaAgg.N[k] = v.N;
  }

  return {
    total_records: total,
    approved_total: approvedTotal,
    rejected_total: rejectedTotal,
    overall_approval_rate: Math.round((approvedTotal / total) * 1000) / 10,
    records,
    aggregates: {
      credit_history: creditAgg,
      property_area: areaAgg,
    },
  };
}
