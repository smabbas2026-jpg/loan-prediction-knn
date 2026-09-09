import { NextResponse } from 'next/server';
import { predictSingle, ApplicantInput } from '@/lib/knnEngine';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const applicants: ApplicantInput[] = body.applicants || [];
    const results = applicants.map((item) => {
      const res = predictSingle(item);
      return {
        applicant: item,
        verdict: res.verdict,
        probability: res.approval_probability,
        dti_pct: res.financials.debt_to_income_pct,
      };
    });
    return NextResponse.json({
      total_evaluated: results.length,
      results,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Batch evaluation failed' }, { status: 500 });
  }
}
