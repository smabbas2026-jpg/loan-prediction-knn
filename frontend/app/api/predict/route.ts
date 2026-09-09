import { NextResponse } from 'next/server';
import { predictSingle } from '@/lib/knnEngine';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const result = predictSingle(body);
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Prediction failed' }, { status: 500 });
  }
}
