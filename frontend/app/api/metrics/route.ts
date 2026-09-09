import { NextResponse } from 'next/server';
import { getMetrics } from '@/lib/knnEngine';

export async function GET() {
  try {
    const data = getMetrics();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Metrics failed' }, { status: 500 });
  }
}
