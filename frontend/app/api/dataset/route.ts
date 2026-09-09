import { NextResponse } from 'next/server';
import { getDatasetSummary } from '@/lib/knnEngine';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '50', 10);
    const offset = parseInt(searchParams.get('offset') || '0', 10);
    const data = getDatasetSummary(limit, offset);
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Dataset fetch failed' }, { status: 500 });
  }
}
