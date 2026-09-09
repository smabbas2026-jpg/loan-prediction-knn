import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    engine: 'Next.js 15 Serverless KNN Engine',
    default_k: 15,
  });
}
