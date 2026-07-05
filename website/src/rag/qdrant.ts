import OpenAI from 'openai';
import fetch from 'node-fetch';

const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION || 'docusaurus_docs';

if (!QDRANT_URL) {
  throw new Error('QDRANT_URL environment variable is required.');
}

if (!QDRANT_API_KEY) {
  throw new Error('QDRANT_API_KEY environment variable is required.');
}

export interface QdrantPointPayload {
  title: string;
  filePath: string;
  slug: string;
  url: string;
  sectionHeading: string;
  text: string;
  modifiedAt: string;
}

export interface QdrantSearchResult {
  id: string;
  score: number;
  payload: QdrantPointPayload;
}

async function qdrantRequest(path: string, body: unknown, method = 'POST') {
  const url = `${QDRANT_URL.replace(/\/$/, '')}${path}`;
  const response = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${QDRANT_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const payload = await response.text();
    throw new Error(`Qdrant request failed: ${response.status} ${response.statusText} - ${payload}`);
  }

  return response.json();
}

export async function ensureQdrantCollection() {
  const collectionUrl = `/collections/${QDRANT_COLLECTION}`;
  const existsResponse = await fetch(`${QDRANT_URL.replace(/\/$/, '')}${collectionUrl}`, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${QDRANT_API_KEY}`,
    },
  });

  if (existsResponse.status === 404) {
    await qdrantRequest(collectionUrl, {
      vectors: {
        size: 1536,
        distance: 'Cosine',
      },
      optimizers_config: {
        default_segment_number: 1,
      },
    });
    return;
  }

  if (!existsResponse.ok) {
    const payload = await existsResponse.text();
    throw new Error(`Qdrant collection check failed: ${existsResponse.status} ${existsResponse.statusText} - ${payload}`);
  }
}

export async function upsertQdrantPoints(points: Array<{ id: string; vector: number[]; payload: QdrantPointPayload }>) {
  if (points.length === 0) {
    return;
  }

  await qdrantRequest(`/collections/${QDRANT_COLLECTION}/points?wait=true`, {
    points,
  });
}

export async function searchQdrant(queryVector: number[], limit = 8): Promise<QdrantSearchResult[]> {
  const response = await qdrantRequest(`/collections/${QDRANT_COLLECTION}/points/search`, {
    vector: queryVector,
    limit,
    with_payload: true,
    with_vectors: false,
  });

  return (response.result || []).map((hit: any) => ({
    id: hit.id,
    score: hit.score,
    payload: hit.payload as QdrantPointPayload,
  }));
}
