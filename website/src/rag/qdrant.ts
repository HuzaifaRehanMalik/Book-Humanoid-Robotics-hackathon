function getQdrantConfig() {
  const url = process.env.QDRANT_URL;
  const apiKey = process.env.QDRANT_API_KEY;
  if (!url) {
    throw new Error('Qdrant is not configured. Set QDRANT_URL to use vector search.');
  }
  return { url: url.replace(/\/$/, ''), apiKey, collection: process.env.QDRANT_COLLECTION || 'docusaurus_docs' };
}

export function isQdrantConfigured(): boolean {
  return Boolean(process.env.QDRANT_URL);
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

function getVectorSizeForModel(model: string): number {
  const mapping: Record<string, number> = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
  };
  return mapping[model] ?? 1536;
}

async function qdrantRequest(path: string, body: unknown, method = 'POST') {
  const config = getQdrantConfig();
  const url = `${config.url}${path}`;
  const response = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const payload = await response.text();
    throw new Error(`Qdrant request failed: ${response.status} ${response.statusText} - ${payload}`);
  }

  return response.json();
}

export async function ensureQdrantCollection(embeddingModel = 'text-embedding-3-small') {
  const config = getQdrantConfig();
  const collectionUrl = `/collections/${config.collection}`;
  const existsResponse = await fetch(`${config.url}${collectionUrl}`, {
    headers: {
      ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
    },
  });

  if (existsResponse.status === 404) {
    await qdrantRequest(collectionUrl, {
      vectors: {
        size: getVectorSizeForModel(embeddingModel),
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

  const { collection } = getQdrantConfig();
  await qdrantRequest(`/collections/${collection}/points?wait=true`, {
    points,
  });
}

export async function searchQdrant(queryVector: number[], limit = 8): Promise<QdrantSearchResult[]> {
  const { collection } = getQdrantConfig();
  const response = await qdrantRequest(`/collections/${collection}/points/search`, {
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
