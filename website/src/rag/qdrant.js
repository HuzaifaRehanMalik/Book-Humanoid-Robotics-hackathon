const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION || 'docusaurus_docs';
if (!QDRANT_URL) {
    throw new Error('QDRANT_URL environment variable is required.');
}
if (!QDRANT_API_KEY) {
    throw new Error('QDRANT_API_KEY environment variable is required.');
}
function getVectorSizeForModel(model) {
    const mapping = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    };
    return mapping[model] ?? 1536;
}
async function qdrantRequest(path, body, method = 'POST') {
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
export async function ensureQdrantCollection(embeddingModel = 'text-embedding-3-small') {
    const collectionUrl = `/collections/${QDRANT_COLLECTION}`;
    const existsResponse = await fetch(`${QDRANT_URL.replace(/\/$/, '')}${collectionUrl}`, {
        headers: {
            Authorization: `Bearer ${QDRANT_API_KEY}`,
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
export async function upsertQdrantPoints(points) {
    if (points.length === 0) {
        return;
    }
    await qdrantRequest(`/collections/${QDRANT_COLLECTION}/points?wait=true`, {
        points,
    });
}
export async function searchQdrant(queryVector, limit = 8) {
    const response = await qdrantRequest(`/collections/${QDRANT_COLLECTION}/points/search`, {
        vector: queryVector,
        limit,
        with_payload: true,
        with_vectors: false,
    });
    return (response.result || []).map((hit) => ({
        id: hit.id,
        score: hit.score,
        payload: hit.payload,
    }));
}
