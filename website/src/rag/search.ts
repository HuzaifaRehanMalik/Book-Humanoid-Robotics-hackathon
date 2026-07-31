import { embedTexts } from './embeddings';
import { isQdrantConfigured, searchQdrant, QdrantSearchResult } from './qdrant';
import { searchLocalDocs } from './local-search';

export interface SearchContextChunk {
  id: string;
  title: string;
  filePath: string;
  slug: string;
  url: string;
  sectionHeading: string;
  text: string;
  modifiedAt: string;
  score: number;
}

export async function retrieveRelevantChunks(query: string, limit = 8): Promise<SearchContextChunk[]> {
  if (!isQdrantConfigured()) {
    return searchLocalDocs(query, limit);
  }

  try {
    const [embedding] = await embedTexts([query]);
    const hits: QdrantSearchResult[] = await searchQdrant(embedding.embedding, limit);

    return hits.map((hit) => ({
      id: hit.id, title: hit.payload.title, filePath: hit.payload.filePath, slug: hit.payload.slug,
      url: hit.payload.url, sectionHeading: hit.payload.sectionHeading, text: hit.payload.text,
      modifiedAt: hit.payload.modifiedAt, score: hit.score,
    }));
  } catch (error) {
    console.warn('Vector search unavailable; using local textbook search.', error);
    return searchLocalDocs(query, limit);
  }
}
