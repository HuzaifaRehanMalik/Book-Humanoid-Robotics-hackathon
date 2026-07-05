import { embedTexts } from './embeddings';
import { searchQdrant, QdrantSearchResult } from './qdrant';

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
  const [embedding] = await embedTexts([query]);
  const hits: QdrantSearchResult[] = await searchQdrant(embedding.embedding, limit);

  return hits.map((hit) => ({
    id: hit.id,
    title: hit.payload.title,
    filePath: hit.payload.filePath,
    slug: hit.payload.slug,
    url: hit.payload.url,
    sectionHeading: hit.payload.sectionHeading,
    text: hit.payload.text,
    modifiedAt: hit.payload.modifiedAt,
    score: hit.score,
  }));
}
