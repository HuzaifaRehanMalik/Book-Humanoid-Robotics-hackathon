import { embedTexts } from './embeddings';
import { searchQdrant } from './qdrant';
export async function retrieveRelevantChunks(query, limit = 8) {
    const [embedding] = await embedTexts([query]);
    const hits = await searchQdrant(embedding.embedding, limit);
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
