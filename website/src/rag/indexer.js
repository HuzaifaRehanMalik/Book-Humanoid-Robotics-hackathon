import path from 'node:path';
import { collectMarkdownFiles, loadDocumentChunks } from './chunker';
import { embedTexts, getOpenAIEmbeddingModel } from './embeddings';
import { ensureQdrantCollection, upsertQdrantPoints } from './qdrant';
const DEFAULT_CHUNK_BATCH_SIZE = 32;
export async function indexDocsIntoQdrant(docsRoot) {
    const absoluteDocsRoot = path.resolve(docsRoot);
    await ensureQdrantCollection(getOpenAIEmbeddingModel());
    const markdownFiles = await collectMarkdownFiles(absoluteDocsRoot);
    const chunksByFile = await Promise.all(markdownFiles.map((filePath) => loadDocumentChunks(filePath, absoluteDocsRoot)));
    const chunks = chunksByFile.flat();
    const uniqueChunks = Array.from(new Map(chunks.map((chunk) => [chunk.id, chunk])).values());
    if (uniqueChunks.length === 0) {
        return { indexed: 0 };
    }
    for (let start = 0; start < uniqueChunks.length; start += DEFAULT_CHUNK_BATCH_SIZE) {
        const batch = uniqueChunks.slice(start, start + DEFAULT_CHUNK_BATCH_SIZE);
        const texts = batch.map((chunk) => chunk.text);
        const embeddings = await embedTexts(texts);
        const points = batch.map((chunk, index) => ({
            id: chunk.id,
            vector: embeddings[index].embedding,
            payload: {
                title: chunk.title,
                filePath: chunk.filePath,
                slug: chunk.slug,
                url: chunk.url,
                sectionHeading: chunk.sectionHeading,
                text: chunk.text,
                modifiedAt: chunk.modifiedAt,
            },
        }));
        await upsertQdrantPoints(points);
    }
    return { indexed: uniqueChunks.length, files: markdownFiles.length };
}
