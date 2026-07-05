import path from 'node:path';
import { collectMarkdownFiles, loadDocumentChunks, DocumentChunk } from './chunker';
import { embedTexts } from './embeddings';
import { ensureQdrantCollection, upsertQdrantPoints } from './qdrant';

const DEFAULT_CHUNK_BATCH_SIZE = 32;

export async function indexDocsIntoQdrant(docsRoot: string) {
  await ensureQdrantCollection();

  const markdownFiles = await collectMarkdownFiles(docsRoot);
  const chunksByFile = await Promise.all(markdownFiles.map((filePath) => loadDocumentChunks(filePath, docsRoot)));
  const chunks = chunksByFile.flat();

  if (chunks.length === 0) {
    return { indexed: 0 };
  }

  for (let start = 0; start < chunks.length; start += DEFAULT_CHUNK_BATCH_SIZE) {
    const batch = chunks.slice(start, start + DEFAULT_CHUNK_BATCH_SIZE);
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

  return { indexed: chunks.length, files: markdownFiles.length };
}
