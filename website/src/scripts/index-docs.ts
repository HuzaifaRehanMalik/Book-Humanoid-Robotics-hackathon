import path from 'node:path';
import { indexDocsIntoQdrant } from '../rag/indexer';

async function main() {
  const docsRoot = process.argv[2] || path.resolve(process.cwd(), 'docs');
  console.log(`Indexing docs from ${docsRoot} into Qdrant...`);

  const result = await indexDocsIntoQdrant(docsRoot);

  console.log(`Indexed ${result.indexed} chunks from ${result.files} markdown files.`);
}

main().catch((error) => {
  console.error('index-docs error:', error);
  process.exit(1);
});
