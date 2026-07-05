import { readFile } from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const FRONTMATTER_REGEX = /^---\s*[\s\S]*?---\s*/;

export interface DocumentChunk {
  id: string;
  title: string;
  filePath: string;
  slug: string;
  url: string;
  sectionHeading: string;
  text: string;
  chunkIndex: number;
  modifiedAt: string;
}

function removeFrontmatter(raw: string): string {
  return raw.replace(FRONTMATTER_REGEX, '').trim();
}

function extractTitle(raw: string, filePath: string): string {
  const cleaned = removeFrontmatter(raw);
  const titleMatch = cleaned.match(/^#\s+(.+)$/m);
  if (titleMatch) {
    return titleMatch[1].trim();
  }

  return path.basename(filePath).replace(/\.mdx?$/i, '');
}

function normalizeSlug(filePath: string, docsRoot: string): string {
  const relativePath = path.relative(docsRoot, filePath).replace(/\\/g, '/');
  let slug = '/' + relativePath.replace(/\.mdx?$/i, '');
  if (slug.endsWith('/index')) {
    slug = slug.slice(0, -'/index'.length) || '/';
  }
  return slug;
}

function extractSections(raw: string, title: string): Array<{ sectionHeading: string; text: string }> {
  const cleaned = removeFrontmatter(raw);
  const lines = cleaned.split(/\r?\n/);
  const sections: Array<{ sectionHeading: string; text: string }> = [];
  let currentHeading = title;
  let currentLines: string[] = [];

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      if (currentLines.length > 0) {
        sections.push({
          sectionHeading: currentHeading,
          text: currentLines.join('\n').trim(),
        });
      }

      currentHeading = headingMatch[2].trim();
      currentLines = [];
      continue;
    }

    currentLines.push(line);
  }

  if (currentLines.length > 0) {
    sections.push({
      sectionHeading: currentHeading,
      text: currentLines.join('\n').trim(),
    });
  }

  return sections.filter((section) => section.text.length > 0);
}

function splitTextIntoChunks(text: string, maxWords = 850, overlapWords = 160): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) {
    return [text.trim()];
  }

  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + maxWords, words.length);
    const slice = words.slice(start, end).join(' ');
    chunks.push(slice.trim());
    if (end === words.length) {
      break;
    }
    start += Math.max(maxWords - overlapWords, 1);
  }

  return chunks;
}

function buildChunkId(slug: string, sectionHeading: string, index: number): string {
  const safeSection = sectionHeading.replace(/[^a-zA-Z0-9-_]/g, '-').slice(0, 64);
  const hash = crypto.createHash('sha256').update(`${slug}:${sectionHeading}:${index}`).digest('hex').slice(0, 10);
  return `${slug}#${safeSection}-${index}-${hash}`;
}

export async function collectMarkdownFiles(rootDir: string): Promise<string[]> {
  const entries = await import('node:fs/promises').then((fs) => fs.readdir(rootDir, { withFileTypes: true }));
  const paths: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.startsWith('.') || entry.name === 'node_modules' || entry.name === '__generated__') {
        continue;
      }
      paths.push(...(await collectMarkdownFiles(fullPath)));
      continue;
    }

    if (/\.(md|mdx)$/i.test(entry.name)) {
      paths.push(fullPath);
    }
  }

  return paths;
}

export async function loadDocumentChunks(filePath: string, docsRoot: string): Promise<DocumentChunk[]> {
  const raw = await readFile(filePath, 'utf8');
  const title = extractTitle(raw, filePath);
  const slug = normalizeSlug(filePath, docsRoot);
  const sections = extractSections(raw, title);
  const stats = await import('node:fs/promises').then((fs) => fs.stat(filePath));
  const modifiedAt = stats.mtime.toISOString();

  const chunks: DocumentChunk[] = [];
  let chunkIndex = 0;

  for (const section of sections) {
    const sectionChunks = splitTextIntoChunks(section.text);
    for (const sectionText of sectionChunks) {
      chunks.push({
        id: buildChunkId(slug, section.sectionHeading, chunkIndex),
        title,
        filePath: path.relative(docsRoot, filePath).replace(/\\/g, '/'),
        slug,
        url: slug,
        sectionHeading: section.sectionHeading,
        text: sectionText,
        chunkIndex,
        modifiedAt,
      });
      chunkIndex += 1;
    }
  }

  return chunks;
}
