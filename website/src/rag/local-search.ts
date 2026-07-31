import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { SearchContextChunk } from './search';

const docsDirectory = join(process.cwd(), 'docs');
const stopWords = new Set(['about', 'after', 'and', 'are', 'for', 'from', 'how', 'into', 'the', 'this', 'what', 'when', 'where', 'with', 'your']);

function getMarkdownFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? getMarkdownFiles(path) : /\.mdx?$/i.test(entry.name) ? [path] : [];
  });
}

function cleanMarkdown(markdown: string): string {
  return markdown.replace(/^---[\s\S]*?---\s*/m, '').replace(/```[\s\S]*?```/g, ' ').replace(/!?(\[[^\]]*\])\([^)]*\)/g, '$1').replace(/[#>*_`|]/g, ' ').replace(/\s+/g, ' ').trim();
}

function titleFor(markdown: string, filePath: string): string {
  return markdown.match(/^#\s+(.+)$/m)?.[1]?.trim() || filePath.replace(/\\/g, '/').replace(/\.mdx?$/i, '').split('/').pop()!.replace(/[-_]/g, ' ');
}

function termsFor(query: string): string[] {
  return query.toLowerCase().match(/[a-z0-9][a-z0-9-]{1,}/g)?.filter((term) => !stopWords.has(term)) || [];
}

export function searchLocalDocs(query: string, limit = 8): SearchContextChunk[] {
  if (!existsSync(docsDirectory)) return [];
  const terms = termsFor(query);
  return getMarkdownFiles(docsDirectory).map((filePath) => {
    const markdown = readFileSync(filePath, 'utf8');
    const text = cleanMarkdown(markdown);
    const lowerText = text.toLowerCase();
    const score = terms.reduce((total, term) => total + (lowerText.match(new RegExp(`\\b${term.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}\\b`, 'g'))?.length || 0), 0);
    const relativePath = relative(docsDirectory, filePath).replace(/\\/g, '/');
    const slug = `/${relativePath.replace(/\.mdx?$/i, '').replace(/\/index$/i, '')}`;
    return { id: createHash('sha1').update(relativePath).digest('hex'), title: titleFor(markdown, relativePath), filePath: relativePath, slug, url: slug, sectionHeading: titleFor(markdown, relativePath), text: text.slice(0, 6000), modifiedAt: '', score };
  }).filter((doc) => doc.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
}
