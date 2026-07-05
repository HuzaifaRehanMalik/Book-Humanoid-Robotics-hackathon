import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import OpenAI from 'openai';

export interface UserPreferences {
  user_id?: string;
  adaptive_difficulty?: string;
  adaptive_code_samples?: boolean;
  preferred_language?: string;
  learning_goals?: string[];
  preferred_topics?: string[];
}

export interface DocumentSource {
  title: string;
  source: string;
  content: string;
}

let docsCache: DocumentSource[] | null = null;

async function collectMarkdownFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectMarkdownFiles(fullPath)));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.md')) {
      files.push(fullPath);
    }
  }

  return files;
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

function scoreDocument(content: string, terms: string[]): number {
  const lowerContent = normalizeText(content);
  return terms.reduce((score, term) => {
    if (!term) return score;
    const occurrences = lowerContent.split(term).length - 1;
    return score + occurrences;
  }, 0);
}

export async function loadTextbookDocuments(): Promise<DocumentSource[]> {
  if (docsCache) {
    return docsCache;
  }

  const docsRoot = path.join(__dirname, '..', '..', 'docs');
  const files = await collectMarkdownFiles(docsRoot);
  const docs: DocumentSource[] = [];

  for (const filePath of files) {
    const raw = await readFile(filePath, 'utf8');
    const titleMatch = raw.match(/^#{1,3}\s+(.*)$/m);
    const title = titleMatch ? titleMatch[1].trim() : path.basename(filePath);
    const source = path.relative(path.join(__dirname, '..', '..'), filePath).replace(/\\/g, '/');
    docs.push({
      title,
      source,
      content: raw,
    });
  }

  docsCache = docs;
  return docs;
}

export async function findRelevantDocs(query: string, limit = 3): Promise<DocumentSource[]> {
  const docs = await loadTextbookDocuments();
  const terms = normalizeText(query).split(/[^a-z0-9]+/).filter(Boolean);

  const scored = docs
    .map((doc) => ({
      doc,
      score: scoreDocument(doc.content, terms),
    }))
    .sort((a, b) => b.score - a.score)
    .filter((item) => item.score > 0)
    .slice(0, limit)
    .map((item) => item.doc);

  return scored.length > 0 ? scored : docs.slice(0, Math.min(limit, docs.length));
}

function buildPreferencePrompt(userPreferences?: UserPreferences): string {
  if (!userPreferences) {
    return '';
  }

  const parts: string[] = [];
  if (userPreferences.adaptive_difficulty) {
    parts.push(`Make the answer suitable for a ${userPreferences.adaptive_difficulty} learner.`);
  }
  if (userPreferences.preferred_language && userPreferences.preferred_language !== 'en') {
    parts.push(`Respond in ${userPreferences.preferred_language} if possible.`);
  }
  if (userPreferences.adaptive_code_samples === false) {
    parts.push('Keep the response conceptual and avoid code samples unless explicitly requested.');
  }

  return parts.join(' ');
}

export function buildChatPrompt(
  query: string,
  docs: DocumentSource[],
  userPreferences?: UserPreferences
): string {
  const context = docs
    .map((doc) => `Source: ${doc.title} (${doc.source})\n${doc.content.trim()}`)
    .join('\n\n---\n\n');

  const preferenceText = buildPreferencePrompt(userPreferences);
  const guidance = `You are an expert textbook assistant for Physical AI & Humanoid Robotics. Use only the provided context from the textbook content to answer the question. Do not invent facts or add information from outside the context. If the answer is not available, say you do not know. ${preferenceText}`.trim();

  return `${guidance}\n\nContext:\n${context}\n\nQuestion:\n${query}`;
}

export async function generateOpenAIAnswer(prompt: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set.');
  }

  const client = new OpenAI({ apiKey });
  const model = process.env.OPENAI_MODEL_NAME || 'gpt-4o-mini';
  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: 'system',
        content:
          'You are an expert textbook assistant providing answers from provided textbook content.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
    temperature: Number(process.env.OPENAI_TEMPERATURE ?? 0.2),
    max_tokens: Number(process.env.OPENAI_MAX_TOKENS ?? 500),
  });

  const answer = response.choices?.[0]?.message?.content;
  return typeof answer === 'string' ? answer.trim() : '';
}
