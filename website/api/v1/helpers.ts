import { retrieveRelevantChunks, SearchContextChunk } from '../../src/rag/search';
import { createOpenAIClient, getOpenAIModel } from '../../src/rag/embeddings';

export interface UserPreferences {
  user_id?: string;
  adaptive_difficulty?: string;
  adaptive_code_samples?: boolean;
  preferred_language?: string;
  learning_goals?: string[];
  preferred_topics?: string[];
}

export async function findRelevantDocs(query: string, limit = 5): Promise<SearchContextChunk[]> {
  return retrieveRelevantChunks(query, limit);
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
  docs: SearchContextChunk[],
  userPreferences?: UserPreferences
): string {
  const context = docs
    .map((doc) => `Source: ${doc.title} (${doc.sectionHeading})\nPath: ${doc.url}\n\n${doc.text}`)
    .join('\n\n---\n\n');

  const preferenceText = buildPreferencePrompt(userPreferences);
  const guidance = `You are an expert textbook assistant for Physical AI & Humanoid Robotics. Use only the provided context from the textbook content to answer the question. Do not invent facts or add information from outside the context. If the answer is not available in the context, say "I could not find information about this in the textbook." ${preferenceText}`.trim();

  return `${guidance}\n\nContext:\n${context}\n\nQuestion:\n${query}`;
}

export async function generateOpenAIAnswer(prompt: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set.');
  }

  const client = createOpenAIClient();
  const model = getOpenAIModel();
  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: 'system',
        content: 'You are an expert textbook assistant providing answers from the provided textbook content.',
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
