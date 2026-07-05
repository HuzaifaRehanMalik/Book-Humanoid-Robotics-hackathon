import OpenAI from 'openai';

export interface EmbeddingResponse {
  input: string;
  embedding: number[];
}

export function getOpenAIEmbeddingModel(): string {
  return process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small';
}

export function getOpenAIModel(): string {
  return process.env.OPENAI_MODEL || 'gpt-4.1-mini';
}

export function createOpenAIClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set.');
  }

  return new OpenAI({ apiKey });
}

export async function embedTexts(texts: string[]): Promise<EmbeddingResponse[]> {
  const client = createOpenAIClient();
  const model = getOpenAIEmbeddingModel();

  const response = await client.embeddings.create({
    model,
    input: texts,
  });

  if (!response.data || response.data.length !== texts.length) {
    throw new Error('Unexpected embedding response from OpenAI.');
  }

  return response.data.map((item) => ({
    input: item.input as string,
    embedding: item.embedding as number[],
  }));
}
