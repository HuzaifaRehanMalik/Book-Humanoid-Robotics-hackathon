import { buildChatPrompt, findRelevantDocs, generateOpenAIAnswer } from './helpers';

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  const { query, user_preferences, context } = req.body ?? {};

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'Query cannot be empty.' });
  }

  try {
    const docs = typeof context === 'string' && context.trim().length > 0
      ? [{ title: 'Provided context', source: 'user', content: context }]
      : await findRelevantDocs(query, 3);

    const prompt = buildChatPrompt(query, docs, user_preferences);
    const answer = await generateOpenAIAnswer(prompt);

    return res.status(200).json({
      response: answer,
      sources: docs.map((doc) => doc.title),
    });
  } catch (error: any) {
    console.error('chat handler error:', error);
    return res.status(500).json({ error: error.message || 'Internal server error.' });
  }
}
