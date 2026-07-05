import { buildChatPrompt, generateOpenAIAnswer } from './helpers';

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  const { query, selected_text, user_preferences } = req.body ?? {};

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'Query cannot be empty.' });
  }

  if (!selected_text || typeof selected_text !== 'string' || !selected_text.trim()) {
    return res.status(400).json({ error: 'Selected text cannot be empty.' });
  }

  try {
    const prompt = buildChatPrompt(query, [
      {
        title: 'Selected Text',
        source: 'selected_text',
        content: selected_text,
      },
    ], user_preferences);

    const answer = await generateOpenAIAnswer(prompt);

    return res.status(200).json({
      response: answer,
      sources: ['Selected Text'],
    });
  } catch (error: any) {
    console.error('selected-text handler error:', error);
    return res.status(500).json({ error: error.message || 'Internal server error.' });
  }
}
