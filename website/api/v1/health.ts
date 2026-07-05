export default function handler(req: any, res: any) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed. Use GET.' });
  }

  return res.status(200).json({
    status: 'healthy',
    service: 'Physical AI & Humanoid Robotics website chatbot API',
  });
}
