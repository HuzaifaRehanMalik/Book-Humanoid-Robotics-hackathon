export default function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ error: 'Method not allowed. Use GET.' });
    }
    return res.status(200).json({
        status: 'healthy',
        service: 'Physical AI & Humanoid Robotics website chatbot API',
    });
}
