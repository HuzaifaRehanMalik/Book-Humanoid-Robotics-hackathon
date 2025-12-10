import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { user } = useAuth();
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // In a real implementation, this would call the RAG API
      // For now, we'll simulate a response
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Get user preferences from localStorage
      const userPreferences = user ? localStorage.getItem('userPreferences') : null;
      let adaptiveDifficulty = 'intermediate';

      if (userPreferences) {
        try {
          const prefs = JSON.parse(userPreferences);
          adaptiveDifficulty = prefs.adaptive_difficulty || 'intermediate';
        } catch (e) {
          console.error('Error parsing user preferences:', e);
        }
      }

      // Simulate API response based on preferences
      let responseContent = `This is a simulated response to your query: "${inputValue}". `;

      if (adaptiveDifficulty === 'beginner') {
        responseContent += 'This explanation is simplified for beginners.';
      } else if (adaptiveDifficulty === 'advanced') {
        responseContent += 'This explanation includes advanced technical details.';
      } else {
        responseContent += 'This is an intermediate-level explanation.';
      }

      // Add assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: responseContent,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, there was an error processing your request.',
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={`chatbot-widget ${isOpen ? 'chatbot-widget--open' : ''}`}>
      {isOpen ? (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>Textbook Assistant</h3>
            <button
              className="chatbot-close-button"
              onClick={toggleChat}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="chatbot-welcome">
                <p>Hello! I'm your Physical AI & Humanoid Robotics textbook assistant.</p>
                <p>Ask me anything about the content, and I'll provide information based on the textbook.</p>
                {user && <p>Using preferences for: {user.name}</p>}
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`chatbot-message chatbot-message--${message.role}`}
                >
                  <div className="chatbot-message-content">
                    {message.content}
                  </div>
                  <div className="chatbot-message-timestamp">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="chatbot-input-form">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about the textbook..."
              disabled={isLoading}
              className="chatbot-input"
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="chatbot-send-button"
            >
              {isLoading ? '...' : '→'}
            </button>
          </form>
        </div>
      ) : (
        <button
          className="chatbot-toggle-button"
          onClick={toggleChat}
          aria-label="Open chat"
        >
          💬
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;