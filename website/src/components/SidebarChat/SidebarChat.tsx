import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import styles from './SidebarChat.module.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: string[];
}

interface SidebarChatProps {
  isOpen: boolean;
  onClose: () => void;
}

const SidebarChat: React.FC<SidebarChatProps> = ({ isOpen, onClose }) => {
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
      // Get user preferences from localStorage
      const userPreferences = user ? localStorage.getItem('userPreferences') : null;
      let preferences = null;

      if (userPreferences) {
        try {
          preferences = JSON.parse(userPreferences);
        } catch (e) {
          console.error('Error parsing user preferences:', e);
        }
      }

      // Determine API base URL based on environment variable
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL ||
        (typeof window !== 'undefined'
          ? window.location.origin
          : '');

      // Call the RAG API
      const response = await fetch(`${apiBaseUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          user_preferences: preferences
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.error || `API request failed with status ${response.status}`);
      }

      const data = await response.json();

      // Add assistant message with the API response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response ?? data.answer ?? 'Sorry, there was no answer returned.',
        role: 'assistant',
        timestamp: new Date(),
        sources: Array.isArray(data.sources)
          ? data.sources.map((source: any) => {
              if (typeof source === 'string') return source;
              return source.title || source.filePath || source.url || '';
            }).filter(Boolean)
          : [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: error instanceof Error
          ? `Sorry, I could not answer that: ${error.message}`
          : 'Sorry, there was an error processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className={styles['sidebar-chat-overlay']}>
      <div className={styles['sidebar-chat']}>
        <div className={styles['sidebar-chat-header']}>
          <h3>Textbook Assistant</h3>
          <button
            className={styles['sidebar-chat-close-button']}
            onClick={onClose}
            aria-label="Close chat"
          >
            ×
          </button>
        </div>

        <div className={styles['sidebar-chat-messages']}>
          {messages.length === 0 ? (
            <div className={styles['sidebar-chat-welcome']}>
              <p>Hello! I'm your Physical AI & Humanoid Robotics textbook assistant.</p>
              <p>Ask me anything about the content, and I'll provide information based on the textbook.</p>
              {user && <p>Using preferences for: {user.name}</p>}
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`${styles['sidebar-chat-message']} ${styles[`sidebar-chat-message--${message.role}`]}`}
              >
                <div className={styles['sidebar-chat-message-content']}>
                  {message.content}
                  {message.sources?.length ? (
                    <div className={styles['sidebar-chat-message-sources']}>
                      <strong>Sources:</strong> {message.sources.join(', ')}
                    </div>
                  ) : null}
                </div>
                <div className={styles['sidebar-chat-message-timestamp']}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className={styles['sidebar-chat-input-form']}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about the textbook..."
            disabled={isLoading}
            className={styles['sidebar-chat-input']}
          />
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className={styles['sidebar-chat-send-button']}
          >
            {isLoading ? '...' : '→'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default SidebarChat;
