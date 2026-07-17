import React, { useState } from "react";
import styles from "./ChatbotWidget.module.css";
function getApiUrl() {
    if (typeof window === 'undefined') {
        return process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
    }
    return (process.env.REACT_APP_API_BASE_URL ||
        window.location.origin);
}
export default function ChatbotWidget() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const sendMessage = async () => {
        if (!input.trim())
            return;
        const userMessage = { role: "user", content: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);
        try {
            const response = await fetch(`${getApiUrl()}/api/v1/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: input }),
            });
            if (!response.ok) {
                throw new Error("API request failed");
            }
            const data = await response.json();
            const assistantMessage = {
                role: "assistant",
                content: data.response ?? data.answer ?? 'Sorry, no answer was returned.',
            };
            setMessages((prev) => [...prev, assistantMessage]);
        }
        catch (error) {
            console.error("Chat error:", error);
            setMessages((prev) => [
                ...prev,
                {
                    role: "assistant",
                    content: "Sorry, there was an error processing your request. Please try again.",
                },
            ]);
        }
        finally {
            setLoading(false);
        }
    };
    return (<div className={styles.chatContainer}>
      <div className={styles.messages}>
        {messages.map((msg, index) => (<div key={index} className={msg.role === "user"
                ? styles.userMessage
                : styles.assistantMessage}>
            {msg.content}
          </div>))}
        {loading && <div className={styles.assistantMessage}>Typing...</div>}
      </div>

      <div className={styles.inputContainer}>
        <input type="text" value={input} placeholder="Ask a question about the textbook..." onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendMessage()}/>
        <button onClick={sendMessage}>→</button>
      </div>
    </div>);
}
