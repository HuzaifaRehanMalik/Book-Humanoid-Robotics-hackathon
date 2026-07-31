import React, { useState } from 'react';
import styles from './SidebarChatToggle.module.css';
import SidebarChat from './SidebarChat/SidebarChat';

const SidebarChatToggle: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <>
      <button
        className={styles['sidebar-chat-toggle-button']}
        onClick={() => setIsSidebarOpen((isOpen) => !isOpen)}
        aria-label={isSidebarOpen ? 'Close chatbot' : 'Open chatbot'}
        aria-expanded={isSidebarOpen}
        type="button"
      >
        <span aria-hidden="true">Chat</span>
      </button>
      <SidebarChat
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />
    </>
  );
};

export default SidebarChatToggle;
