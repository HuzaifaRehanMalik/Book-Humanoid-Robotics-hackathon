import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { AuthProvider } from '../contexts/AuthContext';
import ChatbotWidget from '../components/ChatbotWidget';
import SidebarChatToggle from '../components/SidebarChatToggle';
export default function Layout(props) {
    return (<AuthProvider>
      <OriginalLayout {...props}/>
      <ChatbotWidget />
      <SidebarChatToggle />
    </AuthProvider>);
}
