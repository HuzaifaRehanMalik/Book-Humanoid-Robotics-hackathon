import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { AuthProvider } from '../contexts/AuthContext';
import ChatbotWidget from '../components/ChatbotWidget';
import type { Props } from '@theme/Layout';

export default function Layout(props: Props): JSX.Element {
  return (
    <AuthProvider>
      <OriginalLayout {...props} />
      <ChatbotWidget />
    </AuthProvider>
  );
}