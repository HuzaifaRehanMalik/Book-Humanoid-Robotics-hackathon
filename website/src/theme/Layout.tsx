import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { AuthProvider } from '../contexts/AuthContext';
import SidebarChatToggle from '../components/SidebarChatToggle';
import type { Props } from '@theme/Layout';

export default function Layout(props: Props): React.ReactNode {
  return (
    <AuthProvider>
      <OriginalLayout {...props} />
      <SidebarChatToggle />
    </AuthProvider>
  );
}
