import React from 'react';
import Sidebar from '../components/layout/Sidebar';
import ChatWindow from '../components/layout/ChatWindow';

export default function ChatPage() {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-white dark:bg-dark-bg">
      <Sidebar />
      <div className="flex-1 min-w-0 relative">
        <ChatWindow />
      </div>
    </div>
  );
}
