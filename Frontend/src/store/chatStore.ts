import { create } from 'zustand';
import type { Conversation, Message } from '../types';



interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  messages: Message[];
  isStreaming: boolean;
  streamingContent: string;
  isSidebarOpen: boolean;
  isLoadingHistory: boolean;
  isLoadingMessages: boolean;

  setConversations: (convs: Conversation[]) => void;
  addConversation: (conv: Conversation) => void;
  removeConversation: (id: string) => void;
  updateConversationTitle: (id: string, title: string) => void;
  setActiveConversation: (id: string | null) => void;
  setMessages: (messages: Message[]) => void;
  addUserMessage: (content: string) => void;
  appendStreamToken: (token: string) => void;
  finalizeStream: () => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setLoadingHistory: (loading: boolean) => void;
  setLoadingMessages: (loading: boolean) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  conversations: [],
  activeConversationId: null,
  messages: [],
  isStreaming: false,
  streamingContent: '',
  isSidebarOpen: true,
  isLoadingHistory: false,
  isLoadingMessages: false,

  setConversations: (convs) => set({ conversations: convs }),

  addConversation: (conv) =>
    set((state) => ({
      conversations: [conv, ...state.conversations],
    })),

  removeConversation: (id) =>
    set((state) => ({
      conversations: state.conversations.filter((c) => c.id !== id),
      activeConversationId:
        state.activeConversationId === id ? null : state.activeConversationId,
      messages: state.activeConversationId === id ? [] : state.messages,
    })),

  updateConversationTitle: (id, title) =>
    set((state) => ({
      conversations: state.conversations.map((c) =>
        c.id === id ? { ...c, title } : c
      ),
    })),

  setActiveConversation: (id) => set({ activeConversationId: id }),

  setMessages: (messages) => set({ messages }),

  addUserMessage: (content) => {
    const tempMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };
    set((state) => ({
      messages: [...state.messages, tempMessage],
    }));
  },

  appendStreamToken: (token) =>
    set((state) => ({
      streamingContent: state.streamingContent + token,
    })),

  finalizeStream: () => {
    const { streamingContent } = get();
    if (streamingContent) {
      const assistantMessage: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: streamingContent,
        created_at: new Date().toISOString(),
      };
      set((state) => ({
        messages: [...state.messages, assistantMessage],
        streamingContent: '',
        isStreaming: false,
      }));
    } else {
      set({ streamingContent: '', isStreaming: false });
    }
  },

  toggleSidebar: () =>
    set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

  setSidebarOpen: (open) => set({ isSidebarOpen: open }),

  setLoadingHistory: (loading) => set({ isLoadingHistory: loading }),

  setLoadingMessages: (loading) => set({ isLoadingMessages: loading }),

  reset: () =>
    set({
      conversations: [],
      activeConversationId: null,
      messages: [],
      isStreaming: false,
      streamingContent: '',
    }),
}));
