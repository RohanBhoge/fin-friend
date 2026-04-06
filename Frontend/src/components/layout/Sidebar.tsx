import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PlusCircle,
  MessageSquare,
  Pencil,
  Trash2,
  Settings,
  LogOut,
  X,
  TrendingUp,
} from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { useAuthStore } from '../../store/authStore';
import { conversationsApi } from '../../api/client';
import toast from 'react-hot-toast';

export default function Sidebar() {
  const navigate = useNavigate();
  const {
    conversations,
    activeConversationId,
    isSidebarOpen,
    setConversations,
    addConversation,
    removeConversation,
    setActiveConversation,
    setMessages,
    setLoadingMessages,
    updateConversationTitle,
    setSidebarOpen,
  } = useChatStore();
  const { user, logout } = useAuthStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const editRef = useRef<HTMLInputElement>(null);

  // Fetch conversations on mount
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const data = await conversationsApi.list();
        setConversations(data.items);
      } catch {
        // Silent fail on initial load
      }
    };
    fetchConversations();
  }, [setConversations]);

  const handleNewChat = async () => {
    try {
      const conv = await conversationsApi.create();
      addConversation(conv);
      setActiveConversation(conv.id);
      setMessages([]);
      // Close sidebar on mobile
      if (window.innerWidth < 1024) setSidebarOpen(false);
    } catch {
      toast.error('Failed to create conversation');
    }
  };

  const handleSelectConversation = async (id: string) => {
    setActiveConversation(id);
    setLoadingMessages(true);
    try {
      const detail = await conversationsApi.get(id);
      setMessages(detail.messages);
    } catch {
      toast.error('Failed to load messages');
    } finally {
      setLoadingMessages(false);
    }
    if (window.innerWidth < 1024) setSidebarOpen(false);
  };

  const handleDelete = async (id: string) => {
    try {
      await conversationsApi.delete(id);
      removeConversation(id);
      toast.success('Conversation deleted');
    } catch {
      toast.error('Failed to delete conversation');
    }
  };

  const startRename = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditTitle(currentTitle);
    setTimeout(() => editRef.current?.focus(), 50);
  };

  const finishRename = async () => {
    if (editingId && editTitle.trim()) {
      try {
        await conversationsApi.rename(editingId, editTitle.trim());
        updateConversationTitle(editingId, editTitle.trim());
      } catch {
        toast.error('Failed to rename');
      }
    }
    setEditingId(null);
  };

  // Group conversations by date
  const groupByDate = (convs: typeof conversations) => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today.getTime() - 86400000);
    const week = new Date(today.getTime() - 7 * 86400000);

    const groups: { label: string; items: typeof conversations }[] = [
      { label: 'Today', items: [] },
      { label: 'Yesterday', items: [] },
      { label: 'Last 7 Days', items: [] },
      { label: 'Older', items: [] },
    ];

    convs.forEach((c) => {
      const d = new Date(c.updated_at || c.created_at);
      if (d >= today) groups[0].items.push(c);
      else if (d >= yesterday) groups[1].items.push(c);
      else if (d >= week) groups[2].items.push(c);
      else groups[3].items.push(c);
    });

    return groups.filter((g) => g.items.length > 0);
  };

  const groups = groupByDate(conversations);

  const sidebarContent = (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-dark-border">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-brand rounded-lg flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-gray-900 dark:text-white">
              FinFriend
            </span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-1 rounded hover:bg-gray-200 dark:hover:bg-dark-border"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-brand hover:bg-brand-dark text-white rounded-xl font-medium transition-colors"
          aria-label="Start new chat"
        >
          <PlusCircle className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto p-2">
        {groups.length === 0 && (
          <div className="text-center py-8 text-gray-400 text-sm">
            No conversations yet
          </div>
        )}
        {groups.map((group) => (
          <div key={group.label} className="mb-4">
            <div className="px-3 py-1.5 text-xs font-medium text-gray-500 dark:text-gray-500 uppercase tracking-wider">
              {group.label}
            </div>
            {group.items.map((conv) => (
              <div
                key={conv.id}
                className={`group relative flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-colors mb-0.5 ${
                  activeConversationId === conv.id
                    ? 'bg-brand/10 border-l-2 border-brand'
                    : 'hover:bg-gray-100 dark:hover:bg-dark-border/50'
                }`}
                onClick={() => handleSelectConversation(conv.id)}
              >
                <MessageSquare className="w-4 h-4 text-gray-400 shrink-0" />

                {editingId === conv.id ? (
                  <input
                    ref={editRef}
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={finishRename}
                    onKeyDown={(e) => e.key === 'Enter' && finishRename()}
                    className="flex-1 bg-transparent text-sm text-gray-900 dark:text-gray-100 outline-none border-b border-brand"
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span className="flex-1 text-sm text-gray-700 dark:text-gray-300 truncate">
                    {conv.title}
                  </span>
                )}

                {/* Action buttons */}
                <div className="hidden group-hover:flex items-center gap-1 shrink-0">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startRename(conv.id, conv.title);
                    }}
                    className="p-1 rounded hover:bg-gray-200 dark:hover:bg-dark-border"
                    aria-label="Rename conversation"
                  >
                    <Pencil className="w-3.5 h-3.5 text-gray-400" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(conv.id);
                    }}
                    className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/20"
                    aria-label="Delete conversation"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-red-400" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* User Section */}
      <div className="p-4 border-t border-gray-200 dark:border-dark-border">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-brand/20 flex items-center justify-center shrink-0">
            <span className="text-sm font-medium text-brand">
              {user?.full_name?.[0]?.toUpperCase() ||
                user?.email?.[0]?.toUpperCase() ||
                '?'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {user?.full_name || 'User'}
            </p>
            <p className="text-xs text-gray-500 truncate">{user?.email}</p>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => navigate('/profile')}
              className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-dark-border"
              aria-label="Settings"
            >
              <Settings className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={logout}
              className="p-1.5 rounded hover:bg-red-100 dark:hover:bg-red-900/20"
              aria-label="Log out"
            >
              <LogOut className="w-4 h-4 text-red-400" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex w-[260px] h-full bg-white dark:bg-dark-surface border-r border-gray-200 dark:border-dark-border flex-col shrink-0">
        {sidebarContent}
      </aside>

      {/* Mobile sidebar */}
      <AnimatePresence>
        {isSidebarOpen && (
          <>
            {/* Overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="lg:hidden fixed inset-0 bg-black/50 z-40"
              onClick={() => setSidebarOpen(false)}
            />
            {/* Drawer */}
            <motion.aside
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="lg:hidden fixed left-0 top-0 h-full w-[260px] bg-white dark:bg-dark-surface border-r border-gray-200 dark:border-dark-border z-50 flex flex-col"
            >
              {sidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
