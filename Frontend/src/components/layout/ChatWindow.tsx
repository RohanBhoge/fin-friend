import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import {
  Send,
  User,
  Bot,
  Copy,
  Check,
  Menu,
  TrendingUp,
  MessageSquare,
  DollarSign,
  PieChart,
  ArrowRight,
} from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { useStreamingChat } from '../../hooks/useStreamingChat';
import { useAuthStore } from '../../store/authStore';
import { conversationsApi } from '../../api/client';
import type { ConversationDetail } from '../../types';
import ThemeToggle from '../ui/ThemeToggle';
import toast from 'react-hot-toast';

export default function ChatWindow() {
  const {
    messages,
    activeConversationId,
    isStreaming,
    streamingContent,
    isLoadingMessages,
    toggleSidebar,
    addConversation,
    setActiveConversation,
  } = useChatStore();
  const { user } = useAuthStore();
  const { sendMessage } = useStreamingChat();

  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  // Auto-select latest conversation if none active
  useEffect(() => {
    if (!activeConversationId && useChatStore.getState().conversations.length > 0) {
      const latest = useChatStore.getState().conversations[0];
      setActiveConversation(latest.id);
      
      // Load messages for this conversation
      conversationsApi.get(latest.id).then((detail: ConversationDetail) => {
        useChatStore.getState().setMessages(detail.messages);
      });
    }
  }, [activeConversationId, setActiveConversation]);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;

    let conversationId = activeConversationId;

    // If no active conversation, create one first
    if (!conversationId) {
      try {
        const newConv = await conversationsApi.create('New Conversation');
        addConversation(newConv);
        setActiveConversation(newConv.id);
        conversationId = newConv.id;
      } catch (err) {
        toast.error("Failed to create conversation");
        return;
      }
    }

    const message = input;
    setInput('');
    if (conversationId) {
      await sendMessage(message, conversationId);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success('Copied to clipboard');
    setTimeout(() => setCopiedId(null), 2000);
  };

  const suggestions = [
    { title: 'Analyze budget', prompt: 'Can you help me analyze my monthly budget?' },
    { title: 'Debt plan', prompt: 'What is the best way to pay off my high-interest debt?' },
    { title: 'Investing 101', prompt: 'How should I start investing with a small amount?' },
    { title: 'Savings goal', prompt: 'Help me create a plan to save for a house down payment.' },
  ];

  const handleSuggestion = (prompt: string) => {
    setInput(prompt);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-dark-bg relative overflow-hidden">
      {/* Header */}
      <header className="h-14 flex items-center justify-between px-4 border-b border-gray-200 dark:border-dark-border bg-white/80 dark:bg-dark-bg/80 glass sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <button
            onClick={toggleSidebar}
            className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-border"
          >
            <Menu className="w-5 h-5 text-gray-500" />
          </button>
          <div className="hidden lg:flex items-center gap-2">
             <div className="w-6 h-6 bg-brand rounded-lg flex items-center justify-center">
                <TrendingUp className="w-3.5 h-3.5 text-white" />
             </div>
             <span className="font-semibold text-gray-900 dark:text-gray-100 text-sm">FinFriend Chat</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <div className="h-8 w-8 rounded-full bg-brand/10 flex items-center justify-center overflow-hidden border border-brand/20">
             {user?.avatar_url ? (
               <img src={user.avatar_url} alt={user.full_name || ''} className="w-full h-full object-cover" />
             ) : (
               <User className="w-4 h-4 text-brand" />
             )}
          </div>
        </div>
      </header>

      {/* Message Feed */}
      <main className="flex-1 overflow-y-auto p-4 lg:p-8 space-y-8">
        {!activeConversationId && messages.length === 0 && !isLoadingMessages ? (
          <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="w-20 h-20 bg-brand/10 rounded-3xl flex items-center justify-center mb-4">
              <TrendingUp className="w-10 h-10 text-brand" />
            </div>
            <div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
                How can I help with your finances today?
              </h2>
              <p className="text-gray-500 dark:text-gray-400">
                FinFriend is your personal financial health assistant. Ask me anything about budgeting, debt, or investments.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full pt-4">
              {suggestions.map((s) => (
                <button
                  key={s.title}
                  onClick={() => handleSuggestion(s.prompt)}
                  className="p-4 text-left rounded-2xl border border-gray-200 dark:border-dark-border hover:bg-gray-50 dark:hover:bg-dark-surface hover:border-brand transition-all group"
                >
                  <p className="font-medium text-gray-900 dark:text-white mb-1 group-hover:text-brand">
                    {s.title}
                  </p>
                  <p className="text-xs text-gray-500 truncate">{s.prompt}</p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-8 pb-12">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-4 ${
                  m.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                    m.role === 'user'
                      ? 'bg-brand'
                      : 'bg-indigo-100 dark:bg-indigo-900/30'
                  }`}
                >
                  {m.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-brand" />
                  )}
                </div>

                <div className={`flex flex-col max-w-[85%] ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div
                    className={`p-4 rounded-2xl ${
                      m.role === 'user'
                        ? 'bg-brand text-white rounded-tr-sm'
                        : 'bg-gray-100 dark:bg-dark-surface dark:text-gray-200 rounded-tl-sm'
                    }`}
                  >
                    <div className="prose dark:prose-invert max-w-none prose-sm sm:prose-base">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                      >
                        {m.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                  
                  <div className="mt-2 flex items-center gap-2 px-1">
                    <span className="text-[10px] text-gray-400">
                      {new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    {m.role === 'assistant' && (
                      <button
                        onClick={() => copyToClipboard(m.content, m.id)}
                        className="p-1 hover:text-brand transition-colors"
                        title="Copy message"
                      >
                        {copiedId === m.id ? (
                          <Check className="w-3 h-3" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Streaming Active Message */}
            {isStreaming && streamingContent && (
              <div className="flex gap-4 animate-in fade-in">
                <div className="w-8 h-8 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-brand" />
                </div>
                <div className="flex flex-col max-w-[85%]">
                  <div className="p-4 rounded-2xl bg-gray-100 dark:bg-dark-surface dark:text-gray-200 rounded-tl-sm">
                    <div className="prose dark:prose-invert max-w-none prose-sm sm:prose-base">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                      >
                        {streamingContent}
                      </ReactMarkdown>
                      <span className="inline-block w-2 h-4 bg-brand ml-1 animate-blink"></span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Thinking Indicator */}
            {isStreaming && !streamingContent && (
               <div className="flex gap-4">
                  <div className="w-8 h-8 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg flex items-center justify-center shrink-0">
                    <Bot className="w-4 h-4 text-brand" />
                  </div>
                  <div className="flex gap-1 items-center p-4 bg-gray-100 dark:bg-dark-surface rounded-2xl rounded-tl-sm">
                    <span className="w-1.5 h-1.5 bg-brand rounded-full dot-bounce-1"></span>
                    <span className="w-1.5 h-1.5 bg-brand rounded-full dot-bounce-2"></span>
                    <span className="w-1.5 h-1.5 bg-brand rounded-full dot-bounce-3"></span>
                  </div>
               </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Input Area */}
      <footer className="p-4 bg-white dark:bg-dark-bg border-t border-gray-200 dark:border-dark-border z-20">
        <div className="max-w-3xl mx-auto relative group">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
            placeholder="Message FinFriend..."
            className="w-full pl-4 pr-12 py-3 bg-gray-100 dark:bg-dark-input dark:text-white border-transparent focus:border-brand/30 focus:ring-0 rounded-2xl resize-none transition-all duration-200"
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || isStreaming}
            className={`absolute right-2 bottom-2 p-2 rounded-xl transition-all ${
              !input.trim() || isStreaming
                ? 'text-gray-400 cursor-not-allowed'
                : 'bg-brand text-white hover:bg-brand-dark shadow-sm'
            }`}
          >
            {isStreaming ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        <p className="mt-2 text-[10px] text-center text-gray-500 max-w-md mx-auto">
          FinFriend can make mistakes. Consider checking important information. Consult a certified financial advisor for major decisions.
        </p>
      </footer>
    </div>
  );
}
