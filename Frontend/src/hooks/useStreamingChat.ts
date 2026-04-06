import { useCallback, useRef } from 'react';
import { useAuthStore } from '../store/authStore';
import { useChatStore } from '../store/chatStore';
import { chatApi } from '../api/client';
import type { StreamChunk } from '../types';
import toast from 'react-hot-toast';

export function useStreamingChat() {
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (message: string, conversationId: string) => {
      const { accessToken } = useAuthStore.getState();
      const { addUserMessage, appendStreamToken, finalizeStream, messages } =
        useChatStore.getState();

      // 1. Optimistically add user message to UI
      addUserMessage(message);
      useChatStore.setState({ isStreaming: true, streamingContent: '' });

      // Cancel any previous stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        // 2. Open streaming POST request using native fetch
        const response = await fetch(
          `${import.meta.env.VITE_API_BASE_URL}/api/v1/chat/stream`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${accessToken}`,
            },
            body: JSON.stringify({
              conversation_id: conversationId,
              message,
            }),
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }

        if (!response.body) {
          throw new Error('No response body for streaming');
        }

        // 3. Read the SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let isFirstMessage = messages.length === 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk
            .split('\n')
            .filter((line) => line.startsWith('data: '));

          for (const line of lines) {
            const jsonStr = line.replace('data: ', '').trim();
            if (!jsonStr) continue;

            try {
              const parsed: StreamChunk = JSON.parse(jsonStr);

              if (parsed.error) {
                toast.error('AI error: ' + parsed.error);
                useChatStore.setState({
                  isStreaming: false,
                  streamingContent: '',
                });
                return;
              }

              if (parsed.token) {
                appendStreamToken(parsed.token);
              }

              if (parsed.done) {
                finalizeStream();

                // Auto-generate title after first exchange
                if (isFirstMessage) {
                  try {
                    const { title } =
                      await chatApi.generateTitle(conversationId);
                    useChatStore
                      .getState()
                      .updateConversationTitle(conversationId, title);
                  } catch {
                    // Title generation failure is non-critical
                  }
                }
              }
            } catch {
              // Ignore malformed SSE chunks
            }
          }
        }
      } catch (error: unknown) {
        if (error instanceof Error && error.name === 'AbortError') {
          return; // User cancelled, not an error
        }
        useChatStore.setState({ isStreaming: false, streamingContent: '' });
        toast.error('Failed to get AI response. Please try again.');
      }
    },
    []
  );

  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      useChatStore.setState({ isStreaming: false, streamingContent: '' });
    }
  }, []);

  return { sendMessage, cancelStream };
}
