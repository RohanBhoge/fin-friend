import axios from 'axios';
import { useAuthStore } from '../store/authStore';
import type {
  AuthResponse,
  User,
  Conversation,
  ConversationDetail,
  ConversationList,
} from '../types';

// ── Axios Instance ───────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: `${import.meta.env.VITE_API_BASE_URL}/api/v1`,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Request Interceptor: Inject JWT ──────────────────────────────────────────
api.interceptors.request.use((config) => {
  const { accessToken } = useAuthStore.getState();
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

// ── Response Interceptor: Handle 401 + Auto Refresh ──────────────────────────
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (token: string) => void;
  reject: (error: unknown) => void;
}> = [];

const processQueue = (error: unknown, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token!);
    }
  });
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({
            resolve: (token: string) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(api(originalRequest));
            },
            reject,
          });
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        await useAuthStore.getState().refreshAccessToken();
        const newToken = useAuthStore.getState().accessToken;

        if (newToken) {
          processQueue(null, newToken);
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        processQueue(refreshError, null);
        useAuthStore.getState().logout();
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// ── Auth API ─────────────────────────────────────────────────────────────────
export const authApi = {
  register: async (email: string, password: string, full_name?: string) => {
    const { data } = await api.post<User>('/auth/register', {
      email,
      password,
      full_name,
    });
    return data;
  },

  login: async (email: string, password: string) => {
    const { data } = await api.post<AuthResponse>('/auth/login', {
      email,
      password,
    });
    return data;
  },

  getMe: async () => {
    const { data } = await api.get<User>('/auth/me');
    return data;
  },

  forgotPassword: async (email: string) => {
    const { data } = await api.post('/auth/forgot-password', { email });
    return data;
  },

  resetPassword: async (token: string, new_password: string) => {
    const { data } = await api.post('/auth/reset-password', {
      token,
      new_password,
    });
    return data;
  },
};

// ── Users API ────────────────────────────────────────────────────────────────
export const usersApi = {
  getProfile: async () => {
    const { data } = await api.get<User>('/users/profile');
    return data;
  },

  updateProfile: async (updates: {
    full_name?: string;
    email?: string;
    avatar_url?: string;
  }) => {
    const { data } = await api.patch<User>('/users/profile', updates);
    return data;
  },

  updateSecurity: async (current_password: string, new_password: string) => {
    const { data } = await api.patch('/users/security', {
      current_password,
      new_password,
    });
    return data;
  },

  deleteAccount: async (password: string) => {
    const { data } = await api.delete('/users/account', {
      data: { password },
    });
    return data;
  },
};

// ── Conversations API ────────────────────────────────────────────────────────
export const conversationsApi = {
  list: async (page = 1, limit = 50) => {
    const { data } = await api.get<ConversationList>(
      `/conversations?page=${page}&limit=${limit}`
    );
    return data;
  },

  create: async (title?: string) => {
    const { data } = await api.post<Conversation>('/conversations', {
      title: title || 'New Conversation',
    });
    return data;
  },

  get: async (id: string) => {
    const { data } = await api.get<ConversationDetail>(
      `/conversations/${id}`
    );
    return data;
  },

  rename: async (id: string, title: string) => {
    const { data } = await api.patch<Conversation>(`/conversations/${id}`, {
      title,
    });
    return data;
  },

  delete: async (id: string) => {
    const { data } = await api.delete(`/conversations/${id}`);
    return data;
  },
};

// ── Chat API ─────────────────────────────────────────────────────────────────
export const chatApi = {
  generateTitle: async (conversation_id: string) => {
    const { data } = await api.post<{ title: string }>(
      '/chat/generate-title',
      { conversation_id }
    );
    return data;
  },
};

export default api;
