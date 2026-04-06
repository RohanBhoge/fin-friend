/* TypeScript interfaces matching backend Pydantic responses exactly */

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface ConversationList {
  items: Conversation[];
  total: number;
  page: number;
  limit: number;
}

export interface StreamChunk {
  token?: string;
  done?: boolean;
  error?: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
