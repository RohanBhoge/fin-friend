import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { TrendingUp, Shield, Zap, BarChart3 } from 'lucide-react';

interface AuthLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

const features = [
  { icon: <Shield className="w-4 h-4" />, label: 'RAG Intelligence', desc: 'Answers grounded in trusted financial documents.' },
  { icon: <Zap className="w-4 h-4" />, label: 'Real-time Streaming', desc: 'Token-by-token responses, instantly.' },
  { icon: <BarChart3 className="w-4 h-4" />, label: 'Health Reports', desc: 'AI-generated financial health summaries.' },
];

export default function AuthLayout({ children, title, subtitle }: AuthLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-gray-50 dark:bg-dark-bg">

      {/* ── Left Branding Panel – visible md+ ──────────────────────────── */}
      <div className="hidden lg:flex lg:w-[45%] xl:w-1/2 relative overflow-hidden bg-gradient-to-br from-[#0F0F10] via-[#1A1A2E] to-[#0A0A14] flex-col items-center justify-center p-12">
        {/* Animated blobs */}
        <motion.div
          className="absolute w-[480px] h-[480px] rounded-full bg-brand/20 blur-[100px] -top-20 -left-20"
          animate={{ opacity: [0.3, 0.6, 0.3], scale: [1, 1.1, 1] }}
          transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute w-[320px] h-[320px] rounded-full bg-purple-500/15 blur-[80px] bottom-10 right-0"
          animate={{ opacity: [0.2, 0.4, 0.2], x: [0, 20, 0] }}
          transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
        />

        <motion.div
          className="relative z-10 max-w-sm w-full"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 mb-10 group">
            <div className="w-10 h-10 bg-brand rounded-xl flex items-center justify-center shadow-lg shadow-brand/30 group-hover:scale-105 transition-transform">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">FinFriend</span>
          </Link>

          <h2 className="text-3xl font-extrabold text-white mb-3 leading-tight">
            Your AI Financial Health Advisor
          </h2>
          <p className="text-gray-400 leading-relaxed mb-10">
            Get personalized financial insights grounded in proven principles. Chat naturally, understand deeply, act confidently.
          </p>

          {/* Feature list */}
          <div className="space-y-4">
            {features.map(f => (
              <div key={f.label} className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-brand/20 flex items-center justify-center text-brand shrink-0">
                  {f.icon}
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">{f.label}</p>
                  <p className="text-xs text-gray-500 leading-relaxed">{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* ── Right Form Panel ────────────────────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center px-4 py-10 sm:px-8 lg:px-12">
        <motion.div
          className="w-full max-w-md"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center gap-2 mb-8">
            <Link to="/" className="flex items-center gap-2 group">
              <div className="w-8 h-8 bg-brand rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform">
                <TrendingUp className="w-4 h-4 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">FinFriend</span>
            </Link>
          </div>

          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-gray-900 dark:text-white mb-1">
              {title}
            </h1>
            {subtitle && (
              <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{subtitle}</p>
            )}
          </div>

          <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm p-6 sm:p-8">
            {children}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
