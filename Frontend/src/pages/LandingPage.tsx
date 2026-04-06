import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  TrendingUp,
  Shield,
  Zap,
  MessageSquare,
  BarChart3,
  ChevronRight,
  UserPlus,
  Play,
  CheckCircle2,
  Lock,
  Globe,
  Menu,
  X,
  ArrowRight,
} from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import ThemeToggle from '../components/ui/ThemeToggle';

export default function LandingPage() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();
  const { scrollY } = useScroll();
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);
  const heroY = useTransform(scrollY, [0, 400], [0, -60]);

  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = ['Features', 'How it Works', 'Testimonials'];

  const features = [
    {
      title: 'RAG-Powered Intelligence',
      desc: 'Answers grounded in expert financial documents, providing reliable advice tailored to you.',
      icon: <Shield className="w-5 h-5" />,
      color: 'text-brand bg-brand/10',
    },
    {
      title: 'Bank-Level Security',
      desc: 'Your data is protected with JWT authentication and encrypted with enterprise-grade standards.',
      icon: <Lock className="w-5 h-5" />,
      color: 'text-indigo-500 bg-indigo-500/10',
    },
    {
      title: 'Persistent History',
      desc: 'Never lose a financial insight. Every conversation is saved securely for future reference.',
      icon: <MessageSquare className="w-5 h-5" />,
      color: 'text-purple-500 bg-purple-500/10',
    },
    {
      title: 'Real-time Streaming',
      desc: 'Experience lightning-fast responses token-by-token, making financial planning feel natural.',
      icon: <Zap className="w-5 h-5" />,
      color: 'text-yellow-500 bg-yellow-500/10',
    },
    {
      title: 'Financial Health Reports',
      desc: 'Get structured Markdown reports with actionable steps to improve your economic standing.',
      icon: <BarChart3 className="w-5 h-5" />,
      color: 'text-emerald-500 bg-emerald-500/10',
    },
    {
      title: 'Universal Access',
      desc: 'Manage your finances on the go with a fully responsive interface optimized for all devices.',
      icon: <Globe className="w-5 h-5" />,
      color: 'text-blue-500 bg-blue-500/10',
    },
  ];

  const steps = [
    { num: '01', title: 'Create Account', icon: <UserPlus className="w-6 h-6" />, desc: 'Sign up in seconds and secure your financial workspace.' },
    { num: '02', title: 'Start Chatting', icon: <MessageSquare className="w-6 h-6" />, desc: 'Share your income, expenses, and goals with your AI mentor.' },
    { num: '03', title: 'Get Insights', icon: <TrendingUp className="w-6 h-6" />, desc: 'Receive instant, RAG-grounded reports to optimize your wealth.' },
  ];

  const testimonials = [
    { name: 'Sarah J.', role: 'Product Manager', avatar: 'S', quote: 'FinFriend totally changed how I look at my monthly budget. The RAG intelligence is noticeably more grounded than other bots.' },
    { name: 'David L.', role: 'Software Engineer', avatar: 'D', quote: 'The real-time streaming is incredibly fast. Most useful AI application for my personal finances that I have found so far.' },
    { name: 'Elena K.', role: 'Small Business Owner', avatar: 'E', quote: 'The detailed health reports are exactly what I needed. Professional advice without the high cost of a consultant.' },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-dark-bg selection:bg-brand selection:text-white">
      {/* ── Navbar ─────────────────────────────────────────────────────── */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${isScrolled
        ? 'bg-white/90 dark:bg-dark-bg/90 border-b border-gray-200 dark:border-dark-border glass py-3'
        : 'bg-transparent py-4'
        }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-12">

            {/* Logo */}
            <div className="flex items-center gap-2 shrink-0">
              <div className="w-8 h-8 bg-brand rounded-xl flex items-center justify-center shadow-lg shadow-brand/20">
                <TrendingUp className="w-4 h-4 text-white" />
              </div>
              <span className="text-lg font-bold text-gray-900 dark:text-white tracking-tight">FinFriend</span>
            </div>

            {/* Desktop Links */}
            <div className="hidden md:flex items-center gap-6">
              {navLinks.map(link => (
                <a
                  key={link}
                  href={`#${link.toLowerCase().replace(/\s+/g, '-')}`}
                  className="text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-brand dark:hover:text-white transition-colors focus:outline-none focus-visible:outline-none outline-none"
                >
                  {link}
                </a>
              ))}
            </div>

            {/* Desktop Actions */}
            <div className="hidden md:flex items-center gap-3">
              <ThemeToggle />
              {isAuthenticated ? (
                <button
                  onClick={() => navigate('/chat')}
                  className="px-4 py-2 bg-brand hover:bg-brand-dark text-white rounded-xl text-sm font-semibold transition-all hover:scale-105 active:scale-95"
                >
                  Dashboard
                </button>
              ) : (
                <>
                  <Link to="/login" className="text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-brand transition-colors px-3 py-2">
                    Login
                  </Link>
                  <Link to="/register" className="px-4 py-2 bg-brand hover:bg-brand-dark text-white rounded-xl text-sm font-semibold shadow-lg shadow-brand/20 transition-all hover:scale-105 active:scale-95">
                    Get Started
                  </Link>
                </>
              )}
            </div>

            {/* Mobile: theme + hamburger */}
            <div className="flex md:hidden items-center gap-2">
              <ThemeToggle />
              <button
                onClick={() => setMobileMenuOpen(v => !v)}
                className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-border"
                aria-label="Toggle menu"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu Drawer */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden bg-white dark:bg-dark-surface border-b border-gray-200 dark:border-dark-border px-4 pb-4 pt-2 space-y-1"
          >
            {navLinks.map(link => (
              <a
                key={link}
                href={`#${link.toLowerCase().replace(/\s+/g, '-')}`}
                onClick={() => setMobileMenuOpen(false)}
                className="block py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-brand focus:outline-none focus-visible:outline-none outline-none"
              >
                {link}
              </a>
            ))}
            <div className="pt-3 flex flex-col gap-2">
              {isAuthenticated ? (
                <button onClick={() => navigate('/chat')} className="w-full py-2.5 bg-brand text-white rounded-xl font-semibold text-sm">
                  Dashboard
                </button>
              ) : (
                <>
                  <Link to="/login" onClick={() => setMobileMenuOpen(false)} className="block text-center py-2.5 border border-gray-200 dark:border-dark-border text-gray-700 dark:text-gray-300 rounded-xl font-semibold text-sm">
                    Login
                  </Link>
                  <Link to="/register" onClick={() => setMobileMenuOpen(false)} className="block text-center py-2.5 bg-brand text-white rounded-xl font-semibold text-sm">
                    Get Started Free
                  </Link>
                </>
              )}
            </div>
          </motion.div>
        )}
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden px-4 pt-20">

        {/* Background blobs */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-40 -right-40 w-[600px] h-[600px] bg-brand/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-[500px] h-[500px] bg-purple-500/10 rounded-full blur-3xl" />
        </div>

        <motion.div
          style={{ opacity: heroOpacity, y: heroY }}
          className="relative z-10 text-center max-w-5xl mx-auto space-y-6 py-16"
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-brand/10 border border-brand/20"
          >
            <span className="w-1.5 h-1.5 bg-brand rounded-full animate-pulse" />
            <span className="text-xs font-bold text-brand uppercase tracking-widest">Powered by Gemini 2.5 Flash</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-6xl lg:text-7xl font-black text-gray-900 dark:text-white tracking-tight leading-tight"
          >
            Your AI{' '}
            <span className="gradient-text">Financial Health</span>
            <br />
            Advisor
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="max-w-xl mx-auto text-base sm:text-lg text-gray-500 dark:text-gray-400 leading-relaxed"
          >
            Get personalized financial insights grounded in proven principles. Chat naturally, understand deeply, act confidently.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2"
          >
            <Link
              to="/register"
              className="w-full sm:w-auto px-7 py-3.5 bg-brand hover:bg-brand-dark text-white rounded-2xl font-bold shadow-xl shadow-brand/25 transition-all hover:scale-105 active:scale-95 flex items-center justify-center gap-2"
            >
              Start Free Analysis
              <ChevronRight className="w-4 h-4" />
            </Link>
            <button className="w-full sm:w-auto px-7 py-3.5 bg-gray-100 dark:bg-white/5 hover:bg-gray-200 dark:hover:bg-white/10 text-gray-900 dark:text-white rounded-2xl font-bold transition-all border border-gray-200 dark:border-white/10 flex items-center justify-center gap-2">
              <Play className="w-4 h-4 fill-current" />
              Watch Demo
            </button>
          </motion.div>

          {/* Stat badges */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-4 pt-4"
          >
            {[
              { label: 'Response time', value: '< 2.4s' },
              { label: 'RAG accuracy', value: '99.2%' },
              { label: 'Active users', value: '10K+' },
            ].map(s => (
              <div key={s.label} className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/80 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-sm dark:shadow-none">
                <span className="text-sm font-black text-brand">{s.value}</span>
                <span className="text-xs text-gray-500">{s.label}</span>
              </div>
            ))}
          </motion.div>

          {/* Mockup UI */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="pt-8 max-w-3xl mx-auto"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-brand/20 blur-[60px] rounded-full pointer-events-none" />
              <div className="relative bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border shadow-2xl overflow-hidden animate-float">
                {/* Browser bar */}
                <div className="flex items-center gap-2 px-4 py-3 bg-gray-50 dark:bg-dark-bg border-b border-gray-200 dark:border-dark-border">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-400" />
                    <div className="w-3 h-3 rounded-full bg-yellow-400" />
                    <div className="w-3 h-3 rounded-full bg-green-400" />
                  </div>
                  <div className="flex-1 mx-3 h-6 bg-gray-200 dark:bg-white/5 rounded-md flex items-center px-3">
                    <span className="text-[11px] text-gray-400">https://finfriend.ai/chat</span>
                  </div>
                </div>
                {/* Chat preview */}
                <div className="p-4 sm:p-6 space-y-4 h-48">
                  <div className="flex gap-3 items-end">
                    <div className="w-7 h-7 bg-brand rounded-lg shrink-0" />
                    <div className="bg-gray-100 dark:bg-dark-border p-3 rounded-2xl rounded-bl-none max-w-xs">
                      <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-40 mb-2" />
                      <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-28" />
                    </div>
                  </div>
                  <div className="flex flex-row-reverse gap-3 items-end">
                    <div className="w-7 h-7 bg-brand rounded-lg shrink-0" />
                    <div className="bg-brand p-3 rounded-2xl rounded-br-none max-w-xs">
                      <div className="h-2.5 bg-white/20 rounded w-32 mb-2" />
                      <div className="h-2.5 bg-white/20 rounded w-44" />
                    </div>
                  </div>
                  <div className="flex gap-3 items-end">
                    <div className="w-7 h-7 bg-brand rounded-lg shrink-0" />
                    <div className="bg-gray-100 dark:bg-dark-border p-3 rounded-2xl rounded-bl-none max-w-sm">
                      <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-52 mb-2" />
                      <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-36" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* ── Social Proof Bar ────────────────────────────────────────────── */}
      <section className="py-10 border-y border-gray-100 dark:border-dark-border/50 bg-gray-50/50 dark:bg-dark-surface/30">
        <div className="max-w-5xl mx-auto px-4">
          <p className="text-center text-xs font-bold text-gray-400 uppercase tracking-widest mb-6">Trusted by individuals worldwide</p>
          <div className="flex flex-wrap justify-center items-center gap-6 sm:gap-12 opacity-40 grayscale hover:opacity-60 hover:grayscale-0 transition-all duration-500">
            {['FINANCEINSIDER', 'MONEYMINT', 'TECHTIMES', 'WEALTHWISE'].map(name => (
              <span key={name} className="text-lg sm:text-xl font-black text-gray-900 dark:text-white">{name}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────────────── */}
      <section id="features" className="py-20 sm:py-28 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center space-y-3 mb-16">
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="text-xs font-bold text-brand uppercase tracking-widest"
            >
              Why FinFriend
            </motion.p>
            <motion.h2
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-3xl sm:text-5xl font-extrabold text-gray-900 dark:text-white"
            >
              Smarter Financial <span className="text-brand">Conversations</span>
            </motion.h2>
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="max-w-xl mx-auto text-base text-gray-500 dark:text-gray-400"
            >
              Built with cutting-edge AI to provide institutional-grade insights for everyone.
            </motion.p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="p-6 rounded-2xl bg-white dark:bg-dark-surface border border-gray-100 dark:border-dark-border hover:border-brand/30 hover:shadow-lg hover:shadow-brand/5 transition-all group"
              >
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${f.color}`}>
                  {f.icon}
                </div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2 group-hover:text-brand transition-colors">
                  {f.title}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
                  {f.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it Works ───────────────────────────────────────────────── */}
      <section id="how-it-works" className="py-20 sm:py-28 bg-gray-50 dark:bg-dark-surface/40 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center space-y-3 mb-16">
            <p className="text-xs font-bold text-brand uppercase tracking-widest">Simple Steps</p>
            <h2 className="text-3xl sm:text-5xl font-extrabold text-gray-900 dark:text-white">How it Works</h2>
            <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto text-base">Three simple steps to financial clarity.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((item, i) => (
              <motion.div
                key={item.num}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.15 }}
                className="relative flex flex-col items-center text-center p-6 group"
              >
                {/* Step connector line (desktop) */}
                {i < steps.length - 1 && (
                  <div className="hidden md:block absolute top-[2.5rem] left-[60%] w-full h-px bg-gradient-to-r from-brand/30 to-transparent pointer-events-none" />
                )}
                <div className="w-16 h-16 bg-white dark:bg-dark-surface rounded-2xl border-2 border-gray-100 dark:border-dark-border flex items-center justify-center mb-5 shadow-lg group-hover:bg-brand group-hover:border-brand transition-all duration-300">
                  <div className="text-brand group-hover:text-white transition-colors">{item.icon}</div>
                </div>
                <span className="text-3xl font-black text-brand/20 mb-2">#{item.num}</span>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{item.title}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Testimonials ───────────────────────────────────────────────── */}
      <section id="testimonials" className="py-20 sm:py-28 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6 mb-14">
            <div className="space-y-2">
              <p className="text-xs font-bold text-brand uppercase tracking-widest">Testimonials</p>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white">
                People Love <span className="text-brand">FinFriend</span>
              </h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm max-w-md">
                Join thousands of users who have transformed their financial routine.
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex -space-x-3">
                {['U', 'U', 'U', 'U'].map((_, i) => (
                  <div key={i} className="w-10 h-10 rounded-full border-2 border-white dark:border-dark-bg bg-brand/10 flex items-center justify-center font-bold text-brand text-sm">U</div>
                ))}
              </div>
              <div>
                <p className="text-xs text-gray-500 font-medium">Avg response time</p>
                <p className="text-lg font-black text-brand">Under 2.4s</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {testimonials.map((t, i) => (
              <motion.div
                key={t.name}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -4 }}
                className="p-6 rounded-2xl bg-white dark:bg-dark-surface border border-gray-100 dark:border-dark-border shadow-sm flex flex-col gap-4"
              >
                <div className="flex gap-0.5">
                  {[...Array(5)].map((_, k) => (
                    <CheckCircle2 key={k} className="w-4 h-4 text-emerald-500" />
                  ))}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300 italic leading-relaxed flex-1">"{t.quote}"</p>
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-full bg-brand/10 flex items-center justify-center font-bold text-brand text-sm shrink-0">
                    {t.avatar}
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 dark:text-white">{t.name}</p>
                    <p className="text-xs text-gray-400">{t.role}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ─────────────────────────────────────────────────── */}
      <section className="px-4 py-16">
        <div className="max-w-5xl mx-auto">
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-600 via-brand to-purple-600 p-8 sm:p-14"
          >
            {/* Decorative */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/3 pointer-events-none" />
            <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/4 pointer-events-none" />

            <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-8">
              <div className="text-center lg:text-left space-y-3 max-w-xl">
                <h2 className="text-3xl sm:text-5xl font-black text-white leading-tight">
                  Ready to take control<br className="hidden sm:block" /> of your finances?
                </h2>
                <p className="text-indigo-100 text-base sm:text-lg">
                  Join FinFriend today and get your first AI-generated health report for free.
                </p>
              </div>
              <Link
                to="/register"
                className="shrink-0 flex items-center gap-2 px-8 py-4 bg-white text-indigo-700 hover:bg-gray-50 rounded-2xl font-black text-base sm:text-lg shadow-2xl transition-all hover:scale-105 active:scale-95"
              >
                Get Started Free
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className="py-10 px-4 border-t border-gray-100 dark:border-dark-border">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-brand rounded-lg flex items-center justify-center">
              <TrendingUp className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold text-gray-900 dark:text-white text-sm">FinFriend</span>
          </div>
          <div className="flex gap-6 text-sm font-medium text-gray-400">
            <a href="#" className="hover:text-brand transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-brand transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-brand transition-colors">GitHub</a>
          </div>
          <p className="text-xs text-gray-400">© 2024 FinFriend. Powered by Gemini.</p>
        </div>
      </footer>
    </div>
  );
}
