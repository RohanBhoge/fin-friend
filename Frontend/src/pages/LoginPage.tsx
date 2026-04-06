import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Eye, EyeOff, Loader2, Mail, Lock } from 'lucide-react';
import { authApi } from '../api/client';
import { useAuthStore } from '../store/authStore';
import AuthLayout from '../components/auth/AuthLayout';
import toast from 'react-hot-toast';

const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const login = useAuthStore((state) => state.login);
  
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormValues) => {
    setIsLoading(true);
    try {
      const response = await authApi.login(data.email, data.password);
      login(response);
      toast.success('Welcome back to FinFriend!');
      
      const from = (location.state as any)?.from?.pathname || '/chat';
      navigate(from, { replace: true });
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Invalid email or password';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Enter your credentials to access your financial dashboard."
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">
            Email Address
          </label>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-brand transition-colors">
              <Mail className="w-5 h-5" />
            </div>
            <input
              {...register('email')}
              type="email"
              placeholder="name@example.com"
              className={`w-full pl-10 pr-4 py-3 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.email
                  ? 'border-red-500 focus:ring-red-200'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand focus:ring-2 focus:ring-brand/10'
              } dark:text-white`}
            />
          </div>
          {errors.email && (
            <p className="text-xs text-red-500 font-medium ml-1">
              {errors.email.message}
            </p>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between ml-1">
            <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Password
            </label>
            <Link
              to="/forgot-password"
              className="text-xs font-semibold text-brand hover:text-brand-dark"
            >
              Forgot password?
            </Link>
          </div>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-brand transition-colors">
              <Lock className="w-5 h-5" />
            </div>
            <input
              {...register('password')}
              type={showPassword ? 'text' : 'password'}
              placeholder="••••••••"
              className={`w-full pl-10 pr-12 py-3 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.password
                  ? 'border-red-500 focus:ring-red-200'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand focus:ring-2 focus:ring-brand/10'
              } dark:text-white`}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
              {showPassword ? (
                <EyeOff className="w-5 h-5" />
              ) : (
                <Eye className="w-5 h-5" />
              )}
            </button>
          </div>
          {errors.password && (
            <p className="text-xs text-red-500 font-medium ml-1">
              {errors.password.message}
            </p>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3.5 bg-brand hover:bg-brand-dark text-white rounded-xl font-bold transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-brand/20 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Signing in...
            </>
          ) : (
            'Sign In'
          )}
        </button>

        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          Don't have an account?{' '}
          <Link
            to="/register"
            className="font-bold text-brand hover:text-brand-dark underline decoration-brand/20 underline-offset-4"
          >
            Create account
          </Link>
        </p>
      </form>
    </AuthLayout>
  );
}
