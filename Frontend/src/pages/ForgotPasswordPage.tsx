import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Link } from 'react-router-dom';
import { Loader2, Mail, ArrowLeft, CheckCircle2 } from 'lucide-react';
import { authApi } from '../api/client';
import AuthLayout from '../components/auth/AuthLayout';
import toast from 'react-hot-toast';

const forgotPasswordSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
});

type ForgotPasswordFormValues = z.infer<typeof forgotPasswordSchema>;

export default function ForgotPasswordPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [isSent, setIsSent] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ForgotPasswordFormValues>({
    resolver: zodResolver(forgotPasswordSchema),
  });

  const onSubmit = async (data: ForgotPasswordFormValues) => {
    setIsLoading(true);
    try {
      await authApi.forgotPassword(data.email);
      setIsSent(true);
      toast.success('Reset link sent! Please check your email.');
    } catch (error: any) {
      // For security, we might not want to reveal if email exists, 
      // but the backend is already configured to always return 200.
      setIsSent(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Reset password"
      subtitle="Enter your email address and we'll send you a link to reset your password."
    >
      {isSent ? (
        <div className="space-y-6 text-center animate-in fade-in zoom-in duration-300">
          <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
             <CheckCircle2 className="w-8 h-8 text-emerald-500" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">Check your email</h3>
            <p className="text-gray-500 dark:text-gray-400">
              If an account exists for that email, we've sent instructions to reset your password.
            </p>
          </div>
          <Link
            to="/login"
            className="inline-flex items-center gap-2 text-brand font-bold hover:text-brand-dark transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to login
          </Link>
        </div>
      ) : (
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

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3.5 bg-brand hover:bg-brand-dark text-white rounded-xl font-bold transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-brand/20 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Sending link...
              </>
            ) : (
              'Send Reset Link'
            )}
          </button>

          <Link
            to="/login"
            className="flex items-center justify-center gap-2 text-sm font-bold text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to login
          </Link>
        </form>
      )}
    </AuthLayout>
  );
}
