import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Link, useNavigate } from 'react-router-dom';
import { Loader2, Mail, Lock, User, Check, X, Eye, EyeOff } from 'lucide-react';
import { authApi } from '../api/client';
import { useAuthStore } from '../store/authStore';
import AuthLayout from '../components/auth/AuthLayout';
import toast from 'react-hot-toast';

const registerSchema = z
  .object({
    full_name: z.string().min(2, 'Name must be at least 2 characters'),
    email: z.string().email('Please enter a valid email address'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
      .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
      .regex(/\d/, 'Password must contain at least one digit'),
    confirmPassword: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ['confirmPassword'],
  });

type RegisterFormValues = z.infer<typeof registerSchema>;

export default function RegisterPage() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
  });

  const password = watch('password', '');

  useEffect(() => {
    let strength = 0;
    if (password.length >= 8) strength += 25;
    if (/[A-Z]/.test(password)) strength += 25;
    if (/[a-z]/.test(password)) strength += 25;
    if (/\d/.test(password)) strength += 25;
    setPasswordStrength(strength);
  }, [password]);

  const onSubmit = async (data: RegisterFormValues) => {
    setIsLoading(true);
    try {
      // 1. Register user
      await authApi.register(data.email, data.password, data.full_name);
      
      // 2. Auto-login after successful registration
      const authResponse = await authApi.login(data.email, data.password);
      login(authResponse);
      
      toast.success('Account created successfully! Welcome to FinFriend.');
      navigate('/chat', { replace: true });
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Registration failed. Please try again.';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const getStrengthColor = () => {
    if (passwordStrength <= 25) return 'bg-red-500';
    if (passwordStrength <= 50) return 'bg-yellow-500';
    if (passwordStrength <= 75) return 'bg-blue-500';
    return 'bg-emerald-500';
  };

  const getStrengthText = () => {
    if (passwordStrength <= 25) return 'Weak';
    if (passwordStrength <= 50) return 'Fair';
    if (passwordStrength <= 75) return 'Good';
    return 'Strong';
  };

  return (
    <AuthLayout
      title="Create account"
      subtitle="Join FinFriend and start your journey to financial freedom."
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">
            Full Name
          </label>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-brand transition-colors">
              <User className="w-5 h-5" />
            </div>
            <input
              {...register('full_name')}
              type="text"
              placeholder="John Doe"
              className={`w-full pl-10 pr-4 py-2.5 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.full_name
                  ? 'border-red-500'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand'
              } dark:text-white`}
            />
          </div>
          {errors.full_name && (
            <p className="text-xs text-red-500 font-medium ml-1">
              {errors.full_name.message}
            </p>
          )}
        </div>

        <div className="space-y-1.5">
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
              className={`w-full pl-10 pr-4 py-2.5 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.email
                  ? 'border-red-500'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand '
              } dark:text-white`}
            />
          </div>
          {errors.email && (
            <p className="text-xs text-red-500 font-medium ml-1">
              {errors.email.message}
            </p>
          )}
        </div>

        <div className="space-y-1.5">
          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">
            Password
          </label>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-brand transition-colors">
              <Lock className="w-5 h-5" />
            </div>
            <input
              {...register('password')}
              type={showPassword ? 'text' : 'password'}
              placeholder="••••••••"
              className={`w-full pl-10 pr-12 py-2.5 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.password
                  ? 'border-red-500'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand'
              } dark:text-white`}
            />
            <button
               type="button"
               onClick={() => setShowPassword(!showPassword)}
               className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
               {showPassword ? <EyeOff className="w-4 h-4"/> : <Eye className="w-4 h-4"/>}
            </button>
          </div>

          {/* Password Strength Meter */}
          <div className="mt-2 ml-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400">
                Security
              </span>
              <span className={`text-[10px] font-bold uppercase tracking-wider ${getStrengthColor().replace('bg-', 'text-')}`}>
                {getStrengthText()}
              </span>
            </div>
            <div className="h-1 w-full bg-gray-200 dark:bg-white/5 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${getStrengthColor()}`}
                style={{ width: `${passwordStrength}%` }}
              />
            </div>
          </div>
          
          <ul className="mt-2 space-y-1 ml-1">
             {[
                { label: '8+ Characters', met: password.length >= 8 },
                { label: 'Upper & Lowercase', met: /[A-Z]/.test(password) && /[a-z]/.test(password) },
                { label: 'At least one digit', met: /\d/.test(password) },
             ].map((rule) => (
                <li key={rule.label} className="flex items-center gap-2">
                   {rule.met ? <Check className="w-3 h-3 text-emerald-500"/> : <X className="w-3 h-3 text-gray-300"/>}
                   <span className={`text-[10px] font-medium ${rule.met ? 'text-gray-600 dark:text-gray-300' : 'text-gray-400'}`}>
                      {rule.label}
                   </span>
                </li>
             ))}
          </ul>
        </div>

        <div className="space-y-1.5 pb-2">
          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">
            Confirm Password
          </label>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-brand transition-colors">
              <Lock className="w-5 h-5" />
            </div>
            <input
              {...register('confirmPassword')}
              type="password"
              placeholder="••••••••"
              className={`w-full pl-10 pr-4 py-2.5 bg-white dark:bg-dark-surface border rounded-xl outline-none transition-all ${
                errors.confirmPassword
                  ? 'border-red-500'
                  : 'border-gray-200 dark:border-dark-border focus:border-brand '
              } dark:text-white`}
            />
          </div>
          {errors.confirmPassword && (
            <p className="text-xs text-red-500 font-medium ml-1">
              {errors.confirmPassword.message}
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
              Creating account...
            </>
          ) : (
            'Create Account'
          )}
        </button>

        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
           Already have an account?{' '}
           <Link
             to="/login"
             className="font-bold text-brand hover:text-brand-dark underline decoration-brand/20 underline-offset-4"
           >
             Login
           </Link>
        </p>
      </form>
    </AuthLayout>
  );
}
