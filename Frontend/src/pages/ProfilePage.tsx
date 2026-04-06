import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { useNavigate } from 'react-router-dom';
import {
  User,
  Lock,
  ArrowLeft,
  Settings,
  Shield,
  Trash2,
  Loader2,
  CheckCircle2,
  LogOut,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { usersApi } from '../api/client';
import toast from 'react-hot-toast';

const profileSchema = z.object({
  full_name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Please enter a valid email address'),
});

const passwordSchema = z.object({
  current_password: z.string().min(1, 'Current password is required'),
  new_password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/\d/, 'Password must contain at least one digit'),
  confirm_password: z.string().min(1, 'Please confirm your password'),
}).refine(data => data.new_password === data.confirm_password, {
  message: "Passwords don't match",
  path: ["confirm_password"],
});

type ProfileFormValues = z.infer<typeof profileSchema>;
type PasswordFormValues = z.infer<typeof passwordSchema>;

export default function ProfilePage() {
  const navigate = useNavigate();
  const { user, setUser, logout } = useAuthStore();
  const [activeTab, setActiveTab] = useState<'info' | 'security' | 'danger'>('info');
  const [isLoading, setIsLoading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');

  const {
    register: registerProfile,
    handleSubmit: handleSubmitProfile,
    formState: { errors: profileErrors },
  } = useForm<ProfileFormValues>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      full_name: user?.full_name || '',
      email: user?.email || '',
    },
  });

  const {
    register: registerPassword,
    handleSubmit: handleSubmitPassword,
    reset: resetPasswordForm,
    formState: { errors: passwordErrors },
  } = useForm<PasswordFormValues>({
    resolver: zodResolver(passwordSchema),
  });

  const onUpdateProfile = async (data: ProfileFormValues) => {
    setIsLoading(true);
    try {
      const updatedUser = await usersApi.updateProfile(data);
      setUser(updatedUser);
      toast.success('Profile updated successfully');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to update profile');
    } finally {
      setIsLoading(false);
    }
  };

  const onUpdatePassword = async (data: PasswordFormValues) => {
    setIsLoading(true);
    try {
      await usersApi.updateSecurity(data.current_password, data.new_password);
      toast.success('Password changed successfully');
      resetPasswordForm();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to change password');
    } finally {
      setIsLoading(false);
    }
  };

  const onDeleteAccount = async () => {
    if (!deletePassword) return;
    setIsLoading(true);
    try {
      await usersApi.deleteAccount(deletePassword);
      toast.success('Account deleted successfully');
      logout();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete account');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg transition-colors duration-300">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={() => navigate('/chat')}
            className="flex items-center gap-2 text-gray-500 hover:text-brand transition-colors font-medium"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Chat
          </button>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-brand rounded-xl flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-gray-900 dark:text-white">FinFriend Profile</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar Tabs */}
          <div className="lg:col-span-1 space-y-1">
            {[
              { id: 'info', label: 'Profile Info', icon: <User className="w-4 h-4" /> },
              { id: 'security', label: 'Security', icon: <Shield className="w-4 h-4" /> },
              { id: 'danger', label: 'Danger Zone', icon: <Trash2 className="w-4 h-4" /> },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-semibold transition-all ${
                  activeTab === tab.id
                    ? 'bg-brand text-white shadow-lg shadow-brand/20'
                    : 'text-gray-500 hover:bg-gray-200 dark:hover:bg-dark-surface'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
            <button
              onClick={logout}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-xl font-semibold text-red-500 hover:bg-red-50 dark:hover:bg-red-900/10 mt-4"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-dark-surface rounded-3xl p-8 border border-gray-200 dark:border-dark-border shadow-sm">
              {activeTab === 'info' && (
                <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                  <div className="flex items-center gap-6 pb-6 border-b border-gray-100 dark:border-dark-border">
                    <div className="w-20 h-20 bg-brand/10 rounded-3xl flex items-center justify-center font-black text-2xl text-brand border border-brand/20">
                       {user?.full_name?.[0] || user?.email?.[0]}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white">Profile Details</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Update your personal information and contact details.</p>
                    </div>
                  </div>

                  <form onSubmit={handleSubmitProfile(onUpdateProfile)} className="space-y-6">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">Full Name</label>
                        <input
                          {...registerProfile('full_name')}
                          className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-brand outline-none transition-all dark:text-white"
                        />
                        {profileErrors.full_name && <p className="text-xs text-red-500 font-medium">{profileErrors.full_name.message}</p>}
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">Email Address</label>
                        <input
                          {...registerProfile('email')}
                          className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-brand outline-none transition-all dark:text-white"
                        />
                        {profileErrors.email && <p className="text-xs text-red-500 font-medium">{profileErrors.email.message}</p>}
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={isLoading}
                      className="px-8 py-3 bg-brand text-white font-bold rounded-xl shadow-lg shadow-brand/20 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center gap-2 disabled:opacity-70"
                    >
                      {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <CheckCircle2 className="w-5 h-5" />}
                      Save Changes
                    </button>
                  </form>
                </div>
              )}

              {activeTab === 'security' && (
                <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                  <div className="flex items-center gap-6 pb-6 border-b border-gray-100 dark:border-dark-border">
                    <div className="w-20 h-20 bg-indigo-100 dark:bg-indigo-900/20 rounded-3xl flex items-center justify-center text-brand border border-brand/20">
                       <Shield className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white">Security Settings</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Manage your password and account security preferences.</p>
                    </div>
                  </div>

                  <form onSubmit={handleSubmitPassword(onUpdatePassword)} className="space-y-6">
                    <div className="space-y-4">
                       <div className="space-y-2">
                        <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">Current Password</label>
                        <input
                          {...registerPassword('current_password')}
                          type="password"
                          className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-brand outline-none transition-all dark:text-white"
                        />
                        {passwordErrors.current_password && <p className="text-xs text-red-500 font-medium">{passwordErrors.current_password.message}</p>}
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">New Password</label>
                          <input
                            {...registerPassword('new_password')}
                            type="password"
                            className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-brand outline-none transition-all dark:text-white"
                          />
                          {passwordErrors.new_password && <p className="text-xs text-red-500 font-medium">{passwordErrors.new_password.message}</p>}
                        </div>
                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-gray-700 dark:text-gray-300 ml-1">Confirm New Password</label>
                          <input
                            {...registerPassword('confirm_password')}
                            type="password"
                            className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-brand outline-none transition-all dark:text-white"
                          />
                          {passwordErrors.confirm_password && <p className="text-xs text-red-500 font-medium">{passwordErrors.confirm_password.message}</p>}
                        </div>
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={isLoading}
                      className="px-8 py-3 bg-brand text-white font-bold rounded-xl shadow-lg shadow-brand/20 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center gap-2 disabled:opacity-70"
                    >
                      {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Lock className="w-5 h-5" />}
                      Update Password
                    </button>
                  </form>
                </div>
              )}

              {activeTab === 'danger' && (
                <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                  <div className="flex items-center gap-6 pb-6 border-b border-gray-100 dark:border-dark-border">
                    <div className="w-20 h-20 bg-red-100 dark:bg-red-900/20 rounded-3xl flex items-center justify-center text-red-500 border border-red-500/20">
                       <Trash2 className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-red-500">Danger Zone</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Irreversible actions associated with your account.</p>
                    </div>
                  </div>

                  <div className="p-6 border border-red-200 dark:border-red-900/30 bg-red-50/50 dark:bg-red-950/10 rounded-2xl">
                     <p className="text-sm text-gray-700 dark:text-gray-300 font-medium mb-6">
                        Once you delete your account, there is no going back. This will permanently remove all your conversations, personal data, and analytical reports.
                     </p>
                     <button
                        onClick={() => setShowDeleteModal(true)}
                        className="px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white font-bold rounded-xl transition-all shadow-lg shadow-red-500/20"
                     >
                        Delete My Account
                     </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Delete Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
           <div className="bg-white dark:bg-dark-surface w-full max-w-md rounded-3xl p-8 shadow-2xl border border-gray-200 dark:border-dark-border animate-in zoom-in-95 duration-200">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Are you absolutely sure?</h3>
              <p className="text-gray-500 dark:text-gray-400 mb-6 font-medium">Please enter your password to confirm account deletion.</p>
              
              <input
                type="password"
                placeholder="Enter password"
                value={deletePassword}
                onChange={(e) => setDeletePassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-input border border-gray-200 dark:border-dark-border rounded-xl focus:border-red-500 outline-none transition-all dark:text-white mb-6"
              />

              <div className="flex gap-3">
                 <button
                    onClick={() => setShowDeleteModal(false)}
                    className="flex-1 py-3 bg-gray-100 dark:bg-white/5 text-gray-700 dark:text-white font-bold rounded-xl hover:bg-gray-200 transition-all font-sans"
                 >
                    Cancel
                 </button>
                 <button
                    onClick={onDeleteAccount}
                    disabled={!deletePassword || isLoading}
                    className="flex-1 py-3 bg-red-500 text-white font-bold rounded-xl hover:bg-red-600 transition-all shadow-lg shadow-red-500/20 disabled:opacity-50"
                 >
                    {isLoading ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : 'Delete Account'}
                 </button>
              </div>
           </div>
        </div>
      )}
    </div>
  );
}
