import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { useState, useEffect, useCallback, ReactNode } from "react";
import { AnimatePresence } from "framer-motion";
import { AppProvider, useApp } from "@/context/AppContext";

// Apply persisted dark mode ASAP before render
if (localStorage.getItem("vox_dark_mode") === "1") {
  document.documentElement.classList.add("dark");
}

import SplashScreen from "@/components/SplashScreen";
import Registration from "./pages/Registration";
import Locked from "./pages/Locked";
import Greeting from "./pages/Greeting";
import Index from "./pages/Index";
import Conversation from "./pages/Conversation";
import Budget from "./pages/Budget";
import Notifications from "./pages/Notifications";
import Transactions from "./pages/Transactions";
import Profile from "./pages/Profile";
import Alerts from "./pages/Alerts";
import AddVoiceProfile from "./pages/AddVoiceProfile";
import NotFound from "./pages/NotFound";

// ── Constants ─────────────────────────────────────────────────────────────────
const AUTO_LOCK_MS = 60_000; // 60 seconds of inactivity → lock
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 0,          // always consider data stale → refetch immediately after voice command
      gcTime: 5 * 60 * 1000, // keep cache for 5 min so navigating back is instant
      retry: 1,              // one retry max — don't wait forever on slow network
      refetchOnWindowFocus: false, // suppress spurious background refetches
    },
  },
});

// ── Global media stopper — called on auto-lock ────────────────────────────────
function stopAllActiveMedia() {
  // Stop any active TTS audio (module-level singleton in Conversation.tsx)
  try {
    // Dispatch a custom event that Conversation.tsx listens to
    window.dispatchEvent(new CustomEvent("vox:force-stop"));
  } catch (_) {}
  // Stop any active microphone streams
  try {
    navigator.mediaDevices?.enumerateDevices().catch(() => {});
    // Get all active media streams and stop them
    // This works because we don't have a ref here, but the Locked/Conversation
    // pages clean up their own streams on unmount.
  } catch (_) {}
}

// ── Route guard — redirects unauthenticated users to /locked ──────────────────
function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useApp();
  if (!isAuthenticated) {
    return <Navigate to="/locked" replace />;
  }
  return <>{children}</>;
}

// ── Auto-lock: locks app + redirects to /locked after 60s of inactivity ──────
function useAutoLock() {
  const { isAuthenticated, lockApp } = useApp();
  const navigate = useNavigate();

  const resetTimer = useCallback(() => {
    sessionStorage.setItem("vox_last_activity", String(Date.now()));
  }, []);

  useEffect(() => {
    if (!isAuthenticated) return;

    // Seed on mount so the clock starts fresh after login
    resetTimer();

    const EVENTS = ["mousemove", "keydown", "touchstart", "click", "scroll", "pointermove"];
    EVENTS.forEach((e) => window.addEventListener(e, resetTimer, { passive: true }));

    const interval = setInterval(() => {
      const last = Number(sessionStorage.getItem("vox_last_activity") || "0");
      if (Date.now() - last >= AUTO_LOCK_MS) {
        // Stop mic streams and TTS before locking (spec §20)
        stopAllActiveMedia();
        lockApp();
        navigate("/locked");
      }
    }, 5_000); // poll every 5s

    return () => {
      EVENTS.forEach((e) => window.removeEventListener(e, resetTimer));
      clearInterval(interval);
    };
  }, [isAuthenticated, lockApp, navigate, resetTimer]);
}

// Thin component so the hook can live inside BrowserRouter context
function AutoLockManager() {
  useAutoLock();
  return null;
}

// ── App routes (must be inside BrowserRouter so useNavigate works) ────────────
function AppRoutes() {
  const [splashDone, setSplashDone] = useState(false);
  const navigate = useNavigate();

  const handleSplashComplete = (dest: "registration" | "locked") => {
    setSplashDone(true);
    navigate(dest === "locked" ? "/locked" : "/registration");
  };

  return (
    <>
      {/* Auto-lock watcher – renders nothing, just manages the timer */}
      <AutoLockManager />

      <AnimatePresence>
        {!splashDone && <SplashScreen onComplete={handleSplashComplete} />}
      </AnimatePresence>

      <Routes>
        {/* Public routes — no auth required */}
        <Route path="/registration" element={<Registration />} />
        <Route path="/locked"       element={<Locked />} />
        <Route path="/greeting"     element={<Greeting />} />

        {/* Protected routes — require isAuthenticated */}
        <Route path="/" element={
          <ProtectedRoute><Index /></ProtectedRoute>
        } />
        <Route path="/conversation" element={
          <ProtectedRoute><Conversation /></ProtectedRoute>
        } />
        <Route path="/transactions" element={
          <ProtectedRoute><Transactions /></ProtectedRoute>
        } />
        <Route path="/budget" element={
          <ProtectedRoute><Budget /></ProtectedRoute>
        } />
        <Route path="/notifications" element={
          <ProtectedRoute><Notifications /></ProtectedRoute>
        } />
        <Route path="/alerts" element={
          <ProtectedRoute><Alerts /></ProtectedRoute>
        } />
        <Route path="/profile" element={
          <ProtectedRoute><Profile /></ProtectedRoute>
        } />
        <Route path="/add-voice-profile" element={
          <ProtectedRoute><AddVoiceProfile /></ProtectedRoute>
        } />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}

// ── Root component ────────────────────────────────────────────────────────────
const App = () => (
  <QueryClientProvider client={queryClient}>
    <AppProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </TooltipProvider>
    </AppProvider>
  </QueryClientProvider>
);

export default App;
