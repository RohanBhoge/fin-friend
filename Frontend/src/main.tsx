import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// ── StrictMode ───────────────────────────────────────────────────────────────
// We use StrictMode to help identify potential problems during development.
// It will intentionally double-invoke functional components to ensure purity.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
