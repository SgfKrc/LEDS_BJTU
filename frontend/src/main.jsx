import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import AuthGate from './components/AuthGate.jsx'
import './App.css'

const storedTheme = (() => {
  try { return localStorage.getItem('qlh-theme') } catch (_) { return null }
})()
const themeMode = ['light', 'dark', 'system'].includes(storedTheme) ? storedTheme : 'system'
const initialTheme = themeMode === 'system'
  ? (window.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light')
  : themeMode
document.documentElement.setAttribute('data-theme', initialTheme)
document.documentElement.setAttribute('data-theme-mode', themeMode)

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthGate>
      {({ session, onLogout }) => (
        <App authSession={session} onLogout={onLogout} />
      )}
    </AuthGate>
  </React.StrictMode>,
)
