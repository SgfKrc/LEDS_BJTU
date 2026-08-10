import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import AuthGate from './components/AuthGate.jsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthGate>
      {({ session, onLogout }) => (
        <App authSession={session} onLogout={onLogout} />
      )}
    </AuthGate>
  </React.StrictMode>,
)
