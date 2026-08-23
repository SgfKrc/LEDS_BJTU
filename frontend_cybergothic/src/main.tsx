import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './app/App';
import { reportClientError } from './data/api';
import './styles/tokens.css';
import './styles/global.css';
import './styles/components.css';
import './styles/overview.css';
import './styles/workbench.css';
import './styles/image-studio.css';
import './styles/models.css';
import './styles/settings-workspace.css';
import './styles/settings.css';
import './styles/help.css';
import './styles/cluster-admin.css';
import './styles/account.css';
import './styles/audit.css';
import './styles/activity.css';
import './styles/tasks.css';
import './styles/model-downloads.css';
import './styles/scene-surfaces.css';

function sendClientError(source: string, error: unknown, line = 0, col = 0) {
  const message = error instanceof Error ? error.message : String(error);
  void reportClientError({ source, message, stack: error instanceof Error ? error.stack || '' : '', url: window.location.href, line, col, user_agent: navigator.userAgent, extra: { route: window.location.hash } }).catch(() => undefined);
}

window.addEventListener('error', (event) => sendClientError('window.onerror', event.error || event.message, event.lineno, event.colno));
window.addEventListener('unhandledrejection', (event) => sendClientError('unhandledrejection', event.reason));

const container = document.getElementById('root');
if (!container) throw new Error('#root 未找到，index.html 可能被修改。');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
