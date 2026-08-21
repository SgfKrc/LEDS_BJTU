import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './app/App';
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

const container = document.getElementById('root');
if (!container) throw new Error('#root 未找到，index.html 可能被修改。');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
