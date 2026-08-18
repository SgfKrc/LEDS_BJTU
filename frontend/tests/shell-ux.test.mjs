import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import test from 'node:test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const appSource = readFileSync(path.join(ROOT, 'src', 'App.jsx'), 'utf-8');
const cssSource = readFileSync(path.join(ROOT, 'src', 'App.css'), 'utf-8');
const adminSource = readFileSync(path.join(ROOT, 'src', 'components', 'AdminPanel.jsx'), 'utf-8');
const settingsSource = readFileSync(path.join(ROOT, 'src', 'components', 'SettingsModal.jsx'), 'utf-8');
const diffusionSource = readFileSync(path.join(ROOT, 'src', 'components', 'DiffusionPanel.jsx'), 'utf-8');
const chatSource = readFileSync(path.join(ROOT, 'src', 'components', 'ChatPanel.jsx'), 'utf-8');
const accountSource = readFileSync(path.join(ROOT, 'src', 'components', 'UserManagementPanel.jsx'), 'utf-8');
const authSource = readFileSync(path.join(ROOT, 'src', 'components', 'AuthGate.jsx'), 'utf-8');
const mainSource = readFileSync(path.join(ROOT, 'src', 'main.jsx'), 'utf-8');

test('UX-01 uses accessible line icons in the primary navigation instead of emoji', () => {
  assert.match(appSource, /from 'lucide-react'/);
  const navigation = appSource.match(/<nav className="nav-tabs"[\s\S]*?<\/nav>/)?.[0] || '';
  assert.match(navigation, /MessageSquare/);
  assert.match(navigation, /LayoutDashboard/);
  assert.match(navigation, /aria-label="主导航"/);
  assert.doesNotMatch(navigation, /[💬⚙️🧠🔧]/u);
});

test('UX-01 exposes engineering tokens and a compact sidebar layout', () => {
  for (const token of ['--accent-soft:', '--border-strong:', '--surface-raised:']) {
    assert.match(cssSource, new RegExp(token));
  }
  assert.match(cssSource, /\.nav-tabs\s*\{\s*display: grid;/);
  assert.match(cssSource, /\.sidebar-system-status\s*\{/);
  assert.match(cssSource, /@media \(max-width: 720px\)/);
});

test('UX-02 groups the administration surface into accessible workspaces', () => {
  assert.match(adminSource, /const MASTER_WORKSPACES/);
  assert.match(adminSource, /role="tablist" aria-label="后台管理工作区"/);
  assert.match(adminSource, /data-admin-workspace="overview nodes"/);
  assert.match(adminSource, /data-admin-workspace="availability"/);
  assert.match(cssSource, /\.admin-content\[data-active-workspace\] \.admin-section\[data-admin-workspace\]/);
  assert.match(cssSource, /data-active-workspace="runtime"/);
});

test('UX-02 preserves neutral black and white theme tokens', () => {
  assert.match(cssSource, /--bg-primary: #0b0b0c;/);
  assert.match(cssSource, /--accent: #f5f5f5;/);
  assert.match(cssSource, /\[data-theme="light"\][\s\S]*?--bg-primary: #f7f7f7;/);
  assert.match(cssSource, /\[data-theme="light"\][\s\S]*?--accent: #111111;/);
});

test('UX-03 partitions settings, logs, and image assets into focused workspaces', () => {
  assert.match(settingsSource, /const SETTINGS_WORKSPACES/);
  assert.match(settingsSource, /aria-label="系统设置工作区"/);
  assert.match(settingsSource, /aria-label="日志工作区"/);
  assert.match(settingsSource, /data-settings-workspace="logs"/);
  assert.match(diffusionSource, /data-testid="diffusion-workspace-assets"/);
  assert.match(diffusionSource, /activeDiffusionWorkspace === 'assets'/);
  assert.match(cssSource, /\.diffusion-layout-grid\.workspace-assets/);
  assert.match(cssSource, /grid-template-columns: repeat\(auto-fit, minmax\(250px, 1fr\)\)/);
});

test('UX-03B uses line icons for chat tools and focused workspaces for local identity', () => {
  assert.match(chatSource, /from 'lucide-react'/);
  assert.match(chatSource, /<Settings2/);
  assert.match(chatSource, /<Send/);
  assert.match(chatSource, /<MessageAvatar/);
  assert.doesNotMatch(chatSource, /[💬🧠👤🤖⚠️📎🖼🪙💾⏱⚙️📄]/u);
  assert.match(accountSource, /const ACCOUNT_WORKSPACES/);
  assert.match(accountSource, /aria-label="账户与安全工作区"/);
  assert.match(accountSource, /activeWorkspace === 'network'/);
  assert.match(accountSource, /activeWorkspace === 'users'/);
  assert.match(authSource, /ShieldCheck/);
  assert.match(mainSource, /data-theme-mode/);
  assert.match(cssSource, /\.auth-shell[\s\S]*?background: var\(--bg-primary\);/);
});

test('UX-07 keeps avatars local and maps the model mark by theme', () => {
  assert.match(appSource, /qlhDarkLogo from '..\/..\/qlh\.jpg'/);
  assert.match(appSource, /qlhLightLogo from '..\/..\/qlh-light\.jpg'/);
  assert.match(appSource, /modelAvatarUrl = theme === 'dark' \? qlhDarkLogo : qlhLightLogo/);
  assert.match(appSource, /readStoredUserAvatar/);
  assert.match(chatSource, /userAvatar, modelAvatarUrl/);
  assert.match(chatSource, /message-avatar-image/);
  assert.match(settingsSource, /data-testid="chat-user-avatar-input"/);
  assert.match(settingsSource, /本机浏览器/);
  assert.match(cssSource, /\.file-upload-btn:disabled[\s\S]*?opacity: 1;/);
  assert.match(cssSource, /\[data-theme="light"\] \.file-upload-btn:disabled/);
});

test('UX-03R2 keeps execution-mode hover contrast and exposes the brand account entry', () => {
  assert.match(appSource, /className="brand-mark brand-account-entry"/);
  assert.match(appSource, /aria-label="打开账户与安全"/);
  assert.match(appSource, /className="account-legacy-panel"/);
  assert.match(appSource, /认证服务尚未启用/);
  assert.match(settingsSource, /Sidecar 预检/);
  assert.match(settingsSource, /尚未启动模型加载/);
  assert.match(settingsSource, /aria-label="Sidecar 运行时控制面"/);
  assert.match(settingsSource, /requires_task_contract/);
  assert.match(cssSource, /\.brand-account-entry:hover,[\s\S]*?color: var\(--text-primary\);/);
  assert.match(cssSource, /\.execution-mode-segment button\.active:hover:not\(:disabled\)[\s\S]*?color: var\(--on-accent\);/);
  assert.match(cssSource, /\.execution-mode-segment button:hover:not\(:disabled\)[\s\S]*?background: var\(--bg-secondary\);/);
});

test('admin node RTT never uses heartbeat freshness as a latency measurement', () => {
  assert.match(adminSource, /rttMs = node\.avg_rtt_ms > 0 \? node\.avg_rtt_ms : node\.last_rtt_ms/);
  assert.doesNotMatch(adminSource, /const age = Date\.now\(\) \/ 1000 - tcpDetail\.last_heartbeat/);
  assert.doesNotMatch(adminSource, /rttMs = age \* 1000/);
});
