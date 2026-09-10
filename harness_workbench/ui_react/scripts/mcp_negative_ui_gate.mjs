import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const here = resolve(fileURLToPath(new URL('.', import.meta.url)));
const uiRoot = resolve(here, '..');
const repositoryRoot = resolve(uiRoot, '..', '..');
const require = createRequire(import.meta.url);
const candidates = [
  process.env.HARNESS_PLAYWRIGHT,
  resolve(repositoryRoot, 'frontend_cybergothic/node_modules/playwright'),
  'playwright',
].filter(Boolean);

let playwright;
for (const candidate of candidates) {
  try {
    playwright = require(candidate);
    break;
  } catch {
    // The Harness UI stays independent; callers can provide Playwright explicitly.
  }
}
if (!playwright) {
  console.error('Playwright is required. Set HARNESS_PLAYWRIGHT or install it beside the CyberGothic frontend.');
  process.exit(2);
}

const tool = (name, schema, description = `${name} fixture`) => ({
  name,
  description,
  inputSchema: schema,
  annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  _meta: { qlh: { source: 'builtin', capability: 'configured', configured: true } },
});

const manifest = {
  schema: 'qlh.mcp_http_manifest.v1',
  server: { serverInfo: { name: 'qlh-harness-negative-fixture', version: '1.0.0' } },
  tools: [
    tool('rag_search', { type: 'object', properties: { query: { type: 'string', maxLength: 512 } }, required: ['query'], additionalProperties: false }),
    tool('web_search', { type: 'object', properties: { query: { type: 'string', maxLength: 512 } }, required: ['query'], additionalProperties: false }),
    tool('stale_tool', { type: 'object', properties: { query: { type: 'string', maxLength: 512 } }, required: ['query'], additionalProperties: false }),
  ],
  external_mcp: { configurations: [], configuration_only: true, real_connections: false },
  transports: {
    jsonrpc: { method: 'POST', path: '/v1/mcp/rpc' },
    call: { method: 'POST', path: '/v1/mcp/call' },
    stdio: { available: true, command: 'python -m harness_workbench.mcp_server' },
    sse: { available: true, mode: 'loopback_standalone', path: '/sse' },
  },
};

function json(body, status = 200) {
  return { status, contentType: 'application/json', body: JSON.stringify(body) };
}

function mcpToolError(code, message, retryable = false) {
  return {
    jsonrpc: '2.0',
    id: 'negative-fixture',
    result: {
      content: [{ type: 'text', text: JSON.stringify({ schema: 'qlh.mcp_tool_result.v1', error: { code, message, retryable } }) }],
      isError: true,
    },
  };
}

async function fulfillMcpCall(route) {
  const request = JSON.parse(route.request().postData() || '{}');
  const { name, arguments: args = {} } = request;
  if (name === 'web_search') {
    return route.fulfill(json({ error: { message: 'network scope is denied', code: 'scope_denied', retryable: false } }, 403));
  }
  if (name === 'stale_tool') {
    return route.fulfill(json(mcpToolError('unknown_tool', 'MCP tool is not registered')));
  }
  if (name === 'rag_search' && args.query === 'bad-parameter') {
    return route.fulfill(json({ error: { message: 'query must be a string', code: 'invalid_params', retryable: false } }, 400));
  }
  if (name === 'rag_search' && args.query === 'backend-error') {
    return route.fulfill(json(mcpToolError('backend_busy', 'RAG backend is temporarily unavailable', true)));
  }
  return route.fulfill(json({ jsonrpc: '2.0', id: 'negative-fixture', result: { content: [{ type: 'text', text: '{"hits":[]}' }], isError: false } }));
}

async function mockApi(page) {
  await page.route('**/healthz', (route) => route.fulfill(json({ status: 'ok', backend: 'negative-ui-fixture' })));
  await page.route('**/v1/models', (route) => route.fulfill(json({ object: 'list', data: [{ id: 'fixture-chat', available: true, owned_by: 'negative-ui-fixture' }] })));
  await page.route('**/v1/model-profiles', (route) => route.fulfill(json({ schema: 'fixture', profiles: [] })));
  await page.route('**/v1/model-presets', (route) => route.fulfill(json({ presets: [] })));
  await page.route('**/v1/model-downloads', (route) => route.fulfill(json({ jobs: [] })));
  await page.route('**/v1/sessions**', (route) => {
    if (route.request().method() === 'POST' && route.request().url().endsWith('/v1/sessions')) {
      return route.fulfill(json({ session_id: 'sess_negative_fixture', owner_scope: 'local', title: 'Negative fixture', created_at: 0, updated_at: 0 }));
    }
    if (route.request().method() === 'GET') return route.fulfill(json({ sessions: [] }));
    return route.fulfill(json({ ok: true }));
  });
  await page.route('**/v1/rag/health', (route) => route.fulfill(json({ backend: 'sqlite_fts5', chunks: 0 })));
  await page.route('**/v1/images/capabilities', (route) => route.fulfill(json({ backend: 'negative-ui-fixture', model_ids: [], supports_txt2img: false, supports_edit: false, supports_distributed: false, runtime_available: false, evidence: {} })));
  await page.route('**/v1/mcp/manifest', (route) => route.fulfill(json(manifest)));
  await page.route('**/v1/mcp/call', fulfillMcpCall);
  await page.route('**/v1/chat/completions', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"error":{"message":"stream service unavailable","code":"stream_unavailable","retryable":true}}\n\n' }));
}

async function selectTool(page, name) {
  await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
  await page.getByRole('heading', { name }).waitFor();
}

async function callTool(page, argumentsValue) {
  await page.locator('#mcp-arguments').fill(argumentsValue);
  await page.getByRole('button', { name: '调用工具' }).click();
}

async function requireNotice(page, expected) {
  await page.waitForFunction((text) => {
    const notice = document.querySelector('[data-testid="mcp-notice"]');
    return notice?.textContent?.includes(text);
  }, expected);
}

const vite = await createServer({
  configFile: resolve(uiRoot, 'vite.config.ts'),
  logLevel: 'error',
  server: { host: '127.0.0.1', port: 0, strictPort: false },
});
await vite.listen();
const baseUrl = vite.resolvedUrls?.local?.[0];
if (!baseUrl) throw new Error('Vite did not expose a local URL');

const browser = await playwright.chromium.launch({ channel: 'msedge', headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  await mockApi(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'MCP 控制面' }).click();
  await page.getByRole('heading', { name: 'MCP 控制面' }).waitFor();

  await selectTool(page, 'web_search');
  await callTool(page, '{"query":"policy"}');
  await requireNotice(page, 'network scope is denied [scope_denied] · HTTP 403');
  if (await page.getByTestId('mcp-result').count()) throw new Error('permission HTTP error must not render a successful MCP result');

  await selectTool(page, 'stale_tool');
  await callTool(page, '{"query":"stale"}');
  await requireNotice(page, 'MCP tool is not registered [unknown_tool]');
  await page.getByTestId('mcp-result').getByText('unknown_tool').waitFor();

  await selectTool(page, 'rag_search');
  await callTool(page, '{not json');
  await requireNotice(page, '参数 JSON 无效');

  await callTool(page, '{"query":"bad-parameter"}');
  await requireNotice(page, 'query must be a string [invalid_params] · HTTP 400');

  await callTool(page, '{"query":"backend-error"}');
  await requireNotice(page, 'RAG backend is temporarily unavailable [backend_busy] · 可重试');
  await page.getByTestId('mcp-result').getByText('backend_busy').waitFor();

  await page.getByRole('button', { name: '对话' }).click();
  await page.getByRole('textbox', { name: '消息输入' }).fill('trigger stream failure');
  await page.getByRole('button', { name: '发送消息' }).click();
  await page.getByText('请求失败：stream service unavailable').waitFor();
  await page.getByRole('status').getByText('ONLINE').waitFor();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  if (overflow) throw new Error('negative-state UI has horizontal overflow');
  console.log(JSON.stringify({ status: 'passed', cases: ['permission', 'unknown_tool', 'invalid_arguments', 'backend_error', 'sse_error'] }));
  await context.close();
} finally {
  await browser.close();
  await vite.close();
}
