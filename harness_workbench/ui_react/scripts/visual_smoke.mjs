import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const require = createRequire(import.meta.url);
const candidates = [
  process.env.HARNESS_PLAYWRIGHT,
  resolve(process.cwd(), '../../frontend_cybergothic/node_modules/playwright'),
  'playwright',
].filter(Boolean);

let playwright;
for (const candidate of candidates) {
  try {
    playwright = require(candidate);
    break;
  } catch {
    // The package remains independent; a caller may provide Playwright explicitly.
  }
}
if (!playwright) {
  console.error('Playwright is required for visual smoke. Set HARNESS_PLAYWRIGHT or install playwright.');
  process.exit(2);
}

const baseUrl = process.argv[2] || 'http://127.0.0.1:5181/';
const outputDir = resolve(process.cwd(), 'build/ui-visual-smoke');
mkdirSync(outputDir, { recursive: true });
const browser = await playwright.chromium.launch({ headless: true });
const viewports = [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }];

try {
  for (const viewport of viewports) {
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    await page.route('**/healthz', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', backend: 'visual-fixture' }) }));
    await page.route('**/v1/sessions**', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) }));
    await page.route('**/v1/models', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ object: 'list', data: [{ id: 'Qwen2.5-0.5B', object: 'model', owned_by: 'visual-fixture' }] }) }));
    await page.route('**/v1/model-profiles', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ schema: 'qlh.harness.model_profiles.v1', profiles: [{ model_id: 'Qwen2.5-0.5B', revision: 'fixture-v1', backend: 'llama_server', roles: ['answer'], status: 'candidate', production_eligible: false, context: {}, generation: {}, adaptation: {}, capabilities: {}, evidence: {} }] }) }));
    await page.route('**/v1/rag/health', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ backend: 'sqlite_fts5', chunks: 3 }) }));
    await page.route('**/v1/images/capabilities', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ backend: 'visual-fixture', model_ids: [], supports_txt2img: false, supports_edit: false, supports_distributed: false, runtime_available: false, evidence: {} }) }));
    await page.route('**/v1/mcp/manifest', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      schema: 'qlh.mcp_http_manifest.v1',
      server: { protocolVersion: '2025-06-18', serverInfo: { name: 'qlh-harness', version: '1.0.0' }, capabilities: { tools: { listChanged: false } } },
      tools: [
        { name: 'rag_search', description: 'Search user-owned RAG chunks and return bounded context and citations.', inputSchema: { type: 'object', properties: { query: { type: 'string', maxLength: 512 } }, required: ['query'], additionalProperties: false }, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false }, _meta: { qlh: { source: 'builtin', capability: 'configured', configured: true } } },
        { name: 'session_create', description: 'Create a user-owned harness session.', inputSchema: { type: 'object', properties: { title: { type: 'string', maxLength: 200 } }, required: [], additionalProperties: false }, annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false }, _meta: { qlh: { source: 'builtin', capability: 'configured', configured: true } } },
        { name: 'web_search', description: 'Search through the configured, policy-gated web search provider.', inputSchema: { type: 'object', properties: { query: { type: 'string', maxLength: 512 } }, required: ['query'], additionalProperties: false }, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: true }, _meta: { qlh: { source: 'builtin', capability: 'unconfigured', configured: false } } },
      ],
      external_mcp: { configurations: [{ server_id: 'example', transport: 'sse', enabled: false }], configuration_only: true, real_connections: false },
      transports: { jsonrpc: { method: 'POST', path: '/v1/mcp/rpc' }, call: { method: 'POST', path: '/v1/mcp/call' }, stdio: { available: true, command: 'python -m harness_workbench.mcp_server' }, sse: { available: true, mode: 'loopback_standalone', path: '/sse' } },
    }) }));
    await page.route('**/v1/mcp/call', async (route) => {
      const request = JSON.parse(route.request().postData() || '{}');
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ jsonrpc: '2.0', id: 'fixture', result: { content: [{ type: 'text', text: JSON.stringify({ query: request.arguments?.query || '', hits: [], context: { included_count: 0, omitted_count: 0 } }) }], isError: false } }) });
    });
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(200);

    const initial = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth > window.innerWidth,
      unnamedButtons: [...document.querySelectorAll('button')].filter((button) => !button.getAttribute('aria-label') && !button.textContent?.trim()).length,
      theme: document.documentElement.dataset.theme,
    }));
    if (initial.overflow) throw new Error(`${viewport.name}: horizontal overflow`);
    if (initial.unnamedButtons > 0) throw new Error(`${viewport.name}: unnamed icon button count=${initial.unnamedButtons}`);
    if (await page.locator('[aria-label="选择对话模型"]').count() !== 1) throw new Error(`${viewport.name}: live model selector did not render`);

    if (viewport.name === 'mobile') await page.getByRole('button', { name: '打开导航' }).click();
    await page.getByRole('button', { name: '知识库' }).click();
    await page.waitForTimeout(40);
    const navigation = await page.evaluate(() => ({
      title: document.querySelector('#rag-title')?.textContent,
      focusTarget: document.activeElement?.id,
      reducedTransition: getComputedStyle(document.querySelector('.sidebar')).transitionDuration,
      theme: document.documentElement.dataset.theme,
      overflow: document.documentElement.scrollWidth > window.innerWidth,
    }));
    if (navigation.title !== '知识库检索') throw new Error(`${viewport.name}: navigation target did not render`);
    if (navigation.focusTarget !== 'main-content') throw new Error(`${viewport.name}: main content did not receive focus`);
    if (navigation.overflow) throw new Error(`${viewport.name}: overflow after navigation`);
    if (navigation.reducedTransition !== '0s') throw new Error(`${viewport.name}: reduced-motion transition is ${navigation.reducedTransition}`);

    if (viewport.name === 'mobile') await page.getByRole('button', { name: '打开导航' }).click();
    await page.getByRole('button', { name: 'MCP 控制面' }).click();
    await page.waitForTimeout(40);
    if (await page.locator('#mcp-title').textContent() !== 'MCP 控制面') throw new Error(`${viewport.name}: MCP view did not render`);
    if (await page.getByRole('button', { name: '调用工具' }).isDisabled()) throw new Error(`${viewport.name}: configured MCP tool is not callable`);
    await page.getByRole('button', { name: '调用工具' }).click();
    await page.waitForTimeout(40);
    if (!(await page.locator('.mcp-result').count())) throw new Error(`${viewport.name}: MCP result did not render`);
    const mcpOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    if (mcpOverflow) throw new Error(`${viewport.name}: overflow in MCP view`);

    if (viewport.name === 'mobile') await page.getByRole('button', { name: '打开导航' }).click();
    await page.getByRole('button', { name: '运行时' }).click();
    await page.waitForTimeout(40);
    if (await page.locator('#runtime-title').textContent() !== '运行时能力') throw new Error(`${viewport.name}: runtime view did not render`);
    if (!(await page.locator('.profile-row').count())) throw new Error(`${viewport.name}: model profile registry did not render`);
    if (await page.locator('[aria-label="选择对话模型"]').count() !== 0) throw new Error(`${viewport.name}: chat-only model selector leaked into runtime view`);

    await page.getByRole('button', { name: '切换主题' }).click();
    const lightTheme = await page.evaluate(() => document.documentElement.dataset.theme);
    if (lightTheme !== 'light') throw new Error(`${viewport.name}: light theme toggle failed`);
    await page.screenshot({ path: resolve(outputDir, `${viewport.name}.png`), fullPage: true });
    console.log(JSON.stringify({ viewport: viewport.name, ...initial, ...navigation, lightTheme }));
    await context.close();
  }
} finally {
  await browser.close();
}
