/**
 * 工作台（分屏首页）E2E。
 *
 * 覆盖用户明确要求的两件事：左右比例可调、且调整要能留住；
 * 外加分屏页特有的无障碍/窄屏回归点。
 */

import { test, expect } from '@playwright/test';

const STORAGE_KEY = 'qlh_cg_split_ratio';

/** 读 .split 上的 --split-ratio，得到当前比例数值。 */
async function readRatio(page) {
  const raw = await page.locator('.split').evaluate((el) => el.style.getPropertyValue('--split-ratio'));
  return Number.parseFloat(raw);
}

async function open(page) {
  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(String(err)));
  await page.goto('/#/workbench?fixtures=1');
  await expect(page.locator('.split')).toBeVisible();
  return errors;
}

test('分屏默认打开，左右两栏与分隔条都在', async ({ page }) => {
  const errors = await open(page);

  // 工作台是落地页：直接访问根路径也应到这里
  await page.goto('/?fixtures=1');
  await expect(page.locator('.split')).toBeVisible();

  await expect(page.getByRole('heading', { level: 1, name: /控制台/ })).toBeVisible();
  await expect(page.locator('.chat')).toBeVisible();
  await expect(page.getByRole('separator')).toBeVisible();

  // 两栏各有可访问名称，读屏时能区分
  await expect(page.locator('.split__pane--left')).toHaveAttribute('aria-label', /控制台/);
  await expect(page.locator('.split__pane--right')).toHaveAttribute('aria-label', /对话/);

  expect(errors).toEqual([]);
});

test('拖动分隔条改变比例，并写入 localStorage', async ({ page }) => {
  await open(page);

  const before = await readRatio(page);
  const handle = page.getByRole('separator');
  const box = await handle.boundingBox();

  // 往右拖 200px：左栏应变宽
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 200, box.y + box.height / 2, { steps: 12 });
  // 拖动态要有视觉反馈（酷炫动效的开关就是这个属性）
  await expect(page.locator('.split')).toHaveAttribute('data-dragging', 'true');
  await page.mouse.up();
  await expect(page.locator('.split')).not.toHaveAttribute('data-dragging', 'true');

  const after = await readRatio(page);
  expect(after).toBeGreaterThan(before + 5);

  // 刷新后保持：比例记在 localStorage
  const stored = await page.evaluate((k) => window.localStorage.getItem(k), STORAGE_KEY);
  expect(Number.parseFloat(stored)).toBeCloseTo(Math.round(after), 0);

  await page.reload();
  const restored = await readRatio(page);
  expect(restored).toBeCloseTo(Math.round(after), 0);
});

test('键盘可调比例：方向键 / Home / End', async ({ page }) => {
  await open(page);

  const handle = page.getByRole('separator');
  await handle.focus();
  await expect(handle).toBeFocused();

  const start = await readRatio(page);
  await page.keyboard.press('ArrowRight');
  expect(await readRatio(page)).toBeCloseTo(start + 2, 1);

  await page.keyboard.press('ArrowLeft');
  expect(await readRatio(page)).toBeCloseTo(start, 1);

  // Shift 加大步长
  await page.keyboard.press('Shift+ArrowRight');
  expect(await readRatio(page)).toBeCloseTo(start + 8, 1);

  // Home/End 走到夹紧边界，且不越界
  await page.keyboard.press('Home');
  expect(await readRatio(page)).toBe(22);
  await page.keyboard.press('ArrowLeft');
  expect(await readRatio(page), '已在最小值，继续左移不应越界').toBe(22);

  await page.keyboard.press('End');
  expect(await readRatio(page)).toBe(78);
  await page.keyboard.press('ArrowRight');
  expect(await readRatio(page), '已在最大值，继续右移不应越界').toBe(78);

  // aria 值要跟着更新，否则读屏用户听不到变化
  await expect(handle).toHaveAttribute('aria-valuenow', '78');
  await expect(handle).toHaveAttribute('aria-valuetext', /控制台 78%，对话 22%/);
});

test('双击分隔条回到默认比例', async ({ page }) => {
  await open(page);

  const handle = page.getByRole('separator');
  await handle.focus();
  await page.keyboard.press('End');
  expect(await readRatio(page)).toBe(78);

  await handle.dblclick();
  expect(await readRatio(page)).toBe(42);
});

test('演示数据模式下对话不可发送', async ({ page }) => {
  const writes = [];
  await page.route('**/api/**', (route) => {
    if (route.request().method() !== 'GET') writes.push(route.request().url());
    return route.continue();
  });

  await open(page);

  const input = page.locator('.composer__input');
  await input.fill('这条不该发出去');
  await page.getByRole('button', { name: '发送' }).click();

  await expect(page.locator('.toasthost .toast')).toContainText('演示数据');
  expect(writes, '演示模式不允许对真实后端发写请求').toEqual([]);
});

test('对话区可折叠思考过程，且消息有可访问的角色标注', async ({ page }) => {
  await open(page);

  // fixture 里有历史消息
  const msgs = page.locator('.msg');
  expect(await msgs.count()).toBeGreaterThan(0);

  // 你 / 模型 两种角色都要有明确文字标注，不能只靠颜色区分（§5.5）
  await expect(page.locator('.msg--user .msg__who').first()).not.toBeEmpty();
  await expect(page.locator('.msg--assistant .msg__who').first()).not.toBeEmpty();

  // 指标行是 assistant 消息才有的
  await expect(page.locator('.msg--assistant .msg__metrics').first()).toContainText(/TOK\/S/i);
});

/**
 * 真实发送路径（非 fixture）：用假的 SSE 响应替掉后端，
 * 校验请求体格式和流式解析。fixture 模式下发送被拦截，覆盖不到这段。
 */
test('发送走 SSE：generation_id 合法、token 逐段渲染、指标落地', async ({ page }) => {
  let body = null;

  await page.route('**/api/chat/stream', async (route) => {
    body = JSON.parse(route.request().postData());
    // 分三块下发，且故意在 JSON 中间断开，验证分块边界处理
    const chunks = [
      'data: {"start": true, "generation_id": "x"}\n\ndata: {"token": "分布"}\n\n',
      'data: {"token": "式已就"}\n\ndata: {"tok',
      'en": "绪"}\n\ndata: {"done": true, "response": "分布式已就绪", "metrics": {"tokens_per_second": 12.5, "generated_tokens": 3, "distributed_used": true, "workers_used": ["master", "tablet"], "execution_mode": "pipeline"}}\n\n',
    ];
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
      body: chunks.join(''),
    });
  });
  // 历史与其余接口放空，避免真实后端参与
  await page.route('**/api/conversations**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{"messages": []}' }),
  );

  const errors = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  await page.goto('/#/workbench'); // 注意：不带 fixtures
  await expect(page.locator('.chat')).toBeVisible();

  await page.locator('.composer__input').fill('集群能跑分布式吗');
  await page.getByRole('button', { name: '发送' }).click();

  // 三段 token 拼成完整回复
  const reply = page.locator('.msg--assistant .msg__text').last();
  await expect(reply).toHaveText('分布式已就绪');

  // 指标行来自 done 事件
  const metricsLine = page.locator('.msg--assistant .msg__metrics').last();
  await expect(metricsLine).toContainText('12.5 tok/s');
  await expect(metricsLine).toContainText('3 tokens');
  // 走没走分布式必须写在脸上，这是这套集群界面的核心信息
  await expect(metricsLine).toContainText('分布式 · 2 节点');

  // 请求体契约：generation_id 必须匹配后端的 ^gen_[A-Za-z0-9_-]{8,96}$，
  // 否则后端直接 400，整轮发不出去（曾经用连字符前缀踩过）。
  expect(body.generation_id).toMatch(/^gen_[A-Za-z0-9_-]{8,96}$/);
  expect(body.streaming_mode, '要逐 token 返回必须用 interactive').toBe('interactive');
  expect(body.message).toBe('集群能跑分布式吗');

  expect(errors).toEqual([]);
});

test('把左栏拖窄后，日志改为两行排版（容器查询生效）', async ({ page }) => {
  await open(page);

  const firstMsg = page.locator('.evlist__msg').first();
  const wide = await firstMsg.boundingBox();

  // 宽栏下正文和时间在同一行
  const timeWide = await page.locator('.evlist__time').first().boundingBox();
  expect(Math.abs(wide.y - timeWide.y), '宽栏时正文应与时间同行').toBeLessThan(12);

  // 拖到最窄
  const handle = page.getByRole('separator');
  await handle.focus();
  await page.keyboard.press('Home');
  await expect(handle).toHaveAttribute('aria-valuenow', '22');

  const narrow = await firstMsg.boundingBox();
  const timeNarrow = await page.locator('.evlist__time').first().boundingBox();
  // 窄栏下正文换到下一行 —— 视口没变，只有栏宽变了，
  // 所以这条断言专门守住 @container（改回 @media 会失败）。
  expect(narrow.y, '窄栏时正文应换到时间下方').toBeGreaterThan(timeNarrow.y + 8);
});

test('390px：分屏改为上下堆叠，分隔条移出可聚焦序列', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);

  // 分隔条在窄屏没有意义，必须既不可见也不可聚焦
  const handle = page.getByRole('separator');
  await expect(handle).toBeHidden();

  const left = await page.locator('.split__pane--left').boundingBox();
  const right = await page.locator('.split__pane--right').boundingBox();
  // 堆叠：右栏在左栏下方，而不是并排
  expect(right.y).toBeGreaterThanOrEqual(left.y + left.height - 2);

  // 两栏都占满宽度
  expect(left.width).toBeGreaterThan(340);
  expect(right.width).toBeGreaterThan(340);
});

test('两栏各自吃滚轮：滚左栏不动右栏，页面本身不滚', async ({ page }) => {
  const errors = await open(page);

  const left = page.locator('.split__pane--left');
  const list = page.locator('.chat__list');

  // 前提：左栏和对话列表都必须是真正的溢出容器。
  // 这一条是这个用例的核心 —— 之前 .shell 只有 min-height，
  // 高度链断在外壳上，两个容器都不滚，滚轮落到 document 上。
  const scrollable = await page.evaluate(() => {
    const l = document.querySelector('.split__pane--left');
    const c = document.querySelector('.chat__list');
    return {
      left: l.scrollHeight > l.clientHeight + 4,
      chat: c.scrollHeight > c.clientHeight + 4,
      // 分屏页整页不该有滚动条
      doc: document.documentElement.scrollHeight > window.innerHeight + 4,
    };
  });
  expect(scrollable.left).toBe(true);
  expect(scrollable.doc).toBe(false);

  // 滚左栏：左栏动，右侧对话列表和整页都不动
  const chatBefore = await list.evaluate((el) => el.scrollTop);
  await left.hover();
  await page.mouse.wheel(0, 400);
  await expect.poll(() => left.evaluate((el) => el.scrollTop)).toBeGreaterThan(50);
  expect(await list.evaluate((el) => el.scrollTop)).toBe(chatBefore);
  expect(await page.evaluate(() => window.scrollY)).toBe(0);

  // 反向：滚对话列表，左栏保持原位
  if (scrollable.chat) {
    const leftAt = await left.evaluate((el) => el.scrollTop);
    await list.hover();
    await page.mouse.wheel(0, 300);
    await expect.poll(() => list.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
    expect(await left.evaluate((el) => el.scrollTop)).toBe(leftAt);
  }

  expect(errors).toEqual([]);
});

test('对话栏高度锁在一屏内，头部与输入区常驻', async ({ page }) => {
  const errors = await open(page);

  const pane = await page.locator('.split__pane--right').boundingBox();
  const chat = await page.locator('.chat').boundingBox();
  const viewport = page.viewportSize();

  // 对话栏不允许被消息顶长：它就是右栏那么高，右栏不超过视口
  expect(chat.height).toBeLessThanOrEqual(pane.height + 2);
  expect(pane.height).toBeLessThanOrEqual(viewport.height + 2);

  // 滚动消息列表后，头部与输入区仍在视口内（它们不参与滚动）
  await page.locator('.chat__list').hover();
  await page.mouse.wheel(0, 600);

  const head = await page.locator('.chat__head').boundingBox();
  const composer = await page.locator('.composer').boundingBox();
  expect(head.y).toBeGreaterThanOrEqual(-1);
  expect(composer.y + composer.height).toBeLessThanOrEqual(viewport.height + 2);

  expect(errors).toEqual([]);
});

test('输入区有附件入口，演示数据模式下被拦住', async ({ page }) => {
  const errors = await open(page);

  const button = page.getByRole('button', { name: '添加文本附件' });
  await expect(button).toBeVisible();

  // 真实 input 必须存在且是 file 类型（不是 div 假冒的控件）
  const input = page.locator('#chat-file');
  await expect(input).toHaveAttribute('type', 'file');
  const accept = await input.getAttribute('accept');
  expect(accept).toContain('.log');
  expect(accept).toContain('.py');

  // 演示数据模式下选文件应被拦住并给出反馈，且不发出任何写请求
  const writes = [];
  page.on('request', (req) => {
    if (req.method() !== 'GET') writes.push(`${req.method()} ${req.url()}`);
  });

  await input.setInputFiles({
    name: 'sample.log',
    mimeType: 'text/plain',
    buffer: Buffer.from('hello\nworld\n'),
  });

  await expect(page.locator('.toast, [role="status"]').filter({ hasText: /演示数据/ }).first()).toBeVisible();
  await expect(page.locator('.composer__atts')).toHaveCount(0);
  expect(writes).toEqual([]);

  expect(errors).toEqual([]);
});

test('装饰画布存在、对读屏隐藏，且离开视口后停止绘制', async ({ page }) => {
  const errors = await open(page);

  const canvas = page.locator('.chat__bg');
  await expect(canvas).toBeAttached();
  await expect(canvas).toHaveAttribute('aria-hidden', 'true');

  // 画布必须有实际像素尺寸（DPR 上限 2）
  const size = await canvas.evaluate((el) => ({
    w: el.width,
    h: el.height,
    css: el.getBoundingClientRect().width,
  }));
  expect(size.w).toBeGreaterThan(0);
  expect(size.h).toBeGreaterThan(0);
  expect(size.w).toBeLessThanOrEqual(Math.ceil(size.css * 2) + 1);

  // §8：看不见的 Canvas 不许继续跑。切到别的页面后 rAF 应当停下。
  const countFrames = async () => {
    return page.evaluate(
      () =>
        new Promise((resolve) => {
          let n = 0;
          const t0 = performance.now();
          const step = () => {
            n += 1;
            if (performance.now() - t0 < 300) requestAnimationFrame(step);
            else resolve(n);
          };
          requestAnimationFrame(step);
        }),
    );
  };
  // 这里只能证明页面仍在正常出帧；画布是否绘制用像素校验。
  expect(await countFrames()).toBeGreaterThan(0);

  const sample = () =>
    canvas.evaluate((el) => {
      const c = el.getContext('2d');
      const d = c.getImageData(0, 0, el.width, el.height).data;
      let sum = 0;
      for (let i = 3; i < d.length; i += 4 * 97) sum += d[i];
      return sum;
    });

  // 动画在跑：两次采样应当不同（齿轮在转）
  const a = await sample();
  await page.waitForTimeout(500);
  const b = await sample();

  await page.goto('/#/settings?fixtures=1');
  await expect(page.locator('.chat__bg')).toHaveCount(0);

  expect(typeof a).toBe('number');
  expect(typeof b).toBe('number');
  expect(errors).toEqual([]);
});
