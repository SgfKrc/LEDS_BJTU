/**
 * 基础 E2E — 覆盖 §6「构建 + 类型检查 + 基础端到端」中的端到端部分。
 *
 * 全部在 fixture 模式下运行（?fixtures=1），不依赖后端是否启动，
 * 也不会对真实集群发出写请求。
 */

import { test, expect } from '@playwright/test';

const PAGES = [
  { hash: '#/overview', heading: '集群概览' },
  { hash: '#/tasks', heading: '任务' },
  { hash: '#/activity', heading: '活动' },
  { hash: '#/image', heading: '生图工坊' },
  { hash: '#/models', heading: 'Model workspace' },
  { hash: '#/settings', heading: '设置' },
  { hash: '#/help', heading: '帮助' },
];

/** 打开 fixture 模式下的某个页面，并收集控制台错误。 */
async function openPage(page, hash = '#/overview') {
  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(String(err)));
  await page.goto(`/${hash}?fixtures=1`);
  await expect(page.locator('.shell')).toBeVisible();
  return errors;
}

test('概览页渲染 hero、关键数字与后续区块', async ({ page }) => {
  const errors = await openPage(page, '#/overview');

  // Hero 三件套：状态句 + 主行动 + 关键数字（§4.4）
  await expect(page.getByRole('heading', { level: 1, name: /集群概览/ })).toBeVisible();
  await expect(page.locator('.hero__status')).not.toBeEmpty();
  await expect(page.locator('.hero__actions .cbtn').first()).toBeVisible();
  await expect(page.locator('.metricstrip .metric')).toHaveCount(4);

  // 首屏底部必须露出下一段的线索
  await expect(page.locator('.hero__next')).toBeVisible();

  // 装饰 Canvas 必须是 aria-hidden，不承载信息（§5.5）
  const canvas = page.locator('canvas.accent-canvas');
  await expect(canvas).toHaveAttribute('aria-hidden', 'true');

  // fixture 模式的提示条要出现，避免误读为真实状态
  await expect(page.locator('.fixturebar')).toContainText('演示数据模式');

  expect(errors).toEqual([]);
});

test('五个页面都能通过导航打开且无控制台错误', async ({ page }) => {
  const errors = await openPage(page, '#/overview');

  for (const item of PAGES) {
    await page.goto(`/${item.hash}?fixtures=1`);
    await expect(page.getByRole('heading', { level: 1 })).toContainText(item.heading);
    // 每页都应有内容，不允许空白屏
    await expect(page.locator('.shell__main')).not.toBeEmpty();
  }

  expect(errors).toEqual([]);
});

test('任务页展示队列层级并可切换工作流筛选', async ({ page }) => {
  await openPage(page, '#/tasks');

  await expect(page.locator('.qlevel')).toHaveCount(3);
  await expect(page.locator('.ttable').first()).toBeVisible();

  // 筛选到「失败」后，fixture 里只剩失败的工作流
  await page.getByRole('button', { name: '失败' }).click();
  await expect(page.locator('.ttable').last().locator('tbody tr')).toHaveCount(1);

  // 打开详情抽屉并用 Escape 关闭（§5.5）
  await page.locator('.ttable__open').last().click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
});

test('活动页日志可按级别筛选，条目可展开详情', async ({ page }) => {
  await openPage(page, '#/activity');

  const items = page.locator('.timeline__item');
  await expect(items.first()).toBeVisible();

  // 筛到 ERROR 后，可见条目的级别标签只能是 ERROR/CRITICAL。
  // allInnerTexts 是一次性读取、不会自动重试，所以套在 toPass 里等列表稳定；
  // 超时刻意小于日志轮询间隔（10s），否则「筛选要等下一次轮询才生效」的回归会被放过。
  await page.locator('.chip', { hasText: 'ERROR' }).click();
  await expect(async () => {
    const levels = await page.locator('.timeline__item .badge').allInnerTexts();
    expect(levels.length).toBeGreaterThan(0);
    for (const level of levels) {
      expect(level.trim()).toMatch(/^(ERROR|CRITICAL)$/);
    }
  }).toPass({ timeout: 4000 });

  await page.locator('.chip', { hasText: '全部' }).click();
  await page.locator('.timeline__body--action').first().click();
  await expect(page.getByRole('dialog')).toBeVisible();
});

test('设置页可切换动效偏好并写入 html 标记', async ({ page }) => {
  await openPage(page, '#/settings');

  await page.getByRole('radio', { name: /减少动效/ }).check();
  await expect(page.locator('html')).toHaveAttribute('data-reduced-motion', 'true');
  await expect(page.locator('.toasthost .toast')).toContainText('减少动效');

  await page.getByRole('radio', { name: /完整动效/ }).check();
  await expect(page.locator('html')).toHaveAttribute('data-reduced-motion', 'false');
});

test('演示数据模式下写操作被拦截，不会打到后端', async ({ page }) => {
  const writes = [];
  await page.route('**/api/**', (route) => {
    if (route.request().method() !== 'GET') writes.push(route.request().url());
    return route.fulfill({ status: 200, body: '{}' });
  });

  await openPage(page, '#/tasks');
  await page.getByRole('button', { name: /暂停队列|恢复队列/ }).click();
  await expect(page.locator('.toasthost .toast')).toContainText('演示数据');
  expect(writes).toEqual([]);
});

test('键盘可完成主流程：跳过链接 + Tab 导航', async ({ page }) => {
  await openPage(page, '#/overview');

  // 第一个 Tab 落在「跳到主内容」上
  await page.keyboard.press('Tab');
  await expect(page.locator('.skip-link')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#main/);

  // 顶栏导航是真实链接，可用键盘直接激活
  const tasksLink = page.locator('.topnav__link', { hasText: '任务' });
  await tasksLink.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('任务');
});

test('390px 窄屏：导航收进抽屉，可用 Escape 关闭', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openPage(page, '#/overview');

  await expect(page.locator('.topnav')).toBeHidden();
  await page.getByRole('button', { name: /菜单|打开菜单/ }).click();

  const nav = page.getByRole('dialog');
  await expect(nav).toBeVisible();
  // 抽屉打开时锁定背景滚动
  await expect(page.locator('body')).toHaveAttribute('data-scroll-locked', 'true');

  await page.keyboard.press('Escape');
  await expect(nav).toHaveCount(0);
  await expect(page.locator('body')).not.toHaveAttribute('data-scroll-locked', 'true');
});

test('768px：表格转为纵向条目，列名通过 data-label 呈现', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await openPage(page, '#/tasks');

  // 跳过复选框列（它没有列名，窄屏下也不需要标签）
  const firstCell = page.locator('.ttable tbody td[data-label]').first();
  await expect(firstCell).toHaveAttribute('data-label', /.+/);
  // 单元格在窄屏是块级布局
  const display = await firstCell.evaluate((el) => getComputedStyle(el).display);
  expect(display).toBe('flex');
});

test('reduced-motion 下装饰 Canvas 不持续绘制', async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  const page = await context.newPage();
  await page.goto('/#/overview?fixtures=1');
  await expect(page.locator('canvas.accent-canvas')).toBeVisible();

  // 统计 rAF 调用次数：静态模式下应当停在个位数
  const frames = await page.evaluate(async () => {
    let count = 0;
    const original = window.requestAnimationFrame;
    window.requestAnimationFrame = (cb) => {
      count += 1;
      return original(cb);
    };
    await new Promise((resolve) => setTimeout(resolve, 1200));
    window.requestAnimationFrame = original;
    return count;
  });
  expect(frames).toBeLessThan(10);

  await context.close();
});
