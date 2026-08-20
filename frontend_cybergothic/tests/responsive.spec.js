/**
 * 响应式自检 — §6 要求在 1440 / 1024 / 768 / 390px 下确认版式。
 *
 * 除了截图（存到 ../build/cybergothic-shots，便于人工过一眼），
 * 这里还机械校验两件容易回归的事：
 *   1. 不出现横向滚动；
 *   2. 没有元素溢出视口右边界（clip-path 和硬阴影很容易顶破）。
 */

import { test, expect } from '@playwright/test';

const WIDTHS = [1440, 1024, 768, 390];
const PAGES = ['workbench', 'overview', 'tasks', 'activity', 'image', 'models', 'settings', 'help'];
const SHOTS = '../build/cybergothic-shots';

/** 允许的溢出容差（px）：阴影和 1px 边框的取整误差。 */
const SLOP = 2;

for (const width of WIDTHS) {
  test(`${width}px 下各页面无横向溢出`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });

    for (const name of PAGES) {
      await page.goto(`/#/${name}?fixtures=1`);
      await expect(page.locator('.shell')).toBeVisible();
      // 等首屏数据落位，避免截到骨架屏。
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
      await page.waitForTimeout(500);

      // 滚到底再回顶：既触发 [data-reveal]（否则 fullPage 截图下半页全是空白），
      // 也顺带验证滚动确实让区块显现了。
      await page.evaluate(async () => {
        const step = window.innerHeight * 0.8;
        for (let y = 0; y < document.body.scrollHeight; y += step) {
          window.scrollTo(0, y);
          await new Promise((r) => setTimeout(r, 120));
        }
        window.scrollTo(0, 0);
      });
      await page.waitForTimeout(400);

      const hidden = await page.locator('[data-reveal]:not([data-revealed])').count();
      expect(hidden, `${name} @${width}px 有区块滚动后仍未显现`).toBe(0);

      const overflow = await page.evaluate((slop) => {
        const doc = document.documentElement;
        const scroll = doc.scrollWidth - doc.clientWidth;
        const offenders = [];
        for (const el of document.querySelectorAll('body *')) {
          const style = getComputedStyle(el);
          if (style.position === 'fixed' || style.display === 'none') continue;
          const r = el.getBoundingClientRect();
          if (r.width === 0 && r.height === 0) continue;
          if (r.right > doc.clientWidth + slop) {
            offenders.push(`${el.className || el.tagName} right=${Math.round(r.right)}`);
          }
        }
        return { scroll, offenders: offenders.slice(0, 5) };
      }, SLOP);

      expect(overflow.offenders, `${name} @${width}px 溢出`).toEqual([]);
      expect(overflow.scroll, `${name} @${width}px 横向滚动`).toBeLessThanOrEqual(SLOP);

      await page.screenshot({ path: `${SHOTS}/${width}-${name}.png`, fullPage: true });
    }
  });
}
