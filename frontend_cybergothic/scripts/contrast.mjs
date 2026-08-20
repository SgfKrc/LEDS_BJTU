/**
 * WCAG 对比度自检 — 校验 tokens.css 里的前景/背景组合是否达到 AA（§8）。
 *
 * 用法：node scripts/contrast.mjs
 * 直接从 src/styles/tokens.css 解析 hex，避免和样式脱节。
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, '..', 'src', 'styles', 'tokens.css'), 'utf8');

/** 从 tokens.css 里取出所有 `--name: #hex;` 形式的变量。 */
function readTokens(source) {
  const out = {};
  const re = /--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,8})\s*;/g;
  let m;
  while ((m = re.exec(source)) !== null) out[m[1]] = m[2];
  return out;
}

function toRgb(hex) {
  let h = hex.slice(1);
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
}

function luminance(hex) {
  const [r, g, b] = toRgb(hex).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function ratio(fg, bg) {
  const a = luminance(fg);
  const b = luminance(bg);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

const t = readTokens(css);

/** [前景, 背景, 用途, 需要的最低比值]；大字号/图形按 AA 的 3:1 计。 */
const PAIRS = [
  ['text', 'bg', '正文 / 页面底色', 4.5],
  ['text', 'surface', '正文 / 卡片', 4.5],
  ['text', 'surface-raised', '正文 / 抬升面', 4.5],
  ['muted', 'bg', '次要文字 / 页面底色', 4.5],
  ['muted', 'surface', '次要文字 / 卡片', 4.5],
  ['muted', 'surface-raised', '次要文字 / 抬升面', 4.5],
  ['accent', 'bg', '主色文字 / 页面底色', 4.5],
  ['accent', 'surface', '主色文字 / 卡片', 4.5],
  ['accent', 'surface-raised', '主色文字 / 抬升面', 4.5],
  ['info', 'bg', 'info 文字 / 页面底色', 4.5],
  ['info', 'surface', 'info 文字 / 卡片', 4.5],
  ['danger', 'bg', 'danger 文字 / 页面底色', 4.5],
  ['danger', 'surface', 'danger 文字 / 卡片', 4.5],
  ['bg', 'accent', '主按钮文字 / 主色底', 4.5],
  ['accent', 'surface', '焦点描边 / 卡片', 3],

  // 对话栏（哥特侧）自成一套色板，必须单独校验：
  // 它的底色更深、主色是苍紫，和上面的酸绿没有任何继承关系。
  ['gothic-ink', 'gothic-bg', '哥特正文 / 底色', 4.5],
  ['gothic-ink', 'gothic-surface', '哥特正文 / 输入框', 4.5],
  ['gothic-ink', 'gothic-surface-raised', '哥特正文 / 抬升面', 4.5],
  ['gothic-muted', 'gothic-bg', '哥特次要文字 / 底色', 4.5],
  ['gothic-muted', 'gothic-surface', '哥特次要文字 / 输入框', 4.5],
  ['gothic-accent', 'gothic-bg', '苍紫文字 / 底色', 4.5],
  ['gothic-accent', 'gothic-surface', '苍紫文字 / 输入框', 4.5],
  ['gothic-gold', 'gothic-bg', '描边金文字 / 底色', 4.5],
  ['gothic-gold', 'gothic-surface', '描边金文字 / 输入框', 4.5],
  ['gothic-accent', 'gothic-surface', '哥特焦点描边 / 输入框', 3],
  ['gothic-accent-ink', 'gothic-accent', '哥特主按钮文字 / 苍紫底', 4.5],
  ['danger', 'gothic-bg', '哥特侧错误文字 / 底色', 4.5],
];

let failed = 0;
const rows = PAIRS.map(([fgName, bgName, label, need]) => {
  const fg = t[fgName];
  const bg = t[bgName];
  if (!fg || !bg) {
    failed += 1;
    return `MISS  ${label}  (${fgName} 或 ${bgName} 不是 hex 变量)`;
  }
  const r = ratio(fg, bg);
  const ok = r >= need;
  if (!ok) failed += 1;
  return `${ok ? 'PASS' : 'FAIL'}  ${r.toFixed(2)}:1  (需 ${need}:1)  ${label}  ${fg} on ${bg}`;
});

console.log(rows.join('\n'));
console.log(`\n${PAIRS.length - failed}/${PAIRS.length} 组合达标。`);
process.exit(failed > 0 ? 1 : 0);
