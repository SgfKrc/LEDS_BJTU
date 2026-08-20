# 静态资源

这个目录目前是空的，界面不依赖任何图片或字体文件。这是刻意的：

- **图形**：所有插画、图标、纹理都用内联 SVG、CSS 渐变或 Canvas 生成，避免首屏因图片加载产生布局跳动。
- **字体**：使用系统字体栈（见 `src/styles/tokens.css` 的 `--font-sans` / `--font-mono` / `--font-display`），不加载 CDN 字体。边缘部署常常没有外网，缺字体会直接掉回默认样式。
- **图标**：来自 `lucide-react`，随 JS 打包，不需要额外文件。

## 什么时候往这里放东西

放进 `public/assets/` 的文件会被原样复制到 `dist/assets/`，通过 `/assets/<文件名>` 访问。适合：

- 品牌 logo（如果要替换顶栏那个斜切方块）
- 需要外链分享的截图或 OG 图
- 自托管字体文件（`.woff2`）

引用方式（注意不写 `public` 前缀）：

```jsx
<img src="/assets/logo.svg" alt="" />
```

自托管字体时，在 `src/styles/tokens.css` 顶部加 `@font-face`，然后把字体名插到对应变量的最前面：

```css
@font-face {
  font-family: 'Space Grotesk';
  src: url('/assets/space-grotesk.woff2') format('woff2');
  font-display: swap;
}
```

`--font-sans` 已经把 `'Space Grotesk'` 写在首位，所以放好文件即生效，不需要改组件。

## 换主题色

不要在这里放配色文件。主色只有一处：`src/styles/tokens.css` 里的 `--accent`（以及配套的 `--accent-soft` / `--accent-line` / `--accent-ink`）。改这四个值就换掉整套界面的强调色。
