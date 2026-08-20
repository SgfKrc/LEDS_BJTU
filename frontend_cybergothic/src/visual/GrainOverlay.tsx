/**
 * 颗粒纹理层 — 低透明度噪点，只增加材质（§2.5、§4.1「降低颗粒透明度」）。
 *
 * 用内联 SVG feTurbulence 生成，不加载图片资源。
 * 纯 CSS 静态叠层（无动画），减少动效时由 CSS 直接隐藏。
 */

export function GrainOverlay() {
  return <div className="grain" aria-hidden="true" />;
}
