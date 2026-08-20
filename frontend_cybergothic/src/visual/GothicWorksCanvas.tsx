/**
 * 对话栏背景 — 哥特建筑 + 互相啮合的巨型齿轮钟。
 *
 * 构图：底部一列尖拱连券和两座带尖顶的塔（静态），
 * 右上一面玫瑰窗当表盘，左下三只齿轮咬在一起带动指针（动态）。
 *
 * 工程约束（§5.4 / §5.5 / §6）：
 * - 纯装饰，aria-hidden，不承载任何信息。
 * - 静态部分画进离屏 canvas 缓存，每帧只重绘齿轮和指针。
 * - DPR 上限 2；ResizeObserver 处理尺寸变化。
 * - 离开视口 / 页面切后台立即 cancelAnimationFrame（§8：看不见的 Canvas 不许继续跑）。
 * - prefers-reduced-motion 时只画一帧静态图。
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** 齿的节距（相邻齿中心的弧长基准）。所有齿轮共用，齿才咬得住。 */
const MODULE = 7.2;
/** 齿高（径向），相对模数取值，齿形才不会又尖又长。 */
const TOOTH_H = MODULE * 0.62;

interface Gear {
  x: number;
  y: number;
  /** 齿数。半径由 teeth × MODULE / 2π 决定，不单独给。 */
  teeth: number;
  /** 转向：+1 顺时针，-1 逆时针。相邻齿轮必须相反。 */
  dir: 1 | -1;
  /** 起始相位，由啮合关系算出，见 meshPhase()。 */
  phase: number;
  radius: number;
  spokes: number;
}

function radiusOf(teeth: number): number {
  // 节圆周长 = 齿数 × 节距
  return (teeth * MODULE) / (Math.PI * 2);
}

/**
 * 求从动轮的起始相位，使它的齿槽正对主动轮的齿。
 *
 * 记连心线方向角 α（从从动轮看向主动轮为 α+π）。主动轮的齿位于
 * A_k = θ1 + 2πk/N1，从动轮的齿槽位于 G_j = θ2 + π/N2 + 2πj/N2。
 * 啮合要求：A_k = α 时存在 j 使 G_j = α + π。
 *
 * 代入无滑滚动 θ2 = φ2 − (N1/N2)(θ1 − φ1)，k 的部分正好被 j 抵消
 * （两者都是 2π/N2 的整数倍），于是只剩一个与 k 无关的条件：
 *   φ2 = (1 + N1/N2)·α + π − π/N2 − (N1/N2)·φ1
 * 这个关系一旦成立，就在整个转动过程中自动保持。
 */
function meshPhase(driver: Gear, follower: Gear): number {
  const alpha = Math.atan2(driver.y - follower.y, driver.x - follower.x);
  const ratio = driver.teeth / follower.teeth;
  return (
    (1 + ratio) * alpha + Math.PI - Math.PI / follower.teeth - ratio * driver.phase
  );
}

export function GothicWorksCanvas({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const context = el.getContext('2d');
    if (!context) return;

    // 函数声明会被提升，TS 不会在其内部沿用外层的 null 收窄，先固定成非空 const。
    const canvas: HTMLCanvasElement = el;
    const ctx: CanvasRenderingContext2D = context;

    let width = 0;
    let height = 0;
    let frame = 0;
    let visible = true;
    let running = false;
    let spin = 0;
    let gears: Gear[] = [];
    let clock = { x: 0, y: 0, r: 0 };

    const styles = getComputedStyle(canvas);
    const gold = styles.getPropertyValue('--gothic-gold').trim() || '#d9c27a';
    const pale = styles.getPropertyValue('--gothic-accent').trim() || '#d8b4ff';
    const stone = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    /** 静态层缓存：建筑不动，没必要每帧重画。 */
    let backdrop: HTMLCanvasElement | null = null;

    // ---- 静态：建筑 ----

    /** 一道尖拱：两段圆弧交于顶点，腿落在两侧墩上。 */
    function pointedArch(
      c: CanvasRenderingContext2D,
      x: number,
      y: number,
      span: number,
      rise: number,
    ) {
      // 半径取跨度，圆心分别在两个券脚 —— 交点即尖顶，等边尖拱的标准作法。
      const r = span;
      const apexY = y - Math.sqrt(Math.max(0, r * r - span * span * 0.25)) * (rise / span);
      c.beginPath();
      c.moveTo(x - span / 2, y);
      c.quadraticCurveTo(x - span / 2, apexY + (y - apexY) * 0.18, x, apexY);
      c.quadraticCurveTo(x + span / 2, apexY + (y - apexY) * 0.18, x + span / 2, y);
      c.stroke();
    }

    /** 塔：垂直墩身 + 尖顶 + 两层窗。 */
    function tower(c: CanvasRenderingContext2D, x: number, base: number, w: number, h: number) {
      const top = base - h;
      c.beginPath();
      c.moveTo(x - w / 2, base);
      c.lineTo(x - w / 2, top);
      c.lineTo(x, top - w * 1.15); // 尖顶
      c.lineTo(x + w / 2, top);
      c.lineTo(x + w / 2, base);
      c.stroke();

      // 窗：两层细长柳叶窗
      for (let i = 0; i < 2; i += 1) {
        const wy = top + h * (0.3 + i * 0.28);
        pointedArch(c, x, wy, w * 0.42, w * 0.5);
        c.beginPath();
        c.moveTo(x, wy);
        c.lineTo(x, wy - w * 0.3);
        c.stroke();
      }
    }

    function buildBackdrop() {
      const layer = document.createElement('canvas');
      const dpr = Math.min(DPR_MAX, window.devicePixelRatio || 1);
      layer.width = Math.max(1, Math.round(width * dpr));
      layer.height = Math.max(1, Math.round(height * dpr));
      const c = layer.getContext('2d');
      if (!c) return null;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);

      c.lineWidth = 1.15;

      // 远景：连券。券脚落在同一条基线上，横向铺满。
      const base = height * 0.9;
      const span = Math.max(64, Math.min(120, width / 6));
      c.strokeStyle = stone;
      c.globalAlpha = 0.5;
      for (let x = span * 0.5; x < width + span; x += span) {
        pointedArch(c, x, base, span, span * 1.05);
        // 墩
        c.beginPath();
        c.moveTo(x - span / 2, base);
        c.lineTo(x - span / 2, height);
        c.stroke();
      }
      // 基线
      c.beginPath();
      c.moveTo(0, base);
      c.lineTo(width, base);
      c.stroke();

      // 中景：两座塔贴着左右边缘，只露一半，暗示这里是更大一座建筑的一角。
      c.strokeStyle = gold;
      c.globalAlpha = 0.34;
      tower(c, width * 0.06, base, Math.min(84, width * 0.16), height * 0.52);
      tower(c, width * 0.95, base, Math.min(64, width * 0.13), height * 0.4);

      // 拱肋：从塔顶斜拉到画面中部，把上半部撑满，不至于太空
      c.globalAlpha = 0.16;
      c.beginPath();
      c.moveTo(0, height * 0.36);
      c.quadraticCurveTo(width * 0.5, height * 0.06, width, height * 0.3);
      c.stroke();
      c.beginPath();
      c.moveTo(0, height * 0.22);
      c.quadraticCurveTo(width * 0.46, -height * 0.04, width, height * 0.16);
      c.stroke();

      c.globalAlpha = 1;
      return layer;
    }

    // ---- 动态：齿轮与钟 ----

    function layout() {
      // 齿轮组：大轮压在左下角外侧，只露一部分（「巨型」的观感靠露不全来给）。
      const big = Math.max(120, Math.min(210, Math.min(width, height) * 0.42));
      const teethBig = Math.max(24, Math.round((big * Math.PI * 2) / MODULE));

      const g1: Gear = {
        x: width * 0.1,
        y: height * 0.74,
        teeth: teethBig,
        dir: 1,
        phase: 0,
        radius: radiusOf(teethBig),
        spokes: 8,
      };
      const teeth2 = Math.max(13, Math.round(teethBig * 0.46));
      const g2: Gear = {
        x: 0,
        y: 0,
        teeth: teeth2,
        dir: -1,
        phase: 0,
        radius: radiusOf(teeth2),
        spokes: 6,
      };
      // 中心距 = 两节圆半径之和，方向朝右上，指向表盘。
      const a1 = -Math.PI * 0.36;
      const d1 = g1.radius + g2.radius;
      g2.x = g1.x + Math.cos(a1) * d1;
      g2.y = g1.y + Math.sin(a1) * d1;
      g2.phase = meshPhase(g1, g2);

      const teeth3 = Math.max(11, Math.round(teethBig * 0.3));
      const g3: Gear = {
        x: 0,
        y: 0,
        teeth: teeth3,
        dir: 1,
        phase: 0,
        radius: radiusOf(teeth3),
        spokes: 5,
      };
      const a2 = -Math.PI * 0.06;
      const d2 = g2.radius + g3.radius;
      g3.x = g2.x + Math.cos(a2) * d2;
      g3.y = g2.y + Math.sin(a2) * d2;
      g3.phase = meshPhase(g2, g3);

      gears = [g1, g2, g3];
      clock = { x: width * 0.76, y: height * 0.24, r: Math.min(width, height) * 0.17 };
    }

    /** 齿圈：梯形齿，齿顶宽是齿槽的 0.62，看起来像铸件而不是锯条。 */
    function drawGear(g: Gear, angle: number) {
      const rOuter = g.radius + TOOTH_H * 0.5;
      const rRoot = g.radius - TOOTH_H * 0.5;
      const step = (Math.PI * 2) / g.teeth;
      const halfTop = step * 0.5 * 0.62 * 0.5;
      const halfRoot = step * 0.5 * 0.5;

      ctx.beginPath();
      for (let i = 0; i < g.teeth; i += 1) {
        const c = angle + i * step;
        ctx.lineTo(Math.cos(c - halfRoot) * rRoot + g.x, Math.sin(c - halfRoot) * rRoot + g.y);
        ctx.lineTo(Math.cos(c - halfTop) * rOuter + g.x, Math.sin(c - halfTop) * rOuter + g.y);
        ctx.lineTo(Math.cos(c + halfTop) * rOuter + g.x, Math.sin(c + halfTop) * rOuter + g.y);
        ctx.lineTo(Math.cos(c + halfRoot) * rRoot + g.x, Math.sin(c + halfRoot) * rRoot + g.y);
        // 齿槽底：走一小段根圆
        const next = c + step - halfRoot;
        ctx.arc(g.x, g.y, rRoot, c + halfRoot, next);
      }
      ctx.closePath();
      ctx.stroke();

      // 轮毂与辐条
      const hub = g.radius * 0.22;
      ctx.beginPath();
      ctx.arc(g.x, g.y, hub, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(g.x, g.y, g.radius * 0.72, 0, Math.PI * 2);
      ctx.stroke();
      for (let i = 0; i < g.spokes; i += 1) {
        const c = angle + (i * Math.PI * 2) / g.spokes;
        ctx.beginPath();
        ctx.moveTo(g.x + Math.cos(c) * hub, g.y + Math.sin(c) * hub);
        ctx.lineTo(g.x + Math.cos(c) * g.radius * 0.72, g.y + Math.sin(c) * g.radius * 0.72);
        ctx.stroke();
      }
    }

    /** 玫瑰窗兼表盘：外圈 + 放射花饰 + 一圈小尖拱 + 两根指针。 */
    function drawClock(t: number) {
      const { x, y, r } = clock;

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, r * 0.82, 0, Math.PI * 2);
      ctx.stroke();

      // 放射花饰：12 瓣，每瓣一个朝心的小圆，构成玫瑰窗的花格
      const petals = 12;
      for (let i = 0; i < petals; i += 1) {
        const a = (i * Math.PI * 2) / petals;
        const px = x + Math.cos(a) * r * 0.56;
        const py = y + Math.sin(a) * r * 0.56;
        ctx.beginPath();
        ctx.arc(px, py, r * 0.24, 0, Math.PI * 2);
        ctx.stroke();
        // 时刻线
        ctx.beginPath();
        ctx.moveTo(x + Math.cos(a) * r * 0.82, y + Math.sin(a) * r * 0.82);
        ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, r * 0.2, 0, Math.PI * 2);
      ctx.stroke();

      // 指针：分针跟最小的齿轮同速，时针 12:1 减速
      const minute = t * Math.PI * 2;
      const hour = minute / 12;
      ctx.save();
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.sin(minute) * r * 0.74, y - Math.cos(minute) * r * 0.74);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.sin(hour) * r * 0.46, y - Math.cos(hour) * r * 0.46);
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      if (backdrop) {
        // 缓存层是按 DPR 放大过的，这里按 CSS 尺寸贴回去
        ctx.drawImage(backdrop, 0, 0, width, height);
      }

      const driver = gears[0];
      if (!driver) return;

      ctx.save();
      ctx.lineWidth = 1.2;
      ctx.lineJoin = 'round';
      gears.forEach((g, i) => {
        // 无滑滚动：角速度与齿数成反比，所以乘 N1/Ni
        const angle = g.phase + g.dir * spin * (driver.teeth / g.teeth);
        ctx.strokeStyle = i === 0 ? stone : gold;
        ctx.globalAlpha = i === 0 ? 0.42 : 0.3;
        drawGear(g, angle);
      });
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = pale;
      ctx.globalAlpha = 0.26;
      ctx.lineWidth = 1.1;
      // 表盘走最末一只齿轮的转数，机械上说得通
      const last = gears[gears.length - 1];
      const turns = last ? (spin * (driver.teeth / last.teeth)) / (Math.PI * 2) : 0;
      drawClock(turns);
      ctx.restore();
    }

    function tick() {
      if (!running) return;
      // 慢：大轮每分钟约一圈的量级，不抢注意力
      spin += 0.0016;
      draw();
      frame = requestAnimationFrame(tick);
    }

    function start() {
      if (running || reduced) return;
      running = true;
      frame = requestAnimationFrame(tick);
    }

    function stop() {
      running = false;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const dpr = Math.min(DPR_MAX, window.devicePixelRatio || 1);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      layout();
      backdrop = buildBackdrop();
      draw(); // 尺寸变化后补一帧，静态模式下也不会留白
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const io =
      typeof IntersectionObserver !== 'undefined'
        ? new IntersectionObserver(
            (entries) => {
              visible = entries.some((e) => e.isIntersecting);
              if (visible && document.visibilityState === 'visible') start();
              else stop();
            },
            { threshold: 0.02 },
          )
        : null;
    io?.observe(canvas);

    const onVisibility = () => {
      if (document.visibilityState === 'visible' && visible) start();
      else stop();
    };
    document.addEventListener('visibilitychange', onVisibility);

    if (reduced) draw();
    else if (io === null) start();

    return () => {
      stop();
      resizeObserver.disconnect();
      io?.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return (
    <canvas
      ref={canvasRef}
      className={`gothicworks ${className}`.trim()}
      aria-hidden="true"
      role="presentation"
    />
  );
}
