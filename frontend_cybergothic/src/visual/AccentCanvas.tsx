/**
 * 首屏装饰 Canvas — 斜向网格 + 稀疏节点连线，脉冲沿连线缓慢移动。
 *
 * 工程约束（§6）：
 * - ResizeObserver 处理尺寸，DPR 上限 2。
 * - IntersectionObserver 离开视口即 cancelAnimationFrame。
 * - 页面隐藏（切后台）时暂停。
 * - `prefers-reduced-motion` 时只绘制一帧静态图。
 * - 纯装饰层，aria-hidden；核心信息不在这里呈现（§5.5）。
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

interface AccentCanvasProps {
  /** 参与绘制的节点数（来自真实集群规模），仅影响装饰密度。 */
  nodeCount?: number;
  /** 是否有活动任务：有则脉冲更快一点。 */
  active?: boolean;
  className?: string;
}

interface Node {
  x: number;
  y: number;
  r: number;
  primary: boolean;
}

const DPR_MAX = 2;

export function AccentCanvas({ nodeCount = 3, active = false, className = '' }: AccentCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  // 用 ref 传递可变参数，避免每次 prop 变化都重建整个动画循环。
  const paramsRef = useRef({ nodeCount, active });
  paramsRef.current = { nodeCount, active };

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const context = el.getContext('2d');
    if (!context) return;

    // 下面的函数声明会被提升，TypeScript 不会在其内部沿用外层的 null 收窄，
    // 所以这里把收窄结果固定成非空 const 再用。
    const canvas: HTMLCanvasElement = el;
    const ctx: CanvasRenderingContext2D = context;

    let width = 0;
    let height = 0;
    let frame = 0;
    let visible = true;
    let running = false;
    let phase = 0;

    const accent = getComputedStyle(canvas).getPropertyValue('--accent').trim() || '#c7ff3d';
    const lineColor = 'rgba(255,255,255,0.10)';

    function layoutNodes(): Node[] {
      const count = Math.max(2, Math.min(6, paramsRef.current.nodeCount));
      const nodes: Node[] = [];
      // 主节点固定在偏左上，其余沿斜向弧线分布。
      nodes.push({ x: width * 0.28, y: height * 0.34, r: 7, primary: true });
      for (let i = 1; i < count; i += 1) {
        const t = i / Math.max(1, count - 1);
        nodes.push({
          x: width * (0.42 + 0.44 * t),
          y: height * (0.24 + 0.56 * t),
          r: 4.5,
          primary: false,
        });
      }
      return nodes;
    }

    function drawDiagonals() {
      // 低强度重复斜线，作为材质而非主体（§2.5）。
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.035)';
      ctx.lineWidth = 1;
      const step = 26;
      for (let x = -height; x < width + height; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, height);
        ctx.lineTo(x + height, 0);
        ctx.stroke();
      }
      ctx.restore();
    }

    function draw(t: number) {
      ctx.clearRect(0, 0, width, height);
      drawDiagonals();

      const nodes = layoutNodes();
      const master = nodes[0];
      if (!master) return;

      // 连线
      ctx.save();
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1.25;
      for (let i = 1; i < nodes.length; i += 1) {
        const n = nodes[i];
        if (!n) continue;
        ctx.beginPath();
        ctx.moveTo(master.x, master.y);
        ctx.lineTo(n.x, n.y);
        ctx.stroke();
      }
      ctx.restore();

      // 沿连线移动的脉冲点
      for (let i = 1; i < nodes.length; i += 1) {
        const n = nodes[i];
        if (!n) continue;
        const offset = (t + i * 0.31) % 1;
        const px = master.x + (n.x - master.x) * offset;
        const py = master.y + (n.y - master.y) * offset;
        const fade = Math.sin(offset * Math.PI);
        ctx.save();
        ctx.globalAlpha = 0.5 * fade;
        ctx.fillStyle = accent;
        ctx.beginPath();
        ctx.arc(px, py, 2.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // 节点本体：主节点用主色描边方块，从节点用细线圆点
      nodes.forEach((n, i) => {
        ctx.save();
        if (n.primary) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 2;
          ctx.strokeRect(n.x - n.r, n.y - n.r, n.r * 2, n.r * 2);
          ctx.globalAlpha = 0.16;
          ctx.fillStyle = accent;
          ctx.fillRect(n.x - n.r, n.y - n.r, n.r * 2, n.r * 2);
        } else {
          const breathe = 1 + 0.12 * Math.sin(t * Math.PI * 2 + i);
          ctx.strokeStyle = 'rgba(255,255,255,0.45)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r * breathe, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      });
    }

    function tick() {
      if (!running) return;
      // 有活动任务时脉冲略快；数值保持低速，避免抢夺注意力。
      phase += paramsRef.current.active ? 0.0038 : 0.0016;
      draw(phase % 1);
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
      // 尺寸变化后立即重绘一帧，避免静态模式下留白。
      draw(phase % 1);
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
            { threshold: 0.05 },
          )
        : null;
    io?.observe(canvas);

    const onVisibility = () => {
      if (document.visibilityState === 'visible' && visible) start();
      else stop();
    };
    document.addEventListener('visibilitychange', onVisibility);

    if (reduced) {
      // 静态单帧：保留构图但不消耗 GPU。
      draw(0.35);
    } else if (io === null) {
      start();
    }

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
      className={`accent-canvas ${className}`.trim()}
      aria-hidden="true"
      role="presentation"
    />
  );
}
