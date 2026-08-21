import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** 生图页专用背景：光圈、图像版片和缓慢转动的炼金工坊机械环。 */
export function FoundryCanvas({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let running = false;
    let visible = true;
    let phase = 0;
    const style = getComputedStyle(canvas);
    const gold = style.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const brass = style.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const line = style.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const draw = (t: number) => {
      ctx.clearRect(0, 0, width, height);
      const cx = width * 0.72;
      const cy = height * 0.42;
      const base = Math.min(width, height);

      ctx.save();
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i += 1) {
        const y = height * (0.18 + i * 0.16);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y + Math.sin(i) * 18);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(t * 0.18);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.32;
      ctx.lineWidth = 1.4;
      for (let i = 0; i < 3; i += 1) {
        ctx.beginPath();
        ctx.arc(0, 0, base * (0.18 + i * 0.09), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 0.48;
      for (let i = 0; i < 12; i += 1) {
        const angle = (i / 12) * Math.PI * 2;
        const inner = base * 0.13;
        const outer = base * 0.34;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
        ctx.lineTo(Math.cos(angle + 0.08) * outer, Math.sin(angle + 0.08) * outer);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(width * 0.22, height * 0.78);
      ctx.rotate(-t * 0.08);
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(0, 0, base * 0.2, 0, Math.PI * 2);
      ctx.stroke();
      for (let i = 0; i < 8; i += 1) {
        const angle = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * base * 0.06, Math.sin(angle) * base * 0.06);
        ctx.lineTo(Math.cos(angle) * base * 0.2, Math.sin(angle) * base * 0.2);
        ctx.stroke();
      }
      ctx.restore();
    };

    const tick = () => {
      if (!running) return;
      phase += 0.012;
      draw(phase);
      frame = window.requestAnimationFrame(tick);
    };
    const stop = () => {
      running = false;
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
    };
    const start = () => {
      if (running || reduced) return;
      running = true;
      frame = window.requestAnimationFrame(tick);
    };
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const dpr = Math.min(DPR_MAX, window.devicePixelRatio || 1);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(phase);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();
    const observer = typeof IntersectionObserver === 'undefined'
      ? null
      : new IntersectionObserver((entries) => {
          visible = entries.some((entry) => entry.isIntersecting);
          if (visible && document.visibilityState === 'visible') start();
          else stop();
        }, { threshold: 0.05 });
    observer?.observe(canvas);
    const onVisibility = () => {
      if (visible && document.visibilityState === 'visible') start();
      else stop();
    };
    document.addEventListener('visibilitychange', onVisibility);
    if (reduced) draw(0.4);
    else if (!observer) start();

    return () => {
      stop();
      resizeObserver.disconnect();
      observer?.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className={`foundry-canvas ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
