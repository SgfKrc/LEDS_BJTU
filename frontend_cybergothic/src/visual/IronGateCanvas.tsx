import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Account backdrop: a quiet iron gate, lock ring, and offset chain layers. */
export function IronGateCanvas({ className = '' }: { className?: string }) {
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
    let spin = 0;

    const styles = getComputedStyle(canvas);
    const gold = styles.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const brass = styles.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const stone = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const cx = width * 0.79;
      const cy = height * 0.39;
      const radius = Math.min(width, height) * 0.2;

      ctx.save();
      ctx.strokeStyle = stone;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1;
      for (let i = -2; i < 11; i += 1) {
        const x = width * 0.54 + i * 44;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - height * 0.2, height);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(spin * 0.12);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.arc(0, 0, radius, Math.PI * 0.12, Math.PI * 1.88);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.76, Math.PI * 0.12, Math.PI * 1.88);
      ctx.stroke();
      for (let i = 0; i < 14; i += 1) {
        const angle = (i / 14) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * radius * 0.86, Math.sin(angle) * radius * 0.86);
        ctx.lineTo(Math.cos(angle) * radius, Math.sin(angle) * radius);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(cx, cy + radius * 0.12);
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.38, Math.PI, Math.PI * 2);
      ctx.lineTo(radius * 0.38, radius * 0.47);
      ctx.lineTo(-radius * 0.38, radius * 0.47);
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, radius * 0.48, radius * 0.09, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 2;
      const links = 7;
      for (let i = 0; i < links; i += 1) {
        const x = width * 0.16 + i * 26;
        const y = height * 0.79 + Math.sin(spin * 0.45 + i * 0.6) * 8;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-0.25 + Math.sin(spin * 0.3 + i) * 0.06);
        ctx.beginPath();
        ctx.ellipse(0, 0, 14, 7, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      ctx.restore();
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      width = rect.width;
      height = rect.height;
      const dpr = Math.min(DPR_MAX, window.devicePixelRatio || 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    };
    const tick = () => {
      if (!running) return;
      spin += 0.012;
      draw();
      frame = requestAnimationFrame(tick);
    };
    const start = () => {
      if (running || reduced) return;
      running = true;
      frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      running = false;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    const io = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver((entries) => {
      visible = entries.some((entry) => entry.isIntersecting);
      if (visible && document.visibilityState === 'visible') start();
      else stop();
    }, { threshold: 0.01 });
    io?.observe(canvas);
    const onVisibility = () => document.visibilityState === 'visible' && visible ? start() : stop();
    document.addEventListener('visibilitychange', onVisibility);
    resize();
    if (reduced) draw();
    else if (!io) start();
    return () => {
      stop();
      observer.disconnect();
      io?.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className={`iron-gate ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
