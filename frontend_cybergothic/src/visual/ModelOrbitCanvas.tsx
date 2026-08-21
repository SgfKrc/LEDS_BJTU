import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Model page backdrop: nested orbital tracks and a sparse runtime topology. */
export function ModelOrbitCanvas({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let phase = 0;
    let running = false;
    let visible = true;
    const style = getComputedStyle(canvas);
    const brass = style.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const gold = style.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const line = style.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const draw = (time: number) => {
      ctx.clearRect(0, 0, width, height);
      const base = Math.min(width, height);
      const cx = width * 0.66;
      const cy = height * 0.42;

      ctx.save();
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      for (let i = 0; i < 9; i += 1) {
        const y = height * (0.08 + i * 0.12);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y + Math.sin(i * 0.8) * 12);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(time * 0.07);
      ctx.strokeStyle = brass;
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.23;
      for (let ring = 0; ring < 4; ring += 1) {
        ctx.beginPath();
        ctx.ellipse(0, 0, base * (0.2 + ring * 0.08), base * (0.08 + ring * 0.035), ring * 0.22, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 0.5;
      for (let node = 0; node < 12; node += 1) {
        const angle = node * Math.PI / 6;
        const radius = base * (0.18 + (node % 3) * 0.07);
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius * 0.43;
        ctx.beginPath();
        ctx.arc(x, y, 2.3 + (node % 2), 0, Math.PI * 2);
        ctx.fillStyle = node % 3 === 0 ? gold : brass;
        ctx.fill();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(width * 0.18, height * 0.76);
      ctx.rotate(-time * 0.11);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, base * 0.18, 0, Math.PI * 2);
      ctx.stroke();
      for (let tooth = 0; tooth < 16; tooth += 1) {
        const angle = tooth * Math.PI / 8;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * base * 0.15, Math.sin(angle) * base * 0.15);
        ctx.lineTo(Math.cos(angle) * base * 0.21, Math.sin(angle) * base * 0.21);
        ctx.stroke();
      }
      ctx.restore();
    };

    const tick = () => {
      if (!running) return;
      phase += 0.014;
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
    if (reduced) draw(0.2);
    else if (!observer) start();

    return () => {
      stop();
      resizeObserver.disconnect();
      observer?.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className={`model-orbit-canvas ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
