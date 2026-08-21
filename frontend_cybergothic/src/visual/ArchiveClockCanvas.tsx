import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Audit backdrop: clock tower, archive folios, and slow counter-rotating gears. */
export function ArchiveClockCanvas({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let width = 0;
    let height = 0;
    let phase = 0;
    let pointerX = 0;
    let pointerY = 0;
    let frame = 0;
    let visible = true;
    let running = false;

    const styles = getComputedStyle(canvas);
    const brass = styles.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const gold = styles.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const line = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const gear = (x: number, y: number, radius: number, teeth: number, rotation: number, color: string, alpha: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let tooth = 0; tooth < teeth; tooth += 1) {
        const angle = (tooth / teeth) * Math.PI * 2;
        const outer = radius * (tooth % 2 === 0 ? 1.11 : 0.94);
        const px = Math.cos(angle) * outer;
        const py = Math.sin(angle) * outer;
        if (tooth === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.globalAlpha = alpha * 0.72;
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.72, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 14;
      const driftY = pointerY * 10;

      ctx.save();
      ctx.translate(driftX * 0.18, driftY * 0.16);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1;
      for (let y = height * 0.1; y < height; y += 38) {
        ctx.beginPath();
        ctx.moveTo(width * 0.06, y);
        ctx.lineTo(width * 0.42, y - height * 0.06);
        ctx.stroke();
      }
      ctx.restore();

      const towerX = width * 0.8 + driftX * 0.42;
      const towerY = height * 0.16 + driftY * 0.35;
      const towerWidth = Math.min(width * 0.22, 220);
      const towerHeight = Math.min(height * 0.58, 500);
      ctx.save();
      ctx.translate(towerX, towerY);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.34;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(-towerWidth * 0.5, towerHeight);
      ctx.lineTo(-towerWidth * 0.5, towerHeight * 0.15);
      ctx.lineTo(0, 0);
      ctx.lineTo(towerWidth * 0.5, towerHeight * 0.15);
      ctx.lineTo(towerWidth * 0.5, towerHeight);
      ctx.stroke();
      for (let side = -1; side <= 1; side += 2) {
        ctx.beginPath();
        ctx.moveTo(side * towerWidth * 0.27, towerHeight * 0.82);
        ctx.lineTo(side * towerWidth * 0.27, towerHeight * 0.36);
        ctx.arc(side * towerWidth * 0.16, towerHeight * 0.36, towerWidth * 0.11, Math.PI, 0, side < 0);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(towerX, towerY + towerHeight * 0.42);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.34;
      ctx.lineWidth = 1.1;
      const clockRadius = Math.min(towerWidth * 0.27, 56);
      ctx.beginPath();
      ctx.arc(0, 0, clockRadius, 0, Math.PI * 2);
      ctx.stroke();
      for (let tick = 0; tick < 12; tick += 1) {
        const angle = (tick / 12) * Math.PI * 2 - Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * clockRadius * 0.82, Math.sin(angle) * clockRadius * 0.82);
        ctx.lineTo(Math.cos(angle) * clockRadius * 0.96, Math.sin(angle) * clockRadius * 0.96);
        ctx.stroke();
      }
      ctx.rotate(phase * 0.08);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(clockRadius * 0.56, 0);
      ctx.stroke();
      ctx.rotate(-phase * 0.08 + phase * 0.4);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -clockRadius * 0.7);
      ctx.stroke();
      ctx.restore();

      gear(width * 0.17 + driftX * 0.82, height * 0.72 + driftY * 0.72, Math.min(width, height) * 0.11, 22, phase * 0.12, brass, 0.26);
      gear(width * 0.29 + driftX * 0.65, height * 0.78 + driftY * 0.58, Math.min(width, height) * 0.075, 16, -phase * 0.18, gold, 0.2);

      ctx.save();
      ctx.translate(width * 0.52 + driftX * 0.54, height * 0.77 + driftY * 0.42);
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.16;
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i += 1) {
        ctx.strokeRect(i * 7, -i * 7, 118, 54);
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
      phase += 0.012;
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
    const onPointerMove = (event: PointerEvent) => {
      pointerX = event.clientX / Math.max(1, window.innerWidth) - 0.5;
      pointerY = event.clientY / Math.max(1, window.innerHeight) - 0.5;
      if (reduced) draw();
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
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    resize();
    if (reduced) draw();
    else if (!io) start();
    return () => {
      stop();
      observer.disconnect();
      io?.disconnect();
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className={`archive-clock ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
