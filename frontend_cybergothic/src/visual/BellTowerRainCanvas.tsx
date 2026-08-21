import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Activity backdrop: a bell tower, staggered rain lines, and a slow pendulum. */
export function BellTowerRainCanvas({ className = '' }: { className?: string }) {
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
    let running = false;
    let visible = true;

    const styles = getComputedStyle(canvas);
    const brass = styles.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const gold = styles.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const line = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 12;
      const driftY = pointerY * 8;

      ctx.save();
      ctx.translate(driftX * 0.18, driftY * 0.14);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.16;
      ctx.lineWidth = 1;
      for (let x = -height; x < width + height; x += 44) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - height * 0.42, height);
        ctx.stroke();
      }
      ctx.restore();

      const towerX = width * 0.78 + driftX * 0.45;
      const towerBase = height * 0.88 + driftY * 0.34;
      const towerWidth = Math.min(width * 0.22, 220);
      const towerTop = Math.max(70, height * 0.18);

      ctx.save();
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(towerX - towerWidth * 0.48, towerBase);
      ctx.lineTo(towerX - towerWidth * 0.48, towerTop + 42);
      ctx.lineTo(towerX, towerTop);
      ctx.lineTo(towerX + towerWidth * 0.48, towerTop + 42);
      ctx.lineTo(towerX + towerWidth * 0.48, towerBase);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(towerX - towerWidth * 0.62, towerBase);
      ctx.lineTo(towerX + towerWidth * 0.62, towerBase);
      ctx.stroke();
      for (const side of [-1, 1]) {
        const archX = towerX + side * towerWidth * 0.2;
        ctx.beginPath();
        ctx.moveTo(archX - towerWidth * 0.11, towerTop + 108);
        ctx.lineTo(archX - towerWidth * 0.11, towerTop + 72);
        ctx.arc(archX, towerTop + 72, towerWidth * 0.11, Math.PI, 0, side < 0);
        ctx.lineTo(archX + towerWidth * 0.11, towerTop + 108);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(towerX, towerTop + 46);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1.2;
      const bellRadius = Math.min(towerWidth * 0.17, 38);
      ctx.beginPath();
      ctx.arc(0, 2, bellRadius, Math.PI, 0);
      ctx.lineTo(bellRadius * 1.12, bellRadius * 0.65);
      ctx.lineTo(-bellRadius * 1.12, bellRadius * 0.65);
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.translate(towerX, towerTop + 90);
      ctx.rotate(Math.sin(phase * 0.62) * 0.08);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.26;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, Math.min(height * 0.38, 250));
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, Math.min(height * 0.38, 250), 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.globalAlpha = 0.13;
      ctx.strokeStyle = brass;
      ctx.lineWidth = 1;
      const count = Math.min(34, Math.max(18, Math.floor(width / 34)));
      for (let i = 0; i < count; i += 1) {
        const x = ((i * 97 + 23) % Math.max(1, width)) + driftX * (i % 3 === 0 ? 0.7 : 0.26);
        const depth = i % 3;
        const length = 12 + depth * 8;
        const travel = (phase * (0.7 + depth * 0.18) * 40 + i * 39) % Math.max(1, height + 60);
        ctx.beginPath();
        ctx.moveTo(x, travel - length);
        ctx.lineTo(x - 5, travel);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1;
      ctx.translate(width * 0.2 + driftX * 0.72, height * 0.8 + driftY * 0.56);
      for (let i = 0; i < 3; i += 1) {
        ctx.strokeRect(i * 14, -i * 8, 118, 34);
      }
      ctx.restore();

      const wheel = (x: number, y: number, radius: number, teeth: number, rotation: number, color: string, alpha: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let tooth = 0; tooth < teeth * 2; tooth += 1) {
          const angle = (tooth / (teeth * 2)) * Math.PI * 2;
          const outer = radius * (tooth % 2 === 0 ? 1.1 : 0.9);
          const px = Math.cos(angle) * outer;
          const py = Math.sin(angle) * outer;
          if (tooth === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.66, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      };
      wheel(width * 0.18 + driftX * 0.74, height * 0.33 + driftY * 0.56, Math.min(width, height) * 0.095, 17, phase * 0.045, gold, 0.28);
      wheel(width * 0.3 + driftX * 0.56, height * 0.4 + driftY * 0.42, Math.min(width, height) * 0.06, 12, -phase * 0.07, brass, 0.26);
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
      phase += 0.016;
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

  return <canvas ref={canvasRef} className={`bell-tower-rain ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
