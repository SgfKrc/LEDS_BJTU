import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Help backdrop: lancet windows, leaded glass panels, and a slow reading-light sweep. */
export function StainedGlassScriptoriumCanvas({ className = '' }: { className?: string }) {
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

    const windowPanel = (x: number, y: number, panelWidth: number, panelHeight: number, offset: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(-panelWidth * 0.5, panelHeight);
      ctx.lineTo(-panelWidth * 0.5, panelHeight * 0.22);
      ctx.quadraticCurveTo(-panelWidth * 0.42, -panelHeight * 0.08, 0, -panelHeight * 0.14);
      ctx.quadraticCurveTo(panelWidth * 0.42, -panelHeight * 0.08, panelWidth * 0.5, panelHeight * 0.22);
      ctx.lineTo(panelWidth * 0.5, panelHeight);
      ctx.closePath();
      ctx.stroke();

      const rows = 5;
      const cols = 3;
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const cellWidth = panelWidth * 0.25;
          const cellHeight = panelHeight * 0.13;
          const cellX = -panelWidth * 0.37 + col * panelWidth * 0.27;
          const cellY = panelHeight * 0.16 + row * panelHeight * 0.15;
          const color = (row + col) % 3 === 0 ? brass : gold;
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.06 + Math.max(0, Math.sin(phase * 0.25 + offset + row * 0.6)) * 0.025;
          ctx.fillRect(cellX, cellY, cellWidth, cellHeight);
          ctx.strokeStyle = line;
          ctx.globalAlpha = 0.24;
          ctx.strokeRect(cellX, cellY, cellWidth, cellHeight);
        }
      }
      ctx.restore();
    };

    const rose = (x: number, y: number, radius: number, rotation: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.stroke();
      for (let petal = 0; petal < 12; petal += 1) {
        const angle = (petal / 12) * Math.PI * 2;
        ctx.beginPath();
        ctx.ellipse(Math.cos(angle) * radius * 0.38, Math.sin(angle) * radius * 0.38, radius * 0.42, radius * 0.18, angle, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 12;
      const driftY = pointerY * 8;

      ctx.save();
      ctx.translate(driftX * 0.14, driftY * 0.12);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.15;
      ctx.lineWidth = 1;
      for (let y = height * 0.08; y < height; y += 44) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y + 18);
        ctx.stroke();
      }
      ctx.restore();

      const panelWidth = Math.min(width * 0.16, 160);
      const panelHeight = Math.min(height * 0.62, 500);
      windowPanel(width * 0.7 + driftX * 0.34, height * 0.18 + driftY * 0.3, panelWidth, panelHeight, 0);
      windowPanel(width * 0.87 + driftX * 0.46, height * 0.24 + driftY * 0.38, panelWidth * 0.82, panelHeight * 0.83, 1.7);
      rose(width * 0.2 + driftX * 0.72, height * 0.31 + driftY * 0.56, Math.min(width, height) * 0.09, phase * 0.032);
      rose(width * 0.32 + driftX * 0.54, height * 0.41 + driftY * 0.42, Math.min(width, height) * 0.052, -phase * 0.052);

      ctx.save();
      ctx.translate(width * 0.2 + driftX * 0.72, height * 0.7 + driftY * 0.62);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1;
      for (let page = 0; page < 4; page += 1) {
        ctx.strokeRect(page * 12, -page * 8, 112, 54);
        ctx.beginPath();
        ctx.moveTo(page * 12 + 12, -page * 8 + 16);
        ctx.lineTo(page * 12 + 92, -page * 8 + 16);
        ctx.moveTo(page * 12 + 12, -page * 8 + 28);
        ctx.lineTo(page * 12 + 74, -page * 8 + 28);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      const sweep = ((phase * 34) % (width + 180)) - 90;
      ctx.translate(sweep + driftX * 0.18, 0);
      ctx.fillStyle = gold;
      ctx.globalAlpha = 0.06;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(110, 0);
      ctx.lineTo(310, height);
      ctx.lineTo(190, height);
      ctx.closePath();
      ctx.fill();
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
      phase += 0.009;
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

  return <canvas ref={canvasRef} className={`stained-glass-scriptorium ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
