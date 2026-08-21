import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Settings backdrop: a low-motion archive cabinet, locking dials, and indexed spines. */
export function ClockworkArchiveCanvas({ className = '' }: { className?: string }) {
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

    const dial = (x: number, y: number, radius: number, rotation: number, color: string, alpha: number, marks: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.73, 0, Math.PI * 2);
      ctx.stroke();
      for (let mark = 0; mark < marks; mark += 1) {
        const angle = (mark / marks) * Math.PI * 2;
        const inner = radius * (mark % 4 === 0 ? 0.76 : 0.85);
        const outer = radius * 0.96;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
        ctx.lineTo(Math.cos(angle) * outer, Math.sin(angle) * outer);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const gear = (x: number, y: number, radius: number, teeth: number, rotation: number, color: string, alpha: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let tooth = 0; tooth < teeth * 2; tooth += 1) {
        const angle = (tooth / (teeth * 2)) * Math.PI * 2;
        const outer = radius * (tooth % 2 === 0 ? 1.1 : 0.92);
        const px = Math.cos(angle) * outer;
        const py = Math.sin(angle) * outer;
        if (tooth === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.68, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 14;
      const driftY = pointerY * 10;

      // Deep shelf planes give the scene its slow parallax without competing with controls.
      ctx.save();
      ctx.translate(driftX * 0.14, driftY * 0.12);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.16;
      ctx.lineWidth = 1;
      for (let row = 0; row < 6; row += 1) {
        const y = height * (0.08 + row * 0.15);
        ctx.beginPath();
        ctx.moveTo(width * 0.04, y);
        ctx.lineTo(width * 0.46, y - 16);
        ctx.stroke();
      }
      ctx.restore();

      const cabinetX = width * 0.79 + driftX * 0.44;
      const cabinetY = height * 0.16 + driftY * 0.34;
      const cabinetWidth = Math.min(width * 0.25, 270);
      const cabinetHeight = Math.min(height * 0.62, 510);
      ctx.save();
      ctx.translate(cabinetX, cabinetY);
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(-cabinetWidth * 0.5, 0, cabinetWidth, cabinetHeight);
      ctx.beginPath();
      ctx.moveTo(-cabinetWidth * 0.56, 0);
      ctx.lineTo(0, -42);
      ctx.lineTo(cabinetWidth * 0.56, 0);
      ctx.stroke();
      for (let shelf = 1; shelf < 5; shelf += 1) {
        const y = (cabinetHeight / 5) * shelf;
        ctx.beginPath();
        ctx.moveTo(-cabinetWidth * 0.44, y);
        ctx.lineTo(cabinetWidth * 0.44, y);
        ctx.stroke();
      }
      for (let spine = 0; spine < 9; spine += 1) {
        const x = -cabinetWidth * 0.41 + spine * (cabinetWidth * 0.1);
        const row = spine % 3;
        const top = 18 + row * (cabinetHeight / 5);
        const bookHeight = cabinetHeight * (0.12 + (spine % 4) * 0.012);
        ctx.strokeRect(x, top, cabinetWidth * 0.064, bookHeight);
      }
      ctx.restore();

      const lockX = width * 0.57 + driftX * 0.68;
      const lockY = height * 0.68 + driftY * 0.52;
      ctx.save();
      ctx.translate(lockX, lockY);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.27;
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      ctx.arc(0, -14, 20, Math.PI, 0);
      ctx.lineTo(20, 22);
      ctx.lineTo(-20, 22);
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 2, 4, 0, Math.PI * 2);
      ctx.moveTo(0, 6);
      ctx.lineTo(0, 15);
      ctx.stroke();
      ctx.restore();

      dial(width * 0.2 + driftX * 0.78, height * 0.75 + driftY * 0.67, Math.min(width, height) * 0.115, phase * 0.032, brass, 0.26, 20);
      dial(width * 0.31 + driftX * 0.58, height * 0.7 + driftY * 0.5, Math.min(width, height) * 0.075, -phase * 0.05, gold, 0.22, 16);
      gear(width * 0.17 + driftX * 0.7, height * 0.28 + driftY * 0.48, Math.min(width, height) * 0.075, 15, -phase * 0.045, gold, 0.28);
      gear(width * 0.29 + driftX * 0.54, height * 0.37 + driftY * 0.38, Math.min(width, height) * 0.052, 11, phase * 0.065, brass, 0.27);

      ctx.save();
      ctx.translate(width * 0.12 + driftX * 0.52, height * 0.3 + driftY * 0.3);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.15;
      ctx.lineWidth = 1;
      for (let folio = 0; folio < 4; folio += 1) {
        ctx.strokeRect(folio * 15, folio * 10, 96, 48);
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

  return <canvas ref={canvasRef} className={`clockwork-archive ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
