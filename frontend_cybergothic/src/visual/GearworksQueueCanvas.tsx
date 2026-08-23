import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';
import { drawEtchedFrame, drawRoseWindow } from './canvasOrnaments';

const DPR_MAX = 2;

/** Tasks backdrop: queue gates, moving rack teeth, and deliberately restrained queue pulses. */
export function GearworksQueueCanvas({ className = '' }: { className?: string }) {
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
    const gold = styles.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const brass = styles.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const line = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

    const rack = (x: number, y: number, length: number, step: number, offset: number, color: string, alpha: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(length, 0);
      ctx.stroke();
      for (let tooth = -1; tooth < Math.ceil(length / step) + 1; tooth += 1) {
        const toothX = tooth * step + offset;
        ctx.beginPath();
        ctx.moveTo(toothX, 0);
        ctx.lineTo(toothX + step * 0.34, -7);
        ctx.lineTo(toothX + step * 0.68, 0);
        ctx.stroke();
      }
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
        const outer = radius * (tooth % 2 === 0 ? 1.12 : 0.92);
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
      const driftX = pointerX * 12;
      const driftY = pointerY * 9;

      drawEtchedFrame(ctx, width * 0.065, height * 0.1, width * 0.87, height * 0.75, { stroke: line, alpha: 0.14 });

      ctx.save();
      ctx.translate(driftX * 0.16, driftY * 0.12);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.17;
      ctx.lineWidth = 1;
      for (let y = height * 0.08; y < height; y += 42) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y + width * 0.11);
        ctx.stroke();
      }
      ctx.restore();

      gear(width * 0.19 + driftX * 0.7, height * 0.28 + driftY * 0.54, Math.min(width, height) * 0.11, 18, phase * 0.052, gold, 0.3);
      gear(width * 0.32 + driftX * 0.54, height * 0.37 + driftY * 0.4, Math.min(width, height) * 0.07, 13, -phase * 0.082, brass, 0.28);

      const gateX = width * 0.77 + driftX * 0.48;
      const gateY = height * 0.2 + driftY * 0.38;
      const gateWidth = Math.min(width * 0.23, 245);
      const gateHeight = Math.min(height * 0.53, 430);
      ctx.save();
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.24;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(gateX - gateWidth * 0.5, gateY, gateWidth, gateHeight);
      ctx.beginPath();
      ctx.moveTo(gateX - gateWidth * 0.5, gateY + gateHeight * 0.2);
      ctx.lineTo(gateX, gateY - 34);
      ctx.lineTo(gateX + gateWidth * 0.5, gateY + gateHeight * 0.2);
      ctx.stroke();
      for (let i = 0; i < 4; i += 1) {
        const y = gateY + gateHeight * (0.31 + i * 0.15);
        ctx.beginPath();
        ctx.moveTo(gateX - gateWidth * 0.36, y);
        ctx.lineTo(gateX + gateWidth * 0.36, y);
        ctx.stroke();
      }
      drawRoseWindow(ctx, gateX, gateY + gateHeight * 0.23, Math.min(gateWidth, gateHeight) * 0.18, {
        stroke: gold,
        accent: brass,
        alpha: 0.2,
        rotation: -phase * 0.08,
        petals: 8,
      });
      ctx.restore();

      const laneLength = Math.min(width * 0.46, 520);
      rack(width * 0.08 + driftX * 0.74, height * 0.68 + driftY * 0.62, laneLength, 18, (phase * 24) % 18, brass, 0.24);
      rack(width * 0.13 + driftX * 0.55, height * 0.76 + driftY * 0.5, laneLength * 0.84, 15, (-phase * 30) % 15, gold, 0.19);

      ctx.save();
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 1.1;
      const pulseX = width * 0.08 + ((phase * 75) % Math.max(1, laneLength));
      for (let lane = 0; lane < 3; lane += 1) {
        const y = height * (0.49 + lane * 0.1) + driftY * 0.32;
        ctx.beginPath();
        ctx.arc(pulseX + lane * 28, y, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.translate(width * 0.26 + driftX * 0.48, height * 0.3 + driftY * 0.3);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i += 1) {
        ctx.strokeRect(i * 18, i * 12, 76, 52);
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
      phase += 0.014;
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

  return <canvas ref={canvasRef} className={`gearworks-queue ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
