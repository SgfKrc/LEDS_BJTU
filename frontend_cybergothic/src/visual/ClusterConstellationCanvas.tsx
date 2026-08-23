import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';
import { drawEtchedFrame, drawRoseWindow } from './canvasOrnaments';

const DPR_MAX = 2;

/** Cluster page backdrop: a restrained, layered node constellation with slow parallax. */
export function ClusterConstellationCanvas({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let visible = true;
    let running = false;
    let phase = 0;
    let pointerX = 0;
    let pointerY = 0;
    let resizeObserver: ResizeObserver | null = null;

    const styles = getComputedStyle(canvas);
    const brass = styles.getPropertyValue('--gothic-brass').trim() || '#80672f';
    const gold = styles.getPropertyValue('--gothic-gold-bright').trim() || '#ecd184';
    const line = styles.getPropertyValue('--gothic-line-strong').trim() || '#3a3350';

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

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 10;
      const driftY = pointerY * 8;

      drawEtchedFrame(ctx, width * 0.07, height * 0.1, width * 0.84, height * 0.74, { stroke: line, alpha: 0.14 });

      ctx.save();
      ctx.translate(driftX * 0.24, driftY * 0.2);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1;
      for (let x = -height; x < width + height; x += 54) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - height * 0.62, height);
        ctx.stroke();
      }
      ctx.restore();

      const centerX = width * 0.73 + driftX * 0.5;
      const centerY = height * 0.38 + driftY * 0.45;
      const orbitA = Math.min(width, height) * 0.19;
      const orbitB = Math.min(width, height) * 0.31;

      drawRoseWindow(ctx, centerX, centerY, Math.min(width, height) * 0.105, {
        stroke: gold,
        accent: brass,
        alpha: 0.2,
        rotation: phase * 0.05,
        petals: 10,
      });

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(-0.18 + pointerX * 0.02);
      ctx.strokeStyle = brass;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.24;
      ctx.beginPath();
      ctx.ellipse(0, 0, orbitA, orbitA * 0.46, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 0.13;
      ctx.beginPath();
      ctx.ellipse(0, 0, orbitB, orbitB * 0.44, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      const nodes = [
        { x: centerX, y: centerY, r: 7, tone: gold },
        { x: centerX - orbitA * 0.72, y: centerY - orbitA * 0.18, r: 4, tone: gold },
        { x: centerX + orbitA * 0.58, y: centerY + orbitA * 0.22, r: 4, tone: gold },
        { x: centerX - orbitB * 0.46, y: centerY + orbitB * 0.28, r: 3, tone: brass },
        { x: centerX + orbitB * 0.64, y: centerY - orbitB * 0.08, r: 3, tone: brass },
      ];

      ctx.save();
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.17;
      ctx.setLineDash([3, 8]);
      nodes.slice(1).forEach((node) => {
        ctx.beginPath();
        ctx.moveTo(nodes[0].x, nodes[0].y);
        ctx.lineTo(node.x, node.y);
        ctx.stroke();
      });
      ctx.restore();

      nodes.forEach((node, index) => {
        ctx.save();
        ctx.globalAlpha = index === 0 ? 0.65 : 0.42;
        ctx.fillStyle = node.tone;
        ctx.strokeStyle = node.tone;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r + (index === 0 ? Math.sin(phase) * 1.5 : 0), 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha *= 0.6;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r * 2.25, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      });

      ctx.save();
      ctx.translate(width * 0.16 + driftX * 0.72, height * 0.74 + driftY * 0.65);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.14;
      ctx.lineWidth = 1.2;
      for (let i = 0; i < 7; i += 1) {
        const radius = 18 + i * 13;
        ctx.beginPath();
        ctx.arc(0, 0, radius, -Math.PI * 0.82, Math.PI * 0.1);
        ctx.stroke();
      }
      ctx.restore();
    };

    const tick = () => {
      if (!running) return;
      phase += 0.018;
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
    const onVisibility = () => {
      if (document.visibilityState === 'visible' && visible) start();
      else stop();
    };

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    const io = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver((entries) => {
      visible = entries.some((entry) => entry.isIntersecting);
      if (visible && document.visibilityState === 'visible') start();
      else stop();
    }, { threshold: 0.01 });
    io?.observe(canvas);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    resize();
    if (reduced) draw();
    else if (!io) start();

    return () => {
      stop();
      resizeObserver?.disconnect();
      io?.disconnect();
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className={`cluster-constellation ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
