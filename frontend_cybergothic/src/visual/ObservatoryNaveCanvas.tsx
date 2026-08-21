import { useEffect, useRef } from 'react';
import { useReducedMotion } from '../motion/useReducedMotion';

const DPR_MAX = 2;

/** Overview backdrop: an observatory nave, astrolabe rings, and distant buttresses. */
export function ObservatoryNaveCanvas({ className = '' }: { className?: string }) {
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

    const astrolabe = (x: number, y: number, radius: number, rotation: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.24;
      ctx.lineWidth = 1.1;
      for (const scale of [1, 0.76, 0.49]) {
        ctx.beginPath();
        ctx.arc(0, 0, radius * scale, 0, Math.PI * 2);
        ctx.stroke();
      }
      for (let mark = 0; mark < 16; mark += 1) {
        const angle = (mark / 16) * Math.PI * 2;
        const inner = radius * 0.84;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
        ctx.lineTo(Math.cos(angle) * radius, Math.sin(angle) * radius);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(-radius * 0.9, 0);
      ctx.lineTo(radius * 0.9, 0);
      ctx.stroke();
      ctx.restore();
    };

    const orbit = (x: number, y: number, radiusX: number, radiusY: number, rotation: number, color: string, alpha: number) => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(0, 0, radiusX, radiusY, 0, 0, Math.PI * 2);
      ctx.stroke();
      const planetAngle = phase * 0.18;
      ctx.beginPath();
      ctx.arc(Math.cos(planetAngle) * radiusX, Math.sin(planetAngle) * radiusY, 3, 0, Math.PI * 2);
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
        const outer = radius * (tooth % 2 === 0 ? 1.12 : 0.92);
        const px = Math.cos(angle) * outer;
        const py = Math.sin(angle) * outer;
        if (tooth === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, radius * 0.16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const draw = () => {
      if (!width || !height) return;
      ctx.clearRect(0, 0, width, height);
      const driftX = pointerX * 13;
      const driftY = pointerY * 9;

      ctx.save();
      ctx.translate(driftX * 0.14, driftY * 0.12);
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.17;
      ctx.lineWidth = 1;
      for (let x = -height; x < width + height; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - height * 0.28, height);
        ctx.stroke();
      }
      ctx.restore();

      const naveX = width * 0.79 + driftX * 0.44;
      const naveY = height * 0.12 + driftY * 0.35;
      const naveWidth = Math.min(width * 0.27, 300);
      const naveHeight = Math.min(height * 0.68, 540);
      ctx.save();
      ctx.translate(naveX, naveY);
      ctx.strokeStyle = brass;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(-naveWidth * 0.5, naveHeight);
      ctx.lineTo(-naveWidth * 0.5, naveHeight * 0.28);
      ctx.quadraticCurveTo(-naveWidth * 0.36, naveHeight * 0.02, 0, 0);
      ctx.quadraticCurveTo(naveWidth * 0.36, naveHeight * 0.02, naveWidth * 0.5, naveHeight * 0.28);
      ctx.lineTo(naveWidth * 0.5, naveHeight);
      ctx.stroke();
      for (let rib = -2; rib <= 2; rib += 1) {
        const x = (rib / 4) * naveWidth;
        ctx.beginPath();
        ctx.moveTo(x, naveHeight);
        ctx.quadraticCurveTo(x * 0.7, naveHeight * 0.32, 0, naveHeight * 0.14);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(-naveWidth * 0.58, naveHeight * 0.66);
      ctx.lineTo(naveWidth * 0.58, naveHeight * 0.66);
      ctx.stroke();
      ctx.restore();

      astrolabe(naveX + driftX * 0.1, naveY + naveHeight * 0.46, Math.min(naveWidth * 0.25, 62), phase * 0.038);
      orbit(width * 0.21 + driftX * 0.78, height * 0.7 + driftY * 0.68, Math.min(width * 0.19, 220), Math.min(height * 0.08, 72), -0.24 + phase * 0.016, brass, 0.25);
      orbit(width * 0.3 + driftX * 0.58, height * 0.66 + driftY * 0.46, Math.min(width * 0.13, 160), Math.min(height * 0.055, 48), 0.34 - phase * 0.022, gold, 0.2);
      gear(width * 0.15 + driftX * 0.72, height * 0.27 + driftY * 0.58, Math.min(width, height) * 0.09, 16, phase * 0.035, gold, 0.27);
      gear(width * 0.25 + driftX * 0.56, height * 0.36 + driftY * 0.44, Math.min(width, height) * 0.055, 12, -phase * 0.053, brass, 0.28);

      ctx.save();
      ctx.translate(width * 0.08 + driftX * 0.62, height * 0.36 + driftY * 0.45);
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.17;
      ctx.lineWidth = 1;
      for (let tower = 0; tower < 3; tower += 1) {
        const x = tower * 48;
        const towerHeight = 104 - tower * 18;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x + 13, -towerHeight);
        ctx.lineTo(x + 26, 0);
        ctx.lineTo(x + 26, 64);
        ctx.lineTo(x, 64);
        ctx.closePath();
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 1;
      const stars = Math.max(10, Math.min(24, Math.floor(width / 62)));
      for (let star = 0; star < stars; star += 1) {
        const x = (star * 83 + 37) % Math.max(1, width);
        const y = 42 + ((star * 47) % Math.max(1, Math.floor(height * 0.4)));
        const twinkle = 1 + Math.sin(phase * 0.8 + star) * 0.35;
        ctx.beginPath();
        ctx.arc(x + driftX * 0.24, y + driftY * 0.18, twinkle, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      const cometX = ((phase * 26) % (width + 180)) - 90;
      const cometY = height * 0.16 + Math.sin(phase * 0.34) * 34;
      ctx.strokeStyle = gold;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cometX - 48 + driftX * 0.28, cometY + 22 + driftY * 0.2);
      ctx.lineTo(cometX + driftX * 0.28, cometY + driftY * 0.2);
      ctx.stroke();
      ctx.fillStyle = gold;
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.arc(cometX + driftX * 0.28, cometY + driftY * 0.2, 2.2, 0, Math.PI * 2);
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

  return <canvas ref={canvasRef} className={`observatory-nave ${className}`.trim()} aria-hidden="true" role="presentation" />;
}
