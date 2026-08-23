interface OrnamentOptions {
  stroke: string;
  accent?: string;
  alpha?: number;
  rotation?: number;
  petals?: number;
}

/** Draw a compact rose-window motif with radial ribs, petals, and rim rivets. */
export function drawRoseWindow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  { stroke, accent = stroke, alpha = 0.3, rotation = 0, petals = 12 }: OrnamentOptions,
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);
  ctx.strokeStyle = stroke;
  ctx.fillStyle = accent;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1;

  for (const ring of [1, 0.82, 0.48, 0.2]) {
    ctx.beginPath();
    ctx.arc(0, 0, radius * ring, 0, Math.PI * 2);
    ctx.stroke();
  }

  for (let index = 0; index < petals; index += 1) {
    const angle = (index / petals) * Math.PI * 2;
    const next = angle + (Math.PI * 2) / petals;
    const inner = radius * 0.22;
    const middle = radius * 0.68;
    ctx.beginPath();
    ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
    ctx.quadraticCurveTo(
      Math.cos(angle + (next - angle) * 0.32) * middle,
      Math.sin(angle + (next - angle) * 0.32) * middle,
      Math.cos((angle + next) / 2) * radius * 0.82,
      Math.sin((angle + next) / 2) * radius * 0.82,
    );
    ctx.quadraticCurveTo(
      Math.cos(angle + (next - angle) * 0.68) * middle,
      Math.sin(angle + (next - angle) * 0.68) * middle,
      Math.cos(next) * inner,
      Math.sin(next) * inner,
    );
    ctx.stroke();

    if (index % 2 === 0) {
      const rivetAngle = angle + (next - angle) * 0.5;
      ctx.globalAlpha = alpha * 1.35;
      ctx.beginPath();
      ctx.arc(Math.cos(rivetAngle) * radius * 0.92, Math.sin(rivetAngle) * radius * 0.92, 1.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = alpha;
    }
  }
  ctx.restore();
}

/** A restrained etched frame that adds depth without carrying content or interaction. */
export function drawEtchedFrame(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  { stroke, alpha = 0.18 }: Pick<OrnamentOptions, 'stroke' | 'alpha'>,
) {
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, height);
  ctx.strokeRect(x + 7, y + 7, Math.max(0, width - 14), Math.max(0, height - 14));
  for (const [cornerX, cornerY, directionX, directionY] of [
    [x, y, 1, 1],
    [x + width, y, -1, 1],
    [x, y + height, 1, -1],
    [x + width, y + height, -1, -1],
  ]) {
    ctx.beginPath();
    ctx.moveTo(cornerX, cornerY + directionY * 22);
    ctx.lineTo(cornerX + directionX * 22, cornerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cornerX + directionX * 11, cornerY + directionY * 11, 1.5, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}
