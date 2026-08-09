/**
 * bootstrap 首连引导 — CIDR 信任校验与归一化
 * (对齐 src/bootstrap.py:21-72)
 *
 * 信任源（QLH_TRUSTED_BOOTSTRAP_CIDRS 覆盖，默认对齐 DEFAULT_TRUSTED_CIDRS）：
 *   100.64.0.0/10, 127.0.0.0/8, ::1/128, fd7a:115c:a1e0::/48
 * 仅 IP 字面量参与匹配；hostname 一律不信任（对齐 ipaddress 语义）。
 */

export const DEFAULT_TRUSTED_CIDRS = [
  '100.64.0.0/10',
  '127.0.0.0/8',
  '::1/128',
  'fd7a:115c:a1e0::/48',
];

export function resolveTrustedCidrs(env: NodeJS.ProcessEnv = process.env): string[] {
  const raw = env.QLH_TRUSTED_BOOTSTRAP_CIDRS?.trim();
  if (!raw) return DEFAULT_TRUSTED_CIDRS;
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

/** 对齐 is_trusted_bootstrap_source：host 为 IP 且命中任一信任网段 */
export function isTrustedBootstrapSource(
  host: string,
  cidrs: string[] = DEFAULT_TRUSTED_CIDRS,
): boolean {
  const normalizedHost = normalizeMappedIpv4(host);
  const ip = ipToInt(normalizedHost);
  if (ip === null) return false;
  for (const cidr of cidrs) {
    if (isInCidr(normalizedHost, cidr)) return true;
  }
  return false;
}

/** IPv4/IPv6 CIDR 匹配（手写实现，无外部依赖） */
export function isInCidr(ip: string, cidr: string): boolean {
  const parts = cidr.split('/');
  if (parts.length !== 2) return false;
  const prefix = Number(parts[1]);
  if (!Number.isInteger(prefix) || prefix < 0 || prefix > 128) return false;
  const ipInt = ipToInt(ip);
  const netInt = ipToInt(parts[0]);
  if (ipInt === null || netInt === null) return false;
  const ipV6 = ip.includes(':');
  const netV6 = parts[0].includes(':');
  if (ipV6 !== netV6) return false; // 版本不一致不匹配
  const bits = ipV6 ? 128n : 32n;
  const shift = bits - BigInt(prefix);
  const mask = shift >= 128n ? 0n : ((BigInt(1) << shift) - BigInt(1)) ^ ((BigInt(1) << bits) - BigInt(1));
  return (ipInt & mask) === (netInt & mask);
}

/** 解析 IP 为整数（IPv4 → 32 位，IPv6 → 128 位）；非法返回 null */
export function ipToInt(ip: string): bigint | null {
  const trimmed = (ip || '').trim();
  if (!trimmed) return null;
  if (trimmed.includes(':')) {
    return ipv6ToInt(trimmed);
  }
  if (trimmed.includes('.')) {
    const octets = trimmed.split('.');
    if (octets.length !== 4) return null;
    let value = 0n;
    for (const oct of octets) {
      const n = Number(oct);
      if (!Number.isInteger(n) || n < 0 || n > 255) return null;
      value = (value << 8n) | BigInt(n);
    }
    return value;
  }
  return null;
}

function normalizeMappedIpv4(ip: string): string {
  const trimmed = (ip || '').trim();
  const match = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i.exec(trimmed);
  return match?.[1] ?? trimmed;
}

function ipv6ToInt(ip: string): bigint | null {
  // 处理 :: 压缩与内嵌 IPv4
  let head = ip;
  let tail = '';
  const doubleColon = ip.indexOf('::');
  if (doubleColon >= 0) {
    head = ip.slice(0, doubleColon);
    tail = ip.slice(doubleColon + 2);
  }
  const headParts = head ? head.split(':') : [];
  const tailParts = tail ? tail.split(':') : [];
  // 内嵌 IPv4 段（如 ::ffff:192.168.1.1 的尾段含 .）
  const expand = (parts: string[]): bigint[] | null => {
    const out: bigint[] = [];
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (p.includes('.')) {
        const v4 = ipToInt(p);
        if (v4 === null) return null;
        out.push((v4 >> 16n) & 0xffffn, v4 & 0xffffn);
      } else {
        if (!/^[0-9a-fA-F]{1,4}$/.test(p)) return null;
        out.push(BigInt(parseInt(p, 16)));
      }
    }
    return out;
  };
  const headInts = expand(headParts);
  const tailInts = expand(tailParts);
  if (headInts === null || tailInts === null) return null;
  const total = headInts.length + tailInts.length;
  if (doubleColon >= 0) {
    if (total > 7) return null; // 压缩后最多 7 组显式 + 1 组压缩
  } else if (total !== 8) {
    return null;
  }
  const groups: bigint[] = [];
  groups.push(...headInts);
  while (groups.length < 8 - tailInts.length) groups.push(0n);
  groups.push(...tailInts);
  let value = 0n;
  for (const g of groups) value = (value << 16n) | (g & 0xffffn);
  return value;
}

/** 对齐 normalize_node_type（bootstrap.py:51-55） */
export function normalizeNodeType(nodeType: string | null | undefined): string {
  const value = (nodeType || 'pc').trim().toLowerCase();
  if (value === 'android' || value === 'mobile') return 'android';
  return 'pc';
}

/** 对齐 normalize_node_id（bootstrap.py:58-72）：非法字符归一 + 64 截断 */
export function normalizeNodeId(nodeId: string | null | undefined, nodeType = 'pc'): string {
  let raw = (nodeId || '').trim();
  if (!raw || raw === 'master') {
    const prefix = nodeType === 'android' ? 'android' : 'client';
    raw = `${prefix}_${osHostname()}`;
  }
  let normalized = '';
  for (const ch of raw) {
    if (/[A-Za-z0-9]/.test(ch) || ch === '_' || ch === '-' || ch === '.') {
      normalized += ch;
    } else {
      normalized += '_';
    }
  }
  normalized = normalized.replace(/^[._-]+|[._-]+$/g, '');
  if (!normalized || normalized === 'master') {
    normalized = `client_${osHostname()}`;
  }
  return normalized.slice(0, 64);
}

function osHostname(): string {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const os = require('os') as typeof import('os');
  return os.hostname() || 'unknown';
}
