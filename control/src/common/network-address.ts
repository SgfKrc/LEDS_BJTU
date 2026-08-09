/** Canonical host and URL helpers shared by control-svc endpoints. */

export function canonicalHost(host: string | null | undefined): string {
  const value = String(host ?? '').trim();
  if (value.startsWith('[') && value.endsWith(']')) {
    return value.slice(1, -1);
  }
  return value;
}

export function formatUrlHost(host: string | null | undefined): string {
  const value = canonicalHost(host);
  if (!value.includes(':')) return value;
  return `[${value.replace(/%/g, '%25')}]`;
}

export function buildEndpointUrl(
  scheme: string,
  host: string,
  port: number,
  path = '',
): string {
  const normalizedScheme = scheme.trim().toLowerCase();
  if (!['http', 'https', 'ws', 'wss'].includes(normalizedScheme)) {
    throw new Error(`unsupported URL scheme: ${scheme}`);
  }
  const normalizedPort = Number(port);
  if (!Number.isInteger(normalizedPort) || normalizedPort < 1 || normalizedPort > 65535) {
    throw new Error(`invalid port: ${port}`);
  }
  const authorityHost = formatUrlHost(host);
  if (!authorityHost) throw new Error('host is required');
  const normalizedPath = path && !path.startsWith('/') ? `/${path}` : path;
  return `${normalizedScheme}://${authorityHost}:${normalizedPort}${normalizedPath}`;
}

export function normalizeHttpEndpoint(endpoint: string): URL {
  const value = endpoint.trim();
  if (!value) throw new Error('endpoint is required');
  if (/^https?:\/\//i.test(value)) return new URL(value);
  if (value.startsWith('[')) return new URL(`http://${value}`);
  if ((value.match(/:/g) ?? []).length > 1) {
    return new URL(`http://${formatUrlHost(value)}`);
  }
  return new URL(`http://${value}`);
}
