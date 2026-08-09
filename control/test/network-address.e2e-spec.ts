import {
  buildEndpointUrl,
  canonicalHost,
  formatUrlHost,
  normalizeHttpEndpoint,
} from '../src/common/network-address';

describe('IPv4/IPv6 endpoint helpers', () => {
  it('keeps bare hosts in storage and brackets only URL authorities', () => {
    expect(canonicalHost('[::1]')).toBe('::1');
    expect(formatUrlHost('::1')).toBe('[::1]');
    expect(formatUrlHost('[::1]')).toBe('[::1]');
    expect(formatUrlHost('100.64.0.1')).toBe('100.64.0.1');
  });

  it('builds valid IPv4, IPv6 and MagicDNS URLs', () => {
    expect(buildEndpointUrl('http', '::1', 8000, '/health'))
      .toBe('http://[::1]:8000/health');
    expect(buildEndpointUrl('https', 'node.example.ts.net', 443))
      .toBe('https://node.example.ts.net:443');
  });

  it('normalizes a bare IPv6 profile endpoint', () => {
    const parsed = normalizeHttpEndpoint('fd7a:115c:a1e0::1');
    expect(parsed.href).toBe('http://[fd7a:115c:a1e0::1]/');
  });
});
