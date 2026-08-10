import { parseTailscaleStatusJson } from '../src/data/tailscale-local-status';

describe('MF-AUTH-N2L local Tailscale status projection', () => {
  it('projects only the current local identity candidate', () => {
    const result = parseTailscaleStatusJson(JSON.stringify({
      Version: '1.2.3',
      BackendState: 'Running',
      MagicDNSSuffix: 'example.ts.net',
      CurrentTailnet: { Name: 'Example tailnet', MagicDNSSuffix: 'example.ts.net' },
      Self: {
        ID: 'node-stable-id',
        UserID: 123456789,
        HostName: 'local-node',
        DNSName: 'local-node.example.ts.net.',
        PublicKey: 'nodekey:must-not-leak',
        TailscaleIPs: ['100.64.1.2', 'fd7a:115c:a1e0::1', '192.168.1.4'],
      },
      User: { 123456789: { LoginName: 'private@example.com' } },
      Peer: { secret: { PublicKey: 'peerkey:must-not-leak' } },
      Health: ['private diagnostic'],
    }));

    expect(result).toMatchObject({
      available: true,
      state: 'ready',
      requires_confirmation: true,
      candidate: {
        tailnet_id: 'example.ts.net',
        tailnet_id_source: 'magic_dns_suffix',
        tailnet_display_name: 'Example tailnet',
        tailscale_user_id: '123456789',
        node_id: 'node-stable-id',
        hostname: 'local-node',
        addresses: ['100.64.1.2', 'fd7a:115c:a1e0::1'],
      },
    });
    const serialized = JSON.stringify(result);
    expect(serialized).not.toContain('private@example.com');
    expect(serialized).not.toContain('must-not-leak');
    expect(serialized).not.toContain('private diagnostic');
  });

  it('fails closed when Tailscale is not running or not logged in', () => {
    expect(parseTailscaleStatusJson(JSON.stringify({ BackendState: 'Stopped' })))
      .toMatchObject({ available: false, state: 'not_running', candidate: null });
    expect(parseTailscaleStatusJson(JSON.stringify({ BackendState: 'Running' })))
      .toMatchObject({ available: false, state: 'not_logged_in', candidate: null });
  });

  it('rejects malformed and incomplete identity responses', () => {
    expect(parseTailscaleStatusJson('{invalid'))
      .toMatchObject({ available: false, state: 'invalid_response', candidate: null });
    expect(parseTailscaleStatusJson(JSON.stringify({
      BackendState: 'Running',
      CurrentTailnet: { Name: 'Example' },
      Self: { ID: 'node-id' },
    }))).toMatchObject({ available: false, state: 'incomplete_identity', candidate: null });
  });
});
