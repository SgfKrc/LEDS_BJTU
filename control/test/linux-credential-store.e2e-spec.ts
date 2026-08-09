import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  defaultCredentialProtector,
  LinuxSecretServiceProtector,
  ModelCredentialStore,
  SecretServiceCommandRunner,
  SecretServiceSyncCommandRunner,
} from '../src/data/model-credential-store';

describe('MF-AUTH-N1 Linux Secret Service credential adapter', () => {
  it('stores only an opaque handle and uses secret-tool argument contracts', async () => {
    const calls: Array<{ args: string[]; stdin: string | null }> = [];
    const values = new Map<string, string>();
    const runCommand: SecretServiceCommandRunner = async (args, stdin) => {
      calls.push({ args, stdin });
      const handle = args[args.length - 1];
      if (args[0] === 'store') {
        values.set(handle, stdin ?? '');
        return '';
      }
      if (args[0] === 'lookup') return `${values.get(handle) ?? ''}\n`;
      if (args[0] === 'clear') {
        values.delete(handle);
        return '';
      }
      throw new Error('unexpected command');
    };
    const runCommandSync: SecretServiceSyncCommandRunner = (args) => {
      values.delete(args[args.length - 1]);
    };
    const protector = new LinuxSecretServiceProtector({ runCommand, runCommandSync });
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qlh-linux-credential-'));
    try {
      const vault = new ModelCredentialStore({ rootDir: path.join(dir, 'credentials'), protector });
      const secret = 'linux-totp-secret-should-not-be-on-disk';
      await vault.set('os:qlh/auth/owner/totp', secret);
      const record = fs.readFileSync(path.join(vault.root, fs.readdirSync(vault.root)[0]), 'utf8');
      expect(record).not.toContain(secret);
      expect(await vault.get('os:qlh/auth/owner/totp')).toBe(secret);
      expect(calls[0].args).toEqual(expect.arrayContaining(['store', 'application', 'qlh', 'credential']));
      expect(calls[0].stdin).toBe(secret);
      expect(vault.delete('os:qlh/auth/owner/totp')).toBe(true);
      expect(values.size).toBe(0);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('fails closed for malformed handles and selects Linux adapter by platform', () => {
    const protector = new LinuxSecretServiceProtector({
      runCommand: async () => '',
      runCommandSync: () => undefined,
    });
    expect(protector.unprotect('not-a-handle')).rejects.toThrow('credential handle is invalid');
    expect(defaultCredentialProtector('linux').name).toBe('linux-secret-service');
    expect(defaultCredentialProtector('win32').name).toBe('windows-dpapi-current-user');
    expect(defaultCredentialProtector('darwin').name).toBe('unavailable');
  });
});
