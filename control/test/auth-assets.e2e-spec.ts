import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ArtifactStore } from '../src/data/artifact-store';
import { AuthAssetRepository } from '../src/data/auth-asset-repository';
import { SqliteStore } from '../src/data/sqlite-store';
import { StorageMigrationPackage } from '../src/data/storage-migration-package';

const HASH_A = '$scrypt$n=16384$r=8$p=1$c2FsdC1h$c3RvcmVkLWhhc2gtYS0xMjM0NTY3ODkw';
const HASH_B = '$argon2id$v=19$m=65536,t=3,p=1$c2FsdC1i$c3RvcmVkLWhhc2gtYi0xMjM0NTY3ODkw';

describe('M1.2 main-node authentication asset storage', () => {
  let tmpDir: string;
  let stores: SqliteStore[];

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-auth-assets-'));
    stores = [];
  });

  afterEach(() => {
    for (const store of stores) store.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function trackedStore(filePath = path.join(tmpDir, 'control.sqlite3')): SqliteStore {
    const store = new SqliteStore(filePath);
    stores.push(store);
    store.open();
    return store;
  }

  it('creates schema v7 users with normalized uniqueness and no remote projection', () => {
    const store = trackedStore();
    const repository = new AuthAssetRepository(store);
    const owner = repository.createUser({
      username: 'Local.Owner',
      display_name: 'Local Owner',
      role: 'owner',
    });
    const member = repository.createUser({ username: 'member-one' });

    expect(store.schemaVersion).toBe(7);
    expect(repository.getUserByUsername('LOCAL.OWNER')?.user_id).toBe(owner.user_id);
    expect(() => repository.createUser({ username: 'local.owner' })).toThrow();
    expect(() => repository.createUser({ username: 'second-owner', role: 'owner' })).toThrow();
    expect(() => repository.updateUser(owner.user_id, owner.aggregate_version, {
      status: 'suspended',
    })).toThrow('cannot deactivate the only active owner');
    const updated = repository.updateUser(member.user_id, member.aggregate_version, {
      display_name: 'Member One',
    });
    expect(updated.aggregate_version).toBe(member.aggregate_version + 1);
    expect(() => repository.updateUser(member.user_id, member.aggregate_version, {
      display_name: 'stale update',
    })).toThrow('local user version conflict');
    expect(store.prepare('SELECT COUNT(*) AS count FROM outbox').get()).toEqual({ count: 0 });
  });

  it('stores only TOTP references and encoded recovery hashes', () => {
    const store = trackedStore();
    const repository = new AuthAssetRepository(store);
    const owner = repository.createUser({ username: 'owner', role: 'owner' });
    const first = repository.createTotpAuthenticator({
      user_id: owner.user_id,
      secret_ref: 'os:qlh/auth/owner/totp-1',
    });
    repository.activateTotpAuthenticator(first.authenticator_id);
    const second = repository.createTotpAuthenticator({
      user_id: owner.user_id,
      secret_ref: 'os:qlh/auth/owner/totp-2',
      algorithm: 'SHA256',
    });
    repository.activateTotpAuthenticator(second.authenticator_id);

    expect(repository.listAuthenticators(owner.user_id).map((entry) => entry.state)).toEqual([
      'revoked', 'active',
    ]);
    expect(() => repository.createTotpAuthenticator({
      user_id: owner.user_id,
      secret_ref: 'JBSWY3DPEHPK3PXP',
    })).toThrow('credential_ref is invalid');

    const recovery = repository.replaceRecoveryCodeHashes(owner.user_id, [
      { hash_scheme: 'scrypt', code_hash: HASH_A },
      { hash_scheme: 'argon2id', code_hash: HASH_B },
    ]);
    expect(recovery).toHaveLength(2);
    expect(repository.consumeRecoveryCodeHash(owner.user_id, HASH_A)).toBe(true);
    expect(repository.consumeRecoveryCodeHash(owner.user_id, HASH_A)).toBe(false);
    expect(() => repository.replaceRecoveryCodeHashes(owner.user_id, [{
      hash_scheme: 'scrypt',
      code_hash: 'ABCD-EFGH',
    }])).toThrow('encoded scrypt hash');

    store.close();
    const databaseBytes = fs.readFileSync(store.filePath).toString('utf8');
    expect(databaseBytes).not.toContain('JBSWY3DPEHPK3PXP');
    expect(databaseBytes).not.toContain('ABCD-EFGH');
  });

  it('switches active tailnets atomically and retains the old binding on conflict', () => {
    const store = trackedStore();
    const repository = new AuthAssetRepository(store);
    const owner = repository.createUser({ username: 'owner', role: 'owner' });
    const member = repository.createUser({ username: 'member' });

    const oldBinding = repository.prepareTailscaleBinding({
      user_id: owner.user_id,
      authorization_method: 'local_status',
    });
    repository.confirmTailscaleBinding(oldBinding.binding_id, {
      tailnet_id: 'tailnet-old',
      tailscale_user_id: 'ts-user-owner',
      node_id: 'node-owner',
    });
    const occupied = repository.prepareTailscaleBinding({
      user_id: member.user_id,
      authorization_method: 'tailscale_cli',
    });
    repository.confirmTailscaleBinding(occupied.binding_id, {
      tailnet_id: 'tailnet-occupied',
      tailscale_user_id: 'ts-user-occupied',
    });
    const pending = repository.prepareTailscaleBinding({
      user_id: owner.user_id,
      authorization_method: 'oauth_app',
      credential_ref: 'os:qlh/auth/owner/tailscale-oauth',
    });
    expect(() => repository.prepareTailscaleBinding({
      user_id: owner.user_id,
      authorization_method: 'local_status',
    })).toThrow();

    expect(() => repository.confirmTailscaleBinding(pending.binding_id, {
      tailnet_id: 'tailnet-occupied',
      tailscale_user_id: 'ts-user-occupied',
    })).toThrow('already bound to another local user');
    expect(repository.getTailscaleBinding(oldBinding.binding_id)?.state).toBe('active');
    expect(repository.getTailscaleBinding(pending.binding_id)?.state).toBe('pending');

    repository.confirmTailscaleBinding(pending.binding_id, {
      tailnet_id: 'tailnet-new',
      tailscale_user_id: 'ts-user-owner-new',
      node_id: 'node-owner-new',
    });
    const bindings = repository.listTailscaleBindings(owner.user_id);
    expect(bindings.find((entry) => entry.binding_id === oldBinding.binding_id)?.state).toBe('revoked');
    expect(bindings.find((entry) => entry.binding_id === pending.binding_id)).toMatchObject({
      state: 'active',
      tailnet_id: 'tailnet-new',
      tailscale_user_id: 'ts-user-owner-new',
    });
  });

  it('rejects secret-bearing audit details while retaining local audit evidence', () => {
    const store = trackedStore();
    const repository = new AuthAssetRepository(store);
    const owner = repository.createUser({ username: 'owner', role: 'owner' });

    expect(() => repository.appendAudit({
      user_id: owner.user_id,
      event_type: 'login_failed',
      outcome: 'failure',
      details: { token: 'tskey-secret-value' },
    })).toThrow('cannot contain token');
    expect(() => repository.appendAudit({
      user_id: owner.user_id,
      event_type: 'login_failed',
      outcome: 'failure',
      details: { note: 'Authorization: Bearer secret' },
    })).toThrow('contain secret material');
    repository.appendAudit({
      user_id: owner.user_id,
      event_type: 'login_failed',
      outcome: 'denied',
      reason_code: 'totp_mismatch',
      details: { attempt: 2, source: 'local_console' },
    });

    expect(repository.listAuditEvents(owner.user_id).map((entry) => entry.event_type)).toEqual([
      'user_created', 'login_failed',
    ]);
  });

  it('survives restart and migration package restore without claiming OS credential transfer', async () => {
    const sourceRoot = path.join(tmpDir, 'source');
    const sourceStore = trackedStore(path.join(sourceRoot, 'control.sqlite3'));
    const sourceRepository = new AuthAssetRepository(sourceStore);
    const owner = sourceRepository.createUser({ username: 'owner', role: 'owner' });
    const authenticator = sourceRepository.createTotpAuthenticator({
      user_id: owner.user_id,
      secret_ref: 'os:qlh/auth/owner/totp-primary',
    });
    sourceRepository.activateTotpAuthenticator(authenticator.authenticator_id);
    const sourceCredentials = path.join(sourceRoot, 'credentials');
    fs.mkdirSync(sourceCredentials, { recursive: true });
    fs.writeFileSync(path.join(sourceCredentials, 'credential.json'), 'source-only-secret');

    sourceStore.close();
    const reopenedStore = trackedStore(sourceStore.filePath);
    const reopenedRepository = new AuthAssetRepository(reopenedStore);
    expect(reopenedRepository.getUser(owner.user_id)?.username).toBe('owner');
    expect(reopenedRepository.getAuthenticator(authenticator.authenticator_id)?.state).toBe('active');

    const sourceArtifacts = new ArtifactStore(path.join(sourceRoot, 'model-store'));
    const packagePath = path.join(tmpDir, 'auth-assets.qlhmigrate');
    const passphrase = 'auth assets migration passphrase';
    const exported = await new StorageMigrationPackage(reopenedStore, sourceArtifacts)
      .exportPackage(packagePath, passphrase);
    expect(exported).toMatchObject({ schema_version: 7, manifest_count: 0, blob_count: 0 });

    const targetRoot = path.join(tmpDir, 'target');
    const targetStore = trackedStore(path.join(targetRoot, 'control.sqlite3'));
    const targetArtifacts = new ArtifactStore(path.join(targetRoot, 'model-store'));
    new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase);
    const targetRepository = new AuthAssetRepository(targetStore);
    expect(targetRepository.getUser(owner.user_id)?.username).toBe('owner');
    expect(targetRepository.getAuthenticator(authenticator.authenticator_id)).toMatchObject({
      state: 'active',
      secret_ref: 'os:qlh/auth/owner/totp-primary',
    });
    expect(fs.existsSync(path.join(targetRoot, 'credentials'))).toBe(false);
    expect(fs.readFileSync(packagePath).includes(Buffer.from('source-only-secret'))).toBe(false);
  });
});
