import { spawn } from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { ArtifactStore } from './artifact-store';
import { RuntimeLoadStatus } from './artifact-runtime-repository';

interface ManifestFile {
  path: string;
  size: number;
  sha256: string;
}

export interface RuntimeSidecarResult {
  schema_version: 1;
  request_id: string;
  artifact_id: string;
  status: RuntimeLoadStatus;
  engine: string;
  runtime_profile: string;
  checked_at: string;
  loader_version: string | null;
  load_ms: number | null;
  details: Record<string, unknown>;
  error: { code: string; message: string } | null;
}

export interface RuntimeSidecarOptions {
  executable?: string;
  scriptPath?: string;
  timeoutMs?: number;
  maxOutputBytes?: number;
  env?: NodeJS.ProcessEnv;
}

export interface RuntimeTrialOptions {
  signal?: AbortSignal;
}

interface RuntimeRequest {
  schema_version: 1;
  operation: 'trial_load';
  request_id: string;
  artifact_id: string;
  format: 'gguf' | 'safetensors';
  engine: 'llama_cpp' | 'pytorch_transformers';
  runtime_profile: string;
  model_path: string;
  files: ManifestFile[];
  trust_remote_code: false;
  options: {
    n_ctx: number;
    n_threads: number;
    n_gpu_layers: number;
  };
}

export class ModelRuntimeSidecar {
  private readonly timeoutMs: number;
  private readonly maxOutputBytes: number;

  constructor(
    private readonly store: ArtifactStore,
    private readonly options: RuntimeSidecarOptions = {},
  ) {
    this.timeoutMs = options.timeoutMs ?? 240_000;
    this.maxOutputBytes = options.maxOutputBytes ?? 1024 * 1024;
  }

  async trialLoad(
    manifest: Record<string, unknown>,
    options: RuntimeTrialOptions = {},
  ): Promise<RuntimeSidecarResult> {
    const checkedAt = new Date().toISOString();
    const artifactId = String(manifest.artifact_id ?? '');
    const engine = String(manifest.engine ?? '');
    const requirements = manifest.requirements as Record<string, unknown> | undefined;
    const runtimeProfile = String(requirements?.runtime_profile ?? '');
    const requestId = crypto.randomUUID();

    if (!/^sha256:[0-9a-f]{64}$/.test(artifactId)) {
      throw new Error('manifest artifact_id is invalid');
    }

    let viewDir: string | null = null;
    try {
      const prepared = this.prepareRequest(manifest, requestId);
      viewDir = prepared.viewDir;
      return await this.run(prepared.request, options.signal);
    } catch (error) {
      return this.failure(
        requestId,
        artifactId,
        engine,
        runtimeProfile,
        checkedAt,
        'sidecar_prepare_failed',
        error instanceof Error ? error.message : String(error),
      );
    } finally {
      if (viewDir) this.removeRuntimeView(viewDir);
    }
  }

  private prepareRequest(
    manifest: Record<string, unknown>,
    requestId: string,
  ): { request: RuntimeRequest; viewDir: string } {
    const artifactId = String(manifest.artifact_id);
    const format = String(manifest.format);
    const engine = String(manifest.engine);
    const requirements = manifest.requirements as Record<string, unknown> | undefined;
    const runtimeProfile = String(requirements?.runtime_profile ?? '');
    const capabilities = manifest.capabilities as Record<string, unknown> | undefined;
    const trust = manifest.trust_policy as Record<string, unknown> | undefined;

    if (!runtimeProfile) throw new Error('manifest runtime_profile is required');
    if (trust?.trust_remote_code === true) {
      throw new Error('trust_remote_code must remain false');
    }
    if (format === 'gguf') {
      if (engine !== 'llama_cpp' || capabilities?.llama_cpp !== true) {
        throw new Error('GGUF artifact is not admitted for llama_cpp');
      }
    } else if (format === 'safetensors') {
      if (engine !== 'pytorch_transformers' || capabilities?.full_worker !== true) {
        throw new Error('Safetensors artifact is not admitted for full_worker');
      }
    } else {
      throw new Error(`unsupported runtime format: ${format}`);
    }

    const rawFiles = manifest.files;
    if (!Array.isArray(rawFiles) || rawFiles.length === 0) {
      throw new Error('manifest files are required');
    }
    const files = rawFiles.map((item) => this.validateFile(item));
    const runtimeRoot = path.resolve(this.store.root, 'runtime');
    fs.mkdirSync(runtimeRoot, { recursive: true });
    const viewDir = fs.mkdtempSync(path.join(runtimeRoot, 'check-'));

    try {
      for (const file of files) {
        const source = this.store.blobPath(file.sha256);
        if (!fs.existsSync(source)) throw new Error(`artifact blob is missing: ${file.sha256}`);
        const stat = fs.statSync(source);
        if (!stat.isFile() || stat.size !== file.size) {
          throw new Error(`artifact blob size mismatch: ${file.sha256}`);
        }
        const target = path.resolve(viewDir, file.path);
        if (!target.startsWith(path.resolve(viewDir) + path.sep)) {
          throw new Error(`manifest file path escapes runtime view: ${file.path}`);
        }
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.linkSync(source, target);
      }

      const ggufFiles = files.filter((file) => file.path.toLowerCase().endsWith('.gguf'));
      if (format === 'gguf' && ggufFiles.length !== 1) {
        throw new Error('GGUF runtime view must contain exactly one .gguf file');
      }
      const modelPath = format === 'gguf'
        ? path.resolve(viewDir, ggufFiles[0].path)
        : viewDir;
      const detectedCpuCount = Number(process.env.NUMBER_OF_PROCESSORS || 4);
      const cpuCount = Number.isFinite(detectedCpuCount) && detectedCpuCount > 0
        ? detectedCpuCount : 4;

      return {
        viewDir,
        request: {
          schema_version: 1,
          operation: 'trial_load',
          request_id: requestId,
          artifact_id: artifactId,
          format: format as RuntimeRequest['format'],
          engine: engine as RuntimeRequest['engine'],
          runtime_profile: runtimeProfile,
          model_path: modelPath,
          files,
          trust_remote_code: false,
          options: {
            n_ctx: 128,
            n_threads: Math.max(1, Math.min(8, cpuCount)),
            n_gpu_layers: 0,
          },
        },
      };
    } catch (error) {
      this.removeRuntimeView(viewDir);
      throw error;
    }
  }

  private validateFile(input: unknown): ManifestFile {
    const file = input as Partial<ManifestFile>;
    const relPath = String(file?.path ?? '');
    const normalized = path.normalize(relPath);
    if (!relPath || path.isAbsolute(relPath) || normalized === '..'
        || normalized.startsWith(`..${path.sep}`)) {
      throw new Error(`manifest file path is invalid: ${relPath}`);
    }
    if (!Number.isSafeInteger(file.size) || Number(file.size) < 0) {
      throw new Error(`manifest file size is invalid: ${relPath}`);
    }
    const digest = String(file.sha256 ?? '').toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(digest)) {
      throw new Error(`manifest file digest is invalid: ${relPath}`);
    }
    return { path: normalized, size: Number(file.size), sha256: digest };
  }

  private run(request: RuntimeRequest, signal?: AbortSignal): Promise<RuntimeSidecarResult> {
    const executable = this.options.executable
      ?? process.env.QLH_RUNTIME_PYTHON?.trim()
      ?? (process.platform === 'win32' ? 'python' : 'python3');
    const scriptPath = this.options.scriptPath ?? this.defaultScriptPath();
    const childEnv: NodeJS.ProcessEnv = {
      ...process.env,
      ...this.options.env,
      PYTHONUNBUFFERED: '1',
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
      HF_DATASETS_OFFLINE: '1',
      NO_PROXY: '*',
    };
    for (const key of ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']) {
      delete childEnv[key];
    }

    return new Promise((resolve) => {
      if (signal?.aborted) {
        resolve(this.failure(
          request.request_id, request.artifact_id, request.engine,
          request.runtime_profile, new Date().toISOString(),
          'sidecar_cancelled', 'runtime sidecar was cancelled before start',
        ));
        return;
      }
      const child = spawn(executable, [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        env: childEnv,
      });
      const stdout: Buffer[] = [];
      const stderr: Buffer[] = [];
      let stdoutBytes = 0;
      let stderrBytes = 0;
      let timedOut = false;
      let cancelled = false;
      let outputOverflow = false;
      let settled = false;
      let timer: NodeJS.Timeout | null = null;

      const finish = (result: RuntimeSidecarResult): void => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        signal?.removeEventListener('abort', abortListener);
        resolve(result);
      };
      const abortListener = (): void => {
        cancelled = true;
        child.kill('SIGKILL');
      };
      signal?.addEventListener('abort', abortListener, { once: true });
      const killForOverflow = (): void => {
        outputOverflow = true;
        child.kill('SIGKILL');
      };
      child.stdout.on('data', (chunk: Buffer) => {
        stdoutBytes += chunk.length;
        if (stdoutBytes <= this.maxOutputBytes) stdout.push(chunk);
        else killForOverflow();
      });
      child.stderr.on('data', (chunk: Buffer) => {
        stderrBytes += chunk.length;
        if (stderrBytes <= this.maxOutputBytes) stderr.push(chunk);
        else killForOverflow();
      });
      child.on('error', (error) => {
        finish(this.failure(
          request.request_id, request.artifact_id, request.engine,
          request.runtime_profile, new Date().toISOString(),
          'sidecar_spawn_failed', error.message,
        ));
      });
      child.on('close', (code) => {
        if (cancelled) {
          finish(this.failure(
            request.request_id, request.artifact_id, request.engine,
            request.runtime_profile, new Date().toISOString(),
            'sidecar_cancelled', 'runtime sidecar was cancelled',
          ));
          return;
        }
        if (timedOut) {
          finish(this.failure(
            request.request_id, request.artifact_id, request.engine,
            request.runtime_profile, new Date().toISOString(),
            'sidecar_timeout', `runtime sidecar exceeded ${this.timeoutMs} ms`,
          ));
          return;
        }
        if (outputOverflow) {
          finish(this.failure(
            request.request_id, request.artifact_id, request.engine,
            request.runtime_profile, new Date().toISOString(),
            'sidecar_output_overflow', 'runtime sidecar output exceeded the protocol limit',
          ));
          return;
        }
        const diagnostics = Buffer.concat(stderr).toString('utf-8').trim();
        if (code !== 0) {
          finish(this.failure(
            request.request_id, request.artifact_id, request.engine,
            request.runtime_profile, new Date().toISOString(),
            'sidecar_crashed', diagnostics || `runtime sidecar exited with code ${code}`,
          ));
          return;
        }
        try {
          const lines = Buffer.concat(stdout).toString('utf-8')
            .split(/\r?\n/).filter((line) => line.trim().length > 0);
          if (lines.length !== 1) throw new Error('sidecar must emit exactly one JSON line');
          const result = JSON.parse(lines[0]) as RuntimeSidecarResult;
          this.validateResult(request, result);
          if (diagnostics) result.details.stderr_tail = diagnostics.slice(-4096);
          finish(result);
        } catch (error) {
          finish(this.failure(
            request.request_id, request.artifact_id, request.engine,
            request.runtime_profile, new Date().toISOString(),
            'sidecar_protocol_error', error instanceof Error ? error.message : String(error),
          ));
        }
      });
      child.stdin.on('error', () => undefined);
      child.stdin.end(JSON.stringify(request), 'utf-8');
      timer = setTimeout(() => {
        timedOut = true;
        child.kill('SIGKILL');
      }, this.timeoutMs);
    });
  }

  private validateResult(request: RuntimeRequest, result: RuntimeSidecarResult): void {
    if (result?.schema_version !== 1
        || result.request_id !== request.request_id
        || result.artifact_id !== request.artifact_id
        || result.engine !== request.engine
        || result.runtime_profile !== request.runtime_profile
        || !['ready', 'load_failed', 'resource_rejected'].includes(result.status)) {
      throw new Error('sidecar response does not match the request');
    }
    if (!result.checked_at || !result.details || !Object.prototype.hasOwnProperty.call(result, 'error')) {
      throw new Error('sidecar response is incomplete');
    }
  }

  private failure(
    requestId: string,
    artifactId: string,
    engine: string,
    runtimeProfile: string,
    checkedAt: string,
    code: string,
    message: string,
  ): RuntimeSidecarResult {
    return {
      schema_version: 1,
      request_id: requestId,
      artifact_id: artifactId,
      status: 'load_failed',
      engine,
      runtime_profile: runtimeProfile,
      checked_at: checkedAt,
      loader_version: null,
      load_ms: null,
      details: {},
      error: { code, message: message.slice(0, 4096) },
    };
  }

  private defaultScriptPath(): string {
    const candidates = [
      path.resolve(process.cwd(), 'src', 'inference_service', 'model_runtime_sidecar.py'),
      path.resolve(process.cwd(), '..', 'src', 'inference_service', 'model_runtime_sidecar.py'),
    ];
    const found = candidates.find((candidate) => fs.existsSync(candidate));
    if (!found) throw new Error('model runtime sidecar script was not found');
    return found;
  }

  private removeRuntimeView(viewDir: string): void {
    const runtimeRoot = path.resolve(this.store.root, 'runtime');
    const target = path.resolve(viewDir);
    if (!target.startsWith(runtimeRoot + path.sep)) {
      throw new Error('refusing to remove a path outside the runtime view root');
    }
    fs.rmSync(target, { recursive: true, force: true });
  }
}
