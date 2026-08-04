/**
 * 模型注册表存储 — 对齐 api_server /api/models/registry + db.py
 * save_experimental_model/get_experimental_models/delete_experimental_model
 * (微服务架构改造计划 阶段 3.2 模型注册表域)
 *
 * 存储：JSON 文件（PostgreSQL cluster_config `experimental_model:` 前缀
 * 分支未迁移——沿用降级语义，清理阶段切换时再迁数据）。
 * 并发：Node 单线程 + 原子写（tmp + rename）。
 */
import { Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export type RegisteredModelType = 'safetensors' | 'gguf' | 'both';

export interface RegisteredModel {
  model_id: string;
  name: string;
  model_type: RegisteredModelType;
  model_path: string;
  gguf_path: string;
  recommended_vram_gb: number;
  max_context: number;
  huggingface_id: string;
  description: string;
  quant_types: string[];
}

/** 内置模型 ID 集（对齐 model_config.BUILTIN_MODELS 的 8 个条目）——不可删除 */
export const BUILTIN_MODEL_IDS = new Set([
  'qwen-1_8b',
  'qwen2.5-7b',
  'qwen2.5-14b',
  'qwen2.5-7b-gguf',
  'deepseek-r1-distill-qwen-1.5b',
  'deepseek-r1-distill-qwen-7b',
  'deepseek-r1-distill-qwen-14b',
  'deepseek-r1-distill-qwen-32b',
]);

/** 对齐 api_server.py:5201：非 gguf 模型给 torch 三档，gguf 给 Q4_K_M */
export function quantTypesFor(modelType: RegisteredModelType): string[] {
  return modelType !== 'gguf' ? ['fp16', 'int8', 'int4'] : ['Q4_K_M'];
}

export function resolveRegistryFile(env: NodeJS.ProcessEnv = process.env): string {
  return (
    env.QLH_MODEL_REGISTRY_FILE?.trim() ||
    path.join(process.cwd(), 'model_registry.json')
  );
}

@Injectable()
export class ModelRegistryStore {
  private readonly file: string;

  constructor(@Optional() file?: string) {
    this.file = file ?? resolveRegistryFile();
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
  }

  list(): RegisteredModel[] {
    try {
      const raw = fs.readFileSync(this.file, 'utf-8');
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((d) => d && typeof d === 'object' && d.model_id)
        .map((d) => this.normalize(d));
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code !== 'ENOENT') {
        console.warn(`[control-svc] 模型注册表损坏，重建: ${this.file}`);
      }
      return [];
    }
  }

  get(modelId: string): RegisteredModel | null {
    return this.list().find((m) => m.model_id === modelId) ?? null;
  }

  /** 对齐 save_experimental_model 的 upsert 语义（重复 model_id 覆盖） */
  upsert(model: RegisteredModel): void {
    const models = this.list();
    const idx = models.findIndex((m) => m.model_id === model.model_id);
    if (idx >= 0) models[idx] = model;
    else models.push(model);
    this.saveAll(models);
  }

  delete(modelId: string): boolean {
    const models = this.list();
    const kept = models.filter((m) => m.model_id !== modelId);
    if (kept.length === models.length) return false;
    this.saveAll(kept);
    return true;
  }

  private saveAll(models: RegisteredModel[]): void {
    const tmp = `${this.file}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify(models, null, 2), 'utf-8');
      fs.renameSync(tmp, this.file);
    } catch (err) {
      console.warn(`[control-svc] 写入模型注册表失败: ${this.file}: ${String(err)}`);
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
    }
  }

  private normalize(d: Record<string, unknown>): RegisteredModel {
    const valid: RegisteredModelType[] = ['safetensors', 'gguf', 'both'];
    const modelType = valid.includes(d.model_type as RegisteredModelType)
      ? (d.model_type as RegisteredModelType)
      : 'safetensors';
    return {
      model_id: String(d.model_id),
      name: String(d.name ?? d.model_id),
      model_type: modelType,
      model_path: String(d.model_path ?? ''),
      gguf_path: String(d.gguf_path ?? ''),
      recommended_vram_gb: Number(d.recommended_vram_gb) || 8.0,
      max_context: Number(d.max_context) || 4096,
      huggingface_id: String(d.huggingface_id ?? ''),
      description: String(d.description ?? ''),
      quant_types: Array.isArray(d.quant_types)
        ? d.quant_types.map(String)
        : quantTypesFor(modelType),
    };
  }
}
