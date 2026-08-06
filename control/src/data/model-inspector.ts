/**
 * M2 模型静态 inspector — GGUF 头部与 Safetensors 头部解析（纯 Node，无外部依赖）。
 *
 * 只做静态检查与能力判定（§7.2/§7.3）：
 *  - GGUF：magic/version/KV 元数据（general.architecture、file_type 等）；
 *  - Safetensors 目录：config.json 的 model_type → family 映射 + tokenizer 存在性；
 *  - 能力枚举保持 fail-closed：未知架构全部 false（inspection_only）；
 *  - 不加载权重、不执行任何模型代码（trust_remote_code 默认 false）。
 */
import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export interface InspectionResult {
  ok: boolean;
  format: 'gguf' | 'safetensors' | 'unknown';
  architecture: string | null;
  family: string | null;
  parameter_count: number | null;
  quantization: string | null;
  context_length: number | null;
  has_tokenizer: boolean;
  capabilities: {
    full_worker: boolean;
    pytorch_layer_pipeline: boolean;
    llama_cpp: boolean;
    task_stage: boolean;
  };
  errors: string[];
  warnings: string[];
}

/** 已知 family（与 capabilities 判定的白名单；未知一律 fail-closed）。 */
const KNOWN_FAMILIES = new Set([
  'qwen2', 'qwen2.5', 'llama', 'llama3', 'mistral', 'deepseek',
  'deepseek_v2', 'deepseek_r1', 'gemma', 'phi3',
]);

const GGUF_MAGIC = Buffer.from('GGUF', 'ascii'); // 0x46554747

// GGUF KV value types
enum GgufType {
  UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
  FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
  FLOAT64 = 12,
}

const GGUF_FILE_TYPES: Record<number, string> = {
  0: 'f32', 1: 'f16', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1',
  8: 'q8_0', 10: 'q2_k', 11: 'q3_k', 12: 'q4_k', 13: 'q5_k',
  14: 'q6_k', 15: 'q8_k', 16: 'iq2_xxs', 17: 'iq2_xs', 18: 'iq3_xxs',
};

function familyForArchitecture(arch: string): string | null {
  const a = arch.toLowerCase();
  if (a.includes('qwen2')) return 'qwen2';
  if (a.includes('llama')) return 'llama';
  if (a.includes('mistral')) return 'mistral';
  if (a.includes('deepseek')) return 'deepseek';
  if (a.includes('gemma')) return 'gemma';
  if (a.includes('phi')) return 'phi3';
  return null;
}

@Injectable()
export class ModelInspector {
  /** 解析 GGUF 头（最多读前 1 MiB 元数据区）。 */
  inspectGguf(filePath: string): InspectionResult {
    const result = this.empty('gguf');
    let fd: number | null = null;
    try {
      const stat = fs.statSync(filePath);
      if (stat.size < 8) {
        result.errors.push('文件过小，不是合法 GGUF');
        return result;
      }
      fd = fs.openSync(filePath, 'r');
      const header = Buffer.alloc(8);
      fs.readSync(fd, header, 0, 8, 0);
      if (!header.subarray(0, 4).equals(GGUF_MAGIC)) {
        result.errors.push('GGUF magic 不匹配');
        return result;
      }
      const version = header.readUInt32LE(4);
      if (version < 1) {
        result.errors.push(`不支持的 GGUF version: ${version}`);
        return result;
      }
      // GGUF 头：magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
      const kvCountBuf = Buffer.alloc(8);
      fs.readSync(fd, kvCountBuf, 0, 8, 16);
      const kvCount = Number(kvCountBuf.readBigUInt64LE(0));
      if (kvCount > 100_000) {
        result.errors.push(`KV 数量异常: ${kvCount}`);
        return result;
      }
      let offset = 24;
      const budget = 1024 * 1024; // 只读元数据区
      for (let i = 0; i < kvCount; i++) {
        if (offset > budget) {
          result.errors.push('元数据区超过预算（可能损坏）');
          return result;
        }
        const key = this.readGgufString(fd, offset);
        offset = key.nextOffset;
        const typeBuf = Buffer.alloc(4);
        fs.readSync(fd, typeBuf, 0, 4, offset);
        offset += 4;
        const type = typeBuf.readUInt32LE(0);
        const value = this.readGgufValue(fd, offset, type);
        offset = value.nextOffset;
        this.applyGgufMetadata(result, key.value, value.value);
      }
      result.ok = result.errors.length === 0;
      if (result.architecture) {
        result.family = familyForArchitecture(result.architecture);
        if (result.family && KNOWN_FAMILIES.has(result.family)) {
          result.capabilities.llama_cpp = true;
          result.capabilities.full_worker = false;
        } else {
          result.warnings.push(
            `架构 ${result.architecture} 不在已知列表，仅 inspection`,
          );
        }
      } else {
        result.warnings.push('GGUF 缺少 general.architecture');
      }
      return result;
    } catch (err) {
      result.errors.push(err instanceof Error ? err.message : String(err));
      return result;
    } finally {
      if (fd !== null) fs.closeSync(fd);
    }
  }

  /** 解析 Safetensors 目录：config.json model_type + tokenizer 存在性。 */
  inspectSafetensorsDir(dir: string): InspectionResult {
    const result = this.empty('safetensors');
    const configPath = path.join(dir, 'config.json');
    const hasSafetensors = fs.existsSync(path.join(dir, 'model.safetensors'))
      || fs.readdirSync(dir).some((f) => f.endsWith('.safetensors'));
    if (!hasSafetensors) {
      result.errors.push('目录中没有 .safetensors 权重文件');
      return result;
    }
    if (!fs.existsSync(configPath)) {
      result.errors.push('缺少 config.json（无法判定架构）');
      result.warnings.push('缺少 config.json 的目录只能 inspection');
      return result;
    }
    try {
      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8')) as {
        model_type?: string;
        max_position_embeddings?: number;
        architectures?: string[];
      };
      const modelType = config.model_type ?? '';
      result.architecture = config.architectures?.[0] ?? modelType;
      result.family = familyForArchitecture(modelType);
      result.context_length = Number(config.max_position_embeddings) || null;
      if (result.family && KNOWN_FAMILIES.has(result.family)) {
        result.capabilities.full_worker = true;
      } else {
        result.warnings.push(
          `model_type ${modelType || '(空)'} 不在已知列表，仅 inspection`,
        );
      }
    } catch (err) {
      result.errors.push(`config.json 解析失败: ${err instanceof Error ? err.message : String(err)}`);
      return result;
    }
    result.has_tokenizer = fs.existsSync(path.join(dir, 'tokenizer.json'))
      || fs.existsSync(path.join(dir, 'tokenizer_config.json'))
      || fs.existsSync(path.join(dir, 'tokenizer.model'));
    if (!result.has_tokenizer) {
      result.warnings.push('缺少 tokenizer 文件，试加载可能失败');
    }
    result.ok = result.errors.length === 0;
    return result;
  }

  inspect(pathOrDir: string): InspectionResult {
    const stat = fs.statSync(pathOrDir);
    if (stat.isDirectory()) return this.inspectSafetensorsDir(pathOrDir);
    if (pathOrDir.endsWith('.gguf')) return this.inspectGguf(pathOrDir);
    const result = this.empty('unknown');
    result.errors.push(`未知格式: ${pathOrDir}`);
    return result;
  }

  private empty(format: InspectionResult['format']): InspectionResult {
    return {
      ok: false, format, architecture: null, family: null,
      parameter_count: null, quantization: null, context_length: null,
      has_tokenizer: false,
      capabilities: {
        full_worker: false, pytorch_layer_pipeline: false,
        llama_cpp: false, task_stage: false,
      },
      errors: [], warnings: [],
    };
  }

  private applyGgufMetadata(
    result: InspectionResult,
    key: string,
    value: unknown,
  ): void {
    switch (key) {
      case 'general.architecture':
        result.architecture = String(value);
        break;
      case 'general.file_type':
        result.quantization = GGUF_FILE_TYPES[Number(value)] ?? null;
        break;
      case 'general.context_length':
        result.context_length = Number(value) || null;
        break;
      case 'tokenizer.ggml.model':
        if (String(value).length > 0) result.has_tokenizer = true;
        break;
      default:
        break;
    }
  }

  // ---- GGUF 二进制读取 ----

  private readGgufString(fd: number, offset: number): { value: string; nextOffset: number } {
    const lenBuf = Buffer.alloc(8);
    fs.readSync(fd, lenBuf, 0, 8, offset);
    const len = Number(lenBuf.readBigUInt64LE(0));
    if (len > 10 * 1024 * 1024) {
      throw new Error(`GGUF string 长度异常: ${len}`);
    }
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, offset + 8);
    return { value: buf.toString('utf-8'), nextOffset: offset + 8 + len };
  }

  private readGgufValue(
    fd: number,
    offset: number,
    type: GgufType,
  ): { value: unknown; nextOffset: number } {
    const read = (n: number, at: number): Buffer => {
      const buf = Buffer.alloc(n);
      fs.readSync(fd, buf, 0, n, at);
      return buf;
    };
    switch (type) {
      case GgufType.UINT8: return { value: read(1, offset).readUInt8(0), nextOffset: offset + 1 };
      case GgufType.INT8: return { value: read(1, offset).readInt8(0), nextOffset: offset + 1 };
      case GgufType.UINT16: return { value: read(2, offset).readUInt16LE(0), nextOffset: offset + 2 };
      case GgufType.INT16: return { value: read(2, offset).readInt16LE(0), nextOffset: offset + 2 };
      case GgufType.UINT32: return { value: read(4, offset).readUInt32LE(0), nextOffset: offset + 4 };
      case GgufType.INT32: return { value: read(4, offset).readInt32LE(0), nextOffset: offset + 4 };
      case GgufType.FLOAT32: return { value: read(4, offset).readFloatLE(0), nextOffset: offset + 4 };
      case GgufType.UINT64: return { value: read(8, offset).readBigUInt64LE(0), nextOffset: offset + 8 };
      case GgufType.INT64: return { value: read(8, offset).readBigInt64LE(0), nextOffset: offset + 8 };
      case GgufType.FLOAT64: return { value: read(8, offset).readDoubleLE(0), nextOffset: offset + 8 };
      case GgufType.BOOL: return { value: read(1, offset).readUInt8(0) !== 0, nextOffset: offset + 1 };
      case GgufType.STRING: return this.readGgufString(fd, offset);
      case GgufType.ARRAY: {
        const elemType = read(4, offset).readUInt32LE(0);
        const countBuf = read(8, offset + 4);
        const count = Number(countBuf.readBigUInt64LE(0));
        let cursor = offset + 12;
        const items: unknown[] = [];
        for (let i = 0; i < count && i < 1_000_000; i++) {
          const item = this.readGgufValue(fd, cursor, elemType);
          items.push(item.value);
          cursor = item.nextOffset;
        }
        return { value: items, nextOffset: cursor };
      }
      default:
        throw new Error(`未知 GGUF value type: ${type}`);
    }
  }
}
