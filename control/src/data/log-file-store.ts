/**
 * 日志文件操作 — 对齐 api_server.py 日志域的文件语义
 * (微服务架构改造计划 阶段 3.2 日志域)
 *
 * 语义对齐：
 *  - LOG_DIR：QLH_LOG_DIR 环境变量优先，否则 <cwd>/logs
 *    （config.py:423 LOG_DIR = _APP_ROOT/logs；control-svc 从项目根启动）
 *  - 文件名校验 _is_log_filename：basename、无 ..、^[^/\\]+\.log(?:\.\d+)?$
 *    （正则显式拒绝 Windows 反斜杠，与 Python 一致）
 *  - 列表按 mtime 降序，modified 为本地时间 isoformat（对齐
 *    datetime.fromtimestamp(st.st_mtime).isoformat()）
 *  - 读取返回末 1MB，跳过不完整首行，utf-8 errors=replace
 *  - 删除/清空/导出不依赖 Python 的 logging handlers（TS 无文件句柄占用，
 *    无需 _close_logging_handlers/setup_logging 循环）
 */
import { Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import JSZip from 'jszip';

export interface LogFileMeta {
  name: string;
  size: number;
  modified: string;
}

const LOG_FILE_RE = /^[^/\\]+\.log(?:\.\d+)?$/;

export function resolveLogDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.QLH_LOG_DIR?.trim() || path.join(process.cwd(), 'logs');
}

export function nodeIdOf(env: NodeJS.ProcessEnv = process.env): string {
  return env.QLH_NODE_ID?.trim() || 'master';
}

export function deviceIpOf(env: NodeJS.ProcessEnv = process.env): string {
  return env.QLH_DEVICE_IP?.trim() || '';
}

export function isLogFilename(filename: string): boolean {
  return (
    filename === path.basename(filename) &&
    !filename.includes('..') &&
    LOG_FILE_RE.test(filename)
  );
}

/** 本地时间 isoformat（对齐 Python datetime.isoformat()，无时区后缀） */
export function isoLocal(d: Date = new Date()): string {
  const p = (n: number): string => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` +
    `T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.` +
    `${String(d.getMilliseconds()).padStart(3, '0')}`
  );
}

@Injectable()
export class LogFileStore {
  readonly logDir: string;

  constructor(@Optional() logDir?: string) {
    this.logDir = logDir ?? resolveLogDir();
  }

  private filePath(name: string): string {
    return path.join(this.logDir, name);
  }

  private isDir(): boolean {
    try {
      return fs.statSync(this.logDir).isDirectory();
    } catch {
      return false;
    }
  }

  /** 列出所有 .log 文件，按 mtime 降序（对齐 list_log_files） */
  listFiles(): { files: LogFileMeta[] } {
    if (!this.isDir()) return { files: [] };
    const files: (LogFileMeta & { _mtime: number })[] = [];
    for (const fname of fs.readdirSync(this.logDir)) {
      if (!isLogFilename(fname)) continue;
      try {
        const st = fs.statSync(this.filePath(fname));
        files.push({
          name: fname,
          size: st.size,
          modified: isoLocal(new Date(st.mtimeMs)),
          _mtime: st.mtimeMs,
        });
      } catch {
        /* 文件可能被并发轮转删除，跳过 */
      }
    }
    files.sort((a, b) => b._mtime - a._mtime);
    return { files: files.map(({ _mtime, ...rest }) => rest) };
  }

  /** 文件统计（对齐 get_log_stats 的文件部分：size 为文件字节数） */
  fileStats(): { files: { name: string; size: number }[]; totalBytes: number } {
    if (!this.isDir()) return { files: [], totalBytes: 0 };
    const files: { name: string; size: number }[] = [];
    let totalBytes = 0;
    for (const fname of fs.readdirSync(this.logDir)) {
      if (!isLogFilename(fname)) continue;
      try {
        const st = fs.statSync(this.filePath(fname));
        files.push({ name: fname, size: st.size });
        totalBytes += st.size;
      } catch {
        /* ignore */
      }
    }
    return { files, totalBytes };
  }

  /** 读取文件内容（末 1MB + 跳过不完整首行 + utf-8 replace），对齐 read_log_file */
  readFileContent(name: string): { name: string; content: string; truncated: boolean } {
    const maxBytes = 1024 * 1024;
    const filePath = this.filePath(name);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      throw new Error('文件不存在');
    }
    const fd = fs.openSync(filePath, 'r');
    try {
      const actualSize = fs.fstatSync(fd).size;
      let truncated = false;
      let start = 0;
      if (actualSize > maxBytes) {
        truncated = true;
        start = Math.max(0, actualSize - maxBytes);
        // 跳过不完整首行
        const buf = Buffer.alloc(maxBytes);
        fs.readSync(fd, buf, 0, maxBytes, start);
        const nl = buf.indexOf(0x0a);
        start += nl >= 0 ? nl + 1 : 0;
      }
      const len = actualSize - start;
      const buf = Buffer.alloc(Math.max(0, len));
      fs.readSync(fd, buf, 0, len, start);
      return { name, content: buf.toString('utf-8'), truncated };
    } finally {
      fs.closeSync(fd);
    }
  }

  /** 下载流（调用方负责校验文件名；文件不存在抛 Error('文件不存在')） */
  createDownloadStream(name: string): { stream: fs.ReadStream; name: string; size: number } {
    const filePath = this.filePath(name);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      throw new Error('文件不存在');
    }
    return { stream: fs.createReadStream(filePath), name, size: fs.statSync(filePath).size };
  }

  /** 删除单个文件；不存在抛 Error('文件不存在') */
  deleteFile(name: string): void {
    const filePath = this.filePath(name);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      throw new Error('文件不存在');
    }
    fs.rmSync(filePath, { force: true });
  }

  /** 删除全部 .log（对齐 delete_all_log_files：deleted/failed 列表） */
  deleteAll(): { deleted: string[]; failed: { name: string; error: string }[] } {
    const deleted: string[] = [];
    const failed: { name: string; error: string }[] = [];
    if (!this.isDir()) return { deleted, failed };
    for (const fname of fs.readdirSync(this.logDir)) {
      if (!isLogFilename(fname)) continue;
      try {
        fs.rmSync(this.filePath(fname), { force: true });
        deleted.push(fname);
      } catch (err) {
        failed.push({ name: fname, error: String(err) });
      }
    }
    return { deleted, failed };
  }

  /**
   * 打包全部 .log 为 ZIP（对齐 export_logs_zip）。
   * 目录不存在 → null（404）；目录存在但无 .log → 空 ZIP（200，对齐 Python）。
   */
  async exportZip(): Promise<{ zip: Buffer; empty: boolean } | null> {
    if (!this.isDir()) return null;
    const logFiles = fs
      .readdirSync(this.logDir)
      .filter((f) => isLogFilename(f))
      .sort();
    const zip = new JSZip();
    let added = 0;
    for (const fname of logFiles) {
      try {
        zip.file(fname, fs.readFileSync(this.filePath(fname)));
        added += 1;
      } catch (err) {
        console.warn(`[control-svc] 日志导出跳过 ${fname}: ${String(err)}`);
      }
    }
    const buf = await zip.generateAsync({
      type: 'nodebuffer',
      compression: 'DEFLATE',
    });
    return { zip: buf, empty: logFiles.length === 0 || added === 0 };
  }
}
