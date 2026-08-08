import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { ArtifactStore } from './artifact-store';
import { ResolvedFile } from './hf-resolver';

export interface DiskBudgetResult {
  total_bytes: number;
  existing_bytes: number;
  disk_required_bytes: number;
  disk_available_bytes: number;
  sufficient: boolean;
}

@Injectable()
export class ModelDiskBudget {
  evaluate(files: ResolvedFile[], store: ArtifactStore): DiskBudgetResult {
    const totalBytes = files.reduce((total, file) => total + file.size, 0);
    const existingBytes = files.reduce((total, file) => (
      file.sha256 && store.blobExists(file.sha256) ? total + file.size : total
    ), 0);
    const diskRequiredBytes = Math.max(0, totalBytes - existingBytes);
    const diskAvailableBytes = this.availableBytes(store.root);
    return {
      total_bytes: totalBytes,
      existing_bytes: existingBytes,
      disk_required_bytes: diskRequiredBytes,
      disk_available_bytes: diskAvailableBytes,
      sufficient: diskAvailableBytes >= diskRequiredBytes,
    };
  }

  availableBytes(target: string): number {
    let probe = path.resolve(target);
    while (!fs.existsSync(probe)) {
      const parent = path.dirname(probe);
      if (parent === probe) break;
      probe = parent;
    }
    const stats = fs.statfsSync(probe);
    return Number(stats.bavail) * Number(stats.bsize);
  }
}
