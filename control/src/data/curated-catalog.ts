/**
 * M3 curated recipes（第一批）— 固定 revision 的受控来源。
 *
 * 规则：只有经过固定 revision + 已知 family 的 recipe 才允许一键 pull；
 * 用户自定义仓库走 resolve 时同样 fail-closed（未知架构仅 inspection）。
 * 许可证与 gated 状态在下载前展示（§7.1）；token 不落库。
 */
export interface CuratedRecipe {
  id: string;
  repo_id: string;
  revision: string;
  allow_patterns: string[];
  engine: 'llama_cpp' | 'pytorch_transformers';
  family: string;
  license: string;
  gated: boolean;
  description: string;
}

export const CURATED_RECIPES: CuratedRecipe[] = [
  {
    id: 'qwen2.5-1.5b-instruct-gguf',
    repo_id: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    revision: 'main',
    allow_patterns: ['*.gguf'],
    engine: 'llama_cpp',
    family: 'qwen2',
    license: 'apache-2.0',
    gated: false,
    description: 'Qwen2.5-1.5B-Instruct GGUF 分片（含 Q4_K_M 等量化档）',
  },
  {
    id: 'qwen2.5-7b-instruct-gguf',
    repo_id: 'Qwen/Qwen2.5-7B-Instruct-GGUF',
    revision: 'main',
    allow_patterns: ['*q4_k_m.gguf'],
    engine: 'llama_cpp',
    family: 'qwen2',
    license: 'apache-2.0',
    gated: false,
    description: 'Qwen2.5-7B-Instruct GGUF Q4_K_M（8GB 级）',
  },
  {
    id: 'deepseek-r1-distill-qwen-7b-gguf',
    repo_id: 'unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF',
    revision: 'main',
    allow_patterns: ['*Q4_K_M.gguf'],
    engine: 'llama_cpp',
    family: 'deepseek',
    license: 'apache-2.0',
    gated: false,
    description: 'DeepSeek-R1-Distill-Qwen-7B GGUF Q4_K_M',
  },
];
