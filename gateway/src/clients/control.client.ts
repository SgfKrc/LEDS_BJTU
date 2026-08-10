/**
 * control-svc 客户端（阶段 3 控制面服务）
 *
 * 配置：QLH_CONTROL_URL（默认 http://127.0.0.1:8030）。
 * 内部端点：/sessions /conversations /user/settings /cluster/review /
 * /workflows /bootstrap /auth /users /models/registry|gguf|download /logs（对外 /api/*
 * 去掉 /api 前缀后透传）。
 *
 * 渐进切换：网关一般控制面域默认仍走 legacy-control（并行共存基线），
 * 显式设置 QLH_CONTROL_URL 后已迁移域改走 control-svc。认证与用户域没有
 * legacy 实现，因此始终使用本客户端（未配置时为 loopback :8030）。
 */
import { ForwardClient } from './forward-client';

export class ControlClient extends ForwardClient {
  constructor(baseUrl?: string, timeoutMs?: number) {
    super(
      baseUrl ||
        process.env.QLH_CONTROL_URL ||
        'http://127.0.0.1:8030',
      timeoutMs,
    );
  }
}
