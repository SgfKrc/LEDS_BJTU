import { HttpException, HttpStatus } from '@nestjs/common';
import { JsonDetailFilter } from '../src/common/json-detail.filter';

/** T2：9a71ccb gateway request_id fallback 回归（测试修复票排期 P0）。 */

function buildHost(requestId: string | undefined) {
  const headers: Record<string, string> = {};
  const sent: { body?: unknown } = {};
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const res: any = {
    status: jest.fn().mockReturnThis(),
    header: jest.fn((k: string, v: string) => {
      headers[k] = v;
      return res;
    }),
    send: jest.fn((body: unknown) => {
      sent.body = body;
      return res;
    }),
  };
  const req = {
    requestId,
    method: 'GET',
    url: '/api/test',
  };
  const host = {
    switchToHttp: () => ({ getResponse: () => res, getRequest: () => req }),
  };
  return { headers, sent, res, host };
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

describe('JsonDetailFilter request_id fallback (T2)', () => {
  it('拦截器未执行（req.requestId undefined）时生成真实 uuid，而非 "-"', () => {
    const { headers, sent, host } = buildHost(undefined);
    const filter = new JsonDetailFilter();
    filter.catch(
      new HttpException('bad json', HttpStatus.BAD_REQUEST),
      host as never,
    );
    const requestId = headers['X-Request-ID'];
    expect(requestId).toBeDefined();
    expect(requestId).not.toBe('-');
    expect(requestId).toMatch(UUID_RE);
    expect(sent.body).toMatchObject({ request_id: requestId });
  });

  it('已有 requestId 时保留原值（不改写）', () => {
    const { headers, sent, host } = buildHost('existing-123');
    const filter = new JsonDetailFilter();
    filter.catch(
      new HttpException('bad', HttpStatus.BAD_REQUEST),
      host as never,
    );
    expect(headers['X-Request-ID']).toBe('existing-123');
    expect(sent.body).toMatchObject({ request_id: 'existing-123' });
  });

  it('HttpException 状态码透传', () => {
    const { headers, sent, host } = buildHost(undefined);
    const filter = new JsonDetailFilter();
    filter.catch(
      new HttpException('not found', HttpStatus.NOT_FOUND),
      host as never,
    );
    expect(headers['X-Request-ID']).toMatch(UUID_RE);
    expect(sent.body).toMatchObject({ detail: 'not found' });
  });

  it('500 级异常仍生成 request_id（不依赖拦截器）', () => {
    const { headers, sent, host } = buildHost(undefined);
    const filter = new JsonDetailFilter();
    filter.catch(new Error('boom'), host as never);
    expect(headers['X-Request-ID']).toMatch(UUID_RE);
    expect(sent.body).toMatchObject({ detail: '服务器内部错误，请查看后端日志' });
  });
});
