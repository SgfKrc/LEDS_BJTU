import { HfResolver } from '../src/data/hf-resolver';
import { ModelHttpClient } from '../src/data/model-http-client';

describe('MODEL-FLEET M3 Hugging Face blob resolver', () => {
  it('requests blob metadata and maps LFS size and SHA-256', async () => {
    let requestedUrl = '';
    const digest = 'a'.repeat(64);
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async (url) => {
        requestedUrl = String(url);
        return new Response(JSON.stringify({
          sha: 'b'.repeat(40),
          siblings: [
            { rfilename: 'config.json', size: 12 },
            {
              rfilename: 'model.gguf',
              lfs: { size: 1024, sha256: digest },
            },
          ],
        }));
      },
    });
    const resolved = await new HfResolver(http).resolve(
      'org/model', 'feature/revision', ['*.gguf'],
    );
    expect(requestedUrl).toContain('revision=feature%2Frevision');
    expect(requestedUrl).toContain('blobs=true');
    expect(resolved.files).toEqual([{
      rfilename: 'model.gguf', size: 1024, sha256: digest,
    }]);
  });

  it('fails closed when blob size metadata is missing', async () => {
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response(JSON.stringify({
        sha: 'c'.repeat(40),
        siblings: [{ rfilename: 'model.gguf' }],
      })),
    });
    await expect(new HfResolver(http).resolve('org/model'))
      .rejects.toThrow('HF resolve file size is invalid: model.gguf');
  });

  it('preserves a legitimate zero-byte file', async () => {
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response(JSON.stringify({
        sha: 'd'.repeat(40),
        siblings: [{ rfilename: 'empty.txt', size: 0 }],
      })),
    });
    const resolved = await new HfResolver(http).resolve('org/model');
    expect(resolved.files[0].size).toBe(0);
  });
});
