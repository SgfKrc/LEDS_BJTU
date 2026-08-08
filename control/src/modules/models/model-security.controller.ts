import {
  Body, Controller, Delete, Get, HttpCode, HttpException, Param, Post, Put, Req,
} from '@nestjs/common';
import { FastifyRequest } from 'fastify';
import {
  credentialRefForId, ModelCredentialStore,
} from '../../data/model-credential-store';
import { ModelHttpClient } from '../../data/model-http-client';
import { ModelLicenseAcceptanceRepository } from '../../data/model-license-acceptance';

class CredentialRequest {
  secret?: string;
}

class LicenseAcceptanceRequest {
  repo_id?: string;
  license_id?: string;
  accepted?: boolean;
}

@Controller('models')
export class ModelSecurityController {
  constructor(
    private readonly credentials: ModelCredentialStore,
    private readonly licenses: ModelLicenseAcceptanceRepository,
    private readonly http: ModelHttpClient,
  ) {}

  @Put('credentials/:credentialId')
  async putCredential(
    @Param('credentialId') credentialId: string,
    @Body() body: CredentialRequest,
    @Req() request: FastifyRequest,
  ): Promise<Record<string, unknown>> {
    this.assertLocal(request);
    try {
      const ref = credentialRefForId(credentialId);
      if (typeof body?.secret !== 'string') {
        throw new Error('secret is required');
      }
      return { status: 'saved', credential: await this.credentials.set(ref, body.secret) };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  @Get('credentials/:credentialId')
  getCredential(@Param('credentialId') credentialId: string): Record<string, unknown> {
    try {
      return { credential: this.credentials.status(credentialRefForId(credentialId)) };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  @Delete('credentials/:credentialId')
  @HttpCode(200)
  deleteCredential(
    @Param('credentialId') credentialId: string,
    @Req() request: FastifyRequest,
  ): Record<string, unknown> {
    this.assertLocal(request);
    try {
      const ref = credentialRefForId(credentialId);
      return {
        status: this.credentials.delete(ref) ? 'deleted' : 'not_found',
        credential_ref: ref,
      };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  @Get('network')
  network(): Record<string, unknown> {
    return { proxy: this.http.proxyStatus() };
  }

  @Get('licenses/acceptances')
  listAcceptances(): Record<string, unknown> {
    return { acceptances: this.licenses.list() };
  }

  @Post('licenses/acceptances')
  @HttpCode(200)
  acceptLicense(
    @Body() body: LicenseAcceptanceRequest,
    @Req() request: FastifyRequest,
  ): Record<string, unknown> {
    this.assertLocal(request);
    if (body?.accepted !== true) {
      throw new HttpException('accepted=true is required', 422);
    }
    try {
      return {
        status: 'accepted',
        acceptance: this.licenses.accept(body.repo_id ?? '', body.license_id ?? ''),
      };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  @Delete('licenses/acceptances')
  @HttpCode(200)
  revokeLicense(
    @Body() body: LicenseAcceptanceRequest,
    @Req() request: FastifyRequest,
  ): Record<string, unknown> {
    this.assertLocal(request);
    try {
      return {
        status: this.licenses.revoke(
          body.repo_id ?? '', body.license_id ?? '',
        ) ? 'revoked' : 'not_found',
      };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  private assertLocal(request: FastifyRequest): void {
    const address = request.ip;
    if (address !== '127.0.0.1' && address !== '::1'
        && address !== '::ffff:127.0.0.1') {
      throw new HttpException('model security mutations are local-only', 403);
    }
  }
}
