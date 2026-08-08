import {
  Body, Controller, Get, HttpCode, HttpException, Param, Post,
} from '@nestjs/common';
import {
  DeploymentSimulator, SimulationNode,
} from '../../data/deployment-simulator';

class SimulationRequest {
  artifact_id?: string;
  runtime_profile?: string;
  required_capabilities?: string[];
  nodes?: SimulationNode[];
}

@Controller('models/deployment-simulations')
export class DeploymentSimulationController {
  constructor(private readonly simulator: DeploymentSimulator) {}

  @Post()
  @HttpCode(201)
  create(@Body() body: SimulationRequest): Record<string, unknown> {
    try {
      if (!body?.artifact_id || !body.nodes || !body.runtime_profile) {
        throw new Error('artifact_id, runtime_profile and nodes are required');
      }
      return {
        status: 'created',
        plan: this.simulator.createPlan({
          artifactId: body.artifact_id,
          runtimeProfile: body.runtime_profile,
          nodes: body.nodes,
          requiredCapabilities: body.required_capabilities,
        }),
      };
    } catch (error) {
      throw new HttpException(error instanceof Error ? error.message : String(error), 422);
    }
  }

  @Get(':planId')
  get(@Param('planId') planId: string): Record<string, unknown> {
    const plan = this.simulator.get(planId);
    if (!plan) throw new HttpException(`simulation not found: ${planId}`, 404);
    return { plan };
  }

  @Post(':planId/prepare')
  @HttpCode(200)
  prepare(@Param('planId') planId: string, @Body() body: { nodes?: SimulationNode[] }): Record<string, unknown> {
    try {
      return { plan: this.simulator.prepare(planId, body?.nodes ?? []) };
    } catch (error) {
      throw new HttpException(error instanceof Error ? error.message : String(error), 422);
    }
  }

  @Post(':planId/activate')
  @HttpCode(200)
  activate(@Param('planId') planId: string): Record<string, unknown> {
    try {
      return { plan: this.simulator.activate(planId) };
    } catch (error) {
      throw new HttpException(error instanceof Error ? error.message : String(error), 422);
    }
  }

  @Post(':planId/rollback')
  @HttpCode(200)
  rollback(@Param('planId') planId: string): Record<string, unknown> {
    try {
      return { plan: this.simulator.rollback(planId) };
    } catch (error) {
      throw new HttpException(error instanceof Error ? error.message : String(error), 422);
    }
  }
}
