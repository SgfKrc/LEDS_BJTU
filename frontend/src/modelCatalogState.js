function normalizedKey(value) {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function artifactAliases(artifact) {
  const reference = artifact?.reference || {};
  const source = artifact?.source || {};
  const repoId = String(source.repo_id || '').trim();
  const repoName = repoId.includes('/') ? repoId.split('/').pop() : repoId;
  return [reference.name, repoId, repoName]
    .map(normalizedKey)
    .filter(Boolean);
}

function modelAliases(model) {
  const repoId = String(model?.huggingface_id || '').trim();
  const repoName = repoId.includes('/') ? repoId.split('/').pop() : repoId;
  return [model?.model_id, repoId, repoName]
    .map(normalizedKey)
    .filter(Boolean);
}

function artifactStatusReason(artifact) {
  const status = artifact?.runtime_check?.status;
  if (artifact?.runnable || status === 'ready') {
    return '工件已通过运行时检查，由任务路由或对应 Sidecar 管线加载。';
  }
  if (status === 'stale') return '工件已登记，但运行时检查已失效，请先在模型工件区复检。';
  if (status) return `工件已登记，运行时状态为 ${status}，暂不接入单机模型切换。`;
  return '工件已登记，等待运行时检查；暂不接入单机模型切换。';
}

function artifactModel(artifact, index) {
  const reference = artifact?.reference || {};
  const namespace = String(reference.namespace || 'local');
  const name = String(reference.name || artifact?.family || `artifact-${index + 1}`);
  const tag = String(reference.tag || 'latest');
  const sourceRepo = String(artifact?.source?.repo_id || '').trim();
  const runtimeReady = Boolean(artifact?.runnable || artifact?.runtime_check?.status === 'ready');

  return {
    model_id: `fleet:${namespace}/${name}:${tag}`,
    name: sourceRepo || name,
    description: `MODEL-FLEET 工件 ${namespace}/${name}:${tag}`,
    model_type: String(artifact?.format || 'unknown'),
    max_context: artifact?.context_length ?? null,
    recommended_vram_gb: null,
    available_formats: artifact?.format ? [String(artifact.format)] : [],
    supported_engines: artifact?.engine ? [String(artifact.engine)] : [],
    preferred_engine: String(artifact?.engine || 'auto'),
    is_available: false,
    has_local_asset: true,
    is_builtin: false,
    is_experimental: true,
    is_fleet_artifact: true,
    is_editable: false,
    fleet_runnable: runtimeReady,
    fleet_artifacts: [artifact],
    unavailable_reason: artifactStatusReason(artifact),
    catalog_source: 'fleet',
    _source_index: index,
  };
}

function localAssetAliases(asset) {
  return [asset?.model_id, asset?.huggingface_id, asset?.name, ...(asset?.asset_ids || [])]
    .map(normalizedKey)
    .filter(Boolean);
}

function localAssetReason(asset) {
  return asset?.runtime_hint
    || '本地资产已发现；请使用匹配的专用运行时或任务路由加载。';
}

function localAssetModel(asset, index) {
  const formats = Array.isArray(asset?.available_formats) ? asset.available_formats : [];
  const modelId = String(asset?.model_id || `asset-${index + 1}`);
  return {
    model_id: `local:${modelId}`,
    name: String(asset?.name || modelId),
    description: '根目录 models/ 已发现的本地模型资产',
    model_type: String(asset?.model_type || formats.join(' + ') || 'unknown'),
    max_context: asset?.max_context ?? null,
    recommended_vram_gb: null,
    available_formats: formats,
    supported_engines: asset?.runtime_profile ? [String(asset.runtime_profile)] : [],
    preferred_engine: 'specialized_runtime',
    is_available: false,
    has_local_asset: true,
    is_builtin: false,
    is_experimental: true,
    is_fleet_artifact: false,
    is_local_discovered_asset: true,
    is_editable: false,
    fleet_runnable: false,
    fleet_artifacts: [],
    local_asset: asset,
    runtime_status: String(asset?.runtime_status || 'inventory_only'),
    unavailable_reason: localAssetReason(asset),
    catalog_source: 'local-asset',
    _source_index: index,
  };
}

export function mergeModelCatalog(models = [], artifacts = [], activeModelId = '', localAssets = []) {
  const catalog = (Array.isArray(models) ? models : []).map((model, index) => ({
    ...model,
    has_local_asset: Boolean(model?.is_available),
    is_fleet_artifact: false,
    is_editable: Boolean(model?.is_experimental && !model?.is_builtin),
    fleet_runnable: false,
    fleet_artifacts: [],
    catalog_source: 'registry',
    _source_index: index,
  }));

  const aliases = new Map();
  catalog.forEach((model, index) => {
    modelAliases(model).forEach((alias) => {
      if (!aliases.has(alias)) aliases.set(alias, index);
    });
  });

  (Array.isArray(artifacts) ? artifacts : []).forEach((artifact, index) => {
    const matchIndex = artifactAliases(artifact)
      .map((alias) => aliases.get(alias))
      .find((value) => value !== undefined);
    if (matchIndex === undefined) {
      catalog.push(artifactModel(artifact, catalog.length + index));
      return;
    }

    const current = catalog[matchIndex];
    const fleetArtifacts = [...current.fleet_artifacts, artifact];
    catalog[matchIndex] = {
      ...current,
      has_local_asset: true,
      fleet_runnable: fleetArtifacts.some(
        (entry) => entry?.runnable || entry?.runtime_check?.status === 'ready',
      ),
      fleet_artifacts: fleetArtifacts,
      catalog_source: 'registry+fleet',
      unavailable_reason: current.is_available
        ? current.unavailable_reason
        : artifactStatusReason(artifact),
    };
  });

  const localAliases = new Map();
  catalog.forEach((model, index) => {
    modelAliases(model).forEach((alias) => {
      if (!localAliases.has(alias)) localAliases.set(alias, index);
    });
    (model.fleet_artifacts || []).forEach((artifact) => {
      artifactAliases(artifact).forEach((alias) => {
        if (!localAliases.has(alias)) localAliases.set(alias, index);
      });
    });
  });

  (Array.isArray(localAssets) ? localAssets : []).forEach((asset, index) => {
    const matchIndex = localAssetAliases(asset)
      .map((alias) => localAliases.get(alias))
      .find((value) => value !== undefined);
    if (matchIndex === undefined) {
      const localModel = localAssetModel(asset, catalog.length + index);
      catalog.push(localModel);
      localAssetAliases(asset).forEach((alias) => {
        if (!localAliases.has(alias)) localAliases.set(alias, catalog.length - 1);
      });
      return;
    }

    const current = catalog[matchIndex];
    catalog[matchIndex] = {
      ...current,
      has_local_asset: true,
      is_local_discovered_asset: true,
      local_asset: asset,
      catalog_source: current.catalog_source === 'registry'
        ? 'registry+local-asset'
        : `${current.catalog_source}+local-asset`,
      unavailable_reason: current.is_available
        ? current.unavailable_reason
        : localAssetReason(asset),
      available_formats: [...new Set([
        ...(current.available_formats || []),
        ...(asset?.available_formats || []),
      ])],
      model_type: asset?.model_type || current.model_type,
      max_context: asset?.max_context || current.max_context,
    };
  });

  return catalog
    .sort((left, right) => {
      const isActive = (model) => (
        model.model_id === activeModelId || model.local_asset?.model_id === activeModelId
      );
      const activeDelta = Number(isActive(right)) - Number(isActive(left));
      if (activeDelta) return activeDelta;
      const presentDelta = Number(right.has_local_asset) - Number(left.has_local_asset);
      if (presentDelta) return presentDelta;
      const loadableDelta = Number(right.is_available || right.fleet_runnable)
        - Number(left.is_available || left.fleet_runnable);
      if (loadableDelta) return loadableDelta;
      const nameDelta = String(left.name || left.model_id).localeCompare(
        String(right.name || right.model_id), 'zh-CN', { sensitivity: 'base' },
      );
      return nameDelta || left._source_index - right._source_index;
    })
    .map(({ _source_index, ...model }) => model);
}
