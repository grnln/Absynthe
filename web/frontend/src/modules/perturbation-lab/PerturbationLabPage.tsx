import { useEffect, useMemo, useState } from 'react'
import { FlaskConical, Plus, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getPerturbations } from '@/api/capabilities'
import { applyPerturbation } from '@/api/perturbation'
import type {
  ParamSchema,
  PerturbationConfig,
  PerturbationPreview,
  PerturbationSchema,
} from '@/api/types'
import { GraphDiffView } from '@/components/graph/GraphDiffView'
import { PageContainer, PagePanel } from '@/components/layout/PageContainer'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Slider } from '@/components/ui/Slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useGraphState } from '@/hooks/useGraphState'

function schemaDefaults(schema: PerturbationSchema): Record<string, unknown> {
  const defaults: Record<string, unknown> = {}
  for (const param of schema.params) {
    defaults[param.name] = param.default
  }
  return defaults
}

function parseParamValue(param: ParamSchema, raw: string): string | number {
  if (param.type === 'int') return Number.parseInt(raw || '0', 10)
  if (param.type === 'float') return Number.parseFloat(raw || '0')
  return raw
}

function previewTabLabel(
  preview: PerturbationPreview,
  schemas: Record<string, PerturbationSchema>,
): string {
  const schema = schemas[preview.perturbation_type]
  const base = schema?.label ?? preview.perturbation_type
  return `${base} ${preview.config_index + 1}`
}

export function PerturbationLabPage() {
  const navigate = useNavigate()
  const {
    graphId,
    setPerturbationResult,
    perturbationResult,
    labelingFunctions,
    perturbations: savedPerturbations,
    setPerturbations,
  } = useGraphState()

  const [schemas, setSchemas] = useState<PerturbationSchema[]>([])
  const [configs, setConfigs] = useState<PerturbationConfig[]>(savedPerturbations)
  const [maxIterations, setMaxIterations] = useState(10)
  const [seed, setSeed] = useState<number | undefined>(undefined)
  const [activePreviewTab, setActivePreviewTab] = useState<string>('0')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getPerturbations()
      .then((loaded) => {
        setSchemas(loaded)
        if (configs.length === 0 && loaded.length > 0) {
          setConfigs([{ type: loaded[0].id, params: schemaDefaults(loaded[0]), count: 1 }])
        }
      })
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const schemaMap = useMemo(
    () => Object.fromEntries(schemas.map((schema) => [schema.id, schema])),
    [schemas],
  )

  const previews = perturbationResult?.previews ?? []
  const activePreview =
    previews.find((preview) => String(preview.config_index) === activePreviewTab) ??
    previews[0]

  const addConfig = (type: string) => {
    const schema = schemaMap[type]
    setConfigs((prev) => [
      ...prev,
      {
        type,
        params: schema ? schemaDefaults(schema) : {},
        count: 1,
      },
    ])
  }

  const removeConfig = (idx: number) => {
    setConfigs((prev) => prev.filter((_, i) => i !== idx))
  }

  const updateParam = (idx: number, name: string, value: unknown) => {
    setConfigs((prev) =>
      prev.map((cfg, i) => (i === idx ? { ...cfg, params: { ...cfg.params, [name]: value } } : cfg)),
    )
  }

  const updateCount = (idx: number, count: number) => {
    setConfigs((prev) =>
      prev.map((cfg, i) => (i === idx ? { ...cfg, count } : cfg)),
    )
  }

  const handleApply = async () => {
    if (!graphId || configs.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const res = await applyPerturbation({
        graph_id: graphId,
        labeling_functions: labelingFunctions,
        perturbations: configs,
        max_iterations: maxIterations,
        ...(seed !== undefined && { seed }),
      })
      setPerturbations(configs)
      setPerturbationResult(res, configs)
      const firstSuccess = res.previews.find((preview) => preview.success)
      const fallback = res.previews[0]
      if (firstSuccess) {
        setActivePreviewTab(String(firstSuccess.config_index))
      } else if (fallback) {
        setActivePreviewTab(String(fallback.config_index))
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageContainer split>
      <PagePanel className="w-72 shrink-0 overflow-y-auto">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Perturbations</p>
        <p className="text-xs text-gray-500 leading-relaxed">
          Add one or more perturbation modules and tune each module parameters.
        </p>

        <div className="flex flex-col gap-2">
          {configs.map((cfg, idx) => {
            const schema = schemaMap[cfg.type]
            return (
              <Card key={`${cfg.type}-${idx}`} className="border-white/10">
                <CardContent className="p-3 flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Badge color="#f59e0b">{schema?.label ?? cfg.type}</Badge>
                    <button
                      onClick={() => removeConfig(idx)}
                      className="ml-auto text-gray-600 hover:text-red-400 transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <Slider
                    label="Desired count"
                    value={cfg.count}
                    min={1}
                    max={10}
                    onValueChange={(value) => updateCount(idx, value)}
                  />
                  <p className="text-[11px] text-amber-600 leading-relaxed">
                    Used for dataset generation volume. Preview tabs show one successful example per perturbation.
                  </p>

                  {schema?.params.map((param) => (
                    <div key={`${cfg.type}-${idx}-${param.name}`}>
                      {param.type === 'select' ? (
                        <Select
                          label={param.label}
                          value={String(cfg.params[param.name] ?? param.default)}
                          onValueChange={(value) => updateParam(idx, param.name, value)}
                          options={(param.options ?? []).map((opt) => ({ value: opt, label: opt }))}
                        />
                      ) : (
                        <Input
                          label={param.label}
                          type={param.type === 'string' ? 'text' : 'number'}
                          value={String(cfg.params[param.name] ?? param.default)}
                          onChange={(e) =>
                            updateParam(
                              idx,
                              param.name,
                              parseParamValue(param, e.target.value),
                            )
                          }
                        />
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>
            )
          })}
        </div>

        {schemas.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {schemas.map((schema) => (
              <Button key={schema.id} variant="outline" size="sm" onClick={() => addConfig(schema.id)}>
                <Plus className="w-3 h-3" />
                {schema.label}
              </Button>
            ))}
          </div>
        )}

        <div className="border-t border-white/10 pt-4 flex flex-col gap-3">
          <Slider
            label="Max iterations per perturbation"
            value={maxIterations}
            min={1}
            max={100}
            onValueChange={setMaxIterations}
          />
          <Input
            label="Seed (optional)"
            type="number"
            value={seed !== undefined ? String(seed) : ''}
            onChange={(e) => setSeed(e.target.value ? Number.parseInt(e.target.value, 10) : undefined)}
          />
        </div>

        {!graphId && (
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">
            No graph loaded. Generate and label a graph first.
          </p>
        )}
        {error && <p className="text-xs text-red-400 bg-red-900/20 rounded-lg p-2">{error}</p>}

        {perturbationResult && !perturbationResult.success && (
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">
            {perturbationResult.message}
          </p>
        )}

        <Button
          onClick={handleApply}
          loading={loading}
          disabled={!graphId || configs.length === 0}
          className="w-full"
        >
          <FlaskConical className="w-4 h-4" />
          Apply Perturbations
        </Button>

        {perturbationResult?.success && (
          <Button variant="outline" onClick={() => navigate('/dataset')} className="w-full">
            Dataset Studio →
          </Button>
        )}
      </PagePanel>

      <div className="flex-1 min-w-0 overflow-hidden">
        {perturbationResult && previews.length > 0 ? (
          <Tabs
            value={activePreview ? String(activePreview.config_index) : '0'}
            onValueChange={setActivePreviewTab}
            className="h-full flex flex-col"
          >
            <TabsList className="mb-2 w-fit max-w-full overflow-x-auto">
              {previews.map((preview) => (
                <TabsTrigger
                  key={preview.config_index}
                  value={String(preview.config_index)}
                  className={preview.success ? '' : 'text-red-300 data-[state=active]:text-red-200'}
                >
                  {previewTabLabel(preview, schemaMap)}
                </TabsTrigger>
              ))}
            </TabsList>

            {previews.map((preview) => (
              <TabsContent
                key={preview.config_index}
                value={String(preview.config_index)}
                className="mt-0 flex-1 min-h-0"
              >
                {preview.success ? (
                  <GraphDiffView
                    originalElements={preview.original_elements}
                    perturbedElements={preview.perturbed_elements}
                    removedNodes={preview.removed_nodes}
                    changedNodes={preview.changed_nodes}
                    edgePerturbInfo={preview.edge_perturb_info}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-500">
                    <FlaskConical className="w-10 h-10 opacity-30" />
                    <p className="text-sm text-center max-w-md">{preview.message}</p>
                  </div>
                )}
              </TabsContent>
            ))}
          </Tabs>
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-600">
            <FlaskConical className="w-12 h-12 opacity-30" />
            <p className="text-sm">Configure and apply perturbations to inspect the diff</p>
          </div>
        )}
      </div>
    </PageContainer>
  )
}
