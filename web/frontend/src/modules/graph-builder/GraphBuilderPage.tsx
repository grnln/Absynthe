import { useEffect, useRef, useState } from 'react'
import { Dice5, FolderUp, GripVertical, Plus, Trash2, Upload } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getCompositions, getDistributions, getMotifs } from '@/api/capabilities'
import { generateGraph, uploadGraphs } from '@/api/graph'
import type { CompositionSchema, DistributionSchema, IntDistribution, MotifConfig, MotifSchema } from '@/api/types'
import { GraphCanvas } from '@/components/graph/GraphCanvas'
import { GraphInfoPanel } from '@/components/graph/GraphInfoPanel'
import { PageContainer, PagePanel } from '@/components/layout/PageContainer'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Slider } from '@/components/ui/Slider'
import { CountInput } from '@/components/ui/CountInput'
import { DistributionInput } from '@/components/ui/DistributionInput'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useGraphState } from '@/hooks/useGraphState'

const SUPPORTED_GRAPH_EXTS = ['.graphml', '.rdf', '.ttl', '.nt', '.n3', '.owl'] as const

export function GraphBuilderPage() {
  const navigate = useNavigate()
  const { setGraph, setUploadedGraph, elements, graphId, graphSource, uploadFileCount } = useGraphState()

  // ── Motif mode state ──
  const [motifSchemas, setMotifSchemas] = useState<MotifSchema[]>([])
  const [compSchemas, setCompSchemas] = useState<CompositionSchema[]>([])
  const [distributionSchemas, setDistributionSchemas] = useState<DistributionSchema[]>([])
  const [motifs, setMotifs] = useState<MotifConfig[]>([{ type: 'cycle', params: [4] }, { type: 'house', params: [] }])
  const [composition, setComposition] = useState('sequential')
  const [compParams, setCompParams] = useState<Record<string, number | string>>({})
  const [extraVertices, setExtraVertices] = useState(0)
  const [extraVerticesDist, setExtraVerticesDist] = useState<IntDistribution | undefined>(undefined)
  const [extraEdges, setExtraEdges] = useState(0)
  const [extraEdgesDist, setExtraEdgesDist] = useState<IntDistribution | undefined>(undefined)
  const [seed, setSeed] = useState<number | undefined>(undefined)

  // ── Upload mode state ──
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [iterationOrder, setIterationOrder] = useState('sequential')
  const [exhaustionPolicy, setExhaustionPolicy] = useState('stop')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Shared state ──
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<{ id: string; data: Record<string, unknown> } | null>(null)

  useEffect(() => {
    getMotifs().then(setMotifSchemas).catch(() => {})
    getCompositions().then(setCompSchemas).catch(() => {})
    getDistributions().then(setDistributionSchemas).catch(() => {})
  }, [])

  const selectedCompSchema = compSchemas.find((c) => c.id === composition)

  // ── Motif handlers ──
  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await generateGraph({
        motifs,
        composition,
        composition_params: compParams,
        num_extra_vertices: extraVertices,
        num_extra_edges: extraEdges,
        ...(extraVerticesDist && { num_extra_vertices_distribution: extraVerticesDist }),
        ...(extraEdgesDist && { num_extra_edges_distribution: extraEdgesDist }),
        ...(seed !== undefined && { seed }),
      })
      setGraph(res.graph_id, res.elements)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const addMotif = (type: string) => {
    const schema = motifSchemas.find((m) => m.type === type)
    const params = schema?.params.map((p) => p.default as number) ?? []
    setMotifs((prev) => [...prev, { type, params, count: 1 }])
  }

  const updateMotifParam = (idx: number, paramIdx: number, value: number | string) => {
    setMotifs((prev) =>
      prev.map((m, i) => {
        if (i !== idx) return m
        const newParams = [...m.params]
        newParams[paramIdx] = value
        return { ...m, params: newParams }
      }),
    )
  }

  const updateMotifCount = (idx: number, count: number) => {
    setMotifs((prev) =>
      prev.map((m, i) => (i === idx ? { ...m, count: Math.max(1, count), count_distribution: undefined } : m)),
    )
  }

  const toggleMotifDistribution = (idx: number) => {
    setMotifs((prev) =>
      prev.map((m, i) => {
        if (i !== idx) return m
        if (m.count_distribution) {
          return { ...m, count_distribution: undefined }
        }
        const defaultSchema = distributionSchemas[0]
        const params: Record<string, number> = {}
        for (const p of defaultSchema?.params ?? []) {
          params[p.name] = p.default as number
        }
        return {
          ...m,
          count_distribution: {
            type: (defaultSchema?.type ?? 'uniform') as IntDistribution['type'],
            params,
          },
        }
      }),
    )
  }

  const updateMotifDistribution = (idx: number, dist: IntDistribution) => {
    setMotifs((prev) =>
      prev.map((m, i) => (i === idx ? { ...m, count_distribution: dist } : m)),
    )
  }

  const removeMotif = (idx: number) => setMotifs((prev) => prev.filter((_, i) => i !== idx))

  const parseParam = (type: string, value: string): number | string => {
    if (type === 'int') return Number.parseInt(value || '0', 10)
    if (type === 'float') return Number.parseFloat(value || '0')
    return value
  }

  // ── Upload handlers ──
  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return
    const supportedFiles = Array.from(files).filter((f) =>
      SUPPORTED_GRAPH_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext)),
    )
    if (supportedFiles.length < files.length) {
      setError(`Some files were skipped because they are not supported graph files (${SUPPORTED_GRAPH_EXTS.join(', ')}).`)
    } else {
      setError(null)
    }
    setSelectedFiles((prev) => [...prev, ...supportedFiles])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeFile = (idx: number) => setSelectedFiles((prev) => prev.filter((_, i) => i !== idx))

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const res = await uploadGraphs(selectedFiles)
      setUploadedGraph(res.graph_id, res.elements, res.folder_path, res.file_count)
      if (res.warnings.length > 0) {
        const msgs = res.warnings.map((w) => `${w.filename}: ${w.error}`).join('\n')
        setError(`Some files had issues:\n${msgs}`)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageContainer split>
      {/* ── Left: Config ──────────────────────────────────── */}
      <PagePanel className="w-72 shrink-0 overflow-y-auto">
        <Tabs defaultValue="motif">
          <TabsList className="w-full h-10">
            <TabsTrigger value="motif" className="flex-1 py-1.5 border border-transparent data-[state=active]:border-brand-500/40 data-[state=active]:bg-brand-600/20 data-[state=active]:text-brand-300">From Motifs</TabsTrigger>
            <TabsTrigger value="upload" className="flex-1 py-1.5 border border-transparent data-[state=active]:border-brand-500/40 data-[state=active]:bg-brand-600/20 data-[state=active]:text-brand-300">Upload</TabsTrigger>
          </TabsList>

          {/* ── Tab: Motif generation ── */}
          <TabsContent value="motif">
            <div className="flex flex-col gap-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Motifs</p>

              {/* Motif list */}
              <div className="flex flex-col gap-2">
                {motifs.map((m, idx) => {
                  const schema = motifSchemas.find((s) => s.type === m.type)
                  return (
                    <Card key={idx} className="border-white/10">
                      <CardContent className="p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <GripVertical className="w-4 h-4 text-gray-600 shrink-0" />
                          <Badge color="#6366f1">{m.type}</Badge>
                          <button
                            onClick={() => toggleMotifDistribution(idx)}
                            className={`ml-auto p-1 rounded transition-colors ${m.count_distribution ? 'text-brand-400 bg-brand-500/20' : 'text-gray-600 hover:text-gray-400'}`}
                            title={m.count_distribution ? 'Switch to fixed count' : 'Use random distribution'}
                          >
                            <Dice5 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => removeMotif(idx)}
                            className="text-gray-600 hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <div className="mt-2">
                          {m.count_distribution ? (
                            <DistributionInput
                              value={m.count_distribution}
                              onChange={(dist) => updateMotifDistribution(idx, dist)}
                              schemas={distributionSchemas}
                            />
                          ) : (
                            <Slider
                              label="Count"
                              value={m.count ?? 1}
                              min={1}
                              max={20}
                              onValueChange={(v) => updateMotifCount(idx, v)}
                            />
                          )}
                        </div>
                        {schema?.params.map((p, pi) => (
                          <div key={p.name} className="mt-2">
                            <Slider
                              label={p.label}
                              value={(m.params[pi] as number) ?? (p.default as number)}
                              min={p.min ?? 1}
                              max={(p.max as number) ?? 20}
                              onValueChange={(v) => updateMotifParam(idx, pi, v)}
                            />
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {/* Add motif */}
              {motifSchemas.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {motifSchemas.map((s) => (
                    <Button key={s.type} variant="outline" size="sm" onClick={() => addMotif(s.type)}>
                      <Plus className="w-3 h-3" />
                      {s.label}
                    </Button>
                  ))}
                </div>
              )}

              <div className="border-t border-white/10 pt-4 flex flex-col gap-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Composition</p>
                <Select
                  value={composition}
                  onValueChange={(v) => { setComposition(v); setCompParams({}) }}
                  options={compSchemas.map((c) => ({ value: c.id, label: c.label }))}
                  label="Pattern"
                />
                {selectedCompSchema?.params.map((p) => (
                  <Input
                    key={p.name}
                    label={p.label}
                    type={p.type === 'string' ? 'text' : 'number'}
                    value={String(compParams[p.name] ?? p.default)}
                    onChange={(e) =>
                      setCompParams((prev) => ({
                        ...prev,
                        [p.name]: parseParam(p.type, e.target.value),
                      }))
                    }
                  />
                ))}
              </div>

              <div className="border-t border-white/10 pt-4 flex flex-col gap-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Extras</p>
                <CountInput
                  label="Extra vertices"
                  value={extraVertices}
                  distribution={extraVerticesDist}
                  onValueChange={setExtraVertices}
                  onDistributionChange={setExtraVerticesDist}
                  schemas={distributionSchemas}
                  min={0}
                  max={20}
                />
                <CountInput
                  label="Extra edges"
                  value={extraEdges}
                  distribution={extraEdgesDist}
                  onValueChange={setExtraEdges}
                  onDistributionChange={setExtraEdgesDist}
                  schemas={distributionSchemas}
                  min={0}
                  max={20}
                />
                <Input
                  label="Seed (optional)"
                  type="number"
                  value={seed !== undefined ? String(seed) : ''}
                  onChange={(e) => setSeed(e.target.value ? Number.parseInt(e.target.value, 10) : undefined)}
                />
              </div>

              <Button onClick={handleGenerate} loading={loading} className="w-full">
                Generate Graph
              </Button>
            </div>
          </TabsContent>

          {/* ── Tab: Upload graph files (GraphML / RDF) ── */}
          <TabsContent value="upload">
            <div className="flex flex-col gap-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Graph Files</p>

              {/* Drop zone / file input */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center gap-2 rounded-xl border-2 border-dashed border-white/15 hover:border-brand-500/40 bg-white/5 hover:bg-brand-500/5 p-6 transition-colors cursor-pointer"
              >
                <Upload className="w-6 h-6 text-gray-500" />
                <span className="text-xs text-gray-400">Click to select graph files ({SUPPORTED_GRAPH_EXTS.join(', ')})</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept={SUPPORTED_GRAPH_EXTS.join(',')}
                multiple
                className="hidden"
                onChange={handleFilesSelected}
              />

              {/* File list */}
              {selectedFiles.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  {selectedFiles.map((f, idx) => (
                    <div key={`${f.name}-${idx}`} className="flex items-center gap-2 rounded-lg bg-white/5 px-3 py-2">
                      <FolderUp className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                      <span className="text-xs text-gray-300 truncate flex-1">{f.name}</span>
                      <button
                        onClick={() => removeFile(idx)}
                        className="text-gray-600 hover:text-red-400 transition-colors text-xs"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  <p className="text-xs text-gray-500">{selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected</p>
                </div>
              )}

              {/* Multi-file options */}
              {selectedFiles.length > 1 && (
                <div className="border-t border-white/10 pt-4 flex flex-col gap-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Folder Options</p>
                  <Select
                    label="Iteration Order"
                    value={iterationOrder}
                    onValueChange={setIterationOrder}
                    options={[
                      { value: 'sequential', label: 'Sequential' },
                      { value: 'random', label: 'Random' },
                    ]}
                  />
                  <Select
                    label="Exhaustion Policy"
                    value={exhaustionPolicy}
                    onValueChange={setExhaustionPolicy}
                    options={[
                      { value: 'stop', label: 'Stop' },
                      { value: 'cycle', label: 'Cycle' },
                      { value: 'raise', label: 'Raise' },
                    ]}
                  />
                </div>
              )}

              <Button
                onClick={handleUpload}
                loading={loading}
                disabled={selectedFiles.length === 0}
                className="w-full"
              >
                <Upload className="w-4 h-4" />
                Upload & Preview
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        {/* ── Shared: error + navigation ── */}
        {error && <p className="text-xs text-red-400 bg-red-900/20 rounded-lg p-2 mt-4 whitespace-pre-line">{error}</p>}

        {graphId && graphSource === 'upload' && uploadFileCount > 1 && (
          <p className="text-xs text-brand-300 bg-brand-900/20 rounded-lg p-2 mt-3">
            Showing graph 1 of {uploadFileCount}. All graphs will be available in Dataset Studio.
          </p>
        )}

        {graphId && (
          <Button variant="outline" onClick={() => navigate('/labels')} className="w-full mt-3">
            Assign Labels →
          </Button>
        )}
      </PagePanel>

      {/* ── Right: Graph canvas ────────────────────────────── */}
      <div className="flex-1 min-w-0 relative rounded-2xl border border-white/10 overflow-hidden">
        <GraphCanvas
          elements={elements}
          toolbar
          onNodeClick={(id, data) => setSelectedNode({ id, data: data as Record<string, unknown> })}
          className="h-full"
        />
        {selectedNode && (
          <GraphInfoPanel
            nodeId={selectedNode.id}
            data={selectedNode.data}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>
    </PageContainer>
  )
}
