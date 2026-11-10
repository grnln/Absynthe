// Mirrors the Pydantic models on the backend

export interface CytoscapeElementData {
  id: string
  label?: string
  motif?: string
  motif_id?: string
  expected_ground_truth?: string
  observed_ground_truth?: string
  source?: string
  target?: string
  [key: string]: unknown
}

export interface CytoscapeElement {
  group: 'nodes' | 'edges'
  data: CytoscapeElementData
  classes?: string
  position?: { x: number; y: number }
}

export type ParamType = 'int' | 'float' | 'string' | 'select'

export interface ParamSchema {
  name: string
  label: string
  type: ParamType
  default: number | string
  min?: number
  max?: number
  options?: string[]
}

export interface MotifSchema {
  type: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface StrategySchema {
  id: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface CompositionSchema {
  id: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface LabelingFunctionSchema {
  id: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface PerturbationSchema {
  id: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface IntDistribution {
  type: 'uniform' | 'normal' | 'poisson'
  params: Record<string, number>
}

export interface DistributionSchema {
  type: string
  label: string
  description: string
  params: ParamSchema[]
}

export interface MotifConfig {
  type: string
  params: (number | string)[]
  count?: number
  count_distribution?: IntDistribution
}

export interface LabelingFunctionConfig {
  type: string
  params: Record<string, unknown>
}

export interface PerturbationConfig {
  type: string
  params: Record<string, unknown>
  count: number
  /** Sampled per base graph; overrides `count` when present */
  count_distribution?: IntDistribution
}

/** Extra vertices / edges: fixed counts, or distributions sampled per graph */
export interface ExtrasConfig {
  num_extra_vertices?: number
  num_extra_vertices_distribution?: IntDistribution
  num_extra_edges?: number
  num_extra_edges_distribution?: IntDistribution
}

export interface GraphGenerateRequest extends ExtrasConfig {
  motifs: MotifConfig[]
  composition: string
  composition_params: Record<string, unknown>
  num_extra_vertices: number
  num_extra_edges: number
  seed?: number
}

export interface GraphStats {
  num_nodes: number
  num_edges: number
  motif_counts: Record<string, number>
}

export interface GraphGenerateResponse {
  graph_id: string
  elements: CytoscapeElement[]
  stats: GraphStats
}

export interface GraphUploadResponse {
  graph_id: string
  elements: CytoscapeElement[]
  stats: GraphStats
  file_count: number
  folder_path: string | null
  warnings: { filename: string; error: string }[]
}

export interface FolderSourceConfig {
  folder_path: string
  iteration_order: string
  exhaustion_policy: string
}

export interface LabelAssignRequest {
  graph_id: string
  labeling_functions: LabelingFunctionConfig[]
}

export interface LabelDistribution {
  counts: Record<string, number>
}

export interface LabeledGraphResponse {
  graph_id: string
  elements: CytoscapeElement[]
  label_distribution: LabelDistribution
}

export interface PerturbationRequest {
  graph_id: string
  labeling_functions: LabelingFunctionConfig[]
  perturbations: PerturbationConfig[]
  max_iterations: number
  seed?: number
}

export interface ChangedNode {
  node_id: string
  old_label: string
  new_label: string
}

export interface EdgeChange {
  source: string
  target: string
}

export interface EdgePerturbInfo {
  removed_edges: EdgeChange[]
  added_edges: EdgeChange[]
}

export interface PerturbationPreview {
  config_index: number
  perturbation_type: string
  desired_count: number
  success: boolean
  message: string
  original_elements: CytoscapeElement[]
  perturbed_elements: CytoscapeElement[]
  removed_nodes: string[]
  changed_nodes: ChangedNode[]
  edge_perturb_info: Record<string, EdgePerturbInfo>
}

export interface PerturbationResponse {
  original_graph_id: string
  perturbed_graph_id: string
  original_elements: CytoscapeElement[]
  perturbed_elements: CytoscapeElement[]
  removed_nodes: string[]
  changed_nodes: ChangedNode[]
  edge_perturb_info: Record<string, EdgePerturbInfo>
  previews: PerturbationPreview[]
  success: boolean
  message: string
}

export interface DatasetGenerateRequest extends ExtrasConfig {
  num_graphs: number
  motifs?: MotifConfig[]
  folder_source?: FolderSourceConfig
  composition?: string
  composition_params?: Record<string, unknown>
  labeling_functions: LabelingFunctionConfig[]
  perturbations: PerturbationConfig[]
  max_perturbation_iterations: number
  output_dir: string
  seed?: number
}

export interface TaskStatus {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current: number
  total: number
  output_dir?: string
  error?: string
}
