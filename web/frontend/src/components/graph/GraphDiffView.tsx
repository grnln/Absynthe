import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, Info } from 'lucide-react'
import { GraphCanvas } from './GraphCanvas'
import { Badge } from '@/components/ui/Badge'
import { getLabelColor } from '@/lib/color-map'
import type { CytoscapeElement, ChangedNode, EdgePerturbInfo } from '@/api/types'

interface GraphDiffViewProps {
  originalElements: CytoscapeElement[]
  perturbedElements: CytoscapeElement[]
  removedNodes: string[]
  changedNodes: ChangedNode[]
  edgePerturbInfo?: Record<string, EdgePerturbInfo>
}

function edgeKey(source: string, target: string): string {
  return source < target ? `${source}|${target}` : `${target}|${source}`
}

function edgeKeyFromElement(el: CytoscapeElement): string | null {
  if (el.group !== 'edges') return null
  const source = String(el.data.source ?? '')
  const target = String(el.data.target ?? '')
  if (!source || !target) return null
  return edgeKey(source, target)
}

function addClass(existing: string | undefined, extra: string): string {
  const classes = new Set((existing ?? '').split(' ').filter(Boolean))
  classes.add(extra)
  return Array.from(classes).join(' ')
}

export function GraphDiffView({
  originalElements,
  perturbedElements,
  removedNodes,
  changedNodes,
  edgePerturbInfo = {},
}: GraphDiffViewProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [sharedPositions, setSharedPositions] = useState<Record<string, { x: number; y: number }> | null>(null)

  // Build class maps for both sides
  const originalClasses: Record<string, string> = {}
  const perturbedClasses: Record<string, string> = {}

  for (const nid of removedNodes) {
    originalClasses[nid] = 'removed'
  }
  for (const cn of changedNodes) {
    perturbedClasses[cn.node_id] = 'changed'
  }

  // Collect all edge changes
  const addedEdges = Object.values(edgePerturbInfo).flatMap((i) => i.added_edges)
  const removedEdges = Object.values(edgePerturbInfo).flatMap((i) => i.removed_edges)

  // Mark changed edges in place. Only inject ghost edges if we can't match one.
  const removedEdgeKeys = new Set(removedEdges.map((e) => edgeKey(e.source, e.target)))
  const addedEdgeKeys = new Set(addedEdges.map((e) => edgeKey(e.source, e.target)))

  const seenRemoved = new Set<string>()
  const seenAdded = new Set<string>()

  const originalWithDiff: CytoscapeElement[] = originalElements.map((el) => {
    const key = edgeKeyFromElement(el)
    if (key && removedEdgeKeys.has(key)) {
      seenRemoved.add(key)
      return { ...el, classes: addClass(el.classes, 'removed') }
    }
    return el
  })

  const perturbedWithDiff: CytoscapeElement[] = perturbedElements.map((el) => {
    const key = edgeKeyFromElement(el)
    if (key && addedEdgeKeys.has(key)) {
      seenAdded.add(key)
      return { ...el, classes: addClass(el.classes, 'added') }
    }
    return el
  })

  const missingRemovedGhosts = removedEdges
    .filter((e) => !seenRemoved.has(edgeKey(e.source, e.target)))
    .map((e) => ({
      group: 'edges' as const,
      data: { id: `ghost-rm-${e.source}-${e.target}`, source: e.source, target: e.target },
      classes: 'removed',
    }))

  const missingAddedGhosts = addedEdges
    .filter((e) => !seenAdded.has(edgeKey(e.source, e.target)))
    .map((e) => ({
      group: 'edges' as const,
      data: { id: `ghost-add-${e.source}-${e.target}`, source: e.source, target: e.target },
      classes: 'added',
    }))

  const originalWithGhosts: CytoscapeElement[] = [
    ...originalWithDiff,
    ...missingRemovedGhosts,
  ]
  const perturbedWithGhosts: CytoscapeElement[] = [
    ...perturbedWithDiff,
    ...missingAddedGhosts,
  ]

  const originalNodeKey = useMemo(
    () =>
      originalWithGhosts
        .filter((el) => el.group === 'nodes')
        .map((el) => el.data.id)
        .sort()
        .join(','),
    [originalWithGhosts],
  )

  useEffect(() => {
    setSharedPositions(null)
  }, [originalNodeKey])

  // The graph row keeps a minimum height so a long label-change table cannot
  // collapse it to 0px; the table scrolls internally and the whole view scrolls
  // as a last resort instead of clipping.
  return (
    <div className="flex flex-col gap-4 h-full min-h-0 overflow-y-auto">
      {/* Summary badges */}
      {(changedNodes.length > 0 || removedNodes.length > 0) && (
        <div className="flex flex-wrap gap-2 items-center">
          {removedNodes.length > 0 && (
            <Badge variant="outline" className="text-gray-400">
              {removedNodes.length} node{removedNodes.length !== 1 ? 's' : ''} removed
            </Badge>
          )}
          {changedNodes.length > 0 && (
            <Badge className="bg-orange-50 text-orange-700 border-orange-200">
              {changedNodes.length} label change{changedNodes.length !== 1 ? 's' : ''}
            </Badge>
          )}
          {addedEdges.length > 0 && (
            <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
              +{addedEdges.length} edge{addedEdges.length !== 1 ? 's' : ''}
            </Badge>
          )}
          {removedEdges.length > 0 && (
            <Badge className="bg-red-500/20 text-red-300 border-red-500/30">
              −{removedEdges.length} edge{removedEdges.length !== 1 ? 's' : ''}
            </Badge>
          )}
        </div>
      )}

      {/* Side-by-side graphs */}
      <div className="flex gap-4 flex-1 min-h-[360px]">
        <div className="flex flex-col flex-1 min-w-0">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Before</p>
          <div className="flex-1 rounded-xl border border-white/10 overflow-hidden">
            <GraphCanvas
              elements={originalWithGhosts}
              extraClasses={originalClasses}
              onNodeClick={(id) => setSelectedNode(id === selectedNode ? null : id)}
              defaultLayout="cose-bilkent"
              toolbar={false}
              className="h-full"
              fixedPositions={sharedPositions ?? undefined}
              onPositionsReady={(positions) => {
                if (!sharedPositions) setSharedPositions(positions)
              }}
            />
          </div>
        </div>

        <div className="flex items-center shrink-0">
          <ArrowRight className="w-5 h-5 text-gray-600" />
        </div>

        <div className="flex flex-col flex-1 min-w-0">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">After</p>
          <div className="flex-1 rounded-xl border border-white/10 overflow-hidden">
            {sharedPositions ? (
              <GraphCanvas
                elements={perturbedWithGhosts}
                extraClasses={perturbedClasses}
                onNodeClick={(id) => setSelectedNode(id === selectedNode ? null : id)}
                defaultLayout="cose-bilkent"
                toolbar={false}
                className="h-full"
                fixedPositions={sharedPositions}
                adjustNewNodes
              />
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-gray-500">
                Synchronizing layout...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Changed nodes table */}
      {changedNodes.length > 0 && (
        <div className="shrink-0">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            <Info className="inline w-3 h-3 mr-1" />
            Label changes
          </p>
          <div className="rounded-lg border border-white/10 overflow-hidden">
            <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10 bg-white dark:bg-gray-900">
                <tr className="border-b border-white/10 bg-white/5">
                  <th className="text-left px-3 py-2 text-gray-500 font-medium">Node</th>
                  <th className="text-left px-3 py-2 text-gray-500 font-medium">Before</th>
                  <th className="text-left px-3 py-2 text-gray-500 font-medium">After</th>
                </tr>
              </thead>
              <tbody>
                {changedNodes.map((cn) => {
                  const oldC = getLabelColor(cn.old_label)
                  const newC = getLabelColor(cn.new_label)
                  const isSelected = selectedNode === cn.node_id
                  return (
                    <tr
                      key={cn.node_id}
                      onClick={() => setSelectedNode(cn.node_id === selectedNode ? null : cn.node_id)}
                      className={`border-b border-white/5 cursor-pointer transition-colors ${
                        isSelected ? 'bg-orange-500/10' : 'hover:bg-white/5'
                      }`}
                    >
                      <td className="px-3 py-2 font-mono text-gray-300">{cn.node_id}</td>
                      <td className="px-3 py-2">
                        <Badge variant="dot" color={oldC.text}>{cn.old_label}</Badge>
                      </td>
                      <td className="px-3 py-2">
                        <Badge variant="dot" color={newC.text}>{cn.new_label}</Badge>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
