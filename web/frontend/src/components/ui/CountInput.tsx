import { Dice5 } from 'lucide-react'
import type { DistributionSchema, IntDistribution } from '@/api/types'
import { DistributionInput } from './DistributionInput'
import { Slider } from './Slider'

interface CountInputProps {
  label: string
  /** Fixed value, used when no distribution is set */
  value: number
  /** When present the count is sampled from this distribution instead */
  distribution?: IntDistribution
  onValueChange: (value: number) => void
  onDistributionChange: (distribution: IntDistribution | undefined) => void
  schemas: DistributionSchema[]
  min?: number
  max?: number
  className?: string
}

/** Default distribution built from the first schema the backend advertises. */
export function defaultDistribution(schemas: DistributionSchema[]): IntDistribution {
  const schema = schemas[0]
  const params: Record<string, number> = {}
  for (const p of schema?.params ?? []) {
    params[p.name] = p.default as number
  }
  return { type: (schema?.type ?? 'uniform') as IntDistribution['type'], params }
}

/**
 * A count that is either a fixed number (slider) or a random distribution
 * sampled per graph. The dice button toggles between the two modes, matching
 * the motif `count` / `count_distribution` controls.
 */
export function CountInput({
  label,
  value,
  distribution,
  onValueChange,
  onDistributionChange,
  schemas,
  min = 0,
  max = 20,
  className,
}: CountInputProps) {
  const toggle = () => {
    onDistributionChange(distribution ? undefined : defaultDistribution(schemas))
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-2 mb-1">
        <label className="text-xs font-medium text-gray-400">{label}</label>
        <button
          type="button"
          onClick={toggle}
          className={`ml-auto p-1 rounded transition-colors ${distribution ? 'text-brand-400 bg-brand-500/20' : 'text-gray-600 hover:text-gray-400'}`}
          title={distribution ? 'Switch to fixed count' : 'Sample from a random distribution (per graph)'}
        >
          <Dice5 className="w-3.5 h-3.5" />
        </button>
      </div>
      {distribution ? (
        <DistributionInput value={distribution} onChange={onDistributionChange} schemas={schemas} />
      ) : (
        <Slider value={value} min={min} max={max} onValueChange={onValueChange} />
      )}
    </div>
  )
}
