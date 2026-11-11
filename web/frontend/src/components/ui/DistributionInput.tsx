import { useMemo } from 'react'
import type { DistributionSchema, IntDistribution, ParamSchema } from '@/api/types'
import { Input } from './Input'
import { Select } from './Select'

interface DistributionInputProps {
  value: IntDistribution
  onChange: (value: IntDistribution) => void
  schemas: DistributionSchema[]
  className?: string
}

function parseParamValue(param: ParamSchema, raw: string): number {
  if (param.type === 'int') return Number.parseInt(raw || '0', 10)
  return Number.parseFloat(raw || '0')
}

/** Empty input means "no bound"; otherwise a non-negative integer. */
function parseBound(raw: string): number | undefined {
  if (raw.trim() === '') return undefined
  const n = Number.parseInt(raw, 10)
  return Number.isNaN(n) ? undefined : Math.max(0, n)
}

export function DistributionInput({ value, onChange, schemas, className }: DistributionInputProps) {
  const schema = useMemo(
    () => schemas.find((s) => s.type === value.type),
    [schemas, value.type],
  )

  const typeOptions = useMemo(
    () => schemas.map((s) => ({ value: s.type, label: s.label })),
    [schemas],
  )

  const setBound = (key: 'min' | 'max', raw: string) => {
    const bound = parseBound(raw)
    const next: IntDistribution = { ...value }
    if (bound === undefined) delete next[key]
    else next[key] = bound
    onChange(next)
  }

  const boundsInvalid = value.min !== undefined && value.max !== undefined && value.min > value.max

  return (
    <div className={className}>
      <Select
        label="Distribution"
        value={value.type}
        onValueChange={(type) => {
          const newSchema = schemas.find((s) => s.type === type)
          const params: Record<string, number> = {}
          for (const p of newSchema?.params ?? []) {
            params[p.name] = p.default as number
          }
          // Keep the bounds: they are independent of the distribution shape
          onChange({ ...value, type: type as IntDistribution['type'], params })
        }}
        options={typeOptions}
      />
      {schema?.params.map((param) => (
        <Input
          key={param.name}
          label={param.label}
          type="number"
          step={param.type === 'float' ? '0.1' : '1'}
          min={param.min}
          max={param.max}
          value={String(value.params[param.name] ?? param.default)}
          onChange={(e) =>
            onChange({
              ...value,
              params: { ...value.params, [param.name]: parseParamValue(param, e.target.value) },
            })
          }
          className="mt-1.5"
        />
      ))}
      <div className="grid grid-cols-2 gap-2 mt-1.5">
        <Input
          label="Min (optional)"
          type="number"
          min={0}
          step="1"
          placeholder="default"
          value={value.min !== undefined ? String(value.min) : ''}
          onChange={(e) => setBound('min', e.target.value)}
        />
        <Input
          label="Max (optional)"
          type="number"
          min={0}
          step="1"
          placeholder="none"
          value={value.max !== undefined ? String(value.max) : ''}
          onChange={(e) => setBound('max', e.target.value)}
          error={boundsInvalid ? 'Max must be >= min' : undefined}
        />
      </div>
    </div>
  )
}
