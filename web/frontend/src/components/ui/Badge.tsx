import * as React from 'react'
import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'outline' | 'dot' | 'success' | 'error' | 'info'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant
  color?: string // hex colour
}

// Status variants carry light and dark colours so they stay readable in both themes.
const STATUS_CLASSES: Partial<Record<BadgeVariant, string>> = {
  success:
    'border bg-green-100 text-green-800 border-green-300 dark:bg-green-500/20 dark:text-green-300 dark:border-green-500/30',
  error:
    'border bg-red-100 text-red-800 border-red-300 dark:bg-red-500/20 dark:text-red-300 dark:border-red-500/30',
  info:
    'border bg-brand-100 text-brand-800 border-brand-300 dark:bg-brand-600/20 dark:text-brand-300 dark:border-brand-500/30',
}

export function Badge({ className, variant = 'default', color, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium',
        variant === 'default' && 'bg-brand-600/30 text-brand-300 border border-brand-500/30',
        variant === 'outline' && 'border border-white/20 text-gray-300',
        variant === 'dot' && 'bg-transparent text-gray-300',
        STATUS_CLASSES[variant],
        className,
      )}
      style={color ? { backgroundColor: `${color}22`, color, borderColor: `${color}44`, border: '1px solid' } : undefined}
      {...props}
    >
      {variant === 'dot' && (
        <span
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={color ? { backgroundColor: color } : { backgroundColor: 'currentColor' }}
        />
      )}
      {children}
    </span>
  )
}
