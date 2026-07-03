"use client"

import React, { forwardRef, useState, useRef, useEffect } from 'react'
import { CircleNotch, CaretDown } from '@phosphor-icons/react'
import { cn } from '@/lib/utils'

// Import official shadcn components
import { Button as ShadcnButton } from '@/components/ui/button'
import { Card as ShadcnCard } from '@/components/ui/card'
import { Input as ShadcnInput } from '@/components/ui/input'
import { Badge as ShadcnBadge } from '@/components/ui/badge'
import { Skeleton as ShadcnSkeleton } from '@/components/ui/skeleton'
import { Checkbox as ShadcnCheckbox } from '@/components/ui/checkbox'

/**
 * Button — wrapped shadcn component to support legacy interface (loading, primary/danger variants)
 */
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'danger' | 'ghost' | 'outline' | 'secondary' | 'link'
  size?: 'default' | 'xs' | 'sm' | 'lg' | 'icon' | 'icon-xs' | 'icon-sm' | 'icon-lg'
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    children,
    variant = 'primary',
    size = 'default',
    loading = false,
    disabled = false,
    className,
    style,
    ...props
  },
  ref
) {
  // Map legacy variants to shadcn variants
  let shadcnVariant: 'default' | 'destructive' | 'ghost' | 'outline' | 'secondary' | 'link' = 'default'
  if (variant === 'primary') shadcnVariant = 'default'
  else if (variant === 'danger') shadcnVariant = 'destructive'
  else if (variant === 'ghost') shadcnVariant = 'ghost'
  else if (variant === 'outline') shadcnVariant = 'outline'
  else if (variant === 'secondary') shadcnVariant = 'secondary'
  else if (variant === 'link') shadcnVariant = 'link'

  return (
    <ShadcnButton
      ref={ref}
      disabled={disabled || loading}
      variant={shadcnVariant}
      size={size}
      style={style}
      className={cn("transition-all active:scale-[0.97] select-none", className)}
      {...props}
    >
      {loading && (
        <CircleNotch
          size={14}
          className={cn("animate-spin shrink-0", children ? "mr-1.5" : "")}
        />
      )}
      {children}
    </ShadcnButton>
  )
})

/**
 * Badge — wrapped shadcn component supporting legacy variants
 */
interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'accent' | 'secondary' | 'outline'
}

export function Badge({ children, variant = 'default', className, ...props }: BadgeProps) {
  let v: 'default' | 'secondary' | 'outline' | 'destructive' = 'secondary'
  if (variant === 'primary') v = 'default'
  else if (variant === 'error') v = 'destructive'
  else if (variant === 'secondary') v = 'secondary'
  else if (variant === 'outline') v = 'outline'

  const customClasses = cn(
    variant === 'success' && 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100 dark:bg-emerald-950/20 dark:text-emerald-400 dark:border-emerald-900/50',
    variant === 'warning' && 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100 dark:bg-amber-950/20 dark:text-amber-400 dark:border-amber-900/50',
    variant === 'accent' && 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100 dark:bg-indigo-950/20 dark:text-indigo-400 dark:border-indigo-900/50',
    className
  )

  return (
    <ShadcnBadge variant={v} className={customClasses} {...props}>
      {children}
    </ShadcnBadge>
  )
}

/**
 * Card — wrapped shadcn Card
 */
interface CardProps extends React.HTMLAttributes<HTMLDivElement> { }

export function Card({ children, className, ...props }: CardProps) {
  return (
    <ShadcnCard className={cn("overflow-hidden border-border bg-card", className)} {...props}>
      {children}
    </ShadcnCard>
  )
}

/**
 * Skeleton loader — wrapped shadcn Skeleton
 */
interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> { }

export function Skeleton({ className, ...props }: SkeletonProps) {
  return <ShadcnSkeleton className={cn("w-full rounded-md", className)} {...props} />
}

/**
 * Spinner — standalone loading indicator
 */
interface SpinnerProps {
  size?: number
  className?: string
}

export function Spinner({ size = 16, className }: SpinnerProps) {
  return (
    <CircleNotch
      size={size}
      className={cn("animate-spin text-primary shrink-0", className)}
    />
  )
}

/**
 * Input — wrapped shadcn Input supporting label, error, and hint
 */
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
}

export function Input({ label, error, hint, className, style, ...props }: InputProps) {
  return (
    <div style={style} className="flex flex-col gap-1.5 w-full">
      {label && (
        <label className="text-xs font-medium text-muted-foreground select-none">
          {label}
        </label>
      )}
      <ShadcnInput
        className={cn(error && "border-destructive focus-visible:ring-destructive/20", className)}
        {...props}
      />
      {error && <p className="text-xs text-destructive mt-0.5">{error}</p>}
      {hint && !error && <p className="text-xs text-muted-foreground mt-0.5">{hint}</p>}
    </div>
  )
}

/**
 * Divider
 */
export function Divider() {
  return <hr className="border-t border-border w-full my-4" />
}

/**
 * Status badge with dot
 */
interface StatusBadgeProps {
  status: string
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const map: Record<string, { dot: string; variant: BadgeProps['variant']; label: string }> = {
    pending: { dot: 'idle', variant: 'default', label: 'Pending' },
    processing: { dot: 'running', variant: 'accent', label: 'Processing' },
    processed: { dot: 'active', variant: 'success', label: 'Processed' },
    failed: { dot: 'error', variant: 'error', label: 'Failed' },
    queued: { dot: 'idle', variant: 'default', label: 'Queued' },
    training: { dot: 'running', variant: 'accent', label: 'Training' },
    completed: { dot: 'active', variant: 'success', label: 'Completed' },
    cancelled: { dot: 'error', variant: 'warning', label: 'Cancelled' },
    active: { dot: 'active', variant: 'success', label: 'Active' },
    registered: { dot: 'idle', variant: 'default', label: 'Registered' },
    uploaded: { dot: 'idle', variant: 'default', label: 'Uploaded' },
  }
  const cfg = map[status?.toLowerCase()] ?? { dot: 'idle', variant: 'default', label: status }
  return (
    <Badge variant={cfg.variant}>
      <span className={`status-dot ${cfg.dot}`} />
      {cfg.label}
    </Badge>
  )
}

/**
 * Empty state
 */
interface EmptyStateProps {
  icon?: React.ComponentType<{ size?: number; className?: string }>
  title: string
  description?: string
  action?: React.ReactNode
}

export function EmptyState({ icon: Icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
      {Icon && (
        <div className="w-12 h-12 rounded-xl bg-muted border border-border flex items-center justify-center">
          <Icon size={22} className="text-muted-foreground" />
        </div>
      )}
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium text-foreground">{title}</p>
        {description && (
          <p className="text-xs text-muted-foreground max-w-[40ch]">{description}</p>
        )}
      </div>
      {action}
    </div>
  )
}

/**
 * Premium Dropdown Select (shadcn/ui Select wrapper to prevent clipping)
 */
import {
  Select as ShadcnSelectRoot,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export interface SelectOption {
  value: string
  label: string
}

interface SelectProps {
  value: string
  onChange: (value: string) => void
  options: SelectOption[]
  placeholder?: string
  style?: React.CSSProperties
  className?: string
  width?: string | number
}

export function Select({
  value,
  onChange,
  options,
  placeholder = 'Select option...',
  style,
  className,
  width = '100%',
}: SelectProps) {
  return (
    <div style={{ width: width, ...style }} className={className}>
      <ShadcnSelectRoot value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full h-9 bg-background border border-border font-sans text-xs">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent position="popper" className="bg-popover border border-border text-popover-foreground z-100 font-sans text-xs">
          {options.map((opt) => (
            <SelectItem key={opt.value} value={opt.value} className='hover:text-white'>
              {opt.label}
            </SelectItem>
          ))}
          {options.length === 0 && (
            <div className="p-2 text-center text-xs text-muted-foreground">
              No options available
            </div>
          )}
        </SelectContent>
      </ShadcnSelectRoot>
    </div>
  )
}

/**
 * Premium Checked Checkbox
 */
interface CheckboxProps extends React.ComponentPropsWithoutRef<typeof ShadcnCheckbox> {}

export function Checkbox({ className, ...props }: CheckboxProps) {
  return (
    <ShadcnCheckbox
      className={cn(
        "cursor-pointer data-checked:bg-[#0057B5] data-checked:border-[#0057B5] focus-visible:ring-[#0057B5]/30 focus-visible:border-[#0057B5] border-input",
        className
      )}
      {...props}
    />
  )
}

