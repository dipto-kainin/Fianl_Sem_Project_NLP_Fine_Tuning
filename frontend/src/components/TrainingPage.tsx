"use client"

import React, { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { gsap } from 'gsap'
import { toast } from 'sonner'
import {
  GraduationCap,
  Play,
  Stop,
  ChartLine,
  CaretDown,
  Trash,
} from '@phosphor-icons/react'
import { trainingApi, datasetsApi, TrainingJob } from '@/lib/api'
import { Button, Card, Badge, Skeleton, StatusBadge, EmptyState, Input, Select, Checkbox } from './ui'

function formatDuration(start: string, end?: string) {
  const ms = new Date(end || Date.now()).getTime() - new Date(start).getTime()
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

/* ─── Training run card ──────────────────────────────────────────── */
interface RunCardProps {
  run: any // Let API specify standard types or extend
  onCancel: (id: string) => void
  onRetry: (id: string) => void
  onDelete: (id: string) => void
  animRef: (el: HTMLDivElement | null) => void
}

function RunCard({ run, onCancel, onRetry, onDelete, animRef }: RunCardProps) {
  const [expanded, setExpanded] = useState(false)
  const metrics = run.metrics || {}

  return (
    <div
      ref={animRef}
      style={{
        borderBottom: '1px solid var(--border)',
        transition: 'background 150ms',
      }}
    >
      {/* Header row */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '14px 16px',
          cursor: 'pointer',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--surface-2)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        {/* Status */}
        <StatusBadge status={run.status} />

        {/* ID or Model Name */}
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {run.status === 'completed' && run.model_version ? run.model_version : `${run.id.slice(0, 8)}…`}
        </span>

        {/* Base model */}
        <span style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
          {run.base_model?.split('/').pop()}
        </span>

        {/* Duration */}
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-faint)' }}>
          {run.started_at ? formatDuration(run.started_at, run.completed_at) : '—'}
        </span>

        {/* Cancel */}
        {(run.status === 'queued' || run.status === 'training' || run.status === 'processing') && (
          <Button size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); onCancel(run.id) }}>
            <Stop size={12} weight="fill" />
            Stop
          </Button>
        )}

        {/* Retry */}
        {(run.status === 'failed' || run.status === 'cancelled') && (
          <Button size="sm" variant="primary" onClick={(e) => { e.stopPropagation(); onRetry(run.id) }}>
            <Play size={11} weight="fill" />
            Retrain
          </Button>
        )}

        {/* Delete */}
        {(run.status === 'completed' || run.status === 'failed' || run.status === 'cancelled') && (
          <Button
            size="sm"
            variant="ghost"
            style={{ color: 'var(--error)' }}
            onClick={(e) => {
              e.stopPropagation()
              if (window.confirm('Are you sure you want to delete this training run and permanently remove its weights from disk?')) {
                onDelete(run.id)
              }
            }}
          >
            <Trash size={12} />
            Delete
          </Button>
        )}

        <CaretDown
          size={13}
          style={{
            color: 'var(--ink-faint)',
            transition: 'transform 200ms var(--ease-out)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        />
      </div>

      {/* Expanded metrics */}
      {expanded && (
        <div style={{ padding: '0 16px 14px', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {Object.entries(metrics).map(([k, v]) => (
            <div key={k} style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '8px 12px', minWidth: 100 }}>
              <div style={{ fontSize: 10, color: 'var(--ink-faint)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 3 }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--ink)', fontFamily: 'var(--font-mono)' }}>
                {typeof v === 'number' ? v.toFixed(4) : String(v)}
              </div>
            </div>
          ))}
          {Object.keys(metrics).length === 0 && (
            <p style={{ fontSize: 12, color: 'var(--ink-faint)' }}>No metrics available yet.</p>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Start training form ────────────────────────────────────────── */
interface StartTrainingFormProps {
  onSubmit: (params: any) => void
  loading: boolean
}

function StartTrainingForm({ onSubmit, loading }: StartTrainingFormProps) {
  const [form, setForm] = useState({
    dataset_id: '',
    base_model: '',
    model_name: '',
    epochs: '10',
    batch_size: '2',
    learning_rate: '0.0002',
    lora_rank: '8',
    lora_alpha: '16',
    quantize: false,
    use_context: false,
    user_prompt: 'You are an HR professional carefully reading a candidate\'s resume. Your task is to extract all factual information with maximum precision — especially contact details (phone, email, GitHub, LinkedIn, portfolio links), technical skills, project descriptions with their tech stacks and outcomes, and work experience with company names, durations, and measurable achievements.',
  })

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list({ limit: 50 }),
  })

  const set = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }))

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.dataset_id) { toast.error('Select a dataset first'); return }
    onSubmit({
      ...form,
      epochs: Number(form.epochs),
      batch_size: Number(form.batch_size),
      learning_rate: Number(form.learning_rate),
      lora_rank: Number(form.lora_rank),
      lora_alpha: Number(form.lora_alpha),
      model_name: form.model_name.trim() || undefined,
    })
  }

  const ds = datasets?.datasets ?? []

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '20px 20px' }}>
      {/* Dataset */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-muted)' }}>Dataset</label>
        <Select
          value={form.dataset_id}
          onChange={(val) => set('dataset_id', val)}
          options={[
            { value: '', label: 'Select dataset…' },
            ...ds.map((d: any) => ({
              value: d.id,
              label: d.version || `Dataset ${d.id.slice(0, 8)}`,
            }))
          ]}
          placeholder="Select dataset…"
        />
      </div>

      {/* Base Model */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-muted)' }}>Base Model</label>
        <Input
          placeholder="e.g. Qwen/Qwen2.5-3B-Instruct (uses system default if blank)"
          value={form.base_model}
          onChange={(e) => set('base_model', e.target.value)}
        />
      </div>

      {/* Model Version Name */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-muted)' }}>Model Version Name (Optional)</label>
        <Input
          placeholder="e.g. resume-extractor-v1 (uses sequential naming if blank)"
          value={form.model_name}
          onChange={(e) => set('model_name', e.target.value)}
        />
      </div>

      {/* Params grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Input label="Epochs" type="number" min={1} max={100} value={form.epochs} onChange={(e) => set('epochs', e.target.value)} />
        <Input label="Batch size" type="number" min={1} max={64} value={form.batch_size} onChange={(e) => set('batch_size', e.target.value)} />
        <Input label="Learning rate" type="number" step="0.00001" value={form.learning_rate} onChange={(e) => set('learning_rate', e.target.value)} />
        <Input label="LoRA rank" type="number" value={form.lora_rank} onChange={(e) => set('lora_rank', e.target.value)} />
        <Input label="LoRA alpha" type="number" value={form.lora_alpha} onChange={(e) => set('lora_alpha', e.target.value)} />
      </div>

      {/* Quantize toggle */}
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
        <Checkbox
          checked={form.quantize}
          onCheckedChange={(checked) => set('quantize', !!checked)}
        />
        <span style={{ fontSize: 13, color: 'var(--ink-muted)' }}>Quantize output (GGUF)</span>
      </label>

      {/* Use Context toggle */}
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
        <Checkbox
          checked={form.use_context}
          onCheckedChange={(checked) => set('use_context', !!checked)}
        />
        <span style={{ fontSize: 13, color: 'var(--ink-muted)' }}>Include document context (disable for closed-book/RAG-free memorization)</span>
      </label>

      {/* User Prompt */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-muted)' }}>
          Training Persona Prompt
          <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--ink-subtle)', marginLeft: 6 }}>
            (Gemini will amplify this into a precise extraction directive)
          </span>
        </label>
        <textarea
          id="training-user-prompt"
          value={form.user_prompt}
          onChange={(e) => set('user_prompt', e.target.value)}
          rows={4}
          spellCheck={false}
          placeholder="e.g. You are an HR reading a resume. Focus on contact details, skills, and projects."
          style={{
            background: '#ffffff',
            border: '1px solid rgba(0,0,0,0.15)',
            borderRadius: 8,
            color: '#111827',
            fontSize: 13,
            lineHeight: 1.6,
            padding: '10px 12px',
            resize: 'vertical',
            outline: 'none',
            fontFamily: 'inherit',
            width: '100%',
            boxSizing: 'border-box',
            caretColor: '#111827',
          }}
        />
        <span style={{ fontSize: 11, color: 'var(--ink-subtle)' }}>
          ✦ This prompt will be expanded by Gemini to emphasize exact contact info → skills → projects → experience during dataset Q&amp;A generation.
        </span>
      </div>

      <Button type="submit" loading={loading} disabled={!form.dataset_id} style={{ alignSelf: 'flex-start' }}>
        {!loading && <Play size={14} weight="fill" />}
        Start Training
      </Button>
    </form>
  )
}

/* ─── Training page ──────────────────────────────────────────────── */
export function TrainingPage() {
  const qc = useQueryClient()
  const headerRef = useRef<HTMLDivElement>(null)
  const runRefs = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    if (!headerRef.current) return
    gsap.fromTo(headerRef.current, { opacity: 0, y: -10 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power3.out' })
  }, [])

  const { data, isLoading } = useQuery({
    queryKey: ['training'],
    queryFn: () => trainingApi.list({ limit: 50 }),
    refetchInterval: 8000,
  })

  const startMut = useMutation({
    mutationFn: trainingApi.start,
    onSuccess: () => {
      toast.success('Training run started')
      qc.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const cancelMut = useMutation({
    mutationFn: trainingApi.cancel,
    onSuccess: () => {
      toast.success('Training run stopped')
      qc.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const retryMut = useMutation({
    mutationFn: trainingApi.retry,
    onSuccess: () => {
      toast.success('Retraining started successfully')
      qc.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const deleteMut = useMutation({
    mutationFn: trainingApi.delete,
    onSuccess: () => {
      toast.success('Training run and weights deleted successfully')
      qc.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const runs = data?.runs ?? []

  // Animate rows
  useEffect(() => {
    if (!runs.length) return
    gsap.fromTo(
      runRefs.current.filter(Boolean) as HTMLDivElement[],
      { opacity: 0, x: -8 },
      { opacity: 1, x: 0, duration: 0.35, stagger: 0.05, ease: 'power2.out' }
    )
  }, [runs.length])

  return (
    <div style={{ padding: '32px 36px', maxWidth: 900 }}>
      <div ref={headerRef} style={{ marginBottom: 28 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <GraduationCap size={20} style={{ color: 'var(--primary)' }} weight="fill" />
          <h1 style={{ fontSize: 20, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.02em' }}>
            Training
          </h1>
          <Badge variant="default">{runs.length ?? 0}</Badge>
        </div>
        <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
          Fine-tune a Student model with QLoRA on your generated datasets.
        </p>
      </div>

      {/* Start training form */}
      <Card style={{ marginBottom: 24 }} className="animate-fade-up">
        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)' }}>
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink)' }}>New training run</h2>
        </div>
        <StartTrainingForm onSubmit={(p) => startMut.mutate(p)} loading={startMut.isPending} />
      </Card>

      {/* Runs list */}
      <Card className="animate-fade-up" style={{ animationDelay: '60ms' }}>
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
          <ChartLine size={15} style={{ color: 'var(--ink-muted)' }} />
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink)' }}>Runs</h2>
        </div>
        {isLoading ? (
          <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[...Array(3)].map((_, i) => <Skeleton key={i} style={{ height: 52 }} />)}
          </div>
        ) : runs.length === 0 ? (
          <EmptyState icon={GraduationCap} title="No training runs" description="Start a run above to fine-tune your student model." />
        ) : (
          runs.map((run: any, i: number) => (
            <RunCard
              key={run.id}
              run={run}
              animRef={(el) => { runRefs.current[i] = el }}
              onCancel={(id) => cancelMut.mutate(id)}
              onRetry={(id) => retryMut.mutate(id)}
              onDelete={(id) => deleteMut.mutate(id)}
            />
          ))
        )}
      </Card>
    </div>
  )
}
