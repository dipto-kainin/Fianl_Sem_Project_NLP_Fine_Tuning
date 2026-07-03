"use client"

import React, { useRef, useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { gsap } from 'gsap'
import { toast } from 'sonner'
import {
  HardDrives,
  Database,
  FileText,
  Plus,
  CaretDown,
  CaretRight,
  PencilSimple,
  Trash,
  Check,
  X,
  Warning,
} from '@phosphor-icons/react'
import { datasetsApi } from '@/lib/api'
import { Card, Badge, Skeleton, EmptyState, Button } from './ui'

function formatDate(s?: string) {
  if (!s) return '—'
  return new Date(s).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

/* ─── Priority Badge ─────────────────────────────────────────────── */
const PRIORITY_COLORS: Record<number, { bg: string; text: string; label: string }> = {
  3: { bg: 'rgba(239,68,68,0.15)', text: '#f87171', label: 'High' },
  2: { bg: 'rgba(234,179,8,0.15)',  text: '#facc15', label: 'Med'  },
  1: { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8', label: 'Low' },
}

function PriorityBadge({ priority }: { priority: number }) {
  const p = PRIORITY_COLORS[priority] ?? PRIORITY_COLORS[1]
  return (
    <span style={{
      background: p.bg,
      color: p.text,
      fontSize: 10,
      fontWeight: 600,
      padding: '2px 7px',
      borderRadius: 99,
      letterSpacing: '0.03em',
      whiteSpace: 'nowrap',
      flexShrink: 0,
    }}>
      {p.label}
    </span>
  )
}

/* ─── Sample Editor Row ──────────────────────────────────────────── */
function SampleRow({ sample, datasetId, onDeleted }: {
  sample: any
  datasetId: string
  onDeleted: (id: string) => void
}) {
  const qc = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [instruction, setInstruction] = useState(sample.instruction)
  const [response, setResponse] = useState(sample.response)
  const [priority, setPriority] = useState<number>(sample.priority ?? 1)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const saveMut = useMutation({
    mutationFn: () => datasetsApi.updateSample(datasetId, sample.id, { instruction, response, priority }),
    onSuccess: () => {
      toast.success('Sample saved')
      qc.invalidateQueries({ queryKey: ['dataset-samples', datasetId] })
      setEditing(false)
    },
    onError: (e: any) => toast.error(e.message),
  })

  const deleteMut = useMutation({
    mutationFn: () => datasetsApi.deleteSample(datasetId, sample.id),
    onSuccess: () => {
      toast.success('Sample deleted')
      qc.invalidateQueries({ queryKey: ['datasets'] })
      onDeleted(sample.id)
    },
    onError: (e: any) => toast.error(e.message),
  })

  const cancelEdit = () => {
    setInstruction(sample.instruction)
    setResponse(sample.response)
    setPriority(sample.priority ?? 1)
    setEditing(false)
    setConfirmDelete(false)
  }

  const inputBase: React.CSSProperties = {
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 6,
    color: '#e2e8f0',
    fontSize: 12,
    lineHeight: 1.55,
    padding: '7px 10px',
    resize: 'vertical' as const,
    outline: 'none',
    fontFamily: 'inherit',
    width: '100%',
    boxSizing: 'border-box' as const,
  }

  return (
    <div style={{
      borderBottom: '1px solid rgba(255,255,255,0.06)',
      padding: editing ? '12px 16px' : '10px 16px',
      transition: 'background 120ms',
      background: editing ? 'rgba(99,102,241,0.04)' : 'transparent',
    }}>
      {editing ? (
        /* ── Edit mode ── */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* Priority selector */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--ink-muted)', minWidth: 54 }}>Priority</span>
            {[1, 2, 3].map(p => {
              const c = PRIORITY_COLORS[p]
              const selected = priority === p
              return (
                <button
                  key={p}
                  id={`priority-btn-${sample.id}-${p}`}
                  onClick={() => setPriority(p)}
                  style={{
                    background: selected ? c.bg : 'transparent',
                    border: `1px solid ${selected ? c.text : 'rgba(255,255,255,0.12)'}`,
                    color: selected ? c.text : 'var(--ink-muted)',
                    borderRadius: 99,
                    fontSize: 11,
                    fontWeight: 600,
                    padding: '3px 10px',
                    cursor: 'pointer',
                    transition: 'all 150ms',
                  }}
                >
                  {c.label}
                </button>
              )
            })}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 11, color: 'var(--ink-muted)' }}>Instruction</label>
            <textarea
              value={instruction}
              onChange={e => setInstruction(e.target.value)}
              rows={2}
              spellCheck={false}
              style={inputBase}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 11, color: 'var(--ink-muted)' }}>Response</label>
            <textarea
              value={response}
              onChange={e => setResponse(e.target.value)}
              rows={3}
              spellCheck={false}
              style={inputBase}
            />
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <Button
              size="sm"
              onClick={() => saveMut.mutate()}
              loading={saveMut.isPending}
              style={{ gap: 5 }}
            >
              {!saveMut.isPending && <Check size={12} weight="bold" />}
              Save
            </Button>
            <Button size="sm" variant="ghost" onClick={cancelEdit} style={{ gap: 5 }}>
              <X size={12} weight="bold" />
              Cancel
            </Button>
            {confirmDelete ? (
              <>
                <span style={{ fontSize: 12, color: '#f87171', alignSelf: 'center', marginLeft: 8 }}>Confirm delete?</span>
                <Button
                  size="sm"
                  onClick={() => deleteMut.mutate()}
                  loading={deleteMut.isPending}
                  style={{ background: 'rgba(239,68,68,0.15)', color: '#f87171', gap: 5, marginLeft: 4 }}
                >
                  {!deleteMut.isPending && <Trash size={12} />}
                  Yes, delete
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(false)} style={{ gap: 5 }}>No</Button>
              </>
            ) : (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setConfirmDelete(true)}
                style={{ marginLeft: 'auto', color: '#f87171', gap: 5 }}
              >
                <Trash size={12} />
                Delete
              </Button>
            )}
          </div>
        </div>
      ) : (
        /* ── View mode ── */
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
          <PriorityBadge priority={sample.priority ?? 1} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ fontSize: 12, color: 'var(--ink)', fontWeight: 500, marginBottom: 2, lineHeight: 1.4 }}>
              {sample.instruction}
            </p>
            <p style={{
              fontSize: 11,
              color: 'var(--ink-muted)',
              lineHeight: 1.5,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical' as any,
            }}>
              {sample.response}
            </p>
          </div>
          <button
            id={`edit-sample-${sample.id}`}
            onClick={() => setEditing(true)}
            title="Edit sample"
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--ink-muted)',
              padding: 4,
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              flexShrink: 0,
              opacity: 0.5,
              transition: 'opacity 150ms',
            }}
            onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
            onMouseLeave={e => (e.currentTarget.style.opacity = '0.5')}
          >
            <PencilSimple size={13} />
          </button>
        </div>
      )}
    </div>
  )
}

/* ─── Dataset Row ────────────────────────────────────────────────── */
function DatasetRow({ dataset, rowRef }: { dataset: any; rowRef: (el: HTMLDivElement | null) => void }) {
  const [expanded, setExpanded] = useState(false)
  const [deletedIds, setDeletedIds] = useState<Set<string>>(new Set())

  const { data: samplesData, isLoading: samplesLoading } = useQuery({
    queryKey: ['dataset-samples', dataset.id],
    queryFn: () => datasetsApi.getSamples(dataset.id, { limit: 200 }),
    enabled: expanded,
  })

  const samples: any[] = (samplesData?.samples ?? []).filter((s: any) => !deletedIds.has(s.id))

  const priorityCounts = samples.reduce((acc: Record<number, number>, s: any) => {
    const p = s.priority ?? 1
    acc[p] = (acc[p] ?? 0) + 1
    return acc
  }, {})

  return (
    <div ref={rowRef}>
      {/* Header row */}
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '12px 16px',
          borderBottom: '1px solid var(--border)',
          cursor: 'pointer',
          transition: 'background 150ms',
          userSelect: 'none',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface-2)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
      >
        <div style={{ color: 'var(--ink-muted)', display: 'flex', alignItems: 'center' }}>
          {expanded ? <CaretDown size={13} /> : <CaretRight size={13} />}
        </div>
        <div style={{ width: 32, height: 32, borderRadius: 'var(--r-md)', background: 'var(--surface-2)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Database size={14} style={{ color: 'var(--ink-muted)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>
            {dataset.version || `Dataset ${dataset.id.slice(0, 8)}`}
          </p>
          <div style={{ display: 'flex', gap: 8, marginTop: 2, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 11, color: 'var(--ink-faint)' }}>{formatDate(dataset.created_at)}</span>
            {dataset.description && (
              <span style={{ fontSize: 11, color: 'var(--ink-faint)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>
                {dataset.description}
              </span>
            )}
            {expanded && samples.length > 0 && (
              <div style={{ display: 'flex', gap: 5 }}>
                {[3, 2, 1].map(p => priorityCounts[p] ? (
                  <span key={p} style={{ fontSize: 10, color: PRIORITY_COLORS[p].text, background: PRIORITY_COLORS[p].bg, borderRadius: 99, padding: '1px 6px', fontWeight: 600 }}>
                    P{p}: {priorityCounts[p]}
                  </span>
                ) : null)}
              </div>
            )}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {dataset.total_samples !== undefined && (
            <Badge variant="accent">
              <FileText size={11} />
              {dataset.total_samples - deletedIds.size} samples
            </Badge>
          )}
        </div>
      </div>

      {/* Samples panel */}
      {expanded && (
        <div style={{ borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.18)' }}>
          {samplesLoading ? (
            <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[...Array(4)].map((_, i) => <Skeleton key={i} style={{ height: 44 }} />)}
            </div>
          ) : samples.length === 0 ? (
            <div style={{ padding: '20px 16px', textAlign: 'center', color: 'var(--ink-faint)', fontSize: 12 }}>
              No samples available.
            </div>
          ) : (
            <>
              <div style={{ padding: '8px 16px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Warning size={13} style={{ color: 'var(--ink-muted)' }} />
                <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
                  Click the pencil icon on any sample to edit it. Changes are saved to the JSONL immediately.
                </span>
              </div>
              {samples.map((s: any) => (
                <SampleRow
                  key={s.id}
                  sample={s}
                  datasetId={dataset.id}
                  onDeleted={(id) => setDeletedIds(prev => new Set([...prev, id]))}
                />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Datasets page ──────────────────────────────────────────────── */
export function DatasetsPage() {
  const qc = useQueryClient()
  const headerRef = useRef<HTMLDivElement>(null)
  const rowRefs = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    if (!headerRef.current) return
    gsap.fromTo(headerRef.current, { opacity: 0, y: -10 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power3.out' })
  }, [])

  const { data, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list({ limit: 100 }),
  })

  const generateMut = useMutation({
    mutationFn: () => datasetsApi.generate(),
    onSuccess: (res: any) => {
      toast.success(`Dataset version ${res.version || ''} generated successfully!`)
      qc.invalidateQueries({ queryKey: ['datasets'] })
    },
    onError: (e: any) => {
      toast.error(e.message)
    }
  })

  const datasets = data?.datasets ?? []

  useEffect(() => {
    if (!datasets.length) return
    gsap.fromTo(
      rowRefs.current.filter(Boolean) as HTMLDivElement[],
      { opacity: 0, y: 8 },
      { opacity: 1, y: 0, duration: 0.35, stagger: 0.05, ease: 'power2.out' }
    )
  }, [datasets.length])

  return (
    <div style={{ padding: '32px 36px', maxWidth: 900 }}>
      <div
        ref={headerRef}
        style={{
          marginBottom: 28,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          flexWrap: 'wrap',
          gap: 16
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <HardDrives size={20} style={{ color: 'var(--primary)' }} weight="fill" />
            <h1 style={{ fontSize: 20, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.02em' }}>Datasets</h1>
            <Badge variant="default">{datasets.length ?? 0}</Badge>
          </div>
          <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
            Generated training datasets. Click a dataset to expand samples and edit them before training.
          </p>
          {/* Priority legend */}
          <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
            {[3, 2, 1].map(p => (
              <span key={p} style={{
                fontSize: 10,
                color: PRIORITY_COLORS[p].text,
                background: PRIORITY_COLORS[p].bg,
                borderRadius: 99,
                padding: '2px 9px',
                fontWeight: 600,
              }}>
                {PRIORITY_COLORS[p].label}
              </span>
            ))}
            <span style={{ fontSize: 10, color: 'var(--ink-faint)', alignSelf: 'center' }}>
              — training weight per epoch (High=3×, Med=2×, Low=1×)
            </span>
          </div>
        </div>

        <Button
          onClick={() => generateMut.mutate()}
          loading={generateMut.isPending}
          size="sm"
        >
          {!generateMut.isPending && <Plus size={13} weight="bold" />}
          Generate Dataset
        </Button>
      </div>

      <Card className="animate-fade-up">
        {isLoading ? (
          <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[...Array(4)].map((_, i) => <Skeleton key={i} style={{ height: 60 }} />)}
          </div>
        ) : datasets.length === 0 ? (
          <EmptyState icon={HardDrives} title="No datasets yet" description="Process documents with the Teacher LLM to generate training datasets." />
        ) : (
          datasets.map((d: any, i: number) => (
            <DatasetRow key={d.id} dataset={d} rowRef={(el) => { rowRefs.current[i] = el }} />
          ))
        )}
      </Card>
    </div>
  )
}
