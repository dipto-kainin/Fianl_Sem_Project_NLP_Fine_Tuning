"use client"

import React, { useRef, useCallback, useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { gsap } from 'gsap'
import { toast } from 'sonner'
import {
  Files,
  UploadSimple,
  Trash,
  Play,
  CloudArrowUp,
} from '@phosphor-icons/react'
import { documentsApi, DocumentItem } from '@/lib/api'
import {
  Button,
  Card,
  Skeleton,
  StatusBadge,
  EmptyState,
  Badge,
} from './ui'

function formatBytes(b?: number) {
  if (!b) return '—'
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 ** 2).toFixed(1)} MB`
}

function formatDate(s?: string) {
  if (!s) return '—'
  return new Date(s).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
  })
}

/* ─── Drop zone ──────────────────────────────────────────────────── */
interface DropZoneProps {
  onFile: (file: File) => void
}

function DropZone({ onFile }: DropZoneProps) {
  const [drag, setDrag] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const zoneRef = useRef<HTMLDivElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    if (zoneRef.current) {
      gsap.to(zoneRef.current, { scale: 1, duration: 0.2, ease: 'power2.out' })
    }
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    if (!drag) {
      setDrag(true)
      if (zoneRef.current) {
        gsap.to(zoneRef.current, { scale: 1.01, duration: 0.2, ease: 'power2.out' })
      }
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    if (zoneRef.current) {
      gsap.to(zoneRef.current, { scale: 1, duration: 0.2, ease: 'power2.out' })
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  return (
    <div
      ref={zoneRef}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `1.5px dashed ${drag ? 'var(--primary)' : 'var(--border)'}`,
        borderRadius: 'var(--r-lg)',
        padding: '32px 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        cursor: 'pointer',
        background: drag ? 'var(--primary-glow)' : 'transparent',
        transition: 'all 200ms var(--ease-out)',
        userSelect: 'none',
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: 'var(--r-lg)',
          background: drag ? 'var(--primary-glow)' : 'var(--surface-2)',
          border: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 200ms var(--ease-out)',
        }}
      >
        <CloudArrowUp size={22} style={{ color: drag ? 'var(--primary)' : 'var(--ink-faint)' }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>
          Drop document or click to browse
        </p>
        <p style={{ fontSize: 12, color: 'var(--ink-muted)', marginTop: 4 }}>
          PDF, DOCX, TXT, Markdown, HTML, EPUB
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.html,.epub"
        style={{ display: 'none' }}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          if (e.target.files?.[0]) {
            onFile(e.target.files[0])
          }
          e.target.value = ''
        }}
      />
    </div>
  )
}

/* ─── Document row ───────────────────────────────────────────────── */
interface DocRowProps {
  doc: DocumentItem
  onProcess: (id: string) => void
  onDelete: (id: string) => void
  animRef: (el: HTMLDivElement | null) => void
  isLast?: boolean
}

function DocRow({ doc, onProcess, onDelete, animRef, isLast }: DocRowProps) {
  return (
    <div
      ref={animRef}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        padding: '13px 18px',
        borderBottom: isLast ? 'none' : '1px solid var(--border)',
        transition: 'background 150ms',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--surface-2)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      {/* File icon */}
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 'var(--r-md)',
          background: 'var(--surface-2)',
          border: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <Files size={16} style={{ color: 'var(--ink-muted)' }} />
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {doc.filename}
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 3 }}>
          <span style={{ fontSize: 11, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
            {formatBytes(doc.size_bytes)}
          </span>
          <span style={{ color: 'var(--border)', fontSize: 11 }}>·</span>
          <span style={{ fontSize: 11, color: 'var(--ink-faint)' }}>
            {formatDate(doc.created_at)}
          </span>
        </div>
      </div>

      <StatusBadge status={doc.status} />

      {/* Thin vertical rule separates info from actions */}
      <div style={{ width: 1, height: 18, background: 'var(--border)', flexShrink: 0 }} />

      {/* Actions — 8px gap, delete is icon-only with no border */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        {(doc.status === 'uploaded' || doc.status === 'pending' || doc.status === 'failed') && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onProcess(doc.id)}
            title="Process document"
          >
            <Play size={12} weight="fill" />
            Process
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onDelete(doc.id)}
          title="Delete"
          style={{ color: 'var(--error)', borderColor: 'transparent' }}
        >
          <Trash size={13} />
        </Button>
      </div>
    </div>
  )
}

/* ─── Documents page ─────────────────────────────────────────────── */
export function DocumentsPage() {
  const qc = useQueryClient()
  const listRef = useRef<HTMLDivElement>(null)
  const rowRefs = useRef<(HTMLDivElement | null)[]>([])

  const { data, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsApi.list({ limit: 100 }),
    refetchInterval: 5000, // poll for status updates
  })

  const uploadMut = useMutation({
    mutationFn: documentsApi.upload,
    onSuccess: (doc) => {
      toast.success(`"${doc.filename}" uploaded`)
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const processMut = useMutation({
    mutationFn: (id: string) => documentsApi.process(id),
    onSuccess: () => {
      toast.success('Processing started')
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  const deleteMut = useMutation({
    mutationFn: documentsApi.delete,
    onSuccess: () => {
      toast.success('Document deleted')
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (e: any) => toast.error(e.message),
  })

  // Animate rows in when data loads
  const prevCount = useRef(0)

  useEffect(() => {
    if (data?.documents) {
      const docs = data.documents
      if (docs.length > prevCount.current) {
        requestAnimationFrame(() => {
          const newRefs = rowRefs.current.slice(prevCount.current)
          if (newRefs.length) {
            gsap.fromTo(
              newRefs.filter(Boolean) as HTMLDivElement[],
              { opacity: 0, y: 8 },
              { opacity: 1, y: 0, duration: 0.35, stagger: 0.04, ease: 'power2.out' }
            )
          }
        })
        prevCount.current = docs.length
      }
    }
  }, [data])

  const docs = data?.documents ?? []

  return (
    <div style={{ padding: '32px 36px', maxWidth: 900 }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }} className="animate-fade-up">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <Files size={20} style={{ color: 'var(--primary)' }} weight="fill" />
          <h1 style={{ fontSize: 20, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.02em' }}>
            Documents
          </h1>
          <Badge variant="default">{data?.total ?? 0}</Badge>
        </div>
        <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
          Upload documents to extract structured knowledge and power RAG queries.
        </p>
      </div>

      {/* Drop zone */}
      <div className="animate-fade-up" style={{ animationDelay: '50ms', marginBottom: 24 }}>
        <DropZone onFile={(file) => uploadMut.mutate(file)} />
        {uploadMut.isPending && (
          <p style={{ marginTop: 8, fontSize: 12, color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <UploadSimple size={13} className="animate-[spin_0.7s_linear_infinite]" />
            Uploading…
          </p>
        )}
      </div>

      {/* Documents list */}
      <Card style={{ animationDelay: '100ms' }} className="animate-fade-up p-0 gap-0">
        {isLoading ? (
          <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} style={{ height: 60 }} />
            ))}
          </div>
        ) : docs.length === 0 ? (
          <EmptyState
            icon={Files}
            title="No documents yet"
            description="Drop a PDF, DOCX, or text file above to begin building your knowledge base."
          />
        ) : (
          <div ref={listRef}>
            {docs.map((doc, i) => (
              <DocRow
                key={doc.id}
                doc={doc}
                animRef={(el) => { rowRefs.current[i] = el }}
                onProcess={(id) => processMut.mutate(id)}
                onDelete={(id) => deleteMut.mutate(id)}
                isLast={i === docs.length - 1}
              />
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
