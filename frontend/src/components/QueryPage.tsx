"use client"

import React, { useState, useRef, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { gsap } from 'gsap'
import {
  ChatCircleDots,
  PaperPlaneTilt,
  Brain,
  Robot,
  BookOpen,
  CaretDown,
  CaretUp,
  Sparkle,
  ArrowsClockwise,
} from '@phosphor-icons/react'
import { ragApi, documentsApi, registryApi, RagSource, RagResponse } from '@/lib/api'
import { Button, Skeleton, Badge, Select, Checkbox } from './ui'

/* ─── Answer bubble ──────────────────────────────────────────────── */
interface AnswerBubbleProps {
  answer: string
  label: string
  variant?: 'teacher' | 'default' | 'student'
  sources?: RagSource[]
  useRag?: boolean
}

function AnswerBubble({ answer, label, variant = 'teacher', sources, useRag }: AnswerBubbleProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [showSources, setShowSources] = useState(false)

  useEffect(() => {
    if (!ref.current) return
    gsap.fromTo(
      ref.current,
      { opacity: 0, y: 12, scale: 0.98 },
      { opacity: 1, y: 0, scale: 1, duration: 0.4, ease: 'power3.out' }
    )
  }, [])

  const colors = {
    teacher: {
      bg: 'oklch(0.52 0.18 256 / 0.04)',
      border: 'oklch(0.52 0.18 256 / 0.20)',
      icon: 'var(--primary)',
    },
    default: {
      bg: 'oklch(0.22 0.012 250 / 0.04)',
      border: 'oklch(0.22 0.012 250 / 0.12)',
      icon: 'var(--ink-muted)',
    },
    student: {
      bg: 'oklch(0.60 0.16 195 / 0.04)',
      border: 'oklch(0.60 0.16 195 / 0.20)',
      icon: 'var(--accent)',
    },
  }
  const c = colors[variant] || colors.student
  const Icon = variant === 'teacher' ? Brain : Robot
  const showRagLabel = useRag && sources && sources.length > 0

  return (
    <div
      ref={ref}
      style={{
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 'var(--r-lg)',
        padding: '16px 18px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}
    >
      {/* Label row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: 'var(--r-md)',
            background: c.border,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Icon size={15} style={{ color: c.icon }} weight="fill" />
        </div>
        <span style={{ fontSize: 12, fontWeight: 600, color: c.icon, letterSpacing: '0.03em', textTransform: 'uppercase', marginRight: 'auto' }}>
          {label}
        </span>

        {/* Source indicator */}
        <span style={{
          fontSize: 10,
          fontFamily: 'var(--font-sans)',
          fontWeight: 500,
          color: showRagLabel ? 'var(--primary)' : 'var(--ink-muted)',
          background: showRagLabel ? 'var(--primary-glow)' : 'var(--surface-2)',
          padding: '2px 8px',
          borderRadius: 'var(--r-pill)',
          letterSpacing: '0.02em'
        }}>
          {showRagLabel ? 'RAG Retrieval' : 'Internal Weights (Training)'}
        </span>
      </div>

      {/* Answer text */}
      <p style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
        {answer}
      </p>

      {/* Sources */}
      {showRagLabel && sources && sources.length > 0 && (
        <div>
          <button
            onClick={() => setShowSources(!showSources)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 11,
              color: 'var(--ink-muted)',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px 0',
              fontFamily: 'var(--font-sans)',
              transition: 'color 150ms',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--ink)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--ink-muted)' }}
          >
            <BookOpen size={12} />
            {sources.length} source{sources.length !== 1 ? 's' : ''}
            {showSources ? <CaretUp size={10} /> : <CaretDown size={10} />}
          </button>
          {showSources && (
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {sources.map((s, i) => {
                const score = s.score ?? s.similarity_score;
                return (
                  <div
                    key={i}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--r-md)',
                      background: 'var(--surface)',
                      border: '1px solid var(--border)',
                      fontSize: 12,
                      color: 'var(--ink-muted)',
                      lineHeight: 1.5,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                        {s.document_id ? `doc:${s.document_id.slice(0, 8)}…` : 'chunk'}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 11 }}>
                        {score !== undefined ? `${(score * 100).toFixed(0)}%` : ''}
                      </span>
                    </div>
                    <p style={{ margin: 0 }}>{s.text || s.content}</p>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Message pair (query + answers) ────────────────────────────── */
interface MessagePairProps {
  query: string
  result: RagResponse
  model: string
  baseModelName: string
}

function MessagePair({ query, result, model, baseModelName }: MessagePairProps) {
  const showTeacher = model === 'teacher' || model === 'compare'
  const showDefault = model === 'default' || model === 'compare'
  const showStudent = model === 'student' || model === 'compare'
  const isRagUsed = result.used_rag

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* User query bubble */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div
          style={{
            maxWidth: '70%',
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--r-lg)',
            padding: '10px 14px',
            fontSize: 14,
            color: 'var(--ink)',
            lineHeight: 1.6,
          }}
        >
          {query}
        </div>
      </div>
      {/* Answers */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {showTeacher && result.answer && (
          <AnswerBubble
            answer={result.answer}
            label="Teacher · Gemini"
            variant="teacher"
            sources={result.sources}
            useRag={isRagUsed}
          />
        )}
        {showDefault && result.default_answer && (
          <AnswerBubble
            answer={result.default_answer}
            label={`Default Model · ${baseModelName}`}
            variant="default"
            useRag={isRagUsed}
          />
        )}
        {showStudent && result.student_answer && (
          <AnswerBubble
            answer={result.student_answer}
            label={`Fine-tuned Model · ${result.student_version || 'v1'}`}
            variant="student"
            useRag={isRagUsed}
          />
        )}
      </div>
    </div>
  )
}

/* ─── Query page ─────────────────────────────────────────────────── */
interface QueryHistoryItem {
  query: string
  result: RagResponse
  model: string
}

export function QueryPage() {
  const qc = useQueryClient()
  const [input, setInput] = useState('')
  const [selectedModel, setSelectedModel] = useState('compare')
  const [selectedDoc, setSelectedDoc] = useState('')
  const [useRag, setUseRag] = useState(false)
  const [history, setHistory] = useState<QueryHistoryItem[]>([])
  const [selectedMessageIndex, setSelectedMessageIndex] = useState<number | null>(null)
  const [selectedVersion, setSelectedVersion] = useState('')
  
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const headerRef = useRef<HTMLDivElement>(null)

  // Animate header in
  useEffect(() => {
    if (!headerRef.current) return
    gsap.fromTo(
      headerRef.current,
      { opacity: 0, y: -12 },
      { opacity: 1, y: 0, duration: 0.4, ease: 'power3.out' }
    )
  }, [])

  const { data: docsData } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsApi.list({ limit: 100, status: 'processed' }),
  })

  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => registryApi.list({ limit: 100 }),
  })

  const { data: baseModelData } = useQuery({
    queryKey: ['base-model'],
    queryFn: () => registryApi.getBaseModel(),
  })

  const models = modelsData?.models ?? []
  const baseModelName = baseModelData?.base_model?.split('/').pop() || 'Base Model'

  // Update selected version state to the active model when model list loads
  useEffect(() => {
    if (models.length > 0 && !selectedVersion) {
      const active = models.find((m) => m.is_active)
      if (active) {
        setSelectedVersion(active.id)
      }
    }
  }, [models, selectedVersion])

  const activateMut = useMutation({
    mutationFn: (id: string) => registryApi.activate(id),
    onSuccess: (data) => {
      toast.success(`Activated model version: ${data.version}. Context switched and weights reloaded!`)
      qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (err: any) => {
      toast.error(`Failed to activate model: ${err.message}`)
    },
  })

  const handleVersionChange = (newVal: string) => {
    setSelectedVersion(newVal)
    if (newVal) {
      activateMut.mutate(newVal)
    }
  }

  const reloadMut = useMutation({
    mutationFn: () => ragApi.reloadModel(),
    onSuccess: () => toast.success('Model cache cleared — next query will load fresh weights'),
    onError: (err: any) => toast.error(`Reload failed: ${err.message}`),
  })

  const queryMut = useMutation({
    mutationFn: (payload: any) => ragApi.query(payload),
    onSuccess: (result, variables) => {
      setHistory((h) => {
        const next = [...h, { query: variables.query, result, model: variables.model }]
        setSelectedMessageIndex(next.length - 1)
        return next
      })
      setInput('')
      setTimeout(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
      }, 100)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || queryMut.isPending) return
    
    // Force RAG to true if a specific document filter is selected
    const activeRag = selectedDoc !== '' || useRag

    queryMut.mutate({
      query: input.trim(),
      top_k: 5,
      document_id: selectedDoc || null,
      use_rag: activeRag,
      model: selectedModel,
    })
  }

  const docs = docsData?.documents ?? []



  // Extract sources of active/selected query
  const selectedHistoryItem = selectedMessageIndex !== null ? history[selectedMessageIndex] : history[history.length - 1]
  const activeSources = selectedHistoryItem?.result?.sources ?? []

  return (
    <div style={{ display: 'flex', height: '100dvh', maxWidth: 1440, margin: '0 auto', width: '100%' }}>
      {/* Left side: Main Chat Area */}
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, padding: '0 24px', height: '100dvh' }}>
        {/* Header */}
        <div ref={headerRef} style={{ padding: '28px 0 16px', borderBottom: '1px solid var(--border)', marginBottom: 20, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <ChatCircleDots size={20} style={{ color: 'var(--primary)' }} weight="fill" />
            <h1 style={{ fontSize: 20, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.02em' }}>
              Query
            </h1>
            <div style={{ marginLeft: 'auto' }}>
              <button
                id="reload-model-btn"
                onClick={() => reloadMut.mutate()}
                disabled={reloadMut.isPending}
                title="Flush cached model and reload latest trained weights"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 12px',
                  borderRadius: 'var(--r-md)',
                  border: '1px solid var(--border)',
                  background: 'var(--surface-2)',
                  color: reloadMut.isPending ? 'var(--ink-faint)' : 'var(--ink-muted)',
                  fontSize: 12,
                  fontWeight: 500,
                  fontFamily: 'var(--font-sans)',
                  cursor: reloadMut.isPending ? 'wait' : 'pointer',
                  transition: 'all 150ms var(--ease-out)',
                }}
                onMouseEnter={(e) => { if (!reloadMut.isPending) { e.currentTarget.style.borderColor = 'var(--primary)'; e.currentTarget.style.color = 'var(--primary)' } }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--ink-muted)' }}
              >
                <ArrowsClockwise
                  size={14}
                  weight="bold"
                  style={reloadMut.isPending ? { animation: 'spin 0.8s linear infinite' } : undefined}
                />
                {reloadMut.isPending ? 'Reloading…' : 'Reload Model'}
              </button>
            </div>
          </div>
          <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
            Ask questions over your documents. Compare Teacher (Gemini) and Student model answers.
          </p>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, paddingBottom: 20 }}>
          {history.length === 0 && !queryMut.isPending && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 16, paddingTop: 60 }}>
              <div style={{ width: 52, height: 52, borderRadius: 'var(--r-lg)', background: 'var(--primary-glow)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Sparkle size={24} style={{ color: 'var(--primary)' }} weight="fill" />
              </div>
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: 15, fontWeight: 500, color: 'var(--ink)', marginBottom: 6 }}>Ready to answer</p>
                <p style={{ fontSize: 13, color: 'var(--ink-muted)' }}>
                  Ask anything about your uploaded documents
                </p>
              </div>
              {/* Quick suggestions */}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', marginTop: 4 }}>
                {['Summarize key points', 'What are the main findings?', 'Explain the methodology'].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    style={{
                      padding: '6px 12px',
                      borderRadius: 'var(--r-pill)',
                      border: '1px solid var(--border)',
                      background: 'var(--surface)',
                      color: 'var(--ink-muted)',
                      fontSize: 12,
                      cursor: 'pointer',
                      fontFamily: 'var(--font-sans)',
                      transition: 'all 150ms var(--ease-out)',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--primary)'; e.currentTarget.style.color = 'var(--primary)' }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--ink-muted)' }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {history.map((h, i) => {
            const isSelected = selectedMessageIndex === i || (selectedMessageIndex === null && i === history.length - 1);
            return (
              <div
                key={i}
                onClick={() => setSelectedMessageIndex(i)}
                style={{
                  cursor: 'pointer',
                  padding: '12px 14px',
                  borderRadius: 'var(--r-lg)',
                  border: '1px solid transparent',
                  borderColor: isSelected ? 'oklch(from var(--primary) l c h / 0.15)' : 'transparent',
                  background: isSelected ? 'oklch(from var(--primary) l c h / 0.02)' : 'transparent',
                  transition: 'all 200ms ease',
                }}
              >
                <MessagePair query={h.query} result={h.result} model={h.model} baseModelName={baseModelName} />
              </div>
            )
          })}

          {queryMut.isPending && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '12px 14px' }}>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{ maxWidth: '70%', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '10px 14px', fontSize: 14, color: 'var(--ink)' }}>
                  {queryMut.variables?.query}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '12px 0', color: 'var(--ink-muted)', fontSize: 13 }}>
                <Brain size={15} style={{ color: 'var(--primary)' }} className="animate-[spin_1.5s_linear_infinite]" />
                Generating answer…
              </div>
              <Skeleton style={{ height: 100 }} />
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, paddingBottom: 20, flexShrink: 0 }}>
          {/* Options row */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            {/* Document filter */}
            <Select
              value={selectedDoc}
              onChange={setSelectedDoc}
              options={[
                { value: '', label: 'All documents' },
                ...docs.map((d) => ({ value: d.id, label: d.filename })),
              ]}
              style={{ flex: '1 1 160px', maxWidth: 240 }}
            />

            {/* Model selection dropdown */}
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              options={[
                { value: 'compare', label: 'Compare All Models' },
                { value: 'teacher', label: 'Teacher (Gemini)' },
                { value: 'default', label: `Default Model (${baseModelName})` },
                { value: 'student', label: 'Fine-Tuned Local Models' },
              ]}
              style={{ flex: '1 1 160px', maxWidth: 240 }}
            />

            {/* Model version dropdown */}
            {(selectedModel === 'student' || selectedModel === 'compare') && models.length > 0 && (
              <Select
                value={selectedVersion}
                onChange={handleVersionChange}
                options={models.map((m) => ({
                  value: m.id,
                  label: `${m.version}${m.is_active ? ' (Active)' : ''}`,
                }))}
                style={{ flex: '1 1 160px', maxWidth: 240 }}
              />
            )}

            {/* RAG Toggle checkbox (only shown if no document is selected) */}
            {selectedDoc === '' && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none', marginLeft: 4 }}>
                <Checkbox
                  checked={useRag}
                  onCheckedChange={(checked) => setUseRag(!!checked)}
                />
                <span style={{ fontSize: 13, color: 'var(--ink-muted)' }}>Use RAG</span>
              </label>
            )}
          </div>

          {/* Text input */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 10 }}>
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents…"
              disabled={queryMut.isPending}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              style={{
                flex: 1,
                height: 42,
                padding: '0 14px',
                borderRadius: 'var(--r-md)',
                background: 'var(--surface-2)',
                border: '1px solid var(--border)',
                color: 'var(--ink)',
                fontSize: 14,
                fontFamily: 'var(--font-sans)',
                outline: 'none',
                transition: 'border-color 150ms',
              }}
              onFocus={(e) => {
                if (e.target) {
                  e.target.style.borderColor = 'var(--primary)'
                  e.target.style.boxShadow = '0 0 0 3px var(--primary-glow)'
                }
              }}
              onBlur={(e) => {
                if (e.target) {
                  e.target.style.borderColor = 'var(--border)'
                  e.target.style.boxShadow = 'none'
                }
              }}
            />
            <Button
              type="submit"
              disabled={!input.trim()}
              loading={queryMut.isPending}
              style={{ width: 42, height: 42, padding: 0, borderRadius: 'var(--r-md)' }}
            >
              {queryMut.isPending ? null : <PaperPlaneTilt size={16} weight="fill" />}
            </Button>
          </form>
        </div>
      </div>

      {/* Right Column: RAG Passages Sidebar */}
      <div
        className="hidden lg:flex"
        style={{
          width: 380,
          background: 'var(--surface)',
          borderLeft: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          overflowY: 'auto',
          height: '100dvh',
          position: 'sticky',
          top: 0,
        }}
      >
        <div style={{ padding: '24px 20px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={16} style={{ color: 'var(--primary)' }} weight="bold" />
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.01em' }}>Citations & Passages</h2>
        </div>

        <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 14, flex: 1 }}>
          {activeSources.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 12, opacity: 0.6, padding: '40px 0' }}>
              <BookOpen size={32} style={{ color: 'var(--ink-faint)' }} />
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>No citations active</p>
                <p style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4, maxWidth: '24ch', marginInline: 'auto' }}>
                  Submit a query with RAG active or select a query in history to view sources.
                </p>
              </div>
            </div>
          ) : (
            activeSources.map((s, i) => {
              const score = s.score ?? s.similarity_score;
              return (
                <div
                  key={i}
                  style={{
                    padding: 14,
                    borderRadius: 'var(--r-md)',
                    background: 'var(--surface-2)',
                    border: '1px solid var(--border)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                      {s.filename || `Document ${s.document_id?.slice(0, 8)}…`}
                    </span>
                    <Badge variant="accent">
                      {score !== undefined ? `${(score * 100).toFixed(0)}% match` : 'RAG'}
                    </Badge>
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--ink)', lineHeight: 1.5, margin: 0, whiteSpace: 'pre-wrap' }}>
                    {s.text || s.content}
                  </p>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
