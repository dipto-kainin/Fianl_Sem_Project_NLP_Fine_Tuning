"use client"

import React, { useState, useEffect, useRef } from 'react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { gsap } from 'gsap'
import {
  Files,
  ChatCircleDots,
  GraduationCap,
  HardDrives,
  Brain,
  Pulse,
  CaretRight,
} from '@phosphor-icons/react'
import { healthApi } from '@/lib/api'
import { DocumentsPage } from '@/components/DocumentsPage'
import { QueryPage } from '@/components/QueryPage'
import { TrainingPage } from '@/components/TrainingPage'
import { DatasetsPage } from '@/components/DatasetsPage'

const queryClient = new QueryClient()

const NAV = [
  { id: 'documents', label: 'Documents',  icon: Files,          desc: 'Upload & process' },
  { id: 'query',     label: 'Query',      icon: ChatCircleDots, desc: 'RAG & inference' },
  { id: 'training',  label: 'Training',   icon: GraduationCap,  desc: 'Fine-tune Student' },
  { id: 'datasets',  label: 'Datasets',   icon: HardDrives,     desc: 'Training datasets' },
]

const PAGES: Record<string, React.ComponentType> = {
  documents: DocumentsPage,
  query: QueryPage,
  training: TrainingPage,
  datasets: DatasetsPage,
}

interface SidebarProps {
  page: string
  setPage: (p: string) => void
  health: boolean
}

function Sidebar({ page, setPage, health }: SidebarProps) {
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([])
  const logoRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Logo animation
      gsap.fromTo(logoRef.current,
        { opacity: 0, y: -10 },
        { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out', delay: 0.05 }
      )
      // Nav items stagger
      gsap.fromTo(
        itemRefs.current.filter(Boolean),
        { opacity: 0, x: -14 },
        { opacity: 1, x: 0, duration: 0.4, stagger: 0.07, ease: 'power3.out', delay: 0.15 }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <aside
      style={{
        width: 220,
        minWidth: 220,
        background: 'var(--surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        position: 'sticky',
        top: 0,
        height: '100vh',
        overflowY: 'auto',
        zIndex: 10,
      }}
    >
      {/* Logo */}
      <div ref={logoRef} style={{ padding: '20px 16px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 'var(--r-md)',
              background: 'linear-gradient(135deg, var(--primary), var(--primary-dim))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 12px var(--primary-glow)',
              flexShrink: 0,
            }}
          >
            <Brain size={17} color="white" weight="fill" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: '-0.03em', color: 'var(--ink)' }}>
              DistillAI
            </div>
            <div style={{ fontSize: 10, color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
              knowledge distillation
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '10px 8px' }} aria-label="Main navigation">
        <div style={{
          padding: '4px 8px 6px',
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.12em',
          color: 'var(--ink-faint)',
          textTransform: 'uppercase',
          marginTop: 6,
        }}>
          Pipeline
        </div>
        {NAV.map((item, i) => {
          const Icon = item.icon
          const active = page === item.id
          return (
            <button
              key={item.id}
              ref={(el) => { itemRefs.current[i] = el }}
              id={`nav-${item.id}`}
              onClick={() => setPage(item.id)}
              aria-current={active ? 'page' : undefined}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                width: '100%',
                padding: '8px 10px',
                borderRadius: 'var(--r-md)',
                border: 'none',
                cursor: 'pointer',
                marginBottom: 2,
                background: active ? 'var(--primary-glow)' : 'transparent',
                color: active ? 'var(--primary)' : 'var(--ink-muted)',
                fontSize: 13,
                fontWeight: active ? 600 : 400,
                fontFamily: 'var(--font-sans)',
                transition: 'background 150ms var(--ease-out), color 150ms var(--ease-out)',
                textAlign: 'left',
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  e.currentTarget.style.background = 'var(--surface-2)'
                  e.currentTarget.style.color = 'var(--ink)'
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = 'var(--ink-muted)'
                }
              }}
            >
              <Icon
                size={15}
                weight={active ? 'fill' : 'regular'}
                style={{ flexShrink: 0, opacity: active ? 1 : 0.7 }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div>{item.label}</div>
                <div style={{ fontSize: 11, color: active ? 'var(--primary)' : 'var(--ink-faint)', marginTop: 1, opacity: active ? 0.8 : 1 }}>
                  {item.desc}
                </div>
              </div>
              {active && (
                <CaretRight size={11} weight="bold" style={{ opacity: 0.5, flexShrink: 0 }} />
              )}
            </button>
          )
        })}
      </nav>

      {/* Footer — health */}
      <div style={{ padding: '12px 14px', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Pulse size={13} style={{ color: health ? 'var(--success)' : 'var(--error)', flexShrink: 0 }} />
        <span style={{ fontSize: 11, color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)', flex: 1 }}>
          {health ? 'API online' : 'API offline'}
        </span>
        <span className={`status-dot ${health ? 'active' : 'error'}`} />
      </div>
    </aside>
  )
}

interface PageTransitionProps {
  children: React.ReactNode
  pageKey: string
}

function PageTransition({ children, pageKey }: PageTransitionProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    gsap.fromTo(
      ref.current,
      { opacity: 0, y: 10 },
      { opacity: 1, y: 0, duration: 0.3, ease: 'power3.out' }
    )
  }, [pageKey])

  return (
    <div ref={ref} style={{ flex: 1, minWidth: 0, overflowY: 'auto', minHeight: '100vh' }}>
      {children}
    </div>
  )
}

function MainLayout() {
  const [page, setPage] = useState('documents')

  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    refetchInterval: 30_000,
    retry: false,
  })
  const health = healthData?.status === 'healthy'

  const PageComponent = PAGES[page] ?? DocumentsPage

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg)' }}>
      <Sidebar page={page} setPage={setPage} health={health} />
      <PageTransition pageKey={page}>
        <PageComponent />
      </PageTransition>
    </div>
  )
}

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <MainLayout />
      <Toaster position="top-right" richColors />
    </QueryClientProvider>
  )
}
