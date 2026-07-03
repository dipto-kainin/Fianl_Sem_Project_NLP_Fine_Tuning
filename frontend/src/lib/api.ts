import axios from 'axios'

export const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
  timeout: 60_000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      'Request failed'
    return Promise.reject(new Error(msg))
  },
)

/* ─── Documents ─────────────────────────────────────────────────── */
export interface DocumentItem {
  id: string
  filename: string
  status: 'pending' | 'processing' | 'processed' | 'failed' | 'uploaded'
  size_bytes?: number
  created_at: string
  processed_at?: string
  error_message?: string
}

export interface DocumentListResponse {
  documents: DocumentItem[]
  total: number
}

export const documentsApi = {
  list: (params?: any): Promise<DocumentListResponse> => api.get('/documents/', { params }).then((r) => r.data),
  get: (id: string): Promise<DocumentItem> => api.get(`/documents/${id}`).then((r) => r.data),
  upload: (file: File): Promise<DocumentItem> => {
    const fd = new FormData()
    fd.append('file', file)
    return api.post('/documents/upload', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },
  process: (id: string, config: any = {}): Promise<DocumentItem> =>
    api.post(`/documents/${id}/process`, config).then((r) => r.data),
  delete: (id: string): Promise<any> => api.delete(`/documents/${id}`),
}

/* ─── RAG ───────────────────────────────────────────────────────── */
export interface RagQueryPayload {
  query: string
  top_k?: number
  document_id?: string | null
  use_rag?: boolean
  model?: string
}

export interface RagSource {
  id?: string
  document_id?: string
  filename?: string
  text?: string
  content?: string
  score?: number
  similarity_score?: number
}

export interface RagResponse {
  answer: string
  sources: RagSource[]
  used_rag: boolean
  model: string
  student_answer?: string
  default_answer?: string
  teacher_answer?: string
  student_version?: string
}

export const ragApi = {
  query: (payload: RagQueryPayload): Promise<RagResponse> => api.post('/rag/query', payload).then((r) => r.data),
  search: (payload: any): Promise<any> => api.post('/rag/search', payload).then((r) => r.data),
  reloadModel: (): Promise<any> => api.post('/rag/reload-model').then((r) => r.data),
}

/* ─── Training ──────────────────────────────────────────────────── */
export interface TrainingJob {
  id: string
  status: 'pending' | 'training' | 'completed' | 'failed' | 'cancelled'
  epochs?: number
  learning_rate?: number
  loss_history?: number[]
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  model_version?: string | null
}

export interface TrainingListResponse {
  runs: TrainingJob[]
  total: number
}

export const trainingApi = {
  list: (params?: any): Promise<TrainingListResponse> => api.get('/training/', { params }).then((r) => r.data),
  get: (id: string): Promise<TrainingJob> => api.get(`/training/${id}`).then((r) => r.data),
  start: (payload: any): Promise<TrainingJob> => api.post('/training/start', payload).then((r) => r.data),
  cancel: (id: string): Promise<TrainingJob> => api.post(`/training/${id}/cancel`).then((r) => r.data),
  retry: (id: string): Promise<TrainingJob> => api.post(`/training/${id}/retry`).then((r) => r.data),
  delete: (id: string): Promise<any> => api.delete(`/training/${id}`).then((r) => r.data),
}

/* ─── Registry / Models ─────────────────────────────────────────── */
export interface ModelVersionResponse {
  id: string
  version: string
  training_run_id?: string | null
  base_model: string
  model_path: string
  quantization_format?: string | null
  file_size?: number | null
  metrics?: Record<string, any> | null
  dataset_version?: string | null
  is_active: boolean
  created_at: string
}

export interface ModelVersionListResponse {
  models: ModelVersionResponse[]
  total: number
}

export const registryApi = {
  list: (params?: any): Promise<ModelVersionListResponse> => api.get('/models/', { params }).then((r) => r.data),
  get: (id: string): Promise<ModelVersionResponse> => api.get(`/models/${id}`).then((r) => r.data),
  getActive: (): Promise<ModelVersionResponse | null> => api.get('/models/active').then((r) => r.data),
  activate: (id: string): Promise<ModelVersionResponse> => api.post(`/models/${id}/activate`).then((r) => r.data),
  getMetrics: (id: string): Promise<any> => api.get(`/models/${id}/metrics`).then((r) => r.data),
  getBaseModel: (): Promise<{ base_model: string }> => api.get('/models/base-model').then((r) => r.data),
}

/* ─── Datasets ──────────────────────────────────────────────────── */
export interface DatasetItem {
  id: string
  version: string
  status: 'pending' | 'processing' | 'processed' | 'failed' | 'completed'
  total_samples?: number
  description?: string | null
  created_at: string
}

export interface DatasetListResponse {
  datasets: DatasetItem[]
  total: number
}

export const datasetsApi = {
  list: (params?: any): Promise<DatasetListResponse> => api.get('/datasets/', { params }).then((r) => r.data),
  generate: (payload: any = {}): Promise<DatasetItem> => api.post('/datasets/generate', payload).then((r) => r.data),
  getSamples: (datasetId: string, params?: any): Promise<any> => api.get(`/datasets/${datasetId}/samples`, { params }).then((r) => r.data),
  updateSample: (datasetId: string, sampleId: string, payload: any): Promise<any> => api.patch(`/datasets/${datasetId}/samples/${sampleId}`, payload).then((r) => r.data),
  deleteSample: (datasetId: string, sampleId: string): Promise<any> => api.delete(`/datasets/${datasetId}/samples/${sampleId}`).then((r) => r.data),
}

/* ─── Knowledge ─────────────────────────────────────────────────── */
export const knowledgeApi = {
  list: (params?: any): Promise<any> => api.get('/knowledge/', { params }).then((r) => r.data),
}

/* ─── Health ─────────────────────────────────────────────────────── */
export const healthApi = {
  check: (): Promise<any> => axios.get('/health').then((r) => r.data),
}
