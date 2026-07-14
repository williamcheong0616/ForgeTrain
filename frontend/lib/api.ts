import axios from "axios";
import type { Job, ModelEntry, Dataset, HFSearchResult, SystemStats } from "@/types";

const http = axios.create({ baseURL: "/api" });

// Jobs
export const getJobs = () => http.get<Job[]>("/jobs").then((r) => r.data);
export const getJob = (id: number) => http.get<Job>(`/jobs/${id}`).then((r) => r.data);
export const createJob = (body: {
  name: string;
  training_method: string;
  peft_method: string;
  model_id?: number;
  dataset_id?: number;
  config: Record<string, unknown>;
}) => http.post<Job>("/jobs", body).then((r) => r.data);
export const cancelJob = (id: number) => http.delete(`/jobs/${id}`);
export const updateJobRemarks = (id: number, remarks: string) =>
  http.patch<Job>(`/jobs/${id}/remarks`, { remarks }).then((r) => r.data);
export const purgeJob = (id: number) => http.delete(`/jobs/${id}/purge`);
export const getJobMetrics = (id: number) =>
  http.get<import("@/types").TrainingMetric[]>(`/jobs/${id}/metrics/all`).then((r) => r.data);

// Models
export const getModels = () => http.get<ModelEntry[]>("/models").then((r) => r.data);
export const registerModel = (body: { name: string; hf_repo: string; architecture?: string; template?: string; version?: string }) =>
  http.post<ModelEntry>("/models", body).then((r) => r.data);
export const downloadModel = (id: number) => http.post(`/models/${id}/download`);
export const getModelDownloadStatus = (id: number) =>
  http.get<{ is_downloaded: boolean; local_path: string | null; downloaded_at: string | null }>(`/models/${id}/download-status`).then((r) => r.data);
export const deleteModel = (id: number) => http.delete(`/models/${id}`);
export const searchHub = (q: string) =>
  http.get<HFSearchResult[]>("/models/search/hub", { params: { q } }).then((r) => r.data);

// Datasets
export const getDatasets = () => http.get<Dataset[]>("/datasets").then((r) => r.data);
export const uploadDataset = (form: FormData) =>
  http.post<Dataset>("/datasets", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const deleteDataset = (id: number) => http.delete(`/datasets/${id}`);
export const previewDataset = (id: number) =>
  http.get<{ format: string; total: number | null; samples: Record<string, unknown>[]; valid_targets: string[]; conversion_notes: Record<string, string> }>(`/datasets/${id}/preview`).then((r) => r.data);
export const convertDataset = (id: number, body: { target_format: string; template_name?: string; output_name?: string }) =>
  http.post<Dataset>(`/datasets/${id}/convert`, body).then((r) => r.data);
export const previewASRDataset = (id: number) =>
  http.get<{ total: number | null; columns: string[]; samples: Record<string, string>[] }>(`/asr/datasets/${id}/preview`).then((r) => r.data);

// Exports
export const exportJob = (jobId: number, output_name?: string) =>
  http.post(`/exports/${jobId}`, { output_name }).then((r) => r.data);
export const exportFromPath = (body: { adapter_path: string; output_name?: string }) =>
  http.post("/exports/from-path", body).then((r) => r.data);
export const getExports = () => http.get<{ name: string; path: string; size_mb: number; created_at: string }[]>("/exports").then((r) => r.data);

// Eval
export const startEval = (body: Record<string, unknown>) =>
  http.post<{ run_id: string }>("/eval/run", body).then((r) => r.data);
export const getEvalResult = (runId: string) =>
  http.get<{ status: string; loss?: number; perplexity?: number; output_file?: string; error?: string }>(`/eval/${runId}/result`).then((r) => r.data);

// Chat
export const loadChatModel = (body: { model_path: string; adapter_path?: string; quantization?: string }) =>
  http.post("/chat/load", body).then((r) => r.data);
export const getChatStatus = () =>
  http.get<{ status: string; model_path: string | null; adapter_path: string | null; error: string | null }>("/chat/status").then((r) => r.data);
export const unloadChatModel = () => http.post("/chat/unload").then((r) => r.data);

// Sandbox
export const transcribeAudio = (form: FormData) =>
  http.post<{ transcript: string; model_path: string }>("/sandbox/asr/transcribe", form, {
    headers: { "Content-Type": "multipart/form-data" },
  }).then((r) => r.data);

// System
export const getSystemStats = () => http.get<SystemStats>("/system").then((r) => r.data);

// Conversations
export interface ConversationSummary {
  id: number; title: string; model_path: string | null;
  created_at: string; updated_at: string; message_count: number;
}
export interface ConversationMessage { id: number; role: string; content: string; created_at: string; }
export interface ConversationDetail extends ConversationSummary {
  adapter_path: string | null; system_prompt: string | null; gen_params: Record<string, unknown> | null; messages: ConversationMessage[];
}
export const getConversations = () => http.get<ConversationSummary[]>("/conversations").then((r) => r.data);
export const getConversation = (id: number) => http.get<ConversationDetail>(`/conversations/${id}`).then((r) => r.data);
export const createConversation = (body: { title?: string; model_path?: string; adapter_path?: string; system_prompt?: string; gen_params?: Record<string, unknown> }) =>
  http.post<ConversationSummary>("/conversations", body).then((r) => r.data);
export const updateConversation = (id: number, body: { title?: string; system_prompt?: string; gen_params?: Record<string, unknown> }) =>
  http.patch<ConversationSummary>(`/conversations/${id}`, body).then((r) => r.data);
export const deleteConversation = (id: number) => http.delete(`/conversations/${id}`);
export const addConversationMessage = (id: number, role: string, content: string) =>
  http.post<ConversationMessage>(`/conversations/${id}/messages`, { role, content }).then((r) => r.data);

// Prompt profiles
export interface PromptProfile { id: number; name: string; system_prompt: string | null; gen_params: Record<string, unknown> | null; created_at: string; }
export const getPromptProfiles = () => http.get<PromptProfile[]>("/prompt-profiles").then((r) => r.data);
export const createPromptProfile = (body: { name: string; system_prompt?: string; gen_params?: Record<string, unknown> }) =>
  http.post<PromptProfile>("/prompt-profiles", body).then((r) => r.data);
export const deletePromptProfile = (id: number) => http.delete(`/prompt-profiles/${id}`);

// ASR
export const getASRModels = () => http.get<{ id: string; params: string }[]>("/asr/models").then((r) => r.data);
export const getASRDatasets = () => http.get<Dataset[]>("/asr/datasets").then((r) => r.data);
export const uploadASRDataset = (form: FormData) =>
  http.post<Dataset>("/asr/datasets", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const deleteASRDataset = (id: number) => http.delete(`/asr/datasets/${id}`);
export const uploadASRZip = (form: FormData) =>
  http.post<Dataset>("/asr/datasets/zip", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const getASRJobs = () => http.get<Job[]>("/asr/jobs").then((r) => r.data);
export const getASRJob = (id: number) => http.get<Job>(`/asr/jobs/${id}`).then((r) => r.data);
export const createASRJob = (body: {
  name: string;
  peft_method: string;
  dataset_id?: number;
  val_dataset_id?: number;
  config: Record<string, unknown>;
}) => http.post<Job>("/asr/jobs", body).then((r) => r.data);
export const cancelASRJob = (id: number) => http.delete(`/asr/jobs/${id}`);

// Settings
export interface ProviderSettings { base_url: string; api_key: string; configured: boolean; }
export const getProviderSettings = () => http.get<ProviderSettings>("/settings/provider").then((r) => r.data);
export const updateProviderSettings = (body: { base_url?: string; api_key?: string }) =>
  http.put<ProviderSettings>("/settings/provider", body).then((r) => r.data);
export const generateProviderKey = () =>
  http.post<ProviderSettings>("/settings/provider/generate-key").then((r) => r.data);
