// Shapes returned by agentscan's API (src/agentscan/api.py).

export const API = process.env.NEXT_PUBLIC_API_BASE ?? "";

export type Tier = "high" | "medium" | "low";
export type Label = "agent" | "llm" | "service";

export interface RiskFactor {
  key: string;
  points: number;
  why: string;
  owasp: string[];
  atlas: string[];
}

export interface Risk {
  tier: Tier;
  score: number;
  factors: RiskFactor[];
}

export interface WorkloadRow {
  id: string;
  name: string;
  type: string;
  label: Label;
  confidence: number;
  basis: string | null;
  risk: Risk | null;
  activity: Record<string, number>;
}

export interface AccessRequest {
  source: string;
  status: string;
  permissions: string[];
  why: string;
  workloads: string[];
}

export interface Overview {
  account_id: string;
  generated_at: string;
  regions: string[];
  summary: Record<Label, number>;
  total: number;
  risk_tiers: Record<Tier, number>;
  owasp: Record<string, number>;
  atlas: Record<string, number>;
  activity: Record<string, number>;
  events_by_source: { source: string; action: string; n: number }[];
  coverage: { granted: string[]; missing: string[]; failed: string[] };
  access_requests: AccessRequest[];
  unclassified: { arn: string; type: string; region: string }[];
  changes: string[];
  workloads: WorkloadRow[];
  scans: { id: string; generated_at: string; summary: Record<Label, number>; grants: string[] }[];
}

export interface Evidence {
  id: string;
  source: string;
  signal: string;
  tier: string;
  detail: string;
  ai_weight: number;
  agentic_weight: number;
}

export interface AuditEvent {
  id: string;
  occurred_at: string;
  source: string;
  action: string;
  target: string;
  actor: string;
  correlation_id: string | null;
  call_id: string | null;
}

export interface WorkloadDetail {
  workload: { id: string; name: string; notes: string[]; primary: { type: string } };
  classification: { category: Label; confidence: number; basis: string | null; rationale: string };
  evidence: Evidence[];
  risk: Risk | null;
  events: AuditEvent[];
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText);
  return r.json();
}
