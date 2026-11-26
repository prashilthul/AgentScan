"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { get, type Overview, type Tier, type WorkloadDetail, type WorkloadRow } from "@/lib/api";

const TIER_STYLE: Record<Tier, string> = {
  high: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
  medium: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
  low: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
};
const LABEL_TEXT = { agent: "Agent", llm: "LLM app", service: "Service" } as const;
const SOURCE_TEXT: Record<string, string> = {
  otel: "OpenTelemetry (collector)",
  cloudtrail: "CloudTrail",
  "bedrock-invocation-log": "Bedrock invocation logs",
  "app-log": "CloudWatch app logs",
};
const SOURCE_SHORT: Record<string, string> = {
  otel: "OTel",
  cloudtrail: "CloudTrail",
  "bedrock-invocation-log": "Invocation log",
  "app-log": "App log",
};
const ACTION_TEXT: Record<string, [string, string]> = {
  model_call: ["model call", "model calls"],
  tool_call: ["tool call", "tool calls"],
  tool_requested: ["tool requested", "tools requested"],
  agent_step: ["agent step", "agent steps"],
  agent_invoked: ["agent invocation", "agent invocations"],
};

function TierBadge({ tier }: { tier?: Tier }) {
  if (!tier) return <span className="text-muted-foreground">n/a</span>;
  return <Badge className={TIER_STYLE[tier]}>{tier}</Badge>;
}

function Stat({ title, value, note }: { title: string; value: React.ReactNode; note: string }) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardDescription>{title}</CardDescription>
        <p className="text-3xl font-semibold tabular-nums">{value}</p>
      </CardHeader>
      <CardContent className="text-muted-foreground">{note}</CardContent>
    </Card>
  );
}

function Bars({ data, label }: { data: Record<string, number>; label: string }) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map(([, n]) => n));
  if (!entries.length) return <p className="text-muted-foreground">Nothing mapped yet.</p>;
  return (
    <ul className="space-y-2" aria-label={label}>
      {entries.map(([name, n]) => (
        <li key={name}>
          <div className="flex justify-between gap-4 text-sm">
            <span>{name}</span>
            <span className="shrink-0 whitespace-nowrap tabular-nums text-muted-foreground">
              {n} workload{n === 1 ? "" : "s"}
            </span>
          </div>
          <div className="mt-1 h-2 rounded bg-muted">
            <div className="h-2 rounded bg-primary/70" style={{ width: `${(n / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

const count = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

function activityText(a: Record<string, number>) {
  const parts = Object.entries(a).map(([k, n]) => `${n} ${ACTION_TEXT[k]?.[n === 1 ? 0 : 1] ?? k}`);
  return parts.length ? parts.join(", ") : "none seen";
}

function Detail({ name, onClose }: { name: string; onClose: () => void }) {
  const [d, setD] = useState<WorkloadDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    get<WorkloadDetail>(`/api/workloads/${encodeURIComponent(name)}`)
      .then(setD)
      .catch((e) => setError(String(e.message ?? e)));
  }, [name]);
  return (
    <Card className="lg:sticky lg:top-6">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-lg">{name}</CardTitle>
            <CardDescription>{d?.workload.primary.type}</CardDescription>
          </div>
          <button
            onClick={onClose}
            className="rounded px-2 py-1 text-sm text-muted-foreground hover:bg-muted"
            aria-label="Close details"
          >
            Close
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 text-sm">
        {error && <p className="text-destructive">{error}</p>}
        {!d && !error && <p className="text-muted-foreground">Loading…</p>}
        {d && (
          <>
            <section>
              <h3 className="font-medium">
                {LABEL_TEXT[d.classification.category]} ·{" "}
                {(d.classification.confidence * 100).toFixed(0)}% confidence ({d.classification.basis ?? "no"} evidence)
              </h3>
              <p className="mt-1 text-xs text-muted-foreground">
                {d.classification.judge
                  ? `Rules were unsure, so the LLM judge (${d.classification.judge.model}) reviewed the evidence and made this call.`
                  : "Decided by the rules engine; the LLM judge was not needed."}
              </p>
              <p className="mt-1 text-muted-foreground">{d.classification.rationale}</p>
            </section>
            {d.risk && (
              <section>
                <h3 className="font-medium">
                  Risk: <TierBadge tier={d.risk.tier} /> score {d.risk.score}
                </h3>
                <ul className="mt-2 space-y-2">
                  {d.risk.factors.map((f) => (
                    <li key={f.key} className="rounded-md border p-2">
                      <p>
                        <span className="font-medium">+{f.points}</span> {f.why}
                      </p>
                      <p className="mt-1 flex flex-wrap gap-1">
                        {[...f.owasp, ...f.atlas].map((m) => (
                          <Badge key={m} variant="outline" className="font-normal">
                            {m}
                          </Badge>
                        ))}
                      </p>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <section>
              <h3 className="font-medium">Evidence</h3>
              <ul className="mt-2 space-y-1">
                {d.evidence.map((e) => (
                  <li key={e.id} className="flex gap-2">
                    <Badge variant="secondary" className="shrink-0 font-normal">
                      {e.tier}
                    </Badge>
                    <span>{e.detail}</span>
                  </li>
                ))}
              </ul>
            </section>
            <section>
              <h3 className="font-medium">Audit trail ({d.events.length} most recent)</h3>
              <div className="mt-2 max-h-72 overflow-auto rounded-md border">
                <Table className="text-xs">
                  <TableHeader>
                    <TableRow>
                      <TableHead>When</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead>Source</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {d.events.map((e) => (
                      <TableRow key={e.id}>
                        <TableCell className="tabular-nums">
                          {new Date(e.occurred_at).toLocaleTimeString()}
                        </TableCell>
                        <TableCell>{ACTION_TEXT[e.action]?.[0] ?? e.action}</TableCell>
                        <TableCell className="max-w-28 truncate" title={e.target}>
                          {e.target}
                        </TableCell>
                        <TableCell>{SOURCE_SHORT[e.source] ?? e.source}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </section>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function WorkloadTable({
  rows,
  selected,
  onSelect,
}: {
  rows: WorkloadRow[];
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Workload</TableHead>
          <TableHead>What it is</TableHead>
          <TableHead>Risk</TableHead>
          <TableHead>Top reasons</TableHead>
          <TableHead>Activity seen</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((w) => (
          <TableRow key={w.id} data-state={selected === w.name ? "selected" : undefined}>
            <TableCell>
              <button
                onClick={() => onSelect(w.name)}
                className="text-left font-medium underline-offset-4 hover:underline"
              >
                {w.name}
              </button>
              <div className="text-xs text-muted-foreground">{w.type}</div>
            </TableCell>
            <TableCell>
              {LABEL_TEXT[w.label]}
              <div className="text-xs text-muted-foreground">
                {(w.confidence * 100).toFixed(0)}% · {w.basis ?? "no"} evidence
              </div>
            </TableCell>
            <TableCell>
              <TierBadge tier={w.risk?.tier} />
            </TableCell>
            <TableCell className="max-w-64 whitespace-normal text-muted-foreground">
              {w.risk?.factors
                .toSorted((a, b) => b.points - a.points)
                .slice(0, 2)
                .map((f) => f.key.replaceAll("_", " "))
                .join(", ") || "n/a"}
            </TableCell>
            <TableCell className="whitespace-normal text-muted-foreground">
              {activityText(w.activity)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showServices, setShowServices] = useState(false);

  useEffect(() => {
    get<Overview>("/api/overview")
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error)
    return (
      <main className="mx-auto max-w-3xl p-10">
        <h1 className="text-xl font-semibold">No report to show</h1>
        <p className="mt-2 text-muted-foreground">{error}</p>
      </main>
    );
  if (!data) return <main className="p-10 text-muted-foreground">Loading report…</main>;

  const ai = data.workloads.filter((w) => w.label !== "service");
  const rows = showServices ? data.workloads : ai;
  const bySource: Record<string, number> = {};
  for (const r of data.events_by_source) bySource[r.source] = (bySource[r.source] ?? 0) + r.n;
  const events = Object.values(bySource).reduce((a, b) => a + b, 0);
  const optional = data.coverage.granted.length + data.coverage.missing.length;

  return (
    <main className="mx-auto max-w-7xl space-y-6 p-6 lg:p-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted-foreground">AI governance report</p>
          <h1 className="text-2xl font-semibold">AWS account {data.account_id}</h1>
          <p className="text-sm text-muted-foreground">
            Scanned {new Date(data.generated_at).toLocaleString()} · {data.regions.join(", ")} ·{" "}
            {data.coverage.granted.length} of {optional} optional data sources approved
          </p>
        </div>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Headline numbers">
        <Stat
          title="Workloads using AI"
          value={`${ai.length} of ${data.total}`}
          note={`${data.summary.agent} agents, ${data.summary.llm} single-call LLM apps`}
        />
        <Stat
          title="High-risk AI workloads"
          value={data.risk_tiers.high}
          note={`${data.risk_tiers.medium} medium, ${data.risk_tiers.low} low`}
        />
        <Stat
          title="Audit events recorded"
          value={data.distinct_events}
          note={`${events} source records from ${Object.keys(bySource).length} sources, merged by call`}
        />
        <Stat
          title="Blind spots"
          value={data.unclassified.length + data.access_requests.length}
          note={`${data.unclassified.length} unclassified resources, ${data.access_requests.length} open access requests`}
        />
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_26rem]">
        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle>AI inventory by risk</CardTitle>
                  <CardDescription>
                    Highest risk first. Select a workload for its evidence and audit trail.
                  </CardDescription>
                </div>
                <label className="flex items-center gap-2 text-sm text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={showServices}
                    onChange={(e) => setShowServices(e.target.checked)}
                  />
                  Show plain services ({data.summary.service})
                </label>
              </div>
            </CardHeader>
            <CardContent>
              <WorkloadTable rows={rows} selected={selected} onSelect={setSelected} />
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>OWASP Top 10 for LLM Apps (2026)</CardTitle>
                <CardDescription>AI workloads exposed to each risk</CardDescription>
              </CardHeader>
              <CardContent>
                <Bars data={data.owasp} label="OWASP exposure" />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>MITRE ATLAS techniques</CardTitle>
                <CardDescription>Attack techniques these workloads are open to</CardDescription>
              </CardHeader>
              <CardContent>
                <Bars data={data.atlas} label="ATLAS exposure" />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Data sources and approvals</CardTitle>
              <CardDescription>
                Every source the crawler read, whether it worked, and who on the client side
                approved the optional ones.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {Object.entries(data.approvals)
                  .sort(([, x], [, y]) => x.approved_at.localeCompare(y.approved_at))
                  .map(([src, g]) => (
                <div key={src} className="flex justify-between gap-3">
                  <span className="font-medium">{src}</span>
                  <span className="text-right text-muted-foreground">
                    {g.approved_by}, {new Date(g.approved_at).toLocaleString()}
                  </span>
                </div>
              ))}
              {data.coverage.missing.map((src) => (
                <div key={src} className="flex justify-between gap-3">
                  <span className="font-medium">{src}</span>
                  <span className="text-muted-foreground">not approved</span>
                </div>
              ))}
              <ul className="flex flex-wrap gap-1 pt-2">
                {data.sources.map((src) => (
                  <li key={src.name}>
                    <Badge
                      variant={src.ok ? "outline" : "destructive"}
                      className="font-normal"
                      title={src.detail || (src.ok ? "read without errors" : "failed")}
                    >
                      {src.ok ? "✓" : "✗"} {src.name.split("@")[0]}
                    </Badge>
                  </li>
                ))}
              </ul>
              {data.sources
                .filter((src) => !src.ok)
                .map((src) => (
                  <p key={src.name} className="text-destructive">
                    {src.name}: {src.detail}
                  </p>
                ))}
            </CardContent>
          </Card>
        </div>

        <aside className="min-w-0 space-y-6" aria-label="Workload details, scan history and open items">
          {selected ? (
            <Detail key={selected} name={selected} onClose={() => setSelected(null)} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Scan history</CardTitle>
                <CardDescription>Each scan is kept; nothing is overwritten.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {data.scans.map((s) => (
                  <div key={s.id}>
                    <p className="tabular-nums">{new Date(s.generated_at).toLocaleString()}</p>
                    <p className="text-muted-foreground">
                      {count(s.summary.agent, "agent")} · {count(s.summary.llm, "LLM app")} ·{" "}
                      {count(s.summary.service, "service")} ·{" "}
                      {s.grants.length} of {optional} sources
                    </p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader>
              <CardTitle>Open items</CardTitle>
              <CardDescription>What the crawler can&apos;t see yet, and what changed</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {data.access_requests.map((r) => (
                <div key={r.source}>
                  <p className="font-medium">
                    Access request: {r.source} <span className="text-muted-foreground">({r.status})</span>
                  </p>
                  <p className="text-muted-foreground">{r.why}</p>
                </div>
              ))}
              {data.unclassified.map((u) => (
                <div key={u.arn}>
                  <p>
                    Found, not classified yet: <span className="font-medium">{u.type}</span>
                  </p>
                  <p className="truncate text-muted-foreground" title={u.arn}>
                    {u.arn.split(":").at(-1)}
                  </p>
                </div>
              ))}
              {data.changes.slice(0, 6).map((c) => (
                <p key={c} className="text-muted-foreground">
                  Changed: {c}
                </p>
              ))}
              {!data.access_requests.length && !data.unclassified.length && !data.changes.length && (
                <p className="text-muted-foreground">Nothing open.</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Where the audit trail comes from</CardTitle>
              <CardDescription>
                Instrumented agents send OpenTelemetry to a collector; the rest are read from
                what AWS records. One Bedrock call seen by several sources counts once.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              {Object.entries(bySource).map(([s, n]) => (
                <div key={s} className="flex justify-between">
                  <span>{SOURCE_TEXT[s] ?? s}</span>
                  <span className="tabular-nums text-muted-foreground">{n} records</span>
                </div>
              ))}
              <p className="pt-2 text-muted-foreground">
                Distinct actions: {activityText(data.activity)}.
              </p>
            </CardContent>
          </Card>
        </aside>
      </div>
    </main>
  );
}
