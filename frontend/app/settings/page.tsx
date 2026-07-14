"use client";
import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getProviderSettings, updateProviderSettings, generateProviderKey } from "@/lib/api";

export default function SettingsPage() {
  const qc = useQueryClient();
  const { data: settings, isLoading } = useQuery({
    queryKey: ["provider-settings"],
    queryFn: getProviderSettings,
  });

  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    if (settings) {
      setBaseUrl(settings.base_url);
      setApiKey(settings.api_key);
    }
  }, [settings]);

  const { mutate: save, isPending: saving, error: saveError } = useMutation({
    mutationFn: () => updateProviderSettings({ base_url: baseUrl, api_key: apiKey }),
    onSuccess: (data) => {
      qc.setQueryData(["provider-settings"], data);
      setSavedFlash(true);
      setTimeout(() => setSavedFlash(false), 1800);
    },
  });

  const { mutate: generate, isPending: generating } = useMutation({
    mutationFn: generateProviderKey,
    onSuccess: (data) => {
      qc.setQueryData(["provider-settings"], data);
      setApiKey(data.api_key);
      setShowKey(true);
    },
  });

  const copyKey = async () => {
    try {
      await navigator.clipboard.writeText(apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — silently ignore, key text is still selectable
    }
  };

  const dirty = settings ? (baseUrl !== settings.base_url || apiKey !== settings.api_key) : false;

  return (
    <div style={{ padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "auto" }}>
      <div style={{ maxWidth: 640 }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Settings
        </div>

        <div className="lf-panel" style={{ padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <div className="lf-section" style={{ margin: 0 }}>DataSupportTool Connection</div>
            {settings && (
              <span className={`lf-badge ${settings.configured ? "lf-badge-done" : "lf-badge-pending"}`}>
                {settings.configured ? "configured" : "not configured"}
              </span>
            )}
          </div>
          <p style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", margin: "4px 0 16px", lineHeight: 1.5 }}>
            Connects this dashboard to your Data Preprocessing Pipeline (DataSupportTool) as a
            training-data source and inference provider. Generate a key here, then paste the same
            value into DataSupportTool&apos;s <code>PROVIDER_API_KEY</code> env var.
          </p>

          {isLoading ? (
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>loading…</div>
          ) : (
            <>
              <div style={{ marginBottom: 14 }}>
                <label className="lf-label">Provider Endpoint (base URL)</label>
                <input
                  className="lf-input"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder="http://datasupporttool:8000"
                />
              </div>

              <div style={{ marginBottom: 6 }}>
                <label className="lf-label">Shared API Key</label>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    className="lf-input"
                    style={{ flex: 1, fontFamily: "var(--mono)" }}
                    type={showKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="no key set"
                  />
                  <button className="lf-btn lf-btn-ghost" style={{ padding: "0 10px" }} onClick={() => setShowKey((s) => !s)} type="button">
                    {showKey ? "hide" : "show"}
                  </button>
                  <button className="lf-btn lf-btn-ghost" style={{ padding: "0 10px" }} onClick={copyKey} disabled={!apiKey} type="button">
                    {copied ? "copied" : "copy"}
                  </button>
                </div>
              </div>

              <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                <button
                  className="lf-btn lf-btn-ghost"
                  onClick={() => generate()}
                  disabled={generating}
                  type="button"
                >
                  {generating ? <><span className="lf-spin" /> Generating…</> : "⟳ Auto-generate key"}
                </button>
                <button
                  className="lf-btn lf-btn-primary"
                  style={{ marginLeft: "auto" }}
                  onClick={() => save()}
                  disabled={saving || !dirty}
                  type="button"
                >
                  {saving ? <><span className="lf-spin" /> Saving…</> : savedFlash ? "Saved ✓" : "Save"}
                </button>
              </div>

              {saveError && (
                <div style={{ marginTop: 10, padding: "6px 8px", background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: 3, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>
                  {saveError instanceof Error ? saveError.message : "Failed to save settings"}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
