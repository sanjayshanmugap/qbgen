"use client";

import { useEffect, useRef, useState } from "react";
import { buildApiUrl } from "@/lib/api";

type Status = "checking" | "ready" | "waking" | "down";

interface BackendStatusProps {
  isWorking?: boolean;
}

const POLL_INTERVAL_MS = 60_000;
const HEALTH_TIMEOUT_MS = 15_000;
const WAKING_THRESHOLD_MS = 3_000;

export function BackendStatus({ isWorking = false }: BackendStatusProps) {
  const [status, setStatus] = useState<Status>("checking");
  const pollTimerRef = useRef<number | null>(null);
  const wakingTimerRef = useRef<number | null>(null);
  const inFlightRef = useRef<AbortController | null>(null);

  const cancelPoll = () => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const ping = async () => {
    inFlightRef.current?.abort();

    let url: string;
    try {
      url = buildApiUrl("/api/health");
    } catch {
      setStatus("down");
      return;
    }

    const controller = new AbortController();
    inFlightRef.current = controller;
    const abortTimer = window.setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    const started = performance.now();

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        cache: "no-store",
      });
      const elapsed = performance.now() - started;
      if (!response.ok) {
        setStatus("down");
        return;
      }
      setStatus(elapsed > WAKING_THRESHOLD_MS ? "waking" : "ready");
    } catch {
      setStatus("down");
    } finally {
      window.clearTimeout(abortTimer);
      if (inFlightRef.current === controller) {
        inFlightRef.current = null;
      }
    }
  };

  useEffect(() => {
    ping();

    const schedule = () => {
      cancelPoll();
      if (document.visibilityState !== "visible") return;
      pollTimerRef.current = window.setTimeout(async () => {
        await ping();
        schedule();
      }, POLL_INTERVAL_MS);
    };

    schedule();

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        ping();
        schedule();
      } else {
        cancelPoll();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      cancelPoll();
      inFlightRef.current?.abort();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  useEffect(() => {
    if (wakingTimerRef.current !== null) {
      window.clearTimeout(wakingTimerRef.current);
      wakingTimerRef.current = null;
    }

    if (isWorking) {
      wakingTimerRef.current = window.setTimeout(() => {
        setStatus((current) => (current === "down" ? current : "waking"));
      }, WAKING_THRESHOLD_MS);
    } else {
      ping();
    }

    return () => {
      if (wakingTimerRef.current !== null) {
        window.clearTimeout(wakingTimerRef.current);
        wakingTimerRef.current = null;
      }
    };
  }, [isWorking]);

  const { label, dotClass, srLabel } = STATUS_PRESENTATION[status];

  return (
    <div
      className="inline-flex items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground"
      aria-live="polite"
    >
      <span aria-hidden className={`inline-block h-2 w-2 rounded-full ${dotClass}`} />
      <span className="sr-only">{srLabel}</span>
      <span>{label}</span>
    </div>
  );
}

const STATUS_PRESENTATION: Record<
  Status,
  { label: string; dotClass: string; srLabel: string }
> = {
  checking: {
    label: "Checking backend",
    dotClass: "bg-foreground/30",
    srLabel: "Checking backend status",
  },
  ready: {
    label: "Backend ready",
    dotClass: "bg-emerald-500",
    srLabel: "Backend is online and responsive",
  },
  waking: {
    label: "Waking backend",
    dotClass: "bg-accent animate-pulse",
    srLabel: "Backend is waking up from idle",
  },
  down: {
    label: "Backend offline",
    dotClass: "bg-destructive",
    srLabel: "Backend is not responding",
  },
};
