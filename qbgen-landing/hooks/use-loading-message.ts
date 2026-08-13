"use client";

import { useEffect, useState } from "react";

export function useLoadingMessage(isLoading: boolean, initial: string, slowHint: string) {
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!isLoading) {
      setMessage("");
      return;
    }
    setMessage(initial);
    const coldStartTimer = window.setTimeout(() => {
      setMessage("Waking up the backend. The first request after idle can take a bit longer.");
    }, 4000);
    const upstreamTimer = window.setTimeout(() => setMessage(slowHint), 12000);
    return () => {
      window.clearTimeout(coldStartTimer);
      window.clearTimeout(upstreamTimer);
    };
  }, [isLoading, initial, slowHint]);

  return message;
}
