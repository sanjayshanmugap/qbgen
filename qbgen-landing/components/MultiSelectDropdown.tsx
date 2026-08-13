"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

type MultiSelectDropdownProps = {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  singular: string;
  plural: string;
  optional?: boolean;
};

export function MultiSelectDropdown({
  label, options, selected, onChange, singular, plural, optional = false,
}: MultiSelectDropdownProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggle = (option: string) =>
    onChange(
      selected.includes(option)
        ? selected.filter((item) => item !== option)
        : [...selected, option],
    );

  const summary =
    selected.length === 0
      ? `Select ${plural}`
      : selected.length === 1
      ? `1 ${singular} selected`
      : `${selected.length} ${plural} selected`;

  return (
    <div className="relative" ref={containerRef}>
      <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
        {label}
        {optional && (
          <span className="normal-case tracking-normal text-muted-foreground ml-1">(optional)</span>
        )}
      </label>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full h-12 flex items-center justify-between border-b border-foreground/20 bg-transparent text-left text-foreground hover:border-foreground/40 transition-colors focus:outline-none focus:border-accent focus:border-b-2"
      >
        <span className={selected.length > 0 ? "text-foreground" : "text-muted-foreground"}>
          {summary}
        </span>
        <ChevronDown className="h-4 w-4 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-60 overflow-y-auto">
          {options.map((option) => (
            <label
              key={option}
              className="flex items-center px-3 py-2 hover:bg-foreground/5 cursor-pointer text-foreground"
            >
              <input
                type="checkbox"
                checked={selected.includes(option)}
                onChange={() => toggle(option)}
                className="mr-3 h-4 w-4 accent-accent"
              />
              {option}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
