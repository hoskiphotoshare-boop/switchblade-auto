# Switchblade Quantitative Trading & Dashboard Rules

These project rules enforce development invariants, scheduling standards, and sandbox operational practices across the Switchblade codebase.

---

## 1. Sandboxed Git Commit & Push Workflow
- In Antigravity sandboxed environments, never attempt to run `git push` via shell command line (which triggers network 403 or blocks awaiting user elevation).
- When changes need to be committed and pushed to remote GitHub repositories:
  1. Use the GitHub MCP server tool `push_files` (`call_mcp_tool` with `ServerName: "github"`) to push single or multi-file atomic commits directly to `main`.
  2. Follow up immediately with a local working-tree synchronization: `git fetch origin main && git reset --hard origin/main`.

---

## 2. Wall Street Market Close Dual-Cron Invariant
- Market-close automated workflows must run 17 minutes after the 4:00 PM Eastern equity market close (4:17 PM ET).
- Because GitHub Actions cron schedules operate strictly on UTC and the US alternates between EDT (UTC-4) and EST (UTC-5), never configure a single cron trigger.
- Always implement dual cron schedules paired with runtime timezone offset validation:
  ```yaml
  schedule:
    - cron: '17 20 * * 1-5' # 4:17 PM EDT (Daylight Saving Time)
    - cron: '17 21 * * 1-5' # 4:17 PM EST (Standard Time)
  ```
  with a pre-flight validation step:
  ```bash
  OFFSET=$(TZ="America/New_York" date +%z)
  # Exit 0 with should_run=false if trigger does not match current seasonal offset
  ```

---

## 3. Mobile Viewport & Gesture Containment Invariants
When modifying `docs/index.html` or any web dashboard:
- **Root Viewport Lockdown:** Always keep `html, body { width: 100%; max-width: 100%; overflow-x: hidden; overscroll-behavior-x: none; }`. Never use `width: 100vw`.
- **Static Bottom Navigation:** The bottom navigation bar must remain statically anchored to the bottom of the viewport (`position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000; height: calc(var(--bottom-nav-height) + env(safe-area-inset-bottom, 0px));`).
- **Horizontal Scroll Isolation:** Horizontally scrollable elements (financial tables, chart switchers, quick-jump pills) must enforce `overscroll-behavior-x: contain; overscroll-behavior-y: auto; touch-action: pan-x pan-y;` so swiping horizontally scrolls the element without dragging the page horizontally, while allowing vertical finger scrolling up and down the page.
- **Preformatted Text Wrap:** All console dumps, `<pre>` tags, and code blocks must include `word-break: break-all; overflow-wrap: anywhere;` to prevent unformatted output from breaking mobile container widths.

---

## 4. Headless Execution & Matplotlib Invariant
- In all production Python scripts (`switchblade.py`), set `matplotlib.use('Agg')` strictly **before** importing `matplotlib.pyplot`. Importing `pyplot` first initialises an interactive display server that crashes headless Linux CI runners.

---

## 5. Point-in-Time (PIT) Dataset Integrity
- `data/full_production_holdings.csv` is the canonical master constituent change log. Never hardcode constituent lists or modify historical dates directly in strategy code; all ticker additions and deletions must be grounded in this PIT dataset.
