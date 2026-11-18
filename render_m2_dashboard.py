#!/usr/bin/env python3
"""Render the M2 Pro benchmark dashboard from the saved JSON reports."""

import json
from html import escape
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = (
    ("Qwen2.5-Coder 0.5B", "qwen-coder-0.5b"),
    ("SmolLM2 1.7B", "smollm2-1.7b"),
)
METRICS = (
    ("prompt", "PP512", "Prompt processing"),
    ("decode", "TG128", "Token generation"),
)
SWEEP_RUNS = (
    ("cpu", "CPU only"),
    ("metal_stock", "Metal stock"),
    ("metal", "Metal batch 128"),
    ("metal_batch_256", "Metal batch 256"),
)


def report(slug, verification=False):
    prefix = "verify-" if verification else ""
    return json.loads((RESULTS / f"apple-m2-pro-{prefix}{slug}.json").read_text())


def tps(value):
    return f"{value:,.2f}"


def chart(code, title, stock, tuned):
    delta = (tuned / stock - 1) * 100
    sign = "+" if delta > 0 else "−" if delta < 0 else ""
    modifier = "gain" if delta > 0 else "loss" if delta < 0 else ""
    scale = max(stock, tuned)
    bars = []
    for label, value in (("Stock", stock), ("Tuned", tuned)):
        bars.append(f"""
          <div class="bar-row">
            <span>{label}</span>
            <span class="track" aria-hidden="true">
              <span class="fill {label.lower()}" style="width:{100 * value / scale:.2f}%"></span>
            </span>
            <span class="value">{tps(value)}</span>
          </div>""")
    return f"""
        <section class="metric" aria-label="{escape(title)}">
          <div class="metric-head">
            <h4><span class="code">{code}</span> {title}</h4>
            <span class="delta {modifier}" aria-label="Tuned change versus stock {sign}{abs(delta):.2f} percent">{sign}{abs(delta):.2f}%</span>
          </div>
          {''.join(bars)}
        </section>"""


def verification_card(name, data):
    stock, tuned = (data["runs"][mode]["median_tokens_per_second"] for mode in ("stock", "tuned"))
    return f"""
      <article class="card">
        <h3>{escape(name)}</h3>
        <div class="metrics">
          {''.join(chart(code, title, stock[key], tuned[key]) for key, code, title in METRICS)}
        </div>
      </article>"""


def sweep_table(name, data):
    rows = []
    for mode, label in SWEEP_RUNS:
        run = data["runs"][mode]
        speeds = run["median_tokens_per_second"]
        rows.append(f"""
          <tr><th scope="row">{label}</th><td>{run['batch']:,} / {run['microbatch']:,}</td>
          <td>{tps(speeds['prompt'])}</td><td>{tps(speeds['decode'])}</td></tr>""")
    return f"""
      <table>
        <caption>{escape(name)}</caption>
        <thead><tr><th scope="col">Backend</th><th scope="col">Batch / microbatch</th>
          <th scope="col">PP512</th><th scope="col">TG128</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def prompt_lookup_card(title, sample, assessment=""):
    rows = []
    for mode, label in (("stock", "Stock"), ("lookup", "Prompt lookup")):
        run = sample[mode]
        rows.append(f"""
              <tr><th scope="row">{label}</th><td>{tps(run['median_decode_tps'])}</td>
                <td>{run['median_wall_seconds']:.2f} s</td></tr>""")
    identity = ("All measured replies used identical output token IDs."
                if sample["same_token_ids_all_replies"] else
                "Output token IDs differed across measured replies. Reply content or length may affect the comparison.")
    return f"""
        <article class="card">
          <h3>{title}</h3>
          <p>Prompt lookup generated tokens at <strong>{sample['decode_ratio']:.2f}×</strong> the stock rate.</p>
          <table aria-label="{title} timing">
            <thead><tr><th scope="col">Mode</th><th scope="col">Output tokens/s</th>
              <th scope="col">Request time</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
          <p>{identity} {assessment}</p>
        </article>"""


def prompt_lookup_section():
    cards = []
    original = RESULTS / "apple-m2-pro-ngram-qwen-coder-0.5b.json"
    if original.is_file():
        report = json.loads(original.read_text())
        summary = report["summary"]
        edit = summary["edit"]
        assessment = "The task asked for <code>top_customers</code> to default to three results."
        if edit["edit_ast_correct_all"]:
            assessment += " Every reply matched the expected Python syntax tree."
        elif edit["same_token_ids_all_replies"] and "edit" in report.get("observations", {}):
            assessment += " " + escape(report["observations"]["edit"])
        else:
            assessment += " At least one reply failed the syntax-tree check."
        cards.append(prompt_lookup_card("Change the default limit", edit, assessment))
        explanation = escape(report.get("observations", {}).get("explain", ""))
        cards.append(prompt_lookup_card("Code explanation", summary["explain"], explanation))
    followup = RESULTS / "apple-m2-pro-ngram-rename-qwen-coder-0.5b.json"
    if followup.is_file():
        rename = json.loads(followup.read_text())["summary"]["rename"]
        assessment = "The task was to rename <code>summarize_orders</code> to <code>total_by_customer</code>."
        if rename["functional_pass_all"] and rename["edit_ast_correct_all"]:
            assessment += " Every reply matched the expected Python syntax tree and passed the functional check."
        elif rename["functional_pass_all"]:
            assessment += " Every reply passed the functional check."
        elif rename["same_token_ids_all_replies"]:
            assessment += " Every reply failed the functional check."
        else:
            assessment += " At least one reply failed the functional check."
        cards.append(prompt_lookup_card("Rename the function", rename, assessment))
    if not cards:
        return ""
    return f"""
    <section id="speculative" aria-labelledby="lookup-title">
      <div class="section-head">
        <h2 id="lookup-title">Qwen2.5-Coder 0.5B with prompt lookup</h2>
        <p>Prompt lookup proposes tokens from repeated text for the model to verify.
          Each value is the median of six requests per workload and mode. The four blocks ran stock, lookup, lookup, then stock.</p>
      </div>
      <div class="cards">{''.join(cards)}</div>
    </section>
"""


def render():
    paired = [report(slug, verification=True) for _, slug in MODELS]
    sweep = [report(slug) for _, slug in MODELS]
    hardware = paired[0]
    engine = Path(hardware["runs"]["stock"]["command"][2]).parent.name
    memory = hardware["memory_bytes"] // (1024 ** 3)
    repeats = hardware["repetitions"]
    stock = hardware["runs"]["stock"]
    tuned = hardware["runs"]["tuned"]
    utc_day = hardware["started_utc"].split("T", 1)[0]
    utc_time = hardware["started_utc"].split("T", 1)[1][:5]
    styles = """
    :root {
      color-scheme: light;
      --ink: #182d27; --muted: #4f6058; --border: #d6dfd7;
      --paper: #f4f6f1; --white: #fff; --stock: #596d67; --tuned: #26735e;
      --display: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 var(--display); }
    main { max-width: 76rem; margin: 0 auto; padding: clamp(1rem, 4vw, 3rem) clamp(.75rem, 4vw, 2.5rem) 3rem; }
    header { padding: clamp(1.1rem, 4vw, 2.6rem); border-radius: 1.05rem; background: #182f27; color: #f4faf4; }
    header p { margin: .25rem 0 0; color: #d0e2d4; }
    h1, h2, h3, h4 { margin: 0; line-height: 1.18; }
    h1 { font-size: clamp(1.8rem, 5vw, 3.15rem); letter-spacing: -.045em; }
    h2 { font-size: 1.48rem; letter-spacing: -.027em; }
    h3 { font-size: 1.17rem; letter-spacing: -.014em; }
    h4 { font-size: .89rem; font-weight: 650; }
    .summary { margin: 1.05rem 0 0; max-width: 72ch; font-size: 1.02rem; }
    .summary strong { color: #d8f2a9; }
    .section-head { margin: 2rem 0 .8rem; }
    .section-head p, .method p { color: var(--muted); margin: .36rem 0 0; }
    .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem; }
    .card, details { background: var(--white); border: 1px solid var(--border); border-radius: .82rem; }
    .card { padding: 1.15rem; }
    .card > h3 { padding-bottom: .13rem; }
    .metric { margin-top: .92rem; padding-top: .85rem; border-top: 1px solid #e3e9e3; }
    .metric-head { display: flex; align-items: flex-start; justify-content: space-between; gap: .4rem; }
    .code { color: var(--muted); font: bold .77rem ui-monospace, Menlo, Consolas, monospace; padding-right: .18rem; }
    .delta { flex: none; border-radius: .28rem; padding: .09rem .36rem; color: #3c5548; background: #eff3ef; }
    .delta.gain { color: #155541; background: #def3e8; }
    .delta.loss { color: #7b3727; background: #f9ece5; }
    .delta, .value, td { font-variant-numeric: tabular-nums; }
    .delta, .bar-row { font-size: .88rem; }
    .bar-row { display: grid; grid-template-columns: 2.9rem minmax(.6rem, 1fr) minmax(4.6rem, max-content); gap: .48rem; align-items: center; margin-top: .39rem; }
    .track { display: block; height: .52rem; border-radius: 2rem; background: #e6ece6; }
    .fill { display: block; height: 100%; border-radius: inherit; }
    .fill.stock { background: var(--stock); }
    .fill.tuned { background: var(--tuned); }
    .value { text-align: right; font-weight: 670; }
    details { margin-top: .85rem; }
    summary { cursor: pointer; padding: .72rem 1rem; font-weight: 690; }
    summary:focus-visible { outline: 2px solid #175d49; outline-offset: 3px; border-radius: .5rem; }
    .method { padding: 0 1rem .9rem; }
    ul { margin: .7rem 0 0; padding-left: 1.16rem; }
    code { font: .78rem/1.45 ui-monospace, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    table { border-collapse: collapse; width: 100%; margin-top: 1.15rem; font-size: .9rem; }
    caption { text-align: left; font-size: 1rem; font-weight: 720; margin-bottom: .24rem; }
    th, td { padding: .3rem .35rem; border-bottom: 1px solid #e2e9e3; text-align: right; }
    th:first-child { text-align: left; }
    tbody th { font-weight: 560; }
    @media (max-width: 42rem) { .cards { grid-template-columns: 1fr; } }
    @media (max-width: 24rem) {
      .card { padding: .8rem; }
      .bar-row { gap: .3rem; }
      table { font-size: .76rem; }
      th, td { padding: .27rem .18rem; }
    }
    @media print {
      :root { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
      body { background: white; }
      main { padding: 0; }
      .card, table { break-inside: avoid; }
    }
    """
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M2 Pro inference benchmarks</title>
  <style>{styles}</style>
</head>
<body>
  <main>
    <header>
      <h1>M2 Pro inference benchmarks</h1>
      <p>Mac mini · {memory} GB · macOS {escape(hardware['macos'])} · {escape(engine)}</p>
      <p class="summary"><strong>Stock Metal led prompt processing on both models.</strong>
        Tuning gave Qwen a small decoding gain, while SmolLM2 decoded faster with stock settings.
        Keep stock as the general default based on this check.</p>
    </header>

    <section aria-labelledby="paired-title">
      <div class="section-head">
        <h2 id="paired-title">Paired Metal verification</h2>
        <p>Median tokens/sec from {repeats} timed samples per configuration and metric, in one paired comparison per model.
          Percentages show tuned relative to stock.</p>
      </div>
      <div class="cards">
        {''.join(verification_card(name, data) for (name, _), data in zip(MODELS, paired))}
      </div>
    </section>

    <details>
      <summary>Initial CPU and Metal sweep</summary>
      <div class="method">
        <p>Median tokens/sec from the initial sweep of {sweep[0]['repetitions']} timed samples per metric and configuration.
          CPU and Metal stock used different batch settings.</p>
        {''.join(sweep_table(name, data) for (name, _), data in zip(MODELS, sweep))}
      </div>
    </details>

    <details>
      <summary>Method and source reports</summary>
      <div class="method">
        <p>PP512 processes a 512-token prompt. TG128 generates 128 tokens. Both models use Q4_K_M weights.
          The paired runs were warmed up, used six threads, and ran tuned first, then stock on {escape(utc_day)} at {escape(utc_time)} UTC.
          Stock used batch {stock['batch']:,} and microbatch {stock['microbatch']:,}. Tuned used {tuned['batch']:,} for both.
          The initial sweep selected tuned batch {tuned['batch']:,} from the two explicit batch sizes based on prompt throughput.</p>
        <p>With one pair per model at a fixed run order, the small decoding differences need another check to establish repeatability.</p>
        <ul>
          {''.join(f'<li><code>apple-m2-pro-{prefix}{slug}.json</code></li>' for _, slug in MODELS for prefix in ('', 'verify-'))}
        </ul>
      </div>
    </details>
{prompt_lookup_section()}  </main>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html.split("\n"))


if __name__ == "__main__":
    output = RESULTS / "apple-m2-pro-dashboard.html"
    output.write_text(render(), encoding="utf-8")
    print(output)
