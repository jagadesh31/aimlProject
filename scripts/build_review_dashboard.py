"""Build review charts + self-contained demo HTML pages (no server needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

EMOTIONS = config.EMOTIONS
OUT_DIR = config.DOCS_DIR / "review_demo"
CHART_DIR = OUT_DIR / "charts"


def load_metrics(name: str) -> dict:
    return json.loads((config.METRICS_DIR / name).read_text(encoding="utf-8"))


def plot_improvement():
    cur = load_metrics("test_metrics.json")
    prev = load_metrics("test_metrics_previous.json")
    labels = ["Accuracy", "Weighted F1"]
    old = [prev["accuracy"], prev["weighted_f1"]]
    new = [cur["accuracy"], cur["weighted_f1"]]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar(x - w / 2, old, w, label="Earlier model", color="#94a3b8")
    b2 = ax.bar(x + w / 2, new, w, label="Current model", color="#0f766e")
    ax.axhline(0.48, color="#b45309", linestyle="--", linewidth=1.2, label="Neutral-only baseline (~0.48)")
    ax.set_ylim(0, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("MELD test improvement (utterance-level)")
    ax.legend(loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = CHART_DIR / "improvement_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_per_class():
    # From current classification report (hardcoded from file to avoid parse fragility)
    support = [345, 68, 50, 402, 1256, 208, 281]
    precision = [0.32, 0.00, 0.00, 0.00, 0.51, 0.00, 0.38]
    recall = [0.24, 0.00, 0.00, 0.00, 0.95, 0.00, 0.03]
    f1 = [0.27, 0.00, 0.00, 0.00, 0.67, 0.00, 0.05]
    x = np.arange(len(EMOTIONS))
    w = 0.28
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, precision, w, label="Precision", color="#0369a1")
    ax.bar(x, recall, w, label="Recall", color="#0f766e")
    ax.bar(x + w, f1, w, label="F1", color="#a16207")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(EMOTIONS, support)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Current model — per-class metrics on MELD test")
    ax.legend()
    fig.tight_layout()
    path = CHART_DIR / "per_class_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion():
    cur = load_metrics("test_metrics.json")
    cm = np.array(cur["confusion_matrix"], dtype=float)
    # row-normalize for readability
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cmn = cm / row_sum
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(EMOTIONS, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized) — current MELD test")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = CHART_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_class_support():
    support = [345, 68, 50, 402, 1256, 208, 281]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#dc2626", "#7c3aed", "#4b5563", "#eab308", "#64748b", "#2563eb", "#f97316"]
    ax.bar(EMOTIONS, support, color=colors)
    ax.set_title("MELD test class imbalance (why Neutral dominates accuracy)")
    ax.set_ylabel("Number of utterances")
    for i, v in enumerate(support):
        ax.text(i, v + 20, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    path = CHART_DIR / "class_support.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


COLORS = {
    "anger": "#dc2626",
    "disgust": "#7c3aed",
    "fear": "#57534e",
    "joy": "#ca8a04",
    "neutral": "#64748b",
    "sadness": "#2563eb",
    "surprise": "#ea580c",
}


def timeline_html(payload: dict, out_path: Path):
    data_json = json.dumps(payload)
    colors_json = json.dumps(COLORS)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emotion Timeline — {payload.get("tag", "demo")}</title>
  <style>
    :root {{ --bg:#0b1220; --card:#121a2b; --ink:#e5eefc; --muted:#93a4c3; --line:#243149; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; background:linear-gradient(180deg,#0b1220,#111827); color:var(--ink); }}
    .wrap {{ max-width:1000px; margin:0 auto; padding:1.25rem; }}
    h1 {{ margin:0 0 0.25rem; font-size:1.35rem; }}
    .sub {{ color:var(--muted); margin-bottom:1rem; font-size:0.9rem; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1rem; margin-bottom:1rem; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:0.45rem; margin:0.6rem 0 0.9rem; }}
    .chip {{ font-size:0.75rem; padding:0.2rem 0.5rem; border-radius:999px; color:#fff; }}
    .track {{ position:relative; height:54px; border-radius:10px; background:#0f172a; overflow:hidden; border:1px solid var(--line); }}
    .seg {{ position:absolute; top:0; bottom:0; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:600; color:#fff; padding:0 4px; overflow:hidden; white-space:nowrap; border-right:1px solid rgba(255,255,255,0.15); }}
    .seg.transition {{ opacity:0.75; background-image:repeating-linear-gradient(135deg,rgba(255,255,255,0.18) 0 6px,transparent 6px 12px); }}
    .axis {{ display:flex; justify-content:space-between; color:var(--muted); font-size:0.75rem; margin-top:0.35rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    th, td {{ border-bottom:1px solid var(--line); padding:0.4rem 0.35rem; text-align:left; }}
    th {{ color:var(--muted); font-weight:600; }}
    .note {{ color:var(--muted); font-size:0.84rem; }}
    a {{ color:#5eead4; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Continuous emotion timeline</h1>
    <p class="sub">Tag: <strong id="tag"></strong> · Duration: <strong id="dur"></strong>s · Built from window predictions → smooth → stable/transition</p>
    <div class="card">
      <h3 style="margin:0 0 0.4rem;font-size:1rem;">Predicted timeline (our contribution)</h3>
      <div class="legend" id="legend"></div>
      <div class="track" id="track"></div>
      <div class="axis"><span>0 s</span><span id="endLabel"></span></div>
    </div>
    <div class="card" id="goldCard" style="display:none;">
      <h3 style="margin:0 0 0.4rem;font-size:1rem;">MELD gold utterances (for comparison)</h3>
      <div class="track" id="goldTrack"></div>
      <div class="axis"><span>0 s</span><span id="endLabel2"></span></div>
      <p class="note">Gold is one label per turn. Our timeline is continuous over sliding windows.</p>
    </div>
    <div class="card">
      <h3 style="margin:0 0 0.5rem;font-size:1rem;">Segments</h3>
      <table>
        <thead><tr><th>Start</th><th>End</th><th>Type</th><th>Emotion</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <p class="note"><a href="index.html">← Review demo home</a></p>
  </div>
  <script>
    const data = {data_json};
    const COLORS = {colors_json};
    const dur = data.duration_sec || 1;
    document.getElementById('tag').textContent = data.tag || '';
    document.getElementById('dur').textContent = dur.toFixed(2);
    document.getElementById('endLabel').textContent = dur.toFixed(1) + ' s';
    document.getElementById('endLabel2').textContent = dur.toFixed(1) + ' s';
    const legend = document.getElementById('legend');
    Object.keys(COLORS).forEach(e => {{
      const s = document.createElement('span');
      s.className = 'chip';
      s.style.background = COLORS[e];
      s.textContent = e;
      legend.appendChild(s);
    }});
    const track = document.getElementById('track');
    (data.timeline || []).forEach(seg => {{
      const el = document.createElement('div');
      el.className = 'seg' + (seg.type === 'transition' ? ' transition' : '');
      const emo = seg.type === 'transition' ? (seg.from + '→' + seg.to) : seg.emotion;
      const colorKey = seg.type === 'transition' ? (seg.to || 'neutral') : seg.emotion;
      el.style.left = (100 * seg.start / dur) + '%';
      el.style.width = Math.max(0.8, 100 * (seg.end - seg.start) / dur) + '%';
      el.style.backgroundColor = COLORS[colorKey] || '#475569';
      el.textContent = emo;
      el.title = seg.start.toFixed(1) + '-' + seg.end.toFixed(1) + 's  ' + emo;
      track.appendChild(el);
    }});
    const rows = document.getElementById('rows');
    (data.timeline || []).forEach(seg => {{
      const tr = document.createElement('tr');
      const emo = seg.type === 'transition' ? ((seg.from||'') + ' → ' + (seg.to||'')) : seg.emotion;
      tr.innerHTML = `<td>${{seg.start.toFixed(2)}}</td><td>${{seg.end.toFixed(2)}}</td><td>${{seg.type}}</td><td>${{emo}}</td>`;
      rows.appendChild(tr);
    }});
    if (data.gold_utterances && data.gold_utterances.length) {{
      document.getElementById('goldCard').style.display = 'block';
      const gt = document.getElementById('goldTrack');
      data.gold_utterances.forEach(g => {{
        const el = document.createElement('div');
        el.className = 'seg';
        el.style.left = (100 * g.start / dur) + '%';
        el.style.width = Math.max(0.8, 100 * (g.end - g.start) / dur) + '%';
        el.style.backgroundColor = COLORS[g.gold_emotion] || '#475569';
        el.textContent = g.gold_emotion;
        el.title = (g.utterance || '') + ' [' + g.gold_emotion + ']';
        gt.appendChild(el);
      }});
    }}
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def results_html():
    cur = load_metrics("test_metrics.json")
    prev = load_metrics("test_metrics_previous.json")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MELD Results Dashboard</title>
  <style>
    body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; background:#f5f3ef; color:#1c1917; }}
    .wrap {{ max-width:980px; margin:0 auto; padding:1.25rem; }}
    .card {{ background:#fff; border:1px solid #e7e0d6; border-radius:12px; padding:1rem; margin-bottom:1rem; }}
    h1 {{ margin:0 0 0.4rem; font-size:1.4rem; }}
    .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.7rem; }}
    .kpi {{ background:#f0fdfa; border:1px solid #99f6e4; border-radius:10px; padding:0.75rem; }}
    .kpi b {{ display:block; font-size:1.4rem; color:#0f766e; }}
    img {{ max-width:100%; border-radius:8px; border:1px solid #e7e0d6; }}
    .muted {{ color:#57534e; font-size:0.9rem; }}
    a {{ color:#0f766e; }}
    @media (max-width:700px) {{ .kpis {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MELD test results dashboard</h1>
    <p class="muted">Utterance-level evaluation on 2610 test clips. Audio-only CNN + BiLSTM.</p>
    <div class="kpis">
      <div class="kpi"><span>Current accuracy</span><b>{cur['accuracy']:.1%}</b></div>
      <div class="kpi"><span>Current weighted F1</span><b>{cur['weighted_f1']:.2f}</b></div>
      <div class="kpi"><span>Earlier accuracy</span><b>{prev['accuracy']:.1%}</b></div>
    </div>
    <div class="card">
      <h3>Improvement vs earlier model</h3>
      <img src="charts/improvement_bar.png" alt="improvement" />
      <p class="muted">Neutral-only baseline is about 48% because Neutral is almost half of the test set.</p>
    </div>
    <div class="card">
      <h3>Class imbalance in MELD test</h3>
      <img src="charts/class_support.png" alt="support" />
    </div>
    <div class="card">
      <h3>Per-class precision / recall / F1</h3>
      <img src="charts/per_class_metrics.png" alt="per class" />
      <p class="muted">Neutral and some anger work; minority emotions remain hard for audio-only SER.</p>
    </div>
    <div class="card">
      <h3>Confusion matrix</h3>
      <img src="charts/confusion_matrix.png" alt="confusion" />
    </div>
    <p class="muted"><a href="index.html">← Review demo home</a></p>
  </div>
</body>
</html>
"""
    path = OUT_DIR / "results.html"
    path.write_text(html, encoding="utf-8")
    return path


def index_html():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Continuous SER — Review Demo</title>
  <style>
    body { margin:0; font-family:Segoe UI,Arial,sans-serif; background:#f7f4ef; color:#1c1917; }
    .wrap { max-width:820px; margin:0 auto; padding:1.5rem; }
    .card { background:#fff; border:1px solid #e7e0d6; border-radius:14px; padding:1.1rem 1.2rem; margin-bottom:0.85rem; }
    h1 { margin:0 0 0.35rem; }
    a.btn { display:inline-block; margin:0.25rem 0.35rem 0.25rem 0; padding:0.55rem 0.85rem; background:#0f766e; color:#fff; text-decoration:none; border-radius:8px; font-weight:600; }
    a.btn.secondary { background:#0369a1; }
    a.btn.ghost { background:#fff; color:#0f766e; border:1px solid #0f766e; }
    .muted { color:#57534e; }
    ol { line-height:1.55; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Lab review demo pack</h1>
    <p class="muted">Open these pages during the review. No server needed — double-click any HTML file.</p>
    <div class="card">
      <h3>Show these</h3>
      <a class="btn" href="timeline_demo_sequence.html">1. Timeline visualizer (demo audio)</a>
      <a class="btn secondary" href="timeline_test_dia0.html">2. Timeline + MELD gold turns</a>
      <a class="btn" href="results.html">3. Results charts (accuracy / F1)</a>
      <a class="btn ghost" href="../LAB_REVIEW_GUIDE.html">4. A–Z speaking guide</a>
      <a class="btn ghost" href="../CODE_IN_DEPTH.html">5. Code explanation</a>
    </div>
    <div class="card">
      <h3>Suggested 2-minute demo order</h3>
      <ol>
        <li>Open <b>timeline_demo_sequence</b> — show stable regions and striped transitions.</li>
        <li>Open <b>timeline_test_dia0</b> — compare predicted timeline vs MELD gold turns.</li>
        <li>Open <b>results</b> — show 34% → 49% improvement and confusion matrix.</li>
        <li>If asked for code: open <code>src/temporal_tracking.py</code>.</li>
      </ol>
    </div>
  </div>
</body>
</html>
"""
    path = OUT_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    print("Building charts...")
    plot_improvement()
    plot_per_class()
    plot_confusion()
    plot_class_support()
    print("Building HTML...")
    results_html()
    index_html()
    for name in ["timeline_demo_sequence.json", "timeline_test_dia0.json"]:
        src = config.TIMELINE_DIR / name
        if not src.exists():
            print("Missing", src)
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        out = OUT_DIR / (src.stem + ".html")
        timeline_html(payload, out)
        print("Wrote", out)
    print("Done. Open:", OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
