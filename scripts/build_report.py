"""Generate a self-contained static HTML dashboard from pipeline artifacts.

Reads:  data/processed/dataset.parquet, predictions.parquet, reports/metrics.json
Writes: reports/index.html  (single file, opens in any browser, deploy on GH Pages)

Run:  python -m scripts.build_report
"""
import json

import pandas as pd

from src.config import PROCESSED, ROOT

REPORTS = ROOT / "reports"


def load():
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    return df, preds, metrics


def series_for(df):
    """Pre-aggregate per-ticker time series (downsampled to keep file small)."""
    out = {}
    for tic, g in df.groupby("ticker"):
        g = g.sort_values("date").tail(250)  # ~1 trading year
        out[tic] = {
            "dates": g["date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": g["close"].round(2).tolist(),
            "sentiment": g["sent_mean"].round(4).tolist(),
            "volume": g["msg_volume"].astype(int).tolist(),
        }
    return out


def build_html(df, preds, metrics):
    series = series_for(df)
    preds_records = preds.assign(
        date=preds["date"].astype(str),
        pred_up_proba=preds["pred_up_proba"].round(4),
        sent_mean=preds["sent_mean"].round(4),
        close=preds["close"].round(2),
    ).to_dict(orient="records")

    edge = metrics["edge"]
    edge_class = "positive" if edge > 0.01 else ("negative" if edge < -0.01 else "neutral")
    verdict = ("Signal beats baseline" if edge > 0.01
               else "No edge over baseline" if edge <= 0.01 else "")

    payload = {
        "metrics": metrics,
        "series": series,
        "predictions": preds_records,
        "verdict": verdict,
        "edge_class": edge_class,
    }

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload))

    # Offline-safe option: if a vendored chart.js is present, inline it and drop
    # the CDN <script>. Keeps the file truly self-contained (the skill's promise).
    vendor = REPORTS / "vendor" / "chart.umd.min.js"
    if vendor.exists():
        lib = vendor.read_text(encoding="utf-8")
        html = html.replace(
            '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>',
            f"<script>{lib}</script>",
        ).replace(
            '<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0"></script>',
            "",
        )
        print("[report] inlined vendored chart.js (offline-safe)")
    return html


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Stock Signal — Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--header:#161922;--text:#e6e8ef;--muted:#9aa1b3;
--pos:#34d399;--neg:#f87171;--neutral:#fbbf24;--accent:#60a5fa;--gap:16px;--radius:12px;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);padding:24px;line-height:1.5}
.wrap{max-width:1200px;margin:0 auto}
header{background:var(--header);border-radius:var(--radius);padding:24px;margin-bottom:var(--gap);
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
header h1{font-size:20px;font-weight:600}
header .sub{color:var(--muted);font-size:13px;margin-top:4px}
.disclaimer{background:#2a1f0a;border:1px solid #5c4514;border-radius:8px;padding:12px 16px;
margin-bottom:var(--gap);font-size:13px;color:#fcd34d}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:var(--gap);margin-bottom:var(--gap)}
.card{background:var(--card);border-radius:var(--radius);padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.kpi-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.kpi-value{font-size:30px;font-weight:700}
.kpi-sub{font-size:13px;color:var(--muted);margin-top:4px}
.positive{color:var(--pos)} .negative{color:var(--neg)} .neutral{color:var(--neutral)}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:var(--gap);margin-bottom:var(--gap)}
.charts .full{grid-column:1/-1}
.card h3{font-size:14px;font-weight:600;margin-bottom:14px}
.controls{display:flex;gap:10px;align-items:center}
select{background:#222634;color:var(--text);border:1px solid #333;border-radius:6px;padding:6px 10px;font-size:13px}
canvas{max-height:300px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #262a36}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge.up{background:rgba(52,211,153,.15);color:var(--pos)}
.badge.down{background:rgba(248,113,113,.15);color:var(--neg)}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:8px}
@media(max-width:780px){.charts{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<header>
<div><h1>📈 Sentiment Stock Signal — Monitor</h1>
<div class="sub" id="genat"></div></div>
<div class="controls"><label for="tic" style="color:var(--muted);font-size:13px">Ticker</label>
<select id="tic"></select></div>
</header>

<div class="disclaimer">
Learning/portfolio project. This measures whether social sentiment predicts next-day
returns — it is <b>not</b> investment advice. Metrics below are walk-forward
cross-validated and shown honestly, including when the signal is weak.
</div>

<section class="kpis">
<div class="card"><div class="kpi-label">CV Accuracy</div>
<div class="kpi-value" id="k-acc"></div><div class="kpi-sub" id="k-base"></div></div>
<div class="card"><div class="kpi-label">Edge vs Baseline</div>
<div class="kpi-value" id="k-edge"></div><div class="kpi-sub" id="k-verdict"></div></div>
<div class="card"><div class="kpi-label">ROC AUC</div>
<div class="kpi-value" id="k-auc"></div><div class="kpi-sub">0.50 = coin flip</div></div>
<div class="card"><div class="kpi-label">Data</div>
<div class="kpi-value" id="k-rows"></div><div class="kpi-sub" id="k-tickers"></div></div>
</section>

<section class="charts">
<div class="card full"><h3>Sentiment vs Price (selected ticker)</h3><canvas id="overlay"></canvas></div>
<div class="card"><h3>Message volume</h3><canvas id="vol"></canvas></div>
<div class="card"><h3>Latest predictions — P(up) by ticker</h3><canvas id="predbar"></canvas></div>
</section>

<section class="card">
<h3>Today's signal table (sorted by confidence)</h3>
<table><thead><tr><th>Ticker</th><th>Date</th><th>Close</th><th>Sentiment</th><th>Msgs</th><th>P(up)</th><th>Signal</th></tr></thead>
<tbody id="tbody"></tbody></table>
</section>

<footer>Regenerated automatically by the pipeline. Single static file — no server.</footer>
</div>

<script>
const D = /*__DATA__*/;
const fmtPct = x => (x*100).toFixed(1)+'%';
const hasChart = (typeof Chart !== 'undefined');

// --- KPIs (no library dependency) ---
document.getElementById('genat').textContent = 'Generated ' + D.metrics.generated_at;
document.getElementById('k-acc').textContent = fmtPct(D.metrics.cv_accuracy);
document.getElementById('k-base').textContent = 'baseline ' + fmtPct(D.metrics.cv_baseline);
const edgeEl = document.getElementById('k-edge');
edgeEl.textContent = (D.metrics.edge>=0?'+':'') + fmtPct(D.metrics.edge);
edgeEl.className = 'kpi-value ' + D.edge_class;
document.getElementById('k-verdict').textContent = D.verdict;
document.getElementById('k-auc').textContent = D.metrics.cv_auc.toFixed(3);
document.getElementById('k-rows').textContent = D.metrics.rows.toLocaleString();
document.getElementById('k-tickers').textContent = D.metrics.tickers.join(', ');

// --- Table (no library dependency; render FIRST so charts can't block it) ---
const tb = document.getElementById('tbody');
D.predictions.forEach(p=>{
  const up = p.pred_up_proba>=0.5;
  tb.insertAdjacentHTML('beforeend',
    `<tr><td><b>${p.ticker}</b></td><td>${p.date}</td><td>${p.close}</td>
     <td>${p.sent_mean}</td><td>${p.msg_volume}</td><td>${fmtPct(p.pred_up_proba)}</td>
     <td><span class="badge ${up?'up':'down'}">${up?'▲ up':'▼ down'}</span></td></tr>`);
});

// --- Ticker selector ---
const sel = document.getElementById('tic');
Object.keys(D.series).forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o);});

// --- Charts (guarded: if the library failed to load, show a notice instead of crashing) ---
if(!hasChart){
  document.querySelectorAll('canvas').forEach(c=>{
    const n=document.createElement('div');n.style.cssText='color:#9aa1b3;font-size:13px;padding:24px;text-align:center';
    n.textContent='Charts need the Chart.js library (no internet / blocked CDN). KPIs and the table above still work. Build with a vendored copy in reports/vendor/ for offline charts.';
    c.replaceWith(n);
  });
} else {
  let overlay, vol;
  function draw(tic){
    const s = D.series[tic];
    if(overlay) overlay.destroy(); if(vol) vol.destroy();
    overlay = new Chart(document.getElementById('overlay'),{
      data:{labels:s.dates,datasets:[
        {type:'line',label:'Close',data:s.close,yAxisID:'y',borderColor:'#60a5fa',borderWidth:2,pointRadius:0,tension:.3},
        {type:'line',label:'Sentiment',data:s.sentiment,yAxisID:'y1',borderColor:'#34d399',borderWidth:2,pointRadius:0,tension:.3}
      ]},
      options:{responsive:true,animation:false,interaction:{mode:'index',intersect:false},
        scales:{y:{position:'left',ticks:{color:'#9aa1b3'}},y1:{position:'right',grid:{drawOnChartArea:false},ticks:{color:'#9aa1b3'}},
        x:{ticks:{color:'#9aa1b3',maxTicksLimit:8}}},plugins:{legend:{labels:{color:'#e6e8ef'}}}}
    });
    vol = new Chart(document.getElementById('vol'),{
      type:'bar',data:{labels:s.dates,datasets:[{label:'Msgs',data:s.volume,backgroundColor:'#60a5fa55'}]},
      options:{responsive:true,animation:false,scales:{x:{ticks:{color:'#9aa1b3',maxTicksLimit:8}},y:{ticks:{color:'#9aa1b3'}}},plugins:{legend:{display:false}}}
    });
  }
  sel.addEventListener('change',e=>draw(e.target.value));
  draw(Object.keys(D.series)[0]);

  new Chart(document.getElementById('predbar'),{
    type:'bar',data:{labels:D.predictions.map(p=>p.ticker),
      datasets:[{label:'P(up)',data:D.predictions.map(p=>p.pred_up_proba),
        backgroundColor:D.predictions.map(p=>p.pred_up_proba>=0.5?'#34d399':'#f87171')}]},
    options:{responsive:true,animation:false,scales:{y:{min:0,max:1,ticks:{color:'#9aa1b3'}},x:{ticks:{color:'#9aa1b3'}}},plugins:{legend:{display:false}}}
  });
}
</script>
</body>
</html>
"""


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df, preds, metrics = load()
    html = build_html(df, preds, metrics)
    out = REPORTS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
