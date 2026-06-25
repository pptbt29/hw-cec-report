"""Metrics dashboard generator.

Runs the four offloading policies across one or more models on the same
workload trace, collects per-request records and aggregate metrics, and emits
a single self-contained interactive HTML dashboard (no third-party deps).

Usage:
    python -m sim.dashboard                 # default experiment -> output/dashboard.html
    python -m sim.dashboard --open          # also open in browser
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .compute_simulator import get_hardware
from .data_generator import DataGenerator, WorkloadConfig
from .large_model import get_model
from .network import NetworkSimulator, default_topology
from .router import Policy, simulate_trace


def run_experiments(
    models: Optional[List[str]] = None,
    config: Optional[WorkloadConfig] = None,
    hardware_name: str = "A800T-A2",
    num_nodes: int = 3,
    staleness_ms: float = 0.0,
) -> Dict:
    """Run all policies for each model on one shared trace; collect metrics."""
    config = config or WorkloadConfig.default_experiment()
    requests = DataGenerator(config).generate()
    hw = get_hardware(hardware_name)

    present = []
    seen = set()
    for r in requests:
        if r.model_name not in seen:
            seen.add(r.model_name)
            present.append(r.model_name)
    models = models or present

    data: Dict = {
        "meta": {
            "hardware": hardware_name,
            "num_nodes": num_nodes,
            "staleness_ms": staleness_ms,
            "duration_ms": config.duration_ms,
            "mobility_ratio": config.mobility_ratio,
            "mobility_start_frac": config.mobility_start_frac,
            "total_requests": len(requests),
        },
        "models": {},
    }

    for model_name in models:
        model = get_model(model_name)
        per_policy = {}
        for pol in Policy:
            net = NetworkSimulator(default_topology(num_nodes))
            res = simulate_trace(
                pol, requests, model, hw, net,
                num_nodes=num_nodes, staleness_ms=staleness_ms,
                collect_records=True,
            )
            per_policy[pol.value] = res
        data["models"][model_name] = per_policy
    return data


def export_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def render_html(data: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__DATA__", payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CEC-LLM 卸载模拟 Metrics Dashboard</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --border:#30363d;
    --fg:#e6edf3; --muted:#8b949e; --grid:#21262d;
    --nearest:#8b949e; --greedy:#f0883e; --long_term:#3fb950; --long_term_kv:#58a6ff;
    --good:#3fb950; --bad:#f85149; --warn:#d29922;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
  header{padding:20px 28px;border-bottom:1px solid var(--border);
    background:linear-gradient(180deg,#161b22,#0d1117);}
  header h1{margin:0;font-size:20px;font-weight:650}
  header .sub{color:var(--muted);font-size:13px;margin-top:6px}
  .wrap{padding:20px 28px;max-width:1280px;margin:0 auto}
  .tabs{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
  .tab{padding:7px 16px;border:1px solid var(--border);border-radius:8px;
    background:var(--panel);color:var(--muted);cursor:pointer;font-size:13px;transition:.15s}
  .tab:hover{color:var(--fg);border-color:#484f58}
  .tab.active{background:var(--panel2);color:var(--fg);border-color:#58a6ff}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 18px;font-size:12px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:11px;height:11px;border-radius:3px;display:inline-block}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
  .card .k{color:var(--muted);font-size:12px}
  .card .v{font-size:22px;font-weight:650;margin-top:6px}
  .card .d{font-size:12px;margin-top:4px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin-bottom:18px}
  .panel h3{margin:0 0 4px;font-size:15px}
  .panel .hint{color:var(--muted);font-size:12px;margin-bottom:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  thead th{color:var(--muted);font-weight:600;border-bottom:1px solid var(--border)}
  tbody tr:hover{background:var(--panel2)}
  .best{color:var(--good);font-weight:650}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .axis{stroke:var(--grid);stroke-width:1}
  .axlab{fill:var(--muted);font-size:10px}
  text{font-family:inherit}
  footer{color:var(--muted);font-size:12px;padding:20px 28px;border-top:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <h1>CEC-LLM 推理卸载 · 模拟 Metrics Dashboard</h1>
  <div class="sub" id="meta"></div>
</header>
<div class="wrap">
  <div class="tabs" id="modelTabs"></div>
  <div class="legend" id="legend"></div>
  <div class="cards" id="cards"></div>

  <div class="panel">
    <h3>策略汇总对比</h3>
    <div class="hint">同一条请求轨迹下四种策略的关键指标；每列最优值高亮（绿色）。</div>
    <div id="summaryTable"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>关键指标对比（柱状）</h3>
      <div class="hint">越低越好：P99 TTFT、跨节点比例、迁移字节、owner 切换。</div>
      <div id="barCharts"></div>
    </div>
    <div class="panel">
      <h3>状态获取动作分布</h3>
      <div class="hint">local 命中 / migrate 迁移 / recompute 重算 / fresh 新建。</div>
      <div id="actionDist"></div>
    </div>
  </div>

  <div class="panel">
    <h3>P99 TTFT 时间序列</h3>
    <div class="hint">按到达时间分桶的滚动 P99 TTFT；竖线为用户移动起始时刻。</div>
    <div id="ttftSeries"></div>
  </div>

  <div class="panel">
    <h3>累计跨节点请求（状态黏附曲线）</h3>
    <div class="hint">移动后曲线越陡，说明请求越被吸到远端节点（黏附越严重）。</div>
    <div id="stickiness"></div>
  </div>

  <div class="panel">
    <h3>链路利用率</h3>
    <div class="hint">100G 直连 vs 25G 跨主机 RDMA 在各策略下的占用与传输字节。</div>
    <div id="linkUtil"></div>
  </div>
</div>
<footer>由 <code>sim/dashboard.py</code> 离线生成 · 纯标准库 · 数据内联，可直接分享此 HTML。</footer>

<script>
const DATA = __DATA__;
const POLICIES = ["nearest","greedy","long_term","long_term_kv"];
const PLABEL = {nearest:"Nearest(基线)",greedy:"Greedy",long_term:"Long-term",long_term_kv:"Long-term+KV"};
const PCOLOR = {nearest:"#8b949e",greedy:"#f0883e",long_term:"#3fb950",long_term_kv:"#58a6ff"};
let currentModel = Object.keys(DATA.models)[0];

const meta = DATA.meta;
document.getElementById("meta").innerHTML =
  `硬件 ${meta.hardware} · ${meta.num_nodes} 节点 · 时长 ${(meta.duration_ms/1000).toFixed(0)}s · `+
  `移动比例 ${(meta.mobility_ratio*100).toFixed(0)}% (起于 ${(meta.mobility_start_frac*100).toFixed(0)}%) · `+
  `请求总数 ${meta.total_requests} · staleness ${meta.staleness_ms}ms`;

function tabs(){
  const el=document.getElementById("modelTabs"); el.innerHTML="";
  Object.keys(DATA.models).forEach(m=>{
    const d=document.createElement("div");
    d.className="tab"+(m===currentModel?" active":"");
    d.textContent=m; d.onclick=()=>{currentModel=m;render();}; el.appendChild(d);
  });
}
function legend(){
  document.getElementById("legend").innerHTML = POLICIES.map(p=>
    `<span><i class="dot" style="background:${PCOLOR[p]}"></i>${PLABEL[p]}</span>`).join("");
}

function fmt(x,d=1){return (typeof x==="number"&&isFinite(x))?x.toFixed(d):x;}
function el(tag,attrs,children){
  const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]);
  (children||[]).forEach(c=>e.appendChild(c)); return e;
}
function svg(w,h){return el("svg",{viewBox:`0 0 ${w} ${h}`});}
function txt(x,y,s,cls,anchor){const t=el("text",{x,y,class:cls||"axlab","text-anchor":anchor||"start"});t.textContent=s;return t;}

function cards(){
  const M=DATA.models[currentModel];
  const base=M.nearest, lt=M.long_term_kv;
  const reqs=base.num_requests;
  const e2eImp=(base.avg_e2e_ms-lt.avg_e2e_ms)/base.avg_e2e_ms*100;
  const xnodeImp=(base.cross_node_ratio-lt.cross_node_ratio)*100;
  const migImp=(base.migrate_bytes_mb-lt.migrate_bytes_mb)/Math.max(base.migrate_bytes_mb,1e-9)*100;
  const c=[
    ["请求数(该模型)",reqs,"",""],
    ["P99 TTFT (Greedy)",fmt(M.greedy.p99_ttft_ms)+" ms","",""],
    ["P99 TTFT (LT+KV)",fmt(lt.p99_ttft_ms)+" ms","",""],
    ["跨节点↓ vs 基线",fmt(xnodeImp,1)+" pp",xnodeImp>0?"good":"bad","黏附改善"],
    ["迁移字节↓ vs 基线",fmt(migImp,0)+"%",migImp>0?"good":"bad","状态搬运减少"],
    ["owner 切换 (LT+KV)",lt.owner_switch_count,"",""],
  ];
  document.getElementById("cards").innerHTML = c.map(x=>{
    const col=x[2]==="good"?"var(--good)":x[2]==="bad"?"var(--bad)":"var(--fg)";
    return `<div class="card"><div class="k">${x[0]}</div>`+
      `<div class="v" style="color:${col}">${x[1]}</div>`+
      `<div class="d" style="color:var(--muted)">${x[3]}</div></div>`;
  }).join("");
}

function summaryTable(){
  const M=DATA.models[currentModel];
  const rows=[
    ["avg E2E (ms)","avg_e2e_ms",1,"min"],
    ["P50 TTFT (ms)","p50_ttft_ms",1,"min"],
    ["P95 TTFT (ms)","p95_ttft_ms",1,"min"],
    ["P99 TTFT (ms)","p99_ttft_ms",1,"min"],
    ["SLA 违约率 (%)","sla_violation_ratio",2,"min",100],
    ["不可执行率 (%)","infeasible_ratio",2,"min",100],
    ["跨节点比例 (%)","cross_node_ratio",1,"min",100],
    ["migrate 次数","migrate_count",0,"min"],
    ["recompute 次数","recompute_count",0,"min"],
    ["owner 切换","owner_switch_count",0,"min"],
    ["迁移字节 (MB)","migrate_bytes_mb",1,"min"],
  ];
  let h="<table><thead><tr><th>指标</th>"+POLICIES.map(p=>`<th>${PLABEL[p]}</th>`).join("")+"</tr></thead><tbody>";
  rows.forEach(r=>{
    const vals=POLICIES.map(p=>M[p][r[1]]*(r[4]||1));
    const best=Math.min(...vals);
    h+=`<tr><td>${r[0]}</td>`+vals.map(v=>{
      const cls=Math.abs(v-best)<1e-9?"best":"";
      return `<td class="${cls}">${fmt(v,r[2])}</td>`;
    }).join("")+"</tr>";
  });
  h+="</tbody></table>";
  document.getElementById("summaryTable").innerHTML=h;
}

function barCharts(){
  const M=DATA.models[currentModel];
  const metrics=[
    ["P99 TTFT (ms)","p99_ttft_ms",1],
    ["跨节点比例 (%)","cross_node_ratio",100],
    ["迁移字节 (MB)","migrate_bytes_mb",1],
    ["owner 切换","owner_switch_count",1],
  ];
  const W=560,rowH=78,pad=120,barH=14,gap=4;
  const s=svg(W, metrics.length*rowH+10);
  metrics.forEach((m,mi)=>{
    const y0=mi*rowH+8;
    s.appendChild(txt(0,y0+10,m[0],"axlab"));
    const vals=POLICIES.map(p=>M[p][m[1]]*m[2]);
    const mx=Math.max(...vals,1e-9);
    POLICIES.forEach((p,pi)=>{
      const v=vals[pi], y=y0+18+pi*(barH+gap);
      const w=(W-pad-10)*v/mx;
      s.appendChild(el("rect",{x:pad,y,width:Math.max(w,1),height:barH,rx:3,fill:PCOLOR[p]}));
      s.appendChild(txt(pad-6,y+barH-2,PLABEL[p],"axlab","end"));
      s.appendChild(txt(pad+Math.max(w,1)+6,y+barH-2,fmt(v, m[2]===100?1:(m[1]==="migrate_bytes_mb"?0:1)),"axlab"));
    });
  });
  const box=document.getElementById("barCharts"); box.innerHTML=""; box.appendChild(s);
}

function actionDist(){
  const M=DATA.models[currentModel];
  const modes=[["local","#3fb950"],["migrate","#f0883e"],["recompute","#d29922"],["fresh","#8b949e"]];
  const W=560,rowH=46,pad=120;
  const s=svg(W,POLICIES.length*rowH+30);
  POLICIES.forEach((p,pi)=>{
    const r=M[p];
    const total=modes.reduce((a,m)=>a+(r[m[0]+"_count"]||0),0)||1;
    const y=pi*rowH+10; let x=pad;
    s.appendChild(txt(pad-6,y+18,PLABEL[p],"axlab","end"));
    modes.forEach(m=>{
      const c=r[m[0]+"_count"]||0, w=(W-pad-10)*c/total;
      if(w>0){
        s.appendChild(el("rect",{x,y:y+6,width:w,height:20,fill:m[1],rx:2}));
        if(w>26) s.appendChild(txt(x+w/2,y+20,c,"axlab","middle"));
      }
      x+=w;
    });
  });
  let lx=pad,ly=POLICIES.length*rowH+18;
  modes.forEach(m=>{
    s.appendChild(el("rect",{x:lx,y:ly-9,width:10,height:10,fill:m[1],rx:2}));
    s.appendChild(txt(lx+14,ly,m[0],"axlab")); lx+=100;
  });
  const box=document.getElementById("actionDist"); box.innerHTML=""; box.appendChild(s);
}

function bucketize(records,key,nbuckets,dur,agg){
  const b=Array.from({length:nbuckets},()=>[]);
  records.forEach(r=>{
    let i=Math.min(nbuckets-1,Math.floor(r.t/dur*nbuckets));
    if(i<0)i=0; b[i].push(r[key]);
  });
  return b.map(arr=>{
    if(!arr.length) return null;
    if(agg==="p99"){const s=[...arr].sort((a,b)=>a-b);return s[Math.min(s.length-1,Math.round(0.99*(s.length-1)))];}
    return arr.reduce((a,c)=>a+c,0)/arr.length;
  });
}

function lineChart(boxId,seriesByPolicy,dur,yLabel){
  const W=1180,H=300,L=56,R=18,T=14,B=34;
  const s=svg(W,H);
  let mx=0;
  for(const p of POLICIES) seriesByPolicy[p].forEach(v=>{if(v!=null&&v>mx)mx=v;});
  mx=mx||1;
  const n=seriesByPolicy[POLICIES[0]].length;
  const X=i=>L+(W-L-R)*(i/(Math.max(n-1,1)));
  const Y=v=>T+(H-T-B)*(1-v/mx);
  // grid + y labels
  for(let g=0;g<=4;g++){
    const yy=T+(H-T-B)*g/4, val=mx*(1-g/4);
    s.appendChild(el("line",{x1:L,y1:yy,x2:W-R,y2:yy,class:"axis"}));
    s.appendChild(txt(L-8,yy+3,fmt(val, mx<10?2:0),"axlab","end"));
  }
  // mobility line
  const mob=meta.mobility_start_frac;
  const mx2=L+(W-L-R)*mob;
  s.appendChild(el("line",{x1:mx2,y1:T,x2:mx2,y2:H-B,stroke:"#f85149","stroke-dasharray":"4 4","stroke-width":1}));
  s.appendChild(txt(mx2+4,T+10,"用户移动 →","axlab"));
  // x labels
  for(let g=0;g<=5;g++){
    const xx=L+(W-L-R)*g/5;
    s.appendChild(txt(xx,H-B+16,((dur/1000)*g/5).toFixed(0)+"s","axlab","middle"));
  }
  s.appendChild(txt(L-8,T-2,yLabel,"axlab","end"));
  POLICIES.forEach(p=>{
    const ser=seriesByPolicy[p]; let d=""; let started=false;
    ser.forEach((v,i)=>{ if(v==null)return; d+=(started?"L":"M")+X(i)+" "+Y(v)+" "; started=true; });
    s.appendChild(el("path",{d,fill:"none",stroke:PCOLOR[p],"stroke-width":2,"stroke-linejoin":"round"}));
  });
  const box=document.getElementById(boxId); box.innerHTML=""; box.appendChild(s);
}

function ttftSeries(){
  const M=DATA.models[currentModel], dur=meta.duration_ms, nb=40;
  const series={};
  POLICIES.forEach(p=>{series[p]=bucketize(M[p].records,"ttft",nb,dur,"p99");});
  lineChart("ttftSeries",series,dur,"ms");
}

function stickiness(){
  const M=DATA.models[currentModel], dur=meta.duration_ms, nb=60;
  const series={};
  POLICIES.forEach(p=>{
    const recs=[...M[p].records].sort((a,b)=>a.t-b.t);
    let cum=0; const arr=Array(nb).fill(null);
    // cumulative cross-node count sampled into buckets
    const cumAt=[]; recs.forEach(r=>{cum+=r.cross; cumAt.push([r.t,cum]);});
    for(let i=0;i<nb;i++){
      const t=dur*(i+1)/nb;
      let v=0; for(const [tt,cc] of cumAt){ if(tt<=t) v=cc; else break; }
      arr[i]=v;
    }
    series[p]=arr;
  });
  lineChart("stickiness",series,dur,"累计跨节点请求数");
}

function linkUtil(){
  const M=DATA.models[currentModel];
  const links=Object.keys(M.nearest.link_utilization||{});
  if(!links.length){document.getElementById("linkUtil").innerHTML="<span style='color:var(--muted)'>无链路数据</span>";return;}
  const W=560,pad=130,rowH=22;
  const s=svg(W, links.length*POLICIES.length*rowH + links.length*16 + 20);
  let y=6;
  links.forEach(lk=>{
    s.appendChild(txt(0,y+10,lk,"axlab"));
    y+=16;
    const mx=Math.max(...POLICIES.map(p=>(M[p].link_utilization[lk]||{}).total_bytes||0),1);
    POLICIES.forEach(p=>{
      const u=M[p].link_utilization[lk]||{};
      const bytes=u.total_bytes||0, w=(W-pad-70)*bytes/mx;
      s.appendChild(txt(pad-6,y+13,PLABEL[p],"axlab","end"));
      s.appendChild(el("rect",{x:pad,y:y+2,width:Math.max(w,1),height:14,rx:3,fill:PCOLOR[p]}));
      s.appendChild(txt(pad+Math.max(w,1)+6,y+13,(bytes/1e6).toFixed(0)+"MB","axlab"));
      y+=rowH;
    });
    y+=6;
  });
  const box=document.getElementById("linkUtil"); box.innerHTML=""; box.appendChild(s);
}

function render(){
  tabs(); legend(); cards(); summaryTable(); barCharts();
  actionDist(); ttftSeries(); stickiness(); linkUtil();
}
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    import webbrowser

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    html_path = os.path.join(out_dir, "dashboard.html")
    json_path = os.path.join(out_dir, "metrics.json")

    print("running experiments (4 policies x models on shared trace)...")
    data = run_experiments()
    export_json(data, json_path)
    render_html(data, html_path)

    for model_name, per in data["models"].items():
        print(f"\n[{model_name}]")
        print(f"  {'policy':<14}{'p99_ttft':>9}{'xnode%':>8}{'migrMB':>8}{'ownsw':>7}")
        for pol, m in per.items():
            print(f"  {pol:<14}{m['p99_ttft_ms']:9.1f}"
                  f"{m['cross_node_ratio']*100:8.1f}{m['migrate_bytes_mb']:8.1f}"
                  f"{m['owner_switch_count']:7d}")

    print(f"\nJSON  -> {json_path}")
    print(f"HTML  -> {html_path}")
    if "--open" in sys.argv:
        webbrowser.open("file://" + os.path.abspath(html_path))
