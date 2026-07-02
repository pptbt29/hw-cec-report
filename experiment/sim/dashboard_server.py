"""Interactive local dashboard server.

Unlike ``sim.dashboard``, this module does not run the simulation at startup.
It first serves a configuration page, lets the user review/edit the complete
experiment JSON, and only runs ``run_experiments`` after the user confirms.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import urlparse

from .config import default_config, from_dict, load_config, to_dict
from .dashboard import export_json, render_html, run_experiments


_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CEC-LLM 模拟控制台</title>
<style>
  :root{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;--border:#d0d7de;
    --fg:#1f2328;--muted:#656d76;--accent:#0969da;--good:#1a7f37;--bad:#cf222e;--best:#dafbe1}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:18px 24px;background:var(--panel);border-bottom:1px solid var(--border);
    display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap}
  h1{font-size:20px;margin:0 0 6px}
  .sub{font-size:13px;color:var(--muted)}
  main{padding:18px 24px;max-width:1680px;margin:0 auto}
  .layout{display:grid;grid-template-columns:minmax(560px,1fr) minmax(520px,1fr);gap:16px}
  @media(max-width:1180px){.layout{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px}
  .panel h2{font-size:15px;margin:0 0 12px}
  label{font-size:12px;color:var(--muted);display:block;margin:10px 0 5px}
  input,select,textarea{width:100%;border:1px solid var(--border);border-radius:6px;
    background:#fff;color:var(--fg);padding:8px 9px;font:inherit;font-size:13px}
  textarea{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;min-height:620px;line-height:1.45}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .checks{display:flex;flex-wrap:wrap;gap:8px}
  .check{display:flex;align-items:center;gap:6px;background:var(--panel2);
    border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px}
  .check input{width:auto;margin:0}
  button{border:1px solid var(--border);border-radius:6px;background:var(--panel2);
    padding:8px 12px;cursor:pointer;font:inherit;font-size:13px}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  button:disabled{opacity:.55;cursor:not-allowed}
  .actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
  .status{font-size:13px;color:var(--muted);white-space:pre-wrap}
  .status.ok{color:var(--good)} .status.err{color:var(--bad)}
  .result{margin-top:16px;display:none}
  table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
  th,td{padding:8px;border-bottom:1px solid var(--border);text-align:right}
  th:first-child,td:first-child{text-align:left}
  td.best{background:var(--best);font-weight:650}
  .hint{font-size:12px;color:var(--muted);line-height:1.55}
  .group-list{display:flex;flex-direction:column;gap:12px;max-height:760px;overflow:auto;padding-right:4px}
  .model-groups{border:1px solid var(--border);border-radius:8px;background:var(--panel2)}
  .model-groups>summary{cursor:pointer;padding:11px 12px;font-size:13px;font-weight:650;
    display:flex;align-items:center;justify-content:space-between;gap:12px}
  .model-body{padding:0 10px 10px}
  .model-actions{margin-top:10px}
  .group-card{border:1px solid var(--border);border-radius:8px;background:#fff;padding:12px}
  .group-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
  .group-title{font-weight:650;font-size:13px}
  .group-sub{font-size:12px;color:var(--muted)}
  .field-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 10px}
  .field-grid label{margin:0 0 4px}
  .field-grid input,.field-grid select{padding:7px 8px;font-size:12px}
  .field-wide{grid-column:span 2}
  .dist-title{grid-column:1/-1;font-size:12px;font-weight:650;color:var(--fg);margin-top:4px;padding-top:8px;border-top:1px solid var(--border)}
  @media(max-width:680px){.field-grid{grid-template-columns:1fr 1fr}.field-wide{grid-column:1/-1}}
  .link-btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;
    min-width:190px;text-decoration:none;border-radius:8px;border:1px solid var(--accent);
    background:var(--accent);color:#fff;padding:10px 14px;font-size:13px;font-weight:650}
  .link-btn.secondary{background:var(--panel);color:var(--accent)}
  code{background:var(--panel2);padding:1px 4px;border-radius:4px}
</style>
</head>
<body>
<header>
  <div>
    <h1>CEC-LLM 卸载模拟控制台</h1>
    <div class="sub">先确认实验参数，再启动模拟；完成后在下方展示 metrics dashboard。</div>
  </div>
  <div class="actions">
    <button id="resetBtn">恢复初始配置</button>
    <button class="primary" id="runBtn">开始模拟</button>
  </div>
</header>
<main>
  <div class="layout">
    <section class="panel">
      <h2>常用参数</h2>
      <div class="row">
        <div><label>节点数</label><input id="numNodes" type="number" min="1"></div>
        <div><label>状态滞后 ms</label><input id="stalenessMs" type="number" min="0" step="1"></div>
      </div>
      <div class="row">
        <div><label>实验时长 ms</label><input id="durationMs" type="number" min="1" step="1000"></div>
        <div><label>随机种子</label><input id="seed" type="number" step="1"></div>
      </div>
      <div class="row">
        <div><label>移动比例</label><input id="mobilityRatio" type="number" min="0" max="1" step="0.01"></div>
        <div><label>移动起始占比</label><input id="mobilityStart" type="number" min="0" max="1" step="0.01"></div>
      </div>
      <div class="row">
        <div><label>Markov 驻留轮数</label><input id="mobilityResidency" type="number" min="0" step="1"></div>
        <div><label>Session 启动窗口占比</label><input id="sessionStartSpread" type="number" min="0" max="1" step="0.01"></div>
      </div>
      <div class="row">
        <div><label>Gamma</label><input id="gamma" type="number" min="0" step="0.05"></div>
        <div><label>SLA margin ms</label><input id="slaMargin" type="number" min="0" step="1"></div>
      </div>
      <div class="row">
        <div><label>Token id bytes</label><input id="tokenBytes" type="number" min="1" step="1"></div>
        <div><label>Visual bytes/token</label><input id="visualBytes" type="number" min="0" step="1"></div>
      </div>
      <div class="row">
        <div><label>Request overhead bytes</label><input id="reqOverhead" type="number" min="0" step="1"></div>
        <div><label>Response overhead bytes</label><input id="respOverhead" type="number" min="0" step="1"></div>
      </div>
      <label>移动粒度</label>
      <select id="mobilityGranularity">
        <option value="request">request：逐请求入口抖动</option>
        <option value="session">session：会话持续迁移</option>
        <option value="markov">markov：驻留后可能继续迁移</option>
      </select>
      <p class="hint"><b>移动比例定义：</b><code>request</code> 相对原始 home 逐请求抖动；<code>session</code> 每个 session 抽一次，迁到新入口后不动；<code>markov</code> 维护当前位置，至少驻留指定轮数后再按概率迁到其他节点。</p>
      <label>策略</label>
      <div class="checks" id="policyChecks"></div>
      <h2 style="margin-top:18px">Data Generator 分组</h2>
      <label>全局总并发用户数</label>
      <input id="totalConcurrency" type="number" readonly>
      <p class="hint">只读汇总值。每个配置组可选择直接填写各入口人数，或填写入口比例和该组总并发。</p>
      <div class="group-list" id="groupList"></div>
      <p class="hint">左侧只是快捷项；右侧 JSON 包含所有可配置参数，包括硬件、模型、链路、workload groups、策略和集群配置。</p>
      <div class="actions">
        <button id="syncToJson">快捷项写入 JSON</button>
        <button id="syncFromJson">从 JSON 读取快捷项</button>
      </div>
      <div id="status" class="status"></div>
    </section>
    <section class="panel">
      <h2>完整配置 JSON</h2>
      <textarea id="configText" spellcheck="false"></textarea>
      <div class="actions">
        <button id="formatBtn">格式化 JSON</button>
      </div>
    </section>
  </div>
  <section class="panel result" id="resultBox">
    <h2>模拟结果</h2>
    <div id="summary"></div>
    <div class="actions">
      <a class="link-btn secondary" id="jsonLink" href="/output/metrics.json" target="_blank">打开 Metrics JSON</a>
      <a class="link-btn" id="htmlLink" href="/output/dashboard.html" target="_blank">打开 Metrics Dashboard</a>
    </div>
  </section>
</main>
<script>
const INITIAL_CONFIG = __CONFIG__;
const POLICIES = ["nearest","greedy","long_term","long_term_kv"];
let currentConfig = structuredClone(INITIAL_CONFIG);

const $ = id => document.getElementById(id);
function setStatus(msg, cls=""){ const s=$("status"); s.textContent=msg; s.className="status "+cls; }
function readJson(){
  try { return JSON.parse($("configText").value); }
  catch(e){ throw new Error("JSON 解析失败: "+e.message); }
}
function writeJson(cfg){ $("configText").value = JSON.stringify(cfg, null, 2); }
function distValue(group, key, fallback){
  return group.output_dist ? group.output_dist[key] : fallback;
}
function allocatedEntryCounts(group, nodeCount){
  const clamp = v => Math.max(1, Math.min(256, Math.round(v)));
  if((group.entry_mode||"counts")==="counts"){
    const values = (group.entry_concurrency||[]).slice(0,nodeCount).map(v=>clamp(Number(v)));
    while(values.length<nodeCount) values.push(1);
    return values;
  }
  const total=Math.max(nodeCount,Math.min(256*nodeCount,Math.round(Number(group.concurrency||nodeCount))));
  let ratios=(group.entry_ratios||[]).slice(0,nodeCount).map(v=>Math.max(Number(v)||0,0));
  while(ratios.length<nodeCount) ratios.push(0);
  if(ratios.reduce((a,b)=>a+b,0)<=0) ratios=Array(nodeCount).fill(1);
  const ratioSum=ratios.reduce((a,b)=>a+b,0),remaining=total-nodeCount;
  const raw=ratios.map(v=>remaining*v/ratioSum);
  const counts=raw.map(v=>1+Math.min(255,Math.floor(v)));
  let left=total-counts.reduce((a,b)=>a+b,0);
  const order=raw.map((v,i)=>[i,v-Math.floor(v)]).sort((a,b)=>b[1]-a[1]).map(x=>x[0]);
  while(left>0){
    let changed=false;
    for(const i of order){
      if(counts[i]>=256) continue;
      counts[i]++; left--; changed=true;
      if(left===0) break;
    }
    if(!changed) break;
  }
  return counts;
}
function groupFromRow(row){
  const parseList = key => {
    const raw=row.querySelector(`[data-k="${key}"]`)?.value.trim()||"";
    return raw?raw.split(",").map(x=>Number(x.trim())).filter(Number.isFinite):null;
  };
  return {
    entry_mode:row.querySelector('[data-k="entry_mode"]')?.value||"counts",
    entry_concurrency:parseList("entry_concurrency"),
    entry_ratios:parseList("entry_ratios"),
    concurrency:Number(row.querySelector('[data-k="concurrency"]')?.value)||1,
  };
}
function calculateGlobalConcurrency(){
  const nodeCount=Math.max(Number($("numNodes").value)||1,1);
  let total=0;
  document.querySelectorAll("#groupList .group-card").forEach(row=>{
    total+=allocatedEntryCounts(groupFromRow(row),nodeCount).reduce((a,b)=>a+b,0);
  });
  return total;
}
function updateConcurrencySummary(){
  $("totalConcurrency").value=calculateGlobalConcurrency();
  const nodeCount=Math.max(Number($("numNodes").value)||1,1);
  document.querySelectorAll("#groupList .group-card").forEach(row=>{
    const counts=allocatedEntryCounts(groupFromRow(row),nodeCount);
    const out=row.querySelector("[data-effective-counts]");
    if(out) out.textContent=`实际入口并发 ${counts.join(",")} · 合计 ${counts.reduce((a,b)=>a+b,0)}`;
  });
}
function syncEntryMode(row){
  const mode=row.querySelector('[data-k="entry_mode"]')?.value||"counts";
  row.querySelector('[data-field="counts"]').hidden=mode!=="counts";
  row.querySelector('[data-field="ratios"]').hidden=mode!=="ratios";
  const total=row.querySelector('[data-k="concurrency"]');
  total.readOnly=mode==="counts";
  if(mode==="counts"){
    const values=groupFromRow(row).entry_concurrency||[];
    total.value=values.reduce((a,b)=>a+b,0)||1;
  }
}
function renderGroups(cfg){
  const groups = cfg.workload?.groups || [];
  const nodeCount = Math.max(Number(cfg.cluster?.num_nodes ?? 3),1);
  const field = (label, html, wide=false) =>
    `<div class="${wide ? "field-wide" : ""}"><label>${label}</label>${html}</div>`;
  const modelNames=[...new Set([...(cfg.models||[]).map(m=>m.name),...groups.map(g=>g.model_name)])];
  const groupCard=(g,i)=>{
    const entry = (g.entry_concurrency ?? []).join(",");
    const ratios = (g.entry_ratios ?? []).join(",");
    const mode = g.entry_mode || (g.entry_ratios ? "ratios" : "counts");
    const total = allocatedEntryCounts({...g,entry_mode:mode},nodeCount).reduce((a,b)=>a+b,0);
    return `<div class="group-card" data-i="${i}">`+
      `<div class="group-head"><div><div class="group-title">${g.name ?? g.priority ?? "default"}</div>`+
      `<div class="group-sub" data-effective-counts>实际入口并发 ${allocatedEntryCounts({...g,entry_mode:mode},nodeCount).join(",")} · 合计 ${total}</div></div>`+
      `<button type="button" data-action="delete-group" title="删除配置组">删除</button></div>`+
      `<div class="field-grid">`+
      field("组名", `<input data-k="name" value="${g.name ?? g.priority ?? "default"}">`, true)+
      field("入口配置方式", `<select data-k="entry_mode"><option value="counts">具体数量</option><option value="ratios">比例 + 组总并发</option></select>`)+
      `<div class="field-wide" data-field="counts"><label>各入口并发用户数</label><input data-k="entry_concurrency" value="${entry}" placeholder="16,24,32"></div>`+
      `<div class="field-wide" data-field="ratios"><label>各入口比例</label><input data-k="entry_ratios" value="${ratios}" placeholder="1,1,2"></div>`+
      field("组总并发", `<input data-k="concurrency" type="number" min="${nodeCount}" step="1" value="${g.concurrency ?? total}">`)+
      field("到达率 req/s", `<input data-k="arrival_rate" type="number" min="0" step="0.01" value="${g.arrival_rate ?? ""}" placeholder="auto">`)+
      field("SLA ms", `<input data-k="sla_ms" type="number" min="1" step="1" value="${g.sla_ms ?? ""}">`)+
      field("轮数均值", `<input data-k="turns_mean" type="number" min="1" step="0.1" value="${g.turns_mean ?? 4}">`)+
      field("共享 prefix", `<input data-k="shared_prefix_tokens" type="number" min="0" step="1" value="${g.shared_prefix_tokens ?? 0}">`)+
      field("history growth", `<input data-k="history_growth" type="number" min="0" step="0.05" value="${g.history_growth ?? 0.6}">`)+
      `<div class="dist-title">Prompt 分布</div>`+
      field("kind", `<select data-k="prompt_kind"><option value="fixed">fixed</option><option value="normal">normal</option><option value="lognormal">lognormal</option></select>`)+
      field("mean", `<input data-k="prompt_mean" type="number" min="1" step="1" value="${g.prompt_dist?.mean ?? 128}">`)+
      field("std", `<input data-k="prompt_std" type="number" min="0" step="1" value="${g.prompt_dist?.std ?? 64}">`)+
      field("min", `<input data-k="prompt_minimum" type="number" min="1" step="1" value="${g.prompt_dist?.minimum ?? 1}">`)+
      field("max", `<input data-k="prompt_maximum" type="number" min="1" step="1" value="${g.prompt_dist?.maximum ?? 4096}">`)+
      `<div class="dist-title">Output 分布</div>`+
      field("kind", `<select data-k="output_kind"><option value="">model default</option><option value="fixed">fixed</option><option value="normal">normal</option><option value="lognormal">lognormal</option></select>`)+
      field("mean", `<input data-k="output_mean" type="number" min="1" step="1" value="${distValue(g,"mean","")}">`)+
      field("std", `<input data-k="output_std" type="number" min="0" step="1" value="${distValue(g,"std","")}">`)+
      field("min", `<input data-k="output_minimum" type="number" min="1" step="1" value="${distValue(g,"minimum","")}">`)+
      field("max", `<input data-k="output_maximum" type="number" min="1" step="1" value="${distValue(g,"maximum","")}">`)+
      `</div></div>`;
  };
  $("groupList").innerHTML = modelNames.map(modelName=>{
    const indexes=groups.map((g,i)=>g.model_name===modelName?i:-1).filter(i=>i>=0);
    return `<details class="model-groups" data-model="${modelName}"><summary>`+
      `<span>${modelName}</span><span class="group-sub">${indexes.length} 个配置组</span></summary>`+
      `<div class="model-body">${indexes.map(i=>groupCard(groups[i],i)).join("")}`+
      `<div class="model-actions"><button type="button" data-action="add-group" data-model="${modelName}">新增组</button></div></div></details>`;
  }).join("");
  document.querySelectorAll("#groupList .group-card").forEach(row=>{
    const g = groups[Number(row.dataset.i)] || {};
    row.querySelector('[data-k="prompt_kind"]').value = g.prompt_dist?.kind ?? "lognormal";
    row.querySelector('[data-k="output_kind"]').value = g.output_dist?.kind ?? "";
    row.querySelector('[data-k="entry_mode"]').value = g.entry_mode || (g.entry_ratios ? "ratios" : "counts");
    syncEntryMode(row);
  });
  updateConcurrencySummary();
}
function applyGroups(cfg){
  const groups = cfg.workload?.groups || [];
  document.querySelectorAll("#groupList .group-card").forEach(row=>{
    const i = Number(row.dataset.i);
    const g = groups[i];
    if(!g) return;
    const outputDefault = row.querySelector('[data-k="output_kind"]')?.value === "";
    row.querySelectorAll("input,select").forEach(input=>{
      const k = input.dataset.k;
      const raw = input.value.trim();
      if(k === "name" || k === "entry_mode"){ g[k] = raw; return; }
      if(k === "entry_concurrency" || k === "entry_ratios"){
        g[k] = raw ? raw.split(",").map(x=>Number(x.trim())).filter(x=>Number.isFinite(x)) : null;
        return;
      }
      if(k === "arrival_rate" || k === "sla_ms"){ g[k] = raw === "" ? null : Number(raw); return; }
      if(k === "prompt_kind"){ g.prompt_dist = g.prompt_dist || {}; g.prompt_dist.kind = raw; return; }
      if(k === "prompt_mean"){ g.prompt_dist = g.prompt_dist || {}; g.prompt_dist.mean = Number(raw); return; }
      if(k === "prompt_std"){ g.prompt_dist = g.prompt_dist || {}; g.prompt_dist.std = Number(raw); return; }
      if(k === "prompt_minimum"){ g.prompt_dist = g.prompt_dist || {}; g.prompt_dist.minimum = Number(raw); return; }
      if(k === "prompt_maximum"){ g.prompt_dist = g.prompt_dist || {}; g.prompt_dist.maximum = Number(raw); return; }
      if(k === "output_kind"){
        if(raw === ""){ g.output_dist = null; return; }
        g.output_dist = g.output_dist || {kind:"lognormal", mean:128, std:64, minimum:1, maximum:4096};
        g.output_dist.kind = raw;
        return;
      }
      if(k === "output_mean" || k === "output_std" || k === "output_minimum" || k === "output_maximum"){
        if(outputDefault) return;
        if(raw === "" && !g.output_dist) return;
        g.output_dist = g.output_dist || {kind:"lognormal", mean:128, std:64, minimum:1, maximum:4096};
        const field = k.replace("output_", "");
        g.output_dist[field] = Number(raw);
        return;
      }
      g[k] = Number(raw);
    });
    g.priority=g.name;
    if(g.entry_mode==="counts" && g.entry_concurrency?.length){
      g.concurrency = g.entry_concurrency.reduce((a,b)=>a+b,0);
    }
  });
  return cfg;
}
function fillPolicyChecks(cfg){
  $("policyChecks").innerHTML = POLICIES.map(p =>
    `<label class="check"><input type="checkbox" value="${p}"> ${p}</label>`
  ).join("");
  const enabled = new Set(cfg.policies || []);
  document.querySelectorAll("#policyChecks input").forEach(x => x.checked = enabled.has(x.value));
}
function fillQuick(cfg){
  $("numNodes").value = cfg.cluster?.num_nodes ?? 3;
  $("stalenessMs").value = cfg.cluster?.staleness_ms ?? 0;
  $("durationMs").value = cfg.workload?.duration_ms ?? 60000;
  $("seed").value = cfg.workload?.seed ?? 0;
  $("mobilityRatio").value = cfg.workload?.mobility_ratio ?? 0.2;
  $("mobilityStart").value = cfg.workload?.mobility_start_frac ?? 0.5;
  $("sessionStartSpread").value = cfg.workload?.session_start_spread_frac ?? 0.8;
  $("mobilityGranularity").value = cfg.workload?.mobility_granularity ?? "request";
  $("mobilityResidency").value = cfg.workload?.mobility_residency_turns ?? 2;
  $("gamma").value = cfg.router?.gamma ?? 0.9;
  $("slaMargin").value = cfg.router?.sla_margin_ms ?? 20;
  $("tokenBytes").value = cfg.router?.token_id_bytes ?? cfg.router?.request_bytes_per_token ?? 4;
  $("visualBytes").value = cfg.router?.visual_bytes_per_token ?? 0;
  $("reqOverhead").value = cfg.router?.request_overhead_bytes ?? 4096;
  $("respOverhead").value = cfg.router?.response_overhead_bytes ?? 4096;
  fillPolicyChecks(cfg);
  renderGroups(cfg);
}
function applyQuick(cfg){
  cfg.cluster = cfg.cluster || {};
  cfg.router = cfg.router || {};
  cfg.workload = cfg.workload || {};
  cfg.cluster.num_nodes = Number($("numNodes").value);
  cfg.cluster.staleness_ms = Number($("stalenessMs").value);
  cfg.workload.duration_ms = Number($("durationMs").value);
  cfg.workload.seed = Number($("seed").value);
  cfg.workload.mobility_ratio = Number($("mobilityRatio").value);
  cfg.workload.mobility_start_frac = Number($("mobilityStart").value);
  cfg.workload.session_start_spread_frac = Number($("sessionStartSpread").value);
  cfg.workload.mobility_granularity = $("mobilityGranularity").value;
  cfg.workload.mobility_residency_turns = Number($("mobilityResidency").value);
  delete cfg.workload.entry_concurrency_multiplier;
  cfg.router.gamma = Number($("gamma").value);
  cfg.router.sla_margin_ms = Number($("slaMargin").value);
  cfg.router.token_id_bytes = Number($("tokenBytes").value);
  cfg.router.visual_bytes_per_token = Number($("visualBytes").value);
  cfg.router.request_overhead_bytes = Number($("reqOverhead").value);
  cfg.router.response_overhead_bytes = Number($("respOverhead").value);
  delete cfg.router.request_bytes_per_token;
  delete cfg.router.response_bytes_per_token;
  cfg.policies = [...document.querySelectorAll("#policyChecks input:checked")].map(x=>x.value);
  return applyGroups(cfg);
}
function init(){
  writeJson(currentConfig);
  fillQuick(currentConfig);
  $("formatBtn").onclick = () => { try{ writeJson(readJson()); setStatus("JSON 已格式化","ok"); }catch(e){ setStatus(e.message,"err"); } };
  $("syncToJson").onclick = () => { try{ const cfg=applyQuick(readJson()); writeJson(cfg); setStatus("快捷项已写入 JSON","ok"); }catch(e){ setStatus(e.message,"err"); } };
  $("syncFromJson").onclick = () => { try{ fillQuick(readJson()); setStatus("已从 JSON 读取快捷项","ok"); }catch(e){ setStatus(e.message,"err"); } };
  $("resetBtn").onclick = () => { currentConfig=structuredClone(INITIAL_CONFIG); writeJson(currentConfig); fillQuick(currentConfig); setStatus("已恢复初始配置","ok"); };
  $("numNodes").oninput = updateConcurrencySummary;
  $("groupList").addEventListener("input", ev=>{
    if(["entry_concurrency","entry_ratios","concurrency"].includes(ev.target?.dataset?.k)){
      const row=ev.target.closest(".group-card");
      if(row && row.querySelector('[data-k="entry_mode"]').value==="counts") syncEntryMode(row);
      updateConcurrencySummary();
    }
  });
  $("groupList").addEventListener("change", ev=>{
    if(ev.target?.dataset?.k==="entry_mode"){
      syncEntryMode(ev.target.closest(".group-card"));
      updateConcurrencySummary();
    }
  });
  $("groupList").addEventListener("click", ev=>{
    const action=ev.target?.dataset?.action;
    if(!action) return;
    try{
      const cfg=applyGroups(readJson());
      const groups=cfg.workload.groups;
      if(action==="delete-group"){
        groups.splice(Number(ev.target.closest(".group-card").dataset.i),1);
      }else if(action==="add-group"){
        const modelName=ev.target.dataset.model;
        const source=groups.find(g=>g.model_name===modelName);
        const next=source?structuredClone(source):{
          model_name:modelName,name:"default",entry_mode:"counts",concurrency:3,
          entry_concurrency:[1,1,1],entry_ratios:null,sla_ms:null,arrival_rate:null,
          prompt_dist:{kind:"lognormal",mean:128,std:64,minimum:1,maximum:4096},
          output_dist:null,turns_mean:4,turns_min:1,turns_max:12,
          image_size:[0,0],num_frames:1,shared_prefix_tokens:0,history_growth:.6
        };
        const count=groups.filter(g=>g.model_name===modelName).length+1;
        next.name=`group-${count}`; next.priority=next.name;
        groups.push(next);
      }
      writeJson(cfg); renderGroups(cfg);
    }catch(e){setStatus(e.message,"err");}
  });
  $("runBtn").onclick = runSimulation;
}
function bestClass(v, best){ return Math.abs(v-best)<1e-9 ? " class=\"best\"" : ""; }
function summaryTable(models){
  let rows = "";
  for(const [model, per] of Object.entries(models || {})){
    const vals = Object.values(per);
    const best = {
      avg: Math.min(...vals.map(m=>m.avg_e2e_ms)),
      p99: Math.min(...vals.map(m=>m.p99_ttft_ms)),
      xnode: Math.min(...vals.map(m=>m.cross_node_ratio)),
      moves: Math.min(...vals.map(m=>m.mobility_transition_count)),
      mig: Math.min(...vals.map(m=>m.migrate_bytes_mb)),
      owner: Math.min(...vals.map(m=>m.owner_switch_count)),
    };
    for(const [policy, m] of Object.entries(per)){
      rows += `<tr><td>${model}</td><td>${policy}</td>`+
        `<td${bestClass(m.avg_e2e_ms,best.avg)}>${m.avg_e2e_ms.toFixed(1)}</td>`+
        `<td${bestClass(m.p99_ttft_ms,best.p99)}>${m.p99_ttft_ms.toFixed(1)}</td>`+
        `<td${bestClass(m.cross_node_ratio,best.xnode)}>${(m.cross_node_ratio*100).toFixed(1)}%</td>`+
        `<td${bestClass(m.mobility_transition_count,best.moves)}>${m.mobility_transition_count}</td>`+
        `<td${bestClass(m.migrate_bytes_mb,best.mig)}>${m.migrate_bytes_mb.toFixed(1)}</td>`+
        `<td${bestClass(m.owner_switch_count,best.owner)}>${m.owner_switch_count}</td></tr>`;
    }
  }
  return `<table><thead><tr><th>模型</th><th>策略</th><th>Avg E2E</th><th>P99 TTFT</th><th>跨节点</th><th>入口迁移</th><th>迁移 MB</th><th>Owner 切换</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function latencyBreakdownTable(models){
  let rows = "";
  for(const [model, per] of Object.entries(models || {})){
    const keys = ["avg_e2e_ms","avg_request_network_ms","avg_queue_prefill_ms",
      "avg_queue_recompute_ms","avg_queue_decode_ms","avg_migration_ms",
      "avg_recompute_ms","avg_prefill_ms","avg_decode_ms","avg_response_network_ms",
      "avg_e2e_component_sum_ms"];
    const best = Object.fromEntries(keys.map(k=>[k,Math.min(...Object.values(per).map(m=>m[k]))]));
    for(const [policy, m] of Object.entries(per)){
      rows += `<tr><td>${model}</td><td>${policy}</td>`+
        keys.map(k=>`<td${bestClass(m[k],best[k])}>${m[k].toFixed(2)}</td>`).join("")+
        `</tr>`;
    }
  }
  return `<h2 style="margin-top:22px">Avg E2E 延迟分拆</h2>`+
    `<p class="hint">单位均为 ms/请求；分项合计应与 Avg E2E 一致，每列最优值高亮。</p>`+
    `<div style="overflow-x:auto"><table><thead><tr><th>模型</th><th>策略</th><th>Avg E2E</th>`+
    `<th>请求转发</th><th>排队·Prefill</th><th>排队·重算</th><th>排队·Decode</th>`+
    `<th>KV 迁移</th><th>KV 重算</th><th>Prefill</th><th>Decode</th>`+
    `<th>响应回传</th><th>分项合计</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function migrateReasonTable(models){
  const labels={
    entry_sla:"入口 SLA",entry_memory:"入口 Memory",
    entry_sla_and_memory:"入口 SLA+Memory",
    entry_mixed_constraints:"入口混合约束",
    immediate_cost:"即时成本",future_cost:"FutureCost",unclassified:"未分类"
  };
  let rows="";
  for(const [model,per] of Object.entries(models||{})){
    for(const [policy,m] of Object.entries(per)){
      const counts=m.migrate_reason_counts||{},bytes=m.migrate_reason_bytes_mb||{};
      for(const [reason,count] of Object.entries(counts)){
        rows+=`<tr><td>${model}</td><td>${policy}</td><td>${labels[reason]||reason}</td>`+
          `<td>${count}</td><td>${(bytes[reason]||0).toFixed(2)}</td></tr>`;
      }
    }
  }
  return `<h2 style="margin-top:22px">Migrate 原因拆分</h2>`+
    `<div style="overflow-x:auto"><table><thead><tr><th>模型</th><th>策略</th><th>原因</th>`+
    `<th>次数</th><th>迁移 MB</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
async function runSimulation(){
  let cfg;
  try { cfg = applyQuick(readJson()); writeJson(cfg); }
  catch(e){ setStatus(e.message,"err"); return; }
  $("runBtn").disabled = true;
  setStatus("模拟运行中，请稍候...");
  $("resultBox").style.display = "none";
  try{
    const res = await fetch("/run", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(cfg)
    });
    const body = await res.json();
    if(!res.ok) throw new Error(body.error || "模拟失败");
    setStatus("模拟完成","ok");
    $("summary").innerHTML = summaryTable(body.models) + latencyBreakdownTable(body.models) +
      migrateReasonTable(body.models);
    $("jsonLink").href = "/output/metrics.json?t=" + Date.now();
    $("htmlLink").href = "/output/dashboard.html?t=" + Date.now();
    $("resultBox").style.display = "block";
    $("resultBox").scrollIntoView({behavior:"smooth", block:"start"});
  }catch(e){
    setStatus(e.message,"err");
  }finally{
    $("runBtn").disabled = false;
  }
}
init();
</script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _text_response(
    handler: BaseHTTPRequestHandler,
    code: int,
    payload: str,
    content_type: str = "text/html; charset=utf-8",
) -> None:
    data = payload.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class _DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler, initial_config: Dict, out_dir: str):
        super().__init__(server_address, handler)
        self.initial_config = initial_config
        self.out_dir = out_dir
        self.run_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    server: _DashboardServer

    def log_message(self, fmt: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            cfg = json.dumps(self.server.initial_config, ensure_ascii=False)
            _text_response(self, 200, _PAGE.replace("__CONFIG__", cfg))
            return
        if path == "/api/config":
            _json_response(self, 200, self.server.initial_config)
            return
        if path in ("/output/dashboard.html", "/output/metrics.json"):
            name = os.path.basename(path)
            file_path = os.path.join(self.server.out_dir, name)
            if not os.path.exists(file_path):
                _text_response(self, 404, "not generated yet", "text/plain; charset=utf-8")
                return
            ctype = "application/json; charset=utf-8" if name.endswith(".json") else "text/html; charset=utf-8"
            with open(file_path, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _text_response(self, 404, "not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/run":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            cfg_dict = json.loads(raw)
            experiment = from_dict(cfg_dict)
        except Exception as exc:
            _json_response(self, 400, {"error": f"配置无效: {exc}"})
            return

        if not self.server.run_lock.acquire(blocking=False):
            _json_response(self, 409, {"error": "已有模拟正在运行，请稍后再试"})
            return
        try:
            data = run_experiments(experiment)
            os.makedirs(self.server.out_dir, exist_ok=True)
            json_path = os.path.join(self.server.out_dir, "metrics.json")
            html_path = os.path.join(self.server.out_dir, "dashboard.html")
            export_json(data, json_path)
            render_html(data, html_path)
            _json_response(self, 200, {
                "ok": True,
                "models": data["models"],
                "meta": data["meta"],
                "json_url": "/output/metrics.json",
                "dashboard_url": "/output/dashboard.html",
            })
        except Exception as exc:
            _json_response(self, 500, {"error": f"模拟失败: {exc}"})
        finally:
            self.server.run_lock.release()


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: Optional[str] = None,
    open_browser: bool = False,
) -> None:
    cfg = load_config(config_path) if config_path else default_config()
    initial = to_dict(cfg)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    httpd = _DashboardServer((host, port), _Handler, initial, out_dir)
    url = f"http://{host}:{port}/"
    print(f"interactive dashboard -> {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down dashboard server")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, args.config, args.open)


if __name__ == "__main__":
    main()
