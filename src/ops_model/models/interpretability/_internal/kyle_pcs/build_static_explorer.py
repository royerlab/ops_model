#!/usr/bin/env python3
"""Build a static, self-contained PC Strip Explorer HTML file.

Embeds all crop images as base64 and all data as inline JSON,
producing a single HTML file that works from file:// or any static host.

Required inputs (all in --artifacts-dir):
  gene_names.json        — list of gene names
  gene_pc_scores.npy     — (n_genes, n_pcs) mean PC scores per gene
  gene_pc_analysis.json  — must contain "explained_variance" list
  representatives.json   — per-PC representative cells with metadata
  crops_png/             — cell crop PNGs named pc{NNN}_bin{NN}_row{N}.png

Usage:
  python3 build_static_explorer.py
  python3 build_static_explorer.py --artifacts-dir /path/to/data --output explorer.html
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def build(artifacts_dir, output_path):
    artifacts_dir = Path(artifacts_dir)
    crops_dir = artifacts_dir / "crops_png"

    with open(artifacts_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    scores = np.load(artifacts_dir / "gene_pc_scores.npy")
    with open(artifacts_dir / "gene_pc_analysis.json") as f:
        analysis = json.load(f)
    with open(artifacts_dir / "representatives.json") as f:
        reps_data = json.load(f)

    ev = analysis.get("explained_variance", [])

    overview = {
        "n_genes": len(gene_names),
        "n_pcs": scores.shape[1],
        "n_bins": reps_data.get("cells_per_row", 15),
        "n_rows": reps_data.get("n_rows", 3),
        "crop_size": reps_data.get("crop_size", 96),
        "total_variance": round(sum(ev) * 100, 1),
        "explained_variance": [round(v * 100, 2) for v in ev],
    }

    reps_by_pc = {}
    for r in reps_data["representatives"]:
        reps_by_pc.setdefault(r["pc"], []).append(r)

    pc_data = {}
    for pc_num in range(1, scores.shape[1] + 1):
        idx = pc_num - 1
        col = scores[:, idx]
        top_high = np.argsort(col)[::-1][:15].tolist()
        top_low = np.argsort(col)[:15].tolist()

        entries = reps_by_pc.get(idx, [])
        bins = {}
        for r in entries:
            bins.setdefault(r["bin"], []).append(r)

        strip = []
        for bi in sorted(bins.keys()):
            rows = sorted(bins[bi], key=lambda x: x["row"])
            strip.append({
                "bin": bi,
                "cells": [{
                    "gene": r["gene"],
                    "score": round(r["score"], 2),
                    "experiment": r["experiment"],
                    "well": r["well"],
                    "x": round(r["x"], 1),
                    "y": round(r["y"], 1),
                    "has_crop": r["has_crop"],
                    "img": f"pc{idx:03d}_bin{bi:02d}_row{r['row']}.png",
                } for r in rows]
            })

        pc_data[pc_num] = {
            "pc": pc_num,
            "explained_variance": round(ev[idx] * 100, 2) if idx < len(ev) else None,
            "high_genes": [{"gene": gene_names[i], "score": round(float(col[i]), 3)} for i in top_high],
            "low_genes": [{"gene": gene_names[i], "score": round(float(col[i]), 3)} for i in top_low],
            "strip": strip,
        }

    gene_data = {}
    for gi, name in enumerate(gene_names):
        profile = scores[gi].tolist()
        top_pcs = np.argsort(np.abs(scores[gi]))[::-1][:15].tolist()
        gene_data[name] = {
            "gene": name,
            "top_pcs": [{"pc": int(pc + 1), "score": round(float(profile[pc]), 3)} for pc in top_pcs],
        }

    print("Encoding crop images...")
    crop_b64 = {}
    for png in sorted(crops_dir.glob("*.png")):
        crop_b64[png.name] = base64.b64encode(png.read_bytes()).decode("ascii")
    print(f"  {len(crop_b64)} images encoded")

    print("Building HTML...")
    html = build_html(overview, pc_data, gene_data, gene_names, crop_b64)

    output_path = Path(output_path)
    output_path.write_text(html)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Written {output_path} ({size_mb:.1f} MB)")


def build_html(overview, pc_data, gene_data, gene_names, crop_b64):
    return r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PC Strip Explorer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #e6edf3; }

.app { display: grid; grid-template-columns: 280px 1fr; grid-template-rows: auto 1fr; height: 100vh; }
header { grid-column: 1/-1; padding: 10px 20px; background: #161b22; border-bottom: 1px solid #30363d;
         display: flex; align-items: center; gap: 16px; }
header h1 { font-size: 15px; white-space: nowrap; }
header .stats { font-size: 12px; color: #8b949e; }
.search-box { flex: 1; max-width: 360px; }
.search-box input { width: 100%; padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;
                    background: #0d1117; color: #e6edf3; font-size: 13px; outline: none; }
.search-box input:focus { border-color: #58a6ff; }

.sidebar { overflow-y: auto; border-right: 1px solid #30363d; background: #161b22; }
.main { overflow-y: auto; padding: 20px; }

.pc-list { list-style: none; padding: 8px; }
.pc-item { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; padding: 5px 8px;
           border-radius: 4px; cursor: pointer; font-size: 12px; font-family: monospace; }
.pc-item:hover { background: #1f2937; }
.pc-item.active { background: #1f6feb33; color: #58a6ff; }
.pc-item .pc-row { display: flex; align-items: center; gap: 6px; width: 100%; }
.pc-item .pc-label { width: 42px; flex-shrink: 0; }
.pc-item .var-bar-wrap { flex: 1; height: 10px; background: #21262d; border-radius: 3px; }
.pc-item .var-bar { height: 100%; background: #1f6feb; border-radius: 3px; }
.pc-item .var-pct { width: 42px; text-align: right; color: #8b949e; font-size: 10px; flex-shrink: 0; }

.strip-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }
.strip-header h2 { font-size: 18px; }
.strip-header .var { font-size: 13px; color: #8b949e; }

.strip-container { margin-bottom: 24px; }
.strip-row { display: flex; gap: 2px; margin-bottom: 2px; }
.strip-axis { display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;
              padding: 4px 2px 0; }

.cell-img { width: 96px; height: 96px; image-rendering: pixelated; border-radius: 2px;
            cursor: pointer; border: 2px solid transparent; transition: border-color 0.15s; }
.cell-img:hover { border-color: #58a6ff; }
.cell-img.selected { border-color: #f0883e; }
.cell-placeholder { width: 96px; height: 96px; background: #21262d; border-radius: 2px; }

.cell-detail { position: fixed; bottom: 0; right: 0; width: calc(100% - 280px); background: #161b22;
               border-top: 1px solid #30363d; padding: 12px 20px; display: none;
               font-size: 13px; z-index: 10; }
.cell-detail.open { display: flex; gap: 20px; align-items: center; }
.cell-detail img { width: 128px; height: 128px; image-rendering: pixelated; border-radius: 4px; }
.cell-detail .meta { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; }
.cell-detail .meta dt { color: #8b949e; }
.cell-detail .meta dd { font-family: monospace; }
.cell-detail .close-btn { position: absolute; top: 8px; right: 12px; background: none; border: none;
                          color: #8b949e; cursor: pointer; font-size: 16px; }

.gene-section { margin-bottom: 16px; padding: 0 8px; }
.gene-section h3 { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;
                   margin-bottom: 6px; }
.gene-chips { display: flex; flex-wrap: wrap; gap: 3px; }
.gene-chip { padding: 1px 6px; border-radius: 10px; font-size: 10px; font-family: monospace;
             cursor: pointer; border: 1px solid #30363d; white-space: nowrap; }
.gene-chip:hover { border-color: #58a6ff; }
.gene-chip.high { color: #3fb950; border-color: #23462633; }
.gene-chip.low { color: #f85149; border-color: #46232333; }

.pc-nav { display: flex; gap: 8px; margin-left: auto; }
.pc-nav button { padding: 4px 12px; border-radius: 4px; border: 1px solid #30363d;
                 background: #21262d; color: #e6edf3; cursor: pointer; font-size: 12px; }
.pc-nav button:hover { border-color: #58a6ff; }
.pc-nav button:disabled { opacity: 0.3; cursor: default; }
</style>
</head><body>
<div class="app">
<header>
  <h1>PC Strip Explorer</h1>
  <div class="stats" id="stats"></div>
  <div class="search-box"><input id="search" placeholder="Search gene or PC..." autofocus></div>
  <div class="pc-nav">
    <button id="prevPC" onclick="navPC(-1)">&#9664; Prev</button>
    <button id="nextPC" onclick="navPC(1)">Next &#9654;</button>
  </div>
</header>
<div class="sidebar">
  <ul class="pc-list" id="pcList"></ul>
</div>
<div class="main" id="mainPanel"></div>
</div>

<div class="cell-detail" id="cellDetail">
  <button class="close-btn" onclick="closeDetail()">&times;</button>
  <img id="detailImg" src="">
  <dl class="meta" id="detailMeta"></dl>
</div>

<script>
const overview = """ + json.dumps(overview) + r""";
const pcData = """ + json.dumps(pc_data) + r""";
const geneData = """ + json.dumps(gene_data) + r""";
const geneNames = """ + json.dumps(gene_names) + r""";
const cropB64 = """ + json.dumps(crop_b64) + r""";

let currentPC = null;

function init() {
  document.getElementById('stats').textContent =
    `${overview.n_genes} genes · ${overview.n_pcs} PCs · ${overview.total_variance}% variance · ${overview.n_bins} bins × ${overview.n_rows} rows`;

  buildPCList();
  showPC(1);

  const searchEl = document.getElementById('search');
  searchEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const q = e.target.value.trim();
    const pcMatch = q.match(/^pc\s*(\d+)$/i);
    if (pcMatch) { showPC(parseInt(pcMatch[1])); return; }
    const qUp = q.toUpperCase();
    const exact = geneNames.find(g => g === qUp);
    if (exact) { showGeneOverlay(exact); return; }
    const match = geneNames.find(g => g.includes(qUp));
    if (match) showGeneOverlay(match);
  });
}

function buildPCList() {
  const ul = document.getElementById('pcList');
  const maxVar = Math.max(...overview.explained_variance);
  for (let i = 0; i < overview.n_pcs; i++) {
    const li = document.createElement('li');
    li.className = 'pc-item';
    li.dataset.pc = i + 1;
    const pct = overview.explained_variance[i];
    li.innerHTML = `
      <div class="pc-row">
        <span class="pc-label">PC${i+1}</span>
        <div class="var-bar-wrap"><div class="var-bar" style="width:${pct/maxVar*100}%"></div></div>
        <span class="var-pct">${pct.toFixed(1)}%</span>
      </div>`;
    li.onclick = () => showPC(i + 1);
    ul.appendChild(li);
  }
}

function showPC(num) {
  if (num < 1 || num > overview.n_pcs) return;
  currentPC = num;

  document.querySelectorAll('.pc-item').forEach(li => {
    li.classList.toggle('active', parseInt(li.dataset.pc) === num);
  });
  const activeLi = document.querySelector(`.pc-item[data-pc="${num}"]`);
  if (activeLi) activeLi.scrollIntoView({ block: 'nearest' });

  document.getElementById('prevPC').disabled = num <= 1;
  document.getElementById('nextPC').disabled = num >= overview.n_pcs;

  renderPC(pcData[num]);
}

function navPC(delta) {
  if (currentPC) showPC(currentPC + delta);
}

function imgSrc(filename) {
  const b64 = cropB64[filename];
  return b64 ? `data:image/png;base64,${b64}` : '';
}

function renderPC(data) {
  const panel = document.getElementById('mainPanel');
  let html = `
    <div class="strip-header">
      <h2>PC${data.pc}</h2>
      <span class="var">${data.explained_variance}% explained variance</span>
    </div>`;

  html += '<div class="strip-container">';
  const nRows = overview.n_rows;
  for (let row = 0; row < nRows; row++) {
    html += '<div class="strip-row">';
    for (const bin of data.strip) {
      const cell = bin.cells[row];
      if (cell && cell.has_crop) {
        const src = imgSrc(cell.img);
        html += `<img class="cell-img" src="${src}"
                  data-gene="${cell.gene}" data-score="${cell.score}"
                  data-exp="${cell.experiment}" data-well="${cell.well}"
                  data-x="${cell.x}" data-y="${cell.y}"
                  data-img="${cell.img}"
                  onclick="selectCell(this)"
                  title="${cell.gene} score=${cell.score}">`;
      } else {
        html += '<div class="cell-placeholder"></div>';
      }
    }
    html += '</div>';
  }
  html += `<div class="strip-axis"><span>Low</span><span>High</span></div>`;
  html += '</div>';

  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px">';

  html += '<div class="gene-section"><h3>High-loading genes (positive)</h3><div class="gene-chips">';
  for (const g of data.high_genes) {
    html += `<span class="gene-chip high" onclick="showGeneOverlay('${g.gene}')"
              title="${g.score}">${g.gene} <small>+${g.score.toFixed(1)}</small></span>`;
  }
  html += '</div></div>';

  html += '<div class="gene-section"><h3>Low-loading genes (negative)</h3><div class="gene-chips">';
  for (const g of data.low_genes) {
    html += `<span class="gene-chip low" onclick="showGeneOverlay('${g.gene}')"
              title="${g.score}">${g.gene} <small>${g.score.toFixed(1)}</small></span>`;
  }
  html += '</div></div></div>';

  const geneCounts = {};
  for (const bin of data.strip) {
    for (const cell of bin.cells) {
      geneCounts[cell.gene] = (geneCounts[cell.gene] || 0) + 1;
    }
  }
  const sortedGenes = Object.entries(geneCounts).sort((a,b) => b[1] - a[1]).slice(0, 30);
  html += '<div class="gene-section" style="margin-top:16px"><h3>Most represented genes in this strip</h3>';
  html += '<div class="gene-chips">';
  for (const [gene, count] of sortedGenes) {
    html += `<span class="gene-chip" onclick="showGeneOverlay('${gene}')">${gene} <small>&times;${count}</small></span>`;
  }
  html += '</div></div>';

  panel.innerHTML = html;
  closeDetail();
}

function selectCell(el) {
  document.querySelectorAll('.cell-img').forEach(img => img.classList.remove('selected'));
  el.classList.add('selected');
  const detail = document.getElementById('cellDetail');
  document.getElementById('detailImg').src = imgSrc(el.dataset.img);
  document.getElementById('detailMeta').innerHTML = `
    <dt>Gene</dt><dd><a href="#" onclick="showGeneOverlay('${el.dataset.gene}');return false"
                       style="color:#58a6ff">${el.dataset.gene}</a></dd>
    <dt>Score</dt><dd>${parseFloat(el.dataset.score).toFixed(3)}</dd>
    <dt>Experiment</dt><dd>${el.dataset.exp}</dd>
    <dt>Well</dt><dd>${el.dataset.well}</dd>
    <dt>Position</dt><dd>(${el.dataset.x}, ${el.dataset.y})</dd>`;
  detail.classList.add('open');
}

function closeDetail() {
  document.getElementById('cellDetail').classList.remove('open');
  document.querySelectorAll('.cell-img').forEach(img => img.classList.remove('selected'));
}

function showGeneOverlay(name) {
  const data = geneData[name];
  if (!data) return;

  const maxAbs = Math.max(...data.top_pcs.map(p => Math.abs(p.score)));
  let html = `<div style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);
               z-index:100;display:flex;align-items:center;justify-content:center"
               onclick="if(event.target===this)this.remove()">
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;
                max-width:700px;width:90%;max-height:80vh;overflow-y:auto" onclick="event.stopPropagation()">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <h2 style="font-size:16px">${data.gene}</h2>
        <button onclick="this.closest('[style*=fixed]').remove()" style="background:none;border:none;color:#8b949e;font-size:18px;cursor:pointer">&times;</button>
      </div>
      <div style="font-size:11px;color:#8b949e;margin-bottom:8px">Top PCs by |score| — click to navigate</div>`;

  for (const pc of data.top_pcs) {
    const pct = Math.abs(pc.score) / maxAbs * 45;
    const color = pc.score >= 0 ? '#3fb950' : '#f85149';
    const barStyle = pc.score >= 0
      ? `right:50%;width:${pct}%;background:${color}`
      : `left:50%;width:${pct}%;background:${color}`;
    html += `<div style="display:flex;align-items:center;gap:8px;padding:3px 4px;cursor:pointer;border-radius:4px;font-size:12px"
                  onmouseover="this.style.background='#1f2937'" onmouseout="this.style.background=''"
                  onclick="this.closest('[style*=fixed]').remove();showPC(${pc.pc})">
      <span style="width:42px;text-align:right;font-family:monospace;color:#8b949e">PC${pc.pc}</span>
      <div style="flex:1;height:14px;background:#21262d;border-radius:3px;position:relative">
        <div style="height:100%;border-radius:3px;position:absolute;${barStyle}"></div>
      </div>
      <span style="width:50px;font-family:monospace;font-size:11px;color:${color}">${pc.score > 0 ? '+' : ''}${pc.score.toFixed(2)}</span>
    </div>`;
  }
  html += '</div></div>';

  const overlay = document.createElement('div');
  overlay.innerHTML = html;
  document.body.appendChild(overlay.firstElementChild);
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); navPC(-1); }
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); navPC(1); }
  if (e.key === 'Escape') closeDetail();
});

init();
</script>
</body></html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_artifacts = Path(__file__).resolve().parent.parent / "artifacts"
    parser.add_argument("--artifacts-dir", default=str(default_artifacts),
                        help=f"Directory with input data (default: {default_artifacts})")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: <artifacts-dir>/pc_explorer_static.html)")
    args = parser.parse_args()

    output = args.output or str(Path(args.artifacts_dir) / "pc_explorer_static.html")
    build(args.artifacts_dir, output)
