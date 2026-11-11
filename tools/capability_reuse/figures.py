#!/usr/bin/env python3
"""Figures for the capability-reuse audit (repo-coloured network, orphans red)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/capability_reuse"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]


def short(name: str, n: int = 26) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def load():
    stats = json.loads((ART / "graph_stats.json").read_text())
    edges = json.loads((ART / "edges.json").read_text())["edges"]
    return stats, edges


def fig_network(stats, edges):
    cap_repo = stats["cap_repo"]
    cap_class = stats["cap_class"]
    indeg = stats["indegree"]
    headline = set(stats["headline_edge_kinds"])
    # display graph: headline + artefact tier, so cross-repository coupling is visible
    usable = [e for e in edges if e["kind"] in headline | {"artifact_ref", "ref_symbol"}]
    G = nx.DiGraph()
    for e in usable:
        G.add_edge(e["src"], e["dst"])
    # keep every component with >= 4 nodes (the graph is genuinely fragmented:
    # 2727 components on 1238 determined capabilities -> the figure must show it)
    comps = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
    keep = set()
    for comp in comps:
        if len(comp) < 4:
            break
        keep |= comp
    G = G.subgraph(keep).copy()
    repos = sorted({cap_repo.get(n, "") for n in G.nodes})
    colour = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(repos)}
    pos = {}
    # lay each component out separately, then pack them on a grid
    import math
    comps = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
    cols = max(1, int(math.ceil(math.sqrt(len(comps)))))
    for idx, comp in enumerate(comps):
        sub = G.subgraph(comp)
        k = 0.6 if len(comp) < 20 else 0.35
        p = nx.spring_layout(sub, k=k, iterations=120, seed=11 + idx)
        ox, oy = (idx % cols) * 3.0, -(idx // cols) * 3.0
        for n, xy in p.items():
            pos[n] = (xy[0] + ox, xy[1] + oy)
    plt.figure(figsize=(17, 12))
    widths = [0.4 + 0.35 * G.in_degree(n) for n in G.nodes]
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True, arrowsize=8,
                           edge_color="#555555", width=0.7, node_size=260)
    nx.draw_networkx_nodes(G, pos, node_size=[90 + 34 * G.degree(n) for n in G.nodes],
                           node_color=[("#d62728" if indeg.get(n, 0) == 0 else colour[cap_repo.get(n, "")])
                                       for n in G.nodes],
                           linewidths=0.4, edgecolors="white")
    labels = {n: short(cap_class.get(n, {}).get("name", n), 22) for n in G.nodes if G.degree(n) >= 4}
    nx.draw_networkx_labels(G, pos, labels, font_size=5.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour[r],
                          label=f"{r or '<none>'}") for r in repos]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color="#d62728",
                              label="orphan (in-degree 0 in headline)"))
    plt.legend(handles=handles, fontsize=8, loc="lower left", frameon=True, ncol=2)
    plt.title("Capability reuse network (all components with >=4 capabilities; code+artefact+symbol evidence)\n"
              f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}  "
              f"colour=repository, red=orphan", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    out = FIG / "fig1_reuse_network.png"
    plt.savefig(out, dpi=190)
    plt.close()
    return out, G.number_of_nodes(), G.number_of_edges()


def fig_indegree(stats):
    dist = stats["indegree_distribution"]
    ks = [int(k) for k in dist if k != ">max"]
    ys = [dist[str(k)] for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].bar(ks, ys, color="#1f77b4")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("in-degree (number of distinct reusing capabilities)")
    ax[0].set_ylabel("capabilities (log scale)")
    ax[0].set_title(f"In-degree distribution, determined capabilities n={sum(ys)}")
    for x, y in zip(ks, ys):
        if y:
            ax[0].annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 3), fontsize=7, ha="center")
    tiers = stats["tiers"]
    names = ["strong\n(code)", "strong+medium\n(+artefact)", "strong+usage\n(+symbol)",
             "strong+medium+usage"]
    keys = ["headline_all", "headline_plus_medium", "headline_plus_usage", "headline_plus_all"]
    vals = [100 * tiers[k]["reuse_rate"] for k in keys]
    edges_n = [tiers[k]["edges"] for k in keys]
    bars = ax[1].bar(names, vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    for b, v, e in zip(bars, vals, edges_n):
        ax[1].annotate(f"{v:.1f}%\n{e} edges", (b.get_x() + b.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax[1].set_ylabel("share of determined capabilities with >=1 in-edge [%]")
    ax[1].set_title("Reuse rate by evidence tier (tiers never merged)")
    ax[1].set_ylim(0, max(vals) * 1.35)
    plt.tight_layout()
    out = FIG / "fig2_indegree_and_tiers.png"
    plt.savefig(out, dpi=190)
    plt.close()
    return out


def fig_repo(stats):
    per = stats["per_repo"]
    rows = [(r, v) for r, v in per.items() if v["determined"]]
    rows.sort(key=lambda kv: -kv[1]["determined"])
    names = [r or "<none>" for r, _ in rows]
    det = [v["determined"] for _, v in rows]
    reused = [v["reused"] for _, v in rows]
    rate = [100 * v["reused"] / v["determined"] for _, v in rows]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = range(len(names))
    ax.bar(x, det, color="#c8d8e8", label="determined capabilities")
    ax.bar(x, reused, color="#1f77b4", label="reused (>=1 in-edge)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("capabilities")
    ax2 = ax.twinx()
    ax2.plot(list(x), rate, "o-", color="#d62728", label="reuse rate [%]")
    for xi, r in zip(x, rate):
        ax2.annotate(f"{r:.0f}%", (xi, r), textcoords="offset points", xytext=(0, 5),
                     fontsize=7, ha="center", color="#d62728")
    ax2.set_ylabel("reuse rate [%]", color="#d62728")
    ax.set_title("Per-repository reuse density (headline: code calibres)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    plt.tight_layout()
    out = FIG / "fig3_per_repo.png"
    plt.savefig(out, dpi=190)
    plt.close()
    return out


def fig_hubs(stats):
    indeg = stats["indegree"]
    cap = stats["cap_class"]
    rows = sorted(indeg.items(), key=lambda kv: (-kv[1], kv[0]))
    rows = [(c, n) for c, n in rows if n > 0][:15]
    labels = [short(cap.get(c, {}).get("name", c), 40) for c, _ in rows][::-1]
    vals = [n for _, n in rows][::-1]
    repos = [stats["cap_repo"].get(c, "") for c, _ in rows][::-1]
    order = sorted({r for r in repos})
    colmap = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(order)}
    plt.figure(figsize=(11, 6.4))
    plt.barh(range(len(vals)), vals, color=[colmap[r] for r in repos])
    plt.yticks(range(len(vals)), labels, fontsize=7)
    plt.xlabel("in-degree (reusing capabilities)")
    plt.title("Hub capabilities (single-point-of-failure candidates)")
    plt.tight_layout()
    out = FIG / "fig4_hubs.png"
    plt.savefig(out, dpi=190)
    plt.close()
    return out


def main() -> int:
    stats, edges = load()
    outs = []
    o, n, e = fig_network(stats, edges)
    print("network nodes/edges drawn:", n, e)
    outs.append(o)
    outs.append(fig_indegree(stats))
    outs.append(fig_repo(stats))
    outs.append(fig_hubs(stats))
    print(json.dumps([str(p) for p in outs], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())