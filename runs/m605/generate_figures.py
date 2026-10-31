#!/usr/bin/env python3
"""Generate paper-quality figures for m605 FEM/DEM capability audit report.
All labels in English to avoid CJK font issues.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

BASE = '/root/tp-agent/work/m605'
OUT_DIR = os.path.join(BASE, 'runs', 'm605')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Figure 1: Capability Radar Chart
# ============================================================
def plot_radar():
    categories = [
        'Element\nLibrary', 'Material\nNonlinearity', 'Contact\nAlgorithms',
        'Adaptive\nMesh', 'Parallel\nScaling', 'Fracture/\nBreakage',
        'FSI\nCoupling', 'Time\nIntegration', 'Validation\nAnchors',
        'Engineering\nScale'
    ]
    N = len(categories)
    tensorfem = [4, 3, 3, 1, 2, 6, 3, 7, 5, 2]
    tensordem = [2, 2, 4, 1, 3, 5, 2, 5, 4, 2]
    dealii_ccx = [9, 8, 8, 8, 8, 5, 4, 8, 9, 8]
    liggghts   = [6, 5, 8, 4, 9, 7, 3, 7, 8, 9]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    def add_curve(values, label, color, alpha=0.15):
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.2, label=label, color=color)
        ax.fill(angles, vals, alpha=alpha, color=color)

    add_curve(tensorfem, 'TensorFEM (current)', '#2196F3')
    add_curve(tensordem, 'TensorDEM (current)', '#FF9800')
    add_curve(dealii_ccx, 'deal.II/CalculiX (ref)', '#4CAF50', alpha=0.05)
    add_curve(liggghts, 'LIGGGHTS/ESyS (ref)', '#9C27B0', alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
    ax.set_title('FEM/DEM Capability Radar: TensorPhysics vs Reference Codes',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_capability_radar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path

# ============================================================
# Figure 2: Dependency DAG
# ============================================================
def plot_dag():
    with open(os.path.join(BASE, 'artifacts', 'fem_dem_task_tree.json')) as f:
        tree = json.load(f)

    nodes = {n['id']: n for n in tree['nodes']}
    edges = tree['dag_edges']

    layer_order = ['L1_foundation', 'L2_validation', 'L3_integration']
    layer_y = {'L1_foundation': 3, 'L2_validation': 2, 'L3_integration': 1}
    layer_colors = {'L1_foundation': '#E3F2FD', 'L2_validation': '#FFF3E0', 'L3_integration': '#E8F5E9'}
    layer_borders = {'L1_foundation': '#1565C0', 'L2_validation': '#E65100', 'L3_integration': '#2E7D32'}
    layer_labels = {'L1_foundation': 'L1: Foundation', 'L2_validation': 'L2: Validation Anchors', 'L3_integration': 'L3: Application Integration'}

    layer_nodes = defaultdict(list)
    for n in tree['nodes']:
        layer_nodes[n['layer']].append(n['id'])

    pos = {}
    for layer in layer_order:
        ids = layer_nodes[layer]
        n_ids = len(ids)
        for i, nid in enumerate(ids):
            x = (i + 1) / (n_ids + 1) * 10
            pos[nid] = (x, layer_y[layer])

    fig, ax = plt.subplots(figsize=(16, 10))

    for layer in layer_order:
        y = layer_y[layer]
        rect = mpatches.FancyBboxPatch((0.1, y - 0.45), 9.8, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor=layer_colors[layer],
                                        edgecolor=layer_borders[layer],
                                        linewidth=1.5, alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.3, y + 0.35, layer_labels[layer],
                fontsize=9, color=layer_borders[layer], fontweight='bold')

    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        arrow = FancyArrowPatch((x1, y1 - 0.15), (x2, y2 + 0.15),
                                arrowstyle='->', mutation_scale=15,
                                color='#666', linewidth=1.2, alpha=0.6)
        ax.add_patch(arrow)

    node_colors = {'L1_foundation': '#42A5F5', 'L2_validation': '#FFA726', 'L3_integration': '#66BB6A'}
    m579_block = set(tree.get('m579_blocking_nodes', []))

    for nid, (x, y) in pos.items():
        color = node_colors[nodes[nid]['layer']]
        border = '#D32F2F' if nid in m579_block else '#333'
        lw = 2.5 if nid in m579_block else 1.5
        circle = plt.Circle((x, y), 0.28, facecolor=color, edgecolor=border, linewidth=lw, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, nid, ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

    legend_elements = [
        mpatches.Patch(facecolor='#42A5F5', label='L1: Foundation'),
        mpatches.Patch(facecolor='#FFA726', label='L2: Validation'),
        mpatches.Patch(facecolor='#66BB6A', label='L3: Integration'),
        mpatches.Patch(facecolor='white', edgecolor='#D32F2F', linewidth=2, label='Blocks m579'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('FEM/DEM Task Tree Dependency DAG (16 nodes, 20 edges)\nRed border = blocks m579 G4 critical path',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_dependency_dag.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path

# ============================================================
# Figure 3: Gap Matrix Heatmap
# ============================================================
def plot_heatmap():
    rows = [
        'Element/Particle Shape', 'Material Nonlinearity', 'Contact Algorithms',
        'Fracture/Breakage', 'FSI Coupling', 'Time Integration',
        'Parallel/Scaling', 'Validation Anchors', 'FEM->DEM Handoff', 'Engineering Scale'
    ]
    cols = ['FEM Status\n(vs deal.II/CalculiX)', 'DEM Status\n(vs LIGGGHTS/ESyS)', 'Ship-Ice Needs\n(SIR/G6)']

    # 0=covered, 1=partial, 2=gap, 3=blocking
    data = np.array([
        [1, 2, 2], [2, 2, 1], [1, 1, 3], [1, 1, 2], [2, 3, 3],
        [0, 0, 1], [2, 2, 3], [1, 1, 2], [3, 3, 3], [2, 2, 3],
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)

    label_map = {0: 'Covered', 1: 'Partial', 2: 'Gap', 3: 'BLOCKING'}
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            text = label_map[val]
            color = 'white' if val >= 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.set_ticklabels(['Covered', 'Partial', 'Gap', 'BLOCKING'])
    cbar.set_label('Gap Severity', fontsize=11)
    ax.set_title('FEM/DEM Three-Layer Cross-Reference Gap Matrix\n(rows=capability, cols=assessment perspective)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_gap_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path

if __name__ == '__main__':
    p1 = plot_radar()
    p2 = plot_dag()
    p3 = plot_heatmap()
    print(f'\nAll figures saved to {OUT_DIR}')
