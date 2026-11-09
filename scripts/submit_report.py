"""Submit the report with embedded figures."""
import json
import base64
import os
import sys

# Load the report template
with open('/tmp/report.json', 'r') as f:
    report = json.load(f)

# Add figures
figures_dir = '/root/tp-agent/work/m7382/artifacts'
figure_defs = [
    ('polarization_curve', 'PEMFC polarization curve showing voltage and power vs. current density'),
    ('loss_breakdown', 'Breakdown of polarization losses: activation, ohmic, and concentration'),
    ('energy_conservation', 'Energy evolution in multi-field coupling and conservation residuals'),
    ('fuel_comparison', 'Comparison of hydrogen, methanol, and ammonia fuel pathways'),
]

report['figures'] = []
for name, caption in figure_defs:
    path = os.path.join(figures_dir, f'{name}.png')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        report['figures'].append({
            'name': name,
            'caption': caption,
            'png_base64': b64
        })

# Save final report
with open('/tmp/report_final.json', 'w') as f:
    json.dump(report, f)

print(f"Report prepared with {len(report['figures'])} figures")
print(f"Total size: {os.path.getsize('/tmp/report_final.json') / 1024:.1f} KB")
