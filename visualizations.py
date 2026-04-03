"""
Quantum SQA Visualization Module
=================================
HTML visualization generators for Quantum SQA solver results.
Generates interactive charts in the 'additional_output' directory.
"""

from __future__ import annotations

import numpy as np
import os
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from qcentroid import SQAConfig, QuantumAnnealingSchedule

def generate_quantum_visualizations(
    history: List[Tuple[int, float]],
    best_solution: np.ndarray,
    best_config: Dict,
    best_energy: float,
    best_f1: float,
    top_configs: List[Dict],
    sqa_config: SQAConfig,
    energy_landscape: Dict,
    solver: 'QuantumSQASolver',
    elapsed_time: float,
) -> None:
    """
    Generate HTML visualization files for quantum SQA results.

    Creates visualizations in the 'additional_output' directory:
    - energy_convergence.html: Energy vs sweep with schedules
    - solution_heatmap.html: 12x16 grid of 192-bit solution
    - annealing_schedule.html: Temperature and transverse field schedules
    - top_configurations.html: Table of top 10 configs
    - quantum_dashboard.html: Summary dashboard

    Args:
        history: List of (sweep_number, energy) tuples
        best_solution: 192-bit binary solution vector
        best_config: Decoded hyperparameter configuration dict
        best_energy: Best QUBO energy found
        best_f1: Estimated F1 score
        top_configs: List of top 10 configuration dicts
        sqa_config: SQAConfig dataclass
        energy_landscape: Dict with min/max/mean/std energy stats
        solver: QuantumSQASolver instance
        elapsed_time: Total execution time in seconds
    """
    try:
        os.makedirs('additional_output', exist_ok=True)

        # Color scheme
        colors = {
            'primary': '#7c3aed',      # purple
            'secondary': '#2563eb',    # blue
            'success': '#10b981',      # green
            'warning': '#f59e0b',      # amber
            'danger': '#ef4444'        # red
        }

        # Downsample history if needed
        if len(history) > 200:
            indices = np.linspace(0, len(history) - 1, 200, dtype=int)
            history_downsampled = [history[i] for i in indices]
        else:
            history_downsampled = history

        # 1. Energy Convergence Chart
        _generate_energy_convergence_html(history_downsampled, sqa_config, colors)

        # 2. Solution Heatmap (12x16 grid)
        _generate_solution_heatmap_html(best_solution, colors)

        # 3. Annealing Schedule
        _generate_annealing_schedule_html(sqa_config, colors)

        # 4. Top Configurations Table
        _generate_top_configurations_html(top_configs, colors)

        # 5. Quantum Dashboard
        _generate_quantum_dashboard_html(
            best_config, best_energy, best_f1, sqa_config,
            energy_landscape, solver, elapsed_time, colors
        )

        logger.info("Quantum visualizations generated successfully in 'additional_output' directory")

    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")


def _generate_energy_convergence_html(
    history: List[Tuple[int, float]],
    sqa_config: SQAConfig,
    colors: Dict[str, str]
) -> None:
    """Generate energy convergence chart with temperature and transverse field overlays."""
    sweeps = [h[0] for h in history]
    energies = [h[1] for h in history]

    # Normalize energy for visualization
    if len(energies) > 0:
        min_e = min(energies)
        max_e = max(energies)
        energy_range = max_e - min_e if max_e != min_e else 1.0
        energies_norm = [(e - min_e) / energy_range * 100 for e in energies]
    else:
        energies_norm = []

    # Generate temperature schedule
    temps = []
    for s in sweeps:
        progress = s / sqa_config.num_sweeps
        temp = sqa_config.initial_temperature + progress * (
            sqa_config.final_temperature - sqa_config.initial_temperature
        )
        temps.append(temp)

    # Normalize temperature for visualization
    max_temp = max(temps) if temps else 1.0
    temps_norm = [t / max_temp * 100 for t in temps]

    # Generate transverse field schedule
    from qcentroid import QuantumAnnealingSchedule as _QAS
    schedule = _QAS(sqa_config)
    fields = [schedule.get_transverse_field(int(s)) for s in sweeps]
    max_field = max(fields) if fields else 1.0
    fields_norm = [f / max_field * 100 for f in fields]

    # Build SVG path for energy
    svg_width, svg_height = 800, 400
    padding = 60
    plot_width = svg_width - 2 * padding
    plot_height = svg_height - 2 * padding

    if len(sweeps) > 1:
        x_scale = plot_width / (sweeps[-1] - sweeps[0]) if sweeps[-1] != sweeps[0] else 1.0
        energy_path = "M " + " L ".join(
            f"{padding + (s - sweeps[0]) * x_scale},{svg_height - padding - e}"
            for s, e in zip(sweeps, energies_norm)
        )
        temp_path = "M " + " L ".join(
            f"{padding + (s - sweeps[0]) * x_scale},{svg_height - padding - t}"
            for s, t in zip(sweeps, temps_norm)
        )
        field_path = "M " + " L ".join(
            f"{padding + (s - sweeps[0]) * x_scale},{svg_height - padding - f}"
            for s, f in zip(sweeps, fields_norm)
        )
    else:
        energy_path = temp_path = field_path = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Convergence</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: {colors['primary']};
            margin-top: 0;
        }}
        svg {{
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            font-size: 14px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>QUBO Energy Convergence</h1>
        <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
            <!-- Grid -->
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e5e7eb" stroke-width="0.5"/>
                </pattern>
            </defs>
            <rect width="{svg_width}" height="{svg_height}" fill="url(#grid)"/>

            <!-- Axes -->
            <line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}"
                  stroke="#000" stroke-width="2"/>
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}"
                  stroke="#000" stroke-width="2"/>

            <!-- Axis labels -->
            <text x="{svg_width - padding - 20}" y="{svg_height - padding + 25}" font-size="12" fill="#666">
                Sweep Number
            </text>
            <text x="15" y="{padding - 10}" font-size="12" fill="#666" text-anchor="middle">
                Normalized Energy
            </text>

            <!-- Data lines -->
            <path d="{energy_path}" stroke="{colors['primary']}" stroke-width="2" fill="none" opacity="0.8"/>
            <path d="{temp_path}" stroke="{colors['warning']}" stroke-width="2" fill="none" opacity="0.6" stroke-dasharray="5,5"/>
            <path d="{field_path}" stroke="{colors['secondary']}" stroke-width="2" fill="none" opacity="0.6" stroke-dasharray="5,5"/>

            <!-- Tick marks -->
            <g font-size="11" fill="#666">
                <text x="{padding}" y="{svg_height - padding + 20}" text-anchor="middle">0</text>
                <text x="{svg_width - padding}" y="{svg_height - padding + 20}" text-anchor="middle">
                    {int(sweeps[-1] if sweeps else 0)}
                </text>
                <text x="{padding - 30}" y="{svg_height - padding + 5}" text-anchor="end">0</text>
                <text x="{padding - 30}" y="{padding + 5}" text-anchor="end">100</text>
            </g>
        </svg>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: {colors['primary']};"></div>
                <span>QUBO Energy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {colors['warning']};"></div>
                <span>Temperature Schedule (dashed)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {colors['secondary']};"></div>
                <span>Transverse Field (dashed)</span>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open('additional_output/energy_convergence.html', 'w') as f:
        f.write(html)


def _generate_solution_heatmap_html(solution: np.ndarray, colors: Dict[str, str]) -> None:
    """Generate 12x16 heatmap of the 192-bit solution with hyperparameter legend."""
    # Hyperparameter bit ranges (example mapping)
    hyperparams = [
        ("learning_rate", 0, 3),
        ("warmup_steps", 3, 6),
        ("weight_decay", 6, 9),
        ("dropout_rate", 9, 12),
        ("attention_heads", 12, 15),
        ("hidden_dim", 15, 18),
        ("num_layers", 18, 21),
        ("batch_size", 21, 24),
        ("optimizer", 24, 27),
        ("scheduler", 27, 30),
        ("gradient_clipping", 30, 33),
        ("label_smoothing", 33, 36),
        ("mixed_precision", 36, 39),
        ("activation_fn", 39, 42),
        ("pos_encoding", 42, 45),
        ("layer_norm", 45, 48),
    ]

    # Create color mapping for each bit
    param_colors = {}
    param_hex_colors = {}
    color_palette = [
        '#7c3aed', '#2563eb', '#0891b2', '#059669', '#d97706',
        '#dc2626', '#7c3aed', '#2563eb', '#0891b2', '#059669',
        '#d97706', '#dc2626', '#7c3aed', '#2563eb', '#0891b2', '#059669'
    ]

    for i, (name, start, end) in enumerate(hyperparams):
        hex_color = color_palette[i % len(color_palette)]
        param_hex_colors[name] = hex_color
        for bit_idx in range(start, min(end, 192)):
            param_colors[bit_idx] = name

    # Create 12x16 grid
    rows, cols = 12, 16
    cell_size = 30
    svg_width = cols * cell_size + 100
    svg_height = rows * cell_size + 100

    svg = f"""<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">"""

    for bit_idx in range(192):
        row = bit_idx // cols
        col = bit_idx % cols
        x = 50 + col * cell_size
        y = 50 + row * cell_size
        color = '#10b981' if solution[bit_idx] == 1 else '#e5e7eb'
        param_name = param_colors.get(bit_idx, "unknown")
        param_color = param_hex_colors.get(param_name, '#999')

        svg += f"""
        <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}"
              fill="{color}" stroke="{param_color}" stroke-width="2" opacity="0.8">
            <title>Bit {bit_idx}: {solution[bit_idx]:.0f} ({param_name})</title>
        </rect>
        <text x="{x + cell_size/2}" y="{y + cell_size/2 + 4}" text-anchor="middle"
              font-size="11" font-weight="bold" fill="#000" pointer-events="none">
            {int(solution[bit_idx])}
        </text>"""

    # Add legend
    legend_y = rows * cell_size + 70
    legend_x = 50
    svg += f"""
        <text x="50" y="{legend_y}" font-size="14" font-weight="bold" fill="#000">Hyperparameter Groups:</text>"""

    for i, (name, _, _) in enumerate(hyperparams):
        col_idx = i % 4
        row_idx = i // 4
        lx = legend_x + col_idx * 220
        ly = legend_y + 25 + row_idx * 20
        hex_color = param_hex_colors[name]
        svg += f"""
        <rect x="{lx}" y="{ly - 10}" width="15" height="15" fill="{hex_color}" stroke="#999" stroke-width="1"/>
        <text x="{lx + 20}" y="{ly + 2}" font-size="12" fill="#333">{name}</text>"""

    svg += """
    </svg>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solution Heatmap</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: {colors['primary']};
            margin-top: 0;
        }}
        svg {{
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            display: block;
            margin: 20px auto;
        }}
        .info {{
            background-color: #f3f4f6;
            padding: 12px;
            border-radius: 4px;
            margin-top: 20px;
            font-size: 13px;
            color: #374151;
            border-left: 4px solid {colors['secondary']};
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Solution Heatmap (12x16 Grid)</h1>
        <p>Visualization of the 192-bit solution vector. Green cells indicate bits set to 1, gray cells indicate 0.</p>
        {svg}
        <div class="info">
            <strong>Color Coding:</strong> Each hyperparameter group is outlined with a distinct color.
            Green fill = bit value 1, Gray fill = bit value 0.
        </div>
    </div>
</body>
</html>"""

    with open('additional_output/solution_heatmap.html', 'w') as f:
        f.write(html)


def _generate_annealing_schedule_html(sqa_config: SQAConfig, colors: Dict[str, str]) -> None:
    """Generate dual-axis chart of temperature and transverse field schedules."""
    from qcentroid import QuantumAnnealingSchedule as _QAS
    schedule = _QAS(sqa_config)

    sweeps = np.linspace(0, sqa_config.num_sweeps - 1, 100, dtype=int)
    temps = []
    fields = []

    for s in sweeps:
        progress = s / sqa_config.num_sweeps
        temp = sqa_config.initial_temperature + progress * (
            sqa_config.final_temperature - sqa_config.initial_temperature
        )
        temps.append(temp)
        fields.append(schedule.get_transverse_field(int(s)))

    # Normalize for visualization
    max_temp = max(temps) if temps else 1.0
    max_field = max(fields) if fields else 1.0
    temps_norm = [t / max_temp * 100 for t in temps]
    fields_norm = [f / max_field * 100 for f in fields]

    svg_width, svg_height = 800, 400
    padding = 60
    plot_width = svg_width - 2 * padding
    plot_height = svg_height - 2 * padding

    if len(sweeps) > 1:
        x_scale = plot_width / (sweeps[-1] - sweeps[0]) if sweeps[-1] != sweeps[0] else 1.0
        temp_path = "M " + " L ".join(
            f"{padding + (s - sweeps[0]) * x_scale},{svg_height - padding - t}"
            for s, t in zip(sweeps, temps_norm)
        )
        field_path = "M " + " L ".join(
            f"{padding + (s - sweeps[0]) * x_scale},{svg_height - padding - f}"
            for s, f in zip(sweeps, fields_norm)
        )
    else:
        temp_path = field_path = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annealing Schedule</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: {colors['primary']};
            margin-top: 0;
        }}
        svg {{
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }}
        .schedule-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .info-box {{
            padding: 15px;
            border-radius: 4px;
            background-color: #f3f4f6;
            border-left: 4px solid {colors['secondary']};
        }}
        .info-box h3 {{
            margin: 0 0 8px 0;
            color: {colors['primary']};
            font-size: 14px;
        }}
        .info-box p {{
            margin: 4px 0;
            font-size: 13px;
            color: #374151;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Annealing Schedule</h1>
        <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
            <!-- Grid -->
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e5e7eb" stroke-width="0.5"/>
                </pattern>
            </defs>
            <rect width="{svg_width}" height="{svg_height}" fill="url(#grid)"/>

            <!-- Axes -->
            <line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}"
                  stroke="#000" stroke-width="2"/>
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}"
                  stroke="#000" stroke-width="2"/>

            <!-- Data lines -->
            <path d="{temp_path}" stroke="{colors['warning']}" stroke-width="3" fill="none" opacity="0.8"/>
            <path d="{field_path}" stroke="{colors['secondary']}" stroke-width="3" fill="none" opacity="0.8"/>

            <!-- Axis labels -->
            <text x="{svg_width - padding - 20}" y="{svg_height - padding + 25}" font-size="12" fill="#666">
                Sweep
            </text>
            <text x="15" y="{padding - 10}" font-size="12" fill="#666" text-anchor="middle">
                Normalized Value
            </text>

            <!-- Legend in top-right -->
            <rect x="{svg_width - 250}" y="{padding}" width="220" height="70" fill="white" stroke="#ddd" stroke-width="1"/>
            <line x1="{svg_width - 230}" y1="{padding + 15}" x2="{svg_width - 180}" y2="{padding + 15}"
                  stroke="{colors['warning']}" stroke-width="3"/>
            <text x="{svg_width - 170}" y="{padding + 20}" font-size="12" fill="#000">Temperature</text>
            <line x1="{svg_width - 230}" y1="{padding + 40}" x2="{svg_width - 180}" y2="{padding + 40}"
                  stroke="{colors['secondary']}" stroke-width="3"/>
            <text x="{svg_width - 170}" y="{padding + 45}" font-size="12" fill="#000">Transverse Field</text>
        </svg>

        <div class="schedule-info">
            <div class="info-box">
                <h3>Temperature Schedule</h3>
                <p><strong>Initial:</strong> {sqa_config.initial_temperature:.2f}</p>
                <p><strong>Final:</strong> {sqa_config.final_temperature:.6f}</p>
                <p><strong>Type:</strong> {sqa_config.beta_schedule}</p>
            </div>
            <div class="info-box">
                <h3>Transverse Field Schedule</h3>
                <p><strong>Initial:</strong> {sqa_config.initial_transverse_field:.2f}</p>
                <p><strong>Final:</strong> {sqa_config.final_transverse_field:.6f}</p>
                <p><strong>Type:</strong> {sqa_config.transverse_field_schedule}</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open('additional_output/annealing_schedule.html', 'w') as f:
        f.write(html)


def _generate_top_configurations_html(top_configs: List[Dict], colors: Dict[str, str]) -> None:
    """Generate HTML table of top 10 configurations."""
    rows_html = ""

    for cfg in top_configs:
        rank = cfg.get('rank', 0)
        f1 = cfg.get('estimated_f1_macro', 0.0)
        energy = cfg.get('qubo_energy', 0.0)
        config = cfg.get('config', {})

        # Format config details
        config_str = ", ".join(f"{k}: {v}" for k, v in list(config.items())[:5])
        if len(config) > 5:
            config_str += f", ... (+{len(config)-5} more)"

        # Color code by rank
        if rank == 1:
            rank_color = colors['success']
        elif rank <= 3:
            rank_color = colors['warning']
        else:
            rank_color = colors['secondary']

        rows_html += f"""
        <tr>
            <td style="background-color: {rank_color}; color: white; font-weight: bold; text-align: center;">
                {rank}
            </td>
            <td style="font-weight: mono;">{f1:.6f}</td>
            <td style="font-weight: mono;">{energy:.6f}</td>
            <td style="font-size: 12px; color: #555;">{config_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top Configurations</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: {colors['primary']};
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: {colors['primary']};
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid {colors['primary']};
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
        .numeric {{
            font-family: 'Courier New', monospace;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Top 10 Configurations</h1>
        <p>Ranked by estimated F1-macro score on the validation set.</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 60px;">Rank</th>
                    <th style="width: 120px;">F1-Macro Score</th>
                    <th style="width: 150px;">QUBO Energy</th>
                    <th>Hyperparameter Configuration</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""

    with open('additional_output/top_configurations.html', 'w') as f:
        f.write(html)


def _generate_quantum_dashboard_html(
    best_config: Dict,
    best_energy: float,
    best_f1: float,
    sqa_config: SQAConfig,
    energy_landscape: Dict,
    solver: 'QuantumSQASolver',
    elapsed_time: float,
    colors: Dict[str, str]
) -> None:
    """Generate summary dashboard with key metrics and configuration."""

    # Format config details
    config_items = ""
    for key, value in best_config.items():
        config_items += f"<tr><td>{key}</td><td><strong>{value}</strong></td></tr>"

    # Calculate exchange rate
    exchange_rate = solver.replica_manager.get_exchange_rate() * 100

    # Energy landscape
    min_energy = energy_landscape.get('min_energy', 0.0)
    max_energy = energy_landscape.get('max_energy', 0.0)
    mean_energy = energy_landscape.get('mean_energy', 0.0)
    std_energy = energy_landscape.get('std_energy', 0.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: white;
            text-align: center;
            margin: 0 0 30px 0;
            font-size: 32px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin: 0 0 15px 0;
            font-size: 14px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid {colors['primary']};
            padding-bottom: 10px;
        }}
        .metric {{
            margin: 12px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .metric-label {{
            font-size: 13px;
            color: #666;
            font-weight: 500;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
            color: {colors['primary']};
            font-family: 'Courier New', monospace;
        }}
        .metric-small {{
            font-size: 12px;
            color: #999;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 13px;
        }}
        th {{
            background-color: #f3f4f6;
            padding: 8px;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 1px solid #e5e7eb;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .accent {{
            color: {colors['warning']};
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .progress-bar {{
            height: 8px;
            background-color: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            background-color: {colors['success']};
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantum SQA Dashboard</h1>

        <div class="grid">
            <!-- Best Results -->
            <div class="card">
                <h2>Best Solution</h2>
                <div class="metric">
                    <span class="metric-label">F1-Macro Score</span>
                    <span class="metric-value">{best_f1:.6f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">QUBO Energy</span>
                    <span class="metric-value">{best_energy:.6f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Rank</span>
                    <span class="metric-value">1</span>
                </div>
            </div>

            <!-- SQA Parameters -->
            <div class="card">
                <h2>SQA Configuration</h2>
                <div class="metric">
                    <span class="metric-label">Sweeps</span>
                    <span class="metric-value">{sqa_config.num_sweeps}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Replicas</span>
                    <span class="metric-value">{sqa_config.num_replicas}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trotter Slices</span>
                    <span class="metric-value">{sqa_config.trotter_slices}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Exchange Interval</span>
                    <span class="metric-value">{sqa_config.replica_exchange_interval}</span>
                </div>
            </div>

            <!-- Energy Landscape -->
            <div class="card">
                <h2>Energy Landscape</h2>
                <div class="metric">
                    <span class="metric-label">Min Energy</span>
                    <span class="metric-value">{min_energy:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Energy</span>
                    <span class="metric-value">{max_energy:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mean Energy</span>
                    <span class="metric-value">{mean_energy:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Std Dev</span>
                    <span class="metric-value">{std_energy:.4f}</span>
                </div>
            </div>

            <!-- Convergence Metrics -->
            <div class="card">
                <h2>Convergence</h2>
                <div class="metric">
                    <span class="metric-label">Exchange Rate</span>
                    <span class="metric-value">{exchange_rate:.1f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {exchange_rate}%"></div>
                </div>
                <div class="metric" style="margin-top: 15px;">
                    <span class="metric-label">Initial Temp</span>
                    <span class="metric-value">{sqa_config.initial_temperature:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Final Temp</span>
                    <span class="metric-value">{sqa_config.final_temperature:.6f}</span>
                </div>
            </div>

            <!-- Timing -->
            <div class="card">
                <h2>Execution Profile</h2>
                <div class="metric">
                    <span class="metric-label">Total Time</span>
                    <span class="metric-value">{elapsed_time:.2f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Time/Sweep</span>
                    <span class="metric-value">{elapsed_time / max(sqa_config.num_sweeps, 1):.4f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Time/Replica</span>
                    <span class="metric-value">{elapsed_time / max(sqa_config.num_replicas, 1):.4f}s</span>
                </div>
            </div>

            <!-- Best Configuration Table -->
            <div class="card full-width">
                <h2>Best Hyperparameter Configuration</h2>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40%;">Parameter</th>
                            <th style="width: 60%;">Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {config_items}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open('additional_output/quantum_dashboard.html', 'w') as f:
        f.write(html)
