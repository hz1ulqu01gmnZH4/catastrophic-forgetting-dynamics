#!/usr/bin/env python3
"""
Generate visualizations for the catastrophic forgetting experiment.

Creates:
1. Static summary plots
2. Trajectory animation comparing high vs low forgetting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import torch

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['font.family'] = 'monospace'


def load_all_data():
    """Load data from all phases."""
    base_path = Path(__file__).parent.parent / "results"

    data = {}

    # Phase 1
    p1_path = base_path / "phase1" / "forgetting_data.csv"
    if p1_path.exists():
        data['phase1'] = pd.read_csv(p1_path)

    # Phase 2
    p2_path = base_path / "phase2" / "phase2_data.csv"
    if p2_path.exists():
        data['phase2'] = pd.read_csv(p2_path)

    # Phase 3
    p3_path = base_path / "phase3" / "phase3_data.csv"
    if p3_path.exists():
        data['phase3'] = pd.read_csv(p3_path)

    # Phase 4
    p4_path = base_path / "phase4" / "phase4_data.csv"
    if p4_path.exists():
        data['phase4'] = pd.read_csv(p4_path)

    return data


def plot_phase_comparison(data, output_path):
    """Create summary comparison across phases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Catastrophic Forgetting: Four Phases of Discovery',
                 fontsize=16, fontweight='bold', color='#58a6ff')

    colors = ['#7ee787', '#58a6ff', '#f778ba', '#ffa657']

    # Phase 1: LR vs Forgetting
    ax = axes[0, 0]
    if 'phase1' in data:
        df = data['phase1']
        for i, lr in enumerate(sorted(df['learning_rate'].unique())):
            subset = df[df['learning_rate'] == lr]
            ax.scatter(subset['similarity'], subset['forgetting'],
                      alpha=0.5, s=20, label=f'LR={lr}')
    ax.set_xlabel('Task Similarity')
    ax.set_ylabel('Forgetting')
    ax.set_title('Phase 1: Linear Networks', color=colors[0])
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='#8b949e', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Phase 2: Lazy vs Rich
    ax = axes[0, 1]
    if 'phase2' in data:
        df = data['phase2']
        lazy = df[df['regime_after_t2'] == 'lazy']['forgetting']
        rich = df[df['regime_after_t2'] == 'rich']['forgetting']

        positions = [1, 2]
        bp = ax.boxplot([lazy.dropna(), rich.dropna()], positions=positions,
                       patch_artist=True, widths=0.6)

        bp['boxes'][0].set_facecolor('#7ee787')
        bp['boxes'][1].set_facecolor('#f85149')
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='#c9d1d9')

        ax.set_xticks(positions)
        ax.set_xticklabels(['Lazy\n(90.4%)', 'Rich\n(9.6%)'])
        ax.set_ylabel('Forgetting')
        ax.set_title('Phase 2: Lazy-Rich Transition', color=colors[1])
        ax.axhline(y=0, color='#8b949e', linestyle='--', alpha=0.5)

        # Add means
        ax.scatter([1], [lazy.mean()], color='white', s=100, zorder=5, marker='D')
        ax.scatter([2], [rich.mean()], color='white', s=100, zorder=5, marker='D')
        ax.annotate(f'{lazy.mean():.3f}', (1.15, lazy.mean()), color='#7ee787', fontsize=10)
        ax.annotate(f'{rich.mean():.3f}', (2.15, rich.mean()), color='#f85149', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Phase 3: Deviation correlations
    ax = axes[1, 0]
    if 'phase3' in data:
        df = data['phase3']

        metrics = ['max_deviation_t2', 'deviation_after_t2', 'flr_after_t2']
        labels = ['Max Dev\n(during)', 'Final Dev\n(endpoint)', 'FLR']
        correlations = []

        for m in metrics:
            if m in df.columns:
                clean = df[[m, 'forgetting']].dropna()
                if len(clean) > 10:
                    corr = clean['forgetting'].corr(clean[m])
                    correlations.append(corr)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)

        bars = ax.bar(range(len(labels)), correlations, color=[colors[2], '#8b949e', '#8b949e'])
        bars[0].set_color('#7ee787')  # Max deviation is good
        bars[1].set_color('#f85149')  # Final deviation is bad
        bars[2].set_color('#ffa657')  # FLR

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Correlation with Forgetting')
        ax.set_title('Phase 3: What Predicts Forgetting?', color=colors[2])
        ax.axhline(y=0, color='#8b949e', linestyle='--', alpha=0.5)

        for i, v in enumerate(correlations):
            ax.annotate(f'{v:.2f}', (i, v + 0.02 if v >= 0 else v - 0.08),
                       ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Phase 4: Trajectory vs FLR
    ax = axes[1, 1]
    if 'phase4' in data:
        df = data['phase4']

        # Get correlations for top trajectory metrics vs FLR
        traj_metrics = ['t2_max_deviation', 't2_excursion_intensity',
                       't2_path_integral_deviation', 'flr_after_t2']
        labels = ['Max Dev', 'Excursion', 'Path Int', 'FLR']
        correlations = []

        for m in traj_metrics:
            if m in df.columns:
                clean = df[[m, 'forgetting']].dropna()
                if len(clean) > 10:
                    corr = clean['forgetting'].corr(clean[m])
                    correlations.append(corr)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)

        bar_colors = ['#7ee787', '#7ee787', '#7ee787', '#f85149']
        bars = ax.bar(range(len(labels)), correlations, color=bar_colors)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Correlation with Forgetting')
        ax.set_title('Phase 4: Trajectory Beats FLR', color=colors[3])
        ax.axhline(y=0, color='#8b949e', linestyle='--', alpha=0.5)

        for i, v in enumerate(correlations):
            ax.annotate(f'{v:.2f}', (i, v + 0.02 if v >= 0 else v - 0.08),
                       ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_comparison(data, output_path):
    """Plot trajectory comparison: high vs low forgetting."""
    if 'phase4' not in data:
        print("Phase 4 data not available for trajectory plot")
        return

    df = data['phase4']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Trajectory Shape Determines Forgetting',
                 fontsize=14, fontweight='bold', color='#58a6ff')

    # Separate high and low forgetting
    median_forg = df['forgetting'].median()
    low_forg = df[df['forgetting'] < median_forg]
    high_forg = df[df['forgetting'] >= median_forg]

    # Plot 1: Max deviation distribution
    ax = axes[0]
    if 't2_max_deviation' in df.columns:
        ax.hist(low_forg['t2_max_deviation'].dropna(), bins=20, alpha=0.7,
               color='#7ee787', label=f'Low Forgetting (n={len(low_forg)})')
        ax.hist(high_forg['t2_max_deviation'].dropna(), bins=20, alpha=0.7,
               color='#f85149', label=f'High Forgetting (n={len(high_forg)})')
        ax.set_xlabel('Max Deviation (T2)')
        ax.set_ylabel('Count')
        ax.set_title('Max Deviation Distribution')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Path integral distribution
    ax = axes[1]
    if 't2_path_integral_deviation' in df.columns:
        ax.hist(low_forg['t2_path_integral_deviation'].dropna(), bins=20, alpha=0.7,
               color='#7ee787', label='Low Forgetting')
        ax.hist(high_forg['t2_path_integral_deviation'].dropna(), bins=20, alpha=0.7,
               color='#f85149', label='High Forgetting')
        ax.set_xlabel('Path Integral (∫deviation dt)')
        ax.set_ylabel('Count')
        ax.set_title('Cumulative Deviation Distribution')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Scatter of max deviation vs forgetting
    ax = axes[2]
    if 't2_max_deviation' in df.columns:
        scatter = ax.scatter(df['t2_max_deviation'], df['forgetting'],
                            c=df['similarity'], cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Max Deviation (T2)')
        ax.set_ylabel('Forgetting')
        ax.set_title('Forgetting vs Max Deviation')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Task Similarity')

        # Add trend line
        clean = df[['t2_max_deviation', 'forgetting']].dropna()
        if len(clean) > 10:
            z = np.polyfit(clean['t2_max_deviation'], clean['forgetting'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(clean['t2_max_deviation'].min(),
                                clean['t2_max_deviation'].max(), 100)
            ax.plot(x_line, p(x_line), '--', color='#f0f6fc', linewidth=2,
                   label=f'Trend (r={clean["forgetting"].corr(clean["t2_max_deviation"]):.2f})')
            ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_trajectory_animation(output_path):
    """
    Generate animation showing trajectory evolution for different configurations.

    Simulates deviation trajectories for:
    - Low forgetting case (low LR, high similarity)
    - High forgetting case (high LR, low similarity)
    """
    from src.nonlinear_models import (
        NonlinearTeacher, NonlinearStudent,
        create_nonlinear_task_pair, classify_regime
    )
    from src.universal_subspace import UniversalSubspace

    print("Generating trajectory animation...")

    # Run two representative cases
    torch.manual_seed(42)
    np.random.seed(42)

    d_in, d_hidden, d_out = 50, 64, 5
    n_steps = 300
    track_every = 5

    cases = [
        {'lr': 0.02, 'similarity': 0.8, 'label': 'Low Forgetting\n(LR=0.02, s=0.8)', 'color': '#7ee787'},
        {'lr': 0.15, 'similarity': 0.2, 'label': 'High Forgetting\n(LR=0.15, s=0.2)', 'color': '#f85149'},
    ]

    trajectories = []
    forgetting_values = []

    for case in cases:
        # Create task pair
        task_pair = create_nonlinear_task_pair(
            d_in=d_in, d_hidden=d_hidden, d_out=d_out,
            similarity=case['similarity'], activation='gelu'
        )

        teacher1 = NonlinearTeacher(task_pair.teacher1_W1, task_pair.teacher1_W2, 'gelu')
        teacher2 = NonlinearTeacher(task_pair.teacher2_W1, task_pair.teacher2_W2, 'gelu')

        # Create student
        student = NonlinearStudent(d_in, d_hidden, d_out, 'gelu', init_scale=1.0)

        # Collect weights during training
        weights_t1 = []
        weights_t2 = []

        optimizer = torch.optim.SGD(student.parameters(), lr=case['lr'])

        # Task 1
        W_init = torch.cat([student.W1.flatten(), student.W2.flatten()]).detach().clone()
        weights_t1.append(W_init)

        for step in range(n_steps):
            X, y = teacher1.generate_data(64)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(student(X), y)
            loss.backward()
            optimizer.step()

            if step % track_every == 0:
                W = torch.cat([student.W1.flatten(), student.W2.flatten()]).detach().clone()
                weights_t1.append(W)

        loss_t1_after_t1 = torch.nn.functional.mse_loss(
            student(teacher1.generate_data(500)[0]),
            teacher1.generate_data(500)[1]
        ).item()

        # Task 2
        weights_t2.append(weights_t1[-1])

        for step in range(n_steps):
            X, y = teacher2.generate_data(64)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(student(X), y)
            loss.backward()
            optimizer.step()

            if step % track_every == 0:
                W = torch.cat([student.W1.flatten(), student.W2.flatten()]).detach().clone()
                weights_t2.append(W)

        loss_t1_after_t2 = torch.nn.functional.mse_loss(
            student(teacher1.generate_data(500)[0]),
            teacher1.generate_data(500)[1]
        ).item()

        forgetting = loss_t1_after_t2 - loss_t1_after_t1
        forgetting_values.append(forgetting)

        # Fit subspace and compute deviations
        subspace = UniversalSubspace(target_variance=0.9)
        subspace.fit(weights_t1)

        deviations = []
        for W in weights_t1 + weights_t2:
            try:
                analysis = subspace.analyze(W)
                deviations.append(analysis.deviation_ratio)
            except:
                deviations.append(deviations[-1] if deviations else 0)

        trajectories.append(deviations)

    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Trajectory Through Weight Space: The Journey Matters',
                 fontsize=14, fontweight='bold', color='#58a6ff')

    # Left plot: Deviation over time
    ax1 = axes[0]
    ax1.set_xlim(0, len(trajectories[0]))
    max_dev = max(max(t) for t in trajectories) * 1.1
    ax1.set_ylim(0, max_dev)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Subspace Deviation ||θ⊥|| / ||θ∥||')
    ax1.set_title('Deviation Trajectory')
    ax1.axvline(x=len(weights_t1), color='#8b949e', linestyle='--', alpha=0.5, label='Task Switch')
    ax1.grid(True, alpha=0.3)

    lines = []
    points = []
    for i, case in enumerate(cases):
        line, = ax1.plot([], [], color=case['color'], linewidth=2, label=case['label'])
        point, = ax1.plot([], [], 'o', color=case['color'], markersize=10)
        lines.append(line)
        points.append(point)
    ax1.legend(loc='upper left', fontsize=9)

    # Right plot: Forgetting bar chart (will be revealed at end)
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(min(0, min(forgetting_values) - 0.1), max(forgetting_values) + 0.1)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([c['label'] for c in cases])
    ax2.set_ylabel('Forgetting')
    ax2.set_title('Resulting Forgetting')
    ax2.axhline(y=0, color='#8b949e', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    bars = ax2.bar([0, 1], [0, 0], color=[c['color'] for c in cases], alpha=0.8)

    # Text annotations
    text_annotations = []
    for i in range(2):
        txt = ax2.text(i, 0, '', ha='center', va='bottom', fontsize=12, fontweight='bold')
        text_annotations.append(txt)

    # Animation parameters
    n_frames = len(trajectories[0])

    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])
        for bar in bars:
            bar.set_height(0)
        for txt in text_annotations:
            txt.set_text('')
        return lines + points + list(bars) + text_annotations

    def animate(frame):
        for i, (line, point, traj) in enumerate(zip(lines, points, trajectories)):
            x = list(range(frame + 1))
            y = traj[:frame + 1]
            line.set_data(x, y)
            if frame < len(traj):
                point.set_data([frame], [traj[frame]])

        # Reveal bars progressively at the end
        if frame >= n_frames - 1:
            progress = 1.0
        elif frame >= n_frames * 0.9:
            progress = (frame - n_frames * 0.9) / (n_frames * 0.1)
        else:
            progress = 0

        for i, (bar, forg) in enumerate(zip(bars, forgetting_values)):
            bar.set_height(forg * progress)
            if progress > 0.5:
                text_annotations[i].set_text(f'{forg:.3f}')
                text_annotations[i].set_y(forg + 0.02 if forg >= 0 else forg - 0.05)

        return lines + points + list(bars) + text_annotations

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=30, blit=True)

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=30,
              savefig_kwargs={'facecolor': '#0d1117'})
    plt.close()
    print(f"Saved animation: {output_path}")


def plot_key_findings_infographic(output_path):
    """Create an infographic summarizing key findings."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Catastrophic Forgetting: Key Discoveries',
                 fontsize=18, fontweight='bold', color='#58a6ff', y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 1. Main finding box
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.axis('off')
    ax_main.text(0.5, 0.7, 'The Journey Matters More Than The Destination',
                fontsize=16, ha='center', va='center', fontweight='bold', color='#7ee787')
    ax_main.text(0.5, 0.3,
                'Forgetting ∝ max_t ||θ⊥(t)|| / ||θ∥(t)||',
                fontsize=14, ha='center', va='center',
                fontfamily='monospace', color='#f0f6fc',
                bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#58a6ff'))

    # 2. Phase progression
    ax_phases = fig.add_subplot(gs[1, 0])
    ax_phases.axis('off')
    ax_phases.set_title('Phase Progression', fontsize=12, color='#58a6ff')

    phases = [
        ('Phase 1', 'Linear\nBaseline', '0.63'),
        ('Phase 2', 'Lazy-Rich\nTransition', '0.52'),
        ('Phase 3', 'Subspace\nAnalysis', '0.86'),
        ('Phase 4', 'Trajectory\nHypothesis', '0.84'),
    ]

    for i, (name, desc, r2) in enumerate(phases):
        y = 0.85 - i * 0.22
        ax_phases.add_patch(FancyBboxPatch((0.1, y-0.08), 0.8, 0.16,
                                          boxstyle="round,pad=0.02",
                                          facecolor='#21262d', edgecolor='#30363d'))
        ax_phases.text(0.15, y, name, fontsize=10, fontweight='bold', color='#58a6ff')
        ax_phases.text(0.4, y, desc, fontsize=8, color='#c9d1d9')
        ax_phases.text(0.85, y, f'R²={r2}', fontsize=9, color='#7ee787', ha='right')

    # 3. Key metrics comparison
    ax_metrics = fig.add_subplot(gs[1, 1])
    metrics = ['Trajectory\n(max_dev)', 'Similarity', 'Learning\nRate', 'FLR']
    correlations = [0.68, -0.92, 0.16, 0.03]
    colors = ['#7ee787', '#58a6ff', '#ffa657', '#f85149']

    bars = ax_metrics.barh(range(len(metrics)), [abs(c) for c in correlations], color=colors)
    ax_metrics.set_yticks(range(len(metrics)))
    ax_metrics.set_yticklabels(metrics)
    ax_metrics.set_xlabel('|Correlation with Forgetting|')
    ax_metrics.set_title('Predictor Comparison', fontsize=12, color='#58a6ff')
    ax_metrics.set_xlim(0, 1)

    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax_metrics.text(abs(corr) + 0.02, i, f'{corr:+.2f}', va='center', fontsize=9)
    ax_metrics.grid(True, alpha=0.3, axis='x')

    # 4. Lazy vs Rich
    ax_regime = fig.add_subplot(gs[1, 2])
    regimes = ['Lazy\n(90%)', 'Rich\n(10%)']
    forgetting = [0.060, 0.392]
    colors = ['#7ee787', '#f85149']

    bars = ax_regime.bar(range(len(regimes)), forgetting, color=colors)
    ax_regime.set_xticks(range(len(regimes)))
    ax_regime.set_xticklabels(regimes)
    ax_regime.set_ylabel('Mean Forgetting')
    ax_regime.set_title('Regime Effect (6.6×)', fontsize=12, color='#58a6ff')

    for i, (bar, f) in enumerate(zip(bars, forgetting)):
        ax_regime.text(i, f + 0.02, f'{f:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax_regime.grid(True, alpha=0.3, axis='y')

    # 5. Key insights
    ax_insights = fig.add_subplot(gs[2, :])
    ax_insights.axis('off')
    ax_insights.set_title('Key Insights', fontsize=12, color='#58a6ff')

    insights = [
        ('✓', 'Trajectory > FLR', 'Path metrics beat FLR by 20×', '#7ee787'),
        ('✓', 'Max > Final', 'Peak deviation beats endpoint', '#7ee787'),
        ('✓', 'LR Threshold', 'η ≥ 0.1 triggers rich regime', '#ffa657'),
        ('✓', 'Path Integral', 'Cumulative exposure matters (r=0.60)', '#58a6ff'),
    ]

    for i, (check, title, desc, color) in enumerate(insights):
        x = 0.05 + (i % 2) * 0.5
        y = 0.6 - (i // 2) * 0.4
        ax_insights.text(x, y, check, fontsize=14, color=color, fontweight='bold')
        ax_insights.text(x + 0.03, y, title, fontsize=11, color='#f0f6fc', fontweight='bold')
        ax_insights.text(x + 0.03, y - 0.15, desc, fontsize=9, color='#8b949e')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent.parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_all_data()
    print(f"Loaded phases: {list(data.keys())}")

    print("\n1. Generating phase comparison plot...")
    plot_phase_comparison(data, output_dir / "phase_comparison.png")

    print("\n2. Generating trajectory comparison plot...")
    plot_trajectory_comparison(data, output_dir / "trajectory_comparison.png")

    print("\n3. Generating key findings infographic...")
    plot_key_findings_infographic(output_dir / "key_findings.png")

    print("\n4. Generating trajectory animation...")
    generate_trajectory_animation(output_dir / "trajectory_animation.gif")

    print("\n" + "="*50)
    print("All visualizations saved to:")
    print(f"  {output_dir}/")
    print("="*50)


if __name__ == "__main__":
    main()
