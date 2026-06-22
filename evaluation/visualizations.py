"""
System Visualization Module - Publication-Quality Charts for Multi-Agent Results

OBJECTIVE:
This module provides professional visualization capabilities for multi-agent system
evaluation results. It generates publication-quality charts (300 DPI) suitable for
academic papers, presentations, and technical reports, with clear labeling, professional
styling, and accessible color palettes.

WHY THIS EXISTS:
Evaluation metrics alone aren't sufficient for communication:
- **Stakeholder Communication**: Non-technical audiences need visual representation
- **Pattern Recognition**: Humans detect patterns better in charts than tables
- **Publication Requirements**: Academic papers require professional figures
- **Comparative Analysis**: Visual comparison reveals insights text cannot
- **Presentation Materials**: Talks and slides need high-quality graphics

This module ensures results are:
- Visually appealing and professional
- Print-ready (300 DPI, proper formatting)
- Accessible (colorblind-friendly palettes)
- Clearly labeled (titles, axes, legends)
- Consistently styled (seaborn themes)

FIVE CORE VISUALIZATIONS:

1. **Belief Distribution Plot** (Stacked Bar Chart)
   - Shows how each agent distributes belief across alternatives
   - Stacked bars: 100% height = full belief distribution
   - Color-coded alternatives
   - Value labels on bars
   - Use case: Understanding agent preferences

2. **Consensus Evolution** (Line Chart)
   - Tracks consensus level across iterations/time
   - Shows convergence to threshold (default 0.75)
   - Highlights consensus region (above threshold)
   - Annotates first consensus achievement
   - Use case: Tracking deliberation progress

3. **Criteria Importance** (Radar Chart)
   - Visualizes relative importance of decision criteria
   - Polygon shape shows weight distribution
   - Easy to see dominant criteria
   - Labels show exact weights
   - Use case: Explaining MCDA weighting

4. **Decision Comparison** (Grouped Bar Chart)
   - Compares multi-agent vs. single-agent performance
   - Side-by-side bars for each metric
   - Improvement annotations (green arrows)
   - Value labels on bars
   - Use case: Demonstrating multi-agent value

5. **Agent Network** (Network Graph)
   - Visualizes multi-agent system as a graph
   - Nodes: Agents (colored by type)
   - Edges: Trust/interaction weights (thickness)
   - Labels: Agent names and expertise
   - Use case: System architecture visualization

PUBLICATION STANDARDS:
All plots meet academic publication requirements:
- **Resolution**: 300 DPI (print quality)
- **Format**: PNG with transparency support
- **Font Size**: 11-14pt (readable in print)
- **Line Width**: 1.5-2.5pt (visible but not thick)
- **Colors**: Seaborn "Set2" palette (colorblind-friendly)
- **Margins**: Tight layout with proper spacing
- **Labels**: UTF-8 support for Greek letters and symbols

TYPICAL USAGE:
```python
from evaluation.visualizations import SystemVisualizer

# Initialize with output directory
viz = SystemVisualizer(
    output_dir="thesis_figures",
    style="whitegrid",
    dpi=300
)

# Generate individual plots
viz.plot_belief_distributions(
    agent_assessments,
    "beliefs.png",
    title="Agent Belief Distributions - Flood Scenario"
)

viz.plot_consensus_evolution(
    consensus_history=[0.45, 0.60, 0.72, 0.78, 0.82],
    save_path="consensus.png",
    threshold=0.75
)

viz.plot_criteria_importance(
    criteria_weights={'safety': 0.35, 'cost': 0.25, 'speed': 0.20, 'effectiveness': 0.20},
    save_path="criteria.png"
)

viz.plot_decision_comparison(
    metrics=all_metrics,
    save_path="comparison.png"
)

viz.plot_agent_network(
    agent_profiles=profiles,
    trust_matrix=trust,
    save_path="network.png"
)

# Or generate all plots at once
results = {
    'agent_assessments': {...},
    'consensus_history': [...],
    'criteria_weights': {...},
    'metrics': {...},
    'agent_profiles': {...}
}

paths = viz.generate_all_plots(results, output_subdir="scenario_1")
print(f"Plots saved: {paths}")
```

INPUTS (Typical):
- agent_assessments: Dict[agent_id, assessment] with belief_distribution
- consensus_history: List[float] of consensus levels over iterations
- criteria_weights: Dict[criterion_name, weight]
- metrics: Dict with 'baseline_comparison' data
- agent_profiles: Dict[agent_id, profile] with name, expertise
- trust_matrix: Optional Dict[agent_i, Dict[agent_j, trust_score]]

OUTPUTS (Typical):
- PNG files saved to output_dir
- 300 DPI resolution
- White background
- Tight bounding box (no wasted space)
- Returns full file path strings

COLOR PALETTES:
The visualizer uses professionally designed color palettes:
- **Main palette**: Seaborn "Set2" (8 colors, colorblind-safe)
- **Agent colors**: Seaborn "husl" (high saturation, distinct)
- **Comparison**: Green for positive, red for negative/threshold
- **Network**: Different color per agent type

Colorblind-friendly principles:
- Avoid red-green only distinctions
- Use color + pattern (e.g., solid + hashed)
- High contrast between adjacent colors
- Test with colorblind simulators

STYLING OPTIONS:
Seaborn provides multiple style presets:
- **whitegrid** (default): White background, gray grid lines - clean, professional
- **darkgrid**: Gray background, white grid - for dark presentations
- **white**: White background, no grid - minimal, clean
- **dark**: Gray background, no grid - presentation mode
- **ticks**: White with axis ticks only - publication minimal

Font configuration:
- **Family**: sans-serif (DejaVu Sans, Arial, Helvetica)
- **Scale**: "paper" context (optimized for publications)
- **Unicode**: Properly handles Greek letters, math symbols
- **Minus sign**: Fixed display (matplotlib quirk)

CHART SELECTION GUIDE:
Choose the right visualization for your data:

| Data Type | Best Chart | When to Use |
|-----------|-----------|-------------|
| Distributions | Stacked Bar | Compare belief allocation across agents |
| Time Series | Line Chart | Show convergence or evolution over time |
| Multivariate | Radar Chart | Display 5-8 dimensional data (e.g., criteria) |
| Comparisons | Grouped Bar | Compare 2-4 groups across 2-5 metrics |
| Relationships | Network | Show connections, trust, or structure |

CUSTOMIZATION:
All plot methods accept customization parameters:
- **save_path**: Filename (relative to output_dir)
- **title**: Chart title (supports markdown bold, italics)
- **threshold**: For consensus plots (default 0.75)
- **style**: Seaborn style preset
- **dpi**: Resolution (default 300)

Advanced customization requires matplotlib:
```python
import matplotlib.pyplot as plt

# Modify rcParams before creating visualizer
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

viz = SystemVisualizer()
```

ERROR HANDLING:
- Missing data → Logged warning, empty string returned
- Invalid data format → Logged error, graceful skip
- File I/O errors → Logged error with path details
- Matplotlib failures → Caught and logged, other plots continue

PERFORMANCE:
- Belief distributions: O(N×M) for N agents, M alternatives
- Consensus evolution: O(T) for T time points
- Criteria importance: O(C) for C criteria
- Decision comparison: O(M) for M metrics
- Agent network: O(N²) for N agents (spring layout)

Typical timing:
- Simple plot (5 agents, 5 alternatives): < 100ms
- Complex network (10 agents, full connectivity): < 500ms
- Batch generation (all 5 plots): < 1s

BEST PRACTICES:
1. **Consistent Resolution**: Use same DPI for all plots in a paper (300 for print, 150 for web)
2. **Descriptive Titles**: Include scenario name, metric, and context
3. **Directory Organization**: Use subdirectories for different scenarios/runs
4. **File Naming**: Use clear, consistent names (e.g., "scenario1_beliefs.png")
5. **Batch Generation**: Use generate_all_plots() for consistency

COMMON ISSUES:
1. **Font Warnings**: Install DejaVu Sans if matplotlib complains
2. **Greek Letters**: Use UTF-8 encoding in Python source files
3. **Figure Size**: Adjust figsize if labels are cut off
4. **Color Contrast**: Test with grayscale printing
5. **File Overwrite**: Files are overwritten without warning

INTEGRATION POINTS:
- Used by: main.py for experiment visualization
- Inputs from: MetricsEvaluator results
- Outputs to: figures/ directory (configurable)
- Related: metrics.py for data generation

MATPLOTLIB/SEABORN DEPENDENCIES:
Required libraries:
- matplotlib: Core plotting library
- seaborn: Statistical visualization styling
- numpy: Numerical operations for plots
- networkx: Network graph layouts (for agent_network plot)

Installation:
```bash
pip install matplotlib seaborn numpy networkx
```

ACCESSIBILITY CONSIDERATIONS:
1. **Color Blindness**: Use patterns + colors, test with simulators
2. **Screen Readers**: Include alt-text in papers/presentations
3. **Print Quality**: Ensure visible at 50% scale (common in papers)
4. **Contrast**: Minimum 4.5:1 ratio for text
5. **Font Size**: Minimum 10pt when printed at full size

LATEX INTEGRATION:
For use in LaTeX documents:
```latex
\\usepackage{graphicx}

\\begin{figure}[h]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{figures/belief_distributions.png}
    \\caption{Agent belief distributions for flood evacuation scenario.
             Colors indicate alternatives: green (immediate evacuation),
             blue (staged evacuation), orange (shelter in place).}
    \\label{fig:beliefs}
\\end{figure}
```

LIMITATIONS:
1. **Static Only**: No interactive plots (consider plotly for web)
2. **PNG Format**: No vector formats (SVG, EPS, PDF) currently
3. **Fixed Styling**: Limited runtime style customization
4. **English Only**: Labels assume English text
5. **2D Only**: No 3D visualizations

FUTURE ENHANCEMENTS:
- Interactive HTML plots using plotly
- Vector format exports (SVG, PDF)
- Animated GIFs for time-series
- Customizable color schemes per agent
- Automatic optimal layout selection

RELATED RESEARCH:
- Data visualization best practices (Tufte, 2001)
- Colorblind-safe palettes (Okabe & Ito, 2008)
- Chart selection guidelines (Few, 2012)
- Publication figure requirements (IEEE, ACM standards)

VERSION HISTORY:
- v1.0: Initial implementation (5 chart types)
- v1.1: Enhanced styling and labeling
- v1.2: Added network visualization
- v2.0: Publication-quality defaults (300 DPI)
- v2.1: Comprehensive documentation (Jan 2025)

SEE ALSO:
- metrics.py: Data generation for visualization
- EVALUATION_METHODOLOGY.md: Metric definitions
- matplotlib documentation: https://matplotlib.org/
- seaborn gallery: https://seaborn.pydata.org/examples/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import seaborn as sns
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from models.data_models import BeliefDistribution, AgentAssessment


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemVisualizer:
    """
    Professional visualization generator for multi-agent system results.

    Creates publication-quality charts with:
    - High resolution (300 DPI)
    - Professional styling
    - Clear legends and titles
    - UTF-8 support for Greek letters

    Example:
        >>> viz = SystemVisualizer(output_dir="thesis_figures")
        >>> viz.plot_belief_distributions(assessments, "beliefs.png")
        >>> viz.plot_consensus_evolution(history, "consensus.png")
        >>> viz.generate_all_plots(results, "output")
    """

    def __init__(
        self,
        output_dir: str = "visualizations",
        style: str = "whitegrid",
        dpi: int = 300
    ):
        """
        Initialize the system visualizer.

        Args:
            output_dir: Directory to save visualizations
            style: Seaborn style (whitegrid, darkgrid, white, dark, ticks)
            dpi: Resolution for saved images (default: 300 for print quality)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.dpi = dpi

        # Set professional styling for thesis
        sns.set_style(style)
        sns.set_context("paper", font_scale=1.2)

        # Use a professional color palette
        self.colors = sns.color_palette("Set2", 10)
        self.agent_colors = sns.color_palette("husl", 12)  # Extended to 12 for up to 12 agents

        # Configure matplotlib for better fonts
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

        logger.info(f"SystemVisualizer initialized (output: {self.output_dir}, dpi: {self.dpi})")

    # Canonical colours for the four comparison methods
    METHOD_COLORS = {
        'ER':          '#3498db',   # blue
        'GAT':         '#e74c3c',   # red
        'GAT_TRAINED': '#2ecc71',   # green
        'MCDA':        '#f39c12',   # orange
    }
    _METHOD_LABELS = {
        'ER':          'Evidential Reasoning (ER)',
        'GAT':         'Graph Attention Network (GAT)',
        'GAT_TRAINED': 'GAT Trained',
        'MCDA':        'MCDA (TOPSIS)',
    }

    def plot_belief_distributions(
        self,
        agent_assessments: Dict[str, Any],
        save_path: str,
        title: str = "Agent Belief Distributions"
    ) -> str:
        """
        Plot agent belief distributions as stacked bar chart.

        Shows how each agent distributes belief across alternatives.

        Args:
            agent_assessments: Dictionary of agent assessments with belief_distribution
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot

        Example:
            >>> assessments = {
            ...     'agent_1': {'belief_distribution': {'A1': 0.7, 'A2': 0.3}},
            ...     'agent_2': {'belief_distribution': {'A1': 0.5, 'A2': 0.5}}
            ... }
            >>> viz.plot_belief_distributions(assessments, "beliefs.png")
        """
        logger.info(f"Plotting belief distributions to {save_path}")

        # Extract belief distributions (handles Pydantic models and dicts)
        agents = []
        agent_beliefs = {}  # Store complete belief dists per agent

        for agent_id, assessment in agent_assessments.items():
            # Handle both AgentAssessment (Pydantic) and plain dict
            if isinstance(assessment, AgentAssessment):
                agent_name = assessment.agent_name
                belief_dist = assessment.belief_distribution
            elif isinstance(assessment, dict):
                if 'belief_distribution' not in assessment:
                    continue
                agent_name = assessment.get('agent_name', agent_id)
                belief_dist = assessment['belief_distribution']
            else:
                continue

            agents.append(agent_name)

            # Convert BeliefDistribution model to dict if needed
            if isinstance(belief_dist, BeliefDistribution):
                agent_beliefs[agent_name] = belief_dist.to_dict()
            elif isinstance(belief_dist, dict):
                agent_beliefs[agent_name] = belief_dist

        if not agents or not agent_beliefs:
            logger.warning("No belief distribution data to plot")
            return ""

        # Get all alternatives across all agents
        all_alternatives = set()
        for beliefs in agent_beliefs.values():
            all_alternatives.update(beliefs.keys())
        alternatives = sorted(all_alternatives)

        # Build belief_data with consistent dimensions (all agents × all alternatives)
        belief_data = {}
        for alt_id in alternatives:
            belief_data[alt_id] = []
            for agent_name in agents:
                # Get belief for this alternative, default to 0.0 if missing
                belief = agent_beliefs[agent_name].get(alt_id, 0.0)
                belief_data[alt_id].append(belief)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot stacked bars
        x_pos = np.arange(len(agents))
        bottom = np.zeros(len(agents))

        for i, alt_id in enumerate(alternatives):
            values = np.array(belief_data[alt_id])  # Ensure numpy array
            bars = ax.bar(
                x_pos,
                values,
                bottom=bottom,
                label=f"Alternative {alt_id}",
                color=self.colors[i % len(self.colors)],
                edgecolor='white',
                linewidth=0.5
            )
            bottom = bottom + values  # Element-wise addition

            # Add value labels on bars if space allows
            bar_color = self.colors[i % len(self.colors)]
            # Determine readable text color based on bar background luminance
            try:
                import matplotlib.colors as mcolors
                r, g, b, *_ = mcolors.to_rgba(bar_color)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.55 else 'black'
            except Exception:
                text_color = 'white'

            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.08:  # Only show label if bar is large enough
                    height = bar.get_height()
                    y_pos = bar.get_y() + height / 2
                    txt = ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f'{val:.2f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color=text_color,
                        weight='bold',
                    )
                    # Thin contrasting outline so text is legible on any background
                    txt.set_path_effects([
                        mpe.withStroke(linewidth=2,
                                       foreground='black' if text_color == 'white' else 'white')
                    ])

        # Customize plot
        ax.set_xlabel('Expert Agent', fontsize=12, weight='bold')
        ax.set_ylabel('Belief Distribution', fontsize=12, weight='bold')
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Belief distributions plot saved to {full_path}")
        return str(full_path)

    def plot_consensus_evolution(
        self,
        consensus_history: List[float],
        save_path: str,
        threshold: float = 0.75,
        title: str = "Consensus Evolution Over Iterations"
    ) -> str:
        """
        Plot consensus convergence as line chart over iterations.

        Shows how consensus level changes over time/iterations.

        Args:
            consensus_history: List of consensus levels (0-1) over iterations
            save_path: Filename to save the plot
            threshold: Consensus threshold line (default: 0.75)
            title: Plot title

        Returns:
            Full path to saved plot

        Example:
            >>> history = [0.45, 0.60, 0.72, 0.78, 0.82]
            >>> viz.plot_consensus_evolution(history, "consensus.png")
        """
        logger.info(f"Plotting consensus evolution to {save_path}")

        if not consensus_history:
            logger.warning("No consensus history to plot")
            return ""

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = list(range(1, len(consensus_history) + 1))

        # Plot consensus line
        ax.plot(
            iterations,
            consensus_history,
            marker='o',
            linewidth=2.5,
            markersize=8,
            color=self.colors[0],
            label='Consensus Level',
            markeredgecolor='white',
            markeredgewidth=1.5
        )

        # Add threshold line
        ax.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Consensus Threshold ({threshold:.0%})'
        )

        # Highlight region above threshold
        ax.fill_between(
            iterations,
            threshold,
            1.0,
            alpha=0.1,
            color='green',
            label='Consensus Region'
        )

        # Add annotations for key points
        # Mark when consensus is first reached
        for i, level in enumerate(consensus_history):
            if level >= threshold:
                ax.annotate(
                    f'Consensus\nReached',
                    xy=(iterations[i], level),
                    xytext=(iterations[i] + 0.5, level + 0.05),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10,
                    weight='bold',
                    color='green'
                )
                break

        # Customize plot
        ax.set_xlabel('Iteration', fontsize=12, weight='bold')
        ax.set_ylabel('Consensus Level', fontsize=12, weight='bold')
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.5, len(iterations) + 0.5)
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add percentage formatting to y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Consensus evolution plot saved to {full_path}")
        return str(full_path)

    def plot_criteria_importance(
        self,
        criteria_weights: Dict[str, float],
        save_path: str,
        title: str = "Decision Criteria Importance"
    ) -> str:
        """
        Plot criteria importance as radar chart.

        Visualizes relative importance of different decision criteria.

        Args:
            criteria_weights: Dictionary mapping criterion name to weight
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot

        Example:
            >>> weights = {
            ...     'Safety': 0.35,
            ...     'Cost': 0.25,
            ...     'Speed': 0.20,
            ...     'Effectiveness': 0.20
            ... }
            >>> viz.plot_criteria_importance(weights, "criteria.png")
        """
        logger.info(f"Plotting criteria importance to {save_path}")

        if not criteria_weights:
            logger.warning("No criteria weights to plot")
            return ""

        # Prepare data
        criteria = list(criteria_weights.keys())
        weights = list(criteria_weights.values())

        # Number of variables
        num_vars = len(criteria)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the plot
        weights += weights[:1]
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, weights, 'o-', linewidth=2, color=self.colors[2], label='Weights')
        ax.fill(angles, weights, alpha=0.25, color=self.colors[2])

        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=11, weight='bold')

        # Set y-axis limits
        ax.set_ylim(0, max(weights) * 1.1)

        # Add grid
        ax.grid(True, linestyle=':', alpha=0.5)

        # Add title
        ax.set_title(title, fontsize=14, weight='bold', pad=30)

        # Add value labels
        for angle, weight, criterion in zip(angles[:-1], weights[:-1], criteria):
            ax.text(
                angle,
                weight + 0.03,
                f'{weight:.2f}',
                ha='center',
                va='center',
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Criteria importance plot saved to {full_path}")
        return str(full_path)

    def plot_decision_comparison(
        self,
        metrics: Dict[str, Any],
        save_path: str,
        title: str = "Multi-Agent vs Individual Agents Performance"
    ) -> str:
        """
        Plot decision quality comparison chart.

        NEW: Compares multi-agent consensus against EACH individual agent.
        LEGACY: Falls back to single-agent baseline if individual comparisons unavailable.

        Args:
            metrics: Dictionary with 'individual_comparisons' (NEW) or 'baseline_comparison' (LEGACY)
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot

        Example (NEW format):
            >>> metrics = {
            ...     'individual_comparisons': {
            ...         'multi_agent_quality': 0.774,
            ...         'individual_agents': [
            ...             {'agent_name': 'Agent A', 'decision_quality': {'weighted_score': 0.685}, ...},
            ...             {'agent_name': 'Agent B', 'decision_quality': {'weighted_score': 0.620}, ...}
            ...         ],
            ...         'statistics': {'avg_quality': 0.521, 'min_quality': 0.350, 'max_quality': 0.685}
            ...     }
            ... }
        """
        logger.info(f"Plotting decision comparison to {save_path}")

        # Try NEW comprehensive comparison first
        if 'individual_comparisons' in metrics:
            return self._plot_comprehensive_comparison(metrics, save_path, title)

        # Fall back to LEGACY single-agent baseline
        elif 'baseline_comparison' in metrics:
            return self._plot_legacy_baseline_comparison(metrics, save_path, title)

        else:
            logger.warning("No comparison data available to plot")
            return ""

    def _plot_comprehensive_comparison(
        self,
        metrics: Dict[str, Any],
        save_path: str,
        title: str
    ) -> str:
        """
        Plot comprehensive multi-agent vs individual agents comparison.

        Shows:
        - Multi-agent consensus bar (highlighted)
        - All individual agent bars (color-coded by agreement)
        - Average individual quality line
        - Min/max range indicators
        """
        comp = metrics['individual_comparisons']
        stats = comp['statistics']

        ma_quality = comp['multi_agent_quality']
        avg_quality = stats['avg_quality']
        min_quality = stats['min_quality']
        max_quality = stats['max_quality']

        # Get individual agent data
        individual_agents = comp['individual_agents']

        # Sort by quality (descending)
        sorted_agents = sorted(
            individual_agents,
            key=lambda x: x['decision_quality']['weighted_score'],
            reverse=True
        )

        # Extract data
        agent_names = [ag['agent_name'] for ag in sorted_agents]
        agent_qualities = [ag['decision_quality']['weighted_score'] for ag in sorted_agents]
        agent_agrees = [ag.get('agrees_with_consensus', False) for ag in sorted_agents]

        # Create figure with more height for many agents
        n_agents = len(agent_names)
        fig_height = max(8, n_agents * 0.4)  # Dynamic height based on agent count
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Create horizontal bar chart
        y_pos = np.arange(n_agents + 1)  # +1 for multi-agent bar

        # Color individual agents by agreement
        colors = [self.colors[2] if agrees else self.colors[3] for agrees in agent_agrees]

        # Plot individual agent bars
        bars = ax.barh(
            y_pos[1:],  # Skip first position for multi-agent
            agent_qualities,
            color=colors,
            edgecolor='black',
            linewidth=1.0,
            alpha=0.7
        )

        # Plot multi-agent consensus bar (highlighted)
        ma_bar = ax.barh(
            y_pos[0],
            ma_quality,
            color=self.colors[0],
            edgecolor='black',
            linewidth=2.5,
            alpha=1.0,
            label='Multi-Agent Consensus'
        )

        # Add value labels on bars
        for i, (bar, quality) in enumerate(zip(bars, agent_qualities)):
            agree_marker = "✓" if agent_agrees[i] else "✗"
            ax.text(
                quality + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{quality:.3f} {agree_marker}',
                va='center',
                fontsize=9,
                weight='bold'
            )

        # Add multi-agent label
        ax.text(
            ma_quality + 0.01,
            ma_bar[0].get_y() + ma_bar[0].get_height() / 2,
            f'{ma_quality:.3f}',
            va='center',
            fontsize=11,
            weight='bold',
            color='darkblue'
        )

        # Add average line
        ax.axvline(
            avg_quality,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Average Individual: {avg_quality:.3f}',
            alpha=0.7
        )

        # Add min/max range shading
        ax.axvspan(
            min_quality,
            max_quality,
            alpha=0.1,
            color='gray',
            label=f'Individual Range: {min_quality:.3f}-{max_quality:.3f}'
        )

        # Set labels
        all_labels = ['Multi-Agent\nConsensus'] + agent_names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_labels, fontsize=10)

        # Customize plot
        ax.set_xlabel('Decision Quality Score', fontsize=12, weight='bold')
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle=':')

        # Add improvement annotation
        improvement_pct = ((ma_quality - avg_quality) / avg_quality) * 100
        improvement_text = f'Multi-Agent Improvement: {improvement_pct:+.1f}%'
        ax.text(
            0.98, 0.02,
            improvement_text,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=11,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

        # Add agreement stats
        agreement_rate = stats['agreement_rate_percent']
        num_agree = stats['num_agents_agree']
        total = stats['total_agents']
        agreement_text = f'Agreement: {num_agree}/{total} ({agreement_rate:.1f}%)'
        ax.text(
            0.02, 0.98,
            agreement_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=11,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # Legend
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Comprehensive comparison plot saved to {full_path}")
        return str(full_path)

    def _plot_legacy_baseline_comparison(
        self,
        metrics: Dict[str, Any],
        save_path: str,
        title: str
    ) -> str:
        """
        Plot legacy single-agent baseline comparison (for backward compatibility).

        Simple bar chart comparing multi-agent vs single-agent baseline.
        """
        logger.info("Using legacy baseline comparison format")

        comparison = metrics['baseline_comparison']

        # Extract comparison metrics
        metric_names = []
        multi_agent_scores = []
        single_agent_scores = []

        for metric_type, values in comparison.items():
            if metric_type in ['decision_quality', 'confidence', 'efficiency']:
                if isinstance(values, dict) and 'multi_agent' in values:
                    metric_names.append(metric_type.replace('_', ' ').title())
                    multi_agent_scores.append(values['multi_agent'])
                    single_agent_scores.append(values['single_agent'])

        if not metric_names:
            logger.warning("No valid comparison metrics found")
            return ""

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metric_names))
        width = 0.35

        # Plot bars
        bars1 = ax.bar(
            x - width/2,
            multi_agent_scores,
            width,
            label='Multi-Agent',
            color=self.colors[0],
            edgecolor='black',
            linewidth=1.5
        )

        bars2 = ax.bar(
            x + width/2,
            single_agent_scores,
            width,
            label='Single-Agent Baseline',
            color=self.colors[1],
            edgecolor='black',
            linewidth=1.5
        )

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    weight='bold'
                )

        autolabel(bars1)
        autolabel(bars2)

        # Add improvement annotations
        for i, (multi, single) in enumerate(zip(multi_agent_scores, single_agent_scores)):
            improvement = multi - single
            if improvement > 0:
                mid_height = max(multi, single) + 0.05
                ax.annotate(
                    f'↑ +{improvement:.1%}',
                    xy=(x[i], mid_height),
                    ha='center',
                    fontsize=10,
                    weight='bold',
                    color='green'
                )

        # Customize plot
        ax.set_xlabel('Performance Metric', fontsize=12, weight='bold')
        ax.set_ylabel('Score', fontsize=12, weight='bold')
        ax.set_title(title + ' (Legacy)', fontsize=14, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle=':')

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Legacy comparison plot saved to {full_path}")
        return str(full_path)

    def plot_agent_network(
        self,
        agent_profiles: Dict[str, Any],
        trust_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        save_path: str = "agent_network.png",
        title: str = "Expert Agent Network"
    ) -> str:
        """
        Plot agent network graph showing trust/interaction.

        Visualizes the multi-agent system as a network with agents as nodes
        and trust/interaction strength as edge weights.

        Args:
            agent_profiles: Dictionary of agent profiles with metadata
            trust_matrix: Optional trust/interaction weights between agents
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot

        Example:
            >>> profiles = {
            ...     'agent_1': {'name': 'Meteorologist', 'expertise': 'weather'},
            ...     'agent_2': {'name': 'Operations', 'expertise': 'logistics'}
            ... }
            >>> trust = {'agent_1': {'agent_2': 0.8}, 'agent_2': {'agent_1': 0.7}}
            >>> viz.plot_agent_network(profiles, trust, "network.png")
        """
        logger.info(f"Plotting agent network to {save_path}")

        if not agent_profiles:
            logger.warning("No agent profiles to plot")
            return ""

        # Create network graph
        G = nx.Graph()

        # Add nodes (agents)
        for agent_id, profile in agent_profiles.items():
            agent_name = profile.get('name', agent_id)
            expertise = profile.get('expertise', 'Unknown')
            G.add_node(agent_id, name=agent_name, expertise=expertise)

        # Add edges (trust/interaction)
        if trust_matrix:
            for agent_i, connections in trust_matrix.items():
                for agent_j, weight in connections.items():
                    if agent_i != agent_j and weight > 0:
                        G.add_edge(agent_i, agent_j, weight=weight)
        else:
            # If no trust matrix, create fully connected network
            agent_ids = list(agent_profiles.keys())
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    G.add_edge(agent_ids[i], agent_ids[j], weight=0.5)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        node_colors = [self.agent_colors[i % len(self.agent_colors)]
                       for i in range(len(G.nodes()))]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 5 for w in weights],
            alpha=0.4,
            edge_color='gray',
            ax=ax
        )

        # Draw labels
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=11,
            font_weight='bold',
            font_color='black',
            ax=ax
        )

        # Add expertise as sublabels
        for node, (x, y) in pos.items():
            expertise = G.nodes[node]['expertise']
            ax.text(
                x, y - 0.12,
                expertise,
                fontsize=9,
                ha='center',
                style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.agent_colors[i % len(self.agent_colors)], label=labels[node])
            for i, node in enumerate(G.nodes())
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            frameon=True,
            shadow=True,
            title='Agents'
        )

        # Customize plot
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Agent network plot saved to {full_path}")
        return str(full_path)

    def plot_method_comparison(
        self,
        comparative_results: Dict[str, Any],
        save_path: str = "method_comparison.png",
        title: str = "Aggregation Method Comparison"
    ) -> str:
        """
        Plot N-method comparison (ER, GAT, GAT_TRAINED, MCDA) across four metrics.

        Args:
            comparative_results: Dict with 'methods' keyed by method name
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting method comparison to {save_path}")

        if 'methods' not in comparative_results:
            logger.warning("No method comparison data available")
            return ""

        methods_data = comparative_results['methods']
        method_names = list(methods_data.keys())
        if not method_names:
            return ""

        colors = [self.METHOD_COLORS.get(m, '#95a5a6') for m in method_names]
        short_labels = [self._METHOD_LABELS.get(m, m) for m in method_names]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, weight='bold', y=1.02)

        n = len(method_names)
        x = np.arange(n)
        bar_w = min(0.6, 3.0 / n)

        def _labelled_bar(ax, values, ylabel, subplot_title, fmt='{:.3f}', ylim=(0, 1.0)):
            bars = ax.bar(x, values, bar_w, color=colors, edgecolor='black', linewidth=1.2)
            ax.set_ylabel(ylabel, fontsize=10, weight='bold')
            ax.set_title(subplot_title, fontsize=11, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=8, rotation=15, ha='right')
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        fmt.format(val), ha='center', va='bottom', fontsize=9, weight='bold')
            return bars

        # --- Plot 1: DQS ---
        dqs_vals = [methods_data[m].get('decision_quality_score', 0) for m in method_names]
        _labelled_bar(axes[0, 0], dqs_vals, 'Score', 'Decision Quality Score')

        # --- Plot 2: Consensus Level ---
        cons_vals = [methods_data[m].get('consensus_level', 0) for m in method_names]
        _labelled_bar(axes[0, 1], cons_vals, 'Consensus Level', 'Consensus Level')
        axes[0, 1].axhline(y=0.75, color='green', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='Threshold (75%)')
        axes[0, 1].legend(loc='lower right', fontsize=8)

        # --- Plot 3: Confidence ---
        conf_vals = [methods_data[m].get('confidence', 0) for m in method_names]
        _labelled_bar(axes[1, 0], conf_vals, 'Confidence', 'Decision Confidence',
                      fmt='{:.1%}')

        # --- Plot 4: Processing Time ---
        time_vals = [methods_data[m].get('processing_time_ms', 0) / 1000.0 for m in method_names]
        _labelled_bar(axes[1, 1], time_vals, 'Time (s)', 'Processing Time',
                      fmt='{:.1f}s', ylim=None)

        plt.tight_layout()
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Method comparison plot saved to {full_path}")
        return str(full_path)

    def plot_recommendation_comparison(
        self,
        comparative_results: Dict[str, Any],
        save_path: str = "recommendation_comparison.png",
        title: str = "Recommended Actions by Method"
    ) -> str:
        """
        Plot recommended actions for all methods (ER, GAT, GAT_TRAINED, MCDA).

        Draws one colour-coded box per method arranged in a grid and shows
        whether all methods agree on the top recommendation.

        Args:
            comparative_results: Dictionary with 'methods' keyed by method name
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting recommendation comparison to {save_path}")

        if 'methods' not in comparative_results:
            return ""

        methods_data = comparative_results['methods']
        method_names = list(methods_data.keys())
        if not method_names:
            return ""

        def _short(name: str) -> str:
            return name.replace('action_', '').replace('_', '\n').title()

        recommendations = {m: methods_data[m].get('recommended_alternative', 'Unknown')
                           for m in method_names}
        all_same = len(set(recommendations.values())) == 1

        n = len(method_names)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig_w = 6 * ncols
        fig_h = 3.5 * nrows + 1.5

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        fig.suptitle(title, fontsize=14, weight='bold', y=1.01)

        # Flatten axes to 1D for uniform iteration
        if n == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for idx, method in enumerate(method_names):
            ax = axes[idx]
            color = self.METHOD_COLORS.get(method, '#95a5a6')
            label = self._METHOD_LABELS.get(method, method)
            rec = recommendations[method]
            rec_display = _short(rec)

            # Coloured header bar
            ax.add_patch(plt.Rectangle((0, 0.65), 1, 0.35,
                                       facecolor=color, edgecolor='none', transform=ax.transAxes))
            ax.text(0.5, 0.825, label, ha='center', va='center',
                    fontsize=11, weight='bold', color='white', transform=ax.transAxes)

            # Recommendation body
            ax.text(0.5, 0.35, rec_display, ha='center', va='center',
                    fontsize=10, weight='bold', color='#2c3e50', transform=ax.transAxes,
                    multialignment='center')

            conf = methods_data[method].get('confidence', None)
            if conf is not None:
                ax.text(0.5, 0.08, f'Confidence: {conf:.1%}',
                        ha='center', va='center', fontsize=9, color='#555555',
                        transform=ax.transAxes)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        # Hide any surplus axes
        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        # Agreement banner
        agree_text = '✓ ALL METHODS AGREE' if all_same else '✗ METHODS DISAGREE'
        agree_color = '#27ae60' if all_same else '#e74c3c'
        fig.text(0.5, 0.01, agree_text, ha='center', fontsize=13, weight='bold',
                 color=agree_color,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=agree_color, linewidth=2))

        plt.tight_layout()
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Recommendation comparison plot saved to {full_path}")
        return str(full_path)

    def plot_comparative_summary(
        self,
        comparative_results: Dict[str, Any],
        save_path: str = "comparative_summary.png"
    ) -> str:
        """
        Generate a comprehensive N-method summary (ER, GAT, GAT_TRAINED, MCDA).

        Layout:
          - Top-left (2x2):  radar chart of DQS / Consensus / Confidence
          - Top-right:       recommendation table
          - Middle-right:    processing time bars
          - Bottom row:      delta bars vs ER baseline for each non-ER method

        Args:
            comparative_results: Full comparative analysis results
            save_path: Filename to save

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting comparative summary to {save_path}")

        if 'methods' not in comparative_results:
            return ""

        methods_data = comparative_results['methods']
        method_names = list(methods_data.keys())
        if not method_names:
            return ""

        scenario = comparative_results.get('scenario', 'Unknown')
        scenario_type = comparative_results.get('scenario_type', 'Unknown')

        fig = plt.figure(figsize=(16, 10), layout='constrained')
        fig.suptitle(
            f'Multi-Method Comparative Analysis\nScenario: {scenario} ({scenario_type.upper()})',
            fontsize=15, weight='bold', y=0.99
        )

        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

        # --- Radar chart ---
        ax_radar = fig.add_subplot(gs[0:2, 0:2], projection='polar')
        categories = ['DQS', 'Consensus', 'Confidence']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        for method in method_names:
            d = methods_data[method]
            vals = [d.get('decision_quality_score', 0),
                    d.get('consensus_level', 0),
                    d.get('confidence', 0)]
            vals_closed = vals + vals[:1]
            color = self.METHOD_COLORS.get(method, '#95a5a6')
            short = self._METHOD_LABELS.get(method, method)
            ax_radar.plot(angles_closed, vals_closed, 'o-', linewidth=2,
                          color=color, label=short, markersize=7)
            ax_radar.fill(angles_closed, vals_closed, alpha=0.15, color=color)

        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(categories, fontsize=10, weight='bold')
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Metrics', fontsize=12, weight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

        # --- Recommendations panel ---
        ax_rec = fig.add_subplot(gs[0, 2])
        ax_rec.axis('off')
        ax_rec.set_title('Recommendations', fontsize=11, weight='bold')

        all_recs = [methods_data[m].get('recommended_alternative', '') for m in method_names]
        all_same = len(set(all_recs)) == 1
        y_step = 0.95 / max(len(method_names), 1)
        for i, method in enumerate(method_names):
            color = self.METHOD_COLORS.get(method, '#95a5a6')
            rec = (methods_data[method].get('recommended_alternative', 'Unknown')
                   .replace('action_', '').replace('_', ' ').title())
            y = 0.95 - i * y_step
            ax_rec.text(0.02, y, f'{self._METHOD_LABELS.get(method, method)}:',
                        fontsize=8, weight='bold', color=color, va='top', transform=ax_rec.transAxes)
            ax_rec.text(0.02, y - y_step * 0.45, rec,
                        fontsize=8, va='top', transform=ax_rec.transAxes, color='#2c3e50')

        agree_text = '✓ All agree' if all_same else '✗ Differ'
        agree_color = '#27ae60' if all_same else '#e74c3c'
        ax_rec.text(0.5, 0.04, agree_text, ha='center', fontsize=11, weight='bold',
                    color=agree_color, transform=ax_rec.transAxes)

        # --- Processing time ---
        ax_time = fig.add_subplot(gs[1, 2])
        time_vals = [methods_data[m].get('processing_time_ms', 0) / 1000.0
                     for m in method_names]
        colors_t = [self.METHOD_COLORS.get(m, '#95a5a6') for m in method_names]
        short_names = [m.replace('_', '\n') for m in method_names]
        bars = ax_time.bar(range(len(method_names)), time_vals, color=colors_t,
                           edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, time_vals):
            ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{val:.1f}s', ha='center', fontsize=8, weight='bold')
        ax_time.set_xticks(range(len(method_names)))
        ax_time.set_xticklabels(short_names, fontsize=8)
        ax_time.set_ylabel('Seconds', fontsize=9)
        ax_time.set_title('Processing Time', fontsize=11, weight='bold')

        # --- Delta bars vs ER baseline ---
        ax_delta = fig.add_subplot(gs[2, :])
        non_er = [m for m in method_names if m != 'ER']
        if non_er and 'ER' in methods_data:
            er_base = methods_data['ER']
            metric_keys = [
                ('decision_quality_score', 'DQS'),
                ('consensus_level', 'Consensus'),
                ('confidence', 'Confidence'),
            ]
            n_metrics = len(metric_keys)
            n_methods = len(non_er)
            x = np.arange(n_metrics)
            total_w = 0.7
            w = total_w / n_methods

            for mi, method in enumerate(non_er):
                d = methods_data[method]
                deltas = [d.get(k, 0) - er_base.get(k, 0) for k, _ in metric_keys]
                offset = (mi - (n_methods - 1) / 2) * w
                color = self.METHOD_COLORS.get(method, '#95a5a6')
                short = self._METHOD_LABELS.get(method, method)
                bars_d = ax_delta.bar(x + offset, deltas, w,
                                      color=color, alpha=0.85, edgecolor='black',
                                      linewidth=1.0, label=short)
                for bar, val in zip(bars_d, deltas):
                    if abs(val) > 0.002:
                        ax_delta.text(bar.get_x() + bar.get_width() / 2,
                                      val + (0.004 if val >= 0 else -0.008),
                                      f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                                      fontsize=7)

            ax_delta.axhline(0, color='black', linewidth=1.2)
            ax_delta.set_xticks(x)
            ax_delta.set_xticklabels([lbl for _, lbl in metric_keys], fontsize=10)
            ax_delta.set_ylabel('Delta vs ER', fontsize=10, weight='bold')
            ax_delta.set_title('Improvement vs ER Baseline', fontsize=12, weight='bold')
            ax_delta.legend(fontsize=8, loc='upper right')
            ax_delta.grid(axis='y', alpha=0.3, linestyle=':')
        else:
            ax_delta.axis('off')

        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Comparative summary plot saved to {full_path}")
        return str(full_path)

    def plot_dqs_breakdown(
        self,
        decision: Dict[str, Any],
        save_path: str,
        title: str = "",
        aggregation_method: Optional[str] = None,
    ) -> str:
        """
        Plot per-alternative DQS decomposition as grouped horizontal bars.

        Shows three bars per alternative:
          - Aggregated belief score (labelled with the aggregation method name)
          - MCDA score (orange)
          - Combined DQS = 0.6*belief + 0.4*MCDA (green, bold)

        Alternatives are sorted by combined DQS descending.
        The recommended alternative is highlighted with a gold border.

        Args:
            decision: Decision dict with keys final_scores, er_scores, mcda_scores,
                      recommended_alternative
            save_path: Filename to save the plot
            title: Plot title (auto-generated from aggregation_method when empty)
            aggregation_method: Method name shown in labels/title (e.g. 'ER', 'GAT',
                                'GAT_TRAINED'). Defaults to 'ER' when not provided.

        Returns:
            Full path to saved plot
        """
        final_scores = decision.get('final_scores', {})
        er_scores    = decision.get('er_scores', {})
        mcda_scores  = decision.get('mcda_scores', {})
        recommended  = decision.get('recommended_alternative')

        # Build method-aware labels
        method_tag = (aggregation_method or 'ER').upper()
        METHOD_DISPLAY = {
            'ER':          'Evidential Reasoning (ER)',
            'GAT':         'GAT Belief Score',
            'GAT_TRAINED': 'GAT-Trained Belief Score',
            'MCDA':        'MCDA Score',
        }
        belief_label = METHOD_DISPLAY.get(method_tag, f'{method_tag} Belief Score')

        if not title:
            title = f"Alternative Ranking - DQS Decomposition (60% {method_tag} + 40% MCDA)"

        if not final_scores:
            logger.warning("plot_dqs_breakdown: no final_scores in decision, skipping")
            return ""

        # Sort alternatives by combined DQS descending
        alts = sorted(final_scores.keys(), key=lambda a: final_scores[a], reverse=True)
        n = len(alts)

        # Shorten labels for readability
        def _short(name: str) -> str:
            return name.replace('action_', '').replace('_', ' ').title()

        labels   = [_short(a) for a in alts]
        combined = [final_scores[a] for a in alts]
        er_vals  = [er_scores.get(a, 0.0) for a in alts]
        mcda_vals = [mcda_scores.get(a, 0.0) for a in alts]

        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.9)))

        y = np.arange(n)
        height = 0.25

        bar_er   = ax.barh(y + height,  er_vals,   height, label=belief_label,       color=self.colors[0], alpha=0.85)
        bar_mcda = ax.barh(y,           mcda_vals, height, label='MCDA Score',        color=self.colors[1], alpha=0.85)
        bar_dqs  = ax.barh(y - height,  combined,  height, label='Combined DQS',      color=self.colors[2], alpha=0.95)

        # Highlight recommended alternative with a gold border box
        if recommended in alts:
            idx = alts.index(recommended)
            for bar in [bar_er, bar_mcda, bar_dqs]:
                bar[idx].set_edgecolor('goldenrod')
                bar[idx].set_linewidth(2.0)

        # Value labels on the DQS bars
        for i, (bar, val) in enumerate(zip(bar_dqs, combined)):
            ax.text(
                val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}{"  *" if alts[i] == recommended else ""}',
                va='center', ha='left', fontsize=9,
                fontweight='bold' if alts[i] == recommended else 'normal'
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlim(0, min(1.0, max(combined) * 1.30))
        ax.axvline(x=0, color='grey', linewidth=0.5)
        ax.legend(loc='lower right', fontsize=9)

        # Footnote
        ax.text(
            0.01, -0.06,
            '* Recommended alternative     Gold border = recommended',
            transform=ax.transAxes, fontsize=8, color='grey', style='italic'
        )

        plt.tight_layout()

        full_path = str(self.output_dir / save_path)
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"DQS breakdown plot saved to {full_path}")
        return full_path

    def plot_dqs_method_deviation(
        self,
        comparative_results: Dict[str, Any],
        save_path: str,
        title: str = "DQS per Alternative - Method Comparison"
    ) -> str:
        """
        Plot per-alternative DQS scores for all methods with deviation vs ER.

        Left panel: grouped horizontal bars, one bar-group per alternative,
        one bar per method.  Right panel: delta bars (method - ER) per alternative.
        Recommended alternatives are highlighted with a gold border.

        Args:
            comparative_results: Dict from run_comparative_analysis with
                methods[name]['decision']['final_scores'] and
                methods[name]['recommended_alternative']
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot
        """
        methods_data = comparative_results.get('methods', {})
        method_names = list(methods_data.keys())

        # Collect scores and recommendations for every method
        all_scores: Dict[str, Dict[str, float]] = {}
        recommendations: Dict[str, str] = {}
        for m in method_names:
            dec = methods_data[m].get('decision', {})
            scores = dec.get('final_scores', {})
            all_scores[m] = scores
            recommendations[m] = methods_data[m].get('recommended_alternative', '')

        # Union of all alternatives, sorted by first available method score descending
        all_alts_set: set = set()
        for s in all_scores.values():
            all_alts_set.update(s.keys())

        if not all_alts_set:
            logger.warning("plot_dqs_method_deviation: no final_scores in results, skipping")
            return ""

        primary = method_names[0] if method_names else None
        all_alts = sorted(
            all_alts_set,
            key=lambda a: all_scores.get(primary, {}).get(a, 0.0),
            reverse=True
        )
        n = len(all_alts)

        def _short(name: str) -> str:
            return name.replace('action_', '').replace('_', ' ').title()

        labels = [_short(a) for a in all_alts]
        n_methods = len(method_names)
        height = min(0.7 / max(n_methods, 1), 0.28)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.9 + 1)),
                                 gridspec_kw={'width_ratios': [3, 1]})

        # --- Left panel: grouped bars per alternative ---
        ax = axes[0]
        y = np.arange(n)
        offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * height

        for mi, method in enumerate(method_names):
            scores = all_scores[method]
            vals = [scores.get(a, 0.0) for a in all_alts]
            color = self.METHOD_COLORS.get(method, '#95a5a6')
            short_label = self._METHOD_LABELS.get(method, method)
            rec = recommendations[method]

            bars = ax.barh(y + offsets[mi], vals, height,
                           label=short_label, color=color, alpha=0.85)

            # Gold border on recommended alternative
            if rec in all_alts:
                idx = all_alts.index(rec)
                bars[idx].set_edgecolor('goldenrod')
                bars[idx].set_linewidth(2.0)

            # Score labels
            for bar, val in zip(bars, vals):
                if val > 0.02:
                    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                            f'{val:.3f}', va='center', ha='left', fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Combined DQS Score', fontsize=11)
        ax.set_title('Score per Alternative', fontsize=11)
        all_vals_flat = [v for s in all_scores.values() for v in s.values()]
        ax.set_xlim(0, min(1.0, (max(all_vals_flat) if all_vals_flat else 1.0) * 1.35))
        ax.legend(fontsize=8, loc='lower right')

        # --- Right panel: delta vs ER (or first method) ---
        ax2 = axes[1]
        baseline_method = 'ER' if 'ER' in all_scores else method_names[0]
        baseline_scores = all_scores[baseline_method]

        for mi, method in enumerate(method_names):
            if method == baseline_method:
                continue
            scores = all_scores[method]
            deltas = [scores.get(a, 0.0) - baseline_scores.get(a, 0.0) for a in all_alts]
            color = self.METHOD_COLORS.get(method, '#95a5a6')
            short_label = self._METHOD_LABELS.get(method, method)
            ax2.barh(y + offsets[mi], deltas, height,
                     color=color, alpha=0.75, label=short_label)

        ax2.axvline(x=0, color='black', linewidth=1.0)
        ax2.set_yticks(y)
        ax2.set_yticklabels([])
        ax2.set_xlabel(f'Delta vs {baseline_method}', fontsize=9)
        ax2.set_title('Deviation', fontsize=11)
        if n_methods > 2:
            ax2.legend(fontsize=7)

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

        rec_parts = [f'{m}: {_short(r)}' for m, r in recommendations.items() if r]
        if rec_parts:
            fig.text(0.5, -0.02, '  |  '.join(rec_parts) + '  (gold border = recommended)',
                     ha='center', fontsize=8, color='#555555', style='italic')

        plt.tight_layout()
        full_path = str(self.output_dir / save_path)
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"DQS method deviation plot saved to {full_path}")
        return full_path

    def plot_dqs_er_gat_deviation(
        self,
        comparative_results: Dict[str, Any],
        save_path: str,
        title: str = "DQS per Alternative - ER vs GAT Deviation"
    ) -> str:
        """Backward-compatible alias for plot_dqs_method_deviation."""
        return self.plot_dqs_method_deviation(comparative_results, save_path, title)

    def plot_gat_training_result(
        self,
        weights_path: str = "models/gat_weights/gat_trained_weights.json",
        save_path: str = "gat_training_result.png"
    ) -> str:
        """
        Visualise GAT training outcome: learned vs prior weights, accuracy metrics,
        and run metadata.  Designed to be re-generated after each training run so
        progress can be tracked over time.

        Layout (2 rows):
          Top:    grouped bar chart - prior vs learned weights for all 4 coefficients
          Bottom-left:  metric bars (top-1 accuracy, mean rank percentile)
          Bottom-right: summary table (samples, scenarios, timestamp, convergence status)

        Args:
            weights_path: Path to gat_trained_weights.json
            save_path:    Filename to save inside self.output_dir

        Returns:
            Full path to saved plot, or "" if weights file not found.
        """
        import json as _json
        from pathlib import Path as _Path
        from datetime import datetime as _dt

        wpath = _Path(weights_path)
        if not wpath.exists():
            logger.warning(f"plot_gat_training_result: weights file not found at {wpath}")
            return ""

        with open(wpath) as f:
            data = _json.load(f)

        labels   = data.get("labels", ["w_confidence", "w_relevance", "w_certainty", "w_similarity"])
        learned  = data.get("weights", [0.4, 0.3, 0.3, 0.2])
        prior    = data.get("prior_weights", [0.4, 0.3, 0.3, 0.2])
        deltas   = data.get("weight_delta", [w - p for w, p in zip(learned, prior)])
        metrics  = data.get("training_metrics", {})
        n_samples = data.get("n_training_samples", 0)
        scenarios = data.get("training_scenarios", [])
        timestamp = data.get("timestamp", "")
        try:
            ts_str = _dt.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = timestamp[:16] if timestamp else "unknown"

        prior_m   = metrics.get("prior",   {})
        trained_m = metrics.get("trained", {})

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle("GAT Attention Weight Training - Result Summary",
                     fontsize=15, weight='bold', y=0.99)

        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        er_blue  = self.METHOD_COLORS['ER']
        gat_green = self.METHOD_COLORS['GAT_TRAINED']

        # ---- Top: grouped bar chart ----
        ax_w = fig.add_subplot(gs[0, :])
        x = np.arange(len(labels))
        w = 0.32
        short_labels = [l.replace('w_', '').replace('_', ' ').title() for l in labels]

        bars_p = ax_w.bar(x - w / 2, prior,   w, label='Prior (hand-crafted)',
                          color=er_blue,  alpha=0.85, edgecolor='black', linewidth=1.1)
        bars_l = ax_w.bar(x + w / 2, learned, w, label='Learned',
                          color=gat_green, alpha=0.85, edgecolor='black', linewidth=1.1)

        # Value labels
        for bar, val in zip(bars_p, prior):
            ax_w.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=9, color=er_blue, weight='bold')
        for bar, val, d in zip(bars_l, learned, deltas):
            ax_w.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=9, color=gat_green, weight='bold')
            # Delta annotation below bar
            arrow_col = '#27ae60' if d >= 0 else '#e74c3c'
            ax_w.text(bar.get_x() + bar.get_width() / 2, -0.025,
                      f'{d:+.4f}', ha='center', va='top', fontsize=8, color=arrow_col, style='italic')

        ax_w.set_xticks(x)
        ax_w.set_xticklabels(short_labels, fontsize=11, weight='bold')
        ax_w.set_ylim(-0.05, max(max(prior), max(learned)) * 1.3)
        ax_w.set_ylabel('Weight Value', fontsize=10, weight='bold')
        ax_w.set_title('Attention Weight: Prior vs Learned  (delta shown below bars)', fontsize=11)
        ax_w.axhline(0, color='grey', linewidth=0.5)
        ax_w.legend(fontsize=10, loc='upper right')
        ax_w.grid(axis='y', alpha=0.3, linestyle=':')

        # ---- Bottom-left: metric comparison bars ----
        ax_m = fig.add_subplot(gs[1, 0])
        metric_names  = ['Top-1\nAccuracy', 'Mean Rank\nPercentile']
        prior_vals    = [prior_m.get('top1_accuracy', 0),
                         prior_m.get('mean_rank_percentile', 0)]
        trained_vals  = [trained_m.get('top1_accuracy', 0),
                         trained_m.get('mean_rank_percentile', 0)]

        xm = np.arange(len(metric_names))
        wm = 0.32
        bp = ax_m.bar(xm - wm / 2, prior_vals,   wm, label='Prior',   color=er_blue,   alpha=0.85, edgecolor='black')
        bt = ax_m.bar(xm + wm / 2, trained_vals, wm, label='Learned', color=gat_green, alpha=0.85, edgecolor='black')

        for bar, val in zip(bp, prior_vals):
            ax_m.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{val:.1%}', ha='center', fontsize=9, weight='bold', color=er_blue)
        for bar, val in zip(bt, trained_vals):
            ax_m.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{val:.1%}', ha='center', fontsize=9, weight='bold', color=gat_green)

        ax_m.set_xticks(xm)
        ax_m.set_xticklabels(metric_names, fontsize=10)
        ax_m.set_ylim(0, 1.15)
        ax_m.set_title('Training Metrics', fontsize=11, weight='bold')
        ax_m.legend(fontsize=9)
        ax_m.grid(axis='y', alpha=0.3, linestyle=':')

        # ---- Bottom-right: summary table ----
        ax_t = fig.add_subplot(gs[1, 1])
        ax_t.axis('off')
        ax_t.set_title('Training Run Metadata', fontsize=11, weight='bold')

        n_samp = prior_m.get('n_samples', n_samples)
        mean_rank_p = prior_m.get('mean_rank', 0)
        mean_rank_t = trained_m.get('mean_rank', 0)

        table_rows = [
            ['Training samples',    str(n_samples)],
            ['Evaluated samples',   str(n_samp)],
            ['Scenarios',           ', '.join(s.title() for s in scenarios)],
            ['Timestamp',           ts_str],
            ['Prior top-1',         f"{prior_m.get('top1_accuracy', 0):.1%}"],
            ['Learned top-1',       f"{trained_m.get('top1_accuracy', 0):.1%}"],
            ['Prior mean rank',     f"{mean_rank_p:.3f}"],
            ['Learned mean rank',   f"{mean_rank_t:.3f}"],
            ['Improvement (rank)',  f"{mean_rank_p - mean_rank_t:+.3f}"],
        ]

        y_start = 0.95
        row_h = 0.105
        for i, (key, val) in enumerate(table_rows):
            y = y_start - i * row_h
            bg = '#f0f0f0' if i % 2 == 0 else 'white'
            ax_t.add_patch(plt.Rectangle((0, y - row_h * 0.75), 1, row_h * 0.9,
                                         facecolor=bg, edgecolor='none',
                                         transform=ax_t.transAxes))
            ax_t.text(0.03, y - row_h * 0.3, key + ':', fontsize=9, weight='bold',
                      transform=ax_t.transAxes, va='center', color='#333333')
            ax_t.text(0.55, y - row_h * 0.3, val, fontsize=9,
                      transform=ax_t.transAxes, va='center', color='#555555')

        plt.tight_layout()
        full_path = str(self.output_dir / save_path)
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"GAT training result plot saved to {full_path}")
        return full_path

    def generate_all_plots(
        self,
        results: Dict[str, Any],
        output_subdir: Optional[str] = None,
        aggregation_method: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate all visualization plots from results dictionary.

        Convenience method to create all standard plots at once.

        Args:
            results: Dictionary containing all result data with keys:
                - agent_assessments: For belief distributions
                - consensus_history: For consensus evolution
                - criteria_weights: For criteria importance
                - metrics: For decision comparison
                - agent_profiles: For agent network
                - trust_matrix: Optional, for agent network
            output_subdir: Optional subdirectory within output_dir
            aggregation_method: Aggregation method used ('ER' or 'GAT') - displayed in plot titles

        Returns:
            Dictionary mapping plot type to saved file path

        Example:
            >>> results = {
            ...     'agent_assessments': {...},
            ...     'consensus_history': [...],
            ...     'criteria_weights': {...},
            ...     'metrics': {...}
            ... }
            >>> paths = viz.generate_all_plots(results, "scenario_1", aggregation_method="GAT")
            >>> print(paths['beliefs'])
        """
        logger.info("Generating all visualization plots")

        # Set up output directory
        if output_subdir:
            original_dir = self.output_dir
            self.output_dir = self.output_dir / output_subdir
            self.output_dir.mkdir(exist_ok=True, parents=True)

        saved_paths = {}

        # Build method suffix for titles
        method_label = ""
        if aggregation_method:
            method_upper = aggregation_method.upper()
            if method_upper == 'ER':
                method_label = " [Evidential Reasoning]"
            elif method_upper == 'GAT':
                method_label = " [Graph Attention Network]"
            else:
                method_label = f" [{method_upper}]"

        # 1. Belief Distributions
        if 'agent_assessments' in results:
            try:
                path = self.plot_belief_distributions(
                    results['agent_assessments'],
                    "belief_distributions.png",
                    title=f"Agent Belief Distributions{method_label}"
                )
                saved_paths['beliefs'] = path
            except Exception as e:
                logger.error(f"Failed to plot belief distributions: {e}")

        # 2. Consensus Evolution
        if 'consensus_history' in results:
            try:
                path = self.plot_consensus_evolution(
                    results['consensus_history'],
                    "consensus_evolution.png",
                    title=f"Consensus Evolution Over Iterations{method_label}"
                )
                saved_paths['consensus'] = path
            except Exception as e:
                logger.error(f"Failed to plot consensus evolution: {e}")

        # 3. Criteria Importance
        if 'criteria_weights' in results:
            try:
                path = self.plot_criteria_importance(
                    results['criteria_weights'],
                    "criteria_importance.png",
                    title=f"Decision Criteria Importance{method_label}"
                )
                saved_paths['criteria'] = path
            except Exception as e:
                logger.error(f"Failed to plot criteria importance: {e}")

        # 4. Decision Comparison
        if 'metrics' in results:
            try:
                path = self.plot_decision_comparison(
                    results['metrics'],
                    "decision_comparison.png",
                    title=f"Multi-Agent vs Individual Agents Performance{method_label}"
                )
                saved_paths['comparison'] = path
            except Exception as e:
                logger.error(f"Failed to plot decision comparison: {e}")

        # 5. Agent Network
        if 'agent_profiles' in results:
            try:
                trust_matrix = results.get('trust_matrix', None)
                path = self.plot_agent_network(
                    results['agent_profiles'],
                    trust_matrix,
                    "agent_network.png",
                    title=f"Expert Agent Network{method_label}"
                )
                saved_paths['network'] = path
            except Exception as e:
                logger.error(f"Failed to plot agent network: {e}")

        # 6. DQS Breakdown (per-alternative ER / MCDA / Combined)
        if 'decision' in results:
            try:
                path = self.plot_dqs_breakdown(
                    results['decision'],
                    "dqs_breakdown.png",
                    aggregation_method=aggregation_method,
                )
                if path:
                    saved_paths['dqs_breakdown'] = path
            except Exception as e:
                logger.error(f"Failed to plot DQS breakdown: {e}")

        # Restore original directory if changed
        if output_subdir:
            self.output_dir = original_dir

        logger.info(f"Generated {len(saved_paths)} plots: {list(saved_paths.keys())}")
        return saved_paths

    def __repr__(self) -> str:
        """String representation."""
        return f"SystemVisualizer(output_dir='{self.output_dir}', dpi={self.dpi})"
