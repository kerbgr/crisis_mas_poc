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
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.08:  # Only show label if bar is large enough
                    height = bar.get_height()
                    y_pos = bar.get_y() + height / 2
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f'{val:.2f}',
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='white',
                        weight='bold'
                    )

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
        title: str = "ER vs GAT Aggregation Method Comparison"
    ) -> str:
        """
        Plot ER vs GAT method comparison with multiple metrics.

        Creates a comprehensive comparison chart showing:
        - Decision Quality Score
        - Consensus Level
        - Decision Confidence
        - Processing Time

        Args:
            comparative_results: Dictionary with 'methods' containing ER and GAT results
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting method comparison to {save_path}")

        if 'methods' not in comparative_results:
            logger.warning("No method comparison data available")
            return ""

        methods = comparative_results['methods']
        if 'ER' not in methods or 'GAT' not in methods:
            logger.warning("Missing ER or GAT results for comparison")
            return ""

        er_data = methods['ER']
        gat_data = methods['GAT']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, weight='bold', y=1.02)

        # Color scheme for methods
        er_color = '#3498db'  # Blue for ER
        gat_color = '#e74c3c'  # Red for GAT

        # --- Plot 1: Decision Quality Score ---
        ax1 = axes[0, 0]
        metrics = ['Decision Quality\nScore']
        er_vals = [er_data.get('decision_quality_score', 0)]
        gat_vals = [gat_data.get('decision_quality_score', 0)]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, er_vals, width, label='Evidential Reasoning (ER)',
                        color=er_color, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, gat_vals, width, label='Graph Attention Network (GAT)',
                        color=gat_color, edgecolor='black', linewidth=1.5)

        ax1.set_ylabel('Score', fontsize=11, weight='bold')
        ax1.set_title('Decision Quality Score', fontsize=12, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1.0)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle=':')

        # Add value labels
        for bar, val in zip(bars1, er_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        for bar, val in zip(bars2, gat_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

        # --- Plot 2: Consensus Level ---
        ax2 = axes[0, 1]
        er_consensus = er_data.get('consensus_level', 0)
        gat_consensus = gat_data.get('consensus_level', 0)

        bars1 = ax2.bar(['ER'], [er_consensus], width=0.5, color=er_color,
                        edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(['GAT'], [gat_consensus], width=0.5, color=gat_color,
                        edgecolor='black', linewidth=1.5)

        ax2.set_ylabel('Consensus Level', fontsize=11, weight='bold')
        ax2.set_title('Consensus Level Comparison', fontsize=12, weight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=0.75, color='green', linestyle='--', linewidth=2,
                    alpha=0.7, label='Threshold (75%)')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle=':')

        # Add value labels
        ax2.text(0, er_consensus + 0.02, f'{er_consensus:.1%}',
                 ha='center', va='bottom', fontsize=11, weight='bold')
        ax2.text(1, gat_consensus + 0.02, f'{gat_consensus:.1%}',
                 ha='center', va='bottom', fontsize=11, weight='bold')

        # Delta annotation
        delta = gat_consensus - er_consensus
        delta_color = 'green' if delta >= 0 else 'red'
        ax2.annotate(f'Δ = {delta:+.1%}', xy=(0.5, max(er_consensus, gat_consensus) + 0.08),
                     ha='center', fontsize=10, weight='bold', color=delta_color)

        # --- Plot 3: Decision Confidence ---
        ax3 = axes[1, 0]
        er_conf = er_data.get('confidence', 0)
        gat_conf = gat_data.get('confidence', 0)

        # Create pie-style confidence display
        categories = ['Evidential Reasoning\n(ER)', 'Graph Attention\nNetwork (GAT)']
        confidences = [er_conf, gat_conf]
        colors = [er_color, gat_color]

        bars = ax3.barh(categories, confidences, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_xlim(0, 1.0)
        ax3.set_xlabel('Confidence Score', fontsize=11, weight='bold')
        ax3.set_title('Decision Confidence', fontsize=12, weight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle=':')

        # Add value labels
        for bar, val in zip(bars, confidences):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.1%}', ha='left', va='center', fontsize=11, weight='bold')

        # --- Plot 4: Processing Time ---
        ax4 = axes[1, 1]
        er_time = er_data.get('processing_time_ms', 0) / 1000  # Convert to seconds
        gat_time = gat_data.get('processing_time_ms', 0) / 1000

        bars = ax4.bar(['ER', 'GAT'], [er_time, gat_time], color=[er_color, gat_color],
                       edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Time (seconds)', fontsize=11, weight='bold')
        ax4.set_title('Processing Time', fontsize=12, weight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle=':')

        # Add value labels
        for bar, val in zip(bars, [er_time, gat_time]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}s', ha='center', va='bottom', fontsize=11, weight='bold')

        # Add overhead annotation
        overhead = gat_time - er_time
        ax4.annotate(f'GAT overhead: {overhead:+.1f}s',
                     xy=(0.5, max(er_time, gat_time) * 1.15),
                     ha='center', fontsize=10, weight='bold',
                     color='orange' if overhead > 0 else 'green')

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Method comparison plot saved to {full_path}")
        return str(full_path)

    def plot_recommendation_comparison(
        self,
        comparative_results: Dict[str, Any],
        save_path: str = "recommendation_comparison.png",
        title: str = "ER vs GAT: Recommended Actions"
    ) -> str:
        """
        Plot comparison of recommended actions between ER and GAT.

        Shows which action each method recommends and whether they agree.

        Args:
            comparative_results: Dictionary with method results
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting recommendation comparison to {save_path}")

        if 'methods' not in comparative_results:
            return ""

        methods = comparative_results['methods']
        comparison = comparative_results.get('comparison', {})

        er_rec = methods['ER'].get('recommended_alternative', 'Unknown')
        gat_rec = methods['GAT'].get('recommended_alternative', 'Unknown')
        same = comparison.get('same_recommendation', er_rec == gat_rec)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create comparison visualization
        er_color = '#3498db'
        gat_color = '#e74c3c'
        agree_color = '#27ae60' if same else '#f39c12'

        # Draw method boxes
        er_box = plt.Rectangle((0.1, 0.4), 0.35, 0.4, facecolor=er_color,
                                edgecolor='black', linewidth=2)
        gat_box = plt.Rectangle((0.55, 0.4), 0.35, 0.4, facecolor=gat_color,
                                 edgecolor='black', linewidth=2)
        ax.add_patch(er_box)
        ax.add_patch(gat_box)

        # Method labels
        ax.text(0.275, 0.85, 'Evidential Reasoning (ER)', ha='center', va='center',
                fontsize=14, weight='bold', color=er_color)
        ax.text(0.725, 0.85, 'Graph Attention Network (GAT)', ha='center', va='center',
                fontsize=14, weight='bold', color=gat_color)

        # Recommendation text (clean up action IDs for display)
        er_display = er_rec.replace('action_', '').replace('_', ' ').title()
        gat_display = gat_rec.replace('action_', '').replace('_', ' ').title()

        ax.text(0.275, 0.6, er_display, ha='center', va='center',
                fontsize=11, weight='bold', color='white', wrap=True)
        ax.text(0.725, 0.6, gat_display, ha='center', va='center',
                fontsize=11, weight='bold', color='white', wrap=True)

        # Agreement indicator
        if same:
            ax.annotate('', xy=(0.55, 0.6), xytext=(0.45, 0.6),
                        arrowprops=dict(arrowstyle='<->', color=agree_color, lw=3))
            ax.text(0.5, 0.2, '✓ SAME RECOMMENDATION', ha='center', va='center',
                    fontsize=14, weight='bold', color=agree_color,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=agree_color, linewidth=2))
        else:
            ax.text(0.5, 0.2, '✗ DIFFERENT RECOMMENDATIONS', ha='center', va='center',
                    fontsize=14, weight='bold', color=agree_color,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=agree_color, linewidth=2))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        # Save figure
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
        Generate a comprehensive summary visualization of ER vs GAT comparison.

        Creates a single figure combining all key comparative metrics.

        Args:
            comparative_results: Full comparative analysis results
            save_path: Filename to save

        Returns:
            Full path to saved plot
        """
        logger.info(f"Plotting comparative summary to {save_path}")

        if 'methods' not in comparative_results:
            return ""

        methods = comparative_results['methods']
        comparison = comparative_results.get('comparison', {})
        scenario = comparative_results.get('scenario', 'Unknown')
        scenario_type = comparative_results.get('scenario_type', 'Unknown')

        er = methods['ER']
        gat = methods['GAT']

        # Create figure
        fig = plt.figure(figsize=(16, 10))

        # Title with scenario info
        fig.suptitle(f'ER vs GAT Comparative Analysis\nScenario: {scenario} ({scenario_type.upper()})',
                     fontsize=16, weight='bold', y=0.98)

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Colors
        er_color = '#3498db'
        gat_color = '#e74c3c'

        # --- Subplot 1: Metrics Radar Chart ---
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='polar')

        categories = ['DQS', 'Consensus', 'Confidence']
        er_vals = [er.get('decision_quality_score', 0),
                   er.get('consensus_level', 0),
                   er.get('confidence', 0)]
        gat_vals = [gat.get('decision_quality_score', 0),
                    gat.get('consensus_level', 0),
                    gat.get('confidence', 0)]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        er_vals_closed = er_vals + er_vals[:1]
        gat_vals_closed = gat_vals + gat_vals[:1]
        angles_closed = angles + angles[:1]

        ax1.plot(angles_closed, er_vals_closed, 'o-', linewidth=2, color=er_color,
                 label='ER', markersize=8)
        ax1.fill(angles_closed, er_vals_closed, alpha=0.25, color=er_color)
        ax1.plot(angles_closed, gat_vals_closed, 's-', linewidth=2, color=gat_color,
                 label='GAT', markersize=8)
        ax1.fill(angles_closed, gat_vals_closed, alpha=0.25, color=gat_color)

        ax1.set_xticks(angles)
        ax1.set_xticklabels(categories, fontsize=11, weight='bold')
        ax1.set_ylim(0, 1)
        ax1.set_title('Performance Metrics', fontsize=12, weight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # --- Subplot 2: Recommendations ---
        ax2 = fig.add_subplot(gs[0, 2])
        same_rec = comparison.get('same_recommendation', False)

        er_rec = er.get('recommended_alternative', '').replace('action_', '').replace('_', '\n').title()
        gat_rec = gat.get('recommended_alternative', '').replace('action_', '').replace('_', '\n').title()

        ax2.text(0.5, 0.8, 'ER Recommends:', ha='center', fontsize=10, weight='bold', color=er_color)
        ax2.text(0.5, 0.65, er_rec, ha='center', fontsize=9, wrap=True)
        ax2.text(0.5, 0.4, 'GAT Recommends:', ha='center', fontsize=10, weight='bold', color=gat_color)
        ax2.text(0.5, 0.25, gat_rec, ha='center', fontsize=9, wrap=True)

        status = '✓ AGREE' if same_rec else '✗ DIFFER'
        status_color = '#27ae60' if same_rec else '#f39c12'
        ax2.text(0.5, 0.05, status, ha='center', fontsize=12, weight='bold', color=status_color)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Recommendations', fontsize=12, weight='bold')

        # --- Subplot 3: Processing Time ---
        ax3 = fig.add_subplot(gs[1, 2])
        er_time = er.get('processing_time_ms', 0) / 1000
        gat_time = gat.get('processing_time_ms', 0) / 1000

        bars = ax3.bar(['ER', 'GAT'], [er_time, gat_time], color=[er_color, gat_color],
                       edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Seconds', fontsize=10)
        ax3.set_title('Processing Time', fontsize=12, weight='bold')

        for bar, val in zip(bars, [er_time, gat_time]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}s', ha='center', fontsize=9, weight='bold')

        # --- Subplot 4: Delta Summary ---
        ax4 = fig.add_subplot(gs[2, :])

        delta_labels = ['DQS Δ', 'Consensus Δ', 'Confidence Δ', 'Time Δ']
        delta_vals = [
            comparison.get('decision_quality_delta', 0),
            comparison.get('consensus_delta', 0),
            comparison.get('confidence_delta', 0),
            comparison.get('processing_time_delta_ms', 0) / 1000
        ]

        # Color bars by direction (green = GAT better for quality metrics, blue for time)
        colors = []
        for i, val in enumerate(delta_vals):
            if i < 3:  # Quality metrics
                colors.append('#27ae60' if val >= 0 else '#e74c3c')
            else:  # Time (lower is better)
                colors.append('#e74c3c' if val > 0 else '#27ae60')

        bars = ax4.barh(delta_labels, delta_vals, color=colors, edgecolor='black', linewidth=1.5)

        # Add zero line
        ax4.axvline(x=0, color='black', linewidth=2)

        # Add value labels
        for bar, val in zip(bars, delta_vals):
            x_pos = val + 0.005 if val >= 0 else val - 0.005
            ha = 'left' if val >= 0 else 'right'
            label = f'{val:+.3f}' if abs(val) < 10 else f'{val:+.1f}s'
            ax4.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                     ha=ha, va='center', fontsize=10, weight='bold')

        ax4.set_xlabel('GAT - ER (positive = GAT higher)', fontsize=10, weight='bold')
        ax4.set_title('Difference Analysis (GAT - ER)', fontsize=12, weight='bold')
        ax4.grid(axis='x', alpha=0.3, linestyle=':')

        plt.tight_layout()

        # Save figure
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Comparative summary plot saved to {full_path}")
        return str(full_path)

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

        # Restore original directory if changed
        if output_subdir:
            self.output_dir = original_dir

        logger.info(f"Generated {len(saved_paths)} plots: {list(saved_paths.keys())}")
        return saved_paths

    def __repr__(self) -> str:
        """String representation."""
        return f"SystemVisualizer(output_dir='{self.output_dir}', dpi={self.dpi})"
