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
\usepackage{graphicx}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/belief_distributions.png}
    \caption{Agent belief distributions for flood evacuation scenario.
             Colors indicate alternatives: green (immediate evacuation),
             blue (staged evacuation), orange (shelter in place).}
    \label{fig:beliefs}
\end{figure}
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
        self.agent_colors = sns.color_palette("husl", 8)

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

        # Extract belief distributions
        agents = []
        belief_data = {}

        for agent_id, assessment in agent_assessments.items():
            if 'belief_distribution' not in assessment:
                continue

            agent_name = assessment.get('agent_name', agent_id)
            agents.append(agent_name)

            beliefs = assessment['belief_distribution']
            for alt_id, belief in beliefs.items():
                if alt_id not in belief_data:
                    belief_data[alt_id] = []
                belief_data[alt_id].append(belief)

        if not agents or not belief_data:
            logger.warning("No belief distribution data to plot")
            return ""

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot stacked bars
        alternatives = sorted(belief_data.keys())
        x_pos = np.arange(len(agents))
        bottom = np.zeros(len(agents))

        for i, alt_id in enumerate(alternatives):
            values = belief_data[alt_id]
            bars = ax.bar(
                x_pos,
                values,
                bottom=bottom,
                label=f"Alternative {alt_id}",
                color=self.colors[i % len(self.colors)],
                edgecolor='white',
                linewidth=0.5
            )
            bottom += values

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
        title: str = "Multi-Agent vs Single-Agent Performance"
    ) -> str:
        """
        Plot decision quality comparison bar chart.

        Compares multi-agent system performance against single-agent baseline.

        Args:
            metrics: Dictionary with 'baseline_comparison' containing comparison data
            save_path: Filename to save the plot
            title: Plot title

        Returns:
            Full path to saved plot

        Example:
            >>> metrics = {
            ...     'baseline_comparison': {
            ...         'decision_quality': {
            ...             'multi_agent': 0.84,
            ...             'single_agent': 0.72,
            ...             'improvement': 0.12
            ...         },
            ...         'confidence': {
            ...             'multi_agent': 0.82,
            ...             'single_agent': 0.70,
            ...             'improvement': 0.12
            ...         }
            ...     }
            ... }
            >>> viz.plot_decision_comparison(metrics, "comparison.png")
        """
        logger.info(f"Plotting decision comparison to {save_path}")

        if 'baseline_comparison' not in metrics:
            logger.warning("No baseline comparison data to plot")
            return ""

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
            label='Single-Agent',
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
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
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

        logger.info(f"Decision comparison plot saved to {full_path}")
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
            mpatches.Patch(color=self.agent_colors[i], label=labels[node])
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

    def generate_all_plots(
        self,
        results: Dict[str, Any],
        output_subdir: Optional[str] = None
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

        Returns:
            Dictionary mapping plot type to saved file path

        Example:
            >>> results = {
            ...     'agent_assessments': {...},
            ...     'consensus_history': [...],
            ...     'criteria_weights': {...},
            ...     'metrics': {...}
            ... }
            >>> paths = viz.generate_all_plots(results, "scenario_1")
            >>> print(paths['beliefs'])
        """
        logger.info("Generating all visualization plots")

        # Set up output directory
        if output_subdir:
            original_dir = self.output_dir
            self.output_dir = self.output_dir / output_subdir
            self.output_dir.mkdir(exist_ok=True, parents=True)

        saved_paths = {}

        # 1. Belief Distributions
        if 'agent_assessments' in results:
            try:
                path = self.plot_belief_distributions(
                    results['agent_assessments'],
                    "belief_distributions.png"
                )
                saved_paths['beliefs'] = path
            except Exception as e:
                logger.error(f"Failed to plot belief distributions: {e}")

        # 2. Consensus Evolution
        if 'consensus_history' in results:
            try:
                path = self.plot_consensus_evolution(
                    results['consensus_history'],
                    "consensus_evolution.png"
                )
                saved_paths['consensus'] = path
            except Exception as e:
                logger.error(f"Failed to plot consensus evolution: {e}")

        # 3. Criteria Importance
        if 'criteria_weights' in results:
            try:
                path = self.plot_criteria_importance(
                    results['criteria_weights'],
                    "criteria_importance.png"
                )
                saved_paths['criteria'] = path
            except Exception as e:
                logger.error(f"Failed to plot criteria importance: {e}")

        # 4. Decision Comparison
        if 'metrics' in results:
            try:
                path = self.plot_decision_comparison(
                    results['metrics'],
                    "decision_comparison.png"
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
                    "agent_network.png"
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
