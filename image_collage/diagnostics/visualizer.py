import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional

from .collector import DiagnosticsData, DiagnosticsCollector
from ..cli.helpers import calculate_selection_pressure


class DiagnosticsVisualizer:
    """Creates enhanced comprehensive visualizations of genetic algorithm diagnostics.

    Generates 8+ visual reports including:
    - Dashboard with properly aligned statistics
    - Dual-shaded fitness evolution (best-average AND average-worst regions)
    - Genetic operations with clear effectiveness explanations
    - Accurate performance metrics with corrected success rates
    - Population analysis with selection pressure definition
    - Color-coded evolution grid with explanations and color bar
    - Fitness distribution and convergence analysis
    """

    def __init__(self, style: str = 'default'):
        """Initialize visualizer with plotting style."""
        try:
            # Try seaborn style first, fallback to default
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('ggplot')
            except:
                plt.style.use('default')

        # Set color palette
        self.colors = {
            'best': '#2E86AB',
            'average': '#A23B72',
            'worst': '#F18F01',
            'mutations': '#C73E1D',
            'crossovers': '#F79D1E',
            'diversity': '#06A77D'
        }

    def create_full_report(self, collector: DiagnosticsCollector, output_folder: str):
        """Generate a complete visual diagnostics report."""
        folder_path = Path(output_folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        data = collector.data

        # Create individual plots
        self._plot_fitness_evolution(data, folder_path)
        self._plot_fitness_distribution(data, folder_path)
        self._plot_genetic_operations(data, folder_path)
        self._plot_performance_metrics(data, folder_path)
        self._plot_population_analysis(data, folder_path)
        self._create_evolution_grid(data, folder_path)
        self._create_dashboard(data, folder_path)

        # New comprehensive diversity plots
        self._plot_comprehensive_diversity(data, folder_path)
        self._plot_spatial_diversity(data, folder_path)
        self._plot_advanced_metrics(data, folder_path)

        print(f"10 diagnostics visualizations + 3 data files saved to: {folder_path}")

    def _plot_fitness_evolution(self, data: DiagnosticsData, folder_path: Path):
        """Plot enhanced fitness evolution with dual-shaded regions.

        Creates visualization showing best, average, and worst fitness with:
        - Best-average shaded region (blue)
        - Average-worst shaded region (orange)
        - Fitness improvement rate subplot
        """
        if not data.generations:
            return

        generations = [g.generation for g in data.generations]
        best_fitness = [g.best_fitness for g in data.generations]
        avg_fitness = [g.average_fitness for g in data.generations]
        worst_fitness = [g.worst_fitness for g in data.generations]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Main fitness plot
        ax1.plot(generations, best_fitness, label='Best Fitness',
                color=self.colors['best'], linewidth=2)
        ax1.plot(generations, avg_fitness, label='Average Fitness',
                color=self.colors['average'], linewidth=2)
        ax1.plot(generations, worst_fitness, label='Worst Fitness',
                color=self.colors['worst'], linewidth=1.5, alpha=0.7)

        ax1.fill_between(generations, best_fitness, avg_fitness,
                        alpha=0.3, color=self.colors['best'], label='Best-Average Range')
        ax1.fill_between(generations, avg_fitness, worst_fitness,
                        alpha=0.3, color=self.colors['worst'], label='Average-Worst Range')

        # Add migration event annotations if available
        if data.migration_events:
            self._add_migration_annotations(ax1, data.migration_events, generations, best_fitness, worst_fitness)

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fitness improvement rate
        if len(best_fitness) > 1:
            improvement = np.diff(best_fitness)
            ax2.plot(generations[1:], improvement, color=self.colors['best'],
                    linewidth=2, label='Fitness Improvement')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Fitness Change')
            ax2.set_title('Fitness Improvement Rate', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path / 'fitness_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_fitness_distribution(self, data: DiagnosticsData, folder_path: Path):
        """Plot fitness distribution analysis."""
        if not data.generations:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        generations = [g.generation for g in data.generations]

        # Fitness standard deviation over time
        fitness_std = [g.fitness_std for g in data.generations]
        ax1.plot(generations, fitness_std, color=self.colors['diversity'], linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Standard Deviation')
        ax1.set_title('Population Fitness Diversity', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Convergence metric
        convergence = [g.convergence_metric for g in data.generations]
        ax2.plot(generations, convergence, color=self.colors['best'], linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Convergence Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Population diversity
        diversity = [g.population_diversity for g in data.generations]
        ax3.plot(generations, diversity, color=self.colors['mutations'], linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Diversity')
        ax3.set_title('Genetic Diversity Over Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Processing time per generation
        proc_times = [g.processing_time for g in data.generations]
        ax4.plot(generations, proc_times, color=self.colors['crossovers'], linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Processing Time (seconds)')
        ax4.set_title('Performance Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path / 'fitness_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_genetic_operations(self, data: DiagnosticsData, folder_path: Path):
        """Plot genetic operations effectiveness analysis.

        Creates comprehensive analysis including:
        - Beneficial operations over time
        - Cumulative success tracking
        - Rolling average success rates
        - Distribution pie chart with explanatory text clarifying both operations work together
        """
        if not data.generations:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        generations = [g.generation for g in data.generations]

        # Beneficial operations over time
        mutations = [g.beneficial_mutations for g in data.generations]
        crossovers = [g.beneficial_crossovers for g in data.generations]

        ax1.plot(generations, mutations, label='Beneficial Mutations',
                color=self.colors['mutations'], linewidth=2, marker='o', markersize=3)
        ax1.plot(generations, crossovers, label='Beneficial Crossovers',
                color=self.colors['crossovers'], linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Beneficial Operations')
        ax1.set_title('Beneficial Genetic Operations', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative beneficial operations
        cum_mutations = np.cumsum(mutations)
        cum_crossovers = np.cumsum(crossovers)

        ax2.plot(generations, cum_mutations, label='Cumulative Mutations',
                color=self.colors['mutations'], linewidth=2)
        ax2.plot(generations, cum_crossovers, label='Cumulative Crossovers',
                color=self.colors['crossovers'], linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Cumulative Beneficial Operations')
        ax2.set_title('Cumulative Success of Genetic Operations', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Success rates over time (rolling average)
        window = min(10, len(mutations) // 4)
        if window > 1:
            mut_rolling = np.convolve(mutations, np.ones(window)/window, mode='valid')
            cross_rolling = np.convolve(crossovers, np.ones(window)/window, mode='valid')
            rolling_gens = generations[window-1:]

            ax3.plot(rolling_gens, mut_rolling, label=f'Mutations ({window}-gen avg)',
                    color=self.colors['mutations'], linewidth=2)
            ax3.plot(rolling_gens, cross_rolling, label=f'Crossovers ({window}-gen avg)',
                    color=self.colors['crossovers'], linewidth=2)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Average Beneficial Operations')
            ax3.set_title('Rolling Average Success Rates', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Operations effectiveness distribution
        total_mut = sum(mutations)
        total_cross = sum(crossovers)
        if total_mut + total_cross > 0:
            labels = ['Beneficial Mutations', 'Beneficial Crossovers']
            sizes = [total_mut, total_cross]
            colors = [self.colors['mutations'], self.colors['crossovers']]

            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Distribution of Beneficial Operations', fontweight='bold')

            # Add explanatory text
            explanation = (
                "Shows the relative contribution of mutations vs crossovers\n"
                "to successful fitness improvements. Both operations\n"
                "work together - this shows their relative effectiveness."
            )
            ax4.text(0, -1.5, explanation, ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(folder_path / 'genetic_operations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_metrics(self, data: DiagnosticsData, folder_path: Path):
        """Plot accurate performance and efficiency metrics.

        Features corrected calculations including:
        - Processing time trends with regression analysis
        - Evolution efficiency (fitness improvement per second)
        - Memory usage patterns
        - Accurate success rate summary based on estimated total operations
        """
        if not data.generations:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        generations = [g.generation for g in data.generations]
        proc_times = [g.processing_time for g in data.generations]

        # Processing time trend
        ax1.plot(generations, proc_times, color=self.colors['best'], linewidth=2, marker='o', markersize=3)
        if len(proc_times) > 1:
            # Add trend line
            z = np.polyfit(generations, proc_times, 1)
            p = np.poly1d(z)
            ax1.plot(generations, p(generations), "--", color='red', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time per Generation', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Efficiency metric (fitness improvement per second)
        best_fitness = [g.best_fitness for g in data.generations]
        if len(best_fitness) > 1 and len(proc_times) > 1:
            efficiency = []
            for i in range(1, len(best_fitness)):
                improvement = best_fitness[i-1] - best_fitness[i]  # Lower is better
                time_taken = proc_times[i]
                if time_taken > 0:
                    efficiency.append(improvement / time_taken)
                else:
                    efficiency.append(0)

            if efficiency:
                ax2.plot(generations[1:], efficiency, color=self.colors['average'], linewidth=2, marker='s', markersize=3)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Fitness Improvement per Second')
                ax2.set_title('Evolution Efficiency', fontweight='bold')
                ax2.grid(True, alpha=0.3)

        # Memory usage estimation (based on population diversity)
        diversity = [g.population_diversity for g in data.generations]
        ax3.plot(generations, diversity, color=self.colors['diversity'], linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Diversity (proxy for memory usage)')
        ax3.set_title('Estimated Memory Usage Pattern', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Summary statistics
        if proc_times:
            stats_text = f"""Performance Summary:
    Total Time: {data.total_processing_time:.2f}s
    Avg Time/Gen: {np.mean(proc_times):.3f}s
    Min Time/Gen: {np.min(proc_times):.3f}s
    Max Time/Gen: {np.max(proc_times):.3f}s

    Operation Success Rates:
    Mutations: {data.beneficial_mutation_rate:.3f}
    Crossovers: {data.beneficial_crossover_rate:.3f}

    Final Convergence: {data.final_convergence:.6f}"""

            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Performance Summary', fontweight='bold')

        plt.tight_layout()
        plt.savefig(folder_path / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_population_analysis(self, data: DiagnosticsData, folder_path: Path):
        """Plot population-level analysis."""
        if not data.generations:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        generations = [g.generation for g in data.generations]

        # Fitness range over time
        best_fitness = [g.best_fitness for g in data.generations]
        worst_fitness = [g.worst_fitness for g in data.generations]
        avg_fitness = [g.average_fitness for g in data.generations]

        ax1.fill_between(generations, best_fitness, worst_fitness,
                        alpha=0.3, color=self.colors['average'], label='Fitness Range')
        ax1.plot(generations, avg_fitness, color=self.colors['average'], linewidth=2, label='Average')
        ax1.plot(generations, best_fitness, color=self.colors['best'], linewidth=2, label='Best')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Population Fitness Range', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Selection pressure using standardized definition
        from ..cli.helpers import calculate_selection_pressure
        selection_pressure = [calculate_selection_pressure(np.array([avg, best])) for avg, best in zip(avg_fitness, best_fitness)]
        ax2.plot(generations, selection_pressure, color=self.colors['mutations'], linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Selection Pressure')
        ax2.set_title('Selection Pressure Over Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add explanation text
        explanation = (
            "Selection Pressure = (Average - Best) / Average Fitness\n"
            "Higher values indicate more room for improvement.\n"
            "Decreasing trend shows population convergence."
        )
        ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='lightyellow', alpha=0.8))

        # Population diversity trends
        diversity = [g.population_diversity for g in data.generations]
        ax3.plot(generations, diversity, color=self.colors['diversity'], linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Diversity')
        ax3.set_title('Genetic Diversity Trends', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Stagnation detection (generations without improvement)
        stagnation = []
        last_improvement = 0
        for i, fitness in enumerate(best_fitness):
            if i == 0 or fitness < best_fitness[last_improvement]:
                last_improvement = i
                stagnation.append(0)
            else:
                stagnation.append(i - last_improvement)

        ax4.plot(generations, stagnation, color=self.colors['worst'], linewidth=2, marker='s', markersize=3)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Generations Since Last Improvement')
        ax4.set_title('Stagnation Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path / 'population_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_evolution_grid(self, data: DiagnosticsData, folder_path: Path):
        """Create enhanced grid showing best individuals evolution over time.

        Features include:
        - Color-coded heatmaps representing source image arrangements
        - Comprehensive color bar with "Source Image Index" labeling
        - Explanatory text describing visualization meaning
        - Generation progression from random to optimized layouts
        """
        if not data.generations or len(data.generations) < 2:
            return

        # Sample individuals across evolution
        sample_indices = np.linspace(0, len(data.generations)-1, min(16, len(data.generations)), dtype=int)

        # Create a grid layout
        rows = int(np.ceil(np.sqrt(len(sample_indices))))
        cols = int(np.ceil(len(sample_indices) / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, gen_idx in enumerate(sample_indices):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            generation = data.generations[gen_idx]
            individual = generation.best_individual

            # Create a simple heatmap representation
            im = ax.imshow(individual, cmap='viridis', aspect='auto')
            ax.set_title(f'Gen {generation.generation}\nFitness: {generation.best_fitness:.4f}',
                        fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(len(sample_indices), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        # Add suptitle first
        plt.suptitle('Evolution of Best Individuals', fontsize=16, fontweight='bold')

        # Apply tight layout first to get proper subplot positions
        plt.tight_layout(rect=[0, 0.1, 0.85, 0.95])  # Leave space for colorbar and text

        # Add colorbar with proper positioning
        cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Source Image Index', rotation=270, labelpad=15)

        # Add explanation text with proper positioning to avoid overlap
        explanation = (
            "Each heatmap shows the arrangement of source images (by index) "
            "in the grid for the best individual of that generation. "
            "Colors represent different source images used in each position."
        )
        fig.text(0.5, 0.05, explanation, ha='center', fontsize=10,
                style='italic', wrap=True)

        plt.savefig(folder_path / 'evolution_grid.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dashboard(self, data: DiagnosticsData, folder_path: Path):
        """Create comprehensive dashboard with properly aligned statistics.

        Features enhanced layout with:
        - Main fitness evolution plot with dual shading
        - Genetic operations summary bar chart
        - Population diversity trends
        - Processing time distribution histogram
        - Convergence analysis
        - Evolution summary with properly aligned columns using string formatting
        """
        if not data.generations:
            return

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main fitness plot (large)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        generations = [g.generation for g in data.generations]
        best_fitness = [g.best_fitness for g in data.generations]
        avg_fitness = [g.average_fitness for g in data.generations]

        ax_main.plot(generations, best_fitness, label='Best', color=self.colors['best'], linewidth=3)
        ax_main.plot(generations, avg_fitness, label='Average', color=self.colors['average'], linewidth=2)
        ax_main.fill_between(generations, best_fitness, avg_fitness, alpha=0.3, color=self.colors['best'])
        ax_main.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Genetic operations summary
        ax_ops = fig.add_subplot(gs[0, 2])
        total_mut = sum(g.beneficial_mutations for g in data.generations)
        total_cross = sum(g.beneficial_crossovers for g in data.generations)
        ax_ops.bar(['Mutations', 'Crossovers'], [total_mut, total_cross],
                  color=[self.colors['mutations'], self.colors['crossovers']])
        ax_ops.set_title('Beneficial Operations', fontweight='bold')
        ax_ops.set_ylabel('Count')

        # Diversity trend
        ax_div = fig.add_subplot(gs[1, 2])
        diversity = [g.population_diversity for g in data.generations]
        ax_div.plot(generations, diversity, color=self.colors['diversity'], linewidth=2)
        ax_div.set_title('Population Diversity', fontweight='bold')
        ax_div.set_xlabel('Generation')

        # Performance summary
        ax_perf = fig.add_subplot(gs[0, 3])
        proc_times = [g.processing_time for g in data.generations]
        ax_perf.hist(proc_times, bins=20, color=self.colors['crossovers'], alpha=0.7)
        ax_perf.set_title('Processing Time Distribution', fontweight='bold')
        ax_perf.set_xlabel('Time (seconds)')

        # Convergence analysis
        ax_conv = fig.add_subplot(gs[1, 3])
        convergence = [g.convergence_metric for g in data.generations]
        ax_conv.plot(generations, convergence, color=self.colors['worst'], linewidth=2)
        ax_conv.set_title('Convergence Rate', fontweight='bold')
        ax_conv.set_xlabel('Generation')

        # Statistics summary (bottom half)
        ax_stats = fig.add_subplot(gs[2:4, :])
        ax_stats.axis('off')

        # Create comprehensive statistics
        if data.generations:
            initial_fitness = data.generations[0].best_fitness
            final_fitness = data.generations[-1].best_fitness
            total_improvement = initial_fitness - final_fitness
            improvement_pct = (total_improvement / initial_fitness) * 100 if initial_fitness != 0 else 0

            # Create properly aligned columns using fixed-width formatting
            col1_width = 35
            col2_width = 35
            col3_width = 25

            def format_row(col1, col2, col3):
                return f"{col1:<{col1_width}} {col2:<{col2_width}} {col3:<{col3_width}}"

            stats_text = f"""
EVOLUTION SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════

{format_row("Configuration:", "Performance:", "Success Metrics:")}
{format_row(f"• Grid Size: {str(data.config_used['grid_size'])}", f"• Total Time: {data.total_processing_time:.2f}s", f"• Improvement: {improvement_pct:.2f}%")}
{format_row(f"• Population: {data.config_used['population_size']}", f"• Avg Time/Gen: {np.mean(proc_times):.3f}s", f"• Final Fitness: {final_fitness:.6f}")}
{format_row(f"• Mutation Rate: {data.config_used['mutation_rate']:.3f}", f"• Generations: {len(data.generations)}", f"• Convergence: {data.final_convergence:.6f}")}
{format_row(f"• Crossover Rate: {data.config_used['crossover_rate']:.3f}", f"• Speed: {len(data.generations)/data.total_processing_time:.2f} gen/s", f"• Mutation Success: {data.beneficial_mutation_rate:.3f}")}

{format_row("Fitness Evolution:", "Population Dynamics:", "Operation Effectiveness:")}
{format_row(f"• Initial: {initial_fitness:.6f}", f"• Avg Diversity: {np.mean(diversity):.3f}", f"• Beneficial Mutations: {total_mut}")}
{format_row(f"• Final: {final_fitness:.6f}", f"• Max Diversity: {np.max(diversity):.3f}", f"• Beneficial Crossovers: {total_cross}")}
{format_row(f"• Improvement: {total_improvement:.6f}", f"• Selection Pressure: {np.mean([calculate_selection_pressure(np.array([avg, best])) for avg, best in zip(avg_fitness, best_fitness)]):.6f}", f"• Success Ratio: {total_mut/max(total_cross, 1):.2f}")}
"""

            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=11,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))

        plt.suptitle('GENETIC ALGORITHM DIAGNOSTICS DASHBOARD', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(folder_path / 'dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_comprehensive_diversity(self, data: DiagnosticsData, folder_path: Path):
        """Plot comprehensive diversity metrics from the ComprehensiveDiversityManager."""
        if not data.generations:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        generations = [g.generation for g in data.generations]

        # 1. Hamming Distance Metrics
        hamming_avg = [g.hamming_distance_avg for g in data.generations]
        hamming_std = [g.hamming_distance_std for g in data.generations]

        axes[0].plot(generations, hamming_avg, label='Average Hamming Distance', color=self.colors['best'], linewidth=2)
        axes[0].plot(generations, hamming_std, label='Hamming Distance Std', color=self.colors['average'], linewidth=2)
        axes[0].set_title('Hamming Distance Diversity', fontweight='bold')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Hamming Distance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Position-wise Entropy
        position_entropy = [g.position_wise_entropy for g in data.generations]

        axes[1].plot(generations, position_entropy, color=self.colors['diversity'], linewidth=2)
        axes[1].set_title('Position-wise Entropy', fontweight='bold')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True, alpha=0.3)

        # 3. Unique Individuals Ratio
        unique_ratio = [g.unique_individuals_ratio for g in data.generations]

        axes[2].plot(generations, unique_ratio, color=self.colors['mutations'], linewidth=2)
        axes[2].set_title('Unique Individuals Ratio', fontweight='bold')
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Ratio (0-1)')
        axes[2].grid(True, alpha=0.3)

        # 4. Fitness Coefficient of Variation
        fitness_cv = [g.fitness_coefficient_variation for g in data.generations]

        axes[3].plot(generations, fitness_cv, color=self.colors['crossovers'], linewidth=2)
        axes[3].set_title('Fitness Coefficient of Variation', fontweight='bold')
        axes[3].set_xlabel('Generation')
        axes[3].set_ylabel('CV')
        axes[3].grid(True, alpha=0.3)

        # 5. Cluster Count
        cluster_count = [g.cluster_count for g in data.generations]

        axes[4].plot(generations, cluster_count, color=self.colors['worst'], linewidth=2, marker='o', markersize=3)
        axes[4].set_title('Genetic Cluster Count', fontweight='bold')
        axes[4].set_xlabel('Generation')
        axes[4].set_ylabel('Number of Clusters')
        axes[4].grid(True, alpha=0.3)

        # 6. Normalized Diversity Score
        normalized_div = [g.normalized_diversity for g in data.generations]

        axes[5].plot(generations, normalized_div, color=self.colors['best'], linewidth=3)
        axes[5].set_title('Normalized Diversity Score', fontweight='bold')
        axes[5].set_xlabel('Generation')
        axes[5].set_ylabel('Diversity Score (0-1)')
        axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path / 'comprehensive_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_spatial_diversity(self, data: DiagnosticsData, folder_path: Path):
        """Plot spatial diversity metrics from the SpatialDiversityManager."""
        if not data.generations:
            return

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        generations = [g.generation for g in data.generations]

        # 1. Local Pattern Entropy
        local_pattern = [g.local_pattern_entropy for g in data.generations]
        axes[0].plot(generations, local_pattern, color=self.colors['best'], linewidth=2)
        axes[0].set_title('Local Pattern Entropy', fontweight='bold')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Entropy')
        axes[0].grid(True, alpha=0.3)

        # 2. Spatial Clustering
        spatial_clustering = [g.spatial_clustering for g in data.generations]
        axes[1].plot(generations, spatial_clustering, color=self.colors['average'], linewidth=2)
        axes[1].set_title('Spatial Clustering Diversity', fontweight='bold')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Clustering Variance')
        axes[1].grid(True, alpha=0.3)

        # 3. Edge Pattern Diversity
        edge_pattern = [g.edge_pattern_diversity for g in data.generations]
        axes[2].plot(generations, edge_pattern, color=self.colors['diversity'], linewidth=2)
        axes[2].set_title('Edge Pattern Diversity', fontweight='bold')
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Pattern Entropy')
        axes[2].grid(True, alpha=0.3)

        # 4. Quadrant Diversity
        quadrant_div = [g.quadrant_diversity for g in data.generations]
        axes[3].plot(generations, quadrant_div, color=self.colors['mutations'], linewidth=2)
        axes[3].set_title('Quadrant Diversity', fontweight='bold')
        axes[3].set_xlabel('Generation')
        axes[3].set_ylabel('Inter-quadrant Distance')
        axes[3].grid(True, alpha=0.3)

        # 5. Neighbor Similarity
        neighbor_sim = [g.neighbor_similarity for g in data.generations]
        axes[4].plot(generations, neighbor_sim, color=self.colors['crossovers'], linewidth=2)
        axes[4].set_title('Neighbor Similarity (Inverted)', fontweight='bold')
        axes[4].set_xlabel('Generation')
        axes[4].set_ylabel('1 - Similarity')
        axes[4].grid(True, alpha=0.3)

        # 6. Tile Distribution Variance
        tile_var = [g.tile_distribution_variance for g in data.generations]
        axes[5].plot(generations, tile_var, color=self.colors['worst'], linewidth=2)
        axes[5].set_title('Tile Distribution Variance', fontweight='bold')
        axes[5].set_xlabel('Generation')
        axes[5].set_ylabel('Usage Variance')
        axes[5].grid(True, alpha=0.3)

        # 7. Contiguous Regions
        contiguous = [g.contiguous_regions for g in data.generations]
        axes[6].plot(generations, contiguous, color=self.colors['diversity'], linewidth=2)
        axes[6].set_title('Contiguous Regions Diversity', fontweight='bold')
        axes[6].set_xlabel('Generation')
        axes[6].set_ylabel('Region Size Variance')
        axes[6].grid(True, alpha=0.3)

        # 8. Spatial Autocorrelation
        spatial_autocorr = [g.spatial_autocorrelation for g in data.generations]
        axes[7].plot(generations, spatial_autocorr, color=self.colors['best'], linewidth=2)
        axes[7].set_title('Spatial Autocorrelation Diversity', fontweight='bold')
        axes[7].set_xlabel('Generation')
        axes[7].set_ylabel('Autocorrelation Variance')
        axes[7].grid(True, alpha=0.3)

        # 9. Combined Spatial Diversity Score
        spatial_score = [g.spatial_diversity_score for g in data.generations]
        axes[8].plot(generations, spatial_score, color=self.colors['mutations'], linewidth=3)
        axes[8].set_title('Combined Spatial Diversity Score', fontweight='bold')
        axes[8].set_xlabel('Generation')
        axes[8].set_ylabel('Diversity Score (0-1)')
        axes[8].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path / 'spatial_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_advanced_metrics(self, data: DiagnosticsData, folder_path: Path):
        """Plot advanced evolution metrics and parameter adaptation."""
        if not data.generations:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        generations = [g.generation for g in data.generations]

        # 1. Adaptive Parameters
        mutation_rates = [g.current_mutation_rate for g in data.generations]
        crossover_rates = [g.current_crossover_rate for g in data.generations]

        axes[0, 0].plot(generations, mutation_rates, label='Mutation Rate',
                       color=self.colors['mutations'], linewidth=2)
        axes[0, 0].plot(generations, crossover_rates, label='Crossover Rate',
                       color=self.colors['crossovers'], linewidth=2)
        axes[0, 0].set_title('Adaptive Parameter Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Diversity Comparison
        normalized_div = [g.normalized_diversity for g in data.generations]
        spatial_score = [g.spatial_diversity_score for g in data.generations]
        old_diversity = [g.population_diversity for g in data.generations]

        axes[0, 1].plot(generations, normalized_div, label='Normalized Diversity',
                       color=self.colors['best'], linewidth=2)
        axes[0, 1].plot(generations, spatial_score, label='Spatial Diversity',
                       color=self.colors['diversity'], linewidth=2)
        # Normalize old diversity to 0-1 range for comparison
        if old_diversity:
            max_old = max(old_diversity)
            norm_old = [d / max_old for d in old_diversity] if max_old > 0 else old_diversity
            axes[0, 1].plot(generations, norm_old, label='Basic Diversity (norm)',
                           color=self.colors['average'], linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Diversity Metrics Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Diversity Score (0-1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Population Entropy vs Fitness Variance
        pop_entropy = [g.population_entropy for g in data.generations]
        fitness_var = [g.fitness_variance for g in data.generations]

        # Normalize fitness variance for better visualization
        if fitness_var and max(fitness_var) > 0:
            norm_fitness_var = [fv / max(fitness_var) for fv in fitness_var]
        else:
            norm_fitness_var = fitness_var

        axes[1, 0].plot(generations, pop_entropy, label='Population Entropy',
                       color=self.colors['diversity'], linewidth=2)
        axes[1, 0].plot(generations, norm_fitness_var, label='Fitness Variance (norm)',
                       color=self.colors['worst'], linewidth=2)
        axes[1, 0].set_title('Population Entropy vs Fitness Variance', fontweight='bold')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Normalized Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Selection Pressure Evolution
        selection_pressure = [g.selection_pressure for g in data.generations]

        axes[1, 1].plot(generations, selection_pressure, color=self.colors['crossovers'], linewidth=2)
        axes[1, 1].set_title('Selection Pressure Evolution', fontweight='bold')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Selection Pressure')
        axes[1, 1].grid(True, alpha=0.3)

        # Add explanatory text
        explanation = (
            "Selection Pressure = (Average - Best) / Average Fitness\\n"
            "Higher values indicate more room for improvement"
        )
        axes[1, 1].text(0.02, 0.98, explanation, transform=axes[1, 1].transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(folder_path / 'advanced_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _add_migration_annotations(self, ax, migration_events, generations, best_fitness, worst_fitness):
        """Add migration event annotations to fitness evolution plots.

        Args:
            ax: The matplotlib axes to annotate
            migration_events: List of migration event dictionaries
            generations: List of generation numbers
            best_fitness: List of best fitness values per generation
            worst_fitness: List of worst fitness values per generation
        """
        if not migration_events:
            return

        # Get the fitness range for positioning annotations
        min_fitness = min(best_fitness)
        max_fitness = max(worst_fitness)
        fitness_range = max_fitness - min_fitness

        # Group migration events by generation for cleaner visualization
        migration_by_gen = {}
        for event in migration_events:
            gen = event['generation']
            if gen in migration_by_gen:
                migration_by_gen[gen].append(event)
            else:
                migration_by_gen[gen] = [event]

        # Annotate each generation with migrations
        for gen, events in migration_by_gen.items():
            if gen in generations:
                # Draw vertical line at migration generation
                ax.axvline(x=gen, color='purple', linestyle=':', alpha=0.7, linewidth=1.5)

                # Add migration marker with count if multiple events
                num_events = len(events)
                marker_size = min(60, 20 + num_events * 10)  # Scale marker with event count

                # Position marker in middle of fitness range
                y_pos = min_fitness + fitness_range * 0.1

                ax.scatter([gen], [y_pos], color='purple', s=marker_size,
                          marker='*', alpha=0.8, zorder=5, edgecolors='darkmagenta')

                # Add text annotation for migration details
                if num_events == 1:
                    event = events[0]
                    annotation = f"Migration\\nIsland {event['source_island']}→{event['target_island']}"
                else:
                    annotation = f"{num_events} Migrations\\nGen {gen}"

                # Position text above marker
                ax.annotate(annotation, xy=(gen, y_pos),
                           xytext=(5, 20), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                         color='purple', alpha=0.6))

        # Add legend entry for migration events if any exist
        if migration_by_gen:
            ax.scatter([], [], color='purple', s=40, marker='*', alpha=0.8,
                      label='Migration Events', edgecolors='darkmagenta')