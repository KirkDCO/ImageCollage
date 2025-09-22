"""
Real-time diversity monitoring dashboard for genetic algorithm evolution.

Provides live monitoring of population diversity with alerts and intervention
recommendations as described in DIVERSITY.md.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class DiversityAlert:
    """Represents a diversity-related alert."""
    level: str  # 'info', 'warning', 'critical'
    message: str
    metric: str
    value: float
    threshold: float
    generation: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class DashboardConfig:
    """Configuration for diversity dashboard."""
    update_interval: int = 10
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical_diversity': 0.1,
        'low_diversity': 0.2,
        'stagnation_generations': 30,
        'high_selection_pressure': 0.9,
        'low_fitness_variance': 0.001
    })
    enable_real_time_display: bool = True
    enable_file_logging: bool = True
    max_history_size: int = 1000


class DiversityDashboard:
    """
    Real-time diversity monitoring and alert system.

    Tracks population diversity metrics, detects problematic patterns,
    and provides intervention recommendations.
    """

    def __init__(self, config: DashboardConfig, output_dir: Optional[str] = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None

        # State tracking
        self.diversity_history = deque(maxlen=config.max_history_size)
        self.alert_history = deque(maxlen=config.max_history_size)
        self.intervention_history = []

        # Current state
        self.current_generation = 0
        self.current_metrics = {}
        self.active_alerts = []

        # Performance tracking
        self.start_time = time.time()
        self.last_update_time = time.time()

        # File logging setup
        if self.config.enable_file_logging and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.output_dir / "diversity_dashboard.log"
            self.metrics_file = self.output_dir / "diversity_metrics.json"

        logging.info("Diversity dashboard initialized with real-time monitoring")

    def update_dashboard(self, generation: int, diversity_metrics: Dict[str, float],
                        population_state: Dict[str, Any]) -> List[DiversityAlert]:
        """
        Update dashboard with new generation data.

        Args:
            generation: Current generation number
            diversity_metrics: Diversity metrics from comprehensive analysis
            population_state: Additional population state information

        Returns:
            List of new alerts generated
        """

        self.current_generation = generation
        self.current_metrics = diversity_metrics.copy()
        current_time = time.time()

        # Only do full recording and processing every update_interval generations
        if generation % self.config.update_interval == 0:
            # Record metrics with timestamp
            metrics_record = {
                'generation': generation,
                'timestamp': current_time,
                'metrics': diversity_metrics.copy(),
                'population_state': population_state.copy()
            }
            self.diversity_history.append(metrics_record)

            # Check for alerts
            new_alerts = self._check_diversity_alerts(diversity_metrics, population_state, generation)
            self.active_alerts = new_alerts

            # Add to alert history
            for alert in new_alerts:
                self.alert_history.append(alert)

            # Display dashboard
            if self.config.enable_real_time_display:
                self._display_dashboard(generation, diversity_metrics, new_alerts, population_state)

            # Log to file
            if self.config.enable_file_logging:
                self._log_to_file(metrics_record, new_alerts)

            self.last_update_time = current_time
            return new_alerts
        else:
            # For non-update generations, just return empty list
            return []

    def _check_diversity_alerts(self, metrics: Dict[str, float],
                               population_state: Dict[str, Any],
                               generation: int) -> List[DiversityAlert]:
        """Check for diversity-related issues requiring intervention."""
        alerts = []
        thresholds = self.config.alert_thresholds

        diversity_score = metrics.get('normalized_diversity', 0.5)
        hamming_avg = metrics.get('hamming_distance_avg', 0.0)
        entropy = metrics.get('position_wise_entropy', 0.0)
        fitness_variance = metrics.get('fitness_variance', 0.0)

        # Critical diversity alerts
        if diversity_score < thresholds['critical_diversity']:
            alerts.append(DiversityAlert(
                level='critical',
                message=f'CRITICAL: Extremely low population diversity ({diversity_score:.3f})',
                metric='normalized_diversity',
                value=diversity_score,
                threshold=thresholds['critical_diversity'],
                generation=generation
            ))
        elif diversity_score < thresholds['low_diversity']:
            alerts.append(DiversityAlert(
                level='warning',
                message=f'WARNING: Low population diversity ({diversity_score:.3f})',
                metric='normalized_diversity',
                value=diversity_score,
                threshold=thresholds['low_diversity'],
                generation=generation
            ))

        # Fitness variance alerts
        if fitness_variance < thresholds['low_fitness_variance']:
            alerts.append(DiversityAlert(
                level='warning',
                message=f'WARNING: Very low fitness variance ({fitness_variance:.4f})',
                metric='fitness_variance',
                value=fitness_variance,
                threshold=thresholds['low_fitness_variance'],
                generation=generation
            ))

        # Stagnation detection
        stagnation_check = self._check_stagnation_pattern(generation)
        if stagnation_check['is_stagnant']:
            alerts.append(DiversityAlert(
                level='warning',
                message=f'WARNING: Diversity stagnation detected ({stagnation_check["generations"]} generations)',
                metric='diversity_stagnation',
                value=stagnation_check['generations'],
                threshold=thresholds['stagnation_generations'],
                generation=generation
            ))

        # Selection pressure alerts
        selection_pressure = population_state.get('selection_pressure', 0.5)
        if selection_pressure > thresholds['high_selection_pressure']:
            alerts.append(DiversityAlert(
                level='info',
                message=f'INFO: High selection pressure detected ({selection_pressure:.3f})',
                metric='selection_pressure',
                value=selection_pressure,
                threshold=thresholds['high_selection_pressure'],
                generation=generation
            ))

        # Spatial diversity alerts (if available)
        spatial_diversity = metrics.get('spatial_diversity_score', 0.0)
        if spatial_diversity > 0 and spatial_diversity < 0.2:
            alerts.append(DiversityAlert(
                level='warning',
                message=f'WARNING: Low spatial diversity ({spatial_diversity:.3f})',
                metric='spatial_diversity_score',
                value=spatial_diversity,
                threshold=0.2,
                generation=generation
            ))

        return alerts

    def _check_stagnation_pattern(self, generation: int) -> Dict[str, Any]:
        """Check for diversity stagnation patterns."""
        if len(self.diversity_history) < self.config.alert_thresholds['stagnation_generations']:
            return {'is_stagnant': False, 'generations': 0}

        recent_history = list(self.diversity_history)[-int(self.config.alert_thresholds['stagnation_generations']):]
        diversity_values = [h['metrics'].get('normalized_diversity', 0.5) for h in recent_history]

        # Check if diversity has remained relatively constant
        diversity_range = max(diversity_values) - min(diversity_values)
        is_stagnant = diversity_range < 0.05  # Very small change in diversity

        return {
            'is_stagnant': is_stagnant,
            'generations': len(recent_history) if is_stagnant else 0,
            'diversity_range': diversity_range,
            'recent_values': diversity_values
        }

    def _display_dashboard(self, generation: int, metrics: Dict[str, float],
                          alerts: List[DiversityAlert], population_state: Dict[str, Any]) -> None:
        """Display diversity dashboard information to console."""
        runtime = time.time() - self.start_time


        print(f"\n{'='*60}")
        print(f"🧬 DIVERSITY DASHBOARD - Generation {generation}")
        print(f"{'='*60}")
        print(f"⏱️  Runtime: {runtime:.1f}s | Update interval: {self.config.update_interval} gen")
        import sys
        sys.stdout.flush()  # Force immediate output
        print()

        # Core diversity metrics
        print("📊 DIVERSITY METRICS:")
        diversity_score = metrics.get('normalized_diversity', 0)
        hamming_avg = metrics.get('hamming_distance_avg', 0)
        entropy = metrics.get('position_wise_entropy', 0)
        unique_ratio = metrics.get('unique_individuals_ratio', 0)

        print(f"   Overall Diversity Score: {diversity_score:.3f} {self._get_diversity_emoji(diversity_score)}")
        print(f"   Hamming Distance (avg):  {hamming_avg:.2f}")
        print(f"   Position Entropy:        {entropy:.3f}")
        print(f"   Unique Individuals:      {unique_ratio:.1%}")

        # Fitness diversity
        fitness_var = metrics.get('fitness_variance', 0)
        fitness_range = metrics.get('fitness_range', 0)

        # Use scientific notation for very small fitness variance
        if fitness_var > 0 and fitness_var < 0.0001:
            print(f"   Fitness Variance:        {fitness_var:.2e}")
        else:
            print(f"   Fitness Variance:        {fitness_var:.4f}")
        print(f"   Fitness Range:           {fitness_range:.4f}")

        # Spatial diversity (if available)
        spatial_diversity = metrics.get('spatial_diversity_score', 0)
        if spatial_diversity > 0:
            print(f"   Spatial Diversity:       {spatial_diversity:.3f}")

        # Population state
        print()
        print("🏗️  POPULATION STATE:")
        pop_size = population_state.get('population_size', 0)
        selection_pressure = population_state.get('selection_pressure', 0)
        print(f"   Population Size:         {pop_size}")
        print(f"   Selection Pressure:      {selection_pressure:.3f}")

        # Islands info (if available)
        if 'num_islands' in population_state:
            num_islands = population_state['num_islands']
            migration_rate = population_state.get('migration_rate', 0)
            print(f"   Islands:                 {num_islands}")
            print(f"   Migration Rate:          {migration_rate:.2%}")

        # Alerts section
        if alerts:
            print()
            print("🚨 ALERTS:")
            for alert in alerts:
                emoji = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}[alert.level]
                print(f"   {emoji} {alert.message}")
        else:
            print()
            print("✅ No diversity alerts")

        # Recommendations
        recommendations = self._generate_recommendations(metrics, alerts, population_state)
        if recommendations:
            print()
            print("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")

        print(f"{'='*60}\n")
        import sys
        sys.stdout.flush()  # Force immediate output

    def _get_diversity_emoji(self, diversity_score: float) -> str:
        """Get emoji representation of diversity level."""
        if diversity_score >= 0.7:
            return "🟢"  # High diversity
        elif diversity_score >= 0.4:
            return "🟡"  # Medium diversity
        elif diversity_score >= 0.2:
            return "🟠"  # Low diversity
        else:
            return "🔴"  # Critical diversity

    def _generate_recommendations(self, metrics: Dict[str, float],
                                alerts: List[DiversityAlert],
                                population_state: Dict[str, Any]) -> List[str]:
        """Generate intervention recommendations based on current state."""
        recommendations = []

        diversity_score = metrics.get('normalized_diversity', 0.5)
        selection_pressure = population_state.get('selection_pressure', 0.5)

        # Diversity-based recommendations
        if diversity_score < 0.2:
            recommendations.append("Increase mutation rate to 0.2-0.3 for diversity boost")
            recommendations.append("Consider population restart with elite preservation")
            recommendations.append("Enable fitness sharing to reduce crowding")

        elif diversity_score < 0.4:
            recommendations.append("Increase mutation rate slightly (0.1-0.15)")
            recommendations.append("Consider immigration of random individuals")

        # Selection pressure recommendations
        if selection_pressure > 0.8:
            recommendations.append("Reduce selection pressure (increase tournament size)")
            recommendations.append("Use rank-based selection instead of fitness-proportional")

        # Spatial diversity recommendations (if available)
        spatial_diversity = metrics.get('spatial_diversity_score', 0)
        if spatial_diversity > 0 and spatial_diversity < 0.3:
            recommendations.append("Enable spatial-aware mutation for better tile diversity")
            recommendations.append("Consider spatial fitness sharing")

        # Alert-specific recommendations
        critical_alerts = [a for a in alerts if a.level == 'critical']
        if critical_alerts:
            recommendations.append("URGENT: Consider immediate population restart")

        return recommendations

    def _log_to_file(self, metrics_record: Dict[str, Any],
                    alerts: List[DiversityAlert]) -> None:
        """Log dashboard data to files."""
        if not self.output_dir:
            return

        try:
            # Log metrics to JSON file
            with open(self.metrics_file, 'a') as f:
                json.dump(metrics_record, f, default=str)
                f.write('\n')

            # Log alerts to text file
            if alerts:
                with open(self.log_file, 'a') as f:
                    for alert in alerts:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {alert.level.upper()}: {alert.message}\n")

        except Exception as e:
            logging.warning(f"Failed to log dashboard data: {e}")

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        if not self.diversity_history:
            return {}

        recent_metrics = [h['metrics'] for h in list(self.diversity_history)[-10:]]

        # Calculate trends
        diversity_trend = self._calculate_trend([m.get('normalized_diversity', 0) for m in recent_metrics])
        entropy_trend = self._calculate_trend([m.get('position_wise_entropy', 0) for m in recent_metrics])

        # Alert statistics
        alert_counts = {'critical': 0, 'warning': 0, 'info': 0}
        for alert in self.alert_history:
            alert_counts[alert.level] += 1

        return {
            'total_generations_monitored': len(self.diversity_history),
            'current_diversity': self.current_metrics.get('normalized_diversity', 0),
            'diversity_trend': diversity_trend,
            'entropy_trend': entropy_trend,
            'alert_summary': alert_counts,
            'active_alerts': len(self.active_alerts),
            'monitoring_duration': time.time() - self.start_time,
            'last_update': self.last_update_time
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values."""
        if len(values) < 2:
            return "stable"

        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier_avg = np.mean(values[:-3]) if len(values) >= 6 else values[0]

        diff = recent_avg - earlier_avg
        if abs(diff) < 0.01:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"

    def export_dashboard_data(self, output_file: str) -> None:
        """Export complete dashboard data to file."""
        dashboard_data = {
            'config': {
                'update_interval': self.config.update_interval,
                'alert_thresholds': self.config.alert_thresholds
            },
            'summary': self.get_dashboard_summary(),
            'diversity_history': [
                {
                    'generation': h['generation'],
                    'timestamp': h['timestamp'],
                    'metrics': h['metrics']
                }
                for h in self.diversity_history
            ],
            'alert_history': [
                {
                    'level': a.level,
                    'message': a.message,
                    'metric': a.metric,
                    'value': a.value,
                    'generation': a.generation,
                    'timestamp': a.timestamp
                }
                for a in self.alert_history
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        logging.info(f"Dashboard data exported to {output_file}")