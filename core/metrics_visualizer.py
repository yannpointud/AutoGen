"""
Module de visualisation des métriques du système multi-agents.
Tableaux de bord et rapports visuels.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import statistics

from utils.logger import setup_logger, parse_json_logs


class MetricsCollector:
    """Collecteur de métriques du système."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.logger = setup_logger(f"MetricsCollector.{project_name}")
        self.project_path = Path("projects") / project_name
        
        # Métriques en temps réel
        self.metrics = {
            'agents': defaultdict(dict),
            'tasks': defaultdict(dict),
            'communications': defaultdict(int),
            'performance': defaultdict(list),
            'quality': defaultdict(float),
            'progress': defaultdict(float)
        }
        
        # Historique
        self.history = []
        self.start_time = datetime.now()
    
    def collect_from_logs(self) -> Dict[str, Any]:
        """Collecte les métriques depuis les logs."""
        logs_path = self.project_path / "logs"
        if not logs_path.exists():
            return {}
        
        all_metrics = {
            'agent_activities': defaultdict(list),
            'task_completions': [],
            'errors': [],
            'interactions': [],
            'llm_calls': []
        }
        
        # Parser tous les fichiers de logs
        for log_file in logs_path.glob("*.jsonl"):
            logs = parse_json_logs(log_file)
            
            for entry in logs:
                # Activités des agents
                if 'agent_name' in entry:
                    all_metrics['agent_activities'][entry['agent_name']].append({
                        'timestamp': entry.get('timestamp'),
                        'action': entry.get('interaction_type', 'unknown'),
                        'status': entry.get('status', 'unknown')
                    })
                
                # Tâches complétées
                if entry.get('interaction_type') == 'act' and 'status' in entry:
                    all_metrics['task_completions'].append({
                        'agent': entry.get('agent_name'),
                        'status': entry.get('status'),
                        'timestamp': entry.get('timestamp')
                    })
                
                # Erreurs
                if entry.get('level') == 'ERROR':
                    all_metrics['errors'].append({
                        'agent': entry.get('agent_name', 'system'),
                        'message': entry.get('message'),
                        'timestamp': entry.get('timestamp')
                    })
                
                # Interactions LLM
                if 'llm_model' in entry:
                    all_metrics['llm_calls'].append({
                        'model': entry.get('llm_model'),
                        'tokens': entry.get('tokens_used', 0),
                        'duration': entry.get('duration_seconds', 0),
                        'timestamp': entry.get('timestamp')
                    })
        
        return all_metrics
    
    def collect_from_rag(self, rag_engine: Optional[Any] = None) -> Dict[str, Any]:
        """Collecte les métriques du RAG."""
        if not rag_engine:
            return {}
        
        rag_stats = rag_engine.get_memory_usage()
        rag_summary = rag_engine.create_summary()
        
        return {
            'total_chunks': rag_stats.get('total_vectors', 0),
            'working_memory_chunks': rag_stats.get('working_memory_vectors', 0),
            'memory_usage_mb': rag_summary.get('estimated_memory_mb', 0),
            'compression_needed': rag_stats.get('compression_threshold_reached', False),
            'content_types': rag_summary.get('types', {}),
            'agent_contributions': rag_summary.get('agents', {})
        }
    
    def update_agent_metrics(self, agent_name: str, metrics: Dict[str, Any]) -> None:
        """Met à jour les métriques d'un agent."""
        self.metrics['agents'][agent_name].update(metrics)
        self.metrics['agents'][agent_name]['last_update'] = datetime.now().isoformat()
    
    def update_task_metrics(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """Met à jour les métriques d'une tâche."""
        self.metrics['tasks'][task_id].update(metrics)
        
        # Calculer le taux de complétion
        if 'status' in metrics:
            if metrics['status'] == 'completed':
                self.metrics['progress']['completed_tasks'] = \
                    self.metrics['progress'].get('completed_tasks', 0) + 1
            elif metrics['status'] == 'failed':
                self.metrics['progress']['failed_tasks'] = \
                    self.metrics['progress'].get('failed_tasks', 0) + 1
    
    def record_communication(self, sender: str, recipient: str, message_type: str) -> None:
        """Enregistre une communication."""
        key = f"{sender}->{recipient}:{message_type}"
        self.metrics['communications'][key] += 1
    
    def record_performance(self, operation: str, duration: float) -> None:
        """Enregistre une métrique de performance."""
        self.metrics['performance'][operation].append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques."""
        # Collecter depuis les logs
        log_metrics = self.collect_from_logs()
        
        # Calculer les statistiques
        total_tasks = len(self.metrics['tasks'])
        completed_tasks = self.metrics['progress'].get('completed_tasks', 0)
        failed_tasks = self.metrics['progress'].get('failed_tasks', 0)
        
        # Performance moyenne
        avg_performance = {}
        for op, durations in self.metrics['performance'].items():
            if durations:
                avg_performance[op] = {
                    'avg': statistics.mean(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        
        # Durée totale
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'project': self.project_name,
            'elapsed_time_seconds': elapsed_time,
            'agents': {
                'total': len(self.metrics['agents']),
                'active': len([a for a in self.metrics['agents'].values() 
                              if a.get('status') == 'active'])
            },
            'tasks': {
                'total': total_tasks,
                'completed': completed_tasks,
                'failed': failed_tasks,
                'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            },
            'communications': dict(self.metrics['communications']),
            'performance': avg_performance,
            'errors': log_metrics.get('errors', []),
            'llm_usage': self._calculate_llm_usage(log_metrics.get('llm_calls', []))
        }
    
    def _calculate_llm_usage(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule les statistiques d'utilisation LLM."""
        if not llm_calls:
            return {'total_calls': 0, 'total_tokens': 0, 'total_cost_estimate': 0}
        
        total_calls = len(llm_calls)
        total_tokens = sum(call.get('tokens', 0) for call in llm_calls)
        
        # Estimation du coût (simplifiée)
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        # Grouper par modèle
        by_model = defaultdict(int)
        for call in llm_calls:
            by_model[call.get('model', 'unknown')] += 1
        
        return {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost_estimate': round(total_cost, 4),
            'calls_by_model': dict(by_model)
        }


class MetricsVisualizer:
    """Visualiseur de métriques avec génération HTML."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.logger = setup_logger(f"MetricsVisualizer.{project_name}")
        self.collector = MetricsCollector(project_name)
        self.output_path = Path("projects") / project_name / "metrics"
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(self, rag_engine: Optional[Any] = None) -> str:
        """Génère un tableau de bord HTML complet."""
        summary = self.collector.get_summary()
        rag_metrics = self.collector.collect_from_rag(rag_engine)
        html = self._generate_html_dashboard(summary, rag_metrics)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_path = self.output_path / f"dashboard_{timestamp}.html"
        dashboard_path.write_text(html, encoding='utf-8')
        self.logger.info(f"Dashboard generated: {dashboard_path}")
        return str(dashboard_path)

    def _generate_html_dashboard(self, summary: Dict[str, Any], rag_metrics: Dict[str, Any]) -> str:
        elapsed = summary.get('elapsed_time_seconds', 0)
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_str = f"{hours}h {minutes}m {seconds}s"

        agents_total = summary['agents'].get('total', 0)
        agents_active = summary['agents'].get('active', 0)
        safe_total = max(agents_total, 1)

        t_completed = summary['tasks'].get('completed', 0)
        t_failed = summary['tasks'].get('failed', 0)
        t_total = summary['tasks'].get('total', 0)
        t_pending = max(0, t_total - t_completed - t_failed)
        safe_tasks = max(t_total, 1)
        tasks_data = [t_completed, t_failed, t_pending]

        charts = self._prepare_charts_data(summary, rag_metrics)

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dashboard - {self.project_name}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    html, body {{ height: 100%; margin:0; padding:0; overflow-y:auto; scroll-behavior: smooth; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#f5f7fa; color:#2c3e50; }}
    .header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:1rem; }}
    .container {{ max-width:1200px; margin:1rem auto; padding:1rem; overflow: visible; }}
    .metrics-grid {{ display:grid; grid-gap:1rem; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); }}
    .metric-card {{ background:#fff; border-radius:8px; padding:1rem; box-shadow:0 2px 5px rgba(0,0,0,0.1); }}
    .metric-label {{ font-size:0.85rem; color:#7f8c8d; text-transform:uppercase; }}
    .metric-value {{ font-size:1.8rem; margin:0.5rem 0; font-weight:bold; }}
    .chart-container {{ background:#fff; border-radius:8px; padding:1rem; margin-bottom:1rem; box-shadow:0 2px 5px rgba(0,0,0,0.1); max-height:400px; overflow:auto; }}
    canvas {{ display:block; width:100% !important; height:300px !important; }}
    table {{ width:100%; border-collapse:collapse; margin-top:0.5rem; }}
    th, td {{ padding:0.5rem; border-bottom:1px solid #ddd; text-align:left; }}
    th {{ background:#f0f0f0; }}
  </style>
</head>
<body onload="window.scrollTo(0,0)">
  <div class="header">
    <h1>🚀 {self.project_name} - Dashboard</h1>
    <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Durée: {elapsed_str}</p>
  </div>
  <div class="container">
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Agents actifs</div>
        <div class="metric-value">{agents_active}/{agents_total}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Tâches complétées</div>
        <div class="metric-value">{t_completed}/{t_total}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Utilisation LLM</div>
        <div class="metric-value">{summary['llm_usage']['total_calls']}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Chunks RAG</div>
        <div class="metric-value">{rag_metrics.get('total_chunks',0)}</div>
      </div>
    </div>

    <div class="chart-container">
      <h2>Répartition des tâches</h2>
      <canvas id="tasksChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Activité des agents</h2>
      <canvas id="agentsChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Communications inter-agents</h2>
      <canvas id="commChart"></canvas>
    </div>

    {self._generate_errors_section(summary.get('errors', []))}
    {self._generate_performance_section(summary.get('performance', {}))}
    {self._generate_llm_section(summary.get('llm_usage', {}))}
  </div>
  <script>
    new Chart(document.getElementById('tasksChart'), {{
      type: 'doughnut',
      data: {{
        labels: ['Complétées', 'Échouées', 'En cours'],
        datasets: [{{ data: {tasks_data}, backgroundColor: ['#2ecc71', '#e74c3c', '#f39c12'] }}]
      }}
    }});

    new Chart(document.getElementById('agentsChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(charts['agents']['labels'])},
        datasets: [{{ label: 'Contributions', data: {json.dumps(charts['agents']['values'])} }}]
      }}
    }});

    new Chart(document.getElementById('commChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(charts['communications']['labels'])},
        datasets: [{{ label: 'Messages', data: {json.dumps(charts['communications']['values'])} }}]
      }}
    }});
  </script>
</body>
</html>"""
        return html

    def _prepare_charts_data(self, summary: Dict[str, Any], rag_metrics: Dict[str, Any]) -> Dict[str, Any]:
        agents = rag_metrics.get('agent_contributions', {}) or {}
        comms = summary.get('communications', {}) or {}
        return {
            'agents': {'labels': list(agents.keys()), 'values': list(agents.values())},
            'communications': {'labels': [k.split(':')[0] for k in comms], 'values': list(comms.values())}
        }




    def _generate_errors_section(self, errors: List[Dict[str, Any]]) -> str:
        """Génère la section des erreurs."""
        if not errors:
            return """
        <div class="chart-container">
            <h2 class="chart-title">✅ Aucune Erreur</h2>
            <p style="color: #27ae60;">Aucune erreur détectée durant l'exécution.</p>
        </div>
        """
        
        # Limiter aux 10 dernières erreurs
        recent_errors = errors[-10:]
        
        rows = ""
        for error in recent_errors:
            rows += f"""
            <tr>
                <td>{error.get('timestamp', 'N/A')}</td>
                <td>{error.get('agent', 'system')}</td>
                <td>{error.get('message', 'N/A')[:100]}...</td>
            </tr>
            """
        
        return f"""
        <div class="chart-container">
            <h2 class="chart-title danger">⚠️ Erreurs Récentes ({len(errors)} total)</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Agent</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_performance_section(self, performance: Dict[str, Any]) -> str:
        """Génère la section performance."""
        if not performance:
            return ""
        
        rows = ""
        for operation, stats in performance.items():
            rows += f"""
            <tr>
                <td>{operation}</td>
                <td>{stats['avg']:.2f}s</td>
                <td>{stats['min']:.2f}s</td>
                <td>{stats['max']:.2f}s</td>
            </tr>
            """
        
        return f"""
        <div class="chart-container">
            <h2 class="chart-title">⚡ Performance des Opérations</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Opération</th>
                        <th>Moyenne</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_llm_section(self, llm_usage: Dict[str, Any]) -> str:
        """Génère la section utilisation LLM."""
        rows = ""
        for model, count in llm_usage.get('calls_by_model', {}).items():
            rows += f"""
            <tr>
                <td>{model}</td>
                <td>{count}</td>
                <td>{(count / llm_usage['total_calls'] * 100):.1f}%</td>
            </tr>
            """
        
        return f"""
        <div class="chart-container">
            <h2 class="chart-title">🤖 Utilisation des Modèles LLM</h2>
            <div class="metrics-grid" style="margin-bottom: 1rem;">
                <div class="metric-card">
                    <div class="metric-label">Tokens Total</div>
                    <div class="metric-value">{llm_usage.get('total_tokens', 0):,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Coût Estimé</div>
                    <div class="metric-value">${llm_usage.get('total_cost_estimate', 0):.2f}</div>
                </div>
            </div>
            <table class="table">
                <thead>
                    <tr>
                        <th>Modèle</th>
                        <th>Appels</th>
                        <th>Pourcentage</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_agents_chart_js(self, agents_data: Dict[str, Any]) -> str:
        """Génère le code JavaScript pour le graphique des agents."""
        return f"""
        new Chart(document.getElementById('agentsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(agents_data['labels'])},
                datasets: [{{
                    label: 'Contributions',
                    data: {json.dumps(agents_data['values'])},
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        """
    


    def _generate_communications_chart_js(self, comm_data: Dict[str, Any]) -> str:
        """Génère le code JavaScript pour le graphique des communications."""
        # Gérer le cas où il n'y a pas de données
        if not comm_data['labels'] or not comm_data['values']:
            labels = ['Aucune communication']
            values = [0]
        else:
            # Limiter aux 10 communications les plus fréquentes
            sorted_data = sorted(
                zip(comm_data['labels'], comm_data['values']),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            labels = [d[0] for d in sorted_data]
            values = [d[1] for d in sorted_data]
        
        return f"""
        new Chart(document.getElementById('communicationsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Messages',
                    data: {json.dumps(values)},
                    backgroundColor: 'rgba(155, 89, 182, 0.6)',
                    borderColor: 'rgba(155, 89, 182, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        beginAtZero: true,
                        ticks: {{
                            precision: 0
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        """



   
    def generate_progress_report(self, milestones: List[Dict[str, Any]]) -> str:
        """Génère un rapport de progression détaillé."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rapport de Progression - {self.project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .milestone {{ background: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .milestone.completed {{ border-left: 5px solid #27ae60; }}
        .milestone.pending {{ border-left: 5px solid #f39c12; }}
        .milestone.failed {{ border-left: 5px solid #e74c3c; }}
        .agents {{ margin-top: 10px; }}
        .agent-badge {{ display: inline-block; background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; margin: 2px; }}
        .timeline {{ margin: 20px 0; }}
        .timeline-item {{ margin: 10px 0; padding-left: 30px; position: relative; }}
        .timeline-item::before {{ content: ''; position: absolute; left: 0; top: 5px; width: 10px; height: 10px; background: #3498db; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Rapport de Progression - {self.project_name}</h1>
        <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>🎯 Jalons du Projet</h2>
    """
        
        for milestone in milestones:
            status_class = milestone.get('status', 'pending')
            status_emoji = {
                'completed': '✅',
                'pending': '⏳',
                'failed': '❌'
            }.get(status_class, '❓')
            
            html_content += f"""
    <div class="milestone {status_class}">
        <h3>{status_emoji} {milestone['name']}</h3>
        <p><strong>Description:</strong> {milestone.get('description', 'N/A')}</p>
        <p><strong>Statut:</strong> {milestone.get('status', 'pending').upper()}</p>
        <div class="agents">
            <strong>Agents impliqués:</strong>
            {' '.join([f'<span class="agent-badge">{agent}</span>' for agent in milestone.get('agents_required', [])])}
        </div>
        <p><strong>Durée estimée:</strong> {milestone.get('estimated_duration', 'N/A')}</p>
    </div>
            """
        
        html_content += """
</body>
</html>"""
        
        # Sauvegarder
        report_path = self.output_path / f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path.write_text(html_content, encoding='utf-8')
        
        return str(report_path)
