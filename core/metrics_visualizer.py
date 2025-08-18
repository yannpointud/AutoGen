"""
Module de visualisation des métriques du système multi-agents.
Tableaux de bord et rapports visuels refondus pour les logs actuels.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

from utils.logger import get_project_logger


class ModernMetricsCollector:
    """Collecteur de métriques moderne basé sur les logs actuels."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.logger = get_project_logger(project_name, "MetricsCollector")
        self.project_path = Path("projects") / project_name
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collecte toutes les métriques disponibles."""
        logs_path = self.project_path / "logs"
        if not logs_path.exists():
            self.logger.warning(f"Aucun dossier de logs trouvé: {logs_path}")
            return self._empty_metrics()
        
        # Collecter depuis les différentes sources
        main_logs = self._collect_from_main_logs(logs_path)
        llm_logs = self._collect_from_llm_debug(logs_path / "llm_debug")
        
        # Fusionner et calculer les métriques finales
        return self._calculate_final_metrics(main_logs, llm_logs)
    
    def _collect_from_main_logs(self, logs_path: Path) -> Dict[str, Any]:
        """Collecte les métriques depuis les logs principaux."""
        metrics = {
            'tools_executed': [],
            'errors': [],
            'agent_activities': defaultdict(list),
            'compression_events': [],
            'project_timeline': []
        }
        
        for log_file in logs_path.glob("*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            self._process_main_log_entry(entry, metrics)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.logger.debug(f"Erreur lecture {log_file}: {e}")
        
        return metrics
    
    def _collect_from_llm_debug(self, llm_debug_path: Path) -> Dict[str, Any]:
        """Collecte les métriques depuis les logs LLM debug."""
        metrics = {
            'llm_calls': [],
            'token_usage': [],
            'model_usage': defaultdict(int),
            'agent_llm_stats': defaultdict(lambda: {'calls': 0, 'tokens': 0, 'duration': 0})
        }
        
        if not llm_debug_path.exists():
            return metrics
        
        for log_file in llm_debug_path.glob("*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            self._process_llm_log_entry(entry, metrics)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.logger.debug(f"Erreur lecture LLM {log_file}: {e}")
        
        return metrics
    
    def _process_main_log_entry(self, entry: Dict[str, Any], metrics: Dict[str, Any]):
        """Traite une entrée des logs principaux."""
        timestamp = entry.get('timestamp', '')
        agent_name = entry.get('agent_name', 'system')
        level = entry.get('level', 'INFO')
        message = entry.get('message', '')
        
        # Activités des agents
        metrics['agent_activities'][agent_name].append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # Outils exécutés
        if 'tool_name' in entry:
            metrics['tools_executed'].append({
                'tool': entry['tool_name'],
                'agent': agent_name,
                'status': entry.get('status', 'unknown'),
                'timestamp': timestamp
            })
        
        # Erreurs
        if level == 'ERROR':
            metrics['errors'].append({
                'agent': agent_name,
                'message': message,
                'timestamp': timestamp
            })
        
        # Événements de compression
        if 'compression' in message.lower() or 'compres' in message.lower():
            metrics['compression_events'].append({
                'agent': agent_name,
                'message': message,
                'timestamp': timestamp
            })
        
        # Timeline du projet
        metrics['project_timeline'].append({
            'timestamp': timestamp,
            'agent': agent_name,
            'event': message,
            'level': level
        })
    
    def _process_llm_log_entry(self, entry: Dict[str, Any], metrics: Dict[str, Any]):
        """Traite une entrée des logs LLM debug."""
        agent_name = entry.get('agent_name', 'unknown')
        model = entry.get('model', 'unknown')
        direction = entry.get('direction', 'unknown')
        sequence_id = entry.get('sequence_id', 0)
        
        if direction == 'REQUEST':
            # Stocker les tokens d'entrée par sequence_id (convertir chars -> tokens)
            input_chars = entry.get('prompt_total_length', 0)
            input_tokens = input_chars // 3  # Conversion chars -> tokens
            
            if 'pending_requests' not in metrics:
                metrics['pending_requests'] = {}
            metrics['pending_requests'][sequence_id] = {
                'agent': agent_name,
                'model': model,
                'input_tokens': input_tokens,
                'timestamp': entry.get('timestamp', '')
            }
            
        elif direction == 'RESPONSE':
            output_tokens = entry.get('tokens_used', 0)
            duration = entry.get('duration_seconds', 0)
            timestamp = entry.get('timestamp', '')
            
            # Récupérer les tokens d'entrée correspondants
            pending_requests = metrics.get('pending_requests', {})
            request_info = pending_requests.get(sequence_id, {})
            input_tokens = request_info.get('input_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            metrics['llm_calls'].append({
                'agent': agent_name,
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'duration': duration,
                'timestamp': timestamp
            })
            
            metrics['token_usage'].append(total_tokens)
            metrics['model_usage'][model] += 1
            
            # Stats par agent avec détail entrée/sortie
            agent_stats = metrics['agent_llm_stats'][agent_name]
            agent_stats['calls'] += 1
            agent_stats['total_tokens'] = agent_stats.get('total_tokens', 0) + total_tokens
            agent_stats['input_tokens'] = agent_stats.get('input_tokens', 0) + input_tokens
            agent_stats['output_tokens'] = agent_stats.get('output_tokens', 0) + output_tokens
            agent_stats['duration'] += duration
            
            # Nettoyer la requête traitée
            if sequence_id in pending_requests:
                del pending_requests[sequence_id]
    
    def _calculate_final_metrics(self, main_logs: Dict[str, Any], llm_logs: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les métriques finales à partir des données collectées."""
        
        # Calculer la durée du projet
        all_timestamps = []
        for timeline_entry in main_logs['project_timeline']:
            if timeline_entry['timestamp']:
                all_timestamps.append(timeline_entry['timestamp'])
        
        project_start = min(all_timestamps) if all_timestamps else datetime.now().isoformat()
        project_end = max(all_timestamps) if all_timestamps else datetime.now().isoformat()
        
        try:
            start_dt = datetime.fromisoformat(project_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(project_end.replace('Z', '+00:00'))
            duration_minutes = (end_dt - start_dt).total_seconds() / 60
        except:
            duration_minutes = 0
        
        # Compter les outils
        tools_by_type = Counter(tool['tool'] for tool in main_logs['tools_executed'])
        tools_by_status = Counter(tool['status'] for tool in main_logs['tools_executed'])
        tools_by_agent = Counter(tool['agent'] for tool in main_logs['tools_executed'])
        
        # Stats LLM
        total_tokens = sum(llm_logs['token_usage'])
        total_llm_calls = len(llm_logs['llm_calls'])
        avg_tokens_per_call = total_tokens / total_llm_calls if total_llm_calls > 0 else 0
        
        # Note: Coût supprimé car trop dépendant du modèle
        
        return {
            'project_info': {
                'name': self.project_name,
                'start_time': project_start,
                'end_time': project_end,
                'duration_minutes': round(duration_minutes, 2),
                'total_events': len(main_logs['project_timeline'])
            },
            'agents_stats': {
                'active_agents': list(main_logs['agent_activities'].keys()),
                'agent_activity_counts': {
                    agent: len(activities) 
                    for agent, activities in main_logs['agent_activities'].items()
                },
                'llm_stats_by_agent': dict(llm_logs['agent_llm_stats'])
            },
            'tools_stats': {
                'total_tools_executed': len(main_logs['tools_executed']),
                'tools_by_type': dict(tools_by_type),
                'tools_by_status': dict(tools_by_status),
                'tools_by_agent': dict(tools_by_agent),
                'success_rate': round(
                    tools_by_status.get('success', 0) / max(len(main_logs['tools_executed']), 1) * 100, 1
                )
            },
            'llm_stats': {
                'total_calls': total_llm_calls,
                'total_tokens': total_tokens,
                'average_tokens_per_call': round(avg_tokens_per_call, 1),
                'models_used': dict(llm_logs['model_usage']),
                'compression_events': len(main_logs['compression_events'])
            },
            'errors_stats': {
                'total_errors': len(main_logs['errors']),
                'errors_by_agent': dict(Counter(error['agent'] for error in main_logs['errors'])),
                'recent_errors': main_logs['errors'][-5:] if main_logs['errors'] else []
            },
            'timeline': main_logs['project_timeline'][-20:] if main_logs['project_timeline'] else []
        }
    
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Retourne des métriques vides."""
        return {
            'project_info': {
                'name': self.project_name,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': 0,
                'total_events': 0
            },
            'agents_stats': {
                'active_agents': [],
                'agent_activity_counts': {},
                'llm_stats_by_agent': {}
            },
            'tools_stats': {
                'total_tools_executed': 0,
                'tools_by_type': {},
                'tools_by_status': {},
                'tools_by_agent': {},
                'success_rate': 0
            },
            'llm_stats': {
                'total_calls': 0,
                'total_tokens': 0,
                'average_tokens_per_call': 0,
                'models_used': {},
                'compression_events': 0
            },
            'errors_stats': {
                'total_errors': 0,
                'errors_by_agent': {},
                'recent_errors': []
            },
            'timeline': []
        }


class ModernMetricsVisualizer:
    """Visualiseur de métriques moderne avec HTML autonome."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.logger = get_project_logger(project_name, "MetricsVisualizer")
        self.collector = ModernMetricsCollector(project_name)
        self.output_path = Path("projects") / project_name / "metrics"
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard(self, rag_singleton=None) -> str:
        """Génère un tableau de bord HTML complet et autonome."""
        metrics = self.collector.collect_all_metrics()
        html = self._generate_modern_html_dashboard(metrics)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_path = self.output_path / f"dashboard_{timestamp}.html"
        dashboard_path.write_text(html, encoding='utf-8')
        self.logger.info(f"Dashboard moderne généré: {dashboard_path}")
        return str(dashboard_path)
    
    def _generate_modern_html_dashboard(self, metrics: Dict[str, Any]) -> str:
        """Génère un dashboard HTML moderne et autonome."""
        
        project_info = metrics['project_info']
        agents_stats = metrics['agents_stats']
        tools_stats = metrics['tools_stats']
        llm_stats = metrics['llm_stats']
        errors_stats = metrics['errors_stats']
        
        # Préparer les données pour les graphiques
        agents_data = [
            {'agent': agent, 'count': count} 
            for agent, count in agents_stats['agent_activity_counts'].items()
        ]
        
        tools_data = [
            {'tool': tool, 'count': count}
            for tool, count in tools_stats['tools_by_type'].items()
        ]
        
        models_data = [
            {'model': model.replace('-latest', ''), 'count': count}
            for model, count in llm_stats['models_used'].items()
        ]
        
        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - {project_info['name']}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, var(--primary-color), #3b82f6);
            color: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }}
        
        .metric-card h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 4px;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .chart-container {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }}
        
        .chart-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text-primary);
        }}
        
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .bar-item {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .bar-label {{
            min-width: 120px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-primary);
        }}
        
        .bar-track {{
            flex: 1;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), #3b82f6);
            border-radius: 4px;
            transition: width 0.8s ease;
        }}
        
        .bar-value {{
            min-width: 40px;
            text-align: right;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .timeline {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--border-color);
        }}
        
        .timeline-item {{
            display: flex;
            gap: 16px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .timeline-item:last-child {{
            border-bottom: none;
        }}
        
        .timeline-time {{
            min-width: 80px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: 'Monaco', 'Menlo', monospace;
        }}
        
        .timeline-agent {{
            min-width: 80px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--primary-color);
        }}
        
        .timeline-event {{
            flex: 1;
            font-size: 0.85rem;
            color: var(--text-primary);
        }}
        
        .status-success {{ color: var(--success-color); }}
        .status-warning {{ color: var(--warning-color); }}
        .status-error {{ color: var(--error-color); }}
        
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2rem;
            }}
        }}
        
        .no-data {{
            text-align: center;
            color: var(--text-secondary);
            font-style: italic;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{project_info['name']}</h1>
            <p>Tableau de bord généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Durée du projet</h3>
                <div class="metric-value">{project_info['duration_minutes']:.0f}</div>
                <div class="metric-label">minutes</div>
            </div>
            
            <div class="metric-card">
                <h3>Agents actifs</h3>
                <div class="metric-value">{len(agents_stats['active_agents'])}</div>
                <div class="metric-label">agents</div>
            </div>
            
            <div class="metric-card">
                <h3>Outils exécutés</h3>
                <div class="metric-value">{tools_stats['total_tools_executed']}</div>
                <div class="metric-label">{tools_stats['success_rate']}% de succès</div>
            </div>
            
            <div class="metric-card">
                <h3>Appels LLM</h3>
                <div class="metric-value">{llm_stats['total_calls']}</div>
                <div class="metric-label">{llm_stats['total_tokens']:,} tokens totaux</div>
            </div>
            
            <div class="metric-card">
                <h3>Compression</h3>
                <div class="metric-value">{llm_stats['compression_events']}</div>
                <div class="metric-label">événements de compression</div>
            </div>
            
            <div class="metric-card">
                <h3>Erreurs</h3>
                <div class="metric-value status-error">{errors_stats['total_errors']}</div>
                <div class="metric-label">erreurs rencontrées</div>
            </div>
        </div>
        
        <div class="two-column">
            <div class="chart-container">
                <div class="chart-title">Activité par Agent</div>
                <div class="bar-chart">
                    {self._generate_bar_chart_html(agents_data, 'agent', 'count')}
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Outils Utilisés</div>
                <div class="bar-chart">
                    {self._generate_bar_chart_html(tools_data, 'tool', 'count')}
                </div>
            </div>
        </div>
        
        <div class="two-column">
            <div class="chart-container">
                <div class="chart-title">Modèles LLM</div>
                <div class="bar-chart">
                    {self._generate_bar_chart_html(models_data, 'model', 'count')}
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Stats LLM par Agent</div>
                <div class="bar-chart">
                    {self._generate_llm_agent_stats_html(agents_stats['llm_stats_by_agent'])}
                </div>
            </div>
        </div>
        
        <div class="timeline">
            <div class="chart-title">Timeline Récente</div>
            {self._generate_timeline_html(metrics['timeline'])}
        </div>
    </div>
    
    <script>
        // Animation des barres au chargement
        document.addEventListener('DOMContentLoaded', function() {{
            const bars = document.querySelectorAll('.bar-fill');
            bars.forEach(bar => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, 100);
            }});
        }});
    </script>
</body>
</html>"""
    
    def _generate_bar_chart_html(self, data: List[Dict[str, Any]], label_key: str, value_key: str) -> str:
        """Génère le HTML pour un graphique en barres."""
        if not data:
            return '<div class="no-data">Aucune donnée disponible</div>'
        
        max_value = max(item[value_key] for item in data) if data else 1
        html_parts = []
        
        for item in sorted(data, key=lambda x: x[value_key], reverse=True)[:10]:
            label = item[label_key]
            value = item[value_key]
            percentage = (value / max_value) * 100 if max_value > 0 else 0
            
            html_parts.append(f"""
                <div class="bar-item">
                    <div class="bar-label">{label}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {percentage}%"></div>
                    </div>
                    <div class="bar-value">{value}</div>
                </div>
            """)
        
        return ''.join(html_parts)
    
    def _generate_llm_agent_stats_html(self, llm_stats: Dict[str, Dict[str, Any]]) -> str:
        """Génère le HTML pour les stats LLM par agent."""
        if not llm_stats:
            return '<div class="no-data">Aucune donnée LLM disponible</div>'
        
        html_parts = []
        for agent, stats in llm_stats.items():
            total_tokens = stats.get('total_tokens', 0)
            input_tokens = stats.get('input_tokens', 0)
            output_tokens = stats.get('output_tokens', 0)
            calls = stats.get('calls', 0)
            avg_tokens = round(total_tokens / calls, 1) if calls > 0 else 0
            
            html_parts.append(f"""
                <div class="bar-item">
                    <div class="bar-label">{agent}</div>
                    <div class="timeline-event">{calls} appels • {total_tokens:,} tokens ({input_tokens:,} in + {output_tokens:,} out) • {avg_tokens} moy/appel</div>
                </div>
            """)
        
        return ''.join(html_parts)
    
    def _generate_timeline_html(self, timeline: List[Dict[str, Any]]) -> str:
        """Génère le HTML pour la timeline."""
        if not timeline:
            return '<div class="no-data">Aucun événement dans la timeline</div>'
        
        html_parts = []
        for event in timeline[-15:]:  # Derniers 15 événements
            timestamp = event.get('timestamp', '')
            agent = event.get('agent', 'system')
            message = event.get('event', '')
            level = event.get('level', 'INFO')
            
            # Extraire juste l'heure
            try:
                time_only = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%H:%M:%S')
            except:
                time_only = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            # Tronquer le message s'il est trop long
            if len(message) > 80:
                message = message[:77] + "..."
            
            status_class = {
                'ERROR': 'status-error',
                'WARNING': 'status-warning',
                'INFO': 'status-success'
            }.get(level, '')
            
            html_parts.append(f"""
                <div class="timeline-item">
                    <div class="timeline-time">{time_only}</div>
                    <div class="timeline-agent">{agent}</div>
                    <div class="timeline-event {status_class}">{message}</div>
                </div>
            """)
        
        return ''.join(html_parts)


# Fonctions utilitaires pour maintenir la compatibilité
def get_metrics_collector(project_name: str) -> ModernMetricsCollector:
    """Retourne un collecteur de métriques moderne."""
    return ModernMetricsCollector(project_name)

def generate_dashboard(project_name: str) -> str:
    """Génère un dashboard moderne pour un projet."""
    visualizer = ModernMetricsVisualizer(project_name)
    return visualizer.generate_dashboard()

# Alias pour compatibilité
MetricsCollector = ModernMetricsCollector
MetricsVisualizer = ModernMetricsVisualizer