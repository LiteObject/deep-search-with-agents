"""
Enhanced logging utilities for LangChain Deep Agents
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class DeepAgentFormatter(logging.Formatter):
    """Custom formatter for deep agent logging"""

    def __init__(self):
        super().__init__()
        self.COLORS = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m'       # Reset
        }

    def format(self, record):
        # Add color for console output
        if hasattr(record, 'use_color') and record.use_color:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"

        # Format timestamp
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Add agent context if available
        agent_context = ""
        if hasattr(record, 'agent_name'):
            agent_context = f"[{record.agent_name}] "

        # Format the message
        formatted_message = f"{record.timestamp} - {record.levelname} - {agent_context}{record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted_message += f"\n{self.formatException(record.exc_info)}"

        return formatted_message


class AgentLogger:
    """Specialized logger for deep agents with context tracking"""

    def __init__(self, name: str, log_file: Optional[str] = None, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)

        # Agent-specific context
        self.agent_context = {}

    def _setup_handlers(self, log_file: Optional[str] = None):
        """Setup logging handlers"""
        formatter = DeepAgentFormatter()

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(lambda record: setattr(
            record, 'use_color', True) or True)
        self.logger.addHandler(console_handler)

        # File handler without colors
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(lambda record: setattr(
                record, 'use_color', False) or True)
            self.logger.addHandler(file_handler)

    def set_agent_context(self, **context):
        """Set agent-specific context"""
        self.agent_context.update(context)

    def _add_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add agent context to log record"""
        context = self.agent_context.copy()
        if extra:
            context.update(extra)
        return context

    def debug(self, message: str, **kwargs):
        """Log debug message with agent context"""
        extra = self._add_context(kwargs)
        self.logger.debug(message, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info message with agent context"""
        extra = self._add_context(kwargs)
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with agent context"""
        extra = self._add_context(kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs):
        """Log error message with agent context"""
        extra = self._add_context(kwargs)
        self.logger.error(message, extra=extra)

    def critical(self, message: str, **kwargs):
        """Log critical message with agent context"""
        extra = self._add_context(kwargs)
        self.logger.critical(message, extra=extra)

    def log_agent_start(self, query: str, agent_type: Optional[str] = None):
        """Log agent start with context"""
        context = {
            'agent_name': agent_type or self.name,
            'query': query[:100] + "..." if len(query) > 100 else query,
            'operation': 'agent_start'
        }
        self.info("Starting agent processing", **context)

    def log_agent_end(self, duration: float, success: bool = True):
        """Log agent completion"""
        context = {
            'agent_name': self.name,
            'duration': duration,
            'success': success,
            'operation': 'agent_end'
        }
        if success:
            self.info(
                f"Agent processing completed in {duration:.2f}s", **context)
        else:
            self.warning(
                f"Agent processing failed after {duration:.2f}s", **context)

    def log_tool_usage(self, tool_name: str, result_summary: str = ""):
        """Log tool usage"""
        context = {
            'agent_name': self.name,
            'tool_name': tool_name,
            'result_summary': result_summary[:50] + "..." if len(result_summary) > 50 else result_summary,
            'operation': 'tool_usage'
        }
        self.debug(f"Using tool: {tool_name}", **context)

    def log_reflection(self, reflection_type: str, insights: str):
        """Log reflection activities"""
        context = {
            'agent_name': self.name,
            'reflection_type': reflection_type,
            'insights': insights[:100] + "..." if len(insights) > 100 else insights,
            'operation': 'reflection'
        }
        self.info(f"Reflection completed: {reflection_type}", **context)

    def log_collaboration(self, other_agent: str, collaboration_type: str, outcome: str):
        """Log inter-agent collaboration"""
        context = {
            'agent_name': self.name,
            'other_agent': other_agent,
            'collaboration_type': collaboration_type,
            'outcome': outcome,
            'operation': 'collaboration'
        }
        self.info(
            f"Collaboration with {other_agent}: {collaboration_type}", **context)

    def log_performance_metric(self, metric_name: str, value: Any, unit: str = ""):
        """Log performance metrics"""
        context = {
            'agent_name': self.name,
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'operation': 'performance_metric'
        }
        self.debug(
            f"Performance metric - {metric_name}: {value} {unit}", **context)


class OrchestrationLogger(AgentLogger):
    """Specialized logger for orchestration activities"""

    def __init__(self, log_file: Optional[str] = None, level: str = "INFO"):
        super().__init__("DeepSearchOrchestrator", log_file, level)

    def log_query_analysis(self, query: str, analysis_result: Dict[str, Any]):
        """Log query analysis results"""
        context = {
            'query': query[:100] + "..." if len(query) > 100 else query,
            'complexity': analysis_result.get('complexity_level', 'unknown'),
            'recommended_agents': analysis_result.get('recommended_agents', []),
            'operation': 'query_analysis'
        }
        self.info(
            f"Query analyzed: {analysis_result.get('complexity_level', 'unknown')} complexity", **context)

    def log_agent_selection(self, selected_agents: list, collaboration_mode: str):
        """Log agent selection decisions"""
        context = {
            'selected_agents': selected_agents,
            'collaboration_mode': collaboration_mode,
            'operation': 'agent_selection'
        }
        self.info(
            f"Selected agents: {', '.join(selected_agents)} ({collaboration_mode})", **context)

    def log_orchestration_start(self, query: str, strategy: str):
        """Log orchestration start"""
        context = {
            'query': query[:100] + "..." if len(query) > 100 else query,
            'strategy': strategy,
            'operation': 'orchestration_start'
        }
        self.info(
            f"Starting orchestration with strategy: {strategy}", **context)

    def log_orchestration_end(self, duration: float, success: bool, final_confidence: float):
        """Log orchestration completion"""
        context = {
            'duration': duration,
            'success': success,
            'final_confidence': final_confidence,
            'operation': 'orchestration_end'
        }
        if success:
            self.info(
                f"Orchestration completed in {duration:.2f}s (confidence: {final_confidence:.2f})", **context)
        else:
            self.warning(
                f"Orchestration failed after {duration:.2f}s", **context)

    def log_consensus_result(self, participating_agents: list, consensus_score: float, final_decision: str):
        """Log consensus building results"""
        context = {
            'participating_agents': participating_agents,
            'consensus_score': consensus_score,
            'final_decision': final_decision[:50] + "..." if len(final_decision) > 50 else final_decision,
            'operation': 'consensus'
        }
        self.info(
            f"Consensus reached (score: {consensus_score:.2f}) among {len(participating_agents)} agents", **context)


class PerformanceLogger:
    """Logger for performance monitoring and metrics"""

    def __init__(self, log_file: str = "performance.log"):
        self.logger = AgentLogger("PerformanceMonitor", log_file, "DEBUG")
        self.metrics = {}

    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{datetime.now().timestamp()}"
        self.metrics[timer_id] = {
            'operation': operation,
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """End timing and return duration"""
        if timer_id in self.metrics:
            end_time = datetime.now()
            self.metrics[timer_id]['end_time'] = end_time
            duration = (
                end_time - self.metrics[timer_id]['start_time']).total_seconds()
            self.metrics[timer_id]['duration'] = duration

            self.logger.log_performance_metric(
                f"{self.metrics[timer_id]['operation']}_duration",
                duration,
                "seconds"
            )
            return duration
        return 0.0

    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage"""
        self.logger.log_performance_metric(
            f"{component}_memory", memory_mb, "MB")

    def log_api_call(self, api_name: str, duration: float, success: bool, tokens_used: int = 0):
        """Log API call metrics"""
        context = {
            'api_name': api_name,
            'duration': duration,
            'success': success,
            'tokens_used': tokens_used,
            'operation': 'api_call'
        }
        self.logger.info(
            f"API call to {api_name}: {'success' if success else 'failed'} ({duration:.2f}s)", **context)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_operations': len(self.metrics),
            'operations': {},
            'timestamp': datetime.now().isoformat()
        }

        for metric in self.metrics.values():
            op_name = metric['operation']
            if op_name not in summary['operations']:
                summary['operations'][op_name] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0
                }

            if metric['duration']:
                summary['operations'][op_name]['count'] += 1
                summary['operations'][op_name]['total_duration'] += metric['duration']
                summary['operations'][op_name]['min_duration'] = min(
                    summary['operations'][op_name]['min_duration'],
                    metric['duration']
                )
                summary['operations'][op_name]['max_duration'] = max(
                    summary['operations'][op_name]['max_duration'],
                    metric['duration']
                )

        # Calculate averages
        for op_data in summary['operations'].values():
            if op_data['count'] > 0:
                op_data['avg_duration'] = op_data['total_duration'] / \
                    op_data['count']

        return summary

# Global logger instances


def get_agent_logger(agent_name: str, log_file: Optional[str] = None) -> AgentLogger:
    """Get or create an agent logger"""
    return AgentLogger(agent_name, log_file)


def get_orchestration_logger(log_file: Optional[str] = None) -> OrchestrationLogger:
    """Get orchestration logger"""
    return OrchestrationLogger(log_file)


def get_performance_logger(log_file: str = "performance.log") -> PerformanceLogger:
    """Get performance logger"""
    return PerformanceLogger(log_file)

# Setup basic logging configuration


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file) if log_file else logging.NullHandler()
        ]
    )
