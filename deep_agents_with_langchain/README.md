# LangChain Deep Agents Implementation

This is the **LangChain-powered implementation** of the Deep Search with Agents system, featuring true deep agent capabilities with hierarchical planning, reflection loops, and advanced multi-agent collaboration.

## Architecture Overview

### Deep Agent Features
- **Hierarchical Planning**: Multi-level reasoning and task decomposition
- **Reflection Loops**: Self-assessment and iterative improvement
- **Meta-Reasoning**: Agents that reason about their own reasoning processes
- **Cross-Agent Collaboration**: Dynamic collaboration patterns and consensus building
- **Adaptive Planning**: Dynamic strategy adjustment based on query complexity

### Agent Specializations

1. **Research Deep Agent** (`ResearchDeepAgent`)
   - Academic literature search and analysis
   - Citation analysis and validation
   - Research synthesis and gap identification
   - Quality assessment of academic sources

2. **News Deep Agent** (`NewsDeepAgent`)
   - Real-time news analysis and fact-checking
   - Source credibility assessment
   - Trend analysis and breaking news detection
   - Bias detection and perspective analysis

3. **General Deep Agent** (`GeneralDeepAgent`)
   - Adaptive search strategies across domains
   - Multi-perspective analysis and synthesis
   - Context enrichment and knowledge integration
   - Query type classification and routing

4. **Reflection Agent** (`ReflectionAgent`)
   - Meta-cognitive analysis and quality assessment
   - Bias detection and mitigation strategies
   - Gap analysis and improvement suggestions
   - Process reflection and optimization

5. **Deep Search Orchestrator** (`DeepSearchOrchestrator`)
   - Intelligent agent selection and coordination
   - Multi-modal collaboration strategies
   - Consensus building and result synthesis
   - Performance monitoring and optimization

## Collaboration Modes

### 1. Hierarchical Mode
- **Master-Worker Pattern**: Orchestrator delegates specialized tasks
- **Structured Analysis**: Step-by-step reasoning with clear dependencies
- **Quality Gates**: Validation at each level of the hierarchy
- **Best For**: Complex research queries, academic analysis

### 2. Parallel Mode
- **Simultaneous Processing**: Multiple agents work independently
- **Result Aggregation**: Combine diverse perspectives efficiently
- **Time Optimization**: Faster processing for multi-faceted queries
- **Best For**: News analysis, comparative studies

### 3. Sequential Mode
- **Pipeline Processing**: Results flow from one agent to the next
- **Iterative Refinement**: Each agent builds on previous results
- **Cumulative Knowledge**: Progressive understanding development
- **Best For**: Step-by-step analysis, building complex arguments

### 4. Consensus Mode
- **Collaborative Decision Making**: Agents negotiate and agree
- **Conflict Resolution**: Handle contradictory information
- **Confidence Weighting**: Balance different perspectives
- **Best For**: Controversial topics, conflicting information

## Key Differentiators from Standard Implementation

### Advanced Planning
```python
# Hierarchical planning with decomposition
def plan_hierarchical_search(self, query: str) -> SearchPlan:
    # Analyze query complexity and requirements
    analysis = self.analyze_query_requirements(query)
    
    # Decompose into sub-tasks
    subtasks = self.decompose_query(query, analysis)
    
    # Create execution plan with dependencies
    plan = self.create_execution_plan(subtasks)
    
    return plan
```

### Reflection Loops
```python
# Self-assessment and improvement
def reflect_and_improve(self, result: DeepAgentResult) -> DeepAgentResult:
    # Assess result quality
    quality_score = self.assess_quality(result)
    
    if quality_score < self.quality_threshold:
        # Identify improvement areas
        improvements = self.identify_improvements(result)
        
        # Re-execute with improvements
        return self.execute_with_improvements(improvements)
    
    return result
```

### Meta-Reasoning
```python
# Reasoning about reasoning processes
def meta_analyze_approach(self, query: str, approach: str) -> Dict[str, Any]:
    # Evaluate the effectiveness of the chosen approach
    effectiveness = self.evaluate_approach_effectiveness(query, approach)
    
    # Suggest alternative approaches if needed
    alternatives = self.suggest_alternatives(query, effectiveness)
    
    return {
        'effectiveness': effectiveness,
        'alternatives': alternatives,
        'recommendations': self.generate_recommendations(alternatives)
    }
```

## Installation and Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Set required environment variables
export OPENAI_API_KEY="your_openai_api_key"
export SERPER_API_KEY="your_serper_api_key"  # Optional but recommended
export TAVILY_API_KEY="your_tavily_api_key"  # Optional
```

### 3. Running the Application

#### Command Line Interface
```bash
python main.py
```

#### Web Interface
```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Usage Examples

### Basic Usage
```python
from agents.deep_search_orchestrator import DeepSearchOrchestrator, AgentCollaborationMode

# Initialize orchestrator
orchestrator = DeepSearchOrchestrator()

# Execute search with different collaboration modes
result = await orchestrator.search(
    query="Latest developments in quantum computing",
    collaboration_mode=AgentCollaborationMode.HIERARCHICAL
)

print(f"Final Result: {result['final_result']}")
print(f"Confidence: {result['overall_confidence']}")
```

### Advanced Configuration
```python
# Custom agent configuration
from config.settings import settings

# Update agent settings
settings.update_agent_config('research', {
    'max_iterations': 7,
    'confidence_threshold': 0.9,
    'enable_citation_analysis': True
})

# Enable advanced features
settings.DEEP_AGENT_SETTINGS['enable_meta_reasoning'] = True
settings.DEEP_AGENT_SETTINGS['planning_horizon'] = 7
```

## Configuration Options

### Deep Agent Settings
```python
DEEP_AGENT_SETTINGS = {
    "enable_hierarchical_planning": True,    # Multi-level planning
    "enable_reflection_loops": True,         # Self-improvement
    "enable_meta_reasoning": True,           # Reasoning about reasoning
    "enable_collaborative_reasoning": True,  # Multi-agent collaboration
    "planning_horizon": 5,                   # Planning depth
    "reflection_frequency": 2,               # How often to reflect
    "memory_consolidation_interval": 10,     # Memory management
    "cross_agent_communication": True        # Inter-agent communication
}
```

### Orchestration Settings
```python
ORCHESTRATION_SETTINGS = {
    "query_analysis_model": "gpt-4",         # Model for query analysis
    "route_planning_enabled": True,          # Intelligent routing
    "adaptive_routing": True,                # Dynamic routing adjustment
    "quality_gates_enabled": True,           # Quality checkpoints
    "consensus_voting_enabled": True,        # Consensus mechanisms
    "meta_agent_enabled": True,              # Meta-agent oversight
    "learning_enabled": True                 # System learning
}
```

## Performance and Monitoring

### Built-in Metrics
- **Execution Time**: Per-agent and total execution time
- **Confidence Scores**: Individual and overall confidence levels
- **Quality Metrics**: Result quality assessment
- **Collaboration Effectiveness**: Inter-agent collaboration success
- **Planning Efficiency**: Planning vs. execution time ratios

### Logging and Tracing
```python
from utils.logger import get_agent_logger, get_performance_logger

# Agent-specific logging
logger = get_agent_logger("research_agent")
logger.log_agent_start(query, "research")

# Performance monitoring
perf_logger = get_performance_logger()
timer_id = perf_logger.start_timer("search_execution")
# ... execute search ...
duration = perf_logger.end_timer(timer_id)
```

## Comparison with Standard Implementation

| Feature | Standard Implementation | LangChain Deep Agents |
|---------|------------------------|----------------------|
| Agent Architecture | Simple BaseAgent | Deep agents with planning |
| Reasoning | Single-step | Multi-step with reflection |
| Collaboration | Basic orchestration | Advanced collaboration modes |
| Quality Assurance | Basic validation | Reflection loops + meta-reasoning |
| Planning | Static routing | Dynamic hierarchical planning |
| Learning | No learning | Experience-based improvement |
| Scalability | Limited | Advanced orchestration |

## Advanced Features

### 1. Hierarchical Planning
- Multi-level task decomposition
- Dependency management
- Resource allocation
- Progress monitoring

### 2. Reflection and Meta-Reasoning
- Quality self-assessment
- Approach evaluation
- Strategy optimization
- Learning from experience

### 3. Collaborative Intelligence
- Consensus building
- Conflict resolution
- Knowledge synthesis
- Collective decision making

### 4. Adaptive Behavior
- Dynamic strategy selection
- Context-aware routing
- Performance optimization
- Continuous improvement

## Future Enhancements

- **Memory Systems**: Long-term memory and experience retention
- **Learning Algorithms**: Reinforcement learning for strategy optimization
- **Tool Integration**: Enhanced tool ecosystem and plugin architecture
- **Multi-Modal Capabilities**: Vision and audio processing integration
- **Real-Time Adaptation**: Dynamic model and strategy switching

## Contributing

This implementation showcases advanced AI agent architectures and can be extended with:
- Custom agent specializations
- Additional collaboration modes
- Enhanced planning algorithms
- Advanced reflection mechanisms
- Custom tool integrations

## License

This project is part of the Deep Search with Agents system and follows the same licensing terms.
