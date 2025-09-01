"""
Flask web application for LangChain Deep Agents
"""

import logging
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template_string  # pylint: disable=import-error # type: ignore
from config.settings import settings

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Simple HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain Deep Agents</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .search-form { background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; }
        .results { background: #f9f9f9; padding: 20px; border-radius: 8px; }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        select { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005a87; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .error { color: red; margin: 10px 0; }
        .success { color: green; margin: 10px 0; }
        .agent-result { border-left: 4px solid #007cba; padding: 10px; margin: 10px 0; background: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LangChain Deep Agents</h1>
        <p>Advanced AI-powered search system with specialized agents and collaborative reasoning</p>
    </div>

    <div class="search-form">
        <h2>Search Query</h2>
        <form id="searchForm">
            <label for="query">Enter your search query:</label>
            <input type="text" id="query" name="query" placeholder="e.g., Latest developments in quantum computing..." required>
            
            <label for="mode">Collaboration Mode:</label>
            <select id="mode" name="mode">
                <option value="hierarchical">Hierarchical - Structured analysis</option>
                <option value="parallel">Parallel - Simultaneous processing</option>
                <option value="sequential">Sequential - Step-by-step</option>
                <option value="consensus">Consensus - Collaborative agreement</option>
            </select>
            
            <button type="submit">Search</button>
        </form>
        
        <div class="loading" id="loading">
            <p>🔍 Deep agents are working on your query...</p>
            <p>This may take a few moments for complex queries.</p>
        </div>
        
        <div id="error" class="error"></div>
        <div id="success" class="success"></div>
    </div>

    <div class="results" id="results" style="display: none;">
        <h2>Search Results</h2>
        <div id="resultContent"></div>
    </div>

    <script>
        document.getElementById('searchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const mode = document.getElementById('mode').value;
            
            // Show loading, hide results
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').textContent = '';
            document.getElementById('success').textContent = '';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query, mode: mode })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                    document.getElementById('success').textContent = 'Search completed successfully!';
                } else {
                    document.getElementById('error').textContent = data.error || 'Search failed';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'Network error: ' + error.message;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const resultContent = document.getElementById('resultContent');
            
            let html = `
                <h3>Query: ${data.query}</h3>
                <p><strong>Mode:</strong> ${data.collaboration_mode}</p>
                <p><strong>Execution Time:</strong> ${data.execution_time} seconds</p>
                <p><strong>Overall Confidence:</strong> ${data.overall_confidence}</p>
                
                <h4>Final Result:</h4>
                <div class="agent-result">
                    ${data.final_result || 'No final result available'}
                </div>
            `;
            
            if (data.agent_results) {
                html += '<h4>Individual Agent Results:</h4>';
                for (const [agentName, result] of Object.entries(data.agent_results)) {
                    html += `
                        <div class="agent-result">
                            <h5>${agentName.toUpperCase()}</h5>
                            <p><strong>Confidence:</strong> ${result.confidence || 'N/A'}</p>
                            <p>${result.final_result || result}</p>
                        </div>
                    `;
                }
            }
            
            resultContent.innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400

        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        mode = data.get('mode', 'hierarchical')

        # For now, return a mock response since the orchestrator isn't fully integrated
        # In a real implementation, you would:
        # orchestrator = DeepSearchOrchestrator()
        # result = await orchestrator.search(query, collaboration_mode=mode)

        mock_result = create_mock_search_result(query, mode)

        return jsonify(mock_result)

    except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
        logger.error("Search error: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'LangChain Deep Agents'
    })


@app.route('/config')
def config_info():
    """Return configuration information (non-sensitive)"""
    return jsonify({
        'service': 'LangChain Deep Agents',
        'features': {
            'deep_agent_features': list(getattr(settings, 'DEEP_AGENT_SETTINGS', {}).keys()),
            'available_agents': list(getattr(settings, 'AGENT_CONFIGS', {}).keys()),
            'orchestration_modes': ['hierarchical', 'parallel', 'sequential', 'consensus']
        },
        'status': 'operational'
    })


def create_mock_search_result(query: str, mode: str) -> Dict[str, Any]:
    """Create a mock search result for demonstration"""
    return {
        'query': query,
        'collaboration_mode': mode,
        'execution_time': '2.45',
        'overall_confidence': '0.85',
        'final_result': f"""
Based on a comprehensive analysis using {mode} collaboration mode, here are the key findings for your query: "{query}"

This is a demonstration of the LangChain Deep Agents system. In the full implementation, multiple specialized AI agents would:

1. **Research Agent**: Conduct academic literature searches and analysis
2. **News Agent**: Gather current events and breaking news
3. **General Agent**: Provide comprehensive background information
4. **Reflection Agent**: Validate and improve the overall results

The system uses advanced techniques including:
- Hierarchical planning and reasoning
- Multi-agent collaboration and consensus building
- Reflection loops for quality improvement
- Adaptive search strategies based on query complexity

For a fully functional system, please ensure all dependencies are installed and API keys are configured.
        """,
        'agent_results': {
            'research_agent': {
                'confidence': '0.88',
                'final_result': f'Research analysis results for: {query}'
            },
            'news_agent': {
                'confidence': '0.82',
                'final_result': f'Current news and developments related to: {query}'
            },
            'general_agent': {
                'confidence': '0.85',
                'final_result': f'General information and context for: {query}'
            }
        },
        'search_metadata': {
            'agents_used': ['research_agent', 'news_agent', 'general_agent'],
            'strategy': mode,
            'total_iterations': 3
        }
    }


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    print("Starting LangChain Deep Agents web application...")
    print("Open your browser to: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
