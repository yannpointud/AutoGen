📖 [Version française](README.md)

# AutoGen - Multi-Agent AI Platform

🚀 Intelligent multi-agent platform that orchestrates specialized AI agents to design, develop and document complete software projects autonomously.

*Note: This project is not based on Microsoft's AutoGen framework*
*Initially, I planned to use it, but due to its limitations, I opted to rebuild it from scratch*
*The project name was kept as a nod to this experience* 😉

## ✨ Key Features

- **🤖 Specialized Agents**: A team composed of a Supervisor, an Analyst and a Developer, each with a role, personality and tools defined in default_config.yaml.

- **🧠 RAG Architecture**: Vector search engine (rag_engine.py) based on FAISS to provide persistent and relevant context to agents, with working memory management and automatic compression.

- **🔄 Inter-agent Communication**: Structured and configurable exchanges between agents for collaboration, problem solving and validation.

- **📊 Advanced Monitoring**: Generation of structured JSON logs (logger.py), detailed LLM traces and an HTML dashboard (metrics_visualizer.py) to track system performance.

- **⚡ Intelligent Rate Limiting**: A centralized manager (global_rate_limiter.py) prevents external API quota errors (Mistral, DeepSeek) with a retry policy.

- **⚙️ Centralized Configuration**: The entire platform behavior (LLM models, RAG parameters, agent guidelines) is driven by the config/default_config.yaml file.

- **✅ Automated Installation**: A complete shell script (setup.sh) handles environment creation, dependency installation and project validation.


## 🏗️ Architecture


![Architecture.png](Architecture.png)


## 🚀 Quick Installation

### Prerequisites
- Python 3.10+
- Conda (highly recommended)
- Mistral API key (recommended) or DeepSeek

### Automatic Installation (recommended)

```bash
# Clone the project
git clone https://github.com/yannpointud/AutoGen.git
cd AutoGen

# Complete installation in one command
./setup.sh
```

**Advanced options:**
```bash
./setup.sh --check-only   # Check existing installation
./setup.sh --recreate     # Recreate environment from scratch
./setup.sh --help         # Display help
```


### Manual Installation (if necessary)

```bash
# Create environment with config file
conda env create -f conda-requirements.yml
conda activate AutoGen

# (Alternative) manual creation
conda create -n AutoGen python=3.10
conda activate AutoGen
pip install -r requirements.txt

# Configuration
cp .env.example .env
mkdir -p projects logs
```


## 🔑 Configuration

Create a `.env` file with your API keys:

```env
MISTRAL_API_KEY=your_mistral_api_key_here # required
DEEPSEEK_API_KEY=your_deepseek_api_key_here  # optional
```

*Note: the platform operation is independent of the model used*
*It is planned to test integrating other models*
*At minimum, access to an LLM and an embeddings model is required*

**Get the keys:**
- Mistral: [console.mistral.ai](https://console.mistral.ai/)
- DeepSeek: [platform.deepseek.com](https://platform.deepseek.com/)


## 🎮 Usage / Quick Start

Once installation is complete:

```bash
# Activate the environment
conda activate AutoGen

# Launch the interactive interface
python main.py

# Or check that the environment is ok
./setup.sh --check-only
```

### Available Templates

1. **MLPricePredictor** - ML API for real estate price prediction
2. **Calculator**       - Calculator (Python/CLI)
3. **FileOrganizer**    - Automatic file organizer
4. **ChatBot**          - Simple assistant chatbot  
0. **Custom**           - Create your project with a prompt


## 📊 Monitoring and Metrics

AutoGen automatically generates:

- **Structured JSON logs**: `logs/platform_YYYYMMDD.jsonl`
- **Detailed LLM traces**: `logs/llm_debug/`
- **Interactive dashboard**: Real-time metrics with visualizations
- **Progress reports**: Status of milestones and tasks


## ⚙️ Advanced Configuration

### LLM Models

Customize models in `config/default_config.yaml`:

```yaml
llm:
  default_model: "mistral-small-latest"
  models:
    mistral:
      supervisor: "magistral-medium-latest"
      analyst: "magistral-medium-latest" 
      developer: "codestral-latest"
```

### RAG Parameters

```yaml
rag:
  chunk_size: 1000
  chunk_overlap: 200
  top_k_results: 5
  similarity_threshold: 0.7
```

## 🧪 Tests and Validation

```bash
# Check installation
./setup.sh --check-only

# Run all tests (after environment activation)
conda activate AutoGen
pytest tests/

# Tests with coverage
pytest --cov=. tests/

# Specific tests
python tests/test_phase5.py
```


## 🤝 Contributing

1. Fork the project
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request


## 📝 Changelog

Consult [CHANGELOG.md](CHANGELOG.md) for detailed version history.


## 🛠️ Troubleshooting

### Installation and Environment

**Installation problem**:
```bash
# Complete environment diagnostic
./setup.sh --check-only

# Completely recreate environment
./setup.sh --recreate

# See complete script help
./setup.sh --help
```

**Corrupted conda environment**:
```bash
# Clean and recreate
conda env remove -n AutoGen
./setup.sh
```

**Useful conda commands**:
```bash
# List environments
conda env list

# See installed packages
conda activate AutoGen && conda list

# Clean conda cache
conda clean --all

# Export current configuration
conda env export > my-environment.yml
```


### Runtime Issues

**API timeout error**:
```yaml
# Increase in config/default_config.yaml
general:
  llm_timeout: 180
```

**Rate limit reached**:
```yaml
# Slow down API calls
general:
  api_rate_limit_interval: 3
```


## 📚 Technical Documentation

- **Installation script**: `./setup.sh --help`
- **Version history**: [CHANGELOG.md](CHANGELOG.md)
- **API Reference**: Docstrings in the code




## 📄 License

This project is under MIT license. See the LICENSE file for more details.

## 👨‍💻 Author and Opportunities

Developed by Yann POINTUD / yannpointud. 
Passionate about engineering autonomous, robust and efficient AI systems.
Currently available, feel free to contact me to discuss how my skills can help with the success of your projects.


## 🔗 Useful Links

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Python Packaging Guide](https://packaging.python.org/)

---

  ⭐ If you like this project, feel free to give it a star! ⭐