# Phaze-Particles

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

Elementary Particle Modeling Framework using topological soliton approaches in SU(2)/SU(3) gauge theories.

## Overview

Phaze-Particles is a comprehensive framework for modeling elementary particles, starting with proton models based on three torus configurations. The framework implements topological soliton approaches and provides both static and dynamic modeling capabilities.

## Features

- **Proton Modeling**: Static proton model with three torus configurations (120°, clover, cartesian)
- **CLI Interface**: Command-line interface with comprehensive help system
- **Modular Design**: Extensible architecture for additional particle types
- **Scientific Validation**: Built-in validation against experimental data
- **Graceful Shutdown**: Proper signal handling for clean application termination

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/vasilyvz/phaze-particles.git
cd phaze-particles

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Show help
phaze-particles --help

# Show proton command help
phaze-particles proton --help

# Run static proton model with default parameters
phaze-particles proton static

# Run with custom parameters
phaze-particles proton static --grid-size 128 --config-type 120deg --verbose
```

### Available Commands

- `phaze-particles proton static`: Static proton model with three torus configurations
- `phaze-particles proton dynamic`: Dynamic proton model (planned)

## Project Structure

```
phaze-particles/
├── src/phaze_particles/          # Main package
│   ├── cli/                      # Command-line interface
│   ├── models/                   # Particle models
│   ├── utils/                    # Utility functions
│   └── config/                   # Configuration management
├── tests/                        # Test suite
├── docs/                         # Documentation
│   ├── commands/                 # Command documentation
│   ├── reports/                  # Model reports and bug reports
│   └── tech_spec.md              # Technical specifications
└── scripts/                      # Build and utility scripts
```

## Documentation

- **Technical Specification**: [docs/tech_spec.md](docs/tech_spec.md)
- **Project Standards**: [docs/project_standards.md](docs/project_standards.md)
- **Command Documentation**: [docs/commands/](docs/commands/)

## Development

### Code Quality

The project follows strict coding standards:

- **File Size**: Maximum 350-400 lines per file
- **One Class Per File**: Each class in separate file
- **Documentation**: English docstrings for all classes and methods
- **Linting**: Black, flake8, and mypy compliance required

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=phaze_particles

# Run specific test file
pytest tests/test_models/test_proton.py
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the project standards
4. Add tests for new functionality
5. Ensure all tests pass and code is properly formatted
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on SU(2)/SU(3) Skyrme-like topological soliton theory
- Implements three torus configurations for proton modeling
- Designed for elementary particle physics research

## Support

For questions, issues, or contributions, please contact:
- **Email**: vasilyvz@gmail.com
- **Issues**: [GitHub Issues](https://github.com/vasilyvz/phaze-particles/issues)
