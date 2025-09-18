# Project Standards for Phaze-Particles

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

## 1. Project Structure

### 1.1 Directory Organization

```
phaze-particles/
├── src/                          # Source code
│   ├── phaze_particles/          # Main package
│   │   ├── __init__.py
│   │   ├── cli/                  # CLI commands
│   │   │   ├── __init__.py
│   │   │   ├── main.py           # Main CLI entry point
│   │   │   ├── base.py           # Base command class
│   │   │   └── commands/         # Individual command modules
│   │   │       ├── __init__.py
│   │   │       ├── proton.py     # Proton modeling commands
│   │   │       └── neutron.py    # Neutron modeling commands (future)
│   │   ├── models/               # Particle models
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Base model class
│   │   │   ├── proton.py         # Proton model implementation
│   │   │   └── neutron.py        # Neutron model (future)
│   │   ├── utils/                # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── math.py           # Mathematical utilities
│   │   │   └── io.py             # Input/output utilities
│   │   └── config/               # Configuration
│   │       ├── __init__.py
│   │       └── settings.py       # Application settings
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_models/
│   ├── test_cli/
│   └── test_utils/
├── docs/                         # Documentation
│   ├── commands/                 # Command documentation
│   │   ├── proton/               # Proton command docs
│   │   │   ├── static.md         # Static proton model docs
│   │   │   └── dynamic.md        # Dynamic proton model docs
│   │   └── neutron/              # Neutron command docs (future)
│   ├── reports/                  # Model reports and bug reports
│   ├── tech_spec.md              # Technical specifications
│   ├── project_standards.md      # This file
│   └── README.md                 # Project overview
├── scripts/                      # Build and utility scripts
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── setup.py                      # Package setup
├── .gitignore                    # Git ignore rules
├── LICENSE                       # License file
└── README.md                     # Main project README
```

### 1.2 Directory Purposes

- **src/phaze_particles/**: Main application package
- **src/phaze_particles/cli/**: Command-line interface implementation
- **src/phaze_particles/models/**: Particle model implementations
- **src/phaze_particles/utils/**: Shared utility functions
- **src/phaze_particles/config/**: Configuration management
- **tests/**: Unit and integration tests
- **docs/**: All project documentation
- **docs/commands/**: Command-specific documentation (one subdirectory per command)
- **docs/reports/**: Model execution reports and bug reports
- **scripts/**: Build, deployment, and utility scripts

## 2. Coding Standards

### 2.1 File Naming Conventions

- **Python files**: snake_case (e.g., `proton_model.py`, `base_command.py`)
- **Command modules**: snake_case matching command name (e.g., `proton.py`, `neutron.py`)
- **Test files**: `test_` prefix + snake_case (e.g., `test_proton_model.py`)
- **Documentation**: snake_case with `.md` extension (e.g., `tech_spec.md`)

### 2.2 Class and Function Naming

- **Classes**: PascalCase (e.g., `ProtonModel`, `BaseCommand`)
- **Functions and methods**: snake_case (e.g., `calculate_energy`, `run_simulation`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `PROTON_MASS`, `ELECTRON_CHARGE`)
- **Private methods**: Leading underscore (e.g., `_validate_parameters`)

### 2.3 Code Organization Rules

1. **File Size Limit**: Maximum 350-400 lines per file
2. **One Class Per File**: Each class in separate file (except exceptions, enums, errors)
3. **Large Classes**: Split into facade class and smaller components
4. **Imports**: All imports at file beginning (except lazy loading)
5. **Docstrings**: Required for all classes, methods, and functions (English only)

### 2.4 Documentation Standards

- **Language**: English for all code, comments, docstrings, and tests
- **Communication**: Russian for user interaction
- **Country**: Ukraine (use Ukrainian symbols, not Russian)
- **Headers**: Must include Author and email in every file

## 3. CLI Application Standards

### 3.1 Command Structure

- **Main command**: `phaze-particles`
- **Subcommands**: `proton`, `neutron` (future)
- **Sub-subcommands**: `static`, `dynamic` (for each particle type)
- **Help system**: Required for all commands and subcommands

### 3.2 Command Implementation

- **One command per file**: Each command in separate module
- **Base class**: All commands inherit from `BaseCommand`
- **Parameters**: Each command has its own parameter set
- **Graceful shutdown**: Signal handling for clean exit
- **Error handling**: Comprehensive error reporting

### 3.3 Help System Requirements

- **Command help**: `--help` or `-h` for all commands
- **Parameter help**: Detailed description for each parameter
- **Examples**: Usage examples in help text
- **Exit codes**: Standard exit codes for different scenarios

## 4. Model Implementation Standards

### 4.1 Model Structure

- **Base model class**: Common interface for all particle models
- **Configuration**: Separate configuration classes for each model
- **Validation**: Parameter validation before execution
- **Results**: Standardized result format

### 4.2 Reporting Standards

- **Model reports**: Generated in `docs/reports/` directory
- **Bug reports**: Also stored in `docs/reports/` directory
- **Format**: Markdown with tables, graphs, and analysis
- **Naming**: `YYYY-MM-DD_model-name_report.md`

## 5. Development Workflow

### 5.1 Code Quality

- **Linting**: Run `black`, `flake8`, and `mypy` after production code
- **Fix all errors**: Must resolve all linting errors
- **Testing**: Unit tests for all new functionality

### 5.2 Version Control

- **Commits**: After each major logical change or plan step
- **Push**: Only on user request
- **Branching**: Feature branches for major changes

### 5.3 Environment Management

- **Virtual environment**: Use `.venv` in project root
- **Dependencies**: Never use `--break-system-packages`
- **Activation**: Always activate environment before package installation

## 6. Command Documentation Structure

### 6.1 Command Documentation

Each command has its own subdirectory in `docs/commands/`:
- **proton/**: Proton-related commands
  - `static.md`: Static proton model documentation
  - `dynamic.md`: Dynamic proton model documentation
- **neutron/**: Neutron-related commands (future)
  - `static.md`: Static neutron model documentation
  - `dynamic.md`: Dynamic neutron model documentation

### 6.2 Documentation Content

Each command documentation should include:
- **Purpose**: What the command does
- **Parameters**: All available parameters with descriptions
- **Examples**: Usage examples
- **Output**: Expected output format
- **Technical details**: Implementation specifics
- **References**: Related technical specifications

## 7. Project Naming and Branding

- **Project name**: `phaze-particles`
- **Package name**: `phaze_particles`
- **CLI command**: `phaze-particles`
- **Repository**: `phaze-particles`
- **Country**: Ukraine (use Ukrainian symbols and references)
