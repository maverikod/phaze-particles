# Code Quality and Command Properties Checklist

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

## 1. Code Quality Checklist

### 1.1 File Structure and Organization
- [ ] File size does not exceed 350-400 lines
- [ ] One class per file (except exceptions, enums, errors)
- [ ] Large classes split into facade class and smaller components
- [ ] All imports at file beginning (except lazy loading)
- [ ] File header contains Author and email information

### 1.2 Naming Conventions
- [ ] Python files use snake_case (e.g., `proton_model.py`)
- [ ] Classes use PascalCase (e.g., `ProtonModel`)
- [ ] Functions and methods use snake_case (e.g., `calculate_energy`)
- [ ] Constants use UPPER_SNAKE_CASE (e.g., `PROTON_MASS`)
- [ ] Private methods use leading underscore (e.g., `_validate_parameters`)

### 1.3 Documentation Standards
- [ ] All classes have docstrings in English
- [ ] All methods and functions have docstrings in English
- [ ] Docstrings follow Google/NumPy style
- [ ] Comments in code are in English
- [ ] User communication is in Russian
- [ ] Country references use Ukrainian symbols

### 1.4 Code Quality Tools
- [ ] Code formatted with black (line length 88)
- [ ] No flake8 errors or warnings
- [ ] No mypy type checking errors
- [ ] All tests pass
- [ ] No hardcoded values (use constants or configuration)
- [ ] No `pass` statements in production code (use `NotImplemented` for abstract methods)

## 2. Command Properties Checklist

### 2.1 Command Structure Requirements
- [ ] Command inherits from `BaseCommand`
- [ ] Command implements `add_arguments()` method
- [ ] Command implements `execute()` method
- [ ] Command provides `get_subcommands()` method returning list of subcommands
- [ ] Command provides `get_help()` method returning help text
- [ ] Command supports JSON configuration via `load_config()` method

### 2.2 Command Interface Requirements
- [ ] Command has unique name
- [ ] Command has clear description
- [ ] Command has comprehensive help text
- [ ] Command supports `--help` and `-h` flags
- [ ] Command has proper exit codes (0 for success, non-zero for errors)
- [ ] Command handles graceful shutdown

### 2.3 Parameter Validation
- [ ] All parameters have type hints
- [ ] All parameters have default values where appropriate
- [ ] All parameters have help descriptions
- [ ] Parameter validation is implemented
- [ ] Error messages are clear and helpful

### 2.4 Configuration Support
- [ ] Command supports JSON configuration files
- [ ] Configuration parameters are documented
- [ ] Configuration validation is implemented
- [ ] Default configuration is provided
- [ ] Configuration can override command-line arguments

### 2.5 Results Output Requirements
- [ ] Command saves results to CSV files
- [ ] CSV files follow naming template: `[-short-desc-of-args]-YYYY-MM-DDThh.mm.ss.csv`
- [ ] Results directory structure: `results/command/subcommand/`
- [ ] CSV files use UTF-8 with BOM encoding
- [ ] CSV headers are descriptive and consistent
- [ ] All numerical results are included in CSV
- [ ] Configuration parameters are saved as CSV columns
- [ ] Timestamp is included in ISO 8601 format
- [ ] Missing values are handled consistently

### 2.6 Physical Analysis and Validation Requirements
- [ ] Command performs post-execution physical analysis
- [ ] Calculated values are compared with experimental data
- [ ] Physical validation against known constants is performed
- [ ] Deviation analysis from experimental values is calculated
- [ ] Quality assessment based on physical accuracy is provided
- [ ] Comparison table (calculated vs experimental) is generated
- [ ] Validation status (pass/fail) is determined
- [ ] Recommendations for model improvement are provided
- [ ] All physical parameters meet experimental tolerances

## 3. CLI Integration Checklist

### 3.1 Main CLI Integration
- [ ] Command is registered in main CLI
- [ ] Command appears in help output
- [ ] Command can be executed from main CLI
- [ ] Command subcommands are properly registered
- [ ] Command help is accessible from main CLI

### 3.2 Subcommand Support
- [ ] Command provides list of available subcommands
- [ ] Each subcommand has its own help
- [ ] Subcommands can be executed independently
- [ ] Subcommand parameters are properly parsed
- [ ] Subcommand results are properly returned

### 3.3 Error Handling
- [ ] Command handles invalid arguments gracefully
- [ ] Command provides meaningful error messages
- [ ] Command handles file I/O errors
- [ ] Command handles configuration errors
- [ ] Command handles validation errors

## 4. CSV Output Checklist

### 4.1 File Naming and Structure
- [ ] CSV filename follows template: `[-short-desc-of-args]-YYYY-MM-DDThh.mm.ss.csv`
- [ ] Results directory created: `results/command/subcommand/`
- [ ] Directory structure matches command hierarchy
- [ ] Timestamp format is ISO 8601: YYYY-MM-DDTHH:MM:SS
- [ ] Short description includes key parameters (grid size, config type, etc.)

### 4.2 CSV Content and Format
- [ ] UTF-8 with BOM encoding for Excel compatibility
- [ ] Descriptive column headers in first row
- [ ] Consistent data types per column
- [ ] All input parameters included as columns
- [ ] All calculated results included as columns
- [ ] Timestamp column in ISO 8601 format
- [ ] Model version and metadata included
- [ ] Missing values handled consistently (empty strings or "N/A")

### 4.3 Data Validation
- [ ] CSV file is valid and can be opened in Excel
- [ ] All numerical data is properly formatted
- [ ] No corrupted or malformed data
- [ ] File size is reasonable for data volume
- [ ] CSV can be imported into analysis tools

## 5. Physical Analysis Checklist

### 5.1 Experimental Comparison
- [ ] Electric charge compared with Q = +1.0 (exact)
- [ ] Baryon number compared with B = 1.0 (exact)
- [ ] Mass compared with Mp = 938.272 ± 0.006 MeV
- [ ] Radius compared with rE = 0.841 ± 0.019 fm
- [ ] Magnetic moment compared with μp = 2.793 ± 0.001 μN
- [ ] Energy balance compared with E2/E4 = 50/50 ± 5%

### 5.2 Deviation Analysis
- [ ] Percentage deviations calculated for all parameters
- [ ] Deviations within experimental tolerances
- [ ] Large deviations (>10%) are flagged and explained
- [ ] Systematic errors are identified
- [ ] Random errors are quantified

### 5.3 Quality Assessment
- [ ] Model quality rated (excellent/good/fair/poor)
- [ ] Quality based on physical accuracy criteria
- [ ] Quality assessment is documented and justified
- [ ] Recommendations for improvement are provided
- [ ] Validation status (pass/fail) is determined

### 5.4 Analysis Output
- [ ] Comparison table generated (calculated vs experimental)
- [ ] Deviation analysis included in output
- [ ] Quality assessment reported
- [ ] Validation status clearly indicated
- [ ] Recommendations provided for model improvement

## 6. Testing Checklist

### 6.1 Unit Tests
- [ ] Command has unit tests
- [ ] All public methods are tested
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Configuration loading is tested
- [ ] CSV output generation is tested

### 6.2 Integration Tests
- [ ] Command works with main CLI
- [ ] Command works with subcommands
- [ ] Command works with configuration files
- [ ] Command produces expected output
- [ ] Command handles real-world scenarios
- [ ] CSV files are generated correctly
- [ ] CSV content matches expected results

## 7. Documentation Checklist

### 7.1 Command Documentation
- [ ] Command has dedicated documentation file
- [ ] Documentation includes usage examples
- [ ] Documentation includes parameter descriptions
- [ ] Documentation includes configuration examples
- [ ] Documentation includes troubleshooting guide

### 7.2 API Documentation
- [ ] All public methods are documented
- [ ] Method signatures are documented
- [ ] Return values are documented
- [ ] Exceptions are documented
- [ ] Configuration schema is documented

## 8. Performance Checklist

### 8.1 Execution Performance
- [ ] Command executes within reasonable time
- [ ] Memory usage is reasonable
- [ ] No memory leaks
- [ ] Efficient algorithms are used
- [ ] Large datasets are handled properly

### 8.2 Resource Management
- [ ] Files are properly closed
- [ ] Temporary files are cleaned up
- [ ] Network connections are properly managed
- [ ] System resources are released
- [ ] Graceful shutdown is implemented

## 9. Security Checklist

### 9.1 Input Validation
- [ ] All user inputs are validated
- [ ] File paths are sanitized
- [ ] Configuration files are validated
- [ ] No code injection vulnerabilities
- [ ] No path traversal vulnerabilities

### 9.2 Error Information
- [ ] Error messages don't leak sensitive information
- [ ] Stack traces are not exposed to users
- [ ] Logging doesn't include sensitive data
- [ ] Configuration files are properly secured
- [ ] Temporary files have appropriate permissions

## 10. Maintenance Checklist

### 10.1 Code Maintainability
- [ ] Code is well-structured and readable
- [ ] Dependencies are minimal and justified
- [ ] Code follows DRY principle
- [ ] Functions are focused and single-purpose
- [ ] Complex logic is well-documented

### 10.2 Future Extensibility
- [ ] Command can be easily extended
- [ ] New subcommands can be added easily
- [ ] Configuration schema can be extended
- [ ] API is stable and backward-compatible
- [ ] Migration path is provided for breaking changes

## Usage

This checklist should be used:
1. Before committing code changes
2. During code reviews
3. Before releasing new versions
4. When adding new commands
5. When modifying existing commands

Each item should be checked off as completed, and any issues should be addressed before the code is considered ready for production.
