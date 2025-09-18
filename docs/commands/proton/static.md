# Proton Static Model Command

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

## Overview

The `proton static` command implements a static proton model using topological soliton approaches in SU(2)/SU(3) gauge theories. The model is based on three torus configurations: 120°, clover, and cartesian.

## Usage

```bash
phaze-particles proton static [OPTIONS]
```

## Parameters

### Required Parameters

None - all parameters have default values.

### Optional Parameters

- `--config FILE`: Configuration file path (YAML format)
- `--output DIR`: Output directory for results (default: `proton_static_results`)
- `--grid-size N`: Grid size for numerical calculations (default: 64)
- `--box-size FLOAT`: Box size in femtometers (default: 4.0)
- `--config-type TYPE`: Torus configuration type to run
  - Options: `120deg`, `clover`, `cartesian`, `all`
  - Default: `all`
- `--verbose`: Enable verbose output
- `--save-data`: Save numerical data to files
- `--generate-plots`: Generate visualization plots

## Examples

### Basic Usage

```bash
# Run with default parameters
phaze-particles proton static

# Run with custom grid size
phaze-particles proton static --grid-size 128

# Run specific configuration only
phaze-particles proton static --config-type 120deg
```

### Advanced Usage

```bash
# Run with custom configuration file
phaze-particles proton static --config my_config.yaml

# Run with custom output directory and verbose mode
phaze-particles proton static --output results/proton_analysis --verbose

# Generate plots and save data
phaze-particles proton static --generate-plots --save-data
```

## Output

The command generates the following outputs:

### Console Output

- Model execution status
- Physical parameters (charge, mass, radius, magnetic moment)
- Energy balance information
- Configuration-specific results

### Files (when `--save-data` is used)

- `results.json`: Numerical results in JSON format
- `energy_balance.json`: Energy distribution data
- `parameters.json`: Model parameters used

### Plots (when `--generate-plots` is used)

- `density_distribution.png`: Charge density distribution
- `energy_components.png`: Energy component breakdown
- `radius_analysis.png`: RMS radius analysis

## Technical Details

### Model Implementation

The static proton model implements:

1. **SU(2) Field Construction**: Three toroidal directions (A, B, C)
2. **Energy Density Calculation**: E2, E4, and E6 terms
3. **Physical Parameter Calculation**:
   - Electric charge: Q = +1
   - Baryon number: B = 1
   - Mass: Mp ≈ 938.272 MeV
   - Radius: rE ≈ 0.84 fm
   - Magnetic moment: μp ≈ 2.793 μN

### Validation Criteria

The model validates against:

1. **Energy Balance**: 50-50 balance between E2 and E4 terms
2. **Physical Constants**: Agreement with experimental values
3. **Topological Invariants**: Conservation of baryon number and charge

### Configuration Types

- **120deg**: 120° torus configuration
- **clover**: Clover-shaped torus configuration  
- **cartesian**: Cartesian coordinate torus configuration
- **all**: Run all three configurations and compare

## Exit Codes

- `0`: Success
- `1`: General error
- `130`: Interrupted by user (Ctrl+C)

## Related Commands

- `phaze-particles proton --help`: General proton command help
- `phaze-particles --help`: Main application help

## References

- Technical Specification: `docs/tech_spec.md`
- Project Standards: `docs/project_standards.md`
