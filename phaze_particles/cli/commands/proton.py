#!/usr/bin/env python3
"""
Proton modeling commands for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any

from ..base import BaseCommand
from ...models.proton_integrated import (
    ProtonModel,
    ModelConfig,
    ModelStatus,
)
from ...utils.cuda import get_cuda_status
from ...utils.progress import create_performance_monitor
from ...utils.io import save_results
from ...utils.optimization import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizedProtonModel,
    OptimizationSuite,
)


class ProtonCommand(BaseCommand):
    """
    Proton modeling command with subcommands for different model types.
    """

    def __init__(self) -> None:
        """Initialize proton command."""
        super().__init__(
            name="proton",
            description="Proton modeling using topological soliton approaches",
        )
        self.subcommands = {
            "static": ProtonStaticCommand(),
            "optimize": ProtonOptimizeCommand(),
            "tune": ProtonTuneCommand(),
            "solve": ProtonSolveCommand(),
            "phase": ProtonPhaseEnvironmentCommand(),
            "phase-tails": ProtonPhaseTailsCommand(),
            # Future subcommands will be added here
            # 'dynamic': ProtonDynamicCommand(),
        }

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add proton command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        subparsers = parser.add_subparsers(
            dest="subcommand",
            help="Available proton model types",
            metavar="MODEL_TYPE",
        )

        # Add subcommand parsers
        for subcmd_name, subcmd_instance in self.subcommands.items():
            subcmd_parser = subparsers.add_parser(
                subcmd_name,
                help=subcmd_instance.description,
                description=subcmd_instance.description,
            )
            subcmd_instance.add_arguments(subcmd_parser)

    def get_subcommands(self) -> List[str]:
        """
        Get list of available subcommands.

        Returns:
            List of subcommand names
        """
        return list(self.subcommands.keys())

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        help_text = f"{self.description}\n\n"
        help_text += "Available subcommands:\n"
        for subcmd_name, subcmd_instance in self.subcommands.items():
            help_text += f"  {subcmd_name}: {subcmd_instance.description}\n"
        return help_text

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute proton command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        if not args.subcommand:
            print(
                "Error: No subcommand specified. " "Use --help for available options."
            )
            return 1

        if args.subcommand not in self.subcommands:
            print(f"Error: Unknown subcommand '{args.subcommand}'")
            return 1

        # Execute the subcommand
        subcmd_instance = self.subcommands[args.subcommand]
        return subcmd_instance.execute(args)


class ProtonStaticCommand(BaseCommand):
    """
    Static proton model command based on three torus configurations.
    """

    def __init__(self) -> None:
        """Initialize static proton command."""
        super().__init__(
            name="proton static",
            description=(
                "Static proton model with three torus configurations "
                "(120°, clover, cartesian)"
            ),
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add static proton command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        parser.add_argument(
            "--config", type=str, help="Configuration file path (JSON format)"
        )

        parser.add_argument(
            "--output",
            type=str,
            default="proton_static_results",
            help=("Output directory for results " "(default: proton_static_results)"),
        )

        parser.add_argument(
            "--grid-size",
            type=int,
            default=64,
            help="Grid size for numerical calculations (default: 64)",
        )

        parser.add_argument(
            "--box-size",
            type=float,
            default=4.0,
            help="Box size in femtometers (default: 4.0)",
        )

        parser.add_argument(
            "--config-type",
            type=str,
            choices=["120deg", "clover", "cartesian", "all"],
            default="all",
            help="Torus configuration type to run (default: all)",
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "--save-data", action="store_true", help="Save numerical data to files"
        )

        parser.add_argument(
            "--generate-plots", action="store_true", help="Generate visualization plots"
        )

        # CUDA options (CLI > config > env)
        parser.add_argument(
            "--cuda-mem-target",
            type=float,
            default=None,
            help=(
                "Target fraction of total VRAM to reserve in memory pool (e.g., 0.8). "
                "Env: PHAZE_CUDA_MEM_TARGET"
            ),
        )
        parser.add_argument(
            "--cuda-free-cap-frac",
            type=float,
            default=None,
            help=(
                "Fraction of currently free VRAM allowed for reservation (e.g., 0.75). "
                "Env: PHAZE_CUDA_FREE_CAP_FRAC"
            ),
        )
        parser.add_argument(
            "--cuda-safety-mb",
            type=int,
            default=None,
            help=(
                "Safety margin in MB to avoid OOM during reservation (e.g., 128). "
                "Env: PHAZE_CUDA_SAFETY_MB"
            ),
        )
        parser.add_argument(
            "--cuda-device-id",
            type=int,
            default=None,
            help="CUDA device id to use (default: 0). Env: PHAZE_CUDA_DEVICE_ID",
        )

    def get_subcommands(self) -> List[str]:
        """
        Get list of available subcommands.

        Returns:
            List of subcommand names (empty for leaf commands)
        """
        return []

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        return (
            f"{self.description}\n\n"
            f"This command implements the static proton model "
            f"with three torus configurations."
        )

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate grid size
        grid_size = self.get_config("grid_size", 64)
        if not isinstance(grid_size, int) or grid_size < 16:
            return False

        # Validate box size
        box_size = self.get_config("box_size", 4.0)
        if not isinstance(box_size, (int, float)) or box_size <= 0:
            return False

        # Validate config type
        config_type = self.get_config("config_type", "all")
        if config_type not in ["120deg", "clover", "cartesian", "all"]:
            return False

        return True

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute static proton model.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        try:
            # Load configuration if provided
            if args.config:
                try:
                    model_config = ModelConfig.from_file(args.config)
                    print(f"Configuration loaded from: {args.config}")
                except Exception as e:
                    print(f"Error loading configuration file: {e}", file=sys.stderr)
                    return 1
            else:
                # Use command line arguments to create config
                model_config = ModelConfig(
                    grid_size=args.grid_size,
                    box_size=args.box_size,
                    torus_config=args.config_type,
                    validation_enabled=True,
                    save_reports=True,
                    output_dir=args.output,
                )

            # Override with command line arguments if provided
            grid_size = (
                args.grid_size if args.grid_size != 64 else model_config.grid_size
            )
            box_size = args.box_size if args.box_size != 4.0 else model_config.box_size
            config_type = (
                args.config_type
                if args.config_type != "all"
                else model_config.torus_config
            )
            output_dir = (
                args.output
                if args.output != "proton_static_results"
                else model_config.output_dir
            )
            verbose = args.verbose
            save_data = args.save_data
            generate_plots = args.generate_plots

            # Display CUDA status
            print("Running integrated proton model...")
            print(get_cuda_status())
            print(f"Configuration: {config_type}")
            print(f"Grid size: {grid_size}")
            print(f"Box size: {box_size} fm")
            print(f"Output directory: {output_dir}")

            if verbose:
                print("Verbose mode enabled")

            if save_data:
                print("Data saving enabled")

            if generate_plots:
                print("Plot generation enabled")

            # Update model config with final values
            model_config.grid_size = grid_size
            model_config.box_size = box_size
            model_config.torus_config = config_type
            model_config.output_dir = output_dir

            # Create and run model
            with create_performance_monitor("Proton Static Model") as monitor:
                # Create model
                model = ProtonModel(model_config)

                # Display model CUDA status
                print(f"Model CUDA Status: {model.get_cuda_status()}")

                # Apply CUDA configuration (priority: CLI > config > env > defaults)
                try:
                    from ...utils.cuda import get_cuda_manager

                    cm = get_cuda_manager()
                    # Device selection
                    cuda_device_id = None
                    # from CLI
                    if args.cuda_device_id is not None:
                        cuda_device_id = args.cuda_device_id
                    # from config
                    elif getattr(model_config, "cuda_device_id", None) is not None:
                        cuda_device_id = getattr(model_config, "cuda_device_id")
                    # from env
                    elif os.environ.get("PHAZE_CUDA_DEVICE_ID") is not None:
                        try:
                            cuda_device_id = int(os.environ.get("PHAZE_CUDA_DEVICE_ID"))
                        except Exception:
                            cuda_device_id = None
                    if cuda_device_id is not None:
                        cm.set_device(cuda_device_id)

                    # Memory pool tuning
                    def _pick_float(cli_val, cfg_val, env_key):
                        if cli_val is not None:
                            return float(cli_val)
                        if cfg_val is not None:
                            return float(cfg_val)
                        if os.environ.get(env_key) is not None:
                            try:
                                return float(os.environ.get(env_key))
                            except Exception:
                                return None
                        return None

                    def _pick_int(cli_val, cfg_val, env_key):
                        if cli_val is not None:
                            return int(cli_val)
                        if cfg_val is not None:
                            return int(cfg_val)
                        if os.environ.get(env_key) is not None:
                            try:
                                return int(os.environ.get(env_key))
                            except Exception:
                                return None
                        return None

                    cfg_cuda = getattr(model_config, "cuda", {}) or {}
                    tf = _pick_float(args.cuda_mem_target, cfg_cuda.get("mem_target"), "PHAZE_CUDA_MEM_TARGET")
                    fcf = _pick_float(
                        args.cuda_free_cap_frac,
                        cfg_cuda.get("free_cap_frac"),
                        "PHAZE_CUDA_FREE_CAP_FRAC",
                    )
                    sm = _pick_int(args.cuda_safety_mb, cfg_cuda.get("safety_mb"), "PHAZE_CUDA_SAFETY_MB")

                    if tf is not None or fcf is not None or sm is not None:
                        cm.reconfigure_memory_pools(
                            target_fraction=tf, free_cap_frac=fcf, safety_mb=sm
                        )
                except Exception:
                    pass

                # Run model with progress tracking
                print("Starting proton model calculation...")
                results = model.run()

                # Add performance metrics
                monitor.add_metric("grid_size", grid_size)
                monitor.add_metric("box_size", box_size)
                monitor.add_metric("config_type", config_type)
                monitor.add_metric("iterations", results.iterations)
                monitor.add_metric("execution_time", results.execution_time)

            # Check if model was successful
            if results.status == ModelStatus.FAILED:
                print(
                    f"Model execution failed: {results.error_message}",
                    file=sys.stderr,
                )
                return 1

            # Print results
            print("\n=== PROTON MODEL RESULTS ===")
            print(f"Status: {results.status.value}")
            print(f"Execution time: {results.execution_time:.2f} seconds")
            print(f"Iterations: {results.iterations}")
            print(f"Converged: {results.converged}")
            print()
            print("Physical Parameters:")
            print(f"  Proton mass: {results.proton_mass:.3f} MeV")
            print(f"  Charge radius: {results.charge_radius:.3f} fm")
            print(f"  Magnetic moment: {results.magnetic_moment:.3f} μN")
            print(f"  Electric charge: {results.electric_charge:.3f} e")
            print(f"  Baryon number: {results.baryon_number:.3f}")
            print(f"  Energy balance: {results.energy_balance:.3f}")
            print(f"  Total energy: {results.total_energy:.3f} MeV")

            # Print detailed energy report
            if hasattr(model, 'get_energy_report'):
                print("\n" + "="*60)
                print("DETAILED ENERGY ANALYSIS")
                print("="*60)
                print(model.get_energy_report())
            
            # Print phase environment report
            if hasattr(model, 'get_phase_environment_report'):
                print("\n" + "="*60)
                print("PHASE ENVIRONMENT INTEGRATION REPORT")
                print("="*60)
                print(model.get_phase_environment_report())
            
            # Print phase tail report
            if hasattr(model, 'get_phase_tail_report'):
                print("\n" + "="*60)
                print("PHASE TAIL ANALYSIS REPORT")
                print("="*60)
                print(model.get_phase_tail_report())

            # Print validation results
            if results.validation_status:
                print(f"\nValidation Status: {results.validation_status}")
                if results.validation_score is not None:
                    print(f"Validation Score: {results.validation_score:.3f}")

            # Save results to CSV (always save by default)
            self._save_results_to_csv(
                results, output_dir, config_type, grid_size, box_size
            )

            # Save model results
            results_path = os.path.join(output_dir, "model_results.json")
            os.makedirs(output_dir, exist_ok=True)
            results.save_to_file(results_path)
            print(f"\nResults saved to: {results_path}")

            print("\nStatic proton model execution completed successfully.")
            return 0

        except Exception as e:
            print(f"Error executing static proton model: {e}", file=sys.stderr)
            return 1

    def _save_results_to_csv(
        self,
        results,
        output_dir: str,
        config_type: str,
        grid_size: int,
        box_size: float,
    ) -> None:
        """
        Save results to CSV file.

        Args:
            results: Model results
            output_dir: Output directory
            config_type: Configuration type
            grid_size: Grid size
            box_size: Box size
        """
        try:
            # Create results directory
            results_dir = os.path.join(output_dir, "proton", "static")
            os.makedirs(results_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
            short_desc = f"grid{grid_size}-box{box_size}-{config_type}"
            filename = f"-{short_desc}-{timestamp}.csv"
            csv_path = os.path.join(results_dir, filename)

            # Prepare data for CSV
            csv_data = {
                "timestamp": datetime.now().isoformat(),
                "grid_size": grid_size,
                "box_size": box_size,
                "config_type": config_type,
                "status": results.status.value,
                "execution_time": results.execution_time,
                "iterations": results.iterations,
                "converged": results.converged,
                "proton_mass": results.proton_mass,
                "charge_radius": results.charge_radius,
                "magnetic_moment": results.magnetic_moment,
                "electric_charge": results.electric_charge,
                "baryon_number": results.baryon_number,
                "energy_balance": results.energy_balance,
                "total_energy": results.total_energy,
                "validation_status": results.validation_status or "N/A",
                "validation_score": results.validation_score or "N/A",
            }

            # Save CSV
            save_results(csv_data, csv_path, format="csv")
            print(f"CSV results saved to: {csv_path}")

        except Exception as e:
            print(f"Warning: Failed to save CSV results: {e}", file=sys.stderr)


class ProtonOptimizeCommand(BaseCommand):
    """
    Proton model optimization command.
    """

    def __init__(self) -> None:
        """Initialize proton optimize command."""
        super().__init__(
            name="optimize",
            description=(
                "Optimize proton model performance with CUDA and advanced algorithms"
            ),
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add optimization command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        # Basic model parameters
        parser.add_argument(
            "--grid-size",
            type=int,
            default=64,
            help="Grid size for the model (default: 64)",
        )
        parser.add_argument(
            "--box-size",
            type=float,
            default=4.0,
            help="Box size in fm (default: 4.0)",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=1000,
            help="Maximum number of iterations (default: 1000)",
        )

        # Optimization parameters
        parser.add_argument(
            "--optimization-level",
            type=str,
            choices=["none", "basic", "advanced", "maximum"],
            default="advanced",
            help="Optimization level (default: advanced)",
        )
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            help="Disable CUDA optimization",
        )
        parser.add_argument(
            "--no-memory-opt",
            action="store_true",
            help="Disable memory optimization",
        )
        parser.add_argument(
            "--no-algorithm-opt",
            action="store_true",
            help="Disable algorithm optimization",
        )
        parser.add_argument(
            "--no-adaptive",
            action="store_true",
            help="Disable adaptive parameters",
        )
        parser.add_argument(
            "--no-profiling",
            action="store_true",
            help="Disable performance profiling",
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="Disable result caching",
        )
        parser.add_argument(
            "--no-parallel",
            action="store_true",
            help="Disable parallel processing",
        )

        # Benchmark options
        parser.add_argument(
            "--benchmark",
            action="store_true",
            help="Run optimization benchmark comparison",
        )
        parser.add_argument(
            "--benchmark-grids",
            type=int,
            nargs="+",
            default=[32, 64, 128],
            help="Grid sizes for benchmark (default: 32 64 128)",
        )

        # Output options
        parser.add_argument(
            "--output-dir",
            type=str,
            default="results/proton/optimize",
            help="Output directory for results (default: results/proton/optimize)",
        )
        parser.add_argument(
            "--save-report",
            action="store_true",
            help="Save optimization report to file",
        )

        # Configuration file
        parser.add_argument(
            "--config",
            type=str,
            help="JSON configuration file path",
        )

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        help_text = f"{self.description}\n\n"
        help_text += "This command provides advanced optimization features:\n"
        help_text += "  - CUDA acceleration for GPU computations\n"
        help_text += "  - Memory optimization and caching\n"
        help_text += "  - Algorithm optimization\n"
        help_text += "  - Adaptive parameter tuning\n"
        help_text += "  - Performance profiling and benchmarking\n\n"
        help_text += "Examples:\n"
        help_text += "  # Run with advanced optimization\n"
        help_text += (
            "  phaze-particles proton optimize --optimization-level advanced\n\n"
        )
        help_text += "  # Run benchmark comparison\n"
        help_text += "  phaze-particles proton optimize --benchmark\n\n"
        help_text += "  # Disable CUDA and run CPU-only optimization\n"
        help_text += "  phaze-particles proton optimize --no-cuda\n"
        return help_text

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config: Dict[str, Any] = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}", file=sys.stderr)
            return {}

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute optimization command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        try:
            # Load configuration if provided
            config_data: Dict[str, Any] = {}
            if args.config:
                config_data = self.load_config(args.config)

            # Create model configuration
            model_config = ModelConfig(
                grid_size=config_data.get("grid_size", args.grid_size),
                box_size=config_data.get("box_size", args.box_size),
                max_iterations=config_data.get("max_iterations", args.max_iterations),
                validation_enabled=True,
            )

            # Create optimization configuration
            optimization_level = OptimizationLevel(args.optimization_level)
            optimization_config = OptimizationConfig(
                use_cuda=not args.no_cuda,
                optimization_level=optimization_level,
                memory_optimization=not args.no_memory_opt,
                algorithm_optimization=not args.no_algorithm_opt,
                adaptive_parameters=not args.no_adaptive,
                profiling_enabled=not args.no_profiling,
                cache_enabled=not args.no_cache,
                parallel_processing=not args.no_parallel,
            )

            # Display CUDA status
            print(get_cuda_status())
            print()

            if args.benchmark:
                # Run benchmark comparison
                print("Running optimization benchmark...")
                optimization_suite = OptimizationSuite()
                comparison_results = optimization_suite.run_optimization_comparison(
                    model_config
                )

                # Generate and display report
                report = optimization_suite.generate_comparison_report(
                    comparison_results
                )
                print(report)

                # Save report if requested
                if args.save_report:
                    report_path = os.path.join(
                        args.output_dir, "optimization_benchmark_report.txt"
                    )
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(report)
                    print(f"Benchmark report saved to: {report_path}")

                return 0

            else:
                # Run single optimization
                print(
                    f"Running proton model optimization "
                    f"(level: {args.optimization_level})..."
                )

                # Create optimized model
                optimized_model = OptimizedProtonModel(
                    model_config, optimization_config
                )

                # Run optimization
                results = optimized_model.run_optimized()

                # Display results
                print("\nOptimization Results:")
                print("-" * 50)
                metrics = results["performance_metrics"]
                print(f"Execution time: {metrics.execution_time:.2f} seconds")
                print(f"Memory usage: {metrics.memory_usage:.2f} MB")
                print(f"GPU utilization: {metrics.gpu_utilization:.2f} MB")
                print(f"CPU utilization: {metrics.cpu_utilization:.2f}%")
                print(f"Iterations: {metrics.iterations}")
                print(f"Convergence rate: {metrics.convergence_rate:.6f}")
                print(f"Throughput: {metrics.throughput:.2e} ops/sec")

                # Get and display optimization report
                report = optimized_model.get_optimization_report()
                print(f"\n{report}")

                # Save report if requested
                if args.save_report:
                    os.makedirs(args.output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
                    report_path = os.path.join(
                        args.output_dir, f"optimization_report_{timestamp}.txt"
                    )
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(report)
                    print(f"Optimization report saved to: {report_path}")

                return 0

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            return 1
        except Exception as e:
            print(f"Error during optimization: {e}", file=sys.stderr)
            return 1


class ProtonTuneCommand(BaseCommand):
    """
    Tune Skyrme constants for optimal virial balance and energy balance.
    """

    def __init__(self) -> None:
        """Initialize tune command."""
        super().__init__(
            name="tune",
            description="Tune Skyrme constants for virial balance and energy balance",
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add tune command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        # Basic model parameters
        parser.add_argument(
            "--grid-size",
            type=int,
            default=32,
            help="Grid size for calculation (default: 32)",
        )
        parser.add_argument(
            "--box-size",
            type=float,
            default=2.0,
            help="Box size in fm (default: 2.0)",
        )
        parser.add_argument(
            "--config-type",
            type=str,
            default="120deg",
            choices=["120deg", "clover", "cartesian"],
            help="Torus configuration type (default: 120deg)",
        )

        # Optimization targets
        parser.add_argument(
            "--target-e2-ratio",
            type=float,
            default=0.5,
            help="Target E₂/E_total ratio (default: 0.5)",
        )
        parser.add_argument(
            "--target-e4-ratio",
            type=float,
            default=0.5,
            help="Target E₄/E_total ratio (default: 0.5)",
        )
        parser.add_argument(
            "--target-virial-residual",
            type=float,
            default=0.05,
            help="Target virial residual (default: 0.05)",
        )

        # Optimization parameters
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=100,
            help="Maximum optimization iterations (default: 100)",
        )

        # Initial constants
        parser.add_argument(
            "--initial-c2",
            type=float,
            default=1.0,
            help="Initial c₂ constant (default: 1.0)",
        )
        parser.add_argument(
            "--initial-c4",
            type=float,
            default=1.0,
            help="Initial c₄ constant (default: 1.0)",
        )
        parser.add_argument(
            "--initial-c6",
            type=float,
            default=1.0,
            help="Initial c₆ constant (default: 1.0)",
        )

        # Output options
        parser.add_argument(
            "--output",
            type=str,
            default="proton_tune_results",
            help="Output directory (default: proton_tune_results)",
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Configuration file path (JSON format)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute tune command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        try:
            # Load configuration if provided
            if args.config:
                try:
                    model_config = ModelConfig.from_file(args.config)
                    print(f"Configuration loaded from: {args.config}")
                except Exception as e:
                    print(f"Error loading configuration file: {e}", file=sys.stderr)
                    return 1
            else:
                # Use command line arguments to create config
                model_config = ModelConfig(
                    grid_size=args.grid_size,
                    box_size=args.box_size,
                    torus_config=args.config_type,
                    c2=args.initial_c2,
                    c4=args.initial_c4,
                    c6=args.initial_c6,
                    validation_enabled=True,
                    save_reports=True,
                    output_dir=args.output,
                )

            # Display parameters
            print("Running Skyrme constants tuning...")
            print(f"Configuration: {args.config_type}")
            print(f"Grid size: {args.grid_size}")
            print(f"Box size: {args.box_size} fm")
            print(f"Output directory: {args.output}")
            print(f"Initial constants: c₂={args.initial_c2}, c₄={args.initial_c4}, c₆={args.initial_c6}")
            print(f"Target ratios: E₂/E₄ = {args.target_e2_ratio:.1%}/{args.target_e4_ratio:.1%}")
            print(f"Target virial residual: < {args.target_virial_residual:.1%}")

            # Create and run model
            with create_performance_monitor("Proton Constants Tuning") as monitor:
                # Create model
                model = ProtonModel(model_config)

                # Build geometry and fields
                if not model.create_geometry():
                    print("Failed to create geometry", file=sys.stderr)
                    return 1

                if not model.build_fields():
                    print("Failed to build fields", file=sys.stderr)
                    return 1

                if not model.calculate_energy():
                    print("Failed to calculate energy", file=sys.stderr)
                    return 1

                # Show initial energy analysis
                print("\n" + "="*60)
                print("INITIAL ENERGY ANALYSIS")
                print("="*60)
                print(model.get_energy_report())

                # Optimize constants
                success = model.optimize_skyrme_constants(
                    target_e2_ratio=args.target_e2_ratio,
                    target_e4_ratio=args.target_e4_ratio,
                    target_virial_residual=args.target_virial_residual,
                    max_iterations=args.max_iterations,
                    verbose=args.verbose
                )

                if not success:
                    print("Failed to optimize constants", file=sys.stderr)
                    return 1

                # Show optimization report
                print("\n" + "="*60)
                print("OPTIMIZATION REPORT")
                print("="*60)
                print(model.get_optimization_report())

                # Recalculate physics with optimized constants
                if not model.calculate_physics():
                    print("Failed to recalculate physics", file=sys.stderr)
                    return 1

                # Create results by running the model
                results = model.run()

                # Display results
                print("\n" + "="*60)
                print("TUNED PROTON MODEL RESULTS")
                print("="*60)
                print(f"Status: {results.status}")
                print(f"Execution time: {results.execution_time:.2f} seconds")
                print(f"Optimization iterations: {model.optimization_result.iterations}")
                print(f"Converged: {model.optimization_result.converged}")

                print(f"\nOptimized Constants:")
                print(f"  c₂ = {model.optimization_result.c2:.6f}")
                print(f"  c₄ = {model.optimization_result.c4:.6f}")
                print(f"  c₆ = {model.optimization_result.c6:.6f}")

                print(f"\nPhysical Parameters:")
                print(f"  Proton mass: {results.proton_mass:.3f} MeV")
                print(f"  Charge radius: {results.charge_radius:.3f} fm")
                print(f"  Magnetic moment: {results.magnetic_moment:.3f} μN")
                print(f"  Electric charge: {results.electric_charge:.3f} e")
                print(f"  Baryon number: {results.baryon_number:.3f}")
                print(f"  Energy balance: {results.energy_balance:.3f}")
                print(f"  Total energy: {results.total_energy:.3f} MeV")

                # Show final energy analysis
                print("\n" + "="*60)
                print("FINAL ENERGY ANALYSIS")
                print("="*60)
                print(model.get_energy_report())

                # Save results
                os.makedirs(args.output, exist_ok=True)
                
                # Save optimized constants
                constants_path = os.path.join(args.output, "optimized_constants.json")
                constants_data = {
                    "c2": float(model.optimization_result.c2),
                    "c4": float(model.optimization_result.c4),
                    "c6": float(model.optimization_result.c6),
                    "optimization_targets": {
                        "target_e2_ratio": args.target_e2_ratio,
                        "target_e4_ratio": args.target_e4_ratio,
                        "target_virial_residual": args.target_virial_residual
                    },
                    "achieved_values": {
                        "e2_ratio": float(model.optimization_result.e2_ratio),
                        "e4_ratio": float(model.optimization_result.e4_ratio),
                        "virial_residual": float(model.optimization_result.virial_residual)
                    },
                    "converged": bool(model.optimization_result.converged),
                    "iterations": model.optimization_result.iterations
                }
                
                with open(constants_path, 'w') as f:
                    json.dump(constants_data, f, indent=2)
                print(f"\nOptimized constants saved to: {constants_path}")

                # Save model results
                results_path = os.path.join(args.output, "model_results.json")
                results.save_to_file(results_path)
                print(f"Model results saved to: {results_path}")

                print("\nSkyrme constants tuning completed successfully.")
                return 0

        except Exception as e:
            print(f"Error executing tune command: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1


class ProtonSolveCommand(BaseCommand):
    """Proton solve command using universal solver."""

    def __init__(self) -> None:
        """Initialize proton solve command."""
        super().__init__(
            name="solve",
            description="Solve proton model using universal solver with advanced optimization"
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add solve command arguments."""
        parser.add_argument(
            "--grid-size", type=int, default=32,
            help="Grid size (default: 32)"
        )
        parser.add_argument(
            "--box-size", type=float, default=4.0,
            help="Box size in fm (default: 4.0)"
        )
        parser.add_argument(
            "--config-type", type=str, default="120deg",
            choices=["120deg", "clover", "cartesian"],
            help="Configuration type (default: 120deg)"
        )
        parser.add_argument(
            "--c2", type=float, default=1.0,
            help="Skyrme constant c2 (default: 1.0)"
        )
        parser.add_argument(
            "--c4", type=float, default=1.0,
            help="Skyrme constant c4 (default: 1.0)"
        )
        parser.add_argument(
            "--c6", type=float, default=1.0,
            help="Skyrme constant c6 (default: 1.0)"
        )
        parser.add_argument(
            "--F-pi", type=float, default=186.0,
            help="Pion decay constant in MeV (default: 186.0)"
        )
        parser.add_argument(
            "--e", type=float, default=5.45,
            help="Dimensionless Skyrme constant (default: 5.45)"
        )
        parser.add_argument(
            "--target-mass", type=float,
            help="Target mass in MeV (e.g., 938.272 for proton)"
        )
        parser.add_argument(
            "--target-radius", type=float,
            help="Target radius in fm (e.g., 0.841 for proton)"
        )
        parser.add_argument(
            "--target-magnetic-moment", type=float,
            help="Target magnetic moment in μN (e.g., 2.793 for proton)"
        )
        parser.add_argument(
            "--target-bands", type=int,
            help="Target number of energy bands (e.g., 11)"
        )
        parser.add_argument(
            "--optimization-strategy", type=str, default="auto",
            choices=["auto", "energy_balance", "physical_params", "quantization"],
            help="Optimization strategy (default: auto)"
        )
        parser.add_argument(
            "--output", type=str,
            help="Output directory for results"
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Verbose output"
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the proton solve command."""
        try:
            print("PROTON MODEL WITH UNIVERSAL SOLVER")
            print("=" * 50)
            
            # Display CUDA status
            cuda_status = get_cuda_status()
            print(f"CUDA Status: {cuda_status}")
            print()
            
            # Create model configuration
            config = ModelConfig(
                grid_size=args.grid_size,
                box_size=args.box_size,
                c2=args.c2,
                c4=args.c4,
                c6=args.c6,
                F_pi=args.F_pi,
                e=args.e
            )
            config.config_type = args.config_type
            
            if args.verbose:
                print(f"Configuration:")
                print(f"  Grid size: {config.grid_size}")
                print(f"  Box size: {config.box_size} fm")
                print(f"  Config type: {config.config_type}")
                print(f"  Constants: c2={config.c2}, c4={config.c4}, c6={config.c6}")
                print(f"  Physical: F_π={config.F_pi} MeV, e={config.e}")
                if args.target_mass:
                    print(f"  Target mass: {args.target_mass} MeV")
                if args.target_radius:
                    print(f"  Target radius: {args.target_radius} fm")
                if args.target_magnetic_moment:
                    print(f"  Target magnetic moment: {args.target_magnetic_moment} μN")
                if args.target_bands:
                    print(f"  Target bands: {args.target_bands}")
                print(f"  Strategy: {args.optimization_strategy}")
                print()
            
            # Create and initialize model
            model = ProtonModel(config)
            model.create_geometry()
            model.build_fields()
            model.calculate_energy()
            model.calculate_physics()
            
            # Run universal solver
            print("Running universal solver optimization...")
            solver_result = model.solve_with_universal_solver(
                target_mass=args.target_mass,
                target_radius=args.target_radius,
                target_magnetic_moment=args.target_magnetic_moment,
                target_bands=args.target_bands,
                optimization_strategy=args.optimization_strategy,
                verbose=args.verbose
            )
            
            if not solver_result.success:
                print(f"❌ Universal solver failed: {solver_result.convergence_info}")
                return 1
            
            # Display results
            print("\n" + "="*60)
            print("UNIVERSAL SOLVER RESULTS")
            print("="*60)
            
            print("\nPHYSICAL PARAMETERS:")
            print(f"  Mass: {solver_result.physical_parameters['mass']:.1f} MeV")
            print(f"  Charge radius: {solver_result.physical_parameters['charge_radius']:.3f} fm")
            print(f"  Magnetic moment: {solver_result.physical_parameters['magnetic_moment']:.3f} μN")
            print(f"  Electric charge: {solver_result.physical_parameters['electric_charge']:.3f}")
            print(f"  Baryon number: {solver_result.physical_parameters['baryon_number']:.3f}")
            
            print("\nENERGY ANALYSIS:")
            print(f"  E₂: {solver_result.energy_analysis['e2']:.1f} MeV")
            print(f"  E₄: {solver_result.energy_analysis['e4']:.1f} MeV")
            print(f"  E₆: {solver_result.energy_analysis['e6']:.1f} MeV")
            print(f"  Total: {solver_result.energy_analysis['total']:.1f} MeV")
            print(f"  Ratios: E₂/E₄ = {solver_result.energy_analysis['e2_ratio']:.2f}/{solver_result.energy_analysis['e4_ratio']:.2f}")
            
            print("\nMODE ANALYSIS:")
            print(f"  Total modes: {solver_result.mode_analysis['total_modes']}")
            print(f"  Energy bands: {solver_result.mode_analysis['energy_bands']}")
            print(f"  Core radius: {solver_result.mode_analysis['core_radius']:.3f} fm")
            print(f"  Quantization parameter: {solver_result.mode_analysis['quantization_parameter']:.3f}")
            
            print("\nTOPOLOGICAL ANALYSIS:")
            print(f"  Geometric radius: {solver_result.topological_analysis['geometric_radius']:.3f} fm")
            print(f"  Phase radius: {solver_result.topological_analysis['phase_radius']:.3f} fm")
            print(f"  Effective radius: {solver_result.topological_analysis['effective_radius']:.3f} fm")
            print(f"  Topological charge: {solver_result.topological_analysis['topological_charge']:.3f}")
            print(f"  Phase transitions: {solver_result.topological_analysis['phase_transitions']}")
            
            print("\nINTERFERENCE ANALYSIS:")
            print(f"  Fluctuation energy: {solver_result.interference_analysis['fluctuation_energy']:.1f} MeV")
            print(f"  Background field strength: {solver_result.interference_analysis['background_field_strength']:.3f}")
            print(f"  Constructive regions: {solver_result.interference_analysis['constructive_regions']}")
            print(f"  Destructive regions: {solver_result.interference_analysis['destructive_regions']}")
            print(f"  Interference strength: {solver_result.interference_analysis['interference_strength']:.3f}")
            print(f"  Fluctuation amplitude: {solver_result.interference_analysis['fluctuation_amplitude']:.3f}")
            
            print("\nOPTIMIZED CONSTANTS:")
            print(f"  c₂: {solver_result.optimized_constants['c2']:.6f}")
            print(f"  c₄: {solver_result.optimized_constants['c4']:.6f}")
            print(f"  c₆: {solver_result.optimized_constants['c6']:.6f}")
            print(f"  F_π: {solver_result.optimized_constants['F_pi']:.1f} MeV")
            print(f"  e: {solver_result.optimized_constants['e']:.3f}")
            
            print("\nPERFORMANCE:")
            print(f"  Execution time: {solver_result.execution_time:.1f} seconds")
            print(f"  Iterations: {solver_result.iterations}")
            print(f"  Strategy used: {solver_result.convergence_info['strategy_used']}")
            
            # Save results if output directory specified
            if args.output:
                self._save_results(model, solver_result, args.output, args)
                print(f"\nResults saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error executing proton solve command: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _save_results(self, model, solver_result, output_dir: str, args) -> None:
        """Save results to output directory."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = {
            "input_parameters": {
                "grid_size": args.grid_size,
                "box_size": args.box_size,
                "config_type": args.config_type,
                "c2": args.c2,
                "c4": args.c4,
                "c6": args.c6,
                "F_pi": args.F_pi,
                "e": args.e,
                "target_mass": args.target_mass,
                "target_radius": args.target_radius,
                "target_magnetic_moment": args.target_magnetic_moment,
                "target_bands": args.target_bands,
                "optimization_strategy": args.optimization_strategy
            },
            "solver_results": {
                "success": solver_result.success,
                "physical_parameters": solver_result.physical_parameters,
                "energy_analysis": solver_result.energy_analysis,
                "mode_analysis": solver_result.mode_analysis,
                "topological_analysis": solver_result.topological_analysis,
                "interference_analysis": solver_result.interference_analysis,
                "optimized_constants": solver_result.optimized_constants,
                "convergence_info": solver_result.convergence_info,
                "execution_time": solver_result.execution_time,
                "iterations": solver_result.iterations
            },
            "model_results": {
                "mass": model.physical_quantities.mass,
                "charge_radius": model.physical_quantities.charge_radius,
                "magnetic_moment": model.physical_quantities.magnetic_moment,
                "electric_charge": model.physical_quantities.electric_charge,
                "baryon_number": model.physical_quantities.baryon_number
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if hasattr(obj, 'item'):
                item = obj.item()
                if isinstance(item, complex):
                    return float(item.real)
                return item
            elif isinstance(obj, (np.complex128, np.complex64, complex)):
                return float(obj.real)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        results_data = convert_numpy(results_data)
        
        # Save JSON file
        json_file = output_path / "proton_solver_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary as text
        summary_file = output_path / "proton_solver_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PROTON MODEL WITH UNIVERSAL SOLVER RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PHYSICAL PARAMETERS:\n")
            f.write(f"  Mass: {solver_result.physical_parameters['mass']:.1f} MeV\n")
            f.write(f"  Charge radius: {solver_result.physical_parameters['charge_radius']:.3f} fm\n")
            f.write(f"  Magnetic moment: {solver_result.physical_parameters['magnetic_moment']:.3f} μN\n")
            f.write(f"  Electric charge: {solver_result.physical_parameters['electric_charge']:.3f}\n")
            f.write(f"  Baryon number: {solver_result.physical_parameters['baryon_number']:.3f}\n\n")
            
            f.write("ENERGY ANALYSIS:\n")
            f.write(f"  E₂: {solver_result.energy_analysis['e2']:.1f} MeV\n")
            f.write(f"  E₄: {solver_result.energy_analysis['e4']:.1f} MeV\n")
            f.write(f"  E₆: {solver_result.energy_analysis['e6']:.1f} MeV\n")
            f.write(f"  Total: {solver_result.energy_analysis['total']:.1f} MeV\n")
            f.write(f"  Ratios: E₂/E₄ = {solver_result.energy_analysis['e2_ratio']:.2f}/{solver_result.energy_analysis['e4_ratio']:.2f}\n\n")
            
            f.write("OPTIMIZED CONSTANTS:\n")
            f.write(f"  c₂: {solver_result.optimized_constants['c2']:.6f}\n")
            f.write(f"  c₄: {solver_result.optimized_constants['c4']:.6f}\n")
            f.write(f"  c₆: {solver_result.optimized_constants['c6']:.6f}\n")
            f.write(f"  F_π: {solver_result.optimized_constants['F_pi']:.1f} MeV\n")
            f.write(f"  e: {solver_result.optimized_constants['e']:.3f}\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Execution time: {solver_result.execution_time:.1f} seconds\n")
            f.write(f"  Iterations: {solver_result.iterations}\n")
            f.write(f"  Strategy used: {solver_result.convergence_info['strategy_used']}\n")


class ProtonPhaseEnvironmentCommand(BaseCommand):
    """Proton phase environment analysis command."""

    def __init__(self) -> None:
        """Initialize proton phase environment command."""
        super().__init__(
            name="phase",
            description="Analyze phase environment according to 7D theory"
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add phase environment command arguments."""
        parser.add_argument(
            "--grid-size", type=int, default=32,
            help="Grid size (default: 32)"
        )
        parser.add_argument(
            "--box-size", type=float, default=4.0,
            help="Box size in fm (default: 4.0)"
        )
        parser.add_argument(
            "--config-type", type=str, default="120deg",
            choices=["120deg", "clover", "cartesian"],
            help="Configuration type (default: 120deg)"
        )
        parser.add_argument(
            "--c2", type=float, default=1.0,
            help="Skyrme constant c2 (default: 1.0)"
        )
        parser.add_argument(
            "--c4", type=float, default=1.0,
            help="Skyrme constant c4 (default: 1.0)"
        )
        parser.add_argument(
            "--c6", type=float, default=1.0,
            help="Skyrme constant c6 (default: 1.0)"
        )
        parser.add_argument(
            "--F-pi", type=float, default=186.0,
            help="Pion decay constant in MeV (default: 186.0)"
        )
        parser.add_argument(
            "--e", type=float, default=5.45,
            help="Dimensionless Skyrme constant (default: 5.45)"
        )
        parser.add_argument(
            "--well-depth", type=float, default=1.0,
            help="Phase well depth (default: 1.0)"
        )
        parser.add_argument(
            "--well-width", type=float, default=1.0,
            help="Phase well width (default: 1.0)"
        )
        parser.add_argument(
            "--compression-strength", type=float, default=1.0,
            help="Phase compression strength (default: 1.0)"
        )
        parser.add_argument(
            "--rarefaction-strength", type=float, default=1.0,
            help="Phase rarefaction strength (default: 1.0)"
        )
        parser.add_argument(
            "--output", type=str,
            help="Output directory for results"
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Verbose output"
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the proton phase environment command."""
        try:
            print("PROTON PHASE ENVIRONMENT ANALYSIS")
            print("=" * 50)
            
            # Display CUDA status
            cuda_status = get_cuda_status()
            print(f"CUDA Status: {cuda_status}")
            print()
            
            # Create model configuration
            config = ModelConfig(
                grid_size=args.grid_size,
                box_size=args.box_size,
                c2=args.c2,
                c4=args.c4,
                c6=args.c6,
                F_pi=args.F_pi,
                e=args.e
            )
            config.config_type = args.config_type
            
            if args.verbose:
                print(f"Configuration:")
                print(f"  Grid size: {config.grid_size}")
                print(f"  Box size: {config.box_size} fm")
                print(f"  Config type: {config.config_type}")
                print(f"  Constants: c2={config.c2}, c4={config.c4}, c6={config.c6}")
                print(f"  Physical: F_π={config.F_pi} MeV, e={config.e}")
                print(f"  Phase well: depth={args.well_depth}, width={args.well_width}")
                print(f"  Balance: compression={args.compression_strength}, rarefaction={args.rarefaction_strength}")
                print()
            
            # Create and initialize model
            model = ProtonModel(config)
            model.create_geometry()
            model.build_fields()
            model.calculate_energy()
            model.calculate_physics()
            
            # Configure phase environment
            if model.phase_environment:
                model.phase_environment.well_params.well_depth = args.well_depth
                model.phase_environment.well_params.well_width = args.well_width
                model.phase_environment.well_params.compression_strength = args.compression_strength
                model.phase_environment.well_params.rarefaction_strength = args.rarefaction_strength
            
            # Analyze phase environment
            print("Analyzing phase environment...")
            phase_analysis = model.analyze_phase_environment()
            
            if 'error' in phase_analysis:
                print(f"❌ Phase environment analysis failed: {phase_analysis['error']}")
                return 1
            
            # Display results
            print("\n" + "="*60)
            print("PHASE ENVIRONMENT ANALYSIS RESULTS")
            print("="*60)
            
            print("\n" + phase_analysis['environment_report'])
            
            print("\nCOMPRESSION-RAREFACTION BALANCE:")
            balance = phase_analysis['compression_rarefaction_balance']
            print(f"  Total compression: {balance['total_compression']:.3f}")
            print(f"  Total rarefaction: {balance['total_rarefaction']:.3f}")
            print(f"  Balance ratio: {balance['balance_ratio']:.3f}")
            print(f"  Is stable: {balance['is_stable']}")
            print(f"  Stability margin: {balance['stability_margin']:.3f}")
            
            print("\nIMPEDANCE PARAMETERS:")
            impedance = phase_analysis['impedance_parameters']
            print(f"  K_real: {impedance.K_real}")
            print(f"  K_imag: {impedance.K_imag}")
            print(f"  Boundary radius: {impedance.boundary_radius:.3f} fm")
            print(f"  Phase velocity: {impedance.phase_velocity}")
            
            print("\nSCALE QUANTIZATION:")
            quantization = phase_analysis['scale_quantization']
            print(f"  Natural radius R*: {quantization.natural_radius_R_star:.3f} fm")
            print(f"  Spectral radii Rn: {len(quantization.spectral_radii_Rn)} modes")
            print(f"  Allowed radii: {len(quantization.allowed_radii)} modes")
            print(f"  ΔR: {quantization.delta_R:.3f} fm")
            
            if quantization.allowed_radii:
                print(f"  First few allowed radii: {[f'{r:.3f}' for r in quantization.allowed_radii[:5]]}")
            
            print("\nPHASE FIELD ENERGY:")
            energy = phase_analysis['phase_energy']
            print(f"  Total phase energy: {energy['total_phase_energy']:.3f}")
            print(f"  Interaction energy: {energy['interaction_energy']:.3f}")
            print(f"  Total energy: {energy['total_energy']:.3f}")
            
            # Save results if output directory specified
            if args.output:
                self._save_results(model, phase_analysis, args.output, args)
                print(f"\nResults saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error executing proton phase environment command: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _save_results(self, model, phase_analysis, output_dir: str, args) -> None:
        """Save results to output directory."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = {
            "input_parameters": {
                "grid_size": args.grid_size,
                "box_size": args.box_size,
                "config_type": args.config_type,
                "c2": args.c2,
                "c4": args.c4,
                "c6": args.c6,
                "F_pi": args.F_pi,
                "e": args.e,
                "well_depth": args.well_depth,
                "well_width": args.well_width,
                "compression_strength": args.compression_strength,
                "rarefaction_strength": args.rarefaction_strength
            },
            "phase_analysis": {
                "compression_rarefaction_balance": phase_analysis['compression_rarefaction_balance'],
                "impedance_parameters": {
                    "K_real": phase_analysis['impedance_parameters'].K_real,
                    "K_imag": phase_analysis['impedance_parameters'].K_imag,
                    "boundary_radius": phase_analysis['impedance_parameters'].boundary_radius,
                    "phase_velocity": phase_analysis['impedance_parameters'].phase_velocity
                },
                "scale_quantization": {
                    "natural_radius_R_star": phase_analysis['scale_quantization'].natural_radius_R_star,
                    "spectral_radii_Rn": phase_analysis['scale_quantization'].spectral_radii_Rn,
                    "allowed_radii": phase_analysis['scale_quantization'].allowed_radii,
                    "delta_R": phase_analysis['scale_quantization'].delta_R
                },
                "phase_energy": phase_analysis['phase_energy']
            },
            "model_results": {
                "mass": model.physical_quantities.mass,
                "charge_radius": model.physical_quantities.charge_radius,
                "magnetic_moment": model.physical_quantities.magnetic_moment,
                "electric_charge": model.physical_quantities.electric_charge,
                "baryon_number": model.physical_quantities.baryon_number
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if hasattr(obj, 'item'):
                item = obj.item()
                if isinstance(item, complex):
                    return float(item.real)
                return item
            elif isinstance(obj, (np.complex128, np.complex64, complex)):
                return float(obj.real)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        results_data = convert_numpy(results_data)
        
        # Save JSON file
        json_file = output_path / "proton_phase_environment_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary as text
        summary_file = output_path / "proton_phase_environment_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PROTON PHASE ENVIRONMENT ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(phase_analysis['environment_report'])
            
            f.write("\n\nCOMPRESSION-RAREFACTION BALANCE:\n")
            balance = phase_analysis['compression_rarefaction_balance']
            f.write(f"  Total compression: {balance['total_compression']:.3f}\n")
            f.write(f"  Total rarefaction: {balance['total_rarefaction']:.3f}\n")
            f.write(f"  Balance ratio: {balance['balance_ratio']:.3f}\n")
            f.write(f"  Is stable: {balance['is_stable']}\n")
            f.write(f"  Stability margin: {balance['stability_margin']:.3f}\n")
            
            f.write("\nIMPEDANCE PARAMETERS:\n")
            impedance = phase_analysis['impedance_parameters']
            f.write(f"  K_real: {impedance.K_real}\n")
            f.write(f"  K_imag: {impedance.K_imag}\n")
            f.write(f"  Boundary radius: {impedance.boundary_radius:.3f} fm\n")
            f.write(f"  Phase velocity: {impedance.phase_velocity}\n")
            
            f.write("\nSCALE QUANTIZATION:\n")
            quantization = phase_analysis['scale_quantization']
            f.write(f"  Natural radius R*: {quantization.natural_radius_R_star:.3f} fm\n")
            f.write(f"  Spectral radii Rn: {len(quantization.spectral_radii_Rn)} modes\n")
            f.write(f"  Allowed radii: {len(quantization.allowed_radii)} modes\n")
            f.write(f"  ΔR: {quantization.delta_R:.3f} fm\n")
            
            if quantization.allowed_radii:
                f.write(f"  First few allowed radii: {[f'{r:.3f}' for r in quantization.allowed_radii[:5]]}\n")
            
            f.write("\nPHASE FIELD ENERGY:\n")
            energy = phase_analysis['phase_energy']
            f.write(f"  Total phase energy: {energy['total_phase_energy']:.3f}\n")
            f.write(f"  Interaction energy: {energy['interaction_energy']:.3f}\n")
            f.write(f"  Total energy: {energy['total_energy']:.3f}\n")

    def get_subcommands(self) -> list:
        """Get available subcommands."""
        return []

    def get_help(self) -> str:
        """Get help text."""
        return """
Proton solve command using universal solver with advanced optimization.

This command uses the universal solver to optimize proton model parameters
using various strategies including energy balance, physical parameters,
and quantization analysis.

Examples:
  # Basic proton solve
  phaze-particles proton solve --grid-size 32 --box-size 4.0

  # Optimize for target radius
  phaze-particles proton solve --target-radius 0.841 --optimization-strategy physical_params

  # Optimize for target bands
  phaze-particles proton solve --target-bands 11 --optimization-strategy quantization

  # Full optimization with all targets
  phaze-particles proton solve --target-mass 938.272 --target-radius 0.841 --target-magnetic-moment 2.793
        """


class ProtonPhaseTailsCommand(BaseCommand):
    """Proton phase tails analysis command."""
    
    def __init__(self):
        """Initialize phase tails command."""
        super().__init__(
            name="phase-tails",
            description="Analyze phase tails and interference patterns in proton model"
        )
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add phase tails command arguments."""
        # Grid parameters
        parser.add_argument(
            "--grid-size", type=int, default=32,
            help="Grid size for calculations (default: 32)"
        )
        parser.add_argument(
            "--box-size", type=float, default=4.0,
            help="Box size in fm (default: 4.0)"
        )
        
        # Torus configuration
        parser.add_argument(
            "--torus-config", type=str, default="120deg",
            choices=["120deg", "clover", "cartesian"],
            help="Torus configuration (default: 120deg)"
        )
        parser.add_argument(
            "--r-scale", type=float, default=1.0,
            help="Radial scale parameter (default: 1.0)"
        )
        
        # Skyrme constants
        parser.add_argument(
            "--c2", type=float, default=1.0,
            help="Skyrme constant c2 (default: 1.0)"
        )
        parser.add_argument(
            "--c4", type=float, default=1.0,
            help="Skyrme constant c4 (default: 1.0)"
        )
        parser.add_argument(
            "--c6", type=float, default=0.0,
            help="Skyrme constant c6 (default: 0.0)"
        )
        
        # Physical constants
        parser.add_argument(
            "--F-pi", type=float, default=186.0,
            help="Pion decay constant in MeV (default: 186.0)"
        )
        parser.add_argument(
            "--e", type=float, default=5.45,
            help="Dimensionless Skyrme constant (default: 5.45)"
        )
        
        # Output options
        parser.add_argument(
            "--output-dir", type=str, default="results/proton/phase-tails",
            help="Output directory for results (default: results/proton/phase-tails)"
        )
        parser.add_argument(
            "--save-results", action="store_true",
            help="Save analysis results to files"
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Enable verbose output"
        )
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute phase tails analysis."""
        try:
            print("=" * 60)
            print("PROTON PHASE TAILS ANALYSIS")
            print("=" * 60)
            print()
            
            # Create model configuration
            from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
            
            config = ModelConfig(
                grid_size=args.grid_size,
                box_size=args.box_size,
                torus_config=args.torus_config,
                r_scale=args.r_scale,
                c2=args.c2,
                c4=args.c4,
                c6=args.c6,
                F_pi=args.F_pi,
                e=args.e
            )
            
            print(f"Configuration:")
            print(f"  Grid size: {config.grid_size}")
            print(f"  Box size: {config.box_size} fm")
            print(f"  Torus config: {config.torus_config}")
            print(f"  Skyrme constants: c2={config.c2}, c4={config.c4}, c6={config.c6}")
            print(f"  Physical constants: F_π={config.F_pi} MeV, e={config.e}")
            print()
            
            # Initialize model
            print("Initializing proton model...")
            model = ProtonModel(config)
            
            # Build geometry and fields
            print("Building geometry and fields...")
            model.create_geometry()
            model.build_fields()
            
            # Calculate energy
            print("Calculating energy...")
            model.calculate_energy()
            
            # Analyze phase tails
            print("Analyzing phase tails and interference...")
            phase_tail_result = model.analyze_phase_tails()
            
            # Generate report
            print("Generating analysis report...")
            report = model.get_phase_tail_report()
            
            print(report)
            
            # Save results if requested
            if args.save_results:
                import os
                from datetime import datetime
                
                os.makedirs(args.output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
                filename = f"phase-tails-analysis-{timestamp}.txt"
                filepath = os.path.join(args.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report)
                    f.write(f"\n\nConfiguration:\n")
                    f.write(f"  Grid size: {config.grid_size}\n")
                    f.write(f"  Box size: {config.box_size} fm\n")
                    f.write(f"  Torus config: {config.torus_config}\n")
                    f.write(f"  Skyrme constants: c2={config.c2}, c4={config.c4}, c6={config.c6}\n")
                    f.write(f"  Physical constants: F_π={config.F_pi} MeV, e={config.e}\n")
                
                print(f"\nResults saved to: {filepath}")
            
            print("\nPhase tails analysis completed successfully!")
            return 0
            
        except Exception as e:
            print(f"Error during phase tails analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def get_subcommands(self) -> list:
        """Get available subcommands."""
        return []
    
    def get_help(self) -> str:
        """Get help text."""
        return """
Proton phase tails analysis command.

This command analyzes phase tails and interference patterns in the proton model
based on the 7D phase space-time theory. It examines:

- Phase tail energy and structure
- Interference patterns and coherence
- Resonance modes and quantization
- Energy contributions from tails vs background
- Geometric effects on effective metric
- Stability and coherence assessment

Examples:
  # Basic phase tails analysis
  phaze-particles proton phase-tails

  # Analysis with custom parameters
  phaze-particles proton phase-tails --grid-size 64 --box-size 6.0 --c4 2.0

  # Save results to file
  phaze-particles proton phase-tails --save-results --output-dir results/analysis

  # Verbose output with detailed information
  phaze-particles proton phase-tails --verbose
        """
