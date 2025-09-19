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
                    "c2": model.optimization_result.c2,
                    "c4": model.optimization_result.c4,
                    "c6": model.optimization_result.c6,
                    "optimization_targets": {
                        "target_e2_ratio": args.target_e2_ratio,
                        "target_e4_ratio": args.target_e4_ratio,
                        "target_virial_residual": args.target_virial_residual
                    },
                    "achieved_values": {
                        "e2_ratio": model.optimization_result.e2_ratio,
                        "e4_ratio": model.optimization_result.e4_ratio,
                        "virial_residual": model.optimization_result.virial_residual
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
