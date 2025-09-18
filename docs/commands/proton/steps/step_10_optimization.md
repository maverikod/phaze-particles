# Шаг 10: Оптимизация

## Цель

Реализовать систему оптимизации модели протона:
- CUDA оптимизация для ускорения вычислений
- Оптимизация памяти и алгоритмов
- Масштабирование для больших сеток
- Профилирование и мониторинг производительности
- Автоматическая настройка параметров

## Обзор

Оптимизация является финальным этапом разработки модели, направленным на достижение максимальной производительности при сохранении точности вычислений. Она включает:
- Использование GPU для параллельных вычислений
- Оптимизацию алгоритмов и структур данных
- Масштабирование на большие сетки
- Профилирование и мониторинг
- Автоматическую настройку параметров

## Стратегии оптимизации

### 10.1 CUDA оптимизация

- Параллелизация вычислений на GPU
- Оптимизация доступа к памяти
- Использование shared memory
- Оптимизация блоков и потоков

### 10.2 Алгоритмическая оптимизация

- Оптимизация численных методов
- Улучшение сходимости
- Адаптивные алгоритмы
- Кэширование промежуточных результатов

### 10.3 Масштабирование

- Поддержка больших сеток
- Распределенные вычисления
- Оптимизация памяти
- Адаптивные сетки

## Декларативный код

```python
#!/usr/bin/env python3
"""
Оптимизация модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import time
import psutil
import os
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# CUDA импорты
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available, using CPU only")

# Импорт основных модулей
from .step_01_mathematical_foundations import MathematicalFoundations
from .step_02_torus_geometries import TorusGeometries, TorusConfiguration
from .step_03_su2_fields import SU2Field, SU2FieldBuilder
from .step_04_energy_densities import EnergyDensity, EnergyDensityCalculator
from .step_05_physical_quantities import PhysicalQuantities, PhysicalQuantitiesCalculator
from .step_06_numerical_methods import RelaxationSolver, RelaxationConfig, ConstraintConfig
from .step_08_integration import ProtonModel, ModelConfig


class OptimizationLevel(Enum):
    """Уровни оптимизации."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Метрики производительности."""
    execution_time: float
    memory_usage: float
    gpu_utilization: float
    cpu_utilization: float
    iterations: int
    convergence_rate: float
    throughput: float  # вычислений в секунду


@dataclass
class OptimizationConfig:
    """Конфигурация оптимизации."""
    use_cuda: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    memory_optimization: bool = True
    algorithm_optimization: bool = True
    adaptive_parameters: bool = True
    profiling_enabled: bool = True
    cache_enabled: bool = True
    parallel_processing: bool = True


class CUDAManager:
    """Менеджер CUDA операций."""
    
    def __init__(self):
        """Инициализация CUDA менеджера."""
        self.cuda_available = CUDA_AVAILABLE
        self.device_count = 0
        self.current_device = 0
        
        if self.cuda_available:
            self.device_count = cp.cuda.runtime.getDeviceCount()
            self._initialize_device()
    
    def _initialize_device(self):
        """Инициализация CUDA устройства."""
        if self.cuda_available:
            cp.cuda.Device(self.current_device).use()
            print(f"CUDA device {self.current_device} initialized")
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Получение информации об устройстве.
        
        Returns:
            Информация об устройстве
        """
        if not self.cuda_available:
            return {"available": False}
        
        device = cp.cuda.Device(self.current_device)
        props = device.attributes
        
        return {
            "available": True,
            "device_count": self.device_count,
            "current_device": self.current_device,
            "name": cp.cuda.runtime.getDeviceProperties(self.current_device)['name'].decode(),
            "compute_capability": f"{props['Major']}.{props['Minor']}",
            "total_memory": props['totalGlobalMem'],
            "shared_memory": props['sharedMemPerBlock'],
            "max_threads_per_block": props['maxThreadsPerBlock']
        }
    
    def to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """
        Перенос массива на GPU.
        
        Args:
            array: NumPy массив
            
        Returns:
            CuPy массив
        """
        if self.cuda_available:
            return cp.asarray(array)
        else:
            return array
    
    def to_cpu(self, array: 'cp.ndarray') -> np.ndarray:
        """
        Перенос массива на CPU.
        
        Args:
            array: CuPy массив
            
        Returns:
            NumPy массив
        """
        if self.cuda_available and hasattr(array, 'get'):
            return array.get()
        else:
            return array
    
    def synchronize(self):
        """Синхронизация CUDA операций."""
        if self.cuda_available:
            cp.cuda.Stream.null.synchronize()


class MemoryOptimizer:
    """Оптимизатор памяти."""
    
    def __init__(self, cuda_manager: CUDAManager):
        """
        Инициализация оптимизатора памяти.
        
        Args:
            cuda_manager: Менеджер CUDA
        """
        self.cuda_manager = cuda_manager
        self.memory_pool = {}
        self.cache = {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получение информации об использовании памяти.
        
        Returns:
            Информация об использовании памяти
        """
        memory_info = {}
        
        # CPU память
        process = psutil.Process(os.getpid())
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # GPU память
        if self.cuda_manager.cuda_available:
            try:
                mempool = cp.get_default_memory_pool()
                memory_info['gpu_memory_mb'] = mempool.used_bytes() / 1024 / 1024
                memory_info['gpu_memory_total_mb'] = mempool.total_bytes() / 1024 / 1024
            except:
                memory_info['gpu_memory_mb'] = 0
                memory_info['gpu_memory_total_mb'] = 0
        else:
            memory_info['gpu_memory_mb'] = 0
            memory_info['gpu_memory_total_mb'] = 0
        
        return memory_info
    
    def optimize_memory_layout(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Оптимизация расположения массивов в памяти.
        
        Args:
            arrays: Список массивов
            
        Returns:
            Оптимизированные массивы
        """
        optimized_arrays = []
        
        for array in arrays:
            # Обеспечение непрерывности в памяти
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
            
            # Оптимизация типа данных
            if array.dtype == np.float64:
                array = array.astype(np.float32)
            
            optimized_arrays.append(array)
        
        return optimized_arrays
    
    def cache_result(self, key: str, result: Any):
        """
        Кэширование результата.
        
        Args:
            key: Ключ кэша
            result: Результат для кэширования
        """
        self.cache[key] = result
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Получение результата из кэша.
        
        Args:
            key: Ключ кэша
            
        Returns:
            Кэшированный результат или None
        """
        return self.cache.get(key)


class AlgorithmOptimizer:
    """Оптимизатор алгоритмов."""
    
    def __init__(self, cuda_manager: CUDAManager):
        """
        Инициализация оптимизатора алгоритмов.
        
        Args:
            cuda_manager: Менеджер CUDA
        """
        self.cuda_manager = cuda_manager
        self.optimized_kernels = {}
    
    def optimize_gradient_calculation(self, field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Оптимизированное вычисление градиента.
        
        Args:
            field: Скалярное поле
            dx: Шаг сетки
            
        Returns:
            Компоненты градиента
        """
        if self.cuda_manager.cuda_available:
            # CUDA оптимизированное вычисление
            field_gpu = self.cuda_manager.to_gpu(field)
            
            # Использование CuPy для градиента
            grad_x = cp.gradient(field_gpu, dx, axis=0)
            grad_y = cp.gradient(field_gpu, dx, axis=1)
            grad_z = cp.gradient(field_gpu, dx, axis=2)
            
            # Перенос обратно на CPU
            grad_x = self.cuda_manager.to_cpu(grad_x)
            grad_y = self.cuda_manager.to_cpu(grad_y)
            grad_z = self.cuda_manager.to_cpu(grad_z)
            
            return grad_x, grad_y, grad_z
        else:
            # CPU вычисление
            grad_x = np.gradient(field, dx, axis=0)
            grad_y = np.gradient(field, dx, axis=1)
            grad_z = np.gradient(field, dx, axis=2)
            
            return grad_x, grad_y, grad_z
    
    def optimize_integration(self, field: np.ndarray, dx: float) -> float:
        """
        Оптимизированное интегрирование.
        
        Args:
            field: Поле для интегрирования
            dx: Шаг сетки
            
        Returns:
            Результат интегрирования
        """
        if self.cuda_manager.cuda_available:
            # CUDA оптимизированное интегрирование
            field_gpu = self.cuda_manager.to_gpu(field)
            result = cp.sum(field_gpu) * dx**3
            return float(self.cuda_manager.to_cpu(result))
        else:
            # CPU интегрирование
            return float(np.sum(field) * dx**3)
    
    def optimize_matrix_operations(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Оптимизированные матричные операции.
        
        Args:
            matrices: Список матриц
            
        Returns:
            Результат операций
        """
        if self.cuda_manager.cuda_available:
            # CUDA оптимизированные операции
            matrices_gpu = [self.cuda_manager.to_gpu(m) for m in matrices]
            
            result = matrices_gpu[0]
            for matrix in matrices_gpu[1:]:
                result = cp.dot(result, matrix)
            
            return self.cuda_manager.to_cpu(result)
        else:
            # CPU операции
            result = matrices[0]
            for matrix in matrices[1:]:
                result = np.dot(result, matrix)
            
            return result


class AdaptiveParameterOptimizer:
    """Оптимизатор адаптивных параметров."""
    
    def __init__(self):
        """Инициализация оптимизатора параметров."""
        self.parameter_history = {}
        self.performance_history = {}
    
    def optimize_step_size(self, current_step: float, convergence_rate: float, 
                          iteration: int) -> float:
        """
        Оптимизация размера шага.
        
        Args:
            current_step: Текущий размер шага
            convergence_rate: Скорость сходимости
            iteration: Номер итерации
            
        Returns:
            Оптимизированный размер шага
        """
        # Адаптивная настройка размера шага
        if convergence_rate > 0.1:
            # Быстрая сходимость - увеличиваем шаг
            new_step = current_step * 1.1
        elif convergence_rate < 0.01:
            # Медленная сходимость - уменьшаем шаг
            new_step = current_step * 0.9
        else:
            new_step = current_step
        
        # Ограничения
        new_step = max(0.001, min(0.1, new_step))
        
        return new_step
    
    def optimize_tolerance(self, current_tolerance: float, error: float) -> float:
        """
        Оптимизация допуска.
        
        Args:
            current_tolerance: Текущий допуск
            error: Текущая ошибка
            
        Returns:
            Оптимизированный допуск
        """
        # Адаптивная настройка допуска
        if error < current_tolerance * 0.1:
            # Очень маленькая ошибка - можно увеличить допуск
            new_tolerance = current_tolerance * 1.5
        elif error > current_tolerance * 10:
            # Большая ошибка - уменьшаем допуск
            new_tolerance = current_tolerance * 0.5
        else:
            new_tolerance = current_tolerance
        
        # Ограничения
        new_tolerance = max(1e-8, min(1e-3, new_tolerance))
        
        return new_tolerance
    
    def optimize_constraints(self, current_constraints: Dict[str, float], 
                           violations: Dict[str, float]) -> Dict[str, float]:
        """
        Оптимизация ограничений.
        
        Args:
            current_constraints: Текущие ограничения
            violations: Нарушения ограничений
            
        Returns:
            Оптимизированные ограничения
        """
        new_constraints = {}
        
        for key, current_value in current_constraints.items():
            violation = violations.get(key, 0.0)
            
            if violation > 0.1:
                # Большое нарушение - увеличиваем вес
                new_constraints[key] = current_value * 2.0
            elif violation < 0.01:
                # Малое нарушение - уменьшаем вес
                new_constraints[key] = current_value * 0.5
            else:
                new_constraints[key] = current_value
        
        return new_constraints


class PerformanceProfiler:
    """Профилировщик производительности."""
    
    def __init__(self):
        """Инициализация профилировщика."""
        self.profiles = {}
        self.current_profile = None
    
    def start_profile(self, name: str):
        """
        Начало профилирования.
        
        Args:
            name: Имя профиля
        """
        self.current_profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': psutil.cpu_percent()
        }
    
    def end_profile(self, name: str) -> Dict[str, Any]:
        """
        Завершение профилирования.
        
        Args:
            name: Имя профиля
            
        Returns:
            Результаты профилирования
        """
        if not self.current_profile or self.current_profile['name'] != name:
            raise ValueError(f"Profile {name} not started")
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        profile_result = {
            'name': name,
            'execution_time': end_time - self.current_profile['start_time'],
            'memory_delta': end_memory - self.current_profile['start_memory'],
            'cpu_usage': (self.current_profile['start_cpu'] + end_cpu) / 2,
            'timestamp': end_time
        }
        
        self.profiles[name] = profile_result
        self.current_profile = None
        
        return profile_result
    
    def _get_memory_usage(self) -> float:
        """
        Получение использования памяти.
        
        Returns:
            Использование памяти в МБ
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Получение сводки профилирования.
        
        Returns:
            Сводка профилирования
        """
        if not self.profiles:
            return {}
        
        total_time = sum(p['execution_time'] for p in self.profiles.values())
        total_memory = sum(p['memory_delta'] for p in self.profiles.values())
        
        return {
            'total_execution_time': total_time,
            'total_memory_delta': total_memory,
            'profile_count': len(self.profiles),
            'profiles': self.profiles
        }


class OptimizedProtonModel:
    """Оптимизированная модель протона."""
    
    def __init__(self, config: ModelConfig, optimization_config: OptimizationConfig):
        """
        Инициализация оптимизированной модели.
        
        Args:
            config: Конфигурация модели
            optimization_config: Конфигурация оптимизации
        """
        self.config = config
        self.optimization_config = optimization_config
        
        # Инициализация оптимизаторов
        self.cuda_manager = CUDAManager()
        self.memory_optimizer = MemoryOptimizer(self.cuda_manager)
        self.algorithm_optimizer = AlgorithmOptimizer(self.cuda_manager)
        self.parameter_optimizer = AdaptiveParameterOptimizer()
        self.profiler = PerformanceProfiler()
        
        # Информация об устройстве
        self.device_info = self.cuda_manager.get_device_info()
        
        # Инициализация базовой модели
        self.base_model = ProtonModel(config)
        
        # Метрики производительности
        self.performance_metrics = None
    
    def optimize_configuration(self) -> ModelConfig:
        """
        Оптимизация конфигурации модели.
        
        Returns:
            Оптимизированная конфигурация
        """
        optimized_config = self.config
        
        if self.optimization_config.adaptive_parameters:
            # Адаптивная настройка параметров
            if self.device_info['available']:
                # GPU доступен - увеличиваем сетку
                optimized_config.grid_size = min(128, self.config.grid_size * 2)
                optimized_config.max_iterations = min(2000, self.config.max_iterations * 2)
            else:
                # Только CPU - оптимизируем для CPU
                optimized_config.grid_size = min(64, self.config.grid_size)
                optimized_config.max_iterations = min(1000, self.config.max_iterations)
        
        return optimized_config
    
    def run_optimized(self) -> Dict[str, Any]:
        """
        Запуск оптимизированной модели.
        
        Returns:
            Результаты оптимизированной модели
        """
        # Начало профилирования
        self.profiler.start_profile("total_execution")
        
        # Оптимизация конфигурации
        optimized_config = self.optimize_configuration()
        
        # Создание оптимизированной модели
        optimized_model = ProtonModel(optimized_config)
        
        # Профилирование основных этапов
        self.profiler.start_profile("geometry_creation")
        optimized_model.create_geometry()
        self.profiler.end_profile("geometry_creation")
        
        self.profiler.start_profile("field_building")
        optimized_model.build_fields()
        self.profiler.end_profile("field_building")
        
        self.profiler.start_profile("energy_calculation")
        optimized_model.calculate_energy()
        self.profiler.end_profile("energy_calculation")
        
        self.profiler.start_profile("physics_calculation")
        optimized_model.calculate_physics()
        self.profiler.end_profile("physics_calculation")
        
        self.profiler.start_profile("optimization")
        optimized_model.optimize()
        self.profiler.end_profile("optimization")
        
        self.profiler.start_profile("validation")
        optimized_model.validate()
        self.profiler.end_profile("validation")
        
        # Завершение общего профилирования
        total_profile = self.profiler.end_profile("total_execution")
        
        # Получение результатов
        results = optimized_model.get_results()
        
        # Вычисление метрик производительности
        self.performance_metrics = self._calculate_performance_metrics(
            results, total_profile
        )
        
        # Сводка профилирования
        profile_summary = self.profiler.get_profile_summary()
        
        return {
            'results': results,
            'performance_metrics': self.performance_metrics,
            'profile_summary': profile_summary,
            'device_info': self.device_info,
            'optimization_config': self.optimization_config
        }
    
    def _calculate_performance_metrics(self, results: Any, total_profile: Dict[str, Any]) -> PerformanceMetrics:
        """
        Вычисление метрик производительности.
        
        Args:
            results: Результаты модели
            total_profile: Профиль общего времени выполнения
            
        Returns:
            Метрики производительности
        """
        # Использование памяти
        memory_info = self.memory_optimizer.get_memory_usage()
        
        # Вычисление пропускной способности
        total_operations = self.config.grid_size**3 * results.iterations
        throughput = total_operations / total_profile['execution_time']
        
        # Скорость сходимости
        convergence_rate = 1.0 / results.iterations if results.iterations > 0 else 0.0
        
        return PerformanceMetrics(
            execution_time=total_profile['execution_time'],
            memory_usage=memory_info['cpu_memory_mb'],
            gpu_utilization=memory_info.get('gpu_memory_mb', 0),
            cpu_utilization=total_profile['cpu_usage'],
            iterations=results.iterations,
            convergence_rate=convergence_rate,
            throughput=throughput
        )
    
    def benchmark(self, grid_sizes: List[int] = [32, 64, 128]) -> Dict[str, Any]:
        """
        Бенчмарк производительности.
        
        Args:
            grid_sizes: Размеры сеток для тестирования
            
        Returns:
            Результаты бенчмарка
        """
        benchmark_results = {}
        
        for grid_size in grid_sizes:
            print(f"Benchmarking grid size: {grid_size}")
            
            # Создание конфигурации для бенчмарка
            benchmark_config = ModelConfig(
                grid_size=grid_size,
                box_size=self.config.box_size,
                max_iterations=100,  # Ограниченное количество итераций
                validation_enabled=False
            )
            
            # Создание оптимизированной модели
            benchmark_model = OptimizedProtonModel(
                benchmark_config, 
                self.optimization_config
            )
            
            # Запуск бенчмарка
            start_time = time.time()
            results = benchmark_model.run_optimized()
            end_time = time.time()
            
            # Сохранение результатов
            benchmark_results[grid_size] = {
                'execution_time': end_time - start_time,
                'performance_metrics': results['performance_metrics'],
                'device_info': results['device_info']
            }
        
        return benchmark_results
    
    def get_optimization_report(self) -> str:
        """
        Получение отчета об оптимизации.
        
        Returns:
            Отчет об оптимизации
        """
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ОБ ОПТИМИЗАЦИИ МОДЕЛИ ПРОТОНА")
        report.append("=" * 80)
        
        # Информация об устройстве
        report.append("ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ:")
        report.append("-" * 40)
        if self.device_info['available']:
            report.append(f"CUDA доступен: ДА")
            report.append(f"Устройство: {self.device_info['name']}")
            report.append(f"Вычислительная способность: {self.device_info['compute_capability']}")
            report.append(f"Общая память: {self.device_info['total_memory'] / 1024**3:.2f} ГБ")
        else:
            report.append(f"CUDA доступен: НЕТ")
        report.append("")
        
        # Метрики производительности
        if self.performance_metrics:
            report.append("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
            report.append("-" * 40)
            report.append(f"Время выполнения: {self.performance_metrics.execution_time:.2f} сек")
            report.append(f"Использование памяти: {self.performance_metrics.memory_usage:.2f} МБ")
            report.append(f"Использование GPU: {self.performance_metrics.gpu_utilization:.2f} МБ")
            report.append(f"Использование CPU: {self.performance_metrics.cpu_utilization:.2f}%")
            report.append(f"Количество итераций: {self.performance_metrics.iterations}")
            report.append(f"Скорость сходимости: {self.performance_metrics.convergence_rate:.6f}")
            report.append(f"Пропускная способность: {self.performance_metrics.throughput:.2e} оп/сек")
            report.append("")
        
        # Сводка профилирования
        profile_summary = self.profiler.get_profile_summary()
        if profile_summary:
            report.append("ПРОФИЛИРОВАНИЕ:")
            report.append("-" * 40)
            report.append(f"Общее время выполнения: {profile_summary['total_execution_time']:.2f} сек")
            report.append(f"Изменение памяти: {profile_summary['total_memory_delta']:.2f} МБ")
            report.append(f"Количество профилей: {profile_summary['profile_count']}")
            report.append("")
            
            report.append("ДЕТАЛЬНОЕ ПРОФИЛИРОВАНИЕ:")
            for name, profile in profile_summary['profiles'].items():
                report.append(f"  {name}: {profile['execution_time']:.3f} сек")
            report.append("")
        
        # Конфигурация оптимизации
        report.append("КОНФИГУРАЦИЯ ОПТИМИЗАЦИИ:")
        report.append("-" * 40)
        report.append(f"Использование CUDA: {'ДА' if self.optimization_config.use_cuda else 'НЕТ'}")
        report.append(f"Уровень оптимизации: {self.optimization_config.optimization_level.value}")
        report.append(f"Оптимизация памяти: {'ДА' if self.optimization_config.memory_optimization else 'НЕТ'}")
        report.append(f"Оптимизация алгоритмов: {'ДА' if self.optimization_config.algorithm_optimization else 'НЕТ'}")
        report.append(f"Адаптивные параметры: {'ДА' if self.optimization_config.adaptive_parameters else 'НЕТ'}")
        report.append(f"Профилирование: {'ДА' if self.optimization_config.profiling_enabled else 'НЕТ'}")
        report.append(f"Кэширование: {'ДА' if self.optimization_config.cache_enabled else 'НЕТ'}")
        report.append(f"Параллельная обработка: {'ДА' if self.optimization_config.parallel_processing else 'НЕТ'}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


class OptimizationSuite:
    """Набор оптимизаций."""
    
    def __init__(self):
        """Инициализация набора оптимизаций."""
        self.optimization_levels = {
            OptimizationLevel.NONE: OptimizationConfig(
                use_cuda=False,
                optimization_level=OptimizationLevel.NONE,
                memory_optimization=False,
                algorithm_optimization=False,
                adaptive_parameters=False,
                profiling_enabled=False,
                cache_enabled=False,
                parallel_processing=False
            ),
            OptimizationLevel.BASIC: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.BASIC,
                memory_optimization=True,
                algorithm_optimization=False,
                adaptive_parameters=False,
                profiling_enabled=True,
                cache_enabled=False,
                parallel_processing=False
            ),
            OptimizationLevel.ADVANCED: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.ADVANCED,
                memory_optimization=True,
                algorithm_optimization=True,
                adaptive_parameters=True,
                profiling_enabled=True,
                cache_enabled=True,
                parallel_processing=True
            ),
            OptimizationLevel.MAXIMUM: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.MAXIMUM,
                memory_optimization=True,
                algorithm_optimization=True,
                adaptive_parameters=True,
                profiling_enabled=True,
                cache_enabled=True,
                parallel_processing=True
            )
        }
    
    def run_optimization_comparison(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Сравнение различных уровней оптимизации.
        
        Args:
            config: Конфигурация модели
            
        Returns:
            Результаты сравнения
        """
        comparison_results = {}
        
        for level, optimization_config in self.optimization_levels.items():
            print(f"Running optimization level: {level.value}")
            
            # Создание оптимизированной модели
            optimized_model = OptimizedProtonModel(config, optimization_config)
            
            # Запуск модели
            start_time = time.time()
            results = optimized_model.run_optimized()
            end_time = time.time()
            
            # Сохранение результатов
            comparison_results[level.value] = {
                'execution_time': end_time - start_time,
                'performance_metrics': results['performance_metrics'],
                'device_info': results['device_info'],
                'optimization_report': optimized_model.get_optimization_report()
            }
        
        return comparison_results
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Генерация отчета сравнения.
        
        Args:
            comparison_results: Результаты сравнения
            
        Returns:
            Отчет сравнения
        """
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ СРАВНЕНИЯ ОПТИМИЗАЦИЙ")
        report.append("=" * 80)
        
        # Таблица сравнения
        report.append("СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
        report.append("-" * 80)
        report.append(f"{'Уровень':<15} {'Время (сек)':<12} {'Память (МБ)':<12} {'GPU (МБ)':<10} {'Пропускная способность':<20}")
        report.append("-" * 80)
        
        for level, results in comparison_results.items():
            metrics = results['performance_metrics']
            report.append(f"{level:<15} {metrics.execution_time:<12.2f} {metrics.memory_usage:<12.2f} "
                         f"{metrics.gpu_utilization:<10.2f} {metrics.throughput:<20.2e}")
        
        report.append("")
        
        # Анализ результатов
        report.append("АНАЛИЗ РЕЗУЛЬТАТОВ:")
        report.append("-" * 40)
        
        # Нахождение лучших результатов
        best_time = min(results['performance_metrics'].execution_time 
                       for results in comparison_results.values())
        best_memory = min(results['performance_metrics'].memory_usage 
                         for results in comparison_results.values())
        best_throughput = max(results['performance_metrics'].throughput 
                             for results in comparison_results.values())
        
        for level, results in comparison_results.items():
            metrics = results['performance_metrics']
            if metrics.execution_time == best_time:
                report.append(f"Лучшее время выполнения: {level}")
            if metrics.memory_usage == best_memory:
                report.append(f"Лучшее использование памяти: {level}")
            if metrics.throughput == best_throughput:
                report.append(f"Лучшая пропускная способность: {level}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_optimization_benchmark():
    """Запуск бенчмарка оптимизации."""
    # Создание конфигурации
    config = ModelConfig(
        grid_size=64,
        box_size=4.0,
        max_iterations=500,
        validation_enabled=True
    )
    
    # Создание набора оптимизаций
    optimization_suite = OptimizationSuite()
    
    # Запуск сравнения
    comparison_results = optimization_suite.run_optimization_comparison(config)
    
    # Генерация отчета
    report = optimization_suite.generate_comparison_report(comparison_results)
    
    print(report)
    
    return comparison_results


if __name__ == "__main__":
    run_optimization_benchmark()
```

## Объяснение

### CUDA оптимизация

Класс `CUDAManager` обеспечивает управление CUDA операциями и перенос данных между CPU и GPU.

### Оптимизация памяти

Класс `MemoryOptimizer` оптимизирует использование памяти и обеспечивает кэширование результатов.

### Оптимизация алгоритмов

Класс `AlgorithmOptimizer` предоставляет оптимизированные версии основных алгоритмов для GPU и CPU.

### Адаптивная оптимизация

Класс `AdaptiveParameterOptimizer` автоматически настраивает параметры для достижения лучшей производительности.

### Профилирование

Класс `PerformanceProfiler` обеспечивает детальное профилирование производительности всех компонентов.

### Оптимизированная модель

Класс `OptimizedProtonModel` объединяет все оптимизации и предоставляет единый интерфейс для работы с оптимизированной моделью.

## Заключение

Система оптимизации обеспечивает:
- Значительное ускорение вычислений при использовании GPU
- Оптимизацию использования памяти
- Адаптивную настройку параметров
- Детальное профилирование производительности
- Сравнение различных уровней оптимизации

Это завершает полный цикл разработки модели протона от математических основ до оптимизированной реализации.
