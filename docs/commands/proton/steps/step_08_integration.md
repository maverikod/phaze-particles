# Шаг 8: Интеграция

## Цель

Объединить все компоненты модели протона в единую интегрированную систему:
- Интеграция всех модулей (математические основы, геометрии, поля, энергии, физические величины, численные методы, валидация)
- Создание единого интерфейса для работы с моделью
- Реализация конфигурационной системы
- Обеспечение взаимодействия между компонентами

## Обзор

Интеграция является ключевым этапом, который объединяет все разработанные компоненты в единую рабочую систему. Она включает:
- Создание главного класса модели протона
- Интеграцию всех модулей
- Реализацию единого API
- Управление конфигурацией
- Координацию работы компонентов

## Архитектура интеграции

### 8.1 Структура системы

```
ProtonModel
├── MathematicalFoundations
├── TorusGeometries
├── SU2Fields
├── EnergyDensities
├── PhysicalQuantities
├── NumericalMethods
└── ValidationSystem
```

### 8.2 Поток данных

1. **Инициализация** → Загрузка конфигурации
2. **Геометрия** → Создание тороидальных структур
3. **Поля** → Построение SU(2) полей
4. **Энергия** → Вычисление плотности энергии
5. **Физика** → Расчет физических величин
6. **Оптимизация** → Релаксация к стационарному решению
7. **Валидация** → Проверка результатов

## Декларативный код

```python
#!/usr/bin/env python3
"""
Интеграция модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
from pathlib import Path

# Импорт всех модулей
from .step_01_mathematical_foundations import MathematicalFoundations
from .step_02_torus_geometries import TorusGeometries, TorusConfiguration
from .step_03_su2_fields import SU2Field, SU2FieldBuilder
from .step_04_energy_densities import EnergyDensity, EnergyDensityCalculator
from .step_05_physical_quantities import PhysicalQuantities, PhysicalQuantitiesCalculator
from .step_06_numerical_methods import RelaxationSolver, RelaxationConfig, ConstraintConfig
from .step_07_validation import ValidationSystem, ExperimentalData, CalculatedData


class ModelStatus(Enum):
    """Статус модели."""
    INITIALIZED = "initialized"
    GEOMETRY_CREATED = "geometry_created"
    FIELDS_BUILT = "fields_built"
    ENERGY_CALCULATED = "energy_calculated"
    PHYSICS_CALCULATED = "physics_calculated"
    OPTIMIZED = "optimized"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass
class ModelConfig:
    """Конфигурация модели протона."""
    
    # Геометрические параметры
    grid_size: int = 64
    box_size: float = 4.0
    torus_config: str = "120deg"
    R_torus: float = 1.0
    r_torus: float = 0.2
    
    # Параметры профилей
    profile_type: str = "tanh"
    f_0: float = np.pi
    f_inf: float = 0.0
    r_scale: float = 1.0
    
    # Константы Skyrme
    c2: float = 1.0
    c4: float = 1.0
    c6: float = 1.0
    
    # Параметры релаксации
    max_iterations: int = 1000
    convergence_tol: float = 1e-6
    step_size: float = 0.01
    relaxation_method: str = "gradient_descent"
    
    # Параметры ограничений
    lambda_B: float = 1000.0
    lambda_Q: float = 1000.0
    lambda_virial: float = 1000.0
    
    # Параметры валидации
    validation_enabled: bool = True
    save_reports: bool = True
    output_dir: str = "results"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ModelConfig':
        """
        Загрузка конфигурации из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Конфигурация модели
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: str):
        """
        Сохранение конфигурации в файл.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Валидация конфигурации.
        
        Returns:
            Список ошибок валидации
        """
        errors = []
        
        if self.grid_size <= 0:
            errors.append("grid_size must be positive")
        
        if self.box_size <= 0:
            errors.append("box_size must be positive")
        
        if self.torus_config not in ["120deg", "clover", "cartesian"]:
            errors.append("torus_config must be one of: 120deg, clover, cartesian")
        
        if self.R_torus <= 0:
            errors.append("R_torus must be positive")
        
        if self.r_torus <= 0:
            errors.append("r_torus must be positive")
        
        if self.c2 <= 0 or self.c4 <= 0 or self.c6 <= 0:
            errors.append("Skyrme constants must be positive")
        
        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        
        if self.convergence_tol <= 0:
            errors.append("convergence_tol must be positive")
        
        return errors


@dataclass
class ModelResults:
    """Результаты модели протона."""
    
    # Основные результаты
    status: ModelStatus
    execution_time: float
    iterations: int
    converged: bool
    
    # Физические параметры
    proton_mass: float
    charge_radius: float
    magnetic_moment: float
    electric_charge: float
    baryon_number: float
    energy_balance: float
    total_energy: float
    
    # Результаты валидации
    validation_status: Optional[str] = None
    validation_score: Optional[float] = None
    
    # Дополнительная информация
    config: Optional[ModelConfig] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.
        
        Returns:
            Словарь с результатами
        """
        return asdict(self)
    
    def save_to_file(self, output_path: str):
        """
        Сохранение результатов в файл.
        
        Args:
            output_path: Путь к файлу результатов
        """
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ProtonModel:
    """Основной класс модели протона."""
    
    def __init__(self, config: ModelConfig):
        """
        Инициализация модели протона.
        
        Args:
            config: Конфигурация модели
        """
        self.config = config
        self.status = ModelStatus.INITIALIZED
        
        # Валидация конфигурации
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Результаты
        self.results: Optional[ModelResults] = None
        self.error_message: Optional[str] = None
    
    def _initialize_components(self):
        """Инициализация всех компонентов модели."""
        try:
            # Математические основы
            self.math_foundations = MathematicalFoundations(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size
            )
            
            # Тороидальные геометрии
            self.torus_geometries = TorusGeometries(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size
            )
            
            # SU(2) поля
            self.su2_field_builder = SU2FieldBuilder(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size
            )
            
            # Плотности энергии
            self.energy_calculator = EnergyDensityCalculator(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                c2=self.config.c2,
                c4=self.config.c4,
                c6=self.config.c6
            )
            
            # Физические величины
            self.physics_calculator = PhysicalQuantitiesCalculator(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size
            )
            
            # Численные методы
            relaxation_config = RelaxationConfig(
                method=self.config.relaxation_method,
                max_iterations=self.config.max_iterations,
                convergence_tol=self.config.convergence_tol,
                step_size=self.config.step_size
            )
            
            constraint_config = ConstraintConfig(
                lambda_B=self.config.lambda_B,
                lambda_Q=self.config.lambda_Q,
                lambda_virial=self.config.lambda_virial
            )
            
            self.relaxation_solver = RelaxationSolver(
                relaxation_config,
                constraint_config
            )
            
            # Система валидации
            if self.config.validation_enabled:
                experimental_data = ExperimentalData()
                self.validation_system = ValidationSystem(experimental_data)
            else:
                self.validation_system = None
            
            print("Все компоненты модели успешно инициализированы")
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка инициализации: {str(e)}"
            raise
    
    def create_geometry(self) -> bool:
        """
        Создание тороидальной геометрии.
        
        Returns:
            True если успешно
        """
        try:
            # Определение конфигурации торов
            if self.config.torus_config == "120deg":
                config_type = TorusConfiguration.CONFIG_120_DEG
            elif self.config.torus_config == "clover":
                config_type = TorusConfiguration.CONFIG_CLOVER
            elif self.config.torus_config == "cartesian":
                config_type = TorusConfiguration.CONFIG_CARTESIAN
            else:
                raise ValueError(f"Unknown torus configuration: {self.config.torus_config}")
            
            # Создание геометрии
            self.field_direction = self.torus_geometries.create_field_direction(
                config_type=config_type,
                radius=self.config.R_torus,
                thickness=self.config.r_torus
            )
            
            self.status = ModelStatus.GEOMETRY_CREATED
            print(f"Тороидальная геометрия создана: {self.config.torus_config}")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка создания геометрии: {str(e)}"
            return False
    
    def build_fields(self) -> bool:
        """
        Построение SU(2) полей.
        
        Returns:
            True если успешно
        """
        try:
            if self.status != ModelStatus.GEOMETRY_CREATED:
                raise ValueError("Geometry must be created first")
            
            # Построение SU(2) поля
            self.su2_field = self.su2_field_builder.build_field(
                field_direction=self.field_direction,
                profile_type=self.config.profile_type,
                f_0=self.config.f_0,
                f_inf=self.config.f_inf,
                r_scale=self.config.r_scale
            )
            
            self.status = ModelStatus.FIELDS_BUILT
            print("SU(2) поля построены")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка построения полей: {str(e)}"
            return False
    
    def calculate_energy(self) -> bool:
        """
        Вычисление плотности энергии.
        
        Returns:
            True если успешно
        """
        try:
            if self.status != ModelStatus.FIELDS_BUILT:
                raise ValueError("Fields must be built first")
            
            # Вычисление плотности энергии
            self.energy_density = self.energy_calculator.calculate_energy_density(
                su2_field=self.su2_field
            )
            
            self.status = ModelStatus.ENERGY_CALCULATED
            print("Плотность энергии вычислена")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка вычисления энергии: {str(e)}"
            return False
    
    def calculate_physics(self) -> bool:
        """
        Вычисление физических величин.
        
        Returns:
            True если успешно
        """
        try:
            if self.status != ModelStatus.ENERGY_CALCULATED:
                raise ValueError("Energy must be calculated first")
            
            # Вычисление физических величин
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field,
                energy_density=self.energy_density
            )
            
            self.status = ModelStatus.PHYSICS_CALCULATED
            print("Физические величины вычислены")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка вычисления физики: {str(e)}"
            return False
    
    def optimize(self) -> bool:
        """
        Оптимизация модели.
        
        Returns:
            True если успешно
        """
        try:
            if self.status != ModelStatus.PHYSICS_CALCULATED:
                raise ValueError("Physics must be calculated first")
            
            # Функции для оптимизации
            def energy_function(U):
                return self.energy_calculator.calculate_total_energy(U)
            
            def gradient_function(U):
                return self.energy_calculator.calculate_gradient(U)
            
            def constraint_functions(U):
                return {
                    'baryon_number': self.physics_calculator.calculate_baryon_number(U),
                    'electric_charge': self.physics_calculator.calculate_electric_charge(U),
                    'energy_balance': self.energy_calculator.calculate_energy_balance(U)
                }
            
            # Релаксация
            optimization_results = self.relaxation_solver.solve(
                U_init=self.su2_field,
                energy_function=energy_function,
                gradient_function=gradient_function,
                constraint_functions=constraint_functions
            )
            
            # Обновление поля
            self.su2_field = optimization_results['solution']
            
            # Пересчет физических величин
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field,
                energy_density=self.energy_density
            )
            
            self.status = ModelStatus.OPTIMIZED
            self.optimization_results = optimization_results
            print(f"Модель оптимизирована за {optimization_results['iterations']} итераций")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка оптимизации: {str(e)}"
            return False
    
    def validate(self) -> bool:
        """
        Валидация модели.
        
        Returns:
            True если успешно
        """
        try:
            if self.status != ModelStatus.OPTIMIZED:
                raise ValueError("Model must be optimized first")
            
            if not self.validation_system:
                print("Валидация отключена")
                return True
            
            # Подготовка данных для валидации
            calculated_data = CalculatedData(
                proton_mass=self.physical_quantities.mass,
                charge_radius=self.physical_quantities.charge_radius,
                magnetic_moment=self.physical_quantities.magnetic_moment,
                electric_charge=self.physical_quantities.electric_charge,
                baryon_number=self.physical_quantities.baryon_number,
                energy_balance=self.physical_quantities.energy_balance,
                total_energy=self.physical_quantities.energy,
                execution_time=self.optimization_results['execution_time']
            )
            
            # Валидация
            self.validation_results = self.validation_system.validate_model(calculated_data)
            
            # Сохранение отчетов
            if self.config.save_reports:
                self.validation_system.save_reports(
                    self.validation_results,
                    self.config.output_dir
                )
            
            self.status = ModelStatus.VALIDATED
            print(f"Модель валидирована. Статус: {self.validation_results['overall_status'].value}")
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Ошибка валидации: {str(e)}"
            return False
    
    def run(self) -> ModelResults:
        """
        Запуск полного цикла модели.
        
        Returns:
            Результаты модели
        """
        start_time = time.time()
        
        try:
            # Создание геометрии
            if not self.create_geometry():
                raise RuntimeError("Failed to create geometry")
            
            # Построение полей
            if not self.build_fields():
                raise RuntimeError("Failed to build fields")
            
            # Вычисление энергии
            if not self.calculate_energy():
                raise RuntimeError("Failed to calculate energy")
            
            # Вычисление физики
            if not self.calculate_physics():
                raise RuntimeError("Failed to calculate physics")
            
            # Оптимизация
            if not self.optimize():
                raise RuntimeError("Failed to optimize")
            
            # Валидация
            if not self.validate():
                raise RuntimeError("Failed to validate")
            
            # Создание результатов
            self.results = ModelResults(
                status=self.status,
                execution_time=time.time() - start_time,
                iterations=self.optimization_results['iterations'],
                converged=self.optimization_results['converged'],
                proton_mass=self.physical_quantities.mass,
                charge_radius=self.physical_quantities.charge_radius,
                magnetic_moment=self.physical_quantities.magnetic_moment,
                electric_charge=self.physical_quantities.electric_charge,
                baryon_number=self.physical_quantities.baryon_number,
                energy_balance=self.physical_quantities.energy_balance,
                total_energy=self.physical_quantities.energy,
                validation_status=self.validation_results['overall_status'].value if self.validation_system else None,
                validation_score=self.validation_results['weighted_score'] if self.validation_system else None,
                config=self.config
            )
            
            print("Модель протона успешно выполнена")
            return self.results
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = str(e)
            
            self.results = ModelResults(
                status=self.status,
                execution_time=time.time() - start_time,
                iterations=0,
                converged=False,
                proton_mass=0.0,
                charge_radius=0.0,
                magnetic_moment=0.0,
                electric_charge=0.0,
                baryon_number=0.0,
                energy_balance=0.0,
                total_energy=0.0,
                config=self.config,
                error_message=self.error_message
            )
            
            print(f"Ошибка выполнения модели: {self.error_message}")
            return self.results
    
    def get_status(self) -> ModelStatus:
        """
        Получение текущего статуса модели.
        
        Returns:
            Текущий статус
        """
        return self.status
    
    def get_results(self) -> Optional[ModelResults]:
        """
        Получение результатов модели.
        
        Returns:
            Результаты модели или None
        """
        return self.results
    
    def save_results(self, output_path: str):
        """
        Сохранение результатов в файл.
        
        Args:
            output_path: Путь к файлу результатов
        """
        if self.results:
            self.results.save_to_file(output_path)
        else:
            raise ValueError("No results to save")
    
    def reset(self):
        """Сброс модели к начальному состоянию."""
        self.status = ModelStatus.INITIALIZED
        self.results = None
        self.error_message = None
        
        # Сброс компонентов
        if hasattr(self, 'relaxation_solver'):
            self.relaxation_solver.reset()
        
        print("Модель сброшена к начальному состоянию")


class ProtonModelFactory:
    """Фабрика для создания моделей протона."""
    
    @staticmethod
    def create_from_config(config_path: str) -> ProtonModel:
        """
        Создание модели из конфигурационного файла.
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Модель протона
        """
        config = ModelConfig.from_file(config_path)
        return ProtonModel(config)
    
    @staticmethod
    def create_default() -> ProtonModel:
        """
        Создание модели с конфигурацией по умолчанию.
        
        Returns:
            Модель протона
        """
        config = ModelConfig()
        return ProtonModel(config)
    
    @staticmethod
    def create_quick_test() -> ProtonModel:
        """
        Создание модели для быстрого тестирования.
        
        Returns:
            Модель протона
        """
        config = ModelConfig(
            grid_size=32,
            box_size=2.0,
            max_iterations=100,
            validation_enabled=False
        )
        return ProtonModel(config)
```

## Объяснение

### Основной класс модели

Класс `ProtonModel` является центральным компонентом, который координирует работу всех модулей и обеспечивает единый интерфейс для работы с моделью.

### Конфигурация

Класс `ModelConfig` обеспечивает управление всеми параметрами модели с возможностью загрузки и сохранения конфигурации.

### Результаты

Класс `ModelResults` содержит все результаты выполнения модели и обеспечивает их сохранение.

### Фабрика

Класс `ProtonModelFactory` предоставляет удобные методы для создания моделей с различными конфигурациями.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 9: Тестирование**, где создадим комплексную систему тестирования для всех компонентов модели.
