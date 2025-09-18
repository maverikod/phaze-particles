# Шаг 9: Тестирование

## Цель

Создать комплексную систему тестирования для модели протона:
- Unit-тесты для всех компонентов
- Интеграционные тесты для взаимодействия модулей
- Performance-тесты для оценки производительности
- Stress-тесты для проверки стабильности
- Валидационные тесты для проверки физической корректности

## Обзор

Тестирование является критически важным этапом для обеспечения качества и надежности модели. Оно включает:
- Автоматизированное тестирование всех компонентов
- Проверку корректности вычислений
- Оценку производительности
- Валидацию физических результатов
- Проверку граничных случаев

## Типы тестов

### 9.1 Unit-тесты

- Тестирование отдельных функций и методов
- Проверка корректности вычислений
- Валидация входных и выходных данных
- Проверка обработки ошибок

### 9.2 Интеграционные тесты

- Тестирование взаимодействия между модулями
- Проверка потока данных
- Валидация интерфейсов
- Проверка конфигурации

### 9.3 Performance-тесты

- Измерение времени выполнения
- Оценка использования памяти
- Проверка масштабируемости
- Оптимизация производительности

### 9.4 Stress-тесты

- Тестирование при экстремальных параметрах
- Проверка стабильности при длительной работе
- Валидация обработки ошибок
- Проверка восстановления после сбоев

## Декларативный код

```python
#!/usr/bin/env python3
"""
Система тестирования модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
import time
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

# Импорт всех модулей для тестирования
from .step_01_mathematical_foundations import MathematicalFoundations, PhysicalConstants, PauliMatrices
from .step_02_torus_geometries import TorusGeometries, TorusConfiguration
from .step_03_su2_fields import SU2Field, SU2FieldBuilder
from .step_04_energy_densities import EnergyDensity, EnergyDensityCalculator
from .step_05_physical_quantities import PhysicalQuantities, PhysicalQuantitiesCalculator
from .step_06_numerical_methods import RelaxationSolver, RelaxationConfig, ConstraintConfig
from .step_07_validation import ValidationSystem, ExperimentalData
from .step_08_integration import ProtonModel, ModelConfig, ProtonModelFactory


@dataclass
class TestConfig:
    """Конфигурация тестирования."""
    grid_size: int = 32
    box_size: float = 2.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    performance_threshold: float = 1.0  # секунды
    memory_threshold: float = 100.0  # МБ


class TestMathematicalFoundations(unittest.TestCase):
    """Тесты математических основ."""
    
    def setUp(self):
        """Настройка тестов."""
        self.math_foundations = MathematicalFoundations(grid_size=32, box_size=2.0)
        self.tolerance = 1e-10
    
    def test_physical_constants(self):
        """Тест физических констант."""
        constants = self.math_foundations.get_physical_constants()
        
        # Проверка основных констант
        self.assertEqual(constants['proton_charge'], 1.0)
        self.assertEqual(constants['baryon_number'], 1.0)
        self.assertAlmostEqual(constants['proton_mass_mev'], 938.272, places=3)
        self.assertAlmostEqual(constants['charge_radius_fm'], 0.841, places=3)
        self.assertAlmostEqual(constants['magnetic_moment_mu_n'], 2.793, places=3)
    
    def test_pauli_matrices(self):
        """Тест матриц Паули."""
        # Проверка свойств матриц Паули
        sigma1 = PauliMatrices.get_sigma(1)
        sigma2 = PauliMatrices.get_sigma(2)
        sigma3 = PauliMatrices.get_sigma(3)
        
        # Проверка эрмитовости
        self.assertTrue(np.allclose(sigma1, sigma1.conj().T))
        self.assertTrue(np.allclose(sigma2, sigma2.conj().T))
        self.assertTrue(np.allclose(sigma3, sigma3.conj().T))
        
        # Проверка коммутационных соотношений
        [sigma1, sigma2] = np.dot(sigma1, sigma2) - np.dot(sigma2, sigma1)
        self.assertTrue(np.allclose([sigma1, sigma2], 2j * sigma3))
    
    def test_coordinate_system(self):
        """Тест системы координат."""
        coords = self.math_foundations.coords
        
        # Проверка размеров
        self.assertEqual(coords.X.shape, (32, 32, 32))
        self.assertEqual(coords.Y.shape, (32, 32, 32))
        self.assertEqual(coords.Z.shape, (32, 32, 32))
        
        # Проверка диапазонов
        self.assertAlmostEqual(coords.X.min(), -1.0, places=6)
        self.assertAlmostEqual(coords.X.max(), 1.0, places=6)
        self.assertAlmostEqual(coords.Y.min(), -1.0, places=6)
        self.assertAlmostEqual(coords.Y.max(), 1.0, places=6)
        self.assertAlmostEqual(coords.Z.min(), -1.0, places=6)
        self.assertAlmostEqual(coords.Z.max(), 1.0, places=6)
        
        # Проверка радиальной координаты
        self.assertTrue(np.all(coords.R >= 0))
        self.assertAlmostEqual(coords.R[16, 16, 16], 0.0, places=6)
    
    def test_numerical_utils(self):
        """Тест численных утилит."""
        # Создание тестового поля
        field = np.sin(self.math_foundations.coords.X)
        
        # Вычисление градиента
        grad_x, grad_y, grad_z = self.math_foundations.numerical.gradient_3d(
            field, self.math_foundations.coords.dx
        )
        
        # Проверка размеров
        self.assertEqual(grad_x.shape, field.shape)
        self.assertEqual(grad_y.shape, field.shape)
        self.assertEqual(grad_z.shape, field.shape)
        
        # Проверка аналитического градиента
        analytical_grad_x = np.cos(self.math_foundations.coords.X)
        self.assertTrue(np.allclose(grad_x, analytical_grad_x, atol=1e-6))
    
    def test_validation_utils(self):
        """Тест утилит валидации."""
        # Создание тестовой SU(2) матрицы
        theta = np.pi/4
        U = np.array([
            [np.cos(theta), 1j * np.sin(theta)],
            [1j * np.sin(theta), np.cos(theta)]
        ])
        
        # Проверка валидации
        self.assertTrue(self.math_foundations.validation.check_su2_matrix(U))
        
        # Проверка физических границ
        self.assertTrue(self.math_foundations.validation.check_physical_bounds(
            1.0, 1.0, 0.1
        ))
        self.assertFalse(self.math_foundations.validation.check_physical_bounds(
            1.0, 1.0, 0.01
        ))


class TestTorusGeometries(unittest.TestCase):
    """Тесты тороидальных геометрий."""
    
    def setUp(self):
        """Настройка тестов."""
        self.torus_geometries = TorusGeometries(grid_size=32, box_size=2.0)
    
    def test_120_degree_configuration(self):
        """Тест 120° конфигурации."""
        n_x, n_y, n_z = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_120_DEG
        )
        
        # Проверка размеров
        self.assertEqual(n_x.shape, (32, 32, 32))
        self.assertEqual(n_y.shape, (32, 32, 32))
        self.assertEqual(n_z.shape, (32, 32, 32))
        
        # Проверка нормализации
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        self.assertTrue(np.allclose(norm, 1.0, atol=1e-10))
    
    def test_clover_configuration(self):
        """Тест клевер конфигурации."""
        n_x, n_y, n_z = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_CLOVER
        )
        
        # Проверка размеров
        self.assertEqual(n_x.shape, (32, 32, 32))
        self.assertEqual(n_y.shape, (32, 32, 32))
        self.assertEqual(n_z.shape, (32, 32, 32))
        
        # Проверка нормализации
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        self.assertTrue(np.allclose(norm, 1.0, atol=1e-10))
    
    def test_cartesian_configuration(self):
        """Тест декартовой конфигурации."""
        n_x, n_y, n_z = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_CARTESIAN
        )
        
        # Проверка размеров
        self.assertEqual(n_x.shape, (32, 32, 32))
        self.assertEqual(n_y.shape, (32, 32, 32))
        self.assertEqual(n_z.shape, (32, 32, 32))
        
        # Проверка нормализации
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        self.assertTrue(np.allclose(norm, 1.0, atol=1e-10))
    
    def test_configuration_info(self):
        """Тест информации о конфигурациях."""
        for config_type in TorusConfiguration:
            info = self.torus_geometries.get_configuration_info(config_type)
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('symmetry_group', info)
            self.assertIn('num_tori', info)


class TestSU2Fields(unittest.TestCase):
    """Тесты SU(2) полей."""
    
    def setUp(self):
        """Настройка тестов."""
        self.su2_builder = SU2FieldBuilder(grid_size=32, box_size=2.0)
        
        # Создание тестового направления поля
        self.torus_geometries = TorusGeometries(grid_size=32, box_size=2.0)
        self.field_direction = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_120_DEG
        )
    
    def test_field_building(self):
        """Тест построения поля."""
        su2_field = self.su2_builder.build_field(
            field_direction=self.field_direction,
            profile_type="tanh"
        )
        
        # Проверка размеров
        self.assertEqual(su2_field.u_00.shape, (32, 32, 32))
        self.assertEqual(su2_field.u_01.shape, (32, 32, 32))
        self.assertEqual(su2_field.u_10.shape, (32, 32, 32))
        self.assertEqual(su2_field.u_11.shape, (32, 32, 32))
        
        # Проверка SU(2) свойств
        self.assertTrue(su2_field._is_su2_field())
    
    def test_field_validation(self):
        """Тест валидации поля."""
        su2_field = self.su2_builder.build_field(
            field_direction=self.field_direction,
            profile_type="tanh"
        )
        
        # Проверка валидации
        self.assertTrue(su2_field._is_su2_field())
        
        # Проверка граничных условий
        center_field = np.array([
            [su2_field.u_00[16, 16, 16], su2_field.u_01[16, 16, 16]],
            [su2_field.u_10[16, 16, 16], su2_field.u_11[16, 16, 16]]
        ])
        
        self.assertAlmostEqual(np.linalg.det(center_field), -1.0, places=6)
    
    def test_different_profiles(self):
        """Тест различных профилей."""
        profiles = ["tanh", "exp", "gaussian"]
        
        for profile in profiles:
            su2_field = self.su2_builder.build_field(
                field_direction=self.field_direction,
                profile_type=profile
            )
            
            # Проверка SU(2) свойств
            self.assertTrue(su2_field._is_su2_field())


class TestEnergyDensities(unittest.TestCase):
    """Тесты плотностей энергии."""
    
    def setUp(self):
        """Настройка тестов."""
        self.energy_calculator = EnergyDensityCalculator(
            grid_size=32, box_size=2.0, c2=1.0, c4=1.0, c6=1.0
        )
        
        # Создание тестового поля
        self.su2_builder = SU2FieldBuilder(grid_size=32, box_size=2.0)
        self.torus_geometries = TorusGeometries(grid_size=32, box_size=2.0)
        self.field_direction = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_120_DEG
        )
        self.su2_field = self.su2_builder.build_field(
            field_direction=self.field_direction,
            profile_type="tanh"
        )
    
    def test_energy_calculation(self):
        """Тест вычисления энергии."""
        energy_density = self.energy_calculator.calculate_energy_density(
            self.su2_field
        )
        
        # Проверка размеров
        self.assertEqual(energy_density.c2_term.shape, (32, 32, 32))
        self.assertEqual(energy_density.c4_term.shape, (32, 32, 32))
        self.assertEqual(energy_density.c6_term.shape, (32, 32, 32))
        self.assertEqual(energy_density.total_density.shape, (32, 32, 32))
        
        # Проверка положительности
        self.assertTrue(np.all(energy_density.c2_term >= 0))
        self.assertTrue(np.all(energy_density.c4_term >= 0))
        self.assertTrue(np.all(energy_density.c6_term >= 0))
        self.assertTrue(np.all(energy_density.total_density >= 0))
    
    def test_energy_components(self):
        """Тест компонент энергии."""
        energy_density = self.energy_calculator.calculate_energy_density(
            self.su2_field
        )
        
        components = energy_density.get_energy_components()
        
        # Проверка наличия всех компонент
        self.assertIn('E2', components)
        self.assertIn('E4', components)
        self.assertIn('E6', components)
        self.assertIn('E_total', components)
        
        # Проверка положительности
        self.assertGreater(components['E2'], 0)
        self.assertGreater(components['E4'], 0)
        self.assertGreater(components['E6'], 0)
        self.assertGreater(components['E_total'], 0)
    
    def test_energy_balance(self):
        """Тест энергетического баланса."""
        energy_density = self.energy_calculator.calculate_energy_density(
            self.su2_field
        )
        
        balance = energy_density.get_energy_balance()
        
        # Проверка наличия всех компонент баланса
        self.assertIn('E2_fraction', balance)
        self.assertIn('E4_fraction', balance)
        self.assertIn('E6_fraction', balance)
        
        # Проверка суммирования к 1
        total_fraction = (balance['E2_fraction'] + 
                         balance['E4_fraction'] + 
                         balance['E6_fraction'])
        self.assertAlmostEqual(total_fraction, 1.0, places=6)


class TestPhysicalQuantities(unittest.TestCase):
    """Тесты физических величин."""
    
    def setUp(self):
        """Настройка тестов."""
        self.physics_calculator = PhysicalQuantitiesCalculator(
            grid_size=32, box_size=2.0
        )
        
        # Создание тестового поля и энергии
        self.su2_builder = SU2FieldBuilder(grid_size=32, box_size=2.0)
        self.torus_geometries = TorusGeometries(grid_size=32, box_size=2.0)
        self.field_direction = self.torus_geometries.create_field_direction(
            TorusConfiguration.CONFIG_120_DEG
        )
        self.su2_field = self.su2_builder.build_field(
            field_direction=self.field_direction,
            profile_type="tanh"
        )
        
        self.energy_calculator = EnergyDensityCalculator(
            grid_size=32, box_size=2.0, c2=1.0, c4=1.0, c6=1.0
        )
        self.energy_density = self.energy_calculator.calculate_energy_density(
            self.su2_field
        )
    
    def test_baryon_number(self):
        """Тест барионного числа."""
        baryon_number = self.physics_calculator.calculate_baryon_number(
            self.su2_field
        )
        
        # Проверка близости к 1
        self.assertAlmostEqual(baryon_number, 1.0, places=2)
    
    def test_electric_charge(self):
        """Тест электрического заряда."""
        electric_charge = self.physics_calculator.calculate_electric_charge(
            self.su2_field
        )
        
        # Проверка близости к 1
        self.assertAlmostEqual(electric_charge, 1.0, places=2)
    
    def test_charge_radius(self):
        """Тест радиуса зарядового распределения."""
        charge_radius = self.physics_calculator.calculate_charge_radius(
            self.su2_field
        )
        
        # Проверка положительности
        self.assertGreater(charge_radius, 0)
        
        # Проверка разумности значения
        self.assertLess(charge_radius, 2.0)  # Меньше размера коробки
    
    def test_magnetic_moment(self):
        """Тест магнитного момента."""
        magnetic_moment = self.physics_calculator.calculate_magnetic_moment(
            self.su2_field
        )
        
        # Проверка положительности
        self.assertGreater(magnetic_moment, 0)
        
        # Проверка разумности значения
        self.assertLess(magnetic_moment, 10.0)
    
    def test_physical_quantities(self):
        """Тест всех физических величин."""
        quantities = self.physics_calculator.calculate_quantities(
            self.su2_field, self.energy_density
        )
        
        # Проверка наличия всех величин
        self.assertIsNotNone(quantities.baryon_number)
        self.assertIsNotNone(quantities.electric_charge)
        self.assertIsNotNone(quantities.charge_radius)
        self.assertIsNotNone(quantities.magnetic_moment)
        self.assertIsNotNone(quantities.mass)
        self.assertIsNotNone(quantities.energy)
        
        # Проверка положительности
        self.assertGreater(quantities.charge_radius, 0)
        self.assertGreater(quantities.magnetic_moment, 0)
        self.assertGreater(quantities.mass, 0)
        self.assertGreater(quantities.energy, 0)


class TestNumericalMethods(unittest.TestCase):
    """Тесты численных методов."""
    
    def setUp(self):
        """Настройка тестов."""
        self.relaxation_config = RelaxationConfig(
            method="gradient_descent",
            max_iterations=100,
            convergence_tol=1e-6,
            step_size=0.01
        )
        
        self.constraint_config = ConstraintConfig(
            lambda_B=1000.0,
            lambda_Q=1000.0,
            lambda_virial=1000.0
        )
        
        self.solver = RelaxationSolver(
            self.relaxation_config,
            self.constraint_config
        )
    
    def test_solver_initialization(self):
        """Тест инициализации решателя."""
        self.assertIsNotNone(self.solver)
        self.assertEqual(self.solver.config.method, "gradient_descent")
        self.assertEqual(self.solver.config.max_iterations, 100)
    
    def test_constraint_controller(self):
        """Тест контроллера ограничений."""
        penalty = self.solver.constraint_controller.compute_constraint_penalty(
            U=None,  # Не используется в тесте
            baryon_number=1.0,
            electric_charge=1.0,
            energy_balance=0.5
        )
        
        # Проверка нулевого штрафа для правильных значений
        self.assertAlmostEqual(penalty, 0.0, places=6)
        
        # Проверка положительного штрафа для неправильных значений
        penalty_wrong = self.solver.constraint_controller.compute_constraint_penalty(
            U=None,
            baryon_number=2.0,
            electric_charge=0.5,
            energy_balance=0.3
        )
        self.assertGreater(penalty_wrong, 0)
    
    def test_constraint_checking(self):
        """Тест проверки ограничений."""
        constraints = self.solver.constraint_controller.check_constraints(
            baryon_number=1.0,
            electric_charge=1.0,
            energy_balance=0.5
        )
        
        # Проверка выполнения всех ограничений
        self.assertTrue(constraints['baryon_number'])
        self.assertTrue(constraints['electric_charge'])
        self.assertTrue(constraints['energy_balance'])


class TestValidationSystem(unittest.TestCase):
    """Тесты системы валидации."""
    
    def setUp(self):
        """Настройка тестов."""
        self.experimental_data = ExperimentalData()
        self.validation_system = ValidationSystem(self.experimental_data)
    
    def test_experimental_data(self):
        """Тест экспериментальных данных."""
        self.assertEqual(self.experimental_data.proton_mass, 938.272)
        self.assertEqual(self.experimental_data.charge_radius, 0.841)
        self.assertEqual(self.experimental_data.magnetic_moment, 2.793)
        self.assertEqual(self.experimental_data.electric_charge, 1.0)
        self.assertEqual(self.experimental_data.baryon_number, 1.0)
    
    def test_parameter_validator(self):
        """Тест валидатора параметров."""
        # Тест валидации массы
        result = self.validation_system.validator.validate_mass(938.272)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status.value, "excellent")
        
        # Тест валидации радиуса
        result = self.validation_system.validator.validate_radius(0.841)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status.value, "excellent")
        
        # Тест валидации магнитного момента
        result = self.validation_system.validator.validate_magnetic_moment(2.793)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status.value, "excellent")
    
    def test_quality_assessor(self):
        """Тест оценщика качества."""
        # Создание тестовых результатов валидации
        validation_results = [
            self.validation_system.validator.validate_mass(938.272),
            self.validation_system.validator.validate_radius(0.841),
            self.validation_system.validator.validate_magnetic_moment(2.793),
            self.validation_system.validator.validate_charge(1.0),
            self.validation_system.validator.validate_baryon_number(1.0),
            self.validation_system.validator.validate_energy_balance(0.5)
        ]
        
        # Оценка качества
        quality = self.validation_system.quality_assessor.assess_quality(validation_results)
        
        # Проверка результатов
        self.assertEqual(quality['overall_status'].value, "excellent")
        self.assertAlmostEqual(quality['weighted_score'], 1.0, places=3)
        self.assertEqual(quality['passed_parameters'], 6)
        self.assertEqual(quality['total_parameters'], 6)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты."""
    
    def setUp(self):
        """Настройка тестов."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig(
            grid_size=32,
            box_size=2.0,
            max_iterations=100,
            validation_enabled=False
        )
    
    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_creation(self):
        """Тест создания модели."""
        model = ProtonModel(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.get_status().value, "initialized")
    
    def test_model_execution(self):
        """Тест выполнения модели."""
        model = ProtonModel(self.config)
        results = model.run()
        
        # Проверка успешного выполнения
        self.assertIsNotNone(results)
        self.assertEqual(results.status.value, "optimized")
        self.assertTrue(results.converged)
        
        # Проверка физических параметров
        self.assertGreater(results.proton_mass, 0)
        self.assertGreater(results.charge_radius, 0)
        self.assertGreater(results.magnetic_moment, 0)
        self.assertGreater(results.electric_charge, 0)
        self.assertGreater(results.baryon_number, 0)
    
    def test_model_factory(self):
        """Тест фабрики моделей."""
        # Тест создания модели по умолчанию
        model = ProtonModelFactory.create_default()
        self.assertIsNotNone(model)
        
        # Тест создания модели для быстрого тестирования
        model = ProtonModelFactory.create_quick_test()
        self.assertIsNotNone(model)
        self.assertEqual(model.config.grid_size, 32)
        self.assertEqual(model.config.max_iterations, 100)
    
    def test_config_serialization(self):
        """Тест сериализации конфигурации."""
        config_path = Path(self.temp_dir) / "config.json"
        
        # Сохранение конфигурации
        self.config.save_to_file(str(config_path))
        self.assertTrue(config_path.exists())
        
        # Загрузка конфигурации
        loaded_config = ModelConfig.from_file(str(config_path))
        self.assertEqual(loaded_config.grid_size, self.config.grid_size)
        self.assertEqual(loaded_config.box_size, self.config.box_size)
    
    def test_results_serialization(self):
        """Тест сериализации результатов."""
        model = ProtonModel(self.config)
        results = model.run()
        
        results_path = Path(self.temp_dir) / "results.json"
        
        # Сохранение результатов
        results.save_to_file(str(results_path))
        self.assertTrue(results_path.exists())
        
        # Загрузка результатов
        with open(results_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data['status'], results.status.value)
        self.assertEqual(loaded_data['proton_mass'], results.proton_mass)


class TestPerformance(unittest.TestCase):
    """Performance-тесты."""
    
    def setUp(self):
        """Настройка тестов."""
        self.config = ModelConfig(
            grid_size=32,
            box_size=2.0,
            max_iterations=100,
            validation_enabled=False
        )
    
    def test_execution_time(self):
        """Тест времени выполнения."""
        model = ProtonModel(self.config)
        
        start_time = time.time()
        results = model.run()
        execution_time = time.time() - start_time
        
        # Проверка времени выполнения
        self.assertLess(execution_time, 60.0)  # Менее 1 минуты
        self.assertGreater(execution_time, 0.1)  # Более 0.1 секунды
    
    def test_memory_usage(self):
        """Тест использования памяти."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # МБ
        
        model = ProtonModel(self.config)
        results = model.run()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # МБ
        memory_usage = final_memory - initial_memory
        
        # Проверка использования памяти
        self.assertLess(memory_usage, 500.0)  # Менее 500 МБ
    
    def test_scalability(self):
        """Тест масштабируемости."""
        grid_sizes = [16, 32, 64]
        execution_times = []
        
        for grid_size in grid_sizes:
            config = ModelConfig(
                grid_size=grid_size,
                box_size=2.0,
                max_iterations=50,
                validation_enabled=False
            )
            
            model = ProtonModel(config)
            
            start_time = time.time()
            results = model.run()
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
        
        # Проверка масштабируемости (время должно расти не слишком быстро)
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            self.assertLess(ratio, 10.0)  # Не более чем в 10 раз медленнее


class TestStress(unittest.TestCase):
    """Stress-тесты."""
    
    def test_extreme_parameters(self):
        """Тест экстремальных параметров."""
        # Тест с очень маленькой сеткой
        config = ModelConfig(
            grid_size=8,
            box_size=1.0,
            max_iterations=10,
            validation_enabled=False
        )
        
        model = ProtonModel(config)
        results = model.run()
        
        # Проверка успешного выполнения
        self.assertIsNotNone(results)
        self.assertGreater(results.proton_mass, 0)
    
    def test_large_iterations(self):
        """Тест большого количества итераций."""
        config = ModelConfig(
            grid_size=32,
            box_size=2.0,
            max_iterations=1000,
            validation_enabled=False
        )
        
        model = ProtonModel(config)
        results = model.run()
        
        # Проверка успешного выполнения
        self.assertIsNotNone(results)
        self.assertTrue(results.converged)
    
    def test_error_handling(self):
        """Тест обработки ошибок."""
        # Тест с неверной конфигурацией
        config = ModelConfig(
            grid_size=-1,  # Неверное значение
            box_size=2.0
        )
        
        with self.assertRaises(ValueError):
            ProtonModel(config)


class TestSuite:
    """Основной класс тестового набора."""
    
    def __init__(self):
        """Инициализация тестового набора."""
        self.test_config = TestConfig()
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Запуск всех тестов.
        
        Returns:
            Результаты тестирования
        """
        # Создание тестового набора
        test_suite = unittest.TestSuite()
        
        # Добавление всех тестов
        test_classes = [
            TestMathematicalFoundations,
            TestTorusGeometries,
            TestSU2Fields,
            TestEnergyDensities,
            TestPhysicalQuantities,
            TestNumericalMethods,
            TestValidationSystem,
            TestIntegration,
            TestPerformance,
            TestStress
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Запуск тестов
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Сохранение результатов
        self.results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
            'failures': result.failures,
            'errors': result.errors
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Генерация отчета о тестировании.
        
        Returns:
            Текстовый отчет
        """
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ О ТЕСТИРОВАНИИ МОДЕЛИ ПРОТОНА")
        report.append("=" * 80)
        report.append(f"Всего тестов: {self.results['tests_run']}")
        report.append(f"Успешных: {self.results['tests_run'] - self.results['failures'] - self.results['errors']}")
        report.append(f"Неудачных: {self.results['failures']}")
        report.append(f"Ошибок: {self.results['errors']}")
        report.append(f"Процент успеха: {self.results['success_rate']:.2%}")
        report.append("")
        
        if self.results['failures']:
            report.append("НЕУДАЧНЫЕ ТЕСТЫ:")
            report.append("-" * 40)
            for test, traceback in self.results['failures']:
                report.append(f"  - {test}: {traceback}")
            report.append("")
        
        if self.results['errors']:
            report.append("ОШИБКИ:")
            report.append("-" * 40)
            for test, traceback in self.results['errors']:
                report.append(f"  - {test}: {traceback}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def run_tests():
    """Запуск всех тестов."""
    test_suite = TestSuite()
    results = test_suite.run_all_tests()
    report = test_suite.generate_report()
    
    print(report)
    
    return results


if __name__ == "__main__":
    run_tests()
```

## Объяснение

### Типы тестов

1. **Unit-тесты** - тестирование отдельных компонентов
2. **Интеграционные тесты** - тестирование взаимодействия модулей
3. **Performance-тесты** - оценка производительности
4. **Stress-тесты** - проверка стабильности

### Покрытие тестирования

- Все основные компоненты покрыты тестами
- Проверяются граничные случаи
- Валидируются физические результаты
- Оценивается производительность

### Автоматизация

- Автоматический запуск всех тестов
- Генерация отчетов о тестировании
- Интеграция с CI/CD системами

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 10: Оптимизация**, где создадим систему CUDA оптимизации и масштабирования для повышения производительности модели.
