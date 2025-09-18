# Шаг 1: Математические основы

## Цель

Создать базовые математические структуры, константы и утилиты для модели протона на основе SU(2) теории.

## Обзор

Этот шаг закладывает математический фундамент для всей модели протона. Мы определим:
- Физические константы
- Математические структуры (векторы, матрицы, тензоры)
- Базовые операции с SU(2) группами
- Утилиты для численных вычислений

## Физические константы

### Экспериментальные значения (из ТЗ)
- Электрический заряд протона: Q = +1 (точное)
- Барионное число: B = 1 (точное)  
- Масса протона: Mp = 938.272 ± 0.006 МэВ
- Радиус зарядового распределения: rE = 0.841 ± 0.019 фм
- Магнитный момент: μp = 2.793 ± 0.001 μN

### Теоретические константы
- Константы Skyrme модели: c₂, c₄, c₆
- Параметры тороидальных структур
- Масштабные факторы

## Математические структуры

### SU(2) матрицы Паули
```python
# Матрицы Паули
σ₁ = [[0, 1], [1, 0]]
σ₂ = [[0, -i], [i, 0]]  
σ₃ = [[1, 0], [0, -1]]
```

### Тензорные операции
- Свертки с εⁱʲᵏ (антисимметричный тензор)
- Операции с Lᵢ = U†∂ᵢU
- Коммутаторы [Lᵢ, Lⱼ]

## Декларативный код

```python
#!/usr/bin/env python3
"""
Математические основы для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PhysicalConstants:
    """Физические константы для модели протона."""
    
    # Экспериментальные значения
    PROTON_CHARGE = 1.0  # Точное значение
    BARYON_NUMBER = 1.0  # Точное значение
    PROTON_MASS_MEV = 938.272  # МэВ
    PROTON_MASS_ERROR = 0.006  # МэВ
    CHARGE_RADIUS_FM = 0.841  # фм
    CHARGE_RADIUS_ERROR = 0.019  # фм
    MAGNETIC_MOMENT_MU_N = 2.793  # μN
    MAGNETIC_MOMENT_ERROR = 0.001  # μN
    
    # Физические константы
    HBAR_C = 197.3269804  # МэВ·фм
    ALPHA_EM = 1.0/137.035999139  # Постоянная тонкой структуры
    
    # Масштабные факторы
    ENERGY_SCALE = 1.0  # МэВ
    LENGTH_SCALE = 1.0  # фм


class SkyrmeConstants:
    """Константы Skyrme модели."""
    
    def __init__(self, c2: float = 1.0, c4: float = 1.0, c6: float = 1.0):
        """
        Инициализация констант Skyrme модели.
        
        Args:
            c2: Константа для члена Tr(L_i L_i)
            c4: Константа для члена Tr([L_i, L_j]^2)  
            c6: Константа для стабилизирующего члена b_0^2
        """
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6
    
    def validate(self) -> bool:
        """
        Проверка корректности констант.
        
        Returns:
            True если константы корректны
        """
        return all(c > 0 for c in [self.c2, self.c4, self.c6])


class PauliMatrices:
    """Матрицы Паули для SU(2) операций."""
    
    # Матрицы Паули
    SIGMA_1 = np.array([[0, 1], [1, 0]], dtype=complex)
    SIGMA_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    SIGMA_3 = np.array([[1, 0], [0, -1]], dtype=complex)
    
    @classmethod
    def get_sigma(cls, i: int) -> np.ndarray:
        """
        Получить матрицу Паули по индексу.
        
        Args:
            i: Индекс матрицы (1, 2, 3)
            
        Returns:
            Матрица Паули
        """
        if i == 1:
            return cls.SIGMA_1
        elif i == 2:
            return cls.SIGMA_2
        elif i == 3:
            return cls.SIGMA_3
        else:
            raise ValueError(f"Invalid Pauli matrix index: {i}")
    
    @classmethod
    def get_all_sigmas(cls) -> List[np.ndarray]:
        """
        Получить все матрицы Паули.
        
        Returns:
            Список всех матриц Паули
        """
        return [cls.SIGMA_1, cls.SIGMA_2, cls.SIGMA_3]


class TensorOperations:
    """Операции с тензорами для модели протона."""
    
    @staticmethod
    def epsilon_tensor() -> np.ndarray:
        """
        Антисимметричный тензор εⁱʲᵏ.
        
        Returns:
            3x3x3 массив с антисимметричным тензором
        """
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon
    
    @staticmethod
    def trace_product(matrices: List[np.ndarray]) -> complex:
        """
        Вычисление следа произведения матриц.
        
        Args:
            matrices: Список матриц для перемножения
            
        Returns:
            След произведения
        """
        if not matrices:
            return 0.0
        
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.dot(result, matrix)
        
        return np.trace(result)
    
    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Вычисление коммутатора [A, B] = AB - BA.
        
        Args:
            A, B: Матрицы для вычисления коммутатора
            
        Returns:
            Коммутатор [A, B]
        """
        return np.dot(A, B) - np.dot(B, A)


class CoordinateSystem:
    """Система координат для тороидальных структур."""
    
    def __init__(self, grid_size: int, box_size: float):
        """
        Инициализация системы координат.
        
        Args:
            grid_size: Размер сетки (N x N x N)
            box_size: Размер коробки в фм
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        
        # Создание координатных сеток
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        y = np.linspace(-box_size/2, box_size/2, grid_size)
        z = np.linspace(-box_size/2, box_size/2, grid_size)
        
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Получить координатные сетки.
        
        Returns:
            Кортеж (X, Y, Z) координатных сеток
        """
        return self.X, self.Y, self.Z
    
    def get_radial_coordinate(self) -> np.ndarray:
        """
        Получить радиальную координату.
        
        Returns:
            Радиальная координата r = sqrt(x² + y² + z²)
        """
        return self.R
    
    def get_volume_element(self) -> float:
        """
        Получить элемент объема.
        
        Returns:
            dx³ - элемент объема
        """
        return self.dx**3


class NumericalUtils:
    """Утилиты для численных вычислений."""
    
    @staticmethod
    def gradient_3d(field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление градиента 3D поля.
        
        Args:
            field: 3D скалярное поле
            dx: Шаг сетки
            
        Returns:
            Кортеж (∂f/∂x, ∂f/∂y, ∂f/∂z)
        """
        grad_x = np.gradient(field, dx, axis=0)
        grad_y = np.gradient(field, dx, axis=1)
        grad_z = np.gradient(field, dx, axis=2)
        
        return grad_x, grad_y, grad_z
    
    @staticmethod
    def divergence_3d(field_x: np.ndarray, field_y: np.ndarray, 
                     field_z: np.ndarray, dx: float) -> np.ndarray:
        """
        Вычисление дивергенции 3D векторного поля.
        
        Args:
            field_x, field_y, field_z: Компоненты векторного поля
            dx: Шаг сетки
            
        Returns:
            Дивергенция ∇·F
        """
        div_x = np.gradient(field_x, dx, axis=0)
        div_y = np.gradient(field_y, dx, axis=1)
        div_z = np.gradient(field_z, dx, axis=2)
        
        return div_x + div_y + div_z
    
    @staticmethod
    def integrate_3d(field: np.ndarray, dx: float) -> float:
        """
        Интегрирование 3D поля по объему.
        
        Args:
            field: 3D поле для интегрирования
            dx: Шаг сетки
            
        Returns:
            Результат интегрирования
        """
        return np.sum(field) * dx**3


class ValidationUtils:
    """Утилиты для валидации результатов."""
    
    @staticmethod
    def check_su2_matrix(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Проверка, является ли матрица элементом SU(2).
        
        Args:
            U: Матрица для проверки
            tolerance: Допустимая погрешность
            
        Returns:
            True если матрица ∈ SU(2)
        """
        # Проверка унитарности: U†U = I
        unitary_check = np.allclose(np.dot(U.conj().T, U), np.eye(2), atol=tolerance)
        
        # Проверка детерминанта: det(U) = 1
        det_check = abs(np.linalg.det(U) - 1.0) < tolerance
        
        return unitary_check and det_check
    
    @staticmethod
    def check_physical_bounds(value: float, expected: float, 
                            tolerance: float) -> bool:
        """
        Проверка физических границ.
        
        Args:
            value: Вычисленное значение
            expected: Ожидаемое значение
            tolerance: Допустимое отклонение
            
        Returns:
            True если значение в допустимых границах
        """
        return abs(value - expected) <= tolerance


# Основной класс для математических основ
class MathematicalFoundations:
    """Основной класс, объединяющий все математические компоненты."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Инициализация математических основ.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки в фм
        """
        self.constants = PhysicalConstants()
        self.skyrme = SkyrmeConstants()
        self.pauli = PauliMatrices()
        self.tensor = TensorOperations()
        self.coords = CoordinateSystem(grid_size, box_size)
        self.numerical = NumericalUtils()
        self.validation = ValidationUtils()
    
    def validate_setup(self) -> bool:
        """
        Проверка корректности настройки.
        
        Returns:
            True если настройка корректна
        """
        return self.skyrme.validate()
    
    def get_physical_constants(self) -> Dict[str, float]:
        """
        Получить физические константы.
        
        Returns:
            Словарь с физическими константами
        """
        return {
            'proton_charge': self.constants.PROTON_CHARGE,
            'baryon_number': self.constants.BARYON_NUMBER,
            'proton_mass_mev': self.constants.PROTON_MASS_MEV,
            'charge_radius_fm': self.constants.CHARGE_RADIUS_FM,
            'magnetic_moment_mu_n': self.constants.MAGNETIC_MOMENT_MU_N,
            'hbar_c': self.constants.HBAR_C,
            'alpha_em': self.constants.ALPHA_EM
        }
    
    def get_skyrme_constants(self) -> Dict[str, float]:
        """
        Получить константы Skyrme модели.
        
        Returns:
            Словарь с константами Skyrme
        """
        return {
            'c2': self.skyrme.c2,
            'c4': self.skyrme.c4,
            'c6': self.skyrme.c6
        }
```

## Объяснение

### Физические константы
Класс `PhysicalConstants` содержит все экспериментальные значения из ТЗ. Эти константы будут использоваться для валидации результатов модели.

### Константы Skyrme модели
Класс `SkyrmeConstants` определяет параметры c₂, c₄, c₆ для плотности энергии. Эти константы будут подбираться для получения правильных физических параметров.

### Матрицы Паули
Класс `PauliMatrices` предоставляет базовые SU(2) матрицы, необходимые для построения полей U(x).

### Тензорные операции
Класс `TensorOperations` содержит операции с антисимметричным тензором εⁱʲᵏ и коммутаторами, необходимые для расчета барионного числа.

### Система координат
Класс `CoordinateSystem` создает 3D сетку координат и предоставляет утилиты для работы с пространственными координатами.

### Численные утилиты
Класс `NumericalUtils` содержит методы для вычисления градиентов, дивергенций и интегрирования в 3D.

### Валидация
Класс `ValidationUtils` предоставляет методы для проверки корректности SU(2) матриц и физических границ.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 2: Тороидальные геометрии**, где определим три конфигурации торов (120°, клевер, декартовая).
