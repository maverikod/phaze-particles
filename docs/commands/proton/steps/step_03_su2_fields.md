# Шаг 3: SU(2) поля

## Цель

Реализовать SU(2) поля и операции над ними для модели протона согласно формуле из ТЗ:

$$U(\mathbf{x}) = \cos f(r)\, \mathbf{1} + i \sin f(r)\, \hat{n}(\mathbf{x}) \cdot \vec{\sigma}$$

## Обзор

SU(2) поле является основой модели протона. Оно определяется:
- Радиальным профилем f(r)
- Направлением поля n̂(x) из тороидальных конфигураций
- Матрицами Паули σ⃗

## Математические компоненты

### SU(2) поле
- U(x) ∈ SU(2) - унитарная матрица с детерминантом 1
- f(r) - радиальный профиль (скалярная функция)
- n̂(x) - направление поля (векторная функция)
- σ⃗ - матрицы Паули

### Производные поля
- Lᵢ = U†∂ᵢU - левые токи
- Коммутаторы [Lᵢ, Lⱼ] для плотности энергии
- Трасси для вычисления энергии

### Радиальный профиль
- f(r) должна удовлетворять граничным условиям
- f(0) = π (центр протона)
- f(∞) = 0 (бесконечность)
- Плавный переход между значениями

## Декларативный код

```python
#!/usr/bin/env python3
"""
SU(2) поля для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class SU2Field:
    """SU(2) поле U(x) = cos f(r) 1 + i sin f(r) n̂(x) · σ⃗."""
    
    # Компоненты поля
    u_00: np.ndarray  # U[0,0] элемент
    u_01: np.ndarray  # U[0,1] элемент  
    u_10: np.ndarray  # U[1,0] элемент
    u_11: np.ndarray  # U[1,1] элемент
    
    # Метаданные
    grid_size: int
    box_size: float
    
    def __post_init__(self):
        """Проверка корректности поля после инициализации."""
        if not self._is_su2_field():
            raise ValueError("Field is not a valid SU(2) field")
    
    def _is_su2_field(self, tolerance: float = 1e-10) -> bool:
        """
        Проверка, является ли поле элементом SU(2).
        
        Args:
            tolerance: Допустимая погрешность
            
        Returns:
            True если поле ∈ SU(2)
        """
        # Проверка унитарности: U†U = I
        u_dagger_u_00 = (self.u_00.conj() * self.u_00 + 
                        self.u_10.conj() * self.u_10)
        u_dagger_u_01 = (self.u_00.conj() * self.u_01 + 
                        self.u_10.conj() * self.u_11)
        u_dagger_u_10 = (self.u_01.conj() * self.u_00 + 
                        self.u_11.conj() * self.u_10)
        u_dagger_u_11 = (self.u_01.conj() * self.u_01 + 
                        self.u_11.conj() * self.u_11)
        
        unitary_check = (
            np.allclose(u_dagger_u_00, 1.0, atol=tolerance) and
            np.allclose(u_dagger_u_01, 0.0, atol=tolerance) and
            np.allclose(u_dagger_u_10, 0.0, atol=tolerance) and
            np.allclose(u_dagger_u_11, 1.0, atol=tolerance)
        )
        
        # Проверка детерминанта: det(U) = 1
        det_u = (self.u_00 * self.u_11 - self.u_01 * self.u_10)
        det_check = np.allclose(det_u, 1.0, atol=tolerance)
        
        return unitary_check and det_check
    
    def get_matrix_at_point(self, i: int, j: int, k: int) -> np.ndarray:
        """
        Получить матрицу U в точке (i, j, k).
        
        Args:
            i, j, k: Индексы точки
            
        Returns:
            2x2 матрица U в точке
        """
        return np.array([
            [self.u_00[i, j, k], self.u_01[i, j, k]],
            [self.u_10[i, j, k], self.u_11[i, j, k]]
        ], dtype=complex)
    
    def get_determinant(self) -> np.ndarray:
        """
        Вычислить детерминант поля.
        
        Returns:
            Детерминант в каждой точке
        """
        return self.u_00 * self.u_11 - self.u_01 * self.u_10


class RadialProfile:
    """Радиальный профиль f(r) для SU(2) поля."""
    
    def __init__(self, profile_type: str = "skyrmion", 
                 scale: float = 1.0, center_value: float = math.pi):
        """
        Инициализация радиального профиля.
        
        Args:
            profile_type: Тип профиля ("skyrmion", "exponential", "polynomial")
            scale: Масштабный параметр
            center_value: Значение в центре f(0)
        """
        self.profile_type = profile_type
        self.scale = scale
        self.center_value = center_value
    
    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """
        Вычисление радиального профиля f(r).
        
        Args:
            r: Радиальная координата
            
        Returns:
            Значения профиля f(r)
        """
        if self.profile_type == "skyrmion":
            return self._skyrmion_profile(r)
        elif self.profile_type == "exponential":
            return self._exponential_profile(r)
        elif self.profile_type == "polynomial":
            return self._polynomial_profile(r)
        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")
    
    def _skyrmion_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Стандартный профиль скирмиона.
        
        Args:
            r: Радиальная координата
            
        Returns:
            Профиль f(r) = π * exp(-r/scale)
        """
        return self.center_value * np.exp(-r / self.scale)
    
    def _exponential_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Экспоненциальный профиль.
        
        Args:
            r: Радиальная координата
            
        Returns:
            Профиль f(r) = center_value * exp(-r²/scale²)
        """
        return self.center_value * np.exp(-(r**2) / (self.scale**2))
    
    def _polynomial_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Полиномиальный профиль.
        
        Args:
            r: Радиальная координата
            
        Returns:
            Профиль f(r) = center_value * (1 + r/scale)⁻¹
        """
        return self.center_value / (1 + r / self.scale)
    
    def get_derivative(self, r: np.ndarray, dr: float) -> np.ndarray:
        """
        Вычисление производной профиля df/dr.
        
        Args:
            r: Радиальная координата
            dr: Шаг для численного дифференцирования
            
        Returns:
            Производная df/dr
        """
        f_r = self.evaluate(r)
        f_r_plus_dr = self.evaluate(r + dr)
        
        return (f_r_plus_dr - f_r) / dr


class SU2FieldBuilder:
    """Построитель SU(2) полей."""
    
    def __init__(self, grid_size: int, box_size: float):
        """
        Инициализация построителя полей.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
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
    
    def build_field(self, n_x: np.ndarray, n_y: np.ndarray, n_z: np.ndarray,
                   profile: RadialProfile) -> SU2Field:
        """
        Построение SU(2) поля из направления и профиля.
        
        Args:
            n_x, n_y, n_z: Компоненты направления поля
            profile: Радиальный профиль
            
        Returns:
            SU(2) поле
        """
        # Вычисление радиального профиля
        f_r = profile.evaluate(self.R)
        
        # Вычисление компонент поля
        cos_f = np.cos(f_r)
        sin_f = np.sin(f_r)
        
        # U = cos f(r) 1 + i sin f(r) n̂(x) · σ⃗
        u_00 = cos_f + 1j * sin_f * n_z
        u_01 = 1j * sin_f * (n_x - 1j * n_y)
        u_10 = 1j * sin_f * (n_x + 1j * n_y)
        u_11 = cos_f - 1j * sin_f * n_z
        
        return SU2Field(
            u_00=u_00, u_01=u_01, u_10=u_10, u_11=u_11,
            grid_size=self.grid_size, box_size=self.box_size
        )
    
    def build_from_torus_config(self, torus_config: Any, 
                               profile: RadialProfile) -> SU2Field:
        """
        Построение поля из тороидальной конфигурации.
        
        Args:
            torus_config: Тороидальная конфигурация
            profile: Радиальный профиль
            
        Returns:
            SU(2) поле
        """
        # Получение направления поля из конфигурации
        n_x, n_y, n_z = torus_config.get_field_direction(
            self.X, self.Y, self.Z
        )
        
        return self.build_field(n_x, n_y, n_z, profile)


class SU2FieldOperations:
    """Операции над SU(2) полями."""
    
    def __init__(self, dx: float):
        """
        Инициализация операций.
        
        Args:
            dx: Шаг сетки
        """
        self.dx = dx
    
    def compute_left_currents(self, field: SU2Field) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление левых токов Lᵢ = U†∂ᵢU.
        
        Args:
            field: SU(2) поле
            
        Returns:
            Кортеж (L_x, L_y, L_z) левых токов
        """
        # Вычисление производных поля
        du_dx = self._compute_field_derivative(field, axis=0)
        du_dy = self._compute_field_derivative(field, axis=1)
        du_dz = self._compute_field_derivative(field, axis=2)
        
        # Вычисление Lᵢ = U†∂ᵢU
        l_x = self._multiply_field_dagger_derivative(field, du_dx)
        l_y = self._multiply_field_dagger_derivative(field, du_dy)
        l_z = self._multiply_field_dagger_derivative(field, du_dz)
        
        return l_x, l_y, l_z
    
    def _compute_field_derivative(self, field: SU2Field, axis: int) -> Dict[str, np.ndarray]:
        """
        Вычисление производной поля по заданной оси.
        
        Args:
            field: SU(2) поле
            axis: Ось дифференцирования (0, 1, 2)
            
        Returns:
            Словарь с производными компонент
        """
        du_dx = {
            'u_00': np.gradient(field.u_00, self.dx, axis=axis),
            'u_01': np.gradient(field.u_01, self.dx, axis=axis),
            'u_10': np.gradient(field.u_10, self.dx, axis=axis),
            'u_11': np.gradient(field.u_11, self.dx, axis=axis)
        }
        
        return du_dx
    
    def _multiply_field_dagger_derivative(self, field: SU2Field, 
                                        du: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Вычисление U†∂U.
        
        Args:
            field: SU(2) поле
            du: Производные компонент поля
            
        Returns:
            Словарь с компонентами Lᵢ
        """
        # U† = [[u_00*, u_10*], [u_01*, u_11*]]
        # ∂U = [[du_00, du_01], [du_10, du_11]]
        # L = U†∂U
        
        l_00 = (field.u_00.conj() * du['u_00'] + 
                field.u_10.conj() * du['u_10'])
        l_01 = (field.u_00.conj() * du['u_01'] + 
                field.u_10.conj() * du['u_11'])
        l_10 = (field.u_01.conj() * du['u_00'] + 
                field.u_11.conj() * du['u_10'])
        l_11 = (field.u_01.conj() * du['u_01'] + 
                field.u_11.conj() * du['u_11'])
        
        return {
            'l_00': l_00, 'l_01': l_01,
            'l_10': l_10, 'l_11': l_11
        }
    
    def compute_commutators(self, l_x: Dict[str, np.ndarray], 
                          l_y: Dict[str, np.ndarray],
                          l_z: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Вычисление коммутаторов [Lᵢ, Lⱼ].
        
        Args:
            l_x, l_y, l_z: Левые токи
            
        Returns:
            Словарь с коммутаторами
        """
        commutators = {}
        
        # [L_x, L_y]
        commutators['xy'] = self._compute_commutator(l_x, l_y)
        # [L_y, L_z]
        commutators['yz'] = self._compute_commutator(l_y, l_z)
        # [L_z, L_x]
        commutators['zx'] = self._compute_commutator(l_z, l_x)
        
        return commutators
    
    def _compute_commutator(self, l1: Dict[str, np.ndarray], 
                          l2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Вычисление коммутатора [L1, L2] = L1*L2 - L2*L1.
        
        Args:
            l1, l2: Левые токи
            
        Returns:
            Коммутатор
        """
        # L1*L2
        l1l2_00 = l1['l_00'] * l2['l_00'] + l1['l_01'] * l2['l_10']
        l1l2_01 = l1['l_00'] * l2['l_01'] + l1['l_01'] * l2['l_11']
        l1l2_10 = l1['l_10'] * l2['l_00'] + l1['l_11'] * l2['l_10']
        l1l2_11 = l1['l_10'] * l2['l_01'] + l1['l_11'] * l2['l_11']
        
        # L2*L1
        l2l1_00 = l2['l_00'] * l1['l_00'] + l2['l_01'] * l1['l_10']
        l2l1_01 = l2['l_00'] * l1['l_01'] + l2['l_01'] * l1['l_11']
        l2l1_10 = l2['l_10'] * l1['l_00'] + l2['l_11'] * l1['l_10']
        l2l1_11 = l2['l_10'] * l1['l_01'] + l2['l_11'] * l1['l_11']
        
        # [L1, L2] = L1*L2 - L2*L1
        return {
            'comm_00': l1l2_00 - l2l1_00,
            'comm_01': l1l2_01 - l2l1_01,
            'comm_10': l1l2_10 - l2l1_10,
            'comm_11': l1l2_11 - l2l1_11
        }
    
    def compute_traces(self, l_x: Dict[str, np.ndarray], 
                      l_y: Dict[str, np.ndarray],
                      l_z: Dict[str, np.ndarray],
                      commutators: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Вычисление следов для плотности энергии.
        
        Args:
            l_x, l_y, l_z: Левые токи
            commutators: Коммутаторы
            
        Returns:
            Словарь со следами
        """
        traces = {}
        
        # Tr(Lᵢ Lᵢ) - для c₂ члена
        traces['l_squared'] = (
            l_x['l_00'] * l_x['l_00'] + l_x['l_01'] * l_x['l_10'] +
            l_x['l_10'] * l_x['l_01'] + l_x['l_11'] * l_x['l_11'] +
            l_y['l_00'] * l_y['l_00'] + l_y['l_01'] * l_y['l_10'] +
            l_y['l_10'] * l_y['l_01'] + l_y['l_11'] * l_y['l_11'] +
            l_z['l_00'] * l_z['l_00'] + l_z['l_01'] * l_z['l_10'] +
            l_z['l_10'] * l_z['l_01'] + l_z['l_11'] * l_z['l_11']
        )
        
        # Tr([Lᵢ, Lⱼ]²) - для c₄ члена
        traces['comm_squared'] = (
            commutators['xy']['comm_00'] * commutators['xy']['comm_00'] +
            commutators['xy']['comm_01'] * commutators['xy']['comm_10'] +
            commutators['xy']['comm_10'] * commutators['xy']['comm_01'] +
            commutators['xy']['comm_11'] * commutators['xy']['comm_11'] +
            commutators['yz']['comm_00'] * commutators['yz']['comm_00'] +
            commutators['yz']['comm_01'] * commutators['yz']['comm_10'] +
            commutators['yz']['comm_10'] * commutators['yz']['comm_01'] +
            commutators['yz']['comm_11'] * commutators['yz']['comm_11'] +
            commutators['zx']['comm_00'] * commutators['zx']['comm_00'] +
            commutators['zx']['comm_01'] * commutators['zx']['comm_10'] +
            commutators['zx']['comm_10'] * commutators['zx']['comm_01'] +
            commutators['zx']['comm_11'] * commutators['zx']['comm_11']
        )
        
        return traces


class SU2FieldValidator:
    """Валидатор SU(2) полей."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Инициализация валидатора.
        
        Args:
            tolerance: Допустимая погрешность
        """
        self.tolerance = tolerance
    
    def validate_field(self, field: SU2Field) -> Dict[str, bool]:
        """
        Полная валидация SU(2) поля.
        
        Args:
            field: Поле для валидации
            
        Returns:
            Словарь с результатами проверок
        """
        results = {}
        
        # Проверка унитарности
        results['unitary'] = self._check_unitarity(field)
        
        # Проверка детерминанта
        results['determinant'] = self._check_determinant(field)
        
        # Проверка непрерывности
        results['continuity'] = self._check_continuity(field)
        
        # Проверка граничных условий
        results['boundary_conditions'] = self._check_boundary_conditions(field)
        
        return results
    
    def _check_unitarity(self, field: SU2Field) -> bool:
        """Проверка унитарности поля."""
        # Проверка U†U = I
        u_dagger_u_00 = (field.u_00.conj() * field.u_00 + 
                        field.u_10.conj() * field.u_10)
        u_dagger_u_11 = (field.u_01.conj() * field.u_01 + 
                        field.u_11.conj() * field.u_11)
        
        return (np.allclose(u_dagger_u_00, 1.0, atol=self.tolerance) and
                np.allclose(u_dagger_u_11, 1.0, atol=self.tolerance))
    
    def _check_determinant(self, field: SU2Field) -> bool:
        """Проверка детерминанта поля."""
        det = field.get_determinant()
        return np.allclose(det, 1.0, atol=self.tolerance)
    
    def _check_continuity(self, field: SU2Field) -> bool:
        """Проверка непрерывности поля."""
        # Проверка градиентов
        dx = field.box_size / field.grid_size
        
        grad_u_00 = np.gradient(field.u_00, dx)
        grad_u_01 = np.gradient(field.u_01, dx)
        grad_u_10 = np.gradient(field.u_10, dx)
        grad_u_11 = np.gradient(field.u_11, dx)
        
        # Проверка, что градиенты конечны
        return (np.all(np.isfinite(grad_u_00)) and
                np.all(np.isfinite(grad_u_01)) and
                np.all(np.isfinite(grad_u_10)) and
                np.all(np.isfinite(grad_u_11)))
    
    def _check_boundary_conditions(self, field: SU2Field) -> bool:
        """Проверка граничных условий."""
        # В центре поле должно быть близко к -1 (для скирмиона)
        center_idx = field.grid_size // 2
        center_field = field.get_matrix_at_point(center_idx, center_idx, center_idx)
        
        # Проверка, что в центре поле близко к -I
        center_check = np.allclose(center_field, -np.eye(2), atol=self.tolerance)
        
        return center_check


# Основной класс для SU(2) полей
class SU2Fields:
    """Основной класс для работы с SU(2) полями."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Инициализация SU(2) полей.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        
        self.builder = SU2FieldBuilder(grid_size, box_size)
        self.operations = SU2FieldOperations(self.dx)
        self.validator = SU2FieldValidator()
    
    def create_field_from_torus(self, torus_config: Any, 
                               profile_type: str = "skyrmion",
                               scale: float = 1.0) -> SU2Field:
        """
        Создание SU(2) поля из тороидальной конфигурации.
        
        Args:
            torus_config: Тороидальная конфигурация
            profile_type: Тип радиального профиля
            scale: Масштабный параметр профиля
            
        Returns:
            SU(2) поле
        """
        profile = RadialProfile(profile_type, scale)
        return self.builder.build_from_torus_config(torus_config, profile)
    
    def compute_field_derivatives(self, field: SU2Field) -> Dict[str, Any]:
        """
        Вычисление производных поля.
        
        Args:
            field: SU(2) поле
            
        Returns:
            Словарь с производными и следами
        """
        # Вычисление левых токов
        l_x, l_y, l_z = self.operations.compute_left_currents(field)
        
        # Вычисление коммутаторов
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)
        
        # Вычисление следов
        traces = self.operations.compute_traces(l_x, l_y, l_z, commutators)
        
        return {
            'left_currents': {'x': l_x, 'y': l_y, 'z': l_z},
            'commutators': commutators,
            'traces': traces
        }
    
    def validate_field(self, field: SU2Field) -> Dict[str, bool]:
        """
        Валидация SU(2) поля.
        
        Args:
            field: Поле для валидации
            
        Returns:
            Результаты валидации
        """
        return self.validator.validate_field(field)
    
    def get_field_statistics(self, field: SU2Field) -> Dict[str, float]:
        """
        Получение статистики поля.
        
        Args:
            field: SU(2) поле
            
        Returns:
            Словарь со статистикой
        """
        det = field.get_determinant()
        
        return {
            'mean_determinant': np.mean(det.real),
            'std_determinant': np.std(det.real),
            'min_determinant': np.min(det.real),
            'max_determinant': np.max(det.real),
            'field_norm_mean': np.mean(np.abs(field.u_00)),
            'field_norm_std': np.std(np.abs(field.u_00))
        }
```

## Объяснение

### SU(2) поле

Класс `SU2Field` представляет SU(2) поле как набор четырех комплексных массивов, соответствующих элементам 2x2 матрицы. Поле автоматически проверяется на принадлежность к SU(2) группе.

### Радиальный профиль

Класс `RadialProfile` определяет функцию f(r) с различными типами профилей:
- `skyrmion` - стандартный профиль скирмиона
- `exponential` - экспоненциальный профиль
- `polynomial` - полиномиальный профиль

### Построитель полей

Класс `SU2FieldBuilder` создает SU(2) поля из тороидальных конфигураций и радиальных профилей согласно формуле из ТЗ.

### Операции над полями

Класс `SU2FieldOperations` вычисляет:
- Левые токи Lᵢ = U†∂ᵢU
- Коммутаторы [Lᵢ, Lⱼ]
- Следы для плотности энергии

### Валидация

Класс `SU2FieldValidator` проверяет корректность полей:
- Унитарность U†U = I
- Детерминант det(U) = 1
- Непрерывность
- Граничные условия

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 4: Плотности энергии**, где реализуем расчет плотности энергии с членами c₂, c₄, c₆ согласно ТЗ.
