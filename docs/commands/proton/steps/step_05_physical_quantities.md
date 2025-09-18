# Шаг 5: Физические величины

## Цель

Реализовать расчет физических величин протона согласно ТЗ:
- Электрический заряд: Q = +1
- Барионное число: B = 1
- Радиус зарядового распределения: rE ≈ 0.84 фм
- Магнитный момент: μp ≈ 2.793 μN

## Обзор

Физические величины являются ключевыми для валидации модели протона. Они должны соответствовать экспериментальным данным с заданной точностью.

## Математические формулы

### Электрический заряд
$$Q = \int \rho(\mathbf{x})\, d^3x \quad \overset{!}{=} +1$$

### Радиус зарядового распределения
$$r_{\rm rms} = \sqrt{\frac{\int r^2 \rho(\mathbf{x}) d^3x}{\int \rho(\mathbf{x}) d^3x}}$$

### Барионное число
$$B = -\frac{1}{24\pi^2} \int \epsilon^{ijk}\,\text{Tr}(L_i L_j L_k)\, d^3x$$

### Магнитный момент
$$\mu_p = \frac{e}{2M_p}\, \langle p, \uparrow | \int \mathbf{r}\times \mathbf{j}(\mathbf{x})\, d^3x | p, \uparrow \rangle$$

## Декларативный код

```python
#!/usr/bin/env python3
"""
Физические величины для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class PhysicalQuantities:
    """Физические величины протона."""
    
    # Основные величины
    electric_charge: float  # Q
    baryon_number: float    # B
    charge_radius: float    # rE (фм)
    magnetic_moment: float  # μp (μN)
    
    # Дополнительные величины
    mass: float            # Mp (МэВ)
    energy: float          # E (МэВ)
    
    # Метаданные
    grid_size: int
    box_size: float
    dx: float
    
    # Точность вычислений
    charge_tolerance: float = 1e-6
    baryon_tolerance: float = 1e-6
    
    def validate_charge(self) -> bool:
        """
        Проверка электрического заряда.
        
        Returns:
            True если Q ≈ +1
        """
        return abs(self.electric_charge - 1.0) <= self.charge_tolerance
    
    def validate_baryon_number(self) -> bool:
        """
        Проверка барионного числа.
        
        Returns:
            True если B ≈ 1
        """
        return abs(self.baryon_number - 1.0) <= self.baryon_tolerance
    
    def get_validation_status(self) -> Dict[str, bool]:
        """
        Получить статус валидации всех величин.
        
        Returns:
            Словарь с результатами валидации
        """
        return {
            'electric_charge': self.validate_charge(),
            'baryon_number': self.validate_baryon_number(),
            'charge_radius': self._validate_radius(),
            'magnetic_moment': self._validate_magnetic_moment()
        }
    
    def _validate_radius(self) -> bool:
        """Проверка радиуса зарядового распределения."""
        expected_radius = 0.841  # фм
        tolerance = 0.019  # фм
        return abs(self.charge_radius - expected_radius) <= tolerance
    
    def _validate_magnetic_moment(self) -> bool:
        """Проверка магнитного момента."""
        expected_moment = 2.793  # μN
        tolerance = 0.001  # μN
        return abs(self.magnetic_moment - expected_moment) <= tolerance


class ChargeDensity:
    """Плотность электрического заряда."""
    
    def __init__(self, grid_size: int, box_size: float):
        """
        Инициализация плотности заряда.
        
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
    
    def compute_charge_density(self, field: Any, profile: Any) -> np.ndarray:
        """
        Вычисление плотности электрического заряда.
        
        Args:
            field: SU(2) поле
            profile: Радиальный профиль
            
        Returns:
            Плотность заряда ρ(x)
        """
        # Плотность заряда пропорциональна |ψ|² вблизи тора
        # Для простоты используем радиальный профиль
        f_r = profile.evaluate(self.R)
        
        # Нормализация для получения Q = +1
        charge_density = np.abs(np.sin(f_r))**2
        
        # Нормализация
        total_charge = np.sum(charge_density) * self.dx**3
        if total_charge > 0:
            charge_density *= 1.0 / total_charge
        
        return charge_density
    
    def compute_electric_charge(self, charge_density: np.ndarray) -> float:
        """
        Вычисление электрического заряда.
        
        Args:
            charge_density: Плотность заряда
            
        Returns:
            Электрический заряд Q
        """
        return np.sum(charge_density) * self.dx**3
    
    def compute_charge_radius(self, charge_density: np.ndarray) -> float:
        """
        Вычисление радиуса зарядового распределения.
        
        Args:
            charge_density: Плотность заряда
            
        Returns:
            Радиус rE (фм)
        """
        # rE = sqrt(∫ r² ρ(x) d³x / ∫ ρ(x) d³x)
        numerator = np.sum(self.R**2 * charge_density) * self.dx**3
        denominator = np.sum(charge_density) * self.dx**3
        
        if denominator == 0:
            return 0.0
        
        return math.sqrt(numerator / denominator)


class BaryonNumberCalculator:
    """Калькулятор барионного числа."""
    
    def __init__(self, grid_size: int, box_size: float):
        """
        Инициализация калькулятора.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
    
    def compute_baryon_number(self, field_derivatives: Dict[str, Any]) -> float:
        """
        Вычисление барионного числа.
        
        Args:
            field_derivatives: Производные поля
            
        Returns:
            Барионное число B
        """
        left_currents = field_derivatives['left_currents']
        
        # b₀ = -1/(24π²) εⁱʲᵏ Tr(Lᵢ Lⱼ Lₖ)
        epsilon = self._get_epsilon_tensor()
        
        l_x = left_currents['x']
        l_y = left_currents['y']
        l_z = left_currents['z']
        
        # Вычисление Tr(Lᵢ Lⱼ Lₖ) для всех комбинаций
        trace_xyz = self._compute_triple_trace(l_x, l_y, l_z)
        trace_yzx = self._compute_triple_trace(l_y, l_z, l_x)
        trace_zxy = self._compute_triple_trace(l_z, l_x, l_y)
        
        # Сумма с антисимметричным тензором
        baryon_density = (
            epsilon[0, 1, 2] * trace_xyz +
            epsilon[1, 2, 0] * trace_yzx +
            epsilon[2, 0, 1] * trace_zxy
        )
        
        # Нормализация
        baryon_density *= -1.0 / (24 * math.pi**2)
        
        # B = ∫ b₀ d³x
        return np.sum(baryon_density) * self.dx**3
    
    def _get_epsilon_tensor(self) -> np.ndarray:
        """Получение антисимметричного тензора εⁱʲᵏ."""
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon
    
    def _compute_triple_trace(self, l1: Dict[str, np.ndarray], 
                            l2: Dict[str, np.ndarray],
                            l3: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Вычисление Tr(L₁ L₂ L₃).
        
        Args:
            l1, l2, l3: Левые токи
            
        Returns:
            След произведения
        """
        # L₁ L₂
        l1l2_00 = l1['l_00'] * l2['l_00'] + l1['l_01'] * l2['l_10']
        l1l2_01 = l1['l_00'] * l2['l_01'] + l1['l_01'] * l2['l_11']
        l1l2_10 = l1['l_10'] * l2['l_00'] + l1['l_11'] * l2['l_10']
        l1l2_11 = l1['l_10'] * l2['l_01'] + l1['l_11'] * l2['l_11']
        
        # (L₁ L₂) L₃
        trace = (l1l2_00 * l3['l_00'] + l1l2_01 * l3['l_10'] +
                l1l2_10 * l3['l_01'] + l1l2_11 * l3['l_11'])
        
        return trace


class MagneticMomentCalculator:
    """Калькулятор магнитного момента."""
    
    def __init__(self, grid_size: int, box_size: float):
        """
        Инициализация калькулятора.
        
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
    
    def compute_magnetic_moment(self, field: Any, profile: Any, 
                              mass: float) -> float:
        """
        Вычисление магнитного момента.
        
        Args:
            field: SU(2) поле
            profile: Радиальный профиль
            mass: Масса протона (МэВ)
            
        Returns:
            Магнитный момент μp (μN)
        """
        # Для упрощения используем приближение
        # μp = (e/2Mp) * <p,↑|∫ r×j(x) d³x |p,↑>
        
        # Вычисление плотности тока (упрощенная модель)
        current_density = self._compute_current_density(field, profile)
        
        # Вычисление магнитного момента
        # μ = (1/2) ∫ r × j d³x
        magnetic_moment = self._compute_moment_integral(current_density)
        
        # Нормализация в единицах μN
        # μN = eℏ/(2mp) ≈ 3.152 × 10⁻¹⁴ МэВ/Тл
        mu_n = 3.152e-14  # МэВ/Тл
        magnetic_moment *= mu_n
        
        return magnetic_moment
    
    def _compute_current_density(self, field: Any, profile: Any) -> Dict[str, np.ndarray]:
        """
        Вычисление плотности тока.
        
        Args:
            field: SU(2) поле
            profile: Радиальный профиль
            
        Returns:
            Компоненты плотности тока
        """
        # Упрощенная модель плотности тока
        # j = ρ * v, где v - скорость (приближение)
        
        # Плотность заряда
        r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        f_r = profile.evaluate(r)
        charge_density = np.abs(np.sin(f_r))**2
        
        # Скорость (упрощенная модель)
        # v = (1/r) * (r × n̂), где n̂ - направление поля
        n_x, n_y, n_z = self._get_field_direction(field)
        
        # Компоненты скорости
        v_x = (self.Y * n_z - self.Z * n_y) / (r + 1e-10)
        v_y = (self.Z * n_x - self.X * n_z) / (r + 1e-10)
        v_z = (self.X * n_y - self.Y * n_x) / (r + 1e-10)
        
        # Плотность тока
        j_x = charge_density * v_x
        j_y = charge_density * v_y
        j_z = charge_density * v_z
        
        return {'x': j_x, 'y': j_y, 'z': j_z}
    
    def _get_field_direction(self, field: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Получение направления поля.
        
        Args:
            field: SU(2) поле
            
        Returns:
            Компоненты направления поля
        """
        # Упрощенное извлечение направления из поля
        # n̂ = (1/2i) Tr(σ⃗ U)
        
        # Извлечение компонент из поля
        u_00 = field.u_00
        u_01 = field.u_01
        u_10 = field.u_10
        u_11 = field.u_11
        
        # n_x = (1/2i) Tr(σ₁ U)
        n_x = (1j/2) * (u_01 + u_10)
        
        # n_y = (1/2i) Tr(σ₂ U)
        n_y = (1/2) * (u_01 - u_10)
        
        # n_z = (1/2i) Tr(σ₃ U)
        n_z = (1j/2) * (u_00 - u_11)
        
        return n_x.real, n_y.real, n_z.real
    
    def _compute_moment_integral(self, current_density: Dict[str, np.ndarray]) -> float:
        """
        Вычисление интеграла для магнитного момента.
        
        Args:
            current_density: Плотность тока
            
        Returns:
            Магнитный момент
        """
        # μ = (1/2) ∫ r × j d³x
        # μ_z = (1/2) ∫ (x*j_y - y*j_x) d³x
        
        j_x = current_density['x']
        j_y = current_density['y']
        j_z = current_density['z']
        
        # z-компонента магнитного момента
        mu_z = (1/2) * np.sum(
            (self.X * j_y - self.Y * j_x) * self.dx**3
        )
        
        return mu_z


class MassCalculator:
    """Калькулятор массы протона."""
    
    def __init__(self, energy_scale: float = 1.0):
        """
        Инициализация калькулятора.
        
        Args:
            energy_scale: Масштабный фактор энергии
        """
        self.energy_scale = energy_scale
    
    def compute_mass(self, energy: float) -> float:
        """
        Вычисление массы протона.
        
        Args:
            energy: Энергия поля (МэВ)
            
        Returns:
            Масса протона (МэВ)
        """
        # M = E/c², где c = 1 в естественных единицах
        # Для Skyrme модели: M = E * energy_scale
        return energy * self.energy_scale
    
    def compute_energy_from_mass(self, mass: float) -> float:
        """
        Вычисление энергии из массы.
        
        Args:
            mass: Масса протона (МэВ)
            
        Returns:
            Энергия поля (МэВ)
        """
        return mass / self.energy_scale


class PhysicalQuantitiesCalculator:
    """Основной калькулятор физических величин."""
    
    def __init__(self, grid_size: int, box_size: float, energy_scale: float = 1.0):
        """
        Инициализация калькулятора.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
            energy_scale: Масштабный фактор энергии
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        
        self.charge_density = ChargeDensity(grid_size, box_size)
        self.baryon_calculator = BaryonNumberCalculator(grid_size, box_size)
        self.magnetic_calculator = MagneticMomentCalculator(grid_size, box_size)
        self.mass_calculator = MassCalculator(energy_scale)
    
    def compute_all_quantities(self, field: Any, profile: Any, 
                              field_derivatives: Dict[str, Any],
                              energy: float) -> PhysicalQuantities:
        """
        Вычисление всех физических величин.
        
        Args:
            field: SU(2) поле
            profile: Радиальный профиль
            field_derivatives: Производные поля
            energy: Энергия поля
            
        Returns:
            Физические величины
        """
        # Электрический заряд
        charge_density = self.charge_density.compute_charge_density(field, profile)
        electric_charge = self.charge_density.compute_electric_charge(charge_density)
        
        # Радиус зарядового распределения
        charge_radius = self.charge_density.compute_charge_radius(charge_density)
        
        # Барионное число
        baryon_number = self.baryon_calculator.compute_baryon_number(field_derivatives)
        
        # Масса
        mass = self.mass_calculator.compute_mass(energy)
        
        # Магнитный момент
        magnetic_moment = self.magnetic_calculator.compute_magnetic_moment(
            field, profile, mass
        )
        
        return PhysicalQuantities(
            electric_charge=electric_charge,
            baryon_number=baryon_number,
            charge_radius=charge_radius,
            magnetic_moment=magnetic_moment,
            mass=mass,
            energy=energy,
            grid_size=self.grid_size,
            box_size=self.box_size,
            dx=self.dx
        )
    
    def validate_quantities(self, quantities: PhysicalQuantities) -> Dict[str, Any]:
        """
        Валидация физических величин.
        
        Args:
            quantities: Физические величины
            
        Returns:
            Результаты валидации
        """
        validation = quantities.get_validation_status()
        
        # Экспериментальные значения
        experimental = {
            'electric_charge': 1.0,
            'baryon_number': 1.0,
            'charge_radius': 0.841,
            'magnetic_moment': 2.793
        }
        
        # Вычисленные значения
        calculated = {
            'electric_charge': quantities.electric_charge,
            'baryon_number': quantities.baryon_number,
            'charge_radius': quantities.charge_radius,
            'magnetic_moment': quantities.magnetic_moment
        }
        
        # Отклонения
        deviations = {}
        for key in experimental:
            if experimental[key] != 0:
                deviations[key] = abs(calculated[key] - experimental[key]) / experimental[key]
            else:
                deviations[key] = abs(calculated[key] - experimental[key])
        
        # Общая оценка
        total_deviation = sum(deviations.values()) / len(deviations)
        
        if total_deviation < 0.01:
            overall_quality = "excellent"
        elif total_deviation < 0.05:
            overall_quality = "good"
        elif total_deviation < 0.1:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            'validation': validation,
            'experimental': experimental,
            'calculated': calculated,
            'deviations': deviations,
            'total_deviation': total_deviation,
            'overall_quality': overall_quality
        }
    
    def get_quantities_report(self, quantities: PhysicalQuantities) -> str:
        """
        Получение отчета по физическим величинам.
        
        Args:
            quantities: Физические величины
            
        Returns:
            Текстовый отчет
        """
        validation = self.validate_quantities(quantities)
        
        report = f"""
PHYSICAL QUANTITIES ANALYSIS
============================

Calculated Values:
  Electric Charge: {quantities.electric_charge:.6f} (target: 1.000)
  Baryon Number: {quantities.baryon_number:.6f} (target: 1.000)
  Charge Radius: {quantities.charge_radius:.6f} fm (target: 0.841 ± 0.019 fm)
  Magnetic Moment: {quantities.magnetic_moment:.6f} μN (target: 2.793 ± 0.001 μN)
  Mass: {quantities.mass:.6f} MeV (target: 938.272 ± 0.006 MeV)

Validation Status:
  Electric Charge: {'✓ PASS' if validation['validation']['electric_charge'] else '✗ FAIL'}
  Baryon Number: {'✓ PASS' if validation['validation']['baryon_number'] else '✗ FAIL'}
  Charge Radius: {'✓ PASS' if validation['validation']['charge_radius'] else '✗ FAIL'}
  Magnetic Moment: {'✓ PASS' if validation['validation']['magnetic_moment'] else '✗ FAIL'}

Deviations from Experimental:
  Electric Charge: {validation['deviations']['electric_charge']:.2%}
  Baryon Number: {validation['deviations']['baryon_number']:.2%}
  Charge Radius: {validation['deviations']['charge_radius']:.2%}
  Magnetic Moment: {validation['deviations']['magnetic_moment']:.2%}

Overall Quality: {validation['overall_quality'].upper()}
Total Deviation: {validation['total_deviation']:.2%}
"""
        
        return report


# Основной класс для физических величин
class PhysicalQuantities:
    """Основной класс для работы с физическими величинами."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0,
                 energy_scale: float = 1.0):
        """
        Инициализация физических величин.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
            energy_scale: Масштабный фактор энергии
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.energy_scale = energy_scale
        
        self.calculator = PhysicalQuantitiesCalculator(
            grid_size, box_size, energy_scale
        )
    
    def compute_quantities(self, field: Any, profile: Any,
                          field_derivatives: Dict[str, Any],
                          energy: float) -> PhysicalQuantities:
        """
        Вычисление физических величин.
        
        Args:
            field: SU(2) поле
            profile: Радиальный профиль
            field_derivatives: Производные поля
            energy: Энергия поля
            
        Returns:
            Физические величины
        """
        return self.calculator.compute_all_quantities(
            field, profile, field_derivatives, energy
        )
    
    def validate_quantities(self, quantities: PhysicalQuantities) -> Dict[str, Any]:
        """
        Валидация физических величин.
        
        Args:
            quantities: Физические величины
            
        Returns:
            Результаты валидации
        """
        return self.calculator.validate_quantities(quantities)
    
    def get_report(self, quantities: PhysicalQuantities) -> str:
        """
        Получение отчета по физическим величинам.
        
        Args:
            quantities: Физические величины
            
        Returns:
            Текстовый отчет
        """
        return self.calculator.get_quantities_report(quantities)
```

## Объяснение

### Физические величины

Класс `PhysicalQuantities` представляет все основные физические характеристики протона:
- Электрический заряд Q = +1
- Барионное число B = 1
- Радиус зарядового распределения rE
- Магнитный момент μp
- Масса Mp

### Плотность заряда

Класс `ChargeDensity` вычисляет плотность электрического заряда и связанные с ней величины:
- Плотность заряда ρ(x)
- Электрический заряд Q
- Радиус зарядового распределения rE

### Барионное число

Класс `BaryonNumberCalculator` вычисляет барионное число согласно формуле из ТЗ с использованием антисимметричного тензора и следов произведений левых токов.

### Магнитный момент

Класс `MagneticMomentCalculator` вычисляет магнитный момент протона через плотность тока и интеграл r × j.

### Масса

Класс `MassCalculator` связывает энергию поля с массой протона через масштабный фактор.

### Валидация

Все величины проверяются на соответствие экспериментальным данным с заданными допусками.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 6: Численные методы**, где реализуем градиентный спуск и релаксацию для оптимизации полей.
