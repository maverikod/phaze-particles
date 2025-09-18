# Шаг 4: Плотности энергии

## Цель

Реализовать расчет плотности энергии для модели протона согласно формуле из ТЗ:

$$\mathcal{E} = c_2\, \text{Tr}(L_i L_i) + c_4\, \text{Tr}([L_i,L_j]^2) + c_6 \, b_0^2$$

где:
- Lᵢ = U†∂ᵢU - левые токи
- b₀ - плотность барионного заряда
- c₂, c₄, c₆ - константы Skyrme модели

## Обзор

Плотность энергии является ключевым компонентом модели протона. Она состоит из трех членов:
1. **c₂ член** - кинетическая энергия (Tr(LᵢLᵢ))
2. **c₄ член** - энергия взаимодействия (Tr([Lᵢ,Lⱼ]²))
3. **c₆ член** - стабилизирующий член (b₀²)

## Физический смысл

### c₂ член (кинетическая энергия)
- Описывает "кинетическую" энергию поля
- Зависит от градиентов поля
- Обеспечивает стабильность солитона

### c₄ член (энергия взаимодействия)
- Описывает взаимодействие между компонентами поля
- Квадратичен по коммутаторам
- Обеспечивает компактность солитона

### c₆ член (стабилизирующий)
- Зависит от плотности барионного заряда
- Стабилизирует размер протона
- Обеспечивает правильную массу

## Декларативный код

```python
#!/usr/bin/env python3
"""
Плотности энергии для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class EnergyDensity:
    """Плотность энергии модели протона."""
    
    # Компоненты плотности энергии
    c2_term: np.ndarray  # c₂ Tr(Lᵢ Lᵢ)
    c4_term: np.ndarray  # c₄ Tr([Lᵢ, Lⱼ]²)
    c6_term: np.ndarray  # c₆ b₀²
    
    # Общая плотность энергии
    total_density: np.ndarray  # ℰ = c₂ + c₄ + c₆
    
    # Метаданные
    grid_size: int
    box_size: float
    dx: float
    
    # Константы Skyrme
    c2: float
    c4: float
    c6: float
    
    def get_total_energy(self) -> float:
        """
        Вычисление общей энергии.
        
        Returns:
            Общая энергия E = ∫ ℰ d³x
        """
        return np.sum(self.total_density) * self.dx**3
    
    def get_energy_components(self) -> Dict[str, float]:
        """
        Вычисление компонент энергии.
        
        Returns:
            Словарь с компонентами энергии
        """
        return {
            'E2': np.sum(self.c2_term) * self.dx**3,
            'E4': np.sum(self.c4_term) * self.dx**3,
            'E6': np.sum(self.c6_term) * self.dx**3,
            'E_total': self.get_total_energy()
        }
    
    def get_energy_balance(self) -> Dict[str, float]:
        """
        Вычисление баланса энергии.
        
        Returns:
            Словарь с балансом энергии
        """
        components = self.get_energy_components()
        total = components['E_total']
        
        if total == 0:
            return {'E2_ratio': 0.0, 'E4_ratio': 0.0, 'E6_ratio': 0.0}
        
        return {
            'E2_ratio': components['E2'] / total,
            'E4_ratio': components['E4'] / total,
            'E6_ratio': components['E6'] / total
        }
    
    def check_virial_condition(self, tolerance: float = 0.05) -> bool:
        """
        Проверка вириального условия E₂ = E₄.
        
        Args:
            tolerance: Допустимое отклонение
            
        Returns:
            True если вириальное условие выполнено
        """
        components = self.get_energy_components()
        E2 = components['E2']
        E4 = components['E4']
        
        if E2 == 0 and E4 == 0:
            return True
        
        ratio = abs(E2 - E4) / max(E2, E4)
        return ratio <= tolerance


class BaryonDensity:
    """Плотность барионного заряда b₀."""
    
    def __init__(self, field_operations: Any):
        """
        Инициализация плотности барионного заряда.
        
        Args:
            field_operations: Операции над SU(2) полями
        """
        self.field_ops = field_operations
    
    def compute_baryon_density(self, left_currents: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Вычисление плотности барионного заряда.
        
        Args:
            left_currents: Левые токи Lᵢ
            
        Returns:
            Плотность барионного заряда b₀
        """
        l_x = left_currents['x']
        l_y = left_currents['y']
        l_z = left_currents['z']
        
        # b₀ = -1/(24π²) εⁱʲᵏ Tr(Lᵢ Lⱼ Lₖ)
        epsilon = self._get_epsilon_tensor()
        
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
        
        return baryon_density
    
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


class EnergyDensityCalculator:
    """Калькулятор плотности энергии."""
    
    def __init__(self, grid_size: int, box_size: float, 
                 c2: float = 1.0, c4: float = 1.0, c6: float = 1.0):
        """
        Инициализация калькулятора.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
            c2, c4, c6: Константы Skyrme
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6
        
        self.baryon_density = BaryonDensity(None)
    
    def compute_energy_density(self, field_derivatives: Dict[str, Any]) -> EnergyDensity:
        """
        Вычисление плотности энергии.
        
        Args:
            field_derivatives: Производные поля и следы
            
        Returns:
            Плотность энергии
        """
        traces = field_derivatives['traces']
        left_currents = field_derivatives['left_currents']
        
        # c₂ член: Tr(Lᵢ Lᵢ)
        c2_term = self.c2 * traces['l_squared']
        
        # c₄ член: Tr([Lᵢ, Lⱼ]²)
        c4_term = self.c4 * traces['comm_squared']
        
        # c₆ член: b₀²
        b0 = self.baryon_density.compute_baryon_density(left_currents)
        c6_term = self.c6 * b0**2
        
        # Общая плотность энергии
        total_density = c2_term + c4_term + c6_term
        
        return EnergyDensity(
            c2_term=c2_term,
            c4_term=c4_term,
            c6_term=c6_term,
            total_density=total_density,
            grid_size=self.grid_size,
            box_size=self.box_size,
            dx=self.dx,
            c2=self.c2,
            c4=self.c4,
            c6=self.c6
        )
    
    def compute_baryon_number(self, field_derivatives: Dict[str, Any]) -> float:
        """
        Вычисление барионного числа.
        
        Args:
            field_derivatives: Производные поля
            
        Returns:
            Барионное число B
        """
        left_currents = field_derivatives['left_currents']
        b0 = self.baryon_density.compute_baryon_density(left_currents)
        
        # B = ∫ b₀ d³x
        return np.sum(b0) * self.dx**3


class EnergyAnalyzer:
    """Анализатор энергии."""
    
    def __init__(self, tolerance: float = 0.05):
        """
        Инициализация анализатора.
        
        Args:
            tolerance: Допустимое отклонение для проверок
        """
        self.tolerance = tolerance
    
    def analyze_energy(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Анализ плотности энергии.
        
        Args:
            energy_density: Плотность энергии
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {}
        
        # Компоненты энергии
        analysis['components'] = energy_density.get_energy_components()
        
        # Баланс энергии
        analysis['balance'] = energy_density.get_energy_balance()
        
        # Вириальное условие
        analysis['virial_condition'] = energy_density.check_virial_condition(self.tolerance)
        
        # Статистика плотности
        analysis['density_stats'] = self._compute_density_statistics(energy_density)
        
        # Качество модели
        analysis['quality'] = self._assess_energy_quality(energy_density)
        
        return analysis
    
    def _compute_density_statistics(self, energy_density: EnergyDensity) -> Dict[str, float]:
        """
        Вычисление статистики плотности энергии.
        
        Args:
            energy_density: Плотность энергии
            
        Returns:
            Словарь со статистикой
        """
        return {
            'total_mean': np.mean(energy_density.total_density),
            'total_std': np.std(energy_density.total_density),
            'total_max': np.max(energy_density.total_density),
            'total_min': np.min(energy_density.total_density),
            'c2_mean': np.mean(energy_density.c2_term),
            'c4_mean': np.mean(energy_density.c4_term),
            'c6_mean': np.mean(energy_density.c6_term)
        }
    
    def _assess_energy_quality(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Оценка качества энергетической модели.
        
        Args:
            energy_density: Плотность энергии
            
        Returns:
            Словарь с оценкой качества
        """
        balance = energy_density.get_energy_balance()
        virial_ok = energy_density.check_virial_condition(self.tolerance)
        
        # Оценка баланса E₂/E₄
        e2_ratio = balance['E2_ratio']
        e4_ratio = balance['E4_ratio']
        
        if abs(e2_ratio - 0.5) < 0.1 and abs(e4_ratio - 0.5) < 0.1:
            balance_quality = "excellent"
        elif abs(e2_ratio - 0.5) < 0.2 and abs(e4_ratio - 0.5) < 0.2:
            balance_quality = "good"
        elif abs(e2_ratio - 0.5) < 0.3 and abs(e4_ratio - 0.5) < 0.3:
            balance_quality = "fair"
        else:
            balance_quality = "poor"
        
        # Общая оценка
        if virial_ok and balance_quality in ["excellent", "good"]:
            overall_quality = "excellent"
        elif virial_ok and balance_quality == "fair":
            overall_quality = "good"
        elif not virial_ok and balance_quality in ["excellent", "good"]:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            'overall_quality': overall_quality,
            'balance_quality': balance_quality,
            'virial_condition': virial_ok,
            'recommendations': self._get_energy_recommendations(balance, virial_ok)
        }
    
    def _get_energy_recommendations(self, balance: Dict[str, float], 
                                  virial_ok: bool) -> List[str]:
        """
        Получение рекомендаций по улучшению энергии.
        
        Args:
            balance: Баланс энергии
            virial_ok: Выполнение вириального условия
            
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        if not virial_ok:
            recommendations.append("Adjust Skyrme constants to satisfy virial condition E₂ = E₄")
        
        e2_ratio = balance['E2_ratio']
        e4_ratio = balance['E4_ratio']
        
        if e2_ratio > 0.6:
            recommendations.append("Reduce c₂ constant to decrease E₂ contribution")
        elif e2_ratio < 0.4:
            recommendations.append("Increase c₂ constant to increase E₂ contribution")
        
        if e4_ratio > 0.6:
            recommendations.append("Reduce c₄ constant to decrease E₄ contribution")
        elif e4_ratio < 0.4:
            recommendations.append("Increase c₄ constant to increase E₄ contribution")
        
        if balance['E6_ratio'] > 0.1:
            recommendations.append("Reduce c₆ constant to decrease E₆ contribution")
        
        return recommendations


class EnergyOptimizer:
    """Оптимизатор констант Skyrme."""
    
    def __init__(self, target_e2_ratio: float = 0.5, target_e4_ratio: float = 0.5,
                 tolerance: float = 0.05):
        """
        Инициализация оптимизатора.
        
        Args:
            target_e2_ratio: Целевое отношение E₂/E_total
            target_e4_ratio: Целевое отношение E₄/E_total
            tolerance: Допустимое отклонение
        """
        self.target_e2_ratio = target_e2_ratio
        self.target_e4_ratio = target_e4_ratio
        self.tolerance = tolerance
    
    def optimize_constants(self, initial_c2: float, initial_c4: float, 
                          initial_c6: float, field_derivatives: Dict[str, Any],
                          max_iterations: int = 100) -> Dict[str, float]:
        """
        Оптимизация констант Skyrme.
        
        Args:
            initial_c2, initial_c4, initial_c6: Начальные константы
            field_derivatives: Производные поля
            max_iterations: Максимальное число итераций
            
        Returns:
            Оптимизированные константы
        """
        c2, c4, c6 = initial_c2, initial_c4, initial_c6
        
        for iteration in range(max_iterations):
            # Вычисление плотности энергии с текущими константами
            calculator = EnergyDensityCalculator(
                field_derivatives['left_currents']['x']['l_00'].shape[0],
                field_derivatives['left_currents']['x']['l_00'].shape[0] * 0.1,  # Примерный box_size
                c2, c4, c6
            )
            
            energy_density = calculator.compute_energy_density(field_derivatives)
            balance = energy_density.get_energy_balance()
            
            # Проверка сходимости
            e2_error = abs(balance['E2_ratio'] - self.target_e2_ratio)
            e4_error = abs(balance['E4_ratio'] - self.target_e4_ratio)
            
            if e2_error < self.tolerance and e4_error < self.tolerance:
                break
            
            # Корректировка констант
            if balance['E2_ratio'] > self.target_e2_ratio:
                c2 *= 0.95
            else:
                c2 *= 1.05
            
            if balance['E4_ratio'] > self.target_e4_ratio:
                c4 *= 0.95
            else:
                c4 *= 1.05
        
        return {'c2': c2, 'c4': c4, 'c6': c6}


# Основной класс для плотности энергии
class EnergyDensities:
    """Основной класс для работы с плотностью энергии."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0,
                 c2: float = 1.0, c4: float = 1.0, c6: float = 1.0):
        """
        Инициализация плотности энергии.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки
            c2, c4, c6: Константы Skyrme
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6
        
        self.calculator = EnergyDensityCalculator(grid_size, box_size, c2, c4, c6)
        self.analyzer = EnergyAnalyzer()
        self.optimizer = EnergyOptimizer()
    
    def compute_energy(self, field_derivatives: Dict[str, Any]) -> EnergyDensity:
        """
        Вычисление плотности энергии.
        
        Args:
            field_derivatives: Производные поля
            
        Returns:
            Плотность энергии
        """
        return self.calculator.compute_energy_density(field_derivatives)
    
    def compute_baryon_number(self, field_derivatives: Dict[str, Any]) -> float:
        """
        Вычисление барионного числа.
        
        Args:
            field_derivatives: Производные поля
            
        Returns:
            Барионное число
        """
        return self.calculator.compute_baryon_number(field_derivatives)
    
    def analyze_energy(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Анализ плотности энергии.
        
        Args:
            energy_density: Плотность энергии
            
        Returns:
            Результаты анализа
        """
        return self.analyzer.analyze_energy(energy_density)
    
    def optimize_constants(self, field_derivatives: Dict[str, Any]) -> Dict[str, float]:
        """
        Оптимизация констант Skyrme.
        
        Args:
            field_derivatives: Производные поля
            
        Returns:
            Оптимизированные константы
        """
        return self.optimizer.optimize_constants(
            self.c2, self.c4, self.c6, field_derivatives
        )
    
    def get_energy_report(self, energy_density: EnergyDensity) -> str:
        """
        Получение отчета по энергии.
        
        Args:
            energy_density: Плотность энергии
            
        Returns:
            Текстовый отчет
        """
        analysis = self.analyze_energy(energy_density)
        components = analysis['components']
        balance = analysis['balance']
        quality = analysis['quality']
        
        report = f"""
ENERGY DENSITY ANALYSIS
=======================

Energy Components:
  E₂ (c₂ term): {components['E2']:.6f}
  E₄ (c₄ term): {components['E4']:.6f}
  E₆ (c₆ term): {components['E6']:.6f}
  E_total: {components['E_total']:.6f}

Energy Balance:
  E₂/E_total: {balance['E2_ratio']:.3f} (target: 0.500)
  E₄/E_total: {balance['E4_ratio']:.3f} (target: 0.500)
  E₆/E_total: {balance['E6_ratio']:.3f}

Virial Condition (E₂ = E₄): {'✓ PASS' if analysis['virial_condition'] else '✗ FAIL'}

Quality Assessment:
  Overall Quality: {quality['overall_quality'].upper()}
  Balance Quality: {quality['balance_quality'].upper()}

Recommendations:
"""
        
        for rec in quality['recommendations']:
            report += f"  - {rec}\n"
        
        return report
```

## Объяснение

### Плотность энергии

Класс `EnergyDensity` представляет плотность энергии как сумму трех членов:
- **c₂ член** - кинетическая энергия поля
- **c₄ член** - энергия взаимодействия
- **c₆ член** - стабилизирующий член

### Плотность барионного заряда

Класс `BaryonDensity` вычисляет плотность барионного заряда b₀ согласно формуле:
$$b_0 = -\frac{1}{24\pi^2} \epsilon^{ijk} \text{Tr}(L_i L_j L_k)$$

### Калькулятор энергии

Класс `EnergyDensityCalculator` объединяет все компоненты для вычисления полной плотности энергии и барионного числа.

### Анализатор энергии

Класс `EnergyAnalyzer` анализирует качество энергетической модели:
- Проверяет вириальное условие E₂ = E₄
- Оценивает баланс энергии
- Предоставляет рекомендации по улучшению

### Оптимизатор констант

Класс `EnergyOptimizer` автоматически подбирает константы Skyrme для достижения целевого баланса энергии.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 5: Физические величины**, где реализуем расчет электрического заряда, радиуса и других физических параметров протона.
