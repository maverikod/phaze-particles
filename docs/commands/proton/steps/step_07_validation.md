# Шаг 7: Валидация

## Цель

Реализовать систему валидации модели протона:
- Проверка физических параметров против экспериментальных данных
- Анализ отклонений и оценка качества модели
- Генерация отчетов о валидации
- Рекомендации по улучшению модели

## Обзор

Валидация является критически важным этапом для обеспечения физической корректности модели. Она включает:
- Сравнение вычисленных параметров с экспериментальными значениями
- Анализ статистических отклонений
- Оценку качества модели
- Генерацию детальных отчетов

## Физические параметры для валидации

### 7.1 Основные параметры

**Масса протона:**
$$M_p = 938.272 \pm 0.006 \text{ МэВ}$$

**Радиус зарядового распределения:**
$$r_E = 0.841 \pm 0.019 \text{ фм}$$

**Магнитный момент:**
$$\mu_p = 2.793 \pm 0.001 \mu_N$$

**Электрический заряд:**
$$Q = +1 \text{ (точное)}$$

**Барионное число:**
$$B = 1 \text{ (точное)}$$

### 7.2 Дополнительные параметры

**Энергетический баланс:**
$$\frac{E_{(2)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

**Форм-факторы:**
$$G_E(0) = 1, \quad G_M(0) = 2.793$$

## Декларативный код

```python
#!/usr/bin/env python3
"""
Валидация модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
from datetime import datetime


class ValidationStatus(Enum):
    """Статус валидации."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ExperimentalData:
    """Экспериментальные данные."""
    proton_mass: float = 938.272
    proton_mass_error: float = 0.006
    charge_radius: float = 0.841
    charge_radius_error: float = 0.019
    magnetic_moment: float = 2.793
    magnetic_moment_error: float = 0.001
    electric_charge: float = 1.0
    baryon_number: float = 1.0


@dataclass
class CalculatedData:
    """Вычисленные данные."""
    proton_mass: float
    charge_radius: float
    magnetic_moment: float
    electric_charge: float
    baryon_number: float
    energy_balance: float
    total_energy: float
    execution_time: float


@dataclass
class ValidationResult:
    """Результат валидации."""
    parameter_name: str
    calculated_value: float
    experimental_value: float
    experimental_error: float
    deviation: float
    deviation_percent: float
    within_tolerance: bool
    status: ValidationStatus


class ParameterValidator:
    """Валидатор физических параметров."""
    
    def __init__(self, experimental_data: ExperimentalData):
        """
        Инициализация валидатора.
        
        Args:
            experimental_data: Экспериментальные данные
        """
        self.experimental_data = experimental_data
    
    def validate_mass(self, calculated_mass: float) -> ValidationResult:
        """
        Валидация массы протона.
        
        Args:
            calculated_mass: Вычисленная масса
            
        Returns:
            Результат валидации
        """
        exp_mass = self.experimental_data.proton_mass
        exp_error = self.experimental_data.proton_mass_error
        
        deviation = abs(calculated_mass - exp_mass)
        deviation_percent = (deviation / exp_mass) * 100
        within_tolerance = deviation <= exp_error
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="proton_mass",
            calculated_value=calculated_mass,
            experimental_value=exp_mass,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )
    
    def validate_radius(self, calculated_radius: float) -> ValidationResult:
        """
        Валидация радиуса зарядового распределения.
        
        Args:
            calculated_radius: Вычисленный радиус
            
        Returns:
            Результат валидации
        """
        exp_radius = self.experimental_data.charge_radius
        exp_error = self.experimental_data.charge_radius_error
        
        deviation = abs(calculated_radius - exp_radius)
        deviation_percent = (deviation / exp_radius) * 100
        within_tolerance = deviation <= exp_error
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="charge_radius",
            calculated_value=calculated_radius,
            experimental_value=exp_radius,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )
    
    def validate_magnetic_moment(self, calculated_moment: float) -> ValidationResult:
        """
        Валидация магнитного момента.
        
        Args:
            calculated_moment: Вычисленный магнитный момент
            
        Returns:
            Результат валидации
        """
        exp_moment = self.experimental_data.magnetic_moment
        exp_error = self.experimental_data.magnetic_moment_error
        
        deviation = abs(calculated_moment - exp_moment)
        deviation_percent = (deviation / exp_moment) * 100
        within_tolerance = deviation <= exp_error
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="magnetic_moment",
            calculated_value=calculated_moment,
            experimental_value=exp_moment,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )
    
    def validate_charge(self, calculated_charge: float) -> ValidationResult:
        """
        Валидация электрического заряда.
        
        Args:
            calculated_charge: Вычисленный заряд
            
        Returns:
            Результат валидации
        """
        exp_charge = self.experimental_data.electric_charge
        tolerance = 1e-6  # Точное значение
        
        deviation = abs(calculated_charge - exp_charge)
        deviation_percent = deviation * 100
        within_tolerance = deviation <= tolerance
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 1e-4:
            status = ValidationStatus.GOOD
        elif deviation <= 1e-3:
            status = ValidationStatus.FAIR
        elif deviation <= 1e-2:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="electric_charge",
            calculated_value=calculated_charge,
            experimental_value=exp_charge,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )
    
    def validate_baryon_number(self, calculated_baryon: float) -> ValidationResult:
        """
        Валидация барионного числа.
        
        Args:
            calculated_baryon: Вычисленное барионное число
            
        Returns:
            Результат валидации
        """
        exp_baryon = self.experimental_data.baryon_number
        tolerance = 0.02  # Допуск для барионного числа
        
        deviation = abs(calculated_baryon - exp_baryon)
        deviation_percent = deviation * 100
        within_tolerance = deviation <= tolerance
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 0.05:
            status = ValidationStatus.GOOD
        elif deviation <= 0.1:
            status = ValidationStatus.FAIR
        elif deviation <= 0.2:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="baryon_number",
            calculated_value=calculated_baryon,
            experimental_value=exp_baryon,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )
    
    def validate_energy_balance(self, energy_balance: float) -> ValidationResult:
        """
        Валидация энергетического баланса.
        
        Args:
            energy_balance: Энергетический баланс E₂/E₄
            
        Returns:
            Результат валидации
        """
        target_balance = 0.5
        tolerance = 0.01
        
        deviation = abs(energy_balance - target_balance)
        deviation_percent = deviation * 200  # В процентах от 0.5
        within_tolerance = deviation <= tolerance
        
        # Определение статуса
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 0.02:
            status = ValidationStatus.GOOD
        elif deviation <= 0.05:
            status = ValidationStatus.FAIR
        elif deviation <= 0.1:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            parameter_name="energy_balance",
            calculated_value=energy_balance,
            experimental_value=target_balance,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status
        )


class ModelQualityAssessor:
    """Оценщик качества модели."""
    
    def __init__(self):
        """Инициализация оценщика качества."""
        self.weights = {
            'proton_mass': 0.25,
            'charge_radius': 0.25,
            'magnetic_moment': 0.20,
            'electric_charge': 0.15,
            'baryon_number': 0.10,
            'energy_balance': 0.05
        }
    
    def assess_quality(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Оценка качества модели.
        
        Args:
            validation_results: Результаты валидации
            
        Returns:
            Словарь с оценкой качества
        """
        # Подсчет статусов
        status_counts = {}
        for status in ValidationStatus:
            status_counts[status] = 0
        
        for result in validation_results:
            status_counts[result.status] += 1
        
        # Вычисление взвешенной оценки
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            weight = self.weights.get(result.parameter_name, 0.0)
            total_weight += weight
            
            # Преобразование статуса в числовую оценку
            if result.status == ValidationStatus.EXCELLENT:
                score = 1.0
            elif result.status == ValidationStatus.GOOD:
                score = 0.8
            elif result.status == ValidationStatus.FAIR:
                score = 0.6
            elif result.status == ValidationStatus.POOR:
                score = 0.4
            else:  # FAILED
                score = 0.0
            
            weighted_score += weight * score
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Определение общего статуса
        if weighted_score >= 0.9:
            overall_status = ValidationStatus.EXCELLENT
        elif weighted_score >= 0.7:
            overall_status = ValidationStatus.GOOD
        elif weighted_score >= 0.5:
            overall_status = ValidationStatus.FAIR
        elif weighted_score >= 0.3:
            overall_status = ValidationStatus.POOR
        else:
            overall_status = ValidationStatus.FAILED
        
        return {
            'overall_status': overall_status,
            'weighted_score': weighted_score,
            'status_counts': status_counts,
            'total_parameters': len(validation_results),
            'passed_parameters': sum(1 for r in validation_results if r.within_tolerance)
        }


class ValidationReportGenerator:
    """Генератор отчетов валидации."""
    
    def __init__(self):
        """Инициализация генератора отчетов."""
        self.timestamp = datetime.now()
    
    def generate_text_report(self, validation_results: List[ValidationResult], 
                           quality_assessment: Dict[str, Any]) -> str:
        """
        Генерация текстового отчета.
        
        Args:
            validation_results: Результаты валидации
            quality_assessment: Оценка качества
            
        Returns:
            Текстовый отчет
        """
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ВАЛИДАЦИИ МОДЕЛИ ПРОТОНА")
        report.append("=" * 80)
        report.append(f"Дата и время: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Общая оценка
        report.append("ОБЩАЯ ОЦЕНКА:")
        report.append("-" * 40)
        report.append(f"Статус: {quality_assessment['overall_status'].value.upper()}")
        report.append(f"Взвешенная оценка: {quality_assessment['weighted_score']:.3f}")
        report.append(f"Пройдено параметров: {quality_assessment['passed_parameters']}/{quality_assessment['total_parameters']}")
        report.append("")
        
        # Детальные результаты
        report.append("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        report.append("-" * 40)
        
        for result in validation_results:
            report.append(f"Параметр: {result.parameter_name}")
            report.append(f"  Вычисленное значение: {result.calculated_value:.6f}")
            report.append(f"  Экспериментальное значение: {result.experimental_value:.6f} ± {result.experimental_error:.6f}")
            report.append(f"  Отклонение: {result.deviation:.6f} ({result.deviation_percent:.2f}%)")
            report.append(f"  В пределах допуска: {'ДА' if result.within_tolerance else 'НЕТ'}")
            report.append(f"  Статус: {result.status.value.upper()}")
            report.append("")
        
        # Рекомендации
        report.append("РЕКОМЕНДАЦИИ:")
        report.append("-" * 40)
        
        failed_params = [r for r in validation_results if r.status == ValidationStatus.FAILED]
        poor_params = [r for r in validation_results if r.status == ValidationStatus.POOR]
        
        if failed_params:
            report.append("КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            for param in failed_params:
                report.append(f"  - {param.parameter_name}: требует немедленного исправления")
        
        if poor_params:
            report.append("ПРОБЛЕМЫ ТРЕБУЮЩИЕ ВНИМАНИЯ:")
            for param in poor_params:
                report.append(f"  - {param.parameter_name}: рекомендуется улучшение")
        
        if not failed_params and not poor_params:
            report.append("Модель соответствует всем требованиям валидации.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_json_report(self, validation_results: List[ValidationResult], 
                           quality_assessment: Dict[str, Any]) -> str:
        """
        Генерация JSON отчета.
        
        Args:
            validation_results: Результаты валидации
            quality_assessment: Оценка качества
            
        Returns:
            JSON отчет
        """
        report_data = {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': quality_assessment['overall_status'].value,
            'weighted_score': quality_assessment['weighted_score'],
            'status_counts': {k.value: v for k, v in quality_assessment['status_counts'].items()},
            'total_parameters': quality_assessment['total_parameters'],
            'passed_parameters': quality_assessment['passed_parameters'],
            'validation_results': []
        }
        
        for result in validation_results:
            report_data['validation_results'].append({
                'parameter_name': result.parameter_name,
                'calculated_value': result.calculated_value,
                'experimental_value': result.experimental_value,
                'experimental_error': result.experimental_error,
                'deviation': result.deviation,
                'deviation_percent': result.deviation_percent,
                'within_tolerance': result.within_tolerance,
                'status': result.status.value
            })
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def generate_plots(self, validation_results: List[ValidationResult], 
                      output_dir: str = "plots"):
        """
        Генерация графиков валидации.
        
        Args:
            validation_results: Результаты валидации
            output_dir: Директория для сохранения графиков
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # График отклонений
        fig, ax = plt.subplots(figsize=(12, 8))
        
        param_names = [r.parameter_name for r in validation_results]
        deviations = [r.deviation_percent for r in validation_results]
        colors = ['green' if r.within_tolerance else 'red' for r in validation_results]
        
        bars = ax.bar(param_names, deviations, color=colors, alpha=0.7)
        ax.set_ylabel('Отклонение (%)')
        ax.set_title('Отклонения вычисленных параметров от экспериментальных значений')
        ax.tick_params(axis='x', rotation=45)
        
        # Добавление значений на столбцы
        for bar, deviation in zip(bars, deviations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{deviation:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/validation_deviations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # График сравнения значений
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(param_names))
        width = 0.35
        
        calculated_values = [r.calculated_value for r in validation_results]
        experimental_values = [r.experimental_value for r in validation_results]
        experimental_errors = [r.experimental_error for r in validation_results]
        
        bars1 = ax.bar(x - width/2, calculated_values, width, label='Вычисленные', alpha=0.7)
        bars2 = ax.bar(x + width/2, experimental_values, width, label='Экспериментальные', alpha=0.7)
        
        # Добавление ошибок
        ax.errorbar(x + width/2, experimental_values, yerr=experimental_errors, 
                   fmt='none', color='black', capsize=5)
        
        ax.set_ylabel('Значения параметров')
        ax.set_title('Сравнение вычисленных и экспериментальных значений')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/validation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


class ValidationSystem:
    """Основная система валидации."""
    
    def __init__(self, experimental_data: ExperimentalData):
        """
        Инициализация системы валидации.
        
        Args:
            experimental_data: Экспериментальные данные
        """
        self.experimental_data = experimental_data
        self.validator = ParameterValidator(experimental_data)
        self.quality_assessor = ModelQualityAssessor()
        self.report_generator = ValidationReportGenerator()
    
    def validate_model(self, calculated_data: CalculatedData) -> Dict[str, Any]:
        """
        Валидация модели.
        
        Args:
            calculated_data: Вычисленные данные
            
        Returns:
            Словарь с результатами валидации
        """
        # Валидация всех параметров
        validation_results = [
            self.validator.validate_mass(calculated_data.proton_mass),
            self.validator.validate_radius(calculated_data.charge_radius),
            self.validator.validate_magnetic_moment(calculated_data.magnetic_moment),
            self.validator.validate_charge(calculated_data.electric_charge),
            self.validator.validate_baryon_number(calculated_data.baryon_number),
            self.validator.validate_energy_balance(calculated_data.energy_balance)
        ]
        
        # Оценка качества
        quality_assessment = self.quality_assessor.assess_quality(validation_results)
        
        # Генерация отчетов
        text_report = self.report_generator.generate_text_report(validation_results, quality_assessment)
        json_report = self.report_generator.generate_json_report(validation_results, quality_assessment)
        
        # Генерация графиков
        self.report_generator.generate_plots(validation_results)
        
        return {
            'validation_results': validation_results,
            'quality_assessment': quality_assessment,
            'text_report': text_report,
            'json_report': json_report,
            'overall_status': quality_assessment['overall_status'],
            'weighted_score': quality_assessment['weighted_score']
        }
    
    def save_reports(self, validation_results: Dict[str, Any], output_dir: str = "validation_reports"):
        """
        Сохранение отчетов.
        
        Args:
            validation_results: Результаты валидации
            output_dir: Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение текстового отчета
        with open(f"{output_dir}/validation_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(validation_results['text_report'])
        
        # Сохранение JSON отчета
        with open(f"{output_dir}/validation_report_{timestamp}.json", 'w', encoding='utf-8') as f:
            f.write(validation_results['json_report'])
        
        print(f"Отчеты сохранены в директории: {output_dir}")
```

## Объяснение

### Валидация параметров

Класс `ParameterValidator` проверяет каждый физический параметр против экспериментальных данных и определяет статус валидации.

### Оценка качества

Класс `ModelQualityAssessor` вычисляет взвешенную оценку качества модели на основе результатов валидации всех параметров.

### Генерация отчетов

Класс `ValidationReportGenerator` создает детальные отчеты в текстовом и JSON форматах, а также генерирует графики для визуализации результатов.

### Система валидации

Класс `ValidationSystem` координирует весь процесс валидации и обеспечивает генерацию полных отчетов.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 8: Интеграция**, где объединим все компоненты в единую модель протона.
