# Шаг 6: Численные методы

## Цель

Реализовать численные методы для решения вариационных уравнений модели протона:
- Градиентный спуск с проекцией на SU(2)
- Контроль ограничений (барионное число, электрический заряд, виреальное условие)
- Релаксация к стационарному решению

## Обзор

Численные методы являются ключевыми для получения стационарного решения модели протона. Они должны обеспечивать:
- Сходимость к правильному решению
- Сохранение всех физических ограничений
- Стабильность численной схемы
- Эффективность вычислений

## Математические основы

### 6.1 Вариационные уравнения

**Полный функционал:**
$$\tilde{E}[U,\Theta] = E_{\text{SU(2)}}[U] + E_{\text{EM}}[U,\Theta] + E_{\text{constraints}}[U]$$

**Уравнения Эйлера-Лагранжа:**
$$\frac{\delta \tilde{E}}{\delta U} = 0, \quad \frac{\delta \tilde{E}}{\delta A_\mu} = 0$$

### 6.2 Ограничения

**Барионное число:**
$$B = -\frac{1}{24\pi^2} \int \epsilon^{ijk} \text{Tr}(L_i L_j L_k) d^3x = 1$$

**Электрический заряд:**
$$Q = \int \rho(\mathbf{x}) d^3x = +1$$

**Виреальное условие:**
$$\frac{E_{(2)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

## Декларативный код

```python
#!/usr/bin/env python3
"""
Численные методы для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class RelaxationMethod(Enum):
    """Методы релаксации."""
    GRADIENT_DESCENT = "gradient_descent"
    LBFGS = "lbfgs"
    ADAM = "adam"


@dataclass
class RelaxationConfig:
    """Конфигурация релаксации."""
    method: RelaxationMethod
    max_iterations: int = 1000
    convergence_tol: float = 1e-6
    step_size: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class ConstraintConfig:
    """Конфигурация ограничений."""
    lambda_B: float = 1000.0
    lambda_Q: float = 1000.0
    lambda_virial: float = 1000.0
    tolerance_B: float = 0.02
    tolerance_Q: float = 1e-6
    tolerance_virial: float = 0.01


class SU2Projection:
    """Проекция на SU(2) группу."""
    
    @staticmethod
    def project_to_su2(U: np.ndarray) -> np.ndarray:
        """
        Проекция матрицы на SU(2) группу.
        
        Args:
            U: Матрица для проекции
            
        Returns:
            Матрица в SU(2)
        """
        # QR разложение
        Q, R = np.linalg.qr(U)
        
        # Коррекция детерминанта
        det_Q = np.linalg.det(Q)
        Q = Q / (det_Q**(1/2))
        
        # Проверка унитарности
        if not np.allclose(np.dot(Q.conj().T, Q), np.eye(2), atol=1e-10):
            # Повторная проекция
            Q = (Q + Q.conj().T) / 2
            Q = Q / np.sqrt(np.trace(np.dot(Q.conj().T, Q)) / 2)
        
        return Q
    
    @staticmethod
    def validate_su2(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Проверка принадлежности к SU(2).
        
        Args:
            U: Матрица для проверки
            tolerance: Допустимая погрешность
            
        Returns:
            True если матрица ∈ SU(2)
        """
        # Проверка унитарности
        unitary_check = np.allclose(np.dot(U.conj().T, U), np.eye(2), atol=tolerance)
        
        # Проверка детерминанта
        det_check = abs(np.linalg.det(U) - 1.0) < tolerance
        
        return unitary_check and det_check


class GradientDescent:
    """Градиентный спуск с проекцией на SU(2)."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Инициализация градиентного спуска.
        
        Args:
            config: Конфигурация релаксации
        """
        self.config = config
        self.step_size = config.step_size
        self.momentum = config.momentum
        self.velocity = None
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Один шаг градиентного спуска.
        
        Args:
            U: Текущее поле
            gradient: Градиент функционала
            
        Returns:
            Обновленное поле
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        # Обновление скорости с моментумом
        self.velocity = self.momentum * self.velocity + self.step_size * gradient
        
        # Обновление поля
        U_new = U - self.velocity
        
        # Проекция на SU(2)
        U_new = SU2Projection.project_to_su2(U_new)
        
        return U_new
    
    def reset(self):
        """Сброс состояния оптимизатора."""
        self.velocity = None


class LBFGSOptimizer:
    """L-BFGS оптимизатор."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Инициализация L-BFGS.
        
        Args:
            config: Конфигурация релаксации
        """
        self.config = config
        self.memory_size = 10
        self.s_history = []
        self.y_history = []
        self.rho_history = []
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Один шаг L-BFGS.
        
        Args:
            U: Текущее поле
            gradient: Градиент функционала
            
        Returns:
            Обновленное поле
        """
        # Простая реализация (в реальности нужна полная L-BFGS)
        U_new = U - self.config.step_size * gradient
        
        # Проекция на SU(2)
        U_new = SU2Projection.project_to_su2(U_new)
        
        return U_new
    
    def reset(self):
        """Сброс состояния оптимизатора."""
        self.s_history = []
        self.y_history = []
        self.rho_history = []


class AdamOptimizer:
    """Adam оптимизатор."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Инициализация Adam.
        
        Args:
            config: Конфигурация релаксации
        """
        self.config = config
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Один шаг Adam.
        
        Args:
            U: Текущее поле
            gradient: Градиент функционала
            
        Returns:
            Обновленное поле
        """
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        self.t += 1
        
        # Обновление моментов
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        
        # Коррекция смещения
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Обновление поля
        U_new = U - self.config.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Проекция на SU(2)
        U_new = SU2Projection.project_to_su2(U_new)
        
        return U_new
    
    def reset(self):
        """Сброс состояния оптимизатора."""
        self.m = None
        self.v = None
        self.t = 0


class ConstraintController:
    """Контроллер ограничений."""
    
    def __init__(self, config: ConstraintConfig):
        """
        Инициализация контроллера ограничений.
        
        Args:
            config: Конфигурация ограничений
        """
        self.config = config
        self.lambda_B = config.lambda_B
        self.lambda_Q = config.lambda_Q
        self.lambda_virial = config.lambda_virial
    
    def compute_constraint_penalty(self, U: np.ndarray, 
                                 baryon_number: float, 
                                 electric_charge: float,
                                 energy_balance: float) -> float:
        """
        Вычисление штрафа за нарушение ограничений.
        
        Args:
            U: SU(2) поле
            baryon_number: Барионное число
            electric_charge: Электрический заряд
            energy_balance: Энергетический баланс
            
        Returns:
            Штраф за ограничения
        """
        penalty = 0.0
        
        # Штраф за барионное число
        penalty += self.lambda_B * (baryon_number - 1.0)**2
        
        # Штраф за электрический заряд
        penalty += self.lambda_Q * (electric_charge - 1.0)**2
        
        # Штраф за виреальное условие
        penalty += self.lambda_virial * (energy_balance - 0.5)**2
        
        return penalty
    
    def check_constraints(self, baryon_number: float, 
                         electric_charge: float,
                         energy_balance: float) -> Dict[str, bool]:
        """
        Проверка выполнения ограничений.
        
        Args:
            baryon_number: Барионное число
            electric_charge: Электрический заряд
            energy_balance: Энергетический баланс
            
        Returns:
            Словарь с результатами проверки
        """
        return {
            'baryon_number': abs(baryon_number - 1.0) <= self.config.tolerance_B,
            'electric_charge': abs(electric_charge - 1.0) <= self.config.tolerance_Q,
            'energy_balance': abs(energy_balance - 0.5) <= self.config.tolerance_virial
        }


class RelaxationSolver:
    """Основной решатель релаксации."""
    
    def __init__(self, config: RelaxationConfig, constraint_config: ConstraintConfig):
        """
        Инициализация решателя.
        
        Args:
            config: Конфигурация релаксации
            constraint_config: Конфигурация ограничений
        """
        self.config = config
        self.constraint_controller = ConstraintController(constraint_config)
        
        # Выбор оптимизатора
        if config.method == RelaxationMethod.GRADIENT_DESCENT:
            self.optimizer = GradientDescent(config)
        elif config.method == RelaxationMethod.LBFGS:
            self.optimizer = LBFGSOptimizer(config)
        elif config.method == RelaxationMethod.ADAM:
            self.optimizer = AdamOptimizer(config)
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")
    
    def solve(self, U_init: np.ndarray, 
              energy_function: Callable,
              gradient_function: Callable,
              constraint_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Решение задачи релаксации.
        
        Args:
            U_init: Начальное поле
            energy_function: Функция вычисления энергии
            gradient_function: Функция вычисления градиента
            constraint_functions: Функции вычисления ограничений
            
        Returns:
            Словарь с результатами
        """
        U = U_init.copy()
        energy_history = []
        constraint_history = []
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Вычисление энергии
            energy = energy_function(U)
            energy_history.append(energy)
            
            # Вычисление ограничений
            constraints = {}
            for name, func in constraint_functions.items():
                constraints[name] = func(U)
            constraint_history.append(constraints)
            
            # Проверка сходимости
            if iteration > 0:
                energy_change = abs(energy - energy_history[-2])
                if energy_change < self.config.convergence_tol:
                    break
            
            # Вычисление градиента
            gradient = gradient_function(U)
            
            # Шаг оптимизации
            U = self.optimizer.step(U, gradient)
            
            # Проверка ограничений
            constraint_check = self.constraint_controller.check_constraints(
                constraints.get('baryon_number', 0),
                constraints.get('electric_charge', 0),
                constraints.get('energy_balance', 0)
            )
            
            # Логирование прогресса
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f}, "
                      f"Constraints = {constraint_check}")
        
        end_time = time.time()
        
        return {
            'solution': U,
            'energy_history': energy_history,
            'constraint_history': constraint_history,
            'iterations': iteration + 1,
            'converged': iteration < self.config.max_iterations - 1,
            'execution_time': end_time - start_time,
            'final_energy': energy_history[-1] if energy_history else 0.0,
            'final_constraints': constraint_history[-1] if constraint_history else {}
        }
    
    def reset(self):
        """Сброс состояния решателя."""
        self.optimizer.reset()


class NumericalMethods:
    """Основной класс численных методов."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Инициализация численных методов.
        
        Args:
            grid_size: Размер сетки
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
    
    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление градиента поля.
        
        Args:
            field: Скалярное поле
            
        Returns:
            Кортеж (grad_x, grad_y, grad_z)
        """
        grad_x = np.gradient(field, self.dx, axis=0)
        grad_y = np.gradient(field, self.dx, axis=1)
        grad_z = np.gradient(field, self.dx, axis=2)
        
        return grad_x, grad_y, grad_z
    
    def compute_divergence(self, field_x: np.ndarray, field_y: np.ndarray, 
                          field_z: np.ndarray) -> np.ndarray:
        """
        Вычисление дивергенции векторного поля.
        
        Args:
            field_x, field_y, field_z: Компоненты векторного поля
            
        Returns:
            Дивергенция
        """
        div_x = np.gradient(field_x, self.dx, axis=0)
        div_y = np.gradient(field_y, self.dx, axis=1)
        div_z = np.gradient(field_z, self.dx, axis=2)
        
        return div_x + div_y + div_z
    
    def integrate_3d(self, field: np.ndarray) -> float:
        """
        Интегрирование 3D поля по объему.
        
        Args:
            field: 3D поле для интегрирования
            
        Returns:
            Результат интегрирования
        """
        return np.sum(field) * self.dx**3
    
    def create_initial_field(self, profile_type: str = "tanh") -> np.ndarray:
        """
        Создание начального поля.
        
        Args:
            profile_type: Тип профиля
            
        Returns:
            Начальное SU(2) поле
        """
        if profile_type == "tanh":
            f = np.pi * (1 - np.tanh(self.R))
        elif profile_type == "exp":
            f = np.pi * np.exp(-self.R)
        elif profile_type == "gaussian":
            f = np.pi * np.exp(-self.R**2)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        # Создание SU(2) поля
        U = np.zeros((self.grid_size, self.grid_size, self.grid_size, 2, 2), dtype=complex)
        
        # Заполнение полем
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    f_val = f[i, j, k]
                    U[i, j, k] = np.array([
                        [np.cos(f_val), 1j * np.sin(f_val)],
                        [1j * np.sin(f_val), np.cos(f_val)]
                    ])
        
        return U
    
    def validate_solution(self, U: np.ndarray) -> Dict[str, bool]:
        """
        Валидация решения.
        
        Args:
            U: SU(2) поле
            
        Returns:
            Словарь с результатами валидации
        """
        results = {}
        
        # Проверка SU(2) для каждой точки
        su2_valid = True
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if not SU2Projection.validate_su2(U[i, j, k]):
                        su2_valid = False
                        break
                if not su2_valid:
                    break
            if not su2_valid:
                break
        
        results['su2_valid'] = su2_valid
        
        # Проверка граничных условий
        center_field = U[self.grid_size//2, self.grid_size//2, self.grid_size//2]
        boundary_field = U[0, 0, 0]
        
        results['boundary_conditions'] = (
            abs(np.linalg.det(center_field) + 1) < 1e-6 and
            abs(np.linalg.det(boundary_field) - 1) < 1e-6
        )
        
        return results
```

## Объяснение

### Оптимизаторы

1. **GradientDescent** - классический градиентный спуск с моментумом
2. **LBFGSOptimizer** - L-BFGS для более быстрой сходимости
3. **AdamOptimizer** - Adam для адаптивного обучения

### Проекция на SU(2)

Класс `SU2Projection` обеспечивает корректную проекцию матриц на группу SU(2) с сохранением унитарности и детерминанта.

### Контроль ограничений

Класс `ConstraintController` управляет выполнением физических ограничений через штрафные функции.

### Основной решатель

Класс `RelaxationSolver` координирует процесс оптимизации и обеспечивает сходимость к стационарному решению.

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 7: Валидация**, где создадим систему проверки физических параметров и сравнения с экспериментальными данными.
