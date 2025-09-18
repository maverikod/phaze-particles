# Шаг 2: Тороидальные геометрии

## Цель

Реализовать три тороидальные конфигурации для модели протона: 120°, клевер (clover) и декартовая (cartesian).

## Обзор

Согласно ТЗ, модель протона основана на трех тороидальных структурах, которые определяют направление поля n̂(x) в SU(2) поле. Каждая конфигурация имеет свою геометрию и параметры.

## Три конфигурации

### 1. 120° конфигурация (ПРИОРИТЕТНАЯ)
- Три тора расположены под углом 120° друг к другу
- Симметрия C₃ (поворот на 120°)
- **Базовая конфигурация для калибровки модели**
- **Используется для определения констант Skyrme**
- **Основная конфигурация для валидации физических параметров**

### 2. Клевер (Clover) конфигурация  
- Три тора в форме клевера (альтернативная гипотеза)
- Симметрия C₃ (поворот на 120°)
- Более компактная структура
- **Альтернативная гипотеза для сравнения с основной 120° конфигурацией**

### 3. Декартовая (Cartesian) конфигурация
- Три тора вдоль осей x, y, z
- Симметрия D₄ (диэдральная группа)
- Простейшая геометрия
- **Тестовая конфигурация для отладки**

## Стратегия использования конфигураций

### Этап 1: Калибровка (120° конфигурация)
1. Использовать 120° конфигурацию для определения констант Skyrme
2. Калибровать по экспериментальным данным (масса, радиус, магнитный момент)
3. Проверить виреальное равенство E₂ = E₄ + 3E₆ (при c₆=0 → 50:50)

### Этап 2: Валидация (все конфигурации)
1. Сравнить результаты всех трех конфигураций
2. Проверить стабильность физических параметров
3. Оценить влияние геометрии на результаты

### Этап 3: Оптимизация (120° конфигурация)
1. Использовать 120° конфигурацию для финальной оптимизации
2. Применить фазовые координаты для анализа топологий
3. Провести CUDA оптимизацию

## Математическое описание

Каждая тороидальная структура определяется:
- Центром тора (x₀, y₀, z₀)
- Радиусом тора R
- Направлением оси тора n̂
- Толщиной тора δ

## Декларативный код

```python
#!/usr/bin/env python3
"""
Тороидальные геометрии для модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class TorusConfiguration(Enum):
    """Типы тороидальных конфигураций."""
    CONFIG_120_DEG = "120deg"
    CONFIG_CLOVER = "clover"
    CONFIG_CARTESIAN = "cartesian"


@dataclass
class TorusParameters:
    """Параметры отдельного тора."""
    center: Tuple[float, float, float]  # Центр тора (x₀, y₀, z₀)
    radius: float  # Радиус тора R
    axis: Tuple[float, float, float]  # Направление оси n̂
    thickness: float  # Толщина тора δ
    strength: float  # Сила поля в торе


@dataclass
class TorusGeometry:
    """Геометрия тороидальной конфигурации."""
    config_type: TorusConfiguration
    tori: List[TorusParameters]
    symmetry_group: str
    description: str


class Torus120Degrees:
    """120° тороидальная конфигурация."""
    
    def __init__(self, radius: float = 1.0, thickness: float = 0.2, 
                 strength: float = 1.0):
        """
        Инициализация 120° конфигурации.
        
        Args:
            radius: Радиус тора
            thickness: Толщина тора
            strength: Сила поля
        """
        self.radius = radius
        self.thickness = thickness
        self.strength = strength
        
        # Создание трех торов под углом 120°
        self.tori = self._create_120_degree_tori()
    
    def _create_120_degree_tori(self) -> List[TorusParameters]:
        """
        Создание трех торов с углом 120° между ними.
        
        Returns:
            Список параметров торов
        """
        tori = []
        
        # Тор 1: в плоскости xy, ось z
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(0.0, 0.0, 1.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 2: повернут на 120° вокруг оси z
        angle_120 = 2 * math.pi / 3
        cos_120 = math.cos(angle_120)
        sin_120 = math.sin(angle_120)
        
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(cos_120, sin_120, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 3: повернут на 240° вокруг оси z
        angle_240 = 4 * math.pi / 3
        cos_240 = math.cos(angle_240)
        sin_240 = math.sin(angle_240)
        
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(cos_240, sin_240, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        return tori
    
    def get_field_direction(self, x: np.ndarray, y: np.ndarray, 
                           z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление направления поля n̂(x) для 120° конфигурации.
        
        Args:
            x, y, z: Координатные сетки
            
        Returns:
            Кортеж (n_x, n_y, n_z) компонент направления поля
        """
        n_x = np.zeros_like(x)
        n_y = np.zeros_like(y)
        n_z = np.zeros_like(z)
        
        for torus in self.tori:
            # Вычисление расстояния до тора
            distance = self._distance_to_torus(x, y, z, torus)
            
            # Весовая функция (гауссова)
            weight = np.exp(-(distance**2) / (2 * torus.thickness**2))
            
            # Добавление вклада от тора
            n_x += weight * torus.axis[0] * torus.strength
            n_y += weight * torus.axis[1] * torus.strength
            n_z += weight * torus.axis[2] * torus.strength
        
        # Нормализация
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, 1.0)  # Избегаем деления на ноль
        
        return n_x/norm, n_y/norm, n_z/norm
    
    def _distance_to_torus(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          torus: TorusParameters) -> np.ndarray:
        """
        Вычисление расстояния до тора.
        
        Args:
            x, y, z: Координатные сетки
            torus: Параметры тора
            
        Returns:
            Расстояние до тора
        """
        # Смещение координат относительно центра тора
        dx = x - torus.center[0]
        dy = y - torus.center[1]
        dz = z - torus.center[2]
        
        # Проекция на ось тора
        axis_proj = (dx * torus.axis[0] + 
                    dy * torus.axis[1] + 
                    dz * torus.axis[2])
        
        # Координаты в плоскости, перпендикулярной оси тора
        perp_x = dx - axis_proj * torus.axis[0]
        perp_y = dy - axis_proj * torus.axis[1]
        perp_z = dz - axis_proj * torus.axis[2]
        
        # Расстояние до оси тора
        perp_distance = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
        
        # Расстояние до поверхности тора
        return np.abs(perp_distance - torus.radius)


class TorusClover:
    """Клевер тороидальная конфигурация."""
    
    def __init__(self, radius: float = 1.0, thickness: float = 0.2,
                 strength: float = 1.0):
        """
        Инициализация клевер конфигурации.
        
        Args:
            radius: Радиус тора
            thickness: Толщина тора
            strength: Сила поля
        """
        self.radius = radius
        self.thickness = thickness
        self.strength = strength
        
        # Создание трех торов в форме клевера
        self.tori = self._create_clover_tori()
    
    def _create_clover_tori(self) -> List[TorusParameters]:
        """
        Создание трех торов в форме клевера.
        
        Returns:
            Список параметров торов
        """
        tori = []
        
        # Тор 1: вдоль оси x
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(1.0, 0.0, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 2: вдоль оси y
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(0.0, 1.0, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 3: диагональный в плоскости xy
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(1.0/math.sqrt(2), 1.0/math.sqrt(2), 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        return tori
    
    def get_field_direction(self, x: np.ndarray, y: np.ndarray,
                           z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление направления поля n̂(x) для клевер конфигурации.
        
        Args:
            x, y, z: Координатные сетки
            
        Returns:
            Кортеж (n_x, n_y, n_z) компонент направления поля
        """
        n_x = np.zeros_like(x)
        n_y = np.zeros_like(y)
        n_z = np.zeros_like(z)
        
        for torus in self.tori:
            # Вычисление расстояния до тора
            distance = self._distance_to_torus(x, y, z, torus)
            
            # Весовая функция (гауссова)
            weight = np.exp(-(distance**2) / (2 * torus.thickness**2))
            
            # Добавление вклада от тора
            n_x += weight * torus.axis[0] * torus.strength
            n_y += weight * torus.axis[1] * torus.strength
            n_z += weight * torus.axis[2] * torus.strength
        
        # Нормализация
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, 1.0)
        
        return n_x/norm, n_y/norm, n_z/norm
    
    def _distance_to_torus(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          torus: TorusParameters) -> np.ndarray:
        """
        Вычисление расстояния до тора (аналогично 120° конфигурации).
        """
        dx = x - torus.center[0]
        dy = y - torus.center[1]
        dz = z - torus.center[2]
        
        axis_proj = (dx * torus.axis[0] + 
                    dy * torus.axis[1] + 
                    dz * torus.axis[2])
        
        perp_x = dx - axis_proj * torus.axis[0]
        perp_y = dy - axis_proj * torus.axis[1]
        perp_z = dz - axis_proj * torus.axis[2]
        
        perp_distance = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
        
        return np.abs(perp_distance - torus.radius)


class TorusCartesian:
    """Декартовая тороидальная конфигурация."""
    
    def __init__(self, radius: float = 1.0, thickness: float = 0.2,
                 strength: float = 1.0):
        """
        Инициализация декартовой конфигурации.
        
        Args:
            radius: Радиус тора
            thickness: Толщина тора
            strength: Сила поля
        """
        self.radius = radius
        self.thickness = thickness
        self.strength = strength
        
        # Создание трех торов вдоль осей
        self.tori = self._create_cartesian_tori()
    
    def _create_cartesian_tori(self) -> List[TorusParameters]:
        """
        Создание трех торов вдоль декартовых осей.
        
        Returns:
            Список параметров торов
        """
        tori = []
        
        # Тор 1: вдоль оси x
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(1.0, 0.0, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 2: вдоль оси y
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(0.0, 1.0, 0.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        # Тор 3: вдоль оси z
        tori.append(TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=self.radius,
            axis=(0.0, 0.0, 1.0),
            thickness=self.thickness,
            strength=self.strength
        ))
        
        return tori
    
    def get_field_direction(self, x: np.ndarray, y: np.ndarray,
                           z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление направления поля n̂(x) для декартовой конфигурации.
        
        Args:
            x, y, z: Координатные сетки
            
        Returns:
            Кортеж (n_x, n_y, n_z) компонент направления поля
        """
        n_x = np.zeros_like(x)
        n_y = np.zeros_like(y)
        n_z = np.zeros_like(z)
        
        for torus in self.tori:
            # Вычисление расстояния до тора
            distance = self._distance_to_torus(x, y, z, torus)
            
            # Весовая функция (гауссова)
            weight = np.exp(-(distance**2) / (2 * torus.thickness**2))
            
            # Добавление вклада от тора
            n_x += weight * torus.axis[0] * torus.strength
            n_y += weight * torus.axis[1] * torus.strength
            n_z += weight * torus.axis[2] * torus.strength
        
        # Нормализация
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, 1.0)
        
        return n_x/norm, n_y/norm, n_z/norm
    
    def _distance_to_torus(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          torus: TorusParameters) -> np.ndarray:
        """
        Вычисление расстояния до тора (аналогично другим конфигурациям).
        """
        dx = x - torus.center[0]
        dy = y - torus.center[1]
        dz = z - torus.center[2]
        
        axis_proj = (dx * torus.axis[0] + 
                    dy * torus.axis[1] + 
                    dz * torus.axis[2])
        
        perp_x = dx - axis_proj * torus.axis[0]
        perp_y = dy - axis_proj * torus.axis[1]
        perp_z = dz - axis_proj * torus.axis[2]
        
        perp_distance = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
        
        return np.abs(perp_distance - torus.radius)


class TorusGeometryManager:
    """Менеджер тороидальных геометрий."""
    
    def __init__(self):
        """Инициализация менеджера геометрий."""
        self.configurations = {
            TorusConfiguration.CONFIG_120_DEG: Torus120Degrees,
            TorusConfiguration.CONFIG_CLOVER: TorusClover,
            TorusConfiguration.CONFIG_CARTESIAN: TorusCartesian
        }
    
    def create_configuration(self, config_type: TorusConfiguration,
                           radius: float = 1.0, thickness: float = 0.2,
                           strength: float = 1.0) -> Any:
        """
        Создание тороидальной конфигурации.
        
        Args:
            config_type: Тип конфигурации
            radius: Радиус тора
            thickness: Толщина тора
            strength: Сила поля
            
        Returns:
            Экземпляр конфигурации
        """
        if config_type not in self.configurations:
            raise ValueError(f"Unknown configuration type: {config_type}")
        
        config_class = self.configurations[config_type]
        return config_class(radius, thickness, strength)
    
    def get_available_configurations(self) -> List[TorusConfiguration]:
        """
        Получить список доступных конфигураций.
        
        Returns:
            Список доступных конфигураций
        """
        return list(self.configurations.keys())
    
    def get_configuration_info(self, config_type: TorusConfiguration) -> Dict[str, Any]:
        """
        Получить информацию о конфигурации.
        
        Args:
            config_type: Тип конфигурации
            
        Returns:
            Словарь с информацией о конфигурации
        """
        info = {
            TorusConfiguration.CONFIG_120_DEG: {
                'name': '120° Configuration',
                'description': 'Three tori at 120° angles with C₃ symmetry',
                'symmetry_group': 'C₃',
                'num_tori': 3
            },
            TorusConfiguration.CONFIG_CLOVER: {
                'name': 'Clover Configuration',
                'description': 'Three tori in clover shape with C₃ symmetry',
                'symmetry_group': 'C₃',
                'num_tori': 3
            },
            TorusConfiguration.CONFIG_CARTESIAN: {
                'name': 'Cartesian Configuration',
                'description': 'Three tori along x, y, z axes with D₄ symmetry',
                'symmetry_group': 'D₄',
                'num_tori': 3
            }
        }
        
        return info.get(config_type, {})
    
    def validate_configuration(self, config: Any) -> bool:
        """
        Проверка корректности конфигурации.
        
        Args:
            config: Конфигурация для проверки
            
        Returns:
            True если конфигурация корректна
        """
        if not hasattr(config, 'tori'):
            return False
        
        if not isinstance(config.tori, list):
            return False
        
        if len(config.tori) == 0:
            return False
        
        # Проверка параметров каждого тора
        for torus in config.tori:
            if not isinstance(torus, TorusParameters):
                return False
            
            if torus.radius <= 0 or torus.thickness <= 0:
                return False
            
            # Проверка нормализации оси
            axis_norm = math.sqrt(sum(a**2 for a in torus.axis))
            if abs(axis_norm - 1.0) > 1e-10:
                return False
        
        return True


# Основной класс для тороидальных геометрий
class TorusGeometries:
    """Основной класс для работы с тороидальными геометриями."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Инициализация тороидальных геометрий.
        
        Args:
            grid_size: Размер сетки
            box_size: Размер коробки в фм
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.manager = TorusGeometryManager()
        
        # Создание координатных сеток
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        y = np.linspace(-box_size/2, box_size/2, grid_size)
        z = np.linspace(-box_size/2, box_size/2, grid_size)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
    
    def create_field_direction(self, config_type: TorusConfiguration,
                             radius: float = 1.0, thickness: float = 0.2,
                             strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Создание направления поля для заданной конфигурации.
        
        Args:
            config_type: Тип конфигурации
            radius: Радиус тора
            thickness: Толщина тора
            strength: Сила поля
            
        Returns:
            Кортеж (n_x, n_y, n_z) компонент направления поля
        """
        config = self.manager.create_configuration(
            config_type, radius, thickness, strength
        )
        
        if not self.manager.validate_configuration(config):
            raise ValueError(f"Invalid configuration: {config_type}")
        
        return config.get_field_direction(self.X, self.Y, self.Z)
    
    def get_configuration_info(self, config_type: TorusConfiguration) -> Dict[str, Any]:
        """
        Получить информацию о конфигурации.
        
        Args:
            config_type: Тип конфигурации
            
        Returns:
            Словарь с информацией о конфигурации
        """
        return self.manager.get_configuration_info(config_type)
    
    def list_available_configurations(self) -> List[Dict[str, Any]]:
        """
        Получить список всех доступных конфигураций с информацией.
        
        Returns:
            Список словарей с информацией о конфигурациях
        """
        configs = []
        for config_type in self.manager.get_available_configurations():
            info = self.get_configuration_info(config_type)
            info['type'] = config_type
            configs.append(info)
        
        return configs
```

## Объяснение

### Структура тороидальных конфигураций

Каждая конфигурация реализована как отдельный класс, наследующий общий интерфейс:
- `Torus120Degrees` - три тора под углом 120°
- `TorusClover` - три тора в форме клевера  
- `TorusCartesian` - три тора вдоль декартовых осей

### Параметры тора

Класс `TorusParameters` определяет параметры отдельного тора:
- `center` - центр тора в пространстве
- `radius` - радиус тора
- `axis` - направление оси тора (нормализованный вектор)
- `thickness` - толщина тора (для весовой функции)
- `strength` - сила поля в торе

### Вычисление направления поля

Метод `get_field_direction()` вычисляет направление поля n̂(x) как взвешенную сумму направлений осей торов. Весовая функция использует гауссово распределение по расстоянию до тора.

### Менеджер геометрий

Класс `TorusGeometryManager` предоставляет единый интерфейс для создания и валидации конфигураций. Он также содержит метаинформацию о каждой конфигурации.

### Валидация

Каждая конфигурация проверяется на корректность:
- Положительные радиус и толщина
- Нормализованные оси торов
- Корректная структура данных

## Следующий шаг

После реализации этого шага мы перейдем к **Шагу 3: SU(2) поля**, где создадим классы для работы с SU(2) полями и операциями над ними.
