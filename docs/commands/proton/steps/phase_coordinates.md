# Фазовые координаты для модели протона

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

## 1. Введение в фазовые координаты

### 1.1 Концепция фазовых координат

Фазовые координаты - это редуцированная система переменных, которая позволяет:
- Жестко зашить топологию (B=1) как ограничение на фазу
- Естественно описывать многотороидальные конфигурации
- Упрощать расчет барионного заряда и электрического заряда
- Отделить геометрию от динамики фаз

### 1.2 Преимущества фазового подхода

1. **Топологическая стабильность:** B=1 зашито в фазовую структуру
2. **Вычислительная эффективность:** меньше переменных для оптимизации
3. **Физическая интерпретация:** фазы соответствуют угловым модам
4. **Симметрийный анализ:** легче анализировать симметрии конфигураций

## 2. Определение фазовых переменных

### 2.1 Геометрические фазы

**Для трехтороидальной конфигурации:**

**Азимутальные углы:**
$$\phi_a = \arctan\left(\frac{y - y_a}{x - x_a}\right)$$

**Полярные углы:**
$$\theta_a = \arccos\left(\frac{z - z_a}{r_a}\right)$$

где:
- $a = 1,2,3$ - индекс тора
- $(x_a, y_a, z_a)$ - центр тора
- $r_a$ - радиус тора

### 2.2 Динамические фазы

**Фазовые сдвиги:**
$$\alpha_a = \phi_a + \chi_a$$
$$\beta_a = \theta_a + \eta_a$$

где $\chi_a, \eta_a$ - динамические фазовые параметры.

### 2.3 Фазовые ограничения

**Топологическое условие:**
$$\sum_{a=1}^3 \alpha_a = 2\pi n, \quad n \in \mathbb{Z}$$

**Симметрийные условия:**
- **120° конфигурация:** $\alpha_2 = \alpha_1 + 2\pi/3$, $\alpha_3 = \alpha_1 + 4\pi/3$
- **Клевер конфигурация:** $\alpha_2 = \alpha_1 + \pi/2$, $\alpha_3 = \alpha_1 + \pi$, $\alpha_4 = \alpha_1 + 3\pi/2$
- **Декартовая конфигурация:** $\alpha_1 = 0$, $\alpha_2 = \pi/2$, $\alpha_3 = \pi$, $\alpha_4 = 3\pi/2$

## 3. Параметризация SU(2) поля

### 3.1 Общая формула

$$U(\mathbf{x}) = \exp\left(i \sum_{a=1}^3 F_a(r_a) \hat{\mathbf{n}}_a(\alpha_a, \beta_a) \cdot \boldsymbol{\sigma}\right)$$

где:
$$\hat{\mathbf{n}}_a(\alpha_a, \beta_a) = \begin{pmatrix}
\sin\beta_a \cos\alpha_a \\
\sin\beta_a \sin\alpha_a \\
\cos\beta_a
\end{pmatrix}$$

### 3.1.1 Связь с основным SU(2) полем

**Связь с радиальным профилем:**
$$F_a(r_a) = f(r) \cdot w_a(r_a)$$

где:
- $f(r)$ - основной радиальный профиль из SU(2) поля
- $w_a(r_a)$ - весовая функция для тора $a$
- $r_a$ - расстояние до тора $a$

**Связь с направлением поля:**
$$\hat{\mathbf{n}}(\mathbf{x}) = \frac{\sum_{a=1}^3 w_a(r_a) \hat{\mathbf{n}}_a(\alpha_a, \beta_a)}{\left|\sum_{a=1}^3 w_a(r_a) \hat{\mathbf{n}}_a(\alpha_a, \beta_a)\right|}$$

**Основное SU(2) поле:**
$$U(\mathbf{x}) = \cos f(r) \mathbf{1} + i \sin f(r) \hat{\mathbf{n}}(\mathbf{x}) \cdot \boldsymbol{\sigma}$$

### 3.2 Радиальные профили

**Профиль для тора $a$:**
$$F_a(r_a) = \pi \exp\left(-\frac{r_a^2}{2w_a^2}\right) \left(1 - \frac{r_a^2}{R_a^2}\right)$$

где:
- $w_a$ - ширина профиля
- $R_a$ - радиус тора

### 3.3 Фазовые моды

**Основные моды:**
$$F_a(r_a) = \sum_{n=0}^{N} c_{an} \sin\left(\frac{n\pi r_a}{R_a}\right)$$

**Ортогональные моды:**
$$\int_0^{R_a} F_a(r_a) F_b(r_a) r_a^2 dr_a = \delta_{ab}$$

## 4. Фазовые уравнения движения

### 4.1 Уравнения для фазовых углов

**Азимутальные фазы:**
$$\frac{\partial \alpha_a}{\partial t} = -\frac{\delta E}{\delta \alpha_a}$$

**Полярные фазы:**
$$\frac{\partial \beta_a}{\partial t} = -\frac{\delta E}{\delta \beta_a}$$

### 4.2 Уравнения для радиальных профилей

**Профили:**
$$\frac{\partial F_a}{\partial t} = -\frac{\delta E}{\delta F_a}$$

### 4.3 Фазовые ограничения

**Топологическое ограничение:**
$$\frac{d}{dt}\left(\sum_{a=1}^3 \alpha_a\right) = 0$$

**Симметрийные ограничения:**
$$\frac{d}{dt}\left(\alpha_{a+1} - \alpha_a - \frac{2\pi}{3}\right) = 0$$

## 5. Численные методы в фазовых координатах

### 5.1 Фазовый градиентный спуск

```python
def phase_gradient_descent(initial_phases, max_iterations=1000):
    """
    Градиентный спуск в фазовых координатах.
    
    Args:
        initial_phases: Начальные фазовые углы
        max_iterations: Максимальное число итераций
        
    Returns:
        Оптимизированные фазовые углы
    """
    phases = initial_phases.copy()
    
    for iteration in range(max_iterations):
        # Вычисление градиента
        gradient = compute_phase_gradient(phases)
        
        # Обновление фаз
        phases -= learning_rate * gradient
        
        # Применение фазовых ограничений
        phases = apply_phase_constraints(phases)
        
        # Проверка сходимости
        if np.linalg.norm(gradient) < tolerance:
            break
    
    return phases
```

### 5.2 Проекция на фазовые ограничения

```python
def apply_phase_constraints(phases):
    """
    Применение фазовых ограничений.
    
    Args:
        phases: Фазовые углы
        
    Returns:
        Ограниченные фазовые углы
    """
    # Топологическое ограничение
    total_phase = np.sum(phases)
    phases -= total_phase / len(phases)
    
    # Симметрийные ограничения (для 120° конфигурации)
    if len(phases) == 3:
        phases[1] = phases[0] + 2*np.pi/3
        phases[2] = phases[0] + 4*np.pi/3
    
    return phases
```

### 5.3 Фазовый анализ

```python
def analyze_phase_structure(phases):
    """
    Анализ фазовой структуры.
    
    Args:
        phases: Фазовые углы
        
    Returns:
        Словарь с анализом
    """
    analysis = {
        'total_phase': np.sum(phases),
        'phase_differences': np.diff(phases),
        'symmetry_type': classify_symmetry(phases),
        'topological_charge': compute_topological_charge(phases)
    }
    
    return analysis
```

## 6. Сравнение конфигураций

### 6.1 120° конфигурация

**Фазовые углы:**
$$\alpha_1 = 0, \quad \alpha_2 = \frac{2\pi}{3}, \quad \alpha_3 = \frac{4\pi}{3}$$

**Симметрия:** $C_3$ (поворот на 120°)

**Преимущества:**
- Максимальная симметрия
- Минимальная энергия
- Стабильная топология

### 6.2 Клевер конфигурация

**Фазовые углы:**
$$\alpha_1 = 0, \quad \alpha_2 = \frac{\pi}{2}, \quad \alpha_3 = \pi, \quad \alpha_4 = \frac{3\pi}{2}$$

**Симметрия:** $C_4$ (поворот на 90°)

**Преимущества:**
- Четырехкратная симметрия
- Хорошая стабильность
- Простая геометрия

### 6.3 Декартовая конфигурация

**Фазовые углы:**
$$\alpha_1 = 0, \quad \alpha_2 = \frac{\pi}{2}, \quad \alpha_3 = \pi, \quad \alpha_4 = \frac{3\pi}{2}$$

**Симметрия:** $D_4$ (диэдральная группа)

**Преимущества:**
- Высокая симметрия
- Простая реализация
- Хорошая сходимость

## 7. Оптимизация в фазовых координатах

### 7.1 Целевая функция

$$E_{\text{phase}} = E_{\text{SU(2)}} + E_{\text{constraints}} + E_{\text{symmetry}}$$

где:
- $E_{\text{SU(2)}}$ - энергия SU(2) поля
- $E_{\text{constraints}}$ - ограничения на фазы
- $E_{\text{symmetry}}$ - симметрийные ограничения

### 7.2 Градиенты

**По фазовым углам:**
$$\frac{\partial E_{\text{phase}}}{\partial \alpha_a} = \frac{\partial E_{\text{SU(2)}}}{\partial \alpha_a} + \lambda_{\text{topo}} \frac{\partial B}{\partial \alpha_a}$$

**По радиальным профилям:**
$$\frac{\partial E_{\text{phase}}}{\partial F_a} = \frac{\partial E_{\text{SU(2)}}}{\partial F_a} + \lambda_{\text{profile}} \frac{\partial \text{constraints}}{\partial F_a}$$

### 7.3 Алгоритм оптимизации

```python
def optimize_phase_configuration(initial_config):
    """
    Оптимизация фазовой конфигурации.
    
    Args:
        initial_config: Начальная конфигурация
        
    Returns:
        Оптимизированная конфигурация
    """
    config = initial_config.copy()
    
    for iteration in range(max_iterations):
        # Вычисление энергии
        energy = compute_phase_energy(config)
        
        # Вычисление градиентов
        gradients = compute_phase_gradients(config)
        
        # Обновление конфигурации
        config = update_phase_configuration(config, gradients)
        
        # Проверка сходимости
        if energy_change < tolerance:
            break
    
    return config
```

## 8. Валидация фазовых результатов

### 8.1 Топологические проверки

**Барионное число:**
$$B = \frac{1}{24\pi^2} \sum_{a=1}^3 \int \epsilon^{ijk} \text{Tr}(\tilde{L}_i^a \tilde{L}_j^a \tilde{L}_k^a) d^3\tilde{x} = 1$$

**Электрический заряд:**
$$Q = \int \tilde{\rho}(\mathbf{x}) d^3\tilde{x} = +1$$

### 8.2 Симметрийные проверки

**Симметрия 120°:**
$$\alpha_2 - \alpha_1 = \frac{2\pi}{3}, \quad \alpha_3 - \alpha_2 = \frac{2\pi}{3}$$

**Симметрия клевер:**
$$\alpha_2 - \alpha_1 = \frac{\pi}{2}, \quad \alpha_3 - \alpha_2 = \frac{\pi}{2}$$

### 8.3 Энергетические проверки

**Виреальное условие:**
$$\frac{E_{(2)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

**Энергетический баланс:**
$$\frac{E_{(4)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

## 9. Практические рекомендации

### 9.1 Выбор фазовых координат

1. **Для 120° конфигурации:** использовать 3 фазовых угла
2. **Для клевер конфигурации:** использовать 4 фазовых угла
3. **Для декартовой конфигурации:** использовать 4 фазовых угла

### 9.2 Оптимизация

1. **Сначала:** оптимизировать в фазовых координатах
2. **Затем:** перейти к полным координатам
3. **Проверить:** согласованность результатов

### 9.3 Валидация

1. **Топология:** проверить B=1
2. **Симметрия:** проверить фазовые соотношения
3. **Энергия:** проверить виреальное условие

## 10. Заключение

Фазовые координаты предоставляют мощный инструмент для:
- Анализа топологических свойств
- Оптимизации симметрийных конфигураций
- Упрощения численных расчетов
- Физической интерпретации результатов

Использование фазовых координат в сочетании с безразмерными уравнениями обеспечивает эффективную и стабильную реализацию модели протона.
