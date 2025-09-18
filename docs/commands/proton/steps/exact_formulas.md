# Точные формулы для модели протона

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

## 0. Обезразмеривание уравнений

### 0.1 Безразмерные переменные

**Пространственные координаты:**
$$\tilde{x}_i = \frac{x_i}{L_0}, \quad \tilde{r} = \frac{r}{L_0}$$

где $L_0$ - характерная длина (например, $L_0 = 1$ фм).

**Энергетические переменные:**
$$\tilde{E} = \frac{E}{E_0}, \quad \tilde{c}_2 = \frac{c_2}{E_0 L_0}, \quad \tilde{c}_4 = \frac{c_4}{E_0 L_0^3}, \quad \tilde{c}_6 = \frac{c_6}{E_0 L_0^5}$$

где $E_0$ - характерная энергия (например, $E_0 = 1$ МэВ).

**Электромагнитные переменные:**
$$\tilde{A}_\mu = \frac{A_\mu}{A_0}, \quad \tilde{e} = \frac{e}{\sqrt{E_0 L_0}}$$

где $A_0 = \sqrt{E_0/L_0}$ - характерный потенциал.

### 0.2 Безразмерные параметры

**Основные константы:**
$$\alpha = \frac{e^2}{4\pi} = \frac{1}{137.036}, \quad \tilde{\alpha} = \frac{\alpha E_0 L_0}{E_0} = \alpha L_0$$

**Характерные масштабы:**
$$\tilde{M}_p = \frac{M_p}{E_0}, \quad \tilde{r}_E = \frac{r_E}{L_0}$$

### 0.3 Безразмерный функционал энергии

$$\tilde{E}[U,\Theta] = \int d^3\tilde{x} \left[ \tilde{c}_2 \text{Tr}(\tilde{L}_i \tilde{L}_i) + \tilde{c}_4 \text{Tr}([\tilde{L}_i,\tilde{L}_j]^2) + \tilde{c}_6 \tilde{b}_0^2 + \frac{1}{4} \tilde{F}_{\mu\nu} \tilde{F}^{\mu\nu} + \tilde{j}_\mu \tilde{A}^\mu \right]$$

где:
$$\tilde{L}_i = U^\dagger \partial_{\tilde{x}_i} U, \quad \tilde{b}_0 = \frac{1}{24\pi^2} \epsilon^{ijk} \text{Tr}(\tilde{L}_i \tilde{L}_j \tilde{L}_k)$$

## 1. Полный функционал энергии

### 1.1 Основной функционал

$$\tilde{E}[U,\Theta] = E_{\text{SU(2)}}[U] + E_{\text{EM}}[U,\Theta] + E_{\text{constraints}}[U]$$

где:

$$E_{\text{SU(2)}}[U] = \int d^3\tilde{x} \left[ \tilde{c}_2 \text{Tr}(\tilde{L}_i \tilde{L}_i) + \tilde{c}_4 \text{Tr}([\tilde{L}_i,\tilde{L}_j]^2) + \tilde{c}_6 \tilde{b}_0^2 \right]$$

$$E_{\text{EM}}[U,\Theta] = \int d^3\tilde{x} \left[ \frac{1}{4} \tilde{F}_{\mu\nu} \tilde{F}^{\mu\nu} + \tilde{j}_\mu \tilde{A}^\mu \right]$$

$$E_{\text{constraints}}[U] = \tilde{\lambda}_B (B - 1)^2 + \tilde{\lambda}_Q (Q - 1)^2 + \tilde{\lambda}_{\text{virial}} \left( \frac{E_{(2)}}{E_{\text{tot}}} - 0.5 \right)^2$$

### 1.2 Компоненты SU(2) энергии

**Левые токи:**
$$\tilde{L}_i = U^\dagger \partial_{\tilde{x}_i} U$$

**c₂ член (кинетическая энергия):**
$$\text{Tr}(\tilde{L}_i \tilde{L}_i) = \text{Tr}(U^\dagger \partial_{\tilde{x}_i} U \cdot U^\dagger \partial_{\tilde{x}_i} U)$$

**c₄ член (энергия взаимодействия):**
$$\text{Tr}([\tilde{L}_i,\tilde{L}_j]^2) = \text{Tr}([\tilde{L}_i,\tilde{L}_j] [\tilde{L}_i,\tilde{L}_j])$$

**c₆ член (стабилизирующий):**
$$\tilde{b}_0^2 = \left( \frac{1}{24\pi^2} \epsilon^{ijk} \text{Tr}(\tilde{L}_i \tilde{L}_j \tilde{L}_k) \right)^2$$

### 1.3 Электромагнитная энергия

**Полевой тензор:**
$$\tilde{F}_{\mu\nu} = \partial_{\tilde{x}_\mu} \tilde{A}_\nu - \partial_{\tilde{x}_\nu} \tilde{A}_\mu$$

**Электромагнитный ток:**
$$\tilde{j}_\mu = \frac{\delta E_{\text{SU(2)}}}{\delta \tilde{A}^\mu}$$

## 2. Вариационные уравнения (безразмерные)

### 2.1 Уравнения Эйлера-Лагранжа

**Для SU(2) поля:**
$$\frac{\delta \tilde{E}}{\delta U} = 0$$

$$\Rightarrow \partial_{\tilde{x}_i} \left( \frac{\partial \tilde{\mathcal{L}}}{\partial (\partial_{\tilde{x}_i} U)} \right) - \frac{\partial \tilde{\mathcal{L}}}{\partial U} = 0$$

**Для электромагнитного поля:**
$$\frac{\delta \tilde{E}}{\delta \tilde{A}_\mu} = 0$$

$$\Rightarrow \partial_{\tilde{x}_\nu} \tilde{F}^{\mu\nu} = \tilde{j}^\mu$$

### 2.2 Полные вариационные уравнения (безразмерные)

**SU(2) уравнение:**
$$\tilde{c}_2 \partial_{\tilde{x}_i} (U^\dagger \partial_{\tilde{x}_i} U) + \tilde{c}_4 \partial_{\tilde{x}_i} [\tilde{L}_i, [\tilde{L}_i, \tilde{L}_j]] + \tilde{c}_6 \frac{\partial \tilde{b}_0^2}{\partial U} + \frac{\partial \tilde{E}_{\text{constraints}}}{\partial U} = 0$$

**Электромагнитное уравнение:**
$$\partial_{\tilde{x}_\nu} \tilde{F}^{\mu\nu} = \tilde{j}^\mu$$

где:
$$\tilde{j}^\mu = \frac{\partial \tilde{\mathcal{L}}_{\text{SU(2)}}}{\partial \tilde{A}_\mu}$$

### 2.3 Безразмерные множители Лагранжа

**Для барионного числа:**
$$\tilde{\lambda}_B = \frac{\partial \tilde{E}_{\text{constraints}}}{\partial B} = 2(B - 1)$$

**Для электрического заряда:**
$$\tilde{\lambda}_Q = \frac{\partial \tilde{E}_{\text{constraints}}}{\partial Q} = 2(Q - 1)$$

**Для виреального условия:**
$$\tilde{\lambda}_{\text{virial}} = \frac{\partial \tilde{E}_{\text{constraints}}}{\partial (E_{(2)}/E_{\text{tot}})} = 2\left( \frac{E_{(2)}}{E_{\text{tot}}} - 0.5 \right)$$


## 3. Электромагнитные токи (безразмерные)

### 3.1 Минимальная связь

**Ковариантная производная:**
$$\tilde{D}_\mu U = \partial_{\tilde{x}_\mu} U + i \tilde{e} \tilde{A}_\mu [Q, U]$$

где $Q = \frac{1}{2} \sigma_3$ - генератор электрического заряда.

### 3.2 Электромагнитный ток (безразмерный)

**Векторный ток:**
$$\tilde{j}_i = \frac{\partial \tilde{\mathcal{L}}_{\text{SU(2)}}}{\partial \tilde{A}_i} = i \tilde{e} \text{Tr}(Q [\tilde{L}_i, U^\dagger \partial_{\tilde{x}_i} U])$$

**Скалярный ток (плотность заряда):**
$$\tilde{j}_0 = \tilde{\rho} = \frac{\partial \tilde{\mathcal{L}}_{\text{SU(2)}}}{\partial \tilde{A}_0} = i \tilde{e} \text{Tr}(Q [\tilde{L}_0, U^\dagger \partial_{\tilde{x}_0} U])$$

### 3.3 Полный электромагнитный ток (безразмерный)

$$\tilde{j}_\mu = i \tilde{e} \text{Tr}(Q [\tilde{L}_\mu, U^\dagger \partial_{\tilde{x}_\mu} U])$$

где $\tilde{L}_\mu = U^\dagger \partial_{\tilde{x}_\mu} U$.

### 3.4 Связь с размерными величинами

**Размерный ток:**
$$j_\mu = \frac{E_0^{3/2}}{L_0^{5/2}} \tilde{j}_\mu$$

**Размерная плотность заряда:**
$$\rho = \frac{E_0^{3/2}}{L_0^{5/2}} \tilde{\rho}$$

## 4. Форм-факторы

### 4.1 Электрический форм-фактор

$$G_E(Q^2) = \int d^3x \rho(\mathbf{x}) e^{i \mathbf{q} \cdot \mathbf{x}}$$

где $\mathbf{q}$ - переданный импульс, $Q^2 = \mathbf{q}^2$.

### 4.2 Магнитный форм-фактор

$$G_M(Q^2) = \frac{1}{2M_p} \int d^3x \mathbf{r} \times \mathbf{j}(\mathbf{x}) \cdot \hat{\mathbf{q}} e^{i \mathbf{q} \cdot \mathbf{x}}$$

### 4.3 Связь с экспериментальными данными

**Электрический заряд:**
$$G_E(0) = Q = +1$$

**Магнитный момент:**
$$\mu_p = G_M(0) = 2.793 \mu_N$$

## 5. Магнитный момент

### 5.1 Формула для магнитного момента

$$\mu_p = \frac{e}{2M_p} \langle p, \uparrow | \int \mathbf{r} \times \mathbf{j}(\mathbf{x}) d^3x | p, \uparrow \rangle$$

### 5.2 Вычисление через ток

$$\mu_p = \frac{e}{2M_p} \int d^3x \mathbf{r} \times \mathbf{j}(\mathbf{x})$$

где:
$$\mathbf{j}(\mathbf{x}) = i e \text{Tr}(Q [L_i, U^\dagger \partial_i U]) \hat{\mathbf{e}}_i$$

### 5.3 Процедура вычисления магнитного момента

**Шаг 1: Вычисление электромагнитного тока**
```python
# На сетке (i,j,k)
j_x[i,j,k] = i * e * Tr(Q * [L_x, U_dagger * grad_x_U])
j_y[i,j,k] = i * e * Tr(Q * [L_y, U_dagger * grad_y_U])  
j_z[i,j,k] = i * e * Tr(Q * [L_z, U_dagger * grad_z_U])
```

**Шаг 2: Вычисление магнитного момента**
```python
# Интеграл r × j
mu_p = (e / (2 * M_p)) * sum(r[i,j,k] × j[i,j,k] * dx³)
```

**Шаг 3: Нормировка**
```python
# Переход к безразмерным единицам
mu_p_dimensionless = mu_p / (e * hbar / (2 * m_p))
```

### 5.4 Критерии приемки для магнитного момента

- [ ] **Относительная ошибка:** |μp - 2.793|/2.793 ≤ 5%
- [ ] **Сходимость по сетке:** изменение μp при утроении узлов ≤ 2%
- [ ] **Проверка размерности:** μp имеет правильные единицы μN

### 5.5 Связь с барионным током

$$\mu_p = \frac{e}{2M_p} \int d^3x \mathbf{r} \times \mathbf{j}_{\text{baryon}}(\mathbf{x})$$

где $\mathbf{j}_{\text{baryon}}$ - барионный ток, связанный с топологическим зарядом.

## 6. Граничные условия

### 6.1 Для SU(2) поля

**В центре:**
$$U(0) = -i \sigma_3, \quad f(0) = \pi$$

**На бесконечности:**
$$U(\infty) = \mathbf{1}, \quad f(\infty) = 0$$

### 6.2 Для электромагнитного поля

**В центре:**
$$A_0(0) = 0, \quad \mathbf{A}(0) = 0$$

**На бесконечности:**
$$A_0(\infty) = \frac{Q}{4\pi r}, \quad \mathbf{A}(\infty) = 0$$

### 6.3 Для фазового поля

**В центре:**
$$\Theta(0) = \Theta_0$$

**На бесконечности:**
$$\Theta(\infty) = \Theta_\infty$$

## 7. Численные схемы

### 7.1 Градиентный спуск

$$\frac{\partial U}{\partial t} = -\frac{\delta \tilde{E}}{\delta U}$$

$$\frac{\partial A_\mu}{\partial t} = -\frac{\delta \tilde{E}}{\delta A_\mu}$$

### 7.2 Проекция на SU(2)

После каждого шага:
$$U \rightarrow \frac{U}{\sqrt{\det(U)}}$$

### 7.3 Контроль ограничений

**Барионное число:**
$$B = -\frac{1}{24\pi^2} \int \epsilon^{ijk} \text{Tr}(L_i L_j L_k) d^3x = 1$$

**Электрический заряд:**
$$Q = \int \rho(\mathbf{x}) d^3x = +1$$

**Виреальное условие (со стабилизирующим c₆):**
$$E_{(2)} = E_{(4)} + 3 E_{(6)} \quad (\text{при } c_6 = 0 \Rightarrow E_{(2)}:E_{(4)} = 50{:}50)$$

### 7.4 Механизм наведения виреального баланса

**Алгоритм наведения виреального равенства E₂ = E₄ + 3E₆:**

**Шаг 1: Вычисление текущего баланса**
```python
E_2 = compute_energy_c2_term()
E_4 = compute_energy_c4_term()
E_6 = compute_energy_c6_term()
E_tot = E_2 + E_4 + E_6
virial_residual = (E_2 - (E_4 + 3 * E_6)) / E_tot
```

**Шаг 2: Коррекция через множитель Лагранжа**
```python
if abs(virial_residual) > 0.01:
    lambda_virial = 2 * virial_residual
    # Добавить штраф к энергии
    E_total += lambda_virial * virial_residual**2
```

**Шаг 3: Рескейлинговая коррекция**
```python
if virial_residual > 0.01:  # E2 слишком велик относительно E4+3E6
    c2 *= 0.99
    c4 *= 1.005
    c6 *= 1.005
elif virial_residual < -0.01:  # E2 слишком мал
    c2 *= 1.005
    c4 *= 0.995
    c6 *= 0.995
```

**Шаг 4: Критерий сходимости**
```python
converged = abs(virial_residual) <= 0.01
```

## 8. Проверочные формулы

### 8.1 Энергетический баланс

$$E_{\text{tot}} = E_{(2)} + E_{(4)} + E_{(6)}$$

$$\frac{E_{(2)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

$$\frac{E_{(4)}}{E_{\text{tot}}} = 0.5 \pm 0.01$$

### 8.2 Физические параметры

**Масса протона:**
$$M_p = E_{\text{tot}} = 938.272 \pm 0.006 \text{ МэВ}$$

**Радиус зарядового распределения:**
$$r_E = \sqrt{\frac{\int r^2 \rho(\mathbf{x}) d^3x}{\int \rho(\mathbf{x}) d^3x}} = 0.841 \pm 0.019 \text{ фм}$$

**Магнитный момент:**
$$\mu_p = 2.793 \pm 0.001 \mu_N$$

### 8.3 Топологические инварианты

**Барионное число:**
$$B = -\frac{1}{24\pi^2} \int \epsilon^{ijk} \text{Tr}(L_i L_j L_k) d^3x = 1$$

**Электрический заряд:**
$$Q = \int \rho(\mathbf{x}) d^3x = +1$$

**Изоспин:**
$$I_3 = \frac{1}{2} \text{Tr}(\sigma_3 U(0)) = +\frac{1}{2}$$

## 9. Фазовые координаты

### 9.1 Определение фазовых переменных

**Для трехтороидальной конфигурации:**
$$\phi_a = \arctan\left(\frac{y - y_a}{x - x_a}\right), \quad \theta_a = \arccos\left(\frac{z - z_a}{r_a}\right)$$

где $a = 1,2,3$ - индекс тора, $(x_a, y_a, z_a)$ - центр тора, $r_a$ - радиус тора.

**Фазовые углы:**
$$\alpha_a = \phi_a + \chi_a, \quad \beta_a = \theta_a + \eta_a$$

где $\chi_a, \eta_a$ - фазовые сдвиги.

### 9.2 Параметризация SU(2) поля в фазовых координатах

$$U(\mathbf{x}) = \exp\left(i \sum_{a=1}^3 F_a(r_a) \hat{\mathbf{n}}_a(\alpha_a, \beta_a) \cdot \boldsymbol{\sigma}\right)$$

где:
$$\hat{\mathbf{n}}_a(\alpha_a, \beta_a) = \begin{pmatrix}
\sin\beta_a \cos\alpha_a \\
\sin\beta_a \sin\alpha_a \\
\cos\beta_a
\end{pmatrix}$$

### 9.3 Фазовые ограничения

**Топологическое ограничение:**
$$B = \frac{1}{24\pi^2} \sum_{a=1}^3 \int \epsilon^{ijk} \text{Tr}(\tilde{L}_i^a \tilde{L}_j^a \tilde{L}_k^a) d^3\tilde{x} = 1$$

**Фазовое условие:**
$$\sum_{a=1}^3 \alpha_a = 2\pi n, \quad n \in \mathbb{Z}$$

## 10. Перевод в физические единицы (СИ)

### 10.1 Конверсионные факторы

**Длина:**
$$L_0 = 1 \text{ фм} = 10^{-15} \text{ м}$$

**Энергия:**
$$E_0 = 1 \text{ МэВ} = 1.602 \times 10^{-13} \text{ Дж}$$

**Время:**
$$T_0 = \frac{L_0}{c} = \frac{10^{-15}}{3 \times 10^8} = 3.33 \times 10^{-24} \text{ с}$$

### 10.2 Физические величины

**Масса протона:**
$$M_p = \tilde{M}_p \cdot E_0 = \tilde{M}_p \cdot 1.602 \times 10^{-13} \text{ Дж}$$

**Радиус зарядового распределения:**
$$r_E = \tilde{r}_E \cdot L_0 = \tilde{r}_E \cdot 10^{-15} \text{ м}$$

**Магнитный момент:**
$$\mu_p = \tilde{\mu}_p \cdot \frac{e\hbar}{2m_p} = \tilde{\mu}_p \cdot 1.41 \times 10^{-26} \text{ Дж/Тл}$$

### 10.3 Калибровка по экспериментальным данным

**Калибровочные уравнения:**
$$\tilde{M}_p = \frac{938.272 \text{ МэВ}}{E_0} = 938.272$$

$$\tilde{r}_E = \frac{0.841 \text{ фм}}{L_0} = 0.841$$

**Определение коэффициентов:**
$$\tilde{c}_2 = \frac{c_2}{E_0 L_0}, \quad \tilde{c}_4 = \frac{c_4}{E_0 L_0^3}, \quad \tilde{c}_6 = \frac{c_6}{E_0 L_0^5}$$

### 10.4 Проверка физических параметров

**Барионное число:**
$$B = 1.000 \pm 0.001$$

**Электрический заряд:**
$$Q = +1.000 \pm 0.001$$

**Масса:**
$$M_p = 938.272 \pm 0.006 \text{ МэВ}$$

**Радиус:**
$$r_E = 0.841 \pm 0.019 \text{ фм}$$

**Магнитный момент:**
$$\mu_p = 2.793 \pm 0.001 \mu_N$$
