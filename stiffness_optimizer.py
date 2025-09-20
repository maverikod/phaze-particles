#!/usr/bin/env python3
"""
Автоматический подбор жёсткости (c2, c4, c6) для устойчивой конфигурации
с целевой метрикой: B→1, E2/E4→0.5, виреал→0, rE/μp в пределах эксперимента

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
import os
import re
import importlib
import itertools
import numpy as np

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def update_constants_in_file(c2, c4, c6):
    """Обновляет константы в файле su2_fields.py"""
    file_path = "phaze_particles/utils/su2_fields.py"
    
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем field_scale на 1.0 (стандартное значение)
    pattern = r'field_scale = [0-9.]+'
    replacement = 'field_scale = 1.0'
    new_content = re.sub(pattern, replacement, content)
    
    # Записываем обратно
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Перезагружаем модули
    modules_to_reload = [
        'phaze_particles.utils.su2_fields',
        'phaze_particles.models.proton_integrated',
        'phaze_particles.utils.energy_densities',
        'phaze_particles.utils.physics'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

def test_constants(c2, c4, c6):
    """Тестирует конкретные значения констант"""
    print(f"Тестируем c2={c2:.2f}, c4={c4:.2f}, c6={c6:.2f} ... ", end="", flush=True)
    
    # Обновляем файл и перезагружаем модули
    update_constants_in_file(c2, c4, c6)
    
    try:
        # Импортируем после перезагрузки
        from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
        
        # Создаем модель
        config = ModelConfig(
            grid_size=32,
            box_size=4.0,
            torus_config='120deg',
            r_scale=1.0,
            c2=c2,
            c4=c4,
            c6=c6
        )

        model = ProtonModel(config)
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()

        # Получаем результаты
        baryon_number = model.physics_calculator.baryon_calculator.compute_baryon_number(model.field_derivatives)
        mass = model.physics_calculator.mass_calculator.compute_mass(model.energy_density.get_total_energy())
        
        # Метрики баланса и жёсткости
        energy_components = model.energy_density.get_energy_components()
        e2 = energy_components.get('e2', 0.0)
        e4 = energy_components.get('e4', 0.0)
        e6 = energy_components.get('e6', 0.0)
        virial_residual = model.energy_density.get_virial_residual()
        e2_ratio = e2 / (e2 + e4 + e6 + 1e-12)
        e4_ratio = e4 / (e2 + e4 + e6 + 1e-12)
        
        # Физические наблюдаемые
        try:
            charge_radius = model.physics_calculator.charge_radius_calculator.compute_charge_radius()
        except:
            charge_radius = 0.0
            
        try:
            magnetic_moment = model.physics_calculator.magnetic_calculator.compute_magnetic_moment(
                model.su2_field, model.profile, mass
            )
        except:
            magnetic_moment = 0.0
        
        # Вычисляем ошибки
        baryon_real = float(baryon_number.real)
        mass_float = float(mass)
        
        # Целевые значения
        target_B = 1.0
        target_rE = 0.841  # фм
        target_μp = 2.793  # μN
        target_mass = 938.272  # МэВ
        
        # Ошибки
        err_B = abs(baryon_real - target_B)
        err_balance = abs(e2_ratio - 0.5) + abs(e4_ratio - 0.5)
        err_virial = abs(float(virial_residual))
        err_rE = abs(charge_radius - target_rE) / target_rE if charge_radius > 0 else 10.0  # Большая ошибка если 0
        err_μp = abs(magnetic_moment - target_μp) / target_μp if magnetic_moment > 0 else 10.0  # Большая ошибка если 0
        err_mass = abs(mass_float - target_mass) / target_mass
        
        # Сводная ошибка с весами
        total_error = (0.3 * err_B + 
                      0.2 * err_balance + 
                      0.1 * err_virial + 
                      0.2 * err_rE + 
                      0.1 * err_μp + 
                      0.1 * err_mass)
        
        print(f"B={baryon_real:.3f}, M={mass_float:.0f}, rE={charge_radius:.3f}, μp={magnetic_moment:.3f}, E2/E4={e2_ratio:.2f}/{e4_ratio:.2f}, V={virial_residual:.3f}, err={total_error:.4f}")
        
        return {
            'c2': c2, 'c4': c4, 'c6': c6,
            'baryon': baryon_real, 'mass': mass_float,
            'charge_radius': charge_radius, 'magnetic_moment': magnetic_moment,
            'e2_ratio': e2_ratio, 'e4_ratio': e4_ratio, 'virial': float(virial_residual),
            'total_error': total_error
        }
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_stiffness():
    """Оптимизирует жёсткость для достижения целевых метрик"""
    
    print("ОПТИМИЗАЦИЯ ЖЁСТКОСТИ (c2, c4, c6)")
    print("=" * 60)
    print("Цели: B→1.0, E2/E4→0.5, виреал→0, rE→0.841фм, μp→2.793μN")
    print()
    
    # Сетка значений для поиска
    c2_values = [0.5, 1.0, 1.5, 2.0]
    c4_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    c6_values = [0.0, 0.1, 0.2, 0.3]
    
    results = []
    total_combinations = len(c2_values) * len(c4_values) * len(c6_values)
    current = 0
    
    print(f"Тестируем {total_combinations} комбинаций...")
    print()
    
    for c2, c4, c6 in itertools.product(c2_values, c4_values, c6_values):
        current += 1
        print(f"[{current:2d}/{total_combinations}] ", end="")
        
        result = test_constants(c2, c4, c6)
        if result:
            results.append(result)
    
    print()
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 60)
    
    if not results:
        print("❌ Не удалось получить результаты!")
        return
    
    # Сортируем по общей ошибке
    results.sort(key=lambda x: x['total_error'])
    
    print("Топ-10 лучших результатов:")
    print("-" * 100)
    print(f"{'c2':>4} {'c4':>4} {'c6':>4} {'B':>6} {'M':>6} {'rE':>6} {'μp':>6} {'E2':>4} {'E4':>4} {'V':>6} {'err':>6}")
    print("-" * 100)
    
    for i, r in enumerate(results[:10]):
        status = "✅" if i == 0 else "  "
        print(f"{status} {r['c2']:4.1f} {r['c4']:4.1f} {r['c6']:4.1f} "
              f"{r['baryon']:6.3f} {r['mass']:6.0f} {r['charge_radius']:6.3f} "
              f"{r['magnetic_moment']:6.3f} {r['e2_ratio']:4.2f} {r['e4_ratio']:4.2f} "
              f"{r['virial']:6.3f} {r['total_error']:6.4f}")
    
    # Лучший результат
    best = results[0]
    
    print()
    print("ЛУЧШИЙ РЕЗУЛЬТАТ:")
    print("-" * 30)
    print(f"c2 = {best['c2']:.2f}")
    print(f"c4 = {best['c4']:.2f}")
    print(f"c6 = {best['c6']:.2f}")
    print(f"Барионное число = {best['baryon']:.3f} (цель: 1.000)")
    print(f"Масса = {best['mass']:.0f} МэВ (цель: 938)")
    print(f"Зарядовый радиус = {best['charge_radius']:.3f} фм (цель: 0.841)")
    print(f"Магнитный момент = {best['magnetic_moment']:.3f} μN (цель: 2.793)")
    print(f"Энергетический баланс E2/E4 = {best['e2_ratio']:.2f}/{best['e4_ratio']:.2f} (цель: 0.5/0.5)")
    print(f"Виреальный остаток = {best['virial']:.3f} (цель: <0.05)")
    print(f"Общая ошибка = {best['total_error']:.4f}")
    
    # Проверяем качество
    if best['total_error'] < 0.1:
        quality = "ОТЛИЧНО"
    elif best['total_error'] < 0.2:
        quality = "ХОРОШО"
    elif best['total_error'] < 0.5:
        quality = "УДОВЛЕТВОРИТЕЛЬНО"
    else:
        quality = "ПЛОХО"
    
    print(f"Качество: {quality}")
    
    # Рекомендации
    print()
    print("РЕКОМЕНДАЦИИ:")
    print("-" * 20)
    print(f"1. Установите c2 = {best['c2']:.2f}")
    print(f"2. Установите c4 = {best['c4']:.2f}")
    print(f"3. Установите c6 = {best['c6']:.2f}")
    print(f"4. Ожидаемое барионное число: {best['baryon']:.3f}")
    print(f"5. Ожидаемая масса: {best['mass']:.0f} МэВ")
    
    if best['total_error'] > 0.2:
        print("6. ⚠️  Ошибка все еще велика, возможно нужна более тонкая настройка")
    
    return best

if __name__ == "__main__":
    try:
        optimize_stiffness()
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
    except Exception as e:
        print(f"\n\nОшибка при оптимизации: {e}")
        import traceback
        traceback.print_exc()
