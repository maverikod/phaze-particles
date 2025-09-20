#!/usr/bin/env python3
"""
Финальный скрипт для автоматического подбора оптимального field_scale
с перезагрузкой модулей

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
import os
import re
import importlib

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def update_field_scale_in_file(field_scale):
    """Обновляет field_scale в файле su2_fields.py"""
    file_path = "phaze_particles/utils/su2_fields.py"
    
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем field_scale
    pattern = r'field_scale = [0-9.]+'
    replacement = f'field_scale = {field_scale}'
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

def test_field_scale(field_scale):
    """Тестирует конкретное значение field_scale"""
    print(f"Тестируем field_scale = {field_scale:.6f} ... ", end="", flush=True)
    
    # Обновляем файл и перезагружаем модули
    update_field_scale_in_file(field_scale)
    
    try:
        # Импортируем после перезагрузки
        from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
        
        # Создаем модель
        config = ModelConfig(
            grid_size=32,
            box_size=4.0,
            torus_config='120deg',
            r_scale=1.0,
            c2=1.0,
            c4=1.0,
            c6=0.0
        )

        model = ProtonModel(config)
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()

        # Получаем результаты
        baryon_number = model.physics_calculator.baryon_calculator.compute_baryon_number(model.field_derivatives)
        mass = model.physics_calculator.mass_calculator.compute_mass(model.energy_density.get_total_energy())
        # Метрики баланса
        e2, e4, e6 = model.energy_density.get_energy_components()
        virial_residual = model.energy_density.get_virial_residual()
        e2_ratio = e2 / (e2 + e4 + e6 + 1e-12)
        e4_ratio = e4 / (e2 + e4 + e6 + 1e-12)
        # Сводная ошибка: цель B≈1, E2≈0.5, E4≈0.5, виреал≈0
        baryon_real = float(baryon_number.real)
        mass_float = float(mass)
        err_B = abs(baryon_real - 1.0)
        err_bal = abs(e2_ratio - 0.5) + abs(e4_ratio - 0.5)
        err_vir = abs(float(virial_residual))
        # Веса можно менять при необходимости
        total_error = 0.6 * err_B + 0.3 * err_bal + 0.1 * err_vir
        
        print(f"B = {baryon_real:.6f}, M = {mass_float:.1f} МэВ, E2={e2:.3f}, E4={e4:.3f}, r2={e2_ratio:.3f}, r4={e4_ratio:.3f}, V={virial_residual:.3f}, итог.ошибка = {total_error:.6f}")
        return field_scale, baryon_real, mass_float, total_error
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None

def main():
    """Основная функция"""
    print("ОПТИМИЗАЦИЯ FIELD_SCALE С ПЕРЕЗАГРУЗКОЙ МОДУЛЕЙ")
    print("=" * 60)
    
    # Тестируем несколько значений
    test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    results = []
    
    for scale in test_values:
        result = test_field_scale(scale)
        if result:
            results.append(result)
    
    print()
    print("РЕЗУЛЬТАТЫ:")
    print("-" * 60)
    print(f"{'field_scale':>12} {'B':>10} {'Масса':>8} {'Ошибка':>10}")
    print("-" * 60)
    
    # Сортируем по ошибке
    results.sort(key=lambda x: x[3])
    
    for i, (scale, baryon, mass, error) in enumerate(results):
        status = "✅" if i == 0 else "  "
        print(f"{status} {scale:10.4f} {baryon:10.5f} {mass:8.1f} {error:10.5f}")
    
    if results:
        best_scale, best_baryon, best_mass, best_error = results[0]
        print()
        print("ЛУЧШИЙ РЕЗУЛЬТАТ:")
        print("-" * 30)
        print(f"field_scale = {best_scale:.6f}")
        print(f"Барионное число = {best_baryon:.6f}")
        print(f"Масса = {best_mass:.1f} МэВ")
        print(f"Ошибка = {best_error:.6f}")
        
        # Проверяем качество
        if best_error < 0.01:
            quality = "ОТЛИЧНО"
        elif best_error < 0.05:
            quality = "ХОРОШО"
        elif best_error < 0.1:
            quality = "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            quality = "ПЛОХО"
        
        print(f"Качество: {quality}")
        
        # Рекомендации
        print()
        print("РЕКОМЕНДАЦИИ:")
        print("-" * 20)
        print(f"1. Установите field_scale = {best_scale:.6f}")
        print(f"2. Ожидаемое барионное число: {best_baryon:.6f}")
        print(f"3. Ожидаемая масса: {best_mass:.1f} МэВ")
        
        if best_error > 0.1:
            print("4. ⚠️  Ошибка все еще велика, возможно нужна другая стратегия")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
    except Exception as e:
        print(f"\n\nОшибка при оптимизации: {e}")
        import traceback
        traceback.print_exc()
