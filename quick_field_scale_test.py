#!/usr/bin/env python3
"""
Быстрый тест field_scale для подбора оптимального значения

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig

def test_field_scale(field_scale):
    """Тестирует конкретное значение field_scale"""
    print(f"Тестируем field_scale = {field_scale:.6f} ... ", end="", flush=True)
    
    # Временно изменяем field_scale в файле
    file_path = "phaze_particles/utils/su2_fields.py"
    
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем field_scale
    import re
    pattern = r'field_scale = [0-9.]+'
    replacement = f'field_scale = {field_scale}'
    new_content = re.sub(pattern, replacement, content)
    
    # Записываем обратно
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    try:
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
        
        baryon_real = float(baryon_number.real)
        mass_float = float(mass)
        error = abs(baryon_real - 1.0)
        
        print(f"B = {baryon_real:.6f}, M = {mass_float:.1f} МэВ, ошибка = {error:.6f}")
        return field_scale, baryon_real, mass_float, error
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None

def main():
    """Основная функция"""
    print("БЫСТРЫЙ ТЕСТ FIELD_SCALE")
    print("=" * 40)
    
    # Тестируем несколько значений
    test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    results = []
    
    for scale in test_values:
        result = test_field_scale(scale)
        if result:
            results.append(result)
    
    print()
    print("РЕЗУЛЬТАТЫ:")
    print("-" * 40)
    print(f"{'field_scale':>12} {'B':>10} {'Масса':>8} {'Ошибка':>10}")
    print("-" * 40)
    
    # Сортируем по ошибке
    results.sort(key=lambda x: x[3])
    
    for i, (scale, baryon, mass, error) in enumerate(results):
        status = "✅" if i == 0 else "  "
        print(f"{status} {scale:10.4f} {baryon:10.5f} {mass:8.1f} {error:10.5f}")
    
    if results:
        best_scale, best_baryon, best_mass, best_error = results[0]
        print()
        print(f"ЛУЧШИЙ РЕЗУЛЬТАТ: field_scale = {best_scale:.6f}")
        print(f"Барионное число = {best_baryon:.6f}")
        print(f"Масса = {best_mass:.1f} МэВ")
        print(f"Ошибка = {best_error:.6f}")

if __name__ == "__main__":
    main()
