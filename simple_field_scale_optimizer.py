#!/usr/bin/env python3
"""
Простой скрипт для автоматического подбора оптимального field_scale
для достижения барионного числа B ≈ 1.0

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
import os
import re
import subprocess
import tempfile
import shutil

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

def test_field_scale(field_scale):
    """Тестирует конкретное значение field_scale"""
    print(f"Тестируем field_scale = {field_scale:.6f} ... ", end="", flush=True)
    
    # Обновляем файл
    update_field_scale_in_file(field_scale)
    
    # Запускаем тест
    cmd = [
        "python", "-c", """
import sys
sys.path.append('.')
from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig

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

baryon_number = model.physics_calculator.baryon_calculator.compute_baryon_number(model.field_derivatives)
mass = model.physics_calculator.mass_calculator.compute_mass(model.energy_density.get_total_energy())

print(f'{float(baryon_number.real):.6f},{float(mass):.1f}')
"""
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            output = result.stdout.strip()
            if ',' in output:
                baryon_str, mass_str = output.split(',')
                baryon_number = float(baryon_str)
                mass = float(mass_str)
                error = abs(baryon_number - 1.0)
                print(f"B = {baryon_number:.6f}, M = {mass:.1f} МэВ, ошибка = {error:.6f}")
                return field_scale, baryon_number, mass, error
            else:
                print(f"ОШИБКА: неожиданный вывод: {output}")
                return None
        else:
            print(f"ОШИБКА: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("ТАЙМАУТ")
        return None
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None

def optimize_field_scale():
    """Оптимизирует field_scale для достижения B ≈ 1.0"""
    
    print("ОПТИМИЗАЦИЯ FIELD_SCALE ДЛЯ БАРИОННОГО ЧИСЛА")
    print("=" * 60)
    
    # Диапазон для поиска
    field_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    results = []
    
    print("Тестирование различных значений field_scale...")
    print()
    
    for i, scale in enumerate(field_scales):
        result = test_field_scale(scale)
        if result:
            results.append(result)
    
    print()
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 60)
    
    if not results:
        print("❌ Не удалось получить результаты!")
        return
    
    # Сортируем по ошибке (ближе к 1.0)
    results.sort(key=lambda x: x[3])
    
    print("Все результаты:")
    print("-" * 60)
    print(f"{'field_scale':>12} {'B':>10} {'Масса (МэВ)':>12} {'Ошибка':>10}")
    print("-" * 60)
    
    for i, (scale, baryon, mass, error) in enumerate(results):
        status = "✅" if i == 0 else "  "
        print(f"{status} {scale:10.4f} {baryon:10.5f} {mass:12.1f} {error:10.5f}")
    
    # Лучший результат
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
    
    return best_scale, best_baryon, best_mass, best_error

if __name__ == "__main__":
    try:
        optimize_field_scale()
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
    except Exception as e:
        print(f"\n\nОшибка при оптимизации: {e}")
        import traceback
        traceback.print_exc()
