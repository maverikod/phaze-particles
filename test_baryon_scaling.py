#!/usr/bin/env python3
"""
Тест масштабирования барионного числа.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig

def test_baryon_scaling():
    """Тест масштабирования барионного числа."""
    
    print("ТЕСТ МАСШТАБИРОВАНИЯ БАРИОННОГО ЧИСЛА")
    print("=" * 50)
    
    # Создаем модель с разными масштабами
    configs = [
        {"r_scale": 0.5, "name": "Малый масштаб (0.5)"},
        {"r_scale": 1.0, "name": "Стандартный масштаб (1.0)"},
        {"r_scale": 1.5, "name": "Большой масштаб (1.5)"},
        {"r_scale": 2.0, "name": "Очень большой масштаб (2.0)"},
    ]
    
    results = []
    
    for config_data in configs:
        print(f"\n{config_data['name']}:")
        print("-" * 30)
        
        # Создаем конфигурацию
        config = ModelConfig(
            grid_size=32,
            box_size=4.0,
            c2=1.0,
            c4=1.0,
            c6=1.0,
            F_pi=186.0,
            e=5.45,
            r_scale=config_data["r_scale"]
        )
        config.config_type = "120deg"
        
        # Создаем модель
        model = ProtonModel(config)
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()
        
        # Получаем барионное число
        if hasattr(model, 'physical_quantities') and model.physical_quantities:
            baryon_number = model.physical_quantities.baryon_number
            print(f"  Барионное число: {baryon_number:.6f}")
            
            results.append({
                "r_scale": config_data["r_scale"],
                "baryon_number": baryon_number,
                "name": config_data["name"]
            })
        else:
            print("  Ошибка: физические величины не рассчитаны")
    
    # Анализ результатов
    print(f"\nАНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 50)
    
    best_result = None
    best_error = float('inf')
    
    for result in results:
        error = abs(result["baryon_number"] - 1.0)
        print(f"{result['name']}: B = {result['baryon_number']:.6f}, ошибка = {error:.6f}")
        
        if error < best_error:
            best_error = error
            best_result = result
    
    if best_result:
        print(f"\nЛучший результат: {best_result['name']}")
        print(f"  r_scale = {best_result['r_scale']}")
        print(f"  B = {best_result['baryon_number']:.6f}")
        print(f"  Ошибка = {best_error:.6f}")
        
        if best_error < 0.1:
            print("  ✅ Отличный результат!")
        elif best_error < 0.2:
            print("  ✅ Хороший результат!")
        elif best_error < 0.5:
            print("  ⚠️ Удовлетворительный результат")
        else:
            print("  ❌ Плохой результат")
    
    return best_result

if __name__ == "__main__":
    best_config = test_baryon_scaling()
    if best_config:
        print(f"\nРекомендуемый r_scale: {best_config['r_scale']}")
