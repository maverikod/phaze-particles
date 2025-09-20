#!/usr/bin/env python3
"""
Скрипт для автоматического подбора оптимального field_scale
для достижения барионного числа B ≈ 1.0

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
import numpy as np

def test_field_scale(field_scale, config):
    """Тестирует конкретное значение field_scale"""
    # Временно изменяем field_scale в коде
    from phaze_particles.utils.su2_fields import SU2FieldBuilder
    
    # Создаем модель
    model = ProtonModel(config)
    model.create_geometry()
    
    # Временно изменяем field_scale в builder
    original_build = SU2FieldBuilder._build_field_components
    
    def modified_build(self, n_x, n_y, n_z, profile):
        xp = self.backend.get_array_module()
        
        # Normalize radial distance
        r_norm = xp.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        zero_mask = r_norm < 1e-10
        
        # Field direction (radial direction for skyrmion topology)
        n_x_norm = xp.where(zero_mask, 1.0, self.X / r_norm)  # Default to x-direction at center
        n_y_norm = xp.where(zero_mask, 0.0, self.Y / r_norm)  # Default to y-direction at center
        n_z_norm = xp.where(zero_mask, 0.0, self.Z / r_norm)  # Default to z-direction at center

        # Compute radial profile
        f_r = profile.evaluate(self.R)
        
        # Scale the field for proper baryon number
        f_r = f_r * field_scale

        # Compute field components
        cos_f = xp.cos(f_r)
        sin_f = xp.sin(f_r)

        # Build SU(2) field U = cos(f) I + i sin(f) n̂ · σ⃗
        self.U_0 = cos_f
        self.U_1 = sin_f * n_x_norm
        self.U_2 = sin_f * n_y_norm
        self.U_3 = sin_f * n_z_norm
        
        return self.U_0, self.U_1, self.U_2, self.U_3
    
    # Временно заменяем метод
    SU2FieldBuilder._build_field_components = modified_build
    
    try:
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()
        
        # Получаем барионное число
        baryon_number = model.physics_calculator.baryon_calculator.compute_baryon_number(model.field_derivatives)
        mass = model.physics_calculator.mass_calculator.compute_mass(model.energy_density.get_total_energy())
        
        return float(baryon_number.real), float(mass)
    
    finally:
        # Восстанавливаем оригинальный метод
        SU2FieldBuilder._build_field_components = original_build

def optimize_field_scale():
    """Оптимизирует field_scale для достижения B ≈ 1.0"""
    
    print("ОПТИМИЗАЦИЯ FIELD_SCALE ДЛЯ БАРИОННОГО ЧИСЛА")
    print("=" * 60)
    
    # Конфигурация модели
    config = ModelConfig(
        grid_size=32,
        box_size=4.0,
        torus_config='120deg',
        r_scale=1.0,
        c2=1.0,
        c4=1.0,
        c6=0.0
    )
    
    # Диапазон для поиска
    field_scales = np.linspace(0.5, 8.0, 50)
    results = []
    
    print("Тестирование различных значений field_scale...")
    print()
    
    for i, scale in enumerate(field_scales):
        print(f"Тест {i+1:2d}/50: field_scale = {scale:5.2f}", end=" ... ")
        
        try:
            baryon_number, mass = test_field_scale(scale, config)
            error = abs(baryon_number - 1.0)
            results.append((scale, baryon_number, mass, error))
            print(f"B = {baryon_number:8.5f}, M = {mass:7.1f} МэВ, ошибка = {error:8.5f}")
            
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue
    
    print()
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 60)
    
    if not results:
        print("❌ Не удалось получить результаты!")
        return
    
    # Сортируем по ошибке (ближе к 1.0)
    results.sort(key=lambda x: x[3])
    
    print("Топ-10 лучших результатов:")
    print("-" * 60)
    print(f"{'field_scale':>12} {'B':>10} {'Масса (МэВ)':>12} {'Ошибка':>10}")
    print("-" * 60)
    
    for i, (scale, baryon, mass, error) in enumerate(results[:10]):
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
