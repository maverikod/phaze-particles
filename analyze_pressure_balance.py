#!/usr/bin/env python3
"""
Анализ баланса давлений в модели протона.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
from phaze_particles.utils.mathematical_foundations import ArrayBackend

def analyze_pressure_balance():
    """Анализ баланса давлений в модели протона."""
    
    print("АНАЛИЗ БАЛАНСА ДАВЛЕНИЙ")
    print("=" * 50)
    
    # Создаем модель
    config = ModelConfig(
        grid_size=32,
        box_size=4.0,
        c2=1.0,
        c4=1.0,
        c6=1.0,
        F_pi=186.0,
        e=5.45
    )
    config.config_type = "120deg"
    
    model = ProtonModel(config)
    model.create_geometry()
    model.build_fields()
    model.calculate_energy()
    
    # Получаем координаты и поля
    X = model.su2_field_builder.X
    Y = model.su2_field_builder.Y
    Z = model.su2_field_builder.Z
    R = model.su2_field_builder.R
    
    # Получаем SU(2) поле
    su2_field = model.su2_field
    
    # Анализируем поле по радиальным зонам
    print("\n1. АНАЛИЗ РАДИАЛЬНЫХ ЗОН:")
    print("-" * 30)
    
    # Определяем зоны
    r_max = np.max(R)
    r_center = 0.0
    r_torus = 0.5  # Примерный радиус тора
    r_tail = 1.5   # Начало хвоста
    
    # Зона 1: Центр (r < 0.2)
    center_mask = R < 0.2
    center_count = np.sum(center_mask)
    
    # Зона 2: Торы (0.2 < r < 1.0)
    torus_mask = (R >= 0.2) & (R < 1.0)
    torus_count = np.sum(torus_mask)
    
    # Зона 3: Хвост (1.0 < r < 2.0)
    tail_mask = (R >= 1.0) & (R < 2.0)
    tail_count = np.sum(tail_mask)
    
    # Зона 4: Фон (r > 2.0)
    background_mask = R >= 2.0
    background_count = np.sum(background_mask)
    
    print(f"Центр (r < 0.2): {center_count} точек")
    print(f"Торы (0.2 < r < 1.0): {torus_count} точек")
    print(f"Хвост (1.0 < r < 2.0): {tail_count} точек")
    print(f"Фон (r > 2.0): {background_count} точек")
    
    # Анализируем поле в каждой зоне
    print("\n2. АНАЛИЗ ПОЛЯ ПО ЗОНАМ:")
    print("-" * 30)
    
    # Центр
    if center_count > 0:
        center_u00 = su2_field.u_00[center_mask]
        center_u11 = su2_field.u_11[center_mask]
        center_det = np.real(center_u00 * center_u11 - su2_field.u_01[center_mask] * su2_field.u_10[center_mask])
        
        print(f"Центр:")
        print(f"  |U[0,0]|: {np.mean(np.abs(center_u00)):.3f} ± {np.std(np.abs(center_u00)):.3f}")
        print(f"  |U[1,1]|: {np.mean(np.abs(center_u11)):.3f} ± {np.std(np.abs(center_u11)):.3f}")
        print(f"  det(U): {np.mean(center_det):.3f} ± {np.std(center_det):.3f}")
    
    # Торы
    if torus_count > 0:
        torus_u00 = su2_field.u_00[torus_mask]
        torus_u11 = su2_field.u_11[torus_mask]
        torus_det = np.real(torus_u00 * torus_u11 - su2_field.u_01[torus_mask] * su2_field.u_10[torus_mask])
        
        print(f"Торы:")
        print(f"  |U[0,0]|: {np.mean(np.abs(torus_u00)):.3f} ± {np.std(np.abs(torus_u00)):.3f}")
        print(f"  |U[1,1]|: {np.mean(np.abs(torus_u11)):.3f} ± {np.std(np.abs(torus_u11)):.3f}")
        print(f"  det(U): {np.mean(torus_det):.3f} ± {np.std(torus_det):.3f}")
    
    # Хвост
    if tail_count > 0:
        tail_u00 = su2_field.u_00[tail_mask]
        tail_u11 = su2_field.u_11[tail_mask]
        tail_det = np.real(tail_u00 * tail_u11 - su2_field.u_01[tail_mask] * su2_field.u_10[tail_mask])
        
        print(f"Хвост:")
        print(f"  |U[0,0]|: {np.mean(np.abs(tail_u00)):.3f} ± {np.std(np.abs(tail_u00)):.3f}")
        print(f"  |U[1,1]|: {np.mean(np.abs(tail_u11)):.3f} ± {np.std(np.abs(tail_u11)):.3f}")
        print(f"  det(U): {np.mean(tail_det):.3f} ± {np.std(tail_det):.3f}")
    
    # Фон
    if background_count > 0:
        bg_u00 = su2_field.u_00[background_mask]
        bg_u11 = su2_field.u_11[background_mask]
        bg_det = np.real(bg_u00 * bg_u11 - su2_field.u_01[background_mask] * su2_field.u_10[background_mask])
        
        print(f"Фон:")
        print(f"  |U[0,0]|: {np.mean(np.abs(bg_u00)):.3f} ± {np.std(np.abs(bg_u00)):.3f}")
        print(f"  |U[1,1]|: {np.mean(np.abs(bg_u11)):.3f} ± {np.std(np.abs(bg_u11)):.3f}")
        print(f"  det(U): {np.mean(bg_det):.3f} ± {np.std(bg_det):.3f}")
    
    # Анализ непрерывности
    print("\n3. АНАЛИЗ НЕПРЕРЫВНОСТИ:")
    print("-" * 30)
    
    # Проверяем градиенты поля
    dx = model.config.box_size / model.config.grid_size
    
    # Градиенты U[0,0]
    grad_u00_x = np.gradient(su2_field.u_00, dx, axis=0)
    grad_u00_y = np.gradient(su2_field.u_00, dx, axis=1)
    grad_u00_z = np.gradient(su2_field.u_00, dx, axis=2)
    
    grad_magnitude = np.sqrt(np.abs(grad_u00_x)**2 + np.abs(grad_u00_y)**2 + np.abs(grad_u00_z)**2)
    
    print(f"Градиенты поля:")
    print(f"  |∇U[0,0]|: {np.mean(grad_magnitude):.3f} ± {np.std(grad_magnitude):.3f}")
    print(f"  Максимум: {np.max(grad_magnitude):.3f}")
    print(f"  Минимум: {np.min(grad_magnitude):.3f}")
    
    # Анализ по зонам
    if center_count > 0:
        center_grad = grad_magnitude[center_mask]
        print(f"  Центр: {np.mean(center_grad):.3f} ± {np.std(center_grad):.3f}")
    
    if torus_count > 0:
        torus_grad = grad_magnitude[torus_mask]
        print(f"  Торы: {np.mean(torus_grad):.3f} ± {np.std(torus_grad):.3f}")
    
    if tail_count > 0:
        tail_grad = grad_magnitude[tail_mask]
        print(f"  Хвост: {np.mean(tail_grad):.3f} ± {np.std(tail_grad):.3f}")
    
    if background_count > 0:
        bg_grad = grad_magnitude[background_mask]
        print(f"  Фон: {np.mean(bg_grad):.3f} ± {np.std(bg_grad):.3f}")
    
    # Анализ энергии по зонам
    print("\n4. АНАЛИЗ ЭНЕРГИИ ПО ЗОНАМ:")
    print("-" * 30)
    
    if hasattr(model, 'energy_density') and model.energy_density:
        energy_density = model.energy_density
        
        # Получаем компоненты энергии
        if hasattr(energy_density, 'c2_term'):
            e2_density = energy_density.c2_term
            e4_density = energy_density.c4_term
            e6_density = energy_density.c6_term
            total_density = e2_density + e4_density + e6_density
            
            print(f"Плотность энергии:")
            
            if center_count > 0:
                center_e2 = np.sum(e2_density[center_mask]) * dx**3
                center_e4 = np.sum(e4_density[center_mask]) * dx**3
                center_e6 = np.sum(e6_density[center_mask]) * dx**3
                center_total = np.sum(total_density[center_mask]) * dx**3
                
                print(f"  Центр: E₂={center_e2:.3f}, E₄={center_e4:.3f}, E₆={center_e6:.3f}, Total={center_total:.3f}")
            
            if torus_count > 0:
                torus_e2 = np.sum(e2_density[torus_mask]) * dx**3
                torus_e4 = np.sum(e4_density[torus_mask]) * dx**3
                torus_e6 = np.sum(e6_density[torus_mask]) * dx**3
                torus_total = np.sum(total_density[torus_mask]) * dx**3
                
                print(f"  Торы: E₂={torus_e2:.3f}, E₄={torus_e4:.3f}, E₆={torus_e6:.3f}, Total={torus_total:.3f}")
            
            if tail_count > 0:
                tail_e2 = np.sum(e2_density[tail_mask]) * dx**3
                tail_e4 = np.sum(e4_density[tail_mask]) * dx**3
                tail_e6 = np.sum(e6_density[tail_mask]) * dx**3
                tail_total = np.sum(total_density[tail_mask]) * dx**3
                
                print(f"  Хвост: E₂={tail_e2:.3f}, E₄={tail_e4:.3f}, E₆={tail_e6:.3f}, Total={tail_total:.3f}")
            
            if background_count > 0:
                bg_e2 = np.sum(e2_density[background_mask]) * dx**3
                bg_e4 = np.sum(e4_density[background_mask]) * dx**3
                bg_e6 = np.sum(e6_density[background_mask]) * dx**3
                bg_total = np.sum(total_density[background_mask]) * dx**3
                
                print(f"  Фон: E₂={bg_e2:.3f}, E₄={bg_e4:.3f}, E₆={bg_e6:.3f}, Total={bg_total:.3f}")
    
    # Анализ барионного числа по зонам
    print("\n5. АНАЛИЗ БАРИОННОГО ЧИСЛА ПО ЗОНАМ:")
    print("-" * 30)
    
    if hasattr(model, 'field_derivatives') and model.field_derivatives:
        # Получаем барионную плотность
        baryon_density = model.field_derivatives.get('baryon_density', None)
        
        if baryon_density is not None:
            # Конвертируем в numpy если нужно
            if hasattr(baryon_density, 'get'):
                baryon_density = baryon_density.get()
            
            # Конвертируем маски в numpy если нужно
            if hasattr(center_mask, 'get'):
                center_mask = center_mask.get()
            if hasattr(torus_mask, 'get'):
                torus_mask = torus_mask.get()
            if hasattr(tail_mask, 'get'):
                tail_mask = tail_mask.get()
            if hasattr(background_mask, 'get'):
                background_mask = background_mask.get()
            
            if center_count > 0:
                center_baryon = np.sum(baryon_density[center_mask]) * dx**3
                print(f"  Центр: B = {center_baryon:.6f}")
            
            if torus_count > 0:
                torus_baryon = np.sum(baryon_density[torus_mask]) * dx**3
                print(f"  Торы: B = {torus_baryon:.6f}")
            
            if tail_count > 0:
                tail_baryon = np.sum(baryon_density[tail_mask]) * dx**3
                print(f"  Хвост: B = {tail_baryon:.6f}")
            
            if background_count > 0:
                bg_baryon = np.sum(baryon_density[background_mask]) * dx**3
                print(f"  Фон: B = {bg_baryon:.6f}")
            
            total_baryon = np.sum(baryon_density) * dx**3
            print(f"  Общее: B = {total_baryon:.6f}")
    
    print("\n6. ДИАГНОСТИКА ПРОБЛЕМ:")
    print("-" * 30)
    
    # Проверяем граничные условия
    print("Граничные условия:")
    
    # Центр (должен быть U = -I)
    if center_count > 0:
        center_u00 = su2_field.u_00[center_mask]
        center_u11 = su2_field.u_11[center_mask]
        center_det = np.real(center_u00 * center_u11 - su2_field.u_01[center_mask] * su2_field.u_10[center_mask])
        
        expected_center = -1.0  # U(0) = -I
        center_error = np.mean(np.abs(center_det - expected_center))
        print(f"  Центр: det(U) = {np.mean(center_det):.3f} (ожидается {expected_center:.3f}, ошибка: {center_error:.3f})")
    
    # Граница (должна быть U = I)
    if background_count > 0:
        bg_u00 = su2_field.u_00[background_mask]
        bg_u11 = su2_field.u_11[background_mask]
        bg_det = np.real(bg_u00 * bg_u11 - su2_field.u_01[background_mask] * su2_field.u_10[background_mask])
        
        expected_boundary = 1.0  # U(∞) = I
        boundary_error = np.mean(np.abs(bg_det - expected_boundary))
        print(f"  Граница: det(U) = {np.mean(bg_det):.3f} (ожидается {expected_boundary:.3f}, ошибка: {boundary_error:.3f})")
    
    print("\nАНАЛИЗ ЗАВЕРШЕН")
    print("=" * 50)

if __name__ == "__main__":
    analyze_pressure_balance()
