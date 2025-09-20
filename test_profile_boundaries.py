#!/usr/bin/env python3
"""
Тест граничных условий радиального профиля.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from phaze_particles.utils.su2_fields import RadialProfile
from phaze_particles.utils.mathematical_foundations import ArrayBackend

def test_profile_boundaries():
    """Тест граничных условий профиля."""
    
    print("ТЕСТ ГРАНИЧНЫХ УСЛОВИЙ ПРОФИЛЯ")
    print("=" * 40)
    
    # Создаем профиль
    backend = ArrayBackend()
    profile = RadialProfile(
        profile_type="tanh",
        scale=1.0,
        center_value=np.pi,
        backend=backend
    )
    
    # Тестируем в центре и на границе
    r_center = 0.0
    r_boundary = 10.0  # Большое значение для "бесконечности"
    
    f_center = profile.evaluate(r_center)
    f_boundary = profile.evaluate(r_boundary)
    
    print(f"f(0) = {f_center:.6f} (ожидается π = {np.pi:.6f})")
    print(f"f(∞) = {f_boundary:.6f} (ожидается 0.0)")
    
    # Проверяем граничные условия
    center_ok = abs(f_center - np.pi) < 1e-10
    boundary_ok = abs(f_boundary) < 1e-10
    
    print(f"\nГраничные условия:")
    print(f"  Центр: {'✅' if center_ok else '❌'}")
    print(f"  Граница: {'✅' if boundary_ok else '❌'}")
    
    # Тестируем SU(2) поле
    print(f"\nТЕСТ SU(2) ПОЛЯ:")
    print("-" * 20)
    
    # Создаем простое поле в центре
    cos_f_center = np.cos(f_center)
    sin_f_center = np.sin(f_center)
    
    # U = cos f(r) 1 + i sin f(r) n̂(x) · σ⃗
    # В центре n̂ = (0, 0, 1) для простоты
    u_00_center = cos_f_center + 1j * sin_f_center * 1.0
    u_01_center = 1j * sin_f_center * (0.0 - 1j * 0.0)
    u_10_center = 1j * sin_f_center * (0.0 + 1j * 0.0)
    u_11_center = cos_f_center - 1j * sin_f_center * 1.0
    
    det_center = u_00_center * u_11_center - u_01_center * u_10_center
    
    print(f"В центре:")
    print(f"  cos(f) = {cos_f_center:.6f}")
    print(f"  sin(f) = {sin_f_center:.6f}")
    print(f"  det(U) = {det_center:.6f} (ожидается -1.0)")
    
    # На границе
    cos_f_boundary = np.cos(f_boundary)
    sin_f_boundary = np.sin(f_boundary)
    
    u_00_boundary = cos_f_boundary + 1j * sin_f_boundary * 1.0
    u_01_boundary = 1j * sin_f_boundary * (0.0 - 1j * 0.0)
    u_10_boundary = 1j * sin_f_boundary * (0.0 + 1j * 0.0)
    u_11_boundary = cos_f_boundary - 1j * sin_f_boundary * 1.0
    
    det_boundary = u_00_boundary * u_11_boundary - u_01_boundary * u_10_boundary
    
    print(f"\nНа границе:")
    print(f"  cos(f) = {cos_f_boundary:.6f}")
    print(f"  sin(f) = {sin_f_boundary:.6f}")
    print(f"  det(U) = {det_boundary:.6f} (ожидается 1.0)")
    
    # Проверяем детерминанты
    center_det_ok = abs(det_center - (-1.0)) < 1e-10
    boundary_det_ok = abs(det_boundary - 1.0) < 1e-10
    
    print(f"\nДетерминанты:")
    print(f"  Центр: {'✅' if center_det_ok else '❌'}")
    print(f"  Граница: {'✅' if boundary_det_ok else '❌'}")
    
    return center_ok and boundary_ok and center_det_ok and boundary_det_ok

if __name__ == "__main__":
    success = test_profile_boundaries()
    print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
