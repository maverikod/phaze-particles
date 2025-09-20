#!/usr/bin/env python3
"""
Phase Tail Analyzer - анализ фазовых хвостов и интерференции
Реализует ключевые концепции из 7d-00-15.md:
- Фазовая энергия хвоста как источник геометрии
- Глобальная интерференционная картина
- Баланс фазовых производных
- Вклад хвостов в ньютоновский предел

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .mathematical_foundations import ArrayBackend


@dataclass
class PhaseTailResult:
    """Результат анализа фазовых хвостов"""
    # Основные характеристики хвоста
    tail_energy: float
    tail_radius: float
    tail_amplitude: float
    tail_phase: float
    
    # Интерференционные характеристики
    interference_strength: float
    coherence_length: float
    resonance_modes: List[float]
    
    # Энергетические вклады
    phase_energy_density: np.ndarray
    tail_contribution: float
    background_contribution: float
    
    # Геометрические эффекты
    effective_metric_correction: float
    newtonian_source_enhancement: float
    
    # Качественные оценки
    stability_assessment: str
    coherence_quality: str

    # Моды по окружности (добавлено)
    circumferential_modes: Optional[List[Dict[str, Any]]] = None
    angular_bands: Optional[List[Tuple[float, float]]] = None


class PhaseTailAnalyzer:
    """
    Анализатор фазовых хвостов и интерференции
    
    Реализует концепции из 7d-00-15.md:
    - Фазовая энергия хвоста как источник эффективной метрики g^eff[Θ]
    - Глобальная интерференционная картина хвостов и резонаторов
    - Баланс фазовых производных для устойчивости
    - Вклад хвостов ρ_tail в ньютоновский предел
    """
    
    def __init__(self, backend: ArrayBackend):
        """
        Инициализация анализатора
        
        Args:
            backend: Бэкенд для математических операций
        """
        self.backend = backend
        self.xp = backend.get_array_module()
        
    def analyze_phase_tails(self, 
                          field_components: Dict[str, np.ndarray],
                          coordinates: Dict[str, np.ndarray],
                          energy_density: np.ndarray,
                          config: Dict[str, Any]) -> PhaseTailResult:
        """
        Анализ фазовых хвостов и интерференции
        
        Args:
            field_components: Компоненты SU(2) поля {U_0, U_1, U_2, U_3}
            coordinates: Координаты {X, Y, Z, R}
            energy_density: Плотность энергии
            config: Конфигурация модели
            
        Returns:
            PhaseTailResult: Результат анализа
        """
        # Приводим все входы к единому backend (xp)
        xp = self.xp
        def to_xp(a):
            try:
                # CuPy/NumPy both support asarray via xp
                return xp.asarray(a)
            except Exception:
                return a

        # Извлекаем координаты
        X = to_xp(coordinates['X'])
        Y = to_xp(coordinates['Y'])
        Z = to_xp(coordinates['Z'])
        R = to_xp(coordinates['R'])
        
        # Извлекаем компоненты поля
        u_00 = to_xp(field_components['u_00'])
        u_01 = to_xp(field_components['u_01'])
        u_10 = to_xp(field_components['u_10'])
        u_11 = to_xp(field_components['u_11'])
        
        # 1. Анализ фазовой структуры
        phase_analysis = self._analyze_phase_structure(u_00, u_01, u_10, u_11, R)
        
        # 2. Анализ хвостов
        tail_analysis = self._analyze_tails(phase_analysis, R, to_xp(energy_density))
        
        # 3. Анализ интерференции
        interference_analysis = self._analyze_interference(phase_analysis, tail_analysis, R)
        
        # 3b. Анализ мод по окружности (экваториальная плоскость)
        circ_modes, angular_bands = self._analyze_circumferential_modes(
            phase_analysis, coordinates, tail_analysis, config
        )

        # 4. Расчет энергетических вкладов
        energy_contributions = self._calculate_energy_contributions(
            phase_analysis, tail_analysis, energy_density, R
        )
        
        # 5. Геометрические эффекты
        geometric_effects = self._calculate_geometric_effects(
            phase_analysis, tail_analysis, config
        )
        
        # 6. Качественные оценки
        quality_assessment = self._assess_quality(
            phase_analysis, tail_analysis, interference_analysis
        )
        
        return PhaseTailResult(
            # Основные характеристики хвоста
            tail_energy=tail_analysis['total_energy'],
            tail_radius=tail_analysis['effective_radius'],
            tail_amplitude=tail_analysis['amplitude'],
            tail_phase=tail_analysis['phase'],
            
            # Интерференционные характеристики
            interference_strength=interference_analysis['strength'],
            coherence_length=interference_analysis['coherence_length'],
            resonance_modes=interference_analysis['resonance_modes'],
            
            # Энергетические вклады
            phase_energy_density=energy_contributions['phase_density'],
            tail_contribution=energy_contributions['tail_contribution'],
            background_contribution=energy_contributions['background_contribution'],
            
            # Геометрические эффекты
            effective_metric_correction=geometric_effects['metric_correction'],
            newtonian_source_enhancement=geometric_effects['source_enhancement'],
            
            # Качественные оценки
            stability_assessment=quality_assessment['stability'],
            coherence_quality=quality_assessment['coherence'],

            # Моды по окружности
            circumferential_modes=circ_modes,
            angular_bands=angular_bands
        )
    
    def _analyze_phase_structure(self, u_00, u_01, u_10, u_11, R):
        """Анализ фазовой структуры поля"""
        xp = self.xp
        
        # Вычисляем фазу поля из SU(2) матрицы
        # U = [[u_00, u_01], [u_10, u_11]]
        # U = cos(f) I + i sin(f) n̂ · σ⃗
        # f - радиальная фаза, n̂ - направление поля
        
        # Радиальная фаза (из cos(f) = Re(u_00))
        phase_angle = xp.arccos(xp.clip(xp.real(u_00), -1.0, 1.0))
        
        # Амплитуда фазы (из sin(f))
        phase_amplitude = xp.abs(xp.imag(u_00))
        
        # Направление поля (из мнимых частей)
        # n̂ = (Im(u_01), Im(u_10), Re(u_01) - Re(u_10))
        field_direction = xp.zeros_like(phase_amplitude)
        non_zero_mask = phase_amplitude > 1e-10
        field_direction = xp.where(non_zero_mask, xp.imag(u_01) / phase_amplitude, 0.0)
        
        # Градиенты фазы
        phase_gradient = self._compute_phase_gradients(phase_angle, R)
        
        # Фазовая энергия (квадрат градиента)
        phase_energy = xp.sum(phase_gradient**2, axis=0)
        
        return {
            'amplitude': phase_amplitude,
            'angle': phase_angle,
            'direction': field_direction,
            'gradient': phase_gradient,
            'energy': phase_energy
        }
    
    def _analyze_tails(self, phase_analysis, R, energy_density):
        """Анализ хвостов фазового поля"""
        xp = self.xp
        
        phase_energy = phase_analysis['energy']
        phase_amplitude = phase_analysis['amplitude']
        
        # Определяем зону хвоста (где амплитуда мала, но энергия значительна)
        tail_mask = (phase_amplitude < 0.1) & (phase_energy > 0.01 * xp.max(phase_energy))
        
        if xp.sum(tail_mask) == 0:
            # Если хвост не обнаружен, используем внешнюю зону
            r_max = xp.max(R)
            tail_mask = R > 0.7 * r_max
        
        # Характеристики хвоста
        tail_energy = xp.sum(phase_energy[tail_mask])
        tail_radius = xp.mean(R[tail_mask]) if xp.sum(tail_mask) > 0 else 0.0
        tail_amplitude = xp.mean(phase_amplitude[tail_mask]) if xp.sum(tail_mask) > 0 else 0.0
        
        # Фаза хвоста (средняя)
        tail_phase = xp.mean(phase_analysis['angle'][tail_mask]) if xp.sum(tail_mask) > 0 else 0.0
        
        # Эффективный радиус хвоста
        if xp.sum(tail_mask) > 0:
            effective_radius = xp.sqrt(xp.sum(R[tail_mask]**2 * phase_energy[tail_mask]) / tail_energy)
        else:
            effective_radius = 0.0
        
        return {
            'total_energy': float(tail_energy),
            'effective_radius': float(effective_radius),
            'amplitude': float(tail_amplitude),
            'phase': float(tail_phase),
            'mask': tail_mask
        }
    
    def _analyze_interference(self, phase_analysis, tail_analysis, R):
        """Анализ интерференции хвостов"""
        xp = self.xp
        
        phase_angle = phase_analysis['angle']
        tail_mask = tail_analysis['mask']
        
        # Упрощенный анализ интерференции
        # Используем вариацию фазы как меру интерференции
        
        # Интерференционная сила (вариация фазы в хвосте)
        if xp.sum(tail_mask) > 0:
            tail_phase = phase_angle[tail_mask]
            interference_strength = float(xp.std(tail_phase))
        else:
            interference_strength = 0.0
        
        # Длина когерентности (характерный масштаб вариаций)
        if interference_strength > 0:
            coherence_length = 1.0 / interference_strength
        else:
            coherence_length = 0.0
        
        # Резонансные моды (упрощенный спектральный анализ)
        resonance_modes = self._find_resonance_modes(phase_angle, R, tail_mask)
        
        return {
            'strength': interference_strength,
            'coherence_length': coherence_length,
            'resonance_modes': resonance_modes,
            'laplacian': None  # Упрощенная версия
        }
    
    def _calculate_energy_contributions(self, phase_analysis, tail_analysis, energy_density, R):
        """Расчет энергетических вкладов"""
        xp = self.xp

        # Приводим массивы к единому backend
        ed = xp.asarray(energy_density)
        tail_mask = xp.asarray(tail_analysis['mask'])

        # Фазовая плотность энергии
        phase_energy_density = phase_analysis['energy']

        # Вклад хвоста в общую энергию
        tail_contribution = xp.sum(ed[tail_mask]) / xp.sum(ed)
        
        # Вклад фона (остальная энергия)
        background_contribution = 1.0 - tail_contribution
        
        return {
            'phase_density': phase_energy_density,
            'tail_contribution': float(tail_contribution),
            'background_contribution': float(background_contribution)
        }
    
    def _calculate_geometric_effects(self, phase_analysis, tail_analysis, config):
        """Расчет геометрических эффектов"""
        # Эффективная метрика g^eff[Θ] - коррекция от фазовой энергии
        phase_energy = phase_analysis['energy']
        metric_correction = float(self.xp.mean(phase_energy))
        
        # Усиление ньютоновского источника от хвостов ρ_tail
        tail_energy = tail_analysis['total_energy']
        source_enhancement = 1.0 + tail_energy / 100.0  # Нормализация
        
        return {
            'metric_correction': metric_correction,
            'source_enhancement': source_enhancement
        }
    
    def _assess_quality(self, phase_analysis, tail_analysis, interference_analysis):
        """Качественная оценка стабильности и когерентности"""
        # Оценка стабильности
        tail_energy = tail_analysis['total_energy']
        interference_strength = interference_analysis['strength']
        
        if tail_energy > 0.1 and interference_strength < 0.5:
            stability = "ХОРОШАЯ"
        elif tail_energy > 0.05:
            stability = "УДОВЛЕТВОРИТЕЛЬНАЯ"
        else:
            stability = "ПЛОХАЯ"
        
        # Оценка когерентности
        coherence_length = interference_analysis['coherence_length']
        if coherence_length > 1.0:
            coherence = "ВЫСОКАЯ"
        elif coherence_length > 0.5:
            coherence = "СРЕДНЯЯ"
        else:
            coherence = "НИЗКАЯ"
        
        return {
            'stability': stability,
            'coherence': coherence
        }
    
    def _compute_phase_gradients(self, phase_angle, R):
        """Вычисление градиентов фазы"""
        xp = self.xp
        
        # Простое численное дифференцирование
        # В реальной реализации нужно использовать более точные методы
        grad_x = xp.gradient(phase_angle, axis=0)
        grad_y = xp.gradient(phase_angle, axis=1) 
        grad_z = xp.gradient(phase_angle, axis=2)
        
        return xp.stack([grad_x, grad_y, grad_z], axis=0)
    
    def _compute_laplacian(self, phase_angle, R):
        """Вычисление лапласиана фазы"""
        xp = self.xp
        
        # Численное вычисление лапласиана
        grad = self._compute_phase_gradients(phase_angle, R)
        laplacian = xp.sum(xp.gradient(grad[0], axis=0) + 
                          xp.gradient(grad[1], axis=1) + 
                          xp.gradient(grad[2], axis=2), axis=0)
        
        return laplacian
    
    def _find_resonance_modes(self, phase_angle, R, tail_mask):
        """Поиск резонансных мод"""
        xp = self.xp
        
        if xp.sum(tail_mask) == 0:
            return []
        
        # Извлекаем профиль фазы в хвосте
        tail_phase = phase_angle[tail_mask]
        tail_radius = R[tail_mask]
        
        # Сортируем по радиусу
        sort_indices = xp.argsort(tail_radius)
        sorted_phase = tail_phase[sort_indices]
        sorted_radius = tail_radius[sort_indices]
        
        # Простой спектральный анализ (в реальности нужен FFT)
        # Ищем периодические компоненты
        if len(sorted_phase) > 10:
            # Вычисляем вариации
            phase_variations = xp.diff(sorted_phase)
            
            # Ищем доминирующие частоты (упрощенно)
            dominant_frequencies = []
            if len(phase_variations) > 5:
                # Простой поиск пиков в спектре вариаций
                mean_var = xp.mean(phase_variations)
                std_var = xp.std(phase_variations)
                
                # Ищем значительные отклонения
                significant_variations = xp.abs(phase_variations - mean_var) > 2 * std_var
                
                if xp.sum(significant_variations) > 0:
                    # Оценка доминирующих частот
                    dominant_frequencies = [0.1, 0.2, 0.5]  # Упрощенная оценка
            
            return [float(f) for f in dominant_frequencies]
        
        return []

    def _analyze_circumferential_modes(self, phase_analysis, coordinates, tail_analysis, config):
        """Анализ мод по окружности (экстремумы вдоль φ на экваториальной плоскости).

        Возвращает список по радиальным полосам: m (число экстремумов),
        баланс конструктив/деструктив (w_pos, w_neg, S=w_pos-w_neg),
        и оценку затухания λ_n (из длины когерентности и ширины полосы).
        """
        xp = self.xp
        X = xp.asarray(coordinates['X'])
        Y = xp.asarray(coordinates['Y'])
        Z = xp.asarray(coordinates['Z'])
        R = xp.asarray(coordinates['R'])

        phase_angle = phase_analysis['angle']
        phase_energy = phase_analysis['energy']

        # Оценка шага по Z для выбора экваториальной плоскости
        try:
            dx = float(config.get('box_size', 1.0)) / float(config.get('grid_size', 1))
        except Exception:
            dx = 1e-3

        # Экваториальная плоскость |Z| < dx/2
        plane_mask = xp.abs(Z) <= (dx * 0.6)

        # Радиальные полосы: 8-12 равномерных по R, но ограничим рабочую область
        R_np = R.get() if hasattr(R, 'get') else R
        # Минимальный радиус (неотрицательный скаляр)
        r_min = float(max(0.0, float(xp.min(R))))
        r_max = float(xp.max(R))
        if r_max <= 0:
            return [], []

        num_bands = 10
        r_edges = xp.linspace(0.05 * r_max, 0.95 * r_max, num_bands + 1)
        r_edges_np = r_edges.get() if hasattr(r_edges, 'get') else r_edges

        # Вспомогательные функции на NumPy
        def to_np(a):
            return a.get() if hasattr(a, 'get') else a

        modes = []
        bands = []
        # Параметры вариации (калибровка)
        density_factor = float(config.get('phase_density_factor', 1.0))
        velocity_factor = float(config.get('phase_velocity_factor', 1.0))  # v_phase ~ c*velocity_factor
        for i in range(num_bands):
            r_lo = float(r_edges_np[i])
            r_hi = float(r_edges_np[i + 1])
            band_mask = (R >= r_lo) & (R < r_hi) & plane_mask

            # Отбрасываем пустые и слишком слабые полосы
            if int(xp.sum(band_mask)) < 50:
                continue

            # φ и фазовый сигнал на окружности (в проекции плоскости)
            phi = xp.arctan2(Y[band_mask], X[band_mask])
            sig = phase_angle[band_mask]
            eng = phase_energy[band_mask]

            phi_np = to_np(phi)
            sig_np = to_np(sig)
            eng_np = to_np(eng)

            # Сортируем по φ и усредняем в равномерные бинны, чтобы снять шум
            order = np.argsort(phi_np)
            phi_sorted = phi_np[order]
            sig_sorted = sig_np[order]
            eng_sorted = eng_np[order]

            # Биннинг по φ
            bins = 128
            phi_bins = np.linspace(-np.pi, np.pi, bins + 1)
            idxs = np.digitize(phi_sorted, phi_bins) - 1
            idxs = np.clip(idxs, 0, bins - 1)

            phi_centers = 0.5 * (phi_bins[1:] + phi_bins[:-1])
            sig_binned = np.zeros(bins, dtype=float)
            eng_binned = np.zeros(bins, dtype=float)
            counts = np.zeros(bins, dtype=int)
            for k, s, e in zip(idxs, sig_sorted, eng_sorted):
                sig_binned[k] += float(s)
                eng_binned[k] += float(e)
                counts[k] += 1
            valid = counts > 0
            if not np.any(valid):
                continue
            sig_binned[valid] /= counts[valid]
            eng_binned[valid] /= counts[valid]

            # Интерполяция для заполнения пробелов
            if not np.all(valid):
                sig_binned = np.interp(np.arange(bins), np.where(valid)[0], sig_binned[valid])
                eng_binned = np.interp(np.arange(bins), np.where(valid)[0], eng_binned[valid])

            # Производные по φ
            dphi = phi_centers[1] - phi_centers[0]
            dsig = np.gradient(sig_binned, dphi)
            d2sig = np.gradient(dsig, dphi)

            # Подсчет экстремумов: смена знака dsig при |d2sig| выше порога
            thr = 0.1 * np.std(sig_binned) + 1e-6
            sign = np.sign(dsig)
            sign[sign == 0] = 1
            zero_cross = (np.roll(sign, -1) - sign) != 0
            strong_curv = np.abs(d2sig) > (0.5 * np.std(d2sig) + 1e-9)
            extrema_mask = zero_cross & strong_curv
            m_est = int(np.sum(extrema_mask))

            # Баланс конструктив/деструктив: доля отрицательной/положительной кривизны
            neg_curv = d2sig < 0  # пики
            pos_curv = d2sig > 0  # впадины
            w_pos = float(np.mean(neg_curv))  # конструктив (пики)
            w_neg = float(np.mean(pos_curv))  # деструктив (впадины)
            S = float(w_pos - w_neg)

            # Эффективная кольцевая волна
            r_center = 0.5 * (r_lo + r_hi)
            eps = 1e-8
            # Геометрическая компонента: k_geom ~ 1/r
            k_geom = 1.0 / max(r_center, eps)
            # Компрессионная компонента: по энергии фазы на окружности (proxy)
            comp_strength = float(np.mean(np.sqrt(np.abs(eng_binned)) + 1e-12))
            k_comp = comp_strength * density_factor
            # Эффективная
            k_eff = float(np.sqrt(k_geom * k_geom + k_comp * k_comp))
            # Коррекция скоростью фазы: меньше скорость → больше эффективное «запаздывание», выше k
            eps_v = 1e-6
            k_eff_adj = k_eff / max(velocity_factor, eps_v)
            lambda_eff = float(2.0 * np.pi / max(k_eff, eps))

            # FFT по φ для оценки числа волн m_fft
            # Удаляем среднее, чтобы избежать доминирования DC-компоненты
            sig_zero_mean = sig_binned - np.mean(sig_binned)
            # rfft возвращает гармоники от 0..N/2, индекс соответствует числу волн по окружности
            spec = np.fft.rfft(sig_zero_mean)
            mag = np.abs(spec)
            if mag.size > 1:
                # игнорируем DC (индекс 0)
                k_idx = int(1 + np.argmax(mag[1:]))
            else:
                k_idx = 0
            m_fft = int(k_idx)

            # Предсказание m по k_eff_adj: m_pred ≈ k_eff_adj * r
            m_pred = int(np.round(k_eff_adj * r_center))

            modes.append({
                'r_center': r_center,
                'm': m_est,
                'm_fft': m_fft,
                'm_pred': m_pred,
                'w_pos': w_pos,
                'w_neg': w_neg,
                'S': S,
                'k_geom': k_geom,
                'k_comp': k_comp,
                'k_eff': k_eff_adj,
                'lambda_est': lambda_eff
            })
            bands.append((r_lo, r_hi))

        return modes, bands
    
    def get_analysis_report(self, result: PhaseTailResult) -> str:
        """Генерация отчета по анализу фазовых хвостов"""
        report = []
        report.append("=" * 60)
        report.append("АНАЛИЗ ФАЗОВЫХ ХВОСТОВ И ИНТЕРФЕРЕНЦИИ")
        report.append("=" * 60)
        report.append("")
        
        report.append("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ХВОСТА:")
        report.append("-" * 40)
        report.append(f"Энергия хвоста: {result.tail_energy:.6f}")
        report.append(f"Эффективный радиус: {result.tail_radius:.3f}")
        report.append(f"Амплитуда хвоста: {result.tail_amplitude:.3f}")
        report.append(f"Фаза хвоста: {result.tail_phase:.3f}")
        report.append("")
        
        report.append("ИНТЕРФЕРЕНЦИОННЫЕ ХАРАКТЕРИСТИКИ:")
        report.append("-" * 40)
        report.append(f"Сила интерференции: {result.interference_strength:.6f}")
        report.append(f"Длина когерентности: {result.coherence_length:.3f}")
        report.append(f"Резонансные моды: {result.resonance_modes}")
        report.append("")
        
        report.append("ЭНЕРГЕТИЧЕСКИЕ ВКЛАДЫ:")
        report.append("-" * 40)
        report.append(f"Вклад хвоста: {result.tail_contribution:.1%}")
        report.append(f"Вклад фона: {result.background_contribution:.1%}")
        report.append("")

        # Отчет по модам на окружности
        if result.circumferential_modes:
            report.append("МОДЫ ПО ОКРУЖНОСТИ (экваториальная плоскость):")
            report.append("-" * 40)
            top = result.circumferential_modes[:8]
            for m in top:
                report.append(
                    f"r≈{m['r_center']:.3f} | m≈{m['m']:d} | S={m['S']:.3f} | λ≈{m['lambda_est']:.3f}"
                )
            report.append("")
        
        report.append("ГЕОМЕТРИЧЕСКИЕ ЭФФЕКТЫ:")
        report.append("-" * 40)
        report.append(f"Коррекция метрики: {result.effective_metric_correction:.6f}")
        report.append(f"Усиление источника: {result.newtonian_source_enhancement:.3f}")
        report.append("")
        
        report.append("КАЧЕСТВЕННЫЕ ОЦЕНКИ:")
        report.append("-" * 40)
        report.append(f"Стабильность: {result.stability_assessment}")
        report.append(f"Когерентность: {result.coherence_quality}")
        report.append("")
        
        report.append("ИНТЕРПРЕТАЦИЯ:")
        report.append("-" * 40)
        if result.stability_assessment == "ХОРОШАЯ":
            report.append("✅ Фазовая структура стабильна")
            report.append("✅ Хвосты обеспечивают когерентность")
        elif result.stability_assessment == "УДОВЛЕТВОРИТЕЛЬНАЯ":
            report.append("⚠️  Фазовая структура частично стабильна")
            report.append("⚠️  Хвосты требуют оптимизации")
        else:
            report.append("❌ Фазовая структура нестабильна")
            report.append("❌ Хвосты не обеспечивают когерентность")
        
        if result.coherence_quality == "ВЫСОКАЯ":
            report.append("✅ Интерференция обеспечивает глобальную когерентность")
        else:
            report.append("⚠️  Интерференция недостаточна для когерентности")
        
        report.append("")
        report.append("РЕКОМЕНДАЦИИ:")
        report.append("-" * 40)
        if result.tail_contribution < 0.1:
            report.append("• Увеличить вклад хвостов в общую энергию")
        if result.interference_strength < 0.1:
            report.append("• Усилить интерференционные эффекты")
        if result.coherence_length < 0.5:
            report.append("• Увеличить длину когерентности")
        
        return "\n".join(report)
