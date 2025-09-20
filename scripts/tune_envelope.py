#!/usr/bin/env python3
"""
Envelope parameters tuning for magnetic moment (μ) while preserving rE and balance.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig


def _to_host(x):
	return x.get() if hasattr(x, 'get') else x


def set_env_params(calc, shell_frac: float, width_frac: float, pos_w: float, neg_w: float) -> None:
	# MagneticMomentCalculator lives at physics calculator
	mm = calc.magnetic_calculator
	# set attributes dynamically (calculator tolerates missing via getattr defaults)
	setattr(mm, 'env_shell_fraction', float(shell_frac))
	setattr(mm, 'env_width_fraction', float(width_frac))
	setattr(mm, 'env_pos_weight', float(pos_w))
	setattr(mm, 'env_neg_weight', float(neg_w))


def evaluate_once(grid: int, box: float, config_type: str, r_scale: float,
				shell_frac: float, width_frac: float, pos_w: float, neg_w: float) -> Optional[Dict[str, float]]:
	cfg = ModelConfig(grid_size=grid, box_size=box, torus_config=config_type, r_scale=r_scale, validation_enabled=False, max_iterations=200)
	m = ProtonModel(cfg)
	if not (m.create_geometry() and m.build_fields() and m.calculate_energy()):
		return None
	# set envelope params before physics
	set_env_params(m.physics_calculator, shell_frac, width_frac, pos_w, neg_w)
	if not m.calculate_physics():
		return None
	pq = m.physical_quantities
	# energy balance quick from integrals
	dx = box / grid
	c2 = _to_host(m.energy_density.c2_term); c4 = _to_host(m.energy_density.c4_term); c6 = _to_host(m.energy_density.c6_term)
	e2 = float(np.sum(c2) * dx**3); e4 = float(np.sum(c4) * dx**3); e6 = float(np.sum(c6) * dx**3)
	et = e2 + e4 + e6
	e2r = e2/et if et>0 else 0.0
	vir = (-e2 + e4 + 3.0*e6)/et if et>0 else 0.0
	return dict(
		r_scale=float(r_scale), shell_frac=float(shell_frac), width_frac=float(width_frac), pos_w=float(pos_w), neg_w=float(neg_w),
		mu=float(pq.magnetic_moment), rE=float(pq.charge_radius), B=float(pq.baryon_number), mass=float(pq.mass),
		E2_ratio=e2r, virial=vir
	)


def objective(m: Dict[str, float], w_mu: float, w_r: float, w_e: float, w_v: float, w_B: float) -> float:
	# targets
	t_mu = 2.793; t_rE = 0.841
	mu_term = abs(m['mu'] - t_mu) / t_mu
	r_term = abs(m['rE'] - t_rE) / t_rE
	e_term = abs(m['E2_ratio'] - 0.5)
	v_term = abs(m['virial'])
	B_term = abs(m['B'] - 1.0)
	return w_mu*mu_term + w_r*r_term + w_e*e_term + w_v*v_term + w_B*B_term


def save_csv(rows: List[Dict[str, float]], out_dir: str, short: str) -> str:
	os.makedirs(out_dir, exist_ok=True)
	ts = datetime.now().strftime('%Y-%m-%dT%H.%M.%S')
	path = os.path.join(out_dir, f"{short}-{ts}.csv")
	fields = ['r_scale','shell_frac','width_frac','pos_w','neg_w','mu','rE','B','mass','E2_ratio','virial']
	with open(path, 'w', encoding='utf-8-sig', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for r in rows:
			w.writerow(r)
	return path


def main():
	ap = argparse.ArgumentParser(description='Tune envelope params for μ with constraints')
	ap.add_argument('--grid-size', type=int, default=32)
	ap.add_argument('--box-size', type=float, default=3.0)
	ap.add_argument('--config-type', type=str, default='120deg', choices=['120deg','clover','cartesian'])
	ap.add_argument('--r-scale', type=float, default=0.6)
	ap.add_argument('--shell-frac', type=str, default='0.5,0.8,1.0')
	ap.add_argument('--width-frac', type=str, default='0.3,0.5,0.8')
	ap.add_argument('--pos-w', type=str, default='0.6,0.75,0.9')
	ap.add_argument('--neg-w', type=str, default='0.1,0.25,0.4')
	ap.add_argument('--weights', type=str, default='{"mu":1.0,"r":0.5,"e":0.3,"v":0.3,"B":0.6}')
	ap.add_argument('--out-dir', type=str, default='results/proton/envelope_tuning')
	args = ap.parse_args()

	Ws = json.loads(args.weights)
	w_mu = float(Ws.get('mu',1.0)); w_r=float(Ws.get('r',0.5)); w_e=float(Ws.get('e',0.3)); w_v=float(Ws.get('v',0.3)); w_B=float(Ws.get('B',0.6))

	parse = lambda s: [float(x) for x in s.split(',') if x.strip()]
	shell_list = parse(args.shell_frac)
	width_list = parse(args.width_frac)
	pos_list = parse(args.pos_w)
	neg_list = parse(args.neg_w)

	rows: List[Dict[str, float]] = []
	best: Optional[Any] = None
	for sf in shell_list:
		for wf in width_list:
			for pw in pos_list:
				for nw in neg_list:
					res = evaluate_once(args.grid_size, args.box_size, args.config_type, args.r_scale, sf, wf, pw, nw)
					if res is None:
						continue
					rows.append(res)
					score = objective(res, w_mu, w_r, w_e, w_v, w_B)
					if best is None or score < best[0]:
						best = (score, res)
					print({
						'shell': round(sf,3), 'width': round(wf,3), 'pos': round(pw,3), 'neg': round(nw,3),
						'mu': round(res['mu'],3), 'rE': round(res['rE'],3), 'E2r': round(res['E2_ratio'],3), 'vir': round(res['virial'],3), 'B': round(res['B'],3), 'score': round(score,6)
					})

	short = f"env-grid{args.grid_size}-box{args.box_size}-r{args.r_scale}"
	path = save_csv(rows, args.out_dir, short)
	if best is None:
		print('No successful evaluations')
		return
	print('\nBEST:')
	print(json.dumps(best[1], indent=2))
	print('CSV:', path)


if __name__ == '__main__':
	main()
