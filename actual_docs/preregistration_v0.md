# SA-CE v0 — pre-registration (fill BEFORE the first model run)
# Date: 2026-10-14/15 · Dataset: dev-clean frozen-USM latents (6.27 fps), standardized
#
# Dataset floors (filled by floors_v0.py after the cache lands — copy the printed board here):
#   Δ=1 :  blind ____ | memless ____ | linear ____
#   Δ=4 :  blind ____ | memless ____ | linear ____
#   Δ=18:  blind ____ | memless ____ | linear ____
#   Δ=24:  blind ____ | memless ____ | linear ____
#
# MECHANISM-DERIVED BANDS (Ishaan fills these, with a one-line reason each — no vibes):
# 1. ΔS24 @ 3000 steps (vs the memless floor):  band [ ____ , ____ ]  because: ________
# 2. First overnight ΔS24 (500 steps):          band [ ____ , ____ ]  because: ________
# 3. Which horizon falls fastest relative to floors? ____  because: ________
# 4. Order of floors at Δ=24 (circle): blind > memless > linear / linear > memless > blind
#    — reason: ________ (hint: standardized latents change what "mean prediction" costs)
#
# FALSIFIERS already standing (sa_ce_v1_design.md §6):
#   A. ΔS24 ≤ 0 at step 3000  => frozen-latent space is not a learnable target (kill or rethink)
#   B. 4D board ≤ single-horizon boards => the 4-horizon weighting is decoration
#   C. delta-net ≤ matched-SSM arm => the hybrid engine loses its first head-to-head