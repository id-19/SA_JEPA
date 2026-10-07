# my_sel_s4d.py — rung 3: S4D with input-dependent Delta (selective)
#
# Spec only. I write this file. The 07 Oct stub (blanks for theta / cos / sin /
# complex write coefficient / rotated update) was deleted at my request.
#
# Start from my_s4d.py and alter it. One complex scalar per channel:  a = -rate + i*omega
# ZOH of  h' = a*h + B*x  over one ragged step delta_k gives
#     h_k = e^{a*d_k} * h_{k-1}  +  (1 - e^{a*d_k}) / a  *  B * x_k
#   e^{a*d} = r*(cos(theta) + i*sin(theta)),   r = exp(-rate*d),   theta = omega*d
#   ONE d per step scales BOTH r and theta -- both coefficients are functions of
#   the SAME number e^{a*d}.
#
# contract:
#   in   (B, T, d_input)          out (B, T, d_output)
#   state   h       = (B, d_state, 2)      -- the 2 is Re/Im of the complex state
#   A parts log_rate (d_state,)   omega (d_state,)
#   B       (d_input, d_state, 2)          C (d_state, 2, d_output)
#   new     W_delta (d_state, d_input)     b_delta (d_state,)
#
# pass-bars before this counts (the six):
#   1 determinism   2 shape across several (B,T)   3 no param with grad None
#   4 BIBO on the TRAINED r                       5 T=2 by hand   6 causality
# watch 3: if delta collapses to ~0, e^{a*0}=1 and A stops mattering while every
#          grad still flows -- the check passes and the arm tests nothing.
#
# board: wall 0.20245 / S4D 0.02874 (03 Oct, the number to beat).
# pre-register before running: the loss band, AND whether delta_t moves at all on
# 25 stationary sines.
