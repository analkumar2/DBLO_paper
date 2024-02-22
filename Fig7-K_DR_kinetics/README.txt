>>> m, b, r, pvalue, _ = scs.linregress(np.array(A_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(0.05250563321251393, 0.17238742924651002)
>>> m, b, r, pvalue, _ = scs.linregress(np.array(B_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(0.17269561121420934, 6.210354529530469e-06)
>>> m, b, r, pvalue, _ = scs.linregress(np.array(C_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(0.045309776695944604, 0.23905531017838175)
>>> m, b, r, pvalue, _ = scs.linregress(np.array(D_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(-0.13508180816155552, 0.0004244515788151579)
>>> m, b, r, pvalue, _ = scs.linregress(np.array(F_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(0.06963225417044756, 0.07019856691407486)
>>> m, b, r, pvalue, _ = scs.linregress(np.array(Taum65_list), np.array(DBLO_list)*1e3)
>>> r, pvalue
(-0.47949017877453953, 3.2495057504939605e-40)