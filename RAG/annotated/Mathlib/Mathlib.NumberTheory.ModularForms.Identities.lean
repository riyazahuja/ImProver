theorem vAdd_width_periodic (N : ℕ) (k n : ℤ) (f : SlashInvariantForm (Gamma N) k) (z : ℍ) :
    f (((N * n) : ℝ) +ᵥ z) = f z := by
  /-
    N : Nat
    k n : Int
    f : SlashInvariantForm (CongruenceSubgroup.Gamma N) k
    z : UpperHalfPlane
    ⊢ Eq (f (HVAdd.hVAdd (HMul.hMul ↑N ↑n) z)) (f z)
  -/
  norm_cast
  /-
    N : Nat
    k n : Int
    f : SlashInvariantForm (CongruenceSubgroup.Gamma N) k
    z : UpperHalfPlane
    ⊢ Eq (f (HVAdd.hVAdd (↑(HMul.hMul (↑N) n)) z)) (f z)
  -/
  rw [← modular_T_zpow_smul z (N * n)]
  /-
    N : Nat
    k n : Int
    f : SlashInvariantForm (CongruenceSubgroup.Gamma N) k
    z : UpperHalfPlane
    ⊢ Eq (f (HSMul.hSMul (HPow.hPow ModularGroup.T (HMul.hMul (↑N) n)) z)) (f z)
  -/
  convert slash_action_eqn' f (ModularGroup_T_pow_mem_Gamma N (N * n) (Int.dvd_mul_right N n)) z
  simp only [Fin.isValue, ModularGroup.coe_T_zpow (N * n), of_apply, cons_val', cons_val_zero,
    empty_val', cons_val_fin_one, cons_val_one, head_fin_const, Int.cast_zero, zero_mul, head_cons,
    Int.cast_one, zero_add, one_zpow, one_mul]


theorem T_zpow_width_invariant (N : ℕ) (k n : ℤ) (f : SlashInvariantForm (Gamma N) k) (z : ℍ) :
    f (((ModularGroup.T ^ (N * n))) • z) = f z := by
  /-
    N : Nat
    k n : Int
    f : SlashInvariantForm (CongruenceSubgroup.Gamma N) k
    z : UpperHalfPlane
    ⊢ Eq (f (HSMul.hSMul (HPow.hPow ModularGroup.T (HMul.hMul (↑N) n)) z)) (f z)
  -/
  rw [modular_T_zpow_smul z (N * n)]
  /-
    N : Nat
    k n : Int
    f : SlashInvariantForm (CongruenceSubgroup.Gamma N) k
    z : UpperHalfPlane
    ⊢ Eq (f (HVAdd.hVAdd (↑(HMul.hMul (↑N) n)) z)) (f z)
  -/
  simpa only [Int.cast_mul, Int.cast_natCast] using vAdd_width_periodic N k n f z
  /-
    🎉 no goals
  -/


