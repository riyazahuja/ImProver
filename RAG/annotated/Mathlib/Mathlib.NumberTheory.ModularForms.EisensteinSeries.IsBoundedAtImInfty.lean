lemma summable_norm_eisSummand {k : ℤ} (hk : 3 ≤ k) (z : ℍ) :
    Summable fun (x : Fin 2 → ℤ) ↦ ‖(eisSummand k x z)‖ := by
  /-
    k : Int
    hk : LE.le 3 k
    z : UpperHalfPlane
    ⊢ Summable fun x => Norm.norm (EisensteinSeries.eisSummand k x z)
  -/
  have hk' : (2 : ℝ) < k := by norm_cast
  apply ((summable_one_div_norm_rpow hk').mul_left <| r z ^ (-k : ℝ)).of_nonneg_of_le
    (fun _ => Complex.abs.nonneg _)
  /-
    k : Int
    hk : LE.le 3 k
    z : UpperHalfPlane
    hk' : LT.lt 2 ↑k
    ⊢ ∀ (b : Fin 2 → Int), LE.le (Complex.abs (EisensteinSeries.eisSummand k b z)) …
  -/
  intro b
  /-
    k : Int
    hk : LE.le 3 k
    z : UpperHalfPlane
    hk' : LT.lt 2 ↑k
    b : Fin 2 → Int
    ⊢ LE.le (Complex.abs (EisensteinSeries.eisSummand k b z)) (HMul.hMul (HPow.hPo …
  -/
  simp only [eisSummand, map_zpow₀]
  /-
    k : Int
    hk : LE.le 3 k
    z : UpperHalfPlane
    hk' : LT.lt 2 ↑k
    b : Fin 2 → Int
    ⊢ LE.le (HPow.hPow (Complex.abs (HAdd.hAdd (HMul.hMul ↑(b 0) ↑z) ↑(b 1))) (Neg …
  -/
  exact_mod_cast summand_bound z (show 0 ≤ (k : ℝ) by positivity) b
  /-
    🎉 no goals
  -/


/-- The absolute value of the restricted sum is less than the full sum of the absolute values. -/
lemma abs_le_tsum_abs (N : ℕ) (a : Fin 2 → ZMod N) (k : ℤ) (hk : 3 ≤ k) (z : ℍ) :
    Complex.abs (eisensteinSeries a k z) ≤ ∑' (x : Fin 2 → ℤ), Complex.abs (eisSummand k x z) := by
  /-
    N : Nat
    a : Fin 2 → ZMod N
    k : Int
    hk : LE.le 3 k
    z : UpperHalfPlane
    ⊢ LE.le (Complex.abs (eisensteinSeries a k z)) (tsum fun x => Complex.abs (Eis …
  -/
  simp_rw [← Complex.norm_eq_abs, eisensteinSeries]
  apply le_trans (norm_tsum_le_tsum_norm ((summable_norm_eisSummand hk z).subtype _))
    (tsum_subtype_le (fun (x : Fin 2 → ℤ) ↦ ‖(eisSummand k x z)‖) _ (fun _ ↦ norm_nonneg _)
      (summable_norm_eisSummand hk z))


/-- Eisenstein series are bounded at infinity. -/
theorem isBoundedAtImInfty_eisensteinSeries_SIF {N : ℕ+} (a : Fin 2 → ZMod N) {k : ℤ} (hk : 3 ≤ k)
    (A : SL(2, ℤ)) : IsBoundedAtImInfty ((eisensteinSeries_SIF a k).toFun ∣[k] A) := by
  /-
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex k A (EisensteinSe …
  -/
  simp_rw [UpperHalfPlane.isBoundedAtImInfty_iff, eisensteinSeries_SIF] at *
  /-
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Exists fun M => Exists fun A_1 => ∀ (z : UpperHalfPlane), LE.le A_1 z.im → L …
  -/
  refine ⟨∑'(x : Fin 2 → ℤ), r ⟨⟨N, 2⟩, Nat.ofNat_pos⟩ ^ (-k) * ‖x‖ ^ (-k), 2, ?_⟩
  /-
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ ∀ (z : UpperHalfPlane), LE.le 2 z.im → LE.le (Norm.norm (SlashAction.map Com …
  -/
  intro z hz
  /-
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : LE.le 2 z.im
    ⊢ LE.le (Norm.norm (SlashAction.map Complex k A (eisensteinSeries a k) z)) (ts …
  -/
  obtain ⟨n, hn⟩ := (ModularGroup_T_zpow_mem_verticalStrip z N.2)
  rw [eisensteinSeries_slash_apply, ← eisensteinSeries_SIF_apply,
    ← T_zpow_width_invariant N k n (eisensteinSeries_SIF (a ᵥ* A) k) z]
  /-
    case intro
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : LE.le 2 z.im
    n : Int
    hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
    ⊢ LE.le (Norm.norm ((EisensteinSeries.eisensteinSeries_SIF (Matrix.vecMul a ↑( …
  -/
  apply le_trans (abs_le_tsum_abs N (a ᵥ* A) k hk _)
  /-
    case intro
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : LE.le 2 z.im
    n : Int
    hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
    ⊢ LE.le (tsum fun x => Complex.abs (EisensteinSeries.eisSummand k x (HSMul.hSM …
  -/
  have hk' : (2 : ℝ) < k := by norm_cast
  /-
    case intro
    N : PNat
    a : Fin 2 → ZMod ↑N
    k : Int
    hk : LE.le 3 k
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : LE.le 2 z.im
    n : Int
    hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
    hk' : LT.lt 2 ↑k
    ⊢ LE.le (tsum fun x => Complex.abs (EisensteinSeries.eisSummand k x (HSMul.hSM …
  -/
  apply tsum_le_tsum _ (summable_norm_eisSummand hk _)
    /-
      case intro
      N : PNat
      a : Fin 2 → ZMod ↑N
      k : Int
      hk : LE.le 3 k
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      hz : LE.le 2 z.im
      n : Int
      hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
      hk' : LT.lt 2 ↑k
      ⊢ Summable fun i => HMul.hMul (HPow.hPow (EisensteinSeries.r ⟨{ re := ↑↑N, im  …
    -/
  · exact_mod_cast (summable_one_div_norm_rpow hk').mul_left <| r ⟨⟨N, 2⟩, Nat.ofNat_pos⟩ ^ (-k)
    /-
      🎉 no goals
    -/
    /-
      N : PNat
      a : Fin 2 → ZMod ↑N
      k : Int
      hk : LE.le 3 k
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      hz : LE.le 2 z.im
      n : Int
      hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
      hk' : LT.lt 2 ↑k
      ⊢ ∀ (i : Fin 2 → Int), LE.le (Norm.norm (EisensteinSeries.eisSummand k i (HSMu …
    -/
  · intro x
    /-
      N : PNat
      a : Fin 2 → ZMod ↑N
      k : Int
      hk : LE.le 3 k
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      hz : LE.le 2 z.im
      n : Int
      hn : Membership.mem (UpperHalfPlane.verticalStrip (↑↑N) z.im) (HSMul.hSMul (HP …
      hk' : LT.lt 2 ↑k
      x : Fin 2 → Int
      ⊢ LE.le (Norm.norm (EisensteinSeries.eisSummand k x (HSMul.hSMul (HPow.hPow Mo …
    -/
    simp_rw [eisSummand, norm_zpow]
    exact_mod_cast
      summand_bound_of_mem_verticalStrip (lt_trans two_pos hk').le x two_pos
      (verticalStrip_anti_right N hz hn)


