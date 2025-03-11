/-- Rewrite the Gamma integral as an example of a Mellin transform. -/
theorem GammaIntegral_eq_mellin : GammaIntegral = mellin fun x => ↑(Real.exp (-x)) :=
                     /-
                       s : Complex
                       ⊢ Eq s.GammaIntegral (mellin (fun x => ↑(Real.exp (Neg.neg x))) s)
                     -/
  funext fun s => by simp only [mellin, GammaIntegral, smul_eq_mul, mul_comm]
                     /-
                       🎉 no goals
                     -/


/-- The derivative of the `Γ` integral, at any `s ∈ ℂ` with `1 < re s`, is given by the Mellin
transform of `log t * exp (-t)`. -/
theorem hasDerivAt_GammaIntegral {s : ℂ} (hs : 0 < s.re) :
    HasDerivAt GammaIntegral (∫ t : ℝ in Ioi 0, t ^ (s - 1) * (Real.log t * Real.exp (-t))) s := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ HasDerivAt Complex.GammaIntegral (MeasureTheory.integral (MeasureTheory.Meas …
  -/
  rw [GammaIntegral_eq_mellin]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ HasDerivAt (mellin fun x => ↑(Real.exp (Neg.neg x))) (MeasureTheory.integral …
  -/
  convert (mellin_hasDerivAt_of_isBigO_rpow (E := ℂ) _ _ (lt_add_one _) _ hs).2
    /-
      case convert_2
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ MeasureTheory.LocallyIntegrableOn (fun x => ↑(Real.exp (Neg.neg x))) (Set.Io …
    -/
  · refine (Continuous.continuousOn ?_).locallyIntegrableOn measurableSet_Ioi
    /-
      case convert_2
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Continuous fun x => ↑(Real.exp (Neg.neg x))
    -/
    exact continuous_ofReal.comp (Real.continuous_exp.comp continuous_neg)
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => ↑(Real.exp (Neg.neg x))) fun x =>  …
    -/
  · rw [← isBigO_norm_left]
    /-
      case convert_3
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm ↑(Real.exp (Neg.neg x))) …
    -/
    simp_rw [Complex.norm_eq_abs, abs_ofReal, ← Real.norm_eq_abs, isBigO_norm_left]
    /-
      case convert_3
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Real.exp (Neg.neg x)) fun x => HPo …
    -/
    simpa only [neg_one_mul] using (isLittleO_exp_neg_mul_rpow_atTop zero_lt_one _).isBigO
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => ↑(Real.exp (Neg.neg  …
    -/
  · simp_rw [neg_zero, rpow_zero]
    /-
      case convert_4
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => ↑(Real.exp (Neg.neg  …
    -/
    refine isBigO_const_of_tendsto (?_ : Tendsto _ _ (𝓝 (1 : ℂ))) one_ne_zero
    /-
      case convert_4
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Filter.Tendsto (fun x => ↑(Real.exp (Neg.neg x))) (nhdsWithin 0 (Set.Ioi 0)) …
    -/
    rw [(by simp : (1 : ℂ) = Real.exp (-0))]
    /-
      case convert_4
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Filter.Tendsto (fun x => ↑(Real.exp (Neg.neg x))) (nhdsWithin 0 (Set.Ioi 0)) …
    -/
    exact (continuous_ofReal.comp (Real.continuous_exp.comp continuous_neg)).continuousWithinAt
    /-
      🎉 no goals
    -/


theorem differentiableAt_GammaAux (s : ℂ) (n : ℕ) (h1 : 1 - s.re < n) (h2 : ∀ m : ℕ, s ≠ -m) :
    DifferentiableAt ℂ (GammaAux n) s := by
  /-
    s : Complex
    n : Nat
    h1 : LT.lt (HSub.hSub 1 s.re) ↑n
    h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ DifferentiableAt Complex (Complex.GammaAux n) s
  -/
  induction' n with n hn generalizing s
    /-
      case zero
      s : Complex
      h1 : LT.lt (HSub.hSub 1 s.re) ↑0
      h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      ⊢ DifferentiableAt Complex (Complex.GammaAux 0) s
    -/
  · refine (hasDerivAt_GammaIntegral ?_).differentiableAt
    /-
      case zero
      s : Complex
      h1 : LT.lt (HSub.hSub 1 s.re) ↑0
      h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      ⊢ LT.lt 0 s.re
    -/
    rw [Nat.cast_zero] at h1; linarith
                              /-
                                🎉 no goals
                              -/
    /-
      case succ
      n : Nat
      hn : ∀ (s : Complex), LT.lt (HSub.hSub 1 s.re) ↑n → (∀ (m : Nat), Ne s (Neg.ne …
      s : Complex
      h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
      h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      ⊢ DifferentiableAt Complex (Complex.GammaAux (HAdd.hAdd n 1)) s
    -/
  · dsimp only [GammaAux]
    /-
      case succ
      n : Nat
      hn : ∀ (s : Complex), LT.lt (HSub.hSub 1 s.re) ↑n → (∀ (m : Nat), Ne s (Neg.ne …
      s : Complex
      h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
      h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      ⊢ DifferentiableAt Complex (fun s => HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd  …
    -/
    specialize hn (s + 1)
    have a : 1 - (s + 1).re < ↑n := by
      rw [Nat.cast_succ] at h1; rw [Complex.add_re, Complex.one_re]; linarith
    have b : ∀ m : ℕ, s + 1 ≠ -m := by
      intro m; have := h2 (1 + m)
      contrapose! this
      rw [← eq_sub_iff_add_eq] at this
      simpa using this
    /-
      case succ
      n : Nat
      s : Complex
      h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
      h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      hn : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n → (∀ (m : Nat), Ne (HAdd.hAdd s …
      a : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n
      b : ∀ (m : Nat), Ne (HAdd.hAdd s 1) (Neg.neg ↑m)
      ⊢ DifferentiableAt Complex (fun s => HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd  …
    -/
    refine DifferentiableAt.div (DifferentiableAt.comp _ (hn a b) ?_) ?_ ?_
      /-
        case succ.refine_1
        n : Nat
        s : Complex
        h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
        h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
        hn : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n → (∀ (m : Nat), Ne (HAdd.hAdd s …
        a : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n
        b : ∀ (m : Nat), Ne (HAdd.hAdd s 1) (Neg.neg ↑m)
        ⊢ DifferentiableAt Complex (fun s => HAdd.hAdd s 1) s
      -/
    · rw [differentiableAt_add_const_iff (1 : ℂ)]; exact differentiableAt_id
                                                   /-
                                                     🎉 no goals
                                                   -/
      /-
        case succ.refine_2
        n : Nat
        s : Complex
        h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
        h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
        hn : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n → (∀ (m : Nat), Ne (HAdd.hAdd s …
        a : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n
        b : ∀ (m : Nat), Ne (HAdd.hAdd s 1) (Neg.neg ↑m)
        ⊢ DifferentiableAt Complex (fun s => s) s
      -/
    · exact differentiableAt_id
      /-
        🎉 no goals
      -/
      /-
        case succ.refine_3
        n : Nat
        s : Complex
        h1 : LT.lt (HSub.hSub 1 s.re) ↑(HAdd.hAdd n 1)
        h2 : ∀ (m : Nat), Ne s (Neg.neg ↑m)
        hn : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n → (∀ (m : Nat), Ne (HAdd.hAdd s …
        a : LT.lt (HSub.hSub 1 (HAdd.hAdd s 1).re) ↑n
        b : ∀ (m : Nat), Ne (HAdd.hAdd s 1) (Neg.neg ↑m)
        ⊢ Ne s 0
      -/
    · simpa using h2 0
      /-
        🎉 no goals
      -/


theorem differentiableAt_Gamma (s : ℂ) (hs : ∀ m : ℕ, s ≠ -m) : DifferentiableAt ℂ Gamma s := by
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ DifferentiableAt Complex Complex.Gamma s
  -/
  let n := ⌊1 - s.re⌋₊ + 1
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    ⊢ DifferentiableAt Complex Complex.Gamma s
  -/
  have hn : 1 - s.re < n := mod_cast Nat.lt_floor_add_one (1 - s.re)
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    hn : LT.lt (HSub.hSub 1 s.re) ↑n
    ⊢ DifferentiableAt Complex Complex.Gamma s
  -/
  apply (differentiableAt_GammaAux s n hn hs).congr_of_eventuallyEq
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    hn : LT.lt (HSub.hSub 1 s.re) ↑n
    ⊢ (nhds s).EventuallyEq Complex.Gamma (Complex.GammaAux n)
  -/
  let S := {t : ℂ | 1 - t.re < n}
  have : S ∈ 𝓝 s := by
    rw [mem_nhds_iff]; use S
    refine ⟨Subset.rfl, ?_, hn⟩
    have : S = re ⁻¹' Ioi (1 - n : ℝ) := by
      ext; rw [preimage, Ioi, mem_setOf_eq, mem_setOf_eq, mem_setOf_eq]; exact sub_lt_comm
    rw [this]
    exact Continuous.isOpen_preimage continuous_re _ isOpen_Ioi
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    hn : LT.lt (HSub.hSub 1 s.re) ↑n
    S : Set Complex := setOf fun t => LT.lt (HSub.hSub 1 t.re) ↑n
    this : Membership.mem (nhds s) S
    ⊢ (nhds s).EventuallyEq Complex.Gamma (Complex.GammaAux n)
  -/
  apply eventuallyEq_of_mem this
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    hn : LT.lt (HSub.hSub 1 s.re) ↑n
    S : Set Complex := setOf fun t => LT.lt (HSub.hSub 1 t.re) ↑n
    this : Membership.mem (nhds s) S
    ⊢ Set.EqOn Complex.Gamma (Complex.GammaAux n) S
  -/
  intro t ht; rw [mem_setOf_eq] at ht
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    n : Nat := HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re)) 1
    hn : LT.lt (HSub.hSub 1 s.re) ↑n
    S : Set Complex := setOf fun t => LT.lt (HSub.hSub 1 t.re) ↑n
    this : Membership.mem (nhds s) S
    t : Complex
    ht : LT.lt (HSub.hSub 1 t.re) ↑n
    ⊢ Eq (Complex.Gamma t) (Complex.GammaAux n t)
  -/
  apply Gamma_eq_GammaAux; linarith
                           /-
                             🎉 no goals
                           -/


/-- At `s = 0`, the Gamma function has a simple pole with residue 1. -/
theorem tendsto_self_mul_Gamma_nhds_zero : Tendsto (fun z : ℂ => z * Gamma z) (𝓝[≠] 0) (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun z => HMul.hMul z (Complex.Gamma z)) (nhdsWithin 0 (HasCo …
  -/
  rw [show 𝓝 (1 : ℂ) = 𝓝 (Gamma (0 + 1)) by simp only [zero_add, Complex.Gamma_one]]
  convert (Tendsto.mono_left _ nhdsWithin_le_nhds).congr'
    (eventuallyEq_of_mem self_mem_nhdsWithin Complex.Gamma_add_one) using 1
  /-
    case convert_1
    ⊢ Filter.Tendsto (fun x => Complex.Gamma (HAdd.hAdd x 1)) (nhds 0) (nhds (Comp …
  -/
  refine ContinuousAt.comp (g := Gamma) ?_ (continuous_id.add continuous_const).continuousAt
  /-
    case convert_1
    ⊢ ContinuousAt Complex.Gamma (HAdd.hAdd 0 1)
  -/
  refine (Complex.differentiableAt_Gamma _ fun m => ?_).continuousAt
  /-
    case convert_1
    m : Nat
    ⊢ Ne (HAdd.hAdd 0 1) (Neg.neg ↑m)
  -/
  rw [zero_add, ← ofReal_natCast, ← ofReal_neg, ← ofReal_one, Ne, ofReal_inj]
  /-
    case convert_1
    m : Nat
    ⊢ Not (Eq 1 (Neg.neg ↑m))
  -/
  refine (lt_of_le_of_lt ?_ zero_lt_one).ne'
  /-
    case convert_1
    m : Nat
    ⊢ LE.le (Neg.neg ↑m) 0
  -/
  exact neg_nonpos.mpr (Nat.cast_nonneg _)
  /-
    🎉 no goals
  -/


theorem differentiableAt_Gamma {s : ℝ} (hs : ∀ m : ℕ, s ≠ -m) : DifferentiableAt ℝ Gamma s := by
  /-
    s : Real
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ DifferentiableAt Real Real.Gamma s
  -/
  refine (Complex.differentiableAt_Gamma _ ?_).hasDerivAt.real_of_complex.differentiableAt
  /-
    s : Real
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ ∀ (m : Nat), Ne (↑s) (Neg.neg ↑m)
  -/
  simp_rw [← Complex.ofReal_natCast, ← Complex.ofReal_neg, Ne, Complex.ofReal_inj]
  /-
    s : Real
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ ∀ (m : Nat), Not (Eq s (Neg.neg ↑m))
  -/
  exact hs
  /-
    🎉 no goals
  -/


