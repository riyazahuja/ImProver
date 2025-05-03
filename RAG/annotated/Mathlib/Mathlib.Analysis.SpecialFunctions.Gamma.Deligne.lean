/-- Deligne's archimedean Gamma factor for a real infinite place.

See "Valeurs de fonctions L et periodes d'integrales" § 5.3. Note that this is not the same as
`Real.Gamma`; in particular it is a function `ℂ → ℂ`. -/
noncomputable def Gammaℝ (s : ℂ) := π ^ (-s / 2) * Gamma (s / 2)


lemma Gammaℝ_def (s : ℂ) : Gammaℝ s = π ^ (-s / 2) * Gamma (s / 2) := rfl


/-- Deligne's archimedean Gamma factor for a complex infinite place.

See "Valeurs de fonctions L et periodes d'integrales" § 5.3. (Some authors omit the factor of 2).
Note that this is not the same as `Complex.Gamma`. -/
noncomputable def Gammaℂ (s : ℂ) := 2 * (2 * π) ^ (-s) * Gamma s


lemma Gammaℂ_def (s : ℂ) : Gammaℂ s = 2 * (2 * π) ^ (-s) * Gamma s := rfl


lemma Gammaℝ_add_two {s : ℂ} (hs : s ≠ 0) : Gammaℝ (s + 2) = Gammaℝ s * s / 2 / π := by
  rw [Gammaℝ_def, Gammaℝ_def, neg_div, add_div, neg_add, div_self two_ne_zero,
    Gamma_add_one _ (div_ne_zero hs two_ne_zero),
    cpow_add _ _ (ofReal_ne_zero.mpr pi_ne_zero), cpow_neg_one]
  /-
    s : Complex
    hs : Ne s 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg.neg (HDiv.hDiv s 2))) (I …
  -/
  field_simp [pi_ne_zero]
  /-
    s : Complex
    hs : Ne s 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg s) 2)) (H …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma Gammaℂ_add_one {s : ℂ} (hs : s ≠ 0) : Gammaℂ (s + 1) = Gammaℂ s * s / 2 / π := by
  rw [Gammaℂ_def, Gammaℂ_def, Gamma_add_one _ hs, neg_add,
    cpow_add _ _ (mul_ne_zero two_ne_zero (ofReal_ne_zero.mpr pi_ne_zero)), cpow_neg_one]
  /-
    s : Complex
    hs : Ne s 0
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (HMul.hMul (HPow.hPow (HMul.hMul 2 ↑Real.pi) (Neg …
  -/
  field_simp [pi_ne_zero]
  /-
    s : Complex
    hs : Ne s 0
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (HPow.hPow (HMul.hMul 2 ↑Real.pi) (Neg.neg s))) ( …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma Gammaℝ_ne_zero_of_re_pos {s : ℂ} (hs : 0 < re s) : Gammaℝ s ≠ 0 := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ Ne s.Gammaℝ 0
  -/
  apply mul_ne_zero
    /-
      case ha
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Ne (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg s) 2)) 0
    -/
  · simp [pi_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case hb
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Ne (Complex.Gamma (HDiv.hDiv s 2)) 0
    -/
  · apply Gamma_ne_zero_of_re_pos
    /-
      case hb.hs
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ LT.lt 0 (HDiv.hDiv s 2).re
    -/
    rw [div_ofNat_re]
    /-
      case hb.hs
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ LT.lt 0 (HDiv.hDiv s.re 2)
    -/
    exact div_pos hs two_pos
    /-
      🎉 no goals
    -/


lemma Gammaℝ_eq_zero_iff {s : ℂ} : Gammaℝ s = 0 ↔ ∃ n : ℕ, s = -(2 * n) := by
  /-
    s : Complex
    ⊢ Iff (Eq s.Gammaℝ 0) (Exists fun n => Eq s (Neg.neg (HMul.hMul 2 ↑n)))
  -/
  simp [Gammaℝ_def, Complex.Gamma_eq_zero_iff, pi_ne_zero, div_eq_iff (two_ne_zero' ℂ), mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma Gammaℝ_one : Gammaℝ 1 = 1 := by
  /-
    ⊢ Eq (Complex.Gammaℝ 1) 1
  -/
  rw [Gammaℝ_def, Complex.Gamma_one_half_eq]
  /-
    ⊢ Eq (HMul.hMul (HPow.hPow (↑Real.pi) (-1 / 2)) (HPow.hPow (↑Real.pi) (1 / 2)) …
  -/
  simp [neg_div, cpow_neg, inv_mul_cancel, pi_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma Gammaℂ_one : Gammaℂ 1 = 1 / π := by
  /-
    ⊢ Eq (Complex.Gammaℂ 1) (HDiv.hDiv 1 ↑Real.pi)
  -/
  rw [Gammaℂ_def, cpow_neg_one, Complex.Gamma_one]
  /-
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Inv.inv (HMul.hMul 2 ↑Real.pi))) 1) (HDiv.hDiv 1 …
  -/
  field_simp [pi_ne_zero]
  /-
    🎉 no goals
  -/


lemma differentiable_Gammaℝ_inv : Differentiable ℂ (fun s ↦ (Gammaℝ s)⁻¹) := by
  /-
    ⊢ Differentiable Complex fun s => Inv.inv s.Gammaℝ
  -/
  conv => enter [2, s]; rw [Gammaℝ, mul_inv]
  /-
    ⊢ Differentiable Complex fun s => HMul.hMul (Inv.inv (HPow.hPow (↑Real.pi) (HD …
  -/
  refine Differentiable.mul (fun s ↦ .inv ?_ (by simp [pi_ne_zero])) ?_
    /-
      case refine_1
      s : Complex
      ⊢ DifferentiableAt Complex (fun s => HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg  …
    -/
  · refine ((differentiableAt_id.neg.div_const (2 : ℂ)).const_cpow ?_)
    /-
      case refine_1
      s : Complex
      ⊢ Or (Ne (↑Real.pi) 0) (Ne (HDiv.hDiv (Neg.neg s) 2) 0)
    -/
    exact Or.inl (ofReal_ne_zero.mpr pi_ne_zero)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ Differentiable Complex fun s => Inv.inv (Complex.Gamma (HDiv.hDiv s 2))
    -/
  · exact differentiable_one_div_Gamma.comp (differentiable_id.div_const _)
    /-
      🎉 no goals
    -/


lemma Gammaℝ_residue_zero : Tendsto (fun s ↦ s * Gammaℝ s) (𝓝[≠] 0) (𝓝 2) := by
  have h : Tendsto (fun z : ℂ ↦ z / 2 * Gamma (z / 2)) (𝓝[≠] 0) (𝓝 1) := by
    refine tendsto_self_mul_Gamma_nhds_zero.comp ?_
    rw [tendsto_nhdsWithin_iff, (by simp : 𝓝 (0 : ℂ) = 𝓝 (0 / 2))]
    exact ⟨(tendsto_id.div_const _).mono_left nhdsWithin_le_nhds,
      eventually_of_mem self_mem_nhdsWithin fun x hx ↦ div_ne_zero hx two_ne_zero⟩
  have h' : Tendsto (fun s : ℂ ↦ 2 * (π : ℂ) ^ (-s / 2)) (𝓝[≠] 0) (𝓝 2) := by
    rw [(by simp : 𝓝 2 = 𝓝 (2 * (π : ℂ) ^ (-(0 : ℂ) / 2)))]
    refine Tendsto.mono_left (ContinuousAt.tendsto ?_) nhdsWithin_le_nhds
    exact continuousAt_const.mul ((continuousAt_const_cpow (ofReal_ne_zero.mpr pi_ne_zero)).comp
      (continuousAt_id.neg.div_const _))
  /-
    h : Filter.Tendsto (fun z => HMul.hMul (HDiv.hDiv z 2) (Complex.Gamma (HDiv.hD …
    h' : Filter.Tendsto (fun s => HMul.hMul 2 (HPow.hPow (↑Real.pi) (HDiv.hDiv (Ne …
    ⊢ Filter.Tendsto (fun s => HMul.hMul s s.Gammaℝ) (nhdsWithin 0 (HasCompl.compl …
  -/
  convert mul_one (2 : ℂ) ▸ (h'.mul h) using 2 with z
  /-
    case h.e'_3.h
    h : Filter.Tendsto (fun z => HMul.hMul (HDiv.hDiv z 2) (Complex.Gamma (HDiv.hD …
    h' : Filter.Tendsto (fun s => HMul.hMul 2 (HPow.hPow (↑Real.pi) (HDiv.hDiv (Ne …
    z : Complex
    ⊢ Eq (HMul.hMul z z.Gammaℝ) (HMul.hMul (HMul.hMul 2 (HPow.hPow (↑Real.pi) (HDi …
  -/
  rw [Gammaℝ]
  /-
    case h.e'_3.h
    h : Filter.Tendsto (fun z => HMul.hMul (HDiv.hDiv z 2) (Complex.Gamma (HDiv.hD …
    h' : Filter.Tendsto (fun s => HMul.hMul 2 (HPow.hPow (↑Real.pi) (HDiv.hDiv (Ne …
    z : Complex
    ⊢ Eq (HMul.hMul z (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg z) 2))  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- Reformulation of the doubling formula in terms of `Gammaℝ`. -/
lemma Gammaℝ_mul_Gammaℝ_add_one (s : ℂ) : Gammaℝ s * Gammaℝ (s + 1) = Gammaℂ s := by
  /-
    s : Complex
    ⊢ Eq (HMul.hMul s.Gammaℝ (HAdd.hAdd s 1).Gammaℝ) s.Gammaℂ
  -/
  simp only [Gammaℝ_def, Gammaℂ_def]
  calc
  _ = (π ^ (-s / 2) * π ^ (-(s + 1) / 2)) * (Gamma (s / 2) * Gamma (s / 2 + 1 / 2)) := by ring_nf
  _ = 2 ^ (1 - s) * (π ^ (-1 / 2 - s) * π ^ (1 / 2 : ℂ)) * Gamma s := by
    rw [← cpow_add _ _ (ofReal_ne_zero.mpr pi_ne_zero), Complex.Gamma_mul_Gamma_add_half,
      sqrt_eq_rpow, ofReal_cpow pi_pos.le, ofReal_div, ofReal_one, ofReal_ofNat]
    ring_nf
  _ = 2 * ((2 : ℝ) ^ (-s) * π ^ (-s)) * Gamma s := by
    rw [sub_eq_add_neg, cpow_add _ _ two_ne_zero, cpow_one,
      ← cpow_add _ _ (ofReal_ne_zero.mpr pi_ne_zero), ofReal_ofNat]
    ring_nf
  _ = 2 * (2 * π) ^ (-s) * Gamma s := by
    rw [← mul_cpow_ofReal_nonneg two_pos.le pi_pos.le, ofReal_ofNat]


/-- Reformulation of the reflection formula in terms of `Gammaℝ`. -/
lemma Gammaℝ_one_sub_mul_Gammaℝ_one_add (s : ℂ) :
    Gammaℝ (1 - s) * Gammaℝ (1 + s) = (cos (π * s / 2))⁻¹ :=
  calc Gammaℝ (1 - s) * Gammaℝ (1 + s)
  _ = (π ^ ((s - 1) / 2) * π ^ ((-1 - s) / 2)) *
        (Gamma ((1 - s) / 2) * Gamma (1 - (1 - s) / 2)) := by
    /-
      s : Complex
      ⊢ Eq (HMul.hMul (HSub.hSub 1 s).Gammaℝ (HAdd.hAdd 1 s).Gammaℝ) (HMul.hMul (HMu …
    -/
    simp only [Gammaℝ_def]
    /-
      s : Complex
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg (HSub.hSu …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  _ = (π ^ ((s - 1) / 2) * π ^ ((-1 - s) / 2) * π ^ (1 : ℂ)) / sin (π / 2 - π * s / 2) := by
    /-
      s : Complex
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (HSub.hSub s 1) 2) …
    -/
    rw [Complex.Gamma_mul_Gamma_one_sub, cpow_one]
    /-
      s : Complex
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (HSub.hSub s 1) 2) …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  _ = _ := by
    simp_rw [← cpow_add _ _ (ofReal_ne_zero.mpr pi_ne_zero),
      Complex.sin_pi_div_two_sub]
    /-
      s : Complex
      ⊢ Eq (HDiv.hDiv (HPow.hPow (↑Real.pi) (HAdd.hAdd (HAdd.hAdd (HDiv.hDiv (HSub.h …
    -/
    ring_nf
    /-
      s : Complex
      ⊢ Eq (HMul.hMul (HPow.hPow (↑Real.pi) 0) (Inv.inv (Complex.cos (HMul.hMul (HMu …
    -/
    rw [cpow_zero, one_mul]
    /-
      🎉 no goals
    -/


/-- Another formulation of the reflection formula in terms of `Gammaℝ`. -/
lemma Gammaℝ_div_Gammaℝ_one_sub {s : ℂ} (hs : ∀ (n : ℕ), s ≠ -(2 * n + 1)) :
    Gammaℝ s / Gammaℝ (1 - s) = Gammaℂ s * cos (π * s / 2) := by
  have : Gammaℝ (s + 1) ≠ 0 := by
    simpa only [Ne, Gammaℝ_eq_zero_iff, not_exists, ← eq_sub_iff_add_eq,
      sub_eq_add_neg, ← neg_add]
  calc Gammaℝ s / Gammaℝ (1 - s)
  _ = (Gammaℝ s * Gammaℝ (s + 1)) / (Gammaℝ (1 - s) * Gammaℝ (1 + s)) := by
    rw [add_comm 1 s, mul_comm (Gammaℝ (1 - s)) (Gammaℝ (s + 1)), ← div_div,
      mul_div_cancel_right₀ _ this]
  _ = (2 * (2 * π) ^ (-s) * Gamma s) / ((cos (π * s / 2))⁻¹) := by
    rw [Gammaℝ_one_sub_mul_Gammaℝ_one_add, Gammaℝ_mul_Gammaℝ_add_one, Gammaℂ_def]
  _ = _ := by rw [Gammaℂ_def, div_eq_mul_inv, inv_inv]


/-- Formulation of reflection formula tailored to functional equations of L-functions of even
Dirichlet characters (including Riemann zeta). -/
lemma inv_Gammaℝ_one_sub {s : ℂ} (hs : ∀ (n : ℕ), s ≠ -n) :
    (Gammaℝ (1 - s))⁻¹ = Gammaℂ s * cos (π * s / 2) * (Gammaℝ s)⁻¹ := by
  have h1 : Gammaℝ s ≠ 0 := by
    rw [Ne, Gammaℝ_eq_zero_iff, not_exists]
    intro n h
    specialize hs (2 * n)
    simp_all
  have h2 : ∀ (n : ℕ), s ≠ -(2 * ↑n + 1) := by
    intro n h
    specialize hs (2 * n + 1)
    simp_all
  /-
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h1 : Ne s.Gammaℝ 0
    h2 : ∀ (n : Nat), Ne s (Neg.neg (HAdd.hAdd (HMul.hMul 2 ↑n) 1))
    ⊢ Eq (Inv.inv (HSub.hSub 1 s).Gammaℝ) (HMul.hMul (HMul.hMul s.Gammaℂ (Complex. …
  -/
  rw [← Gammaℝ_div_Gammaℝ_one_sub h2, ← div_eq_mul_inv, div_right_comm, div_self h1, one_div]
  /-
    🎉 no goals
  -/


/-- Formulation of reflection formula tailored to functional equations of L-functions of odd
Dirichlet characters. -/
lemma inv_Gammaℝ_two_sub {s : ℂ} (hs : ∀ (n : ℕ), s ≠ -n) :
    (Gammaℝ (2 - s))⁻¹ = Gammaℂ s * sin (π * s / 2) * (Gammaℝ (s + 1))⁻¹ := by
  /-
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    ⊢ Eq (Inv.inv (HSub.hSub 2 s).Gammaℝ) (HMul.hMul (HMul.hMul s.Gammaℂ (Complex. …
  -/
  by_cases h : s = 1
  · rw [h, (by ring : 2 - 1 = (1 : ℂ)), Gammaℝ_one, Gammaℝ,
    neg_div, (by norm_num : (1 + 1) / 2 = (1 : ℂ)), Complex.Gamma_one, Gammaℂ_one,
    mul_one, Complex.sin_pi_div_two, mul_one, cpow_neg_one, mul_one, inv_inv,
    div_mul_cancel₀ _ (ofReal_ne_zero.mpr pi_ne_zero), inv_one]
  /-
    case neg
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h : Not (Eq s 1)
    ⊢ Eq (Inv.inv (HSub.hSub 2 s).Gammaℝ) (HMul.hMul (HMul.hMul s.Gammaℂ (Complex. …
  -/
  rw [← Ne, ← sub_ne_zero] at h
  have h' (n : ℕ) : s - 1 ≠ -n := by
    cases' n with m
    · rwa [Nat.cast_zero, neg_zero]
    · rw [Ne, sub_eq_iff_eq_add]
      convert hs m using 2
      push_cast
      ring
  rw [(by ring : 2 - s = 1 - (s - 1)), inv_Gammaℝ_one_sub h',
    (by rw [sub_add_cancel] : Gammaℂ s = Gammaℂ (s - 1 + 1)), Gammaℂ_add_one h,
    (by ring : s + 1 = (s - 1) + 2), Gammaℝ_add_two h, mul_sub, sub_div, mul_one,
      Complex.cos_sub_pi_div_two]
  /-
    case neg
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h : Ne (HSub.hSub s 1) 0
    h' : ∀ (n : Nat), Ne (HSub.hSub s 1) (Neg.neg ↑n)
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub s 1).Gammaℂ (Complex.sin (HDiv.hDiv (HMu …
  -/
  simp_rw [mul_div_assoc, mul_inv]
  /-
    case neg
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h : Ne (HSub.hSub s 1) 0
    h' : ∀ (n : Nat), Ne (HSub.hSub s 1) (Neg.neg ↑n)
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub s 1).Gammaℂ (Complex.sin (HMul.hMul (↑Re …
  -/
  generalize (Gammaℝ (s - 1))⁻¹ = A
  /-
    case neg
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h : Ne (HSub.hSub s 1) 0
    h' : ∀ (n : Nat), Ne (HSub.hSub s 1) (Neg.neg ↑n)
    A : Complex
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub s 1).Gammaℂ (Complex.sin (HMul.hMul (↑Re …
  -/
  field_simp [pi_ne_zero]
  /-
    case neg
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    h : Ne (HSub.hSub s 1) 0
    h' : ∀ (n : Nat), Ne (HSub.hSub s 1) (Neg.neg ↑n)
    A : Complex
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HSub.hSub s 1).Gammaℂ (Complex.sin (HDi …
  -/
  ring
  /-
    🎉 no goals
  -/


