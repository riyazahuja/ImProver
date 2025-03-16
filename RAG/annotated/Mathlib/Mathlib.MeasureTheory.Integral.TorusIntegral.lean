local macro:arg t:term:max noWs "ⁿ⁺¹" : term => `(Fin (n + 1) → $t)

local macro:arg t:term:max noWs "ⁿ" : term => `(Fin n → $t)

local macro:arg t:term:max noWs "⁰" : term => `(Fin 0 → $t)

local macro:arg t:term:max noWs "¹" : term => `(Fin 1 → $t)


/-- The n dimensional exponential map $θ_i ↦ c + R e^{θ_i*I}, θ ∈ ℝⁿ$ representing
a torus in `ℂⁿ` with center `c ∈ ℂⁿ` and generalized radius `R ∈ ℝⁿ`, so we can adjust
it to every n axis. -/
def torusMap (c : ℂⁿ) (R : ℝⁿ) : ℝⁿ → ℂⁿ := fun θ i => c i + R i * exp (θ i * I)


theorem torusMap_sub_center (c : ℂⁿ) (R : ℝⁿ) (θ : ℝⁿ) : torusMap c R θ - c = torusMap 0 R θ := by
  /-
    n : Nat
    c : Fin n → Complex
    R θ : Fin n → Real
    ⊢ Eq (HSub.hSub (torusMap c R θ) c) (torusMap 0 R θ)
  -/
  ext1 i; simp [torusMap]
          /-
            🎉 no goals
          -/


theorem torusMap_eq_center_iff {c : ℂⁿ} {R : ℝⁿ} {θ : ℝⁿ} : torusMap c R θ = c ↔ R = 0 := by
  /-
    n : Nat
    c : Fin n → Complex
    R θ : Fin n → Real
    ⊢ Iff (Eq (torusMap c R θ) c) (Eq R 0)
  -/
  simp [funext_iff, torusMap, exp_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem torusMap_zero_radius (c : ℂⁿ) : torusMap c 0 = const ℝⁿ c :=
  funext fun _ ↦ torusMap_eq_center_iff.2 rfl


/-- A function `f : ℂⁿ → E` is integrable on the generalized torus if the function
`f ∘ torusMap c R θ` is integrable on `Icc (0 : ℝⁿ) (fun _ ↦ 2 * π)`. -/
def TorusIntegrable (f : ℂⁿ → E) (c : ℂⁿ) (R : ℝⁿ) : Prop :=
  IntegrableOn (fun θ : ℝⁿ => f (torusMap c R θ)) (Icc (0 : ℝⁿ) fun _ => 2 * π) volume


/-- Constant functions are torus integrable -/
theorem torusIntegrable_const (a : E) (c : ℂⁿ) (R : ℝⁿ) : TorusIntegrable (fun _ => a) c R := by
  /-
    n : Nat
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : E
    c : Fin n → Complex
    R : Fin n → Real
    ⊢ TorusIntegrable (fun x => a) c R
  -/
  simp [TorusIntegrable, measure_Icc_lt_top]
  /-
    🎉 no goals
  -/


/-- If `f` is torus integrable then `-f` is torus integrable. -/
protected nonrec theorem neg (hf : TorusIntegrable f c R) : TorusIntegrable (-f) c R := hf.neg


/-- If `f` and `g` are two torus integrable functions, then so is `f + g`. -/
protected nonrec theorem add (hf : TorusIntegrable f c R) (hg : TorusIntegrable g c R) :
    TorusIntegrable (f + g) c R :=
  hf.add hg


/-- If `f` and `g` are two torus integrable functions, then so is `f - g`. -/
protected nonrec theorem sub (hf : TorusIntegrable f c R) (hg : TorusIntegrable g c R) :
    TorusIntegrable (f - g) c R :=
  hf.sub hg


theorem torusIntegrable_zero_radius {f : ℂⁿ → E} {c : ℂⁿ} : TorusIntegrable f c 0 := by
  /-
    n : Nat
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : (Fin n → Complex) → E
    c : Fin n → Complex
    ⊢ TorusIntegrable f c 0
  -/
  rw [TorusIntegrable, torusMap_zero_radius]
  /-
    n : Nat
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : (Fin n → Complex) → E
    c : Fin n → Complex
    ⊢ MeasureTheory.IntegrableOn (fun θ => f (Function.const (Fin n → Real) c θ))  …
  -/
  apply torusIntegrable_const (f c) c 0
  /-
    🎉 no goals
  -/


/-- The function given in the definition of `torusIntegral` is integrable. -/
theorem function_integrable [NormedSpace ℂ E] (hf : TorusIntegrable f c R) :
    IntegrableOn (fun θ : ℝⁿ => (∏ i, R i * exp (θ i * I) * I : ℂ) • f (torusMap c R θ))
      (Icc (0 : ℝⁿ) fun _ => 2 * π) volume := by
  /-
    n : Nat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    f : (Fin n → Complex) → E
    c : Fin n → Complex
    R : Fin n → Real
    inst✝ : NormedSpace Complex E
    hf : TorusIntegrable f c R
    ⊢ MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (Finset.univ.prod fun i =>  …
  -/
  refine (hf.norm.const_mul (∏ i, |R i|)).mono' ?_ ?_
    /-
      case refine_1
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : (Fin n → Complex) → E
      c : Fin n → Complex
      R : Fin n → Real
      inst✝ : NormedSpace Complex E
      hf : TorusIntegrable f c R
      ⊢ MeasureTheory.AEStronglyMeasurable (fun θ => HSMul.hSMul (Finset.univ.prod f …
    -/
  · refine (Continuous.aestronglyMeasurable ?_).smul hf.1; fun_prop
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    case refine_2
    n : Nat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    f : (Fin n → Complex) → E
    c : Fin n → Complex
    R : Fin n → Real
    inst✝ : NormedSpace Complex E
    hf : TorusIntegrable f c R
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HSMul.hSMul (Finset.univ.prod  …
  -/
  simp [norm_smul, map_prod]
  /-
    🎉 no goals
  -/


/-- The integral over a generalized torus with center `c ∈ ℂⁿ` and radius `R ∈ ℝⁿ`, defined
as the `•`-product of the derivative of `torusMap` and `f (torusMap c R θ)`-/
def torusIntegral (f : ℂⁿ → E) (c : ℂⁿ) (R : ℝⁿ) :=
  ∫ θ : ℝⁿ in Icc (0 : ℝⁿ) fun _ => 2 * π, (∏ i, R i * exp (θ i * I) * I : ℂ) • f (torusMap c R θ)


@[inherit_doc torusIntegral]
notation3"∯ "(...)" in ""T("c", "R")"", "r:(scoped f => torusIntegral f c R) => r


theorem torusIntegral_radius_zero (hn : n ≠ 0) (f : ℂⁿ → E) (c : ℂⁿ) :
    (∯ x in T(c, 0), f x) = 0 := by
  simp only [torusIntegral, Pi.zero_apply, ofReal_zero, mul_zero, zero_mul, Fin.prod_const,
    zero_pow hn, zero_smul, integral_zero]


theorem torusIntegral_neg (f : ℂⁿ → E) (c : ℂⁿ) (R : ℝⁿ) :
                                                        /-
                                                          n : Nat
                                                          E : Type u_1
                                                          inst✝¹ : NormedAddCommGroup E
                                                          inst✝ : NormedSpace Complex E
                                                          f : (Fin n → Complex) → E
                                                          c : Fin n → Complex
                                                          R : Fin n → Real
                                                          ⊢ Eq (torusIntegral (fun x => Neg.neg (f x)) c R) (Neg.neg (torusIntegral (fun …
                                                        -/
    (∯ x in T(c, R), -f x) = -∯ x in T(c, R), f x := by simp [torusIntegral, integral_neg]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem torusIntegral_add (hf : TorusIntegrable f c R) (hg : TorusIntegrable g c R) :
    (∯ x in T(c, R), f x + g x) = (∯ x in T(c, R), f x) + ∯ x in T(c, R), g x := by
  simpa only [torusIntegral, smul_add, Pi.add_apply] using
    integral_add hf.function_integrable hg.function_integrable


theorem torusIntegral_sub (hf : TorusIntegrable f c R) (hg : TorusIntegrable g c R) :
    (∯ x in T(c, R), f x - g x) = (∯ x in T(c, R), f x) - ∯ x in T(c, R), g x := by
  /-
    n : Nat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : (Fin n → Complex) → E
    c : Fin n → Complex
    R : Fin n → Real
    hf : TorusIntegrable f c R
    hg : TorusIntegrable g c R
    ⊢ Eq (torusIntegral (fun x => HSub.hSub (f x) (g x)) c R) (HSub.hSub (torusInt …
  -/
  simpa only [sub_eq_add_neg, ← torusIntegral_neg] using torusIntegral_add hf hg.neg
  /-
    🎉 no goals
  -/


theorem torusIntegral_smul {𝕜 : Type*} [RCLike 𝕜] [NormedSpace 𝕜 E] [SMulCommClass 𝕜 ℂ E] (a : 𝕜)
    (f : ℂⁿ → E) (c : ℂⁿ) (R : ℝⁿ) : (∯ x in T(c, R), a • f x) = a • ∯ x in T(c, R), f x := by
  /-
    n : Nat
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : SMulCommClass 𝕜 Complex E
    a : 𝕜
    f : (Fin n → Complex) → E
    c : Fin n → Complex
    R : Fin n → Real
    ⊢ Eq (torusIntegral (fun x => HSMul.hSMul a (f x)) c R) (HSMul.hSMul a (torusI …
  -/
  simp only [torusIntegral, integral_smul, ← smul_comm a (_ : ℂ) (_ : E)]
  /-
    🎉 no goals
  -/


theorem torusIntegral_const_mul (a : ℂ) (f : ℂⁿ → ℂ) (c : ℂⁿ) (R : ℝⁿ) :
    (∯ x in T(c, R), a * f x) = a * ∯ x in T(c, R), f x :=
  torusIntegral_smul a f c R


/-- If for all `θ : ℝⁿ`, `‖f (torusMap c R θ)‖` is less than or equal to a constant `C : ℝ`, then
`‖∯ x in T(c, R), f x‖` is less than or equal to `(2 * π)^n * (∏ i, |R i|) * C`-/
theorem norm_torusIntegral_le_of_norm_le_const {C : ℝ} (hf : ∀ θ, ‖f (torusMap c R θ)‖ ≤ C) :
    ‖∯ x in T(c, R), f x‖ ≤ ((2 * π) ^ (n : ℕ) * ∏ i, |R i|) * C :=
  calc
    ‖∯ x in T(c, R), f x‖ ≤ (∏ i, |R i|) * C * (volume (Icc (0 : ℝⁿ) fun _ => 2 * π)).toReal :=
      norm_setIntegral_le_of_norm_le_const' measure_Icc_lt_top measurableSet_Icc fun θ _ =>
        calc
          ‖(∏ i : Fin n, R i * exp (θ i * I) * I : ℂ) • f (torusMap c R θ)‖ =
                                                                /-
                                                                  n : Nat
                                                                  E : Type u_1
                                                                  inst✝¹ : NormedAddCommGroup E
                                                                  inst✝ : NormedSpace Complex E
                                                                  f : (Fin n → Complex) → E
                                                                  c : Fin n → Complex
                                                                  R : Fin n → Real
                                                                  C : Real
                                                                  hf : ∀ (θ : Fin n → Real), LE.le (Norm.norm (f (torusMap c R θ))) C
                                                                  θ : Fin n → Real
                                                                  x✝ : Membership.mem (Set.Icc 0 fun x => HMul.hMul 2 Real.pi) θ
                                                                  ⊢ Eq (Norm.norm (HSMul.hSMul (Finset.univ.prod fun i => HMul.hMul (HMul.hMul ( …
                                                                -/
              (∏ i : Fin n, |R i|) * ‖f (torusMap c R θ)‖ := by simp [norm_smul]
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                                 /-
                                                                                   n : Nat
                                                                                   E : Type u_1
                                                                                   inst✝¹ : NormedAddCommGroup E
                                                                                   inst✝ : NormedSpace Complex E
                                                                                   f : (Fin n → Complex) → E
                                                                                   c : Fin n → Complex
                                                                                   R : Fin n → Real
                                                                                   C : Real
                                                                                   hf : ∀ (θ : Fin n → Real), LE.le (Norm.norm (f (torusMap c R θ))) C
                                                                                   θ : Fin n → Real
                                                                                   x✝ : Membership.mem (Set.Icc 0 fun x => HMul.hMul 2 Real.pi) θ
                                                                                   ⊢ LE.le 0 (Finset.univ.prod fun i => abs (R i))
                                                                                 -/
          _ ≤ (∏ i : Fin n, |R i|) * C := mul_le_mul_of_nonneg_left (hf _) <| by positivity
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    _ = ((2 * π) ^ (n : ℕ) * ∏ i, |R i|) * C := by
      simp only [Pi.zero_def, Real.volume_Icc_pi_toReal fun _ => Real.two_pi_pos.le, sub_zero,
        Fin.prod_const, mul_assoc, mul_comm ((2 * π) ^ (n : ℕ))]


@[simp]
theorem torusIntegral_dim0 [CompleteSpace E]
    (f : ℂ⁰ → E) (c : ℂ⁰) (R : ℝ⁰) : (∯ x in T(c, R), f x) = f c := by
  simp only [torusIntegral, Fin.prod_univ_zero, one_smul,
    Subsingleton.elim (fun _ : Fin 0 => 2 * π) 0, Icc_self, Measure.restrict_singleton, volume_pi,
    integral_smul_measure, integral_dirac, Measure.pi_of_empty (fun _ : Fin 0 ↦ volume) 0,
    Measure.dirac_apply_of_mem (mem_singleton _), Subsingleton.elim (torusMap c R 0) c]


/-- In dimension one, `torusIntegral` is the same as `circleIntegral`
(up to the natural equivalence between `ℂ` and `Fin 1 → ℂ`). -/
theorem torusIntegral_dim1 (f : ℂ¹ → E) (c : ℂ¹) (R : ℝ¹) :
    (∯ x in T(c, R), f x) = ∮ z in C(c 0, R 0), f fun _ => z := by
  have H₁ : (((MeasurableEquiv.funUnique _ _).symm) ⁻¹' Icc 0 fun _ => 2 * π) = Icc 0 (2 * π) :=
    (OrderIso.funUnique (Fin 1) ℝ).symm.preimage_Icc _ _
  have H₂ : torusMap c R = fun θ _ ↦ circleMap (c 0) (R 0) (θ 0) := by
    ext θ i : 2
    rw [Subsingleton.elim i 0]; rfl
  rw [torusIntegral, circleIntegral, intervalIntegral.integral_of_le Real.two_pi_pos.le,
    Measure.restrict_congr_set Ioc_ae_eq_Icc,
    ← ((volume_preserving_funUnique (Fin 1) ℝ).symm _).setIntegral_preimage_emb
      (MeasurableEquiv.measurableEmbedding _), H₁, H₂]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : (Fin 1 → Complex) → E
    c : Fin 1 → Complex
    R : Fin 1 → Real
    H₁ : Eq (Set.preimage (⇑(MeasurableEquiv.funUnique (Fin 1) Real).symm) (Set.Ic …
    H₂ : Eq (torusMap c R) fun θ x => circleMap (c 0) (R 0) (θ 0)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simp [circleMap_zero]
  /-
    🎉 no goals
  -/


/-- Recurrent formula for `torusIntegral`, see also `torusIntegral_succ`. -/
theorem torusIntegral_succAbove
    {f : ℂⁿ⁺¹ → E} {c : ℂⁿ⁺¹} {R : ℝⁿ⁺¹} (hf : TorusIntegrable f c R)
    (i : Fin (n + 1)) :
    (∯ x in T(c, R), f x) =
      ∮ x in C(c i, R i), ∯ y in T(c ∘ i.succAbove, R ∘ i.succAbove), f (i.insertNth x y) := by
  /-
    n : Nat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : (Fin (HAdd.hAdd n 1) → Complex) → E
    c : Fin (HAdd.hAdd n 1) → Complex
    R : Fin (HAdd.hAdd n 1) → Real
    hf : TorusIntegrable f c R
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (torusIntegral (fun x => f x) c R) (circleIntegral (fun x => torusIntegra …
  -/
  set e : ℝ × ℝⁿ ≃ᵐ ℝⁿ⁺¹ := (MeasurableEquiv.piFinSuccAbove (fun _ => ℝ) i).symm
  have hem : MeasurePreserving e :=
    (volume_preserving_piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) i).symm _
  have heπ : (e ⁻¹' Icc 0 fun _ => 2 * π) = Icc 0 (2 * π) ×ˢ Icc (0 : ℝⁿ) fun _ => 2 * π :=
    ((Fin.insertNthOrderIso (fun _ => ℝ) i).preimage_Icc _ _).trans (Icc_prod_eq _ _)
  rw [torusIntegral, ← hem.map_eq, setIntegral_map_equiv, heπ, Measure.volume_eq_prod,
    setIntegral_prod, circleIntegral_def_Icc]
    /-
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : (Fin (HAdd.hAdd n 1) → Complex) → E
      c : Fin (HAdd.hAdd n 1) → Complex
      R : Fin (HAdd.hAdd n 1) → Real
      hf : TorusIntegrable f c R
      i : Fin (HAdd.hAdd n 1)
      e : MeasurableEquiv (Prod Real (Fin n → Real)) (Fin (HAdd.hAdd n 1) → Real) := …
      hem : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume M …
      heπ : Eq (Set.preimage (⇑e) (Set.Icc 0 fun x => HMul.hMul 2 Real.pi)) (SProd.s …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
  · refine setIntegral_congr_fun measurableSet_Icc fun θ _ => ?_
    simp (config := { unfoldPartialApp := true }) only [e, torusIntegral, ← integral_smul,
      deriv_circleMap, i.prod_univ_succAbove _, smul_smul, torusMap, circleMap_zero]
    /-
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : (Fin (HAdd.hAdd n 1) → Complex) → E
      c : Fin (HAdd.hAdd n 1) → Complex
      R : Fin (HAdd.hAdd n 1) → Real
      hf : TorusIntegrable f c R
      i : Fin (HAdd.hAdd n 1)
      e : MeasurableEquiv (Prod Real (Fin n → Real)) (Fin (HAdd.hAdd n 1) → Real) := …
      hem : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume M …
      heπ : Eq (Set.preimage (⇑e) (Set.Icc 0 fun x => HMul.hMul 2 Real.pi)) (SProd.s …
      θ : Real
      x✝ : Membership.mem (Set.Icc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
    refine setIntegral_congr_fun measurableSet_Icc fun Θ _ => ?_
    simp only [MeasurableEquiv.piFinSuccAbove_symm_apply, i.insertNth_apply_same,
      i.insertNth_apply_succAbove, (· ∘ ·), Fin.insertNthEquiv, Equiv.coe_fn_mk]
    /-
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : (Fin (HAdd.hAdd n 1) → Complex) → E
      c : Fin (HAdd.hAdd n 1) → Complex
      R : Fin (HAdd.hAdd n 1) → Real
      hf : TorusIntegrable f c R
      i : Fin (HAdd.hAdd n 1)
      e : MeasurableEquiv (Prod Real (Fin n → Real)) (Fin (HAdd.hAdd n 1) → Real) := …
      hem : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume M …
      heπ : Eq (Set.preimage (⇑e) (Set.Icc 0 fun x => HMul.hMul 2 Real.pi)) (SProd.s …
      θ : Real
      x✝¹ : Membership.mem (Set.Icc 0 (HMul.hMul 2 Real.pi)) θ
      Θ : Fin n → Real
      x✝ : Membership.mem (Set.Icc 0 fun x => HMul.hMul 2 Real.pi) Θ
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.hMul (↑(R i)) (Complex.exp (HMul …
    -/
    congr 2
    simp only [funext_iff, i.forall_iff_succAbove, circleMap, Fin.insertNth_apply_same,
      eq_self_iff_true, Fin.insertNth_apply_succAbove, imp_true_iff, and_self_iff]
    /-
      case hf
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : (Fin (HAdd.hAdd n 1) → Complex) → E
      c : Fin (HAdd.hAdd n 1) → Complex
      R : Fin (HAdd.hAdd n 1) → Real
      hf : TorusIntegrable f c R
      i : Fin (HAdd.hAdd n 1)
      e : MeasurableEquiv (Prod Real (Fin n → Real)) (Fin (HAdd.hAdd n 1) → Real) := …
      hem : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume M …
      heπ : Eq (Set.preimage (⇑e) (Set.Icc 0 fun x => HMul.hMul 2 Real.pi)) (SProd.s …
      ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (Finset.univ.prod fun i =>  …
    -/
  · have := hf.function_integrable
    /-
      case hf
      n : Nat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : (Fin (HAdd.hAdd n 1) → Complex) → E
      c : Fin (HAdd.hAdd n 1) → Complex
      R : Fin (HAdd.hAdd n 1) → Real
      hf : TorusIntegrable f c R
      i : Fin (HAdd.hAdd n 1)
      e : MeasurableEquiv (Prod Real (Fin n → Real)) (Fin (HAdd.hAdd n 1) → Real) := …
      hem : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume M …
      heπ : Eq (Set.preimage (⇑e) (Set.Icc 0 fun x => HMul.hMul 2 Real.pi)) (SProd.s …
      this : MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (Finset.univ.prod fun  …
      ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (Finset.univ.prod fun i =>  …
    -/
    rwa [← hem.integrableOn_comp_preimage e.measurableEmbedding, heπ] at this
    /-
      🎉 no goals
    -/


/-- Recurrent formula for `torusIntegral`, see also `torusIntegral_succAbove`. -/
theorem torusIntegral_succ
    {f : ℂⁿ⁺¹ → E} {c : ℂⁿ⁺¹} {R : ℝⁿ⁺¹} (hf : TorusIntegrable f c R) :
    (∯ x in T(c, R), f x) =
      ∮ x in C(c 0, R 0), ∯ y in T(c ∘ Fin.succ, R ∘ Fin.succ), f (Fin.cons x y) := by
  /-
    n : Nat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : (Fin (HAdd.hAdd n 1) → Complex) → E
    c : Fin (HAdd.hAdd n 1) → Complex
    R : Fin (HAdd.hAdd n 1) → Real
    hf : TorusIntegrable f c R
    ⊢ Eq (torusIntegral (fun x => f x) c R) (circleIntegral (fun x => torusIntegra …
  -/
  simpa using torusIntegral_succAbove hf 0
  /-
    🎉 no goals
  -/

