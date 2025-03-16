/-- Taking a nonnegative power of a nonnegative number. This is defined as a standalone definition
in order to speed up automation such as `cfc_cont_tac`. -/
noncomputable abbrev nnrpow (a : ℝ≥0) (b : ℝ≥0) : ℝ≥0 := a ^ (b : ℝ)


@[simp] lemma nnrpow_def (a b : ℝ≥0) : nnrpow a b = a ^ (b : ℝ) := rfl


@[fun_prop]
lemma continuous_nnrpow_const (y : ℝ≥0) : Continuous (nnrpow · y) :=
  continuous_rpow_const zero_le_coe

/- This is a "redeclaration" of the attribute to speed up the proofs in this file. -/

/-- Real powers of operators, based on the non-unital continuous functional calculus. -/
noncomputable def nnrpow (a : A) (y : ℝ≥0) : A := cfcₙ (NNReal.nnrpow · y) a


/-- Enable `a ^ y` notation for `CFC.nnrpow`. This is a low-priority instance to make sure it does
not take priority over other instances when they are available. -/
noncomputable instance (priority := 100) : Pow A ℝ≥0 where
  pow a y := nnrpow a y


@[simp]
lemma nnrpow_eq_pow {a : A} {y : ℝ≥0} : nnrpow a y = a ^ y := rfl


@[simp]
lemma nnrpow_nonneg {a : A} {x : ℝ≥0} : 0 ≤ a ^ x := cfcₙ_predicate _ a


lemma nnrpow_def {a : A} {y : ℝ≥0} : a ^ y = cfcₙ (NNReal.nnrpow · y) a := rfl


lemma nnrpow_add {a : A} {x y : ℝ≥0} (hx : 0 < x) (hy : 0 < y) :
    a ^ (x + y) = a ^ x * a ^ y := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : NNReal
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Eq (HPow.hPow a (HAdd.hAdd x y)) (HMul.hMul (HPow.hPow a x) (HPow.hPow a y))
  -/
  simp only [nnrpow_def]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : NNReal
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Eq (cfcₙ (fun x_1 => x_1.nnrpow (HAdd.hAdd x y)) a) (HMul.hMul (cfcₙ (fun x_ …
  -/
  rw [← cfcₙ_mul _ _ a]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : NNReal
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Eq (cfcₙ (fun x_1 => x_1.nnrpow (HAdd.hAdd x y)) a) (cfcₙ (fun x_1 => HMul.h …
  -/
  congr! 2 with z
  /-
    case h.e'_17.h
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : NNReal
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    z : NNReal
    ⊢ Eq (z.nnrpow (HAdd.hAdd x y)) (HMul.hMul (z.nnrpow x) (z.nnrpow y))
  -/
  exact mod_cast z.rpow_add' <| ne_of_gt (add_pos hx hy)
  /-
    🎉 no goals
  -/


@[simp]
lemma nnrpow_zero {a : A} : a ^ (0 : ℝ≥0) = 0 := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ⊢ Eq (HPow.hPow a 0) 0
  -/
  simp [nnrpow_def, cfcₙ_apply_of_not_map_zero]
  /-
    🎉 no goals
  -/


lemma nnrpow_one (a : A) (ha : 0 ≤ a := by cfc_tac) : a ^ (1 : ℝ≥0) = a := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 1) a
  -/
  simp only [nnrpow_def, NNReal.nnrpow_def, NNReal.coe_one, NNReal.rpow_one]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ (fun x => x) a) a
  -/
  change cfcₙ (id : ℝ≥0 → ℝ≥0) a = a
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ id a) a
  -/
  rw [cfcₙ_id ℝ≥0 a]
  /-
    🎉 no goals
  -/


lemma nnrpow_two (a : A) (ha : 0 ≤ a := by cfc_tac) : a ^ (2 : ℝ≥0) = a * a := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 2) (HMul.hMul a a)
  -/
  simp only [nnrpow_def, NNReal.nnrpow_def, NNReal.coe_ofNat, NNReal.rpow_ofNat, pow_two]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ (fun x => HMul.hMul x x) a) (HMul.hMul a a)
  -/
  change cfcₙ (fun z : ℝ≥0 => id z * id z) a = a * a
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ (fun z => HMul.hMul (id z) (id z)) a) (HMul.hMul a a)
  -/
  rw [cfcₙ_mul id id a, cfcₙ_id ℝ≥0 a]
  /-
    🎉 no goals
  -/


lemma nnrpow_three (a : A) (ha : 0 ≤ a := by cfc_tac) : a ^ (3 : ℝ≥0) = a * a * a := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 3) (HMul.hMul (HMul.hMul a a) a)
  -/
  simp only [nnrpow_def, NNReal.nnrpow_def, NNReal.coe_ofNat, NNReal.rpow_ofNat, pow_three]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ (fun x => HMul.hMul x (HMul.hMul x x)) a) (HMul.hMul (HMul.hMul a a …
  -/
  change cfcₙ (fun z : ℝ≥0 => id z * (id z * id z)) a = a * a * a
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ (fun z => HMul.hMul (id z) (HMul.hMul (id z) (id z))) a) (HMul.hMul …
  -/
  rw [cfcₙ_mul id _ a, cfcₙ_mul id _ a, ← mul_assoc, cfcₙ_id ℝ≥0 a]
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      A : Type u_1
                                                      inst✝⁷ : PartialOrder A
                                                      inst✝⁶ : NonUnitalRing A
                                                      inst✝⁵ : TopologicalSpace A
                                                      inst✝⁴ : StarRing A
                                                      inst✝³ : Module NNReal A
                                                      inst✝² : SMulCommClass NNReal A A
                                                      inst✝¹ : IsScalarTower NNReal A A
                                                      inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                                                      x : NNReal
                                                      ⊢ Eq (HPow.hPow 0 x) 0
                                                    -/
lemma zero_nnrpow {x : ℝ≥0} : (0 : A) ^ x = 0 := by simp [nnrpow_def]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
lemma nnrpow_nnrpow [UniqueNonUnitalContinuousFunctionalCalculus ℝ≥0 A]
    {a : A} {x y : ℝ≥0} : (a ^ x) ^ y = a ^ (x * y) := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x y : NNReal
    ⊢ Eq (HPow.hPow (HPow.hPow a x) y) (HPow.hPow a (HMul.hMul x y))
  -/
  by_cases ha : 0 ≤ a
  case pos =>
    obtain (rfl | hx) := eq_zero_or_pos x <;> obtain (rfl | hy) := eq_zero_or_pos y
    all_goals try simp
    simp only [nnrpow_def, NNReal.coe_mul]
    rw [← cfcₙ_comp _ _ a]
    congr! 2 with u
    ext
    simp [Real.rpow_mul]
  case neg =>
    simp [nnrpow_def, cfcₙ_apply_of_not_predicate a ha]


lemma nnrpow_nnrpow_inv [UniqueNonUnitalContinuousFunctionalCalculus ℝ≥0 A]
    (a : A) {x : ℝ≥0} (hx : x ≠ 0) (ha : 0 ≤ a := by cfc_tac) : (a ^ x) ^ x⁻¹ = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    hx : Ne x 0
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (HPow.hPow a x) (Inv.inv x)) a
  -/
  simp [mul_inv_cancel₀ hx, nnrpow_one _ ha]
  /-
    🎉 no goals
  -/


lemma nnrpow_inv_nnrpow [UniqueNonUnitalContinuousFunctionalCalculus ℝ≥0 A]
    (a : A) {x : ℝ≥0} (hx : x ≠ 0) (ha : 0 ≤ a := by cfc_tac) : (a ^ x⁻¹) ^ x = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    hx : Ne x 0
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (HPow.hPow a (Inv.inv x)) x) a
  -/
  simp [inv_mul_cancel₀ hx, nnrpow_one _ ha]
  /-
    🎉 no goals
  -/


lemma nnrpow_inv_eq [UniqueNonUnitalContinuousFunctionalCalculus ℝ≥0 A]
    (a b : A) {x : ℝ≥0} (hx : x ≠ 0) (ha : 0 ≤ a := by cfc_tac) (hb : 0 ≤ b := by cfc_tac) :
    a ^ x⁻¹ = b ↔ b ^ x = a :=
           /-
             A : Type u_1
             inst✝⁸ : PartialOrder A
             inst✝⁷ : NonUnitalRing A
             inst✝⁶ : TopologicalSpace A
             inst✝⁵ : StarRing A
             inst✝⁴ : Module NNReal A
             inst✝³ : SMulCommClass NNReal A A
             inst✝² : IsScalarTower NNReal A A
             inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
             inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
             a b : A
             x : NNReal
             hx : Ne x 0
             ha : autoParam (LE.le 0 a) _auto✝
             hb : autoParam (LE.le 0 b) _auto✝
             h : Eq (HPow.hPow a (Inv.inv x)) b
             ⊢ LE.le 0 a
           -/
  ⟨fun h ↦ nnrpow_inv_nnrpow a hx ▸ congr($(h) ^ x).symm,
           /-
             🎉 no goals
           -/
            /-
              A : Type u_1
              inst✝⁸ : PartialOrder A
              inst✝⁷ : NonUnitalRing A
              inst✝⁶ : TopologicalSpace A
              inst✝⁵ : StarRing A
              inst✝⁴ : Module NNReal A
              inst✝³ : SMulCommClass NNReal A A
              inst✝² : IsScalarTower NNReal A A
              inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
              inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
              a b : A
              x : NNReal
              hx : Ne x 0
              ha : autoParam (LE.le 0 a) _auto✝
              hb : autoParam (LE.le 0 b) _auto✝
              h : Eq (HPow.hPow b x) a
              ⊢ LE.le 0 b
            -/
    fun h ↦ nnrpow_nnrpow_inv b hx ▸ congr($(h) ^ x⁻¹).symm⟩
            /-
              🎉 no goals
            -/

/- ## `sqrt` -/


/-- Square roots of operators, based on the non-unital continuous functional calculus. -/
noncomputable def sqrt (a : A) : A := cfcₙ NNReal.sqrt a


@[simp]
lemma sqrt_nonneg {a : A} : 0 ≤ sqrt a := cfcₙ_predicate _ a


lemma sqrt_eq_nnrpow {a : A} : sqrt a = a ^ (1 / 2 : ℝ≥0) := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ⊢ Eq (CFC.sqrt a) (HPow.hPow a (1 / 2))
  -/
  simp only [sqrt, nnrpow, NNReal.coe_inv, NNReal.coe_ofNat, NNReal.rpow_eq_pow]
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ⊢ Eq (cfcₙ (⇑NNReal.sqrt) a) (HPow.hPow a (1 / 2))
  -/
  congr
  /-
    case e_f
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ⊢ Eq ⇑NNReal.sqrt fun x => x.nnrpow (1 / 2)
  -/
  ext
  /-
    case e_f.h.a
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : StarRing A
    inst✝³ : Module NNReal A
    inst✝² : SMulCommClass NNReal A A
    inst✝¹ : IsScalarTower NNReal A A
    inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x✝ : NNReal
    ⊢ Eq ↑(NNReal.sqrt x✝) ↑(x✝.nnrpow (1 / 2))
  -/
  exact_mod_cast NNReal.sqrt_eq_rpow _
  /-
    🎉 no goals
  -/


@[simp]
                                         /-
                                           A : Type u_1
                                           inst✝⁷ : PartialOrder A
                                           inst✝⁶ : NonUnitalRing A
                                           inst✝⁵ : TopologicalSpace A
                                           inst✝⁴ : StarRing A
                                           inst✝³ : Module NNReal A
                                           inst✝² : SMulCommClass NNReal A A
                                           inst✝¹ : IsScalarTower NNReal A A
                                           inst✝ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                                           ⊢ Eq (CFC.sqrt 0) 0
                                         -/
lemma sqrt_zero : sqrt (0 : A) = 0 := by simp [sqrt]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
lemma nnrpow_sqrt {a : A} {x : ℝ≥0} : (sqrt a) ^ x = a ^ (x / 2) := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    ⊢ Eq (HPow.hPow (CFC.sqrt a) x) (HPow.hPow a (HDiv.hDiv x 2))
  -/
  rw [sqrt_eq_nnrpow, nnrpow_nnrpow, one_div_mul_eq_div 2 x]
  /-
    🎉 no goals
  -/


lemma nnrpow_sqrt_two (a : A) (ha : 0 ≤ a := by cfc_tac) : (sqrt a) ^ (2 : ℝ≥0) = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (CFC.sqrt a) 2) a
  -/
  simp only [nnrpow_sqrt, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, div_self]
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 1) a
  -/
  rw [nnrpow_one a]
  /-
    🎉 no goals
  -/


lemma sqrt_mul_sqrt_self (a : A) (ha : 0 ≤ a := by cfc_tac) : sqrt a * sqrt a = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HMul.hMul (CFC.sqrt a) (CFC.sqrt a)) a
  -/
  rw [← nnrpow_two _, nnrpow_sqrt_two _]
  /-
    🎉 no goals
  -/


@[simp]
lemma sqrt_nnrpow {a : A} {x : ℝ≥0} : sqrt (a ^ x) = a ^ (x / 2) := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    ⊢ Eq (CFC.sqrt (HPow.hPow a x)) (HPow.hPow a (HDiv.hDiv x 2))
  -/
  simp [sqrt_eq_nnrpow, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


lemma sqrt_nnrpow_two (a : A) (ha : 0 ≤ a := by cfc_tac) : sqrt (a ^ (2 : ℝ≥0)) = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (CFC.sqrt (HPow.hPow a 2)) a
  -/
  simp only [sqrt_nnrpow, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, div_self]
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 1) a
  -/
  rw [nnrpow_one _]
  /-
    🎉 no goals
  -/


lemma sqrt_mul_self (a : A) (ha : 0 ≤ a := by cfc_tac) : sqrt (a * a) = a := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (CFC.sqrt (HMul.hMul a a)) a
  -/
  rw [← nnrpow_two _, sqrt_nnrpow_two _]
  /-
    🎉 no goals
  -/


lemma mul_self_eq {a b : A} (h : sqrt a = b) (ha : 0 ≤ a := by cfc_tac) :
    b * b = a :=
  h ▸ sqrt_mul_sqrt_self _ ha


lemma sqrt_unique {a b : A} (h : b * b = a) (hb : 0 ≤ b := by cfc_tac) :
    sqrt a = b :=
      /-
        A : Type u_1
        inst✝⁸ : PartialOrder A
        inst✝⁷ : NonUnitalRing A
        inst✝⁶ : TopologicalSpace A
        inst✝⁵ : StarRing A
        inst✝⁴ : Module NNReal A
        inst✝³ : SMulCommClass NNReal A A
        inst✝² : IsScalarTower NNReal A A
        inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
        a b : A
        h : Eq (HMul.hMul b b) a
        hb : autoParam (LE.le 0 b) _auto✝
        ⊢ LE.le 0 b
      -/
  h ▸ sqrt_mul_self b
      /-
        🎉 no goals
      -/


lemma sqrt_eq_iff (a b : A) (ha : 0 ≤ a := by cfc_tac) (hb : 0 ≤ b := by cfc_tac) :
    sqrt a = b ↔ b * b = a :=
  ⟨(mul_self_eq ·), (sqrt_unique ·)⟩


lemma sqrt_eq_zero_iff (a : A) (ha : 0 ≤ a := by cfc_tac) : sqrt a = 0 ↔ a = 0 := by
  /-
    A : Type u_1
    inst✝⁸ : PartialOrder A
    inst✝⁷ : NonUnitalRing A
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module NNReal A
    inst✝³ : SMulCommClass NNReal A A
    inst✝² : IsScalarTower NNReal A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Iff (Eq (CFC.sqrt a) 0) (Eq a 0)
  -/
  rw [sqrt_eq_iff a _, mul_zero, eq_comm]
  /-
    🎉 no goals
  -/


/-- Real powers of operators, based on the unital continuous functional calculus. -/
noncomputable def rpow (a : A) (y : ℝ) : A := cfc (fun x : ℝ≥0 => x ^ y) a


/-- Enable `a ^ y` notation for `CFC.rpow`. This is a low-priority instance to make sure it does
not take priority over other instances when they are available (such as `Pow ℝ ℝ`). -/
noncomputable instance (priority := 100) : Pow A ℝ where
  pow a y := rpow a y


@[simp]
lemma rpow_eq_pow {a : A} {y : ℝ} : rpow a y = a ^ y := rfl


@[simp]
lemma rpow_nonneg {a : A} {y : ℝ} : 0 ≤ a ^ y := cfc_predicate _ a


lemma rpow_def {a : A} {y : ℝ} : a ^ y = cfc (fun x : ℝ≥0 => x ^ y) a := rfl


lemma rpow_one (a : A) (ha : 0 ≤ a := by cfc_tac) : a ^ (1 : ℝ) = a := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 1) a
  -/
  simp only [rpow_def, NNReal.coe_one, NNReal.rpow_eq_pow, NNReal.rpow_one, cfc_id' ℝ≥0 a]
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       A : Type u_1
                                                       inst✝⁵ : PartialOrder A
                                                       inst✝⁴ : Ring A
                                                       inst✝³ : StarRing A
                                                       inst✝² : TopologicalSpace A
                                                       inst✝¹ : Algebra Real A
                                                       inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                                                       x : Real
                                                       ⊢ Eq (HPow.hPow 1 x) 1
                                                     -/
lemma one_rpow {x : ℝ} : (1 : A) ^ x = (1 : A) := by simp [rpow_def]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma rpow_zero (a : A) (ha : 0 ≤ a := by cfc_tac) : a ^ (0 : ℝ) = 1 := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a 0) 1
  -/
  simp [rpow_def, cfc_const_one ℝ≥0 a]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  A : Type u_1
                                                                  inst✝⁵ : PartialOrder A
                                                                  inst✝⁴ : Ring A
                                                                  inst✝³ : StarRing A
                                                                  inst✝² : TopologicalSpace A
                                                                  inst✝¹ : Algebra Real A
                                                                  inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                                                                  x : Real
                                                                  hx : Ne x 0
                                                                  ⊢ Eq (CFC.rpow 0 x) 0
                                                                -/
lemma zero_rpow {x : ℝ} (hx : x ≠ 0) : rpow (0 : A) x = 0 := by simp [rpow, NNReal.zero_rpow hx]
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma rpow_natCast (a : A) (n : ℕ) (ha : 0 ≤ a := by cfc_tac) : a ^ (n : ℝ) = a ^ n := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    n : Nat
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow a ↑n) (HPow.hPow a n)
  -/
  rw [← cfc_pow_id (R := ℝ≥0) a n, rpow_def]
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    n : Nat
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow x ↑n) a) (cfc (fun x => HPow.hPow x n) a)
  -/
  congr
  /-
    case e_f
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    n : Nat
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (fun x => HPow.hPow x ↑n) fun x => HPow.hPow x n
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma rpow_algebraMap {x : ℝ≥0} {y : ℝ} :
    (algebraMap ℝ≥0 A x) ^ y = algebraMap ℝ≥0 A (x ^ y) := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    x : NNReal
    y : Real
    ⊢ Eq (HPow.hPow ((algebraMap NNReal A) x) y) ((algebraMap NNReal A) (HPow.hPow …
  -/
  rw [rpow_def, cfc_algebraMap ..]
  /-
    🎉 no goals
  -/


lemma rpow_add {a : A} {x y : ℝ} (ha : 0 ∉ spectrum ℝ≥0 a) :
    a ^ (x + y) = a ^ x * a ^ y := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ⊢ Eq (HPow.hPow a (HAdd.hAdd x y)) (HMul.hMul (HPow.hPow a x) (HPow.hPow a y))
  -/
  simp only [rpow_def, NNReal.rpow_eq_pow]
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ⊢ Eq (cfc (fun x_1 => HPow.hPow x_1 (HAdd.hAdd x y)) a) (HMul.hMul (cfc (fun x …
  -/
  rw [← cfc_mul _ _ a]
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ⊢ Eq (cfc (fun x_1 => HPow.hPow x_1 (HAdd.hAdd x y)) a) (cfc (fun x_1 => HMul. …
  -/
  refine cfc_congr ?_
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ⊢ Set.EqOn (fun x_1 => HPow.hPow x_1 (HAdd.hAdd x y)) (fun x_1 => HMul.hMul (H …
  -/
  intro z hz
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    z : NNReal
    hz : Membership.mem (spectrum NNReal a) z
    ⊢ Eq ((fun x_1 => HPow.hPow x_1 (HAdd.hAdd x y)) z) ((fun x_1 => HMul.hMul (HP …
  -/
  have : z ≠ 0 := by aesop
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x y : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    z : NNReal
    hz : Membership.mem (spectrum NNReal a) z
    this : Ne z 0
    ⊢ Eq ((fun x_1 => HPow.hPow x_1 (HAdd.hAdd x y)) z) ((fun x_1 => HMul.hMul (HP …
  -/
  simp [NNReal.rpow_add this _ _]
  /-
    🎉 no goals
  -/

-- TODO: relate to a strict positivity condition

lemma rpow_rpow [UniqueContinuousFunctionalCalculus ℝ≥0 A]
    (a : A) (x y : ℝ) (ha₁ : 0 ∉ spectrum ℝ≥0 a) (hx : x ≠ 0) (ha₂ : 0 ≤ a := by cfc_tac) :
    (a ^ x) ^ y = a ^ (x * y) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    ha₁ : Not (Membership.mem (spectrum NNReal a) 0)
    hx : Ne x 0
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (HPow.hPow a x) y) (HPow.hPow a (HMul.hMul x y))
  -/
  simp only [rpow_def]
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    ha₁ : Not (Membership.mem (spectrum NNReal a) 0)
    hx : Ne x 0
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow x y) (cfc (fun x_1 => HPow.hPow x_1 x) a)) (cfc  …
  -/
  rw [← cfc_comp _ _ a ha₂]
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    ha₁ : Not (Membership.mem (spectrum NNReal a) 0)
    hx : Ne x 0
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc (Function.comp (fun x => HPow.hPow x y) fun x_1 => HPow.hPow x_1 x)  …
  -/
  refine cfc_congr fun _ _ => ?_
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    ha₁ : Not (Membership.mem (spectrum NNReal a) 0)
    hx : Ne x 0
    ha₂ : autoParam (LE.le 0 a) _auto✝
    x✝¹ : NNReal
    x✝ : Membership.mem (spectrum NNReal a) x✝¹
    ⊢ Eq (Function.comp (fun x => HPow.hPow x y) (fun x_1 => HPow.hPow x_1 x) x✝¹) …
  -/
  simp [NNReal.rpow_mul]
  /-
    🎉 no goals
  -/


lemma rpow_rpow_of_exponent_nonneg [UniqueContinuousFunctionalCalculus ℝ≥0 A] (a : A) (x y : ℝ)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (ha₂ : 0 ≤ a := by cfc_tac) : (a ^ x) ^ y = a ^ (x * y) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (HPow.hPow a x) y) (HPow.hPow a (HMul.hMul x y))
  -/
  simp only [rpow_def]
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow x y) (cfc (fun x_1 => HPow.hPow x_1 x) a)) (cfc  …
  -/
  rw [← cfc_comp _ _ a]
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ha₂ : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc (Function.comp (fun x => HPow.hPow x y) fun x_1 => HPow.hPow x_1 x)  …
  -/
  refine cfc_congr fun _ _ => ?_
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ha₂ : autoParam (LE.le 0 a) _auto✝
    x✝¹ : NNReal
    x✝ : Membership.mem (spectrum NNReal a) x✝¹
    ⊢ Eq (Function.comp (fun x => HPow.hPow x y) (fun x_1 => HPow.hPow x_1 x) x✝¹) …
  -/
  simp [NNReal.rpow_mul]
  /-
    🎉 no goals
  -/


lemma rpow_mul_rpow_neg {a : A} (x : ℝ) (ha : 0 ∉ spectrum ℝ≥0 a)
    (ha' : 0 ≤ a := by cfc_tac) : a ^ x * a ^ (-x) = 1 := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ha' : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HMul.hMul (HPow.hPow a x) (HPow.hPow a (Neg.neg x))) 1
  -/
  rw [← rpow_add ha, add_neg_cancel, rpow_zero a]
  /-
    🎉 no goals
  -/


lemma rpow_neg_mul_rpow {a : A} (x : ℝ) (ha : 0 ∉ spectrum ℝ≥0 a)
    (ha' : 0 ≤ a := by cfc_tac) : a ^ (-x) * a ^ x = 1 := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : A
    x : Real
    ha : Not (Membership.mem (spectrum NNReal a) 0)
    ha' : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HMul.hMul (HPow.hPow a (Neg.neg x)) (HPow.hPow a x)) 1
  -/
  rw [← rpow_add ha, neg_add_cancel, rpow_zero a]
  /-
    🎉 no goals
  -/


lemma rpow_neg_one_eq_inv (a : Aˣ) (ha : (0 : A) ≤ a := by cfc_tac) :
    a ^ (-1 : ℝ) = (↑a⁻¹ : A) := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    ha : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ Eq (HPow.hPow (↑a) (-1)) ↑(Inv.inv a)
  -/
  refine a.inv_eq_of_mul_eq_one_left ?_ |>.symm
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    ha : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ Eq (HMul.hMul (HPow.hPow (↑a) (-1)) ↑a) 1
  -/
  simpa [rpow_one (a : A)] using rpow_neg_mul_rpow 1 (spectrum.zero_not_mem ℝ≥0 a.isUnit)
  /-
    🎉 no goals
  -/


lemma rpow_neg_one_eq_cfc_inv {A : Type*} [PartialOrder A] [NormedRing A] [StarRing A]
    [NormedAlgebra ℝ A] [ContinuousFunctionalCalculus ℝ≥0 ((0 : A) ≤ ·)] (a : A) :
    a ^ (-1 : ℝ) = cfc (·⁻¹ : ℝ≥0 → ℝ≥0) a :=
  cfc_congr fun x _ ↦ NNReal.rpow_neg_one x


lemma rpow_neg [UniqueContinuousFunctionalCalculus ℝ≥0 A] (a : Aˣ) (x : ℝ)
    (ha' : (0 : A) ≤ a := by cfc_tac) : (a : A) ^ (-x) = (↑a⁻¹ : A) ^ x := by
  suffices h₁ : ContinuousOn (fun z ↦ z ^ x) (Inv.inv '' (spectrum ℝ≥0 (a : A))) by
    rw [← cfc_inv_id (R := ℝ≥0) a, rpow_def, rpow_def,
        ← cfc_comp' (fun z => z ^ x) (Inv.inv : ℝ≥0 → ℝ≥0) (a : A) h₁]
    refine cfc_congr fun _ _ => ?_
    simp [NNReal.rpow_neg, NNReal.inv_rpow]
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : Units A
    x : Real
    ha' : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ ContinuousOn (fun z => HPow.hPow z x) (Set.image Inv.inv (spectrum NNReal ↑a))
  -/
  refine NNReal.continuousOn_rpow_const (.inl ?_)
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : Units A
    x : Real
    ha' : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ Not (Membership.mem (Set.image Inv.inv (spectrum NNReal ↑a)) 0)
  -/
  rintro ⟨z, hz, hz'⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : Units A
    x : Real
    ha' : autoParam (LE.le 0 ↑a) _auto✝
    z : NNReal
    hz : Membership.mem (spectrum NNReal ↑a) z
    hz' : Eq (Inv.inv z) 0
    ⊢ False
  -/
  exact spectrum.zero_not_mem ℝ≥0 a.isUnit <| inv_eq_zero.mp hz' ▸ hz
  /-
    🎉 no goals
  -/


lemma rpow_intCast (a : Aˣ) (n : ℤ) (ha : (0 : A) ≤ a := by cfc_tac) :
    (a : A) ^ (n : ℝ) = (↑(a ^ n) : A) := by
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    n : Int
    ha : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ Eq (HPow.hPow ↑a ↑n) ↑(HPow.hPow a n)
  -/
  rw [← cfc_zpow (R := ℝ≥0) a n, rpow_def]
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    n : Int
    ha : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow x ↑n) ↑a) (cfc (fun x => HPow.hPow x n) ↑a)
  -/
  refine cfc_congr fun _ _ => ?_
  /-
    A : Type u_1
    inst✝⁵ : PartialOrder A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    n : Int
    ha : autoParam (LE.le 0 ↑a) _auto✝
    x✝¹ : NNReal
    x✝ : Membership.mem (spectrum NNReal ↑a) x✝¹
    ⊢ Eq (HPow.hPow x✝¹ ↑n) (HPow.hPow x✝¹ n)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma nnrpow_eq_rpow {a : A} {x : ℝ≥0} (hx : 0 < x) : a ^ x = a ^ (x : ℝ) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    hx : LT.lt 0 x
    ⊢ Eq (HPow.hPow a x) (HPow.hPow a ↑x)
  -/
  rw [nnrpow_def (A := A), rpow_def, cfcₙ_eq_cfc]
  /-
    🎉 no goals
  -/


lemma sqrt_eq_rpow {a : A} : sqrt a = a ^ (1 / 2 : ℝ) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ⊢ Eq (CFC.sqrt a) (HPow.hPow a (1 / 2))
  -/
  have : a ^ (1 / 2 : ℝ) = a ^ ((1 / 2 : ℝ≥0) : ℝ) := rfl
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    this : Eq (HPow.hPow a (1 / 2)) (HPow.hPow a ↑(1 / 2))
    ⊢ Eq (CFC.sqrt a) (HPow.hPow a (1 / 2))
  -/
  rw [this, ← nnrpow_eq_rpow (by norm_num), sqrt_eq_nnrpow (A := A)]
  /-
    🎉 no goals
  -/


lemma sqrt_eq_cfc {a : A} : sqrt a = cfc NNReal.sqrt a := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ⊢ Eq (CFC.sqrt a) (cfc (⇑NNReal.sqrt) a)
  -/
  unfold sqrt
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ⊢ Eq (cfcₙ (⇑NNReal.sqrt) a) (cfc (⇑NNReal.sqrt) a)
  -/
  rw [cfcₙ_eq_cfc]
  /-
    🎉 no goals
  -/


lemma sqrt_sq (a : A) (ha : 0 ≤ a := by cfc_tac) : sqrt (a ^ 2) = a := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (CFC.sqrt (HPow.hPow a 2)) a
  -/
  rw [pow_two, sqrt_mul_self (A := A) a]
  /-
    🎉 no goals
  -/


lemma sq_sqrt (a : A) (ha : 0 ≤ a := by cfc_tac) : (sqrt a) ^ 2 = a := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (CFC.sqrt a) 2) a
  -/
  rw [pow_two, sqrt_mul_sqrt_self (A := A) a]
  /-
    🎉 no goals
  -/


@[simp]
lemma sqrt_algebraMap {r : ℝ≥0} : sqrt (algebraMap ℝ≥0 A r) = algebraMap ℝ≥0 A (NNReal.sqrt r) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    r : NNReal
    ⊢ Eq (CFC.sqrt ((algebraMap NNReal A) r)) ((algebraMap NNReal A) (NNReal.sqrt  …
  -/
  rw [sqrt_eq_cfc, cfc_algebraMap]
  /-
    🎉 no goals
  -/


@[simp]
                                        /-
                                          A : Type u_1
                                          inst✝⁶ : PartialOrder A
                                          inst✝⁵ : Ring A
                                          inst✝⁴ : StarRing A
                                          inst✝³ : TopologicalSpace A
                                          inst✝² : Algebra Real A
                                          inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                                          inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
                                          ⊢ Eq (CFC.sqrt 1) 1
                                        -/
lemma sqrt_one : sqrt (1 : A) = 1 := by simp [sqrt_eq_cfc]
                                        /-
                                          🎉 no goals
                                        -/

-- TODO: relate to a strict positivity condition

lemma sqrt_rpow [UniqueContinuousFunctionalCalculus ℝ≥0 A] {a : A} {x : ℝ} (h : 0 ∉ spectrum ℝ≥0 a)
    (hx : x ≠ 0) : sqrt (a ^ x) = a ^ (x / 2) := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : Ring A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Algebra Real A
    inst✝² : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x : Real
    h : Not (Membership.mem (spectrum NNReal a) 0)
    hx : Ne x 0
    ⊢ Eq (CFC.sqrt (HPow.hPow a x)) (HPow.hPow a (HDiv.hDiv x 2))
  -/
  by_cases hnonneg : 0 ≤ a
  case pos =>
    simp only [sqrt_eq_rpow, div_eq_mul_inv, one_mul, rpow_rpow _ _ _ h hx]
  case neg =>
    simp [sqrt_eq_cfc, rpow_def, cfc_apply_of_not_predicate a hnonneg]

-- TODO: relate to a strict positivity condition

lemma rpow_sqrt [UniqueContinuousFunctionalCalculus ℝ≥0 A] (a : A) (x : ℝ) (h : 0 ∉ spectrum ℝ≥0 a)
    (ha : 0 ≤ a := by cfc_tac) : (sqrt a) ^ x = a ^ (x / 2) := by
  rw [sqrt_eq_rpow, div_eq_mul_inv, one_mul,
      rpow_rpow _ _ _ h (by norm_num), inv_mul_eq_div]


lemma sqrt_rpow_nnreal {a : A} {x : ℝ≥0} : sqrt (a ^ (x : ℝ)) = a ^ (x / 2 : ℝ) := by
  /-
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    ⊢ Eq (CFC.sqrt (HPow.hPow a ↑x)) (HPow.hPow a (HDiv.hDiv (↑x) 2))
  -/
  by_cases htriv : 0 ≤ a
  /-
    case pos
    A : Type u_1
    inst✝⁶ : PartialOrder A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    htriv : LE.le 0 a
    ⊢ Eq (CFC.sqrt (HPow.hPow a ↑x)) (HPow.hPow a (HDiv.hDiv (↑x) 2))
  -/
  case neg => simp [sqrt_eq_cfc, rpow_def, cfc_apply_of_not_predicate a htriv]
  case pos =>
    by_cases hx : x = 0
    case pos => simp [hx, rpow_zero _ htriv]
    case neg =>
      have h₁ : 0 < x := lt_of_le_of_ne (by aesop) (Ne.symm hx)
      have h₂ : (x : ℝ) / 2 = NNReal.toReal (x / 2) := rfl
      have h₃ : 0 < x / 2 := by positivity
      rw [← nnrpow_eq_rpow h₁, h₂, ← nnrpow_eq_rpow h₃, sqrt_nnrpow (A := A)]


lemma rpow_sqrt_nnreal [UniqueContinuousFunctionalCalculus ℝ≥0 A] {a : A} {x : ℝ≥0}
    (ha : 0 ≤ a := by cfc_tac) : (sqrt a) ^ (x : ℝ) = a ^ (x / 2 : ℝ) := by
  /-
    A : Type u_1
    inst✝⁷ : PartialOrder A
    inst✝⁶ : Ring A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Algebra Real A
    inst✝² : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus NNReal A
    inst✝ : UniqueContinuousFunctionalCalculus NNReal A
    a : A
    x : NNReal
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (HPow.hPow (CFC.sqrt a) ↑x) (HPow.hPow a (HDiv.hDiv (↑x) 2))
  -/
  by_cases hx : x = 0
  case pos =>
    have ha' : 0 ≤ sqrt a := by exact sqrt_nonneg
    simp [hx, rpow_zero _ ha', rpow_zero _ ha]
  case neg =>
    have h₁ : 0 ≤ (x : ℝ) := by exact NNReal.zero_le_coe
    rw [sqrt_eq_rpow, rpow_rpow_of_exponent_nonneg _ _ _ (by norm_num) h₁, one_div_mul_eq_div]


