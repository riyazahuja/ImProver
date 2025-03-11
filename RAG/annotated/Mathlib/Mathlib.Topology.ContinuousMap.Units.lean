/-- Equivalence between continuous maps into the units of a monoid with continuous multiplication
and the units of the monoid of continuous maps. -/
-- `simps` generates some lemmas here with LHS not in simp normal form,
-- so we write them out manually below.
-- https://github.com/leanprover-community/mathlib4/issues/18942
@[to_additive (attr := simps apply_val_apply symm_apply_apply_val)
"Equivalence between continuous maps into the additive units of an additive monoid with continuous
addition and the additive units of the additive monoid of continuous maps."]
def unitsLift : C(X, Mˣ) ≃ C(X, M)ˣ where
  toFun f :=
    { val := ⟨fun x => f x, Units.continuous_val.comp f.continuous⟩
      inv := ⟨fun x => ↑(f x)⁻¹, Units.continuous_val.comp (continuous_inv.comp f.continuous)⟩
      val_inv := ext fun _ => Units.mul_inv _
      inv_val := ext fun _ => Units.inv_mul _ }
  invFun f :=
    { toFun := fun x =>
        ⟨(f : C(X, M)) x, (↑f⁻¹ : C(X, M)) x,
          ContinuousMap.congr_fun f.mul_inv x, ContinuousMap.congr_fun f.inv_mul x⟩
      continuous_toFun := continuous_induced_rng.2 <|
        (f : C(X, M)).continuous.prod_mk <|
        MulOpposite.continuous_op.comp (↑f⁻¹ : C(X, M)).continuous }
                   /-
                     X : Type u_1
                     M : Type u_2
                     R : Type u_3
                     𝕜 : Type u_4
                     inst✝³ : TopologicalSpace X
                     inst✝² : Monoid M
                     inst✝¹ : TopologicalSpace M
                     inst✝ : ContinuousMul M
                     f : ContinuousMap X (Units M)
                     ⊢ Eq ((fun f => { toFun := fun x => { val := ↑f x, inv := ↑(Inv.inv f) x, val_ …
                   -/
  left_inv f := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      X : Type u_1
                      M : Type u_2
                      R : Type u_3
                      𝕜 : Type u_4
                      inst✝³ : TopologicalSpace X
                      inst✝² : Monoid M
                      inst✝¹ : TopologicalSpace M
                      inst✝ : ContinuousMul M
                      f : Units (ContinuousMap X M)
                      ⊢ Eq ((fun f => { val := { toFun := fun x => ↑(f x), continuous_toFun := ⋯ },  …
                    -/
  right_inv f := by ext; rfl
                         /-
                           🎉 no goals
                         -/


@[to_additive (attr := simp)]
lemma unitsLift_apply_inv_apply (f : C(X, Mˣ)) (x : X) :
    (↑(ContinuousMap.unitsLift f)⁻¹ : C(X, M)) x = (f x)⁻¹ :=
  rfl


@[to_additive (attr := simp)]
lemma unitsLift_symm_apply_apply_inv' (f : C(X, M)ˣ) (x : X) :
    (ContinuousMap.unitsLift.symm f x)⁻¹ = (↑f⁻¹ : C(X, M)) x := by
  /-
    X : Type u_1
    M : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    f : Units (ContinuousMap X M)
    x : X
    ⊢ Eq (↑(Inv.inv ((ContinuousMap.unitsLift.symm f) x))) (↑(Inv.inv f) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem continuous_isUnit_unit {f : C(X, R)} (h : ∀ x, IsUnit (f x)) :
    Continuous fun x => (h x).unit := by
  refine
    continuous_induced_rng.2
      (Continuous.prod_mk f.continuous
        (MulOpposite.continuous_op.comp (continuous_iff_continuousAt.mpr fun x => ?_)))
  /-
    X : Type u_1
    R : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    h : ∀ (x : X), IsUnit (f x)
    x : X
    ⊢ ContinuousAt (fun x => ↑(Inv.inv ((fun x => ⋯.unit) x))) x
  -/
  have := NormedRing.inverse_continuousAt (h x).unit
  /-
    X : Type u_1
    R : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    h : ∀ (x : X), IsUnit (f x)
    x : X
    this : ContinuousAt Ring.inverse ↑⋯.unit
    ⊢ ContinuousAt (fun x => ↑(Inv.inv ((fun x => ⋯.unit) x))) x
  -/
  simp only
  /-
    X : Type u_1
    R : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    h : ∀ (x : X), IsUnit (f x)
    x : X
    this : ContinuousAt Ring.inverse ↑⋯.unit
    ⊢ ContinuousAt (fun x => ↑(Inv.inv ⋯.unit)) x
  -/
  simp only [← Ring.inverse_unit, IsUnit.unit_spec] at this ⊢
  /-
    X : Type u_1
    R : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    h : ∀ (x : X), IsUnit (f x)
    x : X
    this : ContinuousAt Ring.inverse (f x)
    ⊢ ContinuousAt (fun x => Ring.inverse (f x)) x
  -/
  exact this.comp (f.continuousAt x)
  /-
    🎉 no goals
  -/
-- Porting note: this had the worst namespace: `NormedRing`


/-- Construct a continuous map into the group of units of a normed ring from a function into the
normed ring and a proof that every element of the range is a unit. -/
@[simps]
noncomputable def unitsOfForallIsUnit {f : C(X, R)} (h : ∀ x, IsUnit (f x)) : C(X, Rˣ) where
  toFun x := (h x).unit
  continuous_toFun := continuous_isUnit_unit h


instance canLift :
    CanLift C(X, R) C(X, Rˣ) (fun f => ⟨fun x => f x, Units.continuous_val.comp f.continuous⟩)
      fun f => ∀ x, IsUnit (f x) where
                                        /-
                                          X : Type u_1
                                          M : Type u_2
                                          R : Type u_3
                                          𝕜 : Type u_4
                                          inst✝² : TopologicalSpace X
                                          inst✝¹ : NormedRing R
                                          inst✝ : CompleteSpace R
                                          f : ContinuousMap X R
                                          h : ∀ (x : X), IsUnit (f x)
                                          ⊢ Eq { toFun := fun x => ↑((ContinuousMap.unitsOfForallIsUnit h) x), continuou …
                                        -/
  prf f h := ⟨unitsOfForallIsUnit h, by ext; rfl⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem isUnit_iff_forall_isUnit (f : C(X, R)) : IsUnit f ↔ ∀ x, IsUnit (f x) :=
  Iff.intro (fun h => fun x => ⟨unitsLift.symm h.unit x, rfl⟩) fun h =>
                                                         /-
                                                           X : Type u_1
                                                           R : Type u_3
                                                           inst✝² : TopologicalSpace X
                                                           inst✝¹ : NormedRing R
                                                           inst✝ : CompleteSpace R
                                                           f : ContinuousMap X R
                                                           h : ∀ (x : X), IsUnit (f x)
                                                           ⊢ Eq (↑(ContinuousMap.unitsLift (ContinuousMap.unitsOfForallIsUnit h))) f
                                                         -/
    ⟨ContinuousMap.unitsLift (unitsOfForallIsUnit h), by ext; rfl⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem isUnit_iff_forall_ne_zero (f : C(X, R)) : IsUnit f ↔ ∀ x, f x ≠ 0 := by
  /-
    X : Type u_1
    R : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedDivisionRing R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    ⊢ Iff (IsUnit f) (∀ (x : X), Ne (f x) 0)
  -/
  simp_rw [f.isUnit_iff_forall_isUnit, isUnit_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem spectrum_eq_preimage_range (f : C(X, R)) :
    spectrum 𝕜 f = algebraMap _ _ ⁻¹' Set.range f := by
  /-
    X : Type u_1
    R : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedField 𝕜
    inst✝² : NormedDivisionRing R
    inst✝¹ : Algebra 𝕜 R
    inst✝ : CompleteSpace R
    f : ContinuousMap X R
    ⊢ Eq (spectrum 𝕜 f) (Set.preimage (⇑(algebraMap 𝕜 R)) (Set.range ⇑f))
  -/
  ext x
  simp only [spectrum.mem_iff, isUnit_iff_forall_ne_zero, not_forall, sub_apply,
    algebraMap_apply, mul_one, Classical.not_not, Set.mem_range,
    sub_eq_zero, @eq_comm _ (x • 1 : R) _, Set.mem_preimage, Algebra.algebraMap_eq_smul_one,
    smul_apply, one_apply]


theorem spectrum_eq_range [CompleteSpace 𝕜] (f : C(X, 𝕜)) : spectrum 𝕜 f = Set.range f := by
  /-
    X : Type u_1
    𝕜 : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    f : ContinuousMap X 𝕜
    ⊢ Eq (spectrum 𝕜 f) (Set.range ⇑f)
  -/
  rw [spectrum_eq_preimage_range, Algebra.id.map_eq_id]
  /-
    X : Type u_1
    𝕜 : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    f : ContinuousMap X 𝕜
    ⊢ Eq (Set.preimage (⇑(RingHom.id 𝕜)) (Set.range ⇑f)) (Set.range ⇑f)
  -/
  exact Set.preimage_id
  /-
    🎉 no goals
  -/


