/-- An affine map is continuous iff its underlying linear map is continuous. See also
`AffineMap.continuous_linear_iff`. -/
theorem continuous_iff {f : E →ᵃ[R] F} : Continuous f ↔ Continuous f.linear := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Ring R
    inst✝¹ : Module R E
    inst✝ : Module R F
    f : AffineMap R E F
    ⊢ Iff (Continuous ⇑f) (Continuous ⇑f.linear)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      ⊢ Continuous ⇑f → Continuous ⇑f.linear
    -/
  · intro hc
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      hc : Continuous ⇑f
      ⊢ Continuous ⇑f.linear
    -/
    rw [decomp' f]
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      hc : Continuous ⇑f
      ⊢ Continuous (HSub.hSub ⇑f fun x => f 0)
    -/
    exact hc.sub continuous_const
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      ⊢ Continuous ⇑f.linear → Continuous ⇑f
    -/
  · intro hc
    /-
      case mpr
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      hc : Continuous ⇑f.linear
      ⊢ Continuous ⇑f
    -/
    rw [decomp f]
    /-
      case mpr
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Ring R
      inst✝¹ : Module R E
      inst✝ : Module R F
      f : AffineMap R E F
      hc : Continuous ⇑f.linear
      ⊢ Continuous (HAdd.hAdd ⇑f.linear fun x => f 0)
    -/
    exact hc.add continuous_const
    /-
      🎉 no goals
    -/


/-- The line map is continuous. -/
@[continuity]
theorem lineMap_continuous [TopologicalSpace R] [ContinuousSMul R F] {p v : F} :
    Continuous (lineMap p v : R →ᵃ[R] F) :=
  continuous_iff.mpr <|
    (continuous_id.smul continuous_const).add <| @continuous_const _ _ _ _ (0 : F)


@[continuity]
theorem homothety_continuous (x : F) (t : R) : Continuous <| homothety x t := by
  suffices ⇑(homothety x t) = fun y => t • (y - x) + x by
    rw [this]
    exact ((continuous_id.sub continuous_const).const_smul _).add continuous_const
    -- Porting note: proof was `by continuity`
  /-
    R : Type u_1
    F : Type u_3
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : CommRing R
    inst✝¹ : Module R F
    inst✝ : ContinuousConstSMul R F
    x : F
    t : R
    ⊢ Eq ⇑(AffineMap.homothety x t) fun y => HAdd.hAdd (HSMul.hSMul t (HSub.hSub y …
  -/
  ext y
  /-
    case h
    R : Type u_1
    F : Type u_3
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : CommRing R
    inst✝¹ : Module R F
    inst✝ : ContinuousConstSMul R F
    x : F
    t : R
    y : F
    ⊢ Eq ((AffineMap.homothety x t) y) (HAdd.hAdd (HSMul.hSMul t (HSub.hSub y x)) x)
  -/
  simp [homothety_apply]
  /-
    🎉 no goals
  -/


theorem homothety_isOpenMap (x : F) (t : R) (ht : t ≠ 0) : IsOpenMap <| homothety x t := by
  /-
    R : Type u_1
    F : Type u_3
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Field R
    inst✝¹ : Module R F
    inst✝ : ContinuousConstSMul R F
    x : F
    t : R
    ht : Ne t 0
    ⊢ IsOpenMap ⇑(AffineMap.homothety x t)
  -/
  apply IsOpenMap.of_inverse (homothety_continuous x t⁻¹) <;> intro e <;>
    /-
      case l_inv
      R : Type u_1
      F : Type u_3
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup F
      inst✝² : Field R
      inst✝¹ : Module R F
      inst✝ : ContinuousConstSMul R F
      x : F
      t : R
      ht : Ne t 0
      e : F
      ⊢ Eq ((AffineMap.homothety x t) ((AffineMap.homothety x (Inv.inv t)) e)) e
    -/
    /-
      🎉 no goals
    -/
    simp [← AffineMap.comp_apply, ← homothety_mul, ht]
    /-
      🎉 no goals
    -/


