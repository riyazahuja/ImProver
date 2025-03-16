lemma FiniteDimensional.of_fact_finrank_eq_two {K V : Type*} [DivisionRing K]
    [AddCommGroup V] [Module K V] [Fact (finrank K V = 2)] : FiniteDimensional K V :=
  .of_fact_finrank_eq_succ 1


@[deprecated (since := "2024-02-02")]
alias FiniteDimensional.finiteDimensional_of_fact_finrank_eq_two :=
  FiniteDimensional.of_fact_finrank_eq_two


/-- An antisymmetric bilinear form on an oriented real inner product space of dimension 2 (usual
notation `ω`). When evaluated on two vectors, it gives the oriented area of the parallelogram they
span. -/
irreducible_def areaForm : E →ₗ[ℝ] E →ₗ[ℝ] ℝ := by
  let z : E [⋀^Fin 0]→ₗ[ℝ] ℝ ≃ₗ[ℝ] ℝ :=
    AlternatingMap.constLinearEquivOfIsEmpty.symm
  let y : E [⋀^Fin 1]→ₗ[ℝ] ℝ →ₗ[ℝ] E →ₗ[ℝ] ℝ :=
    LinearMap.llcomp ℝ E (E [⋀^Fin 0]→ₗ[ℝ] ℝ) ℝ z ∘ₗ AlternatingMap.curryLeftLinearMap
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    z : LinearEquiv (RingHom.id Real) (AlternatingMap Real E Real (Fin 0)) Real := …
    y : LinearMap (RingHom.id Real) (AlternatingMap Real E Real (Fin 1)) (LinearMa …
    ⊢ LinearMap (RingHom.id Real) E (LinearMap (RingHom.id Real) E Real)
  -/
  exact y ∘ₗ AlternatingMap.curryLeftLinearMap (R' := ℝ) o.volumeForm
  /-
    🎉 no goals
  -/


local notation "ω" => o.areaForm


                                                                              /-
                                                                                E : Type u_1
                                                                                inst✝² : NormedAddCommGroup E
                                                                                inst✝¹ : InnerProductSpace Real E
                                                                                inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                                o : Orientation Real E (Fin 2)
                                                                                x y : E
                                                                                ⊢ Eq ((o.areaForm x) y) (o.volumeForm (Matrix.vecCons x (Matrix.vecCons y Matr …
                                                                              -/
theorem areaForm_to_volumeForm (x y : E) : ω x y = o.volumeForm ![x, y] := by simp [areaForm]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem areaForm_apply_self (x : E) : ω x x = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq ((o.areaForm x) x) 0
  -/
  rw [areaForm_to_volumeForm]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq (o.volumeForm (Matrix.vecCons x (Matrix.vecCons x Matrix.vecEmpty))) 0
  -/
  refine o.volumeForm.map_eq_zero_of_eq ![x, x] ?_ (?_ : (0 : Fin 2) ≠ 1)
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x : E
      ⊢ Eq (Matrix.vecCons x (Matrix.vecCons x Matrix.vecEmpty) 0) (Matrix.vecCons x …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x : E
      ⊢ Ne 0 1
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem areaForm_swap (x y : E) : ω x y = -ω y x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((o.areaForm x) y) (Neg.neg ((o.areaForm y) x))
  -/
  simp only [areaForm_to_volumeForm]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (o.volumeForm (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))) (Neg …
  -/
  convert o.volumeForm.map_swap ![y, x] (_ : (0 : Fin 2) ≠ 1)
    /-
      case h.e'_2.h.e'_6
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      ⊢ Eq (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty)) (Function.comp (Mat …
    -/
  · ext i
    /-
      case h.e'_2.h.e'_6.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      i : Fin (Nat.succ 1)
      ⊢ Eq (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) i) (Function.comp (M …
    -/
                    /-
                      🎉 no goals
                    -/
    fin_cases i <;> rfl
                    /-
                      🎉 no goals
                    -/
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      ⊢ Ne 0 1
    -/
  · norm_num
    /-
      🎉 no goals
    -/


@[simp]
theorem areaForm_neg_orientation : (-o).areaForm = -o.areaForm := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    ⊢ Eq (Neg.neg o).areaForm (Neg.neg o.areaForm)
  -/
  ext x y
  /-
    case h.h
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (((Neg.neg o).areaForm x) y) (((Neg.neg o.areaForm) x) y)
  -/
  simp [areaForm_to_volumeForm]
  /-
    🎉 no goals
  -/


/-- Continuous linear map version of `Orientation.areaForm`, useful for calculus. -/
def areaForm' : E →L[ℝ] E →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    (↑(LinearMap.toContinuousLinearMap : (E →ₗ[ℝ] ℝ) ≃ₗ[ℝ] E →L[ℝ] ℝ) ∘ₗ o.areaForm)


@[simp]
theorem areaForm'_apply (x : E) :
    o.areaForm' x = LinearMap.toContinuousLinearMap (o.areaForm x) :=
  rfl


theorem abs_areaForm_le (x y : E) : |ω x y| ≤ ‖x‖ * ‖y‖ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ LE.le (abs ((o.areaForm x) y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  simpa [areaForm_to_volumeForm, Fin.prod_univ_succ] using o.abs_volumeForm_apply_le ![x, y]
  /-
    🎉 no goals
  -/


theorem areaForm_le (x y : E) : ω x y ≤ ‖x‖ * ‖y‖ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ LE.le ((o.areaForm x) y) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  simpa [areaForm_to_volumeForm, Fin.prod_univ_succ] using o.volumeForm_apply_le ![x, y]
  /-
    🎉 no goals
  -/


theorem abs_areaForm_of_orthogonal {x y : E} (h : ⟪x, y⟫ = 0) : |ω x y| = ‖x‖ * ‖y‖ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (abs ((o.areaForm x) y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  rw [o.areaForm_to_volumeForm, o.abs_volumeForm_apply_of_pairwise_orthogonal]
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      h : Eq (Inner.inner x y) 0
      ⊢ Eq (Finset.univ.prod fun i => Norm.norm (Matrix.vecCons x (Matrix.vecCons y  …
    -/
  · simp [Fin.prod_univ_succ]
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    h : Eq (Inner.inner x y) 0
    ⊢ Pairwise fun i j => Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matr …
  -/
  intro i j hij
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    h : Eq (Inner.inner x y) 0
    i j : Fin 2
    hij : Ne i j
    ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) i) (Mat …
  -/
  fin_cases i <;> fin_cases j
    /-
      case «0».«0»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      h : Eq (Inner.inner x y) 0
      hij : Ne ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)
      ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) ((fun i …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case «0».«1»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      h : Eq (Inner.inner x y) 0
      hij : Ne ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩)
      ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) ((fun i …
    -/
  · simpa using h
    /-
      🎉 no goals
    -/
    /-
      case «1».«0»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      h : Eq (Inner.inner x y) 0
      hij : Ne ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)
      ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) ((fun i …
    -/
  · simpa [real_inner_comm] using h
    /-
      🎉 no goals
    -/
    /-
      case «1».«1»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      h : Eq (Inner.inner x y) 0
      hij : Ne ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩)
      ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) ((fun i …
    -/
  · simp_all
    /-
      🎉 no goals
    -/


theorem areaForm_map {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [hF : Fact (finrank ℝ F = 2)] (φ : E ≃ₗᵢ[ℝ] F) (x y : F) :
    (Orientation.map (Fin 2) φ.toLinearEquiv o).areaForm x y =
    o.areaForm (φ.symm x) (φ.symm y) := by
  have : φ.symm ∘ ![x, y] = ![φ.symm x, φ.symm y] := by
    ext i
    fin_cases i <;> rfl
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x y : F
    this : Eq (Function.comp (⇑φ.symm) (Matrix.vecCons x (Matrix.vecCons y Matrix. …
    ⊢ Eq ((((Orientation.map (Fin 2) φ.toLinearEquiv) o).areaForm x) y) ((o.areaFo …
  -/
  simp [areaForm_to_volumeForm, volumeForm_map, this]
  /-
    🎉 no goals
  -/


/-- The area form is invariant under pullback by a positively-oriented isometric automorphism. -/
theorem areaForm_comp_linearIsometryEquiv (φ : E ≃ₗᵢ[ℝ] E)
    (hφ : 0 < LinearMap.det (φ.toLinearEquiv : E →ₗ[ℝ] E)) (x y : E) :
    o.areaForm (φ x) (φ y) = o.areaForm x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    x y : E
    ⊢ Eq ((o.areaForm (φ x)) (φ y)) ((o.areaForm x) y)
  -/
  convert o.areaForm_map φ (φ x) (φ y)
    /-
      case h.e'_2.h.e'_5.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x y : E
      ⊢ Eq o ((Orientation.map (Fin 2) φ.toLinearEquiv) o)
    -/
  · symm
    /-
      case h.e'_2.h.e'_5.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x y : E
      ⊢ Eq ((Orientation.map (Fin 2) φ.toLinearEquiv) o) o
    -/
    rwa [← o.map_eq_iff_det_pos φ.toLinearEquiv] at hφ
    /-
      case h.e'_2.h.e'_5.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x y : E
      ⊢ Eq (Fintype.card (Fin 2)) (Module.finrank Real E)
    -/
    rw [@Fact.out (finrank ℝ E = 2), Fintype.card_fin]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5.h.e'_6
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x y : E
      ⊢ Eq x (φ.symm (φ x))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x y : E
      ⊢ Eq y (φ.symm (φ y))
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- Auxiliary construction for `Orientation.rightAngleRotation`, rotation by 90 degrees in an
oriented real inner product space of dimension 2. -/
irreducible_def rightAngleRotationAux₁ : E →ₗ[ℝ] E :=
  let to_dual : E ≃ₗ[ℝ] E →ₗ[ℝ] ℝ :=
    (InnerProductSpace.toDual ℝ E).toLinearEquiv ≪≫ₗ LinearMap.toContinuousLinearMap.symm
  ↑to_dual.symm ∘ₗ ω


@[simp]
theorem inner_rightAngleRotationAux₁_left (x y : E) : ⟪o.rightAngleRotationAux₁ x, y⟫ = ω x y := by
  -- Porting note: split `simp only` for greater proof control
  simp only [rightAngleRotationAux₁, LinearEquiv.trans_symm, LinearIsometryEquiv.toLinearEquiv_symm,
    LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, LinearEquiv.trans_apply,
    LinearIsometryEquiv.coe_toLinearEquiv]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner ((InnerProductSpace.toDual Real E).symm (LinearMap.toContinu …
  -/
  rw [InnerProductSpace.toDual_symm_apply]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((LinearMap.toContinuousLinearMap.symm.symm (o.areaForm x)) y) ((o.areaFo …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_rightAngleRotationAux₁_right (x y : E) :
    ⟪x, o.rightAngleRotationAux₁ y⟫ = -ω x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner x (o.rightAngleRotationAux₁ y)) (Neg.neg ((o.areaForm x) y))
  -/
  rw [real_inner_comm]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner (o.rightAngleRotationAux₁ y) x) (Neg.neg ((o.areaForm x) y))
  -/
  simp [o.areaForm_swap y x]
  /-
    🎉 no goals
  -/


/-- Auxiliary construction for `Orientation.rightAngleRotation`, rotation by 90 degrees in an
oriented real inner product space of dimension 2. -/
def rightAngleRotationAux₂ : E →ₗᵢ[ℝ] E :=
  { o.rightAngleRotationAux₁ with
    norm_map' := fun x => by
      /-
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        x : E
        ⊢ Eq (Norm.norm (__src✝ x)) (Norm.norm x)
      -/
      dsimp
      /-
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        x : E
        ⊢ Eq (Norm.norm (o.rightAngleRotationAux₁ x)) (Norm.norm x)
      -/
      refine le_antisymm ?_ ?_
        /-
          case refine_1
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          ⊢ LE.le (Norm.norm (o.rightAngleRotationAux₁ x)) (Norm.norm x)
        -/
      · cases' eq_or_lt_of_le (norm_nonneg (o.rightAngleRotationAux₁ x)) with h h
          /-
            case refine_1.inl
            E : Type u_1
            inst✝² : NormedAddCommGroup E
            inst✝¹ : InnerProductSpace Real E
            inst✝ : Fact (Eq (Module.finrank Real E) 2)
            o : Orientation Real E (Fin 2)
            x : E
            h : Eq 0 (Norm.norm (o.rightAngleRotationAux₁ x))
            ⊢ LE.le (Norm.norm (o.rightAngleRotationAux₁ x)) (Norm.norm x)
          -/
        · rw [← h]
          /-
            case refine_1.inl
            E : Type u_1
            inst✝² : NormedAddCommGroup E
            inst✝¹ : InnerProductSpace Real E
            inst✝ : Fact (Eq (Module.finrank Real E) 2)
            o : Orientation Real E (Fin 2)
            x : E
            h : Eq 0 (Norm.norm (o.rightAngleRotationAux₁ x))
            ⊢ LE.le 0 (Norm.norm x)
          -/
          positivity
          /-
            🎉 no goals
          -/
        /-
          case refine_1.inr
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          h : LT.lt 0 (Norm.norm (o.rightAngleRotationAux₁ x))
          ⊢ LE.le (Norm.norm (o.rightAngleRotationAux₁ x)) (Norm.norm x)
        -/
        refine le_of_mul_le_mul_right ?_ h
        /-
          case refine_1.inr
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          h : LT.lt 0 (Norm.norm (o.rightAngleRotationAux₁ x))
          ⊢ LE.le (HMul.hMul (Norm.norm (o.rightAngleRotationAux₁ x)) (Norm.norm (o.righ …
        -/
        rw [← real_inner_self_eq_norm_mul_norm, o.inner_rightAngleRotationAux₁_left]
        /-
          case refine_1.inr
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          h : LT.lt 0 (Norm.norm (o.rightAngleRotationAux₁ x))
          ⊢ LE.le ((o.areaForm x) (o.rightAngleRotationAux₁ x)) (HMul.hMul (Norm.norm x) …
        -/
        exact o.areaForm_le x (o.rightAngleRotationAux₁ x)
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          ⊢ LE.le (Norm.norm x) (Norm.norm (o.rightAngleRotationAux₁ x))
        -/
      · let K : Submodule ℝ E := ℝ ∙ x
        have : Nontrivial Kᗮ := by
          apply nontrivial_of_finrank_pos (R := ℝ)
          have : finrank ℝ K ≤ Finset.card {x} := by
            rw [← Set.toFinset_singleton]
            exact finrank_span_le_card ({x} : Set E)
          have : Finset.card {x} = 1 := Finset.card_singleton x
          have : finrank ℝ K + finrank ℝ Kᗮ = finrank ℝ E := K.finrank_add_finrank_orthogonal
          have : finrank ℝ E = 2 := Fact.out
          omega
        /-
          case refine_2
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          ⊢ LE.le (Norm.norm x) (Norm.norm (o.rightAngleRotationAux₁ x))
        -/
        obtain ⟨w, hw₀⟩ : ∃ w : Kᗮ, w ≠ 0 := exists_ne 0
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          ⊢ LE.le (Norm.norm x) (Norm.norm (o.rightAngleRotationAux₁ x))
        -/
        have hw' : ⟪x, (w : E)⟫ = 0 := Submodule.mem_orthogonal_singleton_iff_inner_right.mp w.2
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          hw' : Eq (Inner.inner x ↑w) 0
          ⊢ LE.le (Norm.norm x) (Norm.norm (o.rightAngleRotationAux₁ x))
        -/
        have hw : (w : E) ≠ 0 := fun h => hw₀ (Submodule.coe_eq_zero.mp h)
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          hw' : Eq (Inner.inner x ↑w) 0
          hw : Ne (↑w) 0
          ⊢ LE.le (Norm.norm x) (Norm.norm (o.rightAngleRotationAux₁ x))
        -/
        refine le_of_mul_le_mul_right ?_ (by rwa [norm_pos_iff] : 0 < ‖(w : E)‖)
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          hw' : Eq (Inner.inner x ↑w) 0
          hw : Ne (↑w) 0
          ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm ↑w)) (HMul.hMul (Norm.norm (o.righ …
        -/
        rw [← o.abs_areaForm_of_orthogonal hw']
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          hw' : Eq (Inner.inner x ↑w) 0
          hw : Ne (↑w) 0
          ⊢ LE.le (abs ((o.areaForm x) ↑w)) (HMul.hMul (Norm.norm (o.rightAngleRotationA …
        -/
        rw [← o.inner_rightAngleRotationAux₁_left x w]
        /-
          case refine_2.intro
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          K : Submodule Real E := Submodule.span Real (Singleton.singleton x)
          this : Nontrivial (Subtype fun x => Membership.mem K.orthogonal x)
          w : Subtype fun x => Membership.mem K.orthogonal x
          hw₀ : Ne w 0
          hw' : Eq (Inner.inner x ↑w) 0
          hw : Ne (↑w) 0
          ⊢ LE.le (abs (Inner.inner (o.rightAngleRotationAux₁ x) ↑w)) (HMul.hMul (Norm.n …
        -/
        exact abs_real_inner_le_norm (o.rightAngleRotationAux₁ x) w }
        /-
          🎉 no goals
        -/


@[simp]
theorem rightAngleRotationAux₁_rightAngleRotationAux₁ (x : E) :
    o.rightAngleRotationAux₁ (o.rightAngleRotationAux₁ x) = -x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq (o.rightAngleRotationAux₁ (o.rightAngleRotationAux₁ x)) (Neg.neg x)
  -/
  apply ext_inner_left ℝ
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ ∀ (v : E), Eq (Inner.inner v (o.rightAngleRotationAux₁ (o.rightAngleRotation …
  -/
  intro y
  have : ⟪o.rightAngleRotationAux₁ y, o.rightAngleRotationAux₁ x⟫ = ⟪y, x⟫ :=
    LinearIsometry.inner_map_map o.rightAngleRotationAux₂ y x
  rw [o.inner_rightAngleRotationAux₁_right, ← o.inner_rightAngleRotationAux₁_left, this,
    inner_neg_right]


/-- An isometric automorphism of an oriented real inner product space of dimension 2 (usual notation
`J`). This automorphism squares to -1. We will define rotations in such a way that this
automorphism is equal to rotation by 90 degrees. -/
irreducible_def rightAngleRotation : E ≃ₗᵢ[ℝ] E :=
  LinearIsometryEquiv.ofLinearIsometry o.rightAngleRotationAux₂ (-o.rightAngleRotationAux₁)
        /-
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          ⊢ Eq (o.rightAngleRotationAux₂.comp (Neg.neg o.rightAngleRotationAux₁)) Linear …
        -/
             /-
               🎉 no goals
             -/
    (by ext; simp [rightAngleRotationAux₂]) (by ext; simp [rightAngleRotationAux₂])
                                                     /-
                                                       🎉 no goals
                                                     -/


local notation "J" => o.rightAngleRotation


@[simp]
theorem inner_rightAngleRotation_left (x y : E) : ⟪J x, y⟫ = ω x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner (o.rightAngleRotation x) y) ((o.areaForm x) y)
  -/
  rw [rightAngleRotation]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner ((LinearIsometryEquiv.ofLinearIsometry o.rightAngleRotationA …
  -/
  exact o.inner_rightAngleRotationAux₁_left x y
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_rightAngleRotation_right (x y : E) : ⟪x, J y⟫ = -ω x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner x (o.rightAngleRotation y)) (Neg.neg ((o.areaForm x) y))
  -/
  rw [rightAngleRotation]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner x ((LinearIsometryEquiv.ofLinearIsometry o.rightAngleRotatio …
  -/
  exact o.inner_rightAngleRotationAux₁_right x y
  /-
    🎉 no goals
  -/


@[simp]
theorem rightAngleRotation_rightAngleRotation (x : E) : J (J x) = -x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq (o.rightAngleRotation (o.rightAngleRotation x)) (Neg.neg x)
  -/
  rw [rightAngleRotation]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq ((LinearIsometryEquiv.ofLinearIsometry o.rightAngleRotationAux₂ (Neg.neg  …
  -/
  exact o.rightAngleRotationAux₁_rightAngleRotationAux₁ x
  /-
    🎉 no goals
  -/


@[simp]
theorem rightAngleRotation_symm :
    LinearIsometryEquiv.symm J = LinearIsometryEquiv.trans J (LinearIsometryEquiv.neg ℝ) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    ⊢ Eq o.rightAngleRotation.symm (o.rightAngleRotation.trans (LinearIsometryEqui …
  -/
  rw [rightAngleRotation]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    ⊢ Eq (LinearIsometryEquiv.ofLinearIsometry o.rightAngleRotationAux₂ (Neg.neg o …
  -/
  exact LinearIsometryEquiv.toLinearIsometry_injective rfl
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     E : Type u_1
                                                                     inst✝² : NormedAddCommGroup E
                                                                     inst✝¹ : InnerProductSpace Real E
                                                                     inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                     o : Orientation Real E (Fin 2)
                                                                     x : E
                                                                     ⊢ Eq (Inner.inner (o.rightAngleRotation x) x) 0
                                                                   -/
theorem inner_rightAngleRotation_self (x : E) : ⟪J x, x⟫ = 0 := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


                                                                             /-
                                                                               E : Type u_1
                                                                               inst✝² : NormedAddCommGroup E
                                                                               inst✝¹ : InnerProductSpace Real E
                                                                               inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                               o : Orientation Real E (Fin 2)
                                                                               x y : E
                                                                               ⊢ Eq (Inner.inner x (o.rightAngleRotation y)) (Neg.neg (Inner.inner (o.rightAn …
                                                                             -/
theorem inner_rightAngleRotation_swap (x y : E) : ⟪x, J y⟫ = -⟪J x, y⟫ := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem inner_rightAngleRotation_swap' (x y : E) : ⟪J x, y⟫ = -⟪x, J y⟫ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner (o.rightAngleRotation x) y) (Neg.neg (Inner.inner x (o.right …
  -/
  simp [o.inner_rightAngleRotation_swap x y]
  /-
    🎉 no goals
  -/


theorem inner_comp_rightAngleRotation (x y : E) : ⟪J x, J y⟫ = ⟪x, y⟫ :=
  LinearIsometryEquiv.inner_map_map J x y


@[simp]
theorem areaForm_rightAngleRotation_left (x y : E) : ω (J x) y = -⟪x, y⟫ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((o.areaForm (o.rightAngleRotation x)) y) (Neg.neg (Inner.inner x y))
  -/
  rw [← o.inner_comp_rightAngleRotation, o.inner_rightAngleRotation_right, neg_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem areaForm_rightAngleRotation_right (x y : E) : ω x (J y) = ⟪x, y⟫ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((o.areaForm x) (o.rightAngleRotation y)) (Inner.inner x y)
  -/
  rw [← o.inner_rightAngleRotation_left, o.inner_comp_rightAngleRotation]
  /-
    🎉 no goals
  -/


                                                                                 /-
                                                                                   E : Type u_1
                                                                                   inst✝² : NormedAddCommGroup E
                                                                                   inst✝¹ : InnerProductSpace Real E
                                                                                   inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                                   o : Orientation Real E (Fin 2)
                                                                                   x y : E
                                                                                   ⊢ Eq ((o.areaForm (o.rightAngleRotation x)) (o.rightAngleRotation y)) ((o.area …
                                                                                 -/
theorem areaForm_comp_rightAngleRotation (x y : E) : ω (J x) (J y) = ω x y := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem rightAngleRotation_trans_rightAngleRotation :
                                                                    /-
                                                                      E : Type u_1
                                                                      inst✝² : NormedAddCommGroup E
                                                                      inst✝¹ : InnerProductSpace Real E
                                                                      inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                      o : Orientation Real E (Fin 2)
                                                                      ⊢ Eq (o.rightAngleRotation.trans o.rightAngleRotation) (LinearIsometryEquiv.ne …
                                                                    -/
    LinearIsometryEquiv.trans J J = LinearIsometryEquiv.neg ℝ := by ext; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem rightAngleRotation_neg_orientation (x : E) :
    (-o).rightAngleRotation x = -o.rightAngleRotation x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq ((Neg.neg o).rightAngleRotation x) (Neg.neg (o.rightAngleRotation x))
  -/
  apply ext_inner_right ℝ
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ ∀ (v : E), Eq (Inner.inner ((Neg.neg o).rightAngleRotation x) v) (Inner.inne …
  -/
  intro y
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Inner.inner ((Neg.neg o).rightAngleRotation x) y) (Inner.inner (Neg.neg  …
  -/
  rw [inner_rightAngleRotation_left]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (((Neg.neg o).areaForm x) y) (Inner.inner (Neg.neg (o.rightAngleRotation  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem rightAngleRotation_trans_neg_orientation :
    (-o).rightAngleRotation = o.rightAngleRotation.trans (LinearIsometryEquiv.neg ℝ) :=
  LinearIsometryEquiv.ext <| o.rightAngleRotation_neg_orientation


theorem rightAngleRotation_map {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [hF : Fact (finrank ℝ F = 2)] (φ : E ≃ₗᵢ[ℝ] F) (x : F) :
    (Orientation.map (Fin 2) φ.toLinearEquiv o).rightAngleRotation x =
      φ (o.rightAngleRotation (φ.symm x)) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x : F
    ⊢ Eq (((Orientation.map (Fin 2) φ.toLinearEquiv) o).rightAngleRotation x) (φ ( …
  -/
  apply ext_inner_right ℝ
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x : F
    ⊢ ∀ (v : F), Eq (Inner.inner (((Orientation.map (Fin 2) φ.toLinearEquiv) o).ri …
  -/
  intro y
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x y : F
    ⊢ Eq (Inner.inner (((Orientation.map (Fin 2) φ.toLinearEquiv) o).rightAngleRot …
  -/
  rw [inner_rightAngleRotation_left]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x y : F
    ⊢ Eq ((((Orientation.map (Fin 2) φ.toLinearEquiv) o).areaForm x) y) (Inner.inn …
  -/
  trans ⟪J (φ.symm x), φ.symm y⟫
    /-
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace Real E
      inst✝² : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      hF : Fact (Eq (Module.finrank Real F) 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E F
      x y : F
      ⊢ Eq ((((Orientation.map (Fin 2) φ.toLinearEquiv) o).areaForm x) y) (Inner.inn …
    -/
  · simp [o.areaForm_map]
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x y : F
    ⊢ Eq (Inner.inner (o.rightAngleRotation (φ.symm x)) (φ.symm y)) (Inner.inner ( …
  -/
  trans ⟪φ (J (φ.symm x)), φ (φ.symm y)⟫
    /-
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace Real E
      inst✝² : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      hF : Fact (Eq (Module.finrank Real F) 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E F
      x y : F
      ⊢ Eq (Inner.inner (o.rightAngleRotation (φ.symm x)) (φ.symm y)) (Inner.inner ( …
    -/
  · rw [φ.inner_map_map]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace Real E
      inst✝² : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      hF : Fact (Eq (Module.finrank Real F) 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E F
      x y : F
      ⊢ Eq (Inner.inner (φ (o.rightAngleRotation (φ.symm x))) (φ (φ.symm y))) (Inner …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- `J` commutes with any positively-oriented isometric automorphism. -/
theorem linearIsometryEquiv_comp_rightAngleRotation (φ : E ≃ₗᵢ[ℝ] E)
    (hφ : 0 < LinearMap.det (φ.toLinearEquiv : E →ₗ[ℝ] E)) (x : E) : φ (J x) = J (φ x) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    x : E
    ⊢ Eq (φ (o.rightAngleRotation x)) (o.rightAngleRotation (φ x))
  -/
  convert (o.rightAngleRotation_map φ (φ x)).symm
    /-
      case h.e'_2.h.e'_6.h.e'_6
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x : E
      ⊢ Eq x (φ.symm (φ x))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x : E
      ⊢ Eq o ((Orientation.map (Fin 2) φ.toLinearEquiv) o)
    -/
  · symm
    /-
      case h.e'_3.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x : E
      ⊢ Eq ((Orientation.map (Fin 2) φ.toLinearEquiv) o) o
    -/
    rwa [← o.map_eq_iff_det_pos φ.toLinearEquiv] at hφ
    /-
      case h.e'_3.h.e'_5.h.e'_5
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      x : E
      ⊢ Eq (Fintype.card (Fin 2)) (Module.finrank Real E)
    -/
    rw [@Fact.out (finrank ℝ E = 2), Fintype.card_fin]
    /-
      🎉 no goals
    -/


theorem rightAngleRotation_map' {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [Fact (finrank ℝ F = 2)] (φ : E ≃ₗᵢ[ℝ] F) :
    (Orientation.map (Fin 2) φ.toLinearEquiv o).rightAngleRotation =
      (φ.symm.trans o.rightAngleRotation).trans φ :=
  LinearIsometryEquiv.ext <| o.rightAngleRotation_map φ


/-- `J` commutes with any positively-oriented isometric automorphism. -/
theorem linearIsometryEquiv_comp_rightAngleRotation' (φ : E ≃ₗᵢ[ℝ] E)
    (hφ : 0 < LinearMap.det (φ.toLinearEquiv : E →ₗ[ℝ] E)) :
    LinearIsometryEquiv.trans J φ = φ.trans J :=
  LinearIsometryEquiv.ext <| o.linearIsometryEquiv_comp_rightAngleRotation φ hφ


/-- For a nonzero vector `x` in an oriented two-dimensional real inner product space `E`,
`![x, J x]` forms an (orthogonal) basis for `E`. -/
def basisRightAngleRotation (x : E) (hx : x ≠ 0) : Basis (Fin 2) ℝ E :=
  @basisOfLinearIndependentOfCardEqFinrank ℝ _ _ _ _ _ _ _ ![x, J x]
                                                                /-
                                                                  E : Type u_1
                                                                  inst✝² : NormedAddCommGroup E
                                                                  inst✝¹ : InnerProductSpace Real E
                                                                  inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                  o : Orientation Real E (Fin 2)
                                                                  x : E
                                                                  hx : Ne x 0
                                                                  i : Fin 2
                                                                  ⊢ Ne (Matrix.vecCons x (Matrix.vecCons (o.rightAngleRotation x) Matrix.vecEmpt …
                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    (linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => by fin_cases i <;> simp [hx])
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
      (by
        /-
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          hx : Ne x 0
          ⊢ Pairwise fun i j => Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons (o.rig …
        -/
        intro i j hij
        /-
          E : Type u_1
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace Real E
          inst✝ : Fact (Eq (Module.finrank Real E) 2)
          o : Orientation Real E (Fin 2)
          x : E
          hx : Ne x 0
          i j : Fin 2
          hij : Ne i j
          ⊢ Eq (Inner.inner (Matrix.vecCons x (Matrix.vecCons (o.rightAngleRotation x) M …
        -/
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
        fin_cases i <;> fin_cases j <;> simp_all))
                                        /-
                                          🎉 no goals
                                        -/
    (@Fact.out (finrank ℝ E = 2)).symm


@[simp]
theorem coe_basisRightAngleRotation (x : E) (hx : x ≠ 0) :
    ⇑(o.basisRightAngleRotation x hx) = ![x, J x] :=
  coe_basisOfLinearIndependentOfCardEqFinrank _ _


/-- For vectors `a x y : E`, the identity `⟪a, x⟫ * ⟪a, y⟫ + ω a x * ω a y = ‖a‖ ^ 2 * ⟪x, y⟫`. (See
`Orientation.inner_mul_inner_add_areaForm_mul_areaForm` for the "applied" form.)-/
theorem inner_mul_inner_add_areaForm_mul_areaForm' (a x : E) :
    ⟪a, x⟫ • innerₛₗ ℝ a + ω a x • ω a = ‖a‖ ^ 2 • innerₛₗ ℝ x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inner.inner a x) ((innerₛₗ Real) a)) (HSMul.hSMu …
  -/
  by_cases ha : a = 0
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Eq a 0
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inner.inner a x) ((innerₛₗ Real) a)) (HSMul.hSMu …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inner.inner a x) ((innerₛₗ Real) a)) (HSMul.hSMu …
  -/
  apply (o.basisRightAngleRotation a ha).ext
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    ⊢ ∀ (i : Fin 2), Eq ((HAdd.hAdd (HSMul.hSMul (Inner.inner a x) ((innerₛₗ Real) …
  -/
  intro i
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    i : Fin 2
    ⊢ Eq ((HAdd.hAdd (HSMul.hSMul (Inner.inner a x) ((innerₛₗ Real) a)) (HSMul.hSM …
  -/
  fin_cases i
  · simp only [Fin.zero_eta, Fin.isValue, id_eq, coe_basisRightAngleRotation, Nat.succ_eq_add_one,
      Nat.reduceAdd, Matrix.cons_val_zero, LinearMap.add_apply, LinearMap.smul_apply, innerₛₗ_apply,
      real_inner_self_eq_norm_sq, smul_eq_mul, areaForm_apply_self, mul_zero, add_zero,
      real_inner_comm]
    /-
      case neg.«_@»._hyg.6442.«0»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (Inner.inner a x) (HPow.hPow (Norm.norm a) 2)) (HMul.hMul (HPo …
    -/
    ring
    /-
      🎉 no goals
    -/
  · simp only [Fin.mk_one, Fin.isValue, id_eq, coe_basisRightAngleRotation, Nat.succ_eq_add_one,
      Nat.reduceAdd, Matrix.cons_val_one, Matrix.head_cons, LinearMap.add_apply,
      LinearMap.smul_apply, innerₛₗ_apply, inner_rightAngleRotation_right, areaForm_apply_self,
      neg_zero, smul_eq_mul, mul_zero, areaForm_rightAngleRotation_right,
      real_inner_self_eq_norm_sq, zero_add, mul_neg]
    /-
      case neg.«_@»._hyg.6442.«1»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul ((o.areaForm a) x) (HPow.hPow (Norm.norm a) 2)) (Neg.neg (HMul …
    -/
    rw [o.areaForm_swap]
    /-
      case neg.«_@»._hyg.6442.«1»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (Neg.neg ((o.areaForm x) a)) (HPow.hPow (Norm.norm a) 2)) (Neg …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- For vectors `a x y : E`, the identity `⟪a, x⟫ * ⟪a, y⟫ + ω a x * ω a y = ‖a‖ ^ 2 * ⟪x, y⟫`. -/
theorem inner_mul_inner_add_areaForm_mul_areaForm (a x y : E) :
    ⟪a, x⟫ * ⟪a, y⟫ + ω a x * ω a y = ‖a‖ ^ 2 * ⟪x, y⟫ :=
  congr_arg (fun f : E →ₗ[ℝ] ℝ => f y) (o.inner_mul_inner_add_areaForm_mul_areaForm' a x)


theorem inner_sq_add_areaForm_sq (a b : E) : ⟪a, b⟫ ^ 2 + ω a b ^ 2 = ‖a‖ ^ 2 * ‖b‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a b : E
    ⊢ Eq (HAdd.hAdd (HPow.hPow (Inner.inner a b) 2) (HPow.hPow ((o.areaForm a) b)  …
  -/
  simpa [sq, real_inner_self_eq_norm_sq] using o.inner_mul_inner_add_areaForm_mul_areaForm a b b
  /-
    🎉 no goals
  -/


/-- For vectors `a x y : E`, the identity `⟪a, x⟫ * ω a y - ω a x * ⟪a, y⟫ = ‖a‖ ^ 2 * ω x y`. (See
`Orientation.inner_mul_areaForm_sub` for the "applied" form.) -/
theorem inner_mul_areaForm_sub' (a x : E) : ⟪a, x⟫ • ω a - ω a x • innerₛₗ ℝ a = ‖a‖ ^ 2 • ω x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ⊢ Eq (HSub.hSub (HSMul.hSMul (Inner.inner a x) (o.areaForm a)) (HSMul.hSMul (( …
  -/
  by_cases ha : a = 0
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Eq a 0
      ⊢ Eq (HSub.hSub (HSMul.hSMul (Inner.inner a x) (o.areaForm a)) (HSMul.hSMul (( …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    ⊢ Eq (HSub.hSub (HSMul.hSMul (Inner.inner a x) (o.areaForm a)) (HSMul.hSMul (( …
  -/
  apply (o.basisRightAngleRotation a ha).ext
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    ⊢ ∀ (i : Fin 2), Eq ((HSub.hSub (HSMul.hSMul (Inner.inner a x) (o.areaForm a)) …
  -/
  intro i
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x : E
    ha : Not (Eq a 0)
    i : Fin 2
    ⊢ Eq ((HSub.hSub (HSMul.hSMul (Inner.inner a x) (o.areaForm a)) (HSMul.hSMul ( …
  -/
  fin_cases i
  · simp only [o.areaForm_swap a x, neg_smul, sub_neg_eq_add, Fin.zero_eta, Fin.isValue, id_eq,
      coe_basisRightAngleRotation, Nat.succ_eq_add_one, Nat.reduceAdd, Matrix.cons_val_zero,
      LinearMap.add_apply, LinearMap.smul_apply, areaForm_apply_self, smul_eq_mul, mul_zero,
      innerₛₗ_apply, real_inner_self_eq_norm_sq, zero_add]
    /-
      case neg.«_@»._hyg.6856.«0»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul ((o.areaForm x) a) (HPow.hPow (Norm.norm a) 2)) (HMul.hMul (HP …
    -/
    ring
    /-
      🎉 no goals
    -/
  · simp only [Fin.mk_one, Fin.isValue, id_eq, coe_basisRightAngleRotation, Nat.succ_eq_add_one,
      Nat.reduceAdd, Matrix.cons_val_one, Matrix.head_cons, LinearMap.sub_apply,
      LinearMap.smul_apply, areaForm_rightAngleRotation_right, real_inner_self_eq_norm_sq,
      smul_eq_mul, innerₛₗ_apply, inner_rightAngleRotation_right, areaForm_apply_self, neg_zero,
      mul_zero, sub_zero, real_inner_comm]
    /-
      case neg.«_@»._hyg.6856.«1»
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x : E
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (Inner.inner a x) (HPow.hPow (Norm.norm a) 2)) (HMul.hMul (HPo …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- For vectors `a x y : E`, the identity `⟪a, x⟫ * ω a y - ω a x * ⟪a, y⟫ = ‖a‖ ^ 2 * ω x y`. -/
theorem inner_mul_areaForm_sub (a x y : E) : ⟪a, x⟫ * ω a y - ω a x * ⟪a, y⟫ = ‖a‖ ^ 2 * ω x y :=
  congr_arg (fun f : E →ₗ[ℝ] ℝ => f y) (o.inner_mul_areaForm_sub' a x)


theorem nonneg_inner_and_areaForm_eq_zero_iff_sameRay (x y : E) :
    0 ≤ ⟪x, y⟫ ∧ ω x y = 0 ↔ SameRay ℝ x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Iff (And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0)) (SameRay Rea …
  -/
  by_cases hx : x = 0
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Eq x 0
      ⊢ Iff (And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0)) (SameRay Rea …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    hx : Not (Eq x 0)
    ⊢ Iff (And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0)) (SameRay Rea …
  -/
  constructor
    /-
      case neg.mp
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      ⊢ And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0) → SameRay Real x y
    -/
  · let a : ℝ := (o.basisRightAngleRotation x hx).repr y 0
    /-
      case neg.mp
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      ⊢ And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0) → SameRay Real x y
    -/
    let b : ℝ := (o.basisRightAngleRotation x hx).repr y 1
    suffices ↑0 ≤ a * ‖x‖ ^ 2 ∧ b * ‖x‖ ^ 2 = 0 → SameRay ℝ x (a • x + b • J x) by
      rw [← (o.basisRightAngleRotation x hx).sum_repr y]
      simp only [Fin.sum_univ_succ, coe_basisRightAngleRotation, Matrix.cons_val_zero,
        Fin.succ_zero_eq_one', Finset.univ_eq_empty, Finset.sum_empty, areaForm_apply_self,
        map_smul, map_add, real_inner_smul_right, inner_add_right, Matrix.cons_val_one,
        Matrix.head_cons, Algebra.id.smul_eq_mul, areaForm_rightAngleRotation_right,
        mul_zero, add_zero, zero_add, neg_zero, inner_rightAngleRotation_right,
        real_inner_self_eq_norm_sq, zero_smul, one_smul]
      exact this
    /-
      case neg.mp
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      b : Real := ((o.basisRightAngleRotation x hx).repr y) 1
      ⊢ And (LE.le 0 (HMul.hMul a (HPow.hPow (Norm.norm x) 2))) (Eq (HMul.hMul b (HP …
    -/
    rintro ⟨ha, hb⟩
    /-
      case neg.mp.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      b : Real := ((o.basisRightAngleRotation x hx).repr y) 1
      ha : LE.le 0 (HMul.hMul a (HPow.hPow (Norm.norm x) 2))
      hb : Eq (HMul.hMul b (HPow.hPow (Norm.norm x) 2)) 0
      ⊢ SameRay Real x (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b (o.rightAngleRota …
    -/
    have hx' : 0 < ‖x‖ := by simpa using hx
    /-
      case neg.mp.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      b : Real := ((o.basisRightAngleRotation x hx).repr y) 1
      ha : LE.le 0 (HMul.hMul a (HPow.hPow (Norm.norm x) 2))
      hb : Eq (HMul.hMul b (HPow.hPow (Norm.norm x) 2)) 0
      hx' : LT.lt 0 (Norm.norm x)
      ⊢ SameRay Real x (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b (o.rightAngleRota …
    -/
    have ha' : 0 ≤ a := nonneg_of_mul_nonneg_left ha (by positivity)
    /-
      case neg.mp.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      b : Real := ((o.basisRightAngleRotation x hx).repr y) 1
      ha : LE.le 0 (HMul.hMul a (HPow.hPow (Norm.norm x) 2))
      hb : Eq (HMul.hMul b (HPow.hPow (Norm.norm x) 2)) 0
      hx' : LT.lt 0 (Norm.norm x)
      ha' : LE.le 0 a
      ⊢ SameRay Real x (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b (o.rightAngleRota …
    -/
    have hb' : b = 0 := eq_zero_of_ne_zero_of_mul_right_eq_zero (pow_ne_zero 2 hx'.ne') hb
    /-
      case neg.mp.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      a : Real := ((o.basisRightAngleRotation x hx).repr y) 0
      b : Real := ((o.basisRightAngleRotation x hx).repr y) 1
      ha : LE.le 0 (HMul.hMul a (HPow.hPow (Norm.norm x) 2))
      hb : Eq (HMul.hMul b (HPow.hPow (Norm.norm x) 2)) 0
      hx' : LT.lt 0 (Norm.norm x)
      ha' : LE.le 0 a
      hb' : Eq b 0
      ⊢ SameRay Real x (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b (o.rightAngleRota …
    -/
    exact (SameRay.sameRay_nonneg_smul_right x ha').add_right <| by simp [hb']
    /-
      🎉 no goals
    -/
    /-
      case neg.mpr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      ⊢ SameRay Real x y → And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0)
    -/
  · intro h
    /-
      case neg.mpr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Not (Eq x 0)
      h : SameRay Real x y
      ⊢ And (LE.le 0 (Inner.inner x y)) (Eq ((o.areaForm x) y) 0)
    -/
    obtain ⟨r, hr, rfl⟩ := h.exists_nonneg_left hx
    simp only [inner_smul_right, real_inner_self_eq_norm_sq, LinearMap.map_smulₛₗ,
      areaForm_apply_self, Algebra.id.smul_eq_mul, mul_zero, eq_self_iff_true, and_true]
    /-
      case neg.mpr.intro.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x : E
      hx : Not (Eq x 0)
      r : Real
      hr : LE.le 0 r
      h : SameRay Real x (HSMul.hSMul r x)
      ⊢ LE.le 0 (HMul.hMul r (HPow.hPow (Norm.norm x) 2))
    -/
    positivity
    /-
      🎉 no goals
    -/


/-- A complex-valued real-bilinear map on an oriented real inner product space of dimension 2. Its
real part is the inner product and its imaginary part is `Orientation.areaForm`.

On `ℂ` with the standard orientation, `kahler w z = conj w * z`; see `Complex.kahler`. -/
def kahler : E →ₗ[ℝ] E →ₗ[ℝ] ℂ :=
  LinearMap.llcomp ℝ E ℝ ℂ Complex.ofRealCLM ∘ₗ innerₛₗ ℝ +
    LinearMap.llcomp ℝ E ℝ ℂ ((LinearMap.lsmul ℝ ℂ).flip Complex.I) ∘ₗ ω


theorem kahler_apply_apply (x y : E) : o.kahler x y = ⟪x, y⟫ + ω x y • Complex.I :=
  rfl


theorem kahler_swap (x y : E) : o.kahler x y = conj (o.kahler y x) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((o.kahler x) y) ((starRingEnd Complex) ((o.kahler y) x))
  -/
  simp only [kahler_apply_apply]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (HAdd.hAdd (↑(Inner.inner x y)) (HSMul.hSMul ((o.areaForm x) y) Complex.I …
  -/
  rw [real_inner_comm, areaForm_swap]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (HAdd.hAdd (↑(Inner.inner y x)) (HSMul.hSMul (Neg.neg ((o.areaForm y) x)) …
  -/
  simp [Complex.conj_ofReal]
  /-
    🎉 no goals
  -/


@[simp]
theorem kahler_apply_self (x : E) : o.kahler x x = ‖x‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x : E
    ⊢ Eq ((o.kahler x) x) (HPow.hPow (↑(Norm.norm x)) 2)
  -/
  simp [kahler_apply_apply, real_inner_self_eq_norm_sq]
  /-
    🎉 no goals
  -/


@[simp]
theorem kahler_rightAngleRotation_left (x y : E) :
    o.kahler (J x) y = -Complex.I * o.kahler x y := by
  simp only [o.areaForm_rightAngleRotation_left, o.inner_rightAngleRotation_left,
    o.kahler_apply_apply, Complex.ofReal_neg, Complex.real_smul]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (HAdd.hAdd (↑((o.areaForm x) y)) (HMul.hMul (Neg.neg ↑(Inner.inner x y))  …
  -/
  linear_combination ω x y * Complex.I_sq
  /-
    🎉 no goals
  -/


@[simp]
theorem kahler_rightAngleRotation_right (x y : E) :
    o.kahler x (J y) = Complex.I * o.kahler x y := by
  simp only [o.areaForm_rightAngleRotation_right, o.inner_rightAngleRotation_right,
    o.kahler_apply_apply, Complex.ofReal_neg, Complex.real_smul]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (HAdd.hAdd (Neg.neg ↑((o.areaForm x) y)) (HMul.hMul (↑(Inner.inner x y))  …
  -/
  linear_combination -ω x y * Complex.I_sq
  /-
    🎉 no goals
  -/

-- @[simp] -- Porting note: simp normal form is `kahler_comp_rightAngleRotation'`

theorem kahler_comp_rightAngleRotation (x y : E) : o.kahler (J x) (J y) = o.kahler x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq ((o.kahler (o.rightAngleRotation x)) (o.rightAngleRotation y)) ((o.kahler …
  -/
  simp only [kahler_rightAngleRotation_left, kahler_rightAngleRotation_right]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (HMul.hMul Complex.I (HMul.hMul (Neg.neg Complex.I) ((o.kahler x) y))) (( …
  -/
  linear_combination -o.kahler x y * Complex.I_sq
  /-
    🎉 no goals
  -/


theorem kahler_comp_rightAngleRotation' (x y : E) :
    -(Complex.I * (Complex.I * o.kahler x y)) = o.kahler x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Neg.neg (HMul.hMul Complex.I (HMul.hMul Complex.I ((o.kahler x) y)))) (( …
  -/
  linear_combination -o.kahler x y * Complex.I_sq
  /-
    🎉 no goals
  -/


@[simp]
theorem kahler_neg_orientation (x y : E) : (-o).kahler x y = conj (o.kahler x y) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (((Neg.neg o).kahler x) y) ((starRingEnd Complex) ((o.kahler x) y))
  -/
  simp [kahler_apply_apply, Complex.conj_ofReal]
  /-
    🎉 no goals
  -/


theorem kahler_mul (a x y : E) : o.kahler x a * o.kahler a y = ‖a‖ ^ 2 * o.kahler x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    a x y : E
    ⊢ Eq (HMul.hMul ((o.kahler x) a) ((o.kahler a) y)) (HMul.hMul (HPow.hPow (↑(No …
  -/
  trans ((‖a‖ ^ 2 :) : ℂ) * o.kahler x y
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x y : E
      ⊢ Eq (HMul.hMul ((o.kahler x) a) ((o.kahler a) y)) (HMul.hMul (↑(HPow.hPow (No …
    -/
  · apply Complex.ext
    · simp only [o.kahler_apply_apply, Complex.add_im, Complex.add_re, Complex.I_im, Complex.I_re,
        Complex.mul_im, Complex.mul_re, Complex.ofReal_im, Complex.ofReal_re, Complex.real_smul]
      /-
        case a
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        a x y : E
        ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (Inner.inner x a) (HSub.hSub (HMul.hMul  …
      -/
      rw [real_inner_comm a x, o.areaForm_swap x a]
      /-
        case a
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        a x y : E
        ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (Inner.inner a x) (HSub.hSub (HMul.hMul  …
      -/
      linear_combination o.inner_mul_inner_add_areaForm_mul_areaForm a x y
      /-
        🎉 no goals
      -/
    · simp only [o.kahler_apply_apply, Complex.add_im, Complex.add_re, Complex.I_im, Complex.I_re,
        Complex.mul_im, Complex.mul_re, Complex.ofReal_im, Complex.ofReal_re, Complex.real_smul]
      /-
        case a
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        a x y : E
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (Inner.inner x a) (HSub.hSub (HMul.hMul  …
      -/
      rw [real_inner_comm a x, o.areaForm_swap x a]
      /-
        case a
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        inst✝ : Fact (Eq (Module.finrank Real E) 2)
        o : Orientation Real E (Fin 2)
        a x y : E
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (Inner.inner a x) (HSub.hSub (HMul.hMul  …
      -/
      linear_combination o.inner_mul_areaForm_sub a x y
      /-
        🎉 no goals
      -/
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      a x y : E
      ⊢ Eq (HMul.hMul (↑(HPow.hPow (Norm.norm a) 2)) ((o.kahler x) y)) (HMul.hMul (H …
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


theorem normSq_kahler (x y : E) : Complex.normSq (o.kahler x y) = ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Complex.normSq ((o.kahler x) y)) (HMul.hMul (HPow.hPow (Norm.norm x) 2)  …
  -/
  simpa [kahler_apply_apply, Complex.normSq, sq] using o.inner_sq_add_areaForm_sq x y
  /-
    🎉 no goals
  -/


theorem abs_kahler (x y : E) : Complex.abs (o.kahler x y) = ‖x‖ * ‖y‖ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Eq (Complex.abs ((o.kahler x) y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  rw [← sq_eq_sq₀, Complex.sq_abs]
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      ⊢ Eq (Complex.normSq ((o.kahler x) y)) (HPow.hPow (HMul.hMul (Norm.norm x) (No …
    -/
  · linear_combination o.normSq_kahler x y
    /-
      🎉 no goals
    -/
    /-
      case ha
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      ⊢ LE.le 0 (Complex.abs ((o.kahler x) y))
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case hb
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      ⊢ LE.le 0 (HMul.hMul (Norm.norm x) (Norm.norm y))
    -/
  · positivity
    /-
      🎉 no goals
    -/


                                                                 /-
                                                                   E : Type u_1
                                                                   inst✝² : NormedAddCommGroup E
                                                                   inst✝¹ : InnerProductSpace Real E
                                                                   inst✝ : Fact (Eq (Module.finrank Real E) 2)
                                                                   o : Orientation Real E (Fin 2)
                                                                   x y : E
                                                                   ⊢ Eq (Norm.norm ((o.kahler x) y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
                                                                 -/
theorem norm_kahler (x y : E) : ‖o.kahler x y‖ = ‖x‖ * ‖y‖ := by simpa using o.abs_kahler x y
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem eq_zero_or_eq_zero_of_kahler_eq_zero {x y : E} (hx : o.kahler x y = 0) : x = 0 ∨ y = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    hx : Eq ((o.kahler x) y) 0
    ⊢ Or (Eq x 0) (Eq y 0)
  -/
  have : ‖x‖ * ‖y‖ = 0 := by simpa [hx] using (o.norm_kahler x y).symm
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    hx : Eq ((o.kahler x) y) 0
    this : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
    ⊢ Or (Eq x 0) (Eq y 0)
  -/
  cases' eq_zero_or_eq_zero_of_mul_eq_zero this with h h
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Eq ((o.kahler x) y) 0
      this : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
      h : Eq (Norm.norm x) 0
      ⊢ Or (Eq x 0) (Eq y 0)
    -/
  · left
    /-
      case inl.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Eq ((o.kahler x) y) 0
      this : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
      h : Eq (Norm.norm x) 0
      ⊢ Eq x 0
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Eq ((o.kahler x) y) 0
      this : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
      h : Eq (Norm.norm y) 0
      ⊢ Or (Eq x 0) (Eq y 0)
    -/
  · right
    /-
      case inr.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      inst✝ : Fact (Eq (Module.finrank Real E) 2)
      o : Orientation Real E (Fin 2)
      x y : E
      hx : Eq ((o.kahler x) y) 0
      this : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
      h : Eq (Norm.norm y) 0
      ⊢ Eq y 0
    -/
    simpa using h
    /-
      🎉 no goals
    -/


theorem kahler_eq_zero_iff (x y : E) : o.kahler x y = 0 ↔ x = 0 ∨ y = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Iff (Eq ((o.kahler x) y) 0) (Or (Eq x 0) (Eq y 0))
  -/
  refine ⟨o.eq_zero_or_eq_zero_of_kahler_eq_zero, ?_⟩
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Or (Eq x 0) (Eq y 0) → Eq ((o.kahler x) y) 0
  -/
                         /-
                           🎉 no goals
                         -/
  rintro (rfl | rfl) <;> simp
                         /-
                           🎉 no goals
                         -/


theorem kahler_ne_zero {x y : E} (hx : x ≠ 0) (hy : y ≠ 0) : o.kahler x y ≠ 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Ne ((o.kahler x) y) 0
  -/
  apply mt o.eq_zero_or_eq_zero_of_kahler_eq_zero
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Not (Or (Eq x 0) (Eq y 0))
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem kahler_ne_zero_iff (x y : E) : o.kahler x y ≠ 0 ↔ x ≠ 0 ∧ y ≠ 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Iff (Ne ((o.kahler x) y) 0) (And (Ne x 0) (Ne y 0))
  -/
  refine ⟨?_, fun h => o.kahler_ne_zero h.1 h.2⟩
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Ne ((o.kahler x) y) 0 → And (Ne x 0) (Ne y 0)
  -/
  contrapose
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Not (And (Ne x 0) (Ne y 0)) → Not (Ne ((o.kahler x) y) 0)
  -/
  simp only [not_and_or, Classical.not_not, kahler_apply_apply, Complex.real_smul]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    x y : E
    ⊢ Or (Eq x 0) (Eq y 0) → Eq (HAdd.hAdd (↑(Inner.inner x y)) (HMul.hMul (↑((o.a …
  -/
                         /-
                           🎉 no goals
                         -/
  rintro (rfl | rfl) <;> simp
                         /-
                           🎉 no goals
                         -/


theorem kahler_map {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [hF : Fact (finrank ℝ F = 2)] (φ : E ≃ₗᵢ[ℝ] F) (x y : F) :
    (Orientation.map (Fin 2) φ.toLinearEquiv o).kahler x y = o.kahler (φ.symm x) (φ.symm y) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    hF : Fact (Eq (Module.finrank Real F) 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x y : F
    ⊢ Eq ((((Orientation.map (Fin 2) φ.toLinearEquiv) o).kahler x) y) ((o.kahler ( …
  -/
  simp [kahler_apply_apply, areaForm_map]
  /-
    🎉 no goals
  -/


/-- The bilinear map `kahler` is invariant under pullback by a positively-oriented isometric
automorphism. -/
theorem kahler_comp_linearIsometryEquiv (φ : E ≃ₗᵢ[ℝ] E)
    (hφ : 0 < LinearMap.det (φ.toLinearEquiv : E →ₗ[ℝ] E)) (x y : E) :
    o.kahler (φ x) (φ y) = o.kahler x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    x y : E
    ⊢ Eq ((o.kahler (φ x)) (φ y)) ((o.kahler x) y)
  -/
  simp [kahler_apply_apply, o.areaForm_comp_linearIsometryEquiv φ hφ]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem areaForm (w z : ℂ) : Complex.orientation.areaForm w z = (conj w * z).im := by
  /-
    w z : Complex
    ⊢ Eq ((Complex.orientation.areaForm w) z) (HMul.hMul ((starRingEnd Complex) w) …
  -/
  let o := Complex.orientation
  simp only [o, o.areaForm_to_volumeForm,
    o.volumeForm_robust Complex.orthonormalBasisOneI rfl, Basis.det_apply, Matrix.det_fin_two,
    Basis.toMatrix_apply, toBasis_orthonormalBasisOneI, Matrix.cons_val_zero, coe_basisOneI_repr,
    Matrix.cons_val_one, Matrix.head_cons, mul_im, conj_re, conj_im]
  /-
    w z : Complex
    o : Orientation Real Complex (Fin 2) := Complex.orientation
    ⊢ Eq (HSub.hSub (HMul.hMul w.re z.im) (HMul.hMul z.re w.im)) (HAdd.hAdd (HMul. …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
protected theorem rightAngleRotation (z : ℂ) :
    Complex.orientation.rightAngleRotation z = I * z := by
  /-
    z : Complex
    ⊢ Eq (Complex.orientation.rightAngleRotation z) (HMul.hMul Complex.I z)
  -/
  apply ext_inner_right ℝ
  /-
    z : Complex
    ⊢ ∀ (v : Complex), Eq (Inner.inner (Complex.orientation.rightAngleRotation z)  …
  -/
  intro w
  /-
    z w : Complex
    ⊢ Eq (Inner.inner (Complex.orientation.rightAngleRotation z) w) (Inner.inner ( …
  -/
  rw [Orientation.inner_rightAngleRotation_left]
  simp only [Complex.areaForm, Complex.inner, mul_re, mul_im, conj_re, conj_im, map_mul, conj_I,
    neg_re, neg_im, I_re, I_im]
  /-
    z w : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul z.re w.im) (HMul.hMul (Neg.neg z.im) w.re)) (HSub.h …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
protected theorem kahler (w z : ℂ) : Complex.orientation.kahler w z = conj w * z := by
  /-
    w z : Complex
    ⊢ Eq ((Complex.orientation.kahler w) z) (HMul.hMul ((starRingEnd Complex) w) z)
  -/
  rw [Orientation.kahler_apply_apply]
  /-
    w z : Complex
    ⊢ Eq (HAdd.hAdd (↑(Inner.inner w z)) (HSMul.hSMul ((Complex.orientation.areaFo …
  -/
                        /-
                          🎉 no goals
                        -/
  apply Complex.ext <;> simp
                        /-
                          🎉 no goals
                        -/


/-- The area form on an oriented real inner product space of dimension 2 can be evaluated in terms
of a complex-number representation of the space. -/
theorem areaForm_map_complex (f : E ≃ₗᵢ[ℝ] ℂ)
    (hf : Orientation.map (Fin 2) f.toLinearEquiv o = Complex.orientation) (x y : E) :
    ω x y = (conj (f x) * f y).im := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) E Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : E
    ⊢ Eq ((o.areaForm x) y) (HMul.hMul ((starRingEnd Complex) (f x)) (f y)).im
  -/
  rw [← Complex.areaForm, ← hf, areaForm_map (hF := _)]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) E Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : E
    ⊢ Eq ((o.areaForm x) y) ((o.areaForm (f.symm (f x))) (f.symm (f y)))
  -/
  iterate 2 rw [LinearIsometryEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- The rotation by 90 degrees on an oriented real inner product space of dimension 2 can be
evaluated in terms of a complex-number representation of the space. -/
theorem rightAngleRotation_map_complex (f : E ≃ₗᵢ[ℝ] ℂ)
    (hf : Orientation.map (Fin 2) f.toLinearEquiv o = Complex.orientation) (x : E) :
    f (J x) = I * f x := by
  rw [← Complex.rightAngleRotation, ← hf, rightAngleRotation_map (hF := _),
    LinearIsometryEquiv.symm_apply_apply]


/-- The Kahler form on an oriented real inner product space of dimension 2 can be evaluated in terms
of a complex-number representation of the space. -/
theorem kahler_map_complex (f : E ≃ₗᵢ[ℝ] ℂ)
    (hf : Orientation.map (Fin 2) f.toLinearEquiv o = Complex.orientation) (x y : E) :
    o.kahler x y = conj (f x) * f y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) E Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : E
    ⊢ Eq ((o.kahler x) y) (HMul.hMul ((starRingEnd Complex) (f x)) (f y))
  -/
  rw [← Complex.kahler, ← hf, kahler_map (hF := _)]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    inst✝ : Fact (Eq (Module.finrank Real E) 2)
    o : Orientation Real E (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) E Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : E
    ⊢ Eq ((o.kahler x) y) ((o.kahler (f.symm (f x))) (f.symm (f y)))
  -/
  iterate 2 rw [LinearIsometryEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


