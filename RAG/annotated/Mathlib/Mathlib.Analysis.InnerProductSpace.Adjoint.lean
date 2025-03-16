local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- The adjoint, as a continuous conjugate-linear map. This is only meant as an auxiliary
definition for the main definition `adjoint`, where this is bundled as a conjugate-linear isometric
equivalence. -/
noncomputable def adjointAux : (E →L[𝕜] F) →L⋆[𝕜] F →L[𝕜] E :=
  (ContinuousLinearMap.compSL _ _ _ _ _ ((toDual 𝕜 E).symm : NormedSpace.Dual 𝕜 E →L⋆[𝕜] E)).comp
    (toSesqForm : (E →L[𝕜] F) →L[𝕜] F →L⋆[𝕜] NormedSpace.Dual 𝕜 E)


@[simp]
theorem adjointAux_apply (A : E →L[𝕜] F) (x : F) :
    adjointAux A x = ((toDual 𝕜 E).symm : NormedSpace.Dual 𝕜 E → E) ((toSesqForm A) x) :=
  rfl


theorem adjointAux_inner_left (A : E →L[𝕜] F) (x : E) (y : F) : ⟪adjointAux A y, x⟫ = ⟪y, A x⟫ := by
  rw [adjointAux_apply, toDual_symm_apply, toSesqForm_apply_coe, coe_comp', innerSL_apply_coe,
    Function.comp_apply]


theorem adjointAux_inner_right (A : E →L[𝕜] F) (x : E) (y : F) :
    ⟪x, adjointAux A y⟫ = ⟪A x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace E
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    ⊢ Eq (Inner.inner x ((ContinuousLinearMap.adjointAux A) y)) (Inner.inner (A x) …
  -/
  rw [← inner_conj_symm, adjointAux_inner_left, inner_conj_symm]
  /-
    🎉 no goals
  -/


theorem adjointAux_adjointAux (A : E →L[𝕜] F) : adjointAux (adjointAux A) = A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (ContinuousLinearMap.adjointAux (ContinuousLinearMap.adjointAux A)) A
  -/
  ext v
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    v : E
    ⊢ Eq ((ContinuousLinearMap.adjointAux (ContinuousLinearMap.adjointAux A)) v) ( …
  -/
  refine ext_inner_left 𝕜 fun w => ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    v : E
    w : F
    ⊢ Eq (Inner.inner w ((ContinuousLinearMap.adjointAux (ContinuousLinearMap.adjo …
  -/
  rw [adjointAux_inner_right, adjointAux_inner_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjointAux_norm (A : E →L[𝕜] F) : ‖adjointAux A‖ = ‖A‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (Norm.norm (ContinuousLinearMap.adjointAux A)) (Norm.norm A)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ LE.le (Norm.norm (ContinuousLinearMap.adjointAux A)) (Norm.norm A)
    -/
  · refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun x => ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : F
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.adjointAux A) x)) (HMul.hMul (Norm.no …
    -/
    rw [adjointAux_apply, LinearIsometryEquiv.norm_map]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : F
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.toSesqForm A) x)) (HMul.hMul (Norm.no …
    -/
    exact toSesqForm_apply_norm_le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ LE.le (Norm.norm A) (Norm.norm (ContinuousLinearMap.adjointAux A))
    -/
  · nth_rw 1 [← adjointAux_adjointAux A]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ LE.le (Norm.norm (ContinuousLinearMap.adjointAux (ContinuousLinearMap.adjoin …
    -/
    refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun x => ?_
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.adjointAux (ContinuousLinearMap.adjoi …
    -/
    rw [adjointAux_apply, LinearIsometryEquiv.norm_map]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.toSesqForm (ContinuousLinearMap.adjoi …
    -/
    exact toSesqForm_apply_norm_le
    /-
      🎉 no goals
    -/


/-- The adjoint of a bounded operator from Hilbert space `E` to Hilbert space `F`. -/
def adjoint : (E →L[𝕜] F) ≃ₗᵢ⋆[𝕜] F →L[𝕜] E :=
  LinearIsometryEquiv.ofSurjective { adjointAux with norm_map' := adjointAux_norm } fun A =>
    ⟨adjointAux A, adjointAux_adjointAux A⟩


scoped[InnerProduct] postfix:1000 "†" => ContinuousLinearMap.adjoint

/-- The fundamental property of the adjoint. -/
theorem adjoint_inner_left (A : E →L[𝕜] F) (x : E) (y : F) : ⟪(A†) y, x⟫ = ⟪y, A x⟫ :=
  adjointAux_inner_left A x y


/-- The fundamental property of the adjoint. -/
theorem adjoint_inner_right (A : E →L[𝕜] F) (x : E) (y : F) : ⟪x, (A†) y⟫ = ⟪A x, y⟫ :=
  adjointAux_inner_right A x y


/-- The adjoint is involutive. -/
@[simp]
theorem adjoint_adjoint (A : E →L[𝕜] F) : A†† = A :=
  adjointAux_adjointAux A


/-- The adjoint of the composition of two operators is the composition of the two adjoints
in reverse order. -/
@[simp]
theorem adjoint_comp (A : F →L[𝕜] G) (B : E →L[𝕜] F) : (A ∘L B)† = B† ∘L A† := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : CompleteSpace E
    inst✝¹ : CompleteSpace G
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) F G
    B : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (ContinuousLinearMap.adjoint (A.comp B)) ((ContinuousLinearMap.adjoint B) …
  -/
  ext v
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : CompleteSpace E
    inst✝¹ : CompleteSpace G
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) F G
    B : ContinuousLinearMap (RingHom.id 𝕜) E F
    v : G
    ⊢ Eq ((ContinuousLinearMap.adjoint (A.comp B)) v) (((ContinuousLinearMap.adjoi …
  -/
  refine ext_inner_left 𝕜 fun w => ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : CompleteSpace E
    inst✝¹ : CompleteSpace G
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) F G
    B : ContinuousLinearMap (RingHom.id 𝕜) E F
    v : G
    w : E
    ⊢ Eq (Inner.inner w ((ContinuousLinearMap.adjoint (A.comp B)) v)) (Inner.inner …
  -/
  simp only [adjoint_inner_right, ContinuousLinearMap.coe_comp', Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem apply_norm_sq_eq_inner_adjoint_left (A : E →L[𝕜] F) (x : E) :
    ‖A x‖ ^ 2 = re ⟪(A† ∘L A) x, x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (HPow.hPow (Norm.norm (A x)) 2) (RCLike.re (Inner.inner (((ContinuousLine …
  -/
  have h : ⟪(A† ∘L A) x, x⟫ = ⟪A x, A x⟫ := by rw [← adjoint_inner_left]; rfl
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : Eq (Inner.inner (((ContinuousLinearMap.adjoint A).comp A) x) x) (Inner.inn …
    ⊢ Eq (HPow.hPow (Norm.norm (A x)) 2) (RCLike.re (Inner.inner (((ContinuousLine …
  -/
  rw [h, ← inner_self_eq_norm_sq (𝕜 := 𝕜) _]
  /-
    🎉 no goals
  -/


theorem apply_norm_eq_sqrt_inner_adjoint_left (A : E →L[𝕜] F) (x : E) :
    ‖A x‖ = √(re ⟪(A† ∘L A) x, x⟫) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (Norm.norm (A x)) (RCLike.re (Inner.inner (((ContinuousLinearMap.adjoint  …
  -/
  rw [← apply_norm_sq_eq_inner_adjoint_left, Real.sqrt_sq (norm_nonneg _)]
  /-
    🎉 no goals
  -/


theorem apply_norm_sq_eq_inner_adjoint_right (A : E →L[𝕜] F) (x : E) :
    ‖A x‖ ^ 2 = re ⟪x, (A† ∘L A) x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (HPow.hPow (Norm.norm (A x)) 2) (RCLike.re (Inner.inner x (((ContinuousLi …
  -/
  have h : ⟪x, (A† ∘L A) x⟫ = ⟪A x, A x⟫ := by rw [← adjoint_inner_right]; rfl
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : Eq (Inner.inner x (((ContinuousLinearMap.adjoint A).comp A) x)) (Inner.inn …
    ⊢ Eq (HPow.hPow (Norm.norm (A x)) 2) (RCLike.re (Inner.inner x (((ContinuousLi …
  -/
  rw [h, ← inner_self_eq_norm_sq (𝕜 := 𝕜) _]
  /-
    🎉 no goals
  -/


theorem apply_norm_eq_sqrt_inner_adjoint_right (A : E →L[𝕜] F) (x : E) :
    ‖A x‖ = √(re ⟪x, (A† ∘L A) x⟫) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (Norm.norm (A x)) (RCLike.re (Inner.inner x (((ContinuousLinearMap.adjoin …
  -/
  rw [← apply_norm_sq_eq_inner_adjoint_right, Real.sqrt_sq (norm_nonneg _)]
  /-
    🎉 no goals
  -/


/-- The adjoint is unique: a map `A` is the adjoint of `B` iff it satisfies `⟪A x, y⟫ = ⟪x, B y⟫`
for all `x` and `y`. -/
theorem eq_adjoint_iff (A : E →L[𝕜] F) (B : F →L[𝕜] E) : A = B† ↔ ∀ x y, ⟪A x, y⟫ = ⟪x, B y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    B : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ Iff (Eq A (ContinuousLinearMap.adjoint B)) (∀ (x : E) (y : F), Eq (Inner.inn …
  -/
  refine ⟨fun h x y => by rw [h, adjoint_inner_left], fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    B : ContinuousLinearMap (RingHom.id 𝕜) F E
    h : ∀ (x : E) (y : F), Eq (Inner.inner (A x) y) (Inner.inner x (B y))
    ⊢ Eq A (ContinuousLinearMap.adjoint B)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    B : ContinuousLinearMap (RingHom.id 𝕜) F E
    h : ∀ (x : E) (y : F), Eq (Inner.inner (A x) y) (Inner.inner x (B y))
    x : E
    ⊢ Eq (A x) ((ContinuousLinearMap.adjoint B) x)
  -/
  exact ext_inner_right 𝕜 fun y => by simp only [adjoint_inner_left, h x y]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjoint_id :
    ContinuousLinearMap.adjoint (ContinuousLinearMap.id 𝕜 E) = ContinuousLinearMap.id 𝕜 E := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    ⊢ Eq (ContinuousLinearMap.adjoint (ContinuousLinearMap.id 𝕜 E)) (ContinuousLin …
  -/
  refine Eq.symm ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    ⊢ Eq (ContinuousLinearMap.id 𝕜 E) (ContinuousLinearMap.adjoint (ContinuousLine …
  -/
  rw [eq_adjoint_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    ⊢ ∀ (x y : E), Eq (Inner.inner ((ContinuousLinearMap.id 𝕜 E) x) y) (Inner.inne …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem _root_.Submodule.adjoint_subtypeL (U : Submodule 𝕜 E) [CompleteSpace U] :
    U.subtypeL† = orthogonalProjection U := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ Eq (ContinuousLinearMap.adjoint U.subtypeL) (orthogonalProjection U)
  -/
  symm
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ Eq (orthogonalProjection U) (ContinuousLinearMap.adjoint U.subtypeL)
  -/
  rw [eq_adjoint_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ ∀ (x : E) (y : Subtype fun x => Membership.mem U x), Eq (Inner.inner ((ortho …
  -/
  intro x u
  rw [U.coe_inner, inner_orthogonalProjection_left_eq_right,
    orthogonalProjection_mem_subspace_eq_self]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    x : E
    u : Subtype fun x => Membership.mem U x
    ⊢ Eq (Inner.inner x ↑u) (Inner.inner x (U.subtypeL u))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem _root_.Submodule.adjoint_orthogonalProjection (U : Submodule 𝕜 E) [CompleteSpace U] :
    (orthogonalProjection U : E →L[𝕜] U)† = U.subtypeL := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ Eq (ContinuousLinearMap.adjoint (orthogonalProjection U)) U.subtypeL
  -/
  rw [← U.adjoint_subtypeL, adjoint_adjoint]
  /-
    🎉 no goals
  -/


/-- `E →L[𝕜] E` is a star algebra with the adjoint as the star operation. -/
instance : Star (E →L[𝕜] E) :=
  ⟨adjoint⟩


instance : InvolutiveStar (E →L[𝕜] E) :=
  ⟨adjoint_adjoint⟩


instance : StarMul (E →L[𝕜] E) :=
  ⟨adjoint_comp⟩


instance : StarRing (E →L[𝕜] E) :=
  ⟨LinearIsometryEquiv.map_add adjoint⟩


instance : StarModule 𝕜 (E →L[𝕜] E) :=
  ⟨LinearIsometryEquiv.map_smulₛₗ adjoint⟩


theorem star_eq_adjoint (A : E →L[𝕜] E) : star A = A† :=
  rfl


/-- A continuous linear operator is self-adjoint iff it is equal to its adjoint. -/
theorem isSelfAdjoint_iff' {A : E →L[𝕜] E} : IsSelfAdjoint A ↔ ContinuousLinearMap.adjoint A = A :=
  Iff.rfl


theorem norm_adjoint_comp_self (A : E →L[𝕜] F) :
    ‖ContinuousLinearMap.adjoint A ∘L A‖ = ‖A‖ * ‖A‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (Norm.norm ((ContinuousLinearMap.adjoint A).comp A)) (HMul.hMul (Norm.nor …
  -/
  refine le_antisymm ?_ ?_
  · calc
      ‖A† ∘L A‖ ≤ ‖A†‖ * ‖A‖ := opNorm_comp_le _ _
      _ = ‖A‖ * ‖A‖ := by rw [LinearIsometryEquiv.norm_map]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ LE.le (HMul.hMul (Norm.norm A) (Norm.norm A)) (Norm.norm ((ContinuousLinearM …
    -/
  · rw [← sq, ← Real.sqrt_le_sqrt_iff (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ LE.le (Norm.norm A) (Norm.norm ((ContinuousLinearMap.adjoint A).comp A)).sqrt
    -/
    refine opNorm_le_bound _ (Real.sqrt_nonneg _) fun x => ?_
    have :=
      calc
        re ⟪(A† ∘L A) x, x⟫ ≤ ‖(A† ∘L A) x‖ * ‖x‖ := re_inner_le_norm _ _
        _ ≤ ‖A† ∘L A‖ * ‖x‖ * ‖x‖ := mul_le_mul_of_nonneg_right (le_opNorm _ _) (norm_nonneg _)
    calc
      ‖A x‖ = √(re ⟪(A† ∘L A) x, x⟫) := by rw [apply_norm_eq_sqrt_inner_adjoint_left]
      _ ≤ √(‖A† ∘L A‖ * ‖x‖ * ‖x‖) := Real.sqrt_le_sqrt this
      _ = √‖A† ∘L A‖ * ‖x‖ := by
        simp_rw [mul_assoc, Real.sqrt_mul (norm_nonneg _) (‖x‖ * ‖x‖),
          Real.sqrt_mul_self (norm_nonneg x)]


/-- The C⋆-algebra instance when `𝕜 := ℂ` can be found in
`Analysis.CStarAlgebra.ContinuousLinearMap`. -/
instance : CStarRing (E →L[𝕜] E) where
  norm_mul_self_le x := le_of_eq <| Eq.symm <| norm_adjoint_comp_self x


theorem isAdjointPair_inner (A : E →L[𝕜] F) :
    LinearMap.IsAdjointPair (sesqFormOfInner : E →ₗ[𝕜] E →ₗ⋆[𝕜] 𝕜)
      (sesqFormOfInner : F →ₗ[𝕜] F →ₗ⋆[𝕜] 𝕜) A (A†) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ sesqFormOfInner.IsAdjointPair sesqFormOfInner ⇑A ⇑(ContinuousLinearMap.adjoi …
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    ⊢ Eq ((sesqFormOfInner (A x)) y) ((sesqFormOfInner x) ((ContinuousLinearMap.ad …
  -/
  simp only [sesqFormOfInner_apply_apply, adjoint_inner_left, coe_coe]
  /-
    🎉 no goals
  -/


theorem adjoint_eq {A : E →L[𝕜] E} (hA : IsSelfAdjoint A) : ContinuousLinearMap.adjoint A = A :=
  hA


/-- Every self-adjoint operator on an inner product space is symmetric. -/
theorem isSymmetric {A : E →L[𝕜] E} (hA : IsSelfAdjoint A) : (A : E →ₗ[𝕜] E).IsSymmetric := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : ContinuousLinearMap (RingHom.id 𝕜) E E
    hA : IsSelfAdjoint A
    ⊢ (↑A).IsSymmetric
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : ContinuousLinearMap (RingHom.id 𝕜) E E
    hA : IsSelfAdjoint A
    x y : E
    ⊢ Eq (Inner.inner (↑A x) y) (Inner.inner x (↑A y))
  -/
  rw_mod_cast [← A.adjoint_inner_right, hA.adjoint_eq]
  /-
    🎉 no goals
  -/


/-- Conjugating preserves self-adjointness. -/
theorem conj_adjoint {T : E →L[𝕜] E} (hT : IsSelfAdjoint T) (S : E →L[𝕜] F) :
    IsSelfAdjoint (S ∘L T ∘L ContinuousLinearMap.adjoint S) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ IsSelfAdjoint (S.comp (T.comp (ContinuousLinearMap.adjoint S)))
  -/
  rw [isSelfAdjoint_iff'] at hT ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : Eq (ContinuousLinearMap.adjoint T) T
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (ContinuousLinearMap.adjoint (S.comp (T.comp (ContinuousLinearMap.adjoint …
  -/
  simp only [hT, adjoint_comp, adjoint_adjoint]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : Eq (ContinuousLinearMap.adjoint T) T
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq ((S.comp T).comp (ContinuousLinearMap.adjoint S)) (S.comp (T.comp (Contin …
  -/
  exact ContinuousLinearMap.comp_assoc _ _ _
  /-
    🎉 no goals
  -/


/-- Conjugating preserves self-adjointness. -/
theorem adjoint_conj {T : E →L[𝕜] E} (hT : IsSelfAdjoint T) (S : F →L[𝕜] E) :
    IsSelfAdjoint (ContinuousLinearMap.adjoint S ∘L T ∘L S) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    S : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ IsSelfAdjoint ((ContinuousLinearMap.adjoint S).comp (T.comp S))
  -/
  rw [isSelfAdjoint_iff'] at hT ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : Eq (ContinuousLinearMap.adjoint T) T
    S : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ Eq (ContinuousLinearMap.adjoint ((ContinuousLinearMap.adjoint S).comp (T.com …
  -/
  simp only [hT, adjoint_comp, adjoint_adjoint]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : Eq (ContinuousLinearMap.adjoint T) T
    S : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ Eq (((ContinuousLinearMap.adjoint S).comp T).comp S) ((ContinuousLinearMap.a …
  -/
  exact ContinuousLinearMap.comp_assoc _ _ _
  /-
    🎉 no goals
  -/


theorem _root_.ContinuousLinearMap.isSelfAdjoint_iff_isSymmetric {A : E →L[𝕜] E} :
    IsSelfAdjoint A ↔ (A : E →ₗ[𝕜] E).IsSymmetric :=
  ⟨fun hA => hA.isSymmetric, fun hA =>
    ext fun x => ext_inner_right 𝕜 fun y => (A.adjoint_inner_left y x).symm ▸ (hA x y).symm⟩


theorem _root_.LinearMap.IsSymmetric.isSelfAdjoint {A : E →L[𝕜] E}
    (hA : (A : E →ₗ[𝕜] E).IsSymmetric) : IsSelfAdjoint A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : ContinuousLinearMap (RingHom.id 𝕜) E E
    hA : (↑A).IsSymmetric
    ⊢ IsSelfAdjoint A
  -/
  rwa [← ContinuousLinearMap.isSelfAdjoint_iff_isSymmetric] at hA
  /-
    🎉 no goals
  -/


/-- The orthogonal projection is self-adjoint. -/
theorem _root_.orthogonalProjection_isSelfAdjoint (U : Submodule 𝕜 E) [CompleteSpace U] :
    IsSelfAdjoint (U.subtypeL ∘L orthogonalProjection U) :=
  (orthogonalProjection_isSymmetric U).isSelfAdjoint


theorem conj_orthogonalProjection {T : E →L[𝕜] E} (hT : IsSelfAdjoint T) (U : Submodule 𝕜 E)
    [CompleteSpace U] :
    IsSelfAdjoint
      (U.subtypeL ∘L orthogonalProjection U ∘L T ∘L U.subtypeL ∘L orthogonalProjection U) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ IsSelfAdjoint (U.subtypeL.comp ((orthogonalProjection U).comp (T.comp (U.sub …
  -/
  rw [← ContinuousLinearMap.comp_assoc]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ IsSelfAdjoint ((U.subtypeL.comp (orthogonalProjection U)).comp (T.comp (U.su …
  -/
  nth_rw 1 [← (orthogonalProjection_isSelfAdjoint U).adjoint_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ IsSelfAdjoint ((ContinuousLinearMap.adjoint (U.subtypeL.comp (orthogonalProj …
  -/
  exact hT.adjoint_conj _
  /-
    🎉 no goals
  -/


/-- The **Hellinger--Toeplitz theorem**: Construct a self-adjoint operator from an everywhere
  defined symmetric operator. -/
def IsSymmetric.toSelfAdjoint (hT : IsSymmetric T) : selfAdjoint (E →L[𝕜] E) :=
  ⟨⟨T, hT.continuous⟩, ContinuousLinearMap.isSelfAdjoint_iff_isSymmetric.mpr hT⟩


theorem IsSymmetric.coe_toSelfAdjoint (hT : IsSymmetric T) : (hT.toSelfAdjoint : E →ₗ[𝕜] E) = T :=
  rfl


theorem IsSymmetric.toSelfAdjoint_apply (hT : IsSymmetric T) {x : E} :
    (hT.toSelfAdjoint : E → E) x = T x :=
  rfl


/-- The adjoint of an operator from the finite-dimensional inner product space `E` to the
finite-dimensional inner product space `F`. -/
def adjoint : (E →ₗ[𝕜] F) ≃ₗ⋆[𝕜] F →ₗ[𝕜] E :=
  have := FiniteDimensional.complete 𝕜 E
  have := FiniteDimensional.complete 𝕜 F
  /- Note: Instead of the two instances above, the following works:
    ```
      have := FiniteDimensional.complete 𝕜
      have := FiniteDimensional.complete 𝕜
    ```
    But removing one of the `have`s makes it fail. The reason is that `E` and `F` don't live
    in the same universe, so the first `have` can no longer be used for `F` after its universe
    metavariable has been assigned to that of `E`!
  -/
  ((LinearMap.toContinuousLinearMap : (E →ₗ[𝕜] F) ≃ₗ[𝕜] E →L[𝕜] F).trans
      ContinuousLinearMap.adjoint.toLinearEquiv).trans
    LinearMap.toContinuousLinearMap.symm


theorem adjoint_toContinuousLinearMap (A : E →ₗ[𝕜] F) :
    haveI := FiniteDimensional.complete 𝕜 E
    haveI := FiniteDimensional.complete 𝕜 F
    LinearMap.toContinuousLinearMap (LinearMap.adjoint A) =
      ContinuousLinearMap.adjoint (LinearMap.toContinuousLinearMap A) :=
  rfl


theorem adjoint_eq_toCLM_adjoint (A : E →ₗ[𝕜] F) :
    haveI := FiniteDimensional.complete 𝕜 E
    haveI := FiniteDimensional.complete 𝕜 F
    LinearMap.adjoint A = ContinuousLinearMap.adjoint (LinearMap.toContinuousLinearMap A) :=
  rfl


/-- The fundamental property of the adjoint. -/
theorem adjoint_inner_left (A : E →ₗ[𝕜] F) (x : E) (y : F) : ⟪adjoint A y, x⟫ = ⟪y, A x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    ⊢ Eq (Inner.inner ((LinearMap.adjoint A) y) x) (Inner.inner y (A x))
  -/
  haveI := FiniteDimensional.complete 𝕜 E
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this : CompleteSpace E
    ⊢ Eq (Inner.inner ((LinearMap.adjoint A) y) x) (Inner.inner y (A x))
  -/
  haveI := FiniteDimensional.complete 𝕜 F
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this✝ : CompleteSpace E
    this : CompleteSpace F
    ⊢ Eq (Inner.inner ((LinearMap.adjoint A) y) x) (Inner.inner y (A x))
  -/
  rw [← coe_toContinuousLinearMap A, adjoint_eq_toCLM_adjoint]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this✝ : CompleteSpace E
    this : CompleteSpace F
    ⊢ Eq (Inner.inner (↑(ContinuousLinearMap.adjoint (LinearMap.toContinuousLinear …
  -/
  exact ContinuousLinearMap.adjoint_inner_left _ x y
  /-
    🎉 no goals
  -/


/-- The fundamental property of the adjoint. -/
theorem adjoint_inner_right (A : E →ₗ[𝕜] F) (x : E) (y : F) : ⟪x, adjoint A y⟫ = ⟪A x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    ⊢ Eq (Inner.inner x ((LinearMap.adjoint A) y)) (Inner.inner (A x) y)
  -/
  haveI := FiniteDimensional.complete 𝕜 E
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this : CompleteSpace E
    ⊢ Eq (Inner.inner x ((LinearMap.adjoint A) y)) (Inner.inner (A x) y)
  -/
  haveI := FiniteDimensional.complete 𝕜 F
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this✝ : CompleteSpace E
    this : CompleteSpace F
    ⊢ Eq (Inner.inner x ((LinearMap.adjoint A) y)) (Inner.inner (A x) y)
  -/
  rw [← coe_toContinuousLinearMap A, adjoint_eq_toCLM_adjoint]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    this✝ : CompleteSpace E
    this : CompleteSpace F
    ⊢ Eq (Inner.inner x (↑(ContinuousLinearMap.adjoint (LinearMap.toContinuousLine …
  -/
  exact ContinuousLinearMap.adjoint_inner_right _ x y
  /-
    🎉 no goals
  -/


/-- The adjoint is involutive. -/
@[simp]
theorem adjoint_adjoint (A : E →ₗ[𝕜] F) : LinearMap.adjoint (LinearMap.adjoint A) = A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (LinearMap.adjoint (LinearMap.adjoint A)) A
  -/
  ext v
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    v : E
    ⊢ Eq ((LinearMap.adjoint (LinearMap.adjoint A)) v) (A v)
  -/
  refine ext_inner_left 𝕜 fun w => ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    v : E
    w : F
    ⊢ Eq (Inner.inner w ((LinearMap.adjoint (LinearMap.adjoint A)) v)) (Inner.inne …
  -/
  rw [adjoint_inner_right, adjoint_inner_left]
  /-
    🎉 no goals
  -/


/-- The adjoint of the composition of two operators is the composition of the two adjoints
in reverse order. -/
@[simp]
theorem adjoint_comp (A : F →ₗ[𝕜] G) (B : E →ₗ[𝕜] F) :
    LinearMap.adjoint (A ∘ₗ B) = LinearMap.adjoint B ∘ₗ LinearMap.adjoint A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 F
    inst✝ : FiniteDimensional 𝕜 G
    A : LinearMap (RingHom.id 𝕜) F G
    B : LinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (LinearMap.adjoint (A.comp B)) ((LinearMap.adjoint B).comp (LinearMap.adj …
  -/
  ext v
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 F
    inst✝ : FiniteDimensional 𝕜 G
    A : LinearMap (RingHom.id 𝕜) F G
    B : LinearMap (RingHom.id 𝕜) E F
    v : G
    ⊢ Eq ((LinearMap.adjoint (A.comp B)) v) (((LinearMap.adjoint B).comp (LinearMa …
  -/
  refine ext_inner_left 𝕜 fun w => ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : InnerProductSpace 𝕜 E
    inst✝⁴ : InnerProductSpace 𝕜 F
    inst✝³ : InnerProductSpace 𝕜 G
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 F
    inst✝ : FiniteDimensional 𝕜 G
    A : LinearMap (RingHom.id 𝕜) F G
    B : LinearMap (RingHom.id 𝕜) E F
    v : G
    w : E
    ⊢ Eq (Inner.inner w ((LinearMap.adjoint (A.comp B)) v)) (Inner.inner w (((Line …
  -/
  simp only [adjoint_inner_right, LinearMap.coe_comp, Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- The adjoint is unique: a map `A` is the adjoint of `B` iff it satisfies `⟪A x, y⟫ = ⟪x, B y⟫`
for all `x` and `y`. -/
theorem eq_adjoint_iff (A : E →ₗ[𝕜] F) (B : F →ₗ[𝕜] E) :
    A = LinearMap.adjoint B ↔ ∀ x y, ⟪A x, y⟫ = ⟪x, B y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    ⊢ Iff (Eq A (LinearMap.adjoint B)) (∀ (x : E) (y : F), Eq (Inner.inner (A x) y …
  -/
  refine ⟨fun h x y => by rw [h, adjoint_inner_left], fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (x : E) (y : F), Eq (Inner.inner (A x) y) (Inner.inner x (B y))
    ⊢ Eq A (LinearMap.adjoint B)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (x : E) (y : F), Eq (Inner.inner (A x) y) (Inner.inner x (B y))
    x : E
    ⊢ Eq (A x) ((LinearMap.adjoint B) x)
  -/
  exact ext_inner_right 𝕜 fun y => by simp only [adjoint_inner_left, h x y]
  /-
    🎉 no goals
  -/


/-- The adjoint is unique: a map `A` is the adjoint of `B` iff it satisfies `⟪A x, y⟫ = ⟪x, B y⟫`
for all basis vectors `x` and `y`. -/
theorem eq_adjoint_iff_basis {ι₁ : Type*} {ι₂ : Type*} (b₁ : Basis ι₁ 𝕜 E) (b₂ : Basis ι₂ 𝕜 F)
    (A : E →ₗ[𝕜] F) (B : F →ₗ[𝕜] E) :
    A = LinearMap.adjoint B ↔ ∀ (i₁ : ι₁) (i₂ : ι₂), ⟪A (b₁ i₁), b₂ i₂⟫ = ⟪b₁ i₁, B (b₂ i₂)⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι₁ : Type u_5
    ι₂ : Type u_6
    b₁ : Basis ι₁ 𝕜 E
    b₂ : Basis ι₂ 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    ⊢ Iff (Eq A (LinearMap.adjoint B)) (∀ (i₁ : ι₁) (i₂ : ι₂), Eq (Inner.inner (A  …
  -/
  refine ⟨fun h x y => by rw [h, adjoint_inner_left], fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι₁ : Type u_5
    ι₂ : Type u_6
    b₁ : Basis ι₁ 𝕜 E
    b₂ : Basis ι₂ 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (i₁ : ι₁) (i₂ : ι₂), Eq (Inner.inner (A (b₁ i₁)) (b₂ i₂)) (Inner.inner ( …
    ⊢ Eq A (LinearMap.adjoint B)
  -/
  refine Basis.ext b₁ fun i₁ => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι₁ : Type u_5
    ι₂ : Type u_6
    b₁ : Basis ι₁ 𝕜 E
    b₂ : Basis ι₂ 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (i₁ : ι₁) (i₂ : ι₂), Eq (Inner.inner (A (b₁ i₁)) (b₂ i₂)) (Inner.inner ( …
    i₁ : ι₁
    ⊢ Eq (A (b₁ i₁)) ((LinearMap.adjoint B) (b₁ i₁))
  -/
  exact ext_inner_right_basis b₂ fun i₂ => by simp only [adjoint_inner_left, h i₁ i₂]
  /-
    🎉 no goals
  -/


theorem eq_adjoint_iff_basis_left {ι : Type*} (b : Basis ι 𝕜 E) (A : E →ₗ[𝕜] F) (B : F →ₗ[𝕜] E) :
    A = LinearMap.adjoint B ↔ ∀ i y, ⟪A (b i), y⟫ = ⟪b i, B y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι : Type u_5
    b : Basis ι 𝕜 E
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    ⊢ Iff (Eq A (LinearMap.adjoint B)) (∀ (i : ι) (y : F), Eq (Inner.inner (A (b i …
  -/
  refine ⟨fun h x y => by rw [h, adjoint_inner_left], fun h => Basis.ext b fun i => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι : Type u_5
    b : Basis ι 𝕜 E
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (i : ι) (y : F), Eq (Inner.inner (A (b i)) y) (Inner.inner (b i) (B y))
    i : ι
    ⊢ Eq (A (b i)) ((LinearMap.adjoint B) (b i))
  -/
  exact ext_inner_right 𝕜 fun y => by simp only [h i, adjoint_inner_left]
  /-
    🎉 no goals
  -/


theorem eq_adjoint_iff_basis_right {ι : Type*} (b : Basis ι 𝕜 F) (A : E →ₗ[𝕜] F) (B : F →ₗ[𝕜] E) :
    A = LinearMap.adjoint B ↔ ∀ i x, ⟪A x, b i⟫ = ⟪x, B (b i)⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι : Type u_5
    b : Basis ι 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    ⊢ Iff (Eq A (LinearMap.adjoint B)) (∀ (i : ι) (x : E), Eq (Inner.inner (A x) ( …
  -/
  refine ⟨fun h x y => by rw [h, adjoint_inner_left], fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι : Type u_5
    b : Basis ι 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (i : ι) (x : E), Eq (Inner.inner (A x) (b i)) (Inner.inner x (B (b i)))
    ⊢ Eq A (LinearMap.adjoint B)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    ι : Type u_5
    b : Basis ι 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    B : LinearMap (RingHom.id 𝕜) F E
    h : ∀ (i : ι) (x : E), Eq (Inner.inner (A x) (b i)) (Inner.inner x (B (b i)))
    x : E
    ⊢ Eq (A x) ((LinearMap.adjoint B) x)
  -/
  exact ext_inner_right_basis b fun i => by simp only [h i, adjoint_inner_left]
  /-
    🎉 no goals
  -/


/-- `E →ₗ[𝕜] E` is a star algebra with the adjoint as the star operation. -/
instance : Star (E →ₗ[𝕜] E) :=
  ⟨adjoint⟩


instance : InvolutiveStar (E →ₗ[𝕜] E) :=
  ⟨adjoint_adjoint⟩


instance : StarMul (E →ₗ[𝕜] E) :=
  ⟨adjoint_comp⟩


instance : StarRing (E →ₗ[𝕜] E) :=
  ⟨LinearEquiv.map_add adjoint⟩


instance : StarModule 𝕜 (E →ₗ[𝕜] E) :=
  ⟨LinearEquiv.map_smulₛₗ adjoint⟩


theorem star_eq_adjoint (A : E →ₗ[𝕜] E) : star A = LinearMap.adjoint A :=
  rfl


/-- A continuous linear operator is self-adjoint iff it is equal to its adjoint. -/
theorem isSelfAdjoint_iff' {A : E →ₗ[𝕜] E} : IsSelfAdjoint A ↔ LinearMap.adjoint A = A :=
  Iff.rfl


theorem isSymmetric_iff_isSelfAdjoint (A : E →ₗ[𝕜] E) : IsSymmetric A ↔ IsSelfAdjoint A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    A : LinearMap (RingHom.id 𝕜) E E
    ⊢ Iff A.IsSymmetric (IsSelfAdjoint A)
  -/
  rw [isSelfAdjoint_iff', IsSymmetric, ← LinearMap.eq_adjoint_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    A : LinearMap (RingHom.id 𝕜) E E
    ⊢ Iff (Eq A (LinearMap.adjoint A)) (Eq (LinearMap.adjoint A) A)
  -/
  exact eq_comm
  /-
    🎉 no goals
  -/


theorem isAdjointPair_inner (A : E →ₗ[𝕜] F) :
    IsAdjointPair (sesqFormOfInner : E →ₗ[𝕜] E →ₗ⋆[𝕜] 𝕜) (sesqFormOfInner : F →ₗ[𝕜] F →ₗ⋆[𝕜] 𝕜) A
      (LinearMap.adjoint A) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    ⊢ sesqFormOfInner.IsAdjointPair sesqFormOfInner ⇑A ⇑(LinearMap.adjoint A)
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    A : LinearMap (RingHom.id 𝕜) E F
    x : E
    y : F
    ⊢ Eq ((sesqFormOfInner (A x)) y) ((sesqFormOfInner x) ((LinearMap.adjoint A) y))
  -/
  simp only [sesqFormOfInner_apply_apply, adjoint_inner_left]
  /-
    🎉 no goals
  -/


/-- The Gram operator T†T is symmetric. -/
theorem isSymmetric_adjoint_mul_self (T : E →ₗ[𝕜] E) : IsSymmetric (LinearMap.adjoint T * T) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    ⊢ (HMul.hMul (LinearMap.adjoint T) T).IsSymmetric
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x y : E
    ⊢ Eq (Inner.inner ((HMul.hMul (LinearMap.adjoint T) T) x) y) (Inner.inner x (( …
  -/
  simp only [mul_apply, adjoint_inner_left, adjoint_inner_right]
  /-
    🎉 no goals
  -/


/-- The Gram operator T†T is a positive operator. -/
theorem re_inner_adjoint_mul_self_nonneg (T : E →ₗ[𝕜] E) (x : E) :
    0 ≤ re ⟪x, (LinearMap.adjoint T * T) x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x : E
    ⊢ LE.le 0 (RCLike.re (Inner.inner x ((HMul.hMul (LinearMap.adjoint T) T) x)))
  -/
  simp only [mul_apply, adjoint_inner_right, inner_self_eq_norm_sq_to_K]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x : E
    ⊢ LE.le 0 (RCLike.re (HPow.hPow (↑(Norm.norm (T x))) 2))
  -/
  norm_cast
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x : E
    ⊢ LE.le 0 (HPow.hPow (Norm.norm (T x)) 2)
  -/
  exact sq_nonneg _
  /-
    🎉 no goals
  -/


@[simp]
theorem im_inner_adjoint_mul_self_eq_zero (T : E →ₗ[𝕜] E) (x : E) :
    im ⟪x, LinearMap.adjoint T (T x)⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x : E
    ⊢ Eq (RCLike.im (Inner.inner x ((LinearMap.adjoint T) (T x)))) 0
  -/
  simp only [mul_apply, adjoint_inner_right, inner_self_eq_norm_sq_to_K]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    x : E
    ⊢ Eq (RCLike.im (HPow.hPow (↑(Norm.norm (T x))) 2)) 0
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem inner_map_map_iff_adjoint_comp_self (u : H →L[𝕜] K) :
    (∀ x y : H, ⟪u x, u y⟫_𝕜 = ⟪x, y⟫_𝕜) ↔ adjoint u ∘L u = 1 := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : RCLike 𝕜
    H : Type u_5
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : InnerProductSpace 𝕜 H
    inst✝³ : CompleteSpace H
    K : Type u_6
    inst✝² : NormedAddCommGroup K
    inst✝¹ : InnerProductSpace 𝕜 K
    inst✝ : CompleteSpace K
    u : ContinuousLinearMap (RingHom.id 𝕜) H K
    ⊢ Iff (∀ (x y : H), Eq (Inner.inner (u x) (u y)) (Inner.inner x y)) (Eq ((Cont …
  -/
  refine ⟨fun h ↦ ext fun x ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁶ : RCLike 𝕜
      H : Type u_5
      inst✝⁵ : NormedAddCommGroup H
      inst✝⁴ : InnerProductSpace 𝕜 H
      inst✝³ : CompleteSpace H
      K : Type u_6
      inst✝² : NormedAddCommGroup K
      inst✝¹ : InnerProductSpace 𝕜 K
      inst✝ : CompleteSpace K
      u : ContinuousLinearMap (RingHom.id 𝕜) H K
      h : ∀ (x y : H), Eq (Inner.inner (u x) (u y)) (Inner.inner x y)
      x : H
      ⊢ Eq (((ContinuousLinearMap.adjoint u).comp u) x) (1 x)
    -/
  · refine ext_inner_right 𝕜 fun y ↦ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁶ : RCLike 𝕜
      H : Type u_5
      inst✝⁵ : NormedAddCommGroup H
      inst✝⁴ : InnerProductSpace 𝕜 H
      inst✝³ : CompleteSpace H
      K : Type u_6
      inst✝² : NormedAddCommGroup K
      inst✝¹ : InnerProductSpace 𝕜 K
      inst✝ : CompleteSpace K
      u : ContinuousLinearMap (RingHom.id 𝕜) H K
      h : ∀ (x y : H), Eq (Inner.inner (u x) (u y)) (Inner.inner x y)
      x y : H
      ⊢ Eq (Inner.inner (((ContinuousLinearMap.adjoint u).comp u) x) y) (Inner.inner …
    -/
    simpa [star_eq_adjoint, adjoint_inner_left] using h x y
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁶ : RCLike 𝕜
      H : Type u_5
      inst✝⁵ : NormedAddCommGroup H
      inst✝⁴ : InnerProductSpace 𝕜 H
      inst✝³ : CompleteSpace H
      K : Type u_6
      inst✝² : NormedAddCommGroup K
      inst✝¹ : InnerProductSpace 𝕜 K
      inst✝ : CompleteSpace K
      u : ContinuousLinearMap (RingHom.id 𝕜) H K
      h : Eq ((ContinuousLinearMap.adjoint u).comp u) 1
      ⊢ ∀ (x y : H), Eq (Inner.inner (u x) (u y)) (Inner.inner x y)
    -/
  · simp [← adjoint_inner_left, ← comp_apply, h]
    /-
      🎉 no goals
    -/


theorem norm_map_iff_adjoint_comp_self (u : H →L[𝕜] K) :
    (∀ x : H, ‖u x‖ = ‖x‖) ↔ adjoint u ∘L u = 1 := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : RCLike 𝕜
    H : Type u_5
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : InnerProductSpace 𝕜 H
    inst✝³ : CompleteSpace H
    K : Type u_6
    inst✝² : NormedAddCommGroup K
    inst✝¹ : InnerProductSpace 𝕜 K
    inst✝ : CompleteSpace K
    u : ContinuousLinearMap (RingHom.id 𝕜) H K
    ⊢ Iff (∀ (x : H), Eq (Norm.norm (u x)) (Norm.norm x)) (Eq ((ContinuousLinearMa …
  -/
  rw [LinearMap.norm_map_iff_inner_map_map u, u.inner_map_map_iff_adjoint_comp_self]
  /-
    🎉 no goals
  -/


@[simp]
lemma _root_.LinearIsometryEquiv.adjoint_eq_symm (e : H ≃ₗᵢ[𝕜] K) :
    adjoint (e : H →L[𝕜] K) = e.symm :=
  let e' := (e : H →L[𝕜] K)
  calc
    adjoint e' = adjoint e' ∘L (e' ∘L e.symm) := by
      /-
        𝕜 : Type u_1
        inst✝⁶ : RCLike 𝕜
        H : Type u_5
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        K : Type u_6
        inst✝² : NormedAddCommGroup K
        inst✝¹ : InnerProductSpace 𝕜 K
        inst✝ : CompleteSpace K
        e : LinearIsometryEquiv (RingHom.id 𝕜) H K
        e' : ContinuousLinearMap (RingHom.id 𝕜) H K := ↑{ toLinearEquiv := e.toLinearE …
        ⊢ Eq (ContinuousLinearMap.adjoint e') ((ContinuousLinearMap.adjoint e').comp ( …
      -/
      convert (adjoint e').comp_id.symm
      /-
        case h.e'_3.h.e'_24
        𝕜 : Type u_1
        inst✝⁶ : RCLike 𝕜
        H : Type u_5
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        K : Type u_6
        inst✝² : NormedAddCommGroup K
        inst✝¹ : InnerProductSpace 𝕜 K
        inst✝ : CompleteSpace K
        e : LinearIsometryEquiv (RingHom.id 𝕜) H K
        e' : ContinuousLinearMap (RingHom.id 𝕜) H K := ↑{ toLinearEquiv := e.toLinearE …
        ⊢ Eq (e'.comp ↑{ toLinearEquiv := e.symm.toLinearEquiv, continuous_toFun := ⋯, …
      -/
      ext
      /-
        case h.e'_3.h.e'_24.h
        𝕜 : Type u_1
        inst✝⁶ : RCLike 𝕜
        H : Type u_5
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        K : Type u_6
        inst✝² : NormedAddCommGroup K
        inst✝¹ : InnerProductSpace 𝕜 K
        inst✝ : CompleteSpace K
        e : LinearIsometryEquiv (RingHom.id 𝕜) H K
        e' : ContinuousLinearMap (RingHom.id 𝕜) H K := ↑{ toLinearEquiv := e.toLinearE …
        x✝ : K
        ⊢ Eq ((e'.comp ↑{ toLinearEquiv := e.symm.toLinearEquiv, continuous_toFun := ⋯ …
      -/
      simp [e']
      /-
        🎉 no goals
      -/
    _ = e.symm := by
      /-
        𝕜 : Type u_1
        inst✝⁶ : RCLike 𝕜
        H : Type u_5
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        K : Type u_6
        inst✝² : NormedAddCommGroup K
        inst✝¹ : InnerProductSpace 𝕜 K
        inst✝ : CompleteSpace K
        e : LinearIsometryEquiv (RingHom.id 𝕜) H K
        e' : ContinuousLinearMap (RingHom.id 𝕜) H K := ↑{ toLinearEquiv := e.toLinearE …
        ⊢ Eq ((ContinuousLinearMap.adjoint e').comp (e'.comp ↑{ toLinearEquiv := e.sym …
      -/
      rw [← comp_assoc, norm_map_iff_adjoint_comp_self e' |>.mp e.norm_map]
      /-
        𝕜 : Type u_1
        inst✝⁶ : RCLike 𝕜
        H : Type u_5
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        K : Type u_6
        inst✝² : NormedAddCommGroup K
        inst✝¹ : InnerProductSpace 𝕜 K
        inst✝ : CompleteSpace K
        e : LinearIsometryEquiv (RingHom.id 𝕜) H K
        e' : ContinuousLinearMap (RingHom.id 𝕜) H K := ↑{ toLinearEquiv := e.toLinearE …
        ⊢ Eq (ContinuousLinearMap.comp 1 ↑{ toLinearEquiv := e.symm.toLinearEquiv, con …
      -/
      exact (e.symm : K →L[𝕜] H).id_comp
      /-
        🎉 no goals
      -/


@[simp]
lemma _root_.LinearIsometryEquiv.star_eq_symm (e : H ≃ₗᵢ[𝕜] H) :
    star (e : H →L[𝕜] H) = e.symm :=
  e.adjoint_eq_symm


theorem norm_map_of_mem_unitary {u : H →L[𝕜] H} (hu : u ∈ unitary (H →L[𝕜] H)) (x : H) :
    ‖u x‖ = ‖x‖ :=
  -- Elaborates faster with this broken out https://github.com/leanprover-community/mathlib4/issues/11299
  have := unitary.star_mul_self_of_mem hu
  u.norm_map_iff_adjoint_comp_self.mpr this x


theorem inner_map_map_of_mem_unitary {u : H →L[𝕜] H} (hu : u ∈ unitary (H →L[𝕜] H)) (x y : H) :
    ⟪u x, u y⟫_𝕜 = ⟪x, y⟫_𝕜 :=
  -- Elaborates faster with this broken out https://github.com/leanprover-community/mathlib4/issues/11299
  have := unitary.star_mul_self_of_mem hu
  u.inner_map_map_iff_adjoint_comp_self.mpr this x y


theorem norm_map (u : unitary (H →L[𝕜] H)) (x : H) : ‖(u : H →L[𝕜] H) x‖ = ‖x‖ :=
  u.val.norm_map_of_mem_unitary u.property x


theorem inner_map_map (u : unitary (H →L[𝕜] H)) (x y : H) :
    ⟪(u : H →L[𝕜] H) x, (u : H →L[𝕜] H) y⟫_𝕜 = ⟪x, y⟫_𝕜 :=
  u.val.inner_map_map_of_mem_unitary u.property x y


/-- The unitary elements of continuous linear maps on a Hilbert space coincide with the linear
isometric equivalences on that Hilbert space. -/
noncomputable def linearIsometryEquiv : unitary (H →L[𝕜] H) ≃* (H ≃ₗᵢ[𝕜] H) where
  toFun u :=
    { (u : H →L[𝕜] H) with
      norm_map' := norm_map u
      invFun := ↑(star u)
      left_inv := fun x ↦ congr($(star_mul_self u).val x)
      right_inv := fun x ↦ congr($(mul_star_self u).val x) }
  invFun e :=
    { val := e
      property := by
        let e' : (H →L[𝕜] H)ˣ :=
          { val := (e : H →L[𝕜] H)
            inv := (e.symm : H →L[𝕜] H)
            val_inv := by ext; simp
            inv_val := by ext; simp }
        exact IsUnit.mem_unitary_of_star_mul_self ⟨e', rfl⟩ <|
          (e : H →L[𝕜] H).norm_map_iff_adjoint_comp_self.mp e.norm_map }
  left_inv _ := Subtype.ext rfl
  right_inv _ := LinearIsometryEquiv.ext fun _ ↦ rfl
                     /-
                       𝕜 : Type u_1
                       E : Type u_2
                       F : Type u_3
                       G : Type u_4
                       inst✝⁹ : RCLike 𝕜
                       inst✝⁸ : NormedAddCommGroup E
                       inst✝⁷ : NormedAddCommGroup F
                       inst✝⁶ : NormedAddCommGroup G
                       inst✝⁵ : InnerProductSpace 𝕜 E
                       inst✝⁴ : InnerProductSpace 𝕜 F
                       inst✝³ : InnerProductSpace 𝕜 G
                       H : Type u_5
                       inst✝² : NormedAddCommGroup H
                       inst✝¹ : InnerProductSpace 𝕜 H
                       inst✝ : CompleteSpace H
                       u v : Subtype fun x => Membership.mem (unitary (ContinuousLinearMap (RingHom.i …
                       ⊢ Eq
                           ({
                                 toFun := fun u =>
                                   let __src := ↑u;
                                   { toLinearMap := ↑__src, invFun := ⇑↑(Star.star u), left_inv := ⋯, …
                                 invFun := fun e => ⟨↑{ toLinearEquiv := e.toLinearEquiv, continuous_ …
                             (HMul.hMul u v))
                           (HMul.hMul
                             ({
                                   toFun := fun u =>
                                     let __src := ↑u;
                                     { toLinearMap := ↑__src, invFun := ⇑↑(Star.star u), left_inv :=  …
                                   invFun := fun e => ⟨↑{ toLinearEquiv := e.toLinearEquiv, continuou …
                               u)
                             ({
                                   toFun := fun u =>
                                     let __src := ↑u;
                                     { toLinearMap := ↑__src, invFun := ⇑↑(Star.star u), left_inv :=  …
                                   invFun := fun e => ⟨↑{ toLinearEquiv := e.toLinearEquiv, continuou …
                               v))
                     -/
  map_mul' u v := by ext; rfl
                          /-
                            🎉 no goals
                          -/


@[simp]
lemma linearIsometryEquiv_coe_apply (u : unitary (H →L[𝕜] H)) :
    linearIsometryEquiv u = (u : H →L[𝕜] H) :=
  rfl


@[simp]
lemma linearIsometryEquiv_coe_symm_apply (e : H ≃ₗᵢ[𝕜] H) :
    linearIsometryEquiv.symm e = (e : H →L[𝕜] H) :=
  rfl


/-- The linear map associated to the conjugate transpose of a matrix corresponding to two
orthonormal bases is the adjoint of the linear map associated to the matrix. -/
lemma Matrix.toLin_conjTranspose (A : Matrix m n 𝕜) :
    toLin v₂.toBasis v₁.toBasis Aᴴ = adjoint (toLin v₁.toBasis v₂.toBasis A) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : InnerProductSpace 𝕜 E
    inst✝⁶ : InnerProductSpace 𝕜 F
    m : Type u_5
    n : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    v₁ : OrthonormalBasis n 𝕜 E
    v₂ : OrthonormalBasis m 𝕜 F
    A : Matrix m n 𝕜
    ⊢ Eq ((Matrix.toLin v₂.toBasis v₁.toBasis) A.conjTranspose) (LinearMap.adjoint …
  -/
  refine eq_adjoint_iff_basis v₂.toBasis v₁.toBasis _ _ |>.mpr fun i j ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : InnerProductSpace 𝕜 E
    inst✝⁶ : InnerProductSpace 𝕜 F
    m : Type u_5
    n : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : FiniteDimensional 𝕜 F
    v₁ : OrthonormalBasis n 𝕜 E
    v₂ : OrthonormalBasis m 𝕜 F
    A : Matrix m n 𝕜
    i : m
    j : n
    ⊢ Eq (Inner.inner (((Matrix.toLin v₂.toBasis v₁.toBasis) A.conjTranspose) (v₂. …
  -/
  simp_rw [toLin_self]
  simp [sum_inner, inner_smul_left, inner_sum, inner_smul_right,
    orthonormal_iff_ite.mp v₁.orthonormal, orthonormal_iff_ite.mp v₂.orthonormal]


/-- The matrix associated to the adjoint of a linear map corresponding to two orthonormal bases
is the conjugate transpose of the matrix associated to the linear map. -/
lemma LinearMap.toMatrix_adjoint (f : E →ₗ[𝕜] F) :
    toMatrix v₂.toBasis v₁.toBasis (adjoint f) = (toMatrix v₁.toBasis v₂.toBasis f)ᴴ :=
                                                 /-
                                                   𝕜 : Type u_1
                                                   E : Type u_2
                                                   F : Type u_3
                                                   inst✝¹⁰ : RCLike 𝕜
                                                   inst✝⁹ : NormedAddCommGroup E
                                                   inst✝⁸ : NormedAddCommGroup F
                                                   inst✝⁷ : InnerProductSpace 𝕜 E
                                                   inst✝⁶ : InnerProductSpace 𝕜 F
                                                   m : Type u_5
                                                   n : Type u_6
                                                   inst✝⁵ : Fintype m
                                                   inst✝⁴ : DecidableEq m
                                                   inst✝³ : Fintype n
                                                   inst✝² : DecidableEq n
                                                   inst✝¹ : FiniteDimensional 𝕜 E
                                                   inst✝ : FiniteDimensional 𝕜 F
                                                   v₁ : OrthonormalBasis n 𝕜 E
                                                   v₂ : OrthonormalBasis m 𝕜 F
                                                   f : LinearMap (RingHom.id 𝕜) E F
                                                   ⊢ Eq ((Matrix.toLin v₂.toBasis v₁.toBasis) ((LinearMap.toMatrix v₂.toBasis v₁. …
                                                 -/
  toLin v₂.toBasis v₁.toBasis |>.injective <| by simp [toLin_conjTranspose]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The star algebra equivalence between the linear endomorphisms of finite-dimensional inner
product space and square matrices induced by the choice of an orthonormal basis. -/
@[simps]
def LinearMap.toMatrixOrthonormal : (E →ₗ[𝕜] E) ≃⋆ₐ[𝕜] Matrix n n 𝕜 :=
  { LinearMap.toMatrix v₁.toBasis v₁.toBasis with
    map_mul' := LinearMap.toMatrix_mul v₁.toBasis
    map_star' := LinearMap.toMatrix_adjoint v₁ v₁ }


/-- The adjoint of the linear map associated to a matrix is the linear map associated to the
conjugate transpose of that matrix. -/
theorem Matrix.toEuclideanLin_conjTranspose_eq_adjoint (A : Matrix m n 𝕜) :
    Matrix.toEuclideanLin A.conjTranspose = LinearMap.adjoint (Matrix.toEuclideanLin A) :=
  A.toLin_conjTranspose (EuclideanSpace.basisFun n 𝕜) (EuclideanSpace.basisFun m 𝕜)


