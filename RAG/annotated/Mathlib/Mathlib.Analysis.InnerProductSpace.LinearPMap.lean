local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- An operator `T` is a formal adjoint of `S` if for all `x` in the domain of `T` and `y` in the
domain of `S`, we have that `⟪T x, y⟫ = ⟪x, S y⟫`. -/
def IsFormalAdjoint (T : E →ₗ.[𝕜] F) (S : F →ₗ.[𝕜] E) : Prop :=
  ∀ (x : T.domain) (y : S.domain), ⟪T x, y⟫ = ⟪(x : E), S y⟫


@[symm]
protected theorem IsFormalAdjoint.symm (h : T.IsFormalAdjoint S) :
    S.IsFormalAdjoint T := fun y _ => by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    S : LinearPMap 𝕜 F E
    h : T.IsFormalAdjoint S
    y : Subtype fun x => Membership.mem S.domain x
    x✝ : Subtype fun x => Membership.mem T.domain x
    ⊢ Eq (Inner.inner (↑S y) ↑x✝) (Inner.inner (↑y) (↑T x✝))
  -/
  rw [← inner_conj_symm, ← inner_conj_symm (y : F), h]
  /-
    🎉 no goals
  -/


/-- The domain of the adjoint operator.

This definition is needed to construct the adjoint operator and the preferred version to use is
`T.adjoint.domain` instead of `T.adjointDomain`. -/
def adjointDomain : Submodule 𝕜 F where
  carrier := {y | Continuous ((innerₛₗ 𝕜 y).comp T.toFun)}
  zero_mem' := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      T : LinearPMap 𝕜 E F
      S : LinearPMap 𝕜 F E
      ⊢ Membership.mem { carrier := setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).comp …
    -/
    rw [Set.mem_setOf_eq, LinearMap.map_zero, LinearMap.zero_comp]
                       /-
                         𝕜 : Type u_1
                         E : Type u_2
                         F : Type u_3
                         inst✝⁴ : RCLike 𝕜
                         inst✝³ : NormedAddCommGroup E
                         inst✝² : InnerProductSpace 𝕜 E
                         inst✝¹ : NormedAddCommGroup F
                         inst✝ : InnerProductSpace 𝕜 F
                         T : LinearPMap 𝕜 E F
                         S : LinearPMap 𝕜 F E
                         a✝ b✝ : F
                         hx : Membership.mem (setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).comp T.toFun) …
                         hy : Membership.mem (setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).comp T.toFun) …
                         ⊢ Membership.mem (setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).comp T.toFun)) ( …
                       -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      T : LinearPMap 𝕜 E F
      S : LinearPMap 𝕜 F E
      ⊢ Continuous ⇑0
    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    exact continuous_zero
    /-
      🎉 no goals
    -/
  add_mem' hx hy := by rw [Set.mem_setOf_eq, LinearMap.map_add] at *; exact hx.add hy
  smul_mem' a x hx := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      T : LinearPMap 𝕜 E F
      S : LinearPMap 𝕜 F E
      a : 𝕜
      x : F
      hx : Membership.mem { carrier := setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).c …
      ⊢ Membership.mem { carrier := setOf fun y => Continuous ⇑(((innerₛₗ 𝕜) y).comp …
    -/
    rw [Set.mem_setOf_eq, LinearMap.map_smulₛₗ] at *
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      T : LinearPMap 𝕜 E F
      S : LinearPMap 𝕜 F E
      a : 𝕜
      x : F
      hx : Continuous ⇑(((innerₛₗ 𝕜) x).comp T.toFun)
      ⊢ Continuous ⇑((HSMul.hSMul ((starRingEnd 𝕜) a) ((innerₛₗ 𝕜) x)).comp T.toFun)
    -/
    exact hx.const_smul (conj a)
    /-
      🎉 no goals
    -/


/-- The operator `fun x ↦ ⟪y, T x⟫` considered as a continuous linear operator
from `T.adjointDomain` to `𝕜`. -/
def adjointDomainMkCLM (y : T.adjointDomain) : T.domain →L[𝕜] 𝕜 :=
  ⟨(innerₛₗ 𝕜 (y : F)).comp T.toFun, y.prop⟩


theorem adjointDomainMkCLM_apply (y : T.adjointDomain) (x : T.domain) :
    adjointDomainMkCLM T y x = ⟪(y : F), T x⟫ :=
  rfl


/-- The unique continuous extension of the operator `adjointDomainMkCLM` to `E`. -/
def adjointDomainMkCLMExtend (y : T.adjointDomain) : E →L[𝕜] 𝕜 :=
  (T.adjointDomainMkCLM y).extend (Submodule.subtypeL T.domain) hT.denseRange_val
    isUniformEmbedding_subtype_val.isUniformInducing


@[simp]
theorem adjointDomainMkCLMExtend_apply (y : T.adjointDomain) (x : T.domain) :
    adjointDomainMkCLMExtend hT y (x : E) = ⟪(y : F), T x⟫ :=
  ContinuousLinearMap.extend_eq _ _ _ _ _


/-- The adjoint as a linear map from its domain to `E`.

This is an auxiliary definition needed to define the adjoint operator as a `LinearPMap` without
the assumption that `T.domain` is dense. -/
def adjointAux : T.adjointDomain →ₗ[𝕜] E where
  toFun y := (InnerProductSpace.toDual 𝕜 E).symm (adjointDomainMkCLMExtend hT y)
  map_add' x y :=
    hT.eq_of_inner_left fun _ => by
      simp only [inner_add_left, Submodule.coe_add, InnerProductSpace.toDual_symm_apply,
        adjointDomainMkCLMExtend_apply]
  map_smul' _ _ :=
    hT.eq_of_inner_left fun _ => by
      simp only [inner_smul_left, Submodule.coe_smul_of_tower, RingHom.id_apply,
        InnerProductSpace.toDual_symm_apply, adjointDomainMkCLMExtend_apply]


theorem adjointAux_inner (y : T.adjointDomain) (x : T.domain) :
    ⟪adjointAux hT y, x⟫ = ⟪(y : F), T x⟫ := by
  simp only [adjointAux, LinearMap.coe_mk, InnerProductSpace.toDual_symm_apply,
    adjointDomainMkCLMExtend_apply]
  -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5026):
  -- mathlib3 was finished here
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    hT : Dense ↑T.domain
    inst✝ : CompleteSpace E
    y : Subtype fun x => Membership.mem T.adjointDomain x
    x : Subtype fun x => Membership.mem T.domain x
    ⊢ Eq (Inner.inner ({ toFun := fun y => (InnerProductSpace.toDual 𝕜 E).symm (Li …
  -/
  simp only [AddHom.coe_mk, InnerProductSpace.toDual_symm_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    hT : Dense ↑T.domain
    inst✝ : CompleteSpace E
    y : Subtype fun x => Membership.mem T.adjointDomain x
    x : Subtype fun x => Membership.mem T.domain x
    ⊢ Eq ((LinearPMap.adjointDomainMkCLMExtend hT y) ↑x) (Inner.inner (↑y) (↑T x))
  -/
  rw [adjointDomainMkCLMExtend_apply]
  /-
    🎉 no goals
  -/


theorem adjointAux_unique (y : T.adjointDomain) {x₀ : E}
    (hx₀ : ∀ x : T.domain, ⟪x₀, x⟫ = ⟪(y : F), T x⟫) : adjointAux hT y = x₀ :=
  hT.eq_of_inner_left fun v => (adjointAux_inner hT _ _).trans (hx₀ v).symm


open scoped Classical in
/-- The adjoint operator as a partially defined linear operator. -/
def adjoint : F →ₗ.[𝕜] E where
  domain := T.adjointDomain
  toFun := if hT : Dense (T.domain : Set E) then adjointAux hT else 0


scoped postfix:1024 "†" => LinearPMap.adjoint


theorem mem_adjoint_domain_iff (y : F) : y ∈ T†.domain ↔ Continuous ((innerₛₗ 𝕜 y).comp T.toFun) :=
  Iff.rfl


theorem mem_adjoint_domain_of_exists (y : F) (h : ∃ w : E, ∀ x : T.domain, ⟪w, x⟫ = ⟪y, T x⟫) :
    y ∈ T†.domain := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    inst✝ : CompleteSpace E
    y : F
    h : Exists fun w => ∀ (x : Subtype fun x => Membership.mem T.domain x), Eq (In …
    ⊢ Membership.mem T.adjoint.domain y
  -/
  cases' h with w hw
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    inst✝ : CompleteSpace E
    y : F
    w : E
    hw : ∀ (x : Subtype fun x => Membership.mem T.domain x), Eq (Inner.inner w ↑x) …
    ⊢ Membership.mem T.adjoint.domain y
  -/
  rw [T.mem_adjoint_domain_iff]
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    inst✝ : CompleteSpace E
    y : F
    w : E
    hw : ∀ (x : Subtype fun x => Membership.mem T.domain x), Eq (Inner.inner w ↑x) …
    ⊢ Continuous ⇑(((innerₛₗ 𝕜) y).comp T.toFun)
  -/
  have : Continuous ((innerSL 𝕜 w).comp T.domain.subtypeL) := by fun_prop
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    inst✝ : CompleteSpace E
    y : F
    w : E
    hw : ∀ (x : Subtype fun x => Membership.mem T.domain x), Eq (Inner.inner w ↑x) …
    this : Continuous ⇑(((innerSL 𝕜) w).comp T.domain.subtypeL)
    ⊢ Continuous ⇑(((innerₛₗ 𝕜) y).comp T.toFun)
  -/
  convert this using 1
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    T : LinearPMap 𝕜 E F
    inst✝ : CompleteSpace E
    y : F
    w : E
    hw : ∀ (x : Subtype fun x => Membership.mem T.domain x), Eq (Inner.inner w ↑x) …
    this : Continuous ⇑(((innerSL 𝕜) w).comp T.domain.subtypeL)
    ⊢ Eq ⇑(((innerₛₗ 𝕜) y).comp T.toFun) ⇑(((innerSL 𝕜) w).comp T.domain.subtypeL)
  -/
  exact funext fun x => (hw x).symm
  /-
    🎉 no goals
  -/


theorem adjoint_apply_of_not_dense (hT : ¬Dense (T.domain : Set E)) (y : T†.domain) : T† y = 0 := by
  classical
  change (if hT : Dense (T.domain : Set E) then adjointAux hT else 0) y = _
  simp only [hT, not_false_iff, dif_neg, LinearMap.zero_apply]


theorem adjoint_apply_of_dense (y : T†.domain) : T† y = adjointAux hT y := by
  classical
  change (if hT : Dense (T.domain : Set E) then adjointAux hT else 0) y = _
  simp only [hT, dif_pos, LinearMap.coe_mk]


include hT in
theorem adjoint_apply_eq (y : T†.domain) {x₀ : E} (hx₀ : ∀ x : T.domain, ⟪x₀, x⟫ = ⟪(y : F), T x⟫) :
    T† y = x₀ :=
  (adjoint_apply_of_dense hT y).symm ▸ adjointAux_unique hT _ hx₀


include hT in
/-- The fundamental property of the adjoint. -/
theorem adjoint_isFormalAdjoint : T†.IsFormalAdjoint T := fun x =>
  (adjoint_apply_of_dense hT x).symm ▸ adjointAux_inner hT x


include hT in
/-- The adjoint is maximal in the sense that it contains every formal adjoint. -/
theorem IsFormalAdjoint.le_adjoint (h : T.IsFormalAdjoint S) : S ≤ T† :=
  ⟨-- Trivially, every `x : S.domain` is in `T.adjoint.domain`
  fun x hx =>
    mem_adjoint_domain_of_exists _
      ⟨S ⟨x, hx⟩, h.symm ⟨x, hx⟩⟩,-- Equality on `S.domain` follows from equality
  -- `⟪v, S x⟫ = ⟪v, T.adjoint y⟫` for all `v : T.domain`:
                                                    /-
                                                      𝕜 : Type u_1
                                                      E : Type u_2
                                                      F : Type u_3
                                                      inst✝⁵ : RCLike 𝕜
                                                      inst✝⁴ : NormedAddCommGroup E
                                                      inst✝³ : InnerProductSpace 𝕜 E
                                                      inst✝² : NormedAddCommGroup F
                                                      inst✝¹ : InnerProductSpace 𝕜 F
                                                      T : LinearPMap 𝕜 E F
                                                      S : LinearPMap 𝕜 F E
                                                      hT : Dense ↑T.domain
                                                      inst✝ : CompleteSpace E
                                                      h : T.IsFormalAdjoint S
                                                      x✝² : Subtype fun x => Membership.mem S.domain x
                                                      x✝¹ : Subtype fun x => Membership.mem T.adjoint.domain x
                                                      hxy : Eq ↑x✝² ↑x✝¹
                                                      x✝ : Subtype fun x => Membership.mem T.domain x
                                                      ⊢ Eq (Inner.inner (↑S x✝²) ↑x✝) (Inner.inner (↑x✝¹) (↑T x✝))
                                                    -/
  fun _ _ hxy => (adjoint_apply_eq hT _ fun _ => by rw [h.symm, hxy]).symm⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Restricting `A` to a dense submodule and taking the `LinearPMap.adjoint` is the same
as taking the `ContinuousLinearMap.adjoint` interpreted as a `LinearPMap`. -/
theorem toPMap_adjoint_eq_adjoint_toPMap_of_dense (hp : Dense (p : Set E)) :
    (A.toPMap p).adjoint = A.adjoint.toPMap ⊤ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    p : Submodule 𝕜 E
    hp : Dense ↑p
    ⊢ Eq ((↑A).toPMap p).adjoint ((↑(ContinuousLinearMap.adjoint A)).toPMap Top.top)
  -/
  ext x y hxy
  · simp only [LinearMap.toPMap_domain, Submodule.mem_top, iff_true,
      LinearPMap.mem_adjoint_domain_iff, LinearMap.coe_comp, innerₛₗ_apply_coe]
    /-
      case h.h
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      p : Submodule 𝕜 E
      hp : Dense ↑p
      x : F
      ⊢ Continuous (Function.comp (fun w => Inner.inner x w) ⇑((↑A).toPMap p).toFun)
    -/
    exact ((innerSL 𝕜 x).comp <| A.comp <| Submodule.subtypeL _).cont
    /-
      🎉 no goals
    -/
  /-
    case h'
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    A : ContinuousLinearMap (RingHom.id 𝕜) E F
    p : Submodule 𝕜 E
    hp : Dense ↑p
    x : Subtype fun x => Membership.mem ((↑A).toPMap p).adjoint.domain x
    y : Subtype fun x => Membership.mem ((↑(ContinuousLinearMap.adjoint A)).toPMap …
    hxy : Eq ↑x ↑y
    ⊢ Eq (↑((↑A).toPMap p).adjoint x) (↑((↑(ContinuousLinearMap.adjoint A)).toPMap …
  -/
  refine LinearPMap.adjoint_apply_eq ?_ _ fun v => ?_
  · -- Porting note: was simply `hp` as an argument above
    /-
      case h'.refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      p : Submodule 𝕜 E
      hp : Dense ↑p
      x : Subtype fun x => Membership.mem ((↑A).toPMap p).adjoint.domain x
      y : Subtype fun x => Membership.mem ((↑(ContinuousLinearMap.adjoint A)).toPMap …
      hxy : Eq ↑x ↑y
      ⊢ Dense ↑((↑A).toPMap p).domain
    -/
    simpa using hp
    /-
      🎉 no goals
    -/
    /-
      case h'.refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      A : ContinuousLinearMap (RingHom.id 𝕜) E F
      p : Submodule 𝕜 E
      hp : Dense ↑p
      x : Subtype fun x => Membership.mem ((↑A).toPMap p).adjoint.domain x
      y : Subtype fun x => Membership.mem ((↑(ContinuousLinearMap.adjoint A)).toPMap …
      hxy : Eq ↑x ↑y
      v : Subtype fun x => Membership.mem ((↑A).toPMap p).domain x
      ⊢ Eq (Inner.inner (↑((↑(ContinuousLinearMap.adjoint A)).toPMap Top.top) y) ↑v) …
    -/
  · simp only [adjoint_inner_left, hxy, LinearMap.toPMap_apply, coe_coe]
    /-
      🎉 no goals
    -/


instance instStar : Star (E →ₗ.[𝕜] E) where
  star := fun A ↦ A.adjoint


theorem isSelfAdjoint_def : IsSelfAdjoint A ↔ A† = A := Iff.rfl


/-- Every self-adjoint `LinearPMap` has dense domain.

This is not true by definition since we define the adjoint without the assumption that the
domain is dense, but the choice of the junk value implies that a `LinearPMap` cannot be self-adjoint
if it does not have dense domain. -/
theorem _root_.IsSelfAdjoint.dense_domain (hA : IsSelfAdjoint A) : Dense (A.domain : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : LinearPMap 𝕜 E E
    hA : IsSelfAdjoint A
    ⊢ Dense ↑A.domain
  -/
  by_contra h
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : LinearPMap 𝕜 E E
    hA : IsSelfAdjoint A
    h : Not (Dense ↑A.domain)
    ⊢ False
  -/
  rw [isSelfAdjoint_def] at hA
  have h' : A.domain = ⊤ := by
    rw [← hA, Submodule.eq_top_iff']
    intro x
    rw [mem_adjoint_domain_iff, ← hA]
    refine (innerSL 𝕜 x).cont.comp ?_
    simp only [adjoint, h]
    exact continuous_const
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    A : LinearPMap 𝕜 E E
    hA : Eq A.adjoint A
    h : Not (Dense ↑A.domain)
    h' : Eq A.domain Top.top
    ⊢ False
  -/
  simp [h'] at h
  /-
    🎉 no goals
  -/


