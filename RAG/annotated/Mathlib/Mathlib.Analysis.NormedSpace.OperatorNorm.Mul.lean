/-- Multiplication in a non-unital normed algebra as a continuous bilinear map. -/
def mul : 𝕜' →L[𝕜] 𝕜' →L[𝕜] 𝕜' :=
                                                     /-
                                                       𝕜 : Type u_1
                                                       E : Type u_2
                                                       inst✝⁶ : NontriviallyNormedField 𝕜
                                                       inst✝⁵ : SeminormedAddCommGroup E
                                                       inst✝⁴ : NormedSpace 𝕜 E
                                                       𝕜' : Type u_3
                                                       inst✝³ : NonUnitalSeminormedRing 𝕜'
                                                       inst✝² : NormedSpace 𝕜 𝕜'
                                                       inst✝¹ : IsScalarTower 𝕜 𝕜' 𝕜'
                                                       inst✝ : SMulCommClass 𝕜 𝕜' 𝕜'
                                                       x y : 𝕜'
                                                       ⊢ LE.le (Norm.norm (((LinearMap.mul 𝕜 𝕜') x) y)) (HMul.hMul (HMul.hMul 1 (Norm …
                                                     -/
  (LinearMap.mul 𝕜 𝕜').mkContinuous₂ 1 fun x y => by simpa using norm_mul_le x y
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem mul_apply' (x y : 𝕜') : mul 𝕜 𝕜' x y = x * y :=
  rfl


@[simp]
theorem opNorm_mul_apply_le (x : 𝕜') : ‖mul 𝕜 𝕜' x‖ ≤ ‖x‖ :=
  opNorm_le_bound _ (norm_nonneg x) (norm_mul_le x)


@[deprecated (since := "2024-02-02")] alias op_norm_mul_apply_le := opNorm_mul_apply_le


theorem opNorm_mul_le : ‖mul 𝕜 𝕜'‖ ≤ 1 :=
  LinearMap.mkContinuous₂_norm_le _ zero_le_one _


@[deprecated (since := "2024-02-02")] alias op_norm_mul_le := opNorm_mul_le


/-- Multiplication on the left in a non-unital normed algebra `𝕜'` as a non-unital algebra
homomorphism into the algebra of *continuous* linear maps. This is the left regular representation
of `A` acting on itself.

This has more algebraic structure than `ContinuousLinearMap.mul`, but there is no longer continuity
bundled in the first coordinate.  An alternative viewpoint is that this upgrades
`NonUnitalAlgHom.lmul` from a homomorphism into linear maps to a homomorphism into *continuous*
linear maps. -/
def _root_.NonUnitalAlgHom.Lmul : 𝕜' →ₙₐ[𝕜] 𝕜' →L[𝕜] 𝕜' :=
  { mul 𝕜 𝕜' with
    map_mul' := fun _ _ ↦ ext fun _ ↦ mul_assoc _ _ _
    map_zero' := ext fun _ ↦ zero_mul _ }


variable {𝕜 𝕜'} in
@[simp]
theorem _root_.NonUnitalAlgHom.coe_Lmul : ⇑(NonUnitalAlgHom.Lmul 𝕜 𝕜') = mul 𝕜 𝕜' :=
  rfl


/-- Simultaneous left- and right-multiplication in a non-unital normed algebra, considered as a
continuous trilinear map. This is akin to its non-continuous version `LinearMap.mulLeftRight`,
but there is a minor difference: `LinearMap.mulLeftRight` is uncurried. -/
def mulLeftRight : 𝕜' →L[𝕜] 𝕜' →L[𝕜] 𝕜' →L[𝕜] 𝕜' :=
  ((compL 𝕜 𝕜' 𝕜' 𝕜').comp (mul 𝕜 𝕜').flip).flip.comp (mul 𝕜 𝕜')


@[simp]
theorem mulLeftRight_apply (x y z : 𝕜') : mulLeftRight 𝕜 𝕜' x y z = x * z * y :=
  rfl


theorem opNorm_mulLeftRight_apply_apply_le (x y : 𝕜') : ‖mulLeftRight 𝕜 𝕜' x y‖ ≤ ‖x‖ * ‖y‖ :=
  (opNorm_comp_le _ _).trans <|
    (mul_comm _ _).trans_le <|
      mul_le_mul (opNorm_mul_apply_le _ _ _)
        (opNorm_le_bound _ (norm_nonneg _) fun _ => (norm_mul_le _ _).trans_eq (mul_comm _ _))
        (norm_nonneg _) (norm_nonneg _)


@[deprecated (since := "2024-02-02")]
alias op_norm_mulLeftRight_apply_apply_le :=
  opNorm_mulLeftRight_apply_apply_le


theorem opNorm_mulLeftRight_apply_le (x : 𝕜') : ‖mulLeftRight 𝕜 𝕜' x‖ ≤ ‖x‖ :=
  opNorm_le_bound _ (norm_nonneg x) (opNorm_mulLeftRight_apply_apply_le 𝕜 𝕜' x)


@[deprecated (since := "2024-02-02")]
alias op_norm_mulLeftRight_apply_le := opNorm_mulLeftRight_apply_le


set_option maxSynthPendingDepth 2 in
theorem opNorm_mulLeftRight_le :
    ‖mulLeftRight 𝕜 𝕜'‖ ≤ 1 :=
  opNorm_le_bound _ zero_le_one fun x => (one_mul ‖x‖).symm ▸ opNorm_mulLeftRight_apply_le 𝕜 𝕜' x


@[deprecated (since := "2024-02-02")] alias op_norm_mulLeftRight_le := opNorm_mulLeftRight_le


/-- This is a mixin class for non-unital normed algebras which states that the left-regular
representation of the algebra on itself is isometric. Every unital normed algebra with `‖1‖ = 1` is
a regular normed algebra (see `NormedAlgebra.instRegularNormedAlgebra`). In addition, so is every
C⋆-algebra, non-unital included (see `CStarRing.instRegularNormedAlgebra`), but there are yet other
examples. Any algebra with an approximate identity (e.g., $$L^1$$) is also regular.

This is a useful class because it gives rise to a nice norm on the unitization; in particular it is
a C⋆-norm when the norm on `A` is a C⋆-norm. -/
class _root_.RegularNormedAlgebra : Prop where
  /-- The left regular representation of the algebra on itself is an isometry. -/
  isometry_mul' : Isometry (mul 𝕜 𝕜')


/-- Every (unital) normed algebra such that `‖1‖ = 1` is a `RegularNormedAlgebra`. -/
instance _root_.NormedAlgebra.instRegularNormedAlgebra {𝕜 𝕜' : Type*} [NontriviallyNormedField 𝕜]
    [SeminormedRing 𝕜'] [NormedAlgebra 𝕜 𝕜'] [NormOneClass 𝕜'] : RegularNormedAlgebra 𝕜 𝕜' where
  isometry_mul' := AddMonoidHomClass.isometry_of_norm (mul 𝕜 𝕜') <|
    fun x => le_antisymm (opNorm_mul_apply_le _ _ _) <| by
      /-
        𝕜✝ : Type u_1
        E : Type u_2
        inst✝¹⁰ : NontriviallyNormedField 𝕜✝
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : NormedSpace 𝕜✝ E
        𝕜'✝ : Type u_3
        inst✝⁷ : NonUnitalSeminormedRing 𝕜'✝
        inst✝⁶ : NormedSpace 𝕜✝ 𝕜'✝
        inst✝⁵ : IsScalarTower 𝕜✝ 𝕜'✝ 𝕜'✝
        inst✝⁴ : SMulCommClass 𝕜✝ 𝕜'✝ 𝕜'✝
        𝕜 : Type u_4
        𝕜' : Type u_5
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : SeminormedRing 𝕜'
        inst✝¹ : NormedAlgebra 𝕜 𝕜'
        inst✝ : NormOneClass 𝕜'
        x : 𝕜'
        ⊢ LE.le (Norm.norm x) (Norm.norm ((ContinuousLinearMap.mul 𝕜 𝕜') x))
      -/
      convert ratio_le_opNorm ((mul 𝕜 𝕜') x) (1 : 𝕜')
      /-
        case h.e'_3
        𝕜✝ : Type u_1
        E : Type u_2
        inst✝¹⁰ : NontriviallyNormedField 𝕜✝
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : NormedSpace 𝕜✝ E
        𝕜'✝ : Type u_3
        inst✝⁷ : NonUnitalSeminormedRing 𝕜'✝
        inst✝⁶ : NormedSpace 𝕜✝ 𝕜'✝
        inst✝⁵ : IsScalarTower 𝕜✝ 𝕜'✝ 𝕜'✝
        inst✝⁴ : SMulCommClass 𝕜✝ 𝕜'✝ 𝕜'✝
        𝕜 : Type u_4
        𝕜' : Type u_5
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : SeminormedRing 𝕜'
        inst✝¹ : NormedAlgebra 𝕜 𝕜'
        inst✝ : NormOneClass 𝕜'
        x : 𝕜'
        ⊢ Eq (Norm.norm x) (HDiv.hDiv (Norm.norm (((ContinuousLinearMap.mul 𝕜 𝕜') x) 1 …
      -/
      simp [norm_one]
      /-
        🎉 no goals
      -/


lemma isometry_mul : Isometry (mul 𝕜 𝕜') :=
  RegularNormedAlgebra.isometry_mul'


@[simp]
lemma opNorm_mul_apply (x : 𝕜') : ‖mul 𝕜 𝕜' x‖ = ‖x‖ :=
  (AddMonoidHomClass.isometry_iff_norm (mul 𝕜 𝕜')).mp (isometry_mul 𝕜 𝕜') x


@[deprecated (since := "2024-02-02")] alias op_norm_mul_apply := opNorm_mul_apply


@[simp]
lemma opNNNorm_mul_apply (x : 𝕜') : ‖mul 𝕜 𝕜' x‖₊ = ‖x‖₊ :=
  Subtype.ext <| opNorm_mul_apply 𝕜 𝕜' x


@[deprecated (since := "2024-02-02")] alias op_nnnorm_mul_apply := opNNNorm_mul_apply


/-- Multiplication in a normed algebra as a linear isometry to the space of
continuous linear maps. -/
def mulₗᵢ : 𝕜' →ₗᵢ[𝕜] 𝕜' →L[𝕜] 𝕜' where
  toLinearMap := mul 𝕜 𝕜'
  norm_map' x := opNorm_mul_apply 𝕜 𝕜' x


@[simp]
theorem coe_mulₗᵢ : ⇑(mulₗᵢ 𝕜 𝕜') = mul 𝕜 𝕜' :=
  rfl


/-- If `M` is a normed space over `𝕜`, then the space of maps `𝕜 →L[𝕜] M` is linearly equivalent
to `M`. (See `ring_lmap_equiv_self` for a stronger statement.) -/
def ring_lmap_equiv_selfₗ : (𝕜 →L[𝕜] E) ≃ₗ[𝕜] E where
  toFun := fun f ↦ f 1
  invFun := (ContinuousLinearMap.id 𝕜 𝕜).smulRight
                            /-
                              𝕜 : Type u_1
                              E : Type u_2
                              inst✝² : NontriviallyNormedField 𝕜
                              inst✝¹ : SeminormedAddCommGroup E
                              inst✝ : NormedSpace 𝕜 E
                              a : 𝕜
                              f : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E
                              ⊢ Eq ({ toFun := fun f => f 1, map_add' := ⋯ }.toFun (HSMul.hSMul a f)) (HSMul …
                            -/
                           /-
                             𝕜 : Type u_1
                             E : Type u_2
                             inst✝² : NontriviallyNormedField 𝕜
                             inst✝¹ : SeminormedAddCommGroup E
                             inst✝ : NormedSpace 𝕜 E
                             f g : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E
                             ⊢ Eq ((fun f => f 1) (HAdd.hAdd f g)) (HAdd.hAdd ((fun f => f 1) f) ((fun f => …
                           -/
  map_smul' := fun a f ↦ by simp only [coe_smul', Pi.smul_apply, RingHom.id_apply]
                           /-
                             🎉 no goals
                           -/
                            /-
                              🎉 no goals
                            -/
  map_add' := fun f g ↦ by simp only [add_apply]
                         /-
                           𝕜 : Type u_1
                           E : Type u_2
                           inst✝² : NontriviallyNormedField 𝕜
                           inst✝¹ : SeminormedAddCommGroup E
                           inst✝ : NormedSpace 𝕜 E
                           f : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E
                           ⊢ Eq ((ContinuousLinearMap.id 𝕜 𝕜).smulRight ({ toFun := fun f => f 1, map_add …
                         -/
  left_inv := fun f ↦ by ext; simp only [smulRight_apply, coe_id', _root_.id, one_smul]
                              /-
                                🎉 no goals
                              -/
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            inst✝² : NontriviallyNormedField 𝕜
                            inst✝¹ : SeminormedAddCommGroup E
                            inst✝ : NormedSpace 𝕜 E
                            m : E
                            ⊢ Eq ({ toFun := fun f => f 1, map_add' := ⋯, map_smul' := ⋯ }.toFun ((Continu …
                          -/
  right_inv := fun m ↦ by simp only [smulRight_apply, id_apply, one_smul]
                          /-
                            🎉 no goals
                          -/


/-- If `M` is a normed space over `𝕜`, then the space of maps `𝕜 →L[𝕜] M` is linearly isometrically
equivalent to `M`. -/
def ring_lmap_equiv_self : (𝕜 →L[𝕜] E) ≃ₗᵢ[𝕜] E where
  toLinearEquiv := ring_lmap_equiv_selfₗ 𝕜 E
  norm_map' := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      ⊢ ∀ (x : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E), Eq (Norm.norm ((ContinuousLi …
    -/
    refine fun f ↦ le_antisymm ?_ ?_
      /-
        case refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        f : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E
        ⊢ LE.le (Norm.norm ((ContinuousLinearMap.ring_lmap_equiv_selfₗ 𝕜 E) f)) (Norm. …
      -/
    · simpa only [norm_one, mul_one] using le_opNorm f 1
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        f : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 E
        ⊢ LE.le (Norm.norm f) (Norm.norm ((ContinuousLinearMap.ring_lmap_equiv_selfₗ 𝕜 …
      -/
    · refine opNorm_le_bound' f (norm_nonneg <| f 1) (fun x _ ↦ ?_)
      rw [(by rw [smul_eq_mul, mul_one] : f x = f (x • 1)), ContinuousLinearMap.map_smul,
        norm_smul, mul_comm, (by rfl : ring_lmap_equiv_selfₗ 𝕜 E f = f 1)]


/-- Scalar multiplication as a continuous bilinear map. -/
def lsmul : 𝕜' →L[𝕜] E →L[𝕜] E :=
  ((Algebra.lsmul 𝕜 𝕜 E).toLinearMap : 𝕜' →ₗ[𝕜] E →ₗ[𝕜] E).mkContinuous₂ 1 fun c x => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      𝕜' : Type u_3
      inst✝³ : NormedField 𝕜'
      inst✝² : NormedAlgebra 𝕜 𝕜'
      inst✝¹ : NormedSpace 𝕜' E
      inst✝ : IsScalarTower 𝕜 𝕜' E
      c : 𝕜'
      x : E
      ⊢ LE.le (Norm.norm (((Algebra.lsmul 𝕜 𝕜 E).toLinearMap c) x)) (HMul.hMul (HMul …
    -/
    simpa only [one_mul] using norm_smul_le c x
    /-
      🎉 no goals
    -/


@[simp]
theorem lsmul_apply (c : 𝕜') (x : E) : lsmul 𝕜 𝕜' c x = c • x :=
  rfl


theorem norm_toSpanSingleton (x : E) : ‖toSpanSingleton 𝕜 x‖ = ‖x‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ Eq (Norm.norm (ContinuousLinearMap.toSpanSingleton 𝕜 x)) (Norm.norm x)
  -/
  refine opNorm_eq_of_bounds (norm_nonneg _) (fun x => ?_) fun N _ h => ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x✝ : E
      x : 𝕜
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.toSpanSingleton 𝕜 x✝) x)) (HMul.hMul  …
    -/
  · rw [toSpanSingleton_apply, norm_smul, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      N : Real
      x✝ : GE.ge N 0
      h : ∀ (x_1 : 𝕜), LE.le (Norm.norm ((ContinuousLinearMap.toSpanSingleton 𝕜 x) x …
      ⊢ LE.le (Norm.norm x) N
    -/
  · specialize h 1
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      N : Real
      x✝ : GE.ge N 0
      h : LE.le (Norm.norm ((ContinuousLinearMap.toSpanSingleton 𝕜 x) 1)) (HMul.hMul …
      ⊢ LE.le (Norm.norm x) N
    -/
    rw [toSpanSingleton_apply, norm_smul, mul_comm] at h
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      N : Real
      x✝ : GE.ge N 0
      h : LE.le (HMul.hMul (Norm.norm x) (Norm.norm 1)) (HMul.hMul N (Norm.norm 1))
      ⊢ LE.le (Norm.norm x) N
    -/
    exact (mul_le_mul_right (by simp)).mp h
    /-
      🎉 no goals
    -/


theorem opNorm_lsmul_apply_le (x : 𝕜') : ‖(lsmul 𝕜 𝕜' x : E →L[𝕜] E)‖ ≤ ‖x‖ :=
  ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg x) fun y => norm_smul_le x y


@[deprecated (since := "2024-02-02")] alias op_norm_lsmul_apply_le := opNorm_lsmul_apply_le


/-- The norm of `lsmul` is at most 1 in any semi-normed group. -/
theorem opNorm_lsmul_le : ‖(lsmul 𝕜 𝕜' : 𝕜' →L[𝕜] E →L[𝕜] E)‖ ≤ 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝³ : NormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.lsmul 𝕜 𝕜')) 1
  -/
  refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun x => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝³ : NormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    x : 𝕜'
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.hMul 1 (Norm.no …
  -/
  simp_rw [one_mul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝³ : NormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    x : 𝕜'
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (Norm.norm x)
  -/
  exact opNorm_lsmul_apply_le _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_lsmul_le := opNorm_lsmul_le


@[simp]
theorem opNorm_mul : ‖mul 𝕜 𝕜'‖ = 1 :=
  (mulₗᵢ 𝕜 𝕜').norm_toContinuousLinearMap


@[deprecated (since := "2024-02-02")] alias op_norm_mul := opNorm_mul


@[simp]
theorem opNNNorm_mul : ‖mul 𝕜 𝕜'‖₊ = 1 :=
  Subtype.ext <| opNorm_mul 𝕜 𝕜'


@[deprecated (since := "2024-02-02")] alias op_nnnorm_mul := opNNNorm_mul


/-- The norm of `lsmul` equals 1 in any nontrivial normed group.

This is `ContinuousLinearMap.opNorm_lsmul_le` as an equality. -/
@[simp]
theorem opNorm_lsmul [NormedField 𝕜'] [NormedAlgebra 𝕜 𝕜'] [NormedSpace 𝕜' E]
    [IsScalarTower 𝕜 𝕜' E] [Nontrivial E] : ‖(lsmul 𝕜 𝕜' : 𝕜' →L[𝕜] E →L[𝕜] E)‖ = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    ⊢ Eq (Norm.norm (ContinuousLinearMap.lsmul 𝕜 𝕜')) 1
  -/
  refine ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => ?_) fun N _ h => ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      𝕜' : Type u_3
      inst✝⁴ : NormedField 𝕜'
      inst✝³ : NormedAlgebra 𝕜 𝕜'
      inst✝² : NormedSpace 𝕜' E
      inst✝¹ : IsScalarTower 𝕜 𝕜' E
      inst✝ : Nontrivial E
      x : 𝕜'
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.hMul 1 (Norm.no …
    -/
  · rw [one_mul]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      𝕜' : Type u_3
      inst✝⁴ : NormedField 𝕜'
      inst✝³ : NormedAlgebra 𝕜 𝕜'
      inst✝² : NormedSpace 𝕜' E
      inst✝¹ : IsScalarTower 𝕜 𝕜' E
      inst✝ : Nontrivial E
      x : 𝕜'
      ⊢ LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (Norm.norm x)
    -/
    apply opNorm_lsmul_apply_le
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    N : Real
    x✝ : GE.ge N 0
    h : ∀ (x : 𝕜'), LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.h …
    ⊢ LE.le 1 N
  -/
  obtain ⟨y, hy⟩ := exists_ne (0 : E)
  /-
    case refine_2.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    N : Real
    x✝ : GE.ge N 0
    h : ∀ (x : 𝕜'), LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.h …
    y : E
    hy : Ne y 0
    ⊢ LE.le 1 N
  -/
  have := le_of_opNorm_le _ (h 1) y
  /-
    case refine_2.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    N : Real
    x✝ : GE.ge N 0
    h : ∀ (x : 𝕜'), LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.h …
    y : E
    hy : Ne y 0
    this : LE.le (Norm.norm (((ContinuousLinearMap.lsmul 𝕜 𝕜') 1) y)) (HMul.hMul ( …
    ⊢ LE.le 1 N
  -/
  simp_rw [lsmul_apply, one_smul, norm_one, mul_one] at this
  /-
    case refine_2.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    N : Real
    x✝ : GE.ge N 0
    h : ∀ (x : 𝕜'), LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.h …
    y : E
    hy : Ne y 0
    this : LE.le (Norm.norm y) (HMul.hMul N (Norm.norm y))
    ⊢ LE.le 1 N
  -/
  refine le_of_mul_le_mul_right ?_ (norm_pos_iff.mpr hy)
  /-
    case refine_2.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    𝕜' : Type u_3
    inst✝⁴ : NormedField 𝕜'
    inst✝³ : NormedAlgebra 𝕜 𝕜'
    inst✝² : NormedSpace 𝕜' E
    inst✝¹ : IsScalarTower 𝕜 𝕜' E
    inst✝ : Nontrivial E
    N : Real
    x✝ : GE.ge N 0
    h : ∀ (x : 𝕜'), LE.le (Norm.norm ((ContinuousLinearMap.lsmul 𝕜 𝕜') x)) (HMul.h …
    y : E
    hy : Ne y 0
    this : LE.le (Norm.norm y) (HMul.hMul N (Norm.norm y))
    ⊢ LE.le (HMul.hMul 1 (Norm.norm y)) (HMul.hMul N (Norm.norm y))
  -/
  simp_rw [one_mul, this]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_lsmul := opNorm_lsmul


