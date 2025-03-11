local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- A complex polarization identity, with a linear map. -/
theorem inner_map_polarization (T : V →ₗ[ℂ] V) (x y : V) :
    ⟪T y, x⟫_ℂ =
      (⟪T (x + y), x + y⟫_ℂ - ⟪T (x - y), x - y⟫_ℂ +
            Complex.I * ⟪T (x + Complex.I • y), x + Complex.I • y⟫_ℂ -
          Complex.I * ⟪T (x - Complex.I • y), x - Complex.I • y⟫_ℂ) /
        4 := by
  simp only [map_add, map_sub, inner_add_left, inner_add_right, LinearMap.map_smul, inner_smul_left,
    inner_smul_right, Complex.conj_I, ← pow_two, Complex.I_sq, inner_sub_left, inner_sub_right,
    mul_add, ← mul_assoc, mul_neg, neg_neg, sub_neg_eq_add, one_mul, neg_one_mul, mul_sub, sub_sub]
  /-
    V : Type u_4
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    T : LinearMap (RingHom.id Complex) V V
    x y : V
    ⊢ Eq (Inner.inner (T y) x) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem inner_map_polarization' (T : V →ₗ[ℂ] V) (x y : V) :
    ⟪T x, y⟫_ℂ =
      (⟪T (x + y), x + y⟫_ℂ - ⟪T (x - y), x - y⟫_ℂ -
            Complex.I * ⟪T (x + Complex.I • y), x + Complex.I • y⟫_ℂ +
          Complex.I * ⟪T (x - Complex.I • y), x - Complex.I • y⟫_ℂ) /
        4 := by
  simp only [map_add, map_sub, inner_add_left, inner_add_right, LinearMap.map_smul, inner_smul_left,
    inner_smul_right, Complex.conj_I, ← pow_two, Complex.I_sq, inner_sub_left, inner_sub_right,
    mul_add, ← mul_assoc, mul_neg, neg_neg, sub_neg_eq_add, one_mul, neg_one_mul, mul_sub, sub_sub]
  /-
    V : Type u_4
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    T : LinearMap (RingHom.id Complex) V V
    x y : V
    ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- A linear map `T` is zero, if and only if the identity `⟪T x, x⟫_ℂ = 0` holds for all `x`.
-/
theorem inner_map_self_eq_zero (T : V →ₗ[ℂ] V) : (∀ x : V, ⟪T x, x⟫_ℂ = 0) ↔ T = 0 := by
  /-
    V : Type u_4
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    T : LinearMap (RingHom.id Complex) V V
    ⊢ Iff (∀ (x : V), Eq (Inner.inner (T x) x) 0) (Eq T 0)
  -/
  constructor
    /-
      case mp
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      ⊢ (∀ (x : V), Eq (Inner.inner (T x) x) 0) → Eq T 0
    -/
  · intro hT
    /-
      case mp
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      hT : ∀ (x : V), Eq (Inner.inner (T x) x) 0
      ⊢ Eq T 0
    -/
    ext x
    /-
      case mp.h
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      hT : ∀ (x : V), Eq (Inner.inner (T x) x) 0
      x : V
      ⊢ Eq (T x) (0 x)
    -/
    rw [LinearMap.zero_apply, ← @inner_self_eq_zero ℂ V, inner_map_polarization]
    /-
      case mp.h
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      hT : ∀ (x : V), Eq (Inner.inner (T x) x) 0
      x : V
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (Inner.inner (T (HAdd.hAdd (T …
    -/
    simp only [hT]
    /-
      case mp.h
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      hT : ∀ (x : V), Eq (Inner.inner (T x) x) 0
      x : V
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub 0 0) (HMul.hMul Complex.I 0)) …
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      ⊢ Eq T 0 → ∀ (x : V), Eq (Inner.inner (T x) x) 0
    -/
  · rintro rfl x
    /-
      case mpr
      V : Type u_4
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      x : V
      ⊢ Eq (Inner.inner (0 x) x) 0
    -/
    simp only [LinearMap.zero_apply, inner_zero_left]
    /-
      🎉 no goals
    -/


/--
Two linear maps `S` and `T` are equal, if and only if the identity `⟪S x, x⟫_ℂ = ⟪T x, x⟫_ℂ` holds
for all `x`.
-/
theorem ext_inner_map (S T : V →ₗ[ℂ] V) : (∀ x : V, ⟪S x, x⟫_ℂ = ⟪T x, x⟫_ℂ) ↔ S = T := by
  /-
    V : Type u_4
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    S T : LinearMap (RingHom.id Complex) V V
    ⊢ Iff (∀ (x : V), Eq (Inner.inner (S x) x) (Inner.inner (T x) x)) (Eq S T)
  -/
  rw [← sub_eq_zero, ← inner_map_self_eq_zero]
  /-
    V : Type u_4
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    S T : LinearMap (RingHom.id Complex) V V
    ⊢ Iff (∀ (x : V), Eq (Inner.inner (S x) x) (Inner.inner (T x) x)) (∀ (x : V),  …
  -/
  refine forall_congr' fun x => ?_
  /-
    V : Type u_4
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    S T : LinearMap (RingHom.id Complex) V V
    x : V
    ⊢ Iff (Eq (Inner.inner (S x) x) (Inner.inner (T x) x)) (Eq (Inner.inner ((HSub …
  -/
  rw [LinearMap.sub_apply, inner_sub_left, sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- A linear isometry preserves the inner product. -/
@[simp]
theorem LinearIsometry.inner_map_map (f : E →ₗᵢ[𝕜] E') (x y : E) : ⟪f x, f y⟫ = ⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    E' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    f : LinearIsometry (RingHom.id 𝕜) E E'
    x y : E
    ⊢ Eq (Inner.inner (f x) (f y)) (Inner.inner x y)
  -/
  simp [inner_eq_sum_norm_sq_div_four, ← f.norm_map]
  /-
    🎉 no goals
  -/


/-- A linear isometric equivalence preserves the inner product. -/
@[simp]
theorem LinearIsometryEquiv.inner_map_map (f : E ≃ₗᵢ[𝕜] E') (x y : E) : ⟪f x, f y⟫ = ⟪x, y⟫ :=
  f.toLinearIsometry.inner_map_map x y


/-- The adjoint of a linear isometric equivalence is its inverse. -/
theorem LinearIsometryEquiv.inner_map_eq_flip (f : E ≃ₗᵢ[𝕜] E') (x : E) (y : E') :
    ⟪f x, y⟫_𝕜 = ⟪x, f.symm y⟫_𝕜 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    E' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    f : LinearIsometryEquiv (RingHom.id 𝕜) E E'
    x : E
    y : E'
    ⊢ Eq (Inner.inner (f x) y) (Inner.inner x (f.symm y))
  -/
  conv_lhs => rw [← f.apply_symm_apply y, f.inner_map_map]
  /-
    🎉 no goals
  -/


/-- A linear map that preserves the inner product is a linear isometry. -/
def LinearMap.isometryOfInner (f : E →ₗ[𝕜] E') (h : ∀ x y, ⟪f x, f y⟫ = ⟪x, y⟫) : E →ₗᵢ[𝕜] E' :=
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    inst✝⁸ : RCLike 𝕜
                    inst✝⁷ : SeminormedAddCommGroup E
                    inst✝⁶ : InnerProductSpace 𝕜 E
                    inst✝⁵ : SeminormedAddCommGroup F
                    inst✝⁴ : InnerProductSpace Real F
                    ι : Type u_4
                    ι' : Type u_5
                    ι'' : Type u_6
                    E' : Type u_7
                    inst✝³ : SeminormedAddCommGroup E'
                    inst✝² : InnerProductSpace 𝕜 E'
                    E'' : Type u_8
                    inst✝¹ : SeminormedAddCommGroup E''
                    inst✝ : InnerProductSpace 𝕜 E''
                    f : LinearMap (RingHom.id 𝕜) E E'
                    h : ∀ (x y : E), Eq (Inner.inner (f x) (f y)) (Inner.inner x y)
                    x : E
                    ⊢ Eq (Norm.norm (f x)) (Norm.norm x)
                  -/
  ⟨f, fun x => by simp only [@norm_eq_sqrt_inner 𝕜, h]⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem LinearMap.coe_isometryOfInner (f : E →ₗ[𝕜] E') (h) : ⇑(f.isometryOfInner h) = f :=
  rfl


@[simp]
theorem LinearMap.isometryOfInner_toLinearMap (f : E →ₗ[𝕜] E') (h) :
    (f.isometryOfInner h).toLinearMap = f :=
  rfl


/-- A linear equivalence that preserves the inner product is a linear isometric equivalence. -/
def LinearEquiv.isometryOfInner (f : E ≃ₗ[𝕜] E') (h : ∀ x y, ⟪f x, f y⟫ = ⟪x, y⟫) : E ≃ₗᵢ[𝕜] E' :=
  ⟨f, ((f : E →ₗ[𝕜] E').isometryOfInner h).norm_map⟩


@[simp]
theorem LinearEquiv.coe_isometryOfInner (f : E ≃ₗ[𝕜] E') (h) : ⇑(f.isometryOfInner h) = f :=
  rfl


@[simp]
theorem LinearEquiv.isometryOfInner_toLinearEquiv (f : E ≃ₗ[𝕜] E') (h) :
    (f.isometryOfInner h).toLinearEquiv = f :=
  rfl


/-- A linear map is an isometry if and it preserves the inner product. -/
theorem LinearMap.norm_map_iff_inner_map_map {F : Type*} [FunLike F E E'] [LinearMapClass F 𝕜 E E']
    (f : F) : (∀ x, ‖f x‖ = ‖x‖) ↔ (∀ x y, ⟪f x, f y⟫_𝕜 = ⟪x, y⟫_𝕜) :=
  ⟨({ toLinearMap := LinearMapClass.linearMap f, norm_map' := · : E →ₗᵢ[𝕜] E' }.inner_map_map),
    (LinearMapClass.linearMap f |>.isometryOfInner · |>.norm_map)⟩


/-- The inner product as a sesquilinear map. -/
def innerₛₗ : E →ₗ⋆[𝕜] E →ₗ[𝕜] 𝕜 :=
  LinearMap.mk₂'ₛₗ _ _ (fun v w => ⟪v, w⟫) inner_add_left (fun _ _ _ => inner_smul_left _ _ _)
    inner_add_right fun _ _ _ => inner_smul_right _ _ _


@[simp]
theorem innerₛₗ_apply_coe (v : E) : ⇑(innerₛₗ 𝕜 v) = fun w => ⟪v, w⟫ :=
  rfl


@[simp]
theorem innerₛₗ_apply (v w : E) : innerₛₗ 𝕜 v w = ⟪v, w⟫ :=
  rfl


/-- The inner product as a bilinear map in the real case. -/
def innerₗ : F →ₗ[ℝ] F →ₗ[ℝ] ℝ := innerₛₗ ℝ


@[simp] lemma flip_innerₗ : (innerₗ F).flip = innerₗ F := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    ⊢ Eq (innerₗ F).flip (innerₗ F)
  -/
  ext v w
  /-
    case h.h
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    v w : F
    ⊢ Eq (((innerₗ F).flip v) w) (((innerₗ F) v) w)
  -/
  exact real_inner_comm v w
  /-
    🎉 no goals
  -/


@[simp] lemma innerₗ_apply (v w : F) : innerₗ F v w = ⟪v, w⟫_ℝ := rfl


/-- The inner product as a continuous sesquilinear map. Note that `toDualMap` (resp. `toDual`)
in `InnerProductSpace.Dual` is a version of this given as a linear isometry (resp. linear
isometric equivalence). -/
def innerSL : E →L⋆[𝕜] E →L[𝕜] 𝕜 :=
  LinearMap.mkContinuous₂ (innerₛₗ 𝕜) 1 fun x y => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : E
      ⊢ LE.le (Norm.norm (((innerₛₗ 𝕜) x) y)) (HMul.hMul (HMul.hMul 1 (Norm.norm x)) …
    -/
    simp only [norm_inner_le_norm, one_mul, innerₛₗ_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem innerSL_apply_coe (v : E) : ⇑(innerSL 𝕜 v) = fun w => ⟪v, w⟫ :=
  rfl


@[simp]
theorem innerSL_apply (v w : E) : innerSL 𝕜 v w = ⟪v, w⟫ :=
  rfl


/-- The inner product as a continuous sesquilinear map, with the two arguments flipped. -/
def innerSLFlip : E →L[𝕜] E →L⋆[𝕜] 𝕜 :=
  @ContinuousLinearMap.flipₗᵢ' 𝕜 𝕜 𝕜 E E 𝕜 _ _ _ _ _ _ _ _ _ (RingHom.id 𝕜) (starRingEnd 𝕜) _ _
    (innerSL 𝕜)


@[simp]
theorem innerSLFlip_apply (x y : E) : innerSLFlip 𝕜 x y = ⟪y, x⟫ :=
  rfl


variable (F) in
@[simp] lemma innerSL_real_flip : (innerSL ℝ (E := F)).flip = innerSL ℝ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    ⊢ Eq (innerSL Real).flip (innerSL Real)
  -/
  ext v w
  /-
    case h.h
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    v w : F
    ⊢ Eq (((innerSL Real).flip v) w) (((innerSL Real) v) w)
  -/
  exact real_inner_comm _ _
  /-
    🎉 no goals
  -/


/-- Given `f : E →L[𝕜] E'`, construct the continuous sesquilinear form `fun x y ↦ ⟪x, A y⟫`, given
as a continuous linear map. -/
noncomputable def toSesqForm : (E →L[𝕜] E') →L[𝕜] E' →L⋆[𝕜] E →L[𝕜] 𝕜 :=
  (ContinuousLinearMap.flipₗᵢ' E E' 𝕜 (starRingEnd 𝕜) (RingHom.id 𝕜)).toContinuousLinearEquiv ∘L
    ContinuousLinearMap.compSL E E' (E' →L⋆[𝕜] 𝕜) (RingHom.id 𝕜) (RingHom.id 𝕜) (innerSLFlip 𝕜)


@[simp]
theorem toSesqForm_apply_coe (f : E →L[𝕜] E') (x : E') : toSesqForm f x = (innerSL 𝕜 x).comp f :=
  rfl


theorem toSesqForm_apply_norm_le {f : E →L[𝕜] E'} {v : E'} : ‖toSesqForm f v‖ ≤ ‖f‖ * ‖v‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    E' : Type u_4
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    f : ContinuousLinearMap (RingHom.id 𝕜) E E'
    v : E'
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.toSesqForm f) v)) (HMul.hMul (Norm.no …
  -/
  refine opNorm_le_bound _ (by positivity) fun x ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    E' : Type u_4
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    f : ContinuousLinearMap (RingHom.id 𝕜) E E'
    v : E'
    x : E
    ⊢ LE.le (Norm.norm (((ContinuousLinearMap.toSesqForm f) v) x)) (HMul.hMul (HMu …
  -/
  have h₁ : ‖f x‖ ≤ ‖f‖ * ‖x‖ := le_opNorm _ _
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    E' : Type u_4
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    f : ContinuousLinearMap (RingHom.id 𝕜) E E'
    v : E'
    x : E
    h₁ : LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
    ⊢ LE.le (Norm.norm (((ContinuousLinearMap.toSesqForm f) v) x)) (HMul.hMul (HMu …
  -/
  have h₂ := @norm_inner_le_norm 𝕜 E' _ _ _ v (f x)
  calc
    ‖⟪v, f x⟫‖ ≤ ‖v‖ * ‖f x‖ := h₂
    _ ≤ ‖v‖ * (‖f‖ * ‖x‖) := mul_le_mul_of_nonneg_left h₁ (norm_nonneg v)
    _ = ‖f‖ * ‖v‖ * ‖x‖ := by ring


/-- `innerSL` is an isometry. Note that the associated `LinearIsometry` is defined in
`InnerProductSpace.Dual` as `toDualMap`. -/
@[simp]
theorem innerSL_apply_norm (x : E) : ‖innerSL 𝕜 x‖ = ‖x‖ := by
  refine
    le_antisymm ((innerSL 𝕜 x).opNorm_le_bound (norm_nonneg _) fun y => norm_inner_le_norm _ _) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ LE.le (Norm.norm x) (Norm.norm ((innerSL 𝕜) x))
  -/
  rcases (norm_nonneg x).eq_or_gt with (h | h)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x : E
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm x) (Norm.norm ((innerSL 𝕜) x))
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x : E
      h : LT.lt 0 (Norm.norm x)
      ⊢ LE.le (Norm.norm x) (Norm.norm ((innerSL 𝕜) x))
    -/
  · refine (mul_le_mul_right h).mp ?_
    calc
      ‖x‖ * ‖x‖ = ‖(⟪x, x⟫ : 𝕜)‖ := by
        rw [← sq, inner_self_eq_norm_sq_to_K, norm_pow, norm_ofReal, abs_norm]
      _ ≤ ‖innerSL 𝕜 x‖ * ‖x‖ := (innerSL 𝕜 x).le_opNorm _


lemma norm_innerSL_le : ‖innerSL 𝕜 (E := E)‖ ≤ 1 :=
                                                        /-
                                                          𝕜 : Type u_1
                                                          E : Type u_2
                                                          inst✝² : RCLike 𝕜
                                                          inst✝¹ : SeminormedAddCommGroup E
                                                          inst✝ : InnerProductSpace 𝕜 E
                                                          ⊢ ∀ (x : E), LE.le (Norm.norm ((innerSL 𝕜) x)) (HMul.hMul 1 (Norm.norm x))
                                                        -/
  ContinuousLinearMap.opNorm_le_bound _ zero_le_one (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The inner product on an inner product space of dimension 2 can be evaluated in terms
of a complex-number representation of the space. -/
theorem inner_map_complex [SeminormedAddCommGroup G] [InnerProductSpace ℝ G] (f : G ≃ₗᵢ[ℝ] ℂ)
                                                       /-
                                                         G : Type u_4
                                                         inst✝¹ : SeminormedAddCommGroup G
                                                         inst✝ : InnerProductSpace Real G
                                                         f : LinearIsometryEquiv (RingHom.id Real) G Complex
                                                         x y : G
                                                         ⊢ Eq (Inner.inner x y) (HMul.hMul ((starRingEnd Complex) (f x)) (f y)).re
                                                       -/
    (x y : G) : ⟪x, y⟫_ℝ = (conj (f x) * f y).re := by rw [← Complex.inner, f.inner_map_map]
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Extract a real bilinear form from an operator `T`,
by taking the pairing `fun x ↦ re ⟪T x, x⟫`. -/
def ContinuousLinearMap.reApplyInnerSelf (T : E →L[𝕜] E) (x : E) : ℝ :=
  re ⟪T x, x⟫


theorem ContinuousLinearMap.reApplyInnerSelf_apply (T : E →L[𝕜] E) (x : E) :
    T.reApplyInnerSelf x = re ⟪T x, x⟫ :=
  rfl


theorem ContinuousLinearMap.reApplyInnerSelf_continuous (T : E →L[𝕜] E) :
    Continuous T.reApplyInnerSelf :=
  reCLM.continuous.comp <| T.continuous.inner continuous_id


theorem ContinuousLinearMap.reApplyInnerSelf_smul (T : E →L[𝕜] E) (x : E) {c : 𝕜} :
    T.reApplyInnerSelf (c • x) = ‖c‖ ^ 2 * T.reApplyInnerSelf x := by
  simp only [ContinuousLinearMap.map_smul, ContinuousLinearMap.reApplyInnerSelf_apply,
    inner_smul_left, inner_smul_right, ← mul_assoc, mul_conj, ← ofReal_pow, ← smul_re,
    Algebra.smul_def (‖c‖ ^ 2) ⟪T x, x⟫, algebraMap_eq_ofReal]


