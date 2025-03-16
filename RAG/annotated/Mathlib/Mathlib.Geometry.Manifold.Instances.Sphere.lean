/-- Stereographic projection, forward direction. This is a map from an inner product space `E` to
the orthogonal complement of an element `v` of `E`. It is smooth away from the affine hyperplane
through `v` parallel to the orthogonal complement.  It restricts on the sphere to the stereographic
projection. -/
def stereoToFun (x : E) : (ℝ ∙ v)ᗮ :=
  (2 / ((1 : ℝ) - innerSL ℝ v x)) • orthogonalProjection (ℝ ∙ v)ᗮ x


@[simp]
theorem stereoToFun_apply (x : E) :
    stereoToFun v x = (2 / ((1 : ℝ) - innerSL ℝ v x)) • orthogonalProjection (ℝ ∙ v)ᗮ x :=
  rfl


theorem contDiffOn_stereoToFun :
    ContDiffOn ℝ ∞ (stereoToFun v) {x : E | innerSL _ v x ≠ (1 : ℝ)} := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    ⊢ ContDiffOn Real (↑Top.top) (stereoToFun v) (setOf fun x => Ne (((innerSL Rea …
  -/
  refine ContDiffOn.smul ?_ (orthogonalProjection (ℝ ∙ v)ᗮ).contDiff.contDiffOn
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    ⊢ ContDiffOn Real (↑Top.top) (fun x => HDiv.hDiv 2 (HSub.hSub 1 (((innerSL Rea …
  -/
  refine contDiff_const.contDiffOn.div ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      ⊢ ContDiffOn Real (↑Top.top) (fun x => HSub.hSub 1 (((innerSL Real) v) x)) (se …
    -/
  · exact (contDiff_const.sub (innerSL ℝ v).contDiff).contDiffOn
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      ⊢ ∀ (x : E), Membership.mem (setOf fun x => Ne (((innerSL Real) v) x) 1) x → N …
    -/
  · intro x h h'
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v x : E
      h : Membership.mem (setOf fun x => Ne (((innerSL Real) v) x) 1) x
      h' : Eq (HSub.hSub 1 (((innerSL Real) v) x)) 0
      ⊢ False
    -/
    exact h (sub_eq_zero.mp h').symm
    /-
      🎉 no goals
    -/


theorem continuousOn_stereoToFun :
    ContinuousOn (stereoToFun v) {x : E | innerSL _ v x ≠ (1 : ℝ)} :=
  contDiffOn_stereoToFun.continuousOn


/-- Auxiliary function for the construction of the reverse direction of the stereographic
projection.  This is a map from the orthogonal complement of a unit vector `v` in an inner product
space `E` to `E`; we will later prove that it takes values in the unit sphere.

For most purposes, use `stereoInvFun`, not `stereoInvFunAux`. -/
def stereoInvFunAux (w : E) : E :=
  (‖w‖ ^ 2 + 4)⁻¹ • ((4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v)


@[simp]
theorem stereoInvFunAux_apply (w : E) :
    stereoInvFunAux v w = (‖w‖ ^ 2 + 4)⁻¹ • ((4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v) :=
  rfl


theorem stereoInvFunAux_mem (hv : ‖v‖ = 1) {w : E} (hw : w ∈ (ℝ ∙ v)ᗮ) :
    stereoInvFunAux v w ∈ sphere (0 : E) 1 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : E
    hw : Membership.mem (Submodule.span Real (Singleton.singleton v)).orthogonal w
    ⊢ Membership.mem (Metric.sphere 0 1) (stereoInvFunAux v w)
  -/
  have h₁ : (0 : ℝ) < ‖w‖ ^ 2 + 4 := by positivity
  suffices ‖(4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v‖ = ‖w‖ ^ 2 + 4 by
    simp only [mem_sphere_zero_iff_norm, norm_smul, Real.norm_eq_abs, abs_inv, this,
      abs_of_pos h₁, stereoInvFunAux_apply, inv_mul_cancel₀ h₁.ne']
  suffices ‖(4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v‖ ^ 2 = (‖w‖ ^ 2 + 4) ^ 2 by
    simpa only [sq_eq_sq_iff_abs_eq_abs, abs_norm, abs_of_pos h₁] using this
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : E
    hw : Membership.mem (Submodule.span Real (Singleton.singleton v)).orthogonal w
    h₁ : LT.lt 0 (HAdd.hAdd (HPow.hPow (Norm.norm w) 2) 4)
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd (HSMul.hSMul 4 w) (HSMul.hSMul (HSub.hSu …
  -/
  rw [Submodule.mem_orthogonal_singleton_iff_inner_left] at hw
  simp [norm_add_sq_real, norm_smul, inner_smul_left, inner_smul_right, hw, mul_pow,
    Real.norm_eq_abs, hv]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : E
    hw : Eq (Inner.inner w v) 0
    h₁ : LT.lt 0 (HAdd.hAdd (HPow.hPow (Norm.norm w) 2) 4)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow 4 2) (HPow.hPow (Norm.norm w) 2)) (HPow. …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_stereoInvFunAux (v : E) :
    HasFDerivAt (stereoInvFunAux v) (ContinuousLinearMap.id ℝ E) 0 := by
  have h₀ : HasFDerivAt (fun w : E => ‖w‖ ^ 2) (0 : E →L[ℝ] ℝ) 0 := by
    convert (hasStrictFDerivAt_norm_sq (0 : E)).hasFDerivAt
    simp only [map_zero, smul_zero]
  have h₁ : HasFDerivAt (fun w : E => (‖w‖ ^ 2 + 4)⁻¹) (0 : E →L[ℝ] ℝ) 0 := by
    convert (hasFDerivAt_inv _).comp _ (h₀.add (hasFDerivAt_const 4 0)) <;> simp
  have h₂ : HasFDerivAt (fun w => (4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v)
      ((4 : ℝ) • ContinuousLinearMap.id ℝ E) 0 := by
    convert ((hasFDerivAt_const (4 : ℝ) 0).smul (hasFDerivAt_id 0)).add
      ((h₀.sub (hasFDerivAt_const (4 : ℝ) 0)).smul (hasFDerivAt_const v 0)) using 1
    ext w
    simp
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    h₀ : HasFDerivAt (fun w => HPow.hPow (Norm.norm w) 2) 0 0
    h₁ : HasFDerivAt (fun w => Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm w) 2) 4))  …
    h₂ : HasFDerivAt (fun w => HAdd.hAdd (HSMul.hSMul 4 w) (HSMul.hSMul (HSub.hSub …
    ⊢ HasFDerivAt (stereoInvFunAux v) (ContinuousLinearMap.id Real E) 0
  -/
  convert h₁.smul h₂ using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    h₀ : HasFDerivAt (fun w => HPow.hPow (Norm.norm w) 2) 0 0
    h₁ : HasFDerivAt (fun w => Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm w) 2) 4))  …
    h₂ : HasFDerivAt (fun w => HAdd.hAdd (HSMul.hSMul 4 w) (HSMul.hSMul (HSub.hSub …
    ⊢ Eq (ContinuousLinearMap.id Real E) (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hA …
  -/
  ext w
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    h₀ : HasFDerivAt (fun w => HPow.hPow (Norm.norm w) 2) 0 0
    h₁ : HasFDerivAt (fun w => Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm w) 2) 4))  …
    h₂ : HasFDerivAt (fun w => HAdd.hAdd (HSMul.hSMul 4 w) (HSMul.hSMul (HSub.hSub …
    w : E
    ⊢ Eq ((ContinuousLinearMap.id Real E) w) ((HAdd.hAdd (HSMul.hSMul (Inv.inv (HA …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_stereoInvFunAux_comp_coe (v : E) :
    HasFDerivAt (stereoInvFunAux v ∘ ((↑) : (ℝ ∙ v)ᗮ → E)) (ℝ ∙ v)ᗮ.subtypeL 0 := by
  have : HasFDerivAt (stereoInvFunAux v) (ContinuousLinearMap.id ℝ E) ((ℝ ∙ v)ᗮ.subtypeL 0) :=
    hasFDerivAt_stereoInvFunAux v
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    this : HasFDerivAt (stereoInvFunAux v) (ContinuousLinearMap.id Real E) ((Submo …
    ⊢ HasFDerivAt (Function.comp (stereoInvFunAux v) Subtype.val) (Submodule.span  …
  -/
  refine this.comp (0 : (ℝ ∙ v)ᗮ) (by apply ContinuousLinearMap.hasFDerivAt)
  /-
    🎉 no goals
  -/


theorem contDiff_stereoInvFunAux : ContDiff ℝ ∞ (stereoInvFunAux v) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    ⊢ ContDiff Real (↑Top.top) (stereoInvFunAux v)
  -/
  have h₀ : ContDiff ℝ ∞ fun w : E => ‖w‖ ^ 2 := contDiff_norm_sq ℝ
  have h₁ : ContDiff ℝ ∞ fun w : E => (‖w‖ ^ 2 + 4)⁻¹ := by
    refine (h₀.add contDiff_const).inv ?_
    intro x
    nlinarith
  have h₂ : ContDiff ℝ ∞ fun w => (4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v := by
    refine (contDiff_const.smul contDiff_id).add ?_
    exact (h₀.sub contDiff_const).smul contDiff_const
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    h₀ : ContDiff Real ↑Top.top fun w => HPow.hPow (Norm.norm w) 2
    h₁ : ContDiff Real ↑Top.top fun w => Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm  …
    h₂ : ContDiff Real ↑Top.top fun w => HAdd.hAdd (HSMul.hSMul 4 w) (HSMul.hSMul  …
    ⊢ ContDiff Real (↑Top.top) (stereoInvFunAux v)
  -/
  exact h₁.smul h₂
  /-
    🎉 no goals
  -/


/-- Stereographic projection, reverse direction.  This is a map from the orthogonal complement of a
unit vector `v` in an inner product space `E` to the unit sphere in `E`. -/
def stereoInvFun (hv : ‖v‖ = 1) (w : (ℝ ∙ v)ᗮ) : sphere (0 : E) 1 :=
  ⟨stereoInvFunAux v (w : E), stereoInvFunAux_mem hv w.2⟩


@[simp]
theorem stereoInvFun_apply (hv : ‖v‖ = 1) (w : (ℝ ∙ v)ᗮ) :
    (stereoInvFun hv w : E) = (‖w‖ ^ 2 + 4)⁻¹ • ((4 : ℝ) • w + (‖w‖ ^ 2 - 4) • v) :=
  rfl


open scoped InnerProductSpace in
theorem stereoInvFun_ne_north_pole (hv : ‖v‖ = 1) (w : (ℝ ∙ v)ᗮ) :
                                /-
                                  E : Type u_1
                                  inst✝¹ : NormedAddCommGroup E
                                  inst✝ : InnerProductSpace Real E
                                  v : E
                                  hv : Eq (Norm.norm v) 1
                                  w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
                                  ⊢ Membership.mem (Metric.sphere 0 1) v
                                -/
    stereoInvFun hv w ≠ (⟨v, by simp [hv]⟩ : sphere (0 : E) 1) := by
                                /-
                                  🎉 no goals
                                -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    ⊢ Ne (stereoInvFun hv w) ⟨v, ⋯⟩
  -/
  refine Subtype.coe_ne_coe.1 ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    ⊢ Ne ↑(stereoInvFun hv w) ↑⟨v, ⋯⟩
  -/
  rw [← inner_lt_one_iff_real_of_norm_one _ hv]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      ⊢ LT.lt (Inner.inner (↑(stereoInvFun hv w)) v) 1
    -/
  · have hw : ⟪v, w⟫_ℝ = 0 := Submodule.mem_orthogonal_singleton_iff_inner_right.mp w.2
    have hw' : (‖(w : E)‖ ^ 2 + 4)⁻¹ * (‖(w : E)‖ ^ 2 - 4) < 1 := by
      rw [inv_mul_lt_iff₀']
      · linarith
      positivity
    simpa [real_inner_comm, inner_add_right, inner_smul_right, real_inner_self_eq_norm_mul_norm, hw,
      hv] using hw'
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      ⊢ Eq (Norm.norm ↑(stereoInvFun hv w)) 1
    -/
  · simpa using stereoInvFunAux_mem hv w.2
    /-
      🎉 no goals
    -/


theorem continuous_stereoInvFun (hv : ‖v‖ = 1) : Continuous (stereoInvFun hv) :=
  continuous_induced_rng.2 (contDiff_stereoInvFunAux.continuous.comp continuous_subtype_val)


open scoped InnerProductSpace in
attribute [-simp] AddSubgroupClass.coe_norm Submodule.coe_norm in
theorem stereo_left_inv (hv : ‖v‖ = 1) {x : sphere (0 : E) 1} (hx : (x : E) ≠ v) :
    stereoInvFun hv (stereoToFun v x) = x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    ⊢ Eq (stereoInvFun hv (stereoToFun v ↑x)) x
  -/
  ext
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    ⊢ Eq ↑(stereoInvFun hv (stereoToFun v ↑x)) ↑x
  -/
  simp only [stereoToFun_apply, stereoInvFun_apply, smul_add]
  -- name two frequently-occurring quantities and write down their basic properties
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm (HSMul. …
  -/
  set a : ℝ := innerSL _ v x
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    a : Real := ((innerSL Real) v) ↑x
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm (HSMul. …
  -/
  set y := orthogonalProjection (ℝ ∙ v)ᗮ x
  have split : ↑x = a • v + ↑y := by
    convert (orthogonalProjection_add_orthogonalProjection_orthogonal (ℝ ∙ v) x).symm
    exact (orthogonalProjection_unit_singleton ℝ hv x).symm
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    a : Real := ((innerSL Real) v) ↑x
    y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm (HSMul. …
  -/
  have hvy : ⟪v, y⟫_ℝ = 0 := Submodule.mem_orthogonal_singleton_iff_inner_right.mp y.2
  have pythag : 1 = a ^ 2 + ‖y‖ ^ 2 := by
    have hvy' : ⟪a • v, y⟫_ℝ = 0 := by simp only [inner_smul_left, hvy, mul_zero]
    convert norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero _ _ hvy' using 2
    · simp [← split]
    · simp [norm_smul, hv, ← sq, sq_abs]
    · exact sq _
  -- a fact which will be helpful for clearing denominators in the main calculation
  have ha : 0 < 1 - a := by
    have : a < 1 := (inner_lt_one_iff_real_of_norm_one hv (by simp)).mpr hx.symm
    linarith
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    a : Real := ((innerSL Real) v) ↑x
    y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
    hvy : Eq (Inner.inner v ↑y) 0
    pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
    ha : LT.lt 0 (HSub.hSub 1 a)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm (HSMul. …
  -/
  rw [split, Submodule.coe_smul_of_tower]
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    a : Real := ((innerSL Real) v) ↑x
    y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
    hvy : Eq (Inner.inner v ↑y) 0
    pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
    ha : LT.lt 0 (HSub.hSub 1 a)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm (HSMul. …
  -/
  simp only [norm_smul, norm_div, RCLike.norm_ofNat, Real.norm_eq_abs, abs_of_nonneg ha.le]
  /-
    case a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    x : ↑(Metric.sphere 0 1)
    hx : Ne (↑x) v
    a : Real := ((innerSL Real) v) ↑x
    y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
    hvy : Eq (Inner.inner v ↑y) 0
    pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
    ha : LT.lt 0 (HSub.hSub 1 a)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (HMul.hMul (HDiv.h …
  -/
  match_scalars
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hx : Ne (↑x) v
      a : Real := ((innerSL Real) v) ↑x
      y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
      hvy : Eq (Inner.inner v ↑y) 0
      pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
      ha : LT.lt 0 (HSub.hSub 1 a)
      ⊢ Eq (HMul.hMul (Inv.inv (HAdd.hAdd (HPow.hPow (HMul.hMul (HDiv.hDiv 2 (HSub.h …
    -/
  · field_simp
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hx : Ne (↑x) v
      a : Real := ((innerSL Real) v) ↑x
      y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
      hvy : Eq (Inner.inner v ↑y) 0
      pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
      ha : LT.lt 0 (HSub.hSub 1 a)
      ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 a) 2) (HMul.hMul 4 2)) (HMul.hMul (HAd …
    -/
    linear_combination 4 * (1 - a) * pythag
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hx : Ne (↑x) v
      a : Real := ((innerSL Real) v) ↑x
      y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
      hvy : Eq (Inner.inner v ↑y) 0
      pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
      ha : LT.lt 0 (HSub.hSub 1 a)
      ⊢ Eq (HMul.hMul (Inv.inv (HAdd.hAdd (HPow.hPow (HMul.hMul (HDiv.hDiv 2 (HSub.h …
    -/
  · field_simp
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hx : Ne (↑x) v
      a : Real := ((innerSL Real) v) ↑x
      y : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
      split : Eq (↑x) (HAdd.hAdd (HSMul.hSMul a v) ↑y)
      hvy : Eq (Inner.inner v ↑y) 0
      pythag : Eq 1 (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow (Norm.norm y) 2))
      ha : LT.lt 0 (HSub.hSub 1 a)
      ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 a) 2) (HSub.hSub (HPow.hPow (HMul.hMul …
    -/
    linear_combination 4 * (a - 1) ^ 3 * pythag
    /-
      🎉 no goals
    -/


theorem stereo_right_inv (hv : ‖v‖ = 1) (w : (ℝ ∙ v)ᗮ) : stereoToFun v (stereoInvFun hv w) = w := by
  simp only [stereoToFun, stereoInvFun, stereoInvFunAux, smul_add, map_add, map_smul, innerSL_apply,
    orthogonalProjection_mem_subspace_eq_self]
  have h₁ : orthogonalProjection (ℝ ∙ v)ᗮ v = 0 :=
    orthogonalProjection_orthogonalComplement_singleton_eq_zero v
  -- Porting note: was innerSL _ and now just inner
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    h₁ : Eq ((orthogonalProjection (Submodule.span Real (Singleton.singleton v)).o …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv 2 (HSub.hSub 1 (HAdd.hAdd (HSMul.hSMul …
  -/
  have h₂ : inner v w = (0 : ℝ) := Submodule.mem_orthogonal_singleton_iff_inner_right.mp w.2
  -- Porting note: was innerSL _ and now just inner
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    h₁ : Eq ((orthogonalProjection (Submodule.span Real (Singleton.singleton v)).o …
    h₂ : Eq (Inner.inner v ↑w) 0
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv 2 (HSub.hSub 1 (HAdd.hAdd (HSMul.hSMul …
  -/
  have h₃ : inner v v = (1 : ℝ) := by simp [real_inner_self_eq_norm_mul_norm, hv]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    h₁ : Eq ((orthogonalProjection (Submodule.span Real (Singleton.singleton v)).o …
    h₂ : Eq (Inner.inner v ↑w) 0
    h₃ : Eq (Inner.inner v v) 1
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv 2 (HSub.hSub 1 (HAdd.hAdd (HSMul.hSMul …
  -/
  rw [h₁, h₂, h₃]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    h₁ : Eq ((orthogonalProjection (Submodule.span Real (Singleton.singleton v)).o …
    h₂ : Eq (Inner.inner v ↑w) 0
    h₃ : Eq (Inner.inner v v) 1
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv 2 (HSub.hSub 1 (HAdd.hAdd (HSMul.hSMul …
  -/
  match_scalars
  -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp`, but was really slow
  -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
  simp (disch := field_simp_discharge) only [add_div', add_sub_sub_cancel, div_div,
    div_div_eq_mul_div, div_eq_iff, div_mul_eq_mul_div, inv_eq_one_div,
    mul_div_assoc', mul_one, mul_zero, one_mul, smul_eq_mul, sub_div', zero_add, zero_div, zero_mul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : E
    hv : Eq (Norm.norm v) 1
    w : Subtype fun x => Membership.mem (Submodule.span Real (Singleton.singleton  …
    h₁ : Eq ((orthogonalProjection (Submodule.span Real (Singleton.singleton v)).o …
    h₂ : Eq (Inner.inner v ↑w) 0
    h₃ : Eq (Inner.inner v v) 1
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (HPow.hPow (Norm.norm ↑w) 2) 4)) 4) (H …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Stereographic projection from the unit sphere in `E`, centred at a unit vector `v` in `E`;
this is the version as a partial homeomorphism. -/
def stereographic (hv : ‖v‖ = 1) : PartialHomeomorph (sphere (0 : E) 1) (ℝ ∙ v)ᗮ where
  toFun := stereoToFun v ∘ (↑)
  invFun := stereoInvFun hv
                    /-
                      E : Type u_1
                      inst✝¹ : NormedAddCommGroup E
                      inst✝ : InnerProductSpace Real E
                      v : E
                      hv : Eq (Norm.norm v) 1
                      ⊢ Membership.mem (Metric.sphere 0 1) v
                    -/
  source := {⟨v, by simp [hv]⟩}ᶜ
                    /-
                      🎉 no goals
                    -/
  target := Set.univ
                    /-
                      E : Type u_1
                      inst✝¹ : NormedAddCommGroup E
                      inst✝ : InnerProductSpace Real E
                      v : E
                      hv : Eq (Norm.norm v) 1
                      ⊢ ∀ ⦃x : ↑(Metric.sphere 0 1)⦄, Membership.mem (HasCompl.compl (Singleton.sing …
                    -/
  map_source' := by simp
                    /-
                      🎉 no goals
                    -/
  map_target' {w} _ := fun h => (stereoInvFun_ne_north_pole hv w) (Set.eq_of_mem_singleton h)
  left_inv' x hx := stereo_left_inv hv fun h => hx (by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hx : Membership.mem (HasCompl.compl (Singleton.singleton ⟨v, ?m.242673⟩)) x
      h : Eq (↑x) v
      ⊢ Membership.mem (Singleton.singleton ⟨v, ?m.242673⟩) x
    -/
    rw [← h] at hv
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv✝ : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hv : Eq (Norm.norm ↑x) 1
      hx : Membership.mem (HasCompl.compl (Singleton.singleton ⟨v, ?m.242673⟩)) x
      h : Eq (↑x) v
      ⊢ Membership.mem (Singleton.singleton ⟨v, ?m.242673⟩) x
    -/
    apply Subtype.ext
    /-
      case a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv✝ : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hv : Eq (Norm.norm ↑x) 1
      hx : Membership.mem (HasCompl.compl (Singleton.singleton ⟨v, ?m.242673⟩)) x
      h : Eq (↑x) v
      ⊢ Eq ↑x ↑⟨v, ?m.242673⟩
    -/
    dsimp
    /-
      case a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      v : E
      hv✝ : Eq (Norm.norm v) 1
      x : ↑(Metric.sphere 0 1)
      hv : Eq (Norm.norm ↑x) 1
      hx : Membership.mem (HasCompl.compl (Singleton.singleton ⟨v, ?m.242673⟩)) x
      h : Eq (↑x) v
      ⊢ Eq (↑x) v
    -/
    exact h)
    /-
      🎉 no goals
    -/
  right_inv' w _ := stereo_right_inv hv w
  open_source := isOpen_compl_singleton
  open_target := isOpen_univ
  continuousOn_toFun :=
    continuousOn_stereoToFun.comp continuous_subtype_val.continuousOn fun w h => by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        v : E
        hv : Eq (Norm.norm v) 1
        w : ↑(Metric.sphere 0 1)
        h : Membership.mem { toFun := Function.comp (stereoToFun v) Subtype.val, invFu …
        ⊢ Membership.mem (setOf fun x => Ne (((innerSL Real) v) x) 1) ↑w
      -/
      dsimp
      exact
        h ∘ Subtype.ext ∘ Eq.symm ∘ (inner_eq_one_iff_of_norm_one hv (by simp)).mp
  continuousOn_invFun := (continuous_stereoInvFun hv).continuousOn


theorem stereographic_apply (hv : ‖v‖ = 1) (x : sphere (0 : E) 1) :
    stereographic hv x = (2 / ((1 : ℝ) - inner v x)) • orthogonalProjection (ℝ ∙ v)ᗮ x :=
  rfl


@[simp]
                                                                                  /-
                                                                                    E : Type u_1
                                                                                    inst✝¹ : NormedAddCommGroup E
                                                                                    inst✝ : InnerProductSpace Real E
                                                                                    v : E
                                                                                    hv : Eq (Norm.norm v) 1
                                                                                    ⊢ Membership.mem (Metric.sphere 0 1) v
                                                                                  -/
theorem stereographic_source (hv : ‖v‖ = 1) : (stereographic hv).source = {⟨v, by simp [hv]⟩}ᶜ :=
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  rfl


@[simp]
theorem stereographic_target (hv : ‖v‖ = 1) : (stereographic hv).target = Set.univ :=
  rfl


@[simp]
theorem stereographic_apply_neg (v : sphere (0 : E) 1) :
    stereographic (norm_eq_of_mem_sphere v) (-v) = 0 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq (↑(stereographic ⋯) (Neg.neg v)) 0
  -/
  simp [stereographic_apply, orthogonalProjection_orthogonalComplement_singleton_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem stereographic_neg_apply (v : sphere (0 : E) 1) :
    stereographic (norm_eq_of_mem_sphere (-v)) v = 0 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq (↑(stereographic ⋯) v) 0
  -/
  convert stereographic_apply_neg (-v)
  /-
    case h.e'_2.h.e'_6
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq v (Neg.neg (Neg.neg v))
  -/
  ext1
  /-
    case h.e'_2.h.e'_6.a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq ↑v ↑(Neg.neg (Neg.neg v))
  -/
  simp
  /-
    🎉 no goals
  -/


private theorem findim (n : ℕ) [Fact (finrank ℝ E = n + 1)] : FiniteDimensional ℝ E :=
  .of_fact_finrank_eq_succ n


/-- Variant of the stereographic projection, for the sphere in an `n + 1`-dimensional inner product
space `E`.  This version has codomain the Euclidean space of dimension `n`, and is obtained by
composing the original sterographic projection (`stereographic`) with an arbitrary linear isometry
from `(ℝ ∙ v)ᗮ` to the Euclidean space. -/
def stereographic' (n : ℕ) [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
    PartialHomeomorph (sphere (0 : E) 1) (EuclideanSpace ℝ (Fin n)) :=
  stereographic (norm_eq_of_mem_sphere v) ≫ₕ
    (OrthonormalBasis.fromOrthogonalSpanSingleton n
            (ne_zero_of_mem_unit_sphere v)).repr.toHomeomorph.toPartialHomeomorph


@[simp]
theorem stereographic'_source {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
                                             /-
                                               E : Type u_1
                                               inst✝² : NormedAddCommGroup E
                                               inst✝¹ : InnerProductSpace Real E
                                               n : Nat
                                               inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
                                               v : ↑(Metric.sphere 0 1)
                                               ⊢ Eq (stereographic' n v).source (HasCompl.compl (Singleton.singleton v))
                                             -/
    (stereographic' n v).source = {v}ᶜ := by simp [stereographic']
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem stereographic'_target {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
                                                 /-
                                                   E : Type u_1
                                                   inst✝² : NormedAddCommGroup E
                                                   inst✝¹ : InnerProductSpace Real E
                                                   n : Nat
                                                   inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
                                                   v : ↑(Metric.sphere 0 1)
                                                   ⊢ Eq (stereographic' n v).target Set.univ
                                                 -/
    (stereographic' n v).target = Set.univ := by simp [stereographic']
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The unit sphere in an `n + 1`-dimensional inner product space `E` is a charted space
modelled on the Euclidean space of dimension `n`. -/
instance EuclideanSpace.instChartedSpaceSphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] :
    ChartedSpace (EuclideanSpace ℝ (Fin n)) (sphere (0 : E) 1) where
  atlas := {f | ∃ v : sphere (0 : E) 1, f = stereographic' n v}
  chartAt v := stereographic' n (-v)
                           /-
                             E : Type u_1
                             inst✝² : NormedAddCommGroup E
                             inst✝¹ : InnerProductSpace Real E
                             n : Nat
                             inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
                             v : ↑(Metric.sphere 0 1)
                             ⊢ Membership.mem ((fun v => stereographic' n (Neg.neg v)) v).source v
                           -/
  mem_chart_source v := by simpa using ne_neg_of_mem_unit_sphere ℝ v
                           /-
                             🎉 no goals
                           -/
  chart_mem_atlas v := ⟨-v, rfl⟩


instance (n : ℕ) :
    ChartedSpace (EuclideanSpace ℝ (Fin n)) (sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1) :=
  have := Fact.mk (@finrank_euclideanSpace_fin ℝ _ (n + 1))
  EuclideanSpace.instChartedSpaceSphere


theorem sphere_ext_iff (u v : sphere (0 : E) 1) : u = v ↔ ⟪(u : E), v⟫_ℝ = 1 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    u v : ↑(Metric.sphere 0 1)
    ⊢ Iff (Eq u v) (Eq (Inner.inner ↑u ↑v) 1)
  -/
  simp [Subtype.ext_iff, inner_eq_one_iff_of_norm_one]
  /-
    🎉 no goals
  -/


theorem stereographic'_symm_apply {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1)
    (x : EuclideanSpace ℝ (Fin n)) :
    ((stereographic' n v).symm x : E) =
      let U : (ℝ ∙ (v : E))ᗮ ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n) :=
        (OrthonormalBasis.fromOrthogonalSpanSingleton n (ne_zero_of_mem_unit_sphere v)).repr
      (‖(U.symm x : E)‖ ^ 2 + 4)⁻¹ • (4 : ℝ) • (U.symm x : E) +
        (‖(U.symm x : E)‖ ^ 2 + 4)⁻¹ • (‖(U.symm x : E)‖ ^ 2 - 4) • v.val := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    x : EuclideanSpace Real (Fin n)
    ⊢ Eq (↑(↑(stereographic' n v).symm x))
        (let U := (OrthonormalBasis.fromOrthogonalSpanSingleton n ⋯).repr;
        HAdd.hAdd (HSMul.hSMul (Inv.inv (HAdd.hAdd (HPow.hPow (Norm.norm ↑(U.symm  …
  -/
  simp [real_inner_comm, stereographic, stereographic', ← Submodule.coe_norm]
  /-
    🎉 no goals
  -/


/-- The unit sphere in an `n + 1`-dimensional inner product space `E` is a smooth manifold,
modelled on the Euclidean space of dimension `n`. -/
instance EuclideanSpace.instSmoothManifoldWithCornersSphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] :
    SmoothManifoldWithCorners (𝓡 n) (sphere (0 : E) 1) :=
  smoothManifoldWithCorners_of_contDiffOn (𝓡 n) (sphere (0 : E) 1)
    (by
      /-
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        n : Nat
        inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
        ⊢ ∀ (e e' : PartialHomeomorph (↑(Metric.sphere 0 1)) (EuclideanSpace Real (Fin …
      -/
      rintro _ _ ⟨v, rfl⟩ ⟨v', rfl⟩
      let U :=
        (-- Removed type ascription, and this helped for some reason with timeout issues?
            OrthonormalBasis.fromOrthogonalSpanSingleton (𝕜 := ℝ)
            n (ne_zero_of_mem_unit_sphere v)).repr
      let U' :=
        (-- Removed type ascription, and this helped for some reason with timeout issues?
            OrthonormalBasis.fromOrthogonalSpanSingleton (𝕜 := ℝ)
            n (ne_zero_of_mem_unit_sphere v')).repr
      /-
        case intro.intro
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        n : Nat
        inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
        v v' : ↑(Metric.sphere 0 1)
        U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
        U' : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (S …
        ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersSelf Real (Eucl …
      -/
      have H₁ := U'.contDiff.comp_contDiffOn contDiffOn_stereoToFun
      -- Porting note: need to help with implicit variables again
      have H₂ := (contDiff_stereoInvFunAux (v := v.val)|>.comp
        (ℝ ∙ (v : E))ᗮ.subtypeL.contDiff).comp U.symm.contDiff
      /-
        case intro.intro
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        n : Nat
        inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
        v v' : ↑(Metric.sphere 0 1)
        U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
        U' : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (S …
        H₁ : ContDiffOn Real (↑Top.top) (Function.comp (⇑U') (stereoToFun ↑v')) (setOf …
        H₂ : ContDiff Real (↑Top.top) (Function.comp (Function.comp (stereoInvFunAux ↑ …
        ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersSelf Real (Eucl …
      -/
      convert H₁.comp_inter (H₂.contDiffOn : ContDiffOn ℝ ∞ _ Set.univ) using 1
      -- -- squeezed from `ext, simp [sphere_ext_iff, stereographic'_symm_apply, real_inner_comm]`
      simp only [PartialHomeomorph.trans_toPartialEquiv, PartialHomeomorph.symm_toPartialEquiv,
        PartialEquiv.trans_source, PartialEquiv.symm_source, stereographic'_target,
        stereographic'_source]
      simp only [modelWithCornersSelf_coe, modelWithCornersSelf_coe_symm, Set.preimage_id,
        Set.range_id, Set.inter_univ, Set.univ_inter, Set.compl_singleton_eq, Set.preimage_setOf_eq]
      simp only [id, comp_apply, Submodule.subtypeL_apply, PartialHomeomorph.coe_coe_symm,
        innerSL_apply, Ne, sphere_ext_iff, real_inner_comm (v' : E)]
      /-
        case h.e'_11
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace Real E
        n : Nat
        inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
        v v' : ↑(Metric.sphere 0 1)
        U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
        U' : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (S …
        H₁ : ContDiffOn Real (↑Top.top) (Function.comp (⇑U') (stereoToFun ↑v')) (setOf …
        H₂ : ContDiff Real (↑Top.top) (Function.comp (Function.comp (stereoInvFunAux ↑ …
        ⊢ Eq (setOf fun a => Not (Eq (Inner.inner ↑v' ↑(↑(stereographic' n v).symm a)) …
      -/
      rfl)
      /-
        🎉 no goals
      -/


instance (n : ℕ) :
    SmoothManifoldWithCorners (𝓡 n) (sphere (0 :  EuclideanSpace ℝ (Fin (n + 1))) 1) :=
  haveI := Fact.mk (@finrank_euclideanSpace_fin ℝ _ (n + 1))
  EuclideanSpace.instSmoothManifoldWithCornersSphere


/-- The inclusion map (i.e., `coe`) from the sphere in `E` to `E` is smooth. -/
theorem contMDiff_coe_sphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] :
    ContMDiff (𝓡 n) 𝓘(ℝ, E) ⊤ ((↑) : sphere (0 : E) 1 → E) := by
  -- Porting note: trouble with filling these implicit variables in the instance
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) (modelWi …
  -/
  have := EuclideanSpace.instSmoothManifoldWithCornersSphere (E := E) (n := n)
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    this : SmoothManifoldWithCorners (modelWithCornersSelf Real (EuclideanSpace Re …
    ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) (modelWi …
  -/
  rw [contMDiff_iff]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    this : SmoothManifoldWithCorners (modelWithCornersSelf Real (EuclideanSpace Re …
    ⊢ And (Continuous Subtype.val) (∀ (x : Subtype fun x => Membership.mem (Metric …
  -/
  constructor
    /-
      case left
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      this : SmoothManifoldWithCorners (modelWithCornersSelf Real (EuclideanSpace Re …
      ⊢ Continuous Subtype.val
    -/
  · exact continuous_subtype_val
    /-
      🎉 no goals
    -/
    /-
      case right
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      this : SmoothManifoldWithCorners (modelWithCornersSelf Real (EuclideanSpace Re …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Metric.sphere 0 1) x) (y : E), ContD …
    -/
  · intro v _
    let U : _ ≃ₗᵢ[ℝ] _ :=
      (-- Again, partially removing type ascription...
          OrthonormalBasis.fromOrthogonalSpanSingleton
          n (ne_zero_of_mem_unit_sphere (-v))).repr
    exact
      ((contDiff_stereoInvFunAux.comp (ℝ ∙ (-v : E))ᗮ.subtypeL.contDiff).comp
          U.symm.contDiff).contDiffOn


/-- If a `ContMDiff` function `f : M → E`, where `M` is some manifold, takes values in the
sphere, then it restricts to a `ContMDiff` function from `M` to the sphere. -/
theorem ContMDiff.codRestrict_sphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] {m : ℕ∞} {f : M → E}
    (hf : ContMDiff I 𝓘(ℝ, E) m f) (hf' : ∀ x, f x ∈ sphere (0 : E) 1) :
    ContMDiff I (𝓡 n) m (Set.codRestrict _ _ hf' : M → sphere (0 : E) 1) := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    ⊢ ContMDiff I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) m (Set …
  -/
  rw [contMDiff_iff_target]
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    ⊢ And (Continuous (Set.codRestrict f (Metric.sphere 0 1) hf')) (∀ (y : ↑(Metri …
  -/
  refine ⟨continuous_induced_rng.2 hf.continuous, ?_⟩
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    ⊢ ∀ (y : ↑(Metric.sphere 0 1)), ContMDiffOn I (modelWithCornersSelf Real (Eucl …
  -/
  intro v
  let U : _ ≃ₗᵢ[ℝ] _ :=
    (-- Again, partially removing type ascription... Weird that this helps!
        OrthonormalBasis.fromOrthogonalSpanSingleton
        n (ne_zero_of_mem_unit_sphere (-v))).repr
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    ⊢ ContMDiffOn I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) m (F …
  -/
  have h : ContDiffOn ℝ ∞ _ Set.univ := U.contDiff.contDiffOn
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    ⊢ ContMDiffOn I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) m (F …
  -/
  have H₁ := (h.comp_inter contDiffOn_stereoToFun).contMDiffOn
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    H₁ : ContMDiffOn (modelWithCornersSelf Real E) (modelWithCornersSelf Real (Euc …
    ⊢ ContMDiffOn I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) m (F …
  -/
  have H₂ : ContMDiffOn _ _ _ _ Set.univ := hf.contMDiffOn
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    H₁ : ContMDiffOn (modelWithCornersSelf Real E) (modelWithCornersSelf Real (Euc …
    H₂ : ContMDiffOn I (modelWithCornersSelf Real E) m f Set.univ
    ⊢ ContMDiffOn I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) m (F …
  -/
  convert (H₁.of_le le_top).comp' H₂ using 1
  /-
    case h.e'_23
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    H₁ : ContMDiffOn (modelWithCornersSelf Real E) (modelWithCornersSelf Real (Euc …
    H₂ : ContMDiffOn I (modelWithCornersSelf Real E) m f Set.univ
    ⊢ Eq (Set.preimage (Set.codRestrict f (Metric.sphere 0 1) hf') (extChartAt (mo …
  -/
  ext x
  have hfxv : f x = -↑v ↔ ⟪f x, -↑v⟫_ℝ = 1 := by
    have hfx : ‖f x‖ = 1 := by simpa using hf' x
    rw [inner_eq_one_iff_of_norm_one hfx]
    exact norm_eq_of_mem_sphere (-v)
  -- Porting note: unfold more
  /-
    case h.e'_23.h
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    H₁ : ContMDiffOn (modelWithCornersSelf Real E) (modelWithCornersSelf Real (Euc …
    H₂ : ContMDiffOn I (modelWithCornersSelf Real E) m f Set.univ
    x : M
    hfxv : Iff (Eq (f x) (Neg.neg ↑v)) (Eq (Inner.inner (f x) (Neg.neg ↑v)) 1)
    ⊢ Iff (Membership.mem (Set.preimage (Set.codRestrict f (Metric.sphere 0 1) hf' …
  -/
  dsimp [chartAt, Set.codRestrict, ChartedSpace.chartAt]
  /-
    case h.e'_23.h
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : InnerProductSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real F H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    m : ENat
    f : M → E
    hf : ContMDiff I (modelWithCornersSelf Real E) m f
    hf' : ∀ (x : M), Membership.mem (Metric.sphere 0 1) (f x)
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    h : ContDiffOn Real (↑Top.top) (⇑U) Set.univ
    H₁ : ContMDiffOn (modelWithCornersSelf Real E) (modelWithCornersSelf Real (Euc …
    H₂ : ContMDiffOn I (modelWithCornersSelf Real E) m f Set.univ
    x : M
    hfxv : Iff (Eq (f x) (Neg.neg ↑v)) (Eq (Inner.inner (f x) (Neg.neg ↑v)) 1)
    ⊢ Iff (Membership.mem (Inter.inter (Set.preimage (Set.codRestrict f (Metric.sp …
  -/
  simp [not_iff_not, Subtype.ext_iff, hfxv, real_inner_comm]
  /-
    🎉 no goals
  -/


/-- The antipodal map is smooth. -/
theorem contMDiff_neg_sphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] :
    ContMDiff (𝓡 n) (𝓡 n) ⊤ fun x : sphere (0 : E) 1 => -x := by
  -- this doesn't elaborate well in term mode
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) (modelWi …
  -/
  apply ContMDiff.codRestrict_sphere
  /-
    case hf
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) (modelWi …
  -/
  apply contDiff_neg.contMDiff.comp _
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin n))) (modelWi …
  -/
  exact contMDiff_coe_sphere
  /-
    🎉 no goals
  -/


private lemma stereographic'_neg {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
  stereographic' n (-v) v = 0 := by
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      ⊢ Eq (↑(stereographic' n (Neg.neg v)) v) 0
    -/
    dsimp [stereographic']
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      ⊢ Eq ((OrthonormalBasis.fromOrthogonalSpanSingleton n ⋯).repr (↑(stereographic …
    -/
    simp only [EmbeddingLike.map_eq_zero_iff]
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      ⊢ Eq (↑(stereographic ⋯) v) 0
    -/
    apply stereographic_neg_apply
    /-
      🎉 no goals
    -/


/-- Consider the differential of the inclusion of the sphere in `E` at the point `v` as a continuous
linear map from `TangentSpace (𝓡 n) v` to `E`.  The range of this map is the orthogonal complement
of `v` in `E`.

Note that there is an abuse here of the defeq between `E` and the tangent space to `E` at `(v:E`).
In general this defeq is not canonical, but in this case (the tangent space of a vector space) it is
canonical. -/
theorem range_mfderiv_coe_sphere {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
    LinearMap.range (mfderiv (𝓡 n) 𝓘(ℝ, E) ((↑) : sphere (0 : E) 1 → E) v :
    TangentSpace (𝓡 n) v →L[ℝ] E) = (ℝ ∙ (v : E))ᗮ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq (LinearMap.range (mfderiv (modelWithCornersSelf Real (EuclideanSpace Real …
  -/
  rw [((contMDiff_coe_sphere v).mdifferentiableAt le_top).mfderiv]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    ⊢ Eq (LinearMap.range (fderivWithin Real (writtenInExtChartAt (modelWithCorner …
  -/
  dsimp [chartAt]
  simp only [chartAt, stereographic_neg_apply, fderivWithin_univ,
    LinearIsometryEquiv.toHomeomorph_symm, LinearIsometryEquiv.coe_toHomeomorph,
    LinearIsometryEquiv.map_zero, mfld_simps]
  let U := (OrthonormalBasis.fromOrthogonalSpanSingleton (𝕜 := ℝ) n
    (ne_zero_of_mem_unit_sphere (-v))).repr
  -- Porting note: this `suffices` was a `change`
  suffices
      LinearMap.range (fderiv ℝ ((stereoInvFunAux (-v : E) ∘ (↑)) ∘ U.symm) 0) = (ℝ ∙ (v : E))ᗮ by
    convert this using 3
    apply stereographic'_neg
  have :
    HasFDerivAt (stereoInvFunAux (-v : E) ∘ (Subtype.val : (ℝ ∙ (↑(-v) : E))ᗮ → E))
      (ℝ ∙ (↑(-v) : E))ᗮ.subtypeL (U.symm 0) := by
    convert hasFDerivAt_stereoInvFunAux_comp_coe (-v : E)
    simp
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Eq (LinearMap.range (fderiv Real (Function.comp (Function.comp (stereoInvFun …
  -/
  convert congrArg LinearMap.range (this.comp 0 U.symm.toContinuousLinearEquiv.hasFDerivAt).fderiv
  /-
    case h.e'_3
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Eq (Submodule.span Real (Singleton.singleton ↑v)).orthogonal (LinearMap.rang …
  -/
  symm
  convert
    (U.symm : EuclideanSpace ℝ (Fin n) ≃ₗᵢ[ℝ] (ℝ ∙ (↑(-v) : E))ᗮ).range_comp
      (ℝ ∙ (↑(-v) : E))ᗮ.subtype using 1
  /-
    case h.e'_3
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Eq (Submodule.span Real (Singleton.singleton ↑v)).orthogonal (LinearMap.rang …
  -/
  simp only [Submodule.range_subtype, coe_neg_sphere]
  /-
    case h.e'_3
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Eq (Submodule.span Real (Singleton.singleton ↑v)).orthogonal (Submodule.span …
  -/
  congr 1
  -- we must show `Submodule.span ℝ {v} = Submodule.span ℝ {-v}`
  /-
    case h.e'_3.e_K
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Eq (Submodule.span Real (Singleton.singleton ↑v)) (Submodule.span Real (Sing …
  -/
  apply Submodule.span_eq_span
    /-
      case h.e'_3.e_K.hs
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ HasSubset.Subset (Singleton.singleton ↑v) ↑(Submodule.span Real (Singleton.s …
    -/
  · simp only [Set.singleton_subset_iff, SetLike.mem_coe]
    /-
      case h.e'_3.e_K.hs
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ Membership.mem (Submodule.span Real (Singleton.singleton (Neg.neg ↑v))) ↑v
    -/
    rw [← Submodule.neg_mem_iff]
    /-
      case h.e'_3.e_K.hs
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ Membership.mem (Submodule.span Real (Singleton.singleton (Neg.neg ↑v))) (Neg …
    -/
    exact Submodule.mem_span_singleton_self (-v : E)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.e_K.ht
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ HasSubset.Subset (Singleton.singleton (Neg.neg ↑v)) ↑(Submodule.span Real (S …
    -/
  · simp only [Set.singleton_subset_iff, SetLike.mem_coe]
    /-
      case h.e'_3.e_K.ht
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ Membership.mem (Submodule.span Real (Singleton.singleton ↑v)) (Neg.neg ↑v)
    -/
    rw [Submodule.neg_mem_iff]
    /-
      case h.e'_3.e_K.ht
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace Real E
      n : Nat
      inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      v : ↑(Metric.sphere 0 1)
      U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
      this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
      ⊢ Membership.mem (Submodule.span Real (Singleton.singleton ↑v)) ↑v
    -/
    exact Submodule.mem_span_singleton_self (v : E)
    /-
      🎉 no goals
    -/


/-- Consider the differential of the inclusion of the sphere in `E` at the point `v` as a continuous
linear map from `TangentSpace (𝓡 n) v` to `E`.  This map is injective. -/
theorem mfderiv_coe_sphere_injective {n : ℕ} [Fact (finrank ℝ E = n + 1)] (v : sphere (0 : E) 1) :
    Injective (mfderiv (𝓡 n) 𝓘(ℝ, E) ((↑) : sphere (0 : E) 1 → E) v) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    ⊢ Function.Injective ⇑(mfderiv (modelWithCornersSelf Real (EuclideanSpace Real …
  -/
  rw [((contMDiff_coe_sphere v).mdifferentiableAt le_top).mfderiv]
  simp only [chartAt, stereographic', stereographic_neg_apply, fderivWithin_univ,
    LinearIsometryEquiv.toHomeomorph_symm, LinearIsometryEquiv.coe_toHomeomorph,
    LinearIsometryEquiv.map_zero, mfld_simps]
  let U := (OrthonormalBasis.fromOrthogonalSpanSingleton
      (𝕜 := ℝ) n (ne_zero_of_mem_unit_sphere (-v))).repr
  suffices Injective (fderiv ℝ ((stereoInvFunAux (-v : E) ∘ (↑)) ∘ U.symm) 0) by
    convert this using 3
    apply stereographic'_neg
  have : HasFDerivAt (stereoInvFunAux (-v : E) ∘ (Subtype.val : (ℝ ∙ (↑(-v) : E))ᗮ → E))
      (ℝ ∙ (↑(-v) : E))ᗮ.subtypeL (U.symm 0) := by
    convert hasFDerivAt_stereoInvFunAux_comp_coe (-v : E)
    simp
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val)  …
    ⊢ Function.Injective ⇑(fderiv Real (Function.comp (Function.comp (stereoInvFun …
  -/
  have := congr_arg DFunLike.coe <| (this.comp 0 U.symm.toContinuousLinearEquiv.hasFDerivAt).fderiv
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this✝ : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val) …
    this : Eq ⇑(fderiv Real (Function.comp (Function.comp (stereoInvFunAux (Neg.ne …
    ⊢ Function.Injective ⇑(fderiv Real (Function.comp (Function.comp (stereoInvFun …
  -/
  refine Eq.subst this.symm ?_
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this✝ : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val) …
    this : Eq ⇑(fderiv Real (Function.comp (Function.comp (stereoInvFunAux (Neg.ne …
    ⊢ Function.Injective ⇑((Submodule.span Real (Singleton.singleton ↑(Neg.neg v)) …
  -/
  rw [ContinuousLinearMap.coe_comp', ContinuousLinearEquiv.coe_coe]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace Real E
    n : Nat
    inst✝ : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    v : ↑(Metric.sphere 0 1)
    U : LinearIsometryEquiv (RingHom.id Real) (Subtype fun x => Membership.mem (Su …
    this✝ : HasFDerivAt (Function.comp (stereoInvFunAux (Neg.neg ↑v)) Subtype.val) …
    this : Eq ⇑(fderiv Real (Function.comp (Function.comp (stereoInvFunAux (Neg.ne …
    ⊢ Function.Injective (Function.comp ⇑(Submodule.span Real (Singleton.singleton …
  -/
  simpa [- Subtype.val_injective] using Subtype.val_injective
  /-
    🎉 no goals
  -/


theorem finrank_real_complex_fact' : Fact (finrank ℝ ℂ = 1 + 1) :=
  finrank_real_complex_fact


/-- The unit circle in `ℂ` is a charted space modelled on `EuclideanSpace ℝ (Fin 1)`.  This
follows by definition from the corresponding result for `Metric.Sphere`. -/
instance : ChartedSpace (EuclideanSpace ℝ (Fin 1)) Circle :=
  EuclideanSpace.instChartedSpaceSphere


instance : SmoothManifoldWithCorners (𝓡 1) Circle :=
  EuclideanSpace.instSmoothManifoldWithCornersSphere (E := ℂ)


/-- The unit circle in `ℂ` is a Lie group. -/
instance : LieGroup (𝓡 1) Circle where
  smooth_mul := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContMDiff ((modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))).prod (m …
    -/
    apply ContMDiff.codRestrict_sphere
    /-
      case hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContMDiff ((modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))).prod (m …
    -/
    let c : Circle → ℂ := (↑)
    have h₂ : ContMDiff (𝓘(ℝ, ℂ).prod 𝓘(ℝ, ℂ)) 𝓘(ℝ, ℂ) ⊤ fun z : ℂ × ℂ => z.fst * z.snd := by
      rw [contMDiff_iff]
      exact ⟨continuous_mul, fun x y => contDiff_mul.contDiffOn⟩
    -- Porting note: needed to fill in first 3 arguments or could not figure out typeclasses
    suffices h₁ : ContMDiff ((𝓡 1).prod (𝓡 1)) (𝓘(ℝ, ℂ).prod 𝓘(ℝ, ℂ)) ⊤ (Prod.map c c) from
      h₂.comp h₁
    /-
      case hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : Circle → Complex := Subtype.val
      h₂ : ContMDiff ((modelWithCornersSelf Real Complex).prod (modelWithCornersSelf …
      ⊢ ContMDiff ((modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))).prod (m …
    -/
    apply ContMDiff.prod_map <;>
    /-
      case hf.hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : Circle → Complex := Subtype.val
      h₂ : ContMDiff ((modelWithCornersSelf Real Complex).prod (modelWithCornersSelf …
      ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))) (modelWi …
    -/
    /-
      🎉 no goals
    -/
    exact contMDiff_coe_sphere
    /-
      🎉 no goals
    -/
  smooth_inv := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))) (modelWi …
    -/
    apply ContMDiff.codRestrict_sphere
    /-
      case hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))) (modelWi …
    -/
    simp only [← Circle.coe_inv, Circle.coe_inv_eq_conj]
    /-
      case hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      ⊢ ContMDiff (modelWithCornersSelf Real (EuclideanSpace Real (Fin 1))) (modelWi …
    -/
    exact Complex.conjCLE.contDiff.contMDiff.comp contMDiff_coe_sphere
    /-
      🎉 no goals
    -/


/-- The map `fun t ↦ exp (t * I)` from `ℝ` to the unit circle in `ℂ` is smooth. -/
theorem contMDiff_circleExp : ContMDiff 𝓘(ℝ, ℝ) (𝓡 1) ⊤ Circle.exp :=
  (contDiff_exp.comp (contDiff_id.smul contDiff_const)).contMDiff.codRestrict_sphere _


@[deprecated (since := "2024-07-25")] alias contMDiff_expMapCircle := contMDiff_circleExp


