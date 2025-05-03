/-- Construct a continuous linear map from a linear map and a bound on this linear map.
The fact that the norm of the continuous linear map is then controlled is given in
`LinearMap.mkContinuous_norm_le`. -/
def LinearMap.mkContinuous (C : ℝ) (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) : E →SL[σ] F :=
  ⟨f, AddMonoidHomClass.continuous_of_bound f C h⟩


/-- Construct a continuous linear map from a linear map and the existence of a bound on this linear
map. If you have an explicit bound, use `LinearMap.mkContinuous` instead, as a norm estimate will
follow automatically in `LinearMap.mkContinuous_norm_le`. -/
def LinearMap.mkContinuousOfExistsBound (h : ∃ C, ∀ x, ‖f x‖ ≤ C * ‖x‖) : E →SL[σ] F :=
  ⟨f,
    let ⟨C, hC⟩ := h
    AddMonoidHomClass.continuous_of_bound f C hC⟩


theorem continuous_of_linear_of_boundₛₗ {f : E → F} (h_add : ∀ x y, f (x + y) = f x + f y)
    (h_smul : ∀ (c : 𝕜) (x), f (c • x) = σ c • f x) {C : ℝ} (h_bound : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    Continuous f :=
  let φ : E →ₛₗ[σ] F :=
    { toFun := f
      map_add' := h_add
      map_smul' := h_smul }
  AddMonoidHomClass.continuous_of_bound φ C h_bound


theorem continuous_of_linear_of_bound {f : E → G} (h_add : ∀ x y, f (x + y) = f x + f y)
    (h_smul : ∀ (c : 𝕜) (x), f (c • x) = c • f x) {C : ℝ} (h_bound : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    Continuous f :=
  let φ : E →ₗ[𝕜] G :=
    { toFun := f
      map_add' := h_add
      map_smul' := h_smul }
  AddMonoidHomClass.continuous_of_bound φ C h_bound


@[simp, norm_cast]
theorem LinearMap.mkContinuous_coe (C : ℝ) (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    (f.mkContinuous C h : E →ₛₗ[σ] F) = f :=
  rfl


@[simp]
theorem LinearMap.mkContinuous_apply (C : ℝ) (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) (x : E) :
    f.mkContinuous C h x = f x :=
  rfl


@[simp, norm_cast]
theorem LinearMap.mkContinuousOfExistsBound_coe (h : ∃ C, ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    (f.mkContinuousOfExistsBound h : E →ₛₗ[σ] F) = f :=
  rfl


@[simp]
theorem LinearMap.mkContinuousOfExistsBound_apply (h : ∃ C, ∀ x, ‖f x‖ ≤ C * ‖x‖) (x : E) :
    f.mkContinuousOfExistsBound h x = f x :=
  rfl


theorem antilipschitz_of_bound (f : E →SL[σ] F) {K : ℝ≥0} (h : ∀ x, ‖x‖ ≤ K * ‖f x‖) :
    AntilipschitzWith K f :=
  AddMonoidHomClass.antilipschitz_of_bound _ h


theorem bound_of_antilipschitz (f : E →SL[σ] F) {K : ℝ≥0} (h : AntilipschitzWith K f) (x) :
    ‖x‖ ≤ K * ‖f x‖ :=
  ZeroHomClass.bound_of_antilipschitz _ h x


/-- Construct a continuous linear equivalence from a linear equivalence together with
bounds in both directions. -/
def LinearEquiv.toContinuousLinearEquivOfBounds (e : E ≃ₛₗ[σ] F) (C_to C_inv : ℝ)
    (h_to : ∀ x, ‖e x‖ ≤ C_to * ‖x‖) (h_inv : ∀ x : F, ‖e.symm x‖ ≤ C_inv * ‖x‖) : E ≃SL[σ] F where
  toLinearEquiv := e
  continuous_toFun := AddMonoidHomClass.continuous_of_bound e C_to h_to
  continuous_invFun := AddMonoidHomClass.continuous_of_bound e.symm C_inv h_inv


/-- Reinterpret a linear map `𝕜 →ₗ[𝕜] E` as a continuous linear map. This construction
is generalized to the case of any finite dimensional domain
in `LinearMap.toContinuousLinearMap`. -/
def LinearMap.toContinuousLinearMap₁ (f : 𝕜 →ₗ[𝕜] E) : 𝕜 →L[𝕜] E :=
  f.mkContinuous ‖f 1‖ fun x => by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁴ : SeminormedRing 𝕜
      inst✝³ : Ring 𝕜₂
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) 𝕜 E
      x : 𝕜
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm (f 1)) (Norm.norm x))
    -/
    conv_lhs => rw [← mul_one x]
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁴ : SeminormedRing 𝕜
      inst✝³ : Ring 𝕜₂
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) 𝕜 E
      x : 𝕜
      ⊢ LE.le (Norm.norm (f (HMul.hMul x 1))) (HMul.hMul (Norm.norm (f 1)) (Norm.nor …
    -/
    rw [← smul_eq_mul, f.map_smul, mul_comm]; exact norm_smul_le _ _
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem LinearMap.toContinuousLinearMap₁_coe (f : 𝕜 →ₗ[𝕜] E) :
    (f.toContinuousLinearMap₁ : 𝕜 →ₗ[𝕜] E) = f :=
  rfl


@[simp]
theorem LinearMap.toContinuousLinearMap₁_apply (f : 𝕜 →ₗ[𝕜] E) (x) :
    f.toContinuousLinearMap₁ x = f x :=
  rfl


theorem ContinuousLinearMap.isUniformEmbedding_of_bound {K : ℝ≥0} (hf : ∀ x, ‖x‖ ≤ K * ‖f x‖) :
    IsUniformEmbedding f :=
  (AddMonoidHomClass.antilipschitz_of_bound f hf).isUniformEmbedding f.uniformContinuous


@[deprecated (since := "2024-10-01")]
alias ContinuousLinearMap.uniformEmbedding_of_bound :=
  ContinuousLinearMap.isUniformEmbedding_of_bound


/-- A (semi-)linear map which is a homothety is a continuous linear map.
    Since the field `𝕜` need not have `ℝ` as a subfield, this theorem is not directly deducible from
    the corresponding theorem about isometries plus a theorem about scalar multiplication.  Likewise
    for the other theorems about homotheties in this file.
 -/
def ContinuousLinearMap.ofHomothety (f : E →ₛₗ[σ] F) (a : ℝ) (hf : ∀ x, ‖f x‖ = a * ‖x‖) :
    E →SL[σ] F :=
  f.mkContinuous a fun x => le_of_eq (hf x)


theorem ContinuousLinearEquiv.homothety_inverse (a : ℝ) (ha : 0 < a) (f : E ≃ₛₗ[σ] F) :
    (∀ x : E, ‖f x‖ = a * ‖x‖) → ∀ y : F, ‖f.symm y‖ = a⁻¹ * ‖y‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁷ : Ring 𝕜
    inst✝⁶ : Ring 𝕜₂
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : SeminormedAddCommGroup F
    inst✝³ : Module 𝕜 E
    inst✝² : Module 𝕜₂ F
    σ : RingHom 𝕜 𝕜₂
    σ₂₁ : RingHom 𝕜₂ 𝕜
    inst✝¹ : RingHomInvPair σ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ
    a : Real
    ha : LT.lt 0 a
    f : LinearEquiv σ E F
    ⊢ (∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))) → ∀ (y : F), E …
  -/
  intro hf y
  calc
    ‖f.symm y‖ = a⁻¹ * (a * ‖f.symm y‖) := by
      rw [← mul_assoc, inv_mul_cancel₀ (ne_of_lt ha).symm, one_mul]
    _ = a⁻¹ * ‖f (f.symm y)‖ := by rw [hf]
    _ = a⁻¹ * ‖y‖ := by simp


/-- A linear equivalence which is a homothety is a continuous linear equivalence. -/
noncomputable def ContinuousLinearEquiv.ofHomothety (f : E ≃ₛₗ[σ] F) (a : ℝ) (ha : 0 < a)
    (hf : ∀ x, ‖f x‖ = a * ‖x‖) : E ≃SL[σ] F :=
  LinearEquiv.toContinuousLinearEquivOfBounds f a a⁻¹ (fun x => (hf x).le) fun x =>
    (ContinuousLinearEquiv.homothety_inverse a ha f hf x).le


