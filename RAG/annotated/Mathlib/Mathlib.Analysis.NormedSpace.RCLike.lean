                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝¹ : RCLike 𝕜
                                                                 E : Type u_2
                                                                 inst✝ : NormedAddCommGroup E
                                                                 z : E
                                                                 ⊢ Eq (Norm.norm ↑(Norm.norm z)) (Norm.norm z)
                                                               -/
theorem RCLike.norm_coe_norm {z : E} : ‖(‖z‖ : 𝕜)‖ = ‖z‖ := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Lemma to normalize a vector in a normed space `E` over either `ℂ` or `ℝ` to unit length. -/
@[simp]
theorem norm_smul_inv_norm {x : E} (hx : x ≠ 0) : ‖(‖x‖⁻¹ : 𝕜) • x‖ = 1 := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    hx : Ne x 0
    ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm x)) x)) 1
  -/
  have : ‖x‖ ≠ 0 := by simp [hx]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    hx : Ne x 0
    this : Ne (Norm.norm x) 0
    ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm x)) x)) 1
  -/
  field_simp [norm_smul]
  /-
    🎉 no goals
  -/


/-- Lemma to normalize a vector in a normed space `E` over either `ℂ` or `ℝ` to length `r`. -/
theorem norm_smul_inv_norm' {r : ℝ} (r_nonneg : 0 ≤ r) {x : E} (hx : x ≠ 0) :
    ‖((r : 𝕜) * (‖x‖ : 𝕜)⁻¹) • x‖ = r := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_nonneg : LE.le 0 r
    x : E
    hx : Ne x 0
    ⊢ Eq (Norm.norm (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm x))) x)) r
  -/
  have : ‖x‖ ≠ 0 := by simp [hx]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_nonneg : LE.le 0 r
    x : E
    hx : Ne x 0
    this : Ne (Norm.norm x) 0
    ⊢ Eq (Norm.norm (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm x))) x)) r
  -/
  field_simp [norm_smul, r_nonneg, rclike_simps]
  /-
    🎉 no goals
  -/


theorem LinearMap.bound_of_sphere_bound {r : ℝ} (r_pos : 0 < r) (c : ℝ) (f : E →ₗ[𝕜] 𝕜)
    (h : ∀ z ∈ sphere (0 : E) r, ‖f z‖ ≤ c) (z : E) : ‖f z‖ ≤ c / r * ‖z‖ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
    z : E
    ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm z))
  -/
  by_cases z_zero : z = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E 𝕜
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
      z : E
      z_zero : Eq z 0
      ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm z))
    -/
  · rw [z_zero]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E 𝕜
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
      z : E
      z_zero : Eq z 0
      ⊢ LE.le (Norm.norm (f 0)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm 0))
    -/
    simp only [LinearMap.map_zero, norm_zero, mul_zero]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E 𝕜
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
      z : E
      z_zero : Eq z 0
      ⊢ LE.le 0 0
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
    z : E
    z_zero : Not (Eq z 0)
    ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm z))
  -/
  set z₁ := ((r : 𝕜) * (‖z‖ : 𝕜)⁻¹) • z with hz₁
  have norm_f_z₁ : ‖f z₁‖ ≤ c := by
    apply h
    rw [mem_sphere_zero_iff_norm]
    exact norm_smul_inv_norm' r_pos.le z_zero
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
    z : E
    z_zero : Not (Eq z 0)
    z₁ : E := HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z
    hz₁ : Eq z₁ (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z)
    norm_f_z₁ : LE.le (Norm.norm (f z₁)) c
    ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm z))
  -/
  have r_ne_zero : (r : 𝕜) ≠ 0 := RCLike.ofReal_ne_zero.mpr r_pos.ne'
  have eq : f z = ‖z‖ / r * f z₁ := by
    rw [hz₁, LinearMap.map_smul, smul_eq_mul]
    rw [← mul_assoc, ← mul_assoc, div_mul_cancel₀ _ r_ne_zero, mul_inv_cancel₀, one_mul]
    simp only [z_zero, RCLike.ofReal_eq_zero, norm_eq_zero, Ne, not_false_iff]
  rw [eq, norm_mul, norm_div, RCLike.norm_coe_norm, RCLike.norm_of_nonneg r_pos.le,
    div_mul_eq_mul_div, div_mul_eq_mul_div, mul_comm]
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
    z : E
    z_zero : Not (Eq z 0)
    z₁ : E := HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z
    hz₁ : Eq z₁ (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z)
    norm_f_z₁ : LE.le (Norm.norm (f z₁)) c
    r_ne_zero : Ne (↑r) 0
    eq : Eq (f z) (HMul.hMul (HDiv.hDiv ↑(Norm.norm z) ↑r) (f z₁))
    ⊢ LE.le (HDiv.hDiv (HMul.hMul (Norm.norm (f z₁)) (Norm.norm z)) r) (HDiv.hDiv  …
  -/
  apply div_le_div₀ _ _ r_pos rfl.ge
    /-
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E 𝕜
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
      z : E
      z_zero : Not (Eq z 0)
      z₁ : E := HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z
      hz₁ : Eq z₁ (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z)
      norm_f_z₁ : LE.le (Norm.norm (f z₁)) c
      r_ne_zero : Ne (↑r) 0
      eq : Eq (f z) (HMul.hMul (HDiv.hDiv ↑(Norm.norm z) ↑r) (f z₁))
      ⊢ LE.le 0 (HMul.hMul c (Norm.norm z))
    -/
  · exact mul_nonneg ((norm_nonneg _).trans norm_f_z₁) (norm_nonneg z)
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.sphere 0 r) z → LE.le (Norm.norm (f z)) c
    z : E
    z_zero : Not (Eq z 0)
    z₁ : E := HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z
    hz₁ : Eq z₁ (HSMul.hSMul (HMul.hMul (↑r) (Inv.inv ↑(Norm.norm z))) z)
    norm_f_z₁ : LE.le (Norm.norm (f z₁)) c
    r_ne_zero : Ne (↑r) 0
    eq : Eq (f z) (HMul.hMul (HDiv.hDiv ↑(Norm.norm z) ↑r) (f z₁))
    ⊢ LE.le (HMul.hMul (Norm.norm (f z₁)) (Norm.norm z)) (HMul.hMul c (Norm.norm z))
  -/
  apply mul_le_mul norm_f_z₁ rfl.le (norm_nonneg z) ((norm_nonneg _).trans norm_f_z₁)
  /-
    🎉 no goals
  -/


/-- `LinearMap.bound_of_ball_bound` is a version of this over arbitrary nontrivially normed fields.
It produces a less precise bound so we keep both versions. -/
theorem LinearMap.bound_of_ball_bound' {r : ℝ} (r_pos : 0 < r) (c : ℝ) (f : E →ₗ[𝕜] 𝕜)
    (h : ∀ z ∈ closedBall (0 : E) r, ‖f z‖ ≤ c) (z : E) : ‖f z‖ ≤ c / r * ‖z‖ :=
  f.bound_of_sphere_bound r_pos c (fun z hz => h z hz.le) z


theorem ContinuousLinearMap.opNorm_bound_of_ball_bound {r : ℝ} (r_pos : 0 < r) (c : ℝ)
    (f : E →L[𝕜] 𝕜) (h : ∀ z ∈ closedBall (0 : E) r, ‖f z‖ ≤ c) : ‖f‖ ≤ c / r := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.closedBall 0 r) z → LE.le (Norm.norm (f  …
    ⊢ LE.le (Norm.norm f) (HDiv.hDiv c r)
  -/
  apply ContinuousLinearMap.opNorm_le_bound
    /-
      case hMp
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      h : ∀ (z : E), Membership.mem (Metric.closedBall 0 r) z → LE.le (Norm.norm (f  …
      ⊢ LE.le 0 (HDiv.hDiv c r)
    -/
  · apply div_nonneg _ r_pos.le
    exact
      (norm_nonneg _).trans
        (h 0 (by simp only [norm_zero, mem_closedBall, dist_zero_left, r_pos.le]))
  /-
    case hM
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.closedBall 0 r) z → LE.le (Norm.norm (f  …
    ⊢ ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul (HDiv.hDiv c r) (Norm.norm x))
  -/
  apply LinearMap.bound_of_ball_bound' r_pos
  /-
    case hM.h
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    h : ∀ (z : E), Membership.mem (Metric.closedBall 0 r) z → LE.le (Norm.norm (f  …
    ⊢ ∀ (z : E), Membership.mem (Metric.closedBall 0 r) z → LE.le (Norm.norm (↑f z …
  -/
  exact fun z hz => h z hz
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")]
alias ContinuousLinearMap.op_norm_bound_of_ball_bound :=
  ContinuousLinearMap.opNorm_bound_of_ball_bound


include 𝕜 in
theorem NormedSpace.sphere_nonempty_rclike [Nontrivial E] {r : ℝ} (hr : 0 ≤ r) :
    Nonempty (sphere (0 : E) r) :=
  letI : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 E
  (NormedSpace.sphere_nonempty.mpr hr).coe_sort

