theorem nnnorm_def (f : E →SL[σ₁₂] F) : ‖f‖₊ = sInf { c | ∀ x, ‖f x‖₊ ≤ c * ‖x‖₊ } := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_6
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ Eq (NNNorm.nnnorm f) (InfSet.sInf (setOf fun c => ∀ (x : E), LE.le (NNNorm.n …
  -/
  ext
  /-
    case a
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_6
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ Eq ↑(NNNorm.nnnorm f) ↑(InfSet.sInf (setOf fun c => ∀ (x : E), LE.le (NNNorm …
  -/
  rw [NNReal.coe_sInf, coe_nnnorm, norm_def, NNReal.coe_image]
  simp_rw [← NNReal.coe_le_coe, NNReal.coe_mul, coe_nnnorm, mem_setOf_eq, NNReal.coe_mk,
    exists_prop]


/-- If one controls the norm of every `A x`, then one controls the norm of `A`. -/
theorem opNNNorm_le_bound (f : E →SL[σ₁₂] F) (M : ℝ≥0) (hM : ∀ x, ‖f x‖₊ ≤ M * ‖x‖₊) : ‖f‖₊ ≤ M :=
  opNorm_le_bound f (zero_le M) hM


@[deprecated (since := "2024-02-02")] alias op_nnnorm_le_bound := opNNNorm_le_bound


/-- If one controls the norm of every `A x`, `‖x‖₊ ≠ 0`, then one controls the norm of `A`. -/
theorem opNNNorm_le_bound' (f : E →SL[σ₁₂] F) (M : ℝ≥0) (hM : ∀ x, ‖x‖₊ ≠ 0 → ‖f x‖₊ ≤ M * ‖x‖₊) :
    ‖f‖₊ ≤ M :=
                                                        /-
                                                          𝕜 : Type u_1
                                                          𝕜₂ : Type u_2
                                                          E : Type u_4
                                                          F : Type u_6
                                                          inst✝⁶ : SeminormedAddCommGroup E
                                                          inst✝⁵ : SeminormedAddCommGroup F
                                                          inst✝⁴ : NontriviallyNormedField 𝕜
                                                          inst✝³ : NontriviallyNormedField 𝕜₂
                                                          inst✝² : NormedSpace 𝕜 E
                                                          inst✝¹ : NormedSpace 𝕜₂ F
                                                          σ₁₂ : RingHom 𝕜 𝕜₂
                                                          inst✝ : RingHomIsometric σ₁₂
                                                          f : ContinuousLinearMap σ₁₂ E F
                                                          M : NNReal
                                                          hM : ∀ (x : E), Ne (NNNorm.nnnorm x) 0 → LE.le (NNNorm.nnnorm (f x)) (HMul.hMu …
                                                          x : E
                                                          hx : Ne (Norm.norm x) 0
                                                          ⊢ Ne (NNNorm.nnnorm x) 0
                                                        -/
  opNorm_le_bound' f (zero_le M) fun x hx => hM x <| by rwa [← NNReal.coe_ne_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[deprecated (since := "2024-02-02")] alias op_nnnorm_le_bound' := opNNNorm_le_bound'


/-- For a continuous real linear map `f`, if one controls the norm of every `f x`, `‖x‖₊ = 1`, then
one controls the norm of `f`. -/
theorem opNNNorm_le_of_unit_nnnorm [NormedSpace ℝ E] [NormedSpace ℝ F] {f : E →L[ℝ] F} {C : ℝ≥0}
    (hf : ∀ x, ‖x‖₊ = 1 → ‖f x‖₊ ≤ C) : ‖f‖₊ ≤ C :=
                                                             /-
                                                               E : Type u_4
                                                               F : Type u_6
                                                               inst✝³ : SeminormedAddCommGroup E
                                                               inst✝² : SeminormedAddCommGroup F
                                                               inst✝¹ : NormedSpace Real E
                                                               inst✝ : NormedSpace Real F
                                                               f : ContinuousLinearMap (RingHom.id Real) E F
                                                               C : NNReal
                                                               hf : ∀ (x : E), Eq (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm (f x)) C
                                                               x : E
                                                               hx : Eq (Norm.norm x) 1
                                                               ⊢ Eq (NNNorm.nnnorm x) 1
                                                             -/
  opNorm_le_of_unit_norm C.coe_nonneg fun x hx => hf x <| by rwa [← NNReal.coe_eq_one]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-02-02")]
alias op_nnnorm_le_of_unit_nnnorm := opNNNorm_le_of_unit_nnnorm


theorem opNNNorm_le_of_lipschitz {f : E →SL[σ₁₂] F} {K : ℝ≥0} (hf : LipschitzWith K f) :
    ‖f‖₊ ≤ K :=
  opNorm_le_of_lipschitz hf


@[deprecated (since := "2024-02-02")] alias op_nnnorm_le_of_lipschitz := opNNNorm_le_of_lipschitz


theorem opNNNorm_eq_of_bounds {φ : E →SL[σ₁₂] F} (M : ℝ≥0) (h_above : ∀ x, ‖φ x‖₊ ≤ M * ‖x‖₊)
    (h_below : ∀ N, (∀ x, ‖φ x‖₊ ≤ N * ‖x‖₊) → M ≤ N) : ‖φ‖₊ = M :=
  Subtype.ext <| opNorm_eq_of_bounds (zero_le M) h_above <| Subtype.forall'.mpr h_below


@[deprecated (since := "2024-02-02")] alias op_nnnorm_eq_of_bounds := opNNNorm_eq_of_bounds


theorem opNNNorm_le_iff {f : E →SL[σ₁₂] F} {C : ℝ≥0} : ‖f‖₊ ≤ C ↔ ∀ x, ‖f x‖₊ ≤ C * ‖x‖₊ :=
  opNorm_le_iff C.2


@[deprecated (since := "2024-02-02")] alias op_nnnorm_le_iff := opNNNorm_le_iff


theorem isLeast_opNNNorm : IsLeast {C : ℝ≥0 | ∀ x, ‖f x‖₊ ≤ C * ‖x‖₊} ‖f‖₊ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_6
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ IsLeast (setOf fun C => ∀ (x : E), LE.le (NNNorm.nnnorm (f x)) (HMul.hMul C  …
  -/
  simpa only [← opNNNorm_le_iff] using isLeast_Ici
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias isLeast_op_nnnorm := isLeast_opNNNorm


theorem opNNNorm_comp_le [RingHomIsometric σ₁₃] (f : E →SL[σ₁₂] F) : ‖h.comp f‖₊ ≤ ‖h‖₊ * ‖f‖₊ :=
  opNorm_comp_le h f


@[deprecated (since := "2024-02-02")] alias op_nnnorm_comp_le := opNNNorm_comp_le


theorem le_opNNNorm : ‖f x‖₊ ≤ ‖f‖₊ * ‖x‖₊ :=
  f.le_opNorm x


@[deprecated (since := "2024-02-02")] alias le_op_nnnorm := le_opNNNorm


theorem nndist_le_opNNNorm (x y : E) : nndist (f x) (f y) ≤ ‖f‖₊ * nndist x y :=
  dist_le_opNorm f x y


@[deprecated (since := "2024-02-02")] alias nndist_le_op_nnnorm := nndist_le_opNNNorm


/-- continuous linear maps are Lipschitz continuous. -/
theorem lipschitz : LipschitzWith ‖f‖₊ f :=
  AddMonoidHomClass.lipschitz_of_bound_nnnorm f _ f.le_opNNNorm


/-- Evaluation of a continuous linear map `f` at a point is Lipschitz continuous in `f`. -/
theorem lipschitz_apply (x : E) : LipschitzWith ‖x‖₊ fun f : E →SL[σ₁₂] F => f x :=
  lipschitzWith_iff_norm_sub_le.2 fun f g => ((f - g).le_opNorm x).trans_eq (mul_comm _ _)


theorem exists_mul_lt_apply_of_lt_opNNNorm (f : E →SL[σ₁₂] F) {r : ℝ≥0} (hr : r < ‖f‖₊) :
    ∃ x, r * ‖x‖₊ < ‖f x‖₊ := by
  simpa only [not_forall, not_le, Set.mem_setOf] using
    not_mem_of_lt_csInf (nnnorm_def f ▸ hr : r < sInf { c : ℝ≥0 | ∀ x, ‖f x‖₊ ≤ c * ‖x‖₊ })
      (OrderBot.bddBelow _)


@[deprecated (since := "2024-02-02")]
alias exists_mul_lt_apply_of_lt_op_nnnorm := exists_mul_lt_apply_of_lt_opNNNorm


theorem exists_mul_lt_of_lt_opNorm (f : E →SL[σ₁₂] F) {r : ℝ} (hr₀ : 0 ≤ r) (hr : r < ‖f‖) :
    ∃ x, r * ‖x‖ < ‖f x‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_6
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : Real
    hr₀ : LE.le 0 r
    hr : LT.lt r (Norm.norm f)
    ⊢ Exists fun x => LT.lt (HMul.hMul r (Norm.norm x)) (Norm.norm (f x))
  -/
  lift r to ℝ≥0 using hr₀
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_6
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt (↑r) (Norm.norm f)
    ⊢ Exists fun x => LT.lt (HMul.hMul (↑r) (Norm.norm x)) (Norm.norm (f x))
  -/
  exact f.exists_mul_lt_apply_of_lt_opNNNorm hr
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")]
alias exists_mul_lt_of_lt_op_norm := exists_mul_lt_of_lt_opNorm


theorem exists_lt_apply_of_lt_opNNNorm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E]
    [SeminormedAddCommGroup F] [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂}
    [NormedSpace 𝕜 E] [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) {r : ℝ≥0}
    (hr : r < ‖f‖₊) : ∃ x : E, ‖x‖₊ < 1 ∧ r < ‖f x‖₊ := by
  /-
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    ⊢ Exists fun x => And (LT.lt (NNNorm.nnnorm x) 1) (LT.lt r (NNNorm.nnnorm (f x …
  -/
  obtain ⟨y, hy⟩ := f.exists_mul_lt_apply_of_lt_opNNNorm hr
  have hy' : ‖y‖₊ ≠ 0 :=
    nnnorm_ne_zero_iff.2 fun heq => by
      simp [heq, nnnorm_zero, map_zero, not_lt_zero'] at hy
  /-
    case intro
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    y : E
    hy : LT.lt (HMul.hMul r (NNNorm.nnnorm y)) (NNNorm.nnnorm (f y))
    hy' : Ne (NNNorm.nnnorm y) 0
    ⊢ Exists fun x => And (LT.lt (NNNorm.nnnorm x) 1) (LT.lt r (NNNorm.nnnorm (f x …
  -/
  have hfy : ‖f y‖₊ ≠ 0 := (zero_le'.trans_lt hy).ne'
  rw [← inv_inv ‖f y‖₊, NNReal.lt_inv_iff_mul_lt (inv_ne_zero hfy), mul_assoc, mul_comm ‖y‖₊, ←
    mul_assoc, ← NNReal.lt_inv_iff_mul_lt hy'] at hy
  /-
    case intro
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    y : E
    hy : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (Inv.inv (NNNorm.nnno …
    hy' : Ne (NNNorm.nnnorm y) 0
    hfy : Ne (NNNorm.nnnorm (f y)) 0
    ⊢ Exists fun x => And (LT.lt (NNNorm.nnnorm x) 1) (LT.lt r (NNNorm.nnnorm (f x …
  -/
  obtain ⟨k, hk₁, hk₂⟩ := NormedField.exists_lt_nnnorm_lt 𝕜 hy
  /-
    case intro.intro.intro
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    y : E
    hy : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (Inv.inv (NNNorm.nnno …
    hy' : Ne (NNNorm.nnnorm y) 0
    hfy : Ne (NNNorm.nnnorm (f y)) 0
    k : 𝕜
    hk₁ : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (NNNorm.nnnorm k)
    hk₂ : LT.lt (NNNorm.nnnorm k) (Inv.inv (NNNorm.nnnorm y))
    ⊢ Exists fun x => And (LT.lt (NNNorm.nnnorm x) 1) (LT.lt r (NNNorm.nnnorm (f x …
  -/
  refine ⟨k • y, (nnnorm_smul k y).symm ▸ (NNReal.lt_inv_iff_mul_lt hy').1 hk₂, ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    y : E
    hy : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (Inv.inv (NNNorm.nnno …
    hy' : Ne (NNNorm.nnnorm y) 0
    hfy : Ne (NNNorm.nnnorm (f y)) 0
    k : 𝕜
    hk₁ : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (NNNorm.nnnorm k)
    hk₂ : LT.lt (NNNorm.nnnorm k) (Inv.inv (NNNorm.nnnorm y))
    ⊢ LT.lt r (NNNorm.nnnorm (f (HSMul.hSMul k y)))
  -/
  have : ‖σ₁₂ k‖₊ = ‖k‖₊ := Subtype.ext RingHomIsometric.is_iso
  /-
    case intro.intro.intro
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : NNReal
    hr : LT.lt r (NNNorm.nnnorm f)
    y : E
    hy : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (Inv.inv (NNNorm.nnno …
    hy' : Ne (NNNorm.nnnorm y) 0
    hfy : Ne (NNNorm.nnnorm (f y)) 0
    k : 𝕜
    hk₁ : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm (f y)))) (NNNorm.nnnorm k)
    hk₂ : LT.lt (NNNorm.nnnorm k) (Inv.inv (NNNorm.nnnorm y))
    this : Eq (NNNorm.nnnorm (σ₁₂ k)) (NNNorm.nnnorm k)
    ⊢ LT.lt r (NNNorm.nnnorm (f (HSMul.hSMul k y)))
  -/
  rwa [map_smulₛₗ f, nnnorm_smul, ← div_lt_iff₀ hfy.bot_lt, div_eq_mul_inv, this]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")]
alias exists_lt_apply_of_lt_op_nnnorm := exists_lt_apply_of_lt_opNNNorm


theorem exists_lt_apply_of_lt_opNorm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E]
    [SeminormedAddCommGroup F] [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂}
    [NormedSpace 𝕜 E] [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) {r : ℝ}
    (hr : r < ‖f‖) : ∃ x : E, ‖x‖ < 1 ∧ r < ‖f x‖ := by
  /-
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    r : Real
    hr : LT.lt r (Norm.norm f)
    ⊢ Exists fun x => And (LT.lt (Norm.norm x) 1) (LT.lt r (Norm.norm (f x)))
  -/
  by_cases hr₀ : r < 0
    /-
      case pos
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      r : Real
      hr : LT.lt r (Norm.norm f)
      hr₀ : LT.lt r 0
      ⊢ Exists fun x => And (LT.lt (Norm.norm x) 1) (LT.lt r (Norm.norm (f x)))
    -/
  · exact ⟨0, by simpa using hr₀⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      r : Real
      hr : LT.lt r (Norm.norm f)
      hr₀ : Not (LT.lt r 0)
      ⊢ Exists fun x => And (LT.lt (Norm.norm x) 1) (LT.lt r (Norm.norm (f x)))
    -/
  · lift r to ℝ≥0 using not_lt.1 hr₀
    /-
      case neg.intro
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      r : NNReal
      hr : LT.lt (↑r) (Norm.norm f)
      hr₀ : Not (LT.lt (↑r) 0)
      ⊢ Exists fun x => And (LT.lt (Norm.norm x) 1) (LT.lt (↑r) (Norm.norm (f x)))
    -/
    exact f.exists_lt_apply_of_lt_opNNNorm hr
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")]
alias exists_lt_apply_of_lt_op_norm := exists_lt_apply_of_lt_opNorm


theorem sSup_unit_ball_eq_nnnorm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E]
    [SeminormedAddCommGroup F] [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂}
    [NormedSpace 𝕜 E] [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) :
    sSup ((fun x => ‖f x‖₊) '' ball 0 1) = ‖f‖₊ := by
  refine csSup_eq_of_forall_le_of_forall_lt_exists_gt ((nonempty_ball.mpr zero_lt_one).image _) ?_
    fun ub hub => ?_
    /-
      case refine_1
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ⊢ ∀ (a : NNReal), Membership.mem (Set.image (fun x => NNNorm.nnnorm (f x)) (Me …
    -/
  · rintro - ⟨x, hx, rfl⟩
    /-
      case refine_1.intro.intro
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      x : E
      hx : Membership.mem (Metric.ball 0 1) x
      ⊢ LE.le ((fun x => NNNorm.nnnorm (f x)) x) (NNNorm.nnnorm f)
    -/
    simpa only [mul_one] using f.le_opNorm_of_le (mem_ball_zero_iff.1 hx).le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ub : NNReal
      hub : LT.lt ub (NNNorm.nnnorm f)
      ⊢ Exists fun a => And (Membership.mem (Set.image (fun x => NNNorm.nnnorm (f x) …
    -/
  · obtain ⟨x, hx, hxf⟩ := f.exists_lt_apply_of_lt_opNNNorm hub
    /-
      case refine_2.intro.intro
      𝕜 : Type u_11
      𝕜₂ : Type u_12
      E : Type u_13
      F : Type u_14
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : DenselyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ub : NNReal
      hub : LT.lt ub (NNNorm.nnnorm f)
      x : E
      hx : LT.lt (NNNorm.nnnorm x) 1
      hxf : LT.lt ub (NNNorm.nnnorm (f x))
      ⊢ Exists fun a => And (Membership.mem (Set.image (fun x => NNNorm.nnnorm (f x) …
    -/
    exact ⟨_, ⟨x, mem_ball_zero_iff.2 hx, rfl⟩, hxf⟩
    /-
      🎉 no goals
    -/


theorem sSup_unit_ball_eq_norm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E] [SeminormedAddCommGroup F]
    [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂} [NormedSpace 𝕜 E]
    [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) :
    sSup ((fun x => ‖f x‖) '' ball 0 1) = ‖f‖ := by
  /-
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ Eq (SupSet.sSup (Set.image (fun x => Norm.norm (f x)) (Metric.ball 0 1))) (N …
  -/
  simpa only [NNReal.coe_sSup, Set.image_image] using NNReal.coe_inj.2 f.sSup_unit_ball_eq_nnnorm
  /-
    🎉 no goals
  -/


theorem sSup_unitClosedBall_eq_nnnorm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E]
    [SeminormedAddCommGroup F] [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂}
    [NormedSpace 𝕜 E] [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) :
    sSup ((fun x => ‖f x‖₊) '' closedBall 0 1) = ‖f‖₊ := by
  have hbdd : ∀ y ∈ (fun x => ‖f x‖₊) '' closedBall 0 1, y ≤ ‖f‖₊ := by
    rintro - ⟨x, hx, rfl⟩
    exact f.unit_le_opNorm x (mem_closedBall_zero_iff.1 hx)
  /-
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    hbdd : ∀ (y : NNReal), Membership.mem (Set.image (fun x => NNNorm.nnnorm (f x) …
    ⊢ Eq (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (f x)) (Metric.closedBall …
  -/
  refine le_antisymm (csSup_le ((nonempty_closedBall.mpr zero_le_one).image _) hbdd) ?_
  /-
    𝕜 : Type u_11
    𝕜₂ : Type u_12
    E : Type u_13
    F : Type u_14
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : DenselyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    hbdd : ∀ (y : NNReal), Membership.mem (Set.image (fun x => NNNorm.nnnorm (f x) …
    ⊢ LE.le (NNNorm.nnnorm f) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (f x …
  -/
  rw [← sSup_unit_ball_eq_nnnorm]
  exact csSup_le_csSup ⟨‖f‖₊, hbdd⟩ ((nonempty_ball.2 zero_lt_one).image _)
    (Set.image_subset _ ball_subset_closedBall)


@[deprecated (since := "2024-12-01")]
alias sSup_closed_unit_ball_eq_nnnorm := sSup_unitClosedBall_eq_nnnorm


theorem sSup_unitClosedBall_eq_norm {𝕜 𝕜₂ E F : Type*} [NormedAddCommGroup E]
    [SeminormedAddCommGroup F] [DenselyNormedField 𝕜] [NontriviallyNormedField 𝕜₂] {σ₁₂ : 𝕜 →+* 𝕜₂}
    [NormedSpace 𝕜 E] [NormedSpace 𝕜₂ F] [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) :
    sSup ((fun x => ‖f x‖) '' closedBall 0 1) = ‖f‖ := by
  simpa only [NNReal.coe_sSup, Set.image_image] using
    NNReal.coe_inj.2 f.sSup_unitClosedBall_eq_nnnorm


@[deprecated (since := "2024-12-01")]
alias sSup_closed_unit_ball_eq_norm := sSup_unitClosedBall_eq_norm


protected theorem lipschitz : LipschitzWith ‖(e : E →SL[σ₁₂] F)‖₊ e :=
  (e : E →SL[σ₁₂] F).lipschitz


