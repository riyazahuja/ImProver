/-- If `‖x‖ = 0` and `f` is continuous then `‖f x‖ = 0`. -/
theorem norm_image_of_norm_zero [SemilinearMapClass 𝓕 σ₁₂ E F] (f : 𝓕) (hf : Continuous f) {x : E}
    (hx : ‖x‖ = 0) : ‖f x‖ = 0 := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    𝓕 : Type u_8
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : SemilinearMapClass 𝓕 σ₁₂ E F
    f : 𝓕
    hf : Continuous ⇑f
    x : E
    hx : Eq (Norm.norm x) 0
    ⊢ Eq (Norm.norm (f x)) 0
  -/
  rw [← mem_closure_zero_iff_norm, ← specializes_iff_mem_closure, ← map_zero f] at *
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    𝓕 : Type u_8
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : SemilinearMapClass 𝓕 σ₁₂ E F
    f : 𝓕
    hf : Continuous ⇑f
    x : E
    hx : Specializes 0 x
    ⊢ Specializes (f 0) (f x)
  -/
  exact hx.map hf
  /-
    🎉 no goals
  -/


theorem SemilinearMapClass.bound_of_shell_semi_normed [SemilinearMapClass 𝓕 σ₁₂ E F] (f : 𝓕)
    {ε C : ℝ} (ε_pos : 0 < ε) {c : 𝕜} (hc : 1 < ‖c‖)
    (hf : ∀ x, ε / ‖c‖ ≤ ‖x‖ → ‖x‖ < ε → ‖f x‖ ≤ C * ‖x‖) {x : E} (hx : ‖x‖ ≠ 0) :
    ‖f x‖ ≤ C * ‖x‖ :=
  (normSeminorm 𝕜 E).bound_of_shell ((normSeminorm 𝕜₂ F).comp ⟨⟨f, map_add f⟩, map_smulₛₗ f⟩)
    ε_pos hc hf hx


/-- A continuous linear map between seminormed spaces is bounded when the field is nontrivially
normed. The continuity ensures boundedness on a ball of some radius `ε`. The nontriviality of the
norm is then used to rescale any element into an element of norm in `[ε/C, ε]`, whose image has a
controlled norm. The norm control for the original element follows by rescaling. -/
theorem SemilinearMapClass.bound_of_continuous [SemilinearMapClass 𝓕 σ₁₂ E F] (f : 𝓕)
    (hf : Continuous f) : ∃ C, 0 < C ∧ ∀ x : E, ‖f x‖ ≤ C * ‖x‖ :=
  let φ : E →ₛₗ[σ₁₂] F := ⟨⟨f, map_add f⟩, map_smulₛₗ f⟩
  ((normSeminorm 𝕜₂ F).comp φ).bound_of_continuous_normedSpace (continuous_norm.comp hf)


theorem bound [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) : ∃ C, 0 < C ∧ ∀ x : E, ‖f x‖ ≤ C * ‖x‖ :=
  SemilinearMapClass.bound_of_continuous f f.2


/-- Given a unit-length element `x` of a normed space `E` over a field `𝕜`, the natural linear
    isometry map from `𝕜` to `E` by taking multiples of `x`. -/
def _root_.LinearIsometry.toSpanSingleton {v : E} (hv : ‖v‖ = 1) : 𝕜 →ₗᵢ[𝕜] E :=
                                                                  /-
                                                                    𝕜 : Type u_1
                                                                    𝕜₂ : Type u_2
                                                                    𝕜₃ : Type u_3
                                                                    E : Type u_4
                                                                    F : Type u_5
                                                                    Fₗ : Type u_6
                                                                    G : Type u_7
                                                                    𝓕 : Type u_8
                                                                    inst✝¹² : SeminormedAddCommGroup E
                                                                    inst✝¹¹ : SeminormedAddCommGroup F
                                                                    inst✝¹⁰ : SeminormedAddCommGroup Fₗ
                                                                    inst✝⁹ : SeminormedAddCommGroup G
                                                                    inst✝⁸ : NontriviallyNormedField 𝕜
                                                                    inst✝⁷ : NontriviallyNormedField 𝕜₂
                                                                    inst✝⁶ : NontriviallyNormedField 𝕜₃
                                                                    inst✝⁵ : NormedSpace 𝕜 E
                                                                    inst✝⁴ : NormedSpace 𝕜₂ F
                                                                    inst✝³ : NormedSpace 𝕜 Fₗ
                                                                    inst✝² : NormedSpace 𝕜₃ G
                                                                    σ₁₂ : RingHom 𝕜 𝕜₂
                                                                    σ₂₃ : RingHom 𝕜₂ 𝕜₃
                                                                    σ₁₃ : RingHom 𝕜 𝕜₃
                                                                    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                                                    inst✝ : FunLike 𝓕 E F
                                                                    v : E
                                                                    hv : Eq (Norm.norm v) 1
                                                                    x : 𝕜
                                                                    ⊢ Eq (Norm.norm (__src✝ x)) (Norm.norm x)
                                                                  -/
  { LinearMap.toSpanSingleton 𝕜 E v with norm_map' := fun x => by simp [norm_smul, hv] }
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem _root_.LinearIsometry.toSpanSingleton_apply {v : E} (hv : ‖v‖ = 1) (a : 𝕜) :
    LinearIsometry.toSpanSingleton 𝕜 E hv a = a • v :=
  rfl


@[simp]
theorem _root_.LinearIsometry.coe_toSpanSingleton {v : E} (hv : ‖v‖ = 1) :
    (LinearIsometry.toSpanSingleton 𝕜 E hv).toLinearMap = LinearMap.toSpanSingleton 𝕜 E v :=
  rfl


/-- The operator norm of a continuous linear map is the inf of all its bounds. -/
def opNorm (f : E →SL[σ₁₂] F) :=
  sInf { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ }


instance hasOpNorm : Norm (E →SL[σ₁₂] F) :=
  ⟨opNorm⟩


theorem norm_def (f : E →SL[σ₁₂] F) : ‖f‖ = sInf { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  rfl

-- So that invocations of `le_csInf` make sense: we show that the set of
-- bounds is nonempty and bounded below.

theorem bounds_nonempty [RingHomIsometric σ₁₂] {f : E →SL[σ₁₂] F} :
    ∃ c, c ∈ { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  let ⟨M, hMp, hMb⟩ := f.bound
  ⟨M, le_of_lt hMp, hMb⟩


theorem bounds_bddBelow {f : E →SL[σ₁₂] F} : BddBelow { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  ⟨0, fun _ ⟨hn, _⟩ => hn⟩


theorem isLeast_opNorm [RingHomIsometric σ₁₂] (f : E →SL[σ₁₂] F) :
    IsLeast {c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖} ‖f‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ IsLeast (setOf fun c => And (LE.le 0 c) (∀ (x : E), LE.le (Norm.norm (f x))  …
  -/
  refine IsClosed.isLeast_csInf ?_ bounds_nonempty bounds_bddBelow
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ IsClosed (setOf fun c => And (LE.le 0 c) (∀ (x : E), LE.le (Norm.norm (f x)) …
  -/
  simp only [setOf_and, setOf_forall]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ⊢ IsClosed (Inter.inter (setOf fun a => LE.le 0 a) (Set.iInter fun i => setOf  …
  -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  refine isClosed_Ici.inter <| isClosed_iInter fun _ ↦ isClosed_le ?_ ?_ <;> continuity
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[deprecated (since := "2024-02-02")] alias isLeast_op_norm := isLeast_opNorm


/-- If one controls the norm of every `A x`, then one controls the norm of `A`. -/
theorem opNorm_le_bound (f : E →SL[σ₁₂] F) {M : ℝ} (hMp : 0 ≤ M) (hM : ∀ x, ‖f x‖ ≤ M * ‖x‖) :
    ‖f‖ ≤ M :=
  csInf_le bounds_bddBelow ⟨hMp, hM⟩


@[deprecated (since := "2024-02-02")] alias op_norm_le_bound := opNorm_le_bound


/-- If one controls the norm of every `A x`, `‖x‖ ≠ 0`, then one controls the norm of `A`. -/
theorem opNorm_le_bound' (f : E →SL[σ₁₂] F) {M : ℝ} (hMp : 0 ≤ M)
    (hM : ∀ x, ‖x‖ ≠ 0 → ‖f x‖ ≤ M * ‖x‖) : ‖f‖ ≤ M :=
  opNorm_le_bound f hMp fun x =>
    (ne_or_eq ‖x‖ 0).elim (hM x) fun h => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_4
        F : Type u_5
        inst✝⁵ : SeminormedAddCommGroup E
        inst✝⁴ : SeminormedAddCommGroup F
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NontriviallyNormedField 𝕜₂
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜₂ F
        σ₁₂ : RingHom 𝕜 𝕜₂
        f : ContinuousLinearMap σ₁₂ E F
        M : Real
        hMp : LE.le 0 M
        hM : ∀ (x : E), Ne (Norm.norm x) 0 → LE.le (Norm.norm (f x)) (HMul.hMul M (Nor …
        x : E
        h : Eq (Norm.norm x) 0
        ⊢ LE.le (Norm.norm (f x)) (HMul.hMul M (Norm.norm x))
      -/
      simp only [h, mul_zero, norm_image_of_norm_zero f f.2 h, le_refl]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-02-02")] alias op_norm_le_bound' := opNorm_le_bound'


theorem opNorm_le_of_lipschitz {f : E →SL[σ₁₂] F} {K : ℝ≥0} (hf : LipschitzWith K f) : ‖f‖ ≤ K :=
  f.opNorm_le_bound K.2 fun x => by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NontriviallyNormedField 𝕜₂
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      f : ContinuousLinearMap σ₁₂ E F
      K : NNReal
      hf : LipschitzWith K ⇑f
      x : E
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (↑K) (Norm.norm x))
    -/
    simpa only [dist_zero_right, f.map_zero] using hf.dist_le_mul x 0
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_lipschitz := opNorm_le_of_lipschitz


theorem opNorm_eq_of_bounds {φ : E →SL[σ₁₂] F} {M : ℝ} (M_nonneg : 0 ≤ M)
    (h_above : ∀ x, ‖φ x‖ ≤ M * ‖x‖) (h_below : ∀ N ≥ 0, (∀ x, ‖φ x‖ ≤ N * ‖x‖) → M ≤ N) :
    ‖φ‖ = M :=
  le_antisymm (φ.opNorm_le_bound M_nonneg h_above)
    ((le_csInf_iff ContinuousLinearMap.bounds_bddBelow ⟨M, M_nonneg, h_above⟩).mpr
      fun N ⟨N_nonneg, hN⟩ => h_below N N_nonneg hN)


@[deprecated (since := "2024-02-02")] alias op_norm_eq_of_bounds := opNorm_eq_of_bounds


                                                         /-
                                                           𝕜 : Type u_1
                                                           𝕜₂ : Type u_2
                                                           E : Type u_4
                                                           F : Type u_5
                                                           inst✝⁵ : SeminormedAddCommGroup E
                                                           inst✝⁴ : SeminormedAddCommGroup F
                                                           inst✝³ : NontriviallyNormedField 𝕜
                                                           inst✝² : NontriviallyNormedField 𝕜₂
                                                           inst✝¹ : NormedSpace 𝕜 E
                                                           inst✝ : NormedSpace 𝕜₂ F
                                                           σ₁₂ : RingHom 𝕜 𝕜₂
                                                           f : ContinuousLinearMap σ₁₂ E F
                                                           ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
                                                         -/
theorem opNorm_neg (f : E →SL[σ₁₂] F) : ‖-f‖ = ‖f‖ := by simp only [norm_def, neg_apply, norm_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-02-02")] alias op_norm_neg := opNorm_neg


theorem opNorm_nonneg (f : E →SL[σ₁₂] F) : 0 ≤ ‖f‖ :=
  Real.sInf_nonneg fun _ ↦ And.left


@[deprecated (since := "2024-02-02")] alias op_norm_nonneg := opNorm_nonneg


/-- The norm of the `0` operator is `0`. -/
theorem opNorm_zero : ‖(0 : E →SL[σ₁₂] F)‖ = 0 :=
                                                   /-
                                                     𝕜 : Type u_1
                                                     𝕜₂ : Type u_2
                                                     E : Type u_4
                                                     F : Type u_5
                                                     inst✝⁵ : SeminormedAddCommGroup E
                                                     inst✝⁴ : SeminormedAddCommGroup F
                                                     inst✝³ : NontriviallyNormedField 𝕜
                                                     inst✝² : NontriviallyNormedField 𝕜₂
                                                     inst✝¹ : NormedSpace 𝕜 E
                                                     inst✝ : NormedSpace 𝕜₂ F
                                                     σ₁₂ : RingHom 𝕜 𝕜₂
                                                     x✝ : E
                                                     ⊢ LE.le (Norm.norm (0 x✝)) (HMul.hMul 0 (Norm.norm x✝))
                                                   -/
  le_antisymm (opNorm_le_bound _ le_rfl fun _ ↦ by simp) (opNorm_nonneg _)
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-02-02")] alias op_norm_zero := opNorm_zero


/-- The norm of the identity is at most `1`. It is in fact `1`, except when the space is trivial
where it is `0`. It means that one can not do better than an inequality in general. -/
theorem norm_id_le : ‖id 𝕜 E‖ ≤ 1 :=
                                            /-
                                              𝕜 : Type u_1
                                              E : Type u_4
                                              inst✝² : SeminormedAddCommGroup E
                                              inst✝¹ : NontriviallyNormedField 𝕜
                                              inst✝ : NormedSpace 𝕜 E
                                              x : E
                                              ⊢ LE.le (Norm.norm ((ContinuousLinearMap.id 𝕜 E) x)) (HMul.hMul 1 (Norm.norm x))
                                            -/
  opNorm_le_bound _ zero_le_one fun x => by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- The fundamental property of the operator norm: `‖f x‖ ≤ ‖f‖ * ‖x‖`. -/
theorem le_opNorm : ‖f x‖ ≤ ‖f‖ * ‖x‖ := (isLeast_opNorm f).1.2 x


@[deprecated (since := "2024-02-02")] alias le_op_norm := le_opNorm


theorem dist_le_opNorm (x y : E) : dist (f x) (f y) ≤ ‖f‖ * dist x y := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    x y : E
    ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul (Norm.norm f) (Dist.dist x y))
  -/
  simp_rw [dist_eq_norm, ← map_sub, f.le_opNorm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias dist_le_op_norm := dist_le_opNorm


theorem le_of_opNorm_le_of_le {x} {a b : ℝ} (hf : ‖f‖ ≤ a) (hx : ‖x‖ ≤ b) :
    ‖f x‖ ≤ a * b :=
                              /-
                                𝕜 : Type u_1
                                𝕜₂ : Type u_2
                                E : Type u_4
                                F : Type u_5
                                inst✝⁶ : SeminormedAddCommGroup E
                                inst✝⁵ : SeminormedAddCommGroup F
                                inst✝⁴ : NontriviallyNormedField 𝕜
                                inst✝³ : NontriviallyNormedField 𝕜₂
                                inst✝² : NormedSpace 𝕜 E
                                inst✝¹ : NormedSpace 𝕜₂ F
                                σ₁₂ : RingHom 𝕜 𝕜₂
                                inst✝ : RingHomIsometric σ₁₂
                                f : ContinuousLinearMap σ₁₂ E F
                                x : E
                                a b : Real
                                hf : LE.le (Norm.norm f) a
                                hx : LE.le (Norm.norm x) b
                                ⊢ LE.le (HMul.hMul (Norm.norm f) (Norm.norm x)) (HMul.hMul a b)
                              -/
  (f.le_opNorm x).trans <| by gcongr; exact (opNorm_nonneg f).trans hf
                                      /-
                                        🎉 no goals
                                      -/


@[deprecated (since := "2024-02-02")] alias le_of_op_norm_le_of_le := le_of_opNorm_le_of_le


theorem le_opNorm_of_le {c : ℝ} {x} (h : ‖x‖ ≤ c) : ‖f x‖ ≤ ‖f‖ * c :=
  f.le_of_opNorm_le_of_le le_rfl h


@[deprecated (since := "2024-02-02")] alias le_op_norm_of_le := le_opNorm_of_le


theorem le_of_opNorm_le {c : ℝ} (h : ‖f‖ ≤ c) (x : E) : ‖f x‖ ≤ c * ‖x‖ :=
  f.le_of_opNorm_le_of_le h le_rfl


@[deprecated (since := "2024-02-02")] alias le_of_op_norm_le := le_of_opNorm_le


theorem opNorm_le_iff {f : E →SL[σ₁₂] F} {M : ℝ} (hMp : 0 ≤ M) :
    ‖f‖ ≤ M ↔ ∀ x, ‖f x‖ ≤ M * ‖x‖ :=
  ⟨f.le_of_opNorm_le, opNorm_le_bound f hMp⟩


@[deprecated (since := "2024-02-02")] alias op_norm_le_iff := opNorm_le_iff


theorem ratio_le_opNorm : ‖f x‖ / ‖x‖ ≤ ‖f‖ :=
  div_le_of_le_mul₀ (norm_nonneg _) f.opNorm_nonneg (le_opNorm _ _)


@[deprecated (since := "2024-02-02")] alias ratio_le_op_norm := ratio_le_opNorm


/-- The image of the unit ball under a continuous linear map is bounded. -/
theorem unit_le_opNorm : ‖x‖ ≤ 1 → ‖f x‖ ≤ ‖f‖ :=
  mul_one ‖f‖ ▸ f.le_opNorm_of_le


@[deprecated (since := "2024-02-02")] alias unit_le_op_norm := unit_le_opNorm


theorem opNorm_le_of_shell {f : E →SL[σ₁₂] F} {ε C : ℝ} (ε_pos : 0 < ε) (hC : 0 ≤ C) {c : 𝕜}
    (hc : 1 < ‖c‖) (hf : ∀ x, ε / ‖c‖ ≤ ‖x‖ → ‖x‖ < ε → ‖f x‖ ≤ C * ‖x‖) : ‖f‖ ≤ C :=
  f.opNorm_le_bound' hC fun _ hx => SemilinearMapClass.bound_of_shell_semi_normed f ε_pos hc hf hx


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_shell := opNorm_le_of_shell


theorem opNorm_le_of_ball {f : E →SL[σ₁₂] F} {ε : ℝ} {C : ℝ} (ε_pos : 0 < ε) (hC : 0 ≤ C)
    (hf : ∀ x ∈ ball (0 : E) ε, ‖f x‖ ≤ C * ‖x‖) : ‖f‖ ≤ C := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    hC : LE.le 0 C
    hf : ∀ (x : E), Membership.mem (Metric.ball 0 ε) x → LE.le (Norm.norm (f x)) ( …
    ⊢ LE.le (Norm.norm f) C
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    hC : LE.le 0 C
    hf : ∀ (x : E), Membership.mem (Metric.ball 0 ε) x → LE.le (Norm.norm (f x)) ( …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ LE.le (Norm.norm f) C
  -/
  refine opNorm_le_of_shell ε_pos hC hc fun x _ hx => hf x ?_
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    hC : LE.le 0 C
    hf : ∀ (x : E), Membership.mem (Metric.ball 0 ε) x → LE.le (Norm.norm (f x)) ( …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    x✝ : LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm x)
    hx : LT.lt (Norm.norm x) ε
    ⊢ Membership.mem (Metric.ball 0 ε) x
  -/
  rwa [ball_zero_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_ball := opNorm_le_of_ball


theorem opNorm_le_of_nhds_zero {f : E →SL[σ₁₂] F} {C : ℝ} (hC : 0 ≤ C)
    (hf : ∀ᶠ x in 𝓝 (0 : E), ‖f x‖ ≤ C * ‖x‖) : ‖f‖ ≤ C :=
  let ⟨_, ε0, hε⟩ := Metric.eventually_nhds_iff_ball.1 hf
  opNorm_le_of_ball ε0 hC hε


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_nhds_zero := opNorm_le_of_nhds_zero


theorem opNorm_le_of_shell' {f : E →SL[σ₁₂] F} {ε C : ℝ} (ε_pos : 0 < ε) (hC : 0 ≤ C) {c : 𝕜}
    (hc : ‖c‖ < 1) (hf : ∀ x, ε * ‖c‖ ≤ ‖x‖ → ‖x‖ < ε → ‖f x‖ ≤ C * ‖x‖) : ‖f‖ ≤ C := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    hC : LE.le 0 C
    c : 𝕜
    hc : LT.lt (Norm.norm c) 1
    hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
    ⊢ LE.le (Norm.norm f) C
  -/
  by_cases h0 : c = 0
    /-
      case pos
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ε C : Real
      ε_pos : LT.lt 0 ε
      hC : LE.le 0 C
      c : 𝕜
      hc : LT.lt (Norm.norm c) 1
      hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
      h0 : Eq c 0
      ⊢ LE.le (Norm.norm f) C
    -/
  · refine opNorm_le_of_ball ε_pos hC fun x hx => hf x ?_ ?_
      /-
        case pos.refine_1
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_4
        F : Type u_5
        inst✝⁶ : SeminormedAddCommGroup E
        inst✝⁵ : SeminormedAddCommGroup F
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NontriviallyNormedField 𝕜₂
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedSpace 𝕜₂ F
        σ₁₂ : RingHom 𝕜 𝕜₂
        inst✝ : RingHomIsometric σ₁₂
        f : ContinuousLinearMap σ₁₂ E F
        ε C : Real
        ε_pos : LT.lt 0 ε
        hC : LE.le 0 C
        c : 𝕜
        hc : LT.lt (Norm.norm c) 1
        hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
        h0 : Eq c 0
        x : E
        hx : Membership.mem (Metric.ball 0 ε) x
        ⊢ LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x)
      -/
    · simp [h0]
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_4
        F : Type u_5
        inst✝⁶ : SeminormedAddCommGroup E
        inst✝⁵ : SeminormedAddCommGroup F
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NontriviallyNormedField 𝕜₂
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedSpace 𝕜₂ F
        σ₁₂ : RingHom 𝕜 𝕜₂
        inst✝ : RingHomIsometric σ₁₂
        f : ContinuousLinearMap σ₁₂ E F
        ε C : Real
        ε_pos : LT.lt 0 ε
        hC : LE.le 0 C
        c : 𝕜
        hc : LT.lt (Norm.norm c) 1
        hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
        h0 : Eq c 0
        x : E
        hx : Membership.mem (Metric.ball 0 ε) x
        ⊢ LT.lt (Norm.norm x) ε
      -/
    · rwa [ball_zero_eq] at hx
      /-
        🎉 no goals
      -/
    /-
      case neg
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ε C : Real
      ε_pos : LT.lt 0 ε
      hC : LE.le 0 C
      c : 𝕜
      hc : LT.lt (Norm.norm c) 1
      hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
      h0 : Not (Eq c 0)
      ⊢ LE.le (Norm.norm f) C
    -/
  · rw [← inv_inv c, norm_inv, inv_lt_one₀ (norm_pos_iff.2 <| inv_ne_zero h0)] at hc
    /-
      case neg
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ε C : Real
      ε_pos : LT.lt 0 ε
      hC : LE.le 0 C
      c : 𝕜
      hc : LT.lt 1 (Norm.norm (Inv.inv c))
      hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
      h0 : Not (Eq c 0)
      ⊢ LE.le (Norm.norm f) C
    -/
    refine opNorm_le_of_shell ε_pos hC hc ?_
    /-
      case neg
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      f : ContinuousLinearMap σ₁₂ E F
      ε C : Real
      ε_pos : LT.lt 0 ε
      hC : LE.le 0 C
      c : 𝕜
      hc : LT.lt 1 (Norm.norm (Inv.inv c))
      hf : ∀ (x : E), LE.le (HMul.hMul ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
      h0 : Not (Eq c 0)
      ⊢ ∀ (x : E), LE.le (HDiv.hDiv ε (Norm.norm (Inv.inv c))) (Norm.norm x) → LT.lt …
    -/
    rwa [norm_inv, div_eq_mul_inv, inv_inv]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_shell' := opNorm_le_of_shell'


/-- For a continuous real linear map `f`, if one controls the norm of every `f x`, `‖x‖ = 1`, then
one controls the norm of `f`. -/
theorem opNorm_le_of_unit_norm [NormedSpace ℝ E] [NormedSpace ℝ F] {f : E →L[ℝ] F} {C : ℝ}
    (hC : 0 ≤ C) (hf : ∀ x, ‖x‖ = 1 → ‖f x‖ ≤ C) : ‖f‖ ≤ C := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hC : LE.le 0 C
    hf : ∀ (x : E), Eq (Norm.norm x) 1 → LE.le (Norm.norm (f x)) C
    ⊢ LE.le (Norm.norm f) C
  -/
  refine opNorm_le_bound' f hC fun x hx => ?_
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hC : LE.le 0 C
    hf : ∀ (x : E), Eq (Norm.norm x) 1 → LE.le (Norm.norm (f x)) C
    x : E
    hx : Ne (Norm.norm x) 0
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
  -/
  have H₁ : ‖‖x‖⁻¹ • x‖ = 1 := by rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hx]
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hC : LE.le 0 C
    hf : ∀ (x : E), Eq (Norm.norm x) 1 → LE.le (Norm.norm (f x)) C
    x : E
    hx : Ne (Norm.norm x) 0
    H₁ : Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm x)) x)) 1
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
  -/
  have H₂ := hf _ H₁
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hC : LE.le 0 C
    hf : ∀ (x : E), Eq (Norm.norm x) 1 → LE.le (Norm.norm (f x)) C
    x : E
    hx : Ne (Norm.norm x) 0
    H₁ : Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm x)) x)) 1
    H₂ : LE.le (Norm.norm (f (HSMul.hSMul (Inv.inv (Norm.norm x)) x))) C
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
  -/
  rwa [map_smul, norm_smul, norm_inv, norm_norm, ← div_eq_inv_mul, div_le_iff₀] at H₂
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hC : LE.le 0 C
    hf : ∀ (x : E), Eq (Norm.norm x) 1 → LE.le (Norm.norm (f x)) C
    x : E
    hx : Ne (Norm.norm x) 0
    H₁ : Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm x)) x)) 1
    H₂ : LE.le (HDiv.hDiv (Norm.norm (f x)) (Norm.norm x)) C
    ⊢ LT.lt 0 (Norm.norm x)
  -/
  exact (norm_nonneg x).lt_of_ne' hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_le_of_unit_norm := opNorm_le_of_unit_norm


/-- The operator norm satisfies the triangle inequality. -/
theorem opNorm_add_le : ‖f + g‖ ≤ ‖f‖ + ‖g‖ :=
  (f + g).opNorm_le_bound (add_nonneg f.opNorm_nonneg g.opNorm_nonneg) fun x =>
    (norm_add_le_of_le (f.le_opNorm x) (g.le_opNorm x)).trans_eq (add_mul _ _ _).symm


@[deprecated (since := "2024-02-02")] alias op_norm_add_le := opNorm_add_le


/-- If there is an element with norm different from `0`, then the norm of the identity equals `1`.
(Since we are working with seminorms supposing that the space is non-trivial is not enough.) -/
theorem norm_id_of_nontrivial_seminorm (h : ∃ x : E, ‖x‖ ≠ 0) : ‖id 𝕜 E‖ = 1 :=
  le_antisymm norm_id_le <| by
    /-
      𝕜 : Type u_1
      E : Type u_4
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      h : Exists fun x => Ne (Norm.norm x) 0
      ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.id 𝕜 E))
    -/
    let ⟨x, hx⟩ := h
    /-
      𝕜 : Type u_1
      E : Type u_4
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      h : Exists fun x => Ne (Norm.norm x) 0
      x : E
      hx : Ne (Norm.norm x) 0
      ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.id 𝕜 E))
    -/
    have := (id 𝕜 E).ratio_le_opNorm x
    /-
      𝕜 : Type u_1
      E : Type u_4
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      h : Exists fun x => Ne (Norm.norm x) 0
      x : E
      hx : Ne (Norm.norm x) 0
      this : LE.le (HDiv.hDiv (Norm.norm ((ContinuousLinearMap.id 𝕜 E) x)) (Norm.nor …
      ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.id 𝕜 E))
    -/
    rwa [id_apply, div_self hx] at this
    /-
      🎉 no goals
    -/


theorem opNorm_smul_le {𝕜' : Type*} [NormedField 𝕜'] [NormedSpace 𝕜' F] [SMulCommClass 𝕜₂ 𝕜' F]
    (c : 𝕜') (f : E →SL[σ₁₂] F) : ‖c • f‖ ≤ ‖c‖ * ‖f‖ :=
  (c • f).opNorm_le_bound (mul_nonneg (norm_nonneg _) (opNorm_nonneg _)) fun _ => by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : SeminormedAddCommGroup E
      inst✝⁸ : SeminormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝³ : RingHomIsometric σ₁₂
      𝕜' : Type u_9
      inst✝² : NormedField 𝕜'
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : SMulCommClass 𝕜₂ 𝕜' F
      c : 𝕜'
      f : ContinuousLinearMap σ₁₂ E F
      x✝ : E
      ⊢ LE.le (Norm.norm ((HSMul.hSMul c f) x✝)) (HMul.hMul (HMul.hMul (Norm.norm c) …
    -/
    rw [smul_apply, norm_smul, mul_assoc]
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : SeminormedAddCommGroup E
      inst✝⁸ : SeminormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝³ : RingHomIsometric σ₁₂
      𝕜' : Type u_9
      inst✝² : NormedField 𝕜'
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : SMulCommClass 𝕜₂ 𝕜' F
      c : 𝕜'
      f : ContinuousLinearMap σ₁₂ E F
      x✝ : E
      ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm (f x✝))) (HMul.hMul (Norm.norm c)  …
    -/
    exact mul_le_mul_of_nonneg_left (le_opNorm _ _) (norm_nonneg _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_norm_smul_le := opNorm_smul_le


/-- Operator seminorm on the space of continuous (semi)linear maps, as `Seminorm`.

We use this seminorm to define a `SeminormedGroup` structure on `E →SL[σ] F`,
but we have to override the projection `UniformSpace`
so that it is definitionally equal to the one coming from the topologies on `E` and `F`. -/
protected def seminorm : Seminorm 𝕜₂ (E →SL[σ₁₂] F) :=
  .ofSMulLE norm opNorm_zero opNorm_add_le opNorm_smul_le


private lemma uniformity_eq_seminorm :
    𝓤 (E →SL[σ₁₂] F) = ⨅ r > 0, 𝓟 {f | ‖f.1 - f.2‖ < r} := by
  refine ContinuousLinearMap.seminorm (σ₁₂ := σ₁₂) (E := E) (F := F) |>.uniformity_eq_of_hasBasis
    (ContinuousLinearMap.hasBasis_nhds_zero_of_basis Metric.nhds_basis_closedBall)
    ?_ fun (s, r) ⟨hs, hr⟩ ↦ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      ⊢ Exists fun r => Membership.mem (nhds 0) (ContinuousLinearMap.seminorm.closed …
    -/
  · rcases NormedField.exists_lt_norm 𝕜 1 with ⟨c, hc⟩
    refine ⟨‖c‖, ContinuousLinearMap.hasBasis_nhds_zero.mem_iff.2
      ⟨(closedBall 0 1, closedBall 0 1), ?_⟩⟩
    suffices ∀ f : E →SL[σ₁₂] F, (∀ x, ‖x‖ ≤ 1 → ‖f x‖ ≤ 1) → ‖f‖ ≤ ‖c‖ by
      simpa [NormedSpace.isVonNBounded_closedBall, closedBall_mem_nhds, subset_def] using this
    /-
      case refine_1.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ⊢ ∀ (f : ContinuousLinearMap σ₁₂ E F), (∀ (x : E), LE.le (Norm.norm x) 1 → LE. …
    -/
    intro f hf
    /-
      case refine_1.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      f : ContinuousLinearMap σ₁₂ E F
      hf : ∀ (x : E), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (f x)) 1
      ⊢ LE.le (Norm.norm f) (Norm.norm c)
    -/
    refine opNorm_le_of_shell (f := f) one_pos (norm_nonneg c) hc fun x hcx hx ↦ ?_
    /-
      case refine_1.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      f : ContinuousLinearMap σ₁₂ E F
      hf : ∀ (x : E), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (f x)) 1
      x : E
      hcx : LE.le (HDiv.hDiv 1 (Norm.norm c)) (Norm.norm x)
      hx : LT.lt (Norm.norm x) 1
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm c) (Norm.norm x))
    -/
    exact (hf x hx.le).trans ((div_le_iff₀' <| one_pos.trans hc).1 hcx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset (ContinuousLinearMap.s …
    -/
  · rcases (NormedSpace.isVonNBounded_iff' _).1 hs with ⟨ε, hε⟩
    /-
      case refine_2.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : E), Membership.mem { fst := s, snd := r }.1 x → LE.le (Norm.norm x …
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset (ContinuousLinearMap.s …
    -/
    rcases exists_pos_mul_lt hr ε with ⟨δ, hδ₀, hδ⟩
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : E), Membership.mem { fst := s, snd := r }.1 x → LE.le (Norm.norm x …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul ε δ) { fst := s, snd := r }.2
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset (ContinuousLinearMap.s …
    -/
    refine ⟨δ, hδ₀, fun f hf x hx ↦ ?_⟩
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : E), Membership.mem { fst := s, snd := r }.1 x → LE.le (Norm.norm x …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul ε δ) { fst := s, snd := r }.2
      f : ContinuousLinearMap σ₁₂ E F
      hf : Membership.mem (ContinuousLinearMap.seminorm.ball 0 δ) f
      x : E
      hx : Membership.mem { fst := s, snd := r }.1 x
      ⊢ Membership.mem (Metric.closedBall 0 { fst := s, snd := r }.2) (f x)
    -/
    simp only [Seminorm.mem_ball_zero, mem_closedBall_zero_iff] at hf ⊢
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : E), Membership.mem { fst := s, snd := r }.1 x → LE.le (Norm.norm x …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul ε δ) { fst := s, snd := r }.2
      f : ContinuousLinearMap σ₁₂ E F
      x : E
      hx : Membership.mem { fst := s, snd := r }.1 x
      hf : LT.lt (ContinuousLinearMap.seminorm f) δ
      ⊢ LE.le (Norm.norm (f x)) r
    -/
    rw [mul_comm] at hδ
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁶ : SeminormedAddCommGroup E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NontriviallyNormedField 𝕜₂
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝ : RingHomIsometric σ₁₂
      x✝¹ : Prod (Set E) Real
      s : Set E
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : E), Membership.mem { fst := s, snd := r }.1 x → LE.le (Norm.norm x …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul δ ε) { fst := s, snd := r }.2
      f : ContinuousLinearMap σ₁₂ E F
      x : E
      hx : Membership.mem { fst := s, snd := r }.1 x
      hf : LT.lt (ContinuousLinearMap.seminorm f) δ
      ⊢ LE.le (Norm.norm (f x)) r
    -/
    exact le_trans (le_of_opNorm_le_of_le _ hf.le (hε _ hx)) hδ.le
    /-
      🎉 no goals
    -/


instance toPseudoMetricSpace : PseudoMetricSpace (E →SL[σ₁₂] F) := .replaceUniformity
  ContinuousLinearMap.seminorm.toSeminormedAddCommGroup.toPseudoMetricSpace uniformity_eq_seminorm


/-- Continuous linear maps themselves form a seminormed space with respect to
    the operator norm. -/
instance toSeminormedAddCommGroup : SeminormedAddCommGroup (E →SL[σ₁₂] F) where
  dist_eq _ _ := rfl


instance toNormedSpace {𝕜' : Type*} [NormedField 𝕜'] [NormedSpace 𝕜' F] [SMulCommClass 𝕜₂ 𝕜' F] :
    NormedSpace 𝕜' (E →SL[σ₁₂] F) :=
  ⟨opNorm_smul_le⟩


/-- The operator norm is submultiplicative. -/
theorem opNorm_comp_le (f : E →SL[σ₁₂] F) : ‖h.comp f‖ ≤ ‖h‖ * ‖f‖ :=
  csInf_le bounds_bddBelow
    ⟨mul_nonneg (opNorm_nonneg _) (opNorm_nonneg _), fun x => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_7
        inst✝¹¹ : SeminormedAddCommGroup E
        inst✝¹⁰ : SeminormedAddCommGroup F
        inst✝⁹ : SeminormedAddCommGroup G
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NontriviallyNormedField 𝕜₂
        inst✝⁶ : NontriviallyNormedField 𝕜₃
        inst✝⁵ : NormedSpace 𝕜 E
        inst✝⁴ : NormedSpace 𝕜₂ F
        inst✝³ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹ : RingHomIsometric σ₁₂
        inst✝ : RingHomIsometric σ₂₃
        h : ContinuousLinearMap σ₂₃ F G
        f : ContinuousLinearMap σ₁₂ E F
        x : E
        ⊢ LE.le (Norm.norm ((h.comp f) x)) (HMul.hMul (HMul.hMul (Norm.norm h) (Norm.n …
      -/
      rw [mul_assoc]
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_7
        inst✝¹¹ : SeminormedAddCommGroup E
        inst✝¹⁰ : SeminormedAddCommGroup F
        inst✝⁹ : SeminormedAddCommGroup G
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NontriviallyNormedField 𝕜₂
        inst✝⁶ : NontriviallyNormedField 𝕜₃
        inst✝⁵ : NormedSpace 𝕜 E
        inst✝⁴ : NormedSpace 𝕜₂ F
        inst✝³ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹ : RingHomIsometric σ₁₂
        inst✝ : RingHomIsometric σ₂₃
        h : ContinuousLinearMap σ₂₃ F G
        f : ContinuousLinearMap σ₁₂ E F
        x : E
        ⊢ LE.le (Norm.norm ((h.comp f) x)) (HMul.hMul (Norm.norm h) (HMul.hMul (Norm.n …
      -/
      exact h.le_opNorm_of_le (f.le_opNorm x)⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-02-02")] alias op_norm_comp_le := opNorm_comp_le


/-- Continuous linear maps form a seminormed ring with respect to the operator norm. -/
instance toSemiNormedRing : SeminormedRing (E →L[𝕜] E) :=
  { ContinuousLinearMap.toSeminormedAddCommGroup, ContinuousLinearMap.ring with
    norm_mul := fun f g => opNorm_comp_le f g }


/-- For a normed space `E`, continuous linear endomorphisms form a normed algebra with
respect to the operator norm. -/
instance toNormedAlgebra : NormedAlgebra 𝕜 (E →L[𝕜] E) :=
  { algebra with
    norm_smul_le := by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_5
        Fₗ : Type u_6
        G : Type u_7
        𝓕 : Type u_8
        inst✝¹⁴ : SeminormedAddCommGroup E
        inst✝¹³ : SeminormedAddCommGroup F
        inst✝¹² : SeminormedAddCommGroup Fₗ
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NontriviallyNormedField 𝕜₂
        inst✝⁸ : NontriviallyNormedField 𝕜₃
        inst✝⁷ : NormedSpace 𝕜 E
        inst✝⁶ : NormedSpace 𝕜₂ F
        inst✝⁵ : NormedSpace 𝕜 Fₗ
        inst✝⁴ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝² : FunLike 𝓕 E F
        inst✝¹ : RingHomIsometric σ₁₂
        inst✝ : RingHomIsometric σ₂₃
        f g : ContinuousLinearMap σ₁₂ E F
        h : ContinuousLinearMap σ₂₃ F G
        x : E
        ⊢ ∀ (r : 𝕜) (x : ContinuousLinearMap (RingHom.id 𝕜) E E), LE.le (Norm.norm (HS …
      -/
      intro c f
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_5
        Fₗ : Type u_6
        G : Type u_7
        𝓕 : Type u_8
        inst✝¹⁴ : SeminormedAddCommGroup E
        inst✝¹³ : SeminormedAddCommGroup F
        inst✝¹² : SeminormedAddCommGroup Fₗ
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NontriviallyNormedField 𝕜₂
        inst✝⁸ : NontriviallyNormedField 𝕜₃
        inst✝⁷ : NormedSpace 𝕜 E
        inst✝⁶ : NormedSpace 𝕜₂ F
        inst✝⁵ : NormedSpace 𝕜 Fₗ
        inst✝⁴ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝² : FunLike 𝓕 E F
        inst✝¹ : RingHomIsometric σ₁₂
        inst✝ : RingHomIsometric σ₂₃
        f✝ g : ContinuousLinearMap σ₁₂ E F
        h : ContinuousLinearMap σ₂₃ F G
        x : E
        c : 𝕜
        f : ContinuousLinearMap (RingHom.id 𝕜) E E
        ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
      -/
      apply opNorm_smul_le c f}
      /-
        🎉 no goals
      -/


@[simp, nontriviality]
theorem opNorm_subsingleton [Subsingleton E] : ‖f‖ = 0 := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    inst✝ : Subsingleton E
    ⊢ Eq (Norm.norm f) 0
  -/
  refine le_antisymm ?_ (norm_nonneg _)
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    inst✝ : Subsingleton E
    ⊢ LE.le (Norm.norm f) 0
  -/
  apply opNorm_le_bound _ rfl.ge
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    inst✝ : Subsingleton E
    ⊢ ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul 0 (Norm.norm x))
  -/
  intro x
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    f : ContinuousLinearMap σ₁₂ E F
    inst✝ : Subsingleton E
    x : E
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul 0 (Norm.norm x))
  -/
  simp [Subsingleton.elim x 0]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_subsingleton := opNorm_subsingleton


@[simp]
theorem norm_restrictScalars (f : E →L[𝕜] Fₗ) : ‖f.restrictScalars 𝕜'‖ = ‖f‖ :=
  le_antisymm (opNorm_le_bound _ (norm_nonneg _) fun x => f.le_opNorm x)
    (opNorm_le_bound _ (norm_nonneg _) fun x => f.le_opNorm x)


/-- `ContinuousLinearMap.restrictScalars` as a `LinearIsometry`. -/
def restrictScalarsIsometry : (E →L[𝕜] Fₗ) →ₗᵢ[𝕜''] E →L[𝕜'] Fₗ :=
  ⟨restrictScalarsₗ 𝕜 E Fₗ 𝕜' 𝕜'', norm_restrictScalars⟩


@[simp]
theorem coe_restrictScalarsIsometry :
    ⇑(restrictScalarsIsometry 𝕜 E Fₗ 𝕜' 𝕜'') = restrictScalars 𝕜' :=
  rfl


@[simp]
theorem restrictScalarsIsometry_toLinearMap :
    (restrictScalarsIsometry 𝕜 E Fₗ 𝕜' 𝕜'').toLinearMap = restrictScalarsₗ 𝕜 E Fₗ 𝕜' 𝕜'' :=
  rfl


lemma norm_pi_le_of_le {ι : Type*} [Fintype ι]
    {M : ι → Type*} [∀ i, SeminormedAddCommGroup (M i)] [∀ i, NormedSpace 𝕜 (M i)] {C : ℝ}
    {L : (i : ι) → (E →L[𝕜] M i)} (hL : ∀ i, ‖L i‖ ≤ C) (hC : 0 ≤ C) :
    ‖pi L‖ ≤ C := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    M : ι → Type u_10
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (M i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
    C : Real
    L : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) E (M i)
    hL : ∀ (i : ι), LE.le (Norm.norm (L i)) C
    hC : LE.le 0 C
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.pi L)) C
  -/
  refine opNorm_le_bound _ hC (fun x ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    M : ι → Type u_10
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (M i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
    C : Real
    L : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) E (M i)
    hL : ∀ (i : ι), LE.le (Norm.norm (L i)) C
    hC : LE.le 0 C
    x : E
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.pi L) x)) (HMul.hMul C (Norm.norm x))
  -/
  refine (pi_norm_le_iff_of_nonneg (by positivity)).mpr (fun i ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    M : ι → Type u_10
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (M i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
    C : Real
    L : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) E (M i)
    hL : ∀ (i : ι), LE.le (Norm.norm (L i)) C
    hC : LE.le 0 C
    x : E
    i : ι
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.pi L) x i)) (HMul.hMul C (Norm.norm x))
  -/
  exact (L i).le_of_opNorm_le (hL i) _
  /-
    🎉 no goals
  -/


/-- If a continuous linear map is constructed from a linear map via the constructor `mkContinuous`,
then its norm is bounded by the bound given to the constructor if it is nonnegative. -/
theorem mkContinuous_norm_le (f : E →ₛₗ[σ₁₂] F) {C : ℝ} (hC : 0 ≤ C) (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    ‖f.mkContinuous C h‖ ≤ C :=
  ContinuousLinearMap.opNorm_le_bound _ hC h


/-- If a continuous linear map is constructed from a linear map via the constructor `mkContinuous`,
then its norm is bounded by the bound or zero if bound is negative. -/
theorem mkContinuous_norm_le' (f : E →ₛₗ[σ₁₂] F) {C : ℝ} (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    ‖f.mkContinuous C h‖ ≤ max C 0 :=
  ContinuousLinearMap.opNorm_le_bound _ (le_max_right _ _) fun x =>
    (h x).trans <| mul_le_mul_of_nonneg_right (le_max_left _ _) (norm_nonneg x)


theorem norm_toContinuousLinearMap_le (f : E →ₛₗᵢ[σ₁₂] F) : ‖f.toContinuousLinearMap‖ ≤ 1 :=
                                                                  /-
                                                                    𝕜 : Type u_1
                                                                    𝕜₂ : Type u_2
                                                                    E : Type u_4
                                                                    F : Type u_5
                                                                    inst✝⁵ : SeminormedAddCommGroup E
                                                                    inst✝⁴ : SeminormedAddCommGroup F
                                                                    inst✝³ : NontriviallyNormedField 𝕜
                                                                    inst✝² : NontriviallyNormedField 𝕜₂
                                                                    inst✝¹ : NormedSpace 𝕜 E
                                                                    inst✝ : NormedSpace 𝕜₂ F
                                                                    σ₁₂ : RingHom 𝕜 𝕜₂
                                                                    f : LinearIsometry σ₁₂ E F
                                                                    x : E
                                                                    ⊢ LE.le (Norm.norm (f.toContinuousLinearMap x)) (HMul.hMul 1 (Norm.norm x))
                                                                  -/
  f.toContinuousLinearMap.opNorm_le_bound zero_le_one fun x => by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem norm_subtypeL_le (K : Submodule 𝕜 E) : ‖K.subtypeL‖ ≤ 1 :=
  K.subtypeₗᵢ.norm_toContinuousLinearMap_le


