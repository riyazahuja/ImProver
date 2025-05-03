theorem bound_of_shell [RingHomIsometric σ₁₂] (f : E →ₛₗ[σ₁₂] F) {ε C : ℝ} (ε_pos : 0 < ε) {c : 𝕜}
    (hc : 1 < ‖c‖) (hf : ∀ x, ε / ‖c‖ ≤ ‖x‖ → ‖x‖ < ε → ‖f x‖ ≤ C * ‖x‖) (x : E) :
    ‖f x‖ ≤ C * ‖x‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
    x : E
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NontriviallyNormedField 𝕜₂
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    ε C : Real
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm x) → LT.lt (Norm. …
    x : E
    hx : Not (Eq x 0)
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
  -/
  exact SemilinearMapClass.bound_of_shell_semi_normed f ε_pos hc hf (norm_ne_zero_iff.2 hx)
  /-
    🎉 no goals
  -/


/-- `LinearMap.bound_of_ball_bound'` is a version of this lemma over a field satisfying `RCLike`
that produces a concrete bound.
-/
theorem bound_of_ball_bound {r : ℝ} (r_pos : 0 < r) (c : ℝ) (f : E →ₗ[𝕜] Fₗ)
    (h : ∀ z ∈ Metric.ball (0 : E) r, ‖f z‖ ≤ c) : ∃ C, ∀ z : E, ‖f z‖ ≤ C * ‖z‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_6
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup Fₗ
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 Fₗ
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E Fₗ
    h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
    ⊢ Exists fun C => ∀ (z : E), LE.le (Norm.norm (f z)) (HMul.hMul C (Norm.norm z))
  -/
  cases' @NontriviallyNormedField.non_trivial 𝕜 _ with k hk
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_6
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup Fₗ
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 Fₗ
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E Fₗ
    h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
    k : 𝕜
    hk : LT.lt 1 (Norm.norm k)
    ⊢ Exists fun C => ∀ (z : E), LE.le (Norm.norm (f z)) (HMul.hMul C (Norm.norm z))
  -/
  use c * (‖k‖ / r)
  /-
    case h
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_6
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup Fₗ
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 Fₗ
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E Fₗ
    h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
    k : 𝕜
    hk : LT.lt 1 (Norm.norm k)
    ⊢ ∀ (z : E), LE.le (Norm.norm (f z)) (HMul.hMul (HMul.hMul c (HDiv.hDiv (Norm. …
  -/
  intro z
  /-
    case h
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_6
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup Fₗ
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 Fₗ
    r : Real
    r_pos : LT.lt 0 r
    c : Real
    f : LinearMap (RingHom.id 𝕜) E Fₗ
    h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
    k : 𝕜
    hk : LT.lt 1 (Norm.norm k)
    z : E
    ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HMul.hMul c (HDiv.hDiv (Norm.norm k) r)) …
  -/
  refine bound_of_shell _ r_pos hk (fun x hko hxo => ?_) _
  calc
    ‖f x‖ ≤ c := h _ (mem_ball_zero_iff.mpr hxo)
    _ ≤ c * (‖x‖ * ‖k‖ / r) := le_mul_of_one_le_right ?_ ?_
    _ = _ := by ring
    /-
      case h.calc_1
      𝕜 : Type u_1
      E : Type u_4
      Fₗ : Type u_6
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup Fₗ
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 Fₗ
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E Fₗ
      h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
      k : 𝕜
      hk : LT.lt 1 (Norm.norm k)
      z x : E
      hko : LE.le (HDiv.hDiv r (Norm.norm k)) (Norm.norm x)
      hxo : LT.lt (Norm.norm x) r
      ⊢ LE.le 0 c
    -/
  · exact le_trans (norm_nonneg _) (h 0 (by simp [r_pos]))
    /-
      🎉 no goals
    -/
    /-
      case h.calc_2
      𝕜 : Type u_1
      E : Type u_4
      Fₗ : Type u_6
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup Fₗ
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 Fₗ
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E Fₗ
      h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
      k : 𝕜
      hk : LT.lt 1 (Norm.norm k)
      z x : E
      hko : LE.le (HDiv.hDiv r (Norm.norm k)) (Norm.norm x)
      hxo : LT.lt (Norm.norm x) r
      ⊢ LE.le 1 (HDiv.hDiv (HMul.hMul (Norm.norm x) (Norm.norm k)) r)
    -/
  · rw [div_le_iff₀ (zero_lt_one.trans hk)] at hko
    /-
      case h.calc_2
      𝕜 : Type u_1
      E : Type u_4
      Fₗ : Type u_6
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup Fₗ
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 Fₗ
      r : Real
      r_pos : LT.lt 0 r
      c : Real
      f : LinearMap (RingHom.id 𝕜) E Fₗ
      h : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (f z)) c
      k : 𝕜
      hk : LT.lt 1 (Norm.norm k)
      z x : E
      hko : LE.le r (HMul.hMul (Norm.norm x) (Norm.norm k))
      hxo : LT.lt (Norm.norm x) r
      ⊢ LE.le 1 (HDiv.hDiv (HMul.hMul (Norm.norm x) (Norm.norm k)) r)
    -/
    exact (one_le_div r_pos).mpr hko
    /-
      🎉 no goals
    -/


theorem antilipschitz_of_comap_nhds_le [h : RingHomIsometric σ₁₂] (f : E →ₛₗ[σ₁₂] F)
    (hf : (𝓝 0).comap f ≤ 𝓝 0) : ∃ K, AntilipschitzWith K f := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ⊢ Exists fun K => AntilipschitzWith K ⇑f
  -/
  rcases ((nhds_basis_ball.comap _).le_basis_iff nhds_basis_ball).1 hf 1 one_pos with ⟨ε, ε0, hε⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : Real
    ε0 : LT.lt 0 ε
    hε : HasSubset.Subset (Set.preimage (⇑f) (Metric.ball 0 ε)) (Metric.ball 0 1)
    ⊢ Exists fun K => AntilipschitzWith K ⇑f
  -/
  simp only [Set.subset_def, Set.mem_preimage, mem_ball_zero_iff] at hε
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ε → LT.lt (Norm.norm x) 1
    ⊢ Exists fun K => AntilipschitzWith K ⇑f
  -/
  lift ε to ℝ≥0 using ε0.le
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    ⊢ Exists fun K => AntilipschitzWith K ⇑f
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ Exists fun K => AntilipschitzWith K ⇑f
  -/
  refine ⟨ε⁻¹ * ‖c‖₊, AddMonoidHomClass.antilipschitz_of_bound f fun x => ?_⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
  -/
  by_cases hx : f x = 0
    /-
      case pos
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NontriviallyNormedField 𝕜₂
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      h : RingHomIsometric σ₁₂
      f : LinearMap σ₁₂ E F
      hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
      ε : NNReal
      ε0 : LT.lt 0 ↑ε
      hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      x : E
      hx : Eq (f x) 0
      ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
    -/
  · rw [← hx] at hf
    obtain rfl : x = 0 := Specializes.eq (specializes_iff_pure.2 <|
      ((Filter.tendsto_pure_pure _ _).mono_right (pure_le_nhds _)).le_comap.trans hf)
    /-
      case pos
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NontriviallyNormedField 𝕜₂
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      h : RingHomIsometric σ₁₂
      f : LinearMap σ₁₂ E F
      ε : NNReal
      ε0 : LT.lt 0 ↑ε
      hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      hf : LE.le (Filter.comap (⇑f) (nhds (f 0))) (nhds 0)
      hx : Eq (f 0) 0
      ⊢ LE.le (Norm.norm 0) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
    -/
    exact norm_zero.trans_le (mul_nonneg (NNReal.coe_nonneg _) (norm_nonneg _))
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Not (Eq (f x) 0)
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
  -/
  have hc₀ : c ≠ 0 := norm_pos_iff.1 (one_pos.trans hc)
  /-
    case neg
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Not (Eq (f x) 0)
    hc₀ : Ne c 0
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
  -/
  rw [← h.1] at hc
  /-
    case neg
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm (σ₁₂ c))
    x : E
    hx : Not (Eq (f x) 0)
    hc₀ : Ne c 0
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
  -/
  rcases rescale_to_shell_zpow hc ε0 hx with ⟨n, -, hlt, -, hle⟩
  /-
    case neg.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NontriviallyNormedField 𝕜₂
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    h : RingHomIsometric σ₁₂
    f : LinearMap σ₁₂ E F
    hf : LE.le (Filter.comap (⇑f) (nhds 0)) (nhds 0)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hε : ∀ (x : E), LT.lt (Norm.norm (f x)) ↑ε → LT.lt (Norm.norm x) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm (σ₁₂ c))
    x : E
    hx : Not (Eq (f x) 0)
    hc₀ : Ne c 0
    n : Int
    hlt : LT.lt (Norm.norm (HSMul.hSMul (HPow.hPow (σ₁₂ c) n) (f x))) ↑ε
    hle : LE.le (Inv.inv (Norm.norm (HPow.hPow (σ₁₂ c) n))) (HMul.hMul (HMul.hMul  …
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(HMul.hMul (Inv.inv ε) (NNNorm.nnnorm c)))  …
  -/
  simp only [← map_zpow₀, h.1, ← map_smulₛₗ] at hlt hle
  calc
    ‖x‖ = ‖c ^ n‖⁻¹ * ‖c ^ n • x‖ := by
      rwa [← norm_inv, ← norm_smul, inv_smul_smul₀ (zpow_ne_zero _ _)]
    _ ≤ ‖c ^ n‖⁻¹ * 1 := (mul_le_mul_of_nonneg_left (hε _ hlt).le (inv_nonneg.2 (norm_nonneg _)))
    _ ≤ ε⁻¹ * ‖c‖ * ‖f x‖ := by rwa [mul_one]


/-- An operator is zero iff its norm vanishes. -/
theorem opNorm_zero_iff [RingHomIsometric σ₁₂] : ‖f‖ = 0 ↔ f = 0 :=
  Iff.intro
    (fun hn => ContinuousLinearMap.ext fun x => norm_le_zero_iff.1
      (calc
        _ ≤ ‖f‖ * ‖x‖ := le_opNorm _ _
                    /-
                      𝕜 : Type u_1
                      𝕜₂ : Type u_2
                      E : Type u_4
                      F : Type u_5
                      inst✝⁶ : NormedAddCommGroup E
                      inst✝⁵ : NormedAddCommGroup F
                      inst✝⁴ : NontriviallyNormedField 𝕜
                      inst✝³ : NontriviallyNormedField 𝕜₂
                      inst✝² : NormedSpace 𝕜 E
                      inst✝¹ : NormedSpace 𝕜₂ F
                      σ₁₂ : RingHom 𝕜 𝕜₂
                      f : ContinuousLinearMap σ₁₂ E F
                      inst✝ : RingHomIsometric σ₁₂
                      hn : Eq (Norm.norm f) 0
                      x : E
                      ⊢ Eq (HMul.hMul (Norm.norm f) (Norm.norm x)) 0
                    -/
        _ = _ := by rw [hn, zero_mul]))
                    /-
                      🎉 no goals
                    -/
    (by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NontriviallyNormedField 𝕜₂
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedSpace 𝕜₂ F
        σ₁₂ : RingHom 𝕜 𝕜₂
        f : ContinuousLinearMap σ₁₂ E F
        inst✝ : RingHomIsometric σ₁₂
        ⊢ Eq f 0 → Eq (Norm.norm f) 0
      -/
      rintro rfl
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NontriviallyNormedField 𝕜₂
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedSpace 𝕜₂ F
        σ₁₂ : RingHom 𝕜 𝕜₂
        inst✝ : RingHomIsometric σ₁₂
        ⊢ Eq (Norm.norm 0) 0
      -/
      exact opNorm_zero)
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-02-02")] alias op_norm_zero_iff := opNorm_zero_iff


/-- If a normed space is non-trivial, then the norm of the identity equals `1`. -/
@[simp]
theorem norm_id [Nontrivial E] : ‖id 𝕜 E‖ = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    ⊢ Eq (Norm.norm (ContinuousLinearMap.id 𝕜 E)) 1
  -/
  refine norm_id_of_nontrivial_seminorm ?_
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    ⊢ Exists fun x => Ne (Norm.norm x) 0
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : E)
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    x : E
    hx : Ne x 0
    ⊢ Exists fun x => Ne (Norm.norm x) 0
  -/
  exact ⟨x, ne_of_gt (norm_pos_iff.2 hx)⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma nnnorm_id [Nontrivial E] : ‖id 𝕜 E‖₊ = 1 := NNReal.eq norm_id


instance normOneClass [Nontrivial E] : NormOneClass (E →L[𝕜] E) :=
  ⟨norm_id⟩


/-- Continuous linear maps themselves form a normed space with respect to
    the operator norm. -/
instance toNormedAddCommGroup [RingHomIsometric σ₁₂] : NormedAddCommGroup (E →SL[σ₁₂] F) :=
  NormedAddCommGroup.ofSeparation fun f => (opNorm_zero_iff f).mp


/-- Continuous linear maps form a normed ring with respect to the operator norm. -/
instance toNormedRing : NormedRing (E →L[𝕜] E) :=
  { ContinuousLinearMap.toNormedAddCommGroup, ContinuousLinearMap.toSemiNormedRing with }


theorem homothety_norm [RingHomIsometric σ₁₂] [Nontrivial E] (f : E →SL[σ₁₂] F) {a : ℝ}
    (hf : ∀ x, ‖f x‖ = a * ‖x‖) : ‖f‖ = a := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    f : ContinuousLinearMap σ₁₂ E F
    a : Real
    hf : ∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))
    ⊢ Eq (Norm.norm f) a
  -/
  obtain ⟨x, hx⟩ : ∃ x : E, x ≠ 0 := exists_ne 0
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    f : ContinuousLinearMap σ₁₂ E F
    a : Real
    hf : ∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))
    x : E
    hx : Ne x 0
    ⊢ Eq (Norm.norm f) a
  -/
  rw [← norm_pos_iff] at hx
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    f : ContinuousLinearMap σ₁₂ E F
    a : Real
    hf : ∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))
    x : E
    hx : LT.lt 0 (Norm.norm x)
    ⊢ Eq (Norm.norm f) a
  -/
  have ha : 0 ≤ a := by simpa only [hf, hx, mul_nonneg_iff_of_pos_right] using norm_nonneg (f x)
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    f : ContinuousLinearMap σ₁₂ E F
    a : Real
    hf : ∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))
    x : E
    hx : LT.lt 0 (Norm.norm x)
    ha : LE.le 0 a
    ⊢ Eq (Norm.norm f) a
  -/
  apply le_antisymm (f.opNorm_le_bound ha fun y => le_of_eq (hf y))
  /-
    case intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    f : ContinuousLinearMap σ₁₂ E F
    a : Real
    hf : ∀ (x : E), Eq (Norm.norm (f x)) (HMul.hMul a (Norm.norm x))
    x : E
    hx : LT.lt 0 (Norm.norm x)
    ha : LE.le 0 a
    ⊢ LE.le a (Norm.norm f)
  -/
  simpa only [hf, hx, mul_le_mul_right] using f.le_opNorm x
  /-
    🎉 no goals
  -/


/-- If a continuous linear map is a topology embedding, then it is expands the distances
by a positive factor. -/
theorem antilipschitz_of_isEmbedding (f : E →L[𝕜] Fₗ) (hf : IsEmbedding f) :
    ∃ K, AntilipschitzWith K f :=
  f.toLinearMap.antilipschitz_of_comap_nhds_le <| map_zero f ▸ (hf.nhds_eq_comap 0).ge


@[deprecated (since := "2024-10-26")]
alias antilipschitz_of_embedding := antilipschitz_of_isEmbedding


@[simp]
theorem norm_toContinuousLinearMap [Nontrivial E] [RingHomIsometric σ₁₂] (f : E →ₛₗᵢ[σ₁₂] F) :
    ‖f.toContinuousLinearMap‖ = 1 :=
                                               /-
                                                 𝕜 : Type u_1
                                                 𝕜₂ : Type u_2
                                                 E : Type u_4
                                                 F : Type u_5
                                                 inst✝⁷ : NormedAddCommGroup E
                                                 inst✝⁶ : NormedAddCommGroup F
                                                 inst✝⁵ : NontriviallyNormedField 𝕜
                                                 inst✝⁴ : NontriviallyNormedField 𝕜₂
                                                 inst✝³ : NormedSpace 𝕜 E
                                                 inst✝² : NormedSpace 𝕜₂ F
                                                 σ₁₂ : RingHom 𝕜 𝕜₂
                                                 inst✝¹ : Nontrivial E
                                                 inst✝ : RingHomIsometric σ₁₂
                                                 f : LinearIsometry σ₁₂ E F
                                                 ⊢ ∀ (x : E), Eq (Norm.norm (f.toContinuousLinearMap x)) (HMul.hMul 1 (Norm.nor …
                                               -/
  f.toContinuousLinearMap.homothety_norm <| by simp
                                               /-
                                                 🎉 no goals
                                               -/


/-- Postcomposition of a continuous linear map with a linear isometry preserves
the operator norm. -/
theorem norm_toContinuousLinearMap_comp [RingHomIsometric σ₁₂] (f : F →ₛₗᵢ[σ₂₃] G)
    {g : E →SL[σ₁₂] F} : ‖f.toContinuousLinearMap.comp g‖ = ‖g‖ :=
  opNorm_ext (f.toContinuousLinearMap.comp g) g fun x => by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_7
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NontriviallyNormedField 𝕜₃
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜₃ G
      σ₁₂ : RingHom 𝕜 𝕜₂
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      σ₁₃ : RingHom 𝕜 𝕜₃
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝ : RingHomIsometric σ₁₂
      f : LinearIsometry σ₂₃ F G
      g : ContinuousLinearMap σ₁₂ E F
      x : E
      ⊢ Eq (Norm.norm ((f.toContinuousLinearMap.comp g) x)) (Norm.norm (g x))
    -/
    simp only [norm_map, coe_toContinuousLinearMap, coe_comp', Function.comp_apply]
    /-
      🎉 no goals
    -/


/-- Composing on the left with a linear isometry gives a linear isometry between spaces of
continuous linear maps. -/
def postcomp [RingHomIsometric σ₁₂] [RingHomIsometric σ₁₃] (a : F →ₛₗᵢ[σ₂₃] G) :
    (E →SL[σ₁₂] F) →ₛₗᵢ[σ₂₃] (E →SL[σ₁₃] G) where
  toFun f := a.toContinuousLinearMap.comp f
                     /-
                       𝕜 : Type u_1
                       𝕜₂ : Type u_2
                       𝕜₃ : Type u_3
                       E : Type u_4
                       F : Type u_5
                       Fₗ : Type u_6
                       G : Type u_7
                       inst✝¹³ : NormedAddCommGroup E
                       inst✝¹² : NormedAddCommGroup F
                       inst✝¹¹ : NormedAddCommGroup G
                       inst✝¹⁰ : NormedAddCommGroup Fₗ
                       inst✝⁹ : NontriviallyNormedField 𝕜
                       inst✝⁸ : NontriviallyNormedField 𝕜₂
                       inst✝⁷ : NontriviallyNormedField 𝕜₃
                       inst✝⁶ : NormedSpace 𝕜 E
                       inst✝⁵ : NormedSpace 𝕜₂ F
                       inst✝⁴ : NormedSpace 𝕜₃ G
                       inst✝³ : NormedSpace 𝕜 Fₗ
                       σ₁₂ : RingHom 𝕜 𝕜₂
                       σ₂₃ : RingHom 𝕜₂ 𝕜₃
                       f✝ : ContinuousLinearMap σ₁₂ E F
                       σ₁₃ : RingHom 𝕜 𝕜₃
                       inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                       inst✝¹ : RingHomIsometric σ₁₂
                       inst✝ : RingHomIsometric σ₁₃
                       a : LinearIsometry σ₂₃ F G
                       f g : ContinuousLinearMap σ₁₂ E F
                       ⊢ Eq ((fun f => a.toContinuousLinearMap.comp f) (HAdd.hAdd f g)) (HAdd.hAdd (( …
                     -/
  map_add' f g := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        𝕜 : Type u_1
                        𝕜₂ : Type u_2
                        𝕜₃ : Type u_3
                        E : Type u_4
                        F : Type u_5
                        Fₗ : Type u_6
                        G : Type u_7
                        inst✝¹³ : NormedAddCommGroup E
                        inst✝¹² : NormedAddCommGroup F
                        inst✝¹¹ : NormedAddCommGroup G
                        inst✝¹⁰ : NormedAddCommGroup Fₗ
                        inst✝⁹ : NontriviallyNormedField 𝕜
                        inst✝⁸ : NontriviallyNormedField 𝕜₂
                        inst✝⁷ : NontriviallyNormedField 𝕜₃
                        inst✝⁶ : NormedSpace 𝕜 E
                        inst✝⁵ : NormedSpace 𝕜₂ F
                        inst✝⁴ : NormedSpace 𝕜₃ G
                        inst✝³ : NormedSpace 𝕜 Fₗ
                        σ₁₂ : RingHom 𝕜 𝕜₂
                        σ₂₃ : RingHom 𝕜₂ 𝕜₃
                        f✝ : ContinuousLinearMap σ₁₂ E F
                        σ₁₃ : RingHom 𝕜 𝕜₃
                        inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        inst✝¹ : RingHomIsometric σ₁₂
                        inst✝ : RingHomIsometric σ₁₃
                        a : LinearIsometry σ₂₃ F G
                        c : 𝕜₂
                        f : ContinuousLinearMap σ₁₂ E F
                        ⊢ Eq ({ toFun := fun f => a.toContinuousLinearMap.comp f, map_add' := ⋯ }.toFu …
                      -/
  map_smul' c f := by simp
                      /-
                        🎉 no goals
                      -/
                    /-
                      𝕜 : Type u_1
                      𝕜₂ : Type u_2
                      𝕜₃ : Type u_3
                      E : Type u_4
                      F : Type u_5
                      Fₗ : Type u_6
                      G : Type u_7
                      inst✝¹³ : NormedAddCommGroup E
                      inst✝¹² : NormedAddCommGroup F
                      inst✝¹¹ : NormedAddCommGroup G
                      inst✝¹⁰ : NormedAddCommGroup Fₗ
                      inst✝⁹ : NontriviallyNormedField 𝕜
                      inst✝⁸ : NontriviallyNormedField 𝕜₂
                      inst✝⁷ : NontriviallyNormedField 𝕜₃
                      inst✝⁶ : NormedSpace 𝕜 E
                      inst✝⁵ : NormedSpace 𝕜₂ F
                      inst✝⁴ : NormedSpace 𝕜₃ G
                      inst✝³ : NormedSpace 𝕜 Fₗ
                      σ₁₂ : RingHom 𝕜 𝕜₂
                      σ₂₃ : RingHom 𝕜₂ 𝕜₃
                      f✝ : ContinuousLinearMap σ₁₂ E F
                      σ₁₃ : RingHom 𝕜 𝕜₃
                      inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                      inst✝¹ : RingHomIsometric σ₁₂
                      inst✝ : RingHomIsometric σ₁₃
                      a : LinearIsometry σ₂₃ F G
                      f : ContinuousLinearMap σ₁₂ E F
                      ⊢ Eq (Norm.norm ({ toFun := fun f => a.toContinuousLinearMap.comp f, map_add'  …
                    -/
  norm_map' f := by simp [a.norm_toContinuousLinearMap_comp]
                    /-
                      🎉 no goals
                    -/


/-- Precomposition with a linear isometry preserves the operator norm. -/
theorem opNorm_comp_linearIsometryEquiv (f : F →SL[σ₂₃] G) (g : F' ≃ₛₗᵢ[σ₂'] F) :
    ‖f.comp g.toLinearIsometry.toContinuousLinearMap‖ = ‖f‖ := by
  /-
    𝕜₂ : Type u_2
    𝕜₃ : Type u_3
    F : Type u_5
    G : Type u_7
    inst✝¹⁶ : NormedAddCommGroup F
    inst✝¹⁵ : NormedAddCommGroup G
    inst✝¹⁴ : NontriviallyNormedField 𝕜₂
    inst✝¹³ : NontriviallyNormedField 𝕜₃
    inst✝¹² : NormedSpace 𝕜₂ F
    inst✝¹¹ : NormedSpace 𝕜₃ G
    σ₂₃ : RingHom 𝕜₂ 𝕜₃
    𝕜₂' : Type u_8
    inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
    F' : Type u_9
    inst✝⁹ : NormedAddCommGroup F'
    inst✝⁸ : NormedSpace 𝕜₂' F'
    σ₂' : RingHom 𝕜₂' 𝕜₂
    σ₂'' : RingHom 𝕜₂ 𝕜₂'
    σ₂₃' : RingHom 𝕜₂' 𝕜₃
    inst✝⁷ : RingHomInvPair σ₂' σ₂''
    inst✝⁶ : RingHomInvPair σ₂'' σ₂'
    inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
    inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
    inst✝³ : RingHomIsometric σ₂₃
    inst✝² : RingHomIsometric σ₂'
    inst✝¹ : RingHomIsometric σ₂''
    inst✝ : RingHomIsometric σ₂₃'
    f : ContinuousLinearMap σ₂₃ F G
    g : LinearIsometryEquiv σ₂' F' F
    ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (Norm.norm f)
  -/
  cases subsingleton_or_nontrivial F'
    /-
      case inl
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Subsingleton F'
      ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (Norm.norm f)
    -/
  · haveI := g.symm.toLinearEquiv.toEquiv.subsingleton
    /-
      case inl
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Subsingleton F'
      this : Subsingleton F
      ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (Norm.norm f)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜₂ : Type u_2
    𝕜₃ : Type u_3
    F : Type u_5
    G : Type u_7
    inst✝¹⁶ : NormedAddCommGroup F
    inst✝¹⁵ : NormedAddCommGroup G
    inst✝¹⁴ : NontriviallyNormedField 𝕜₂
    inst✝¹³ : NontriviallyNormedField 𝕜₃
    inst✝¹² : NormedSpace 𝕜₂ F
    inst✝¹¹ : NormedSpace 𝕜₃ G
    σ₂₃ : RingHom 𝕜₂ 𝕜₃
    𝕜₂' : Type u_8
    inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
    F' : Type u_9
    inst✝⁹ : NormedAddCommGroup F'
    inst✝⁸ : NormedSpace 𝕜₂' F'
    σ₂' : RingHom 𝕜₂' 𝕜₂
    σ₂'' : RingHom 𝕜₂ 𝕜₂'
    σ₂₃' : RingHom 𝕜₂' 𝕜₃
    inst✝⁷ : RingHomInvPair σ₂' σ₂''
    inst✝⁶ : RingHomInvPair σ₂'' σ₂'
    inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
    inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
    inst✝³ : RingHomIsometric σ₂₃
    inst✝² : RingHomIsometric σ₂'
    inst✝¹ : RingHomIsometric σ₂''
    inst✝ : RingHomIsometric σ₂₃'
    f : ContinuousLinearMap σ₂₃ F G
    g : LinearIsometryEquiv σ₂' F' F
    h✝ : Nontrivial F'
    ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (Norm.norm f)
  -/
  refine le_antisymm ?_ ?_
    /-
      case inr.refine_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Nontrivial F'
      ⊢ LE.le (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (Norm.no …
    -/
  · convert f.opNorm_comp_le g.toLinearIsometry.toContinuousLinearMap
    /-
      case h.e'_4
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Nontrivial F'
      ⊢ Eq (Norm.norm f) (HMul.hMul (Norm.norm f) (Norm.norm g.toLinearIsometry.toCo …
    -/
    simp [g.toLinearIsometry.norm_toContinuousLinearMap]
    /-
      🎉 no goals
    -/
  · convert (f.comp g.toLinearIsometry.toContinuousLinearMap).opNorm_comp_le
        g.symm.toLinearIsometry.toContinuousLinearMap
      /-
        case h.e'_3.h.e'_3
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        F : Type u_5
        G : Type u_7
        inst✝¹⁶ : NormedAddCommGroup F
        inst✝¹⁵ : NormedAddCommGroup G
        inst✝¹⁴ : NontriviallyNormedField 𝕜₂
        inst✝¹³ : NontriviallyNormedField 𝕜₃
        inst✝¹² : NormedSpace 𝕜₂ F
        inst✝¹¹ : NormedSpace 𝕜₃ G
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        𝕜₂' : Type u_8
        inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
        F' : Type u_9
        inst✝⁹ : NormedAddCommGroup F'
        inst✝⁸ : NormedSpace 𝕜₂' F'
        σ₂' : RingHom 𝕜₂' 𝕜₂
        σ₂'' : RingHom 𝕜₂ 𝕜₂'
        σ₂₃' : RingHom 𝕜₂' 𝕜₃
        inst✝⁷ : RingHomInvPair σ₂' σ₂''
        inst✝⁶ : RingHomInvPair σ₂'' σ₂'
        inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
        inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
        inst✝³ : RingHomIsometric σ₂₃
        inst✝² : RingHomIsometric σ₂'
        inst✝¹ : RingHomIsometric σ₂''
        inst✝ : RingHomIsometric σ₂₃'
        f : ContinuousLinearMap σ₂₃ F G
        g : LinearIsometryEquiv σ₂' F' F
        h✝ : Nontrivial F'
        ⊢ Eq f ((f.comp g.toLinearIsometry.toContinuousLinearMap).comp g.symm.toLinear …
      -/
    · ext
      /-
        case h.e'_3.h.e'_3.h
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        F : Type u_5
        G : Type u_7
        inst✝¹⁶ : NormedAddCommGroup F
        inst✝¹⁵ : NormedAddCommGroup G
        inst✝¹⁴ : NontriviallyNormedField 𝕜₂
        inst✝¹³ : NontriviallyNormedField 𝕜₃
        inst✝¹² : NormedSpace 𝕜₂ F
        inst✝¹¹ : NormedSpace 𝕜₃ G
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        𝕜₂' : Type u_8
        inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
        F' : Type u_9
        inst✝⁹ : NormedAddCommGroup F'
        inst✝⁸ : NormedSpace 𝕜₂' F'
        σ₂' : RingHom 𝕜₂' 𝕜₂
        σ₂'' : RingHom 𝕜₂ 𝕜₂'
        σ₂₃' : RingHom 𝕜₂' 𝕜₃
        inst✝⁷ : RingHomInvPair σ₂' σ₂''
        inst✝⁶ : RingHomInvPair σ₂'' σ₂'
        inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
        inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
        inst✝³ : RingHomIsometric σ₂₃
        inst✝² : RingHomIsometric σ₂'
        inst✝¹ : RingHomIsometric σ₂''
        inst✝ : RingHomIsometric σ₂₃'
        f : ContinuousLinearMap σ₂₃ F G
        g : LinearIsometryEquiv σ₂' F' F
        h✝ : Nontrivial F'
        x✝ : F
        ⊢ Eq (f x✝) (((f.comp g.toLinearIsometry.toContinuousLinearMap).comp g.symm.to …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case h.e'_4
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Nontrivial F'
      ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (HMul.hMul  …
    -/
    haveI := g.symm.surjective.nontrivial
    /-
      case h.e'_4
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      F : Type u_5
      G : Type u_7
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedAddCommGroup G
      inst✝¹⁴ : NontriviallyNormedField 𝕜₂
      inst✝¹³ : NontriviallyNormedField 𝕜₃
      inst✝¹² : NormedSpace 𝕜₂ F
      inst✝¹¹ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      𝕜₂' : Type u_8
      inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
      F' : Type u_9
      inst✝⁹ : NormedAddCommGroup F'
      inst✝⁸ : NormedSpace 𝕜₂' F'
      σ₂' : RingHom 𝕜₂' 𝕜₂
      σ₂'' : RingHom 𝕜₂ 𝕜₂'
      σ₂₃' : RingHom 𝕜₂' 𝕜₃
      inst✝⁷ : RingHomInvPair σ₂' σ₂''
      inst✝⁶ : RingHomInvPair σ₂'' σ₂'
      inst✝⁵ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
      inst✝⁴ : RingHomCompTriple σ₂'' σ₂₃' σ₂₃
      inst✝³ : RingHomIsometric σ₂₃
      inst✝² : RingHomIsometric σ₂'
      inst✝¹ : RingHomIsometric σ₂''
      inst✝ : RingHomIsometric σ₂₃'
      f : ContinuousLinearMap σ₂₃ F G
      g : LinearIsometryEquiv σ₂' F' F
      h✝ : Nontrivial F'
      this : Nontrivial F
      ⊢ Eq (Norm.norm (f.comp g.toLinearIsometry.toContinuousLinearMap)) (HMul.hMul  …
    -/
    simp [g.symm.toLinearIsometry.norm_toContinuousLinearMap]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")]
alias op_norm_comp_linearIsometryEquiv := opNorm_comp_linearIsometryEquiv


@[simp]
theorem norm_smulRightL (c : E →L[𝕜] 𝕜) [Nontrivial Fₗ] : ‖smulRightL 𝕜 E Fₗ c‖ = ‖c‖ :=
  ContinuousLinearMap.homothety_norm _ c.norm_smulRight_apply


set_option maxSynthPendingDepth 2 in
lemma norm_smulRightL_le : ‖smulRightL 𝕜 E Fₗ‖ ≤ 1 :=
  LinearMap.mkContinuous₂_norm_le _ zero_le_one _


theorem norm_subtypeL (K : Submodule 𝕜 E) [Nontrivial K] : ‖K.subtypeL‖ = 1 :=
  K.subtypeₗᵢ.norm_toContinuousLinearMap


protected theorem antilipschitz (e : E ≃SL[σ₁₂] F) :
    AntilipschitzWith ‖(e.symm : F →SL[σ₂₁] E)‖₊ e :=
  e.symm.lipschitz.to_rightInverse e.left_inv


theorem one_le_norm_mul_norm_symm [RingHomIsometric σ₁₂] [Nontrivial E] (e : E ≃SL[σ₁₂] F) :
    1 ≤ ‖(e : E →SL[σ₁₂] F)‖ * ‖(e.symm : F →SL[σ₂₁] E)‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    σ₂₁ : RingHom 𝕜₂ 𝕜
    inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
    inst✝³ : RingHomInvPair σ₂₁ σ₁₂
    inst✝² : RingHomIsometric σ₂₁
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    e : ContinuousLinearEquiv σ₁₂ E F
    ⊢ LE.le 1 (HMul.hMul (Norm.norm ↑e) (Norm.norm ↑e.symm))
  -/
  rw [mul_comm]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    σ₂₁ : RingHom 𝕜₂ 𝕜
    inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
    inst✝³ : RingHomInvPair σ₂₁ σ₁₂
    inst✝² : RingHomIsometric σ₂₁
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    e : ContinuousLinearEquiv σ₁₂ E F
    ⊢ LE.le 1 (HMul.hMul (Norm.norm ↑e.symm) (Norm.norm ↑e))
  -/
  convert (e.symm : F →SL[σ₂₁] E).opNorm_comp_le (e : E →SL[σ₁₂] F)
  /-
    case h.e'_3
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    σ₂₁ : RingHom 𝕜₂ 𝕜
    inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
    inst✝³ : RingHomInvPair σ₂₁ σ₁₂
    inst✝² : RingHomIsometric σ₂₁
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : Nontrivial E
    e : ContinuousLinearEquiv σ₁₂ E F
    ⊢ Eq 1 (Norm.norm ((↑e.symm).comp ↑e))
  -/
  rw [e.coe_symm_comp_coe, ContinuousLinearMap.norm_id]
  /-
    🎉 no goals
  -/


theorem norm_pos [RingHomIsometric σ₁₂] [Nontrivial E] (e : E ≃SL[σ₁₂] F) :
    0 < ‖(e : E →SL[σ₁₂] F)‖ :=
  pos_of_mul_pos_left (lt_of_lt_of_le zero_lt_one e.one_le_norm_mul_norm_symm) (norm_nonneg _)


theorem norm_symm_pos [RingHomIsometric σ₁₂] [Nontrivial E] (e : E ≃SL[σ₁₂] F) :
    0 < ‖(e.symm : F →SL[σ₂₁] E)‖ :=
  pos_of_mul_pos_right (zero_lt_one.trans_le e.one_le_norm_mul_norm_symm) (norm_nonneg _)


theorem nnnorm_symm_pos [RingHomIsometric σ₁₂] [Nontrivial E] (e : E ≃SL[σ₁₂] F) :
    0 < ‖(e.symm : F →SL[σ₂₁] E)‖₊ :=
  e.norm_symm_pos


theorem subsingleton_or_norm_symm_pos [RingHomIsometric σ₁₂] (e : E ≃SL[σ₁₂] F) :
    Subsingleton E ∨ 0 < ‖(e.symm : F →SL[σ₂₁] E)‖ := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜₂
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    σ₂₁ : RingHom 𝕜₂ 𝕜
    inst✝³ : RingHomInvPair σ₁₂ σ₂₁
    inst✝² : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : RingHomIsometric σ₂₁
    inst✝ : RingHomIsometric σ₁₂
    e : ContinuousLinearEquiv σ₁₂ E F
    ⊢ Or (Subsingleton E) (LT.lt 0 (Norm.norm ↑e.symm))
  -/
  rcases subsingleton_or_nontrivial E with (_i | _i)
    /-
      case inl
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      σ₂₁ : RingHom 𝕜₂ 𝕜
      inst✝³ : RingHomInvPair σ₁₂ σ₂₁
      inst✝² : RingHomInvPair σ₂₁ σ₁₂
      inst✝¹ : RingHomIsometric σ₂₁
      inst✝ : RingHomIsometric σ₁₂
      e : ContinuousLinearEquiv σ₁₂ E F
      _i : Subsingleton E
      ⊢ Or (Subsingleton E) (LT.lt 0 (Norm.norm ↑e.symm))
    -/
  · left
    /-
      case inl.h
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      σ₂₁ : RingHom 𝕜₂ 𝕜
      inst✝³ : RingHomInvPair σ₁₂ σ₂₁
      inst✝² : RingHomInvPair σ₂₁ σ₁₂
      inst✝¹ : RingHomIsometric σ₂₁
      inst✝ : RingHomIsometric σ₁₂
      e : ContinuousLinearEquiv σ₁₂ E F
      _i : Subsingleton E
      ⊢ Subsingleton E
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      σ₂₁ : RingHom 𝕜₂ 𝕜
      inst✝³ : RingHomInvPair σ₁₂ σ₂₁
      inst✝² : RingHomInvPair σ₂₁ σ₁₂
      inst✝¹ : RingHomIsometric σ₂₁
      inst✝ : RingHomIsometric σ₁₂
      e : ContinuousLinearEquiv σ₁₂ E F
      _i : Nontrivial E
      ⊢ Or (Subsingleton E) (LT.lt 0 (Norm.norm ↑e.symm))
    -/
  · right
    /-
      case inr.h
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_4
      F : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      σ₂₁ : RingHom 𝕜₂ 𝕜
      inst✝³ : RingHomInvPair σ₁₂ σ₂₁
      inst✝² : RingHomInvPair σ₂₁ σ₁₂
      inst✝¹ : RingHomIsometric σ₂₁
      inst✝ : RingHomIsometric σ₁₂
      e : ContinuousLinearEquiv σ₁₂ E F
      _i : Nontrivial E
      ⊢ LT.lt 0 (Norm.norm ↑e.symm)
    -/
    exact e.norm_symm_pos
    /-
      🎉 no goals
    -/


theorem subsingleton_or_nnnorm_symm_pos [RingHomIsometric σ₁₂] (e : E ≃SL[σ₁₂] F) :
    Subsingleton E ∨ 0 < ‖(e.symm : F →SL[σ₂₁] E)‖₊ :=
  subsingleton_or_norm_symm_pos e


@[simp]
theorem coord_norm (x : E) (h : x ≠ 0) : ‖coord 𝕜 x h‖ = ‖x‖⁻¹ := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    ⊢ Eq (Norm.norm (ContinuousLinearEquiv.coord 𝕜 x h)) (Inv.inv (Norm.norm x))
  -/
  have hx : 0 < ‖x‖ := norm_pos_iff.mpr h
  /-
    𝕜 : Type u_1
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    hx : LT.lt 0 (Norm.norm x)
    ⊢ Eq (Norm.norm (ContinuousLinearEquiv.coord 𝕜 x h)) (Inv.inv (Norm.norm x))
  -/
  haveI : Nontrivial (𝕜 ∙ x) := Submodule.nontrivial_span_singleton h
  exact ContinuousLinearMap.homothety_norm _ fun y =>
    homothety_inverse _ hx _ (LinearEquiv.toSpanNonzeroSingleton_homothety 𝕜 x h) _


/-- A bounded bilinear form `B` in a real normed space is *coercive*
if there is some positive constant C such that `C * ‖u‖ * ‖u‖ ≤ B u u`.
-/
def IsCoercive [NormedAddCommGroup E] [NormedSpace ℝ E] (B : E →L[ℝ] E →L[ℝ] ℝ) : Prop :=
  ∃ C, 0 < C ∧ ∀ u, C * ‖u‖ * ‖u‖ ≤ B u u


/-- Equivalent characterizations for equicontinuity of a family of continuous linear maps
between normed spaces. See also `WithSeminorms.equicontinuous_TFAE` for similar characterizations
between spaces satisfying `WithSeminorms`. -/
protected theorem NormedSpace.equicontinuous_TFAE : List.TFAE
    [ EquicontinuousAt ((↑) ∘ f) 0,
      Equicontinuous ((↑) ∘ f),
      UniformEquicontinuous ((↑) ∘ f),
      ∃ C, ∀ i x, ‖f i x‖ ≤ C * ‖x‖,
      ∃ C ≥ 0, ∀ i x, ‖f i x‖ ≤ C * ‖x‖,
      ∃ C, ∀ i, ‖f i‖ ≤ C,
      ∃ C ≥ 0, ∀ i, ‖f i‖ ≤ C,
      BddAbove (Set.range (‖f ·‖)),
      (⨆ i, (‖f i‖₊ : ENNReal)) < ⊤ ] := by
  -- `1 ↔ 2 ↔ 3` follows from `uniformEquicontinuous_of_equicontinuousAt_zero`
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    ι : Type u_8
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    f : ι → ContinuousLinearMap σ₁₂ E F
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 1 → 3 := uniformEquicontinuous_of_equicontinuousAt_zero f
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    ι : Type u_8
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    f : ι → ContinuousLinearMap σ₁₂ E F
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 3 → 2 := UniformEquicontinuous.equicontinuous
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    ι : Type u_8
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    f : ι → ContinuousLinearMap σ₁₂ E F
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    tfae_3_to_2 : UniformEquicontinuous (Function.comp DFunLike.coe f) → Equiconti …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 2 → 1 := fun H ↦ H 0
  -- `4 ↔ 5 ↔ 6 ↔ 7 ↔ 8 ↔ 9` is morally trivial, we just have to use a lot of rewriting
  -- and `congr` lemmas
  tfae_have 4 ↔ 5 := by
    rw [exists_ge_and_iff_exists]
    exact fun C₁ C₂ hC ↦ forall₂_imp fun i x ↦ le_trans' <| by gcongr
  tfae_have 5 ↔ 7 := by
    refine exists_congr (fun C ↦ and_congr_right fun hC ↦ forall_congr' fun i ↦ ?_)
    rw [ContinuousLinearMap.opNorm_le_iff hC]
  tfae_have 7 ↔ 8 := by
    simp_rw [bddAbove_iff_exists_ge (0 : ℝ), Set.forall_mem_range]
  tfae_have 6 ↔ 8 := by
    simp_rw [bddAbove_def, Set.forall_mem_range]
  tfae_have 8 ↔ 9 := by
    rw [ENNReal.iSup_coe_lt_top, ← NNReal.bddAbove_coe, ← Set.range_comp]
    rfl
  -- `3 ↔ 4` is the interesting part of the result. It is essentially a combination of
  -- `WithSeminorms.uniformEquicontinuous_iff_exists_continuous_seminorm` which turns
  -- equicontinuity into existence of some continuous seminorm and
  -- `Seminorm.bound_of_continuous_normedSpace` which characterize such seminorms.
  tfae_have 3 ↔ 4 := by
    refine ((norm_withSeminorms 𝕜₂ F).uniformEquicontinuous_iff_exists_continuous_seminorm _).trans
      ?_
    rw [forall_const]
    constructor
    · intro ⟨p, hp, hpf⟩
      rcases p.bound_of_continuous_normedSpace hp with ⟨C, -, hC⟩
      exact ⟨C, fun i x ↦ (hpf i x).trans (hC x)⟩
    · intro ⟨C, hC⟩
      refine ⟨C.toNNReal • normSeminorm 𝕜 E,
        ((norm_withSeminorms 𝕜 E).continuous_seminorm 0).const_smul C.toNNReal, fun i x ↦ ?_⟩
      exact (hC i x).trans (mul_le_mul_of_nonneg_right (C.le_coe_toNNReal) (norm_nonneg x))
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_4
    F : Type u_5
    ι : Type u_8
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜₂ F
    f : ι → ContinuousLinearMap σ₁₂ E F
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    tfae_3_to_2 : UniformEquicontinuous (Function.comp DFunLike.coe f) → Equiconti …
    tfae_2_to_1 : Equicontinuous (Function.comp DFunLike.coe f) → EquicontinuousAt …
    tfae_4_iff_5 : Iff (Exists fun C => ∀ (i : ι) (x : E), LE.le (Norm.norm ((f i) …
    tfae_5_iff_7 : Iff (Exists fun C => And (GE.ge C 0) (∀ (i : ι) (x : E), LE.le  …
    tfae_7_iff_8 : Iff (Exists fun C => And (GE.ge C 0) (∀ (i : ι), LE.le (Norm.no …
    tfae_6_iff_8 : Iff (Exists fun C => ∀ (i : ι), LE.le (Norm.norm (f i)) C) (Bdd …
    tfae_8_iff_9 : Iff (BddAbove (Set.range fun x => Norm.norm (f x))) (LT.lt (iSu …
    tfae_3_iff_4 : Iff (UniformEquicontinuous (Function.comp DFunLike.coe f)) (Exi …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


