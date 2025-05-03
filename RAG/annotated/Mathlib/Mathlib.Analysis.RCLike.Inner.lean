/-- Weighted inner product giving rise to the L2 norm. -/
def wInner (w : ι → ℝ) (f g : ∀ i, E i) : 𝕜 := ∑ i, w i • inner (f i) (g i)


/-- The weight function making `wInner` into the compact inner product. -/
noncomputable abbrev cWeight : ι → ℝ := Function.const _ (Fintype.card ι)⁻¹


@[inherit_doc] notation "⟪" f ", " g "⟫_[" 𝕝 ", " w "]" => wInner (𝕜 := 𝕝) w f g


/-- Discrete inner product giving rise to the discrete L2 norm. -/
notation "⟪" f ", " g "⟫_[" 𝕝 "]" => ⟪f, g⟫_[𝕝, 1]


/-- Compact inner product giving rise to the compact L2 norm. -/
notation "⟪" f ", " g "⟫ₙ_[" 𝕝 "]" => ⟪f, g⟫_[𝕝, cWeight]


lemma wInner_cWeight_eq_smul_wInner_one (f g : ∀ i, E i) :
    ⟪f, g⟫ₙ_[𝕜] = (Fintype.card ι : ℚ≥0)⁻¹ • ⟪f, g⟫_[𝕜] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner RCLike.cWeight f g) (HSMul.hSMul (Inv.inv ↑(Fintype.card ι …
  -/
  simp [wInner, smul_sum, ← NNRat.cast_smul_eq_nnqsmul ℝ]
  /-
    🎉 no goals
  -/


@[simp] lemma conj_wInner_symm (w : ι → ℝ) (f g : ∀ i, E i) :
    conj ⟪f, g⟫_[𝕜, w] = ⟪g, f⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f g : (i : ι) → E i
    ⊢ Eq ((starRingEnd 𝕜) (RCLike.wInner w f g)) (RCLike.wInner w g f)
  -/
  simp [wInner, map_sum, inner_conj_symm, rclike_simps]
  /-
    🎉 no goals
  -/


                                                                                    /-
                                                                                      ι : Type u_1
                                                                                      𝕜 : Type u_3
                                                                                      E : ι → Type u_4
                                                                                      inst✝³ : Fintype ι
                                                                                      inst✝² : RCLike 𝕜
                                                                                      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
                                                                                      inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
                                                                                      w : ι → Real
                                                                                      g : (i : ι) → E i
                                                                                      ⊢ Eq (RCLike.wInner w 0 g) 0
                                                                                    -/
@[simp] lemma wInner_zero_left (w : ι → ℝ) (g : ∀ i, E i) : ⟪0, g⟫_[𝕜, w] = 0 := by simp [wInner]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

                                                                                     /-
                                                                                       ι : Type u_1
                                                                                       𝕜 : Type u_3
                                                                                       E : ι → Type u_4
                                                                                       inst✝³ : Fintype ι
                                                                                       inst✝² : RCLike 𝕜
                                                                                       inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
                                                                                       inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
                                                                                       w : ι → Real
                                                                                       f : (i : ι) → E i
                                                                                       ⊢ Eq (RCLike.wInner w f 0) 0
                                                                                     -/
@[simp] lemma wInner_zero_right (w : ι → ℝ) (f : ∀ i, E i) : ⟪f, 0⟫_[𝕜, w] = 0 := by simp [wInner]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma wInner_add_left (w : ι → ℝ) (f₁ f₂ g : ∀ i, E i) :
    ⟪f₁ + f₂, g⟫_[𝕜, w] = ⟪f₁, g⟫_[𝕜, w] + ⟪f₂, g⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f₁ f₂ g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w (HAdd.hAdd f₁ f₂) g) (HAdd.hAdd (RCLike.wInner w f₁ g) ( …
  -/
  simp [wInner, inner_add_left, smul_add, sum_add_distrib]
  /-
    🎉 no goals
  -/


lemma wInner_add_right (w : ι → ℝ) (f g₁ g₂ : ∀ i, E i) :
    ⟪f, g₁ + g₂⟫_[𝕜, w] = ⟪f, g₁⟫_[𝕜, w] + ⟪f, g₂⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f g₁ g₂ : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w f (HAdd.hAdd g₁ g₂)) (HAdd.hAdd (RCLike.wInner w f g₁) ( …
  -/
  simp [wInner, inner_add_right, smul_add, sum_add_distrib]
  /-
    🎉 no goals
  -/


@[simp] lemma wInner_neg_left (w : ι → ℝ) (f g : ∀ i, E i) : ⟪-f, g⟫_[𝕜, w] = -⟪f, g⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w (Neg.neg f) g) (Neg.neg (RCLike.wInner w f g))
  -/
  simp [wInner]
  /-
    🎉 no goals
  -/


@[simp] lemma wInner_neg_right (w : ι → ℝ) (f g : ∀ i, E i) : ⟪f, -g⟫_[𝕜, w] = -⟪f, g⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w f (Neg.neg g)) (Neg.neg (RCLike.wInner w f g))
  -/
  simp [wInner]
  /-
    🎉 no goals
  -/


lemma wInner_sub_left (w : ι → ℝ) (f₁ f₂ g : ∀ i, E i) :
    ⟪f₁ - f₂, g⟫_[𝕜, w] = ⟪f₁, g⟫_[𝕜, w] - ⟪f₂, g⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f₁ f₂ g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w (HSub.hSub f₁ f₂) g) (HSub.hSub (RCLike.wInner w f₁ g) ( …
  -/
  simp_rw [sub_eq_add_neg, wInner_add_left, wInner_neg_left]
  /-
    🎉 no goals
  -/


lemma wInner_sub_right (w : ι → ℝ) (f g₁ g₂ : ∀ i, E i) :
    ⟪f, g₁ - g₂⟫_[𝕜, w] = ⟪f, g₁⟫_[𝕜, w] - ⟪f, g₂⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    w : ι → Real
    f g₁ g₂ : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w f (HSub.hSub g₁ g₂)) (HSub.hSub (RCLike.wInner w f g₁) ( …
  -/
  simp_rw [sub_eq_add_neg, wInner_add_right, wInner_neg_right]
  /-
    🎉 no goals
  -/


@[simp] lemma wInner_of_isEmpty [IsEmpty ι] (w : ι → ℝ) (f g : ∀ i, E i) : ⟪f, g⟫_[𝕜, w] = 0 := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝⁴ : Fintype ι
    inst✝³ : RCLike 𝕜
    inst✝² : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (E i)
    inst✝ : IsEmpty ι
    w : ι → Real
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w f g) 0
  -/
  simp [Subsingleton.elim f 0]
  /-
    🎉 no goals
  -/


lemma wInner_smul_left {𝕝 : Type*} [CommSemiring 𝕝] [StarRing 𝕝] [Algebra 𝕝 𝕜] [StarModule 𝕝 𝕜]
    [SMulCommClass ℝ 𝕝 𝕜] [∀ i, Module 𝕝 (E i)] [∀ i, IsScalarTower 𝕝 𝕜 (E i)] (c : 𝕝)
    (w : ι → ℝ) (f g : ∀ i, E i) : ⟪c • f, g⟫_[𝕜, w] = star c • ⟪f, g⟫_[𝕜, w] := by
  simp_rw [wInner, Pi.smul_apply, inner_smul_left_eq_star_smul, starRingEnd_apply, smul_sum,
    smul_comm (w _)]


lemma wInner_smul_right {𝕝 : Type*} [CommSemiring 𝕝] [StarRing 𝕝] [Algebra 𝕝 𝕜] [StarModule 𝕝 𝕜]
    [SMulCommClass ℝ 𝕝 𝕜] [∀ i, Module 𝕝 (E i)] [∀ i, IsScalarTower 𝕝 𝕜 (E i)] (c : 𝕝)
    (w : ι → ℝ) (f g : ∀ i, E i) : ⟪f, c • g⟫_[𝕜, w] = c • ⟪f, g⟫_[𝕜, w] := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝¹⁰ : Fintype ι
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁷ : (i : ι) → InnerProductSpace 𝕜 (E i)
    𝕝 : Type u_5
    inst✝⁶ : CommSemiring 𝕝
    inst✝⁵ : StarRing 𝕝
    inst✝⁴ : Algebra 𝕝 𝕜
    inst✝³ : StarModule 𝕝 𝕜
    inst✝² : SMulCommClass Real 𝕝 𝕜
    inst✝¹ : (i : ι) → Module 𝕝 (E i)
    inst✝ : ∀ (i : ι), IsScalarTower 𝕝 𝕜 (E i)
    c : 𝕝
    w : ι → Real
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner w f (HSMul.hSMul c g)) (HSMul.hSMul c (RCLike.wInner w f g))
  -/
  simp_rw [wInner, Pi.smul_apply, inner_smul_right_eq_smul, smul_sum, smul_comm]
  /-
    🎉 no goals
  -/


lemma mul_wInner_left (c : 𝕜) (w : ι → ℝ) (f g : ∀ i, E i) :
                                                     /-
                                                       ι : Type u_1
                                                       𝕜 : Type u_3
                                                       E : ι → Type u_4
                                                       inst✝³ : Fintype ι
                                                       inst✝² : RCLike 𝕜
                                                       inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
                                                       inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
                                                       c : 𝕜
                                                       w : ι → Real
                                                       f g : (i : ι) → E i
                                                       ⊢ Eq (HMul.hMul c (RCLike.wInner w f g)) (RCLike.wInner w (HSMul.hSMul (Star.s …
                                                     -/
    c * ⟪f, g⟫_[𝕜, w] = ⟪star c • f, g⟫_[𝕜, w] := by rw [wInner_smul_left, star_star, smul_eq_mul]
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                                                     /-
                                                                                       ι : Type u_1
                                                                                       𝕜 : Type u_3
                                                                                       E : ι → Type u_4
                                                                                       inst✝³ : Fintype ι
                                                                                       inst✝² : RCLike 𝕜
                                                                                       inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
                                                                                       inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
                                                                                       f g : (i : ι) → E i
                                                                                       ⊢ Eq (RCLike.wInner 1 f g) (Finset.univ.sum fun i => Inner.inner (f i) (g i))
                                                                                     -/
lemma wInner_one_eq_sum (f g : ∀ i, E i) : ⟪f, g⟫_[𝕜] = ∑ i, inner (f i) (g i) := by simp [wInner]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/

lemma wInner_cWeight_eq_expect (f g : ∀ i, E i) : ⟪f, g⟫ₙ_[𝕜] = 𝔼 i, inner (f i) (g i) := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    E : ι → Type u_4
    inst✝³ : Fintype ι
    inst✝² : RCLike 𝕜
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (E i)
    f g : (i : ι) → E i
    ⊢ Eq (RCLike.wInner RCLike.cWeight f g) (Finset.univ.expect fun i => Inner.inn …
  -/
  simp [wInner, expect, smul_sum, ← NNRat.cast_smul_eq_nnqsmul ℝ]
  /-
    🎉 no goals
  -/


lemma wInner_const_left (a : 𝕜) (f : ι → 𝕜) :
                                                          /-
                                                            ι : Type u_1
                                                            𝕜 : Type u_3
                                                            inst✝¹ : Fintype ι
                                                            inst✝ : RCLike 𝕜
                                                            w : ι → Real
                                                            a : 𝕜
                                                            f : ι → 𝕜
                                                            ⊢ Eq (RCLike.wInner w (Function.const ι a) f) (HMul.hMul ((starRingEnd 𝕜) a) ( …
                                                          -/
    ⟪const _ a, f⟫_[𝕜, w] = conj a * ∑ i, w i • f i := by simp [wInner, const_apply, mul_sum]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma wInner_const_right (f : ι → 𝕜) (a : 𝕜) :
                                                              /-
                                                                ι : Type u_1
                                                                𝕜 : Type u_3
                                                                inst✝¹ : Fintype ι
                                                                inst✝ : RCLike 𝕜
                                                                w : ι → Real
                                                                f : ι → 𝕜
                                                                a : 𝕜
                                                                ⊢ Eq (RCLike.wInner w f (Function.const ι a)) (HMul.hMul (Finset.univ.sum fun  …
                                                              -/
    ⟪f, const _ a⟫_[𝕜, w] = (∑ i, w i • conj (f i)) * a := by simp [wInner, const_apply, sum_mul]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] lemma wInner_one_const_left (a : 𝕜) (f : ι → 𝕜) :
                                                 /-
                                                   ι : Type u_1
                                                   𝕜 : Type u_3
                                                   inst✝¹ : Fintype ι
                                                   inst✝ : RCLike 𝕜
                                                   a : 𝕜
                                                   f : ι → 𝕜
                                                   ⊢ Eq (RCLike.wInner 1 (Function.const ι a) f) (HMul.hMul ((starRingEnd 𝕜) a) ( …
                                                 -/
    ⟪const _ a, f⟫_[𝕜] = conj a * ∑ i, f i := by simp [wInner_one_eq_sum, mul_sum]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp] lemma wInner_one_const_right (f : ι → 𝕜) (a : 𝕜) :
                                                     /-
                                                       ι : Type u_1
                                                       𝕜 : Type u_3
                                                       inst✝¹ : Fintype ι
                                                       inst✝ : RCLike 𝕜
                                                       f : ι → 𝕜
                                                       a : 𝕜
                                                       ⊢ Eq (RCLike.wInner 1 f (Function.const ι a)) (HMul.hMul (Finset.univ.sum fun  …
                                                     -/
    ⟪f, const _ a⟫_[𝕜] = (∑ i, conj (f i)) * a := by simp [wInner_one_eq_sum, sum_mul]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma wInner_cWeight_const_left (a : 𝕜) (f : ι → 𝕜) :
                                                  /-
                                                    ι : Type u_1
                                                    𝕜 : Type u_3
                                                    inst✝¹ : Fintype ι
                                                    inst✝ : RCLike 𝕜
                                                    a : 𝕜
                                                    f : ι → 𝕜
                                                    ⊢ Eq (RCLike.wInner RCLike.cWeight (Function.const ι a) f) (HMul.hMul ((starRi …
                                                  -/
    ⟪const _ a, f⟫ₙ_[𝕜] = conj a * 𝔼 i, f i := by simp [wInner_cWeight_eq_expect, mul_expect]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] lemma wInner_cWeight_const_right (f : ι → 𝕜) (a : 𝕜) :
                                                      /-
                                                        ι : Type u_1
                                                        𝕜 : Type u_3
                                                        inst✝¹ : Fintype ι
                                                        inst✝ : RCLike 𝕜
                                                        f : ι → 𝕜
                                                        a : 𝕜
                                                        ⊢ Eq (RCLike.wInner RCLike.cWeight f (Function.const ι a)) (HMul.hMul (Finset. …
                                                      -/
    ⟪f, const _ a⟫ₙ_[𝕜] = (𝔼 i, conj (f i)) * a := by simp [wInner_cWeight_eq_expect, expect_mul]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma wInner_one_eq_inner (f g : ι → 𝕜) :
    ⟪f, g⟫_[𝕜, 1] = inner ((WithLp.equiv 2 _).symm f) ((WithLp.equiv 2 _).symm g) := by
  /-
    ι : Type u_1
    𝕜 : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : RCLike 𝕜
    f g : ι → 𝕜
    ⊢ Eq (RCLike.wInner 1 f g) (Inner.inner ((WithLp.equiv 2 (ι → 𝕜)).symm f) ((Wi …
  -/
  simp [wInner]
  /-
    🎉 no goals
  -/


lemma inner_eq_wInner_one (f g : PiLp 2 fun _i : ι ↦ 𝕜) :
                                                                      /-
                                                                        ι : Type u_1
                                                                        𝕜 : Type u_3
                                                                        inst✝¹ : Fintype ι
                                                                        inst✝ : RCLike 𝕜
                                                                        f g : PiLp 2 fun _i => 𝕜
                                                                        ⊢ Eq (Inner.inner f g) (RCLike.wInner 1 ((WithLp.equiv 2 (ι → 𝕜)) f) ((WithLp. …
                                                                      -/
    inner f g = ⟪WithLp.equiv 2 _ f, WithLp.equiv 2 _ g⟫_[𝕜, 1] := by simp [wInner]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma linearIndependent_of_ne_zero_of_wInner_one_eq_zero {f : κ → ι → 𝕜} (hf : ∀ k, f k ≠ 0)
    (hinner : Pairwise fun k₁ k₂ ↦ ⟪f k₁, f k₂⟫_[𝕜] = 0) : LinearIndependent 𝕜 f := by
  /-
    ι : Type u_1
    κ : Type u_2
    𝕜 : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : RCLike 𝕜
    f : κ → ι → 𝕜
    hf : ∀ (k : κ), Ne (f k) 0
    hinner : Pairwise fun k₁ k₂ => Eq (RCLike.wInner 1 (f k₁) (f k₂)) 0
    ⊢ LinearIndependent 𝕜 f
  -/
  simp_rw [wInner_one_eq_inner] at hinner
  /-
    ι : Type u_1
    κ : Type u_2
    𝕜 : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : RCLike 𝕜
    f : κ → ι → 𝕜
    hf : ∀ (k : κ), Ne (f k) 0
    hinner : Pairwise fun k₁ k₂ => Eq (Inner.inner ((WithLp.equiv 2 (ι → 𝕜)).symm  …
    ⊢ LinearIndependent 𝕜 f
  -/
  have := linearIndependent_of_ne_zero_of_inner_eq_zero ?_ hinner
  /-
    case refine_2
    ι : Type u_1
    κ : Type u_2
    𝕜 : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : RCLike 𝕜
    f : κ → ι → 𝕜
    hf : ∀ (k : κ), Ne (f k) 0
    hinner : Pairwise fun k₁ k₂ => Eq (Inner.inner ((WithLp.equiv 2 (ι → 𝕜)).symm  …
    this : LinearIndependent 𝕜 fun i => (WithLp.equiv 2 (ι → 𝕜)).symm (f i)
    ⊢ LinearIndependent 𝕜 f
  -/
  exacts [this, hf]
  /-
    🎉 no goals
  -/


lemma linearIndependent_of_ne_zero_of_wInner_cWeight_eq_zero {f : κ → ι → 𝕜} (hf : ∀ k, f k ≠ 0)
    (hinner : Pairwise fun k₁ k₂ ↦ ⟪f k₁, f k₂⟫ₙ_[𝕜] = 0) : LinearIndependent 𝕜 f := by
  /-
    ι : Type u_1
    κ : Type u_2
    𝕜 : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : RCLike 𝕜
    f : κ → ι → 𝕜
    hf : ∀ (k : κ), Ne (f k) 0
    hinner : Pairwise fun k₁ k₂ => Eq (RCLike.wInner RCLike.cWeight (f k₁) (f k₂)) 0
    ⊢ LinearIndependent 𝕜 f
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Type u_1
      κ : Type u_2
      𝕜 : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : RCLike 𝕜
      f : κ → ι → 𝕜
      hf : ∀ (k : κ), Ne (f k) 0
      hinner : Pairwise fun k₁ k₂ => Eq (RCLike.wInner RCLike.cWeight (f k₁) (f k₂)) 0
      h✝ : IsEmpty ι
      ⊢ LinearIndependent 𝕜 f
    -/
  · have : IsEmpty κ := ⟨fun k ↦ hf k <| Subsingleton.elim ..⟩
    /-
      case inl
      ι : Type u_1
      κ : Type u_2
      𝕜 : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : RCLike 𝕜
      f : κ → ι → 𝕜
      hf : ∀ (k : κ), Ne (f k) 0
      hinner : Pairwise fun k₁ k₂ => Eq (RCLike.wInner RCLike.cWeight (f k₁) (f k₂)) 0
      h✝ : IsEmpty ι
      this : IsEmpty κ
      ⊢ LinearIndependent 𝕜 f
    -/
    exact linearIndependent_empty_type
    /-
      🎉 no goals
    -/
  · exact linearIndependent_of_ne_zero_of_wInner_one_eq_zero hf <| by
      simpa [wInner_cWeight_eq_smul_wInner_one, ← NNRat.cast_smul_eq_nnqsmul 𝕜] using hinner


lemma wInner_nonneg (hw : 0 ≤ w) (hf : 0 ≤ f) (hg : 0 ≤ g) : 0 ≤ ⟪f, g⟫_[𝕜, w] :=
  sum_nonneg fun _ _ ↦ smul_nonneg (hw _) <| mul_nonneg (star_nonneg_iff.2 (hf _)) (hg _)


lemma norm_wInner_le (hw : 0 ≤ w) : ‖⟪f, g⟫_[𝕜, w]‖ ≤ ⟪fun i ↦ ‖f i‖, fun i ↦ ‖g i‖⟫_[ℝ, w] :=
  (norm_sum_le ..).trans_eq <| sum_congr rfl fun i _ ↦ by
    /-
      ι : Type u_1
      𝕜 : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : RCLike 𝕜
      w : ι → Real
      f g : ι → 𝕜
      hw : LE.le 0 w
      i : ι
      x✝ : Membership.mem Finset.univ i
      ⊢ Eq (Norm.norm (HSMul.hSMul (w i) (Inner.inner (f i) (g i)))) (HSMul.hSMul (w …
    -/
    simp [Algebra.smul_def, norm_mul, abs_of_nonneg (hw i)]
    /-
      🎉 no goals
    -/


lemma abs_wInner_le (hw : 0 ≤ w) : |⟪f, g⟫_[ℝ, w]| ≤ ⟪|f|, |g|⟫_[ℝ, w] := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    w f g : ι → Real
    hw : LE.le 0 w
    ⊢ LE.le (abs (RCLike.wInner w f g)) (RCLike.wInner w (abs f) (abs g))
  -/
  simpa using norm_wInner_le (𝕜 := ℝ) hw
  /-
    🎉 no goals
  -/


