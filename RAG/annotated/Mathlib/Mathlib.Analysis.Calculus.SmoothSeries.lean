/-- Consider a series of functions `∑' n, f n x` on a preconnected open set. If the series converges
at a point, and all functions in the series are differentiable with a summable bound on the
derivatives, then the series converges everywhere on the set. -/
theorem summable_of_summable_hasFDerivAt_of_isPreconnected (hu : Summable u) (hs : IsOpen s)
    (h's : IsPreconnected s) (hf : ∀ n x, x ∈ s → HasFDerivAt (f n) (f' n x) x)
    (hf' : ∀ n x, x ∈ s → ‖f' n x‖ ≤ u n) (hx₀ : x₀ ∈ s) (hf0 : Summable (f · x₀))
    (hx : x ∈ s) : Summable fun n => f n x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set E
    x₀ x : E
    hu : Summable u
    hs : IsOpen s
    h's : IsPreconnected s
    hf : ∀ (n : α) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), Membership.mem s x → LE.le (Norm.norm (f' n x)) (u n)
    hx₀ : Membership.mem s x₀
    hf0 : Summable fun x => f x x₀
    hx : Membership.mem s x
    ⊢ Summable fun n => f n x
  -/
  haveI := Classical.decEq α
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set E
    x₀ x : E
    hu : Summable u
    hs : IsOpen s
    h's : IsPreconnected s
    hf : ∀ (n : α) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), Membership.mem s x → LE.le (Norm.norm (f' n x)) (u n)
    hx₀ : Membership.mem s x₀
    hf0 : Summable fun x => f x x₀
    hx : Membership.mem s x
    this : DecidableEq α
    ⊢ Summable fun n => f n x
  -/
  rw [summable_iff_cauchySeq_finset] at hf0 ⊢
  have A : UniformCauchySeqOn (fun t : Finset α => fun x => ∑ i ∈ t, f' i x) atTop s :=
    (tendstoUniformlyOn_tsum hu hf').uniformCauchySeqOn
  -- Porting note: Lean 4 failed to find `f` by unification
  refine cauchy_map_of_uniformCauchySeqOn_fderiv (f := fun t x ↦ ∑ i ∈ t, f i x)
    hs h's A (fun t y hy => ?_) hx₀ hx hf0
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set E
    x₀ x : E
    hu : Summable u
    hs : IsOpen s
    h's : IsPreconnected s
    hf : ∀ (n : α) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), Membership.mem s x → LE.le (Norm.norm (f' n x)) (u n)
    hx₀ : Membership.mem s x₀
    hf0 : CauchySeq fun s => s.sum fun b => f b x₀
    hx : Membership.mem s x
    this : DecidableEq α
    A : UniformCauchySeqOn (fun t x => t.sum fun i => f' i x) Filter.atTop s
    t : Finset α
    y : E
    hy : Membership.mem s y
    ⊢ HasFDerivAt ((fun t x => t.sum fun i => f i x) t) (t.sum fun i => f' i y) y
  -/
  exact HasFDerivAt.sum fun i _ => hf i y hy
  /-
    🎉 no goals
  -/


/-- Consider a series of functions `∑' n, f n x` on a preconnected open set. If the series converges
at a point, and all functions in the series are differentiable with a summable bound on the
derivatives, then the series converges everywhere on the set. -/
theorem summable_of_summable_hasDerivAt_of_isPreconnected (hu : Summable u) (ht : IsOpen t)
    (h't : IsPreconnected t) (hg : ∀ n y, y ∈ t → HasDerivAt (g n) (g' n y) y)
    (hg' : ∀ n y, y ∈ t → ‖g' n y‖ ≤ u n) (hy₀ : y₀ ∈ t) (hg0 : Summable (g · y₀))
    (hy : y ∈ t) : Summable fun n => g n y := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    t : Set 𝕜
    y₀ y : 𝕜
    hu : Summable u
    ht : IsOpen t
    h't : IsPreconnected t
    hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasDerivAt (g n) (g' n y) y
    hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
    hy₀ : Membership.mem t y₀
    hg0 : Summable fun x => g x y₀
    hy : Membership.mem t y
    ⊢ Summable fun n => g n y
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hg
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    t : Set 𝕜
    y₀ y : 𝕜
    hu : Summable u
    ht : IsOpen t
    h't : IsPreconnected t
    hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
    hy₀ : Membership.mem t y₀
    hg0 : Summable fun x => g x y₀
    hy : Membership.mem t y
    hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasFDerivAt (g n) (ContinuousLine …
    ⊢ Summable fun n => g n y
  -/
  refine summable_of_summable_hasFDerivAt_of_isPreconnected hu ht h't hg ?_ hy₀ hg0 hy
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    t : Set 𝕜
    y₀ y : 𝕜
    hu : Summable u
    ht : IsOpen t
    h't : IsPreconnected t
    hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
    hy₀ : Membership.mem t y₀
    hg0 : Summable fun x => g x y₀
    hy : Membership.mem t y
    hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasFDerivAt (g n) (ContinuousLine …
    ⊢ ∀ (n : α) (x : 𝕜), Membership.mem t x → LE.le (Norm.norm (ContinuousLinearMa …
  -/
  simpa? says simpa only [ContinuousLinearMap.norm_smulRight_apply, norm_one, one_mul]
  /-
    🎉 no goals
  -/


/-- Consider a series of functions `∑' n, f n x` on a preconnected open set. If the series converges
at a point, and all functions in the series are differentiable with a summable bound on the
derivatives, then the series is differentiable on the set and its derivative is the sum of the
derivatives. -/
theorem hasFDerivAt_tsum_of_isPreconnected (hu : Summable u) (hs : IsOpen s)
    (h's : IsPreconnected s) (hf : ∀ n x, x ∈ s → HasFDerivAt (f n) (f' n x) x)
    (hf' : ∀ n x, x ∈ s → ‖f' n x‖ ≤ u n) (hx₀ : x₀ ∈ s) (hf0 : Summable fun n => f n x₀)
    (hx : x ∈ s) : HasFDerivAt (fun y => ∑' n, f n y) (∑' n, f' n x) x := by
  classical
    have A :
      ∀ x : E, x ∈ s → Tendsto (fun t : Finset α => ∑ n ∈ t, f n x) atTop (𝓝 (∑' n, f n x)) := by
      intro y hy
      apply Summable.hasSum
      exact summable_of_summable_hasFDerivAt_of_isPreconnected hu hs h's hf hf' hx₀ hf0 hy
    refine hasFDerivAt_of_tendstoUniformlyOn hs (tendstoUniformlyOn_tsum hu hf')
      (fun t y hy => ?_) A hx
    exact HasFDerivAt.sum fun n _ => hf n y hy


/-- Consider a series of functions `∑' n, f n x` on a preconnected open set. If the series converges
at a point, and all functions in the series are differentiable with a summable bound on the
derivatives, then the series is differentiable on the set and its derivative is the sum of the
derivatives. -/
theorem hasDerivAt_tsum_of_isPreconnected (hu : Summable u) (ht : IsOpen t)
    (h't : IsPreconnected t) (hg : ∀ n y, y ∈ t → HasDerivAt (g n) (g' n y) y)
    (hg' : ∀ n y, y ∈ t → ‖g' n y‖ ≤ u n) (hy₀ : y₀ ∈ t) (hg0 : Summable fun n => g n y₀)
    (hy : y ∈ t) : HasDerivAt (fun z => ∑' n, g n z) (∑' n, g' n y) y := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    t : Set 𝕜
    y₀ y : 𝕜
    hu : Summable u
    ht : IsOpen t
    h't : IsPreconnected t
    hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasDerivAt (g n) (g' n y) y
    hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
    hy₀ : Membership.mem t y₀
    hg0 : Summable fun n => g n y₀
    hy : Membership.mem t y
    ⊢ HasDerivAt (fun z => tsum fun n => g n z) (tsum fun n => g' n y) y
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hg ⊢
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    t : Set 𝕜
    y₀ y : 𝕜
    hu : Summable u
    ht : IsOpen t
    h't : IsPreconnected t
    hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
    hy₀ : Membership.mem t y₀
    hg0 : Summable fun n => g n y₀
    hy : Membership.mem t y
    hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasFDerivAt (g n) (ContinuousLine …
    ⊢ HasFDerivAt (fun z => tsum fun n => g n z) (ContinuousLinearMap.smulRight 1  …
  -/
  convert hasFDerivAt_tsum_of_isPreconnected hu ht h't hg ?_ hy₀ hg0 hy
  · exact (ContinuousLinearMap.smulRightL 𝕜 𝕜 F 1).map_tsum <|
      .of_norm_bounded u hu fun n ↦ hg' n y hy
    /-
      α : Type u_1
      𝕜 : Type u_3
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      g g' : α → 𝕜 → F
      t : Set 𝕜
      y₀ y : 𝕜
      hu : Summable u
      ht : IsOpen t
      h't : IsPreconnected t
      hg' : ∀ (n : α) (y : 𝕜), Membership.mem t y → LE.le (Norm.norm (g' n y)) (u n)
      hy₀ : Membership.mem t y₀
      hg0 : Summable fun n => g n y₀
      hy : Membership.mem t y
      hg : ∀ (n : α) (y : 𝕜), Membership.mem t y → HasFDerivAt (g n) (ContinuousLine …
      ⊢ ∀ (n : α) (x : 𝕜), Membership.mem t x → LE.le (Norm.norm (ContinuousLinearMa …
    -/
  · simpa? says simpa only [ContinuousLinearMap.norm_smulRight_apply, norm_one, one_mul]
    /-
      🎉 no goals
    -/


/-- Consider a series of functions `∑' n, f n x`. If the series converges at a
point, and all functions in the series are differentiable with a summable bound on the derivatives,
then the series converges everywhere. -/
theorem summable_of_summable_hasFDerivAt (hu : Summable u)
    (hf : ∀ n x, HasFDerivAt (f n) (f' n x) x) (hf' : ∀ n x, ‖f' n x‖ ≤ u n)
    (hf0 : Summable fun n => f n x₀) (x : E) : Summable fun n => f n x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
    hf0 : Summable fun n => f n x₀
    x : E
    ⊢ Summable fun n => f n x
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
    hf0 : Summable fun n => f n x₀
    x : E
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Summable fun n => f n x
  -/
  let _ : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 _
  exact summable_of_summable_hasFDerivAt_of_isPreconnected hu isOpen_univ isPreconnected_univ
    (fun n x _ => hf n x) (fun n x _ => hf' n x) (mem_univ _) hf0 (mem_univ _)


/-- Consider a series of functions `∑' n, f n x`. If the series converges at a
point, and all functions in the series are differentiable with a summable bound on the derivatives,
then the series converges everywhere. -/
theorem summable_of_summable_hasDerivAt (hu : Summable u)
    (hg : ∀ n y, HasDerivAt (g n) (g' n y) y) (hg' : ∀ n y, ‖g' n y‖ ≤ u n)
    (hg0 : Summable fun n => g n y₀) (y : 𝕜) : Summable fun n => g n y := by
  exact summable_of_summable_hasDerivAt_of_isPreconnected hu isOpen_univ isPreconnected_univ
    (fun n x _ => hg n x) (fun n x _ => hg' n x) (mem_univ _) hg0 (mem_univ _)


/-- Consider a series of functions `∑' n, f n x`. If the series converges at a
point, and all functions in the series are differentiable with a summable bound on the derivatives,
then the series is differentiable and its derivative is the sum of the derivatives. -/
theorem hasFDerivAt_tsum (hu : Summable u) (hf : ∀ n x, HasFDerivAt (f n) (f' n x) x)
    (hf' : ∀ n x, ‖f' n x‖ ≤ u n) (hf0 : Summable fun n => f n x₀) (x : E) :
    HasFDerivAt (fun y => ∑' n, f n y) (∑' n, f' n x) x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
    hf0 : Summable fun n => f n x₀
    x : E
    ⊢ HasFDerivAt (fun y => tsum fun n => f n y) (tsum fun n => f' n x) x
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
    hf0 : Summable fun n => f n x₀
    x : E
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ HasFDerivAt (fun y => tsum fun n => f n y) (tsum fun n => f' n x) x
  -/
  let A : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 _
  exact hasFDerivAt_tsum_of_isPreconnected hu isOpen_univ isPreconnected_univ
    (fun n x _ => hf n x) (fun n x _ => hf' n x) (mem_univ _) hf0 (mem_univ _)


/-- Consider a series of functions `∑' n, f n x`. If the series converges at a
point, and all functions in the series are differentiable with a summable bound on the derivatives,
then the series is differentiable and its derivative is the sum of the derivatives. -/
theorem hasDerivAt_tsum (hu : Summable u) (hg : ∀ n y, HasDerivAt (g n) (g' n y) y)
    (hg' : ∀ n y, ‖g' n y‖ ≤ u n) (hg0 : Summable fun n => g n y₀) (y : 𝕜) :
    HasDerivAt (fun z => ∑' n, g n z) (∑' n, g' n y) y := by
  exact hasDerivAt_tsum_of_isPreconnected hu isOpen_univ isPreconnected_univ
    (fun n y _ => hg n y) (fun n y _ => hg' n y) (mem_univ _) hg0 (mem_univ _)


/-- Consider a series of functions `∑' n, f n x`. If all functions in the series are differentiable
with a summable bound on the derivatives, then the series is differentiable.
Note that our assumptions do not ensure the pointwise convergence, but if there is no pointwise
convergence then the series is zero everywhere so the result still holds. -/
theorem differentiable_tsum (hu : Summable u) (hf : ∀ n x, HasFDerivAt (f n) (f' n x) x)
    (hf' : ∀ n x, ‖f' n x‖ ≤ u n) : Differentiable 𝕜 fun y => ∑' n, f n y := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
    hu : Summable u
    hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
    ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
  -/
  by_cases h : ∃ x₀, Summable fun n => f n x₀
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      h : Exists fun x₀ => Summable fun n => f n x₀
      ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
    -/
  · rcases h with ⟨x₀, hf0⟩
    /-
      case pos.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      x₀ : E
      hf0 : Summable fun n => f n x₀
      ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
    -/
    intro x
    /-
      case pos.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      x₀ : E
      hf0 : Summable fun n => f n x₀
      x : E
      ⊢ DifferentiableAt 𝕜 (fun y => tsum fun n => f n y) x
    -/
    exact (hasFDerivAt_tsum hu hf hf' hf0 x).differentiableAt
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      h : Not (Exists fun x₀ => Summable fun n => f n x₀)
      ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
    -/
  · push_neg at h
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      h : ∀ (x₀ : E), Not (Summable fun n => f n x₀)
      ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
    -/
    have : (fun x => ∑' n, f n x) = 0 := by ext1 x; exact tsum_eq_zero_of_not_summable (h x)
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      h : ∀ (x₀ : E), Not (Summable fun n => f n x₀)
      this : Eq (fun x => tsum fun n => f n x) 0
      ⊢ Differentiable 𝕜 fun y => tsum fun n => f n y
    -/
    rw [this]
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      u : α → Real
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      f' : α → E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : Summable u
      hf : ∀ (n : α) (x : E), HasFDerivAt (f n) (f' n x) x
      hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (f' n x)) (u n)
      h : ∀ (x₀ : E), Not (Summable fun n => f n x₀)
      this : Eq (fun x => tsum fun n => f n x) 0
      ⊢ Differentiable 𝕜 0
    -/
    exact differentiable_const 0
    /-
      🎉 no goals
    -/


/-- Consider a series of functions `∑' n, f n x`. If all functions in the series are differentiable
with a summable bound on the derivatives, then the series is differentiable.
Note that our assumptions do not ensure the pointwise convergence, but if there is no pointwise
convergence then the series is zero everywhere so the result still holds. -/
theorem differentiable_tsum' (hu : Summable u) (hg : ∀ n y, HasDerivAt (g n) (g' n y) y)
    (hg' : ∀ n y, ‖g' n y‖ ≤ u n) : Differentiable 𝕜 fun z => ∑' n, g n z := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    hu : Summable u
    hg : ∀ (n : α) (y : 𝕜), HasDerivAt (g n) (g' n y) y
    hg' : ∀ (n : α) (y : 𝕜), LE.le (Norm.norm (g' n y)) (u n)
    ⊢ Differentiable 𝕜 fun z => tsum fun n => g n z
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hg
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    hu : Summable u
    hg' : ∀ (n : α) (y : 𝕜), LE.le (Norm.norm (g' n y)) (u n)
    hg : ∀ (n : α) (y : 𝕜), HasFDerivAt (g n) (ContinuousLinearMap.smulRight 1 (g' …
    ⊢ Differentiable 𝕜 fun z => tsum fun n => g n z
  -/
  refine differentiable_tsum hu hg ?_
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g g' : α → 𝕜 → F
    hu : Summable u
    hg' : ∀ (n : α) (y : 𝕜), LE.le (Norm.norm (g' n y)) (u n)
    hg : ∀ (n : α) (y : 𝕜), HasFDerivAt (g n) (ContinuousLinearMap.smulRight 1 (g' …
    ⊢ ∀ (n : α) (x : 𝕜), LE.le (Norm.norm (ContinuousLinearMap.smulRight 1 (g' n x …
  -/
  simpa? says simpa only [ContinuousLinearMap.norm_smulRight_apply, norm_one, one_mul]
  /-
    🎉 no goals
  -/


theorem fderiv_tsum_apply (hu : Summable u) (hf : ∀ n, Differentiable 𝕜 (f n))
    (hf' : ∀ n x, ‖fderiv 𝕜 (f n) x‖ ≤ u n) (hf0 : Summable fun n => f n x₀) (x : E) :
    fderiv 𝕜 (fun y => ∑' n, f n y) x = ∑' n, fderiv 𝕜 (f n) x :=
  (hasFDerivAt_tsum hu (fun n x => (hf n x).hasFDerivAt) hf' hf0 _).fderiv


theorem deriv_tsum_apply (hu : Summable u) (hg : ∀ n, Differentiable 𝕜 (g n))
    (hg' : ∀ n y, ‖deriv (g n) y‖ ≤ u n) (hg0 : Summable fun n => g n y₀) (y : 𝕜) :
    deriv (fun z => ∑' n, g n z) y = ∑' n, deriv (g n) y :=
  (hasDerivAt_tsum hu (fun n y => (hg n y).hasDerivAt) hg' hg0 _).deriv


theorem fderiv_tsum (hu : Summable u) (hf : ∀ n, Differentiable 𝕜 (f n))
    (hf' : ∀ n x, ‖fderiv 𝕜 (f n) x‖ ≤ u n) (hf0 : Summable fun n => f n x₀) :
    (fderiv 𝕜 fun y => ∑' n, f n y) = fun x => ∑' n, fderiv 𝕜 (f n) x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α), Differentiable 𝕜 (f n)
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (fderiv 𝕜 (f n) x)) (u n)
    hf0 : Summable fun n => f n x₀
    ⊢ Eq (fderiv 𝕜 fun y => tsum fun n => f n y) fun x => tsum fun n => fderiv 𝕜 ( …
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    x₀ : E
    hu : Summable u
    hf : ∀ (n : α), Differentiable 𝕜 (f n)
    hf' : ∀ (n : α) (x : E), LE.le (Norm.norm (fderiv 𝕜 (f n) x)) (u n)
    hf0 : Summable fun n => f n x₀
    x : E
    ⊢ Eq (fderiv 𝕜 (fun y => tsum fun n => f n y) x) (tsum fun n => fderiv 𝕜 (f n) …
  -/
  exact fderiv_tsum_apply hu hf hf' hf0 x
  /-
    🎉 no goals
  -/


theorem deriv_tsum (hu : Summable u) (hg : ∀ n, Differentiable 𝕜 (g n))
    (hg' : ∀ n y, ‖deriv (g n) y‖ ≤ u n) (hg0 : Summable fun n => g n y₀) :
    (deriv fun y => ∑' n, g n y) = fun y => ∑' n, deriv (g n) y := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g : α → 𝕜 → F
    y₀ : 𝕜
    hu : Summable u
    hg : ∀ (n : α), Differentiable 𝕜 (g n)
    hg' : ∀ (n : α) (y : 𝕜), LE.le (Norm.norm (deriv (g n) y)) (u n)
    hg0 : Summable fun n => g n y₀
    ⊢ Eq (deriv fun y => tsum fun n => g n y) fun y => tsum fun n => deriv (g n) y
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : NormedSpace 𝕜 F
    g : α → 𝕜 → F
    y₀ : 𝕜
    hu : Summable u
    hg : ∀ (n : α), Differentiable 𝕜 (g n)
    hg' : ∀ (n : α) (y : 𝕜), LE.le (Norm.norm (deriv (g n) y)) (u n)
    hg0 : Summable fun n => g n y₀
    x : 𝕜
    ⊢ Eq (deriv (fun y => tsum fun n => g n y) x) (tsum fun n => deriv (g n) x)
  -/
  exact deriv_tsum_apply hu hg hg' hg0 x
  /-
    🎉 no goals
  -/


/-- Consider a series of smooth functions, with summable uniform bounds on the successive
derivatives. Then the iterated derivative of the sum is the sum of the iterated derivative. -/
theorem iteratedFDeriv_tsum (hf : ∀ i, ContDiff 𝕜 N (f i))
    (hv : ∀ k : ℕ, (k : ℕ∞) ≤ N → Summable (v k))
    (h'f : ∀ (k : ℕ) (i : α) (x : E), (k : ℕ∞) ≤ N → ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i) {k : ℕ}
    (hk : (k : ℕ∞) ≤ N) :
    (iteratedFDeriv 𝕜 k fun y => ∑' n, f n y) = fun x => ∑' n, iteratedFDeriv 𝕜 k (f n) x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    v : Nat → α → Real
    N : ENat
    hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
    hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
    h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
    k : Nat
    hk : LE.le (↑k) N
    ⊢ Eq (iteratedFDeriv 𝕜 k fun y => tsum fun n => f n y) fun x => tsum fun n =>  …
  -/
  induction' k with k IH
    /-
      case zero
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      hk : LE.le (↑0) N
      ⊢ Eq (iteratedFDeriv 𝕜 0 fun y => tsum fun n => f n y) fun x => tsum fun n =>  …
    -/
  · ext1 x
    /-
      case zero.h
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      hk : LE.le (↑0) N
      x : E
      ⊢ Eq (iteratedFDeriv 𝕜 0 (fun y => tsum fun n => f n y) x) (tsum fun n => iter …
    -/
    simp_rw [iteratedFDeriv_zero_eq_comp]
    /-
      case zero.h
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      hk : LE.le (↑0) N
      x : E
      ⊢ Eq (Function.comp (⇑(continuousMultilinearCurryFin0 𝕜 E F).symm) (fun y => t …
    -/
    exact (continuousMultilinearCurryFin0 𝕜 E F).symm.toContinuousLinearEquiv.map_tsum
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      k : Nat
      IH : LE.le (↑k) N → Eq (iteratedFDeriv 𝕜 k fun y => tsum fun n => f n y) fun x …
      hk : LE.le (↑(HAdd.hAdd k 1)) N
      ⊢ Eq (iteratedFDeriv 𝕜 (HAdd.hAdd k 1) fun y => tsum fun n => f n y) fun x =>  …
    -/
  · have h'k : (k : ℕ∞) < N := lt_of_lt_of_le (WithTop.coe_lt_coe.2 (Nat.lt_succ_self _)) hk
    have A : Summable fun n => iteratedFDeriv 𝕜 k (f n) 0 :=
      .of_norm_bounded (v k) (hv k h'k.le) fun n => h'f k n 0 h'k.le
    /-
      case succ
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      k : Nat
      IH : LE.le (↑k) N → Eq (iteratedFDeriv 𝕜 k fun y => tsum fun n => f n y) fun x …
      hk : LE.le (↑(HAdd.hAdd k 1)) N
      h'k : LT.lt (↑k) N
      A : Summable fun n => iteratedFDeriv 𝕜 k (f n) 0
      ⊢ Eq (iteratedFDeriv 𝕜 (HAdd.hAdd k 1) fun y => tsum fun n => f n y) fun x =>  …
    -/
    simp_rw [iteratedFDeriv_succ_eq_comp_left, IH h'k.le]
    rw [fderiv_tsum (hv _ hk) (fun n => (hf n).differentiable_iteratedFDeriv
        (mod_cast h'k)) _ A]
      /-
        case succ
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        k : Nat
        IH : LE.le (↑k) N → Eq (iteratedFDeriv 𝕜 k fun y => tsum fun n => f n y) fun x …
        hk : LE.le (↑(HAdd.hAdd k 1)) N
        h'k : LT.lt (↑k) N
        A : Summable fun n => iteratedFDeriv 𝕜 k (f n) 0
        ⊢ Eq (Function.comp ⇑(continuousMultilinearCurryLeftEquiv 𝕜 (fun x => E) F).sy …
      -/
    · ext1 x
      exact (continuousMultilinearCurryLeftEquiv 𝕜
        (fun _ : Fin (k + 1) => E) F).symm.toContinuousLinearEquiv.map_tsum
      /-
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        k : Nat
        IH : LE.le (↑k) N → Eq (iteratedFDeriv 𝕜 k fun y => tsum fun n => f n y) fun x …
        hk : LE.le (↑(HAdd.hAdd k 1)) N
        h'k : LT.lt (↑k) N
        A : Summable fun n => iteratedFDeriv 𝕜 k (f n) 0
        ⊢ ∀ (n : α) (x : E), LE.le (Norm.norm (fderiv 𝕜 (fun x => iteratedFDeriv 𝕜 k ( …
      -/
    · intro n x
      simpa only [iteratedFDeriv_succ_eq_comp_left, LinearIsometryEquiv.norm_map, comp_apply]
        using h'f k.succ n x hk


/-- Consider a series of smooth functions, with summable uniform bounds on the successive
derivatives. Then the iterated derivative of the sum is the sum of the iterated derivative. -/
theorem iteratedFDeriv_tsum_apply (hf : ∀ i, ContDiff 𝕜 N (f i))
    (hv : ∀ k : ℕ, (k : ℕ∞) ≤ N → Summable (v k))
    (h'f : ∀ (k : ℕ) (i : α) (x : E), (k : ℕ∞) ≤ N → ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i) {k : ℕ}
    (hk : (k : ℕ∞) ≤ N) (x : E) :
    iteratedFDeriv 𝕜 k (fun y => ∑' n, f n y) x = ∑' n, iteratedFDeriv 𝕜 k (f n) x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    v : Nat → α → Real
    N : ENat
    hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
    hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
    h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
    k : Nat
    hk : LE.le (↑k) N
    x : E
    ⊢ Eq (iteratedFDeriv 𝕜 k (fun y => tsum fun n => f n y) x) (tsum fun n => iter …
  -/
  rw [iteratedFDeriv_tsum hf hv h'f hk]
  /-
    🎉 no goals
  -/


/-- Consider a series of functions `∑' i, f i x`. Assume that each individual function `f i` is of
class `C^N`, and moreover there is a uniform summable upper bound on the `k`-th derivative
for each `k ≤ N`. Then the series is also `C^N`. -/
theorem contDiff_tsum (hf : ∀ i, ContDiff 𝕜 N (f i)) (hv : ∀ k : ℕ, (k : ℕ∞) ≤ N → Summable (v k))
    (h'f : ∀ (k : ℕ) (i : α) (x : E), k ≤ N → ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i) :
    ContDiff 𝕜 N fun x => ∑' i, f i x := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    v : Nat → α → Real
    N : ENat
    hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
    hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
    h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
    ⊢ ContDiff 𝕜 ↑N fun x => tsum fun i => f i x
  -/
  rw [contDiff_iff_continuous_differentiable]
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    v : Nat → α → Real
    N : ENat
    hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
    hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
    h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
    ⊢ And (∀ (m : Nat), LE.le (↑m) N → Continuous fun x => iteratedFDeriv 𝕜 m (fun …
  -/
  constructor
    /-
      case left
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      ⊢ ∀ (m : Nat), LE.le (↑m) N → Continuous fun x => iteratedFDeriv 𝕜 m (fun x => …
    -/
  · intro m hm
    /-
      case left
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LE.le (↑m) N
      ⊢ Continuous fun x => iteratedFDeriv 𝕜 m (fun x => tsum fun i => f i x) x
    -/
    rw [iteratedFDeriv_tsum hf hv h'f hm]
    /-
      case left
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LE.le (↑m) N
      ⊢ Continuous fun x => (fun x => tsum fun n => iteratedFDeriv 𝕜 m (f n) x) x
    -/
    refine continuous_tsum ?_ (hv m hm) ?_
      /-
        case left.refine_1
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        m : Nat
        hm : LE.le (↑m) N
        ⊢ ∀ (i : α), Continuous (iteratedFDeriv 𝕜 m (f i))
      -/
    · intro i
      /-
        case left.refine_1
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        m : Nat
        hm : LE.le (↑m) N
        i : α
        ⊢ Continuous (iteratedFDeriv 𝕜 m (f i))
      -/
      exact ContDiff.continuous_iteratedFDeriv (mod_cast hm) (hf i)
      /-
        🎉 no goals
      -/
      /-
        case left.refine_2
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        m : Nat
        hm : LE.le (↑m) N
        ⊢ ∀ (n : α) (x : E), LE.le (Norm.norm (iteratedFDeriv 𝕜 m (f n) x)) (v m n)
      -/
    · intro n x
      /-
        case left.refine_2
        α : Type u_1
        𝕜 : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : IsRCLikeNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : CompleteSpace F
        inst✝ : NormedSpace 𝕜 F
        f : α → E → F
        v : Nat → α → Real
        N : ENat
        hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
        hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
        h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
        m : Nat
        hm : LE.le (↑m) N
        n : α
        x : E
        ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 m (f n) x)) (v m n)
      -/
      exact h'f _ _ _ hm
      /-
        🎉 no goals
      -/
    /-
      case right
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      ⊢ ∀ (m : Nat), LT.lt (↑m) N → Differentiable 𝕜 fun x => iteratedFDeriv 𝕜 m (fu …
    -/
  · intro m hm
    have h'm : ((m + 1 : ℕ) : ℕ∞) ≤ N := by
      simpa only [ENat.coe_add, ENat.coe_one] using Order.add_one_le_of_lt hm
    /-
      case right
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LT.lt (↑m) N
      h'm : LE.le (↑(HAdd.hAdd m 1)) N
      ⊢ Differentiable 𝕜 fun x => iteratedFDeriv 𝕜 m (fun x => tsum fun i => f i x) x
    -/
    rw [iteratedFDeriv_tsum hf hv h'f hm.le]
    have A n x : HasFDerivAt (iteratedFDeriv 𝕜 m (f n)) (fderiv 𝕜 (iteratedFDeriv 𝕜 m (f n)) x) x :=
      (ContDiff.differentiable_iteratedFDeriv (mod_cast hm)
        (hf n)).differentiableAt.hasFDerivAt
    /-
      case right
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LT.lt (↑m) N
      h'm : LE.le (↑(HAdd.hAdd m 1)) N
      A : ∀ (n : α) (x : E), HasFDerivAt (iteratedFDeriv 𝕜 m (f n)) (fderiv 𝕜 (itera …
      ⊢ Differentiable 𝕜 fun x => (fun x => tsum fun n => iteratedFDeriv 𝕜 m (f n) x …
    -/
    refine differentiable_tsum (hv _ h'm) A fun n x => ?_
    /-
      case right
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LT.lt (↑m) N
      h'm : LE.le (↑(HAdd.hAdd m 1)) N
      A : ∀ (n : α) (x : E), HasFDerivAt (iteratedFDeriv 𝕜 m (f n)) (fderiv 𝕜 (itera …
      n : α
      x : E
      ⊢ LE.le (Norm.norm (fderiv 𝕜 (iteratedFDeriv 𝕜 m (f n)) x)) (v (HAdd.hAdd m 1) …
    -/
    rw [fderiv_iteratedFDeriv, comp_apply, LinearIsometryEquiv.norm_map]
    /-
      case right
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : IsRCLikeNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : CompleteSpace F
      inst✝ : NormedSpace 𝕜 F
      f : α → E → F
      v : Nat → α → Real
      N : ENat
      hf : ∀ (i : α), ContDiff 𝕜 (↑N) (f i)
      hv : ∀ (k : Nat), LE.le (↑k) N → Summable (v k)
      h'f : ∀ (k : Nat) (i : α) (x : E), LE.le (↑k) N → LE.le (Norm.norm (iteratedFD …
      m : Nat
      hm : LT.lt (↑m) N
      h'm : LE.le (↑(HAdd.hAdd m 1)) N
      A : ∀ (n : α) (x : E), HasFDerivAt (iteratedFDeriv 𝕜 m (f n)) (fderiv 𝕜 (itera …
      n : α
      x : E
      ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 (HAdd.hAdd m 1) (f n) x)) (v (HAdd.hAdd m …
    -/
    exact h'f _ _ _ h'm
    /-
      🎉 no goals
    -/


/-- Consider a series of functions `∑' i, f i x`. Assume that each individual function `f i` is of
class `C^N`, and moreover there is a uniform summable upper bound on the `k`-th derivative
for each `k ≤ N` (except maybe for finitely many `i`s). Then the series is also `C^N`. -/
theorem contDiff_tsum_of_eventually (hf : ∀ i, ContDiff 𝕜 N (f i))
    (hv : ∀ k : ℕ, k ≤ N → Summable (v k))
    (h'f : ∀ k : ℕ, k ≤ N →
      ∀ᶠ i in (Filter.cofinite : Filter α), ∀ x : E, ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i) :
    ContDiff 𝕜 N fun x => ∑' i, f i x := by
  classical
    refine contDiff_iff_forall_nat_le.2 fun m hm => ?_
    let t : Set α :=
      { i : α | ¬∀ k : ℕ, k ∈ Finset.range (m + 1) → ∀ x, ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i }
    have ht : Set.Finite t :=
      haveI A :
        ∀ᶠ i in (Filter.cofinite : Filter α),
          ∀ k : ℕ, k ∈ Finset.range (m + 1) → ∀ x : E, ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i := by
        rw [eventually_all_finset]
        intro i hi
        apply h'f
        simp only [Finset.mem_range_succ_iff] at hi
        exact (WithTop.coe_le_coe.2 hi).trans hm
      eventually_cofinite.2 A
    let T : Finset α := ht.toFinset
    have : (fun x => ∑' i, f i x) = (fun x => ∑ i ∈ T, f i x) +
        fun x => ∑' i : { i // i ∉ T }, f i x := by
      ext1 x
      refine (sum_add_tsum_subtype_compl ?_ T).symm
      refine .of_norm_bounded_eventually _ (hv 0 (zero_le _)) ?_
      filter_upwards [h'f 0 (zero_le _)] with i hi
      simpa only [norm_iteratedFDeriv_zero] using hi x
    rw [this]
    apply (ContDiff.sum fun i _ => (hf i).of_le (mod_cast hm)).add
    have h'u : ∀ k : ℕ, (k : ℕ∞) ≤ m → Summable (v k ∘ ((↑) : { i // i ∉ T } → α)) := fun k hk =>
      (hv k (hk.trans hm)).subtype _
    refine contDiff_tsum (fun i => (hf i).of_le (mod_cast hm)) h'u ?_
    rintro k ⟨i, hi⟩ x hk
    simp only [t, T, Finite.mem_toFinset, mem_setOf_eq, Finset.mem_range, not_forall, not_le,
      exists_prop, not_exists, not_and, not_lt] at hi
    exact hi k (Nat.lt_succ_iff.2 (WithTop.coe_le_coe.1 hk)) x

