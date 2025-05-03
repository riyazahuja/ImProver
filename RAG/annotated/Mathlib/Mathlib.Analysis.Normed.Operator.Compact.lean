/-- A compact operator between two topological vector spaces. This definition is usually
given as "there exists a neighborhood of zero whose image is contained in a compact set",
but we choose a definition which involves fewer existential quantifiers and replaces images
with preimages.

We prove the equivalence in `isCompactOperator_iff_exists_mem_nhds_image_subset_compact`. -/
def IsCompactOperator {M₁ M₂ : Type*} [Zero M₁] [TopologicalSpace M₁] [TopologicalSpace M₂]
    (f : M₁ → M₂) : Prop :=
  ∃ K, IsCompact K ∧ f ⁻¹' K ∈ (𝓝 0 : Filter M₁)


theorem isCompactOperator_zero {M₁ M₂ : Type*} [Zero M₁] [TopologicalSpace M₁]
    [TopologicalSpace M₂] [Zero M₂] : IsCompactOperator (0 : M₁ → M₂) :=
  ⟨{0}, isCompact_singleton, mem_of_superset univ_mem fun _ _ => rfl⟩


theorem isCompactOperator_iff_exists_mem_nhds_image_subset_compact (f : M₁ → M₂) :
    IsCompactOperator f ↔ ∃ V ∈ (𝓝 0 : Filter M₁), ∃ K : Set M₂, IsCompact K ∧ f '' V ⊆ K :=
  ⟨fun ⟨K, hK, hKf⟩ => ⟨f ⁻¹' K, hKf, K, hK, image_preimage_subset _ _⟩, fun ⟨_, hV, K, hK, hVK⟩ =>
    ⟨K, hK, mem_of_superset hV (image_subset_iff.mp hVK)⟩⟩


theorem isCompactOperator_iff_exists_mem_nhds_isCompact_closure_image [T2Space M₂] (f : M₁ → M₂) :
    IsCompactOperator f ↔ ∃ V ∈ (𝓝 0 : Filter M₁), IsCompact (closure <| f '' V) := by
  /-
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝³ : TopologicalSpace M₁
    inst✝² : AddCommMonoid M₁
    inst✝¹ : TopologicalSpace M₂
    inst✝ : T2Space M₂
    f : M₁ → M₂
    ⊢ Iff (IsCompactOperator f) (Exists fun V => And (Membership.mem (nhds 0) V) ( …
  -/
  rw [isCompactOperator_iff_exists_mem_nhds_image_subset_compact]
  exact
    ⟨fun ⟨V, hV, K, hK, hKV⟩ => ⟨V, hV, hK.closure_of_subset hKV⟩,
      fun ⟨V, hV, hVc⟩ => ⟨V, hV, closure (f '' V), hVc, subset_closure⟩⟩


theorem IsCompactOperator.image_subset_compact_of_isVonNBounded {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) {S : Set M₁} (hS : IsVonNBounded 𝕜₁ S) :
    ∃ K : Set M₂, IsCompact K ∧ f '' S ⊆ K :=
  let ⟨K, hK, hKf⟩ := hf
  let ⟨r, hr, hrS⟩ := (hS hKf).exists_pos
  let ⟨c, hc⟩ := NormedField.exists_lt_norm 𝕜₁ r
  let this := ne_zero_of_norm_ne_zero (hr.trans hc).ne.symm
  ⟨σ₁₂ c • K, hK.image <| continuous_id.const_smul (σ₁₂ c), by
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      inst✝⁸ : NontriviallyNormedField 𝕜₁
      inst✝⁷ : SeminormedRing 𝕜₂
      σ₁₂ : RingHom 𝕜₁ 𝕜₂
      M₁ : Type u_3
      M₂ : Type u_4
      inst✝⁶ : TopologicalSpace M₁
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module 𝕜₁ M₁
      inst✝¹ : Module 𝕜₂ M₂
      inst✝ : ContinuousConstSMul 𝕜₂ M₂
      f : LinearMap σ₁₂ M₁ M₂
      hf : IsCompactOperator ⇑f
      S : Set M₁
      hS : Bornology.IsVonNBounded 𝕜₁ S
      K : Set M₂
      hK : IsCompact K
      hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
      r : Real
      hr : GT.gt r 0
      hrS : ∀ (c : 𝕜₁), LE.le r (Norm.norm c) → HasSubset.Subset S (HSMul.hSMul c (S …
      c : 𝕜₁
      hc : LT.lt r (Norm.norm c)
      this : Ne c 0 := ne_zero_of_norm_ne_zero (Ne.symm (LT.lt.ne (LT.lt.trans hr hc …
      ⊢ HasSubset.Subset (Set.image (⇑f) S) (HSMul.hSMul (σ₁₂ c) K)
    -/
    rw [image_subset_iff, preimage_smul_setₛₗ _ _ _ f this.isUnit]; exact hrS c hc.le⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem IsCompactOperator.isCompact_closure_image_of_isVonNBounded [T2Space M₂] {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) {S : Set M₁} (hS : IsVonNBounded 𝕜₁ S) :
    IsCompact (closure <| f '' S) :=
  let ⟨_, hK, hKf⟩ := hf.image_subset_compact_of_isVonNBounded hS
  hK.closure_of_subset hKf


theorem IsCompactOperator.image_subset_compact_of_bounded [ContinuousConstSMul 𝕜₂ M₂]
    {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) {S : Set M₁} (hS : Bornology.IsBounded S) :
    ∃ K : Set M₂, IsCompact K ∧ f '' S ⊆ K :=
                                                 /-
                                                   𝕜₁ : Type u_1
                                                   𝕜₂ : Type u_2
                                                   inst✝⁷ : NontriviallyNormedField 𝕜₁
                                                   inst✝⁶ : SeminormedRing 𝕜₂
                                                   σ₁₂ : RingHom 𝕜₁ 𝕜₂
                                                   M₁ : Type u_3
                                                   M₂ : Type u_4
                                                   inst✝⁵ : SeminormedAddCommGroup M₁
                                                   inst✝⁴ : TopologicalSpace M₂
                                                   inst✝³ : AddCommMonoid M₂
                                                   inst✝² : NormedSpace 𝕜₁ M₁
                                                   inst✝¹ : Module 𝕜₂ M₂
                                                   inst✝ : ContinuousConstSMul 𝕜₂ M₂
                                                   f : LinearMap σ₁₂ M₁ M₂
                                                   hf : IsCompactOperator ⇑f
                                                   S : Set M₁
                                                   hS : Bornology.IsBounded S
                                                   ⊢ Bornology.IsVonNBounded 𝕜₁ S
                                                 -/
  hf.image_subset_compact_of_isVonNBounded <| by rwa [NormedSpace.isVonNBounded_iff]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem IsCompactOperator.isCompact_closure_image_of_bounded [ContinuousConstSMul 𝕜₂ M₂]
    [T2Space M₂] {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) {S : Set M₁}
    (hS : Bornology.IsBounded S) : IsCompact (closure <| f '' S) :=
                                                    /-
                                                      𝕜₁ : Type u_1
                                                      𝕜₂ : Type u_2
                                                      inst✝⁸ : NontriviallyNormedField 𝕜₁
                                                      inst✝⁷ : SeminormedRing 𝕜₂
                                                      σ₁₂ : RingHom 𝕜₁ 𝕜₂
                                                      M₁ : Type u_3
                                                      M₂ : Type u_4
                                                      inst✝⁶ : SeminormedAddCommGroup M₁
                                                      inst✝⁵ : TopologicalSpace M₂
                                                      inst✝⁴ : AddCommMonoid M₂
                                                      inst✝³ : NormedSpace 𝕜₁ M₁
                                                      inst✝² : Module 𝕜₂ M₂
                                                      inst✝¹ : ContinuousConstSMul 𝕜₂ M₂
                                                      inst✝ : T2Space M₂
                                                      f : LinearMap σ₁₂ M₁ M₂
                                                      hf : IsCompactOperator ⇑f
                                                      S : Set M₁
                                                      hS : Bornology.IsBounded S
                                                      ⊢ Bornology.IsVonNBounded 𝕜₁ S
                                                    -/
  hf.isCompact_closure_image_of_isVonNBounded <| by rwa [NormedSpace.isVonNBounded_iff]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem IsCompactOperator.image_ball_subset_compact [ContinuousConstSMul 𝕜₂ M₂] {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) (r : ℝ) : ∃ K : Set M₂, IsCompact K ∧ f '' Metric.ball 0 r ⊆ K :=
  hf.image_subset_compact_of_isVonNBounded (NormedSpace.isVonNBounded_ball 𝕜₁ M₁ r)


theorem IsCompactOperator.image_closedBall_subset_compact [ContinuousConstSMul 𝕜₂ M₂]
    {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) (r : ℝ) :
    ∃ K : Set M₂, IsCompact K ∧ f '' Metric.closedBall 0 r ⊆ K :=
  hf.image_subset_compact_of_isVonNBounded (NormedSpace.isVonNBounded_closedBall 𝕜₁ M₁ r)


theorem IsCompactOperator.isCompact_closure_image_ball [ContinuousConstSMul 𝕜₂ M₂] [T2Space M₂]
    {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) (r : ℝ) :
    IsCompact (closure <| f '' Metric.ball 0 r) :=
  hf.isCompact_closure_image_of_isVonNBounded (NormedSpace.isVonNBounded_ball 𝕜₁ M₁ r)


theorem IsCompactOperator.isCompact_closure_image_closedBall [ContinuousConstSMul 𝕜₂ M₂]
    [T2Space M₂] {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) (r : ℝ) :
    IsCompact (closure <| f '' Metric.closedBall 0 r) :=
  hf.isCompact_closure_image_of_isVonNBounded (NormedSpace.isVonNBounded_closedBall 𝕜₁ M₁ r)


theorem isCompactOperator_iff_image_ball_subset_compact [ContinuousConstSMul 𝕜₂ M₂]
    (f : M₁ →ₛₗ[σ₁₂] M₂) {r : ℝ} (hr : 0 < r) :
    IsCompactOperator f ↔ ∃ K : Set M₂, IsCompact K ∧ f '' Metric.ball 0 r ⊆ K :=
  ⟨fun hf => hf.image_ball_subset_compact r, fun ⟨K, hK, hKr⟩ =>
    (isCompactOperator_iff_exists_mem_nhds_image_subset_compact f).mpr
      ⟨Metric.ball 0 r, ball_mem_nhds _ hr, K, hK, hKr⟩⟩


theorem isCompactOperator_iff_image_closedBall_subset_compact [ContinuousConstSMul 𝕜₂ M₂]
    (f : M₁ →ₛₗ[σ₁₂] M₂) {r : ℝ} (hr : 0 < r) :
    IsCompactOperator f ↔ ∃ K : Set M₂, IsCompact K ∧ f '' Metric.closedBall 0 r ⊆ K :=
  ⟨fun hf => hf.image_closedBall_subset_compact r, fun ⟨K, hK, hKr⟩ =>
    (isCompactOperator_iff_exists_mem_nhds_image_subset_compact f).mpr
      ⟨Metric.closedBall 0 r, closedBall_mem_nhds _ hr, K, hK, hKr⟩⟩


theorem isCompactOperator_iff_isCompact_closure_image_ball [ContinuousConstSMul 𝕜₂ M₂] [T2Space M₂]
    (f : M₁ →ₛₗ[σ₁₂] M₂) {r : ℝ} (hr : 0 < r) :
    IsCompactOperator f ↔ IsCompact (closure <| f '' Metric.ball 0 r) :=
  ⟨fun hf => hf.isCompact_closure_image_ball r, fun hf =>
    (isCompactOperator_iff_exists_mem_nhds_isCompact_closure_image f).mpr
      ⟨Metric.ball 0 r, ball_mem_nhds _ hr, hf⟩⟩


theorem isCompactOperator_iff_isCompact_closure_image_closedBall [ContinuousConstSMul 𝕜₂ M₂]
    [T2Space M₂] (f : M₁ →ₛₗ[σ₁₂] M₂) {r : ℝ} (hr : 0 < r) :
    IsCompactOperator f ↔ IsCompact (closure <| f '' Metric.closedBall 0 r) :=
  ⟨fun hf => hf.isCompact_closure_image_closedBall r, fun hf =>
    (isCompactOperator_iff_exists_mem_nhds_isCompact_closure_image f).mpr
      ⟨Metric.closedBall 0 r, closedBall_mem_nhds _ hr, hf⟩⟩


theorem IsCompactOperator.smul {S : Type*} [Monoid S] [DistribMulAction S M₂]
    [ContinuousConstSMul S M₂] {f : M₁ → M₂} (hf : IsCompactOperator f) (c : S) :
    IsCompactOperator (c • f) :=
  let ⟨K, hK, hKf⟩ := hf
  ⟨c • K, hK.image <| continuous_id.const_smul c,
    mem_of_superset hKf fun _ hx => smul_mem_smul_set hx⟩


theorem IsCompactOperator.add [ContinuousAdd M₂] {f g : M₁ → M₂} (hf : IsCompactOperator f)
    (hg : IsCompactOperator g) : IsCompactOperator (f + g) :=
  let ⟨A, hA, hAf⟩ := hf
  let ⟨B, hB, hBg⟩ := hg
  ⟨A + B, hA.add hB,
    mem_of_superset (inter_mem hAf hBg) fun _ ⟨hxA, hxB⟩ => Set.add_mem_add hxA hxB⟩


theorem IsCompactOperator.neg [ContinuousNeg M₄] {f : M₁ → M₄} (hf : IsCompactOperator f) :
    IsCompactOperator (-f) :=
  let ⟨K, hK, hKf⟩ := hf
  ⟨-K, hK.neg, mem_of_superset hKf fun x (hx : f x ∈ K) => Set.neg_mem_neg.mpr hx⟩


theorem IsCompactOperator.sub [TopologicalAddGroup M₄] {f g : M₁ → M₄} (hf : IsCompactOperator f)
    (hg : IsCompactOperator g) : IsCompactOperator (f - g) := by
  /-
    M₁ : Type u_3
    M₄ : Type u_5
    inst✝⁴ : TopologicalSpace M₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : TopologicalSpace M₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : TopologicalAddGroup M₄
    f g : M₁ → M₄
    hf : IsCompactOperator f
    hg : IsCompactOperator g
    ⊢ IsCompactOperator (HSub.hSub f g)
  -/
  rw [sub_eq_add_neg]; exact hf.add hg.neg
                       /-
                         🎉 no goals
                       -/


/-- The submodule of compact continuous linear maps. -/
def compactOperator [Module R₁ M₁] [Module R₄ M₄] [ContinuousConstSMul R₄ M₄]
    [TopologicalAddGroup M₄] : Submodule R₄ (M₁ →SL[σ₁₄] M₄) where
  carrier := { f | IsCompactOperator f }
  add_mem' hf hg := hf.add hg
  zero_mem' := isCompactOperator_zero
  smul_mem' c _ hf := hf.smul c


theorem IsCompactOperator.comp_clm [AddCommMonoid M₂] [Module R₂ M₂] {f : M₂ → M₃}
    (hf : IsCompactOperator f) (g : M₁ →SL[σ₁₂] M₂) : IsCompactOperator (f ∘ g) := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁸ : Semiring R₁
    inst✝⁷ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₁
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    f : M₂ → M₃
    hf : IsCompactOperator f
    g : ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ IsCompactOperator (Function.comp f ⇑g)
  -/
  have := g.continuous.tendsto 0
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁸ : Semiring R₁
    inst✝⁷ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₁
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    f : M₂ → M₃
    hf : IsCompactOperator f
    g : ContinuousLinearMap σ₁₂ M₁ M₂
    this : Filter.Tendsto (⇑g) (nhds 0) (nhds (g 0))
    ⊢ IsCompactOperator (Function.comp f ⇑g)
  -/
  rw [map_zero] at this
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁸ : Semiring R₁
    inst✝⁷ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₁
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    f : M₂ → M₃
    hf : IsCompactOperator f
    g : ContinuousLinearMap σ₁₂ M₁ M₂
    this : Filter.Tendsto (⇑g) (nhds 0) (nhds 0)
    ⊢ IsCompactOperator (Function.comp f ⇑g)
  -/
  rcases hf with ⟨K, hK, hKf⟩
  /-
    case intro.intro
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁸ : Semiring R₁
    inst✝⁷ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₁
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    f : M₂ → M₃
    g : ContinuousLinearMap σ₁₂ M₁ M₂
    this : Filter.Tendsto (⇑g) (nhds 0) (nhds 0)
    K : Set M₃
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage f K)
    ⊢ IsCompactOperator (Function.comp f ⇑g)
  -/
  exact ⟨K, hK, this hKf⟩
  /-
    🎉 no goals
  -/


theorem IsCompactOperator.continuous_comp {f : M₁ → M₂} (hf : IsCompactOperator f) {g : M₂ → M₃}
    (hg : Continuous g) : IsCompactOperator (g ∘ f) := by
  /-
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝³ : TopologicalSpace M₁
    inst✝² : TopologicalSpace M₂
    inst✝¹ : TopologicalSpace M₃
    inst✝ : AddCommMonoid M₁
    f : M₁ → M₂
    hf : IsCompactOperator f
    g : M₂ → M₃
    hg : Continuous g
    ⊢ IsCompactOperator (Function.comp g f)
  -/
  rcases hf with ⟨K, hK, hKf⟩
  /-
    case intro.intro
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝³ : TopologicalSpace M₁
    inst✝² : TopologicalSpace M₂
    inst✝¹ : TopologicalSpace M₃
    inst✝ : AddCommMonoid M₁
    f : M₁ → M₂
    g : M₂ → M₃
    hg : Continuous g
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage f K)
    ⊢ IsCompactOperator (Function.comp g f)
  -/
  refine ⟨g '' K, hK.image hg, mem_of_superset hKf ?_⟩
  /-
    case intro.intro
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝³ : TopologicalSpace M₁
    inst✝² : TopologicalSpace M₂
    inst✝¹ : TopologicalSpace M₃
    inst✝ : AddCommMonoid M₁
    f : M₁ → M₂
    g : M₂ → M₃
    hg : Continuous g
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage f K)
    ⊢ HasSubset.Subset (Set.preimage f K) (Set.preimage (Function.comp g f) (Set.i …
  -/
  rw [preimage_comp]
  /-
    case intro.intro
    M₁ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝³ : TopologicalSpace M₁
    inst✝² : TopologicalSpace M₂
    inst✝¹ : TopologicalSpace M₃
    inst✝ : AddCommMonoid M₁
    f : M₁ → M₂
    g : M₂ → M₃
    hg : Continuous g
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage f K)
    ⊢ HasSubset.Subset (Set.preimage f K) (Set.preimage f (Set.preimage g (Set.ima …
  -/
  exact preimage_mono (subset_preimage_image _ _)
  /-
    🎉 no goals
  -/


theorem IsCompactOperator.clm_comp [AddCommMonoid M₂] [Module R₂ M₂] [AddCommMonoid M₃]
    [Module R₃ M₃] {f : M₁ → M₂} (hf : IsCompactOperator f) (g : M₂ →SL[σ₂₃] M₃) :
    IsCompactOperator (g ∘ f) :=
  hf.continuous_comp g.continuous


theorem IsCompactOperator.codRestrict {f : M₁ → M₂} (hf : IsCompactOperator f) {V : Submodule R₂ M₂}
    (hV : ∀ x, f x ∈ V) (h_closed : IsClosed (V : Set M₂)) :
    IsCompactOperator (Set.codRestrict f V hV) :=
  let ⟨_, hK, hKf⟩ := hf
  ⟨_, h_closed.isClosedEmbedding_subtypeVal.isCompact_preimage hK, hKf⟩


/-- If a compact operator preserves a closed submodule, its restriction to that submodule is
compact.

Note that, following mathlib's convention in linear algebra, `restrict` designates the restriction
of an endomorphism `f : E →ₗ E` to an endomorphism `f' : ↥V →ₗ ↥V`. To prove that the restriction
`f' : ↥U →ₛₗ ↥V` of a compact operator `f : E →ₛₗ F` is compact, apply
`IsCompactOperator.codRestrict` to `f ∘ U.subtypeL`, which is compact by
`IsCompactOperator.comp_clm`. -/
theorem IsCompactOperator.restrict {f : M₁ →ₗ[R₁] M₁} (hf : IsCompactOperator f)
    {V : Submodule R₁ M₁} (hV : ∀ v ∈ V, f v ∈ V) (h_closed : IsClosed (V : Set M₁)) :
    IsCompactOperator (f.restrict hV) :=
  (hf.comp_clm V.subtypeL).codRestrict (SetLike.forall.2 hV) h_closed


/-- If a compact operator preserves a complete submodule, its restriction to that submodule is
compact.

Note that, following mathlib's convention in linear algebra, `restrict` designates the restriction
of an endomorphism `f : E →ₗ E` to an endomorphism `f' : ↥V →ₗ ↥V`. To prove that the restriction
`f' : ↥U →ₛₗ ↥V` of a compact operator `f : E →ₛₗ F` is compact, apply
`IsCompactOperator.codRestrict` to `f ∘ U.subtypeL`, which is compact by
`IsCompactOperator.comp_clm`. -/
theorem IsCompactOperator.restrict' [T0Space M₂] {f : M₂ →ₗ[R₂] M₂}
    (hf : IsCompactOperator f) {V : Submodule R₂ M₂} (hV : ∀ v ∈ V, f v ∈ V)
    [hcomplete : CompleteSpace V] : IsCompactOperator (f.restrict hV) :=
  hf.restrict hV (completeSpace_coe_iff_isComplete.mp hcomplete).isClosed


@[continuity]
theorem IsCompactOperator.continuous {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) :
    Continuous f := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    hf : IsCompactOperator ⇑f
    ⊢ Continuous ⇑f
  -/
  letI : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace _
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    hf : IsCompactOperator ⇑f
    this : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    ⊢ Continuous ⇑f
  -/
  haveI : UniformAddGroup M₂ := comm_topologicalAddGroup_is_uniform
  -- Since `f` is linear, we only need to show that it is continuous at zero.
  -- Let `U` be a neighborhood of `0` in `M₂`.
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    hf : IsCompactOperator ⇑f
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    ⊢ Continuous ⇑f
  -/
  refine continuous_of_continuousAt_zero f fun U hU => ?_
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    hf : IsCompactOperator ⇑f
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds (f 0)) U
    ⊢ Membership.mem (Filter.map (⇑f) (nhds 0)) U
  -/
  rw [map_zero] at hU
  -- The compactness of `f` gives us a compact set `K : Set M₂` such that `f ⁻¹' K` is a
  -- neighborhood of `0` in `M₁`.
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    hf : IsCompactOperator ⇑f
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    ⊢ Membership.mem (Filter.map (⇑f) (nhds 0)) U
  -/
  rcases hf with ⟨K, hK, hKf⟩
  -- But any compact set is totally bounded, hence Von-Neumann bounded. Thus, `K` absorbs `U`.
  -- This gives `r > 0` such that `∀ a : 𝕜₂, r ≤ ‖a‖ → K ⊆ a • U`.
  /-
    case intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    ⊢ Membership.mem (Filter.map (⇑f) (nhds 0)) U
  -/
  rcases (hK.totallyBounded.isVonNBounded 𝕜₂ hU).exists_pos with ⟨r, hr, hrU⟩
  -- Choose `c : 𝕜₂` with `r < ‖c‖`.
  /-
    case intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    ⊢ Membership.mem (Filter.map (⇑f) (nhds 0)) U
  -/
  rcases NormedField.exists_lt_norm 𝕜₁ r with ⟨c, hc⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    c : 𝕜₁
    hc : LT.lt r (Norm.norm c)
    ⊢ Membership.mem (Filter.map (⇑f) (nhds 0)) U
  -/
  have hcnz : c ≠ 0 := ne_zero_of_norm_ne_zero (hr.trans hc).ne.symm
  -- We have `f ⁻¹' ((σ₁₂ c⁻¹) • K) = c⁻¹ • f ⁻¹' K ∈ 𝓝 0`. Thus, showing that
  -- `(σ₁₂ c⁻¹) • K ⊆ U` is enough to deduce that `f ⁻¹' U ∈ 𝓝 0`.
  suffices (σ₁₂ <| c⁻¹) • K ⊆ U by
    refine mem_of_superset ?_ this
    have : IsUnit c⁻¹ := hcnz.isUnit.inv
    rwa [mem_map, preimage_smul_setₛₗ _ _ _ f this, set_smul_mem_nhds_zero_iff (inv_ne_zero hcnz)]
  -- Since `σ₁₂ c⁻¹` = `(σ₁₂ c)⁻¹`, we have to prove that `K ⊆ σ₁₂ c • U`.
  /-
    case intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    c : 𝕜₁
    hc : LT.lt r (Norm.norm c)
    hcnz : Ne c 0
    ⊢ HasSubset.Subset (HSMul.hSMul (σ₁₂ (Inv.inv c)) K) U
  -/
  rw [map_inv₀, ← subset_set_smul_iff₀ ((map_ne_zero σ₁₂).mpr hcnz)]
  -- But `σ₁₂` is isometric, so `‖σ₁₂ c‖ = ‖c‖ > r`, which concludes the argument since
  -- `∀ a : 𝕜₂, r ≤ ‖a‖ → K ⊆ a • U`.
  /-
    case intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    c : 𝕜₁
    hc : LT.lt r (Norm.norm c)
    hcnz : Ne c 0
    ⊢ HasSubset.Subset K (HSMul.hSMul (σ₁₂ c) U)
  -/
  refine hrU (σ₁₂ c) ?_
  /-
    case intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    c : 𝕜₁
    hc : LT.lt r (Norm.norm c)
    hcnz : Ne c 0
    ⊢ LE.le r (Norm.norm (σ₁₂ c))
  -/
  rw [RingHomIsometric.is_iso]
  /-
    case intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜₁
    inst✝¹¹ : NontriviallyNormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    inst✝¹⁰ : RingHomIsometric σ₁₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module 𝕜₁ M₁
    inst✝⁴ : Module 𝕜₂ M₂
    inst✝³ : TopologicalAddGroup M₁
    inst✝² : ContinuousConstSMul 𝕜₁ M₁
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : ContinuousSMul 𝕜₂ M₂
    f : LinearMap σ₁₂ M₁ M₂
    this✝ : UniformSpace M₂ := TopologicalAddGroup.toUniformSpace M₂
    this : UniformAddGroup M₂
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    K : Set M₂
    hK : IsCompact K
    hKf : Membership.mem (nhds 0) (Set.preimage (⇑f) K)
    r : Real
    hr : GT.gt r 0
    hrU : ∀ (c : 𝕜₂), LE.le r (Norm.norm c) → HasSubset.Subset K (HSMul.hSMul c U)
    c : 𝕜₁
    hc : LT.lt r (Norm.norm c)
    hcnz : Ne c 0
    ⊢ LE.le r (Norm.norm c)
  -/
  exact hc.le
  /-
    🎉 no goals
  -/


/-- Upgrade a compact `LinearMap` to a `ContinuousLinearMap`. -/
def ContinuousLinearMap.mkOfIsCompactOperator {f : M₁ →ₛₗ[σ₁₂] M₂} (hf : IsCompactOperator f) :
    M₁ →SL[σ₁₂] M₂ :=
  ⟨f, hf.continuous⟩


@[simp]
theorem ContinuousLinearMap.mkOfIsCompactOperator_to_linearMap {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) :
    (ContinuousLinearMap.mkOfIsCompactOperator hf : M₁ →ₛₗ[σ₁₂] M₂) = f :=
  rfl


@[simp]
theorem ContinuousLinearMap.coe_mkOfIsCompactOperator {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) : (ContinuousLinearMap.mkOfIsCompactOperator hf : M₁ → M₂) = f :=
  rfl


theorem ContinuousLinearMap.mkOfIsCompactOperator_mem_compactOperator {f : M₁ →ₛₗ[σ₁₂] M₂}
    (hf : IsCompactOperator f) :
    ContinuousLinearMap.mkOfIsCompactOperator hf ∈ compactOperator σ₁₂ M₁ M₂ :=
  hf


/-- The set of compact operators from a normed space to a complete topological vector space is
closed. -/
theorem isClosed_setOf_isCompactOperator {𝕜₁ 𝕜₂ : Type*} [NontriviallyNormedField 𝕜₁]
    [NormedField 𝕜₂] {σ₁₂ : 𝕜₁ →+* 𝕜₂} {M₁ M₂ : Type*} [SeminormedAddCommGroup M₁]
    [AddCommGroup M₂] [NormedSpace 𝕜₁ M₁] [Module 𝕜₂ M₂] [UniformSpace M₂] [UniformAddGroup M₂]
    [ContinuousConstSMul 𝕜₂ M₂] [T2Space M₂] [CompleteSpace M₂] :
    IsClosed { f : M₁ →SL[σ₁₂] M₂ | IsCompactOperator f } := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    ⊢ IsClosed (setOf fun f => IsCompactOperator ⇑f)
  -/
  refine isClosed_of_closure_subset ?_
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    ⊢ HasSubset.Subset (closure (setOf fun f => IsCompactOperator ⇑f)) (setOf fun  …
  -/
  rintro u hu
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : Membership.mem (closure (setOf fun f => IsCompactOperator ⇑f)) u
    ⊢ Membership.mem (setOf fun f => IsCompactOperator ⇑f) u
  -/
  rw [mem_closure_iff_nhds_zero] at hu
  suffices TotallyBounded (u '' Metric.closedBall 0 1) by
    change IsCompactOperator (u : M₁ →ₛₗ[σ₁₂] M₂)
    rw [isCompactOperator_iff_isCompact_closure_image_closedBall (u : M₁ →ₛₗ[σ₁₂] M₂) zero_lt_one]
    exact isCompact_of_totallyBounded_isClosed this.closure isClosed_closure
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    ⊢ TotallyBounded (Set.image (⇑u) (Metric.closedBall 0 1))
  -/
  rw [totallyBounded_iff_subset_finite_iUnion_nhds_zero]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    ⊢ ∀ (U : Set M₂), Membership.mem (nhds 0) U → Exists fun t => And t.Finite (Ha …
  -/
  intro U hU
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.image (⇑u) (Metric.close …
  -/
  rcases exists_nhds_zero_half hU with ⟨V, hV, hVU⟩
  /-
    case intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.image (⇑u) (Metric.close …
  -/
  let SV : Set M₁ × Set M₂ := ⟨closedBall 0 1, -V⟩
  rcases hu { f | ∀ x ∈ SV.1, f x ∈ SV.2 }
      (ContinuousLinearMap.hasBasis_nhds_zero.mem_of_mem
        ⟨NormedSpace.isVonNBounded_closedBall _ _ _, neg_mem_nhds_zero M₂ hV⟩) with
    ⟨v, hv, huv⟩
  rcases totallyBounded_iff_subset_finite_iUnion_nhds_zero.mp
      (hv.isCompact_closure_image_closedBall 1).totallyBounded V hV with
    ⟨T, hT, hTv⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.image (⇑u) (Metric.close …
  -/
  have hTv : v '' closedBall 0 1 ⊆ _ := subset_closure.trans hTv
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    hTv : HasSubset.Subset (Set.image (⇑v) (Metric.closedBall 0 1)) (Set.iUnion fu …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.image (⇑u) (Metric.close …
  -/
  refine ⟨T, hT, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    hTv : HasSubset.Subset (Set.image (⇑v) (Metric.closedBall 0 1)) (Set.iUnion fu …
    ⊢ HasSubset.Subset (Set.image (⇑u) (Metric.closedBall 0 1)) (Set.iUnion fun y  …
  -/
  rw [image_subset_iff, preimage_iUnion₂] at hTv ⊢
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    hTv : HasSubset.Subset (Metric.closedBall 0 1) (Set.iUnion fun i => Set.iUnion …
    ⊢ HasSubset.Subset (Metric.closedBall 0 1) (Set.iUnion fun i => Set.iUnion fun …
  -/
  intro x hx
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    hTv : HasSubset.Subset (Metric.closedBall 0 1) (Set.iUnion fun i => Set.iUnion …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.preimage (⇑u) (H …
  -/
  specialize hTv hx
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    hTv : Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.preimage (⇑v …
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.preimage (⇑u) (H …
  -/
  rw [mem_iUnion₂] at hTv ⊢
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv✝ : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (S …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    hTv : Exists fun i => Exists fun j => Membership.mem (Set.preimage (⇑v) (HVAdd …
    ⊢ Exists fun i => Exists fun j => Membership.mem (Set.preimage (⇑u) (HVAdd.hVA …
  -/
  rcases hTv with ⟨t, ht, htx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    t : M₂
    ht : Membership.mem T t
    htx : Membership.mem (Set.preimage (⇑v) (HVAdd.hVAdd t V)) x
    ⊢ Exists fun i => Exists fun j => Membership.mem (Set.preimage (⇑u) (HVAdd.hVA …
  -/
  refine ⟨t, ht, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    t : M₂
    ht : Membership.mem T t
    htx : Membership.mem (Set.preimage (⇑v) (HVAdd.hVAdd t V)) x
    ⊢ Membership.mem (Set.preimage (⇑u) (HVAdd.hVAdd t U)) x
  -/
  rw [mem_preimage, mem_vadd_set_iff_neg_vadd_mem, vadd_eq_add, neg_add_eq_sub] at htx ⊢
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    t : M₂
    ht : Membership.mem T t
    htx : Membership.mem V (HSub.hSub (v x) t)
    ⊢ Membership.mem U (HSub.hSub (u x) t)
  -/
  convert hVU _ htx _ (huv x hx) using 1
  /-
    case h.e'_5
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    t : M₂
    ht : Membership.mem T t
    htx : Membership.mem V (HSub.hSub (v x) t)
    ⊢ Eq (HSub.hSub (u x) t) (HAdd.hAdd (HSub.hSub (v x) t) (Neg.neg ((HSub.hSub v …
  -/
  rw [ContinuousLinearMap.sub_apply]
  /-
    case h.e'_5
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ₁₂ : RingHom 𝕜₁ 𝕜₂
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁸ : SeminormedAddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : NormedSpace 𝕜₁ M₁
    inst✝⁵ : Module 𝕜₂ M₂
    inst✝⁴ : UniformSpace M₂
    inst✝³ : UniformAddGroup M₂
    inst✝² : ContinuousConstSMul 𝕜₂ M₂
    inst✝¹ : T2Space M₂
    inst✝ : CompleteSpace M₂
    u : ContinuousLinearMap σ₁₂ M₁ M₂
    hu : ∀ (U : Set (ContinuousLinearMap σ₁₂ M₁ M₂)), Membership.mem (nhds 0) U →  …
    U : Set M₂
    hU : Membership.mem (nhds 0) U
    V : Set M₂
    hV : Membership.mem (nhds 0) V
    hVU : ∀ (v : M₂), Membership.mem V v → ∀ (w : M₂), Membership.mem V w → Member …
    SV : Prod (Set M₁) (Set M₂) := { fst := Metric.closedBall 0 1, snd := Neg.neg  …
    v : ContinuousLinearMap σ₁₂ M₁ M₂
    hv : Membership.mem (setOf fun f => IsCompactOperator ⇑f) v
    huv : Membership.mem (setOf fun f => ∀ (x : M₁), Membership.mem SV.1 x → Membe …
    T : Set M₂
    hT : T.Finite
    hTv : HasSubset.Subset (closure (Set.image (⇑↑v) (Metric.closedBall 0 1))) (Se …
    x : M₁
    hx : Membership.mem (Metric.closedBall 0 1) x
    t : M₂
    ht : Membership.mem T t
    htx : Membership.mem V (HSub.hSub (v x) t)
    ⊢ Eq (HSub.hSub (u x) t) (HAdd.hAdd (HSub.hSub (v x) t) (Neg.neg (HSub.hSub (v …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem compactOperator_topologicalClosure {𝕜₁ 𝕜₂ : Type*} [NontriviallyNormedField 𝕜₁]
    [NormedField 𝕜₂] {σ₁₂ : 𝕜₁ →+* 𝕜₂} {M₁ M₂ : Type*} [SeminormedAddCommGroup M₁]
    [AddCommGroup M₂] [NormedSpace 𝕜₁ M₁] [Module 𝕜₂ M₂] [UniformSpace M₂] [UniformAddGroup M₂]
    [ContinuousConstSMul 𝕜₂ M₂] [T2Space M₂] [CompleteSpace M₂] :
    (compactOperator σ₁₂ M₁ M₂).topologicalClosure = compactOperator σ₁₂ M₁ M₂ :=
  SetLike.ext' isClosed_setOf_isCompactOperator.closure_eq


theorem isCompactOperator_of_tendsto {ι 𝕜₁ 𝕜₂ : Type*} [NontriviallyNormedField 𝕜₁]
    [NormedField 𝕜₂] {σ₁₂ : 𝕜₁ →+* 𝕜₂} {M₁ M₂ : Type*} [SeminormedAddCommGroup M₁]
    [AddCommGroup M₂] [NormedSpace 𝕜₁ M₁] [Module 𝕜₂ M₂] [UniformSpace M₂] [UniformAddGroup M₂]
    [ContinuousConstSMul 𝕜₂ M₂] [T2Space M₂] [CompleteSpace M₂] {l : Filter ι} [l.NeBot]
    {F : ι → M₁ →SL[σ₁₂] M₂} {f : M₁ →SL[σ₁₂] M₂} (hf : Tendsto F l (𝓝 f))
    (hF : ∀ᶠ i in l, IsCompactOperator (F i)) : IsCompactOperator f :=
  isClosed_setOf_isCompactOperator.mem_of_tendsto hf hF

