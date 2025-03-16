/-- A function `f : X → Y` between two `PseudoEMetricSpace`s is Hölder continuous with constant
`C : ℝ≥0` and exponent `r : ℝ≥0`, if `edist (f x) (f y) ≤ C * edist x y ^ r` for all `x y : X`. -/
def HolderWith (C r : ℝ≥0) (f : X → Y) : Prop :=
  ∀ x y, edist (f x) (f y) ≤ (C : ℝ≥0∞) * edist x y ^ (r : ℝ)


/-- A function `f : X → Y` between two `PseudoEMetricSpace`s is Hölder continuous with constant
`C : ℝ≥0` and exponent `r : ℝ≥0` on a set `s : Set X`, if `edist (f x) (f y) ≤ C * edist x y ^ r`
for all `x y ∈ s`. -/
def HolderOnWith (C r : ℝ≥0) (f : X → Y) (s : Set X) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, edist (f x) (f y) ≤ (C : ℝ≥0∞) * edist x y ^ (r : ℝ)


@[simp]
theorem holderOnWith_empty (C r : ℝ≥0) (f : X → Y) : HolderOnWith C r f ∅ := fun _ hx => hx.elim


@[simp]
theorem holderOnWith_singleton (C r : ℝ≥0) (f : X → Y) (x : X) : HolderOnWith C r f {x} := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    x : X
    ⊢ HolderOnWith C r f (Singleton.singleton x)
  -/
  rintro a (rfl : a = x) b (rfl : b = a)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    b : X
    ⊢ LE.le (EDist.edist (f b) (f b)) (HMul.hMul (↑C) (HPow.hPow (EDist.edist b b) …
  -/
  rw [edist_self]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    b : X
    ⊢ LE.le 0 (HMul.hMul (↑C) (HPow.hPow (EDist.edist b b) ↑r))
  -/
  exact zero_le _
  /-
    🎉 no goals
  -/


theorem Set.Subsingleton.holderOnWith {s : Set X} (hs : s.Subsingleton) (C r : ℝ≥0) (f : X → Y) :
    HolderOnWith C r f s :=
  hs.induction_on (holderOnWith_empty C r f) (holderOnWith_singleton C r f)


theorem holderOnWith_univ {C r : ℝ≥0} {f : X → Y} : HolderOnWith C r f univ ↔ HolderWith C r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    ⊢ Iff (HolderOnWith C r f Set.univ) (HolderWith C r f)
  -/
  simp only [HolderOnWith, HolderWith, mem_univ, true_imp_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem holderOnWith_one {C : ℝ≥0} {f : X → Y} {s : Set X} :
    HolderOnWith C 1 f s ↔ LipschitzOnWith C f s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C : NNReal
    f : X → Y
    s : Set X
    ⊢ Iff (HolderOnWith C 1 f s) (LipschitzOnWith C f s)
  -/
  simp only [HolderOnWith, LipschitzOnWith, NNReal.coe_one, ENNReal.rpow_one]
  /-
    🎉 no goals
  -/


alias ⟨_, LipschitzOnWith.holderOnWith⟩ := holderOnWith_one


@[simp]
theorem holderWith_one {C : ℝ≥0} {f : X → Y} : HolderWith C 1 f ↔ LipschitzWith C f :=
  holderOnWith_univ.symm.trans <| holderOnWith_one.trans lipschitzOnWith_univ


alias ⟨_, LipschitzWith.holderWith⟩ := holderWith_one


theorem holderWith_id : HolderWith 1 1 (id : X → X) :=
  LipschitzWith.id.holderWith


protected theorem HolderWith.holderOnWith {C r : ℝ≥0} {f : X → Y} (h : HolderWith C r f)
    (s : Set X) : HolderOnWith C r f s := fun x _ y _ => h x y


theorem edist_le (h : HolderOnWith C r f s) {x y : X} (hx : x ∈ s) (hy : y ∈ s) :
    edist (f x) (f y) ≤ (C : ℝ≥0∞) * edist x y ^ (r : ℝ) :=
  h x hx y hy


theorem edist_le_of_le (h : HolderOnWith C r f s) {x y : X} (hx : x ∈ s) (hy : y ∈ s) {d : ℝ≥0∞}
    (hd : edist x y ≤ d) : edist (f x) (f y) ≤ (C : ℝ≥0∞) * d ^ (r : ℝ) :=
                                 /-
                                   X : Type u_1
                                   Y : Type u_2
                                   inst✝¹ : PseudoEMetricSpace X
                                   inst✝ : PseudoEMetricSpace Y
                                   C r : NNReal
                                   f : X → Y
                                   s : Set X
                                   h : HolderOnWith C r f s
                                   x y : X
                                   hx : Membership.mem s x
                                   hy : Membership.mem s y
                                   d : ENNReal
                                   hd : LE.le (EDist.edist x y) d
                                   ⊢ LE.le (HMul.hMul (↑C) (HPow.hPow (EDist.edist x y) ↑r)) (HMul.hMul (↑C) (HPo …
                                 -/
  (h.edist_le hx hy).trans <| by gcongr
                                 /-
                                   🎉 no goals
                                 -/


theorem comp {Cg rg : ℝ≥0} {g : Y → Z} {t : Set Y} (hg : HolderOnWith Cg rg g t) {Cf rf : ℝ≥0}
    {f : X → Y} (hf : HolderOnWith Cf rf f s) (hst : MapsTo f s t) :
    HolderOnWith (Cg * Cf ^ (rg : ℝ)) (rg * rf) (g ∘ f) s := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : PseudoEMetricSpace X
    inst✝¹ : PseudoEMetricSpace Y
    inst✝ : PseudoEMetricSpace Z
    s : Set X
    Cg rg : NNReal
    g : Y → Z
    t : Set Y
    hg : HolderOnWith Cg rg g t
    Cf rf : NNReal
    f : X → Y
    hf : HolderOnWith Cf rf f s
    hst : Set.MapsTo f s t
    ⊢ HolderOnWith (HMul.hMul Cg (HPow.hPow Cf ↑rg)) (HMul.hMul rg rf) (Function.c …
  -/
  intro x hx y hy
  rw [ENNReal.coe_mul, mul_comm rg, NNReal.coe_mul, ENNReal.rpow_mul, mul_assoc,
    ENNReal.coe_rpow_of_nonneg _ rg.coe_nonneg, ← ENNReal.mul_rpow_of_nonneg _ _ rg.coe_nonneg]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : PseudoEMetricSpace X
    inst✝¹ : PseudoEMetricSpace Y
    inst✝ : PseudoEMetricSpace Z
    s : Set X
    Cg rg : NNReal
    g : Y → Z
    t : Set Y
    hg : HolderOnWith Cg rg g t
    Cf rf : NNReal
    f : X → Y
    hf : HolderOnWith Cf rf f s
    hst : Set.MapsTo f s t
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist (Function.comp g f x) (Function.comp g f y)) (HMul.hMul ( …
  -/
  exact hg.edist_le_of_le (hst hx) (hst hy) (hf.edist_le hx hy)
  /-
    🎉 no goals
  -/


theorem comp_holderWith {Cg rg : ℝ≥0} {g : Y → Z} {t : Set Y} (hg : HolderOnWith Cg rg g t)
    {Cf rf : ℝ≥0} {f : X → Y} (hf : HolderWith Cf rf f) (ht : ∀ x, f x ∈ t) :
    HolderWith (Cg * Cf ^ (rg : ℝ)) (rg * rf) (g ∘ f) :=
  holderOnWith_univ.mp <| hg.comp (hf.holderOnWith univ) fun x _ => ht x


/-- A Hölder continuous function is uniformly continuous -/
protected theorem uniformContinuousOn (hf : HolderOnWith C r f s) (h0 : 0 < r) :
    UniformContinuousOn f s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    hf : HolderOnWith C r f s
    h0 : LT.lt 0 r
    ⊢ UniformContinuousOn f s
  -/
  refine EMetric.uniformContinuousOn_iff.2 fun ε εpos => ?_
  have : Tendsto (fun d : ℝ≥0∞ => (C : ℝ≥0∞) * d ^ (r : ℝ)) (𝓝 0) (𝓝 0) :=
    ENNReal.tendsto_const_mul_rpow_nhds_zero_of_pos ENNReal.coe_ne_top h0
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    hf : HolderOnWith C r f s
    h0 : LT.lt 0 r
    ε : ENNReal
    εpos : GT.gt ε 0
    this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a : X}, Membership.mem s a → ∀ {b : X},  …
  -/
  rcases ENNReal.nhds_zero_basis.mem_iff.1 (this (gt_mem_nhds εpos)) with ⟨δ, δ0, H⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    hf : HolderOnWith C r f s
    h0 : LT.lt 0 r
    ε : ENNReal
    εpos : GT.gt ε 0
    this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
    δ : ENNReal
    δ0 : LT.lt 0 δ
    H : HasSubset.Subset (Set.Iio δ) (Set.preimage (fun d => HMul.hMul (↑C) (HPow. …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a : X}, Membership.mem s a → ∀ {b : X},  …
  -/
  exact ⟨δ, δ0, fun hx y hy h => (hf.edist_le hx hy).trans_lt (H h)⟩
  /-
    🎉 no goals
  -/


protected theorem continuousOn (hf : HolderOnWith C r f s) (h0 : 0 < r) : ContinuousOn f s :=
  (hf.uniformContinuousOn h0).continuousOn


protected theorem mono (hf : HolderOnWith C r f s) (ht : t ⊆ s) : HolderOnWith C r f t :=
  fun _ hx _ hy => hf.edist_le (ht hx) (ht hy)


theorem ediam_image_le_of_le (hf : HolderOnWith C r f s) {d : ℝ≥0∞} (hd : EMetric.diam s ≤ d) :
    EMetric.diam (f '' s) ≤ (C : ℝ≥0∞) * d ^ (r : ℝ) :=
  EMetric.diam_image_le_iff.2 fun _ hx _ hy =>
    hf.edist_le_of_le hx hy <| (EMetric.edist_le_diam_of_mem hx hy).trans hd


theorem ediam_image_le (hf : HolderOnWith C r f s) :
    EMetric.diam (f '' s) ≤ (C : ℝ≥0∞) * EMetric.diam s ^ (r : ℝ) :=
  hf.ediam_image_le_of_le le_rfl


theorem ediam_image_le_of_subset (hf : HolderOnWith C r f s) (ht : t ⊆ s) :
    EMetric.diam (f '' t) ≤ (C : ℝ≥0∞) * EMetric.diam t ^ (r : ℝ) :=
  (hf.mono ht).ediam_image_le


theorem ediam_image_le_of_subset_of_le (hf : HolderOnWith C r f s) (ht : t ⊆ s) {d : ℝ≥0∞}
    (hd : EMetric.diam t ≤ d) : EMetric.diam (f '' t) ≤ (C : ℝ≥0∞) * d ^ (r : ℝ) :=
  (hf.mono ht).ediam_image_le_of_le hd


theorem ediam_image_inter_le_of_le (hf : HolderOnWith C r f s) {d : ℝ≥0∞}
    (hd : EMetric.diam t ≤ d) : EMetric.diam (f '' (t ∩ s)) ≤ (C : ℝ≥0∞) * d ^ (r : ℝ) :=
  hf.ediam_image_le_of_subset_of_le inter_subset_right <|
    (EMetric.diam_mono inter_subset_left).trans hd


theorem ediam_image_inter_le (hf : HolderOnWith C r f s) (t : Set X) :
    EMetric.diam (f '' (t ∩ s)) ≤ (C : ℝ≥0∞) * EMetric.diam t ^ (r : ℝ) :=
  hf.ediam_image_inter_le_of_le le_rfl


theorem restrict_iff {s : Set X} : HolderWith C r (s.restrict f) ↔ HolderOnWith C r f s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    ⊢ Iff (HolderWith C r (s.restrict f)) (HolderOnWith C r f s)
  -/
  simp [HolderWith, HolderOnWith]
  /-
    🎉 no goals
  -/


protected alias ⟨_, _root_.HolderOnWith.holderWith⟩ := restrict_iff


theorem edist_le (h : HolderWith C r f) (x y : X) :
    edist (f x) (f y) ≤ (C : ℝ≥0∞) * edist x y ^ (r : ℝ) :=
  h x y


theorem edist_le_of_le (h : HolderWith C r f) {x y : X} {d : ℝ≥0∞} (hd : edist x y ≤ d) :
    edist (f x) (f y) ≤ (C : ℝ≥0∞) * d ^ (r : ℝ) :=
  (h.holderOnWith univ).edist_le_of_le trivial trivial hd


theorem comp {Cg rg : ℝ≥0} {g : Y → Z} (hg : HolderWith Cg rg g) {Cf rf : ℝ≥0} {f : X → Y}
    (hf : HolderWith Cf rf f) : HolderWith (Cg * Cf ^ (rg : ℝ)) (rg * rf) (g ∘ f) :=
  (hg.holderOnWith univ).comp_holderWith hf fun _ => trivial


theorem comp_holderOnWith {Cg rg : ℝ≥0} {g : Y → Z} (hg : HolderWith Cg rg g) {Cf rf : ℝ≥0}
    {f : X → Y} {s : Set X} (hf : HolderOnWith Cf rf f s) :
    HolderOnWith (Cg * Cf ^ (rg : ℝ)) (rg * rf) (g ∘ f) s :=
  (hg.holderOnWith univ).comp hf fun _ _ => trivial


/-- A Hölder continuous function is uniformly continuous -/
protected theorem uniformContinuous (hf : HolderWith C r f) (h0 : 0 < r) : UniformContinuous f :=
  uniformContinuousOn_univ.mp <| (hf.holderOnWith univ).uniformContinuousOn h0


protected theorem continuous (hf : HolderWith C r f) (h0 : 0 < r) : Continuous f :=
  (hf.uniformContinuous h0).continuous


theorem ediam_image_le (hf : HolderWith C r f) (s : Set X) :
    EMetric.diam (f '' s) ≤ (C : ℝ≥0∞) * EMetric.diam s ^ (r : ℝ) :=
  EMetric.diam_image_le_iff.2 fun _ hx _ hy =>
    hf.edist_le_of_le <| EMetric.edist_le_diam_of_mem hx hy


lemma const {y : Y} :
    HolderWith C r (Function.const X y) := fun x₁ x₂ => by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    C r : NNReal
    y : Y
    x₁ x₂ : X
    ⊢ LE.le (EDist.edist (Function.const X y x₁) (Function.const X y x₂)) (HMul.hM …
  -/
  simp only [Function.const_apply, edist_self, zero_le]
  /-
    🎉 no goals
  -/


lemma zero [Zero Y] : HolderWith C r (0 : X → Y) := .const


lemma of_isEmpty [IsEmpty X] : HolderWith C r f := isEmptyElim


lemma mono {C' : ℝ≥0} (hf : HolderWith C r f) (h : C ≤ C') :
    HolderWith C' r f :=
  fun x₁ x₂ ↦ (hf x₁ x₂).trans (mul_right_mono (coe_le_coe.2 h))


theorem nndist_le_of_le (hf : HolderOnWith C r f s) (hx : x ∈ s) (hy : y ∈ s)
    {d : ℝ≥0} (hd : nndist x y ≤ d) : nndist (f x) (f y) ≤ C * d ^ (r : ℝ) := by
  rw [← ENNReal.coe_le_coe, ← edist_nndist, ENNReal.coe_mul,
    ENNReal.coe_rpow_of_nonneg _ r.coe_nonneg]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : NNReal
    hd : LE.le (NNDist.nndist x y) d
    ⊢ LE.le (EDist.edist (f x) (f y)) (HMul.hMul (↑C) (HPow.hPow ↑d ↑r))
  -/
  apply hf.edist_le_of_le hx hy
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : NNReal
    hd : LE.le (NNDist.nndist x y) d
    ⊢ LE.le (EDist.edist x y) ↑d
  -/
  rwa [edist_nndist, ENNReal.coe_le_coe]
  /-
    🎉 no goals
  -/


theorem nndist_le (hf : HolderOnWith C r f s) (hx : x ∈ s) (hy : y ∈ s) :
    nndist (f x) (f y) ≤ C * nndist x y ^ (r : ℝ) :=
  hf.nndist_le_of_le hx hy le_rfl


theorem dist_le_of_le (hf : HolderOnWith C r f s) (hx : x ∈ s) (hy : y ∈ s)
    {d : ℝ} (hd : dist x y ≤ d) : dist (f x) (f y) ≤ C * d ^ (r : ℝ) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : Real
    hd : LE.le (Dist.dist x y) d
    ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul (↑C) (HPow.hPow d ↑r))
  -/
  lift d to ℝ≥0 using dist_nonneg.trans hd
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : NNReal
    hd : LE.le (Dist.dist x y) ↑d
    ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul (↑C) (HPow.hPow ↑d ↑r))
  -/
  rw [dist_nndist] at hd ⊢
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : NNReal
    hd : LE.le ↑(NNDist.nndist x y) ↑d
    ⊢ LE.le (↑(NNDist.nndist (f x) (f y))) (HMul.hMul (↑C) (HPow.hPow ↑d ↑r))
  -/
  norm_cast at hd ⊢
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : PseudoMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    x y : X
    hf : HolderOnWith C r f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    d : NNReal
    hd : LE.le (NNDist.nndist x y) d
    ⊢ LE.le (NNDist.nndist (f x) (f y)) (HMul.hMul C (HPow.hPow d ↑r))
  -/
  exact hf.nndist_le_of_le hx hy hd
  /-
    🎉 no goals
  -/


theorem dist_le (hf : HolderOnWith C r f s) (hx : x ∈ s) (hy : y ∈ s) :
    dist (f x) (f y) ≤ C * dist x y ^ (r : ℝ) :=
  hf.dist_le_of_le hx hy le_rfl


theorem nndist_le_of_le (hf : HolderWith C r f) {x y : X} {d : ℝ≥0} (hd : nndist x y ≤ d) :
    nndist (f x) (f y) ≤ C * d ^ (r : ℝ) :=
  (hf.holderOnWith univ).nndist_le_of_le (mem_univ x) (mem_univ y) hd


theorem nndist_le (hf : HolderWith C r f) (x y : X) :
    nndist (f x) (f y) ≤ C * nndist x y ^ (r : ℝ) :=
  hf.nndist_le_of_le le_rfl


theorem dist_le_of_le (hf : HolderWith C r f) {x y : X} {d : ℝ} (hd : dist x y ≤ d) :
    dist (f x) (f y) ≤ C * d ^ (r : ℝ) :=
  (hf.holderOnWith univ).dist_le_of_le (mem_univ x) (mem_univ y) hd


theorem dist_le (hf : HolderWith C r f) (x y : X) : dist (f x) (f y) ≤ C * dist x y ^ (r : ℝ) :=
  hf.dist_le_of_le le_rfl


@[simp]
lemma holderWith_zero_iff : HolderWith 0 r f ↔ ∀ x₁ x₂, f x₁ = f x₂ := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : MetricSpace Y
    r : NNReal
    f : X → Y
    ⊢ Iff (HolderWith 0 r f) (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂))
  -/
  refine ⟨fun h x₁ x₂ => ?_, fun h x₁ x₂ => h x₁ x₂ ▸ ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : PseudoMetricSpace X
      inst✝ : MetricSpace Y
      r : NNReal
      f : X → Y
      h : HolderWith 0 r f
      x₁ x₂ : X
      ⊢ Eq (f x₁) (f x₂)
    -/
  · specialize h x₁ x₂
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : PseudoMetricSpace X
      inst✝ : MetricSpace Y
      r : NNReal
      f : X → Y
      x₁ x₂ : X
      h : LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑0) (HPow.hPow (EDist.edist  …
      ⊢ Eq (f x₁) (f x₂)
    -/
    simp [ENNReal.coe_zero, zero_mul, nonpos_iff_eq_zero, edist_eq_zero] at h
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : PseudoMetricSpace X
      inst✝ : MetricSpace Y
      r : NNReal
      f : X → Y
      x₁ x₂ : X
      h : Eq (f x₁) (f x₂)
      ⊢ Eq (f x₁) (f x₂)
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝¹ : PseudoMetricSpace X
      inst✝ : MetricSpace Y
      r : NNReal
      f : X → Y
      h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
      x₁ x₂ : X
      ⊢ LE.le (EDist.edist (f x₁) (f x₁)) (HMul.hMul (↑0) (HPow.hPow (EDist.edist x₁ …
    -/
  · simp only [edist_self, ENNReal.coe_zero, zero_mul, le_refl]
    /-
      🎉 no goals
    -/


lemma add (hf : HolderWith C r f) (hg : HolderWith C' r g) :
    HolderWith (C + C') r (f + g) := fun x₁ x₂ => by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : SeminormedAddCommGroup Y
    C C' r : NNReal
    f g : X → Y
    hf : HolderWith C r f
    hg : HolderWith C' r g
    x₁ x₂ : X
    ⊢ LE.le (EDist.edist (HAdd.hAdd f g x₁) (HAdd.hAdd f g x₂)) (HMul.hMul (↑(HAdd …
  -/
  refine le_trans (edist_add_add_le _ _ _ _) <| le_trans (add_le_add (hf x₁ x₂) (hg x₁ x₂)) ?_
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoMetricSpace X
    inst✝ : SeminormedAddCommGroup Y
    C C' r : NNReal
    f g : X → Y
    hf : HolderWith C r f
    hg : HolderWith C' r g
    x₁ x₂ : X
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (↑C) (HPow.hPow (EDist.edist x₁ x₂) ↑r)) (HMul.h …
  -/
  rw [coe_add, add_mul]
  /-
    🎉 no goals
  -/


lemma smul {α} [NormedDivisionRing α] [Module α Y] [BoundedSMul α Y] (a : α)
    (hf : HolderWith C r f) : HolderWith (C * ‖a‖₊) r (a • f) := fun x₁ x₂ => by
  rw [Pi.smul_apply, coe_mul, Pi.smul_apply, edist_smul₀, mul_comm (C : ℝ≥0∞),
    ENNReal.smul_def, smul_eq_mul, mul_assoc]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : PseudoMetricSpace X
    inst✝³ : SeminormedAddCommGroup Y
    C r : NNReal
    f : X → Y
    α : Type u_4
    inst✝² : NormedDivisionRing α
    inst✝¹ : Module α Y
    inst✝ : BoundedSMul α Y
    a : α
    hf : HolderWith C r f
    x₁ x₂ : X
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm a)) (EDist.edist (f x₁) (f x₂))) (HMul.hMu …
  -/
  gcongr
  /-
    case bc
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : PseudoMetricSpace X
    inst✝³ : SeminormedAddCommGroup Y
    C r : NNReal
    f : X → Y
    α : Type u_4
    inst✝² : NormedDivisionRing α
    inst✝¹ : Module α Y
    inst✝ : BoundedSMul α Y
    a : α
    hf : HolderWith C r f
    x₁ x₂ : X
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑C) (HPow.hPow (EDist.edist x₁ …
  -/
  exact hf x₁ x₂
  /-
    🎉 no goals
  -/


