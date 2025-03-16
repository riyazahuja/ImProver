/-- This is really an auxiliary result en route to `blimsup_cthickening_ae_le_of_eventually_mul_le`
(which is itself an auxiliary result en route to `blimsup_cthickening_mul_ae_eq`).

NB: The `: Set α` type ascription is present because of
https://github.com/leanprover-community/mathlib/issues/16932. -/
theorem blimsup_cthickening_ae_le_of_eventually_mul_le_aux (p : ℕ → Prop) {s : ℕ → Set α}
    (hs : ∀ i, IsClosed (s i)) {r₁ r₂ : ℕ → ℝ} (hr : Tendsto r₁ atTop (𝓝[>] 0)) (hrp : 0 ≤ r₁)
    {M : ℝ} (hM : 0 < M) (hM' : M < 1) (hMr : ∀ᶠ i in atTop, M * r₁ i ≤ r₂ i) :
    (blimsup (fun i => cthickening (r₁ i) (s i)) atTop p : Set α) ≤ᵐ[μ]
      (blimsup (fun i => cthickening (r₂ i) (s i)) atTop p : Set α) := by
  /- Sketch of proof:

  Assume that `p` is identically true for simplicity. Let `Y₁ i = cthickening (r₁ i) (s i)`, define
  `Y₂` similarly except using `r₂`, and let `(Z i) = ⋃_{j ≥ i} (Y₂ j)`. Our goal is equivalent to
  showing that `μ ((limsup Y₁) \ (Z i)) = 0` for all `i`.

  Assume for contradiction that `μ ((limsup Y₁) \ (Z i)) ≠ 0` for some `i` and let
  `W = (limsup Y₁) \ (Z i)`. Apply Lebesgue's density theorem to obtain a point `d` in `W` of
  density `1`. Since `d ∈ limsup Y₁`, there is a subsequence of `j ↦ Y₁ j`, indexed by
  `f 0 < f 1 < ...`, such that `d ∈ Y₁ (f j)` for all `j`. For each `j`, we may thus choose
  `w j ∈ s (f j)` such that `d ∈ B j`, where `B j = closedBall (w j) (r₁ (f j))`. Note that
  since `d` has density one, `μ (W ∩ (B j)) / μ (B j) → 1`.

  We obtain our contradiction by showing that there exists `η < 1` such that
  `μ (W ∩ (B j)) / μ (B j) ≤ η` for sufficiently large `j`. In fact we claim that `η = 1 - C⁻¹`
  is such a value where `C` is the scaling constant of `M⁻¹` for the uniformly locally doubling
  measure `μ`.

  To prove the claim, let `b j = closedBall (w j) (M * r₁ (f j))` and for given `j` consider the
  sets `b j` and `W ∩ (B j)`. These are both subsets of `B j` and are disjoint for large enough `j`
  since `M * r₁ j ≤ r₂ j` and thus `b j ⊆ Z i ⊆ Wᶜ`. We thus have:
  `μ (b j) + μ (W ∩ (B j)) ≤ μ (B j)`. Combining this with `μ (B j) ≤ C * μ (b j)` we obtain
  the required inequality. -/
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  set Y₁ : ℕ → Set α := fun i => cthickening (r₁ i) (s i)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup Y₁ Filter.atTop p) (Filter …
  -/
  set Y₂ : ℕ → Set α := fun i => cthickening (r₂ i) (s i)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup Y₁ Filter.atTop p) (Filter …
  -/
  let Z : ℕ → Set α := fun i => ⋃ (j) (_ : p j ∧ i ≤ j), Y₂ j
  suffices ∀ i, μ (atTop.blimsup Y₁ p \ Z i) = 0 by
    rwa [ae_le_set, @blimsup_eq_iInf_biSup_of_nat _ _ _ Y₂, iInf_eq_iInter, diff_iInter,
      measure_iUnion_null_iff]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    ⊢ ∀ (i : Nat), Eq (μ (SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i))) 0
  -/
  intros i
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    ⊢ Eq (μ (SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i))) 0
  -/
  set W := atTop.blimsup Y₁ p \ Z i
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    ⊢ Eq (μ W) 0
  -/
  by_contra contra
  obtain ⟨d, hd, hd'⟩ : ∃ d, d ∈ W ∧ ∀ {ι : Type _} {l : Filter ι} (w : ι → α) (δ : ι → ℝ),
      Tendsto δ l (𝓝[>] 0) → (∀ᶠ j in l, d ∈ closedBall (w j) (2 * δ j)) →
        Tendsto (fun j => μ (W ∩ closedBall (w j) (δ j)) / μ (closedBall (w j) (δ j))) l (𝓝 1) :=
    Measure.exists_mem_of_measure_ne_zero_of_ae contra
      (IsUnifLocDoublingMeasure.ae_tendsto_measure_inter_div μ W 2)
  /-
    case intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd : Membership.mem W d
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    ⊢ False
  -/
  replace hd : d ∈ blimsup Y₁ atTop p := ((mem_diff _).mp hd).1
  /-
    case intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    ⊢ False
  -/
  obtain ⟨f : ℕ → ℕ, hf⟩ := exists_forall_mem_of_hasBasis_mem_blimsup' atTop_basis hd
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf : ∀ (i : Nat), And (Membership.mem (Y₁ (f i)) d) (And (p (f i)) (Membership …
    ⊢ False
  -/
  simp only [forall_and] at hf
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf : And (∀ (x : Nat), Membership.mem (Y₁ (f x)) d) (And (∀ (x : Nat), p (f x) …
    ⊢ False
  -/
  obtain ⟨hf₀ : ∀ j, d ∈ cthickening (r₁ (f j)) (s (f j)), hf₁, hf₂ : ∀ j, j ≤ f j⟩ := hf
  have hf₃ : Tendsto f atTop atTop :=
    tendsto_atTop_atTop.mpr fun j => ⟨f j, fun i hi => (hf₂ j).trans (hi.trans <| hf₂ i)⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₀ : ∀ (j : Nat), Membership.mem (Metric.cthickening (r₁ (f j)) (s (f j))) d
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    ⊢ False
  -/
  replace hr : Tendsto (r₁ ∘ f) atTop (𝓝[>] 0) := hr.comp hf₃
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₀ : ∀ (j : Nat), Membership.mem (Metric.cthickening (r₁ (f j)) (s (f j))) d
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    ⊢ False
  -/
  replace hMr : ∀ᶠ j in atTop, M * r₁ (f j) ≤ r₂ (f j) := hf₃.eventually hMr
  replace hf₀ : ∀ j, ∃ w ∈ s (f j), d ∈ closedBall w (2 * r₁ (f j)) := by
    intro j
    specialize hrp (f j)
    rw [Pi.zero_apply] at hrp
    rcases eq_or_lt_of_le hrp with (hr0 | hrp')
    · specialize hf₀ j
      rw [← hr0, cthickening_zero, (hs (f j)).closure_eq] at hf₀
      exact ⟨d, hf₀, by simp [← hr0]⟩
    · simpa using mem_iUnion₂.mp (cthickening_subset_iUnion_closedBall_of_lt (s (f j))
        (by positivity) (lt_two_mul_self hrp') (hf₀ j))
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    hf₀ : ∀ (j : Nat), Exists fun w => And (Membership.mem (s (f j)) w) (Membershi …
    ⊢ False
  -/
  choose w hw hw' using hf₀
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type ?u.5847} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.T …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    ⊢ False
  -/
  let C := IsUnifLocDoublingMeasure.scalingConstantOf μ M⁻¹
  have hC : 0 < C :=
    lt_of_lt_of_le zero_lt_one (IsUnifLocDoublingMeasure.one_le_scalingConstantOf μ M⁻¹)
  suffices ∃ η < (1 : ℝ≥0),
      ∀ᶠ j in atTop, μ (W ∩ closedBall (w j) (r₁ (f j))) / μ (closedBall (w j) (r₁ (f j))) ≤ η by
    obtain ⟨η, hη, hη'⟩ := this
    replace hη' : 1 ≤ η := by
      simpa only [ENNReal.one_le_coe_iff] using
        le_of_tendsto (hd' w (fun j => r₁ (f j)) hr <| Eventually.of_forall hw') hη'
    exact (lt_self_iff_false _).mp (lt_of_lt_of_le hη hη')
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : LT.lt 0 C
    ⊢ Exists fun η => And (LT.lt η 1) (Filter.Eventually (fun j => LE.le (HDiv.hDi …
  -/
  refine ⟨1 - C⁻¹, tsub_lt_self zero_lt_one (inv_pos.mpr hC), ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : LT.lt 0 C
    ⊢ Filter.Eventually (fun j => LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.close …
  -/
  replace hC : C ≠ 0 := ne_of_gt hC
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    ⊢ Filter.Eventually (fun j => LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.close …
  -/
  let b : ℕ → Set α := fun j => closedBall (w j) (M * r₁ (f j))
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    ⊢ Filter.Eventually (fun j => LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.close …
  -/
  let B : ℕ → Set α := fun j => closedBall (w j) (r₁ (f j))
  have h₁ : ∀ j, b j ⊆ B j := fun j =>
    closedBall_subset_closedBall (mul_le_of_le_one_left (hrp (f j)) hM'.le)
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    ⊢ Filter.Eventually (fun j => LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.close …
  -/
  have h₂ : ∀ j, W ∩ B j ⊆ B j := fun j => inter_subset_right
  have h₃ : ∀ᶠ j in atTop, Disjoint (b j) (W ∩ B j) := by
    apply hMr.mp
    rw [eventually_atTop]
    refine
      ⟨i, fun j hj hj' => Disjoint.inf_right (B j) <| Disjoint.inf_right' (blimsup Y₁ atTop p) ?_⟩
    change Disjoint (b j) (Z i)ᶜ
    rw [disjoint_compl_right_iff_subset]
    refine (closedBall_subset_cthickening (hw j) (M * r₁ (f j))).trans
      ((cthickening_mono hj' _).trans fun a ha => ?_)
    simp only [Z, mem_iUnion, exists_prop]
    exact ⟨f j, ⟨hf₁ j, hj.le.trans (hf₂ j)⟩, ha⟩
  have h₄ : ∀ᶠ j in atTop, μ (B j) ≤ C * μ (b j) :=
    (hr.eventually (IsUnifLocDoublingMeasure.eventually_measure_le_scaling_constant_mul'
      μ M hM)).mono fun j hj => hj (w j)
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    ⊢ Filter.Eventually (fun j => LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.close …
  -/
  refine (h₃.and h₄).mono fun j hj₀ => ?_
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hj₀ : And (Disjoint (b j) (Inter.inter W (B j))) (LE.le (μ (B j)) (HMul.hMul ( …
    ⊢ LE.le (HDiv.hDiv (μ (Inter.inter W (Metric.closedBall (w j) (r₁ (f j))))) (μ …
  -/
  change μ (W ∩ B j) / μ (B j) ≤ ↑(1 - C⁻¹)
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hj₀ : And (Disjoint (b j) (Inter.inter W (B j))) (LE.le (μ (B j)) (HMul.hMul ( …
    ⊢ LE.le (HDiv.hDiv (μ (Inter.inter W (B j))) (μ (B j))) ↑(HSub.hSub 1 (Inv.inv …
  -/
  rcases eq_or_ne (μ (B j)) ∞ with (hB | hB); · simp [hB]
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    case intro.intro.intro.intro.intro.inr
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hj₀ : And (Disjoint (b j) (Inter.inter W (B j))) (LE.le (μ (B j)) (HMul.hMul ( …
    hB : Ne (μ (B j)) Top.top
    ⊢ LE.le (HDiv.hDiv (μ (Inter.inter W (B j))) (μ (B j))) ↑(HSub.hSub 1 (Inv.inv …
  -/
  apply ENNReal.div_le_of_le_mul
  /-
    case intro.intro.intro.intro.intro.inr.h
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hj₀ : And (Disjoint (b j) (Inter.inter W (B j))) (LE.le (μ (B j)) (HMul.hMul ( …
    hB : Ne (μ (B j)) Top.top
    ⊢ LE.le (μ (Inter.inter W (B j))) (HMul.hMul (↑(HSub.hSub 1 (Inv.inv C))) (μ ( …
  -/
  rw [ENNReal.coe_sub, ENNReal.coe_one, ENNReal.sub_mul fun _ _ => hB, one_mul]
  replace hB : ↑C⁻¹ * μ (B j) ≠ ∞ := by
    refine ENNReal.mul_ne_top ?_ hB
    rwa [ENNReal.coe_inv hC, Ne, ENNReal.inv_eq_top, ENNReal.coe_eq_zero]
  /-
    case intro.intro.intro.intro.intro.inr.h
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hj₀ : And (Disjoint (b j) (Inter.inter W (B j))) (LE.le (μ (B j)) (HMul.hMul ( …
    hB : Ne (HMul.hMul (↑(Inv.inv C)) (μ (B j))) Top.top
    ⊢ LE.le (μ (Inter.inter W (B j))) (HSub.hSub (μ (B j)) (HMul.hMul (↑(Inv.inv C …
  -/
  obtain ⟨hj₁ : Disjoint (b j) (W ∩ B j), hj₂ : μ (B j) ≤ C * μ (b j)⟩ := hj₀
  replace hj₂ : ↑C⁻¹ * μ (B j) ≤ μ (b j) := by
    rw [ENNReal.coe_inv hC, ← ENNReal.div_eq_inv_mul]
    exact ENNReal.div_le_of_le_mul' hj₂
  have hj₃ : ↑C⁻¹ * μ (B j) + μ (W ∩ B j) ≤ μ (B j) := by
    refine le_trans (add_le_add_right hj₂ _) ?_
    rw [← measure_union' hj₁ measurableSet_closedBall]
    exact measure_mono (union_subset (h₁ j) (h₂ j))
  /-
    case intro.intro.intro.intro.intro.inr.h.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hB : Ne (HMul.hMul (↑(Inv.inv C)) (μ (B j))) Top.top
    hj₁ : Disjoint (b j) (Inter.inter W (B j))
    hj₂ : LE.le (HMul.hMul (↑(Inv.inv C)) (μ (B j))) (μ (b j))
    hj₃ : LE.le (HAdd.hAdd (HMul.hMul (↑(Inv.inv C)) (μ (B j))) (μ (Inter.inter W  …
    ⊢ LE.le (μ (Inter.inter W (B j))) (HSub.hSub (μ (B j)) (HMul.hMul (↑(Inv.inv C …
  -/
  replace hj₃ := tsub_le_tsub_right hj₃ (↑C⁻¹ * μ (B j))
  /-
    case intro.intro.intro.intro.intro.inr.h.intro
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    hs : ∀ (i : Nat), IsClosed (s i)
    r₁ r₂ : Nat → Real
    hrp : LE.le 0 r₁
    M : Real
    hM : LT.lt 0 M
    hM' : LT.lt M 1
    Y₁ : Nat → Set α := fun i => Metric.cthickening (r₁ i) (s i)
    Y₂ : Nat → Set α := fun i => Metric.cthickening (r₂ i) (s i)
    Z : Nat → Set α := fun i => Set.iUnion fun j => Set.iUnion fun x => Y₂ j
    i : Nat
    W : Set α := SDiff.sdiff (Filter.blimsup Y₁ Filter.atTop p) (Z i)
    contra : Not (Eq (μ W) 0)
    d : α
    hd' : ∀ {ι : Type} {l : Filter ι} (w : ι → α) (δ : ι → Real), Filter.Tendsto δ …
    hd : Membership.mem (Filter.blimsup Y₁ Filter.atTop p) d
    f : Nat → Nat
    hf₁ : ∀ (x : Nat), p (f x)
    hf₂ : ∀ (j : Nat), LE.le j (f j)
    hf₃ : Filter.Tendsto f Filter.atTop Filter.atTop
    hr : Filter.Tendsto (Function.comp r₁ f) Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun j => LE.le (HMul.hMul M (r₁ (f j))) (r₂ (f j))) F …
    w : Nat → α
    hw : ∀ (j : Nat), Membership.mem (s (f j)) (w j)
    hw' : ∀ (j : Nat), Membership.mem (Metric.closedBall (w j) (HMul.hMul 2 (r₁ (f …
    C : NNReal := IsUnifLocDoublingMeasure.scalingConstantOf μ (Inv.inv M)
    hC : Ne C 0
    b : Nat → Set α := fun j => Metric.closedBall (w j) (HMul.hMul M (r₁ (f j)))
    B : Nat → Set α := fun j => Metric.closedBall (w j) (r₁ (f j))
    h₁ : ∀ (j : Nat), HasSubset.Subset (b j) (B j)
    h₂ : ∀ (j : Nat), HasSubset.Subset (Inter.inter W (B j)) (B j)
    h₃ : Filter.Eventually (fun j => Disjoint (b j) (Inter.inter W (B j))) Filter. …
    h₄ : Filter.Eventually (fun j => LE.le (μ (B j)) (HMul.hMul (↑C) (μ (b j)))) F …
    j : Nat
    hB : Ne (HMul.hMul (↑(Inv.inv C)) (μ (B j))) Top.top
    hj₁ : Disjoint (b j) (Inter.inter W (B j))
    hj₂ : LE.le (HMul.hMul (↑(Inv.inv C)) (μ (B j))) (μ (b j))
    hj₃ : LE.le (HSub.hSub (HAdd.hAdd (HMul.hMul (↑(Inv.inv C)) (μ (B j))) (μ (Int …
    ⊢ LE.le (μ (Inter.inter W (B j))) (HSub.hSub (μ (B j)) (HMul.hMul (↑(Inv.inv C …
  -/
  rwa [ENNReal.add_sub_cancel_left hB] at hj₃
  /-
    🎉 no goals
  -/


/-- This is really an auxiliary result en route to `blimsup_cthickening_mul_ae_eq`.

NB: The `: Set α` type ascription is present because of
https://github.com/leanprover-community/mathlib/issues/16932. -/
theorem blimsup_cthickening_ae_le_of_eventually_mul_le (p : ℕ → Prop) {s : ℕ → Set α} {M : ℝ}
    (hM : 0 < M) {r₁ r₂ : ℕ → ℝ} (hr : Tendsto r₁ atTop (𝓝[>] 0))
    (hMr : ∀ᶠ i in atTop, M * r₁ i ≤ r₂ i) :
    (blimsup (fun i => cthickening (r₁ i) (s i)) atTop p : Set α) ≤ᵐ[μ]
      (blimsup (fun i => cthickening (r₂ i) (s i)) atTop p : Set α) := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  let R₁ i := max 0 (r₁ i)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  let R₂ i := max 0 (r₂ i)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (r₁ i)) (r₂ i)) Filter.at …
    R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
    R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  have hRp : 0 ≤ R₁ := fun i => le_max_left 0 (r₁ i)
  replace hMr : ∀ᶠ i in atTop, M * R₁ i ≤ R₂ i := by
    refine hMr.mono fun i hi ↦ ?_
    rw [mul_max_of_nonneg _ _ hM.le, mul_zero]
    exact max_le_max (le_refl 0) hi
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
    R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
    hRp : LE.le 0 R₁
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  simp_rw [← cthickening_max_zero (r₁ _), ← cthickening_max_zero (r₂ _)]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r₁ r₂ : Nat → Real
    hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
    R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
    hRp : LE.le 0 R₁
    hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  rcases le_or_lt 1 M with hM' | hM'
    /-
      case inl
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LE.le 1 M
      ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
    -/
  · apply HasSubset.Subset.eventuallyLE
    /-
      case inl.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LE.le 1 M
      ⊢ HasSubset.Subset (Filter.blimsup (fun i => Metric.cthickening (Max.max 0 (r₁ …
    -/
    change _ ≤ _
    /-
      case inl.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LE.le 1 M
      ⊢ LE.le (Filter.blimsup (fun i => Metric.cthickening (Max.max 0 (r₁ i)) (s i)) …
    -/
    refine mono_blimsup' (hMr.mono fun i hi _ => cthickening_mono ?_ (s i))
    /-
      case inl.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LE.le 1 M
      i : Nat
      hi : LE.le (HMul.hMul M (R₁ i)) (R₂ i)
      x✝ : p i
      ⊢ LE.le (Max.max 0 (r₁ i)) (Max.max 0 (r₂ i))
    -/
    exact (le_mul_of_one_le_left (hRp i) hM').trans hi
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LT.lt M 1
      ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
    -/
  · simp only [← @cthickening_closure _ _ _ (s _)]
    /-
      case inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      M : Real
      hM : LT.lt 0 M
      r₁ r₂ : Nat → Real
      hr : Filter.Tendsto r₁ Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      R₁ : Nat → Real := fun i => Max.max 0 (r₁ i)
      R₂ : Nat → Real := fun i => Max.max 0 (r₂ i)
      hRp : LE.le 0 R₁
      hMr : Filter.Eventually (fun i => LE.le (HMul.hMul M (R₁ i)) (R₂ i)) Filter.at …
      hM' : LT.lt M 1
      ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
    -/
    have hs : ∀ i, IsClosed (closure (s i)) := fun i => isClosed_closure
    exact blimsup_cthickening_ae_le_of_eventually_mul_le_aux μ p hs
      (tendsto_nhds_max_right hr) hRp hM hM' hMr


/-- Given a sequence of subsets `sᵢ` of a metric space, together with a sequence of radii `rᵢ`
such that `rᵢ → 0`, the set of points which belong to infinitely many of the closed
`rᵢ`-thickenings of `sᵢ` is unchanged almost everywhere for a uniformly locally doubling measure if
the `rᵢ` are all scaled by a positive constant.

This lemma is a generalisation of Lemma 9 appearing on page 217 of
[J.W.S. Cassels, *Some metrical theorems in Diophantine approximation. I*](cassels1950).

See also `blimsup_thickening_mul_ae_eq`.

NB: The `: Set α` type ascription is present because of
https://github.com/leanprover-community/mathlib/issues/16932. -/
theorem blimsup_cthickening_mul_ae_eq (p : ℕ → Prop) (s : ℕ → Set α) {M : ℝ} (hM : 0 < M)
    (r : ℕ → ℝ) (hr : Tendsto r atTop (𝓝 0)) :
    (blimsup (fun i => cthickening (M * r i) (s i)) atTop p : Set α) =ᵐ[μ]
      (blimsup (fun i => cthickening (r i) (s i)) atTop p : Set α) := by
  have : ∀ (p : ℕ → Prop) {r : ℕ → ℝ} (_ : Tendsto r atTop (𝓝[>] 0)),
      (blimsup (fun i => cthickening (M * r i) (s i)) atTop p : Set α) =ᵐ[μ]
        (blimsup (fun i => cthickening (r i) (s i)) atTop p : Set α) := by
    clear p hr r; intro p r hr
    have hr' : Tendsto (fun i => M * r i) atTop (𝓝[>] 0) := by
      convert TendstoNhdsWithinIoi.const_mul hM hr <;> simp only [mul_zero]
    refine eventuallyLE_antisymm_iff.mpr ⟨?_, ?_⟩
    · exact blimsup_cthickening_ae_le_of_eventually_mul_le μ p (inv_pos.mpr hM) hr'
        (Eventually.of_forall fun i => by rw [inv_mul_cancel_left₀ hM.ne' (r i)])
    · exact blimsup_cthickening_ae_le_of_eventually_mul_le μ p hM hr
        (Eventually.of_forall fun i => le_refl _)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    this : ∀ (p : Nat → Prop) {r : Nat → Real}, Filter.Tendsto r Filter.atTop (nhd …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  let r' : ℕ → ℝ := fun i => if 0 < r i then r i else 1 / ((i : ℝ) + 1)
  have hr' : Tendsto r' atTop (𝓝[>] 0) := by
    refine tendsto_nhdsWithin_iff.mpr
      ⟨Tendsto.if' hr tendsto_one_div_add_atTop_nhds_zero_nat, Eventually.of_forall fun i => ?_⟩
    by_cases hi : 0 < r i
    · simp [r', hi]
    · simp only [r', hi, one_div, mem_Ioi, if_false, inv_pos]; positivity
  have h₀ : ∀ i, p i ∧ 0 < r i → cthickening (r i) (s i) = cthickening (r' i) (s i) := by
    rintro i ⟨-, hi⟩; congr! 1; change r i = ite (0 < r i) (r i) _; simp [hi]
  have h₁ : ∀ i, p i ∧ 0 < r i → cthickening (M * r i) (s i) = cthickening (M * r' i) (s i) := by
    rintro i ⟨-, hi⟩; simp only [r', hi, mul_ite, if_true]
  have h₂ : ∀ i, p i ∧ r i ≤ 0 → cthickening (M * r i) (s i) = cthickening (r i) (s i) := by
    rintro i ⟨-, hi⟩
    have hi' : M * r i ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hM.le hi
    rw [cthickening_of_nonpos hi, cthickening_of_nonpos hi']
  have hp : p = fun i => p i ∧ 0 < r i ∨ p i ∧ r i ≤ 0 := by
    ext i; simp [← and_or_left, lt_or_le 0 (r i)]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    this : ∀ (p : Nat → Prop) {r : Nat → Real}, Filter.Tendsto r Filter.atTop (nhd …
    r' : Nat → Real := fun i => ite (LT.lt 0 (r i)) (r i) (HDiv.hDiv 1 (HAdd.hAdd  …
    hr' : Filter.Tendsto r' Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    h₀ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (r i) (s  …
    h₁ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (HMul.hMu …
    h₂ : ∀ (i : Nat), And (p i) (LE.le (r i) 0) → Eq (Metric.cthickening (HMul.hMu …
    hp : Eq p fun i => Or (And (p i) (LT.lt 0 (r i))) (And (p i) (LE.le (r i) 0))
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  rw [hp, blimsup_or_eq_sup, blimsup_or_eq_sup]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    this : ∀ (p : Nat → Prop) {r : Nat → Real}, Filter.Tendsto r Filter.atTop (nhd …
    r' : Nat → Real := fun i => ite (LT.lt 0 (r i)) (r i) (HDiv.hDiv 1 (HAdd.hAdd  …
    hr' : Filter.Tendsto r' Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    h₀ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (r i) (s  …
    h₁ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (HMul.hMu …
    h₂ : ∀ (i : Nat), And (p i) (LE.le (r i) 0) → Eq (Metric.cthickening (HMul.hMu …
    hp : Eq p fun i => Or (And (p i) (LT.lt 0 (r i))) (And (p i) (LE.le (r i) 0))
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Max.max (Filter.blimsup (fun i => Metric. …
  -/
  simp only [sup_eq_union]
  rw [blimsup_congr (Eventually.of_forall h₀), blimsup_congr (Eventually.of_forall h₁),
    blimsup_congr (Eventually.of_forall h₂)]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    this : ∀ (p : Nat → Prop) {r : Nat → Real}, Filter.Tendsto r Filter.atTop (nhd …
    r' : Nat → Real := fun i => ite (LT.lt 0 (r i)) (r i) (HDiv.hDiv 1 (HAdd.hAdd  …
    hr' : Filter.Tendsto r' Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    h₀ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (r i) (s  …
    h₁ : ∀ (i : Nat), And (p i) (LT.lt 0 (r i)) → Eq (Metric.cthickening (HMul.hMu …
    h₂ : ∀ (i : Nat), And (p i) (LE.le (r i) 0) → Eq (Metric.cthickening (HMul.hMu …
    hp : Eq p fun i => Or (And (p i) (LT.lt 0 (r i))) (And (p i) (LE.le (r i) 0))
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union (Filter.blimsup (fun x => Met …
  -/
  exact ae_eq_set_union (this (fun i => p i ∧ 0 < r i) hr') (ae_eq_refl _)
  /-
    🎉 no goals
  -/


theorem blimsup_cthickening_ae_eq_blimsup_thickening {p : ℕ → Prop} {s : ℕ → Set α} {r : ℕ → ℝ}
    (hr : Tendsto r atTop (𝓝 0)) (hr' : ∀ᶠ i in atTop, p i → 0 < r i) :
    (blimsup (fun i => cthickening (r i) (s i)) atTop p : Set α) =ᵐ[μ]
      (blimsup (fun i => thickening (r i) (s i)) atTop p : Set α) := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthickeni …
  -/
  refine eventuallyLE_antisymm_iff.mpr ⟨?_, HasSubset.Subset.eventuallyLE (?_ : _ ≤ _)⟩
  · rw [eventuallyLE_congr (blimsup_cthickening_mul_ae_eq μ p s (@one_half_pos ℝ _) r hr).symm
      EventuallyEq.rfl]
    /-
      case refine_1
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      r : Nat → Real
      hr : Filter.Tendsto r Filter.atTop (nhds 0)
      hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
      ⊢ (MeasureTheory.ae μ).EventuallyLE (Filter.blimsup (fun i => Metric.cthickeni …
    -/
    apply HasSubset.Subset.eventuallyLE
    /-
      case refine_1.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      r : Nat → Real
      hr : Filter.Tendsto r Filter.atTop (nhds 0)
      hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
      ⊢ HasSubset.Subset (Filter.blimsup (fun i => Metric.cthickening (HMul.hMul (1  …
    -/
    change _ ≤ _
    /-
      case refine_1.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      r : Nat → Real
      hr : Filter.Tendsto r Filter.atTop (nhds 0)
      hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
      ⊢ LE.le (Filter.blimsup (fun i => Metric.cthickening (HMul.hMul (1 / 2) (r i)) …
    -/
    refine mono_blimsup' (hr'.mono fun i hi pi => cthickening_subset_thickening' (hi pi) ?_ (s i))
    /-
      case refine_1.h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      r : Nat → Real
      hr : Filter.Tendsto r Filter.atTop (nhds 0)
      hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
      i : Nat
      hi : p i → LT.lt 0 (r i)
      pi : p i
      ⊢ LT.lt (HMul.hMul (1 / 2) (r i)) (r i)
    -/
    nlinarith [hi pi]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : IsUnifLocDoublingMeasure μ
      p : Nat → Prop
      s : Nat → Set α
      r : Nat → Real
      hr : Filter.Tendsto r Filter.atTop (nhds 0)
      hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
      ⊢ LE.le (Filter.blimsup (fun i => Metric.thickening (r i) (s i)) Filter.atTop  …
    -/
  · exact mono_blimsup fun i _ => thickening_subset_cthickening _ _
    /-
      🎉 no goals
    -/


/-- An auxiliary result en route to `blimsup_thickening_mul_ae_eq`. -/
theorem blimsup_thickening_mul_ae_eq_aux (p : ℕ → Prop) (s : ℕ → Set α) {M : ℝ} (hM : 0 < M)
    (r : ℕ → ℝ) (hr : Tendsto r atTop (𝓝 0)) (hr' : ∀ᶠ i in atTop, p i → 0 < r i) :
    (blimsup (fun i => thickening (M * r i) (s i)) atTop p : Set α) =ᵐ[μ]
      (blimsup (fun i => thickening (r i) (s i)) atTop p : Set α) := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  have h₁ := blimsup_cthickening_ae_eq_blimsup_thickening (s := s) μ hr hr'
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
    h₁ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  have h₂ := blimsup_cthickening_mul_ae_eq μ p s hM r hr
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
    h₁ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    h₂ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  replace hr : Tendsto (fun i => M * r i) atTop (𝓝 0) := by convert hr.const_mul M; simp
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (r i)) Filter.atTop
    h₁ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    h₂ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    hr : Filter.Tendsto (fun i => HMul.hMul M (r i)) Filter.atTop (nhds 0)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  replace hr' : ∀ᶠ i in atTop, p i → 0 < M * r i := hr'.mono fun i hi hip ↦ mul_pos hM (hi hip)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    h₁ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    h₂ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    hr : Filter.Tendsto (fun i => HMul.hMul M (r i)) Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (HMul.hMul M (r i))) Filter.at …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  have h₃ := blimsup_cthickening_ae_eq_blimsup_thickening (s := s) μ hr hr'
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    h₁ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    h₂ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    hr : Filter.Tendsto (fun i => HMul.hMul M (r i)) Filter.atTop (nhds 0)
    hr' : Filter.Eventually (fun i => p i → LT.lt 0 (HMul.hMul M (r i))) Filter.at …
    h₃ : (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.cthick …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  exact h₃.symm.trans (h₂.trans h₁)
  /-
    🎉 no goals
  -/


/-- Given a sequence of subsets `sᵢ` of a metric space, together with a sequence of radii `rᵢ`
such that `rᵢ → 0`, the set of points which belong to infinitely many of the
`rᵢ`-thickenings of `sᵢ` is unchanged almost everywhere for a uniformly locally doubling measure if
the `rᵢ` are all scaled by a positive constant.

This lemma is a generalisation of Lemma 9 appearing on page 217 of
[J.W.S. Cassels, *Some metrical theorems in Diophantine approximation. I*](cassels1950).

See also `blimsup_cthickening_mul_ae_eq`.

NB: The `: Set α` type ascription is present because of
https://github.com/leanprover-community/mathlib/issues/16932. -/
theorem blimsup_thickening_mul_ae_eq (p : ℕ → Prop) (s : ℕ → Set α) {M : ℝ} (hM : 0 < M) (r : ℕ → ℝ)
    (hr : Tendsto r atTop (𝓝 0)) :
    (blimsup (fun i => thickening (M * r i) (s i)) atTop p : Set α) =ᵐ[μ]
      (blimsup (fun i => thickening (r i) (s i)) atTop p : Set α) := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  let q : ℕ → Prop := fun i => p i ∧ 0 < r i
  have h₁ : blimsup (fun i => thickening (r i) (s i)) atTop p =
      blimsup (fun i => thickening (r i) (s i)) atTop q := by
    refine blimsup_congr' (Eventually.of_forall fun i h => ?_)
    replace hi : 0 < r i := by contrapose! h; apply thickening_of_nonpos h
    simp only [q, hi, iff_self_and, imp_true_iff]
  have h₂ : blimsup (fun i => thickening (M * r i) (s i)) atTop p =
      blimsup (fun i => thickening (M * r i) (s i)) atTop q := by
    refine blimsup_congr' (Eventually.of_forall fun i h ↦ ?_)
    replace h : 0 < r i := by
      rw [← mul_pos_iff_of_pos_left hM]; contrapose! h; apply thickening_of_nonpos h
    simp only [q, h, iff_self_and, imp_true_iff]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    q : Nat → Prop := fun i => And (p i) (LT.lt 0 (r i))
    h₁ : Eq (Filter.blimsup (fun i => Metric.thickening (r i) (s i)) Filter.atTop  …
    h₂ : Eq (Filter.blimsup (fun i => Metric.thickening (HMul.hMul M (r i)) (s i)) …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  rw [h₁, h₂]
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : IsUnifLocDoublingMeasure μ
    p : Nat → Prop
    s : Nat → Set α
    M : Real
    hM : LT.lt 0 M
    r : Nat → Real
    hr : Filter.Tendsto r Filter.atTop (nhds 0)
    q : Nat → Prop := fun i => And (p i) (LT.lt 0 (r i))
    h₁ : Eq (Filter.blimsup (fun i => Metric.thickening (r i) (s i)) Filter.atTop  …
    h₂ : Eq (Filter.blimsup (fun i => Metric.thickening (HMul.hMul M (r i)) (s i)) …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.blimsup (fun i => Metric.thickenin …
  -/
  exact blimsup_thickening_mul_ae_eq_aux μ q s hM r hr (Eventually.of_forall fun i hi => hi.2)
  /-
    🎉 no goals
  -/

