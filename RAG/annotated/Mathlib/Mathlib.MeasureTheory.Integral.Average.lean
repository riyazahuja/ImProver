/-- Average value of an `ℝ≥0∞`-valued function `f` w.r.t. a measure `μ`, denoted `⨍⁻ x, f x ∂μ`.

It is equal to `(μ univ)⁻¹ * ∫⁻ x, f x ∂μ`, so it takes value zero if `μ` is an infinite measure. If
`μ` is a probability measure, then the average of any function is equal to its integral.

For the average on a set, use `⨍⁻ x in s, f x ∂μ`, defined as `⨍⁻ x, f x ∂(μ.restrict s)`. For the
average w.r.t. the volume, one can omit `∂volume`. -/
noncomputable def laverage (f : α → ℝ≥0∞) := ∫⁻ x, f x ∂(μ univ)⁻¹ • μ


/-- Average value of an `ℝ≥0∞`-valued function `f` w.r.t. a measure `μ`.

It is equal to `(μ univ)⁻¹ * ∫⁻ x, f x ∂μ`, so it takes value zero if `μ` is an infinite measure. If
`μ` is a probability measure, then the average of any function is equal to its integral.

For the average on a set, use `⨍⁻ x in s, f x ∂μ`, defined as `⨍⁻ x, f x ∂(μ.restrict s)`. For the
average w.r.t. the volume, one can omit `∂volume`. -/
notation3 "⨍⁻ "(...)", "r:60:(scoped f => f)" ∂"μ:70 => laverage μ r


/-- Average value of an `ℝ≥0∞`-valued function `f` w.r.t. to the standard measure.

It is equal to `(volume univ)⁻¹ * ∫⁻ x, f x`, so it takes value zero if the space has infinite
measure. In a probability space, the average of any function is equal to its integral.

For the average on a set, use `⨍⁻ x in s, f x`, defined as `⨍⁻ x, f x ∂(volume.restrict s)`. -/
notation3 "⨍⁻ "(...)", "r:60:(scoped f => laverage volume f) => r


/-- Average value of an `ℝ≥0∞`-valued function `f` w.r.t. a measure `μ` on a set `s`.

It is equal to `(μ s)⁻¹ * ∫⁻ x, f x ∂μ`, so it takes value zero if `s` has infinite measure. If `s`
has measure `1`, then the average of any function is equal to its integral.

For the average w.r.t. the volume, one can omit `∂volume`. -/
notation3 "⨍⁻ "(...)" in "s", "r:60:(scoped f => f)" ∂"μ:70 => laverage (Measure.restrict μ s) r


/-- Average value of an `ℝ≥0∞`-valued function `f` w.r.t. to the standard measure on a set `s`.

It is equal to `(volume s)⁻¹ * ∫⁻ x, f x`, so it takes value zero if `s` has infinite measure. If
`s` has measure `1`, then the average of any function is equal to its integral. -/
notation3 (prettyPrint := false)
  "⨍⁻ "(...)" in "s", "r:60:(scoped f => laverage Measure.restrict volume s f) => r


@[simp]
                                                       /-
                                                         α : Type u_1
                                                         m0 : MeasurableSpace α
                                                         μ : MeasureTheory.Measure α
                                                         ⊢ Eq (MeasureTheory.laverage μ fun _x => 0) 0
                                                       -/
theorem laverage_zero : ⨍⁻ _x, (0 : ℝ≥0∞) ∂μ = 0 := by rw [laverage, lintegral_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
                                                                                    /-
                                                                                      α : Type u_1
                                                                                      m0 : MeasurableSpace α
                                                                                      f : α → ENNReal
                                                                                      ⊢ Eq (MeasureTheory.laverage 0 fun x => f x) 0
                                                                                    -/
theorem laverage_zero_measure (f : α → ℝ≥0∞) : ⨍⁻ x, f x ∂(0 : Measure α) = 0 := by simp [laverage]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem laverage_eq' (f : α → ℝ≥0∞) : ⨍⁻ x, f x ∂μ = ∫⁻ x, f x ∂(μ univ)⁻¹ • μ := rfl


theorem laverage_eq (f : α → ℝ≥0∞) : ⨍⁻ x, f x ∂μ = (∫⁻ x, f x ∂μ) / μ univ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.laverage μ fun x => f x) (HDiv.hDiv (MeasureTheory.lintegr …
  -/
  rw [laverage_eq', lintegral_smul_measure, ENNReal.div_eq_inv_mul]
  /-
    🎉 no goals
  -/


theorem laverage_eq_lintegral [IsProbabilityMeasure μ] (f : α → ℝ≥0∞) :
                                      /-
                                        α : Type u_1
                                        m0 : MeasurableSpace α
                                        μ : MeasureTheory.Measure α
                                        inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                        f : α → ENNReal
                                        ⊢ Eq (MeasureTheory.laverage μ fun x => f x) (MeasureTheory.lintegral μ fun x  …
                                      -/
    ⨍⁻ x, f x ∂μ = ∫⁻ x, f x ∂μ := by rw [laverage, measure_univ, inv_one, one_smul]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem measure_mul_laverage [IsFiniteMeasure μ] (f : α → ℝ≥0∞) :
    μ univ * ⨍⁻ x, f x ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → ENNReal
    ⊢ Eq (HMul.hMul (μ Set.univ) (MeasureTheory.laverage μ fun x => f x)) (Measure …
  -/
  rcases eq_or_ne μ 0 with hμ | hμ
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → ENNReal
      hμ : Eq μ 0
      ⊢ Eq (HMul.hMul (μ Set.univ) (MeasureTheory.laverage μ fun x => f x)) (Measure …
    -/
  · rw [hμ, lintegral_zero_measure, laverage_zero_measure, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → ENNReal
      hμ : Ne μ 0
      ⊢ Eq (HMul.hMul (μ Set.univ) (MeasureTheory.laverage μ fun x => f x)) (Measure …
    -/
  · rw [laverage_eq, ENNReal.mul_div_cancel (measure_univ_ne_zero.2 hμ) (measure_ne_top _ _)]
    /-
      🎉 no goals
    -/


theorem setLaverage_eq (f : α → ℝ≥0∞) (s : Set α) :
                                                        /-
                                                          α : Type u_1
                                                          m0 : MeasurableSpace α
                                                          μ : MeasureTheory.Measure α
                                                          f : α → ENNReal
                                                          s : Set α
                                                          ⊢ Eq (MeasureTheory.laverage (μ.restrict s) fun x => f x) (HDiv.hDiv (MeasureT …
                                                        -/
    ⨍⁻ x in s, f x ∂μ = (∫⁻ x in s, f x ∂μ) / μ s := by rw [laverage_eq, restrict_apply_univ]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem setLaverage_eq' (f : α → ℝ≥0∞) (s : Set α) :
    ⨍⁻ x in s, f x ∂μ = ∫⁻ x, f x ∂(μ s)⁻¹ • μ.restrict s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    ⊢ Eq (MeasureTheory.laverage (μ.restrict s) fun x => f x) (MeasureTheory.linte …
  -/
  simp only [laverage_eq', restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem laverage_congr {f g : α → ℝ≥0∞} (h : f =ᵐ[μ] g) : ⨍⁻ x, f x ∂μ = ⨍⁻ x, g x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (MeasureTheory.laverage μ fun x => f x) (MeasureTheory.laverage μ fun x = …
  -/
  simp only [laverage_eq, lintegral_congr_ae h]
  /-
    🎉 no goals
  -/


theorem setLaverage_congr (h : s =ᵐ[μ] t) : ⨍⁻ x in s, f x ∂μ = ⨍⁻ x in t, f x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    f : α → ENNReal
    h : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Eq (MeasureTheory.laverage (μ.restrict s) fun x => f x) (MeasureTheory.laver …
  -/
  simp only [setLaverage_eq, setLIntegral_congr h, measure_congr h]
  /-
    🎉 no goals
  -/


theorem setLaverage_congr_fun (hs : MeasurableSet s) (h : ∀ᵐ x ∂μ, x ∈ s → f x = g x) :
    ⨍⁻ x in s, f x ∂μ = ⨍⁻ x in s, g x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → ENNReal
    hs : MeasurableSet s
    h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (MeasureT …
    ⊢ Eq (MeasureTheory.laverage (μ.restrict s) fun x => f x) (MeasureTheory.laver …
  -/
  simp only [laverage_eq, setLIntegral_congr_fun hs h]
  /-
    🎉 no goals
  -/


theorem laverage_lt_top (hf : ∫⁻ x, f x ∂μ ≠ ∞) : ⨍⁻ x, f x ∂μ < ∞ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ⊢ LT.lt (MeasureTheory.laverage μ fun x => f x) Top.top
  -/
  obtain rfl | hμ := eq_or_ne μ 0
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral 0 fun x => f x) Top.top
      ⊢ LT.lt (MeasureTheory.laverage 0 fun x => f x) Top.top
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      hμ : Ne μ 0
      ⊢ LT.lt (MeasureTheory.laverage μ fun x => f x) Top.top
    -/
  · rw [laverage_eq]
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      hμ : Ne μ 0
      ⊢ LT.lt (HDiv.hDiv (MeasureTheory.lintegral μ fun x => f x) (μ Set.univ)) Top. …
    -/
    exact div_lt_top hf (measure_univ_ne_zero.2 hμ)
    /-
      🎉 no goals
    -/


theorem setLaverage_lt_top : ∫⁻ x in s, f x ∂μ ≠ ∞ → ⨍⁻ x in s, f x ∂μ < ∞ :=
  laverage_lt_top


theorem laverage_add_measure :
    ⨍⁻ x, f x ∂(μ + ν) =
      μ univ / (μ univ + ν univ) * ⨍⁻ x, f x ∂μ + ν univ / (μ univ + ν univ) * ⨍⁻ x, f x ∂ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
  -/
  by_cases hμ : IsFiniteMeasure μ; swap
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : α → ENNReal
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
    -/
  · rw [not_isFiniteMeasure_iff] at hμ
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : α → ENNReal
      hμ : Eq (μ Set.univ) Top.top
      ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
    -/
    simp [laverage_eq, hμ]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → ENNReal
    hμ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
  -/
  by_cases hν : IsFiniteMeasure ν; swap
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : α → ENNReal
      hμ : MeasureTheory.IsFiniteMeasure μ
      hν : Not (MeasureTheory.IsFiniteMeasure ν)
      ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
    -/
  · rw [not_isFiniteMeasure_iff] at hν
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : α → ENNReal
      hμ : MeasureTheory.IsFiniteMeasure μ
      hν : Eq (ν Set.univ) Top.top
      ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
    -/
    simp [laverage_eq, hν]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → ENNReal
    hμ : MeasureTheory.IsFiniteMeasure μ
    hν : MeasureTheory.IsFiniteMeasure ν
    ⊢ Eq (MeasureTheory.laverage (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (HMul.hM …
  -/
  haveI := hμ; haveI := hν
  simp only [← ENNReal.mul_div_right_comm, measure_mul_laverage, ← ENNReal.add_div,
    ← lintegral_add_measure, ← Measure.add_apply, ← laverage_eq]


theorem measure_mul_setLaverage (f : α → ℝ≥0∞) (h : μ s ≠ ∞) :
    μ s * ⨍⁻ x in s, f x ∂μ = ∫⁻ x in s, f x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    h : Ne (μ s) Top.top
    ⊢ Eq (HMul.hMul (μ s) (MeasureTheory.laverage (μ.restrict s) fun x => f x)) (M …
  -/
  have := Fact.mk h.lt_top
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    h : Ne (μ s) Top.top
    this : Fact (LT.lt (μ s) Top.top)
    ⊢ Eq (HMul.hMul (μ s) (MeasureTheory.laverage (μ.restrict s) fun x => f x)) (M …
  -/
  rw [← measure_mul_laverage, restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem laverage_union (hd : AEDisjoint μ s t) (ht : NullMeasurableSet t μ) :
    ⨍⁻ x in s ∪ t, f x ∂μ =
      μ s / (μ s + μ t) * ⨍⁻ x in s, f x ∂μ + μ t / (μ s + μ t) * ⨍⁻ x in t, f x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    f : α → ENNReal
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    ⊢ Eq (MeasureTheory.laverage (μ.restrict (Union.union s t)) fun x => f x) (HAd …
  -/
  rw [restrict_union₀ hd ht, laverage_add_measure, restrict_apply_univ, restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem laverage_union_mem_openSegment (hd : AEDisjoint μ s t) (ht : NullMeasurableSet t μ)
    (hs₀ : μ s ≠ 0) (ht₀ : μ t ≠ 0) (hsμ : μ s ≠ ∞) (htμ : μ t ≠ ∞) :
    ⨍⁻ x in s ∪ t, f x ∂μ ∈ openSegment ℝ≥0∞ (⨍⁻ x in s, f x ∂μ) (⨍⁻ x in t, f x ∂μ) := by
  refine
    ⟨μ s / (μ s + μ t), μ t / (μ s + μ t), ENNReal.div_pos hs₀ <| add_ne_top.2 ⟨hsμ, htμ⟩,
      ENNReal.div_pos ht₀ <| add_ne_top.2 ⟨hsμ, htμ⟩, ?_, (laverage_union hd ht).symm⟩
  rw [← ENNReal.add_div,
    ENNReal.div_self (add_eq_zero.not.2 fun h => hs₀ h.1) (add_ne_top.2 ⟨hsμ, htμ⟩)]


theorem laverage_union_mem_segment (hd : AEDisjoint μ s t) (ht : NullMeasurableSet t μ)
    (hsμ : μ s ≠ ∞) (htμ : μ t ≠ ∞) :
    ⨍⁻ x in s ∪ t, f x ∂μ ∈ [⨍⁻ x in s, f x ∂μ -[ℝ≥0∞] ⨍⁻ x in t, f x ∂μ] := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    f : α → ENNReal
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    ⊢ Membership.mem (segment ENNReal (MeasureTheory.laverage (μ.restrict s) fun x …
  -/
  by_cases hs₀ : μ s = 0
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : Set α
      f : α → ENNReal
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hs₀ : Eq (μ s) 0
      ⊢ Membership.mem (segment ENNReal (MeasureTheory.laverage (μ.restrict s) fun x …
    -/
  · rw [← ae_eq_empty] at hs₀
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : Set α
      f : α → ENNReal
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hs₀ : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
      ⊢ Membership.mem (segment ENNReal (MeasureTheory.laverage (μ.restrict s) fun x …
    -/
    rw [restrict_congr_set (hs₀.union EventuallyEq.rfl), empty_union]
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : Set α
      f : α → ENNReal
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hs₀ : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
      ⊢ Membership.mem (segment ENNReal (MeasureTheory.laverage (μ.restrict s) fun x …
    -/
    exact right_mem_segment _ _ _
    /-
      🎉 no goals
    -/
  · refine
      ⟨μ s / (μ s + μ t), μ t / (μ s + μ t), zero_le _, zero_le _, ?_, (laverage_union hd ht).symm⟩
    rw [← ENNReal.add_div,
      ENNReal.div_self (add_eq_zero.not.2 fun h => hs₀ h.1) (add_ne_top.2 ⟨hsμ, htμ⟩)]


theorem laverage_mem_openSegment_compl_self [IsFiniteMeasure μ] (hs : NullMeasurableSet s μ)
    (hs₀ : μ s ≠ 0) (hsc₀ : μ sᶜ ≠ 0) :
    ⨍⁻ x, f x ∂μ ∈ openSegment ℝ≥0∞ (⨍⁻ x in s, f x ∂μ) (⨍⁻ x in sᶜ, f x ∂μ) := by
  simpa only [union_compl_self, restrict_univ] using
    laverage_union_mem_openSegment aedisjoint_compl_right hs.compl hs₀ hsc₀ (measure_ne_top _ _)
      (measure_ne_top _ _)


@[simp]
theorem laverage_const (μ : Measure α) [IsFiniteMeasure μ] [h : NeZero μ] (c : ℝ≥0∞) :
    ⨍⁻ _x, c ∂μ = c := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : NeZero μ
    c : ENNReal
    ⊢ Eq (MeasureTheory.laverage μ fun _x => c) c
  -/
  simp only [laverage, lintegral_const, measure_univ, mul_one]
  /-
    🎉 no goals
  -/


theorem setLaverage_const (hs₀ : μ s ≠ 0) (hs : μ s ≠ ∞) (c : ℝ≥0∞) : ⨍⁻ _x in s, c ∂μ = c := by
  simp only [setLaverage_eq, lintegral_const, Measure.restrict_apply, MeasurableSet.univ,
    univ_inter, div_eq_mul_inv, mul_assoc, ENNReal.mul_inv_cancel hs₀ hs, mul_one]


theorem laverage_one [IsFiniteMeasure μ] [NeZero μ] : ⨍⁻ _x, (1 : ℝ≥0∞) ∂μ = 1 :=
  laverage_const _ _


theorem setLaverage_one (hs₀ : μ s ≠ 0) (hs : μ s ≠ ∞) : ⨍⁻ _x in s, (1 : ℝ≥0∞) ∂μ = 1 :=
  setLaverage_const hs₀ hs _

-- Porting note: Dropped `simp` because of `simp` seeing through `1 : α → ℝ≥0∞` and applying
-- `lintegral_const`. This is suboptimal.

theorem lintegral_laverage (μ : Measure α) [IsFiniteMeasure μ] (f : α → ℝ≥0∞) :
    ∫⁻ _x, ⨍⁻ a, f a ∂μ ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun _x => MeasureTheory.laverage μ fun a => f  …
  -/
  obtain rfl | hμ := eq_or_ne μ 0
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      f : α → ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure 0
      ⊢ Eq (MeasureTheory.lintegral 0 fun _x => MeasureTheory.laverage 0 fun a => f  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [lintegral_const, laverage_eq,
      ENNReal.div_mul_cancel (measure_univ_ne_zero.2 hμ) (measure_ne_top _ _)]


theorem setLintegral_setLaverage (μ : Measure α) [IsFiniteMeasure μ] (f : α → ℝ≥0∞) (s : Set α) :
    ∫⁻ _x in s, ⨍⁻ a in s, f a ∂μ ∂μ = ∫⁻ x in s, f x ∂μ :=
  lintegral_laverage _ _


/-- Average value of a function `f` w.r.t. a measure `μ`, denoted `⨍ x, f x ∂μ`.

It is equal to `(μ univ).toReal⁻¹ • ∫ x, f x ∂μ`, so it takes value zero if `f` is not integrable or
if `μ` is an infinite measure. If `μ` is a probability measure, then the average of any function is
equal to its integral.

For the average on a set, use `⨍ x in s, f x ∂μ`, defined as `⨍ x, f x ∂(μ.restrict s)`. For the
average w.r.t. the volume, one can omit `∂volume`. -/
noncomputable def average (f : α → E) :=
  ∫ x, f x ∂(μ univ)⁻¹ • μ


/-- Average value of a function `f` w.r.t. a measure `μ`.

It is equal to `(μ univ).toReal⁻¹ • ∫ x, f x ∂μ`, so it takes value zero if `f` is not integrable or
if `μ` is an infinite measure. If `μ` is a probability measure, then the average of any function is
equal to its integral.

For the average on a set, use `⨍ x in s, f x ∂μ`, defined as `⨍ x, f x ∂(μ.restrict s)`. For the
average w.r.t. the volume, one can omit `∂volume`. -/
notation3 "⨍ "(...)", "r:60:(scoped f => f)" ∂"μ:70 => average μ r


/-- Average value of a function `f` w.r.t. to the standard measure.

It is equal to `(volume univ).toReal⁻¹ * ∫ x, f x`, so it takes value zero if `f` is not integrable
or if the space has infinite measure. In a probability space, the average of any function is equal
to its integral.

For the average on a set, use `⨍ x in s, f x`, defined as `⨍ x, f x ∂(volume.restrict s)`. -/
notation3 "⨍ "(...)", "r:60:(scoped f => average volume f) => r


/-- Average value of a function `f` w.r.t. a measure `μ` on a set `s`.

It is equal to `(μ s).toReal⁻¹ * ∫ x, f x ∂μ`, so it takes value zero if `f` is not integrable on
`s` or if `s` has infinite measure. If `s` has measure `1`, then the average of any function is
equal to its integral.

For the average w.r.t. the volume, one can omit `∂volume`. -/
notation3 "⨍ "(...)" in "s", "r:60:(scoped f => f)" ∂"μ:70 => average (Measure.restrict μ s) r


/-- Average value of a function `f` w.r.t. to the standard measure on a set `s`.

It is equal to `(volume s).toReal⁻¹ * ∫ x, f x`, so it takes value zero `f` is not integrable on `s`
or if `s` has infinite measure. If `s` has measure `1`, then the average of any function is equal to
its integral. -/
notation3 "⨍ "(...)" in "s", "r:60:(scoped f => average (Measure.restrict volume s) f) => r


@[simp]
                                                 /-
                                                   α : Type u_1
                                                   E : Type u_2
                                                   m0 : MeasurableSpace α
                                                   inst✝¹ : NormedAddCommGroup E
                                                   inst✝ : NormedSpace Real E
                                                   μ : MeasureTheory.Measure α
                                                   ⊢ Eq (MeasureTheory.average μ fun x => 0) 0
                                                 -/
theorem average_zero : ⨍ _, (0 : E) ∂μ = 0 := by rw [average, integral_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem average_zero_measure (f : α → E) : ⨍ x, f x ∂(0 : Measure α) = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    ⊢ Eq (MeasureTheory.average 0 fun x => f x) 0
  -/
  rw [average, smul_zero, integral_zero_measure]
  /-
    🎉 no goals
  -/


@[simp]
theorem average_neg (f : α → E) : ⨍ x, -f x ∂μ = -⨍ x, f x ∂μ :=
  integral_neg f


theorem average_eq' (f : α → E) : ⨍ x, f x ∂μ = ∫ x, f x ∂(μ univ)⁻¹ • μ :=
  rfl


theorem average_eq (f : α → E) : ⨍ x, f x ∂μ = (μ univ).toReal⁻¹ • ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    ⊢ Eq (MeasureTheory.average μ fun x => f x) (HSMul.hSMul (Inv.inv (μ Set.univ) …
  -/
  rw [average_eq', integral_smul_measure, ENNReal.toReal_inv]
  /-
    🎉 no goals
  -/


theorem average_eq_integral [IsProbabilityMeasure μ] (f : α → E) : ⨍ x, f x ∂μ = ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    f : α → E
    ⊢ Eq (MeasureTheory.average μ fun x => f x) (MeasureTheory.integral μ fun x => …
  -/
  rw [average, measure_univ, inv_one, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_smul_average [IsFiniteMeasure μ] (f : α → E) :
    (μ univ).toReal • ⨍ x, f x ∂μ = ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (MeasureTheory.average μ fun x => f x))  …
  -/
  rcases eq_or_ne μ 0 with hμ | hμ
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hμ : Eq μ 0
      ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (MeasureTheory.average μ fun x => f x))  …
    -/
  · rw [hμ, integral_zero_measure, average_zero_measure, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hμ : Ne μ 0
      ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (MeasureTheory.average μ fun x => f x))  …
    -/
  · rw [average_eq, smul_inv_smul₀]
    /-
      case inr.ha
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hμ : Ne μ 0
      ⊢ Ne (μ Set.univ).toReal 0
    -/
    refine (ENNReal.toReal_pos ?_ <| measure_ne_top _ _).ne'
    /-
      case inr.ha
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hμ : Ne μ 0
      ⊢ Ne (μ Set.univ) 0
    -/
    rwa [Ne, measure_univ_eq_zero]
    /-
      🎉 no goals
    -/


theorem setAverage_eq (f : α → E) (s : Set α) :
                                                               /-
                                                                 α : Type u_1
                                                                 E : Type u_2
                                                                 m0 : MeasurableSpace α
                                                                 inst✝¹ : NormedAddCommGroup E
                                                                 inst✝ : NormedSpace Real E
                                                                 μ : MeasureTheory.Measure α
                                                                 f : α → E
                                                                 s : Set α
                                                                 ⊢ Eq (MeasureTheory.average (μ.restrict s) fun x => f x) (HSMul.hSMul (Inv.inv …
                                                               -/
    ⨍ x in s, f x ∂μ = (μ s).toReal⁻¹ • ∫ x in s, f x ∂μ := by rw [average_eq, restrict_apply_univ]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem setAverage_eq' (f : α → E) (s : Set α) :
    ⨍ x in s, f x ∂μ = ∫ x, f x ∂(μ s)⁻¹ • μ.restrict s := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    ⊢ Eq (MeasureTheory.average (μ.restrict s) fun x => f x) (MeasureTheory.integr …
  -/
  simp only [average_eq', restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem average_congr {f g : α → E} (h : f =ᵐ[μ] g) : ⨍ x, f x ∂μ = ⨍ x, g x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f g : α → E
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (MeasureTheory.average μ fun x => f x) (MeasureTheory.average μ fun x =>  …
  -/
  simp only [average_eq, integral_congr_ae h]
  /-
    🎉 no goals
  -/


theorem setAverage_congr (h : s =ᵐ[μ] t) : ⨍ x in s, f x ∂μ = ⨍ x in t, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    s t : Set α
    f : α → E
    h : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Eq (MeasureTheory.average (μ.restrict s) fun x => f x) (MeasureTheory.averag …
  -/
  simp only [setAverage_eq, setIntegral_congr_set h, measure_congr h]
  /-
    🎉 no goals
  -/


theorem setAverage_congr_fun (hs : MeasurableSet s) (h : ∀ᵐ x ∂μ, x ∈ s → f x = g x) :
                                              /-
                                                α : Type u_1
                                                E : Type u_2
                                                m0 : MeasurableSpace α
                                                inst✝¹ : NormedAddCommGroup E
                                                inst✝ : NormedSpace Real E
                                                μ : MeasureTheory.Measure α
                                                s : Set α
                                                f g : α → E
                                                hs : MeasurableSet s
                                                h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (MeasureT …
                                                ⊢ Eq (MeasureTheory.average (μ.restrict s) fun x => f x) (MeasureTheory.averag …
                                              -/
    ⨍ x in s, f x ∂μ = ⨍ x in s, g x ∂μ := by simp only [average_eq, setIntegral_congr_ae hs h]
                                              /-
                                                🎉 no goals
                                              -/


theorem average_add_measure [IsFiniteMeasure μ] {ν : Measure α} [IsFiniteMeasure ν] {f : α → E}
    (hμ : Integrable f μ) (hν : Integrable f ν) :
    ⨍ x, f x ∂(μ + ν) =
      ((μ univ).toReal / ((μ univ).toReal + (ν univ).toReal)) • ⨍ x, f x ∂μ +
        ((ν univ).toReal / ((μ univ).toReal + (ν univ).toReal)) • ⨍ x, f x ∂ν := by
  simp only [div_eq_inv_mul, mul_smul, measure_smul_average, ← smul_add,
    ← integral_add_measure hμ hν, ← ENNReal.toReal_add (measure_ne_top μ _) (measure_ne_top ν _)]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    f : α → E
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    ⊢ Eq (MeasureTheory.average (HAdd.hAdd μ ν) fun x => f x) (HSMul.hSMul (Inv.in …
  -/
  rw [average_eq, Measure.add_apply]
  /-
    🎉 no goals
  -/


theorem average_pair [CompleteSpace E]
    {f : α → E} {g : α → F} (hfi : Integrable f μ) (hgi : Integrable g μ) :
    ⨍ x, (f x, g x) ∂μ = (⨍ x, f x ∂μ, ⨍ x, g x ∂μ) :=
  integral_pair hfi.to_average hgi.to_average


theorem measure_smul_setAverage (f : α → E) {s : Set α} (h : μ s ≠ ∞) :
    (μ s).toReal • ⨍ x in s, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    h : Ne (μ s) Top.top
    ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.average (μ.restrict s) fun x =>  …
  -/
  haveI := Fact.mk h.lt_top
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    h : Ne (μ s) Top.top
    this : Fact (LT.lt (μ s) Top.top)
    ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.average (μ.restrict s) fun x =>  …
  -/
  rw [← measure_smul_average, restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem average_union {f : α → E} {s t : Set α} (hd : AEDisjoint μ s t) (ht : NullMeasurableSet t μ)
    (hsμ : μ s ≠ ∞) (htμ : μ t ≠ ∞) (hfs : IntegrableOn f s μ) (hft : IntegrableOn f t μ) :
    ⨍ x in s ∪ t, f x ∂μ =
      ((μ s).toReal / ((μ s).toReal + (μ t).toReal)) • ⨍ x in s, f x ∂μ +
        ((μ t).toReal / ((μ s).toReal + (μ t).toReal)) • ⨍ x in t, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s t : Set α
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    ⊢ Eq (MeasureTheory.average (μ.restrict (Union.union s t)) fun x => f x) (HAdd …
  -/
  haveI := Fact.mk hsμ.lt_top; haveI := Fact.mk htμ.lt_top
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s t : Set α
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    this✝ : Fact (LT.lt (μ s) Top.top)
    this : Fact (LT.lt (μ t) Top.top)
    ⊢ Eq (MeasureTheory.average (μ.restrict (Union.union s t)) fun x => f x) (HAdd …
  -/
  rw [restrict_union₀ hd ht, average_add_measure hfs hft, restrict_apply_univ, restrict_apply_univ]
  /-
    🎉 no goals
  -/


theorem average_union_mem_openSegment {f : α → E} {s t : Set α} (hd : AEDisjoint μ s t)
    (ht : NullMeasurableSet t μ) (hs₀ : μ s ≠ 0) (ht₀ : μ t ≠ 0) (hsμ : μ s ≠ ∞) (htμ : μ t ≠ ∞)
    (hfs : IntegrableOn f s μ) (hft : IntegrableOn f t μ) :
    ⨍ x in s ∪ t, f x ∂μ ∈ openSegment ℝ (⨍ x in s, f x ∂μ) (⨍ x in t, f x ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s t : Set α
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hs₀ : Ne (μ s) 0
    ht₀ : Ne (μ t) 0
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    ⊢ Membership.mem (openSegment Real (MeasureTheory.average (μ.restrict s) fun x …
  -/
  replace hs₀ : 0 < (μ s).toReal := ENNReal.toReal_pos hs₀ hsμ
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s t : Set α
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    ht₀ : Ne (μ t) 0
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    hs₀ : LT.lt 0 (μ s).toReal
    ⊢ Membership.mem (openSegment Real (MeasureTheory.average (μ.restrict s) fun x …
  -/
  replace ht₀ : 0 < (μ t).toReal := ENNReal.toReal_pos ht₀ htμ
  exact mem_openSegment_iff_div.mpr
    ⟨(μ s).toReal, (μ t).toReal, hs₀, ht₀, (average_union hd ht hsμ htμ hfs hft).symm⟩


theorem average_union_mem_segment {f : α → E} {s t : Set α} (hd : AEDisjoint μ s t)
    (ht : NullMeasurableSet t μ) (hsμ : μ s ≠ ∞) (htμ : μ t ≠ ∞) (hfs : IntegrableOn f s μ)
    (hft : IntegrableOn f t μ) :
    ⨍ x in s ∪ t, f x ∂μ ∈ [⨍ x in s, f x ∂μ -[ℝ] ⨍ x in t, f x ∂μ] := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    s t : Set α
    hd : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hsμ : Ne (μ s) Top.top
    htμ : Ne (μ t) Top.top
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    ⊢ Membership.mem (segment Real (MeasureTheory.average (μ.restrict s) fun x =>  …
  -/
  by_cases hse : μ s = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      f : α → E
      s t : Set α
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hfs : MeasureTheory.IntegrableOn f s μ
      hft : MeasureTheory.IntegrableOn f t μ
      hse : Eq (μ s) 0
      ⊢ Membership.mem (segment Real (MeasureTheory.average (μ.restrict s) fun x =>  …
    -/
  · rw [← ae_eq_empty] at hse
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      f : α → E
      s t : Set α
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hfs : MeasureTheory.IntegrableOn f s μ
      hft : MeasureTheory.IntegrableOn f t μ
      hse : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
      ⊢ Membership.mem (segment Real (MeasureTheory.average (μ.restrict s) fun x =>  …
    -/
    rw [restrict_congr_set (hse.union EventuallyEq.rfl), empty_union]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      μ : MeasureTheory.Measure α
      f : α → E
      s t : Set α
      hd : MeasureTheory.AEDisjoint μ s t
      ht : MeasureTheory.NullMeasurableSet t μ
      hsμ : Ne (μ s) Top.top
      htμ : Ne (μ t) Top.top
      hfs : MeasureTheory.IntegrableOn f s μ
      hft : MeasureTheory.IntegrableOn f t μ
      hse : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
      ⊢ Membership.mem (segment Real (MeasureTheory.average (μ.restrict s) fun x =>  …
    -/
    exact right_mem_segment _ _ _
    /-
      🎉 no goals
    -/
  · refine
      mem_segment_iff_div.mpr
        ⟨(μ s).toReal, (μ t).toReal, ENNReal.toReal_nonneg, ENNReal.toReal_nonneg, ?_,
          (average_union hd ht hsμ htμ hfs hft).symm⟩
    calc
      0 < (μ s).toReal := ENNReal.toReal_pos hse hsμ
      _ ≤ _ := le_add_of_nonneg_right ENNReal.toReal_nonneg


theorem average_mem_openSegment_compl_self [IsFiniteMeasure μ] {f : α → E} {s : Set α}
    (hs : NullMeasurableSet s μ) (hs₀ : μ s ≠ 0) (hsc₀ : μ sᶜ ≠ 0) (hfi : Integrable f μ) :
    ⨍ x, f x ∂μ ∈ openSegment ℝ (⨍ x in s, f x ∂μ) (⨍ x in sᶜ, f x ∂μ) := by
  simpa only [union_compl_self, restrict_univ] using
    average_union_mem_openSegment aedisjoint_compl_right hs.compl hs₀ hsc₀ (measure_ne_top _ _)
      (measure_ne_top _ _) hfi.integrableOn hfi.integrableOn


@[simp]
theorem average_const (μ : Measure α) [IsFiniteMeasure μ] [h : NeZero μ] (c : E) :
    ⨍ _x, c ∂μ = c := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : NeZero μ
    c : E
    ⊢ Eq (MeasureTheory.average μ fun _x => c) c
  -/
  rw [average, integral_const, measure_univ, ENNReal.one_toReal, one_smul]
  /-
    🎉 no goals
  -/


theorem setAverage_const {s : Set α} (hs₀ : μ s ≠ 0) (hs : μ s ≠ ∞) (c : E) :
    ⨍ _ in s, c ∂μ = c :=
  have := NeZero.mk hs₀; have := Fact.mk hs.lt_top; average_const _ _


theorem integral_average (μ : Measure α) [IsFiniteMeasure μ] (f : α → E) :
                                            /-
                                              α : Type u_1
                                              E : Type u_2
                                              m0 : MeasurableSpace α
                                              inst✝³ : NormedAddCommGroup E
                                              inst✝² : NormedSpace Real E
                                              inst✝¹ : CompleteSpace E
                                              μ : MeasureTheory.Measure α
                                              inst✝ : MeasureTheory.IsFiniteMeasure μ
                                              f : α → E
                                              ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.average μ fun a => f a)  …
                                            -/
    ∫ _, ⨍ a, f a ∂μ ∂μ = ∫ x, f x ∂μ := by simp
                                            /-
                                              🎉 no goals
                                            -/


theorem setIntegral_setAverage (μ : Measure α) [IsFiniteMeasure μ] (f : α → E) (s : Set α) :
    ∫ _ in s, ⨍ a in s, f a ∂μ ∂μ = ∫ x in s, f x ∂μ :=
  integral_average _ _


theorem integral_sub_average (μ : Measure α) [IsFiniteMeasure μ] (f : α → E) :
    ∫ x, f x - ⨍ a, f a ∂μ ∂μ = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (f x) (MeasureTheory.average …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (f x) (MeasureTheory.average …
    -/
  · rw [integral_sub hf (integrable_const _), integral_average, sub_self]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hf : Not (MeasureTheory.Integrable f μ)
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (f x) (MeasureTheory.average …
  -/
  refine integral_undef fun h => hf ?_
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hf : Not (MeasureTheory.Integrable f μ)
    h : MeasureTheory.Integrable (fun x => HSub.hSub (f x) (MeasureTheory.average  …
    ⊢ MeasureTheory.Integrable f μ
  -/
  convert h.add (integrable_const (⨍ a, f a ∂μ))
  /-
    case h.e'_6
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hf : Not (MeasureTheory.Integrable f μ)
    h : MeasureTheory.Integrable (fun x => HSub.hSub (f x) (MeasureTheory.average  …
    ⊢ Eq f (HAdd.hAdd (fun x => HSub.hSub (f x) (MeasureTheory.average μ fun a =>  …
  -/
  exact (sub_add_cancel _ _).symm
  /-
    🎉 no goals
  -/


theorem setAverage_sub_setAverage (hs : μ s ≠ ∞) (f : α → E) :
    ∫ x in s, f x - ⨍ a in s, f a ∂μ ∂μ = 0 :=
  haveI : Fact (μ s < ∞) := ⟨lt_top_iff_ne_top.2 hs⟩
  integral_sub_average _ _


theorem integral_average_sub [IsFiniteMeasure μ] (hf : Integrable f μ) :
    ∫ x, ⨍ a, f a ∂μ - f x ∂μ = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (MeasureTheory.average μ fun …
  -/
  rw [integral_sub (integrable_const _) hf, integral_average, sub_self]
  /-
    🎉 no goals
  -/


theorem setIntegral_setAverage_sub (hs : μ s ≠ ∞) (hf : IntegrableOn f s μ) :
    ∫ x in s, ⨍ a in s, f a ∂μ - f x ∂μ = 0 :=
  haveI : Fact (μ s < ∞) := ⟨lt_top_iff_ne_top.2 hs⟩
  integral_average_sub hf


theorem ofReal_average {f : α → ℝ} (hf : Integrable f μ) (hf₀ : 0 ≤ᵐ[μ] f) :
    ENNReal.ofReal (⨍ x, f x ∂μ) = (∫⁻ x, ENNReal.ofReal (f x) ∂μ) / μ univ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf₀ : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.average μ fun x => f x)) (HDiv.hDiv (Measu …
  -/
  obtain rfl | hμ := eq_or_ne μ 0
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      f : α → Real
      hf : MeasureTheory.Integrable f 0
      hf₀ : (MeasureTheory.ae 0).EventuallyLE 0 f
      ⊢ Eq (ENNReal.ofReal (MeasureTheory.average 0 fun x => f x)) (HDiv.hDiv (Measu …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [average_eq, smul_eq_mul, ← toReal_inv, ofReal_mul toReal_nonneg,
      ofReal_toReal (inv_ne_top.2 <| measure_univ_ne_zero.2 hμ),
      ofReal_integral_eq_lintegral_ofReal hf hf₀, ENNReal.div_eq_inv_mul]


theorem ofReal_setAverage {f : α → ℝ} (hf : IntegrableOn f s μ) (hf₀ : 0 ≤ᵐ[μ.restrict s] f) :
    ENNReal.ofReal (⨍ x in s, f x ∂μ) = (∫⁻ x in s, ENNReal.ofReal (f x) ∂μ) / μ s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hf : MeasureTheory.IntegrableOn f s μ
    hf₀ : (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 f
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.average (μ.restrict s) fun x => f x)) (HDi …
  -/
  simpa using ofReal_average hf hf₀
  /-
    🎉 no goals
  -/


theorem toReal_laverage {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hf' : ∀ᵐ x ∂μ, f x ≠ ∞) :
    (⨍⁻ x, f x ∂μ).toReal = ⨍ x, (f x).toReal ∂μ := by
    rw [average_eq, laverage_eq, smul_eq_mul, toReal_div, div_eq_inv_mul, ←
      integral_toReal hf (hf'.mono fun _ => lt_top_iff_ne_top.2)]


theorem toReal_setLaverage {f : α → ℝ≥0∞} (hf : AEMeasurable f (μ.restrict s))
    (hf' : ∀ᵐ x ∂μ.restrict s, f x ≠ ∞) :
    (⨍⁻ x in s, f x ∂μ).toReal = ⨍ x in s, (f x).toReal ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hf : AEMeasurable f (μ.restrict s)
    hf' : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae (μ.restr …
    ⊢ Eq (MeasureTheory.laverage (μ.restrict s) fun x => f x).toReal (MeasureTheor …
  -/
  simpa [laverage_eq] using toReal_laverage hf hf'
  /-
    🎉 no goals
  -/


/-- **First moment method**. An integrable function is smaller than its mean on a set of positive
measure. -/
theorem measure_le_setAverage_pos (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞) (hf : IntegrableOn f s μ) :
    0 < μ ({x ∈ s | f x ≤ ⨍ a in s, f a ∂μ}) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : MeasureTheory.IntegrableOn f s μ
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (f x) (MeasureThe …
  -/
  refine pos_iff_ne_zero.2 fun H => ?_
  replace H : (μ.restrict s) {x | f x ≤ ⨍ a in s, f a ∂μ} = 0 := by
    rwa [restrict_apply₀, inter_comm]
    exact AEStronglyMeasurable.nullMeasurableSet_le hf.1 aestronglyMeasurable_const
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : MeasureTheory.IntegrableOn f s μ
    H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
    ⊢ False
  -/
  haveI := Fact.mk hμ₁.lt_top
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : MeasureTheory.IntegrableOn f s μ
    H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
    this : Fact (LT.lt (μ s) Top.top)
    ⊢ False
  -/
  refine (integral_sub_average (μ.restrict s) f).not_gt ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : MeasureTheory.IntegrableOn f s μ
    H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
    this : Fact (LT.lt (μ s) Top.top)
    ⊢ LT.lt 0 (MeasureTheory.integral (μ.restrict s) fun x => HSub.hSub (f x) (Mea …
  -/
  refine (setIntegral_pos_iff_support_of_nonneg_ae ?_ ?_).2 ?_
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 fun x => HSub.hSub (f x) (M …
    -/
  · refine measure_mono_null (fun x hx ↦ ?_) H
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      x : α
      hx : Membership.mem (HasCompl.compl (setOf fun x => (fun x => LE.le (0 x) ((fu …
      ⊢ Membership.mem (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.restric …
    -/
    simp only [Pi.zero_apply, sub_nonneg, mem_compl_iff, mem_setOf_eq, not_le] at hx
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      x : α
      hx : LT.lt (f x) (MeasureTheory.average (μ.restrict s) fun a => f a)
      ⊢ Membership.mem (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.restric …
    -/
    exact hx.le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ MeasureTheory.IntegrableOn (fun x => HSub.hSub (f x) (MeasureTheory.average  …
    -/
  · exact hf.sub (integrableOn_const.2 <| Or.inr <| lt_top_iff_ne_top.2 hμ₁)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ LT.lt 0 (μ (Inter.inter (Function.support fun x => HSub.hSub (f x) (MeasureT …
    -/
  · rwa [pos_iff_ne_zero, inter_comm, ← diff_compl, ← diff_inter_self_eq_diff, measure_diff_null]
    /-
      case refine_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ Eq (μ (Inter.inter (HasCompl.compl (Function.support fun x => HSub.hSub (f x …
    -/
    refine measure_mono_null ?_ (measure_inter_eq_zero_of_restrict H)
    /-
      case refine_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : MeasureTheory.IntegrableOn f s μ
      H : Eq ((μ.restrict s) (setOf fun x => LE.le (f x) (MeasureTheory.average (μ.r …
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl (Function.support fun x => HSu …
    -/
    exact inter_subset_inter_left _ fun a ha => (sub_eq_zero.1 <| of_not_not ha).le
    /-
      🎉 no goals
    -/


/-- **First moment method**. An integrable function is greater than its mean on a set of positive
measure. -/
theorem measure_setAverage_le_pos (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞) (hf : IntegrableOn f s μ) :
    0 < μ ({x ∈ s | ⨍ a in s, f a ∂μ ≤ f x}) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : MeasureTheory.IntegrableOn f s μ
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.av …
  -/
  simpa [integral_neg, neg_div] using measure_le_setAverage_pos hμ hμ₁ hf.neg
  /-
    🎉 no goals
  -/


/-- **First moment method**. The minimum of an integrable function is smaller than its mean. -/
theorem exists_le_setAverage (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞) (hf : IntegrableOn f s μ) :
    ∃ x ∈ s, f x ≤ ⨍ a in s, f a ∂μ :=
  let ⟨x, hx, h⟩ := nonempty_of_measure_ne_zero (measure_le_setAverage_pos hμ hμ₁ hf).ne'
  ⟨x, hx, h⟩


/-- **First moment method**. The maximum of an integrable function is greater than its mean. -/
theorem exists_setAverage_le (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞) (hf : IntegrableOn f s μ) :
    ∃ x ∈ s, ⨍ a in s, f a ∂μ ≤ f x :=
  let ⟨x, hx, h⟩ := nonempty_of_measure_ne_zero (measure_setAverage_le_pos hμ hμ₁ hf).ne'
  ⟨x, hx, h⟩


/-- **First moment method**. An integrable function is smaller than its mean on a set of positive
measure. -/
theorem measure_le_average_pos (hμ : μ ≠ 0) (hf : Integrable f μ) :
    0 < μ {x | f x ≤ ⨍ a, f a ∂μ} := by
  simpa using measure_le_setAverage_pos (Measure.measure_univ_ne_zero.2 hμ) (measure_ne_top _ _)
    hf.integrableOn


/-- **First moment method**. An integrable function is greater than its mean on a set of positive
measure. -/
theorem measure_average_le_pos (hμ : μ ≠ 0) (hf : Integrable f μ) :
    0 < μ {x | ⨍ a, f a ∂μ ≤ f x} := by
  simpa using measure_setAverage_le_pos (Measure.measure_univ_ne_zero.2 hμ) (measure_ne_top _ _)
    hf.integrableOn


/-- **First moment method**. The minimum of an integrable function is smaller than its mean. -/
theorem exists_le_average (hμ : μ ≠ 0) (hf : Integrable f μ) : ∃ x, f x ≤ ⨍ a, f a ∂μ :=
  let ⟨x, hx⟩ := nonempty_of_measure_ne_zero (measure_le_average_pos hμ hf).ne'
  ⟨x, hx⟩


/-- **First moment method**. The maximum of an integrable function is greater than its mean. -/
theorem exists_average_le (hμ : μ ≠ 0) (hf : Integrable f μ) : ∃ x, ⨍ a, f a ∂μ ≤ f x :=
  let ⟨x, hx⟩ := nonempty_of_measure_ne_zero (measure_average_le_pos hμ hf).ne'
  ⟨x, hx⟩


/-- **First moment method**. The minimum of an integrable function is smaller than its mean, while
avoiding a null set. -/
theorem exists_not_mem_null_le_average (hμ : μ ≠ 0) (hf : Integrable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ f x ≤ ⨍ a, f a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : MeasureTheory.Integrable f μ
    hN : Eq (μ N) 0
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.a …
  -/
  have := measure_le_average_pos hμ hf
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : MeasureTheory.Integrable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (setOf fun x => LE.le (f x) (MeasureTheory.average μ fun a = …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.a …
  -/
  rw [← measure_diff_null hN] at this
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : MeasureTheory.Integrable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (f x) (MeasureTheory.aver …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.a …
  -/
  obtain ⟨x, hx, hxN⟩ := nonempty_of_measure_ne_zero this.ne'
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : MeasureTheory.Integrable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (f x) (MeasureTheory.aver …
    x : α
    hx : Membership.mem (setOf fun x => LE.le (f x) (MeasureTheory.average μ fun a …
    hxN : Not (Membership.mem N x)
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.a …
  -/
  exact ⟨x, hxN, hx⟩
  /-
    🎉 no goals
  -/


/-- **First moment method**. The maximum of an integrable function is greater than its mean, while
avoiding a null set. -/
theorem exists_not_mem_null_average_le (hμ : μ ≠ 0) (hf : Integrable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ ⨍ a, f a ∂μ ≤ f x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : MeasureTheory.Integrable f μ
    hN : Eq (μ N) 0
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (MeasureTheory.average …
  -/
  simpa [integral_neg, neg_div] using exists_not_mem_null_le_average hμ hf.neg hN
  /-
    🎉 no goals
  -/


/-- **First moment method**. An integrable function is smaller than its integral on a set of
positive measure. -/
theorem measure_le_integral_pos (hf : Integrable f μ) : 0 < μ {x | f x ≤ ∫ a, f a ∂μ} := by
  simpa only [average_eq_integral] using
    measure_le_average_pos (IsProbabilityMeasure.ne_zero μ) hf


/-- **First moment method**. An integrable function is greater than its integral on a set of
positive measure. -/
theorem measure_integral_le_pos (hf : Integrable f μ) : 0 < μ {x | ∫ a, f a ∂μ ≤ f x} := by
  simpa only [average_eq_integral] using
    measure_average_le_pos (IsProbabilityMeasure.ne_zero μ) hf


/-- **First moment method**. The minimum of an integrable function is smaller than its integral. -/
theorem exists_le_integral (hf : Integrable f μ) : ∃ x, f x ≤ ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : MeasureTheory.Integrable f μ
    ⊢ Exists fun x => LE.le (f x) (MeasureTheory.integral μ fun a => f a)
  -/
  simpa only [average_eq_integral] using exists_le_average (IsProbabilityMeasure.ne_zero μ) hf
  /-
    🎉 no goals
  -/


/-- **First moment method**. The maximum of an integrable function is greater than its integral. -/
theorem exists_integral_le (hf : Integrable f μ) : ∃ x, ∫ a, f a ∂μ ≤ f x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : MeasureTheory.Integrable f μ
    ⊢ Exists fun x => LE.le (MeasureTheory.integral μ fun a => f a) (f x)
  -/
  simpa only [average_eq_integral] using exists_average_le (IsProbabilityMeasure.ne_zero μ) hf
  /-
    🎉 no goals
  -/


/-- **First moment method**. The minimum of an integrable function is smaller than its integral,
while avoiding a null set. -/
theorem exists_not_mem_null_le_integral (hf : Integrable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ f x ≤ ∫ a, f a ∂μ := by
  simpa only [average_eq_integral] using
    exists_not_mem_null_le_average (IsProbabilityMeasure.ne_zero μ) hf hN


/-- **First moment method**. The maximum of an integrable function is greater than its integral,
while avoiding a null set. -/
theorem exists_not_mem_null_integral_le (hf : Integrable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ ∫ a, f a ∂μ ≤ f x := by
  simpa only [average_eq_integral] using
    exists_not_mem_null_average_le (IsProbabilityMeasure.ne_zero μ) hf hN


/-- **First moment method**. A measurable function is smaller than its mean on a set of positive
measure. -/
theorem measure_le_setLaverage_pos (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞)
    (hf : AEMeasurable f (μ.restrict s)) : 0 < μ {x ∈ s | f x ≤ ⨍⁻ a in s, f a ∂μ} := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (f x) (MeasureThe …
  -/
  obtain h | h := eq_or_ne (∫⁻ a in s, f a ∂μ) ∞
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hμ : Ne (μ s) 0
      hμ₁ : Ne (μ s) Top.top
      hf : AEMeasurable f (μ.restrict s)
      h : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
      ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (f x) (MeasureThe …
    -/
  · simpa [mul_top, hμ₁, laverage, h, top_div_of_ne_top hμ₁, pos_iff_ne_zero] using hμ
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (f x) (MeasureThe …
  -/
  have := measure_le_setAverage_pos hμ hμ₁ (integrable_toReal_of_lintegral_ne_top hf h)
  rw [← setOf_inter_eq_sep, ← Measure.restrict_apply₀
    (hf.aestronglyMeasurable.nullMeasurableSet_le aestronglyMeasurable_const)]
  rw [← setOf_inter_eq_sep, ← Measure.restrict_apply₀
    (hf.ennreal_toReal.aestronglyMeasurable.nullMeasurableSet_le aestronglyMeasurable_const),
    ← measure_diff_null (measure_eq_top_of_lintegral_ne_top hf h)] at this
  /-
    case inr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    ⊢ LT.lt 0 ((μ.restrict s) (setOf fun a => LE.le (f a) (MeasureTheory.laverage  …
  -/
  refine this.trans_le (measure_mono ?_)
  /-
    case inr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    ⊢ HasSubset.Subset (SDiff.sdiff (setOf fun a => LE.le (f a).toReal (MeasureThe …
  -/
  rintro x ⟨hfx, hx⟩
  /-
    case inr.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    x : α
    hfx : Membership.mem (setOf fun a => LE.le (f a).toReal (MeasureTheory.average …
    hx : Not (Membership.mem (setOf fun x => Eq (f x) Top.top) x)
    ⊢ Membership.mem (setOf fun a => LE.le (f a) (MeasureTheory.laverage (μ.restri …
  -/
  dsimp at hfx
  /-
    case inr.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    x : α
    hfx : LE.le (f x).toReal (MeasureTheory.average (μ.restrict s) fun a => (f a). …
    hx : Not (Membership.mem (setOf fun x => Eq (f x) Top.top) x)
    ⊢ Membership.mem (setOf fun a => LE.le (f a) (MeasureTheory.laverage (μ.restri …
  -/
  rwa [← toReal_laverage hf, toReal_le_toReal hx (setLaverage_lt_top h).ne] at hfx
  /-
    case inr.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    x : α
    hfx : LE.le (f x).toReal (MeasureTheory.average (μ.restrict s) fun a => (f a). …
    hx : Not (Membership.mem (setOf fun x => Eq (f x) Top.top) x)
    ⊢ Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae (μ.restrict  …
  -/
  simp_rw [ae_iff, not_ne_iff]
  /-
    case inr.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hμ₁ : Ne (μ s) Top.top
    hf : AEMeasurable f (μ.restrict s)
    h : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (f a).toReal …
    x : α
    hfx : LE.le (f x).toReal (MeasureTheory.average (μ.restrict s) fun a => (f a). …
    hx : Not (Membership.mem (setOf fun x => Eq (f x) Top.top) x)
    ⊢ Eq ((μ.restrict s) (setOf fun a => Eq (f a) Top.top)) 0
  -/
  exact measure_eq_top_of_lintegral_ne_top hf h
  /-
    🎉 no goals
  -/


/-- **First moment method**. A measurable function is greater than its mean on a set of positive
measure. -/
theorem measure_setLaverage_le_pos (hμ : μ s ≠ 0) (hs : NullMeasurableSet s μ)
    (hint : ∫⁻ a in s, f a ∂μ ≠ ∞) : 0 < μ {x ∈ s | ⨍⁻ a in s, f a ∂μ ≤ f x} := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
  -/
  obtain hμ₁ | hμ₁ := eq_or_ne (μ s) ∞
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hμ : Ne (μ s) 0
      hs : MeasureTheory.NullMeasurableSet s μ
      hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
      hμ₁ : Eq (μ s) Top.top
      ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
    -/
  · simp [setLaverage_eq, hμ₁]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    hμ₁ : Ne (μ s) Top.top
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
  -/
  obtain ⟨g, hg, hgf, hfg⟩ := exists_measurable_le_lintegral_eq (μ.restrict s) f
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
  -/
  have hfg' : ⨍⁻ a in s, f a ∂μ = ⨍⁻ a in s, g a ∂μ := by simp_rw [laverage_eq, hfg]
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => f a) Top.top
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
  -/
  rw [hfg] at hint
  have :=
    measure_setAverage_le_pos hμ hμ₁ (integrable_toReal_of_lintegral_ne_top hg.aemeasurable hint)
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    this : LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheo …
    ⊢ LT.lt 0 (μ (setOf fun x => And (Membership.mem s x) (LE.le (MeasureTheory.la …
  -/
  simp_rw [← setOf_inter_eq_sep, ← Measure.restrict_apply₀' hs, hfg']
  rw [← setOf_inter_eq_sep, ← Measure.restrict_apply₀' hs, ←
    measure_diff_null (measure_eq_top_of_lintegral_ne_top hg.aemeasurable hint)] at this
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
    ⊢ LT.lt 0 ((μ.restrict s) (setOf fun a => LE.le (MeasureTheory.laverage (μ.res …
  -/
  refine this.trans_le (measure_mono ?_)
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
    ⊢ HasSubset.Subset (SDiff.sdiff (setOf fun a => LE.le (MeasureTheory.average ( …
  -/
  rintro x ⟨hfx, hx⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
    x : α
    hfx : Membership.mem (setOf fun a => LE.le (MeasureTheory.average (μ.restrict  …
    hx : Not (Membership.mem (setOf fun x => Eq (g x) Top.top) x)
    ⊢ Membership.mem (setOf fun a => LE.le (MeasureTheory.laverage (μ.restrict s)  …
  -/
  dsimp at hfx
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hμ : Ne (μ s) 0
    hs : MeasureTheory.NullMeasurableSet s μ
    hμ₁ : Ne (μ s) Top.top
    g : α → ENNReal
    hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
    hg : Measurable g
    hgf : LE.le g f
    hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
    this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
    x : α
    hfx : LE.le (MeasureTheory.average (μ.restrict s) fun a => (g a).toReal) (g x) …
    hx : Not (Membership.mem (setOf fun x => Eq (g x) Top.top) x)
    ⊢ Membership.mem (setOf fun a => LE.le (MeasureTheory.laverage (μ.restrict s)  …
  -/
  rw [← toReal_laverage hg.aemeasurable, toReal_le_toReal (setLaverage_lt_top hint).ne hx] at hfx
    /-
      case inr.intro.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hμ : Ne (μ s) 0
      hs : MeasureTheory.NullMeasurableSet s μ
      hμ₁ : Ne (μ s) Top.top
      g : α → ENNReal
      hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
      hg : Measurable g
      hgf : LE.le g f
      hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
      hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
      this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
      x : α
      hfx : LE.le (MeasureTheory.laverage (μ.restrict s) fun x => g x) (g x)
      hx : Not (Membership.mem (setOf fun x => Eq (g x) Top.top) x)
      ⊢ Membership.mem (setOf fun a => LE.le (MeasureTheory.laverage (μ.restrict s)  …
    -/
  · exact hfx.trans (hgf _)
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hμ : Ne (μ s) 0
      hs : MeasureTheory.NullMeasurableSet s μ
      hμ₁ : Ne (μ s) Top.top
      g : α → ENNReal
      hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
      hg : Measurable g
      hgf : LE.le g f
      hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
      hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
      this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
      x : α
      hfx : LE.le (MeasureTheory.average (μ.restrict s) fun a => (g a).toReal) (g x) …
      hx : Not (Membership.mem (setOf fun x => Eq (g x) Top.top) x)
      ⊢ Filter.Eventually (fun x => Ne (g x) Top.top) (MeasureTheory.ae (μ.restrict  …
    -/
  · simp_rw [ae_iff, not_ne_iff]
    /-
      case inr.intro.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hμ : Ne (μ s) 0
      hs : MeasureTheory.NullMeasurableSet s μ
      hμ₁ : Ne (μ s) Top.top
      g : α → ENNReal
      hint : Ne (MeasureTheory.lintegral (μ.restrict s) fun a => g a) Top.top
      hg : Measurable g
      hgf : LE.le g f
      hfg : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
      hfg' : Eq (MeasureTheory.laverage (μ.restrict s) fun a => f a) (MeasureTheory. …
      this : LT.lt 0 ((μ.restrict s) (SDiff.sdiff (setOf fun a => LE.le (MeasureTheo …
      x : α
      hfx : LE.le (MeasureTheory.average (μ.restrict s) fun a => (g a).toReal) (g x) …
      hx : Not (Membership.mem (setOf fun x => Eq (g x) Top.top) x)
      ⊢ Eq ((μ.restrict s) (setOf fun a => Eq (g a) Top.top)) 0
    -/
    exact measure_eq_top_of_lintegral_ne_top hg.aemeasurable hint
    /-
      🎉 no goals
    -/


/-- **First moment method**. The minimum of a measurable function is smaller than its mean. -/
theorem exists_le_setLaverage (hμ : μ s ≠ 0) (hμ₁ : μ s ≠ ∞) (hf : AEMeasurable f (μ.restrict s)) :
    ∃ x ∈ s, f x ≤ ⨍⁻ a in s, f a ∂μ :=
  let ⟨x, hx, h⟩ := nonempty_of_measure_ne_zero (measure_le_setLaverage_pos hμ hμ₁ hf).ne'
  ⟨x, hx, h⟩


/-- **First moment method**. The maximum of a measurable function is greater than its mean. -/
theorem exists_setLaverage_le (hμ : μ s ≠ 0) (hs : NullMeasurableSet s μ)
    (hint : ∫⁻ a in s, f a ∂μ ≠ ∞) : ∃ x ∈ s, ⨍⁻ a in s, f a ∂μ ≤ f x :=
  let ⟨x, hx, h⟩ := nonempty_of_measure_ne_zero (measure_setLaverage_le_pos hμ hs hint).ne'
  ⟨x, hx, h⟩


/-- **First moment method**. A measurable function is greater than its mean on a set of positive
measure. -/
theorem measure_laverage_le_pos (hμ : μ ≠ 0) (hint : ∫⁻ a, f a ∂μ ≠ ∞) :
    0 < μ {x | ⨍⁻ a, f a ∂μ ≤ f x} := by
  simpa [hint] using
    @measure_setLaverage_le_pos _ _ _ _ f (measure_univ_ne_zero.2 hμ) nullMeasurableSet_univ


/-- **First moment method**. The maximum of a measurable function is greater than its mean. -/
theorem exists_laverage_le (hμ : μ ≠ 0) (hint : ∫⁻ a, f a ∂μ ≠ ∞) : ∃ x, ⨍⁻ a, f a ∂μ ≤ f x :=
  let ⟨x, hx⟩ := nonempty_of_measure_ne_zero (measure_laverage_le_pos hμ hint).ne'
  ⟨x, hx⟩


/-- **First moment method**. The maximum of a measurable function is greater than its mean, while
avoiding a null set. -/
theorem exists_not_mem_null_laverage_le (hμ : μ ≠ 0) (hint : ∫⁻ a : α, f a ∂μ ≠ ∞) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ ⨍⁻ a, f a ∂μ ≤ f x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    hμ : Ne μ 0
    hint : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    hN : Eq (μ N) 0
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (MeasureTheory.laverag …
  -/
  have := measure_laverage_le_pos hμ hint
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    hμ : Ne μ 0
    hint : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (setOf fun x => LE.le (MeasureTheory.laverage μ fun a => f a …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (MeasureTheory.laverag …
  -/
  rw [← measure_diff_null hN] at this
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    hμ : Ne μ 0
    hint : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (MeasureTheory.laverage μ …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (MeasureTheory.laverag …
  -/
  obtain ⟨x, hx, hxN⟩ := nonempty_of_measure_ne_zero this.ne'
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    hμ : Ne μ 0
    hint : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (MeasureTheory.laverage μ …
    x : α
    hx : Membership.mem (setOf fun x => LE.le (MeasureTheory.laverage μ fun a => f …
    hxN : Not (Membership.mem N x)
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (MeasureTheory.laverag …
  -/
  exact ⟨x, hxN, hx⟩
  /-
    🎉 no goals
  -/


/-- **First moment method**. A measurable function is smaller than its mean on a set of positive
measure. -/
theorem measure_le_laverage_pos (hμ : μ ≠ 0) (hf : AEMeasurable f μ) :
    0 < μ {x | f x ≤ ⨍⁻ a, f a ∂μ} := by
  simpa using
    measure_le_setLaverage_pos (measure_univ_ne_zero.2 hμ) (measure_ne_top _ _) hf.restrict


/-- **First moment method**. The minimum of a measurable function is smaller than its mean. -/
theorem exists_le_laverage (hμ : μ ≠ 0) (hf : AEMeasurable f μ) : ∃ x, f x ≤ ⨍⁻ a, f a ∂μ :=
  let ⟨x, hx⟩ := nonempty_of_measure_ne_zero (measure_le_laverage_pos hμ hf).ne'
  ⟨x, hx⟩


/-- **First moment method**. The minimum of a measurable function is smaller than its mean, while
avoiding a null set. -/
theorem exists_not_mem_null_le_laverage (hμ : μ ≠ 0) (hf : AEMeasurable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ f x ≤ ⨍⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : AEMeasurable f μ
    hN : Eq (μ N) 0
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.l …
  -/
  have := measure_le_laverage_pos hμ hf
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : AEMeasurable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (setOf fun x => LE.le (f x) (MeasureTheory.laverage μ fun a  …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.l …
  -/
  rw [← measure_diff_null hN] at this
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : AEMeasurable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (f x) (MeasureTheory.lave …
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.l …
  -/
  obtain ⟨x, hx, hxN⟩ := nonempty_of_measure_ne_zero this.ne'
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    N : Set α
    f : α → ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    hf : AEMeasurable f μ
    hN : Eq (μ N) 0
    this : LT.lt 0 (μ (SDiff.sdiff (setOf fun x => LE.le (f x) (MeasureTheory.lave …
    x : α
    hx : Membership.mem (setOf fun x => LE.le (f x) (MeasureTheory.laverage μ fun  …
    hxN : Not (Membership.mem N x)
    ⊢ Exists fun x => And (Not (Membership.mem N x)) (LE.le (f x) (MeasureTheory.l …
  -/
  exact ⟨x, hxN, hx⟩
  /-
    🎉 no goals
  -/


/-- **First moment method**. A measurable function is smaller than its integral on a set f
positive measure. -/
theorem measure_le_lintegral_pos (hf : AEMeasurable f μ) : 0 < μ {x | f x ≤ ∫⁻ a, f a ∂μ} := by
  simpa only [laverage_eq_lintegral] using
    measure_le_laverage_pos (IsProbabilityMeasure.ne_zero μ) hf


/-- **First moment method**. A measurable function is greater than its integral on a set f
positive measure. -/
theorem measure_lintegral_le_pos (hint : ∫⁻ a, f a ∂μ ≠ ∞) : 0 < μ {x | ∫⁻ a, f a ∂μ ≤ f x} := by
  simpa only [laverage_eq_lintegral] using
    measure_laverage_le_pos (IsProbabilityMeasure.ne_zero μ) hint


/-- **First moment method**. The minimum of a measurable function is smaller than its integral. -/
theorem exists_le_lintegral (hf : AEMeasurable f μ) : ∃ x, f x ≤ ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : AEMeasurable f μ
    ⊢ Exists fun x => LE.le (f x) (MeasureTheory.lintegral μ fun a => f a)
  -/
  simpa only [laverage_eq_lintegral] using exists_le_laverage (IsProbabilityMeasure.ne_zero μ) hf
  /-
    🎉 no goals
  -/


/-- **First moment method**. The maximum of a measurable function is greater than its integral. -/
theorem exists_lintegral_le (hint : ∫⁻ a, f a ∂μ ≠ ∞) : ∃ x, ∫⁻ a, f a ∂μ ≤ f x := by
  simpa only [laverage_eq_lintegral] using
    exists_laverage_le (IsProbabilityMeasure.ne_zero μ) hint


/-- **First moment method**. The minimum of a measurable function is smaller than its integral,
while avoiding a null set. -/
theorem exists_not_mem_null_le_lintegral (hf : AEMeasurable f μ) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ f x ≤ ∫⁻ a, f a ∂μ := by
  simpa only [laverage_eq_lintegral] using
    exists_not_mem_null_le_laverage (IsProbabilityMeasure.ne_zero μ) hf hN


/-- **First moment method**. The maximum of a measurable function is greater than its integral,
while avoiding a null set. -/
theorem exists_not_mem_null_lintegral_le (hint : ∫⁻ a, f a ∂μ ≠ ∞) (hN : μ N = 0) :
    ∃ x, x ∉ N ∧ ∫⁻ a, f a ∂μ ≤ f x := by
  simpa only [laverage_eq_lintegral] using
    exists_not_mem_null_laverage_le (IsProbabilityMeasure.ne_zero μ) hint hN


/-- If the average of a function `f` along a sequence of sets `aₙ` converges to `c` (more precisely,
we require that `⨍ y in a i, ‖f y - c‖ ∂μ` tends to `0`), then the integral of `gₙ • f` also tends
to `c` if `gₙ` is supported in `aₙ`, has integral converging to one and supremum at most `K / μ aₙ`.
-/
theorem tendsto_integral_smul_of_tendsto_average_norm_sub
    [CompleteSpace E]
    {ι : Type*} {a : ι → Set α} {l : Filter ι} {f : α → E} {c : E} {g : ι → α → ℝ} (K : ℝ)
    (hf : Tendsto (fun i ↦ ⨍ y in a i, ‖f y - c‖ ∂μ) l (𝓝 0))
    (f_int : ∀ᶠ i in l, IntegrableOn f (a i) μ)
    (hg : Tendsto (fun i ↦ ∫ y, g i y ∂μ) l (𝓝 1))
    (g_supp : ∀ᶠ i in l, Function.support (g i) ⊆ a i)
    (g_bound : ∀ᶠ i in l, ∀ x, |g i x| ≤ K / (μ (a i)).toReal) :
    Tendsto (fun i ↦ ∫ y, g i y • f y ∂μ) l (𝓝 c) := by
  have g_int : ∀ᶠ i in l, Integrable (g i) μ := by
    filter_upwards [(tendsto_order.1 hg).1 _ zero_lt_one] with i hi
    contrapose hi
    simp only [integral_undef hi, lt_self_iff_false, not_false_eq_true]
  have I : ∀ᶠ i in l, ∫ y, g i y • (f y - c) ∂μ + (∫ y, g i y ∂μ) • c = ∫ y, g i y • f y ∂μ := by
    filter_upwards [f_int, g_int, g_supp, g_bound] with i hif hig hisupp hibound
    rw [← integral_smul_const, ← integral_add]
    · simp only [smul_sub, sub_add_cancel]
    · simp_rw [smul_sub]
      apply Integrable.sub _ (hig.smul_const _)
      have A : Function.support (fun y ↦ g i y • f y) ⊆ a i := by
        apply Subset.trans _ hisupp
        exact Function.support_smul_subset_left _ _
      rw [← integrableOn_iff_integrable_of_support_subset A]
      apply Integrable.smul_of_top_right hif
      exact memℒp_top_of_bound hig.aestronglyMeasurable.restrict
        (K / (μ (a i)).toReal) (Eventually.of_forall hibound)
    · exact hig.smul_const _
  have L0 : Tendsto (fun i ↦ ∫ y, g i y • (f y - c) ∂μ) l (𝓝 0) := by
    have := hf.const_mul K
    simp only [mul_zero] at this
    refine squeeze_zero_norm' ?_ this
    filter_upwards [g_supp, g_bound, f_int, (tendsto_order.1 hg).1 _ zero_lt_one]
      with i hi h'i h''i hi_int
    have mu_ai : μ (a i) < ∞ := by
      rw [lt_top_iff_ne_top]
      intro h
      simp only [h, ENNReal.top_toReal, _root_.div_zero, abs_nonpos_iff] at h'i
      have : ∫ (y : α), g i y ∂μ = ∫ (y : α), 0 ∂μ := by congr; ext y; exact h'i y
      simp [this] at hi_int
    apply (norm_integral_le_integral_norm _).trans
    simp_rw [average_eq, smul_eq_mul, ← integral_mul_left, norm_smul, ← mul_assoc, ← div_eq_mul_inv]
    have : ∀ x, x ∉ a i → ‖g i x‖ * ‖(f x - c)‖ = 0 := by
      intro x hx
      have : g i x = 0 := by rw [← Function.nmem_support]; exact fun h ↦ hx (hi h)
      simp [this]
    rw [← setIntegral_eq_integral_of_forall_compl_eq_zero this (μ := μ)]
    refine integral_mono_of_nonneg (Eventually.of_forall (fun x ↦ by positivity)) ?_
      (Eventually.of_forall (fun x ↦ ?_))
    · apply (Integrable.sub h''i _).norm.const_mul
      change IntegrableOn (fun _ ↦ c) (a i) μ
      simp [integrableOn_const, mu_ai]
    · dsimp; gcongr; simpa using h'i x
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace E
    ι : Type u_4
    a : ι → Set α
    l : Filter ι
    f : α → E
    c : E
    g : ι → α → Real
    K : Real
    hf : Filter.Tendsto (fun i => MeasureTheory.average (μ.restrict (a i)) fun y = …
    f_int : Filter.Eventually (fun i => MeasureTheory.IntegrableOn f (a i) μ) l
    hg : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => g i y) l (nhds …
    g_supp : Filter.Eventually (fun i => HasSubset.Subset (Function.support (g i)) …
    g_bound : Filter.Eventually (fun i => ∀ (x : α), LE.le (abs (g i x)) (HDiv.hDi …
    g_int : Filter.Eventually (fun i => MeasureTheory.Integrable (g i) μ) l
    I : Filter.Eventually (fun i => Eq (HAdd.hAdd (MeasureTheory.integral μ fun y  …
    L0 : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g i  …
  -/
  have := L0.add (hg.smul_const c)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace E
    ι : Type u_4
    a : ι → Set α
    l : Filter ι
    f : α → E
    c : E
    g : ι → α → Real
    K : Real
    hf : Filter.Tendsto (fun i => MeasureTheory.average (μ.restrict (a i)) fun y = …
    f_int : Filter.Eventually (fun i => MeasureTheory.IntegrableOn f (a i) μ) l
    hg : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => g i y) l (nhds …
    g_supp : Filter.Eventually (fun i => HasSubset.Subset (Function.support (g i)) …
    g_bound : Filter.Eventually (fun i => ∀ (x : α), LE.le (abs (g i x)) (HDiv.hDi …
    g_int : Filter.Eventually (fun i => MeasureTheory.Integrable (g i) μ) l
    I : Filter.Eventually (fun i => Eq (HAdd.hAdd (MeasureTheory.integral μ fun y  …
    L0 : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g …
    this : Filter.Tendsto (fun x => HAdd.hAdd (MeasureTheory.integral μ fun y => H …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g i  …
  -/
  simp only [one_smul, zero_add] at this
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace E
    ι : Type u_4
    a : ι → Set α
    l : Filter ι
    f : α → E
    c : E
    g : ι → α → Real
    K : Real
    hf : Filter.Tendsto (fun i => MeasureTheory.average (μ.restrict (a i)) fun y = …
    f_int : Filter.Eventually (fun i => MeasureTheory.IntegrableOn f (a i) μ) l
    hg : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => g i y) l (nhds …
    g_supp : Filter.Eventually (fun i => HasSubset.Subset (Function.support (g i)) …
    g_bound : Filter.Eventually (fun i => ∀ (x : α), LE.le (abs (g i x)) (HDiv.hDi …
    g_int : Filter.Eventually (fun i => MeasureTheory.Integrable (g i) μ) l
    I : Filter.Eventually (fun i => Eq (HAdd.hAdd (MeasureTheory.integral μ fun y  …
    L0 : Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g …
    this : Filter.Tendsto (fun x => HAdd.hAdd (MeasureTheory.integral μ fun y => H …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => HSMul.hSMul (g i  …
  -/
  exact Tendsto.congr' I this
  /-
    🎉 no goals
  -/


