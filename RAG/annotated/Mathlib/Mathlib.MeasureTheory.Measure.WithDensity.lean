/-- Given a measure `μ : Measure α` and a function `f : α → ℝ≥0∞`, `μ.withDensity f` is the
measure such that for a measurable set `s` we have `μ.withDensity f s = ∫⁻ a in s, f a ∂μ`. -/
noncomputable
def Measure.withDensity {m : MeasurableSpace α} (μ : Measure α) (f : α → ℝ≥0∞) : Measure α :=
                                                          /-
                                                            α : Type u_1
                                                            m0 : MeasurableSpace α
                                                            μ✝ : MeasureTheory.Measure α
                                                            m : MeasurableSpace α
                                                            μ : MeasureTheory.Measure α
                                                            f : α → ENNReal
                                                            ⊢ Eq ((fun s x => MeasureTheory.lintegral (μ.restrict s) fun a => f a) EmptyCo …
                                                          -/
  Measure.ofMeasurable (fun s _ => ∫⁻ a in s, f a ∂μ) (by simp) fun _ hs hd =>
                                                          /-
                                                            🎉 no goals
                                                          -/
    lintegral_iUnion hs hd _


@[simp]
theorem withDensity_apply (f : α → ℝ≥0∞) {s : Set α} (hs : MeasurableSet s) :
    μ.withDensity f s = ∫⁻ a in s, f a ∂μ :=
  Measure.ofMeasurable_apply s hs


theorem withDensity_apply_le (f : α → ℝ≥0∞) (s : Set α) :
    ∫⁻ a in s, f a ∂μ ≤ μ.withDensity f s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) ((μ.withDensity  …
  -/
  let t := toMeasurable (μ.withDensity f) s
  calc
  ∫⁻ a in s, f a ∂μ ≤ ∫⁻ a in t, f a ∂μ :=
    lintegral_mono_set (subset_toMeasurable (withDensity μ f) s)
  _ = μ.withDensity f t :=
    (withDensity_apply f (measurableSet_toMeasurable (withDensity μ f) s)).symm
  _ = μ.withDensity f s := measure_toMeasurable s



theorem withDensity_apply' [SFinite μ] (f : α → ℝ≥0∞) (s : Set α) :
    μ.withDensity f s = ∫⁻ a in s, f a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    s : Set α
    ⊢ Eq ((μ.withDensity f) s) (MeasureTheory.lintegral (μ.restrict s) fun a => f a)
  -/
  apply le_antisymm ?_ (withDensity_apply_le f s)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    s : Set α
    ⊢ LE.le ((μ.withDensity f) s) (MeasureTheory.lintegral (μ.restrict s) fun a => …
  -/
  let t := toMeasurable μ s
  calc
  μ.withDensity f s ≤ μ.withDensity f t := measure_mono (subset_toMeasurable μ s)
  _ = ∫⁻ a in t, f a ∂μ := withDensity_apply f (measurableSet_toMeasurable μ s)
  _ = ∫⁻ a in s, f a ∂μ := by congr 1; exact restrict_toMeasurable_of_sFinite s


@[simp]
lemma withDensity_zero_left (f : α → ℝ≥0∞) : (0 : Measure α).withDensity f = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.Measure.withDensity 0 f) 0
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    f : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.withDensity 0 f) s) (0 s)
  -/
  rw [withDensity_apply _ hs]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    f : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.restrict 0 s) fun a => f  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withDensity_congr_ae {f g : α → ℝ≥0∞} (h : f =ᵐ[μ] g) :
    μ.withDensity f = μ.withDensity g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (μ.withDensity f) (μ.withDensity g)
  -/
  refine Measure.ext fun s hs => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : (MeasureTheory.ae μ).EventuallyEq f g
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity f) s) ((μ.withDensity g) s)
  -/
  rw [withDensity_apply _ hs, withDensity_apply _ hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : (MeasureTheory.ae μ).EventuallyEq f g
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.lint …
  -/
  exact lintegral_congr_ae (ae_restrict_of_ae h)
  /-
    🎉 no goals
  -/


lemma withDensity_mono {f g : α → ℝ≥0∞} (hfg : f ≤ᵐ[μ] g) :
    μ.withDensity f ≤ μ.withDensity g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le (μ.withDensity f) (μ.withDensity g)
  -/
  refine le_iff.2 fun s hs ↦ ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    s : Set α
    hs : MeasurableSet s
    ⊢ LE.le ((μ.withDensity f) s) ((μ.withDensity g) s)
  -/
  rw [withDensity_apply _ hs, withDensity_apply _ hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    s : Set α
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.l …
  -/
  refine setLIntegral_mono_ae' hs ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Measure …
  -/
  filter_upwards [hfg] with x h_le using fun _ ↦ h_le
  /-
    🎉 no goals
  -/


theorem withDensity_add_left {f : α → ℝ≥0∞} (hf : Measurable f) (g : α → ℝ≥0∞) :
    μ.withDensity (f + g) = μ.withDensity f + μ.withDensity g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    g : α → ENNReal
    ⊢ Eq (μ.withDensity (HAdd.hAdd f g)) (HAdd.hAdd (μ.withDensity f) (μ.withDensi …
  -/
  refine Measure.ext fun s hs => ?_
  rw [withDensity_apply _ hs, Measure.add_apply, withDensity_apply _ hs, withDensity_apply _ hs,
    ← lintegral_add_left hf]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HAdd.hAdd f g a) (Measur …
  -/
  simp only [Pi.add_apply]
  /-
    🎉 no goals
  -/


theorem withDensity_add_right (f : α → ℝ≥0∞) {g : α → ℝ≥0∞} (hg : Measurable g) :
    μ.withDensity (f + g) = μ.withDensity f + μ.withDensity g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : Measurable g
    ⊢ Eq (μ.withDensity (HAdd.hAdd f g)) (HAdd.hAdd (μ.withDensity f) (μ.withDensi …
  -/
  simpa only [add_comm] using withDensity_add_left hg f
  /-
    🎉 no goals
  -/


theorem withDensity_add_measure {m : MeasurableSpace α} (μ ν : Measure α) (f : α → ℝ≥0∞) :
    (μ + ν).withDensity f = μ.withDensity f + ν.withDensity f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq ((HAdd.hAdd μ ν).withDensity f) (HAdd.hAdd (μ.withDensity f) (ν.withDensi …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (((HAdd.hAdd μ ν).withDensity f) s) ((HAdd.hAdd (μ.withDensity f) (ν.with …
  -/
  simp only [withDensity_apply f hs, restrict_add, lintegral_add_measure, Measure.add_apply]
  /-
    🎉 no goals
  -/


theorem withDensity_sum {ι : Type*} {m : MeasurableSpace α} (μ : ι → Measure α) (f : α → ℝ≥0∞) :
    (sum μ).withDensity f = sum fun n => (μ n).withDensity f := by
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq ((MeasureTheory.Measure.sum μ).withDensity f) (MeasureTheory.Measure.sum  …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.Measure.sum μ).withDensity f) s) ((MeasureTheory.Measure …
  -/
  simp_rw [sum_apply _ hs, withDensity_apply f hs, restrict_sum μ hs, lintegral_sum_measure]
  /-
    🎉 no goals
  -/


theorem withDensity_smul (r : ℝ≥0∞) {f : α → ℝ≥0∞} (hf : Measurable f) :
    μ.withDensity (r • f) = r • μ.withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (μ.withDensity (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensity f))
  -/
  refine Measure.ext fun s hs => ?_
  rw [withDensity_apply _ hs, Measure.coe_smul, Pi.smul_apply, withDensity_apply _ hs,
    smul_eq_mul, ← lintegral_const_mul r hf]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HSMul.hSMul r f a) (Meas …
  -/
  simp only [Pi.smul_apply, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem withDensity_smul' (r : ℝ≥0∞) (f : α → ℝ≥0∞) (hr : r ≠ ∞) :
    μ.withDensity (r • f) = r • μ.withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    ⊢ Eq (μ.withDensity (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensity f))
  -/
  refine Measure.ext fun s hs => ?_
  rw [withDensity_apply _ hs, Measure.coe_smul, Pi.smul_apply, withDensity_apply _ hs,
    smul_eq_mul, ← lintegral_const_mul' r f hr]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HSMul.hSMul r f a) (Meas …
  -/
  simp only [Pi.smul_apply, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem withDensity_smul_measure (r : ℝ≥0∞) (f : α → ℝ≥0∞) :
    (r • μ).withDensity f = r • μ.withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    ⊢ Eq ((HSMul.hSMul r μ).withDensity f) (HSMul.hSMul r (μ.withDensity f))
  -/
  ext s hs
  rw [withDensity_apply _ hs, Measure.coe_smul, Pi.smul_apply, withDensity_apply _ hs,
    smul_eq_mul, setLIntegral_smul_measure]


theorem isFiniteMeasure_withDensity {f : α → ℝ≥0∞} (hf : ∫⁻ a, f a ∂μ ≠ ∞) :
    IsFiniteMeasure (μ.withDensity f) :=
  { measure_univ_lt_top := by
      /-
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
        ⊢ LT.lt ((μ.withDensity f) Set.univ) Top.top
      -/
      rwa [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ, lt_top_iff_ne_top] }
      /-
        🎉 no goals
      -/


theorem withDensity_absolutelyContinuous {m : MeasurableSpace α} (μ : Measure α) (f : α → ℝ≥0∞) :
    μ.withDensity f ≪ μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ (μ.withDensity f).AbsolutelyContinuous μ
  -/
  refine AbsolutelyContinuous.mk fun s hs₁ hs₂ => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hs₁ : MeasurableSet s
    hs₂ : Eq (μ s) 0
    ⊢ Eq ((μ.withDensity f) s) 0
  -/
  rw [withDensity_apply _ hs₁]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hs₁ : MeasurableSet s
    hs₂ : Eq (μ s) 0
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) 0
  -/
  exact setLIntegral_measure_zero _ _ hs₂
  /-
    🎉 no goals
  -/


@[simp]
theorem withDensity_zero : μ.withDensity 0 = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.withDensity 0) 0
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity 0) s) (0 s)
  -/
  simp [withDensity_apply _ hs]
  /-
    🎉 no goals
  -/


@[simp]
theorem withDensity_one : μ.withDensity 1 = μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.withDensity 1) μ
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity 1) s) (μ s)
  -/
  simp [withDensity_apply _ hs]
  /-
    🎉 no goals
  -/


@[simp]
theorem withDensity_const (c : ℝ≥0∞) : μ.withDensity (fun _ ↦ c) = c • μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    ⊢ Eq (μ.withDensity fun x => c) (HSMul.hSMul c μ)
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity fun x => c) s) ((HSMul.hSMul c μ) s)
  -/
  simp [withDensity_apply _ hs]
  /-
    🎉 no goals
  -/


theorem withDensity_tsum {ι : Type*} [Countable ι] {f : ι → α → ℝ≥0∞} (h : ∀ i, Measurable (f i)) :
    μ.withDensity (∑' n, f n) = sum fun n => μ.withDensity (f n) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Eq (μ.withDensity (tsum fun n => f n)) (MeasureTheory.Measure.sum fun n => μ …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity (tsum fun n => f n)) s) ((MeasureTheory.Measure.sum fun n …
  -/
  simp_rw [sum_apply _ hs, withDensity_apply _ hs]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => tsum (fun n => f n) a) ( …
  -/
  change ∫⁻ x in s, (∑' n, f n) x ∂μ = ∑' i, ∫⁻ x, f i x ∂μ.restrict s
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => tsum (fun n => f n) x) ( …
  -/
  rw [← lintegral_tsum fun i => (h i).aemeasurable]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => tsum (fun n => f n) x) ( …
  -/
  exact lintegral_congr fun x => tsum_apply (Pi.summable.2 fun _ => ENNReal.summable)
  /-
    🎉 no goals
  -/


theorem withDensity_indicator {s : Set α} (hs : MeasurableSet s) (f : α → ℝ≥0∞) :
    μ.withDensity (s.indicator f) = (μ.restrict s).withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ Eq (μ.withDensity (s.indicator f)) ((μ.restrict s).withDensity f)
  -/
  ext1 t ht
  rw [withDensity_apply _ ht, lintegral_indicator hs, restrict_comm hs, ←
    withDensity_apply _ ht]


theorem withDensity_indicator_one {s : Set α} (hs : MeasurableSet s) :
    μ.withDensity (s.indicator 1) = μ.restrict s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (μ.withDensity (s.indicator 1)) (μ.restrict s)
  -/
  rw [withDensity_indicator hs, withDensity_one]
  /-
    🎉 no goals
  -/


theorem withDensity_ofReal_mutuallySingular {f : α → ℝ} (hf : Measurable f) :
    (μ.withDensity fun x => ENNReal.ofReal <| f x) ⟂ₘ
      μ.withDensity fun x => ENNReal.ofReal <| -f x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    ⊢ (μ.withDensity fun x => ENNReal.ofReal (f x)).MutuallySingular (μ.withDensit …
  -/
  set S : Set α := { x | f x < 0 }
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    S : Set α := setOf fun x => LT.lt (f x) 0
    ⊢ (μ.withDensity fun x => ENNReal.ofReal (f x)).MutuallySingular (μ.withDensit …
  -/
  have hS : MeasurableSet S := measurableSet_lt hf measurable_const
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    S : Set α := setOf fun x => LT.lt (f x) 0
    hS : MeasurableSet S
    ⊢ (μ.withDensity fun x => ENNReal.ofReal (f x)).MutuallySingular (μ.withDensit …
  -/
  refine ⟨S, hS, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      S : Set α := setOf fun x => LT.lt (f x) 0
      hS : MeasurableSet S
      ⊢ Eq ((μ.withDensity fun x => ENNReal.ofReal (f x)) S) 0
    -/
  · rw [withDensity_apply _ hS, lintegral_eq_zero_iff hf.ennreal_ofReal, EventuallyEq]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      S : Set α := setOf fun x => LT.lt (f x) 0
      hS : MeasurableSet S
      ⊢ Filter.Eventually (fun x => Eq (ENNReal.ofReal (f x)) (0 x)) (MeasureTheory. …
    -/
    exact (ae_restrict_mem hS).mono fun x hx => ENNReal.ofReal_eq_zero.2 (le_of_lt hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      S : Set α := setOf fun x => LT.lt (f x) 0
      hS : MeasurableSet S
      ⊢ Eq ((μ.withDensity fun x => ENNReal.ofReal (Neg.neg (f x))) (HasCompl.compl  …
    -/
  · rw [withDensity_apply _ hS.compl, lintegral_eq_zero_iff hf.neg.ennreal_ofReal, EventuallyEq]
    exact
      (ae_restrict_mem hS.compl).mono fun x hx =>
        ENNReal.ofReal_eq_zero.2 (not_lt.1 <| mt neg_pos.1 hx)


theorem restrict_withDensity {s : Set α} (hs : MeasurableSet s) (f : α → ℝ≥0∞) :
    (μ.withDensity f).restrict s = (μ.restrict s).withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ Eq ((μ.withDensity f).restrict s) ((μ.restrict s).withDensity f)
  -/
  ext1 t ht
  rw [restrict_apply ht, withDensity_apply _ ht, withDensity_apply _ (ht.inter hs),
    restrict_restrict ht]


theorem restrict_withDensity' [SFinite μ] (s : Set α) (f : α → ℝ≥0∞) :
    (μ.withDensity f).restrict s = (μ.restrict s).withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    f : α → ENNReal
    ⊢ Eq ((μ.withDensity f).restrict s) ((μ.restrict s).withDensity f)
  -/
  ext1 t ht
  rw [restrict_apply ht, withDensity_apply _ ht, withDensity_apply' _ (t ∩ s),
    restrict_restrict ht]


lemma trim_withDensity {m m0 : MeasurableSpace α} {μ : Measure α}
    (hm : m ≤ m0) {f : α → ℝ≥0∞} (hf : Measurable[m] f) :
    (μ.withDensity f).trim hm = (μ.trim hm).withDensity f := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq ((μ.withDensity f).trim hm) ((μ.trim hm).withDensity f)
  -/
  refine @Measure.ext _ m _ _ (fun s hs ↦ ?_)
  rw [withDensity_apply _ hs, restrict_trim _ _ hs, lintegral_trim _ hf, trim_measurableSet_eq _ hs,
    withDensity_apply _ (hm s hs)]


lemma Measure.MutuallySingular.withDensity {ν : Measure α} {f : α → ℝ≥0∞} (h : μ ⟂ₘ ν) :
    μ.withDensity f ⟂ₘ ν :=
  MutuallySingular.mono_ac h (withDensity_absolutelyContinuous _ _) AbsolutelyContinuous.rfl


@[simp]
theorem withDensity_eq_zero_iff {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
    μ.withDensity f = 0 ↔ f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Iff (Eq (μ.withDensity f) 0) ((MeasureTheory.ae μ).EventuallyEq f 0)
  -/
  rw [← measure_univ_eq_zero, withDensity_apply _ .univ, restrict_univ, lintegral_eq_zero_iff' hf]
  /-
    🎉 no goals
  -/


alias ⟨withDensity_eq_zero, _⟩ := withDensity_eq_zero_iff


theorem withDensity_apply_eq_zero' {f : α → ℝ≥0∞} {s : Set α} (hf : AEMeasurable f μ) :
    μ.withDensity f s = 0 ↔ μ ({ x | f x ≠ 0 } ∩ s) = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hf : AEMeasurable f μ
    ⊢ Iff (Eq ((μ.withDensity f) s) 0) (Eq (μ (Inter.inter (setOf fun x => Ne (f x …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      ⊢ Eq ((μ.withDensity f) s) 0 → Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0)  …
    -/
  · intro hs
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
    -/
    let t := toMeasurable (μ.withDensity f) s
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
    -/
    apply measure_mono_null (inter_subset_inter_right _ (subset_toMeasurable (μ.withDensity f) s))
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) (MeasureTheory.toMeasurable ( …
    -/
    have A : μ.withDensity f t = 0 := by rw [measure_toMeasurable, hs]
    rw [withDensity_apply f (measurableSet_toMeasurable _ s),
      lintegral_eq_zero_iff' (AEMeasurable.restrict hf),
      EventuallyEq, ae_restrict_iff'₀, ae_iff] at A
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      A : Eq (μ (setOf fun a => Not (Membership.mem (MeasureTheory.toMeasurable (μ.w …
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) (MeasureTheory.toMeasurable ( …
    -/
    swap
      /-
        case mp
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        hs : Eq ((μ.withDensity f) s) 0
        t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
        A : Filter.Eventually (fun x => Eq (f x) (0 x)) (MeasureTheory.ae (μ.restrict  …
        ⊢ MeasureTheory.NullMeasurableSet (MeasureTheory.toMeasurable (μ.withDensity f …
      -/
    · simp only [measurableSet_toMeasurable, MeasurableSet.nullMeasurableSet]
      /-
        🎉 no goals
      -/
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      A : Eq (μ (setOf fun a => Not (Membership.mem (MeasureTheory.toMeasurable (μ.w …
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) (MeasureTheory.toMeasurable ( …
    -/
    simp only [Pi.zero_apply, mem_setOf_eq, Filter.mem_mk] at A
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      A : Eq (μ (setOf fun a => Not (Membership.mem (MeasureTheory.toMeasurable (μ.w …
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) (MeasureTheory.toMeasurable ( …
    -/
    convert A using 2
    /-
      case h.e'_2.h.e'_6
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq ((μ.withDensity f) s) 0
      t : Set α := MeasureTheory.toMeasurable (μ.withDensity f) s
      A : Eq (μ (setOf fun a => Not (Membership.mem (MeasureTheory.toMeasurable (μ.w …
      ⊢ Eq (Inter.inter (setOf fun x => Ne (f x) 0) (MeasureTheory.toMeasurable (μ.w …
    -/
    ext x
    simp only [and_comm, exists_prop, mem_inter_iff, mem_setOf_eq,
      mem_compl_iff, not_forall]
    /-
      case mpr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      ⊢ Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0 → Eq ((μ.withDensity f) …
    -/
  · intro hs
    /-
      case mpr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      ⊢ Eq ((μ.withDensity f) s) 0
    -/
    let t := toMeasurable μ ({ x | f x ≠ 0 } ∩ s)
    have A : s ⊆ t ∪ { x | f x = 0 } := by
      intro x hx
      rcases eq_or_ne (f x) 0 with (fx | fx)
      · simp only [fx, mem_union, mem_setOf_eq, eq_self_iff_true, or_true]
      · left
        apply subset_toMeasurable _ _
        exact ⟨fx, hx⟩
    /-
      case mpr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f x …
      A : HasSubset.Subset s (Union.union t (setOf fun x => Eq (f x) 0))
      ⊢ Eq ((μ.withDensity f) s) 0
    -/
    apply measure_mono_null A (measure_union_null _ _)
      /-
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
        t : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f x …
        A : HasSubset.Subset s (Union.union t (setOf fun x => Eq (f x) 0))
        ⊢ Eq ((μ.withDensity f) t) 0
      -/
    · apply withDensity_absolutelyContinuous
      /-
        case a
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
        t : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f x …
        A : HasSubset.Subset s (Union.union t (setOf fun x => Eq (f x) 0))
        ⊢ Eq (μ t) 0
      -/
      rwa [measure_toMeasurable]
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f x …
      A : HasSubset.Subset s (Union.union t (setOf fun x => Eq (f x) 0))
      ⊢ Eq ((μ.withDensity f) (setOf fun x => Eq (f x) 0)) 0
    -/
    rcases hf with ⟨g, hg, hfg⟩
    have t : {x | f x = 0} =ᵐ[μ.withDensity f] {x | g x = 0} := by
      apply withDensity_absolutelyContinuous
      filter_upwards [hfg] with a ha
      rw [eq_iff_iff]
      exact ⟨fun h ↦ by rw [h] at ha; exact ha.symm,
             fun h ↦ by rw [h] at ha; exact ha⟩
    /-
      case intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t✝ : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f  …
      A : HasSubset.Subset s (Union.union t✝ (setOf fun x => Eq (f x) 0))
      g : α → ENNReal
      hg : Measurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      t : (MeasureTheory.ae (μ.withDensity f)).EventuallyEq (setOf fun x => Eq (f x) …
      ⊢ Eq ((μ.withDensity f) (setOf fun x => Eq (f x) 0)) 0
    -/
    rw [measure_congr t, withDensity_congr_ae hfg]
    /-
      case intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t✝ : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f  …
      A : HasSubset.Subset s (Union.union t✝ (setOf fun x => Eq (f x) 0))
      g : α → ENNReal
      hg : Measurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      t : (MeasureTheory.ae (μ.withDensity f)).EventuallyEq (setOf fun x => Eq (f x) …
      ⊢ Eq ((μ.withDensity g) (setOf fun x => Eq (g x) 0)) 0
    -/
    have M : MeasurableSet { x : α | g x = 0 } := hg (measurableSet_singleton _)
    /-
      case intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t✝ : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f  …
      A : HasSubset.Subset s (Union.union t✝ (setOf fun x => Eq (f x) 0))
      g : α → ENNReal
      hg : Measurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      t : (MeasureTheory.ae (μ.withDensity f)).EventuallyEq (setOf fun x => Eq (f x) …
      M : MeasurableSet (setOf fun x => Eq (g x) 0)
      ⊢ Eq ((μ.withDensity g) (setOf fun x => Eq (g x) 0)) 0
    -/
    rw [withDensity_apply _ M, lintegral_eq_zero_iff hg]
    /-
      case intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t✝ : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f  …
      A : HasSubset.Subset s (Union.union t✝ (setOf fun x => Eq (f x) 0))
      g : α → ENNReal
      hg : Measurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      t : (MeasureTheory.ae (μ.withDensity f)).EventuallyEq (setOf fun x => Eq (f x) …
      M : MeasurableSet (setOf fun x => Eq (g x) 0)
      ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (g x) 0))).EventuallyEq g 0
    -/
    filter_upwards [ae_restrict_mem M]
    /-
      case h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      hs : Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) s)) 0
      t✝ : Set α := MeasureTheory.toMeasurable μ (Inter.inter (setOf fun x => Ne (f  …
      A : HasSubset.Subset s (Union.union t✝ (setOf fun x => Eq (f x) 0))
      g : α → ENNReal
      hg : Measurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      t : (MeasureTheory.ae (μ.withDensity f)).EventuallyEq (setOf fun x => Eq (f x) …
      M : MeasurableSet (setOf fun x => Eq (g x) 0)
      ⊢ ∀ (a : α), Eq (g a) 0 → Eq (g a) (0 a)
    -/
    simp only [imp_self, Pi.zero_apply, imp_true_iff]
    /-
      🎉 no goals
    -/


theorem withDensity_apply_eq_zero {f : α → ℝ≥0∞} {s : Set α} (hf : Measurable f) :
    μ.withDensity f s = 0 ↔ μ ({ x | f x ≠ 0 } ∩ s) = 0 :=
  withDensity_apply_eq_zero' <| hf.aemeasurable


theorem ae_withDensity_iff' {p : α → Prop} {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
    (∀ᵐ x ∂μ.withDensity f, p x) ↔ ∀ᵐ x ∂μ, f x ≠ 0 → p x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : α → Prop
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.withDensity f)))  …
  -/
  rw [ae_iff, ae_iff, withDensity_apply_eq_zero' hf, iff_iff_eq]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : α → Prop
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Eq (Eq (μ (Inter.inter (setOf fun x => Ne (f x) 0) (setOf fun a => Not (p a) …
  -/
  congr
  /-
    case e_a.h.e_6.h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : α → Prop
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Eq (Inter.inter (setOf fun x => Ne (f x) 0) (setOf fun a => Not (p a))) (set …
  -/
  ext x
  /-
    case e_a.h.e_6.h.h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : α → Prop
    f : α → ENNReal
    hf : AEMeasurable f μ
    x : α
    ⊢ Iff (Membership.mem (Inter.inter (setOf fun x => Ne (f x) 0) (setOf fun a => …
  -/
  simp only [exists_prop, mem_inter_iff, mem_setOf_eq, not_forall]
  /-
    🎉 no goals
  -/


theorem ae_withDensity_iff {p : α → Prop} {f : α → ℝ≥0∞} (hf : Measurable f) :
    (∀ᵐ x ∂μ.withDensity f, p x) ↔ ∀ᵐ x ∂μ, f x ≠ 0 → p x :=
  ae_withDensity_iff' <| hf.aemeasurable


theorem ae_withDensity_iff_ae_restrict' {p : α → Prop} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) :
    (∀ᵐ x ∂μ.withDensity f, p x) ↔ ∀ᵐ x ∂μ.restrict { x | f x ≠ 0 }, p x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : α → Prop
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.withDensity f)))  …
  -/
  rw [ae_withDensity_iff' hf, ae_restrict_iff'₀]
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : α → Prop
      f : α → ENNReal
      hf : AEMeasurable f μ
      ⊢ Iff (Filter.Eventually (fun x => Ne (f x) 0 → p x) (MeasureTheory.ae μ)) (Fi …
    -/
  · simp only [mem_setOf]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : α → Prop
      f : α → ENNReal
      hf : AEMeasurable f μ
      ⊢ MeasureTheory.NullMeasurableSet (setOf fun x => Ne (f x) 0) μ
    -/
  · rcases hf with ⟨g, hg, hfg⟩
    have nonneg_eq_ae : {x | g x ≠ 0} =ᵐ[μ] {x | f x ≠ 0} := by
      filter_upwards [hfg] with a ha
      simp only [eq_iff_iff]
      exact ⟨fun (h : g a ≠ 0) ↦ by rwa [← ha] at h,
             fun (h : f a ≠ 0) ↦ by rwa [ha] at h⟩
    exact NullMeasurableSet.congr
      (MeasurableSet.nullMeasurableSet
        <| hg (measurableSet_singleton _)).compl
      nonneg_eq_ae


theorem ae_withDensity_iff_ae_restrict {p : α → Prop} {f : α → ℝ≥0∞} (hf : Measurable f) :
    (∀ᵐ x ∂μ.withDensity f, p x) ↔ ∀ᵐ x ∂μ.restrict { x | f x ≠ 0 }, p x :=
  ae_withDensity_iff_ae_restrict' <| hf.aemeasurable


theorem aemeasurable_withDensity_ennreal_iff' {f : α → ℝ≥0}
    (hf : AEMeasurable f μ) {g : α → ℝ≥0∞} :
    AEMeasurable g (μ.withDensity fun x => (f x : ℝ≥0∞)) ↔
      AEMeasurable (fun x => (f x : ℝ≥0∞) * g x) μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    ⊢ Iff (AEMeasurable g (μ.withDensity fun x => ↑(f x))) (AEMeasurable (fun x => …
  -/
  have t : ∃ f', Measurable f' ∧ f =ᵐ[μ] f' := hf
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    t : Exists fun f' => And (Measurable f') ((MeasureTheory.ae μ).EventuallyEq f  …
    ⊢ Iff (AEMeasurable g (μ.withDensity fun x => ↑(f x))) (AEMeasurable (fun x => …
  -/
  rcases t with ⟨f', hf'_m, hf'_ae⟩
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    f' : α → NNReal
    hf'_m : Measurable f'
    hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ Iff (AEMeasurable g (μ.withDensity fun x => ↑(f x))) (AEMeasurable (fun x => …
  -/
  constructor
    /-
      case intro.intro.mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      ⊢ AEMeasurable g (μ.withDensity fun x => ↑(f x)) → AEMeasurable (fun x => HMul …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case intro.intro.mp.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      ⊢ AEMeasurable (fun x => HMul.hMul (↑(f x)) (g x)) μ
    -/
    have A : MeasurableSet {x | f' x ≠ 0} := hf'_m (measurableSet_singleton _).compl
    /-
      case intro.intro.mp.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      A : MeasurableSet (setOf fun x => Ne (f' x) 0)
      ⊢ AEMeasurable (fun x => HMul.hMul (↑(f x)) (g x)) μ
    -/
    refine ⟨fun x => f' x * g' x, hf'_m.coe_nnreal_ennreal.smul g'meas, ?_⟩
    /-
      case intro.intro.mp.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      A : MeasurableSet (setOf fun x => Ne (f' x) 0)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) fun x  …
    -/
    apply ae_of_ae_restrict_of_ae_restrict_compl { x | f' x ≠ 0 }
      /-
        case intro.intro.mp.intro.intro.ht
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HMul.hMul (↑(f x)) (g x)) x) ((fun …
      -/
    · rw [EventuallyEq, ae_withDensity_iff' hf.coe_nnreal_ennreal] at hg'
      /-
        case intro.intro.mp.intro.intro.ht
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HMul.hMul (↑(f x)) (g x)) x) ((fun …
      -/
      rw [ae_restrict_iff' A]
      /-
        case intro.intro.mp.intro.intro.ht
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => Ne (f' x) 0) x →  …
      -/
      filter_upwards [hg', hf'_ae] with a ha h'a h_a_nonneg
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Ne (↑(f a)) 0 → Eq (g a) (g' a)
        h'a : Eq (f a) (f' a)
        h_a_nonneg : Ne (f' a) 0
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑(f' a)) (g' a))
      -/
      have : (f' a : ℝ≥0∞) ≠ 0 := by simpa only [Ne, ENNReal.coe_eq_zero] using h_a_nonneg
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Ne (↑(f a)) 0 → Eq (g a) (g' a)
        h'a : Eq (f a) (f' a)
        h_a_nonneg : Ne (f' a) 0
        this : Ne (↑(f' a)) 0
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑(f' a)) (g' a))
      -/
      rw [← h'a] at this ⊢
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Ne (↑(f a)) 0 → Eq (g a) (g' a)
        h'a : Eq (f a) (f' a)
        h_a_nonneg : Ne (f' a) 0
        this : Ne (↑(f a)) 0
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑(f a)) (g' a))
      -/
      rw [ha this]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.mp.intro.intro.htc
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HMul.hMul (↑(f x)) (g x)) x) ((fun …
      -/
    · rw [ae_restrict_iff' A.compl]
      /-
        case intro.intro.mp.intro.intro.htc
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (setOf fun x => N …
      -/
      filter_upwards [hf'_ae] with a ha ha_null
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Eq (f a) (f' a)
        ha_null : Membership.mem (HasCompl.compl (setOf fun x => Ne (f' x) 0)) a
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑(f' a)) (g' a))
      -/
      have ha_null : f' a = 0 := Function.nmem_support.mp ha_null
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Eq (f a) (f' a)
        ha_null✝ : Membership.mem (HasCompl.compl (setOf fun x => Ne (f' x) 0)) a
        ha_null : Eq (f' a) 0
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑(f' a)) (g' a))
      -/
      rw [ha_null] at ha ⊢
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Eq (f a) 0
        ha_null✝ : Membership.mem (HasCompl.compl (setOf fun x => Ne (f' x) 0)) a
        ha_null : Eq (f' a) 0
        ⊢ Eq (HMul.hMul (↑(f a)) (g a)) (HMul.hMul (↑0) (g' a))
      -/
      rw [ha]
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → ENNReal
        f' : α → NNReal
        hf'_m : Measurable f'
        hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
        g' : α → ENNReal
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f' x) 0)
        a : α
        ha : Eq (f a) 0
        ha_null✝ : Membership.mem (HasCompl.compl (setOf fun x => Ne (f' x) 0)) a
        ha_null : Eq (f' a) 0
        ⊢ Eq (HMul.hMul (↑0) (g a)) (HMul.hMul (↑0) (g' a))
      -/
      simp only [ENNReal.coe_zero, zero_mul]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.mpr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      ⊢ AEMeasurable (fun x => HMul.hMul (↑(f x)) (g x)) μ → AEMeasurable g (μ.withD …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case intro.intro.mpr.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) g'
      ⊢ AEMeasurable g (μ.withDensity fun x => ↑(f x))
    -/
    refine ⟨fun x => ((f' x)⁻¹ : ℝ≥0∞) * g' x, hf'_m.coe_nnreal_ennreal.inv.smul g'meas, ?_⟩
    /-
      case intro.intro.mpr.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) g'
      ⊢ (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g fun x => H …
    -/
    rw [EventuallyEq, ae_withDensity_iff' hf.coe_nnreal_ennreal]
    /-
      case intro.intro.mpr.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) g'
      ⊢ Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (HMul.hMul (Inv.inv ↑(f …
    -/
    filter_upwards [hg', hf'_ae] with a hfga hff'a h'a
    /-
      case h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) g'
      a : α
      hfga : Eq (HMul.hMul (↑(f a)) (g a)) (g' a)
      hff'a : Eq (f a) (f' a)
      h'a : Ne (↑(f a)) 0
      ⊢ Eq (g a) (HMul.hMul (Inv.inv ↑(f' a)) (g' a))
    -/
    rw [hff'a] at hfga h'a
    /-
      case h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      hf : AEMeasurable f μ
      g : α → ENNReal
      f' : α → NNReal
      hf'_m : Measurable f'
      hf'_ae : (MeasureTheory.ae μ).EventuallyEq f f'
      g' : α → ENNReal
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (↑(f x)) (g x)) g'
      a : α
      hfga : Eq (HMul.hMul (↑(f' a)) (g a)) (g' a)
      hff'a : Eq (f a) (f' a)
      h'a : Ne (↑(f' a)) 0
      ⊢ Eq (g a) (HMul.hMul (Inv.inv ↑(f' a)) (g' a))
    -/
    rw [← hfga, ← mul_assoc, ENNReal.inv_mul_cancel h'a ENNReal.coe_ne_top, one_mul]
    /-
      🎉 no goals
    -/


theorem aemeasurable_withDensity_ennreal_iff {f : α → ℝ≥0} (hf : Measurable f) {g : α → ℝ≥0∞} :
    AEMeasurable g (μ.withDensity fun x => (f x : ℝ≥0∞)) ↔
      AEMeasurable (fun x => (f x : ℝ≥0∞) * g x) μ :=
  aemeasurable_withDensity_ennreal_iff' <| hf.aemeasurable


/-- This is Exercise 1.2.1 from [tao2010]. It allows you to express integration of a measurable
function with respect to `(μ.withDensity f)` as an integral with respect to `μ`, called the base
measure. `μ` is often the Lebesgue measure, and in this circumstance `f` is the probability density
function, and `(μ.withDensity f)` represents any continuous random variable as a
probability measure, such as the uniform distribution between 0 and 1, the Gaussian distribution,
the exponential distribution, the Beta distribution, or the Cauchy distribution (see Section 2.4
of [wasserman2004]). Thus, this method shows how to one can calculate expectations, variances,
and other moments as a function of the probability density function.
 -/
theorem lintegral_withDensity_eq_lintegral_mul (μ : Measure α) {f : α → ℝ≥0∞}
    (h_mf : Measurable f) :
    ∀ {g : α → ℝ≥0∞}, Measurable g → ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h_mf : Measurable f
    ⊢ ∀ {g : α → ENNReal}, Measurable g → Eq (MeasureTheory.lintegral (μ.withDensi …
  -/
  apply Measurable.ennreal_induction
    /-
      case h_ind
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      ⊢ ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → Eq (MeasureTheory.lintegral ( …
    -/
  · intro c s h_ms
    /-
      case h_ind
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      c : ENNReal
      s : Set α
      h_ms : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => s.indicator (fun x => …
    -/
    simp [*, mul_comm _ c, ← indicator_mul_right]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      ⊢ ∀ ⦃f_1 g : α → ENNReal⦄, Disjoint (Function.support f_1) (Function.support g …
    -/
  · intro g h _ h_mea_g _ h_ind_g h_ind_h
    /-
      case h_add
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      g h : α → ENNReal
      a✝¹ : Disjoint (Function.support g) (Function.support h)
      h_mea_g : Measurable g
      a✝ : Measurable h
      h_ind_g : Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (Measure …
      h_ind_h : Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => h a) (Measure …
      ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => HAdd.hAdd g h a) (Mea …
    -/
    simp [mul_add, *, Measurable.mul]
    /-
      🎉 no goals
    -/
    /-
      case h_iSup
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      ⊢ ∀ ⦃f_1 : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f_1 n)) → Monotone f_ …
    -/
  · intro g h_mea_g h_mono_g h_ind
    /-
      case h_iSup
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      g : Nat → α → ENNReal
      h_mea_g : ∀ (n : Nat), Measurable (g n)
      h_mono_g : Monotone g
      h_ind : ∀ (n : Nat), Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g  …
      ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => (fun x => iSup fun n  …
    -/
    have : Monotone fun n a => f a * g n a := fun m n hmn x => mul_le_mul_left' (h_mono_g hmn x) _
    /-
      case h_iSup
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h_mf : Measurable f
      g : Nat → α → ENNReal
      h_mea_g : ∀ (n : Nat), Measurable (g n)
      h_mono_g : Monotone g
      h_ind : ∀ (n : Nat), Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g  …
      this : Monotone fun n a => HMul.hMul (f a) (g n a)
      ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => (fun x => iSup fun n  …
    -/
    simp [lintegral_iSup, ENNReal.mul_iSup, h_mf.mul (h_mea_g _), *]
    /-
      🎉 no goals
    -/


theorem setLIntegral_withDensity_eq_setLIntegral_mul (μ : Measure α) {f g : α → ℝ≥0∞}
    (hf : Measurable f) (hg : Measurable g) {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ x in s, g x ∂μ.withDensity f = ∫⁻ x in s, (f * g) x ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((μ.withDensity f).restrict s) fun x => g x) (Me …
  -/
  rw [restrict_withDensity hs, lintegral_withDensity_eq_lintegral_mul _ hf hg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_set_lintegral_mul := setLIntegral_withDensity_eq_setLIntegral_mul


/-- The Lebesgue integral of `g` with respect to the measure `μ.withDensity f` coincides with
the integral of `f * g`. This version assumes that `g` is almost everywhere measurable. For a
version without conditions on `g` but requiring that `f` is almost everywhere finite, see
`lintegral_withDensity_eq_lintegral_mul_non_measurable` -/
theorem lintegral_withDensity_eq_lintegral_mul₀' {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) {g : α → ℝ≥0∞} (hg : AEMeasurable g (μ.withDensity f)) :
    ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheory.l …
  -/
  let f' := hf.mk f
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    f' : α → ENNReal := AEMeasurable.mk f hf
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheory.l …
  -/
  have : μ.withDensity f = μ.withDensity f' := withDensity_congr_ae hf.ae_eq_mk
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    f' : α → ENNReal := AEMeasurable.mk f hf
    this : Eq (μ.withDensity f) (μ.withDensity f')
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheory.l …
  -/
  rw [this] at hg ⊢
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    f' : α → ENNReal := AEMeasurable.mk f hf
    hg : AEMeasurable g (μ.withDensity f')
    this : Eq (μ.withDensity f) (μ.withDensity f')
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f') fun a => g a) (MeasureTheory. …
  -/
  let g' := hg.mk g
  calc
    ∫⁻ a, g a ∂μ.withDensity f' = ∫⁻ a, g' a ∂μ.withDensity f' := lintegral_congr_ae hg.ae_eq_mk
    _ = ∫⁻ a, (f' * g') a ∂μ :=
      (lintegral_withDensity_eq_lintegral_mul _ hf.measurable_mk hg.measurable_mk)
    _ = ∫⁻ a, (f' * g) a ∂μ := by
      apply lintegral_congr_ae
      apply ae_of_ae_restrict_of_ae_restrict_compl { x | f' x ≠ 0 }
      · have Z := hg.ae_eq_mk
        rw [EventuallyEq, ae_withDensity_iff_ae_restrict hf.measurable_mk] at Z
        filter_upwards [Z]
        intro x hx
        simp only [g', hx, Pi.mul_apply]
      · have M : MeasurableSet { x : α | f' x ≠ 0 }ᶜ :=
          (hf.measurable_mk (measurableSet_singleton 0).compl).compl
        filter_upwards [ae_restrict_mem M]
        intro x hx
        simp only [Classical.not_not, mem_setOf_eq, mem_compl_iff] at hx
        simp only [hx, zero_mul, Pi.mul_apply]
    _ = ∫⁻ a : α, (f * g) a ∂μ := by
      apply lintegral_congr_ae
      filter_upwards [hf.ae_eq_mk]
      intro x hx
      simp only [f', hx, Pi.mul_apply]


lemma setLIntegral_withDensity_eq_lintegral_mul₀' {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) {g : α → ℝ≥0∞} (hg : AEMeasurable g (μ.withDensity f))
    {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ a in s, g a ∂μ.withDensity f = ∫⁻ a in s, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((μ.withDensity f).restrict s) fun a => g a) (Me …
  -/
  rw [restrict_withDensity hs, lintegral_withDensity_eq_lintegral_mul₀' hf.restrict]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    s : Set α
    hs : MeasurableSet s
    ⊢ AEMeasurable g ((μ.restrict s).withDensity f)
  -/
  rw [← restrict_withDensity hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    g : α → ENNReal
    hg : AEMeasurable g (μ.withDensity f)
    s : Set α
    hs : MeasurableSet s
    ⊢ AEMeasurable g ((μ.withDensity f).restrict s)
  -/
  exact hg.restrict
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_lintegral_mul₀' := setLIntegral_withDensity_eq_lintegral_mul₀'


theorem lintegral_withDensity_eq_lintegral_mul₀ {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) {g : α → ℝ≥0∞} (hg : AEMeasurable g μ) :
    ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, (f * g) a ∂μ :=
  lintegral_withDensity_eq_lintegral_mul₀' hf (hg.mono' (withDensity_absolutelyContinuous μ f))


lemma setLIntegral_withDensity_eq_lintegral_mul₀ {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) {g : α → ℝ≥0∞} (hg : AEMeasurable g μ)
    {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ a in s, g a ∂μ.withDensity f = ∫⁻ a in s, (f * g) a ∂μ :=
  setLIntegral_withDensity_eq_lintegral_mul₀' hf
    (hg.mono' (MeasureTheory.withDensity_absolutelyContinuous μ f)) hs


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_lintegral_mul₀ := setLIntegral_withDensity_eq_lintegral_mul₀


theorem lintegral_withDensity_le_lintegral_mul (μ : Measure α) {f : α → ℝ≥0∞}
    (f_meas : Measurable f) (g : α → ℝ≥0∞) : (∫⁻ a, g a ∂μ.withDensity f) ≤ ∫⁻ a, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheor …
  -/
  rw [← iSup_lintegral_measurable_le_eq_lintegral, ← iSup_lintegral_measurable_le_eq_lintegral]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g : α → ENNReal
    ⊢ LE.le (iSup fun g_1 => iSup fun x => iSup fun x => MeasureTheory.lintegral ( …
  -/
  refine iSup₂_le fun i i_meas => iSup_le fun hi => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i g
    ⊢ LE.le (MeasureTheory.lintegral (μ.withDensity f) fun a => i a) (iSup fun g_1 …
  -/
  have A : f * i ≤ f * g := fun x => mul_le_mul_left' (hi x) _
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i g
    A : LE.le (HMul.hMul f i) (HMul.hMul f g)
    ⊢ LE.le (MeasureTheory.lintegral (μ.withDensity f) fun a => i a) (iSup fun g_1 …
  -/
  refine le_iSup₂_of_le (f * i) (f_meas.mul i_meas) ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i g
    A : LE.le (HMul.hMul f i) (HMul.hMul f g)
    ⊢ LE.le (MeasureTheory.lintegral (μ.withDensity f) fun a => i a) (iSup fun x = …
  -/
  exact le_iSup_of_le A (le_of_eq (lintegral_withDensity_eq_lintegral_mul _ f_meas i_meas))
  /-
    🎉 no goals
  -/


theorem lintegral_withDensity_eq_lintegral_mul_non_measurable (μ : Measure α) {f : α → ℝ≥0∞}
    (f_meas : Measurable f) (hf : ∀ᵐ x ∂μ, f x < ∞) (g : α → ℝ≥0∞) :
    ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheory.l …
  -/
  refine le_antisymm (lintegral_withDensity_le_lintegral_mul μ f_meas g) ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (MeasureTheory.li …
  -/
  rw [← iSup_lintegral_measurable_le_eq_lintegral, ← iSup_lintegral_measurable_le_eq_lintegral]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → ENNReal
    ⊢ LE.le (iSup fun g_1 => iSup fun x => iSup fun x => MeasureTheory.lintegral μ …
  -/
  refine iSup₂_le fun i i_meas => iSup_le fun hi => ?_
  have A : (fun x => (f x)⁻¹ * i x) ≤ g := by
    intro x
    dsimp
    rw [mul_comm, ← div_eq_mul_inv]
    exact div_le_of_le_mul' (hi x)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => i a) (iSup fun g_1 => iSup fun x = …
  -/
  refine le_iSup_of_le (fun x => (f x)⁻¹ * i x) (le_iSup_of_le (f_meas.inv.mul i_meas) ?_)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => i a) (iSup fun x => MeasureTheory. …
  -/
  refine le_iSup_of_le A ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => i a) (MeasureTheory.lintegral (μ.w …
  -/
  rw [lintegral_withDensity_eq_lintegral_mul _ f_meas (f_meas.inv.mul i_meas)]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => i a) (MeasureTheory.lintegral μ fu …
  -/
  apply lintegral_mono_ae
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ Filter.Eventually (fun a => LE.le (i a) (HMul.hMul f (fun a => HMul.hMul (In …
  -/
  filter_upwards [hf]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    ⊢ ∀ (a : α), LT.lt (f a) Top.top → LE.le (i a) (HMul.hMul f (fun a => HMul.hMu …
  -/
  intro x h'x
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g i : α → ENNReal
    i_meas : Measurable i
    hi : LE.le i (HMul.hMul f g)
    A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
    x : α
    h'x : LT.lt (f x) Top.top
    ⊢ LE.le (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
  -/
  rcases eq_or_ne (f x) 0 with (hx | hx)
    /-
      case h.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Eq (f x) 0
      ⊢ LE.le (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
    -/
  · have := hi x
    /-
      case h.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Eq (f x) 0
      this : LE.le (i x) (HMul.hMul f g x)
      ⊢ LE.le (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
    -/
    simp only [hx, zero_mul, Pi.mul_apply, nonpos_iff_eq_zero] at this
    /-
      case h.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Eq (f x) 0
      this : Eq (i x) 0
      ⊢ LE.le (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
    -/
    simp [this]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Ne (f x) 0
      ⊢ LE.le (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
    -/
  · apply le_of_eq _
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Ne (f x) 0
      ⊢ Eq (i x) (HMul.hMul f (fun a => HMul.hMul (Inv.inv (f a)) (i a)) x)
    -/
    dsimp
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      f_meas : Measurable f
      hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g i : α → ENNReal
      i_meas : Measurable i
      hi : LE.le i (HMul.hMul f g)
      A : LE.le (fun x => HMul.hMul (Inv.inv (f x)) (i x)) g
      x : α
      h'x : LT.lt (f x) Top.top
      hx : Ne (f x) 0
      ⊢ Eq (i x) (HMul.hMul (f x) (HMul.hMul (Inv.inv (f x)) (i x)))
    -/
    rw [← mul_assoc, ENNReal.mul_inv_cancel hx h'x.ne, one_mul]
    /-
      🎉 no goals
    -/


theorem setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable (μ : Measure α) {f : α → ℝ≥0∞}
    (f_meas : Measurable f) (g : α → ℝ≥0∞) {s : Set α} (hs : MeasurableSet s)
    (hf : ∀ᵐ x ∂μ.restrict s, f x < ∞) :
    ∫⁻ a in s, g a ∂μ.withDensity f = ∫⁻ a in s, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    f_meas : Measurable f
    g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae (μ.res …
    ⊢ Eq (MeasureTheory.lintegral ((μ.withDensity f).restrict s) fun a => g a) (Me …
  -/
  rw [restrict_withDensity hs, lintegral_withDensity_eq_lintegral_mul_non_measurable _ f_meas hf]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_set_lintegral_mul_non_measurable :=
  setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable


theorem lintegral_withDensity_eq_lintegral_mul_non_measurable₀ (μ : Measure α) {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (h'f : ∀ᵐ x ∂μ, f x < ∞) (g : α → ℝ≥0∞) :
    ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.withDensity f) fun a => g a) (MeasureTheory.l …
  -/
  let f' := hf.mk f
  calc
    ∫⁻ a, g a ∂μ.withDensity f = ∫⁻ a, g a ∂μ.withDensity f' := by
      rw [withDensity_congr_ae hf.ae_eq_mk]
    _ = ∫⁻ a, (f' * g) a ∂μ := by
      apply lintegral_withDensity_eq_lintegral_mul_non_measurable _ hf.measurable_mk
      filter_upwards [h'f, hf.ae_eq_mk]
      intro x hx h'x
      rwa [← h'x]
    _ = ∫⁻ a, (f * g) a ∂μ := by
      apply lintegral_congr_ae
      filter_upwards [hf.ae_eq_mk]
      intro x hx
      simp only [f', hx, Pi.mul_apply]


theorem setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀ (μ : Measure α)
    {f : α → ℝ≥0∞} {s : Set α} (hf : AEMeasurable f (μ.restrict s)) (g : α → ℝ≥0∞)
    (hs : MeasurableSet s) (h'f : ∀ᵐ x ∂μ.restrict s, f x < ∞) :
    ∫⁻ a in s, g a ∂μ.withDensity f = ∫⁻ a in s, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hf : AEMeasurable f (μ.restrict s)
    g : α → ENNReal
    hs : MeasurableSet s
    h'f : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae (μ.re …
    ⊢ Eq (MeasureTheory.lintegral ((μ.withDensity f).restrict s) fun a => g a) (Me …
  -/
  rw [restrict_withDensity hs, lintegral_withDensity_eq_lintegral_mul_non_measurable₀ _ hf h'f]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_set_lintegral_mul_non_measurable₀ :=
  setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀


theorem setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀' (μ : Measure α) [SFinite μ]
    {f : α → ℝ≥0∞} (s : Set α) (hf : AEMeasurable f (μ.restrict s)) (g : α → ℝ≥0∞)
    (h'f : ∀ᵐ x ∂μ.restrict s, f x < ∞) :
    ∫⁻ a in s, g a ∂μ.withDensity f = ∫⁻ a in s, (f * g) a ∂μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    s : Set α
    hf : AEMeasurable f (μ.restrict s)
    g : α → ENNReal
    h'f : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae (μ.re …
    ⊢ Eq (MeasureTheory.lintegral ((μ.withDensity f).restrict s) fun a => g a) (Me …
  -/
  rw [restrict_withDensity' s, lintegral_withDensity_eq_lintegral_mul_non_measurable₀ _ hf h'f]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_withDensity_eq_set_lintegral_mul_non_measurable₀' :=
  setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀'


theorem withDensity_mul₀ {μ : Measure α} {f g : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    μ.withDensity (f * g) = (μ.withDensity f).withDensity g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ Eq (μ.withDensity (HMul.hMul f g)) ((μ.withDensity f).withDensity g)
  -/
  ext1 s hs
  rw [withDensity_apply _ hs, withDensity_apply _ hs, restrict_withDensity hs,
    lintegral_withDensity_eq_lintegral_mul₀ hf.restrict hg.restrict]


theorem withDensity_mul (μ : Measure α) {f g : α → ℝ≥0∞} (hf : Measurable f) (hg : Measurable g) :
    μ.withDensity (f * g) = (μ.withDensity f).withDensity g :=
  withDensity_mul₀ hf.aemeasurable hg.aemeasurable


lemma withDensity_inv_same_le {μ : Measure α} {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
    (μ.withDensity f).withDensity f⁻¹ ≤ μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ LE.le ((μ.withDensity f).withDensity (Inv.inv f)) μ
  -/
  change (μ.withDensity f).withDensity (fun x ↦ (f x)⁻¹) ≤ μ
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ LE.le ((μ.withDensity f).withDensity fun x => Inv.inv (f x)) μ
  -/
  rw [← withDensity_mul₀ hf hf.inv]
  suffices (f * fun x ↦ (f x)⁻¹) ≤ᵐ[μ] 1 by
    refine (withDensity_mono this).trans ?_
    rw [withDensity_one]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyLE (HMul.hMul f fun x => Inv.inv (f x)) 1
  -/
  filter_upwards with x
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    x : α
    ⊢ LE.le (HMul.hMul f (fun x => Inv.inv (f x)) x) (1 x)
  -/
  simp only [Pi.mul_apply, Pi.one_apply]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    x : α
    ⊢ LE.le (HMul.hMul (f x) (Inv.inv (f x))) 1
  -/
  by_cases hx_top : f x = ∞
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : AEMeasurable f μ
      x : α
      hx_top : Eq (f x) Top.top
      ⊢ LE.le (HMul.hMul (f x) (Inv.inv (f x))) 1
    -/
  · simp only [hx_top, ENNReal.inv_top, mul_zero, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    x : α
    hx_top : Not (Eq (f x) Top.top)
    ⊢ LE.le (HMul.hMul (f x) (Inv.inv (f x))) 1
  -/
  by_cases hx_zero : f x = 0
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : AEMeasurable f μ
      x : α
      hx_top : Not (Eq (f x) Top.top)
      hx_zero : Eq (f x) 0
      ⊢ LE.le (HMul.hMul (f x) (Inv.inv (f x))) 1
    -/
  · simp only [hx_zero, ENNReal.inv_zero, zero_mul, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    x : α
    hx_top : Not (Eq (f x) Top.top)
    hx_zero : Not (Eq (f x) 0)
    ⊢ LE.le (HMul.hMul (f x) (Inv.inv (f x))) 1
  -/
  rw [ENNReal.mul_inv_cancel hx_zero hx_top]
  /-
    🎉 no goals
  -/


lemma withDensity_inv_same₀ {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_ne_zero : ∀ᵐ x ∂μ, f x ≠ 0) (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    (μ.withDensity f).withDensity (fun x ↦ (f x)⁻¹) = μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ Eq ((μ.withDensity f).withDensity fun x => Inv.inv (f x)) μ
  -/
  rw [← withDensity_mul₀ hf hf.inv]
  suffices (f * fun x ↦ (f x)⁻¹) =ᵐ[μ] 1 by
    rw [withDensity_congr_ae this, withDensity_one]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f fun x => Inv.inv (f x)) 1
  -/
  filter_upwards [hf_ne_zero, hf_ne_top] with x hf_ne_zero hf_ne_top
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero✝ : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    hf_ne_top✝ : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    x : α
    hf_ne_zero : Ne (f x) 0
    hf_ne_top : Ne (f x) Top.top
    ⊢ Eq (HMul.hMul f (fun x => Inv.inv (f x)) x) (1 x)
  -/
  simp only [Pi.mul_apply]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero✝ : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    hf_ne_top✝ : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    x : α
    hf_ne_zero : Ne (f x) 0
    hf_ne_top : Ne (f x) Top.top
    ⊢ Eq (HMul.hMul (f x) (Inv.inv (f x))) (1 x)
  -/
  rw [ENNReal.mul_inv_cancel hf_ne_zero hf_ne_top, Pi.one_apply]
  /-
    🎉 no goals
  -/


lemma withDensity_inv_same {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : Measurable f) (hf_ne_zero : ∀ᵐ x ∂μ, f x ≠ 0) (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    (μ.withDensity f).withDensity (fun x ↦ (f x)⁻¹) = μ :=
  withDensity_inv_same₀ hf.aemeasurable hf_ne_zero hf_ne_top


/-- If `f` is almost everywhere positive, then `μ ≪ μ.withDensity f`. See also
`withDensity_absolutelyContinuous` for the reverse direction, which always holds. -/
lemma withDensity_absolutelyContinuous' {μ : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_ne_zero : ∀ᵐ x ∂μ, f x ≠ 0) :
    μ ≪ μ.withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    ⊢ μ.AbsolutelyContinuous (μ.withDensity f)
  -/
  refine Measure.AbsolutelyContinuous.mk (fun s hs hμs ↦ ?_)
  rw [withDensity_apply _ hs, lintegral_eq_zero_iff' hf.restrict,
    ae_eq_restrict_iff_indicator_ae_eq hs, Set.indicator_zero', Filter.EventuallyEq, ae_iff] at hμs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae μ)
    s : Set α
    hs : MeasurableSet s
    hμs : Eq (μ (setOf fun a => Not (Eq (s.indicator f a) (0 a)))) 0
    ⊢ Eq (μ s) 0
  -/
  simp only [ae_iff, ne_eq, not_not] at hf_ne_zero
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    s : Set α
    hs : MeasurableSet s
    hμs : Eq (μ (setOf fun a => Not (Eq (s.indicator f a) (0 a)))) 0
    hf_ne_zero : Eq (μ (setOf fun a => Eq (f a) 0)) 0
    ⊢ Eq (μ s) 0
  -/
  simp only [Pi.zero_apply, Set.indicator_apply_eq_zero, not_forall, exists_prop] at hμs
  have hle : s ⊆ {a | a ∈ s ∧ ¬f a = 0} ∪ {a | f a = 0} :=
    fun x hx ↦ or_iff_not_imp_right.mpr <| fun hnx ↦ ⟨hx, hnx⟩
  exact measure_mono_null hle <| nonpos_iff_eq_zero.1 <| le_trans (measure_union_le _ _)
    <| hμs.symm ▸ zero_add _ |>.symm ▸ hf_ne_zero.le


theorem withDensity_ae_eq {β : Type} {f g : α → β} {d : α → ℝ≥0∞}
    (hd : AEMeasurable d μ) (h_ae_nonneg : ∀ᵐ x ∂μ, d x ≠ 0) :
    f =ᵐ[μ.withDensity d] g ↔ f =ᵐ[μ] g :=
  Iff.intro
  (fun h ↦ Measure.AbsolutelyContinuous.ae_eq
    (withDensity_absolutelyContinuous' hd h_ae_nonneg) h)
  (fun h ↦ Measure.AbsolutelyContinuous.ae_eq
    (withDensity_absolutelyContinuous μ d) h)


/-- If `μ` is a σ-finite measure, then so is `μ.withDensity fun x ↦ f x`
for any `ℝ≥0`-valued function `f`. -/
protected instance SigmaFinite.withDensity [SigmaFinite μ] (f : α → ℝ≥0) :
    SigmaFinite (μ.withDensity (fun x ↦ f x)) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    ⊢ MeasureTheory.SigmaFinite (μ.withDensity fun x => ↑(f x))
  -/
  refine ⟨⟨⟨fun n ↦ spanningSets μ n ∩ f ⁻¹' (Iic n), fun _ ↦ trivial, fun n ↦ ?_, ?_⟩⟩⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      n : Nat
      ⊢ LT.lt ((μ.withDensity fun x => ↑(f x)) ((fun n => Inter.inter (MeasureTheory …
    -/
  · rw [withDensity_apply']
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      n : Nat
      ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict ((fun n => Inter.inter (MeasureTh …
    -/
    apply setLIntegral_lt_top_of_bddAbove
      /-
        case refine_1.hs
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        n : Nat
        ⊢ Ne (μ ((fun n => Inter.inter (MeasureTheory.spanningSets μ n) (Set.preimage  …
      -/
    · exact ((measure_mono inter_subset_left).trans_lt (measure_spanningSets_lt_top μ n)).ne
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hbdd
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        n : Nat
        ⊢ BddAbove (Set.image f ((fun n => Inter.inter (MeasureTheory.spanningSets μ n …
      -/
    · exact ⟨n, forall_mem_image.2 fun x hx ↦ hx.2⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      ⊢ Eq (Set.iUnion fun i => (fun n => Inter.inter (MeasureTheory.spanningSets μ  …
    -/
  · rw [iUnion_eq_univ_iff]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      ⊢ ∀ (x : α), Exists fun i => Membership.mem (Inter.inter (MeasureTheory.spanni …
    -/
    refine fun x ↦ ⟨max (spanningSetsIndex μ x) ⌈f x⌉₊, ?_, ?_⟩
      /-
        case refine_2.refine_1
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        x : α
        ⊢ Membership.mem (MeasureTheory.spanningSets μ (Max.max (MeasureTheory.spannin …
      -/
    · exact mem_spanningSets_of_index_le _ _ (le_max_left ..)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        x : α
        ⊢ Membership.mem (Set.preimage f (Set.Iic ↑(Max.max (MeasureTheory.spanningSet …
      -/
    · simp [Nat.le_ceil]
      /-
        🎉 no goals
      -/


lemma SigmaFinite.withDensity_of_ne_top [SigmaFinite μ] {f : α → ℝ≥0∞}
    (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) : SigmaFinite (μ.withDensity f) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ MeasureTheory.SigmaFinite (μ.withDensity f)
  -/
  have : f =ᵐ[μ] fun x ↦ (f x).toNNReal := hf_ne_top.mono fun x hx ↦ (ENNReal.coe_toNNReal hx).symm
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    this : (MeasureTheory.ae μ).EventuallyEq f fun x => ↑(f x).toNNReal
    ⊢ MeasureTheory.SigmaFinite (μ.withDensity f)
  -/
  rw [withDensity_congr_ae this]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    this : (MeasureTheory.ae μ).EventuallyEq f fun x => ↑(f x).toNNReal
    ⊢ MeasureTheory.SigmaFinite (μ.withDensity fun x => ↑(f x).toNNReal)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma SigmaFinite.withDensity_of_ne_top' [SigmaFinite μ] {f : α → ℝ≥0∞} (hf_ne_top : ∀ x, f x ≠ ∞) :
    SigmaFinite (μ.withDensity f) :=
  SigmaFinite.withDensity_of_ne_top <| ae_of_all _ hf_ne_top


instance SigmaFinite.withDensity_ofReal [SigmaFinite μ] (f : α → ℝ) :
    SigmaFinite (μ.withDensity (fun x ↦ ENNReal.ofReal (f x))) :=
  .withDensity _


variable (μ) in
theorem exists_measurable_le_withDensity_eq [SFinite μ] (f : α → ℝ≥0∞) :
    ∃ g, Measurable g ∧ g ≤ f ∧ μ.withDensity g = μ.withDensity f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (μ.withDensity g) (μ …
  -/
  obtain ⟨g, hgm, hgf, hint⟩ := exists_measurable_le_forall_setLIntegral_eq μ f
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f g : α → ENNReal
    hgm : Measurable g
    hgf : LE.le g f
    hint : ∀ (s : Set α), Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) …
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (μ.withDensity g) (μ …
  -/
  use g, hgm, hgf
  /-
    case right
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f g : α → ENNReal
    hgm : Measurable g
    hgf : LE.le g f
    hint : ∀ (s : Set α), Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) …
    ⊢ Eq (μ.withDensity g) (μ.withDensity f)
  -/
  ext s hs
  /-
    case right.h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f g : α → ENNReal
    hgm : Measurable g
    hgf : LE.le g f
    hint : ∀ (s : Set α), Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) …
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity g) s) ((μ.withDensity f) s)
  -/
  simp only [hint, withDensity_apply _ hs]
  /-
    🎉 no goals
  -/


/-- If `μ` is an `s`-finite measure, then so is `μ.withDensity f`. -/
instance Measure.withDensity.instSFinite [SFinite μ] {f : α → ℝ≥0∞} :
    SFinite (μ.withDensity f) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    ⊢ MeasureTheory.SFinite (μ.withDensity f)
  -/
  wlog hfm : Measurable f generalizing f
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      this : ∀ {f : α → ENNReal}, Measurable f → MeasureTheory.SFinite (μ.withDensit …
      hfm : Not (Measurable f)
      ⊢ MeasureTheory.SFinite (μ.withDensity f)
    -/
  · rcases exists_measurable_le_withDensity_eq μ f with ⟨g, hgm, -, h⟩
    /-
      case inr.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      this : ∀ {f : α → ENNReal}, Measurable f → MeasureTheory.SFinite (μ.withDensit …
      hfm : Not (Measurable f)
      g : α → ENNReal
      hgm : Measurable g
      h : Eq (μ.withDensity g) (μ.withDensity f)
      ⊢ MeasureTheory.SFinite (μ.withDensity f)
    -/
    exact h ▸ this hgm
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    hfm : Measurable f
    ⊢ MeasureTheory.SFinite (μ.withDensity f)
  -/
  wlog hμ : IsFiniteMeasure μ generalizing μ
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      hfm : Measurable f
      this : ∀ {μ : MeasureTheory.Measure α} [inst : MeasureTheory.SFinite μ], Measu …
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ MeasureTheory.SFinite (μ.withDensity f)
    -/
  · rw [← sum_sfiniteSeq μ, withDensity_sum]
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      hfm : Measurable f
      this : ∀ {μ : MeasureTheory.Measure α} [inst : MeasureTheory.SFinite μ], Measu …
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun n => (MeasureTheory.sfi …
    -/
    have (n : ℕ) : SFinite ((sfiniteSeq μ n).withDensity f) := this inferInstance
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      hfm : Measurable f
      this✝ : ∀ {μ : MeasureTheory.Measure α} [inst : MeasureTheory.SFinite μ], Meas …
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      this : ∀ (n : Nat), MeasureTheory.SFinite ((MeasureTheory.sfiniteSeq μ n).with …
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun n => (MeasureTheory.sfi …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    ⊢ MeasureTheory.SFinite (μ.withDensity f)
  -/
  set s := {x | f x = ∞}
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Set α := setOf fun x => Eq (f x) Top.top
    ⊢ MeasureTheory.SFinite (μ.withDensity f)
  -/
  have hs : MeasurableSet s := hfm (measurableSet_singleton _)
  have key := calc
    μ.withDensity f = μ.withDensity (sᶜ.indicator f) + μ.withDensity (s.indicator f) := by
      simp (disch := measurability) [withDensity_indicator, ← restrict_withDensity]
    _ = μ.withDensity (sᶜ.indicator f) + .sum fun _ : ℕ ↦ μ.withDensity (s.indicator 1) := by
      rw [← withDensity_tsum (by measurability)]
      congr 2 with x
      rw [ENNReal.tsum_apply]
      if hx : x ∈ s then simpa [hx, ENNReal.tsum_const_eq_top_of_ne_zero]
      else simp [hx]
  have : SigmaFinite (μ.withDensity (sᶜ.indicator f)) := by
    refine SigmaFinite.withDensity_of_ne_top <| ae_of_all _ fun x hx ↦ ?_
    simp [indicator_apply, ite_eq_iff, s] at hx
  have : SigmaFinite (μ.withDensity (s.indicator 1)) := by
    rw [withDensity_indicator hs]
    exact SigmaFinite.withDensity 1
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Set α := setOf fun x => Eq (f x) Top.top
    hs : MeasurableSet s
    key : Eq (μ.withDensity f) (HAdd.hAdd (μ.withDensity ((HasCompl.compl s).indic …
    this✝ : MeasureTheory.SigmaFinite (μ.withDensity ((HasCompl.compl s).indicator …
    this : MeasureTheory.SigmaFinite (μ.withDensity (s.indicator 1))
    ⊢ MeasureTheory.SFinite (μ.withDensity f)
  -/
  rw [key]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Set α := setOf fun x => Eq (f x) Top.top
    hs : MeasurableSet s
    key : Eq (μ.withDensity f) (HAdd.hAdd (μ.withDensity ((HasCompl.compl s).indic …
    this✝ : MeasureTheory.SigmaFinite (μ.withDensity ((HasCompl.compl s).indicator …
    this : MeasureTheory.SigmaFinite (μ.withDensity (s.indicator 1))
    ⊢ MeasureTheory.SFinite (HAdd.hAdd (μ.withDensity ((HasCompl.compl s).indicato …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[deprecated Measure.withDensity.instSFinite (since := "2024-07-14"), nolint unusedArguments]
lemma sFinite_withDensity_of_sigmaFinite_of_measurable (μ : Measure α) [SigmaFinite μ]
    {f : α → ℝ≥0∞} (_hf : Measurable f) :
    SFinite (μ.withDensity f) :=
  inferInstance


@[deprecated Measure.withDensity.instSFinite (since := "2024-07-14"), nolint unusedArguments]
lemma sFinite_withDensity_of_measurable (μ : Measure α) [SFinite μ]
    {f : α → ℝ≥0∞} (_hf : Measurable f) :
    SFinite (μ.withDensity f) :=
  inferInstance


instance [SFinite μ] (c : ℝ≥0∞) : SFinite (c • μ) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    c : ENNReal
    ⊢ MeasureTheory.SFinite (HSMul.hSMul c μ)
  -/
  rw [← withDensity_const]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    c : ENNReal
    ⊢ MeasureTheory.SFinite (μ.withDensity fun x => c)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `μ ≪ ν` and `ν` is s-finite, then `μ` is s-finite. -/
theorem sFinite_of_absolutelyContinuous {ν : Measure α} [SFinite ν] (hμν : μ ≪ ν) :
    SFinite μ := by
  rw [← Measure.restrict_add_restrict_compl (μ := μ) measurableSet_sigmaFiniteSetWRT,
    restrict_compl_sigmaFiniteSetWRT hμν]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ MeasureTheory.SFinite (HAdd.hAdd (μ.restrict (μ.sigmaFiniteSetWRT ν)) (HSMul …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma IsLocallyFiniteMeasure.withDensity_coe {f : α → ℝ≥0} (hf : Continuous f) :
    IsLocallyFiniteMeasure (μ.withDensity fun x ↦ f x) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → NNReal
    hf : Continuous f
    ⊢ MeasureTheory.IsLocallyFiniteMeasure (μ.withDensity fun x => ↑(f x))
  -/
  refine ⟨fun x ↦ ?_⟩
  rcases (μ.finiteAt_nhds x).exists_mem_basis ((nhds_basis_opens' x).restrict_subset
    ((hf.tendsto x).eventually_le_const (lt_add_one _))) with ⟨U, ⟨⟨hUx, hUo⟩, hUf⟩, hμU⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → NNReal
    hf : Continuous f
    x : α
    U : Set α
    hμU : LT.lt (μ U) Top.top
    hUf : HasSubset.Subset U (setOf fun x_1 => (fun a => LE.le (f a) (HAdd.hAdd (f …
    hUx : Membership.mem (nhds x) U
    hUo : IsOpen U
    ⊢ (μ.withDensity fun x => ↑(f x)).FiniteAtFilter (nhds x)
  -/
  refine ⟨U, hUx, ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → NNReal
    hf : Continuous f
    x : α
    U : Set α
    hμU : LT.lt (μ U) Top.top
    hUf : HasSubset.Subset U (setOf fun x_1 => (fun a => LE.le (f a) (HAdd.hAdd (f …
    hUx : Membership.mem (nhds x) U
    hUo : IsOpen U
    ⊢ LT.lt ((μ.withDensity fun x => ↑(f x)) U) Top.top
  -/
  rw [withDensity_apply _ hUo.measurableSet]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → NNReal
    hf : Continuous f
    x : α
    U : Set α
    hμU : LT.lt (μ U) Top.top
    hUf : HasSubset.Subset U (setOf fun x_1 => (fun a => LE.le (f a) (HAdd.hAdd (f …
    hUx : Membership.mem (nhds x) U
    hUo : IsOpen U
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict U) fun a => ↑(f a)) Top.top
  -/
  exact setLIntegral_lt_top_of_bddAbove hμU.ne ⟨f x + 1, forall_mem_image.2 hUf⟩
  /-
    🎉 no goals
  -/


lemma IsLocallyFiniteMeasure.withDensity_ofReal {f : α → ℝ} (hf : Continuous f) :
    IsLocallyFiniteMeasure (μ.withDensity fun x ↦ .ofReal (f x)) :=
  .withDensity_coe <| continuous_real_toNNReal.comp hf


