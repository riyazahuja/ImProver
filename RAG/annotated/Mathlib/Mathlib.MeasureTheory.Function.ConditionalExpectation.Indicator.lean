theorem condexp_ae_eq_restrict_zero (hs : MeasurableSet[m] s) (hf : f =ᵐ[μ.restrict s] 0) :
    μ[f|m] =ᵐ[μ.restrict s] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  swap; · simp_rw [condexp_of_not_le hm]; rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  have : SigmaFinite ((μ.restrict s).trim hm) := by
    rw [← restrict_trim hm _ hs]
    exact Restrict.sigmaFinite _ s
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hf_int : Integrable f μ
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  swap; · rw [condexp_undef hf_int]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) 0
  -/
  refine ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' hm ?_ ?_ ?_ ?_ ?_
    /-
      case pos.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt ((μ.restrict s) s_1) Top.top → Me …
    -/
  · exact fun t _ _ => integrable_condexp.integrableOn.integrableOn
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt ((μ.restrict s) s_1) Top.top → Me …
    -/
  · exact fun t _ _ => (integrable_zero _ _ _).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_3
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt ((μ.restrict s) s_1) Top.top → Eq …
    -/
  · intro t ht _
    rw [Measure.restrict_restrict (hm _ ht), setIntegral_condexp hm hf_int (ht.inter hs), ←
      Measure.restrict_restrict (hm _ ht)]
    /-
      case pos.refine_3
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      t : Set α
      ht : MeasurableSet t
      a✝ : LT.lt ((μ.restrict s) t) Top.top
      ⊢ Eq (MeasureTheory.integral ((μ.restrict s).restrict t) fun x => f x) (Measur …
    -/
    refine setIntegral_congr_ae (hm _ ht) ?_
    /-
      case pos.refine_3
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      t : Set α
      ht : MeasurableSet t
      a✝ : LT.lt ((μ.restrict s) t) Top.top
      ⊢ Filter.Eventually (fun x => Membership.mem t x → Eq (f x) (0 x)) (MeasureThe …
    -/
    filter_upwards [hf] with x hx _ using hx
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_4
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m (MeasureTheory.condexp m μ f) (μ.restr …
    -/
  · exact stronglyMeasurable_condexp.aeStronglyMeasurable'
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_5
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      hf_int : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m 0 (μ.restrict s)
    -/
  · exact stronglyMeasurable_zero.aeStronglyMeasurable'
    /-
      🎉 no goals
    -/


/-- Auxiliary lemma for `condexp_indicator`. -/
theorem condexp_indicator_aux (hs : MeasurableSet[m] s) (hf : f =ᵐ[μ.restrict sᶜ] 0) :
    μ[s.indicator f|m] =ᵐ[μ] s.indicator (μ[f|m]) := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f 0
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  swap; · simp_rw [condexp_of_not_le hm, Set.indicator_zero']; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/
  have hsf_zero : ∀ g : α → E, g =ᵐ[μ.restrict sᶜ] 0 → s.indicator g =ᵐ[μ] g := fun g =>
    indicator_ae_eq_of_restrict_compl_ae_eq_zero (hm _ hs)
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f 0
    hm : LE.le m m0
    hsf_zero : ∀ (g : α → E), (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).E …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  refine ((hsf_zero (μ[f|m]) (condexp_ae_eq_restrict_zero hs.compl hf)).trans ?_).symm
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f 0
    hm : LE.le m m0
    hsf_zero : ∀ (g : α → E), (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).E …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  exact condexp_congr_ae (hsf_zero f hf).symm
  /-
    🎉 no goals
  -/


/-- The conditional expectation of the indicator of a function over an `m`-measurable set with
respect to the σ-algebra `m` is a.e. equal to the indicator of the conditional expectation. -/
theorem condexp_indicator (hf_int : Integrable f μ) (hs : MeasurableSet[m] s) :
    μ[s.indicator f|m] =ᵐ[μ] s.indicator (μ[f|m]) := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  swap; · simp_rw [condexp_of_not_le hm, Set.indicator_zero']; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm, Set.indicator_zero']; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  -- use `have` to perform what should be the first calc step because of an error I don't
  -- understand
  have : s.indicator (μ[f|m]) =ᵐ[μ] s.indicator (μ[s.indicator f + sᶜ.indicator f|m]) := by
    rw [Set.indicator_self_add_compl s f]
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hf_int : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    this : (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator f) …
  -/
  refine (this.trans ?_).symm
  calc
    s.indicator (μ[s.indicator f + sᶜ.indicator f|m]) =ᵐ[μ]
        s.indicator (μ[s.indicator f|m] + μ[sᶜ.indicator f|m]) := by
      have : μ[s.indicator f + sᶜ.indicator f|m] =ᵐ[μ] μ[s.indicator f|m] + μ[sᶜ.indicator f|m] :=
        condexp_add (hf_int.indicator (hm _ hs)) (hf_int.indicator (hm _ hs.compl))
      filter_upwards [this] with x hx
      classical rw [Set.indicator_apply, Set.indicator_apply, hx]
    _ = s.indicator (μ[s.indicator f|m]) + s.indicator (μ[sᶜ.indicator f|m]) :=
      (s.indicator_add' _ _)
    _ =ᵐ[μ] s.indicator (μ[s.indicator f|m]) +
        s.indicator (sᶜ.indicator (μ[sᶜ.indicator f|m])) := by
      refine Filter.EventuallyEq.rfl.add ?_
      have : sᶜ.indicator (μ[sᶜ.indicator f|m]) =ᵐ[μ] μ[sᶜ.indicator f|m] := by
        refine (condexp_indicator_aux hs.compl ?_).symm.trans ?_
        · exact indicator_ae_eq_restrict_compl (hm _ hs.compl)
        · rw [Set.indicator_indicator, Set.inter_self]
      filter_upwards [this] with x hx
      by_cases hxs : x ∈ s
      · simp only [hx, hxs, Set.indicator_of_mem]
      · simp only [hxs, Set.indicator_of_not_mem, not_false_iff]
    _ =ᵐ[μ] s.indicator (μ[s.indicator f|m]) := by
      rw [Set.indicator_indicator, Set.inter_compl_self, Set.indicator_empty', add_zero]
    _ =ᵐ[μ] μ[s.indicator f|m] := by
      refine (condexp_indicator_aux hs ?_).symm.trans ?_
      · exact indicator_ae_eq_restrict_compl (hm _ hs)
      · rw [Set.indicator_indicator, Set.inter_self]


theorem condexp_restrict_ae_eq_restrict (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (hs_m : MeasurableSet[m] s) (hf_int : Integrable f μ) :
    (μ.restrict s)[f|m] =ᵐ[μ.restrict s] μ[f|m] := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs_m : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m (μ.r …
  -/
  have : SigmaFinite ((μ.restrict s).trim hm) := by rw [← restrict_trim hm _ hs_m]; infer_instance
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs_m : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m (μ.r …
  -/
  rw [ae_eq_restrict_iff_indicator_ae_eq (hm _ hs_m)]
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs_m : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m (μ.r …
  -/
  refine EventuallyEq.trans ?_ (condexp_indicator hf_int hs_m)
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs_m : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m (μ.r …
  -/
  refine ae_eq_condexp_of_forall_setIntegral_eq hm (hf_int.indicator (hm _ hs_m)) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs_m : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → MeasureTheory.I …
    -/
  · intro t ht _
    rw [← integrable_indicator_iff (hm _ ht), Set.indicator_indicator, Set.inter_comm, ←
      Set.indicator_indicator]
    suffices h_int_restrict : Integrable (t.indicator ((μ.restrict s)[f|m])) (μ.restrict s) by
      rw [integrable_indicator_iff (hm _ hs_m), IntegrableOn]
      rw [integrable_indicator_iff (hm _ ht), IntegrableOn] at h_int_restrict ⊢
      exact h_int_restrict
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs_m : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      t : Set α
      ht : MeasurableSet t
      a✝ : LT.lt (μ t) Top.top
      ⊢ MeasureTheory.Integrable (t.indicator (MeasureTheory.condexp m (μ.restrict s …
    -/
    exact integrable_condexp.indicator (hm _ ht)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs_m : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → Eq (MeasureTheo …
    -/
  · intro t ht _
    calc
      ∫ x in t, s.indicator ((μ.restrict s)[f|m]) x ∂μ =
          ∫ x in t, ((μ.restrict s)[f|m]) x ∂μ.restrict s := by
        rw [integral_indicator (hm _ hs_m), Measure.restrict_restrict (hm _ hs_m),
          Measure.restrict_restrict (hm _ ht), Set.inter_comm]
      _ = ∫ x in t, f x ∂μ.restrict s := setIntegral_condexp hm hf_int.integrableOn ht
      _ = ∫ x in t, s.indicator f x ∂μ := by
        rw [integral_indicator (hm _ hs_m), Measure.restrict_restrict (hm _ hs_m),
          Measure.restrict_restrict (hm _ ht), Set.inter_comm]
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs_m : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      this : MeasureTheory.SigmaFinite ((μ.restrict s).trim hm)
      ⊢ MeasureTheory.AEStronglyMeasurable' m (s.indicator (MeasureTheory.condexp m  …
    -/
  · exact (stronglyMeasurable_condexp.indicator hs_m).aeStronglyMeasurable'
    /-
      🎉 no goals
    -/


/-- If the restriction to an `m`-measurable set `s` of a σ-algebra `m` is equal to the restriction
to `s` of another σ-algebra `m₂` (hypothesis `hs`), then `μ[f | m] =ᵐ[μ.restrict s] μ[f | m₂]`. -/
theorem condexp_ae_eq_restrict_of_measurableSpace_eq_on {m m₂ m0 : MeasurableSpace α}
    {μ : Measure α} (hm : m ≤ m0) (hm₂ : m₂ ≤ m0) [SigmaFinite (μ.trim hm)]
    [SigmaFinite (μ.trim hm₂)] (hs_m : MeasurableSet[m] s)
    (hs : ∀ t, MeasurableSet[m] (s ∩ t) ↔ MeasurableSet[m₂] (s ∩ t)) :
    μ[f|m] =ᵐ[μ.restrict s] μ[f|m₂] := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp m μ f) …
  -/
  rw [ae_eq_restrict_iff_indicator_ae_eq (hm _ hs_m)]
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m μ f) …
  -/
  have hs_m₂ : MeasurableSet[m₂] s := by rwa [← Set.inter_univ s, ← hs Set.univ, Set.inter_univ]
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m μ f) …
  -/
  by_cases hf_int : Integrable f μ
  /-
    case pos
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m μ f) …
  -/
  swap; · simp_rw [condexp_undef hf_int]; rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m μ f) …
  -/
  refine ((condexp_indicator hf_int hs_m).symm.trans ?_).trans (condexp_indicator hf_int hs_m₂)
  refine ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' hm₂
    (fun s _ _ => integrable_condexp.integrableOn)
    (fun s _ _ => integrable_condexp.integrableOn) ?_ ?_
    stronglyMeasurable_condexp.aeStronglyMeasurable'
  /-
    case pos.refine_1
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → Eq (MeasureTheo …
  -/
  swap
    /-
      case pos.refine_2
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      f : α → E
      s : Set α
      m m₂ m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hm₂ : LE.le m₂ m0
      inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
      hs_m : MeasurableSet s
      hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
      hs_m₂ : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m₂ (MeasureTheory.condexp m μ (s.indicat …
    -/
  · have : StronglyMeasurable[m] (μ[s.indicator f|m]) := stronglyMeasurable_condexp
    refine this.aeStronglyMeasurable'.aeStronglyMeasurable'_of_measurableSpace_le_on hm hs_m
      (fun t => (hs t).mp) ?_
    /-
      case pos.refine_2
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      f : α → E
      s : Set α
      m m₂ m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hm₂ : LE.le m₂ m0
      inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
      hs_m : MeasurableSet s
      hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
      hs_m₂ : MeasurableSet s
      hf_int : MeasureTheory.Integrable f μ
      this : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ (s.indicato …
      ⊢ (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq (MeasureTheo …
    -/
    exact condexp_ae_eq_restrict_zero hs_m.compl (indicator_ae_eq_restrict_compl (hm _ hs_m))
    /-
      🎉 no goals
    -/
  /-
    case pos.refine_1
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → Eq (MeasureTheo …
  -/
  intro t ht _
  have : ∫ x in t, (μ[s.indicator f|m]) x ∂μ = ∫ x in s ∩ t, (μ[s.indicator f|m]) x ∂μ := by
    rw [← integral_add_compl (hm _ hs_m) integrable_condexp.integrableOn]
    suffices ∫ x in sᶜ, (μ[s.indicator f|m]) x ∂μ.restrict t = 0 by
      rw [this, add_zero, Measure.restrict_restrict (hm _ hs_m)]
    rw [Measure.restrict_restrict (MeasurableSet.compl (hm _ hs_m))]
    suffices μ[s.indicator f|m] =ᵐ[μ.restrict sᶜ] 0 by
      rw [Set.inter_comm, ← Measure.restrict_restrict (hm₂ _ ht)]
      calc
        ∫ x : α in t, (μ[s.indicator f|m]) x ∂μ.restrict sᶜ =
            ∫ x : α in t, 0 ∂μ.restrict sᶜ := by
          refine setIntegral_congr_ae (hm₂ _ ht) ?_
          filter_upwards [this] with x hx _ using hx
        _ = 0 := integral_zero _ _
    refine condexp_ae_eq_restrict_zero hs_m.compl ?_
    exact indicator_ae_eq_restrict_compl (hm _ hs_m)
  /-
    case pos.refine_1
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : α → E
    s : Set α
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m0
    inst✝¹ : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), Iff (MeasurableSet (Inter.inter s t)) (MeasurableSet (Inte …
    hs_m₂ : MeasurableSet s
    hf_int : MeasureTheory.Integrable f μ
    t : Set α
    ht : MeasurableSet t
    a✝ : LT.lt (μ t) Top.top
    this : Eq (MeasureTheory.integral (μ.restrict t) fun x => MeasureTheory.condex …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => MeasureTheory.condexp m μ …
  -/
  have hst_m : MeasurableSet[m] (s ∩ t) := (hs _).mpr (hs_m₂.inter ht)
  simp_rw [this, setIntegral_condexp hm₂ (hf_int.indicator (hm _ hs_m)) ht,
    setIntegral_condexp hm (hf_int.indicator (hm _ hs_m)) hst_m, integral_indicator (hm _ hs_m),
    Measure.restrict_restrict (hm _ hs_m), ← Set.inter_assoc, Set.inter_self]


