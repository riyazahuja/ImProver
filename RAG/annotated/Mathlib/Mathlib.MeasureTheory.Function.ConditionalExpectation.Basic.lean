/-- Conditional expectation of a function. It is defined as 0 if any one of the following conditions
is true:
- `m` is not a sub-σ-algebra of `m0`,
- `μ` is not σ-finite with respect to `m`,
- `f` is not integrable. -/
noncomputable irreducible_def condexp (m : MeasurableSpace α) {m0 : MeasurableSpace α}
    (μ : Measure α) (f : α → F') : α → F' :=
  if hm : m ≤ m0 then
    if h : SigmaFinite (μ.trim hm) ∧ Integrable f μ then
      if StronglyMeasurable[m] f then f
      else (@aestronglyMeasurable'_condexpL1 _ _ _ _ _ m m0 μ hm h.1 _).mk
        (@condexpL1 _ _ _ _ _ _ _ hm μ h.1 f)
    else 0
  else 0

-- We define notation `μ[f|m]` for the conditional expectation of `f` with respect to `m`.

scoped notation μ "[" f "|" m "]" => MeasureTheory.condexp m μ f


                                                                /-
                                                                  α : Type u_1
                                                                  F' : Type u_3
                                                                  inst✝² : NormedAddCommGroup F'
                                                                  inst✝¹ : NormedSpace Real F'
                                                                  inst✝ : CompleteSpace F'
                                                                  m m0 : MeasurableSpace α
                                                                  μ : MeasureTheory.Measure α
                                                                  f : α → F'
                                                                  hm_not : Not (LE.le m m0)
                                                                  ⊢ Eq (MeasureTheory.condexp m μ f) 0
                                                                -/
theorem condexp_of_not_le (hm_not : ¬m ≤ m0) : μ[f|m] = 0 := by rw [condexp, dif_neg hm_not]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem condexp_of_not_sigmaFinite (hm : m ≤ m0) (hμm_not : ¬SigmaFinite (μ.trim hm)) :
                     /-
                       α : Type u_1
                       F' : Type u_3
                       inst✝² : NormedAddCommGroup F'
                       inst✝¹ : NormedSpace Real F'
                       inst✝ : CompleteSpace F'
                       m m0 : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       f : α → F'
                       hm : LE.le m m0
                       hμm_not : Not (MeasureTheory.SigmaFinite (μ.trim hm))
                       ⊢ Eq (MeasureTheory.condexp m μ f) 0
                     -/
    μ[f|m] = 0 := by rw [condexp, dif_pos hm, dif_neg]; push_neg; exact fun h => absurd h hμm_not
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem condexp_of_sigmaFinite (hm : m ≤ m0) [hμm : SigmaFinite (μ.trim hm)] :
    μ[f|m] =
      if Integrable f μ then
        if StronglyMeasurable[m] f then f
        else aestronglyMeasurable'_condexpL1.mk (condexpL1 hm μ f)
      else 0 := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ f) (ite (MeasureTheory.Integrable f μ) (ite (M …
  -/
  rw [condexp, dif_pos hm]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (dite (And (MeasureTheory.SigmaFinite (μ.trim hm)) (MeasureTheory.Integra …
  -/
  simp only [hμm, Ne, true_and]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (dite (MeasureTheory.Integrable f μ) (fun h => ite (MeasureTheory.Strongl …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (dite (MeasureTheory.Integrable f μ) (fun h => ite (MeasureTheory.Strongl …
    -/
  · rw [dif_pos hf, if_pos hf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (dite (MeasureTheory.Integrable f μ) (fun h => ite (MeasureTheory.Strongl …
    -/
  · rw [dif_neg hf, if_neg hf]
    /-
      🎉 no goals
    -/


theorem condexp_of_stronglyMeasurable (hm : m ≤ m0) [hμm : SigmaFinite (μ.trim hm)] {f : α → F'}
    (hf : StronglyMeasurable[m] f) (hfi : Integrable f μ) : μ[f|m] = f := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hf : MeasureTheory.StronglyMeasurable f
    hfi : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.condexp m μ f) f
  -/
  rw [condexp_of_sigmaFinite hm, if_pos hfi, if_pos hf]
  /-
    🎉 no goals
  -/


theorem condexp_const (hm : m ≤ m0) (c : F') [IsFiniteMeasure μ] :
    μ[fun _ : α => c|m] = fun _ => c :=
  condexp_of_stronglyMeasurable hm (@stronglyMeasurable_const _ _ m _ _) (integrable_const c)


theorem condexp_ae_eq_condexpL1 (hm : m ≤ m0) [hμm : SigmaFinite (μ.trim hm)] (f : α → F') :
    μ[f|m] =ᵐ[μ] condexpL1 hm μ f := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) ↑↑(MeasureTh …
  -/
  rw [condexp_of_sigmaFinite hm]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.Integrable f μ) (ite ( …
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hfi : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.Integrable f μ) (ite ( …
    -/
  · rw [if_pos hfi]
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hfi : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.StronglyMeasurable f)  …
    -/
    by_cases hfm : StronglyMeasurable[m] f
      /-
        case pos
        α : Type u_1
        F' : Type u_3
        inst✝² : NormedAddCommGroup F'
        inst✝¹ : NormedSpace Real F'
        inst✝ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        hμm : MeasureTheory.SigmaFinite (μ.trim hm)
        f : α → F'
        hfi : MeasureTheory.Integrable f μ
        hfm : MeasureTheory.StronglyMeasurable f
        ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.StronglyMeasurable f)  …
      -/
    · rw [if_pos hfm]
      exact (condexpL1_of_aestronglyMeasurable' (StronglyMeasurable.aeStronglyMeasurable' hfm)
        hfi).symm
      /-
        case neg
        α : Type u_1
        F' : Type u_3
        inst✝² : NormedAddCommGroup F'
        inst✝¹ : NormedSpace Real F'
        inst✝ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        hμm : MeasureTheory.SigmaFinite (μ.trim hm)
        f : α → F'
        hfi : MeasureTheory.Integrable f μ
        hfm : Not (MeasureTheory.StronglyMeasurable f)
        ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.StronglyMeasurable f)  …
      -/
    · rw [if_neg hfm]
      /-
        case neg
        α : Type u_1
        F' : Type u_3
        inst✝² : NormedAddCommGroup F'
        inst✝¹ : NormedSpace Real F'
        inst✝ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        hμm : MeasureTheory.SigmaFinite (μ.trim hm)
        f : α → F'
        hfi : MeasureTheory.Integrable f μ
        hfm : Not (MeasureTheory.StronglyMeasurable f)
        ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.AEStronglyMeasurable'.mk ↑↑ …
      -/
      exact (AEStronglyMeasurable'.ae_eq_mk aestronglyMeasurable'_condexpL1).symm
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfi : Not (MeasureTheory.Integrable f μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (ite (MeasureTheory.Integrable f μ) (ite ( …
  -/
  rw [if_neg hfi, condexpL1_undef hfi]
  /-
    case neg
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfi : Not (MeasureTheory.Integrable f μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq 0 ↑↑0
  -/
  exact (coeFn_zero _ _ _).symm
  /-
    🎉 no goals
  -/


theorem condexp_ae_eq_condexpL1CLM (hm : m ≤ m0) [SigmaFinite (μ.trim hm)] (hf : Integrable f μ) :
    μ[f|m] =ᵐ[μ] condexpL1CLM F' hm μ (hf.toL1 f) := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hf : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) ↑↑((MeasureT …
  -/
  refine (condexp_ae_eq_condexpL1 hm f).trans (Eventually.of_forall fun x => ?_)
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hf : MeasureTheory.Integrable f μ
    x : α
    ⊢ Eq (↑↑(MeasureTheory.condexpL1 hm μ f) x) (↑↑((MeasureTheory.condexpL1CLM F' …
  -/
  rw [condexpL1_eq hf]
  /-
    🎉 no goals
  -/


theorem condexp_undef (hf : ¬Integrable f μ) : μ[f|m] = 0 := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    hm : LE.le m m0
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  swap; · rw [condexp_of_not_le hm]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    hm : LE.le m m0
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  swap; · rw [condexp_of_not_sigmaFinite hm hμm]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hf : Not (MeasureTheory.Integrable f μ)
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ f) 0
  -/
  rw [condexp_of_sigmaFinite, if_neg hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem condexp_zero : μ[(0 : α → F')|m] = 0 := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.condexp m μ 0) 0
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    ⊢ Eq (MeasureTheory.condexp m μ 0) 0
  -/
  swap; · rw [condexp_of_not_le hm]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    ⊢ Eq (MeasureTheory.condexp m μ 0) 0
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ 0) 0
  -/
  swap; · rw [condexp_of_not_sigmaFinite hm hμm]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexp m μ 0) 0
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  exact
    condexp_of_stronglyMeasurable hm (@stronglyMeasurable_zero _ _ m _ _) (integrable_zero _ _ _)


theorem stronglyMeasurable_condexp : StronglyMeasurable[m] (μ[f|m]) := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  swap; · rw [condexp_of_not_le hm]; exact stronglyMeasurable_zero
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  swap; · rw [condexp_of_not_sigmaFinite hm hμm]; exact stronglyMeasurable_zero
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.condexp m μ f)
  -/
  rw [condexp_of_sigmaFinite hm]
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.StronglyMeasurable (ite (MeasureTheory.Integrable f μ) (ite (M …
  -/
  split_ifs with hfi hfm
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
      hfi : MeasureTheory.Integrable f μ
      hfm : MeasureTheory.StronglyMeasurable f
      ⊢ MeasureTheory.StronglyMeasurable f
    -/
  · exact hfm
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
      hfi : MeasureTheory.Integrable f μ
      hfm : Not (MeasureTheory.StronglyMeasurable f)
      ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable'.mk ↑↑( …
    -/
  · exact AEStronglyMeasurable'.stronglyMeasurable_mk _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ MeasureTheory.StronglyMeasurable 0
    -/
  · exact stronglyMeasurable_zero
    /-
      🎉 no goals
    -/


theorem condexp_congr_ae (h : f =ᵐ[μ] g) : μ[f|m] =ᵐ[μ] μ[g|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    h : (MeasureTheory.ae μ).EventuallyEq f g
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  swap; · simp_rw [condexp_of_not_le hm]; rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    h : (MeasureTheory.ae μ).EventuallyEq f g
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    h : (MeasureTheory.ae μ).EventuallyEq f g
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    h : (MeasureTheory.ae μ).EventuallyEq f g
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  exact (condexp_ae_eq_condexpL1 hm f).trans
    (Filter.EventuallyEq.trans (by rw [condexpL1_congr_ae hm h])
      (condexp_ae_eq_condexpL1 hm g).symm)


theorem condexp_of_aestronglyMeasurable' (hm : m ≤ m0) [hμm : SigmaFinite (μ.trim hm)] {f : α → F'}
    (hf : AEStronglyMeasurable' m f μ) (hfi : Integrable f μ) : μ[f|m] =ᵐ[μ] f := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) f
  -/
  refine ((condexp_congr_ae hf.ae_eq_mk).trans ?_).trans hf.ae_eq_mk.symm
  rw [condexp_of_stronglyMeasurable hm hf.stronglyMeasurable_mk
    ((integrable_congr hf.ae_eq_mk).mp hfi)]


theorem integrable_condexp : Integrable (μ[f|m]) μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  swap; · rw [condexp_of_not_le hm]; exact integrable_zero _ _ _
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  swap; · rw [condexp_of_not_sigmaFinite hm hμm]; exact integrable_zero _ _ _
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ MeasureTheory.Integrable (MeasureTheory.condexp m μ f) μ
  -/
  exact (integrable_condexpL1 f).congr (condexp_ae_eq_condexpL1 hm f).symm
  /-
    🎉 no goals
  -/


/-- The integral of the conditional expectation `μ[f|hm]` over an `m`-measurable set is equal to
the integral of `f` on that set. -/
theorem setIntegral_condexp (hm : m ≤ m0) [SigmaFinite (μ.trim hm)] (hf : Integrable f μ)
    (hs : MeasurableSet[m] s) : ∫ x in s, (μ[f|m]) x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hf : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => MeasureTheory.condexp m μ …
  -/
  rw [setIntegral_congr_ae (hm s hs) ((condexp_ae_eq_condexpL1 hm f).mono fun x hx _ => hx)]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    s : Set α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hf : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑(MeasureTheory.condexpL …
  -/
  exact setIntegral_condexpL1 hf hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")] alias set_integral_condexp := setIntegral_condexp


theorem integral_condexp (hm : m ≤ m0) [hμm : SigmaFinite (μ.trim hm)] :
    ∫ x, (μ[f|m]) x ∂μ = ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.condexp m μ f x) (Measur …
  -/
  by_cases hf : Integrable f μ
  · suffices ∫ x in Set.univ, (μ[f|m]) x ∂μ = ∫ x in Set.univ, f x ∂μ by
      simp_rw [setIntegral_univ] at this; exact this
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral (μ.restrict Set.univ) fun x => MeasureTheory.cond …
    -/
    exact setIntegral_condexp hm hf (@MeasurableSet.univ _ m)
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    hf : Not (MeasureTheory.Integrable f μ)
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.condexp m μ f x) (Measur …
  -/
  simp only [condexp_undef hf, Pi.zero_apply, integral_zero, integral_undef hf]
  /-
    🎉 no goals
  -/


/-- Total probability law using `condexp` as conditional probability. -/
theorem integral_condexp_indicator [mF : MeasurableSpace F] {Y : α → F} (hY : Measurable Y)
    [SigmaFinite (μ.trim hY.comap_le)] {A : Set α} (hA : MeasurableSet A) :
    ∫ x, (μ[(A.indicator fun _ ↦ (1 : ℝ)) | mF.comap Y]) x ∂μ = (μ A).toReal := by
  /-
    α : Type u_1
    F : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mF : MeasurableSpace F
    Y : α → F
    hY : Measurable Y
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    A : Set α
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.condexp (MeasurableSpace …
  -/
  rw [integral_condexp, integral_indicator hA, setIntegral_const, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


/-- **Uniqueness of the conditional expectation**
If a function is a.e. `m`-measurable, verifies an integrability condition and has same integral
as `f` on all `m`-measurable sets, then it is a.e. equal to `μ[f|hm]`. -/
theorem ae_eq_condexp_of_forall_setIntegral_eq (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    {f g : α → F'} (hf : Integrable f μ)
    (hg_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn g s μ)
    (hg_eq : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, g x ∂μ = ∫ x in s, f x ∂μ)
    (hgm : AEStronglyMeasurable' m g μ) : g =ᵐ[μ] μ[f|m] := by
  refine ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' hm hg_int_finite
    (fun s _ _ => integrable_condexp.integrableOn) (fun s hs hμs => ?_) hgm
    (StronglyMeasurable.aeStronglyMeasurable' stronglyMeasurable_condexp)
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheo …
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => g x) (MeasureTheory.integ …
  -/
  rw [hg_eq s hs hμs, setIntegral_condexp hm hf hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_condexp_of_forall_set_integral_eq := ae_eq_condexp_of_forall_setIntegral_eq


theorem condexp_bot' [hμ : NeZero μ] (f : α → F') :
    μ[f|⊥] = fun _ => (μ Set.univ).toReal⁻¹ • ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
  -/
  by_cases hμ_finite : IsFiniteMeasure μ
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
  -/
  swap
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : NeZero μ
      f : α → F'
      hμ_finite : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
    -/
  · have h : ¬SigmaFinite (μ.trim bot_le) := by rwa [sigmaFinite_trim_bot_iff]
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : NeZero μ
      f : α → F'
      hμ_finite : Not (MeasureTheory.IsFiniteMeasure μ)
      h : Not (MeasureTheory.SigmaFinite (μ.trim ⋯))
      ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
    -/
    rw [not_isFiniteMeasure_iff] at hμ_finite
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : NeZero μ
      f : α → F'
      hμ_finite : Eq (μ Set.univ) Top.top
      h : Not (MeasureTheory.SigmaFinite (μ.trim ⋯))
      ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
    -/
    rw [condexp_of_not_sigmaFinite bot_le h]
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : NeZero μ
      f : α → F'
      hμ_finite : Eq (μ Set.univ) Top.top
      h : Not (MeasureTheory.SigmaFinite (μ.trim ⋯))
      ⊢ Eq 0 fun x => HSMul.hSMul (Inv.inv (μ Set.univ).toReal) (MeasureTheory.integ …
    -/
    simp only [hμ_finite, ENNReal.top_toReal, inv_zero, zero_smul]
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : NeZero μ
      f : α → F'
      hμ_finite : Eq (μ Set.univ) Top.top
      h : Not (MeasureTheory.SigmaFinite (μ.trim ⋯))
      ⊢ Eq 0 fun x => 0
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
  -/
  have h_meas : StronglyMeasurable[⊥] (μ[f|⊥]) := stronglyMeasurable_condexp
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
  -/
  obtain ⟨c, h_eq⟩ := stronglyMeasurable_bot_iff.mp h_meas
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => HSMul.hSMul (Inv.inv (μ Set. …
  -/
  rw [h_eq]
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    ⊢ Eq (fun x => c) fun x => HSMul.hSMul (Inv.inv (μ Set.univ).toReal) (MeasureT …
  -/
  have h_integral : ∫ x, (μ[f|⊥]) x ∂μ = ∫ x, f x ∂μ := integral_condexp bot_le
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    h_integral : Eq (MeasureTheory.integral μ fun x => MeasureTheory.condexp Bot.b …
    ⊢ Eq (fun x => c) fun x => HSMul.hSMul (Inv.inv (μ Set.univ).toReal) (MeasureT …
  -/
  simp_rw [h_eq, integral_const] at h_integral
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    h_integral : Eq (HSMul.hSMul (μ Set.univ).toReal c) (MeasureTheory.integral μ  …
    ⊢ Eq (fun x => c) fun x => HSMul.hSMul (Inv.inv (μ Set.univ).toReal) (MeasureT …
  -/
  rw [← h_integral, ← smul_assoc, smul_eq_mul, inv_mul_cancel₀, one_smul]
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    h_integral : Eq (HSMul.hSMul (μ Set.univ).toReal c) (MeasureTheory.integral μ  …
    ⊢ Ne (μ Set.univ).toReal 0
  -/
  rw [Ne, ENNReal.toReal_eq_zero_iff, not_or]
  /-
    case pos.intro
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : NeZero μ
    f : α → F'
    hμ_finite : MeasureTheory.IsFiniteMeasure μ
    h_meas : MeasureTheory.StronglyMeasurable (MeasureTheory.condexp Bot.bot μ f)
    c : F'
    h_eq : Eq (MeasureTheory.condexp Bot.bot μ f) fun x => c
    h_integral : Eq (HSMul.hSMul (μ Set.univ).toReal c) (MeasureTheory.integral μ  …
    ⊢ And (Not (Eq (μ Set.univ) 0)) (Not (Eq (μ Set.univ) Top.top))
  -/
  exact ⟨NeZero.ne _, measure_ne_top μ Set.univ⟩
  /-
    🎉 no goals
  -/


theorem condexp_bot_ae_eq (f : α → F') :
    μ[f|⊥] =ᵐ[μ] fun _ => (μ Set.univ).toReal⁻¹ • ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp Bot.bot μ f) fun x  …
  -/
  rcases eq_zero_or_neZero μ with rfl | hμ
    /-
      case inl
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      f : α → F'
      ⊢ (MeasureTheory.ae 0).EventuallyEq (MeasureTheory.condexp Bot.bot 0 f) fun x  …
    -/
  · rw [ae_zero]; exact eventually_bot
                  /-
                    🎉 no goals
                  -/
    /-
      case inr
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → F'
      hμ : NeZero μ
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp Bot.bot μ f) fun x  …
    -/
  · exact Eventually.of_forall <| congr_fun (condexp_bot' f)
    /-
      🎉 no goals
    -/


theorem condexp_bot [IsProbabilityMeasure μ] (f : α → F') : μ[f|⊥] = fun _ => ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    f : α → F'
    ⊢ Eq (MeasureTheory.condexp Bot.bot μ f) fun x => MeasureTheory.integral μ fun …
  -/
  refine (condexp_bot' f).trans ?_; rw [measure_univ, ENNReal.one_toReal, inv_one, one_smul]
                                    /-
                                      🎉 no goals
                                    -/


theorem condexp_add (hf : Integrable f μ) (hg : Integrable g μ) :
    μ[f + g|m] =ᵐ[μ] μ[f|m] + μ[g|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  swap; · simp_rw [condexp_of_not_le hm]; simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; simp
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f g) …
  -/
  refine (condexp_ae_eq_condexpL1 hm _).trans ?_
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpL1 hm μ (HAdd.hAd …
  -/
  rw [condexpL1_add hf hg]
  exact (coeFn_add _ _).trans
    ((condexp_ae_eq_condexpL1 hm _).symm.add (condexp_ae_eq_condexpL1 hm _).symm)


theorem condexp_finset_sum {ι : Type*} {s : Finset ι} {f : ι → α → F'}
    (hf : ∀ i ∈ s, Integrable (f i) μ) : μ[∑ i ∈ s, f i|m] =ᵐ[μ] ∑ i ∈ s, μ[f i|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    s : Finset ι
    f : ι → α → F'
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.sum fun i => …
  -/
  induction' s using Finset.induction_on with i s his heq hf
    /-
      case empty
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      f : ι → α → F'
      hf : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → MeasureTheo …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (EmptyCollectio …
    -/
  · rw [Finset.sum_empty, Finset.sum_empty, condexp_zero]
    /-
      🎉 no goals
    -/
    /-
      case insert
      α : Type u_1
      F' : Type u_3
      inst✝² : NormedAddCommGroup F'
      inst✝¹ : NormedSpace Real F'
      inst✝ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      f : ι → α → F'
      i : ι
      s : Finset ι
      his : Not (Membership.mem s i)
      heq : (∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ) → (Me …
      hf : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → MeasureTheory.Integ …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ ((Insert.insert …
    -/
  · rw [Finset.sum_insert his, Finset.sum_insert his]
    exact (condexp_add (hf i <| Finset.mem_insert_self i s) <|
      integrable_finset_sum' _ fun j hmem => hf j <| Finset.mem_insert_of_mem hmem).trans
        ((EventuallyEq.refl _ _).add (heq fun j hmem => hf j <| Finset.mem_insert_of_mem hmem))


theorem condexp_smul (c : 𝕜) (f : α → F') : μ[c • f|m] =ᵐ[μ] c • μ[f|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  swap; · simp_rw [condexp_of_not_le hm]; simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; simp
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSMul.hSMul c  …
  -/
  refine (condexp_ae_eq_condexpL1 hm _).trans ?_
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpL1 hm μ (HSMul.hS …
  -/
  rw [condexpL1_smul c f]
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(HSMul.hSMul c (MeasureTheory.condexpL1 …
  -/
  refine (@condexp_ae_eq_condexpL1 _ _ _ _ _ m _ _ hm _ f).mp ?_
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Filter.Eventually (fun x => Eq (MeasureTheory.condexp m μ f x) (↑↑(MeasureTh …
  -/
  refine (coeFn_smul c (condexpL1 hm μ f)).mono fun x hx1 hx2 => ?_
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace 𝕜 F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : 𝕜
    f : α → F'
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    x : α
    hx1 : Eq (↑↑(HSMul.hSMul c (MeasureTheory.condexpL1 hm μ f)) x) (HSMul.hSMul c …
    hx2 : Eq (MeasureTheory.condexp m μ f x) (↑↑(MeasureTheory.condexpL1 hm μ f) x)
    ⊢ Eq (↑↑(HSMul.hSMul c (MeasureTheory.condexpL1 hm μ f)) x) (HSMul.hSMul c (Me …
  -/
  simp only [hx1, hx2, Pi.smul_apply]
  /-
    🎉 no goals
  -/


theorem condexp_neg (f : α → F') : μ[-f|m] =ᵐ[μ] -μ[f|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → F'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (Neg.neg f)) (N …
  -/
  letI : Module ℝ (α → F') := @Pi.module α (fun _ => F') ℝ _ _ fun _ => inferInstance
  calc
    μ[-f|m] = μ[(-1 : ℝ) • f|m] := by rw [neg_one_smul ℝ f]
    _ =ᵐ[μ] (-1 : ℝ) • μ[f|m] := condexp_smul (-1) f
    _ = -μ[f|m] := neg_one_smul ℝ (μ[f|m])


theorem condexp_sub (hf : Integrable f μ) (hg : Integrable g μ) :
    μ[f - g|m] =ᵐ[μ] μ[f|m] - μ[g|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HSub.hSub f g) …
  -/
  simp_rw [sub_eq_add_neg]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → F'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HAdd.hAdd f (N …
  -/
  exact (condexp_add hf hg.neg).trans (EventuallyEq.rfl.add (condexp_neg g))
  /-
    🎉 no goals
  -/


theorem condexp_condexp_of_le {m₁ m₂ m0 : MeasurableSpace α} {μ : Measure α} (hm₁₂ : m₁ ≤ m₂)
    (hm₂ : m₂ ≤ m0) [SigmaFinite (μ.trim hm₂)] : μ[μ[f|m₂]|m₁] =ᵐ[μ] μ[f|m₁] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₁ μ (MeasureTheory …
  -/
  by_cases hμm₁ : SigmaFinite (μ.trim (hm₁₂.trans hm₂))
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₁ μ (MeasureTheory …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite (hm₁₂.trans hm₂) hμm₁]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₁ μ (MeasureTheory …
  -/
  haveI : SigmaFinite (μ.trim (hm₁₂.trans hm₂)) := hμm₁
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₁ μ (MeasureTheory …
  -/
  by_cases hf : Integrable f μ
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hf : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₁ μ (MeasureTheory …
  -/
  swap; · simp_rw [condexp_undef hf, condexp_zero]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  refine ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' (hm₁₂.trans hm₂)
    (fun s _ _ => integrable_condexp.integrableOn)
    (fun s _ _ => integrable_condexp.integrableOn) ?_
    (StronglyMeasurable.aeStronglyMeasurable' stronglyMeasurable_condexp)
    (StronglyMeasurable.aeStronglyMeasurable' stronglyMeasurable_condexp)
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hf : MeasureTheory.Integrable f μ
    ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
  -/
  intro s hs _
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hf : MeasureTheory.Integrable f μ
    s : Set α
    hs : MeasurableSet s
    a✝ : LT.lt (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => MeasureTheory.condexp m₁  …
  -/
  rw [setIntegral_condexp (hm₁₂.trans hm₂) integrable_condexp hs]
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    f : α → F'
    m₁ m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm₂)
    hμm₁ this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hf : MeasureTheory.Integrable f μ
    s : Set α
    hs : MeasurableSet s
    a✝ : LT.lt (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => MeasureTheory.condexp m₂  …
  -/
  rw [setIntegral_condexp (hm₁₂.trans hm₂) hf hs, setIntegral_condexp hm₂ hf (hm₁₂ s hs)]
  /-
    🎉 no goals
  -/


theorem condexp_mono {E} [NormedLatticeAddCommGroup E] [CompleteSpace E] [NormedSpace ℝ E]
    [OrderedSMul ℝ E] {f g : α → E} (hf : Integrable f μ) (hg : Integrable g μ) (hfg : f ≤ᵐ[μ] g) :
    μ[f|m] ≤ᵐ[μ] μ[g|m] := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  swap; · simp_rw [condexp_of_not_le hm]; rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  exact (condexp_ae_eq_condexpL1 hm _).trans_le
    ((condexpL1_mono hf hg hfg).trans_eq (condexp_ae_eq_condexpL1 hm _).symm)


theorem condexp_nonneg {E} [NormedLatticeAddCommGroup E] [CompleteSpace E] [NormedSpace ℝ E]
    [OrderedSMul ℝ E] {f : α → E} (hf : 0 ≤ᵐ[μ] f) : 0 ≤ᵐ[μ] μ[f|m] := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f : α → E
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp m μ f)
  -/
  by_cases hfint : Integrable f μ
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp m μ f)
    -/
  · rw [(condexp_zero.symm : (0 : α → E) = μ[0|m])]
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ 0) (MeasureTheo …
    -/
    exact condexp_mono (integrable_zero _ _ _) hfint hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp m μ f)
    -/
  · rw [condexp_undef hfint]
    /-
      🎉 no goals
    -/


theorem condexp_nonpos {E} [NormedLatticeAddCommGroup E] [CompleteSpace E] [NormedSpace ℝ E]
    [OrderedSMul ℝ E] {f : α → E} (hf : f ≤ᵐ[μ] 0) : μ[f|m] ≤ᵐ[μ] 0 := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f : α → E
    hf : (MeasureTheory.ae μ).EventuallyLE f 0
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) 0
  -/
  by_cases hfint : Integrable f μ
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE f 0
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) 0
    -/
  · rw [(condexp_zero.symm : (0 : α → E) = μ[0|m])]
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE f 0
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) (MeasureTheo …
    -/
    exact condexp_mono hfint (integrable_zero _ _ _) hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝³ : NormedLatticeAddCommGroup E
      inst✝² : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      inst✝ : OrderedSMul Real E
      f : α → E
      hf : (MeasureTheory.ae μ).EventuallyLE f 0
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp m μ f) 0
    -/
  · rw [condexp_undef hfint]
    /-
      🎉 no goals
    -/


/-- **Lebesgue dominated convergence theorem**: sufficient conditions under which almost
  everywhere convergence of a sequence of functions implies the convergence of their image by
  `condexpL1`. -/
theorem tendsto_condexpL1_of_dominated_convergence (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    {fs : ℕ → α → F'} {f : α → F'} (bound_fs : α → ℝ)
    (hfs_meas : ∀ n, AEStronglyMeasurable (fs n) μ) (h_int_bound_fs : Integrable bound_fs μ)
    (hfs_bound : ∀ n, ∀ᵐ x ∂μ, ‖fs n x‖ ≤ bound_fs x)
    (hfs : ∀ᵐ x ∂μ, Tendsto (fun n => fs n x) atTop (𝓝 (f x))) :
    Tendsto (fun n => condexpL1 hm μ (fs n)) atTop (𝓝 (condexpL1 hm μ f)) :=
  tendsto_setToFun_of_dominated_convergence _ bound_fs hfs_meas h_int_bound_fs hfs_bound hfs


/-- If two sequences of functions have a.e. equal conditional expectations at each step, converge
and verify dominated convergence hypotheses, then the conditional expectations of their limits are
a.e. equal. -/
theorem tendsto_condexp_unique (fs gs : ℕ → α → F') (f g : α → F')
    (hfs_int : ∀ n, Integrable (fs n) μ) (hgs_int : ∀ n, Integrable (gs n) μ)
    (hfs : ∀ᵐ x ∂μ, Tendsto (fun n => fs n x) atTop (𝓝 (f x)))
    (hgs : ∀ᵐ x ∂μ, Tendsto (fun n => gs n x) atTop (𝓝 (g x))) (bound_fs : α → ℝ)
    (h_int_bound_fs : Integrable bound_fs μ) (bound_gs : α → ℝ)
    (h_int_bound_gs : Integrable bound_gs μ) (hfs_bound : ∀ n, ∀ᵐ x ∂μ, ‖fs n x‖ ≤ bound_fs x)
    (hgs_bound : ∀ n, ∀ᵐ x ∂μ, ‖gs n x‖ ≤ bound_gs x) (hfg : ∀ n, μ[fs n|m] =ᵐ[μ] μ[gs n|m]) :
    μ[f|m] =ᵐ[μ] μ[g|m] := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hm : m ≤ m0; swap; · simp_rw [condexp_of_not_le hm]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm); swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ f) (MeasureTheo …
  -/
  refine (condexp_ae_eq_condexpL1 hm f).trans ((condexp_ae_eq_condexpL1 hm g).trans ?_).symm
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpL1 hm μ g) ↑↑(Meas …
  -/
  rw [← Lp.ext_iff]
  have hn_eq : ∀ n, condexpL1 hm μ (gs n) = condexpL1 hm μ (fs n) := by
    intro n
    ext1
    refine (condexp_ae_eq_condexpL1 hm (gs n)).symm.trans ((hfg n).symm.trans ?_)
    exact condexp_ae_eq_condexpL1 hm (fs n)
  have hcond_fs : Tendsto (fun n => condexpL1 hm μ (fs n)) atTop (𝓝 (condexpL1 hm μ f)) :=
    tendsto_condexpL1_of_dominated_convergence hm _ (fun n => (hfs_int n).1) h_int_bound_fs
      hfs_bound hfs
  have hcond_gs : Tendsto (fun n => condexpL1 hm μ (gs n)) atTop (𝓝 (condexpL1 hm μ g)) :=
    tendsto_condexpL1_of_dominated_convergence hm _ (fun n => (hgs_int n).1) h_int_bound_gs
      hgs_bound hgs
  /-
    case pos
    α : Type u_1
    F' : Type u_3
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    fs gs : Nat → α → F'
    f g : α → F'
    hfs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    hgs_int : ∀ (n : Nat), MeasureTheory.Integrable (gs n) μ
    hfs : Filter.Eventually (fun x => Filter.Tendsto (fun n => fs n x) Filter.atTo …
    hgs : Filter.Eventually (fun x => Filter.Tendsto (fun n => gs n x) Filter.atTo …
    bound_fs : α → Real
    h_int_bound_fs : MeasureTheory.Integrable bound_fs μ
    bound_gs : α → Real
    h_int_bound_gs : MeasureTheory.Integrable bound_gs μ
    hfs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (fs n x) …
    hgs_bound : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm (gs n x) …
    hfg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m  …
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    hn_eq : ∀ (n : Nat), Eq (MeasureTheory.condexpL1 hm μ (gs n)) (MeasureTheory.c …
    hcond_fs : Filter.Tendsto (fun n => MeasureTheory.condexpL1 hm μ (fs n)) Filte …
    hcond_gs : Filter.Tendsto (fun n => MeasureTheory.condexpL1 hm μ (gs n)) Filte …
    ⊢ Eq (MeasureTheory.condexpL1 hm μ g) (MeasureTheory.condexpL1 hm μ f)
  -/
  exact tendsto_nhds_unique_of_eventuallyEq hcond_gs hcond_fs (Eventually.of_forall hn_eq)
  /-
    🎉 no goals
  -/


