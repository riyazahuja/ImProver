theorem lpMeas.ae_eq_zero_of_forall_setIntegral_eq_zero (hm : m ≤ m0) (f : lpMeas E' 𝕜 m p μ)
    (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    -- Porting note: needed to add explicit casts in the next two hypotheses
    (hf_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn (f : Lp E' p μ) s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, (f : Lp E' p μ) x ∂μ = 0) :
    f =ᵐ[μ] (0 : α → E') := by
  /-
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑↑f) 0
  -/
  obtain ⟨g, hg_sm, hfg⟩ := lpMeas.ae_fin_strongly_measurable' hm f hp_ne_zero hp_ne_top
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    g : α → E'
    hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑↑f) g
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑↑f) 0
  -/
  refine hfg.trans ?_
  -- Porting note: added
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    g : α → E'
    hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑↑f) g
    ⊢ (MeasureTheory.ae μ).EventuallyEq g 0
  -/
  unfold Filter.EventuallyEq at hfg
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    g : α → E'
    hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
    hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq g 0
  -/
  refine ae_eq_zero_of_forall_setIntegral_eq_of_finStronglyMeasurable_trim hm ?_ ?_ hg_sm
    /-
      case intro.intro.refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory.Integra …
    -/
  · intro s hs hμs
    /-
      case intro.intro.refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.IntegrableOn g s μ
    -/
    have hfg_restrict : f =ᵐ[μ.restrict s] g := ae_restrict_of_ae hfg
    /-
      case intro.intro.refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑↑f) g
      ⊢ MeasureTheory.IntegrableOn g s μ
    -/
    rw [IntegrableOn, integrable_congr hfg_restrict.symm]
    /-
      case intro.intro.refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑↑f) g
      ⊢ MeasureTheory.Integrable (↑↑↑f) (μ.restrict s)
    -/
    exact hf_int_finite s hs hμs
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    /-
      case intro.intro.refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => g x) 0
    -/
    have hfg_restrict : f =ᵐ[μ.restrict s] g := ae_restrict_of_ae hfg
    /-
      case intro.intro.refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑↑f) g
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => g x) 0
    -/
    rw [integral_congr_ae hfg_restrict.symm]
    /-
      case intro.intro.refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      g : α → E'
      hg_sm : MeasureTheory.FinStronglyMeasurable g (μ.trim hm)
      hfg : Filter.Eventually (fun x => Eq (↑↑↑f x) (g x)) (MeasureTheory.ae μ)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑↑f) g
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => ↑↑↑f a) 0
    -/
    exact hf_zero s hs hμs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias lpMeas.ae_eq_zero_of_forall_set_integral_eq_zero :=
  lpMeas.ae_eq_zero_of_forall_setIntegral_eq_zero


include 𝕜 in
theorem Lp.ae_eq_zero_of_forall_setIntegral_eq_zero' (hm : m ≤ m0) (f : Lp E' p μ)
    (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, f x ∂μ = 0)
    (hf_meas : AEStronglyMeasurable' m f μ) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑f) 0
  -/
  let f_meas : lpMeas E' 𝕜 m p μ := ⟨f, hf_meas⟩
  -- Porting note: `simp only` does not call `rfl` to try to close the goal. See https://github.com/leanprover-community/mathlib4/issues/5025
  /-
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑f) 0
  -/
  have hf_f_meas : f =ᵐ[μ] f_meas := by simp only [f_meas, Subtype.coe_mk]; rfl
  /-
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
    hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑f) 0
  -/
  refine hf_f_meas.trans ?_
  /-
    α : Type u_1
    E' : Type u_2
    𝕜 : Type u_4
    p : ENNReal
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
    hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑↑f_meas) 0
  -/
  refine lpMeas.ae_eq_zero_of_forall_setIntegral_eq_zero hm f_meas hp_ne_zero hp_ne_top ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory.Integra …
    -/
  · intro s hs hμs
    /-
      case refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.IntegrableOn (↑↑↑f_meas) s μ
    -/
    have hfg_restrict : f =ᵐ[μ.restrict s] f_meas := ae_restrict_of_ae hf_f_meas
    /-
      case refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ MeasureTheory.IntegrableOn (↑↑↑f_meas) s μ
    -/
    rw [IntegrableOn, integrable_congr hfg_restrict.symm]
    /-
      case refine_1
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ MeasureTheory.Integrable (↑↑f) (μ.restrict s)
    -/
    exact hf_int_finite s hs hμs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    /-
      case refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f_meas x) 0
    -/
    have hfg_restrict : f =ᵐ[μ.restrict s] f_meas := ae_restrict_of_ae hf_f_meas
    /-
      case refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f_meas x) 0
    -/
    rw [integral_congr_ae hfg_restrict.symm]
    /-
      case refine_2
      α : Type u_1
      E' : Type u_2
      𝕜 : Type u_4
      p : ENNReal
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' p μ) x
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f_meas : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E' 𝕜 m p μ) x : …
      hf_f_meas : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑↑f_meas
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfg_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑f ↑↑↑f_meas
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => ↑↑f a) 0
    -/
    exact hf_zero s hs hμs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias Lp.ae_eq_zero_of_forall_set_integral_eq_zero' :=
  Lp.ae_eq_zero_of_forall_setIntegral_eq_zero'


include 𝕜 in
/-- **Uniqueness of the conditional expectation** -/
theorem Lp.ae_eq_of_forall_setIntegral_eq' (hm : m ≤ m0) (f g : Lp E' p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) (hf_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn g s μ)
    (hfg : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ)
    (hf_meas : AEStronglyMeasurable' m f μ) (hg_meas : AEStronglyMeasurable' m g μ) :
    f =ᵐ[μ] g := by
  suffices h_sub : ⇑(f - g) =ᵐ[μ] 0 by
    rw [← sub_ae_eq_zero]; exact (Lp.coeFn_sub f g).symm.trans h_sub
  have hfg' : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → (∫ x in s, (f - g) x ∂μ) = 0 := by
    intro s hs hμs
    rw [integral_congr_ae (ae_restrict_of_ae (Lp.coeFn_sub f g))]
    rw [integral_sub' (hf_int_finite s hs hμs) (hg_int_finite s hs hμs)]
    exact sub_eq_zero.mpr (hfg s hs hμs)
  have hfg_int : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn (⇑(f - g)) s μ := by
    intro s hs hμs
    rw [IntegrableOn, integrable_congr (ae_restrict_of_ae (Lp.coeFn_sub f g))]
    exact (hf_int_finite s hs hμs).sub (hg_int_finite s hs hμs)
  have hfg_meas : AEStronglyMeasurable' m (⇑(f - g)) μ :=
    AEStronglyMeasurable'.congr (hf_meas.sub hg_meas) (Lp.coeFn_sub f g).symm
  exact
    Lp.ae_eq_zero_of_forall_setIntegral_eq_zero' 𝕜 hm (f - g) hp_ne_zero hp_ne_top hfg_int hfg'
      hfg_meas


@[deprecated (since := "2024-04-17")]
alias Lp.ae_eq_of_forall_set_integral_eq' := Lp.ae_eq_of_forall_setIntegral_eq'


theorem ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    {f g : α → F'} (hf_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn g s μ)
    (hfg_eq : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ)
    (hfm : AEStronglyMeasurable' m f μ) (hgm : AEStronglyMeasurable' m g μ) : f =ᵐ[μ] g := by
  /-
    α : Type u_1
    F' : Type u_3
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f g : α → F'
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  rw [← ae_eq_trim_iff_of_aeStronglyMeasurable' hm hfm hgm]
  have hf_mk_int_finite (s) :
      MeasurableSet[m] s → μ.trim hm s < ∞ → @IntegrableOn _ _ m _ _ (hfm.mk f) s (μ.trim hm) := by
    intro hs hμs
    rw [trim_measurableSet_eq hm hs] at hμs
    -- Porting note: `rw [IntegrableOn]` fails with
    -- synthesized type class instance is not definitionally equal to expression inferred by typing
    -- rules, synthesized m0 inferred m
    unfold IntegrableOn
    rw [restrict_trim hm _ hs]
    refine Integrable.trim hm ?_ hfm.stronglyMeasurable_mk
    exact Integrable.congr (hf_int_finite s hs hμs) (ae_restrict_of_ae hfm.ae_eq_mk)
  have hg_mk_int_finite (s) :
      MeasurableSet[m] s → μ.trim hm s < ∞ → @IntegrableOn _ _ m _ _ (hgm.mk g) s (μ.trim hm) := by
    intro hs hμs
    rw [trim_measurableSet_eq hm hs] at hμs
    -- Porting note: `rw [IntegrableOn]` fails with
    -- synthesized type class instance is not definitionally equal to expression inferred by typing
    -- rules, synthesized m0 inferred m
    unfold IntegrableOn
    rw [restrict_trim hm _ hs]
    refine Integrable.trim hm ?_ hgm.stronglyMeasurable_mk
    exact Integrable.congr (hg_int_finite s hs hμs) (ae_restrict_of_ae hgm.ae_eq_mk)
  have hfg_mk_eq :
    ∀ s : Set α,
      MeasurableSet[m] s →
        μ.trim hm s < ∞ → ∫ x in s, hfm.mk f x ∂μ.trim hm = ∫ x in s, hgm.mk g x ∂μ.trim hm := by
    intro s hs hμs
    rw [trim_measurableSet_eq hm hs] at hμs
    rw [restrict_trim hm _ hs, ← integral_trim hm hfm.stronglyMeasurable_mk, ←
      integral_trim hm hgm.stronglyMeasurable_mk,
      integral_congr_ae (ae_restrict_of_ae hfm.ae_eq_mk.symm),
      integral_congr_ae (ae_restrict_of_ae hgm.ae_eq_mk.symm)]
    exact hfg_eq s hs hμs
  /-
    α : Type u_1
    F' : Type u_3
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f g : α → F'
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    hf_mk_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt ((μ.trim hm) s) Top. …
    hg_mk_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt ((μ.trim hm) s) Top. …
    hfg_mk_eq : ∀ (s : Set α), MeasurableSet s → LT.lt ((μ.trim hm) s) Top.top → E …
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq (MeasureTheory.AEStronglyMeasura …
  -/
  exact ae_eq_of_forall_setIntegral_eq_of_sigmaFinite hf_mk_int_finite hg_mk_int_finite hfg_mk_eq
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_of_forall_set_integral_eq_of_sigmaFinite' :=
  ae_eq_of_forall_setIntegral_eq_of_sigmaFinite'


/-- Let `m` be a sub-σ-algebra of `m0`, `f` an `m0`-measurable function and `g` an `m`-measurable
function, such that their integrals coincide on `m`-measurable sets with finite measure.
Then `∫ x in s, ‖g x‖ ∂μ ≤ ∫ x in s, ‖f x‖ ∂μ` on all `m`-measurable sets with finite measure. -/
theorem integral_norm_le_of_forall_fin_meas_integral_eq (hm : m ≤ m0) {f g : α → ℝ}
    (hf : StronglyMeasurable f) (hfi : IntegrableOn f s μ) (hg : StronglyMeasurable[m] g)
    (hgi : IntegrableOn g s μ)
    (hgf : ∀ t, MeasurableSet[m] t → μ t < ∞ → ∫ x in t, g x ∂μ = ∫ x in t, f x ∂μ)
    (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) : (∫ x in s, ‖g x‖ ∂μ) ≤ ∫ x in s, ‖f x‖ ∂μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfi : MeasureTheory.IntegrableOn f s μ
    hg : MeasureTheory.StronglyMeasurable g
    hgi : MeasureTheory.IntegrableOn g s μ
    hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => Norm.norm (g x)) (Meas …
  -/
  rw [integral_norm_eq_pos_sub_neg hgi, integral_norm_eq_pos_sub_neg hfi]
  have h_meas_nonneg_g : MeasurableSet[m] {x | 0 ≤ g x} :=
    (@stronglyMeasurable_const _ _ m _ _).measurableSet_le hg
  have h_meas_nonneg_f : MeasurableSet {x | 0 ≤ f x} :=
    stronglyMeasurable_const.measurableSet_le hf
  have h_meas_nonpos_g : MeasurableSet[m] {x | g x ≤ 0} :=
    hg.measurableSet_le (@stronglyMeasurable_const _ _ m _ _)
  have h_meas_nonpos_f : MeasurableSet {x | f x ≤ 0} :=
    hf.measurableSet_le stronglyMeasurable_const
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfi : MeasureTheory.IntegrableOn f s μ
    hg : MeasureTheory.StronglyMeasurable g
    hgi : MeasureTheory.IntegrableOn g s μ
    hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h_meas_nonneg_g : MeasurableSet (setOf fun x => LE.le 0 (g x))
    h_meas_nonneg_f : MeasurableSet (setOf fun x => LE.le 0 (f x))
    h_meas_nonpos_g : MeasurableSet (setOf fun x => LE.le (g x) 0)
    h_meas_nonpos_f : MeasurableSet (setOf fun x => LE.le (f x) 0)
    ⊢ LE.le (HSub.hSub (MeasureTheory.integral ((μ.restrict s).restrict (setOf fun …
  -/
  refine sub_le_sub ?_ ?_
  · rw [Measure.restrict_restrict (hm _ h_meas_nonneg_g), Measure.restrict_restrict h_meas_nonneg_f,
      hgf _ (@MeasurableSet.inter α m _ _ h_meas_nonneg_g hs)
        ((measure_mono Set.inter_subset_right).trans_lt (lt_top_iff_ne_top.mpr hμs)),
      ← Measure.restrict_restrict (hm _ h_meas_nonneg_g), ←
      Measure.restrict_restrict h_meas_nonneg_f]
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfi : MeasureTheory.IntegrableOn f s μ
      hg : MeasureTheory.StronglyMeasurable g
      hgi : MeasureTheory.IntegrableOn g s μ
      hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      h_meas_nonneg_g : MeasurableSet (setOf fun x => LE.le 0 (g x))
      h_meas_nonneg_f : MeasurableSet (setOf fun x => LE.le 0 (f x))
      h_meas_nonpos_g : MeasurableSet (setOf fun x => LE.le (g x) 0)
      h_meas_nonpos_f : MeasurableSet (setOf fun x => LE.le (f x) 0)
      ⊢ LE.le (MeasureTheory.integral ((μ.restrict s).restrict (setOf fun x => LE.le …
    -/
    exact setIntegral_le_nonneg (hm _ h_meas_nonneg_g) hf hfi
    /-
      🎉 no goals
    -/
  · rw [Measure.restrict_restrict (hm _ h_meas_nonpos_g), Measure.restrict_restrict h_meas_nonpos_f,
      hgf _ (@MeasurableSet.inter α m _ _ h_meas_nonpos_g hs)
        ((measure_mono Set.inter_subset_right).trans_lt (lt_top_iff_ne_top.mpr hμs)),
      ← Measure.restrict_restrict (hm _ h_meas_nonpos_g), ←
      Measure.restrict_restrict h_meas_nonpos_f]
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfi : MeasureTheory.IntegrableOn f s μ
      hg : MeasureTheory.StronglyMeasurable g
      hgi : MeasureTheory.IntegrableOn g s μ
      hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      h_meas_nonneg_g : MeasurableSet (setOf fun x => LE.le 0 (g x))
      h_meas_nonneg_f : MeasurableSet (setOf fun x => LE.le 0 (f x))
      h_meas_nonpos_g : MeasurableSet (setOf fun x => LE.le (g x) 0)
      h_meas_nonpos_f : MeasurableSet (setOf fun x => LE.le (f x) 0)
      ⊢ LE.le (MeasureTheory.integral ((μ.restrict s).restrict (setOf fun x => LE.le …
    -/
    exact setIntegral_nonpos_le (hm _ h_meas_nonpos_g) hf hfi
    /-
      🎉 no goals
    -/


/-- Let `m` be a sub-σ-algebra of `m0`, `f` an `m0`-measurable function and `g` an `m`-measurable
function, such that their integrals coincide on `m`-measurable sets with finite measure.
Then `∫⁻ x in s, ‖g x‖₊ ∂μ ≤ ∫⁻ x in s, ‖f x‖₊ ∂μ` on all `m`-measurable sets with finite
measure. -/
theorem lintegral_nnnorm_le_of_forall_fin_meas_integral_eq (hm : m ≤ m0) {f g : α → ℝ}
    (hf : StronglyMeasurable f) (hfi : IntegrableOn f s μ) (hg : StronglyMeasurable[m] g)
    (hgi : IntegrableOn g s μ)
    (hgf : ∀ t, MeasurableSet[m] t → μ t < ∞ → ∫ x in t, g x ∂μ = ∫ x in t, f x ∂μ)
    (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) : (∫⁻ x in s, ‖g x‖₊ ∂μ) ≤ ∫⁻ x in s, ‖f x‖₊ ∂μ := by
  rw [← ofReal_integral_norm_eq_lintegral_nnnorm hfi, ←
    ofReal_integral_norm_eq_lintegral_nnnorm hgi, ENNReal.ofReal_le_ofReal_iff]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfi : MeasureTheory.IntegrableOn f s μ
      hg : MeasureTheory.StronglyMeasurable g
      hgi : MeasureTheory.IntegrableOn g s μ
      hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => Norm.norm (g x)) (Meas …
    -/
  · exact integral_norm_le_of_forall_fin_meas_integral_eq hm hf hfi hg hgi hgf hs hμs
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfi : MeasureTheory.IntegrableOn f s μ
      hg : MeasureTheory.StronglyMeasurable g
      hgi : MeasureTheory.IntegrableOn g s μ
      hgf : ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory …
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x => Norm.norm (f x))
    -/
  · positivity
    /-
      🎉 no goals
    -/


