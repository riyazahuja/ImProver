/-- A family of functions `f : ι → Ω → E` is a martingale with respect to a filtration `ℱ` if `f`
is adapted with respect to `ℱ` and for all `i ≤ j`, `μ[f j | ℱ i] =ᵐ[μ] f i`. -/
def Martingale (f : ι → Ω → E) (ℱ : Filtration ι m0) (μ : Measure Ω) : Prop :=
  Adapted ℱ f ∧ ∀ i j, i ≤ j → μ[f j|ℱ i] =ᵐ[μ] f i


/-- A family of integrable functions `f : ι → Ω → E` is a supermartingale with respect to a
filtration `ℱ` if `f` is adapted with respect to `ℱ` and for all `i ≤ j`,
`μ[f j | ℱ.le i] ≤ᵐ[μ] f i`. -/
def Supermartingale [LE E] (f : ι → Ω → E) (ℱ : Filtration ι m0) (μ : Measure Ω) : Prop :=
  Adapted ℱ f ∧ (∀ i j, i ≤ j → μ[f j|ℱ i] ≤ᵐ[μ] f i) ∧ ∀ i, Integrable (f i) μ


/-- A family of integrable functions `f : ι → Ω → E` is a submartingale with respect to a
filtration `ℱ` if `f` is adapted with respect to `ℱ` and for all `i ≤ j`,
`f i ≤ᵐ[μ] μ[f j | ℱ.le i]`. -/
def Submartingale [LE E] (f : ι → Ω → E) (ℱ : Filtration ι m0) (μ : Measure Ω) : Prop :=
  Adapted ℱ f ∧ (∀ i j, i ≤ j → f i ≤ᵐ[μ] μ[f j|ℱ i]) ∧ ∀ i, Integrable (f i) μ


theorem martingale_const (ℱ : Filtration ι m0) (μ : Measure Ω) [IsFiniteMeasure μ] (x : E) :
    Martingale (fun _ _ => x) ℱ μ :=
                                      /-
                                        Ω : Type u_1
                                        E : Type u_2
                                        ι : Type u_3
                                        inst✝⁴ : Preorder ι
                                        m0 : MeasurableSpace Ω
                                        inst✝³ : NormedAddCommGroup E
                                        inst✝² : NormedSpace Real E
                                        inst✝¹ : CompleteSpace E
                                        ℱ : MeasureTheory.Filtration ι m0
                                        μ : MeasureTheory.Measure Ω
                                        inst✝ : MeasureTheory.IsFiniteMeasure μ
                                        x : E
                                        i j : ι
                                        x✝ : LE.le i j
                                        ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ((fun x_1  …
                                      -/
  ⟨adapted_const ℱ _, fun i j _ => by rw [condexp_const (ℱ.le _)]⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem martingale_const_fun [OrderBot ι] (ℱ : Filtration ι m0) (μ : Measure Ω) [IsFiniteMeasure μ]
    {f : Ω → E} (hf : StronglyMeasurable[ℱ ⊥] f) (hfint : Integrable f μ) :
    Martingale (fun _ => f) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : OrderBot ι
    ℱ : MeasureTheory.Filtration ι m0
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Ω → E
    hf : MeasureTheory.StronglyMeasurable f
    hfint : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Martingale (fun x => f) ℱ μ
  -/
  refine ⟨fun i => hf.mono <| ℱ.mono bot_le, fun i j _ => ?_⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : OrderBot ι
    ℱ : MeasureTheory.Filtration ι m0
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Ω → E
    hf : MeasureTheory.StronglyMeasurable f
    hfint : MeasureTheory.Integrable f μ
    i j : ι
    x✝ : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ((fun x => …
  -/
  rw [condexp_of_stronglyMeasurable (ℱ.le _) (hf.mono <| ℱ.mono bot_le) hfint]
  /-
    🎉 no goals
  -/


theorem martingale_zero (ℱ : Filtration ι m0) (μ : Measure Ω) : Martingale (0 : ι → Ω → E) ℱ μ :=
                                     /-
                                       Ω : Type u_1
                                       E : Type u_2
                                       ι : Type u_3
                                       inst✝³ : Preorder ι
                                       m0 : MeasurableSpace Ω
                                       inst✝² : NormedAddCommGroup E
                                       inst✝¹ : NormedSpace Real E
                                       inst✝ : CompleteSpace E
                                       ℱ : MeasureTheory.Filtration ι m0
                                       μ : MeasureTheory.Measure Ω
                                       i j : ι
                                       x✝ : LE.le i j
                                       ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ (0 j)) (0 i)
                                     -/
  ⟨adapted_zero E ℱ, fun i j _ => by rw [Pi.zero_apply, condexp_zero]; simp⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


protected theorem adapted (hf : Martingale f ℱ μ) : Adapted ℱ f :=
  hf.1


protected theorem stronglyMeasurable (hf : Martingale f ℱ μ) (i : ι) :
    StronglyMeasurable[ℱ i] (f i) :=
  hf.adapted i


theorem condexp_ae_eq (hf : Martingale f ℱ μ) {i j : ι} (hij : i ≤ j) : μ[f j|ℱ i] =ᵐ[μ] f i :=
  hf.2 i j hij


protected theorem integrable (hf : Martingale f ℱ μ) (i : ι) : Integrable (f i) μ :=
  integrable_condexp.congr (hf.condexp_ae_eq (le_refl i))


theorem setIntegral_eq [SigmaFiniteFiltration μ ℱ] (hf : Martingale f ℱ μ) {i j : ι} (hij : i ≤ j)
    {s : Set Ω} (hs : MeasurableSet[ℱ i] s) : ∫ ω in s, f i ω ∂μ = ∫ ω in s, f j ω ∂μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    hf : MeasureTheory.Martingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory.int …
  -/
  rw [← @setIntegral_condexp _ _ _ _ _ (ℱ i) m0 _ _ _ (ℱ.le i) _ (hf.integrable j) hs]
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    hf : MeasureTheory.Martingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory.int …
  -/
  refine setIntegral_congr_ae (ℱ.le i s hs) ?_
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    hf : MeasureTheory.Martingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → Eq (f i x) (MeasureTheory.c …
  -/
  filter_upwards [hf.2 i j hij] with _ heq _ using heq.symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_eq := setIntegral_eq


theorem add (hf : Martingale f ℱ μ) (hg : Martingale g ℱ μ) : Martingale (f + g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Martingale g ℱ μ
    ⊢ MeasureTheory.Martingale (HAdd.hAdd f g) ℱ μ
  -/
  refine ⟨hf.adapted.add hg.adapted, fun i j hij => ?_⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Martingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ (HAdd.hAdd …
  -/
  exact (condexp_add (hf.integrable j) (hg.integrable j)).trans ((hf.2 i j hij).add (hg.2 i j hij))
  /-
    🎉 no goals
  -/


theorem neg (hf : Martingale f ℱ μ) : Martingale (-f) ℱ μ :=
  ⟨hf.adapted.neg, fun i j hij => (condexp_neg (f j)).trans (hf.2 i j hij).neg⟩


theorem sub (hf : Martingale f ℱ μ) (hg : Martingale g ℱ μ) : Martingale (f - g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Martingale g ℱ μ
    ⊢ MeasureTheory.Martingale (HSub.hSub f g) ℱ μ
  -/
  rw [sub_eq_add_neg]; exact hf.add hg.neg
                       /-
                         🎉 no goals
                       -/


theorem smul (c : ℝ) (hf : Martingale f ℱ μ) : Martingale (c • f) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    c : Real
    hf : MeasureTheory.Martingale f ℱ μ
    ⊢ MeasureTheory.Martingale (HSMul.hSMul c f) ℱ μ
  -/
  refine ⟨hf.adapted.smul c, fun i j hij => ?_⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    c : Real
    hf : MeasureTheory.Martingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ (HSMul.hSM …
  -/
  refine (condexp_smul c (f j)).trans ((hf.2 i j hij).mono fun x hx => ?_)
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    c : Real
    hf : MeasureTheory.Martingale f ℱ μ
    i j : ι
    hij : LE.le i j
    x : Ω
    hx : Eq (MeasureTheory.condexp (↑ℱ i) μ (f j) x) (f i x)
    ⊢ Eq (HSMul.hSMul c (MeasureTheory.condexp (↑ℱ i) μ (f j)) x) (HSMul.hSMul c f …
  -/
  simp only [Pi.smul_apply, hx]
  /-
    🎉 no goals
  -/


theorem supermartingale [Preorder E] (hf : Martingale f ℱ μ) : Supermartingale f ℱ μ :=
  ⟨hf.1, fun i j hij => (hf.2 i j hij).le, fun i => hf.integrable i⟩


theorem submartingale [Preorder E] (hf : Martingale f ℱ μ) : Submartingale f ℱ μ :=
  ⟨hf.1, fun i j hij => (hf.2 i j hij).symm.le, fun i => hf.integrable i⟩


theorem martingale_iff [PartialOrder E] :
    Martingale f ℱ μ ↔ Supermartingale f ℱ μ ∧ Submartingale f ℱ μ :=
  ⟨fun hf => ⟨hf.supermartingale, hf.submartingale⟩, fun ⟨hf₁, hf₂⟩ =>
    ⟨hf₁.1, fun i j hij => (hf₁.2.1 i j hij).antisymm (hf₂.2.1 i j hij)⟩⟩


theorem martingale_condexp (f : Ω → E) (ℱ : Filtration ι m0) (μ : Measure Ω)
    [SigmaFiniteFiltration μ ℱ] : Martingale (fun i => μ[f|ℱ i]) ℱ μ :=
  ⟨fun _ => stronglyMeasurable_condexp, fun _ j hij => condexp_condexp_of_le (ℱ.mono hij) (ℱ.le j)⟩


protected theorem adapted [LE E] (hf : Supermartingale f ℱ μ) : Adapted ℱ f :=
  hf.1


protected theorem stronglyMeasurable [LE E] (hf : Supermartingale f ℱ μ) (i : ι) :
    StronglyMeasurable[ℱ i] (f i) :=
  hf.adapted i


protected theorem integrable [LE E] (hf : Supermartingale f ℱ μ) (i : ι) : Integrable (f i) μ :=
  hf.2.2 i


theorem condexp_ae_le [LE E] (hf : Supermartingale f ℱ μ) {i j : ι} (hij : i ≤ j) :
    μ[f j|ℱ i] ≤ᵐ[μ] f i :=
  hf.2.1 i j hij


theorem setIntegral_le [SigmaFiniteFiltration μ ℱ] {f : ι → Ω → ℝ} (hf : Supermartingale f ℱ μ)
    {i j : ι} (hij : i ≤ j) {s : Set Ω} (hs : MeasurableSet[ℱ i] s) :
    ∫ ω in s, f j ω ∂μ ≤ ∫ ω in s, f i ω ∂μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f : ι → Ω → Real
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f j ω) (MeasureTheory. …
  -/
  rw [← setIntegral_condexp (ℱ.le i) (hf.integrable j) hs]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f : ι → Ω → Real
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => MeasureTheory.condexp  …
  -/
  refine setIntegral_mono_ae integrable_condexp.integrableOn (hf.integrable i).integrableOn ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f : ι → Ω → Real
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp (↑ℱ i) μ (f j)) (f i)
  -/
  filter_upwards [hf.2.1 i j hij] with _ heq using heq
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_le := setIntegral_le


theorem add [Preorder E] [AddLeftMono E] (hf : Supermartingale f ℱ μ)
    (hg : Supermartingale g ℱ μ) : Supermartingale (f + g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    ⊢ MeasureTheory.Supermartingale (HAdd.hAdd f g) ℱ μ
  -/
  refine ⟨hf.1.add hg.1, fun i j hij => ?_, fun i => (hf.2.2 i).add (hg.2.2 i)⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp (↑ℱ i) μ (HAdd.hAdd …
  -/
  refine (condexp_add (hf.integrable j) (hg.integrable j)).le.trans ?_
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (HAdd.hAdd (MeasureTheory.condexp (↑ℱ i) μ …
  -/
  filter_upwards [hf.2.1 i j hij, hg.2.1 i j hij]
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ ∀ (a : Ω), LE.le (MeasureTheory.condexp (↑ℱ i) μ (f j) a) (f i a) → LE.le (M …
  -/
  intros
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    a✝² : Ω
    a✝¹ : LE.le (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝²) (f i a✝²)
    a✝ : LE.le (MeasureTheory.condexp (↑ℱ i) μ (g j) a✝²) (g i a✝²)
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.condexp (↑ℱ i) μ (f j)) (MeasureTheory.conde …
  -/
                              /-
                                🎉 no goals
                              -/
  refine add_le_add ?_ ?_ <;> assumption
                              /-
                                🎉 no goals
                              -/


theorem add_martingale [Preorder E] [AddLeftMono E]
    (hf : Supermartingale f ℱ μ) (hg : Martingale g ℱ μ) : Supermartingale (f + g) ℱ μ :=
  hf.add hg.supermartingale


theorem neg [Preorder E] [AddLeftMono E] (hf : Supermartingale f ℱ μ) :
    Submartingale (-f) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    ⊢ MeasureTheory.Submartingale (Neg.neg f) ℱ μ
  -/
  refine ⟨hf.1.neg, fun i j hij => ?_, fun i => (hf.2.2 i).neg⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Neg.neg f i) (MeasureTheory.condexp (↑ℱ i …
  -/
  refine EventuallyLE.trans ?_ (condexp_neg (f j)).symm.le
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Neg.neg f i) (Neg.neg (MeasureTheory.cond …
  -/
  filter_upwards [hf.2.1 i j hij] with _ _
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    a✝¹ : Ω
    a✝ : LE.le (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝¹) (f i a✝¹)
    ⊢ LE.le (Neg.neg f i a✝¹) (Neg.neg (MeasureTheory.condexp (↑ℱ i) μ (f j)) a✝¹)
  -/
  simpa
  /-
    🎉 no goals
  -/


protected theorem adapted [LE E] (hf : Submartingale f ℱ μ) : Adapted ℱ f :=
  hf.1


protected theorem stronglyMeasurable [LE E] (hf : Submartingale f ℱ μ) (i : ι) :
    StronglyMeasurable[ℱ i] (f i) :=
  hf.adapted i


protected theorem integrable [LE E] (hf : Submartingale f ℱ μ) (i : ι) : Integrable (f i) μ :=
  hf.2.2 i


theorem ae_le_condexp [LE E] (hf : Submartingale f ℱ μ) {i j : ι} (hij : i ≤ j) :
    f i ≤ᵐ[μ] μ[f j|ℱ i] :=
  hf.2.1 i j hij


theorem add [Preorder E] [AddLeftMono E] (hf : Submartingale f ℱ μ)
    (hg : Submartingale g ℱ μ) : Submartingale (f + g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    ⊢ MeasureTheory.Submartingale (HAdd.hAdd f g) ℱ μ
  -/
  refine ⟨hf.1.add hg.1, fun i j hij => ?_, fun i => (hf.2.2 i).add (hg.2.2 i)⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (HAdd.hAdd f g i) (MeasureTheory.condexp ( …
  -/
  refine EventuallyLE.trans ?_ (condexp_add (hf.integrable j) (hg.integrable j)).symm.le
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (HAdd.hAdd f g i) (HAdd.hAdd (MeasureTheor …
  -/
  filter_upwards [hf.2.1 i j hij, hg.2.1 i j hij]
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ ∀ (a : Ω), LE.le (f i a) (MeasureTheory.condexp (↑ℱ i) μ (f j) a) → LE.le (g …
  -/
  intros
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    a✝² : Ω
    a✝¹ : LE.le (f i a✝²) (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝²)
    a✝ : LE.le (g i a✝²) (MeasureTheory.condexp (↑ℱ i) μ (g j) a✝²)
    ⊢ LE.le (HAdd.hAdd f g i a✝²) (HAdd.hAdd (MeasureTheory.condexp (↑ℱ i) μ (f j) …
  -/
                              /-
                                🎉 no goals
                              -/
  refine add_le_add ?_ ?_ <;> assumption
                              /-
                                🎉 no goals
                              -/


theorem add_martingale [Preorder E] [AddLeftMono E] (hf : Submartingale f ℱ μ)
    (hg : Martingale g ℱ μ) : Submartingale (f + g) ℱ μ :=
  hf.add hg.submartingale


theorem neg [Preorder E] [AddLeftMono E] (hf : Submartingale f ℱ μ) :
    Supermartingale (-f) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ MeasureTheory.Supermartingale (Neg.neg f) ℱ μ
  -/
  refine ⟨hf.1.neg, fun i j hij => (condexp_neg (f j)).le.trans ?_, fun i => (hf.2.2 i).neg⟩
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Neg.neg (MeasureTheory.condexp (↑ℱ i) μ ( …
  -/
  filter_upwards [hf.2.1 i j hij] with _ _
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    a✝¹ : Ω
    a✝ : LE.le (f i a✝¹) (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝¹)
    ⊢ LE.le (Neg.neg (MeasureTheory.condexp (↑ℱ i) μ (f j)) a✝¹) (Neg.neg f i a✝¹)
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- The converse of this lemma is `MeasureTheory.submartingale_of_setIntegral_le`. -/
theorem setIntegral_le [SigmaFiniteFiltration μ ℱ] {f : ι → Ω → ℝ} (hf : Submartingale f ℱ μ)
    {i j : ι} (hij : i ≤ j) {s : Set Ω} (hs : MeasurableSet[ℱ i] s) :
    ∫ ω in s, f i ω ∂μ ≤ ∫ ω in s, f j ω ∂μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
  -/
  rw [← neg_le_neg_iff, ← integral_neg, ← integral_neg]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun a => Neg.neg (f j a)) (Meas …
  -/
  exact Supermartingale.setIntegral_le hf.neg hij hs
  /-
    🎉 no goals
  -/


theorem sub_supermartingale [Preorder E] [AddLeftMono E]
    (hf : Submartingale f ℱ μ) (hg : Supermartingale g ℱ μ) : Submartingale (f - g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Supermartingale g ℱ μ
    ⊢ MeasureTheory.Submartingale (HSub.hSub f g) ℱ μ
  -/
  rw [sub_eq_add_neg]; exact hf.add hg.neg
                       /-
                         🎉 no goals
                       -/


theorem sub_martingale [Preorder E] [AddLeftMono E] (hf : Submartingale f ℱ μ)
    (hg : Martingale g ℱ μ) : Submartingale (f - g) ℱ μ :=
  hf.sub_supermartingale hg.supermartingale


protected theorem sup {f g : ι → Ω → ℝ} (hf : Submartingale f ℱ μ) (hg : Submartingale g ℱ μ) :
    Submartingale (f ⊔ g) ℱ μ := by
  refine ⟨fun i => @StronglyMeasurable.sup _ _ _ _ (ℱ i) _ _ _ (hf.adapted i) (hg.adapted i),
    fun i j hij => ?_, fun i => Integrable.sup (hf.integrable _) (hg.integrable _)⟩
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    f g : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Max.max f g i) (MeasureTheory.condexp (↑ℱ …
  -/
  refine EventuallyLE.sup_le ?_ ?_
  · exact EventuallyLE.trans (hf.2.1 i j hij)
      (condexp_mono (hf.integrable _) (Integrable.sup (hf.integrable j) (hg.integrable j))
        (Eventually.of_forall fun x => le_max_left _ _))
  · exact EventuallyLE.trans (hg.2.1 i j hij)
      (condexp_mono (hg.integrable _) (Integrable.sup (hf.integrable j) (hg.integrable j))
        (Eventually.of_forall fun x => le_max_right _ _))


protected theorem pos {f : ι → Ω → ℝ} (hf : Submartingale f ℱ μ) : Submartingale (f⁺) ℱ μ :=
  hf.sup (martingale_zero _ _ _).submartingale


theorem submartingale_of_setIntegral_le [IsFiniteMeasure μ] {f : ι → Ω → ℝ} (hadp : Adapted ℱ f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ i j : ι,
      i ≤ j → ∀ s : Set Ω, MeasurableSet[ℱ i] s → ∫ ω in s, f i ω ∂μ ≤ ∫ ω in s, f j ω ∂μ) :
    Submartingale f ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → ∀ (s : Set Ω), MeasurableSet s → LE.le (MeasureT …
    ⊢ MeasureTheory.Submartingale f ℱ μ
  -/
  refine ⟨hadp, fun i j hij => ?_, hint⟩
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → ∀ (s : Set Ω), MeasurableSet s → LE.le (MeasureT …
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.condexp (↑ℱ i) μ (f j))
  -/
  suffices f i ≤ᵐ[μ.trim (ℱ.le i)] μ[f j|ℱ i] by exact ae_le_of_ae_le_trim this
  suffices 0 ≤ᵐ[μ.trim (ℱ.le i)] μ[f j|ℱ i] - f i by
    filter_upwards [this] with x hx
    rwa [← sub_nonneg]
  refine ae_nonneg_of_forall_setIntegral_nonneg
    ((integrable_condexp.sub (hint i)).trim _ (stronglyMeasurable_condexp.sub <| hadp i))
      fun s hs _ => ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → ∀ (s : Set Ω), MeasurableSet s → LE.le (MeasureT …
    i j : ι
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    x✝ : LT.lt ((μ.trim ⋯) s) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral ((μ.trim ⋯).restrict s) fun x => HSub.hSub ( …
  -/
  specialize hf i j hij s hs
  rwa [← setIntegral_trim _ (stronglyMeasurable_condexp.sub <| hadp i) hs,
    integral_sub' integrable_condexp.integrableOn (hint i).integrableOn, sub_nonneg,
    setIntegral_condexp (ℱ.le i) (hint j) hs]


@[deprecated (since := "2024-04-17")]
alias submartingale_of_set_integral_le := submartingale_of_setIntegral_le


theorem submartingale_of_condexp_sub_nonneg [IsFiniteMeasure μ] {f : ι → Ω → ℝ} (hadp : Adapted ℱ f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ i j, i ≤ j → 0 ≤ᵐ[μ] μ[f j - f i|ℱ i]) :
    Submartingale f ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheo …
    ⊢ MeasureTheory.Submartingale f ℱ μ
  -/
  refine ⟨hadp, fun i j hij => ?_, hint⟩
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheo …
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.condexp (↑ℱ i) μ (f j))
  -/
  rw [← condexp_of_stronglyMeasurable (ℱ.le _) (hadp _) (hint _), ← eventually_sub_nonneg]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝¹ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : ι → Ω → Real
    hadp : MeasureTheory.Adapted ℱ f
    hint : ∀ (i : ι), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i j : ι), LE.le i j → (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheo …
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (HSub.hSub (MeasureTheory.condexp (↑ℱ i) …
  -/
  exact EventuallyLE.trans (hf i j hij) (condexp_sub (hint _) (hint _)).le
  /-
    🎉 no goals
  -/


theorem Submartingale.condexp_sub_nonneg {f : ι → Ω → ℝ} (hf : Submartingale f ℱ μ) {i j : ι}
    (hij : i ≤ j) : 0 ≤ᵐ[μ] μ[f j - f i|ℱ i] := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑ℱ i) μ (HSub.hS …
  -/
  by_cases h : SigmaFinite (μ.trim (ℱ.le i))
  /-
    case pos
    Ω : Type u_1
    ι : Type u_3
    inst✝ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    h : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑ℱ i) μ (HSub.hS …
  -/
  swap; · rw [condexp_of_not_sigmaFinite (ℱ.le i) h]
          /-
            🎉 no goals
          -/
  /-
    case pos
    Ω : Type u_1
    ι : Type u_3
    inst✝ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    h : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑ℱ i) μ (HSub.hS …
  -/
  refine EventuallyLE.trans ?_ (condexp_sub (hf.integrable _) (hf.integrable _)).symm.le
  rw [eventually_sub_nonneg,
    condexp_of_stronglyMeasurable (ℱ.le _) (hf.adapted _) (hf.integrable _)]
  /-
    case pos
    Ω : Type u_1
    ι : Type u_3
    inst✝ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    f : ι → Ω → Real
    hf : MeasureTheory.Submartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    h : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.condexp (↑ℱ i) μ (f j))
  -/
  exact hf.2.1 i j hij
  /-
    🎉 no goals
  -/


theorem submartingale_iff_condexp_sub_nonneg [IsFiniteMeasure μ] {f : ι → Ω → ℝ} :
    Submartingale f ℱ μ ↔
      Adapted ℱ f ∧ (∀ i, Integrable (f i) μ) ∧ ∀ i j, i ≤ j → 0 ≤ᵐ[μ] μ[f j - f i|ℱ i] :=
  ⟨fun h => ⟨h.adapted, h.integrable, fun _ _ => h.condexp_sub_nonneg⟩, fun ⟨hadp, hint, h⟩ =>
    submartingale_of_condexp_sub_nonneg hadp hint h⟩


theorem sub_submartingale [Preorder E] [AddLeftMono E]
    (hf : Supermartingale f ℱ μ) (hg : Submartingale g ℱ μ) : Supermartingale (f - g) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝⁵ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f g : ι → Ω → E
    ℱ : MeasureTheory.Filtration ι m0
    inst✝¹ : Preorder E
    inst✝ : AddLeftMono E
    hf : MeasureTheory.Supermartingale f ℱ μ
    hg : MeasureTheory.Submartingale g ℱ μ
    ⊢ MeasureTheory.Supermartingale (HSub.hSub f g) ℱ μ
  -/
  rw [sub_eq_add_neg]; exact hf.add hg.neg
                       /-
                         🎉 no goals
                       -/


theorem sub_martingale [Preorder E] [AddLeftMono E]
    (hf : Supermartingale f ℱ μ) (hg : Martingale g ℱ μ) : Supermartingale (f - g) ℱ μ :=
  hf.sub_submartingale hg.submartingale


theorem smul_nonneg {f : ι → Ω → F} {c : ℝ} (hc : 0 ≤ c) (hf : Supermartingale f ℱ μ) :
    Supermartingale (c • f) ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Supermartingale f ℱ μ
    ⊢ MeasureTheory.Supermartingale (HSMul.hSMul c f) ℱ μ
  -/
  refine ⟨hf.1.smul c, fun i j hij => ?_, fun i => (hf.2.2 i).smul c⟩
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp (↑ℱ i) μ (HSMul.hSM …
  -/
  refine (condexp_smul c (f j)).le.trans ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    ⊢ (MeasureTheory.ae μ).EventuallyLE (HSMul.hSMul c (MeasureTheory.condexp (↑ℱ  …
  -/
  filter_upwards [hf.2.1 i j hij] with _ hle
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    a✝ : Ω
    hle : LE.le (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝) (f i a✝)
    ⊢ LE.le (HSMul.hSMul c (MeasureTheory.condexp (↑ℱ i) μ (f j)) a✝) (HSMul.hSMul …
  -/
  simp_rw [Pi.smul_apply]
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Supermartingale f ℱ μ
    i j : ι
    hij : LE.le i j
    a✝ : Ω
    hle : LE.le (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝) (f i a✝)
    ⊢ LE.le (HSMul.hSMul c (MeasureTheory.condexp (↑ℱ i) μ (f j) a✝)) (HSMul.hSMul …
  -/
  exact smul_le_smul_of_nonneg_left hle hc
  /-
    🎉 no goals
  -/


theorem smul_nonpos {f : ι → Ω → F} {c : ℝ} (hc : c ≤ 0) (hf : Supermartingale f ℱ μ) :
    Submartingale (c • f) ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le c 0
    hf : MeasureTheory.Supermartingale f ℱ μ
    ⊢ MeasureTheory.Submartingale (HSMul.hSMul c f) ℱ μ
  -/
  rw [← neg_neg c, (by ext (i x); simp : - -c • f = -(-c • f))]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le c 0
    hf : MeasureTheory.Supermartingale f ℱ μ
    ⊢ MeasureTheory.Submartingale (Neg.neg (HSMul.hSMul (Neg.neg c) f)) ℱ μ
  -/
  exact (hf.smul_nonneg <| neg_nonneg.2 hc).neg
  /-
    🎉 no goals
  -/


theorem smul_nonneg {f : ι → Ω → F} {c : ℝ} (hc : 0 ≤ c) (hf : Submartingale f ℱ μ) :
    Submartingale (c • f) ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ MeasureTheory.Submartingale (HSMul.hSMul c f) ℱ μ
  -/
  rw [← neg_neg c, (by ext (i x); simp : - -c • f = -(c • -f))]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le 0 c
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ MeasureTheory.Submartingale (Neg.neg (HSMul.hSMul c (Neg.neg f))) ℱ μ
  -/
  exact Supermartingale.neg (hf.neg.smul_nonneg hc)
  /-
    🎉 no goals
  -/


theorem smul_nonpos {f : ι → Ω → F} {c : ℝ} (hc : c ≤ 0) (hf : Submartingale f ℱ μ) :
    Supermartingale (c • f) ℱ μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le c 0
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ MeasureTheory.Supermartingale (HSMul.hSMul c f) ℱ μ
  -/
  rw [← neg_neg c, (by ext (i x); simp : - -c • f = -(-c • f))]
  /-
    Ω : Type u_1
    ι : Type u_3
    inst✝⁴ : Preorder ι
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m0
    F : Type u_4
    inst✝³ : NormedLatticeAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : OrderedSMul Real F
    f : ι → Ω → F
    c : Real
    hc : LE.le c 0
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ MeasureTheory.Supermartingale (Neg.neg (HSMul.hSMul (Neg.neg c) f)) ℱ μ
  -/
  exact (hf.smul_nonneg <| neg_nonneg.2 hc).neg
  /-
    🎉 no goals
  -/


theorem submartingale_of_setIntegral_le_succ [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ}
    (hadp : Adapted 𝒢 f) (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, ∀ s : Set Ω, MeasurableSet[𝒢 i] s → ∫ ω in s, f i ω ∂μ ≤ ∫ ω in s, f (i + 1) ω ∂μ) :
    Submartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
    ⊢ MeasureTheory.Submartingale f 𝒢 μ
  -/
  refine submartingale_of_setIntegral_le hadp hint fun i j hij s hs => ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
    i j : Nat
    hij : LE.le i j
    s : Set Ω
    hs : MeasurableSet s
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
  -/
  induction' hij with k hk₁ hk₂
    /-
      case refl
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : Nat → Ω → Real
      hadp : MeasureTheory.Adapted 𝒢 f
      hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
      hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
      i j : Nat
      s : Set Ω
      hs : MeasurableSet s
      ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case step
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : Nat → Ω → Real
      hadp : MeasureTheory.Adapted 𝒢 f
      hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
      hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
      i j : Nat
      s : Set Ω
      hs : MeasurableSet s
      k : Nat
      hk₁ : i.le k
      hk₂ : LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureThe …
      ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
    -/
  · exact le_trans hk₂ (hf k s (𝒢.mono hk₁ _ hs))
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias submartingale_of_set_integral_le_succ := submartingale_of_setIntegral_le_succ


theorem supermartingale_of_setIntegral_succ_le [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ}
    (hadp : Adapted 𝒢 f) (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, ∀ s : Set Ω, MeasurableSet[𝒢 i] s → ∫ ω in s, f (i + 1) ω ∂μ ≤ ∫ ω in s, f i ω ∂μ) :
    Supermartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
    ⊢ MeasureTheory.Supermartingale f 𝒢 μ
  -/
  rw [← neg_neg f]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
    ⊢ MeasureTheory.Supermartingale (Neg.neg (Neg.neg f)) 𝒢 μ
  -/
  refine (submartingale_of_setIntegral_le_succ hadp.neg (fun i => (hint i).neg) ?_).neg
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral  …
    ⊢ ∀ (i : Nat) (s : Set Ω), MeasurableSet s → LE.le (MeasureTheory.integral (μ. …
  -/
  simpa only [integral_neg, Pi.neg_apply, neg_le_neg_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias supermartingale_of_set_integral_succ_le := supermartingale_of_setIntegral_succ_le


theorem martingale_of_setIntegral_eq_succ [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ} (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, ∀ s : Set Ω, MeasurableSet[𝒢 i] s → ∫ ω in s, f i ω ∂μ = ∫ ω in s, f (i + 1) ω ∂μ) :
    Martingale f 𝒢 μ :=
  martingale_iff.2 ⟨supermartingale_of_setIntegral_succ_le hadp hint fun i s hs => (hf i s hs).ge,
    submartingale_of_setIntegral_le_succ hadp hint fun i s hs => (hf i s hs).le⟩


@[deprecated (since := "2024-04-17")]
alias martingale_of_set_integral_eq_succ := martingale_of_setIntegral_eq_succ


theorem submartingale_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ} (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ i, f i ≤ᵐ[μ] μ[f (i + 1)|𝒢 i]) :
    Submartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.conde …
    ⊢ MeasureTheory.Submartingale f 𝒢 μ
  -/
  refine submartingale_of_setIntegral_le_succ hadp hint fun i s hs => ?_
  have : ∫ ω in s, f (i + 1) ω ∂μ = ∫ ω in s, (μ[f (i + 1)|𝒢 i]) ω ∂μ :=
    (setIntegral_condexp (𝒢.le i) (hint _) hs).symm
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.conde …
    i : Nat
    s : Set Ω
    hs : MeasurableSet s
    this : Eq (MeasureTheory.integral (μ.restrict s) fun ω => f (HAdd.hAdd i 1) ω) …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
  -/
  rw [this]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.conde …
    i : Nat
    s : Set Ω
    hs : MeasurableSet s
    this : Eq (MeasureTheory.integral (μ.restrict s) fun ω => f (HAdd.hAdd i 1) ω) …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun ω => f i ω) (MeasureTheory. …
  -/
  exact setIntegral_mono_ae (hint i).integrableOn integrable_condexp.integrableOn (hf i)
  /-
    🎉 no goals
  -/


theorem supermartingale_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ} (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ i, μ[f (i + 1)|𝒢 i] ≤ᵐ[μ] f i) :
    Supermartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp (↑𝒢 …
    ⊢ MeasureTheory.Supermartingale f 𝒢 μ
  -/
  rw [← neg_neg f]
  refine (submartingale_nat hadp.neg (fun i => (hint i).neg) fun i =>
    EventuallyLE.trans ?_ (condexp_neg _).symm.le).neg
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE (MeasureTheory.condexp (↑𝒢 …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE (Neg.neg f i) (Neg.neg (MeasureTheory.cond …
  -/
  filter_upwards [hf i] with x hx using neg_le_neg hx
  /-
    🎉 no goals
  -/


theorem martingale_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ} (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ i, f i =ᵐ[μ] μ[f (i + 1)|𝒢 i]) : Martingale f 𝒢 μ :=
  martingale_iff.2 ⟨supermartingale_nat hadp hint fun i => (hf i).symm.le,
    submartingale_nat hadp hint fun i => (hf i).le⟩


theorem submartingale_of_condexp_sub_nonneg_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ}
    (hadp : Adapted 𝒢 f) (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, 0 ≤ᵐ[μ] μ[f (i + 1) - f i|𝒢 i]) : Submartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    ⊢ MeasureTheory.Submartingale f 𝒢 μ
  -/
  refine submartingale_nat hadp hint fun i => ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f i) (MeasureTheory.condexp (↑𝒢 i) μ (f ( …
  -/
  rw [← condexp_of_stronglyMeasurable (𝒢.le _) (hadp _) (hint _), ← eventually_sub_nonneg]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (HSub.hSub (MeasureTheory.condexp (↑𝒢 i) …
  -/
  exact EventuallyLE.trans (hf i) (condexp_sub (hint _) (hint _)).le
  /-
    🎉 no goals
  -/


theorem supermartingale_of_condexp_sub_nonneg_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ}
    (hadp : Adapted 𝒢 f) (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, 0 ≤ᵐ[μ] μ[f i - f (i + 1)|𝒢 i]) : Supermartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    ⊢ MeasureTheory.Supermartingale f 𝒢 μ
  -/
  rw [← neg_neg f]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    ⊢ MeasureTheory.Supermartingale (Neg.neg (Neg.neg f)) 𝒢 μ
  -/
  refine (submartingale_of_condexp_sub_nonneg_nat hadp.neg (fun i => (hint i).neg) ?_).neg
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp ( …
    ⊢ ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑𝒢  …
  -/
  simpa only [Pi.zero_apply, Pi.neg_apply, neg_sub_neg]
  /-
    🎉 no goals
  -/


theorem martingale_of_condexp_sub_eq_zero_nat [IsFiniteMeasure μ] {f : ℕ → Ω → ℝ}
    (hadp : Adapted 𝒢 f) (hint : ∀ i, Integrable (f i) μ)
    (hf : ∀ i, μ[f (i + 1) - f i|𝒢 i] =ᵐ[μ] 0) : Martingale f 𝒢 μ := by
  refine martingale_iff.2 ⟨supermartingale_of_condexp_sub_nonneg_nat hadp hint fun i => ?_,
    submartingale_of_condexp_sub_nonneg_nat hadp hint fun i => (hf i).symm.le⟩
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑𝒢 …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑𝒢 i) μ (HSub.hS …
  -/
  rw [← neg_sub]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑𝒢 …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.condexp (↑𝒢 i) μ (Neg.neg …
  -/
  refine (EventuallyEq.trans ?_ (condexp_neg _).symm).le
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑𝒢 …
    i : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq 0 (Neg.neg (MeasureTheory.condexp (↑𝒢 i) μ …
  -/
  filter_upwards [hf i] with x hx
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Ω → Real
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (i : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑𝒢 …
    i : Nat
    x : Ω
    hx : Eq (MeasureTheory.condexp (↑𝒢 i) μ (HSub.hSub (f (HAdd.hAdd i 1)) (f i))  …
    ⊢ Eq (0 x) (Neg.neg (MeasureTheory.condexp (↑𝒢 i) μ (HSub.hSub (f (HAdd.hAdd i …
  -/
  simpa only [Pi.zero_apply, Pi.neg_apply, zero_eq_neg]
  /-
    🎉 no goals
  -/

-- Note that one cannot use `Submartingale.zero_le_of_predictable` to prove the other two
-- corresponding lemmas without imposing more restrictions to the ordering of `E`

/-- A predictable submartingale is a.e. greater equal than its initial state. -/
theorem Submartingale.zero_le_of_predictable [Preorder E] [SigmaFiniteFiltration μ 𝒢]
    {f : ℕ → Ω → E} (hfmgle : Submartingale f 𝒢 μ) (hfadp : Adapted 𝒢 fun n => f (n + 1)) (n : ℕ) :
    f 0 ≤ᵐ[μ] f n := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝¹ : Preorder E
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
    f : Nat → Ω → E
    hfmgle : MeasureTheory.Submartingale f 𝒢 μ
    hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f 0) (f n)
  -/
  induction' n with k ih
    /-
      case zero
      Ω : Type u_1
      E : Type u_2
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      𝒢 : MeasureTheory.Filtration Nat m0
      inst✝¹ : Preorder E
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      f : Nat → Ω → E
      hfmgle : MeasureTheory.Submartingale f 𝒢 μ
      hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
      ⊢ (MeasureTheory.ae μ).EventuallyLE (f 0) (f 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · exact ih.trans ((hfmgle.2.1 k (k + 1) k.le_succ).trans_eq <| Germ.coe_eq.mp <|
    congr_arg Germ.ofFun <| condexp_of_stronglyMeasurable (𝒢.le _) (hfadp _) <| hfmgle.integrable _)


/-- A predictable supermartingale is a.e. less equal than its initial state. -/
theorem Supermartingale.le_zero_of_predictable [Preorder E] [SigmaFiniteFiltration μ 𝒢]
    {f : ℕ → Ω → E} (hfmgle : Supermartingale f 𝒢 μ) (hfadp : Adapted 𝒢 fun n => f (n + 1))
    (n : ℕ) : f n ≤ᵐ[μ] f 0 := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝¹ : Preorder E
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
    f : Nat → Ω → E
    hfmgle : MeasureTheory.Supermartingale f 𝒢 μ
    hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyLE (f n) (f 0)
  -/
  induction' n with k ih
    /-
      case zero
      Ω : Type u_1
      E : Type u_2
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      𝒢 : MeasureTheory.Filtration Nat m0
      inst✝¹ : Preorder E
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      f : Nat → Ω → E
      hfmgle : MeasureTheory.Supermartingale f 𝒢 μ
      hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
      ⊢ (MeasureTheory.ae μ).EventuallyLE (f 0) (f 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · exact ((Germ.coe_eq.mp <| congr_arg Germ.ofFun <| condexp_of_stronglyMeasurable (𝒢.le _)
      (hfadp _) <| hfmgle.integrable _).symm.trans_le (hfmgle.2.1 k (k + 1) k.le_succ)).trans ih


/-- A predictable martingale is a.e. equal to its initial state. -/
theorem Martingale.eq_zero_of_predictable [SigmaFiniteFiltration μ 𝒢] {f : ℕ → Ω → E}
    (hfmgle : Martingale f 𝒢 μ) (hfadp : Adapted 𝒢 fun n => f (n + 1)) (n : ℕ) : f n =ᵐ[μ] f 0 := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
    f : Nat → Ω → E
    hfmgle : MeasureTheory.Martingale f 𝒢 μ
    hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (f n) (f 0)
  -/
  induction' n with k ih
    /-
      case zero
      Ω : Type u_1
      E : Type u_2
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      𝒢 : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      f : Nat → Ω → E
      hfmgle : MeasureTheory.Martingale f 𝒢 μ
      hfadp : MeasureTheory.Adapted 𝒢 fun n => f (HAdd.hAdd n 1)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (f 0) (f 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · exact ((Germ.coe_eq.mp (congr_arg Germ.ofFun <| condexp_of_stronglyMeasurable (𝒢.le _) (hfadp _)
      (hfmgle.integrable _))).symm.trans (hfmgle.2 k (k + 1) k.le_succ)).trans ih


protected theorem integrable_stoppedValue [LE E] {f : ℕ → Ω → E} (hf : Submartingale f 𝒢 μ)
    {τ : Ω → ℕ} (hτ : IsStoppingTime 𝒢 τ) {N : ℕ} (hbdd : ∀ ω, τ ω ≤ N) :
    Integrable (stoppedValue f τ) μ :=
  integrable_stoppedValue ℕ hτ hf.integrable hbdd


theorem Submartingale.sum_mul_sub [IsFiniteMeasure μ] {R : ℝ} {ξ f : ℕ → Ω → ℝ}
    (hf : Submartingale f 𝒢 μ) (hξ : Adapted 𝒢 ξ) (hbdd : ∀ n ω, ξ n ω ≤ R)
    (hnonneg : ∀ n ω, 0 ≤ ξ n ω) :
    Submartingale (fun n => ∑ k ∈ Finset.range n, ξ k * (f (k + 1) - f k)) 𝒢 μ := by
  have hξbdd : ∀ i, ∃ C, ∀ ω, |ξ i ω| ≤ C := fun i =>
    ⟨R, fun ω => (abs_of_nonneg (hnonneg i ω)).trans_le (hbdd i ω)⟩
  have hint : ∀ m, Integrable (∑ k ∈ Finset.range m, ξ k * (f (k + 1) - f k)) μ := fun m =>
    integrable_finset_sum' _ fun i _ => Integrable.bdd_mul ((hf.integrable _).sub (hf.integrable _))
      hξ.stronglyMeasurable.aestronglyMeasurable (hξbdd _)
  have hadp : Adapted 𝒢 fun n => ∑ k ∈ Finset.range n, ξ k * (f (k + 1) - f k) := by
    intro m
    refine Finset.stronglyMeasurable_sum' _ fun i hi => ?_
    rw [Finset.mem_range] at hi
    exact (hξ.stronglyMeasurable_le hi.le).mul
      ((hf.adapted.stronglyMeasurable_le (Nat.succ_le_of_lt hi)).sub
        (hf.adapted.stronglyMeasurable_le hi.le))
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    R : Real
    ξ f : Nat → Ω → Real
    hf : MeasureTheory.Submartingale f 𝒢 μ
    hξ : MeasureTheory.Adapted 𝒢 ξ
    hbdd : ∀ (n : Nat) (ω : Ω), LE.le (ξ n ω) R
    hnonneg : ∀ (n : Nat) (ω : Ω), LE.le 0 (ξ n ω)
    hξbdd : ∀ (i : Nat), Exists fun C => ∀ (ω : Ω), LE.le (abs (ξ i ω)) C
    hint : ∀ (m : Nat), MeasureTheory.Integrable ((Finset.range m).sum fun k => HM …
    hadp : MeasureTheory.Adapted 𝒢 fun n => (Finset.range n).sum fun k => HMul.hMu …
    ⊢ MeasureTheory.Submartingale (fun n => (Finset.range n).sum fun k => HMul.hMu …
  -/
  refine submartingale_of_condexp_sub_nonneg_nat hadp hint fun i => ?_
  simp only [← Finset.sum_Ico_eq_sub _ (Nat.le_succ _), Finset.sum_apply, Pi.mul_apply,
    Pi.sub_apply, Nat.Ico_succ_singleton, Finset.sum_singleton]
  exact EventuallyLE.trans (EventuallyLE.mul_nonneg (Eventually.of_forall (hnonneg _))
    (hf.condexp_sub_nonneg (Nat.le_succ _))) (condexp_stronglyMeasurable_mul (hξ _)
    (((hf.integrable _).sub (hf.integrable _)).bdd_mul
      hξ.stronglyMeasurable.aestronglyMeasurable (hξbdd _))
    ((hf.integrable _).sub (hf.integrable _))).symm.le


/-- Given a discrete submartingale `f` and a predictable process `ξ` (i.e. `ξ (n + 1)` is adapted)
the process defined by `fun n => ∑ k ∈ Finset.range n, ξ (k + 1) * (f (k + 1) - f k)` is also a
submartingale. -/
theorem Submartingale.sum_mul_sub' [IsFiniteMeasure μ] {R : ℝ} {ξ f : ℕ → Ω → ℝ}
    (hf : Submartingale f 𝒢 μ) (hξ : Adapted 𝒢 fun n => ξ (n + 1)) (hbdd : ∀ n ω, ξ n ω ≤ R)
    (hnonneg : ∀ n ω, 0 ≤ ξ n ω) :
    Submartingale (fun n => ∑ k ∈ Finset.range n, ξ (k + 1) * (f (k + 1) - f k)) 𝒢 μ :=
  hf.sum_mul_sub hξ (fun _ => hbdd _) fun _ => hnonneg _


