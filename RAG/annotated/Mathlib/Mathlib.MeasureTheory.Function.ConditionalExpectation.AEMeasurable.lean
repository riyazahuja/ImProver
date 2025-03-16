/-- A function `f` verifies `AEStronglyMeasurable' m f μ` if it is `μ`-a.e. equal to
an `m`-strongly measurable function. This is similar to `AEStronglyMeasurable`, but the
`MeasurableSpace` structures used for the measurability statement and for the measure are
different. -/
def AEStronglyMeasurable' {α β} [TopologicalSpace β] (m : MeasurableSpace α)
    {_ : MeasurableSpace α} (f : α → β) (μ : Measure α) : Prop :=
  ∃ g : α → β, StronglyMeasurable[m] g ∧ f =ᵐ[μ] g


theorem congr (hf : AEStronglyMeasurable' m f μ) (hfg : f =ᵐ[μ] g) :
    AEStronglyMeasurable' m g μ := by
  /-
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ MeasureTheory.AEStronglyMeasurable' m g μ
  -/
  obtain ⟨f', hf'_meas, hff'⟩ := hf; exact ⟨f', hf'_meas, hfg.symm.trans hff'⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem mono {m'} (hf : AEStronglyMeasurable' m f μ) (hm : m ≤ m') :
    AEStronglyMeasurable' m' f μ :=
  let ⟨f', hf'_meas, hff'⟩ := hf; ⟨f', hf'_meas.mono hm, hff'⟩


theorem add [Add β] [ContinuousAdd β] (hf : AEStronglyMeasurable' m f μ)
    (hg : AEStronglyMeasurable' m g μ) : AEStronglyMeasurable' m (f + g) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f g : α → β
    inst✝¹ : Add β
    inst✝ : ContinuousAdd β
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.AEStronglyMeasurable' m g μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HAdd.hAdd f g) μ
  -/
  rcases hf with ⟨f', h_f'_meas, hff'⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f g : α → β
    inst✝¹ : Add β
    inst✝ : ContinuousAdd β
    hg : MeasureTheory.AEStronglyMeasurable' m g μ
    f' : α → β
    h_f'_meas : MeasureTheory.StronglyMeasurable f'
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HAdd.hAdd f g) μ
  -/
  rcases hg with ⟨g', h_g'_meas, hgg'⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f g : α → β
    inst✝¹ : Add β
    inst✝ : ContinuousAdd β
    f' : α → β
    h_f'_meas : MeasureTheory.StronglyMeasurable f'
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    g' : α → β
    h_g'_meas : MeasureTheory.StronglyMeasurable g'
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HAdd.hAdd f g) μ
  -/
  exact ⟨f' + g', h_f'_meas.add h_g'_meas, hff'.add hgg'⟩
  /-
    🎉 no goals
  -/


theorem neg [AddGroup β] [TopologicalAddGroup β] {f : α → β} (hfm : AEStronglyMeasurable' m f μ) :
    AEStronglyMeasurable' m (-f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f : α → β
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (Neg.neg f) μ
  -/
  rcases hfm with ⟨f', hf'_meas, hf_ae⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (Neg.neg f) μ
  -/
  refine ⟨-f', hf'_meas.neg, hf_ae.mono fun x hx => ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    x : α
    hx : Eq (f x) (f' x)
    ⊢ Eq (Neg.neg f x) (Neg.neg f' x)
  -/
  simp_rw [Pi.neg_apply]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    x : α
    hx : Eq (f x) (f' x)
    ⊢ Eq (Neg.neg (f x)) (Neg.neg (f' x))
  -/
  rw [hx]
  /-
    🎉 no goals
  -/


theorem sub [AddGroup β] [TopologicalAddGroup β] {f g : α → β} (hfm : AEStronglyMeasurable' m f μ)
    (hgm : AEStronglyMeasurable' m g μ) : AEStronglyMeasurable' m (f - g) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f g : α → β
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HSub.hSub f g) μ
  -/
  rcases hfm with ⟨f', hf'_meas, hf_ae⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f g : α → β
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HSub.hSub f g) μ
  -/
  rcases hgm with ⟨g', hg'_meas, hg_ae⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f g f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    g' : α → β
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg_ae : (MeasureTheory.ae μ).EventuallyEq g g'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HSub.hSub f g) μ
  -/
  refine ⟨f' - g', hf'_meas.sub hg'_meas, hf_ae.mp (hg_ae.mono fun x hx1 hx2 => ?_)⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f g f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    g' : α → β
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg_ae : (MeasureTheory.ae μ).EventuallyEq g g'
    x : α
    hx1 : Eq (g x) (g' x)
    hx2 : Eq (f x) (f' x)
    ⊢ Eq (HSub.hSub f g x) (HSub.hSub f' g' x)
  -/
  simp_rw [Pi.sub_apply]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f g f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    g' : α → β
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg_ae : (MeasureTheory.ae μ).EventuallyEq g g'
    x : α
    hx1 : Eq (g x) (g' x)
    hx2 : Eq (f x) (f' x)
    ⊢ Eq (HSub.hSub (f x) (g x)) (HSub.hSub (f' x) (g' x))
  -/
  rw [hx1, hx2]
  /-
    🎉 no goals
  -/


theorem const_smul [SMul 𝕜 β] [ContinuousConstSMul 𝕜 β] (c : 𝕜) (hf : AEStronglyMeasurable' m f μ) :
    AEStronglyMeasurable' m (c • f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    𝕜 : Type u_3
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : SMul 𝕜 β
    inst✝ : ContinuousConstSMul 𝕜 β
    c : 𝕜
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HSMul.hSMul c f) μ
  -/
  rcases hf with ⟨f', h_f'_meas, hff'⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    𝕜 : Type u_3
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : SMul 𝕜 β
    inst✝ : ContinuousConstSMul 𝕜 β
    c : 𝕜
    f' : α → β
    h_f'_meas : MeasureTheory.StronglyMeasurable f'
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (HSMul.hSMul c f) μ
  -/
  refine ⟨c • f', h_f'_meas.const_smul c, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    𝕜 : Type u_3
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : SMul 𝕜 β
    inst✝ : ContinuousConstSMul 𝕜 β
    c : 𝕜
    f' : α → β
    h_f'_meas : MeasureTheory.StronglyMeasurable f'
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul c f) (HSMul.hSMul c f')
  -/
  exact EventuallyEq.fun_comp hff' fun x => c • x
  /-
    🎉 no goals
  -/


theorem const_inner {𝕜 β} [RCLike 𝕜] [NormedAddCommGroup β] [InnerProductSpace 𝕜 β] {f : α → β}
    (hfm : AEStronglyMeasurable' m f μ) (c : β) :
    AEStronglyMeasurable' m (fun x => (inner c (f x) : 𝕜)) μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_4
    β : Type u_5
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup β
    inst✝ : InnerProductSpace 𝕜 β
    f : α → β
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    c : β
    ⊢ MeasureTheory.AEStronglyMeasurable' m (fun x => Inner.inner c (f x)) μ
  -/
  rcases hfm with ⟨f', hf'_meas, hf_ae⟩
  refine
    ⟨fun x => (inner c (f' x) : 𝕜), (@stronglyMeasurable_const _ _ m _ c).inner hf'_meas,
      hf_ae.mono fun x hx => ?_⟩
  /-
    case intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_4
    β : Type u_5
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup β
    inst✝ : InnerProductSpace 𝕜 β
    f : α → β
    c : β
    f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    x : α
    hx : Eq (f x) (f' x)
    ⊢ Eq ((fun x => Inner.inner c (f x)) x) ((fun x => Inner.inner c (f' x)) x)
  -/
  dsimp only
  /-
    case intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_4
    β : Type u_5
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup β
    inst✝ : InnerProductSpace 𝕜 β
    f : α → β
    c : β
    f' : α → β
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf_ae : (MeasureTheory.ae μ).EventuallyEq f f'
    x : α
    hx : Eq (f x) (f' x)
    ⊢ Eq (Inner.inner c (f x)) (Inner.inner c (f' x))
  -/
  rw [hx]
  /-
    🎉 no goals
  -/


@[simp] theorem of_subsingleton [Subsingleton β] : AEStronglyMeasurable' m f μ :=
         /-
           α : Type u_1
           β : Type u_2
           m m0 : MeasurableSpace α
           μ : MeasureTheory.Measure α
           inst✝¹ : TopologicalSpace β
           f : α → β
           inst✝ : Subsingleton β
           ⊢ MeasureTheory.StronglyMeasurable f
         -/
         /-
           🎉 no goals
         -/
  ⟨f, by simp, by simp⟩
                  /-
                    🎉 no goals
                  -/


@[simp] theorem of_subsingleton' [Subsingleton α] : AEStronglyMeasurable' m f μ :=
         /-
           α : Type u_1
           β : Type u_2
           m m0 : MeasurableSpace α
           μ : MeasureTheory.Measure α
           inst✝¹ : TopologicalSpace β
           f : α → β
           inst✝ : Subsingleton α
           ⊢ MeasureTheory.StronglyMeasurable f
         -/
         /-
           🎉 no goals
         -/
  ⟨f, by simp, by simp⟩
                  /-
                    🎉 no goals
                  -/


/-- An `m`-strongly measurable function almost everywhere equal to `f`. -/
noncomputable def mk (f : α → β) (hfm : AEStronglyMeasurable' m f μ) : α → β :=
  hfm.choose


theorem stronglyMeasurable_mk {f : α → β} (hfm : AEStronglyMeasurable' m f μ) :
    StronglyMeasurable[m] (hfm.mk f) :=
  hfm.choose_spec.1


theorem ae_eq_mk {f : α → β} (hfm : AEStronglyMeasurable' m f μ) : f =ᵐ[μ] hfm.mk f :=
  hfm.choose_spec.2


theorem continuous_comp {γ} [TopologicalSpace γ] {f : α → β} {g : β → γ} (hg : Continuous g)
    (hf : AEStronglyMeasurable' m f μ) : AEStronglyMeasurable' m (g ∘ f) μ :=
  ⟨fun x => g (hf.mk _ x),
    @Continuous.comp_stronglyMeasurable _ _ _ m _ _ _ _ hg hf.stronglyMeasurable_mk,
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      m m0 : MeasurableSpace α
                                      μ : MeasureTheory.Measure α
                                      inst✝¹ : TopologicalSpace β
                                      γ : Type u_4
                                      inst✝ : TopologicalSpace γ
                                      f : α → β
                                      g : β → γ
                                      hg : Continuous g
                                      hf : MeasureTheory.AEStronglyMeasurable' m f μ
                                      x : α
                                      hx : Eq (f x) (MeasureTheory.AEStronglyMeasurable'.mk f hf x)
                                      ⊢ Eq (Function.comp g f x) ((fun x => g (MeasureTheory.AEStronglyMeasurable'.m …
                                    -/
    hf.ae_eq_mk.mono fun x hx => by rw [Function.comp_apply, hx]⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem aeStronglyMeasurable'_of_aeStronglyMeasurable'_trim {α β} {m m0 m0' : MeasurableSpace α}
    [TopologicalSpace β] (hm0 : m0 ≤ m0') {μ : Measure α} {f : α → β}
    (hf : AEStronglyMeasurable' m f (μ.trim hm0)) : AEStronglyMeasurable' m f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m m0 m0' : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hm0 : LE.le m0 m0'
    μ : MeasureTheory.Measure α
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable' m f (μ.trim hm0)
    ⊢ MeasureTheory.AEStronglyMeasurable' m f μ
  -/
  obtain ⟨g, hg_meas, hfg⟩ := hf; exact ⟨g, hg_meas, ae_eq_of_ae_eq_trim hfg⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem StronglyMeasurable.aeStronglyMeasurable' {α β} {m _ : MeasurableSpace α}
    [TopologicalSpace β] {μ : Measure α} {f : α → β} (hf : StronglyMeasurable[m] f) :
    AEStronglyMeasurable' m f μ :=
  ⟨f, hf, ae_eq_refl _⟩


theorem ae_eq_trim_iff_of_aeStronglyMeasurable' {α β} [TopologicalSpace β] [MetrizableSpace β]
    {m m0 : MeasurableSpace α} {μ : Measure α} {f g : α → β} (hm : m ≤ m0)
    (hfm : AEStronglyMeasurable' m f μ) (hgm : AEStronglyMeasurable' m g μ) :
    hfm.mk f =ᵐ[μ.trim hm] hgm.mk g ↔ f =ᵐ[μ] g :=
  (ae_eq_trim_iff hm hfm.stronglyMeasurable_mk hgm.stronglyMeasurable_mk).trans
    ⟨fun h => hfm.ae_eq_mk.trans (h.trans hgm.ae_eq_mk.symm), fun h =>
      hfm.ae_eq_mk.symm.trans (h.trans hgm.ae_eq_mk)⟩


theorem AEStronglyMeasurable.comp_ae_measurable' {α β γ : Type*} [TopologicalSpace β]
    {mα : MeasurableSpace α} {_ : MeasurableSpace γ} {f : α → β} {μ : Measure γ} {g : γ → α}
    (hf : AEStronglyMeasurable f (μ.map g)) (hg : AEMeasurable g μ) :
    AEStronglyMeasurable' (mα.comap g) (f ∘ g) μ :=
  ⟨hf.mk f ∘ g, hf.stronglyMeasurable_mk.comp_measurable (measurable_iff_comap_le.mpr le_rfl),
    ae_eq_comp hg hf.ae_eq_mk⟩


/-- If the restriction to a set `s` of a σ-algebra `m` is included in the restriction to `s` of
another σ-algebra `m₂` (hypothesis `hs`), the set `s` is `m` measurable and a function `f` almost
everywhere supported on `s` is `m`-ae-strongly-measurable, then `f` is also
`m₂`-ae-strongly-measurable. -/
theorem AEStronglyMeasurable'.aeStronglyMeasurable'_of_measurableSpace_le_on {α E}
    {m m₂ m0 : MeasurableSpace α} {μ : Measure α} [TopologicalSpace E] [Zero E] (hm : m ≤ m0)
    {s : Set α} {f : α → E} (hs_m : MeasurableSet[m] s)
    (hs : ∀ t, MeasurableSet[m] (s ∩ t) → MeasurableSet[m₂] (s ∩ t))
    (hf : AEStronglyMeasurable' m f μ) (hf_zero : f =ᵐ[μ.restrict sᶜ] 0) :
    AEStronglyMeasurable' m₂ f μ := by
  have h_ind_eq : s.indicator (hf.mk f) =ᵐ[μ] f := by
    refine Filter.EventuallyEq.trans ?_ <|
      indicator_ae_eq_of_restrict_compl_ae_eq_zero (hm _ hs_m) hf_zero
    filter_upwards [hf.ae_eq_mk] with x hx
    by_cases hxs : x ∈ s
    · simp [hxs, hx]
    · simp [hxs]
  suffices StronglyMeasurable[m₂] (s.indicator (hf.mk f)) from
    AEStronglyMeasurable'.congr this.aeStronglyMeasurable' h_ind_eq
  have hf_ind : StronglyMeasurable[m] (s.indicator (hf.mk f)) :=
    hf.stronglyMeasurable_mk.indicator hs_m
  exact
    hf_ind.stronglyMeasurable_of_measurableSpace_le_on hs_m hs fun x hxs =>
      Set.indicator_of_not_mem hxs _


/-- `lpMeasSubgroup F m p μ` is the subspace of `Lp F p μ` containing functions `f` verifying
`AEStronglyMeasurable' m f μ`, i.e. functions which are `μ`-a.e. equal to
an `m`-strongly measurable function. -/
def lpMeasSubgroup (m : MeasurableSpace α) [MeasurableSpace α] (p : ℝ≥0∞) (μ : Measure α) :
    AddSubgroup (Lp F p μ) where
  carrier := {f : Lp F p μ | AEStronglyMeasurable' m f μ}
  zero_mem' := ⟨(0 : α → F), @stronglyMeasurable_zero _ _ m _ _, Lp.coeFn_zero _ _ _⟩
  add_mem' {f g} hf hg := (hf.add hg).congr (Lp.coeFn_add f g).symm
  neg_mem' {f} hf := AEStronglyMeasurable'.congr hf.neg (Lp.coeFn_neg f).symm


/-- `lpMeas F 𝕜 m p μ` is the subspace of `Lp F p μ` containing functions `f` verifying
`AEStronglyMeasurable' m f μ`, i.e. functions which are `μ`-a.e. equal to
an `m`-strongly measurable function. -/
def lpMeas (m : MeasurableSpace α) [MeasurableSpace α] (p : ℝ≥0∞) (μ : Measure α) :
    Submodule 𝕜 (Lp F p μ) where
  carrier := {f : Lp F p μ | AEStronglyMeasurable' m f μ}
  zero_mem' := ⟨(0 : α → F), @stronglyMeasurable_zero _ _ m _ _, Lp.coeFn_zero _ _ _⟩
  add_mem' {f g} hf hg := (hf.add hg).congr (Lp.coeFn_add f g).symm
  smul_mem' c f hf := (hf.const_smul c).congr (Lp.coeFn_smul c f).symm


theorem mem_lpMeasSubgroup_iff_aeStronglyMeasurable' {m m0 : MeasurableSpace α} {μ : Measure α}
    {f : Lp F p μ} : f ∈ lpMeasSubgroup F m p μ ↔ AEStronglyMeasurable' m f μ := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    ⊢ Iff (Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f) (MeasureTheory …
  -/
  rw [← AddSubgroup.mem_carrier, lpMeasSubgroup, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem mem_lpMeas_iff_aeStronglyMeasurable' {m m0 : MeasurableSpace α} {μ : Measure α}
    {f : Lp F p μ} : f ∈ lpMeas F 𝕜 m p μ ↔ AEStronglyMeasurable' m f μ := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    ⊢ Iff (Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) f) (MeasureTheory.AEStr …
  -/
  rw [← SetLike.mem_coe, ← Submodule.mem_carrier, lpMeas, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem lpMeas.aeStronglyMeasurable' {m _ : MeasurableSpace α} {μ : Measure α}
    (f : lpMeas F 𝕜 m p μ) : AEStronglyMeasurable' (β := F) m f μ :=
  mem_lpMeas_iff_aeStronglyMeasurable'.mp f.mem


theorem mem_lpMeas_self {m0 : MeasurableSpace α} (μ : Measure α) (f : Lp F p μ) :
    f ∈ lpMeas F 𝕜 m0 p μ :=
  mem_lpMeas_iff_aeStronglyMeasurable'.mpr (Lp.aestronglyMeasurable f)


theorem lpMeasSubgroup_coe {m _ : MeasurableSpace α} {μ : Measure α} {f : lpMeasSubgroup F m p μ} :
    (f : _ → _) = (f : Lp F p μ) :=
  rfl


theorem lpMeas_coe {m _ : MeasurableSpace α} {μ : Measure α} {f : lpMeas F 𝕜 m p μ} :
    (f : _ → _) = (f : Lp F p μ) :=
  rfl


theorem mem_lpMeas_indicatorConstLp {m m0 : MeasurableSpace α} (hm : m ≤ m0) {μ : Measure α}
    {s : Set α} (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) {c : F} :
    indicatorConstLp p (hm s hs) hμs c ∈ lpMeas F 𝕜 m p μ :=
  ⟨s.indicator fun _ : α => c, (@stronglyMeasurable_const _ _ m _ _).indicator hs,
    indicatorConstLp_coeFn⟩


/-- If `f` belongs to `lpMeasSubgroup F m p μ`, then the measurable function it is almost
everywhere equal to (given by `AEMeasurable.mk`) belongs to `ℒp` for the measure `μ.trim hm`. -/
theorem memℒp_trim_of_mem_lpMeasSubgroup (hm : m ≤ m0) (f : Lp F p μ)
    (hf_meas : f ∈ lpMeasSubgroup F m p μ) :
    Memℒp (mem_lpMeasSubgroup_iff_aeStronglyMeasurable'.mp hf_meas).choose p (μ.trim hm) := by
  have hf : AEStronglyMeasurable' m f μ :=
    mem_lpMeasSubgroup_iff_aeStronglyMeasurable'.mp hf_meas
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    ⊢ MeasureTheory.Memℒp (Exists.choose ⋯) p (μ.trim hm)
  -/
  let g := hf.choose
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    g : α → F := Exists.choose hf
    ⊢ MeasureTheory.Memℒp (Exists.choose ⋯) p (μ.trim hm)
  -/
  obtain ⟨hg, hfg⟩ := hf.choose_spec
  /-
    case intro
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    g : α → F := Exists.choose hf
    hg : MeasureTheory.StronglyMeasurable (Exists.choose hf)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑f) (Exists.choose hf)
    ⊢ MeasureTheory.Memℒp (Exists.choose ⋯) p (μ.trim hm)
  -/
  change Memℒp g p (μ.trim hm)
  /-
    case intro
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    g : α → F := Exists.choose hf
    hg : MeasureTheory.StronglyMeasurable (Exists.choose hf)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑f) (Exists.choose hf)
    ⊢ MeasureTheory.Memℒp g p (μ.trim hm)
  -/
  refine ⟨hg.aestronglyMeasurable, ?_⟩
  have h_eLpNorm_fg : eLpNorm g p (μ.trim hm) = eLpNorm f p μ := by
    rw [eLpNorm_trim hm hg]
    exact eLpNorm_congr_ae hfg.symm
  /-
    case intro
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    g : α → F := Exists.choose hf
    hg : MeasureTheory.StronglyMeasurable (Exists.choose hf)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑f) (Exists.choose hf)
    h_eLpNorm_fg : Eq (MeasureTheory.eLpNorm g p (μ.trim hm)) (MeasureTheory.eLpNo …
    ⊢ LT.lt (MeasureTheory.eLpNorm g p (μ.trim hm)) Top.top
  -/
  rw [h_eLpNorm_fg]
  /-
    case intro
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf_meas : Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) f
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    g : α → F := Exists.choose hf
    hg : MeasureTheory.StronglyMeasurable (Exists.choose hf)
    hfg : (MeasureTheory.ae μ).EventuallyEq (↑↑f) (Exists.choose hf)
    h_eLpNorm_fg : Eq (MeasureTheory.eLpNorm g p (μ.trim hm)) (MeasureTheory.eLpNo …
    ⊢ LT.lt (MeasureTheory.eLpNorm (↑↑f) p μ) Top.top
  -/
  exact Lp.eLpNorm_lt_top f
  /-
    🎉 no goals
  -/


/-- If `f` belongs to `Lp` for the measure `μ.trim hm`, then it belongs to the subgroup
`lpMeasSubgroup F m p μ`. -/
theorem mem_lpMeasSubgroup_toLp_of_trim (hm : m ≤ m0) (f : Lp F p (μ.trim hm)) :
    (memℒp_of_memℒp_trim hm (Lp.memℒp f)).toLp f ∈ lpMeasSubgroup F m p μ := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    ⊢ Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) (MeasureTheory.Memℒp.t …
  -/
  let hf_mem_ℒp := memℒp_of_memℒp_trim hm (Lp.memℒp f)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    hf_mem_ℒp : MeasureTheory.Memℒp (↑↑f) p μ := MeasureTheory.memℒp_of_memℒp_trim …
    ⊢ Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) (MeasureTheory.Memℒp.t …
  -/
  rw [mem_lpMeasSubgroup_iff_aeStronglyMeasurable']
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    hf_mem_ℒp : MeasureTheory.Memℒp (↑↑f) p μ := MeasureTheory.memℒp_of_memℒp_trim …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.Memℒp.toLp ↑↑f ⋯)) μ
  -/
  refine AEStronglyMeasurable'.congr ?_ (Memℒp.coeFn_toLp hf_mem_ℒp).symm
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    hf_mem_ℒp : MeasureTheory.Memℒp (↑↑f) p μ := MeasureTheory.memℒp_of_memℒp_trim …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
  -/
  refine aeStronglyMeasurable'_of_aeStronglyMeasurable'_trim hm ?_
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    hf_mem_ℒp : MeasureTheory.Memℒp (↑↑f) p μ := MeasureTheory.memℒp_of_memℒp_trim …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑f) (μ.trim hm)
  -/
  exact Lp.aestronglyMeasurable f
  /-
    🎉 no goals
  -/


/-- Map from `lpMeasSubgroup` to `Lp F p (μ.trim hm)`. -/
noncomputable def lpMeasSubgroupToLpTrim (hm : m ≤ m0) (f : lpMeasSubgroup F m p μ) :
    Lp F p (μ.trim hm) :=
  Memℒp.toLp (mem_lpMeasSubgroup_iff_aeStronglyMeasurable'.mp f.mem).choose
    -- Porting note: had to replace `f` with `f.1` here.
    (memℒp_trim_of_mem_lpMeasSubgroup hm f.1 f.mem)


/-- Map from `lpMeas` to `Lp F p (μ.trim hm)`. -/
noncomputable def lpMeasToLpTrim (hm : m ≤ m0) (f : lpMeas F 𝕜 m p μ) : Lp F p (μ.trim hm) :=
  Memℒp.toLp (mem_lpMeas_iff_aeStronglyMeasurable'.mp f.mem).choose
    -- Porting note: had to replace `f` with `f.1` here.
    (memℒp_trim_of_mem_lpMeasSubgroup hm f.1 f.mem)


/-- Map from `Lp F p (μ.trim hm)` to `lpMeasSubgroup`, inverse of
`lpMeasSubgroupToLpTrim`. -/
noncomputable def lpTrimToLpMeasSubgroup (hm : m ≤ m0) (f : Lp F p (μ.trim hm)) :
    lpMeasSubgroup F m p μ :=
  ⟨(memℒp_of_memℒp_trim hm (Lp.memℒp f)).toLp f, mem_lpMeasSubgroup_toLp_of_trim hm f⟩


/-- Map from `Lp F p (μ.trim hm)` to `lpMeas`, inverse of `Lp_meas_to_Lp_trim`. -/
noncomputable def lpTrimToLpMeas (hm : m ≤ m0) (f : Lp F p (μ.trim hm)) : lpMeas F 𝕜 m p μ :=
  ⟨(memℒp_of_memℒp_trim hm (Lp.memℒp f)).toLp f, mem_lpMeasSubgroup_toLp_of_trim hm f⟩


theorem lpMeasSubgroupToLpTrim_ae_eq (hm : m ≤ m0) (f : lpMeasSubgroup F m p μ) :
    lpMeasSubgroupToLpTrim F p μ hm f =ᵐ[μ] f :=
  -- Porting note: replaced `(↑f)` with `f.1` here.
  (ae_eq_of_ae_eq_trim (Memℒp.coeFn_toLp (memℒp_trim_of_mem_lpMeasSubgroup hm f.1 f.mem))).trans
    (mem_lpMeasSubgroup_iff_aeStronglyMeasurable'.mp f.mem).choose_spec.2.symm


theorem lpTrimToLpMeasSubgroup_ae_eq (hm : m ≤ m0) (f : Lp F p (μ.trim hm)) :
    lpTrimToLpMeasSubgroup F p μ hm f =ᵐ[μ] f :=
  -- Porting note: filled in the argument
  Memℒp.coeFn_toLp (memℒp_of_memℒp_trim hm (Lp.memℒp f))


theorem lpMeasToLpTrim_ae_eq (hm : m ≤ m0) (f : lpMeas F 𝕜 m p μ) :
    lpMeasToLpTrim F 𝕜 p μ hm f =ᵐ[μ] f :=
  -- Porting note: replaced `(↑f)` with `f.1` here.
  (ae_eq_of_ae_eq_trim (Memℒp.coeFn_toLp (memℒp_trim_of_mem_lpMeasSubgroup hm f.1 f.mem))).trans
    (mem_lpMeasSubgroup_iff_aeStronglyMeasurable'.mp f.mem).choose_spec.2.symm


theorem lpTrimToLpMeas_ae_eq (hm : m ≤ m0) (f : Lp F p (μ.trim hm)) :
    lpTrimToLpMeas F 𝕜 p μ hm f =ᵐ[μ] f :=
  -- Porting note: filled in the argument
  Memℒp.coeFn_toLp (memℒp_of_memℒp_trim hm (Lp.memℒp f))


/-- `lpTrimToLpMeasSubgroup` is a right inverse of `lpMeasSubgroupToLpTrim`. -/
theorem lpMeasSubgroupToLpTrim_right_inv (hm : m ≤ m0) :
    Function.RightInverse (lpTrimToLpMeasSubgroup F p μ hm) (lpMeasSubgroupToLpTrim F p μ hm) := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    ⊢ Function.RightInverse (MeasureTheory.lpTrimToLpMeasSubgroup F p μ hm) (Measu …
  -/
  intro f
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    ⊢ Eq (MeasureTheory.lpMeasSubgroupToLpTrim F p μ hm (MeasureTheory.lpTrimToLpM …
  -/
  ext1
  refine
    ae_eq_trim_of_stronglyMeasurable hm (Lp.stronglyMeasurable _) (Lp.stronglyMeasurable _) ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.lpMeasSubgroupToLpTrim F  …
  -/
  exact (lpMeasSubgroupToLpTrim_ae_eq hm _).trans (lpTrimToLpMeasSubgroup_ae_eq hm _)
  /-
    🎉 no goals
  -/


/-- `lpTrimToLpMeasSubgroup` is a left inverse of `lpMeasSubgroupToLpTrim`. -/
theorem lpMeasSubgroupToLpTrim_left_inv (hm : m ≤ m0) :
    Function.LeftInverse (lpTrimToLpMeasSubgroup F p μ hm) (lpMeasSubgroupToLpTrim F p μ hm) := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    ⊢ Function.LeftInverse (MeasureTheory.lpTrimToLpMeasSubgroup F p μ hm) (Measur …
  -/
  intro f
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ Eq (MeasureTheory.lpTrimToLpMeasSubgroup F p μ hm (MeasureTheory.lpMeasSubgr …
  -/
  ext1
  /-
    case a
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ Eq ↑(MeasureTheory.lpTrimToLpMeasSubgroup F p μ hm (MeasureTheory.lpMeasSubg …
  -/
  ext1
  /-
    case a.h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(MeasureTheory.lpTrimToLpMeasSubgroup F …
  -/
  rw [← lpMeasSubgroup_coe]
  /-
    case a.h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(MeasureTheory.lpTrimToLpMeasSubgroup F …
  -/
  exact (lpTrimToLpMeasSubgroup_ae_eq hm _).trans (lpMeasSubgroupToLpTrim_ae_eq hm _)
  /-
    🎉 no goals
  -/


theorem lpMeasSubgroupToLpTrim_add (hm : m ≤ m0) (f g : lpMeasSubgroup F m p μ) :
    lpMeasSubgroupToLpTrim F p μ hm (f + g) =
      lpMeasSubgroupToLpTrim F p μ hm f + lpMeasSubgroupToLpTrim F p μ hm g := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ Eq (MeasureTheory.lpMeasSubgroupToLpTrim F p μ hm (HAdd.hAdd f g)) (HAdd.hAd …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq ↑↑(MeasureTheory.lpMeasSubgroupT …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_add _ _).symm
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq (↑↑(MeasureTheory.lpMeasSubgroup …
  -/
  refine ae_eq_trim_of_stronglyMeasurable hm (Lp.stronglyMeasurable _) ?_ ?_
    /-
      case h.refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝ : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
      ⊢ MeasureTheory.StronglyMeasurable (HAdd.hAdd ↑↑(MeasureTheory.lpMeasSubgroupT …
    -/
  · exact (Lp.stronglyMeasurable _).add (Lp.stronglyMeasurable _)
    /-
      🎉 no goals
    -/
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.lpMeasSubgroupToLpTrim F …
  -/
  refine (lpMeasSubgroupToLpTrim_ae_eq hm _).trans ?_
  refine
    EventuallyEq.trans ?_
      (EventuallyEq.add (lpMeasSubgroupToLpTrim_ae_eq hm f).symm
        (lpMeasSubgroupToLpTrim_ae_eq hm g).symm)
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(HAdd.hAdd f g) fun x => HAdd.hAdd (↑↑↑ …
  -/
  refine (Lp.coeFn_add _ _).trans ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd ↑↑↑f ↑↑↑g) fun x => HAdd.hAdd ( …
  -/
  simp_rw [lpMeasSubgroup_coe]
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd ↑↑↑f ↑↑↑g) fun x => HAdd.hAdd ( …
  -/
  filter_upwards with x using rfl
  /-
    🎉 no goals
  -/


theorem lpMeasSubgroupToLpTrim_neg (hm : m ≤ m0) (f : lpMeasSubgroup F m p μ) :
    lpMeasSubgroupToLpTrim F p μ hm (-f) = -lpMeasSubgroupToLpTrim F p μ hm f := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ Eq (MeasureTheory.lpMeasSubgroupToLpTrim F p μ hm (Neg.neg f)) (Neg.neg (Mea …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq ↑↑(MeasureTheory.lpMeasSubgroupT …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_neg _).symm
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq (↑↑(MeasureTheory.lpMeasSubgroup …
  -/
  refine ae_eq_trim_of_stronglyMeasurable hm (Lp.stronglyMeasurable _) ?_ ?_
    /-
      case h.refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝ : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
      ⊢ MeasureTheory.StronglyMeasurable (Neg.neg ↑↑(MeasureTheory.lpMeasSubgroupToL …
    -/
  · exact @StronglyMeasurable.neg _ _ _ m _ _ _ (Lp.stronglyMeasurable _)
    /-
      🎉 no goals
    -/
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.lpMeasSubgroupToLpTrim F …
  -/
  refine (lpMeasSubgroupToLpTrim_ae_eq hm _).trans ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑↑(Neg.neg f)) (Neg.neg ↑↑(MeasureTheory …
  -/
  refine EventuallyEq.trans ?_ (EventuallyEq.neg (lpMeasSubgroupToLpTrim_ae_eq hm f).symm)
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(Neg.neg f) fun x => Neg.neg (↑↑↑f x)
  -/
  refine (Lp.coeFn_neg _).trans ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Neg.neg ↑↑↑f) fun x => Neg.neg (↑↑↑f x)
  -/
  simp_rw [lpMeasSubgroup_coe]
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Neg.neg ↑↑↑f) fun x => Neg.neg (↑↑↑f x)
  -/
  exact Eventually.of_forall fun x => by rfl
  /-
    🎉 no goals
  -/


theorem lpMeasSubgroupToLpTrim_sub (hm : m ≤ m0) (f g : lpMeasSubgroup F m p μ) :
    lpMeasSubgroupToLpTrim F p μ hm (f - g) =
      lpMeasSubgroupToLpTrim F p μ hm f - lpMeasSubgroupToLpTrim F p μ hm g := by
  rw [sub_eq_add_neg, sub_eq_add_neg, lpMeasSubgroupToLpTrim_add,
    lpMeasSubgroupToLpTrim_neg]


theorem lpMeasToLpTrim_smul (hm : m ≤ m0) (c : 𝕜) (f : lpMeas F 𝕜 m p μ) :
    lpMeasToLpTrim F 𝕜 p μ hm (c • f) = c • lpMeasToLpTrim F 𝕜 p μ hm f := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ Eq (MeasureTheory.lpMeasToLpTrim F 𝕜 p μ hm (HSMul.hSMul c f)) (HSMul.hSMul  …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq ↑↑(MeasureTheory.lpMeasToLpTrim  …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_smul _ _).symm
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq (↑↑(MeasureTheory.lpMeasToLpTrim …
  -/
  refine ae_eq_trim_of_stronglyMeasurable hm (Lp.stronglyMeasurable _) ?_ ?_
    /-
      case h.refine_1
      α : Type u_1
      F : Type u_2
      𝕜 : Type u_3
      p : ENNReal
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      c : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
      ⊢ MeasureTheory.StronglyMeasurable (HSMul.hSMul c ↑↑(MeasureTheory.lpMeasToLpT …
    -/
  · exact (Lp.stronglyMeasurable _).const_smul c
    /-
      🎉 no goals
    -/
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.lpMeasToLpTrim F 𝕜 p μ h …
  -/
  refine (lpMeasToLpTrim_ae_eq hm _).trans ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HSMul.hSMul c f)) (HSMul.hSMul c ↑↑(M …
  -/
  refine (Lp.coeFn_smul _ _).trans ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul c ↑↑↑f) (HSMul.hSMul c ↑↑(Mea …
  -/
  refine (lpMeasToLpTrim_ae_eq hm f).mono fun x hx => ?_
  /-
    case h.refine_2
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p μ) x
    x : α
    hx : Eq (↑↑(MeasureTheory.lpMeasToLpTrim F 𝕜 p μ hm f) x) (↑↑↑f x)
    ⊢ Eq (HSMul.hSMul c (↑↑↑f) x) (HSMul.hSMul c (↑↑(MeasureTheory.lpMeasToLpTrim  …
  -/
  simp only [Pi.smul_apply, hx]
  /-
    🎉 no goals
  -/


/-- `lpMeasSubgroupToLpTrim` preserves the norm. -/
theorem lpMeasSubgroupToLpTrim_norm_map [hp : Fact (1 ≤ p)] (hm : m ≤ m0)
    (f : lpMeasSubgroup F m p μ) : ‖lpMeasSubgroupToLpTrim F p μ hm f‖ = ‖f‖ := by
  rw [Lp.norm_def, eLpNorm_trim hm (Lp.stronglyMeasurable _),
    eLpNorm_congr_ae (lpMeasSubgroupToLpTrim_ae_eq hm _), lpMeasSubgroup_coe, ← Lp.norm_def]
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hp : Fact (LE.le 1 p)
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup F m p μ) x
    ⊢ Eq (Norm.norm ↑f) (Norm.norm f)
  -/
  congr
  /-
    🎉 no goals
  -/


theorem isometry_lpMeasSubgroupToLpTrim [hp : Fact (1 ≤ p)] (hm : m ≤ m0) :
    Isometry (lpMeasSubgroupToLpTrim F p μ hm) :=
  Isometry.of_dist_eq fun f g => by
    rw [dist_eq_norm, ← lpMeasSubgroupToLpTrim_sub, lpMeasSubgroupToLpTrim_norm_map,
      dist_eq_norm]


/-- `lpMeasSubgroup` and `Lp F p (μ.trim hm)` are isometric. -/
noncomputable def lpMeasSubgroupToLpTrimIso [Fact (1 ≤ p)] (hm : m ≤ m0) :
    lpMeasSubgroup F m p μ ≃ᵢ Lp F p (μ.trim hm) where
  toFun := lpMeasSubgroupToLpTrim F p μ hm
  invFun := lpTrimToLpMeasSubgroup F p μ hm
  left_inv := lpMeasSubgroupToLpTrim_left_inv hm
  right_inv := lpMeasSubgroupToLpTrim_right_inv hm
  isometry_toFun := isometry_lpMeasSubgroupToLpTrim hm


/-- `lpMeasSubgroup` and `lpMeas` are isometric. -/
noncomputable def lpMeasSubgroupToLpMeasIso [Fact (1 ≤ p)] :
    lpMeasSubgroup F m p μ ≃ᵢ lpMeas F 𝕜 m p μ :=
  IsometryEquiv.refl (lpMeasSubgroup F m p μ)


/-- `lpMeas` and `Lp F p (μ.trim hm)` are isometric, with a linear equivalence. -/
noncomputable def lpMeasToLpTrimLie [Fact (1 ≤ p)] (hm : m ≤ m0) :
    lpMeas F 𝕜 m p μ ≃ₗᵢ[𝕜] Lp F p (μ.trim hm) where
  toFun := lpMeasToLpTrim F 𝕜 p μ hm
  invFun := lpTrimToLpMeas F 𝕜 p μ hm
  left_inv := lpMeasSubgroupToLpTrim_left_inv hm
  right_inv := lpMeasSubgroupToLpTrim_right_inv hm
  map_add' := lpMeasSubgroupToLpTrim_add hm
  map_smul' := lpMeasToLpTrim_smul hm
  norm_map' := lpMeasSubgroupToLpTrim_norm_map hm


instance [hm : Fact (m ≤ m0)] [CompleteSpace F] [hp : Fact (1 ≤ p)] :
    CompleteSpace (lpMeasSubgroup F m p μ) := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : Fact (LE.le m m0)
    inst✝ : CompleteSpace F
    hp : Fact (LE.le 1 p)
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup …
  -/
  rw [(lpMeasSubgroupToLpTrimIso F p μ hm.elim).completeSpace_iff]; infer_instance
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

-- For now just no-lint this; lean4's tree-based logging will make this easier to debug.
-- One possible change might be to generalize `𝕜` from `RCLike` to `NormedField`, as this
-- result may well hold there.
-- Porting note: removed @[nolint fails_quickly]

instance [hm : Fact (m ≤ m0)] [CompleteSpace F] [hp : Fact (1 ≤ p)] :
    CompleteSpace (lpMeas F 𝕜 m p μ) := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    p : ENNReal
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : Fact (LE.le m m0)
    inst✝ : CompleteSpace F
    hp : Fact (LE.le 1 p)
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.lpMeas F 𝕜 m p …
  -/
  rw [(lpMeasSubgroupToLpMeasIso F 𝕜 p μ).symm.completeSpace_iff]; infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem isComplete_aeStronglyMeasurable' [hp : Fact (1 ≤ p)] [CompleteSpace F] (hm : m ≤ m0) :
    IsComplete {f : Lp F p μ | AEStronglyMeasurable' m f μ} := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hp : Fact (LE.le 1 p)
    inst✝ : CompleteSpace F
    hm : LE.le m m0
    ⊢ IsComplete (setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ)
  -/
  rw [← completeSpace_coe_iff_isComplete]
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hp : Fact (LE.le 1 p)
    inst✝ : CompleteSpace F
    hm : LE.le m m0
    ⊢ CompleteSpace ↑(setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ)
  -/
  haveI : Fact (m ≤ m0) := ⟨hm⟩
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hp : Fact (LE.le 1 p)
    inst✝ : CompleteSpace F
    hm : LE.le m m0
    this : Fact (LE.le m m0)
    ⊢ CompleteSpace ↑(setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ)
  -/
  change CompleteSpace (lpMeasSubgroup F m p μ)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hp : Fact (LE.le 1 p)
    inst✝ : CompleteSpace F
    hm : LE.le m m0
    this : Fact (LE.le m m0)
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.lpMeasSubgroup …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem isClosed_aeStronglyMeasurable' [Fact (1 ≤ p)] [CompleteSpace F] (hm : m ≤ m0) :
    IsClosed {f : Lp F p μ | AEStronglyMeasurable' m f μ} :=
  IsComplete.isClosed (isComplete_aeStronglyMeasurable' hm)


/-- We do not get `ae_fin_strongly_measurable f (μ.trim hm)`, since we don't have
`f =ᵐ[μ.trim hm] Lp_meas_to_Lp_trim F 𝕜 p μ hm f` but only the weaker
`f =ᵐ[μ] Lp_meas_to_Lp_trim F 𝕜 p μ hm f`. -/
theorem lpMeas.ae_fin_strongly_measurable' (hm : m ≤ m0) (f : lpMeas F 𝕜 m p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) :
    -- Porting note: changed `f` to `f.1` in the next line. Not certain this is okay.
    ∃ g, FinStronglyMeasurable g (μ.trim hm) ∧ f.1 =ᵐ[μ] g :=
  ⟨lpMeasSubgroupToLpTrim F p μ hm f, Lp.finStronglyMeasurable _ hp_ne_zero hp_ne_top,
    (lpMeasSubgroupToLpTrim_ae_eq hm f).symm⟩


/-- When applying the inverse of `lpMeasToLpTrimLie` (which takes a function in the Lp space of
the sub-sigma algebra and returns its version in the larger Lp space) to an indicator of the
sub-sigma-algebra, we obtain an indicator in the Lp space of the larger sigma-algebra. -/
theorem lpMeasToLpTrimLie_symm_indicator [one_le_p : Fact (1 ≤ p)] [NormedSpace ℝ F] {hm : m ≤ m0}
    {s : Set α} {μ : Measure α} (hs : MeasurableSet[m] s) (hμs : μ.trim hm s ≠ ∞) (c : F) :
    ((lpMeasToLpTrimLie F ℝ p μ hm).symm (indicatorConstLp p hs hμs c) : Lp F p μ) =
      indicatorConstLp p (hm s hs) ((le_trim hm).trans_lt hμs.lt_top).ne c := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    s : Set α
    μ : MeasureTheory.Measure α
    hs : MeasurableSet s
    hμs : Ne ((μ.trim hm) s) Top.top
    c : F
    ⊢ Eq (↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory.in …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    s : Set α
    μ : MeasureTheory.Measure α
    hs : MeasurableSet s
    hμs : Ne ((μ.trim hm) s) Top.top
    c : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.lpMeasToLpTrimLie F Rea …
  -/
  rw [← lpMeas_coe]
  change
    lpTrimToLpMeas F ℝ p μ hm (indicatorConstLp p hs hμs c) =ᵐ[μ]
      (indicatorConstLp p _ _ c : α → F)
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    s : Set α
    μ : MeasureTheory.Measure α
    hs : MeasurableSet s
    hμs : Ne ((μ.trim hm) s) Top.top
    c : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(MeasureTheory.lpTrimToLpMeas F Real p  …
  -/
  refine (lpTrimToLpMeas_ae_eq hm _).trans ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    s : Set α
    μ : MeasureTheory.Measure α
    hs : MeasurableSet s
    hμs : Ne ((μ.trim hm) s) Top.top
    c : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.indicatorConstLp p hs hμs …
  -/
  exact (ae_eq_of_ae_eq_trim indicatorConstLp_coeFn).trans indicatorConstLp_coeFn.symm
  /-
    🎉 no goals
  -/


theorem lpMeasToLpTrimLie_symm_toLp [one_le_p : Fact (1 ≤ p)] [NormedSpace ℝ F] (hm : m ≤ m0)
    (f : α → F) (hf : Memℒp f p (μ.trim hm)) :
    ((lpMeasToLpTrimLie F ℝ p μ hm).symm (hf.toLp f) : Lp F p μ) =
      (memℒp_of_memℒp_trim hm hf).toLp f := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    f : α → F
    hf : MeasureTheory.Memℒp f p (μ.trim hm)
    ⊢ Eq (↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory.Me …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    f : α → F
    hf : MeasureTheory.Memℒp f p (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.lpMeasToLpTrimLie F Rea …
  -/
  rw [← lpMeas_coe]
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    f : α → F
    hf : MeasureTheory.Memℒp f p (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.lpMeasToLpTrimLie F Rea …
  -/
  refine (lpTrimToLpMeas_ae_eq hm _).trans ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    one_le_p : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    f : α → F
    hf : MeasureTheory.Memℒp f p (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp f hf) ↑↑(Measu …
  -/
  exact (ae_eq_of_ae_eq_trim (Memℒp.coeFn_toLp hf)).trans (Memℒp.coeFn_toLp _).symm
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `Lp.induction_stronglyMeasurable`. -/
@[elab_as_elim]
theorem Lp.induction_stronglyMeasurable_aux (hm : m ≤ m0) (hp_ne_top : p ≠ ∞) (P : Lp F p μ → Prop)
    (h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet[m] s) (hμs : μ s < ∞),
      P (Lp.simpleFunc.indicatorConst p (hm s hs) hμs.ne c))
    (h_add : ∀ ⦃f g⦄, ∀ hf : Memℒp f p μ, ∀ hg : Memℒp g p μ, AEStronglyMeasurable' m f μ →
      AEStronglyMeasurable' m g μ → Disjoint (Function.support f) (Function.support g) →
        P (hf.toLp f) → P (hg.toLp g) → P (hf.toLp f + hg.toLp g))
    (h_closed : IsClosed {f : lpMeas F ℝ m p μ | P f}) :
    ∀ f : Lp F p μ, AEStronglyMeasurable' m f μ → P f := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    ⊢ ∀ (f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x), MeasureT …
  -/
  intro f hf
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    ⊢ P f
  -/
  let f' := (⟨f, hf⟩ : lpMeas F ℝ m p μ)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
    ⊢ P f
  -/
  let g := lpMeasToLpTrimLie F ℝ p μ hm f'
  have hfg : f' = (lpMeasToLpTrimLie F ℝ p μ hm).symm g := by
    simp only [f', g, LinearIsometryEquiv.symm_apply_apply]
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
    hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
    ⊢ P f
  -/
  change P ↑f'
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
    hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
    ⊢ P ↑f'
  -/
  rw [hfg]
  refine
    @Lp.induction α F m _ p (μ.trim hm) _ hp_ne_top
      (fun g => P ((lpMeasToLpTrimLie F ℝ p μ hm).symm g)) ?_ ?_ ?_ g
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      ⊢ ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt ((μ.trim hm) s) To …
    -/
  · intro b t ht hμt
    -- Porting note: needed to pass `m` to `Lp.simpleFunc.coe_indicatorConst` to avoid
    -- synthesized type class instance is not definitionally equal to expression inferred by typing
    -- rules, synthesized m0 inferred m
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      b : F
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      ⊢ P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm ↑(MeasureTheory.Lp. …
    -/
    rw [@Lp.simpleFunc.coe_indicatorConst _ _ m, lpMeasToLpTrimLie_symm_indicator ht hμt.ne b]
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      b : F
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      ⊢ P (MeasureTheory.indicatorConstLp p ⋯ ⋯ b)
    -/
    have hμt' : μ t < ∞ := (le_trim hm).trans_lt hμt
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      b : F
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      hμt' : LT.lt (μ t) Top.top
      ⊢ P (MeasureTheory.indicatorConstLp p ⋯ ⋯ b)
    -/
    specialize h_ind b ht hμt'
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      b : F
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      hμt' : LT.lt (μ t) Top.top
      h_ind : P ↑(MeasureTheory.Lp.simpleFunc.indicatorConst p ⋯ ⋯ b)
      ⊢ P (MeasureTheory.indicatorConstLp p ⋯ ⋯ b)
    -/
    rwa [Lp.simpleFunc.coe_indicatorConst] at h_ind
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      ⊢ ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p (μ.trim hm)) (hg : MeasureTheo …
    -/
  · intro f g hf hg h_disj hfP hgP
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := ( …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g✝)
      f g : α → F
      hf : MeasureTheory.Memℒp f p (μ.trim hm)
      hg : MeasureTheory.Memℒp g p (μ.trim hm)
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      hgP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      ⊢ P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (HAdd.hAdd (Measure …
    -/
    rw [LinearIsometryEquiv.map_add]
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := ( …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g✝)
      f g : α → F
      hf : MeasureTheory.Memℒp f p (μ.trim hm)
      hg : MeasureTheory.Memℒp g p (μ.trim hm)
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      hgP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      ⊢ P ↑(HAdd.hAdd ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (Measure …
    -/
    push_cast
    have h_eq :
      ∀ (f : α → F) (hf : Memℒp f p (μ.trim hm)),
        ((lpMeasToLpTrimLie F ℝ p μ hm).symm (Memℒp.toLp f hf) : Lp F p μ) =
          (memℒp_of_memℒp_trim hm hf).toLp f :=
      lpMeasToLpTrimLie_symm_toLp hm
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := ( …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g✝)
      f g : α → F
      hf : MeasureTheory.Memℒp f p (μ.trim hm)
      hg : MeasureTheory.Memℒp g p (μ.trim hm)
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      hgP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      h_eq : ∀ (f : α → F) (hf : MeasureTheory.Memℒp f p (μ.trim hm)), Eq (↑((Measur …
      ⊢ P (HAdd.hAdd ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (Measure …
    -/
    rw [h_eq f hf] at hfP ⊢
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := ( …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g✝)
      f g : α → F
      hf : MeasureTheory.Memℒp f p (μ.trim hm)
      hg : MeasureTheory.Memℒp g p (μ.trim hm)
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P (MeasureTheory.Memℒp.toLp f ⋯)
      hgP : P ↑((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm (MeasureTheory. …
      h_eq : ∀ (f : α → F) (hf : MeasureTheory.Memℒp f p (μ.trim hm)), Eq (↑((Measur …
      ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f ⋯) ↑((MeasureTheory.lpMeasToLpTrimL …
    -/
    rw [h_eq g hg] at hgP ⊢
    exact
      h_add (memℒp_of_memℒp_trim hm hf) (memℒp_of_memℒp_trim hm hg)
        (aeStronglyMeasurable'_of_aeStronglyMeasurable'_trim hm hf.aestronglyMeasurable)
        (aeStronglyMeasurable'_of_aeStronglyMeasurable'_trim hm hg.aestronglyMeasurable)
        h_disj hfP hgP
    /-
      case refine_3
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      ⊢ IsClosed (setOf fun f => (fun g => P ↑((MeasureTheory.lpMeasToLpTrimLie F Re …
    -/
  · change IsClosed ((lpMeasToLpTrimLie F ℝ p μ hm).symm ⁻¹' {g : lpMeas F ℝ m p μ | P ↑g})
    /-
      case refine_3
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
      h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P ↑f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
      f' : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F Real m p μ) x :=  …
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p (μ.trim hm)) x := (M …
      hfg : Eq f' ((MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).symm g)
      ⊢ IsClosed (Set.preimage (⇑(MeasureTheory.lpMeasToLpTrimLie F Real p μ hm).sym …
    -/
    exact IsClosed.preimage (LinearIsometryEquiv.continuous _) h_closed
    /-
      🎉 no goals
    -/


/-- To prove something for an `Lp` function a.e. strongly measurable with respect to a
sub-σ-algebra `m` in a normed space, it suffices to show that
* the property holds for (multiples of) characteristic functions which are measurable w.r.t. `m`;
* is closed under addition;
* the set of functions in `Lp` strongly measurable w.r.t. `m` for which the property holds is
  closed.
-/
@[elab_as_elim]
theorem Lp.induction_stronglyMeasurable (hm : m ≤ m0) (hp_ne_top : p ≠ ∞) (P : Lp F p μ → Prop)
    (h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet[m] s) (hμs : μ s < ∞),
      P (Lp.simpleFunc.indicatorConst p (hm s hs) hμs.ne c))
    (h_add : ∀ ⦃f g⦄, ∀ hf : Memℒp f p μ, ∀ hg : Memℒp g p μ, StronglyMeasurable[m] f →
      StronglyMeasurable[m] g → Disjoint (Function.support f) (Function.support g) →
        P (hf.toLp f) → P (hg.toLp g) → P (hf.toLp f + hg.toLp g))
    (h_closed : IsClosed {f : lpMeas F ℝ m p μ | P f}) :
    ∀ f : Lp F p μ, AEStronglyMeasurable' m f μ → P f := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    ⊢ ∀ (f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x), MeasureT …
  -/
  intro f hf
  suffices h_add_ae :
    ∀ ⦃f g⦄, ∀ hf : Memℒp f p μ, ∀ hg : Memℒp g p μ, AEStronglyMeasurable' m f μ →
      AEStronglyMeasurable' m g μ → Disjoint (Function.support f) (Function.support g) →
        P (hf.toLp f) → P (hg.toLp g) → P (hf.toLp f + hg.toLp g) from
    Lp.induction_stronglyMeasurable_aux hm hp_ne_top _ h_ind h_add_ae h_closed f hf
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf : MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ
    ⊢ ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.Memℒp g …
  -/
  intro f g hf hg hfm hgm h_disj hPf hPg
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  let s_f : Set α := Function.support (hfm.mk f)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hs_f : MeasurableSet[m] s_f := hfm.stronglyMeasurable_mk.measurableSet_support
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hs_f_eq : s_f =ᵐ[μ] Function.support f := hfm.ae_eq_mk.symm.support
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  let s_g : Set α := Function.support (hgm.mk g)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hs_g : MeasurableSet[m] s_g := hgm.stronglyMeasurable_mk.measurableSet_support
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hs_g_eq : s_g =ᵐ[μ] Function.support g := hgm.ae_eq_mk.symm.support
  have h_inter_empty : (s_f ∩ s_g : Set α) =ᵐ[μ] (∅ : Set α) := by
    refine (hs_f_eq.inter hs_g_eq).trans ?_
    suffices Function.support f ∩ Function.support g = ∅ by rw [this]
    exact Set.disjoint_iff_inter_eq_empty.mp h_disj
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  let f' := (s_f \ s_g).indicator (hfm.mk f)
  have hff' : f =ᵐ[μ] f' := by
    have : s_f \ s_g =ᵐ[μ] s_f := by
      rw [← Set.diff_inter_self_eq_diff, Set.inter_comm]
      refine ((ae_eq_refl s_f).diff h_inter_empty).trans ?_
      rw [Set.diff_empty]
    refine ((indicator_ae_eq_of_ae_eq_set this).trans ?_).symm
    rw [Set.indicator_support]
    exact hfm.ae_eq_mk.symm
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hf'_meas : StronglyMeasurable[m] f' := hfm.stronglyMeasurable_mk.indicator (hs_f.diff hs_g)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hf'_Lp : Memℒp f' p μ := hf.ae_eq hff'
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  let g' := (s_g \ s_f).indicator (hgm.mk g)
  have hgg' : g =ᵐ[μ] g' := by
    have : s_g \ s_f =ᵐ[μ] s_g := by
      rw [← Set.diff_inter_self_eq_diff]
      refine ((ae_eq_refl s_g).diff h_inter_empty).trans ?_
      rw [Set.diff_empty]
    refine ((indicator_ae_eq_of_ae_eq_set this).trans ?_).symm
    rw [Set.indicator_support]
    exact hgm.ae_eq_mk.symm
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    g' : α → F := (SDiff.sdiff s_g s_f).indicator (MeasureTheory.AEStronglyMeasura …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hg'_meas : StronglyMeasurable[m] g' := hgm.stronglyMeasurable_mk.indicator (hs_g.diff hs_f)
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    g' : α → F := (SDiff.sdiff s_g s_f).indicator (MeasureTheory.AEStronglyMeasura …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  have hg'_Lp : Memℒp g' p μ := hg.ae_eq hgg'
  have h_disj : Disjoint (Function.support f') (Function.support g') :=
    haveI : Disjoint (s_f \ s_g) (s_g \ s_f) := disjoint_sdiff_sdiff
    this.mono Set.support_indicator_subset Set.support_indicator_subset
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj✝ : Disjoint (Function.support f) (Function.support g)
    hPf : P (MeasureTheory.Memℒp.toLp f hf)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    g' : α → F := (SDiff.sdiff s_g s_f).indicator (MeasureTheory.AEStronglyMeasura …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg'_Lp : MeasureTheory.Memℒp g' p μ
    h_disj : Disjoint (Function.support f') (Function.support g')
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg))
  -/
  rw [← Memℒp.toLp_congr hf'_Lp hf hff'.symm] at hPf ⊢
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj✝ : Disjoint (Function.support f) (Function.support g)
    hPg : P (MeasureTheory.Memℒp.toLp g hg)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    hPf : P (MeasureTheory.Memℒp.toLp f' hf'_Lp)
    g' : α → F := (SDiff.sdiff s_g s_f).indicator (MeasureTheory.AEStronglyMeasura …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg'_Lp : MeasureTheory.Memℒp g' p μ
    h_disj : Disjoint (Function.support f') (Function.support g')
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f' hf'_Lp) (MeasureTheory.Memℒp.toLp  …
  -/
  rw [← Memℒp.toLp_congr hg'_Lp hg hgg'.symm] at hPg ⊢
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x) → Prop
    h_ind : ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P ↑f)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    hf✝ : MeasureTheory.AEStronglyMeasurable' m (↑↑f✝) μ
    f g : α → F
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hgm : MeasureTheory.AEStronglyMeasurable' m g μ
    h_disj✝ : Disjoint (Function.support f) (Function.support g)
    s_f : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk f hfm)
    hs_f : MeasurableSet s_f
    hs_f_eq : (MeasureTheory.ae μ).EventuallyEq s_f (Function.support f)
    s_g : Set α := Function.support (MeasureTheory.AEStronglyMeasurable'.mk g hgm)
    hs_g : MeasurableSet s_g
    hs_g_eq : (MeasureTheory.ae μ).EventuallyEq s_g (Function.support g)
    h_inter_empty : (MeasureTheory.ae μ).EventuallyEq (Inter.inter s_f s_g) EmptyC …
    f' : α → F := (SDiff.sdiff s_f s_g).indicator (MeasureTheory.AEStronglyMeasura …
    hff' : (MeasureTheory.ae μ).EventuallyEq f f'
    hf'_meas : MeasureTheory.StronglyMeasurable f'
    hf'_Lp : MeasureTheory.Memℒp f' p μ
    hPf : P (MeasureTheory.Memℒp.toLp f' hf'_Lp)
    g' : α → F := (SDiff.sdiff s_g s_f).indicator (MeasureTheory.AEStronglyMeasura …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'_meas : MeasureTheory.StronglyMeasurable g'
    hg'_Lp : MeasureTheory.Memℒp g' p μ
    hPg : P (MeasureTheory.Memℒp.toLp g' hg'_Lp)
    h_disj : Disjoint (Function.support f') (Function.support g')
    ⊢ P (HAdd.hAdd (MeasureTheory.Memℒp.toLp f' hf'_Lp) (MeasureTheory.Memℒp.toLp  …
  -/
  exact h_add hf'_Lp hg'_Lp hf'_meas hg'_meas h_disj hPf hPg
  /-
    🎉 no goals
  -/


/-- To prove something for an arbitrary `Memℒp` function a.e. strongly measurable with respect
to a sub-σ-algebra `m` in a normed space, it suffices to show that
* the property holds for (multiples of) characteristic functions which are measurable w.r.t. `m`;
* is closed under addition;
* the set of functions in the `Lᵖ` space strongly measurable w.r.t. `m` for which the property
  holds is closed.
* the property is closed under the almost-everywhere equal relation.
-/
@[elab_as_elim]
theorem Memℒp.induction_stronglyMeasurable (hm : m ≤ m0) (hp_ne_top : p ≠ ∞) (P : (α → F) → Prop)
    (h_ind : ∀ (c : F) ⦃s⦄, MeasurableSet[m] s → μ s < ∞ → P (s.indicator fun _ => c))
    (h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →
      Memℒp f p μ → Memℒp g p μ → StronglyMeasurable[m] f → StronglyMeasurable[m] g →
        P f → P g → P (f + g))
    (h_closed : IsClosed {f : lpMeas F ℝ m p μ | P f})
    (h_ae : ∀ ⦃f g⦄, f =ᵐ[μ] g → Memℒp f p μ → P f → P g) :
    ∀ ⦃f : α → F⦄, Memℒp f p μ → AEStronglyMeasurable' m f μ → P f := by
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    ⊢ ∀ ⦃f : α → F⦄, MeasureTheory.Memℒp f p μ → MeasureTheory.AEStronglyMeasurabl …
  -/
  intro f hf hfm
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    f : α → F
    hf : MeasureTheory.Memℒp f p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    ⊢ P f
  -/
  let f_Lp := hf.toLp f
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    f : α → F
    hf : MeasureTheory.Memℒp f p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
    ⊢ P f
  -/
  have hfm_Lp : AEStronglyMeasurable' m f_Lp μ := hfm.congr hf.coeFn_toLp.symm
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    f : α → F
    hf : MeasureTheory.Memℒp f p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
    hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
    ⊢ P f
  -/
  refine h_ae hf.coeFn_toLp (Lp.memℒp _) ?_
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    f : α → F
    hf : MeasureTheory.Memℒp f p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
    hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
    ⊢ P ↑↑(MeasureTheory.Memℒp.toLp f hf)
  -/
  change P f_Lp
  /-
    α : Type u_1
    F : Type u_2
    p : ENNReal
    inst✝² : NormedAddCommGroup F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Fact (LE.le 1 p)
    inst✝ : NormedSpace Real F
    hm : LE.le m m0
    hp_ne_top : Ne p Top.top
    P : (α → F) → Prop
    h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑↑f)
    h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    f : α → F
    hf : MeasureTheory.Memℒp f p μ
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
    hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
    ⊢ P ↑↑f_Lp
  -/
  refine Lp.induction_stronglyMeasurable hm hp_ne_top (fun f => P f) ?_ ?_ h_closed f_Lp hfm_Lp
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f : α → F
      hf : MeasureTheory.Memℒp f p μ
      hfm : MeasureTheory.AEStronglyMeasurable' m f μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      ⊢ ∀ (c : F) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.top), (f …
    -/
  · intro c s hs hμs
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f : α → F
      hf : MeasureTheory.Memℒp f p μ
      hfm : MeasureTheory.AEStronglyMeasurable' m f μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      c : F
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ P ↑↑↑(MeasureTheory.Lp.simpleFunc.indicatorConst p ⋯ ⋯ c)
    -/
    rw [Lp.simpleFunc.coe_indicatorConst]
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f : α → F
      hf : MeasureTheory.Memℒp f p μ
      hfm : MeasureTheory.AEStronglyMeasurable' m f μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      c : F
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ P ↑↑(MeasureTheory.indicatorConstLp p ⋯ ⋯ c)
    -/
    refine h_ae indicatorConstLp_coeFn.symm ?_ (h_ind c hs hμs)
    /-
      case refine_1
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f : α → F
      hf : MeasureTheory.Memℒp f p μ
      hfm : MeasureTheory.AEStronglyMeasurable' m f μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      c : F
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.Memℒp (s.indicator fun x => c) p μ
    -/
    exact memℒp_indicator_const p (hm s hs) c (Or.inr hμs.ne)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f : α → F
      hf : MeasureTheory.Memℒp f p μ
      hfm : MeasureTheory.AEStronglyMeasurable' m f μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      ⊢ ∀ ⦃f g : α → F⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.Memℒp g …
    -/
  · intro f g hf_mem hg_mem hfm hgm h_disj hfP hgP
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f✝ : α → F
      hf : MeasureTheory.Memℒp f✝ p μ
      hfm✝ : MeasureTheory.AEStronglyMeasurable' m f✝ μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      f g : α → F
      hf_mem : MeasureTheory.Memℒp f p μ
      hg_mem : MeasureTheory.Memℒp g p μ
      hfm : MeasureTheory.StronglyMeasurable f
      hgm : MeasureTheory.StronglyMeasurable g
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑↑(MeasureTheory.Memℒp.toLp f hf_mem)
      hgP : P ↑↑(MeasureTheory.Memℒp.toLp g hg_mem)
      ⊢ P ↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf_mem) (MeasureTheory.Memℒp.toLp …
    -/
    have hfP' : P f := h_ae hf_mem.coeFn_toLp (Lp.memℒp _) hfP
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f✝ : α → F
      hf : MeasureTheory.Memℒp f✝ p μ
      hfm✝ : MeasureTheory.AEStronglyMeasurable' m f✝ μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      f g : α → F
      hf_mem : MeasureTheory.Memℒp f p μ
      hg_mem : MeasureTheory.Memℒp g p μ
      hfm : MeasureTheory.StronglyMeasurable f
      hgm : MeasureTheory.StronglyMeasurable g
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑↑(MeasureTheory.Memℒp.toLp f hf_mem)
      hgP : P ↑↑(MeasureTheory.Memℒp.toLp g hg_mem)
      hfP' : P f
      ⊢ P ↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf_mem) (MeasureTheory.Memℒp.toLp …
    -/
    have hgP' : P g := h_ae hg_mem.coeFn_toLp (Lp.memℒp _) hgP
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_add : ∀ ⦃f g : α → F⦄, Disjoint (Function.support f) (Function.support g) →  …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f✝ : α → F
      hf : MeasureTheory.Memℒp f✝ p μ
      hfm✝ : MeasureTheory.AEStronglyMeasurable' m f✝ μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      f g : α → F
      hf_mem : MeasureTheory.Memℒp f p μ
      hg_mem : MeasureTheory.Memℒp g p μ
      hfm : MeasureTheory.StronglyMeasurable f
      hgm : MeasureTheory.StronglyMeasurable g
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑↑(MeasureTheory.Memℒp.toLp f hf_mem)
      hgP : P ↑↑(MeasureTheory.Memℒp.toLp g hg_mem)
      hfP' : P f
      hgP' : P g
      ⊢ P ↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf_mem) (MeasureTheory.Memℒp.toLp …
    -/
    specialize h_add h_disj hf_mem hg_mem hfm hgm hfP' hgP'
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f✝ : α → F
      hf : MeasureTheory.Memℒp f✝ p μ
      hfm✝ : MeasureTheory.AEStronglyMeasurable' m f✝ μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      f g : α → F
      hf_mem : MeasureTheory.Memℒp f p μ
      hg_mem : MeasureTheory.Memℒp g p μ
      hfm : MeasureTheory.StronglyMeasurable f
      hgm : MeasureTheory.StronglyMeasurable g
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑↑(MeasureTheory.Memℒp.toLp f hf_mem)
      hgP : P ↑↑(MeasureTheory.Memℒp.toLp g hg_mem)
      hfP' : P f
      hgP' : P g
      h_add : P (HAdd.hAdd f g)
      ⊢ P ↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf_mem) (MeasureTheory.Memℒp.toLp …
    -/
    refine h_ae ?_ (hf_mem.add hg_mem) h_add
    /-
      case refine_2
      α : Type u_1
      F : Type u_2
      p : ENNReal
      inst✝² : NormedAddCommGroup F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : Fact (LE.le 1 p)
      inst✝ : NormedSpace Real F
      hm : LE.le m m0
      hp_ne_top : Ne p Top.top
      P : (α → F) → Prop
      h_ind : ∀ (c : F) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
      h_closed : IsClosed (setOf fun f => P ↑↑↑f)
      h_ae : ∀ ⦃f g : α → F⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
      f✝ : α → F
      hf : MeasureTheory.Memℒp f✝ p μ
      hfm✝ : MeasureTheory.AEStronglyMeasurable' m f✝ μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x := MeasureTh …
      hfm_Lp : MeasureTheory.AEStronglyMeasurable' m (↑↑f_Lp) μ
      f g : α → F
      hf_mem : MeasureTheory.Memℒp f p μ
      hg_mem : MeasureTheory.Memℒp g p μ
      hfm : MeasureTheory.StronglyMeasurable f
      hgm : MeasureTheory.StronglyMeasurable g
      h_disj : Disjoint (Function.support f) (Function.support g)
      hfP : P ↑↑(MeasureTheory.Memℒp.toLp f hf_mem)
      hgP : P ↑↑(MeasureTheory.Memℒp.toLp g hg_mem)
      hfP' : P f
      hgP' : P g
      h_add : P (HAdd.hAdd f g)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd f g) ↑↑(HAdd.hAdd (MeasureTheor …
    -/
    exact (hf_mem.coeFn_toLp.symm.add hg_mem.coeFn_toLp.symm).trans (Lp.coeFn_add _ _).symm
    /-
      🎉 no goals
    -/


