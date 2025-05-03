/-- If `f = O[l] g` on measurably generated `l`, `f` is strongly measurable at `l`,
and `g` is integrable at `l`, then `f` is integrable at `l`. -/
theorem IsBigO.integrableAtFilter [IsMeasurablyGenerated l]
    (hf : f =O[l] g) (hfm : StronglyMeasurableAtFilter f l μ) (hg : IntegrableAtFilter g l μ) :
    IntegrableAtFilter f l μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    f : α → E
    g : α → F
    l : Filter α
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    inst✝ : l.IsMeasurablyGenerated
    hf : Asymptotics.IsBigO l f g
    hfm : StronglyMeasurableAtFilter f l μ
    hg : MeasureTheory.IntegrableAtFilter g l μ
    ⊢ MeasureTheory.IntegrableAtFilter f l μ
  -/
  obtain ⟨C, hC⟩ := hf.bound
  obtain ⟨s, hsl, hsm, hfg, hf, hg⟩ :=
    (hC.smallSets.and <| hfm.eventually.and hg.eventually).exists_measurable_mem_of_smallSets
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    f : α → E
    g : α → F
    l : Filter α
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    inst✝ : l.IsMeasurablyGenerated
    hf✝ : Asymptotics.IsBigO l f g
    hfm : StronglyMeasurableAtFilter f l μ
    hg✝ : MeasureTheory.IntegrableAtFilter g l μ
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfg : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) (HMul.hMul C (No …
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hg : MeasureTheory.IntegrableOn g s μ
    ⊢ MeasureTheory.IntegrableAtFilter f l μ
  -/
  refine ⟨s, hsl, (hg.norm.const_mul C).mono hf ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    f : α → E
    g : α → F
    l : Filter α
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    inst✝ : l.IsMeasurablyGenerated
    hf✝ : Asymptotics.IsBigO l f g
    hfm : StronglyMeasurableAtFilter f l μ
    hg✝ : MeasureTheory.IntegrableAtFilter g l μ
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfg : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) (HMul.hMul C (No …
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hg : MeasureTheory.IntegrableOn g s μ
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f a)) (Norm.norm (HMul.hMul C  …
  -/
  refine (ae_restrict_mem hsm).mono fun x hx ↦ ?_
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    f : α → E
    g : α → F
    l : Filter α
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    inst✝ : l.IsMeasurablyGenerated
    hf✝ : Asymptotics.IsBigO l f g
    hfm : StronglyMeasurableAtFilter f l μ
    hg✝ : MeasureTheory.IntegrableAtFilter g l μ
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfg : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) (HMul.hMul C (No …
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hg : MeasureTheory.IntegrableOn g s μ
    x : α
    hx : Membership.mem s x
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm (HMul.hMul C (Norm.norm (g x))))
  -/
  exact (hfg x hx).trans (le_abs_self _)
  /-
    🎉 no goals
  -/


/-- Variant of `MeasureTheory.Integrable.mono` taking `f =O[⊤] (g)` instead of `‖f(x)‖ ≤ ‖g(x)‖` -/
theorem IsBigO.integrable (hfm : AEStronglyMeasurable f μ)
    (hf : f =O[⊤] g) (hg : Integrable g μ) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hf : Asymptotics.IsBigO Top.top f g
    hg : MeasureTheory.Integrable g μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  rewrite [← integrableAtFilter_top] at *
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hf : Asymptotics.IsBigO Top.top f g
    hg : MeasureTheory.IntegrableAtFilter g Top.top μ
    ⊢ MeasureTheory.IntegrableAtFilter f Top.top μ
  -/
  exact hf.integrableAtFilter ⟨univ, univ_mem, hfm.restrict⟩ hg
  /-
    🎉 no goals
  -/


/-- Let `f : X x Y → Z`. If as `y` tends to `l`, `f(x, y) = O(g(y))` uniformly on `s : Set X`
of finite measure, then f is eventually (as `y` tends to `l`) integrable along `s`. -/
theorem IsBigO.eventually_integrableOn [Norm F]
    (hf : f =O[𝓟 s ×ˢ l] (g ∘ Prod.snd))
    (hfm : ∀ᶠ x in l, AEStronglyMeasurable (fun i ↦ f (i, x)) (μ.restrict s))
    (hs : MeasurableSet s) (hμ : μ s < ⊤) :
    ∀ᶠ x in l, IntegrableOn (fun i ↦ f (i, x)) s μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    ⊢ Filter.Eventually (fun x => MeasureTheory.IntegrableOn (fun i => f { fst :=  …
  -/
  obtain ⟨C, hC⟩ := hf.bound
  /-
    case intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    ⊢ Filter.Eventually (fun x => MeasureTheory.IntegrableOn (fun i => f { fst :=  …
  -/
  obtain ⟨t, htl, ht⟩ := hC.exists_mem
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    ⊢ Filter.Eventually (fun x => MeasureTheory.IntegrableOn (fun i => f { fst :=  …
  -/
  obtain ⟨u, hu, v, hv, huv⟩ := Filter.mem_prod_iff.mp htl
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    ⊢ Filter.Eventually (fun x => MeasureTheory.IntegrableOn (fun i => f { fst :=  …
  -/
  obtain ⟨w, hwl, hw⟩ := hfm.exists_mem
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    ⊢ Filter.Eventually (fun x => MeasureTheory.IntegrableOn (fun i => f { fst :=  …
  -/
  refine eventually_iff_exists_mem.mpr ⟨w ∩ v, inter_mem hwl hv, fun x hx ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    x : α
    hx : Membership.mem (Inter.inter w v) x
    ⊢ MeasureTheory.IntegrableOn (fun i => f { fst := i, snd := x }) s μ
  -/
  haveI : IsFiniteMeasure (μ.restrict s) := ⟨Measure.restrict_apply_univ s ▸ hμ⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    x : α
    hx : Membership.mem (Inter.inter w v) x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ MeasureTheory.IntegrableOn (fun i => f { fst := i, snd := x }) s μ
  -/
  refine Integrable.mono' (integrable_const (C * ‖g x‖)) (hw x hx.1) ?_
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    x : α
    hx : Membership.mem (Inter.inter w v) x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f { fst := a, snd := x })) (HM …
  -/
  filter_upwards [MeasureTheory.self_mem_ae_restrict hs]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    x : α
    hx : Membership.mem (Inter.inter w v) x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ ∀ (a : ι), Membership.mem s a → LE.le (Norm.norm (f { fst := a, snd := x })) …
  -/
  intro y hy
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝¹ : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝ : Norm F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hfm : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun i => …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    w : Set α
    hwl : Membership.mem l w
    hw : ∀ (y : α), Membership.mem w y → MeasureTheory.AEStronglyMeasurable (fun i …
    x : α
    hx : Membership.mem (Inter.inter w v) x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    y : ι
    hy : Membership.mem s y
    ⊢ LE.le (Norm.norm (f { fst := y, snd := x })) (HMul.hMul C (Norm.norm (g x)))
  -/
  exact ht (y, x) <| huv ⟨hu hy, hx.2⟩
  /-
    🎉 no goals
  -/


/-- Let `f : X x Y → Z`. If as `y` tends to `l`, `f(x, y) = O(g(y))` uniformly on `s : Set X`
of finite measure, then the integral of `f` along `s` is `O(g(y))`. -/
theorem IsBigO.set_integral_isBigO
    (hf : f =O[𝓟 s ×ˢ l] (g ∘ Prod.snd)) (hs : MeasurableSet s) (hμ : μ s < ⊤)  :
    (fun x ↦ ∫ i in s, f (i, x) ∂μ) =O[l] g := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    ⊢ Asymptotics.IsBigO l (fun x => MeasureTheory.integral (μ.restrict s) fun i = …
  -/
  obtain ⟨C, hC⟩ := hf.bound
  /-
    case intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    ⊢ Asymptotics.IsBigO l (fun x => MeasureTheory.integral (μ.restrict s) fun i = …
  -/
  obtain ⟨t, htl, ht⟩ := hC.exists_mem
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    ⊢ Asymptotics.IsBigO l (fun x => MeasureTheory.integral (μ.restrict s) fun i = …
  -/
  obtain ⟨u, hu, v, hv, huv⟩ := Filter.mem_prod_iff.mp htl
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    ⊢ Asymptotics.IsBigO l (fun x => MeasureTheory.integral (μ.restrict s) fun i = …
  -/
  refine isBigO_iff.mpr ⟨C * (μ s).toReal, eventually_iff_exists_mem.mpr ⟨v, hv, fun x hx ↦ ?_⟩⟩
  rw [mul_assoc, ← smul_eq_mul (a' := ‖g x‖), ← MeasureTheory.Measure.restrict_apply_univ,
    ← integral_const, mul_comm, ← smul_eq_mul, ← integral_smul_const]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    x : α
    hx : Membership.mem v x
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun i => f { fst :=  …
  -/
  haveI : IsFiniteMeasure (μ.restrict s) := ⟨by rw [Measure.restrict_apply_univ s]; exact hμ⟩
  refine (norm_integral_le_integral_norm _).trans <|
    integral_mono_of_nonneg (univ_mem' fun _ ↦ norm_nonneg _) (integrable_const _) ?_
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    x : α
    hx : Membership.mem v x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE (fun a => Norm.norm (f { fst  …
  -/
  filter_upwards [MeasureTheory.self_mem_ae_restrict hs]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    x : α
    hx : Membership.mem v x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ ∀ (a : ι), Membership.mem s a → LE.le (Norm.norm (f { fst := a, snd := x })) …
  -/
  intro y hy
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    x : α
    hx : Membership.mem v x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    y : ι
    hy : Membership.mem s y
    ⊢ LE.le (Norm.norm (f { fst := y, snd := x })) (HSMul.hSMul (Norm.norm (g x)) C)
  -/
  rw [smul_eq_mul, mul_comm]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    g : α → F
    l : Filter α
    ι : Type u_4
    inst✝² : MeasurableSpace ι
    f : Prod ι α → E
    s : Set ι
    μ : MeasureTheory.Measure ι
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedAddCommGroup F
    hf : Asymptotics.IsBigO (SProd.sprod (Filter.principal s) l) f (Function.comp  …
    hs : MeasurableSet s
    hμ : LT.lt (μ s) Top.top
    C : Real
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.no …
    t : Set (Prod ι α)
    htl : Membership.mem (SProd.sprod (Filter.principal s) l) t
    ht : ∀ (y : Prod ι α), Membership.mem t y → LE.le (Norm.norm (f y)) (HMul.hMul …
    u : Set ι
    hu : Membership.mem (Filter.principal s) u
    v : Set α
    hv : Membership.mem l v
    huv : HasSubset.Subset (SProd.sprod u v) t
    x : α
    hx : Membership.mem v x
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    y : ι
    hy : Membership.mem s y
    ⊢ LE.le (Norm.norm (f { fst := y, snd := x })) (HMul.hMul C (Norm.norm (g x)))
  -/
  exact ht (y, x) <| huv ⟨hu hy, hx⟩
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable, and `f =O[cocompact] g` for some `g` integrable at `cocompact`,
then `f` is integrable. -/
theorem LocallyIntegrable.integrable_of_isBigO_cocompact [IsMeasurablyGenerated (cocompact α)]
    (hf : LocallyIntegrable f μ) (ho : f =O[cocompact α] g)
    (hg : IntegrableAtFilter g (cocompact α) μ) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁴ : TopologicalSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : (Filter.cocompact α).IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO (Filter.cocompact α) f g
    hg : MeasureTheory.IntegrableAtFilter g (Filter.cocompact α) μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine integrable_iff_integrableAtFilter_cocompact.mpr ⟨ho.integrableAtFilter ?_ hg, hf⟩
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁴ : TopologicalSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : (Filter.cocompact α).IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO (Filter.cocompact α) f g
    hg : MeasureTheory.IntegrableAtFilter g (Filter.cocompact α) μ
    ⊢ StronglyMeasurableAtFilter f (Filter.cocompact α) μ
  -/
  exact hf.aestronglyMeasurable.stronglyMeasurableAtFilter
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable, and `f =O[atBot] g`, `f =O[atTop] g'` for some
`g`, `g'` integrable at `atBot` and `atTop` respectively, then `f` is integrable. -/
theorem LocallyIntegrable.integrable_of_isBigO_atBot_atTop
    [IsMeasurablyGenerated (atBot (α := α))] [IsMeasurablyGenerated (atTop (α := α))]
    (hf : LocallyIntegrable f μ)
    (ho : f =O[atBot] g) (hg : IntegrableAtFilter g atBot μ)
    (ho' : f =O[atTop] g') (hg' : IntegrableAtFilter g' atTop μ) : Integrable f μ := by
  refine integrable_iff_integrableAtFilter_atBot_atTop.mpr
    ⟨⟨ho.integrableAtFilter ?_ hg, ho'.integrableAtFilter ?_ hg'⟩, hf⟩
  /-
    case refine_1
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : SecondCountableTopology α
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : LinearOrder α
    inst✝² : CompactIccSpace α
    g' : α → F
    inst✝¹ : Filter.atBot.IsMeasurablyGenerated
    inst✝ : Filter.atTop.IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO Filter.atBot f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atBot μ
    ho' : Asymptotics.IsBigO Filter.atTop f g'
    hg' : MeasureTheory.IntegrableAtFilter g' Filter.atTop μ
    ⊢ StronglyMeasurableAtFilter f Filter.atBot μ
  -/
  all_goals exact hf.aestronglyMeasurable.stronglyMeasurableAtFilter
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable on `(∞, a]`, and `f =O[atBot] g`, for some
`g` integrable at `atBot`, then `f` is integrable on `(∞, a]`. -/
theorem LocallyIntegrableOn.integrableOn_of_isBigO_atBot [IsMeasurablyGenerated (atBot (α := α))]
    (hf : LocallyIntegrableOn f (Iic a) μ) (ho : f =O[atBot] g)
    (hg : IntegrableAtFilter g atBot μ) : IntegrableOn f (Iic a) μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    f : α → E
    g : α → F
    a : α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    inst✝² : LinearOrder α
    inst✝¹ : CompactIccSpace α
    inst✝ : Filter.atBot.IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    ho : Asymptotics.IsBigO Filter.atBot f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atBot μ
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic a) μ
  -/
  refine integrableOn_Iic_iff_integrableAtFilter_atBot.mpr ⟨ho.integrableAtFilter ?_ hg, hf⟩
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    f : α → E
    g : α → F
    a : α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    inst✝² : LinearOrder α
    inst✝¹ : CompactIccSpace α
    inst✝ : Filter.atBot.IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    ho : Asymptotics.IsBigO Filter.atBot f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atBot μ
    ⊢ StronglyMeasurableAtFilter f Filter.atBot μ
  -/
  exact ⟨Iic a, Iic_mem_atBot a, hf.aestronglyMeasurable⟩
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable on `[a, ∞)`, and `f =O[atTop] g`, for some
`g` integrable at `atTop`, then `f` is integrable on `[a, ∞)`. -/
theorem LocallyIntegrableOn.integrableOn_of_isBigO_atTop [IsMeasurablyGenerated (atTop (α := α))]
    (hf : LocallyIntegrableOn f (Ici a) μ) (ho : f =O[atTop] g)
    (hg : IntegrableAtFilter g atTop μ) : IntegrableOn f (Ici a) μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    f : α → E
    g : α → F
    a : α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    inst✝² : LinearOrder α
    inst✝¹ : CompactIccSpace α
    inst✝ : Filter.atTop.IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrableOn f (Set.Ici a) μ
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    ⊢ MeasureTheory.IntegrableOn f (Set.Ici a) μ
  -/
  refine integrableOn_Ici_iff_integrableAtFilter_atTop.mpr ⟨ho.integrableAtFilter ?_ hg, hf⟩
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    f : α → E
    g : α → F
    a : α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    inst✝² : LinearOrder α
    inst✝¹ : CompactIccSpace α
    inst✝ : Filter.atTop.IsMeasurablyGenerated
    hf : MeasureTheory.LocallyIntegrableOn f (Set.Ici a) μ
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    ⊢ StronglyMeasurableAtFilter f Filter.atTop μ
  -/
  exact ⟨Ici a, Ici_mem_atTop a, hf.aestronglyMeasurable⟩
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable, `f` has a top element, and `f =O[atBot] g`, for some
`g` integrable at `atBot`, then `f` is integrable. -/
theorem LocallyIntegrable.integrable_of_isBigO_atBot [IsMeasurablyGenerated (atBot (α := α))]
    [OrderTop α] (hf : LocallyIntegrable f μ) (ho : f =O[atBot] g)
    (hg : IntegrableAtFilter g atBot μ) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : SecondCountableTopology α
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : LinearOrder α
    inst✝² : CompactIccSpace α
    inst✝¹ : Filter.atBot.IsMeasurablyGenerated
    inst✝ : OrderTop α
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO Filter.atBot f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atBot μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine integrable_iff_integrableAtFilter_atBot.mpr ⟨ho.integrableAtFilter ?_ hg, hf⟩
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : SecondCountableTopology α
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : LinearOrder α
    inst✝² : CompactIccSpace α
    inst✝¹ : Filter.atBot.IsMeasurablyGenerated
    inst✝ : OrderTop α
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO Filter.atBot f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atBot μ
    ⊢ StronglyMeasurableAtFilter f Filter.atBot μ
  -/
  exact hf.aestronglyMeasurable.stronglyMeasurableAtFilter
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable, `f` has a bottom element, and `f =O[atTop] g`, for some
`g` integrable at `atTop`, then `f` is integrable. -/
theorem LocallyIntegrable.integrable_of_isBigO_atTop [IsMeasurablyGenerated (atTop (α := α))]
    [OrderBot α] (hf : LocallyIntegrable f μ) (ho : f =O[atTop] g)
    (hg : IntegrableAtFilter g atTop μ) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : SecondCountableTopology α
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : LinearOrder α
    inst✝² : CompactIccSpace α
    inst✝¹ : Filter.atTop.IsMeasurablyGenerated
    inst✝ : OrderBot α
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine integrable_iff_integrableAtFilter_atTop.mpr ⟨ho.integrableAtFilter ?_ hg, hf⟩
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : SecondCountableTopology α
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : LinearOrder α
    inst✝² : CompactIccSpace α
    inst✝¹ : Filter.atTop.IsMeasurablyGenerated
    inst✝ : OrderBot α
    hf : MeasureTheory.LocallyIntegrable f μ
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    ⊢ StronglyMeasurableAtFilter f Filter.atTop μ
  -/
  exact hf.aestronglyMeasurable.stronglyMeasurableAtFilter
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable, `‖f(-x)‖ = ‖f(x)‖`, and `f =O[atTop] g`, for some
`g` integrable at `atTop`, then `f` is integrable. -/
theorem LocallyIntegrable.integrable_of_isBigO_atTop_of_norm_isNegInvariant
    [IsMeasurablyGenerated (atTop (α := α))] [MeasurableNeg α] [μ.IsNegInvariant]
    (hf : LocallyIntegrable f μ) (hsymm : norm ∘ f =ᵐ[μ] norm ∘ f ∘ Neg.neg) (ho : f =O[atTop] g)
    (hg : IntegrableAtFilter g atTop μ) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  have h_int := (hf.locallyIntegrableOn (Ici 0)).integrableOn_of_isBigO_atTop ho hg
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  rw [← integrableOn_univ, ← Iic_union_Ici_of_le le_rfl, integrableOn_union]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    ⊢ And (MeasureTheory.IntegrableOn f (Set.Iic ?m.44918) μ) (MeasureTheory.Integ …
  -/
  refine ⟨?_, h_int⟩
  have h_map_neg : (μ.restrict (Ici 0)).map Neg.neg = μ.restrict (Iic 0) := by
    conv => rhs; rw [← Measure.map_neg_eq_self μ, measurableEmbedding_neg.restrict_map]
    simp
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    h_map_neg : Eq (MeasureTheory.Measure.map Neg.neg (μ.restrict (Set.Ici 0))) (μ …
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic 0) μ
  -/
  rw [IntegrableOn, ← h_map_neg, measurableEmbedding_neg.integrable_map_iff]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    h_map_neg : Eq (MeasureTheory.Measure.map Neg.neg (μ.restrict (Set.Ici 0))) (μ …
    ⊢ MeasureTheory.Integrable (Function.comp f Neg.neg) (μ.restrict (Set.Ici 0))
  -/
  refine h_int.congr' ?_ hsymm.restrict
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    h_map_neg : Eq (MeasureTheory.Measure.map Neg.neg (μ.restrict (Set.Ici 0))) (μ …
    ⊢ MeasureTheory.AEStronglyMeasurable (Function.comp f Neg.neg) (μ.restrict (Se …
  -/
  refine AEStronglyMeasurable.comp_aemeasurable ?_ measurable_neg.aemeasurable
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    f : α → E
    g : α → F
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : SecondCountableTopology α
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : LinearOrderedAddCommGroup α
    inst✝³ : CompactIccSpace α
    inst✝² : Filter.atTop.IsMeasurablyGenerated
    inst✝¹ : MeasurableNeg α
    inst✝ : μ.IsNegInvariant
    hf : MeasureTheory.LocallyIntegrable f μ
    hsymm : (MeasureTheory.ae μ).EventuallyEq (Function.comp Norm.norm f) (Functio …
    ho : Asymptotics.IsBigO Filter.atTop f g
    hg : MeasureTheory.IntegrableAtFilter g Filter.atTop μ
    h_int : MeasureTheory.IntegrableOn f (Set.Ici 0) μ
    h_map_neg : Eq (MeasureTheory.Measure.map Neg.neg (μ.restrict (Set.Ici 0))) (μ …
    ⊢ MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map Neg.neg (μ.r …
  -/
  exact h_map_neg ▸ hf.aestronglyMeasurable.restrict
  /-
    🎉 no goals
  -/


