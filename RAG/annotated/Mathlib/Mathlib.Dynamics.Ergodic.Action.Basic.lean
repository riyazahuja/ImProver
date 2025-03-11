/--
An additive group action of `G` on a space `α` with measure `μ` is called *ergodic*,
if for any (null) measurable set `s`,
if it is a.e.-invariant under each scalar addition `(g +ᵥ ·)`, `g : G`,
then it is either null or conull.
-/
class ErgodicVAdd (G α : Type*) [VAdd G α] {_ : MeasurableSpace α} (μ : Measure α)
    extends VAddInvariantMeasure G α μ : Prop where
  aeconst_of_forall_preimage_vadd_ae_eq {s : Set α} : MeasurableSet s →
    (∀ g : G, (g +ᵥ ·) ⁻¹' s =ᵐ[μ] s) → EventuallyConst s (ae μ)


/--
A group action of `G` on a space `α` with measure `μ` is called *ergodic*,
if for any (null) measurable set `s`,
if it is a.e.-invariant under each scalar multiplication `(g • ·)`, `g : G`,
then it is either null or conull.
-/
@[to_additive, mk_iff]
class ErgodicSMul (G α : Type*) [SMul G α] {_ : MeasurableSpace α} (μ : Measure α)
    extends SMulInvariantMeasure G α μ : Prop where
  aeconst_of_forall_preimage_smul_ae_eq {s : Set α} : MeasurableSet s →
    (∀ g : G, (g • ·) ⁻¹' s =ᵐ[μ] s) → EventuallyConst s (ae μ)


attribute [to_additive] ergodicSMul_iff


@[to_additive]
theorem aeconst_of_forall_preimage_smul_ae_eq [SMul G α] [ErgodicSMul G α μ] {s : Set α}
    (hm : NullMeasurableSet s μ) (h : ∀ g : G, (g • ·) ⁻¹' s =ᵐ[μ] s) :
    EventuallyConst s (ae μ) := by
  /-
    G : Type u_1
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : SMul G α
    inst✝ : ErgodicSMul G α μ
    s : Set α
    hm : MeasureTheory.NullMeasurableSet s μ
    h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul …
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  rcases hm with ⟨t, htm, hst⟩
  /-
    case intro.intro
    G : Type u_1
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : SMul G α
    inst✝ : ErgodicSMul G α μ
    s : Set α
    h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul …
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  refine .congr ?_ hst.symm
  /-
    case intro.intro
    G : Type u_1
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : SMul G α
    inst✝ : ErgodicSMul G α μ
    s : Set α
    h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul …
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Filter.EventuallyConst t (MeasureTheory.ae μ)
  -/
  refine ErgodicSMul.aeconst_of_forall_preimage_smul_ae_eq htm fun g : G ↦ ?_
  /-
    case intro.intro
    G : Type u_1
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : SMul G α
    inst✝ : ErgodicSMul G α μ
    s : Set α
    h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul …
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    g : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.hSMul g x) t …
  -/
  refine .trans (.trans ?_ (h g)) hst
  /-
    case intro.intro
    G : Type u_1
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : SMul G α
    inst✝ : ErgodicSMul G α μ
    s : Set α
    h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul …
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    g : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.hSMul g x) t …
  -/
  exact tendsto_smul_ae _ _ hst.symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem aeconst_of_forall_smul_ae_eq (hm : NullMeasurableSet s μ) (h : ∀ g : G, g • s =ᵐ[μ] s) :
    EventuallyConst s (ae μ) :=
  aeconst_of_forall_preimage_smul_ae_eq G hm fun g ↦ by
    /-
      G : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : ErgodicSMul G α μ
      s : Set α
      hm : MeasureTheory.NullMeasurableSet s μ
      h : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul g s) s
      g : G
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.hSMul g x) s …
    -/
    simpa only [preimage_smul] using h g⁻¹
    /-
      🎉 no goals
    -/


@[to_additive]
theorem _root_.MulAction.aeconst_of_aestabilizer_eq_top
    (hm : NullMeasurableSet s μ) (h : aestabilizer G μ s = ⊤) : EventuallyConst s (ae μ) :=
  aeconst_of_forall_smul_ae_eq G hm <| (Subgroup.eq_top_iff' _).1 h


theorem _root_.ErgodicSMul.of_aestabilizer [Group G] [MulAction G α] [SMulInvariantMeasure G α μ]
    (h : ∀ s, MeasurableSet s → aestabilizer G μ s = ⊤ → EventuallyConst s (ae μ)) :
    ErgodicSMul G α μ :=
  ⟨fun hm hs ↦ h _ hm <| (Subgroup.eq_top_iff' _).2 fun g ↦ by
    /-
      G : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
      h : ∀ (s : Set α), MeasurableSet s → Eq (MulAction.aestabilizer G μ s) Top.top …
      s✝ : Set α
      hm : MeasurableSet s✝
      hs : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
      g : G
      ⊢ Membership.mem (MulAction.aestabilizer G μ s✝) g
    -/
    simpa only [preimage_smul_inv] using hs g⁻¹⟩
    /-
      🎉 no goals
    -/


theorem ergodicSMul_iterateMulAct {f : α → α} (hf : Measurable f) :
    ErgodicSMul (IterateMulAct f) α μ ↔ Ergodic f μ := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    hf : Measurable f
    ⊢ Iff (ErgodicSMul (IterateMulAct f) α μ) (Ergodic f μ)
  -/
  simp only [ergodicSMul_iff, smulInvariantMeasure_iterateMulAct, hf]
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    hf : Measurable f
    ⊢ Iff (And (MeasureTheory.MeasurePreserving f μ μ) (∀ {s : Set α}, MeasurableS …
  -/
  refine ⟨fun ⟨h₁, h₂⟩ ↦ ⟨h₁, ⟨?_⟩⟩, fun h ↦ ⟨h.1, ?_⟩⟩
    /-
      case refine_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      x✝ : And (MeasureTheory.MeasurePreserving f μ μ) (∀ {s : Set α}, MeasurableSet …
      h₁ : MeasureTheory.MeasurePreserving f μ μ
      h₂ : ∀ {s : Set α}, MeasurableSet s → (∀ (g : IterateMulAct f), (MeasureTheory …
      ⊢ ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (Set.preimage f s) s → Filter.Eventually …
    -/
  · intro s hm hs
    /-
      case refine_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      x✝ : And (MeasureTheory.MeasurePreserving f μ μ) (∀ {s : Set α}, MeasurableSet …
      h₁ : MeasureTheory.MeasurePreserving f μ μ
      h₂ : ∀ {s : Set α}, MeasurableSet s → (∀ (g : IterateMulAct f), (MeasureTheory …
      s : Set α
      hm : MeasurableSet s
      hs : Eq (Set.preimage f s) s
      ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
    -/
    refine h₂ hm fun n ↦ ?_
    /-
      case refine_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      x✝ : And (MeasureTheory.MeasurePreserving f μ μ) (∀ {s : Set α}, MeasurableSet …
      h₁ : MeasureTheory.MeasurePreserving f μ μ
      h₂ : ∀ {s : Set α}, MeasurableSet s → (∀ (g : IterateMulAct f), (MeasureTheory …
      s : Set α
      hm : MeasurableSet s
      hs : Eq (Set.preimage f s) s
      n : IterateMulAct f
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.hSMul n x) s …
    -/
    nth_rewrite 2 [← Function.IsFixedPt.preimage_iterate hs n.val]
    /-
      case refine_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      x✝ : And (MeasureTheory.MeasurePreserving f μ μ) (∀ {s : Set α}, MeasurableSet …
      h₁ : MeasureTheory.MeasurePreserving f μ μ
      h₂ : ∀ {s : Set α}, MeasurableSet s → (∀ (g : IterateMulAct f), (MeasureTheory …
      s : Set α
      hm : MeasurableSet s
      hs : Eq (Set.preimage f s) s
      n : IterateMulAct f
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.hSMul n x) s …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      h : Ergodic f μ
      ⊢ ∀ {s : Set α}, MeasurableSet s → (∀ (g : IterateMulAct f), (MeasureTheory.ae …
    -/
  · intro s hm hs
    /-
      case refine_2
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      hf : Measurable f
      h : Ergodic f μ
      s : Set α
      hm : MeasurableSet s
      hs : ∀ (g : IterateMulAct f), (MeasureTheory.ae μ).EventuallyEq (Set.preimage  …
      ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
    -/
    exact h.quasiErgodic.aeconst_set₀ hm.nullMeasurableSet <| hs (.mk 1)
    /-
      🎉 no goals
    -/


