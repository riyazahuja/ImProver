/-- This is an auxiliary lemma for `measurable_kernel_prod_mk_left`. -/
theorem measurable_kernel_prod_mk_left_of_finite {t : Set (α × β)} (ht : MeasurableSet t)
    (hκs : ∀ a, IsFiniteMeasure (κ a)) : Measurable fun a => κ a (Prod.mk a ⁻¹' t) := by
  -- `t` is a measurable set in the product `α × β`: we use that the product σ-algebra is generated
  -- by boxes to prove the result by induction.
  induction t, ht
    using MeasurableSpace.induction_on_inter generateFrom_prod.symm isPiSystem_prod with
  | empty =>
    simp only [preimage_empty, measure_empty, measurable_const]
  | basic t ht =>
    simp only [Set.mem_image2, Set.mem_setOf_eq] at ht
    obtain ⟨t₁, ht₁, t₂, ht₂, rfl⟩ := ht
    classical
    simp_rw [mk_preimage_prod_right_eq_if]
    have h_eq_ite : (fun a => κ a (ite (a ∈ t₁) t₂ ∅)) = fun a => ite (a ∈ t₁) (κ a t₂) 0 := by
      ext1 a
      split_ifs
      exacts [rfl, measure_empty]
    rw [h_eq_ite]
    exact Measurable.ite ht₁ (Kernel.measurable_coe κ ht₂) measurable_const
  | compl t htm iht =>
    have h_eq_sdiff : ∀ a, Prod.mk a ⁻¹' tᶜ = Set.univ \ Prod.mk a ⁻¹' t := by
      intro a
      ext1 b
      simp only [mem_compl_iff, mem_preimage, mem_diff, mem_univ, true_and]
    simp_rw [h_eq_sdiff]
    have :
      (fun a => κ a (Set.univ \ Prod.mk a ⁻¹' t)) = fun a =>
        κ a Set.univ - κ a (Prod.mk a ⁻¹' t) := by
      ext1 a
      rw [← Set.diff_inter_self_eq_diff, Set.inter_univ, measure_diff (Set.subset_univ _)]
      · exact (measurable_prod_mk_left htm).nullMeasurableSet
      · exact measure_ne_top _ _
    rw [this]
    exact Measurable.sub (Kernel.measurable_coe κ MeasurableSet.univ) iht
  | iUnion f h_disj hf_meas hf =>
    have (a : α) : κ a (Prod.mk a ⁻¹' ⋃ i, f i) = ∑' i, κ a (Prod.mk a ⁻¹' f i) := by
      rw [preimage_iUnion, measure_iUnion]
      · exact h_disj.mono fun _ _ ↦ .preimage _
      · exact fun i ↦ measurable_prod_mk_left (hf_meas i)
    simpa only [this] using Measurable.ennreal_tsum hf


theorem measurable_kernel_prod_mk_left [IsSFiniteKernel κ] {t : Set (α × β)}
    (ht : MeasurableSet t) : Measurable fun a => κ a (Prod.mk a ⁻¹' t) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    t : Set (Prod α β)
    ht : MeasurableSet t
    ⊢ Measurable fun a => (κ a) (Set.preimage (Prod.mk a) t)
  -/
  rw [← Kernel.kernel_sum_seq κ]
  have : ∀ a, Kernel.sum (Kernel.seq κ) a (Prod.mk a ⁻¹' t) =
      ∑' n, Kernel.seq κ n a (Prod.mk a ⁻¹' t) := fun a =>
    Kernel.sum_apply' _ _ (measurable_prod_mk_left ht)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    t : Set (Prod α β)
    ht : MeasurableSet t
    this : ∀ (a : α), Eq (((ProbabilityTheory.Kernel.sum κ.seq) a) (Set.preimage ( …
    ⊢ Measurable fun a => ((ProbabilityTheory.Kernel.sum κ.seq) a) (Set.preimage ( …
  -/
  simp_rw [this]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    t : Set (Prod α β)
    ht : MeasurableSet t
    this : ∀ (a : α), Eq (((ProbabilityTheory.Kernel.sum κ.seq) a) (Set.preimage ( …
    ⊢ Measurable fun a => tsum fun n => ((κ.seq n) a) (Set.preimage (Prod.mk a) t)
  -/
  refine Measurable.ennreal_tsum fun n => ?_
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    t : Set (Prod α β)
    ht : MeasurableSet t
    this : ∀ (a : α), Eq (((ProbabilityTheory.Kernel.sum κ.seq) a) (Set.preimage ( …
    n : Nat
    ⊢ Measurable fun a => ((κ.seq n) a) (Set.preimage (Prod.mk a) t)
  -/
  exact measurable_kernel_prod_mk_left_of_finite ht inferInstance
  /-
    🎉 no goals
  -/


theorem measurable_kernel_prod_mk_left' [IsSFiniteKernel η] {s : Set (β × γ)} (hs : MeasurableSet s)
    (a : α) : Measurable fun b => η (a, b) (Prod.mk b ⁻¹' s) := by
  have : ∀ b, Prod.mk b ⁻¹' s = {c | ((a, b), c) ∈ {p : (α × β) × γ | (p.1.2, p.2) ∈ s}} := by
    intro b; rfl
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set (Prod β γ)
    hs : MeasurableSet s
    a : α
    this : ∀ (b : β), Eq (Set.preimage (Prod.mk b) s) (setOf fun c => Membership.m …
    ⊢ Measurable fun b => (η { fst := a, snd := b }) (Set.preimage (Prod.mk b) s)
  -/
  simp_rw [this]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set (Prod β γ)
    hs : MeasurableSet s
    a : α
    this : ∀ (b : β), Eq (Set.preimage (Prod.mk b) s) (setOf fun c => Membership.m …
    ⊢ Measurable fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.me …
  -/
  refine (measurable_kernel_prod_mk_left ?_).comp measurable_prod_mk_left
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set (Prod β γ)
    hs : MeasurableSet s
    a : α
    this : ∀ (b : β), Eq (Set.preimage (Prod.mk b) s) (setOf fun c => Membership.m …
    ⊢ MeasurableSet (setOf fun p => Membership.mem s { fst := p.1.2, snd := p.2 })
  -/
  exact (measurable_fst.snd.prod_mk measurable_snd) hs
  /-
    🎉 no goals
  -/


theorem measurable_kernel_prod_mk_right [IsSFiniteKernel κ] {s : Set (β × α)}
    (hs : MeasurableSet s) : Measurable fun y => κ y ((fun x => (x, y)) ⁻¹' s) :=
  measurable_kernel_prod_mk_left (measurableSet_swap_iff.mpr hs)


/-- Auxiliary lemma for `Measurable.lintegral_kernel_prod_right`. -/
theorem Kernel.measurable_lintegral_indicator_const {t : Set (α × β)} (ht : MeasurableSet t)
    (c : ℝ≥0∞) : Measurable fun a => ∫⁻ b, t.indicator (Function.const (α × β) c) (a, b) ∂κ a := by
  -- Porting note: was originally by
  -- `simp_rw [lintegral_indicator_const_comp measurable_prod_mk_left ht _]`
  -- but this has no effect, so added the `conv` below
  conv =>
    congr
    ext
    erw [lintegral_indicator_const_comp measurable_prod_mk_left ht _]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    t : Set (Prod α β)
    ht : MeasurableSet t
    c : ENNReal
    ⊢ Measurable fun x => HMul.hMul c ((κ x) (Set.preimage (Prod.mk x) t))
  -/
  exact Measurable.const_mul (measurable_kernel_prod_mk_left ht) c
  /-
    🎉 no goals
  -/


/-- For an s-finite kernel `κ` and a function `f : α → β → ℝ≥0∞` which is measurable when seen as a
map from `α × β` (hypothesis `Measurable (uncurry f)`), the integral `a ↦ ∫⁻ b, f a b ∂(κ a)` is
measurable. -/
theorem _root_.Measurable.lintegral_kernel_prod_right {f : α → β → ℝ≥0∞}
    (hf : Measurable (uncurry f)) : Measurable fun a => ∫⁻ b, f a b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => f a b
  -/
  let F : ℕ → SimpleFunc (α × β) ℝ≥0∞ := SimpleFunc.eapprox (uncurry f)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => f a b
  -/
  have h : ∀ a, ⨆ n, F n a = uncurry f a := SimpleFunc.iSup_eapprox_apply hf
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : Prod α β), Eq (iSup fun n => (F n) a) (Function.uncurry f a)
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => f a b
  -/
  simp only [Prod.forall, uncurry_apply_pair] at h
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => f a b
  -/
  simp_rw [← h]
  have : ∀ a, (∫⁻ b, ⨆ n, F n (a, b) ∂κ a) = ⨆ n, ∫⁻ b, F n (a, b) ∂κ a := by
    intro a
    rw [lintegral_iSup]
    · exact fun n => (F n).measurable.comp measurable_prod_mk_left
    · exact fun i j hij b => SimpleFunc.monotone_eapprox (uncurry f) hij _
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
  -/
  simp_rw [this]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
    ⊢ Measurable fun a => iSup fun n => MeasureTheory.lintegral (κ a) fun b => (F  …
  -/
  refine .iSup fun n => ?_
  refine SimpleFunc.induction
    (P := fun f => Measurable (fun (a : α) => ∫⁻ (b : β), f (a, b) ∂κ a)) ?_ ?_ (F n)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      ⊢ ∀ (c : ENNReal) {s : Set (Prod α β)} (hs : MeasurableSet s), (fun f => Measu …
    -/
  · intro c t ht
    simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_const,
      SimpleFunc.coe_zero, Set.piecewise_eq_indicator]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      c : ENNReal
      t : Set (Prod α β)
      ht : MeasurableSet t
      ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => t.indicator (Func …
    -/
    exact Kernel.measurable_lintegral_indicator_const (κ := κ) ht c
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      ⊢ ∀ ⦃f g : MeasureTheory.SimpleFunc (Prod α β) ENNReal⦄, Disjoint (Function.su …
    -/
  · intro g₁ g₂ _ hm₁ hm₂
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      g₁ g₂ : MeasureTheory.SimpleFunc (Prod α β) ENNReal
      a✝ : Disjoint (Function.support ⇑g₁) (Function.support ⇑g₂)
      hm₁ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₁ { fst := a …
      hm₂ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₂ { fst := a …
      ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => (HAdd.hAdd g₁ g₂) …
    -/
    simp only [SimpleFunc.coe_add, Pi.add_apply]
    have h_add :
      (fun a => ∫⁻ b, g₁ (a, b) + g₂ (a, b) ∂κ a) =
        (fun a => ∫⁻ b, g₁ (a, b) ∂κ a) + fun a => ∫⁻ b, g₂ (a, b) ∂κ a := by
      ext1 a
      rw [Pi.add_apply]
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): was `rw` (`Function.comp` reducibility)
      erw [lintegral_add_left (g₁.measurable.comp measurable_prod_mk_left)]
      simp_rw [Function.comp_apply]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      g₁ g₂ : MeasureTheory.SimpleFunc (Prod α β) ENNReal
      a✝ : Disjoint (Function.support ⇑g₁) (Function.support ⇑g₂)
      hm₁ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₁ { fst := a …
      hm₂ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₂ { fst := a …
      h_add : Eq (fun a => MeasureTheory.lintegral (κ a) fun b => HAdd.hAdd (g₁ { fs …
      ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => HAdd.hAdd (g₁ { f …
    -/
    rw [h_add]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod α β) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : α) (b : β), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      this : ∀ (a : α), Eq (MeasureTheory.lintegral (κ a) fun b => iSup fun n => (F  …
      n : Nat
      g₁ g₂ : MeasureTheory.SimpleFunc (Prod α β) ENNReal
      a✝ : Disjoint (Function.support ⇑g₁) (Function.support ⇑g₂)
      hm₁ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₁ { fst := a …
      hm₂ : Measurable fun a => MeasureTheory.lintegral (κ a) fun b => g₂ { fst := a …
      h_add : Eq (fun a => MeasureTheory.lintegral (κ a) fun b => HAdd.hAdd (g₁ { fs …
      ⊢ Measurable (HAdd.hAdd (fun a => MeasureTheory.lintegral (κ a) fun b => g₁ {  …
    -/
    exact Measurable.add hm₁ hm₂
    /-
      🎉 no goals
    -/


theorem _root_.Measurable.lintegral_kernel_prod_right' {f : α × β → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun a => ∫⁻ b, f (a, b) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => f { fst := a, snd …
  -/
  refine Measurable.lintegral_kernel_prod_right ?_
  have : (uncurry fun (a : α) (b : β) => f (a, b)) = f := by
    ext x; rw [uncurry_apply_pair]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    this : Eq (Function.uncurry fun a b => f { fst := a, snd := b }) f
    ⊢ Measurable (Function.uncurry fun a b => f { fst := a, snd := b })
  -/
  rwa [this]
  /-
    🎉 no goals
  -/


theorem _root_.Measurable.lintegral_kernel_prod_right'' {f : β × γ → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun x => ∫⁻ y, f (x, y) ∂η (a, x) := by
  -- Porting note: used `Prod.mk a` instead of `fun x => (a, x)` below
  change
    Measurable
      ((fun x => ∫⁻ y, (fun u : (α × β) × γ => f (u.1.2, u.2)) (x, y) ∂η x) ∘ Prod.mk a)
  -- Porting note: specified `κ`, `f`.
  refine (Measurable.lintegral_kernel_prod_right' (κ := η)
    (f := (fun u ↦ f (u.fst.snd, u.snd))) ?_).comp measurable_prod_mk_left
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    a : α
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    f : Prod β γ → ENNReal
    hf : Measurable f
    ⊢ Measurable fun u => f { fst := u.1.2, snd := u.2 }
  -/
  exact hf.comp (measurable_fst.snd.prod_mk measurable_snd)
  /-
    🎉 no goals
  -/


theorem _root_.Measurable.setLIntegral_kernel_prod_right {f : α → β → ℝ≥0∞}
    (hf : Measurable (uncurry f)) {s : Set β} (hs : MeasurableSet s) :
    Measurable fun a => ∫⁻ b in s, f a b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun a => MeasureTheory.lintegral ((κ a).restrict s) fun b => f a b
  -/
  simp_rw [← lintegral_restrict κ hs]; exact hf.lintegral_kernel_prod_right
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-06-29")]
alias _root_.Measurable.set_lintegral_kernel_prod_right :=
  _root_.Measurable.setLIntegral_kernel_prod_right


theorem _root_.Measurable.lintegral_kernel_prod_left' {f : β × α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun y => ∫⁻ x, f (x, y) ∂κ y :=
  (measurable_swap_iff.mpr hf).lintegral_kernel_prod_right'


theorem _root_.Measurable.lintegral_kernel_prod_left {f : β → α → ℝ≥0∞}
    (hf : Measurable (uncurry f)) : Measurable fun y => ∫⁻ x, f x y ∂κ y :=
  hf.lintegral_kernel_prod_left'


theorem _root_.Measurable.setLIntegral_kernel_prod_left {f : β → α → ℝ≥0∞}
    (hf : Measurable (uncurry f)) {s : Set β} (hs : MeasurableSet s) :
    Measurable fun b => ∫⁻ a in s, f a b ∂κ b := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : β → α → ENNReal
    hf : Measurable (Function.uncurry f)
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun b => MeasureTheory.lintegral ((κ b).restrict s) fun a => f a b
  -/
  simp_rw [← lintegral_restrict κ hs]; exact hf.lintegral_kernel_prod_left
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-06-29")]
alias _root_.Measurable.set_lintegral_kernel_prod_left :=
  _root_.Measurable.setLIntegral_kernel_prod_left


theorem _root_.Measurable.lintegral_kernel {f : β → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun a => ∫⁻ b, f b ∂κ a :=
  Measurable.lintegral_kernel_prod_right (hf.comp measurable_snd)


theorem _root_.Measurable.setLIntegral_kernel {f : β → ℝ≥0∞} (hf : Measurable f) {s : Set β}
    (hs : MeasurableSet s) : Measurable fun a => ∫⁻ b in s, f b ∂κ a := by
  -- Porting note: was term mode proof (`Function.comp` reducibility)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : β → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun a => MeasureTheory.lintegral ((κ a).restrict s) fun b => f b
  -/
  refine Measurable.setLIntegral_kernel_prod_right ?_ hs
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : β → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable (Function.uncurry fun a => f)
  -/
  convert hf.comp measurable_snd
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias _root_.Measurable.set_lintegral_kernel := _root_.Measurable.setLIntegral_kernel


