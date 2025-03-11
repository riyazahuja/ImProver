/-- Auxiliary function for the definition of the composition-product of two kernels.
For all `a : α`, `compProdFun κ η a` is a countably additive function with value zero on the empty
set, and the composition-product of kernels is defined in `Kernel.compProd` through
`Measure.ofMeasurable`. -/
noncomputable def compProdFun (κ : Kernel α β) (η : Kernel (α × β) γ) (a : α) (s : Set (β × γ)) :
    ℝ≥0∞ :=
  ∫⁻ b, η (a, b) {c | (b, c) ∈ s} ∂κ a


theorem compProdFun_empty (κ : Kernel α β) (η : Kernel (α × β) γ) (a : α) :
    compProdFun κ η a ∅ = 0 := by
  simp only [compProdFun, Set.mem_empty_iff_false, Set.setOf_false, measure_empty,
    MeasureTheory.lintegral_const, zero_mul]


theorem compProdFun_iUnion (κ : Kernel α β) (η : Kernel (α × β) γ) [IsSFiniteKernel η] (a : α)
    (f : ℕ → Set (β × γ)) (hf_meas : ∀ i, MeasurableSet (f i))
    (hf_disj : Pairwise (Disjoint on f)) :
    compProdFun κ η a (⋃ i, f i) = ∑' i, compProdFun κ η a (f i) := by
  have h_Union : (fun b ↦ η (a, b) {c : γ | (b, c) ∈ ⋃ i, f i})
      = fun b ↦ η (a, b) (⋃ i, {c : γ | (b, c) ∈ f i}) := by
    ext1 b
    congr with c
    simp only [Set.mem_iUnion, Set.iSup_eq_iUnion, Set.mem_setOf_eq]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Nat → Set (Prod β γ)
    hf_meas : ∀ (i : Nat), MeasurableSet (f i)
    hf_disj : Pairwise (Function.onFun Disjoint f)
    h_Union : Eq (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.m …
    ⊢ Eq (κ.compProdFun η a (Set.iUnion fun i => f i)) (tsum fun i => κ.compProdFu …
  -/
  rw [compProdFun, h_Union]
  have h_tsum : (fun b ↦ η (a, b) (⋃ i, {c : γ | (b, c) ∈ f i}))
      = fun b ↦ ∑' i, η (a, b) {c : γ | (b, c) ∈ f i} := by
    ext1 b
    rw [measure_iUnion]
    · intro i j hij s hsi hsj c hcs
      have hbci : {(b, c)} ⊆ f i := by rw [Set.singleton_subset_iff]; exact hsi hcs
      have hbcj : {(b, c)} ⊆ f j := by rw [Set.singleton_subset_iff]; exact hsj hcs
      simpa only [Set.bot_eq_empty, Set.le_eq_subset, Set.singleton_subset_iff,
        Set.mem_empty_iff_false] using hf_disj hij hbci hbcj
    · exact fun i ↦ measurable_prod_mk_left (hf_meas i)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Nat → Set (Prod β γ)
    hf_meas : ∀ (i : Nat), MeasurableSet (f i)
    hf_disj : Pairwise (Function.onFun Disjoint f)
    h_Union : Eq (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.m …
    h_tsum : Eq (fun b => (η { fst := a, snd := b }) (Set.iUnion fun i => setOf fu …
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η { fst := a, snd := b }) (Set.i …
  -/
  rw [h_tsum, lintegral_tsum]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Nat → Set (Prod β γ)
      hf_meas : ∀ (i : Nat), MeasurableSet (f i)
      hf_disj : Pairwise (Function.onFun Disjoint f)
      h_Union : Eq (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.m …
      h_tsum : Eq (fun b => (η { fst := a, snd := b }) (Set.iUnion fun i => setOf fu …
      ⊢ Eq (tsum fun i => MeasureTheory.lintegral (κ a) fun a_1 => (η { fst := a, sn …
    -/
  · simp [compProdFun]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Nat → Set (Prod β γ)
      hf_meas : ∀ (i : Nat), MeasurableSet (f i)
      hf_disj : Pairwise (Function.onFun Disjoint f)
      h_Union : Eq (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.m …
      h_tsum : Eq (fun b => (η { fst := a, snd := b }) (Set.iUnion fun i => setOf fu …
      ⊢ ∀ (i : Nat), AEMeasurable (fun b => (η { fst := a, snd := b }) (setOf fun c  …
    -/
  · intro i
    have hm : MeasurableSet {p : (α × β) × γ | (p.1.2, p.2) ∈ f i} :=
      measurable_fst.snd.prod_mk measurable_snd (hf_meas i)
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Nat → Set (Prod β γ)
      hf_meas : ∀ (i : Nat), MeasurableSet (f i)
      hf_disj : Pairwise (Function.onFun Disjoint f)
      h_Union : Eq (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.m …
      h_tsum : Eq (fun b => (η { fst := a, snd := b }) (Set.iUnion fun i => setOf fu …
      i : Nat
      hm : MeasurableSet (setOf fun p => Membership.mem (f i) { fst := p.1.2, snd := …
      ⊢ AEMeasurable (fun b => (η { fst := a, snd := b }) (setOf fun c => Membership …
    -/
    exact ((measurable_kernel_prod_mk_left hm).comp measurable_prod_mk_left).aemeasurable
    /-
      🎉 no goals
    -/


theorem compProdFun_tsum_right (κ : Kernel α β) (η : Kernel (α × β) γ) [IsSFiniteKernel η] (a : α)
    (hs : MeasurableSet s) : compProdFun κ η a s = ∑' n, compProdFun κ (seq η n) a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (κ.compProdFun η a s) (tsum fun n => κ.compProdFun (η.seq n) a s)
  -/
  simp_rw [compProdFun, (measure_sum_seq η _).symm]
  have : ∫⁻ b, Measure.sum (fun n => seq η n (a, b)) {c : γ | (b, c) ∈ s} ∂κ a
      = ∫⁻ b, ∑' n, seq η n (a, b) {c : γ | (b, c) ∈ s} ∂κ a := by
    congr with b
    rw [Measure.sum_apply]
    exact measurable_prod_mk_left hs
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    this : Eq (MeasureTheory.lintegral (κ a) fun b => (MeasureTheory.Measure.sum f …
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (MeasureTheory.Measure.sum fun n  …
  -/
  rw [this, lintegral_tsum]
  exact fun n ↦ ((measurable_kernel_prod_mk_left (κ := (seq η n))
    ((measurable_fst.snd.prod_mk measurable_snd) hs)).comp measurable_prod_mk_left).aemeasurable


theorem compProdFun_tsum_left (κ : Kernel α β) (η : Kernel (α × β) γ) [IsSFiniteKernel κ] (a : α)
    (s : Set (β × γ)) : compProdFun κ η a s = ∑' n, compProdFun (seq κ n) η a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    a : α
    s : Set (Prod β γ)
    ⊢ Eq (κ.compProdFun η a s) (tsum fun n => (κ.seq n).compProdFun η a s)
  -/
  simp_rw [compProdFun, (measure_sum_seq κ _).symm, lintegral_sum_measure]
  /-
    🎉 no goals
  -/


theorem compProdFun_eq_tsum (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) (hs : MeasurableSet s) :
    compProdFun κ η a s = ∑' (n) (m), compProdFun (seq κ n) (seq η m) a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (κ.compProdFun η a s) (tsum fun n => tsum fun m => (κ.seq n).compProdFun  …
  -/
  simp_rw [compProdFun_tsum_left κ η a s, compProdFun_tsum_right _ η a hs]
  /-
    🎉 no goals
  -/


theorem measurable_compProdFun (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (hs : MeasurableSet s) :
    Measurable fun a ↦ compProdFun κ η a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hs : MeasurableSet s
    ⊢ Measurable fun a => κ.compProdFun η a s
  -/
  simp only [compProdFun]
  have h_meas : Measurable (Function.uncurry fun a b => η (a, b) {c : γ | (b, c) ∈ s}) := by
    have : (Function.uncurry fun a b => η (a, b) {c : γ | (b, c) ∈ s})
        = fun p ↦ η p {c : γ | (p.2, c) ∈ s} := by
      ext1 p
      rw [Function.uncurry_apply_pair]
    rw [this]
    exact measurable_kernel_prod_mk_left (measurable_fst.snd.prod_mk measurable_snd hs)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hs : MeasurableSet s
    h_meas : Measurable (Function.uncurry fun a b => (η { fst := a, snd := b }) (s …
    ⊢ Measurable fun a => MeasureTheory.lintegral (κ a) fun b => (η { fst := a, sn …
  -/
  exact h_meas.lintegral_kernel_prod_right
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-30")]
alias measurable_compProdFun_of_finite := measurable_compProdFun


/-- Composition-Product of kernels. For s-finite kernels, it satisfies
`∫⁻ bc, f bc ∂(compProd κ η a) = ∫⁻ b, ∫⁻ c, f (b, c) ∂(η (a, b)) ∂(κ a)`
(see `ProbabilityTheory.Kernel.lintegral_compProd`).
If either of the kernels is not s-finite, `compProd` is given the junk value 0. -/
noncomputable def compProd (κ : Kernel α β) (η : Kernel (α × β) γ) : Kernel α (β × γ) :=
  if h : IsSFiniteKernel κ ∧ IsSFiniteKernel η then
  { toFun := fun a ↦
      have : IsSFiniteKernel η := h.2
      Measure.ofMeasurable (fun s _ ↦ compProdFun κ η a s) (compProdFun_empty κ η a)
        (compProdFun_iUnion κ η a)
    measurable' := by
      /-
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        s : Set (Prod β γ)
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel (Prod α β) γ
        h : And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKern …
        ⊢ Measurable fun a => letFun ⋯ fun this => MeasureTheory.Measure.ofMeasurable  …
      -/
      have : IsSFiniteKernel κ := h.1
      /-
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        s : Set (Prod β γ)
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel (Prod α β) γ
        h : And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKern …
        this : ProbabilityTheory.IsSFiniteKernel κ
        ⊢ Measurable fun a => letFun ⋯ fun this => MeasureTheory.Measure.ofMeasurable  …
      -/
      have : IsSFiniteKernel η := h.2
      /-
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        s : Set (Prod β γ)
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel (Prod α β) γ
        h : And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKern …
        this✝ : ProbabilityTheory.IsSFiniteKernel κ
        this : ProbabilityTheory.IsSFiniteKernel η
        ⊢ Measurable fun a => letFun ⋯ fun this => MeasureTheory.Measure.ofMeasurable  …
      -/
      refine Measure.measurable_of_measurable_coe _ fun s hs ↦ ?_
      have : (fun a ↦ Measure.ofMeasurable (fun s _ ↦ compProdFun κ η a s) (compProdFun_empty κ η a)
              (compProdFun_iUnion κ η a) s)
          = fun a ↦ compProdFun κ η a s := by
        ext1 a; rwa [Measure.ofMeasurable_apply]
      /-
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        s✝ : Set (Prod β γ)
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel (Prod α β) γ
        h : And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKern …
        this✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        this✝ : ProbabilityTheory.IsSFiniteKernel η
        s : Set (Prod β γ)
        hs : MeasurableSet s
        this : Eq (fun a => (MeasureTheory.Measure.ofMeasurable (fun s x => κ.compProd …
        ⊢ Measurable fun b => (letFun ⋯ fun this => MeasureTheory.Measure.ofMeasurable …
      -/
      rw [this]
      /-
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        s✝ : Set (Prod β γ)
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel (Prod α β) γ
        h : And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKern …
        this✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        this✝ : ProbabilityTheory.IsSFiniteKernel η
        s : Set (Prod β γ)
        hs : MeasurableSet s
        this : Eq (fun a => (MeasureTheory.Measure.ofMeasurable (fun s x => κ.compProd …
        ⊢ Measurable fun a => κ.compProdFun η a s
      -/
      exact measurable_compProdFun κ η hs }
      /-
        🎉 no goals
      -/
  else 0


@[inherit_doc]
scoped[ProbabilityTheory] infixl:100 " ⊗ₖ " => ProbabilityTheory.Kernel.compProd


theorem compProd_apply_eq_compProdFun (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) (hs : MeasurableSet s) :
    (κ ⊗ₖ η) a s = compProdFun κ η a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (((κ.compProd η) a) s) (κ.compProdFun η a s)
  -/
  rw [compProd, dif_pos]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (({ toFun := fun a => letFun ⋯ fun this => MeasureTheory.Measure.ofMeasur …
  -/
  swap
    /-
      case hc
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      hs : MeasurableSet s
      ⊢ And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteKernel …
    -/
                    /-
                      🎉 no goals
                    -/
  · constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/
  change
    Measure.ofMeasurable (fun s _ => compProdFun κ η a s) (compProdFun_empty κ η a)
        (compProdFun_iUnion κ η a) s =
      ∫⁻ b, η (a, b) {c | (b, c) ∈ s} ∂κ a
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.ofMeasurable (fun s x => κ.compProdFun η a s) ⋯ ⋯ …
  -/
  rw [Measure.ofMeasurable_apply _ hs]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (κ.compProdFun η a s) (MeasureTheory.lintegral (κ a) fun b => (η { fst := …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem compProd_of_not_isSFiniteKernel_left (κ : Kernel α β) (η : Kernel (α × β) γ)
    (h : ¬ IsSFiniteKernel κ) :
    κ ⊗ₖ η = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : Not (ProbabilityTheory.IsSFiniteKernel κ)
    ⊢ Eq (κ.compProd η) 0
  -/
  rw [compProd, dif_neg]
  /-
    case hnc
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : Not (ProbabilityTheory.IsSFiniteKernel κ)
    ⊢ Not (And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteK …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem compProd_of_not_isSFiniteKernel_right (κ : Kernel α β) (η : Kernel (α × β) γ)
    (h : ¬ IsSFiniteKernel η) :
    κ ⊗ₖ η = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : Not (ProbabilityTheory.IsSFiniteKernel η)
    ⊢ Eq (κ.compProd η) 0
  -/
  rw [compProd, dif_neg]
  /-
    case hnc
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : Not (ProbabilityTheory.IsSFiniteKernel η)
    ⊢ Not (And (ProbabilityTheory.IsSFiniteKernel κ) (ProbabilityTheory.IsSFiniteK …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem compProd_apply (hs : MeasurableSet s) (κ : Kernel α β) [IsSFiniteKernel κ]
    (η : Kernel (α × β) γ) [IsSFiniteKernel η] (a : α) :
    (κ ⊗ₖ η) a s = ∫⁻ b, η (a, b) {c | (b, c) ∈ s} ∂κ a :=
  compProd_apply_eq_compProdFun κ η a hs


theorem le_compProd_apply (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) (s : Set (β × γ)) :
    ∫⁻ b, η (a, b) {c | (b, c) ∈ s} ∂κ a ≤ (κ ⊗ₖ η) a s :=
  calc
    ∫⁻ b, η (a, b) {c | (b, c) ∈ s} ∂κ a ≤
        ∫⁻ b, η (a, b) {c | (b, c) ∈ toMeasurable ((κ ⊗ₖ η) a) s} ∂κ a :=
      lintegral_mono fun _ => measure_mono fun _ h_mem => subset_toMeasurable _ _ h_mem
    _ = (κ ⊗ₖ η) a (toMeasurable ((κ ⊗ₖ η) a) s) :=
      (Kernel.compProd_apply_eq_compProdFun κ η a (measurableSet_toMeasurable _ _)).symm
    _ = (κ ⊗ₖ η) a s := measure_toMeasurable s


@[simp]
lemma compProd_zero_left (κ : Kernel (α × β) γ) :
    (0 : Kernel α β) ⊗ₖ κ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel (Prod α β) γ
    ⊢ Eq (ProbabilityTheory.Kernel.compProd 0 κ) 0
  -/
  by_cases h : IsSFiniteKernel κ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel (Prod α β) γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      ⊢ Eq (ProbabilityTheory.Kernel.compProd 0 κ) 0
    -/
  · ext a s hs
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel (Prod α β) γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      ⊢ Eq (((ProbabilityTheory.Kernel.compProd 0 κ) a) s) ((0 a) s)
    -/
    rw [Kernel.compProd_apply hs]
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel (Prod α β) γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (0 a) fun b => (κ { fst := a, snd := b }) (setOf …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel (Prod α β) γ
      h : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ Eq (ProbabilityTheory.Kernel.compProd 0 κ) 0
    -/
  · rw [Kernel.compProd_of_not_isSFiniteKernel_right _ _ h]
    /-
      🎉 no goals
    -/


@[simp]
lemma compProd_zero_right (κ : Kernel α β) (γ : Type*) {mγ : MeasurableSpace γ} :
    κ ⊗ₖ (0 : Kernel (α × β) γ) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    ⊢ Eq (κ.compProd 0) 0
  -/
  by_cases h : IsSFiniteKernel κ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      ⊢ Eq (κ.compProd 0) 0
    -/
  · ext a s hs
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      ⊢ Eq (((κ.compProd 0) a) s) ((0 a) s)
    -/
    rw [Kernel.compProd_apply hs]
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      h : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (0 { fst := a, snd := b }) (setOf …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      h : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ Eq (κ.compProd 0) 0
    -/
  · rw [Kernel.compProd_of_not_isSFiniteKernel_left _ _ h]
    /-
      🎉 no goals
    -/


lemma compProd_preimage_fst {s : Set β} (hs : MeasurableSet s) (κ : Kernel α β)
    (η : Kernel (α × β) γ) [IsSFiniteKernel κ] [IsMarkovKernel η] (x : α) :
    (κ ⊗ₖ η) x (Prod.fst ⁻¹' s) = κ x s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set β
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    x : α
    ⊢ Eq (((κ.compProd η) x) (Set.preimage Prod.fst s)) ((κ x) s)
  -/
  rw [compProd_apply (measurable_fst hs)]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set β
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    x : α
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => (η { fst := x, snd := b }) (setOf …
  -/
  simp only [Set.mem_preimage]
  classical
  have : ∀ b : β, η (x, b) {_c | b ∈ s} = s.indicator (fun _ ↦ 1) b := by
    intro b
    by_cases hb : b ∈ s <;> simp [hb]
  simp_rw [this]
  rw [lintegral_indicator_const hs, one_mul]


lemma compProd_deterministic_apply [MeasurableSingletonClass γ] {f : α × β → γ} (hf : Measurable f)
    {s : Set (β × γ)} (hs : MeasurableSet s) (κ : Kernel α β) [IsSFiniteKernel κ] (x : α) :
    (κ ⊗ₖ deterministic f hf) x s = κ x {b | (b, f (x, b)) ∈ s} := by
  simp only [deterministic_apply, measurableSet_setOf, Set.mem_setOf_eq, Measure.dirac_apply,
    Set.mem_setOf_eq, Set.indicator_apply, Pi.one_apply, compProd_apply hs]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSingletonClass γ
    f : Prod α β → γ
    hf : Measurable f
    s : Set (Prod β γ)
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    x : α
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => ite (Membership.mem s { fst := b, …
  -/
  let t := {b | (b, f (x, b)) ∈ s}
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSingletonClass γ
    f : Prod α β → γ
    hf : Measurable f
    s : Set (Prod β γ)
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    x : α
    t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => ite (Membership.mem s { fst := b, …
  -/
  have ht : MeasurableSet t := (measurable_id.prod_mk (hf.comp measurable_prod_mk_left)) hs
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSingletonClass γ
    f : Prod α β → γ
    hf : Measurable f
    s : Set (Prod β γ)
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    x : α
    t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => ite (Membership.mem s { fst := b, …
  -/
  rw [← lintegral_add_compl _ ht]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSingletonClass γ
    f : Prod α β → γ
    hf : Measurable f
    s : Set (Prod β γ)
    hs : MeasurableSet s
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    x : α
    t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
    ht : MeasurableSet t
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral ((κ x).restrict t) fun x_1 => ite (Me …
  -/
  convert add_zero _
  · suffices ∀ b ∈ tᶜ, (if (b, f (x, b)) ∈ s then (1 : ℝ≥0∞) else 0) = 0 by
      rw [setLIntegral_congr_fun ht.compl (ae_of_all _ this), lintegral_zero]
    /-
      case h.e'_2.h.e'_6
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      ⊢ ∀ (b : β), Membership.mem (HasCompl.compl t) b → Eq (ite (Membership.mem s { …
    -/
    intro b hb
    /-
      case h.e'_2.h.e'_6
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      b : β
      hb : Membership.mem (HasCompl.compl t) b
      ⊢ Eq (ite (Membership.mem s { fst := b, snd := f { fst := x, snd := b } }) 1 0 …
    -/
    simp only [t, Set.mem_compl_iff, Set.mem_setOf_eq] at hb
    /-
      case h.e'_2.h.e'_6
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      b : β
      hb : Not (Membership.mem s { fst := b, snd := f { fst := x, snd := b } })
      ⊢ Eq (ite (Membership.mem s { fst := b, snd := f { fst := x, snd := b } }) 1 0 …
    -/
    simp [hb]
    /-
      🎉 no goals
    -/
  · suffices ∀ b ∈ t, (if (b, f (x, b)) ∈ s then (1 : ℝ≥0∞) else 0) = 1 by
      rw [setLIntegral_congr_fun ht (ae_of_all _ this), setLIntegral_one]
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      ⊢ ∀ (b : β), Membership.mem t b → Eq (ite (Membership.mem s { fst := b, snd := …
    -/
    intro b hb
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      b : β
      hb : Membership.mem t b
      ⊢ Eq (ite (Membership.mem s { fst := b, snd := f { fst := x, snd := b } }) 1 0 …
    -/
    simp only [t, Set.mem_setOf_eq] at hb
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSingletonClass γ
      f : Prod α β → γ
      hf : Measurable f
      s : Set (Prod β γ)
      hs : MeasurableSet s
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      x : α
      t : Set β := setOf fun b => Membership.mem s { fst := b, snd := f { fst := x,  …
      ht : MeasurableSet t
      b : β
      hb : Membership.mem s { fst := b, snd := f { fst := x, snd := b } }
      ⊢ Eq (ite (Membership.mem s { fst := b, snd := f { fst := x, snd := b } }) 1 0 …
    -/
    simp [hb]
    /-
      🎉 no goals
    -/


theorem ae_kernel_lt_top (a : α) (h2s : (κ ⊗ₖ η) a s ≠ ∞) :
    ∀ᵐ b ∂κ a, η (a, b) (Prod.mk b ⁻¹' s) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h2s : Ne (((κ.compProd η) a) s) Top.top
    ⊢ Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set.preimage  …
  -/
  let t := toMeasurable ((κ ⊗ₖ η) a) s
  have : ∀ b : β, η (a, b) (Prod.mk b ⁻¹' s) ≤ η (a, b) (Prod.mk b ⁻¹' t) := fun b =>
    measure_mono (Set.preimage_mono (subset_toMeasurable _ _))
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h2s : Ne (((κ.compProd η) a) s) Top.top
    t : Set (Prod β γ) := MeasureTheory.toMeasurable ((κ.compProd η) a) s
    this : ∀ (b : β), LE.le ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b)  …
    ⊢ Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set.preimage  …
  -/
  have ht : MeasurableSet t := measurableSet_toMeasurable _ _
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h2s : Ne (((κ.compProd η) a) s) Top.top
    t : Set (Prod β γ) := MeasureTheory.toMeasurable ((κ.compProd η) a) s
    this : ∀ (b : β), LE.le ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b)  …
    ht : MeasurableSet t
    ⊢ Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set.preimage  …
  -/
  have h2t : (κ ⊗ₖ η) a t ≠ ∞ := by rwa [measure_toMeasurable]
  have ht_lt_top : ∀ᵐ b ∂κ a, η (a, b) (Prod.mk b ⁻¹' t) < ∞ := by
    rw [Kernel.compProd_apply ht] at h2t
    exact ae_lt_top (Kernel.measurable_kernel_prod_mk_left' ht a) h2t
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h2s : Ne (((κ.compProd η) a) s) Top.top
    t : Set (Prod β γ) := MeasureTheory.toMeasurable ((κ.compProd η) a) s
    this : ∀ (b : β), LE.le ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b)  …
    ht : MeasurableSet t
    h2t : Ne (((κ.compProd η) a) t) Top.top
    ht_lt_top : Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set …
    ⊢ Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set.preimage  …
  -/
  filter_upwards [ht_lt_top] with b hb
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h2s : Ne (((κ.compProd η) a) s) Top.top
    t : Set (Prod β γ) := MeasureTheory.toMeasurable ((κ.compProd η) a) s
    this : ∀ (b : β), LE.le ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b)  …
    ht : MeasurableSet t
    h2t : Ne (((κ.compProd η) a) t) Top.top
    ht_lt_top : Filter.Eventually (fun b => LT.lt ((η { fst := a, snd := b }) (Set …
    b : β
    hb : LT.lt ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b) t)) Top.top
    ⊢ LT.lt ((η { fst := a, snd := b }) (Set.preimage (Prod.mk b) s)) Top.top
  -/
  exact (this b).trans_lt hb
  /-
    🎉 no goals
  -/


theorem compProd_null (a : α) (hs : MeasurableSet s) :
    (κ ⊗ₖ η) a s = 0 ↔ (fun b => η (a, b) (Prod.mk b ⁻¹' s)) =ᵐ[κ a] 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Iff (Eq (((κ.compProd η) a) s) 0) ((MeasureTheory.ae (κ a)).EventuallyEq (fu …
  -/
  rw [Kernel.compProd_apply hs, lintegral_eq_zero_iff]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      hs : MeasurableSet s
      ⊢ Iff ((MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      hs : MeasurableSet s
      ⊢ Measurable fun b => (η { fst := a, snd := b }) (setOf fun c => Membership.me …
    -/
  · exact Kernel.measurable_kernel_prod_mk_left' hs a
    /-
      🎉 no goals
    -/


theorem ae_null_of_compProd_null (h : (κ ⊗ₖ η) a s = 0) :
    (fun b => η (a, b) (Prod.mk b ⁻¹' s)) =ᵐ[κ a] 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h : Eq (((κ.compProd η) a) s) 0
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b }) ( …
  -/
  obtain ⟨t, hst, mt, ht⟩ := exists_measurable_superset_of_null h
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h : Eq (((κ.compProd η) a) s) 0
    t : Set (Prod β γ)
    hst : HasSubset.Subset s t
    mt : MeasurableSet t
    ht : Eq (((κ.compProd η) a) t) 0
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b }) ( …
  -/
  simp_rw [compProd_null a mt] at ht
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    h : Eq (((κ.compProd η) a) s) 0
    t : Set (Prod β γ)
    hst : HasSubset.Subset s t
    mt : MeasurableSet t
    ht : (MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b } …
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b }) ( …
  -/
  rw [Filter.eventuallyLE_antisymm_iff]
  exact
    ⟨Filter.EventuallyLE.trans_eq
        (Filter.Eventually.of_forall fun x => (measure_mono (Set.preimage_mono hst) : _)) ht,
      Filter.Eventually.of_forall fun x => zero_le _⟩


theorem ae_ae_of_ae_compProd {p : β × γ → Prop} (h : ∀ᵐ bc ∂(κ ⊗ₖ η) a, p bc) :
    ∀ᵐ b ∂κ a, ∀ᵐ c ∂η (a, b), p (b, c) :=
  ae_null_of_compProd_null h


lemma ae_compProd_of_ae_ae {κ : Kernel α β} {η : Kernel (α × β) γ}
    {p : β × γ → Prop} (hp : MeasurableSet {x | p x})
    (h : ∀ᵐ b ∂κ a, ∀ᵐ c ∂η (a, b), p (b, c)) :
    ∀ᵐ bc ∂(κ ⊗ₖ η) a, p bc := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    h : Filter.Eventually (fun b => Filter.Eventually (fun c => p { fst := b, snd  …
    ⊢ Filter.Eventually (fun bc => p bc) (MeasureTheory.ae ((κ.compProd η) a))
  -/
  by_cases hκ : IsSFiniteKernel κ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    h : Filter.Eventually (fun b => Filter.Eventually (fun c => p { fst := b, snd  …
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Filter.Eventually (fun bc => p bc) (MeasureTheory.ae ((κ.compProd η) a))
  -/
  swap; · simp [compProd_of_not_isSFiniteKernel_left _ _ hκ]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    h : Filter.Eventually (fun b => Filter.Eventually (fun c => p { fst := b, snd  …
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Filter.Eventually (fun bc => p bc) (MeasureTheory.ae ((κ.compProd η) a))
  -/
  by_cases hη : IsSFiniteKernel η
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    h : Filter.Eventually (fun b => Filter.Eventually (fun c => p { fst := b, snd  …
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Filter.Eventually (fun bc => p bc) (MeasureTheory.ae ((κ.compProd η) a))
  -/
  swap; · simp [compProd_of_not_isSFiniteKernel_right _ _ hη]
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    h : Filter.Eventually (fun b => Filter.Eventually (fun c => p { fst := b, snd  …
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Filter.Eventually (fun bc => p bc) (MeasureTheory.ae ((κ.compProd η) a))
  -/
  simp_rw [ae_iff] at h ⊢
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    a : α
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    p : Prod β γ → Prop
    hp : MeasurableSet (setOf fun x => p x)
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    h : Eq ((κ a) (setOf fun a_1 => Not (Eq ((η { fst := a, snd := a_1 }) (setOf f …
    ⊢ Eq (((κ.compProd η) a) (setOf fun a => Not (p a))) 0
  -/
  rw [compProd_null]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      a : α
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      p : Prod β γ → Prop
      hp : MeasurableSet (setOf fun x => p x)
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      hη : ProbabilityTheory.IsSFiniteKernel η
      h : Eq ((κ a) (setOf fun a_1 => Not (Eq ((η { fst := a, snd := a_1 }) (setOf f …
      ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun b => (η { fst := a, snd := b }) ( …
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case pos.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      a : α
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      p : Prod β γ → Prop
      hp : MeasurableSet (setOf fun x => p x)
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      hη : ProbabilityTheory.IsSFiniteKernel η
      h : Eq ((κ a) (setOf fun a_1 => Not (Eq ((η { fst := a, snd := a_1 }) (setOf f …
      ⊢ MeasurableSet (setOf fun a => Not (p a))
    -/
  · exact hp.compl
    /-
      🎉 no goals
    -/


lemma ae_compProd_iff {p : β × γ → Prop} (hp : MeasurableSet {x | p x}) :
    (∀ᵐ bc ∂(κ ⊗ₖ η) a, p bc) ↔ ∀ᵐ b ∂κ a, ∀ᵐ c ∂η (a, b), p (b, c) :=
  ⟨fun h ↦ ae_ae_of_ae_compProd h, fun h ↦ ae_compProd_of_ae_ae hp h⟩


theorem compProd_restrict {s : Set β} {t : Set γ} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    Kernel.restrict κ hs ⊗ₖ Kernel.restrict η ht = Kernel.restrict (κ ⊗ₖ η) (hs.prod ht) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set β
    t : Set γ
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq ((κ.restrict hs).compProd (η.restrict ht)) ((κ.compProd η).restrict ⋯)
  -/
  ext a u hu
  rw [compProd_apply hu, restrict_apply' _ _ _ hu,
    compProd_apply (hu.inter (hs.prod ht))]
  simp only [Kernel.restrict_apply, Measure.restrict_apply' ht, Set.mem_inter_iff,
    Set.prod_mk_mem_set_prod_eq]
  have :
    ∀ b,
      η (a, b) {c : γ | (b, c) ∈ u ∧ b ∈ s ∧ c ∈ t} =
        s.indicator (fun b => η (a, b) ({c : γ | (b, c) ∈ u} ∩ t)) b := by
    intro b
    classical
    rw [Set.indicator_apply]
    split_ifs with h
    · simp only [h, true_and, Set.inter_def, Set.mem_setOf]
    · simp only [h, false_and, and_false, Set.setOf_false, measure_empty]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set β
    t : Set γ
    hs : MeasurableSet s
    ht : MeasurableSet t
    a : α
    u : Set (Prod β γ)
    hu : MeasurableSet u
    this : ∀ (b : β), Eq ((η { fst := a, snd := b }) (setOf fun c => And (Membersh …
    ⊢ Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => (η { fst := a, snd : …
  -/
  simp_rw [this]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set β
    t : Set γ
    hs : MeasurableSet s
    ht : MeasurableSet t
    a : α
    u : Set (Prod β γ)
    hu : MeasurableSet u
    this : ∀ (b : β), Eq ((η { fst := a, snd := b }) (setOf fun c => And (Membersh …
    ⊢ Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => (η { fst := a, snd : …
  -/
  rw [lintegral_indicator hs]
  /-
    🎉 no goals
  -/


theorem compProd_restrict_left {s : Set β} (hs : MeasurableSet s) :
    Kernel.restrict κ hs ⊗ₖ η = Kernel.restrict (κ ⊗ₖ η) (hs.prod MeasurableSet.univ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((κ.restrict hs).compProd η) ((κ.compProd η).restrict ⋯)
  -/
  rw [← compProd_restrict]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      s : Set β
      hs : MeasurableSet s
      ⊢ Eq ((κ.restrict hs).compProd η) ((κ.restrict ?hs).compProd (η.restrict ?ht))
    -/
  · congr; exact Kernel.restrict_univ.symm
           /-
             🎉 no goals
           -/


theorem compProd_restrict_right {t : Set γ} (ht : MeasurableSet t) :
    κ ⊗ₖ Kernel.restrict η ht = Kernel.restrict (κ ⊗ₖ η) (MeasurableSet.univ.prod ht) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    t : Set γ
    ht : MeasurableSet t
    ⊢ Eq (κ.compProd (η.restrict ht)) ((κ.compProd η).restrict ⋯)
  -/
  rw [← compProd_restrict]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      t : Set γ
      ht : MeasurableSet t
      ⊢ Eq (κ.compProd (η.restrict ht)) ((κ.restrict ?hs).compProd (η.restrict ?ht))
    -/
  · congr; exact Kernel.restrict_univ.symm
           /-
             🎉 no goals
           -/


/-- Lebesgue integral against the composition-product of two kernels. -/
theorem lintegral_compProd' (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) {f : β → γ → ℝ≥0∞} (hf : Measurable (Function.uncurry f)) :
    ∫⁻ bc, f bc.1 bc.2 ∂(κ ⊗ₖ η) a = ∫⁻ b, ∫⁻ c, f b c ∂η (a, b) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc.1 bc.2) (Measu …
  -/
  let F : ℕ → SimpleFunc (β × γ) ℝ≥0∞ := SimpleFunc.eapprox (Function.uncurry f)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc.1 bc.2) (Measu …
  -/
  have h : ∀ a, ⨆ n, F n a = Function.uncurry f a := SimpleFunc.iSup_eapprox_apply hf
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : Prod β γ), Eq (iSup fun n => (F n) a) (Function.uncurry f a)
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc.1 bc.2) (Measu …
  -/
  simp only [Prod.forall, Function.uncurry_apply_pair] at h
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc.1 bc.2) (Measu …
  -/
  simp_rw [← h]
  have h_mono : Monotone F := fun i j hij b =>
    SimpleFunc.monotone_eapprox (Function.uncurry f) hij _
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => iSup fun n => (F n) …
  -/
  rw [lintegral_iSup (fun n => (F n).measurable) h_mono]
  have : ∀ b, ∫⁻ c, ⨆ n, F n (b, c) ∂η (a, b) = ⨆ n, ∫⁻ c, F n (b, c) ∂η (a, b) := by
    intro a
    rw [lintegral_iSup]
    · exact fun n => (F n).measurable.comp measurable_prod_mk_left
    · exact fun i j hij b => h_mono hij _
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    ⊢ Eq (iSup fun n => MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n)  …
  -/
  simp_rw [this]
  have h_some_meas_integral :
    ∀ f' : SimpleFunc (β × γ) ℝ≥0∞, Measurable fun b => ∫⁻ c, f' (b, c) ∂η (a, b) := by
    intro f'
    have :
      (fun b => ∫⁻ c, f' (b, c) ∂η (a, b)) =
        (fun ab => ∫⁻ c, f' (ab.2, c) ∂η ab) ∘ fun b => (a, b) := by
      ext1 ab; rfl
    rw [this]
    apply Measurable.comp _ (measurable_prod_mk_left (m := mα))
    exact Measurable.lintegral_kernel_prod_right
      ((SimpleFunc.measurable _).comp (measurable_fst.snd.prod_mk measurable_snd))
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
    ⊢ Eq (iSup fun n => MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n)  …
  -/
  rw [lintegral_iSup]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
    ⊢ Eq (iSup fun n => MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n)  …
  -/
  rotate_left
    /-
      case hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : β → γ → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      ⊢ ∀ (n : Nat), Measurable fun b => MeasureTheory.lintegral (η { fst := a, snd  …
    -/
  · exact fun n => h_some_meas_integral (F n)
    /-
      🎉 no goals
    -/
    /-
      case h_mono
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : β → γ → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      ⊢ Monotone fun n b => MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    -/
  · exact fun i j hij b => lintegral_mono fun c => h_mono hij _
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
    ⊢ Eq (iSup fun n => MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n)  …
  -/
  congr
  /-
    case e_s
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
    ⊢ Eq (fun n => MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n) a) fu …
  -/
  ext1 n
  /-
    case e_s.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : β → γ → ENNReal
    hf : Measurable (Function.uncurry f)
    F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
    h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
    h_mono : Monotone F
    this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
    h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
    n : Nat
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => (F n) a) (MeasureThe …
  -/
  refine SimpleFunc.induction ?_ ?_ (F n)
    /-
      case e_s.h.refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : β → γ → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      ⊢ ∀ (c : ENNReal) {s : Set (Prod β γ)} (hs : MeasurableSet s), Eq (MeasureTheo …
    -/
  · intro c s hs
    classical -- Porting note: Added `classical` for `Set.piecewise_eq_indicator`
    simp (config := { unfoldPartialApp := true }) only [SimpleFunc.const_zero,
      SimpleFunc.coe_piecewise, SimpleFunc.coe_const, SimpleFunc.coe_zero,
      Set.piecewise_eq_indicator, Function.const, lintegral_indicator_const hs]
    rw [compProd_apply hs, ← lintegral_const_mul c _]
    swap
    · exact (measurable_kernel_prod_mk_left ((measurable_fst.snd.prod_mk measurable_snd) hs)).comp
        measurable_prod_mk_left
    congr
    ext1 b
    rw [lintegral_indicator_const_comp measurable_prod_mk_left hs]
    rfl
    /-
      case e_s.h.refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : β → γ → ENNReal
      hf : Measurable (Function.uncurry f)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      ⊢ ∀ ⦃f g : MeasureTheory.SimpleFunc (Prod β γ) ENNReal⦄, Disjoint (Function.su …
    -/
  · intro f f' _ hf_eq hf'_eq
    /-
      case e_s.h.refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => (HAdd.hAdd f f') a)  …
    -/
    simp_rw [SimpleFunc.coe_add, Pi.add_apply]
    change
      ∫⁻ x, (f : β × γ → ℝ≥0∞) x + f' x ∂(κ ⊗ₖ η) a =
        ∫⁻ b, ∫⁻ c : γ, f (b, c) + f' (b, c) ∂η (a, b) ∂κ a
    /-
      case e_s.h.refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun x => HAdd.hAdd (f x) (f'  …
    -/
    rw [lintegral_add_left (SimpleFunc.measurable _), hf_eq, hf'_eq, ← lintegral_add_left]
    /-
      case e_s.h.refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun a_1 => HAdd.hAdd (MeasureTheory.linteg …
    -/
    swap
      /-
        case e_s.h.refine_2.hf
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        κ : ProbabilityTheory.Kernel α β
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝ : ProbabilityTheory.IsSFiniteKernel η
        a : α
        f✝ : β → γ → ENNReal
        hf : Measurable (Function.uncurry f✝)
        F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
        h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
        h_mono : Monotone F
        this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
        h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
        n : Nat
        f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
        a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
        hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
        hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
        ⊢ Measurable fun a_1 => MeasureTheory.lintegral (η { fst := a, snd := a_1 }) f …
      -/
    · exact h_some_meas_integral f
      /-
        🎉 no goals
      -/
    /-
      case e_s.h.refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun a_1 => HAdd.hAdd (MeasureTheory.linteg …
    -/
    congr with b
    /-
      case e_s.h.refine_2.e_f.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      b : β
      ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c => f …
    -/
    rw [lintegral_add_left]
    /-
      case e_s.h.refine_2.e_f.h.hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f✝ : β → γ → ENNReal
      hf : Measurable (Function.uncurry f✝)
      F : Nat → MeasureTheory.SimpleFunc (Prod β γ) ENNReal := MeasureTheory.SimpleF …
      h : ∀ (a : β) (b : γ), Eq (iSup fun n => (F n) { fst := a, snd := b }) (f✝ a b)
      h_mono : Monotone F
      this : ∀ (b : β), Eq (MeasureTheory.lintegral (η { fst := a, snd := b }) fun c …
      h_some_meas_integral : ∀ (f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal), M …
      n : Nat
      f f' : MeasureTheory.SimpleFunc (Prod β γ) ENNReal
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f a) (MeasureT …
      hf'_eq : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun a => f' a) (Measur …
      b : β
      ⊢ Measurable fun c => f { fst := b, snd := c }
    -/
    exact (SimpleFunc.measurable _).comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/


/-- Lebesgue integral against the composition-product of two kernels. -/
theorem lintegral_compProd (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) {f : β × γ → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ bc, f bc ∂(κ ⊗ₖ η) a = ∫⁻ b, ∫⁻ c, f (b, c) ∂η (a, b) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc) (MeasureTheor …
  -/
  let g := Function.curry f
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : Measurable f
    g : β → γ → ENNReal := Function.curry f
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc) (MeasureTheor …
  -/
  change ∫⁻ bc, f bc ∂(κ ⊗ₖ η) a = ∫⁻ b, ∫⁻ c, g b c ∂η (a, b) ∂κ a
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : Measurable f
    g : β → γ → ENNReal := Function.curry f
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc) (MeasureTheor …
  -/
  rw [← lintegral_compProd']
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → ENNReal
      hf : Measurable f
      g : β → γ → ENNReal := Function.curry f
      ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun bc => f bc) (MeasureTheor …
    -/
  · simp_rw [g, Function.curry_apply]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → ENNReal
      hf : Measurable f
      g : β → γ → ENNReal := Function.curry f
      ⊢ Measurable (Function.uncurry g)
    -/
  · simp_rw [g, Function.uncurry_curry]; exact hf
                                         /-
                                           🎉 no goals
                                         -/


/-- Lebesgue integral against the composition-product of two kernels. -/
theorem lintegral_compProd₀ (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) {f : β × γ → ℝ≥0∞} (hf : AEMeasurable f ((κ ⊗ₖ η) a)) :
    ∫⁻ z, f z ∂(κ ⊗ₖ η) a = ∫⁻ x, ∫⁻ y, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : AEMeasurable f ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun z => f z) (MeasureTheory. …
  -/
  have A : ∫⁻ z, f z ∂(κ ⊗ₖ η) a = ∫⁻ z, hf.mk f z ∂(κ ⊗ₖ η) a := lintegral_congr_ae hf.ae_eq_mk
  have B : ∫⁻ x, ∫⁻ y, f (x, y) ∂η (a, x) ∂κ a = ∫⁻ x, ∫⁻ y, hf.mk f (x, y) ∂η (a, x) ∂κ a := by
    apply lintegral_congr_ae
    filter_upwards [ae_ae_of_ae_compProd hf.ae_eq_mk] with _ ha using lintegral_congr_ae ha
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : AEMeasurable f ((κ.compProd η) a)
    A : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
    B : Eq (MeasureTheory.lintegral (κ a) fun x => MeasureTheory.lintegral (η { fs …
    ⊢ Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun z => f z) (MeasureTheory. …
  -/
  rw [A, B, lintegral_compProd]
  /-
    case hf
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : AEMeasurable f ((κ.compProd η) a)
    A : Eq (MeasureTheory.lintegral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
    B : Eq (MeasureTheory.lintegral (κ a) fun x => MeasureTheory.lintegral (η { fs …
    ⊢ Measurable (AEMeasurable.mk f hf)
  -/
  exact hf.measurable_mk
  /-
    🎉 no goals
  -/


theorem setLIntegral_compProd (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) {f : β × γ → ℝ≥0∞} (hf : Measurable f) {s : Set β} {t : Set γ}
    (hs : MeasurableSet s) (ht : MeasurableSet t) :
    ∫⁻ z in s ×ˢ t, f z ∂(κ ⊗ₖ η) a = ∫⁻ x in s, ∫⁻ y in t, f (x, y) ∂η (a, x) ∂κ a := by
  simp_rw [← Kernel.restrict_apply (κ ⊗ₖ η) (hs.prod ht), ← compProd_restrict hs ht,
    lintegral_compProd _ _ _ hf, Kernel.restrict_apply]


@[deprecated (since := "2024-06-29")]
alias set_lintegral_compProd := setLIntegral_compProd


theorem setLIntegral_compProd_univ_right (κ : Kernel α β) [IsSFiniteKernel κ]
    (η : Kernel (α × β) γ) [IsSFiniteKernel η] (a : α) {f : β × γ → ℝ≥0∞} (hf : Measurable f)
    {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ z in s ×ˢ Set.univ, f z ∂(κ ⊗ₖ η) a = ∫⁻ x in s, ∫⁻ y, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (((κ.compProd η) a).restrict (SProd.sprod s Set. …
  -/
  simp_rw [setLIntegral_compProd κ η a hf hs MeasurableSet.univ, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_compProd_univ_right := setLIntegral_compProd_univ_right


theorem setLIntegral_compProd_univ_left (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) {f : β × γ → ℝ≥0∞} (hf : Measurable f) {t : Set γ}
    (ht : MeasurableSet t) :
    ∫⁻ z in Set.univ ×ˢ t, f z ∂(κ ⊗ₖ η) a = ∫⁻ x, ∫⁻ y in t, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → ENNReal
    hf : Measurable f
    t : Set γ
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (((κ.compProd η) a).restrict (SProd.sprod Set.un …
  -/
  simp_rw [setLIntegral_compProd κ η a hf MeasurableSet.univ ht, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_compProd_univ_left := setLIntegral_compProd_univ_left


theorem compProd_eq_tsum_compProd (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] (a : α) (hs : MeasurableSet s) :
    (κ ⊗ₖ η) a s = ∑' (n : ℕ) (m : ℕ), (seq κ n ⊗ₖ seq η m) a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    hs : MeasurableSet s
    ⊢ Eq (((κ.compProd η) a) s) (tsum fun n => tsum fun m => (((κ.seq n).compProd  …
  -/
  simp_rw [compProd_apply_eq_compProdFun _ _ _ hs]; exact compProdFun_eq_tsum κ η a hs
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem compProd_eq_sum_compProd (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] : κ ⊗ₖ η = Kernel.sum fun n => Kernel.sum fun m => seq κ n ⊗ₖ seq η m := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.K …
  -/
  ext a s hs; simp_rw [Kernel.sum_apply' _ a hs]; rw [compProd_eq_tsum_compProd κ η a hs]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem compProd_eq_sum_compProd_left (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ) :
    κ ⊗ₖ η = Kernel.sum fun n => seq κ n ⊗ₖ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).compProd η)
  -/
  by_cases h : IsSFiniteKernel η
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).compProd η)
  -/
  swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h : Not (ProbabilityTheory.IsSFiniteKernel η)
      ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).compProd η)
    -/
  · simp_rw [compProd_of_not_isSFiniteKernel_right _ _ h]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h : Not (ProbabilityTheory.IsSFiniteKernel η)
      ⊢ Eq 0 (ProbabilityTheory.Kernel.sum fun n => 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).compProd η)
  -/
  rw [compProd_eq_sum_compProd]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.sum fun m …
  -/
  congr with n a s hs
  simp_rw [Kernel.sum_apply' _ _ hs, compProd_apply_eq_compProdFun _ _ _ hs,
    compProdFun_tsum_right _ η a hs]


theorem compProd_eq_sum_compProd_right (κ : Kernel α β) (η : Kernel (α × β) γ)
    [IsSFiniteKernel η] : κ ⊗ₖ η = Kernel.sum fun n => κ ⊗ₖ seq η n := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => κ.compProd (η.seq n))
  -/
  by_cases hκ : IsSFiniteKernel κ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => κ.compProd (η.seq n))
  -/
  swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => κ.compProd (η.seq n))
    -/
  · simp_rw [compProd_of_not_isSFiniteKernel_left _ _ hκ]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ Eq 0 (ProbabilityTheory.Kernel.sum fun n => 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (κ.compProd η) (ProbabilityTheory.Kernel.sum fun n => κ.compProd (η.seq n))
  -/
  rw [compProd_eq_sum_compProd]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.sum fun m …
  -/
  simp_rw [compProd_eq_sum_compProd_left κ _]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.sum fun m …
  -/
  rw [Kernel.sum_comm]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.compProd (κ : Kernel α β) [IsMarkovKernel κ] (η : Kernel (α × β) γ)
    [IsMarkovKernel η] : IsMarkovKernel (κ ⊗ₖ η) where
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  mα : MeasurableSpace α
                                  mβ : MeasurableSpace β
                                  γ : Type u_3
                                  mγ : MeasurableSpace γ
                                  s : Set (Prod β γ)
                                  κ : ProbabilityTheory.Kernel α β
                                  inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
                                  η : ProbabilityTheory.Kernel (Prod α β) γ
                                  inst✝ : ProbabilityTheory.IsMarkovKernel η
                                  a : α
                                  ⊢ Eq (((κ.compProd η) a) Set.univ) 1
                                -/
  isProbabilityMeasure a := ⟨by simp [compProd_apply]⟩
                                /-
                                  🎉 no goals
                                -/


theorem compProd_apply_univ_le (κ : Kernel α β) (η : Kernel (α × β) γ) [IsFiniteKernel η] (a : α) :
    (κ ⊗ₖ η) a Set.univ ≤ κ a Set.univ * IsFiniteKernel.bound η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (((κ.compProd η) a) Set.univ) (HMul.hMul ((κ a) Set.univ) (Probability …
  -/
  by_cases hκ : IsSFiniteKernel κ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ LE.le (((κ.compProd η) a) Set.univ) (HMul.hMul ((κ a) Set.univ) (Probability …
  -/
  swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ LE.le (((κ.compProd η) a) Set.univ) (HMul.hMul ((κ a) Set.univ) (Probability …
    -/
  · rw [compProd_of_not_isSFiniteKernel_left _ _ hκ]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ LE.le ((0 a) Set.univ) (HMul.hMul ((κ a) Set.univ) (ProbabilityTheory.IsFini …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ LE.le (((κ.compProd η) a) Set.univ) (HMul.hMul ((κ a) Set.univ) (Probability …
  -/
  rw [compProd_apply .univ]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ LE.le (MeasureTheory.lintegral (κ a) fun b => (η { fst := a, snd := b }) (se …
  -/
  simp only [Set.mem_univ, Set.setOf_true]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ LE.le (MeasureTheory.lintegral (κ a) fun b => (η { fst := a, snd := b }) Set …
  -/
  let Cη := IsFiniteKernel.bound η
  calc
    ∫⁻ b, η (a, b) Set.univ ∂κ a ≤ ∫⁻ _, Cη ∂κ a :=
      lintegral_mono fun b => measure_le_bound η (a, b) Set.univ
    _ = Cη * κ a Set.univ := MeasureTheory.lintegral_const Cη
    _ = κ a Set.univ * Cη := mul_comm _ _


instance IsFiniteKernel.compProd (κ : Kernel α β) [IsFiniteKernel κ] (η : Kernel (α × β) γ)
    [IsFiniteKernel η] : IsFiniteKernel (κ ⊗ₖ η) :=
  ⟨⟨IsFiniteKernel.bound κ * IsFiniteKernel.bound η,
      ENNReal.mul_lt_top (IsFiniteKernel.bound_lt_top κ) (IsFiniteKernel.bound_lt_top η), fun a =>
      calc
        (κ ⊗ₖ η) a Set.univ ≤ κ a Set.univ * IsFiniteKernel.bound η := compProd_apply_univ_le κ η a
        _ ≤ IsFiniteKernel.bound κ * IsFiniteKernel.bound η :=
          mul_le_mul (measure_le_bound κ a Set.univ) le_rfl (zero_le _) (zero_le _)⟩⟩


instance IsSFiniteKernel.compProd (κ : Kernel α β) (η : Kernel (α × β) γ) :
    IsSFiniteKernel (κ ⊗ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
  -/
  by_cases h : IsSFiniteKernel κ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
  -/
  swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
    -/
  · rw [compProd_of_not_isSFiniteKernel_left _ _ h]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ ProbabilityTheory.IsSFiniteKernel 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
  -/
  by_cases h : IsSFiniteKernel η
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h✝ : ProbabilityTheory.IsSFiniteKernel κ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
  -/
  swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h✝ : ProbabilityTheory.IsSFiniteKernel κ
      h : Not (ProbabilityTheory.IsSFiniteKernel η)
      ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
    -/
  · rw [compProd_of_not_isSFiniteKernel_right _ _ h]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      s : Set (Prod β γ)
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel (Prod α β) γ
      h✝ : ProbabilityTheory.IsSFiniteKernel κ
      h : Not (ProbabilityTheory.IsSFiniteKernel η)
      ⊢ ProbabilityTheory.IsSFiniteKernel 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h✝ : ProbabilityTheory.IsSFiniteKernel κ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.compProd η)
  -/
  rw [compProd_eq_sum_compProd]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    s : Set (Prod β γ)
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    h✝ : ProbabilityTheory.IsSFiniteKernel κ
    h : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum fun n => Pro …
  -/
  exact Kernel.isSFiniteKernel_sum fun n => Kernel.isSFiniteKernel_sum inferInstance
  /-
    🎉 no goals
  -/


lemma compProd_add_left (μ κ : Kernel α β) (η : Kernel (α × β) γ)
    [IsSFiniteKernel μ] [IsSFiniteKernel κ] [IsSFiniteKernel η] :
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           mα : MeasurableSpace α
                                           mβ : MeasurableSpace β
                                           γ : Type u_3
                                           mγ : MeasurableSpace γ
                                           μ κ : ProbabilityTheory.Kernel α β
                                           η : ProbabilityTheory.Kernel (Prod α β) γ
                                           inst✝² : ProbabilityTheory.IsSFiniteKernel μ
                                           inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
                                           inst✝ : ProbabilityTheory.IsSFiniteKernel η
                                           ⊢ Eq ((HAdd.hAdd μ κ).compProd η) (HAdd.hAdd (μ.compProd η) (κ.compProd η))
                                         -/
    (μ + κ) ⊗ₖ η = μ ⊗ₖ η + κ ⊗ₖ η := by ext _ _ hs; simp [compProd_apply hs]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma compProd_add_right (μ : Kernel α β) (κ η : Kernel (α × β) γ)
    [IsSFiniteKernel μ] [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    μ ⊗ₖ (κ + η) = μ ⊗ₖ κ + μ ⊗ₖ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : ProbabilityTheory.Kernel α β
    κ η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝² : ProbabilityTheory.IsSFiniteKernel μ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (μ.compProd (HAdd.hAdd κ η)) (HAdd.hAdd (μ.compProd κ) (μ.compProd η))
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : ProbabilityTheory.Kernel α β
    κ η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝² : ProbabilityTheory.IsSFiniteKernel μ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((μ.compProd (HAdd.hAdd κ η)) a) s) (((HAdd.hAdd (μ.compProd κ) (μ.compP …
  -/
  simp only [compProd_apply hs, coe_add, Pi.add_apply, Measure.coe_add]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : ProbabilityTheory.Kernel α β
    κ η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝² : ProbabilityTheory.IsSFiniteKernel μ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (μ a) fun b => HAdd.hAdd ((κ { fst := a, snd :=  …
  -/
  rw [lintegral_add_left]
  /-
    case h.h.hf
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : ProbabilityTheory.Kernel α β
    κ η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝² : ProbabilityTheory.IsSFiniteKernel μ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Measurable fun b => (κ { fst := a, snd := b }) (setOf fun c => Membership.me …
  -/
  exact measurable_kernel_prod_mk_left' hs a
  /-
    🎉 no goals
  -/


lemma comapRight_compProd_id_prod {δ : Type*} {mδ : MeasurableSpace δ}
    (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel (α × β) γ) [IsSFiniteKernel η]
    {f : δ → γ} (hf : MeasurableEmbedding f) :
    comapRight (κ ⊗ₖ η) (MeasurableEmbedding.id.prodMap hf) = κ ⊗ₖ (comapRight η hf) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    δ : Type u_4
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    f : δ → γ
    hf : MeasurableEmbedding f
    ⊢ Eq ((κ.compProd η).comapRight ⋯) (κ.compProd (η.comapRight hf))
  -/
  ext a t ht
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    δ : Type u_4
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    f : δ → γ
    hf : MeasurableEmbedding f
    a : α
    t : Set (Prod β δ)
    ht : MeasurableSet t
    ⊢ Eq ((((κ.compProd η).comapRight ⋯) a) t) (((κ.compProd (η.comapRight hf)) a) …
  -/
  rw [comapRight_apply' _ _ _ ht, compProd_apply, compProd_apply ht]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      δ : Type u_4
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : δ → γ
      hf : MeasurableEmbedding f
      a : α
      t : Set (Prod β δ)
      ht : MeasurableSet t
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η { fst := a, snd := b }) (setOf …
    -/
  · refine lintegral_congr fun b ↦ ?_
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      δ : Type u_4
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : δ → γ
      hf : MeasurableEmbedding f
      a : α
      t : Set (Prod β δ)
      ht : MeasurableSet t
      b : β
      ⊢ Eq ((η { fst := a, snd := b }) (setOf fun c => Membership.mem (Set.image (Pr …
    -/
    rw [comapRight_apply']
      /-
        case h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        δ : Type u_4
        mδ : MeasurableSpace δ
        κ : ProbabilityTheory.Kernel α β
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝ : ProbabilityTheory.IsSFiniteKernel η
        f : δ → γ
        hf : MeasurableEmbedding f
        a : α
        t : Set (Prod β δ)
        ht : MeasurableSet t
        b : β
        ⊢ Eq ((η { fst := a, snd := b }) (setOf fun c => Membership.mem (Set.image (Pr …
      -/
    · congr with x
      /-
        case h.h.h.e_6.h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        δ : Type u_4
        mδ : MeasurableSpace δ
        κ : ProbabilityTheory.Kernel α β
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝ : ProbabilityTheory.IsSFiniteKernel η
        f : δ → γ
        hf : MeasurableEmbedding f
        a : α
        t : Set (Prod β δ)
        ht : MeasurableSet t
        b : β
        x : γ
        ⊢ Iff (Membership.mem (setOf fun c => Membership.mem (Set.image (Prod.map id f …
      -/
      aesop
      /-
        🎉 no goals
      -/
      /-
        case h.h.ht
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        mγ : MeasurableSpace γ
        δ : Type u_4
        mδ : MeasurableSpace δ
        κ : ProbabilityTheory.Kernel α β
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝ : ProbabilityTheory.IsSFiniteKernel η
        f : δ → γ
        hf : MeasurableEmbedding f
        a : α
        t : Set (Prod β δ)
        ht : MeasurableSet t
        b : β
        ⊢ MeasurableSet (setOf fun c => Membership.mem t { fst := b, snd := c })
      -/
    · exact measurable_prod_mk_left ht
      /-
        🎉 no goals
      -/
    /-
      case h.h.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      δ : Type u_4
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : δ → γ
      hf : MeasurableEmbedding f
      a : α
      t : Set (Prod β δ)
      ht : MeasurableSet t
      ⊢ MeasurableSet (Set.image (Prod.map id f) t)
    -/
  · exact (MeasurableEmbedding.id.prodMap hf).measurableSet_image.mpr ht
    /-
      🎉 no goals
    -/


/-- The pushforward of a kernel along a measurable function. This is an implementation detail,
use `map κ f` instead. -/
noncomputable def mapOfMeasurable (κ : Kernel α β) (f : β → γ) (hf : Measurable f) :
    Kernel α γ where
  toFun a := (κ a).map f
  measurable' := (Measure.measurable_map _ hf).comp (Kernel.measurable κ)


open Classical in
/-- The pushforward of a kernel along a function.
If the function is not measurable, we use zero instead. This choice of junk
value ensures that typeclass inference can infer that the `map` of a kernel
satisfying `IsZeroOrMarkovKernel` again satisfies this property. -/
noncomputable def map [MeasurableSpace γ] (κ : Kernel α β) (f : β → γ) : Kernel α γ :=
  if hf : Measurable f then mapOfMeasurable κ f hf else 0


theorem map_of_not_measurable (κ : Kernel α β) {f : β → γ} (hf : ¬(Measurable f)) :
    map κ f = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    f : β → γ
    hf : Not (Measurable f)
    ⊢ Eq (κ.map f) 0
  -/
  simp [map, hf]
  /-
    🎉 no goals
  -/


@[simp] theorem mapOfMeasurable_eq_map (κ : Kernel α β) {f : β → γ} (hf : Measurable f) :
    mapOfMeasurable κ f hf = map κ f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    f : β → γ
    hf : Measurable f
    ⊢ Eq (κ.mapOfMeasurable f hf) (κ.map f)
  -/
  simp [map, hf]
  /-
    🎉 no goals
  -/


theorem map_apply (κ : Kernel α β) (hf : Measurable f) (a : α) : map κ f a = (κ a).map f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    κ : ProbabilityTheory.Kernel α β
    hf : Measurable f
    a : α
    ⊢ Eq ((κ.map f) a) (MeasureTheory.Measure.map f (κ a))
  -/
  simp only [map, hf, ↓reduceDIte, mapOfMeasurable, coe_mk]
  /-
    🎉 no goals
  -/


theorem map_apply' (κ : Kernel α β) (hf : Measurable f) (a : α) {s : Set γ} (hs : MeasurableSet s) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        mα : MeasurableSpace α
                                        mβ : MeasurableSpace β
                                        γ : Type u_3
                                        mγ : MeasurableSpace γ
                                        f : β → γ
                                        κ : ProbabilityTheory.Kernel α β
                                        hf : Measurable f
                                        a : α
                                        s : Set γ
                                        hs : MeasurableSet s
                                        ⊢ Eq (((κ.map f) a) s) ((κ a) (Set.preimage f s))
                                      -/
    map κ f a s = κ a (f ⁻¹' s) := by rw [map_apply _ hf, Measure.map_apply hf hs]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
lemma map_zero : Kernel.map (0 : Kernel α β) f = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    ⊢ Eq (ProbabilityTheory.Kernel.map 0 f) 0
  -/
  ext
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    a✝¹ : α
    s✝ : Set γ
    a✝ : MeasurableSet s✝
    ⊢ Eq (((ProbabilityTheory.Kernel.map 0 f) a✝¹) s✝) ((0 a✝¹) s✝)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      f : β → γ
      a✝¹ : α
      s✝ : Set γ
      a✝ : MeasurableSet s✝
      hf : Measurable f
      ⊢ Eq (((ProbabilityTheory.Kernel.map 0 f) a✝¹) s✝) ((0 a✝¹) s✝)
    -/
  · simp [map_apply, hf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      f : β → γ
      a✝¹ : α
      s✝ : Set γ
      a✝ : MeasurableSet s✝
      hf : Not (Measurable f)
      ⊢ Eq (((ProbabilityTheory.Kernel.map 0 f) a✝¹) s✝) ((0 a✝¹) s✝)
    -/
  · simp [map_of_not_measurable _ hf, map_apply]
    /-
      🎉 no goals
    -/


@[simp]
lemma map_id (κ : Kernel α β) : map κ id = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq (κ.map id) κ
  -/
  ext a
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    a : α
    s✝ : Set β
    a✝ : MeasurableSet s✝
    ⊢ Eq (((κ.map id) a) s✝) ((κ a) s✝)
  -/
  simp [map_apply, measurable_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_id' (κ : Kernel α β) : map κ (fun a ↦ a) = κ := map_id κ


nonrec theorem lintegral_map (κ : Kernel α β) (hf : Measurable f) (a : α) {g' : γ → ℝ≥0∞}
    (hg : Measurable g') : ∫⁻ b, g' b ∂map κ f a = ∫⁻ a, g' (f a) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    κ : ProbabilityTheory.Kernel α β
    hf : Measurable f
    a : α
    g' : γ → ENNReal
    hg : Measurable g'
    ⊢ Eq (MeasureTheory.lintegral ((κ.map f) a) fun b => g' b) (MeasureTheory.lint …
  -/
  rw [map_apply _ hf, lintegral_map hg hf]
  /-
    🎉 no goals
  -/


theorem sum_map_seq (κ : Kernel α β) [IsSFiniteKernel κ] (f : β → γ) :
    (Kernel.sum fun n => map (seq κ n) f) = map κ f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : β → γ
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).map f) (κ.map f)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : β → γ
      hf : Measurable f
      ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).map f) (κ.map f)
    -/
  · ext a s hs
    rw [Kernel.sum_apply, map_apply' κ hf a hs, Measure.sum_apply _ hs, ← measure_sum_seq κ,
      Measure.sum_apply _ (hf hs)]
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : β → γ
      hf : Measurable f
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ Eq (tsum fun i => (((κ.seq i).map f) a) s) (tsum fun i => ((κ.seq i) a) (Set …
    -/
    simp_rw [map_apply' _ hf _ hs]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : β → γ
      hf : Not (Measurable f)
      ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).map f) (κ.map f)
    -/
  · simp [map_of_not_measurable _ hf]
    /-
      🎉 no goals
    -/


lemma IsMarkovKernel.map (κ : Kernel α β) [IsMarkovKernel κ] (hf : Measurable f) :
    IsMarkovKernel (map κ f) :=
                /-
                  α : Type u_1
                  β : Type u_2
                  mα : MeasurableSpace α
                  mβ : MeasurableSpace β
                  γ : Type u_3
                  mγ : MeasurableSpace γ
                  f : β → γ
                  κ : ProbabilityTheory.Kernel α β
                  inst✝ : ProbabilityTheory.IsMarkovKernel κ
                  hf : Measurable f
                  a : α
                  ⊢ Eq (((κ.map f) a) Set.univ) 1
                -/
  ⟨fun a => ⟨by rw [map_apply' κ hf a MeasurableSet.univ, Set.preimage_univ, measure_univ]⟩⟩
                /-
                  🎉 no goals
                -/


instance IsZeroOrMarkovKernel.map (κ : Kernel α β) [IsZeroOrMarkovKernel κ] (f : β → γ) :
    IsZeroOrMarkovKernel (map κ f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f✝ : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    f : β → γ
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.map f)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f✝ : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      f : β → γ
      hf : Measurable f
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.map f)
    -/
  · rcases eq_zero_or_isMarkovKernel κ with rfl | h
      /-
        case pos.inl
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        δ : Type u_4
        mγ : MeasurableSpace γ
        mδ : MeasurableSpace δ
        f✝ : β → γ
        g : γ → α
        f : β → γ
        hf : Measurable f
        inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
        ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.map 0 f)
      -/
    · simp only [map_zero]; infer_instance
                            /-
                              🎉 no goals
                            -/
      /-
        case pos.inr
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        δ : Type u_4
        mγ : MeasurableSpace γ
        mδ : MeasurableSpace δ
        f✝ : β → γ
        g : γ → α
        κ : ProbabilityTheory.Kernel α β
        inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
        f : β → γ
        hf : Measurable f
        h : ProbabilityTheory.IsMarkovKernel κ
        ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.map f)
      -/
    · have := IsMarkovKernel.map κ hf; infer_instance
                                       /-
                                         🎉 no goals
                                       -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f✝ : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      f : β → γ
      hf : Not (Measurable f)
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.map f)
    -/
  · simp only [map_of_not_measurable _ hf]; infer_instance
                                            /-
                                              🎉 no goals
                                            -/


instance IsFiniteKernel.map (κ : Kernel α β) [IsFiniteKernel κ] (f : β → γ) :
    IsFiniteKernel (map κ f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f✝ : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : β → γ
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.map f)
  -/
  refine ⟨⟨IsFiniteKernel.bound κ, IsFiniteKernel.bound_lt_top κ, fun a => ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f✝ : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : β → γ
    a : α
    ⊢ LE.le (((κ.map f) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound κ)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f✝ : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      f : β → γ
      a : α
      hf : Measurable f
      ⊢ LE.le (((κ.map f) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound κ)
    -/
  · rw [map_apply' κ hf a MeasurableSet.univ]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f✝ : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      f : β → γ
      a : α
      hf : Measurable f
      ⊢ LE.le ((κ a) (Set.preimage f Set.univ)) (ProbabilityTheory.IsFiniteKernel.bo …
    -/
    exact measure_le_bound κ a _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f✝ : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      f : β → γ
      a : α
      hf : Not (Measurable f)
      ⊢ LE.le (((κ.map f) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound κ)
    -/
  · simp [map_of_not_measurable _ hf]
    /-
      🎉 no goals
    -/


instance IsSFiniteKernel.map (κ : Kernel α β) [IsSFiniteKernel κ] (f : β → γ) :
    IsSFiniteKernel (map κ f) :=
  ⟨⟨fun n => Kernel.map (seq κ n) f, inferInstance, (sum_map_seq κ f).symm⟩⟩


@[simp]
lemma map_const (μ : Measure α) {f : α → β} (hf : Measurable f) :
    map (const γ μ) f = const γ (μ.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    ⊢ Eq ((ProbabilityTheory.Kernel.const γ μ).map f) (ProbabilityTheory.Kernel.co …
  -/
  ext x s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    x : γ
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((((ProbabilityTheory.Kernel.const γ μ).map f) x) s) (((ProbabilityTheory …
  -/
  rw [map_apply' _ hf _ hs, const_apply, const_apply, Measure.map_apply hf hs]
  /-
    🎉 no goals
  -/


/-- Pullback of a kernel, such that for each set s `comap κ g hg c s = κ (g c) s`.
We include measurability in the assumptions instead of using junk values
to make sure that typeclass inference can infer that the `comap` of a Markov kernel
is again a Markov kernel. -/
def comap (κ : Kernel α β) (g : γ → α) (hg : Measurable g) : Kernel γ β where
  toFun a := κ (g a)
  measurable' := κ.measurable.comp hg


@[simp, norm_cast]
lemma coe_comap (κ : Kernel α β) (g : γ → α) (hg : Measurable g) : κ.comap g hg = κ ∘ g := rfl


theorem comap_apply (κ : Kernel α β) (hg : Measurable g) (c : γ) : comap κ g hg c = κ (g c) :=
  rfl


theorem comap_apply' (κ : Kernel α β) (hg : Measurable g) (c : γ) (s : Set β) :
    comap κ g hg c s = κ (g c) s :=
  rfl


@[simp]
lemma comap_zero (hg : Measurable g) : Kernel.comap (0 : Kernel α β) g hg = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    g : γ → α
    hg : Measurable g
    ⊢ Eq (ProbabilityTheory.Kernel.comap 0 g hg) 0
  -/
  ext; rw [Kernel.comap_apply]; simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       mα : MeasurableSpace α
                                                                       mβ : MeasurableSpace β
                                                                       κ : ProbabilityTheory.Kernel α β
                                                                       ⊢ Eq (κ.comap id ⋯) κ
                                                                     -/
lemma comap_id (κ : Kernel α β) : comap κ id measurable_id = κ := by ext a; rw [comap_apply]; simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp]
lemma comap_id' (κ : Kernel α β) : comap κ (fun a ↦ a) measurable_id = κ := comap_id κ


theorem lintegral_comap (κ : Kernel α β) (hg : Measurable g) (c : γ) (g' : β → ℝ≥0∞) :
    ∫⁻ b, g' b ∂comap κ g hg c = ∫⁻ b, g' b ∂κ (g c) :=
  rfl


theorem sum_comap_seq (κ : Kernel α β) [IsSFiniteKernel κ] (hg : Measurable g) :
    (Kernel.sum fun n => comap (seq κ n) g hg) = comap κ g hg := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hg : Measurable g
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).comap g hg) (κ.comap g hg)
  -/
  ext a s hs
  rw [Kernel.sum_apply, comap_apply' κ hg a s, Measure.sum_apply _ hs, ← measure_sum_seq κ,
    Measure.sum_apply _ hs]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hg : Measurable g
    a : γ
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (tsum fun i => (((κ.seq i).comap g hg) a) s) (tsum fun i => ((κ.seq i) (g …
  -/
  simp_rw [comap_apply' _ hg _ s]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.comap (κ : Kernel α β) [IsMarkovKernel κ] (hg : Measurable g) :
    IsMarkovKernel (comap κ g hg) :=
                /-
                  α : Type u_1
                  β : Type u_2
                  mα : MeasurableSpace α
                  mβ : MeasurableSpace β
                  γ : Type u_3
                  δ : Type u_4
                  mγ : MeasurableSpace γ
                  mδ : MeasurableSpace δ
                  f : β → γ
                  g : γ → α
                  κ : ProbabilityTheory.Kernel α β
                  inst✝ : ProbabilityTheory.IsMarkovKernel κ
                  hg : Measurable g
                  a : γ
                  ⊢ Eq (((κ.comap g hg) a) Set.univ) 1
                -/
  ⟨fun a => ⟨by rw [comap_apply' κ hg a Set.univ, measure_univ]⟩⟩
                /-
                  🎉 no goals
                -/


instance IsZeroOrMarkovKernel.comap (κ : Kernel α β) [IsZeroOrMarkovKernel κ] (hg : Measurable g) :
    IsZeroOrMarkovKernel (comap κ g hg) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    hg : Measurable g
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.comap g hg)
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f : β → γ
      g : γ → α
      hg : Measurable g
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.comap 0 g hg)
    -/
  · simp only [comap_zero]; infer_instance
                            /-
                              🎉 no goals
                            -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      f : β → γ
      g : γ → α
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      hg : Measurable g
      h : ProbabilityTheory.IsMarkovKernel κ
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.comap g hg)
    -/
  · have := IsMarkovKernel.comap κ hg; infer_instance
                                       /-
                                         🎉 no goals
                                       -/


instance IsFiniteKernel.comap (κ : Kernel α β) [IsFiniteKernel κ] (hg : Measurable g) :
    IsFiniteKernel (comap κ g hg) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hg : Measurable g
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.comap g hg)
  -/
  refine ⟨⟨IsFiniteKernel.bound κ, IsFiniteKernel.bound_lt_top κ, fun a => ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hg : Measurable g
    a : γ
    ⊢ LE.le (((κ.comap g hg) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound κ)
  -/
  rw [comap_apply' κ hg a Set.univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    f : β → γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hg : Measurable g
    a : γ
    ⊢ LE.le ((κ (g a)) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound κ)
  -/
  exact measure_le_bound κ _ _
  /-
    🎉 no goals
  -/


instance IsSFiniteKernel.comap (κ : Kernel α β) [IsSFiniteKernel κ] (hg : Measurable g) :
    IsSFiniteKernel (comap κ g hg) :=
  ⟨⟨fun n => Kernel.comap (seq κ n) g hg, inferInstance, (sum_comap_seq κ hg).symm⟩⟩


lemma comap_map_comm (κ : Kernel β γ) {f : α → β} {g : γ → δ}
    (hf : Measurable f) (hg : Measurable g) :
    comap (map κ g) f hf = map (comap κ f hf) g := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel β γ
    f : α → β
    g : γ → δ
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq ((κ.map g).comap f hf) ((κ.comap f hf).map g)
  -/
  ext x s _
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel β γ
    f : α → β
    g : γ → δ
    hf : Measurable f
    hg : Measurable g
    x : α
    s : Set δ
    a✝ : MeasurableSet s
    ⊢ Eq ((((κ.map g).comap f hf) x) s) ((((κ.comap f hf).map g) x) s)
  -/
  rw [comap_apply, map_apply _ hg, map_apply _ hg, comap_apply]
  /-
    🎉 no goals
  -/


/-- Define a `Kernel (γ × α) β` from a `Kernel α β` by taking the comap of the projection. -/
def prodMkLeft (γ : Type*) [MeasurableSpace γ] (κ : Kernel α β) : Kernel (γ × α) β :=
  comap κ Prod.snd measurable_snd


/-- Define a `Kernel (α × γ) β` from a `Kernel α β` by taking the comap of the projection. -/
def prodMkRight (γ : Type*) [MeasurableSpace γ] (κ : Kernel α β) : Kernel (α × γ) β :=
  comap κ Prod.fst measurable_fst


@[simp]
theorem prodMkLeft_apply (κ : Kernel α β) (ca : γ × α) : prodMkLeft γ κ ca = κ ca.snd :=
  rfl


@[simp]
theorem prodMkRight_apply (κ : Kernel α β) (ca : α × γ) : prodMkRight γ κ ca = κ ca.fst := rfl


theorem prodMkLeft_apply' (κ : Kernel α β) (ca : γ × α) (s : Set β) :
    prodMkLeft γ κ ca s = κ ca.snd s :=
  rfl


theorem prodMkRight_apply' (κ : Kernel α β) (ca : α × γ) (s : Set β) :
    prodMkRight γ κ ca s = κ ca.fst s := rfl


@[simp]
lemma prodMkLeft_zero : Kernel.prodMkLeft α (0 : Kernel β γ) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    ⊢ Eq (ProbabilityTheory.Kernel.prodMkLeft α 0) 0
  -/
  ext x s _; simp
             /-
               🎉 no goals
             -/


@[simp]
lemma prodMkRight_zero : Kernel.prodMkRight α (0 : Kernel β γ) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    ⊢ Eq (ProbabilityTheory.Kernel.prodMkRight α 0) 0
  -/
  ext x s _; simp
             /-
               🎉 no goals
             -/


@[simp]
lemma prodMkLeft_add (κ η : Kernel α β) :
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   mα : MeasurableSpace α
                                                                   mβ : MeasurableSpace β
                                                                   γ : Type u_4
                                                                   mγ : MeasurableSpace γ
                                                                   κ η : ProbabilityTheory.Kernel α β
                                                                   ⊢ Eq (ProbabilityTheory.Kernel.prodMkLeft γ (HAdd.hAdd κ η)) (HAdd.hAdd (Proba …
                                                                 -/
    prodMkLeft γ (κ + η) = prodMkLeft γ κ + prodMkLeft γ η := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
lemma prodMkRight_add (κ η : Kernel α β) :
                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      mα : MeasurableSpace α
                                                                      mβ : MeasurableSpace β
                                                                      γ : Type u_4
                                                                      mγ : MeasurableSpace γ
                                                                      κ η : ProbabilityTheory.Kernel α β
                                                                      ⊢ Eq (ProbabilityTheory.Kernel.prodMkRight γ (HAdd.hAdd κ η)) (HAdd.hAdd (Prob …
                                                                    -/
    prodMkRight γ (κ + η) = prodMkRight γ κ + prodMkRight γ η := by ext; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem lintegral_prodMkLeft (κ : Kernel α β) (ca : γ × α) (g : β → ℝ≥0∞) :
    ∫⁻ b, g b ∂prodMkLeft γ κ ca = ∫⁻ b, g b ∂κ ca.snd := rfl


theorem lintegral_prodMkRight (κ : Kernel α β) (ca : α × γ) (g : β → ℝ≥0∞) :
    ∫⁻ b, g b ∂prodMkRight γ κ ca = ∫⁻ b, g b ∂κ ca.fst := rfl


instance IsMarkovKernel.prodMkLeft (κ : Kernel α β) [IsMarkovKernel κ] :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            mα : MeasurableSpace α
                                            mβ : MeasurableSpace β
                                            δ : Type u_3
                                            mδ : MeasurableSpace δ
                                            γ : Type u_4
                                            mγ : MeasurableSpace γ
                                            κ : ProbabilityTheory.Kernel α β
                                            inst✝ : ProbabilityTheory.IsMarkovKernel κ
                                            ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.prodMkLeft γ κ)
                                          -/
    IsMarkovKernel (prodMkLeft γ κ) := by rw [Kernel.prodMkLeft]; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance IsMarkovKernel.prodMkRight (κ : Kernel α β) [IsMarkovKernel κ] :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             mα : MeasurableSpace α
                                             mβ : MeasurableSpace β
                                             δ : Type u_3
                                             mδ : MeasurableSpace δ
                                             γ : Type u_4
                                             mγ : MeasurableSpace γ
                                             κ : ProbabilityTheory.Kernel α β
                                             inst✝ : ProbabilityTheory.IsMarkovKernel κ
                                             ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.prodMkRight γ κ)
                                           -/
    IsMarkovKernel (prodMkRight γ κ) := by rw [Kernel.prodMkRight]; infer_instance
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance IsZeroOrMarkovKernel.prodMkLeft (κ : Kernel α β) [IsZeroOrMarkovKernel κ] :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  mα : MeasurableSpace α
                                                  mβ : MeasurableSpace β
                                                  δ : Type u_3
                                                  mδ : MeasurableSpace δ
                                                  γ : Type u_4
                                                  mγ : MeasurableSpace γ
                                                  κ : ProbabilityTheory.Kernel α β
                                                  inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
                                                  ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.prodMkLeft  …
                                                -/
    IsZeroOrMarkovKernel (prodMkLeft γ κ) := by rw [Kernel.prodMkLeft]; infer_instance
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


instance IsZeroOrMarkovKernel.prodMkRight (κ : Kernel α β) [IsZeroOrMarkovKernel κ] :
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   mα : MeasurableSpace α
                                                   mβ : MeasurableSpace β
                                                   δ : Type u_3
                                                   mδ : MeasurableSpace δ
                                                   γ : Type u_4
                                                   mγ : MeasurableSpace γ
                                                   κ : ProbabilityTheory.Kernel α β
                                                   inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
                                                   ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.prodMkRight …
                                                 -/
    IsZeroOrMarkovKernel (prodMkRight γ κ) := by rw [Kernel.prodMkRight]; infer_instance
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


instance IsFiniteKernel.prodMkLeft (κ : Kernel α β) [IsFiniteKernel κ] :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            mα : MeasurableSpace α
                                            mβ : MeasurableSpace β
                                            δ : Type u_3
                                            mδ : MeasurableSpace δ
                                            γ : Type u_4
                                            mγ : MeasurableSpace γ
                                            κ : ProbabilityTheory.Kernel α β
                                            inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                            ⊢ ProbabilityTheory.IsFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft γ κ)
                                          -/
    IsFiniteKernel (prodMkLeft γ κ) := by rw [Kernel.prodMkLeft]; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance IsFiniteKernel.prodMkRight (κ : Kernel α β) [IsFiniteKernel κ] :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             mα : MeasurableSpace α
                                             mβ : MeasurableSpace β
                                             δ : Type u_3
                                             mδ : MeasurableSpace δ
                                             γ : Type u_4
                                             mγ : MeasurableSpace γ
                                             κ : ProbabilityTheory.Kernel α β
                                             inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                             ⊢ ProbabilityTheory.IsFiniteKernel (ProbabilityTheory.Kernel.prodMkRight γ κ)
                                           -/
    IsFiniteKernel (prodMkRight γ κ) := by rw [Kernel.prodMkRight]; infer_instance
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance IsSFiniteKernel.prodMkLeft (κ : Kernel α β) [IsSFiniteKernel κ] :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             mα : MeasurableSpace α
                                             mβ : MeasurableSpace β
                                             δ : Type u_3
                                             mδ : MeasurableSpace δ
                                             γ : Type u_4
                                             mγ : MeasurableSpace γ
                                             κ : ProbabilityTheory.Kernel α β
                                             inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                             ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft γ κ)
                                           -/
    IsSFiniteKernel (prodMkLeft γ κ) := by rw [Kernel.prodMkLeft]; infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance IsSFiniteKernel.prodMkRight (κ : Kernel α β) [IsSFiniteKernel κ] :
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              mα : MeasurableSpace α
                                              mβ : MeasurableSpace β
                                              δ : Type u_3
                                              mδ : MeasurableSpace δ
                                              γ : Type u_4
                                              mγ : MeasurableSpace γ
                                              κ : ProbabilityTheory.Kernel α β
                                              inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                              ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkRight γ κ)
                                            -/
    IsSFiniteKernel (prodMkRight γ κ) := by rw [Kernel.prodMkRight]; infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma isSFiniteKernel_prodMkLeft_unit {κ : Kernel α β} :
    IsSFiniteKernel (prodMkLeft Unit κ) ↔ IsSFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Iff (ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft  …
  -/
  refine ⟨fun _ ↦ ?_, fun _ ↦ inferInstance⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    x✝ : ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft Un …
    ⊢ ProbabilityTheory.IsSFiniteKernel κ
  -/
  change IsSFiniteKernel ((prodMkLeft Unit κ).comap (fun a ↦ ((), a)) (by fun_prop))
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    x✝ : ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft Un …
    ⊢ ProbabilityTheory.IsSFiniteKernel ((ProbabilityTheory.Kernel.prodMkLeft Unit …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isSFiniteKernel_prodMkRight_unit {κ : Kernel α β} :
    IsSFiniteKernel (prodMkRight Unit κ) ↔ IsSFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Iff (ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkRight …
  -/
  refine ⟨fun _ ↦ ?_, fun _ ↦ inferInstance⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    x✝ : ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkRight U …
    ⊢ ProbabilityTheory.IsSFiniteKernel κ
  -/
  change IsSFiniteKernel ((prodMkRight Unit κ).comap (fun a ↦ (a, ())) (by fun_prop))
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    x✝ : ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkRight U …
    ⊢ ProbabilityTheory.IsSFiniteKernel ((ProbabilityTheory.Kernel.prodMkRight Uni …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma map_prodMkLeft (γ : Type*) [MeasurableSpace γ] (κ : Kernel α β) (f : β → δ) :
    map (prodMkLeft γ κ) f = prodMkLeft γ (map κ f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_5
    inst✝ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    f : β → δ
    ⊢ Eq ((ProbabilityTheory.Kernel.prodMkLeft γ κ).map f) (ProbabilityTheory.Kern …
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_5
      inst✝ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → δ
      hf : Measurable f
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkLeft γ κ).map f) (ProbabilityTheory.Kern …
    -/
  · simp only [map, hf, ↓reduceDIte]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_5
      inst✝ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → δ
      hf : Measurable f
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkLeft γ κ).mapOfMeasurable f ⋯) (Probabil …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_5
      inst✝ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → δ
      hf : Not (Measurable f)
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkLeft γ κ).map f) (ProbabilityTheory.Kern …
    -/
  · simp [map_of_not_measurable _ hf]
    /-
      🎉 no goals
    -/


lemma map_prodMkRight (κ : Kernel α β) (γ : Type*) {mγ : MeasurableSpace γ} (f : β → δ) :
    map (prodMkRight γ κ) f = prodMkRight γ (map κ f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    γ : Type u_5
    mγ : MeasurableSpace γ
    f : β → δ
    ⊢ Eq ((ProbabilityTheory.Kernel.prodMkRight γ κ).map f) (ProbabilityTheory.Ker …
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_5
      mγ : MeasurableSpace γ
      f : β → δ
      hf : Measurable f
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkRight γ κ).map f) (ProbabilityTheory.Ker …
    -/
  · simp only [map, hf, ↓reduceDIte]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_5
      mγ : MeasurableSpace γ
      f : β → δ
      hf : Measurable f
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkRight γ κ).mapOfMeasurable f ⋯) (Probabi …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      γ : Type u_5
      mγ : MeasurableSpace γ
      f : β → δ
      hf : Not (Measurable f)
      ⊢ Eq ((ProbabilityTheory.Kernel.prodMkRight γ κ).map f) (ProbabilityTheory.Ker …
    -/
  · simp [map_of_not_measurable _ hf]
    /-
      🎉 no goals
    -/


/-- Define a `Kernel (β × α) γ` from a `Kernel (α × β) γ` by taking the comap of `Prod.swap`. -/
def swapLeft (κ : Kernel (α × β) γ) : Kernel (β × α) γ :=
  comap κ Prod.swap measurable_swap


@[simp]
theorem swapLeft_apply (κ : Kernel (α × β) γ) (a : β × α) : swapLeft κ a = κ a.swap := rfl


theorem swapLeft_apply' (κ : Kernel (α × β) γ) (a : β × α) (s : Set γ) :
    swapLeft κ a s = κ a.swap s := rfl


theorem lintegral_swapLeft (κ : Kernel (α × β) γ) (a : β × α) (g : γ → ℝ≥0∞) :
    ∫⁻ c, g c ∂swapLeft κ a = ∫⁻ c, g c ∂κ a.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel (Prod α β) γ
    a : Prod β α
    g : γ → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (κ.swapLeft a) fun c => g c) (MeasureTheory.lint …
  -/
  rw [swapLeft_apply]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.swapLeft (κ : Kernel (α × β) γ) [IsMarkovKernel κ] :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        mα : MeasurableSpace α
                                        mβ : MeasurableSpace β
                                        δ : Type u_3
                                        mδ : MeasurableSpace δ
                                        γ : Type u_4
                                        mγ : MeasurableSpace γ
                                        κ : ProbabilityTheory.Kernel (Prod α β) γ
                                        inst✝ : ProbabilityTheory.IsMarkovKernel κ
                                        ⊢ ProbabilityTheory.IsMarkovKernel κ.swapLeft
                                      -/
    IsMarkovKernel (swapLeft κ) := by rw [Kernel.swapLeft]; infer_instance
                                                            /-
                                                              🎉 no goals
                                                            -/


instance IsFiniteKernel.swapLeft (κ : Kernel (α × β) γ) [IsFiniteKernel κ] :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        mα : MeasurableSpace α
                                        mβ : MeasurableSpace β
                                        δ : Type u_3
                                        mδ : MeasurableSpace δ
                                        γ : Type u_4
                                        mγ : MeasurableSpace γ
                                        κ : ProbabilityTheory.Kernel (Prod α β) γ
                                        inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                        ⊢ ProbabilityTheory.IsFiniteKernel κ.swapLeft
                                      -/
    IsFiniteKernel (swapLeft κ) := by rw [Kernel.swapLeft]; infer_instance
                                                            /-
                                                              🎉 no goals
                                                            -/


instance IsSFiniteKernel.swapLeft (κ : Kernel (α × β) γ) [IsSFiniteKernel κ] :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         mα : MeasurableSpace α
                                         mβ : MeasurableSpace β
                                         δ : Type u_3
                                         mδ : MeasurableSpace δ
                                         γ : Type u_4
                                         mγ : MeasurableSpace γ
                                         κ : ProbabilityTheory.Kernel (Prod α β) γ
                                         inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                         ⊢ ProbabilityTheory.IsSFiniteKernel κ.swapLeft
                                       -/
    IsSFiniteKernel (swapLeft κ) := by rw [Kernel.swapLeft]; infer_instance
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp] lemma swapLeft_prodMkLeft (κ : Kernel α β) (γ : Type*) {_ : MeasurableSpace γ} :
    swapLeft (prodMkLeft γ κ) = prodMkRight γ κ := rfl


@[simp] lemma swapLeft_prodMkRight (κ : Kernel α β) (γ : Type*) {_ : MeasurableSpace γ} :
    swapLeft (prodMkRight γ κ) = prodMkLeft γ κ := rfl


/-- Define a `Kernel α (γ × β)` from a `Kernel α (β × γ)` by taking the map of `Prod.swap`.
We use `mapOfMeasurable` in the definition for better defeqs. -/
noncomputable def swapRight (κ : Kernel α (β × γ)) : Kernel α (γ × β) :=
  mapOfMeasurable κ Prod.swap measurable_swap


lemma swapRight_eq (κ : Kernel α (β × γ)) : swapRight κ = map κ Prod.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    ⊢ Eq κ.swapRight (κ.map Prod.swap)
  -/
  simp [swapRight]
  /-
    🎉 no goals
  -/


theorem swapRight_apply (κ : Kernel α (β × γ)) (a : α) : swapRight κ a = (κ a).map Prod.swap :=
  rfl


theorem swapRight_apply' (κ : Kernel α (β × γ)) (a : α) {s : Set (γ × β)} (hs : MeasurableSet s) :
    swapRight κ a s = κ a {p | p.swap ∈ s} := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    s : Set (Prod γ β)
    hs : MeasurableSet s
    ⊢ Eq ((κ.swapRight a) s) ((κ a) (setOf fun p => Membership.mem s p.swap))
  -/
  rw [swapRight_apply, Measure.map_apply measurable_swap hs]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem lintegral_swapRight (κ : Kernel α (β × γ)) (a : α) {g : γ × β → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ c, g c ∂swapRight κ a = ∫⁻ bc : β × γ, g bc.swap ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    g : Prod γ β → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral (κ.swapRight a) fun c => g c) (MeasureTheory.lin …
  -/
  rw [swapRight_eq, lintegral_map _ measurable_swap a hg]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.swapRight (κ : Kernel α (β × γ)) [IsMarkovKernel κ] :
    IsMarkovKernel (swapRight κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.swapRight
  -/
  rw [Kernel.swapRight_eq]; exact IsMarkovKernel.map _ measurable_swap
                            /-
                              🎉 no goals
                            -/


instance IsZeroOrMarkovKernel.swapRight (κ : Kernel α (β × γ)) [IsZeroOrMarkovKernel κ] :
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               mα : MeasurableSpace α
                                               mβ : MeasurableSpace β
                                               δ : Type u_3
                                               mδ : MeasurableSpace δ
                                               γ : Type u_4
                                               mγ : MeasurableSpace γ
                                               κ : ProbabilityTheory.Kernel α (Prod β γ)
                                               inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
                                               ⊢ ProbabilityTheory.IsZeroOrMarkovKernel κ.swapRight
                                             -/
    IsZeroOrMarkovKernel (swapRight κ) := by rw [Kernel.swapRight_eq]; infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance IsFiniteKernel.swapRight (κ : Kernel α (β × γ)) [IsFiniteKernel κ] :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         mα : MeasurableSpace α
                                         mβ : MeasurableSpace β
                                         δ : Type u_3
                                         mδ : MeasurableSpace δ
                                         γ : Type u_4
                                         mγ : MeasurableSpace γ
                                         κ : ProbabilityTheory.Kernel α (Prod β γ)
                                         inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                         ⊢ ProbabilityTheory.IsFiniteKernel κ.swapRight
                                       -/
    IsFiniteKernel (swapRight κ) := by rw [Kernel.swapRight_eq]; infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance IsSFiniteKernel.swapRight (κ : Kernel α (β × γ)) [IsSFiniteKernel κ] :
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          mα : MeasurableSpace α
                                          mβ : MeasurableSpace β
                                          δ : Type u_3
                                          mδ : MeasurableSpace δ
                                          γ : Type u_4
                                          mγ : MeasurableSpace γ
                                          κ : ProbabilityTheory.Kernel α (Prod β γ)
                                          inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                          ⊢ ProbabilityTheory.IsSFiniteKernel κ.swapRight
                                        -/
    IsSFiniteKernel (swapRight κ) := by rw [Kernel.swapRight_eq]; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Define a `Kernel α β` from a `Kernel α (β × γ)` by taking the map of the first projection.
We use `mapOfMeasurable` for better defeqs. -/
noncomputable def fst (κ : Kernel α (β × γ)) : Kernel α β :=
  mapOfMeasurable κ Prod.fst measurable_fst


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       mα : MeasurableSpace α
                                                                       mβ : MeasurableSpace β
                                                                       γ : Type u_4
                                                                       mγ : MeasurableSpace γ
                                                                       κ : ProbabilityTheory.Kernel α (Prod β γ)
                                                                       ⊢ Eq κ.fst (κ.map Prod.fst)
                                                                     -/
theorem fst_eq (κ : Kernel α (β × γ)) : fst κ = map κ Prod.fst := by simp [fst]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem fst_apply (κ : Kernel α (β × γ)) (a : α) : fst κ a = (κ a).map Prod.fst :=
  rfl


theorem fst_apply' (κ : Kernel α (β × γ)) (a : α) {s : Set β} (hs : MeasurableSet s) :
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          mα : MeasurableSpace α
                                          mβ : MeasurableSpace β
                                          γ : Type u_4
                                          mγ : MeasurableSpace γ
                                          κ : ProbabilityTheory.Kernel α (Prod β γ)
                                          a : α
                                          s : Set β
                                          hs : MeasurableSet s
                                          ⊢ Eq ((κ.fst a) s) ((κ a) (setOf fun p => Membership.mem s p.1))
                                        -/
    fst κ a s = κ a {p | p.1 ∈ s} := by rw [fst_apply, Measure.map_apply measurable_fst hs]; rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp]
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        mα : MeasurableSpace α
                                                        mβ : MeasurableSpace β
                                                        γ : Type u_4
                                                        mγ : MeasurableSpace γ
                                                        ⊢ Eq (ProbabilityTheory.Kernel.fst 0) 0
                                                      -/
lemma fst_zero : fst (0 : Kernel α (β × γ)) = 0 := by simp [fst]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem lintegral_fst (κ : Kernel α (β × γ)) (a : α) {g : β → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ c, g c ∂fst κ a = ∫⁻ bc : β × γ, g bc.fst ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    g : β → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun c => g c) (MeasureTheory.lintegral …
  -/
  rw [fst_eq, lintegral_map _ measurable_fst a hg]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.fst (κ : Kernel α (β × γ)) [IsMarkovKernel κ] : IsMarkovKernel (fst κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.fst
  -/
  rw [Kernel.fst_eq]; exact IsMarkovKernel.map _ measurable_fst
                      /-
                        🎉 no goals
                      -/


instance IsZeroOrMarkovKernel.fst (κ : Kernel α (β × γ)) [IsZeroOrMarkovKernel κ] :
    IsZeroOrMarkovKernel (fst κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel κ.fst
  -/
  rw [Kernel.fst_eq]; infer_instance
                      /-
                        🎉 no goals
                      -/


instance IsFiniteKernel.fst (κ : Kernel α (β × γ)) [IsFiniteKernel κ] : IsFiniteKernel (fst κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsFiniteKernel κ.fst
  -/
  rw [Kernel.fst_eq]; infer_instance
                      /-
                        🎉 no goals
                      -/


instance IsSFiniteKernel.fst (κ : Kernel α (β × γ)) [IsSFiniteKernel κ] :
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    mα : MeasurableSpace α
                                    mβ : MeasurableSpace β
                                    δ : Type u_3
                                    mδ : MeasurableSpace δ
                                    γ : Type u_4
                                    mγ : MeasurableSpace γ
                                    κ : ProbabilityTheory.Kernel α (Prod β γ)
                                    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                    ⊢ ProbabilityTheory.IsSFiniteKernel κ.fst
                                  -/
    IsSFiniteKernel (fst κ) := by rw [Kernel.fst_eq]; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (priority := 100) isFiniteKernel_of_isFiniteKernel_fst {κ : Kernel α (β × γ)}
    [h : IsFiniteKernel (fst κ)] :
    IsFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.fst
    ⊢ ProbabilityTheory.IsFiniteKernel κ
  -/
  refine ⟨h.bound, h.bound_lt_top, fun a ↦ le_trans ?_ (measure_le_bound (fst κ) a Set.univ)⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.fst
    a : α
    ⊢ LE.le ((κ a) Set.univ) ((κ.fst a) Set.univ)
  -/
  rw [fst_apply' _ _ MeasurableSet.univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.fst
    a : α
    ⊢ LE.le ((κ a) Set.univ) ((κ a) (setOf fun p => Membership.mem Set.univ p.1))
  -/
  simp
  /-
    🎉 no goals
  -/


lemma fst_map_prod (κ : Kernel α β) {f : β → γ} {g : β → δ} (hg : Measurable g) :
    fst (map κ (fun x ↦ (f x, g x))) = map κ f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    f : β → γ
    g : β → δ
    hg : Measurable g
    ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).fst (κ.map f)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hg : Measurable g
      hf : Measurable f
      ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).fst (κ.map f)
    -/
  · ext x s hs
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hg : Measurable g
      hf : Measurable f
      x : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ Eq (((κ.map fun x => { fst := f x, snd := g x }).fst x) s) (((κ.map f) x) s)
    -/
    rw [fst_apply' _ _ hs, map_apply' _ (hf.prod hg) _, map_apply' _ hf _ hs]
      /-
        case pos.h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        δ : Type u_3
        mδ : MeasurableSpace δ
        γ : Type u_4
        mγ : MeasurableSpace γ
        κ : ProbabilityTheory.Kernel α β
        f : β → γ
        g : β → δ
        hg : Measurable g
        hf : Measurable f
        x : α
        s : Set γ
        hs : MeasurableSet s
        ⊢ Eq ((κ x) (Set.preimage (fun x => { fst := f x, snd := g x }) (setOf fun p = …
      -/
    · simp only [Set.preimage, Set.mem_setOf]
      /-
        🎉 no goals
      -/
      /-
        case pos.h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        δ : Type u_3
        mδ : MeasurableSpace δ
        γ : Type u_4
        mγ : MeasurableSpace γ
        κ : ProbabilityTheory.Kernel α β
        f : β → γ
        g : β → δ
        hg : Measurable g
        hf : Measurable f
        x : α
        s : Set γ
        hs : MeasurableSet s
        ⊢ MeasurableSet (setOf fun p => Membership.mem s p.1)
      -/
    · exact measurable_fst hs
      /-
        🎉 no goals
      -/
  · have : ¬ Measurable (fun x ↦ (f x, g x)) := by
      contrapose! hf; exact hf.fst
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hg : Measurable g
      hf : Not (Measurable f)
      this : Not (Measurable fun x => { fst := f x, snd := g x })
      ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).fst (κ.map f)
    -/
    simp [map_of_not_measurable _ hf, map_of_not_measurable _ this]
    /-
      🎉 no goals
    -/


lemma fst_map_id_prod (κ : Kernel α β) {γ : Type*} {mγ : MeasurableSpace γ}
    {f : β → γ} (hf : Measurable f) :
    fst (map κ (fun a ↦ (a, f a))) = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    γ : Type u_5
    mγ : MeasurableSpace γ
    f : β → γ
    hf : Measurable f
    ⊢ Eq (κ.map fun a => { fst := a, snd := f a }).fst κ
  -/
  rw [fst_map_prod _ hf, Kernel.map_id']
  /-
    🎉 no goals
  -/


/-- If `η` is a Markov kernel, use instead `fst_compProd` to get `(κ ⊗ₖ η).fst = κ`. -/
lemma fst_compProd_apply (κ : Kernel α β) (η : Kernel (α × β) γ)
    [IsSFiniteKernel κ] [IsSFiniteKernel η] (x : α) {s : Set β} (hs : MeasurableSet s) :
    (κ ⊗ₖ η).fst x s = ∫⁻ b, s.indicator (fun b ↦ η (x, b) Set.univ) b ∂(κ x) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((κ.compProd η).fst x) s) (MeasureTheory.lintegral (κ x) fun b => s.indi …
  -/
  rw [Kernel.fst_apply' _ _ hs, Kernel.compProd_apply]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => (η { fst := x, snd := b }) (setOf …
  -/
  swap; · exact measurable_fst hs
          /-
            🎉 no goals
          -/
  have h_eq b : η (x, b) {c | b ∈ s} = s.indicator (fun b ↦ η (x, b) Set.univ) b := by
    by_cases hb : b ∈ s <;> simp [hb]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    s : Set β
    hs : MeasurableSet s
    h_eq : ∀ (b : β), Eq ((η { fst := x, snd := b }) (setOf fun c => Membership.me …
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => (η { fst := x, snd := b }) (setOf …
  -/
  simp_rw [Set.mem_setOf_eq, h_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma fst_compProd (κ : Kernel α β) (η : Kernel (α × β) γ) [IsSFiniteKernel κ] [IsMarkovKernel η] :
    fst (κ ⊗ₖ η) = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    ⊢ Eq (κ.compProd η).fst κ
  -/
  ext x s hs; simp [fst_compProd_apply, hs]
              /-
                🎉 no goals
              -/


lemma fst_prodMkLeft (δ : Type*) [MeasurableSpace δ] (κ : Kernel α (β × γ)) :
    fst (prodMkLeft δ κ) = prodMkLeft δ (fst κ) := rfl


lemma fst_prodMkRight (κ : Kernel α (β × γ)) (δ : Type*) [MeasurableSpace δ] :
    fst (prodMkRight δ κ) = prodMkRight δ (fst κ) := rfl


/-- Define a `Kernel α γ` from a `Kernel α (β × γ)` by taking the map of the second projection.
We use `mapOfMeasurable` for better defeqs. -/
noncomputable def snd (κ : Kernel α (β × γ)) : Kernel α γ :=
  mapOfMeasurable κ Prod.snd measurable_snd


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       mα : MeasurableSpace α
                                                                       mβ : MeasurableSpace β
                                                                       γ : Type u_4
                                                                       mγ : MeasurableSpace γ
                                                                       κ : ProbabilityTheory.Kernel α (Prod β γ)
                                                                       ⊢ Eq κ.snd (κ.map Prod.snd)
                                                                     -/
theorem snd_eq (κ : Kernel α (β × γ)) : snd κ = map κ Prod.snd := by simp [snd]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem snd_apply (κ : Kernel α (β × γ)) (a : α) : snd κ a = (κ a).map Prod.snd :=
  rfl


theorem snd_apply' (κ : Kernel α (β × γ)) (a : α) {s : Set γ} (hs : MeasurableSet s) :
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          mα : MeasurableSpace α
                                          mβ : MeasurableSpace β
                                          γ : Type u_4
                                          mγ : MeasurableSpace γ
                                          κ : ProbabilityTheory.Kernel α (Prod β γ)
                                          a : α
                                          s : Set γ
                                          hs : MeasurableSet s
                                          ⊢ Eq ((κ.snd a) s) ((κ a) (setOf fun p => Membership.mem s p.2))
                                        -/
    snd κ a s = κ a {p | p.2 ∈ s} := by rw [snd_apply, Measure.map_apply measurable_snd hs]; rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp]
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        mα : MeasurableSpace α
                                                        mβ : MeasurableSpace β
                                                        γ : Type u_4
                                                        mγ : MeasurableSpace γ
                                                        ⊢ Eq (ProbabilityTheory.Kernel.snd 0) 0
                                                      -/
lemma snd_zero : snd (0 : Kernel α (β × γ)) = 0 := by simp [snd]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem lintegral_snd (κ : Kernel α (β × γ)) (a : α) {g : γ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ c, g c ∂snd κ a = ∫⁻ bc : β × γ, g bc.snd ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    g : γ → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral (κ.snd a) fun c => g c) (MeasureTheory.lintegral …
  -/
  rw [snd_eq, lintegral_map _ measurable_snd a hg]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.snd (κ : Kernel α (β × γ)) [IsMarkovKernel κ] : IsMarkovKernel (snd κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.snd
  -/
  rw [Kernel.snd_eq]; exact IsMarkovKernel.map _ measurable_snd
                      /-
                        🎉 no goals
                      -/


instance IsZeroOrMarkovKernel.snd (κ : Kernel α (β × γ)) [IsZeroOrMarkovKernel κ] :
    IsZeroOrMarkovKernel (snd κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel κ.snd
  -/
  rw [Kernel.snd_eq]; infer_instance
                      /-
                        🎉 no goals
                      -/


instance IsFiniteKernel.snd (κ : Kernel α (β × γ)) [IsFiniteKernel κ] : IsFiniteKernel (snd κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsFiniteKernel κ.snd
  -/
  rw [Kernel.snd_eq]; infer_instance
                      /-
                        🎉 no goals
                      -/


instance IsSFiniteKernel.snd (κ : Kernel α (β × γ)) [IsSFiniteKernel κ] :
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    mα : MeasurableSpace α
                                    mβ : MeasurableSpace β
                                    δ : Type u_3
                                    mδ : MeasurableSpace δ
                                    γ : Type u_4
                                    mγ : MeasurableSpace γ
                                    κ : ProbabilityTheory.Kernel α (Prod β γ)
                                    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                    ⊢ ProbabilityTheory.IsSFiniteKernel κ.snd
                                  -/
    IsSFiniteKernel (snd κ) := by rw [Kernel.snd_eq]; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (priority := 100) isFiniteKernel_of_isFiniteKernel_snd {κ : Kernel α (β × γ)}
    [h : IsFiniteKernel (snd κ)] :
    IsFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.snd
    ⊢ ProbabilityTheory.IsFiniteKernel κ
  -/
  refine ⟨h.bound, h.bound_lt_top, fun a ↦ le_trans ?_ (measure_le_bound (snd κ) a Set.univ)⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.snd
    a : α
    ⊢ LE.le ((κ a) Set.univ) ((κ.snd a) Set.univ)
  -/
  rw [snd_apply' _ _ MeasurableSet.univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    h : ProbabilityTheory.IsFiniteKernel κ.snd
    a : α
    ⊢ LE.le ((κ a) Set.univ) ((κ a) (setOf fun p => Membership.mem Set.univ p.2))
  -/
  simp
  /-
    🎉 no goals
  -/


lemma snd_map_prod (κ : Kernel α β) {f : β → γ} {g : β → δ} (hf : Measurable f) :
    snd (map κ (fun x ↦ (f x, g x))) = map κ g := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    δ : Type u_3
    mδ : MeasurableSpace δ
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    f : β → γ
    g : β → δ
    hf : Measurable f
    ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).snd (κ.map g)
  -/
  by_cases hg : Measurable g
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hf : Measurable f
      hg : Measurable g
      ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).snd (κ.map g)
    -/
  · ext x s hs
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hf : Measurable f
      hg : Measurable g
      x : α
      s : Set δ
      hs : MeasurableSet s
      ⊢ Eq (((κ.map fun x => { fst := f x, snd := g x }).snd x) s) (((κ.map g) x) s)
    -/
    rw [snd_apply' _ _ hs, map_apply' _ (hf.prod hg), map_apply' _ hg _ hs]
      /-
        case pos.h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        δ : Type u_3
        mδ : MeasurableSpace δ
        γ : Type u_4
        mγ : MeasurableSpace γ
        κ : ProbabilityTheory.Kernel α β
        f : β → γ
        g : β → δ
        hf : Measurable f
        hg : Measurable g
        x : α
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq ((κ x) (Set.preimage (fun x => { fst := f x, snd := g x }) (setOf fun p = …
      -/
    · simp only [Set.preimage, Set.mem_setOf]
      /-
        🎉 no goals
      -/
      /-
        case pos.h.h.hs
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        δ : Type u_3
        mδ : MeasurableSpace δ
        γ : Type u_4
        mγ : MeasurableSpace γ
        κ : ProbabilityTheory.Kernel α β
        f : β → γ
        g : β → δ
        hf : Measurable f
        hg : Measurable g
        x : α
        s : Set δ
        hs : MeasurableSet s
        ⊢ MeasurableSet (setOf fun p => Membership.mem s p.2)
      -/
    · exact measurable_snd hs
      /-
        🎉 no goals
      -/
  · have : ¬ Measurable (fun x ↦ (f x, g x)) := by
      contrapose! hg; exact hg.snd
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      δ : Type u_3
      mδ : MeasurableSpace δ
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      f : β → γ
      g : β → δ
      hf : Measurable f
      hg : Not (Measurable g)
      this : Not (Measurable fun x => { fst := f x, snd := g x })
      ⊢ Eq (κ.map fun x => { fst := f x, snd := g x }).snd (κ.map g)
    -/
    simp [map_of_not_measurable _ hg, map_of_not_measurable _ this]
    /-
      🎉 no goals
    -/


lemma snd_map_prod_id (κ : Kernel α β) {γ : Type*} {mγ : MeasurableSpace γ}
    {f : β → γ} (hf : Measurable f) :
    snd (map κ (fun a ↦ (f a, a))) = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    γ : Type u_5
    mγ : MeasurableSpace γ
    f : β → γ
    hf : Measurable f
    ⊢ Eq (κ.map fun a => { fst := f a, snd := a }).snd κ
  -/
  rw [snd_map_prod _ hf, Kernel.map_id']
  /-
    🎉 no goals
  -/


lemma snd_prodMkLeft (δ : Type*) [MeasurableSpace δ] (κ : Kernel α (β × γ)) :
    snd (prodMkLeft δ κ) = prodMkLeft δ (snd κ) := rfl


lemma snd_prodMkRight (κ : Kernel α (β × γ)) (δ : Type*) [MeasurableSpace δ] :
    snd (prodMkRight δ κ) = prodMkRight δ (snd κ) := rfl


@[simp]
lemma fst_swapRight (κ : Kernel α (β × γ)) : fst (swapRight κ) = snd κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    ⊢ Eq κ.swapRight.fst κ.snd
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq ((κ.swapRight.fst a) s) ((κ.snd a) s)
  -/
  rw [fst_apply' _ _ hs, swapRight_apply', snd_apply' _ _ hs]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α (Prod β γ)
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ Eq ((κ a) (setOf fun p => Membership.mem (setOf fun p => Membership.mem s p. …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.h.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α (Prod β γ)
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ MeasurableSet (setOf fun p => Membership.mem s p.1)
    -/
  · exact measurable_fst hs
    /-
      🎉 no goals
    -/


@[simp]
lemma snd_swapRight (κ : Kernel α (β × γ)) : snd (swapRight κ) = fst κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    ⊢ Eq κ.swapRight.snd κ.fst
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((κ.swapRight.snd a) s) ((κ.fst a) s)
  -/
  rw [snd_apply' _ _ hs, swapRight_apply', fst_apply' _ _ hs]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α (Prod β γ)
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ Eq ((κ a) (setOf fun p => Membership.mem (setOf fun p => Membership.mem s p. …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.h.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_4
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α (Prod β γ)
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasurableSet (setOf fun p => Membership.mem s p.2)
    -/
  · exact measurable_snd hs
    /-
      🎉 no goals
    -/


/-- Composition of two kernels. -/
noncomputable def comp (η : Kernel β γ) (κ : Kernel α β) : Kernel α γ where
  toFun a := (κ a).bind η
  measurable' := (Measure.measurable_bind' η.measurable).comp κ.measurable


@[inherit_doc]
scoped[ProbabilityTheory] infixl:100 " ∘ₖ " => ProbabilityTheory.Kernel.comp


theorem comp_apply (η : Kernel β γ) (κ : Kernel α β) (a : α) : (η ∘ₖ κ) a = (κ a).bind η :=
  rfl


theorem comp_apply' (η : Kernel β γ) (κ : Kernel α β) (a : α) {s : Set γ} (hs : MeasurableSet s) :
    (η ∘ₖ κ) a s = ∫⁻ b, η b s ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    κ : ProbabilityTheory.Kernel α β
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (((η.comp κ) a) s) (MeasureTheory.lintegral (κ a) fun b => (η b) s)
  -/
  rw [comp_apply, Measure.bind_apply hs (Kernel.measurable _)]
  /-
    🎉 no goals
  -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             mα : MeasurableSpace α
                                                                             mβ : MeasurableSpace β
                                                                             γ : Type u_3
                                                                             mγ : MeasurableSpace γ
                                                                             κ : ProbabilityTheory.Kernel α β
                                                                             ⊢ Eq (ProbabilityTheory.Kernel.comp 0 κ) 0
                                                                           -/
@[simp] lemma zero_comp (κ : Kernel α β) : (0 : Kernel β γ) ∘ₖ κ = 0 := by ext; simp [comp_apply]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             mα : MeasurableSpace α
                                                                             mβ : MeasurableSpace β
                                                                             γ : Type u_3
                                                                             mγ : MeasurableSpace γ
                                                                             κ : ProbabilityTheory.Kernel β γ
                                                                             ⊢ Eq (κ.comp 0) 0
                                                                           -/
@[simp] lemma comp_zero (κ : Kernel β γ) : κ ∘ₖ (0 : Kernel α β) = 0 := by ext; simp [comp_apply]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem comp_eq_snd_compProd (η : Kernel β γ) [IsSFiniteKernel η] (κ : Kernel α β)
    [IsSFiniteKernel κ] : η ∘ₖ κ = snd (κ ⊗ₖ prodMkLeft α η) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (η.comp κ) (κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η)).snd
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (((η.comp κ) a) s) (((κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η …
  -/
  rw [comp_apply' _ _ _ hs, snd_apply' _ _ hs, compProd_apply]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η b) s) (MeasureTheory.lintegral …
  -/
  swap
    /-
      case h.h.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      η : ProbabilityTheory.Kernel β γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ MeasurableSet (setOf fun p => Membership.mem s p.2)
    -/
  · exact measurable_snd hs
    /-
      🎉 no goals
    -/
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η b) s) (MeasureTheory.lintegral …
  -/
  simp only [Set.mem_setOf_eq, Set.setOf_mem_eq, prodMkLeft_apply' _ _ s]
  /-
    🎉 no goals
  -/


theorem lintegral_comp (η : Kernel β γ) (κ : Kernel α β) (a : α) {g : γ → ℝ≥0∞}
    (hg : Measurable g) : ∫⁻ c, g c ∂(η ∘ₖ κ) a = ∫⁻ b, ∫⁻ c, g c ∂η b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel β γ
    κ : ProbabilityTheory.Kernel α β
    a : α
    g : γ → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral ((η.comp κ) a) fun c => g c) (MeasureTheory.lint …
  -/
  rw [comp_apply, Measure.lintegral_bind (Kernel.measurable _) hg]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.comp (η : Kernel β γ) [IsMarkovKernel η] (κ : Kernel α β)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         mα : MeasurableSpace α
                                                         mβ : MeasurableSpace β
                                                         γ : Type u_3
                                                         δ : Type u_4
                                                         mγ : MeasurableSpace γ
                                                         mδ : MeasurableSpace δ
                                                         f : β → γ
                                                         g : γ → α
                                                         η : ProbabilityTheory.Kernel β γ
                                                         inst✝¹ : ProbabilityTheory.IsMarkovKernel η
                                                         κ : ProbabilityTheory.Kernel α β
                                                         inst✝ : ProbabilityTheory.IsMarkovKernel κ
                                                         ⊢ ProbabilityTheory.IsMarkovKernel (η.comp κ)
                                                       -/
    [IsMarkovKernel κ] : IsMarkovKernel (η ∘ₖ κ) := by rw [comp_eq_snd_compProd]; infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance IsFiniteKernel.comp (η : Kernel β γ) [IsFiniteKernel η] (κ : Kernel α β)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         mα : MeasurableSpace α
                                                         mβ : MeasurableSpace β
                                                         γ : Type u_3
                                                         δ : Type u_4
                                                         mγ : MeasurableSpace γ
                                                         mδ : MeasurableSpace δ
                                                         f : β → γ
                                                         g : γ → α
                                                         η : ProbabilityTheory.Kernel β γ
                                                         inst✝¹ : ProbabilityTheory.IsFiniteKernel η
                                                         κ : ProbabilityTheory.Kernel α β
                                                         inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                                         ⊢ ProbabilityTheory.IsFiniteKernel (η.comp κ)
                                                       -/
    [IsFiniteKernel κ] : IsFiniteKernel (η ∘ₖ κ) := by rw [comp_eq_snd_compProd]; infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance IsSFiniteKernel.comp (η : Kernel β γ) [IsSFiniteKernel η] (κ : Kernel α β)
                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           mα : MeasurableSpace α
                                                           mβ : MeasurableSpace β
                                                           γ : Type u_3
                                                           δ : Type u_4
                                                           mγ : MeasurableSpace γ
                                                           mδ : MeasurableSpace δ
                                                           f : β → γ
                                                           g : γ → α
                                                           η : ProbabilityTheory.Kernel β γ
                                                           inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
                                                           κ : ProbabilityTheory.Kernel α β
                                                           inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                                           ⊢ ProbabilityTheory.IsSFiniteKernel (η.comp κ)
                                                         -/
    [IsSFiniteKernel κ] : IsSFiniteKernel (η ∘ₖ κ) := by rw [comp_eq_snd_compProd]; infer_instance
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- Composition of kernels is associative. -/
theorem comp_assoc {δ : Type*} {mδ : MeasurableSpace δ} (ξ : Kernel γ δ) [IsSFiniteKernel ξ]
    (η : Kernel β γ) (κ : Kernel α β) : ξ ∘ₖ η ∘ₖ κ = ξ ∘ₖ (η ∘ₖ κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    δ : Type u_5
    mδ : MeasurableSpace δ
    ξ : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel ξ
    η : ProbabilityTheory.Kernel β γ
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq ((ξ.comp η).comp κ) (ξ.comp (η.comp κ))
  -/
  refine ext_fun fun a f hf => ?_
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    δ : Type u_5
    mδ : MeasurableSpace δ
    ξ : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel ξ
    η : ProbabilityTheory.Kernel β γ
    κ : ProbabilityTheory.Kernel α β
    a : α
    f : δ → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (((ξ.comp η).comp κ) a) fun b => f b) (MeasureTh …
  -/
  simp_rw [lintegral_comp _ _ _ hf, lintegral_comp _ _ _ hf.lintegral_kernel]
  /-
    🎉 no goals
  -/


theorem deterministic_comp_eq_map (hf : Measurable f) (κ : Kernel α β) :
    deterministic f hf ∘ₖ κ = map κ f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    hf : Measurable f
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq ((ProbabilityTheory.Kernel.deterministic f hf).comp κ) (κ.map f)
  -/
  ext a s hs
  simp_rw [map_apply' _ hf _ hs, comp_apply' _ _ _ hs, deterministic_apply' hf _ hs,
    lintegral_indicator_const_comp hf hs, one_mul]


theorem comp_deterministic_eq_comap (κ : Kernel α β) (hg : Measurable g) :
    κ ∘ₖ deterministic g hg = comap κ g hg := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    g : γ → α
    κ : ProbabilityTheory.Kernel α β
    hg : Measurable g
    ⊢ Eq (κ.comp (ProbabilityTheory.Kernel.deterministic g hg)) (κ.comap g hg)
  -/
  ext a s hs
  simp_rw [comap_apply' _ _ _ s, comp_apply' _ _ _ hs, deterministic_apply hg a,
    lintegral_dirac' _ (Kernel.measurable_coe κ hs)]


lemma deterministic_comp_deterministic (hf : Measurable f) (hg : Measurable g) :
    (deterministic g hg) ∘ₖ (deterministic f hf) = deterministic (g ∘ f) (hg.comp hf) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : β → γ
    g : γ → α
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq ((ProbabilityTheory.Kernel.deterministic g hg).comp (ProbabilityTheory.Ke …
  -/
  ext; simp [comp_deterministic_eq_comap, comap_apply, deterministic_apply]
       /-
         🎉 no goals
       -/


@[simp]
lemma comp_id (κ : Kernel α β) : κ ∘ₖ Kernel.id = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq (κ.comp ProbabilityTheory.Kernel.id) κ
  -/
  rw [Kernel.id, comp_deterministic_eq_comap, comap_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma id_comp (κ : Kernel α β) : Kernel.id ∘ₖ κ = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq (ProbabilityTheory.Kernel.id.comp κ) κ
  -/
  rw [Kernel.id, deterministic_comp_eq_map, map_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_discard (κ : Kernel α β) [IsMarkovKernel κ] : discard β ∘ₖ κ = discard α := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Eq ((ProbabilityTheory.Kernel.discard β).comp κ) (ProbabilityTheory.Kernel.d …
  -/
  ext a s hs; simp [comp_apply' _ _ _ hs]
              /-
                🎉 no goals
              -/


@[simp]
lemma swap_copy : (swap α α) ∘ₖ (copy α) = copy α := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ⊢ Eq ((ProbabilityTheory.Kernel.swap α α).comp (ProbabilityTheory.Kernel.copy  …
  -/
  ext a s hs
  rw [comp_apply, copy_apply, Measure.dirac_bind (Kernel.measurable _), swap_apply' _ hs,
    Measure.dirac_apply' _ hs]
  /-
    case h.h
    α : Type u_1
    mα : MeasurableSpace α
    a : α
    s : Set (Prod α α)
    hs : MeasurableSet s
    ⊢ Eq (s.indicator 1 { fst := a, snd := a }.swap) (s.indicator 1 { fst := a, sn …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
lemma swap_swap : (swap α β) ∘ₖ (swap β α) = Kernel.id := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    ⊢ Eq ((ProbabilityTheory.Kernel.swap α β).comp (ProbabilityTheory.Kernel.swap  …
  -/
  simp_rw [swap, Kernel.deterministic_comp_deterministic, Prod.swap_swap_eq, Kernel.id]
  /-
    🎉 no goals
  -/


lemma swap_comp_eq_map {κ : Kernel α (β × γ)} : (swap β γ) ∘ₖ κ = κ.map Prod.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α (Prod β γ)
    ⊢ Eq ((ProbabilityTheory.Kernel.swap β γ).comp κ) (κ.map Prod.swap)
  -/
  rw [swap, deterministic_comp_eq_map]
  /-
    🎉 no goals
  -/


lemma const_comp (μ : Measure γ) (κ : Kernel α β) :
    const β μ ∘ₖ κ = fun a ↦ (κ a) Set.univ • μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure γ
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq ⇑((ProbabilityTheory.Kernel.const β μ).comp κ) fun a => HSMul.hSMul ((κ a …
  -/
  ext _ _ hs
  simp_rw [comp_apply' _ _ _ hs, const_apply, MeasureTheory.lintegral_const, Measure.smul_apply,
    smul_eq_mul, mul_comm]


@[simp]
lemma const_comp' (μ : Measure γ) (κ : Kernel α β) [IsMarkovKernel κ] :
    const β μ ∘ₖ κ = const α μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure γ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Eq ((ProbabilityTheory.Kernel.const β μ).comp κ) (ProbabilityTheory.Kernel.c …
  -/
  ext; simp_rw [const_comp, measure_univ, one_smul, const_apply]
       /-
         🎉 no goals
       -/


lemma map_comp (κ : Kernel α β) (η : Kernel β γ) (f : γ → δ) :
    (η ∘ₖ κ).map f = (η.map f) ∘ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel β γ
    f : γ → δ
    ⊢ Eq ((η.comp κ).map f) ((η.map f).comp κ)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel β γ
      f : γ → δ
      hf : Measurable f
      ⊢ Eq ((η.comp κ).map f) ((η.map f).comp κ)
    -/
  · ext a s hs
    /-
      case pos.h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel β γ
      f : γ → δ
      hf : Measurable f
      a : α
      s : Set δ
      hs : MeasurableSet s
      ⊢ Eq ((((η.comp κ).map f) a) s) ((((η.map f).comp κ) a) s)
    -/
    rw [map_apply' _ hf _ hs, comp_apply', comp_apply' _ _ _ hs]
      /-
        case pos.h.h
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        δ : Type u_4
        mγ : MeasurableSpace γ
        mδ : MeasurableSpace δ
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel β γ
        f : γ → δ
        hf : Measurable f
        a : α
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η b) (Set.preimage f s)) (Measur …
      -/
    · simp_rw [map_apply' _ hf _ hs]
      /-
        🎉 no goals
      -/
      /-
        case pos.h.h.hs
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        γ : Type u_3
        δ : Type u_4
        mγ : MeasurableSpace γ
        mδ : MeasurableSpace δ
        κ : ProbabilityTheory.Kernel α β
        η : ProbabilityTheory.Kernel β γ
        f : γ → δ
        hf : Measurable f
        a : α
        s : Set δ
        hs : MeasurableSet s
        ⊢ MeasurableSet (Set.preimage f s)
      -/
    · exact hf hs
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel β γ
      f : γ → δ
      hf : Not (Measurable f)
      ⊢ Eq ((η.comp κ).map f) ((η.map f).comp κ)
    -/
  · simp [map_of_not_measurable _ hf]
    /-
      🎉 no goals
    -/


lemma fst_comp (κ : Kernel α β) (η : Kernel β (γ × δ)) : (η ∘ₖ κ).fst = η.fst ∘ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel β (Prod γ δ)
    ⊢ Eq (η.comp κ).fst (η.fst.comp κ)
  -/
  simp [fst_eq, map_comp κ η _]
  /-
    🎉 no goals
  -/


lemma snd_comp (κ : Kernel α β) (η : Kernel β (γ × δ)) : (η ∘ₖ κ).snd = η.snd ∘ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel β (Prod γ δ)
    ⊢ Eq (η.comp κ).snd (η.snd.comp κ)
  -/
  simp_rw [snd_eq, map_comp κ η _]
  /-
    🎉 no goals
  -/


@[simp] lemma snd_compProd_prodMkLeft
    (κ : Kernel α β) (η : Kernel β γ) [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    snd (κ ⊗ₖ prodMkLeft α η) = η ∘ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η)).snd (η.comp κ)
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel β γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (((κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η)).snd a) s) (((η.c …
  -/
  rw [snd_apply' _ _ hs, compProd_apply, comp_apply' _ _ _ hs]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel β γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => ((ProbabilityTheory.Kernel.prodMk …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.h.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      η : ProbabilityTheory.Kernel β γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ MeasurableSet (setOf fun p => Membership.mem s p.2)
    -/
  · exact measurable_snd hs
    /-
      🎉 no goals
    -/


/-- Product of two kernels. This is meaningful only when the kernels are s-finite. -/
noncomputable def prod (κ : Kernel α β) (η : Kernel α γ) : Kernel α (β × γ) :=
  κ ⊗ₖ swapLeft (prodMkLeft β η)


scoped[ProbabilityTheory] infixl:100 " ×ₖ " => ProbabilityTheory.Kernel.prod


theorem prod_apply' (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel α γ) [IsSFiniteKernel η]
    (a : α) {s : Set (β × γ)} (hs : MeasurableSet s) :
    (κ ×ₖ η) a s = ∫⁻ b : β, (η a) {c : γ | (b, c) ∈ s} ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((κ.prod η) a) s) (MeasureTheory.lintegral (κ a) fun b => (η a) (setOf f …
  -/
  simp_rw [prod, compProd_apply hs, swapLeft_apply _ _, prodMkLeft_apply, Prod.swap_prod_mk]
  /-
    🎉 no goals
  -/


lemma prod_apply (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel α γ) [IsSFiniteKernel η]
    (a : α) :
    (κ ×ₖ η) a = (κ a).prod (η a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Eq ((κ.prod η) a) ((κ a).prod (η a))
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((κ.prod η) a) s) (((κ a).prod (η a)) s)
  -/
  rw [prod_apply' _ _ _ hs, Measure.prod_apply hs]
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η a) (setOf fun c => Membership. …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma prod_const (μ : Measure β) [SFinite μ] (ν : Measure γ) [SFinite ν] :
    const α μ ×ₖ const α ν = const α (μ.prod ν) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    ν : MeasureTheory.Measure γ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq ((ProbabilityTheory.Kernel.const α μ).prod (ProbabilityTheory.Kernel.cons …
  -/
  ext x
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    μ : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    ν : MeasureTheory.Measure γ
    inst✝ : MeasureTheory.SFinite ν
    x : α
    s✝ : Set (Prod β γ)
    a✝ : MeasurableSet s✝
    ⊢ Eq ((((ProbabilityTheory.Kernel.const α μ).prod (ProbabilityTheory.Kernel.co …
  -/
  rw [const_apply, prod_apply, const_apply, const_apply]
  /-
    🎉 no goals
  -/


theorem lintegral_prod (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel α γ) [IsSFiniteKernel η]
    (a : α) {g : β × γ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ c, g c ∂(κ ×ₖ η) a = ∫⁻ b, ∫⁻ c, g (b, c) ∂η a ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    g : Prod β γ → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral ((κ.prod η) a) fun c => g c) (MeasureTheory.lint …
  -/
  simp_rw [prod, lintegral_compProd _ _ _ hg, swapLeft_apply, prodMkLeft_apply, Prod.swap_prod_mk]
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.prod (κ : Kernel α β) [IsMarkovKernel κ] (η : Kernel α γ)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         mα : MeasurableSpace α
                                                         mβ : MeasurableSpace β
                                                         γ : Type u_3
                                                         δ : Type u_4
                                                         mγ : MeasurableSpace γ
                                                         mδ : MeasurableSpace δ
                                                         κ : ProbabilityTheory.Kernel α β
                                                         inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
                                                         η : ProbabilityTheory.Kernel α γ
                                                         inst✝ : ProbabilityTheory.IsMarkovKernel η
                                                         ⊢ ProbabilityTheory.IsMarkovKernel (κ.prod η)
                                                       -/
    [IsMarkovKernel η] : IsMarkovKernel (κ ×ₖ η) := by rw [Kernel.prod]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


nonrec instance IsZeroOrMarkovKernel.prod (κ : Kernel α β) [h : IsZeroOrMarkovKernel κ]
    (η : Kernel α γ) [IsZeroOrMarkovKernel η] : IsZeroOrMarkovKernel (κ ×ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    h : ProbabilityTheory.IsZeroOrMarkovKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel η
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.prod η)
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      η : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel η
      h : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.prod 0 η)
    -/
  · simp only [prod, swapLeft_prodMkLeft, compProd_zero_left]; infer_instance
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    h✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel η
    h : ProbabilityTheory.IsMarkovKernel κ
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.prod η)
  -/
  rcases eq_zero_or_isMarkovKernel η with rfl | h'
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      γ : Type u_3
      δ : Type u_4
      mγ : MeasurableSpace γ
      mδ : MeasurableSpace δ
      κ : ProbabilityTheory.Kernel α β
      h✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      h : ProbabilityTheory.IsMarkovKernel κ
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.prod 0)
    -/
  · simp only [prod, swapLeft, prodMkLeft_zero, comap_zero, compProd_zero_right]; infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    h✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel η
    h : ProbabilityTheory.IsMarkovKernel κ
    h' : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.prod η)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance IsFiniteKernel.prod (κ : Kernel α β) [IsFiniteKernel κ] (η : Kernel α γ)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         mα : MeasurableSpace α
                                                         mβ : MeasurableSpace β
                                                         γ : Type u_3
                                                         δ : Type u_4
                                                         mγ : MeasurableSpace γ
                                                         mδ : MeasurableSpace δ
                                                         κ : ProbabilityTheory.Kernel α β
                                                         inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
                                                         η : ProbabilityTheory.Kernel α γ
                                                         inst✝ : ProbabilityTheory.IsFiniteKernel η
                                                         ⊢ ProbabilityTheory.IsFiniteKernel (κ.prod η)
                                                       -/
    [IsFiniteKernel η] : IsFiniteKernel (κ ×ₖ η) := by rw [Kernel.prod]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance IsSFiniteKernel.prod (κ : Kernel α β) (η : Kernel α γ) :
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     mα : MeasurableSpace α
                                     mβ : MeasurableSpace β
                                     γ : Type u_3
                                     δ : Type u_4
                                     mγ : MeasurableSpace γ
                                     mδ : MeasurableSpace δ
                                     κ : ProbabilityTheory.Kernel α β
                                     η : ProbabilityTheory.Kernel α γ
                                     ⊢ ProbabilityTheory.IsSFiniteKernel (κ.prod η)
                                   -/
    IsSFiniteKernel (κ ×ₖ η) := by rw [Kernel.prod]; infer_instance
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma fst_prod (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel α γ) [IsMarkovKernel η] :
    fst (κ ×ₖ η) = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    ⊢ Eq (κ.prod η).fst κ
  -/
  rw [prod]; simp
             /-
               🎉 no goals
             -/


@[simp] lemma snd_prod (κ : Kernel α β) [IsMarkovKernel κ] (η : Kernel α γ) [IsSFiniteKernel η] :
    snd (κ ×ₖ η) = η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.prod η).snd η
  -/
  ext x; simp [snd_apply, prod_apply]
         /-
           🎉 no goals
         -/


lemma comap_prod_swap (κ : Kernel α β) (η : Kernel γ δ) [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    comap (prodMkRight α η ×ₖ prodMkLeft γ κ) Prod.swap measurable_swap
      = map (prodMkRight γ κ ×ₖ prodMkLeft α η) Prod.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (((ProbabilityTheory.Kernel.prodMkRight α η).prod (ProbabilityTheory.Kern …
  -/
  rw [ext_fun_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ∀ (a : Prod α γ) (f : Prod δ β → ENNReal), Measurable f → Eq (MeasureTheory. …
  -/
  intro x f hf
  rw [lintegral_comap, lintegral_map _ measurable_swap _ hf, lintegral_prod _ _ _ hf,
    lintegral_prod]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : Prod α γ
    f : Prod δ β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.prodMkRight α η) x.sw …
  -/
  swap; · exact hf.comp measurable_swap
          /-
            🎉 no goals
          -/
  simp only [prodMkRight_apply, Prod.fst_swap, Prod.swap_prod_mk, lintegral_prodMkLeft,
    Prod.snd_swap]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : Prod α γ
    f : Prod δ β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (η x.2) fun b => MeasureTheory.lintegral (κ x.1) …
  -/
  refine (lintegral_lintegral_swap ?_).symm
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    δ : Type u_4
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : Prod α γ
    f : Prod δ β → ENNReal
    hf : Measurable f
    ⊢ AEMeasurable (Function.uncurry fun c b => f { fst := b, snd := c }) ((κ x.1) …
  -/
  exact (hf.comp measurable_swap).aemeasurable
  /-
    🎉 no goals
  -/


lemma map_prod_swap (κ : Kernel α β) (η : Kernel α γ) [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    map (κ ×ₖ η) Prod.swap = η ×ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((κ.prod η).map Prod.swap) (η.prod κ)
  -/
  rw [ext_fun_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ∀ (a : α) (f : Prod γ β → ENNReal), Measurable f → Eq (MeasureTheory.lintegr …
  -/
  intro x f hf
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    f : Prod γ β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (((κ.prod η).map Prod.swap) x) fun b => f b) (Me …
  -/
  rw [lintegral_map _ measurable_swap _ hf, lintegral_prod, lintegral_prod _ _ _ hf]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    f : Prod γ β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => MeasureTheory.lintegral (η x) fun …
  -/
  swap; · exact hf.comp measurable_swap
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    f : Prod γ β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (κ x) fun b => MeasureTheory.lintegral (η x) fun …
  -/
  refine (lintegral_lintegral_swap ?_).symm
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : α
    f : Prod γ β → ENNReal
    hf : Measurable f
    ⊢ AEMeasurable (Function.uncurry fun c b => f { fst := b, snd := c }.swap) ((η …
  -/
  exact hf.aemeasurable
  /-
    🎉 no goals
  -/


@[simp]
lemma swap_prod {κ : Kernel α β} [IsSFiniteKernel κ] {η : Kernel α γ} [IsSFiniteKernel η] :
    (swap β γ) ∘ₖ (κ ×ₖ η) = (η ×ₖ κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((ProbabilityTheory.Kernel.swap β γ).comp (κ.prod η)) (η.prod κ)
  -/
  rw [swap_comp_eq_map, map_prod_swap]
  /-
    🎉 no goals
  -/


lemma deterministic_prod_deterministic {f : α → β} {g : α → γ}
    (hf : Measurable f) (hg : Measurable g) :
    deterministic f hf ×ₖ deterministic g hg
      = deterministic (fun a ↦ (f a, g a)) (hf.prod_mk hg) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → β
    g : α → γ
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq ((ProbabilityTheory.Kernel.deterministic f hf).prod (ProbabilityTheory.Ke …
  -/
  ext; simp_rw [prod_apply, deterministic_apply, Measure.dirac_prod_dirac]
       /-
         🎉 no goals
       -/


lemma compProd_prodMkLeft_eq_comp
    (κ : Kernel α β) [IsSFiniteKernel κ] (η : Kernel β γ) [IsSFiniteKernel η] :
    κ ⊗ₖ (prodMkLeft α η) = (Kernel.id ×ₖ η) ∘ₖ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η)) ((ProbabilityTheor …
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((κ.compProd (ProbabilityTheory.Kernel.prodMkLeft α η)) a) s) ((((Probab …
  -/
  rw [comp_eq_snd_compProd, compProd_apply hs, snd_apply' _ _ hs, compProd_apply]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => ((ProbabilityTheory.Kernel.prodMk …
  -/
  swap; · exact measurable_snd hs
          /-
            🎉 no goals
          -/
  simp only [prodMkLeft_apply, Set.mem_setOf_eq, Set.setOf_mem_eq, prod_apply' _ _ _ hs,
    id_apply, id_eq]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η b) (setOf fun c => Membership. …
  -/
  congr with b
  /-
    case h.h.e_f.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    b : β
    ⊢ Eq ((η b) (setOf fun c => Membership.mem s { fst := b, snd := c })) (Measure …
  -/
  rw [lintegral_dirac']
  /-
    case h.h.e_f.h.hf
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_3
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel β γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    b : β
    ⊢ Measurable fun b_1 => (η b) (setOf fun c => Membership.mem s { fst := b_1, s …
  -/
  exact measurable_measure_prod_mk_left hs
  /-
    🎉 no goals
  -/


