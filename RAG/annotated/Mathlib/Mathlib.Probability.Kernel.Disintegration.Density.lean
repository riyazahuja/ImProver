/-- An `ℕ`-indexed martingale that is a density for `κ` with respect to `ν` on the sets in
`countablePartition γ n`. Used to define its limit `ProbabilityTheory.Kernel.density`, which is
a density for those kernels for all measurable sets. -/
noncomputable
def densityProcess (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ) (a : α) (x : γ) (s : Set β) :
    ℝ :=
  (κ a (countablePartitionSet n x ×ˢ s) / ν a (countablePartitionSet n x)).toReal


lemma densityProcess_def (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ) (a : α) (s : Set β) :
    (fun t ↦ densityProcess κ ν n a t s)
      = fun t ↦ (κ a (countablePartitionSet n t ×ˢ s) / ν a (countablePartitionSet n t)).toReal :=
  rfl


lemma measurable_densityProcess_countableFiltration_aux (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    (n : ℕ) {s : Set β} (hs : MeasurableSet s) :
    Measurable[mα.prod (countableFiltration γ n)] (fun (p : α × γ) ↦
      κ p.1 (countablePartitionSet n p.2 ×ˢ s) / ν p.1 (countablePartitionSet n p.2)) := by
  change Measurable[mα.prod (countableFiltration γ n)]
      ((fun (p : α × countablePartition γ n) ↦ κ p.1 (↑p.2 ×ˢ s) / ν p.1 p.2)
        ∘ (fun (p : α × γ) ↦ (p.1, ⟨countablePartitionSet n p.2, countablePartitionSet_mem n p.2⟩)))
  have h1 : @Measurable _ _ (mα.prod ⊤) _
      (fun p : α × countablePartition γ n ↦ κ p.1 (↑p.2 ×ˢ s) / ν p.1 p.2) := by
    refine Measurable.div ?_ ?_
    · refine measurable_from_prod_countable (fun t ↦ ?_)
      exact Kernel.measurable_coe _ ((measurableSet_countablePartition _ t.prop).prod hs)
    · refine measurable_from_prod_countable ?_
      rintro ⟨t, ht⟩
      exact Kernel.measurable_coe _ (measurableSet_countablePartition _ ht)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    s : Set β
    hs : MeasurableSet s
    h1 : Measurable fun p => HDiv.hDiv ((κ p.1) (SProd.sprod (↑p.2) s)) ((ν p.1) ↑ …
    ⊢ Measurable (Function.comp (fun p => HDiv.hDiv ((κ p.1) (SProd.sprod (↑p.2) s …
  -/
  refine h1.comp (measurable_fst.prod_mk ?_)
  change @Measurable (α × γ) (countablePartition γ n) (mα.prod (countableFiltration γ n)) ⊤
    ((fun c ↦ ⟨countablePartitionSet n c, countablePartitionSet_mem n c⟩) ∘ (fun p : α × γ ↦ p.2))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    s : Set β
    hs : MeasurableSet s
    h1 : Measurable fun p => HDiv.hDiv ((κ p.1) (SProd.sprod (↑p.2) s)) ((ν p.1) ↑ …
    ⊢ Measurable (Function.comp (fun c => ⟨MeasurableSpace.countablePartitionSet n …
  -/
  exact (measurable_countablePartitionSet_subtype n ⊤).comp measurable_snd
  /-
    🎉 no goals
  -/


lemma measurable_densityProcess_aux (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    {s : Set β} (hs : MeasurableSet s) :
    Measurable (fun (p : α × γ) ↦
      κ p.1 (countablePartitionSet n p.2 ×ˢ s) / ν p.1 (countablePartitionSet n p.2)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun p => HDiv.hDiv ((κ p.1) (SProd.sprod (MeasurableSpace.countab …
  -/
  refine Measurable.mono (measurable_densityProcess_countableFiltration_aux κ ν n hs) ?_ le_rfl
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    s : Set β
    hs : MeasurableSet s
    ⊢ LE.le (mα.prod (↑(ProbabilityTheory.countableFiltration γ) n)) Prod.instMeas …
  -/
  exact sup_le_sup le_rfl (comap_mono ((countableFiltration γ).le _))
  /-
    🎉 no goals
  -/


lemma measurable_densityProcess (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    {s : Set β} (hs : MeasurableSet s) :
    Measurable (fun (p : α × γ) ↦ densityProcess κ ν n p.1 p.2 s) :=
  (measurable_densityProcess_aux κ ν n hs).ennreal_toReal


lemma measurable_densityProcess_left (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    (x : γ) {s : Set β} (hs : MeasurableSet s) :
    Measurable (fun a ↦ densityProcess κ ν n a x s) :=
  (measurable_densityProcess κ ν n hs).comp (measurable_id.prod_mk measurable_const)


lemma measurable_densityProcess_right (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    {s : Set β} (a : α) (hs : MeasurableSet s) :
    Measurable (fun x ↦ densityProcess κ ν n a x s) :=
  (measurable_densityProcess κ ν n hs).comp (measurable_const.prod_mk measurable_id)


lemma measurable_countableFiltration_densityProcess (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Measurable[countableFiltration γ n] (fun x ↦ densityProcess κ ν n a x s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun x => κ.densityProcess ν n a x s
  -/
  refine @Measurable.ennreal_toReal _ (countableFiltration γ n) _ ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun x => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countable …
  -/
  exact (measurable_densityProcess_countableFiltration_aux κ ν n hs).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma stronglyMeasurable_countableFiltration_densityProcess (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    (n : ℕ) (a : α) {s : Set β} (hs : MeasurableSet s) :
    StronglyMeasurable[countableFiltration γ n] (fun x ↦ densityProcess κ ν n a x s) :=
  (measurable_countableFiltration_densityProcess κ ν n a hs).stronglyMeasurable


lemma adapted_densityProcess (κ : Kernel α (γ × β)) (ν : Kernel α γ) (a : α)
    {s : Set β} (hs : MeasurableSet s) :
    Adapted (countableFiltration γ) (fun n x ↦ densityProcess κ ν n a x s) :=
  fun n ↦ stronglyMeasurable_countableFiltration_densityProcess κ ν n a hs


lemma densityProcess_nonneg (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ)
    (a : α) (x : γ) (s : Set β) :
    0 ≤ densityProcess κ ν n a x s :=
  ENNReal.toReal_nonneg


lemma meas_countablePartitionSet_le_of_fst_le (hκν : fst κ ≤ ν) (n : ℕ) (a : α) (x : γ)
    (s : Set β) :
    κ a (countablePartitionSet n x ×ˢ s) ≤ ν a (countablePartitionSet n x) := by
  calc κ a (countablePartitionSet n x ×ˢ s)
    ≤ fst κ a (countablePartitionSet n x) := by
        rw [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)]
        refine measure_mono (fun x ↦ ?_)
        simp only [mem_prod, mem_setOf_eq, and_imp]
        exact fun h _ ↦ h
  _ ≤ ν a (countablePartitionSet n x) := hκν a _


lemma densityProcess_le_one (hκν : fst κ ≤ ν) (n : ℕ) (a : α) (x : γ) (s : Set β) :
    densityProcess κ ν n a x s ≤ 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le (κ.densityProcess ν n a x s) 1
  -/
  refine ENNReal.toReal_le_of_le_ofReal zero_le_one (ENNReal.div_le_of_le_mul ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) (H …
  -/
  rw [ENNReal.ofReal_one, one_mul]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) (( …
  -/
  exact meas_countablePartitionSet_le_of_fst_le hκν n a x s
  /-
    🎉 no goals
  -/


lemma eLpNorm_densityProcess_le (hκν : fst κ ≤ ν) (n : ℕ) (a : α) (s : Set β) :
    eLpNorm (fun x ↦ densityProcess κ ν n a x s) 1 (ν a) ≤ ν a univ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    s : Set β
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => κ.densityProcess ν n a x s) 1 (ν a))  …
  -/
  refine (eLpNorm_le_of_ae_bound (C := 1) (ae_of_all _ (fun x ↦ ?_))).trans ?_
  · simp only [Real.norm_eq_abs, abs_of_nonneg (densityProcess_nonneg κ ν n a x s),
      densityProcess_le_one hκν n a x s]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      s : Set β
      ⊢ LE.le (HMul.hMul (HPow.hPow ((ν a) Set.univ) (Inv.inv (ENNReal.toReal 1))) ( …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_densityProcess_le := eLpNorm_densityProcess_le


lemma integrable_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν] (n : ℕ)
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Integrable (fun x ↦ densityProcess κ ν n a x s) (ν a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasureTheory.Integrable (fun x => κ.densityProcess ν n a x s) (ν a)
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasureTheory.Memℒp (fun x => κ.densityProcess ν n a x s) 1 (ν a)
  -/
  refine ⟨Measurable.aestronglyMeasurable ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun x => κ.densityProcess ν n a x s
    -/
  · exact measurable_densityProcess_right κ ν n a hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => κ.densityProcess ν n a x s) 1 (ν a))  …
    -/
  · exact (eLpNorm_densityProcess_le hκν n a s).trans_lt (measure_lt_top _ _)
    /-
      🎉 no goals
    -/


lemma setIntegral_densityProcess_of_mem (hκν : fst κ ≤ ν) [hν : IsFiniteKernel ν]
    (n : ℕ) (a : α) {s : Set β} (hs : MeasurableSet s) {u : Set γ}
    (hu : u ∈ countablePartition γ n) :
    ∫ x in u, densityProcess κ ν n a x s ∂(ν a) = (κ a (u ×ˢ s)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict u) fun x => κ.densityProcess ν n  …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict u) fun x => κ.densityProcess ν n  …
  -/
  have hu_meas : MeasurableSet u := measurableSet_countablePartition n hu
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict u) fun x => κ.densityProcess ν n  …
  -/
  simp_rw [densityProcess]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict u) fun x => (HDiv.hDiv ((κ a) (SP …
  -/
  rw [integral_toReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict u) fun a_1 => HDiv.hDiv ((κ a) ( …
  -/
  rotate_left
    /-
      case hfm
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      ⊢ AEMeasurable (fun x => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.counta …
    -/
  · refine Measurable.aemeasurable ?_
    change Measurable ((fun (p : α × _) ↦ κ p.1 (countablePartitionSet n p.2 ×ˢ s)
      / ν p.1 (countablePartitionSet n p.2)) ∘ (fun x ↦ (a, x)))
    /-
      case hfm
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      ⊢ Measurable (Function.comp (fun p => HDiv.hDiv ((κ p.1) (SProd.sprod (Measura …
    -/
    exact (measurable_densityProcess_aux κ ν n hs).comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      ⊢ Filter.Eventually (fun x => LT.lt (HDiv.hDiv ((κ a) (SProd.sprod (Measurable …
    -/
  · refine ae_of_all _ (fun x ↦ ?_)
    /-
      case hf
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      x : γ
      ⊢ LT.lt (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
    by_cases h0 : ν a (countablePartitionSet n x) = 0
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x : γ
        h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
        ⊢ LT.lt (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
      -/
    · suffices κ a (countablePartitionSet n x ×ˢ s) = 0 by simp [h0, this]
      have h0' : fst κ a (countablePartitionSet n x) = 0 :=
        le_antisymm ((hκν a _).trans h0.le) zero_le'
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x : γ
        h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
        h0' : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
        ⊢ Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) 0
      -/
      rw [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)] at h0'
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x : γ
        h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
        h0' : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countableParti …
        ⊢ Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) 0
      -/
      refine measure_mono_null (fun x ↦ ?_) h0'
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x✝ : γ
        h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x✝)) 0
        h0' : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countableParti …
        x : Prod γ β
        ⊢ Membership.mem (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) s)  …
      -/
      simp only [mem_prod, mem_setOf_eq, and_imp]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x✝ : γ
        h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x✝)) 0
        h0' : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countableParti …
        x : Prod γ β
        ⊢ Membership.mem (MeasurableSpace.countablePartitionSet n x✝) x.1 → Membership …
      -/
      exact fun h _ ↦ h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        hν : ProbabilityTheory.IsFiniteKernel ν
        n : Nat
        a : α
        s : Set β
        hs : MeasurableSet s
        u : Set γ
        hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
        this : ProbabilityTheory.IsFiniteKernel κ
        hu_meas : MeasurableSet u
        x : γ
        h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
        ⊢ LT.lt (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
      -/
    · exact ENNReal.div_lt_top (measure_ne_top _ _) h0
      /-
        🎉 no goals
      -/
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict u) fun a_1 => HDiv.hDiv ((κ a) ( …
  -/
  congr
  have : ∫⁻ x in u, κ a (countablePartitionSet n x ×ˢ s) / ν a (countablePartitionSet n x) ∂(ν a)
      = ∫⁻ _ in u, κ a (u ×ˢ s) / ν a u ∂(ν a) := by
    refine setLIntegral_congr_fun hu_meas (ae_of_all _ (fun t ht ↦ ?_))
    rw [countablePartitionSet_of_mem hu ht]
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this✝ : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict u) fun a_1 => HDiv.hDiv ((κ a) ( …
  -/
  rw [this]
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this✝ : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a) (SP …
  -/
  simp only [MeasureTheory.lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter]
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this✝ : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
    ⊢ Eq (HMul.hMul (HDiv.hDiv ((κ a) (SProd.sprod u s)) ((ν a) u)) ((ν a) u)) ((κ …
  -/
  by_cases h0 : ν a u = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      ⊢ Eq (HMul.hMul (HDiv.hDiv ((κ a) (SProd.sprod u s)) ((ν a) u)) ((ν a) u)) ((κ …
    -/
  · simp only [h0, mul_zero]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      ⊢ Eq 0 ((κ a) (SProd.sprod u s))
    -/
    have h0' : fst κ a u = 0 := le_antisymm ((hκν a _).trans h0.le) zero_le'
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      h0' : Eq ((κ.fst a) u) 0
      ⊢ Eq 0 ((κ a) (SProd.sprod u s))
    -/
    rw [fst_apply' _ _ hu_meas] at h0'
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      h0' : Eq ((κ a) (setOf fun p => Membership.mem u p.1)) 0
      ⊢ Eq 0 ((κ a) (SProd.sprod u s))
    -/
    refine (measure_mono_null ?_ h0').symm
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      h0' : Eq ((κ a) (setOf fun p => Membership.mem u p.1)) 0
      ⊢ HasSubset.Subset (SProd.sprod u s) (setOf fun p => Membership.mem u p.1)
    -/
    intro p
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      h0' : Eq ((κ a) (setOf fun p => Membership.mem u p.1)) 0
      p : Prod γ β
      ⊢ Membership.mem (SProd.sprod u s) p → Membership.mem (setOf fun p => Membersh …
    -/
    simp only [mem_prod, mem_setOf_eq, and_imp]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      hν : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      u : Set γ
      hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
      this✝ : ProbabilityTheory.IsFiniteKernel κ
      hu_meas : MeasurableSet u
      this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
      h0 : Eq ((ν a) u) 0
      h0' : Eq ((κ a) (setOf fun p => Membership.mem u p.1)) 0
      p : Prod γ β
      ⊢ Membership.mem u p.1 → Membership.mem s p.2 → Membership.mem u p.1
    -/
    exact fun h _ ↦ h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this✝ : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
    h0 : Not (Eq ((ν a) u) 0)
    ⊢ Eq (HMul.hMul (HDiv.hDiv ((κ a) (SProd.sprod u s)) ((ν a) u)) ((ν a) u)) ((κ …
  -/
  rw [div_eq_mul_inv, mul_assoc, ENNReal.inv_mul_cancel h0, mul_one]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    hν : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    u : Set γ
    hu : Membership.mem (MeasurableSpace.countablePartition γ n) u
    this✝ : ProbabilityTheory.IsFiniteKernel κ
    hu_meas : MeasurableSet u
    this : Eq (MeasureTheory.lintegral ((ν a).restrict u) fun x => HDiv.hDiv ((κ a …
    h0 : Not (Eq ((ν a) u) 0)
    ⊢ Ne ((ν a) u) Top.top
  -/
  exact measure_ne_top _ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_densityProcess_of_mem := setIntegral_densityProcess_of_mem


lemma setIntegral_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (n : ℕ) (a : α) {s : Set β} (hs : MeasurableSet s) {A : Set γ}
    (hA : MeasurableSet[countableFiltration γ n] A) :
    ∫ x in A, densityProcess κ ν n a x s ∂(ν a) = (κ a (A ×ˢ s)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun x => κ.densityProcess ν n  …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun x => κ.densityProcess ν n  …
  -/
  obtain ⟨S, hS_subset, rfl⟩ := (measurableSet_generateFrom_countablePartition_iff _ _).mp hA
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    this : ProbabilityTheory.IsFiniteKernel κ
    S : Finset (Set γ)
    hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
    hA : MeasurableSet (↑S).sUnion
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict (↑S).sUnion) fun x => κ.densityPr …
  -/
  simp_rw [sUnion_eq_iUnion]
  have h_disj : Pairwise (Disjoint on fun i : S ↦ (i : Set γ)) := by
    intro u v huv
    #adaptation_note /-- nightly-2024-03-16
    Previously `Function.onFun` unfolded in the following `simp only`,
    but now needs a `rw`.
    This may be a bug: a no import minimization may be required.
    simp only [Finset.coe_sort_coe, Function.onFun] -/
    rw [Function.onFun]
    refine disjoint_countablePartition (hS_subset (by simp)) (hS_subset (by simp)) ?_
    rwa [ne_eq, ← Subtype.ext_iff]
  rw [integral_iUnion, iUnion_prod_const, measure_iUnion,
      ENNReal.tsum_toReal_eq (fun _ ↦ measure_ne_top _ _)]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ Eq (tsum fun n_1 => MeasureTheory.integral ((ν a).restrict ↑n_1) fun x => κ. …
    -/
  · congr with u
    /-
      case intro.intro.e_f.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      u : ↑↑S
      ⊢ Eq (MeasureTheory.integral ((ν a).restrict ↑u) fun x => κ.densityProcess ν n …
    -/
    rw [setIntegral_densityProcess_of_mem hκν _ _ hs (hS_subset (by simp))]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ Pairwise (Function.onFun Disjoint fun i => SProd.sprod (↑i) s)
    -/
  · intro u v huv
    /-
      case intro.intro.hn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      u v : ↑↑S
      huv : Ne u v
      ⊢ Function.onFun Disjoint (fun i => SProd.sprod (↑i) s) u v
    -/
    simp only [Finset.coe_sort_coe, Set.disjoint_prod, disjoint_self, bot_eq_empty]
    /-
      case intro.intro.hn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      u v : ↑↑S
      huv : Ne u v
      ⊢ Or (Disjoint ↑u ↑v) (Eq s EmptyCollection.emptyCollection)
    -/
    exact Or.inl (h_disj huv)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ ∀ (i : ↑↑S), MeasurableSet (SProd.sprod (↑i) s)
    -/
  · exact fun _ ↦ (measurableSet_countablePartition n (hS_subset (by simp))).prod hs
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hm
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ ∀ (i : ↑↑S), MeasurableSet ↑i
    -/
  · exact fun _ ↦ measurableSet_countablePartition n (hS_subset (by simp))
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hd
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ Pairwise (Function.onFun Disjoint Subtype.val)
    -/
  · exact h_disj
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hfi
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      n : Nat
      a : α
      s : Set β
      hs : MeasurableSet s
      this : ProbabilityTheory.IsFiniteKernel κ
      S : Finset (Set γ)
      hS_subset : HasSubset.Subset (↑S) (MeasurableSpace.countablePartition γ n)
      hA : MeasurableSet (↑S).sUnion
      h_disj : Pairwise (Function.onFun Disjoint fun i => ↑i)
      ⊢ MeasureTheory.IntegrableOn (fun x => κ.densityProcess ν n a x s) (Set.iUnion …
    -/
  · exact (integrable_densityProcess hκν _ _ hs).integrableOn
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_densityProcess := setIntegral_densityProcess


lemma integral_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (n : ℕ) (a : α) {s : Set β} (hs : MeasurableSet s) :
    ∫ x, densityProcess κ ν n a x s ∂(ν a) = (κ a (univ ×ˢ s)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (ν a) fun x => κ.densityProcess ν n a x s) ((κ a) …
  -/
  rw [← setIntegral_univ, setIntegral_densityProcess hκν _ _ hs MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma setIntegral_densityProcess_of_le (hκν : fst κ ≤ ν)
    [IsFiniteKernel ν] {n m : ℕ} (hnm : n ≤ m) (a : α) {s : Set β} (hs : MeasurableSet s)
    {A : Set γ} (hA : MeasurableSet[countableFiltration γ n] A) :
    ∫ x in A, densityProcess κ ν m a x s ∂(ν a) = (κ a (A ×ˢ s)).toReal :=
  setIntegral_densityProcess hκν m a hs ((countableFiltration γ).mono hnm A hA)


@[deprecated (since := "2024-04-17")]
alias set_integral_densityProcess_of_le := setIntegral_densityProcess_of_le


lemma condexp_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    {i j : ℕ} (hij : i ≤ j) (a : α) {s : Set β} (hs : MeasurableSet s) :
    (ν a)[fun x ↦ densityProcess κ ν j a x s | countableFiltration γ i]
      =ᵐ[ν a] fun x ↦ densityProcess κ ν i a x s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    i j : Nat
    hij : LE.le i j
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (ν a)).EventuallyEq (MeasureTheory.condexp (↑(ProbabilityT …
  -/
  refine (ae_eq_condexp_of_forall_setIntegral_eq ?_ ?_ ?_ ?_ ?_).symm
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      i j : Nat
      hij : LE.le i j
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable (fun x => κ.densityProcess ν j a x s) (ν a)
    -/
  · exact integrable_densityProcess hκν j a hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      i j : Nat
      hij : LE.le i j
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ ∀ (s_1 : Set γ), MeasurableSet s_1 → LT.lt ((ν a) s_1) Top.top → MeasureTheo …
    -/
  · exact fun _ _ _ ↦ (integrable_densityProcess hκν _ _ hs).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      i j : Nat
      hij : LE.le i j
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ ∀ (s_1 : Set γ), MeasurableSet s_1 → LT.lt ((ν a) s_1) Top.top → Eq (Measure …
    -/
  · intro x hx _
    rw [setIntegral_densityProcess hκν i a hs hx,
      setIntegral_densityProcess_of_le hκν hij a hs hx]
  · exact StronglyMeasurable.aeStronglyMeasurable'
      (stronglyMeasurable_countableFiltration_densityProcess κ ν i a hs)


lemma martingale_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Martingale (fun n x ↦ densityProcess κ ν n a x s) (countableFiltration γ) (ν a) :=
  ⟨adapted_densityProcess κ ν a hs, fun _ _ h ↦ condexp_densityProcess hκν h a hs⟩


lemma densityProcess_mono_set (hκν : fst κ ≤ ν) (n : ℕ) (a : α) (x : γ)
    {s s' : Set β} (h : s ⊆ s') :
    densityProcess κ ν n a x s ≤ densityProcess κ ν n a x s' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s s' : Set β
    h : HasSubset.Subset s s'
    ⊢ LE.le (κ.densityProcess ν n a x s) (κ.densityProcess ν n a x s')
  -/
  unfold densityProcess
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s s' : Set β
    h : HasSubset.Subset s s'
    ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
  -/
  obtain h₀ | h₀ := eq_or_ne (ν a (countablePartitionSet n x)) 0
    /-
      case inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      h₀ : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
  · simp [h₀]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      h₀ : Ne ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
  · gcongr
    /-
      case inr.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      h₀ : Ne ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Ne (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
    simp only [ne_eq, ENNReal.div_eq_top, h₀, and_false, false_or, not_and, not_not]
    /-
      case inr.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      h₀ : Ne ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s')) Top. …
    -/
    exact eq_top_mono (meas_countablePartitionSet_le_of_fst_le hκν n a x s')
    /-
      🎉 no goals
    -/


lemma densityProcess_mono_kernel_left {κ' : Kernel α (γ × β)} (hκκ' : κ ≤ κ')
    (hκ'ν : fst κ' ≤ ν) (n : ℕ) (a : α) (x : γ) (s : Set β) :
    densityProcess κ ν n a x s ≤ densityProcess κ' ν n a x s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    κ' : ProbabilityTheory.Kernel α (Prod γ β)
    hκκ' : LE.le κ κ'
    hκ'ν : LE.le κ'.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le (κ.densityProcess ν n a x s) (κ'.densityProcess ν n a x s)
  -/
  unfold densityProcess
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    κ' : ProbabilityTheory.Kernel α (Prod γ β)
    hκκ' : LE.le κ κ'
    hκ'ν : LE.le κ'.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
  -/
  by_cases h0 : ν a (countablePartitionSet n x) = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      κ' : ProbabilityTheory.Kernel α (Prod γ β)
      hκκ' : LE.le κ κ'
      hκ'ν : LE.le κ'.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
  · rw [h0, ENNReal.toReal_div, ENNReal.toReal_div]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      κ' : ProbabilityTheory.Kernel α (Prod γ β)
      hκκ' : LE.le κ κ'
      hκ'ν : LE.le κ'.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
    simp
    /-
      🎉 no goals
    -/
  have h_le : κ' a (countablePartitionSet n x ×ˢ s) ≤ ν a (countablePartitionSet n x) :=
    meas_countablePartitionSet_le_of_fst_le hκ'ν n a x s
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    κ' : ProbabilityTheory.Kernel α (Prod γ β)
    hκκ' : LE.le κ κ'
    hκ'ν : LE.le κ'.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
    h_le : LE.le ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x)  …
    ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
  -/
  gcongr
    /-
      case neg.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      κ' : ProbabilityTheory.Kernel α (Prod γ β)
      hκκ' : LE.le κ κ'
      hκ'ν : LE.le κ'.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      h_le : LE.le ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x)  …
      ⊢ Ne (HDiv.hDiv ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n  …
    -/
  · simp only [ne_eq, ENNReal.div_eq_top, h0, and_false, false_or, not_and, not_not]
    /-
      case neg.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      κ' : ProbabilityTheory.Kernel α (Prod γ β)
      hκκ' : LE.le κ κ'
      hκ'ν : LE.le κ'.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      h_le : LE.le ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x)  …
      ⊢ Eq ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) Top. …
    -/
    exact fun h_top ↦ eq_top_mono h_le h_top
    /-
      🎉 no goals
    -/
    /-
      case neg.h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      κ' : ProbabilityTheory.Kernel α (Prod γ β)
      hκκ' : LE.le κ κ'
      hκ'ν : LE.le κ'.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      h_le : LE.le ((κ' a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x)  …
      ⊢ LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) (( …
    -/
  · apply hκκ'
    /-
      🎉 no goals
    -/


lemma densityProcess_antitone_kernel_right {ν' : Kernel α γ}
    (hνν' : ν ≤ ν') (hκν : fst κ ≤ ν) (n : ℕ) (a : α) (x : γ) (s : Set β) :
    densityProcess κ ν' n a x s ≤ densityProcess κ ν n a x s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν ν' : ProbabilityTheory.Kernel α γ
    hνν' : LE.le ν ν'
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    ⊢ LE.le (κ.densityProcess ν' n a x s) (κ.densityProcess ν n a x s)
  -/
  unfold densityProcess
  have h_le : κ a (countablePartitionSet n x ×ˢ s) ≤ ν a (countablePartitionSet n x) :=
    meas_countablePartitionSet_le_of_fst_le hκν n a x s
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν ν' : ProbabilityTheory.Kernel α γ
    hνν' : LE.le ν ν'
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
    ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
  -/
  by_cases h0 : ν a (countablePartitionSet n x) = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν ν' : ProbabilityTheory.Kernel α γ
      hνν' : LE.le ν ν'
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
      h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
  · simp [le_antisymm (h_le.trans h0.le) zero_le', h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν ν' : ProbabilityTheory.Kernel α γ
    hνν' : LE.le ν ν'
    hκν : LE.le κ.fst ν
    n : Nat
    a : α
    x : γ
    s : Set β
    h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
    h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
    ⊢ LE.le (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
  -/
  gcongr
    /-
      case neg.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν ν' : ProbabilityTheory.Kernel α γ
      hνν' : LE.le ν ν'
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Ne (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
  · simp only [ne_eq, ENNReal.div_eq_top, h0, and_false, false_or, not_and, not_not]
    /-
      case neg.hb
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν ν' : ProbabilityTheory.Kernel α γ
      hνν' : LE.le ν ν'
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s)) Top.t …
    -/
    exact fun h_top ↦ eq_top_mono h_le h_top
    /-
      🎉 no goals
    -/
    /-
      case neg.h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν ν' : ProbabilityTheory.Kernel α γ
      hνν' : LE.le ν ν'
      hκν : LE.le κ.fst ν
      n : Nat
      a : α
      x : γ
      s : Set β
      h_le : LE.le ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) s …
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ LE.le ((ν a) (MeasurableSpace.countablePartitionSet n x)) ((ν' a) (Measurabl …
    -/
  · apply hνν'
    /-
      🎉 no goals
    -/


@[simp]
lemma densityProcess_empty (κ : Kernel α (γ × β)) (ν : Kernel α γ) (n : ℕ) (a : α) (x : γ) :
    densityProcess κ ν n a x ∅ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    n : Nat
    a : α
    x : γ
    ⊢ Eq (κ.densityProcess ν n a x EmptyCollection.emptyCollection) 0
  -/
  simp [densityProcess]
  /-
    🎉 no goals
  -/


lemma tendsto_densityProcess_atTop_empty_of_antitone (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    [IsFiniteKernel κ] (n : ℕ) (a : α) (x : γ)
    (seq : ℕ → Set β) (hseq : Antitone seq) (hseq_iInter : ⋂ i, seq i = ∅)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    Tendsto (fun m ↦ densityProcess κ ν n a x (seq m)) atTop
      (𝓝 (densityProcess κ ν n a x ∅)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => κ.densityProcess ν n a x (seq m)) Filter.atTop (nhd …
  -/
  simp_rw [densityProcess]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.cou …
  -/
  by_cases h0 : ν a (countablePartitionSet n x) = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Filter.Tendsto (fun m => (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.cou …
    -/
  · simp_rw [h0, ENNReal.toReal_div]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      h0 : Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Filter.Tendsto (fun m => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.coun …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
    ⊢ Filter.Tendsto (fun m => (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.cou …
  -/
  refine (ENNReal.tendsto_toReal ?_).comp ?_
    /-
      case neg.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Ne (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
  · rw [ne_eq, ENNReal.div_eq_top]
    /-
      case neg.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Not (Or (And (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
    push_neg
    /-
      case neg.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ And (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Empt …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case neg.refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
    ⊢ Filter.Tendsto (fun m => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.coun …
  -/
  refine ENNReal.Tendsto.div_const ?_ (.inr h0)
  have : Tendsto (fun m ↦ κ a (countablePartitionSet n x ×ˢ seq m)) atTop
      (𝓝 ((κ a) (⋂ n_1, countablePartitionSet n x ×ˢ seq n_1))) := by
    apply tendsto_measure_iInter_atTop
    · measurability
    · exact fun _ _ h ↦ prod_mono_right <| hseq h
    · exact ⟨0, measure_ne_top _ _⟩
  /-
    case neg.refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    h0 : Not (Eq ((ν a) (MeasurableSpace.countablePartitionSet n x)) 0)
    this : Filter.Tendsto (fun m => (κ a) (SProd.sprod (MeasurableSpace.countableP …
    ⊢ Filter.Tendsto (fun m => (κ a) (SProd.sprod (MeasurableSpace.countablePartit …
  -/
  simpa only [← prod_iInter, hseq_iInter] using this
  /-
    🎉 no goals
  -/


lemma tendsto_densityProcess_atTop_of_antitone (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    [IsFiniteKernel κ] (n : ℕ) (a : α) (x : γ)
    (seq : ℕ → Set β) (hseq : Antitone seq) (hseq_iInter : ⋂ i, seq i = ∅)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    Tendsto (fun m ↦ densityProcess κ ν n a x (seq m)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => κ.densityProcess ν n a x (seq m)) Filter.atTop (nhd …
  -/
  rw [← densityProcess_empty κ ν n a x]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => κ.densityProcess ν n a x (seq m)) Filter.atTop (nhd …
  -/
  exact tendsto_densityProcess_atTop_empty_of_antitone κ ν n a x seq hseq hseq_iInter hseq_meas
  /-
    🎉 no goals
  -/


lemma tendsto_densityProcess_limitProcess (hκν : fst κ ≤ ν)
    [IsFiniteKernel ν] (a : α) {s : Set β} (hs : MeasurableSet s) :
    ∀ᵐ x ∂(ν a), Tendsto (fun n ↦ densityProcess κ ν n a x s) atTop
      (𝓝 ((countableFiltration γ).limitProcess
      (fun n x ↦ densityProcess κ ν n a x s) (ν a) x)) := by
  refine Submartingale.ae_tendsto_limitProcess (martingale_densityProcess hκν a hs).submartingale
    (R := (ν a univ).toNNReal) (fun n ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => κ.densityProcess ν n a x s) 1 (ν a))  …
  -/
  refine (eLpNorm_densityProcess_le hκν n a s).trans_eq ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ Eq ((ν a) Set.univ) ↑((ν a) Set.univ).toNNReal
  -/
  rw [ENNReal.coe_toNNReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ Ne ((ν a) Set.univ) Top.top
  -/
  exact measure_ne_top _ _
  /-
    🎉 no goals
  -/


lemma memL1_limitProcess_densityProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Memℒp ((countableFiltration γ).limitProcess
      (fun n x ↦ densityProcess κ ν n a x s) (ν a)) 1 (ν a) := by
  refine Submartingale.memℒp_limitProcess (martingale_densityProcess hκν a hs).submartingale
    (R := (ν a univ).toNNReal) (fun n ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => κ.densityProcess ν n a x s) 1 (ν a))  …
  -/
  refine (eLpNorm_densityProcess_le hκν n a s).trans_eq ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ Eq ((ν a) Set.univ) ↑((ν a) Set.univ).toNNReal
  -/
  rw [ENNReal.coe_toNNReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    n : Nat
    ⊢ Ne ((ν a) Set.univ) Top.top
  -/
  exact measure_ne_top _ _
  /-
    🎉 no goals
  -/


lemma tendsto_eLpNorm_one_densityProcess_limitProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Tendsto (fun n ↦ eLpNorm ((fun x ↦ densityProcess κ ν n a x s)
      - (countableFiltration γ).limitProcess (fun n x ↦ densityProcess κ ν n a x s) (ν a))
      1 (ν a)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (fun x => κ.densit …
  -/
  refine Submartingale.tendsto_eLpNorm_one_limitProcess ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Submartingale (fun n x => κ.densityProcess ν n a x s) (Probabi …
    -/
  · exact (martingale_densityProcess hκν a hs).submartingale
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.UniformIntegrable (fun n x => κ.densityProcess ν n a x s) 1 (ν …
    -/
  · refine uniformIntegrable_of le_rfl ENNReal.one_ne_top ?_ ?_
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        s : Set β
        hs : MeasurableSet s
        ⊢ ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (fun x => κ.densityProcess ν …
      -/
    · exact fun n ↦ (measurable_densityProcess_right κ ν n a hs).aestronglyMeasurable
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        s : Set β
        hs : MeasurableSet s
        ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory. …
      -/
    · refine fun ε _ ↦ ⟨2, fun n ↦ le_of_eq_of_le ?_ (?_ : 0 ≤ ENNReal.ofReal ε)⟩
        /-
          case refine_2.refine_2.refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          ⊢ Eq (MeasureTheory.eLpNorm ((setOf fun x => LE.le 2 (NNNorm.nnnorm (κ.density …
        -/
      · suffices {x | 2 ≤ ‖densityProcess κ ν n a x s‖₊} = ∅ by simp [this]
        /-
          case refine_2.refine_2.refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          ⊢ Eq (setOf fun x => LE.le 2 (NNNorm.nnnorm (κ.densityProcess ν n a x s))) Emp …
        -/
        ext x
        /-
          case refine_2.refine_2.refine_1.h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          x : γ
          ⊢ Iff (Membership.mem (setOf fun x => LE.le 2 (NNNorm.nnnorm (κ.densityProcess …
        -/
        simp only [mem_setOf_eq, mem_empty_iff_false, iff_false, not_le]
        /-
          case refine_2.refine_2.refine_1.h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          x : γ
          ⊢ LT.lt (NNNorm.nnnorm (κ.densityProcess ν n a x s)) 2
        -/
        refine (?_ : _ ≤ (1 : ℝ≥0)).trans_lt one_lt_two
        /-
          case refine_2.refine_2.refine_1.h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          x : γ
          ⊢ LE.le (NNNorm.nnnorm (κ.densityProcess ν n a x s)) 1
        -/
        rw [Real.nnnorm_of_nonneg (densityProcess_nonneg _ _ _ _ _ _)]
        /-
          case refine_2.refine_2.refine_1.h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          x : γ
          ⊢ LE.le ⟨κ.densityProcess ν n a x s, ⋯⟩ 1
        -/
        exact mod_cast (densityProcess_le_one hκν _ _ _ _)
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.refine_2
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          inst✝¹ : MeasurableSpace.CountablyGenerated γ
          κ : ProbabilityTheory.Kernel α (Prod γ β)
          ν : ProbabilityTheory.Kernel α γ
          hκν : LE.le κ.fst ν
          inst✝ : ProbabilityTheory.IsFiniteKernel ν
          a : α
          s : Set β
          hs : MeasurableSet s
          ε : Real
          x✝ : LT.lt 0 ε
          n : Nat
          ⊢ LE.le 0 (ENNReal.ofReal ε)
        -/
      · simp
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-07-27")]
alias tendsto_snorm_one_densityProcess_limitProcess :=
  tendsto_eLpNorm_one_densityProcess_limitProcess


lemma tendsto_eLpNorm_one_restrict_densityProcess_limitProcess [IsFiniteKernel ν]
    (hκν : fst κ ≤ ν) (a : α) {s : Set β} (hs : MeasurableSet s) (A : Set γ) :
    Tendsto (fun n ↦ eLpNorm ((fun x ↦ densityProcess κ ν n a x s)
      - (countableFiltration γ).limitProcess (fun n x ↦ densityProcess κ ν n a x s) (ν a))
      1 ((ν a).restrict A)) atTop (𝓝 0) :=
  tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (tendsto_eLpNorm_one_densityProcess_limitProcess hκν a hs) (fun _ ↦ zero_le')
    (fun _ ↦ eLpNorm_restrict_le _ _ _ _)


@[deprecated (since := "2024-07-27")]
alias tendsto_snorm_one_restrict_densityProcess_limitProcess :=
  tendsto_eLpNorm_one_restrict_densityProcess_limitProcess


/-- Density of the kernel `κ` with respect to `ν`. This is a function `α → γ → Set β → ℝ` which
is measurable on `α × γ` for all measurable sets `s : Set β` and satisfies that
`∫ x in A, density κ ν a x s ∂(ν a) = (κ a (A ×ˢ s)).toReal` for all measurable `A : Set γ`. -/
noncomputable
def density (κ : Kernel α (γ × β)) (ν : Kernel α γ) (a : α) (x : γ) (s : Set β) : ℝ :=
  limsup (fun n ↦ densityProcess κ ν n a x s) atTop


lemma density_ae_eq_limitProcess (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    (fun x ↦ density κ ν a x s)
      =ᵐ[ν a] (countableFiltration γ).limitProcess
        (fun n x ↦ densityProcess κ ν n a x s) (ν a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (ν a)).EventuallyEq (fun x => κ.density ν a x s) (MeasureT …
  -/
  filter_upwards [tendsto_densityProcess_limitProcess hκν a hs] with t ht using ht.limsup_eq
  /-
    🎉 no goals
  -/


lemma tendsto_m_density (hκν : fst κ ≤ ν) (a : α) [IsFiniteKernel ν]
    {s : Set β} (hs : MeasurableSet s) :
    ∀ᵐ x ∂(ν a),
      Tendsto (fun n ↦ densityProcess κ ν n a x s) atTop (𝓝 (density κ ν a x s)) := by
  filter_upwards [tendsto_densityProcess_limitProcess hκν a hs, density_ae_eq_limitProcess hκν a hs]
    with t h1 h2 using h2 ▸ h1


lemma measurable_density (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    {s : Set β} (hs : MeasurableSet s) :
    Measurable (fun (p : α × γ) ↦ density κ ν p.1 p.2 s) :=
  .limsup (fun n ↦ measurable_densityProcess κ ν n hs)


lemma measurable_density_left (κ : Kernel α (γ × β)) (ν : Kernel α γ) (x : γ)
    {s : Set β} (hs : MeasurableSet s) :
    Measurable (fun a ↦ density κ ν a x s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    x : γ
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun a => κ.density ν a x s
  -/
  change Measurable ((fun (p : α × γ) ↦ density κ ν p.1 p.2 s) ∘ (fun a ↦ (a, x)))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    x : γ
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable (Function.comp (fun p => κ.density ν p.1 p.2 s) fun a => { fst := …
  -/
  exact (measurable_density κ ν hs).comp measurable_prod_mk_right
  /-
    🎉 no goals
  -/


lemma measurable_density_right (κ : Kernel α (γ × β)) (ν : Kernel α γ)
    {s : Set β} (hs : MeasurableSet s) (a : α) :
    Measurable (fun x ↦ density κ ν a x s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    s : Set β
    hs : MeasurableSet s
    a : α
    ⊢ Measurable fun x => κ.density ν a x s
  -/
  change Measurable ((fun (p : α × γ) ↦ density κ ν p.1 p.2 s) ∘ (fun x ↦ (a, x)))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    s : Set β
    hs : MeasurableSet s
    a : α
    ⊢ Measurable (Function.comp (fun p => κ.density ν p.1 p.2 s) fun x => { fst := …
  -/
  exact (measurable_density κ ν hs).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma density_mono_set (hκν : fst κ ≤ ν) (a : α) (x : γ) {s s' : Set β} (h : s ⊆ s') :
    density κ ν a x s ≤ density κ ν a x s' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    a : α
    x : γ
    s s' : Set β
    h : HasSubset.Subset s s'
    ⊢ LE.le (κ.density ν a x s) (κ.density ν a x s')
  -/
  refine limsup_le_limsup ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      ⊢ Filter.atTop.EventuallyLE (fun n => κ.densityProcess ν n a x s) fun n => κ.d …
    -/
  · exact Eventually.of_forall (fun n ↦ densityProcess_mono_set hκν n a x h)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => κ.d …
    -/
  · exact isCoboundedUnder_le_of_le atTop (fun i ↦ densityProcess_nonneg _ _ _ _ _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s s' : Set β
      h : HasSubset.Subset s s'
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => κ.den …
    -/
  · exact isBoundedUnder_of ⟨1, fun n ↦ densityProcess_le_one hκν _ _ _ _⟩
    /-
      🎉 no goals
    -/


lemma density_nonneg (hκν : fst κ ≤ ν) (a : α) (x : γ) (s : Set β) :
    0 ≤ density κ ν a x s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    a : α
    x : γ
    s : Set β
    ⊢ LE.le 0 (κ.density ν a x s)
  -/
  refine le_limsup_of_frequently_le ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s : Set β
      ⊢ Filter.Frequently (fun x_1 => LE.le 0 (κ.densityProcess ν x_1 a x s)) Filter …
    -/
  · exact Frequently.of_forall (fun n ↦ densityProcess_nonneg _ _ _ _ _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s : Set β
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => κ.den …
    -/
  · exact isBoundedUnder_of ⟨1, fun n ↦ densityProcess_le_one hκν _ _ _ _⟩
    /-
      🎉 no goals
    -/


lemma density_le_one (hκν : fst κ ≤ ν) (a : α) (x : γ) (s : Set β) :
    density κ ν a x s ≤ 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    a : α
    x : γ
    s : Set β
    ⊢ LE.le (κ.density ν a x s) 1
  -/
  refine limsup_le_of_le ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s : Set β
      ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => κ.d …
    -/
  · exact isCoboundedUnder_le_of_le atTop (fun i ↦ densityProcess_nonneg _ _ _ _ _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      x : γ
      s : Set β
      ⊢ Filter.Eventually (fun n => LE.le (κ.densityProcess ν n a x s) 1) Filter.atTop
    -/
  · exact Eventually.of_forall (fun n ↦ densityProcess_le_one hκν _ _ _ _)
    /-
      🎉 no goals
    -/


lemma eLpNorm_density_le (hκν : fst κ ≤ ν) (a : α) (s : Set β) :
    eLpNorm (fun x ↦ density κ ν a x s) 1 (ν a) ≤ ν a univ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    a : α
    s : Set β
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => κ.density ν a x s) 1 (ν a)) ((ν a) Se …
  -/
  refine (eLpNorm_le_of_ae_bound (C := 1) (ae_of_all _ (fun t ↦ ?_))).trans ?_
  · simp only [Real.norm_eq_abs, abs_of_nonneg (density_nonneg hκν a t s),
      density_le_one hκν a t s]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      a : α
      s : Set β
      ⊢ LE.le (HMul.hMul (HPow.hPow ((ν a) Set.univ) (Inv.inv (ENNReal.toReal 1))) ( …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_density_le := eLpNorm_density_le


lemma integrable_density (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    Integrable (fun x ↦ density κ ν a x s) (ν a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasureTheory.Integrable (fun x => κ.density ν a x s) (ν a)
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasureTheory.Memℒp (fun x => κ.density ν a x s) 1 (ν a)
  -/
  refine ⟨Measurable.aestronglyMeasurable ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun x => κ.density ν a x s
    -/
  · exact measurable_density_right κ ν hs a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => κ.density ν a x s) 1 (ν a)) Top.top
    -/
  · exact (eLpNorm_density_le hκν a s).trans_lt (measure_lt_top _ _)
    /-
      🎉 no goals
    -/


lemma tendsto_setIntegral_densityProcess (hκν : fst κ ≤ ν)
    [IsFiniteKernel ν] (a : α) {s : Set β} (hs : MeasurableSet s) (A : Set γ) :
    Tendsto (fun i ↦ ∫ x in A, densityProcess κ ν i a x s ∂(ν a)) atTop
      (𝓝 (∫ x in A, density κ ν a x s ∂(ν a))) := by
  refine tendsto_setIntegral_of_L1' (μ := ν a) (fun x ↦ density κ ν a x s)
    (integrable_density hκν a hs) (F := fun i x ↦ densityProcess κ ν i a x s) (l := atTop)
    (Eventually.of_forall (fun n ↦ integrable_densityProcess hκν _ _ hs)) ?_ A
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    ⊢ Filter.Tendsto (fun i => MeasureTheory.eLpNorm (HSub.hSub ((fun i x => κ.den …
  -/
  refine (tendsto_congr fun n ↦ ?_).mp (tendsto_eLpNorm_one_densityProcess_limitProcess hκν a hs)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    n : Nat
    ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub (fun x => κ.densityProcess ν n a x s) ( …
  -/
  refine eLpNorm_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    n : Nat
    ⊢ (MeasureTheory.ae (ν a)).EventuallyEq (HSub.hSub (fun x => κ.densityProcess  …
  -/
  exact EventuallyEq.rfl.sub (density_ae_eq_limitProcess hκν a hs).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias tendsto_set_integral_densityProcess := tendsto_setIntegral_densityProcess


/-- Auxiliary lemma for `setIntegral_density`. -/
lemma setIntegral_density_of_measurableSet (hκν : fst κ ≤ ν)
    [IsFiniteKernel ν] (n : ℕ) (a : α) {s : Set β} (hs : MeasurableSet s) {A : Set γ}
    (hA : MeasurableSet[countableFiltration γ n] A) :
    ∫ x in A, density κ ν a x s ∂(ν a) = (κ a (A ×ˢ s)).toReal := by
  suffices ∫ x in A, density κ ν a x s ∂(ν a) = ∫ x in A, densityProcess κ ν n a x s ∂(ν a) by
    exact this ▸ setIntegral_densityProcess hκν _ _ hs hA
  suffices ∫ x in A, density κ ν a x s ∂(ν a)
      = limsup (fun i ↦ ∫ x in A, densityProcess κ ν i a x s ∂(ν a)) atTop by
    rw [this, ← limsup_const (α := ℕ) (f := atTop) (∫ x in A, densityProcess κ ν n a x s ∂(ν a)),
      limsup_congr]
    simp only [eventually_atTop]
    refine ⟨n, fun m hnm ↦ ?_⟩
    rw [setIntegral_densityProcess_of_le hκν hnm _ hs hA,
      setIntegral_densityProcess hκν _ _ hs hA]
  -- use L1 convergence
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun x => κ.density ν a x s) (F …
  -/
  have h := tendsto_setIntegral_densityProcess hκν a hs A
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    n : Nat
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    h : Filter.Tendsto (fun i => MeasureTheory.integral ((ν a).restrict A) fun x = …
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun x => κ.density ν a x s) (F …
  -/
  rw [h.limsup_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_density_of_measurableSet := setIntegral_density_of_measurableSet


lemma integral_density (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    ∫ x, density κ ν a x s ∂(ν a) = (κ a (univ ×ˢ s)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (ν a) fun x => κ.density ν a x s) ((κ a) (SProd.s …
  -/
  rw [← setIntegral_univ, setIntegral_density_of_measurableSet hκν 0 a hs MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma setIntegral_density (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) {A : Set γ} (hA : MeasurableSet A) :
    ∫ x in A, density κ ν a x s ∂(ν a) = (κ a (A ×ˢ s)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun x => κ.density ν a x s) (( …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  have hgen : ‹MeasurableSpace γ› =
      .generateFrom {s | ∃ n, MeasurableSet[countableFiltration γ n] s} := by
    rw [setOf_exists, generateFrom_iUnion_measurableSet (countableFiltration γ),
      iSup_countableFiltration]
  have hpi : IsPiSystem {s | ∃ n, MeasurableSet[countableFiltration γ n] s} := by
    rw [setOf_exists]
    exact isPiSystem_iUnion_of_monotone _
      (fun n ↦ @isPiSystem_measurableSet _ (countableFiltration γ n))
      fun _ _ ↦ (countableFiltration γ).mono
  induction A, hA using induction_on_inter hgen hpi with
  | empty => simp
  | basic s hs =>
    rcases hs with ⟨n, hn⟩
    exact setIntegral_density_of_measurableSet hκν n a hs hn
  | compl A hA hA_eq =>
    have h := integral_add_compl hA (integrable_density hκν a hs)
    rw [hA_eq, integral_density hκν a hs] at h
    have : Aᶜ ×ˢ s = univ ×ˢ s \ A ×ˢ s := by
      rw [prod_diff_prod, compl_eq_univ_diff]
      simp
    rw [this, measure_diff (by intro; simp) (hA.prod hs).nullMeasurableSet (measure_ne_top (κ a) _),
      ENNReal.toReal_sub_of_le (measure_mono (by intro x; simp)) (measure_ne_top _ _)]
    rw [eq_tsub_iff_add_eq_of_le, add_comm]
    · exact h
    · gcongr <;> simp
  | iUnion f hf_disj hf h_eq =>
    rw [integral_iUnion hf hf_disj (integrable_density hκν _ hs).integrableOn]
    simp_rw [h_eq]
    rw [← ENNReal.tsum_toReal_eq (fun _ ↦ measure_ne_top _ _)]
    congr
    rw [iUnion_prod_const, measure_iUnion]
    · exact hf_disj.mono fun _ _ h ↦ h.set_prod_left _ _
    · exact fun i ↦ (hf i).prod hs


@[deprecated (since := "2024-04-17")]
alias set_integral_density := setIntegral_density


lemma setLIntegral_density (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) {A : Set γ} (hA : MeasurableSet A) :
    ∫⁻ x in A, ENNReal.ofReal (density κ ν a x s) ∂(ν a) = κ a (A ×ˢ s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict A) fun x => ENNReal.ofReal (κ.de …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    A : Set γ
    hA : MeasurableSet A
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict A) fun x => ENNReal.ofReal (κ.de …
  -/
  rw [← ofReal_integral_eq_lintegral_ofReal]
  · rw [setIntegral_density hκν a hs hA,
      ENNReal.ofReal_toReal (measure_ne_top _ _)]
    /-
      case hfi
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      A : Set γ
      hA : MeasurableSet A
      this : ProbabilityTheory.IsFiniteKernel κ
      ⊢ MeasureTheory.Integrable (fun x => κ.density ν a x s) ((ν a).restrict A)
    -/
  · exact (integrable_density hκν a hs).restrict
    /-
      🎉 no goals
    -/
    /-
      case f_nn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      s : Set β
      hs : MeasurableSet s
      A : Set γ
      hA : MeasurableSet A
      this : ProbabilityTheory.IsFiniteKernel κ
      ⊢ (MeasureTheory.ae ((ν a).restrict A)).EventuallyLE 0 fun x => κ.density ν a  …
    -/
  · exact ae_of_all _ (fun _ ↦ density_nonneg hκν _ _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_density := setLIntegral_density


lemma lintegral_density (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ x, ENNReal.ofReal (density κ ν a x s) ∂(ν a) = κ a (univ ×ˢ s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν a) fun x => ENNReal.ofReal (κ.density ν a x s …
  -/
  rw [← setLIntegral_univ]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict Set.univ) fun x => ENNReal.ofRea …
  -/
  exact setLIntegral_density hκν a hs MeasurableSet.univ
  /-
    🎉 no goals
  -/


lemma tendsto_integral_density_of_monotone (hκν : fst κ ≤ ν) [IsFiniteKernel ν]
    (a : α) (seq : ℕ → Set β) (hseq : Monotone seq) (hseq_iUnion : ⋃ i, seq i = univ)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    Tendsto (fun m ↦ ∫ x, density κ ν a x (seq m) ∂(ν a)) atTop (𝓝 (κ a univ).toReal) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (ν a) fun x => κ.density ν a …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (ν a) fun x => κ.density ν a …
  -/
  simp_rw [integral_density hκν a (hseq_meas _)]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  have h_cont := ENNReal.continuousOn_toReal.continuousAt (x := κ a univ) ?_
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal ((κ a) Set.univ)
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  swap
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      this : ProbabilityTheory.IsFiniteKernel κ
      ⊢ Membership.mem (nhds ((κ a) Set.univ)) (setOf fun a => Ne a Top.top)
    -/
  · rw [mem_nhds_iff]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      this : ProbabilityTheory.IsFiniteKernel κ
      ⊢ Exists fun t => And (HasSubset.Subset t (setOf fun a => Ne a Top.top)) (And  …
    -/
    refine ⟨Iio (κ a univ + 1), fun x hx ↦ ne_top_of_lt (?_ : x < κ a univ + 1), isOpen_Iio, ?_⟩
      /-
        case refine_1.refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
        this : ProbabilityTheory.IsFiniteKernel κ
        x : ENNReal
        hx : Membership.mem (Set.Iio (HAdd.hAdd ((κ a) Set.univ) 1)) x
        ⊢ LT.lt x (HAdd.hAdd ((κ a) Set.univ) 1)
      -/
    · simpa using hx
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
        this : ProbabilityTheory.IsFiniteKernel κ
        ⊢ Membership.mem (Set.Iio (HAdd.hAdd ((κ a) Set.univ) 1)) ((κ a) Set.univ)
      -/
    · simp only [mem_Iio]
      /-
        case refine_1.refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        ν : ProbabilityTheory.Kernel α γ
        hκν : LE.le κ.fst ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
        this : ProbabilityTheory.IsFiniteKernel κ
        ⊢ LT.lt ((κ a) Set.univ) (HAdd.hAdd ((κ a) Set.univ) 1)
      -/
      exact ENNReal.lt_add_right (measure_ne_top _ _) one_ne_zero
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal ((κ a) Set.univ)
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  refine h_cont.tendsto.comp ?_
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal ((κ a) Set.univ)
    ⊢ Filter.Tendsto (fun m => (κ a) (SProd.sprod Set.univ (seq m))) Filter.atTop  …
  -/
  convert tendsto_measure_iUnion_atTop (monotone_const.set_prod hseq)
  /-
    case h.e'_5.h.e'_3.h.e'_6
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal ((κ a) Set.univ)
    ⊢ Eq Set.univ (Set.iUnion fun n => SProd.sprod Set.univ (seq n))
  -/
  rw [← prod_iUnion, hseq_iUnion, univ_prod_univ]
  /-
    🎉 no goals
  -/


lemma tendsto_integral_density_of_antitone (hκν : fst κ ≤ ν) [IsFiniteKernel ν] (a : α)
    (seq : ℕ → Set β) (hseq : Antitone seq) (hseq_iInter : ⋂ i, seq i = ∅)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    Tendsto (fun m ↦ ∫ x, density κ ν a x (seq m) ∂(ν a)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (ν a) fun x => κ.density ν a …
  -/
  have : IsFiniteKernel κ := isFiniteKernel_of_isFiniteKernel_fst (h := isFiniteKernel_of_le hκν)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (ν a) fun x => κ.density ν a …
  -/
  simp_rw [integral_density hκν a (hseq_meas _)]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  rw [← ENNReal.zero_toReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  have h_cont := ENNReal.continuousAt_toReal ENNReal.zero_ne_top
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal 0
    ⊢ Filter.Tendsto (fun m => ((κ a) (SProd.sprod Set.univ (seq m))).toReal) Filt …
  -/
  refine h_cont.tendsto.comp ?_
  have h : Tendsto (fun m ↦ κ a (univ ×ˢ seq m)) atTop
      (𝓝 ((κ a) (⋂ n, (fun m ↦ univ ×ˢ seq m) n))) := by
    apply tendsto_measure_iInter_atTop
    · measurability
    · exact antitone_const.set_prod hseq
    · exact ⟨0, measure_ne_top _ _⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    this : ProbabilityTheory.IsFiniteKernel κ
    h_cont : ContinuousAt ENNReal.toReal 0
    h : Filter.Tendsto (fun m => (κ a) (SProd.sprod Set.univ (seq m))) Filter.atTo …
    ⊢ Filter.Tendsto (fun m => (κ a) (SProd.sprod Set.univ (seq m))) Filter.atTop  …
  -/
  simpa [← prod_iInter, hseq_iInter] using h
  /-
    🎉 no goals
  -/


lemma tendsto_density_atTop_ae_of_antitone (hκν : fst κ ≤ ν) [IsFiniteKernel ν] (a : α)
    (seq : ℕ → Set β) (hseq : Antitone seq) (hseq_iInter : ⋂ i, seq i = ∅)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    ∀ᵐ x ∂(ν a), Tendsto (fun m ↦ density κ ν a x (seq m)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    ν : ProbabilityTheory.Kernel α γ
    hκν : LE.le κ.fst ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Set β
    hseq : Antitone seq
    hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun m => κ.density ν a x (seq m) …
  -/
  refine tendsto_of_integral_tendsto_of_antitone ?_ (integrable_const _) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ ∀ (n : Nat), MeasureTheory.Integrable (fun x => κ.density ν a x (seq n)) (ν a)
    -/
  · exact fun m ↦ integrable_density hκν _ (hseq_meas m)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => κ.density ν …
    -/
  · rw [integral_zero]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => κ.density ν …
    -/
    exact tendsto_integral_density_of_antitone hκν a seq hseq hseq_iInter hseq_meas
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Eventually (fun a_1 => Antitone fun i => κ.density ν a a_1 (seq i)) ( …
    -/
  · exact ae_of_all _ (fun c n m hnm ↦ density_mono_set hκν a c (hseq hnm))
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      ν : ProbabilityTheory.Kernel α γ
      hκν : LE.le κ.fst ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Set β
      hseq : Antitone seq
      hseq_iInter : Eq (Set.iInter fun i => seq i) EmptyCollection.emptyCollection
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Eventually (fun a_1 => ∀ (i : Nat), LE.le 0 (κ.density ν a a_1 (seq i …
    -/
  · exact ae_of_all _ (fun x m ↦ density_nonneg hκν a x (seq m))
    /-
      🎉 no goals
    -/


lemma densityProcess_fst_univ [IsFiniteKernel κ] (n : ℕ) (a : α) (x : γ) :
    densityProcess κ (fst κ) n a x univ
      = if fst κ a (countablePartitionSet n x) = 0 then 0 else 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    ⊢ Eq (κ.densityProcess κ.fst n a x Set.univ) (ite (Eq ((κ.fst a) (MeasurableSp …
  -/
  rw [densityProcess]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    x : γ
    ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      h : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
  · simp only [h]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      h : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
    by_cases h' : κ a (countablePartitionSet n x ×ˢ univ) = 0
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
        h' : Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.un …
        ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
      -/
    · simp [h']
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
        h' : Not (Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) S …
        ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
      -/
    · rw [ENNReal.div_zero h']
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
        h' : Not (Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) S …
        ⊢ Eq Top.top.toReal 0
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      h : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
  · rw [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)]
    have : countablePartitionSet n x ×ˢ univ = {p : γ × β | p.1 ∈ countablePartitionSet n x} := by
      ext x
      simp
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      x : γ
      h : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
      this : Eq (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) ( …
      ⊢ Eq (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
    rw [this, ENNReal.div_self]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
        this : Eq (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) ( …
        ⊢ Eq (ENNReal.toReal 1) 1
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg.h0
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
        this : Eq (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) ( …
        ⊢ Ne ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartition …
      -/
    · rwa [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)] at h
      /-
        🎉 no goals
      -/
      /-
        case neg.hI
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        x : γ
        h : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
        this : Eq (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) ( …
        ⊢ Ne ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartition …
      -/
    · exact measure_ne_top _ _
      /-
        🎉 no goals
      -/


lemma densityProcess_fst_univ_ae (κ : Kernel α (γ × β)) [IsFiniteKernel κ] (n : ℕ) (a : α) :
    ∀ᵐ x ∂(fst κ a), densityProcess κ (fst κ) n a x univ = 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    ⊢ Filter.Eventually (fun x => Eq (κ.densityProcess κ.fst n a x Set.univ) 1) (M …
  -/
  rw [ae_iff]
  have : {x | ¬ densityProcess κ (fst κ) n a x univ = 1}
      ⊆ {x | fst κ a (countablePartitionSet n x) = 0} := by
    intro x hx
    simp only [mem_setOf_eq] at hx ⊢
    rw [densityProcess_fst_univ] at hx
    simpa using hx
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    this : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x  …
    ⊢ Eq ((κ.fst a) (setOf fun a_1 => Not (Eq (κ.densityProcess κ.fst n a a_1 Set. …
  -/
  refine measure_mono_null this ?_
  have : {x | fst κ a (countablePartitionSet n x) = 0}
      ⊆ ⋃ (u) (_ : u ∈ countablePartition γ n) (_ : fst κ a u = 0), u := by
    intro t ht
    simp only [mem_setOf_eq, mem_iUnion, exists_prop] at ht ⊢
    exact ⟨countablePartitionSet n t, countablePartitionSet_mem _ _, ht,
      mem_countablePartitionSet _ _⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
    this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
    ⊢ Eq ((κ.fst a) (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countablePartit …
  -/
  refine measure_mono_null this ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
    this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
    ⊢ Eq ((κ.fst a) (Set.iUnion fun u => Set.iUnion fun x => Set.iUnion fun x => u …
  -/
  rw [measure_biUnion]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      ⊢ Eq (tsum fun p => (κ.fst a) (Set.iUnion fun x => ↑p)) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hs
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      ⊢ (MeasurableSpace.countablePartition γ n).Countable
    -/
  · exact (finite_countablePartition _ _).countable
    /-
      🎉 no goals
    -/
    /-
      case hd
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      ⊢ (MeasurableSpace.countablePartition γ n).PairwiseDisjoint fun u => Set.iUnio …
    -/
  · intro s hs t ht hst
    /-
      case hd
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      s : Set γ
      hs : Membership.mem (MeasurableSpace.countablePartition γ n) s
      t : Set γ
      ht : Membership.mem (MeasurableSpace.countablePartition γ n) t
      hst : Ne s t
      ⊢ Function.onFun Disjoint (fun u => Set.iUnion fun x => u) s t
    -/
    simp only [disjoint_iUnion_right, disjoint_iUnion_left]
    /-
      case hd
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      s : Set γ
      hs : Membership.mem (MeasurableSpace.countablePartition γ n) s
      t : Set γ
      ht : Membership.mem (MeasurableSpace.countablePartition γ n) t
      hst : Ne s t
      ⊢ Eq ((κ.fst a) t) 0 → Eq ((κ.fst a) s) 0 → Disjoint s t
    -/
    exact fun _ _ ↦ disjoint_countablePartition hs ht hst
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      ⊢ ∀ (b : Set γ), Membership.mem (MeasurableSpace.countablePartition γ n) b → M …
    -/
  · intro s hs
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      n : Nat
      a : α
      this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
      this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
      s : Set γ
      hs : Membership.mem (MeasurableSpace.countablePartition γ n) s
      ⊢ MeasurableSet (Set.iUnion fun x => s)
    -/
    by_cases h : fst κ a s = 0
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
        this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
        s : Set γ
        hs : Membership.mem (MeasurableSpace.countablePartition γ n) s
        h : Eq ((κ.fst a) s) 0
        ⊢ MeasurableSet (Set.iUnion fun x => s)
      -/
    · simp [h, measurableSet_countablePartition n hs]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        a : α
        this✝ : HasSubset.Subset (setOf fun x => Not (Eq (κ.densityProcess κ.fst n a x …
        this : HasSubset.Subset (setOf fun x => Eq ((κ.fst a) (MeasurableSpace.countab …
        s : Set γ
        hs : Membership.mem (MeasurableSpace.countablePartition γ n) s
        h : Not (Eq ((κ.fst a) s) 0)
        ⊢ MeasurableSet (Set.iUnion fun x => s)
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


lemma tendsto_densityProcess_fst_atTop_univ_of_monotone (κ : Kernel α (γ × β)) (n : ℕ) (a : α)
    (x : γ) (seq : ℕ → Set β) (hseq : Monotone seq) (hseq_iUnion : ⋃ i, seq i = univ) :
    Tendsto (fun m ↦ densityProcess κ (fst κ) n a x (seq m)) atTop
      (𝓝 (densityProcess κ (fst κ) n a x univ)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    ⊢ Filter.Tendsto (fun m => κ.densityProcess κ.fst n a x (seq m)) Filter.atTop  …
  -/
  simp_rw [densityProcess]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    ⊢ Filter.Tendsto (fun m => (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.cou …
  -/
  refine (ENNReal.tendsto_toReal ?_).comp ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      ⊢ Ne (HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x …
    -/
  · rw [ne_eq, ENNReal.div_eq_top]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      ⊢ Not (Or (And (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet  …
    -/
    push_neg
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      ⊢ And (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set. …
    -/
    simp_rw [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      ⊢ And (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set. …
    -/
    constructor
      /-
        case refine_1.left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        ⊢ Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) …
      -/
    · refine fun h h0 ↦ h (measure_mono_null (fun x ↦ ?_) h0)
      /-
        case refine_1.left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x✝ : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        h : Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Set.un …
        h0 : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartit …
        x : Prod γ β
        ⊢ Membership.mem (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Set …
      -/
      simp only [mem_prod, mem_setOf_eq, and_imp]
      /-
        case refine_1.left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x✝ : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        h : Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Set.un …
        h0 : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartit …
        x : Prod γ β
        ⊢ Membership.mem (MeasurableSpace.countablePartitionSet n x✝) x.1 → Membership …
      -/
      exact fun h _ ↦ h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        ⊢ Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) …
      -/
    · refine fun h_top ↦ eq_top_mono (measure_mono (fun x ↦ ?_)) h_top
      /-
        case refine_1.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x✝ : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        h_top : Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Se …
        x : Prod γ β
        ⊢ Membership.mem (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Set …
      -/
      simp only [mem_prod, mem_setOf_eq, and_imp]
      /-
        case refine_1.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ β)
        n : Nat
        a : α
        x✝ : γ
        seq : Nat → Set β
        hseq : Monotone seq
        hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
        h_top : Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) Se …
        x : Prod γ β
        ⊢ Membership.mem (MeasurableSpace.countablePartitionSet n x✝) x.1 → Membership …
      -/
      exact fun h _ ↦ h
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    ⊢ Filter.Tendsto (fun m => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.coun …
  -/
  by_cases h0 : fst κ a (countablePartitionSet n x) = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0
      ⊢ Filter.Tendsto (fun m => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.coun …
    -/
  · rw [fst_apply' _ _ (measurableSet_countablePartitionSet _ _)] at h0 ⊢
    suffices ∀ m, κ a (countablePartitionSet n x ×ˢ seq m) = 0 by
      simp only [this, h0, ENNReal.zero_div, tendsto_const_nhds_iff]
      suffices κ a (countablePartitionSet n x ×ˢ univ) = 0 by
        simp only [this, ENNReal.zero_div]
      convert h0
      ext x
      simp only [mem_prod, mem_univ, and_true, mem_setOf_eq]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartit …
      ⊢ ∀ (m : Nat), Eq ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n …
    -/
    refine fun m ↦ measure_mono_null (fun x ↦ ?_) h0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x✝ : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartit …
      m : Nat
      x : Prod γ β
      ⊢ Membership.mem (SProd.sprod (MeasurableSpace.countablePartitionSet n x✝) (se …
    -/
    simp only [mem_prod, mem_setOf_eq, and_imp]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x✝ : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Eq ((κ a) (setOf fun p => Membership.mem (MeasurableSpace.countablePartit …
      m : Nat
      x : Prod γ β
      ⊢ Membership.mem (MeasurableSpace.countablePartitionSet n x✝) x.1 → Membership …
    -/
    exact fun h _ ↦ h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    n : Nat
    a : α
    x : γ
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    h0 : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
    ⊢ Filter.Tendsto (fun m => HDiv.hDiv ((κ a) (SProd.sprod (MeasurableSpace.coun …
  -/
  refine ENNReal.Tendsto.div_const ?_ ?_
    /-
      case neg.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Filter.Tendsto (fun m => (κ a) (SProd.sprod (MeasurableSpace.countablePartit …
    -/
  · convert tendsto_measure_iUnion_atTop (monotone_const.set_prod hseq)
    /-
      case h.e'_5.h.e'_3.h.e'_6
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Eq (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.univ) (Set.i …
    -/
    rw [← prod_iUnion, hseq_iUnion]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      n : Nat
      a : α
      x : γ
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      h0 : Not (Eq ((κ.fst a) (MeasurableSpace.countablePartitionSet n x)) 0)
      ⊢ Or (Ne ((κ a) (SProd.sprod (MeasurableSpace.countablePartitionSet n x) Set.u …
    -/
  · exact Or.inr h0
    /-
      🎉 no goals
    -/


lemma tendsto_densityProcess_fst_atTop_ae_of_monotone (κ : Kernel α (γ × β)) [IsFiniteKernel κ]
    (n : ℕ) (a : α) (seq : ℕ → Set β) (hseq : Monotone seq) (hseq_iUnion : ⋃ i, seq i = univ) :
    ∀ᵐ x ∂(fst κ a), Tendsto (fun m ↦ densityProcess κ (fst κ) n a x (seq m)) atTop (𝓝 1) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun m => κ.densityProcess κ.fst  …
  -/
  filter_upwards [densityProcess_fst_univ_ae κ n a] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    x : γ
    hx : Eq (κ.densityProcess κ.fst n a x Set.univ) 1
    ⊢ Filter.Tendsto (fun m => κ.densityProcess κ.fst n a x (seq m)) Filter.atTop  …
  -/
  rw [← hx]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    n : Nat
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    x : γ
    hx : Eq (κ.densityProcess κ.fst n a x Set.univ) 1
    ⊢ Filter.Tendsto (fun m => κ.densityProcess κ.fst n a x (seq m)) Filter.atTop  …
  -/
  exact tendsto_densityProcess_fst_atTop_univ_of_monotone κ n a x seq hseq hseq_iUnion
  /-
    🎉 no goals
  -/


lemma density_fst_univ (κ : Kernel α (γ × β)) [IsFiniteKernel κ] (a : α) :
    ∀ᵐ x ∂(fst κ a), density κ (fst κ) a x univ = 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    ⊢ Filter.Eventually (fun x => Eq (κ.density κ.fst a x Set.univ) 1) (MeasureThe …
  -/
  have h := fun n ↦ densityProcess_fst_univ_ae κ n a
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    h : ∀ (n : Nat), Filter.Eventually (fun x => Eq (κ.densityProcess κ.fst n a x  …
    ⊢ Filter.Eventually (fun x => Eq (κ.density κ.fst a x Set.univ) 1) (MeasureThe …
  -/
  rw [← ae_all_iff] at h
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    h : Filter.Eventually (fun a_1 => ∀ (i : Nat), Eq (κ.densityProcess κ.fst i a  …
    ⊢ Filter.Eventually (fun x => Eq (κ.density κ.fst a x Set.univ) 1) (MeasureThe …
  -/
  filter_upwards [h] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    h : Filter.Eventually (fun a_1 => ∀ (i : Nat), Eq (κ.densityProcess κ.fst i a  …
    x : γ
    hx : ∀ (i : Nat), Eq (κ.densityProcess κ.fst i a x Set.univ) 1
    ⊢ Eq (κ.density κ.fst a x Set.univ) 1
  -/
  simp [density, hx]
  /-
    🎉 no goals
  -/


lemma tendsto_density_fst_atTop_ae_of_monotone [IsFiniteKernel κ]
    (a : α) (seq : ℕ → Set β) (hseq : Monotone seq) (hseq_iUnion : ⋃ i, seq i = univ)
    (hseq_meas : ∀ m, MeasurableSet (seq m)) :
    ∀ᵐ x ∂(fst κ a), Tendsto (fun m ↦ density κ (fst κ) a x (seq m)) atTop (𝓝 1) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ β)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    seq : Nat → Set β
    hseq : Monotone seq
    hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
    hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun m => κ.density κ.fst a x (se …
  -/
  refine tendsto_of_integral_tendsto_of_monotone ?_ (integrable_const _) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ ∀ (n : Nat), MeasureTheory.Integrable (fun x => κ.density κ.fst a x (seq n)) …
    -/
  · exact fun m ↦ integrable_density le_rfl _ (hseq_meas m)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (κ.fst a) fun a_1 => κ.densi …
    -/
  · rw [MeasureTheory.integral_const, smul_eq_mul, mul_one]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (κ.fst a) fun a_1 => κ.densi …
    -/
    convert tendsto_integral_density_of_monotone (κ := κ) le_rfl a seq hseq hseq_iUnion hseq_meas
    /-
      case h.e'_5.h.e'_3.h.e'_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Eq ((κ.fst a) Set.univ) ((κ a) Set.univ)
    -/
    rw [fst_apply' _ _ MeasurableSet.univ]
    /-
      case h.e'_5.h.e'_3.h.e'_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Eq ((κ a) (setOf fun p => Membership.mem Set.univ p.1)) ((κ a) Set.univ)
    -/
    simp only [mem_univ, setOf_true]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Eventually (fun a_1 => Monotone fun i => κ.density κ.fst a a_1 (seq i …
    -/
  · exact ae_of_all _ (fun c n m hnm ↦ density_mono_set le_rfl a c (hseq hnm))
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ β)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      seq : Nat → Set β
      hseq : Monotone seq
      hseq_iUnion : Eq (Set.iUnion fun i => seq i) Set.univ
      hseq_meas : ∀ (m : Nat), MeasurableSet (seq m)
      ⊢ Filter.Eventually (fun a_1 => ∀ (i : Nat), LE.le (κ.density κ.fst a a_1 (seq …
    -/
  · exact ae_of_all _ (fun x m ↦ density_le_one le_rfl a x (seq m))
    /-
      🎉 no goals
    -/


