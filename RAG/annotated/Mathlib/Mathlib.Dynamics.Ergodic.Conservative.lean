/-- We say that a non-singular (`MeasureTheory.QuasiMeasurePreserving`) self-map is
*conservative* if for any measurable set `s` of positive measure there exists `x ∈ s` such that `x`
returns back to `s` under some iteration of `f`. -/
structure Conservative (f : α → α) (μ : Measure α) extends QuasiMeasurePreserving f μ μ : Prop where
  /-- If `f` is a conservative self-map and `s` is a measurable set of nonzero measure,
  then there exists a point `x ∈ s` that returns to `s` under a non-zero iteration of `f`. -/
  exists_mem_iterate_mem' : ∀ ⦃s⦄, MeasurableSet s → μ s ≠ 0 → ∃ x ∈ s, ∃ m ≠ 0, f^[m] x ∈ s


/-- A self-map preserving a finite measure is conservative. -/
protected theorem MeasurePreserving.conservative [IsFiniteMeasure μ] (h : MeasurePreserving f μ μ) :
    Conservative f μ :=
  ⟨h.quasiMeasurePreserving, fun _ hsm h0 => h.exists_mem_iterate_mem hsm.nullMeasurableSet h0⟩


/-- The identity map is conservative w.r.t. any measure. -/
protected theorem id (μ : Measure α) : Conservative id μ :=
  { toQuasiMeasurePreserving := QuasiMeasurePreserving.id μ
    exists_mem_iterate_mem' := fun _ _ h0 => by
      /-
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        x✝¹ : Set α
        x✝ : MeasurableSet x✝¹
        h0 : Ne (μ x✝¹) 0
        ⊢ Exists fun x => And (Membership.mem x✝¹ x) (Exists fun m => And (Ne m 0) (Me …
      -/
      simpa [exists_ne] using nonempty_of_measure_ne_zero h0 }
      /-
        🎉 no goals
      -/


theorem of_absolutelyContinuous {ν : Measure α} (h : Conservative f μ) (hν : ν ≪ μ)
    (h' : QuasiMeasurePreserving f ν ν) : Conservative f ν :=
  ⟨h', fun _ hsm h0 ↦ h.exists_mem_iterate_mem' hsm (mt (@hν _) h0)⟩


/-- Restriction of a conservative system to an invariant set is a conservative system,
formulated in terms of the restriction of the measure. -/
theorem measureRestrict (h : Conservative f μ) (hs : MapsTo f s s) :
    Conservative f (μ.restrict s) :=
  .of_absolutelyContinuous h (absolutelyContinuous_of_le restrict_le_self) <|
    h.toQuasiMeasurePreserving.restrict hs


/-- If `f` is a conservative self-map and `s` is a null measurable set of nonzero measure,
then there exists a point `x ∈ s` that returns to `s` under a non-zero iteration of `f`. -/
theorem exists_mem_iterate_mem (hf : Conservative f μ)
    (hsm : NullMeasurableSet s μ) (hs₀ : μ s ≠ 0) :
    ∃ x ∈ s, ∃ m ≠ 0, f^[m] x ∈ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hsm : MeasureTheory.NullMeasurableSet s μ
    hs₀ : Ne (μ s) 0
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rcases hsm.exists_measurable_subset_ae_eq with ⟨t, hsub, htm, hts⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hsm : MeasureTheory.NullMeasurableSet s μ
    hs₀ : Ne (μ s) 0
    t : Set α
    hsub : HasSubset.Subset t s
    htm : MeasurableSet t
    hts : (MeasureTheory.ae μ).EventuallyEq t s
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rcases hf.exists_mem_iterate_mem' htm (by rwa [measure_congr hts]) with ⟨x, hxt, m, hm₀, hmt⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hsm : MeasureTheory.NullMeasurableSet s μ
    hs₀ : Ne (μ s) 0
    t : Set α
    hsub : HasSubset.Subset t s
    htm : MeasurableSet t
    hts : (MeasureTheory.ae μ).EventuallyEq t s
    x : α
    hxt : Membership.mem t x
    m : Nat
    hm₀ : Ne m 0
    hmt : Membership.mem t (Nat.iterate f m x)
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  exact ⟨x, hsub hxt, m, hm₀, hsub hmt⟩
  /-
    🎉 no goals
  -/


/-- If `f` is a conservative map and `s` is a measurable set of nonzero measure, then
for infinitely many values of `m` a positive measure of points `x ∈ s` returns back to `s`
after `m` iterations of `f`. -/
theorem frequently_measure_inter_ne_zero (hf : Conservative f μ) (hs : NullMeasurableSet s μ)
    (h0 : μ s ≠ 0) : ∃ᶠ m in atTop, μ (s ∩ f^[m] ⁻¹' s) ≠ 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    ⊢ Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.iterate  …
  -/
  set t : ℕ → Set α := fun n ↦ s ∩ f^[n] ⁻¹' s
  -- Assume that `μ (t n) ≠ 0`, where `t n = s ∩ f^[n] ⁻¹' s`, only for finitely many `n`.
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
    ⊢ Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.iterate  …
  -/
  by_contra H
  -- Let `N` be the maximal `n` such that `μ (t n) ≠ 0`.
  obtain ⟨N, hN, hmax⟩ : ∃ N, μ (t N) ≠ 0 ∧ ∀ n > N, μ (t n) = 0 := by
    rw [Nat.frequently_atTop_iff_infinite, not_infinite] at H
    convert exists_max_image _ (·) H ⟨0, by simpa⟩ using 4
    rw [gt_iff_lt, ← not_le, not_imp_comm, mem_setOf]
  have htm {n : ℕ} : NullMeasurableSet (t n) μ :=
    hs.inter <| hs.preimage <| hf.toQuasiMeasurePreserving.iterate n
  -- Then all `t n`, `n > N`, are null sets, hence `T = t N \ ⋃ n > N, t n` has positive measure.
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
    H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
    N : Nat
    hN : Ne (μ (t N)) 0
    hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
    htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
    ⊢ False
  -/
  set T := t N \ ⋃ n > N, t n with hT
  have hμT : μ T ≠ 0 := by
    rwa [hT, measure_diff_null]
    exact (measure_biUnion_null_iff {n | N < n}.to_countable).2 hmax
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
    H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
    N : Nat
    hN : Ne (μ (t N)) 0
    hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
    htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
    T : Set α := SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n)
    hT : Eq T (SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n))
    hμT : Ne (μ T) 0
    ⊢ False
  -/
  have hTm : NullMeasurableSet T μ := htm.diff <| .biUnion {n | N < n}.to_countable fun _ _ ↦ htm
  -- Take `x ∈ T` and `m ≠ 0` such that `f^[m] x ∈ T`.
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
    H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
    N : Nat
    hN : Ne (μ (t N)) 0
    hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
    htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
    T : Set α := SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n)
    hT : Eq T (SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n))
    hμT : Ne (μ T) 0
    hTm : MeasureTheory.NullMeasurableSet T μ
    ⊢ False
  -/
  rcases hf.exists_mem_iterate_mem hTm hμT with ⟨x, hxt, m, hm₀, hmt⟩
  -- Then `N + m > N`, `x ∈ s`, and `f^[N + m] x = f^[N] (f^[m] x) ∈ s`.
  -- This contradicts `x ∈ T ⊆ (⋃ n > N, t n)ᶜ`.
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h0 : Ne (μ s) 0
    t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
    H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
    N : Nat
    hN : Ne (μ (t N)) 0
    hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
    htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
    T : Set α := SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n)
    hT : Eq T (SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n))
    hμT : Ne (μ T) 0
    hTm : MeasureTheory.NullMeasurableSet T μ
    x : α
    hxt : Membership.mem T x
    m : Nat
    hm₀ : Ne m 0
    hmt : Membership.mem T (Nat.iterate f m x)
    ⊢ False
  -/
  refine hxt.2 <| mem_iUnion₂.2 ⟨N + m, ?_, hxt.1.1, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      s : Set α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      hs : MeasureTheory.NullMeasurableSet s μ
      h0 : Ne (μ s) 0
      t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
      H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
      N : Nat
      hN : Ne (μ (t N)) 0
      hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
      htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
      T : Set α := SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n)
      hT : Eq T (SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n))
      hμT : Ne (μ T) 0
      hTm : MeasureTheory.NullMeasurableSet T μ
      x : α
      hxt : Membership.mem T x
      m : Nat
      hm₀ : Ne m 0
      hmt : Membership.mem T (Nat.iterate f m x)
      ⊢ GT.gt (HAdd.hAdd N m) N
    -/
  · simpa [pos_iff_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      s : Set α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      hs : MeasureTheory.NullMeasurableSet s μ
      h0 : Ne (μ s) 0
      t : Nat → Set α := fun n => Inter.inter s (Set.preimage (Nat.iterate f n) s)
      H : Not (Filter.Frequently (fun m => Ne (μ (Inter.inter s (Set.preimage (Nat.i …
      N : Nat
      hN : Ne (μ (t N)) 0
      hmax : ∀ (n : Nat), GT.gt n N → Eq (μ (t n)) 0
      htm : ∀ {n : Nat}, MeasureTheory.NullMeasurableSet (t n) μ
      T : Set α := SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n)
      hT : Eq T (SDiff.sdiff (t N) (Set.iUnion fun n => Set.iUnion fun h => t n))
      hμT : Ne (μ T) 0
      hTm : MeasureTheory.NullMeasurableSet T μ
      x : α
      hxt : Membership.mem T x
      m : Nat
      hm₀ : Ne m 0
      hmt : Membership.mem T (Nat.iterate f m x)
      ⊢ Membership.mem (Set.preimage (Nat.iterate f (HAdd.hAdd N m)) s) x
    -/
  · simpa only [iterate_add] using hmt.1.2
    /-
      🎉 no goals
    -/


/-- If `f` is a conservative map and `s` is a measurable set of nonzero measure, then
for an arbitrarily large `m` a positive measure of points `x ∈ s` returns back to `s`
after `m` iterations of `f`. -/
theorem exists_gt_measure_inter_ne_zero (hf : Conservative f μ) (hs : NullMeasurableSet s μ)
    (h0 : μ s ≠ 0) (N : ℕ) : ∃ m > N, μ (s ∩ f^[m] ⁻¹' s) ≠ 0 :=
  let ⟨m, hm, hmN⟩ :=
    ((hf.frequently_measure_inter_ne_zero hs h0).and_eventually (eventually_gt_atTop N)).exists
  ⟨m, hmN, hm⟩


/-- Poincaré recurrence theorem: given a conservative map `f` and a measurable set `s`, the set
of points `x ∈ s` such that `x` does not return to `s` after `≥ n` iterations has measure zero. -/
theorem measure_mem_forall_ge_image_not_mem_eq_zero (hf : Conservative f μ)
    (hs : NullMeasurableSet s μ) (n : ℕ) :
    μ ({ x ∈ s | ∀ m ≥ n, f^[m] x ∉ s }) = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ⊢ Eq (μ (setOf fun x => And (Membership.mem s x) (∀ (m : Nat), GE.ge m n → Not …
  -/
  by_contra H
  have : NullMeasurableSet (s ∩ { x | ∀ m ≥ n, f^[m] x ∉ s }) μ := by
    simp only [setOf_forall, ← compl_setOf]
    exact hs.inter <| .biInter (to_countable _) fun m _ ↦
      (hs.preimage <| hf.toQuasiMeasurePreserving.iterate m).compl
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    H : Not (Eq (μ (setOf fun x => And (Membership.mem s x) (∀ (m : Nat), GE.ge m  …
    this : MeasureTheory.NullMeasurableSet (Inter.inter s (setOf fun x => ∀ (m : N …
    ⊢ False
  -/
  rcases (hf.exists_gt_measure_inter_ne_zero this H) n with ⟨m, hmn, hm⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    H : Not (Eq (μ (setOf fun x => And (Membership.mem s x) (∀ (m : Nat), GE.ge m  …
    this : MeasureTheory.NullMeasurableSet (Inter.inter s (setOf fun x => ∀ (m : N …
    m : Nat
    hmn : GT.gt m n
    hm : Ne (μ (Inter.inter (Inter.inter s (setOf fun x => ∀ (m : Nat), GE.ge m n  …
    ⊢ False
  -/
  rcases nonempty_of_measure_ne_zero hm with ⟨x, ⟨_, hxn⟩, hxm, -⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    H : Not (Eq (μ (setOf fun x => And (Membership.mem s x) (∀ (m : Nat), GE.ge m  …
    this : MeasureTheory.NullMeasurableSet (Inter.inter s (setOf fun x => ∀ (m : N …
    m : Nat
    hmn : GT.gt m n
    hm : Ne (μ (Inter.inter (Inter.inter s (setOf fun x => ∀ (m : Nat), GE.ge m n  …
    x : α
    left✝ : Membership.mem s x
    hxn : Membership.mem (setOf fun x => ∀ (m : Nat), GE.ge m n → Not (Membership. …
    hxm : Membership.mem s (Nat.iterate f m x)
    ⊢ False
  -/
  exact hxn m hmn.lt.le hxm
  /-
    🎉 no goals
  -/


/-- Poincaré recurrence theorem: given a conservative map `f` and a measurable set `s`,
almost every point `x ∈ s` returns back to `s` infinitely many times. -/
theorem ae_mem_imp_frequently_image_mem (hf : Conservative f μ) (hs : NullMeasurableSet s μ) :
    ∀ᵐ x ∂μ, x ∈ s → ∃ᶠ n in atTop, f^[n] x ∈ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Filter.Eventually (fun x => Membership.mem s x → Filter.Frequently (fun n => …
  -/
  simp only [frequently_atTop, @forall_swap (_ ∈ s), ae_all_iff]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ ∀ (i : Nat), Filter.Eventually (fun a => Membership.mem s a → Exists fun b = …
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ⊢ Filter.Eventually (fun a => Membership.mem s a → Exists fun b => And (GE.ge  …
  -/
  filter_upwards [measure_zero_iff_ae_nmem.1 (hf.measure_mem_forall_ge_image_not_mem_eq_zero hs n)]
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ⊢ ∀ (a : α), Not (And (Membership.mem s a) (∀ (m : Nat), GE.ge m n → Not (Memb …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem inter_frequently_image_mem_ae_eq (hf : Conservative f μ) (hs : NullMeasurableSet s μ) :
    (s ∩ { x | ∃ᶠ n in atTop, f^[n] x ∈ s } : Set α) =ᵐ[μ] s :=
  inter_eventuallyEq_left.2 <| hf.ae_mem_imp_frequently_image_mem hs


theorem measure_inter_frequently_image_mem_eq (hf : Conservative f μ) (hs : NullMeasurableSet s μ) :
    μ (s ∩ { x | ∃ᶠ n in atTop, f^[n] x ∈ s }) = μ s :=
  measure_congr (hf.inter_frequently_image_mem_ae_eq hs)


/-- Poincaré recurrence theorem: if `f` is a conservative dynamical system and `s` is a measurable
set, then for `μ`-a.e. `x`, if the orbit of `x` visits `s` at least once, then it visits `s`
infinitely many times. -/
theorem ae_forall_image_mem_imp_frequently_image_mem (hf : Conservative f μ)
    (hs : NullMeasurableSet s μ) : ∀ᵐ x ∂μ, ∀ k, f^[k] x ∈ s → ∃ᶠ n in atTop, f^[n] x ∈ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Filter.Eventually (fun x => ∀ (k : Nat), Membership.mem s (Nat.iterate f k x …
  -/
  refine ae_all_iff.2 fun k => ?_
  refine (hf.ae_mem_imp_frequently_image_mem
    (hs.preimage <| hf.toQuasiMeasurePreserving.iterate k)).mono fun x hx hk => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    k : Nat
    x : α
    hx : Membership.mem (Set.preimage (Nat.iterate f k) s) x → Filter.Frequently ( …
    hk : Membership.mem s (Nat.iterate f k x)
    ⊢ Filter.Frequently (fun n => Membership.mem s (Nat.iterate f n x)) Filter.atTop
  -/
  rw [← map_add_atTop_eq_nat k, frequently_map]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    k : Nat
    x : α
    hx : Membership.mem (Set.preimage (Nat.iterate f k) s) x → Filter.Frequently ( …
    hk : Membership.mem s (Nat.iterate f k x)
    ⊢ Filter.Frequently (fun a => Membership.mem s (Nat.iterate f (HAdd.hAdd a k)  …
  -/
  refine (hx hk).mono fun n hn => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    k : Nat
    x : α
    hx : Membership.mem (Set.preimage (Nat.iterate f k) s) x → Filter.Frequently ( …
    hk : Membership.mem s (Nat.iterate f k x)
    n : Nat
    hn : Membership.mem (Set.preimage (Nat.iterate f k) s) (Nat.iterate f n x)
    ⊢ Membership.mem s (Nat.iterate f (HAdd.hAdd n k) x)
  -/
  rwa [add_comm, iterate_add_apply]
  /-
    🎉 no goals
  -/


/-- If `f` is a conservative self-map and `s` is a measurable set of positive measure, then
`ae μ`-frequently we have `x ∈ s` and `s` returns to `s` under infinitely many iterations of `f`. -/
theorem frequently_ae_mem_and_frequently_image_mem (hf : Conservative f μ)
    (hs : NullMeasurableSet s μ) (h0 : μ s ≠ 0) : ∃ᵐ x ∂μ, x ∈ s ∧ ∃ᶠ n in atTop, f^[n] x ∈ s :=
  ((frequently_ae_mem_iff.2 h0).and_eventually (hf.ae_mem_imp_frequently_image_mem hs)).mono
    fun _ hx => ⟨hx.1, hx.2 hx.1⟩


/-- Poincaré recurrence theorem. Let `f : α → α` be a conservative dynamical system on a topological
space with second countable topology and measurable open sets. Then almost every point `x : α`
is recurrent: it visits every neighborhood `s ∈ 𝓝 x` infinitely many times. -/
theorem ae_frequently_mem_of_mem_nhds [TopologicalSpace α] [SecondCountableTopology α]
    [OpensMeasurableSpace α] {f : α → α} {μ : Measure α} (h : Conservative f μ) :
    ∀ᵐ x ∂μ, ∀ s ∈ 𝓝 x, ∃ᶠ n in atTop, f^[n] x ∈ s := by
  have : ∀ s ∈ countableBasis α, ∀ᵐ x ∂μ, x ∈ s → ∃ᶠ n in atTop, f^[n] x ∈ s := fun s hs =>
    h.ae_mem_imp_frequently_image_mem (isOpen_of_mem_countableBasis hs).nullMeasurableSet
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : OpensMeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.Conservative f μ
    this : ∀ (s : Set α), Membership.mem (TopologicalSpace.countableBasis α) s → F …
    ⊢ Filter.Eventually (fun x => ∀ (s : Set α), Membership.mem (nhds x) s → Filte …
  -/
  refine ((ae_ball_iff <| countable_countableBasis α).2 this).mono fun x hx s hs => ?_
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : OpensMeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.Conservative f μ
    this : ∀ (s : Set α), Membership.mem (TopologicalSpace.countableBasis α) s → F …
    x : α
    hx : ∀ (i : Set α), Membership.mem (TopologicalSpace.countableBasis α) i → Mem …
    s : Set α
    hs : Membership.mem (nhds x) s
    ⊢ Filter.Frequently (fun n => Membership.mem s (Nat.iterate f n x)) Filter.atTop
  -/
  rcases (isBasis_countableBasis α).mem_nhds_iff.1 hs with ⟨o, hoS, hxo, hos⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : OpensMeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.Conservative f μ
    this : ∀ (s : Set α), Membership.mem (TopologicalSpace.countableBasis α) s → F …
    x : α
    hx : ∀ (i : Set α), Membership.mem (TopologicalSpace.countableBasis α) i → Mem …
    s : Set α
    hs : Membership.mem (nhds x) s
    o : Set α
    hoS : Membership.mem (TopologicalSpace.countableBasis α) o
    hxo : Membership.mem o x
    hos : HasSubset.Subset o s
    ⊢ Filter.Frequently (fun n => Membership.mem s (Nat.iterate f n x)) Filter.atTop
  -/
  exact (hx o hoS hxo).mono fun n hn => hos hn
  /-
    🎉 no goals
  -/


/-- Iteration of a conservative system is a conservative system. -/
protected theorem iterate (hf : Conservative f μ) (n : ℕ) : Conservative f^[n] μ := by
  -- Discharge the trivial case `n = 0`
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    ⊢ MeasureTheory.Conservative (Nat.iterate f n) μ
  -/
  cases' n with n
    /-
      case zero
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      ⊢ MeasureTheory.Conservative (Nat.iterate f 0) μ
    -/
  · exact Conservative.id μ
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    ⊢ MeasureTheory.Conservative (Nat.iterate f (HAdd.hAdd n 1)) μ
  -/
  refine ⟨hf.1.iterate _, fun s hs hs0 => ?_⟩
  rcases (hf.frequently_ae_mem_and_frequently_image_mem hs.nullMeasurableSet hs0).exists
    with ⟨x, _, hx⟩
  /- We take a point `x ∈ s` such that `f^[k] x ∈ s` for infinitely many values of `k`,
    then we choose two of these values `k < l` such that `k ≡ l [MOD (n + 1)]`.
    Then `f^[k] x ∈ s` and `f^[n + 1]^[(l - k) / (n + 1)] (f^[k] x) = f^[l] x ∈ s`. -/
  /-
    case succ.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    s : Set α
    hs : MeasurableSet s
    hs0 : Ne (μ s) 0
    x : α
    left✝ : Membership.mem s x
    hx : Filter.Frequently (fun n => Membership.mem s (Nat.iterate f n x)) Filter. …
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rw [Nat.frequently_atTop_iff_infinite] at hx
  /-
    case succ.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    s : Set α
    hs : MeasurableSet s
    hs0 : Ne (μ s) 0
    x : α
    left✝ : Membership.mem s x
    hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rcases Nat.exists_lt_modEq_of_infinite hx n.succ_pos with ⟨k, hk, l, hl, hkl, hn⟩
  /-
    case succ.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    s : Set α
    hs : MeasurableSet s
    hs0 : Ne (μ s) 0
    x : α
    left✝ : Membership.mem s x
    hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
    k : Nat
    hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
    l : Nat
    hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
    hkl : LT.lt k l
    hn : n.succ.ModEq k l
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  set m := (l - k) / (n + 1)
  have : (n + 1) * m = l - k := by
    apply Nat.mul_div_cancel'
    exact (Nat.modEq_iff_dvd' hkl.le).1 hn
  /-
    case succ.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.Conservative f μ
    n : Nat
    s : Set α
    hs : MeasurableSet s
    hs0 : Ne (μ s) 0
    x : α
    left✝ : Membership.mem s x
    hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
    k : Nat
    hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
    l : Nat
    hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
    hkl : LT.lt k l
    hn : n.succ.ModEq k l
    m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
    this : Eq (HMul.hMul (HAdd.hAdd n 1) m) (HSub.hSub l k)
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  refine ⟨f^[k] x, hk, m, ?_, ?_⟩
    /-
      case succ.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hs0 : Ne (μ s) 0
      x : α
      left✝ : Membership.mem s x
      hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
      k : Nat
      hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
      l : Nat
      hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
      hkl : LT.lt k l
      hn : n.succ.ModEq k l
      m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
      this : Eq (HMul.hMul (HAdd.hAdd n 1) m) (HSub.hSub l k)
      ⊢ Ne m 0
    -/
  · intro hm
    /-
      case succ.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hs0 : Ne (μ s) 0
      x : α
      left✝ : Membership.mem s x
      hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
      k : Nat
      hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
      l : Nat
      hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
      hkl : LT.lt k l
      hn : n.succ.ModEq k l
      m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
      this : Eq (HMul.hMul (HAdd.hAdd n 1) m) (HSub.hSub l k)
      hm : Eq m 0
      ⊢ False
    -/
    rw [hm, mul_zero, eq_comm, tsub_eq_zero_iff_le] at this
    /-
      case succ.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hs0 : Ne (μ s) 0
      x : α
      left✝ : Membership.mem s x
      hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
      k : Nat
      hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
      l : Nat
      hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
      hkl : LT.lt k l
      hn : n.succ.ModEq k l
      m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
      this : LE.le l k
      hm : Eq m 0
      ⊢ False
    -/
    exact this.not_lt hkl
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hs0 : Ne (μ s) 0
      x : α
      left✝ : Membership.mem s x
      hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
      k : Nat
      hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
      l : Nat
      hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
      hkl : LT.lt k l
      hn : n.succ.ModEq k l
      m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
      this : Eq (HMul.hMul (HAdd.hAdd n 1) m) (HSub.hSub l k)
      ⊢ Membership.mem s (Nat.iterate (Nat.iterate f (HAdd.hAdd n 1)) m (Nat.iterate …
    -/
  · rwa [← iterate_mul, this, ← iterate_add_apply, tsub_add_cancel_of_le]
    /-
      case succ.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      hf : MeasureTheory.Conservative f μ
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hs0 : Ne (μ s) 0
      x : α
      left✝ : Membership.mem s x
      hx : (setOf fun n => Membership.mem s (Nat.iterate f n x)).Infinite
      k : Nat
      hk : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) k
      l : Nat
      hl : Membership.mem (setOf fun n => Membership.mem s (Nat.iterate f n x)) l
      hkl : LT.lt k l
      hn : n.succ.ModEq k l
      m : Nat := HDiv.hDiv (HSub.hSub l k) (HAdd.hAdd n 1)
      this : Eq (HMul.hMul (HAdd.hAdd n 1) m) (HSub.hSub l k)
      ⊢ LE.le k l
    -/
    exact hkl.le
    /-
      🎉 no goals
    -/


