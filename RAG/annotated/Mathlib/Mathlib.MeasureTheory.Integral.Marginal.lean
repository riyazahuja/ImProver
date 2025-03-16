/-- Integrate `f(x₁,…,xₙ)` over all variables `xᵢ` where `i ∈ s`. Return a function in the
  remaining variables (it will be constant in the `xᵢ` for `i ∈ s`).
  This is the marginal distribution of all variables not in `s` when the considered measure
  is the product measure. -/
def lmarginal (μ : ∀ i, Measure (π i)) (s : Finset δ) (f : (∀ i, π i) → ℝ≥0∞)
    (x : ∀ i, π i) : ℝ≥0∞ :=
  ∫⁻ y : ∀ i : s, π i, f (updateFinset x s y) ∂Measure.pi fun i : s => μ i

-- Note: this notation is not a binder. This is more convenient since it returns a function.

@[inherit_doc]
notation "∫⋯∫⁻_" s ", " f " ∂" μ:70 => lmarginal μ s f


@[inherit_doc]
notation "∫⋯∫⁻_" s ", " f => lmarginal (fun _ ↦ volume) s f


theorem _root_.Measurable.lmarginal [∀ i, SigmaFinite (μ i)] (hf : Measurable f) :
    Measurable (∫⋯∫⁻_s, f ∂μ) := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    f : ((i : δ) → π i) → ENNReal
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    hf : Measurable f
    ⊢ Measurable (MeasureTheory.lmarginal μ s f)
  -/
  refine Measurable.lintegral_prod_right ?_
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    f : ((i : δ) → π i) → ENNReal
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    hf : Measurable f
    ⊢ Measurable (Function.uncurry fun x y => f (Function.updateFinset x s y))
  -/
  refine hf.comp ?_
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    f : ((i : δ) → π i) → ENNReal
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    hf : Measurable f
    ⊢ Measurable fun a => Function.updateFinset a.1 s a.2
  -/
  rw [measurable_pi_iff]; intro i
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    f : ((i : δ) → π i) → ENNReal
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    hf : Measurable f
    i : δ
    ⊢ Measurable fun x => Function.updateFinset x.1 s x.2 i
  -/
  by_cases hi : i ∈ s
    /-
      case pos
      δ : Type u_1
      π : δ → Type u_3
      inst✝² : (x : δ) → MeasurableSpace (π x)
      μ : (i : δ) → MeasureTheory.Measure (π i)
      inst✝¹ : DecidableEq δ
      s : Finset δ
      f : ((i : δ) → π i) → ENNReal
      inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
      hf : Measurable f
      i : δ
      hi : Membership.mem s i
      ⊢ Measurable fun x => Function.updateFinset x.1 s x.2 i
    -/
  · simpa [hi, updateFinset] using measurable_pi_iff.1 measurable_snd _
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ : Type u_1
      π : δ → Type u_3
      inst✝² : (x : δ) → MeasurableSpace (π x)
      μ : (i : δ) → MeasureTheory.Measure (π i)
      inst✝¹ : DecidableEq δ
      s : Finset δ
      f : ((i : δ) → π i) → ENNReal
      inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
      hf : Measurable f
      i : δ
      hi : Not (Membership.mem s i)
      ⊢ Measurable fun x => Function.updateFinset x.1 s x.2 i
    -/
  · simpa [hi, updateFinset] using measurable_pi_iff.1 measurable_fst _
    /-
      🎉 no goals
    -/


@[simp] theorem lmarginal_empty (f : (∀ i, π i) → ℝ≥0∞) : ∫⋯∫⁻_∅, f ∂μ = f := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    ⊢ Eq (MeasureTheory.lmarginal μ EmptyCollection.emptyCollection f) f
  -/
  ext1 x
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lmarginal μ EmptyCollection.emptyCollection f x) (f x)
  -/
  simp_rw [lmarginal, Measure.pi_of_empty fun i : (∅ : Finset δ) => μ i]
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac fun a => isEmptyEli …
  -/
  apply lintegral_dirac'
  /-
    case h.hf
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    ⊢ Measurable fun a => f (Function.updateFinset x EmptyCollection.emptyCollecti …
  -/
  exact Subsingleton.measurable
  /-
    🎉 no goals
  -/


/-- The marginal distribution is independent of the variables in `s`. -/
theorem lmarginal_congr {x y : ∀ i, π i} (f : (∀ i, π i) → ℝ≥0∞)
    (h : ∀ i ∉ s, x i = y i) :
    (∫⋯∫⁻_s, f ∂μ) x = (∫⋯∫⁻_s, f ∂μ) y := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    s : Finset δ
    x y : (i : δ) → π i
    f : ((i : δ) → π i) → ENNReal
    h : ∀ (i : δ), Not (Membership.mem s i) → Eq (x i) (y i)
    ⊢ Eq (MeasureTheory.lmarginal μ s f x) (MeasureTheory.lmarginal μ s f y)
  -/
  dsimp [lmarginal, updateFinset_def]; rcongr; exact h _ ‹_›
                                               /-
                                                 🎉 no goals
                                               -/


theorem lmarginal_update_of_mem {i : δ} (hi : i ∈ s)
    (f : (∀ i, π i) → ℝ≥0∞) (x : ∀ i, π i) (y : π i) :
    (∫⋯∫⁻_s, f ∂μ) (Function.update x i y) = (∫⋯∫⁻_s, f ∂μ) x := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    s : Finset δ
    i : δ
    hi : Membership.mem s i
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    y : π i
    ⊢ Eq (MeasureTheory.lmarginal μ s f (Function.update x i y)) (MeasureTheory.lm …
  -/
  apply lmarginal_congr
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    s : Finset δ
    i : δ
    hi : Membership.mem s i
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    y : π i
    ⊢ ∀ (i_1 : δ), Not (Membership.mem s i_1) → Eq (Function.update x i y i_1) (x  …
  -/
  intro j hj
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    s : Finset δ
    i : δ
    hi : Membership.mem s i
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    y : π i
    j : δ
    hj : Not (Membership.mem s j)
    ⊢ Eq (Function.update x i y j) (x j)
  -/
  have : j ≠ i := by rintro rfl; exact hj hi
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    s : Finset δ
    i : δ
    hi : Membership.mem s i
    f : ((i : δ) → π i) → ENNReal
    x : (i : δ) → π i
    y : π i
    j : δ
    hj : Not (Membership.mem s j)
    this : Ne j i
    ⊢ Eq (Function.update x i y j) (x j)
  -/
  apply update_of_ne this
  /-
    🎉 no goals
  -/


variable {μ} in
theorem lmarginal_singleton (f : (∀ i, π i) → ℝ≥0∞) (i : δ) :
    ∫⋯∫⁻_{i}, f ∂μ = fun x => ∫⁻ xᵢ, f (Function.update x i xᵢ) ∂μ i := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    i : δ
    ⊢ Eq (MeasureTheory.lmarginal μ (Singleton.singleton i) f) fun x => MeasureThe …
  -/
  let α : Type _ := ({i} : Finset δ)
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    i : δ
    α : Type u_1 := Subtype fun x => Membership.mem (Singleton.singleton i) x
    ⊢ Eq (MeasureTheory.lmarginal μ (Singleton.singleton i) f) fun x => MeasureThe …
  -/
  let e := (MeasurableEquiv.piUnique fun j : α ↦ π j).symm
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝¹ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝ : DecidableEq δ
    f : ((i : δ) → π i) → ENNReal
    i : δ
    α : Type u_1 := Subtype fun x => Membership.mem (Singleton.singleton i) x
    e : MeasurableEquiv (π ↑Inhabited.default) ((i_1 : α) → π ↑i_1) := (Measurable …
    ⊢ Eq (MeasureTheory.lmarginal μ (Singleton.singleton i) f) fun x => MeasureThe …
  -/
  ext1 x
  calc (∫⋯∫⁻_{i}, f ∂μ) x
      = ∫⁻ (y : π (default : α)), f (updateFinset x {i} (e y)) ∂μ (default : α) := by
        simp_rw [lmarginal,
          measurePreserving_piUnique (fun j : ({i} : Finset δ) ↦ μ j) |>.symm _
            |>.lintegral_map_equiv]
    _ = ∫⁻ xᵢ, f (Function.update x i xᵢ) ∂μ i := by simp [update_eq_updateFinset]; rfl


variable {μ} in
@[gcongr]
theorem lmarginal_mono {f g : (∀ i, π i) → ℝ≥0∞} (hfg : f ≤ g) : ∫⋯∫⁻_s, f ∂μ ≤ ∫⋯∫⁻_s, g ∂μ :=
  fun _ => lintegral_mono fun _ => hfg _


theorem lmarginal_union (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f)
    (hst : Disjoint s t) : ∫⋯∫⁻_s ∪ t, f ∂μ = ∫⋯∫⁻_s, ∫⋯∫⁻_t, f ∂μ ∂μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s t : Finset δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hst : Disjoint s t
    ⊢ Eq (MeasureTheory.lmarginal μ (Union.union s t) f) (MeasureTheory.lmarginal  …
  -/
  ext1 x
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s t : Finset δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hst : Disjoint s t
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lmarginal μ (Union.union s t) f x) (MeasureTheory.lmargina …
  -/
  let e := MeasurableEquiv.piFinsetUnion π hst
  calc (∫⋯∫⁻_s ∪ t, f ∂μ) x
      = ∫⁻ (y : (i : ↥(s ∪ t)) → π i), f (updateFinset x (s ∪ t) y)
          ∂.pi fun i' : ↥(s ∪ t) ↦ μ i' := rfl
    _ = ∫⁻ (y : ((i : s) → π i) × ((j : t) → π j)), f (updateFinset x (s ∪ t) _)
          ∂(Measure.pi fun i : s ↦ μ i).prod (.pi fun j : t ↦ μ j) := by
        rw [measurePreserving_piFinsetUnion hst μ |>.lintegral_map_equiv]
    _ = ∫⁻ (y : (i : s) → π i), ∫⁻ (z : (j : t) → π j), f (updateFinset x (s ∪ t) (e (y, z)))
          ∂.pi fun j : t ↦ μ j ∂.pi fun i : s ↦ μ i := by
        apply lintegral_prod
        apply Measurable.aemeasurable
        exact hf.comp <| measurable_updateFinset.comp e.measurable
    _ = (∫⋯∫⁻_s, ∫⋯∫⁻_t, f ∂μ ∂μ) x := by
        simp_rw [lmarginal, updateFinset_updateFinset hst]
        rfl


theorem lmarginal_union' (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f) {s t : Finset δ}
    (hst : Disjoint s t) : ∫⋯∫⁻_s ∪ t, f ∂μ = ∫⋯∫⁻_t, ∫⋯∫⁻_s, f ∂μ ∂μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    s t : Finset δ
    hst : Disjoint s t
    ⊢ Eq (MeasureTheory.lmarginal μ (Union.union s t) f) (MeasureTheory.lmarginal  …
  -/
  rw [Finset.union_comm, lmarginal_union μ f hf hst.symm]
  /-
    🎉 no goals
  -/


/-- Peel off a single integral from a `lmarginal` integral at the beginning (compare with
`lmarginal_insert'`, which peels off an integral at the end). -/
theorem lmarginal_insert (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f) {i : δ}
    (hi : i ∉ s) (x : ∀ i, π i) :
    (∫⋯∫⁻_insert i s, f ∂μ) x = ∫⁻ xᵢ, (∫⋯∫⁻_s, f ∂μ) (Function.update x i xᵢ) ∂μ i := by
  rw [Finset.insert_eq, lmarginal_union μ f hf (Finset.disjoint_singleton_left.mpr hi),
    lmarginal_singleton]


/-- Peel off a single integral from a `lmarginal` integral at the beginning (compare with
`lmarginal_erase'`, which peels off an integral at the end). -/
theorem lmarginal_erase (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f) {i : δ}
    (hi : i ∈ s) (x : ∀ i, π i) :
    (∫⋯∫⁻_s, f ∂μ) x = ∫⁻ xᵢ, (∫⋯∫⁻_(erase s i), f ∂μ) (Function.update x i xᵢ) ∂μ i := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    i : δ
    hi : Membership.mem s i
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lmarginal μ s f x) (MeasureTheory.lintegral (μ i) fun xᵢ = …
  -/
  simpa [insert_erase hi] using lmarginal_insert _ hf (not_mem_erase i s) x
  /-
    🎉 no goals
  -/


/-- Peel off a single integral from a `lmarginal` integral at the end (compare with
`lmarginal_insert`, which peels off an integral at the beginning). -/
theorem lmarginal_insert' (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f) {i : δ}
    (hi : i ∉ s) :
    ∫⋯∫⁻_insert i s, f ∂μ = ∫⋯∫⁻_s, (fun x ↦ ∫⁻ xᵢ, f (Function.update x i xᵢ) ∂μ i) ∂μ := by
  rw [Finset.insert_eq, Finset.union_comm,
    lmarginal_union (s := s) μ f hf (Finset.disjoint_singleton_right.mpr hi), lmarginal_singleton]


/-- Peel off a single integral from a `lmarginal` integral at the end (compare with
`lmarginal_erase`, which peels off an integral at the beginning). -/
theorem lmarginal_erase' (f : (∀ i, π i) → ℝ≥0∞) (hf : Measurable f) {i : δ}
    (hi : i ∈ s) :
    ∫⋯∫⁻_s, f ∂μ = ∫⋯∫⁻_(erase s i), (fun x ↦ ∫⁻ xᵢ, f (Function.update x i xᵢ) ∂μ i) ∂μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s : Finset δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    i : δ
    hi : Membership.mem s i
    ⊢ Eq (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ (s.erase i) fu …
  -/
  simpa [insert_erase hi] using lmarginal_insert' _ hf (not_mem_erase i s)
  /-
    🎉 no goals
  -/


@[simp] theorem lmarginal_univ [Fintype δ] {f : (∀ i, π i) → ℝ≥0∞} :
    ∫⋯∫⁻_univ, f ∂μ = fun _ => ∫⁻ x, f x ∂Measure.pi μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    f : ((i : δ) → π i) → ENNReal
    ⊢ Eq (MeasureTheory.lmarginal μ Finset.univ f) fun x => MeasureTheory.lintegra …
  -/
  let e : { j // j ∈ Finset.univ } ≃ δ := Equiv.subtypeUnivEquiv mem_univ
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    f : ((i : δ) → π i) → ENNReal
    e : Equiv (Subtype fun j => Membership.mem Finset.univ j) δ := Equiv.subtypeUn …
    ⊢ Eq (MeasureTheory.lmarginal μ Finset.univ f) fun x => MeasureTheory.lintegra …
  -/
  ext1 x
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    f : ((i : δ) → π i) → ENNReal
    e : Equiv (Subtype fun j => Membership.mem Finset.univ j) δ := Equiv.subtypeUn …
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lmarginal μ Finset.univ f x) (MeasureTheory.lintegral (Mea …
  -/
  simp_rw [lmarginal, measurePreserving_piCongrLeft μ e |>.lintegral_map_equiv, updateFinset_def]
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    f : ((i : δ) → π i) → ENNReal
    e : Equiv (Subtype fun j => Membership.mem Finset.univ j) δ := Equiv.subtypeUn …
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi fun i => μ ↑i) fun y = …
  -/
  simp
  /-
    case h
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    f : ((i : δ) → π i) → ENNReal
    e : Equiv (Subtype fun j => Membership.mem Finset.univ j) δ := Equiv.subtypeUn …
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi fun i => μ ↑i) fun y = …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem lintegral_eq_lmarginal_univ [Fintype δ] {f : (∀ i, π i) → ℝ≥0∞} (x : ∀ i, π i) :
                                                        /-
                                                          δ : Type u_1
                                                          π : δ → Type u_3
                                                          inst✝³ : (x : δ) → MeasurableSpace (π x)
                                                          μ : (i : δ) → MeasureTheory.Measure (π i)
                                                          inst✝² : DecidableEq δ
                                                          inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
                                                          inst✝ : Fintype δ
                                                          f : ((i : δ) → π i) → ENNReal
                                                          x : (i : δ) → π i
                                                          ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (Meas …
                                                        -/
    ∫⁻ x, f x ∂Measure.pi μ = (∫⋯∫⁻_univ, f ∂μ) x := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem lmarginal_image [DecidableEq δ'] {e : δ' → δ} (he : Injective e) (s : Finset δ')
    {f : (∀ i, π (e i)) → ℝ≥0∞} (hf : Measurable f) (x : ∀ i, π i) :
      (∫⋯∫⁻_s.image e, f ∘ (· ∘' e) ∂μ) x = (∫⋯∫⁻_s, f ∂μ ∘' e) (x ∘' e) := by
  have h : Measurable ((· ∘' e) : (∀ i, π i) → _) :=
    measurable_pi_iff.mpr <| fun i ↦ measurable_pi_apply (e i)
  induction s using Finset.induction generalizing x with
  | empty => simp
  | insert hi ih =>
    rw [image_insert, lmarginal_insert _ (hf.comp h) (he.mem_finset_image.not.mpr hi),
      lmarginal_insert _ hf hi]
    simp_rw [ih, ← update_comp_eq_of_injective' x he]


theorem lmarginal_update_of_not_mem {i : δ}
    {f : (∀ i, π i) → ℝ≥0∞} (hf : Measurable f) (hi : i ∉ s) (x : ∀ i, π i) (y : π i) :
    (∫⋯∫⁻_s, f ∂μ) (Function.update x i y) = (∫⋯∫⁻_s, f ∘ (Function.update · i y) ∂μ) x := by
  induction s using Finset.induction generalizing x with
  | empty => simp
  | @insert i' s hi' ih =>
    rw [lmarginal_insert _ hf hi', lmarginal_insert _ (hf.comp measurable_update_left) hi']
    have hii' : i ≠ i' := mt (by rintro rfl; exact mem_insert_self i s) hi
    simp_rw [update_comm hii', ih (mt Finset.mem_insert_of_mem hi)]


theorem lmarginal_eq_of_subset {f g : (∀ i, π i) → ℝ≥0∞} (hst : s ⊆ t)
    (hf : Measurable f) (hg : Measurable g) (hfg : ∫⋯∫⁻_s, f ∂μ = ∫⋯∫⁻_s, g ∂μ) :
    ∫⋯∫⁻_t, f ∂μ = ∫⋯∫⁻_t, g ∂μ := by
  rw [← union_sdiff_of_subset hst, lmarginal_union' μ f hf disjoint_sdiff,
    lmarginal_union' μ g hg disjoint_sdiff, hfg]


theorem lmarginal_le_of_subset {f g : (∀ i, π i) → ℝ≥0∞} (hst : s ⊆ t)
    (hf : Measurable f) (hg : Measurable g) (hfg : ∫⋯∫⁻_s, f ∂μ ≤ ∫⋯∫⁻_s, g ∂μ) :
    ∫⋯∫⁻_t, f ∂μ ≤ ∫⋯∫⁻_t, g ∂μ := by
  rw [← union_sdiff_of_subset hst, lmarginal_union' μ f hf disjoint_sdiff,
    lmarginal_union' μ g hg disjoint_sdiff]
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝² : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝¹ : DecidableEq δ
    s t : Finset δ
    inst✝ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    f g : ((i : δ) → π i) → ENNReal
    hst : HasSubset.Subset s t
    hf : Measurable f
    hg : Measurable g
    hfg : LE.le (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
    ⊢ LE.le (MeasureTheory.lmarginal μ (SDiff.sdiff t s) (MeasureTheory.lmarginal  …
  -/
  exact lmarginal_mono hfg
  /-
    🎉 no goals
  -/


theorem lintegral_eq_of_lmarginal_eq [Fintype δ] (s : Finset δ) {f g : (∀ i, π i) → ℝ≥0∞}
    (hf : Measurable f) (hg : Measurable g) (hfg : ∫⋯∫⁻_s, f ∂μ = ∫⋯∫⁻_s, g ∂μ) :
    ∫⁻ x, f x ∂Measure.pi μ = ∫⁻ x, g x ∂Measure.pi μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    s : Finset δ
    f g : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hg : Measurable g
    hfg : Eq (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (Meas …
  -/
  rcases isEmpty_or_nonempty (∀ i, π i) with h|⟨⟨x⟩⟩
    /-
      case inl
      δ : Type u_1
      π : δ → Type u_3
      inst✝³ : (x : δ) → MeasurableSpace (π x)
      μ : (i : δ) → MeasureTheory.Measure (π i)
      inst✝² : DecidableEq δ
      inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
      inst✝ : Fintype δ
      s : Finset δ
      f g : ((i : δ) → π i) → ENNReal
      hf : Measurable f
      hg : Measurable g
      hfg : Eq (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
      h : IsEmpty ((i : δ) → π i)
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (Meas …
    -/
  · simp_rw [lintegral_of_isEmpty]
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    s : Finset δ
    f g : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hg : Measurable g
    hfg : Eq (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
    x : (i : δ) → π i
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (Meas …
  -/
  simp_rw [lintegral_eq_lmarginal_univ x, lmarginal_eq_of_subset (Finset.subset_univ s) hf hg hfg]
  /-
    🎉 no goals
  -/


theorem lintegral_le_of_lmarginal_le [Fintype δ] (s : Finset δ) {f g : (∀ i, π i) → ℝ≥0∞}
    (hf : Measurable f) (hg : Measurable g) (hfg : ∫⋯∫⁻_s, f ∂μ ≤ ∫⋯∫⁻_s, g ∂μ) :
    ∫⁻ x, f x ∂Measure.pi μ ≤ ∫⁻ x, g x ∂Measure.pi μ := by
  /-
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    s : Finset δ
    f g : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hg : Measurable g
    hfg : LE.le (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (M …
  -/
  rcases isEmpty_or_nonempty (∀ i, π i) with h|⟨⟨x⟩⟩
    /-
      case inl
      δ : Type u_1
      π : δ → Type u_3
      inst✝³ : (x : δ) → MeasurableSpace (π x)
      μ : (i : δ) → MeasureTheory.Measure (π i)
      inst✝² : DecidableEq δ
      inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
      inst✝ : Fintype δ
      s : Finset δ
      f g : ((i : δ) → π i) → ENNReal
      hf : Measurable f
      hg : Measurable g
      hfg : LE.le (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
      h : IsEmpty ((i : δ) → π i)
      ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (M …
    -/
  · simp_rw [lintegral_of_isEmpty, le_rfl]
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    δ : Type u_1
    π : δ → Type u_3
    inst✝³ : (x : δ) → MeasurableSpace (π x)
    μ : (i : δ) → MeasureTheory.Measure (π i)
    inst✝² : DecidableEq δ
    inst✝¹ : ∀ (i : δ), MeasureTheory.SigmaFinite (μ i)
    inst✝ : Fintype δ
    s : Finset δ
    f g : ((i : δ) → π i) → ENNReal
    hf : Measurable f
    hg : Measurable g
    hfg : LE.le (MeasureTheory.lmarginal μ s f) (MeasureTheory.lmarginal μ s g)
    x : (i : δ) → π i
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.pi μ) fun x => f x) (M …
  -/
  simp_rw [lintegral_eq_lmarginal_univ x, lmarginal_le_of_subset (Finset.subset_univ s) hf hg hfg x]
  /-
    🎉 no goals
  -/


