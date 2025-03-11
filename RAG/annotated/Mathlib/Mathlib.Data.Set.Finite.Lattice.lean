instance fintypeiUnion [DecidableEq α] [Fintype (PLift ι)] (f : ι → Set α) [∀ i, Fintype (f i)] :
    Fintype (⋃ i, f i) :=
                                                                                      /-
                                                                                        α : Type u
                                                                                        β : Type v
                                                                                        ι : Sort w
                                                                                        γ : Type x
                                                                                        inst✝² : DecidableEq α
                                                                                        inst✝¹ : Fintype (PLift ι)
                                                                                        f : ι → Set α
                                                                                        inst✝ : (i : ι) → Fintype ↑(f i)
                                                                                        ⊢ ∀ (x : α), Iff (Membership.mem (Finset.univ.biUnion fun i => (f i.down).toFi …
                                                                                      -/
  Fintype.ofFinset (Finset.univ.biUnion fun i : PLift ι => (f i.down).toFinset) <| by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance fintypesUnion [DecidableEq α] {s : Set (Set α)} [Fintype s]
    [H : ∀ t : s, Fintype (t : Set α)] : Fintype (⋃₀ s) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝¹ : DecidableEq α
    s : Set (Set α)
    inst✝ : Fintype ↑s
    H : (t : ↑s) → Fintype ↑↑t
    ⊢ Fintype ↑s.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝¹ : DecidableEq α
    s : Set (Set α)
    inst✝ : Fintype ↑s
    H : (t : ↑s) → Fintype ↑↑t
    ⊢ Fintype ↑(Set.iUnion fun i => ↑i)
  -/
  exact @Set.fintypeiUnion _ _ _ _ _ H
  /-
    🎉 no goals
  -/


lemma toFinset_iUnion [Fintype β] [DecidableEq α] (f : β → Set α)
    [∀ w, Fintype (f w)] :
    Set.toFinset (⋃ (x : β), f x) =
    Finset.biUnion (Finset.univ : Finset β) (fun x => (f x).toFinset) := by
  /-
    α : Type u
    β : Type v
    inst✝² : Fintype β
    inst✝¹ : DecidableEq α
    f : β → Set α
    inst✝ : (w : β) → Fintype ↑(f w)
    ⊢ Eq (Set.iUnion fun x => f x).toFinset (Finset.univ.biUnion fun x => (f x).to …
  -/
  ext v
  /-
    case h
    α : Type u
    β : Type v
    inst✝² : Fintype β
    inst✝¹ : DecidableEq α
    f : β → Set α
    inst✝ : (w : β) → Fintype ↑(f w)
    v : α
    ⊢ Iff (Membership.mem (Set.iUnion fun x => f x).toFinset v) (Membership.mem (F …
  -/
  simp only [mem_toFinset, mem_iUnion, Finset.mem_biUnion, Finset.mem_univ, true_and]
  /-
    🎉 no goals
  -/


instance finite_iUnion [Finite ι] (f : ι → Set α) [∀ i, Finite (f i)] : Finite (⋃ i, f i) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝¹ : Finite ι
    f : ι → Set α
    inst✝ : ∀ (i : ι), Finite ↑(f i)
    ⊢ Finite ↑(Set.iUnion fun i => f i)
  -/
  rw [iUnion_eq_range_psigma]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    inst✝¹ : Finite ι
    f : ι → Set α
    inst✝ : ∀ (i : ι), Finite ↑(f i)
    ⊢ Finite ↑(Set.range fun a => ↑a.snd)
  -/
  apply Set.finite_range
  /-
    🎉 no goals
  -/


instance finite_sUnion {s : Set (Set α)} [Finite s] [H : ∀ t : s, Finite (t : Set α)] :
    Finite (⋃₀ s) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s : Set (Set α)
    inst✝ : Finite ↑s
    H : ∀ (t : ↑s), Finite ↑↑t
    ⊢ Finite ↑s.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s : Set (Set α)
    inst✝ : Finite ↑s
    H : ∀ (t : ↑s), Finite ↑↑t
    ⊢ Finite ↑(Set.iUnion fun i => ↑i)
  -/
  exact @Finite.Set.finite_iUnion _ _ _ _ H
  /-
    🎉 no goals
  -/


theorem finite_biUnion {ι : Type*} (s : Set ι) [Finite s] (t : ι → Set α)
    (H : ∀ i ∈ s, Finite (t i)) : Finite (⋃ x ∈ s, t x) := by
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    inst✝ : Finite ↑s
    t : ι → Set α
    H : ∀ (i : ι), Membership.mem s i → Finite ↑(t i)
    ⊢ Finite ↑(Set.iUnion fun x => Set.iUnion fun h => t x)
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    inst✝ : Finite ↑s
    t : ι → Set α
    H : ∀ (i : ι), Membership.mem s i → Finite ↑(t i)
    ⊢ Finite ↑(Set.iUnion fun x => t ↑x)
  -/
  haveI : ∀ i : s, Finite (t i) := fun i => H i i.property
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    inst✝ : Finite ↑s
    t : ι → Set α
    H : ∀ (i : ι), Membership.mem s i → Finite ↑(t i)
    this : ∀ (i : ↑s), Finite ↑(t ↑i)
    ⊢ Finite ↑(Set.iUnion fun x => t ↑x)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance finite_biUnion' {ι : Type*} (s : Set ι) [Finite s] (t : ι → Set α) [∀ i, Finite (t i)] :
    Finite (⋃ x ∈ s, t x) :=
  finite_biUnion s t fun _ _ => inferInstance


/-- Example: `Finite (⋃ (i < n), f i)` where `f : ℕ → Set α` and `[∀ i, Finite (f i)]`
(when given instances from `Order.Interval.Finset.Nat`).
-/
instance finite_biUnion'' {ι : Type*} (p : ι → Prop) [h : Finite { x | p x }] (t : ι → Set α)
    [∀ i, Finite (t i)] : Finite (⋃ (x) (_ : p x), t x) :=
  @Finite.Set.finite_biUnion' _ _ (setOf p) h t _


instance finite_iInter {ι : Sort*} [Nonempty ι] (t : ι → Set α) [∀ i, Finite (t i)] :
    Finite (⋂ i, t i) :=
  Finite.Set.subset (t <| Classical.arbitrary ι) (iInter_subset _ _)


theorem finite_iUnion [Finite ι] {f : ι → Set α} (H : ∀ i, (f i).Finite) : (⋃ i, f i).Finite :=
  haveI := fun i => (H i).to_subtype
  toFinite _


/-- Dependent version of `Finite.biUnion`. -/
theorem Finite.biUnion' {ι} {s : Set ι} (hs : s.Finite) {t : ∀ i ∈ s, Set α}
    (ht : ∀ i (hi : i ∈ s), (t i hi).Finite) : (⋃ i ∈ s, t i ‹_›).Finite := by
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    hs : s.Finite
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (hi : Membership.mem s i), (t i hi).Finite
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => t i h).Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    hs : s.Finite
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (hi : Membership.mem s i), (t i hi).Finite
    this : Finite ↑s
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => t i h).Finite
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u
    ι : Type u_1
    s : Set ι
    hs : s.Finite
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (hi : Membership.mem s i), (t i hi).Finite
    this : Finite ↑s
    ⊢ (Set.iUnion fun x => t ↑x ⋯).Finite
  -/
  apply finite_iUnion fun i : s => ht i.1 i.2
  /-
    🎉 no goals
  -/


theorem Finite.biUnion {ι} {s : Set ι} (hs : s.Finite) {t : ι → Set α}
    (ht : ∀ i ∈ s, (t i).Finite) : (⋃ i ∈ s, t i).Finite :=
  hs.biUnion' ht


theorem Finite.sUnion {s : Set (Set α)} (hs : s.Finite) (H : ∀ t ∈ s, Set.Finite t) :
    (⋃₀ s).Finite := by
  /-
    α : Type u
    s : Set (Set α)
    hs : s.Finite
    H : ∀ (t : Set α), Membership.mem s t → t.Finite
    ⊢ s.sUnion.Finite
  -/
  simpa only [sUnion_eq_biUnion] using hs.biUnion H
  /-
    🎉 no goals
  -/


theorem Finite.sInter {α : Type*} {s : Set (Set α)} {t : Set α} (ht : t ∈ s) (hf : t.Finite) :
    (⋂₀ s).Finite :=
  hf.subset (sInter_subset_of_mem ht)


/-- If sets `s i` are finite for all `i` from a finite set `t` and are empty for `i ∉ t`, then the
union `⋃ i, s i` is a finite set. -/
theorem Finite.iUnion {ι : Type*} {s : ι → Set α} {t : Set ι} (ht : t.Finite)
    (hs : ∀ i ∈ t, (s i).Finite) (he : ∀ i, i ∉ t → s i = ∅) : (⋃ i, s i).Finite := by
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    t : Set ι
    ht : t.Finite
    hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
    he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
    ⊢ (Set.iUnion fun i => s i).Finite
  -/
  suffices ⋃ i, s i ⊆ ⋃ i ∈ t, s i by exact (ht.biUnion hs).subset this
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    t : Set ι
    ht : t.Finite
    hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
    he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
    ⊢ HasSubset.Subset (Set.iUnion fun i => s i) (Set.iUnion fun i => Set.iUnion f …
  -/
  refine iUnion_subset fun i x hx => ?_
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    t : Set ι
    ht : t.Finite
    hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
    he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
    i : ι
    x : α
    hx : Membership.mem (s i) x
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => s i) x
  -/
  by_cases hi : i ∈ t
    /-
      case pos
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set ι
      ht : t.Finite
      hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
      he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
      i : ι
      x : α
      hx : Membership.mem (s i) x
      hi : Membership.mem t i
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => s i) x
    -/
  · exact mem_biUnion hi hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set ι
      ht : t.Finite
      hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
      he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
      i : ι
      x : α
      hx : Membership.mem (s i) x
      hi : Not (Membership.mem t i)
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => s i) x
    -/
  · rw [he i hi, mem_empty_iff_false] at hx
    /-
      case neg
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set ι
      ht : t.Finite
      hs : ∀ (i : ι), Membership.mem t i → (s i).Finite
      he : ∀ (i : ι), Not (Membership.mem t i) → Eq (s i) EmptyCollection.emptyColle …
      i : ι
      x : α
      hx : False
      hi : Not (Membership.mem t i)
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => s i) x
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem Finite.preimage' (h : s.Finite) (hf : ∀ b ∈ s, (f ⁻¹' {b}).Finite) :
    (f ⁻¹' s).Finite := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : s.Finite
    hf : ∀ (b : β), Membership.mem s b → (Set.preimage f (Singleton.singleton b)). …
    ⊢ (Set.preimage f s).Finite
  -/
  rw [← Set.biUnion_preimage_singleton]
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : s.Finite
    hf : ∀ (b : β), Membership.mem s b → (Set.preimage f (Singleton.singleton b)). …
    ⊢ (Set.iUnion fun y => Set.iUnion fun h => Set.preimage f (Singleton.singleton …
  -/
  exact Set.Finite.biUnion h hf
  /-
    🎉 no goals
  -/


/-- A finite union of finsets is finite. -/
theorem union_finset_finite_of_range_finite (f : α → Finset β) (h : (range f).Finite) :
    (⋃ a, (f a : Set β)).Finite := by
  /-
    α : Type u
    β : Type v
    f : α → Finset β
    h : (Set.range f).Finite
    ⊢ (Set.iUnion fun a => ↑(f a)).Finite
  -/
  rw [← biUnion_range]
  /-
    α : Type u
    β : Type v
    f : α → Finset β
    h : (Set.range f).Finite
    ⊢ (Set.iUnion fun x => Set.iUnion fun h => ↑x).Finite
  -/
  exact h.biUnion fun y _ => y.finite_toSet
  /-
    🎉 no goals
  -/


theorem finite_subset_iUnion {s : Set α} (hs : s.Finite) {ι} {t : ι → Set α} (h : s ⊆ ⋃ i, t i) :
    ∃ I : Set ι, I.Finite ∧ s ⊆ ⋃ i ∈ I, t i := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ι : Type u_1
    t : ι → Set α
    h : HasSubset.Subset s (Set.iUnion fun i => t i)
    ⊢ Exists fun I => And I.Finite (HasSubset.Subset s (Set.iUnion fun i => Set.iU …
  -/
  have := hs.to_subtype
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ι : Type u_1
    t : ι → Set α
    h : HasSubset.Subset s (Set.iUnion fun i => t i)
    this : Finite ↑s
    ⊢ Exists fun I => And I.Finite (HasSubset.Subset s (Set.iUnion fun i => Set.iU …
  -/
  choose f hf using show ∀ x : s, ∃ i, x.1 ∈ t i by simpa [subset_def] using h
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ι : Type u_1
    t : ι → Set α
    h : HasSubset.Subset s (Set.iUnion fun i => t i)
    this : Finite ↑s
    f : ↑s → ι
    hf : ∀ (x : ↑s), Membership.mem (t (f x)) ↑x
    ⊢ Exists fun I => And I.Finite (HasSubset.Subset s (Set.iUnion fun i => Set.iU …
  -/
  refine ⟨range f, finite_range f, fun x hx => ?_⟩
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ι : Type u_1
    t : ι → Set α
    h : HasSubset.Subset s (Set.iUnion fun i => t i)
    this : Finite ↑s
    f : ↑s → ι
    hf : ∀ (x : ↑s), Membership.mem (t (f x)) ↑x
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => t i) x
  -/
  rw [biUnion_range, mem_iUnion]
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ι : Type u_1
    t : ι → Set α
    h : HasSubset.Subset s (Set.iUnion fun i => t i)
    this : Finite ↑s
    f : ↑s → ι
    hf : ∀ (x : ↑s), Membership.mem (t (f x)) ↑x
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun i => Membership.mem (t (f i)) x
  -/
  exact ⟨⟨x, hx⟩, hf _⟩
  /-
    🎉 no goals
  -/


theorem eq_finite_iUnion_of_finite_subset_iUnion {ι} {s : ι → Set α} {t : Set α} (tfin : t.Finite)
    (h : t ⊆ ⋃ i, s i) :
    ∃ I : Set ι,
      I.Finite ∧
        ∃ σ : { i | i ∈ I } → Set α, (∀ i, (σ i).Finite) ∧ (∀ i, σ i ⊆ s i) ∧ t = ⋃ i, σ i :=
  let ⟨I, Ifin, hI⟩ := finite_subset_iUnion tfin h
  ⟨I, Ifin, fun x => s x ∩ t, fun _ => tfin.subset inter_subset_right, fun _ =>
    inter_subset_left, by
    /-
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set α
      tfin : t.Finite
      h : HasSubset.Subset t (Set.iUnion fun i => s i)
      I : Set ι
      Ifin : I.Finite
      hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
      ⊢ Eq t (Set.iUnion fun i => (fun x => Inter.inter (s ↑x) t) i)
    -/
    ext x
    /-
      case h
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set α
      tfin : t.Finite
      h : HasSubset.Subset t (Set.iUnion fun i => s i)
      I : Set ι
      Ifin : I.Finite
      hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
      x : α
      ⊢ Iff (Membership.mem t x) (Membership.mem (Set.iUnion fun i => (fun x => Inte …
    -/
    rw [mem_iUnion]
    /-
      case h
      α : Type u
      ι : Type u_1
      s : ι → Set α
      t : Set α
      tfin : t.Finite
      h : HasSubset.Subset t (Set.iUnion fun i => s i)
      I : Set ι
      Ifin : I.Finite
      hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
      x : α
      ⊢ Iff (Membership.mem t x) (Exists fun i => Membership.mem (Inter.inter (s ↑i) …
    -/
    constructor
      /-
        case h.mp
        α : Type u
        ι : Type u_1
        s : ι → Set α
        t : Set α
        tfin : t.Finite
        h : HasSubset.Subset t (Set.iUnion fun i => s i)
        I : Set ι
        Ifin : I.Finite
        hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
        x : α
        ⊢ Membership.mem t x → Exists fun i => Membership.mem (Inter.inter (s ↑i) t) x
      -/
    · intro x_in
      /-
        case h.mp
        α : Type u
        ι : Type u_1
        s : ι → Set α
        t : Set α
        tfin : t.Finite
        h : HasSubset.Subset t (Set.iUnion fun i => s i)
        I : Set ι
        Ifin : I.Finite
        hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
        x : α
        x_in : Membership.mem t x
        ⊢ Exists fun i => Membership.mem (Inter.inter (s ↑i) t) x
      -/
      rcases mem_iUnion.mp (hI x_in) with ⟨i, _, ⟨hi, rfl⟩, H⟩
      /-
        case h.mp.intro.intro.intro.intro
        α : Type u
        ι : Type u_1
        s : ι → Set α
        t : Set α
        tfin : t.Finite
        h : HasSubset.Subset t (Set.iUnion fun i => s i)
        I : Set ι
        Ifin : I.Finite
        hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
        x : α
        x_in : Membership.mem t x
        i : ι
        hi : Membership.mem I i
        H : Membership.mem ((fun h => s i) hi) x
        ⊢ Exists fun i => Membership.mem (Inter.inter (s ↑i) t) x
      -/
      exact ⟨⟨i, hi⟩, ⟨H, x_in⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        α : Type u
        ι : Type u_1
        s : ι → Set α
        t : Set α
        tfin : t.Finite
        h : HasSubset.Subset t (Set.iUnion fun i => s i)
        I : Set ι
        Ifin : I.Finite
        hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
        x : α
        ⊢ (Exists fun i => Membership.mem (Inter.inter (s ↑i) t) x) → Membership.mem t x
      -/
    · rintro ⟨i, -, H⟩
      /-
        case h.mpr.intro.intro
        α : Type u
        ι : Type u_1
        s : ι → Set α
        t : Set α
        tfin : t.Finite
        h : HasSubset.Subset t (Set.iUnion fun i => s i)
        I : Set ι
        Ifin : I.Finite
        hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
        x : α
        i : ↑(setOf fun i => Membership.mem I i)
        H : Membership.mem t x
        ⊢ Membership.mem t x
      -/
      exact H⟩
      /-
        🎉 no goals
      -/


theorem infinite_iUnion {ι : Type*} [Infinite ι] {s : ι → Set α} (hs : Function.Injective s) :
    (⋃ i, s i).Infinite :=
  fun hfin ↦ @not_injective_infinite_finite ι _ _ hfin.finite_subsets.to_subtype
                                                             /-
                                                               α : Type u
                                                               ι : Type u_1
                                                               inst✝ : Infinite ι
                                                               s : ι → Set α
                                                               hs : Function.Injective s
                                                               hfin : (Set.iUnion fun i => s i).Finite
                                                               i j : ι
                                                               h_eq : Eq ((fun i => ⟨s i, ⋯⟩) i) ((fun i => ⟨s i, ⋯⟩) j)
                                                               ⊢ Eq (s i) (s j)
                                                             -/
    (fun i ↦ ⟨s i, subset_iUnion _ _⟩) fun i j h_eq ↦ hs (by simpa using h_eq)
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Infinite.biUnion {ι : Type*} {s : ι → Set α} {a : Set ι} (ha : a.Infinite)
    (hs : a.InjOn s) : (⋃ i ∈ a, s i).Infinite := by
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    a : Set ι
    ha : a.Infinite
    hs : Set.InjOn s a
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => s i).Infinite
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    a : Set ι
    ha : a.Infinite
    hs : Set.InjOn s a
    ⊢ (Set.iUnion fun x => s ↑x).Infinite
  -/
  have _ := ha.to_subtype
  /-
    α : Type u
    ι : Type u_1
    s : ι → Set α
    a : Set ι
    ha : a.Infinite
    hs : Set.InjOn s a
    x✝ : Infinite ↑a
    ⊢ (Set.iUnion fun x => s ↑x).Infinite
  -/
  exact infinite_iUnion fun ⟨i,hi⟩ ⟨j,hj⟩ hij ↦ by simp [hs hi hj hij]
  /-
    🎉 no goals
  -/


theorem Infinite.sUnion {s : Set (Set α)} (hs : s.Infinite) : (⋃₀ s).Infinite := by
  /-
    α : Type u
    s : Set (Set α)
    hs : s.Infinite
    ⊢ s.sUnion.Infinite
  -/
  rw [sUnion_eq_iUnion]
  /-
    α : Type u
    s : Set (Set α)
    hs : s.Infinite
    ⊢ (Set.iUnion fun i => ↑i).Infinite
  -/
  have _ := hs.to_subtype
  /-
    α : Type u
    s : Set (Set α)
    hs : s.Infinite
    x✝ : Infinite ↑s
    ⊢ (Set.iUnion fun i => ↑i).Infinite
  -/
  exact infinite_iUnion Subtype.coe_injective
  /-
    🎉 no goals
  -/


lemma map_finite_biSup {F ι : Type*} [CompleteLattice α] [CompleteLattice β] [FunLike F α β]
    [SupBotHomClass F α β] {s : Set ι} (hs : s.Finite) (f : F) (g : ι → α) :
    f (⨆ x ∈ s, g x) = ⨆ x ∈ s, f (g x) := by
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : SupBotHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    ⊢ Eq (f (iSup fun x => iSup fun h => g x)) (iSup fun x => iSup fun h => f (g x))
  -/
  have := map_finset_sup f hs.toFinset g
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : SupBotHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    this : Eq (f (hs.toFinset.sup g)) (hs.toFinset.sup (Function.comp (⇑f) g))
    ⊢ Eq (f (iSup fun x => iSup fun h => g x)) (iSup fun x => iSup fun h => f (g x))
  -/
  simp only [Finset.sup_eq_iSup, hs.mem_toFinset, comp_apply] at this
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : SupBotHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    this : Eq (f (iSup fun a => iSup fun x => g a)) (iSup fun a => iSup fun x => f …
    ⊢ Eq (f (iSup fun x => iSup fun h => g x)) (iSup fun x => iSup fun h => f (g x))
  -/
  exact this
  /-
    🎉 no goals
  -/


lemma map_finite_biInf {F ι : Type*} [CompleteLattice α] [CompleteLattice β] [FunLike F α β]
    [InfTopHomClass F α β] {s : Set ι} (hs : s.Finite) (f : F) (g : ι → α) :
    f (⨅ x ∈ s, g x) = ⨅ x ∈ s, f (g x) := by
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : InfTopHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    ⊢ Eq (f (iInf fun x => iInf fun h => g x)) (iInf fun x => iInf fun h => f (g x))
  -/
  have := map_finset_inf f hs.toFinset g
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : InfTopHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    this : Eq (f (hs.toFinset.inf g)) (hs.toFinset.inf (Function.comp (⇑f) g))
    ⊢ Eq (f (iInf fun x => iInf fun h => g x)) (iInf fun x => iInf fun h => f (g x))
  -/
  simp only [Finset.inf_eq_iInf, hs.mem_toFinset, comp_apply] at this
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝³ : CompleteLattice α
    inst✝² : CompleteLattice β
    inst✝¹ : FunLike F α β
    inst✝ : InfTopHomClass F α β
    s : Set ι
    hs : s.Finite
    f : F
    g : ι → α
    this : Eq (f (iInf fun a => iInf fun x => g a)) (iInf fun a => iInf fun x => f …
    ⊢ Eq (f (iInf fun x => iInf fun h => g x)) (iInf fun x => iInf fun h => f (g x))
  -/
  exact this
  /-
    🎉 no goals
  -/


lemma map_finite_iSup {F ι : Type*} [CompleteLattice α] [CompleteLattice β] [FunLike F α β]
    [SupBotHomClass F α β] [Finite ι] (f : F) (g : ι → α) :
    f (⨆ i, g i) = ⨆ i, f (g i) := by
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝⁴ : CompleteLattice α
    inst✝³ : CompleteLattice β
    inst✝² : FunLike F α β
    inst✝¹ : SupBotHomClass F α β
    inst✝ : Finite ι
    f : F
    g : ι → α
    ⊢ Eq (f (iSup fun i => g i)) (iSup fun i => f (g i))
  -/
  rw [← iSup_univ (f := g), ← iSup_univ (f := fun i ↦ f (g i))]
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝⁴ : CompleteLattice α
    inst✝³ : CompleteLattice β
    inst✝² : FunLike F α β
    inst✝¹ : SupBotHomClass F α β
    inst✝ : Finite ι
    f : F
    g : ι → α
    ⊢ Eq (f (iSup fun x => iSup fun h => g x)) (iSup fun x => iSup fun h => f (g x))
  -/
  exact map_finite_biSup finite_univ f g
  /-
    🎉 no goals
  -/


lemma map_finite_iInf {F ι : Type*} [CompleteLattice α] [CompleteLattice β] [FunLike F α β]
    [InfTopHomClass F α β] [Finite ι] (f : F) (g : ι → α) :
    f (⨅ i, g i) = ⨅ i, f (g i) := by
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝⁴ : CompleteLattice α
    inst✝³ : CompleteLattice β
    inst✝² : FunLike F α β
    inst✝¹ : InfTopHomClass F α β
    inst✝ : Finite ι
    f : F
    g : ι → α
    ⊢ Eq (f (iInf fun i => g i)) (iInf fun i => f (g i))
  -/
  rw [← iInf_univ (f := g), ← iInf_univ (f := fun i ↦ f (g i))]
  /-
    α : Type u
    β : Type v
    F : Type u_1
    ι : Type u_2
    inst✝⁴ : CompleteLattice α
    inst✝³ : CompleteLattice β
    inst✝² : FunLike F α β
    inst✝¹ : InfTopHomClass F α β
    inst✝ : Finite ι
    f : F
    g : ι → α
    ⊢ Eq (f (iInf fun x => iInf fun h => g x)) (iInf fun x => iInf fun h => f (g x))
  -/
  exact map_finite_biInf finite_univ f g
  /-
    🎉 no goals
  -/


theorem Finite.iSup_biInf_of_monotone {ι ι' α : Type*} [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (· ≤ ·)] [Order.Frame α] {s : Set ι} (hs : s.Finite) {f : ι → ι' → α}
    (hf : ∀ i ∈ s, Monotone (f i)) : ⨆ j, ⨅ i ∈ s, f i j = ⨅ i ∈ s, ⨆ j, f i j := by
  induction s, hs using Set.Finite.dinduction_on with
  | H0 => simp [iSup_const]
  | H1 _ _ ihs =>
    rw [forall_mem_insert] at hf
    simp only [iInf_insert, ← ihs hf.2]
    exact iSup_inf_of_monotone hf.1 fun j₁ j₂ hj => iInf₂_mono fun i hi => hf.2 i hi hj


theorem Finite.iSup_biInf_of_antitone {ι ι' α : Type*} [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (swap (· ≤ ·))] [Order.Frame α] {s : Set ι} (hs : s.Finite) {f : ι → ι' → α}
    (hf : ∀ i ∈ s, Antitone (f i)) : ⨆ j, ⨅ i ∈ s, f i j = ⨅ i ∈ s, ⨆ j, f i j :=
  @Finite.iSup_biInf_of_monotone ι ι'ᵒᵈ α _ _ _ _ _ hs _ fun i hi => (hf i hi).dual_left


theorem Finite.iInf_biSup_of_monotone {ι ι' α : Type*} [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (swap (· ≤ ·))] [Order.Coframe α] {s : Set ι} (hs : s.Finite) {f : ι → ι' → α}
    (hf : ∀ i ∈ s, Monotone (f i)) : ⨅ j, ⨆ i ∈ s, f i j = ⨆ i ∈ s, ⨅ j, f i j :=
  hs.iSup_biInf_of_antitone (α := αᵒᵈ) fun i hi => (hf i hi).dual_right


theorem Finite.iInf_biSup_of_antitone {ι ι' α : Type*} [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (· ≤ ·)] [Order.Coframe α] {s : Set ι} (hs : s.Finite) {f : ι → ι' → α}
    (hf : ∀ i ∈ s, Antitone (f i)) : ⨅ j, ⨆ i ∈ s, f i j = ⨆ i ∈ s, ⨅ j, f i j :=
  hs.iSup_biInf_of_monotone (α := αᵒᵈ) fun i hi => (hf i hi).dual_right


theorem iSup_iInf_of_monotone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (· ≤ ·)] [Order.Frame α] {f : ι → ι' → α} (hf : ∀ i, Monotone (f i)) :
    ⨆ j, ⨅ i, f i j = ⨅ i, ⨆ j, f i j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : Type u_3
    inst✝⁴ : Finite ι
    inst✝³ : Preorder ι'
    inst✝² : Nonempty ι'
    inst✝¹ : IsDirected ι' fun x1 x2 => LE.le x1 x2
    inst✝ : Order.Frame α
    f : ι → ι' → α
    hf : ∀ (i : ι), Monotone (f i)
    ⊢ Eq (iSup fun j => iInf fun i => f i j) (iInf fun i => iSup fun j => f i j)
  -/
  simpa only [iInf_univ] using finite_univ.iSup_biInf_of_monotone fun i _ => hf i
  /-
    🎉 no goals
  -/


theorem iSup_iInf_of_antitone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (swap (· ≤ ·))] [Order.Frame α] {f : ι → ι' → α} (hf : ∀ i, Antitone (f i)) :
    ⨆ j, ⨅ i, f i j = ⨅ i, ⨆ j, f i j :=
  @iSup_iInf_of_monotone ι ι'ᵒᵈ α _ _ _ _ _ _ fun i => (hf i).dual_left


theorem iInf_iSup_of_monotone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (swap (· ≤ ·))] [Order.Coframe α] {f : ι → ι' → α} (hf : ∀ i, Monotone (f i)) :
    ⨅ j, ⨆ i, f i j = ⨆ i, ⨅ j, f i j :=
  iSup_iInf_of_antitone (α := αᵒᵈ) fun i => (hf i).dual_right


theorem iInf_iSup_of_antitone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [Nonempty ι']
    [IsDirected ι' (· ≤ ·)] [Order.Coframe α] {f : ι → ι' → α} (hf : ∀ i, Antitone (f i)) :
    ⨅ j, ⨆ i, f i j = ⨆ i, ⨅ j, f i j :=
  iSup_iInf_of_monotone (α := αᵒᵈ) fun i => (hf i).dual_right


/-- An increasing union distributes over finite intersection. -/
theorem iUnion_iInter_of_monotone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [IsDirected ι' (· ≤ ·)]
    [Nonempty ι'] {s : ι → ι' → Set α} (hs : ∀ i, Monotone (s i)) :
    ⋃ j : ι', ⋂ i : ι, s i j = ⋂ i : ι, ⋃ j : ι', s i j :=
  iSup_iInf_of_monotone hs


/-- A decreasing union distributes over finite intersection. -/
theorem iUnion_iInter_of_antitone {ι ι' α : Type*} [Finite ι] [Preorder ι']
    [IsDirected ι' (swap (· ≤ ·))] [Nonempty ι'] {s : ι → ι' → Set α} (hs : ∀ i, Antitone (s i)) :
    ⋃ j : ι', ⋂ i : ι, s i j = ⋂ i : ι, ⋃ j : ι', s i j :=
  iSup_iInf_of_antitone hs


/-- An increasing intersection distributes over finite union. -/
theorem iInter_iUnion_of_monotone {ι ι' α : Type*} [Finite ι] [Preorder ι']
    [IsDirected ι' (swap (· ≤ ·))] [Nonempty ι'] {s : ι → ι' → Set α} (hs : ∀ i, Monotone (s i)) :
    ⋂ j : ι', ⋃ i : ι, s i j = ⋃ i : ι, ⋂ j : ι', s i j :=
  iInf_iSup_of_monotone hs


/-- A decreasing intersection distributes over finite union. -/
theorem iInter_iUnion_of_antitone {ι ι' α : Type*} [Finite ι] [Preorder ι'] [IsDirected ι' (· ≤ ·)]
    [Nonempty ι'] {s : ι → ι' → Set α} (hs : ∀ i, Antitone (s i)) :
    ⋂ j : ι', ⋃ i : ι, s i j = ⋃ i : ι, ⋂ j : ι', s i j :=
  iInf_iSup_of_antitone hs


theorem iUnion_pi_of_monotone {ι ι' : Type*} [LinearOrder ι'] [Nonempty ι'] {α : ι → Type*}
    {I : Set ι} {s : ∀ i, ι' → Set (α i)} (hI : I.Finite) (hs : ∀ i ∈ I, Monotone (s i)) :
    ⋃ j : ι', I.pi (fun i => s i j) = I.pi fun i => ⋃ j, s i j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    inst✝¹ : LinearOrder ι'
    inst✝ : Nonempty ι'
    α : ι → Type u_3
    I : Set ι
    s : (i : ι) → ι' → Set (α i)
    hI : I.Finite
    hs : ∀ (i : ι), Membership.mem I i → Monotone (s i)
    ⊢ Eq (Set.iUnion fun j => I.pi fun i => s i j) (I.pi fun i => Set.iUnion fun j …
  -/
  simp only [pi_def, biInter_eq_iInter, preimage_iUnion]
  /-
    ι : Type u_1
    ι' : Type u_2
    inst✝¹ : LinearOrder ι'
    inst✝ : Nonempty ι'
    α : ι → Type u_3
    I : Set ι
    s : (i : ι) → ι' → Set (α i)
    hI : I.Finite
    hs : ∀ (i : ι), Membership.mem I i → Monotone (s i)
    ⊢ Eq (Set.iUnion fun j => Set.iInter fun x => Set.preimage (Function.eval ↑x)  …
  -/
  haveI := hI.fintype.finite
  /-
    ι : Type u_1
    ι' : Type u_2
    inst✝¹ : LinearOrder ι'
    inst✝ : Nonempty ι'
    α : ι → Type u_3
    I : Set ι
    s : (i : ι) → ι' → Set (α i)
    hI : I.Finite
    hs : ∀ (i : ι), Membership.mem I i → Monotone (s i)
    this : Finite ↑I
    ⊢ Eq (Set.iUnion fun j => Set.iInter fun x => Set.preimage (Function.eval ↑x)  …
  -/
  refine iUnion_iInter_of_monotone (ι' := ι') (fun (i : I) j₁ j₂ h => ?_)
  /-
    ι : Type u_1
    ι' : Type u_2
    inst✝¹ : LinearOrder ι'
    inst✝ : Nonempty ι'
    α : ι → Type u_3
    I : Set ι
    s : (i : ι) → ι' → Set (α i)
    hI : I.Finite
    hs : ∀ (i : ι), Membership.mem I i → Monotone (s i)
    this : Finite ↑I
    i : ↑I
    j₁ j₂ : ι'
    h : LE.le j₁ j₂
    ⊢ LE.le (Set.preimage (Function.eval ↑i) (s (↑i) j₁)) (Set.preimage (Function. …
  -/
  exact preimage_mono <| hs i i.2 h
  /-
    🎉 no goals
  -/


theorem iUnion_univ_pi_of_monotone {ι ι' : Type*} [LinearOrder ι'] [Nonempty ι'] [Finite ι]
    {α : ι → Type*} {s : ∀ i, ι' → Set (α i)} (hs : ∀ i, Monotone (s i)) :
    ⋃ j : ι', pi univ (fun i => s i j) = pi univ fun i => ⋃ j, s i j :=
  iUnion_pi_of_monotone finite_univ fun i _ => hs i


/-- A finite set is bounded above. -/
protected theorem Finite.bddAbove (hs : s.Finite) : BddAbove s :=
  Finite.induction_on hs bddAbove_empty fun _ _ h => h.insert _


/-- A finite union of sets which are all bounded above is still bounded above. -/
theorem Finite.bddAbove_biUnion {I : Set β} {S : β → Set α} (H : I.Finite) :
    BddAbove (⋃ i ∈ I, S i) ↔ ∀ i ∈ I, BddAbove (S i) :=
                            /-
                              α : Type u
                              β : Type v
                              inst✝² : Preorder α
                              inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
                              inst✝ : Nonempty α
                              I : Set β
                              S : β → Set α
                              H : I.Finite
                              ⊢ Iff (BddAbove (Set.iUnion fun i => Set.iUnion fun h => S i)) (∀ (i : β), Mem …
                            -/
  Finite.induction_on H (by simp only [biUnion_empty, bddAbove_empty, forall_mem_empty])
                            /-
                              🎉 no goals
                            -/
                     /-
                       α : Type u
                       β : Type v
                       inst✝² : Preorder α
                       inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
                       inst✝ : Nonempty α
                       I : Set β
                       S : β → Set α
                       H : I.Finite
                       a✝ : β
                       s✝ : Set β
                       x✝¹ : Not (Membership.mem s✝ a✝)
                       x✝ : s✝.Finite
                       hs : Iff (BddAbove (Set.iUnion fun i => Set.iUnion fun h => S i)) (∀ (i : β),  …
                       ⊢ Iff (BddAbove (Set.iUnion fun i => Set.iUnion fun h => S i)) (∀ (i : β), Mem …
                     -/
    fun _ _ hs => by simp only [biUnion_insert, forall_mem_insert, bddAbove_union, hs]
                     /-
                       🎉 no goals
                     -/


theorem infinite_of_not_bddAbove : ¬BddAbove s → s.Infinite :=
  mt Finite.bddAbove


/-- A finite set is bounded below. -/
protected theorem Finite.bddBelow (hs : s.Finite) : BddBelow s :=
  Finite.bddAbove (α := αᵒᵈ) hs


/-- A finite union of sets which are all bounded below is still bounded below. -/
theorem Finite.bddBelow_biUnion {I : Set β} {S : β → Set α} (H : I.Finite) :
    BddBelow (⋃ i ∈ I, S i) ↔ ∀ i ∈ I, BddBelow (S i) :=
  Finite.bddAbove_biUnion (α := αᵒᵈ) H


theorem infinite_of_not_bddBelow : ¬BddBelow s → s.Infinite := mt Finite.bddBelow


/-- A finset is bounded above. -/
protected theorem bddAbove [SemilatticeSup α] [Nonempty α] (s : Finset α) : BddAbove (↑s : Set α) :=
  s.finite_toSet.bddAbove


/-- A finset is bounded below. -/
protected theorem bddBelow [SemilatticeInf α] [Nonempty α] (s : Finset α) : BddBelow (↑s : Set α) :=
  s.finite_toSet.bddBelow


lemma Set.finite_diff_iUnion_Ioo (s : Set α) : (s \ ⋃ (x ∈ s) (y ∈ s), Ioo x y).Finite :=
  Set.finite_of_forall_not_lt_lt fun _x hx _y hy _z hz hxy hyz => hy.2 <| mem_iUnion₂_of_mem hx.1 <|
    mem_iUnion₂_of_mem hz.1 ⟨hxy, hyz⟩


lemma Set.finite_diff_iUnion_Ioo' (s : Set α) : (s \ ⋃ x : s × s, Ioo x.1 x.2).Finite := by
  /-
    α : Type u
    inst✝ : LinearOrder α
    s : Set α
    ⊢ (SDiff.sdiff s (Set.iUnion fun x => Set.Ioo ↑x.1 ↑x.2)).Finite
  -/
  simpa only [iUnion, iSup_prod, iSup_subtype] using s.finite_diff_iUnion_Ioo
  /-
    🎉 no goals
  -/


theorem DirectedOn.exists_mem_subset_of_finset_subset_biUnion {α ι : Type*} {f : ι → Set α}
    {c : Set ι} (hn : c.Nonempty) (hc : DirectedOn (fun i j => f i ⊆ f j) c) {s : Finset α}
    (hs : (s : Set α) ⊆ ⋃ i ∈ c, f i) : ∃ i ∈ c, (s : Set α) ⊆ f i := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    c : Set ι
    hn : c.Nonempty
    hc : DirectedOn (fun i j => HasSubset.Subset (f i) (f j)) c
    s : Finset α
    hs : HasSubset.Subset (↑s) (Set.iUnion fun i => Set.iUnion fun h => f i)
    ⊢ Exists fun i => And (Membership.mem c i) (HasSubset.Subset (↑s) (f i))
  -/
  rw [Set.biUnion_eq_iUnion] at hs
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    c : Set ι
    hn : c.Nonempty
    hc : DirectedOn (fun i j => HasSubset.Subset (f i) (f j)) c
    s : Finset α
    hs : HasSubset.Subset (↑s) (Set.iUnion fun x => f ↑x)
    ⊢ Exists fun i => And (Membership.mem c i) (HasSubset.Subset (↑s) (f i))
  -/
  haveI := hn.coe_sort
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    c : Set ι
    hn : c.Nonempty
    hc : DirectedOn (fun i j => HasSubset.Subset (f i) (f j)) c
    s : Finset α
    hs : HasSubset.Subset (↑s) (Set.iUnion fun x => f ↑x)
    this : Nonempty ↑c
    ⊢ Exists fun i => And (Membership.mem c i) (HasSubset.Subset (↑s) (f i))
  -/
  simpa using (directed_comp.2 hc.directed_val).exists_mem_subset_of_finset_subset_biUnion hs
  /-
    🎉 no goals
  -/


