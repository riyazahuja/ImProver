/-- A filter basis `B` on a type `α` is a nonempty collection of sets of `α`
such that the intersection of two elements of this collection contains some element
of the collection. -/
structure FilterBasis (α : Type*) where
  /-- Sets of a filter basis. -/
  sets : Set (Set α)
  /-- The set of filter basis sets is nonempty. -/
  nonempty : sets.Nonempty
  /-- The set of filter basis sets is directed downwards. -/
  inter_sets {x y} : x ∈ sets → y ∈ sets → ∃ z ∈ sets, z ⊆ x ∩ y


instance FilterBasis.nonempty_sets (B : FilterBasis α) : Nonempty B.sets :=
  B.nonempty.to_subtype

-- Porting note: this instance was reducible but it doesn't work the same way in Lean 4

/-- If `B` is a filter basis on `α`, and `U` a subset of `α` then we can write `U ∈ B` as
on paper. -/
instance {α : Type*} : Membership (Set α) (FilterBasis α) :=
  ⟨fun B U => U ∈ B.sets⟩


@[simp] theorem FilterBasis.mem_sets {s : Set α} {B : FilterBasis α} : s ∈ B.sets ↔ s ∈ B := Iff.rfl

-- For illustration purposes, the filter basis defining `(atTop : Filter ℕ)`

instance : Inhabited (FilterBasis ℕ) :=
  ⟨{  sets := range Ici
      nonempty := ⟨Ici 0, mem_range_self 0⟩
      inter_sets := by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          ι' : Sort u_5
          ⊢ ∀ {x y : Set Nat}, Membership.mem (Set.range Set.Ici) x → Membership.mem (Se …
        -/
        rintro _ _ ⟨n, rfl⟩ ⟨m, rfl⟩
        /-
          case intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          ι' : Sort u_5
          n m : Nat
          ⊢ Exists fun z => And (Membership.mem (Set.range Set.Ici) z) (HasSubset.Subset …
        -/
        exact ⟨Ici (max n m), mem_range_self _, Ici_inter_Ici.symm.subset⟩ }⟩
        /-
          🎉 no goals
        -/


/-- View a filter as a filter basis. -/
def Filter.asBasis (f : Filter α) : FilterBasis α :=
  ⟨f.sets, ⟨univ, univ_mem⟩, fun {x y} hx hy => ⟨x ∩ y, inter_mem hx hy, subset_rfl⟩⟩

-- Porting note: was `protected` in Lean 3 but `protected` didn't work; removed

/-- `IsBasis p s` means the image of `s` bounded by `p` is a filter basis. -/
structure Filter.IsBasis (p : ι → Prop) (s : ι → Set α) : Prop where
  /-- There exists at least one `i` that satisfies `p`. -/
  nonempty : ∃ i, p i
  /-- `s` is directed downwards on `i` such that `p i`. -/
  inter : ∀ {i j}, p i → p j → ∃ k, p k ∧ s k ⊆ s i ∩ s j


/-- Constructs a filter basis from an indexed family of sets satisfying `IsBasis`. -/
protected def filterBasis {p : ι → Prop} {s : ι → Set α} (h : IsBasis p s) : FilterBasis α where
  sets := { t | ∃ i, p i ∧ s i = t }
  nonempty :=
    let ⟨i, hi⟩ := h.nonempty
    ⟨s i, ⟨i, hi, rfl⟩⟩
  inter_sets := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      p : ι → Prop
      s : ι → Set α
      h : Filter.IsBasis p s
      ⊢ ∀ {x y : Set α}, Membership.mem (setOf fun t => Exists fun i => And (p i) (E …
    -/
    rintro _ _ ⟨i, hi, rfl⟩ ⟨j, hj, rfl⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      p : ι → Prop
      s : ι → Set α
      h : Filter.IsBasis p s
      i : ι
      hi : p i
      j : ι
      hj : p j
      ⊢ Exists fun z => And (Membership.mem (setOf fun t => Exists fun i => And (p i …
    -/
    rcases h.inter hi hj with ⟨k, hk, hk'⟩
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      p : ι → Prop
      s : ι → Set α
      h : Filter.IsBasis p s
      i : ι
      hi : p i
      j : ι
      hj : p j
      k : ι
      hk : p k
      hk' : HasSubset.Subset (s k) (Inter.inter (s i) (s j))
      ⊢ Exists fun z => And (Membership.mem (setOf fun t => Exists fun i => And (p i …
    -/
    exact ⟨_, ⟨k, hk, rfl⟩, hk'⟩
    /-
      🎉 no goals
    -/


theorem mem_filterBasis_iff {U : Set α} : U ∈ h.filterBasis ↔ ∃ i, p i ∧ s i = U :=
  Iff.rfl


/-- The filter associated to a filter basis. -/
protected def filter (B : FilterBasis α) : Filter α where
  sets := { s | ∃ t ∈ B, t ⊆ s }
  univ_sets := B.nonempty.imp fun s s_in => ⟨s_in, s.subset_univ⟩
  sets_of_superset := fun ⟨s, s_in, h⟩ hxy => ⟨s, s_in, Set.Subset.trans h hxy⟩
  inter_sets := fun ⟨_s, s_in, hs⟩ ⟨_t, t_in, ht⟩ =>
    let ⟨u, u_in, u_sub⟩ := B.inter_sets s_in t_in
    ⟨u, u_in, u_sub.trans (inter_subset_inter hs ht)⟩


theorem mem_filter_iff (B : FilterBasis α) {U : Set α} : U ∈ B.filter ↔ ∃ s ∈ B, s ⊆ U :=
  Iff.rfl


theorem mem_filter_of_mem (B : FilterBasis α) {U : Set α} : U ∈ B → U ∈ B.filter := fun U_in =>
  ⟨U, U_in, Subset.refl _⟩


theorem eq_iInf_principal (B : FilterBasis α) : B.filter = ⨅ s : B.sets, 𝓟 s := by
  have : Directed (· ≥ ·) fun s : B.sets => 𝓟 (s : Set α) := by
    rintro ⟨U, U_in⟩ ⟨V, V_in⟩
    rcases B.inter_sets U_in V_in with ⟨W, W_in, W_sub⟩
    use ⟨W, W_in⟩
    simp only [le_principal_iff, mem_principal, Subtype.coe_mk]
    exact subset_inter_iff.mp W_sub
  /-
    α : Type u_1
    B : FilterBasis α
    this : Directed (fun x1 x2 => GE.ge x1 x2) fun s => Filter.principal ↑s
    ⊢ Eq B.filter (iInf fun s => Filter.principal ↑s)
  -/
  ext U
  /-
    case h
    α : Type u_1
    B : FilterBasis α
    this : Directed (fun x1 x2 => GE.ge x1 x2) fun s => Filter.principal ↑s
    U : Set α
    ⊢ Iff (Membership.mem B.filter U) (Membership.mem (iInf fun s => Filter.princi …
  -/
  simp [mem_filter_iff, mem_iInf_of_directed this]
  /-
    🎉 no goals
  -/


protected theorem generate (B : FilterBasis α) : generate B.sets = B.filter := by
  /-
    α : Type u_1
    B : FilterBasis α
    ⊢ Eq (Filter.generate B.sets) B.filter
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      B : FilterBasis α
      ⊢ LE.le (Filter.generate B.sets) B.filter
    -/
  · intro U U_in
    /-
      case a
      α : Type u_1
      B : FilterBasis α
      U : Set α
      U_in : Membership.mem B.filter U
      ⊢ Membership.mem (Filter.generate B.sets) U
    -/
    rcases B.mem_filter_iff.mp U_in with ⟨V, V_in, h⟩
    /-
      case a.intro.intro
      α : Type u_1
      B : FilterBasis α
      U : Set α
      U_in : Membership.mem B.filter U
      V : Set α
      V_in : Membership.mem B V
      h : HasSubset.Subset V U
      ⊢ Membership.mem (Filter.generate B.sets) U
    -/
    exact GenerateSets.superset (GenerateSets.basic V_in) h
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      B : FilterBasis α
      ⊢ LE.le B.filter (Filter.generate B.sets)
    -/
  · rw [le_generate_iff]
    /-
      case a
      α : Type u_1
      B : FilterBasis α
      ⊢ HasSubset.Subset B.sets B.filter.sets
    -/
    apply mem_filter_of_mem
    /-
      🎉 no goals
    -/


/-- Constructs a filter from an indexed family of sets satisfying `IsBasis`. -/
protected def filter (h : IsBasis p s) : Filter α :=
  h.filterBasis.filter


protected theorem mem_filter_iff (h : IsBasis p s) {U : Set α} :
    U ∈ h.filter ↔ ∃ i, p i ∧ s i ⊆ U := by
  simp only [IsBasis.filter, FilterBasis.mem_filter_iff, mem_filterBasis_iff,
    exists_exists_and_eq_and]


theorem filter_eq_generate (h : IsBasis p s) : h.filter = generate { U | ∃ i, p i ∧ s i = U } := by
  /-
    α : Type u_1
    ι : Sort u_4
    p : ι → Prop
    s : ι → Set α
    h : Filter.IsBasis p s
    ⊢ Eq h.filter (Filter.generate (setOf fun U => Exists fun i => And (p i) (Eq ( …
  -/
  erw [h.filterBasis.generate]; rfl
                                /-
                                  🎉 no goals
                                -/


/-- We say that a filter `l` has a basis `s : ι → Set α` bounded by `p : ι → Prop`,
if `t ∈ l` if and only if `t` includes `s i` for some `i` such that `p i`. -/
structure HasBasis (l : Filter α) (p : ι → Prop) (s : ι → Set α) : Prop where
  /-- A set `t` belongs to a filter `l` iff it includes an element of the basis. -/
  mem_iff' : ∀ t : Set α, t ∈ l ↔ ∃ i, p i ∧ s i ⊆ t


theorem hasBasis_generate (s : Set (Set α)) :
    (generate s).HasBasis (fun t => Set.Finite t ∧ t ⊆ s) fun t => ⋂₀ t :=
               /-
                 α : Type u_1
                 s : Set (Set α)
                 U : Set α
                 ⊢ Iff (Membership.mem (Filter.generate s) U) (Exists fun i => And (And i.Finit …
               -/
  ⟨fun U => by simp only [mem_generate_iff, exists_prop, and_assoc, and_left_comm]⟩
               /-
                 🎉 no goals
               -/


/-- The smallest filter basis containing a given collection of sets. -/
def FilterBasis.ofSets (s : Set (Set α)) : FilterBasis α where
  sets := sInter '' { t | Set.Finite t ∧ t ⊆ s }
  nonempty := ⟨univ, ∅, ⟨⟨finite_empty, empty_subset s⟩, sInter_empty⟩⟩
  inter_sets := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s✝ : ι → Set α
      t : Set α
      i : ι
      p' : ι' → Prop
      s' : ι' → Set α
      i' : ι'
      s : Set (Set α)
      ⊢ ∀ {x y : Set α}, Membership.mem (Set.image Set.sInter (setOf fun t => And t. …
    -/
    rintro _ _ ⟨a, ⟨fina, suba⟩, rfl⟩ ⟨b, ⟨finb, subb⟩, rfl⟩
    exact ⟨⋂₀ (a ∪ b), mem_image_of_mem _ ⟨fina.union finb, union_subset suba subb⟩,
        (sInter_union _ _).subset⟩


lemma FilterBasis.ofSets_sets (s : Set (Set α)) :
    (FilterBasis.ofSets s).sets = sInter '' { t | Set.Finite t ∧ t ⊆ s } :=
  rfl

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

/-- Definition of `HasBasis` unfolded with implicit set argument. -/
theorem HasBasis.mem_iff (hl : l.HasBasis p s) : t ∈ l ↔ ∃ i, p i ∧ s i ⊆ t :=
  hl.mem_iff' t


theorem HasBasis.eq_of_same_basis (hl : l.HasBasis p s) (hl' : l'.HasBasis p s) : l = l' := by
  /-
    α : Type u_1
    ι : Sort u_4
    l l' : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    hl' : l'.HasBasis p s
    ⊢ Eq l l'
  -/
  ext t
  /-
    case h
    α : Type u_1
    ι : Sort u_4
    l l' : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    hl' : l'.HasBasis p s
    t : Set α
    ⊢ Iff (Membership.mem l t) (Membership.mem l' t)
  -/
  rw [hl.mem_iff, hl'.mem_iff]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem hasBasis_iff : l.HasBasis p s ↔ ∀ t, t ∈ l ↔ ∃ i, p i ∧ s i ⊆ t :=
  ⟨fun ⟨h⟩ => h, fun h => ⟨h⟩⟩


theorem HasBasis.ex_mem (h : l.HasBasis p s) : ∃ i, p i :=
  (h.mem_iff.mp univ_mem).imp fun _ => And.left


protected theorem HasBasis.nonempty (h : l.HasBasis p s) : Nonempty ι :=
  nonempty_of_exists h.ex_mem


protected theorem IsBasis.hasBasis (h : IsBasis p s) : HasBasis h.filter p s :=
               /-
                 α : Type u_1
                 ι : Sort u_4
                 p : ι → Prop
                 s : ι → Set α
                 h : Filter.IsBasis p s
                 t : Set α
                 ⊢ Iff (Membership.mem h.filter t) (Exists fun i => And (p i) (HasSubset.Subset …
               -/
  ⟨fun t => by simp only [h.mem_filter_iff, exists_prop]⟩
               /-
                 🎉 no goals
               -/


protected theorem HasBasis.mem_of_superset (hl : l.HasBasis p s) (hi : p i) (ht : s i ⊆ t) :
    t ∈ l :=
  hl.mem_iff.2 ⟨i, hi, ht⟩


theorem HasBasis.mem_of_mem (hl : l.HasBasis p s) (hi : p i) : s i ∈ l :=
  hl.mem_of_superset hi Subset.rfl


/-- Index of a basis set such that `s i ⊆ t` as an element of `Subtype p`. -/
noncomputable def HasBasis.index (h : l.HasBasis p s) (t : Set α) (ht : t ∈ l) : { i : ι // p i } :=
  ⟨(h.mem_iff.1 ht).choose, (h.mem_iff.1 ht).choose_spec.1⟩


theorem HasBasis.property_index (h : l.HasBasis p s) (ht : t ∈ l) : p (h.index t ht) :=
  (h.index t ht).2


theorem HasBasis.set_index_mem (h : l.HasBasis p s) (ht : t ∈ l) : s (h.index t ht) ∈ l :=
  h.mem_of_mem <| h.property_index _


theorem HasBasis.set_index_subset (h : l.HasBasis p s) (ht : t ∈ l) : s (h.index t ht) ⊆ t :=
  (h.mem_iff.1 ht).choose_spec.2


theorem HasBasis.isBasis (h : l.HasBasis p s) : IsBasis p s where
  nonempty := h.ex_mem
  inter hi hj := by
    /-
      α : Type u_1
      ι : Sort u_4
      l : Filter α
      p : ι → Prop
      s : ι → Set α
      h : l.HasBasis p s
      i✝ j✝ : ι
      hi : p i✝
      hj : p j✝
      ⊢ Exists fun k => And (p k) (HasSubset.Subset (s k) (Inter.inter (s i✝) (s j✝)))
    -/
    simpa only [h.mem_iff] using inter_mem (h.mem_of_mem hi) (h.mem_of_mem hj)
    /-
      🎉 no goals
    -/


theorem HasBasis.filter_eq (h : l.HasBasis p s) : h.isBasis.filter = l := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    ⊢ Eq ⋯.filter l
  -/
  ext U
  /-
    case h
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    U : Set α
    ⊢ Iff (Membership.mem ⋯.filter U) (Membership.mem l U)
  -/
  simp [h.mem_iff, IsBasis.mem_filter_iff]
  /-
    🎉 no goals
  -/


theorem HasBasis.eq_generate (h : l.HasBasis p s) : l = generate { U | ∃ i, p i ∧ s i = U } := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    ⊢ Eq l (Filter.generate (setOf fun U => Exists fun i => And (p i) (Eq (s i) U)))
  -/
  rw [← h.isBasis.filter_eq_generate, h.filter_eq]
  /-
    🎉 no goals
  -/


theorem generate_eq_generate_inter (s : Set (Set α)) :
    generate s = generate (sInter '' { t | Set.Finite t ∧ t ⊆ s }) := by
  /-
    α : Type u_1
    s : Set (Set α)
    ⊢ Eq (Filter.generate s) (Filter.generate (Set.image Set.sInter (setOf fun t = …
  -/
  rw [← FilterBasis.ofSets_sets, FilterBasis.generate, ← (hasBasis_generate s).filter_eq]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem ofSets_filter_eq_generate (s : Set (Set α)) :
    (FilterBasis.ofSets s).filter = generate s := by
  /-
    α : Type u_1
    s : Set (Set α)
    ⊢ Eq (Filter.FilterBasis.ofSets s).filter (Filter.generate s)
  -/
  rw [← (FilterBasis.ofSets s).generate, FilterBasis.ofSets_sets, ← generate_eq_generate_inter]
  /-
    🎉 no goals
  -/


protected theorem _root_.FilterBasis.hasBasis (B : FilterBasis α) :
    HasBasis B.filter (fun s : Set α => s ∈ B) id :=
  ⟨fun _ => B.mem_filter_iff⟩


theorem HasBasis.to_hasBasis' (hl : l.HasBasis p s) (h : ∀ i, p i → ∃ i', p' i' ∧ s' i' ⊆ s i)
    (h' : ∀ i', p' i' → s' i' ∈ l) : l.HasBasis p' s' := by
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
    h' : ∀ (i' : ι'), p' i' → Membership.mem l (s' i')
    ⊢ l.HasBasis p' s'
  -/
  refine ⟨fun t => ⟨fun ht => ?_, fun ⟨i', hi', ht⟩ => mem_of_superset (h' i' hi') ht⟩⟩
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
    h' : ∀ (i' : ι'), p' i' → Membership.mem l (s' i')
    t : Set α
    ht : Membership.mem l t
    ⊢ Exists fun i => And (p' i) (HasSubset.Subset (s' i) t)
  -/
  rcases hl.mem_iff.1 ht with ⟨i, hi, ht⟩
  /-
    case intro.intro
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
    h' : ∀ (i' : ι'), p' i' → Membership.mem l (s' i')
    t : Set α
    ht✝ : Membership.mem l t
    i : ι
    hi : p i
    ht : HasSubset.Subset (s i) t
    ⊢ Exists fun i => And (p' i) (HasSubset.Subset (s' i) t)
  -/
  rcases h i hi with ⟨i', hi', hs's⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
    h' : ∀ (i' : ι'), p' i' → Membership.mem l (s' i')
    t : Set α
    ht✝ : Membership.mem l t
    i : ι
    hi : p i
    ht : HasSubset.Subset (s i) t
    i' : ι'
    hi' : p' i'
    hs's : HasSubset.Subset (s' i') (s i)
    ⊢ Exists fun i => And (p' i) (HasSubset.Subset (s' i) t)
  -/
  exact ⟨i', hi', hs's.trans ht⟩
  /-
    🎉 no goals
  -/


theorem HasBasis.to_hasBasis (hl : l.HasBasis p s) (h : ∀ i, p i → ∃ i', p' i' ∧ s' i' ⊆ s i)
    (h' : ∀ i', p' i' → ∃ i, p i ∧ s i ⊆ s' i') : l.HasBasis p' s' :=
  hl.to_hasBasis' h fun i' hi' =>
    let ⟨i, hi, hss'⟩ := h' i' hi'
    hl.mem_iff.2 ⟨i, hi, hss'⟩


protected lemma HasBasis.congr (hl : l.HasBasis p s) {p' s'} (hp : ∀ i, p i ↔ p' i)
    (hs : ∀ i, p i → s i = s' i) : l.HasBasis p' s' :=
              /-
                α : Type u_1
                ι : Sort u_4
                l : Filter α
                p : ι → Prop
                s : ι → Set α
                hl : l.HasBasis p s
                p' : ι → Prop
                s' : ι → Set α
                hp : ∀ (i : ι), Iff (p i) (p' i)
                hs : ∀ (i : ι), p i → Eq (s i) (s' i)
                t : Set α
                ⊢ Iff (Membership.mem l t) (Exists fun i => And (p' i) (HasSubset.Subset (s' i …
              -/
  ⟨fun t ↦ by simp only [hl.mem_iff, ← hp]; exact exists_congr fun i ↦
    and_congr_right fun hi ↦ hs i hi ▸ Iff.rfl⟩


theorem HasBasis.to_subset (hl : l.HasBasis p s) {t : ι → Set α} (h : ∀ i, p i → t i ⊆ s i)
    (ht : ∀ i, p i → t i ∈ l) : l.HasBasis p t :=
  hl.to_hasBasis' (fun i hi => ⟨i, hi, h i hi⟩) ht


theorem HasBasis.eventually_iff (hl : l.HasBasis p s) {q : α → Prop} :
                                                             /-
                                                               α : Type u_1
                                                               ι : Sort u_4
                                                               l : Filter α
                                                               p : ι → Prop
                                                               s : ι → Set α
                                                               hl : l.HasBasis p s
                                                               q : α → Prop
                                                               ⊢ Iff (Filter.Eventually (fun x => q x) l) (Exists fun i => And (p i) (∀ ⦃x :  …
                                                             -/
    (∀ᶠ x in l, q x) ↔ ∃ i, p i ∧ ∀ ⦃x⦄, x ∈ s i → q x := by simpa using hl.mem_iff
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem HasBasis.frequently_iff (hl : l.HasBasis p s) {q : α → Prop} :
    (∃ᶠ x in l, q x) ↔ ∀ i, p i → ∃ x ∈ s i, q x := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    q : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => q x) l) (∀ (i : ι), p i → Exists fun x => A …
  -/
  simp only [Filter.Frequently, hl.eventually_iff]; push_neg; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.exists_iff (hl : l.HasBasis p s) {P : Set α → Prop}
    (mono : ∀ ⦃s t⦄, s ⊆ t → P t → P s) : (∃ s ∈ l, P s) ↔ ∃ i, p i ∧ P (s i) :=
  ⟨fun ⟨_s, hs, hP⟩ =>
    let ⟨i, hi, his⟩ := hl.mem_iff.1 hs
    ⟨i, hi, mono his hP⟩,
    fun ⟨i, hi, hP⟩ => ⟨s i, hl.mem_of_mem hi, hP⟩⟩


theorem HasBasis.forall_iff (hl : l.HasBasis p s) {P : Set α → Prop}
    (mono : ∀ ⦃s t⦄, s ⊆ t → P s → P t) : (∀ s ∈ l, P s) ↔ ∀ i, p i → P (s i) :=
  ⟨fun H i hi => H (s i) <| hl.mem_of_mem hi, fun H _s hs =>
    let ⟨i, hi, his⟩ := hl.mem_iff.1 hs
    mono his (H i hi)⟩


protected theorem HasBasis.neBot_iff (hl : l.HasBasis p s) :
    NeBot l ↔ ∀ {i}, p i → (s i).Nonempty :=
  forall_mem_nonempty_iff_neBot.symm.trans <| hl.forall_iff fun _ _ => Nonempty.mono


theorem HasBasis.eq_bot_iff (hl : l.HasBasis p s) : l = ⊥ ↔ ∃ i, p i ∧ s i = ∅ :=
  not_iff_not.1 <| neBot_iff.symm.trans <|
                             /-
                               α : Type u_1
                               ι : Sort u_4
                               l : Filter α
                               p : ι → Prop
                               s : ι → Set α
                               hl : l.HasBasis p s
                               ⊢ Iff (∀ {i : ι}, p i → (s i).Nonempty) (Not (Exists fun i => And (p i) (Eq (s …
                             -/
    hl.neBot_iff.trans <| by simp only [not_exists, not_and, nonempty_iff_ne_empty]
                             /-
                               🎉 no goals
                             -/


theorem generate_neBot_iff {s : Set (Set α)} :
    NeBot (generate s) ↔ ∀ t, t ⊆ s → t.Finite → (⋂₀ t).Nonempty :=
                                              /-
                                                α : Type u_1
                                                s : Set (Set α)
                                                ⊢ Iff (∀ {i : Set (Set α)}, And i.Finite (HasSubset.Subset i s) → i.sInter.Non …
                                              -/
  (hasBasis_generate s).neBot_iff.trans <| by simp only [← and_imp, and_comm]
                                              /-
                                                🎉 no goals
                                              -/


theorem basis_sets (l : Filter α) : l.HasBasis (fun s : Set α => s ∈ l) id :=
  ⟨fun _ => exists_mem_subset_iff.symm⟩


theorem asBasis_filter (f : Filter α) : f.asBasis.filter = f :=
  Filter.ext fun _ => exists_mem_subset_iff


theorem hasBasis_self {l : Filter α} {P : Set α → Prop} :
    HasBasis l (fun s => s ∈ l ∧ P s) id ↔ ∀ t ∈ l, ∃ r ∈ l, P r ∧ r ⊆ t := by
  /-
    α : Type u_1
    l : Filter α
    P : Set α → Prop
    ⊢ Iff (l.HasBasis (fun s => And (Membership.mem l s) (P s)) id) (∀ (t : Set α) …
  -/
  simp only [hasBasis_iff, id, and_assoc]
  exact forall_congr' fun s =>
    ⟨fun h => h.1, fun h => ⟨h, fun ⟨t, hl, _, hts⟩ => mem_of_superset hl hts⟩⟩


theorem HasBasis.comp_surjective (h : l.HasBasis p s) {g : ι' → ι} (hg : Function.Surjective g) :
    l.HasBasis (p ∘ g) (s ∘ g) :=
  ⟨fun _ => h.mem_iff.trans hg.exists⟩


theorem HasBasis.comp_equiv (h : l.HasBasis p s) (e : ι' ≃ ι) : l.HasBasis (p ∘ e) (s ∘ e) :=
  h.comp_surjective e.surjective


theorem HasBasis.to_image_id' (h : l.HasBasis p s) : l.HasBasis (fun t ↦ ∃ i, p i ∧ s i = t) id :=
              /-
                α : Type u_1
                ι : Sort u_4
                l : Filter α
                p : ι → Prop
                s : ι → Set α
                h : l.HasBasis p s
                x✝ : Set α
                ⊢ Iff (Membership.mem l x✝) (Exists fun i => And (Exists fun i_1 => And (p i_1 …
              -/
  ⟨fun _ ↦ by simp [h.mem_iff]⟩
              /-
                🎉 no goals
              -/


theorem HasBasis.to_image_id {ι : Type*} {p : ι → Prop} {s : ι → Set α} (h : l.HasBasis p s) :
    l.HasBasis (· ∈ s '' {i | p i}) id :=
  h.to_image_id'


/-- If `{s i | p i}` is a basis of a filter `l` and each `s i` includes `s j` such that
`p j ∧ q j`, then `{s j | p j ∧ q j}` is a basis of `l`. -/
theorem HasBasis.restrict (h : l.HasBasis p s) {q : ι → Prop}
    (hq : ∀ i, p i → ∃ j, p j ∧ q j ∧ s j ⊆ s i) : l.HasBasis (fun i => p i ∧ q i) s := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    q : ι → Prop
    hq : ∀ (i : ι), p i → Exists fun j => And (p j) (And (q j) (HasSubset.Subset ( …
    ⊢ l.HasBasis (fun i => And (p i) (q i)) s
  -/
  refine ⟨fun t => ⟨fun ht => ?_, fun ⟨i, hpi, hti⟩ => h.mem_iff.2 ⟨i, hpi.1, hti⟩⟩⟩
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    q : ι → Prop
    hq : ∀ (i : ι), p i → Exists fun j => And (p j) (And (q j) (HasSubset.Subset ( …
    t : Set α
    ht : Membership.mem l t
    ⊢ Exists fun i => And (And (p i) (q i)) (HasSubset.Subset (s i) t)
  -/
  rcases h.mem_iff.1 ht with ⟨i, hpi, hti⟩
  /-
    case intro.intro
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    q : ι → Prop
    hq : ∀ (i : ι), p i → Exists fun j => And (p j) (And (q j) (HasSubset.Subset ( …
    t : Set α
    ht : Membership.mem l t
    i : ι
    hpi : p i
    hti : HasSubset.Subset (s i) t
    ⊢ Exists fun i => And (And (p i) (q i)) (HasSubset.Subset (s i) t)
  -/
  rcases hq i hpi with ⟨j, hpj, hqj, hji⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    q : ι → Prop
    hq : ∀ (i : ι), p i → Exists fun j => And (p j) (And (q j) (HasSubset.Subset ( …
    t : Set α
    ht : Membership.mem l t
    i : ι
    hpi : p i
    hti : HasSubset.Subset (s i) t
    j : ι
    hpj : p j
    hqj : q j
    hji : HasSubset.Subset (s j) (s i)
    ⊢ Exists fun i => And (And (p i) (q i)) (HasSubset.Subset (s i) t)
  -/
  exact ⟨j, ⟨hpj, hqj⟩, hji.trans hti⟩
  /-
    🎉 no goals
  -/


/-- If `{s i | p i}` is a basis of a filter `l` and `V ∈ l`, then `{s i | p i ∧ s i ⊆ V}`
is a basis of `l`. -/
theorem HasBasis.restrict_subset (h : l.HasBasis p s) {V : Set α} (hV : V ∈ l) :
    l.HasBasis (fun i => p i ∧ s i ⊆ V) s :=
  h.restrict fun _i hi => (h.mem_iff.1 (inter_mem hV (h.mem_of_mem hi))).imp fun _j hj =>
    ⟨hj.1, subset_inter_iff.1 hj.2⟩


theorem HasBasis.hasBasis_self_subset {p : Set α → Prop} (h : l.HasBasis (fun s => s ∈ l ∧ p s) id)
    {V : Set α} (hV : V ∈ l) : l.HasBasis (fun s => s ∈ l ∧ p s ∧ s ⊆ V) id := by
  /-
    α : Type u_1
    l : Filter α
    p : Set α → Prop
    h : l.HasBasis (fun s => And (Membership.mem l s) (p s)) id
    V : Set α
    hV : Membership.mem l V
    ⊢ l.HasBasis (fun s => And (Membership.mem l s) (And (p s) (HasSubset.Subset s …
  -/
  simpa only [and_assoc] using h.restrict_subset hV
  /-
    🎉 no goals
  -/


theorem HasBasis.ge_iff (hl' : l'.HasBasis p' s') : l ≤ l' ↔ ∀ i', p' i' → s' i' ∈ l :=
  ⟨fun h _i' hi' => h <| hl'.mem_of_mem hi', fun h _s hs =>
    let ⟨_i', hi', hs⟩ := hl'.mem_iff.1 hs
    mem_of_superset (h _ hi') hs⟩

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.le_iff (hl : l.HasBasis p s) : l ≤ l' ↔ ∀ t ∈ l', ∃ i, p i ∧ s i ⊆ t := by
  /-
    α : Type u_1
    ι : Sort u_4
    l l' : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    ⊢ Iff (LE.le l l') (∀ (t : Set α), Membership.mem l' t → Exists fun i => And ( …
  -/
  simp only [le_def, hl.mem_iff]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.le_basis_iff (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    l ≤ l' ↔ ∀ i', p' i' → ∃ i, p i ∧ s i ⊆ s' i' := by
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l l' : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    hl' : l'.HasBasis p' s'
    ⊢ Iff (LE.le l l') (∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset. …
  -/
  simp only [hl'.ge_iff, hl.mem_iff]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.ext (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s')
    (h : ∀ i, p i → ∃ i', p' i' ∧ s' i' ⊆ s i) (h' : ∀ i', p' i' → ∃ i, p i ∧ s i ⊆ s' i') :
    l = l' := by
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    l l' : Filter α
    p : ι → Prop
    s : ι → Set α
    p' : ι' → Prop
    s' : ι' → Set α
    hl : l.HasBasis p s
    hl' : l'.HasBasis p' s'
    h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
    h' : ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s …
    ⊢ Eq l l'
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
      h' : ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s …
      ⊢ LE.le l l'
    -/
  · rw [hl.le_basis_iff hl']
    /-
      case a
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
      h' : ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s …
      ⊢ ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s' i …
    -/
    simpa using h'
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
      h' : ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s …
      ⊢ LE.le l' l
    -/
  · rw [hl'.le_basis_iff hl]
    /-
      case a
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      h : ∀ (i : ι), p i → Exists fun i' => And (p' i') (HasSubset.Subset (s' i') (s …
      h' : ∀ (i' : ι'), p' i' → Exists fun i => And (p i) (HasSubset.Subset (s i) (s …
      ⊢ ∀ (i' : ι), p i' → Exists fun i => And (p' i) (HasSubset.Subset (s' i) (s i'))
    -/
    simpa using h
    /-
      🎉 no goals
    -/


theorem HasBasis.inf' (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    (l ⊓ l').HasBasis (fun i : PProd ι ι' => p i.1 ∧ p' i.2) fun i => s i.1 ∩ s' i.2 :=
  ⟨by
    /-
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      ⊢ ∀ (t : Set α), Iff (Membership.mem (Min.min l l') t) (Exists fun i => And (A …
    -/
    intro t
    /-
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      t : Set α
      ⊢ Iff (Membership.mem (Min.min l l') t) (Exists fun i => And (And (p i.fst) (p …
    -/
    constructor
      /-
        case mp
        α : Type u_1
        ι : Sort u_4
        ι' : Sort u_5
        l l' : Filter α
        p : ι → Prop
        s : ι → Set α
        p' : ι' → Prop
        s' : ι' → Set α
        hl : l.HasBasis p s
        hl' : l'.HasBasis p' s'
        t : Set α
        ⊢ Membership.mem (Min.min l l') t → Exists fun i => And (And (p i.fst) (p' i.s …
      -/
    · simp only [mem_inf_iff, hl.mem_iff, hl'.mem_iff]
      /-
        case mp
        α : Type u_1
        ι : Sort u_4
        ι' : Sort u_5
        l l' : Filter α
        p : ι → Prop
        s : ι → Set α
        p' : ι' → Prop
        s' : ι' → Set α
        hl : l.HasBasis p s
        hl' : l'.HasBasis p' s'
        t : Set α
        ⊢ (Exists fun t₁ => And (Exists fun i => And (p i) (HasSubset.Subset (s i) t₁) …
      -/
      rintro ⟨t, ⟨i, hi, ht⟩, t', ⟨i', hi', ht'⟩, rfl⟩
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        ι : Sort u_4
        ι' : Sort u_5
        l l' : Filter α
        p : ι → Prop
        s : ι → Set α
        p' : ι' → Prop
        s' : ι' → Set α
        hl : l.HasBasis p s
        hl' : l'.HasBasis p' s'
        t : Set α
        i : ι
        hi : p i
        ht : HasSubset.Subset (s i) t
        t' : Set α
        i' : ι'
        hi' : p' i'
        ht' : HasSubset.Subset (s' i') t'
        ⊢ Exists fun i => And (And (p i.fst) (p' i.snd)) (HasSubset.Subset (Inter.inte …
      -/
      exact ⟨⟨i, i'⟩, ⟨hi, hi'⟩, inter_subset_inter ht ht'⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        ι : Sort u_4
        ι' : Sort u_5
        l l' : Filter α
        p : ι → Prop
        s : ι → Set α
        p' : ι' → Prop
        s' : ι' → Set α
        hl : l.HasBasis p s
        hl' : l'.HasBasis p' s'
        t : Set α
        ⊢ (Exists fun i => And (And (p i.fst) (p' i.snd)) (HasSubset.Subset (Inter.int …
      -/
    · rintro ⟨⟨i, i'⟩, ⟨hi, hi'⟩, H⟩
      /-
        case mpr.intro.mk.intro.intro
        α : Type u_1
        ι : Sort u_4
        ι' : Sort u_5
        l l' : Filter α
        p : ι → Prop
        s : ι → Set α
        p' : ι' → Prop
        s' : ι' → Set α
        hl : l.HasBasis p s
        hl' : l'.HasBasis p' s'
        t : Set α
        i : ι
        i' : ι'
        H : HasSubset.Subset (Inter.inter (s { fst := i, snd := i' }.fst) (s' { fst := …
        hi : p { fst := i, snd := i' }.fst
        hi' : p' { fst := i, snd := i' }.snd
        ⊢ Membership.mem (Min.min l l') t
      -/
      exact mem_inf_of_inter (hl.mem_of_mem hi) (hl'.mem_of_mem hi') H⟩
      /-
        🎉 no goals
      -/


theorem HasBasis.inf {ι ι' : Type*} {p : ι → Prop} {s : ι → Set α} {p' : ι' → Prop}
    {s' : ι' → Set α} (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    (l ⊓ l').HasBasis (fun i : ι × ι' => p i.1 ∧ p' i.2) fun i => s i.1 ∩ s' i.2 :=
  (hl.inf' hl').comp_equiv Equiv.pprodEquivProd.symm


theorem hasBasis_iInf' {ι : Type*} {ι' : ι → Type*} {l : ι → Filter α} {p : ∀ i, ι' i → Prop}
    {s : ∀ i, ι' i → Set α} (hl : ∀ i, (l i).HasBasis (p i) (s i)) :
    (⨅ i, l i).HasBasis (fun If : Set ι × ∀ i, ι' i => If.1.Finite ∧ ∀ i ∈ If.1, p i (If.2 i))
      fun If : Set ι × ∀ i, ι' i => ⋂ i ∈ If.1, s i (If.2 i) :=
  ⟨by
    /-
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      ⊢ ∀ (t : Set α), Iff (Membership.mem (iInf fun i => l i) t) (Exists fun i => A …
    -/
    intro t
    /-
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      ⊢ Iff (Membership.mem (iInf fun i => l i) t) (Exists fun i => And (And i.1.Fin …
    -/
    constructor
      /-
        case mp
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        t : Set α
        ⊢ Membership.mem (iInf fun i => l i) t → Exists fun i => And (And i.1.Finite ( …
      -/
    · simp only [mem_iInf', (hl _).mem_iff]
      /-
        case mp
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        t : Set α
        ⊢ (Exists fun I => And I.Finite (Exists fun V => And (∀ (i : ι), Exists fun i_ …
      -/
      rintro ⟨I, hI, V, hV, -, rfl, -⟩
      /-
        case mp.intro.intro.intro.intro.intro.intro
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        I : Set ι
        hI : I.Finite
        V : ι → Set α
        hV : ∀ (i : ι), Exists fun i_1 => And (p i i_1) (HasSubset.Subset (s i i_1) (V …
        ⊢ Exists fun i => And (And i.1.Finite (∀ (i_1 : ι), Membership.mem i.1 i_1 → p …
      -/
      choose u hu using hV
      /-
        case mp.intro.intro.intro.intro.intro.intro
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        I : Set ι
        hI : I.Finite
        V : ι → Set α
        u : (i : ι) → ι' i
        hu : ∀ (i : ι), And (p i (u i)) (HasSubset.Subset (s i (u i)) (V i))
        ⊢ Exists fun i => And (And i.1.Finite (∀ (i_1 : ι), Membership.mem i.1 i_1 → p …
      -/
      exact ⟨⟨I, u⟩, ⟨hI, fun i _ => (hu i).1⟩, iInter₂_mono fun i _ => (hu i).2⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        t : Set α
        ⊢ (Exists fun i => And (And i.1.Finite (∀ (i_1 : ι), Membership.mem i.1 i_1 →  …
      -/
    · rintro ⟨⟨I, f⟩, ⟨hI₁, hI₂⟩, hsub⟩
      /-
        case mpr.intro.mk.intro.intro
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        t : Set α
        I : Set ι
        f : (i : ι) → ι' i
        hsub : HasSubset.Subset (Set.iInter fun i => Set.iInter fun h => s i ({ fst := …
        hI₁ : { fst := I, snd := f }.1.Finite
        hI₂ : ∀ (i : ι), Membership.mem { fst := I, snd := f }.1 i → p i ({ fst := I,  …
        ⊢ Membership.mem (iInf fun i => l i) t
      -/
      refine mem_of_superset ?_ hsub
      /-
        case mpr.intro.mk.intro.intro
        α : Type u_1
        ι : Type u_6
        ι' : ι → Type u_7
        l : ι → Filter α
        p : (i : ι) → ι' i → Prop
        s : (i : ι) → ι' i → Set α
        hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
        t : Set α
        I : Set ι
        f : (i : ι) → ι' i
        hsub : HasSubset.Subset (Set.iInter fun i => Set.iInter fun h => s i ({ fst := …
        hI₁ : { fst := I, snd := f }.1.Finite
        hI₂ : ∀ (i : ι), Membership.mem { fst := I, snd := f }.1 i → p i ({ fst := I,  …
        ⊢ Membership.mem (iInf fun i => l i) (Set.iInter fun i => Set.iInter fun h =>  …
      -/
      exact (biInter_mem hI₁).mpr fun i hi => mem_iInf_of_mem i <| (hl i).mem_of_mem <| hI₂ _ hi⟩
      /-
        🎉 no goals
      -/


theorem hasBasis_iInf {ι : Type*} {ι' : ι → Type*} {l : ι → Filter α} {p : ∀ i, ι' i → Prop}
    {s : ∀ i, ι' i → Set α} (hl : ∀ i, (l i).HasBasis (p i) (s i)) :
    (⨅ i, l i).HasBasis
      (fun If : Σ I : Set ι, ∀ i : I, ι' i => If.1.Finite ∧ ∀ i : If.1, p i (If.2 i)) fun If =>
      ⋂ i : If.1, s i (If.2 i) := by
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    l : ι → Filter α
    p : (i : ι) → ι' i → Prop
    s : (i : ι) → ι' i → Set α
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    ⊢ (iInf fun i => l i).HasBasis (fun If => And If.fst.Finite (∀ (i : ↑If.fst),  …
  -/
  refine ⟨fun t => ⟨fun ht => ?_, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      ht : Membership.mem (iInf fun i => l i) t
      ⊢ Exists fun i => And (And i.fst.Finite (∀ (i_1 : ↑i.fst), p (↑i_1) (i.snd i_1 …
    -/
  · rcases (hasBasis_iInf' hl).mem_iff.mp ht with ⟨⟨I, f⟩, ⟨hI, hf⟩, hsub⟩
    /-
      case refine_1.intro.mk.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      ht : Membership.mem (iInf fun i => l i) t
      I : Set ι
      f : (i : ι) → ι' i
      hsub : HasSubset.Subset (Set.iInter fun i => Set.iInter fun h => s i ({ fst := …
      hI : { fst := I, snd := f }.1.Finite
      hf : ∀ (i : ι), Membership.mem { fst := I, snd := f }.1 i → p i ({ fst := I, s …
      ⊢ Exists fun i => And (And i.fst.Finite (∀ (i_1 : ↑i.fst), p (↑i_1) (i.snd i_1 …
    -/
    exact ⟨⟨I, fun i => f i⟩, ⟨hI, Subtype.forall.mpr hf⟩, trans (iInter_subtype _ _) hsub⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      ⊢ (Exists fun i => And (And i.fst.Finite (∀ (i_1 : ↑i.fst), p (↑i_1) (i.snd i_ …
    -/
  · rintro ⟨⟨I, f⟩, ⟨hI, hf⟩, hsub⟩
    /-
      case refine_2.intro.mk.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      I : Set ι
      f : (i : ↑I) → ι' ↑i
      hsub : HasSubset.Subset (Set.iInter fun i => s (↑i) (⟨I, f⟩.snd i)) t
      hI : ⟨I, f⟩.fst.Finite
      hf : ∀ (i : ↑⟨I, f⟩.fst), p (↑i) (⟨I, f⟩.snd i)
      ⊢ Membership.mem (iInf fun i => l i) t
    -/
    refine mem_of_superset ?_ hsub
    /-
      case refine_2.intro.mk.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      I : Set ι
      f : (i : ↑I) → ι' ↑i
      hsub : HasSubset.Subset (Set.iInter fun i => s (↑i) (⟨I, f⟩.snd i)) t
      hI : ⟨I, f⟩.fst.Finite
      hf : ∀ (i : ↑⟨I, f⟩.fst), p (↑i) (⟨I, f⟩.snd i)
      ⊢ Membership.mem (iInf fun i => l i) (Set.iInter fun i => s (↑i) (⟨I, f⟩.snd i))
    -/
    cases hI.nonempty_fintype
    /-
      case refine_2.intro.mk.intro.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      l : ι → Filter α
      p : (i : ι) → ι' i → Prop
      s : (i : ι) → ι' i → Set α
      hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
      t : Set α
      I : Set ι
      f : (i : ↑I) → ι' ↑i
      hsub : HasSubset.Subset (Set.iInter fun i => s (↑i) (⟨I, f⟩.snd i)) t
      hI : ⟨I, f⟩.fst.Finite
      hf : ∀ (i : ↑⟨I, f⟩.fst), p (↑i) (⟨I, f⟩.snd i)
      val✝ : Fintype ↑⟨I, f⟩.fst
      ⊢ Membership.mem (iInf fun i => l i) (Set.iInter fun i => s (↑i) (⟨I, f⟩.snd i))
    -/
    exact iInter_mem.2 fun i => mem_iInf_of_mem ↑i <| (hl i).mem_of_mem <| hf _
    /-
      🎉 no goals
    -/


theorem hasBasis_iInf_of_directed' {ι : Type*} {ι' : ι → Sort _} [Nonempty ι] {l : ι → Filter α}
    (s : ∀ i, ι' i → Set α) (p : ∀ i, ι' i → Prop) (hl : ∀ i, (l i).HasBasis (p i) (s i))
    (h : Directed (· ≥ ·) l) :
    (⨅ i, l i).HasBasis (fun ii' : Σi, ι' i => p ii'.1 ii'.2) fun ii' => s ii'.1 ii'.2 := by
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    ⊢ (iInf fun i => l i).HasBasis (fun ii' => p ii'.fst ii'.snd) fun ii' => s ii' …
  -/
  refine ⟨fun t => ?_⟩
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    t : Set α
    ⊢ Iff (Membership.mem (iInf fun i => l i) t) (Exists fun i => And (p i.fst i.s …
  -/
  rw [mem_iInf_of_directed h, Sigma.exists]
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    t : Set α
    ⊢ Iff (Exists fun i => Membership.mem (l i) t) (Exists fun a => Exists fun b = …
  -/
  exact exists_congr fun i => (hl i).mem_iff
  /-
    🎉 no goals
  -/


theorem hasBasis_iInf_of_directed {ι : Type*} {ι' : Sort _} [Nonempty ι] {l : ι → Filter α}
    (s : ι → ι' → Set α) (p : ι → ι' → Prop) (hl : ∀ i, (l i).HasBasis (p i) (s i))
    (h : Directed (· ≥ ·) l) :
    (⨅ i, l i).HasBasis (fun ii' : ι × ι' => p ii'.1 ii'.2) fun ii' => s ii'.1 ii'.2 := by
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    ⊢ (iInf fun i => l i).HasBasis (fun ii' => p ii'.1 ii'.2) fun ii' => s ii'.1 i …
  -/
  refine ⟨fun t => ?_⟩
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    t : Set α
    ⊢ Iff (Membership.mem (iInf fun i => l i) t) (Exists fun i => And (p i.1 i.2)  …
  -/
  rw [mem_iInf_of_directed h, Prod.exists]
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    inst✝ : Nonempty ι
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), (l i).HasBasis (p i) (s i)
    h : Directed (fun x1 x2 => GE.ge x1 x2) l
    t : Set α
    ⊢ Iff (Exists fun i => Membership.mem (l i) t) (Exists fun a => Exists fun b = …
  -/
  exact exists_congr fun i => (hl i).mem_iff
  /-
    🎉 no goals
  -/


theorem hasBasis_biInf_of_directed' {ι : Type*} {ι' : ι → Sort _} {dom : Set ι}
    (hdom : dom.Nonempty) {l : ι → Filter α} (s : ∀ i, ι' i → Set α) (p : ∀ i, ι' i → Prop)
    (hl : ∀ i ∈ dom, (l i).HasBasis (p i) (s i)) (h : DirectedOn (l ⁻¹'o GE.ge) dom) :
    (⨅ i ∈ dom, l i).HasBasis (fun ii' : Σi, ι' i => ii'.1 ∈ dom ∧ p ii'.1 ii'.2) fun ii' =>
      s ii'.1 ii'.2 := by
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    ⊢ (iInf fun i => iInf fun h => l i).HasBasis (fun ii' => And (Membership.mem d …
  -/
  refine ⟨fun t => ?_⟩
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    t : Set α
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => l i) t) (Exists fun i => An …
  -/
  rw [mem_biInf_of_directed h hdom, Sigma.exists]
  /-
    α : Type u_1
    ι : Type u_6
    ι' : ι → Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : (i : ι) → ι' i → Set α
    p : (i : ι) → ι' i → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    t : Set α
    ⊢ Iff (Exists fun i => And (Membership.mem dom i) (Membership.mem (l i) t)) (E …
  -/
  refine exists_congr fun i => ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : (i : ι) → ι' i → Set α
      p : (i : ι) → ι' i → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      ⊢ And (Membership.mem dom i) (Membership.mem (l i) t) → Exists fun b => And (A …
    -/
  · rintro ⟨hi, hti⟩
    /-
      case refine_1.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : (i : ι) → ι' i → Set α
      p : (i : ι) → ι' i → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      hi : Membership.mem dom i
      hti : Membership.mem (l i) t
      ⊢ Exists fun b => And (And (Membership.mem dom ⟨i, b⟩.fst) (p ⟨i, b⟩.fst ⟨i, b …
    -/
    rcases (hl i hi).mem_iff.mp hti with ⟨b, hb, hbt⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : (i : ι) → ι' i → Set α
      p : (i : ι) → ι' i → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      hi : Membership.mem dom i
      hti : Membership.mem (l i) t
      b : ι' i
      hb : p i b
      hbt : HasSubset.Subset (s i b) t
      ⊢ Exists fun b => And (And (Membership.mem dom ⟨i, b⟩.fst) (p ⟨i, b⟩.fst ⟨i, b …
    -/
    exact ⟨b, ⟨hi, hb⟩, hbt⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : (i : ι) → ι' i → Set α
      p : (i : ι) → ι' i → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      ⊢ (Exists fun b => And (And (Membership.mem dom ⟨i, b⟩.fst) (p ⟨i, b⟩.fst ⟨i,  …
    -/
  · rintro ⟨b, ⟨hi, hb⟩, hibt⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : ι → Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : (i : ι) → ι' i → Set α
      p : (i : ι) → ι' i → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      b : ι' i
      hibt : HasSubset.Subset (s ⟨i, b⟩.fst ⟨i, b⟩.snd) t
      hi : Membership.mem dom ⟨i, b⟩.fst
      hb : p ⟨i, b⟩.fst ⟨i, b⟩.snd
      ⊢ And (Membership.mem dom i) (Membership.mem (l i) t)
    -/
    exact ⟨hi, (hl i hi).mem_iff.mpr ⟨b, hb, hibt⟩⟩
    /-
      🎉 no goals
    -/


theorem hasBasis_biInf_of_directed {ι : Type*} {ι' : Sort _} {dom : Set ι} (hdom : dom.Nonempty)
    {l : ι → Filter α} (s : ι → ι' → Set α) (p : ι → ι' → Prop)
    (hl : ∀ i ∈ dom, (l i).HasBasis (p i) (s i)) (h : DirectedOn (l ⁻¹'o GE.ge) dom) :
    (⨅ i ∈ dom, l i).HasBasis (fun ii' : ι × ι' => ii'.1 ∈ dom ∧ p ii'.1 ii'.2) fun ii' =>
      s ii'.1 ii'.2 := by
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    ⊢ (iInf fun i => iInf fun h => l i).HasBasis (fun ii' => And (Membership.mem d …
  -/
  refine ⟨fun t => ?_⟩
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    t : Set α
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => l i) t) (Exists fun i => An …
  -/
  rw [mem_biInf_of_directed h hdom, Prod.exists]
  /-
    α : Type u_1
    ι : Type u_6
    ι' : Type u_7
    dom : Set ι
    hdom : dom.Nonempty
    l : ι → Filter α
    s : ι → ι' → Set α
    p : ι → ι' → Prop
    hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
    h : DirectedOn (Order.Preimage l GE.ge) dom
    t : Set α
    ⊢ Iff (Exists fun i => And (Membership.mem dom i) (Membership.mem (l i) t)) (E …
  -/
  refine exists_congr fun i => ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u_6
      ι' : Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : ι → ι' → Set α
      p : ι → ι' → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      ⊢ And (Membership.mem dom i) (Membership.mem (l i) t) → Exists fun b => And (A …
    -/
  · rintro ⟨hi, hti⟩
    /-
      case refine_1.intro
      α : Type u_1
      ι : Type u_6
      ι' : Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : ι → ι' → Set α
      p : ι → ι' → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      hi : Membership.mem dom i
      hti : Membership.mem (l i) t
      ⊢ Exists fun b => And (And (Membership.mem dom { fst := i, snd := b }.1) (p {  …
    -/
    rcases (hl i hi).mem_iff.mp hti with ⟨b, hb, hbt⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : ι → ι' → Set α
      p : ι → ι' → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      hi : Membership.mem dom i
      hti : Membership.mem (l i) t
      b : ι'
      hb : p i b
      hbt : HasSubset.Subset (s i b) t
      ⊢ Exists fun b => And (And (Membership.mem dom { fst := i, snd := b }.1) (p {  …
    -/
    exact ⟨b, ⟨hi, hb⟩, hbt⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u_6
      ι' : Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : ι → ι' → Set α
      p : ι → ι' → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      ⊢ (Exists fun b => And (And (Membership.mem dom { fst := i, snd := b }.1) (p { …
    -/
  · rintro ⟨b, ⟨hi, hb⟩, hibt⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      ι : Type u_6
      ι' : Type u_7
      dom : Set ι
      hdom : dom.Nonempty
      l : ι → Filter α
      s : ι → ι' → Set α
      p : ι → ι' → Prop
      hl : ∀ (i : ι), Membership.mem dom i → (l i).HasBasis (p i) (s i)
      h : DirectedOn (Order.Preimage l GE.ge) dom
      t : Set α
      i : ι
      b : ι'
      hibt : HasSubset.Subset (s { fst := i, snd := b }.1 { fst := i, snd := b }.2) t
      hi : Membership.mem dom { fst := i, snd := b }.1
      hb : p { fst := i, snd := b }.1 { fst := i, snd := b }.2
      ⊢ And (Membership.mem dom i) (Membership.mem (l i) t)
    -/
    exact ⟨hi, (hl i hi).mem_iff.mpr ⟨b, hb, hibt⟩⟩
    /-
      🎉 no goals
    -/


theorem hasBasis_principal (t : Set α) : (𝓟 t).HasBasis (fun _ : Unit => True) fun _ => t :=
               /-
                 α : Type u_1
                 t U : Set α
                 ⊢ Iff (Membership.mem (Filter.principal t) U) (Exists fun i => And True (HasSu …
               -/
  ⟨fun U => by simp⟩
               /-
                 🎉 no goals
               -/


theorem hasBasis_pure (x : α) :
    (pure x : Filter α).HasBasis (fun _ : Unit => True) fun _ => {x} := by
  /-
    α : Type u_1
    x : α
    ⊢ (Pure.pure x).HasBasis (fun x => True) fun x_1 => Singleton.singleton x
  -/
  simp only [← principal_singleton, hasBasis_principal]
  /-
    🎉 no goals
  -/


theorem HasBasis.sup' (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    (l ⊔ l').HasBasis (fun i : PProd ι ι' => p i.1 ∧ p' i.2) fun i => s i.1 ∪ s' i.2 :=
  ⟨by
    /-
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      ⊢ ∀ (t : Set α), Iff (Membership.mem (Max.max l l') t) (Exists fun i => And (A …
    -/
    intro t
    simp_rw [mem_sup, hl.mem_iff, hl'.mem_iff, PProd.exists, union_subset_iff,
       ← exists_and_right, ← exists_and_left]
    /-
      α : Type u_1
      ι : Sort u_4
      ι' : Sort u_5
      l l' : Filter α
      p : ι → Prop
      s : ι → Set α
      p' : ι' → Prop
      s' : ι' → Set α
      hl : l.HasBasis p s
      hl' : l'.HasBasis p' s'
      t : Set α
      ⊢ Iff (Exists fun x => Exists fun x_1 => And (And (p x) (HasSubset.Subset (s x …
    -/
    simp only [and_assoc, and_left_comm]⟩
    /-
      🎉 no goals
    -/


theorem HasBasis.sup {ι ι' : Type*} {p : ι → Prop} {s : ι → Set α} {p' : ι' → Prop}
    {s' : ι' → Set α} (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    (l ⊔ l').HasBasis (fun i : ι × ι' => p i.1 ∧ p' i.2) fun i => s i.1 ∪ s' i.2 :=
  (hl.sup' hl').comp_equiv Equiv.pprodEquivProd.symm


theorem hasBasis_iSup {ι : Sort*} {ι' : ι → Type*} {l : ι → Filter α} {p : ∀ i, ι' i → Prop}
    {s : ∀ i, ι' i → Set α} (hl : ∀ i, (l i).HasBasis (p i) (s i)) :
    (⨆ i, l i).HasBasis (fun f : ∀ i, ι' i => ∀ i, p i (f i)) fun f : ∀ i, ι' i => ⋃ i, s i (f i) :=
  hasBasis_iff.mpr fun t => by
    simp only [hasBasis_iff, (hl _).mem_iff, Classical.skolem, forall_and, iUnion_subset_iff,
      mem_iSup]


theorem HasBasis.sup_principal (hl : l.HasBasis p s) (t : Set α) :
    (l ⊔ 𝓟 t).HasBasis p fun i => s i ∪ t :=
  ⟨fun u => by
    simp only [(hl.sup' (hasBasis_principal t)).mem_iff, PProd.exists, exists_prop, and_true,
      Unique.exists_iff]⟩


theorem HasBasis.sup_pure (hl : l.HasBasis p s) (x : α) :
    (l ⊔ pure x).HasBasis p fun i => s i ∪ {x} := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    x : α
    ⊢ (Max.max l (Pure.pure x)).HasBasis p fun i => Union.union (s i) (Singleton.s …
  -/
  simp only [← principal_singleton, hl.sup_principal]
  /-
    🎉 no goals
  -/


theorem HasBasis.inf_principal (hl : l.HasBasis p s) (s' : Set α) :
    (l ⊓ 𝓟 s').HasBasis p fun i => s i ∩ s' :=
  ⟨fun t => by
    /-
      α : Type u_1
      ι : Sort u_4
      l : Filter α
      p : ι → Prop
      s : ι → Set α
      hl : l.HasBasis p s
      s' t : Set α
      ⊢ Iff (Membership.mem (Min.min l (Filter.principal s')) t) (Exists fun i => An …
    -/
    simp only [mem_inf_principal, hl.mem_iff, subset_def, mem_setOf_eq, mem_inter_iff, and_imp]⟩
    /-
      🎉 no goals
    -/


theorem HasBasis.principal_inf (hl : l.HasBasis p s) (s' : Set α) :
    (𝓟 s' ⊓ l).HasBasis p fun i => s' ∩ s i := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    hl : l.HasBasis p s
    s' : Set α
    ⊢ (Min.min (Filter.principal s') l).HasBasis p fun i => Inter.inter s' (s i)
  -/
  simpa only [inf_comm, inter_comm] using hl.inf_principal s'
  /-
    🎉 no goals
  -/


theorem HasBasis.inf_basis_neBot_iff (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    NeBot (l ⊓ l') ↔ ∀ ⦃i⦄, p i → ∀ ⦃i'⦄, p' i' → (s i ∩ s' i').Nonempty :=
                                      /-
                                        α : Type u_1
                                        ι : Sort u_4
                                        ι' : Sort u_5
                                        l l' : Filter α
                                        p : ι → Prop
                                        s : ι → Set α
                                        p' : ι' → Prop
                                        s' : ι' → Set α
                                        hl : l.HasBasis p s
                                        hl' : l'.HasBasis p' s'
                                        ⊢ Iff (∀ {i : PProd ι ι'}, And (p i.fst) (p' i.snd) → (Inter.inter (s i.fst) ( …
                                      -/
  (hl.inf' hl').neBot_iff.trans <| by simp [@forall_swap _ ι']
                                      /-
                                        🎉 no goals
                                      -/


theorem HasBasis.inf_neBot_iff (hl : l.HasBasis p s) :
    NeBot (l ⊓ l') ↔ ∀ ⦃i⦄, p i → ∀ ⦃s'⦄, s' ∈ l' → (s i ∩ s').Nonempty :=
  hl.inf_basis_neBot_iff l'.basis_sets


theorem HasBasis.inf_principal_neBot_iff (hl : l.HasBasis p s) {t : Set α} :
    NeBot (l ⊓ 𝓟 t) ↔ ∀ ⦃i⦄, p i → (s i ∩ t).Nonempty :=
  (hl.inf_principal t).neBot_iff

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.disjoint_iff (hl : l.HasBasis p s) (hl' : l'.HasBasis p' s') :
    Disjoint l l' ↔ ∃ i, p i ∧ ∃ i', p' i' ∧ Disjoint (s i) (s' i') :=
  not_iff_not.mp <| by simp only [_root_.disjoint_iff, ← Ne.eq_def, ← neBot_iff, inf_eq_inter,
    hl.inf_basis_neBot_iff hl', not_exists, not_and, bot_eq_empty, ← nonempty_iff_ne_empty]

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem _root_.Disjoint.exists_mem_filter_basis (h : Disjoint l l') (hl : l.HasBasis p s)
    (hl' : l'.HasBasis p' s') : ∃ i, p i ∧ ∃ i', p' i' ∧ Disjoint (s i) (s' i') :=
  (hl.disjoint_iff hl').1 h


theorem _root_.Pairwise.exists_mem_filter_basis_of_disjoint {I} [Finite I] {l : I → Filter α}
    {ι : I → Sort*} {p : ∀ i, ι i → Prop} {s : ∀ i, ι i → Set α} (hd : Pairwise (Disjoint on l))
    (h : ∀ i, (l i).HasBasis (p i) (s i)) :
    ∃ ind : ∀ i, ι i, (∀ i, p i (ind i)) ∧ Pairwise (Disjoint on fun i => s i (ind i)) := by
  /-
    α : Type u_1
    I : Type u_7
    inst✝ : Finite I
    l : I → Filter α
    ι : I → Sort u_6
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    hd : Pairwise (Function.onFun Disjoint l)
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (Pairwise (Function.onFun Dis …
  -/
  rcases hd.exists_mem_filter_of_disjoint with ⟨t, htl, hd⟩
  /-
    case intro.intro
    α : Type u_1
    I : Type u_7
    inst✝ : Finite I
    l : I → Filter α
    ι : I → Sort u_6
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    hd✝ : Pairwise (Function.onFun Disjoint l)
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    t : I → Set α
    htl : ∀ (i : I), Membership.mem (l i) (t i)
    hd : Pairwise (Function.onFun Disjoint t)
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (Pairwise (Function.onFun Dis …
  -/
  choose ind hp ht using fun i => (h i).mem_iff.1 (htl i)
  /-
    case intro.intro
    α : Type u_1
    I : Type u_7
    inst✝ : Finite I
    l : I → Filter α
    ι : I → Sort u_6
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    hd✝ : Pairwise (Function.onFun Disjoint l)
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    t : I → Set α
    htl : ∀ (i : I), Membership.mem (l i) (t i)
    hd : Pairwise (Function.onFun Disjoint t)
    ind : (i : I) → ι i
    hp : ∀ (i : I), p i (ind i)
    ht : ∀ (i : I), HasSubset.Subset (s i (ind i)) (t i)
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (Pairwise (Function.onFun Dis …
  -/
  exact ⟨ind, hp, hd.mono fun i j hij => hij.mono (ht _) (ht _)⟩
  /-
    🎉 no goals
  -/


theorem _root_.Set.PairwiseDisjoint.exists_mem_filter_basis {I : Type*} {l : I → Filter α}
    {ι : I → Sort*} {p : ∀ i, ι i → Prop} {s : ∀ i, ι i → Set α} {S : Set I}
    (hd : S.PairwiseDisjoint l) (hS : S.Finite) (h : ∀ i, (l i).HasBasis (p i) (s i)) :
    ∃ ind : ∀ i, ι i, (∀ i, p i (ind i)) ∧ S.PairwiseDisjoint fun i => s i (ind i) := by
  /-
    α : Type u_1
    I : Type u_6
    l : I → Filter α
    ι : I → Sort u_7
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    S : Set I
    hd : S.PairwiseDisjoint l
    hS : S.Finite
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (S.PairwiseDisjoint fun i =>  …
  -/
  rcases hd.exists_mem_filter hS with ⟨t, htl, hd⟩
  /-
    case intro.intro
    α : Type u_1
    I : Type u_6
    l : I → Filter α
    ι : I → Sort u_7
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    S : Set I
    hd✝ : S.PairwiseDisjoint l
    hS : S.Finite
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    t : I → Set α
    htl : ∀ (i : I), Membership.mem (l i) (t i)
    hd : S.PairwiseDisjoint t
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (S.PairwiseDisjoint fun i =>  …
  -/
  choose ind hp ht using fun i => (h i).mem_iff.1 (htl i)
  /-
    case intro.intro
    α : Type u_1
    I : Type u_6
    l : I → Filter α
    ι : I → Sort u_7
    p : (i : I) → ι i → Prop
    s : (i : I) → ι i → Set α
    S : Set I
    hd✝ : S.PairwiseDisjoint l
    hS : S.Finite
    h : ∀ (i : I), (l i).HasBasis (p i) (s i)
    t : I → Set α
    htl : ∀ (i : I), Membership.mem (l i) (t i)
    hd : S.PairwiseDisjoint t
    ind : (i : I) → ι i
    hp : ∀ (i : I), p i (ind i)
    ht : ∀ (i : I), HasSubset.Subset (s i (ind i)) (t i)
    ⊢ Exists fun ind => And (∀ (i : I), p i (ind i)) (S.PairwiseDisjoint fun i =>  …
  -/
  exact ⟨ind, hp, hd.mono ht⟩
  /-
    🎉 no goals
  -/


theorem inf_neBot_iff :
    NeBot (l ⊓ l') ↔ ∀ ⦃s : Set α⦄, s ∈ l → ∀ ⦃s'⦄, s' ∈ l' → (s ∩ s').Nonempty :=
  l.basis_sets.inf_neBot_iff


theorem inf_principal_neBot_iff {s : Set α} : NeBot (l ⊓ 𝓟 s) ↔ ∀ U ∈ l, (U ∩ s).Nonempty :=
  l.basis_sets.inf_principal_neBot_iff


theorem mem_iff_inf_principal_compl {f : Filter α} {s : Set α} : s ∈ f ↔ f ⊓ 𝓟 sᶜ = ⊥ := by
  /-
    α : Type u_1
    f : Filter α
    s : Set α
    ⊢ Iff (Membership.mem f s) (Eq (Min.min f (Filter.principal (HasCompl.compl s) …
  -/
  refine not_iff_not.1 ((inf_principal_neBot_iff.trans ?_).symm.trans neBot_iff)
  exact
    ⟨fun h hs => by simpa [Set.not_nonempty_empty] using h s hs, fun hs t ht =>
      inter_compl_nonempty_iff.2 fun hts => hs <| mem_of_superset ht hts⟩


theorem not_mem_iff_inf_principal_compl {f : Filter α} {s : Set α} : s ∉ f ↔ NeBot (f ⊓ 𝓟 sᶜ) :=
  (not_congr mem_iff_inf_principal_compl).trans neBot_iff.symm


@[simp]
theorem disjoint_principal_right {f : Filter α} {s : Set α} : Disjoint f (𝓟 s) ↔ sᶜ ∈ f := by
  /-
    α : Type u_1
    f : Filter α
    s : Set α
    ⊢ Iff (Disjoint f (Filter.principal s)) (Membership.mem f (HasCompl.compl s))
  -/
  rw [mem_iff_inf_principal_compl, compl_compl, disjoint_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_principal_left {f : Filter α} {s : Set α} : Disjoint (𝓟 s) f ↔ sᶜ ∈ f := by
  /-
    α : Type u_1
    f : Filter α
    s : Set α
    ⊢ Iff (Disjoint (Filter.principal s) f) (Membership.mem f (HasCompl.compl s))
  -/
  rw [disjoint_comm, disjoint_principal_right]
  /-
    🎉 no goals
  -/


@[simp 1100] -- Porting note: higher priority for linter
theorem disjoint_principal_principal {s t : Set α} : Disjoint (𝓟 s) (𝓟 t) ↔ Disjoint s t := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (Disjoint (Filter.principal s) (Filter.principal t)) (Disjoint s t)
  -/
  rw [← subset_compl_iff_disjoint_left, disjoint_principal_left, mem_principal]
  /-
    🎉 no goals
  -/


alias ⟨_, _root_.Disjoint.filter_principal⟩ := disjoint_principal_principal


@[simp]
theorem disjoint_pure_pure {x y : α} : Disjoint (pure x : Filter α) (pure y) ↔ x ≠ y := by
  /-
    α : Type u_1
    x y : α
    ⊢ Iff (Disjoint (Pure.pure x) (Pure.pure y)) (Ne x y)
  -/
  simp only [← principal_singleton, disjoint_principal_principal, disjoint_singleton]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.disjoint_iff_left (h : l.HasBasis p s) :
    Disjoint l l' ↔ ∃ i, p i ∧ (s i)ᶜ ∈ l' := by
  simp only [h.disjoint_iff l'.basis_sets, id, ← disjoint_principal_left,
    (hasBasis_principal _).disjoint_iff l'.basis_sets, true_and, Unique.exists_iff]

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.disjoint_iff_right (h : l.HasBasis p s) :
    Disjoint l' l ↔ ∃ i, p i ∧ (s i)ᶜ ∈ l' :=
  disjoint_comm.trans h.disjoint_iff_left


theorem le_iff_forall_inf_principal_compl {f g : Filter α} : f ≤ g ↔ ∀ V ∈ g, f ⊓ 𝓟 Vᶜ = ⊥ :=
  forall₂_congr fun _ _ => mem_iff_inf_principal_compl


theorem inf_neBot_iff_frequently_left {f g : Filter α} :
    NeBot (f ⊓ g) ↔ ∀ {p : α → Prop}, (∀ᶠ x in f, p x) → ∃ᶠ x in g, p x := by
  /-
    α : Type u_1
    f g : Filter α
    ⊢ Iff (Min.min f g).NeBot (∀ {p : α → Prop}, Filter.Eventually (fun x => p x)  …
  -/
  simp only [inf_neBot_iff, frequently_iff, and_comm]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem inf_neBot_iff_frequently_right {f g : Filter α} :
    NeBot (f ⊓ g) ↔ ∀ {p : α → Prop}, (∀ᶠ x in g, p x) → ∃ᶠ x in f, p x := by
  /-
    α : Type u_1
    f g : Filter α
    ⊢ Iff (Min.min f g).NeBot (∀ {p : α → Prop}, Filter.Eventually (fun x => p x)  …
  -/
  rw [inf_comm]
  /-
    α : Type u_1
    f g : Filter α
    ⊢ Iff (Min.min g f).NeBot (∀ {p : α → Prop}, Filter.Eventually (fun x => p x)  …
  -/
  exact inf_neBot_iff_frequently_left
  /-
    🎉 no goals
  -/


theorem HasBasis.eq_biInf (h : l.HasBasis p s) : l = ⨅ (i) (_ : p i), 𝓟 (s i) :=
                                               /-
                                                 α : Type u_1
                                                 ι : Sort u_4
                                                 l : Filter α
                                                 p : ι → Prop
                                                 s : ι → Set α
                                                 h : l.HasBasis p s
                                                 x✝ : Set α
                                                 ⊢ Iff (Membership.mem l x✝) (Exists fun i => And (p i) (Membership.mem (Filter …
                                               -/
  eq_biInf_of_mem_iff_exists_mem fun {_} => by simp only [h.mem_iff, mem_principal, exists_prop]
                                               /-
                                                 🎉 no goals
                                               -/


theorem HasBasis.eq_iInf (h : l.HasBasis (fun _ => True) s) : l = ⨅ i, 𝓟 (s i) := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    s : ι → Set α
    h : l.HasBasis (fun x => True) s
    ⊢ Eq l (iInf fun i => Filter.principal (s i))
  -/
  simpa only [iInf_true] using h.eq_biInf
  /-
    🎉 no goals
  -/


theorem hasBasis_iInf_principal {s : ι → Set α} (h : Directed (· ≥ ·) s) [Nonempty ι] :
    (⨅ i, 𝓟 (s i)).HasBasis (fun _ => True) s :=
  ⟨fun t => by
    /-
      α : Type u_1
      ι : Sort u_4
      s : ι → Set α
      h : Directed (fun x1 x2 => GE.ge x1 x2) s
      inst✝ : Nonempty ι
      t : Set α
      ⊢ Iff (Membership.mem (iInf fun i => Filter.principal (s i)) t) (Exists fun i  …
    -/
    simpa only [true_and] using mem_iInf_of_directed (h.mono_comp _ monotone_principal.dual) t⟩
    /-
      🎉 no goals
    -/


/-- If `s : ι → Set α` is an indexed family of sets, then finite intersections of `s i` form a basis
of `⨅ i, 𝓟 (s i)`. -/
theorem hasBasis_iInf_principal_finite {ι : Type*} (s : ι → Set α) :
    (⨅ i, 𝓟 (s i)).HasBasis (fun t : Set ι => t.Finite) fun t => ⋂ i ∈ t, s i := by
  /-
    α : Type u_1
    ι : Type u_6
    s : ι → Set α
    ⊢ (iInf fun i => Filter.principal (s i)).HasBasis (fun t => t.Finite) fun t => …
  -/
  refine ⟨fun U => (mem_iInf_finite _).trans ?_⟩
  simp only [iInf_principal_finset, mem_iUnion, mem_principal, exists_prop,
    exists_finite_iff_finset, Finset.set_biInter_coe]


theorem hasBasis_biInf_principal {s : β → Set α} {S : Set β} (h : DirectedOn (s ⁻¹'o (· ≥ ·)) S)
    (ne : S.Nonempty) : (⨅ i ∈ S, 𝓟 (s i)).HasBasis (fun i => i ∈ S) s :=
  ⟨fun t => by
    /-
      α : Type u_1
      β : Type u_2
      s : β → Set α
      S : Set β
      h : DirectedOn (Order.Preimage s fun x1 x2 => GE.ge x1 x2) S
      ne : S.Nonempty
      t : Set α
      ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => Filter.principal (s i)) t)  …
    -/
    refine mem_biInf_of_directed ?_ ne
    /-
      α : Type u_1
      β : Type u_2
      s : β → Set α
      S : Set β
      h : DirectedOn (Order.Preimage s fun x1 x2 => GE.ge x1 x2) S
      ne : S.Nonempty
      t : Set α
      ⊢ DirectedOn (Order.Preimage (fun i => Filter.principal (s i)) fun x1 x2 => GE …
    -/
    rw [directedOn_iff_directed, ← directed_comp] at h ⊢
    /-
      α : Type u_1
      β : Type u_2
      s : β → Set α
      S : Set β
      h : Directed (fun x1 x2 => GE.ge x1 x2) (Function.comp s Subtype.val)
      ne : S.Nonempty
      t : Set α
      ⊢ Directed (fun x1 x2 => GE.ge x1 x2) (Function.comp (fun i => Filter.principa …
    -/
    refine h.mono_comp _ ?_
    /-
      α : Type u_1
      β : Type u_2
      s : β → Set α
      S : Set β
      h : Directed (fun x1 x2 => GE.ge x1 x2) (Function.comp s Subtype.val)
      ne : S.Nonempty
      t : Set α
      ⊢ ∀ ⦃x y : Set α⦄, GE.ge x y → GE.ge (Filter.principal x) (Filter.principal y)
    -/
    exact fun _ _ => principal_mono.2⟩
    /-
      🎉 no goals
    -/


theorem hasBasis_biInf_principal' {ι : Type*} {p : ι → Prop} {s : ι → Set α}
    (h : ∀ i, p i → ∀ j, p j → ∃ k, p k ∧ s k ⊆ s i ∧ s k ⊆ s j) (ne : ∃ i, p i) :
    (⨅ (i) (_ : p i), 𝓟 (s i)).HasBasis p s :=
  Filter.hasBasis_biInf_principal h ne


theorem HasBasis.map (f : α → β) (hl : l.HasBasis p s) : (l.map f).HasBasis p fun i => f '' s i :=
               /-
                 α : Type u_1
                 β : Type u_2
                 ι : Sort u_4
                 l : Filter α
                 p : ι → Prop
                 s : ι → Set α
                 f : α → β
                 hl : l.HasBasis p s
                 t : Set β
                 ⊢ Iff (Membership.mem (Filter.map f l) t) (Exists fun i => And (p i) (HasSubse …
               -/
  ⟨fun t => by simp only [mem_map, image_subset_iff, hl.mem_iff, preimage]⟩
               /-
                 🎉 no goals
               -/


theorem HasBasis.comap (f : β → α) (hl : l.HasBasis p s) :
    (l.comap f).HasBasis p fun i => f ⁻¹' s i :=
  ⟨fun t => by
    /-
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      l : Filter α
      p : ι → Prop
      s : ι → Set α
      f : β → α
      hl : l.HasBasis p s
      t : Set β
      ⊢ Iff (Membership.mem (Filter.comap f l) t) (Exists fun i => And (p i) (HasSub …
    -/
    simp only [mem_comap', hl.mem_iff]
    /-
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      l : Filter α
      p : ι → Prop
      s : ι → Set α
      f : β → α
      hl : l.HasBasis p s
      t : Set β
      ⊢ Iff (Exists fun i => And (p i) (HasSubset.Subset (s i) (setOf fun y => ∀ ⦃x  …
    -/
    refine exists_congr (fun i => Iff.rfl.and ?_)
    /-
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      l : Filter α
      p : ι → Prop
      s : ι → Set α
      f : β → α
      hl : l.HasBasis p s
      t : Set β
      i : ι
      ⊢ Iff (HasSubset.Subset (s i) (setOf fun y => ∀ ⦃x : β⦄, Eq (f x) y → Membersh …
    -/
    exact ⟨fun h x hx => h hx rfl, fun h y hy x hx => h <| by rwa [mem_preimage, hx]⟩⟩
    /-
      🎉 no goals
    -/


theorem comap_hasBasis (f : α → β) (l : Filter β) :
    HasBasis (comap f l) (fun s : Set β => s ∈ l) fun s => f ⁻¹' s :=
  ⟨fun _ => mem_comap⟩


theorem HasBasis.forall_mem_mem (h : HasBasis l p s) {x : α} :
    (∀ t ∈ l, x ∈ t) ↔ ∀ i, p i → x ∈ s i := by
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    x : α
    ⊢ Iff (∀ (t : Set α), Membership.mem l t → Membership.mem t x) (∀ (i : ι), p i …
  -/
  simp only [h.mem_iff, exists_imp, and_imp]
  /-
    α : Type u_1
    ι : Sort u_4
    l : Filter α
    p : ι → Prop
    s : ι → Set α
    h : l.HasBasis p s
    x : α
    ⊢ Iff (∀ (t : Set α) (x_1 : ι), p x_1 → HasSubset.Subset (s x_1) t → Membershi …
  -/
  exact ⟨fun h i hi => h (s i) i hi Subset.rfl, fun h t i hi ht => ht (h i hi)⟩
  /-
    🎉 no goals
  -/


protected theorem HasBasis.biInf_mem [CompleteLattice β] {f : Set α → β} (h : HasBasis l p s)
    (hf : Monotone f) : ⨅ t ∈ l, f t = ⨅ (i) (_ : p i), f (s i) :=
  le_antisymm (le_iInf₂ fun i hi => iInf₂_le (s i) (h.mem_of_mem hi)) <|
    le_iInf₂ fun _t ht =>
      let ⟨i, hpi, hi⟩ := h.mem_iff.1 ht
      iInf₂_le_of_le i hpi (hf hi)


protected theorem HasBasis.biInter_mem {f : Set α → Set β} (h : HasBasis l p s) (hf : Monotone f) :
    ⋂ t ∈ l, f t = ⋂ (i) (_ : p i), f (s i) :=
  h.biInf_mem hf


protected theorem HasBasis.ker (h : HasBasis l p s) : l.ker = ⋂ (i) (_ : p i), s i :=
  sInter_eq_biInter.trans <| h.biInter_mem monotone_id


/-- `IsAntitoneBasis s` means the image of `s` is a filter basis such that `s` is decreasing. -/
structure IsAntitoneBasis extends IsBasis (fun _ => True) s'' : Prop where
  /-- The sequence of sets is antitone. -/
  protected antitone : Antitone s''


/-- We say that a filter `l` has an antitone basis `s : ι → Set α`, if `t ∈ l` if and only if `t`
includes `s i` for some `i`, and `s` is decreasing. -/
structure HasAntitoneBasis (l : Filter α) (s : ι'' → Set α)
    extends HasBasis l (fun _ => True) s : Prop where
  /-- The sequence of sets is antitone. -/
  protected antitone : Antitone s


protected theorem HasAntitoneBasis.map {l : Filter α} {s : ι'' → Set α}
    (hf : HasAntitoneBasis l s) (m : α → β) : HasAntitoneBasis (map m l) (m '' s ·) :=
  ⟨HasBasis.map _ hf.toHasBasis, fun _ _ h => image_subset _ <| hf.2 h⟩


protected theorem HasAntitoneBasis.comap {l : Filter α} {s : ι'' → Set α}
    (hf : HasAntitoneBasis l s) (m : β → α) : HasAntitoneBasis (comap m l) (m ⁻¹' s ·) :=
  ⟨hf.1.comap _, fun _ _ h ↦ preimage_mono (hf.2 h)⟩


lemma HasAntitoneBasis.iInf_principal {ι : Type*} [Preorder ι] [Nonempty ι] [IsDirected ι (· ≤ ·)]
    {s : ι → Set α} (hs : Antitone s) : (⨅ i, 𝓟 (s i)).HasAntitoneBasis s :=
  ⟨hasBasis_iInf_principal hs.directed_ge, hs⟩


theorem HasBasis.tendsto_left_iff (hla : la.HasBasis pa sa) :
    Tendsto f la lb ↔ ∀ t ∈ lb, ∃ i, pa i ∧ MapsTo f (sa i) t := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    la : Filter α
    pa : ι → Prop
    sa : ι → Set α
    lb : Filter β
    f : α → β
    hla : la.HasBasis pa sa
    ⊢ Iff (Filter.Tendsto f la lb) (∀ (t : Set β), Membership.mem lb t → Exists fu …
  -/
  simp only [Tendsto, (hla.map f).le_iff, image_subset_iff]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    la : Filter α
    pa : ι → Prop
    sa : ι → Set α
    lb : Filter β
    f : α → β
    hla : la.HasBasis pa sa
    ⊢ Iff (∀ (t : Set β), Membership.mem lb t → Exists fun i => And (pa i) (HasSub …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem HasBasis.tendsto_right_iff (hlb : lb.HasBasis pb sb) :
    Tendsto f la lb ↔ ∀ i, pb i → ∀ᶠ x in la, f x ∈ sb i := by
  /-
    α : Type u_1
    β : Type u_2
    ι' : Sort u_5
    la : Filter α
    lb : Filter β
    pb : ι' → Prop
    sb : ι' → Set β
    f : α → β
    hlb : lb.HasBasis pb sb
    ⊢ Iff (Filter.Tendsto f la lb) (∀ (i : ι'), pb i → Filter.Eventually (fun x => …
  -/
  simp only [Tendsto, hlb.ge_iff, mem_map', Filter.Eventually]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem HasBasis.tendsto_iff (hla : la.HasBasis pa sa) (hlb : lb.HasBasis pb sb) :
    Tendsto f la lb ↔ ∀ ib, pb ib → ∃ ia, pa ia ∧ ∀ x ∈ sa ia, f x ∈ sb ib := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    ι' : Sort u_5
    la : Filter α
    pa : ι → Prop
    sa : ι → Set α
    lb : Filter β
    pb : ι' → Prop
    sb : ι' → Set β
    f : α → β
    hla : la.HasBasis pa sa
    hlb : lb.HasBasis pb sb
    ⊢ Iff (Filter.Tendsto f la lb) (∀ (ib : ι'), pb ib → Exists fun ia => And (pa  …
  -/
  simp [hlb.tendsto_right_iff, hla.eventually_iff]
  /-
    🎉 no goals
  -/

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem Tendsto.basis_left (H : Tendsto f la lb) (hla : la.HasBasis pa sa) :
    ∀ t ∈ lb, ∃ i, pa i ∧ MapsTo f (sa i) t :=
  hla.tendsto_left_iff.1 H


theorem Tendsto.basis_right (H : Tendsto f la lb) (hlb : lb.HasBasis pb sb) :
    ∀ i, pb i → ∀ᶠ x in la, f x ∈ sb i :=
  hlb.tendsto_right_iff.1 H

-- Porting note: use `∃ i, p i ∧ _` instead of `∃ i (hi : p i), _`.

theorem Tendsto.basis_both (H : Tendsto f la lb) (hla : la.HasBasis pa sa)
    (hlb : lb.HasBasis pb sb) :
    ∀ ib, pb ib → ∃ ia, pa ia ∧ MapsTo f (sa ia) (sb ib) :=
  (hla.tendsto_iff hlb).1 H


theorem HasBasis.prod_pprod (hla : la.HasBasis pa sa) (hlb : lb.HasBasis pb sb) :
    (la ×ˢ lb).HasBasis (fun i : PProd ι ι' => pa i.1 ∧ pb i.2) fun i => sa i.1 ×ˢ sb i.2 :=
  (hla.comap Prod.fst).inf' (hlb.comap Prod.snd)


theorem HasBasis.prod {ι ι' : Type*} {pa : ι → Prop} {sa : ι → Set α} {pb : ι' → Prop}
    {sb : ι' → Set β} (hla : la.HasBasis pa sa) (hlb : lb.HasBasis pb sb) :
    (la ×ˢ lb).HasBasis (fun i : ι × ι' => pa i.1 ∧ pb i.2) fun i => sa i.1 ×ˢ sb i.2 :=
  (hla.comap Prod.fst).inf (hlb.comap Prod.snd)


theorem HasBasis.prod_same_index {p : ι → Prop} {sb : ι → Set β} (hla : la.HasBasis p sa)
    (hlb : lb.HasBasis p sb) (h_dir : ∀ {i j}, p i → p j → ∃ k, p k ∧ sa k ⊆ sa i ∧ sb k ⊆ sb j) :
    (la ×ˢ lb).HasBasis p fun i => sa i ×ˢ sb i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    la : Filter α
    sa : ι → Set α
    lb : Filter β
    p : ι → Prop
    sb : ι → Set β
    hla : la.HasBasis p sa
    hlb : lb.HasBasis p sb
    h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
    ⊢ (SProd.sprod la lb).HasBasis p fun i => SProd.sprod (sa i) (sb i)
  -/
  simp only [hasBasis_iff, (hla.prod_pprod hlb).mem_iff]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    la : Filter α
    sa : ι → Set α
    lb : Filter β
    p : ι → Prop
    sb : ι → Set β
    hla : la.HasBasis p sa
    hlb : lb.HasBasis p sb
    h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
    ⊢ ∀ (t : Set (Prod α β)), Iff (Exists fun i => And (And (p i.fst) (p i.snd)) ( …
  -/
  refine fun t => ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      la : Filter α
      sa : ι → Set α
      lb : Filter β
      p : ι → Prop
      sb : ι → Set β
      hla : la.HasBasis p sa
      hlb : lb.HasBasis p sb
      h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
      t : Set (Prod α β)
      ⊢ (Exists fun i => And (And (p i.fst) (p i.snd)) (HasSubset.Subset (SProd.spro …
    -/
  · rintro ⟨⟨i, j⟩, ⟨hi, hj⟩, hsub : sa i ×ˢ sb j ⊆ t⟩
    /-
      case refine_1.intro.mk.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      la : Filter α
      sa : ι → Set α
      lb : Filter β
      p : ι → Prop
      sb : ι → Set β
      hla : la.HasBasis p sa
      hlb : lb.HasBasis p sb
      h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
      t : Set (Prod α β)
      i j : ι
      hsub : HasSubset.Subset (SProd.sprod (sa i) (sb j)) t
      hi : p { fst := i, snd := j }.fst
      hj : p { fst := i, snd := j }.snd
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (SProd.sprod (sa i) (sb i)) t)
    -/
    rcases h_dir hi hj with ⟨k, hk, ki, kj⟩
    /-
      case refine_1.intro.mk.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      la : Filter α
      sa : ι → Set α
      lb : Filter β
      p : ι → Prop
      sb : ι → Set β
      hla : la.HasBasis p sa
      hlb : lb.HasBasis p sb
      h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
      t : Set (Prod α β)
      i j : ι
      hsub : HasSubset.Subset (SProd.sprod (sa i) (sb j)) t
      hi : p { fst := i, snd := j }.fst
      hj : p { fst := i, snd := j }.snd
      k : ι
      hk : p k
      ki : HasSubset.Subset (sa k) (sa { fst := i, snd := j }.fst)
      kj : HasSubset.Subset (sb k) (sb { fst := i, snd := j }.snd)
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (SProd.sprod (sa i) (sb i)) t)
    -/
    exact ⟨k, hk, (Set.prod_mono ki kj).trans hsub⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      la : Filter α
      sa : ι → Set α
      lb : Filter β
      p : ι → Prop
      sb : ι → Set β
      hla : la.HasBasis p sa
      hlb : lb.HasBasis p sb
      h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
      t : Set (Prod α β)
      ⊢ (Exists fun i => And (p i) (HasSubset.Subset (SProd.sprod (sa i) (sb i)) t)) …
    -/
  · rintro ⟨i, hi, h⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Sort u_4
      la : Filter α
      sa : ι → Set α
      lb : Filter β
      p : ι → Prop
      sb : ι → Set β
      hla : la.HasBasis p sa
      hlb : lb.HasBasis p sb
      h_dir : ∀ {i j : ι}, p i → p j → Exists fun k => And (p k) (And (HasSubset.Sub …
      t : Set (Prod α β)
      i : ι
      hi : p i
      h : HasSubset.Subset (SProd.sprod (sa i) (sb i)) t
      ⊢ Exists fun i => And (And (p i.fst) (p i.snd)) (HasSubset.Subset (SProd.sprod …
    -/
    exact ⟨⟨i, i⟩, ⟨hi, hi⟩, h⟩
    /-
      🎉 no goals
    -/


theorem HasBasis.prod_same_index_mono {ι : Type*} [LinearOrder ι] {p : ι → Prop} {sa : ι → Set α}
    {sb : ι → Set β} (hla : la.HasBasis p sa) (hlb : lb.HasBasis p sb)
    (hsa : MonotoneOn sa { i | p i }) (hsb : MonotoneOn sb { i | p i }) :
    (la ×ˢ lb).HasBasis p fun i => sa i ×ˢ sb i :=
  hla.prod_same_index hlb fun {i j} hi hj =>
    have : p (min i j) := min_rec' _ hi hj
    ⟨min i j, this, hsa this hi <| min_le_left _ _, hsb this hj <| min_le_right _ _⟩


theorem HasBasis.prod_same_index_anti {ι : Type*} [LinearOrder ι] {p : ι → Prop} {sa : ι → Set α}
    {sb : ι → Set β} (hla : la.HasBasis p sa) (hlb : lb.HasBasis p sb)
    (hsa : AntitoneOn sa { i | p i }) (hsb : AntitoneOn sb { i | p i }) :
    (la ×ˢ lb).HasBasis p fun i => sa i ×ˢ sb i :=
  @HasBasis.prod_same_index_mono _ _ _ _ ιᵒᵈ _ _ _ _ hla hlb hsa.dual_left hsb.dual_left


theorem HasBasis.prod_self (hl : la.HasBasis pa sa) :
    (la ×ˢ la).HasBasis pa fun i => sa i ×ˢ sa i :=
  hl.prod_same_index hl fun {i j} hi hj => by
    simpa only [exists_prop, subset_inter_iff] using
      hl.mem_iff.1 (inter_mem (hl.mem_of_mem hi) (hl.mem_of_mem hj))


theorem mem_prod_self_iff {s} : s ∈ la ×ˢ la ↔ ∃ t ∈ la, t ×ˢ t ⊆ s :=
  la.basis_sets.prod_self.mem_iff


lemma eventually_prod_self_iff {r : α → α → Prop} :
    (∀ᶠ x in la ×ˢ la, r x.1 x.2) ↔ ∃ t ∈ la, ∀ x ∈ t, ∀ y ∈ t, r x y :=
                                /-
                                  α : Type u_1
                                  la : Filter α
                                  r : α → α → Prop
                                  ⊢ Iff (Exists fun t => And (Membership.mem la t) (HasSubset.Subset (SProd.spro …
                                -/
  mem_prod_self_iff.trans <| by simp only [prod_subset_iff, mem_setOf_eq]
                                /-
                                  🎉 no goals
                                -/


theorem HasAntitoneBasis.prod {ι : Type*} [LinearOrder ι] {f : Filter α} {g : Filter β}
    {s : ι → Set α} {t : ι → Set β} (hf : HasAntitoneBasis f s) (hg : HasAntitoneBasis g t) :
    HasAntitoneBasis (f ×ˢ g) fun n => s n ×ˢ t n :=
  ⟨hf.1.prod_same_index_anti hg.1 (hf.2.antitoneOn _) (hg.2.antitoneOn _), hf.2.set_prod hg.2⟩


theorem HasBasis.coprod {ι ι' : Type*} {pa : ι → Prop} {sa : ι → Set α} {pb : ι' → Prop}
    {sb : ι' → Set β} (hla : la.HasBasis pa sa) (hlb : lb.HasBasis pb sb) :
    (la.coprod lb).HasBasis (fun i : ι × ι' => pa i.1 ∧ pb i.2) fun i =>
      Prod.fst ⁻¹' sa i.1 ∪ Prod.snd ⁻¹' sb i.2 :=
  (hla.comap Prod.fst).sup (hlb.comap Prod.snd)


theorem map_sigma_mk_comap {π : α → Type*} {π' : β → Type*} {f : α → β}
    (hf : Function.Injective f) (g : ∀ a, π a → π' (f a)) (a : α) (l : Filter (π' (f a))) :
    map (Sigma.mk a) (comap (g a) l) = comap (Sigma.map f g) (map (Sigma.mk (f a)) l) := by
  /-
    α : Type u_1
    β : Type u_2
    π : α → Type u_6
    π' : β → Type u_7
    f : α → β
    hf : Function.Injective f
    g : (a : α) → π a → π' (f a)
    a : α
    l : Filter (π' (f a))
    ⊢ Eq (Filter.map (Sigma.mk a) (Filter.comap (g a) l)) (Filter.comap (Sigma.map …
  -/
  refine (((basis_sets _).comap _).map _).eq_of_same_basis ?_
  /-
    α : Type u_1
    β : Type u_2
    π : α → Type u_6
    π' : β → Type u_7
    f : α → β
    hf : Function.Injective f
    g : (a : α) → π a → π' (f a)
    a : α
    l : Filter (π' (f a))
    ⊢ (Filter.comap (Sigma.map f g) (Filter.map (Sigma.mk (f a)) l)).HasBasis (fun …
  -/
  convert ((basis_sets l).map (Sigma.mk (f a))).comap (Sigma.map f g)
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    π : α → Type u_6
    π' : β → Type u_7
    f : α → β
    hf : Function.Injective f
    g : (a : α) → π a → π' (f a)
    a : α
    l : Filter (π' (f a))
    x✝ : Set (π' (f a))
    ⊢ Eq (Set.image (Sigma.mk a) (Set.preimage (g a) (id x✝))) (Set.preimage (Sigm …
  -/
  apply image_sigmaMk_preimage_sigmaMap hf
  /-
    🎉 no goals
  -/


