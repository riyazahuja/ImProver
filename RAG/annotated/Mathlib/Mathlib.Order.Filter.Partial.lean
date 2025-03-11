/-- The forward map of a filter under a relation. Generalization of `Filter.map` to relations. Note
that `Rel.core` generalizes `Set.preimage`. -/
def rmap (r : Rel α β) (l : Filter α) : Filter β where
  sets := { s | r.core s ∈ l }
                  /-
                    α : Type u
                    β : Type v
                    γ : Type w
                    r : Rel α β
                    l : Filter α
                    ⊢ Membership.mem (setOf fun s => Membership.mem l (r.core s)) Set.univ
                  -/
  univ_sets := by simp
                  /-
                    🎉 no goals
                  -/
  sets_of_superset hs st := mem_of_superset hs (Rel.core_mono _ st)
  inter_sets hs ht := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      l : Filter α
      x✝ y✝ : Set β
      hs : Membership.mem (setOf fun s => Membership.mem l (r.core s)) x✝
      ht : Membership.mem (setOf fun s => Membership.mem l (r.core s)) y✝
      ⊢ Membership.mem (setOf fun s => Membership.mem l (r.core s)) (Inter.inter x✝  …
    -/
    simp only [Set.mem_setOf_eq]
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      l : Filter α
      x✝ y✝ : Set β
      hs : Membership.mem (setOf fun s => Membership.mem l (r.core s)) x✝
      ht : Membership.mem (setOf fun s => Membership.mem l (r.core s)) y✝
      ⊢ Membership.mem l (r.core (Inter.inter x✝ y✝))
    -/
    convert inter_mem hs ht
    /-
      case h.e'_5
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      l : Filter α
      x✝ y✝ : Set β
      hs : Membership.mem (setOf fun s => Membership.mem l (r.core s)) x✝
      ht : Membership.mem (setOf fun s => Membership.mem l (r.core s)) y✝
      ⊢ Eq (r.core (Inter.inter x✝ y✝)) (Inter.inter (r.core x✝) (r.core y✝))
    -/
    rw [← Rel.core_inter]
    /-
      🎉 no goals
    -/


theorem rmap_sets (r : Rel α β) (l : Filter α) : (l.rmap r).sets = r.core ⁻¹' l.sets :=
  rfl


@[simp]
theorem mem_rmap (r : Rel α β) (l : Filter α) (s : Set β) : s ∈ l.rmap r ↔ r.core s ∈ l :=
  Iff.rfl


@[simp]
theorem rmap_rmap (r : Rel α β) (s : Rel β γ) (l : Filter α) :
    rmap s (rmap r l) = rmap (r.comp s) l :=
                  /-
                    α : Type u
                    β : Type v
                    γ : Type w
                    r : Rel α β
                    s : Rel β γ
                    l : Filter α
                    ⊢ Eq (Filter.rmap s (Filter.rmap r l)).sets (Filter.rmap (r.comp s) l).sets
                  -/
  filter_eq <| by simp [rmap_sets, Set.preimage, Rel.core_comp]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem rmap_compose (r : Rel α β) (s : Rel β γ) : rmap s ∘ rmap r = rmap (r.comp s) :=
  funext <| rmap_rmap _ _


/-- Generic "limit of a relation" predicate. `RTendsto r l₁ l₂` asserts that for every
`l₂`-neighborhood `a`, the `r`-core of `a` is an `l₁`-neighborhood. One generalization of
`Filter.Tendsto` to relations. -/
def RTendsto (r : Rel α β) (l₁ : Filter α) (l₂ : Filter β) :=
  l₁.rmap r ≤ l₂


theorem rtendsto_def (r : Rel α β) (l₁ : Filter α) (l₂ : Filter β) :
    RTendsto r l₁ l₂ ↔ ∀ s ∈ l₂, r.core s ∈ l₁ :=
  Iff.rfl


/-- One way of taking the inverse map of a filter under a relation. One generalization of
`Filter.comap` to relations. Note that `Rel.core` generalizes `Set.preimage`. -/
def rcomap (r : Rel α β) (f : Filter β) : Filter α where
  sets := Rel.image (fun s t => r.core s ⊆ t) f.sets
  univ_sets := ⟨Set.univ, univ_mem, Set.subset_univ _⟩
  sets_of_superset := fun ⟨a', ha', ma'a⟩ ab => ⟨a', ha', ma'a.trans ab⟩
  inter_sets := fun ⟨a', ha₁, ha₂⟩ ⟨b', hb₁, hb₂⟩ =>
    ⟨a' ∩ b', inter_mem ha₁ hb₁, (r.core_inter a' b').subset.trans (Set.inter_subset_inter ha₂ hb₂)⟩


theorem rcomap_sets (r : Rel α β) (f : Filter β) :
    (rcomap r f).sets = Rel.image (fun s t => r.core s ⊆ t) f.sets :=
  rfl


theorem rcomap_rcomap (r : Rel α β) (s : Rel β γ) (l : Filter γ) :
    rcomap r (rcomap s l) = rcomap (r.comp s) l :=
  filter_eq <| by
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      ⊢ Eq (Filter.rcomap r (Filter.rcomap s l)).sets (Filter.rcomap (r.comp s) l).s …
    -/
    ext t; simp only [rcomap_sets, Rel.image, Filter.mem_sets, Set.mem_setOf_eq, Rel.core_comp]
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t : Set α
      ⊢ Iff (Exists fun x => And (Exists fun x_1 => And (Membership.mem l x_1) (HasS …
    -/
    constructor
      /-
        case h.mp
        α : Type u
        β : Type v
        γ : Type w
        r : Rel α β
        s : Rel β γ
        l : Filter γ
        t : Set α
        ⊢ (Exists fun x => And (Exists fun x_1 => And (Membership.mem l x_1) (HasSubse …
      -/
    · rintro ⟨u, ⟨v, vsets, hv⟩, h⟩
      /-
        case h.mp.intro.intro.intro.intro
        α : Type u
        β : Type v
        γ : Type w
        r : Rel α β
        s : Rel β γ
        l : Filter γ
        t : Set α
        u : Set β
        h : HasSubset.Subset (r.core u) t
        v : Set γ
        vsets : Membership.mem l v
        hv : HasSubset.Subset (s.core v) u
        ⊢ Exists fun x => And (Membership.mem l x) (HasSubset.Subset (r.core (s.core x …
      -/
      exact ⟨v, vsets, Set.Subset.trans (Rel.core_mono _ hv) h⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t : Set α
      ⊢ (Exists fun x => And (Membership.mem l x) (HasSubset.Subset (r.core (s.core  …
    -/
    rintro ⟨t, tsets, ht⟩
    /-
      case h.mpr.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t✝ : Set α
      t : Set γ
      tsets : Membership.mem l t
      ht : HasSubset.Subset (r.core (s.core t)) t✝
      ⊢ Exists fun x => And (Exists fun x_1 => And (Membership.mem l x_1) (HasSubset …
    -/
    exact ⟨Rel.core s t, ⟨t, tsets, Set.Subset.rfl⟩, ht⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem rcomap_compose (r : Rel α β) (s : Rel β γ) : rcomap r ∘ rcomap s = rcomap (r.comp s) :=
  funext <| rcomap_rcomap _ _


theorem rtendsto_iff_le_rcomap (r : Rel α β) (l₁ : Filter α) (l₂ : Filter β) :
    RTendsto r l₁ l₂ ↔ l₁ ≤ l₂.rcomap r := by
  /-
    α : Type u
    β : Type v
    r : Rel α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Iff (Filter.RTendsto r l₁ l₂) (LE.le l₁ (Filter.rcomap r l₂))
  -/
  rw [rtendsto_def]
  /-
    α : Type u
    β : Type v
    r : Rel α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Iff (∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (r.core s)) (LE. …
  -/
  simp_rw [← l₂.mem_sets]
  /-
    α : Type u
    β : Type v
    r : Rel α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Iff (∀ (s : Set β), Membership.mem l₂.sets s → Membership.mem l₁ (r.core s)) …
  -/
  constructor
    /-
      case mp
      α : Type u
      β : Type v
      r : Rel α β
      l₁ : Filter α
      l₂ : Filter β
      ⊢ (∀ (s : Set β), Membership.mem l₂.sets s → Membership.mem l₁ (r.core s)) → L …
    -/
  · simpa [Filter.le_def, rcomap, Rel.mem_image] using fun h s t tl₂ => mem_of_superset (h t tl₂)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      r : Rel α β
      l₁ : Filter α
      l₂ : Filter β
      ⊢ LE.le l₁ (Filter.rcomap r l₂) → ∀ (s : Set β), Membership.mem l₂.sets s → Me …
    -/
  · simpa [Filter.le_def, rcomap, Rel.mem_image] using fun h t tl₂ => h _ t tl₂ Set.Subset.rfl
    /-
      🎉 no goals
    -/

-- Interestingly, there does not seem to be a way to express this relation using a forward map.
-- Given a filter `f` on `α`, we want a filter `f'` on `β` such that `r.preimage s ∈ f` if
-- and only if `s ∈ f'`. But the intersection of two sets satisfying the lhs may be empty.

/-- One way of taking the inverse map of a filter under a relation. Generalization of `Filter.comap`
to relations. -/
def rcomap' (r : Rel α β) (f : Filter β) : Filter α where
  sets := Rel.image (fun s t => r.preimage s ⊆ t) f.sets
  univ_sets := ⟨Set.univ, univ_mem, Set.subset_univ _⟩
  sets_of_superset := fun ⟨a', ha', ma'a⟩ ab => ⟨a', ha', ma'a.trans ab⟩
  inter_sets := fun ⟨a', ha₁, ha₂⟩ ⟨b', hb₁, hb₂⟩ =>
    ⟨a' ∩ b', inter_mem ha₁ hb₁,
      (@Rel.preimage_inter _ _ r _ _).trans (Set.inter_subset_inter ha₂ hb₂)⟩


@[simp]
theorem mem_rcomap' (r : Rel α β) (l : Filter β) (s : Set α) :
    s ∈ l.rcomap' r ↔ ∃ t ∈ l, r.preimage t ⊆ s :=
  Iff.rfl


theorem rcomap'_sets (r : Rel α β) (f : Filter β) :
    (rcomap' r f).sets = Rel.image (fun s t => r.preimage s ⊆ t) f.sets :=
  rfl


@[simp]
theorem rcomap'_rcomap' (r : Rel α β) (s : Rel β γ) (l : Filter γ) :
    rcomap' r (rcomap' s l) = rcomap' (r.comp s) l :=
  Filter.ext fun t => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t : Set α
      ⊢ Iff (Membership.mem (Filter.rcomap' r (Filter.rcomap' s l)) t) (Membership.m …
    -/
    simp only [mem_rcomap', Rel.preimage_comp]
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t : Set α
      ⊢ Iff (Exists fun t_1 => And (Exists fun t => And (Membership.mem l t) (HasSub …
    -/
    constructor
      /-
        case mp
        α : Type u
        β : Type v
        γ : Type w
        r : Rel α β
        s : Rel β γ
        l : Filter γ
        t : Set α
        ⊢ (Exists fun t_1 => And (Exists fun t => And (Membership.mem l t) (HasSubset. …
      -/
    · rintro ⟨u, ⟨v, vsets, hv⟩, h⟩
      /-
        case mp.intro.intro.intro.intro
        α : Type u
        β : Type v
        γ : Type w
        r : Rel α β
        s : Rel β γ
        l : Filter γ
        t : Set α
        u : Set β
        h : HasSubset.Subset (r.preimage u) t
        v : Set γ
        vsets : Membership.mem l v
        hv : HasSubset.Subset (s.preimage v) u
        ⊢ Exists fun t_1 => And (Membership.mem l t_1) (HasSubset.Subset (r.preimage ( …
      -/
      exact ⟨v, vsets, (Rel.preimage_mono _ hv).trans h⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t : Set α
      ⊢ (Exists fun t_1 => And (Membership.mem l t_1) (HasSubset.Subset (r.preimage  …
    -/
    rintro ⟨t, tsets, ht⟩
    /-
      case mpr.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      r : Rel α β
      s : Rel β γ
      l : Filter γ
      t✝ : Set α
      t : Set γ
      tsets : Membership.mem l t
      ht : HasSubset.Subset (r.preimage (s.preimage t)) t✝
      ⊢ Exists fun t => And (Exists fun t_1 => And (Membership.mem l t_1) (HasSubset …
    -/
    exact ⟨s.preimage t, ⟨t, tsets, Set.Subset.rfl⟩, ht⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem rcomap'_compose (r : Rel α β) (s : Rel β γ) : rcomap' r ∘ rcomap' s = rcomap' (r.comp s) :=
  funext <| rcomap'_rcomap' _ _


/-- Generic "limit of a relation" predicate. `RTendsto' r l₁ l₂` asserts that for every
`l₂`-neighborhood `a`, the `r`-preimage of `a` is an `l₁`-neighborhood. One generalization of
`Filter.Tendsto` to relations. -/
def RTendsto' (r : Rel α β) (l₁ : Filter α) (l₂ : Filter β) :=
  l₁ ≤ l₂.rcomap' r


theorem rtendsto'_def (r : Rel α β) (l₁ : Filter α) (l₂ : Filter β) :
    RTendsto' r l₁ l₂ ↔ ∀ s ∈ l₂, r.preimage s ∈ l₁ := by
  /-
    α : Type u
    β : Type v
    r : Rel α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Iff (Filter.RTendsto' r l₁ l₂) (∀ (s : Set β), Membership.mem l₂ s → Members …
  -/
  unfold RTendsto' rcomap'; constructor
    /-
      case mp
      α : Type u
      β : Type v
      r : Rel α β
      l₁ : Filter α
      l₂ : Filter β
      ⊢ LE.le l₁ { sets := Rel.image (fun s t => HasSubset.Subset (r.preimage s) t)  …
    -/
  · simpa [le_def, Rel.mem_image] using fun h s hs => h _ _ hs Set.Subset.rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      r : Rel α β
      l₁ : Filter α
      l₂ : Filter β
      ⊢ (∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (r.preimage s)) → LE …
    -/
  · simpa [le_def, Rel.mem_image] using fun h s t ht => mem_of_superset (h t ht)
    /-
      🎉 no goals
    -/


theorem tendsto_iff_rtendsto (l₁ : Filter α) (l₂ : Filter β) (f : α → β) :
    Tendsto f l₁ l₂ ↔ RTendsto (Function.graph f) l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    l₁ : Filter α
    l₂ : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto f l₁ l₂) (Filter.RTendsto (Function.graph f) l₁ l₂)
  -/
  simp [tendsto_def, Function.graph, rtendsto_def, Rel.core, Set.preimage]
  /-
    🎉 no goals
  -/


theorem tendsto_iff_rtendsto' (l₁ : Filter α) (l₂ : Filter β) (f : α → β) :
    Tendsto f l₁ l₂ ↔ RTendsto' (Function.graph f) l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    l₁ : Filter α
    l₂ : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto f l₁ l₂) (Filter.RTendsto' (Function.graph f) l₁ l₂)
  -/
  simp [tendsto_def, Function.graph, rtendsto'_def, Rel.preimage_def, Set.preimage]
  /-
    🎉 no goals
  -/


/-- The forward map of a filter under a partial function. Generalization of `Filter.map` to partial
functions. -/
def pmap (f : α →. β) (l : Filter α) : Filter β :=
  Filter.rmap f.graph' l


@[simp]
theorem mem_pmap (f : α →. β) (l : Filter α) (s : Set β) : s ∈ l.pmap f ↔ f.core s ∈ l :=
  Iff.rfl


/-- Generic "limit of a partial function" predicate. `PTendsto r l₁ l₂` asserts that for every
`l₂`-neighborhood `a`, the `p`-core of `a` is an `l₁`-neighborhood. One generalization of
`Filter.Tendsto` to partial function. -/
def PTendsto (f : α →. β) (l₁ : Filter α) (l₂ : Filter β) :=
  l₁.pmap f ≤ l₂


theorem ptendsto_def (f : α →. β) (l₁ : Filter α) (l₂ : Filter β) :
    PTendsto f l₁ l₂ ↔ ∀ s ∈ l₂, f.core s ∈ l₁ :=
  Iff.rfl


theorem ptendsto_iff_rtendsto (l₁ : Filter α) (l₂ : Filter β) (f : α →. β) :
    PTendsto f l₁ l₂ ↔ RTendsto f.graph' l₁ l₂ :=
  Iff.rfl


theorem pmap_res (l : Filter α) (s : Set α) (f : α → β) :
    pmap (PFun.res f s) l = map f (l ⊓ 𝓟 s) := by
  /-
    α : Type u
    β : Type v
    l : Filter α
    s : Set α
    f : α → β
    ⊢ Eq (Filter.pmap (PFun.res f s) l) (Filter.map f (Min.min l (Filter.principal …
  -/
  ext t
  /-
    case h
    α : Type u
    β : Type v
    l : Filter α
    s : Set α
    f : α → β
    t : Set β
    ⊢ Iff (Membership.mem (Filter.pmap (PFun.res f s) l) t) (Membership.mem (Filte …
  -/
  simp only [PFun.core_res, mem_pmap, mem_map, mem_inf_principal, imp_iff_not_or]
  /-
    case h
    α : Type u
    β : Type v
    l : Filter α
    s : Set α
    f : α → β
    t : Set β
    ⊢ Iff (Membership.mem l (Union.union (HasCompl.compl s) (Set.preimage f t))) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tendsto_iff_ptendsto (l₁ : Filter α) (l₂ : Filter β) (s : Set α) (f : α → β) :
    Tendsto f (l₁ ⊓ 𝓟 s) l₂ ↔ PTendsto (PFun.res f s) l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    l₁ : Filter α
    l₂ : Filter β
    s : Set α
    f : α → β
    ⊢ Iff (Filter.Tendsto f (Min.min l₁ (Filter.principal s)) l₂) (Filter.PTendsto …
  -/
  simp only [Tendsto, PTendsto, pmap_res]
  /-
    🎉 no goals
  -/


theorem tendsto_iff_ptendsto_univ (l₁ : Filter α) (l₂ : Filter β) (f : α → β) :
    Tendsto f l₁ l₂ ↔ PTendsto (PFun.res f Set.univ) l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    l₁ : Filter α
    l₂ : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto f l₁ l₂) (Filter.PTendsto (PFun.res f Set.univ) l₁ l₂)
  -/
  rw [← tendsto_iff_ptendsto]
  /-
    α : Type u
    β : Type v
    l₁ : Filter α
    l₂ : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto f l₁ l₂) (Filter.Tendsto f (Min.min l₁ (Filter.principal …
  -/
  simp [principal_univ]
  /-
    🎉 no goals
  -/


/-- Inverse map of a filter under a partial function. One generalization of `Filter.comap` to
partial functions. -/
def pcomap' (f : α →. β) (l : Filter β) : Filter α :=
  Filter.rcomap' f.graph' l


/-- Generic "limit of a partial function" predicate. `PTendsto' r l₁ l₂` asserts that for every
`l₂`-neighborhood `a`, the `p`-preimage of `a` is an `l₁`-neighborhood. One generalization of
`Filter.Tendsto` to partial functions. -/
def PTendsto' (f : α →. β) (l₁ : Filter α) (l₂ : Filter β) :=
  l₁ ≤ l₂.rcomap' f.graph'


theorem ptendsto'_def (f : α →. β) (l₁ : Filter α) (l₂ : Filter β) :
    PTendsto' f l₁ l₂ ↔ ∀ s ∈ l₂, f.preimage s ∈ l₁ :=
  rtendsto'_def _ _ _


theorem ptendsto_of_ptendsto' {f : α →. β} {l₁ : Filter α} {l₂ : Filter β} :
    PTendsto' f l₁ l₂ → PTendsto f l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Filter.PTendsto' f l₁ l₂ → Filter.PTendsto f l₁ l₂
  -/
  rw [ptendsto_def, ptendsto'_def]
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ (∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (f.preimage s)) → ∀  …
  -/
  exact fun h s sl₂ => mem_of_superset (h s sl₂) (PFun.preimage_subset_core _ _)
  /-
    🎉 no goals
  -/


theorem ptendsto'_of_ptendsto {f : α →. β} {l₁ : Filter α} {l₂ : Filter β} (h : f.Dom ∈ l₁) :
    PTendsto f l₁ l₂ → PTendsto' f l₁ l₂ := by
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    h : Membership.mem l₁ f.Dom
    ⊢ Filter.PTendsto f l₁ l₂ → Filter.PTendsto' f l₁ l₂
  -/
  rw [ptendsto_def, ptendsto'_def]
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    h : Membership.mem l₁ f.Dom
    ⊢ (∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (f.core s)) → ∀ (s : …
  -/
  intro h' s sl₂
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    h : Membership.mem l₁ f.Dom
    h' : ∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (f.core s)
    s : Set β
    sl₂ : Membership.mem l₂ s
    ⊢ Membership.mem l₁ (f.preimage s)
  -/
  rw [PFun.preimage_eq]
  /-
    α : Type u
    β : Type v
    f : PFun α β
    l₁ : Filter α
    l₂ : Filter β
    h : Membership.mem l₁ f.Dom
    h' : ∀ (s : Set β), Membership.mem l₂ s → Membership.mem l₁ (f.core s)
    s : Set β
    sl₂ : Membership.mem l₂ s
    ⊢ Membership.mem l₁ (Inter.inter (f.core s) f.Dom)
  -/
  exact inter_mem (h' s sl₂) h
  /-
    🎉 no goals
  -/


