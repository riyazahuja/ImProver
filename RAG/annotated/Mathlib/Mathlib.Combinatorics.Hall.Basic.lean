/-- The set of matchings for `t` when restricted to a `Finset` of `ι`. -/
def hallMatchingsOn {ι : Type u} {α : Type v} (t : ι → Finset α) (ι' : Finset ι) :=
  { f : ι' → α | Function.Injective f ∧ ∀ (x : {x // x ∈ ι'}), f x ∈ t x }


/-- Given a matching on a finset, construct the restriction of that matching to a subset. -/
def hallMatchingsOn.restrict {ι : Type u} {α : Type v} (t : ι → Finset α) {ι' ι'' : Finset ι}
    (h : ι' ⊆ ι'') (f : hallMatchingsOn t ι'') : hallMatchingsOn t ι' := by
  /-
    ι : Type u
    α : Type v
    t : ι → Finset α
    ι' ι'' : Finset ι
    h : HasSubset.Subset ι' ι''
    f : ↑(hallMatchingsOn t ι'')
    ⊢ ↑(hallMatchingsOn t ι')
  -/
  refine ⟨fun i => f.val ⟨i, h i.property⟩, ?_⟩
  /-
    ι : Type u
    α : Type v
    t : ι → Finset α
    ι' ι'' : Finset ι
    h : HasSubset.Subset ι' ι''
    f : ↑(hallMatchingsOn t ι'')
    ⊢ Membership.mem (hallMatchingsOn t ι') fun i => ↑f ⟨↑i, ⋯⟩
  -/
  cases' f.property with hinj hc
  /-
    case intro
    ι : Type u
    α : Type v
    t : ι → Finset α
    ι' ι'' : Finset ι
    h : HasSubset.Subset ι' ι''
    f : ↑(hallMatchingsOn t ι'')
    hinj : Function.Injective ↑f
    hc : ∀ (x : Subtype fun x => Membership.mem ι'' x), Membership.mem (t ↑x) (↑f x)
    ⊢ Membership.mem (hallMatchingsOn t ι') fun i => ↑f ⟨↑i, ⋯⟩
  -/
  refine ⟨?_, fun i => hc ⟨i, h i.property⟩⟩
  /-
    case intro
    ι : Type u
    α : Type v
    t : ι → Finset α
    ι' ι'' : Finset ι
    h : HasSubset.Subset ι' ι''
    f : ↑(hallMatchingsOn t ι'')
    hinj : Function.Injective ↑f
    hc : ∀ (x : Subtype fun x => Membership.mem ι'' x), Membership.mem (t ↑x) (↑f x)
    ⊢ Function.Injective fun i => ↑f ⟨↑i, ⋯⟩
  -/
  rintro ⟨i, hi⟩ ⟨j, hj⟩ hh
  /-
    case intro.mk.mk
    ι : Type u
    α : Type v
    t : ι → Finset α
    ι' ι'' : Finset ι
    h : HasSubset.Subset ι' ι''
    f : ↑(hallMatchingsOn t ι'')
    hinj : Function.Injective ↑f
    hc : ∀ (x : Subtype fun x => Membership.mem ι'' x), Membership.mem (t ↑x) (↑f x)
    i : ι
    hi : Membership.mem ι' i
    j : ι
    hj : Membership.mem ι' j
    hh : Eq ((fun i => ↑f ⟨↑i, ⋯⟩) ⟨i, hi⟩) ((fun i => ↑f ⟨↑i, ⋯⟩) ⟨j, hj⟩)
    ⊢ Eq ⟨i, hi⟩ ⟨j, hj⟩
  -/
  simpa only [Subtype.mk_eq_mk] using hinj hh
  /-
    🎉 no goals
  -/


/-- When the Hall condition is satisfied, the set of matchings on a finite set is nonempty.
This is where `Finset.all_card_le_biUnion_card_iff_existsInjective'` comes into the argument. -/
theorem hallMatchingsOn.nonempty {ι : Type u} {α : Type v} [DecidableEq α] (t : ι → Finset α)
    (h : ∀ s : Finset ι, #s ≤ #(s.biUnion t)) (ι' : Finset ι) :
    Nonempty (hallMatchingsOn t ι') := by
  classical
    refine ⟨Classical.indefiniteDescription _ ?_⟩
    apply (all_card_le_biUnion_card_iff_existsInjective' fun i : ι' => t i).mp
    intro s'
    convert h (s'.image (↑)) using 1
    · simp only [card_image_of_injective s' Subtype.coe_injective]
    · rw [image_biUnion]


/-- This is the `hallMatchingsOn` sets assembled into a directed system.
-/
def hallMatchingsFunctor {ι : Type u} {α : Type v} (t : ι → Finset α) :
    (Finset ι)ᵒᵖ ⥤ Type max u v where
  obj ι' := hallMatchingsOn t ι'.unop
  map {_ _} g f := hallMatchingsOn.restrict t (CategoryTheory.leOfHom g.unop) f


instance hallMatchingsOn.finite {ι : Type u} {α : Type v} (t : ι → Finset α) (ι' : Finset ι) :
    Finite (hallMatchingsOn t ι') := by
  classical
    rw [hallMatchingsOn]
    let g : hallMatchingsOn t ι' → ι' → ι'.biUnion t := by
      rintro f i
      refine ⟨f.val i, ?_⟩
      rw [mem_biUnion]
      exact ⟨i, i.property, f.property.2 i⟩
    apply Finite.of_injective g
    intro f f' h
    ext a
    rw [funext_iff] at h
    simpa [g] using h a


/-- This is the version of **Hall's Marriage Theorem** in terms of indexed
families of finite sets `t : ι → Finset α`.  It states that there is a
set of distinct representatives if and only if every union of `k` of the
sets has at least `k` elements.

Recall that `s.biUnion t` is the union of all the sets `t i` for `i ∈ s`.

This theorem is bootstrapped from `Finset.all_card_le_biUnion_card_iff_exists_injective'`,
which has the additional constraint that `ι` is a `Fintype`.
-/
theorem Finset.all_card_le_biUnion_card_iff_exists_injective {ι : Type u} {α : Type v}
    [DecidableEq α] (t : ι → Finset α) :
    (∀ s : Finset ι, #s ≤ #(s.biUnion t)) ↔
      ∃ f : ι → α, Function.Injective f ∧ ∀ x, f x ∈ t x := by
  /-
    ι : Type u
    α : Type v
    inst✝ : DecidableEq α
    t : ι → Finset α
    ⊢ Iff (∀ (s : Finset ι), LE.le s.card (s.biUnion t).card) (Exists fun f => And …
  -/
  constructor
    /-
      case mp
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      ⊢ (∀ (s : Finset ι), LE.le s.card (s.biUnion t).card) → Exists fun f => And (F …
    -/
  · intro h
    -- Set up the functor
    haveI : ∀ ι' : (Finset ι)ᵒᵖ, Nonempty ((hallMatchingsFunctor t).obj ι') := fun ι' =>
      hallMatchingsOn.nonempty t h ι'.unop
    classical
      haveI : ∀ ι' : (Finset ι)ᵒᵖ, Finite ((hallMatchingsFunctor t).obj ι') := by
        intro ι'
        rw [hallMatchingsFunctor]
        infer_instance
      -- Apply the compactness argument
      obtain ⟨u, hu⟩ := nonempty_sections_of_finite_inverse_system (hallMatchingsFunctor t)
      -- Interpret the resulting section of the inverse limit
      refine ⟨?_, ?_, ?_⟩
      ·-- Build the matching function from the section
        exact fun i =>
          (u (Opposite.op ({i} : Finset ι))).val ⟨i, by simp only [Opposite.unop_op, mem_singleton]⟩
      · -- Show that it is injective
        intro i i'
        have subi : ({i} : Finset ι) ⊆ {i, i'} := by simp
        have subi' : ({i'} : Finset ι) ⊆ {i, i'} := by simp
        rw [← Finset.le_iff_subset] at subi subi'
        simp only
        rw [← hu (CategoryTheory.homOfLE subi).op, ← hu (CategoryTheory.homOfLE subi').op]
        let uii' := u (Opposite.op ({i, i'} : Finset ι))
        exact fun h => Subtype.mk_eq_mk.mp (uii'.property.1 h)
      · -- Show that it maps each index to the corresponding finite set
        intro i
        apply (u (Opposite.op ({i} : Finset ι))).property.2
  · -- The reverse direction is a straightforward cardinality argument
    /-
      case mpr
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      ⊢ (Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x) …
    -/
    rintro ⟨f, hf₁, hf₂⟩ s
    /-
      case mpr.intro.intro
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ LE.le s.card (s.biUnion t).card
    -/
    rw [← Finset.card_image_of_injective s hf₁]
    /-
      case mpr.intro.intro
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ LE.le (Finset.image f s).card (s.biUnion t).card
    -/
    apply Finset.card_le_card
    /-
      case mpr.intro.intro.a
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ HasSubset.Subset (Finset.image f s) (s.biUnion t)
    -/
    intro
    /-
      case mpr.intro.intro.a
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      a✝ : α
      ⊢ Membership.mem (Finset.image f s) a✝ → Membership.mem (s.biUnion t) a✝
    -/
    rw [Finset.mem_image, Finset.mem_biUnion]
    /-
      case mpr.intro.intro.a
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      a✝ : α
      ⊢ (Exists fun a => And (Membership.mem s a) (Eq (f a) a✝)) → Exists fun a => A …
    -/
    rintro ⟨x, hx, rfl⟩
    /-
      case mpr.intro.intro.a.intro.intro
      ι : Type u
      α : Type v
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      x : ι
      hx : Membership.mem s x
      ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem (t a) (f x))
    -/
    exact ⟨x, hx, hf₂ x⟩
    /-
      🎉 no goals
    -/


/-- Given a relation such that the image of every singleton set is finite, then the image of every
finite set is finite. -/
instance {α : Type u} {β : Type v} [DecidableEq β] (r : α → β → Prop)
    [∀ a : α, Fintype (Rel.image r {a})] (A : Finset α) : Fintype (Rel.image r A) := by
  have h : Rel.image r A = (A.biUnion fun a => (Rel.image r {a}).toFinset : Set β) := by
    ext
    simp [Rel.image]
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    A : Finset α
    h : Eq (Rel.image r ↑A) ↑(A.biUnion fun a => (Rel.image r (Singleton.singleton …
    ⊢ Fintype ↑(Rel.image r ↑A)
  -/
  rw [h]
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    A : Finset α
    h : Eq (Rel.image r ↑A) ↑(A.biUnion fun a => (Rel.image r (Singleton.singleton …
    ⊢ Fintype ↑↑(A.biUnion fun a => (Rel.image r (Singleton.singleton a)).toFinset)
  -/
  apply FinsetCoe.fintype
  /-
    🎉 no goals
  -/


/-- This is a version of **Hall's Marriage Theorem** in terms of a relation
between types `α` and `β` such that `α` is finite and the image of
each `x : α` is finite (it suffices for `β` to be finite; see
`Fintype.all_card_le_filter_rel_iff_exists_injective`).  There is
a transversal of the relation (an injective function `α → β` whose graph is
a subrelation of the relation) iff every subset of
`k` terms of `α` is related to at least `k` terms of `β`.

Note: if `[Fintype β]`, then there exist instances for `[∀ (a : α), Fintype (Rel.image r {a})]`.
-/
theorem Fintype.all_card_le_rel_image_card_iff_exists_injective {α : Type u} {β : Type v}
    [DecidableEq β] (r : α → β → Prop) [∀ a : α, Fintype (Rel.image r {a})] :
    (∀ A : Finset α, #A ≤ Fintype.card (Rel.image r A)) ↔
      ∃ f : α → β, Function.Injective f ∧ ∀ x, r x (f x) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Fintype.card ↑(Rel.image r ↑A))) (Exist …
  -/
  let r' a := (Rel.image r {a}).toFinset
  have h : ∀ A : Finset α, Fintype.card (Rel.image r A) = #(A.biUnion r') := by
    intro A
    rw [← Set.toFinset_card]
    apply congr_arg
    ext b
    simp [r', Rel.image]
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    r' : α → Finset β := fun a => (Rel.image r (Singleton.singleton a)).toFinset
    h : ∀ (A : Finset α), Eq (Fintype.card ↑(Rel.image r ↑A)) (A.biUnion r').card
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Fintype.card ↑(Rel.image r ↑A))) (Exist …
  -/
  have h' : ∀ (f : α → β) (x), r x (f x) ↔ f x ∈ r' x := by simp [r', Rel.image]
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    r' : α → Finset β := fun a => (Rel.image r (Singleton.singleton a)).toFinset
    h : ∀ (A : Finset α), Eq (Fintype.card ↑(Rel.image r ↑A)) (A.biUnion r').card
    h' : ∀ (f : α → β) (x : α), Iff (r x (f x)) (Membership.mem (r' x) (f x))
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Fintype.card ↑(Rel.image r ↑A))) (Exist …
  -/
  simp only [h, h']
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq β
    r : α → β → Prop
    inst✝ : (a : α) → Fintype ↑(Rel.image r (Singleton.singleton a))
    r' : α → Finset β := fun a => (Rel.image r (Singleton.singleton a)).toFinset
    h : ∀ (A : Finset α), Eq (Fintype.card ↑(Rel.image r ↑A)) (A.biUnion r').card
    h' : ∀ (f : α → β) (x : α), Iff (r x (f x)) (Membership.mem (r' x) (f x))
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (A.biUnion r').card) (Exists fun f => An …
  -/
  apply Finset.all_card_le_biUnion_card_iff_exists_injective
  /-
    🎉 no goals
  -/

-- TODO: decidable_pred makes Yael sad. When an appropriate decidable_rel-like exists, fix it.

/-- This is a version of **Hall's Marriage Theorem** in terms of a relation to a finite type.
There is a transversal of the relation (an injective function `α → β` whose graph is a subrelation
of the relation) iff every subset of `k` terms of `α` is related to at least `k` terms of `β`.

It is like `Fintype.all_card_le_rel_image_card_iff_exists_injective` but uses `Finset.filter`
rather than `Rel.image`.
-/
theorem Fintype.all_card_le_filter_rel_iff_exists_injective {α : Type u} {β : Type v} [Fintype β]
    (r : α → β → Prop) [∀ a, DecidablePred (r a)] :
    (∀ A : Finset α, #A ≤ #{b | ∃ a ∈ A, r a b}) ↔ ∃ f : α → β, Injective f ∧ ∀ x, r x (f x) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype β
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Finset.filter (fun b => Exists fun a => …
  -/
  haveI := Classical.decEq β
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype β
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    this : DecidableEq β
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Finset.filter (fun b => Exists fun a => …
  -/
  let r' a : Finset β := {b | r a b}
  have h : ∀ A : Finset α, ({b | ∃ a ∈ A, r a b} : Finset _) = A.biUnion r' := by
    intro A
    ext b
    simp [r']
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype β
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    this : DecidableEq β
    r' : α → Finset β := fun a => Finset.filter (fun b => r a b) Finset.univ
    h : ∀ (A : Finset α), Eq (Finset.filter (fun b => Exists fun a => And (Members …
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Finset.filter (fun b => Exists fun a => …
  -/
  have h' : ∀ (f : α → β) (x), r x (f x) ↔ f x ∈ r' x := by simp [r']
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype β
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    this : DecidableEq β
    r' : α → Finset β := fun a => Finset.filter (fun b => r a b) Finset.univ
    h : ∀ (A : Finset α), Eq (Finset.filter (fun b => Exists fun a => And (Members …
    h' : ∀ (f : α → β) (x : α), Iff (r x (f x)) (Membership.mem (r' x) (f x))
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (Finset.filter (fun b => Exists fun a => …
  -/
  simp_rw [h, h']
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype β
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    this : DecidableEq β
    r' : α → Finset β := fun a => Finset.filter (fun b => r a b) Finset.univ
    h : ∀ (A : Finset α), Eq (Finset.filter (fun b => Exists fun a => And (Members …
    h' : ∀ (f : α → β) (x : α), Iff (r x (f x)) (Membership.mem (r' x) (f x))
    ⊢ Iff (∀ (A : Finset α), LE.le A.card (A.biUnion r').card) (Exists fun f => An …
  -/
  apply Finset.all_card_le_biUnion_card_iff_exists_injective
  /-
    🎉 no goals
  -/

