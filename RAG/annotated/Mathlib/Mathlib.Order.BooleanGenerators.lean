/--
An alternative constructor for boolean algebras.

A set of *boolean generators* in a compactly generated complete lattice is a subset `S` such that

* the elements of `S` are all atoms, and
* the set `S` satisfies an atomicity condition:
  any compact element below the supremum of a finite subset `s` of generators
  is equal to the supremum of a subset of `s`.

If the supremum of `S` is the whole lattice,
then the lattice is a boolean algebra
(see `IsCompactlyGenerated.BooleanGenerators.booleanAlgebra_of_sSup_eq_top`).
-/
structure BooleanGenerators (S : Set α) : Prop where
  /-- The elements in a collection of boolean generators are all atoms. -/
  isAtom : ∀ I ∈ S, IsAtom I
  /-- The elements in a collection of boolean generators satisfy an atomicity condition:
  any compact element below the supremum of a finite subset `s` of generators
  is equal to the supremum of a subset of `s`. -/
  finitelyAtomistic : ∀ (s : Finset α) (a : α),
      ↑s ⊆ S → IsCompactElement a → a ≤ s.sup id → ∃ t ⊆ s, a = t.sup id


lemma mono (hS : BooleanGenerators S) {T : Set α} (hTS : T ⊆ S) : BooleanGenerators T where
  isAtom I hI := hS.isAtom I (hTS hI)
  finitelyAtomistic := fun s a hs ↦ hS.finitelyAtomistic s a (le_trans hs hTS)


lemma atomistic (hS : BooleanGenerators S) (a : α) (ha : a ≤ sSup S) : ∃ T ⊆ S, a = sSup T := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    a : α
    ha : LE.le a (SupSet.sSup S)
    ⊢ Exists fun T => And (HasSubset.Subset T S) (Eq a (SupSet.sSup T))
  -/
  obtain ⟨C, hC, rfl⟩ := IsCompactlyGenerated.exists_sSup_eq a
  have aux : ∀ b : α, IsCompactElement b → b ≤ sSup S → ∃ T ⊆ S, b = sSup T := by
    intro b hb hbS
    obtain ⟨s, hs₁, hs₂⟩ := hb S hbS
    obtain ⟨t, ht, rfl⟩ := hS.finitelyAtomistic s b hs₁ hb hs₂
    refine ⟨t, ?_, Finset.sup_id_eq_sSup t⟩
    refine Set.Subset.trans ?_ hs₁
    simpa only [Finset.coe_subset] using ht
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    C : Set α
    hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
    ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
    aux : ∀ (b : α), CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S)  …
    ⊢ Exists fun T => And (HasSubset.Subset T S) (Eq (SupSet.sSup C) (SupSet.sSup  …
  -/
  choose T hT₁ hT₂ using aux
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    C : Set α
    hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
    ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
    T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
    hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
    hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
    ⊢ Exists fun T => And (HasSubset.Subset T S) (Eq (SupSet.sSup C) (SupSet.sSup  …
  -/
  use sSup {T c h₁ h₂ | (c ∈ C) (h₁ : IsCompactElement c) (h₂ : c ≤ sSup S)}
  /-
    case h
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    C : Set α
    hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
    ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
    T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
    hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
    hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
    ⊢ And (HasSubset.Subset (SupSet.sSup (setOf fun x => Exists fun c => And (Memb …
  -/
  constructor
    /-
      case h.left
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      C : Set α
      hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
      ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
      T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
      hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      ⊢ HasSubset.Subset (SupSet.sSup (setOf fun x => Exists fun c => And (Membershi …
    -/
  · apply _root_.sSup_le
    /-
      case h.left.a
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      C : Set α
      hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
      ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
      T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
      hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      ⊢ ∀ (b : Set α), Membership.mem (setOf fun x => Exists fun c => And (Membershi …
    -/
    rintro _ ⟨c, -, h₁, h₂, rfl⟩
    /-
      case h.left.a.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      C : Set α
      hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
      ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
      T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
      hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      c : α
      h₁ : CompleteLattice.IsCompactElement c
      h₂ : LE.le c (SupSet.sSup S)
      ⊢ LE.le (T c h₁ h₂) S
    -/
    apply hT₁
    /-
      🎉 no goals
    -/
    /-
      case h.right
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      C : Set α
      hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
      ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
      T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
      hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
      ⊢ Eq (SupSet.sSup C) (SupSet.sSup (SupSet.sSup (setOf fun x => Exists fun c => …
    -/
  · apply le_antisymm
      /-
        case h.right.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        ⊢ LE.le (SupSet.sSup C) (SupSet.sSup (SupSet.sSup (setOf fun x => Exists fun c …
      -/
    · apply _root_.sSup_le
      /-
        case h.right.a.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        ⊢ ∀ (b : α), Membership.mem C b → LE.le b (SupSet.sSup (SupSet.sSup (setOf fun …
      -/
      intro c hc
      /-
        case h.right.a.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        c : α
        hc : Membership.mem C c
        ⊢ LE.le c (SupSet.sSup (SupSet.sSup (setOf fun x => Exists fun c => And (Membe …
      -/
      rw [hT₂ c (hC _ hc) ((le_sSup hc).trans ha)]
      /-
        case h.right.a.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        c : α
        hc : Membership.mem C c
        ⊢ LE.le (SupSet.sSup (T c ⋯ ⋯)) (SupSet.sSup (SupSet.sSup (setOf fun x => Exis …
      -/
      apply sSup_le_sSup
      /-
        case h.right.a.a.h
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        c : α
        hc : Membership.mem C c
        ⊢ HasSubset.Subset (T c ⋯ ⋯) (SupSet.sSup (setOf fun x => Exists fun c => And  …
      -/
      apply _root_.le_sSup
      /-
        case h.right.a.a.h.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        c : α
        hc : Membership.mem C c
        ⊢ Membership.mem (setOf fun x => Exists fun c => And (Membership.mem C c) (Exi …
      -/
      use c, hc, hC _ hc, (le_sSup hc).trans ha
      /-
        🎉 no goals
      -/
    · simp only [Set.sSup_eq_sUnion, sSup_le_iff, Set.mem_sUnion, Set.mem_setOf_eq,
        forall_exists_index, and_imp]
      /-
        case h.right.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        ⊢ ∀ (b : α) (x : Set α) (x_1 : α), Membership.mem C x_1 → ∀ (x_2 : CompleteLat …
      -/
      rintro a T b hbC hb hbS rfl haT
      /-
        case h.right.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        a b : α
        hbC : Membership.mem C b
        hb : CompleteLattice.IsCompactElement b
        hbS : LE.le b (SupSet.sSup S)
        haT : Membership.mem (T b hb hbS) a
        ⊢ LE.le a (SupSet.sSup C)
      -/
      apply (le_sSup haT).trans
      /-
        case h.right.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        a b : α
        hbC : Membership.mem C b
        hb : CompleteLattice.IsCompactElement b
        hbS : LE.le b (SupSet.sSup S)
        haT : Membership.mem (T b hb hbS) a
        ⊢ LE.le (SupSet.sSup (T b hb hbS)) (SupSet.sSup C)
      -/
      rw [← hT₂]
      /-
        case h.right.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        C : Set α
        hC : ∀ (x : α), Membership.mem C x → CompleteLattice.IsCompactElement x
        ha : LE.le (SupSet.sSup C) (SupSet.sSup S)
        T : (b : α) → CompleteLattice.IsCompactElement b → LE.le b (SupSet.sSup S) → S …
        hT₁ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        hT₂ : ∀ (b : α) (a : CompleteLattice.IsCompactElement b) (a_1 : LE.le b (SupSe …
        a b : α
        hbC : Membership.mem C b
        hb : CompleteLattice.IsCompactElement b
        hbS : LE.le b (SupSet.sSup S)
        haT : Membership.mem (T b hb hbS) a
        ⊢ LE.le b (SupSet.sSup C)
      -/
      exact le_sSup hbC
      /-
        🎉 no goals
      -/


lemma isAtomistic_of_sSup_eq_top (hS : BooleanGenerators S) (h : sSup S = ⊤) :
    IsAtomistic α := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    ⊢ IsAtomistic α
  -/
  refine ⟨fun a ↦ ?_⟩
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    a : α
    ⊢ Exists fun s => And (Eq a (SupSet.sSup s)) (∀ (a : α), Membership.mem s a →  …
  -/
  obtain ⟨s, hs, hs'⟩ := hS.atomistic a (h ▸ le_top)
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    a : α
    s : Set α
    hs : HasSubset.Subset s S
    hs' : Eq a (SupSet.sSup s)
    ⊢ Exists fun s => And (Eq a (SupSet.sSup s)) (∀ (a : α), Membership.mem s a →  …
  -/
  exact ⟨s, hs', fun I hI ↦ hS.isAtom I (hs hI)⟩
  /-
    🎉 no goals
  -/


lemma mem_of_isAtom_of_le_sSup_atoms (hS : BooleanGenerators S) (a : α) (ha : IsAtom a)
    (haS : a ≤ sSup S) : a ∈ S := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    a : α
    ha : IsAtom a
    haS : LE.le a (SupSet.sSup S)
    ⊢ Membership.mem S a
  -/
  obtain ⟨T, hT, rfl⟩ := hS.atomistic a haS
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T : Set α
    hT : HasSubset.Subset T S
    ha : IsAtom (SupSet.sSup T)
    haS : LE.le (SupSet.sSup T) (SupSet.sSup S)
    ⊢ Membership.mem S (SupSet.sSup T)
  -/
  obtain rfl | ⟨a, haT⟩ := T.eq_empty_or_nonempty
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      hT : HasSubset.Subset EmptyCollection.emptyCollection S
      ha : IsAtom (SupSet.sSup EmptyCollection.emptyCollection)
      haS : LE.le (SupSet.sSup EmptyCollection.emptyCollection) (SupSet.sSup S)
      ⊢ Membership.mem S (SupSet.sSup EmptyCollection.emptyCollection)
    -/
  · simp only [sSup_empty] at ha
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      hT : HasSubset.Subset EmptyCollection.emptyCollection S
      haS : LE.le (SupSet.sSup EmptyCollection.emptyCollection) (SupSet.sSup S)
      ha : IsAtom Bot.bot
      ⊢ Membership.mem S (SupSet.sSup EmptyCollection.emptyCollection)
    -/
    exact (ha.1 rfl).elim
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T : Set α
    hT : HasSubset.Subset T S
    ha : IsAtom (SupSet.sSup T)
    haS : LE.le (SupSet.sSup T) (SupSet.sSup S)
    a : α
    haT : Membership.mem T a
    ⊢ Membership.mem S (SupSet.sSup T)
  -/
  suffices sSup T = a from this ▸ hT haT
  /-
    case intro.intro.inr.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T : Set α
    hT : HasSubset.Subset T S
    ha : IsAtom (SupSet.sSup T)
    haS : LE.le (SupSet.sSup T) (SupSet.sSup S)
    a : α
    haT : Membership.mem T a
    ⊢ Eq (SupSet.sSup T) a
  -/
  have : a ≤ sSup T := le_sSup haT
  /-
    case intro.intro.inr.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T : Set α
    hT : HasSubset.Subset T S
    ha : IsAtom (SupSet.sSup T)
    haS : LE.le (SupSet.sSup T) (SupSet.sSup S)
    a : α
    haT : Membership.mem T a
    this : LE.le a (SupSet.sSup T)
    ⊢ Eq (SupSet.sSup T) a
  -/
  rwa [ha.le_iff_eq, eq_comm] at this
  /-
    case intro.intro.inr.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T : Set α
    hT : HasSubset.Subset T S
    ha : IsAtom (SupSet.sSup T)
    haS : LE.le (SupSet.sSup T) (SupSet.sSup S)
    a : α
    haT : Membership.mem T a
    this : LE.le a (SupSet.sSup T)
    ⊢ Ne a Bot.bot
  -/
  exact (hS.isAtom a (hT haT)).1
  /-
    🎉 no goals
  -/


lemma sSup_inter (hS : BooleanGenerators S) {T₁ T₂ : Set α} (hT₁ : T₁ ⊆ S) (hT₂ : T₂ ⊆ S) :
    sSup (T₁ ∩ T₂) = (sSup T₁) ⊓ (sSup T₂) := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    ⊢ Eq (SupSet.sSup (Inter.inter T₁ T₂)) (Min.min (SupSet.sSup T₁) (SupSet.sSup  …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      T₁ T₂ : Set α
      hT₁ : HasSubset.Subset T₁ S
      hT₂ : HasSubset.Subset T₂ S
      ⊢ LE.le (SupSet.sSup (Inter.inter T₁ T₂)) (Min.min (SupSet.sSup T₁) (SupSet.sS …
    -/
  · apply le_inf
      /-
        case a.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        ⊢ LE.le (SupSet.sSup (Inter.inter T₁ T₂)) (SupSet.sSup T₁)
      -/
    · apply sSup_le_sSup Set.inter_subset_left
      /-
        🎉 no goals
      -/
      /-
        case a.a
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        ⊢ LE.le (SupSet.sSup (Inter.inter T₁ T₂)) (SupSet.sSup T₂)
      -/
    · apply sSup_le_sSup Set.inter_subset_right
      /-
        🎉 no goals
      -/
  /-
    case a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    ⊢ LE.le (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup (Inter.inter  …
  -/
  obtain ⟨X, hX, hX'⟩ := hS.atomistic (sSup T₁ ⊓ sSup T₂) (inf_le_left.trans (sSup_le_sSup hT₁))
  /-
    case a.intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    X : Set α
    hX : HasSubset.Subset X S
    hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
    ⊢ LE.le (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup (Inter.inter  …
  -/
  rw [hX']
  /-
    case a.intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    X : Set α
    hX : HasSubset.Subset X S
    hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
    ⊢ LE.le (SupSet.sSup X) (SupSet.sSup (Inter.inter T₁ T₂))
  -/
  apply _root_.sSup_le
  /-
    case a.intro.intro.a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    X : Set α
    hX : HasSubset.Subset X S
    hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
    ⊢ ∀ (b : α), Membership.mem X b → LE.le b (SupSet.sSup (Inter.inter T₁ T₂))
  -/
  intro I hI
  /-
    case a.intro.intro.a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    X : Set α
    hX : HasSubset.Subset X S
    hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
    I : α
    hI : Membership.mem X I
    ⊢ LE.le I (SupSet.sSup (Inter.inter T₁ T₂))
  -/
  apply _root_.le_sSup
  /-
    case a.intro.intro.a.a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    T₁ T₂ : Set α
    hT₁ : HasSubset.Subset T₁ S
    hT₂ : HasSubset.Subset T₂ S
    X : Set α
    hX : HasSubset.Subset X S
    hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
    I : α
    hI : Membership.mem X I
    ⊢ Membership.mem (Inter.inter T₁ T₂) I
  -/
  constructor
    /-
      case a.intro.intro.a.a.left
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      T₁ T₂ : Set α
      hT₁ : HasSubset.Subset T₁ S
      hT₂ : HasSubset.Subset T₂ S
      X : Set α
      hX : HasSubset.Subset X S
      hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
      I : α
      hI : Membership.mem X I
      ⊢ Membership.mem T₁ I
    -/
  · apply (hS.mono hT₁).mem_of_isAtom_of_le_sSup_atoms _ _ _
      /-
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        X : Set α
        hX : HasSubset.Subset X S
        hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
        I : α
        hI : Membership.mem X I
        ⊢ IsAtom I
      -/
    · exact (hS.mono hX).isAtom I hI
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        X : Set α
        hX : HasSubset.Subset X S
        hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
        I : α
        hI : Membership.mem X I
        ⊢ LE.le I (SupSet.sSup T₁)
      -/
    · exact (_root_.le_sSup hI).trans (hX'.ge.trans inf_le_left)
      /-
        🎉 no goals
      -/
    /-
      case a.intro.intro.a.a.right
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      T₁ T₂ : Set α
      hT₁ : HasSubset.Subset T₁ S
      hT₂ : HasSubset.Subset T₂ S
      X : Set α
      hX : HasSubset.Subset X S
      hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
      I : α
      hI : Membership.mem X I
      ⊢ Membership.mem T₂ I
    -/
  · apply (hS.mono hT₂).mem_of_isAtom_of_le_sSup_atoms _ _ _
      /-
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        X : Set α
        hX : HasSubset.Subset X S
        hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
        I : α
        hI : Membership.mem X I
        ⊢ IsAtom I
      -/
    · exact (hS.mono hX).isAtom I hI
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝¹ : CompleteLattice α
        S : Set α
        inst✝ : IsCompactlyGenerated α
        hS : IsCompactlyGenerated.BooleanGenerators S
        T₁ T₂ : Set α
        hT₁ : HasSubset.Subset T₁ S
        hT₂ : HasSubset.Subset T₂ S
        X : Set α
        hX : HasSubset.Subset X S
        hX' : Eq (Min.min (SupSet.sSup T₁) (SupSet.sSup T₂)) (SupSet.sSup X)
        I : α
        hI : Membership.mem X I
        ⊢ LE.le I (SupSet.sSup T₂)
      -/
    · exact (_root_.le_sSup hI).trans (hX'.ge.trans inf_le_right)
      /-
        🎉 no goals
      -/


/-- A lattice generated by boolean generators is a distributive lattice. -/
def distribLattice_of_sSup_eq_top (hS : BooleanGenerators S) (h : sSup S = ⊤) :
    DistribLattice α where
  le_sup_inf a b c := by
    /-
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      a b c : α
      ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
    -/
    obtain ⟨Ta, hTa, rfl⟩ := hS.atomistic a (h ▸ le_top)
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      b c : α
      Ta : Set α
      hTa : HasSubset.Subset Ta S
      ⊢ LE.le (Min.min (Max.max (SupSet.sSup Ta) b) (Max.max (SupSet.sSup Ta) c)) (M …
    -/
    obtain ⟨Tb, hTb, rfl⟩ := hS.atomistic b (h ▸ le_top)
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      c : α
      Ta : Set α
      hTa : HasSubset.Subset Ta S
      Tb : Set α
      hTb : HasSubset.Subset Tb S
      ⊢ LE.le (Min.min (Max.max (SupSet.sSup Ta) (SupSet.sSup Tb)) (Max.max (SupSet. …
    -/
    obtain ⟨Tc, hTc, rfl⟩ := hS.atomistic c (h ▸ le_top)
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      Ta : Set α
      hTa : HasSubset.Subset Ta S
      Tb : Set α
      hTb : HasSubset.Subset Tb S
      Tc : Set α
      hTc : HasSubset.Subset Tc S
      ⊢ LE.le (Min.min (Max.max (SupSet.sSup Ta) (SupSet.sSup Tb)) (Max.max (SupSet. …
    -/
    apply le_of_eq
    /-
      case intro.intro.intro.intro.intro.intro.hab
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      Ta : Set α
      hTa : HasSubset.Subset Ta S
      Tb : Set α
      hTb : HasSubset.Subset Tb S
      Tc : Set α
      hTc : HasSubset.Subset Tc S
      ⊢ Eq (Min.min (Max.max (SupSet.sSup Ta) (SupSet.sSup Tb)) (Max.max (SupSet.sSu …
    -/
    rw [← sSup_union, ← sSup_union, ← hS.sSup_inter hTb hTc, ← hS.sSup_inter, ← sSup_union]
    /-
      case intro.intro.intro.intro.intro.intro.hab
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      Ta : Set α
      hTa : HasSubset.Subset Ta S
      Tb : Set α
      hTb : HasSubset.Subset Tb S
      Tc : Set α
      hTc : HasSubset.Subset Tc S
      ⊢ Eq (SupSet.sSup (Inter.inter (Union.union Ta Tb) (Union.union Ta Tc))) (SupS …
    -/
    on_goal 1 => congr 1; ext
    all_goals
      simp only [Set.union_subset_iff, Set.mem_inter_iff, Set.mem_union]
      tauto


lemma complementedLattice_of_sSup_eq_top (hS : BooleanGenerators S) (h : sSup S = ⊤) :
    ComplementedLattice α := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    ⊢ ComplementedLattice α
  -/
  let _i := hS.distribLattice_of_sSup_eq_top h
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    _i : DistribLattice α := hS.distribLattice_of_sSup_eq_top h
    ⊢ ComplementedLattice α
  -/
  have _i₁ := isAtomistic_of_sSup_eq_top hS h
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    _i : DistribLattice α := hS.distribLattice_of_sSup_eq_top h
    _i₁ : IsAtomistic α
    ⊢ ComplementedLattice α
  -/
  apply complementedLattice_of_isAtomistic
  /-
    🎉 no goals
  -/


/-- A compactly generated complete lattice generated by boolean generators is a boolean algebra. -/
noncomputable
def booleanAlgebra_of_sSup_eq_top (hS : BooleanGenerators S) (h : sSup S = ⊤) : BooleanAlgebra α :=
  let _i := hS.distribLattice_of_sSup_eq_top h
  have := hS.complementedLattice_of_sSup_eq_top h
  DistribLattice.booleanAlgebraOfComplemented α


lemma sSup_le_sSup_iff_of_atoms (hS : BooleanGenerators S) (X Y : Set α) (hX : X ⊆ S) (hY : Y ⊆ S) :
    sSup X ≤ sSup Y ↔ X ⊆ Y := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    X Y : Set α
    hX : HasSubset.Subset X S
    hY : HasSubset.Subset Y S
    ⊢ Iff (LE.le (SupSet.sSup X) (SupSet.sSup Y)) (HasSubset.Subset X Y)
  -/
  refine ⟨?_, sSup_le_sSup⟩
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    X Y : Set α
    hX : HasSubset.Subset X S
    hY : HasSubset.Subset Y S
    ⊢ LE.le (SupSet.sSup X) (SupSet.sSup Y) → HasSubset.Subset X Y
  -/
  intro h a ha
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    X Y : Set α
    hX : HasSubset.Subset X S
    hY : HasSubset.Subset Y S
    h : LE.le (SupSet.sSup X) (SupSet.sSup Y)
    a : α
    ha : Membership.mem X a
    ⊢ Membership.mem Y a
  -/
  apply (hS.mono hY).mem_of_isAtom_of_le_sSup_atoms _ _ ((le_sSup ha).trans h)
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    X Y : Set α
    hX : HasSubset.Subset X S
    hY : HasSubset.Subset Y S
    h : LE.le (SupSet.sSup X) (SupSet.sSup Y)
    a : α
    ha : Membership.mem X a
    ⊢ IsAtom a
  -/
  exact (hS.mono hX).isAtom a ha
  /-
    🎉 no goals
  -/


lemma eq_atoms_of_sSup_eq_top (hS : BooleanGenerators S) (h : sSup S = ⊤) :
    S = {a : α | IsAtom a} := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    ⊢ Eq S (setOf fun a => IsAtom a)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : CompleteLattice α
      S : Set α
      inst✝ : IsCompactlyGenerated α
      hS : IsCompactlyGenerated.BooleanGenerators S
      h : Eq (SupSet.sSup S) Top.top
      ⊢ LE.le S (setOf fun a => IsAtom a)
    -/
  · exact hS.isAtom
    /-
      🎉 no goals
    -/
  /-
    case a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    ⊢ LE.le (setOf fun a => IsAtom a) S
  -/
  intro a ha
  /-
    case a
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    ⊢ Membership.mem S a
  -/
  obtain ⟨T, hT, rfl⟩ := hS.atomistic a (le_top.trans h.ge)
  /-
    case a.intro.intro
    α : Type u_1
    inst✝¹ : CompleteLattice α
    S : Set α
    inst✝ : IsCompactlyGenerated α
    hS : IsCompactlyGenerated.BooleanGenerators S
    h : Eq (SupSet.sSup S) Top.top
    T : Set α
    hT : HasSubset.Subset T S
    ha : Membership.mem (setOf fun a => IsAtom a) (SupSet.sSup T)
    ⊢ Membership.mem S (SupSet.sSup T)
  -/
  exact hS.mem_of_isAtom_of_le_sSup_atoms _ ha (sSup_le_sSup hT)
  /-
    🎉 no goals
  -/


