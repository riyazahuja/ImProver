/-- Constructs the `⊥` of a finite nonempty `SemilatticeInf`. -/
abbrev toOrderBot [SemilatticeInf α] : OrderBot α where
  bot := univ.inf' univ_nonempty id
  bot_le a := inf'_le _ <| mem_univ a

-- See note [reducible non-instances]

/-- Constructs the `⊤` of a finite nonempty `SemilatticeSup` -/
abbrev toOrderTop [SemilatticeSup α] : OrderTop α where
  top := univ.sup' univ_nonempty id
  -- Porting note: needed to make `id` explicit
  le_top a := le_sup' id <| mem_univ a

-- See note [reducible non-instances]

/-- Constructs the `⊤` and `⊥` of a finite nonempty `Lattice`. -/
abbrev toBoundedOrder [Lattice α] : BoundedOrder α :=
  { toOrderBot α, toOrderTop α with }


open scoped Classical in
-- See note [reducible non-instances]
/-- A finite bounded lattice is complete. -/
noncomputable abbrev toCompleteLattice [Lattice α] [BoundedOrder α] : CompleteLattice α where
  __ := ‹Lattice α›
  __ := ‹BoundedOrder α›
  sSup := fun s => s.toFinset.sup id
  sInf := fun s => s.toFinset.inf id
  le_sSup := fun _ _ ha => Finset.le_sup (f := id) (Set.mem_toFinset.mpr ha)
  sSup_le := fun _ _ ha => Finset.sup_le fun _ hb => ha _ <| Set.mem_toFinset.mp hb
  sInf_le := fun _ _ ha => Finset.inf_le (Set.mem_toFinset.mpr ha)
  le_sInf := fun _ _ ha => Finset.le_inf fun _ hb => ha _ <| Set.mem_toFinset.mp hb

-- See note [reducible non-instances]

/-- A finite bounded distributive lattice is completely distributive. -/
noncomputable abbrev toCompleteDistribLatticeMinimalAxioms [DistribLattice α] [BoundedOrder α] :
    CompleteDistribLattice.MinimalAxioms α where
  __ := toCompleteLattice α
  iInf_sup_le_sup_sInf := fun a s => by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    convert (Finset.inf_sup_distrib_left s.toFinset id a).ge using 1
    /-
      case h.e'_3
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iInf fun b => iInf fun h => Max.max a b) (s.toFinset.inf fun i => Max.ma …
    -/
    rw [Finset.inf_eq_iInf]
    /-
      case h.e'_3
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iInf fun b => iInf fun h => Max.max a b) (iInf fun a_1 => iInf fun h =>  …
    -/
    /-
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    simp_rw [Set.mem_toFinset]
    /-
      case h.e'_4
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iSup fun b => iSup fun h => Min.min a b) (s.toFinset.sup fun i => Min.mi …
    -/
    /-
      case h.e'_3
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iInf fun b => iInf fun x => Max.max a b) (iInf fun a_1 => iInf fun x =>  …
    -/
    /-
      case h.e'_4
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iSup fun b => iSup fun h => Min.min a b) (iSup fun a_1 => iSup fun h =>  …
    -/
    rfl
    /-
      case h.e'_4
      ι : Type u_1
      α : Type u_2
      inst✝³ : Fintype ι
      inst✝² : Fintype α
      inst✝¹ : DistribLattice α
      inst✝ : BoundedOrder α
      a : α
      s : Set α
      ⊢ Eq (iSup fun b => iSup fun x => Min.min a b) (iSup fun a_1 => iSup fun x =>  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  inf_sSup_le_iSup_inf := fun a s => by
    convert (Finset.sup_inf_distrib_left s.toFinset id a).le using 1
    rw [Finset.sup_eq_iSup]
    simp_rw [Set.mem_toFinset]
    rfl

-- See note [reducible non-instances]

/-- A finite bounded distributive lattice is completely distributive. -/
noncomputable abbrev toCompleteDistribLattice [DistribLattice α] [BoundedOrder α] :
    CompleteDistribLattice α := .ofMinimalAxioms (toCompleteDistribLatticeMinimalAxioms _)

-- See note [reducible non-instances]

/-- A finite bounded linear order is complete. -/
noncomputable abbrev toCompleteLinearOrder
    [LinearOrder α] [BoundedOrder α] : CompleteLinearOrder α :=
  { toCompleteLattice α, ‹LinearOrder α›, LinearOrder.toBiheytingAlgebra with }

-- See note [reducible non-instances]

/-- A finite boolean algebra is complete. -/
noncomputable abbrev toCompleteBooleanAlgebra [BooleanAlgebra α] : CompleteBooleanAlgebra α where
  __ := ‹BooleanAlgebra α›
  __ := Fintype.toCompleteDistribLattice α

-- See note [reducible non-instances]

/-- A finite boolean algebra is complete and atomic. -/
noncomputable abbrev toCompleteAtomicBooleanAlgebra [BooleanAlgebra α] :
    CompleteAtomicBooleanAlgebra α :=
  (toCompleteBooleanAlgebra α).toCompleteAtomicBooleanAlgebra


/-- A nonempty finite lattice is complete. If the lattice is already a `BoundedOrder`, then use
`Fintype.toCompleteLattice` instead, as this gives definitional equality for `⊥` and `⊤`. -/
noncomputable abbrev toCompleteLatticeOfNonempty [Lattice α] : CompleteLattice α :=
  @toCompleteLattice _ _ _ <| @toBoundedOrder α _ ⟨Classical.arbitrary α⟩ _

-- See note [reducible non-instances]

/-- A nonempty finite linear order is complete. If the linear order is already a `BoundedOrder`,
then use `Fintype.toCompleteLinearOrder` instead, as this gives definitional equality for `⊥` and
`⊤`. -/
noncomputable abbrev toCompleteLinearOrderOfNonempty [LinearOrder α] : CompleteLinearOrder α := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : Fintype ι
    inst✝² : Fintype α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder α
    ⊢ CompleteLinearOrder α
  -/
  let _ := toBoundedOrder α
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : Fintype ι
    inst✝² : Fintype α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder α
    x✝ : BoundedOrder α := Fintype.toBoundedOrder α
    ⊢ CompleteLinearOrder α
  -/
  exact { toCompleteLatticeOfNonempty α, ‹LinearOrder α›, LinearOrder.toBiheytingAlgebra with }
  /-
    🎉 no goals
  -/


lemma Finite.exists_minimal_le [Finite α] (h : p a) : ∃ b, b ≤ a ∧ Minimal p b := by
  obtain ⟨b, ⟨hba, hb⟩, hbmin⟩ :=
    Set.Finite.exists_minimal_wrt id {x | x ≤ a ∧ p x} (Set.toFinite _) ⟨a, rfl.le, h⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    p : α → Prop
    inst✝ : Finite α
    h : p a
    b : α
    hbmin : ∀ (a' : α), Membership.mem (setOf fun x => And (LE.le x a) (p x)) a' → …
    hba : LE.le b a
    hb : p b
    ⊢ Exists fun b => And (LE.le b a) (Minimal p b)
  -/
  exact ⟨b, hba, hb, fun x hx hxb ↦ (hbmin x ⟨hxb.trans hba, hx⟩ hxb).le⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-23")] alias Finite.exists_ge_minimal := Finite.exists_minimal_le


lemma Finite.exists_le_maximal [Finite α] (h : p a) : ∃ b, a ≤ b ∧ Maximal p b :=
  Finite.exists_minimal_le (α := αᵒᵈ) h


lemma Finset.exists_minimal_le (s : Finset α) (h : a ∈ s) : ∃ b, b ≤ a ∧ Minimal (· ∈ s) b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s : Finset α
    h : Membership.mem s a
    ⊢ Exists fun b => And (LE.le b a) (Minimal (fun x => Membership.mem s x) b)
  -/
  obtain ⟨⟨b, _⟩, lb, minb⟩ := @Finite.exists_minimal_le s _ ⟨a, h⟩ (·.1 ∈ s) _ h
  /-
    case intro.mk.intro
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s : Finset α
    h : Membership.mem s a
    b : α
    property✝ : Membership.mem s b
    lb : LE.le ⟨b, property✝⟩ ⟨a, h⟩
    minb : Minimal (fun x => Membership.mem s ↑x) ⟨b, property✝⟩
    ⊢ Exists fun b => And (LE.le b a) (Minimal (fun x => Membership.mem s x) b)
  -/
  use b, lb; rwa [minimal_subtype, inf_idem] at minb
             /-
               🎉 no goals
             -/


lemma Finset.exists_le_maximal (s : Finset α) (h : a ∈ s) : ∃ b, a ≤ b ∧ Maximal (· ∈ s) b :=
  s.exists_minimal_le (α := αᵒᵈ) h


lemma Set.Finite.exists_minimal_le {s : Set α} (hs : s.Finite) (h : a ∈ s) :
    ∃ b, b ≤ a ∧ Minimal (· ∈ s) b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s : Set α
    hs : s.Finite
    h : Membership.mem s a
    ⊢ Exists fun b => And (LE.le b a) (Minimal (fun x => Membership.mem s x) b)
  -/
  obtain ⟨b, lb, minb⟩ := hs.toFinset.exists_minimal_le (hs.mem_toFinset.mpr h)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s : Set α
    hs : s.Finite
    h : Membership.mem s a
    b : α
    lb : LE.le b a
    minb : Minimal (fun x => Membership.mem hs.toFinset x) b
    ⊢ Exists fun b => And (LE.le b a) (Minimal (fun x => Membership.mem s x) b)
  -/
  use b, lb; simpa using minb
             /-
               🎉 no goals
             -/


lemma Set.Finite.exists_le_maximal {s : Set α} (hs : s.Finite) (h : a ∈ s) :
    ∃ b, a ≤ b ∧ Maximal (· ∈ s) b :=
  hs.exists_minimal_le (α := αᵒᵈ) h


noncomputable instance Fin.completeLinearOrder {n : ℕ} [NeZero n] : CompleteLinearOrder (Fin n) :=
  Fintype.toCompleteLinearOrder _


noncomputable instance Bool.completeLinearOrder : CompleteLinearOrder Bool :=
  Fintype.toCompleteLinearOrder _


noncomputable instance Bool.completeBooleanAlgebra : CompleteBooleanAlgebra Bool :=
  Fintype.toCompleteBooleanAlgebra _


noncomputable instance Bool.completeAtomicBooleanAlgebra : CompleteAtomicBooleanAlgebra Bool :=
  Fintype.toCompleteAtomicBooleanAlgebra _


theorem Directed.finite_set_le (D : Directed r f) {s : Set γ} (hs : s.Finite) :
    ∃ z, ∀ i ∈ s, r (f i) (f z) := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : IsTrans α r
    γ : Type u_3
    inst✝ : Nonempty γ
    f : γ → α
    D : Directed r f
    s : Set γ
    hs : s.Finite
    ⊢ Exists fun z => ∀ (i : γ), Membership.mem s i → r (f i) (f z)
  -/
  convert D.finset_le hs.toFinset using 3; rw [Set.Finite.mem_toFinset]
                                           /-
                                             🎉 no goals
                                           -/


theorem Directed.finite_le (D : Directed r f) (g : β → γ) : ∃ z, ∀ i, r (f (g i)) (f z) := by
  classical
    obtain ⟨z, hz⟩ := D.finite_set_le (Set.finite_range g)
    exact ⟨z, fun i => hz (g i) ⟨i, rfl⟩⟩


theorem Finite.exists_le [IsDirected α (· ≤ ·)] (f : β → α) : ∃ M, ∀ i, f i ≤ M :=
  directed_id.finite_le _


theorem Finite.exists_ge [IsDirected α (· ≥ ·)] (f : β → α) : ∃ M, ∀ i, M ≤ f i :=
  directed_id.finite_le (r := (· ≥ ·)) _


theorem Set.Finite.exists_le [IsDirected α (· ≤ ·)] {s : Set α} (hs : s.Finite) :
    ∃ M, ∀ i ∈ s, i ≤ M :=
  directed_id.finite_set_le hs


theorem Set.Finite.exists_ge [IsDirected α (· ≥ ·)] {s : Set α} (hs : s.Finite) :
    ∃ M, ∀ i ∈ s, M ≤ i :=
  directed_id.finite_set_le (r := (· ≥ ·)) hs


@[simp]
theorem Finite.bddAbove_range [IsDirected α (· ≤ ·)] (f : β → α) : BddAbove (Set.range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    f : β → α
    ⊢ BddAbove (Set.range f)
  -/
  obtain ⟨M, hM⟩ := Finite.exists_le f
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le (f i) M
    ⊢ BddAbove (Set.range f)
  -/
  refine ⟨M, fun a ha => ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le (f i) M
    a : α
    ha : Membership.mem (Set.range f) a
    ⊢ LE.le a M
  -/
  obtain ⟨b, rfl⟩ := ha
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le (f i) M
    b : β
    ⊢ LE.le (f b) M
  -/
  exact hM b
  /-
    🎉 no goals
  -/


@[simp]
theorem Finite.bddBelow_range [IsDirected α (· ≥ ·)] (f : β → α) : BddBelow (Set.range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => GE.ge x1 x2
    f : β → α
    ⊢ BddBelow (Set.range f)
  -/
  obtain ⟨M, hM⟩ := Finite.exists_ge f
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => GE.ge x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le M (f i)
    ⊢ BddBelow (Set.range f)
  -/
  refine ⟨M, fun a ha => ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => GE.ge x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le M (f i)
    a : α
    ha : Membership.mem (Set.range f) a
    ⊢ LE.le M a
  -/
  obtain ⟨b, rfl⟩ := ha
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Finite β
    inst✝² : Nonempty α
    inst✝¹ : Preorder α
    inst✝ : IsDirected α fun x1 x2 => GE.ge x1 x2
    f : β → α
    M : α
    hM : ∀ (i : β), LE.le M (f i)
    b : β
    ⊢ LE.le M (f b)
  -/
  exact hM b
  /-
    🎉 no goals
  -/

