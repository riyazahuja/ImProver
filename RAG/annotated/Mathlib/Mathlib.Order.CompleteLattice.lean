@[simp] lemma iSup_ulift {ι : Type*} [SupSet α] (f : ULift ι → α) :
                                              /-
                                                α : Type u_1
                                                ι : Type u_8
                                                inst✝ : SupSet α
                                                f : ULift.{u_9, u_8} ι → α
                                                ⊢ Eq (iSup fun i => f i) (iSup fun i => f { down := i })
                                              -/
    ⨆ i : ULift ι, f i = ⨆ i, f (.up i) := by simp [iSup]; congr with x; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma iInf_ulift {ι : Type*} [InfSet α] (f : ULift ι → α) :
                                              /-
                                                α : Type u_1
                                                ι : Type u_8
                                                inst✝ : InfSet α
                                                f : ULift.{u_9, u_8} ι → α
                                                ⊢ Eq (iInf fun i => f i) (iInf fun i => f { down := i })
                                              -/
    ⨅ i : ULift ι, f i = ⨅ i, f (.up i) := by simp [iInf]; congr with x; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance OrderDual.supSet (α) [InfSet α] : SupSet αᵒᵈ :=
  ⟨(sInf : Set α → α)⟩


instance OrderDual.infSet (α) [SupSet α] : InfSet αᵒᵈ :=
  ⟨(sSup : Set α → α)⟩


/-- Note that we rarely use `CompleteSemilatticeSup`
(in fact, any such object is always a `CompleteLattice`, so it's usually best to start there).

Nevertheless it is sometimes a useful intermediate step in constructions.
-/
class CompleteSemilatticeSup (α : Type*) extends PartialOrder α, SupSet α where
  /-- Any element of a set is less than the set supremum. -/
  le_sSup : ∀ s, ∀ a ∈ s, a ≤ sSup s
  /-- Any upper bound is more than the set supremum. -/
  sSup_le : ∀ s a, (∀ b ∈ s, b ≤ a) → sSup s ≤ a


theorem le_sSup : a ∈ s → a ≤ sSup s :=
  CompleteSemilatticeSup.le_sSup s a


theorem sSup_le : (∀ b ∈ s, b ≤ a) → sSup s ≤ a :=
  CompleteSemilatticeSup.sSup_le s a


theorem isLUB_sSup (s : Set α) : IsLUB s (sSup s) :=
  ⟨fun _ ↦ le_sSup, fun _ ↦ sSup_le⟩


lemma isLUB_iff_sSup_eq : IsLUB s a ↔ sSup s = a :=
                             /-
                               α : Type u_1
                               inst✝ : CompleteSemilatticeSup α
                               s : Set α
                               a : α
                               ⊢ Eq (SupSet.sSup s) a → IsLUB s a
                             -/
  ⟨(isLUB_sSup s).unique, by rintro rfl; exact isLUB_sSup _⟩
                                         /-
                                           🎉 no goals
                                         -/


alias ⟨IsLUB.sSup_eq, _⟩ := isLUB_iff_sSup_eq


theorem le_sSup_of_le (hb : b ∈ s) (h : a ≤ b) : a ≤ sSup s :=
  le_trans h (le_sSup hb)


@[gcongr]
theorem sSup_le_sSup (h : s ⊆ t) : sSup s ≤ sSup t :=
  (isLUB_sSup s).mono (isLUB_sSup t) h


@[simp]
theorem sSup_le_iff : sSup s ≤ a ↔ ∀ b ∈ s, b ≤ a :=
  isLUB_le_iff (isLUB_sSup s)


theorem le_sSup_iff : a ≤ sSup s ↔ ∀ b ∈ upperBounds s, a ≤ b :=
  ⟨fun h _ hb => le_trans h (sSup_le hb), fun hb => hb _ fun _ => le_sSup⟩


theorem le_iSup_iff {s : ι → α} : a ≤ iSup s ↔ ∀ b, (∀ i, s i ≤ b) → a ≤ b := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteSemilatticeSup α
    a : α
    s : ι → α
    ⊢ Iff (LE.le a (iSup s)) (∀ (b : α), (∀ (i : ι), LE.le (s i) b) → LE.le a b)
  -/
  simp [iSup, le_sSup_iff, upperBounds]
  /-
    🎉 no goals
  -/


theorem sSup_le_sSup_of_forall_exists_le (h : ∀ x ∈ s, ∃ y ∈ t, x ≤ y) : sSup s ≤ sSup t :=
  le_sSup_iff.2 fun _ hb =>
    sSup_le fun a ha =>
      let ⟨_, hct, hac⟩ := h a ha
      hac.trans (hb hct)

-- We will generalize this to conditionally complete lattices in `csSup_singleton`.

theorem sSup_singleton {a : α} : sSup {a} = a :=
  isLUB_singleton.sSup_eq


/-- Note that we rarely use `CompleteSemilatticeInf`
(in fact, any such object is always a `CompleteLattice`, so it's usually best to start there).

Nevertheless it is sometimes a useful intermediate step in constructions.
-/
class CompleteSemilatticeInf (α : Type*) extends PartialOrder α, InfSet α where
  /-- Any element of a set is more than the set infimum. -/
  sInf_le : ∀ s, ∀ a ∈ s, sInf s ≤ a
  /-- Any lower bound is less than the set infimum. -/
  le_sInf : ∀ s a, (∀ b ∈ s, a ≤ b) → a ≤ sInf s


theorem sInf_le : a ∈ s → sInf s ≤ a :=
  CompleteSemilatticeInf.sInf_le s a


theorem le_sInf : (∀ b ∈ s, a ≤ b) → a ≤ sInf s :=
  CompleteSemilatticeInf.le_sInf s a


theorem isGLB_sInf (s : Set α) : IsGLB s (sInf s) :=
  ⟨fun _ => sInf_le, fun _ => le_sInf⟩


lemma isGLB_iff_sInf_eq : IsGLB s a ↔ sInf s = a :=
                             /-
                               α : Type u_1
                               inst✝ : CompleteSemilatticeInf α
                               s : Set α
                               a : α
                               ⊢ Eq (InfSet.sInf s) a → IsGLB s a
                             -/
  ⟨(isGLB_sInf s).unique, by rintro rfl; exact isGLB_sInf _⟩
                                         /-
                                           🎉 no goals
                                         -/


alias ⟨IsGLB.sInf_eq, _⟩ := isGLB_iff_sInf_eq


theorem sInf_le_of_le (hb : b ∈ s) (h : b ≤ a) : sInf s ≤ a :=
  le_trans (sInf_le hb) h


@[gcongr]
theorem sInf_le_sInf (h : s ⊆ t) : sInf t ≤ sInf s :=
  (isGLB_sInf s).mono (isGLB_sInf t) h


@[simp]
theorem le_sInf_iff : a ≤ sInf s ↔ ∀ b ∈ s, a ≤ b :=
  le_isGLB_iff (isGLB_sInf s)


theorem sInf_le_iff : sInf s ≤ a ↔ ∀ b ∈ lowerBounds s, b ≤ a :=
  ⟨fun h _ hb => le_trans (le_sInf hb) h, fun hb => hb _ fun _ => sInf_le⟩


theorem iInf_le_iff {s : ι → α} : iInf s ≤ a ↔ ∀ b, (∀ i, b ≤ s i) → b ≤ a := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteSemilatticeInf α
    a : α
    s : ι → α
    ⊢ Iff (LE.le (iInf s) a) (∀ (b : α), (∀ (i : ι), LE.le b (s i)) → LE.le b a)
  -/
  simp [iInf, sInf_le_iff, lowerBounds]
  /-
    🎉 no goals
  -/


theorem sInf_le_sInf_of_forall_exists_le (h : ∀ x ∈ s, ∃ y ∈ t, y ≤ x) : sInf t ≤ sInf s :=
  le_sInf fun x hx ↦ let ⟨_y, hyt, hyx⟩ := h x hx; sInf_le_of_le hyt hyx

-- We will generalize this to conditionally complete lattices in `csInf_singleton`.

theorem sInf_singleton {a : α} : sInf {a} = a :=
  isGLB_singleton.sInf_eq


instance {α : Type*} [CompleteSemilatticeInf α] : CompleteSemilatticeSup αᵒᵈ where
  le_sSup := @CompleteSemilatticeInf.sInf_le α _
  sSup_le := @CompleteSemilatticeInf.le_sInf α _


instance {α : Type*} [CompleteSemilatticeSup α] : CompleteSemilatticeInf αᵒᵈ where
  le_sInf := @CompleteSemilatticeSup.sSup_le α _
  sInf_le := @CompleteSemilatticeSup.le_sSup α _



/-- A complete lattice is a bounded lattice which has suprema and infima for every subset. -/
class CompleteLattice (α : Type*) extends Lattice α, CompleteSemilatticeSup α,
  CompleteSemilatticeInf α, Top α, Bot α where
  /-- Any element is less than the top one. -/
  protected le_top : ∀ x : α, x ≤ ⊤
  /-- Any element is more than the bottom one. -/
  protected bot_le : ∀ x : α, ⊥ ≤ x

-- see Note [lower instance priority]

instance (priority := 100) CompleteLattice.toBoundedOrder [h : CompleteLattice α] :
    BoundedOrder α :=
  { h with }


/-- Create a `CompleteLattice` from a `PartialOrder` and `InfSet`
that returns the greatest lower bound of a set. Usually this constructor provides
poor definitional equalities.  If other fields are known explicitly, they should be
provided; for example, if `inf` is known explicitly, construct the `CompleteLattice`
instance as
```
instance : CompleteLattice my_T where
  inf := better_inf
  le_inf := ...
  inf_le_right := ...
  inf_le_left := ...
  -- don't care to fix sup, sSup, bot, top
  __ := completeLatticeOfInf my_T _
```
-/
def completeLatticeOfInf (α : Type*) [H1 : PartialOrder α] [H2 : InfSet α]
    (isGLB_sInf : ∀ s : Set α, IsGLB s (sInf s)) : CompleteLattice α where
  __ := H1; __ := H2
  bot := sInf univ
  bot_le _ := (isGLB_sInf univ).1 trivial
  top := sInf ∅
                                     /-
                                       α✝ : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       ι : Sort u_4
                                       ι' : Sort u_5
                                       κ : ι → Sort u_6
                                       κ' : ι' → Sort u_7
                                       α : Type u_8
                                       H1 : PartialOrder α
                                       H2 : InfSet α
                                       isGLB_sInf : ∀ (s : Set α), IsGLB s (InfSet.sInf s)
                                       a : α
                                       ⊢ Membership.mem (lowerBounds EmptyCollection.emptyCollection) a
                                     -/
  le_top a := (isGLB_sInf ∅).2 <| by simp
                                     /-
                                       🎉 no goals
                                     -/
  sup a b := sInf { x : α | a ≤ x ∧ b ≤ x }
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      κ : ι → Sort u_6
      κ' : ι' → Sort u_7
      α : Type u_8
      H1 : PartialOrder α
      H2 : InfSet α
      isGLB_sInf : ∀ (s : Set α), IsGLB s (InfSet.sInf s)
      a b c : α
      hab : LE.le a b
      hac : LE.le a c
      ⊢ LE.le a ((fun a b => InfSet.sInf (Insert.insert a (Singleton.singleton b)))  …
    -/
  inf a b := sInf {a, b}
    /-
      case a
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      κ : ι → Sort u_6
      κ' : ι' → Sort u_7
      α : Type u_8
      H1 : PartialOrder α
      H2 : InfSet α
      isGLB_sInf : ∀ (s : Set α), IsGLB s (InfSet.sInf s)
      a b c : α
      hab : LE.le a b
      hac : LE.le a c
      ⊢ Membership.mem (lowerBounds (Insert.insert b (Singleton.singleton c))) a
    -/
  le_inf a b c hab hac := by
                                                 /-
                                                   α✝ : Type u_1
                                                   β : Type u_2
                                                   γ : Type u_3
                                                   ι : Sort u_4
                                                   ι' : Sort u_5
                                                   κ : ι → Sort u_6
                                                   κ' : ι' → Sort u_7
                                                   α : Type u_8
                                                   H1 : PartialOrder α
                                                   H2 : InfSet α
                                                   isGLB_sInf : ∀ (s : Set α), IsGLB s (InfSet.sInf s)
                                                   a b c : α
                                                   hac : LE.le a c
                                                   hbc : LE.le b c
                                                   ⊢ Membership.mem (setOf fun x => And (LE.le a x) (LE.le b x)) c
                                                 -/
    /-
      🎉 no goals
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    apply (isGLB_sInf _).2
    simp [*]
  inf_le_right _ _ := (isGLB_sInf _).1 <| mem_insert_of_mem _ <| mem_singleton _
  inf_le_left _ _ := (isGLB_sInf _).1 <| mem_insert _ _
  sup_le a b c hac hbc := (isGLB_sInf _).1 <| by simp [*]
  le_sup_left _ _ := (isGLB_sInf _).2 fun _ => And.left
  le_sup_right _ _ := (isGLB_sInf _).2 fun _ => And.right
  le_sInf s _ ha := (isGLB_sInf s).2 ha
  sInf_le s _ ha := (isGLB_sInf s).1 ha
  sSup s := sInf (upperBounds s)
  le_sSup s _ ha := (isGLB_sInf (upperBounds s)).2 fun _ hb => hb ha
  sSup_le s _ ha := (isGLB_sInf (upperBounds s)).1 ha


/-- Any `CompleteSemilatticeInf` is in fact a `CompleteLattice`.

Note that this construction has bad definitional properties:
see the doc-string on `completeLatticeOfInf`.
-/
def completeLatticeOfCompleteSemilatticeInf (α : Type*) [CompleteSemilatticeInf α] :
    CompleteLattice α :=
  completeLatticeOfInf α fun s => isGLB_sInf s


/-- Create a `CompleteLattice` from a `PartialOrder` and `SupSet`
that returns the least upper bound of a set. Usually this constructor provides
poor definitional equalities.  If other fields are known explicitly, they should be
provided; for example, if `inf` is known explicitly, construct the `CompleteLattice`
instance as
```
instance : CompleteLattice my_T where
  inf := better_inf
  le_inf := ...
  inf_le_right := ...
  inf_le_left := ...
  -- don't care to fix sup, sInf, bot, top
  __ := completeLatticeOfSup my_T _
```
-/
def completeLatticeOfSup (α : Type*) [H1 : PartialOrder α] [H2 : SupSet α]
    (isLUB_sSup : ∀ s : Set α, IsLUB s (sSup s)) : CompleteLattice α where
  __ := H1; __ := H2
  top := sSup univ
  le_top _ := (isLUB_sSup univ).1 trivial
  bot := sSup ∅
                                     /-
                                       α✝ : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       ι : Sort u_4
                                       ι' : Sort u_5
                                       κ : ι → Sort u_6
                                       κ' : ι' → Sort u_7
                                       α : Type u_8
                                       H1 : PartialOrder α
                                       H2 : SupSet α
                                       isLUB_sSup : ∀ (s : Set α), IsLUB s (SupSet.sSup s)
                                       x : α
                                       ⊢ Membership.mem (upperBounds EmptyCollection.emptyCollection) x
                                     -/
  bot_le x := (isLUB_sSup ∅).2 <| by simp
                                               /-
                                                 α✝ : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 ι : Sort u_4
                                                 ι' : Sort u_5
                                                 κ : ι → Sort u_6
                                                 κ' : ι' → Sort u_7
                                                 α : Type u_8
                                                 H1 : PartialOrder α
                                                 H2 : SupSet α
                                                 isLUB_sSup : ∀ (s : Set α), IsLUB s (SupSet.sSup s)
                                                 a b c : α
                                                 hac : LE.le a c
                                                 hbc : LE.le b c
                                                 ⊢ Membership.mem (upperBounds (Insert.insert a (Singleton.singleton b))) c
                                               -/
                                     /-
                                       🎉 no goals
                                     -/
                                               /-
                                                 🎉 no goals
                                               -/
  sup a b := sSup {a, b}
  sup_le a b c hac hbc := (isLUB_sSup _).2 (by simp [*])
  le_sup_left _ _ := (isLUB_sSup _).1 <| mem_insert _ _
                                                 /-
                                                   α✝ : Type u_1
                                                   β : Type u_2
                                                   γ : Type u_3
                                                   ι : Sort u_4
                                                   ι' : Sort u_5
                                                   κ : ι → Sort u_6
                                                   κ' : ι' → Sort u_7
                                                   α : Type u_8
                                                   H1 : PartialOrder α
                                                   H2 : SupSet α
                                                   isLUB_sSup : ∀ (s : Set α), IsLUB s (SupSet.sSup s)
                                                   a b c : α
                                                   hab : LE.le a b
                                                   hac : LE.le a c
                                                   ⊢ Membership.mem (setOf fun x => And (LE.le x b) (LE.le x c)) a
                                                 -/
  le_sup_right _ _ := (isLUB_sSup _).1 <| mem_insert_of_mem _ <| mem_singleton _
                                                 /-
                                                   🎉 no goals
                                                 -/
  inf a b := sSup { x | x ≤ a ∧ x ≤ b }
  le_inf a b c hab hac := (isLUB_sSup _).1 <| by simp [*]
  inf_le_left _ _ := (isLUB_sSup _).2 fun _ => And.left
  inf_le_right _ _ := (isLUB_sSup _).2 fun _ => And.right
  sInf s := sSup (lowerBounds s)
  sSup_le s _ ha := (isLUB_sSup s).2 ha
  le_sSup s _ ha := (isLUB_sSup s).1 ha
  sInf_le s _ ha := (isLUB_sSup (lowerBounds s)).2 fun _ hb => hb ha
  le_sInf s _ ha := (isLUB_sSup (lowerBounds s)).1 ha


/-- Any `CompleteSemilatticeSup` is in fact a `CompleteLattice`.

Note that this construction has bad definitional properties:
see the doc-string on `completeLatticeOfSup`.
-/
def completeLatticeOfCompleteSemilatticeSup (α : Type*) [CompleteSemilatticeSup α] :
    CompleteLattice α :=
  completeLatticeOfSup α fun s => isLUB_sSup s

-- Porting note: as we cannot rename fields while extending,
-- `CompleteLinearOrder` does not directly extend `LinearOrder`.
-- Instead we add the fields by hand, and write a manual instance.


/-- A complete linear order is a linear order whose lattice structure is complete. -/
class CompleteLinearOrder (α : Type*) extends CompleteLattice α, BiheytingAlgebra α where
  /-- A linear order is total. -/
  le_total (a b : α) : a ≤ b ∨ b ≤ a
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableLE : DecidableRel (· ≤ · : α → α → Prop)
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableEq : DecidableEq α := @decidableEqOfDecidableLE _ _ decidableLE
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableLT : DecidableRel (· < · : α → α → Prop) :=
    @decidableLTOfDecidableLE _ _ decidableLE


instance CompleteLinearOrder.toLinearOrder [i : CompleteLinearOrder α] : LinearOrder α where
  __ := i
  min_def a b := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      κ : ι → Sort u_6
      κ' : ι' → Sort u_7
      i : CompleteLinearOrder α
      a b : α
      ⊢ Eq (Min.min a b) (ite (LE.le a b) a b)
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Sort u_4
        ι' : Sort u_5
        κ : ι → Sort u_6
        κ' : ι' → Sort u_7
        i : CompleteLinearOrder α
        a b : α
        h : LE.le a b
        ⊢ Eq (Min.min a b) a
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Sort u_4
        ι' : Sort u_5
        κ : ι → Sort u_6
        κ' : ι' → Sort u_7
        i : CompleteLinearOrder α
        a b : α
        h : Not (LE.le a b)
        ⊢ Eq (Min.min a b) b
      -/
    · simp [(CompleteLinearOrder.le_total a b).resolve_left h]
      /-
        🎉 no goals
      -/
  max_def a b := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Sort u_4
      ι' : Sort u_5
      κ : ι → Sort u_6
      κ' : ι' → Sort u_7
      i : CompleteLinearOrder α
      a b : α
      ⊢ Eq (Max.max a b) (ite (LE.le a b) b a)
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Sort u_4
        ι' : Sort u_5
        κ : ι → Sort u_6
        κ' : ι' → Sort u_7
        i : CompleteLinearOrder α
        a b : α
        h : LE.le a b
        ⊢ Eq (Max.max a b) b
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Sort u_4
        ι' : Sort u_5
        κ : ι → Sort u_6
        κ' : ι' → Sort u_7
        i : CompleteLinearOrder α
        a b : α
        h : Not (LE.le a b)
        ⊢ Eq (Max.max a b) a
      -/
    · simp [(CompleteLinearOrder.le_total a b).resolve_left h]
      /-
        🎉 no goals
      -/


instance instCompleteLattice [CompleteLattice α] : CompleteLattice αᵒᵈ where
  __ := instBoundedOrder α
  le_sSup := @CompleteLattice.sInf_le α _
  sSup_le := @CompleteLattice.le_sInf α _
  sInf_le := @CompleteLattice.le_sSup α _
  le_sInf := @CompleteLattice.sSup_le α _


instance instCompleteLinearOrder [CompleteLinearOrder α] : CompleteLinearOrder αᵒᵈ where
  __ := instCompleteLattice
  __ := instBiheytingAlgebra
  __ := instLinearOrder α


@[simp]
theorem toDual_sSup [SupSet α] (s : Set α) : toDual (sSup s) = sInf (ofDual ⁻¹' s) :=
  rfl


@[simp]
theorem toDual_sInf [InfSet α] (s : Set α) : toDual (sInf s) = sSup (ofDual ⁻¹' s) :=
  rfl


@[simp]
theorem ofDual_sSup [InfSet α] (s : Set αᵒᵈ) : ofDual (sSup s) = sInf (toDual ⁻¹' s) :=
  rfl


@[simp]
theorem ofDual_sInf [SupSet α] (s : Set αᵒᵈ) : ofDual (sInf s) = sSup (toDual ⁻¹' s) :=
  rfl


@[simp]
theorem toDual_iSup [SupSet α] (f : ι → α) : toDual (⨆ i, f i) = ⨅ i, toDual (f i) :=
  rfl


@[simp]
theorem toDual_iInf [InfSet α] (f : ι → α) : toDual (⨅ i, f i) = ⨆ i, toDual (f i) :=
  rfl


@[simp]
theorem ofDual_iSup [InfSet α] (f : ι → αᵒᵈ) : ofDual (⨆ i, f i) = ⨅ i, ofDual (f i) :=
  rfl


@[simp]
theorem ofDual_iInf [SupSet α] (f : ι → αᵒᵈ) : ofDual (⨅ i, f i) = ⨆ i, ofDual (f i) :=
  rfl


theorem sInf_le_sSup (hs : s.Nonempty) : sInf s ≤ sSup s :=
  isGLB_le_isLUB (isGLB_sInf s) (isLUB_sSup s) hs


theorem sSup_union {s t : Set α} : sSup (s ∪ t) = sSup s ⊔ sSup t :=
  ((isLUB_sSup s).union (isLUB_sSup t)).sSup_eq


theorem sInf_union {s t : Set α} : sInf (s ∪ t) = sInf s ⊓ sInf t :=
  ((isGLB_sInf s).union (isGLB_sInf t)).sInf_eq


theorem sSup_inter_le {s t : Set α} : sSup (s ∩ t) ≤ sSup s ⊓ sSup t :=
  sSup_le fun _ hb => le_inf (le_sSup hb.1) (le_sSup hb.2)


theorem le_sInf_inter {s t : Set α} : sInf s ⊔ sInf t ≤ sInf (s ∩ t) :=
  @sSup_inter_le αᵒᵈ _ _ _


@[simp]
theorem sSup_empty : sSup ∅ = (⊥ : α) :=
  (@isLUB_empty α _ _).sSup_eq


@[simp]
theorem sInf_empty : sInf ∅ = (⊤ : α) :=
  (@isGLB_empty α _ _).sInf_eq


@[simp]
theorem sSup_univ : sSup univ = (⊤ : α) :=
  (@isLUB_univ α _ _).sSup_eq


@[simp]
theorem sInf_univ : sInf univ = (⊥ : α) :=
  (@isGLB_univ α _ _).sInf_eq

-- TODO(Jeremy): get this automatically

@[simp]
theorem sSup_insert {a : α} {s : Set α} : sSup (insert a s) = a ⊔ sSup s :=
  ((isLUB_sSup s).insert a).sSup_eq


@[simp]
theorem sInf_insert {a : α} {s : Set α} : sInf (insert a s) = a ⊓ sInf s :=
  ((isGLB_sInf s).insert a).sInf_eq


theorem sSup_le_sSup_of_subset_insert_bot (h : s ⊆ insert ⊥ t) : sSup s ≤ sSup t :=
  (sSup_le_sSup h).trans_eq (sSup_insert.trans (bot_sup_eq _))


theorem sInf_le_sInf_of_subset_insert_top (h : s ⊆ insert ⊤ t) : sInf t ≤ sInf s :=
  (sInf_le_sInf h).trans_eq' (sInf_insert.trans (top_inf_eq _)).symm


@[simp]
theorem sSup_diff_singleton_bot (s : Set α) : sSup (s \ {⊥}) = sSup s :=
  (sSup_le_sSup diff_subset).antisymm <|
    sSup_le_sSup_of_subset_insert_bot <| subset_insert_diff_singleton _ _


@[simp]
theorem sInf_diff_singleton_top (s : Set α) : sInf (s \ {⊤}) = sInf s :=
  @sSup_diff_singleton_bot αᵒᵈ _ s


theorem sSup_pair {a b : α} : sSup {a, b} = a ⊔ b :=
  (@isLUB_pair α _ a b).sSup_eq


theorem sInf_pair {a b : α} : sInf {a, b} = a ⊓ b :=
  (@isGLB_pair α _ a b).sInf_eq


@[simp]
theorem sSup_eq_bot : sSup s = ⊥ ↔ ∀ a ∈ s, a = ⊥ :=
  ⟨fun h _ ha => bot_unique <| h ▸ le_sSup ha, fun h =>
    bot_unique <| sSup_le fun a ha => le_bot_iff.2 <| h a ha⟩


@[simp]
theorem sInf_eq_top : sInf s = ⊤ ↔ ∀ a ∈ s, a = ⊤ :=
  @sSup_eq_bot αᵒᵈ _ _


lemma sSup_eq_bot' {s : Set α} : sSup s = ⊥ ↔ s = ∅ ∨ s = {⊥} := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Iff (Eq (SupSet.sSup s) Bot.bot) (Or (Eq s EmptyCollection.emptyCollection)  …
  -/
  rw [sSup_eq_bot, ← subset_singleton_iff_eq, subset_singleton_iff]
  /-
    🎉 no goals
  -/


theorem eq_singleton_bot_of_sSup_eq_bot_of_nonempty {s : Set α} (h_sup : sSup s = ⊥)
    (hne : s.Nonempty) : s = {⊥} := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    h_sup : Eq (SupSet.sSup s) Bot.bot
    hne : s.Nonempty
    ⊢ Eq s (Singleton.singleton Bot.bot)
  -/
  rw [Set.eq_singleton_iff_nonempty_unique_mem]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    h_sup : Eq (SupSet.sSup s) Bot.bot
    hne : s.Nonempty
    ⊢ And s.Nonempty (∀ (x : α), Membership.mem s x → Eq x Bot.bot)
  -/
  rw [sSup_eq_bot] at h_sup
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    h_sup : ∀ (a : α), Membership.mem s a → Eq a Bot.bot
    hne : s.Nonempty
    ⊢ And s.Nonempty (∀ (x : α), Membership.mem s x → Eq x Bot.bot)
  -/
  exact ⟨hne, h_sup⟩
  /-
    🎉 no goals
  -/


theorem eq_singleton_top_of_sInf_eq_top_of_nonempty : sInf s = ⊤ → s.Nonempty → s = {⊤} :=
  @eq_singleton_bot_of_sSup_eq_bot_of_nonempty αᵒᵈ _ _


/-- Introduction rule to prove that `b` is the supremum of `s`: it suffices to check that `b`
is larger than all elements of `s`, and that this is not the case of any `w < b`.
See `csSup_eq_of_forall_le_of_forall_lt_exists_gt` for a version in conditionally complete
lattices. -/
theorem sSup_eq_of_forall_le_of_forall_lt_exists_gt (h₁ : ∀ a ∈ s, a ≤ b)
    (h₂ : ∀ w, w < b → ∃ a ∈ s, w < a) : sSup s = b :=
  (sSup_le h₁).eq_of_not_lt fun h =>
    let ⟨_, ha, ha'⟩ := h₂ _ h
    ((le_sSup ha).trans_lt ha').false


/-- Introduction rule to prove that `b` is the infimum of `s`: it suffices to check that `b`
is smaller than all elements of `s`, and that this is not the case of any `w > b`.
See `csInf_eq_of_forall_ge_of_forall_gt_exists_lt` for a version in conditionally complete
lattices. -/
theorem sInf_eq_of_forall_ge_of_forall_gt_exists_lt :
    (∀ a ∈ s, b ≤ a) → (∀ w, b < w → ∃ a ∈ s, a < w) → sInf s = b :=
  @sSup_eq_of_forall_le_of_forall_lt_exists_gt αᵒᵈ _ _ _


theorem lt_sSup_iff : b < sSup s ↔ ∃ a ∈ s, b < a :=
  lt_isLUB_iff <| isLUB_sSup s


theorem sInf_lt_iff : sInf s < b ↔ ∃ a ∈ s, a < b :=
  isGLB_lt_iff <| isGLB_sInf s


theorem sSup_eq_top : sSup s = ⊤ ↔ ∀ b < ⊤, ∃ a ∈ s, b < a :=
  ⟨fun h _ hb => lt_sSup_iff.1 <| hb.trans_eq h.symm, fun h =>
    top_unique <|
      le_of_not_gt fun h' =>
        let ⟨_, ha, h⟩ := h _ h'
        (h.trans_le <| le_sSup ha).false⟩


theorem sInf_eq_bot : sInf s = ⊥ ↔ ∀ b > ⊥, ∃ a ∈ s, a < b :=
  @sSup_eq_top αᵒᵈ _ _


theorem lt_iSup_iff {f : ι → α} : a < iSup f ↔ ∃ i, a < f i :=
  lt_sSup_iff.trans exists_range_iff


theorem iInf_lt_iff {f : ι → α} : iInf f < a ↔ ∃ i, f i < a :=
  sInf_lt_iff.trans exists_range_iff


theorem sSup_range : sSup (range f) = iSup f :=
  rfl


                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : SupSet α
                                                                      s : Set α
                                                                      ⊢ Eq (SupSet.sSup s) (iSup fun a => ↑a)
                                                                    -/
theorem sSup_eq_iSup' (s : Set α) : sSup s = ⨆ a : s, (a : α) := by rw [iSup, Subtype.range_coe]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem iSup_congr (h : ∀ i, f i = g i) : ⨆ i, f i = ⨆ i, g i :=
  congr_arg _ <| funext h


theorem biSup_congr {p : ι → Prop} (h : ∀ i, p i → f i = g i) :
    ⨆ (i) (_ : p i), f i = ⨆ (i) (_ : p i), g i :=
  iSup_congr fun i ↦ iSup_congr (h i)


theorem biSup_congr' {p : ι → Prop} {f g : (i : ι) → p i → α}
    (h : ∀ i (hi : p i), f i hi = g i hi) :
    ⨆ i, ⨆ (hi : p i), f i hi = ⨆ i, ⨆ (hi : p i), g i hi := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : SupSet α
    p : ι → Prop
    f g : (i : ι) → p i → α
    h : ∀ (i : ι) (hi : p i), Eq (f i hi) (g i hi)
    ⊢ Eq (iSup fun i => iSup fun hi => f i hi) (iSup fun i => iSup fun hi => g i hi)
  -/
  congr; ext i; congr; ext hi; exact h i hi
                               /-
                                 🎉 no goals
                               -/


theorem Function.Surjective.iSup_comp {f : ι → ι'} (hf : Surjective f) (g : ι' → α) :
    ⨆ x, g (f x) = ⨆ y, g y := by
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    inst✝ : SupSet α
    f : ι → ι'
    hf : Function.Surjective f
    g : ι' → α
    ⊢ Eq (iSup fun x => g (f x)) (iSup fun y => g y)
  -/
  simp only [iSup.eq_1]
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    inst✝ : SupSet α
    f : ι → ι'
    hf : Function.Surjective f
    g : ι' → α
    ⊢ Eq (SupSet.sSup (Set.range fun x => g (f x))) (SupSet.sSup (Set.range fun y  …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    inst✝ : SupSet α
    f : ι → ι'
    hf : Function.Surjective f
    g : ι' → α
    ⊢ Eq (Set.range fun x => g (f x)) (Set.range fun y => g y)
  -/
  exact hf.range_comp g
  /-
    🎉 no goals
  -/


theorem Equiv.iSup_comp {g : ι' → α} (e : ι ≃ ι') : ⨆ x, g (e x) = ⨆ y, g y :=
  e.surjective.iSup_comp _


protected theorem Function.Surjective.iSup_congr {g : ι' → α} (h : ι → ι') (h1 : Surjective h)
    (h2 : ∀ x, g (h x) = f x) : ⨆ x, f x = ⨆ y, g y := by
  /-
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    inst✝ : SupSet α
    f : ι → α
    g : ι' → α
    h : ι → ι'
    h1 : Function.Surjective h
    h2 : ∀ (x : ι), Eq (g (h x)) (f x)
    ⊢ Eq (iSup fun x => f x) (iSup fun y => g y)
  -/
  convert h1.iSup_comp g
  /-
    case h.e'_2.h.e'_4.h
    α : Type u_1
    ι : Sort u_4
    ι' : Sort u_5
    inst✝ : SupSet α
    f : ι → α
    g : ι' → α
    h : ι → ι'
    h1 : Function.Surjective h
    h2 : ∀ (x : ι), Eq (g (h x)) (f x)
    x✝ : ι
    ⊢ Eq (f x✝) (g (h x✝))
  -/
  exact (h2 _).symm
  /-
    🎉 no goals
  -/


protected theorem Equiv.iSup_congr {g : ι' → α} (e : ι ≃ ι') (h : ∀ x, g (e x) = f x) :
    ⨆ x, f x = ⨆ y, g y :=
  e.surjective.iSup_congr _ h


@[congr]
theorem iSup_congr_Prop {p q : Prop} {f₁ : p → α} {f₂ : q → α} (pq : p ↔ q)
    (f : ∀ x, f₁ (pq.mpr x) = f₂ x) : iSup f₁ = iSup f₂ := by
  /-
    α : Type u_1
    inst✝ : SupSet α
    p q : Prop
    f₁ : p → α
    f₂ : q → α
    pq : Iff p q
    f : ∀ (x : q), Eq (f₁ ⋯) (f₂ x)
    ⊢ Eq (iSup f₁) (iSup f₂)
  -/
  obtain rfl := propext pq
  /-
    α : Type u_1
    inst✝ : SupSet α
    p : Prop
    f₁ f₂ : p → α
    pq : Iff p p
    f : ∀ (x : p), Eq (f₁ ⋯) (f₂ x)
    ⊢ Eq (iSup f₁) (iSup f₂)
  -/
  congr with x
  /-
    case e_s.h
    α : Type u_1
    inst✝ : SupSet α
    p : Prop
    f₁ f₂ : p → α
    pq : Iff p p
    f : ∀ (x : p), Eq (f₁ ⋯) (f₂ x)
    x : p
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  apply f
  /-
    🎉 no goals
  -/


theorem iSup_plift_up (f : PLift ι → α) : ⨆ i, f (PLift.up i) = ⨆ i, f i :=
  (PLift.up_surjective.iSup_congr _) fun _ => rfl


theorem iSup_plift_down (f : ι → α) : ⨆ i, f (PLift.down i) = ⨆ i, f i :=
  (PLift.down_surjective.iSup_congr _) fun _ => rfl


theorem iSup_range' (g : β → α) (f : ι → β) : ⨆ b : range f, g b = ⨆ i, g (f i) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝ : SupSet α
    g : β → α
    f : ι → β
    ⊢ Eq (iSup fun b => g ↑b) (iSup fun i => g (f i))
  -/
  rw [iSup, iSup, ← image_eq_range, ← range_comp]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝ : SupSet α
    g : β → α
    f : ι → β
    ⊢ Eq (SupSet.sSup (Set.range (Function.comp g f))) (SupSet.sSup (Set.range fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem sSup_image' {s : Set β} {f : β → α} : sSup (f '' s) = ⨆ a : s, f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : SupSet α
    s : Set β
    f : β → α
    ⊢ Eq (SupSet.sSup (Set.image f s)) (iSup fun a => f ↑a)
  -/
  rw [iSup, image_eq_range]
  /-
    🎉 no goals
  -/


theorem sInf_range : sInf (range f) = iInf f :=
  rfl


theorem sInf_eq_iInf' (s : Set α) : sInf s = ⨅ a : s, (a : α) :=
  @sSup_eq_iSup' αᵒᵈ _ _


theorem iInf_congr (h : ∀ i, f i = g i) : ⨅ i, f i = ⨅ i, g i :=
  congr_arg _ <| funext h


theorem biInf_congr {p : ι → Prop} (h : ∀ i, p i → f i = g i) :
    ⨅ (i) (_ : p i), f i = ⨅ (i) (_ : p i), g i :=
  biSup_congr (α := αᵒᵈ) h


theorem biInf_congr' {p : ι → Prop} {f g : (i : ι) → p i → α}
    (h : ∀ i (hi : p i), f i hi = g i hi) :
    ⨅ i, ⨅ (hi : p i), f i hi = ⨅ i, ⨅ (hi : p i), g i hi := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : InfSet α
    p : ι → Prop
    f g : (i : ι) → p i → α
    h : ∀ (i : ι) (hi : p i), Eq (f i hi) (g i hi)
    ⊢ Eq (iInf fun i => iInf fun hi => f i hi) (iInf fun i => iInf fun hi => g i hi)
  -/
  congr; ext i; congr; ext hi; exact h i hi
                               /-
                                 🎉 no goals
                               -/


theorem Function.Surjective.iInf_comp {f : ι → ι'} (hf : Surjective f) (g : ι' → α) :
    ⨅ x, g (f x) = ⨅ y, g y :=
  @Function.Surjective.iSup_comp αᵒᵈ _ _ _ f hf g


theorem Equiv.iInf_comp {g : ι' → α} (e : ι ≃ ι') : ⨅ x, g (e x) = ⨅ y, g y :=
  @Equiv.iSup_comp αᵒᵈ _ _ _ _ e


protected theorem Function.Surjective.iInf_congr {g : ι' → α} (h : ι → ι') (h1 : Surjective h)
    (h2 : ∀ x, g (h x) = f x) : ⨅ x, f x = ⨅ y, g y :=
  @Function.Surjective.iSup_congr αᵒᵈ _ _ _ _ _ h h1 h2


protected theorem Equiv.iInf_congr {g : ι' → α} (e : ι ≃ ι') (h : ∀ x, g (e x) = f x) :
    ⨅ x, f x = ⨅ y, g y :=
  @Equiv.iSup_congr αᵒᵈ _ _ _ _ _ e h


@[congr]
theorem iInf_congr_Prop {p q : Prop} {f₁ : p → α} {f₂ : q → α} (pq : p ↔ q)
    (f : ∀ x, f₁ (pq.mpr x) = f₂ x) : iInf f₁ = iInf f₂ :=
  @iSup_congr_Prop αᵒᵈ _ p q f₁ f₂ pq f


theorem iInf_plift_up (f : PLift ι → α) : ⨅ i, f (PLift.up i) = ⨅ i, f i :=
  (PLift.up_surjective.iInf_congr _) fun _ => rfl


theorem iInf_plift_down (f : ι → α) : ⨅ i, f (PLift.down i) = ⨅ i, f i :=
  (PLift.down_surjective.iInf_congr _) fun _ => rfl


theorem iInf_range' (g : β → α) (f : ι → β) : ⨅ b : range f, g b = ⨅ i, g (f i) :=
  @iSup_range' αᵒᵈ _ _ _ _ _


theorem sInf_image' {s : Set β} {f : β → α} : sInf (f '' s) = ⨅ a : s, f a :=
  @sSup_image' αᵒᵈ _ _ _ _


theorem le_iSup (f : ι → α) (i : ι) : f i ≤ iSup f :=
  le_sSup ⟨i, rfl⟩


theorem iInf_le (f : ι → α) (i : ι) : iInf f ≤ f i :=
  sInf_le ⟨i, rfl⟩


@[deprecated le_iSup (since := "2024-12-13")]
theorem le_iSup' (f : ι → α) (i : ι) : f i ≤ iSup f := le_iSup f i


@[deprecated iInf_le (since := "2024-12-13")]
theorem iInf_le' (f : ι → α) (i : ι) : iInf f ≤ f i := iInf_le f i


theorem isLUB_iSup : IsLUB (range f) (⨆ j, f j) :=
  isLUB_sSup _


theorem isGLB_iInf : IsGLB (range f) (⨅ j, f j) :=
  isGLB_sInf _


theorem IsLUB.iSup_eq (h : IsLUB (range f) a) : ⨆ j, f j = a :=
  h.sSup_eq


theorem IsGLB.iInf_eq (h : IsGLB (range f) a) : ⨅ j, f j = a :=
  h.sInf_eq


theorem le_iSup_of_le (i : ι) (h : a ≤ f i) : a ≤ iSup f :=
  h.trans <| le_iSup _ i


theorem iInf_le_of_le (i : ι) (h : f i ≤ a) : iInf f ≤ a :=
  (iInf_le _ i).trans h


theorem le_iSup₂ {f : ∀ i, κ i → α} (i : ι) (j : κ i) : f i j ≤ ⨆ (i) (j), f i j :=
  le_iSup_of_le i <| le_iSup (f i) j


theorem iInf₂_le {f : ∀ i, κ i → α} (i : ι) (j : κ i) : ⨅ (i) (j), f i j ≤ f i j :=
  iInf_le_of_le i <| iInf_le (f i) j


theorem le_iSup₂_of_le {f : ∀ i, κ i → α} (i : ι) (j : κ i) (h : a ≤ f i j) :
    a ≤ ⨆ (i) (j), f i j :=
  h.trans <| le_iSup₂ i j


theorem iInf₂_le_of_le {f : ∀ i, κ i → α} (i : ι) (j : κ i) (h : f i j ≤ a) :
    ⨅ (i) (j), f i j ≤ a :=
  (iInf₂_le i j).trans h


theorem iSup_le (h : ∀ i, f i ≤ a) : iSup f ≤ a :=
  sSup_le fun _ ⟨i, Eq⟩ => Eq ▸ h i


theorem le_iInf (h : ∀ i, a ≤ f i) : a ≤ iInf f :=
  le_sInf fun _ ⟨i, Eq⟩ => Eq ▸ h i


theorem iSup₂_le {f : ∀ i, κ i → α} (h : ∀ i j, f i j ≤ a) : ⨆ (i) (j), f i j ≤ a :=
  iSup_le fun i => iSup_le <| h i


theorem le_iInf₂ {f : ∀ i, κ i → α} (h : ∀ i j, a ≤ f i j) : a ≤ ⨅ (i) (j), f i j :=
  le_iInf fun i => le_iInf <| h i


theorem iSup₂_le_iSup (κ : ι → Sort*) (f : ι → α) : ⨆ (i) (_ : κ i), f i ≤ ⨆ i, f i :=
  iSup₂_le fun i _ => le_iSup f i


theorem iInf_le_iInf₂ (κ : ι → Sort*) (f : ι → α) : ⨅ i, f i ≤ ⨅ (i) (_ : κ i), f i :=
  le_iInf₂ fun i _ => iInf_le f i


@[gcongr]
theorem iSup_mono (h : ∀ i, f i ≤ g i) : iSup f ≤ iSup g :=
  iSup_le fun i => le_iSup_of_le i <| h i


@[gcongr]
theorem iInf_mono (h : ∀ i, f i ≤ g i) : iInf f ≤ iInf g :=
  le_iInf fun i => iInf_le_of_le i <| h i


theorem iSup₂_mono {f g : ∀ i, κ i → α} (h : ∀ i j, f i j ≤ g i j) :
    ⨆ (i) (j), f i j ≤ ⨆ (i) (j), g i j :=
  iSup_mono fun i => iSup_mono <| h i


theorem iInf₂_mono {f g : ∀ i, κ i → α} (h : ∀ i j, f i j ≤ g i j) :
    ⨅ (i) (j), f i j ≤ ⨅ (i) (j), g i j :=
  iInf_mono fun i => iInf_mono <| h i


theorem iSup_mono' {g : ι' → α} (h : ∀ i, ∃ i', f i ≤ g i') : iSup f ≤ iSup g :=
  iSup_le fun i => Exists.elim (h i) le_iSup_of_le


theorem iInf_mono' {g : ι' → α} (h : ∀ i', ∃ i, f i ≤ g i') : iInf f ≤ iInf g :=
  le_iInf fun i' => Exists.elim (h i') iInf_le_of_le


theorem iSup₂_mono' {f : ∀ i, κ i → α} {g : ∀ i', κ' i' → α} (h : ∀ i j, ∃ i' j', f i j ≤ g i' j') :
    ⨆ (i) (j), f i j ≤ ⨆ (i) (j), g i j :=
  iSup₂_le fun i j =>
    let ⟨i', j', h⟩ := h i j
    le_iSup₂_of_le i' j' h


theorem iInf₂_mono' {f : ∀ i, κ i → α} {g : ∀ i', κ' i' → α} (h : ∀ i j, ∃ i' j', f i' j' ≤ g i j) :
    ⨅ (i) (j), f i j ≤ ⨅ (i) (j), g i j :=
  le_iInf₂ fun i j =>
    let ⟨i', j', h⟩ := h i j
    iInf₂_le_of_le i' j' h


theorem iSup_const_mono (h : ι → ι') : ⨆ _ : ι, a ≤ ⨆ _ : ι', a :=
  iSup_le <| le_iSup _ ∘ h


theorem iInf_const_mono (h : ι' → ι) : ⨅ _ : ι, a ≤ ⨅ _ : ι', a :=
  le_iInf <| iInf_le _ ∘ h


theorem iSup_iInf_le_iInf_iSup (f : ι → ι' → α) : ⨆ i, ⨅ j, f i j ≤ ⨅ j, ⨆ i, f i j :=
  iSup_le fun i => iInf_mono fun j => le_iSup (fun i => f i j) i


theorem biSup_mono {p q : ι → Prop} (hpq : ∀ i, p i → q i) :
    ⨆ (i) (_ : p i), f i ≤ ⨆ (i) (_ : q i), f i :=
  iSup_mono fun i => iSup_const_mono (hpq i)


theorem biInf_mono {p q : ι → Prop} (hpq : ∀ i, p i → q i) :
    ⨅ (i) (_ : q i), f i ≤ ⨅ (i) (_ : p i), f i :=
  iInf_mono fun i => iInf_const_mono (hpq i)


@[simp]
theorem iSup_le_iff : iSup f ≤ a ↔ ∀ i, f i ≤ a :=
  (isLUB_le_iff isLUB_iSup).trans forall_mem_range


@[simp]
theorem le_iInf_iff : a ≤ iInf f ↔ ∀ i, a ≤ f i :=
  (le_isGLB_iff isGLB_iInf).trans forall_mem_range


theorem iSup₂_le_iff {f : ∀ i, κ i → α} : ⨆ (i) (j), f i j ≤ a ↔ ∀ i j, f i j ≤ a := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLattice α
    a : α
    f : (i : ι) → κ i → α
    ⊢ Iff (LE.le (iSup fun i => iSup fun j => f i j) a) (∀ (i : ι) (j : κ i), LE.l …
  -/
  simp_rw [iSup_le_iff]
  /-
    🎉 no goals
  -/


theorem le_iInf₂_iff {f : ∀ i, κ i → α} : (a ≤ ⨅ (i) (j), f i j) ↔ ∀ i j, a ≤ f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLattice α
    a : α
    f : (i : ι) → κ i → α
    ⊢ Iff (LE.le a (iInf fun i => iInf fun j => f i j)) (∀ (i : ι) (j : κ i), LE.l …
  -/
  simp_rw [le_iInf_iff]
  /-
    🎉 no goals
  -/


theorem iSup_lt_iff : iSup f < a ↔ ∃ b, b < a ∧ ∀ i, f i ≤ b :=
  ⟨fun h => ⟨iSup f, h, le_iSup f⟩, fun ⟨_, h, hb⟩ => (iSup_le hb).trans_lt h⟩


theorem lt_iInf_iff : a < iInf f ↔ ∃ b, a < b ∧ ∀ i, b ≤ f i :=
  ⟨fun h => ⟨iInf f, h, iInf_le f⟩, fun ⟨_, h, hb⟩ => h.trans_le <| le_iInf hb⟩


theorem sSup_eq_iSup {s : Set α} : sSup s = ⨆ a ∈ s, a :=
  le_antisymm (sSup_le le_iSup₂) (iSup₂_le fun _ => le_sSup)


theorem sInf_eq_iInf {s : Set α} : sInf s = ⨅ a ∈ s, a :=
  @sSup_eq_iSup αᵒᵈ _ _


lemma sSup_lowerBounds_eq_sInf (s : Set α) : sSup (lowerBounds s) = sInf s :=
  (isLUB_sSup _).unique (isGLB_sInf _).isLUB


lemma sInf_upperBounds_eq_csSup (s : Set α) : sInf (upperBounds s) = sSup s :=
  (isGLB_sInf _).unique (isLUB_sSup _).isGLB


theorem Monotone.le_map_iSup [CompleteLattice β] {f : α → β} (hf : Monotone f) :
    ⨆ i, f (s i) ≤ f (iSup s) :=
  iSup_le fun _ => hf <| le_iSup _ _


theorem Antitone.le_map_iInf [CompleteLattice β] {f : α → β} (hf : Antitone f) :
    ⨆ i, f (s i) ≤ f (iInf s) :=
  hf.dual_left.le_map_iSup


theorem Monotone.le_map_iSup₂ [CompleteLattice β] {f : α → β} (hf : Monotone f) (s : ∀ i, κ i → α) :
    ⨆ (i) (j), f (s i j) ≤ f (⨆ (i) (j), s i j) :=
  iSup₂_le fun _ _ => hf <| le_iSup₂ _ _


theorem Antitone.le_map_iInf₂ [CompleteLattice β] {f : α → β} (hf : Antitone f) (s : ∀ i, κ i → α) :
    ⨆ (i) (j), f (s i j) ≤ f (⨅ (i) (j), s i j) :=
  hf.dual_left.le_map_iSup₂ _


theorem Monotone.le_map_sSup [CompleteLattice β] {s : Set α} {f : α → β} (hf : Monotone f) :
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      inst✝¹ : CompleteLattice α
                                      inst✝ : CompleteLattice β
                                      s : Set α
                                      f : α → β
                                      hf : Monotone f
                                      ⊢ LE.le (iSup fun a => iSup fun h => f a) (f (SupSet.sSup s))
                                    -/
    ⨆ a ∈ s, f a ≤ f (sSup s) := by rw [sSup_eq_iSup]; exact hf.le_map_iSup₂ _
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Antitone.le_map_sInf [CompleteLattice β] {s : Set α} {f : α → β} (hf : Antitone f) :
    ⨆ a ∈ s, f a ≤ f (sInf s) :=
  hf.dual_left.le_map_sSup


theorem OrderIso.map_iSup [CompleteLattice β] (f : α ≃o β) (x : ι → α) :
    f (⨆ i, x i) = ⨆ i, f (x i) :=
  eq_of_forall_ge_iff <| f.surjective.forall.2
              /-
                α : Type u_1
                β : Type u_2
                ι : Sort u_4
                inst✝¹ : CompleteLattice α
                inst✝ : CompleteLattice β
                f : OrderIso α β
                x✝ : ι → α
                x : α
                ⊢ Iff (LE.le (f (iSup fun i => x✝ i)) (f x)) (LE.le (iSup fun i => f (x✝ i)) ( …
              -/
  fun x => by simp only [f.le_iff_le, iSup_le_iff]
              /-
                🎉 no goals
              -/


theorem OrderIso.map_iInf [CompleteLattice β] (f : α ≃o β) (x : ι → α) :
    f (⨅ i, x i) = ⨅ i, f (x i) :=
  OrderIso.map_iSup f.dual _


theorem OrderIso.map_sSup [CompleteLattice β] (f : α ≃o β) (s : Set α) :
    f (sSup s) = ⨆ a ∈ s, f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : OrderIso α β
    s : Set α
    ⊢ Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h => f a)
  -/
  simp only [sSup_eq_iSup, OrderIso.map_iSup]
  /-
    🎉 no goals
  -/


theorem OrderIso.map_sInf [CompleteLattice β] (f : α ≃o β) (s : Set α) :
    f (sInf s) = ⨅ a ∈ s, f a :=
  OrderIso.map_sSup f.dual _


theorem iSup_comp_le {ι' : Sort*} (f : ι' → α) (g : ι → ι') : ⨆ x, f (g x) ≤ ⨆ y, f y :=
  iSup_mono' fun _ => ⟨_, le_rfl⟩


theorem le_iInf_comp {ι' : Sort*} (f : ι' → α) (g : ι → ι') : ⨅ y, f y ≤ ⨅ x, f (g x) :=
  iInf_mono' fun _ => ⟨_, le_rfl⟩


theorem Monotone.iSup_comp_eq [Preorder β] {f : β → α} (hf : Monotone f) {s : ι → β}
    (hs : ∀ x, ∃ i, x ≤ s i) : ⨆ x, f (s x) = ⨆ y, f y :=
  le_antisymm (iSup_comp_le _ _) (iSup_mono' fun x => (hs x).imp fun _ hi => hf hi)


theorem Monotone.iInf_comp_eq [Preorder β] {f : β → α} (hf : Monotone f) {s : ι → β}
    (hs : ∀ x, ∃ i, s i ≤ x) : ⨅ x, f (s x) = ⨅ y, f y :=
  le_antisymm (iInf_mono' fun x => (hs x).imp fun _ hi => hf hi) (le_iInf_comp _ _)


theorem Antitone.map_iSup_le [CompleteLattice β] {f : α → β} (hf : Antitone f) :
    f (iSup s) ≤ ⨅ i, f (s i) :=
  le_iInf fun _ => hf <| le_iSup _ _


theorem Monotone.map_iInf_le [CompleteLattice β] {f : α → β} (hf : Monotone f) :
    f (iInf s) ≤ ⨅ i, f (s i) :=
  hf.dual_left.map_iSup_le


theorem Antitone.map_iSup₂_le [CompleteLattice β] {f : α → β} (hf : Antitone f) (s : ∀ i, κ i → α) :
    f (⨆ (i) (j), s i j) ≤ ⨅ (i) (j), f (s i j) :=
  hf.dual.le_map_iInf₂ _


theorem Monotone.map_iInf₂_le [CompleteLattice β] {f : α → β} (hf : Monotone f) (s : ∀ i, κ i → α) :
    f (⨅ (i) (j), s i j) ≤ ⨅ (i) (j), f (s i j) :=
  hf.dual.le_map_iSup₂ _


theorem Antitone.map_sSup_le [CompleteLattice β] {s : Set α} {f : α → β} (hf : Antitone f) :
    f (sSup s) ≤ ⨅ a ∈ s, f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    s : Set α
    f : α → β
    hf : Antitone f
    ⊢ LE.le (f (SupSet.sSup s)) (iInf fun a => iInf fun h => f a)
  -/
  rw [sSup_eq_iSup]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    s : Set α
    f : α → β
    hf : Antitone f
    ⊢ LE.le (f (iSup fun a => iSup fun h => a)) (iInf fun a => iInf fun h => f a)
  -/
  exact hf.map_iSup₂_le _
  /-
    🎉 no goals
  -/


theorem Monotone.map_sInf_le [CompleteLattice β] {s : Set α} {f : α → β} (hf : Monotone f) :
    f (sInf s) ≤ ⨅ a ∈ s, f a :=
  hf.dual_left.map_sSup_le


theorem iSup_const_le : ⨆ _ : ι, a ≤ a :=
  iSup_le fun _ => le_rfl


theorem le_iInf_const : a ≤ ⨅ _ : ι, a :=
  le_iInf fun _ => le_rfl

-- We generalize this to conditionally complete lattices in `ciSup_const` and `ciInf_const`.

                                                       /-
                                                         α : Type u_1
                                                         ι : Sort u_4
                                                         inst✝¹ : CompleteLattice α
                                                         a : α
                                                         inst✝ : Nonempty ι
                                                         ⊢ Eq (iSup fun x => a) a
                                                       -/
theorem iSup_const [Nonempty ι] : ⨆ _ : ι, a = a := by rw [iSup, range_const, sSup_singleton]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem iInf_const [Nonempty ι] : ⨅ _ : ι, a = a :=
  @iSup_const αᵒᵈ _ _ a _


lemma iSup_unique [Unique ι] (f : ι → α) : ⨆ i, f i = f default := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Unique ι
    f : ι → α
    ⊢ Eq (iSup fun i => f i) (f Inhabited.default)
  -/
  simp only [congr_arg f (Unique.eq_default _), iSup_const]
  /-
    🎉 no goals
  -/


lemma iInf_unique [Unique ι] (f : ι → α) : ⨅ i, f i = f default := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Unique ι
    f : ι → α
    ⊢ Eq (iInf fun i => f i) (f Inhabited.default)
  -/
  simp only [congr_arg f (Unique.eq_default _), iInf_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem iSup_bot : (⨆ _ : ι, ⊥ : α) = ⊥ :=
  bot_unique iSup_const_le


@[simp]
theorem iInf_top : (⨅ _ : ι, ⊤ : α) = ⊤ :=
  top_unique le_iInf_const


@[simp]
theorem iSup_eq_bot : iSup s = ⊥ ↔ ∀ i, s i = ⊥ :=
  sSup_eq_bot.trans forall_mem_range


@[simp]
theorem iInf_eq_top : iInf s = ⊤ ↔ ∀ i, s i = ⊤ :=
  sInf_eq_top.trans forall_mem_range


                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                inst✝ : CompleteLattice α
                                                                s : ι → α
                                                                ⊢ Iff (LT.lt Bot.bot (iSup fun i => s i)) (Exists fun i => LT.lt Bot.bot (s i))
                                                              -/
@[simp] lemma bot_lt_iSup : ⊥ < ⨆ i, s i ↔ ∃ i, ⊥ < s i := by simp [bot_lt_iff_ne_bot]
                                                              /-
                                                                🎉 no goals
                                                              -/

                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                inst✝ : CompleteLattice α
                                                                s : ι → α
                                                                ⊢ Iff (LT.lt (iInf fun i => s i) Top.top) (Exists fun i => LT.lt (s i) Top.top)
                                                              -/
@[simp] lemma iInf_lt_top : ⨅ i, s i < ⊤ ↔ ∃ i, s i < ⊤ := by simp [lt_top_iff_ne_top]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem iSup₂_eq_bot {f : ∀ i, κ i → α} : ⨆ (i) (j), f i j = ⊥ ↔ ∀ i j, f i j = ⊥ := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLattice α
    f : (i : ι) → κ i → α
    ⊢ Iff (Eq (iSup fun i => iSup fun j => f i j) Bot.bot) (∀ (i : ι) (j : κ i), E …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iInf₂_eq_top {f : ∀ i, κ i → α} : ⨅ (i) (j), f i j = ⊤ ↔ ∀ i j, f i j = ⊤ := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLattice α
    f : (i : ι) → κ i → α
    ⊢ Iff (Eq (iInf fun i => iInf fun j => f i j) Top.top) (∀ (i : ι) (j : κ i), E …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem iSup_pos {p : Prop} {f : p → α} (hp : p) : ⨆ h : p, f h = f hp :=
  le_antisymm (iSup_le fun _ => le_rfl) (le_iSup _ _)


@[simp]
theorem iInf_pos {p : Prop} {f : p → α} (hp : p) : ⨅ h : p, f h = f hp :=
  le_antisymm (iInf_le _ _) (le_iInf fun _ => le_rfl)


@[simp]
theorem iSup_neg {p : Prop} {f : p → α} (hp : ¬p) : ⨆ h : p, f h = ⊥ :=
  le_antisymm (iSup_le fun h => (hp h).elim) bot_le


@[simp]
theorem iInf_neg {p : Prop} {f : p → α} (hp : ¬p) : ⨅ h : p, f h = ⊤ :=
  le_antisymm le_top <| le_iInf fun h => (hp h).elim


/-- Introduction rule to prove that `b` is the supremum of `f`: it suffices to check that `b`
is larger than `f i` for all `i`, and that this is not the case of any `w<b`.
See `ciSup_eq_of_forall_le_of_forall_lt_exists_gt` for a version in conditionally complete
lattices. -/
theorem iSup_eq_of_forall_le_of_forall_lt_exists_gt {f : ι → α} (h₁ : ∀ i, f i ≤ b)
    (h₂ : ∀ w, w < b → ∃ i, w < f i) : ⨆ i : ι, f i = b :=
  sSup_eq_of_forall_le_of_forall_lt_exists_gt (forall_mem_range.mpr h₁) fun w hw =>
    exists_range_iff.mpr <| h₂ w hw


/-- Introduction rule to prove that `b` is the infimum of `f`: it suffices to check that `b`
is smaller than `f i` for all `i`, and that this is not the case of any `w>b`.
See `ciInf_eq_of_forall_ge_of_forall_gt_exists_lt` for a version in conditionally complete
lattices. -/
theorem iInf_eq_of_forall_ge_of_forall_gt_exists_lt :
    (∀ i, b ≤ f i) → (∀ w, b < w → ∃ i, f i < w) → ⨅ i, f i = b :=
  @iSup_eq_of_forall_le_of_forall_lt_exists_gt αᵒᵈ _ _ _ _


theorem iSup_eq_dif {p : Prop} [Decidable p] (a : p → α) :
                                                  /-
                                                    α : Type u_1
                                                    inst✝¹ : CompleteLattice α
                                                    p : Prop
                                                    inst✝ : Decidable p
                                                    a : p → α
                                                    ⊢ Eq (iSup fun h => a h) (dite p (fun h => a h) fun h => Bot.bot)
                                                  -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    ⨆ h : p, a h = if h : p then a h else ⊥ := by by_cases h : p <;> simp [h]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem iSup_eq_if {p : Prop} [Decidable p] (a : α) : ⨆ _ : p, a = if p then a else ⊥ :=
  iSup_eq_dif fun _ => a


theorem iInf_eq_dif {p : Prop} [Decidable p] (a : p → α) :
    ⨅ h : p, a h = if h : p then a h else ⊤ :=
  @iSup_eq_dif αᵒᵈ _ _ _ _


theorem iInf_eq_if {p : Prop} [Decidable p] (a : α) : ⨅ _ : p, a = if p then a else ⊤ :=
  iInf_eq_dif fun _ => a


theorem iSup_comm {f : ι → ι' → α} : ⨆ (i) (j), f i j = ⨆ (j) (i), f i j :=
  le_antisymm (iSup_le fun i => iSup_mono fun j => le_iSup (fun i => f i j) i)
    (iSup_le fun _ => iSup_mono fun _ => le_iSup _ _)


theorem iInf_comm {f : ι → ι' → α} : ⨅ (i) (j), f i j = ⨅ (j) (i), f i j :=
  @iSup_comm αᵒᵈ _ _ _ _


theorem iSup₂_comm {ι₁ ι₂ : Sort*} {κ₁ : ι₁ → Sort*} {κ₂ : ι₂ → Sort*}
    (f : ∀ i₁, κ₁ i₁ → ∀ i₂, κ₂ i₂ → α) :
    ⨆ (i₁) (j₁) (i₂) (j₂), f i₁ j₁ i₂ j₂ = ⨆ (i₂) (j₂) (i₁) (j₁), f i₁ j₁ i₂ j₂ := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι₁ : Sort u_8
    ι₂ : Sort u_9
    κ₁ : ι₁ → Sort u_10
    κ₂ : ι₂ → Sort u_11
    f : (i₁ : ι₁) → κ₁ i₁ → (i₂ : ι₂) → κ₂ i₂ → α
    ⊢ Eq (iSup fun i₁ => iSup fun j₁ => iSup fun i₂ => iSup fun j₂ => f i₁ j₁ i₂ j …
  -/
  simp only [@iSup_comm _ (κ₁ _), @iSup_comm _ ι₁]
  /-
    🎉 no goals
  -/


theorem iInf₂_comm {ι₁ ι₂ : Sort*} {κ₁ : ι₁ → Sort*} {κ₂ : ι₂ → Sort*}
    (f : ∀ i₁, κ₁ i₁ → ∀ i₂, κ₂ i₂ → α) :
    ⨅ (i₁) (j₁) (i₂) (j₂), f i₁ j₁ i₂ j₂ = ⨅ (i₂) (j₂) (i₁) (j₁), f i₁ j₁ i₂ j₂ := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι₁ : Sort u_8
    ι₂ : Sort u_9
    κ₁ : ι₁ → Sort u_10
    κ₂ : ι₂ → Sort u_11
    f : (i₁ : ι₁) → κ₁ i₁ → (i₂ : ι₂) → κ₂ i₂ → α
    ⊢ Eq (iInf fun i₁ => iInf fun j₁ => iInf fun i₂ => iInf fun j₂ => f i₁ j₁ i₂ j …
  -/
  simp only [@iInf_comm _ (κ₁ _), @iInf_comm _ ι₁]
  /-
    🎉 no goals
  -/

/- TODO: this is strange. In the proof below, we get exactly the desired
   among the equalities, but close does not get it.
begin
  apply @le_antisymm,
    simp, intros,
    begin [smt]
      ematch, ematch, ematch, trace_state, have := le_refl (f i_1 i),
      trace_state, close
    end
end
-/

@[simp]
theorem iSup_iSup_eq_left {b : β} {f : ∀ x : β, x = b → α} : ⨆ x, ⨆ h : x = b, f x h = f b rfl :=
  (@le_iSup₂ _ _ _ _ f b rfl).antisymm'
    (iSup_le fun c =>
      iSup_le <| by
        /-
          α : Type u_1
          β : Type u_2
          inst✝ : CompleteLattice α
          b : β
          f : (x : β) → Eq x b → α
          c : β
          ⊢ ∀ (i : Eq c b), LE.le (f c i) (f b ⋯)
        -/
        rintro rfl
        /-
          α : Type u_1
          β : Type u_2
          inst✝ : CompleteLattice α
          c : β
          f : (x : β) → Eq x c → α
          ⊢ LE.le (f c ⋯) (f c ⋯)
        -/
        rfl)
        /-
          🎉 no goals
        -/


@[simp]
theorem iInf_iInf_eq_left {b : β} {f : ∀ x : β, x = b → α} : ⨅ x, ⨅ h : x = b, f x h = f b rfl :=
  @iSup_iSup_eq_left αᵒᵈ _ _ _ _


@[simp]
theorem iSup_iSup_eq_right {b : β} {f : ∀ x : β, b = x → α} : ⨆ x, ⨆ h : b = x, f x h = f b rfl :=
  (le_iSup₂ b rfl).antisymm'
    (iSup₂_le fun c => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : CompleteLattice α
        b : β
        f : (x : β) → Eq b x → α
        c : β
        ⊢ ∀ (j : Eq b c), LE.le (f c j) (f b ⋯)
      -/
      rintro rfl
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : CompleteLattice α
        b : β
        f : (x : β) → Eq b x → α
        ⊢ LE.le (f b ⋯) (f b ⋯)
      -/
      rfl)
      /-
        🎉 no goals
      -/


@[simp]
theorem iInf_iInf_eq_right {b : β} {f : ∀ x : β, b = x → α} : ⨅ x, ⨅ h : b = x, f x h = f b rfl :=
  @iSup_iSup_eq_right αᵒᵈ _ _ _ _


theorem iSup_subtype {p : ι → Prop} {f : Subtype p → α} : iSup f = ⨆ (i) (h : p i), f ⟨i, h⟩ :=
  le_antisymm (iSup_le fun ⟨i, h⟩ => @le_iSup₂ _ _ p _ (fun i h => f ⟨i, h⟩) i h)
    (iSup₂_le fun _ _ => le_iSup _ _)


theorem iInf_subtype : ∀ {p : ι → Prop} {f : Subtype p → α}, iInf f = ⨅ (i) (h : p i), f ⟨i, h⟩ :=
  @iSup_subtype αᵒᵈ _ _


theorem iSup_subtype' {p : ι → Prop} {f : ∀ i, p i → α} :
    ⨆ (i) (h), f i h = ⨆ x : Subtype p, f x x.property :=
  (@iSup_subtype _ _ _ p fun x => f x.val x.property).symm


theorem iInf_subtype' {p : ι → Prop} {f : ∀ i, p i → α} :
    ⨅ (i) (h : p i), f i h = ⨅ x : Subtype p, f x x.property :=
  (@iInf_subtype _ _ _ p fun x => f x.val x.property).symm


theorem iSup_subtype'' {ι} (s : Set ι) (f : ι → α) : ⨆ i : s, f i = ⨆ (t : ι) (_ : t ∈ s), f t :=
  iSup_subtype


theorem iInf_subtype'' {ι} (s : Set ι) (f : ι → α) : ⨅ i : s, f i = ⨅ (t : ι) (_ : t ∈ s), f t :=
  iInf_subtype


theorem biSup_const {a : α} {s : Set β} (hs : s.Nonempty) : ⨆ i ∈ s, a = a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    a : α
    s : Set β
    hs : s.Nonempty
    ⊢ Eq (iSup fun i => iSup fun h => a) a
  -/
  haveI : Nonempty s := Set.nonempty_coe_sort.mpr hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    a : α
    s : Set β
    hs : s.Nonempty
    this : Nonempty ↑s
    ⊢ Eq (iSup fun i => iSup fun h => a) a
  -/
  rw [← iSup_subtype'', iSup_const]
  /-
    🎉 no goals
  -/


theorem biInf_const {a : α} {s : Set β} (hs : s.Nonempty) : ⨅ i ∈ s, a = a :=
  biSup_const (α := αᵒᵈ) hs


theorem iSup_sup_eq : ⨆ x, f x ⊔ g x = (⨆ x, f x) ⊔ ⨆ x, g x :=
  le_antisymm (iSup_le fun _ => sup_le_sup (le_iSup _ _) <| le_iSup _ _)
    (sup_le (iSup_mono fun _ => le_sup_left) <| iSup_mono fun _ => le_sup_right)


theorem iInf_inf_eq : ⨅ x, f x ⊓ g x = (⨅ x, f x) ⊓ ⨅ x, g x :=
  @iSup_sup_eq αᵒᵈ _ _ _ _


lemma Equiv.biSup_comp {ι ι' : Type*} {g : ι' → α} (e : ι ≃ ι') (s : Set ι') :
    ⨆ i ∈ e.symm '' s, g (e i) = ⨆ i ∈ s, g i := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Type u_8
    ι' : Type u_9
    g : ι' → α
    e : Equiv ι ι'
    s : Set ι'
    ⊢ Eq (iSup fun i => iSup fun h => g (e i)) (iSup fun i => iSup fun h => g i)
  -/
  simpa only [iSup_subtype'] using (image e.symm s).symm.iSup_comp (g := g ∘ (↑))
  /-
    🎉 no goals
  -/


lemma Equiv.biInf_comp {ι ι' : Type*} {g : ι' → α} (e : ι ≃ ι') (s : Set ι') :
    ⨅ i ∈ e.symm '' s, g (e i) = ⨅ i ∈ s, g i :=
  e.biSup_comp s (α := αᵒᵈ)


lemma biInf_le {ι : Type*} {s : Set ι} (f : ι → α) {i : ι} (hi : i ∈ s) :
    ⨅ i ∈ s, f i ≤ f i := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Type u_8
    s : Set ι
    f : ι → α
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (iInf fun i => iInf fun h => f i) (f i)
  -/
  simpa only [iInf_subtype'] using iInf_le (ι := s) (f := f ∘ (↑)) ⟨i, hi⟩
  /-
    🎉 no goals
  -/


lemma le_biSup {ι : Type*} {s : Set ι} (f : ι → α) {i : ι} (hi : i ∈ s) :
    f i ≤ ⨆ i ∈ s, f i :=
  biInf_le (α := αᵒᵈ) f hi

/- TODO: here is another example where more flexible pattern matching
   might help.

begin
  apply @le_antisymm,
  safe, pose h := f a ⊓ g a, begin [smt] ematch, ematch end
end
-/

theorem iSup_sup [Nonempty ι] {f : ι → α} {a : α} : (⨆ x, f x) ⊔ a = ⨆ x, f x ⊔ a := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Nonempty ι
    f : ι → α
    a : α
    ⊢ Eq (Max.max (iSup fun x => f x) a) (iSup fun x => Max.max (f x) a)
  -/
  rw [iSup_sup_eq, iSup_const]
  /-
    🎉 no goals
  -/


theorem iInf_inf [Nonempty ι] {f : ι → α} {a : α} : (⨅ x, f x) ⊓ a = ⨅ x, f x ⊓ a := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Nonempty ι
    f : ι → α
    a : α
    ⊢ Eq (Min.min (iInf fun x => f x) a) (iInf fun x => Min.min (f x) a)
  -/
  rw [iInf_inf_eq, iInf_const]
  /-
    🎉 no goals
  -/


theorem sup_iSup [Nonempty ι] {f : ι → α} {a : α} : (a ⊔ ⨆ x, f x) = ⨆ x, a ⊔ f x := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Nonempty ι
    f : ι → α
    a : α
    ⊢ Eq (Max.max a (iSup fun x => f x)) (iSup fun x => Max.max a (f x))
  -/
  rw [iSup_sup_eq, iSup_const]
  /-
    🎉 no goals
  -/


theorem inf_iInf [Nonempty ι] {f : ι → α} {a : α} : (a ⊓ ⨅ x, f x) = ⨅ x, a ⊓ f x := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    inst✝ : Nonempty ι
    f : ι → α
    a : α
    ⊢ Eq (Min.min a (iInf fun x => f x)) (iInf fun x => Min.min a (f x))
  -/
  rw [iInf_inf_eq, iInf_const]
  /-
    🎉 no goals
  -/


theorem biSup_sup {p : ι → Prop} {f : ∀ i, p i → α} {a : α} (h : ∃ i, p i) :
    (⨆ (i) (h : p i), f i h) ⊔ a = ⨆ (i) (h : p i), f i h ⊔ a := by
  haveI : Nonempty { i // p i } :=
    let ⟨i, hi⟩ := h
    ⟨⟨i, hi⟩⟩
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLattice α
    p : ι → Prop
    f : (i : ι) → p i → α
    a : α
    h : Exists fun i => p i
    this : Nonempty (Subtype fun i => p i)
    ⊢ Eq (Max.max (iSup fun i => iSup fun h => f i h) a) (iSup fun i => iSup fun h …
  -/
  rw [iSup_subtype', iSup_subtype', iSup_sup]
  /-
    🎉 no goals
  -/


theorem sup_biSup {p : ι → Prop} {f : ∀ i, p i → α} {a : α} (h : ∃ i, p i) :
    (a ⊔ ⨆ (i) (h : p i), f i h) = ⨆ (i) (h : p i), a ⊔ f i h := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLattice α
    p : ι → Prop
    f : (i : ι) → p i → α
    a : α
    h : Exists fun i => p i
    ⊢ Eq (Max.max a (iSup fun i => iSup fun h => f i h)) (iSup fun i => iSup fun h …
  -/
  simpa only [sup_comm] using @biSup_sup α _ _ p _ _ h
  /-
    🎉 no goals
  -/


theorem biInf_inf {p : ι → Prop} {f : ∀ i, p i → α} {a : α} (h : ∃ i, p i) :
    (⨅ (i) (h : p i), f i h) ⊓ a = ⨅ (i) (h : p i), f i h ⊓ a :=
  @biSup_sup αᵒᵈ ι _ p f _ h


theorem inf_biInf {p : ι → Prop} {f : ∀ i, p i → α} {a : α} (h : ∃ i, p i) :
    (a ⊓ ⨅ (i) (h : p i), f i h) = ⨅ (i) (h : p i), a ⊓ f i h :=
  @sup_biSup αᵒᵈ ι _ p f _ h


lemma biSup_lt_eq_iSup {ι : Type*} [LT ι] [NoMaxOrder ι] {f : ι → α} :
    ⨆ (i) (j < i), f j = ⨆ i, f i := by
  /-
    α : Type u_1
    inst✝² : CompleteLattice α
    ι : Type u_8
    inst✝¹ : LT ι
    inst✝ : NoMaxOrder ι
    f : ι → α
    ⊢ Eq (iSup fun i => iSup fun j => iSup fun h => f j) (iSup fun i => f i)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝² : CompleteLattice α
      ι : Type u_8
      inst✝¹ : LT ι
      inst✝ : NoMaxOrder ι
      f : ι → α
      ⊢ LE.le (iSup fun i => iSup fun j => iSup fun h => f j) (iSup fun i => f i)
    -/
  · exact iSup_le fun _ ↦ iSup₂_le fun _ _ ↦ le_iSup _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝² : CompleteLattice α
      ι : Type u_8
      inst✝¹ : LT ι
      inst✝ : NoMaxOrder ι
      f : ι → α
      ⊢ LE.le (iSup fun i => f i) (iSup fun i => iSup fun j => iSup fun h => f j)
    -/
  · refine iSup_le fun j ↦ ?_
    /-
      case a
      α : Type u_1
      inst✝² : CompleteLattice α
      ι : Type u_8
      inst✝¹ : LT ι
      inst✝ : NoMaxOrder ι
      f : ι → α
      j : ι
      ⊢ LE.le (f j) (iSup fun i => iSup fun j => iSup fun h => f j)
    -/
    obtain ⟨i, jlt⟩ := exists_gt j
    /-
      case a.intro
      α : Type u_1
      inst✝² : CompleteLattice α
      ι : Type u_8
      inst✝¹ : LT ι
      inst✝ : NoMaxOrder ι
      f : ι → α
      j i : ι
      jlt : LT.lt j i
      ⊢ LE.le (f j) (iSup fun i => iSup fun j => iSup fun h => f j)
    -/
    exact le_iSup_of_le i (le_iSup₂_of_le j jlt le_rfl)
    /-
      🎉 no goals
    -/


lemma biSup_le_eq_iSup {ι : Type*} [Preorder ι] {f : ι → α} :
    ⨆ (i) (j ≤ i), f j = ⨆ i, f i := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    ι : Type u_8
    inst✝ : Preorder ι
    f : ι → α
    ⊢ Eq (iSup fun i => iSup fun j => iSup fun h => f j) (iSup fun i => f i)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : CompleteLattice α
      ι : Type u_8
      inst✝ : Preorder ι
      f : ι → α
      ⊢ LE.le (iSup fun i => iSup fun j => iSup fun h => f j) (iSup fun i => f i)
    -/
  · exact iSup_le fun _ ↦ iSup₂_le fun _ _ ↦ le_iSup _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝¹ : CompleteLattice α
      ι : Type u_8
      inst✝ : Preorder ι
      f : ι → α
      ⊢ LE.le (iSup fun i => f i) (iSup fun i => iSup fun j => iSup fun h => f j)
    -/
  · exact iSup_le fun j ↦ le_iSup_of_le j (le_iSup₂_of_le j le_rfl le_rfl)
    /-
      🎉 no goals
    -/


lemma biInf_lt_eq_iInf {ι : Type*} [LT ι] [NoMaxOrder ι] {f : ι → α} :
    ⨅ (i) (j < i), f j = ⨅ i, f i :=
  biSup_lt_eq_iSup (α := αᵒᵈ)


lemma biInf_le_eq_iInf {ι : Type*} [Preorder ι] {f : ι → α} : ⨅ (i) (j ≤ i), f j = ⨅ i, f i :=
  biSup_le_eq_iSup (α := αᵒᵈ)


lemma biSup_gt_eq_iSup {ι : Type*} [LT ι] [NoMinOrder ι] {f : ι → α} :
    ⨆ (i) (j > i), f j = ⨆ i, f i :=
  biSup_lt_eq_iSup (ι := ιᵒᵈ)


lemma biSup_ge_eq_iSup {ι : Type*} [Preorder ι] {f : ι → α} : ⨆ (i) (j ≥ i), f j = ⨆ i, f i :=
  biSup_le_eq_iSup (ι := ιᵒᵈ)


lemma biInf_gt_eq_iInf {ι : Type*} [LT ι] [NoMinOrder ι] {f : ι → α} :
    ⨅ (i) (j > i), f j = ⨅ i, f i :=
  biInf_lt_eq_iInf (ι := ιᵒᵈ)


lemma biInf_ge_eq_iInf {ι : Type*} [Preorder ι] {f : ι → α} : ⨅ (i) (j ≥ i), f j = ⨅ i, f i :=
  biInf_le_eq_iInf (ι := ιᵒᵈ)


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : CompleteLattice α
                                                        s : False → α
                                                        ⊢ Eq (iSup s) Bot.bot
                                                      -/
theorem iSup_false {s : False → α} : iSup s = ⊥ := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : CompleteLattice α
                                                        s : False → α
                                                        ⊢ Eq (iInf s) Top.top
                                                      -/
theorem iInf_false {s : False → α} : iInf s = ⊤ := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem iSup_true {s : True → α} : iSup s = s trivial :=
  iSup_pos trivial


theorem iInf_true {s : True → α} : iInf s = s trivial :=
  iInf_pos trivial


@[simp]
theorem iSup_exists {p : ι → Prop} {f : Exists p → α} : ⨆ x, f x = ⨆ (i) (h), f ⟨i, h⟩ :=
  le_antisymm (iSup_le fun ⟨i, h⟩ => @le_iSup₂ _ _ _ _ (fun _ _ => _) i h)
    (iSup₂_le fun _ _ => le_iSup _ _)


@[simp]
theorem iInf_exists {p : ι → Prop} {f : Exists p → α} : ⨅ x, f x = ⨅ (i) (h), f ⟨i, h⟩ :=
  @iSup_exists αᵒᵈ _ _ _ _


theorem iSup_and {p q : Prop} {s : p ∧ q → α} : iSup s = ⨆ (h₁) (h₂), s ⟨h₁, h₂⟩ :=
  le_antisymm (iSup_le fun ⟨i, h⟩ => @le_iSup₂ _ _ _ _ (fun _ _ => _) i h)
    (iSup₂_le fun _ _ => le_iSup _ _)


theorem iInf_and {p q : Prop} {s : p ∧ q → α} : iInf s = ⨅ (h₁) (h₂), s ⟨h₁, h₂⟩ :=
  @iSup_and αᵒᵈ _ _ _ _


/-- The symmetric case of `iSup_and`, useful for rewriting into a supremum over a conjunction -/
theorem iSup_and' {p q : Prop} {s : p → q → α} :
    ⨆ (h₁ : p) (h₂ : q), s h₁ h₂ = ⨆ h : p ∧ q, s h.1 h.2 :=
  Eq.symm iSup_and


/-- The symmetric case of `iInf_and`, useful for rewriting into an infimum over a conjunction -/
theorem iInf_and' {p q : Prop} {s : p → q → α} :
    ⨅ (h₁ : p) (h₂ : q), s h₁ h₂ = ⨅ h : p ∧ q, s h.1 h.2 :=
  Eq.symm iInf_and


theorem iSup_or {p q : Prop} {s : p ∨ q → α} :
    ⨆ x, s x = (⨆ i, s (Or.inl i)) ⊔ ⨆ j, s (Or.inr j) :=
  le_antisymm
    (iSup_le fun i =>
      match i with
      | Or.inl _ => le_sup_of_le_left <| le_iSup (fun _ => s _) _
      | Or.inr _ => le_sup_of_le_right <| le_iSup (fun _ => s _) _)
    (sup_le (iSup_comp_le _ _) (iSup_comp_le _ _))


theorem iInf_or {p q : Prop} {s : p ∨ q → α} :
    ⨅ x, s x = (⨅ i, s (Or.inl i)) ⊓ ⨅ j, s (Or.inr j) :=
  @iSup_or αᵒᵈ _ _ _ _


theorem iSup_dite (f : ∀ i, p i → α) (g : ∀ i, ¬p i → α) :
    ⨆ i, (if h : p i then f i h else g i h) = (⨆ (i) (h : p i), f i h) ⊔ ⨆ (i) (h : ¬p i),
    g i h := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    p : ι → Prop
    inst✝ : DecidablePred p
    f : (i : ι) → p i → α
    g : (i : ι) → Not (p i) → α
    ⊢ Eq (iSup fun i => dite (p i) (fun h => f i h) fun h => g i h) (Max.max (iSup …
  -/
  rw [← iSup_sup_eq]
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    p : ι → Prop
    inst✝ : DecidablePred p
    f : (i : ι) → p i → α
    g : (i : ι) → Not (p i) → α
    ⊢ Eq (iSup fun i => dite (p i) (fun h => f i h) fun h => g i h) (iSup fun x => …
  -/
  congr 1 with i
  /-
    case e_s.h
    α : Type u_1
    ι : Sort u_4
    inst✝¹ : CompleteLattice α
    p : ι → Prop
    inst✝ : DecidablePred p
    f : (i : ι) → p i → α
    g : (i : ι) → Not (p i) → α
    i : ι
    ⊢ Eq (dite (p i) (fun h => f i h) fun h => g i h) (Max.max (iSup fun h => f i  …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem iInf_dite (f : ∀ i, p i → α) (g : ∀ i, ¬p i → α) :
    ⨅ i, (if h : p i then f i h else g i h) = (⨅ (i) (h : p i), f i h) ⊓ ⨅ (i) (h : ¬p i), g i h :=
  iSup_dite p (show ∀ i, p i → αᵒᵈ from f) g


theorem iSup_ite (f g : ι → α) :
    ⨆ i, (if p i then f i else g i) = (⨆ (i) (_ : p i), f i) ⊔ ⨆ (i) (_ : ¬p i), g i :=
  iSup_dite _ _ _


theorem iInf_ite (f g : ι → α) :
    ⨅ i, (if p i then f i else g i) = (⨅ (i) (_ : p i), f i) ⊓ ⨅ (i) (_ : ¬p i), g i :=
  iInf_dite _ _ _


theorem iSup_range {g : β → α} {f : ι → β} : ⨆ b ∈ range f, g b = ⨆ i, g (f i) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝ : CompleteLattice α
    g : β → α
    f : ι → β
    ⊢ Eq (iSup fun b => iSup fun h => g b) (iSup fun i => g (f i))
  -/
  rw [← iSup_subtype'', iSup_range']
  /-
    🎉 no goals
  -/


theorem iInf_range : ∀ {g : β → α} {f : ι → β}, ⨅ b ∈ range f, g b = ⨅ i, g (f i) :=
  @iSup_range αᵒᵈ _ _ _


theorem sSup_image {s : Set β} {f : β → α} : sSup (f '' s) = ⨆ a ∈ s, f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    s : Set β
    f : β → α
    ⊢ Eq (SupSet.sSup (Set.image f s)) (iSup fun a => iSup fun h => f a)
  -/
  rw [← iSup_subtype'', sSup_image']
  /-
    🎉 no goals
  -/


theorem sInf_image {s : Set β} {f : β → α} : sInf (f '' s) = ⨅ a ∈ s, f a :=
  @sSup_image αᵒᵈ _ _ _ _


theorem OrderIso.map_sSup_eq_sSup_symm_preimage [CompleteLattice β] (f : α ≃o β) (s : Set α) :
    f (sSup s) = sSup (f.symm ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : OrderIso α β
    s : Set α
    ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.preimage (⇑f.symm) s))
  -/
  rw [map_sSup, ← sSup_image, f.image_eq_preimage]
  /-
    🎉 no goals
  -/


theorem OrderIso.map_sInf_eq_sInf_symm_preimage [CompleteLattice β] (f : α ≃o β) (s : Set α) :
    f (sInf s) = sInf (f.symm ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : OrderIso α β
    s : Set α
    ⊢ Eq (f (InfSet.sInf s)) (InfSet.sInf (Set.preimage (⇑f.symm) s))
  -/
  rw [map_sInf, ← sInf_image, f.image_eq_preimage]
  /-
    🎉 no goals
  -/

/-
### iSup and iInf under set constructions
-/

                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝ : CompleteLattice α
                                                                       f : β → α
                                                                       ⊢ Eq (iSup fun x => iSup fun h => f x) Bot.bot
                                                                     -/
theorem iSup_emptyset {f : β → α} : ⨆ x ∈ (∅ : Set β), f x = ⊥ := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝ : CompleteLattice α
                                                                       f : β → α
                                                                       ⊢ Eq (iInf fun x => iInf fun h => f x) Top.top
                                                                     -/
theorem iInf_emptyset {f : β → α} : ⨅ x ∈ (∅ : Set β), f x = ⊤ := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             inst✝ : CompleteLattice α
                                                                             f : β → α
                                                                             ⊢ Eq (iSup fun x => iSup fun h => f x) (iSup fun x => f x)
                                                                           -/
theorem iSup_univ {f : β → α} : ⨆ x ∈ (univ : Set β), f x = ⨆ x, f x := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             inst✝ : CompleteLattice α
                                                                             f : β → α
                                                                             ⊢ Eq (iInf fun x => iInf fun h => f x) (iInf fun x => f x)
                                                                           -/
theorem iInf_univ {f : β → α} : ⨅ x ∈ (univ : Set β), f x = ⨅ x, f x := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem iSup_union {f : β → α} {s t : Set β} :
    ⨆ x ∈ s ∪ t, f x = (⨆ x ∈ s, f x) ⊔ ⨆ x ∈ t, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    s t : Set β
    ⊢ Eq (iSup fun x => iSup fun h => f x) (Max.max (iSup fun x => iSup fun h => f …
  -/
  simp_rw [mem_union, iSup_or, iSup_sup_eq]
  /-
    🎉 no goals
  -/


theorem iInf_union {f : β → α} {s t : Set β} : ⨅ x ∈ s ∪ t, f x = (⨅ x ∈ s, f x) ⊓ ⨅ x ∈ t, f x :=
  @iSup_union αᵒᵈ _ _ _ _ _


theorem iSup_split (f : β → α) (p : β → Prop) :
    ⨆ i, f i = (⨆ (i) (_ : p i), f i) ⊔ ⨆ (i) (_ : ¬p i), f i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    p : β → Prop
    ⊢ Eq (iSup fun i => f i) (Max.max (iSup fun i => iSup fun x => f i) (iSup fun  …
  -/
  simpa [Classical.em] using @iSup_union _ _ _ f { i | p i } { i | ¬p i }
  /-
    🎉 no goals
  -/


theorem iInf_split :
    ∀ (f : β → α) (p : β → Prop), ⨅ i, f i = (⨅ (i) (_ : p i), f i) ⊓ ⨅ (i) (_ : ¬p i), f i :=
  @iSup_split αᵒᵈ _ _


theorem iSup_split_single (f : β → α) (i₀ : β) : ⨆ i, f i = f i₀ ⊔ ⨆ (i) (_ : i ≠ i₀), f i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    i₀ : β
    ⊢ Eq (iSup fun i => f i) (Max.max (f i₀) (iSup fun i => iSup fun x => f i))
  -/
  convert iSup_split f (fun i => i = i₀)
  /-
    case h.e'_3.h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    i₀ : β
    ⊢ Eq (f i₀) (iSup fun i => iSup fun x => f i)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iInf_split_single (f : β → α) (i₀ : β) : ⨅ i, f i = f i₀ ⊓ ⨅ (i) (_ : i ≠ i₀), f i :=
  @iSup_split_single αᵒᵈ _ _ _ _


theorem iSup_le_iSup_of_subset {f : β → α} {s t : Set β} : s ⊆ t → ⨆ x ∈ s, f x ≤ ⨆ x ∈ t, f x :=
  biSup_mono


theorem iInf_le_iInf_of_subset {f : β → α} {s t : Set β} : s ⊆ t → ⨅ x ∈ t, f x ≤ ⨅ x ∈ s, f x :=
  biInf_mono


theorem iSup_insert {f : β → α} {s : Set β} {b : β} :
    ⨆ x ∈ insert b s, f x = f b ⊔ ⨆ x ∈ s, f x :=
  Eq.trans iSup_union <| congr_arg (fun x => x ⊔ ⨆ x ∈ s, f x) iSup_iSup_eq_left


theorem iInf_insert {f : β → α} {s : Set β} {b : β} :
    ⨅ x ∈ insert b s, f x = f b ⊓ ⨅ x ∈ s, f x :=
  Eq.trans iInf_union <| congr_arg (fun x => x ⊓ ⨅ x ∈ s, f x) iInf_iInf_eq_left


                                                                                          /-
                                                                                            α : Type u_1
                                                                                            β : Type u_2
                                                                                            inst✝ : CompleteLattice α
                                                                                            f : β → α
                                                                                            b : β
                                                                                            ⊢ Eq (iSup fun x => iSup fun h => f x) (f b)
                                                                                          -/
theorem iSup_singleton {f : β → α} {b : β} : ⨆ x ∈ (singleton b : Set β), f x = f b := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


                                                                                          /-
                                                                                            α : Type u_1
                                                                                            β : Type u_2
                                                                                            inst✝ : CompleteLattice α
                                                                                            f : β → α
                                                                                            b : β
                                                                                            ⊢ Eq (iInf fun x => iInf fun h => f x) (f b)
                                                                                          -/
theorem iInf_singleton {f : β → α} {b : β} : ⨅ x ∈ (singleton b : Set β), f x = f b := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem iSup_pair {f : β → α} {a b : β} : ⨆ x ∈ ({a, b} : Set β), f x = f a ⊔ f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    a b : β
    ⊢ Eq (iSup fun x => iSup fun h => f x) (Max.max (f a) (f b))
  -/
  rw [iSup_insert, iSup_singleton]
  /-
    🎉 no goals
  -/


theorem iInf_pair {f : β → α} {a b : β} : ⨅ x ∈ ({a, b} : Set β), f x = f a ⊓ f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : β → α
    a b : β
    ⊢ Eq (iInf fun x => iInf fun h => f x) (Min.min (f a) (f b))
  -/
  rw [iInf_insert, iInf_singleton]
  /-
    🎉 no goals
  -/


theorem iSup_image {γ} {f : β → γ} {g : γ → α} {t : Set β} :
    ⨆ c ∈ f '' t, g c = ⨆ b ∈ t, g (f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    γ : Type u_8
    f : β → γ
    g : γ → α
    t : Set β
    ⊢ Eq (iSup fun c => iSup fun h => g c) (iSup fun b => iSup fun h => g (f b))
  -/
  rw [← sSup_image, ← sSup_image, ← image_comp, comp_def]
  /-
    🎉 no goals
  -/


theorem iInf_image :
    ∀ {γ} {f : β → γ} {g : γ → α} {t : Set β}, ⨅ c ∈ f '' t, g c = ⨅ b ∈ t, g (f b) :=
  @iSup_image αᵒᵈ _ _


theorem iSup_extend_bot {e : ι → β} (he : Injective e) (f : ι → α) :
    ⨆ j, extend e f ⊥ j = ⨆ i, f i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝ : CompleteLattice α
    e : ι → β
    he : Function.Injective e
    f : ι → α
    ⊢ Eq (iSup fun j => Function.extend e f Bot.bot j) (iSup fun i => f i)
  -/
  rw [iSup_split _ fun j => ∃ i, e i = j]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝ : CompleteLattice α
    e : ι → β
    he : Function.Injective e
    f : ι → α
    ⊢ Eq (Max.max (iSup fun i => iSup fun x => Function.extend e f Bot.bot i) (iSu …
  -/
  simp +contextual [he.extend_apply, extend_apply', @iSup_comm _ β ι]
  /-
    🎉 no goals
  -/


theorem iInf_extend_top {e : ι → β} (he : Injective e) (f : ι → α) :
    ⨅ j, extend e f ⊤ j = iInf f :=
  @iSup_extend_bot αᵒᵈ _ _ _ _ he _


theorem iSup_of_empty' {α ι} [SupSet α] [IsEmpty ι] (f : ι → α) : iSup f = sSup (∅ : Set α) :=
  congr_arg sSup (range_eq_empty f)


theorem iInf_of_isEmpty {α ι} [InfSet α] [IsEmpty ι] (f : ι → α) : iInf f = sInf (∅ : Set α) :=
  congr_arg sInf (range_eq_empty f)


theorem iSup_of_empty [IsEmpty ι] (f : ι → α) : iSup f = ⊥ :=
  (iSup_of_empty' f).trans sSup_empty


theorem iInf_of_empty [IsEmpty ι] (f : ι → α) : iInf f = ⊤ :=
  @iSup_of_empty αᵒᵈ _ _ _ f


theorem iSup_bool_eq {f : Bool → α} : ⨆ b : Bool, f b = f true ⊔ f false := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Bool → α
    ⊢ Eq (iSup fun b => f b) (Max.max (f Bool.true) (f Bool.false))
  -/
  rw [iSup, Bool.range_eq, sSup_pair, sup_comm]
  /-
    🎉 no goals
  -/


theorem iInf_bool_eq {f : Bool → α} : ⨅ b : Bool, f b = f true ⊓ f false :=
  @iSup_bool_eq αᵒᵈ _ _


theorem sup_eq_iSup (x y : α) : x ⊔ y = ⨆ b : Bool, cond b x y := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    x y : α
    ⊢ Eq (Max.max x y) (iSup fun b => cond b x y)
  -/
  rw [iSup_bool_eq, Bool.cond_true, Bool.cond_false]
  /-
    🎉 no goals
  -/


theorem inf_eq_iInf (x y : α) : x ⊓ y = ⨅ b : Bool, cond b x y :=
  @sup_eq_iSup αᵒᵈ _ _ _


theorem isGLB_biInf {s : Set β} {f : β → α} : IsGLB (f '' s) (⨅ x ∈ s, f x) := by
  simpa only [range_comp, Subtype.range_coe, iInf_subtype'] using
    @isGLB_iInf α s _ (f ∘ fun x => (x : β))


theorem isLUB_biSup {s : Set β} {f : β → α} : IsLUB (f '' s) (⨆ x ∈ s, f x) := by
  simpa only [range_comp, Subtype.range_coe, iSup_subtype'] using
    @isLUB_iSup α s _ (f ∘ fun x => (x : β))


theorem iSup_sigma {p : β → Type*} {f : Sigma p → α} : ⨆ x, f x = ⨆ (i) (j), f ⟨i, j⟩ :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    inst✝ : CompleteLattice α
                                    p : β → Type u_8
                                    f : Sigma p → α
                                    c : α
                                    ⊢ Iff (LE.le (iSup fun x => f x) c) (LE.le (iSup fun i => iSup fun j => f ⟨i,  …
                                  -/
  eq_of_forall_ge_iff fun c => by simp only [iSup_le_iff, Sigma.forall]
                                  /-
                                    🎉 no goals
                                  -/


theorem iInf_sigma {p : β → Type*} {f : Sigma p → α} : ⨅ x, f x = ⨅ (i) (j), f ⟨i, j⟩ :=
  @iSup_sigma αᵒᵈ _ _ _ _


lemma iSup_sigma' {κ : β → Type*} (f : ∀ i, κ i → α) :
    (⨆ i, ⨆ j, f i j) = ⨆ x : Σ i, κ i, f x.1 x.2 := (iSup_sigma (f := fun x ↦ f x.1 x.2)).symm


lemma iInf_sigma' {κ : β → Type*} (f : ∀ i, κ i → α) :
    (⨅ i, ⨅ j, f i j) = ⨅ x : Σ i, κ i, f x.1 x.2 := (iInf_sigma (f := fun x ↦ f x.1 x.2)).symm


lemma iSup_psigma {ι : Sort*} {κ : ι → Sort*} (f : (Σ' i, κ i) → α) :
    ⨆ ij, f ij = ⨆ i, ⨆ j, f ⟨i, j⟩ :=
                                 /-
                                   α : Type u_1
                                   inst✝ : CompleteLattice α
                                   ι : Sort u_8
                                   κ : ι → Sort u_9
                                   f : (PSigma fun i => κ i) → α
                                   c : α
                                   ⊢ Iff (LE.le (iSup fun ij => f ij) c) (LE.le (iSup fun i => iSup fun j => f ⟨i …
                                 -/
  eq_of_forall_ge_iff fun c ↦ by simp only [iSup_le_iff, PSigma.forall]
                                 /-
                                   🎉 no goals
                                 -/


lemma iInf_psigma {ι : Sort*} {κ : ι → Sort*} (f : (Σ' i, κ i) → α) :
    ⨅ ij, f ij = ⨅ i, ⨅ j, f ⟨i, j⟩ :=
                                 /-
                                   α : Type u_1
                                   inst✝ : CompleteLattice α
                                   ι : Sort u_8
                                   κ : ι → Sort u_9
                                   f : (PSigma fun i => κ i) → α
                                   c : α
                                   ⊢ Iff (LE.le c (iInf fun ij => f ij)) (LE.le c (iInf fun i => iInf fun j => f  …
                                 -/
  eq_of_forall_le_iff fun c ↦ by simp only [le_iInf_iff, PSigma.forall]
                                 /-
                                   🎉 no goals
                                 -/


lemma iSup_psigma' {ι : Sort*} {κ : ι → Sort*} (f : ∀ i, κ i → α) :
    (⨆ i, ⨆ j, f i j) = ⨆ ij : Σ' i, κ i, f ij.1 ij.2 := (iSup_psigma fun x ↦ f x.1 x.2).symm


lemma iInf_psigma' {ι : Sort*} {κ : ι → Sort*} (f : ∀ i, κ i → α) :
    (⨅ i, ⨅ j, f i j) = ⨅ ij : Σ' i, κ i, f ij.1 ij.2 := (iInf_psigma fun x ↦ f x.1 x.2).symm


theorem iSup_prod {f : β × γ → α} : ⨆ x, f x = ⨆ (i) (j), f (i, j) :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    γ : Type u_3
                                    inst✝ : CompleteLattice α
                                    f : Prod β γ → α
                                    c : α
                                    ⊢ Iff (LE.le (iSup fun x => f x) c) (LE.le (iSup fun i => iSup fun j => f { fs …
                                  -/
  eq_of_forall_ge_iff fun c => by simp only [iSup_le_iff, Prod.forall]
                                  /-
                                    🎉 no goals
                                  -/


theorem iInf_prod {f : β × γ → α} : ⨅ x, f x = ⨅ (i) (j), f (i, j) :=
  @iSup_prod αᵒᵈ _ _ _ _


lemma iSup_prod' (f : β → γ → α) : (⨆ i, ⨆ j, f i j) = ⨆ x : β × γ, f x.1 x.2 :=
(iSup_prod (f := fun x ↦ f x.1 x.2)).symm


lemma iInf_prod' (f : β → γ → α) : (⨅ i, ⨅ j, f i j) = ⨅ x : β × γ, f x.1 x.2 :=
(iInf_prod (f := fun x ↦ f x.1 x.2)).symm


theorem biSup_prod {f : β × γ → α} {s : Set β} {t : Set γ} :
    ⨆ x ∈ s ×ˢ t, f x = ⨆ (a ∈ s) (b ∈ t), f (a, b) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : CompleteLattice α
    f : Prod β γ → α
    s : Set β
    t : Set γ
    ⊢ Eq (iSup fun x => iSup fun h => f x) (iSup fun a => iSup fun h => iSup fun b …
  -/
  simp_rw [iSup_prod, mem_prod, iSup_and]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : CompleteLattice α
    f : Prod β γ → α
    s : Set β
    t : Set γ
    ⊢ Eq (iSup fun i => iSup fun j => iSup fun h₁ => iSup fun h₂ => f { fst := i,  …
  -/
  exact iSup_congr fun _ => iSup_comm
  /-
    🎉 no goals
  -/


theorem biInf_prod {f : β × γ → α} {s : Set β} {t : Set γ} :
    ⨅ x ∈ s ×ˢ t, f x = ⨅ (a ∈ s) (b ∈ t), f (a, b) :=
  @biSup_prod αᵒᵈ _ _ _ _ _ _


theorem iSup_image2 {γ δ} (f : β → γ → δ) (s : Set β) (t : Set γ) (g : δ → α) :
    ⨆ d ∈ image2 f s t, g d = ⨆ b ∈ s, ⨆ c ∈ t, g (f b c) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    γ : Type u_8
    δ : Type u_9
    f : β → γ → δ
    s : Set β
    t : Set γ
    g : δ → α
    ⊢ Eq (iSup fun d => iSup fun h => g d) (iSup fun b => iSup fun h => iSup fun c …
  -/
  rw [← image_prod, iSup_image, biSup_prod]
  /-
    🎉 no goals
  -/


theorem iInf_image2 {γ δ} (f : β → γ → δ) (s : Set β) (t : Set γ) (g : δ → α) :
    ⨅ d ∈ image2 f s t, g d = ⨅ b ∈ s, ⨅ c ∈ t, g (f b c) :=
  iSup_image2 f s t (toDual ∘ g)


theorem iSup_sum {f : β ⊕ γ → α} : ⨆ x, f x = (⨆ i, f (Sum.inl i)) ⊔ ⨆ j, f (Sum.inr j) :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    γ : Type u_3
                                    inst✝ : CompleteLattice α
                                    f : Sum β γ → α
                                    c : α
                                    ⊢ Iff (LE.le (iSup fun x => f x) c) (LE.le (Max.max (iSup fun i => f (Sum.inl  …
                                  -/
  eq_of_forall_ge_iff fun c => by simp only [sup_le_iff, iSup_le_iff, Sum.forall]
                                  /-
                                    🎉 no goals
                                  -/


theorem iInf_sum {f : β ⊕ γ → α} : ⨅ x, f x = (⨅ i, f (Sum.inl i)) ⊓ ⨅ j, f (Sum.inr j) :=
  @iSup_sum αᵒᵈ _ _ _ _


theorem iSup_option (f : Option β → α) : ⨆ o, f o = f none ⊔ ⨆ b, f (Option.some b) :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    inst✝ : CompleteLattice α
                                    f : Option β → α
                                    c : α
                                    ⊢ Iff (LE.le (iSup fun o => f o) c) (LE.le (Max.max (f Option.none) (iSup fun  …
                                  -/
  eq_of_forall_ge_iff fun c => by simp only [iSup_le_iff, sup_le_iff, Option.forall]
                                  /-
                                    🎉 no goals
                                  -/


theorem iInf_option (f : Option β → α) : ⨅ o, f o = f none ⊓ ⨅ b, f (Option.some b) :=
  @iSup_option αᵒᵈ _ _ _


/-- A version of `iSup_option` useful for rewriting right-to-left. -/
theorem iSup_option_elim (a : α) (f : β → α) : ⨆ o : Option β, o.elim a f = a ⊔ ⨆ b, f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    a : α
    f : β → α
    ⊢ Eq (iSup fun o => o.elim a f) (Max.max a (iSup fun b => f b))
  -/
  simp [iSup_option]
  /-
    🎉 no goals
  -/


/-- A version of `iInf_option` useful for rewriting right-to-left. -/
theorem iInf_option_elim (a : α) (f : β → α) : ⨅ o : Option β, o.elim a f = a ⊓ ⨅ b, f b :=
  @iSup_option_elim αᵒᵈ _ _ _ _


/-- When taking the supremum of `f : ι → α`, the elements of `ι` on which `f` gives `⊥` can be
dropped, without changing the result. -/
@[simp]
theorem iSup_ne_bot_subtype (f : ι → α) : ⨆ i : { i // f i ≠ ⊥ }, f i = ⨆ i, f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLattice α
    f : ι → α
    ⊢ Eq (iSup fun i => f ↑i) (iSup fun i => f i)
  -/
  by_cases htriv : ∀ i, f i = ⊥
    /-
      case pos
      α : Type u_1
      ι : Sort u_4
      inst✝ : CompleteLattice α
      f : ι → α
      htriv : ∀ (i : ι), Eq (f i) Bot.bot
      ⊢ Eq (iSup fun i => f ↑i) (iSup fun i => f i)
    -/
  · simp only [iSup_bot, (funext htriv : f = _)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLattice α
    f : ι → α
    htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
    ⊢ Eq (iSup fun i => f ↑i) (iSup fun i => f i)
  -/
  refine (iSup_comp_le f _).antisymm (iSup_mono' fun i => ?_)
  /-
    case neg
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLattice α
    f : ι → α
    htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
    i : ι
    ⊢ Exists fun i' => LE.le (f i) (f ↑i')
  -/
  by_cases hi : f i = ⊥
    /-
      case pos
      α : Type u_1
      ι : Sort u_4
      inst✝ : CompleteLattice α
      f : ι → α
      htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
      i : ι
      hi : Eq (f i) Bot.bot
      ⊢ Exists fun i' => LE.le (f i) (f ↑i')
    -/
  · rw [hi]
    /-
      case pos
      α : Type u_1
      ι : Sort u_4
      inst✝ : CompleteLattice α
      f : ι → α
      htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
      i : ι
      hi : Eq (f i) Bot.bot
      ⊢ Exists fun i' => LE.le Bot.bot (f ↑i')
    -/
    obtain ⟨i₀, hi₀⟩ := not_forall.mp htriv
    /-
      case pos.intro
      α : Type u_1
      ι : Sort u_4
      inst✝ : CompleteLattice α
      f : ι → α
      htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
      i : ι
      hi : Eq (f i) Bot.bot
      i₀ : ι
      hi₀ : Not (Eq (f i₀) Bot.bot)
      ⊢ Exists fun i' => LE.le Bot.bot (f ↑i')
    -/
    exact ⟨⟨i₀, hi₀⟩, bot_le⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      ι : Sort u_4
      inst✝ : CompleteLattice α
      f : ι → α
      htriv : Not (∀ (i : ι), Eq (f i) Bot.bot)
      i : ι
      hi : Not (Eq (f i) Bot.bot)
      ⊢ Exists fun i' => LE.le (f i) (f ↑i')
    -/
  · exact ⟨⟨i, hi⟩, rfl.le⟩
    /-
      🎉 no goals
    -/


/-- When taking the infimum of `f : ι → α`, the elements of `ι` on which `f` gives `⊤` can be
dropped, without changing the result. -/
theorem iInf_ne_top_subtype (f : ι → α) : ⨅ i : { i // f i ≠ ⊤ }, f i = ⨅ i, f i :=
  @iSup_ne_bot_subtype αᵒᵈ ι _ f


theorem sSup_image2 {f : β → γ → α} {s : Set β} {t : Set γ} :
                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           γ : Type u_3
                                                           inst✝ : CompleteLattice α
                                                           f : β → γ → α
                                                           s : Set β
                                                           t : Set γ
                                                           ⊢ Eq (SupSet.sSup (Set.image2 f s t)) (iSup fun a => iSup fun h => iSup fun b  …
                                                         -/
    sSup (image2 f s t) = ⨆ (a ∈ s) (b ∈ t), f a b := by rw [← image_prod, sSup_image, biSup_prod]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem sInf_image2 {f : β → γ → α} {s : Set β} {t : Set γ} :
                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           γ : Type u_3
                                                           inst✝ : CompleteLattice α
                                                           f : β → γ → α
                                                           s : Set β
                                                           t : Set γ
                                                           ⊢ Eq (InfSet.sInf (Set.image2 f s t)) (iInf fun a => iInf fun h => iInf fun b  …
                                                         -/
    sInf (image2 f s t) = ⨅ (a ∈ s) (b ∈ t), f a b := by rw [← image_prod, sInf_image, biInf_prod]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem iSup_ge_eq_iSup_nat_add (u : ℕ → α) (n : ℕ) : ⨆ i ≥ n, u i = ⨆ i, u (i + n) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (iSup fun i => iSup fun h => u i) (iSup fun i => u (HAdd.hAdd i n))
  -/
  apply le_antisymm <;> simp only [iSup_le_iff]
    /-
      case a
      α : Type u_1
      inst✝ : CompleteLattice α
      u : Nat → α
      n : Nat
      ⊢ ∀ (i : Nat), GE.ge i n → LE.le (u i) (iSup fun i => u (HAdd.hAdd i n))
    -/
  · refine fun i hi => le_sSup ⟨i - n, ?_⟩
    /-
      case a
      α : Type u_1
      inst✝ : CompleteLattice α
      u : Nat → α
      n i : Nat
      hi : GE.ge i n
      ⊢ Eq ((fun i => u (HAdd.hAdd i n)) (HSub.hSub i n)) (u i)
    -/
    dsimp only
    /-
      case a
      α : Type u_1
      inst✝ : CompleteLattice α
      u : Nat → α
      n i : Nat
      hi : GE.ge i n
      ⊢ Eq (u (HAdd.hAdd (HSub.hSub i n) n)) (u i)
    -/
    rw [Nat.sub_add_cancel hi]
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : CompleteLattice α
      u : Nat → α
      n : Nat
      ⊢ ∀ (i : Nat), LE.le (u (HAdd.hAdd i n)) (iSup fun i => iSup fun h => u i)
    -/
  · exact fun i => le_sSup ⟨i + n, iSup_pos (Nat.le_add_left _ _)⟩
    /-
      🎉 no goals
    -/


theorem iInf_ge_eq_iInf_nat_add (u : ℕ → α) (n : ℕ) : ⨅ i ≥ n, u i = ⨅ i, u (i + n) :=
  @iSup_ge_eq_iSup_nat_add αᵒᵈ _ _ _


theorem Monotone.iSup_nat_add {f : ℕ → α} (hf : Monotone f) (k : ℕ) : ⨆ n, f (n + k) = ⨆ n, f n :=
  le_antisymm (iSup_le fun i => le_iSup _ (i + k)) <| iSup_mono fun i => hf <| Nat.le_add_right i k


theorem Antitone.iInf_nat_add {f : ℕ → α} (hf : Antitone f) (k : ℕ) : ⨅ n, f (n + k) = ⨅ n, f n :=
  hf.dual_right.iSup_nat_add k

-- Porting note: the linter doesn't like this being marked as `@[simp]`,
-- saying that it doesn't work when called on its LHS.
-- Mysteriously, it *does* work. Nevertheless, per
-- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/complete_lattice.20and.20has_sup/near/316497982
-- "the subterm ?f (i + ?k) produces an ugly higher-order unification problem."
-- @[simp]

theorem iSup_iInf_ge_nat_add (f : ℕ → α) (k : ℕ) :
    ⨆ n, ⨅ i ≥ n, f (i + k) = ⨆ n, ⨅ i ≥ n, f i := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → α
    k : Nat
    ⊢ Eq (iSup fun n => iInf fun i => iInf fun h => f (HAdd.hAdd i k)) (iSup fun n …
  -/
  have hf : Monotone fun n => ⨅ i ≥ n, f i := fun n m h => biInf_mono fun i => h.trans
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → α
    k : Nat
    hf : Monotone fun n => iInf fun i => iInf fun h => f i
    ⊢ Eq (iSup fun n => iInf fun i => iInf fun h => f (HAdd.hAdd i k)) (iSup fun n …
  -/
  rw [← Monotone.iSup_nat_add hf k]
    /-
      α : Type u_1
      inst✝ : CompleteLattice α
      f : Nat → α
      k : Nat
      hf : Monotone fun n => iInf fun i => iInf fun h => f i
      ⊢ Eq (iSup fun n => iInf fun i => iInf fun h => f (HAdd.hAdd i k)) (iSup fun n …
    -/
  · simp_rw [iInf_ge_eq_iInf_nat_add, ← Nat.add_assoc]
    /-
      🎉 no goals
    -/

-- Porting note: removing `@[simp]`, see discussion on `iSup_iInf_ge_nat_add`.
-- @[simp]

theorem iInf_iSup_ge_nat_add :
    ∀ (f : ℕ → α) (k : ℕ), ⨅ n, ⨆ i ≥ n, f (i + k) = ⨅ n, ⨆ i ≥ n, f i :=
  @iSup_iInf_ge_nat_add αᵒᵈ _


theorem sup_iSup_nat_succ (u : ℕ → α) : (u 0 ⊔ ⨆ i, u (i + 1)) = ⨆ i, u i :=
  calc
    (u 0 ⊔ ⨆ i, u (i + 1)) = ⨆ x ∈ {0} ∪ range Nat.succ, u x := by
      /-
        α : Type u_1
        inst✝ : CompleteLattice α
        u : Nat → α
        ⊢ Eq (Max.max (u 0) (iSup fun i => u (HAdd.hAdd i 1))) (iSup fun x => iSup fun …
      -/
      { rw [iSup_union, iSup_singleton, iSup_range] }
      /-
        🎉 no goals
      -/
                       /-
                         α : Type u_1
                         inst✝ : CompleteLattice α
                         u : Nat → α
                         ⊢ Eq (iSup fun x => iSup fun h => u x) (iSup fun i => u i)
                       -/
    _ = ⨆ i, u i := by rw [Nat.zero_union_range_succ, iSup_univ]
                       /-
                         🎉 no goals
                       -/


theorem inf_iInf_nat_succ (u : ℕ → α) : (u 0 ⊓ ⨅ i, u (i + 1)) = ⨅ i, u i :=
  @sup_iSup_nat_succ αᵒᵈ _ u


theorem iInf_nat_gt_zero_eq (f : ℕ → α) : ⨅ i > 0, f i = ⨅ i, f (i + 1) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → α
    ⊢ Eq (iInf fun i => iInf fun h => f i) (iInf fun i => f (HAdd.hAdd i 1))
  -/
  rw [← iInf_range, Nat.range_succ]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → α
    ⊢ Eq (iInf fun i => iInf fun h => f i) (iInf fun b => iInf fun h => f b)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iSup_nat_gt_zero_eq (f : ℕ → α) : ⨆ i > 0, f i = ⨆ i, f (i + 1) :=
  @iInf_nat_gt_zero_eq αᵒᵈ _ f


theorem iSup_eq_top (f : ι → α) : iSup f = ⊤ ↔ ∀ b < ⊤, ∃ i, b < f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLinearOrder α
    f : ι → α
    ⊢ Iff (Eq (iSup f) Top.top) (∀ (b : α), LT.lt b Top.top → Exists fun i => LT.l …
  -/
  simp only [← sSup_range, sSup_eq_top, Set.exists_range_iff]
  /-
    🎉 no goals
  -/


theorem iInf_eq_bot (f : ι → α) : iInf f = ⊥ ↔ ∀ b > ⊥, ∃ i, f i < b := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : CompleteLinearOrder α
    f : ι → α
    ⊢ Iff (Eq (iInf f) Bot.bot) (∀ (b : α), GT.gt b Bot.bot → Exists fun i => LT.l …
  -/
  simp only [← sInf_range, sInf_eq_bot, Set.exists_range_iff]
  /-
    🎉 no goals
  -/


lemma iSup₂_eq_top (f : ∀ i, κ i → α) : ⨆ i, ⨆ j, f i j = ⊤ ↔ ∀ b < ⊤, ∃ i j, b < f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLinearOrder α
    f : (i : ι) → κ i → α
    ⊢ Iff (Eq (iSup fun i => iSup fun j => f i j) Top.top) (∀ (b : α), LT.lt b Top …
  -/
  simp_rw [iSup_psigma', iSup_eq_top, PSigma.exists]
  /-
    🎉 no goals
  -/


lemma iInf₂_eq_bot (f : ∀ i, κ i → α) : ⨅ i, ⨅ j, f i j = ⊥ ↔ ∀ b > ⊥, ∃ i j, f i j < b := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_6
    inst✝ : CompleteLinearOrder α
    f : (i : ι) → κ i → α
    ⊢ Iff (Eq (iInf fun i => iInf fun j => f i j) Bot.bot) (∀ (b : α), GT.gt b Bot …
  -/
  simp_rw [iInf_psigma', iInf_eq_bot, PSigma.exists]
  /-
    🎉 no goals
  -/


instance Prop.instCompleteLattice : CompleteLattice Prop where
  __ := Prop.instBoundedOrder
  __ := Prop.instDistribLattice
  sSup s := ∃ a ∈ s, a
  le_sSup _ a h p := ⟨a, h, p⟩
  sSup_le _ _ h := fun ⟨b, h', p⟩ => h b h' p
  sInf s := ∀ a, a ∈ s → a
  sInf_le _ a h p := p a h
  le_sInf _ _ h p b hb := h b hb p


noncomputable instance Prop.instCompleteLinearOrder : CompleteLinearOrder Prop where
  __ := Prop.instCompleteLattice
  __ := Prop.linearOrder
  __ := BooleanAlgebra.toBiheytingAlgebra


@[simp]
theorem sSup_Prop_eq {s : Set Prop} : sSup s = ∃ p ∈ s, p :=
  rfl


@[simp]
theorem sInf_Prop_eq {s : Set Prop} : sInf s = ∀ p ∈ s, p :=
  rfl


@[simp]
theorem iSup_Prop_eq {p : ι → Prop} : ⨆ i, p i = ∃ i, p i :=
  le_antisymm (fun ⟨_, ⟨i, (eq : p i = _)⟩, hq⟩ => ⟨i, eq.symm ▸ hq⟩) fun ⟨i, hi⟩ =>
    ⟨p i, ⟨i, rfl⟩, hi⟩


@[simp]
theorem iInf_Prop_eq {p : ι → Prop} : ⨅ i, p i = ∀ i, p i :=
  le_antisymm (fun h i => h _ ⟨i, rfl⟩) fun h _ ⟨i, Eq⟩ => Eq ▸ h i


instance Pi.supSet {α : Type*} {β : α → Type*} [∀ i, SupSet (β i)] : SupSet (∀ i, β i) :=
  ⟨fun s i => ⨆ f : s, (f : ∀ i, β i) i⟩


instance Pi.infSet {α : Type*} {β : α → Type*} [∀ i, InfSet (β i)] : InfSet (∀ i, β i) :=
  ⟨fun s i => ⨅ f : s, (f : ∀ i, β i) i⟩


instance Pi.instCompleteLattice {α : Type*} {β : α → Type*} [∀ i, CompleteLattice (β i)] :
    CompleteLattice (∀ i, β i) where
  __ := instBoundedOrder
  le_sSup s f hf := fun i => le_iSup (fun f : s => (f : ∀ i, β i) i) ⟨f, hf⟩
  sInf_le s f hf := fun i => iInf_le (fun f : s => (f : ∀ i, β i) i) ⟨f, hf⟩
  sSup_le _ _ hf := fun i => iSup_le fun g => hf g g.2 i
  le_sInf _ _ hf := fun i => le_iInf fun g => hf g g.2 i


@[simp]
theorem sSup_apply {α : Type*} {β : α → Type*} [∀ i, SupSet (β i)] {s : Set (∀ a, β a)} {a : α} :
    (sSup s) a = ⨆ f : s, (f : ∀ a, β a) a :=
  rfl


@[simp]
theorem sInf_apply {α : Type*} {β : α → Type*} [∀ i, InfSet (β i)] {s : Set (∀ a, β a)} {a : α} :
    sInf s a = ⨅ f : s, (f : ∀ a, β a) a :=
  rfl


@[simp]
theorem iSup_apply {α : Type*} {β : α → Type*} {ι : Sort*} [∀ i, SupSet (β i)] {f : ι → ∀ a, β a}
    {a : α} : (⨆ i, f i) a = ⨆ i, f i a := by
  rw [iSup, sSup_apply, iSup, iSup, ← image_eq_range (fun f : ∀ i, β i => f a) (range f), ←
                 /-
                   α : Type u_8
                   β : α → Type u_9
                   ι : Sort u_10
                   inst✝ : (i : α) → SupSet (β i)
                   f : ι → (a : α) → β a
                   a : α
                   ⊢ Eq (SupSet.sSup (Set.range (Function.comp (fun f => f a) f))) (SupSet.sSup ( …
                 -/
    range_comp]; rfl
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem iInf_apply {α : Type*} {β : α → Type*} {ι : Sort*} [∀ i, InfSet (β i)] {f : ι → ∀ a, β a}
    {a : α} : (⨅ i, f i) a = ⨅ i, f i a :=
  @iSup_apply α (fun i => (β i)ᵒᵈ) _ _ _ _


theorem unary_relation_sSup_iff {α : Type*} (s : Set (α → Prop)) {a : α} :
    sSup s a ↔ ∃ r : α → Prop, r ∈ s ∧ r a := by
  /-
    α : Type u_8
    s : Set (α → Prop)
    a : α
    ⊢ Iff (SupSet.sSup s a) (Exists fun r => And (Membership.mem s r) (r a))
  -/
  rw [sSup_apply]
  /-
    α : Type u_8
    s : Set (α → Prop)
    a : α
    ⊢ Iff (iSup fun f => ↑f a) (Exists fun r => And (Membership.mem s r) (r a))
  -/
  simp [← eq_iff_iff]
  /-
    🎉 no goals
  -/


theorem unary_relation_sInf_iff {α : Type*} (s : Set (α → Prop)) {a : α} :
    sInf s a ↔ ∀ r : α → Prop, r ∈ s → r a := by
  /-
    α : Type u_8
    s : Set (α → Prop)
    a : α
    ⊢ Iff (InfSet.sInf s a) (∀ (r : α → Prop), Membership.mem s r → r a)
  -/
  rw [sInf_apply]
  /-
    α : Type u_8
    s : Set (α → Prop)
    a : α
    ⊢ Iff (iInf fun f => ↑f a) (∀ (r : α → Prop), Membership.mem s r → r a)
  -/
  simp [← eq_iff_iff]
  /-
    🎉 no goals
  -/


theorem binary_relation_sSup_iff {α β : Type*} (s : Set (α → β → Prop)) {a : α} {b : β} :
    sSup s a b ↔ ∃ r : α → β → Prop, r ∈ s ∧ r a b := by
  /-
    α : Type u_8
    β : Type u_9
    s : Set (α → β → Prop)
    a : α
    b : β
    ⊢ Iff (SupSet.sSup s a b) (Exists fun r => And (Membership.mem s r) (r a b))
  -/
  rw [sSup_apply]
  /-
    α : Type u_8
    β : Type u_9
    s : Set (α → β → Prop)
    a : α
    b : β
    ⊢ Iff (iSup (fun f => ↑f a) b) (Exists fun r => And (Membership.mem s r) (r a  …
  -/
  simp [← eq_iff_iff]
  /-
    🎉 no goals
  -/


theorem binary_relation_sInf_iff {α β : Type*} (s : Set (α → β → Prop)) {a : α} {b : β} :
    sInf s a b ↔ ∀ r : α → β → Prop, r ∈ s → r a b := by
  /-
    α : Type u_8
    β : Type u_9
    s : Set (α → β → Prop)
    a : α
    b : β
    ⊢ Iff (InfSet.sInf s a b) (∀ (r : α → β → Prop), Membership.mem s r → r a b)
  -/
  rw [sInf_apply]
  /-
    α : Type u_8
    β : Type u_9
    s : Set (α → β → Prop)
    a : α
    b : β
    ⊢ Iff (iInf (fun f => ↑f a) b) (∀ (r : α → β → Prop), Membership.mem s r → r a …
  -/
  simp [← eq_iff_iff]
  /-
    🎉 no goals
  -/


protected lemma Monotone.sSup (hs : ∀ f ∈ s, Monotone f) : Monotone (sSup s) :=
  fun _ _ h ↦ iSup_mono fun f ↦ hs f f.2 h


protected lemma Monotone.sInf (hs : ∀ f ∈ s, Monotone f) : Monotone (sInf s) :=
  fun _ _ h ↦ iInf_mono fun f ↦ hs f f.2 h


protected lemma Antitone.sSup (hs : ∀ f ∈ s, Antitone f) : Antitone (sSup s) :=
  fun _ _ h ↦ iSup_mono fun f ↦ hs f f.2 h


protected lemma Antitone.sInf (hs : ∀ f ∈ s, Antitone f) : Antitone (sInf s) :=
  fun _ _ h ↦ iInf_mono fun f ↦ hs f f.2 h


@[deprecated (since := "2024-05-29")] alias monotone_sSup_of_monotone := Monotone.sSup

@[deprecated (since := "2024-05-29")] alias monotone_sInf_of_monotone := Monotone.sInf


protected lemma Monotone.iSup (hf : ∀ i, Monotone (f i)) : Monotone (⨆ i, f i) :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Sort u_4
                      inst✝¹ : Preorder α
                      inst✝ : CompleteLattice β
                      f : ι → α → β
                      hf : ∀ (i : ι), Monotone (f i)
                      ⊢ ∀ (f_1 : α → β), Membership.mem (Set.range fun i => f i) f_1 → Monotone f_1
                    -/
  Monotone.sSup (by simpa)
                    /-
                      🎉 no goals
                    -/

protected lemma Monotone.iInf (hf : ∀ i, Monotone (f i)) : Monotone (⨅ i, f i) :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Sort u_4
                      inst✝¹ : Preorder α
                      inst✝ : CompleteLattice β
                      f : ι → α → β
                      hf : ∀ (i : ι), Monotone (f i)
                      ⊢ ∀ (f_1 : α → β), Membership.mem (Set.range fun i => f i) f_1 → Monotone f_1
                    -/
  Monotone.sInf (by simpa)
                    /-
                      🎉 no goals
                    -/

protected lemma Antitone.iSup (hf : ∀ i, Antitone (f i)) : Antitone (⨆ i, f i) :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Sort u_4
                      inst✝¹ : Preorder α
                      inst✝ : CompleteLattice β
                      f : ι → α → β
                      hf : ∀ (i : ι), Antitone (f i)
                      ⊢ ∀ (f_1 : α → β), Membership.mem (Set.range fun i => f i) f_1 → Antitone f_1
                    -/
  Antitone.sSup (by simpa)
                    /-
                      🎉 no goals
                    -/

protected lemma Antitone.iInf (hf : ∀ i, Antitone (f i)) : Antitone (⨅ i, f i) :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Sort u_4
                      inst✝¹ : Preorder α
                      inst✝ : CompleteLattice β
                      f : ι → α → β
                      hf : ∀ (i : ι), Antitone (f i)
                      ⊢ ∀ (f_1 : α → β), Membership.mem (Set.range fun i => f i) f_1 → Antitone f_1
                    -/
  Antitone.sInf (by simpa)
                    /-
                      🎉 no goals
                    -/


instance supSet [SupSet α] [SupSet β] : SupSet (α × β) :=
  ⟨fun s => (sSup (Prod.fst '' s), sSup (Prod.snd '' s))⟩


instance infSet [InfSet α] [InfSet β] : InfSet (α × β) :=
  ⟨fun s => (sInf (Prod.fst '' s), sInf (Prod.snd '' s))⟩


theorem fst_sInf [InfSet α] [InfSet β] (s : Set (α × β)) : (sInf s).fst = sInf (Prod.fst '' s) :=
  rfl


theorem snd_sInf [InfSet α] [InfSet β] (s : Set (α × β)) : (sInf s).snd = sInf (Prod.snd '' s) :=
  rfl


theorem swap_sInf [InfSet α] [InfSet β] (s : Set (α × β)) : (sInf s).swap = sInf (Prod.swap '' s) :=
  Prod.ext (congr_arg sInf <| image_comp Prod.fst swap s)
    (congr_arg sInf <| image_comp Prod.snd swap s)


theorem fst_sSup [SupSet α] [SupSet β] (s : Set (α × β)) : (sSup s).fst = sSup (Prod.fst '' s) :=
  rfl


theorem snd_sSup [SupSet α] [SupSet β] (s : Set (α × β)) : (sSup s).snd = sSup (Prod.snd '' s) :=
  rfl


theorem swap_sSup [SupSet α] [SupSet β] (s : Set (α × β)) : (sSup s).swap = sSup (Prod.swap '' s) :=
  Prod.ext (congr_arg sSup <| image_comp Prod.fst swap s)
    (congr_arg sSup <| image_comp Prod.snd swap s)


theorem fst_iInf [InfSet α] [InfSet β] (f : ι → α × β) : (iInf f).fst = ⨅ i, (f i).fst :=
  congr_arg sInf (range_comp _ _).symm


theorem snd_iInf [InfSet α] [InfSet β] (f : ι → α × β) : (iInf f).snd = ⨅ i, (f i).snd :=
  congr_arg sInf (range_comp _ _).symm


theorem swap_iInf [InfSet α] [InfSet β] (f : ι → α × β) : (iInf f).swap = ⨅ i, (f i).swap := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝¹ : InfSet α
    inst✝ : InfSet β
    f : ι → Prod α β
    ⊢ Eq (iInf f).swap (iInf fun i => (f i).swap)
  -/
  simp_rw [iInf, swap_sInf, ← range_comp, comp_def]  -- Porting note: need to unfold `∘`
  /-
    🎉 no goals
  -/


theorem iInf_mk [InfSet α] [InfSet β] (f : ι → α) (g : ι → β) :
    ⨅ i, (f i, g i) = (⨅ i, f i, ⨅ i, g i) :=
  congr_arg₂ Prod.mk (fst_iInf _) (snd_iInf _)


theorem fst_iSup [SupSet α] [SupSet β] (f : ι → α × β) : (iSup f).fst = ⨆ i, (f i).fst :=
  congr_arg sSup (range_comp _ _).symm


theorem snd_iSup [SupSet α] [SupSet β] (f : ι → α × β) : (iSup f).snd = ⨆ i, (f i).snd :=
  congr_arg sSup (range_comp _ _).symm


theorem swap_iSup [SupSet α] [SupSet β] (f : ι → α × β) : (iSup f).swap = ⨆ i, (f i).swap := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_4
    inst✝¹ : SupSet α
    inst✝ : SupSet β
    f : ι → Prod α β
    ⊢ Eq (iSup f).swap (iSup fun i => (f i).swap)
  -/
  simp_rw [iSup, swap_sSup, ← range_comp, comp_def]  -- Porting note: need to unfold `∘`
  /-
    🎉 no goals
  -/


theorem iSup_mk [SupSet α] [SupSet β] (f : ι → α) (g : ι → β) :
    ⨆ i, (f i, g i) = (⨆ i, f i, ⨆ i, g i) :=
  congr_arg₂ Prod.mk (fst_iSup _) (snd_iSup _)


instance instCompleteLattice [CompleteLattice α] [CompleteLattice β] : CompleteLattice (α × β) where
  __ := instBoundedOrder α β
  le_sSup _ _ hab := ⟨le_sSup <| mem_image_of_mem _ hab, le_sSup <| mem_image_of_mem _ hab⟩
  sSup_le _ _ h :=
    ⟨sSup_le <| forall_mem_image.2 fun p hp => (h p hp).1,
      sSup_le <| forall_mem_image.2 fun p hp => (h p hp).2⟩
  sInf_le _ _ hab := ⟨sInf_le <| mem_image_of_mem _ hab, sInf_le <| mem_image_of_mem _ hab⟩
  le_sInf _ _ h :=
    ⟨le_sInf <| forall_mem_image.2 fun p hp => (h p hp).1,
      le_sInf <| forall_mem_image.2 fun p hp => (h p hp).2⟩


lemma sInf_prod [InfSet α] [InfSet β] {s : Set α} {t : Set β} (hs : s.Nonempty) (ht : t.Nonempty) :
    sInf (s ×ˢ t) = (sInf s, sInf t) :=
congr_arg₂ Prod.mk (congr_arg sInf <| fst_image_prod _ ht) (congr_arg sInf <| snd_image_prod hs _)


lemma sSup_prod [SupSet α] [SupSet β] {s : Set α} {t : Set β} (hs : s.Nonempty) (ht : t.Nonempty) :
    sSup (s ×ˢ t) = (sSup s, sSup t) :=
congr_arg₂ Prod.mk (congr_arg sSup <| fst_image_prod _ ht) (congr_arg sSup <| snd_image_prod hs _)


/-- This is a weaker version of `sup_sInf_eq` -/
theorem sup_sInf_le_iInf_sup : a ⊔ sInf s ≤ ⨅ b ∈ s, a ⊔ b :=
  le_iInf₂ fun _ h => sup_le_sup_left (sInf_le h) _


/-- This is a weaker version of `inf_sSup_eq` -/
theorem iSup_inf_le_inf_sSup : ⨆ b ∈ s, a ⊓ b ≤ a ⊓ sSup s :=
  @sup_sInf_le_iInf_sup αᵒᵈ _ _ _


/-- This is a weaker version of `sInf_sup_eq` -/
theorem sInf_sup_le_iInf_sup : sInf s ⊔ a ≤ ⨅ b ∈ s, b ⊔ a :=
  le_iInf₂ fun _ h => sup_le_sup_right (sInf_le h) _


/-- This is a weaker version of `sSup_inf_eq` -/
theorem iSup_inf_le_sSup_inf : ⨆ b ∈ s, b ⊓ a ≤ sSup s ⊓ a :=
  @sInf_sup_le_iInf_sup αᵒᵈ _ _ _


theorem le_iSup_inf_iSup (f g : ι → α) : ⨆ i, f i ⊓ g i ≤ (⨆ i, f i) ⊓ ⨆ i, g i :=
  le_inf (iSup_mono fun _ => inf_le_left) (iSup_mono fun _ => inf_le_right)


theorem iInf_sup_iInf_le (f g : ι → α) : (⨅ i, f i) ⊔ ⨅ i, g i ≤ ⨅ i, f i ⊔ g i :=
  @le_iSup_inf_iSup αᵒᵈ ι _ f g


theorem disjoint_sSup_left {a : Set α} {b : α} (d : Disjoint (sSup a) b) {i} (hi : i ∈ a) :
    Disjoint i b :=
  disjoint_iff_inf_le.mpr (iSup₂_le_iff.1 (iSup_inf_le_sSup_inf.trans d.le_bot) i hi : _)


theorem disjoint_sSup_right {a : Set α} {b : α} (d : Disjoint b (sSup a)) {i} (hi : i ∈ a) :
    Disjoint b i :=
  disjoint_iff_inf_le.mpr (iSup₂_le_iff.mp (iSup_inf_le_inf_sSup.trans d.le_bot) i hi : _)


/-- Pullback a `CompleteLattice` along an injection. -/
protected abbrev Function.Injective.completeLattice [Max α] [Min α] [SupSet α] [InfSet α] [Top α]
    [Bot α] [CompleteLattice β] (f : α → β) (hf : Function.Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) : CompleteLattice α where
  -- we cannot use BoundedOrder.lift here as the `LE` instance doesn't exist yet
  __ := hf.lattice f map_sup map_inf
  le_sSup _ a h := (le_iSup₂ a h).trans (map_sSup _).ge
  sSup_le _ _ h := (map_sSup _).trans_le <| iSup₂_le h
  sInf_le _ a h := (map_sInf _).trans_le <| iInf₂_le a h
  le_sInf _ _ h := (le_iInf₂ h).trans (map_sInf _).ge
  top := ⊤
  le_top _ := (@le_top β _ _ _).trans map_top.ge
  bot := ⊥
  bot_le _ := map_bot.le.trans bot_le


instance supSet [SupSet α] : SupSet (ULift.{v} α) where sSup s := ULift.up (sSup <| ULift.up ⁻¹' s)


theorem down_sSup [SupSet α] (s : Set (ULift.{v} α)) : (sSup s).down = sSup (ULift.up ⁻¹' s) := rfl

theorem up_sSup [SupSet α] (s : Set α) : up (sSup s) = sSup (ULift.down ⁻¹' s) := rfl


instance infSet [InfSet α] : InfSet (ULift.{v} α) where sInf s := ULift.up (sInf <| ULift.up ⁻¹' s)


theorem down_sInf [InfSet α] (s : Set (ULift.{v} α)) : (sInf s).down = sInf (ULift.up ⁻¹' s) := rfl

theorem up_sInf [InfSet α] (s : Set α) : up (sInf s) = sInf (ULift.down ⁻¹' s) := rfl


theorem down_iSup [SupSet α] (f : ι → ULift.{v} α) : (⨆ i, f i).down = ⨆ i, (f i).down :=
  congr_arg sSup <| (preimage_eq_iff_eq_image ULift.up_bijective).mpr <|
    Eq.symm (range_comp _ _).symm

theorem up_iSup [SupSet α] (f : ι → α) : up (⨆ i, f i) = ⨆ i, up (f i) :=
  congr_arg ULift.up <| (down_iSup _).symm


theorem down_iInf [InfSet α] (f : ι → ULift.{v} α) : (⨅ i, f i).down = ⨅ i, (f i).down :=
  congr_arg sInf <| (preimage_eq_iff_eq_image ULift.up_bijective).mpr <|
    Eq.symm (range_comp _ _).symm

theorem up_iInf [InfSet α] (f : ι → α) : up (⨅ i, f i) = ⨅ i, up (f i) :=
  congr_arg ULift.up <| (down_iInf _).symm


instance instCompleteLattice [CompleteLattice α] : CompleteLattice (ULift.{v} α) :=
  ULift.down_injective.completeLattice _ down_sup down_inf
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   ι : Sort u_4
                   ι' : Sort u_5
                   κ : ι → Sort u_6
                   κ' : ι' → Sort u_7
                   inst✝ : CompleteLattice α
                   s : Set (ULift.{v, u_1} α)
                   ⊢ Eq (SupSet.sSup s).down (iSup fun a => iSup fun h => a.down)
                 -/
    (fun s => by rw [sSup_eq_iSup', down_iSup, iSup_subtype''])
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   ι : Sort u_4
                   ι' : Sort u_5
                   κ : ι → Sort u_6
                   κ' : ι' → Sort u_7
                   inst✝ : CompleteLattice α
                   s : Set (ULift.{v, u_1} α)
                   ⊢ Eq (InfSet.sInf s).down (iInf fun a => iInf fun h => a.down)
                 -/
    (fun s => by rw [sInf_eq_iInf', down_iInf, iInf_subtype'']) down_top down_bot
                 /-
                   🎉 no goals
                 -/


instance instCompleteLinearOrder : CompleteLinearOrder PUnit where
  __ := instBooleanAlgebra
  __ := instLinearOrder
  sSup := fun _ => unit
  sInf := fun _ => unit
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  ι : Sort u_4
                  ι' : Sort u_5
                  κ : ι → Sort u_6
                  κ' : ι' → Sort u_7
                  ⊢ ∀ (s : Set PUnit.{?u.204283 + 1}) (a : PUnit.{?u.204283 + 1}), Membership.me …
                -/
  le_sSup := by intros; trivial
                        /-
                          🎉 no goals
                        -/
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  ι : Sort u_4
                  ι' : Sort u_5
                  κ : ι → Sort u_6
                  κ' : ι' → Sort u_7
                  ⊢ ∀ (s : Set PUnit.{?u.204283 + 1}) (a : PUnit.{?u.204283 + 1}), (∀ (b : PUnit …
                -/
  sSup_le := by intros; trivial
                        /-
                          🎉 no goals
                        -/
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  ι : Sort u_4
                  ι' : Sort u_5
                  κ : ι → Sort u_6
                  κ' : ι' → Sort u_7
                  ⊢ ∀ (s : Set PUnit.{?u.204283 + 1}) (a : PUnit.{?u.204283 + 1}), Membership.me …
                -/
  sInf_le := by intros; trivial
                        /-
                          🎉 no goals
                        -/
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  ι : Sort u_4
                  ι' : Sort u_5
                  κ : ι → Sort u_6
                  κ' : ι' → Sort u_7
                  ⊢ ∀ (s : Set PUnit.{?u.204283 + 1}) (a : PUnit.{?u.204283 + 1}), (∀ (b : PUnit …
                -/
  le_sInf := by intros; trivial
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      ι : Sort u_4
                      ι' : Sort u_5
                      κ : ι → Sort u_6
                      κ' : ι' → Sort u_7
                      ⊢ ∀ (a b c : PUnit.{?u.204283 + 1}), Iff (LE.le a (HImp.himp b c)) (LE.le (Min …
                    -/
  le_himp_iff := by intros; trivial
                            /-
                              🎉 no goals
                            -/
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   ι : Sort u_4
                   ι' : Sort u_5
                   κ : ι → Sort u_6
                   κ' : ι' → Sort u_7
                   ⊢ ∀ (a : PUnit.{?u.204283 + 1}), Eq (HImp.himp a Bot.bot) (HasCompl.compl a)
                 -/
  himp_bot := by intros; trivial
                         /-
                           🎉 no goals
                         -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       ι : Sort u_4
                       ι' : Sort u_5
                       κ : ι → Sort u_6
                       κ' : ι' → Sort u_7
                       ⊢ ∀ (a b c : PUnit.{?u.204283 + 1}), Iff (LE.le (SDiff.sdiff a b) c) (LE.le a  …
                     -/
  sdiff_le_iff := by intros; trivial
                             /-
                               🎉 no goals
                             -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    ι : Sort u_4
                    ι' : Sort u_5
                    κ : ι → Sort u_6
                    κ' : ι' → Sort u_7
                    ⊢ ∀ (a : PUnit.{?u.204283 + 1}), Eq (SDiff.sdiff Top.top a) (HNot.hnot a)
                  -/
  top_sdiff := by intros; trivial
                          /-
                            🎉 no goals
                          -/


