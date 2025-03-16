/-- Structure containing the minimal axioms required to check that an order is a frame. Do NOT use,
except for implementing `Order.Frame` via `Order.Frame.ofMinimalAxioms`.

This structure omits the `himp`, `compl` fields, which can be recovered using
`Order.Frame.ofMinimalAxioms`. -/
class Order.Frame.MinimalAxioms (α : Type u) extends CompleteLattice α where
  inf_sSup_le_iSup_inf (a : α) (s : Set α) : a ⊓ sSup s ≤ ⨆ b ∈ s, a ⊓ b


/-- Structure containing the minimal axioms required to check that an order is a coframe. Do NOT
use, except for implementing `Order.Coframe` via `Order.Coframe.ofMinimalAxioms`.

This structure omits the `sdiff`, `hnot` fields, which can be recovered using
`Order.Coframe.ofMinimalAxioms`. -/
class Order.Coframe.MinimalAxioms (α : Type u) extends CompleteLattice α where
  iInf_sup_le_sup_sInf (a : α) (s : Set α) : ⨅ b ∈ s, a ⊔ b ≤ a ⊔ sInf s


/-- A frame, aka complete Heyting algebra, is a complete lattice whose `⊓` distributes over `⨆`. -/
class Order.Frame (α : Type*) extends CompleteLattice α, HeytingAlgebra α where
  /-- `⊓` distributes over `⨆`. -/
  inf_sSup_le_iSup_inf (a : α) (s : Set α) : a ⊓ sSup s ≤ ⨆ b ∈ s, a ⊓ b


/-- A coframe, aka complete Brouwer algebra or complete co-Heyting algebra, is a complete lattice
whose `⊔` distributes over `⨅`. -/
class Order.Coframe (α : Type*) extends CompleteLattice α, CoheytingAlgebra α where
  /-- `⊔` distributes over `⨅`. -/
  iInf_sup_le_sup_sInf (a : α) (s : Set α) : ⨅ b ∈ s, a ⊔ b ≤ a ⊔ sInf s


/-- Structure containing the minimal axioms required to check that an order is a complete
distributive lattice. Do NOT use, except for implementing `CompleteDistribLattice` via
`CompleteDistribLattice.ofMinimalAxioms`.

This structure omits the `himp`, `compl`, `sdiff`, `hnot` fields, which can be recovered using
`CompleteDistribLattice.ofMinimalAxioms`. -/
structure CompleteDistribLattice.MinimalAxioms (α : Type u)
    extends CompleteLattice α, Frame.MinimalAxioms α, Coframe.MinimalAxioms α where

-- We give those projections better name further down

/-- A complete distributive lattice is a complete lattice whose `⊔` and `⊓` respectively
distribute over `⨅` and `⨆`. -/
class CompleteDistribLattice (α : Type*) extends Frame α, Coframe α, BiheytingAlgebra α


/-- Structure containing the minimal axioms required to check that an order is a completely
distributive. Do NOT use, except for implementing `CompletelyDistribLattice` via
`CompletelyDistribLattice.ofMinimalAxioms`.

This structure omits the `himp`, `compl`, `sdiff`, `hnot` fields, which can be recovered using
`CompletelyDistribLattice.ofMinimalAxioms`. -/
structure CompletelyDistribLattice.MinimalAxioms (α : Type u) extends CompleteLattice α where
  protected iInf_iSup_eq {ι : Type u} {κ : ι → Type u} (f : ∀ a, κ a → α) :
    (⨅ a, ⨆ b, f a b) = ⨆ g : ∀ a, κ a, ⨅ a, f a (g a)


/-- A completely distributive lattice is a complete lattice whose `⨅` and `⨆`
distribute over each other. -/
class CompletelyDistribLattice (α : Type u) extends CompleteLattice α, BiheytingAlgebra α where
  protected iInf_iSup_eq {ι : Type u} {κ : ι → Type u} (f : ∀ a, κ a → α) :
    (⨅ a, ⨆ b, f a b) = ⨆ g : ∀ a, κ a, ⨅ a, f a (g a)


theorem le_iInf_iSup [CompleteLattice α] {f : ∀ a, κ a → α} :
    (⨆ g : ∀ a, κ a, ⨅ a, f a (g a)) ≤ ⨅ a, ⨆ b, f a b :=
  iSup_le fun _ => le_iInf fun a => le_trans (iInf_le _ a) (le_iSup _ _)


lemma iSup_iInf_le [CompleteLattice α] {f : ∀ a, κ a → α} :
    ⨆ a, ⨅ b, f a b ≤ ⨅ g : ∀ a, κ a, ⨆ a, f a (g a) :=
  le_iInf_iSup (α := αᵒᵈ)


lemma inf_sSup_eq : a ⊓ sSup s = ⨆ b ∈ s, a ⊓ b :=
  (minAx.inf_sSup_le_iSup_inf _ _).antisymm iSup_inf_le_inf_sSup


lemma sSup_inf_eq : sSup s ⊓ b = ⨆ a ∈ s, a ⊓ b := by
  /-
    α : Type u
    minAx : Order.Frame.MinimalAxioms α
    s : Set α
    b : α
    ⊢ Eq (Min.min (SupSet.sSup s) b) (iSup fun a => iSup fun h => Min.min a b)
  -/
  simpa only [inf_comm] using @inf_sSup_eq α _ s b
  /-
    🎉 no goals
  -/


lemma iSup_inf_eq (f : ι → α) (a : α) : (⨆ i, f i) ⊓ a = ⨆ i, f i ⊓ a := by
  /-
    α : Type u
    ι : Sort w
    minAx : Order.Frame.MinimalAxioms α
    f : ι → α
    a : α
    ⊢ Eq (Min.min (iSup fun i => f i) a) (iSup fun i => Min.min (f i) a)
  -/
  rw [iSup, sSup_inf_eq, iSup_range]
  /-
    🎉 no goals
  -/


lemma inf_iSup_eq (a : α) (f : ι → α) : (a ⊓ ⨆ i, f i) = ⨆ i, a ⊓ f i := by
  /-
    α : Type u
    ι : Sort w
    minAx : Order.Frame.MinimalAxioms α
    a : α
    f : ι → α
    ⊢ Eq (Min.min a (iSup fun i => f i)) (iSup fun i => Min.min a (f i))
  -/
  simpa only [inf_comm] using minAx.iSup_inf_eq f a
  /-
    🎉 no goals
  -/


lemma inf_iSup₂_eq {f : ∀ i, κ i → α} (a : α) : (a ⊓ ⨆ i, ⨆ j, f i j) = ⨆ i, ⨆ j, a ⊓ f i j := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : Order.Frame.MinimalAxioms α
    f : (i : ι) → κ i → α
    a : α
    ⊢ Eq (Min.min a (iSup fun i => iSup fun j => f i j)) (iSup fun i => iSup fun j …
  -/
  simp only [inf_iSup_eq]
  /-
    🎉 no goals
  -/


/-- The `Order.Frame.MinimalAxioms` element corresponding to a frame. -/
def of [Frame α] : MinimalAxioms α := { ‹Frame α› with }


/-- Construct a frame instance using the minimal amount of work needed.

This sets `a ⇨ b := sSup {c | c ⊓ a ≤ b}` and `aᶜ := a ⇨ ⊥`. -/
-- See note [reducible non instances]
abbrev ofMinimalAxioms (minAx : MinimalAxioms α) : Frame α where
  __ := minAx
  himp a b := sSup {c | c ⊓ a ≤ b}
  le_himp_iff _ b c :=
                                              /-
                                                α : Type u
                                                β : Type v
                                                ι : Sort w
                                                κ : ι → Sort w'
                                                minAx : Order.Frame.MinimalAxioms α
                                                x✝ b c : α
                                                h : LE.le x✝ (HImp.himp b c)
                                                ⊢ LE.le (Min.min (HImp.himp b c) b) c
                                              -/
    ⟨fun h ↦ (inf_le_inf_right _ h).trans (by simp [minAx.sSup_inf_eq]), fun h ↦ le_sSup h⟩
                                              /-
                                                🎉 no goals
                                              -/
  himp_bot _ := rfl


lemma sup_sInf_eq : a ⊔ sInf s = ⨅ b ∈ s, a ⊔ b :=
  sup_sInf_le_iInf_sup.antisymm (minAx.iInf_sup_le_sup_sInf _ _)


lemma sInf_sup_eq : sInf s ⊔ b = ⨅ a ∈ s, a ⊔ b := by
  /-
    α : Type u
    minAx : Order.Coframe.MinimalAxioms α
    s : Set α
    b : α
    ⊢ Eq (Max.max (InfSet.sInf s) b) (iInf fun a => iInf fun h => Max.max a b)
  -/
  simpa only [sup_comm] using @sup_sInf_eq α _ s b
  /-
    🎉 no goals
  -/


lemma iInf_sup_eq (f : ι → α) (a : α) : (⨅ i, f i) ⊔ a = ⨅ i, f i ⊔ a := by
  /-
    α : Type u
    ι : Sort w
    minAx : Order.Coframe.MinimalAxioms α
    f : ι → α
    a : α
    ⊢ Eq (Max.max (iInf fun i => f i) a) (iInf fun i => Max.max (f i) a)
  -/
  rw [iInf, sInf_sup_eq, iInf_range]
  /-
    🎉 no goals
  -/


lemma sup_iInf_eq (a : α) (f : ι → α) : (a ⊔ ⨅ i, f i) = ⨅ i, a ⊔ f i := by
  /-
    α : Type u
    ι : Sort w
    minAx : Order.Coframe.MinimalAxioms α
    a : α
    f : ι → α
    ⊢ Eq (Max.max a (iInf fun i => f i)) (iInf fun i => Max.max a (f i))
  -/
  simpa only [sup_comm] using minAx.iInf_sup_eq f a
  /-
    🎉 no goals
  -/


lemma sup_iInf₂_eq {f : ∀ i, κ i → α} (a : α) : (a ⊔ ⨅ i, ⨅ j, f i j) = ⨅ i, ⨅ j, a ⊔ f i j := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : Order.Coframe.MinimalAxioms α
    f : (i : ι) → κ i → α
    a : α
    ⊢ Eq (Max.max a (iInf fun i => iInf fun j => f i j)) (iInf fun i => iInf fun j …
  -/
  simp only [sup_iInf_eq]
  /-
    🎉 no goals
  -/


/-- The `Order.Coframe.MinimalAxioms` element corresponding to a frame. -/
def of [Coframe α] : MinimalAxioms α := { ‹Coframe α› with }


/-- Construct a coframe instance using the minimal amount of work needed.

This sets `a \ b := sInf {c | a ≤ b ⊔ c}` and `￢a := ⊤ \ a`. -/
-- See note [reducible non instances]
abbrev ofMinimalAxioms (minAx : MinimalAxioms α) : Coframe α where
  __ := minAx
  sdiff a b := sInf {c | a ≤ b ⊔ c}
  sdiff_le_iff a b _ :=
                                              /-
                                                α : Type u
                                                β : Type v
                                                ι : Sort w
                                                κ : ι → Sort w'
                                                minAx : Order.Coframe.MinimalAxioms α
                                                a b x✝ : α
                                                h : LE.le (SDiff.sdiff a b) x✝
                                                ⊢ LE.le a (Max.max b (SDiff.sdiff a b))
                                              -/
    ⟨fun h ↦ (sup_le_sup_left h _).trans' (by simp [minAx.sup_sInf_eq]), fun h ↦ sInf_le h⟩
                                              /-
                                                🎉 no goals
                                              -/
  top_sdiff _ := rfl


/-- The `CompleteDistribLattice.MinimalAxioms` element corresponding to a complete distrib lattice.
-/
def of [CompleteDistribLattice α] : MinimalAxioms α := { ‹CompleteDistribLattice α› with }


/-- Turn minimal axioms for `CompleteDistribLattice` into minimal axioms for `Order.Frame`. -/
abbrev toFrame : Frame.MinimalAxioms α := minAx.toMinimalAxioms


/-- Turn minimal axioms for `CompleteDistribLattice` into minimal axioms for `Order.Coframe`. -/
abbrev toCoframe : Coframe.MinimalAxioms α where __ := minAx


/-- Construct a complete distrib lattice instance using the minimal amount of work needed.

This sets `a ⇨ b := sSup {c | c ⊓ a ≤ b}`, `aᶜ := a ⇨ ⊥`, `a \ b := sInf {c | a ≤ b ⊔ c}` and
`￢a := ⊤ \ a`. -/
-- See note [reducible non instances]
abbrev ofMinimalAxioms (minAx : MinimalAxioms α) : CompleteDistribLattice α where
  __ := Frame.ofMinimalAxioms minAx.toFrame
  __ := Coframe.ofMinimalAxioms minAx.toCoframe


lemma iInf_iSup_eq' (f : ∀ a, κ a → α) :
    let _ := minAx.toCompleteLattice
    ⨅ i, ⨆ j, f i j = ⨆ g : ∀ i, κ i, ⨅ i, f i (g i) := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (a : ι) → κ a → α
    ⊢ let x := minAx.toCompleteLattice;
      Eq (iInf fun i => iSup fun j => f i j) (iSup fun g => iInf fun i => f i (g i))
  -/
  let _ := minAx.toCompleteLattice
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (a : ι) → κ a → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    ⊢ let x := minAx.toCompleteLattice;
      Eq (iInf fun i => iSup fun j => f i j) (iSup fun g => iInf fun i => f i (g i))
  -/
  refine le_antisymm ?_ le_iInf_iSup
  calc
    _ = ⨅ a : range (range <| f ·), ⨆ b : a.1, b.1 := by
      simp_rw [iInf_subtype, iInf_range, iSup_subtype, iSup_range]
    _ = _ := minAx.iInf_iSup_eq _
    _ ≤ _ := iSup_le fun g => by
      refine le_trans ?_ <| le_iSup _ fun a => Classical.choose (g ⟨_, a, rfl⟩).2
      refine le_iInf fun a => le_trans (iInf_le _ ⟨range (f a), a, rfl⟩) ?_
      rw [← Classical.choose_spec (g ⟨_, a, rfl⟩).2]


lemma iSup_iInf_eq (f : ∀ i, κ i → α) :
    let _ := minAx.toCompleteLattice
    ⨆ i, ⨅ j, f i j = ⨅ g : ∀ i, κ i, ⨆ i, f i (g i) := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    ⊢ let x := minAx.toCompleteLattice;
      Eq (iSup fun i => iInf fun j => f i j) (iInf fun g => iSup fun i => f i (g i))
  -/
  let _ := minAx.toCompleteLattice
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    ⊢ let x := minAx.toCompleteLattice;
      Eq (iSup fun i => iInf fun j => f i j) (iInf fun g => iSup fun i => f i (g i))
  -/
  refine le_antisymm iSup_iInf_le ?_
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    ⊢ LE.le (iInf fun g => iSup fun i => f i (g i)) (iSup fun i => iInf fun j => f …
  -/
  rw [minAx.iInf_iSup_eq']
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    ⊢ LE.le (iSup fun g => iInf fun i => f (g i) (i (g i))) (iSup fun i => iInf fu …
  -/
  refine iSup_le fun g => ?_
  have ⟨a, ha⟩ : ∃ a, ∀ b, ∃ f, ∃ h : a = g f, h ▸ b = f (g f) := of_not_not fun h => by
    push_neg at h
    choose h hh using h
    have := hh _ h rfl
    contradiction
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    g : ((i : ι) → κ i) → ι
    a : ι
    ha : ∀ (b : κ a), Exists fun f => Exists fun h => Eq (Eq.rec b h) (f (g f))
    ⊢ LE.le (iInf fun i => f (g i) (i (g i))) (iSup fun i => iInf fun j => f i j)
  -/
  refine le_trans ?_ (le_iSup _ a)
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    g : ((i : ι) → κ i) → ι
    a : ι
    ha : ∀ (b : κ a), Exists fun f => Exists fun h => Eq (Eq.rec b h) (f (g f))
    ⊢ LE.le (iInf fun i => f (g i) (i (g i))) (iInf fun j => f a j)
  -/
  refine le_iInf fun b => ?_
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    g : ((i : ι) → κ i) → ι
    a : ι
    ha : ∀ (b : κ a), Exists fun f => Exists fun h => Eq (Eq.rec b h) (f (g f))
    b : κ a
    ⊢ LE.le (iInf fun i => f (g i) (i (g i))) (f a b)
  -/
  obtain ⟨h, rfl, rfl⟩ := ha b
  /-
    case intro.intro
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    minAx : CompletelyDistribLattice.MinimalAxioms α
    f : (i : ι) → κ i → α
    x✝ : CompleteLattice α := minAx.toCompleteLattice
    g : ((i : ι) → κ i) → ι
    h : (i : ι) → κ i
    ha : ∀ (b : κ (g h)), Exists fun f => Exists fun h_1 => Eq (Eq.rec b h_1) (f ( …
    ⊢ LE.le (iInf fun i => f (g i) (i (g i))) (f (g h) (h (g h)))
  -/
  exact iInf_le _ _
  /-
    🎉 no goals
  -/


/-- Turn minimal axioms for `CompletelyDistribLattice` into minimal axioms for
`CompleteDistribLattice`. -/
abbrev toCompleteDistribLattice : CompleteDistribLattice.MinimalAxioms α where
  __ := minAx
  inf_sSup_le_iSup_inf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      minAx : CompletelyDistribLattice.MinimalAxioms α
      a : α
      s : Set α
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    let _ := minAx.toCompleteLattice
    calc
      _ = ⨅ i : ULift.{u} Bool, ⨆ j : match i with | .up true => PUnit.{u + 1} | .up false => s,
          match i with
          | .up true => a
          | .up false => j := by simp [sSup_eq_iSup', iSup_unique, iInf_bool_eq]
      _ ≤ _ := by
        simp only [minAx.iInf_iSup_eq, iInf_ulift, iInf_bool_eq, iSup_le_iff]
        exact fun x ↦ le_biSup _ (x (.up false)).2
  iInf_sup_le_sup_sInf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      minAx : CompletelyDistribLattice.MinimalAxioms α
      a : α
      s : Set α
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    let _ := minAx.toCompleteLattice
    calc
      _ ≤ ⨆ i : ULift.{u} Bool, ⨅ j : match i with | .up true => PUnit.{u + 1} | .up false => s,
          match i with
          | .up true => a
          | .up false => j := by
        simp only [minAx.iSup_iInf_eq, iSup_ulift, iSup_bool_eq, le_iInf_iff]
        exact fun x ↦ biInf_le _ (x (.up false)).2
      _ = _ := by simp [sInf_eq_iInf', iInf_unique, iSup_bool_eq]


/-- The `CompletelyDistribLattice.MinimalAxioms` element corresponding to a frame. -/
def of [CompletelyDistribLattice α] : MinimalAxioms α := { ‹CompletelyDistribLattice α› with }


/-- Construct a completely distributive lattice instance using the minimal amount of work needed.

This sets `a ⇨ b := sSup {c | c ⊓ a ≤ b}`, `aᶜ := a ⇨ ⊥`, `a \ b := sInf {c | a ≤ b ⊔ c}` and
`￢a := ⊤ \ a`. -/
-- See note [reducible non instances]
abbrev ofMinimalAxioms (minAx : MinimalAxioms α) : CompletelyDistribLattice α where
  __ := minAx
  __ := CompleteDistribLattice.ofMinimalAxioms minAx.toCompleteDistribLattice


theorem iInf_iSup_eq [CompletelyDistribLattice α] {f : ∀ a, κ a → α} :
    (⨅ a, ⨆ b, f a b) = ⨆ g : ∀ a, κ a, ⨅ a, f a (g a) :=
  CompletelyDistribLattice.MinimalAxioms.of.iInf_iSup_eq' _


theorem iSup_iInf_eq [CompletelyDistribLattice α] {f : ∀ a, κ a → α} :
    (⨆ a, ⨅ b, f a b) = ⨅ g : ∀ a, κ a, ⨆ a, f a (g a) :=
  CompletelyDistribLattice.MinimalAxioms.of.iSup_iInf_eq _


instance (priority := 100) CompletelyDistribLattice.toCompleteDistribLattice
    [CompletelyDistribLattice α] : CompleteDistribLattice α where
  __ := ‹CompletelyDistribLattice α›
  __ := CompleteDistribLattice.ofMinimalAxioms MinimalAxioms.of.toCompleteDistribLattice

-- See note [lower instance priority]

instance (priority := 100) CompleteLinearOrder.toCompletelyDistribLattice [CompleteLinearOrder α] :
    CompletelyDistribLattice α where
  __ := ‹CompleteLinearOrder α›
  iInf_iSup_eq {α β} g := by
    /-
      α✝ : Type u
      β✝ : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝ : CompleteLinearOrder α✝
      α : Type u
      β : α → Type u
      g : (a : α) → β a → α✝
      ⊢ Eq (iInf fun a => iSup fun b => g a b) (iSup fun g_1 => iInf fun a => g a (g …
    -/
    let lhs := ⨅ a, ⨆ b, g a b
    /-
      α✝ : Type u
      β✝ : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝ : CompleteLinearOrder α✝
      α : Type u
      β : α → Type u
      g : (a : α) → β a → α✝
      lhs : α✝ := iInf fun a => iSup fun b => g a b
      ⊢ Eq (iInf fun a => iSup fun b => g a b) (iSup fun g_1 => iInf fun a => g a (g …
    -/
    let rhs := ⨆ h : ∀ a, β a, ⨅ a, g a (h a)
    /-
      α✝ : Type u
      β✝ : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝ : CompleteLinearOrder α✝
      α : Type u
      β : α → Type u
      g : (a : α) → β a → α✝
      lhs : α✝ := iInf fun a => iSup fun b => g a b
      rhs : α✝ := iSup fun h => iInf fun a => g a (h a)
      ⊢ Eq (iInf fun a => iSup fun b => g a b) (iSup fun g_1 => iInf fun a => g a (g …
    -/
    suffices lhs ≤ rhs from le_antisymm this le_iInf_iSup
    if h : ∃ x, rhs < x ∧ x < lhs then
      rcases h with ⟨x, hr, hl⟩
      suffices rhs ≥ x from nomatch not_lt.2 this hr
      have : ∀ a, ∃ b, x < g a b := fun a =>
        lt_iSup_iff.1 <| lt_of_not_le fun h =>
            lt_irrefl x (lt_of_lt_of_le hl (le_trans (iInf_le _ a) h))
      choose f hf using this
      refine le_trans ?_ (le_iSup _ f)
      exact le_iInf fun a => le_of_lt (hf a)
    else
      refine le_of_not_lt fun hrl : rhs < lhs => not_le_of_lt hrl ?_
      replace h : ∀ x, x ≤ rhs ∨ lhs ≤ x := by
        simpa only [not_exists, not_and_or, not_or, not_lt] using h
      have : ∀ a, ∃ b, rhs < g a b := fun a =>
        lt_iSup_iff.1 <| lt_of_lt_of_le hrl (iInf_le _ a)
      choose f hf using this
      have : ∀ a, lhs ≤ g a (f a) := fun a =>
        (h (g a (f a))).resolve_left (by simpa using hf a)
      refine le_trans ?_ (le_iSup _ f)
      exact le_iInf fun a => this _


instance OrderDual.instCoframe : Coframe αᵒᵈ where
  __ := instCompleteLattice
  __ := instCoheytingAlgebra
  iInf_sup_le_sup_sInf := @Frame.inf_sSup_le_iSup_inf α _


theorem inf_sSup_eq : a ⊓ sSup s = ⨆ b ∈ s, a ⊓ b :=
  (Frame.inf_sSup_le_iSup_inf _ _).antisymm iSup_inf_le_inf_sSup


theorem sSup_inf_eq : sSup s ⊓ b = ⨆ a ∈ s, a ⊓ b := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    s : Set α
    b : α
    ⊢ Eq (Min.min (SupSet.sSup s) b) (iSup fun a => iSup fun h => Min.min a b)
  -/
  simpa only [inf_comm] using @inf_sSup_eq α _ s b
  /-
    🎉 no goals
  -/


theorem iSup_inf_eq (f : ι → α) (a : α) : (⨆ i, f i) ⊓ a = ⨆ i, f i ⊓ a := by
  /-
    α : Type u
    ι : Sort w
    inst✝ : Order.Frame α
    f : ι → α
    a : α
    ⊢ Eq (Min.min (iSup fun i => f i) a) (iSup fun i => Min.min (f i) a)
  -/
  rw [iSup, sSup_inf_eq, iSup_range]
  /-
    🎉 no goals
  -/


theorem inf_iSup_eq (a : α) (f : ι → α) : (a ⊓ ⨆ i, f i) = ⨆ i, a ⊓ f i := by
  /-
    α : Type u
    ι : Sort w
    inst✝ : Order.Frame α
    a : α
    f : ι → α
    ⊢ Eq (Min.min a (iSup fun i => f i)) (iSup fun i => Min.min a (f i))
  -/
  simpa only [inf_comm] using iSup_inf_eq f a
  /-
    🎉 no goals
  -/


theorem iSup₂_inf_eq {f : ∀ i, κ i → α} (a : α) :
    (⨆ (i) (j), f i j) ⊓ a = ⨆ (i) (j), f i j ⊓ a := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    inst✝ : Order.Frame α
    f : (i : ι) → κ i → α
    a : α
    ⊢ Eq (Min.min (iSup fun i => iSup fun j => f i j) a) (iSup fun i => iSup fun j …
  -/
  simp only [iSup_inf_eq]
  /-
    🎉 no goals
  -/


theorem inf_iSup₂_eq {f : ∀ i, κ i → α} (a : α) :
    (a ⊓ ⨆ (i) (j), f i j) = ⨆ (i) (j), a ⊓ f i j := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    inst✝ : Order.Frame α
    f : (i : ι) → κ i → α
    a : α
    ⊢ Eq (Min.min a (iSup fun i => iSup fun j => f i j)) (iSup fun i => iSup fun j …
  -/
  simp only [inf_iSup_eq]
  /-
    🎉 no goals
  -/


theorem iSup_inf_iSup {ι ι' : Type*} {f : ι → α} {g : ι' → α} :
    ((⨆ i, f i) ⊓ ⨆ j, g j) = ⨆ i : ι × ι', f i.1 ⊓ g i.2 := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    ι : Type u_1
    ι' : Type u_2
    f : ι → α
    g : ι' → α
    ⊢ Eq (Min.min (iSup fun i => f i) (iSup fun j => g j)) (iSup fun i => Min.min  …
  -/
  simp_rw [iSup_inf_eq, inf_iSup_eq, iSup_prod]
  /-
    🎉 no goals
  -/


theorem biSup_inf_biSup {ι ι' : Type*} {f : ι → α} {g : ι' → α} {s : Set ι} {t : Set ι'} :
    ((⨆ i ∈ s, f i) ⊓ ⨆ j ∈ t, g j) = ⨆ p ∈ s ×ˢ t, f (p : ι × ι').1 ⊓ g p.2 := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    ι : Type u_1
    ι' : Type u_2
    f : ι → α
    g : ι' → α
    s : Set ι
    t : Set ι'
    ⊢ Eq (Min.min (iSup fun i => iSup fun h => f i) (iSup fun j => iSup fun h => g …
  -/
  simp only [iSup_subtype', iSup_inf_iSup]
  /-
    α : Type u
    inst✝ : Order.Frame α
    ι : Type u_1
    ι' : Type u_2
    f : ι → α
    g : ι' → α
    s : Set ι
    t : Set ι'
    ⊢ Eq (iSup fun i => Min.min (f ↑i.1) (g ↑i.2)) (iSup fun x => Min.min (f (↑x). …
  -/
  exact (Equiv.surjective _).iSup_congr (Equiv.Set.prod s t).symm fun x => rfl
  /-
    🎉 no goals
  -/


theorem sSup_inf_sSup : sSup s ⊓ sSup t = ⨆ p ∈ s ×ˢ t, (p : α × α).1 ⊓ p.2 := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    s t : Set α
    ⊢ Eq (Min.min (SupSet.sSup s) (SupSet.sSup t)) (iSup fun p => iSup fun h => Mi …
  -/
  simp only [sSup_eq_iSup, biSup_inf_biSup]
  /-
    🎉 no goals
  -/


theorem iSup_disjoint_iff {f : ι → α} : Disjoint (⨆ i, f i) a ↔ ∀ i, Disjoint (f i) a := by
  /-
    α : Type u
    ι : Sort w
    inst✝ : Order.Frame α
    a : α
    f : ι → α
    ⊢ Iff (Disjoint (iSup fun i => f i) a) (∀ (i : ι), Disjoint (f i) a)
  -/
  simp only [disjoint_iff, iSup_inf_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


theorem disjoint_iSup_iff {f : ι → α} : Disjoint a (⨆ i, f i) ↔ ∀ i, Disjoint a (f i) := by
  /-
    α : Type u
    ι : Sort w
    inst✝ : Order.Frame α
    a : α
    f : ι → α
    ⊢ Iff (Disjoint a (iSup fun i => f i)) (∀ (i : ι), Disjoint a (f i))
  -/
  simpa only [disjoint_comm] using @iSup_disjoint_iff
  /-
    🎉 no goals
  -/


theorem iSup₂_disjoint_iff {f : ∀ i, κ i → α} :
    Disjoint (⨆ (i) (j), f i j) a ↔ ∀ i j, Disjoint (f i j) a := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    inst✝ : Order.Frame α
    a : α
    f : (i : ι) → κ i → α
    ⊢ Iff (Disjoint (iSup fun i => iSup fun j => f i j) a) (∀ (i : ι) (j : κ i), D …
  -/
  simp_rw [iSup_disjoint_iff]
  /-
    🎉 no goals
  -/


theorem disjoint_iSup₂_iff {f : ∀ i, κ i → α} :
    Disjoint a (⨆ (i) (j), f i j) ↔ ∀ i j, Disjoint a (f i j) := by
  /-
    α : Type u
    ι : Sort w
    κ : ι → Sort w'
    inst✝ : Order.Frame α
    a : α
    f : (i : ι) → κ i → α
    ⊢ Iff (Disjoint a (iSup fun i => iSup fun j => f i j)) (∀ (i : ι) (j : κ i), D …
  -/
  simp_rw [disjoint_iSup_iff]
  /-
    🎉 no goals
  -/


theorem sSup_disjoint_iff {s : Set α} : Disjoint (sSup s) a ↔ ∀ b ∈ s, Disjoint b a := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    a : α
    s : Set α
    ⊢ Iff (Disjoint (SupSet.sSup s) a) (∀ (b : α), Membership.mem s b → Disjoint b …
  -/
  simp only [disjoint_iff, sSup_inf_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


theorem disjoint_sSup_iff {s : Set α} : Disjoint a (sSup s) ↔ ∀ b ∈ s, Disjoint a b := by
  /-
    α : Type u
    inst✝ : Order.Frame α
    a : α
    s : Set α
    ⊢ Iff (Disjoint a (SupSet.sSup s)) (∀ (b : α), Membership.mem s b → Disjoint a …
  -/
  simpa only [disjoint_comm] using @sSup_disjoint_iff
  /-
    🎉 no goals
  -/


theorem iSup_inf_of_monotone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {f g : ι → α}
    (hf : Monotone f) (hg : Monotone g) : ⨆ i, f i ⊓ g i = (⨆ i, f i) ⊓ ⨆ i, g i := by
  /-
    α : Type u
    inst✝² : Order.Frame α
    ι : Type u_1
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f g : ι → α
    hf : Monotone f
    hg : Monotone g
    ⊢ Eq (iSup fun i => Min.min (f i) (g i)) (Min.min (iSup fun i => f i) (iSup fu …
  -/
  refine (le_iSup_inf_iSup f g).antisymm ?_
  /-
    α : Type u
    inst✝² : Order.Frame α
    ι : Type u_1
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f g : ι → α
    hf : Monotone f
    hg : Monotone g
    ⊢ LE.le (Min.min (iSup fun i => f i) (iSup fun i => g i)) (iSup fun i => Min.m …
  -/
  rw [iSup_inf_iSup]
  /-
    α : Type u
    inst✝² : Order.Frame α
    ι : Type u_1
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f g : ι → α
    hf : Monotone f
    hg : Monotone g
    ⊢ LE.le (iSup fun i => Min.min (f i.1) (g i.2)) (iSup fun i => Min.min (f i) ( …
  -/
  refine iSup_mono' fun i => ?_
  /-
    α : Type u
    inst✝² : Order.Frame α
    ι : Type u_1
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f g : ι → α
    hf : Monotone f
    hg : Monotone g
    i : Prod ι ι
    ⊢ Exists fun i' => LE.le (Min.min (f i.1) (g i.2)) (Min.min (f i') (g i'))
  -/
  rcases directed_of (· ≤ ·) i.1 i.2 with ⟨j, h₁, h₂⟩
  /-
    case intro.intro
    α : Type u
    inst✝² : Order.Frame α
    ι : Type u_1
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f g : ι → α
    hf : Monotone f
    hg : Monotone g
    i : Prod ι ι
    j : ι
    h₁ : LE.le i.1 j
    h₂ : LE.le i.2 j
    ⊢ Exists fun i' => LE.le (Min.min (f i.1) (g i.2)) (Min.min (f i') (g i'))
  -/
  exact ⟨j, inf_le_inf (hf h₁) (hg h₂)⟩
  /-
    🎉 no goals
  -/


theorem iSup_inf_of_antitone {ι : Type*} [Preorder ι] [IsDirected ι (swap (· ≤ ·))] {f g : ι → α}
    (hf : Antitone f) (hg : Antitone g) : ⨆ i, f i ⊓ g i = (⨆ i, f i) ⊓ ⨆ i, g i :=
  @iSup_inf_of_monotone α _ ιᵒᵈ _ _ f g hf.dual_left hg.dual_left

-- see Note [lower instance priority]

instance (priority := 100) Frame.toDistribLattice : DistribLattice α :=
  DistribLattice.ofInfSupLe fun a b c => by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝ : Order.Frame α
      s t : Set α
      a✝ b✝ a b c : α
      ⊢ LE.le (Min.min a (Max.max b c)) (Max.max (Min.min a b) (Min.min a c))
    -/
    rw [← sSup_pair, ← sSup_pair, inf_sSup_eq, ← sSup_image, image_pair]
    /-
      🎉 no goals
    -/


instance Prod.instFrame [Frame α] [Frame β] : Frame (α × β) where
  __ := instCompleteLattice
  __ := instHeytingAlgebra
  inf_sSup_le_iSup_inf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝² : Order.Frame α
      s✝ t : Set α
      a✝ b : α
      inst✝¹ : Order.Frame α
      inst✝ : Order.Frame β
      a : Prod α β
      s : Set (Prod α β)
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    simp [Prod.le_def, sSup_eq_iSup, fst_iSup, snd_iSup, fst_iInf, snd_iInf, inf_iSup_eq]
    /-
      🎉 no goals
    -/


instance Pi.instFrame {ι : Type*} {π : ι → Type*} [∀ i, Frame (π i)] : Frame (∀ i, π i) where
  __ := instCompleteLattice
  __ := instHeytingAlgebra
  inf_sSup_le_iSup_inf a s i := by
    /-
      α : Type u
      β : Type v
      ι✝ : Sort w
      κ : ι✝ → Sort w'
      inst✝¹ : Order.Frame α
      s✝ t : Set α
      a✝ b : α
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → Order.Frame (π i)
      a : (i : ι) → π i
      s : Set ((i : ι) → π i)
      i : ι
      ⊢ LE.le (Min.min a (SupSet.sSup s) i) (iSup (fun b => iSup fun h => Min.min a  …
    -/
    simp only [sSup_apply, iSup_apply, inf_apply, inf_iSup_eq, ← iSup_subtype'']; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance OrderDual.instFrame : Frame αᵒᵈ where
  __ := instCompleteLattice
  __ := instHeytingAlgebra
  inf_sSup_le_iSup_inf := @Coframe.iInf_sup_le_sup_sInf α _


theorem sup_sInf_eq : a ⊔ sInf s = ⨅ b ∈ s, a ⊔ b :=
  @inf_sSup_eq αᵒᵈ _ _ _


theorem sInf_sup_eq : sInf s ⊔ b = ⨅ a ∈ s, a ⊔ b :=
  @sSup_inf_eq αᵒᵈ _ _ _


theorem iInf_sup_eq (f : ι → α) (a : α) : (⨅ i, f i) ⊔ a = ⨅ i, f i ⊔ a :=
  @iSup_inf_eq αᵒᵈ _ _ _ _


theorem sup_iInf_eq (a : α) (f : ι → α) : (a ⊔ ⨅ i, f i) = ⨅ i, a ⊔ f i :=
  @inf_iSup_eq αᵒᵈ _ _ _ _


theorem iInf₂_sup_eq {f : ∀ i, κ i → α} (a : α) : (⨅ (i) (j), f i j) ⊔ a = ⨅ (i) (j), f i j ⊔ a :=
  @iSup₂_inf_eq αᵒᵈ _ _ _ _ _


theorem sup_iInf₂_eq {f : ∀ i, κ i → α} (a : α) : (a ⊔ ⨅ (i) (j), f i j) = ⨅ (i) (j), a ⊔ f i j :=
  @inf_iSup₂_eq αᵒᵈ _ _ _ _ _


theorem iInf_sup_iInf {ι ι' : Type*} {f : ι → α} {g : ι' → α} :
    ((⨅ i, f i) ⊔ ⨅ i, g i) = ⨅ i : ι × ι', f i.1 ⊔ g i.2 :=
  @iSup_inf_iSup αᵒᵈ _ _ _ _ _


theorem biInf_sup_biInf {ι ι' : Type*} {f : ι → α} {g : ι' → α} {s : Set ι} {t : Set ι'} :
    ((⨅ i ∈ s, f i) ⊔ ⨅ j ∈ t, g j) = ⨅ p ∈ s ×ˢ t, f (p : ι × ι').1 ⊔ g p.2 :=
  @biSup_inf_biSup αᵒᵈ _ _ _ _ _ _ _


theorem sInf_sup_sInf : sInf s ⊔ sInf t = ⨅ p ∈ s ×ˢ t, (p : α × α).1 ⊔ p.2 :=
  @sSup_inf_sSup αᵒᵈ _ _ _


theorem iInf_sup_of_monotone {ι : Type*} [Preorder ι] [IsDirected ι (swap (· ≤ ·))] {f g : ι → α}
    (hf : Monotone f) (hg : Monotone g) : ⨅ i, f i ⊔ g i = (⨅ i, f i) ⊔ ⨅ i, g i :=
  @iSup_inf_of_antitone αᵒᵈ _ _ _ _ _ _ hf.dual_right hg.dual_right


theorem iInf_sup_of_antitone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {f g : ι → α}
    (hf : Antitone f) (hg : Antitone g) : ⨅ i, f i ⊔ g i = (⨅ i, f i) ⊔ ⨅ i, g i :=
  @iSup_inf_of_monotone αᵒᵈ _ _ _ _ _ _ hf.dual_right hg.dual_right

-- see Note [lower instance priority]

instance (priority := 100) Coframe.toDistribLattice : DistribLattice α where
  __ := ‹Coframe α›
  le_sup_inf a b c := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝ : Order.Coframe α
      s t : Set α
      a✝ b✝ a b c : α
      ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
    -/
    rw [← sInf_pair, ← sInf_pair, sup_sInf_eq, ← sInf_image, image_pair]
    /-
      🎉 no goals
    -/


instance Prod.instCoframe [Coframe β] : Coframe (α × β) where
  __ := instCompleteLattice
  __ := instCoheytingAlgebra
  iInf_sup_le_sup_sInf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝¹ : Order.Coframe α
      s✝ t : Set α
      a✝ b : α
      inst✝ : Order.Coframe β
      a : Prod α β
      s : Set (Prod α β)
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    simp [Prod.le_def, sInf_eq_iInf, fst_iSup, snd_iSup, fst_iInf, snd_iInf, sup_iInf_eq]
    /-
      🎉 no goals
    -/


instance Pi.instCoframe {ι : Type*} {π : ι → Type*} [∀ i, Coframe (π i)] : Coframe (∀ i, π i) where
  __ := instCompleteLattice
  __ := instCoheytingAlgebra
  iInf_sup_le_sup_sInf a s i := by
    /-
      α : Type u
      β : Type v
      ι✝ : Sort w
      κ : ι✝ → Sort w'
      inst✝¹ : Order.Coframe α
      s✝ t : Set α
      a✝ b : α
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → Order.Coframe (π i)
      a : (i : ι) → π i
      s : Set ((i : ι) → π i)
      i : ι
      ⊢ LE.le (iInf (fun b => iInf fun h => Max.max a b) i) (Max.max a (InfSet.sInf  …
    -/
    simp only [sInf_apply, iInf_apply, sup_apply, sup_iInf_eq, ← iInf_subtype'']; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance OrderDual.instCompleteDistribLattice [CompleteDistribLattice α] :
    CompleteDistribLattice αᵒᵈ where
  __ := instFrame
  __ := instCoframe


instance Prod.instCompleteDistribLattice [CompleteDistribLattice β] :
    CompleteDistribLattice (α × β) where
  __ := instFrame
  __ := instCoframe


instance Pi.instCompleteDistribLattice {ι : Type*} {π : ι → Type*}
    [∀ i, CompleteDistribLattice (π i)] : CompleteDistribLattice (∀ i, π i) where
  __ := instFrame
  __ := instCoframe


instance OrderDual.instCompletelyDistribLattice [CompletelyDistribLattice α] :
    CompletelyDistribLattice αᵒᵈ where
  __ := instFrame
  __ := instCoframe
  iInf_iSup_eq _ := iSup_iInf_eq (α := α)


instance Prod.instCompletelyDistribLattice [CompletelyDistribLattice α]
    [CompletelyDistribLattice β] : CompletelyDistribLattice (α × β) where
  __ := instFrame
  __ := instCoframe
                       /-
                         α : Type u
                         β : Type v
                         ι : Sort w
                         κ : ι → Sort w'
                         inst✝¹ : CompletelyDistribLattice α
                         inst✝ : CompletelyDistribLattice β
                         ι✝ : Type (max u v)
                         κ✝ : ι✝ → Type (max u v)
                         f : (a : ι✝) → κ✝ a → Prod α β
                         ⊢ Eq (iInf fun a => iSup fun b => f a b) (iSup fun g => iInf fun a => f a (g a))
                       -/
                               /-
                                 🎉 no goals
                               -/
  iInf_iSup_eq f := by ext <;> simp [fst_iSup, fst_iInf, snd_iSup, snd_iInf, iInf_iSup_eq]
                               /-
                                 🎉 no goals
                               -/


instance Pi.instCompletelyDistribLattice {ι : Type*} {π : ι → Type*}
    [∀ i, CompletelyDistribLattice (π i)] : CompletelyDistribLattice (∀ i, π i) where
  __ := instFrame
  __ := instCoframe
                       /-
                         α : Type u
                         β : Type v
                         ι✝¹ : Sort w
                         κ : ι✝¹ → Sort w'
                         ι : Type u_1
                         π : ι → Type u_2
                         inst✝ : (i : ι) → CompletelyDistribLattice (π i)
                         ι✝ : Type (max u_1 u_2)
                         κ✝ : ι✝ → Type (max u_1 u_2)
                         f : (a : ι✝) → κ✝ a → (i : ι) → π i
                         ⊢ Eq (iInf fun a => iSup fun b => f a b) (iSup fun g => iInf fun a => f a (g a))
                       -/
  iInf_iSup_eq f := by ext i; simp only [iInf_apply, iSup_apply, iInf_iSup_eq]
                              /-
                                🎉 no goals
                              -/


/--
A complete Boolean algebra is a Boolean algebra that is also a complete distributive lattice.

It is only completely distributive if it is also atomic.
-/
-- We do not directly extend `CompleteDistribLattice` to avoid having the `hnot` field
class CompleteBooleanAlgebra (α) extends CompleteLattice α, BooleanAlgebra α where
  /-- `⊓` distributes over `⨆`. -/
  inf_sSup_le_iSup_inf (a : α) (s : Set α) : a ⊓ sSup s ≤ ⨆ b ∈ s, a ⊓ b
  /-- `⊔` distributes over `⨅`. -/
  iInf_sup_le_sup_sInf (a : α) (s : Set α) : ⨅ b ∈ s, a ⊔ b ≤ a ⊔ sInf s

-- See note [lower instance priority]

instance (priority := 100) CompleteBooleanAlgebra.toCompleteDistribLattice
    [CompleteBooleanAlgebra α] : CompleteDistribLattice α where
  __ := ‹CompleteBooleanAlgebra α›
  __ := BooleanAlgebra.toBiheytingAlgebra


instance Prod.instCompleteBooleanAlgebra [CompleteBooleanAlgebra α] [CompleteBooleanAlgebra β] :
    CompleteBooleanAlgebra (α × β) where
  __ := instBooleanAlgebra
  __ := instCompleteDistribLattice


instance Pi.instCompleteBooleanAlgebra {ι : Type*} {π : ι → Type*}
    [∀ i, CompleteBooleanAlgebra (π i)] : CompleteBooleanAlgebra (∀ i, π i) where
  __ := instBooleanAlgebra
  __ := instCompleteDistribLattice


instance OrderDual.instCompleteBooleanAlgebra [CompleteBooleanAlgebra α] :
    CompleteBooleanAlgebra αᵒᵈ where
  __ := instBooleanAlgebra
  __ := instCompleteDistribLattice


theorem compl_iInf : (iInf f)ᶜ = ⨆ i, (f i)ᶜ :=
  le_antisymm
    (compl_le_of_compl_le <| le_iInf fun i => compl_le_of_compl_le <|
      le_iSup (HasCompl.compl ∘ f) i)
    (iSup_le fun _ => compl_le_compl <| iInf_le _ _)


theorem compl_iSup : (iSup f)ᶜ = ⨅ i, (f i)ᶜ :=
                      /-
                        α : Type u
                        ι : Sort w
                        inst✝ : CompleteBooleanAlgebra α
                        f : ι → α
                        ⊢ Eq (HasCompl.compl (HasCompl.compl (iSup f))) (HasCompl.compl (iInf fun i => …
                      -/
  compl_injective (by simp [compl_iInf])
                      /-
                        🎉 no goals
                      -/


                                                   /-
                                                     α : Type u
                                                     inst✝ : CompleteBooleanAlgebra α
                                                     s : Set α
                                                     ⊢ Eq (HasCompl.compl (InfSet.sInf s)) (iSup fun i => iSup fun h => HasCompl.co …
                                                   -/
theorem compl_sInf : (sInf s)ᶜ = ⨆ i ∈ s, iᶜ := by simp only [sInf_eq_iInf, compl_iInf]
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                   /-
                                                     α : Type u
                                                     inst✝ : CompleteBooleanAlgebra α
                                                     s : Set α
                                                     ⊢ Eq (HasCompl.compl (SupSet.sSup s)) (iInf fun i => iInf fun h => HasCompl.co …
                                                   -/
theorem compl_sSup : (sSup s)ᶜ = ⨅ i ∈ s, iᶜ := by simp only [sSup_eq_iSup, compl_iSup]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem compl_sInf' : (sInf s)ᶜ = sSup (HasCompl.compl '' s) :=
  compl_sInf.trans sSup_image.symm


theorem compl_sSup' : (sSup s)ᶜ = sInf (HasCompl.compl '' s) :=
  compl_sSup.trans sInf_image.symm


open scoped symmDiff in
/-- The symmetric difference of two `iSup`s is at most the `iSup` of the symmetric differences. -/
theorem iSup_symmDiff_iSup_le {g : ι → α} : (⨆ i, f i) ∆ (⨆ i, g i) ≤ ⨆ i, ((f i) ∆ (g i)) := by
  /-
    α : Type u
    ι : Sort w
    inst✝ : CompleteBooleanAlgebra α
    f g : ι → α
    ⊢ LE.le (symmDiff (iSup fun i => f i) (iSup fun i => g i)) (iSup fun i => symm …
  -/
  simp_rw [symmDiff_le_iff, ← iSup_sup_eq]
  exact ⟨iSup_mono fun i ↦ sup_comm (g i) _ ▸ le_symmDiff_sup_right ..,
    iSup_mono fun i ↦ sup_comm (f i) _ ▸ symmDiff_comm (f i) _ ▸ le_symmDiff_sup_right ..⟩


open scoped symmDiff in
/-- A `biSup` version of `iSup_symmDiff_iSup_le`. -/
theorem biSup_symmDiff_biSup_le {p : ι → Prop} {f g : (i : ι) → p i → α} :
    (⨆ i, ⨆ (h : p i), f i h) ∆ (⨆ i, ⨆ (h : p i), g i h) ≤
    ⨆ i, ⨆ (h : p i), ((f i h) ∆ (g i h)) :=
  le_trans iSup_symmDiff_iSup_le <|iSup_mono fun _ ↦ iSup_symmDiff_iSup_le


/--
A complete atomic Boolean algebra is a complete Boolean algebra
that is also completely distributive.

We take iSup_iInf_eq as the definition here,
and prove later on that this implies atomicity.
-/
-- We do not directly extend `CompletelyDistribLattice` to avoid having the `hnot` field
-- We do not directly extend `CompleteBooleanAlgebra` to avoid having the `inf_sSup_le_iSup_inf` and
-- `iInf_sup_le_sup_sInf` fields
class CompleteAtomicBooleanAlgebra (α : Type u) extends CompleteLattice α, BooleanAlgebra α where
  protected iInf_iSup_eq {ι : Type u} {κ : ι → Type u} (f : ∀ a, κ a → α) :
    (⨅ a, ⨆ b, f a b) = ⨆ g : ∀ a, κ a, ⨅ a, f a (g a)

-- See note [lower instance priority]

instance (priority := 100) CompleteAtomicBooleanAlgebra.toCompletelyDistribLattice
    [CompleteAtomicBooleanAlgebra α] : CompletelyDistribLattice α where
  __ := ‹CompleteAtomicBooleanAlgebra α›
  __ := BooleanAlgebra.toBiheytingAlgebra

-- See note [lower instance priority]

instance (priority := 100) CompleteAtomicBooleanAlgebra.toCompleteBooleanAlgebra
    [CompleteAtomicBooleanAlgebra α] : CompleteBooleanAlgebra α where
  __ := ‹CompleteAtomicBooleanAlgebra α›
  __ := CompletelyDistribLattice.toCompleteDistribLattice


instance Prod.instCompleteAtomicBooleanAlgebra [CompleteAtomicBooleanAlgebra α]
    [CompleteAtomicBooleanAlgebra β] : CompleteAtomicBooleanAlgebra (α × β) where
  __ := instBooleanAlgebra
  __ := instCompletelyDistribLattice


instance Pi.instCompleteAtomicBooleanAlgebra {ι : Type*} {π : ι → Type*}
    [∀ i, CompleteAtomicBooleanAlgebra (π i)] : CompleteAtomicBooleanAlgebra (∀ i, π i) where
  __ := Pi.instCompleteBooleanAlgebra
                       /-
                         α : Type u
                         β : Type v
                         ι✝¹ : Sort w
                         κ : ι✝¹ → Sort w'
                         ι : Type u_1
                         π : ι → Type u_2
                         inst✝ : (i : ι) → CompleteAtomicBooleanAlgebra (π i)
                         ι✝ : Type (max u_1 u_2)
                         κ✝ : ι✝ → Type (max u_1 u_2)
                         f : (a : ι✝) → κ✝ a → (i : ι) → π i
                         ⊢ Eq (iInf fun a => iSup fun b => f a b) (iSup fun g => iInf fun a => f a (g a))
                       -/
  iInf_iSup_eq f := by ext; rw [iInf_iSup_eq]
                            /-
                              🎉 no goals
                            -/


instance OrderDual.instCompleteAtomicBooleanAlgebra [CompleteAtomicBooleanAlgebra α] :
    CompleteAtomicBooleanAlgebra αᵒᵈ where
  __ := instCompleteBooleanAlgebra
  __ := instCompletelyDistribLattice


instance Prop.instCompleteAtomicBooleanAlgebra : CompleteAtomicBooleanAlgebra Prop where
  __ := Prop.instCompleteLattice
  __ := Prop.instBooleanAlgebra
                       /-
                         α : Type u
                         β : Type v
                         ι : Sort w
                         κ : ι → Sort w'
                         ι✝ : Type
                         κ✝ : ι✝ → Type
                         f : (a : ι✝) → κ✝ a → Prop
                         ⊢ Eq (iInf fun a => iSup fun b => f a b) (iSup fun g => iInf fun a => f a (g a))
                       -/
  iInf_iSup_eq f := by simp [Classical.skolem]
                       /-
                         🎉 no goals
                       -/


instance Prop.instCompleteBooleanAlgebra : CompleteBooleanAlgebra Prop := inferInstance


/-- Pullback an `Order.Frame.MinimalAxioms` along an injection. -/
protected abbrev Function.Injective.frameMinimalAxioms [Max α] [Min α] [SupSet α] [InfSet α] [Top α]
    [Bot α] (minAx : Frame.MinimalAxioms β) (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) : Frame.MinimalAxioms α where
  __ := hf.completeLattice f map_sup map_inf map_sSup map_sInf map_top map_bot
  inf_sSup_le_iSup_inf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Frame.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    change f (a ⊓ sSup s) ≤ f _
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Frame.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (f (Min.min a (SupSet.sSup s))) (f (iSup fun b => iSup fun h => Min.mi …
    -/
    rw [← sSup_image, map_inf, map_sSup s, minAx.inf_iSup₂_eq]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Frame.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (iSup fun i => iSup fun j => Min.min (f a) (f i)) (f (SupSet.sSup (Set …
    -/
    simp_rw [← map_inf]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Frame.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (iSup fun i => iSup fun x => f (Min.min a i)) (f (SupSet.sSup (Set.ima …
    -/
    exact ((map_sSup _).trans iSup_image).ge
    /-
      🎉 no goals
    -/

-- See note [reducible non-instances]

/-- Pullback an `Order.Coframe.MinimalAxioms` along an injection. -/
protected abbrev Function.Injective.coframeMinimalAxioms [Max α] [Min α] [SupSet α] [InfSet α]
    [Top α] [Bot α] (minAx : Coframe.MinimalAxioms β) (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) : Coframe.MinimalAxioms α where
  __ := hf.completeLattice f map_sup map_inf map_sSup map_sInf map_top map_bot
  iInf_sup_le_sup_sInf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Coframe.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    change f _ ≤ f (a ⊔ sInf s)
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Coframe.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (f (iInf fun b => iInf fun h => Max.max a b)) (f (Max.max a (InfSet.sI …
    -/
    rw [← sInf_image, map_sup, map_sInf s, minAx.sup_iInf₂_eq]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Coframe.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (f (InfSet.sInf (Set.image (Max.max a) s))) (iInf fun i => iInf fun j  …
    -/
    simp_rw [← map_sup]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁵ : Max α
      inst✝⁴ : Min α
      inst✝³ : SupSet α
      inst✝² : InfSet α
      inst✝¹ : Top α
      inst✝ : Bot α
      minAx : Order.Coframe.MinimalAxioms β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      a : α
      s : Set α
      ⊢ LE.le (f (InfSet.sInf (Set.image (fun a_1 => Max.max a a_1) s))) (iInf fun i …
    -/
    exact ((map_sInf _).trans iInf_image).le
    /-
      🎉 no goals
    -/

-- See note [reducible non-instances]

/-- Pullback an `Order.Frame` along an injection. -/
protected abbrev Function.Injective.frame [Max α] [Min α] [SupSet α] [InfSet α] [Top α] [Bot α]
    [HasCompl α] [HImp α] [Frame β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) (map_compl : ∀ a, f aᶜ = (f a)ᶜ)
    (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b) : Frame α where
  __ := hf.frameMinimalAxioms .of f map_sup map_inf map_sSup map_sInf map_top map_bot
  __ := hf.heytingAlgebra f map_sup map_inf map_top map_bot map_compl map_himp

-- See note [reducible non-instances]

/-- Pullback an `Order.Coframe` along an injection. -/
protected abbrev Function.Injective.coframe [Max α] [Min α] [SupSet α] [InfSet α] [Top α] [Bot α]
    [HNot α] [SDiff α] [Coframe β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) (map_hnot : ∀ a, f (￢a) = ￢f a)
    (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) : Coframe α where
  __ := hf.coframeMinimalAxioms .of f map_sup map_inf map_sSup map_sInf map_top map_bot
  __ := hf.coheytingAlgebra f map_sup map_inf map_top map_bot map_hnot map_sdiff

-- See note [reducible non-instances]

/-- Pullback a `CompleteDistribLattice.MinimalAxioms` along an injection. -/
protected abbrev Function.Injective.completeDistribLatticeMinimalAxioms [Max α] [Min α] [SupSet α]
    [InfSet α] [Top α] [Bot α] (minAx : CompleteDistribLattice.MinimalAxioms β) (f : α → β)
    (hf : Injective f) (map_sup : let _ := minAx.toCompleteLattice
      ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : let _ := minAx.toCompleteLattice
      ∀ a b, f (a ⊓ b) = f a ⊓ f b) (map_sSup : let _ := minAx.toCompleteLattice
      ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : let _ := minAx.toCompleteLattice
      ∀ s, f (sInf s) = ⨅ a ∈ s, f a) (map_top : let _ := minAx.toCompleteLattice
      f ⊤ = ⊤) (map_bot : let _ := minAx.toCompleteLattice
      f ⊥ = ⊥) :
    CompleteDistribLattice.MinimalAxioms α where
  __ := hf.frameMinimalAxioms minAx.toFrame f map_sup map_inf map_sSup map_sInf map_top map_bot
  __ := hf.coframeMinimalAxioms minAx.toCoframe f map_sup map_inf map_sSup map_sInf map_top map_bot

-- See note [reducible non-instances]

/-- Pullback a `CompleteDistribLattice` along an injection. -/
protected abbrev Function.Injective.completeDistribLattice [Max α] [Min α] [SupSet α] [InfSet α]
    [Top α] [Bot α] [HasCompl α] [HImp α] [HNot α] [SDiff α] [CompleteDistribLattice β] (f : α → β)
    (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥)
    (map_compl : ∀ a, f aᶜ = (f a)ᶜ) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b)
    (map_hnot : ∀ a, f (￢a) = ￢f a) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    CompleteDistribLattice α where
  __ := hf.frame f map_sup map_inf map_sSup map_sInf map_top map_bot map_compl map_himp
  __ := hf.coframe f map_sup map_inf map_sSup map_sInf map_top map_bot map_hnot map_sdiff

-- See note [reducible non-instances]

/-- Pullback a `CompletelyDistribLattice.MinimalAxioms` along an injection. -/
protected abbrev Function.Injective.completelyDistribLatticeMinimalAxioms [Max α] [Min α] [SupSet α]
    [InfSet α] [Top α] [Bot α] (minAx : CompletelyDistribLattice.MinimalAxioms β) (f : α → β)
    (hf : Injective f) (map_sup : let _ := minAx.toCompleteLattice
      ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : let _ := minAx.toCompleteLattice
      ∀ a b, f (a ⊓ b) = f a ⊓ f b) (map_sSup : let _ := minAx.toCompleteLattice
      ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : let _ := minAx.toCompleteLattice
      ∀ s, f (sInf s) = ⨅ a ∈ s, f a) (map_top : let _ := minAx.toCompleteLattice
      f ⊤ = ⊤) (map_bot : let _ := minAx.toCompleteLattice
      f ⊥ = ⊥) :
    CompletelyDistribLattice.MinimalAxioms α where
  __ := hf.completeDistribLatticeMinimalAxioms minAx.toCompleteDistribLattice f map_sup map_inf
    map_sSup map_sInf map_top map_bot
  iInf_iSup_eq g := hf <| by
    simp_rw [iInf, map_sInf, iInf_range, iSup, map_sSup, iSup_range, map_sInf, iInf_range,
      minAx.iInf_iSup_eq']

-- See note [reducible non-instances]

/-- Pullback a `CompletelyDistribLattice` along an injection. -/
protected abbrev Function.Injective.completelyDistribLattice [Max α] [Min α] [SupSet α] [InfSet α]
    [Top α] [Bot α] [HasCompl α] [HImp α] [HNot α] [SDiff α] [CompletelyDistribLattice β]
    (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥)
    (map_compl : ∀ a, f aᶜ = (f a)ᶜ) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b)
    (map_hnot : ∀ a, f (￢a) = ￢f a) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    CompletelyDistribLattice α where
  __ := hf.completeLattice f map_sup map_inf map_sSup map_sInf map_top map_bot
  __ := hf.biheytingAlgebra f map_sup map_inf map_top map_bot map_compl map_hnot map_himp map_sdiff
  iInf_iSup_eq g := hf <| by
    simp_rw [iInf, map_sInf, iInf_range, iSup, map_sSup, iSup_range, map_sInf, iInf_range,
      iInf_iSup_eq]

-- See note [reducible non-instances]

/-- Pullback a `CompleteBooleanAlgebra` along an injection. -/
protected abbrev Function.Injective.completeBooleanAlgebra [Max α] [Min α] [SupSet α] [InfSet α]
    [Top α] [Bot α] [HasCompl α] [HImp α] [SDiff α] [CompleteBooleanAlgebra β] (f : α → β)
    (hf : Injective f) (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b)
    (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b) (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a)
    (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a) (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥)
    (map_compl : ∀ a, f aᶜ = (f a)ᶜ) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b)
    (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    CompleteBooleanAlgebra α where
  __ := hf.completeLattice f map_sup map_inf map_sSup map_sInf map_top map_bot
  __ := hf.booleanAlgebra f map_sup map_inf map_top map_bot map_compl map_sdiff map_himp
  inf_sSup_le_iSup_inf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    change f (a ⊓ sSup s) ≤ f _
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (f (Min.min a (SupSet.sSup s))) (f (iSup fun b => iSup fun h => Min.mi …
    -/
    rw [← sSup_image, map_inf, map_sSup s, inf_iSup₂_eq]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (iSup fun i => iSup fun j => Min.min (f a) (f i)) (f (SupSet.sSup (Set …
    -/
    simp_rw [← map_inf]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (iSup fun i => iSup fun x => f (Min.min a i)) (f (SupSet.sSup (Set.ima …
    -/
    exact ((map_sSup _).trans iSup_image).ge
    /-
      🎉 no goals
    -/
  iInf_sup_le_sup_sInf a s := by
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    change f _ ≤ f (a ⊔ sInf s)
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (f (iInf fun b => iInf fun h => Max.max a b)) (f (Max.max a (InfSet.sI …
    -/
    rw [← sInf_image, map_sup, map_sInf s, sup_iInf₂_eq]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (f (InfSet.sInf (Set.image (Max.max a) s))) (iInf fun i => iInf fun j  …
    -/
    simp_rw [← map_sup]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      κ : ι → Sort w'
      inst✝⁹ : Max α
      inst✝⁸ : Min α
      inst✝⁷ : SupSet α
      inst✝⁶ : InfSet α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : HImp α
      inst✝¹ : SDiff α
      inst✝ : CompleteBooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_sSup : ∀ (s : Set α), Eq (f (SupSet.sSup s)) (iSup fun a => iSup fun h =>  …
      map_sInf : ∀ (s : Set α), Eq (f (InfSet.sInf s)) (iInf fun a => iInf fun h =>  …
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      a : α
      s : Set α
      ⊢ LE.le (f (InfSet.sInf (Set.image (fun a_1 => Max.max a a_1) s))) (iInf fun i …
    -/
    exact ((map_sInf _).trans iInf_image).le
    /-
      🎉 no goals
    -/

-- See note [reducible non-instances]

/-- Pullback a `CompleteAtomicBooleanAlgebra` along an injection. -/
protected abbrev Function.Injective.completeAtomicBooleanAlgebra [Max α] [Min α] [SupSet α]
    [InfSet α] [Top α] [Bot α] [HasCompl α] [HImp α] [HNot α] [SDiff α]
    [CompleteAtomicBooleanAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_sSup : ∀ s, f (sSup s) = ⨆ a ∈ s, f a) (map_sInf : ∀ s, f (sInf s) = ⨅ a ∈ s, f a)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥)
    (map_compl : ∀ a, f aᶜ = (f a)ᶜ) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b)
    (map_hnot : ∀ a, f (￢a) = ￢f a) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    CompleteAtomicBooleanAlgebra α where
  __ := hf.completelyDistribLattice f map_sup map_inf map_sSup map_sInf map_top map_bot map_compl
    map_himp map_hnot map_sdiff
  __ := hf.booleanAlgebra f map_sup map_inf map_top map_bot map_compl map_sdiff map_himp


instance instCompleteAtomicBooleanAlgebra : CompleteAtomicBooleanAlgebra PUnit where
  __ := PUnit.instBooleanAlgebra
  sSup _ := unit
  sInf _ := unit
  le_sSup _ _ _ := trivial
  sSup_le _ _ _ := trivial
  sInf_le _ _ _ := trivial
  le_sInf _ _ _ := trivial
  iInf_iSup_eq _ := rfl


instance instCompleteBooleanAlgebra : CompleteBooleanAlgebra PUnit := inferInstance


@[simp]
theorem sSup_eq : sSup s = unit :=
  rfl


@[simp]
theorem sInf_eq : sInf s = unit :=
  rfl


