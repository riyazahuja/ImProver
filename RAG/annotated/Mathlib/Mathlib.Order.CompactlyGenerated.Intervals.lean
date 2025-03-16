theorem isCompactElement {a : α} {b : Iic a} (h : CompleteLattice.IsCompactElement (b : α)) :
    CompleteLattice.IsCompactElement b := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    b : ↑(Set.Iic a)
    h : CompleteLattice.IsCompactElement ↑b
    ⊢ CompleteLattice.IsCompactElement b
  -/
  simp only [CompleteLattice.isCompactElement_iff, Finset.sup_eq_iSup] at h ⊢
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    b : ↑(Set.Iic a)
    h : ∀ (ι : Type u_2) (s : ι → α), LE.le (↑b) (iSup s) → Exists fun t => LE.le  …
    ⊢ ∀ (ι : Type u_2) (s : ι → ↑(Set.Iic a)), LE.le b (iSup s) → Exists fun t =>  …
  -/
  intro ι s hb
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    b : ↑(Set.Iic a)
    h : ∀ (ι : Type u_2) (s : ι → α), LE.le (↑b) (iSup s) → Exists fun t => LE.le  …
    ι : Type u_2
    s : ι → ↑(Set.Iic a)
    hb : LE.le b (iSup s)
    ⊢ Exists fun t => LE.le b (iSup fun a_1 => iSup fun h => s a_1)
  -/
  replace hb : (b : α) ≤ iSup ((↑) ∘ s) := le_trans hb <| (coe_iSup s) ▸ le_refl _
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    b : ↑(Set.Iic a)
    h : ∀ (ι : Type u_2) (s : ι → α), LE.le (↑b) (iSup s) → Exists fun t => LE.le  …
    ι : Type u_2
    s : ι → ↑(Set.Iic a)
    hb : LE.le (↑b) (iSup (Function.comp Subtype.val s))
    ⊢ Exists fun t => LE.le b (iSup fun a_1 => iSup fun h => s a_1)
  -/
  obtain ⟨t, ht⟩ := h ι ((↑) ∘ s) hb
  /-
    case intro
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    b : ↑(Set.Iic a)
    h : ∀ (ι : Type u_2) (s : ι → α), LE.le (↑b) (iSup s) → Exists fun t => LE.le  …
    ι : Type u_2
    s : ι → ↑(Set.Iic a)
    hb : LE.le (↑b) (iSup (Function.comp Subtype.val s))
    t : Finset ι
    ht : LE.le (↑b) (iSup fun a_1 => iSup fun h => Function.comp Subtype.val s a_1)
    ⊢ Exists fun t => LE.le b (iSup fun a_1 => iSup fun h => s a_1)
  -/
  exact ⟨t, (by simpa using ht : (b : α) ≤ _)⟩
  /-
    🎉 no goals
  -/


instance instIsCompactlyGenerated [IsCompactlyGenerated α] {a : α} :
    IsCompactlyGenerated (Iic a) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    ⊢ IsCompactlyGenerated ↑(Set.Iic a)
  -/
  refine ⟨fun ⟨x, (hx : x ≤ a)⟩ ↦ ?_⟩
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    x✝ : ↑(Set.Iic a)
    x : α
    hx : LE.le x a
    ⊢ Exists fun s => And (∀ (x : ↑(Set.Iic a)), Membership.mem s x → CompleteLatt …
  -/
  obtain ⟨s, hs, rfl⟩ := IsCompactlyGenerated.exists_sSup_eq x
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    x✝ : ↑(Set.Iic a)
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
    hx : LE.le (SupSet.sSup s) a
    ⊢ Exists fun s_1 => And (∀ (x : ↑(Set.Iic a)), Membership.mem s_1 x → Complete …
  -/
  rw [sSup_le_iff] at hx
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    x✝ : ↑(Set.Iic a)
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
    hx✝ : LE.le (SupSet.sSup s) a
    hx : ∀ (b : α), Membership.mem s b → LE.le b a
    ⊢ Exists fun s_1 => And (∀ (x : ↑(Set.Iic a)), Membership.mem s_1 x → Complete …
  -/
  let f : s → Iic a := fun y ↦ ⟨y, hx _ y.property⟩
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    x✝ : ↑(Set.Iic a)
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
    hx✝ : LE.le (SupSet.sSup s) a
    hx : ∀ (b : α), Membership.mem s b → LE.le b a
    f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
    ⊢ Exists fun s_1 => And (∀ (x : ↑(Set.Iic a)), Membership.mem s_1 x → Complete …
  -/
  refine ⟨range f, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      ⊢ ∀ (x : ↑(Set.Iic a)), Membership.mem (Set.range f) x → CompleteLattice.IsCom …
    -/
  · rintro - ⟨⟨y, hy⟩, hy', rfl⟩
    /-
      case intro.intro.refine_1.intro.mk.refl
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      y : α
      hy : Membership.mem s y
      ⊢ CompleteLattice.IsCompactElement (f ⟨y, hy⟩)
    -/
    exact isCompactElement (hs _ hy)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      ⊢ Eq (SupSet.sSup (Set.range f)) ⟨SupSet.sSup s, hx✝⟩
    -/
  · rw [Subtype.ext_iff]
    /-
      case intro.intro.refine_2
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      ⊢ Eq ↑(SupSet.sSup (Set.range f)) ↑⟨SupSet.sSup s, hx✝⟩
    -/
    change sSup (((↑) : Iic a → α) '' (range f)) = sSup s
    /-
      case intro.intro.refine_2
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      ⊢ Eq (SupSet.sSup (Set.image Subtype.val (Set.range f))) (SupSet.sSup s)
    -/
    congr
    /-
      case intro.intro.refine_2.e_a
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      ⊢ Eq (Set.image Subtype.val (Set.range f)) s
    -/
    ext b
    /-
      case intro.intro.refine_2.e_a.h
      ι : Type u_1
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a : α
      x✝ : ↑(Set.Iic a)
      s : Set α
      hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
      hx✝ : LE.le (SupSet.sSup s) a
      hx : ∀ (b : α), Membership.mem s b → LE.le b a
      f : ↑s → ↑(Set.Iic a) := fun y => ⟨↑y, ⋯⟩
      b : α
      ⊢ Iff (Membership.mem (Set.image Subtype.val (Set.range f)) b) (Membership.mem …
    -/
    simpa [f] using hx b
    /-
      🎉 no goals
    -/


theorem complementedLattice_of_complementedLattice_Iic
    [IsModularLattice α] [IsCompactlyGenerated α]
    {s : Set ι} {f : ι → α}
    (h : ∀ i ∈ s, ComplementedLattice <| Iic (f i))
    (h' : ⨆ i ∈ s, f i = ⊤) :
    ComplementedLattice α := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    ⊢ ComplementedLattice α
  -/
  apply complementedLattice_of_sSup_atoms_eq_top
  have : ∀ i ∈ s, ∃ t : Set α, f i = sSup t ∧ ∀ a ∈ t, IsAtom a := fun i hi ↦ by
    replace h := complementedLattice_iff_isAtomistic.mp (h i hi)
    obtain ⟨u, hu, hu'⟩ := eq_sSup_atoms (⊤ : Iic (f i))
    refine ⟨(↑) '' u, ?_, ?_⟩
    · replace hu : f i = ↑(sSup u) := Subtype.ext_iff.mp hu
      simp_rw [hu, Iic.coe_sSup]
    · rintro b ⟨⟨a, ha'⟩, ha, rfl⟩
      exact IsAtom.of_isAtom_coe_Iic (hu' _ ha)
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    this : ∀ (i : ι), Membership.mem s i → Exists fun t => And (Eq (f i) (SupSet.s …
    ⊢ Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
  -/
  choose t ht ht' using this
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (a : Membership.mem s i), Eq (f i) (SupSet.sSup (t i a))
    ht' : ∀ (i : ι) (a : Membership.mem s i) (a_1 : α), Membership.mem (t i a) a_1 …
    ⊢ Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
  -/
  let u : Set α := ⋃ i, ⋃ hi : i ∈ s, t i hi
  have hu₁ : u ⊆ {a | IsAtom a} := by
    rintro a ⟨-, ⟨i, rfl⟩, ⟨-, ⟨hi, rfl⟩, ha : a ∈ t i hi⟩⟩
    exact ht' i hi a ha
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (a : Membership.mem s i), Eq (f i) (SupSet.sSup (t i a))
    ht' : ∀ (i : ι) (a : Membership.mem s i) (a_1 : α), Membership.mem (t i a) a_1 …
    u : Set α := Set.iUnion fun i => Set.iUnion fun hi => t i hi
    hu₁ : HasSubset.Subset u (setOf fun a => IsAtom a)
    ⊢ Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
  -/
  have hu₂ : sSup u = ⨆ i ∈ s, f i := by simp_rw [u, sSup_iUnion, biSup_congr' ht]
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (a : Membership.mem s i), Eq (f i) (SupSet.sSup (t i a))
    ht' : ∀ (i : ι) (a : Membership.mem s i) (a_1 : α), Membership.mem (t i a) a_1 …
    u : Set α := Set.iUnion fun i => Set.iUnion fun hi => t i hi
    hu₁ : HasSubset.Subset u (setOf fun a => IsAtom a)
    hu₂ : Eq (SupSet.sSup u) (iSup fun i => iSup fun h => f i)
    ⊢ Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
  -/
  rw [eq_top_iff, ← h', ← hu₂]
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set ι
    f : ι → α
    h : ∀ (i : ι), Membership.mem s i → ComplementedLattice ↑(Set.Iic (f i))
    h' : Eq (iSup fun i => iSup fun h => f i) Top.top
    t : (i : ι) → Membership.mem s i → Set α
    ht : ∀ (i : ι) (a : Membership.mem s i), Eq (f i) (SupSet.sSup (t i a))
    ht' : ∀ (i : ι) (a : Membership.mem s i) (a_1 : α), Membership.mem (t i a) a_1 …
    u : Set α := Set.iUnion fun i => Set.iUnion fun hi => t i hi
    hu₁ : HasSubset.Subset u (setOf fun a => IsAtom a)
    hu₂ : Eq (SupSet.sSup u) (iSup fun i => iSup fun h => f i)
    ⊢ LE.le (SupSet.sSup u) (SupSet.sSup (setOf fun a => IsAtom a))
  -/
  exact sSup_le_sSup hu₁
  /-
    🎉 no goals
  -/

