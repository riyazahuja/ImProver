theorem IsCofinal.of_isEmpty [IsEmpty α] (s : Set α) : IsCofinal s :=
  fun a ↦ isEmptyElim a


theorem isCofinal_empty_iff : IsCofinal (∅ : Set α) ↔ IsEmpty α := by
  /-
    α : Type u_1
    inst✝ : LE α
    ⊢ Iff (IsCofinal EmptyCollection.emptyCollection) (IsEmpty α)
  -/
  refine ⟨fun h ↦ ⟨fun a ↦ ?_⟩, fun h ↦ .of_isEmpty _⟩
  /-
    α : Type u_1
    inst✝ : LE α
    h : IsCofinal EmptyCollection.emptyCollection
    a : α
    ⊢ False
  -/
  simpa using h a
  /-
    🎉 no goals
  -/


theorem IsCofinal.singleton_top [OrderTop α] : IsCofinal {(⊤ : α)} :=
  fun _ ↦ ⟨⊤, Set.mem_singleton _, le_top⟩


theorem IsCofinal.mono {s t : Set α} (h : s ⊆ t) (hs : IsCofinal s) : IsCofinal t := by
  /-
    α : Type u_1
    inst✝ : LE α
    s t : Set α
    h : HasSubset.Subset s t
    hs : IsCofinal s
    ⊢ IsCofinal t
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : LE α
    s t : Set α
    h : HasSubset.Subset s t
    hs : IsCofinal s
    a : α
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le a y)
  -/
  obtain ⟨b, hb, hb'⟩ := hs a
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LE α
    s t : Set α
    h : HasSubset.Subset s t
    hs : IsCofinal s
    a b : α
    hb : Membership.mem s b
    hb' : LE.le a b
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le a y)
  -/
  exact ⟨b, h hb, hb'⟩
  /-
    🎉 no goals
  -/


theorem IsCofinal.univ : IsCofinal (@Set.univ α) :=
  fun a ↦ ⟨a, ⟨⟩, le_rfl⟩


instance : Inhabited {s : Set α // IsCofinal s} :=
  ⟨_, .univ⟩


/-- A cofinal subset of a cofinal subset is cofinal. -/
theorem IsCofinal.trans {s : Set α} {t : Set s} (hs : IsCofinal s) (ht : IsCofinal t) :
    IsCofinal (Subtype.val '' t) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    t : Set ↑s
    hs : IsCofinal s
    ht : IsCofinal t
    ⊢ IsCofinal (Set.image Subtype.val t)
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    t : Set ↑s
    hs : IsCofinal s
    ht : IsCofinal t
    a : α
    ⊢ Exists fun y => And (Membership.mem (Set.image Subtype.val t) y) (LE.le a y)
  -/
  obtain ⟨b, hb, hb'⟩ := hs a
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    t : Set ↑s
    hs : IsCofinal s
    ht : IsCofinal t
    a b : α
    hb : Membership.mem s b
    hb' : LE.le a b
    ⊢ Exists fun y => And (Membership.mem (Set.image Subtype.val t) y) (LE.le a y)
  -/
  obtain ⟨c, hc, hc'⟩ := ht ⟨b, hb⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    t : Set ↑s
    hs : IsCofinal s
    ht : IsCofinal t
    a b : α
    hb : Membership.mem s b
    hb' : LE.le a b
    c : ↑s
    hc : Membership.mem t c
    hc' : LE.le ⟨b, hb⟩ c
    ⊢ Exists fun y => And (Membership.mem (Set.image Subtype.val t) y) (LE.le a y)
  -/
  exact ⟨c, Set.mem_image_of_mem _ hc, hb'.trans hc'⟩
  /-
    🎉 no goals
  -/


theorem GaloisConnection.map_cofinal [Preorder β] {f : β → α} {g : α → β}
    (h : GaloisConnection f g) {s : Set α} (hs : IsCofinal s) : IsCofinal (g '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : β → α
    g : α → β
    h : GaloisConnection f g
    s : Set α
    hs : IsCofinal s
    ⊢ IsCofinal (Set.image g s)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : β → α
    g : α → β
    h : GaloisConnection f g
    s : Set α
    hs : IsCofinal s
    a : β
    ⊢ Exists fun y => And (Membership.mem (Set.image g s) y) (LE.le a y)
  -/
  obtain ⟨b, hb, hb'⟩ := hs (f a)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : β → α
    g : α → β
    h : GaloisConnection f g
    s : Set α
    hs : IsCofinal s
    a : β
    b : α
    hb : Membership.mem s b
    hb' : LE.le (f a) b
    ⊢ Exists fun y => And (Membership.mem (Set.image g s) y) (LE.le a y)
  -/
  exact ⟨g b, Set.mem_image_of_mem _ hb, h.le_iff_le.1 hb'⟩
  /-
    🎉 no goals
  -/


theorem OrderIso.map_cofinal [Preorder β] (e : α ≃o β) {s : Set α} (hs : IsCofinal s) :
    IsCofinal (e '' s) :=
  e.symm.to_galoisConnection.map_cofinal hs


theorem IsCofinal.mem_of_isMax {s : Set α} {a : α} (ha : IsMax a) (hs : IsCofinal s) : a ∈ s := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    a : α
    ha : IsMax a
    hs : IsCofinal s
    ⊢ Membership.mem s a
  -/
  obtain ⟨b, hb, hb'⟩ := hs a
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    a : α
    ha : IsMax a
    hs : IsCofinal s
    b : α
    hb : Membership.mem s b
    hb' : LE.le a b
    ⊢ Membership.mem s a
  -/
  rwa [ha.eq_of_ge hb'] at hb
  /-
    🎉 no goals
  -/


theorem IsCofinal.top_mem [OrderTop α] {s : Set α} (hs : IsCofinal s) : ⊤ ∈ s :=
  hs.mem_of_isMax isMax_top


@[simp]
theorem isCofinal_iff_top_mem [OrderTop α] {s : Set α} : IsCofinal s ↔ ⊤ ∈ s :=
  ⟨IsCofinal.top_mem, fun hs _ ↦ ⟨⊤, hs, le_top⟩⟩


theorem not_isCofinal_iff {s : Set α} : ¬ IsCofinal s ↔ ∃ x, ∀ y ∈ s, y < x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ Iff (Not (IsCofinal s)) (Exists fun x => ∀ (y : α), Membership.mem s y → LT. …
  -/
  simp [IsCofinal]
  /-
    🎉 no goals
  -/


theorem BddAbove.of_not_isCofinal {s : Set α} (h : ¬ IsCofinal s) : BddAbove s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    h : Not (IsCofinal s)
    ⊢ BddAbove s
  -/
  rw [not_isCofinal_iff] at h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    h : Exists fun x => ∀ (y : α), Membership.mem s y → LT.lt y x
    ⊢ BddAbove s
  -/
  obtain ⟨x, h⟩ := h
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : α
    h : ∀ (y : α), Membership.mem s y → LT.lt y x
    ⊢ BddAbove s
  -/
  exact ⟨x, fun y hy ↦ (h y hy).le⟩
  /-
    🎉 no goals
  -/


theorem IsCofinal.of_not_bddAbove {s : Set α} (h : ¬ BddAbove s) : IsCofinal s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    h : Not (BddAbove s)
    ⊢ IsCofinal s
  -/
  contrapose! h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    h : Not (IsCofinal s)
    ⊢ BddAbove s
  -/
  exact .of_not_isCofinal h
  /-
    🎉 no goals
  -/


/-- In a linear order with no maximum, cofinal sets are the same as unbounded sets. -/
theorem not_isCofinal_iff_bddAbove [NoMaxOrder α] {s : Set α} : ¬ IsCofinal s ↔ BddAbove s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    s : Set α
    ⊢ Iff (Not (IsCofinal s)) (BddAbove s)
  -/
  use .of_not_isCofinal
  /-
    case mpr
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    s : Set α
    ⊢ BddAbove s → Not (IsCofinal s)
  -/
  rw [not_isCofinal_iff]
  /-
    case mpr
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    s : Set α
    ⊢ BddAbove s → Exists fun x => ∀ (y : α), Membership.mem s y → LT.lt y x
  -/
  rintro ⟨x, h⟩
  /-
    case mpr.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    s : Set α
    x : α
    h : Membership.mem (upperBounds s) x
    ⊢ Exists fun x => ∀ (y : α), Membership.mem s y → LT.lt y x
  -/
  obtain ⟨z, hz⟩ := exists_gt x
  /-
    case mpr.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    s : Set α
    x : α
    h : Membership.mem (upperBounds s) x
    z : α
    hz : LT.lt x z
    ⊢ Exists fun x => ∀ (y : α), Membership.mem s y → LT.lt y x
  -/
  exact ⟨z, fun y hy ↦ (h hy).trans_lt hz⟩
  /-
    🎉 no goals
  -/


/-- In a linear order with no maximum, cofinal sets are the same as unbounded sets. -/
theorem not_bddAbove_iff_isCofinal [NoMaxOrder α] {s : Set α} : ¬ BddAbove s ↔ IsCofinal s :=
  not_iff_comm.1 not_isCofinal_iff_bddAbove


/-- The set of "records" (the smallest inputs yielding the highest values) with respect to a
well-ordering of `α` is a cofinal set. -/
theorem isCofinal_setOf_imp_lt (r : α → α → Prop) [h : IsWellFounded α r] :
    IsCofinal { a | ∀ b, r b a → b < a } := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    r : α → α → Prop
    h : IsWellFounded α r
    ⊢ IsCofinal (setOf fun a => ∀ (b : α), r b a → LT.lt b a)
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    r : α → α → Prop
    h : IsWellFounded α r
    a : α
    ⊢ Exists fun y => And (Membership.mem (setOf fun a => ∀ (b : α), r b a → LT.lt …
  -/
  obtain ⟨b, hb, hb'⟩ := h.wf.has_min (Set.Ici a) Set.nonempty_Ici
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    r : α → α → Prop
    h : IsWellFounded α r
    a b : α
    hb : Membership.mem (Set.Ici a) b
    hb' : ∀ (x : α), Membership.mem (Set.Ici a) x → Not (r x b)
    ⊢ Exists fun y => And (Membership.mem (setOf fun a => ∀ (b : α), r b a → LT.lt …
  -/
  refine ⟨b, fun c hc ↦ ?_, hb⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    r : α → α → Prop
    h : IsWellFounded α r
    a b : α
    hb : Membership.mem (Set.Ici a) b
    hb' : ∀ (x : α), Membership.mem (Set.Ici a) x → Not (r x b)
    c : α
    hc : r c b
    ⊢ LT.lt c b
  -/
  by_contra! hc'
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    r : α → α → Prop
    h : IsWellFounded α r
    a b : α
    hb : Membership.mem (Set.Ici a) b
    hb' : ∀ (x : α), Membership.mem (Set.Ici a) x → Not (r x b)
    c : α
    hc : r c b
    hc' : LE.le b c
    ⊢ False
  -/
  exact hb' c (hb.trans hc') hc
  /-
    🎉 no goals
  -/


