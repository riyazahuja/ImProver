/-- This is important for iterating Kruskal-Katona: the shadow of an initial segment is also an
initial segment. -/
lemma shadow_initSeg [Fintype α] (hs : s.Nonempty) :
    ∂ (initSeg s) = initSeg (erase s <| min' s hs) := by
  -- This is a pretty painful proof, with lots of cases.
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    ⊢ Eq (Finset.Colex.initSeg s).shadow (Finset.Colex.initSeg (s.erase (s.min' hs …
  -/
  ext t
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    ⊢ Iff (Membership.mem (Finset.Colex.initSeg s).shadow t) (Membership.mem (Fins …
  -/
  simp only [mem_shadow_iff_insert_mem, mem_initSeg, exists_prop]
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    ⊢ Iff (Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert. …
  -/
  constructor
  -- First show that if t ∪ a ≤ s, then t ≤ s - min s
    /-
      case h.mp
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      ⊢ (Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert.inse …
    -/
  · rintro ⟨a, ha, hst, hts⟩
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      a : α
      ha : Not (Membership.mem t a)
      hst : Eq s.card (Insert.insert a t).card
      hts : LE.le { ofColex := Insert.insert a t } { ofColex := s }
      ⊢ And (Eq (s.erase (s.min' hs)).card t.card) (LE.le { ofColex := t } { ofColex …
    -/
    constructor
      /-
        case h.mp.intro.intro.intro.left
        α : Type u_1
        inst✝¹ : LinearOrder α
        s : Finset α
        inst✝ : Fintype α
        hs : s.Nonempty
        t : Finset α
        a : α
        ha : Not (Membership.mem t a)
        hst : Eq s.card (Insert.insert a t).card
        hts : LE.le { ofColex := Insert.insert a t } { ofColex := s }
        ⊢ Eq (s.erase (s.min' hs)).card t.card
      -/
    · rw [card_erase_of_mem (min'_mem _ _), hst, card_insert_of_not_mem ha, add_tsub_cancel_right]
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.right
        α : Type u_1
        inst✝¹ : LinearOrder α
        s : Finset α
        inst✝ : Fintype α
        hs : s.Nonempty
        t : Finset α
        a : α
        ha : Not (Membership.mem t a)
        hst : Eq s.card (Insert.insert a t).card
        hts : LE.le { ofColex := Insert.insert a t } { ofColex := s }
        ⊢ LE.le { ofColex := t } { ofColex := s.erase (s.min' hs) }
      -/
    · simpa [ha] using erase_le_erase_min' hts hst.ge (mem_insert_self _ _)
      /-
        🎉 no goals
      -/
  -- Now show that if t ≤ s - min s, there is j such that t ∪ j ≤ s
  -- We choose j as the smallest thing not in t
  /-
    case h.mpr
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    ⊢ And (Eq (s.erase (s.min' hs)).card t.card) (LE.le { ofColex := t } { ofColex …
  -/
  simp_rw [le_iff_eq_or_lt, lt_iff_exists_filter_lt, mem_sdiff, filter_inj, and_assoc]
  /-
    case h.mpr
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    ⊢ And (Eq (s.erase (s.min' hs)).card t.card) (Or (Eq { ofColex := t } { ofCole …
  -/
  simp only [toColex_inj, ofColex_toColex, ne_eq, and_imp]
  /-
    case h.mpr
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    ⊢ Eq (s.erase (s.min' hs)).card t.card → Or (Eq t (s.erase (s.min' hs))) (Exis …
  -/
  rintro cards' (rfl | ⟨k, hks, hkt, z⟩)
  -- If t = s - min s, then use j = min s so t ∪ j = s
    /-
      case h.mpr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      cards' : Eq (s.erase (s.min' hs)).card (s.erase (s.min' hs)).card
      ⊢ Exists fun a => And (Not (Membership.mem (s.erase (s.min' hs)) a)) (And (Eq  …
    -/
  · refine ⟨min' s hs, not_mem_erase _ _, ?_⟩
    /-
      case h.mpr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      cards' : Eq (s.erase (s.min' hs)).card (s.erase (s.min' hs)).card
      ⊢ And (Eq s.card (Insert.insert (s.min' hs) (s.erase (s.min' hs))).card) (Or ( …
    -/
    rw [insert_erase (min'_mem _ _)]
    /-
      case h.mpr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      cards' : Eq (s.erase (s.min' hs)).card (s.erase (s.min' hs)).card
      ⊢ And (Eq s.card s.card) (Or (Eq s s) (Exists fun w => And (Membership.mem s w …
    -/
    exact ⟨rfl, Or.inl rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr.inr.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    ⊢ Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert.inser …
  -/
  set j := min' tᶜ ⟨k, mem_compl.2 hkt⟩
  -- Assume first t < s - min s, and take k as the colex witness for this
  /-
    case h.mpr.inr.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    ⊢ Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert.inser …
  -/
  have hjk : j ≤ k := min'_le _ _ (mem_compl.2 ‹k ∉ t›)
  /-
    case h.mpr.inr.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    hjk : LE.le j k
    ⊢ Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert.inser …
  -/
  have : j ∉ t := mem_compl.1 (min'_mem _ _)
  have hcard : #s = #(insert j t) := by
    rw [card_insert_of_not_mem ‹j ∉ t›, ← ‹_ = #t›, card_erase_add_one (min'_mem _ _)]
  /-
    case h.mpr.inr.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    hjk : LE.le j k
    this : Not (Membership.mem t j)
    hcard : Eq s.card (Insert.insert j t).card
    ⊢ Exists fun a => And (Not (Membership.mem t a)) (And (Eq s.card (Insert.inser …
  -/
  refine ⟨j, ‹_›, hcard, ?_⟩
  -- Cases on j < k or j = k
  /-
    case h.mpr.inr.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    hjk : LE.le j k
    this : Not (Membership.mem t j)
    hcard : Eq s.card (Insert.insert j t).card
    ⊢ Or (Eq (Insert.insert j t) s) (Exists fun w => And (Membership.mem s w) (And …
  -/
  obtain hjk | r₁ := hjk.lt_or_eq
  -- if j < k, k is our colex witness for t ∪ {j} < s
  · refine Or.inr ⟨k, mem_of_mem_erase ‹_›, fun hk ↦ hkt <| mem_of_mem_insert_of_ne hk hjk.ne',
      fun x hx ↦ ?_⟩
    simpa only [mem_insert, z hx, (hjk.trans hx).ne', mem_erase, Ne, false_or,
      and_iff_right_iff_imp] using fun _ ↦ ((min'_le _ _ <| mem_of_mem_erase hks).trans_lt hx).ne'
  -- if j = k, all of range k is in t so by sizes t ∪ {j} = s
  /-
    case h.mpr.inr.intro.intro.intro.inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    hjk : LE.le j k
    this : Not (Membership.mem t j)
    hcard : Eq s.card (Insert.insert j t).card
    r₁ : Eq j k
    ⊢ Or (Eq (Insert.insert j t) s) (Exists fun w => And (Membership.mem s w) (And …
  -/
  refine Or.inl (eq_of_subset_of_card_le (fun a ha ↦ ?_) hcard.ge).symm
  /-
    case h.mpr.inr.intro.intro.intro.inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    hs : s.Nonempty
    t : Finset α
    cards' : Eq (s.erase (s.min' hs)).card t.card
    k : α
    hks : Membership.mem (s.erase (s.min' hs)) k
    hkt : Not (Membership.mem t k)
    z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
    j : α := (HasCompl.compl t).min' ⋯
    hjk : LE.le j k
    this : Not (Membership.mem t j)
    hcard : Eq s.card (Insert.insert j t).card
    r₁ : Eq j k
    a : α
    ha : Membership.mem s a
    ⊢ Membership.mem (Insert.insert j t) a
  -/
  rcases lt_trichotomy k a with (lt | rfl | gt)
    /-
      case h.mpr.inr.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      lt : LT.lt k a
      ⊢ Membership.mem (Insert.insert j t) a
    -/
  · apply mem_insert_of_mem
    /-
      case h.mpr.inr.intro.intro.intro.inr.inl.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      lt : LT.lt k a
      ⊢ Membership.mem t a
    -/
    rw [z lt]
    /-
      case h.mpr.inr.intro.intro.intro.inr.inl.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      lt : LT.lt k a
      ⊢ Membership.mem (s.erase (s.min' hs)) a
    -/
    refine mem_erase_of_ne_of_mem (lt_of_le_of_lt ?_ lt).ne' ha
    /-
      case h.mpr.inr.intro.intro.intro.inr.inl.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      lt : LT.lt k a
      ⊢ LE.le (s.min' hs) k
    -/
    apply min'_le _ _ (mem_of_mem_erase ‹_›)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.inr.intro.intro.intro.inr.inr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      ha : Membership.mem s k
      ⊢ Membership.mem (Insert.insert j t) k
    -/
  · rw [r₁]; apply mem_insert_self
             /-
               🎉 no goals
             -/
    /-
      case h.mpr.inr.intro.intro.intro.inr.inr.inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      gt : LT.lt a k
      ⊢ Membership.mem (Insert.insert j t) a
    -/
  · apply mem_insert_of_mem
    /-
      case h.mpr.inr.intro.intro.intro.inr.inr.inr.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      gt : LT.lt a k
      ⊢ Membership.mem t a
    -/
    rw [← r₁] at gt
    /-
      case h.mpr.inr.intro.intro.intro.inr.inr.inr.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      gt : LT.lt a j
      ⊢ Membership.mem t a
    -/
    by_contra
    /-
      case h.mpr.inr.intro.intro.intro.inr.inr.inr.h
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      gt : LT.lt a j
      x✝ : Not (Membership.mem t a)
      ⊢ False
    -/
    apply (min'_le tᶜ _ _).not_lt gt
    /-
      α : Type u_1
      inst✝¹ : LinearOrder α
      s : Finset α
      inst✝ : Fintype α
      hs : s.Nonempty
      t : Finset α
      cards' : Eq (s.erase (s.min' hs)).card t.card
      k : α
      hks : Membership.mem (s.erase (s.min' hs)) k
      hkt : Not (Membership.mem t k)
      z : ∀ ⦃a : α⦄, LT.lt k a → Iff (Membership.mem t a) (Membership.mem (s.erase ( …
      j : α := (HasCompl.compl t).min' ⋯
      hjk : LE.le j k
      this : Not (Membership.mem t j)
      hcard : Eq s.card (Insert.insert j t).card
      r₁ : Eq j k
      a : α
      ha : Membership.mem s a
      gt : LT.lt a j
      x✝ : Not (Membership.mem t a)
      ⊢ Membership.mem (HasCompl.compl t) a
    -/
    rwa [mem_compl]
    /-
      🎉 no goals
    -/


/-- The shadow of an initial segment is also an initial segment. -/
protected lemma IsInitSeg.shadow [Finite α] (h₁ : IsInitSeg 𝒜 r) : IsInitSeg (∂ 𝒜) (r - 1) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Finite α
    h₁ : Finset.Colex.IsInitSeg 𝒜 r
    ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub r 1)
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Finite α
    h₁ : Finset.Colex.IsInitSeg 𝒜 r
    val✝ : Fintype α
    ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub r 1)
  -/
  obtain rfl | hr := Nat.eq_zero_or_pos r
    /-
      case intro.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      inst✝ : Finite α
      val✝ : Fintype α
      h₁ : Finset.Colex.IsInitSeg 𝒜 0
      ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub 0 1)
    -/
  · have : 𝒜 ⊆ {∅} := fun s hs ↦ by rw [mem_singleton, ← Finset.card_eq_zero]; exact h₁.1 hs
    /-
      case intro.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      inst✝ : Finite α
      val✝ : Fintype α
      h₁ : Finset.Colex.IsInitSeg 𝒜 0
      this : HasSubset.Subset 𝒜 (Singleton.singleton EmptyCollection.emptyCollection)
      ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub 0 1)
    -/
    have := shadow_monotone this
    /-
      case intro.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      inst✝ : Finite α
      val✝ : Fintype α
      h₁ : Finset.Colex.IsInitSeg 𝒜 0
      this✝ : HasSubset.Subset 𝒜 (Singleton.singleton EmptyCollection.emptyCollection)
      this : LE.le 𝒜.shadow (Singleton.singleton EmptyCollection.emptyCollection).sh …
      ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub 0 1)
    -/
    simp only [subset_empty, le_eq_subset, shadow_singleton_empty] at this
    /-
      case intro.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      inst✝ : Finite α
      val✝ : Fintype α
      h₁ : Finset.Colex.IsInitSeg 𝒜 0
      this✝ : HasSubset.Subset 𝒜 (Singleton.singleton EmptyCollection.emptyCollection)
      this : Eq 𝒜.shadow EmptyCollection.emptyCollection
      ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub 0 1)
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Finite α
    h₁ : Finset.Colex.IsInitSeg 𝒜 r
    val✝ : Fintype α
    hr : GT.gt r 0
    ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub r 1)
  -/
  obtain rfl | h𝒜 := 𝒜.eq_empty_or_nonempty
    /-
      case intro.inr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      r : Nat
      inst✝ : Finite α
      val✝ : Fintype α
      hr : GT.gt r 0
      h₁ : Finset.Colex.IsInitSeg EmptyCollection.emptyCollection r
      ⊢ Finset.Colex.IsInitSeg EmptyCollection.emptyCollection.shadow (HSub.hSub r 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case intro.inr.inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Finite α
    h₁ : Finset.Colex.IsInitSeg 𝒜 r
    val✝ : Fintype α
    hr : GT.gt r 0
    h𝒜 : 𝒜.Nonempty
    ⊢ Finset.Colex.IsInitSeg 𝒜.shadow (HSub.hSub r 1)
  -/
  obtain ⟨s, rfl, rfl⟩ := h₁.exists_initSeg h𝒜
  /-
    case intro.inr.inr.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Finite α
    val✝ : Fintype α
    s : Finset α
    hr : GT.gt s.card 0
    h𝒜 : (Finset.Colex.initSeg s).Nonempty
    h₁ : Finset.Colex.IsInitSeg (Finset.Colex.initSeg s) s.card
    ⊢ Finset.Colex.IsInitSeg (Finset.Colex.initSeg s).shadow (HSub.hSub s.card 1)
  -/
  rw [shadow_initSeg (card_pos.1 hr), ← card_erase_of_mem (min'_mem _ _)]
  /-
    case intro.inr.inr.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Finite α
    val✝ : Fintype α
    s : Finset α
    hr : GT.gt s.card 0
    h𝒜 : (Finset.Colex.initSeg s).Nonempty
    h₁ : Finset.Colex.IsInitSeg (Finset.Colex.initSeg s) s.card
    ⊢ Finset.Colex.IsInitSeg (Finset.Colex.initSeg (s.erase (s.min' ⋯))) (s.erase  …
  -/
  exact isInitSeg_initSeg
  /-
    🎉 no goals
  -/


/-- Applying the compression makes the set smaller in colex. This is intuitive since a portion of
the set is being "shifted down" as `max U < max V`. -/
lemma toColex_compress_lt_toColex {hU : U.Nonempty} {hV : V.Nonempty} (h : max' U hU < max' V hV)
    (hA : compress U V s ≠ s) : toColex (compress U V s) < toColex s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : Ne (UV.compress U V s) s
    ⊢ LT.lt { ofColex := UV.compress U V s } { ofColex := s }
  -/
  rw [compress, ite_ne_right_iff] at hA
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    ⊢ LT.lt { ofColex := UV.compress U V s } { ofColex := s }
  -/
  rw [compress, if_pos hA.1, lt_iff_exists_filter_lt]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    ⊢ Exists fun w => And (Membership.mem (SDiff.sdiff s (SDiff.sdiff (Max.max s U …
  -/
  simp_rw [mem_sdiff (s := s), filter_inj, and_assoc]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    ⊢ Exists fun w => And (Membership.mem s w) (And (Not (Membership.mem (SDiff.sd …
  -/
  refine ⟨_, hA.1.2 <| max'_mem _ hV, not_mem_sdiff_of_mem_right <| max'_mem _ _, fun a ha ↦ ?_⟩
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    a : α
    ha : LT.lt (V.max' hV) a
    ⊢ Iff (Membership.mem (SDiff.sdiff (Max.max s U) V) a) (Membership.mem s a)
  -/
  have : a ∉ V := fun H ↦ ha.not_le (le_max' _ _ H)
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    a : α
    ha : LT.lt (V.max' hV) a
    this : Not (Membership.mem V a)
    ⊢ Iff (Membership.mem (SDiff.sdiff (Max.max s U) V) a) (Membership.mem s a)
  -/
  have : a ∉ U := fun H ↦ ha.not_lt ((le_max' _ _ H).trans_lt h)
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s U V : Finset α
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    hA : And (And (Disjoint U s) (LE.le V s)) (Ne (SDiff.sdiff (Max.max s U) V) s)
    a : α
    ha : LT.lt (V.max' hV) a
    this✝ : Not (Membership.mem V a)
    this : Not (Membership.mem U a)
    ⊢ Iff (Membership.mem (SDiff.sdiff (Max.max s U) V) a) (Membership.mem s a)
  -/
  simp [‹a ∉ U›, ‹a ∉ V›]
  /-
    🎉 no goals
  -/


/-- These are the compressions which we will apply to decrease the "measure" of a family of sets.-/
private def UsefulCompression (U V : Finset α) : Prop :=
  Disjoint U V ∧ #U = #V ∧ ∃ (HU : U.Nonempty) (HV : V.Nonempty), max' U HU < max' V HV


private instance UsefulCompression.instDecidableRel :
    DecidableRel (α := Finset α) UsefulCompression :=
  fun _ _ ↦ inferInstanceAs (Decidable (_ ∧ _))


/-- Applying a good compression will decrease measure, keep cardinality, keep sizes and decrease
shadow. In particular, 'good' means it's useful, and every smaller compression won't make a
difference. -/
private lemma compression_improved (𝒜 : Finset (Finset α)) (h₁ : UsefulCompression U V)
    (h₂ : ∀ ⦃U₁ V₁⦄, UsefulCompression U₁ V₁ → #U₁ < #U → IsCompressed U₁ V₁ 𝒜) :
    #(∂ (𝓒 U V 𝒜)) ≤ #(∂ 𝒜) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    U V : Finset α
    𝒜 : Finset (Finset α)
    h₁ : Finset.UV.UsefulCompression U V
    h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
    ⊢ LE.le (UV.compression U V 𝒜).shadow.card 𝒜.shadow.card
  -/
  obtain ⟨UVd, same_size, hU, hV, max_lt⟩ := h₁
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    U V : Finset α
    𝒜 : Finset (Finset α)
    h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
    UVd : Disjoint U V
    same_size : Eq U.card V.card
    hU : U.Nonempty
    hV : V.Nonempty
    max_lt : LT.lt (U.max' hU) (V.max' hV)
    ⊢ LE.le (UV.compression U V 𝒜).shadow.card 𝒜.shadow.card
  -/
  refine card_shadow_compression_le _ _ fun x Hx ↦ ⟨min' V hV, min'_mem _ _, ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    U V : Finset α
    𝒜 : Finset (Finset α)
    h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
    UVd : Disjoint U V
    same_size : Eq U.card V.card
    hU : U.Nonempty
    hV : V.Nonempty
    max_lt : LT.lt (U.max' hU) (V.max' hV)
    x : α
    Hx : Membership.mem U x
    ⊢ UV.IsCompressed (U.erase x) (V.erase (V.min' hV)) 𝒜
  -/
  obtain hU' | hU' := eq_or_lt_of_le (succ_le_iff.2 hU.card_pos)
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq U.card V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : Eq (Nat.succ 0) U.card
      ⊢ UV.IsCompressed (U.erase x) (V.erase (V.min' hV)) 𝒜
    -/
  · rw [← hU'] at same_size
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq (Nat.succ 0) V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : Eq (Nat.succ 0) U.card
      ⊢ UV.IsCompressed (U.erase x) (V.erase (V.min' hV)) 𝒜
    -/
    have : erase U x = ∅ := by rw [← Finset.card_eq_zero, card_erase_of_mem Hx, ← hU']
    have : erase V (min' V hV) = ∅ := by
      rw [← Finset.card_eq_zero, card_erase_of_mem (min'_mem _ _), ← same_size]
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq (Nat.succ 0) V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : Eq (Nat.succ 0) U.card
      this✝ : Eq (U.erase x) EmptyCollection.emptyCollection
      this : Eq (V.erase (V.min' hV)) EmptyCollection.emptyCollection
      ⊢ UV.IsCompressed (U.erase x) (V.erase (V.min' hV)) 𝒜
    -/
    rw [‹erase U x = ∅›, ‹erase V (min' V hV) = ∅›]
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq (Nat.succ 0) V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : Eq (Nat.succ 0) U.card
      this✝ : Eq (U.erase x) EmptyCollection.emptyCollection
      this : Eq (V.erase (V.min' hV)) EmptyCollection.emptyCollection
      ⊢ UV.IsCompressed EmptyCollection.emptyCollection EmptyCollection.emptyCollect …
    -/
    exact isCompressed_self _ _
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.inr
    α : Type u_1
    inst✝ : LinearOrder α
    U V : Finset α
    𝒜 : Finset (Finset α)
    h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
    UVd : Disjoint U V
    same_size : Eq U.card V.card
    hU : U.Nonempty
    hV : V.Nonempty
    max_lt : LT.lt (U.max' hU) (V.max' hV)
    x : α
    Hx : Membership.mem U x
    hU' : LT.lt (Nat.succ 0) U.card
    ⊢ UV.IsCompressed (U.erase x) (V.erase (V.min' hV)) 𝒜
  -/
  refine h₂ ⟨UVd.mono (erase_subset ..) (erase_subset ..), ?_, ?_, ?_, ?_⟩ (card_erase_lt_of_mem Hx)
    /-
      case intro.intro.intro.intro.inr.refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq U.card V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : LT.lt (Nat.succ 0) U.card
      ⊢ Eq (U.erase x).card (V.erase (V.min' hV)).card
    -/
  · rw [card_erase_of_mem (min'_mem _ _), card_erase_of_mem Hx, same_size]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq U.card V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : LT.lt (Nat.succ 0) U.card
      ⊢ (U.erase x).Nonempty
    -/
  · rwa [← card_pos, card_erase_of_mem Hx, tsub_pos_iff_lt]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.refine_3
      α : Type u_1
      inst✝ : LinearOrder α
      U V : Finset α
      𝒜 : Finset (Finset α)
      h₂ : ∀ ⦃U₁ V₁ : Finset α⦄, Finset.UV.UsefulCompression U₁ V₁ → LT.lt U₁.card U …
      UVd : Disjoint U V
      same_size : Eq U.card V.card
      hU : U.Nonempty
      hV : V.Nonempty
      max_lt : LT.lt (U.max' hU) (V.max' hV)
      x : α
      Hx : Membership.mem U x
      hU' : LT.lt (Nat.succ 0) U.card
      ⊢ (V.erase (V.min' hV)).Nonempty
    -/
  · rwa [← Finset.card_pos, card_erase_of_mem (min'_mem _ _), ← same_size, tsub_pos_iff_lt]
    /-
      🎉 no goals
    -/
  · exact (Finset.max'_subset _ <| erase_subset _ _).trans_lt (max_lt.trans_le <| le_max' _ _ <|
      mem_erase.2 ⟨(min'_lt_max'_of_card _ (by rwa [← same_size])).ne', max'_mem _ _⟩)


/-- If we're compressed by all useful compressions, then we're an initial segment. This is the other
key Kruskal-Katona part. -/
lemma isInitSeg_of_compressed {ℬ : Finset (Finset α)} {r : ℕ} (h₁ : (ℬ : Set (Finset α)).Sized r)
    (h₂ : ∀ U V, UsefulCompression U V → IsCompressed U V ℬ) : IsInitSeg ℬ r := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    ⊢ Finset.Colex.IsInitSeg ℬ r
  -/
  refine ⟨h₁, ?_⟩
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    ⊢ ∀ ⦃s t : Finset α⦄, Membership.mem ℬ s → And (LT.lt { ofColex := t } { ofCol …
  -/
  rintro A B hA ⟨hBA, sizeA⟩
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    ⊢ Membership.mem ℬ B
  -/
  by_contra hB
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    ⊢ False
  -/
  have hAB : A ≠ B := ne_of_mem_of_not_mem hA hB
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    ⊢ False
  -/
  have hAB' : #A = #B := (h₁ hA).trans sizeA.symm
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    hAB' : Eq A.card B.card
    ⊢ False
  -/
  have hU : (A \ B).Nonempty := sdiff_nonempty.2 fun h ↦ hAB <| eq_of_subset_of_card_le h hAB'.ge
  have hV : (B \ A).Nonempty :=
    sdiff_nonempty.2 fun h ↦ hAB.symm <| eq_of_subset_of_card_le h hAB'.le
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    hAB' : Eq A.card B.card
    hU : (SDiff.sdiff A B).Nonempty
    hV : (SDiff.sdiff B A).Nonempty
    ⊢ False
  -/
  have disj : Disjoint (B \ A) (A \ B) := disjoint_sdiff.mono_left sdiff_subset
  have smaller : max' _ hV < max' _ hU := by
    obtain hlt | heq | hgt := lt_trichotomy (max' _ hU) (max' _ hV)
    · rw [← compress_sdiff_sdiff A B] at hAB hBA
      cases hBA.not_lt <| toColex_compress_lt_toColex hlt hAB
    · exact (disjoint_right.1 disj (max'_mem _ hU) <| heq.symm ▸ max'_mem _ _).elim
    · assumption
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    hAB' : Eq A.card B.card
    hU : (SDiff.sdiff A B).Nonempty
    hV : (SDiff.sdiff B A).Nonempty
    disj : Disjoint (SDiff.sdiff B A) (SDiff.sdiff A B)
    smaller : LT.lt ((SDiff.sdiff B A).max' hV) ((SDiff.sdiff A B).max' hU)
    ⊢ False
  -/
  refine hB ?_
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    hAB' : Eq A.card B.card
    hU : (SDiff.sdiff A B).Nonempty
    hV : (SDiff.sdiff B A).Nonempty
    disj : Disjoint (SDiff.sdiff B A) (SDiff.sdiff A B)
    smaller : LT.lt ((SDiff.sdiff B A).max' hV) ((SDiff.sdiff A B).max' hU)
    ⊢ Membership.mem ℬ B
  -/
  rw [← (h₂ _ _ ⟨disj, card_sdiff_comm hAB'.symm, hV, hU, smaller⟩).eq]
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    ℬ : Finset (Finset α)
    r : Nat
    h₁ : Set.Sized r ↑ℬ
    h₂ : ∀ (U V : Finset α), Finset.UV.UsefulCompression U V → UV.IsCompressed U V ℬ
    A B : Finset α
    hA : Membership.mem ℬ A
    hBA : LT.lt { ofColex := B } { ofColex := A }
    sizeA : Eq B.card r
    hB : Not (Membership.mem ℬ B)
    hAB : Ne A B
    hAB' : Eq A.card B.card
    hU : (SDiff.sdiff A B).Nonempty
    hV : (SDiff.sdiff B A).Nonempty
    disj : Disjoint (SDiff.sdiff B A) (SDiff.sdiff A B)
    smaller : LT.lt ((SDiff.sdiff B A).max' hV) ((SDiff.sdiff A B).max' hU)
    ⊢ Membership.mem (UV.compression (SDiff.sdiff B A) (SDiff.sdiff A B) ℬ) B
  -/
  exact mem_compression.2 (Or.inr ⟨hB, A, hA, compress_sdiff_sdiff _ _⟩)
  /-
    🎉 no goals
  -/


/-- This measures roughly how compressed the family is.

Note that this does depend on the order of the ground set, unlike the Kruskal-Katona theorem itself
(although `kruskal_katona` currently is stated in an order-dependent manner). -/
private def familyMeasure (𝒜 : Finset (Finset (Fin n))) : ℕ := ∑ A in 𝒜, ∑ a in A, 2 ^ (a : ℕ)


/-- Applying a compression strictly decreases the measure. This helps show that "compress until we
can't any more" is a terminating process. -/
private lemma familyMeasure_compression_lt_familyMeasure {U V : Finset (Fin n)} {hU : U.Nonempty}
    {hV : V.Nonempty} (h : max' U hU < max' V hV) {𝒜 : Finset (Finset (Fin n))} (a : 𝓒 U V 𝒜 ≠ 𝒜) :
    familyMeasure (𝓒 U V 𝒜) < familyMeasure 𝒜 := by
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (UV.compression U V 𝒜) 𝒜
    ⊢ LT.lt (Finset.UV.familyMeasure (UV.compression U V 𝒜)) (Finset.UV.familyMeas …
  -/
  rw [compression] at a ⊢
  have q : ∀ Q ∈ {A ∈ 𝒜 | compress U V A ∉ 𝒜}, compress U V Q ≠ Q := by
    simp_rw [mem_filter]
    intro Q hQ h
    rw [h] at hQ
    exact hQ.2 hQ.1
  have uA : {A ∈ 𝒜 | compress U V A ∈ 𝒜} ∪ {A ∈ 𝒜 | compress U V A ∉ 𝒜} = 𝒜 :=
    filter_union_filter_neg_eq _ _
  have ne₂ : {A ∈ 𝒜 | compress U V A ∉ 𝒜}.Nonempty := by
    refine nonempty_iff_ne_empty.2 fun z ↦ a ?_
    rw [filter_image, z, image_empty, union_empty]
    rwa [z, union_empty] at uA
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (Union.union (Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
    q : ∀ (Q : Finset (Fin n)), Membership.mem (Finset.filter (fun A => Not (Membe …
    uA : Eq (Union.union (Finset.filter (fun A => Membership.mem 𝒜 (UV.compress U  …
    ne₂ : (Finset.filter (fun A => Not (Membership.mem 𝒜 (UV.compress U V A))) 𝒜). …
    ⊢ LT.lt (Finset.UV.familyMeasure (Union.union (Finset.filter (fun a => Members …
  -/
  rw [familyMeasure, familyMeasure, sum_union compress_disjoint]
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (Union.union (Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
    q : ∀ (Q : Finset (Fin n)), Membership.mem (Finset.filter (fun A => Not (Membe …
    uA : Eq (Union.union (Finset.filter (fun A => Membership.mem 𝒜 (UV.compress U  …
    ne₂ : (Finset.filter (fun A => Not (Membership.mem 𝒜 (UV.compress U V A))) 𝒜). …
    ⊢ LT.lt (HAdd.hAdd ((Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
  -/
  conv_rhs => rw [← uA]
  rw [sum_union (disjoint_filter_filter_neg _ _ _), add_lt_add_iff_left, filter_image,
    sum_image compress_injOn]
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (Union.union (Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
    q : ∀ (Q : Finset (Fin n)), Membership.mem (Finset.filter (fun A => Not (Membe …
    uA : Eq (Union.union (Finset.filter (fun A => Membership.mem 𝒜 (UV.compress U  …
    ne₂ : (Finset.filter (fun A => Not (Membership.mem 𝒜 (UV.compress U V A))) 𝒜). …
    ⊢ LT.lt ((Finset.filter (fun a => Not (Membership.mem 𝒜 (UV.compress U V a)))  …
  -/
  refine sum_lt_sum_of_nonempty ne₂ fun A hA ↦ ?_
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (Union.union (Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
    q : ∀ (Q : Finset (Fin n)), Membership.mem (Finset.filter (fun A => Not (Membe …
    uA : Eq (Union.union (Finset.filter (fun A => Membership.mem 𝒜 (UV.compress U  …
    ne₂ : (Finset.filter (fun A => Not (Membership.mem 𝒜 (UV.compress U V A))) 𝒜). …
    A : Finset (Fin n)
    hA : Membership.mem (Finset.filter (fun a => Not (Membership.mem 𝒜 (UV.compres …
    ⊢ LT.lt ((UV.compress U V A).sum fun a => HPow.hPow 2 ↑a) (A.sum fun a => HPow …
  -/
  simp_rw [← sum_image Fin.val_injective.injOn]
  rw [geomSum_lt_geomSum_iff_toColex_lt_toColex le_rfl,
    toColex_image_lt_toColex_image Fin.val_strictMono]
  /-
    n : Nat
    U V : Finset (Fin n)
    hU : U.Nonempty
    hV : V.Nonempty
    h : LT.lt (U.max' hU) (V.max' hV)
    𝒜 : Finset (Finset (Fin n))
    a : Ne (Union.union (Finset.filter (fun a => Membership.mem 𝒜 (UV.compress U V …
    q : ∀ (Q : Finset (Fin n)), Membership.mem (Finset.filter (fun A => Not (Membe …
    uA : Eq (Union.union (Finset.filter (fun A => Membership.mem 𝒜 (UV.compress U  …
    ne₂ : (Finset.filter (fun A => Not (Membership.mem 𝒜 (UV.compress U V A))) 𝒜). …
    A : Finset (Fin n)
    hA : Membership.mem (Finset.filter (fun a => Not (Membership.mem 𝒜 (UV.compres …
    ⊢ LT.lt { ofColex := UV.compress U V A } { ofColex := A }
  -/
  exact toColex_compress_lt_toColex h <| q _ hA
  /-
    🎉 no goals
  -/


/-- The main Kruskal-Katona helper: use induction with our measure to keep compressing until
we can't any more, which gives a set family which is fully compressed and has the nice properties we
want. -/
private lemma kruskal_katona_helper {r : ℕ} (𝒜 : Finset (Finset (Fin n)))
    (h : (𝒜 : Set (Finset (Fin n))).Sized r) :
    ∃ ℬ : Finset (Finset (Fin n)), #(∂ ℬ) ≤ #(∂ 𝒜) ∧ #𝒜 = #ℬ ∧
      (ℬ : Set (Finset (Fin n))).Sized r ∧ ∀ U V, UsefulCompression U V → IsCompressed U V ℬ := by
  classical
  -- Are there any compressions we can make now?
  set usable : Finset (Finset (Fin n) × Finset (Fin n)) :=
    {t | UsefulCompression t.1 t.2 ∧ ¬ IsCompressed t.1 t.2 𝒜}
  obtain husable | husable := usable.eq_empty_or_nonempty
  -- No. Then where we are is the required set family.
  · refine ⟨𝒜, le_rfl, rfl, h, fun U V hUV ↦ ?_⟩
    rw [eq_empty_iff_forall_not_mem] at husable
    by_contra h
    exact husable ⟨U, V⟩ <| mem_filter.2 ⟨mem_univ _, hUV, h⟩
  -- Yes. Then apply the smallest compression, then keep going
  obtain ⟨⟨U, V⟩, hUV, t⟩ := exists_min_image usable (fun t ↦ #t.1) husable
  rw [mem_filter] at hUV
  have h₂ : ∀ U₁ V₁, UsefulCompression U₁ V₁ → #U₁ < #U → IsCompressed U₁ V₁ 𝒜 := by
    rintro U₁ V₁ huseful hUcard
    by_contra h
    exact hUcard.not_le <| t ⟨U₁, V₁⟩ <| mem_filter.2 ⟨mem_univ _, huseful, h⟩
  have p1 : #(∂ (𝓒 U V 𝒜)) ≤ #(∂ 𝒜) := compression_improved _ hUV.2.1 h₂
  obtain ⟨-, hUV', hu, hv, hmax⟩ := hUV.2.1
  have := familyMeasure_compression_lt_familyMeasure hmax hUV.2.2
  obtain ⟨t, q1, q2, q3, q4⟩ := UV.kruskal_katona_helper (𝓒 U V 𝒜) (h.uvCompression hUV')
  exact ⟨t, q1.trans p1, (card_compression _ _ _).symm.trans q2, q3, q4⟩
termination_by familyMeasure 𝒜


/-- The **Kruskal-Katona theorem**.

Given a set family `𝒜` consisting of `r`-sets, and `𝒞` an initial segment of the colex order of the
same size, the shadow of `𝒞` is smaller than the shadow of `𝒜`. In particular, this gives that the
minimum shadow size is achieved by initial segments of colex. -/
theorem kruskal_katona (h𝒜r : (𝒜 : Set (Finset (Fin n))).Sized r) (h𝒞𝒜 : #𝒞 ≤ #𝒜)
    (h𝒞 : IsInitSeg 𝒞 r) : #(∂ 𝒞) ≤ #(∂ 𝒜) := by
  -- WLOG `|𝒜| = |𝒞|`
  /-
    n r : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h𝒜r : Set.Sized r ↑𝒜
    h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
    h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
    ⊢ LE.le 𝒞.shadow.card 𝒜.shadow.card
  -/
  obtain ⟨𝒜', h𝒜, h𝒜𝒞⟩ := exists_subset_card_eq h𝒞𝒜
  -- By `kruskal_katona_helper`, we find a fully compressed family `ℬ` of the same size as `𝒜`
  -- whose shadow is no bigger.
  /-
    case intro.intro
    n r : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h𝒜r : Set.Sized r ↑𝒜
    h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
    h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
    𝒜' : Finset (Finset (Fin n))
    h𝒜 : HasSubset.Subset 𝒜' 𝒜
    h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
    ⊢ LE.le 𝒞.shadow.card 𝒜.shadow.card
  -/
  obtain ⟨ℬ, hℬ𝒜, h𝒜ℬ, hℬr, hℬ⟩ := UV.kruskal_katona_helper 𝒜' (h𝒜r.mono (by gcongr))
  -- This means that `ℬ` is an initial segment of the same size as `𝒞`. Hence they are equal and
  -- we are done.
  /-
    case intro.intro.intro.intro.intro.intro
    n r : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h𝒜r : Set.Sized r ↑𝒜
    h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
    h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
    𝒜' : Finset (Finset (Fin n))
    h𝒜 : HasSubset.Subset 𝒜' 𝒜
    h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
    ℬ : Finset (Finset (Fin n))
    hℬ𝒜 : LE.le ℬ.shadow.card 𝒜'.shadow.card
    h𝒜ℬ : Eq 𝒜'.card ℬ.card
    hℬr : Set.Sized r ↑ℬ
    hℬ : ∀ (U V : Finset (Fin n)), Finset.UV.UsefulCompression U V → UV.IsCompress …
    ⊢ LE.le 𝒞.shadow.card 𝒜.shadow.card
  -/
  suffices ℬ = 𝒞 by subst 𝒞; exact hℬ𝒜.trans (by gcongr)
  /-
    case intro.intro.intro.intro.intro.intro
    n r : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h𝒜r : Set.Sized r ↑𝒜
    h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
    h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
    𝒜' : Finset (Finset (Fin n))
    h𝒜 : HasSubset.Subset 𝒜' 𝒜
    h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
    ℬ : Finset (Finset (Fin n))
    hℬ𝒜 : LE.le ℬ.shadow.card 𝒜'.shadow.card
    h𝒜ℬ : Eq 𝒜'.card ℬ.card
    hℬr : Set.Sized r ↑ℬ
    hℬ : ∀ (U V : Finset (Fin n)), Finset.UV.UsefulCompression U V → UV.IsCompress …
    ⊢ Eq ℬ 𝒞
  -/
  have hcard : #ℬ = #𝒞 := h𝒜ℬ.symm.trans h𝒜𝒞
  /-
    case intro.intro.intro.intro.intro.intro
    n r : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h𝒜r : Set.Sized r ↑𝒜
    h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
    h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
    𝒜' : Finset (Finset (Fin n))
    h𝒜 : HasSubset.Subset 𝒜' 𝒜
    h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
    ℬ : Finset (Finset (Fin n))
    hℬ𝒜 : LE.le ℬ.shadow.card 𝒜'.shadow.card
    h𝒜ℬ : Eq 𝒜'.card ℬ.card
    hℬr : Set.Sized r ↑ℬ
    hℬ : ∀ (U V : Finset (Fin n)), Finset.UV.UsefulCompression U V → UV.IsCompress …
    hcard : Eq ℬ.card 𝒞.card
    ⊢ Eq ℬ 𝒞
  -/
  obtain h𝒞ℬ | hℬ𝒞 := h𝒞.total (UV.isInitSeg_of_compressed hℬr hℬ)
    /-
      case intro.intro.intro.intro.intro.intro.inl
      n r : Nat
      𝒜 𝒞 : Finset (Finset (Fin n))
      h𝒜r : Set.Sized r ↑𝒜
      h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
      h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
      𝒜' : Finset (Finset (Fin n))
      h𝒜 : HasSubset.Subset 𝒜' 𝒜
      h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
      ℬ : Finset (Finset (Fin n))
      hℬ𝒜 : LE.le ℬ.shadow.card 𝒜'.shadow.card
      h𝒜ℬ : Eq 𝒜'.card ℬ.card
      hℬr : Set.Sized r ↑ℬ
      hℬ : ∀ (U V : Finset (Fin n)), Finset.UV.UsefulCompression U V → UV.IsCompress …
      hcard : Eq ℬ.card 𝒞.card
      h𝒞ℬ : HasSubset.Subset 𝒞 ℬ
      ⊢ Eq ℬ 𝒞
    -/
  · exact (eq_of_subset_of_card_le h𝒞ℬ hcard.le).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.inr
      n r : Nat
      𝒜 𝒞 : Finset (Finset (Fin n))
      h𝒜r : Set.Sized r ↑𝒜
      h𝒞𝒜 : LE.le 𝒞.card 𝒜.card
      h𝒞 : Finset.Colex.IsInitSeg 𝒞 r
      𝒜' : Finset (Finset (Fin n))
      h𝒜 : HasSubset.Subset 𝒜' 𝒜
      h𝒜𝒞 : Eq 𝒜'.card 𝒞.card
      ℬ : Finset (Finset (Fin n))
      hℬ𝒜 : LE.le ℬ.shadow.card 𝒜'.shadow.card
      h𝒜ℬ : Eq 𝒜'.card ℬ.card
      hℬr : Set.Sized r ↑ℬ
      hℬ : ∀ (U V : Finset (Fin n)), Finset.UV.UsefulCompression U V → UV.IsCompress …
      hcard : Eq ℬ.card 𝒞.card
      hℬ𝒞 : HasSubset.Subset ℬ 𝒞
      ⊢ Eq ℬ 𝒞
    -/
  · exact eq_of_subset_of_card_le hℬ𝒞 hcard.ge
    /-
      🎉 no goals
    -/


/-- An iterated form of the Kruskal-Katona theorem. In particular, the minimum possible iterated
shadow size is attained by initial segments. -/
theorem iterated_kk (h₁ : (𝒜 : Set (Finset (Fin n))).Sized r) (h₂ : #𝒞 ≤ #𝒜) (h₃ : IsInitSeg 𝒞 r) :
    #(∂^[k] 𝒞) ≤ #(∂^[k] 𝒜) := by
  /-
    n r k : Nat
    𝒜 𝒞 : Finset (Finset (Fin n))
    h₁ : Set.Sized r ↑𝒜
    h₂ : LE.le 𝒞.card 𝒜.card
    h₃ : Finset.Colex.IsInitSeg 𝒞 r
    ⊢ LE.le (Nat.iterate Finset.shadow k 𝒞).card (Nat.iterate Finset.shadow k 𝒜).c …
  -/
  induction' k with _k ih generalizing r 𝒜 𝒞
    /-
      case zero
      n r : Nat
      𝒜 𝒞 : Finset (Finset (Fin n))
      h₁ : Set.Sized r ↑𝒜
      h₂ : LE.le 𝒞.card 𝒜.card
      h₃ : Finset.Colex.IsInitSeg 𝒞 r
      ⊢ LE.le (Nat.iterate Finset.shadow 0 𝒞).card (Nat.iterate Finset.shadow 0 𝒜).c …
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case succ
      n _k : Nat
      ih : ∀ {r : Nat} {𝒜 𝒞 : Finset (Finset (Fin n))}, Set.Sized r ↑𝒜 → LE.le 𝒞.car …
      r : Nat
      𝒜 𝒞 : Finset (Finset (Fin n))
      h₁ : Set.Sized r ↑𝒜
      h₂ : LE.le 𝒞.card 𝒜.card
      h₃ : Finset.Colex.IsInitSeg 𝒞 r
      ⊢ LE.le (Nat.iterate Finset.shadow (HAdd.hAdd _k 1) 𝒞).card (Nat.iterate Finse …
    -/
  · refine ih h₁.shadow (kruskal_katona h₁ h₂ h₃) ?_
    /-
      case succ
      n _k : Nat
      ih : ∀ {r : Nat} {𝒜 𝒞 : Finset (Finset (Fin n))}, Set.Sized r ↑𝒜 → LE.le 𝒞.car …
      r : Nat
      𝒜 𝒞 : Finset (Finset (Fin n))
      h₁ : Set.Sized r ↑𝒜
      h₂ : LE.le 𝒞.card 𝒜.card
      h₃ : Finset.Colex.IsInitSeg 𝒞 r
      ⊢ Finset.Colex.IsInitSeg 𝒞.shadow (HSub.hSub r 1)
    -/
    convert h₃.shadow
    /-
      🎉 no goals
    -/


/-- The **Lovasz formulation of the Kruskal-Katona theorem**.

If `|𝒜| ≥ k choose r`, (and everything in `𝒜` has size `r`) then the initial segment we compare to
is just all the subsets of `{0, ..., k - 1}` of size `r`. The `i`-th iterated shadow of this is all
the subsets of `{0, ..., k - 1}` of size `r - i`, so the `i`-th iterated shadow of `𝒜` has at least
`k.choose (r - i)` elements. -/
theorem kruskal_katona_lovasz_form (hir : i ≤ r) (hrk : r ≤ k) (hkn : k ≤ n)
    (h₁ : (𝒜 : Set (Finset (Fin n))).Sized r) (h₂ : k.choose r ≤ #𝒜) :
    k.choose (r - i) ≤ #(∂^[i] 𝒜) := by
  set range'k : Finset (Fin n) :=
    attachFin (range k) fun m ↦ by rw [mem_range]; apply forall_lt_iff_le.2 hkn
  /-
    n r k i : Nat
    𝒜 : Finset (Finset (Fin n))
    hir : LE.le i r
    hrk : LE.le r k
    hkn : LE.le k n
    h₁ : Set.Sized r ↑𝒜
    h₂ : LE.le (k.choose r) 𝒜.card
    range'k : Finset (Fin n) := (Finset.range k).attachFin ⋯
    ⊢ LE.le (k.choose (HSub.hSub r i)) (Nat.iterate Finset.shadow i 𝒜).card
  -/
  set 𝒞 : Finset (Finset (Fin n)) := powersetCard r range'k
  /-
    n r k i : Nat
    𝒜 : Finset (Finset (Fin n))
    hir : LE.le i r
    hrk : LE.le r k
    hkn : LE.le k n
    h₁ : Set.Sized r ↑𝒜
    h₂ : LE.le (k.choose r) 𝒜.card
    range'k : Finset (Fin n) := (Finset.range k).attachFin ⋯
    𝒞 : Finset (Finset (Fin n)) := Finset.powersetCard r range'k
    ⊢ LE.le (k.choose (HSub.hSub r i)) (Nat.iterate Finset.shadow i 𝒜).card
  -/
  have : (𝒞 : Set (Finset (Fin n))).Sized r := Set.sized_powersetCard _ _
  calc
    k.choose (r - i)
    _ = #(powersetCard (r - i) range'k) := by rw [card_powersetCard, card_attachFin, card_range]
    _ = #(∂^[i] 𝒞) := by
      congr!
      ext B
      rw [mem_powersetCard, mem_shadow_iterate_iff_exists_sdiff]
      constructor
      · rintro ⟨hBk, hB⟩
        have := exists_subsuperset_card_eq hBk (Nat.le_add_left _ i) <| by
          rwa [hB, card_attachFin, card_range, ← Nat.add_sub_assoc hir, Nat.add_sub_cancel_left]
        obtain ⟨C, BsubC, hCrange, hcard⟩ := this
        rw [hB, ← Nat.add_sub_assoc hir, Nat.add_sub_cancel_left] at hcard
        refine ⟨C, mem_powersetCard.2 ⟨hCrange, hcard⟩, BsubC, ?_⟩
        rw [card_sdiff BsubC, hcard, hB, Nat.sub_sub_self hir]
      · rintro ⟨A, Ah, hBA, card_sdiff_i⟩
        rw [mem_powersetCard] at Ah
        refine ⟨hBA.trans Ah.1, eq_tsub_of_add_eq ?_⟩
        rw [← Ah.2, ← card_sdiff_i, add_comm, card_sdiff_add_card_eq_card hBA]
    _ ≤ #(∂ ^[i] 𝒜) := by
      refine iterated_kk h₁ ?_ ⟨‹_›, ?_⟩
      · rwa [card_powersetCard, card_attachFin, card_range]
      simp_rw [𝒞, mem_powersetCard]
      rintro A B hA ⟨HB₁, HB₂⟩
      refine ⟨fun t ht ↦ ?_, ‹_›⟩
      rw [mem_attachFin, mem_range]
      have : toColex (image Fin.val B) < toColex (image Fin.val A) := by
        rwa [toColex_image_lt_toColex_image Fin.val_strictMono]
      apply Colex.forall_lt_mono this.le _ t (mem_image.2 ⟨t, ht, rfl⟩)
      simp_rw [mem_image]
      rintro _ ⟨a, ha, hab⟩
      simpa [range'k, hab] using hA.1 ha


/-- The **Erdős–Ko–Rado theorem**.

The maximum size of an intersecting family in `α` where all sets have size `r` is bounded by
`(card α - 1).choose (r - 1)`. This bound is sharp. -/
theorem erdos_ko_rado {𝒜 : Finset (Finset (Fin n))} {r : ℕ}
    (h𝒜 : (𝒜 : Set (Finset (Fin n))).Intersecting) (h₂ : (𝒜 : Set (Finset (Fin n))).Sized r)
    (h₃ : r ≤ n / 2) :
    #𝒜 ≤ (n - 1).choose (r - 1) := by
  -- Take care of the r=0 case first: it's not very interesting.
  /-
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    ⊢ LE.le 𝒜.card ((HSub.hSub n 1).choose (HSub.hSub r 1))
  -/
  cases' Nat.eq_zero_or_pos r with b h1r
    /-
      case inl
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      b : Eq r 0
      ⊢ LE.le 𝒜.card ((HSub.hSub n 1).choose (HSub.hSub r 1))
    -/
  · convert Nat.zero_le _
    /-
      case h.e'_3
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      b : Eq r 0
      ⊢ Eq 𝒜.card 0
    -/
    rw [Finset.card_eq_zero, eq_empty_iff_forall_not_mem]
    /-
      case h.e'_3
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      b : Eq r 0
      ⊢ ∀ (x : Finset (Fin n)), Not (Membership.mem 𝒜 x)
    -/
    refine fun A HA ↦ h𝒜 HA HA ?_
    /-
      case h.e'_3
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      b : Eq r 0
      A : Finset (Fin n)
      HA : Membership.mem 𝒜 A
      ⊢ Disjoint A A
    -/
    rw [disjoint_self_iff_empty, ← Finset.card_eq_zero, ← b]
    /-
      case h.e'_3
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      b : Eq r 0
      A : Finset (Fin n)
      HA : Membership.mem 𝒜 A
      ⊢ Eq A.card r
    -/
    exact h₂ HA
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    ⊢ LE.le 𝒜.card ((HSub.hSub n 1).choose (HSub.hSub r 1))
  -/
  refine le_of_not_lt fun size ↦ ?_
  -- Consider 𝒜ᶜˢ = {sᶜ | s ∈ 𝒜}
  -- Its iterated shadow (∂^[n-2k] 𝒜ᶜˢ) is disjoint from 𝒜 by intersecting-ness
  have : Disjoint 𝒜 (∂^[n - 2 * r] 𝒜ᶜˢ) := disjoint_right.2 fun A hAbar hA ↦ by
    simp [mem_shadow_iterate_iff_exists_sdiff, mem_compls] at hAbar
    obtain ⟨C, hC, hAC, _⟩ := hAbar
    exact h𝒜 hA hC (disjoint_of_subset_left hAC disjoint_compl_right)
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜.c …
    ⊢ False
  -/
  have : r ≤ n := h₃.trans (Nat.div_le_self n 2)
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜. …
    this : LE.le r n
    ⊢ False
  -/
  have : 1 ≤ n := ‹1 ≤ r›.trans ‹r ≤ n›
  -- We know the size of 𝒜ᶜˢ since it's the same size as 𝒜
  have z : (n - 1).choose (n - r) < #𝒜ᶜˢ := by
    rwa [card_compls, choose_symm_of_eq_add (tsub_add_tsub_cancel ‹r ≤ n› ‹1 ≤ r›).symm]
  -- and everything in 𝒜ᶜˢ has size n-r.
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝¹ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝ : LE.le r n
    this : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    ⊢ False
  -/
  have h𝒜bar : (𝒜ᶜˢ : Set (Finset (Fin n))).Sized (n - r) := by simpa using h₂.compls
  have : n - 2 * r ≤ n - r := by
    rw [tsub_le_tsub_iff_left ‹r ≤ n›]
    exact Nat.le_mul_of_pos_left _ zero_lt_two
  -- We can use the Lovasz form of Kruskal-Katona to get |∂^[n-2k] 𝒜ᶜˢ| ≥ (n-1) choose r
  have kk := kruskal_katona_lovasz_form ‹n - 2 * r ≤ n - r› ((tsub_le_tsub_iff_left ‹1 ≤ n›).2 h1r)
      tsub_le_self h𝒜bar z.le
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝² : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝¹ : LE.le r n
    this✝ : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HM …
    ⊢ False
  -/
  have : n - r - (n - 2 * r) = r := by omega
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝³ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝² : LE.le r n
    this✝¹ : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HM …
    this : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    ⊢ False
  -/
  rw [this] at kk
  -- But this gives a contradiction: `n choose r < |𝒜| + |∂^[n-2k] 𝒜ᶜˢ|`
  have : n.choose r < #(𝒜 ∪ ∂^[n - 2 * r] 𝒜ᶜˢ) := by
    rw [card_union_of_disjoint ‹_›]
    convert lt_of_le_of_lt (add_le_add_left kk _) (add_lt_add_right size _) using 1
    convert Nat.choose_succ_succ _ _ using 3
    all_goals rwa [Nat.sub_one, Nat.succ_pred_eq_of_pos]
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ False
  -/
  apply this.not_le
  /-
    case inr
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ LE.le (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r) …
  -/
  convert Set.Sized.card_le _
    /-
      case h.e'_4.h.e'_1
      n : Nat
      𝒜 : Finset (Finset (Fin n))
      r : Nat
      h𝒜 : (↑𝒜).Intersecting
      h₂ : Set.Sized r ↑𝒜
      h₃ : LE.le r (HDiv.hDiv n 2)
      h1r : GT.gt r 0
      size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
      this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
      this✝³ : LE.le r n
      this✝² : LE.le 1 n
      z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
      h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
      this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
      kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
      this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
      this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
      ⊢ Eq n (Fintype.card (Fin n))
    -/
  · rw [Fintype.card_fin]
    /-
      🎉 no goals
    -/
  /-
    case inr.convert_5
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ Set.Sized r ↑(Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hM …
  -/
  rw [coe_union, Set.sized_union]
  /-
    case inr.convert_5
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ And (Set.Sized r ↑𝒜) (Set.Sized r ↑(Nat.iterate Finset.shadow (HSub.hSub n ( …
  -/
  refine ⟨‹_›, ?_⟩
  /-
    case inr.convert_5
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ Set.Sized r ↑(Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜.comp …
  -/
  convert h𝒜bar.shadow_iterate
  /-
    case h.e'_2
    n : Nat
    𝒜 : Finset (Finset (Fin n))
    r : Nat
    h𝒜 : (↑𝒜).Intersecting
    h₂ : Set.Sized r ↑𝒜
    h₃ : LE.le r (HDiv.hDiv n 2)
    h1r : GT.gt r 0
    size : LT.lt ((HSub.hSub n 1).choose (HSub.hSub r 1)) 𝒜.card
    this✝⁴ : Disjoint 𝒜 (Nat.iterate Finset.shadow (HSub.hSub n (HMul.hMul 2 r)) 𝒜 …
    this✝³ : LE.le r n
    this✝² : LE.le 1 n
    z : LT.lt ((HSub.hSub n 1).choose (HSub.hSub n r)) 𝒜.compls.card
    h𝒜bar : Set.Sized (HSub.hSub n r) ↑𝒜.compls
    this✝¹ : LE.le (HSub.hSub n (HMul.hMul 2 r)) (HSub.hSub n r)
    kk : LE.le ((HSub.hSub n 1).choose r) (Nat.iterate Finset.shadow (HSub.hSub n  …
    this✝ : Eq (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r))) r
    this : LT.lt (n.choose r) (Union.union 𝒜 (Nat.iterate Finset.shadow (HSub.hSub …
    ⊢ Eq r (HSub.hSub (HSub.hSub n r) (HSub.hSub n (HMul.hMul 2 r)))
  -/
  omega
  /-
    🎉 no goals
  -/


