lemma csSup_mem_of_not_isSuccPrelimit
    (hne : s.Nonempty) (hbdd : BddAbove s) (hlim : ¬ IsSuccPrelimit (sSup s)) : sSup s ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddAbove s
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  obtain ⟨y, hy⟩ := not_forall_not.mp hlim
  /-
    case intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddAbove s
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    y : α
    hy : CovBy y (SupSet.sSup s)
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  obtain ⟨i, his, hi⟩ := exists_lt_of_lt_csSup hne hy.lt
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddAbove s
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    y : α
    hy : CovBy y (SupSet.sSup s)
    i : α
    his : Membership.mem s i
    hi : LT.lt y i
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  exact eq_of_le_of_not_lt (le_csSup hbdd his) (hy.2 hi) ▸ his
  /-
    🎉 no goals
  -/


@[deprecated csSup_mem_of_not_isSuccPrelimit (since := "2024-09-05")]
alias csSup_mem_of_not_isSuccLimit := csSup_mem_of_not_isSuccPrelimit


lemma csInf_mem_of_not_isPredPrelimit
    (hne : s.Nonempty) (hbdd : BddBelow s) (hlim : ¬ IsPredPrelimit (sInf s)) : sInf s ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddBelow s
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  obtain ⟨y, hy⟩ := not_forall_not.mp hlim
  /-
    case intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddBelow s
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    y : α
    hy : CovBy (InfSet.sInf s) y
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  obtain ⟨i, his, hi⟩ := exists_lt_of_csInf_lt hne hy.lt
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hne : s.Nonempty
    hbdd : BddBelow s
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    y : α
    hy : CovBy (InfSet.sInf s) y
    i : α
    his : Membership.mem s i
    hi : LT.lt i y
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  exact eq_of_le_of_not_lt (csInf_le hbdd his) (hy.2 · hi) ▸ his
  /-
    🎉 no goals
  -/


@[deprecated csInf_mem_of_not_isPredPrelimit (since := "2024-09-05")]
alias csInf_mem_of_not_isPredLimit := csInf_mem_of_not_isPredPrelimit


lemma exists_eq_ciSup_of_not_isSuccPrelimit
    (hf : BddAbove (range f)) (hf' : ¬ IsSuccPrelimit (⨆ i, f i)) : ∃ i, f i = ⨆ i, f i :=
  csSup_mem_of_not_isSuccPrelimit (range_nonempty f) hf hf'


@[deprecated exists_eq_ciSup_of_not_isSuccPrelimit (since := "2024-09-05")]
alias exists_eq_ciSup_of_not_isSuccLimit := exists_eq_ciSup_of_not_isSuccPrelimit


lemma exists_eq_ciInf_of_not_isPredPrelimit
    (hf : BddBelow (range f)) (hf' : ¬ IsPredPrelimit (⨅ i, f i)) : ∃ i, f i = ⨅ i, f i :=
  csInf_mem_of_not_isPredPrelimit (range_nonempty f) hf hf'


@[deprecated exists_eq_ciInf_of_not_isPredPrelimit (since := "2024-09-05")]
alias exists_eq_ciInf_of_not_isPredLimit := exists_eq_ciInf_of_not_isPredPrelimit


lemma IsLUB.mem_of_nonempty_of_not_isSuccPrelimit
    (hs : IsLUB s x) (hne : s.Nonempty) (hx : ¬ IsSuccPrelimit x) : x ∈ s :=
  hs.csSup_eq hne ▸ csSup_mem_of_not_isSuccPrelimit hne hs.bddAbove (hs.csSup_eq hne ▸ hx)


@[deprecated IsLUB.mem_of_nonempty_of_not_isSuccPrelimit (since := "2024-09-05")]
alias IsLUB.mem_of_nonempty_of_not_isSuccLimit := IsLUB.mem_of_nonempty_of_not_isSuccPrelimit


lemma IsGLB.mem_of_nonempty_of_not_isPredPrelimit
    (hs : IsGLB s x) (hne : s.Nonempty) (hx : ¬ IsPredPrelimit x) : x ∈ s :=
  hs.csInf_eq hne ▸ csInf_mem_of_not_isPredPrelimit hne hs.bddBelow (hs.csInf_eq hne ▸ hx)


@[deprecated IsGLB.mem_of_nonempty_of_not_isPredPrelimit (since := "2024-09-05")]
alias IsGLB.mem_of_nonempty_of_not_isPredLimit := IsGLB.mem_of_nonempty_of_not_isPredPrelimit


lemma IsLUB.exists_of_nonempty_of_not_isSuccPrelimit
    (hf : IsLUB (range f) x) (hx : ¬ IsSuccPrelimit x) : ∃ i, f i = x :=
  hf.mem_of_nonempty_of_not_isSuccPrelimit (range_nonempty f) hx


@[deprecated IsLUB.exists_of_nonempty_of_not_isSuccPrelimit (since := "2024-09-05")]
alias IsLUB.exists_of_nonempty_of_not_isSuccLimit := IsLUB.exists_of_nonempty_of_not_isSuccPrelimit


lemma IsGLB.exists_of_nonempty_of_not_isPredPrelimit
    (hf : IsGLB (range f) x) (hx : ¬ IsPredPrelimit x) : ∃ i, f i = x :=
  hf.mem_of_nonempty_of_not_isPredPrelimit (range_nonempty f) hx


@[deprecated IsGLB.exists_of_nonempty_of_not_isPredPrelimit (since := "2024-09-05")]
alias IsGLB.exists_of_nonempty_of_not_isPredLimit := IsGLB.exists_of_nonempty_of_not_isPredPrelimit


open Classical in
/-- Every conditionally complete linear order with well-founded `<` is a successor order, by setting
the successor of an element to be the infimum of all larger elements. -/
noncomputable def ConditionallyCompleteLinearOrder.toSuccOrder [WellFoundedLT α] :
    SuccOrder α where
  succ a := if IsMax a then a else sInf {b | a < b}
  le_succ a := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a : α
      ⊢ LE.le a ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b))) …
    -/
    by_cases h : IsMax a
      /-
        case pos
        ι : Type u_1
        α : Type u_2
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : Nonempty ι
        f : ι → α
        s : Set α
        x : α
        inst✝ : WellFoundedLT α
        a : α
        h : IsMax a
        ⊢ LE.le a ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b))) …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        α : Type u_2
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : Nonempty ι
        f : ι → α
        s : Set α
        x : α
        inst✝ : WellFoundedLT α
        a : α
        h : Not (IsMax a)
        ⊢ LE.le a ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b))) …
      -/
    · simp only [h, ↓reduceIte]
      /-
        case neg
        ι : Type u_1
        α : Type u_2
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : Nonempty ι
        f : ι → α
        s : Set α
        x : α
        inst✝ : WellFoundedLT α
        a : α
        h : Not (IsMax a)
        ⊢ LE.le a (InfSet.sInf (setOf fun b => LT.lt a b))
      -/
      rw [not_isMax_iff] at h
      /-
        case neg
        ι : Type u_1
        α : Type u_2
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : Nonempty ι
        f : ι → α
        s : Set α
        x : α
        inst✝ : WellFoundedLT α
        a : α
        h : Exists fun b => LT.lt a b
        ⊢ LE.le a (InfSet.sInf (setOf fun b => LT.lt a b))
      -/
      exact le_csInf h (fun b => le_of_lt)
      /-
        🎉 no goals
      -/
  max_of_succ_le hs := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a✝ : α
      hs : LE.le ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b)) …
      ⊢ IsMax a✝
    -/
    by_contra h
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a✝ : α
      hs : LE.le ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b)) …
      h : Not (IsMax a✝)
      ⊢ False
    -/
    simp [h] at hs
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a✝ : α
      h : Not (IsMax a✝)
      hs : LE.le (InfSet.sInf (setOf fun b => LT.lt a✝ b)) a✝
      ⊢ False
    -/
    rw [not_isMax_iff] at h
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a✝ : α
      h : Exists fun b => LT.lt a✝ b
      hs : LE.le (InfSet.sInf (setOf fun b => LT.lt a✝ b)) a✝
      ⊢ False
    -/
    exact hs.not_lt (csInf_mem h)
    /-
      🎉 no goals
    -/
  succ_le_of_lt {a b} ha := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a b : α
      ha : LT.lt a b
      ⊢ LE.le ((fun a => ite (IsMax a) a (InfSet.sInf (setOf fun b => LT.lt a b))) a …
    -/
    simp [ha.not_isMax]
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : Nonempty ι
      f : ι → α
      s : Set α
      x : α
      inst✝ : WellFoundedLT α
      a b : α
      ha : LT.lt a b
      ⊢ LE.le (InfSet.sInf (setOf fun b => LT.lt a b)) b
    -/
    exact csInf_le ⟨a, fun _ hc => hc.le⟩ ha
    /-
      🎉 no goals
    -/


/-- See `csSup_mem_of_not_isSuccPrelimit` for the `ConditionallyCompleteLinearOrder` version. -/
lemma csSup_mem_of_not_isSuccPrelimit'
    (hbdd : BddAbove s) (hlim : ¬ IsSuccPrelimit (sSup s)) : sSup s ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    s : Set α
    hbdd : BddAbove s
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      hbdd : BddAbove EmptyCollection.emptyCollection
      hlim : Not (Order.IsSuccPrelimit (SupSet.sSup EmptyCollection.emptyCollection))
      ⊢ Membership.mem EmptyCollection.emptyCollection (SupSet.sSup EmptyCollection. …
    -/
  · simp [isSuccPrelimit_bot] at hlim
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      s : Set α
      hbdd : BddAbove s
      hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
      hs : s.Nonempty
      ⊢ Membership.mem s (SupSet.sSup s)
    -/
  · exact csSup_mem_of_not_isSuccPrelimit hs hbdd hlim
    /-
      🎉 no goals
    -/


@[deprecated csSup_mem_of_not_isSuccPrelimit' (since := "2024-09-05")]
alias csSup_mem_of_not_isSuccLimit' := csSup_mem_of_not_isSuccPrelimit'


/-- See `exists_eq_ciSup_of_not_isSuccPrelimit` for the
`ConditionallyCompleteLinearOrder` version. -/
lemma exists_eq_ciSup_of_not_isSuccPrelimit'
    (hf : BddAbove (range f)) (hf' : ¬ IsSuccPrelimit (⨆ i, f i)) : ∃ i, f i = ⨆ i, f i :=
  csSup_mem_of_not_isSuccPrelimit' hf hf'


@[deprecated exists_eq_ciSup_of_not_isSuccPrelimit' (since := "2024-09-05")]
alias exists_eq_ciSup_of_not_isSuccLimit' := exists_eq_ciSup_of_not_isSuccPrelimit'


lemma IsLUB.mem_of_not_isSuccPrelimit (hs : IsLUB s x) (hx : ¬ IsSuccPrelimit x) : x ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    s : Set α
    x : α
    hs : IsLUB s x
    hx : Not (Order.IsSuccPrelimit x)
    ⊢ Membership.mem s x
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      x : α
      hx : Not (Order.IsSuccPrelimit x)
      hs : IsLUB EmptyCollection.emptyCollection x
      ⊢ Membership.mem EmptyCollection.emptyCollection x
    -/
  · simp [show x = ⊥ by simpa using hs, isSuccPrelimit_bot] at hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      s : Set α
      x : α
      hs : IsLUB s x
      hx : Not (Order.IsSuccPrelimit x)
      hs' : s.Nonempty
      ⊢ Membership.mem s x
    -/
  · exact hs.mem_of_nonempty_of_not_isSuccPrelimit hs' hx
    /-
      🎉 no goals
    -/


@[deprecated IsLUB.mem_of_not_isSuccPrelimit (since := "2024-09-05")]
alias IsLUB.mem_of_not_isSuccLimit := IsLUB.mem_of_not_isSuccPrelimit


lemma IsLUB.exists_of_not_isSuccPrelimit (hf : IsLUB (range f) x) (hx : ¬ IsSuccPrelimit x) :
    ∃ i, f i = x :=
  hf.mem_of_not_isSuccPrelimit hx


@[deprecated IsLUB.exists_of_not_isSuccPrelimit (since := "2024-09-05")]
alias IsLUB.exists_of_not_isSuccLimit := IsLUB.exists_of_not_isSuccPrelimit


theorem Order.IsSuccPrelimit.sSup_Iio (h : IsSuccPrelimit x) : sSup (Iio x) = x := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    h : Order.IsSuccPrelimit x
    ⊢ Eq (SupSet.sSup (Set.Iio x)) x
  -/
  obtain rfl | hx := eq_bot_or_bot_lt x
    /-
      case inl
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      h : Order.IsSuccPrelimit Bot.bot
      ⊢ Eq (SupSet.sSup (Set.Iio Bot.bot)) Bot.bot
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      x : α
      h : Order.IsSuccPrelimit x
      hx : LT.lt Bot.bot x
      ⊢ Eq (SupSet.sSup (Set.Iio x)) x
    -/
  · refine (csSup_le ⟨⊥, hx⟩ fun a ha ↦ ha.le).antisymm <| le_of_forall_lt fun a ha ↦ ?_
    /-
      case inr
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      x : α
      h : Order.IsSuccPrelimit x
      hx : LT.lt Bot.bot x
      a : α
      ha : LT.lt a x
      ⊢ LT.lt a (SupSet.sSup (Set.Iio x))
    -/
    rw [lt_csSup_iff' bddAbove_Iio]
    /-
      case inr
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      x : α
      h : Order.IsSuccPrelimit x
      hx : LT.lt Bot.bot x
      a : α
      ha : LT.lt a x
      ⊢ Exists fun b => And (Membership.mem (Set.Iio x) b) (LT.lt a b)
    -/
    obtain ⟨b, hb', hb⟩ := (not_covBy_iff ha).1 (h a)
    /-
      case inr.intro.intro
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot α
      x : α
      h : Order.IsSuccPrelimit x
      hx : LT.lt Bot.bot x
      a : α
      ha : LT.lt a x
      b : α
      hb' : LT.lt a b
      hb : LT.lt b x
      ⊢ Exists fun b => And (Membership.mem (Set.Iio x) b) (LT.lt a b)
    -/
    use b, hb
    /-
      🎉 no goals
    -/


theorem Order.IsSuccPrelimit.iSup_Iio (h : IsSuccPrelimit x) : ⨆ a : Iio x, a.1 = x := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    h : Order.IsSuccPrelimit x
    ⊢ Eq (iSup fun a => ↑a) x
  -/
  rw [← sSup_eq_iSup', h.sSup_Iio]
  /-
    🎉 no goals
  -/


theorem Order.IsSuccLimit.sSup_Iio (h : IsSuccLimit x) : sSup (Iio x) = x :=
  h.isSuccPrelimit.sSup_Iio


theorem Order.IsSuccLimit.iSup_Iio (h : IsSuccLimit x) : ⨆ a : Iio x, a.1 = x :=
  h.isSuccPrelimit.iSup_Iio


theorem sSup_Iio_eq_self_iff_isSuccPrelimit : sSup (Iio x) = x ↔ IsSuccPrelimit x := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    ⊢ Iff (Eq (SupSet.sSup (Set.Iio x)) x) (Order.IsSuccPrelimit x)
  -/
  refine ⟨fun h ↦ ?_, IsSuccPrelimit.sSup_Iio⟩
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    h : Eq (SupSet.sSup (Set.Iio x)) x
    ⊢ Order.IsSuccPrelimit x
  -/
  by_contra hx
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    h : Eq (SupSet.sSup (Set.Iio x)) x
    hx : Not (Order.IsSuccPrelimit x)
    ⊢ False
  -/
  rw [← h] at hx
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    h : Eq (SupSet.sSup (Set.Iio x)) x
    hx : Not (Order.IsSuccPrelimit (SupSet.sSup (Set.Iio x)))
    ⊢ False
  -/
  simpa [h] using csSup_mem_of_not_isSuccPrelimit' bddAbove_Iio hx
  /-
    🎉 no goals
  -/


theorem iSup_Iio_eq_self_iff_isSuccPrelimit : ⨆ a : Iio x, a.1 = x ↔ IsSuccPrelimit x := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot α
    x : α
    ⊢ Iff (Eq (iSup fun a => ↑a) x) (Order.IsSuccPrelimit x)
  -/
  rw [← sSup_eq_iSup', sSup_Iio_eq_self_iff_isSuccPrelimit]
  /-
    🎉 no goals
  -/


lemma sSup_mem_of_not_isSuccPrelimit (hlim : ¬ IsSuccPrelimit (sSup s)) : sSup s ∈ s := by
  /-
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  obtain ⟨y, hy⟩ := not_forall_not.mp hlim
  /-
    case intro
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    y : α
    hy : CovBy y (SupSet.sSup s)
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  obtain ⟨i, his, hi⟩ := lt_sSup_iff.mp hy.lt
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsSuccPrelimit (SupSet.sSup s))
    y : α
    hy : CovBy y (SupSet.sSup s)
    i : α
    his : Membership.mem s i
    hi : LT.lt y i
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  exact eq_of_le_of_not_lt (le_sSup his) (hy.2 hi) ▸ his
  /-
    🎉 no goals
  -/


@[deprecated sSup_mem_of_not_isSuccPrelimit (since := "2024-09-05")]
alias sSup_mem_of_not_isSuccLimit := sSup_mem_of_not_isSuccPrelimit


lemma sInf_mem_of_not_isPredPrelimit (hlim : ¬ IsPredPrelimit (sInf s)) : sInf s ∈ s := by
  /-
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  obtain ⟨y, hy⟩ := not_forall_not.mp hlim
  /-
    case intro
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    y : α
    hy : CovBy (InfSet.sInf s) y
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  obtain ⟨i, his, hi⟩ := sInf_lt_iff.mp hy.lt
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : CompleteLinearOrder α
    s : Set α
    hlim : Not (Order.IsPredPrelimit (InfSet.sInf s))
    y : α
    hy : CovBy (InfSet.sInf s) y
    i : α
    his : Membership.mem s i
    hi : LT.lt i y
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  exact eq_of_le_of_not_lt (sInf_le his) (hy.2 · hi) ▸ his
  /-
    🎉 no goals
  -/


@[deprecated sInf_mem_of_not_isPredPrelimit (since := "2024-09-05")]
alias sInf_mem_of_not_isPredLimit := sInf_mem_of_not_isPredPrelimit


lemma exists_eq_iSup_of_not_isSuccPrelimit (hf : ¬ IsSuccPrelimit (⨆ i, f i)) :
    ∃ i, f i = ⨆ i, f i :=
  sSup_mem_of_not_isSuccPrelimit hf


@[deprecated exists_eq_iSup_of_not_isSuccPrelimit (since := "2024-09-05")]
alias exists_eq_iSup_of_not_isSuccLimit := exists_eq_iSup_of_not_isSuccPrelimit


lemma exists_eq_iInf_of_not_isPredPrelimit (hf : ¬ IsPredPrelimit (⨅ i, f i)) :
    ∃ i, f i = ⨅ i, f i :=
  sInf_mem_of_not_isPredPrelimit hf


@[deprecated exists_eq_iInf_of_not_isPredPrelimit (since := "2024-09-05")]
alias exists_eq_iInf_of_not_isPredLimit := exists_eq_iInf_of_not_isPredPrelimit


lemma IsGLB.mem_of_not_isPredPrelimit (hs : IsGLB s x) (hx : ¬ IsPredPrelimit x) : x ∈ s :=
  hs.sInf_eq ▸ sInf_mem_of_not_isPredPrelimit (hs.sInf_eq ▸ hx)


@[deprecated IsGLB.mem_of_not_isPredPrelimit (since := "2024-09-05")]
alias IsGLB.mem_of_not_isPredLimit := IsGLB.mem_of_not_isPredPrelimit


lemma IsGLB.exists_of_not_isPredPrelimit (hf : IsGLB (range f) x) (hx : ¬ IsPredPrelimit x) :
    ∃ i, f i = x :=
  hf.mem_of_not_isPredPrelimit hx


@[deprecated IsGLB.exists_of_not_isPredPrelimit (since := "2024-09-05")]
alias IsGLB.exists_of_not_isPredLimit := IsGLB.exists_of_not_isPredPrelimit


