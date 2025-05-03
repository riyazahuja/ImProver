/-- The rank of an element `a` accessible under a relation `r` is defined recursively as the
smallest ordinal greater than the ranks of all elements below it (i.e. elements `b` such that
`r b a`). -/
noncomputable def rank (h : Acc r a) : Ordinal.{u} :=
  Acc.recOn h fun a _h ih => ⨆ b : { b // r b a }, Order.succ (ih b b.2)


theorem rank_eq (h : Acc r a) :
    h.rank = ⨆ b : { b // r b a }, Order.succ (h.inv b.2).rank := by
  /-
    α : Type u
    a : α
    r : α → α → Prop
    h : Acc r a
    ⊢ Eq h.rank (iSup fun b => Order.succ ⋯.rank)
  -/
  change (Acc.intro a fun _ => h.inv).rank = _
  /-
    α : Type u
    a : α
    r : α → α → Prop
    h : Acc r a
    ⊢ Eq ⋯.rank (iSup fun b => Order.succ ⋯.rank)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- if `r a b` then the rank of `a` is less than the rank of `b`. -/
theorem rank_lt_of_rel (hb : Acc r b) (h : r a b) : (hb.inv h).rank < hb.rank :=
  (Order.lt_succ _).trans_le <| by
    /-
      α : Type u
      a b : α
      r : α → α → Prop
      hb : Acc r b
      h : r a b
      ⊢ LE.le (Order.succ ⋯.rank) hb.rank
    -/
    rw [hb.rank_eq]
    /-
      α : Type u
      a b : α
      r : α → α → Prop
      hb : Acc r b
      h : r a b
      ⊢ LE.le (Order.succ ⋯.rank) (iSup fun b_1 => Order.succ ⋯.rank)
    -/
    exact Ordinal.le_iSup _ (⟨a, h⟩ : {a // r a b})
    /-
      🎉 no goals
    -/


theorem mem_range_rank_of_le {o : Ordinal} (ha : Acc r a) (ho : o ≤ ha.rank) :
    ∃ (b : α) (hb : Acc r b), hb.rank = o := by
  /-
    α : Type u
    a : α
    r : α → α → Prop
    o : Ordinal.{u}
    ha : Acc r a
    ho : LE.le o ha.rank
    ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
  -/
  obtain rfl | ho := ho.eq_or_lt
    /-
      case inl
      α : Type u
      a : α
      r : α → α → Prop
      ha : Acc r a
      ho : LE.le ha.rank ha.rank
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank ha.rank
    -/
  · exact ⟨a, ha, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      a : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha : Acc r a
      ho✝ : LE.le o ha.rank
      ho : LT.lt o ha.rank
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
    -/
  · revert ho
    /-
      case inr
      α : Type u
      a : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha : Acc r a
      ho : LE.le o ha.rank
      ⊢ LT.lt o ha.rank → Exists fun b => Exists fun hb => Eq hb.rank o
    -/
    refine ha.recOn fun a ha IH ho ↦ ?_
    /-
      case inr
      α : Type u
      a✝ : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha✝ : Acc r a✝
      ho✝ : LE.le o ha✝.rank
      a : α
      ha : ∀ (y : α), r y a → Acc r y
      IH : ∀ (y : α) (a : r y a), LT.lt o ⋯.rank → Exists fun b => Exists fun hb =>  …
      ho : LT.lt o ⋯.rank
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
    -/
    rw [rank_eq, Ordinal.lt_iSup_iff] at ho
    /-
      case inr
      α : Type u
      a✝ : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha✝ : Acc r a✝
      ho✝ : LE.le o ha✝.rank
      a : α
      ha : ∀ (y : α), r y a → Acc r y
      IH : ∀ (y : α) (a : r y a), LT.lt o ⋯.rank → Exists fun b => Exists fun hb =>  …
      ho : Exists fun i => LT.lt o (Order.succ ⋯.rank)
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
    -/
    obtain ⟨⟨b, hb⟩, ho⟩ := ho
    /-
      case inr.intro.mk
      α : Type u
      a✝ : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha✝ : Acc r a✝
      ho✝ : LE.le o ha✝.rank
      a : α
      ha : ∀ (y : α), r y a → Acc r y
      IH : ∀ (y : α) (a : r y a), LT.lt o ⋯.rank → Exists fun b => Exists fun hb =>  …
      b : α
      hb : r b a
      ho : LT.lt o (Order.succ ⋯.rank)
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
    -/
    rw [Order.lt_succ_iff] at ho
    /-
      case inr.intro.mk
      α : Type u
      a✝ : α
      r : α → α → Prop
      o : Ordinal.{u}
      ha✝ : Acc r a✝
      ho✝ : LE.le o ha✝.rank
      a : α
      ha : ∀ (y : α), r y a → Acc r y
      IH : ∀ (y : α) (a : r y a), LT.lt o ⋯.rank → Exists fun b => Exists fun hb =>  …
      b : α
      hb : r b a
      ho : LE.le o ⋯.rank
      ⊢ Exists fun b => Exists fun hb => Eq hb.rank o
    -/
    obtain rfl | ho := ho.eq_or_lt
    /-
      case inr.intro.mk.inl
      α : Type u
      a✝ : α
      r : α → α → Prop
      ha✝ : Acc r a✝
      a : α
      ha : ∀ (y : α), r y a → Acc r y
      b : α
      hb : r b a
      ho✝ : LE.le ⋯.rank ha✝.rank
      IH : ∀ (y : α) (a_1 : r y a), LT.lt ⋯.rank ⋯.rank → Exists fun b_1 => Exists f …
      ho : LE.le ⋯.rank ⋯.rank
      ⊢ Exists fun b_1 => Exists fun hb_1 => Eq hb_1.rank ⋯.rank
    -/
    exacts [⟨b, ha b hb, rfl⟩, IH _ hb ho]
    /-
      🎉 no goals
    -/


/-- The rank of an element `a` under a well-founded relation `r` is defined recursively as the
smallest ordinal greater than the ranks of all elements below it (i.e. elements `b` such that
`r b a`). -/
noncomputable def rank (a : α) : Ordinal.{u} :=
  (hwf.apply r a).rank


theorem rank_eq (a : α) : rank r a = ⨆ b : { b // r b a }, Order.succ (rank r b) :=
  (hwf.apply r a).rank_eq


theorem rank_lt_of_rel (h : r a b) : rank r a < rank r b :=
  Acc.rank_lt_of_rel _ h


theorem mem_range_rank_of_le {o : Ordinal} (h : o ≤ rank r a) : o ∈ Set.range (rank r) := by
  /-
    α : Type u
    a : α
    r : α → α → Prop
    hwf : IsWellFounded α r
    o : Ordinal.{u}
    h : LE.le o (IsWellFounded.rank r a)
    ⊢ Membership.mem (Set.range (IsWellFounded.rank r)) o
  -/
  obtain ⟨b, hb, rfl⟩ := Acc.mem_range_rank_of_le (hwf.apply r a) h
  /-
    case intro.intro
    α : Type u
    a : α
    r : α → α → Prop
    hwf : IsWellFounded α r
    b : α
    hb : Acc r b
    h : LE.le hb.rank (IsWellFounded.rank r a)
    ⊢ Membership.mem (Set.range (IsWellFounded.rank r)) hb.rank
  -/
  exact ⟨b, rfl⟩
  /-
    🎉 no goals
  -/


theorem WellFoundedLT.rank_strictMono [Preorder α] [WellFoundedLT α] :
    StrictMono (IsWellFounded.rank (α := α) (· < ·)) :=
  fun _ _ => IsWellFounded.rank_lt_of_rel


theorem WellFoundedGT.rank_strictAnti [Preorder α] [WellFoundedGT α] :
    StrictAnti (IsWellFounded.rank (α := α) (· > ·)) :=
  fun _ _ a => IsWellFounded.rank_lt_of_rel a


@[simp]
theorem IsWellFounded.rank_eq_typein (r) [IsWellOrder α r] : rank r = Ordinal.typein r := by
  classical
  letI := linearOrderOfSTO r
  ext a
  exact InitialSeg.eq (⟨(OrderEmbedding.ofStrictMono _ WellFoundedLT.rank_strictMono).ltEmbedding,
    fun a b h ↦ mem_range_rank_of_le h.le⟩) (Ordinal.typein r) a


/-- The rank of an element `a` under a well-founded relation `r` is defined inductively as the
smallest ordinal greater than the ranks of all elements below it (i.e. elements `b` such that
`r b a`). -/
@[deprecated IsWellFounded.rank (since := "2024-09-07")]
noncomputable def rank (a : α) : Ordinal.{u} :=
  (hwf.apply a).rank


set_option linter.deprecated false in
@[deprecated IsWellFounded.rank_eq (since := "2024-09-07")]
theorem rank_eq : hwf.rank a = ⨆ b : { b // r b a }, Order.succ (hwf.rank b) :=
  (hwf.apply a).rank_eq


set_option linter.deprecated false in
@[deprecated IsWellFounded.rank_lt_of_rel (since := "2024-09-07")]
theorem rank_lt_of_rel (h : r a b) : hwf.rank a < hwf.rank b :=
  Acc.rank_lt_of_rel _ h


set_option linter.deprecated false in
@[deprecated WellFoundedLT.rank_strictMono (since := "2024-09-07")]
theorem rank_strictMono [Preorder α] [WellFoundedLT α] :
    StrictMono (rank <| @wellFounded_lt α _ _) := fun _ _ => rank_lt_of_rel _


set_option linter.deprecated false in
@[deprecated WellFoundedGT.rank_strictAnti (since := "2024-09-07")]
theorem rank_strictAnti [Preorder α] [WellFoundedGT α] :
    StrictAnti (rank <| @wellFounded_gt α _ _) := fun _ _ => rank_lt_of_rel wellFounded_gt


