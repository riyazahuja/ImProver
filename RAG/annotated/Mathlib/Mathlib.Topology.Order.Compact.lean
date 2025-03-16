/-- This typeclass says that all closed intervals in `α` are compact. This is true for all
conditionally complete linear orders with order topology and products (finite or infinite)
of such spaces. -/
class CompactIccSpace (α : Type*) [TopologicalSpace α] [Preorder α] : Prop where
  /-- A closed interval `Set.Icc a b` is a compact set for all `a` and `b`. -/
  isCompact_Icc : ∀ {a b : α}, IsCompact (Icc a b)


lemma CompactIccSpace.mk' [TopologicalSpace α] [Preorder α]
    (h : ∀ {a b : α}, a ≤ b → IsCompact (Icc a b)) : CompactIccSpace α where
                                                  /-
                                                    α : Type u_1
                                                    inst✝¹ : TopologicalSpace α
                                                    inst✝ : Preorder α
                                                    h : ∀ {a b : α}, LE.le a b → IsCompact (Set.Icc a b)
                                                    a b : α
                                                    hab : Not (LE.le a b)
                                                    ⊢ IsCompact (Set.Icc a b)
                                                  -/
  isCompact_Icc {a b} := by_cases h fun hab => by rw [Icc_eq_empty hab]; exact isCompact_empty
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: drop one `'`

lemma CompactIccSpace.mk'' [TopologicalSpace α] [PartialOrder α]
    (h : ∀ {a b : α}, a < b → IsCompact (Icc a b)) : CompactIccSpace α :=
                                        /-
                                          α : Type u_1
                                          inst✝¹ : TopologicalSpace α
                                          inst✝ : PartialOrder α
                                          h : ∀ {a b : α}, LT.lt a b → IsCompact (Set.Icc a b)
                                          a✝ b✝ : α
                                          hab : LE.le a✝ b✝
                                          ⊢ Eq a✝ b✝ → IsCompact (Set.Icc a✝ b✝)
                                        -/
  .mk' fun hab => hab.eq_or_lt.elim (by rintro rfl; simp) h
                                                    /-
                                                      🎉 no goals
                                                    -/


instance [TopologicalSpace α] [Preorder α] [CompactIccSpace α] : CompactIccSpace (αᵒᵈ) where
  isCompact_Icc := by
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : CompactIccSpace α
      ⊢ ∀ {a b : OrderDual α}, IsCompact (Set.Icc a b)
    -/
    intro a b
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : CompactIccSpace α
      a b : OrderDual α
      ⊢ IsCompact (Set.Icc a b)
    -/
    convert isCompact_Icc (α := α) (a := b) (b := a) using 1
    /-
      case h.e'_3.h
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : CompactIccSpace α
      a b : OrderDual α
      e_1✝ : Eq (OrderDual α) α
      ⊢ Eq (Set.Icc a b) (Set.Icc b a)
    -/
    exact dual_Icc (α := α)
    /-
      🎉 no goals
    -/


/-- A closed interval in a conditionally complete linear order is compact. -/
instance (priority := 100) ConditionallyCompleteLinearOrder.toCompactIccSpace (α : Type*)
    [ConditionallyCompleteLinearOrder α] [TopologicalSpace α] [OrderTopology α] :
    CompactIccSpace α := by
  /-
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    ⊢ CompactIccSpace α
  -/
  refine .mk'' fun {a b} hlt => ?_
  /-
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    ⊢ IsCompact (Set.Icc a b)
  -/
  rcases le_or_lt a b with hab | hab
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    ⊢ IsCompact (Set.Icc a b)
  -/
  swap
    /-
      case inr
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      hlt : LT.lt a b
      hab : LT.lt b a
      ⊢ IsCompact (Set.Icc a b)
    -/
  · simp [hab]
    /-
      🎉 no goals
    -/
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    ⊢ IsCompact (Set.Icc a b)
  -/
  refine isCompact_iff_ultrafilter_le_nhds.2 fun f hf => ?_
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : LE.le (↑f) (Filter.principal (Set.Icc a b))
    ⊢ Exists fun x => And (Membership.mem (Set.Icc a b) x) (LE.le (↑f) (nhds x))
  -/
  contrapose! hf
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    ⊢ Not (LE.le (↑f) (Filter.principal (Set.Icc a b)))
  -/
  rw [le_principal_iff]
  have hpt : ∀ x ∈ Icc a b, {x} ∉ f := fun x hx hxf =>
    hf x hx ((le_pure_iff.2 hxf).trans (pure_le_nhds x))
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  set s := { x ∈ Icc a b | Icc a x ∉ f }
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  have hsb : b ∈ upperBounds s := fun x hx => hx.1.2
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  have sbd : BddAbove s := ⟨b, hsb⟩
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  have ha : a ∈ s := by simp [s, hpt, hab]
  /-
    case inl
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  rcases hab.eq_or_lt with (rfl | _hlt)
    /-
      case inl.inl
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a : α
      f : Ultrafilter α
      hlt : LT.lt a a
      hab : LE.le a a
      hf : ∀ (x : α), Membership.mem (Set.Icc a a) x → Not (LE.le (↑f) (nhds x))
      hpt : ∀ (x : α), Membership.mem (Set.Icc a a) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a a) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) a
      sbd : BddAbove s
      ha : Membership.mem s a
      ⊢ Not (Membership.mem (↑f) (Set.Icc a a))
    -/
  · exact ha.2
    /-
      🎉 no goals
    -/
  -- Porting note: the `obtain` below was instead
  -- `set c := Sup s`
  -- `have hsc : IsLUB s c := isLUB_csSup ⟨a, ha⟩ sbd`
  /-
    case inl.inr
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  obtain ⟨c, hsc⟩ : ∃ c, IsLUB s c := ⟨sSup s, isLUB_csSup ⟨a, ha⟩ ⟨b, hsb⟩⟩
  /-
    case inl.inr.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  have hc : c ∈ Icc a b := ⟨hsc.1 ha, hsc.2 hsb⟩
  /-
    case inl.inr.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hf : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (LE.le (↑f) (nhds x))
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  specialize hf c hc
  have hcs : c ∈ s := by
    rcases hc.1.eq_or_lt with (rfl | hlt); · assumption
    refine ⟨hc, fun hcf => hf fun U hU => ?_⟩
    rcases (mem_nhdsLE_iff_exists_Ioc_subset' hlt).1 (mem_nhdsWithin_of_mem_nhds hU)
      with ⟨x, hxc, hxU⟩
    rcases ((hsc.frequently_mem ⟨a, ha⟩).and_eventually (Ioc_mem_nhdsLE hxc)).exists
      with ⟨y, ⟨_hyab, hyf⟩, hy⟩
    refine mem_of_superset (f.diff_mem_iff.2 ⟨hcf, hyf⟩) (Subset.trans ?_ hxU)
    rw [diff_subset_iff]
    exact Subset.trans Icc_subset_Icc_union_Ioc <| union_subset_union Subset.rfl <|
      Ioc_subset_Ioc_left hy.1.le
  /-
    case inl.inr.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  rcases hc.2.eq_or_lt with (rfl | hlt)
    /-
      case inl.inr.intro.inl
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a : α
      f : Ultrafilter α
      c : α
      hf : Not (LE.le (↑f) (nhds c))
      hlt : LT.lt a c
      hab : LE.le a c
      hpt : ∀ (x : α), Membership.mem (Set.Icc a c) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a c) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) c
      sbd : BddAbove s
      ha : Membership.mem s a
      _hlt : LT.lt a c
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a c) c
      hcs : Membership.mem s c
      ⊢ Not (Membership.mem (↑f) (Set.Icc a c))
    -/
  · exact hcs.2
    /-
      🎉 no goals
    -/
  /-
    case inl.inr.intro.inr
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt✝ : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    hlt : LT.lt c b
    ⊢ Not (Membership.mem (↑f) (Set.Icc a b))
  -/
  exfalso
  /-
    case inl.inr.intro.inr
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt✝ : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    hlt : LT.lt c b
    ⊢ False
  -/
  refine hf fun U hU => ?_
  rcases (mem_nhdsGE_iff_exists_mem_Ioc_Ico_subset hlt).1 (mem_nhdsWithin_of_mem_nhds hU)
    with ⟨y, hxy, hyU⟩
  /-
    case inl.inr.intro.inr.intro.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt✝ : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    hlt : LT.lt c b
    U : Set α
    hU : Membership.mem (nhds c) U
    y : α
    hxy : Membership.mem (Set.Ioc c b) y
    hyU : HasSubset.Subset (Set.Ico c y) U
    ⊢ Membership.mem (↑f) U
  -/
  refine mem_of_superset ?_ hyU; clear! U
  /-
    case inl.inr.intro.inr.intro.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt✝ : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    hlt : LT.lt c b
    y : α
    hxy : Membership.mem (Set.Ioc c b) y
    ⊢ Membership.mem (↑f) (Set.Ico c y)
  -/
  have hy : y ∈ Icc a b := ⟨hc.1.trans hxy.1.le, hxy.2⟩
  /-
    case inl.inr.intro.inr.intro.intro
    α✝ : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    hlt✝ : LT.lt a b
    hab : LE.le a b
    f : Ultrafilter α
    hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
    s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ha : Membership.mem s a
    _hlt : LT.lt a b
    c : α
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    hf : Not (LE.le (↑f) (nhds c))
    hcs : Membership.mem s c
    hlt : LT.lt c b
    y : α
    hxy : Membership.mem (Set.Ioc c b) y
    hy : Membership.mem (Set.Icc a b) y
    ⊢ Membership.mem (↑f) (Set.Ico c y)
  -/
  by_cases hay : Icc a y ∈ f
    /-
      case pos
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      hlt✝ : LT.lt a b
      hab : LE.le a b
      f : Ultrafilter α
      hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      ha : Membership.mem s a
      _hlt : LT.lt a b
      c : α
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      hf : Not (LE.le (↑f) (nhds c))
      hcs : Membership.mem s c
      hlt : LT.lt c b
      y : α
      hxy : Membership.mem (Set.Ioc c b) y
      hy : Membership.mem (Set.Icc a b) y
      hay : Membership.mem f (Set.Icc a y)
      ⊢ Membership.mem (↑f) (Set.Ico c y)
    -/
  · refine mem_of_superset (f.diff_mem_iff.2 ⟨f.diff_mem_iff.2 ⟨hay, hcs.2⟩, hpt y hy⟩) ?_
    /-
      case pos
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      hlt✝ : LT.lt a b
      hab : LE.le a b
      f : Ultrafilter α
      hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      ha : Membership.mem s a
      _hlt : LT.lt a b
      c : α
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      hf : Not (LE.le (↑f) (nhds c))
      hcs : Membership.mem s c
      hlt : LT.lt c b
      y : α
      hxy : Membership.mem (Set.Ioc c b) y
      hy : Membership.mem (Set.Icc a b) y
      hay : Membership.mem f (Set.Icc a y)
      ⊢ HasSubset.Subset (SDiff.sdiff (SDiff.sdiff (Set.Icc a y) (Set.Icc a c)) (Sin …
    -/
    rw [diff_subset_iff, union_comm, Ico_union_right hxy.1.le, diff_subset_iff]
    /-
      case pos
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      hlt✝ : LT.lt a b
      hab : LE.le a b
      f : Ultrafilter α
      hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      ha : Membership.mem s a
      _hlt : LT.lt a b
      c : α
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      hf : Not (LE.le (↑f) (nhds c))
      hcs : Membership.mem s c
      hlt : LT.lt c b
      y : α
      hxy : Membership.mem (Set.Ioc c b) y
      hy : Membership.mem (Set.Icc a b) y
      hay : Membership.mem f (Set.Icc a y)
      ⊢ HasSubset.Subset (Set.Icc a y) (Union.union (Set.Icc a c) (Set.Icc c y))
    -/
    exact Icc_subset_Icc_union_Icc
    /-
      🎉 no goals
    -/
    /-
      case neg
      α✝ : Type u_1
      α : Type u_2
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      hlt✝ : LT.lt a b
      hab : LE.le a b
      f : Ultrafilter α
      hpt : ∀ (x : α), Membership.mem (Set.Icc a b) x → Not (Membership.mem f (Singl …
      s : Set α := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Not (Members …
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      ha : Membership.mem s a
      _hlt : LT.lt a b
      c : α
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      hf : Not (LE.le (↑f) (nhds c))
      hcs : Membership.mem s c
      hlt : LT.lt c b
      y : α
      hxy : Membership.mem (Set.Ioc c b) y
      hy : Membership.mem (Set.Icc a b) y
      hay : Not (Membership.mem f (Set.Icc a y))
      ⊢ Membership.mem (↑f) (Set.Ico c y)
    -/
  · exact ((hsc.1 ⟨hy, hay⟩).not_lt hxy.1).elim
    /-
      🎉 no goals
    -/


instance {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] [∀ i, TopologicalSpace (α i)]
    [∀ i, CompactIccSpace (α i)] : CompactIccSpace (∀ i, α i) :=
  ⟨fun {a b} => (pi_univ_Icc a b ▸ isCompact_univ_pi) fun _ => isCompact_Icc⟩


instance Pi.compact_Icc_space' {α β : Type*} [Preorder β] [TopologicalSpace β]
    [CompactIccSpace β] : CompactIccSpace (α → β) :=
  inferInstance


instance {α β : Type*} [Preorder α] [TopologicalSpace α] [CompactIccSpace α] [Preorder β]
    [TopologicalSpace β] [CompactIccSpace β] : CompactIccSpace (α × β) :=
  ⟨fun {a b} => (Icc_prod_eq a b).symm ▸ isCompact_Icc.prod isCompact_Icc⟩


/-- An unordered closed interval is compact. -/
theorem isCompact_uIcc {α : Type*} [LinearOrder α] [TopologicalSpace α] [CompactIccSpace α]
    {a b : α} : IsCompact (uIcc a b) :=
  isCompact_Icc

-- See note [lower instance priority]

/-- A complete linear order is a compact space.

We do not register an instance for a `[CompactIccSpace α]` because this would only add instances
for products (indexed or not) of complete linear orders, and we have instances with higher priority
that cover these cases. -/
instance (priority := 100) compactSpace_of_completeLinearOrder {α : Type*} [CompleteLinearOrder α]
    [TopologicalSpace α] [OrderTopology α] : CompactSpace α :=
      /-
        α✝ : Type u_1
        α : Type u_2
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        ⊢ IsCompact Set.univ
      -/
  ⟨by simp only [← Icc_bot_top, isCompact_Icc]⟩
      /-
        🎉 no goals
      -/


instance compactSpace_Icc (a b : α) : CompactSpace (Icc a b) :=
  isCompact_iff_compactSpace.mp isCompact_Icc


theorem IsCompact.exists_isLeast [ClosedIicTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : ∃ x, IsLeast s x := by
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : IsCompact s
    ne_s : s.Nonempty
    ⊢ Exists fun x => IsLeast s x
  -/
  haveI : Nonempty s := ne_s.to_subtype
  suffices (s ∩ ⋂ x ∈ s, Iic x).Nonempty from
    ⟨this.choose, this.choose_spec.1, mem_iInter₂.mp this.choose_spec.2⟩
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : IsCompact s
    ne_s : s.Nonempty
    this : Nonempty ↑s
    ⊢ (Inter.inter s (Set.iInter fun x => Set.iInter fun h => Set.Iic x)).Nonempty
  -/
  rw [biInter_eq_iInter]
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : IsCompact s
    ne_s : s.Nonempty
    this : Nonempty ↑s
    ⊢ (Inter.inter s (Set.iInter fun x => Set.Iic ↑x)).Nonempty
  -/
  by_contra H
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : IsCompact s
    ne_s : s.Nonempty
    this : Nonempty ↑s
    H : Not (Inter.inter s (Set.iInter fun x => Set.Iic ↑x)).Nonempty
    ⊢ False
  -/
  rw [not_nonempty_iff_eq_empty] at H
  rcases hs.elim_directed_family_closed (fun x : s => Iic ↑x) (fun x => isClosed_Iic) H
      (Monotone.directed_ge fun _ _ h => Iic_subset_Iic.mpr h) with ⟨x, hx⟩
  /-
    case intro
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : IsCompact s
    ne_s : s.Nonempty
    this : Nonempty ↑s
    H : Eq (Inter.inter s (Set.iInter fun x => Set.Iic ↑x)) EmptyCollection.emptyC …
    x : ↑s
    hx : Eq (Inter.inter s (Set.Iic ↑x)) EmptyCollection.emptyCollection
    ⊢ False
  -/
  exact not_nonempty_iff_eq_empty.mpr hx ⟨x, x.2, le_rfl⟩
  /-
    🎉 no goals
  -/


theorem IsCompact.exists_isGreatest [ClosedIciTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : ∃ x, IsGreatest s x :=
  IsCompact.exists_isLeast (α := αᵒᵈ) hs ne_s


theorem IsCompact.exists_isGLB [ClosedIicTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : ∃ x ∈ s, IsGLB s x :=
  (hs.exists_isLeast ne_s).imp (fun x (hx : IsLeast s x) => ⟨hx.1, hx.isGLB⟩)


theorem IsCompact.exists_isLUB [ClosedIciTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : ∃ x ∈ s, IsLUB s x :=
  IsCompact.exists_isGLB (α := αᵒᵈ) hs ne_s


theorem cocompact_le_atBot_atTop [CompactIccSpace α] :
    cocompact α ≤ atBot ⊔ atTop := by
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactIccSpace α
    ⊢ LE.le (Filter.cocompact α) (Max.max Filter.atBot Filter.atTop)
  -/
  refine fun s hs ↦ mem_cocompact.mpr <| (isEmpty_or_nonempty α).casesOn ?_ ?_ <;> intro
    /-
      case refine_1
      α : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : CompactIccSpace α
      s : Set α
      hs : Membership.mem (Max.max Filter.atBot Filter.atTop) s
      h✝ : IsEmpty α
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) s)
    -/
  · exact ⟨∅, isCompact_empty, fun x _ ↦ (IsEmpty.false x).elim⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : CompactIccSpace α
      s : Set α
      hs : Membership.mem (Max.max Filter.atBot Filter.atTop) s
      h✝ : Nonempty α
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) s)
    -/
  · obtain ⟨t, ht⟩ := mem_atBot_sets.mp hs.1
    /-
      case refine_2.intro
      α : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : CompactIccSpace α
      s : Set α
      hs : Membership.mem (Max.max Filter.atBot Filter.atTop) s
      h✝ : Nonempty α
      t : α
      ht : ∀ (b : α), LE.le b t → Membership.mem s b
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) s)
    -/
    obtain ⟨u, hu⟩ := mem_atTop_sets.mp hs.2
    /-
      case refine_2.intro.intro
      α : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : CompactIccSpace α
      s : Set α
      hs : Membership.mem (Max.max Filter.atBot Filter.atTop) s
      h✝ : Nonempty α
      t : α
      ht : ∀ (b : α), LE.le b t → Membership.mem s b
      u : α
      hu : ∀ (b : α), GE.ge b u → Membership.mem s b
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) s)
    -/
    refine ⟨Icc t u, isCompact_Icc, fun x hx ↦ ?_⟩
    /-
      case refine_2.intro.intro
      α : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : CompactIccSpace α
      s : Set α
      hs : Membership.mem (Max.max Filter.atBot Filter.atTop) s
      h✝ : Nonempty α
      t : α
      ht : ∀ (b : α), LE.le b t → Membership.mem s b
      u : α
      hu : ∀ (b : α), GE.ge b u → Membership.mem s b
      x : α
      hx : Membership.mem (HasCompl.compl (Set.Icc t u)) x
      ⊢ Membership.mem s x
    -/
    exact (not_and_or.mp hx).casesOn (fun h ↦ ht x (le_of_not_le h)) fun h ↦ hu x (le_of_not_le h)
    /-
      🎉 no goals
    -/


theorem cocompact_le_atBot [OrderTop α] [CompactIccSpace α] :
    cocompact α ≤ atBot := by
  /-
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTop α
    inst✝ : CompactIccSpace α
    ⊢ LE.le (Filter.cocompact α) Filter.atBot
  -/
  refine fun _ hs ↦ mem_cocompact.mpr <| (isEmpty_or_nonempty α).casesOn ?_ ?_ <;> intro
    /-
      case refine_1
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTop α
      inst✝ : CompactIccSpace α
      x✝ : Set α
      hs : Membership.mem Filter.atBot x✝
      h✝ : IsEmpty α
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) x✝)
    -/
  · exact ⟨∅, isCompact_empty, fun x _ ↦ (IsEmpty.false x).elim⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTop α
      inst✝ : CompactIccSpace α
      x✝ : Set α
      hs : Membership.mem Filter.atBot x✝
      h✝ : Nonempty α
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) x✝)
    -/
  · obtain ⟨t, ht⟩ := mem_atBot_sets.mp hs
    /-
      case refine_2.intro
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTop α
      inst✝ : CompactIccSpace α
      x✝ : Set α
      hs : Membership.mem Filter.atBot x✝
      h✝ : Nonempty α
      t : α
      ht : ∀ (b : α), LE.le b t → Membership.mem x✝ b
      ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) x✝)
    -/
    refine ⟨Icc t ⊤, isCompact_Icc, fun _ hx ↦ ?_⟩
    /-
      case refine_2.intro
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTop α
      inst✝ : CompactIccSpace α
      x✝¹ : Set α
      hs : Membership.mem Filter.atBot x✝¹
      h✝ : Nonempty α
      t : α
      ht : ∀ (b : α), LE.le b t → Membership.mem x✝¹ b
      x✝ : α
      hx : Membership.mem (HasCompl.compl (Set.Icc t Top.top)) x✝
      ⊢ Membership.mem x✝¹ x✝
    -/
    exact (not_and_or.mp hx).casesOn (fun h ↦ ht _ (le_of_not_le h)) (fun h ↦ (h le_top).elim)
    /-
      🎉 no goals
    -/


theorem cocompact_le_atTop [OrderBot α] [CompactIccSpace α] :
    cocompact α ≤ atTop :=
  cocompact_le_atBot (α := αᵒᵈ)


theorem atBot_le_cocompact [NoMinOrder α] [ClosedIicTopology α] :
    atBot ≤ cocompact α := by
  /-
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : NoMinOrder α
    inst✝ : ClosedIicTopology α
    ⊢ LE.le Filter.atBot (Filter.cocompact α)
  -/
  refine fun s hs ↦ ?_
  /-
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : NoMinOrder α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : Membership.mem (Filter.cocompact α) s
    ⊢ Membership.mem Filter.atBot s
  -/
  obtain ⟨t, ht, hts⟩ := mem_cocompact.mp hs
  /-
    case intro.intro
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : NoMinOrder α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : Membership.mem (Filter.cocompact α) s
    t : Set α
    ht : IsCompact t
    hts : HasSubset.Subset (HasCompl.compl t) s
    ⊢ Membership.mem Filter.atBot s
  -/
  refine (Set.eq_empty_or_nonempty t).casesOn (fun h_empty ↦ ?_) (fun h_nonempty ↦ ?_)
    /-
      case intro.intro.refine_1
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : NoMinOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Membership.mem (Filter.cocompact α) s
      t : Set α
      ht : IsCompact t
      hts : HasSubset.Subset (HasCompl.compl t) s
      h_empty : Eq t EmptyCollection.emptyCollection
      ⊢ Membership.mem Filter.atBot s
    -/
  · rewrite [compl_univ_iff.mpr h_empty, univ_subset_iff] at hts
    /-
      case intro.intro.refine_1
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : NoMinOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Membership.mem (Filter.cocompact α) s
      t : Set α
      ht : IsCompact t
      hts : Eq s Set.univ
      h_empty : Eq t EmptyCollection.emptyCollection
      ⊢ Membership.mem Filter.atBot s
    -/
    convert univ_mem
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : NoMinOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Membership.mem (Filter.cocompact α) s
      t : Set α
      ht : IsCompact t
      hts : HasSubset.Subset (HasCompl.compl t) s
      h_nonempty : t.Nonempty
      ⊢ Membership.mem Filter.atBot s
    -/
  · haveI := h_nonempty.nonempty
    /-
      case intro.intro.refine_2
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : NoMinOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Membership.mem (Filter.cocompact α) s
      t : Set α
      ht : IsCompact t
      hts : HasSubset.Subset (HasCompl.compl t) s
      h_nonempty : t.Nonempty
      this : Nonempty α
      ⊢ Membership.mem Filter.atBot s
    -/
    obtain ⟨a, ha⟩ := ht.exists_isLeast h_nonempty
    /-
      case intro.intro.refine_2.intro
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : NoMinOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Membership.mem (Filter.cocompact α) s
      t : Set α
      ht : IsCompact t
      hts : HasSubset.Subset (HasCompl.compl t) s
      h_nonempty : t.Nonempty
      this : Nonempty α
      a : α
      ha : IsLeast t a
      ⊢ Membership.mem Filter.atBot s
    -/
    obtain ⟨b, hb⟩ := exists_lt a
    exact Filter.mem_atBot_sets.mpr ⟨b, fun b' hb' ↦ hts <| Classical.byContradiction
      fun hc ↦ LT.lt.false <| hb'.trans_lt <| hb.trans_le <| ha.2 (not_not_mem.mp hc)⟩


theorem atTop_le_cocompact [NoMaxOrder α] [ClosedIciTopology α] :
    atTop ≤ cocompact α :=
  atBot_le_cocompact (α := αᵒᵈ)


theorem atBot_atTop_le_cocompact [NoMinOrder α] [NoMaxOrder α]
    [OrderClosedTopology α] : atBot ⊔ atTop ≤ cocompact α :=
  sup_le atBot_le_cocompact atTop_le_cocompact


@[simp 900]
theorem cocompact_eq_atBot_atTop [NoMaxOrder α] [NoMinOrder α]
    [OrderClosedTopology α] [CompactIccSpace α] : cocompact α = atBot ⊔ atTop :=
  cocompact_le_atBot_atTop.antisymm atBot_atTop_le_cocompact


@[simp]
theorem cocompact_eq_atBot [NoMinOrder α] [OrderTop α]
    [ClosedIicTopology α] [CompactIccSpace α] : cocompact α = atBot :=
  cocompact_le_atBot.antisymm atBot_le_cocompact


@[simp]
theorem cocompact_eq_atTop [NoMaxOrder α] [OrderBot α]
    [ClosedIciTopology α] [CompactIccSpace α] : cocompact α = atTop :=
  cocompact_le_atTop.antisymm atTop_le_cocompact


/-- The **extreme value theorem**: a continuous function realizes its minimum on a compact set. -/
theorem IsCompact.exists_isMinOn [ClosedIicTopology α] {s : Set β} (hs : IsCompact s)
    (ne_s : s.Nonempty) {f : β → α} (hf : ContinuousOn f s) : ∃ x ∈ s, IsMinOn f s x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    hs : IsCompact s
    ne_s : s.Nonempty
    f : β → α
    hf : ContinuousOn f s
    ⊢ Exists fun x => And (Membership.mem s x) (IsMinOn f s x)
  -/
  rcases (hs.image_of_continuousOn hf).exists_isLeast (ne_s.image f) with ⟨_, ⟨x, hxs, rfl⟩, hx⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    hs : IsCompact s
    ne_s : s.Nonempty
    f : β → α
    hf : ContinuousOn f s
    x : β
    hxs : Membership.mem s x
    hx : Membership.mem (lowerBounds (Set.image f s)) (f x)
    ⊢ Exists fun x => And (Membership.mem s x) (IsMinOn f s x)
  -/
  refine ⟨x, hxs, forall_mem_image.1 (fun _ hb => hx <| mem_image_of_mem f ?_)⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    hs : IsCompact s
    ne_s : s.Nonempty
    f : β → α
    hf : ContinuousOn f s
    x : β
    hxs : Membership.mem s x
    hx : Membership.mem (lowerBounds (Set.image f s)) (f x)
    x✝ : β
    hb : Membership.mem (Set.image (fun a => a) s) x✝
    ⊢ Membership.mem s x✝
  -/
  rwa [(image_id' s).symm]
  /-
    🎉 no goals
  -/


/-- If a continuous function lies strictly above `a` on a compact set,
  it has a lower bound strictly above `a`. -/
theorem IsCompact.exists_forall_le' [ClosedIicTopology α] [NoMaxOrder α] {f : β → α}
    {s : Set β} (hs : IsCompact s) (hf : ContinuousOn f s) {a : α} (hf' : ∀ b ∈ s, a < f b) :
    ∃ a', a < a' ∧ ∀ b ∈ s, a' ≤ f b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : ClosedIicTopology α
    inst✝ : NoMaxOrder α
    f : β → α
    s : Set β
    hs : IsCompact s
    hf : ContinuousOn f s
    a : α
    hf' : ∀ (b : β), Membership.mem s b → LT.lt a (f b)
    ⊢ Exists fun a' => And (LT.lt a a') (∀ (b : β), Membership.mem s b → LE.le a'  …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hs')
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝⁴ : LinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : ClosedIicTopology α
      inst✝ : NoMaxOrder α
      f : β → α
      a : α
      hs : IsCompact EmptyCollection.emptyCollection
      hf : ContinuousOn f EmptyCollection.emptyCollection
      hf' : ∀ (b : β), Membership.mem EmptyCollection.emptyCollection b → LT.lt a (f …
      ⊢ Exists fun a' => And (LT.lt a a') (∀ (b : β), Membership.mem EmptyCollection …
    -/
  · obtain ⟨a', ha'⟩ := exists_gt a
    /-
      case inl.intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : LinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : ClosedIicTopology α
      inst✝ : NoMaxOrder α
      f : β → α
      a : α
      hs : IsCompact EmptyCollection.emptyCollection
      hf : ContinuousOn f EmptyCollection.emptyCollection
      hf' : ∀ (b : β), Membership.mem EmptyCollection.emptyCollection b → LT.lt a (f …
      a' : α
      ha' : LT.lt a a'
      ⊢ Exists fun a' => And (LT.lt a a') (∀ (b : β), Membership.mem EmptyCollection …
    -/
    exact ⟨a', ha', fun _ a ↦ a.elim⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝⁴ : LinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : ClosedIicTopology α
      inst✝ : NoMaxOrder α
      f : β → α
      s : Set β
      hs : IsCompact s
      hf : ContinuousOn f s
      a : α
      hf' : ∀ (b : β), Membership.mem s b → LT.lt a (f b)
      hs' : s.Nonempty
      ⊢ Exists fun a' => And (LT.lt a a') (∀ (b : β), Membership.mem s b → LE.le a'  …
    -/
  · obtain ⟨x, hx, hx'⟩ := hs.exists_isMinOn hs' hf
    /-
      case inr.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : LinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : ClosedIicTopology α
      inst✝ : NoMaxOrder α
      f : β → α
      s : Set β
      hs : IsCompact s
      hf : ContinuousOn f s
      a : α
      hf' : ∀ (b : β), Membership.mem s b → LT.lt a (f b)
      hs' : s.Nonempty
      x : β
      hx : Membership.mem s x
      hx' : IsMinOn f s x
      ⊢ Exists fun a' => And (LT.lt a a') (∀ (b : β), Membership.mem s b → LE.le a'  …
    -/
    exact ⟨f x, hf' x hx, hx'⟩
    /-
      🎉 no goals
    -/


/-- The **extreme value theorem**: a continuous function realizes its maximum on a compact set. -/
theorem IsCompact.exists_isMaxOn [ClosedIciTopology α] {s : Set β} (hs : IsCompact s)
    (ne_s : s.Nonempty) {f : β → α} (hf : ContinuousOn f s) : ∃ x ∈ s, IsMaxOn f s x :=
  IsCompact.exists_isMinOn (α := αᵒᵈ) hs ne_s hf


/-- The **extreme value theorem**: if a function `f` is continuous on a closed set `s` and it is
larger than a value in its image away from compact sets, then it has a minimum on this set. -/
theorem ContinuousOn.exists_isMinOn' [ClosedIicTopology α] {s : Set β} {f : β → α}
    (hf : ContinuousOn f s) (hsc : IsClosed s) {x₀ : β} (h₀ : x₀ ∈ s)
    (hc : ∀ᶠ x in cocompact β ⊓ 𝓟 s, f x₀ ≤ f x) : ∃ x ∈ s, IsMinOn f s x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    f : β → α
    hf : ContinuousOn f s
    hsc : IsClosed s
    x₀ : β
    h₀ : Membership.mem s x₀
    hc : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompac …
    ⊢ Exists fun x => And (Membership.mem s x) (IsMinOn f s x)
  -/
  rcases (hasBasis_cocompact.inf_principal _).eventually_iff.1 hc with ⟨K, hK, hKf⟩
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    f : β → α
    hf : ContinuousOn f s
    hsc : IsClosed s
    x₀ : β
    h₀ : Membership.mem s x₀
    hc : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompac …
    K : Set β
    hK : IsCompact K
    hKf : ∀ ⦃x : β⦄, Membership.mem (Inter.inter (HasCompl.compl K) s) x → LE.le ( …
    ⊢ Exists fun x => And (Membership.mem s x) (IsMinOn f s x)
  -/
  have hsub : insert x₀ (K ∩ s) ⊆ s := insert_subset_iff.2 ⟨h₀, inter_subset_right⟩
  obtain ⟨x, hx, hxf⟩ : ∃ x ∈ insert x₀ (K ∩ s), ∀ y ∈ insert x₀ (K ∩ s), f x ≤ f y :=
    ((hK.inter_right hsc).insert x₀).exists_isMinOn (insert_nonempty _ _) (hf.mono hsub)
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    f : β → α
    hf : ContinuousOn f s
    hsc : IsClosed s
    x₀ : β
    h₀ : Membership.mem s x₀
    hc : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompac …
    K : Set β
    hK : IsCompact K
    hKf : ∀ ⦃x : β⦄, Membership.mem (Inter.inter (HasCompl.compl K) s) x → LE.le ( …
    hsub : HasSubset.Subset (Insert.insert x₀ (Inter.inter K s)) s
    x : β
    hx : Membership.mem (Insert.insert x₀ (Inter.inter K s)) x
    hxf : ∀ (y : β), Membership.mem (Insert.insert x₀ (Inter.inter K s)) y → LE.le …
    ⊢ Exists fun x => And (Membership.mem s x) (IsMinOn f s x)
  -/
  refine ⟨x, hsub hx, fun y hy => ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    f : β → α
    hf : ContinuousOn f s
    hsc : IsClosed s
    x₀ : β
    h₀ : Membership.mem s x₀
    hc : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompac …
    K : Set β
    hK : IsCompact K
    hKf : ∀ ⦃x : β⦄, Membership.mem (Inter.inter (HasCompl.compl K) s) x → LE.le ( …
    hsub : HasSubset.Subset (Insert.insert x₀ (Inter.inter K s)) s
    x : β
    hx : Membership.mem (Insert.insert x₀ (Inter.inter K s)) x
    hxf : ∀ (y : β), Membership.mem (Insert.insert x₀ (Inter.inter K s)) y → LE.le …
    y : β
    hy : Membership.mem s y
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (f x) (f x_2)) x_1) y
  -/
  by_cases hyK : y ∈ K
  /-
    case pos
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIicTopology α
    s : Set β
    f : β → α
    hf : ContinuousOn f s
    hsc : IsClosed s
    x₀ : β
    h₀ : Membership.mem s x₀
    hc : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompac …
    K : Set β
    hK : IsCompact K
    hKf : ∀ ⦃x : β⦄, Membership.mem (Inter.inter (HasCompl.compl K) s) x → LE.le ( …
    hsub : HasSubset.Subset (Insert.insert x₀ (Inter.inter K s)) s
    x : β
    hx : Membership.mem (Insert.insert x₀ (Inter.inter K s)) x
    hxf : ∀ (y : β), Membership.mem (Insert.insert x₀ (Inter.inter K s)) y → LE.le …
    y : β
    hy : Membership.mem s y
    hyK : Membership.mem K y
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (f x) (f x_2)) x_1) y
  -/
  exacts [hxf _ (Or.inr ⟨hyK, hy⟩), (hxf _ (Or.inl rfl)).trans (hKf ⟨hyK, hy⟩)]
  /-
    🎉 no goals
  -/


/-- The **extreme value theorem**: if a function `f` is continuous on a closed set `s` and it is
smaller than a value in its image away from compact sets, then it has a maximum on this set. -/
theorem ContinuousOn.exists_isMaxOn' [ClosedIciTopology α] {s : Set β} {f : β → α}
    (hf : ContinuousOn f s) (hsc : IsClosed s) {x₀ : β} (h₀ : x₀ ∈ s)
    (hc : ∀ᶠ x in cocompact β ⊓ 𝓟 s, f x ≤ f x₀) : ∃ x ∈ s, IsMaxOn f s x :=
  ContinuousOn.exists_isMinOn' (α := αᵒᵈ) hf hsc h₀ hc


/-- The **extreme value theorem**: if a continuous function `f` is larger than a value in its range
away from compact sets, then it has a global minimum. -/
theorem Continuous.exists_forall_le' [ClosedIicTopology α] {f : β → α} (hf : Continuous f)
    (x₀ : β) (h : ∀ᶠ x in cocompact β, f x₀ ≤ f x) : ∃ x : β, ∀ y : β, f x ≤ f y :=
  let ⟨x, _, hx⟩ := hf.continuousOn.exists_isMinOn' isClosed_univ (mem_univ x₀)
        /-
          α : Type u_2
          β : Type u_3
          inst✝³ : LinearOrder α
          inst✝² : TopologicalSpace α
          inst✝¹ : TopologicalSpace β
          inst✝ : ClosedIicTopology α
          f : β → α
          hf : Continuous f
          x₀ : β
          h : Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Filter.cocompact β)
          ⊢ Filter.Eventually (fun x => LE.le (f x₀) (f x)) (Min.min (Filter.cocompact β …
        -/
    (by rwa [principal_univ, inf_top_eq])
        /-
          🎉 no goals
        -/
  ⟨x, fun y => hx (mem_univ y)⟩


/-- The **extreme value theorem**: if a continuous function `f` is smaller than a value in its range
away from compact sets, then it has a global maximum. -/
theorem Continuous.exists_forall_ge' [ClosedIciTopology α] {f : β → α} (hf : Continuous f)
    (x₀ : β) (h : ∀ᶠ x in cocompact β, f x ≤ f x₀) : ∃ x : β, ∀ y : β, f y ≤ f x :=
  Continuous.exists_forall_le' (α := αᵒᵈ) hf x₀ h


/-- The **extreme value theorem**: if a continuous function `f` tends to infinity away from compact
sets, then it has a global minimum. -/
theorem Continuous.exists_forall_le [ClosedIicTopology α] [Nonempty β] {f : β → α}
    (hf : Continuous f) (hlim : Tendsto f (cocompact β) atTop) : ∃ x, ∀ y, f x ≤ f y := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : ClosedIicTopology α
    inst✝ : Nonempty β
    f : β → α
    hf : Continuous f
    hlim : Filter.Tendsto f (Filter.cocompact β) Filter.atTop
    ⊢ Exists fun x => ∀ (y : β), LE.le (f x) (f y)
  -/
  inhabit β
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : ClosedIicTopology α
    inst✝ : Nonempty β
    f : β → α
    hf : Continuous f
    hlim : Filter.Tendsto f (Filter.cocompact β) Filter.atTop
    inhabited_h : Inhabited β
    ⊢ Exists fun x => ∀ (y : β), LE.le (f x) (f y)
  -/
  exact hf.exists_forall_le' default (hlim.eventually <| eventually_ge_atTop _)
  /-
    🎉 no goals
  -/


/-- The **extreme value theorem**: if a continuous function `f` tends to negative infinity away from
compact sets, then it has a global maximum. -/
theorem Continuous.exists_forall_ge [ClosedIciTopology α] [Nonempty β] {f : β → α}
    (hf : Continuous f) (hlim : Tendsto f (cocompact β) atBot) : ∃ x, ∀ y, f y ≤ f x :=
  Continuous.exists_forall_le (α := αᵒᵈ) hf hlim


/-- A continuous function with compact support has a global minimum. -/
@[to_additive "A continuous function with compact support has a global minimum."]
theorem Continuous.exists_forall_le_of_hasCompactMulSupport [ClosedIicTopology α] [Nonempty β]
    [One α] {f : β → α} (hf : Continuous f) (h : HasCompactMulSupport f) :
    ∃ x : β, ∀ y : β, f x ≤ f y := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : ClosedIicTopology α
    inst✝¹ : Nonempty β
    inst✝ : One α
    f : β → α
    hf : Continuous f
    h : HasCompactMulSupport f
    ⊢ Exists fun x => ∀ (y : β), LE.le (f x) (f y)
  -/
  obtain ⟨_, ⟨x, rfl⟩, hx⟩ := (h.isCompact_range hf).exists_isLeast (range_nonempty _)
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : ClosedIicTopology α
    inst✝¹ : Nonempty β
    inst✝ : One α
    f : β → α
    hf : Continuous f
    h : HasCompactMulSupport f
    x : β
    hx : Membership.mem (lowerBounds (Set.range f)) (f x)
    ⊢ Exists fun x => ∀ (y : β), LE.le (f x) (f y)
  -/
  rw [mem_lowerBounds, forall_mem_range] at hx
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : ClosedIicTopology α
    inst✝¹ : Nonempty β
    inst✝ : One α
    f : β → α
    hf : Continuous f
    h : HasCompactMulSupport f
    x : β
    hx : ∀ (i : β), LE.le (f x) (f i)
    ⊢ Exists fun x => ∀ (y : β), LE.le (f x) (f y)
  -/
  exact ⟨x, hx⟩
  /-
    🎉 no goals
  -/


/-- A continuous function with compact support has a global maximum. -/
@[to_additive "A continuous function with compact support has a global maximum."]
theorem Continuous.exists_forall_ge_of_hasCompactMulSupport [ClosedIciTopology α] [Nonempty β]
    [One α] {f : β → α} (hf : Continuous f) (h : HasCompactMulSupport f) :
    ∃ x : β, ∀ y : β, f y ≤ f x :=
  Continuous.exists_forall_le_of_hasCompactMulSupport (α := αᵒᵈ) hf h


/-- A compact set is bounded below -/
theorem IsCompact.bddBelow [ClosedIicTopology α] [Nonempty α] {s : Set α} (hs : IsCompact s) :
    BddBelow s := by
  /-
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : ClosedIicTopology α
    inst✝ : Nonempty α
    s : Set α
    hs : IsCompact s
    ⊢ BddBelow s
  -/
  rcases s.eq_empty_or_nonempty with rfl | hne
    /-
      case inl
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : ClosedIicTopology α
      inst✝ : Nonempty α
      hs : IsCompact EmptyCollection.emptyCollection
      ⊢ BddBelow EmptyCollection.emptyCollection
    -/
  · exact bddBelow_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : ClosedIicTopology α
      inst✝ : Nonempty α
      s : Set α
      hs : IsCompact s
      hne : s.Nonempty
      ⊢ BddBelow s
    -/
  · obtain ⟨a, -, has⟩ := hs.exists_isLeast hne
    /-
      case inr.intro.intro
      α : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : ClosedIicTopology α
      inst✝ : Nonempty α
      s : Set α
      hs : IsCompact s
      hne : s.Nonempty
      a : α
      has : Membership.mem (lowerBounds s) a
      ⊢ BddBelow s
    -/
    exact ⟨a, has⟩
    /-
      🎉 no goals
    -/


/-- A compact set is bounded above -/
theorem IsCompact.bddAbove [ClosedIciTopology α] [Nonempty α] {s : Set α} (hs : IsCompact s) :
    BddAbove s :=
  IsCompact.bddBelow (α := αᵒᵈ) hs


/-- A continuous function is bounded below on a compact set. -/
theorem IsCompact.bddBelow_image [ClosedIicTopology α] [Nonempty α] {f : β → α} {K : Set β}
    (hK : IsCompact K) (hf : ContinuousOn f K) : BddBelow (f '' K) :=
  (hK.image_of_continuousOn hf).bddBelow


/-- A continuous function is bounded above on a compact set. -/
theorem IsCompact.bddAbove_image [ClosedIciTopology α] [Nonempty α] {f : β → α} {K : Set β}
    (hK : IsCompact K) (hf : ContinuousOn f K) : BddAbove (f '' K) :=
  IsCompact.bddBelow_image (α := αᵒᵈ) hK hf


/-- A continuous function with compact support is bounded below. -/
@[to_additive " A continuous function with compact support is bounded below. "]
theorem Continuous.bddBelow_range_of_hasCompactMulSupport [ClosedIicTopology α] [One α]
    {f : β → α} (hf : Continuous f) (h : HasCompactMulSupport f) : BddBelow (range f) :=
  (h.isCompact_range hf).bddBelow


/-- A continuous function with compact support is bounded above. -/
@[to_additive " A continuous function with compact support is bounded above. "]
theorem Continuous.bddAbove_range_of_hasCompactMulSupport [ClosedIciTopology α] [One α]
    {f : β → α} (hf : Continuous f) (h : HasCompactMulSupport f) : BddAbove (range f) :=
  Continuous.bddBelow_range_of_hasCompactMulSupport (α := αᵒᵈ) hf h


theorem IsCompact.sSup_lt_iff_of_continuous [ClosedIciTopology α] {f : β → α} {K : Set β}
    (hK : IsCompact K) (h0K : K.Nonempty) (hf : ContinuousOn f K) (y : α) :
    sSup (f '' K) < y ↔ ∀ x ∈ K, f x < y := by
  refine ⟨fun h x hx => (le_csSup (hK.bddAbove_image hf) <| mem_image_of_mem f hx).trans_lt h,
    fun h => ?_⟩
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIciTopology α
    f : β → α
    K : Set β
    hK : IsCompact K
    h0K : K.Nonempty
    hf : ContinuousOn f K
    y : α
    h : ∀ (x : β), Membership.mem K x → LT.lt (f x) y
    ⊢ LT.lt (SupSet.sSup (Set.image f K)) y
  -/
  obtain ⟨x, hx, h2x⟩ := hK.exists_isMaxOn h0K hf
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIciTopology α
    f : β → α
    K : Set β
    hK : IsCompact K
    h0K : K.Nonempty
    hf : ContinuousOn f K
    y : α
    h : ∀ (x : β), Membership.mem K x → LT.lt (f x) y
    x : β
    hx : Membership.mem K x
    h2x : IsMaxOn f K x
    ⊢ LT.lt (SupSet.sSup (Set.image f K)) y
  -/
  refine (csSup_le (h0K.image f) ?_).trans_lt (h x hx)
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : ClosedIciTopology α
    f : β → α
    K : Set β
    hK : IsCompact K
    h0K : K.Nonempty
    hf : ContinuousOn f K
    y : α
    h : ∀ (x : β), Membership.mem K x → LT.lt (f x) y
    x : β
    hx : Membership.mem K x
    h2x : IsMaxOn f K x
    ⊢ ∀ (b : α), Membership.mem (Set.image f K) b → LE.le b (f x)
  -/
  rintro _ ⟨x', hx', rfl⟩; exact h2x hx'
                           /-
                             🎉 no goals
                           -/


theorem IsCompact.lt_sInf_iff_of_continuous [ClosedIicTopology α] {f : β → α} {K : Set β}
    (hK : IsCompact K) (h0K : K.Nonempty) (hf : ContinuousOn f K) (y : α) :
    y < sInf (f '' K) ↔ ∀ x ∈ K, y < f x :=
  IsCompact.sSup_lt_iff_of_continuous (α := αᵒᵈ) hK h0K hf y


theorem IsCompact.sInf_mem [ClosedIicTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : sInf s ∈ s :=
  let ⟨_a, ha⟩ := hs.exists_isLeast ne_s
  ha.csInf_mem


theorem IsCompact.sSup_mem [ClosedIciTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : sSup s ∈ s :=
  IsCompact.sInf_mem (α := αᵒᵈ) hs ne_s


theorem IsCompact.isGLB_sInf [ClosedIicTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : IsGLB s (sInf s) :=
  isGLB_csInf ne_s hs.bddBelow


theorem IsCompact.isLUB_sSup [ClosedIciTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : IsLUB s (sSup s) :=
  IsCompact.isGLB_sInf (α := αᵒᵈ) hs ne_s


theorem IsCompact.isLeast_sInf [ClosedIicTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : IsLeast s (sInf s) :=
  ⟨hs.sInf_mem ne_s, (hs.isGLB_sInf ne_s).1⟩


theorem IsCompact.isGreatest_sSup [ClosedIciTopology α] {s : Set α} (hs : IsCompact s)
    (ne_s : s.Nonempty) : IsGreatest s (sSup s) :=
  IsCompact.isLeast_sInf (α := αᵒᵈ) hs ne_s


theorem IsCompact.exists_sInf_image_eq_and_le [ClosedIicTopology α] {s : Set β}
    (hs : IsCompact s) (ne_s : s.Nonempty) {f : β → α} (hf : ContinuousOn f s) :
    ∃ x ∈ s, sInf (f '' s) = f x ∧ ∀ y ∈ s, f x ≤ f y :=
  let ⟨x, hxs, hx⟩ := (hs.image_of_continuousOn hf).sInf_mem (ne_s.image f)
  ⟨x, hxs, hx.symm, fun _y hy =>
    hx.trans_le <| csInf_le (hs.image_of_continuousOn hf).bddBelow <| mem_image_of_mem f hy⟩


theorem IsCompact.exists_sSup_image_eq_and_ge [ClosedIciTopology α] {s : Set β}
    (hs : IsCompact s) (ne_s : s.Nonempty) {f : β → α} (hf : ContinuousOn f s) :
    ∃ x ∈ s, sSup (f '' s) = f x ∧ ∀ y ∈ s, f y ≤ f x :=
  IsCompact.exists_sInf_image_eq_and_le (α := αᵒᵈ) hs ne_s hf


theorem IsCompact.exists_sInf_image_eq [ClosedIicTopology α] {s : Set β} (hs : IsCompact s)
    (ne_s : s.Nonempty) {f : β → α} (hf : ContinuousOn f s) : ∃ x ∈ s, sInf (f '' s) = f x :=
  let ⟨x, hxs, hx, _⟩ := hs.exists_sInf_image_eq_and_le ne_s hf
  ⟨x, hxs, hx⟩


theorem IsCompact.exists_sSup_image_eq [ClosedIciTopology α] {s : Set β} (hs : IsCompact s)
    (ne_s : s.Nonempty) : ∀ {f : β → α}, ContinuousOn f s → ∃ x ∈ s, sSup (f '' s) = f x :=
  IsCompact.exists_sInf_image_eq (α := αᵒᵈ) hs ne_s


theorem IsCompact.exists_isMinOn_mem_subset [ClosedIicTopology α] {f : β → α} {s t : Set β}
    {z : β} (ht : IsCompact t) (hf : ContinuousOn f t) (hz : z ∈ t)
    (hfz : ∀ z' ∈ t \ s, f z < f z') : ∃ x ∈ s, IsMinOn f t x :=
  let ⟨x, hxt, hfx⟩ := ht.exists_isMinOn ⟨z, hz⟩ hf
  ⟨x, by_contra fun hxs => (hfz x ⟨hxt, hxs⟩).not_le (hfx hz), hfx⟩


theorem IsCompact.exists_isMaxOn_mem_subset [ClosedIciTopology α] {f : β → α} {s t : Set β}
    {z : β} (ht : IsCompact t) (hf : ContinuousOn f t) (hz : z ∈ t)
    (hfz : ∀ z' ∈ t \ s, f z' < f z) : ∃ x ∈ s, IsMaxOn f t x :=
  let ⟨x, hxt, hfx⟩ := ht.exists_isMaxOn ⟨z, hz⟩ hf
  ⟨x, by_contra fun hxs => (hfz x ⟨hxt, hxs⟩).not_le (hfx hz), hfx⟩

-- Porting note: rfc: assume `t ∈ 𝓝ˢ s` (a.k.a. `s ⊆ interior t`) instead of `s ⊆ t` and
-- `IsOpen s`?

theorem IsCompact.exists_isLocalMin_mem_open [ClosedIicTopology α] {f : β → α} {s t : Set β}
    {z : β} (ht : IsCompact t) (hst : s ⊆ t) (hf : ContinuousOn f t) (hz : z ∈ t)
    (hfz : ∀ z' ∈ t \ s, f z < f z') (hs : IsOpen s) : ∃ x ∈ s, IsLocalMin f x :=
  let ⟨x, hxs, h⟩ := ht.exists_isMinOn_mem_subset hf hz hfz
  ⟨x, hxs, h.isLocalMin <| mem_nhds_iff.2 ⟨s, hst, hs, hxs⟩⟩


theorem IsCompact.exists_isLocalMax_mem_open [ClosedIciTopology α] {f : β → α} {s t : Set β}
    {z : β} (ht : IsCompact t) (hst : s ⊆ t) (hf : ContinuousOn f t) (hz : z ∈ t)
    (hfz : ∀ z' ∈ t \ s, f z' < f z) (hs : IsOpen s) : ∃ x ∈ s, IsLocalMax f x :=
  let ⟨x, hxs, h⟩ := ht.exists_isMaxOn_mem_subset hf hz hfz
  ⟨x, hxs, h.isLocalMax <| mem_nhds_iff.2 ⟨s, hst, hs, hxs⟩⟩


theorem eq_Icc_of_connected_compact {s : Set α} (h₁ : IsConnected s) (h₂ : IsCompact s) :
    s = Icc (sInf s) (sSup s) :=
  eq_Icc_csInf_csSup_of_connected_bdd_closed h₁ h₂.bddBelow h₂.bddAbove h₂.isClosed


/-- If `f : γ → β → α` is a function that is continuous as a function on `γ × β`, `α` is a
conditionally complete linear order, and `K : Set β` is a compact set, then
`fun x ↦ sSup (f x '' K)` is a continuous function. -/
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: generalize. The following version seems to be true:
```
theorem IsCompact.tendsto_sSup {f : γ → β → α} {g : β → α} {K : Set β} {l : Filter γ}
    (hK : IsCompact K) (hf : ∀ y ∈ K, Tendsto ↿f (l ×ˢ 𝓝[K] y) (𝓝 (g y)))
    (hgc : ContinuousOn g K) :
    Tendsto (fun x => sSup (f x '' K)) l (𝓝 (sSup (g '' K))) := _
```
Moreover, it seems that `hgc` follows from `hf` (Yury Kudryashov). -/
theorem IsCompact.continuous_sSup {f : γ → β → α} {K : Set β} (hK : IsCompact K)
    (hf : Continuous ↿f) : Continuous fun x => sSup (f x '' K) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : γ → β → α
    K : Set β
    hK : IsCompact K
    hf : Continuous (Function.HasUncurry.uncurry f)
    ⊢ Continuous fun x => SupSet.sSup (Set.image (f x) K)
  -/
  rcases eq_empty_or_nonempty K with (rfl | h0K)
    /-
      case inl
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : γ → β → α
      hf : Continuous (Function.HasUncurry.uncurry f)
      hK : IsCompact EmptyCollection.emptyCollection
      ⊢ Continuous fun x => SupSet.sSup (Set.image (f x) EmptyCollection.emptyCollec …
    -/
  · simp_rw [image_empty]
    /-
      case inl
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : γ → β → α
      hf : Continuous (Function.HasUncurry.uncurry f)
      hK : IsCompact EmptyCollection.emptyCollection
      ⊢ Continuous fun x => SupSet.sSup EmptyCollection.emptyCollection
    -/
    exact continuous_const
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : γ → β → α
    K : Set β
    hK : IsCompact K
    hf : Continuous (Function.HasUncurry.uncurry f)
    h0K : K.Nonempty
    ⊢ Continuous fun x => SupSet.sSup (Set.image (f x) K)
  -/
  rw [continuous_iff_continuousAt]
  /-
    case inr
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : γ → β → α
    K : Set β
    hK : IsCompact K
    hf : Continuous (Function.HasUncurry.uncurry f)
    h0K : K.Nonempty
    ⊢ ∀ (x : γ), ContinuousAt (fun x => SupSet.sSup (Set.image (f x) K)) x
  -/
  intro x
  obtain ⟨y, hyK, h2y, hy⟩ :=
    hK.exists_sSup_image_eq_and_ge h0K
      (show Continuous fun y => f x y from hf.comp <| Continuous.Prod.mk x).continuousOn
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : γ → β → α
    K : Set β
    hK : IsCompact K
    hf : Continuous (Function.HasUncurry.uncurry f)
    h0K : K.Nonempty
    x : γ
    y : β
    hyK : Membership.mem K y
    h2y : Eq (SupSet.sSup (Set.image (fun y => f x y) K)) (f x y)
    hy : ∀ (y_1 : β), Membership.mem K y_1 → LE.le (f x y_1) (f x y)
    ⊢ ContinuousAt (fun x => SupSet.sSup (Set.image (f x) K)) x
  -/
  rw [ContinuousAt, h2y, tendsto_order]
  have := tendsto_order.mp ((show Continuous fun x => f x y
    from hf.comp <| continuous_id.prod_mk continuous_const).tendsto x)
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : γ → β → α
    K : Set β
    hK : IsCompact K
    hf : Continuous (Function.HasUncurry.uncurry f)
    h0K : K.Nonempty
    x : γ
    y : β
    hyK : Membership.mem K y
    h2y : Eq (SupSet.sSup (Set.image (fun y => f x y) K)) (f x y)
    hy : ∀ (y_1 : β), Membership.mem K y_1 → LE.le (f x y_1) (f x y)
    this : And (∀ (a' : α), LT.lt a' (f x y) → Filter.Eventually (fun b => LT.lt a …
    ⊢ And (∀ (a' : α), LT.lt a' (f x y) → Filter.Eventually (fun b => LT.lt a' (Su …
  -/
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · refine (this.1 z hz).mono fun x' hx' =>
      hx'.trans_le <| le_csSup ?_ <| mem_image_of_mem (f x') hyK
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : γ → β → α
      K : Set β
      hK : IsCompact K
      hf : Continuous (Function.HasUncurry.uncurry f)
      h0K : K.Nonempty
      x : γ
      y : β
      hyK : Membership.mem K y
      h2y : Eq (SupSet.sSup (Set.image (fun y => f x y) K)) (f x y)
      hy : ∀ (y_1 : β), Membership.mem K y_1 → LE.le (f x y_1) (f x y)
      this : And (∀ (a' : α), LT.lt a' (f x y) → Filter.Eventually (fun b => LT.lt a …
      z : α
      hz : LT.lt z (f x y)
      x' : γ
      hx' : LT.lt z (f x' y)
      ⊢ BddAbove (Set.image (f x') K)
    -/
    exact hK.bddAbove_image (hf.comp <| Continuous.Prod.mk x').continuousOn
    /-
      🎉 no goals
    -/
  · have h : ({x} : Set γ) ×ˢ K ⊆ ↿f ⁻¹' Iio z := by
      rintro ⟨x', y'⟩ ⟨(rfl : x' = x), hy'⟩
      exact (hy y' hy').trans_lt hz
    obtain ⟨u, v, hu, _, hxu, hKv, huv⟩ :=
      generalized_tube_lemma isCompact_singleton hK (isOpen_Iio.preimage hf) h
    /-
      case inr.intro.intro.intro.refine_2.intro.intro.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : γ → β → α
      K : Set β
      hK : IsCompact K
      hf : Continuous (Function.HasUncurry.uncurry f)
      h0K : K.Nonempty
      x : γ
      y : β
      hyK : Membership.mem K y
      h2y : Eq (SupSet.sSup (Set.image (fun y => f x y) K)) (f x y)
      hy : ∀ (y_1 : β), Membership.mem K y_1 → LE.le (f x y_1) (f x y)
      this : And (∀ (a' : α), LT.lt a' (f x y) → Filter.Eventually (fun b => LT.lt a …
      z : α
      hz : GT.gt z (f x y)
      h : HasSubset.Subset (SProd.sprod (Singleton.singleton x) K) (Set.preimage (Fu …
      u : Set γ
      v : Set β
      hu : IsOpen u
      left✝ : IsOpen v
      hxu : HasSubset.Subset (Singleton.singleton x) u
      hKv : HasSubset.Subset K v
      huv : HasSubset.Subset (SProd.sprod u v) (Set.preimage (Function.HasUncurry.un …
      ⊢ Filter.Eventually (fun b => LT.lt (SupSet.sSup (Set.image (f b) K)) z) (nhds …
    -/
    refine eventually_of_mem (hu.mem_nhds (singleton_subset_iff.mp hxu)) fun x' hx' => ?_
    rw [hK.sSup_lt_iff_of_continuous h0K
        (show Continuous (f x') from hf.comp <| Continuous.Prod.mk x').continuousOn]
    /-
      case inr.intro.intro.intro.refine_2.intro.intro.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : γ → β → α
      K : Set β
      hK : IsCompact K
      hf : Continuous (Function.HasUncurry.uncurry f)
      h0K : K.Nonempty
      x : γ
      y : β
      hyK : Membership.mem K y
      h2y : Eq (SupSet.sSup (Set.image (fun y => f x y) K)) (f x y)
      hy : ∀ (y_1 : β), Membership.mem K y_1 → LE.le (f x y_1) (f x y)
      this : And (∀ (a' : α), LT.lt a' (f x y) → Filter.Eventually (fun b => LT.lt a …
      z : α
      hz : GT.gt z (f x y)
      h : HasSubset.Subset (SProd.sprod (Singleton.singleton x) K) (Set.preimage (Fu …
      u : Set γ
      v : Set β
      hu : IsOpen u
      left✝ : IsOpen v
      hxu : HasSubset.Subset (Singleton.singleton x) u
      hKv : HasSubset.Subset K v
      huv : HasSubset.Subset (SProd.sprod u v) (Set.preimage (Function.HasUncurry.un …
      x' : γ
      hx' : Membership.mem u x'
      ⊢ ∀ (x : β), Membership.mem K x → LT.lt (f x' x) z
    -/
    exact fun y' hy' => huv (mk_mem_prod hx' (hKv hy'))
    /-
      🎉 no goals
    -/


theorem IsCompact.continuous_sInf {f : γ → β → α} {K : Set β} (hK : IsCompact K)
    (hf : Continuous ↿f) : Continuous fun x => sInf (f x '' K) :=
  IsCompact.continuous_sSup (α := αᵒᵈ) hK hf


theorem image_Icc (hab : a ≤ b) (h : ContinuousOn f <| Icc a b) :
    f '' Icc a b = Icc (sInf <| f '' Icc a b) (sSup <| f '' Icc a b) :=
  eq_Icc_of_connected_compact ⟨(nonempty_Icc.2 hab).image f, isPreconnected_Icc.image f h⟩
    (isCompact_Icc.image_of_continuousOn h)


theorem image_uIcc_eq_Icc (h : ContinuousOn f [[a, b]]) :
    f '' [[a, b]] = Icc (sInf (f '' [[a, b]])) (sSup (f '' [[a, b]])) :=
  image_Icc min_le_max h


theorem image_uIcc (h : ContinuousOn f <| [[a, b]]) :
    f '' [[a, b]] = [[sInf (f '' [[a, b]]), sSup (f '' [[a, b]])]] := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b : α
    h : ContinuousOn f (Set.uIcc a b)
    ⊢ Eq (Set.image f (Set.uIcc a b)) (Set.uIcc (InfSet.sInf (Set.image f (Set.uIc …
  -/
  refine h.image_uIcc_eq_Icc.trans (uIcc_of_le ?_).symm
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b : α
    h : ContinuousOn f (Set.uIcc a b)
    ⊢ LE.le (InfSet.sInf (Set.image f (Set.uIcc a b))) (SupSet.sSup (Set.image f ( …
  -/
  refine csInf_le_csSup ?_ ?_ (nonempty_uIcc.image _) <;> rw [h.image_uIcc_eq_Icc]
  /-
    case refine_1
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b : α
    h : ContinuousOn f (Set.uIcc a b)
    ⊢ BddBelow (Set.Icc (InfSet.sInf (Set.image f (Set.uIcc a b))) (SupSet.sSup (S …
  -/
  exacts [bddBelow_Icc, bddAbove_Icc]
  /-
    🎉 no goals
  -/


theorem sInf_image_Icc_le (h : ContinuousOn f <| Icc a b) (hc : c ∈ Icc a b) :
    sInf (f '' Icc a b) ≤ f c := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    ⊢ LE.le (InfSet.sInf (Set.image f (Set.Icc a b))) (f c)
  -/
  have := mem_image_of_mem f hc
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    this : Membership.mem (Set.image f (Set.Icc a b)) (f c)
    ⊢ LE.le (InfSet.sInf (Set.image f (Set.Icc a b))) (f c)
  -/
  rw [h.image_Icc (hc.1.trans hc.2)] at this
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    this : Membership.mem (Set.Icc (InfSet.sInf (Set.image f (Set.Icc a b))) (SupS …
    ⊢ LE.le (InfSet.sInf (Set.image f (Set.Icc a b))) (f c)
  -/
  exact this.1
  /-
    🎉 no goals
  -/


theorem le_sSup_image_Icc (h : ContinuousOn f <| Icc a b) (hc : c ∈ Icc a b) :
    f c ≤ sSup (f '' Icc a b) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    ⊢ LE.le (f c) (SupSet.sSup (Set.image f (Set.Icc a b)))
  -/
  have := mem_image_of_mem f hc
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    this : Membership.mem (Set.image f (Set.Icc a b)) (f c)
    ⊢ LE.le (f c) (SupSet.sSup (Set.image f (Set.Icc a b)))
  -/
  rw [h.image_Icc (hc.1.trans hc.2)] at this
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : TopologicalSpace β
    inst✝² : DenselyOrdered α
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a b c : α
    h : ContinuousOn f (Set.Icc a b)
    hc : Membership.mem (Set.Icc a b) c
    this : Membership.mem (Set.Icc (InfSet.sInf (Set.image f (Set.Icc a b))) (SupS …
    ⊢ LE.le (f c) (SupSet.sSup (Set.image f (Set.Icc a b)))
  -/
  exact this.2
  /-
    🎉 no goals
  -/


