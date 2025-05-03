/-- Order-connected component of a point `x` in a set `s`. It is defined as the set of `y` such that
`Set.uIcc x y ⊆ s`. Note that it is empty if and only if `x ∉ s`. -/
def ordConnectedComponent (s : Set α) (x : α) : Set α :=
  { y | [[x, y]] ⊆ s }


theorem mem_ordConnectedComponent : y ∈ ordConnectedComponent s x ↔ [[x, y]] ⊆ s :=
  Iff.rfl


theorem dual_ordConnectedComponent :
    ordConnectedComponent (ofDual ⁻¹' s) (toDual x) = ofDual ⁻¹' ordConnectedComponent s x :=
  ext <| (Surjective.forall toDual.surjective).2 fun x => by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      s : Set α
      x✝ x : α
      ⊢ Iff (Membership.mem ((Set.preimage (⇑OrderDual.ofDual) s).ordConnectedCompon …
    -/
    rw [mem_ordConnectedComponent, dual_uIcc]
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      s : Set α
      x✝ x : α
      ⊢ Iff (HasSubset.Subset (Set.preimage (⇑OrderDual.ofDual) (Set.uIcc x✝ x)) (Se …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem ordConnectedComponent_subset : ordConnectedComponent s x ⊆ s := fun _ hy =>
  hy right_mem_uIcc


theorem subset_ordConnectedComponent {t} [h : OrdConnected s] (hs : x ∈ s) (ht : s ⊆ t) :
    s ⊆ ordConnectedComponent t x := fun _ hy => (h.uIcc_subset hs hy).trans ht


@[simp]
theorem self_mem_ordConnectedComponent : x ∈ ordConnectedComponent s x ↔ x ∈ s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (s.ordConnectedComponent x) x) (Membership.mem s x)
  -/
  rw [mem_ordConnectedComponent, uIcc_self, singleton_subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonempty_ordConnectedComponent : (ordConnectedComponent s x).Nonempty ↔ x ∈ s :=
  ⟨fun ⟨_, hy⟩ => hy <| left_mem_uIcc, fun h => ⟨x, self_mem_ordConnectedComponent.2 h⟩⟩


@[simp]
theorem ordConnectedComponent_eq_empty : ordConnectedComponent s x = ∅ ↔ x ∉ s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : α
    ⊢ Iff (Eq (s.ordConnectedComponent x) EmptyCollection.emptyCollection) (Not (M …
  -/
  rw [← not_nonempty_iff_eq_empty, nonempty_ordConnectedComponent]
  /-
    🎉 no goals
  -/


@[simp]
theorem ordConnectedComponent_empty : ordConnectedComponent ∅ x = ∅ :=
  ordConnectedComponent_eq_empty.2 (not_mem_empty x)


@[simp]
theorem ordConnectedComponent_univ : ordConnectedComponent univ x = univ := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x : α
    ⊢ Eq (Set.univ.ordConnectedComponent x) Set.univ
  -/
  simp [ordConnectedComponent]
  /-
    🎉 no goals
  -/


theorem ordConnectedComponent_inter (s t : Set α) (x : α) :
    ordConnectedComponent (s ∩ t) x = ordConnectedComponent s x ∩ ordConnectedComponent t x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x : α
    ⊢ Eq ((Inter.inter s t).ordConnectedComponent x) (Inter.inter (s.ordConnectedC …
  -/
  simp [ordConnectedComponent, setOf_and]
  /-
    🎉 no goals
  -/


theorem mem_ordConnectedComponent_comm :
    y ∈ ordConnectedComponent s x ↔ x ∈ ordConnectedComponent s y := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x y : α
    ⊢ Iff (Membership.mem (s.ordConnectedComponent x) y) (Membership.mem (s.ordCon …
  -/
  rw [mem_ordConnectedComponent, mem_ordConnectedComponent, uIcc_comm]
  /-
    🎉 no goals
  -/


theorem mem_ordConnectedComponent_trans (hxy : y ∈ ordConnectedComponent s x)
    (hyz : z ∈ ordConnectedComponent s y) : z ∈ ordConnectedComponent s x :=
  calc
    [[x, z]] ⊆ [[x, y]] ∪ [[y, z]] := uIcc_subset_uIcc_union_uIcc
    _ ⊆ s := union_subset hxy hyz


theorem ordConnectedComponent_eq (h : [[x, y]] ⊆ s) :
    ordConnectedComponent s x = ordConnectedComponent s y :=
  ext fun _ =>
    ⟨mem_ordConnectedComponent_trans (mem_ordConnectedComponent_comm.2 h),
      mem_ordConnectedComponent_trans h⟩


instance : OrdConnected (ordConnectedComponent s x) :=
  ordConnected_of_uIcc_subset_left fun _ hy _ hz => (uIcc_subset_uIcc_left hz).trans hy


/-- Projection from `s : Set α` to `α` sending each order connected component of `s` to a single
point of this component. -/
noncomputable def ordConnectedProj (s : Set α) : s → α := fun x : s =>
  (nonempty_ordConnectedComponent.2 x.2).some


theorem ordConnectedProj_mem_ordConnectedComponent (s : Set α) (x : s) :
    ordConnectedProj s x ∈ ordConnectedComponent s x :=
  Nonempty.some_mem _


theorem mem_ordConnectedComponent_ordConnectedProj (s : Set α) (x : s) :
    ↑x ∈ ordConnectedComponent s (ordConnectedProj s x) :=
  mem_ordConnectedComponent_comm.2 <| ordConnectedProj_mem_ordConnectedComponent s x


@[simp]
theorem ordConnectedComponent_ordConnectedProj (s : Set α) (x : s) :
    ordConnectedComponent s (ordConnectedProj s x) = ordConnectedComponent s x :=
  ordConnectedComponent_eq <| mem_ordConnectedComponent_ordConnectedProj _ _


@[simp]
theorem ordConnectedProj_eq {x y : s} :
    ordConnectedProj s x = ordConnectedProj s y ↔ [[(x : α), y]] ⊆ s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x y : ↑s
    ⊢ Iff (Eq (s.ordConnectedProj x) (s.ordConnectedProj y)) (HasSubset.Subset (Se …
  -/
  constructor <;> intro h
  · rw [← mem_ordConnectedComponent, ← ordConnectedComponent_ordConnectedProj, h,
      ordConnectedComponent_ordConnectedProj, self_mem_ordConnectedComponent]
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrder α
      s : Set α
      x y : ↑s
      h : Eq (s.ordConnectedProj x) (s.ordConnectedProj y)
      ⊢ Membership.mem s ↑y
    -/
    exact y.2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : LinearOrder α
      s : Set α
      x y : ↑s
      h : HasSubset.Subset (Set.uIcc ↑x ↑y) s
      ⊢ Eq (s.ordConnectedProj x) (s.ordConnectedProj y)
    -/
  · simp only [ordConnectedProj, ordConnectedComponent_eq h]
    /-
      🎉 no goals
    -/


/-- A set that intersects each order connected component of a set by a single point. Defined as the
range of `Set.ordConnectedProj s`. -/
def ordConnectedSection (s : Set α) : Set α :=
  range <| ordConnectedProj s


theorem dual_ordConnectedSection (s : Set α) :
    ordConnectedSection (ofDual ⁻¹' s) = ofDual ⁻¹' ordConnectedSection s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ Eq (Set.preimage (⇑OrderDual.ofDual) s).ordConnectedSection (Set.preimage (⇑ …
  -/
  simp only [ordConnectedSection]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ Eq (Set.range (Set.preimage (⇑OrderDual.ofDual) s).ordConnectedProj) (Set.pr …
  -/
  simp (config := { unfoldPartialApp := true }) only [ordConnectedProj]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ Eq (Set.range fun x => ⋯.some) (Set.preimage (⇑OrderDual.ofDual) (Set.range  …
  -/
  ext x
  simp only [mem_range, Subtype.exists, mem_preimage, OrderDual.exists, dual_ordConnectedComponent,
    ofDual_toDual]
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : OrderDual α
    ⊢ Iff (Exists fun a => Exists fun h => Eq ⋯.some x) (Exists fun a => Exists fu …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem ordConnectedSection_subset : ordConnectedSection s ⊆ s :=
  range_subset_iff.2 fun _ => ordConnectedComponent_subset <| Nonempty.some_mem _


theorem eq_of_mem_ordConnectedSection_of_uIcc_subset (hx : x ∈ ordConnectedSection s)
    (hy : y ∈ ordConnectedSection s) (h : [[x, y]] ⊆ s) : x = y := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x y : α
    hx : Membership.mem s.ordConnectedSection x
    hy : Membership.mem s.ordConnectedSection y
    h : HasSubset.Subset (Set.uIcc x y) s
    ⊢ Eq x y
  -/
  rcases hx with ⟨x, rfl⟩; rcases hy with ⟨y, rfl⟩
  exact
    ordConnectedProj_eq.2
      (mem_ordConnectedComponent_trans
        (mem_ordConnectedComponent_trans (ordConnectedProj_mem_ordConnectedComponent _ _) h)
        (mem_ordConnectedComponent_ordConnectedProj _ _))


/-- Given two sets `s t : Set α`, the set `Set.orderSeparatingSet s t` is the set of points that
belong both to some `Set.ordConnectedComponent tᶜ x`, `x ∈ s`, and to some
`Set.ordConnectedComponent sᶜ x`, `x ∈ t`. In the case of two disjoint closed sets, this is the
union of all open intervals $(a, b)$ such that their endpoints belong to different sets. -/
def ordSeparatingSet (s t : Set α) : Set α :=
  (⋃ x ∈ s, ordConnectedComponent tᶜ x) ∩ ⋃ x ∈ t, ordConnectedComponent sᶜ x


theorem ordSeparatingSet_comm (s t : Set α) : ordSeparatingSet s t = ordSeparatingSet t s :=
  inter_comm _ _


theorem disjoint_left_ordSeparatingSet : Disjoint s (ordSeparatingSet s t) :=
  Disjoint.inter_right' _ <|
    disjoint_iUnion₂_right.2 fun _ _ =>
      disjoint_compl_right.mono_right <| ordConnectedComponent_subset


theorem disjoint_right_ordSeparatingSet : Disjoint t (ordSeparatingSet s t) :=
  ordSeparatingSet_comm t s ▸ disjoint_left_ordSeparatingSet


theorem dual_ordSeparatingSet :
    ordSeparatingSet (ofDual ⁻¹' s) (ofDual ⁻¹' t) = ofDual ⁻¹' ordSeparatingSet s t := by
  simp only [ordSeparatingSet, mem_preimage, ← toDual.surjective.iUnion_comp, ofDual_toDual,
    dual_ordConnectedComponent, ← preimage_compl, preimage_inter, preimage_iUnion]


/-- An auxiliary neighborhood that will be used in the proof of
    `OrderTopology.CompletelyNormalSpace`. -/
def ordT5Nhd (s t : Set α) : Set α :=
  ⋃ x ∈ s, ordConnectedComponent (tᶜ ∩ (ordConnectedSection <| ordSeparatingSet s t)ᶜ) x


theorem disjoint_ordT5Nhd : Disjoint (ordT5Nhd s t) (ordT5Nhd t s) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    ⊢ Disjoint (s.ordT5Nhd t) (t.ordT5Nhd s)
  -/
  rw [disjoint_iff_inf_le]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    ⊢ LE.le (Min.min (s.ordT5Nhd t) (t.ordT5Nhd s)) Bot.bot
  -/
  rintro x ⟨hx₁, hx₂⟩
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x : α
    hx₁ : Membership.mem (s.ordT5Nhd t) x
    hx₂ : Membership.mem (t.ordT5Nhd s) x
    ⊢ Membership.mem Bot.bot x
  -/
  rcases mem_iUnion₂.1 hx₁ with ⟨a, has, ha⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x : α
    hx₁ : Membership.mem (s.ordT5Nhd t) x
    hx₂ : Membership.mem (t.ordT5Nhd s) x
    a : α
    has : Membership.mem s a
    ha : Membership.mem ((Inter.inter (HasCompl.compl t) (HasCompl.compl (s.ordSep …
    ⊢ Membership.mem Bot.bot x
  -/
  clear hx₁
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x : α
    hx₂ : Membership.mem (t.ordT5Nhd s) x
    a : α
    has : Membership.mem s a
    ha : Membership.mem ((Inter.inter (HasCompl.compl t) (HasCompl.compl (s.ordSep …
    ⊢ Membership.mem Bot.bot x
  -/
  rcases mem_iUnion₂.1 hx₂ with ⟨b, hbt, hb⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x : α
    hx₂ : Membership.mem (t.ordT5Nhd s) x
    a : α
    has : Membership.mem s a
    ha : Membership.mem ((Inter.inter (HasCompl.compl t) (HasCompl.compl (s.ordSep …
    b : α
    hbt : Membership.mem t b
    hb : Membership.mem ((Inter.inter (HasCompl.compl s) (HasCompl.compl (t.ordSep …
    ⊢ Membership.mem Bot.bot x
  -/
  clear hx₂
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    ha : Membership.mem ((Inter.inter (HasCompl.compl t) (HasCompl.compl (s.ordSep …
    b : α
    hbt : Membership.mem t b
    hb : Membership.mem ((Inter.inter (HasCompl.compl s) (HasCompl.compl (t.ordSep …
    ⊢ Membership.mem Bot.bot x
  -/
  rw [mem_ordConnectedComponent, subset_inter_iff] at ha hb
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    ha : And (HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)) (HasSubset.Subse …
    b : α
    hbt : Membership.mem t b
    hb : And (HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)) (HasSubset.Subse …
    ⊢ Membership.mem Bot.bot x
  -/
  wlog hab : a ≤ b with H
    /-
      case intro.intro.intro.intro.intro.inr
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Set α
      x a : α
      has : Membership.mem s a
      ha : And (HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)) (HasSubset.Subse …
      b : α
      hbt : Membership.mem t b
      hb : And (HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)) (HasSubset.Subse …
      H : ∀ {α : Type u_1} [inst : LinearOrder α] {s t : Set α} ⦃x : α⦄ (a : α), Mem …
      hab : Not (LE.le a b)
      ⊢ Membership.mem Bot.bot x
    -/
  · exact H b hbt hb a has ha (le_of_not_le hab)
    /-
      🎉 no goals
    -/
  /-
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    ha : And (HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)) (HasSubset.Subse …
    b : α
    hbt : Membership.mem t b
    hb : And (HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)) (HasSubset.Subse …
    hab : LE.le a b
    ⊢ Membership.mem Bot.bot x
  -/
  cases' ha with ha ha'
  /-
    case intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hb : And (HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)) (HasSubset.Subse …
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    ha' : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl (s.ordSeparatingSet t).o …
    ⊢ Membership.mem Bot.bot x
  -/
  cases' hb with hb hb'
  have hsub : [[a, b]] ⊆ (ordSeparatingSet s t).ordConnectedSectionᶜ := by
    rw [ordSeparatingSet_comm, uIcc_comm] at hb'
    calc
      [[a, b]] ⊆ [[a, x]] ∪ [[x, b]] := uIcc_subset_uIcc_union_uIcc
      _ ⊆ (ordSeparatingSet s t).ordConnectedSectionᶜ := union_subset ha' hb'
  /-
    case intro.intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    ha' : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl (s.ordSeparatingSet t).o …
    hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
    hb' : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl (t.ordSeparatingSet s).o …
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    ⊢ Membership.mem Bot.bot x
  -/
  clear ha' hb'
  /-
    case intro.intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    ⊢ Membership.mem Bot.bot x
  -/
  rcases le_total x a with hxa | hax
    /-
      case intro.intro.inl
      α✝ : Type u_1
      inst✝¹ : LinearOrder α✝
      s✝ t✝ : Set α✝
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Set α
      x a : α
      has : Membership.mem s a
      b : α
      hbt : Membership.mem t b
      hab : LE.le a b
      ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
      hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
      hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
      hxa : LE.le x a
      ⊢ Membership.mem Bot.bot x
    -/
  · exact hb (Icc_subset_uIcc' ⟨hxa, hab⟩) has
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    hax : LE.le a x
    ⊢ Membership.mem Bot.bot x
  -/
  rcases le_total b x with hbx | hxb
    /-
      case intro.intro.inr.inl
      α✝ : Type u_1
      inst✝¹ : LinearOrder α✝
      s✝ t✝ : Set α✝
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Set α
      x a : α
      has : Membership.mem s a
      b : α
      hbt : Membership.mem t b
      hab : LE.le a b
      ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
      hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
      hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
      hax : LE.le a x
      hbx : LE.le b x
      ⊢ Membership.mem Bot.bot x
    -/
  · exact ha (Icc_subset_uIcc ⟨hab, hbx⟩) hbt
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.inr
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    hax : LE.le a x
    hxb : LE.le x b
    ⊢ Membership.mem Bot.bot x
  -/
  have h' : x ∈ ordSeparatingSet s t := ⟨mem_iUnion₂.2 ⟨a, has, ha⟩, mem_iUnion₂.2 ⟨b, hbt, hb⟩⟩
  /-
    case intro.intro.inr.inr
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    x a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    ha : HasSubset.Subset (Set.uIcc a x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b x) (HasCompl.compl s)
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    hax : LE.le a x
    hxb : LE.le x b
    h' : Membership.mem (s.ordSeparatingSet t) x
    ⊢ Membership.mem Bot.bot x
  -/
  lift x to ordSeparatingSet s t using h'
  suffices ordConnectedComponent (ordSeparatingSet s t) x ⊆ [[a, b]] from
    hsub (this <| ordConnectedProj_mem_ordConnectedComponent _ x) (mem_range_self _)
  /-
    case intro.intro.inr.inr.intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    x : Subtype fun x => Membership.mem (s.ordSeparatingSet t) x
    ha : HasSubset.Subset (Set.uIcc a ↑x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b ↑x) (HasCompl.compl s)
    hax : LE.le a ↑x
    hxb : LE.le (↑x) b
    ⊢ HasSubset.Subset ((s.ordSeparatingSet t).ordConnectedComponent ↑x) (Set.uIcc …
  -/
  rintro y hy
  /-
    case intro.intro.inr.inr.intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    x : Subtype fun x => Membership.mem (s.ordSeparatingSet t) x
    ha : HasSubset.Subset (Set.uIcc a ↑x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b ↑x) (HasCompl.compl s)
    hax : LE.le a ↑x
    hxb : LE.le (↑x) b
    y : α
    hy : Membership.mem ((s.ordSeparatingSet t).ordConnectedComponent ↑x) y
    ⊢ Membership.mem (Set.uIcc a b) y
  -/
  rw [uIcc_of_le hab, mem_Icc, ← not_lt, ← not_lt]
  have sol1 := fun (hya : y < a) =>
      (disjoint_left (t := ordSeparatingSet s t)).1 disjoint_left_ordSeparatingSet has
        (hy <| Icc_subset_uIcc' ⟨hya.le, hax⟩)
  have sol2 := fun (hby : b < y) =>
      (disjoint_left (t := ordSeparatingSet s t)).1 disjoint_right_ordSeparatingSet hbt
        (hy <| Icc_subset_uIcc ⟨hxb, hby.le⟩)
  /-
    case intro.intro.inr.inr.intro
    α✝ : Type u_1
    inst✝¹ : LinearOrder α✝
    s✝ t✝ : Set α✝
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    a : α
    has : Membership.mem s a
    b : α
    hbt : Membership.mem t b
    hab : LE.le a b
    hsub : HasSubset.Subset (Set.uIcc a b) (HasCompl.compl (s.ordSeparatingSet t). …
    x : Subtype fun x => Membership.mem (s.ordSeparatingSet t) x
    ha : HasSubset.Subset (Set.uIcc a ↑x) (HasCompl.compl t)
    hb : HasSubset.Subset (Set.uIcc b ↑x) (HasCompl.compl s)
    hax : LE.le a ↑x
    hxb : LE.le (↑x) b
    y : α
    hy : Membership.mem ((s.ordSeparatingSet t).ordConnectedComponent ↑x) y
    sol1 : LT.lt y a → False
    sol2 : LT.lt b y → False
    ⊢ And (Not (LT.lt y a)) (Not (LT.lt b y))
  -/
  exact ⟨sol1, sol2⟩
  /-
    🎉 no goals
  -/


