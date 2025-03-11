/-- Elements of `𝒜` that do not contain `a`. -/
def nonMemberSubfamily (a : α) (𝒜 : Finset (Finset α)) : Finset (Finset α) := {s ∈ 𝒜 | a ∉ s}


/-- Image of the elements of `𝒜` which contain `a` under removing `a`. Finsets that do not contain
`a` such that `insert a s ∈ 𝒜`. -/
def memberSubfamily (a : α) (𝒜 : Finset (Finset α)) : Finset (Finset α) :=
  {s ∈ 𝒜 | a ∈ s}.image fun s => erase s a


@[simp]
theorem mem_nonMemberSubfamily : s ∈ 𝒜.nonMemberSubfamily a ↔ s ∈ 𝒜 ∧ a ∉ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Iff (Membership.mem (Finset.nonMemberSubfamily a 𝒜) s) (And (Membership.mem  …
  -/
  simp [nonMemberSubfamily]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_memberSubfamily : s ∈ 𝒜.memberSubfamily a ↔ insert a s ∈ 𝒜 ∧ a ∉ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Iff (Membership.mem (Finset.memberSubfamily a 𝒜) s) (And (Membership.mem 𝒜 ( …
  -/
  simp_rw [memberSubfamily, mem_image, mem_filter]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Iff (Exists fun a_1 => And (And (Membership.mem 𝒜 a_1) (Membership.mem a_1 a …
  -/
  refine ⟨?_, fun h => ⟨insert a s, ⟨h.1, by simp⟩, erase_insert h.2⟩⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ (Exists fun a_1 => And (And (Membership.mem 𝒜 a_1) (Membership.mem a_1 a)) ( …
  -/
  rintro ⟨s, ⟨hs1, hs2⟩, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    s : Finset α
    hs1 : Membership.mem 𝒜 s
    hs2 : Membership.mem s a
    ⊢ And (Membership.mem 𝒜 (Insert.insert a (s.erase a))) (Not (Membership.mem (s …
  -/
  rw [insert_erase hs2]
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    s : Finset α
    hs1 : Membership.mem 𝒜 s
    hs2 : Membership.mem s a
    ⊢ And (Membership.mem 𝒜 s) (Not (Membership.mem (s.erase a) a))
  -/
  exact ⟨hs1, not_mem_erase _ _⟩
  /-
    🎉 no goals
  -/


theorem nonMemberSubfamily_inter (a : α) (𝒜 ℬ : Finset (Finset α)) :
    (𝒜 ∩ ℬ).nonMemberSubfamily a = 𝒜.nonMemberSubfamily a ∩ ℬ.nonMemberSubfamily a :=
  filter_inter_distrib _ _ _


theorem memberSubfamily_inter (a : α) (𝒜 ℬ : Finset (Finset α)) :
    (𝒜 ∩ ℬ).memberSubfamily a = 𝒜.memberSubfamily a ∩ ℬ.memberSubfamily a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (Finset.memberSubfamily a (Inter.inter 𝒜 ℬ)) (Inter.inter (Finset.memberS …
  -/
  unfold memberSubfamily
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (Finset.image (fun s => s.erase a) (Finset.filter (fun s => Membership.me …
  -/
  rw [filter_inter_distrib, image_inter_of_injOn _ _ ((erase_injOn' _).mono _)]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 ℬ : Finset (Finset α)
    ⊢ HasSubset.Subset (Union.union ↑(Finset.filter (fun s => Membership.mem s a)  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nonMemberSubfamily_union (a : α) (𝒜 ℬ : Finset (Finset α)) :
    (𝒜 ∪ ℬ).nonMemberSubfamily a = 𝒜.nonMemberSubfamily a ∪ ℬ.nonMemberSubfamily a :=
  filter_union _ _ _


theorem memberSubfamily_union (a : α) (𝒜 ℬ : Finset (Finset α)) :
    (𝒜 ∪ ℬ).memberSubfamily a = 𝒜.memberSubfamily a ∪ ℬ.memberSubfamily a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (Finset.memberSubfamily a (Union.union 𝒜 ℬ)) (Union.union (Finset.memberS …
  -/
  simp_rw [memberSubfamily, filter_union, image_union]
  /-
    🎉 no goals
  -/


theorem card_memberSubfamily_add_card_nonMemberSubfamily (a : α) (𝒜 : Finset (Finset α)) :
    #(𝒜.memberSubfamily a) + #(𝒜.nonMemberSubfamily a) = #𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (Finset.memberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a …
  -/
  rw [memberSubfamily, nonMemberSubfamily, card_image_of_injOn]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      ⊢ Eq (HAdd.hAdd (Finset.filter (fun s => Membership.mem s a) 𝒜).card (Finset.f …
    -/
  · conv_rhs => rw [← filter_card_add_filter_neg_card_eq_card (fun s => (a ∈ s))]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      ⊢ Set.InjOn (fun s => s.erase a) ↑(Finset.filter (fun s => Membership.mem s a) …
    -/
  · apply (erase_injOn' _).mono
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      ⊢ HasSubset.Subset (↑(Finset.filter (fun s => Membership.mem s a) 𝒜)) (setOf f …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem memberSubfamily_union_nonMemberSubfamily (a : α) (𝒜 : Finset (Finset α)) :
    𝒜.memberSubfamily a ∪ 𝒜.nonMemberSubfamily a = 𝒜.image fun s => s.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    ⊢ Eq (Union.union (Finset.memberSubfamily a 𝒜) (Finset.nonMemberSubfamily a 𝒜) …
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Union.union (Finset.memberSubfamily a 𝒜) (Finset.nonMem …
  -/
  simp only [mem_union, mem_memberSubfamily, mem_nonMemberSubfamily, mem_image, exists_prop]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Or (And (Membership.mem 𝒜 (Insert.insert a s)) (Not (Membership.mem s a …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ Or (And (Membership.mem 𝒜 (Insert.insert a s)) (Not (Membership.mem s a))) ( …
    -/
  · rintro (h | h)
      /-
        case h.mp.inl
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        𝒜 : Finset (Finset α)
        s : Finset α
        h : And (Membership.mem 𝒜 (Insert.insert a s)) (Not (Membership.mem s a))
        ⊢ Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (a_1.erase a) s)
      -/
    · exact ⟨_, h.1, erase_insert h.2⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mp.inr
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        𝒜 : Finset (Finset α)
        s : Finset α
        h : And (Membership.mem 𝒜 s) (Not (Membership.mem s a))
        ⊢ Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (a_1.erase a) s)
      -/
    · exact ⟨_, h.1, erase_eq_of_not_mem h.2⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ (Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (a_1.erase a) s)) → Or (An …
    -/
  · rintro ⟨s, hs, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Membership.mem 𝒜 s
      ⊢ Or (And (Membership.mem 𝒜 (Insert.insert a (s.erase a))) (Not (Membership.me …
    -/
    by_cases ha : a ∈ s
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        𝒜 : Finset (Finset α)
        s : Finset α
        hs : Membership.mem 𝒜 s
        ha : Membership.mem s a
        ⊢ Or (And (Membership.mem 𝒜 (Insert.insert a (s.erase a))) (Not (Membership.me …
      -/
    · exact Or.inl ⟨by rwa [insert_erase ha], not_mem_erase _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        𝒜 : Finset (Finset α)
        s : Finset α
        hs : Membership.mem 𝒜 s
        ha : Not (Membership.mem s a)
        ⊢ Or (And (Membership.mem 𝒜 (Insert.insert a (s.erase a))) (Not (Membership.me …
      -/
    · exact Or.inr ⟨by rwa [erase_eq_of_not_mem ha], not_mem_erase _ _⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem memberSubfamily_memberSubfamily : (𝒜.memberSubfamily a).memberSubfamily a = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.memberSubfamily a (Finset.memberSubfamily a 𝒜)) EmptyCollection.e …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.memberSubfamily a (Finset.memberSubfamily a 𝒜))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem memberSubfamily_nonMemberSubfamily : (𝒜.nonMemberSubfamily a).memberSubfamily a = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.memberSubfamily a (Finset.nonMemberSubfamily a 𝒜)) EmptyCollectio …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.memberSubfamily a (Finset.nonMemberSubfamily a 𝒜 …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem nonMemberSubfamily_memberSubfamily :
    (𝒜.memberSubfamily a).nonMemberSubfamily a = 𝒜.memberSubfamily a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.nonMemberSubfamily a (Finset.memberSubfamily a 𝒜)) (Finset.member …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.nonMemberSubfamily a (Finset.memberSubfamily a 𝒜 …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem nonMemberSubfamily_nonMemberSubfamily :
    (𝒜.nonMemberSubfamily a).nonMemberSubfamily a = 𝒜.nonMemberSubfamily a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.nonMemberSubfamily a (Finset.nonMemberSubfamily a 𝒜)) (Finset.non …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.nonMemberSubfamily a (Finset.nonMemberSubfamily  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma memberSubfamily_image_insert (h𝒜 : ∀ s ∈ 𝒜, a ∉ s) :
    (𝒜.image <| insert a).memberSubfamily a = 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h𝒜 : ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a)
    ⊢ Eq (Finset.memberSubfamily a (Finset.image (Insert.insert a) 𝒜)) 𝒜
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h𝒜 : ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a)
    s : Finset α
    ⊢ Iff (Membership.mem (Finset.memberSubfamily a (Finset.image (Insert.insert a …
  -/
  simp only [mem_memberSubfamily, mem_image]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h𝒜 : ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a)
    s : Finset α
    ⊢ Iff (And (Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (Insert.insert a  …
  -/
  refine ⟨?_, fun hs ↦ ⟨⟨s, hs, rfl⟩, h𝒜 _ hs⟩⟩
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h𝒜 : ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a)
    s : Finset α
    ⊢ And (Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (Insert.insert a a_1)  …
  -/
  rintro ⟨⟨t, ht, hts⟩, hs⟩
  /-
    case h.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h𝒜 : ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a)
    s : Finset α
    hs : Not (Membership.mem s a)
    t : Finset α
    ht : Membership.mem 𝒜 t
    hts : Eq (Insert.insert a t) (Insert.insert a s)
    ⊢ Membership.mem 𝒜 s
  -/
  rwa [← insert_erase_invOn.2.injOn (h𝒜 _ ht) hs hts]
  /-
    🎉 no goals
  -/


@[simp] lemma nonMemberSubfamily_image_insert : (𝒜.image <| insert a).nonMemberSubfamily a = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.nonMemberSubfamily a (Finset.image (Insert.insert a) 𝒜)) EmptyCol …
  -/
  simp [eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


@[simp] lemma memberSubfamily_image_erase : (𝒜.image (erase · a)).memberSubfamily a = ∅ := by
  simp [eq_empty_iff_forall_not_mem,
    (ne_of_mem_of_not_mem' (mem_insert_self _ _) (not_mem_erase _ _)).symm]


lemma image_insert_memberSubfamily (𝒜 : Finset (Finset α)) (a : α) :
    (𝒜.memberSubfamily a).image (insert a) = {s ∈ 𝒜 | a ∈ s} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    ⊢ Eq (Finset.image (Insert.insert a) (Finset.memberSubfamily a 𝒜)) (Finset.fil …
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    s : Finset α
    ⊢ Iff (Membership.mem (Finset.image (Insert.insert a) (Finset.memberSubfamily  …
  -/
  simp only [mem_memberSubfamily, mem_image, mem_filter]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    s : Finset α
    ⊢ Iff (Exists fun a_1 => And (And (Membership.mem 𝒜 (Insert.insert a a_1)) (No …
  -/
  refine ⟨?_, fun ⟨hs, ha⟩ ↦ ⟨erase s a, ⟨?_, not_mem_erase _ _⟩, insert_erase ha⟩⟩
    /-
      case h.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      a : α
      s : Finset α
      ⊢ (Exists fun a_1 => And (And (Membership.mem 𝒜 (Insert.insert a a_1)) (Not (M …
    -/
  · rintro ⟨s, ⟨hs, -⟩, rfl⟩
    /-
      case h.refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      a : α
      s : Finset α
      hs : Membership.mem 𝒜 (Insert.insert a s)
      ⊢ And (Membership.mem 𝒜 (Insert.insert a s)) (Membership.mem (Insert.insert a  …
    -/
    exact ⟨hs, mem_insert_self _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      a : α
      s : Finset α
      x✝ : And (Membership.mem 𝒜 s) (Membership.mem s a)
      hs : Membership.mem 𝒜 s
      ha : Membership.mem s a
      ⊢ Membership.mem 𝒜 (Insert.insert a (s.erase a))
    -/
  · rwa [insert_erase ha]
    /-
      🎉 no goals
    -/


/-- Induction principle for finset families. To prove a statement for every finset family,
it suffices to prove it for
* the empty finset family.
* the finset family which only contains the empty finset.
* `ℬ ∪ {s ∪ {a} | s ∈ 𝒞}` assuming the property for `ℬ` and `𝒞`, where `a` is an element of the
  ground type and `𝒜` and `ℬ` are families of finsets not containing `a`.
  Note that instead of giving `ℬ` and `𝒞`, the `subfamily` case gives you
  `𝒜 = ℬ ∪ {s ∪ {a} | s ∈ 𝒞}`, so that `ℬ = 𝒜.nonMemberSubfamily` and `𝒞 = 𝒜.memberSubfamily`.

This is a way of formalising induction on `n` where `𝒜` is a finset family on `n` elements.

See also `Finset.family_induction_on.`-/
@[elab_as_elim]
lemma memberFamily_induction_on {p : Finset (Finset α) → Prop}
    (𝒜 : Finset (Finset α)) (empty : p ∅) (singleton_empty : p {∅})
    (subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄,
      p (𝒜.nonMemberSubfamily a) → p (𝒜.memberSubfamily a) → p 𝒜) : p 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜 : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
    ⊢ p 𝒜
  -/
  set u := 𝒜.sup id
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜 : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
    u : Finset α := 𝒜.sup id
    ⊢ p 𝒜
  -/
  have hu : ∀ s ∈ 𝒜, s ⊆ u := fun s ↦ le_sup (f := id)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜 : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
    u : Finset α := 𝒜.sup id
    hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s u
    ⊢ p 𝒜
  -/
  clear_value u
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜 : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
    u : Finset α
    hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s u
    ⊢ p 𝒜
  -/
  induction' u using Finset.induction with a u _ ih generalizing 𝒜
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s EmptyCollection …
      ⊢ p 𝒜
    -/
  · simp_rw [subset_empty] at hu
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → Eq s EmptyCollection.emptyCollection
      ⊢ p 𝒜
    -/
    rw [← subset_singleton_iff', subset_singleton_iff] at hu
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      𝒜 : Finset (Finset α)
      hu : Or (Eq 𝒜 EmptyCollection.emptyCollection) (Eq 𝒜 (Singleton.singleton Empt …
      ⊢ p 𝒜
    -/
                               /-
                                 🎉 no goals
                               -/
    obtain rfl | rfl := hu <;> assumption
                               /-
                                 🎉 no goals
                               -/
  /-
    case insert
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
    a : α
    u : Finset α
    a✝ : Not (Membership.mem u a)
    ih : ∀ (𝒜 : Finset (Finset α)), (∀ (s : Finset α), Membership.mem 𝒜 s → HasSub …
    𝒜 : Finset (Finset α)
    hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s (Insert.insert  …
    ⊢ p 𝒜
  -/
  refine subfamily a (ih _ ?_) (ih _ ?_)
    /-
      case insert.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      a : α
      u : Finset α
      a✝ : Not (Membership.mem u a)
      ih : ∀ (𝒜 : Finset (Finset α)), (∀ (s : Finset α), Membership.mem 𝒜 s → HasSub …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s (Insert.insert  …
      ⊢ ∀ (s : Finset α), Membership.mem (Finset.nonMemberSubfamily a 𝒜) s → HasSubs …
    -/
  · simp only [mem_nonMemberSubfamily, and_imp]
    /-
      case insert.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      a : α
      u : Finset α
      a✝ : Not (Membership.mem u a)
      ih : ∀ (𝒜 : Finset (Finset α)), (∀ (s : Finset α), Membership.mem 𝒜 s → HasSub …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s (Insert.insert  …
      ⊢ ∀ (s : Finset α), Membership.mem 𝒜 s → Not (Membership.mem s a) → HasSubset. …
    -/
    exact fun s hs has ↦ (subset_insert_iff_of_not_mem has).1 <| hu _ hs
    /-
      🎉 no goals
    -/
    /-
      case insert.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      a : α
      u : Finset α
      a✝ : Not (Membership.mem u a)
      ih : ∀ (𝒜 : Finset (Finset α)), (∀ (s : Finset α), Membership.mem 𝒜 s → HasSub …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s (Insert.insert  …
      ⊢ ∀ (s : Finset α), Membership.mem (Finset.memberSubfamily a 𝒜) s → HasSubset. …
    -/
  · simp only [mem_memberSubfamily, and_imp]
    /-
      case insert.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      p : Finset (Finset α) → Prop
      empty : p EmptyCollection.emptyCollection
      singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
      subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.nonMemberSubfamily a  …
      a : α
      u : Finset α
      a✝ : Not (Membership.mem u a)
      ih : ∀ (𝒜 : Finset (Finset α)), (∀ (s : Finset α), Membership.mem 𝒜 s → HasSub …
      𝒜 : Finset (Finset α)
      hu : ∀ (s : Finset α), Membership.mem 𝒜 s → HasSubset.Subset s (Insert.insert  …
      ⊢ ∀ (s : Finset α), Membership.mem 𝒜 (Insert.insert a s) → Not (Membership.mem …
    -/
    exact fun s hs ha ↦ (insert_subset_insert_iff ha).1 <| hu _ hs
    /-
      🎉 no goals
    -/


/-- Induction principle for finset families. To prove a statement for every finset family,
it suffices to prove it for
* the empty finset family.
* the finset family which only contains the empty finset.
* `{s ∪ {a} | s ∈ 𝒜}` assuming the property for `𝒜` a family of finsets not containing `a`.
* `ℬ ∪ 𝒞` assuming the property for `ℬ` and `𝒞`, where `a` is an element of the ground type and
  `ℬ`is a family of finsets not containing `a` and `𝒞` a family of finsets containing `a`.
  Note that instead of giving `ℬ` and `𝒞`, the `subfamily` case gives you `𝒜 = ℬ ∪ 𝒞`, so that
  `ℬ = {s ∈ 𝒜 | a ∉ s}` and `𝒞 = {s ∈ 𝒜 | a ∈ s}`.

This is a way of formalising induction on `n` where `𝒜` is a finset family on `n` elements.

See also `Finset.memberFamily_induction_on.`-/
@[elab_as_elim]
protected lemma family_induction_on {p : Finset (Finset α) → Prop}
    (𝒜 : Finset (Finset α)) (empty : p ∅) (singleton_empty : p {∅})
    (image_insert : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄,
      (∀ s ∈ 𝒜, a ∉ s) → p 𝒜 → p (𝒜.image <| insert a))
    (subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄,
      p {s ∈ 𝒜 | a ∉ s} → p {s ∈ 𝒜 | a ∈ s} → p 𝒜) : p 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜 : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    image_insert : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, (∀ (s : Finset α), Membershi …
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.filter (fun s => Not  …
    ⊢ p 𝒜
  -/
  refine memberFamily_induction_on 𝒜 empty singleton_empty fun a 𝒜 h𝒜₀ h𝒜₁ ↦ subfamily a h𝒜₀ ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜✝ : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    image_insert : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, (∀ (s : Finset α), Membershi …
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.filter (fun s => Not  …
    a : α
    𝒜 : Finset (Finset α)
    h𝒜₀ : p (Finset.nonMemberSubfamily a 𝒜)
    h𝒜₁ : p (Finset.memberSubfamily a 𝒜)
    ⊢ p (Finset.filter (fun s => Membership.mem s a) 𝒜)
  -/
  rw [← image_insert_memberSubfamily]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : Finset (Finset α) → Prop
    𝒜✝ : Finset (Finset α)
    empty : p EmptyCollection.emptyCollection
    singleton_empty : p (Singleton.singleton EmptyCollection.emptyCollection)
    image_insert : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, (∀ (s : Finset α), Membershi …
    subfamily : ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, p (Finset.filter (fun s => Not  …
    a : α
    𝒜 : Finset (Finset α)
    h𝒜₀ : p (Finset.nonMemberSubfamily a 𝒜)
    h𝒜₁ : p (Finset.memberSubfamily a 𝒜)
    ⊢ p (Finset.image (Insert.insert a) (Finset.memberSubfamily a 𝒜))
  -/
  exact image_insert _ (by simp) h𝒜₁
  /-
    🎉 no goals
  -/


/-- `a`-down-compressing `𝒜` means removing `a` from the elements of `𝒜` that contain it, when the
resulting Finset is not already in `𝒜`. -/
def compression (a : α) (𝒜 : Finset (Finset α)) : Finset (Finset α) :=
  {s ∈ 𝒜 | erase s a ∈ 𝒜}.disjUnion {s ∈ 𝒜.image fun s ↦ erase s a | s ∉ 𝒜} <|
    disjoint_left.2 fun _s h₁ h₂ ↦ (mem_filter.1 h₂).2 (mem_filter.1 h₁).1


@[inherit_doc]
scoped[FinsetFamily] notation "𝓓 " => Down.compression
-- Porting note: had to open this

/-- `a` is in the down-compressed family iff it's in the original and its compression is in the
original, or it's not in the original but it's the compression of something in the original. -/
theorem mem_compression : s ∈ 𝓓 a 𝒜 ↔ s ∈ 𝒜 ∧ s.erase a ∈ 𝒜 ∨ s ∉ 𝒜 ∧ insert a s ∈ 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Iff (Membership.mem (Down.compression a 𝒜) s) (Or (And (Membership.mem 𝒜 s)  …
  -/
  simp_rw [compression, mem_disjUnion, mem_filter, mem_image, and_comm (a := (¬ s ∈ 𝒜))]
  refine
    or_congr_right
      (and_congr_left fun hs =>
        ⟨?_, fun h => ⟨_, h, erase_insert <| insert_ne_self.1 <| ne_of_mem_of_not_mem h hs⟩⟩)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : Not (Membership.mem 𝒜 s)
    ⊢ (Exists fun a_1 => And (Membership.mem 𝒜 a_1) (Eq (a_1.erase a) s)) → Member …
  -/
  rintro ⟨t, ht, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    t : Finset α
    ht : Membership.mem 𝒜 t
    hs : Not (Membership.mem 𝒜 (t.erase a))
    ⊢ Membership.mem 𝒜 (Insert.insert a (t.erase a))
  -/
  rwa [insert_erase (erase_ne_self.1 (ne_of_mem_of_not_mem ht hs).symm)]
  /-
    🎉 no goals
  -/


theorem erase_mem_compression (hs : s ∈ 𝒜) : s.erase a ∈ 𝓓 a 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : Membership.mem 𝒜 s
    ⊢ Membership.mem (Down.compression a 𝒜) (s.erase a)
  -/
  simp_rw [mem_compression, erase_idem, and_self_iff]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : Membership.mem 𝒜 s
    ⊢ Or (Membership.mem 𝒜 (s.erase a)) (And (Not (Membership.mem 𝒜 (s.erase a)))  …
  -/
  refine (em _).imp_right fun h => ⟨h, ?_⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : Membership.mem 𝒜 s
    h : Not (Membership.mem 𝒜 (s.erase a))
    ⊢ Membership.mem 𝒜 (Insert.insert a (s.erase a))
  -/
  rwa [insert_erase (erase_ne_self.1 (ne_of_mem_of_not_mem hs h).symm)]
  /-
    🎉 no goals
  -/

-- This is a special case of `erase_mem_compression` once we have `compression_idem`.

theorem erase_mem_compression_of_mem_compression : s ∈ 𝓓 a 𝒜 → s.erase a ∈ 𝓓 a 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Membership.mem (Down.compression a 𝒜) s → Membership.mem (Down.compression a …
  -/
  simp_rw [mem_compression, erase_idem]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    ⊢ Or (And (Membership.mem 𝒜 s) (Membership.mem 𝒜 (s.erase a))) (And (Not (Memb …
  -/
  refine Or.imp (fun h => ⟨h.2, h.2⟩) fun h => ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    h : And (Not (Membership.mem 𝒜 s)) (Membership.mem 𝒜 (Insert.insert a s))
    ⊢ And (Not (Membership.mem 𝒜 (s.erase a))) (Membership.mem 𝒜 (Insert.insert a  …
  -/
  rwa [erase_eq_of_not_mem (insert_ne_self.1 <| ne_of_mem_of_not_mem h.2 h.1)]
  /-
    🎉 no goals
  -/


theorem mem_compression_of_insert_mem_compression (h : insert a s ∈ 𝓓 a 𝒜) : s ∈ 𝓓 a 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    h : Membership.mem (Down.compression a 𝒜) (Insert.insert a s)
    ⊢ Membership.mem (Down.compression a 𝒜) s
  -/
  by_cases ha : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      h : Membership.mem (Down.compression a 𝒜) (Insert.insert a s)
      ha : Membership.mem s a
      ⊢ Membership.mem (Down.compression a 𝒜) s
    -/
  · rwa [insert_eq_of_mem ha] at h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      h : Membership.mem (Down.compression a 𝒜) (Insert.insert a s)
      ha : Not (Membership.mem s a)
      ⊢ Membership.mem (Down.compression a 𝒜) s
    -/
  · rw [← erase_insert ha]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      h : Membership.mem (Down.compression a 𝒜) (Insert.insert a s)
      ha : Not (Membership.mem s a)
      ⊢ Membership.mem (Down.compression a 𝒜) ((Insert.insert a s).erase a)
    -/
    exact erase_mem_compression_of_mem_compression h
    /-
      🎉 no goals
    -/


/-- Down-compressing a family is idempotent. -/
@[simp]
theorem compression_idem (a : α) (𝒜 : Finset (Finset α)) : 𝓓 a (𝓓 a 𝒜) = 𝓓 a 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    ⊢ Eq (Down.compression a (Down.compression a 𝒜)) (Down.compression a 𝒜)
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Down.compression a (Down.compression a 𝒜)) s) (Membersh …
  -/
  refine mem_compression.trans ⟨?_, fun h => Or.inl ⟨h, erase_mem_compression_of_mem_compression h⟩⟩
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Or (And (Membership.mem (Down.compression a 𝒜) s) (Membership.mem (Down.comp …
  -/
  rintro (h | h)
    /-
      case h.inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      s : Finset α
      h : And (Membership.mem (Down.compression a 𝒜) s) (Membership.mem (Down.compre …
      ⊢ Membership.mem (Down.compression a 𝒜) s
    -/
  · exact h.1
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      s : Finset α
      h : And (Not (Membership.mem (Down.compression a 𝒜) s)) (Membership.mem (Down. …
      ⊢ Membership.mem (Down.compression a 𝒜) s
    -/
  · cases h.1 (mem_compression_of_insert_mem_compression h.2)
    /-
      🎉 no goals
    -/


/-- Down-compressing a family doesn't change its size. -/
@[simp]
theorem card_compression (a : α) (𝒜 : Finset (Finset α)) : #(𝓓 a 𝒜) = #𝒜 := by
  rw [compression, card_disjUnion, filter_image,
    card_image_of_injOn ((erase_injOn' _).mono fun s hs => _), ← card_union_of_disjoint]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      ⊢ Eq (Union.union (Finset.filter (fun s => Membership.mem 𝒜 (s.erase a)) 𝒜) (F …
    -/
  · conv_rhs => rw [← filter_union_filter_neg_eq (fun s => (erase s a ∈ 𝒜)) 𝒜]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      𝒜 : Finset (Finset α)
      ⊢ Disjoint (Finset.filter (fun s => Membership.mem 𝒜 (s.erase a)) 𝒜) (Finset.f …
    -/
  · exact disjoint_filter_filter_neg 𝒜 𝒜 (fun s => (erase s a ∈ 𝒜))
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    ⊢ ∀ (s : Finset α), Membership.mem (↑(Finset.filter (fun a_1 => Not (Membershi …
  -/
  intro s hs
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    hs : Membership.mem (↑(Finset.filter (fun a_1 => Not (Membership.mem 𝒜 (a_1.er …
    ⊢ Membership.mem (setOf fun s => Membership.mem s a) s
  -/
  rw [mem_coe, mem_filter] at hs
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    s : Finset α
    hs : And (Membership.mem 𝒜 s) (Not (Membership.mem 𝒜 (s.erase a)))
    ⊢ Membership.mem (setOf fun s => Membership.mem s a) s
  -/
  exact not_imp_comm.1 erase_eq_of_not_mem (ne_of_mem_of_not_mem hs.1 hs.2).symm
  /-
    🎉 no goals
  -/


