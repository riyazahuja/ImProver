/-- A set `s` is a `Subsingleton` if it has at most one element. -/
protected def Subsingleton (s : Set α) : Prop :=
  ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), x = y


theorem Subsingleton.anti (ht : t.Subsingleton) (hst : s ⊆ t) : s.Subsingleton := fun _ hx _ hy =>
  ht (hst hx) (hst hy)


theorem Subsingleton.eq_singleton_of_mem (hs : s.Subsingleton) {x : α} (hx : x ∈ s) : s = {x} :=
  ext fun _ => ⟨fun hy => hs hx hy ▸ mem_singleton _, fun hy => (eq_of_mem_singleton hy).symm ▸ hx⟩


@[simp]
theorem subsingleton_empty : (∅ : Set α).Subsingleton := fun _ => False.elim


@[simp]
theorem subsingleton_singleton {a} : ({a} : Set α).Subsingleton := fun _ hx _ hy =>
  (eq_of_mem_singleton hx).symm ▸ (eq_of_mem_singleton hy).symm ▸ rfl


theorem subsingleton_of_subset_singleton (h : s ⊆ {a}) : s.Subsingleton :=
  subsingleton_singleton.anti h


theorem subsingleton_of_forall_eq (a : α) (h : ∀ b ∈ s, b = a) : s.Subsingleton := fun _ hb _ hc =>
  (h _ hb).trans (h _ hc).symm


theorem subsingleton_iff_singleton {x} (hx : x ∈ s) : s.Subsingleton ↔ s = {x} :=
  ⟨fun h => h.eq_singleton_of_mem hx, fun h => h.symm ▸ subsingleton_singleton⟩


theorem Subsingleton.eq_empty_or_singleton (hs : s.Subsingleton) : s = ∅ ∨ ∃ x, s = {x} :=
  s.eq_empty_or_nonempty.elim Or.inl fun ⟨x, hx⟩ => Or.inr ⟨x, hs.eq_singleton_of_mem hx⟩


theorem Subsingleton.induction_on {p : Set α → Prop} (hs : s.Subsingleton) (he : p ∅)
    (h₁ : ∀ x, p {x}) : p s := by
  /-
    α : Type u
    s : Set α
    p : Set α → Prop
    hs : s.Subsingleton
    he : p EmptyCollection.emptyCollection
    h₁ : ∀ (x : α), p (Singleton.singleton x)
    ⊢ p s
  -/
  rcases hs.eq_empty_or_singleton with (rfl | ⟨x, rfl⟩)
  /-
    case inl
    α : Type u
    p : Set α → Prop
    he : p EmptyCollection.emptyCollection
    h₁ : ∀ (x : α), p (Singleton.singleton x)
    hs : EmptyCollection.emptyCollection.Subsingleton
    ⊢ p EmptyCollection.emptyCollection
  -/
  exacts [he, h₁ _]
  /-
    🎉 no goals
  -/


theorem subsingleton_univ [Subsingleton α] : (univ : Set α).Subsingleton := fun x _ y _ =>
  Subsingleton.elim x y


theorem subsingleton_of_univ_subsingleton (h : (univ : Set α).Subsingleton) : Subsingleton α :=
  ⟨fun a b => h (mem_univ a) (mem_univ b)⟩


@[simp]
theorem subsingleton_univ_iff : (univ : Set α).Subsingleton ↔ Subsingleton α :=
  ⟨subsingleton_of_univ_subsingleton, fun h => @subsingleton_univ _ h⟩


lemma Subsingleton.inter_singleton : (s ∩ {a}).Subsingleton :=
  Set.subsingleton_of_subset_singleton Set.inter_subset_right


lemma Subsingleton.singleton_inter : ({a} ∩ s).Subsingleton :=
  Set.subsingleton_of_subset_singleton Set.inter_subset_left


theorem subsingleton_of_subsingleton [Subsingleton α] {s : Set α} : Set.Subsingleton s :=
  subsingleton_univ.anti (subset_univ s)


theorem subsingleton_isTop (α : Type*) [PartialOrder α] : Set.Subsingleton { x : α | IsTop x } :=
  fun x hx _ hy => hx.isMax.eq_of_le (hy x)


theorem subsingleton_isBot (α : Type*) [PartialOrder α] : Set.Subsingleton { x : α | IsBot x } :=
  fun x hx _ hy => hx.isMin.eq_of_ge (hy x)


theorem exists_eq_singleton_iff_nonempty_subsingleton :
    (∃ a : α, s = {a}) ↔ s.Nonempty ∧ s.Subsingleton := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Exists fun a => Eq s (Singleton.singleton a)) (And s.Nonempty s.Subsing …
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u
      s : Set α
      ⊢ (Exists fun a => Eq s (Singleton.singleton a)) → And s.Nonempty s.Subsingleton
    -/
  · rintro ⟨a, rfl⟩
    /-
      case refine_1.intro
      α : Type u
      a : α
      ⊢ And (Singleton.singleton a).Nonempty (Singleton.singleton a).Subsingleton
    -/
    exact ⟨singleton_nonempty a, subsingleton_singleton⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      s : Set α
      h : And s.Nonempty s.Subsingleton
      ⊢ Exists fun a => Eq s (Singleton.singleton a)
    -/
  · exact h.2.eq_empty_or_singleton.resolve_left h.1.ne_empty
    /-
      🎉 no goals
    -/


/-- `s`, coerced to a type, is a subsingleton type if and only if `s` is a subsingleton set. -/
@[simp, norm_cast]
theorem subsingleton_coe (s : Set α) : Subsingleton s ↔ s.Subsingleton := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Subsingleton ↑s) s.Subsingleton
  -/
  constructor
    /-
      case mp
      α : Type u
      s : Set α
      ⊢ Subsingleton ↑s → s.Subsingleton
    -/
  · refine fun h => fun a ha b hb => ?_
    /-
      case mp
      α : Type u
      s : Set α
      h : Subsingleton ↑s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      ⊢ Eq a b
    -/
    exact SetCoe.ext_iff.2 (@Subsingleton.elim s h ⟨a, ha⟩ ⟨b, hb⟩)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      s : Set α
      ⊢ s.Subsingleton → Subsingleton ↑s
    -/
  · exact fun h => Subsingleton.intro fun a b => SetCoe.ext (h a.property b.property)
    /-
      🎉 no goals
    -/


theorem Subsingleton.coe_sort {s : Set α} : s.Subsingleton → Subsingleton s :=
  s.subsingleton_coe.2


/-- The `coe_sort` of a set `s` in a subsingleton type is a subsingleton.
For the corresponding result for `Subtype`, see `subtype.subsingleton`. -/
instance subsingleton_coe_of_subsingleton [Subsingleton α] {s : Set α} : Subsingleton s := by
  /-
    α : Type u
    a : α
    s✝ t : Set α
    inst✝ : Subsingleton α
    s : Set α
    ⊢ Subsingleton ↑s
  -/
  rw [s.subsingleton_coe]
  /-
    α : Type u
    a : α
    s✝ t : Set α
    inst✝ : Subsingleton α
    s : Set α
    ⊢ s.Subsingleton
  -/
  exact subsingleton_of_subsingleton
  /-
    🎉 no goals
  -/


/-- A set `s` is `Set.Nontrivial` if it has at least two distinct elements. -/
protected def Nontrivial (s : Set α) : Prop :=
  ∃ x ∈ s, ∃ y ∈ s, x ≠ y


theorem nontrivial_of_mem_mem_ne {x y} (hx : x ∈ s) (hy : y ∈ s) (hxy : x ≠ y) : s.Nontrivial :=
  ⟨x, hx, y, hy, hxy⟩

-- Porting note: following the pattern for `Exists`, we have renamed `some` to `choose`.


/-- Extract witnesses from s.nontrivial. This function might be used instead of case analysis on the
argument. Note that it makes a proof depend on the classical.choice axiom. -/
protected noncomputable def Nontrivial.choose (hs : s.Nontrivial) : α × α :=
  (Exists.choose hs, hs.choose_spec.right.choose)


protected theorem Nontrivial.choose_fst_mem (hs : s.Nontrivial) : hs.choose.fst ∈ s :=
  hs.choose_spec.left


protected theorem Nontrivial.choose_snd_mem (hs : s.Nontrivial) : hs.choose.snd ∈ s :=
  hs.choose_spec.right.choose_spec.left


protected theorem Nontrivial.choose_fst_ne_choose_snd (hs : s.Nontrivial) :
    hs.choose.fst ≠ hs.choose.snd :=
  hs.choose_spec.right.choose_spec.right


theorem Nontrivial.mono (hs : s.Nontrivial) (hst : s ⊆ t) : t.Nontrivial :=
  let ⟨x, hx, y, hy, hxy⟩ := hs
  ⟨x, hst hx, y, hst hy, hxy⟩


theorem nontrivial_pair {x y} (hxy : x ≠ y) : ({x, y} : Set α).Nontrivial :=
  ⟨x, mem_insert _ _, y, mem_insert_of_mem _ (mem_singleton _), hxy⟩


theorem nontrivial_of_pair_subset {x y} (hxy : x ≠ y) (h : {x, y} ⊆ s) : s.Nontrivial :=
  (nontrivial_pair hxy).mono h


theorem Nontrivial.pair_subset (hs : s.Nontrivial) : ∃ x y, x ≠ y ∧ {x, y} ⊆ s :=
  let ⟨x, hx, y, hy, hxy⟩ := hs
  ⟨x, y, hxy, insert_subset hx <| singleton_subset_iff.2 hy⟩


theorem nontrivial_iff_pair_subset : s.Nontrivial ↔ ∃ x y, x ≠ y ∧ {x, y} ⊆ s :=
  ⟨Nontrivial.pair_subset, fun H =>
    let ⟨_, _, hxy, h⟩ := H
    nontrivial_of_pair_subset hxy h⟩


theorem nontrivial_of_exists_ne {x} (hx : x ∈ s) (h : ∃ y ∈ s, y ≠ x) : s.Nontrivial :=
  let ⟨y, hy, hyx⟩ := h
  ⟨y, hy, x, hx, hyx⟩


theorem Nontrivial.exists_ne (hs : s.Nontrivial) (z) : ∃ x ∈ s, x ≠ z := by
  /-
    α : Type u
    s : Set α
    hs : s.Nontrivial
    z : α
    ⊢ Exists fun x => And (Membership.mem s x) (Ne x z)
  -/
  by_contra! H
  /-
    α : Type u
    s : Set α
    hs : s.Nontrivial
    z : α
    H : ∀ (x : α), Membership.mem s x → Eq x z
    ⊢ False
  -/
  rcases hs with ⟨x, hx, y, hy, hxy⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    s : Set α
    z : α
    H : ∀ (x : α), Membership.mem s x → Eq x z
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ False
  -/
  rw [H x hx, H y hy] at hxy
  /-
    case intro.intro.intro.intro
    α : Type u
    s : Set α
    z : α
    H : ∀ (x : α), Membership.mem s x → Eq x z
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : Ne z z
    ⊢ False
  -/
  exact hxy rfl
  /-
    🎉 no goals
  -/


theorem nontrivial_iff_exists_ne {x} (hx : x ∈ s) : s.Nontrivial ↔ ∃ y ∈ s, y ≠ x :=
  ⟨fun H => H.exists_ne _, nontrivial_of_exists_ne hx⟩


theorem nontrivial_of_lt [Preorder α] {x y} (hx : x ∈ s) (hy : y ∈ s) (hxy : x < y) :
    s.Nontrivial :=
  ⟨x, hx, y, hy, ne_of_lt hxy⟩


theorem nontrivial_of_exists_lt [Preorder α]
    (H : ∃ᵉ (x ∈ s) (y ∈ s), x < y) : s.Nontrivial :=
  let ⟨_, hx, _, hy, hxy⟩ := H
  nontrivial_of_lt hx hy hxy


theorem Nontrivial.exists_lt [LinearOrder α] (hs : s.Nontrivial) : ∃ᵉ (x ∈ s) (y ∈ s), x < y :=
  let ⟨x, hx, y, hy, hxy⟩ := hs
  Or.elim (lt_or_gt_of_ne hxy) (fun H => ⟨x, hx, y, hy, H⟩) fun H => ⟨y, hy, x, hx, H⟩


theorem nontrivial_iff_exists_lt [LinearOrder α] :
    s.Nontrivial ↔ ∃ᵉ (x ∈ s) (y ∈ s), x < y :=
  ⟨Nontrivial.exists_lt, nontrivial_of_exists_lt⟩


protected theorem Nontrivial.nonempty (hs : s.Nontrivial) : s.Nonempty :=
  let ⟨x, hx, _⟩ := hs
  ⟨x, hx⟩


protected theorem Nontrivial.ne_empty (hs : s.Nontrivial) : s ≠ ∅ :=
  hs.nonempty.ne_empty


theorem Nontrivial.not_subset_empty (hs : s.Nontrivial) : ¬s ⊆ ∅ :=
  hs.nonempty.not_subset_empty


@[simp]
theorem not_nontrivial_empty : ¬(∅ : Set α).Nontrivial := fun h => h.ne_empty rfl


@[simp]
theorem not_nontrivial_singleton {x} : ¬({x} : Set α).Nontrivial := fun H => by
  /-
    α : Type u
    x : α
    H : (Singleton.singleton x).Nontrivial
    ⊢ False
  -/
  rw [nontrivial_iff_exists_ne (mem_singleton x)] at H
  /-
    α : Type u
    x : α
    H : Exists fun y => And (Membership.mem (Singleton.singleton x) y) (Ne y x)
    ⊢ False
  -/
  let ⟨y, hy, hya⟩ := H
  /-
    α : Type u
    x : α
    H : Exists fun y => And (Membership.mem (Singleton.singleton x) y) (Ne y x)
    y : α
    hy : Membership.mem (Singleton.singleton x) y
    hya : Ne y x
    ⊢ False
  -/
  exact hya (mem_singleton_iff.1 hy)
  /-
    🎉 no goals
  -/


theorem Nontrivial.ne_singleton {x} (hs : s.Nontrivial) : s ≠ {x} := fun H => by
  /-
    α : Type u
    s : Set α
    x : α
    hs : s.Nontrivial
    H : Eq s (Singleton.singleton x)
    ⊢ False
  -/
  rw [H] at hs
  /-
    α : Type u
    s : Set α
    x : α
    hs : (Singleton.singleton x).Nontrivial
    H : Eq s (Singleton.singleton x)
    ⊢ False
  -/
  exact not_nontrivial_singleton hs
  /-
    🎉 no goals
  -/


theorem Nontrivial.not_subset_singleton {x} (hs : s.Nontrivial) : ¬s ⊆ {x} :=
  (not_congr subset_singleton_iff_eq).2 (not_or_intro hs.ne_empty hs.ne_singleton)


theorem nontrivial_univ [Nontrivial α] : (univ : Set α).Nontrivial :=
  let ⟨x, y, hxy⟩ := exists_pair_ne α
  ⟨x, mem_univ _, y, mem_univ _, hxy⟩


theorem nontrivial_of_univ_nontrivial (h : (univ : Set α).Nontrivial) : Nontrivial α :=
  let ⟨x, _, y, _, hxy⟩ := h
  ⟨⟨x, y, hxy⟩⟩


@[simp]
theorem nontrivial_univ_iff : (univ : Set α).Nontrivial ↔ Nontrivial α :=
  ⟨nontrivial_of_univ_nontrivial, fun h => @nontrivial_univ _ h⟩


theorem nontrivial_of_nontrivial (hs : s.Nontrivial) : Nontrivial α :=
  let ⟨x, _, y, _, hxy⟩ := hs
  ⟨⟨x, y, hxy⟩⟩


/-- `s`, coerced to a type, is a nontrivial type if and only if `s` is a nontrivial set. -/
@[simp, norm_cast]
theorem nontrivial_coe_sort {s : Set α} : Nontrivial s ↔ s.Nontrivial := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Nontrivial ↑s) s.Nontrivial
  -/
  simp [← nontrivial_univ_iff, Set.Nontrivial]
  /-
    🎉 no goals
  -/


alias ⟨_, Nontrivial.coe_sort⟩ := nontrivial_coe_sort


/-- A type with a set `s` whose `coe_sort` is a nontrivial type is nontrivial.
For the corresponding result for `Subtype`, see `Subtype.nontrivial_iff_exists_ne`. -/
theorem nontrivial_of_nontrivial_coe (hs : Nontrivial s) : Nontrivial α :=
  nontrivial_of_nontrivial <| nontrivial_coe_sort.1 hs


theorem nontrivial_mono {α : Type*} {s t : Set α} (hst : s ⊆ t) (hs : Nontrivial s) :
    Nontrivial t :=
  Nontrivial.coe_sort <| (nontrivial_coe_sort.1 hs).mono hst


@[simp]
theorem not_subsingleton_iff : ¬s.Subsingleton ↔ s.Nontrivial := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Not s.Subsingleton) s.Nontrivial
  -/
  simp_rw [Set.Subsingleton, Set.Nontrivial, not_forall, exists_prop]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_nontrivial_iff : ¬s.Nontrivial ↔ s.Subsingleton :=
  Iff.not_left not_subsingleton_iff.symm


alias ⟨_, Subsingleton.not_nontrivial⟩ := not_nontrivial_iff


alias ⟨_, Nontrivial.not_subsingleton⟩ := not_subsingleton_iff


protected lemma subsingleton_or_nontrivial (s : Set α) : s.Subsingleton ∨ s.Nontrivial := by
  /-
    α : Type u
    s : Set α
    ⊢ Or s.Subsingleton s.Nontrivial
  -/
  simp [or_iff_not_imp_right]
  /-
    🎉 no goals
  -/


lemma eq_singleton_or_nontrivial (ha : a ∈ s) : s = {a} ∨ s.Nontrivial := by
  /-
    α : Type u
    a : α
    s : Set α
    ha : Membership.mem s a
    ⊢ Or (Eq s (Singleton.singleton a)) s.Nontrivial
  -/
  rw [← subsingleton_iff_singleton ha]; exact s.subsingleton_or_nontrivial
                                        /-
                                          🎉 no goals
                                        -/


lemma nontrivial_iff_ne_singleton (ha : a ∈ s) : s.Nontrivial ↔ s ≠ {a} :=
  ⟨Nontrivial.ne_singleton, (eq_singleton_or_nontrivial ha).resolve_left⟩


lemma Nonempty.exists_eq_singleton_or_nontrivial : s.Nonempty → (∃ a, s = {a}) ∨ s.Nontrivial :=
  fun ⟨a, ha⟩ ↦ (eq_singleton_or_nontrivial ha).imp_left <| Exists.intro a


theorem univ_eq_true_false : univ = ({True, False} : Set Prop) :=
  Eq.symm <| eq_univ_of_forall fun x => by
    /-
      x : Prop
      ⊢ Membership.mem (Insert.insert True (Singleton.singleton False)) x
    -/
    rw [mem_insert_iff, mem_singleton_iff]
    /-
      x : Prop
      ⊢ Or (Eq x True) (Eq x False)
    -/
    exact Classical.propComplete x
    /-
      🎉 no goals
    -/


protected theorem Subsingleton.monotoneOn (h : s.Subsingleton) : MonotoneOn f s :=
  fun _ ha _ hb _ => (congr_arg _ (h ha hb)).le


protected theorem Subsingleton.antitoneOn (h : s.Subsingleton) : AntitoneOn f s :=
  fun _ ha _ hb _ => (congr_arg _ (h hb ha)).le


protected theorem Subsingleton.strictMonoOn (h : s.Subsingleton) : StrictMonoOn f s :=
  fun _ ha _ hb hlt => (hlt.ne (h ha hb)).elim


protected theorem Subsingleton.strictAntiOn (h : s.Subsingleton) : StrictAntiOn f s :=
  fun _ ha _ hb hlt => (hlt.ne (h ha hb)).elim


@[simp]
theorem monotoneOn_singleton : MonotoneOn f {a} :=
  subsingleton_singleton.monotoneOn f


@[simp]
theorem antitoneOn_singleton : AntitoneOn f {a} :=
  subsingleton_singleton.antitoneOn f


@[simp]
theorem strictMonoOn_singleton : StrictMonoOn f {a} :=
  subsingleton_singleton.strictMonoOn f


@[simp]
theorem strictAntiOn_singleton : StrictAntiOn f {a} :=
  subsingleton_singleton.strictAntiOn f


