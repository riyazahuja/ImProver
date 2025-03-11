/-- `{a} : Finset a` is the set `{a}` containing `a` and nothing else.

This differs from `insert a ∅` in that it does not require a `DecidableEq` instance for `α`.
-/
instance : Singleton α (Finset α) :=
  ⟨fun a => ⟨{a}, nodup_singleton a⟩⟩


@[simp]
theorem singleton_val (a : α) : ({a} : Finset α).1 = {a} :=
  rfl


@[simp]
theorem mem_singleton {a b : α} : b ∈ ({a} : Finset α) ↔ b = a :=
  Multiset.mem_singleton


theorem eq_of_mem_singleton {x y : α} (h : x ∈ ({y} : Finset α)) : x = y :=
  mem_singleton.1 h


theorem not_mem_singleton {a b : α} : a ∉ ({b} : Finset α) ↔ a ≠ b :=
  not_congr mem_singleton


theorem mem_singleton_self (a : α) : a ∈ ({a} : Finset α) :=
  -- Porting note: was `Or.inl rfl`
  mem_singleton.mpr rfl


@[simp]
theorem val_eq_singleton_iff {a : α} {s : Finset α} : s.val = {a} ↔ s = {a} := by
  /-
    α : Type u_1
    a : α
    s : Finset α
    ⊢ Iff (Eq s.val (Singleton.singleton a)) (Eq s (Singleton.singleton a))
  -/
  rw [← val_inj]
  /-
    α : Type u_1
    a : α
    s : Finset α
    ⊢ Iff (Eq s.val (Singleton.singleton a)) (Eq s.val (Singleton.singleton a).val)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem singleton_injective : Injective (singleton : α → Finset α) := fun _a _b h =>
  mem_singleton.1 (h ▸ mem_singleton_self _)


@[simp]
theorem singleton_inj : ({a} : Finset α) = {b} ↔ a = b :=
  singleton_injective.eq_iff


@[simp, aesop safe apply (rule_sets := [finsetNonempty])]
theorem singleton_nonempty (a : α) : ({a} : Finset α).Nonempty :=
  ⟨a, mem_singleton_self a⟩


@[simp]
theorem singleton_ne_empty (a : α) : ({a} : Finset α) ≠ ∅ :=
  (singleton_nonempty a).ne_empty


theorem empty_ssubset_singleton : (∅ : Finset α) ⊂ {a} :=
  (singleton_nonempty _).empty_ssubset


@[simp, norm_cast]
theorem coe_singleton (a : α) : (({a} : Finset α) : Set α) = {a} := by
  /-
    α : Type u_1
    a : α
    ⊢ Eq (↑(Singleton.singleton a)) (Singleton.singleton a)
  -/
  ext
  /-
    case h
    α : Type u_1
    a x✝ : α
    ⊢ Iff (Membership.mem (↑(Singleton.singleton a)) x✝) (Membership.mem (Singleto …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_eq_singleton {s : Finset α} {a : α} : (s : Set α) = {a} ↔ s = {a} := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (Eq (↑s) (Singleton.singleton a)) (Eq s (Singleton.singleton a))
  -/
  rw [← coe_singleton, coe_inj]
  /-
    🎉 no goals
  -/


@[norm_cast]
                                                               /-
                                                                 α : Type u_1
                                                                 s : Finset α
                                                                 a : α
                                                                 ⊢ Iff (HasSubset.Subset (↑s) (Singleton.singleton a)) (HasSubset.Subset s (Sin …
                                                               -/
lemma coe_subset_singleton : (s : Set α) ⊆ {a} ↔ s ⊆ {a} := by rw [← coe_subset, coe_singleton]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[norm_cast]
                                                               /-
                                                                 α : Type u_1
                                                                 s : Finset α
                                                                 a : α
                                                                 ⊢ Iff (HasSubset.Subset (Singleton.singleton a) ↑s) (HasSubset.Subset (Singlet …
                                                               -/
lemma singleton_subset_coe : {a} ⊆ (s : Set α) ↔ {a} ⊆ s := by rw [← coe_subset, coe_singleton]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem eq_singleton_iff_unique_mem {s : Finset α} {a : α} : s = {a} ↔ a ∈ s ∧ ∀ x ∈ s, x = a := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (Eq s (Singleton.singleton a)) (And (Membership.mem s a) (∀ (x : α), Mem …
  -/
  constructor <;> intro t
    /-
      case mp
      α : Type u_1
      s : Finset α
      a : α
      t : Eq s (Singleton.singleton a)
      ⊢ And (Membership.mem s a) (∀ (x : α), Membership.mem s x → Eq x a)
    -/
  · rw [t]
    /-
      case mp
      α : Type u_1
      s : Finset α
      a : α
      t : Eq s (Singleton.singleton a)
      ⊢ And (Membership.mem (Singleton.singleton a) a) (∀ (x : α), Membership.mem (S …
    -/
    exact ⟨Finset.mem_singleton_self _, fun _ => Finset.mem_singleton.1⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Finset α
      a : α
      t : And (Membership.mem s a) (∀ (x : α), Membership.mem s x → Eq x a)
      ⊢ Eq s (Singleton.singleton a)
    -/
  · ext
    /-
      case mpr.h
      α : Type u_1
      s : Finset α
      a : α
      t : And (Membership.mem s a) (∀ (x : α), Membership.mem s x → Eq x a)
      a✝ : α
      ⊢ Iff (Membership.mem s a✝) (Membership.mem (Singleton.singleton a) a✝)
    -/
    rw [Finset.mem_singleton]
    /-
      case mpr.h
      α : Type u_1
      s : Finset α
      a : α
      t : And (Membership.mem s a) (∀ (x : α), Membership.mem s x → Eq x a)
      a✝ : α
      ⊢ Iff (Membership.mem s a✝) (Eq a✝ a)
    -/
    exact ⟨t.right _, fun r => r.symm ▸ t.left⟩
    /-
      🎉 no goals
    -/


theorem eq_singleton_iff_nonempty_unique_mem {s : Finset α} {a : α} :
    s = {a} ↔ s.Nonempty ∧ ∀ x ∈ s, x = a := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (Eq s (Singleton.singleton a)) (And s.Nonempty (∀ (x : α), Membership.me …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s : Finset α
      a : α
      ⊢ Eq s (Singleton.singleton a) → And s.Nonempty (∀ (x : α), Membership.mem s x …
    -/
  · rintro rfl
    /-
      case mp
      α : Type u_1
      a : α
      ⊢ And (Singleton.singleton a).Nonempty (∀ (x : α), Membership.mem (Singleton.s …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Finset α
      a : α
      ⊢ And s.Nonempty (∀ (x : α), Membership.mem s x → Eq x a) → Eq s (Singleton.si …
    -/
  · rintro ⟨hne, h_uniq⟩
    /-
      case mpr.intro
      α : Type u_1
      s : Finset α
      a : α
      hne : s.Nonempty
      h_uniq : ∀ (x : α), Membership.mem s x → Eq x a
      ⊢ Eq s (Singleton.singleton a)
    -/
    rw [eq_singleton_iff_unique_mem]
    /-
      case mpr.intro
      α : Type u_1
      s : Finset α
      a : α
      hne : s.Nonempty
      h_uniq : ∀ (x : α), Membership.mem s x → Eq x a
      ⊢ And (Membership.mem s a) (∀ (x : α), Membership.mem s x → Eq x a)
    -/
    refine ⟨?_, h_uniq⟩
    /-
      case mpr.intro
      α : Type u_1
      s : Finset α
      a : α
      hne : s.Nonempty
      h_uniq : ∀ (x : α), Membership.mem s x → Eq x a
      ⊢ Membership.mem s a
    -/
    rw [← h_uniq hne.choose hne.choose_spec]
    /-
      case mpr.intro
      α : Type u_1
      s : Finset α
      a : α
      hne : s.Nonempty
      h_uniq : ∀ (x : α), Membership.mem s x → Eq x a
      ⊢ Membership.mem s (Exists.choose hne)
    -/
    exact hne.choose_spec
    /-
      🎉 no goals
    -/


theorem nonempty_iff_eq_singleton_default [Unique α] {s : Finset α} :
    s.Nonempty ↔ s = {default} := by
  /-
    α : Type u_1
    inst✝ : Unique α
    s : Finset α
    ⊢ Iff s.Nonempty (Eq s (Singleton.singleton Inhabited.default))
  -/
  simp [eq_singleton_iff_nonempty_unique_mem, eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


alias ⟨Nonempty.eq_singleton_default, _⟩ := nonempty_iff_eq_singleton_default


theorem singleton_iff_unique_mem (s : Finset α) : (∃ a, s = {a}) ↔ ∃! a, a ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Exists fun a => Eq s (Singleton.singleton a)) (ExistsUnique fun a => Me …
  -/
  simp only [eq_singleton_iff_unique_mem, ExistsUnique]
  /-
    🎉 no goals
  -/


theorem singleton_subset_set_iff {s : Set α} {a : α} : ↑({a} : Finset α) ⊆ s ↔ a ∈ s := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ⊢ Iff (HasSubset.Subset (↑(Singleton.singleton a)) s) (Membership.mem s a)
  -/
  rw [coe_singleton, Set.singleton_subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_subset_iff {s : Finset α} {a : α} : {a} ⊆ s ↔ a ∈ s :=
  singleton_subset_set_iff


@[simp]
theorem subset_singleton_iff {s : Finset α} {a : α} : s ⊆ {a} ↔ s = ∅ ∨ s = {a} := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (HasSubset.Subset s (Singleton.singleton a)) (Or (Eq s EmptyCollection.e …
  -/
  rw [← coe_subset, coe_singleton, Set.subset_singleton_iff_eq, coe_eq_empty, coe_eq_singleton]
  /-
    🎉 no goals
  -/


                                                                          /-
                                                                            α : Type u_1
                                                                            a b : α
                                                                            ⊢ Iff (HasSubset.Subset (Singleton.singleton a) (Singleton.singleton b)) (Eq a …
                                                                          -/
theorem singleton_subset_singleton : ({a} : Finset α) ⊆ {b} ↔ a = b := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


protected theorem Nonempty.subset_singleton_iff {s : Finset α} {a : α} (h : s.Nonempty) :
    s ⊆ {a} ↔ s = {a} :=
  subset_singleton_iff.trans <| or_iff_right h.ne_empty


theorem subset_singleton_iff' {s : Finset α} {a : α} : s ⊆ {a} ↔ ∀ b ∈ s, b = a :=
  forall₂_congr fun _ _ => mem_singleton


@[simp]
theorem ssubset_singleton_iff {s : Finset α} {a : α} : s ⊂ {a} ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (HasSSubset.SSubset s (Singleton.singleton a)) (Eq s EmptyCollection.emp …
  -/
  rw [← coe_ssubset, coe_singleton, Set.ssubset_singleton_iff, coe_eq_empty]
  /-
    🎉 no goals
  -/


theorem eq_empty_of_ssubset_singleton {s : Finset α} {x : α} (hs : s ⊂ {x}) : s = ∅ :=
  ssubset_singleton_iff.1 hs


/-- A finset is nontrivial if it has at least two elements. -/
protected abbrev Nontrivial (s : Finset α) : Prop := (s : Set α).Nontrivial


nonrec lemma Nontrivial.nonempty (hs : s.Nontrivial) : s.Nonempty := hs.nonempty


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   ⊢ Not EmptyCollection.emptyCollection.Nontrivial
                                                                 -/
theorem not_nontrivial_empty : ¬ (∅ : Finset α).Nontrivial := by simp [Finset.Nontrivial]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
                                                                       /-
                                                                         α : Type u_1
                                                                         a : α
                                                                         ⊢ Not (Singleton.singleton a).Nontrivial
                                                                       -/
theorem not_nontrivial_singleton : ¬ ({a} : Finset α).Nontrivial := by simp [Finset.Nontrivial]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem Nontrivial.ne_singleton (hs : s.Nontrivial) : s ≠ {a} := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    hs : s.Nontrivial
    ⊢ Ne s (Singleton.singleton a)
  -/
  rintro rfl; exact not_nontrivial_singleton hs
              /-
                🎉 no goals
              -/


nonrec lemma Nontrivial.exists_ne (hs : s.Nontrivial) (a : α) : ∃ b ∈ s, b ≠ a := hs.exists_ne _


theorem eq_singleton_or_nontrivial (ha : a ∈ s) : s = {a} ∨ s.Nontrivial := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ Or (Eq s (Singleton.singleton a)) s.Nontrivial
  -/
  rw [← coe_eq_singleton]; exact Set.eq_singleton_or_nontrivial ha
                           /-
                             🎉 no goals
                           -/


theorem nontrivial_iff_ne_singleton (ha : a ∈ s) : s.Nontrivial ↔ s ≠ {a} :=
  ⟨Nontrivial.ne_singleton, (eq_singleton_or_nontrivial ha).resolve_left⟩


theorem Nonempty.exists_eq_singleton_or_nontrivial : s.Nonempty → (∃ a, s = {a}) ∨ s.Nontrivial :=
  fun ⟨a, ha⟩ => (eq_singleton_or_nontrivial ha).imp_left <| Exists.intro a


instance instNontrivial [Nonempty α] : Nontrivial (Finset α) :=
  ‹Nonempty α›.elim fun a => ⟨⟨{a}, ∅, singleton_ne_empty _⟩⟩


instance [IsEmpty α] : Unique (Finset α) where
  default := ∅
  uniq _ := eq_empty_of_forall_not_mem isEmptyElim


instance (i : α) : Unique ({i} : Finset α) where
  default := ⟨i, mem_singleton_self i⟩
  uniq j := Subtype.ext <| mem_singleton.mp j.2


@[simp]
lemma default_singleton (i : α) : ((default : ({i} : Finset α)) : α) = i := rfl


instance Nontrivial.instDecidablePred [DecidableEq α] :
    DecidablePred (Finset.Nontrivial (α := α)) :=
  inferInstanceAs (DecidablePred fun s ↦ ∃ a ∈ s, ∃ b ∈ s, a ≠ b)


/-- `cons a s h` is the set `{a} ∪ s` containing `a` and the elements of `s`. It is the same as
`insert a s` when it is defined, but unlike `insert a s` it does not require `DecidableEq α`,
and the union is guaranteed to be disjoint. -/
def cons (a : α) (s : Finset α) (h : a ∉ s) : Finset α :=
  ⟨a ::ₘ s.1, nodup_cons.2 ⟨h, s.2⟩⟩


@[simp]
theorem mem_cons {h} : b ∈ s.cons a h ↔ b = a ∨ b ∈ s :=
  Multiset.mem_cons


theorem mem_cons_of_mem {a b : α} {s : Finset α} {hb : b ∉ s} (ha : a ∈ s) : a ∈ cons b s hb :=
  Multiset.mem_cons_of_mem ha


theorem mem_cons_self (a : α) (s : Finset α) {h} : a ∈ cons a s h :=
  Multiset.mem_cons_self _ _


@[simp]
theorem cons_val (h : a ∉ s) : (cons a s h).1 = a ::ₘ s.1 :=
  rfl


theorem eq_of_mem_cons_of_not_mem (has : a ∉ s) (h : b ∈ cons a s has) (hb : b ∉ s) : b = a :=
  (mem_cons.1 h).resolve_right hb


theorem mem_of_mem_cons_of_ne {s : Finset α} {a : α} {has} {i : α}
    (hi : i ∈ cons a s has) (hia : i ≠ a) : i ∈ s :=
  (mem_cons.1 hi).resolve_left hia


theorem forall_mem_cons (h : a ∉ s) (p : α → Prop) :
    (∀ x, x ∈ cons a s h → p x) ↔ p a ∧ ∀ x, x ∈ s → p x := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    h : Not (Membership.mem s a)
    p : α → Prop
    ⊢ Iff (∀ (x : α), Membership.mem (Finset.cons a s h) x → p x) (And (p a) (∀ (x …
  -/
  simp only [mem_cons, or_imp, forall_and, forall_eq]
  /-
    🎉 no goals
  -/


/-- Useful in proofs by induction. -/
theorem forall_of_forall_cons {p : α → Prop} {h : a ∉ s} (H : ∀ x, x ∈ cons a s h → p x) (x)
    (h : x ∈ s) : p x :=
  H _ <| mem_cons.2 <| Or.inr h


@[simp]
theorem mk_cons {s : Multiset α} (h : (a ::ₘ s).Nodup) :
    (⟨a ::ₘ s, h⟩ : Finset α) = cons a ⟨s, (nodup_cons.1 h).2⟩ (nodup_cons.1 h).1 :=
  rfl


@[simp]
theorem cons_empty (a : α) : cons a ∅ (not_mem_empty _) = {a} := rfl


@[simp, aesop safe apply (rule_sets := [finsetNonempty])]
theorem cons_nonempty (h : a ∉ s) : (cons a s h).Nonempty :=
  ⟨a, mem_cons.2 <| Or.inl rfl⟩


@[deprecated (since := "2024-09-19")] alias nonempty_cons := cons_nonempty


@[simp] theorem cons_ne_empty (h : a ∉ s) : cons a s h ≠ ∅ := (cons_nonempty _).ne_empty


@[simp]
theorem nonempty_mk {m : Multiset α} {hm} : (⟨m, hm⟩ : Finset α).Nonempty ↔ m ≠ 0 := by
  /-
    α : Type u_1
    m : Multiset α
    hm : m.Nodup
    ⊢ Iff { val := m, nodup := hm }.Nonempty (Ne m 0)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction m using Multiset.induction_on <;> simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem coe_cons {a s h} : (@cons α a s h : Set α) = insert a (s : Set α) := by
  /-
    α : Type u_1
    a : α
    s : Finset α
    h : Not (Membership.mem s a)
    ⊢ Eq (↑(Finset.cons a s h)) (Insert.insert a ↑s)
  -/
  ext
  /-
    case h
    α : Type u_1
    a : α
    s : Finset α
    h : Not (Membership.mem s a)
    x✝ : α
    ⊢ Iff (Membership.mem (↑(Finset.cons a s h)) x✝) (Membership.mem (Insert.inser …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem subset_cons (h : a ∉ s) : s ⊆ s.cons a h :=
  Multiset.subset_cons _ _


theorem ssubset_cons (h : a ∉ s) : s ⊂ s.cons a h :=
  Multiset.ssubset_cons h


theorem cons_subset {h : a ∉ s} : s.cons a h ⊆ t ↔ a ∈ t ∧ s ⊆ t :=
  Multiset.cons_subset


@[simp]
theorem cons_subset_cons {hs ht} : s.cons a hs ⊆ t.cons a ht ↔ s ⊆ t := by
  /-
    α : Type u_1
    s t : Finset α
    a : α
    hs : Not (Membership.mem s a)
    ht : Not (Membership.mem t a)
    ⊢ Iff (HasSubset.Subset (Finset.cons a s hs) (Finset.cons a t ht)) (HasSubset. …
  -/
  rwa [← coe_subset, coe_cons, coe_cons, Set.insert_subset_insert_iff, coe_subset]
  /-
    🎉 no goals
  -/


theorem ssubset_iff_exists_cons_subset : s ⊂ t ↔ ∃ (a : _) (h : a ∉ s), s.cons a h ⊆ t := by
  /-
    α : Type u_1
    s t : Finset α
    ⊢ Iff (HasSSubset.SSubset s t) (Exists fun a => Exists fun h => HasSubset.Subs …
  -/
  refine ⟨fun h => ?_, fun ⟨a, ha, h⟩ => ssubset_of_ssubset_of_subset (ssubset_cons _) h⟩
  /-
    α : Type u_1
    s t : Finset α
    h : HasSSubset.SSubset s t
    ⊢ Exists fun a => Exists fun h => HasSubset.Subset (Finset.cons a s h) t
  -/
  obtain ⟨a, hs, ht⟩ := not_subset.1 h.2
  /-
    case intro.intro
    α : Type u_1
    s t : Finset α
    h : HasSSubset.SSubset s t
    a : α
    hs : Membership.mem t a
    ht : Not (Membership.mem s a)
    ⊢ Exists fun a => Exists fun h => HasSubset.Subset (Finset.cons a s h) t
  -/
  exact ⟨a, ht, cons_subset.2 ⟨hs, h.subset⟩⟩
  /-
    🎉 no goals
  -/


theorem cons_swap (hb : b ∉ s) (ha : a ∉ s.cons b hb) :
    (s.cons b hb).cons a ha = (s.cons a fun h ↦ ha (mem_cons.mpr (.inr h))).cons b fun h ↦
      ha (mem_cons.mpr (.inl ((mem_cons.mp h).elim symm (fun h ↦ False.elim (hb h))))) :=
  eq_of_veq <| Multiset.cons_swap a b s.val


/-- Split the added element of cons off a Pi type. -/
@[simps!]
def consPiProd (f : α → Type*) (has : a ∉ s) (x : Π i ∈ cons a s has, f i) : f a × Π i ∈ s, f i :=
  (x a (mem_cons_self a s), fun i hi => x i (mem_cons_of_mem hi))


/-- Combine a product with a pi type to pi of cons. -/
def prodPiCons [DecidableEq α] (f : α → Type*) {a : α} (has : a ∉ s) (x : f a × Π i ∈ s, f i) :
    (Π i ∈ cons a s has, f i) :=
  fun i hi =>
    if h : i = a then cast (congrArg f h.symm) x.1 else x.2 i (mem_of_mem_cons_of_ne hi h)


/-- The equivalence between pi types on cons and the product. -/
def consPiProdEquiv [DecidableEq α] {s : Finset α} (f : α → Type*) {a : α} (has : a ∉ s) :
    (Π i ∈ cons a s has, f i) ≃ f a × Π i ∈ s, f i where
  toFun := consPiProd f has
  invFun := prodPiCons f has
  left_inv _ := by
    /-
      α : Type u_1
      β : Type u_2
      s✝ t : Finset α
      a✝ b : α
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
      ⊢ Eq (Finset.prodPiCons f has (Finset.consPiProd f has x✝)) x✝
    -/
    ext i _
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      s✝ t : Finset α
      a✝ b : α
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝¹ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
      i : α
      x✝ : Membership.mem (Finset.cons a s has) i
      ⊢ Eq (Finset.prodPiCons f has (Finset.consPiProd f has x✝¹) i x✝) (x✝¹ i x✝)
    -/
    dsimp only [prodPiCons, consPiProd]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      s✝ t : Finset α
      a✝ b : α
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝¹ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
      i : α
      x✝ : Membership.mem (Finset.cons a s has) i
      ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
    -/
    by_cases h : i = a
      /-
        case pos
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
        i : α
        x✝ : Membership.mem (Finset.cons a s has) i
        h : Eq i a
        ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
      -/
    · rw [dif_pos h]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
        i : α
        x✝ : Membership.mem (Finset.cons a s has) i
        h : Eq i a
        ⊢ Eq (cast ⋯ (x✝¹ a ⋯)) (x✝¹ i x✝)
      -/
      subst h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        i : α
        has : Not (Membership.mem s i)
        x✝¹ : (i_1 : α) → Membership.mem (Finset.cons i s has) i_1 → f i_1
        x✝ : Membership.mem (Finset.cons i s has) i
        ⊢ Eq (cast ⋯ (x✝¹ i ⋯)) (x✝¹ i x✝)
      -/
      simp_all only [cast_eq]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Finset.cons a s has) i → f i
        i : α
        x✝ : Membership.mem (Finset.cons a s has) i
        h : Not (Eq i a)
        ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
      -/
    · rw [dif_neg h]
      /-
        🎉 no goals
      -/
  right_inv _ := by
    /-
      α : Type u_1
      β : Type u_2
      s✝ t : Finset α
      a✝ b : α
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝ : Prod (f a) ((i : α) → Membership.mem s i → f i)
      ⊢ Eq (Finset.consPiProd f has (Finset.prodPiCons f has x✝)) x✝
    -/
    ext _ hi
      /-
        case fst
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        ⊢ Eq (Finset.consPiProd f has (Finset.prodPiCons f has x✝)).1 x✝.1
      -/
    · simp [prodPiCons]
      /-
        🎉 no goals
      -/
      /-
        case snd.h.h
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        x✝ : α
        hi : Membership.mem s x✝
        ⊢ Eq ((Finset.consPiProd f has (Finset.prodPiCons f has x✝¹)).2 x✝ hi) (x✝¹.2  …
      -/
    · simp only [consPiProd_snd]
      /-
        case snd.h.h
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b : α
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        x✝ : α
        hi : Membership.mem s x✝
        ⊢ Eq (Finset.prodPiCons f has x✝¹ x✝ ⋯) (x✝¹.2 x✝ hi)
      -/
      exact dif_neg (ne_of_mem_of_not_mem hi has)
      /-
        🎉 no goals
      -/


/-- `insert a s` is the set `{a} ∪ s` containing `a` and the elements of `s`. -/
instance : Insert α (Finset α) :=
  ⟨fun a s => ⟨_, s.2.ndinsert a⟩⟩


theorem insert_def (a : α) (s : Finset α) : insert a s = ⟨_, s.2.ndinsert a⟩ :=
  rfl


@[simp]
theorem insert_val (a : α) (s : Finset α) : (insert a s).1 = ndinsert a s.1 :=
  rfl


theorem insert_val' (a : α) (s : Finset α) : (insert a s).1 = dedup (a ::ₘ s.1) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Insert.insert a s).val (Multiset.cons a s.val).dedup
  -/
  rw [dedup_cons, dedup_eq_self]; rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem insert_val_of_not_mem {a : α} {s : Finset α} (h : a ∉ s) : (insert a s).1 = a ::ₘ s.1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    h : Not (Membership.mem s a)
    ⊢ Eq (Insert.insert a s).val (Multiset.cons a s.val)
  -/
  rw [insert_val, ndinsert_of_not_mem h]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_insert : a ∈ insert b s ↔ a = b ∨ a ∈ s :=
  mem_ndinsert


theorem mem_insert_self (a : α) (s : Finset α) : a ∈ insert a s :=
  mem_ndinsert_self a s.1


theorem mem_insert_of_mem (h : a ∈ s) : a ∈ insert b s :=
  mem_ndinsert_of_mem h


theorem mem_of_mem_insert_of_ne (h : b ∈ insert a s) : b ≠ a → b ∈ s :=
  (mem_insert.1 h).resolve_left


theorem eq_of_mem_insert_of_not_mem (ha : b ∈ insert a s) (hb : b ∉ s) : b = a :=
  (mem_insert.1 ha).resolve_right hb


/-- A version of `LawfulSingleton.insert_emptyc_eq` that works with `dsimp`. -/
@[simp] lemma insert_empty : insert a (∅ : Finset α) = {a} := rfl


@[simp]
theorem cons_eq_insert (a s h) : @cons α a s h = insert a s :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    a✝ : α
                    s : Finset α
                    h : Not (Membership.mem s a✝)
                    a : α
                    ⊢ Iff (Membership.mem (Finset.cons a✝ s h) a) (Membership.mem (Insert.insert a …
                  -/
  ext fun a => by simp
                  /-
                    🎉 no goals
                  -/


@[simp, norm_cast]
theorem coe_insert (a : α) (s : Finset α) : ↑(insert a s) = (insert a s : Set α) :=
                      /-
                        α : Type u_1
                        inst✝ : DecidableEq α
                        a : α
                        s : Finset α
                        x : α
                        ⊢ Iff (Membership.mem (↑(Insert.insert a s)) x) (Membership.mem (Insert.insert …
                      -/
  Set.ext fun x => by simp only [mem_coe, mem_insert, Set.mem_insert_iff]
                      /-
                        🎉 no goals
                      -/


theorem mem_insert_coe {s : Finset α} {x y : α} : x ∈ insert y s ↔ x ∈ insert y (s : Set α) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    ⊢ Iff (Membership.mem (Insert.insert y s) x) (Membership.mem (Insert.insert y  …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : LawfulSingleton α (Finset α) :=
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝ : DecidableEq α
                 s t : Finset α
                 a✝ b : α
                 f : α → β
                 a : α
                 ⊢ Eq (Insert.insert a EmptyCollection.emptyCollection) (Singleton.singleton a)
               -/
  ⟨fun a => by ext; simp⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem insert_eq_of_mem (h : a ∈ s) : insert a s = s :=
  eq_of_veq <| ndinsert_of_mem h


@[simp]
theorem insert_eq_self : insert a s = s ↔ a ∈ s :=
  ⟨fun h => h ▸ mem_insert_self _ _, insert_eq_of_mem⟩


theorem insert_ne_self : insert a s ≠ s ↔ a ∉ s :=
  insert_eq_self.not


theorem pair_eq_singleton (a : α) : ({a, a} : Finset α) = {a} :=
  insert_eq_of_mem <| mem_singleton_self _


theorem insert_comm (a b : α) (s : Finset α) : insert a (insert b s) = insert b (insert a s) :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    a b : α
                    s : Finset α
                    x : α
                    ⊢ Iff (Membership.mem (Insert.insert a (Insert.insert b s)) x) (Membership.mem …
                  -/
  ext fun x => by simp only [mem_insert, or_left_comm]
                  /-
                    🎉 no goals
                  -/


@[deprecated (since := "2024-11-29")] alias Insert.comm := insert_comm


@[norm_cast]
theorem coe_pair {a b : α} : (({a, b} : Finset α) : Set α) = {a, b} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq (↑(Insert.insert a (Singleton.singleton b))) (Insert.insert a (Singleton. …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a b x✝ : α
    ⊢ Iff (Membership.mem (↑(Insert.insert a (Singleton.singleton b))) x✝) (Member …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_eq_pair {s : Finset α} {a b : α} : (s : Set α) = {a, b} ↔ s = {a, b} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a b : α
    ⊢ Iff (Eq (↑s) (Insert.insert a (Singleton.singleton b))) (Eq s (Insert.insert …
  -/
  rw [← coe_pair, coe_inj]
  /-
    🎉 no goals
  -/


theorem pair_comm (a b : α) : ({a, b} : Finset α) = {b, a} :=
  insert_comm a b ∅


theorem insert_idem (a : α) (s : Finset α) : insert a (insert a s) = insert a s :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    a : α
                    s : Finset α
                    x : α
                    ⊢ Iff (Membership.mem (Insert.insert a (Insert.insert a s)) x) (Membership.mem …
                  -/
  ext fun x => by simp only [mem_insert, ← or_assoc, or_self_iff]
                  /-
                    🎉 no goals
                  -/


@[simp, aesop safe apply (rule_sets := [finsetNonempty])]
theorem insert_nonempty (a : α) (s : Finset α) : (insert a s).Nonempty :=
  ⟨a, mem_insert_self a s⟩


@[simp]
theorem insert_ne_empty (a : α) (s : Finset α) : insert a s ≠ ∅ :=
  (insert_nonempty a s).ne_empty

-- Porting note: explicit universe annotation is no longer required.

instance (i : α) (s : Finset α) : Nonempty ((insert i s : Finset α) : Set α) :=
  (Finset.coe_nonempty.mpr (s.insert_nonempty i)).to_subtype


theorem ne_insert_of_not_mem (s t : Finset α) {a : α} (h : a ∉ s) : s ≠ insert a t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Ne s (Insert.insert a t)
  -/
  contrapose! h
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    h : Eq s (Insert.insert a t)
    ⊢ Membership.mem s a
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem insert_subset_iff : insert a s ⊆ t ↔ a ∈ t ∧ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Iff (HasSubset.Subset (Insert.insert a s) t) (And (Membership.mem t a) (HasS …
  -/
  simp only [subset_iff, mem_insert, forall_eq, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem insert_subset (ha : a ∈ t) (hs : s ⊆ t) : insert a s ⊆ t :=
  insert_subset_iff.mpr ⟨ha,hs⟩


@[simp] theorem subset_insert (a : α) (s : Finset α) : s ⊆ insert a s := fun _b => mem_insert_of_mem


@[gcongr]
theorem insert_subset_insert (a : α) {s t : Finset α} (h : s ⊆ t) : insert a s ⊆ insert a t :=
  insert_subset_iff.2 ⟨mem_insert_self _ _, Subset.trans h (subset_insert _ _)⟩


@[simp] lemma insert_subset_insert_iff (ha : a ∉ s) : insert a s ⊆ insert a t ↔ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Iff (HasSubset.Subset (Insert.insert a s) (Insert.insert a t)) (HasSubset.Su …
  -/
  simp_rw [← coe_subset]; simp [-coe_subset, ha]
                          /-
                            🎉 no goals
                          -/


theorem insert_inj (ha : a ∉ s) : insert a s = insert b s ↔ a = b :=
  ⟨fun h => eq_of_mem_insert_of_not_mem (h ▸ mem_insert_self _ _) ha, congr_arg (insert · s)⟩


theorem insert_inj_on (s : Finset α) : Set.InjOn (fun a => insert a s) sᶜ := fun _ h _ _ =>
  (insert_inj h).1


theorem ssubset_iff : s ⊂ t ↔ ∃ a ∉ s, insert a s ⊆ t := mod_cast @Set.ssubset_iff_insert α s t


theorem ssubset_insert (h : a ∉ s) : s ⊂ insert a s :=
  ssubset_iff.mpr ⟨a, h, Subset.rfl⟩


@[elab_as_elim]
theorem cons_induction {α : Type*} {p : Finset α → Prop} (empty : p ∅)
    (cons : ∀ (a : α) (s : Finset α) (h : a ∉ s), p s → p (cons a s h)) : ∀ s, p s
  | ⟨s, nd⟩ => by
    induction s using Multiset.induction with
    | empty => exact empty
    | cons a s IH =>
      rw [mk_cons nd]
      exact cons a _ _ (IH _)


@[elab_as_elim]
theorem cons_induction_on {α : Type*} {p : Finset α → Prop} (s : Finset α) (h₁ : p ∅)
    (h₂ : ∀ ⦃a : α⦄ {s : Finset α} (h : a ∉ s), p s → p (cons a s h)) : p s :=
  cons_induction h₁ h₂ s


@[elab_as_elim]
protected theorem induction {α : Type*} {p : Finset α → Prop} [DecidableEq α] (empty : p ∅)
    (insert : ∀ ⦃a : α⦄ {s : Finset α}, a ∉ s → p s → p (insert a s)) : ∀ s, p s :=
  cons_induction empty fun a s ha => (s.cons_eq_insert a ha).symm ▸ insert ha


/-- To prove a proposition about an arbitrary `Finset α`,
it suffices to prove it for the empty `Finset`,
and to show that if it holds for some `Finset α`,
then it holds for the `Finset` obtained by inserting a new element.
-/
@[elab_as_elim]
protected theorem induction_on {α : Type*} {p : Finset α → Prop} [DecidableEq α] (s : Finset α)
    (empty : p ∅) (insert : ∀ ⦃a : α⦄ {s : Finset α}, a ∉ s → p s → p (insert a s)) : p s :=
  Finset.induction empty insert s


/-- To prove a proposition about `S : Finset α`,
it suffices to prove it for the empty `Finset`,
and to show that if it holds for some `Finset α ⊆ S`,
then it holds for the `Finset` obtained by inserting a new element of `S`.
-/
@[elab_as_elim]
theorem induction_on' {α : Type*} {p : Finset α → Prop} [DecidableEq α] (S : Finset α) (h₁ : p ∅)
    (h₂ : ∀ {a s}, a ∈ S → s ⊆ S → a ∉ s → p s → p (insert a s)) : p S :=
  @Finset.induction_on α (fun T => T ⊆ S → p T) _ S (fun _ => h₁)
    (fun _ _ has hqs hs =>
      let ⟨hS, sS⟩ := Finset.insert_subset_iff.1 hs
      h₂ hS sS has (hqs sS))
    (Finset.Subset.refl S)


/-- To prove a proposition about a nonempty `s : Finset α`, it suffices to show it holds for all
singletons and that if it holds for nonempty `t : Finset α`, then it also holds for the `Finset`
obtained by inserting an element in `t`. -/
@[elab_as_elim]
theorem Nonempty.cons_induction {α : Type*} {p : ∀ s : Finset α, s.Nonempty → Prop}
    (singleton : ∀ a, p {a} (singleton_nonempty _))
    (cons : ∀ a s (h : a ∉ s) (hs), p s hs → p (Finset.cons a s h) (cons_nonempty h))
    {s : Finset α} (hs : s.Nonempty) : p s hs := by
  induction s using Finset.cons_induction with
  | empty => exact (not_nonempty_empty hs).elim
  | cons a t ha h =>
    obtain rfl | ht := t.eq_empty_or_nonempty
    · exact singleton a
    · exact cons a t ha ht (h ht)

-- We use a fresh `α` here to exclude the unneeded `DecidableEq α` instance from the section.

lemma Nonempty.exists_cons_eq {α} {s : Finset α} (hs : s.Nonempty) : ∃ t a ha, cons a t ha = s :=
  hs.cons_induction (fun a ↦ ⟨∅, a, _, cons_empty _⟩) fun _ _ _ _ _ ↦ ⟨_, _, _, rfl⟩


/-- Inserting an element to a finite set is equivalent to the option type. -/
def subtypeInsertEquivOption {t : Finset α} {x : α} (h : x ∉ t) :
    { i // i ∈ insert x t } ≃ Option { i // i ∈ t } where
  toFun y := if h : ↑y = x then none else some ⟨y, (mem_insert.mp y.2).resolve_left h⟩
  invFun y := (y.elim ⟨x, mem_insert_self _ _⟩) fun z => ⟨z, mem_insert_of_mem z.2⟩
  left_inv y := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s t✝ : Finset α
      a b : α
      f : α → β
      t : Finset α
      x : α
      h : Not (Membership.mem t x)
      y : Subtype fun i => Membership.mem (Insert.insert x t) i
      ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
    -/
    by_cases h : ↑y = x
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        s t✝ : Finset α
        a b : α
        f : α → β
        t : Finset α
        x : α
        h✝ : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem (Insert.insert x t) i
        h : Eq (↑y) x
        ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
      -/
    · simp only [Subtype.ext_iff, h, Option.elim, dif_pos, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        s t✝ : Finset α
        a b : α
        f : α → β
        t : Finset α
        x : α
        h✝ : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem (Insert.insert x t) i
        h : Not (Eq (↑y) x)
        ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
      -/
    · simp only [h, Option.elim, dif_neg, not_false_iff, Subtype.coe_eta, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s t✝ : Finset α
      a b : α
      f : α → β
      t : Finset α
      x : α
      h : Not (Membership.mem t x)
      ⊢ Function.RightInverse (fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) fun y => dit …
    -/
    rintro (_ | y)
      /-
        case none
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        s t✝ : Finset α
        a b : α
        f : α → β
        t : Finset α
        x : α
        h : Not (Membership.mem t x)
        ⊢ Eq ((fun y => dite (Eq (↑y) x) (fun h => Option.none) fun h => Option.some ⟨ …
      -/
    · simp only [Option.elim, dif_pos]
      /-
        🎉 no goals
      -/
    · have : ↑y ≠ x := by
        rintro ⟨⟩
        exact h y.2
      /-
        case some
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        s t✝ : Finset α
        a b : α
        f : α → β
        t : Finset α
        x : α
        h : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem t i
        this : Ne (↑y) x
        ⊢ Eq ((fun y => dite (Eq (↑y) x) (fun h => Option.none) fun h => Option.some ⟨ …
      -/
      simp only [this, Option.elim, Subtype.eta, dif_neg, not_false_iff, Subtype.coe_mk]
      /-
        🎉 no goals
      -/


/-- Split the added element of insert off a Pi type. -/
@[simps!]
def insertPiProd (f : α → Type*) (x : Π i ∈ insert a s, f i) : f a × Π i ∈ s, f i :=
  (x a (mem_insert_self a s), fun i hi => x i (mem_insert_of_mem hi))


/-- Combine a product with a pi type to pi of insert. -/
def prodPiInsert (f : α → Type*) {a : α} (x : f a × Π i ∈ s, f i) : (Π i ∈ insert a s, f i) :=
  fun i hi =>
    if h : i = a then cast (congrArg f h.symm) x.1 else x.2 i (mem_of_mem_insert_of_ne hi h)


/-- The equivalence between pi types on insert and the product. -/
def insertPiProdEquiv [DecidableEq α] {s : Finset α} (f : α → Type*) {a : α} (has : a ∉ s) :
    (Π i ∈ insert a s, f i) ≃ f a × Π i ∈ s, f i where
  toFun := insertPiProd f
  invFun := prodPiInsert f
  left_inv _ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      s✝ t : Finset α
      a✝ b : α
      f✝ : α → β
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝ : (i : α) → Membership.mem (Insert.insert a s) i → f i
      ⊢ Eq (Finset.prodPiInsert f (Finset.insertPiProd f x✝)) x✝
    -/
    ext i _
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      s✝ t : Finset α
      a✝ b : α
      f✝ : α → β
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝¹ : (i : α) → Membership.mem (Insert.insert a s) i → f i
      i : α
      x✝ : Membership.mem (Insert.insert a s) i
      ⊢ Eq (Finset.prodPiInsert f (Finset.insertPiProd f x✝¹) i x✝) (x✝¹ i x✝)
    -/
    dsimp only [prodPiInsert, insertPiProd]
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      s✝ t : Finset α
      a✝ b : α
      f✝ : α → β
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝¹ : (i : α) → Membership.mem (Insert.insert a s) i → f i
      i : α
      x✝ : Membership.mem (Insert.insert a s) i
      ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
    -/
    by_cases h : i = a
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Insert.insert a s) i → f i
        i : α
        x✝ : Membership.mem (Insert.insert a s) i
        h : Eq i a
        ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
      -/
    · rw [dif_pos h]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Insert.insert a s) i → f i
        i : α
        x✝ : Membership.mem (Insert.insert a s) i
        h : Eq i a
        ⊢ Eq (cast ⋯ (x✝¹ a ⋯)) (x✝¹ i x✝)
      -/
      subst h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        i : α
        has : Not (Membership.mem s i)
        x✝¹ : (i_1 : α) → Membership.mem (Insert.insert i s) i_1 → f i_1
        x✝ : Membership.mem (Insert.insert i s) i
        ⊢ Eq (cast ⋯ (x✝¹ i ⋯)) (x✝¹ i x✝)
      -/
      simp_all only [cast_eq]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : (i : α) → Membership.mem (Insert.insert a s) i → f i
        i : α
        x✝ : Membership.mem (Insert.insert a s) i
        h : Not (Eq i a)
        ⊢ Eq (dite (Eq i a) (fun h => cast ⋯ (x✝¹ a ⋯)) fun h => x✝¹ i ⋯) (x✝¹ i x✝)
      -/
    · rw [dif_neg h]
      /-
        🎉 no goals
      -/
  right_inv _ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      s✝ t : Finset α
      a✝ b : α
      f✝ : α → β
      inst✝ : DecidableEq α
      s : Finset α
      f : α → Type u_3
      a : α
      has : Not (Membership.mem s a)
      x✝ : Prod (f a) ((i : α) → Membership.mem s i → f i)
      ⊢ Eq (Finset.insertPiProd f (Finset.prodPiInsert f x✝)) x✝
    -/
    ext _ hi
      /-
        case fst
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        ⊢ Eq (Finset.insertPiProd f (Finset.prodPiInsert f x✝)).1 x✝.1
      -/
    · simp [prodPiInsert]
      /-
        🎉 no goals
      -/
      /-
        case snd.h.h
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        x✝ : α
        hi : Membership.mem s x✝
        ⊢ Eq ((Finset.insertPiProd f (Finset.prodPiInsert f x✝¹)).2 x✝ hi) (x✝¹.2 x✝ hi)
      -/
    · simp only [insertPiProd_snd]
      /-
        case snd.h.h
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        s✝ t : Finset α
        a✝ b : α
        f✝ : α → β
        inst✝ : DecidableEq α
        s : Finset α
        f : α → Type u_3
        a : α
        has : Not (Membership.mem s a)
        x✝¹ : Prod (f a) ((i : α) → Membership.mem s i → f i)
        x✝ : α
        hi : Membership.mem s x✝
        ⊢ Eq (Finset.prodPiInsert f x✝¹ x✝ ⋯) (x✝¹.2 x✝ hi)
      -/
      exact dif_neg (ne_of_mem_of_not_mem hi has)
      /-
        🎉 no goals
      -/

-- useful rules for calculations with quantifiers

theorem exists_mem_insert (a : α) (s : Finset α) (p : α → Prop) :
    (∃ x, x ∈ insert a s ∧ p x) ↔ p a ∨ ∃ x, x ∈ s ∧ p x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    p : α → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem (Insert.insert a s) x) (p x)) (Or ( …
  -/
  simp only [mem_insert, or_and_right, exists_or, exists_eq_left]
  /-
    🎉 no goals
  -/


theorem forall_mem_insert (a : α) (s : Finset α) (p : α → Prop) :
    (∀ x, x ∈ insert a s → p x) ↔ p a ∧ ∀ x, x ∈ s → p x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    p : α → Prop
    ⊢ Iff (∀ (x : α), Membership.mem (Insert.insert a s) x → p x) (And (p a) (∀ (x …
  -/
  simp only [mem_insert, or_imp, forall_and, forall_eq]
  /-
    🎉 no goals
  -/


/-- Useful in proofs by induction. -/
theorem forall_of_forall_insert {p : α → Prop} {a : α} {s : Finset α}
    (H : ∀ x, x ∈ insert a s → p x) (x) (h : x ∈ s) : p x :=
  H _ <| mem_insert_of_mem h


@[simp]
theorem toFinset_zero : toFinset (0 : Multiset α) = ∅ :=
  rfl


@[simp]
theorem toFinset_cons (a : α) (s : Multiset α) : toFinset (a ::ₘ s) = insert a (toFinset s) :=
  Finset.eq_of_veq dedup_cons


@[simp]
theorem toFinset_singleton (a : α) : toFinset ({a} : Multiset α) = {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Singleton.singleton a).toFinset (Singleton.singleton a)
  -/
  rw [← cons_zero, toFinset_cons, toFinset_zero, LawfulSingleton.insert_emptyc_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_nil : toFinset (@nil α) = ∅ :=
  rfl


@[simp]
theorem toFinset_cons : toFinset (a :: l) = insert a (toFinset l) :=
                         /-
                           α : Type u_1
                           inst✝ : DecidableEq α
                           l : List α
                           a : α
                           ⊢ Eq (List.cons a l).toFinset.val (Insert.insert a l.toFinset).val
                         -/
                                                /-
                                                  🎉 no goals
                                                -/
  Finset.eq_of_veq <| by by_cases h : a ∈ l <;> simp [Finset.insert_val', Multiset.dedup_cons, h]
                                                /-
                                                  🎉 no goals
                                                -/


theorem toFinset_replicate_of_ne_zero {n : ℕ} (hn : n ≠ 0) :
    (List.replicate n a).toFinset = {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    hn : Ne n 0
    ⊢ Eq (List.replicate n a).toFinset (Singleton.singleton a)
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    hn : Ne n 0
    x : α
    ⊢ Iff (Membership.mem (List.replicate n a).toFinset x) (Membership.mem (Single …
  -/
  simp [hn, List.mem_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_eq_empty_iff (l : List α) : l.toFinset = ∅ ↔ l = nil := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ Iff (Eq l.toFinset EmptyCollection.emptyCollection) (Eq l List.nil)
  -/
              /-
                🎉 no goals
              -/
  cases l <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem toFinset_nonempty_iff (l : List α) : l.toFinset.Nonempty ↔ l ≠ [] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ Iff l.toFinset.Nonempty (Ne l List.nil)
  -/
  simp [Finset.nonempty_iff_ne_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem toList_eq_singleton_iff {a : α} {s : Finset α} : s.toList = [a] ↔ s = {a} := by
  /-
    α : Type u_1
    a : α
    s : Finset α
    ⊢ Iff (Eq s.toList (List.cons a List.nil)) (Eq s (Singleton.singleton a))
  -/
  rw [toList, Multiset.toList_eq_singleton_iff, val_eq_singleton_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem toList_singleton : ∀ a, ({a} : Finset α).toList = [a] :=
  Multiset.toList_singleton


open scoped List in
theorem toList_cons {a : α} {s : Finset α} (h : a ∉ s) : (cons a s h).toList ~ a :: s.toList :=
                                                   /-
                                                     α : Type u_1
                                                     a : α
                                                     s : Finset α
                                                     h : Not (Membership.mem s a)
                                                     ⊢ (List.cons a s.toList).Nodup
                                                   -/
  (List.perm_ext_iff_of_nodup (nodup_toList _) (by simp [h, nodup_toList s])).2 fun x => by
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      α : Type u_1
      a : α
      s : Finset α
      h : Not (Membership.mem s a)
      x : α
      ⊢ Iff (Membership.mem (Finset.cons a s h).toList x) (Membership.mem (List.cons …
    -/
    simp only [List.mem_cons, Finset.mem_toList, Finset.mem_cons]
    /-
      🎉 no goals
    -/


open scoped List in
theorem toList_insert [DecidableEq α] {a : α} {s : Finset α} (h : a ∉ s) :
    (insert a s).toList ~ a :: s.toList :=
  cons_eq_insert _ _ h ▸ toList_cons _


theorem pairwise_cons' {a : α} (ha : a ∉ s) (r : β → β → Prop) (f : α → β) :
    Pairwise (r on fun a : s.cons a ha => f a) ↔
    Pairwise (r on fun a : s => f a) ∧ ∀ b ∈ s, r (f a) (f b) ∧ r (f b) (f a) := by
  simp only [pairwise_subtype_iff_pairwise_finset', Finset.coe_cons, Set.pairwise_insert,
    Finset.mem_coe, and_congr_right_iff]
  exact fun _ =>
    ⟨fun h b hb =>
      h b hb <| by
        rintro rfl
        contradiction,
      fun h b hb _ => h b hb⟩


theorem pairwise_cons {a : α} (ha : a ∉ s) (r : α → α → Prop) :
    Pairwise (r on fun a : s.cons a ha => a) ↔
      Pairwise (r on fun a : s => a) ∧ ∀ b ∈ s, r a b ∧ r b a :=
  pairwise_cons' ha r id


