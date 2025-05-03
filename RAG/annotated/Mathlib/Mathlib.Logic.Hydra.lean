/-- The relation that specifies valid moves in our hydra game. `CutExpand r s' s`
  means that `s'` is obtained by removing one head `a ∈ s` and adding back an arbitrary
  multiset `t` of heads such that all `a' ∈ t` satisfy `r a' a`.

  This is most directly translated into `s' = s.erase a + t`, but `Multiset.erase` requires
  `DecidableEq α`, so we use the equivalent condition `s' + {a} = s + t` instead, which
  is also easier to verify for explicit multisets `s'`, `s` and `t`.

  We also don't include the condition `a ∈ s` because `s' + {a} = s + t` already
  guarantees `a ∈ s + t`, and if `r` is irreflexive then `a ∉ t`, which is the
  case when `r` is well-founded, the case we are primarily interested in.

  The lemma `Relation.cutExpand_iff` below converts between this convenient definition
  and the direct translation when `r` is irreflexive. -/
def CutExpand (r : α → α → Prop) (s' s : Multiset α) : Prop :=
  ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ s' + {a} = s + t


theorem cutExpand_le_invImage_lex [DecidableEq α] [IsIrrefl α r] :
    CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ (· ≠ ·)) (· < ·)) toFinsupp := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    ⊢ LE.le (Relation.CutExpand r) (InvImage (Finsupp.Lex (Min.min (HasCompl.compl …
  -/
  rintro s t ⟨u, a, hr, he⟩
  /-
    case intro.intro.intro
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s t u : Multiset α
    a : α
    hr : ∀ (a' : α), Membership.mem u a' → r a' a
    he : Eq (HAdd.hAdd s (Singleton.singleton a)) (HAdd.hAdd t u)
    ⊢ InvImage (Finsupp.Lex (Min.min (HasCompl.compl r) fun x1 x2 => Ne x1 x2) fun …
  -/
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)]
      using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)),
      add_zero] at he
    exact he ▸ Nat.lt_succ_self _


theorem cutExpand_singleton {s x} (h : ∀ x' ∈ s, r x' x) : CutExpand r s {x} :=
  ⟨s, x, h, add_comm s _⟩


theorem cutExpand_singleton_singleton {x' x} (h : r x' x) : CutExpand r {x'} {x} :=
                                   /-
                                     α : Type u_1
                                     r : α → α → Prop
                                     x' x : α
                                     h✝ : r x' x
                                     a : α
                                     h : Membership.mem (Singleton.singleton x') a
                                     ⊢ r a x
                                   -/
  cutExpand_singleton fun a h ↦ by rwa [mem_singleton.1 h]
                                   /-
                                     🎉 no goals
                                   -/


theorem cutExpand_add_left {t u} (s) : CutExpand r (s + t) (s + u) ↔ CutExpand r t u :=
                                                  /-
                                                    α : Type u_1
                                                    r : α → α → Prop
                                                    t u s x✝¹ : Multiset α
                                                    x✝ : α
                                                    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd s t) (Singleton.singleton x✝)) (HAdd.hAdd (HAd …
                                                  -/
  exists₂_congr fun _ _ ↦ and_congr Iff.rfl <| by rw [add_assoc, add_assoc, add_left_cancel_iff]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma cutExpand_add_right {s' s} (t) : CutExpand r (s' + t) (s + t) ↔ CutExpand r s' s := by
  /-
    α : Type u_1
    r : α → α → Prop
    s' s t : Multiset α
    ⊢ Iff (Relation.CutExpand r (HAdd.hAdd s' t) (HAdd.hAdd s t)) (Relation.CutExp …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  convert cutExpand_add_left t using 2 <;> apply add_comm
                                           /-
                                             🎉 no goals
                                           -/


theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s' s : Multiset α
    ⊢ Iff (Relation.CutExpand r s' s) (Exists fun t => Exists fun a => And (∀ (a'  …
  -/
  simp_rw [CutExpand, add_singleton_eq_iff]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s' s : Multiset α
    ⊢ Iff (Exists fun t => Exists fun a => And (∀ (a' : α), Membership.mem t a' →  …
  -/
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      r : α → α → Prop
      inst✝¹ : DecidableEq α
      inst✝ : IsIrrefl α r
      s' s t : Multiset α
      a : α
      ⊢ And (∀ (a' : α), Membership.mem t a' → r a' a) (And (Membership.mem (HAdd.hA …
    -/
  · rintro ⟨ht, ha, rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝¹ : DecidableEq α
      inst✝ : IsIrrefl α r
      s t : Multiset α
      a : α
      ht : ∀ (a' : α), Membership.mem t a' → r a' a
      ha : Membership.mem (HAdd.hAdd s t) a
      ⊢ And (∀ (a' : α), Membership.mem t a' → r a' a) (And (Membership.mem s a) (Eq …
    -/
    obtain h | h := mem_add.1 ha
    /-
      case refine_1.intro.intro.inl
      α : Type u_1
      r : α → α → Prop
      inst✝¹ : DecidableEq α
      inst✝ : IsIrrefl α r
      s t : Multiset α
      a : α
      ht : ∀ (a' : α), Membership.mem t a' → r a' a
      ha : Membership.mem (HAdd.hAdd s t) a
      h : Membership.mem s a
      ⊢ And (∀ (a' : α), Membership.mem t a' → r a' a) (And (Membership.mem s a) (Eq …
    -/
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      r : α → α → Prop
      inst✝¹ : DecidableEq α
      inst✝ : IsIrrefl α r
      s' s t : Multiset α
      a : α
      ⊢ And (∀ (a' : α), Membership.mem t a' → r a' a) (And (Membership.mem s a) (Eq …
    -/
  · rintro ⟨ht, h, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝¹ : DecidableEq α
      inst✝ : IsIrrefl α r
      s t : Multiset α
      a : α
      ht : ∀ (a' : α), Membership.mem t a' → r a' a
      h : Membership.mem s a
      ⊢ And (∀ (a' : α), Membership.mem t a' → r a' a) (And (Membership.mem (HAdd.hA …
    -/
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩
    /-
      🎉 no goals
    -/


theorem not_cutExpand_zero [IsIrrefl α r] (s) : ¬CutExpand r s 0 := by
  classical
  rw [cutExpand_iff]
  rintro ⟨_, _, _, ⟨⟩, _⟩


lemma cutExpand_zero {x} : CutExpand r 0 {x} := ⟨0, x, nofun, add_comm 0 _⟩


/-- For any relation `r` on `α`, multiset addition `Multiset α × Multiset α → Multiset α` is a
  fibration between the game sum of `CutExpand r` with itself and `CutExpand r` itself. -/
theorem cutExpand_fibration (r : α → α → Prop) :
    Fibration (GameAdd (CutExpand r) (CutExpand r)) (CutExpand r) fun s ↦ s.1 + s.2 := by
  /-
    α : Type u_1
    r : α → α → Prop
    ⊢ Relation.Fibration (Prod.GameAdd (Relation.CutExpand r) (Relation.CutExpand  …
  -/
  rintro ⟨s₁, s₂⟩ s ⟨t, a, hr, he⟩; dsimp at he ⊢
  classical
  obtain ⟨ha, rfl⟩ := add_singleton_eq_iff.1 he
  rw [add_assoc, mem_add] at ha
  obtain h | h := ha
  · refine ⟨(s₁.erase a + t, s₂), GameAdd.fst ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, ← add_assoc, singleton_add, cons_erase h]
    · rw [add_assoc s₁, erase_add_left_pos _ h, add_right_comm, add_assoc]
  · refine ⟨(s₁, (s₂ + t).erase a), GameAdd.snd ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, singleton_add, cons_erase h]
    · rw [add_assoc, erase_add_right_pos _ h]


/-- `CutExpand` preserves leftward-closedness under a relation. -/
lemma cutExpand_closed [IsIrrefl α r] (p : α → Prop)
    (h : ∀ {a' a}, r a' a → p a → p a') :
    ∀ {s' s}, CutExpand r s' s → (∀ a ∈ s, p a) → ∀ a ∈ s', p a := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsIrrefl α r
    p : α → Prop
    h : ∀ {a' a : α}, r a' a → p a → p a'
    ⊢ ∀ {s' s : Multiset α}, Relation.CutExpand r s' s → (∀ (a : α), Membership.me …
  -/
  intros s' s
  classical
  rw [cutExpand_iff]
  rintro ⟨t, a, hr, ha, rfl⟩ hsp a' h'
  obtain (h'|h') := mem_add.1 h'
  exacts [hsp a' (mem_of_mem_erase h'), h (hr a' h') (hsp a ha)]


lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    /-
      α : Type u_1
      r : α → α → Prop
      a a₁ a₂ : α
      h₁ : r a₁ a
      h₂ : r a₂ a
      ⊢ ∀ (x' : α), Membership.mem (Insert.insert a₁ (Singleton.singleton a₂)) x' →  …
    -/
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    /-
      α : Type u_1
      r : α → α → Prop
      a a₁ a₂ : α
      h₁ : r a₁ a
      h₂ : r a₂ a
      ⊢ And (r a₁ a) (r a₂ a)
    -/
    tauto
    /-
      🎉 no goals
    -/


lemma cutExpand_pair_left {a' a b} (hr : r a' a) : CutExpand r {a', b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_singleton_singleton hr)


lemma cutExpand_pair_right {a b' b} (hr : r b' b) : CutExpand r {a, b'} {a, b} :=
  (cutExpand_add_left {a}).2 (cutExpand_singleton_singleton hr)


lemma cutExpand_double_left {a a₁ a₂ b} (h₁ : r a₁ a) (h₂ : r a₂ a) :
    CutExpand r {a₁, a₂, b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_double h₁ h₂)


/-- A multiset is accessible under `CutExpand` if all its singleton subsets are,
  assuming `r` is irreflexive. -/
theorem acc_of_singleton [IsIrrefl α r] {s : Multiset α} (hs : ∀ a ∈ s, Acc (CutExpand r) {a}) :
    Acc (CutExpand r) s := by
  induction s using Multiset.induction with
  | empty => exact Acc.intro 0 fun s h ↦ (not_cutExpand_zero s h).elim
  | cons a s ihs =>
    rw [← s.singleton_add a]
    rw [forall_mem_cons] at hs
    exact (hs.1.prod_gameAdd <| ihs fun a ha ↦ hs.2 a ha).of_fibration _ (cutExpand_fibration r)


/-- A singleton `{a}` is accessible under `CutExpand r` if `a` is accessible under `r`,
  assuming `r` is irreflexive. -/
theorem _root_.Acc.cutExpand [IsIrrefl α r] {a : α} (hacc : Acc r a) : Acc (CutExpand r) {a} := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsIrrefl α r
    a : α
    hacc : Acc r a
    ⊢ Acc (Relation.CutExpand r) (Singleton.singleton a)
  -/
  induction' hacc with a h ih
  /-
    case intro
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsIrrefl α r
    a✝ a : α
    h : ∀ (y : α), r y a → Acc r y
    ih : ∀ (y : α), r y a → Acc (Relation.CutExpand r) (Singleton.singleton y)
    ⊢ Acc (Relation.CutExpand r) (Singleton.singleton a)
  -/
  refine Acc.intro _ fun s ↦ ?_
  classical
  simp only [cutExpand_iff, mem_singleton]
  rintro ⟨t, a, hr, rfl, rfl⟩
  refine acc_of_singleton fun a' ↦ ?_
  rw [erase_singleton, zero_add]
  exact ih a' ∘ hr a'


/-- `CutExpand r` is well-founded when `r` is. -/
theorem _root_.WellFounded.cutExpand (hr : WellFounded r) : WellFounded (CutExpand r) :=
  ⟨have := hr.isIrrefl; fun _ ↦ acc_of_singleton fun a _ ↦ (hr.apply a).cutExpand⟩


