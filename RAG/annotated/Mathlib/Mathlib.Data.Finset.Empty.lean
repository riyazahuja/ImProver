/-- The property `s.Nonempty` expresses the fact that the finset `s` is not empty. It should be used
in theorem assumptions instead of `∃ x, x ∈ s` or `s ≠ ∅` as it gives access to a nice API thanks
to the dot notation. -/
protected def Nonempty (s : Finset α) : Prop := ∃ x : α, x ∈ s

-- Porting note: Much longer than in Lean3

instance decidableNonempty {s : Finset α} : Decidable s.Nonempty :=
  Quotient.recOnSubsingleton (motive := fun s : Multiset α => Decidable (∃ a, a ∈ s)) s.1
    (fun l : List α =>
      match l with
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              s : Finset α
                              l : List α
                              ⊢ Not (Exists fun a => Membership.mem (Quotient.mk (List.isSetoid α) List.nil) …
                            -/
      | [] => isFalse <| by simp
                            /-
                              🎉 no goals
                            -/
                              /-
                                α : Type u_1
                                β : Type u_2
                                γ : Type u_3
                                s : Finset α
                                l✝ : List α
                                a : α
                                l : List α
                                ⊢ Membership.mem (Quotient.mk (List.isSetoid α) (List.cons a l)) a
                              -/
      | a::l => isTrue ⟨a, by simp⟩)
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
theorem coe_nonempty {s : Finset α} : (s : Set α).Nonempty ↔ s.Nonempty :=
  Iff.rfl

-- Porting note: Left-hand side simplifies @[simp]

theorem nonempty_coe_sort {s : Finset α} : Nonempty (s : Type _) ↔ s.Nonempty :=
  nonempty_subtype


alias ⟨_, Nonempty.to_set⟩ := coe_nonempty


alias ⟨_, Nonempty.coe_sort⟩ := nonempty_coe_sort


theorem Nonempty.exists_mem {s : Finset α} (h : s.Nonempty) : ∃ x : α, x ∈ s :=
  h

@[deprecated (since := "2024-03-23")] alias Nonempty.bex := Nonempty.exists_mem


theorem Nonempty.mono {s t : Finset α} (hst : s ⊆ t) (hs : s.Nonempty) : t.Nonempty :=
  Set.Nonempty.mono hst hs


theorem Nonempty.forall_const {s : Finset α} (h : s.Nonempty) {p : Prop} : (∀ x ∈ s, p) ↔ p :=
  let ⟨x, hx⟩ := h
  ⟨fun h => h x hx, fun h _ _ => h⟩


theorem Nonempty.to_subtype {s : Finset α} : s.Nonempty → Nonempty s :=
  nonempty_coe_sort.2


theorem Nonempty.to_type {s : Finset α} : s.Nonempty → Nonempty α := fun ⟨x, _hx⟩ => ⟨x⟩


/-- The empty finset -/
protected def empty : Finset α :=
  ⟨0, nodup_zero⟩


instance : EmptyCollection (Finset α) :=
  ⟨Finset.empty⟩


instance inhabitedFinset : Inhabited (Finset α) :=
  ⟨∅⟩


@[simp]
theorem empty_val : (∅ : Finset α).1 = 0 :=
  rfl


@[simp]
theorem not_mem_empty (a : α) : a ∉ (∅ : Finset α) := by
  -- Porting note: was `id`. `a ∈ List.nil` is no longer definitionally equal to `False`
  /-
    α : Type u_1
    a : α
    ⊢ Not (Membership.mem EmptyCollection.emptyCollection a)
  -/
  simp only [mem_def, empty_val, not_mem_zero, not_false_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_nonempty_empty : ¬(∅ : Finset α).Nonempty := fun ⟨x, hx⟩ => not_mem_empty x hx


@[simp]
theorem mk_zero : (⟨0, nodup_zero⟩ : Finset α) = ∅ :=
  rfl


theorem ne_empty_of_mem {a : α} {s : Finset α} (h : a ∈ s) : s ≠ ∅ := fun e =>
  not_mem_empty a <| e ▸ h


theorem Nonempty.ne_empty {s : Finset α} (h : s.Nonempty) : s ≠ ∅ :=
  (Exists.elim h) fun _a => ne_empty_of_mem


@[simp]
theorem empty_subset (s : Finset α) : ∅ ⊆ s :=
  zero_subset _


theorem eq_empty_of_forall_not_mem {s : Finset α} (H : ∀ x, x ∉ s) : s = ∅ :=
  eq_of_veq (eq_zero_of_forall_not_mem H)


theorem eq_empty_iff_forall_not_mem {s : Finset α} : s = ∅ ↔ ∀ x, x ∉ s :=
  -- Porting note: used `id`
      /-
        α : Type u_1
        s : Finset α
        ⊢ Eq s EmptyCollection.emptyCollection → ∀ (x : α), Not (Membership.mem s x)
      -/
  ⟨by rintro rfl x; apply not_mem_empty, fun h => eq_empty_of_forall_not_mem h⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem val_eq_zero {s : Finset α} : s.1 = 0 ↔ s = ∅ :=
  @val_inj _ s ∅


@[simp] lemma subset_empty : s ⊆ ∅ ↔ s = ∅ := subset_zero.trans val_eq_zero


@[simp]
theorem not_ssubset_empty (s : Finset α) : ¬s ⊂ ∅ := fun h =>
  let ⟨_, he, _⟩ := exists_of_ssubset h
  -- Porting note: was `he`
  not_mem_empty _ he


theorem nonempty_of_ne_empty {s : Finset α} (h : s ≠ ∅) : s.Nonempty :=
  exists_mem_of_ne_zero (mt val_eq_zero.1 h)


theorem nonempty_iff_ne_empty {s : Finset α} : s.Nonempty ↔ s ≠ ∅ :=
  ⟨Nonempty.ne_empty, nonempty_of_ne_empty⟩


@[simp]
theorem not_nonempty_iff_eq_empty {s : Finset α} : ¬s.Nonempty ↔ s = ∅ :=
  nonempty_iff_ne_empty.not.trans not_not


theorem eq_empty_or_nonempty (s : Finset α) : s = ∅ ∨ s.Nonempty :=
  by_cases Or.inl fun h => Or.inr (nonempty_of_ne_empty h)


@[simp, norm_cast]
theorem coe_empty : ((∅ : Finset α) : Set α) = ∅ :=
                /-
                  α : Type u_1
                  ⊢ ∀ (x : α), Iff (Membership.mem (↑EmptyCollection.emptyCollection) x) (Member …
                -/
  Set.ext <| by simp
                /-
                  🎉 no goals
                -/


@[simp, norm_cast]
                                                                    /-
                                                                      α : Type u_1
                                                                      s : Finset α
                                                                      ⊢ Iff (Eq (↑s) EmptyCollection.emptyCollection) (Eq s EmptyCollection.emptyCol …
                                                                    -/
theorem coe_eq_empty {s : Finset α} : (s : Set α) = ∅ ↔ s = ∅ := by rw [← coe_empty, coe_inj]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

-- Porting note: Left-hand side simplifies @[simp]

theorem isEmpty_coe_sort {s : Finset α} : IsEmpty (s : Type _) ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (IsEmpty (Subtype fun x => Membership.mem s x)) (Eq s EmptyCollection.em …
  -/
  simpa using @Set.isEmpty_coe_sort α s
  /-
    🎉 no goals
  -/


instance instIsEmpty : IsEmpty (∅ : Finset α) :=
  isEmpty_coe_sort.2 rfl


/-- A `Finset` for an empty type is empty. -/
theorem eq_empty_of_isEmpty [IsEmpty α] (s : Finset α) : s = ∅ :=
  Finset.eq_empty_of_forall_not_mem isEmptyElim


instance : OrderBot (Finset α) where
  bot := ∅
  bot_le := empty_subset


@[simp]
theorem bot_eq_empty : (⊥ : Finset α) = ∅ :=
  rfl


@[simp]
theorem empty_ssubset : ∅ ⊂ s ↔ s.Nonempty :=
  (@bot_lt_iff_ne_bot (Finset α) _ _ _).trans nonempty_iff_ne_empty.symm


alias ⟨_, Nonempty.empty_ssubset⟩ := empty_ssubset

-- useful rules for calculations with quantifiers

theorem exists_mem_empty_iff (p : α → Prop) : (∃ x, x ∈ (∅ : Finset α) ∧ p x) ↔ False := by
  /-
    α : Type u_1
    p : α → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem EmptyCollection.emptyCollection x)  …
  -/
  simp only [not_mem_empty, false_and, exists_false]
  /-
    🎉 no goals
  -/


theorem forall_mem_empty_iff (p : α → Prop) : (∀ x, x ∈ (∅ : Finset α) → p x) ↔ True :=
  iff_true_intro fun _ h => False.elim <| not_mem_empty _ h


/-- Attempt to prove that a finset is nonempty using the `finsetNonempty` aesop rule-set.

You can add lemmas to the rule-set by tagging them with either:
* `aesop safe apply (rule_sets := [finsetNonempty])` if they are always a good idea to follow or
* `aesop unsafe apply (rule_sets := [finsetNonempty])` if they risk directing the search to a blind
  alley.

TODO: should some of the lemmas be `aesop safe simp` instead?
-/
def proveFinsetNonempty {u : Level} {α : Q(Type u)} (s : Q(Finset $α)) :
    MetaM (Option Q(Finset.Nonempty $s)) := do
  -- Aesop expects to operate on goals, so we're going to make a new goal.
  let goal ← Lean.Meta.mkFreshExprMVar q(Finset.Nonempty $s)
  let mvar := goal.mvarId!
  -- We want this to be fast, so use only the basic and `Finset.Nonempty`-specific rules.
  let rulesets ← Aesop.Frontend.getGlobalRuleSets #[`builtin, `finsetNonempty]
  let options : Aesop.Options' :=
    { terminal := true -- Fail if the new goal is not closed.
      generateScript := false
      useDefaultSimpSet := false -- Avoiding the whole simp set to speed up the tactic.
      warnOnNonterminal := false -- Don't show a warning on failure, simply return `none`.
      forwardMaxDepth? := none }
  let rules ← Aesop.mkLocalRuleSet rulesets options
  let (remainingGoals, _) ←
    try Aesop.search (options := options.toOptions) mvar (.some rules)
    catch _ => return none
  -- Fail if there are open goals remaining, this serves as an extra check for the
  -- Aesop configuration option `terminal := true`.
  if remainingGoals.size > 0 then return none
  Lean.getExprMVarAssignment? mvar


