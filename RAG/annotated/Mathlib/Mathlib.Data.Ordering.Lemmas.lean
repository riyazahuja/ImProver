@[simp]
theorem ite_eq_lt_distrib (c : Prop) [Decidable c] (a b : Ordering) :
    ((if c then a else b) = Ordering.lt) = if c then a = Ordering.lt else b = Ordering.lt := by
  /-
    c : Prop
    inst✝ : Decidable c
    a b : Ordering
    ⊢ Eq (Eq (ite c a b) Ordering.lt) (ite c (Eq a Ordering.lt) (Eq b Ordering.lt))
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases c <;> simp [*]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem ite_eq_eq_distrib (c : Prop) [Decidable c] (a b : Ordering) :
    ((if c then a else b) = Ordering.eq) = if c then a = Ordering.eq else b = Ordering.eq := by
  /-
    c : Prop
    inst✝ : Decidable c
    a b : Ordering
    ⊢ Eq (Eq (ite c a b) Ordering.eq) (ite c (Eq a Ordering.eq) (Eq b Ordering.eq))
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases c <;> simp [*]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem ite_eq_gt_distrib (c : Prop) [Decidable c] (a b : Ordering) :
    ((if c then a else b) = Ordering.gt) = if c then a = Ordering.gt else b = Ordering.gt := by
  /-
    c : Prop
    inst✝ : Decidable c
    a b : Ordering
    ⊢ Eq (Eq (ite c a b) Ordering.gt) (ite c (Eq a Ordering.gt) (Eq b Ordering.gt))
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases c <;> simp [*]
                 /-
                   🎉 no goals
                 -/


@[simp]
lemma dthen_eq_then (o₁ o₂ : Ordering) : o₁.dthen (fun _ => o₂) = o₁.then o₂ := by
  /-
    o₁ o₂ : Ordering
    ⊢ Eq (o₁.dthen fun x => o₂) (o₁.then o₂)
  -/
               /-
                 🎉 no goals
               -/
               /-
                 🎉 no goals
               -/
  cases o₁ <;> rfl
               /-
                 🎉 no goals
               -/


attribute [local simp] cmpUsing


@[simp]
theorem cmpUsing_eq_lt (a b : α) : (cmpUsing lt a b = Ordering.lt) = lt a b := by
  /-
    α : Type u
    lt : α → α → Prop
    inst✝ : DecidableRel lt
    a b : α
    ⊢ Eq (Eq (cmpUsing lt a b) Ordering.lt) (lt a b)
  -/
  simp only [cmpUsing, Ordering.ite_eq_lt_distrib, ite_self, if_false_right, and_true, reduceCtorEq]
  /-
    🎉 no goals
  -/


@[simp]
theorem cmpUsing_eq_gt [IsStrictOrder α lt] (a b : α) : cmpUsing lt a b = Ordering.gt ↔ lt b a := by
  simp only [cmpUsing, Ordering.ite_eq_gt_distrib, if_false_right, and_true, if_false_left,
    and_iff_right_iff_imp, reduceCtorEq]
  /-
    α : Type u
    lt : α → α → Prop
    inst✝¹ : DecidableRel lt
    inst✝ : IsStrictOrder α lt
    a b : α
    ⊢ lt b a → Not (lt a b)
  -/
  exact fun hba hab ↦ (irrefl a) (_root_.trans hab hba)
  /-
    🎉 no goals
  -/


@[simp]
                                                                                           /-
                                                                                             α : Type u
                                                                                             lt : α → α → Prop
                                                                                             inst✝ : DecidableRel lt
                                                                                             a b : α
                                                                                             ⊢ Iff (Eq (cmpUsing lt a b) Ordering.eq) (And (Not (lt a b)) (Not (lt b a)))
                                                                                           -/
theorem cmpUsing_eq_eq (a b : α) : cmpUsing lt a b = Ordering.eq ↔ ¬lt a b ∧ ¬lt b a := by simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


