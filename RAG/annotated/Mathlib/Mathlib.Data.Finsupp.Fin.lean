/-- `tail` for maps `Fin (n + 1) →₀ M`. See `Fin.tail` for more details. -/
def tail (s : Fin (n + 1) →₀ M) : Fin n →₀ M :=
  Finsupp.equivFunOnFinite.symm (Fin.tail s)


/-- `cons` for maps `Fin n →₀ M`. See `Fin.cons` for more details. -/
def cons (y : M) (s : Fin n →₀ M) : Fin (n + 1) →₀ M :=
  Finsupp.equivFunOnFinite.symm (Fin.cons y s : Fin (n + 1) → M)


theorem tail_apply : tail t i = t i.succ :=
  rfl


@[simp]
theorem cons_zero : cons y s 0 = y :=
  rfl


@[simp]
theorem cons_succ : cons y s i.succ = s i :=
  -- Porting note: was Fin.cons_succ _ _ _
  rfl


@[simp]
theorem tail_cons : tail (cons y s) = s :=
                  /-
                    n : Nat
                    M : Type u_1
                    inst✝ : Zero M
                    y : M
                    s : Finsupp (Fin n) M
                    k : Fin n
                    ⊢ Eq ((Finsupp.cons y s).tail k) (s k)
                  -/
  ext fun k => by simp only [tail_apply, cons_succ]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem cons_tail : cons (t 0) (tail t) = t := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    t : Finsupp (Fin (HAdd.hAdd n 1)) M
    ⊢ Eq (Finsupp.cons (t 0) t.tail) t
  -/
  ext a
  /-
    case h
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    t : Finsupp (Fin (HAdd.hAdd n 1)) M
    a : Fin (HAdd.hAdd n 1)
    ⊢ Eq ((Finsupp.cons (t 0) t.tail) a) (t a)
  -/
  by_cases c_a : a = 0
    /-
      case pos
      n : Nat
      M : Type u_1
      inst✝ : Zero M
      t : Finsupp (Fin (HAdd.hAdd n 1)) M
      a : Fin (HAdd.hAdd n 1)
      c_a : Eq a 0
      ⊢ Eq ((Finsupp.cons (t 0) t.tail) a) (t a)
    -/
  · rw [c_a, cons_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      M : Type u_1
      inst✝ : Zero M
      t : Finsupp (Fin (HAdd.hAdd n 1)) M
      a : Fin (HAdd.hAdd n 1)
      c_a : Not (Eq a 0)
      ⊢ Eq ((Finsupp.cons (t 0) t.tail) a) (t a)
    -/
  · rw [← Fin.succ_pred a c_a, cons_succ, ← tail_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem cons_zero_zero : cons 0 (0 : Fin n →₀ M) = 0 := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    ⊢ Eq (Finsupp.cons 0 0) 0
  -/
  ext a
  /-
    case h
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    a : Fin (HAdd.hAdd n 1)
    ⊢ Eq ((Finsupp.cons 0 0) a) (0 a)
  -/
  by_cases c : a = 0
    /-
      case pos
      n : Nat
      M : Type u_1
      inst✝ : Zero M
      a : Fin (HAdd.hAdd n 1)
      c : Eq a 0
      ⊢ Eq ((Finsupp.cons 0 0) a) (0 a)
    -/
  · simp [c]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      M : Type u_1
      inst✝ : Zero M
      a : Fin (HAdd.hAdd n 1)
      c : Not (Eq a 0)
      ⊢ Eq ((Finsupp.cons 0 0) a) (0 a)
    -/
  · rw [← Fin.succ_pred a c, cons_succ]
    /-
      case neg
      n : Nat
      M : Type u_1
      inst✝ : Zero M
      a : Fin (HAdd.hAdd n 1)
      c : Not (Eq a 0)
      ⊢ Eq (0 (a.pred c)) (0 (a.pred c).succ)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem cons_ne_zero_of_left (h : y ≠ 0) : cons y s ≠ 0 := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    h : Ne y 0
    ⊢ Ne (Finsupp.cons y s) 0
  -/
  contrapose! h with c
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    c : Eq (Finsupp.cons y s) 0
    ⊢ Eq y 0
  -/
  rw [← cons_zero y s, c, Finsupp.coe_zero, Pi.zero_apply]
  /-
    🎉 no goals
  -/


theorem cons_ne_zero_of_right (h : s ≠ 0) : cons y s ≠ 0 := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    h : Ne s 0
    ⊢ Ne (Finsupp.cons y s) 0
  -/
  contrapose! h with c
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    c : Eq (Finsupp.cons y s) 0
    ⊢ Eq s 0
  -/
  ext a
  /-
    case h
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    c : Eq (Finsupp.cons y s) 0
    a : Fin n
    ⊢ Eq (s a) (0 a)
  -/
  simp [← cons_succ a y s, c]
  /-
    🎉 no goals
  -/


theorem cons_ne_zero_iff : cons y s ≠ 0 ↔ y ≠ 0 ∨ s ≠ 0 := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    ⊢ Iff (Ne (Finsupp.cons y s) 0) (Or (Ne y 0) (Ne s 0))
  -/
  refine ⟨fun h => ?_, fun h => h.casesOn cons_ne_zero_of_left cons_ne_zero_of_right⟩
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    h : Ne (Finsupp.cons y s) 0
    ⊢ Or (Ne y 0) (Ne s 0)
  -/
  refine imp_iff_not_or.1 fun h' c => h ?_
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    h : Ne (Finsupp.cons y s) 0
    h' : Eq y 0
    c : Eq s 0
    ⊢ Eq (Finsupp.cons y s) 0
  -/
  rw [h', c, Finsupp.cons_zero_zero]
  /-
    🎉 no goals
  -/


lemma cons_support : (s.cons y).support ⊆ insert 0 (s.support.map (Fin.succEmb n)) := by
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    ⊢ HasSubset.Subset (Finsupp.cons y s).support (Insert.insert 0 (Finset.map (Fi …
  -/
  intro i hi
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    i : Fin (HAdd.hAdd n 1)
    hi : Membership.mem (Finsupp.cons y s).support i
    ⊢ Membership.mem (Insert.insert 0 (Finset.map (Fin.succEmb n) s.support)) i
  -/
  suffices i = 0 ∨ ∃ a, ¬s a = 0 ∧ a.succ = i by simpa
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    i : Fin (HAdd.hAdd n 1)
    hi : Membership.mem (Finsupp.cons y s).support i
    ⊢ Or (Eq i 0) (Exists fun a => And (Not (Eq (s a) 0)) (Eq a.succ i))
  -/
  apply (Fin.eq_zero_or_eq_succ i).imp id (Exists.imp _)
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    i : Fin (HAdd.hAdd n 1)
    hi : Membership.mem (Finsupp.cons y s).support i
    ⊢ ∀ (a : Fin n), Eq i a.succ → And (Not (Eq (s a) 0)) (Eq a.succ i)
  -/
  rintro i rfl
  /-
    n : Nat
    M : Type u_1
    inst✝ : Zero M
    y : M
    s : Finsupp (Fin n) M
    i : Fin n
    hi : Membership.mem (Finsupp.cons y s).support i.succ
    ⊢ And (Not (Eq (s i) 0)) (Eq i.succ i.succ)
  -/
  simpa [Finsupp.mem_support_iff] using hi
  /-
    🎉 no goals
  -/


lemma cons_right_injective {n : ℕ} {M : Type*} [Zero M] (y : M) :
    Injective (Finsupp.cons y : (Fin n →₀ M) → Fin (n + 1) →₀ M) :=
  (equivFunOnFinite.symm.injective.comp ((Fin.cons_right_injective _).comp DFunLike.coe_injective))


