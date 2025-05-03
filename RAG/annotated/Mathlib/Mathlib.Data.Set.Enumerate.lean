/-- Given a choice function `sel`, enumerates the elements of a set in the order
`a 0 = sel s`, `a 1 = sel (s \ {a 0})`, `a 2 = sel (s \ {a 0, a 1})`, ... and stops when
`sel (s \ {a 0, ..., a n}) = none`. Note that we don't require `sel` to be a choice function. -/
def enumerate : Set α → ℕ → Option α
  | s, 0 => sel s
  | s, n + 1 => do
    let a ← sel s
    enumerate (s \ {a}) n


theorem enumerate_eq_none_of_sel {s : Set α} (h : sel s = none) : ∀ {n}, enumerate sel s n = none
            /-
              α : Type u_1
              sel : Set α → Option α
              s : Set α
              h : Eq (sel s) Option.none
              ⊢ Eq (Set.enumerate sel s 0) Option.none
            -/
  | 0 => by simp [h, enumerate]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  sel : Set α → Option α
                  s : Set α
                  h : Eq (sel s) Option.none
                  n : Nat
                  ⊢ Eq (Set.enumerate sel s (HAdd.hAdd n 1)) Option.none
                -/
  | n + 1 => by simp [h, enumerate]
                /-
                  🎉 no goals
                -/


theorem enumerate_eq_none :
    ∀ {s n₁ n₂}, enumerate sel s n₁ = none → n₁ ≤ n₂ → enumerate sel s n₂ = none
  | _, 0, _ => fun h _ ↦ enumerate_eq_none_of_sel sel h
  | s, n + 1, m => fun h hm ↦ by
    /-
      α : Type u_1
      sel : Set α → Option α
      s : Set α
      n m : Nat
      h : Eq (Set.enumerate sel s (HAdd.hAdd n 1)) Option.none
      hm : LE.le (HAdd.hAdd n 1) m
      ⊢ Eq (Set.enumerate sel s m) Option.none
    -/
    cases hs : sel s
      /-
        case none
        α : Type u_1
        sel : Set α → Option α
        s : Set α
        n m : Nat
        h : Eq (Set.enumerate sel s (HAdd.hAdd n 1)) Option.none
        hm : LE.le (HAdd.hAdd n 1) m
        hs : Eq (sel s) Option.none
        ⊢ Eq (Set.enumerate sel s m) Option.none
      -/
    · exact enumerate_eq_none_of_sel sel hs
      /-
        🎉 no goals
      -/
    · cases m with
      | zero => contradiction
      | succ m' =>
        simp? [hs, enumerate] at h ⊢ says
          simp only [enumerate, hs, Option.bind_eq_bind, Option.some_bind] at h ⊢
        have hm : n ≤ m' := Nat.le_of_succ_le_succ hm
        exact enumerate_eq_none h hm


theorem enumerate_mem (h_sel : ∀ s a, sel s = some a → a ∈ s) :
    ∀ {s n a}, enumerate sel s n = some a → a ∈ s
  | s, 0, a => h_sel s a
  | s, n + 1, a => by
    cases h : sel s with
    | none => simp [enumerate_eq_none_of_sel, h]
    | some a' =>
      simp only [enumerate, h, Nat.add_eq, add_zero]
      exact fun h' : enumerate sel (s \ {a'}) n = some a ↦
        have : a ∈ s \ {a'} := enumerate_mem h_sel h'
        this.left


theorem enumerate_inj {n₁ n₂ : ℕ} {a : α} {s : Set α} (h_sel : ∀ s a, sel s = some a → a ∈ s)
    (h₁ : enumerate sel s n₁ = some a) (h₂ : enumerate sel s n₂ = some a) : n₁ = n₂ := by
  /- Porting note: The `rcase, on_goal, all_goals` has been used instead of
     the not-yet-ported `wlog` -/
  /-
    α : Type u_1
    sel : Set α → Option α
    n₁ n₂ : Nat
    a : α
    s : Set α
    h_sel : ∀ (s : Set α) (a : α), Eq (sel s) (Option.some a) → Membership.mem s a
    h₁ : Eq (Set.enumerate sel s n₁) (Option.some a)
    h₂ : Eq (Set.enumerate sel s n₂) (Option.some a)
    ⊢ Eq n₁ n₂
  -/
  rcases le_total n₁ n₂ with (hn|hn)
  /-
    case inl
    α : Type u_1
    sel : Set α → Option α
    n₁ n₂ : Nat
    a : α
    s : Set α
    h_sel : ∀ (s : Set α) (a : α), Eq (sel s) (Option.some a) → Membership.mem s a
    h₁ : Eq (Set.enumerate sel s n₁) (Option.some a)
    h₂ : Eq (Set.enumerate sel s n₂) (Option.some a)
    hn : LE.le n₁ n₂
    ⊢ Eq n₁ n₂
  -/
  on_goal 2 => swap_var n₁ ↔ n₂, h₁ ↔ h₂
  all_goals
    rcases Nat.le.dest hn with ⟨m, rfl⟩
    clear hn
    induction n₁ generalizing s with
    | zero =>
      cases m with
      | zero => rfl
      | succ m =>
        have h' : enumerate sel (s \ {a}) m = some a := by
          simp_all only [enumerate, Nat.add_eq, zero_add]; exact h₂
        have : a ∈ s \ {a} := enumerate_mem sel h_sel h'
        simp_all [Set.mem_diff_singleton]
    | succ k ih =>
      cases h : sel s with
      /- Porting note: The original covered both goals with just `simp_all <;> tauto` -/
      | none =>
        simp_all only [add_comm, self_eq_add_left, Nat.add_succ, enumerate_eq_none_of_sel _ h,
          reduceCtorEq]
      | some =>
        simp_all only [add_comm, self_eq_add_left, enumerate, Option.some.injEq,
                       Nat.add_succ, Nat.succ.injEq]
        exact ih h₁ h₂


