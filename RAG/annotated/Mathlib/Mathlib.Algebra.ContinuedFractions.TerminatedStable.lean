/-- If a gcf terminated at position `n`, it also terminated at `m ≥ n`. -/
theorem terminated_stable (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.TerminatedAt m :=
  g.s.terminated_stable n_le_m terminatedAt_n


theorem contsAux_stable_step_of_terminated (terminatedAt_n : g.TerminatedAt n) :
    g.contsAux (n + 2) = g.contsAux (n + 1) := by
  /-
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    terminatedAt_n : g.TerminatedAt n
    ⊢ Eq (g.contsAux (HAdd.hAdd n 2)) (g.contsAux (HAdd.hAdd n 1))
  -/
  rw [terminatedAt_iff_s_none] at terminatedAt_n
  /-
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    terminatedAt_n : Eq (g.s.get? n) Option.none
    ⊢ Eq (g.contsAux (HAdd.hAdd n 2)) (g.contsAux (HAdd.hAdd n 1))
  -/
  simp only [contsAux, Nat.add_eq, Nat.add_zero, terminatedAt_n]
  /-
    🎉 no goals
  -/


theorem contsAux_stable_of_terminated (n_lt_m : n < m) (terminatedAt_n : g.TerminatedAt n) :
    g.contsAux m = g.contsAux (n + 1) := by
  /-
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_lt_m : LT.lt n m
    terminatedAt_n : g.TerminatedAt n
    ⊢ Eq (g.contsAux m) (g.contsAux (HAdd.hAdd n 1))
  -/
  refine Nat.le_induction rfl (fun k hnk hk => ?_) _ n_lt_m
  /-
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_lt_m : LT.lt n m
    terminatedAt_n : g.TerminatedAt n
    k : Nat
    hnk : LE.le n.succ k
    hk : Eq (g.contsAux k) (g.contsAux (HAdd.hAdd n 1))
    ⊢ Eq (g.contsAux (HAdd.hAdd k 1)) (g.contsAux (HAdd.hAdd n 1))
  -/
  rcases Nat.exists_eq_add_of_lt hnk with ⟨k, rfl⟩
  /-
    case intro
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_lt_m : LT.lt n m
    terminatedAt_n : g.TerminatedAt n
    k : Nat
    hnk : LE.le n.succ (HAdd.hAdd (HAdd.hAdd n k) 1)
    hk : Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd n k) 1)) (g.contsAux (HAdd.hAdd n 1))
    ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n k) 1) 1)) (g.contsAux (HAd …
  -/
  refine (contsAux_stable_step_of_terminated ?_).trans hk
  /-
    case intro
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_lt_m : LT.lt n m
    terminatedAt_n : g.TerminatedAt n
    k : Nat
    hnk : LE.le n.succ (HAdd.hAdd (HAdd.hAdd n k) 1)
    hk : Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd n k) 1)) (g.contsAux (HAdd.hAdd n 1))
    ⊢ g.TerminatedAt (HAdd.hAdd n k)
  -/
  exact terminated_stable (Nat.le_add_right _ _) terminatedAt_n
  /-
    🎉 no goals
  -/


theorem convs'Aux_stable_step_of_terminated {s : Stream'.Seq <| Pair K}
    (terminatedAt_n : s.TerminatedAt n) : convs'Aux s (n + 1) = convs'Aux s n := by
  /-
    K : Type u_1
    n : Nat
    inst✝ : DivisionRing K
    s : Stream'.Seq (GenContFract.Pair K)
    terminatedAt_n : s.TerminatedAt n
    ⊢ Eq (GenContFract.convs'Aux s (HAdd.hAdd n 1)) (GenContFract.convs'Aux s n)
  -/
  change s.get? n = none at terminatedAt_n
  induction n generalizing s with
  | zero => simp only [convs'Aux, terminatedAt_n, Stream'.Seq.head]
  | succ n IH =>
    cases s_head_eq : s.head with
    | none => simp only [convs'Aux, s_head_eq]
    | some gp_head =>
      have : s.tail.TerminatedAt n := by
        simp only [Stream'.Seq.TerminatedAt, s.get?_tail, terminatedAt_n]
      have := IH this
      rw [convs'Aux] at this
      simp [this, Nat.add_eq, add_zero, convs'Aux, s_head_eq]


theorem convs'Aux_stable_of_terminated {s : Stream'.Seq <| Pair K} (n_le_m : n ≤ m)
    (terminatedAt_n : s.TerminatedAt n) : convs'Aux s m = convs'Aux s n := by
  induction n_le_m with
  | refl => rfl
  | step n_le_m IH =>
    refine (convs'Aux_stable_step_of_terminated (?_)).trans IH
    exact s.terminated_stable n_le_m terminatedAt_n


theorem conts_stable_of_terminated (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.conts m = g.conts n := by
  simp only [nth_cont_eq_succ_nth_contAux,
    contsAux_stable_of_terminated (Nat.pred_le_iff.mp n_le_m) terminatedAt_n]


theorem nums_stable_of_terminated (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.nums m = g.nums n := by
  /-
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_le_m : LE.le n m
    terminatedAt_n : g.TerminatedAt n
    ⊢ Eq (g.nums m) (g.nums n)
  -/
  simp only [num_eq_conts_a, conts_stable_of_terminated n_le_m terminatedAt_n]
  /-
    🎉 no goals
  -/


theorem dens_stable_of_terminated (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.dens m = g.dens n := by
  /-
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_le_m : LE.le n m
    terminatedAt_n : g.TerminatedAt n
    ⊢ Eq (g.dens m) (g.dens n)
  -/
  simp only [den_eq_conts_b, conts_stable_of_terminated n_le_m terminatedAt_n]
  /-
    🎉 no goals
  -/


theorem convs_stable_of_terminated (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.convs m = g.convs n := by
  simp only [convs, dens_stable_of_terminated n_le_m terminatedAt_n,
    nums_stable_of_terminated n_le_m terminatedAt_n]


theorem convs'_stable_of_terminated (n_le_m : n ≤ m) (terminatedAt_n : g.TerminatedAt n) :
    g.convs' m = g.convs' n := by
  /-
    K : Type u_1
    g : GenContFract K
    n m : Nat
    inst✝ : DivisionRing K
    n_le_m : LE.le n m
    terminatedAt_n : g.TerminatedAt n
    ⊢ Eq (g.convs' m) (g.convs' n)
  -/
  simp only [convs', convs'Aux_stable_of_terminated n_le_m terminatedAt_n]
  /-
    🎉 no goals
  -/


