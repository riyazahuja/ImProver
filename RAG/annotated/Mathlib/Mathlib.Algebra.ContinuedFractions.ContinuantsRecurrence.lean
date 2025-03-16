theorem contsAux_recurrence {gp ppred pred : Pair K} (nth_s_eq : g.s.get? n = some gp)
    (nth_contsAux_eq : g.contsAux n = ppred)
    (succ_nth_contsAux_eq : g.contsAux (n + 1) = pred) :
    g.contsAux (n + 2) = ⟨gp.b * pred.a + gp.a * ppred.a, gp.b * pred.b + gp.a * ppred.b⟩ := by
  /-
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    gp ppred pred : GenContFract.Pair K
    nth_s_eq : Eq (g.s.get? n) (Option.some gp)
    nth_contsAux_eq : Eq (g.contsAux n) ppred
    succ_nth_contsAux_eq : Eq (g.contsAux (HAdd.hAdd n 1)) pred
    ⊢ Eq (g.contsAux (HAdd.hAdd n 2)) { a := HAdd.hAdd (HMul.hMul gp.b pred.a) (HM …
  -/
  simp [*, contsAux, nextConts, nextDen, nextNum]
  /-
    🎉 no goals
  -/


theorem conts_recurrenceAux {gp ppred pred : Pair K} (nth_s_eq : g.s.get? n = some gp)
    (nth_contsAux_eq : g.contsAux n = ppred)
    (succ_nth_contsAux_eq : g.contsAux (n + 1) = pred) :
    g.conts (n + 1) = ⟨gp.b * pred.a + gp.a * ppred.a, gp.b * pred.b + gp.a * ppred.b⟩ := by
  simp [nth_cont_eq_succ_nth_contAux,
    contsAux_recurrence nth_s_eq nth_contsAux_eq succ_nth_contsAux_eq]


/-- Shows that `Aₙ = bₙ * Aₙ₋₁ + aₙ * Aₙ₋₂` and `Bₙ = bₙ * Bₙ₋₁ + aₙ * Bₙ₋₂`. -/
theorem conts_recurrence {gp ppred pred : Pair K} (succ_nth_s_eq : g.s.get? (n + 1) = some gp)
    (nth_conts_eq : g.conts n = ppred) (succ_nth_conts_eq : g.conts (n + 1) = pred) :
    g.conts (n + 2) = ⟨gp.b * pred.a + gp.a * ppred.a, gp.b * pred.b + gp.a * ppred.b⟩ := by
  /-
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    gp ppred pred : GenContFract.Pair K
    succ_nth_s_eq : Eq (g.s.get? (HAdd.hAdd n 1)) (Option.some gp)
    nth_conts_eq : Eq (g.conts n) ppred
    succ_nth_conts_eq : Eq (g.conts (HAdd.hAdd n 1)) pred
    ⊢ Eq (g.conts (HAdd.hAdd n 2)) { a := HAdd.hAdd (HMul.hMul gp.b pred.a) (HMul. …
  -/
  rw [nth_cont_eq_succ_nth_contAux] at nth_conts_eq succ_nth_conts_eq
  /-
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    gp ppred pred : GenContFract.Pair K
    succ_nth_s_eq : Eq (g.s.get? (HAdd.hAdd n 1)) (Option.some gp)
    nth_conts_eq : Eq (g.contsAux (HAdd.hAdd n 1)) ppred
    succ_nth_conts_eq : Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd n 1) 1)) pred
    ⊢ Eq (g.conts (HAdd.hAdd n 2)) { a := HAdd.hAdd (HMul.hMul gp.b pred.a) (HMul. …
  -/
  exact conts_recurrenceAux succ_nth_s_eq nth_conts_eq succ_nth_conts_eq
  /-
    🎉 no goals
  -/


/-- Shows that `Aₙ = bₙ * Aₙ₋₁ + aₙ * Aₙ₋₂`. -/
theorem nums_recurrence {gp : Pair K} {ppredA predA : K}
    (succ_nth_s_eq : g.s.get? (n + 1) = some gp) (nth_num_eq : g.nums n = ppredA)
    (succ_nth_num_eq : g.nums (n + 1) = predA) :
    g.nums (n + 2) = gp.b * predA + gp.a * ppredA := by
  obtain ⟨ppredConts, nth_conts_eq, ⟨rfl⟩⟩ : ∃ conts, g.conts n = conts ∧ conts.a = ppredA :=
    exists_conts_a_of_num nth_num_eq
  obtain ⟨predConts, succ_nth_conts_eq, ⟨rfl⟩⟩ :
      ∃ conts, g.conts (n + 1) = conts ∧ conts.a = predA :=
    exists_conts_a_of_num succ_nth_num_eq
  /-
    case intro.intro.refl.intro.intro.refl
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    gp : GenContFract.Pair K
    succ_nth_s_eq : Eq (g.s.get? (HAdd.hAdd n 1)) (Option.some gp)
    ppredConts : GenContFract.Pair K
    nth_conts_eq : Eq (g.conts n) ppredConts
    nth_num_eq : Eq (g.nums n) ppredConts.a
    predConts : GenContFract.Pair K
    succ_nth_conts_eq : Eq (g.conts (HAdd.hAdd n 1)) predConts
    succ_nth_num_eq : Eq (g.nums (HAdd.hAdd n 1)) predConts.a
    ⊢ Eq (g.nums (HAdd.hAdd n 2)) (HAdd.hAdd (HMul.hMul gp.b predConts.a) (HMul.hM …
  -/
  rw [num_eq_conts_a, conts_recurrence succ_nth_s_eq nth_conts_eq succ_nth_conts_eq]
  /-
    🎉 no goals
  -/


/-- Shows that `Bₙ = bₙ * Bₙ₋₁ + aₙ * Bₙ₋₂`. -/
theorem dens_recurrence {gp : Pair K} {ppredB predB : K}
    (succ_nth_s_eq : g.s.get? (n + 1) = some gp) (nth_den_eq : g.dens n = ppredB)
    (succ_nth_den_eq : g.dens (n + 1) = predB) :
    g.dens (n + 2) = gp.b * predB + gp.a * ppredB := by
  obtain ⟨ppredConts, nth_conts_eq, ⟨rfl⟩⟩ : ∃ conts, g.conts n = conts ∧ conts.b = ppredB :=
    exists_conts_b_of_den nth_den_eq
  obtain ⟨predConts, succ_nth_conts_eq, ⟨rfl⟩⟩ :
      ∃ conts, g.conts (n + 1) = conts ∧ conts.b = predB :=
    exists_conts_b_of_den succ_nth_den_eq
  /-
    case intro.intro.refl.intro.intro.refl
    K : Type u_1
    g : GenContFract K
    n : Nat
    inst✝ : DivisionRing K
    gp : GenContFract.Pair K
    succ_nth_s_eq : Eq (g.s.get? (HAdd.hAdd n 1)) (Option.some gp)
    ppredConts : GenContFract.Pair K
    nth_conts_eq : Eq (g.conts n) ppredConts
    nth_den_eq : Eq (g.dens n) ppredConts.b
    predConts : GenContFract.Pair K
    succ_nth_conts_eq : Eq (g.conts (HAdd.hAdd n 1)) predConts
    succ_nth_den_eq : Eq (g.dens (HAdd.hAdd n 1)) predConts.b
    ⊢ Eq (g.dens (HAdd.hAdd n 2)) (HAdd.hAdd (HMul.hMul gp.b predConts.b) (HMul.hM …
  -/
  rw [den_eq_conts_b, conts_recurrence succ_nth_s_eq nth_conts_eq succ_nth_conts_eq]
  /-
    🎉 no goals
  -/


