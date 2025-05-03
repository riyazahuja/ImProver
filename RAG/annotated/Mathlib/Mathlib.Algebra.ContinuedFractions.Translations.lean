                                                                                      /-
                                                                                        α : Type u_1
                                                                                        g : GenContFract α
                                                                                        n : Nat
                                                                                        ⊢ Iff (g.TerminatedAt n) (g.s.TerminatedAt n)
                                                                                      -/
theorem terminatedAt_iff_s_terminatedAt : g.TerminatedAt n ↔ g.s.TerminatedAt n := by rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


                                                                             /-
                                                                               α : Type u_1
                                                                               g : GenContFract α
                                                                               n : Nat
                                                                               ⊢ Iff (g.TerminatedAt n) (Eq (g.s.get? n) Option.none)
                                                                             -/
theorem terminatedAt_iff_s_none : g.TerminatedAt n ↔ g.s.get? n = none := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem partNum_none_iff_s_none : g.partNums.get? n = none ↔ g.s.get? n = none := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    ⊢ Iff (Eq (g.partNums.get? n) Option.none) (Eq (g.s.get? n) Option.none)
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases s_nth_eq : g.s.get? n <;> simp [partNums, s_nth_eq]
                                  /-
                                    🎉 no goals
                                  -/


theorem terminatedAt_iff_partNum_none : g.TerminatedAt n ↔ g.partNums.get? n = none := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    ⊢ Iff (g.TerminatedAt n) (Eq (g.partNums.get? n) Option.none)
  -/
  rw [terminatedAt_iff_s_none, partNum_none_iff_s_none]
  /-
    🎉 no goals
  -/


theorem partDen_none_iff_s_none : g.partDens.get? n = none ↔ g.s.get? n = none := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    ⊢ Iff (Eq (g.partDens.get? n) Option.none) (Eq (g.s.get? n) Option.none)
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases s_nth_eq : g.s.get? n <;> simp [partDens, s_nth_eq]
                                  /-
                                    🎉 no goals
                                  -/


theorem terminatedAt_iff_partDen_none : g.TerminatedAt n ↔ g.partDens.get? n = none := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    ⊢ Iff (g.TerminatedAt n) (Eq (g.partDens.get? n) Option.none)
  -/
  rw [terminatedAt_iff_s_none, partDen_none_iff_s_none]
  /-
    🎉 no goals
  -/


theorem partNum_eq_s_a {gp : Pair α} (s_nth_eq : g.s.get? n = some gp) :
                                        /-
                                          α : Type u_1
                                          g : GenContFract α
                                          n : Nat
                                          gp : GenContFract.Pair α
                                          s_nth_eq : Eq (g.s.get? n) (Option.some gp)
                                          ⊢ Eq (g.partNums.get? n) (Option.some gp.a)
                                        -/
    g.partNums.get? n = some gp.a := by simp [partNums, s_nth_eq]
                                        /-
                                          🎉 no goals
                                        -/


theorem partDen_eq_s_b {gp : Pair α} (s_nth_eq : g.s.get? n = some gp) :
                                        /-
                                          α : Type u_1
                                          g : GenContFract α
                                          n : Nat
                                          gp : GenContFract.Pair α
                                          s_nth_eq : Eq (g.s.get? n) (Option.some gp)
                                          ⊢ Eq (g.partDens.get? n) (Option.some gp.b)
                                        -/
    g.partDens.get? n = some gp.b := by simp [partDens, s_nth_eq]
                                        /-
                                          🎉 no goals
                                        -/


theorem exists_s_a_of_partNum {a : α} (nth_partNum_eq : g.partNums.get? n = some a) :
    ∃ gp, g.s.get? n = some gp ∧ gp.a = a := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    a : α
    nth_partNum_eq : Eq (g.partNums.get? n) (Option.some a)
    ⊢ Exists fun gp => And (Eq (g.s.get? n) (Option.some gp)) (Eq gp.a a)
  -/
  simpa [partNums, Stream'.Seq.map_get?] using nth_partNum_eq
  /-
    🎉 no goals
  -/


theorem exists_s_b_of_partDen {b : α}
    (nth_partDen_eq : g.partDens.get? n = some b) :
    ∃ gp, g.s.get? n = some gp ∧ gp.b = b := by
  /-
    α : Type u_1
    g : GenContFract α
    n : Nat
    b : α
    nth_partDen_eq : Eq (g.partDens.get? n) (Option.some b)
    ⊢ Exists fun gp => And (Eq (g.s.get? n) (Option.some gp)) (Eq gp.b b)
  -/
  simpa [partDens, Stream'.Seq.map_get?] using nth_partDen_eq
  /-
    🎉 no goals
  -/


theorem nth_cont_eq_succ_nth_contAux : g.conts n = g.contsAux (n + 1) :=
  rfl


theorem num_eq_conts_a : g.nums n = (g.conts n).a :=
  rfl


theorem den_eq_conts_b : g.dens n = (g.conts n).b :=
  rfl


theorem conv_eq_num_div_den : g.convs n = g.nums n / g.dens n :=
  rfl


theorem conv_eq_conts_a_div_conts_b :
    g.convs n = (g.conts n).a / (g.conts n).b :=
  rfl


theorem exists_conts_a_of_num {A : K} (nth_num_eq : g.nums n = A) :
                                                   /-
                                                     K : Type u_1
                                                     g : GenContFract K
                                                     n : Nat
                                                     inst✝ : DivisionRing K
                                                     A : K
                                                     nth_num_eq : Eq (g.nums n) A
                                                     ⊢ Exists fun conts => And (Eq (g.conts n) conts) (Eq conts.a A)
                                                   -/
    ∃ conts, g.conts n = conts ∧ conts.a = A := by simpa
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem exists_conts_b_of_den {B : K} (nth_denom_eq : g.dens n = B) :
                                                   /-
                                                     K : Type u_1
                                                     g : GenContFract K
                                                     n : Nat
                                                     inst✝ : DivisionRing K
                                                     B : K
                                                     nth_denom_eq : Eq (g.dens n) B
                                                     ⊢ Exists fun conts => And (Eq (g.conts n) conts) (Eq conts.b B)
                                                   -/
    ∃ conts, g.conts n = conts ∧ conts.b = B := by simpa
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem zeroth_contAux_eq_one_zero : g.contsAux 0 = ⟨1, 0⟩ :=
  rfl


@[simp]
theorem first_contAux_eq_h_one : g.contsAux 1 = ⟨g.h, 1⟩ :=
  rfl


@[simp]
theorem zeroth_cont_eq_h_one : g.conts 0 = ⟨g.h, 1⟩ :=
  rfl


@[simp]
theorem zeroth_num_eq_h : g.nums 0 = g.h :=
  rfl


@[simp]
theorem zeroth_den_eq_one : g.dens 0 = 1 :=
  rfl


@[simp]
theorem zeroth_conv_eq_h : g.convs 0 = g.h := by
  /-
    K : Type u_1
    g : GenContFract K
    inst✝ : DivisionRing K
    ⊢ Eq (g.convs 0) g.h
  -/
  simp [conv_eq_num_div_den, num_eq_conts_a, den_eq_conts_b, div_one]
  /-
    🎉 no goals
  -/


theorem second_contAux_eq {gp : Pair K} (zeroth_s_eq : g.s.get? 0 = some gp) :
    g.contsAux 2 = ⟨gp.b * g.h + gp.a, gp.b⟩ := by
  /-
    K : Type u_1
    g : GenContFract K
    inst✝ : DivisionRing K
    gp : GenContFract.Pair K
    zeroth_s_eq : Eq (g.s.get? 0) (Option.some gp)
    ⊢ Eq (g.contsAux 2) { a := HAdd.hAdd (HMul.hMul gp.b g.h) gp.a, b := gp.b }
  -/
  simp [zeroth_s_eq, contsAux, nextConts, nextDen, nextNum]
  /-
    🎉 no goals
  -/


theorem first_cont_eq {gp : Pair K} (zeroth_s_eq : g.s.get? 0 = some gp) :
    g.conts 1 = ⟨gp.b * g.h + gp.a, gp.b⟩ := by
  /-
    K : Type u_1
    g : GenContFract K
    inst✝ : DivisionRing K
    gp : GenContFract.Pair K
    zeroth_s_eq : Eq (g.s.get? 0) (Option.some gp)
    ⊢ Eq (g.conts 1) { a := HAdd.hAdd (HMul.hMul gp.b g.h) gp.a, b := gp.b }
  -/
  simp [nth_cont_eq_succ_nth_contAux]
  -- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
  -- simp used to work here, but now it can't figure out that 1 + 1 = 2
  /-
    K : Type u_1
    g : GenContFract K
    inst✝ : DivisionRing K
    gp : GenContFract.Pair K
    zeroth_s_eq : Eq (g.s.get? 0) (Option.some gp)
    ⊢ Eq (g.contsAux 2) { a := HAdd.hAdd (HMul.hMul gp.b g.h) gp.a, b := gp.b }
  -/
  convert second_contAux_eq zeroth_s_eq
  /-
    🎉 no goals
  -/


theorem first_num_eq {gp : Pair K} (zeroth_s_eq : g.s.get? 0 = some gp) :
                                       /-
                                         K : Type u_1
                                         g : GenContFract K
                                         inst✝ : DivisionRing K
                                         gp : GenContFract.Pair K
                                         zeroth_s_eq : Eq (g.s.get? 0) (Option.some gp)
                                         ⊢ Eq (g.nums 1) (HAdd.hAdd (HMul.hMul gp.b g.h) gp.a)
                                       -/
    g.nums 1 = gp.b * g.h + gp.a := by simp [num_eq_conts_a, first_cont_eq zeroth_s_eq]
                                       /-
                                         🎉 no goals
                                       -/


theorem first_den_eq {gp : Pair K} (zeroth_s_eq : g.s.get? 0 = some gp) :
                          /-
                            K : Type u_1
                            g : GenContFract K
                            inst✝ : DivisionRing K
                            gp : GenContFract.Pair K
                            zeroth_s_eq : Eq (g.s.get? 0) (Option.some gp)
                            ⊢ Eq (g.dens 1) gp.b
                          -/
    g.dens 1 = gp.b := by simp [den_eq_conts_b, first_cont_eq zeroth_s_eq]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem zeroth_conv'Aux_eq_zero {s : Stream'.Seq <| Pair K} :
    convs'Aux s 0 = (0 : K) :=
  rfl


@[simp]
                                                   /-
                                                     K : Type u_1
                                                     g : GenContFract K
                                                     inst✝ : DivisionRing K
                                                     ⊢ Eq (g.convs' 0) g.h
                                                   -/
theorem zeroth_conv'_eq_h : g.convs' 0 = g.h := by simp [convs']
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem convs'Aux_succ_none {s : Stream'.Seq (Pair K)} (h : s.head = none) (n : ℕ) :
                                  /-
                                    K : Type u_1
                                    inst✝ : DivisionRing K
                                    s : Stream'.Seq (GenContFract.Pair K)
                                    h : Eq s.head Option.none
                                    n : Nat
                                    ⊢ Eq (GenContFract.convs'Aux s (HAdd.hAdd n 1)) 0
                                  -/
    convs'Aux s (n + 1) = 0 := by simp [convs'Aux, h]
                                  /-
                                    🎉 no goals
                                  -/


theorem convs'Aux_succ_some {s : Stream'.Seq (Pair K)} {p : Pair K} (h : s.head = some p)
    (n : ℕ) : convs'Aux s (n + 1) = p.a / (p.b + convs'Aux s.tail n) := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    s : Stream'.Seq (GenContFract.Pair K)
    p : GenContFract.Pair K
    h : Eq s.head (Option.some p)
    n : Nat
    ⊢ Eq (GenContFract.convs'Aux s (HAdd.hAdd n 1)) (HDiv.hDiv p.a (HAdd.hAdd p.b  …
  -/
  simp [convs'Aux, h]
  /-
    🎉 no goals
  -/


