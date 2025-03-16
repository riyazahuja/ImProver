theorem of_convs_eq_convs' : (of v).convs = (of v).convs' :=
  @ContFract.convs_eq_convs' _ _ (ContFract.of v)


/-- The recurrence relation for the convergents of the continued fraction expansion
of an element `v` of `K` in terms of the convergents of the inverse of its fractional part.
-/
theorem convs_succ (n : ℕ) :
    (of v).convs (n + 1) = ⌊v⌋ + 1 / (of (Int.fract v)⁻¹).convs n := by
  /-
    K : Type u_1
    v : K
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    n : Nat
    ⊢ Eq ((GenContFract.of v).convs (HAdd.hAdd n 1)) (HAdd.hAdd (↑(Int.floor v)) ( …
  -/
  rw [of_convs_eq_convs', convs'_succ, of_convs_eq_convs']
  /-
    🎉 no goals
  -/


theorem of_convergence_epsilon :
    ∀ ε > (0 : K), ∃ N : ℕ, ∀ n ≥ N, |v - (of v).convs n| < ε := by
  /-
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ⊢ ∀ (ε : K), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (abs ( …
  -/
  intro ε ε_pos
  -- use the archimedean property to obtain a suitable N
  /-
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (abs (HSub.hSub v ((GenContFr …
  -/
  rcases (exists_nat_gt (1 / ε) : ∃ N' : ℕ, 1 / ε < N') with ⟨N', one_div_ε_lt_N'⟩
  /-
    case intro
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    N' : Nat
    one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (abs (HSub.hSub v ((GenContFr …
  -/
  let N := max N' 5
  -- set minimum to 5 to have N ≤ fib N work
  /-
    case intro
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    N' : Nat
    one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
    N : Nat := Max.max N' 5
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (abs (HSub.hSub v ((GenContFr …
  -/
  exists N
  /-
    case intro
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    N' : Nat
    one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
    N : Nat := Max.max N' 5
    ⊢ ∀ (n : Nat), GE.ge n N → LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs  …
  -/
  intro n n_ge_N
  /-
    case intro
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    N' : Nat
    one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
    N : Nat := Max.max N' 5
    n : Nat
    n_ge_N : GE.ge n N
    ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
  -/
  let g := of v
  /-
    case intro
    K : Type u_1
    v : K
    inst✝² : LinearOrderedField K
    inst✝¹ : FloorRing K
    inst✝ : Archimedean K
    ε : K
    ε_pos : GT.gt ε 0
    N' : Nat
    one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
    N : Nat := Max.max N' 5
    n : Nat
    n_ge_N : GE.ge n N
    g : GenContFract K := GenContFract.of v
    ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
  -/
  rcases Decidable.em (g.TerminatedAt n) with terminatedAt_n | not_terminatedAt_n
    /-
      case intro.inl
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      terminatedAt_n : g.TerminatedAt n
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
  · have : v = g.convs n := of_correctness_of_terminatedAt terminatedAt_n
    /-
      case intro.inl
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      terminatedAt_n : g.TerminatedAt n
      this : Eq v (g.convs n)
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
    have : v - g.convs n = 0 := sub_eq_zero.mpr this
    /-
      case intro.inl
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      terminatedAt_n : g.TerminatedAt n
      this✝ : Eq v (g.convs n)
      this : Eq (HSub.hSub v (g.convs n)) 0
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
    rw [this]
    /-
      case intro.inl
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      terminatedAt_n : g.TerminatedAt n
      this✝ : Eq v (g.convs n)
      this : Eq (HSub.hSub v (g.convs n)) 0
      ⊢ LT.lt (abs 0) ε
    -/
    exact mod_cast ε_pos
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
  · let B := g.dens n
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
    let nB := g.dens (n + 1)
    have abs_v_sub_conv_le : |v - g.convs n| ≤ 1 / (B * nB) :=
      abs_sub_convs_le not_terminatedAt_n
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      ⊢ LT.lt (abs (HSub.hSub v ((GenContFract.of v).convs n))) ε
    -/
    suffices 1 / (B * nB) < ε from lt_of_le_of_lt abs_v_sub_conv_le this
    -- show that `0 < (B * nB)` and then multiply by `B * nB` to get rid of the division
    have nB_ineq : (fib (n + 2) : K) ≤ nB :=
      haveI : ¬g.TerminatedAt (n + 1 - 1) := not_terminatedAt_n
      succ_nth_fib_le_of_nth_den (Or.inr this)
    have B_ineq : (fib (n + 1) : K) ≤ B :=
      haveI : ¬g.TerminatedAt (n - 1) := mt (terminated_stable n.pred_le) not_terminatedAt_n
      succ_nth_fib_le_of_nth_den (Or.inr this)
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      nB_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) nB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      ⊢ LT.lt (HDiv.hDiv 1 (HMul.hMul B nB)) ε
    -/
    have zero_lt_B : 0 < B := B_ineq.trans_lt' <| mod_cast fib_pos.2 n.succ_pos
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      nB_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) nB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      ⊢ LT.lt (HDiv.hDiv 1 (HMul.hMul B nB)) ε
    -/
    have nB_pos : 0 < nB := nB_ineq.trans_lt' <| mod_cast fib_pos.2 <| succ_pos _
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      nB_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) nB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      nB_pos : LT.lt 0 nB
      ⊢ LT.lt (HDiv.hDiv 1 (HMul.hMul B nB)) ε
    -/
    have zero_lt_mul_conts : 0 < B * nB := by positivity
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      nB_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) nB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      nB_pos : LT.lt 0 nB
      zero_lt_mul_conts : LT.lt 0 (HMul.hMul B nB)
      ⊢ LT.lt (HDiv.hDiv 1 (HMul.hMul B nB)) ε
    -/
    suffices 1 < ε * (B * nB) from (div_lt_iff₀ zero_lt_mul_conts).mpr this
    -- use that `N' ≥ n` was obtained from the archimedean property to show the following
    calc 1 < ε * (N' : K) := (div_lt_iff₀' ε_pos).mp one_div_ε_lt_N'
      _ ≤ ε * (B * nB) := ?_
    -- cancel `ε`
    /-
      case intro.inr
      K : Type u_1
      v : K
      inst✝² : LinearOrderedField K
      inst✝¹ : FloorRing K
      inst✝ : Archimedean K
      ε : K
      ε_pos : GT.gt ε 0
      N' : Nat
      one_div_ε_lt_N' : LT.lt (HDiv.hDiv 1 ε) ↑N'
      N : Nat := Max.max N' 5
      n : Nat
      n_ge_N : GE.ge n N
      g : GenContFract K := GenContFract.of v
      not_terminatedAt_n : Not (g.TerminatedAt n)
      B : K := g.dens n
      nB : K := g.dens (HAdd.hAdd n 1)
      abs_v_sub_conv_le : LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.h …
      nB_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) nB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      nB_pos : LT.lt 0 nB
      zero_lt_mul_conts : LT.lt 0 (HMul.hMul B nB)
      ⊢ LE.le (HMul.hMul ε ↑N') (HMul.hMul ε (HMul.hMul B nB))
    -/
    gcongr
    calc
      (N' : K) ≤ (N : K) := by exact_mod_cast le_max_left _ _
      _ ≤ n := by exact_mod_cast n_ge_N
      _ ≤ fib n := by exact_mod_cast le_fib_self <| le_trans (le_max_right N' 5) n_ge_N
      _ ≤ fib (n + 1) := by exact_mod_cast fib_le_fib_succ
      _ ≤ fib (n + 1) * fib (n + 1) := by exact_mod_cast (fib (n + 1)).le_mul_self
      _ ≤ fib (n + 1) * fib (n + 2) := by gcongr; exact_mod_cast fib_le_fib_succ
      _ ≤ B * nB := by gcongr


theorem of_convergence [TopologicalSpace K] [OrderTopology K] :
    Filter.Tendsto (of v).convs Filter.atTop <| 𝓝 v := by
  /-
    K : Type u_1
    v : K
    inst✝⁴ : LinearOrderedField K
    inst✝³ : FloorRing K
    inst✝² : Archimedean K
    inst✝¹ : TopologicalSpace K
    inst✝ : OrderTopology K
    ⊢ Filter.Tendsto (GenContFract.of v).convs Filter.atTop (nhds v)
  -/
  simpa [LinearOrderedAddCommGroup.tendsto_nhds, abs_sub_comm] using of_convergence_epsilon v
  /-
    🎉 no goals
  -/


