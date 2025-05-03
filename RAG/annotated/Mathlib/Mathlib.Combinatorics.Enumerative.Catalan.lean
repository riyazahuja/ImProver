/-- The recursive definition of the sequence of Catalan numbers:
`catalan (n + 1) = ∑ i : Fin n.succ, catalan i * catalan (n - i)` -/
def catalan : ℕ → ℕ
  | 0 => 1
  | n + 1 =>
    ∑ i : Fin n.succ,
      catalan i * catalan (n - i)


@[simp]
                                           /-
                                             ⊢ Eq (catalan 0) 1
                                           -/
theorem catalan_zero : catalan 0 = 1 := by rw [catalan]
                                           /-
                                             🎉 no goals
                                           -/


theorem catalan_succ (n : ℕ) : catalan (n + 1) = ∑ i : Fin n.succ, catalan i * catalan (n - i) := by
  /-
    n : Nat
    ⊢ Eq (catalan (HAdd.hAdd n 1)) (Finset.univ.sum fun i => HMul.hMul (catalan ↑i …
  -/
  rw [catalan]
  /-
    🎉 no goals
  -/


theorem catalan_succ' (n : ℕ) :
    catalan (n + 1) = ∑ ij ∈ antidiagonal n, catalan ij.1 * catalan ij.2 := by
  rw [catalan_succ, Nat.sum_antidiagonal_eq_sum_range_succ (fun x y => catalan x * catalan y) n,
    sum_range]


@[simp]
                                          /-
                                            ⊢ Eq (catalan 1) 1
                                          -/
theorem catalan_one : catalan 1 = 1 := by simp [catalan_succ]
                                          /-
                                            🎉 no goals
                                          -/


/-- A helper sequence that can be used to prove the equality of the recursive and the explicit
definition using a telescoping sum argument. -/
private def gosperCatalan (n j : ℕ) : ℚ :=
  Nat.centralBinom j * Nat.centralBinom (n - j) * (2 * j - n) / (2 * n * (n + 1))


private theorem gosper_trick {n i : ℕ} (h : i ≤ n) :
    gosperCatalan (n + 1) (i + 1) - gosperCatalan (n + 1) i =
      Nat.centralBinom i / (i + 1) * Nat.centralBinom (n - i) / (n - i + 1) := by
  /-
    n i : Nat
    h : LE.le i n
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd i 1)) (gosperCatalan …
  -/
  have l₁ : (i : ℚ) + 1 ≠ 0 := by norm_cast
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd i 1)) (gosperCatalan …
  -/
  have l₂ : (n : ℚ) - i + 1 ≠ 0 := by norm_cast
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd i 1)) (gosperCatalan …
  -/
  have h₁ := (mul_div_cancel_left₀ (↑(Nat.centralBinom (i + 1))) l₁).symm
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd i 1)) (gosperCatalan …
  -/
  have h₂ := (mul_div_cancel_left₀ (↑(Nat.centralBinom (n - i + 1))) l₂).symm
  have h₃ : ((i : ℚ) + 1) * (i + 1).centralBinom = 2 * (2 * i + 1) * i.centralBinom :=
    mod_cast Nat.succ_mul_centralBinom_succ i
  have h₄ :
    ((n : ℚ) - i + 1) * (n - i + 1).centralBinom = 2 * (2 * (n - i) + 1) * (n - i).centralBinom :=
      mod_cast Nat.succ_mul_centralBinom_succ (n - i)
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    h₂ : Eq (↑(HAdd.hAdd (HSub.hSub n i) 1).centralBinom) (HDiv.hDiv (HMul.hMul (H …
    h₃ : Eq (HMul.hMul (HAdd.hAdd (↑i) 1) ↑(HAdd.hAdd i 1).centralBinom) (HMul.hMu …
    h₄ : Eq (HMul.hMul (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) ↑(HAdd.hAdd (HSub.hSub n i) …
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd i 1)) (gosperCatalan …
  -/
  simp only [gosperCatalan]
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    h₂ : Eq (↑(HAdd.hAdd (HSub.hSub n i) 1).centralBinom) (HDiv.hDiv (HMul.hMul (H …
    h₃ : Eq (HMul.hMul (HAdd.hAdd (↑i) 1) ↑(HAdd.hAdd i 1).centralBinom) (HMul.hMu …
    h₄ : Eq (HMul.hMul (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) ↑(HAdd.hAdd (HSub.hSub n i) …
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HMul.hMul (HMul.hMul ↑(HAdd.hAdd i 1).centralBinom …
  -/
  push_cast
  rw [show n + 1 - i = n - i + 1 by rw [Nat.add_comm (n - i) 1, ← (Nat.add_sub_assoc h 1),
    add_comm]]
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    h₂ : Eq (↑(HAdd.hAdd (HSub.hSub n i) 1).centralBinom) (HDiv.hDiv (HMul.hMul (H …
    h₃ : Eq (HMul.hMul (HAdd.hAdd (↑i) 1) ↑(HAdd.hAdd i 1).centralBinom) (HMul.hMu …
    h₄ : Eq (HMul.hMul (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) ↑(HAdd.hAdd (HSub.hSub n i) …
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HMul.hMul (HMul.hMul ↑(HAdd.hAdd i 1).centralBinom …
  -/
  rw [h₁, h₂, h₃, h₄]
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    h₂ : Eq (↑(HAdd.hAdd (HSub.hSub n i) 1).centralBinom) (HDiv.hDiv (HMul.hMul (H …
    h₃ : Eq (HMul.hMul (HAdd.hAdd (↑i) 1) ↑(HAdd.hAdd i 1).centralBinom) (HMul.hMu …
    h₄ : Eq (HMul.hMul (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) ↑(HAdd.hAdd (HSub.hSub n i) …
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.h …
  -/
  field_simp
  /-
    n i : Nat
    h : LE.le i n
    l₁ : Ne (HAdd.hAdd (↑i) 1) 0
    l₂ : Ne (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) 0
    h₁ : Eq (↑(HAdd.hAdd i 1).centralBinom) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑i)  …
    h₂ : Eq (↑(HAdd.hAdd (HSub.hSub n i) 1).centralBinom) (HDiv.hDiv (HMul.hMul (H …
    h₃ : Eq (HMul.hMul (HAdd.hAdd (↑i) 1) ↑(HAdd.hAdd i 1).centralBinom) (HMul.hMu …
    h₄ : Eq (HMul.hMul (HAdd.hAdd (HSub.hSub ↑n ↑i) 1) ↑(HAdd.hAdd (HSub.hSub n i) …
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.h …
  -/
  ring
  /-
    🎉 no goals
  -/


private theorem gosper_catalan_sub_eq_central_binom_div (n : ℕ) : gosperCatalan (n + 1) (n + 1) -
    gosperCatalan (n + 1) 0 = Nat.centralBinom (n + 1) / (n + 2) := by
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (gosperCatalan …
  -/
  have : (n : ℚ) + 1 ≠ 0 := by norm_cast
  /-
    n : Nat
    this : Ne (HAdd.hAdd (↑n) 1) 0
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (gosperCatalan …
  -/
  have : (n : ℚ) + 1 + 1 ≠ 0 := by norm_cast
  /-
    n : Nat
    this✝ : Ne (HAdd.hAdd (↑n) 1) 0
    this : Ne (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) 0
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (gosperCatalan …
  -/
  have h : (n : ℚ) + 2 ≠ 0 := by norm_cast
  /-
    n : Nat
    this✝ : Ne (HAdd.hAdd (↑n) 1) 0
    this : Ne (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) 0
    h : Ne (HAdd.hAdd (↑n) 2) 0
    ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (gosperCatalan …
  -/
  simp only [gosperCatalan, Nat.sub_zero, Nat.centralBinom_zero, Nat.sub_self]
  /-
    n : Nat
    this✝ : Ne (HAdd.hAdd (↑n) 1) 0
    this : Ne (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) 0
    h : Ne (HAdd.hAdd (↑n) 2) 0
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HMul.hMul (HMul.hMul ↑(HAdd.hAdd n 1).centralBinom …
  -/
  field_simp
  /-
    n : Nat
    this✝ : Ne (HAdd.hAdd (↑n) 1) 0
    this : Ne (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) 0
    h : Ne (HAdd.hAdd (↑n) 2) 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (↑(HAdd.hAdd n 1).centralBinom) (HSub.hS …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem catalan_eq_centralBinom_div (n : ℕ) : catalan n = n.centralBinom / (n + 1) := by
  suffices (catalan n : ℚ) = Nat.centralBinom n / (n + 1) by
    have h := Nat.succ_dvd_centralBinom n
    exact mod_cast this
  /-
    n : Nat
    ⊢ Eq (↑(catalan n)) (HDiv.hDiv (↑n.centralBinom) (HAdd.hAdd (↑n) 1))
  -/
  induction' n using Nat.case_strong_induction_on with d hd
    /-
      case hz
      ⊢ Eq (↑(catalan 0)) (HDiv.hDiv (↑(Nat.centralBinom 0)) (HAdd.hAdd (↑0) 1))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hi
      d : Nat
      hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
      ⊢ Eq (↑(catalan (HAdd.hAdd d 1))) (HDiv.hDiv (↑(HAdd.hAdd d 1).centralBinom) ( …
    -/
  · simp_rw [catalan_succ, Nat.cast_sum, Nat.cast_mul]
    trans (∑ i : Fin d.succ, Nat.centralBinom i / (i + 1) *
                             (Nat.centralBinom (d - i) / (d - i + 1)) : ℚ)
      /-
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        ⊢ Eq (Finset.univ.sum fun x => HMul.hMul ↑(catalan ↑x) ↑(catalan (HSub.hSub d  …
      -/
    · congr
      /-
        case e_f
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        ⊢ Eq (fun x => HMul.hMul ↑(catalan ↑x) ↑(catalan (HSub.hSub d ↑x))) fun i => H …
      -/
      ext1 x
      /-
        case e_f.h
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        x : Fin d.succ
        ⊢ Eq (HMul.hMul ↑(catalan ↑x) ↑(catalan (HSub.hSub d ↑x))) (HMul.hMul (HDiv.hD …
      -/
      have m_le_d : x.val ≤ d := by omega
      /-
        case e_f.h
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        x : Fin d.succ
        m_le_d : LE.le (↑x) d
        ⊢ Eq (HMul.hMul ↑(catalan ↑x) ↑(catalan (HSub.hSub d ↑x))) (HMul.hMul (HDiv.hD …
      -/
      have d_minus_x_le_d : (d - x.val) ≤ d := tsub_le_self
      /-
        case e_f.h
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        x : Fin d.succ
        m_le_d : LE.le (↑x) d
        d_minus_x_le_d : LE.le (HSub.hSub d ↑x) d
        ⊢ Eq (HMul.hMul ↑(catalan ↑x) ↑(catalan (HSub.hSub d ↑x))) (HMul.hMul (HDiv.hD …
      -/
      rw [hd _ m_le_d, hd _ d_minus_x_le_d]
      /-
        case e_f.h
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        x : Fin d.succ
        m_le_d : LE.le (↑x) d
        d_minus_x_le_d : LE.le (HSub.hSub d ↑x) d
        ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(↑x).centralBinom) (HAdd.hAdd (↑↑x) 1)) (HDiv.hDi …
      -/
      norm_cast
      /-
        🎉 no goals
      -/
      /-
        d : Nat
        hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
        ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HDiv.hDiv (↑(↑i).centralBinom) (HAdd …
      -/
    · trans (∑ i : Fin d.succ, (gosperCatalan (d + 1) (i + 1) - gosperCatalan (d + 1) i))
        /-
          d : Nat
          hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
          ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HDiv.hDiv (↑(↑i).centralBinom) (HAdd …
        -/
      · refine sum_congr rfl fun i _ => ?_
        /-
          d : Nat
          hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
          i : Fin d.succ
          x✝ : Membership.mem Finset.univ i
          ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(↑i).centralBinom) (HAdd.hAdd (↑↑i) 1)) (HDiv.hDi …
        -/
        rw [gosper_trick i.is_le, mul_div]
        /-
          🎉 no goals
        -/
      · rw [← sum_range fun i => gosperCatalan (d + 1) (i + 1) - gosperCatalan (d + 1) i,
            sum_range_sub, Nat.succ_eq_add_one]
        /-
          d : Nat
          hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
          ⊢ Eq (HSub.hSub (gosperCatalan (HAdd.hAdd d 1) (HAdd.hAdd d 1)) (gosperCatalan …
        -/
        rw [gosper_catalan_sub_eq_central_binom_div d]
        /-
          d : Nat
          hd : ∀ (m : Nat), LE.le m d → Eq (↑(catalan m)) (HDiv.hDiv (↑m.centralBinom) ( …
          ⊢ Eq (HDiv.hDiv (↑(HAdd.hAdd d 1).centralBinom) (HAdd.hAdd (↑d) 2)) (HDiv.hDiv …
        -/
        norm_cast
        /-
          🎉 no goals
        -/


theorem succ_mul_catalan_eq_centralBinom (n : ℕ) : (n + 1) * catalan n = n.centralBinom :=
  (Nat.eq_mul_of_div_eq_right n.succ_dvd_centralBinom (catalan_eq_centralBinom_div n).symm).symm


theorem catalan_two : catalan 2 = 2 := by
  /-
    ⊢ Eq (catalan 2) 2
  -/
  norm_num [catalan_eq_centralBinom_div, Nat.centralBinom, Nat.choose]
  /-
    🎉 no goals
  -/


theorem catalan_three : catalan 3 = 5 := by
  /-
    ⊢ Eq (catalan 3) 5
  -/
  norm_num [catalan_eq_centralBinom_div, Nat.centralBinom, Nat.choose]
  /-
    🎉 no goals
  -/


/-- Given two finsets, find all trees that can be formed with
  left child in `a` and right child in `b` -/
abbrev pairwiseNode (a b : Finset (Tree Unit)) : Finset (Tree Unit) :=
                                                                         /-
                                                                           a b : Finset (Tree Unit)
                                                                           x✝¹ x✝ : Prod (Tree Unit) (Tree Unit)
                                                                           x₁ x₂ y₁ y₂ : Tree Unit
                                                                           h : Eq ((fun x => Tree.node Unit.unit x.1 x.2) { fst := x₁, snd := x₂ }) ((fun …
                                                                           ⊢ Eq { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
                                                                         -/
  (a ×ˢ b).map ⟨fun x => x.1 △ x.2, fun ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ => fun h => by simpa using h⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- A Finset of all trees with `n` nodes. See `mem_treesOfNodesEq` -/
def treesOfNumNodesEq : ℕ → Finset (Tree Unit)
  | 0 => {nil}
  | n + 1 =>
    (antidiagonal n).attach.biUnion fun ijh =>
      -- Porting note: `unusedHavesSuffices` linter is not happy with this. Commented out.
      -- have := Nat.lt_succ_of_le (fst_le ijh.2)
      -- have := Nat.lt_succ_of_le (snd_le ijh.2)
      pairwiseNode (treesOfNumNodesEq ijh.1.1) (treesOfNumNodesEq ijh.1.2)
  -- Porting note: Add this to satisfy the linter.
  decreasing_by
    · simp_wf; have := fst_le ijh.2; omega
    · simp_wf; have := snd_le ijh.2; omega


@[simp]
                                                                   /-
                                                                     ⊢ Eq (Tree.treesOfNumNodesEq 0) (Singleton.singleton Tree.nil)
                                                                   -/
theorem treesOfNumNodesEq_zero : treesOfNumNodesEq 0 = {nil} := by rw [treesOfNumNodesEq]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem treesOfNumNodesEq_succ (n : ℕ) :
    treesOfNumNodesEq (n + 1) =
      (antidiagonal n).biUnion fun ij =>
        pairwiseNode (treesOfNumNodesEq ij.1) (treesOfNumNodesEq ij.2) := by
  /-
    n : Nat
    ⊢ Eq (Tree.treesOfNumNodesEq (HAdd.hAdd n 1)) ((Finset.HasAntidiagonal.antidia …
  -/
  rw [treesOfNumNodesEq]
  /-
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).attach.biUnion fun ijh => Tree.p …
  -/
  ext
  /-
    case h
    n : Nat
    a✝ : Tree Unit
    ⊢ Iff (Membership.mem ((Finset.HasAntidiagonal.antidiagonal n).attach.biUnion  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_treesOfNumNodesEq {x : Tree Unit} {n : ℕ} :
    x ∈ treesOfNumNodesEq n ↔ x.numNodes = n := by
  /-
    x : Tree Unit
    n : Nat
    ⊢ Iff (Membership.mem (Tree.treesOfNumNodesEq n) x) (Eq x.numNodes n)
  -/
  induction x using Tree.unitRecOn generalizing n <;> cases n <;>
    /-
      case base.zero
      ⊢ Iff (Membership.mem (Tree.treesOfNumNodesEq 0) Tree.nil) (Eq Tree.nil.numNod …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [treesOfNumNodesEq_succ, *]
    /-
      🎉 no goals
    -/


theorem mem_treesOfNumNodesEq_numNodes (x : Tree Unit) : x ∈ treesOfNumNodesEq x.numNodes :=
  mem_treesOfNumNodesEq.mpr rfl


@[simp, norm_cast]
theorem coe_treesOfNumNodesEq (n : ℕ) :
    ↑(treesOfNumNodesEq n) = { x : Tree Unit | x.numNodes = n } :=
              /-
                n : Nat
                ⊢ ∀ (x : Tree Unit), Iff (Membership.mem (↑(Tree.treesOfNumNodesEq n)) x) (Mem …
              -/
  Set.ext (by simp)
              /-
                🎉 no goals
              -/


theorem treesOfNumNodesEq_card_eq_catalan (n : ℕ) : #(treesOfNumNodesEq n) = catalan n := by
  /-
    n : Nat
    ⊢ Eq (Tree.treesOfNumNodesEq n).card (catalan n)
  -/
  induction' n using Nat.case_strong_induction_on with n ih
    /-
      case hz
      ⊢ Eq (Tree.treesOfNumNodesEq 0).card (catalan 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case hi
    n : Nat
    ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
    ⊢ Eq (Tree.treesOfNumNodesEq (HAdd.hAdd n 1)).card (catalan (HAdd.hAdd n 1))
  -/
  rw [treesOfNumNodesEq_succ, card_biUnion, catalan_succ']
    /-
      case hi
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun u => (Tree.pairwiseNode  …
    -/
  · apply sum_congr rfl
    /-
      case hi
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
    rintro ⟨i, j⟩ H
    /-
      case hi.mk
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      i j : Nat
      H : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd :=  …
      ⊢ Eq (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd := j }.1) (Tre …
    -/
    rw [card_map, card_product, ih _ (fst_le H), ih _ (snd_le H)]
    /-
      🎉 no goals
    -/
    /-
      case hi
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
  · simp_rw [disjoint_left]
    /-
      case hi
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
    rintro ⟨i, j⟩ _ ⟨i', j'⟩ _
    -- Porting note: was clear * -; tidy
    /-
      case hi.mk.mk
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      i j : Nat
      a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
      i' j' : Nat
      a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
      ⊢ Ne { fst := i, snd := j } { fst := i', snd := j' } → ∀ ⦃a : Tree Unit⦄, Memb …
    -/
    intros h a
    /-
      case hi.mk.mk
      n : Nat
      ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
      i j : Nat
      a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
      i' j' : Nat
      a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
      h : Ne { fst := i, snd := j } { fst := i', snd := j' }
      a : Tree Unit
      ⊢ Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd := …
    -/
    cases' a with a l r
      /-
        case hi.mk.mk.nil
        n : Nat
        ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
        i j : Nat
        a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
        i' j' : Nat
        a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
        h : Ne { fst := i, snd := j } { fst := i', snd := j' }
        ⊢ Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd := …
      -/
    · intro h; simp at h
               /-
                 🎉 no goals
               -/
      /-
        case hi.mk.mk.node
        n : Nat
        ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
        i j : Nat
        a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
        i' j' : Nat
        a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
        h : Ne { fst := i, snd := j } { fst := i', snd := j' }
        a : Unit
        l r : Tree Unit
        ⊢ Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd := …
      -/
    · intro h1 h2
      /-
        case hi.mk.mk.node
        n : Nat
        ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
        i j : Nat
        a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
        i' j' : Nat
        a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
        h : Ne { fst := i, snd := j } { fst := i', snd := j' }
        a : Unit
        l r : Tree Unit
        h1 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd …
        h2 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i', sn …
        ⊢ False
      -/
      apply h
      /-
        case hi.mk.mk.node
        n : Nat
        ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
        i j : Nat
        a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
        i' j' : Nat
        a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
        h : Ne { fst := i, snd := j } { fst := i', snd := j' }
        a : Unit
        l r : Tree Unit
        h1 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd …
        h2 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i', sn …
        ⊢ Eq { fst := i, snd := j } { fst := i', snd := j' }
      -/
      trans (numNodes l, numNodes r)
        /-
          n : Nat
          ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
          i j : Nat
          a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
          i' j' : Nat
          a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
          h : Ne { fst := i, snd := j } { fst := i', snd := j' }
          a : Unit
          l r : Tree Unit
          h1 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd …
          h2 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i', sn …
          ⊢ Eq { fst := i, snd := j } { fst := l.numNodes, snd := r.numNodes }
        -/
      · simp at h1; simp [h1]
                    /-
                      🎉 no goals
                    -/
        /-
          n : Nat
          ih : ∀ (m : Nat), LE.le m n → Eq (Tree.treesOfNumNodesEq m).card (catalan m)
          i j : Nat
          a✝¹ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
          i' j' : Nat
          a✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i', snd : …
          h : Ne { fst := i, snd := j } { fst := i', snd := j' }
          a : Unit
          l r : Tree Unit
          h1 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i, snd …
          h2 : Membership.mem (Tree.pairwiseNode (Tree.treesOfNumNodesEq { fst := i', sn …
          ⊢ Eq { fst := l.numNodes, snd := r.numNodes } { fst := i', snd := j' }
        -/
      · simp at h2; simp [h2]
                    /-
                      🎉 no goals
                    -/


