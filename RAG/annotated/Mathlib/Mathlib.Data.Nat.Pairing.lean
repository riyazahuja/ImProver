/-- Pairing function for the natural numbers. -/
@[pp_nodot]
def pair (a b : ℕ) : ℕ :=
  if a < b then b * b + a else a * a + a + b


/-- Unpairing function for the natural numbers. -/
@[pp_nodot]
def unpair (n : ℕ) : ℕ × ℕ :=
  let s := sqrt n
  if n - s * s < s then (n - s * s, s) else (s, n - s * s - s)


@[simp]
theorem pair_unpair (n : ℕ) : pair (unpair n).1 (unpair n).2 = n := by
  /-
    n : Nat
    ⊢ Eq (Nat.pair (Nat.unpair n).1 (Nat.unpair n).2) n
  -/
  dsimp only [unpair]; let s := sqrt n
  /-
    n : Nat
    s : Nat := n.sqrt
    ⊢ Eq (Nat.pair (ite (LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n.sqrt) { f …
  -/
  have sm : s * s + (n - s * s) = n := Nat.add_sub_cancel' (sqrt_le _)
  /-
    n : Nat
    s : Nat := n.sqrt
    sm : Eq (HAdd.hAdd (HMul.hMul s s) (HSub.hSub n (HMul.hMul s s))) n
    ⊢ Eq (Nat.pair (ite (LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n.sqrt) { f …
  -/
  split_ifs with h
    /-
      case pos
      n : Nat
      s : Nat := n.sqrt
      sm : Eq (HAdd.hAdd (HMul.hMul s s) (HSub.hSub n (HMul.hMul s s))) n
      h : LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n.sqrt
      ⊢ Eq (Nat.pair { fst := HSub.hSub n (HMul.hMul n.sqrt n.sqrt), snd := n.sqrt } …
    -/
  · simp [s, pair, h, sm]
    /-
      🎉 no goals
    -/
  · have hl : n - s * s - s ≤ s := Nat.sub_le_iff_le_add.2
      (Nat.sub_le_iff_le_add'.2 <| by rw [← Nat.add_assoc]; apply sqrt_le_add)
    /-
      case neg
      n : Nat
      s : Nat := n.sqrt
      sm : Eq (HAdd.hAdd (HMul.hMul s s) (HSub.hSub n (HMul.hMul s s))) n
      h : Not (LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n.sqrt)
      hl : LE.le (HSub.hSub (HSub.hSub n (HMul.hMul s s)) s) s
      ⊢ Eq (Nat.pair { fst := n.sqrt, snd := HSub.hSub (HSub.hSub n (HMul.hMul n.sqr …
    -/
    simp [s, pair, hl.not_lt, Nat.add_assoc, Nat.add_sub_cancel' (le_of_not_gt h), sm]
    /-
      🎉 no goals
    -/


theorem pair_unpair' {n a b} (H : unpair n = (a, b)) : pair a b = n := by
  /-
    n a b : Nat
    H : Eq (Nat.unpair n) { fst := a, snd := b }
    ⊢ Eq (Nat.pair a b) n
  -/
  simpa [H] using pair_unpair n
  /-
    🎉 no goals
  -/


@[simp]
theorem unpair_pair (a b : ℕ) : unpair (pair a b) = (a, b) := by
  /-
    a b : Nat
    ⊢ Eq (Nat.unpair (Nat.pair a b)) { fst := a, snd := b }
  -/
  dsimp only [pair]; split_ifs with h
    /-
      case pos
      a b : Nat
      h : LT.lt a b
      ⊢ Eq (Nat.unpair (HAdd.hAdd (HMul.hMul b b) a)) { fst := a, snd := b }
    -/
  · show unpair (b * b + a) = (a, b)
    /-
      case pos
      a b : Nat
      h : LT.lt a b
      ⊢ Eq (Nat.unpair (HAdd.hAdd (HMul.hMul b b) a)) { fst := a, snd := b }
    -/
    have be : sqrt (b * b + a) = b := sqrt_add_eq _ (le_trans (le_of_lt h) (Nat.le_add_left _ _))
    /-
      case pos
      a b : Nat
      h : LT.lt a b
      be : Eq (HAdd.hAdd (HMul.hMul b b) a).sqrt b
      ⊢ Eq (Nat.unpair (HAdd.hAdd (HMul.hMul b b) a)) { fst := a, snd := b }
    -/
    simp [unpair, be, Nat.add_sub_cancel_left, h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : Nat
      h : Not (LT.lt a b)
      ⊢ Eq (Nat.unpair (HAdd.hAdd (HAdd.hAdd (HMul.hMul a a) a) b)) { fst := a, snd  …
    -/
  · show unpair (a * a + a + b) = (a, b)
    have ae : sqrt (a * a + (a + b)) = a := by
      rw [sqrt_add_eq]
      exact Nat.add_le_add_left (le_of_not_gt h) _
    /-
      case neg
      a b : Nat
      h : Not (LT.lt a b)
      ae : Eq (HAdd.hAdd (HMul.hMul a a) (HAdd.hAdd a b)).sqrt a
      ⊢ Eq (Nat.unpair (HAdd.hAdd (HAdd.hAdd (HMul.hMul a a) a) b)) { fst := a, snd  …
    -/
    simp [unpair, ae, Nat.not_lt_zero, Nat.add_assoc, Nat.add_sub_cancel_left]
    /-
      🎉 no goals
    -/


/-- An equivalence between `ℕ × ℕ` and `ℕ`. -/
@[simps (config := .asFn)]
def pairEquiv : ℕ × ℕ ≃ ℕ :=
  ⟨uncurry pair, unpair, fun ⟨a, b⟩ => unpair_pair a b, pair_unpair⟩


theorem surjective_unpair : Surjective unpair :=
  pairEquiv.symm.surjective


@[simp]
theorem pair_eq_pair {a b c d : ℕ} : pair a b = pair c d ↔ a = c ∧ b = d :=
  pairEquiv.injective.eq_iff.trans (@Prod.ext_iff ℕ ℕ (a, b) (c, d))


theorem unpair_lt {n : ℕ} (n1 : 1 ≤ n) : (unpair n).1 < n := by
  /-
    n : Nat
    n1 : LE.le 1 n
    ⊢ LT.lt (Nat.unpair n).1 n
  -/
  let s := sqrt n
  /-
    n : Nat
    n1 : LE.le 1 n
    s : Nat := n.sqrt
    ⊢ LT.lt (Nat.unpair n).1 n
  -/
  simp only [unpair, Nat.sub_le_iff_le_add]
  /-
    n : Nat
    n1 : LE.le 1 n
    s : Nat := n.sqrt
    ⊢ LT.lt (ite (LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n.sqrt) { fst := H …
  -/
  by_cases h : n - s * s < s <;> simp [s, h, ↓reduceIte]
    /-
      case pos
      n : Nat
      n1 : LE.le 1 n
      s : Nat := n.sqrt
      h : LT.lt (HSub.hSub n (HMul.hMul s s)) s
      ⊢ LT.lt (HSub.hSub n (HMul.hMul n.sqrt n.sqrt)) n
    -/
  · exact lt_of_lt_of_le h (sqrt_le_self _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      n1 : LE.le 1 n
      s : Nat := n.sqrt
      h : Not (LT.lt (HSub.hSub n (HMul.hMul s s)) s)
      ⊢ LT.lt n.sqrt n
    -/
  · simp only [not_lt] at h
    /-
      case neg
      n : Nat
      n1 : LE.le 1 n
      s : Nat := n.sqrt
      h : LE.le s (HSub.hSub n (HMul.hMul s s))
      ⊢ LT.lt n.sqrt n
    -/
    have s0 : 0 < s := sqrt_pos.2 n1
    /-
      case neg
      n : Nat
      n1 : LE.le 1 n
      s : Nat := n.sqrt
      h : LE.le s (HSub.hSub n (HMul.hMul s s))
      s0 : LT.lt 0 s
      ⊢ LT.lt n.sqrt n
    -/
    exact lt_of_le_of_lt h (Nat.sub_lt n1 (Nat.mul_pos s0 s0))
    /-
      🎉 no goals
    -/


@[simp]
theorem unpair_zero : unpair 0 = 0 := by
  /-
    ⊢ Eq (Nat.unpair 0) 0
  -/
  rw [unpair]
  /-
    ⊢ Eq (ite (LT.lt (HSub.hSub 0 (HMul.hMul (Nat.sqrt 0) (Nat.sqrt 0))) (Nat.sqrt …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem unpair_left_le : ∀ n : ℕ, (unpair n).1 ≤ n
            /-
              ⊢ LE.le (Nat.unpair 0).1 0
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | _ + 1 => le_of_lt (unpair_lt (Nat.succ_pos _))


                                                    /-
                                                      a b : Nat
                                                      ⊢ LE.le a (Nat.pair a b)
                                                    -/
theorem left_le_pair (a b : ℕ) : a ≤ pair a b := by simpa using unpair_left_le (pair a b)
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem right_le_pair (a b : ℕ) : b ≤ pair a b := by
  /-
    a b : Nat
    ⊢ LE.le b (Nat.pair a b)
  -/
  by_cases h : a < b
    /-
      case pos
      a b : Nat
      h : LT.lt a b
      ⊢ LE.le b (Nat.pair a b)
    -/
  · simpa [pair, h] using le_trans (le_mul_self _) (Nat.le_add_right _ _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : Nat
      h : Not (LT.lt a b)
      ⊢ LE.le b (Nat.pair a b)
    -/
  · simp [pair, h]
    /-
      🎉 no goals
    -/


theorem unpair_right_le (n : ℕ) : (unpair n).2 ≤ n := by
  /-
    n : Nat
    ⊢ LE.le (Nat.unpair n).2 n
  -/
  simpa using right_le_pair n.unpair.1 n.unpair.2
  /-
    🎉 no goals
  -/


theorem pair_lt_pair_left {a₁ a₂} (b) (h : a₁ < a₂) : pair a₁ b < pair a₂ b := by
  /-
    a₁ a₂ b : Nat
    h : LT.lt a₁ a₂
    ⊢ LT.lt (Nat.pair a₁ b) (Nat.pair a₂ b)
  -/
  by_cases h₁ : a₁ < b <;> simp [pair, h₁, Nat.add_assoc]
    /-
      case pos
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : LT.lt a₁ b
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul b b) a₁) (ite (LT.lt a₂ b) (HAdd.hAdd (HMul.hMul …
    -/
                             /-
                               🎉 no goals
                             -/
  · by_cases h₂ : a₂ < b <;> simp [pair, h₂, h]
    /-
      case neg
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : LT.lt a₁ b
      h₂ : Not (LT.lt a₂ b)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul b b) a₁) (HAdd.hAdd (HMul.hMul a₂ a₂) (HAdd.hAdd …
    -/
    simp? at h₂ says simp only [not_lt] at h₂
    /-
      case neg
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : LT.lt a₁ b
      h₂ : LE.le b a₂
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul b b) a₁) (HAdd.hAdd (HMul.hMul a₂ a₂) (HAdd.hAdd …
    -/
    apply Nat.add_lt_add_of_le_of_lt
      /-
        case neg.hle
        a₁ a₂ b : Nat
        h : LT.lt a₁ a₂
        h₁ : LT.lt a₁ b
        h₂ : LE.le b a₂
        ⊢ LE.le (HMul.hMul b b) (HMul.hMul a₂ a₂)
      -/
    · exact Nat.mul_self_le_mul_self h₂
      /-
        🎉 no goals
      -/
      /-
        case neg.hlt
        a₁ a₂ b : Nat
        h : LT.lt a₁ a₂
        h₁ : LT.lt a₁ b
        h₂ : LE.le b a₂
        ⊢ LT.lt a₁ (HAdd.hAdd a₂ b)
      -/
    · exact Nat.lt_add_right _ h
      /-
        🎉 no goals
      -/
    /-
      case neg
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : Not (LT.lt a₁ b)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a₁ a₁) (HAdd.hAdd a₁ b)) (ite (LT.lt a₂ b) (HAdd …
    -/
  · simp at h₁
    /-
      case neg
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : LE.le b a₁
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a₁ a₁) (HAdd.hAdd a₁ b)) (ite (LT.lt a₂ b) (HAdd …
    -/
    simp only [not_lt_of_gt (lt_of_le_of_lt h₁ h), ite_false]
    /-
      case neg
      a₁ a₂ b : Nat
      h : LT.lt a₁ a₂
      h₁ : LE.le b a₁
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a₁ a₁) (HAdd.hAdd a₁ b)) (HAdd.hAdd (HMul.hMul a …
    -/
    apply add_lt_add
      /-
        case neg.h₁
        a₁ a₂ b : Nat
        h : LT.lt a₁ a₂
        h₁ : LE.le b a₁
        ⊢ LT.lt (HMul.hMul a₁ a₁) (HMul.hMul a₂ a₂)
      -/
    · exact Nat.mul_self_lt_mul_self h
      /-
        🎉 no goals
      -/
      /-
        case neg.h₂
        a₁ a₂ b : Nat
        h : LT.lt a₁ a₂
        h₁ : LE.le b a₁
        ⊢ LT.lt (HAdd.hAdd a₁ b) (HAdd.hAdd a₂ b)
      -/
    · apply Nat.add_lt_add_right; assumption
                                  /-
                                    🎉 no goals
                                  -/


theorem pair_lt_pair_right (a) {b₁ b₂} (h : b₁ < b₂) : pair a b₁ < pair a b₂ := by
  /-
    a b₁ b₂ : Nat
    h : LT.lt b₁ b₂
    ⊢ LT.lt (Nat.pair a b₁) (Nat.pair a b₂)
  -/
  by_cases h₁ : a < b₁
    /-
      case pos
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₁ : LT.lt a b₁
      ⊢ LT.lt (Nat.pair a b₁) (Nat.pair a b₂)
    -/
  · simpa [pair, h₁, Nat.add_assoc, lt_trans h₁ h, h] using mul_self_lt_mul_self h
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₁ : Not (LT.lt a b₁)
      ⊢ LT.lt (Nat.pair a b₁) (Nat.pair a b₂)
    -/
  · simp only [pair, h₁, ↓reduceIte, Nat.add_assoc]
    /-
      case neg
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₁ : Not (LT.lt a b₁)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a a) (HAdd.hAdd a b₁)) (ite (LT.lt a b₂) (HAdd.h …
    -/
    by_cases h₂ : a < b₂ <;> simp [pair, h₂, h]
                             /-
                               🎉 no goals
                             -/
    /-
      case pos
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₁ : Not (LT.lt a b₁)
      h₂ : LT.lt a b₂
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a a) (HAdd.hAdd a b₁)) (HAdd.hAdd (HMul.hMul b₂  …
    -/
    simp? at h₁ says simp only [not_lt] at h₁
    /-
      case pos
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₂ : LT.lt a b₂
      h₁ : LE.le b₁ a
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul a a) (HAdd.hAdd a b₁)) (HAdd.hAdd (HMul.hMul b₂  …
    -/
    rw [Nat.add_comm, Nat.add_comm _ a, Nat.add_assoc, Nat.add_lt_add_iff_left]
    /-
      case pos
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₂ : LT.lt a b₂
      h₁ : LE.le b₁ a
      ⊢ LT.lt (HAdd.hAdd b₁ (HMul.hMul a a)) (HMul.hMul b₂ b₂)
    -/
    rwa [Nat.add_comm, ← sqrt_lt, sqrt_add_eq]
    /-
      case pos.h
      a b₁ b₂ : Nat
      h : LT.lt b₁ b₂
      h₂ : LT.lt a b₂
      h₁ : LE.le b₁ a
      ⊢ LE.le b₁ (HAdd.hAdd a a)
    -/
    exact le_trans h₁ (Nat.le_add_left _ _)
    /-
      🎉 no goals
    -/


theorem pair_lt_max_add_one_sq (m n : ℕ) : pair m n < (max m n + 1) ^ 2 := by
  /-
    m n : Nat
    ⊢ LT.lt (Nat.pair m n) (HPow.hPow (HAdd.hAdd (Max.max m n) 1) 2)
  -/
  simp only [pair, Nat.pow_two, Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul, Nat.add_assoc]
  /-
    m n : Nat
    ⊢ LT.lt (ite (LT.lt m n) (HAdd.hAdd (HMul.hMul n n) m) (HAdd.hAdd (HMul.hMul m …
  -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
  split_ifs <;> simp [Nat.max_eq_left, Nat.max_eq_right, Nat.le_of_lt,  not_lt.1, *] <;> omega
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem max_sq_add_min_le_pair (m n : ℕ) : max m n ^ 2 + min m n ≤ pair m n := by
  /-
    m n : Nat
    ⊢ LE.le (HAdd.hAdd (HPow.hPow (Max.max m n) 2) (Min.min m n)) (Nat.pair m n)
  -/
  rw [pair]
  /-
    m n : Nat
    ⊢ LE.le (HAdd.hAdd (HPow.hPow (Max.max m n) 2) (Min.min m n)) (ite (LT.lt m n) …
  -/
  cases' lt_or_le m n with h h
    /-
      case inl
      m n : Nat
      h : LT.lt m n
      ⊢ LE.le (HAdd.hAdd (HPow.hPow (Max.max m n) 2) (Min.min m n)) (ite (LT.lt m n) …
    -/
  · rw [if_pos h, max_eq_right h.le, min_eq_left h.le, Nat.pow_two]
    /-
      🎉 no goals
    -/
  rw [if_neg h.not_lt, max_eq_left h, min_eq_right h, Nat.pow_two, Nat.add_assoc,
    Nat.add_le_add_iff_left]
  /-
    case inr
    m n : Nat
    h : LE.le n m
    ⊢ LE.le n (HAdd.hAdd m n)
  -/
  exact Nat.le_add_left _ _
  /-
    🎉 no goals
  -/


theorem add_le_pair (m n : ℕ) : m + n ≤ pair m n := by
  /-
    m n : Nat
    ⊢ LE.le (HAdd.hAdd m n) (Nat.pair m n)
  -/
  simp only [pair, Nat.add_assoc]
  /-
    m n : Nat
    ⊢ LE.le (HAdd.hAdd m n) (ite (LT.lt m n) (HAdd.hAdd (HMul.hMul n n) m) (HAdd.h …
  -/
  split_ifs
    /-
      case pos
      m n : Nat
      h✝ : LT.lt m n
      ⊢ LE.le (HAdd.hAdd m n) (HAdd.hAdd (HMul.hMul n n) m)
    -/
  · have := le_mul_self n
    /-
      case pos
      m n : Nat
      h✝ : LT.lt m n
      this : LE.le n (HMul.hMul n n)
      ⊢ LE.le (HAdd.hAdd m n) (HAdd.hAdd (HMul.hMul n n) m)
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      h✝ : Not (LT.lt m n)
      ⊢ LE.le (HAdd.hAdd m n) (HAdd.hAdd (HMul.hMul m m) (HAdd.hAdd m n))
    -/
  · exact Nat.le_add_left _ _
    /-
      🎉 no goals
    -/


theorem unpair_add_le (n : ℕ) : (unpair n).1 + (unpair n).2 ≤ n :=
  (add_le_pair _ _).trans_eq (pair_unpair _)


theorem iSup_unpair {α} [CompleteLattice α] (f : ℕ → ℕ → α) :
    ⨆ n : ℕ, f n.unpair.1 n.unpair.2 = ⨆ (i : ℕ) (j : ℕ), f i j := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → Nat → α
    ⊢ Eq (iSup fun n => f (Nat.unpair n).1 (Nat.unpair n).2) (iSup fun i => iSup f …
  -/
  rw [← (iSup_prod : ⨆ i : ℕ × ℕ, f i.1 i.2 = _), ← Nat.surjective_unpair.iSup_comp]
  /-
    🎉 no goals
  -/

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

theorem iInf_unpair {α} [CompleteLattice α] (f : ℕ → ℕ → α) :
    ⨅ n : ℕ, f n.unpair.1 n.unpair.2 = ⨅ (i : ℕ) (j : ℕ), f i j :=
  iSup_unpair (show ℕ → ℕ → αᵒᵈ from f)


theorem iUnion_unpair_prod {α β} {s : ℕ → Set α} {t : ℕ → Set β} :
    ⋃ n : ℕ, s n.unpair.fst ×ˢ t n.unpair.snd = (⋃ n, s n) ×ˢ ⋃ n, t n := by
  /-
    α : Type u_1
    β : Type u_2
    s : Nat → Set α
    t : Nat → Set β
    ⊢ Eq (Set.iUnion fun n => SProd.sprod (s (Nat.unpair n).1) (t (Nat.unpair n).2 …
  -/
  rw [← Set.iUnion_prod]
  /-
    α : Type u_1
    β : Type u_2
    s : Nat → Set α
    t : Nat → Set β
    ⊢ Eq (Set.iUnion fun n => SProd.sprod (s (Nat.unpair n).1) (t (Nat.unpair n).2 …
  -/
  exact surjective_unpair.iUnion_comp (fun x => s x.fst ×ˢ t x.snd)
  /-
    🎉 no goals
  -/

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

theorem iUnion_unpair {α} (f : ℕ → ℕ → Set α) :
    ⋃ n : ℕ, f n.unpair.1 n.unpair.2 = ⋃ (i : ℕ) (j : ℕ), f i j :=
  iSup_unpair f

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

theorem iInter_unpair {α} (f : ℕ → ℕ → Set α) :
    ⋂ n : ℕ, f n.unpair.1 n.unpair.2 = ⋂ (i : ℕ) (j : ℕ), f i j :=
  iInf_unpair f


