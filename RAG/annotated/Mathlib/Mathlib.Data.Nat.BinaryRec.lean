/-- `bit b` appends the digit `b` to the binary representation of its natural number input. -/
def bit (b : Bool) : Nat → Nat := cond b (2 * · + 1) (2 * ·)


theorem shiftRight_one (n) : n >>> 1 = n / 2 := rfl


@[simp]
theorem bit_decide_mod_two_eq_one_shiftRight_one (n : Nat) : bit (n % 2 = 1) (n >>> 1) = n := by
  /-
    n : Nat
    ⊢ Eq (Nat.bit (Decidable.decide (Eq (HMod.hMod n 2) 1)) (HShiftRight.hShiftRig …
  -/
  simp only [bit, shiftRight_one]
  /-
    n : Nat
    ⊢ Eq (cond (Decidable.decide (Eq (HMod.hMod n 2) 1)) (fun x => HAdd.hAdd (HMul …
  -/
  /-
    🎉 no goals
  -/
  cases mod_two_eq_zero_or_one n with | _ h => simpa [h] using Nat.div_add_mod n 2


theorem bit_testBit_zero_shiftRight_one (n : Nat) : bit (n.testBit 0) (n >>> 1) = n := by
  /-
    n : Nat
    ⊢ Eq (Nat.bit (n.testBit 0) (HShiftRight.hShiftRight n 1)) n
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem bit_eq_zero_iff {n : Nat} {b : Bool} : bit b n = 0 ↔ n = 0 ∧ b = false := by
  /-
    n : Nat
    b : Bool
    ⊢ Iff (Eq (Nat.bit b n) 0) (And (Eq n 0) (Eq b Bool.false))
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
  cases n <;> cases b <;> simp [bit, Nat.shiftLeft_succ, Nat.two_mul, ← Nat.add_assoc]
                          /-
                            🎉 no goals
                          -/


/-- For a predicate `motive : Nat → Sort u`, if instances can be
  constructed for natural numbers of the form `bit b n`,
  they can be constructed for any given natural number. -/
@[inline]
def bitCasesOn {motive : Nat → Sort u} (n) (h : ∀ b n, motive (bit b n)) : motive n :=
  -- `1 &&& n != 0` is faster than `n.testBit 0`. This may change when we have faster `testBit`.
  let x := h (1 &&& n != 0) (n >>> 1)
  -- `congrArg motive _ ▸ x` is defeq to `x` in non-dependent case
  congrArg motive n.bit_testBit_zero_shiftRight_one ▸ x


/-- A recursion principle for `bit` representations of natural numbers.
  For a predicate `motive : Nat → Sort u`, if instances can be
  constructed for natural numbers of the form `bit b n`,
  they can be constructed for all natural numbers. -/
@[elab_as_elim, specialize]
def binaryRec {motive : Nat → Sort u} (z : motive 0) (f : ∀ b n, motive n → motive (bit b n))
    (n : Nat) : motive n :=
  if n0 : n = 0 then congrArg motive n0 ▸ z
  else
    let x := f (1 &&& n != 0) (n >>> 1) (binaryRec z f (n >>> 1))
    congrArg motive n.bit_testBit_zero_shiftRight_one ▸ x
/-
  n : Nat
  n0 : Not (Eq n 0)
  ⊢ LT.lt (HShiftRight.hShiftRight n 1) n
-/
decreasing_by exact bitwise_rec_lemma n0
/-
  🎉 no goals
-/


/-- The same as `binaryRec`, but the induction step can assume that if `n=0`,
  the bit being appended is `true`-/
@[elab_as_elim, specialize]
def binaryRec' {motive : Nat → Sort u} (z : motive 0)
    (f : ∀ b n, (n = 0 → b = true) → motive n → motive (bit b n)) :
    ∀ n, motive n :=
  binaryRec z fun b n ih =>
    if h : n = 0 → b = true then f b n h ih
    else
      have : bit b n = 0 := by
        /-
          motive : Nat → Sort u
          z : motive 0
          f : (b : Bool) → (n : Nat) → (Eq n 0 → Eq b Bool.true) → motive n → motive (Na …
          b : Bool
          n : Nat
          ih : motive n
          h : Not (Eq n 0 → Eq b Bool.true)
          ⊢ Eq (Nat.bit b n) 0
        -/
        rw [bit_eq_zero_iff]
        /-
          motive : Nat → Sort u
          z : motive 0
          f : (b : Bool) → (n : Nat) → (Eq n 0 → Eq b Bool.true) → motive n → motive (Na …
          b : Bool
          n : Nat
          ih : motive n
          h : Not (Eq n 0 → Eq b Bool.true)
          ⊢ And (Eq n 0) (Eq b Bool.false)
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
        cases n <;> cases b <;> simp at h ⊢
                                /-
                                  🎉 no goals
                                -/
      congrArg motive this ▸ z


/-- The same as `binaryRec`, but special casing both 0 and 1 as base cases -/
@[elab_as_elim, specialize]
def binaryRecFromOne {motive : Nat → Sort u} (z₀ : motive 0) (z₁ : motive 1)
    (f : ∀ b n, n ≠ 0 → motive n → motive (bit b n)) :
    ∀ n, motive n :=
  binaryRec' z₀ fun b n h ih =>
    if h' : n = 0 then
      have : bit b n = bit true 0 := by
        /-
          motive : Nat → Sort u
          z₀ : motive 0
          z₁ : motive 1
          f : (b : Bool) → (n : Nat) → Ne n 0 → motive n → motive (Nat.bit b n)
          b : Bool
          n : Nat
          h : Eq n 0 → Eq b Bool.true
          ih : motive n
          h' : Eq n 0
          ⊢ Eq (Nat.bit b n) (Nat.bit Bool.true 0)
        -/
        rw [h', h h']
        /-
          🎉 no goals
        -/
      congrArg motive this ▸ z₁
    else f b n h' ih


theorem bit_val (b n) : bit b n = 2 * n + b.toNat := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (Nat.bit b n) (HAdd.hAdd (HMul.hMul 2 n) b.toNat)
  -/
              /-
                🎉 no goals
              -/
  cases b <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem bit_div_two (b n) : bit b n / 2 = n := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (HDiv.hDiv (Nat.bit b n) 2) n
  -/
  rw [bit_val, Nat.add_comm, add_mul_div_left, div_eq_of_lt, Nat.zero_add]
    /-
      b : Bool
      n : Nat
      ⊢ LT.lt b.toNat 2
    -/
                /-
                  🎉 no goals
                -/
  · cases b <;> decide
                /-
                  🎉 no goals
                -/
    /-
      case H
      b : Bool
      n : Nat
      ⊢ LT.lt 0 2
    -/
  · decide
    /-
      🎉 no goals
    -/


@[simp]
theorem bit_mod_two (b n) : bit b n % 2 = b.toNat := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (HMod.hMod (Nat.bit b n) 2) b.toNat
  -/
              /-
                🎉 no goals
              -/
  cases b <;> simp [bit_val, mul_add_mod]
              /-
                🎉 no goals
              -/


@[simp]
theorem bit_shiftRight_one (b n) : bit b n >>> 1 = n :=
  bit_div_two b n


theorem testBit_bit_zero (b n) : (bit b n).testBit 0 = b := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq ((Nat.bit b n).testBit 0) b
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem bitCasesOn_bit (h : ∀ b n, motive (bit b n)) (b : Bool) (n : Nat) :
    bitCasesOn (bit b n) h = h b n := by
  /-
    motive : Nat → Sort u
    h : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    b : Bool
    n : Nat
    ⊢ Eq (Nat.bitCasesOn (Nat.bit b n) h) (h b n)
  -/
  change congrArg motive (bit b n).bit_testBit_zero_shiftRight_one ▸ h _ _ = h b n
  /-
    motive : Nat → Sort u
    h : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    b : Bool
    n : Nat
    ⊢ Eq (Eq.rec (h ((Nat.bit b n).testBit 0) (HShiftRight.hShiftRight (Nat.bit b  …
  -/
  generalize congrArg motive (bit b n).bit_testBit_zero_shiftRight_one = e; revert e
  /-
    motive : Nat → Sort u
    h : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    b : Bool
    n : Nat
    ⊢ ∀ (e : Eq (motive (Nat.bit ((Nat.bit b n).testBit 0) (HShiftRight.hShiftRigh …
  -/
  rw [testBit_bit_zero, bit_shiftRight_one]
  /-
    motive : Nat → Sort u
    h : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    b : Bool
    n : Nat
    ⊢ ∀ (e : Eq (motive (Nat.bit b n)) (motive (Nat.bit b n))), Eq (Eq.rec (h b n) …
  -/
  intros; rfl
          /-
            🎉 no goals
          -/


unseal binaryRec in
@[simp]
theorem binaryRec_zero (z : motive 0) (f : ∀ b n, motive n → motive (bit b n)) :
    binaryRec z f 0 = z :=
  rfl


@[simp]
theorem binaryRec_one (z : motive 0) (f : ∀ b n, motive n → motive (bit b n)) :
    binaryRec (motive := motive) z f 1 = f true 0 z := by
  /-
    motive : Nat → Sort u
    z : motive 0
    f : (b : Bool) → (n : Nat) → motive n → motive (Nat.bit b n)
    ⊢ Eq (Nat.binaryRec z f 1) (f Bool.true 0 z)
  -/
  rw [binaryRec]
  /-
    motive : Nat → Sort u
    z : motive 0
    f : (b : Bool) → (n : Nat) → motive n → motive (Nat.bit b n)
    ⊢ Eq
        (dite (Eq 1 0) (fun n0 => Eq.rec z ⋯) fun n0 =>
          let x := f (bne (HAnd.hAnd 1 1) 0) (HShiftRight.hShiftRight 1 1) (Nat.bi …
          Eq.rec x ⋯)
        (f Bool.true 0 z)
  -/
  simp only [add_one_ne_zero, ↓reduceDIte, Nat.reduceShiftRight, binaryRec_zero]
  /-
    motive : Nat → Sort u
    z : motive 0
    f : (b : Bool) → (n : Nat) → motive n → motive (Nat.bit b n)
    ⊢ Eq (f (bne (HAnd.hAnd 1 1) 0) 0 z) (f Bool.true 0 z)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem binaryRec_eq {z : motive 0} {f : ∀ b n, motive n → motive (bit b n)}
    (b n) (h : f false 0 z = z ∨ (n = 0 → b = true)) :
    binaryRec z f (bit b n) = f b n (binaryRec z f n) := by
  /-
    motive : Nat → Sort u
    z : motive 0
    f : (b : Bool) → (n : Nat) → motive n → motive (Nat.bit b n)
    b : Bool
    n : Nat
    h : Or (Eq (f Bool.false 0 z) z) (Eq n 0 → Eq b Bool.true)
    ⊢ Eq (Nat.binaryRec z f (Nat.bit b n)) (f b n (Nat.binaryRec z f n))
  -/
  by_cases h' : bit b n = 0
  case pos =>
    obtain ⟨rfl, rfl⟩ := bit_eq_zero_iff.mp h'
    simp only [Bool.false_eq_true, imp_false, not_true_eq_false, or_false] at h
    unfold binaryRec
    exact h.symm
  case neg =>
    rw [binaryRec, dif_neg h']
    change congrArg motive (bit b n).bit_testBit_zero_shiftRight_one ▸ f _ _ _ = _
    generalize congrArg motive (bit b n).bit_testBit_zero_shiftRight_one = e; revert e
    rw [testBit_bit_zero, bit_shiftRight_one]
    intros; rfl


@[deprecated (since := "2024-10-21")] alias binaryRec_eq' := binaryRec_eq


