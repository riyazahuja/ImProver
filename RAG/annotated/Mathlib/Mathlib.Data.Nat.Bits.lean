/-- `bxor` denotes the `xor` function i.e. the exclusive-or function on type `Bool`. -/
local notation "bxor" => xor


/-- `boddDiv2 n` returns a 2-tuple of type `(Bool, Nat)` where the `Bool` value indicates whether
`n` is odd or not and the `Nat` value returns `⌊n/2⌋` -/
def boddDiv2 : ℕ → Bool × ℕ
  | 0 => (false, 0)
  | succ n =>
    match boddDiv2 n with
    | (false, m) => (true, m)
    | (true, m) => (false, succ m)


/-- `div2 n = ⌊n/2⌋` the greatest integer smaller than `n/2`-/
def div2 (n : ℕ) : ℕ := (boddDiv2 n).2


/-- `bodd n` returns `true` if `n` is odd -/
def bodd (n : ℕ) : Bool := (boddDiv2 n).1


@[simp] lemma bodd_zero : bodd 0 = false := rfl


@[simp] lemma bodd_one : bodd 1 = true := rfl


lemma bodd_two : bodd 2 = false := rfl


@[simp]
lemma bodd_succ (n : ℕ) : bodd (succ n) = not (bodd n) := by
  /-
    n : Nat
    ⊢ Eq n.succ.bodd n.bodd.not
  -/
  simp only [bodd, boddDiv2]
  /-
    n : Nat
    ⊢ Eq (Nat.boddDiv2.match_1 (fun x => Prod Bool Nat) n.boddDiv2 (fun m => { fst …
  -/
  let ⟨b,m⟩ := boddDiv2 n
  /-
    n : Nat
    b : Bool
    m : Nat
    ⊢ Eq (Nat.boddDiv2.match_1 (fun x => Prod Bool Nat) { fst := b, snd := m } (fu …
  -/
              /-
                🎉 no goals
              -/
  cases b <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
lemma bodd_add (m n : ℕ) : bodd (m + n) = bxor (bodd m) (bodd n) := by
  /-
    m n : Nat
    ⊢ Eq (HAdd.hAdd m n).bodd (m.bodd.xor n.bodd)
  -/
  induction n
  /-
    case zero
    m : Nat
    ⊢ Eq (HAdd.hAdd m 0).bodd (m.bodd.xor (Nat.bodd 0))
  -/
  case zero => simp
  /-
    case succ
    m n✝ : Nat
    a✝ : Eq (HAdd.hAdd m n✝).bodd (m.bodd.xor n✝.bodd)
    ⊢ Eq (HAdd.hAdd m (HAdd.hAdd n✝ 1)).bodd (m.bodd.xor (HAdd.hAdd n✝ 1).bodd)
  -/
  case succ n ih => simp [← Nat.add_assoc, Bool.xor_not, ih]
  /-
    🎉 no goals
  -/


@[simp]
lemma bodd_mul (m n : ℕ) : bodd (m * n) = (bodd m && bodd n) := by
  induction n with
  | zero => simp
  | succ n IH =>
    simp only [mul_succ, bodd_add, IH, bodd_succ]
    cases bodd m <;> cases bodd n <;> rfl


lemma mod_two_of_bodd (n : ℕ) : n % 2 = (bodd n).toNat := by
  /-
    n : Nat
    ⊢ Eq (HMod.hMod n 2) n.bodd.toNat
  -/
  have := congr_arg bodd (mod_add_div n 2)
  simp? [not] at this says
    simp only [bodd_add, bodd_mul, bodd_succ, not, bodd_zero, Bool.false_and, Bool.bne_false]
      at this
  have _ : ∀ b, and false b = false := by
    intro b
    cases b <;> rfl
  have _ : ∀ b, bxor b false = b := by
    intro b
    cases b <;> rfl
  /-
    n : Nat
    this : Eq (HMod.hMod n 2).bodd n.bodd
    x✝¹ : ∀ (b : Bool), Eq (Bool.false.and b) Bool.false
    x✝ : ∀ (b : Bool), Eq (b.xor Bool.false) b
    ⊢ Eq (HMod.hMod n 2) n.bodd.toNat
  -/
  rw [← this]
  /-
    n : Nat
    this : Eq (HMod.hMod n 2).bodd n.bodd
    x✝¹ : ∀ (b : Bool), Eq (Bool.false.and b) Bool.false
    x✝ : ∀ (b : Bool), Eq (b.xor Bool.false) b
    ⊢ Eq (HMod.hMod n 2) (HMod.hMod n 2).bodd.toNat
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rcases mod_two_eq_zero_or_one n with h | h <;> rw [h] <;> rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp] lemma div2_zero : div2 0 = 0 := rfl


@[simp] lemma div2_one : div2 1 = 0 := rfl


lemma div2_two : div2 2 = 1 := rfl


@[simp]
lemma div2_succ (n : ℕ) : div2 (n + 1) = cond (bodd n) (succ (div2 n)) (div2 n) := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd n 1).div2 (cond n.bodd n.div2.succ n.div2)
  -/
  simp only [bodd, boddDiv2, div2]
  /-
    n : Nat
    ⊢ Eq (Nat.boddDiv2.match_1 (fun x => Prod Bool Nat) n.boddDiv2 (fun m => { fst …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rcases boddDiv2 n with ⟨_|_, _⟩ <;> simp
                                      /-
                                        🎉 no goals
                                      -/


lemma bodd_add_div2 : ∀ n, (bodd n).toNat + 2 * div2 n = n
  | 0 => rfl
  | succ n => by
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd n.succ.bodd.toNat (HMul.hMul 2 n.succ.div2)) n.succ
    -/
    simp only [bodd_succ, Bool.cond_not, div2_succ, Nat.mul_comm]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd n.bodd.not.toNat (HMul.hMul (cond n.bodd n.div2.succ n.div2) 2 …
    -/
    refine Eq.trans ?_ (congr_arg succ (bodd_add_div2 n))
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd n.bodd.not.toNat (HMul.hMul (cond n.bodd n.div2.succ n.div2) 2 …
    -/
    cases bodd n
      /-
        case false
        n : Nat
        ⊢ Eq (HAdd.hAdd Bool.false.not.toNat (HMul.hMul (cond Bool.false n.div2.succ n …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case true
        n : Nat
        ⊢ Eq (HAdd.hAdd Bool.true.not.toNat (HMul.hMul (cond Bool.true n.div2.succ n.d …
      -/
    · simp; omega
            /-
              🎉 no goals
            -/


lemma div2_val (n) : div2 n = n / 2 := by
  refine Nat.eq_of_mul_eq_mul_left (by decide)
    (Nat.add_left_cancel (Eq.trans ?_ (Nat.mod_add_div n 2).symm))
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMod.hMod n 2) (HMul.hMul 2 n.div2)) n
  -/
  rw [mod_two_of_bodd, bodd_add_div2]
  /-
    🎉 no goals
  -/


lemma bit_decomp (n : Nat) : bit (bodd n) (div2 n) = n :=
  (bit_val _ _).trans <| (Nat.add_comm _ _).trans <| bodd_add_div2 _


lemma bit_zero : bit false 0 = 0 :=
  rfl


/-- `shiftLeft' b m n` performs a left shift of `m` `n` times
 and adds the bit `b` as the least significant bit each time.
 Returns the corresponding natural number -/
def shiftLeft' (b : Bool) (m : ℕ) : ℕ → ℕ
  | 0 => m
  | n + 1 => bit b (shiftLeft' b m n)


@[simp]
lemma shiftLeft'_false : ∀ n, shiftLeft' false m n = m <<< n
  | 0 => rfl
  | n + 1 => by
    have : 2 * (m * 2^n) = 2^(n+1)*m := by
      rw [Nat.mul_comm, Nat.mul_assoc, ← Nat.pow_succ]; simp
    /-
      m n : Nat
      this : Eq (HMul.hMul 2 (HMul.hMul m (HPow.hPow 2 n))) (HMul.hMul (HPow.hPow 2  …
      ⊢ Eq (Nat.shiftLeft' Bool.false m (HAdd.hAdd n 1)) (HShiftLeft.hShiftLeft m (H …
    -/
    simp [shiftLeft_eq, shiftLeft', bit_val, shiftLeft'_false, this]
    /-
      🎉 no goals
    -/


/-- Lean takes the unprimed name for `Nat.shiftLeft_eq m n : m <<< n = m * 2 ^ n`. -/
@[simp] lemma shiftLeft_eq' (m n : Nat) : shiftLeft m n = m <<< n := rfl

@[simp] lemma shiftRight_eq (m n : Nat) : shiftRight m n = m >>> n := rfl


lemma binaryRec_decreasing (h : n ≠ 0) : div2 n < n := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ LT.lt n.div2 n
  -/
  rw [div2_val]
  /-
    n : Nat
    h : Ne n 0
    ⊢ LT.lt (HDiv.hDiv n 2) n
  -/
  apply (div_lt_iff_lt_mul <| succ_pos 1).2
  have := Nat.mul_lt_mul_of_pos_left (lt_succ_self 1)
    (lt_of_le_of_ne n.zero_le h.symm)
  /-
    n : Nat
    h : Ne n 0
    this : LT.lt (HMul.hMul n 1) (HMul.hMul n (Nat.succ 1))
    ⊢ LT.lt n (HMul.hMul n (Nat.succ 1))
  -/
  rwa [Nat.mul_one] at this
  /-
    🎉 no goals
  -/


/-- `size n` : Returns the size of a natural number in
bits i.e. the length of its binary representation -/
def size : ℕ → ℕ :=
  binaryRec 0 fun _ _ => succ


/-- `bits n` returns a list of Bools which correspond to the binary representation of n, where
    the head of the list represents the least significant bit -/
def bits : ℕ → List Bool :=
  binaryRec [] fun b _ IH => b :: IH


/-- `ldiff a b` performs bitwise set difference. For each corresponding
  pair of bits taken as booleans, say `aᵢ` and `bᵢ`, it applies the
  boolean operation `aᵢ ∧ ¬bᵢ` to obtain the `iᵗʰ` bit of the result. -/
def ldiff : ℕ → ℕ → ℕ :=
  bitwise fun a b => a && not b


lemma bodd_bit (b n) : bodd (bit b n) = b := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (Nat.bit b n).bodd b
  -/
  rw [bit_val]
  simp only [Nat.mul_comm, Nat.add_comm, bodd_add, bodd_mul, bodd_succ, bodd_zero, Bool.not_false,
    Bool.not_true, Bool.and_false, Bool.xor_false]
  /-
    b : Bool
    n : Nat
    ⊢ Eq b.toNat.bodd b
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
  cases b <;> cases bodd n <;> rfl
                               /-
                                 🎉 no goals
                               -/


lemma div2_bit (b n) : div2 (bit b n) = n := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (Nat.bit b n).div2 n
  -/
  rw [bit_val, div2_val, Nat.add_comm, add_mul_div_left, div_eq_of_lt, Nat.zero_add]
      /-
        b : Bool
        n : Nat
        ⊢ LT.lt b.toNat 2
      -/
  <;> cases b
      /-
        case false
        n : Nat
        ⊢ LT.lt Bool.false.toNat 2
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
  <;> decide
      /-
        🎉 no goals
      -/


lemma shiftLeft'_add (b m n) : ∀ k, shiftLeft' b m (n + k) = shiftLeft' b (shiftLeft' b m n) k
  | 0 => rfl
  | k + 1 => congr_arg (bit b) (shiftLeft'_add b m n k)


lemma shiftLeft'_sub (b m) : ∀ {n k}, k ≤ n → shiftLeft' b m (n - k) = (shiftLeft' b m n) >>> k
  | _, 0, _ => rfl
  | n + 1, k + 1, h => by
    /-
      b : Bool
      m n k : Nat
      h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
      ⊢ Eq (Nat.shiftLeft' b m (HSub.hSub (HAdd.hAdd n 1) (HAdd.hAdd k 1))) (HShiftR …
    -/
    rw [succ_sub_succ_eq_sub, shiftLeft', Nat.add_comm, shiftRight_add]
    /-
      b : Bool
      m n k : Nat
      h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
      ⊢ Eq (Nat.shiftLeft' b m (HSub.hSub n k)) (HShiftRight.hShiftRight (HShiftRigh …
    -/
    simp only [shiftLeft'_sub, Nat.le_of_succ_le_succ h, shiftRight_succ, shiftRight_zero]
    /-
      b : Bool
      m n k : Nat
      h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
      ⊢ Eq (HShiftRight.hShiftRight (Nat.shiftLeft' b m n) k) (HShiftRight.hShiftRig …
    -/
    simp [← div2_val, div2_bit]
    /-
      🎉 no goals
    -/


lemma shiftLeft_sub : ∀ (m : Nat) {n k}, k ≤ n → m <<< (n - k) = (m <<< n) >>> k :=
                     /-
                       x✝² x✝¹ x✝ : Nat
                       hk : LE.le x✝ x✝¹
                       ⊢ Eq (HShiftLeft.hShiftLeft x✝² (HSub.hSub x✝¹ x✝)) (HShiftRight.hShiftRight ( …
                     -/
  fun _ _ _ hk => by simp only [← shiftLeft'_false, shiftLeft'_sub false _ hk]
                     /-
                       🎉 no goals
                     -/


lemma bodd_eq_one_and_ne_zero : ∀ n, bodd n = (1 &&& n != 0)
  | 0 => rfl
  | 1 => rfl
                /-
                  n : Nat
                  ⊢ Eq (HAdd.hAdd n 2).bodd (bne (HAnd.hAnd 1 (HAdd.hAdd n 2)) 0)
                -/
  | n + 2 => by simpa using bodd_eq_one_and_ne_zero n
                /-
                  🎉 no goals
                -/


lemma testBit_bit_succ (m b n) : testBit (bit b n) (succ m) = testBit n m := by
  have : bodd (((bit b n) >>> 1) >>> m) = bodd (n >>> m) := by
    simp only [shiftRight_eq_div_pow]
    simp [← div2_val, div2_bit]
  /-
    m : Nat
    b : Bool
    n : Nat
    this : Eq (HShiftRight.hShiftRight (HShiftRight.hShiftRight (Nat.bit b n) 1) m …
    ⊢ Eq ((Nat.bit b n).testBit m.succ) (n.testBit m)
  -/
  rw [← shiftRight_add, Nat.add_comm] at this
  /-
    m : Nat
    b : Bool
    n : Nat
    this : Eq (HShiftRight.hShiftRight (Nat.bit b n) (HAdd.hAdd m 1)).bodd (HShift …
    ⊢ Eq ((Nat.bit b n).testBit m.succ) (n.testBit m)
  -/
  simp only [bodd_eq_one_and_ne_zero] at this
  /-
    m : Nat
    b : Bool
    n : Nat
    this : Eq (bne (HAnd.hAnd 1 (HShiftRight.hShiftRight (Nat.bit b n) (HAdd.hAdd  …
    ⊢ Eq ((Nat.bit b n).testBit m.succ) (n.testBit m)
  -/
  exact this
  /-
    🎉 no goals
  -/


@[simp]
theorem boddDiv2_eq (n : ℕ) : boddDiv2 n = (bodd n, div2 n) := rfl


@[simp]
theorem div2_bit0 (n) : div2 (2 * n) = n :=
  div2_bit false n

-- simp can prove this

theorem div2_bit1 (n) : div2 (2 * n + 1) = n :=
  div2_bit true n


theorem bit_add : ∀ (b : Bool) (n m : ℕ), bit b (n + m) = bit false n + bit b m
                      /-
                        x✝¹ x✝ : Nat
                        ⊢ Eq (Nat.bit Bool.true (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Nat.bit Bool.false x✝¹ …
                      -/
  | true,  _, _ => by dsimp [bit]; omega
                                   /-
                                     🎉 no goals
                                   -/
                      /-
                        x✝¹ x✝ : Nat
                        ⊢ Eq (Nat.bit Bool.false (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Nat.bit Bool.false x✝ …
                      -/
  | false, _, _ => by dsimp [bit]; omega
                                   /-
                                     🎉 no goals
                                   -/


theorem bit_add' : ∀ (b : Bool) (n m : ℕ), bit b (n + m) = bit b n + bit false m
                      /-
                        x✝¹ x✝ : Nat
                        ⊢ Eq (Nat.bit Bool.true (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Nat.bit Bool.true x✝¹) …
                      -/
  | true,  _, _ => by dsimp [bit]; omega
                                   /-
                                     🎉 no goals
                                   -/
                      /-
                        x✝¹ x✝ : Nat
                        ⊢ Eq (Nat.bit Bool.false (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Nat.bit Bool.false x✝ …
                      -/
  | false, _, _ => by dsimp [bit]; omega
                                   /-
                                     🎉 no goals
                                   -/


theorem bit_ne_zero (b) {n} (h : n ≠ 0) : bit b n ≠ 0 := by
  /-
    b : Bool
    n : Nat
    h : Ne n 0
    ⊢ Ne (Nat.bit b n) 0
  -/
                              /-
                                🎉 no goals
                              -/
  cases b <;> dsimp [bit] <;> omega
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem bitCasesOn_bit0 {motive : ℕ → Sort u} (H : ∀ b n, motive (bit b n)) (n : ℕ) :
    bitCasesOn (2 * n) H = H false n :=
  bitCasesOn_bit H false n


@[simp]
theorem bitCasesOn_bit1 {motive : ℕ → Sort u} (H : ∀ b n, motive (bit b n)) (n : ℕ) :
    bitCasesOn (2 * n + 1) H = H true n :=
  bitCasesOn_bit H true n


theorem bit_cases_on_injective {motive : ℕ → Sort u} :
    Function.Injective fun H : ∀ b n, motive (bit b n) => fun n => bitCasesOn n H := by
  /-
    motive : Nat → Sort u
    ⊢ Function.Injective fun H n => Nat.bitCasesOn n H
  -/
  intro H₁ H₂ h
  /-
    motive : Nat → Sort u
    H₁ H₂ : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    h : Eq ((fun H n => Nat.bitCasesOn n H) H₁) ((fun H n => Nat.bitCasesOn n H) H₂)
    ⊢ Eq H₁ H₂
  -/
  ext b n
  /-
    case h.h
    motive : Nat → Sort u
    H₁ H₂ : (b : Bool) → (n : Nat) → motive (Nat.bit b n)
    h : Eq ((fun H n => Nat.bitCasesOn n H) H₁) ((fun H n => Nat.bitCasesOn n H) H₂)
    b : Bool
    n : Nat
    ⊢ Eq (H₁ b n) (H₂ b n)
  -/
  simpa only [bitCasesOn_bit] using congr_fun h (bit b n)
  /-
    🎉 no goals
  -/


@[simp]
theorem bit_cases_on_inj {motive : ℕ → Sort u} (H₁ H₂ : ∀ b n, motive (bit b n)) :
    ((fun n => bitCasesOn n H₁) = fun n => bitCasesOn n H₂) ↔ H₁ = H₂ :=
  bit_cases_on_injective.eq_iff


lemma bit_le : ∀ (b : Bool) {m n : ℕ}, m ≤ n → bit b m ≤ bit b n
                        /-
                          x✝¹ x✝ : Nat
                          h : LE.le x✝¹ x✝
                          ⊢ LE.le (Nat.bit Bool.true x✝¹) (Nat.bit Bool.true x✝)
                        -/
  | true, _, _, h => by dsimp [bit]; omega
                                     /-
                                       🎉 no goals
                                     -/
                         /-
                           x✝¹ x✝ : Nat
                           h : LE.le x✝¹ x✝
                           ⊢ LE.le (Nat.bit Bool.false x✝¹) (Nat.bit Bool.false x✝)
                         -/
  | false, _, _, h => by dsimp [bit]; omega
                                      /-
                                        🎉 no goals
                                      -/


lemma bit_lt_bit (a b) (h : m < n) : bit a m < bit b n := calc
                          /-
                            m n : Nat
                            a b : Bool
                            h : LT.lt m n
                            ⊢ LT.lt (Nat.bit a m) (HMul.hMul 2 n)
                          -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  bit a m < 2 * n   := by cases a <;> dsimp [bit] <;> omega
                                                      /-
                                                        🎉 no goals
                                                      -/
                          /-
                            m n : Nat
                            a b : Bool
                            h : LT.lt m n
                            ⊢ LE.le (HMul.hMul 2 n) (Nat.bit b n)
                          -/
                                                      /-
                                                        🎉 no goals
                                                      -/
        _ ≤ bit b n := by cases b <;> dsimp [bit] <;> omega
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                      /-
                                        ⊢ Eq (Nat.bits 0) List.nil
                                      -/
theorem zero_bits : bits 0 = [] := by simp [Nat.bits]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem bits_append_bit (n : ℕ) (b : Bool) (hn : n = 0 → b = true) :
    (bit b n).bits = b :: n.bits := by
  /-
    n : Nat
    b : Bool
    hn : Eq n 0 → Eq b Bool.true
    ⊢ Eq (Nat.bit b n).bits (List.cons b n.bits)
  -/
  rw [Nat.bits, Nat.bits, binaryRec_eq]
  /-
    case h
    n : Nat
    b : Bool
    hn : Eq n 0 → Eq b Bool.true
    ⊢ Or (Eq (List.cons Bool.false List.nil) List.nil) (Eq n 0 → Eq b Bool.true)
  -/
  simpa
  /-
    🎉 no goals
  -/


@[simp]
theorem bit0_bits (n : ℕ) (hn : n ≠ 0) : (2 * n).bits = false :: n.bits :=
  bits_append_bit n false fun hn' => absurd hn' hn


@[simp]
theorem bit1_bits (n : ℕ) : (2 * n + 1).bits = true :: n.bits :=
  bits_append_bit n true fun _ => rfl


@[simp]
theorem one_bits : Nat.bits 1 = [true] := by
  /-
    ⊢ Eq (Nat.bits 1) (List.cons Bool.true List.nil)
  -/
  convert bit1_bits 0
  /-
    case h.e'_3.h.e'_3
    ⊢ Eq List.nil (Nat.bits 0)
  -/
  simp
  /-
    🎉 no goals
  -/

-- TODO Find somewhere this can live.
-- example : bits 3423 = [true, true, true, true, true, false, true, false, true, false, true, true]
-- := by norm_num


theorem bodd_eq_bits_head (n : ℕ) : n.bodd = n.bits.headI := by
  induction n using Nat.binaryRec' with
  | z => simp
  | f _ _ h _ => simp [bodd_bit, bits_append_bit _ _ h]


theorem div2_bits_eq_tail (n : ℕ) : n.div2.bits = n.bits.tail := by
  induction n using Nat.binaryRec' with
  | z => simp
  | f _ _ h _ => simp [div2_bit, bits_append_bit _ _ h]


