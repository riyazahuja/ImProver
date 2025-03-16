/-- `div2 n = n/2`-/
def div2 : ℤ → ℤ
  | (n : ℕ) => n.div2
  | -[n +1] => negSucc n.div2


/-- `bodd n` returns `true` if `n` is odd -/
def bodd : ℤ → Bool
  | (n : ℕ) => n.bodd
  | -[n +1] => not (n.bodd)


/-- `bit b` appends the digit `b` to the binary representation of
  its integer input. -/
def bit (b : Bool) : ℤ → ℤ :=
  cond b (2 * · + 1) (2 * ·)


/-- `testBit m n` returns whether the `(n+1)ˢᵗ` least significant bit is `1` or `0`-/
def testBit : ℤ → ℕ → Bool
  | (m : ℕ), n => Nat.testBit m n
  | -[m +1], n => !(Nat.testBit m n)


/-- `Int.natBitwise` is an auxiliary definition for `Int.bitwise`. -/
def natBitwise (f : Bool → Bool → Bool) (m n : ℕ) : ℤ :=
  cond (f false false) -[ Nat.bitwise (fun x y => not (f x y)) m n +1] (Nat.bitwise f m n)


/-- `Int.bitwise` applies the function `f` to pairs of bits in the same position in
  the binary representations of its inputs. -/
def bitwise (f : Bool → Bool → Bool) : ℤ → ℤ → ℤ
  | (m : ℕ), (n : ℕ) => natBitwise f m n
  | (m : ℕ), -[n +1] => natBitwise (fun x y => f x (not y)) m n
  | -[m +1], (n : ℕ) => natBitwise (fun x y => f (not x) y) m n
  | -[m +1], -[n +1] => natBitwise (fun x y => f (not x) (not y)) m n


/-- `lnot` flips all the bits in the binary representation of its input -/
def lnot : ℤ → ℤ
  | (m : ℕ) => -[m +1]
  | -[m +1] => m


/-- `lor` takes two integers and returns their bitwise `or`-/
def lor : ℤ → ℤ → ℤ
  | (m : ℕ), (n : ℕ) => m ||| n
  | (m : ℕ), -[n +1] => -[Nat.ldiff n m +1]
  | -[m +1], (n : ℕ) => -[Nat.ldiff m n +1]
  | -[m +1], -[n +1] => -[m &&& n +1]


/-- `land` takes two integers and returns their bitwise `and`-/
def land : ℤ → ℤ → ℤ
  | (m : ℕ), (n : ℕ) => m &&& n
  | (m : ℕ), -[n +1] => Nat.ldiff m n
  | -[m +1], (n : ℕ) => Nat.ldiff n m
  | -[m +1], -[n +1] => -[m ||| n +1]

-- Porting note: I don't know why `Nat.ldiff` got the prime, but I'm matching this change here

/-- `ldiff a b` performs bitwise set difference. For each corresponding
  pair of bits taken as booleans, say `aᵢ` and `bᵢ`, it applies the
  boolean operation `aᵢ ∧ bᵢ` to obtain the `iᵗʰ` bit of the result. -/
def ldiff : ℤ → ℤ → ℤ
  | (m : ℕ), (n : ℕ) => Nat.ldiff m n
  | (m : ℕ), -[n +1] => m &&& n
  | -[m +1], (n : ℕ) => -[m ||| n +1]
  | -[m +1], -[n +1] => Nat.ldiff n m

-- Porting note: I don't know why `Nat.xor'` got the prime, but I'm matching this change here

/-- `xor` computes the bitwise `xor` of two natural numbers -/
protected def xor : ℤ → ℤ → ℤ
  | (m : ℕ), (n : ℕ) => (m ^^^ n)
  | (m : ℕ), -[n +1] => -[(m ^^^ n) +1]
  | -[m +1], (n : ℕ) => -[(m ^^^ n) +1]
  | -[m +1], -[n +1] => (m ^^^ n)


/-- `m <<< n` produces an integer whose binary representation
  is obtained by left-shifting the binary representation of `m` by `n` places -/
instance : ShiftLeft ℤ where
  shiftLeft
  | (m : ℕ), (n : ℕ) => Nat.shiftLeft' false m n
  | (m : ℕ), -[n +1] => m >>> (Nat.succ n)
  | -[m +1], (n : ℕ) => -[Nat.shiftLeft' true m n +1]
  | -[m +1], -[n +1] => -[m >>> (Nat.succ n) +1]


/-- `m >>> n` produces an integer whose binary representation
  is obtained by right-shifting the binary representation of `m` by `n` places -/
instance : ShiftRight ℤ where
  shiftRight m n := m <<< (-n)


@[simp]
theorem bodd_zero : bodd 0 = false :=
  rfl


@[simp]
theorem bodd_one : bodd 1 = true :=
  rfl


theorem bodd_two : bodd 2 = false :=
  rfl


@[simp, norm_cast]
theorem bodd_coe (n : ℕ) : Int.bodd n = Nat.bodd n :=
  rfl


@[simp]
theorem bodd_subNatNat (m n : ℕ) : bodd (subNatNat m n) = xor m.bodd n.bodd := by
  /-
    m n : Nat
    ⊢ Eq (Int.subNatNat m n).bodd (m.bodd.xor n.bodd)
  -/
  apply subNatNat_elim m n fun m n i => bodd i = xor m.bodd n.bodd <;>
  /-
    case hp
    m n : Nat
    ⊢ ∀ (i n : Nat), Eq (↑i).bodd ((HAdd.hAdd n i).bodd.xor n.bodd)
  -/
  intros i j <;>
  /-
    case hp
    m n i j : Nat
    ⊢ Eq (↑i).bodd ((HAdd.hAdd j i).bodd.xor j.bodd)
  -/
  simp only [Int.bodd, Int.bodd_coe, Nat.bodd_add] <;>
  /-
    case hp
    m n i j : Nat
    ⊢ Eq i.bodd ((j.bodd.xor i.bodd).xor j.bodd)
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
  cases Nat.bodd i <;> simp
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem bodd_negOfNat (n : ℕ) : bodd (negOfNat n) = n.bodd := by
  /-
    n : Nat
    ⊢ Eq (Int.negOfNat n).bodd n.bodd
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp +decide
  /-
    case succ
    n✝ : Nat
    ⊢ Eq (Int.negOfNat (HAdd.hAdd n✝ 1)).bodd n✝.bodd.not
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem bodd_neg (n : ℤ) : bodd (-n) = bodd n := by
  cases n with
  | ofNat =>
    rw [← negOfNat_eq, bodd_negOfNat]
    simp
  | negSucc n =>
    rw [neg_negSucc, bodd_coe, Nat.bodd_succ]
    change (!Nat.bodd n) = !(bodd n)
    rw [bodd_coe]
-- Porting note: Heavily refactored proof, used to work all with `simp`:
-- `cases n <;> simp [Neg.neg, Int.natCast_eq_ofNat, Int.neg, bodd, -of_nat_eq_coe]`


@[simp]
theorem bodd_add (m n : ℤ) : bodd (m + n) = xor (bodd m) (bodd n) := by
  /-
    m n : Int
    ⊢ Eq (HAdd.hAdd m n).bodd (m.bodd.xor n.bodd)
  -/
  cases' m with m m <;>
  /-
    case ofNat
    n : Int
    m : Nat
    ⊢ Eq (HAdd.hAdd (Int.ofNat m) n).bodd ((Int.ofNat m).bodd.xor n.bodd)
  -/
  cases' n with n n <;>
  simp only [ofNat_eq_coe, ofNat_add_negSucc, negSucc_add_ofNat,
             negSucc_add_negSucc, bodd_subNatNat] <;>
  /-
    case ofNat.ofNat
    m n : Nat
    ⊢ Eq (HAdd.hAdd ↑m ↑n).bodd ((↑m).bodd.xor (↑n).bodd)
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
  simp only [negSucc_coe, bodd_neg, bodd_coe, ← Nat.bodd_add, Bool.xor_comm, ← Nat.cast_add]
  /-
    case negSucc.negSucc
    m n : Nat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n).succ 1).bodd (HAdd.hAdd (HAdd.hAdd m 1) (HAdd. …
  -/
  rw [← Nat.succ_add, add_assoc]
  /-
    🎉 no goals
  -/
-- Porting note: Heavily refactored proof, used to work all with `simp`:
-- `by cases m with m m; cases n with n n; unfold has_add.add;`
-- `simp [int.add, -of_nat_eq_coe, bool.xor_comm]`


@[simp]
theorem bodd_mul (m n : ℤ) : bodd (m * n) = (bodd m && bodd n) := by
  /-
    m n : Int
    ⊢ Eq (HMul.hMul m n).bodd (m.bodd.and n.bodd)
  -/
  cases' m with m m <;> cases' n with n n <;>
  simp only [ofNat_eq_coe, ofNat_mul_negSucc, negSucc_mul_ofNat, ofNat_mul_ofNat,
             negSucc_mul_negSucc] <;>
  /-
    case ofNat.ofNat
    m n : Nat
    ⊢ Eq (↑(HMul.hMul m n)).bodd ((↑m).bodd.and (↑n).bodd)
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
  simp only [negSucc_coe, bodd_neg, bodd_coe, ← Nat.bodd_mul]
  /-
    🎉 no goals
  -/
-- Porting note: Heavily refactored proof, used to be:
-- `by cases m with m m; cases n with n n;`
-- `simp [← int.mul_def, int.mul, -of_nat_eq_coe, bool.xor_comm]`


theorem bodd_add_div2 : ∀ n, cond (bodd n) 1 0 + 2 * div2 n = n
  | (n : ℕ) => by
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (cond (↑n).bodd 1 0) (HMul.hMul 2 (↑n).div2)) ↑n
    -/
    rw [show (cond (bodd n) 1 0 : ℤ) = (cond (bodd n) 1 0 : ℕ) by cases bodd n <;> rfl]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (↑(cond (↑n).bodd 1 0)) (HMul.hMul 2 (↑n).div2)) ↑n
    -/
    exact congr_arg ofNat n.bodd_add_div2
    /-
      🎉 no goals
    -/
  | -[n+1] => by
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (cond (Int.negSucc n).bodd 1 0) (HMul.hMul 2 (Int.negSucc n).d …
    -/
    refine Eq.trans ?_ (congr_arg negSucc n.bodd_add_div2)
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (cond (Int.negSucc n).bodd 1 0) (HMul.hMul 2 (Int.negSucc n).d …
    -/
    dsimp [bodd]; cases Nat.bodd n <;> dsimp [cond, not, div2, Int.mul]
      /-
        case false
        n : Nat
        ⊢ Eq (HAdd.hAdd 1 (HMul.hMul 2 (Int.negSucc n.div2))) (Int.negSucc (HAdd.hAdd  …
      -/
    · change -[2 * Nat.div2 n+1] = _
      /-
        case false
        n : Nat
        ⊢ Eq (Int.negSucc (HMul.hMul 2 n.div2)) (Int.negSucc (HAdd.hAdd 0 (HMul.hMul 2 …
      -/
      rw [zero_add]
      /-
        🎉 no goals
      -/
      /-
        case true
        n : Nat
        ⊢ Eq (HAdd.hAdd 0 (HMul.hMul 2 (Int.negSucc n.div2))) (Int.negSucc (HAdd.hAdd  …
      -/
    · rw [zero_add, add_comm]
      /-
        case true
        n : Nat
        ⊢ Eq (HMul.hMul 2 (Int.negSucc n.div2)) (Int.negSucc (HAdd.hAdd (HMul.hMul 2 n …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem div2_val : ∀ n, div2 n = n / 2
  | (n : ℕ) => congr_arg ofNat n.div2_val
  | -[n+1] => congr_arg negSucc n.div2_val


theorem bit_val (b n) : bit b n = 2 * n + cond b 1 0 := by
  /-
    b : Bool
    n : Int
    ⊢ Eq (Int.bit b n) (HAdd.hAdd (HMul.hMul 2 n) (cond b 1 0))
  -/
  cases b
    /-
      case false
      n : Int
      ⊢ Eq (Int.bit Bool.false n) (HAdd.hAdd (HMul.hMul 2 n) (cond Bool.false 1 0))
    -/
  · apply (add_zero _).symm
    /-
      🎉 no goals
    -/
    /-
      case true
      n : Int
      ⊢ Eq (Int.bit Bool.true n) (HAdd.hAdd (HMul.hMul 2 n) (cond Bool.true 1 0))
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem bit_decomp (n : ℤ) : bit (bodd n) (div2 n) = n :=
  (bit_val _ _).trans <| (add_comm _ _).trans <| bodd_add_div2 _


/-- Defines a function from `ℤ` conditionally, if it is defined for odd and even integers separately
  using `bit`. -/
def bitCasesOn.{u} {C : ℤ → Sort u} (n) (h : ∀ b n, C (bit b n)) : C n := by
  /-
    C : Int → Sort u
    n : Int
    h : (b : Bool) → (n : Int) → C (Int.bit b n)
    ⊢ C n
  -/
  rw [← bit_decomp n]
  /-
    C : Int → Sort u
    n : Int
    h : (b : Bool) → (n : Int) → C (Int.bit b n)
    ⊢ C (Int.bit n.bodd n.div2)
  -/
  apply h
  /-
    🎉 no goals
  -/


@[simp]
theorem bit_zero : bit false 0 = 0 :=
  rfl


@[simp]
theorem bit_coe_nat (b) (n : ℕ) : bit b n = Nat.bit b n := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (Int.bit b ↑n) ↑(Nat.bit b n)
  -/
  rw [bit_val, Nat.bit_val]
  /-
    b : Bool
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 ↑n) (cond b 1 0)) ↑(HAdd.hAdd (HMul.hMul 2 n) b.t …
  -/
              /-
                🎉 no goals
              -/
  cases b <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem bit_negSucc (b) (n : ℕ) : bit b -[n+1] = -[Nat.bit (not b) n+1] := by
  /-
    b : Bool
    n : Nat
    ⊢ Eq (Int.bit b (Int.negSucc n)) (Int.negSucc (Nat.bit b.not n))
  -/
  rw [bit_val, Nat.bit_val]
  /-
    b : Bool
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (Int.negSucc n)) (cond b 1 0)) (Int.negSucc (HAdd …
  -/
              /-
                🎉 no goals
              -/
  cases b <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem bodd_bit (b n) : bodd (bit b n) = b := by
  /-
    b : Bool
    n : Int
    ⊢ Eq (Int.bit b n).bodd b
  -/
  rw [bit_val]
  /-
    b : Bool
    n : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 n) (cond b 1 0)).bodd b
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
  cases b <;> cases bodd n <;> simp [(show bodd 2 = false by rfl)]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem testBit_bit_zero (b) : ∀ n, testBit (bit b n) 0 = b
                  /-
                    b : Bool
                    n : Nat
                    ⊢ Eq ((Int.bit b ↑n).testBit 0) b
                  -/
  | (n : ℕ) => by rw [bit_coe_nat]; apply Nat.testBit_bit_zero
                                    /-
                                      🎉 no goals
                                    -/
  | -[n+1] => by
    /-
      b : Bool
      n : Nat
      ⊢ Eq ((Int.bit b (Int.negSucc n)).testBit 0) b
    -/
    rw [bit_negSucc]; dsimp [testBit]; rw [Nat.testBit_bit_zero]; clear testBit_bit_zero
    /-
      b : Bool
      n : Nat
      ⊢ Eq b.not.not b
    -/
    cases b <;>
      /-
        case false
        n : Nat
        ⊢ Eq Bool.false.not.not Bool.false
      -/
      /-
        🎉 no goals
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem testBit_bit_succ (m b) : ∀ n, testBit (bit b n) (Nat.succ m) = testBit n m
                  /-
                    m : Nat
                    b : Bool
                    n : Nat
                    ⊢ Eq ((Int.bit b ↑n).testBit m.succ) ((↑n).testBit m)
                  -/
  | (n : ℕ) => by rw [bit_coe_nat]; apply Nat.testBit_bit_succ
                                    /-
                                      🎉 no goals
                                    -/
  | -[n+1] => by
    /-
      m : Nat
      b : Bool
      n : Nat
      ⊢ Eq ((Int.bit b (Int.negSucc n)).testBit m.succ) ((Int.negSucc n).testBit m)
    -/
    dsimp only [testBit]
    /-
      m : Nat
      b : Bool
      n : Nat
      ⊢ Eq (Int.testBit.match_1 (fun x x => Bool) (Int.bit b (Int.negSucc n)) m.succ …
    -/
    simp only [bit_negSucc]
    /-
      m : Nat
      b : Bool
      n : Nat
      ⊢ Eq ((Nat.bit b.not n).testBit m.succ).not (n.testBit m).not
    -/
                /-
                  🎉 no goals
                -/
    cases b <;> simp only [Bool.not_false, Bool.not_true, Nat.testBit_bit_succ]
                /-
                  🎉 no goals
                -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO
-- private unsafe def bitwise_tac : tactic Unit :=
--   sorry

-- Porting note: Was `bitwise_tac` in mathlib

theorem bitwise_or : bitwise or = lor := by
  /-
    ⊢ Eq (Int.bitwise Bool.or) Int.lor
  -/
  funext m n
  /-
    case h.h
    m n : Int
    ⊢ Eq (Int.bitwise Bool.or m n) (m.lor n)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  cases' m with m m <;> cases' n with n n <;> try {rfl}
    <;> simp only [bitwise, natBitwise, Bool.not_false, Bool.or_true, cond_true, lor, Nat.ldiff,
      negSucc.injEq, Bool.true_or, Nat.land]
    /-
      case h.h.ofNat.negSucc
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.or y.not).not) m n) (Nat.bitwise (fun a b =>  …
    -/
  · rw [Nat.bitwise_swap, Function.swap]
    /-
      case h.h.ofNat.negSucc
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun y x => (x.or y.not).not) n m) (Nat.bitwise (fun a b =>  …
    -/
    congr
    /-
      case h.h.ofNat.negSucc.e_f
      m n : Nat
      ⊢ Eq (fun y x => (x.or y.not).not) fun a b => a.and b.not
    -/
    funext x y
    /-
      case h.h.ofNat.negSucc.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (y.or x.not).not (x.and y.not)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.ofNat
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.not.or y).not) m n) (Nat.bitwise (fun a b =>  …
    -/
  · congr
    /-
      case h.h.negSucc.ofNat.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.not.or y).not) fun a b => a.and b.not
    -/
    funext x y
    /-
      case h.h.negSucc.ofNat.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.or y).not (x.and y.not)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.negSucc
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.not.or y.not).not) m n) (HAnd.hAnd m n)
    -/
  · congr
    /-
      case h.h.negSucc.negSucc.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.not.or y.not).not) Bool.and
    -/
    funext x y
    /-
      case h.h.negSucc.negSucc.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.or y.not).not (x.and y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/

-- Porting note: Was `bitwise_tac` in mathlib

theorem bitwise_and : bitwise and = land := by
  /-
    ⊢ Eq (Int.bitwise Bool.and) Int.land
  -/
  funext m n
  /-
    case h.h
    m n : Int
    ⊢ Eq (Int.bitwise Bool.and m n) (m.land n)
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  cases' m with m m <;> cases' n with n n <;> try {rfl}
    <;> simp only [bitwise, natBitwise, Bool.not_false, Bool.or_true,
      cond_false, cond_true, lor, Nat.ldiff, Bool.and_true, negSucc.injEq,
      Bool.and_false, Nat.land]
    /-
      case h.h.negSucc.ofNat
      m n : Nat
      ⊢ Eq (↑(Nat.bitwise (fun x y => x.not.and y) m n)) ((Int.negSucc m).land (Int. …
    -/
  · rw [Nat.bitwise_swap, Function.swap]
    /-
      case h.h.negSucc.ofNat
      m n : Nat
      ⊢ Eq (↑(Nat.bitwise (fun y x => x.not.and y) n m)) ((Int.negSucc m).land (Int. …
    -/
    congr
    /-
      case h.h.negSucc.ofNat.e_a.e_f
      m n : Nat
      ⊢ Eq (fun y x => x.not.and y) fun a b => a.and b.not
    -/
    funext x y
    /-
      case h.h.negSucc.ofNat.e_a.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (y.not.and x) (x.and y.not)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.negSucc
      m n : Nat
      ⊢ Eq (Int.negSucc (Nat.bitwise (fun x y => (x.not.and y.not).not) m n)) ((Int. …
    -/
  · congr
    /-
      case h.h.negSucc.negSucc.e_a.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.not.and y.not).not) Bool.or
    -/
    funext x y
    /-
      case h.h.negSucc.negSucc.e_a.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.and y.not).not (x.or y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/

-- Porting note: Was `bitwise_tac` in mathlib

theorem bitwise_diff : (bitwise fun a b => a && not b) = ldiff := by
  /-
    ⊢ Eq (Int.bitwise fun a b => a.and b.not) Int.ldiff
  -/
  funext m n
  /-
    case h.h
    m n : Int
    ⊢ Eq (Int.bitwise (fun a b => a.and b.not) m n) (m.ldiff n)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  cases' m with m m <;> cases' n with n n <;> try {rfl}
    <;> simp only [bitwise, natBitwise, Bool.not_false, Bool.or_true,
      cond_false, cond_true, lor, Nat.ldiff, Bool.and_true, negSucc.injEq,
      Bool.and_false, Nat.land, Bool.not_true, ldiff, Nat.lor]
    /-
      case h.h.ofNat.negSucc
      m n : Nat
      ⊢ Eq ↑(Nat.bitwise (fun x y => x.and y.not.not) m n) ↑(HAnd.hAnd m n)
    -/
  · congr
    /-
      case h.h.ofNat.negSucc.e_a.e_f
      m n : Nat
      ⊢ Eq (fun x y => x.and y.not.not) Bool.and
    -/
    funext x y
    /-
      case h.h.ofNat.negSucc.e_a.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.and y.not.not) (x.and y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.ofNat
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.not.and y.not).not) m n) (HOr.hOr m n)
    -/
  · congr
    /-
      case h.h.negSucc.ofNat.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.not.and y.not).not) Bool.or
    -/
    funext x y
    /-
      case h.h.negSucc.ofNat.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.and y.not).not (x.or y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.negSucc
      m n : Nat
      ⊢ Eq ↑(Nat.bitwise (fun x y => x.not.and y.not.not) m n) ↑(Nat.bitwise (fun a  …
    -/
  · rw [Nat.bitwise_swap, Function.swap]
    /-
      case h.h.negSucc.negSucc
      m n : Nat
      ⊢ Eq ↑(Nat.bitwise (fun y x => x.not.and y.not.not) n m) ↑(Nat.bitwise (fun a  …
    -/
    congr
    /-
      case h.h.negSucc.negSucc.e_a.e_f
      m n : Nat
      ⊢ Eq (fun y x => x.not.and y.not.not) fun a b => a.and b.not
    -/
    funext x y
    /-
      case h.h.negSucc.negSucc.e_a.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (y.not.and x.not.not) (x.and y.not)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/

-- Porting note: Was `bitwise_tac` in mathlib

theorem bitwise_xor : bitwise xor = Int.xor := by
  /-
    ⊢ Eq (Int.bitwise Bool.xor) Int.xor
  -/
  funext m n
  /-
    case h.h
    m n : Int
    ⊢ Eq (Int.bitwise Bool.xor m n) (m.xor n)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  cases' m with m m <;> cases' n with n n <;> try {rfl}
    <;> simp only [bitwise, natBitwise, Bool.not_false, Bool.or_true, Bool.bne_eq_xor,
      cond_false, cond_true, lor, Nat.ldiff, Bool.and_true, negSucc.injEq, Bool.false_xor,
      Bool.true_xor, Bool.and_false, Nat.land, Bool.not_true, ldiff,
      HOr.hOr, OrOp.or, Nat.lor, Int.xor, HXor.hXor, Xor.xor, Nat.xor]
    /-
      case h.h.ofNat.negSucc
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.xor y.not).not) m n) (Nat.bitwise Bool.xor m n)
    -/
  · congr
    /-
      case h.h.ofNat.negSucc.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.xor y.not).not) Bool.xor
    -/
    funext x y
    /-
      case h.h.ofNat.negSucc.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.xor y.not).not (x.xor y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.ofNat
      m n : Nat
      ⊢ Eq (Nat.bitwise (fun x y => (x.not.xor y).not) m n) (Nat.bitwise Bool.xor m n)
    -/
  · congr
    /-
      case h.h.negSucc.ofNat.e_f
      m n : Nat
      ⊢ Eq (fun x y => (x.not.xor y).not) Bool.xor
    -/
    funext x y
    /-
      case h.h.negSucc.ofNat.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.xor y).not (x.xor y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/
    /-
      case h.h.negSucc.negSucc
      m n : Nat
      ⊢ Eq ↑(Nat.bitwise (fun x y => x.not.xor y.not) m n) ↑(Nat.bitwise Bool.xor m n)
    -/
  · congr
    /-
      case h.h.negSucc.negSucc.e_a.e_f
      m n : Nat
      ⊢ Eq (fun x y => x.not.xor y.not) Bool.xor
    -/
    funext x y
    /-
      case h.h.negSucc.negSucc.e_a.e_f.h.h
      m n : Nat
      x y : Bool
      ⊢ Eq (x.not.xor y.not) (x.xor y)
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
    cases x <;> cases y <;> rfl
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem bitwise_bit (f : Bool → Bool → Bool) (a m b n) :
    bitwise f (bit a m) (bit b n) = bit (f a b) (bitwise f m n) := by
  /-
    f : Bool → Bool → Bool
    a : Bool
    m : Int
    b : Bool
    n : Int
    ⊢ Eq (Int.bitwise f (Int.bit a m) (Int.bit b n)) (Int.bit (f a b) (Int.bitwise …
  -/
  cases' m with m m <;> cases' n with n n <;>
  simp [bitwise, ofNat_eq_coe, bit_coe_nat, natBitwise, Bool.not_false, Bool.not_eq_false',
    bit_negSucc]
    /-
      case ofNat.ofNat
      f : Bool → Bool → Bool
      a b : Bool
      m n : Nat
      ⊢ Eq (cond (f Bool.false Bool.false) (Int.negSucc (Nat.bitwise (fun x y => (f  …
    -/
                                   /-
                                     🎉 no goals
                                   -/
  · by_cases h : f false false <;> simp +decide [h]
                                   /-
                                     🎉 no goals
                                   -/
    /-
      case ofNat.negSucc
      f : Bool → Bool → Bool
      a b : Bool
      m n : Nat
      ⊢ Eq (cond (f Bool.false Bool.true) (Int.negSucc (Nat.bitwise (fun x y => (f x …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · by_cases h : f false true <;> simp +decide [h]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case negSucc.ofNat
      f : Bool → Bool → Bool
      a b : Bool
      m n : Nat
      ⊢ Eq (cond (f Bool.true Bool.false) (Int.negSucc (Nat.bitwise (fun x y => (f x …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · by_cases h : f true false <;> simp +decide [h]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case negSucc.negSucc
      f : Bool → Bool → Bool
      a b : Bool
      m n : Nat
      ⊢ Eq (cond (f Bool.true Bool.true) (Int.negSucc (Nat.bitwise (fun x y => (f x. …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · by_cases h : f true true <;> simp +decide [h]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem lor_bit (a m b n) : lor (bit a m) (bit b n) = bit (a || b) (lor m n) := by
  /-
    a : Bool
    m : Int
    b : Bool
    n : Int
    ⊢ Eq ((Int.bit a m).lor (Int.bit b n)) (Int.bit (a.or b) (m.lor n))
  -/
  rw [← bitwise_or, bitwise_bit]
  /-
    🎉 no goals
  -/


@[simp]
theorem land_bit (a m b n) : land (bit a m) (bit b n) = bit (a && b) (land m n) := by
  /-
    a : Bool
    m : Int
    b : Bool
    n : Int
    ⊢ Eq ((Int.bit a m).land (Int.bit b n)) (Int.bit (a.and b) (m.land n))
  -/
  rw [← bitwise_and, bitwise_bit]
  /-
    🎉 no goals
  -/


@[simp]
theorem ldiff_bit (a m b n) : ldiff (bit a m) (bit b n) = bit (a && not b) (ldiff m n) := by
  /-
    a : Bool
    m : Int
    b : Bool
    n : Int
    ⊢ Eq ((Int.bit a m).ldiff (Int.bit b n)) (Int.bit (a.and b.not) (m.ldiff n))
  -/
  rw [← bitwise_diff, bitwise_bit]
  /-
    🎉 no goals
  -/


@[simp]
theorem lxor_bit (a m b n) : Int.xor (bit a m) (bit b n) = bit (xor a b) (Int.xor m n) := by
  /-
    a : Bool
    m : Int
    b : Bool
    n : Int
    ⊢ Eq ((Int.bit a m).xor (Int.bit b n)) (Int.bit (a.xor b) (m.xor n))
  -/
  rw [← bitwise_xor, bitwise_bit]
  /-
    🎉 no goals
  -/


@[simp]
theorem lnot_bit (b) : ∀ n, lnot (bit b n) = bit (not b) (lnot n)
                  /-
                    b : Bool
                    n : Nat
                    ⊢ Eq (Int.bit b ↑n).lnot (Int.bit b.not (↑n).lnot)
                  -/
  | (n : ℕ) => by simp [lnot]
                  /-
                    🎉 no goals
                  -/
                 /-
                   b : Bool
                   n : Nat
                   ⊢ Eq (Int.bit b (Int.negSucc n)).lnot (Int.bit b.not (Int.negSucc n).lnot)
                 -/
  | -[n+1] => by simp [lnot]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem testBit_bitwise (f : Bool → Bool → Bool) (m n k) :
    testBit (bitwise f m n) k = f (testBit m k) (testBit n k) := by
  /-
    f : Bool → Bool → Bool
    m n : Int
    k : Nat
    ⊢ Eq ((Int.bitwise f m n).testBit k) (f (m.testBit k) (n.testBit k))
  -/
  cases m <;> cases n <;> simp only [testBit, bitwise, natBitwise]
    /-
      case ofNat.ofNat
      f : Bool → Bool → Bool
      k a✝¹ a✝ : Nat
      ⊢ Eq (Int.testBit.match_1 (fun x x => Bool) (cond (f Bool.false Bool.false) (I …
    -/
                                   /-
                                     🎉 no goals
                                   -/
  · by_cases h : f false false <;> simp [h]
                                   /-
                                     🎉 no goals
                                   -/
    /-
      case ofNat.negSucc
      f : Bool → Bool → Bool
      k a✝¹ a✝ : Nat
      ⊢ Eq (Int.testBit.match_1 (fun x x => Bool) (cond (f Bool.false Bool.false.not …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · by_cases h : f false true <;> simp [h]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case negSucc.ofNat
      f : Bool → Bool → Bool
      k a✝¹ a✝ : Nat
      ⊢ Eq (Int.testBit.match_1 (fun x x => Bool) (cond (f Bool.false.not Bool.false …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · by_cases h : f true false <;> simp [h]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case negSucc.negSucc
      f : Bool → Bool → Bool
      k a✝¹ a✝ : Nat
      ⊢ Eq (Int.testBit.match_1 (fun x x => Bool) (cond (f Bool.false.not Bool.false …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · by_cases h : f true true <;> simp [h]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem testBit_lor (m n k) : testBit (lor m n) k = (testBit m k || testBit n k) := by
  /-
    m n : Int
    k : Nat
    ⊢ Eq ((m.lor n).testBit k) ((m.testBit k).or (n.testBit k))
  -/
  rw [← bitwise_or, testBit_bitwise]
  /-
    🎉 no goals
  -/


@[simp]
theorem testBit_land (m n k) : testBit (land m n) k = (testBit m k && testBit n k) := by
  /-
    m n : Int
    k : Nat
    ⊢ Eq ((m.land n).testBit k) ((m.testBit k).and (n.testBit k))
  -/
  rw [← bitwise_and, testBit_bitwise]
  /-
    🎉 no goals
  -/


@[simp]
theorem testBit_ldiff (m n k) : testBit (ldiff m n) k = (testBit m k && not (testBit n k)) := by
  /-
    m n : Int
    k : Nat
    ⊢ Eq ((m.ldiff n).testBit k) ((m.testBit k).and (n.testBit k).not)
  -/
  rw [← bitwise_diff, testBit_bitwise]
  /-
    🎉 no goals
  -/


@[simp]
theorem testBit_lxor (m n k) : testBit (Int.xor m n) k = xor (testBit m k) (testBit n k) := by
  /-
    m n : Int
    k : Nat
    ⊢ Eq ((m.xor n).testBit k) ((m.testBit k).xor (n.testBit k))
  -/
  rw [← bitwise_xor, testBit_bitwise]
  /-
    🎉 no goals
  -/


@[simp]
theorem testBit_lnot : ∀ n k, testBit (lnot n) k = not (testBit n k)
                     /-
                       n k : Nat
                       ⊢ Eq ((↑n).lnot.testBit k) ((↑n).testBit k).not
                     -/
  | (n : ℕ), k => by simp [lnot, testBit]
                     /-
                       🎉 no goals
                     -/
                    /-
                      n k : Nat
                      ⊢ Eq ((Int.negSucc n).lnot.testBit k) ((Int.negSucc n).testBit k).not
                    -/
  | -[n+1], k => by simp [lnot, testBit]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem shiftLeft_neg (m n : ℤ) : m <<< (-n) = m >>> n :=
  rfl


@[simp]
                                                              /-
                                                                m n : Int
                                                                ⊢ Eq (HShiftRight.hShiftRight m (Neg.neg n)) (HShiftLeft.hShiftLeft m n)
                                                              -/
theorem shiftRight_neg (m n : ℤ) : m >>> (-n) = m <<< n := by rw [← shiftLeft_neg, neg_neg]
                                                              /-
                                                                🎉 no goals
                                                              -/

-- Porting note: what's the correct new name?

@[simp]
theorem shiftLeft_coe_nat (m n : ℕ) : (m : ℤ) <<< (n : ℤ) = ↑(m <<< n) := by
  /-
    m n : Nat
    ⊢ Eq (HShiftLeft.hShiftLeft ↑m ↑n) ↑(HShiftLeft.hShiftLeft m n)
  -/
  unfold_projs; simp
                /-
                  🎉 no goals
                -/

-- Porting note: what's the correct new name?

@[simp]
                                                                           /-
                                                                             m n : Nat
                                                                             ⊢ Eq (HShiftRight.hShiftRight ↑m ↑n) ↑(HShiftRight.hShiftRight m n)
                                                                           -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
theorem shiftRight_coe_nat (m n : ℕ) : (m : ℤ) >>> (n : ℤ) = m >>> n := by cases n <;> rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem shiftLeft_negSucc (m n : ℕ) : -[m+1] <<< (n : ℤ) = -[Nat.shiftLeft' true m n+1] :=
  rfl


@[simp]
                                                                               /-
                                                                                 m n : Nat
                                                                                 ⊢ Eq (HShiftRight.hShiftRight (Int.negSucc m) ↑n) (Int.negSucc (HShiftRight.hS …
                                                                               -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
theorem shiftRight_negSucc (m n : ℕ) : -[m+1] >>> (n : ℤ) = -[m >>> n+1] := by cases n <;> rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


/-- Compare with `Int.shiftRight_add`, which doesn't have the coercions `ℕ → ℤ`. -/
theorem shiftRight_add' : ∀ (m : ℤ) (n k : ℕ), m >>> (n + k : ℤ) = (m >>> (n : ℤ)) >>> (k : ℤ)
  | (m : ℕ), n, k => by
    rw [shiftRight_coe_nat, shiftRight_coe_nat, ← Int.ofNat_add, shiftRight_coe_nat,
      Nat.shiftRight_add]
  | -[m+1], n, k => by
    rw [shiftRight_negSucc, shiftRight_negSucc, ← Int.ofNat_add, shiftRight_negSucc,
      Nat.shiftRight_add]


theorem shiftLeft_add : ∀ (m : ℤ) (n : ℕ) (k : ℤ), m <<< (n + k) = (m <<< (n : ℤ)) <<< k
  | (m : ℕ), n, (k : ℕ) =>
                        /-
                          m n k : Nat
                          ⊢ Eq (Nat.shiftLeft' Bool.false m (HAdd.hAdd n k)) (Nat.shiftLeft' Bool.false  …
                        -/
    congr_arg ofNat (by simp [Nat.shiftLeft_eq, Nat.pow_add, mul_assoc])
                        /-
                          🎉 no goals
                        -/
  | -[_+1], _, (k : ℕ) => congr_arg negSucc (Nat.shiftLeft'_add _ _ _ _)
  | (m : ℕ), n, -[k+1] =>
    subNatNat_elim n k.succ (fun n k i => (↑m) <<< i = (Nat.shiftLeft' false m n) >>> k)
      (fun (i n : ℕ) =>
           /-
             m n✝ k i n : Nat
             ⊢ (fun n k i => Eq (HShiftLeft.hShiftLeft (↑m) i) ↑(HShiftRight.hShiftRight (N …
           -/
        by dsimp; simp [← Nat.shiftLeft_sub _ , Nat.add_sub_cancel_left])
                  /-
                    🎉 no goals
                  -/
      fun i n => by
        /-
          m n✝ k i n : Nat
          ⊢ (fun n k i => Eq (HShiftLeft.hShiftLeft (↑m) i) ↑(HShiftRight.hShiftRight (N …
        -/
        dsimp
        simp_rw [negSucc_eq, shiftLeft_neg, Nat.shiftLeft'_false, Nat.shiftRight_add,
          ← Nat.shiftLeft_sub _ le_rfl, Nat.sub_self, Nat.shiftLeft_zero, ← shiftRight_coe_nat,
          ← shiftRight_add', Nat.cast_one]
  | -[m+1], n, -[k+1] =>
    subNatNat_elim n k.succ
      (fun n k i => -[m+1] <<< i = -[(Nat.shiftLeft' true m n) >>> k+1])
      (fun i n =>
        congr_arg negSucc <| by
          /-
            m n✝ k i n : Nat
            ⊢ Eq (Nat.shiftLeft' Bool.true m i) (HShiftRight.hShiftRight (Nat.shiftLeft' B …
          -/
          rw [← Nat.shiftLeft'_sub, Nat.add_sub_cancel_left]; apply Nat.le_add_right)
                                                              /-
                                                                🎉 no goals
                                                              -/
      fun i n =>
      congr_arg negSucc <| by rw [add_assoc, Nat.shiftRight_add, ← Nat.shiftLeft'_sub _ _ le_rfl,
          Nat.sub_self, Nat.shiftLeft']


theorem shiftLeft_sub (m : ℤ) (n : ℕ) (k : ℤ) : m <<< (n - k) = (m <<< (n : ℤ)) >>> k :=
  shiftLeft_add _ _ _


theorem shiftLeft_eq_mul_pow : ∀ (m : ℤ) (n : ℕ), m <<< (n : ℤ) = m * (2 ^ n : ℕ)
                                              /-
                                                m x✝ : Nat
                                                ⊢ Eq (Nat.shiftLeft' Bool.false m x✝) (HMul.hMul m (HPow.hPow 2 x✝))
                                              -/
  | (m : ℕ), _ => congr_arg ((↑) : ℕ → ℤ) (by simp [Nat.shiftLeft_eq])
                                              /-
                                                🎉 no goals
                                              -/
  | -[_+1], _ => @congr_arg ℕ ℤ _ _ (fun i => -i) (Nat.shiftLeft'_tt_eq_mul_pow _ _)


theorem one_shiftLeft (n : ℕ) : 1 <<< (n : ℤ) = (2 ^ n : ℕ) :=
                              /-
                                n : Nat
                                ⊢ Eq (Nat.shiftLeft' Bool.false 1 n) (HPow.hPow 2 n)
                              -/
  congr_arg ((↑) : ℕ → ℤ) (by simp [Nat.shiftLeft_eq])
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem zero_shiftLeft : ∀ n : ℤ, 0 <<< n = 0
                                           /-
                                             n : Nat
                                             ⊢ Eq (Nat.shiftLeft' Bool.false 0 n) 0
                                           -/
  | (n : ℕ) => congr_arg ((↑) : ℕ → ℤ) (by simp)
                                           /-
                                             🎉 no goals
                                           -/
                                          /-
                                            a✝ : Nat
                                            ⊢ Eq (HShiftRight.hShiftRight 0 a✝.succ) 0
                                          -/
  | -[_+1] => congr_arg ((↑) : ℕ → ℤ) (by simp)
                                          /-
                                            🎉 no goals
                                          -/


/-- Compare with `Int.zero_shiftRight`, which has `n : ℕ`. -/
@[simp]
theorem zero_shiftRight' (n : ℤ) : 0 >>> n = 0 :=
  zero_shiftLeft _


