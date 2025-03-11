@[simp]
lemma bitwise_zero_left (m : Nat) : bitwise f 0 m = if f false true then m else 0 := by
  /-
    f : Bool → Bool → Bool
    m : Nat
    ⊢ Eq (Nat.bitwise f 0 m) (ite (Eq (f Bool.false Bool.true) Bool.true) m 0)
  -/
  simp [bitwise]
  /-
    🎉 no goals
  -/


@[simp]
lemma bitwise_zero_right (n : Nat) : bitwise f n 0 = if f true false then n else 0 := by
  /-
    f : Bool → Bool → Bool
    n : Nat
    ⊢ Eq (Nat.bitwise f n 0) (ite (Eq (f Bool.true Bool.false) Bool.true) n 0)
  -/
  unfold bitwise
  /-
    f : Bool → Bool → Bool
    n : Nat
    ⊢ Eq
        (ite (Eq n 0) (ite (Eq (f Bool.false Bool.true) Bool.true) 0 0)
          (ite (Eq 0 0) (ite (Eq (f Bool.true Bool.false) Bool.true) n 0)
            (let n' := HDiv.hDiv n 2;
            let m' := 0 / 2;
            let b₁ := Eq (HMod.hMod n 2) 1;
            let b₂ := Eq (HMod.hMod 0 2) 1;
            let r := Nat.bitwise f n' m';
            ite (Eq (f (Decidable.decide b₁) (Decidable.decide b₂)) Bool.true) (HA …
        (ite (Eq (f Bool.true Bool.false) Bool.true) n 0)
  -/
  simp only [ite_self, decide_false, Nat.zero_div, ite_true, ite_eq_right_iff]
  /-
    f : Bool → Bool → Bool
    n : Nat
    ⊢ Eq n 0 → Eq 0 (ite (Eq (f Bool.true Bool.false) Bool.true) n 0)
  -/
  rintro ⟨⟩
  /-
    case refl
    f : Bool → Bool → Bool
    ⊢ Eq 0 (ite (Eq (f Bool.true Bool.false) Bool.true) 0 0)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


lemma bitwise_zero : bitwise f 0 0 = 0 := by
  /-
    f : Bool → Bool → Bool
    ⊢ Eq (Nat.bitwise f 0 0) 0
  -/
  simp only [bitwise_zero_right, ite_self]
  /-
    🎉 no goals
  -/


lemma bitwise_of_ne_zero {n m : Nat} (hn : n ≠ 0) (hm : m ≠ 0) :
    bitwise f n m = bit (f (bodd n) (bodd m)) (bitwise f (n / 2) (m / 2)) := by
  /-
    f : Bool → Bool → Bool
    n m : Nat
    hn : Ne n 0
    hm : Ne m 0
    ⊢ Eq (Nat.bitwise f n m) (Nat.bit (f n.bodd m.bodd) (Nat.bitwise f (HDiv.hDiv  …
  -/
  conv_lhs => unfold bitwise
  have mod_two_iff_bod x : (x % 2 = 1 : Bool) = bodd x := by
    simp only [mod_two_of_bodd, cond]; cases bodd x <;> rfl
  /-
    f : Bool → Bool → Bool
    n m : Nat
    hn : Ne n 0
    hm : Ne m 0
    mod_two_iff_bod : ∀ (x : Nat), Eq (Decidable.decide (Eq (HMod.hMod x 2) 1)) x. …
    ⊢ Eq
        (ite (Eq n 0) (ite (Eq (f Bool.false Bool.true) Bool.true) m 0)
          (ite (Eq m 0) (ite (Eq (f Bool.true Bool.false) Bool.true) n 0)
            (let n' := HDiv.hDiv n 2;
            let m' := HDiv.hDiv m 2;
            let b₁ := Eq (HMod.hMod n 2) 1;
            let b₂ := Eq (HMod.hMod m 2) 1;
            let r := Nat.bitwise f n' m';
            ite (Eq (f (Decidable.decide b₁) (Decidable.decide b₂)) Bool.true) (HA …
        (Nat.bit (f n.bodd m.bodd) (Nat.bitwise f (HDiv.hDiv n 2) (HDiv.hDiv m 2)))
  -/
  simp only [hn, hm, mod_two_iff_bod, ite_false, bit, two_mul, Bool.cond_eq_ite]
  /-
    f : Bool → Bool → Bool
    n m : Nat
    hn : Ne n 0
    hm : Ne m 0
    mod_two_iff_bod : ∀ (x : Nat), Eq (Decidable.decide (Eq (HMod.hMod x 2) 1)) x. …
    ⊢ Eq (ite (Eq (f n.bodd m.bodd) Bool.true) (HAdd.hAdd (HAdd.hAdd (Nat.bitwise  …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem binaryRec_of_ne_zero {C : Nat → Sort*} (z : C 0) (f : ∀ b n, C n → C (bit b n)) {n}
    (h : n ≠ 0) :
    binaryRec z f n = bit_decomp n ▸ f (bodd n) (div2 n) (binaryRec z f (div2 n)) := by
  cases n using bitCasesOn with
  | h b n =>
    rw [binaryRec_eq _ _ (by right; simpa [bit_eq_zero_iff] using h)]
    generalize_proofs h; revert h
    rw [bodd_bit, div2_bit]
    simp


@[simp]
lemma bitwise_bit {f : Bool → Bool → Bool} (h : f false false = false := by rfl) (a m b n) :
    bitwise f (bit a m) (bit b n) = bit (f a b) (bitwise f m n) := by
  /-
    f : Bool → Bool → Bool
    h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ⊢ Eq (Nat.bitwise f (Nat.bit a m) (Nat.bit b n)) (Nat.bit (f a b) (Nat.bitwise …
  -/
  conv_lhs => unfold bitwise
  #adaptation_note /-- nightly-2024-03-16: simp was
  -- simp (config := { unfoldPartialApp := true }) only [bit, bit1, bit0, Bool.cond_eq_ite] -/
  /-
    f : Bool → Bool → Bool
    h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ⊢ Eq
        (ite (Eq (Nat.bit a m) 0) (ite (Eq (f Bool.false Bool.true) Bool.true) (Na …
          (ite (Eq (Nat.bit b n) 0) (ite (Eq (f Bool.true Bool.false) Bool.true) ( …
            (let n' := HDiv.hDiv (Nat.bit a m) 2;
            let m' := HDiv.hDiv (Nat.bit b n) 2;
            let b₁ := Eq (HMod.hMod (Nat.bit a m) 2) 1;
            let b₂ := Eq (HMod.hMod (Nat.bit b n) 2) 1;
            let r := Nat.bitwise f n' m';
            ite (Eq (f (Decidable.decide b₁) (Decidable.decide b₂)) Bool.true) (HA …
        (Nat.bit (f a b) (Nat.bitwise f m n))
  -/
  simp only [bit, ite_apply, Bool.cond_eq_ite]
  /-
    f : Bool → Bool → Bool
    h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ⊢ Eq (ite (Eq (ite (Eq a Bool.true) (HAdd.hAdd (HMul.hMul 2 m) 1) (HMul.hMul 2 …
  -/
  have h2 x : (x + x + 1) % 2 = 1 := by rw [← two_mul, add_comm]; apply add_mul_mod_self_left
  /-
    f : Bool → Bool → Bool
    h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    h2 : ∀ (x : Nat), Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd x x) 1) 2) 1
    ⊢ Eq (ite (Eq (ite (Eq a Bool.true) (HAdd.hAdd (HMul.hMul 2 m) 1) (HMul.hMul 2 …
  -/
  have h4 x : (x + x + 1) / 2 = x := by rw [← two_mul, add_comm]; simp [add_mul_div_left]
  /-
    f : Bool → Bool → Bool
    h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    h2 : ∀ (x : Nat), Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd x x) 1) 2) 1
    h4 : ∀ (x : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd x x) 1) 2) x
    ⊢ Eq (ite (Eq (ite (Eq a Bool.true) (HAdd.hAdd (HMul.hMul 2 m) 1) (HMul.hMul 2 …
  -/
  cases a <;> cases b <;> simp [h2, h4] <;> split_ifs
        /-
          case pos
          f : Bool → Bool → Bool
          h : autoParam (Eq (f Bool.false Bool.false) Bool.false) _auto✝
          m n : Nat
          h2 : ∀ (x : Nat), Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd x x) 1) 2) 1
          h4 : ∀ (x : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd x x) 1) 2) x
          h✝² : Eq (HMul.hMul 2 m) 0
          h✝¹ : Eq (f Bool.false Bool.true) Bool.true
          h✝ : Eq (f Bool.false Bool.false) Bool.true
          ⊢ Eq (HMul.hMul 2 n) (HAdd.hAdd (HMul.hMul 2 (Nat.bitwise f m n)) 1)
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> simp_all +decide [two_mul]
        /-
          🎉 no goals
        -/


lemma bit_mod_two_eq_zero_iff (a x) :
    bit a x % 2 = 0 ↔ !a := by
  /-
    a : Bool
    x : Nat
    ⊢ Iff (Eq (HMod.hMod (Nat.bit a x) 2) 0) (Eq a.not Bool.true)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma bit_mod_two_eq_one_iff (a x) :
    bit a x % 2 = 1 ↔ a := by
  /-
    a : Bool
    x : Nat
    ⊢ Iff (Eq (HMod.hMod (Nat.bit a x) 2) 1) (Eq a Bool.true)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lor_bit : ∀ a m b n, bit a m ||| bit b n = bit (a || b) (m ||| n) :=
  /-
    ⊢ Eq (Bool.false.or Bool.false) Bool.false
  -/
  bitwise_bit
  /-
    🎉 no goals
  -/


@[simp]
theorem land_bit : ∀ a m b n, bit a m &&& bit b n = bit (a && b) (m &&& n) :=
  /-
    ⊢ Eq (Bool.false.and Bool.false) Bool.false
  -/
  bitwise_bit
  /-
    🎉 no goals
  -/


@[simp]
theorem ldiff_bit : ∀ a m b n, ldiff (bit a m) (bit b n) = bit (a && not b) (ldiff m n) :=
  /-
    ⊢ Eq (Bool.false.and Bool.false.not) Bool.false
  -/
  bitwise_bit
  /-
    🎉 no goals
  -/


@[simp]
theorem xor_bit : ∀ a m b n, bit a m ^^^ bit b n = bit (bne a b) (m ^^^ n) :=
  /-
    ⊢ Eq (bne Bool.false Bool.false) Bool.false
  -/
  bitwise_bit
  /-
    🎉 no goals
  -/


theorem testBit_lor : ∀ m n k, testBit (m ||| n) k = (testBit m k || testBit n k) :=
  testBit_bitwise rfl


theorem testBit_land : ∀ m n k, testBit (m &&& n) k = (testBit m k && testBit n k) :=
  testBit_bitwise rfl


@[simp]
theorem testBit_ldiff : ∀ m n k, testBit (ldiff m n) k = (testBit m k && not (testBit n k)) :=
  testBit_bitwise rfl


@[simp]
theorem bit_false : bit false = (2 * ·) :=
  rfl


@[simp]
theorem bit_true : bit true = (2 * · + 1) :=
  rfl


@[deprecated (since := "2024-10-19")] alias bit_eq_zero := bit_eq_zero_iff


theorem bit_ne_zero_iff {n : ℕ} {b : Bool} : n.bit b ≠ 0 ↔ n = 0 → b = true := by
  /-
    n : Nat
    b : Bool
    ⊢ Iff (Ne (Nat.bit b n) 0) (Eq n 0 → Eq b Bool.true)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An alternative for `bitwise_bit` which replaces the `f false false = false` assumption
with assumptions that neither `bit a m` nor `bit b n` are `0`
(albeit, phrased as the implications `m = 0 → a = true` and `n = 0 → b = true`) -/
lemma bitwise_bit' {f : Bool → Bool → Bool} (a : Bool) (m : Nat) (b : Bool) (n : Nat)
    (ham : m = 0 → a = true) (hbn : n = 0 → b = true) :
    bitwise f (bit a m) (bit b n) = bit (f a b) (bitwise f m n) := by
  /-
    f : Bool → Bool → Bool
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ham : Eq m 0 → Eq a Bool.true
    hbn : Eq n 0 → Eq b Bool.true
    ⊢ Eq (Nat.bitwise f (Nat.bit a m) (Nat.bit b n)) (Nat.bit (f a b) (Nat.bitwise …
  -/
  conv_lhs => unfold bitwise
  /-
    f : Bool → Bool → Bool
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ham : Eq m 0 → Eq a Bool.true
    hbn : Eq n 0 → Eq b Bool.true
    ⊢ Eq
        (ite (Eq (Nat.bit a m) 0) (ite (Eq (f Bool.false Bool.true) Bool.true) (Na …
          (ite (Eq (Nat.bit b n) 0) (ite (Eq (f Bool.true Bool.false) Bool.true) ( …
            (let n' := HDiv.hDiv (Nat.bit a m) 2;
            let m' := HDiv.hDiv (Nat.bit b n) 2;
            let b₁ := Eq (HMod.hMod (Nat.bit a m) 2) 1;
            let b₂ := Eq (HMod.hMod (Nat.bit b n) 2) 1;
            let r := Nat.bitwise f n' m';
            ite (Eq (f (Decidable.decide b₁) (Decidable.decide b₂)) Bool.true) (HA …
        (Nat.bit (f a b) (Nat.bitwise f m n))
  -/
  rw [← bit_ne_zero_iff] at ham hbn
  simp only [ham, hbn, bit_mod_two_eq_one_iff, Bool.decide_coe, ← div2_val, div2_bit, ne_eq,
    ite_false]
  /-
    f : Bool → Bool → Bool
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ham : Ne (Nat.bit a m) 0
    hbn : Ne (Nat.bit b n) 0
    ⊢ Eq (ite (Eq (f a b) Bool.true) (HAdd.hAdd (HAdd.hAdd (Nat.bitwise f m n) (Na …
  -/
  conv_rhs => simp only [bit, two_mul, Bool.cond_eq_ite]
  /-
    f : Bool → Bool → Bool
    a : Bool
    m : Nat
    b : Bool
    n : Nat
    ham : Ne (Nat.bit a m) 0
    hbn : Ne (Nat.bit b n) 0
    ⊢ Eq (ite (Eq (f a b) Bool.true) (HAdd.hAdd (HAdd.hAdd (Nat.bitwise f m n) (Na …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hf <;> rfl
                        /-
                          🎉 no goals
                        -/


lemma bitwise_eq_binaryRec (f : Bool → Bool → Bool) :
    bitwise f =
    binaryRec (fun n => cond (f false true) n 0) fun a m Ia =>
      binaryRec (cond (f true false) (bit a m) 0) fun b n _ => bit (f a b) (Ia n) := by
  /-
    f : Bool → Bool → Bool
    ⊢ Eq (Nat.bitwise f) fun n => Nat.binaryRec (motive := fun x => Nat → Nat) (fu …
  -/
  funext x y
  induction x using binaryRec' generalizing y with
  | z => simp only [bitwise_zero_left, binaryRec_zero, Bool.cond_eq_ite]
  | f xb x hxb ih =>
    rw [← bit_ne_zero_iff] at hxb
    simp_rw [binaryRec_of_ne_zero _ _ hxb, bodd_bit, div2_bit, eq_rec_constant]
    induction y using binaryRec' with
    | z => simp only [bitwise_zero_right, binaryRec_zero, Bool.cond_eq_ite]
    | f yb y hyb =>
      rw [← bit_ne_zero_iff] at hyb
      simp_rw [binaryRec_of_ne_zero _ _ hyb, bitwise_of_ne_zero hxb hyb, bodd_bit, ← div2_val,
        div2_bit, eq_rec_constant, ih]


theorem zero_of_testBit_eq_false {n : ℕ} (h : ∀ i, testBit n i = false) : n = 0 := by
  /-
    n : Nat
    h : ∀ (i : Nat), Eq (n.testBit i) Bool.false
    ⊢ Eq n 0
  -/
  induction' n using Nat.binaryRec with b n hn
    /-
      case z
      h : ∀ (i : Nat), Eq (Nat.testBit 0 i) Bool.false
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case f
      b : Bool
      n : Nat
      hn : (∀ (i : Nat), Eq (n.testBit i) Bool.false) → Eq n 0
      h : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false
      ⊢ Eq (Nat.bit b n) 0
    -/
  · have : b = false := by simpa using h 0
    /-
      case f
      b : Bool
      n : Nat
      hn : (∀ (i : Nat), Eq (n.testBit i) Bool.false) → Eq n 0
      h : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false
      this : Eq b Bool.false
      ⊢ Eq (Nat.bit b n) 0
    -/
    rw [this, bit_false, hn fun i => by rw [← h (i + 1), testBit_bit_succ]]
    /-
      🎉 no goals
    -/


theorem testBit_eq_false_of_lt {n i} (h : n < 2 ^ i) : n.testBit i = false := by
  /-
    n i : Nat
    h : LT.lt n (HPow.hPow 2 i)
    ⊢ Eq (n.testBit i) Bool.false
  -/
  simp [testBit, shiftRight_eq_div_pow, Nat.div_eq_of_lt h]
  /-
    🎉 no goals
  -/


/-- The ith bit is the ith element of `n.bits`. -/
theorem testBit_eq_inth (n i : ℕ) : n.testBit i = n.bits.getI i := by
  /-
    n i : Nat
    ⊢ Eq (n.testBit i) (n.bits.getI i)
  -/
  induction' i with i ih generalizing n
  · simp only [testBit, zero_eq, shiftRight_zero, one_and_eq_mod_two, mod_two_of_bodd,
      bodd_eq_bits_head, List.getI_zero_eq_headI]
    /-
      case zero
      n : Nat
      ⊢ Eq (bne n.bits.headI.toNat 0) n.bits.headI
    -/
                                  /-
                                    🎉 no goals
                                  -/
    cases List.headI (bits n) <;> rfl
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case succ
    i : Nat
    ih : ∀ (n : Nat), Eq (n.testBit i) (n.bits.getI i)
    n : Nat
    ⊢ Eq (n.testBit (HAdd.hAdd i 1)) (n.bits.getI (HAdd.hAdd i 1))
  -/
  conv_lhs => rw [← bit_decomp n]
  /-
    case succ
    i : Nat
    ih : ∀ (n : Nat), Eq (n.testBit i) (n.bits.getI i)
    n : Nat
    ⊢ Eq ((Nat.bit n.bodd n.div2).testBit (HAdd.hAdd i 1)) (n.bits.getI (HAdd.hAdd …
  -/
  rw [testBit_bit_succ, ih n.div2, div2_bits_eq_tail]
  /-
    case succ
    i : Nat
    ih : ∀ (n : Nat), Eq (n.testBit i) (n.bits.getI i)
    n : Nat
    ⊢ Eq (n.bits.tail.getI i) (n.bits.getI (HAdd.hAdd i 1))
  -/
                   /-
                     🎉 no goals
                   -/
  cases n.bits <;> simp
                   /-
                     🎉 no goals
                   -/


theorem exists_most_significant_bit {n : ℕ} (h : n ≠ 0) :
    ∃ i, testBit n i = true ∧ ∀ j, i < j → testBit n j = false := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), LT.lt i j → E …
  -/
  induction' n using Nat.binaryRec with b n hn
    /-
      case z
      h : Ne 0 0
      ⊢ Exists fun i => And (Eq (Nat.testBit 0 i) Bool.true) (∀ (j : Nat), LT.lt i j …
    -/
  · exact False.elim (h rfl)
    /-
      🎉 no goals
    -/
  /-
    case f
    b : Bool
    n : Nat
    hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
    h : Ne (Nat.bit b n) 0
    ⊢ Exists fun i => And (Eq ((Nat.bit b n).testBit i) Bool.true) (∀ (j : Nat), L …
  -/
  by_cases h' : n = 0
    /-
      case pos
      b : Bool
      n : Nat
      hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
      h : Ne (Nat.bit b n) 0
      h' : Eq n 0
      ⊢ Exists fun i => And (Eq ((Nat.bit b n).testBit i) Bool.true) (∀ (j : Nat), L …
    -/
  · subst h'
    rw [show b = true by
        revert h
        cases b <;> simp]
    /-
      case pos
      b : Bool
      hn : Ne 0 0 → Exists fun i => And (Eq (Nat.testBit 0 i) Bool.true) (∀ (j : Nat …
      h : Ne (Nat.bit b 0) 0
      ⊢ Exists fun i => And (Eq ((Nat.bit Bool.true 0).testBit i) Bool.true) (∀ (j : …
    -/
    refine ⟨0, ⟨by rw [testBit_bit_zero], fun j hj => ?_⟩⟩
    /-
      case pos
      b : Bool
      hn : Ne 0 0 → Exists fun i => And (Eq (Nat.testBit 0 i) Bool.true) (∀ (j : Nat …
      h : Ne (Nat.bit b 0) 0
      j : Nat
      hj : LT.lt 0 j
      ⊢ Eq ((Nat.bit Bool.true 0).testBit j) Bool.false
    -/
    obtain ⟨j', rfl⟩ := exists_eq_succ_of_ne_zero (ne_of_gt hj)
    /-
      case pos.intro
      b : Bool
      hn : Ne 0 0 → Exists fun i => And (Eq (Nat.testBit 0 i) Bool.true) (∀ (j : Nat …
      h : Ne (Nat.bit b 0) 0
      j' : Nat
      hj : LT.lt 0 j'.succ
      ⊢ Eq ((Nat.bit Bool.true 0).testBit j'.succ) Bool.false
    -/
    rw [testBit_bit_succ, zero_testBit]
    /-
      🎉 no goals
    -/
    /-
      case neg
      b : Bool
      n : Nat
      hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
      h : Ne (Nat.bit b n) 0
      h' : Not (Eq n 0)
      ⊢ Exists fun i => And (Eq ((Nat.bit b n).testBit i) Bool.true) (∀ (j : Nat), L …
    -/
  · obtain ⟨k, ⟨hk, hk'⟩⟩ := hn h'
    /-
      case neg.intro.intro
      b : Bool
      n : Nat
      hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
      h : Ne (Nat.bit b n) 0
      h' : Not (Eq n 0)
      k : Nat
      hk : Eq (n.testBit k) Bool.true
      hk' : ∀ (j : Nat), LT.lt k j → Eq (n.testBit j) Bool.false
      ⊢ Exists fun i => And (Eq ((Nat.bit b n).testBit i) Bool.true) (∀ (j : Nat), L …
    -/
    refine ⟨k + 1, ⟨by rw [testBit_bit_succ, hk], fun j hj => ?_⟩⟩
    /-
      case neg.intro.intro
      b : Bool
      n : Nat
      hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
      h : Ne (Nat.bit b n) 0
      h' : Not (Eq n 0)
      k : Nat
      hk : Eq (n.testBit k) Bool.true
      hk' : ∀ (j : Nat), LT.lt k j → Eq (n.testBit j) Bool.false
      j : Nat
      hj : LT.lt (HAdd.hAdd k 1) j
      ⊢ Eq ((Nat.bit b n).testBit j) Bool.false
    -/
    obtain ⟨j', rfl⟩ := exists_eq_succ_of_ne_zero (show j ≠ 0 by intro x; subst x; simp at hj)
    /-
      case neg.intro.intro.intro
      b : Bool
      n : Nat
      hn : Ne n 0 → Exists fun i => And (Eq (n.testBit i) Bool.true) (∀ (j : Nat), L …
      h : Ne (Nat.bit b n) 0
      h' : Not (Eq n 0)
      k : Nat
      hk : Eq (n.testBit k) Bool.true
      hk' : ∀ (j : Nat), LT.lt k j → Eq (n.testBit j) Bool.false
      j' : Nat
      hj : LT.lt (HAdd.hAdd k 1) j'.succ
      ⊢ Eq ((Nat.bit b n).testBit j'.succ) Bool.false
    -/
    exact (testBit_bit_succ _ _ _).trans (hk' _ (lt_of_succ_lt_succ hj))
    /-
      🎉 no goals
    -/


theorem lt_of_testBit {n m : ℕ} (i : ℕ) (hn : testBit n i = false) (hm : testBit m i = true)
    (hnm : ∀ j, i < j → testBit n j = testBit m j) : n < m := by
  /-
    n m i : Nat
    hn : Eq (n.testBit i) Bool.false
    hm : Eq (m.testBit i) Bool.true
    hnm : ∀ (j : Nat), LT.lt i j → Eq (n.testBit j) (m.testBit j)
    ⊢ LT.lt n m
  -/
  induction' n using Nat.binaryRec with b n hn' generalizing i m
    /-
      case z
      m i : Nat
      hn : Eq (Nat.testBit 0 i) Bool.false
      hm : Eq (m.testBit i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq (Nat.testBit 0 j) (m.testBit j)
      ⊢ LT.lt 0 m
    -/
  · rw [Nat.pos_iff_ne_zero]
    /-
      case z
      m i : Nat
      hn : Eq (Nat.testBit 0 i) Bool.false
      hm : Eq (m.testBit i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq (Nat.testBit 0 j) (m.testBit j)
      ⊢ Ne m 0
    -/
    rintro rfl
    /-
      case z
      i : Nat
      hn : Eq (Nat.testBit 0 i) Bool.false
      hm : Eq (Nat.testBit 0 i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq (Nat.testBit 0 j) (Nat.testBit 0 j)
      ⊢ False
    -/
    simp at hm
    /-
      🎉 no goals
    -/
  /-
    case f
    b : Bool
    n : Nat
    hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
    m i : Nat
    hn : Eq ((Nat.bit b n).testBit i) Bool.false
    hm : Eq (m.testBit i) Bool.true
    hnm : ∀ (j : Nat), LT.lt i j → Eq ((Nat.bit b n).testBit j) (m.testBit j)
    ⊢ LT.lt (Nat.bit b n) m
  -/
  induction' m using Nat.binaryRec with b' m hm' generalizing i
    /-
      case f.z
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      i : Nat
      hn : Eq ((Nat.bit b n).testBit i) Bool.false
      hm : Eq (Nat.testBit 0 i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq ((Nat.bit b n).testBit j) (Nat.testBit 0 j)
      ⊢ LT.lt (Nat.bit b n) 0
    -/
  · exact False.elim (Bool.false_ne_true ((zero_testBit i).symm.trans hm))
    /-
      🎉 no goals
    -/
  /-
    case f.f
    b : Bool
    n : Nat
    hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
    b' : Bool
    m : Nat
    hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
    i : Nat
    hn : Eq ((Nat.bit b n).testBit i) Bool.false
    hm : Eq ((Nat.bit b' m).testBit i) Bool.true
    hnm : ∀ (j : Nat), LT.lt i j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
    ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
  -/
  by_cases hi : i = 0
    /-
      case pos
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      i : Nat
      hn : Eq ((Nat.bit b n).testBit i) Bool.false
      hm : Eq ((Nat.bit b' m).testBit i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
      hi : Eq i 0
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
  · subst hi
    /-
      case pos
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      hn : Eq ((Nat.bit b n).testBit 0) Bool.false
      hm : Eq ((Nat.bit b' m).testBit 0) Bool.true
      hnm : ∀ (j : Nat), LT.lt 0 j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
    simp only [testBit_bit_zero] at hn hm
    have : n = m :=
      eq_of_testBit_eq fun i => by convert hnm (i + 1) (Nat.zero_lt_succ _) using 1
      <;> rw [testBit_bit_succ]
    /-
      case pos
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      hnm : ∀ (j : Nat), LT.lt 0 j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
      hn : Eq b Bool.false
      hm : Eq b' Bool.true
      this : Eq n m
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
    rw [hn, hm, this, bit_false, bit_true]
    /-
      case pos
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      hnm : ∀ (j : Nat), LT.lt 0 j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
      hn : Eq b Bool.false
      hm : Eq b' Bool.true
      this : Eq n m
      ⊢ LT.lt ((fun x => HMul.hMul 2 x) m) ((fun x => HAdd.hAdd (HMul.hMul 2 x) 1) m)
    -/
    exact Nat.lt_succ_self _
    /-
      🎉 no goals
    -/
    /-
      case neg
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      i : Nat
      hn : Eq ((Nat.bit b n).testBit i) Bool.false
      hm : Eq ((Nat.bit b' m).testBit i) Bool.true
      hnm : ∀ (j : Nat), LT.lt i j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' m).te …
      hi : Not (Eq i 0)
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
  · obtain ⟨i', rfl⟩ := exists_eq_succ_of_ne_zero hi
    /-
      case neg.intro
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      i' : Nat
      hn : Eq ((Nat.bit b n).testBit i'.succ) Bool.false
      hm : Eq ((Nat.bit b' m).testBit i'.succ) Bool.true
      hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' …
      hi : Not (Eq i'.succ 0)
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
    simp only [testBit_bit_succ] at hn hm
    have := hn' _ hn hm fun j hj => by
      convert hnm j.succ (succ_lt_succ hj) using 1 <;> rw [testBit_bit_succ]
    /-
      case neg.intro
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      i' : Nat
      hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' …
      hi : Not (Eq i'.succ 0)
      hn : Eq (n.testBit i') Bool.false
      hm : Eq (m.testBit i') Bool.true
      this : LT.lt n m
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
    have this' : 2 * n < 2 * m := Nat.mul_lt_mul_of_le_of_lt (le_refl _) this Nat.two_pos
    /-
      case neg.intro
      b : Bool
      n : Nat
      hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
      b' : Bool
      m : Nat
      hm' : ∀ (i : Nat), Eq ((Nat.bit b n).testBit i) Bool.false → Eq (m.testBit i)  …
      i' : Nat
      hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit b n).testBit j) ((Nat.bit b' …
      hi : Not (Eq i'.succ 0)
      hn : Eq (n.testBit i') Bool.false
      hm : Eq (m.testBit i') Bool.true
      this : LT.lt n m
      this' : LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
      ⊢ LT.lt (Nat.bit b n) (Nat.bit b' m)
    -/
    cases b <;> cases b'
        /-
          case neg.intro.false.false
          n : Nat
          hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
          m i' : Nat
          hi : Not (Eq i'.succ 0)
          hn : Eq (n.testBit i') Bool.false
          hm : Eq (m.testBit i') Bool.true
          this : LT.lt n m
          this' : LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
          hm' : ∀ (i : Nat), Eq ((Nat.bit Bool.false n).testBit i) Bool.false → Eq (m.te …
          hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit Bool.false n).testBit j) ((N …
          ⊢ LT.lt (Nat.bit Bool.false n) (Nat.bit Bool.false m)
        -/
    <;> simp only [bit_false, bit_true]
      /-
        case neg.intro.false.false
        n : Nat
        hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
        m i' : Nat
        hi : Not (Eq i'.succ 0)
        hn : Eq (n.testBit i') Bool.false
        hm : Eq (m.testBit i') Bool.true
        this : LT.lt n m
        this' : LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
        hm' : ∀ (i : Nat), Eq ((Nat.bit Bool.false n).testBit i) Bool.false → Eq (m.te …
        hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit Bool.false n).testBit j) ((N …
        ⊢ LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
      -/
    · exact this'
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.false.true
        n : Nat
        hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
        m i' : Nat
        hi : Not (Eq i'.succ 0)
        hn : Eq (n.testBit i') Bool.false
        hm : Eq (m.testBit i') Bool.true
        this : LT.lt n m
        this' : LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
        hm' : ∀ (i : Nat), Eq ((Nat.bit Bool.false n).testBit i) Bool.false → Eq (m.te …
        hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit Bool.false n).testBit j) ((N …
        ⊢ LT.lt (HMul.hMul 2 n) (HAdd.hAdd (HMul.hMul 2 m) 1)
      -/
    · exact Nat.lt_add_right 1 this'
      /-
        🎉 no goals
      -/
    · calc
        2 * n + 1 < 2 * n + 2 := lt.base _
        _ ≤ 2 * m := mul_le_mul_left 2 this
      /-
        case neg.intro.true.true
        n : Nat
        hn' : ∀ {m : Nat} (i : Nat), Eq (n.testBit i) Bool.false → Eq (m.testBit i) Bo …
        m i' : Nat
        hi : Not (Eq i'.succ 0)
        hn : Eq (n.testBit i') Bool.false
        hm : Eq (m.testBit i') Bool.true
        this : LT.lt n m
        this' : LT.lt (HMul.hMul 2 n) (HMul.hMul 2 m)
        hm' : ∀ (i : Nat), Eq ((Nat.bit Bool.true n).testBit i) Bool.false → Eq (m.tes …
        hnm : ∀ (j : Nat), LT.lt i'.succ j → Eq ((Nat.bit Bool.true n).testBit j) ((Na …
        ⊢ LT.lt (HAdd.hAdd (HMul.hMul 2 n) 1) (HAdd.hAdd (HMul.hMul 2 m) 1)
      -/
    · exact Nat.succ_lt_succ this'
      /-
        🎉 no goals
      -/


theorem bitwise_swap {f : Bool → Bool → Bool} :
    bitwise (Function.swap f) = Function.swap (bitwise f) := by
  /-
    f : Bool → Bool → Bool
    ⊢ Eq (Nat.bitwise (Function.swap f)) (Function.swap (Nat.bitwise f))
  -/
  funext m n
  /-
    case h.h
    f : Bool → Bool → Bool
    m n : Nat
    ⊢ Eq (Nat.bitwise (Function.swap f) m n) (Function.swap (Nat.bitwise f) m n)
  -/
  simp only [Function.swap]
  /-
    case h.h
    f : Bool → Bool → Bool
    m n : Nat
    ⊢ Eq (Nat.bitwise (Function.swap f) m n) (Nat.bitwise f n m)
  -/
  induction' m using Nat.strongRecOn with m ih generalizing n
  /-
    case h.h.ind
    f : Bool → Bool → Bool
    m : Nat
    ih : ∀ (m_1 : Nat), LT.lt m_1 m → ∀ (n : Nat), Eq (Nat.bitwise (Function.swap  …
    n : Nat
    ⊢ Eq (Nat.bitwise (Function.swap f) m n) (Nat.bitwise f n m)
  -/
  cases' m with m
      /-
        case h.h.ind.zero
        f : Bool → Bool → Bool
        n : Nat
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ (n : Nat), Eq (Nat.bitwise (Function.swap f) m …
        ⊢ Eq (Nat.bitwise (Function.swap f) 0 n) (Nat.bitwise f n 0)
      -/
  <;> cases' n with n
      /-
        case h.h.ind.zero.zero
        f : Bool → Bool → Bool
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ (n : Nat), Eq (Nat.bitwise (Function.swap f) m …
        ⊢ Eq (Nat.bitwise (Function.swap f) 0 0) (Nat.bitwise f 0 0)
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
  <;> try rw [bitwise_zero_left, bitwise_zero_right]
    /-
      case h.h.ind.succ.succ
      f : Bool → Bool → Bool
      m : Nat
      ih : ∀ (m_1 : Nat), LT.lt m_1 (HAdd.hAdd m 1) → ∀ (n : Nat), Eq (Nat.bitwise ( …
      n : Nat
      ⊢ Eq (Nat.bitwise (Function.swap f) (HAdd.hAdd m 1) (HAdd.hAdd n 1)) (Nat.bitw …
    -/
  · specialize ih ((m+1) / 2) (div_lt_self' ..)
    /-
      case h.h.ind.succ.succ
      f : Bool → Bool → Bool
      m n : Nat
      ih : ∀ (n : Nat), Eq (Nat.bitwise (Function.swap f) (HDiv.hDiv (HAdd.hAdd m 1) …
      ⊢ Eq (Nat.bitwise (Function.swap f) (HAdd.hAdd m 1) (HAdd.hAdd n 1)) (Nat.bitw …
    -/
    simp [bitwise_of_ne_zero, ih]
    /-
      🎉 no goals
    -/


/-- If `f` is a commutative operation on bools such that `f false false = false`, then `bitwise f`
    is also commutative. -/
theorem bitwise_comm {f : Bool → Bool → Bool} (hf : ∀ b b', f b b' = f b' b) (n m : ℕ) :
    bitwise f n m = bitwise f m n :=
                                           /-
                                             f : Bool → Bool → Bool
                                             hf : ∀ (b b' : Bool), Eq (f b b') (f b' b)
                                             n m : Nat
                                             this : Eq (Nat.bitwise f) (Function.swap (Nat.bitwise f))
                                             ⊢ Eq (Nat.bitwise f n m) (Nat.bitwise f m n)
                                           -/
  suffices bitwise f = swap (bitwise f) by conv_lhs => rw [this]
                                           /-
                                             🎉 no goals
                                           -/
  calc
    bitwise f = bitwise (swap f) := congr_arg _ <| funext fun _ => funext <| hf _
    _ = swap (bitwise f) := bitwise_swap


theorem lor_comm (n m : ℕ) : n ||| m = m ||| n :=
  bitwise_comm Bool.or_comm n m


theorem land_comm (n m : ℕ) : n &&& m = m &&& n :=
  bitwise_comm Bool.and_comm n m


lemma and_two_pow (n i : ℕ) : n &&& 2 ^ i = (n.testBit i).toNat * 2 ^ i := by
  /-
    n i : Nat
    ⊢ Eq (HAnd.hAnd n (HPow.hPow 2 i)) (HMul.hMul (n.testBit i).toNat (HPow.hPow 2 …
  -/
  refine eq_of_testBit_eq fun j => ?_
  /-
    n i j : Nat
    ⊢ Eq ((HAnd.hAnd n (HPow.hPow 2 i)).testBit j) ((HMul.hMul (n.testBit i).toNat …
  -/
  obtain rfl | hij := Decidable.eq_or_ne i j <;> cases' h : n.testBit i
    /-
      case inl.false
      n i : Nat
      h : Eq (n.testBit i) Bool.false
      ⊢ Eq ((HAnd.hAnd n (HPow.hPow 2 i)).testBit i) ((HMul.hMul Bool.false.toNat (H …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inl.true
      n i : Nat
      h : Eq (n.testBit i) Bool.true
      ⊢ Eq ((HAnd.hAnd n (HPow.hPow 2 i)).testBit i) ((HMul.hMul Bool.true.toNat (HP …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr.false
      n i j : Nat
      hij : Ne i j
      h : Eq (n.testBit i) Bool.false
      ⊢ Eq ((HAnd.hAnd n (HPow.hPow 2 i)).testBit j) ((HMul.hMul Bool.false.toNat (H …
    -/
  · simp [h, testBit_two_pow_of_ne hij]
    /-
      🎉 no goals
    -/
    /-
      case inr.true
      n i j : Nat
      hij : Ne i j
      h : Eq (n.testBit i) Bool.true
      ⊢ Eq ((HAnd.hAnd n (HPow.hPow 2 i)).testBit j) ((HMul.hMul Bool.true.toNat (HP …
    -/
  · simp [h, testBit_two_pow_of_ne hij]
    /-
      🎉 no goals
    -/


lemma two_pow_and (n i : ℕ) : 2 ^ i &&& n = 2 ^ i * (n.testBit i).toNat := by
  /-
    n i : Nat
    ⊢ Eq (HAnd.hAnd (HPow.hPow 2 i) n) (HMul.hMul (HPow.hPow 2 i) (n.testBit i).to …
  -/
  rw [mul_comm, land_comm, and_two_pow]
  /-
    🎉 no goals
  -/


/-- Proving associativity of bitwise operations in general essentially boils down to a huge case
    distinction, so it is shorter to use this tactic instead of proving it in the general case. -/
macro "bitwise_assoc_tac" : tactic => set_option hygiene false in `(tactic| (
  induction' n using Nat.binaryRec with b n hn generalizing m k
  · simp
  induction' m using Nat.binaryRec with b' m hm
  · simp
  induction' k using Nat.binaryRec with b'' k hk
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [hn]`
  -- This is necessary because these are simp lemmas in mathlib
  <;> simp [hn, Bool.or_assoc, Bool.and_assoc, Bool.bne_eq_xor]))


                                                                         /-
                                                                           n m k : Nat
                                                                           ⊢ Eq (HAnd.hAnd (HAnd.hAnd n m) k) (HAnd.hAnd n (HAnd.hAnd m k))
                                                                         -/
theorem land_assoc (n m k : ℕ) : (n &&& m) &&& k = n &&& (m &&& k) := by bitwise_assoc_tac
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                                                        /-
                                                                          n m k : Nat
                                                                          ⊢ Eq (HOr.hOr (HOr.hOr n m) k) (HOr.hOr n (HOr.hOr m k))
                                                                        -/
theorem lor_assoc (n m k : ℕ) : (n ||| m) ||| k = n ||| (m ||| k) := by bitwise_assoc_tac
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

-- These lemmas match `mul_inv_cancel_right` and `mul_inv_cancel_left`.

theorem xor_cancel_right (n m : ℕ) : (m ^^^ n) ^^^ n = m := by
  /-
    n m : Nat
    ⊢ Eq (HXor.hXor (HXor.hXor m n) n) m
  -/
  rw [Nat.xor_assoc, Nat.xor_self, xor_zero]
  /-
    🎉 no goals
  -/


theorem xor_cancel_left (n m : ℕ) : n ^^^ (n ^^^ m) = m := by
  /-
    n m : Nat
    ⊢ Eq (HXor.hXor n (HXor.hXor n m)) m
  -/
  rw [← Nat.xor_assoc, Nat.xor_self, zero_xor]
  /-
    🎉 no goals
  -/


theorem xor_right_injective {n : ℕ} : Function.Injective (HXor.hXor n : ℕ → ℕ) := fun m m' h => by
  /-
    n m m' : Nat
    h : Eq (HXor.hXor n m) (HXor.hXor n m')
    ⊢ Eq m m'
  -/
  rw [← xor_cancel_left n m, ← xor_cancel_left n m', h]
  /-
    🎉 no goals
  -/


theorem xor_left_injective {n : ℕ} : Function.Injective fun m => m ^^^ n :=
  fun m m' (h : m ^^^ n = m' ^^^ n) => by
  /-
    n m m' : Nat
    h : Eq (HXor.hXor m n) (HXor.hXor m' n)
    ⊢ Eq m m'
  -/
  rw [← xor_cancel_right n m, ← xor_cancel_right n m', h]
  /-
    🎉 no goals
  -/


@[simp]
theorem xor_right_inj {n m m' : ℕ} : n ^^^ m = n ^^^ m' ↔ m = m' :=
  xor_right_injective.eq_iff


@[simp]
theorem xor_left_inj {n m m' : ℕ} : m ^^^ n = m' ^^^ n ↔ m = m' :=
  xor_left_injective.eq_iff


@[simp]
theorem xor_eq_zero {n m : ℕ} : n ^^^ m = 0 ↔ n = m := by
  /-
    n m : Nat
    ⊢ Iff (Eq (HXor.hXor n m) 0) (Eq n m)
  -/
  rw [← Nat.xor_self n, xor_right_inj, eq_comm]
  /-
    🎉 no goals
  -/


theorem xor_ne_zero {n m : ℕ} : n ^^^ m ≠ 0 ↔ n ≠ m :=
  xor_eq_zero.not


theorem xor_trichotomy {a b c : ℕ} (h : a ^^^ b ^^^ c ≠ 0) :
    b ^^^ c < a ∨ c ^^^ a < b ∨ a ^^^ b < c := by
  /-
    a b c : Nat
    h : Ne (HXor.hXor (HXor.hXor a b) c) 0
    ⊢ Or (LT.lt (HXor.hXor b c) a) (Or (LT.lt (HXor.hXor c a) b) (LT.lt (HXor.hXor …
  -/
  set v := a ^^^ b ^^^ c with hv
  -- The xor of any two of `a`, `b`, `c` is the xor of `v` and the third.
  have hab : a ^^^ b = c ^^^ v := by
    rw [Nat.xor_comm c, xor_cancel_right]
  have hbc : b ^^^ c = a ^^^ v := by
    rw [← Nat.xor_assoc, xor_cancel_left]
  have hca : c ^^^ a = b ^^^ v := by
    rw [hv, Nat.xor_assoc, Nat.xor_comm a, ← Nat.xor_assoc, xor_cancel_left]
  -- If `i` is the position of the most significant bit of `v`, then at least one of `a`, `b`, `c`
  -- has a one bit at position `i`.
  /-
    a b c : Nat
    v : Nat := HXor.hXor (HXor.hXor a b) c
    h : Ne v 0
    hv : Eq v (HXor.hXor (HXor.hXor a b) c)
    hab : Eq (HXor.hXor a b) (HXor.hXor c v)
    hbc : Eq (HXor.hXor b c) (HXor.hXor a v)
    hca : Eq (HXor.hXor c a) (HXor.hXor b v)
    ⊢ Or (LT.lt (HXor.hXor b c) a) (Or (LT.lt (HXor.hXor c a) b) (LT.lt (HXor.hXor …
  -/
  obtain ⟨i, ⟨hi, hi'⟩⟩ := exists_most_significant_bit h
  have : testBit a i ∨ testBit b i ∨ testBit c i := by
    contrapose! hi
    simp_rw [Bool.eq_false_eq_not_eq_true] at hi ⊢
    rw [testBit_xor, testBit_xor, hi.1, hi.2.1, hi.2.2]
    rfl
  -- If, say, `a` has a one bit at position `i`, then `a xor v` has a zero bit at position `i`, but
  -- the same bits as `a` in positions greater than `j`, so `a xor v < a`.
  /-
    case intro.intro
    a b c : Nat
    v : Nat := HXor.hXor (HXor.hXor a b) c
    h : Ne v 0
    hv : Eq v (HXor.hXor (HXor.hXor a b) c)
    hab : Eq (HXor.hXor a b) (HXor.hXor c v)
    hbc : Eq (HXor.hXor b c) (HXor.hXor a v)
    hca : Eq (HXor.hXor c a) (HXor.hXor b v)
    i : Nat
    hi : Eq (v.testBit i) Bool.true
    hi' : ∀ (j : Nat), LT.lt i j → Eq (v.testBit j) Bool.false
    this : Or (Eq (a.testBit i) Bool.true) (Or (Eq (b.testBit i) Bool.true) (Eq (c …
    ⊢ Or (LT.lt (HXor.hXor b c) a) (Or (LT.lt (HXor.hXor c a) b) (LT.lt (HXor.hXor …
  -/
  obtain h | h | h := this
  /-
    case intro.intro.inl
    a b c : Nat
    v : Nat := HXor.hXor (HXor.hXor a b) c
    h✝ : Ne v 0
    hv : Eq v (HXor.hXor (HXor.hXor a b) c)
    hab : Eq (HXor.hXor a b) (HXor.hXor c v)
    hbc : Eq (HXor.hXor b c) (HXor.hXor a v)
    hca : Eq (HXor.hXor c a) (HXor.hXor b v)
    i : Nat
    hi : Eq (v.testBit i) Bool.true
    hi' : ∀ (j : Nat), LT.lt i j → Eq (v.testBit j) Bool.false
    h : Eq (a.testBit i) Bool.true
    ⊢ Or (LT.lt (HXor.hXor b c) a) (Or (LT.lt (HXor.hXor c a) b) (LT.lt (HXor.hXor …
  -/
  on_goal 1 => left; rw [hbc]
  /-
    case intro.intro.inl.h
    a b c : Nat
    v : Nat := HXor.hXor (HXor.hXor a b) c
    h✝ : Ne v 0
    hv : Eq v (HXor.hXor (HXor.hXor a b) c)
    hab : Eq (HXor.hXor a b) (HXor.hXor c v)
    hbc : Eq (HXor.hXor b c) (HXor.hXor a v)
    hca : Eq (HXor.hXor c a) (HXor.hXor b v)
    i : Nat
    hi : Eq (v.testBit i) Bool.true
    hi' : ∀ (j : Nat), LT.lt i j → Eq (v.testBit j) Bool.false
    h : Eq (a.testBit i) Bool.true
    ⊢ LT.lt (HXor.hXor a v) a
  -/
  on_goal 2 => right; left; rw [hca]
  /-
    case intro.intro.inl.h
    a b c : Nat
    v : Nat := HXor.hXor (HXor.hXor a b) c
    h✝ : Ne v 0
    hv : Eq v (HXor.hXor (HXor.hXor a b) c)
    hab : Eq (HXor.hXor a b) (HXor.hXor c v)
    hbc : Eq (HXor.hXor b c) (HXor.hXor a v)
    hca : Eq (HXor.hXor c a) (HXor.hXor b v)
    i : Nat
    hi : Eq (v.testBit i) Bool.true
    hi' : ∀ (j : Nat), LT.lt i j → Eq (v.testBit j) Bool.false
    h : Eq (a.testBit i) Bool.true
    ⊢ LT.lt (HXor.hXor a v) a
  -/
  on_goal 3 => right; right; rw [hab]
  all_goals
    refine lt_of_testBit i ?_ h fun j hj => ?_
    · rw [testBit_xor, h, hi]
      rfl
    · simp only [testBit_xor, hi' _ hj, Bool.bne_false]


theorem lt_xor_cases {a b c : ℕ} (h : a < b ^^^ c) : a ^^^ c < b ∨ a ^^^ b < c := by
  /-
    a b c : Nat
    h : LT.lt a (HXor.hXor b c)
    ⊢ Or (LT.lt (HXor.hXor a c) b) (LT.lt (HXor.hXor a b) c)
  -/
  obtain ha | hb | hc := xor_trichotomy <| Nat.xor_assoc _ _ _ ▸ xor_ne_zero.2 h.ne
  /-
    case inl
    a b c : Nat
    h : LT.lt a (HXor.hXor b c)
    ha : LT.lt (HXor.hXor b c) a
    ⊢ Or (LT.lt (HXor.hXor a c) b) (LT.lt (HXor.hXor a b) c)
  -/
  exacts [(h.asymm ha).elim, Or.inl <| Nat.xor_comm _ _ ▸ hb, Or.inr hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem xor_mod_two_eq {m n : ℕ} : (m ^^^ n) % 2 = (m + n) % 2 := by
  /-
    m n : Nat
    ⊢ Eq (HMod.hMod (HXor.hXor m n) 2) (HMod.hMod (HAdd.hAdd m n) 2)
  -/
  by_cases h : (m + n) % 2 = 0
  · simp only [h, mod_two_eq_zero_iff_testBit_zero, testBit_zero, xor_mod_two_eq_one, decide_not,
      Bool.decide_iff_dist, Bool.not_eq_false', beq_iff_eq, decide_eq_decide]
    /-
      case pos
      m n : Nat
      h : Eq (HMod.hMod (HAdd.hAdd m n) 2) 0
      ⊢ Iff (Eq (HMod.hMod m 2) 1) (Eq (HMod.hMod n 2) 1)
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      h : Not (Eq (HMod.hMod (HAdd.hAdd m n) 2) 0)
      ⊢ Eq (HMod.hMod (HXor.hXor m n) 2) (HMod.hMod (HAdd.hAdd m n) 2)
    -/
  · simp only [mod_two_ne_zero] at h
    /-
      case neg
      m n : Nat
      h : Eq (HMod.hMod (HAdd.hAdd m n) 2) 1
      ⊢ Eq (HMod.hMod (HXor.hXor m n) 2) (HMod.hMod (HAdd.hAdd m n) 2)
    -/
    simp only [h, xor_mod_two_eq_one]
    /-
      case neg
      m n : Nat
      h : Eq (HMod.hMod (HAdd.hAdd m n) 2) 1
      ⊢ Not (Iff (Eq (HMod.hMod m 2) 1) (Eq (HMod.hMod n 2) 1))
    -/
    omega
    /-
      🎉 no goals
    -/


@[simp]
theorem even_xor {m n : ℕ} : Even (m ^^^ n) ↔ (Even m ↔ Even n) := by
  /-
    m n : Nat
    ⊢ Iff (Even (HXor.hXor m n)) (Iff (Even m) (Even n))
  -/
  simp only [even_iff, xor_mod_two_eq]
  /-
    m n : Nat
    ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd m n) 2) 0) (Iff (Eq (HMod.hMod m 2) 0) (Eq (HM …
  -/
  omega
  /-
    🎉 no goals
  -/


@[simp] theorem bit_lt_two_pow_succ_iff {b x n} : bit b x < 2 ^ (n + 1) ↔ x < 2 ^ n := by
  /-
    b : Bool
    x n : Nat
    ⊢ Iff (LT.lt (Nat.bit b x) (HPow.hPow 2 (HAdd.hAdd n 1))) (LT.lt x (HPow.hPow  …
  -/
                       /-
                         🎉 no goals
                       -/
  cases b <;> simp <;> omega
                       /-
                         🎉 no goals
                       -/


/-- If `x` and `y` fit within `n` bits, then the result of any bitwise operation on `x` and `y` also
fits within `n` bits -/
theorem bitwise_lt {f x y n} (hx : x < 2 ^ n) (hy : y < 2 ^ n) :
    bitwise f x y < 2 ^ n := by
  induction x using Nat.binaryRec' generalizing n y with
  | z =>
    simp only [bitwise_zero_left]
    split <;> assumption
  | @f bx nx hnx ih =>
    cases y using Nat.binaryRec' with
    | z =>
      simp only [bitwise_zero_right]
      split <;> assumption
    | f «by» ny hny =>
      rw [bitwise_bit' _ _ _ _ hnx hny]
      cases n <;> simp_all


lemma shiftLeft_lt {x n m : ℕ} (h : x < 2 ^ n) : x <<< m < 2 ^ (n + m) := by
  /-
    x n m : Nat
    h : LT.lt x (HPow.hPow 2 n)
    ⊢ LT.lt (HShiftLeft.hShiftLeft x m) (HPow.hPow 2 (HAdd.hAdd n m))
  -/
  simp only [Nat.pow_add, shiftLeft_eq, Nat.mul_lt_mul_right (Nat.two_pow_pos _), h]
  /-
    🎉 no goals
  -/


/-- Note that the LHS is the expression used within `Std.BitVec.append`, hence the name. -/
lemma append_lt {x y n m} (hx : x < 2 ^ n) (hy : y < 2 ^ m) : y <<< n ||| x < 2 ^ (n + m) := by
  /-
    x y n m : Nat
    hx : LT.lt x (HPow.hPow 2 n)
    hy : LT.lt y (HPow.hPow 2 m)
    ⊢ LT.lt (HOr.hOr (HShiftLeft.hShiftLeft y n) x) (HPow.hPow 2 (HAdd.hAdd n m))
  -/
  apply bitwise_lt
    /-
      case hx
      x y n m : Nat
      hx : LT.lt x (HPow.hPow 2 n)
      hy : LT.lt y (HPow.hPow 2 m)
      ⊢ LT.lt (HShiftLeft.hShiftLeft y n) (HPow.hPow 2 (HAdd.hAdd n m))
    -/
  · rw [add_comm]; apply shiftLeft_lt hy
                   /-
                     🎉 no goals
                   -/
    /-
      case hy
      x y n m : Nat
      hx : LT.lt x (HPow.hPow 2 n)
      hy : LT.lt y (HPow.hPow 2 m)
      ⊢ LT.lt x (HPow.hPow 2 (HAdd.hAdd n m))
    -/
  · apply lt_of_lt_of_le hx <| Nat.pow_le_pow_right (le_succ _) (le_add_right _ _)
    /-
      🎉 no goals
    -/


