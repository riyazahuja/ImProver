/-- If `p ≠ 0` and `p ≠ 1`, then `padicValNat p p` is `1`. -/
@[simp]
theorem self (hp : 1 < p) : padicValNat p p = 1 := by
  /-
    p : Nat
    hp : LT.lt 1 p
    ⊢ Eq (padicValNat p p) 1
  -/
  simp [padicValNat_def', zero_lt_one.trans hp, hp.ne']
  /-
    🎉 no goals
  -/


theorem eq_zero_of_not_dvd {n : ℕ} (h : ¬p ∣ n) : padicValNat p n = 0 :=
  eq_zero_iff.2 <| Or.inr <| Or.inr h


theorem maxPowDiv_eq_emultiplicity {p n : ℕ} (hp : 1 < p) (hn : 0 < n) :
    p.maxPowDiv n = emultiplicity p n := by
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    ⊢ Eq (↑(p.maxPowDiv n)) (emultiplicity p n)
  -/
  apply (emultiplicity_eq_of_dvd_of_not_dvd (pow_dvd p n) _).symm
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (p.maxPowDiv n) 1)) n)
  -/
  intro h
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd (p.maxPowDiv n) 1)) n
    ⊢ False
  -/
  apply Nat.not_lt.mpr <| le_of_dvd hp hn h
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd (p.maxPowDiv n) 1)) n
    ⊢ LT.lt (p.maxPowDiv n) (HAdd.hAdd (p.maxPowDiv n) 1)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem maxPowDiv_eq_multiplicity {p n : ℕ} (hp : 1 < p) (hn : 0 < n) (h : FiniteMultiplicity p n) :
    p.maxPowDiv n = multiplicity p n := by
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    h : FiniteMultiplicity p n
    ⊢ Eq (p.maxPowDiv n) (multiplicity p n)
  -/
  exact_mod_cast h.emultiplicity_eq_multiplicity ▸ maxPowDiv_eq_emultiplicity hp hn
  /-
    🎉 no goals
  -/


/-- Allows for more efficient code for `padicValNat` -/
@[csimp]
theorem padicValNat_eq_maxPowDiv : @padicValNat = @maxPowDiv := by
  /-
    ⊢ Eq padicValNat Nat.maxPowDiv
  -/
  ext p n
  /-
    case h.h
    p n : Nat
    ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
  -/
  by_cases h : 1 < p ∧ 0 < n
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (LT.lt 0 n)
      ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
    -/
  · rw [padicValNat_def' h.1.ne' h.2, maxPowDiv_eq_multiplicity h.1 h.2]
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (LT.lt 0 n)
      ⊢ FiniteMultiplicity p n
    -/
    exact Nat.finiteMultiplicity_iff.2 ⟨h.1.ne', h.2⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      p n : Nat
      h : Not (And (LT.lt 1 p) (LT.lt 0 n))
      ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
    -/
  · simp only [not_and_or,not_gt_eq,Nat.le_zero] at h
    /-
      case neg
      p n : Nat
      h : Or (LE.le p 1) (Eq n 0)
      ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
    -/
    apply h.elim
      /-
        case neg.left
        p n : Nat
        h : Or (LE.le p 1) (Eq n 0)
        ⊢ LE.le p 1 → Eq (padicValNat p n) (p.maxPowDiv n)
      -/
    · intro h
      /-
        case neg.left
        p n : Nat
        h✝ : Or (LE.le p 1) (Eq n 0)
        h : LE.le p 1
        ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
      -/
      interval_cases p
        /-
          case neg.left.«0»
          p n : Nat
          h✝ : Or (LE.le 0 1) (Eq n 0)
          h : LE.le 0 1
          ⊢ Eq (padicValNat 0 n) (Nat.maxPowDiv 0 n)
        -/
      · simp [Classical.em]
        /-
          🎉 no goals
        -/
        /-
          case neg.left.«1»
          p n : Nat
          h✝ : Or (LE.le 1 1) (Eq n 0)
          h : LE.le 1 1
          ⊢ Eq (padicValNat 1 n) (Nat.maxPowDiv 1 n)
        -/
      · dsimp [padicValNat, maxPowDiv]
        /-
          case neg.left.«1»
          p n : Nat
          h✝ : Or (LE.le 1 1) (Eq n 0)
          h : LE.le 1 1
          ⊢ Eq 0 (Nat.maxPowDiv.go 0 1 n)
        -/
        rw [go, if_neg]; simp
                         /-
                           🎉 no goals
                         -/
      /-
        case neg.right
        p n : Nat
        h : Or (LE.le p 1) (Eq n 0)
        ⊢ Eq n 0 → Eq (padicValNat p n) (p.maxPowDiv n)
      -/
    · intro h
      /-
        case neg.right
        p n : Nat
        h✝ : Or (LE.le p 1) (Eq n 0)
        h : Eq n 0
        ⊢ Eq (padicValNat p n) (p.maxPowDiv n)
      -/
      simp [h]
      /-
        🎉 no goals
      -/


/-- For `p ≠ 1`, the `p`-adic valuation of an integer `z ≠ 0` is the largest natural number `k` such
that `p^k` divides `z`. If `x = 0` or `p = 1`, then `padicValInt p q` defaults to `0`. -/
def padicValInt (p : ℕ) (z : ℤ) : ℕ :=
  padicValNat p z.natAbs


theorem of_ne_one_ne_zero {z : ℤ} (hp : p ≠ 1) (hz : z ≠ 0) :
    padicValInt p z = multiplicity (p : ℤ) z:= by
  /-
    p : Nat
    z : Int
    hp : Ne p 1
    hz : Ne z 0
    ⊢ Eq (padicValInt p z) (multiplicity (↑p) z)
  -/
  rw [padicValInt, padicValNat_def' hp (Int.natAbs_pos.mpr hz)]
  /-
    p : Nat
    z : Int
    hp : Ne p 1
    hz : Ne z 0
    ⊢ Eq (multiplicity p z.natAbs) (multiplicity (↑p) z)
  -/
  apply Int.multiplicity_natAbs
  /-
    🎉 no goals
  -/


/-- `padicValInt p 0` is `0` for any `p`. -/
@[simp]
                                                   /-
                                                     p : Nat
                                                     ⊢ Eq (padicValInt p 0) 0
                                                   -/
protected theorem zero : padicValInt p 0 = 0 := by simp [padicValInt]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `padicValInt p 1` is `0` for any `p`. -/
@[simp]
                                                  /-
                                                    p : Nat
                                                    ⊢ Eq (padicValInt p 1) 0
                                                  -/
protected theorem one : padicValInt p 1 = 0 := by simp [padicValInt]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The `p`-adic value of a natural is its `p`-adic value as an integer. -/
@[simp]
                                                                 /-
                                                                   p n : Nat
                                                                   ⊢ Eq (padicValInt p ↑n) (padicValNat p n)
                                                                 -/
theorem of_nat {n : ℕ} : padicValInt p n = padicValNat p n := by simp [padicValInt]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- If `p ≠ 0` and `p ≠ 1`, then `padicValInt p p` is `1`. -/
                                                      /-
                                                        p : Nat
                                                        hp : LT.lt 1 p
                                                        ⊢ Eq (padicValInt p ↑p) 1
                                                      -/
theorem self (hp : 1 < p) : padicValInt p p = 1 := by simp [padicValNat.self hp]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem eq_zero_of_not_dvd {z : ℤ} (h : ¬(p : ℤ) ∣ z) : padicValInt p z = 0 := by
  /-
    p : Nat
    z : Int
    h : Not (Dvd.dvd (↑p) z)
    ⊢ Eq (padicValInt p z) 0
  -/
  rw [padicValInt, padicValNat.eq_zero_iff]
  /-
    p : Nat
    z : Int
    h : Not (Dvd.dvd (↑p) z)
    ⊢ Or (Eq p 1) (Or (Eq z.natAbs 0) (Not (Dvd.dvd p z.natAbs)))
  -/
  right; right
  /-
    case h.h
    p : Nat
    z : Int
    h : Not (Dvd.dvd (↑p) z)
    ⊢ Not (Dvd.dvd p z.natAbs)
  -/
  rwa [← Int.ofNat_dvd_left]
  /-
    🎉 no goals
  -/


/-- `padicValRat` defines the valuation of a rational `q` to be the valuation of `q.num` minus the
valuation of `q.den`. If `q = 0` or `p = 1`, then `padicValRat p q` defaults to `0`. -/
def padicValRat (p : ℕ) (q : ℚ) : ℤ :=
  padicValInt p q.num - padicValNat p q.den


lemma padicValRat_def (p : ℕ) (q : ℚ) :
    padicValRat p q = padicValInt p q.num - padicValNat p q.den :=
  rfl


/-- `padicValRat p q` is symmetric in `q`. -/
@[simp]
protected theorem neg (q : ℚ) : padicValRat p (-q) = padicValRat p q := by
  /-
    p : Nat
    q : Rat
    ⊢ Eq (padicValRat p (Neg.neg q)) (padicValRat p q)
  -/
  simp [padicValRat, padicValInt]
  /-
    🎉 no goals
  -/


/-- `padicValRat p 0` is `0` for any `p`. -/
@[simp]
                                                   /-
                                                     p : Nat
                                                     ⊢ Eq (padicValRat p 0) 0
                                                   -/
protected theorem zero : padicValRat p 0 = 0 := by simp [padicValRat]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `padicValRat p 1` is `0` for any `p`. -/
@[simp]
                                                  /-
                                                    p : Nat
                                                    ⊢ Eq (padicValRat p 1) 0
                                                  -/
protected theorem one : padicValRat p 1 = 0 := by simp [padicValRat]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The `p`-adic value of an integer `z ≠ 0` is its `p`-adic_value as a rational. -/
@[simp]
                                                                 /-
                                                                   p : Nat
                                                                   z : Int
                                                                   ⊢ Eq (padicValRat p ↑z) ↑(padicValInt p z)
                                                                 -/
theorem of_int {z : ℤ} : padicValRat p z = padicValInt p z := by simp [padicValRat]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The `p`-adic value of an integer `z ≠ 0` is the multiplicity of `p` in `z`. -/
theorem of_int_multiplicity {z : ℤ} (hp : p ≠ 1) (hz : z ≠ 0) :
    padicValRat p (z : ℚ) = multiplicity (p : ℤ) z := by
  /-
    p : Nat
    z : Int
    hp : Ne p 1
    hz : Ne z 0
    ⊢ Eq (padicValRat p ↑z) ↑(multiplicity (↑p) z)
  -/
  rw [of_int, padicValInt.of_ne_one_ne_zero hp hz]
  /-
    🎉 no goals
  -/


theorem multiplicity_sub_multiplicity {q : ℚ} (hp : p ≠ 1) (hq : q ≠ 0) :
    padicValRat p q = multiplicity (p : ℤ) q.num - multiplicity p q.den := by
  rw [padicValRat, padicValInt.of_ne_one_ne_zero hp (Rat.num_ne_zero.2 hq),
    padicValNat_def' hp q.pos]


/-- The `p`-adic value of an integer `z ≠ 0` is its `p`-adic value as a rational. -/
@[simp]
                                                                 /-
                                                                   p n : Nat
                                                                   ⊢ Eq (padicValRat p ↑n) ↑(padicValNat p n)
                                                                 -/
theorem of_nat {n : ℕ} : padicValRat p n = padicValNat p n := by simp [padicValRat]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- If `p ≠ 0` and `p ≠ 1`, then `padicValRat p p` is `1`. -/
                                                      /-
                                                        p : Nat
                                                        hp : LT.lt 1 p
                                                        ⊢ Eq (padicValRat p ↑p) 1
                                                      -/
theorem self (hp : 1 < p) : padicValRat p p = 1 := by simp [hp]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                                       /-
                                                                         p n : Nat
                                                                         ⊢ LE.le 0 (padicValRat p ↑n)
                                                                       -/
theorem zero_le_padicValRat_of_nat (n : ℕ) : 0 ≤ padicValRat p n := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- `padicValRat` coincides with `padicValNat`. -/
@[norm_cast]
                                                                                /-
                                                                                  p n : Nat
                                                                                  ⊢ Eq (↑(padicValNat p n)) (padicValRat p ↑n)
                                                                                -/
theorem padicValRat_of_nat (n : ℕ) : ↑(padicValNat p n) = padicValRat p n := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem padicValNat_self [Fact p.Prime] : padicValNat p p = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (padicValNat p p) 1
  -/
  rw [padicValNat_def (@Fact.out p.Prime).pos]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (multiplicity p p) 1
  -/
  simp
  /-
    🎉 no goals
  -/


theorem one_le_padicValNat_of_dvd {n : ℕ} [hp : Fact p.Prime] (hn : 0 < n) (div : p ∣ n) :
    1 ≤ padicValNat p n := by
  rwa [← WithTop.coe_le_coe, ENat.some_eq_coe, padicValNat_eq_emultiplicity hn,
    ← pow_dvd_iff_le_emultiplicity, pow_one]


theorem dvd_iff_padicValNat_ne_zero {p n : ℕ} [Fact p.Prime] (hn0 : n ≠ 0) :
    p ∣ n ↔ padicValNat p n ≠ 0 :=
  ⟨fun h => one_le_iff_ne_zero.mp (one_le_padicValNat_of_dvd hn0.bot_lt h), fun h =>
    Classical.not_not.1 (mt padicValNat.eq_zero_of_not_dvd h)⟩


/-- The multiplicity of `p : ℕ` in `a : ℤ` is finite exactly when `a ≠ 0`. -/
theorem finite_int_prime_iff {a : ℤ} : FiniteMultiplicity (p : ℤ) a ↔ a ≠ 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : Int
    ⊢ Iff (FiniteMultiplicity (↑p) a) (Ne a 0)
  -/
  simp [Int.finiteMultiplicity_iff, hp.1.ne_one]
  /-
    🎉 no goals
  -/


/-- A rewrite lemma for `padicValRat p q` when `q` is expressed in terms of `Rat.mk`. -/
protected theorem defn (p : ℕ) [hp : Fact p.Prime] {q : ℚ} {n d : ℤ} (hqz : q ≠ 0)
    (qdf : q = n /. d) :
    padicValRat p q = multiplicity (p : ℤ) n - multiplicity (p : ℤ) d := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    n d : Int
    hqz : Ne q 0
    qdf : Eq q (Rat.divInt n d)
    ⊢ Eq (padicValRat p q) (HSub.hSub ↑(multiplicity (↑p) n) ↑(multiplicity (↑p) d))
  -/
  have hd : d ≠ 0 := Rat.mk_denom_ne_zero_of_ne_zero hqz qdf
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    n d : Int
    hqz : Ne q 0
    qdf : Eq q (Rat.divInt n d)
    hd : Ne d 0
    ⊢ Eq (padicValRat p q) (HSub.hSub ↑(multiplicity (↑p) n) ↑(multiplicity (↑p) d))
  -/
  let ⟨c, hc1, hc2⟩ := Rat.num_den_mk hd qdf
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    n d : Int
    hqz : Ne q 0
    qdf : Eq q (Rat.divInt n d)
    hd : Ne d 0
    c : Int
    hc1 : Eq n (HMul.hMul c q.num)
    hc2 : Eq d (HMul.hMul c ↑q.den)
    ⊢ Eq (padicValRat p q) (HSub.hSub ↑(multiplicity (↑p) n) ↑(multiplicity (↑p) d))
  -/
  rw [padicValRat.multiplicity_sub_multiplicity hp.1.ne_one hqz]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    n d : Int
    hqz : Ne q 0
    qdf : Eq q (Rat.divInt n d)
    hd : Ne d 0
    c : Int
    hc1 : Eq n (HMul.hMul c q.num)
    hc2 : Eq d (HMul.hMul c ↑q.den)
    ⊢ Eq (HSub.hSub ↑(multiplicity (↑p) q.num) ↑(multiplicity p q.den)) (HSub.hSub …
  -/
  simp only [Nat.isUnit_iff, hc1, hc2]
  rw [multiplicity_mul (Nat.prime_iff_prime_int.1 hp.1),
    multiplicity_mul (Nat.prime_iff_prime_int.1 hp.1)]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      n d : Int
      hqz : Ne q 0
      qdf : Eq q (Rat.divInt n d)
      hd : Ne d 0
      c : Int
      hc1 : Eq n (HMul.hMul c q.num)
      hc2 : Eq d (HMul.hMul c ↑q.den)
      ⊢ Eq (HSub.hSub ↑(multiplicity (↑p) q.num) ↑(multiplicity p q.den)) (HSub.hSub …
    -/
  · rw [Nat.cast_add, Nat.cast_add]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      n d : Int
      hqz : Ne q 0
      qdf : Eq q (Rat.divInt n d)
      hd : Ne d 0
      c : Int
      hc1 : Eq n (HMul.hMul c q.num)
      hc2 : Eq d (HMul.hMul c ↑q.den)
      ⊢ Eq (HSub.hSub ↑(multiplicity (↑p) q.num) ↑(multiplicity p q.den)) (HSub.hSub …
    -/
    simp_rw [Int.natCast_multiplicity p q.den]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      n d : Int
      hqz : Ne q 0
      qdf : Eq q (Rat.divInt n d)
      hd : Ne d 0
      c : Int
      hc1 : Eq n (HMul.hMul c q.num)
      hc2 : Eq d (HMul.hMul c ↑q.den)
      ⊢ Eq (HSub.hSub ↑(multiplicity (↑p) q.num) ↑(multiplicity p q.den)) (HSub.hSub …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      n d : Int
      hqz : Ne q 0
      qdf : Eq q (Rat.divInt n d)
      hd : Ne d 0
      c : Int
      hc1 : Eq n (HMul.hMul c q.num)
      hc2 : Eq d (HMul.hMul c ↑q.den)
      ⊢ FiniteMultiplicity (↑p) (HMul.hMul c ↑q.den)
    -/
  · simpa [finite_int_prime_iff, hc2] using hd
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      n d : Int
      hqz : Ne q 0
      qdf : Eq q (Rat.divInt n d)
      hd : Ne d 0
      c : Int
      hc1 : Eq n (HMul.hMul c q.num)
      hc2 : Eq d (HMul.hMul c ↑q.den)
      ⊢ FiniteMultiplicity (↑p) (HMul.hMul c q.num)
    -/
  · simpa [finite_int_prime_iff, hqz, hc2] using hd
    /-
      🎉 no goals
    -/


/-- A rewrite lemma for `padicValRat p (q * r)` with conditions `q ≠ 0`, `r ≠ 0`. -/
protected theorem mul {q r : ℚ} (hq : q ≠ 0) (hr : r ≠ 0) :
    padicValRat p (q * r) = padicValRat p q + padicValRat p r := by
  have : q * r = (q.num * r.num) /. (q.den * r.den) := by
    rw [Rat.mul_eq_mkRat, Rat.mkRat_eq_divInt, Nat.cast_mul]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
    ⊢ Eq (padicValRat p (HMul.hMul q r)) (HAdd.hAdd (padicValRat p q) (padicValRat …
  -/
  have hq' : q.num /. q.den ≠ 0 := by rwa [Rat.num_divInt_den]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
    hq' : Ne (Rat.divInt q.num ↑q.den) 0
    ⊢ Eq (padicValRat p (HMul.hMul q r)) (HAdd.hAdd (padicValRat p q) (padicValRat …
  -/
  have hr' : r.num /. r.den ≠ 0 := by rwa [Rat.num_divInt_den]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
    hq' : Ne (Rat.divInt q.num ↑q.den) 0
    hr' : Ne (Rat.divInt r.num ↑r.den) 0
    ⊢ Eq (padicValRat p (HMul.hMul q r)) (HAdd.hAdd (padicValRat p q) (padicValRat …
  -/
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.1 hp.1
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
    hq' : Ne (Rat.divInt q.num ↑q.den) 0
    hr' : Ne (Rat.divInt r.num ↑r.den) 0
    hp' : Prime ↑p
    ⊢ Eq (padicValRat p (HMul.hMul q r)) (HAdd.hAdd (padicValRat p q) (padicValRat …
  -/
  rw [padicValRat.defn p (mul_ne_zero hq hr) this]
  conv_rhs =>
    rw [← q.num_divInt_den, padicValRat.defn p hq', ← r.num_divInt_den, padicValRat.defn p hr']
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
    hq' : Ne (Rat.divInt q.num ↑q.den) 0
    hr' : Ne (Rat.divInt r.num ↑r.den) 0
    hp' : Prime ↑p
    ⊢ Eq (HSub.hSub ↑(multiplicity (↑p) (HMul.hMul q.num r.num)) ↑(multiplicity (↑ …
  -/
  rw [multiplicity_mul hp', multiplicity_mul hp', Nat.cast_add, Nat.cast_add]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hq : Ne q 0
      hr : Ne r 0
      this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
      hq' : Ne (Rat.divInt q.num ↑q.den) 0
      hr' : Ne (Rat.divInt r.num ↑r.den) 0
      hp' : Prime ↑p
      ⊢ Eq (HSub.hSub (HAdd.hAdd ↑(multiplicity (↑p) q.num) ↑(multiplicity (↑p) r.nu …
    -/
  · ring
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hq : Ne q 0
      hr : Ne r 0
      this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
      hq' : Ne (Rat.divInt q.num ↑q.den) 0
      hr' : Ne (Rat.divInt r.num ↑r.den) 0
      hp' : Prime ↑p
      ⊢ FiniteMultiplicity (↑p) (HMul.hMul ↑q.den ↑r.den)
    -/
  · simp [finite_int_prime_iff]
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hq : Ne q 0
      hr : Ne r 0
      this : Eq (HMul.hMul q r) (Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.de …
      hq' : Ne (Rat.divInt q.num ↑q.den) 0
      hr' : Ne (Rat.divInt r.num ↑r.den) 0
      hp' : Prime ↑p
      ⊢ FiniteMultiplicity (↑p) (HMul.hMul q.num r.num)
    -/
  · simp [finite_int_prime_iff, hq, hr]
    /-
      🎉 no goals
    -/


/-- A rewrite lemma for `padicValRat p (q^k)` with condition `q ≠ 0`. -/
protected theorem pow {q : ℚ} (hq : q ≠ 0) {k : ℕ} :
    padicValRat p (q ^ k) = k * padicValRat p q := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    k : Nat
    ⊢ Eq (padicValRat p (HPow.hPow q k)) (HMul.hMul (↑k) (padicValRat p q))
  -/
  induction k <;>
    /-
      case zero
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Ne q 0
      ⊢ Eq (padicValRat p (HPow.hPow q 0)) (HMul.hMul (↑0) (padicValRat p q))
    -/
    /-
      🎉 no goals
    -/
    simp [*, padicValRat.mul hq (pow_ne_zero _ hq), _root_.pow_succ', add_mul, add_comm]
    /-
      🎉 no goals
    -/


/-- A rewrite lemma for `padicValRat p (q⁻¹)` with condition `q ≠ 0`. -/
protected theorem inv (q : ℚ) : padicValRat p q⁻¹ = -padicValRat p q := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    ⊢ Eq (padicValRat p (Inv.inv q)) (Neg.neg (padicValRat p q))
  -/
  by_cases hq : q = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Eq q 0
      ⊢ Eq (padicValRat p (Inv.inv q)) (Neg.neg (padicValRat p q))
    -/
  · simp [hq]
    /-
      🎉 no goals
    -/
  · rw [eq_neg_iff_add_eq_zero, ← padicValRat.mul (inv_ne_zero hq) hq, inv_mul_cancel₀ hq,
      padicValRat.one]


/-- A rewrite lemma for `padicValRat p (q / r)` with conditions `q ≠ 0`, `r ≠ 0`. -/
protected theorem div {q r : ℚ} (hq : q ≠ 0) (hr : r ≠ 0) :
    padicValRat p (q / r) = padicValRat p q - padicValRat p r := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hq : Ne q 0
    hr : Ne r 0
    ⊢ Eq (padicValRat p (HDiv.hDiv q r)) (HSub.hSub (padicValRat p q) (padicValRat …
  -/
  rw [div_eq_mul_inv, padicValRat.mul hq (inv_ne_zero hr), padicValRat.inv r, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- A condition for `padicValRat p (n₁ / d₁) ≤ padicValRat p (n₂ / d₂)`, in terms of
divisibility by `p^n`. -/
theorem padicValRat_le_padicValRat_iff {n₁ n₂ d₁ d₂ : ℤ} (hn₁ : n₁ ≠ 0) (hn₂ : n₂ ≠ 0)
    (hd₁ : d₁ ≠ 0) (hd₂ : d₂ ≠ 0) :
    padicValRat p (n₁ /. d₁) ≤ padicValRat p (n₂ /. d₂) ↔
      ∀ n : ℕ, (p : ℤ) ^ n ∣ n₁ * d₂ → (p : ℤ) ^ n ∣ n₂ * d₁ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n₁ n₂ d₁ d₂ : Int
    hn₁ : Ne n₁ 0
    hn₂ : Ne n₂ 0
    hd₁ : Ne d₁ 0
    hd₂ : Ne d₂ 0
    ⊢ Iff (LE.le (padicValRat p (Rat.divInt n₁ d₁)) (padicValRat p (Rat.divInt n₂  …
  -/
  have hf1 : FiniteMultiplicity (p : ℤ) (n₁ * d₂) := finite_int_prime_iff.2 (mul_ne_zero hn₁ hd₂)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n₁ n₂ d₁ d₂ : Int
    hn₁ : Ne n₁ 0
    hn₂ : Ne n₂ 0
    hd₁ : Ne d₁ 0
    hd₂ : Ne d₂ 0
    hf1 : FiniteMultiplicity (↑p) (HMul.hMul n₁ d₂)
    ⊢ Iff (LE.le (padicValRat p (Rat.divInt n₁ d₁)) (padicValRat p (Rat.divInt n₂  …
  -/
  have hf2 : FiniteMultiplicity (p : ℤ) (n₂ * d₁) := finite_int_prime_iff.2 (mul_ne_zero hn₂ hd₁)
  conv =>
    lhs
    rw [padicValRat.defn p (Rat.divInt_ne_zero_of_ne_zero hn₁ hd₁) rfl,
      padicValRat.defn p (Rat.divInt_ne_zero_of_ne_zero hn₂ hd₂) rfl, sub_le_iff_le_add', ←
      add_sub_assoc, _root_.le_sub_iff_add_le]
    norm_cast
    rw [← multiplicity_mul (Nat.prime_iff_prime_int.1 hp.1) hf1, add_comm,
        ← multiplicity_mul (Nat.prime_iff_prime_int.1 hp.1) hf2,
        hf1.multiplicity_le_multiplicity_iff hf2]


/-- Sufficient conditions to show that the `p`-adic valuation of `q` is less than or equal to the
`p`-adic valuation of `q + r`. -/
theorem le_padicValRat_add_of_le {q r : ℚ} (hqr : q + r ≠ 0)
    (h : padicValRat p q ≤ padicValRat p r) : padicValRat p q ≤ padicValRat p (q + r) :=
                        /-
                          p : Nat
                          hp : Fact (Nat.Prime p)
                          q r : Rat
                          hqr : Ne (HAdd.hAdd q r) 0
                          h : LE.le (padicValRat p q) (padicValRat p r)
                          hq : Eq q 0
                          ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
                        -/
  if hq : q = 0 then by simpa [hq] using h
                        /-
                          🎉 no goals
                        -/
  else
                          /-
                            p : Nat
                            hp : Fact (Nat.Prime p)
                            q r : Rat
                            hqr : Ne (HAdd.hAdd q r) 0
                            h : LE.le (padicValRat p q) (padicValRat p r)
                            hq : Not (Eq q 0)
                            hr : Eq r 0
                            ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
                          -/
    if hr : r = 0 then by simp [hr]
                          /-
                            🎉 no goals
                          -/
    else by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hqn : q.num ≠ 0 := Rat.num_ne_zero.2 hq
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hqd : (q.den : ℤ) ≠ 0 := mod_cast Rat.den_nz _
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        hqd : Ne (↑q.den) 0
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hrn : r.num ≠ 0 := Rat.num_ne_zero.2 hr
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        hqd : Ne (↑q.den) 0
        hrn : Ne r.num 0
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hrd : (r.den : ℤ) ≠ 0 := mod_cast Rat.den_nz _
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        hqd : Ne (↑q.den) 0
        hrn : Ne r.num 0
        hrd : Ne (↑r.den) 0
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hqreq : q + r = (q.num * r.den + q.den * r.num) /. (q.den * r.den) := Rat.add_num_den _ _
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        hqd : Ne (↑q.den) 0
        hrn : Ne r.num 0
        hrd : Ne (↑r.den) 0
        hqreq : Eq (HAdd.hAdd q r) (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HM …
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      have hqrd : q.num * r.den + q.den * r.num ≠ 0 := Rat.mk_num_ne_zero_of_ne_zero hqr hqreq
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hqr : Ne (HAdd.hAdd q r) 0
        h : LE.le (padicValRat p q) (padicValRat p r)
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        hqn : Ne q.num 0
        hqd : Ne (↑q.den) 0
        hrn : Ne r.num 0
        hrd : Ne (↑r.den) 0
        hqreq : Eq (HAdd.hAdd q r) (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HM …
        hqrd : Ne (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul (↑q.den) r.num)) 0
        ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
      -/
      conv_lhs => rw [← q.num_divInt_den]
      rw [hqreq, padicValRat_le_padicValRat_iff hqn hqrd hqd (mul_ne_zero hqd hrd), ←
        emultiplicity_le_emultiplicity_iff, mul_left_comm,
        emultiplicity_mul (Nat.prime_iff_prime_int.1 hp.1), add_mul]
      rw [← q.num_divInt_den, ← r.num_divInt_den, padicValRat_le_padicValRat_iff hqn hrn hqd hrd, ←
        emultiplicity_le_emultiplicity_iff] at h
      calc
        _ ≤
            min (emultiplicity (↑p) (q.num * r.den * q.den))
              (emultiplicity (↑p) (↑q.den * r.num * ↑q.den)) :=
          le_min
            (by rw [emultiplicity_mul (a :=_ * _) (Nat.prime_iff_prime_int.1 hp.1), add_comm])
            (by
              rw [mul_assoc,
                  emultiplicity_mul (b := _ * _) (Nat.prime_iff_prime_int.1 hp.1)]
              exact add_le_add_left h _)
        _ ≤ _ := min_le_emultiplicity_add


/-- The minimum of the valuations of `q` and `r` is at most the valuation of `q + r`. -/
theorem min_le_padicValRat_add {q r : ℚ} (hqr : q + r ≠ 0) :
    min (padicValRat p q) (padicValRat p r) ≤ padicValRat p (q + r) :=
  (le_total (padicValRat p q) (padicValRat p r)).elim
               /-
                 p : Nat
                 hp : Fact (Nat.Prime p)
                 q r : Rat
                 hqr : Ne (HAdd.hAdd q r) 0
                 h : LE.le (padicValRat p q) (padicValRat p r)
                 ⊢ LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd.hAd …
               -/
  (fun h => by rw [min_eq_left h]; exact le_padicValRat_add_of_le hqr h)
                                   /-
                                     🎉 no goals
                                   -/
               /-
                 p : Nat
                 hp : Fact (Nat.Prime p)
                 q r : Rat
                 hqr : Ne (HAdd.hAdd q r) 0
                 h : LE.le (padicValRat p r) (padicValRat p q)
                 ⊢ LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd.hAd …
               -/
  (fun h => by rw [min_eq_right h, add_comm]; exact le_padicValRat_add_of_le (by rwa [add_comm]) h)
                                              /-
                                                🎉 no goals
                                              -/


/-- Ultrametric property of a p-adic valuation. -/
lemma add_eq_min {q r : ℚ} (hqr : q + r ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
    (hval : padicValRat p q ≠ padicValRat p r) :
    padicValRat p (q + r) = min (padicValRat p q) (padicValRat p r) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  have h1 := min_le_padicValRat_add (p := p) hqr
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    h1 : LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd. …
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  have h2 := min_le_padicValRat_add (p := p) (ne_of_eq_of_ne (add_neg_cancel_right q r) hq)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    h1 : LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd. …
    h2 : LE.le (Min.min (padicValRat p (HAdd.hAdd q r)) (padicValRat p (Neg.neg r) …
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  have h3 := min_le_padicValRat_add (p := p) (ne_of_eq_of_ne (add_neg_cancel_right r q) hr)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    h1 : LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd. …
    h2 : LE.le (Min.min (padicValRat p (HAdd.hAdd q r)) (padicValRat p (Neg.neg r) …
    h3 : LE.le (Min.min (padicValRat p (HAdd.hAdd r q)) (padicValRat p (Neg.neg q) …
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  rw [add_neg_cancel_right, padicValRat.neg] at h2 h3
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    h1 : LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd. …
    h2 : LE.le (Min.min (padicValRat p (HAdd.hAdd q r)) (padicValRat p r)) (padicV …
    h3 : LE.le (Min.min (padicValRat p (HAdd.hAdd r q)) (padicValRat p q)) (padicV …
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  rw [add_comm] at h3
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : Ne (padicValRat p q) (padicValRat p r)
    h1 : LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd. …
    h2 : LE.le (Min.min (padicValRat p (HAdd.hAdd q r)) (padicValRat p r)) (padicV …
    h3 : LE.le (Min.min (padicValRat p (HAdd.hAdd q r)) (padicValRat p q)) (padicV …
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (Min.min (padicValRat p q) (padicValRat p …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma add_eq_of_lt {q r : ℚ} (hqr : q + r ≠ 0)
    (hq : q ≠ 0) (hr : r ≠ 0) (hval : padicValRat p q < padicValRat p r) :
    padicValRat p (q + r) = padicValRat p q := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hqr : Ne (HAdd.hAdd q r) 0
    hq : Ne q 0
    hr : Ne r 0
    hval : LT.lt (padicValRat p q) (padicValRat p r)
    ⊢ Eq (padicValRat p (HAdd.hAdd q r)) (padicValRat p q)
  -/
  rw [add_eq_min hqr hq hr (ne_of_lt hval), min_eq_left (le_of_lt hval)]
  /-
    🎉 no goals
  -/


lemma lt_add_of_lt {q r₁ r₂ : ℚ} (hqr : r₁ + r₂ ≠ 0)
    (hval₁ : padicValRat p q < padicValRat p r₁) (hval₂ : padicValRat p q < padicValRat p r₂) :
    padicValRat p q < padicValRat p (r₁ + r₂) :=
  lt_of_lt_of_le (lt_min hval₁ hval₂) (padicValRat.min_le_padicValRat_add hqr)


@[simp]
lemma self_pow_inv (r : ℕ) : padicValRat p ((p : ℚ) ^ r)⁻¹ = -r := by
  rw [padicValRat.inv, neg_inj, padicValRat.pow (Nat.cast_ne_zero.mpr hp.elim.ne_zero),
      padicValRat.self hp.elim.one_lt, mul_one]


/-- A finite sum of rationals with positive `p`-adic valuation has positive `p`-adic valuation
(if the sum is non-zero). -/
theorem sum_pos_of_pos {n : ℕ} {F : ℕ → ℚ} (hF : ∀ i, i < n → 0 < padicValRat p (F i))
    (hn0 : ∑ i ∈ Finset.range n, F i ≠ 0) : 0 < padicValRat p (∑ i ∈ Finset.range n, F i) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    F : Nat → Rat
    hF : ∀ (i : Nat), LT.lt i n → LT.lt 0 (padicValRat p (F i))
    hn0 : Ne ((Finset.range n).sum fun i => F i) 0
    ⊢ LT.lt 0 (padicValRat p ((Finset.range n).sum fun i => F i))
  -/
  induction' n with d hd
    /-
      case zero
      p : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      hF : ∀ (i : Nat), LT.lt i 0 → LT.lt 0 (padicValRat p (F i))
      hn0 : Ne ((Finset.range 0).sum fun i => F i) 0
      ⊢ LT.lt 0 (padicValRat p ((Finset.range 0).sum fun i => F i))
    -/
  · exact False.elim (hn0 rfl)
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      d : Nat
      hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
      hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
      hn0 : Ne ((Finset.range (HAdd.hAdd d 1)).sum fun i => F i) 0
      ⊢ LT.lt 0 (padicValRat p ((Finset.range (HAdd.hAdd d 1)).sum fun i => F i))
    -/
  · rw [Finset.sum_range_succ] at hn0 ⊢
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      d : Nat
      hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
      hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
      hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
      ⊢ LT.lt 0 (padicValRat p (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)))
    -/
    by_cases h : ∑ x ∈ Finset.range d, F x = 0
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        F : Nat → Rat
        d : Nat
        hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
        hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
        hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
        h : Eq ((Finset.range d).sum fun x => F x) 0
        ⊢ LT.lt 0 (padicValRat p (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)))
      -/
    · rw [h, zero_add]
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        F : Nat → Rat
        d : Nat
        hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
        hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
        hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
        h : Eq ((Finset.range d).sum fun x => F x) 0
        ⊢ LT.lt 0 (padicValRat p (F d))
      -/
      exact hF d (lt_add_one _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        F : Nat → Rat
        d : Nat
        hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
        hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
        hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
        h : Not (Eq ((Finset.range d).sum fun x => F x) 0)
        ⊢ LT.lt 0 (padicValRat p (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)))
      -/
    · refine lt_of_lt_of_le ?_ (min_le_padicValRat_add hn0)
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        F : Nat → Rat
        d : Nat
        hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
        hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
        hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
        h : Not (Eq ((Finset.range d).sum fun x => F x) 0)
        ⊢ LT.lt 0 (Min.min (padicValRat p ((Finset.range d).sum fun x => F x)) (padicV …
      -/
      refine lt_min (hd (fun i hi => ?_) h) (hF d (lt_add_one _))
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        F : Nat → Rat
        d : Nat
        hd : (∀ (i : Nat), LT.lt i d → LT.lt 0 (padicValRat p (F i))) → Ne ((Finset.ra …
        hF : ∀ (i : Nat), LT.lt i (HAdd.hAdd d 1) → LT.lt 0 (padicValRat p (F i))
        hn0 : Ne (HAdd.hAdd ((Finset.range d).sum fun x => F x) (F d)) 0
        h : Not (Eq ((Finset.range d).sum fun x => F x) 0)
        i : Nat
        hi : LT.lt i d
        ⊢ LT.lt 0 (padicValRat p (F i))
      -/
      exact hF _ (lt_trans hi (lt_add_one _))
      /-
        🎉 no goals
      -/


/-- If the p-adic valuation of a finite set of positive rationals is greater than a given rational
number, then the p-adic valuation of their sum is also greater than the same rational number. -/
theorem lt_sum_of_lt {p j : ℕ} [hp : Fact (Nat.Prime p)] {F : ℕ → ℚ} {S : Finset ℕ}
    (hS : S.Nonempty) (hF : ∀ i, i ∈ S → padicValRat p (F j) < padicValRat p (F i))
    (hn1 : ∀ i : ℕ, 0 < F i) : padicValRat p (F j) < padicValRat p (∑ i ∈ S, F i) := by
  /-
    p j : Nat
    hp : Fact (Nat.Prime p)
    F : Nat → Rat
    S : Finset Nat
    hS : S.Nonempty
    hF : ∀ (i : Nat), Membership.mem S i → LT.lt (padicValRat p (F j)) (padicValRa …
    hn1 : ∀ (i : Nat), LT.lt 0 (F i)
    ⊢ LT.lt (padicValRat p (F j)) (padicValRat p (S.sum fun i => F i))
  -/
  induction' hS using Finset.Nonempty.cons_induction with k s S' Hnot Hne Hind
    /-
      case singleton
      p j : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      S : Finset Nat
      hn1 : ∀ (i : Nat), LT.lt 0 (F i)
      k : Nat
      hF : ∀ (i : Nat), Membership.mem (Singleton.singleton k) i → LT.lt (padicValRa …
      ⊢ LT.lt (padicValRat p (F j)) (padicValRat p ((Singleton.singleton k).sum fun  …
    -/
  · rw [Finset.sum_singleton]
    /-
      case singleton
      p j : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      S : Finset Nat
      hn1 : ∀ (i : Nat), LT.lt 0 (F i)
      k : Nat
      hF : ∀ (i : Nat), Membership.mem (Singleton.singleton k) i → LT.lt (padicValRa …
      ⊢ LT.lt (padicValRat p (F j)) (padicValRat p (F k))
    -/
    exact hF k (by simp)
    /-
      🎉 no goals
    -/
    /-
      case cons
      p j : Nat
      hp : Fact (Nat.Prime p)
      F : Nat → Rat
      S : Finset Nat
      hn1 : ∀ (i : Nat), LT.lt 0 (F i)
      s : Nat
      S' : Finset Nat
      Hnot : Not (Membership.mem S' s)
      Hne : S'.Nonempty
      Hind : (∀ (i : Nat), Membership.mem S' i → LT.lt (padicValRat p (F j)) (padicV …
      hF : ∀ (i : Nat), Membership.mem (Finset.cons s S' Hnot) i → LT.lt (padicValRa …
      ⊢ LT.lt (padicValRat p (F j)) (padicValRat p ((Finset.cons s S' Hnot).sum fun  …
    -/
  · rw [Finset.cons_eq_insert, Finset.sum_insert Hnot]
    exact padicValRat.lt_add_of_lt
      (ne_of_gt (add_pos (hn1 s) (Finset.sum_pos (fun i _ => hn1 i) Hne)))
      (hF _ (by simp [Finset.mem_insert, true_or]))
      (Hind (fun i hi => hF _ (by rw [Finset.cons_eq_insert,Finset.mem_insert]; exact Or.inr hi)))


/-- A rewrite lemma for `padicValNat p (a * b)` with conditions `a ≠ 0`, `b ≠ 0`. -/
protected theorem mul : a ≠ 0 → b ≠ 0 → padicValNat p (a * b) = padicValNat p a + padicValNat p b :=
  mod_cast padicValRat.mul (p := p) (q := a) (r := b)


protected theorem div_of_dvd (h : b ∣ a) :
    padicValNat p (a / b) = padicValNat p a - padicValNat p b := by
  /-
    p a b : Nat
    hp : Fact (Nat.Prime p)
    h : Dvd.dvd b a
    ⊢ Eq (padicValNat p (HDiv.hDiv a b)) (HSub.hSub (padicValNat p a) (padicValNat …
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      p b : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd b 0
      ⊢ Eq (padicValNat p (HDiv.hDiv 0 b)) (HSub.hSub (padicValNat p 0) (padicValNat …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p a b : Nat
    hp : Fact (Nat.Prime p)
    h : Dvd.dvd b a
    ha : Ne a 0
    ⊢ Eq (padicValNat p (HDiv.hDiv a b)) (HSub.hSub (padicValNat p a) (padicValNat …
  -/
  obtain ⟨k, rfl⟩ := h
  /-
    case inr.intro
    p b : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ha : Ne (HMul.hMul b k) 0
    ⊢ Eq (padicValNat p (HDiv.hDiv (HMul.hMul b k) b)) (HSub.hSub (padicValNat p ( …
  -/
  obtain ⟨hb, hk⟩ := mul_ne_zero_iff.mp ha
  /-
    case inr.intro.intro
    p b : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ha : Ne (HMul.hMul b k) 0
    hb : Ne b 0
    hk : Ne k 0
    ⊢ Eq (padicValNat p (HDiv.hDiv (HMul.hMul b k) b)) (HSub.hSub (padicValNat p ( …
  -/
  rw [mul_comm, k.mul_div_cancel hb.bot_lt, padicValNat.mul hk hb, Nat.add_sub_cancel]
  /-
    🎉 no goals
  -/


/-- Dividing out by a prime factor reduces the `padicValNat` by `1`. -/
protected theorem div (dvd : p ∣ b) : padicValNat p (b / p) = padicValNat p b - 1 := by
  /-
    p b : Nat
    hp : Fact (Nat.Prime p)
    dvd : Dvd.dvd p b
    ⊢ Eq (padicValNat p (HDiv.hDiv b p)) (HSub.hSub (padicValNat p b) 1)
  -/
  rw [padicValNat.div_of_dvd dvd, padicValNat_self]
  /-
    🎉 no goals
  -/


/-- A version of `padicValRat.pow` for `padicValNat`. -/
protected theorem pow (n : ℕ) (ha : a ≠ 0) : padicValNat p (a ^ n) = n * padicValNat p a := by
  /-
    p a : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ha : Ne a 0
    ⊢ Eq (padicValNat p (HPow.hPow a n)) (HMul.hMul n (padicValNat p a))
  -/
  simpa only [← @Nat.cast_inj ℤ, push_cast] using padicValRat.pow (Nat.cast_ne_zero.mpr ha)
  /-
    🎉 no goals
  -/


@[simp]
protected theorem prime_pow (n : ℕ) : padicValNat p (p ^ n) = n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (padicValNat p (HPow.hPow p n)) n
  -/
  rw [padicValNat.pow _ (@Fact.out p.Prime).ne_zero, padicValNat_self, mul_one]
  /-
    🎉 no goals
  -/


protected theorem div_pow (dvd : p ^ a ∣ b) : padicValNat p (b / p ^ a) = padicValNat p b - a := by
  /-
    p a b : Nat
    hp : Fact (Nat.Prime p)
    dvd : Dvd.dvd (HPow.hPow p a) b
    ⊢ Eq (padicValNat p (HDiv.hDiv b (HPow.hPow p a))) (HSub.hSub (padicValNat p b …
  -/
  rw [padicValNat.div_of_dvd dvd, padicValNat.prime_pow]
  /-
    🎉 no goals
  -/


protected theorem div' {m : ℕ} (cpm : Coprime p m) {b : ℕ} (dvd : m ∣ b) :
    padicValNat p (b / m) = padicValNat p b := by
  rw [padicValNat.div_of_dvd dvd, eq_zero_of_not_dvd (hp.out.coprime_iff_not_dvd.mp cpm),
    Nat.sub_zero]


theorem dvd_of_one_le_padicValNat {n : ℕ} (hp : 1 ≤ padicValNat p n) : p ∣ n := by
  /-
    p n : Nat
    hp : LE.le 1 (padicValNat p n)
    ⊢ Dvd.dvd p n
  -/
  by_contra h
  /-
    p n : Nat
    hp : LE.le 1 (padicValNat p n)
    h : Not (Dvd.dvd p n)
    ⊢ False
  -/
  rw [padicValNat.eq_zero_of_not_dvd h] at hp
  /-
    p n : Nat
    hp : LE.le 1 0
    h : Not (Dvd.dvd p n)
    ⊢ False
  -/
  exact lt_irrefl 0 (lt_of_lt_of_le zero_lt_one hp)
  /-
    🎉 no goals
  -/


theorem pow_padicValNat_dvd {n : ℕ} : p ^ padicValNat p n ∣ n := by
  /-
    p n : Nat
    ⊢ Dvd.dvd (HPow.hPow p (padicValNat p n)) n
  -/
  rcases n.eq_zero_or_pos with (rfl | hn); · simp
                                             /-
                                               🎉 no goals
                                             -/
  /-
    case inr
    p n : Nat
    hn : GT.gt n 0
    ⊢ Dvd.dvd (HPow.hPow p (padicValNat p n)) n
  -/
  rcases eq_or_ne p 1 with (rfl | hp); · simp
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr.inr
    p n : Nat
    hn : GT.gt n 0
    hp : Ne p 1
    ⊢ Dvd.dvd (HPow.hPow p (padicValNat p n)) n
  -/
  apply pow_dvd_of_le_multiplicity
  /-
    case inr.inr.hk
    p n : Nat
    hn : GT.gt n 0
    hp : Ne p 1
    ⊢ LE.le (padicValNat p n) (multiplicity p n)
  -/
                            /-
                              🎉 no goals
                            -/
  rw [padicValNat_def'] <;> assumption
                            /-
                              🎉 no goals
                            -/


theorem padicValNat_dvd_iff_le [hp : Fact p.Prime] {a n : ℕ} (ha : a ≠ 0) :
    p ^ n ∣ a ↔ n ≤ padicValNat p a := by
  rw [pow_dvd_iff_le_emultiplicity, ← padicValNat_eq_emultiplicity (Nat.pos_of_ne_zero ha),
    Nat.cast_le]


theorem padicValNat_dvd_iff (n : ℕ) [hp : Fact p.Prime] (a : ℕ) :
    p ^ n ∣ a ↔ a = 0 ∨ n ≤ padicValNat p a := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    a : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow p n) a) (Or (Eq a 0) (LE.le n (padicValNat p a)))
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      p n : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Iff (Dvd.dvd (HPow.hPow p n) 0) (Or (Eq 0 0) (LE.le n (padicValNat p 0)))
    -/
  · exact iff_of_true (dvd_zero _) (Or.inl rfl)
    /-
      🎉 no goals
    -/
    /-
      case inr
      p n : Nat
      hp : Fact (Nat.Prime p)
      a : Nat
      ha : Ne a 0
      ⊢ Iff (Dvd.dvd (HPow.hPow p n) a) (Or (Eq a 0) (LE.le n (padicValNat p a)))
    -/
  · rw [padicValNat_dvd_iff_le ha, or_iff_right ha]
    /-
      🎉 no goals
    -/


theorem pow_succ_padicValNat_not_dvd {n : ℕ} [hp : Fact p.Prime] (hn : n ≠ 0) :
    ¬p ^ (padicValNat p n + 1) ∣ n := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (padicValNat p n) 1)) n)
  -/
  rw [padicValNat_dvd_iff_le hn, not_le]
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    ⊢ LT.lt (padicValNat p n) (HAdd.hAdd (padicValNat p n) 1)
  -/
  exact Nat.lt_succ_self _
  /-
    🎉 no goals
  -/


theorem padicValNat_primes {q : ℕ} [hp : Fact p.Prime] [hq : Fact q.Prime] (neq : p ≠ q) :
    padicValNat p q = 0 :=
  @padicValNat.eq_zero_of_not_dvd p q <|
    (not_congr (Iff.symm (prime_dvd_prime_iff_eq hp.1 hq.1))).mp neq


theorem padicValNat_prime_prime_pow {q : ℕ} [hp : Fact p.Prime] [hq : Fact q.Prime]
    (n : ℕ) (neq : p ≠ q) : padicValNat p (q ^ n) = 0 := by
  /-
    p q : Nat
    hp : Fact (Nat.Prime p)
    hq : Fact (Nat.Prime q)
    n : Nat
    neq : Ne p q
    ⊢ Eq (padicValNat p (HPow.hPow q n)) 0
  -/
  rw [padicValNat.pow _ <| Nat.Prime.ne_zero hq.elim, padicValNat_primes neq, mul_zero]
  /-
    🎉 no goals
  -/


theorem padicValNat_mul_pow_left {q : ℕ} [hp : Fact p.Prime] [hq : Fact q.Prime]
    (n m : ℕ) (neq : p ≠ q) : padicValNat p (p^n * q^m) = n := by
  rw [padicValNat.mul (NeZero.ne' (p^n)).symm (NeZero.ne' (q^m)).symm,
    padicValNat.prime_pow, padicValNat_prime_prime_pow m neq, add_zero]


theorem padicValNat_mul_pow_right {q : ℕ} [hp : Fact p.Prime] [hq : Fact q.Prime]
    (n m : ℕ) (neq : q ≠ p) : padicValNat q (p^n * q^m) = m := by
  /-
    p q : Nat
    hp : Fact (Nat.Prime p)
    hq : Fact (Nat.Prime q)
    n m : Nat
    neq : Ne q p
    ⊢ Eq (padicValNat q (HMul.hMul (HPow.hPow p n) (HPow.hPow q m))) m
  -/
  rw [mul_comm (p^n) (q^m)]
  /-
    p q : Nat
    hp : Fact (Nat.Prime p)
    hq : Fact (Nat.Prime q)
    n m : Nat
    neq : Ne q p
    ⊢ Eq (padicValNat q (HMul.hMul (HPow.hPow q m) (HPow.hPow p n))) m
  -/
  exact padicValNat_mul_pow_left m n neq
  /-
    🎉 no goals
  -/


/-- The p-adic valuation of `n` is less than or equal to its logarithm w.r.t `p`. -/
lemma padicValNat_le_nat_log (n : ℕ) : padicValNat p n ≤ Nat.log p n := by
  /-
    p n : Nat
    ⊢ LE.le (padicValNat p n) (Nat.log p n)
  -/
  rcases n with _ | n
    /-
      case zero
      p : Nat
      ⊢ LE.le (padicValNat p 0) (Nat.log p 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    p n : Nat
    ⊢ LE.le (padicValNat p (HAdd.hAdd n 1)) (Nat.log p (HAdd.hAdd n 1))
  -/
  rcases p with _ | _ | p
    /-
      case succ.zero
      n : Nat
      ⊢ LE.le (padicValNat 0 (HAdd.hAdd n 1)) (Nat.log 0 (HAdd.hAdd n 1))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ.succ.zero
      n : Nat
      ⊢ LE.le (padicValNat (HAdd.hAdd 0 1) (HAdd.hAdd n 1)) (Nat.log (HAdd.hAdd 0 1) …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ.succ.succ
    n p : Nat
    ⊢ LE.le (padicValNat (HAdd.hAdd (HAdd.hAdd p 1) 1) (HAdd.hAdd n 1)) (Nat.log ( …
  -/
  exact Nat.le_log_of_pow_le p.one_lt_succ_succ (le_of_dvd n.succ_pos pow_padicValNat_dvd)
  /-
    🎉 no goals
  -/


/-- The p-adic valuation of `n` is equal to the logarithm w.r.t `p` iff
    `n` is less than `p` raised to one plus the p-adic valuation of `n`. -/
lemma nat_log_eq_padicValNat_iff {n : ℕ} [hp : Fact (Nat.Prime p)] (hn : 0 < n) :
    Nat.log p n = padicValNat p n ↔ n < p ^ (padicValNat p n + 1) := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : LT.lt 0 n
    ⊢ Iff (Eq (Nat.log p n) (padicValNat p n)) (LT.lt n (HPow.hPow p (HAdd.hAdd (p …
  -/
  rw [Nat.log_eq_iff (Or.inr ⟨(Nat.Prime.one_lt' p).out, by omega⟩), and_iff_right_iff_imp]
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : LT.lt 0 n
    ⊢ LT.lt n (HPow.hPow p (HAdd.hAdd (padicValNat p n) 1)) → LE.le (HPow.hPow p ( …
  -/
  exact fun _ => Nat.le_of_dvd hn pow_padicValNat_dvd
  /-
    🎉 no goals
  -/


lemma Nat.log_ne_padicValNat_succ {n : ℕ} (hn : n ≠ 0) : log 2 n ≠ padicValNat 2 (n + 1) := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Ne (Nat.log 2 n) (padicValNat 2 (HAdd.hAdd n 1))
  -/
  rw [Ne, log_eq_iff (by simp [hn])]
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Not (And (LE.le (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) n) (LT.lt n (H …
  -/
  rintro ⟨h1, h2⟩
  /-
    case intro
    n : Nat
    hn : Ne n 0
    h1 : LE.le (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) n
    h2 : LT.lt n (HPow.hPow 2 (HAdd.hAdd (padicValNat 2 (HAdd.hAdd n 1)) 1))
    ⊢ False
  -/
  rw [← Nat.lt_add_one_iff, ← mul_one (2 ^ _)] at h1
  /-
    case intro
    n : Nat
    hn : Ne n 0
    h1 : LT.lt (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) 1) (HAdd.h …
    h2 : LT.lt n (HPow.hPow 2 (HAdd.hAdd (padicValNat 2 (HAdd.hAdd n 1)) 1))
    ⊢ False
  -/
  rw [← add_one_le_iff, Nat.pow_succ] at h2
  /-
    case intro
    n : Nat
    hn : Ne n 0
    h1 : LT.lt (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) 1) (HAdd.h …
    h2 : LE.le (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n …
    ⊢ False
  -/
  refine not_dvd_of_between_consec_multiples h1 (lt_of_le_of_ne' h2 ?_) pow_padicValNat_dvd
  -- TODO(kmill): Why is this `p := 2` necessary?
  /-
    case intro
    n : Nat
    hn : Ne n 0
    h1 : LT.lt (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) 1) (HAdd.h …
    h2 : LE.le (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n …
    ⊢ Ne (HMul.hMul (HPow.hPow 2 (padicValNat 2 (HAdd.hAdd n 1))) (HAdd.hAdd 1 1)) …
  -/
  exact pow_succ_padicValNat_not_dvd (p := 2) n.succ_ne_zero ∘ dvd_of_eq
  /-
    🎉 no goals
  -/


lemma Nat.max_log_padicValNat_succ_eq_log_succ (n : ℕ) :
    max (log 2 n) (padicValNat 2 (n + 1)) = log 2 (n + 1) := by
  apply le_antisymm (max_le (le_log_of_pow_le one_lt_two (pow_log_le_add_one 2 n))
    (padicValNat_le_nat_log (n + 1)))
  /-
    n : Nat
    ⊢ LE.le (Nat.log 2 (HAdd.hAdd n 1)) (Max.max (Nat.log 2 n) (padicValNat 2 (HAd …
  -/
  rw [le_max_iff, or_iff_not_imp_left, not_le]
  /-
    n : Nat
    ⊢ LT.lt (Nat.log 2 n) (Nat.log 2 (HAdd.hAdd n 1)) → LE.le (Nat.log 2 (HAdd.hAd …
  -/
  intro h
  replace h := le_antisymm (add_one_le_iff.mpr (lt_pow_of_log_lt one_lt_two h))
    (pow_log_le_self 2 n.succ_ne_zero)
  /-
    n : Nat
    h : Eq (HAdd.hAdd n 1) (HPow.hPow 2 (Nat.log 2 (HAdd.hAdd n 1)))
    ⊢ LE.le (Nat.log 2 (HAdd.hAdd n 1)) (padicValNat 2 (HAdd.hAdd n 1))
  -/
  rw [h, padicValNat.prime_pow, ← h]
  /-
    🎉 no goals
  -/


theorem range_pow_padicValNat_subset_divisors {n : ℕ} (hn : n ≠ 0) :
    (Finset.range (padicValNat p n + 1)).image (p ^ ·) ⊆ n.divisors := by
  /-
    p n : Nat
    hn : Ne n 0
    ⊢ HasSubset.Subset (Finset.image (fun x => HPow.hPow p x) (Finset.range (HAdd. …
  -/
  intro t ht
  /-
    p n : Nat
    hn : Ne n 0
    t : Nat
    ht : Membership.mem (Finset.image (fun x => HPow.hPow p x) (Finset.range (HAdd …
    ⊢ Membership.mem n.divisors t
  -/
  simp only [exists_prop, Finset.mem_image, Finset.mem_range] at ht
  /-
    p n : Nat
    hn : Ne n 0
    t : Nat
    ht : Exists fun a => And (LT.lt a (HAdd.hAdd (padicValNat p n) 1)) (Eq (HPow.h …
    ⊢ Membership.mem n.divisors t
  -/
  obtain ⟨k, hk, rfl⟩ := ht
  /-
    case intro.intro
    p n : Nat
    hn : Ne n 0
    k : Nat
    hk : LT.lt k (HAdd.hAdd (padicValNat p n) 1)
    ⊢ Membership.mem n.divisors (HPow.hPow p k)
  -/
  rw [Nat.mem_divisors]
  /-
    case intro.intro
    p n : Nat
    hn : Ne n 0
    k : Nat
    hk : LT.lt k (HAdd.hAdd (padicValNat p n) 1)
    ⊢ And (Dvd.dvd (HPow.hPow p k) n) (Ne n 0)
  -/
  exact ⟨(pow_dvd_pow p <| by omega).trans pow_padicValNat_dvd, hn⟩
  /-
    🎉 no goals
  -/


theorem range_pow_padicValNat_subset_divisors' {n : ℕ} [hp : Fact p.Prime] :
    ((Finset.range (padicValNat p n)).image fun t => p ^ (t + 1)) ⊆ n.divisors.erase 1 := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ HasSubset.Subset (Finset.image (fun t => HPow.hPow p (HAdd.hAdd t 1)) (Finse …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ HasSubset.Subset (Finset.image (fun t => HPow.hPow p (HAdd.hAdd t 1)) (Finse …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    ⊢ HasSubset.Subset (Finset.image (fun t => HPow.hPow p (HAdd.hAdd t 1)) (Finse …
  -/
  intro t ht
  /-
    case inr
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    t : Nat
    ht : Membership.mem (Finset.image (fun t => HPow.hPow p (HAdd.hAdd t 1)) (Fins …
    ⊢ Membership.mem (n.divisors.erase 1) t
  -/
  simp only [exists_prop, Finset.mem_image, Finset.mem_range] at ht
  /-
    case inr
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    t : Nat
    ht : Exists fun a => And (LT.lt a (padicValNat p n)) (Eq (HPow.hPow p (HAdd.hA …
    ⊢ Membership.mem (n.divisors.erase 1) t
  -/
  obtain ⟨k, hk, rfl⟩ := ht
  /-
    case inr.intro.intro
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    k : Nat
    hk : LT.lt k (padicValNat p n)
    ⊢ Membership.mem (n.divisors.erase 1) (HPow.hPow p (HAdd.hAdd k 1))
  -/
  rw [Finset.mem_erase, Nat.mem_divisors]
  /-
    case inr.intro.intro
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    k : Nat
    hk : LT.lt k (padicValNat p n)
    ⊢ And (Ne (HPow.hPow p (HAdd.hAdd k 1)) 1) (And (Dvd.dvd (HPow.hPow p (HAdd.hA …
  -/
  refine ⟨?_, (pow_dvd_pow p <| succ_le_iff.2 hk).trans pow_padicValNat_dvd, hn⟩
  /-
    case inr.intro.intro
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Ne n 0
    k : Nat
    hk : LT.lt k (padicValNat p n)
    ⊢ Ne (HPow.hPow p (HAdd.hAdd k 1)) 1
  -/
  exact (Nat.one_lt_pow k.succ_ne_zero hp.out.one_lt).ne'
  /-
    🎉 no goals
  -/


/-- The `p`-adic valuation of `(p * n)!` is `n` more than that of `n!`. -/
theorem padicValNat_factorial_mul (n : ℕ) [hp : Fact p.Prime] :
    padicValNat p (p * n) ! = padicValNat p n ! + n := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (padicValNat p (HMul.hMul p n).factorial) (HAdd.hAdd (padicValNat p n.fac …
  -/
  apply Nat.cast_injective (R := ℕ∞)
  rw [padicValNat_eq_emultiplicity <| factorial_pos (p * n), Nat.cast_add,
      padicValNat_eq_emultiplicity <| factorial_pos n]
  /-
    case a
    p n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (emultiplicity p (HMul.hMul p n).factorial) (HAdd.hAdd (emultiplicity p n …
  -/
  exact Prime.emultiplicity_factorial_mul hp.out
  /-
    🎉 no goals
  -/


/-- The `p`-adic valuation of `m` equals zero if it is between `p * k` and `p * (k + 1)` for
some `k`. -/
theorem padicValNat_eq_zero_of_mem_Ioo {m k : ℕ}
    (hm : m ∈ Set.Ioo (p * k) (p * (k + 1))) : padicValNat p m = 0 :=
  padicValNat.eq_zero_of_not_dvd <| not_dvd_of_between_consec_multiples hm.1 hm.2


theorem padicValNat_factorial_mul_add {n : ℕ} (m : ℕ) [hp : Fact p.Prime] (h : n < p) :
    padicValNat p (p * m + n) ! = padicValNat p (p * m) ! := by
  induction n with
  | zero => rw [add_zero]
  | succ n hn =>
    rw [add_succ, factorial_succ,
      padicValNat.mul (succ_ne_zero (p * m + n)) <| factorial_ne_zero (p * m + _),
      hn <| lt_of_succ_lt h, ← add_succ,
      padicValNat_eq_zero_of_mem_Ioo ⟨(Nat.lt_add_of_pos_right <| succ_pos n),
        (Nat.mul_add _ _ _▸ Nat.mul_one _ ▸ ((add_lt_add_iff_left (p * m)).mpr h))⟩,
      zero_add]


/-- The `p`-adic valuation of `n!` is equal to the `p`-adic valuation of the factorial of the
largest multiple of `p` below `n`, i.e. `(p * ⌊n / p⌋)!`. -/
@[simp] theorem padicValNat_mul_div_factorial (n : ℕ) [hp : Fact p.Prime] :
    padicValNat p (p * (n / p))! = padicValNat p n ! := by
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (padicValNat p (HMul.hMul p (HDiv.hDiv n p)).factorial) (padicValNat p n. …
  -/
  nth_rw 2 [← div_add_mod n p]
  /-
    p n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (padicValNat p (HMul.hMul p (HDiv.hDiv n p)).factorial) (padicValNat p (H …
  -/
  exact (padicValNat_factorial_mul_add (n / p) <| mod_lt n hp.out.pos).symm
  /-
    🎉 no goals
  -/


/-- **Legendre's Theorem**

The `p`-adic valuation of `n!` is the sum of the quotients `n / p ^ i`. This sum is expressed
over the finset `Ico 1 b` where `b` is any bound greater than `log p n`. -/
theorem padicValNat_factorial {n b : ℕ} [hp : Fact p.Prime] (hnb : log p n < b) :
    padicValNat p (n !) = ∑ i ∈ Finset.Ico 1 b, n / p ^ i := by
  exact_mod_cast ((padicValNat_eq_emultiplicity (p := p) <| factorial_pos _) ▸
      Prime.emultiplicity_factorial hp.out hnb)


/-- **Legendre's Theorem**

Taking (`p - 1`) times the `p`-adic valuation of `n!` equals `n` minus the sum of base `p` digits
of `n`. -/
theorem sub_one_mul_padicValNat_factorial [hp : Fact p.Prime] (n : ℕ) :
    (p - 1) * padicValNat p (n !) = n - (p.digits n).sum := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) (padicValNat p n.factorial)) (HSub.hSub n (p.d …
  -/
  rw [padicValNat_factorial <| lt_succ_of_lt <| lt.base (log p n)]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.Ico 1 (Nat.log p n).succ.succ).sum fu …
  -/
  nth_rw 2 [← zero_add 1]
  rw [Nat.succ_eq_add_one, ← Finset.sum_Ico_add' _ 0 _ 1,
    Ico_zero_eq_range, ← sub_one_mul_sum_log_div_pow_eq_sub_sum_digits, Nat.succ_eq_add_one]


/-- **Kummer's Theorem**

The `p`-adic valuation of `n.choose k` is the number of carries when `k` and `n - k` are added
in base `p`. This sum is expressed over the finset `Ico 1 b` where `b` is any bound greater than
`log p n`. -/
theorem padicValNat_choose {n k b : ℕ} [hp : Fact p.Prime] (hkn : k ≤ n) (hnb : log p n < b) :
    padicValNat p (choose n k) =
    ((Finset.Ico 1 b).filter fun i => p ^ i ≤ k % p ^ i + (n - k) % p ^ i).card := by
  exact_mod_cast (padicValNat_eq_emultiplicity (p := p) <| choose_pos hkn) ▸
    Prime.emultiplicity_choose hp.out hkn hnb


/-- **Kummer's Theorem**

The `p`-adic valuation of `(n + k).choose k` is the number of carries when `k` and `n` are added
in base `p`. This sum is expressed over the finset `Ico 1 b` where `b` is any bound greater than
`log p (n + k)`. -/
theorem padicValNat_choose' {n k b : ℕ} [hp : Fact p.Prime] (hnb : log p (n + k) < b) :
    padicValNat p (choose (n + k) k) =
    ((Finset.Ico 1 b).filter fun i => p ^ i ≤ k % p ^ i + n % p ^ i).card := by
  exact_mod_cast (padicValNat_eq_emultiplicity (p := p) <| choose_pos <|
    Nat.le_add_left k n)▸ Prime.emultiplicity_choose' hp.out hnb


/-- **Kummer's Theorem**
Taking (`p - 1`) times the `p`-adic valuation of the binomial `n + k` over `k` equals the sum of the
digits of `k` plus the sum of the digits of `n` minus the sum of digits of `n + k`, all base `p`.
-/
theorem sub_one_mul_padicValNat_choose_eq_sub_sum_digits' {k n : ℕ} [hp : Fact p.Prime] :
    (p - 1) * padicValNat p (choose (n + k) k) =
    (p.digits k).sum + (p.digits n).sum - (p.digits (n + k)).sum := by
  /-
    p k n : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) (padicValNat p ((HAdd.hAdd n k).choose k))) (H …
  -/
  have h : k ≤ n + k := by exact Nat.le_add_left k n
  /-
    p k n : Nat
    hp : Fact (Nat.Prime p)
    h : LE.le k (HAdd.hAdd n k)
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) (padicValNat p ((HAdd.hAdd n k).choose k))) (H …
  -/
  simp only [Nat.choose_eq_factorial_div_factorial h]
  rw [padicValNat.div_of_dvd <| factorial_mul_factorial_dvd_factorial h, Nat.mul_sub_left_distrib,
      padicValNat.mul (factorial_ne_zero _) (factorial_ne_zero _), Nat.mul_add]
  /-
    p k n : Nat
    hp : Fact (Nat.Prime p)
    h : LE.le k (HAdd.hAdd n k)
    ⊢ Eq (HSub.hSub (HMul.hMul (HSub.hSub p 1) (padicValNat p (HAdd.hAdd n k).fact …
  -/
  simp only [sub_one_mul_padicValNat_factorial]
  rw [← Nat.sub_add_comm <| digit_sum_le p k, Nat.add_sub_cancel n k, ← Nat.add_sub_assoc <|
      digit_sum_le p n, Nat.sub_sub (k + n), ← Nat.sub_right_comm, Nat.sub_sub, sub_add_eq,
      add_comm, tsub_tsub_assoc (Nat.le_refl (k + n)) <| (add_comm k n) ▸ (Nat.add_le_add
      (digit_sum_le p n) (digit_sum_le p k)), Nat.sub_self (k + n), zero_add, add_comm]


/-- **Kummer's Theorem**
Taking (`p - 1`) times the `p`-adic valuation of the binomial `n` over `k` equals the sum of the
digits of `k` plus the sum of the digits of `n - k` minus the sum of digits of `n`, all base `p`.
-/
theorem sub_one_mul_padicValNat_choose_eq_sub_sum_digits {k n : ℕ} [hp : Fact p.Prime]
    (h : k ≤ n) : (p - 1) * padicValNat p (choose n k) =
    (p.digits k).sum + (p.digits (n - k)).sum - (p.digits n).sum := by
  /-
    p k n : Nat
    hp : Fact (Nat.Prime p)
    h : LE.le k n
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) (padicValNat p (n.choose k))) (HSub.hSub (HAdd …
  -/
  convert @sub_one_mul_padicValNat_choose_eq_sub_sum_digits' _ _ _ ‹_›
  /-
    case h.e'_2.h.e'_6.h.e'_2.h.e'_1
    p k n : Nat
    hp : Fact (Nat.Prime p)
    h : LE.le k n
    ⊢ Eq n (HAdd.hAdd (HSub.hSub n k) k)
  -/
  all_goals omega
  /-
    🎉 no goals
  -/


theorem padicValInt_dvd_iff (n : ℕ) (a : ℤ) : (p : ℤ) ^ n ∣ a ↔ a = 0 ∨ n ≤ padicValInt p a := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    a : Int
    ⊢ Iff (Dvd.dvd (HPow.hPow (↑p) n) a) (Or (Eq a 0) (LE.le n (padicValInt p a)))
  -/
  rw [padicValInt, ← Int.natAbs_eq_zero, ← padicValNat_dvd_iff, ← Int.natCast_dvd, Int.natCast_pow]
  /-
    🎉 no goals
  -/


theorem padicValInt_dvd (a : ℤ) : (p : ℤ) ^ padicValInt p a ∣ a := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : Int
    ⊢ Dvd.dvd (HPow.hPow (↑p) (padicValInt p a)) a
  -/
  rw [padicValInt_dvd_iff]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : Int
    ⊢ Or (Eq a 0) (LE.le (padicValInt p a) (padicValInt p a))
  -/
  exact Or.inr le_rfl
  /-
    🎉 no goals
  -/


theorem padicValInt_self : padicValInt p p = 1 :=
  padicValInt.self hp.out.one_lt


theorem padicValInt.mul {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) :
    padicValInt p (a * b) = padicValInt p a + padicValInt p b := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a b : Int
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (padicValInt p (HMul.hMul a b)) (HAdd.hAdd (padicValInt p a) (padicValInt …
  -/
  simp_rw [padicValInt]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a b : Int
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (padicValNat p (HMul.hMul a b).natAbs) (HAdd.hAdd (padicValNat p a.natAbs …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  rw [Int.natAbs_mul, padicValNat.mul] <;> rwa [Int.natAbs_ne_zero]
                                           /-
                                             🎉 no goals
                                           -/


theorem padicValInt_mul_eq_succ (a : ℤ) (ha : a ≠ 0) :
    padicValInt p (a * p) = padicValInt p a + 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : Int
    ha : Ne a 0
    ⊢ Eq (padicValInt p (HMul.hMul a ↑p)) (HAdd.hAdd (padicValInt p a) 1)
  -/
  rw [padicValInt.mul ha (Int.natCast_ne_zero.mpr hp.out.ne_zero)]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : Int
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (padicValInt p a) (padicValInt p ↑p)) (HAdd.hAdd (padicValInt  …
  -/
  simp only [eq_self_iff_true, padicValInt.of_nat, padicValNat_self]
  /-
    🎉 no goals
  -/


