/-- Auxiliary function for computing the smallest prime factor of a `PosNum`. Unlike
`Nat.minFacAux`, we use a natural number `fuel` variable that is set to an upper bound on the
number of iterations. It is initialized to the number `n` we are determining primality for. Even
though this is exponential in the input (since it is a `Nat`, not a `Num`), it will get lazily
evaluated during kernel reduction, so we will only require about `sqrt n` unfoldings, for the
`sqrt n` iterations of the loop. -/
def minFacAux (n : PosNum) : ℕ → PosNum → PosNum
  | 0, _ => n
  | fuel + 1, k =>
    if n < k.bit1 * k.bit1 then n else if k.bit1 ∣ n then k.bit1 else minFacAux n fuel k.succ


theorem minFacAux_to_nat {fuel : ℕ} {n k : PosNum} (h : Nat.sqrt n < fuel + k.bit1) :
    (minFacAux n fuel k : ℕ) = Nat.minFacAux n k.bit1 := by
  /-
    fuel : Nat
    n k : PosNum
    h : LT.lt (↑n).sqrt (HAdd.hAdd fuel ↑k.bit1)
    ⊢ Eq (↑(n.minFacAux fuel k)) ((↑n).minFacAux ↑k.bit1)
  -/
  induction' fuel with fuel ih generalizing k <;> rw [minFacAux, Nat.minFacAux]
    /-
      case zero
      n k : PosNum
      h : LT.lt (↑n).sqrt (HAdd.hAdd 0 ↑k.bit1)
      ⊢ Eq (↑n) (ite (LT.lt (↑n) (HMul.hMul ↑k.bit1 ↑k.bit1)) (↑n) (ite (Dvd.dvd ↑k. …
    -/
  · rw [Nat.zero_add, Nat.sqrt_lt] at h
    /-
      case zero
      n k : PosNum
      h : LT.lt (↑n) (HMul.hMul ↑k.bit1 ↑k.bit1)
      ⊢ Eq (↑n) (ite (LT.lt (↑n) (HMul.hMul ↑k.bit1 ↑k.bit1)) (↑n) (ite (Dvd.dvd ↑k. …
    -/
    simp only [h, ite_true]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : PosNum
    fuel : Nat
    ih : ∀ {k : PosNum}, LT.lt (↑n).sqrt (HAdd.hAdd fuel ↑k.bit1) → Eq (↑(n.minFac …
    k : PosNum
    h : LT.lt (↑n).sqrt (HAdd.hAdd (HAdd.hAdd fuel 1) ↑k.bit1)
    ⊢ Eq (↑(ite (LT.lt n (HMul.hMul k.bit1 k.bit1)) n (ite (Dvd.dvd k.bit1 n) k.bi …
  -/
  simp_rw [← mul_to_nat]
  /-
    case succ
    n : PosNum
    fuel : Nat
    ih : ∀ {k : PosNum}, LT.lt (↑n).sqrt (HAdd.hAdd fuel ↑k.bit1) → Eq (↑(n.minFac …
    k : PosNum
    h : LT.lt (↑n).sqrt (HAdd.hAdd (HAdd.hAdd fuel 1) ↑k.bit1)
    ⊢ Eq (↑(ite (LT.lt n (HMul.hMul k.bit1 k.bit1)) n (ite (Dvd.dvd k.bit1 n) k.bi …
  -/
  simp only [cast_lt, dvd_to_nat]
  /-
    case succ
    n : PosNum
    fuel : Nat
    ih : ∀ {k : PosNum}, LT.lt (↑n).sqrt (HAdd.hAdd fuel ↑k.bit1) → Eq (↑(n.minFac …
    k : PosNum
    h : LT.lt (↑n).sqrt (HAdd.hAdd (HAdd.hAdd fuel 1) ↑k.bit1)
    ⊢ Eq (↑(ite (LT.lt n (HMul.hMul k.bit1 k.bit1)) n (ite (Dvd.dvd k.bit1 n) k.bi …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> try rfl
  /-
    case neg
    n : PosNum
    fuel : Nat
    ih : ∀ {k : PosNum}, LT.lt (↑n).sqrt (HAdd.hAdd fuel ↑k.bit1) → Eq (↑(n.minFac …
    k : PosNum
    h : LT.lt (↑n).sqrt (HAdd.hAdd (HAdd.hAdd fuel 1) ↑k.bit1)
    h✝¹ : Not (LT.lt n (HMul.hMul k.bit1 k.bit1))
    h✝ : Not (Dvd.dvd k.bit1 n)
    ⊢ Eq (↑(n.minFacAux fuel k.succ)) ((↑n).minFacAux (HAdd.hAdd (↑k.bit1) 2))
  -/
  rw [ih] <;> [congr; convert Nat.lt_succ_of_lt h using 1] <;>
    simp only [cast_bit1, cast_succ, Nat.succ_eq_add_one, add_assoc,
      add_left_comm, ← one_add_one_eq_two]


/-- Returns the smallest prime factor of `n ≠ 1`. -/
def minFac : PosNum → PosNum
  | 1 => 1
  | bit0 _ => 2
  | bit1 n => minFacAux (bit1 n) n 1


@[simp]
theorem minFac_to_nat (n : PosNum) : (minFac n : ℕ) = Nat.minFac n := by
  /-
    n : PosNum
    ⊢ Eq (↑n.minFac) (↑n).minFac
  -/
  cases' n with n
    /-
      case one
      ⊢ Eq (↑PosNum.one.minFac) (↑PosNum.one).minFac
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case bit1
      n : PosNum
      ⊢ Eq (↑n.bit1.minFac) (↑n.bit1).minFac
    -/
  · rw [minFac, Nat.minFac_eq, if_neg]
    /-
      case bit1
      n : PosNum
      ⊢ Eq (↑(n.bit1.minFacAux (↑n) 1)) ((↑n.bit1).minFacAux 3)
    -/
    swap
      /-
        case bit1.hnc
        n : PosNum
        ⊢ Not (Dvd.dvd 2 ↑n.bit1)
      -/
    · simp [← two_mul]
      /-
        🎉 no goals
      -/
    /-
      case bit1
      n : PosNum
      ⊢ Eq (↑(n.bit1.minFacAux (↑n) 1)) ((↑n.bit1).minFacAux 3)
    -/
    rw [minFacAux_to_nat]
      /-
        case bit1
        n : PosNum
        ⊢ Eq ((↑n.bit1).minFacAux ↑(PosNum.bit1 1)) ((↑n.bit1).minFacAux 3)
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case bit1
      n : PosNum
      ⊢ LT.lt (↑n.bit1).sqrt (HAdd.hAdd ↑n ↑(PosNum.bit1 1))
    -/
    simp only [cast_one, cast_bit1]
    /-
      case bit1
      n : PosNum
      ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd ↑n ↑n) 1).sqrt (HAdd.hAdd (↑n) (HAdd.hAdd (HAdd. …
    -/
    rw [Nat.sqrt_lt]
    calc
      (n : ℕ) + (n : ℕ) + 1 ≤ (n : ℕ) + (n : ℕ) + (n : ℕ) := by simp
      _ = (n : ℕ) * (1 + 1 + 1) := by simp only [mul_add, mul_one]
      _ < _ := by
        set_option simprocs false in simp [mul_lt_mul]
    /-
      case bit0
      a✝ : PosNum
      ⊢ Eq (↑a✝.bit0.minFac) (↑a✝.bit0).minFac
    -/
  · rw [minFac, Nat.minFac_eq, if_pos]
      /-
        case bit0
        a✝ : PosNum
        ⊢ Eq (↑2) 2
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case bit0.hc
      a✝ : PosNum
      ⊢ Dvd.dvd 2 ↑a✝.bit0
    -/
    simp [← two_mul]
    /-
      🎉 no goals
    -/


/-- Primality predicate for a `PosNum`. -/
@[simp]
def Prime (n : PosNum) : Prop :=
  Nat.Prime n


instance decidablePrime : DecidablePred PosNum.Prime
  | 1 => Decidable.isFalse Nat.not_prime_one
  | bit0 n =>
    decidable_of_iff' (n = 1)
      (by
        /-
          n : PosNum
          ⊢ Iff n.bit0.Prime (Eq n 1)
        -/
        refine Nat.prime_def_minFac.trans ((and_iff_right ?_).trans <| eq_comm.trans ?_)
          /-
            case refine_1
            n : PosNum
            ⊢ LE.le 2 ↑n.bit0
          -/
        · exact add_le_add (Nat.succ_le_of_lt (to_nat_pos _)) (Nat.succ_le_of_lt (to_nat_pos _))
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          n : PosNum
          ⊢ Iff (Eq (↑n.bit0) (↑n.bit0).minFac) (Eq n 1)
        -/
        rw [← minFac_to_nat, to_nat_inj]
        /-
          case refine_2
          n : PosNum
          ⊢ Iff (Eq n.bit0 n.bit0.minFac) (Eq n 1)
        -/
        exact ⟨bit0.inj, congr_arg _⟩)
        /-
          🎉 no goals
        -/
  | bit1 n =>
    decidable_of_iff' (minFacAux (bit1 n) n 1 = bit1 n) <| by
        /-
          n : PosNum
          ⊢ Iff n.bit1.Prime (Eq (n.bit1.minFacAux (↑n) 1) n.bit1)
        -/
        refine Nat.prime_def_minFac.trans ((and_iff_right ?_).trans ?_)
          /-
            case refine_1
            n : PosNum
            ⊢ LE.le 2 ↑n.bit1
          -/
        · simp only [cast_bit1]
          /-
            case refine_1
            n : PosNum
            ⊢ LE.le 2 (HAdd.hAdd (HAdd.hAdd ↑n ↑n) 1)
          -/
          have := to_nat_pos n
          /-
            case refine_1
            n : PosNum
            this : LT.lt 0 ↑n
            ⊢ LE.le 2 (HAdd.hAdd (HAdd.hAdd ↑n ↑n) 1)
          -/
          omega
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          n : PosNum
          ⊢ Iff (Eq (↑n.bit1).minFac ↑n.bit1) (Eq (n.bit1.minFacAux (↑n) 1) n.bit1)
        -/
        rw [← minFac_to_nat, to_nat_inj]; rfl
                                          /-
                                            🎉 no goals
                                          -/


/-- Returns the smallest prime factor of `n ≠ 1`. -/
def minFac : Num → PosNum
  | 0 => 2
  | pos n => n.minFac


@[simp]
theorem minFac_to_nat : ∀ n : Num, (minFac n : ℕ) = Nat.minFac n
  | 0 => rfl
  | pos _ => PosNum.minFac_to_nat _


/-- Primality predicate for a `Num`. -/
@[simp]
def Prime (n : Num) : Prop :=
  Nat.Prime n


instance decidablePrime : DecidablePred Num.Prime
  | 0 => Decidable.isFalse Nat.not_prime_zero
  | pos n => PosNum.decidablePrime n


