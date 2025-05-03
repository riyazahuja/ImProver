/-- The numeral `((0+1)+⋯)+1`. -/
protected def Nat.unaryCast [One R] [Zero R] [Add R] : ℕ → R
  | 0 => 0
  | n + 1 => Nat.unaryCast n + 1

-- the following four declarations are not in mathlib3 and are relevant to the way numeric
-- literals are handled in Lean 4.


/-- A type class for natural numbers which are greater than or equal to `2`. -/
class Nat.AtLeastTwo (n : ℕ) : Prop where
  prop : n ≥ 2


instance instNatAtLeastTwo {n : ℕ} : Nat.AtLeastTwo (n + 2) where
  prop := Nat.succ_le_succ <| Nat.succ_le_succ <| Nat.zero_le _


lemma one_lt : 1 < n := prop

lemma ne_one : n ≠ 1 := Nat.ne_of_gt one_lt


/-- Recognize numeric literals which are at least `2` as terms of `R` via `Nat.cast`. This
instance is what makes things like `37 : R` type check.  Note that `0` and `1` are not needed
because they are recognized as terms of `R` (at least when `R` is an `AddMonoidWithOne`) through
`Zero` and `One`, respectively. -/
@[nolint unusedArguments]
instance (priority := 100) instOfNatAtLeastTwo {n : ℕ} [NatCast R] [Nat.AtLeastTwo n] :
    OfNat R n where
  ofNat := n.cast


@[simp, norm_cast] theorem Nat.cast_ofNat {n : ℕ} [NatCast R] [Nat.AtLeastTwo n] :
  (Nat.cast ofNat(n) : R) = ofNat(n) := rfl


@[deprecated Nat.cast_ofNat (since := "2024-12-22")]
theorem Nat.cast_eq_ofNat {n : ℕ} [NatCast R] [Nat.AtLeastTwo n] :
    (Nat.cast n : R) = OfNat.ofNat n :=
  rfl


/-- An `AddMonoidWithOne` is an `AddMonoid` with a `1`.
It also contains data for the unique homomorphism `ℕ → R`. -/
class AddMonoidWithOne (R : Type*) extends NatCast R, AddMonoid R, One R where
  natCast := Nat.unaryCast
  /-- The canonical map `ℕ → R` sends `0 : ℕ` to `0 : R`. -/
  natCast_zero : natCast 0 = 0 := by intros; rfl
  /-- The canonical map `ℕ → R` is a homomorphism. -/
  natCast_succ : ∀ n, natCast (n + 1) = natCast n + 1 := by intros; rfl


/-- An `AddCommMonoidWithOne` is an `AddMonoidWithOne` satisfying `a + b = b + a`. -/
class AddCommMonoidWithOne (R : Type*) extends AddMonoidWithOne R, AddCommMonoid R


@[simp, norm_cast]
theorem cast_zero : ((0 : ℕ) : R) = 0 :=
  AddMonoidWithOne.natCast_zero

-- Lemmas about `Nat.succ` need to get a low priority, so that they are tried last.
-- This is because `Nat.succ _` matches `1`, `3`, `x+1`, etc.
-- Rewriting would then produce really wrong terms.

@[norm_cast 500]
theorem cast_succ (n : ℕ) : ((succ n : ℕ) : R) = n + 1 :=
  AddMonoidWithOne.natCast_succ _


theorem cast_add_one (n : ℕ) : ((n + 1 : ℕ) : R) = n + 1 :=
  cast_succ _


@[simp, norm_cast]
theorem cast_ite (P : Prop) [Decidable P] (m n : ℕ) :
    ((ite P m n : ℕ) : R) = ite P (m : R) (n : R) := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    P : Prop
    inst✝ : Decidable P
    m n : Nat
    ⊢ Eq (↑(ite P m n)) (ite P ↑m ↑n)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


@[simp, norm_cast]
theorem cast_one [AddMonoidWithOne R] : ((1 : ℕ) : R) = 1 := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (↑1) 1
  -/
  rw [cast_succ, Nat.cast_zero, zero_add]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_add [AddMonoidWithOne R] (m n : ℕ) : ((m + n : ℕ) : R) = m + n := by
  induction n with
  | zero => simp
  | succ n ih => rw [add_succ, cast_succ, ih, cast_succ, add_assoc]


/-- Computationally friendlier cast than `Nat.unaryCast`, using binary representation. -/
protected def binCast [Zero R] [One R] [Add R] : ℕ → R
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0
    then (Nat.binCast ((n + 1) / 2)) + (Nat.binCast ((n + 1) / 2))
    else (Nat.binCast ((n + 1) / 2)) + (Nat.binCast ((n + 1) / 2)) + 1


@[simp]
theorem binCast_eq [AddMonoidWithOne R] (n : ℕ) :
    (Nat.binCast n : R) = ((n : ℕ) : R) := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    n : Nat
    ⊢ Eq n.binCast ↑n
  -/
  induction n using Nat.strongRecOn with | ind k hk => ?_
  cases k with
  | zero => rw [Nat.binCast, Nat.cast_zero]
  | succ k =>
      rw [Nat.binCast]
      by_cases h : (k + 1) % 2 = 0
      · conv => rhs; rw [← Nat.mod_add_div (k+1) 2]
        rw [if_pos h, hk _ <| Nat.div_lt_self (Nat.succ_pos k) (Nat.le_refl 2), ← Nat.cast_add]
        rw [h, Nat.zero_add, Nat.succ_mul, Nat.one_mul]
      · conv => rhs; rw [← Nat.mod_add_div (k+1) 2]
        rw [if_neg h, hk _ <| Nat.div_lt_self (Nat.succ_pos k) (Nat.le_refl 2), ← Nat.cast_add]
        have h1 := Or.resolve_left (Nat.mod_two_eq_zero_or_one (succ k)) h
        rw [h1, Nat.add_comm 1, Nat.succ_mul, Nat.one_mul]
        simp only [Nat.cast_add, Nat.cast_one]


theorem cast_two [AddMonoidWithOne R] : ((2 : ℕ) : R) = (2 : R) := rfl


theorem cast_three [AddMonoidWithOne R] : ((3 : ℕ) : R) = (3 : R) := rfl


theorem cast_four [AddMonoidWithOne R] : ((4 : ℕ) : R) = (4 : R) := rfl


/-- `AddMonoidWithOne` implementation using unary recursion. -/
protected abbrev AddMonoidWithOne.unary [AddMonoid R] [One R] : AddMonoidWithOne R :=
  { ‹One R›, ‹AddMonoid R› with }


/-- `AddMonoidWithOne` implementation using binary recursion. -/
protected abbrev AddMonoidWithOne.binary [AddMonoid R] [One R] : AddMonoidWithOne R :=
  { ‹One R›, ‹AddMonoid R› with
    natCast := Nat.binCast,
                       /-
                         R : Type u_1
                         inst✝¹ : AddMonoid R
                         inst✝ : One R
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
    natCast_zero := by simp only [Nat.binCast, Nat.cast],
                       /-
                         🎉 no goals
                       -/
    natCast_succ := fun n => by
      /-
        R : Type u_1
        inst✝¹ : AddMonoid R
        inst✝ : One R
        n : Nat
        ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
      -/
      dsimp only [NatCast.natCast]
      /-
        R : Type u_1
        inst✝¹ : AddMonoid R
        inst✝ : One R
        n : Nat
        ⊢ Eq (HAdd.hAdd n 1).binCast (HAdd.hAdd n.binCast 1)
      -/
      letI : AddMonoidWithOne R := AddMonoidWithOne.unary
      /-
        R : Type u_1
        inst✝¹ : AddMonoid R
        inst✝ : One R
        n : Nat
        this : AddMonoidWithOne R := AddMonoidWithOne.unary
        ⊢ Eq (HAdd.hAdd n 1).binCast (HAdd.hAdd n.binCast 1)
      -/
      rw [Nat.binCast_eq, Nat.binCast_eq, Nat.cast_succ] }
      /-
        🎉 no goals
      -/


theorem one_add_one_eq_two [AddMonoidWithOne R] : 1 + 1 = (2 : R) := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (HAdd.hAdd 1 1) 2
  -/
  rw [← Nat.cast_one, ← Nat.cast_add]
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (↑(HAdd.hAdd 1 1)) 2
  -/
  apply congrArg
  /-
    case h
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (HAdd.hAdd 1 1) 2
  -/
  decide
  /-
    🎉 no goals
  -/


theorem two_add_one_eq_three [AddMonoidWithOne R] : 2 + 1 = (3 : R) := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (HAdd.hAdd 2 1) 3
  -/
  rw [← one_add_one_eq_two, ← Nat.cast_one, ← Nat.cast_add, ← Nat.cast_add]
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (↑(HAdd.hAdd (HAdd.hAdd 1 1) 1)) 3
  -/
  apply congrArg
  /-
    case h
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 1) 1) 3
  -/
  decide
  /-
    🎉 no goals
  -/


theorem three_add_one_eq_four [AddMonoidWithOne R] : 3 + 1 = (4 : R) := by
  rw [← two_add_one_eq_three, ← one_add_one_eq_two, ← Nat.cast_one,
    ← Nat.cast_add, ← Nat.cast_add, ← Nat.cast_add]
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (↑(HAdd.hAdd (HAdd.hAdd (HAdd.hAdd 1 1) 1) 1)) 4
  -/
  apply congrArg
  /-
    case h
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd 1 1) 1) 1) 4
  -/
  decide
  /-
    🎉 no goals
  -/


theorem two_add_two_eq_four [AddMonoidWithOne R] : 2 + 2 = (4 : R) := by
  simp [← one_add_one_eq_two, ← Nat.cast_one, ← three_add_one_eq_four,
    ← two_add_one_eq_three, add_assoc]


@[simp] lemma nsmul_one {A} [AddMonoidWithOne A] : ∀ n : ℕ, n • (1 : A) = n
            /-
              A : Type u_2
              inst✝ : AddMonoidWithOne A
              ⊢ Eq (HSMul.hSMul 0 1) ↑0
            -/
  | 0 => by simp [zero_nsmul]
            /-
              🎉 no goals
            -/
                /-
                  A : Type u_2
                  inst✝ : AddMonoidWithOne A
                  n : Nat
                  ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) 1) ↑(HAdd.hAdd n 1)
                -/
  | n + 1 => by simp [succ_nsmul, nsmul_one n]
                /-
                  🎉 no goals
                -/


