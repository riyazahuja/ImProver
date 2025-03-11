theorem IsSquare.nonneg [Semiring R] [LinearOrder R] [IsRightCancelAdd R]
    [ZeroLEOneClass R] [ExistsAddOfLE R] [PosMulMono R] [AddLeftStrictMono R]
    {x : R} (h : IsSquare x) : 0 ≤ x := by
  /-
    R : Type u_3
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder R
    inst✝⁴ : IsRightCancelAdd R
    inst✝³ : ZeroLEOneClass R
    inst✝² : ExistsAddOfLE R
    inst✝¹ : PosMulMono R
    inst✝ : AddLeftStrictMono R
    x : R
    h : IsSquare x
    ⊢ LE.le 0 x
  -/
  rcases h with ⟨y, rfl⟩
  /-
    case intro
    R : Type u_3
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder R
    inst✝⁴ : IsRightCancelAdd R
    inst✝³ : ZeroLEOneClass R
    inst✝² : ExistsAddOfLE R
    inst✝¹ : PosMulMono R
    inst✝ : AddLeftStrictMono R
    y : R
    ⊢ LE.le 0 (HMul.hMul y y)
  -/
  exact mul_self_nonneg y
  /-
    🎉 no goals
  -/


theorem map_neg_one : f (-1) = 1 :=
                                                /-
                                                  M : Type u_2
                                                  R : Type u_3
                                                  inst✝³ : Ring R
                                                  inst✝² : Monoid M
                                                  inst✝¹ : LinearOrder M
                                                  inst✝ : MulLeftMono M
                                                  f : MonoidHom R M
                                                  ⊢ Eq (HPow.hPow (f (-1)) (Nat.succ 1)) 1
                                                -/
  (pow_eq_one_iff (Nat.succ_ne_zero 1)).1 <| by rw [← map_pow, neg_one_sq, map_one]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                             /-
                                               M : Type u_2
                                               R : Type u_3
                                               inst✝³ : Ring R
                                               inst✝² : Monoid M
                                               inst✝¹ : LinearOrder M
                                               inst✝ : MulLeftMono M
                                               f : MonoidHom R M
                                               x : R
                                               ⊢ Eq (f (Neg.neg x)) (f x)
                                             -/
theorem map_neg (x : R) : f (-x) = f x := by rw [← neg_one_mul, map_mul, map_neg_one, one_mul]
                                             /-
                                               🎉 no goals
                                             -/


                                                             /-
                                                               M : Type u_2
                                                               R : Type u_3
                                                               inst✝³ : Ring R
                                                               inst✝² : Monoid M
                                                               inst✝¹ : LinearOrder M
                                                               inst✝ : MulLeftMono M
                                                               f : MonoidHom R M
                                                               x y : R
                                                               ⊢ Eq (f (HSub.hSub x y)) (f (HSub.hSub y x))
                                                             -/
theorem map_sub_swap (x y : R) : f (x - y) = f (y - x) := by rw [← map_neg, neg_sub]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem pow_add_pow_le (hx : 0 ≤ x) (hy : 0 ≤ y) (hn : n ≠ 0) : x ^ n + y ^ n ≤ (x + y) ^ n := by
  /-
    R : Type u_3
    inst✝ : OrderedSemiring R
    x y : R
    n : Nat
    hx : LE.le 0 x
    hy : LE.le 0 y
    hn : Ne n 0
    ⊢ LE.le (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n)) (HPow.hPow (HAdd.hAdd x y) …
  -/
  rcases Nat.exists_eq_add_one_of_ne_zero hn with ⟨k, rfl⟩
  induction k with
  | zero => simp only [zero_add, pow_one, le_refl]
  | succ k ih =>
    let n := k.succ
    have h1 := add_nonneg (mul_nonneg hx (pow_nonneg hy n)) (mul_nonneg hy (pow_nonneg hx n))
    have h2 := add_nonneg hx hy
    calc
      x ^ (n + 1) + y ^ (n + 1) ≤ x * x ^ n + y * y ^ n + (x * y ^ n + y * x ^ n) := by
        rw [pow_succ' _ n, pow_succ' _ n]
        exact le_add_of_nonneg_right h1
      _ = (x + y) * (x ^ n + y ^ n) := by
        rw [add_mul, mul_add, mul_add, add_comm (y * x ^ n), ← add_assoc, ← add_assoc,
          add_assoc (x * x ^ n) (x * y ^ n), add_comm (x * y ^ n) (y * y ^ n), ← add_assoc]
      _ ≤ (x + y) ^ (n + 1) := by
        rw [pow_succ' _ n]
        exact mul_le_mul_of_nonneg_left (ih (Nat.succ_ne_zero k)) h2


@[deprecated (since := "2024-09-28")] alias mul_le_one := mul_le_one₀

@[deprecated (since := "2024-09-28")] alias pow_le_one := pow_le_one₀

@[deprecated (since := "2024-09-28")] alias pow_lt_one := pow_lt_one₀

@[deprecated (since := "2024-09-28")] alias one_le_pow_of_one_le := one_le_pow₀

@[deprecated (since := "2024-09-28")] alias one_lt_pow := one_lt_pow₀

@[deprecated (since := "2024-10-04")] alias pow_right_mono := pow_right_mono₀

@[deprecated (since := "2024-10-04")] alias pow_le_pow_right := pow_le_pow_right₀

@[deprecated (since := "2024-10-04")] alias le_self_pow := le_self_pow₀


@[deprecated pow_le_pow_left₀ (since := "2024-11-13")]
theorem pow_le_pow_left {a b : R} (ha : 0 ≤ a) (hab : a ≤ b) : ∀ n, a ^ n ≤ b ^ n :=
  pow_le_pow_left₀ ha hab


lemma pow_add_pow_le' (ha : 0 ≤ a) (hb : 0 ≤ b) : a ^ n + b ^ n ≤ 2 * (a + b) ^ n := by
  /-
    R : Type u_3
    inst✝ : OrderedSemiring R
    a b : R
    n : Nat
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HMul.hMul 2 (HPow.hPow (H …
  -/
  rw [two_mul]
  exact add_le_add (pow_le_pow_left₀ ha (le_add_of_nonneg_right hb) _)
    (pow_le_pow_left₀ hb (le_add_of_nonneg_left ha) _)


/-- Turn an ordered domain into a strict ordered ring. -/
abbrev OrderedRing.toStrictOrderedRing (α : Type*)
    [OrderedRing α] [NoZeroDivisors α] [Nontrivial α] : StrictOrderedRing α where
  __ := ‹OrderedRing α›
  __ := ‹NoZeroDivisors α›
  mul_pos _ _ ap bp := (mul_nonneg ap.le bp.le).lt_of_ne' (mul_ne_zero ap.ne' bp.ne')


@[deprecated pow_lt_pow_left₀ (since := "2024-11-13")]
theorem pow_lt_pow_left (h : x < y) (hx : 0 ≤ x) : ∀ {n : ℕ}, n ≠ 0 → x ^ n < y ^ n :=
  pow_lt_pow_left₀ h hx


@[deprecated pow_left_strictMonoOn₀ (since := "2024-11-13")]
lemma pow_left_strictMonoOn (hn : n ≠ 0) : StrictMonoOn (· ^ n : R → R) {a | 0 ≤ a} :=
  pow_left_strictMonoOn₀ hn


@[deprecated pow_right_strictMono₀ (since := "2024-11-13")]
lemma pow_right_strictMono (h : 1 < a) : StrictMono (a ^ ·) :=
  pow_right_strictMono₀ h


@[deprecated pow_lt_pow_right₀ (since := "2024-11-13")]
theorem pow_lt_pow_right (h : 1 < a) (hmn : m < n) : a ^ m < a ^ n :=
  pow_lt_pow_right₀ h hmn


@[deprecated pow_lt_pow_iff_right₀ (since := "2024-11-13")]
lemma pow_lt_pow_iff_right (h : 1 < a) : a ^ n < a ^ m ↔ n < m := pow_lt_pow_iff_right₀ h


@[deprecated pow_le_pow_iff_right₀ (since := "2024-11-13")]
lemma pow_le_pow_iff_right (h : 1 < a) : a ^ n ≤ a ^ m ↔ n ≤ m := pow_le_pow_iff_right₀ h


@[deprecated lt_self_pow₀ (since := "2024-11-13")]
theorem lt_self_pow (h : 1 < a) (hm : 1 < m) : a < a ^ m := lt_self_pow₀ h hm


@[deprecated pow_right_strictAnti₀ (since := "2024-11-13")]
theorem pow_right_strictAnti (h₀ : 0 < a) (h₁ : a < 1) : StrictAnti (a ^ ·) :=
  pow_right_strictAnti₀ h₀ h₁


@[deprecated pow_lt_pow_iff_right_of_lt_one₀ (since := "2024-11-13")]
theorem pow_lt_pow_iff_right_of_lt_one (h₀ : 0 < a) (h₁ : a < 1) : a ^ m < a ^ n ↔ n < m :=
  pow_lt_pow_iff_right_of_lt_one₀ h₀ h₁


@[deprecated pow_lt_pow_right_of_lt_one₀ (since := "2024-11-13")]
theorem pow_lt_pow_right_of_lt_one (h₀ : 0 < a) (h₁ : a < 1) (hmn : m < n) : a ^ n < a ^ m :=
  pow_lt_pow_right_of_lt_one₀ h₀ h₁ hmn


@[deprecated pow_lt_self_of_lt_one₀ (since := "2024-11-13")]
theorem pow_lt_self_of_lt_one (h₀ : 0 < a) (h₁ : a < 1) (hn : 1 < n) : a ^ n < a :=
  pow_lt_self_of_lt_one₀ h₀ h₁ hn


                                                   /-
                                                     R : Type u_3
                                                     inst✝ : StrictOrderedRing R
                                                     a : R
                                                     ha : LT.lt a 0
                                                     ⊢ LT.lt 0 (HPow.hPow a 2)
                                                   -/
lemma sq_pos_of_neg (ha : a < 0) : 0 < a ^ 2 := by rw [sq]; exact mul_pos_of_neg_of_neg ha ha
                                                            /-
                                                              🎉 no goals
                                                            -/


@[deprecated pow_le_pow_iff_left₀ (since := "2024-11-12")]
lemma pow_le_pow_iff_left (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n ≤ b ^ n ↔ a ≤ b :=
  pow_le_pow_iff_left₀ ha hb hn


@[deprecated pow_lt_pow_iff_left₀ (since := "2024-11-12")]
lemma pow_lt_pow_iff_left (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n < b ^ n ↔ a < b :=
  pow_lt_pow_iff_left₀ ha hb hn


@[deprecated pow_left_inj₀ (since := "2024-11-12")]
lemma pow_left_inj (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n = b ^ n ↔ a = b :=
  pow_left_inj₀ ha hb hn


@[deprecated pow_right_injective₀ (since := "2024-11-12")]
lemma pow_right_injective (ha₀ : 0 < a) (ha₁ : a ≠ 1) : Injective (a ^ ·) :=
  pow_right_injective₀ ha₀ ha₁


@[deprecated pow_right_inj₀ (since := "2024-11-12")]
lemma pow_right_inj (ha₀ : 0 < a) (ha₁ : a ≠ 1) : a ^ m = a ^ n ↔ m = n := pow_right_inj₀ ha₀ ha₁


@[deprecated sq_le_one_iff₀ (since := "2024-11-12")]
theorem sq_le_one_iff {a : R} (ha : 0 ≤ a) : a ^ 2 ≤ 1 ↔ a ≤ 1 := sq_le_one_iff₀ ha


@[deprecated sq_lt_one_iff₀ (since := "2024-11-12")]
theorem sq_lt_one_iff {a : R} (ha : 0 ≤ a) : a ^ 2 < 1 ↔ a < 1 := sq_lt_one_iff₀ ha


@[deprecated one_le_sq_iff₀ (since := "2024-11-12")]
theorem one_le_sq_iff {a : R} (ha : 0 ≤ a) : 1 ≤ a ^ 2 ↔ 1 ≤ a := one_le_sq_iff₀ ha


@[deprecated one_lt_sq_iff₀ (since := "2024-11-12")]
theorem one_lt_sq_iff {a : R} (ha : 0 ≤ a) : 1 < a ^ 2 ↔ 1 < a := one_lt_sq_iff₀ ha


@[deprecated lt_of_pow_lt_pow_left₀ (since := "2024-11-12")]
theorem lt_of_pow_lt_pow_left (n : ℕ) (hb : 0 ≤ b) (h : a ^ n < b ^ n) : a < b :=
  lt_of_pow_lt_pow_left₀ n hb h


@[deprecated le_of_pow_le_pow_left₀ (since := "2024-11-12")]
theorem le_of_pow_le_pow_left (hn : n ≠ 0) (hb : 0 ≤ b) (h : a ^ n ≤ b ^ n) : a ≤ b :=
  le_of_pow_le_pow_left₀ hn hb h


@[deprecated sq_eq_sq₀ (since := "2024-11-12")]
theorem sq_eq_sq {a b : R} (ha : 0 ≤ a) (hb : 0 ≤ b) : a ^ 2 = b ^ 2 ↔ a = b := sq_eq_sq₀ ha hb


@[deprecated lt_of_mul_self_lt_mul_self₀ (since := "2024-11-12")]
theorem lt_of_mul_self_lt_mul_self (hb : 0 ≤ b) : a * a < b * b → a < b :=
  lt_of_mul_self_lt_mul_self₀ hb


lemma add_sq_le : (a + b) ^ 2 ≤ 2 * (a ^ 2 + b ^ 2) := by
  calc
    (a + b) ^ 2 = a ^ 2 + b ^ 2 + (a * b + b * a) := by
        simp_rw [pow_succ', pow_zero, mul_one, add_mul, mul_add, add_comm (b * a), add_add_add_comm]
    _ ≤ a ^ 2 + b ^ 2 + (a * a + b * b) := add_le_add_left ?_ _
    _ = _ := by simp_rw [pow_succ', pow_zero, mul_one, two_mul]
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a b : R
    inst✝ : ExistsAddOfLE R
    ⊢ LE.le (HAdd.hAdd (HMul.hMul a b) (HMul.hMul b a)) (HAdd.hAdd (HMul.hMul a a) …
  -/
  cases le_total a b
    /-
      case inl
      R : Type u_3
      inst✝¹ : LinearOrderedSemiring R
      a b : R
      inst✝ : ExistsAddOfLE R
      h✝ : LE.le a b
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a b) (HMul.hMul b a)) (HAdd.hAdd (HMul.hMul a a) …
    -/
  · exact mul_add_mul_le_mul_add_mul ‹_› ‹_›
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_3
      inst✝¹ : LinearOrderedSemiring R
      a b : R
      inst✝ : ExistsAddOfLE R
      h✝ : LE.le b a
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a b) (HMul.hMul b a)) (HAdd.hAdd (HMul.hMul a a) …
    -/
  · exact mul_add_mul_le_mul_add_mul' ‹_› ‹_›
    /-
      🎉 no goals
    -/

-- TODO: Use `gcongr`, `positivity`, `ring` once those tactics are made available here

lemma add_pow_le (ha : 0 ≤ a) (hb : 0 ≤ b) : ∀ n, (a + b) ^ n ≤ 2 ^ (n - 1) * (a ^ n + b ^ n)
            /-
              R : Type u_3
              inst✝¹ : LinearOrderedSemiring R
              a b : R
              inst✝ : ExistsAddOfLE R
              ha : LE.le 0 a
              hb : LE.le 0 b
              ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) 0) (HMul.hMul (HPow.hPow 2 (HSub.hSub 0 1)) …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              R : Type u_3
              inst✝¹ : LinearOrderedSemiring R
              a b : R
              inst✝ : ExistsAddOfLE R
              ha : LE.le 0 a
              hb : LE.le 0 b
              ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) 1) (HMul.hMul (HPow.hPow 2 (HSub.hSub 1 1)) …
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
  | n + 2 => by
    /-
      R : Type u_3
      inst✝¹ : LinearOrderedSemiring R
      a b : R
      inst✝ : ExistsAddOfLE R
      ha : LE.le 0 a
      hb : LE.le 0 b
      n : Nat
      ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) (HAdd.hAdd n 2)) (HMul.hMul (HPow.hPow 2 (H …
    -/
    rw [pow_succ]
    calc
      _ ≤ 2 ^ n * (a ^ (n + 1) + b ^ (n + 1)) * (a + b) :=
          mul_le_mul_of_nonneg_right (add_pow_le ha hb (n + 1)) <| add_nonneg ha hb
      _ = 2 ^ n * (a ^ (n + 2) + b ^ (n + 2) + (a ^ (n + 1) * b + b ^ (n + 1) * a)) := by
          rw [mul_assoc, mul_add, add_mul, add_mul, ← pow_succ, ← pow_succ, add_comm _ (b ^ _),
            add_add_add_comm, add_comm (_ * a)]
      _ ≤ 2 ^ n * (a ^ (n + 2) + b ^ (n + 2) + (a ^ (n + 1) * a + b ^ (n + 1) * b)) :=
          mul_le_mul_of_nonneg_left (add_le_add_left ?_ _) <| pow_nonneg (zero_le_two (α := R)) _
      _ = _ := by simp only [← pow_succ, ← two_mul, ← mul_assoc]; rfl
      /-
        R : Type u_3
        inst✝¹ : LinearOrderedSemiring R
        a b : R
        inst✝ : ExistsAddOfLE R
        ha : LE.le 0 a
        hb : LE.le 0 b
        n : Nat
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a (HAdd.hAdd n 1)) b) (HMul.hMul (HPo …
      -/
    · obtain hab | hba := le_total a b
        /-
          case inl
          R : Type u_3
          inst✝¹ : LinearOrderedSemiring R
          a b : R
          inst✝ : ExistsAddOfLE R
          ha : LE.le 0 a
          hb : LE.le 0 b
          n : Nat
          hab : LE.le a b
          ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a (HAdd.hAdd n 1)) b) (HMul.hMul (HPo …
        -/
      · exact mul_add_mul_le_mul_add_mul (pow_le_pow_left₀ ha hab _) hab
        /-
          🎉 no goals
        -/
        /-
          case inr
          R : Type u_3
          inst✝¹ : LinearOrderedSemiring R
          a b : R
          inst✝ : ExistsAddOfLE R
          ha : LE.le 0 a
          hb : LE.le 0 b
          n : Nat
          hba : LE.le b a
          ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a (HAdd.hAdd n 1)) b) (HMul.hMul (HPo …
        -/
      · exact mul_add_mul_le_mul_add_mul' (pow_le_pow_left₀ hb hba _) hba
        /-
          🎉 no goals
        -/


protected lemma Even.add_pow_le (hn : Even n) :
    (a + b) ^ n ≤ 2 ^ (n - 1) * (a ^ n + b ^ n) := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a b : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Even n
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) n) (HMul.hMul (HPow.hPow 2 (HSub.hSub n 1)) …
  -/
  obtain ⟨n, rfl⟩ := hn
  /-
    case intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a b : R
    inst✝ : ExistsAddOfLE R
    n : Nat
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) (HAdd.hAdd n n)) (HMul.hMul (HPow.hPow 2 (H …
  -/
  rw [← two_mul, pow_mul]
  calc
    _ ≤ (2 * (a ^ 2 + b ^ 2)) ^ n := pow_le_pow_left₀ (sq_nonneg _) add_sq_le _
    _ = 2 ^ n * (a ^ 2 + b ^ 2) ^ n := by -- TODO: Should be `Nat.cast_commute`
        rw [Commute.mul_pow]; simp [Commute, SemiconjBy, two_mul, mul_two]
    _ ≤ 2 ^ n * (2 ^ (n - 1) * ((a ^ 2) ^ n + (b ^ 2) ^ n)) := mul_le_mul_of_nonneg_left
          (add_pow_le (sq_nonneg _) (sq_nonneg _) _) <| pow_nonneg (zero_le_two (α := R)) _
    _ = _ := by
      simp only [← mul_assoc, ← pow_add, ← pow_mul]
      cases n
      · rfl
      · simp [Nat.two_mul]


lemma Even.pow_nonneg (hn : Even n) (a : R) : 0 ≤ a ^ n := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Even n
    a : R
    ⊢ LE.le 0 (HPow.hPow a n)
  -/
  obtain ⟨k, rfl⟩ := hn; rw [pow_add]; exact mul_self_nonneg _
                                       /-
                                         🎉 no goals
                                       -/


lemma Even.pow_pos (hn : Even n) (ha : a ≠ 0) : 0 < a ^ n :=
  (hn.pow_nonneg _).lt_of_ne' (pow_ne_zero _ ha)


lemma Even.pow_pos_iff (hn : Even n) (h₀ : n ≠ 0) : 0 < a ^ n ↔ a ≠ 0 := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Even n
    h₀ : Ne n 0
    ⊢ Iff (LT.lt 0 (HPow.hPow a n)) (Ne a 0)
  -/
  obtain ⟨k, rfl⟩ := hn; rw [pow_add, mul_self_pos (α := R), pow_ne_zero_iff (by simpa using h₀)]
                         /-
                           🎉 no goals
                         -/


lemma Odd.pow_neg_iff (hn : Odd n) : a ^ n < 0 ↔ a < 0 := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    ⊢ Iff (LT.lt (HPow.hPow a n) 0) (LT.lt a 0)
  -/
  refine ⟨lt_imp_lt_of_le_imp_le (pow_nonneg · _), fun ha ↦ ?_⟩
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    ha : LT.lt a 0
    ⊢ LT.lt (HPow.hPow a n) 0
  -/
  obtain ⟨k, rfl⟩ := hn
  /-
    case intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    inst✝ : ExistsAddOfLE R
    ha : LT.lt a 0
    k : Nat
    ⊢ LT.lt (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 k) 1)) 0
  -/
  rw [pow_succ]
  /-
    case intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    inst✝ : ExistsAddOfLE R
    ha : LT.lt a 0
    k : Nat
    ⊢ LT.lt (HMul.hMul (HPow.hPow a (HMul.hMul 2 k)) a) 0
  -/
  exact mul_neg_of_pos_of_neg ((even_two_mul _).pow_pos ha.ne) ha
  /-
    🎉 no goals
  -/


lemma Odd.pow_nonneg_iff (hn : Odd n) : 0 ≤ a ^ n ↔ 0 ≤ a :=
  le_iff_le_iff_lt_iff_lt.2 hn.pow_neg_iff


lemma Odd.pow_nonpos_iff (hn : Odd n) : a ^ n ≤ 0 ↔ a ≤ 0 := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    ⊢ Iff (LE.le (HPow.hPow a n) 0) (LE.le a 0)
  -/
  rw [le_iff_lt_or_eq, le_iff_lt_or_eq, hn.pow_neg_iff, pow_eq_zero_iff]
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    a : R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    ⊢ Ne n 0
  -/
  rintro rfl; simp [Odd, eq_comm (a := 0)] at hn
              /-
                🎉 no goals
              -/


lemma Odd.pow_pos_iff (hn : Odd n) : 0 < a ^ n ↔ 0 < a := lt_iff_lt_of_le_iff_le hn.pow_nonpos_iff


alias ⟨_, Odd.pow_nonpos⟩ := Odd.pow_nonpos_iff

alias ⟨_, Odd.pow_neg⟩ := Odd.pow_neg_iff


lemma Odd.strictMono_pow (hn : Odd n) : StrictMono fun a : R => a ^ n := by
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    ⊢ StrictMono fun a => HPow.hPow a n
  -/
  have hn₀ : n ≠ 0 := by rintro rfl; simp [Odd, eq_comm (a := 0)] at hn
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    ⊢ StrictMono fun a => HPow.hPow a n
  -/
  intro a b hab
  /-
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  obtain ha | ha := le_total 0 a
    /-
      case inl
      R : Type u_3
      inst✝¹ : LinearOrderedSemiring R
      n : Nat
      inst✝ : ExistsAddOfLE R
      hn : Odd n
      hn₀ : Ne n 0
      a b : R
      hab : LT.lt a b
      ha : LE.le 0 a
      ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
    -/
  · exact pow_lt_pow_left₀ hab ha hn₀
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  obtain hb | hb := lt_or_le 0 b
    /-
      case inr.inl
      R : Type u_3
      inst✝¹ : LinearOrderedSemiring R
      n : Nat
      inst✝ : ExistsAddOfLE R
      hn : Odd n
      hn₀ : Ne n 0
      a b : R
      hab : LT.lt a b
      ha : LE.le a 0
      hb : LT.lt 0 b
      ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
    -/
  · exact (hn.pow_nonpos ha).trans_lt (pow_pos hb _)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  obtain ⟨c, hac⟩ := exists_add_of_le ha
  /-
    case inr.inr.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  obtain ⟨d, hbd⟩ := exists_add_of_le hb
  /-
    case inr.inr.intro.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    d : R
    hbd : Eq 0 (HAdd.hAdd b d)
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  have hd := nonneg_of_le_add_right (hb.trans_eq hbd)
  /-
    case inr.inr.intro.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    d : R
    hbd : Eq 0 (HAdd.hAdd b d)
    hd : LE.le 0 d
    ⊢ LT.lt ((fun a => HPow.hPow a n) a) ((fun a => HPow.hPow a n) b)
  -/
  refine lt_of_add_lt_add_right (a := c ^ n + d ^ n) ?_
  /-
    case inr.inr.intro.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    d : R
    hbd : Eq 0 (HAdd.hAdd b d)
    hd : LE.le 0 d
    ⊢ LT.lt (HAdd.hAdd ((fun a => HPow.hPow a n) a) (HAdd.hAdd (HPow.hPow c n) (HP …
  -/
  dsimp
  calc
    a ^ n + (c ^ n + d ^ n) = d ^ n := by
      rw [← add_assoc, hn.pow_add_pow_eq_zero hac.symm, zero_add]
    _ < c ^ n := pow_lt_pow_left₀ ?_ hd hn₀
    _ = b ^ n + (c ^ n + d ^ n) := by rw [add_left_comm, hn.pow_add_pow_eq_zero hbd.symm, add_zero]
  /-
    case inr.inr.intro.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    d : R
    hbd : Eq 0 (HAdd.hAdd b d)
    hd : LE.le 0 d
    ⊢ LT.lt d c
  -/
  refine lt_of_add_lt_add_right (a := a + b) ?_
  /-
    case inr.inr.intro.intro
    R : Type u_3
    inst✝¹ : LinearOrderedSemiring R
    n : Nat
    inst✝ : ExistsAddOfLE R
    hn : Odd n
    hn₀ : Ne n 0
    a b : R
    hab : LT.lt a b
    ha : LE.le a 0
    hb : LE.le b 0
    c : R
    hac : Eq 0 (HAdd.hAdd a c)
    d : R
    hbd : Eq 0 (HAdd.hAdd b d)
    hd : LE.le 0 d
    ⊢ LT.lt (HAdd.hAdd d (HAdd.hAdd a b)) (HAdd.hAdd c (HAdd.hAdd a b))
  -/
  rwa [add_rotate', ← hbd, add_zero, add_left_comm, ← add_assoc, ← hac, zero_add]
  /-
    🎉 no goals
  -/


lemma sq_pos_iff {a : R} : 0 < a ^ 2 ↔ a ≠ 0 := even_two.pow_pos_iff two_ne_zero


alias ⟨_, sq_pos_of_ne_zero⟩ := sq_pos_iff

alias pow_two_pos_of_ne_zero := sq_pos_of_ne_zero


lemma pow_four_le_pow_two_of_pow_two_le (h : a ^ 2 ≤ b) : a ^ 4 ≤ b ^ 2 :=
  (pow_mul a 2 2).symm ▸ pow_le_pow_left₀ (sq_nonneg a) h 2


