/-- A mixin for power-associative multiplication. -/
class NatPowAssoc (M : Type*) [MulOneClass M] [Pow M ℕ] : Prop where
  /-- Multiplication is power-associative. -/
  protected npow_add : ∀ (k n : ℕ) (x : M), x ^ (k + n) = x ^ k * x ^ n
  /-- Exponent zero is one. -/
  protected npow_zero : ∀ (x : M), x ^ 0 = 1
  /-- Exponent one is identity. -/
  protected npow_one : ∀ (x : M), x ^ 1 = x


theorem npow_add (k n : ℕ) (x : M) : x ^ (k + n) = x ^ k * x ^ n  :=
  NatPowAssoc.npow_add k n x


@[simp]
theorem npow_zero (x : M) : x ^ 0 = 1 :=
  NatPowAssoc.npow_zero x


@[simp]
theorem npow_one (x : M) : x ^ 1 = x :=
  NatPowAssoc.npow_one x


theorem npow_mul_assoc (k m n : ℕ) (x : M) :
    (x ^ k * x ^ m) * x ^ n = x ^ k * (x ^ m * x ^ n) := by
  /-
    M : Type u_1
    inst✝² : MulOneClass M
    inst✝¹ : Pow M Nat
    inst✝ : NatPowAssoc M
    k m n : Nat
    x : M
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x k) (HPow.hPow x m)) (HPow.hPow x n)) ( …
  -/
  simp only [← npow_add, add_assoc]
  /-
    🎉 no goals
  -/


theorem npow_mul_comm (m n : ℕ) (x : M) :
                                        /-
                                          M : Type u_1
                                          inst✝² : MulOneClass M
                                          inst✝¹ : Pow M Nat
                                          inst✝ : NatPowAssoc M
                                          m n : Nat
                                          x : M
                                          ⊢ Eq (HMul.hMul (HPow.hPow x m) (HPow.hPow x n)) (HMul.hMul (HPow.hPow x n) (H …
                                        -/
    x ^ m * x ^ n = x ^ n * x ^ m := by simp only [← npow_add, add_comm]
                                        /-
                                          🎉 no goals
                                        -/


theorem npow_mul (x : M) (m n : ℕ) : x ^ (m * n) = (x ^ m) ^ n := by
  induction n with
  | zero => rw [npow_zero, Nat.mul_zero, npow_zero]
  | succ n ih => rw [mul_add, npow_add, ih, mul_one, npow_add, npow_one]


theorem npow_mul' (x : M) (m n : ℕ) : x ^ (m * n) = (x ^ n) ^ m := by
  /-
    M : Type u_1
    inst✝² : MulOneClass M
    inst✝¹ : Pow M Nat
    inst✝ : NatPowAssoc M
    x : M
    m n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul m n)) (HPow.hPow (HPow.hPow x n) m)
  -/
  rw [mul_comm]
  /-
    M : Type u_1
    inst✝² : MulOneClass M
    inst✝¹ : Pow M Nat
    inst✝ : NatPowAssoc M
    x : M
    m n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul n m)) (HPow.hPow (HPow.hPow x n) m)
  -/
  exact npow_mul x n m
  /-
    🎉 no goals
  -/


theorem neg_npow_assoc {R : Type*} [NonAssocRing R] [Pow R ℕ] [NatPowAssoc R] (a b : R) (k : ℕ) :
    (-1)^k * a * b = (-1)^k * (a * b) := by
  induction k with
  | zero => simp only [npow_zero, one_mul]
  | succ k ih =>
    rw [npow_add, npow_one, ← neg_mul_comm, mul_one]
    simp only [neg_mul, ih]


instance Pi.instNatPowAssoc {ι : Type*} {α : ι → Type*} [∀ i, MulOneClass <| α i] [∀ i, Pow (α i) ℕ]
    [∀ i, NatPowAssoc <| α i] : NatPowAssoc (∀ i, α i) where
                         /-
                           M : Type u_1
                           ι : Type u_2
                           α : ι → Type u_3
                           inst✝² : (i : ι) → MulOneClass (α i)
                           inst✝¹ : (i : ι) → Pow (α i) Nat
                           inst✝ : ∀ (i : ι), NatPowAssoc (α i)
                           x✝² x✝¹ : Nat
                           x✝ : (i : ι) → α i
                           ⊢ Eq (HPow.hPow x✝ (HAdd.hAdd x✝² x✝¹)) (HMul.hMul (HPow.hPow x✝ x✝²) (HPow.hP …
                         -/
    npow_add _ _ _ := by ext; simp [npow_add]
                              /-
                                🎉 no goals
                              -/
                      /-
                        M : Type u_1
                        ι : Type u_2
                        α : ι → Type u_3
                        inst✝² : (i : ι) → MulOneClass (α i)
                        inst✝¹ : (i : ι) → Pow (α i) Nat
                        inst✝ : ∀ (i : ι), NatPowAssoc (α i)
                        x✝ : (i : ι) → α i
                        ⊢ Eq (HPow.hPow x✝ 0) 1
                      -/
    npow_zero _ := by ext; simp
                           /-
                             🎉 no goals
                           -/
                     /-
                       M : Type u_1
                       ι : Type u_2
                       α : ι → Type u_3
                       inst✝² : (i : ι) → MulOneClass (α i)
                       inst✝¹ : (i : ι) → Pow (α i) Nat
                       inst✝ : ∀ (i : ι), NatPowAssoc (α i)
                       x✝ : (i : ι) → α i
                       ⊢ Eq (HPow.hPow x✝ 1) x✝
                     -/
    npow_one _ := by ext; simp
                          /-
                            🎉 no goals
                          -/


instance Prod.instNatPowAssoc {N : Type*} [MulOneClass M] [Pow M ℕ] [NatPowAssoc M] [MulOneClass N]
    [Pow N ℕ] [NatPowAssoc N] : NatPowAssoc (M × N) where
                       /-
                         M : Type u_1
                         N : Type u_2
                         inst✝⁵ : MulOneClass M
                         inst✝⁴ : Pow M Nat
                         inst✝³ : NatPowAssoc M
                         inst✝² : MulOneClass N
                         inst✝¹ : Pow N Nat
                         inst✝ : NatPowAssoc N
                         x✝² x✝¹ : Nat
                         x✝ : Prod M N
                         ⊢ Eq (HPow.hPow x✝ (HAdd.hAdd x✝² x✝¹)) (HMul.hMul (HPow.hPow x✝ x✝²) (HPow.hP …
                       -/
                               /-
                                 🎉 no goals
                               -/
  npow_add _ _ _ := by ext <;> simp [npow_add]
                               /-
                                 🎉 no goals
                               -/
                    /-
                      M : Type u_1
                      N : Type u_2
                      inst✝⁵ : MulOneClass M
                      inst✝⁴ : Pow M Nat
                      inst✝³ : NatPowAssoc M
                      inst✝² : MulOneClass N
                      inst✝¹ : Pow N Nat
                      inst✝ : NatPowAssoc N
                      x✝ : Prod M N
                      ⊢ Eq (HPow.hPow x✝ 0) 1
                    -/
                            /-
                              🎉 no goals
                            -/
  npow_zero _ := by ext <;> simp
                            /-
                              🎉 no goals
                            -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     inst✝⁵ : MulOneClass M
                     inst✝⁴ : Pow M Nat
                     inst✝³ : NatPowAssoc M
                     inst✝² : MulOneClass N
                     inst✝¹ : Pow N Nat
                     inst✝ : NatPowAssoc N
                     x✝ : Prod M N
                     ⊢ Eq (HPow.hPow x✝ 1) x✝
                   -/
                           /-
                             🎉 no goals
                           -/
  npow_one _ := by ext <;> simp
                           /-
                             🎉 no goals
                           -/


instance Monoid.PowAssoc : NatPowAssoc M where
  npow_add _ _ _ := pow_add _ _ _
  npow_zero _ := pow_zero _
  npow_one _ := pow_one _


@[simp, norm_cast]
theorem Nat.cast_npow (R : Type*) [NonAssocSemiring R] [Pow R ℕ] [NatPowAssoc R] (n m : ℕ) :
    (↑(n ^ m) : R) = (↑n : R) ^ m := by
  induction m with
  | zero => simp only [pow_zero, Nat.cast_one, npow_zero]
  | succ m ih => rw [npow_add, npow_add, Nat.cast_mul, ih, npow_one, npow_one]


@[simp, norm_cast]
theorem Int.cast_npow (R : Type*) [NonAssocRing R] [Pow R ℕ] [NatPowAssoc R]
    (n : ℤ) : ∀(m : ℕ), @Int.cast R NonAssocRing.toIntCast (n ^ m) = (n : R) ^ m
  | 0 => by
    /-
      R : Type u_2
      inst✝² : NonAssocRing R
      inst✝¹ : Pow R Nat
      inst✝ : NatPowAssoc R
      n : Int
      ⊢ Eq (↑(HPow.hPow n 0)) (HPow.hPow (↑n) 0)
    -/
    rw [pow_zero, npow_zero, Int.cast_one]
    /-
      🎉 no goals
    -/
  | m + 1 => by
    /-
      R : Type u_2
      inst✝² : NonAssocRing R
      inst✝¹ : Pow R Nat
      inst✝ : NatPowAssoc R
      n : Int
      m : Nat
      ⊢ Eq (↑(HPow.hPow n (HAdd.hAdd m 1))) (HPow.hPow (↑n) (HAdd.hAdd m 1))
    -/
    rw [npow_add, npow_one, Int.cast_mul, Int.cast_npow R n m, npow_add, npow_one]
    /-
      🎉 no goals
    -/


