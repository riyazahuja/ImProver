/-- A `Prop`-valued mixin for power-associative multiplication in the non-unital setting. -/
class PNatPowAssoc (M : Type*) [Mul M] [Pow M ℕ+] : Prop where
  /-- Multiplication is power-associative. -/
  protected ppow_add : ∀ (k n : ℕ+) (x : M), x ^ (k + n) = x ^ k * x ^ n
  /-- Exponent one is identity. -/
  protected ppow_one : ∀ (x : M), x ^ (1 : ℕ+) = x


theorem ppow_add (k n : ℕ+) (x : M) : x ^ (k + n) = x ^ k * x ^ n :=
  PNatPowAssoc.ppow_add k n x


@[simp]
theorem ppow_one (x : M) : x ^ (1 : ℕ+) = x :=
  PNatPowAssoc.ppow_one x


theorem ppow_mul_assoc (k m n : ℕ+) (x : M) :
    (x ^ k * x ^ m) * x ^ n = x ^ k * (x ^ m * x ^ n) := by
  /-
    M : Type u_1
    inst✝² : Mul M
    inst✝¹ : Pow M PNat
    inst✝ : PNatPowAssoc M
    k m n : PNat
    x : M
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x k) (HPow.hPow x m)) (HPow.hPow x n)) ( …
  -/
  simp only [← ppow_add, add_assoc]
  /-
    🎉 no goals
  -/


theorem ppow_mul_comm (m n : ℕ+) (x : M) :
                                        /-
                                          M : Type u_1
                                          inst✝² : Mul M
                                          inst✝¹ : Pow M PNat
                                          inst✝ : PNatPowAssoc M
                                          m n : PNat
                                          x : M
                                          ⊢ Eq (HMul.hMul (HPow.hPow x m) (HPow.hPow x n)) (HMul.hMul (HPow.hPow x n) (H …
                                        -/
    x ^ m * x ^ n = x ^ n * x ^ m := by simp only [← ppow_add, add_comm]
                                        /-
                                          🎉 no goals
                                        -/


theorem ppow_mul (x : M) (m n : ℕ+) : x ^ (m * n) = (x ^ m) ^ n := by
  /-
    M : Type u_1
    inst✝² : Mul M
    inst✝¹ : Pow M PNat
    inst✝ : PNatPowAssoc M
    x : M
    m n : PNat
    ⊢ Eq (HPow.hPow x (HMul.hMul m n)) (HPow.hPow (HPow.hPow x m) n)
  -/
  refine PNat.recOn n ?_ fun k hk ↦ ?_
    /-
      case refine_1
      M : Type u_1
      inst✝² : Mul M
      inst✝¹ : Pow M PNat
      inst✝ : PNatPowAssoc M
      x : M
      m n : PNat
      ⊢ Eq (HPow.hPow x (HMul.hMul m 1)) (HPow.hPow (HPow.hPow x m) 1)
    -/
  · rw [ppow_one, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝² : Mul M
      inst✝¹ : Pow M PNat
      inst✝ : PNatPowAssoc M
      x : M
      m n k : PNat
      hk : Eq (HPow.hPow x (HMul.hMul m k)) (HPow.hPow (HPow.hPow x m) k)
      ⊢ Eq (HPow.hPow x (HMul.hMul m (HAdd.hAdd k 1))) (HPow.hPow (HPow.hPow x m) (H …
    -/
  · rw [ppow_add, ppow_one, mul_add, ppow_add, mul_one, hk]
    /-
      🎉 no goals
    -/


theorem ppow_mul' (x : M) (m n : ℕ+) : x ^ (m * n) = (x ^ n) ^ m := by
  /-
    M : Type u_1
    inst✝² : Mul M
    inst✝¹ : Pow M PNat
    inst✝ : PNatPowAssoc M
    x : M
    m n : PNat
    ⊢ Eq (HPow.hPow x (HMul.hMul m n)) (HPow.hPow (HPow.hPow x n) m)
  -/
  rw [mul_comm]
  /-
    M : Type u_1
    inst✝² : Mul M
    inst✝¹ : Pow M PNat
    inst✝ : PNatPowAssoc M
    x : M
    m n : PNat
    ⊢ Eq (HPow.hPow x (HMul.hMul n m)) (HPow.hPow (HPow.hPow x n) m)
  -/
  exact ppow_mul x n m
  /-
    🎉 no goals
  -/


instance Pi.instPNatPowAssoc {ι : Type*} {α : ι → Type*} [∀ i, Mul <| α i] [∀ i, Pow (α i) ℕ+]
    [∀ i, PNatPowAssoc <| α i] : PNatPowAssoc (∀ i, α i) where
                       /-
                         M : Type u_1
                         ι : Type u_2
                         α : ι → Type u_3
                         inst✝² : (i : ι) → Mul (α i)
                         inst✝¹ : (i : ι) → Pow (α i) PNat
                         inst✝ : ∀ (i : ι), PNatPowAssoc (α i)
                         x✝² x✝¹ : PNat
                         x✝ : (i : ι) → α i
                         ⊢ Eq (HPow.hPow x✝ (HAdd.hAdd x✝² x✝¹)) (HMul.hMul (HPow.hPow x✝ x✝²) (HPow.hP …
                       -/
  ppow_add _ _ _ := by ext; simp [ppow_add]
                            /-
                              🎉 no goals
                            -/
                   /-
                     M : Type u_1
                     ι : Type u_2
                     α : ι → Type u_3
                     inst✝² : (i : ι) → Mul (α i)
                     inst✝¹ : (i : ι) → Pow (α i) PNat
                     inst✝ : ∀ (i : ι), PNatPowAssoc (α i)
                     x✝ : (i : ι) → α i
                     ⊢ Eq (HPow.hPow x✝ 1) x✝
                   -/
  ppow_one _ := by ext; simp
                        /-
                          🎉 no goals
                        -/


instance Prod.instPNatPowAssoc {N : Type*} [Mul M] [Pow M ℕ+] [PNatPowAssoc M] [Mul N] [Pow N ℕ+]
    [PNatPowAssoc N] : PNatPowAssoc (M × N) where
                       /-
                         M : Type u_1
                         N : Type u_2
                         inst✝⁵ : Mul M
                         inst✝⁴ : Pow M PNat
                         inst✝³ : PNatPowAssoc M
                         inst✝² : Mul N
                         inst✝¹ : Pow N PNat
                         inst✝ : PNatPowAssoc N
                         x✝² x✝¹ : PNat
                         x✝ : Prod M N
                         ⊢ Eq (HPow.hPow x✝ (HAdd.hAdd x✝² x✝¹)) (HMul.hMul (HPow.hPow x✝ x✝²) (HPow.hP …
                       -/
                               /-
                                 🎉 no goals
                               -/
  ppow_add _ _ _ := by ext <;> simp [ppow_add]
                               /-
                                 🎉 no goals
                               -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     inst✝⁵ : Mul M
                     inst✝⁴ : Pow M PNat
                     inst✝³ : PNatPowAssoc M
                     inst✝² : Mul N
                     inst✝¹ : Pow N PNat
                     inst✝ : PNatPowAssoc N
                     x✝ : Prod M N
                     ⊢ Eq (HPow.hPow x✝ 1) x✝
                   -/
                           /-
                             🎉 no goals
                           -/
  ppow_one _ := by ext <;> simp
                           /-
                             🎉 no goals
                           -/


theorem ppow_eq_pow [Monoid M] [Pow M ℕ+] [PNatPowAssoc M] (x : M) (n : ℕ+) :
    x ^ n = x ^ (n : ℕ) := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    inst✝¹ : Pow M PNat
    inst✝ : PNatPowAssoc M
    x : M
    n : PNat
    ⊢ Eq (HPow.hPow x n) (HPow.hPow x ↑n)
  -/
  refine PNat.recOn n ?_ fun k hk ↦ ?_
    /-
      case refine_1
      M : Type u_1
      inst✝² : Monoid M
      inst✝¹ : Pow M PNat
      inst✝ : PNatPowAssoc M
      x : M
      n : PNat
      ⊢ Eq (HPow.hPow x 1) (HPow.hPow x ↑1)
    -/
  · rw [ppow_one, PNat.one_coe, pow_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝² : Monoid M
      inst✝¹ : Pow M PNat
      inst✝ : PNatPowAssoc M
      x : M
      n k : PNat
      hk : Eq (HPow.hPow x k) (HPow.hPow x ↑k)
      ⊢ Eq (HPow.hPow x (HAdd.hAdd k 1)) (HPow.hPow x ↑(HAdd.hAdd k 1))
    -/
  · rw [ppow_add, ppow_one, PNat.add_coe, pow_add, PNat.one_coe, pow_one, ← hk]
    /-
      🎉 no goals
    -/

