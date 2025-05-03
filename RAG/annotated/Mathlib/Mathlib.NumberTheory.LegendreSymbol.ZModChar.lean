/-- Define the nontrivial quadratic character on `ZMod 4`, `χ₄`.
It corresponds to the extension `ℚ(√-1)/ℚ`. -/
@[simps]
def χ₄ : MulChar (ZMod 4) ℤ where
  toFun a :=
    match a with
    | 0 | 2 => 0
    | 1 => 1
    | 3 => -1
  map_one' := rfl
                 /-
                   ⊢ ∀ (x y : ZMod 4), Eq ({ toFun := fun a => ZMod.χ₄.match_1 (fun a => Int) a ( …
                 -/
  map_mul' := by decide
                 /-
                   🎉 no goals
                 -/
                     /-
                       ⊢ ∀ (a : ZMod 4), Not (IsUnit a) → Eq ((↑{ toFun := fun a => ZMod.χ₄.match_1 ( …
                     -/
  map_nonunit' := by decide
                     /-
                       🎉 no goals
                     -/


/-- `χ₄` takes values in `{0, 1, -1}` -/
theorem isQuadratic_χ₄ : χ₄.IsQuadratic := by
  /-
    ⊢ ZMod.χ₄.IsQuadratic
  -/
  unfold MulChar.IsQuadratic
  /-
    ⊢ ∀ (a : ZMod 4), Or (Eq (ZMod.χ₄ a) 0) (Or (Eq (ZMod.χ₄ a) 1) (Eq (ZMod.χ₄ a) …
  -/
  decide
  /-
    🎉 no goals
  -/


/-- The value of `χ₄ n`, for `n : ℕ`, depends only on `n % 4`. -/
theorem χ₄_nat_mod_four (n : ℕ) : χ₄ n = χ₄ (n % 4 : ℕ) := by
  /-
    n : Nat
    ⊢ Eq (ZMod.χ₄ ↑n) (ZMod.χ₄ ↑(HMod.hMod n 4))
  -/
  rw [← ZMod.natCast_mod n 4]
  /-
    🎉 no goals
  -/


/-- The value of `χ₄ n`, for `n : ℤ`, depends only on `n % 4`. -/
theorem χ₄_int_mod_four (n : ℤ) : χ₄ n = χ₄ (n % 4 : ℤ) := by
  /-
    n : Int
    ⊢ Eq (ZMod.χ₄ ↑n) (ZMod.χ₄ ↑(HMod.hMod n 4))
  -/
  rw [← ZMod.intCast_mod n 4, Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


/-- An explicit description of `χ₄` on integers / naturals -/
theorem χ₄_int_eq_if_mod_four (n : ℤ) :
    χ₄ n = if n % 2 = 0 then 0 else if n % 4 = 1 then 1 else -1 := by
  have help : ∀ m : ℤ, 0 ≤ m → m < 4 → χ₄ m = if m % 2 = 0 then 0 else if m = 1 then 1 else -1 := by
    decide
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 4 → Eq (ZMod.χ₄ ↑m) (ite (Eq (HMod.hMo …
    ⊢ Eq (ZMod.χ₄ ↑n) (ite (Eq (HMod.hMod n 2) 0) 0 (ite (Eq (HMod.hMod n 4) 1) 1  …
  -/
  rw [← Int.emod_emod_of_dvd n (by omega : (2 : ℤ) ∣ 4), ← ZMod.intCast_mod n 4]
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 4 → Eq (ZMod.χ₄ ↑m) (ite (Eq (HMod.hMo …
    ⊢ Eq (ZMod.χ₄ ↑(HMod.hMod n ↑4)) (ite (Eq (HMod.hMod (HMod.hMod n 4) 2) 0) 0 ( …
  -/
  exact help (n % 4) (Int.emod_nonneg n (by omega)) (Int.emod_lt n (by omega))
  /-
    🎉 no goals
  -/


theorem χ₄_nat_eq_if_mod_four (n : ℕ) :
    χ₄ n = if n % 2 = 0 then 0 else if n % 4 = 1 then 1 else -1 :=
  mod_cast χ₄_int_eq_if_mod_four n


/-- Alternative description of `χ₄ n` for odd `n : ℕ` in terms of powers of `-1` -/
theorem χ₄_eq_neg_one_pow {n : ℕ} (hn : n % 2 = 1) : χ₄ n = (-1) ^ (n / 2) := by
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq (ZMod.χ₄ ↑n) (HPow.hPow (-1) (HDiv.hDiv n 2))
  -/
  rw [χ₄_nat_eq_if_mod_four]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq (ite (Eq (HMod.hMod n 2) 0) 0 (ite (Eq (HMod.hMod n 4) 1) 1 (-1))) (HPow. …
  -/
  simp only [hn, Nat.one_ne_zero, if_false]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq (ite (Eq (HMod.hMod n 4) 1) 1 (-1)) (HPow.hPow (-1) (HDiv.hDiv n 2))
  -/
  nth_rewrite 3 [← Nat.div_add_mod n 4]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq (ite (Eq (HMod.hMod n 4) 1) 1 (-1)) (HPow.hPow (-1) (HDiv.hDiv (HAdd.hAdd …
  -/
  nth_rewrite 3 [show 4 = 2 * 2 by omega]
  rw [mul_assoc, add_comm, Nat.add_mul_div_left _ _ zero_lt_two, pow_add, pow_mul,
    neg_one_sq, one_pow, mul_one]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq (ite (Eq (HMod.hMod n 4) 1) 1 (-1)) (HPow.hPow (-1) (HDiv.hDiv (HMod.hMod …
  -/
  have help : ∀ m : ℕ, m < 4 → m % 2 = 1 → ite (m = 1) (1 : ℤ) (-1) = (-1) ^ (m / 2) := by decide
  /-
    n : Nat
    hn : Eq (HMod.hMod n 2) 1
    help : ∀ (m : Nat), LT.lt m 4 → Eq (HMod.hMod m 2) 1 → Eq (ite (Eq m 1) 1 (-1) …
    ⊢ Eq (ite (Eq (HMod.hMod n 4) 1) 1 (-1)) (HPow.hPow (-1) (HDiv.hDiv (HMod.hMod …
  -/
  exact help _ (Nat.mod_lt n (by omega)) <| (Nat.mod_mod_of_dvd n (by omega : 2 ∣ 4)).trans hn
  /-
    🎉 no goals
  -/


/-- If `n % 4 = 1`, then `χ₄ n = 1`. -/
theorem χ₄_nat_one_mod_four {n : ℕ} (hn : n % 4 = 1) : χ₄ n = 1 := by
  /-
    n : Nat
    hn : Eq (HMod.hMod n 4) 1
    ⊢ Eq (ZMod.χ₄ ↑n) 1
  -/
  rw [χ₄_nat_mod_four, hn]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 4) 1
    ⊢ Eq (ZMod.χ₄ ↑1) 1
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `n % 4 = 3`, then `χ₄ n = -1`. -/
theorem χ₄_nat_three_mod_four {n : ℕ} (hn : n % 4 = 3) : χ₄ n = -1 := by
  /-
    n : Nat
    hn : Eq (HMod.hMod n 4) 3
    ⊢ Eq (ZMod.χ₄ ↑n) (-1)
  -/
  rw [χ₄_nat_mod_four, hn]
  /-
    n : Nat
    hn : Eq (HMod.hMod n 4) 3
    ⊢ Eq (ZMod.χ₄ ↑3) (-1)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `n % 4 = 1`, then `χ₄ n = 1`. -/
theorem χ₄_int_one_mod_four {n : ℤ} (hn : n % 4 = 1) : χ₄ n = 1 := by
  /-
    n : Int
    hn : Eq (HMod.hMod n 4) 1
    ⊢ Eq (ZMod.χ₄ ↑n) 1
  -/
  rw [χ₄_int_mod_four, hn]
  /-
    n : Int
    hn : Eq (HMod.hMod n 4) 1
    ⊢ Eq (ZMod.χ₄ ↑1) 1
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `n % 4 = 3`, then `χ₄ n = -1`. -/
theorem χ₄_int_three_mod_four {n : ℤ} (hn : n % 4 = 3) : χ₄ n = -1 := by
  /-
    n : Int
    hn : Eq (HMod.hMod n 4) 3
    ⊢ Eq (ZMod.χ₄ ↑n) (-1)
  -/
  rw [χ₄_int_mod_four, hn]
  /-
    n : Int
    hn : Eq (HMod.hMod n 4) 3
    ⊢ Eq (ZMod.χ₄ ↑3) (-1)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `n % 4 = 1`, then `(-1)^(n/2) = 1`. -/
theorem neg_one_pow_div_two_of_one_mod_four {n : ℕ} (hn : n % 4 = 1) : (-1 : ℤ) ^ (n / 2) = 1 :=
  χ₄_eq_neg_one_pow (Nat.odd_of_mod_four_eq_one hn) ▸ χ₄_nat_one_mod_four hn


/-- If `n % 4 = 3`, then `(-1)^(n/2) = -1`. -/
theorem neg_one_pow_div_two_of_three_mod_four {n : ℕ} (hn : n % 4 = 3) : (-1 : ℤ) ^ (n / 2) = -1 :=
  χ₄_eq_neg_one_pow (Nat.odd_of_mod_four_eq_three hn) ▸ χ₄_nat_three_mod_four hn


/-- Define the first primitive quadratic character on `ZMod 8`, `χ₈`.
It corresponds to the extension `ℚ(√2)/ℚ`. -/
@[simps]
def χ₈ : MulChar (ZMod 8) ℤ where
  toFun a :=
    match a with
    | 0 | 2 | 4 | 6 => 0
    | 1 | 7 => 1
    | 3 | 5 => -1
  map_one' := rfl
                 /-
                   ⊢ ∀ (x y : ZMod 8), Eq ({ toFun := fun a => ZMod.χ₈.match_1 (fun a => Int) a ( …
                 -/
  map_mul' := by decide
                 /-
                   🎉 no goals
                 -/
                     /-
                       ⊢ ∀ (a : ZMod 8), Not (IsUnit a) → Eq ((↑{ toFun := fun a => ZMod.χ₈.match_1 ( …
                     -/
  map_nonunit' := by decide
                     /-
                       🎉 no goals
                     -/


/-- `χ₈` takes values in `{0, 1, -1}` -/
theorem isQuadratic_χ₈ : χ₈.IsQuadratic := by
  /-
    ⊢ ZMod.χ₈.IsQuadratic
  -/
  unfold MulChar.IsQuadratic
  /-
    ⊢ ∀ (a : ZMod 8), Or (Eq (ZMod.χ₈ a) 0) (Or (Eq (ZMod.χ₈ a) 1) (Eq (ZMod.χ₈ a) …
  -/
  decide
  /-
    🎉 no goals
  -/


/-- The value of `χ₈ n`, for `n : ℕ`, depends only on `n % 8`. -/
theorem χ₈_nat_mod_eight (n : ℕ) : χ₈ n = χ₈ (n % 8 : ℕ) := by
  /-
    n : Nat
    ⊢ Eq (ZMod.χ₈ ↑n) (ZMod.χ₈ ↑(HMod.hMod n 8))
  -/
  rw [← ZMod.natCast_mod n 8]
  /-
    🎉 no goals
  -/


/-- The value of `χ₈ n`, for `n : ℤ`, depends only on `n % 8`. -/
theorem χ₈_int_mod_eight (n : ℤ) : χ₈ n = χ₈ (n % 8 : ℤ) := by
  /-
    n : Int
    ⊢ Eq (ZMod.χ₈ ↑n) (ZMod.χ₈ ↑(HMod.hMod n 8))
  -/
  rw [← ZMod.intCast_mod n 8, Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


/-- An explicit description of `χ₈` on integers / naturals -/
theorem χ₈_int_eq_if_mod_eight (n : ℤ) :
    χ₈ n = if n % 2 = 0 then 0 else if n % 8 = 1 ∨ n % 8 = 7 then 1 else -1 := by
  have help :
    ∀ m : ℤ, 0 ≤ m → m < 8 → χ₈ m = if m % 2 = 0 then 0 else if m = 1 ∨ m = 7 then 1 else -1 := by
    decide
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 8 → Eq (ZMod.χ₈ ↑m) (ite (Eq (HMod.hMo …
    ⊢ Eq (ZMod.χ₈ ↑n) (ite (Eq (HMod.hMod n 2) 0) 0 (ite (Or (Eq (HMod.hMod n 8) 1 …
  -/
  rw [← Int.emod_emod_of_dvd n (by omega : (2 : ℤ) ∣ 8), ← ZMod.intCast_mod n 8]
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 8 → Eq (ZMod.χ₈ ↑m) (ite (Eq (HMod.hMo …
    ⊢ Eq (ZMod.χ₈ ↑(HMod.hMod n ↑8)) (ite (Eq (HMod.hMod (HMod.hMod n 8) 2) 0) 0 ( …
  -/
  exact help (n % 8) (Int.emod_nonneg n (by omega)) (Int.emod_lt n (by omega))
  /-
    🎉 no goals
  -/


theorem χ₈_nat_eq_if_mod_eight (n : ℕ) :
    χ₈ n = if n % 2 = 0 then 0 else if n % 8 = 1 ∨ n % 8 = 7 then 1 else -1 :=
  mod_cast χ₈_int_eq_if_mod_eight n


/-- Define the second primitive quadratic character on `ZMod 8`, `χ₈'`.
It corresponds to the extension `ℚ(√-2)/ℚ`. -/
@[simps]
def χ₈' : MulChar (ZMod 8) ℤ where
  toFun a :=
    match a with
    | 0 | 2 | 4 | 6 => 0
    | 1 | 3 => 1
    | 5 | 7 => -1
  map_one' := rfl
                 /-
                   ⊢ ∀ (x y : ZMod 8), Eq ({ toFun := fun a => ZMod.χ₈'.match_1 (fun a => Int) a  …
                 -/
  map_mul' := by decide
                 /-
                   🎉 no goals
                 -/
                     /-
                       ⊢ ∀ (a : ZMod 8), Not (IsUnit a) → Eq ((↑{ toFun := fun a => ZMod.χ₈'.match_1  …
                     -/
  map_nonunit' := by decide
                     /-
                       🎉 no goals
                     -/


/-- `χ₈'` takes values in `{0, 1, -1}` -/
theorem isQuadratic_χ₈' : χ₈'.IsQuadratic := by
  /-
    ⊢ ZMod.χ₈'.IsQuadratic
  -/
  unfold MulChar.IsQuadratic
  /-
    ⊢ ∀ (a : ZMod 8), Or (Eq (ZMod.χ₈' a) 0) (Or (Eq (ZMod.χ₈' a) 1) (Eq (ZMod.χ₈' …
  -/
  decide
  /-
    🎉 no goals
  -/


/-- An explicit description of `χ₈'` on integers / naturals -/
theorem χ₈'_int_eq_if_mod_eight (n : ℤ) :
    χ₈' n = if n % 2 = 0 then 0 else if n % 8 = 1 ∨ n % 8 = 3 then 1 else -1 := by
  have help :
    ∀ m : ℤ, 0 ≤ m → m < 8 → χ₈' m = if m % 2 = 0 then 0 else if m = 1 ∨ m = 3 then 1 else -1 := by
    decide
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 8 → Eq (ZMod.χ₈' ↑m) (ite (Eq (HMod.hM …
    ⊢ Eq (ZMod.χ₈' ↑n) (ite (Eq (HMod.hMod n 2) 0) 0 (ite (Or (Eq (HMod.hMod n 8)  …
  -/
  rw [← Int.emod_emod_of_dvd n (by omega : (2 : ℤ) ∣ 8), ← ZMod.intCast_mod n 8]
  /-
    n : Int
    help : ∀ (m : Int), LE.le 0 m → LT.lt m 8 → Eq (ZMod.χ₈' ↑m) (ite (Eq (HMod.hM …
    ⊢ Eq (ZMod.χ₈' ↑(HMod.hMod n ↑8)) (ite (Eq (HMod.hMod (HMod.hMod n 8) 2) 0) 0  …
  -/
  exact help (n % 8) (Int.emod_nonneg n (by omega)) (Int.emod_lt n (by omega))
  /-
    🎉 no goals
  -/


theorem χ₈'_nat_eq_if_mod_eight (n : ℕ) :
    χ₈' n = if n % 2 = 0 then 0 else if n % 8 = 1 ∨ n % 8 = 3 then 1 else -1 :=
  mod_cast χ₈'_int_eq_if_mod_eight n


/-- The relation between `χ₄`, `χ₈` and `χ₈'` -/
theorem χ₈'_eq_χ₄_mul_χ₈ : ∀ a : ZMod 8, χ₈' a = χ₄ (cast a) * χ₈ a := by
  /-
    ⊢ ∀ (a : ZMod 8), Eq (ZMod.χ₈' a) (HMul.hMul (ZMod.χ₄ a.cast) (ZMod.χ₈ a))
  -/
  decide
  /-
    🎉 no goals
  -/


theorem χ₈'_int_eq_χ₄_mul_χ₈ (a : ℤ) : χ₈' a = χ₄ a * χ₈ a := by
  /-
    a : Int
    ⊢ Eq (ZMod.χ₈' ↑a) (HMul.hMul (ZMod.χ₄ ↑a) (ZMod.χ₈ ↑a))
  -/
  rw [← @cast_intCast 8 (ZMod 4) _ 4 _ (by omega) a]
  /-
    a : Int
    ⊢ Eq (ZMod.χ₈' ↑a) (HMul.hMul (ZMod.χ₄ (↑a).cast) (ZMod.χ₈ ↑a))
  -/
  exact χ₈'_eq_χ₄_mul_χ₈ a
  /-
    🎉 no goals
  -/


