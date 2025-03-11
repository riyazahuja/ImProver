/-- the probabilists' Hermite polynomials. -/
noncomputable def hermite : ℕ → Polynomial ℤ
  | 0 => 1
  | n + 1 => X * hermite n - derivative (hermite n)


/-- The recursion `hermite (n+1) = (x - d/dx) (hermite n)` -/
@[simp]
theorem hermite_succ (n : ℕ) : hermite (n + 1) = X * hermite n - derivative (hermite n) := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.hermite (HAdd.hAdd n 1)) (HSub.hSub (HMul.hMul Polynomial.X ( …
  -/
  rw [hermite]
  /-
    🎉 no goals
  -/


theorem hermite_eq_iterate (n : ℕ) : hermite n = (fun p => X * p - derivative p)^[n] 1 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Function.iterate_succ_apply', ← ih, hermite_succ]


@[simp]
theorem hermite_zero : hermite 0 = C 1 :=
  rfl


theorem hermite_one : hermite 1 = X := by
  /-
    ⊢ Eq (Polynomial.hermite 1) Polynomial.X
  -/
  rw [hermite_succ, hermite_zero]
  /-
    ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.C 1)) (Polynomial.derivati …
  -/
  simp only [map_one, mul_one, derivative_one, sub_zero]
  /-
    🎉 no goals
  -/


theorem coeff_hermite_succ_zero (n : ℕ) : coeff (hermite (n + 1)) 0 = -coeff (hermite n) 1 := by
  /-
    n : Nat
    ⊢ Eq ((Polynomial.hermite (HAdd.hAdd n 1)).coeff 0) (Neg.neg ((Polynomial.herm …
  -/
  simp [coeff_derivative]
  /-
    🎉 no goals
  -/


theorem coeff_hermite_succ_succ (n k : ℕ) : coeff (hermite (n + 1)) (k + 1) =
    coeff (hermite n) k - (k + 2) * coeff (hermite n) (k + 2) := by
  /-
    n k : Nat
    ⊢ Eq ((Polynomial.hermite (HAdd.hAdd n 1)).coeff (HAdd.hAdd k 1)) (HSub.hSub ( …
  -/
  rw [hermite_succ, coeff_sub, coeff_X_mul, coeff_derivative, mul_comm]
  /-
    n k : Nat
    ⊢ Eq (HSub.hSub ((Polynomial.hermite n).coeff k) (HMul.hMul (HAdd.hAdd (↑(HAdd …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem coeff_hermite_of_lt {n k : ℕ} (hnk : n < k) : coeff (hermite n) k = 0 := by
  /-
    n k : Nat
    hnk : LT.lt n k
    ⊢ Eq ((Polynomial.hermite n).coeff k) 0
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hnk
  /-
    case intro
    n k : Nat
    hnk : LT.lt n (HAdd.hAdd (HAdd.hAdd n k) 1)
    ⊢ Eq ((Polynomial.hermite n).coeff (HAdd.hAdd (HAdd.hAdd n k) 1)) 0
  -/
  clear hnk
  induction n generalizing k with
  | zero => exact coeff_C
  | succ n ih =>
    have : n + k + 1 + 2 = n + (k + 2) + 1 := by ring
    rw [coeff_hermite_succ_succ, add_right_comm, this, ih k, ih (k + 2), mul_zero, sub_zero]


@[simp]
theorem coeff_hermite_self (n : ℕ) : coeff (hermite n) n = 1 := by
  induction n with
  | zero => exact coeff_C
  | succ n ih =>
    rw [coeff_hermite_succ_succ, ih, coeff_hermite_of_lt, mul_zero, sub_zero]
    simp


@[simp]
theorem degree_hermite (n : ℕ) : (hermite n).degree = n := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.hermite n).degree ↑n
  -/
  rw [degree_eq_of_le_of_coeff_ne_zero]
    /-
      case pn
      n : Nat
      ⊢ LE.le (Polynomial.hermite n).degree ↑n
    -/
  · simp_rw [degree_le_iff_coeff_zero, Nat.cast_lt]
    /-
      case pn
      n : Nat
      ⊢ ∀ (m : Nat), LT.lt n m → Eq ((Polynomial.hermite n).coeff m) 0
    -/
    rintro m hnm
    /-
      case pn
      n m : Nat
      hnm : LT.lt n m
      ⊢ Eq ((Polynomial.hermite n).coeff m) 0
    -/
    exact coeff_hermite_of_lt hnm
    /-
      🎉 no goals
    -/
    /-
      case p1
      n : Nat
      ⊢ Ne ((Polynomial.hermite n).coeff n) 0
    -/
  · simp [coeff_hermite_self n]
    /-
      🎉 no goals
    -/


@[simp]
theorem natDegree_hermite {n : ℕ} : (hermite n).natDegree = n :=
  natDegree_eq_of_degree_eq_some (degree_hermite n)


@[simp]
theorem leadingCoeff_hermite (n : ℕ) : (hermite n).leadingCoeff = 1 := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.hermite n).leadingCoeff 1
  -/
  rw [← coeff_natDegree, natDegree_hermite, coeff_hermite_self]
  /-
    🎉 no goals
  -/


theorem hermite_monic (n : ℕ) : (hermite n).Monic :=
  leadingCoeff_hermite n


theorem coeff_hermite_of_odd_add {n k : ℕ} (hnk : Odd (n + k)) : coeff (hermite n) k = 0 := by
  induction n generalizing k with
  | zero =>
    rw [zero_add k] at hnk
    exact coeff_hermite_of_lt hnk.pos
  | succ n ih =>
    cases k with
    | zero =>
      rw [Nat.succ_add_eq_add_succ] at hnk
      rw [coeff_hermite_succ_zero, ih hnk, neg_zero]
    | succ k =>
      rw [coeff_hermite_succ_succ, ih, ih, mul_zero, sub_zero]
      · rwa [Nat.succ_add_eq_add_succ] at hnk
      · rw [(by rw [Nat.succ_add, Nat.add_succ] : n.succ + k.succ = n + k + 2)] at hnk
        exact (Nat.odd_add.mp hnk).mpr even_two


/-- Because of `coeff_hermite_of_odd_add`, every nonzero coefficient is described as follows. -/
theorem coeff_hermite_explicit :
    ∀ n k : ℕ, coeff (hermite (2 * n + k)) k = (-1) ^ n * (2 * n - 1)‼ * Nat.choose (2 * n + k) k
               /-
                 x✝ : Nat
                 ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 0) x✝)).coeff x✝) (HMul.hMul …
               -/
  | 0, _ => by simp
               /-
                 🎉 no goals
               -/
  | n + 1, 0 => by
    /-
      n : Nat
      ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) 0)).coeff 0 …
    -/
    convert coeff_hermite_succ_zero (2 * n + 1) using 1
    -- Porting note: ring_nf did not solve the goal on line 165
    rw [coeff_hermite_explicit n 1, (by rw [Nat.left_distrib, mul_one, Nat.add_one_sub_one] :
      2 * (n + 1) - 1 = 2 * n + 1), Nat.doubleFactorial_add_one, Nat.choose_zero_right,
      Nat.choose_one_right, pow_succ]
    /-
      case h.e'_3
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) n) (-1)) ↑(HMul.hMul (HA …
    -/
    push_cast
    /-
      case h.e'_3
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) n) (-1)) (HMul.hMul (HAd …
    -/
    ring
    /-
      🎉 no goals
    -/
  | n + 1, k + 1 => by
    let hermite_explicit : ℕ → ℕ → ℤ := fun n k =>
      (-1) ^ n * (2 * n - 1)‼ * Nat.choose (2 * n + k) k
    have hermite_explicit_recur :
      ∀ n k : ℕ,
        hermite_explicit (n + 1) (k + 1) =
          hermite_explicit (n + 1) k - (k + 2) * hermite_explicit n (k + 2) := by
      intro n k
      simp only [hermite_explicit]
      -- Factor out (-1)'s.
      rw [mul_comm (↑k + _ : ℤ), sub_eq_add_neg]
      nth_rw 3 [neg_eq_neg_one_mul]
      simp only [mul_assoc, ← mul_add, pow_succ']
      congr 2
      -- Factor out double factorials.
      norm_cast
      -- Porting note: ring_nf did not solve the goal on line 186
      rw [(by rw [Nat.left_distrib, mul_one, Nat.add_one_sub_one] : 2 * (n + 1) - 1 = 2 * n + 1),
        Nat.doubleFactorial_add_one, mul_comm (2 * n + 1)]
      simp only [mul_assoc, ← mul_add]
      congr 1
      -- Match up binomial coefficients using `Nat.choose_succ_right_eq`.
      rw [(by ring : 2 * (n + 1) + (k + 1) = 2 * n + 1 + (k + 1) + 1),
        (by ring : 2 * (n + 1) + k = 2 * n + 1 + (k + 1)),
        (by ring : 2 * n + (k + 2) = 2 * n + 1 + (k + 1))]
      rw [Nat.choose, Nat.choose_succ_right_eq (2 * n + 1 + (k + 1)) (k + 1), Nat.add_sub_cancel]
      ring
    /-
      n k : Nat
      hermite_explicit : Nat → Nat → Int := fun n k => HMul.hMul (HMul.hMul (HPow.hP …
      hermite_explicit_recur : ∀ (n k : Nat), Eq (hermite_explicit (HAdd.hAdd n 1) ( …
      ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) (HAdd.hAdd  …
    -/
    change _ = hermite_explicit _ _
    /-
      n k : Nat
      hermite_explicit : Nat → Nat → Int := fun n k => HMul.hMul (HMul.hMul (HPow.hP …
      hermite_explicit_recur : ∀ (n k : Nat), Eq (hermite_explicit (HAdd.hAdd n 1) ( …
      ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) (HAdd.hAdd  …
    -/
    rw [← add_assoc, coeff_hermite_succ_succ, hermite_explicit_recur]
    /-
      n k : Nat
      hermite_explicit : Nat → Nat → Int := fun n k => HMul.hMul (HMul.hMul (HPow.hP …
      hermite_explicit_recur : ∀ (n k : Nat), Eq (hermite_explicit (HAdd.hAdd n 1) ( …
      ⊢ Eq (HSub.hSub ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1))  …
    -/
    congr
      /-
        case e_a
        n k : Nat
        hermite_explicit : Nat → Nat → Int := fun n k => HMul.hMul (HMul.hMul (HPow.hP …
        hermite_explicit_recur : ∀ (n k : Nat), Eq (hermite_explicit (HAdd.hAdd n 1) ( …
        ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) k)).coeff k …
      -/
    · rw [coeff_hermite_explicit (n + 1) k]
      /-
        🎉 no goals
      -/
      /-
        case e_a.e_a
        n k : Nat
        hermite_explicit : Nat → Nat → Int := fun n k => HMul.hMul (HMul.hMul (HPow.hP …
        hermite_explicit_recur : ∀ (n k : Nat), Eq (hermite_explicit (HAdd.hAdd n 1) ( …
        ⊢ Eq ((Polynomial.hermite (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) k)).coeff ( …
      -/
    · rw [(by ring : 2 * (n + 1) + k = 2 * n + (k + 2)), coeff_hermite_explicit n (k + 2)]
      /-
        🎉 no goals
      -/


theorem coeff_hermite_of_even_add {n k : ℕ} (hnk : Even (n + k)) :
    coeff (hermite n) k = (-1) ^ ((n - k) / 2) * (n - k - 1)‼ * Nat.choose n k := by
  /-
    n k : Nat
    hnk : Even (HAdd.hAdd n k)
    ⊢ Eq ((Polynomial.hermite n).coeff k) (HMul.hMul (HMul.hMul (HPow.hPow (-1) (H …
  -/
  rcases le_or_lt k n with h_le | h_lt
    /-
      case inl
      n k : Nat
      hnk : Even (HAdd.hAdd n k)
      h_le : LE.le k n
      ⊢ Eq ((Polynomial.hermite n).coeff k) (HMul.hMul (HMul.hMul (HPow.hPow (-1) (H …
    -/
  · rw [Nat.even_add, ← Nat.even_sub h_le] at hnk
    /-
      case inl
      n k : Nat
      hnk : Even (HSub.hSub n k)
      h_le : LE.le k n
      ⊢ Eq ((Polynomial.hermite n).coeff k) (HMul.hMul (HMul.hMul (HPow.hPow (-1) (H …
    -/
    obtain ⟨m, hm⟩ := hnk
    -- Porting note: linarith failed to find a contradiction by itself
    rw [(by omega : n = 2 * m + k),
      Nat.add_sub_cancel, Nat.mul_div_cancel_left _ (Nat.succ_pos 1), coeff_hermite_explicit]
    /-
      case inr
      n k : Nat
      hnk : Even (HAdd.hAdd n k)
      h_lt : LT.lt n k
      ⊢ Eq ((Polynomial.hermite n).coeff k) (HMul.hMul (HMul.hMul (HPow.hPow (-1) (H …
    -/
  · simp [Nat.choose_eq_zero_of_lt h_lt, coeff_hermite_of_lt h_lt]
    /-
      🎉 no goals
    -/


theorem coeff_hermite (n k : ℕ) :
    coeff (hermite n) k =
      if Even (n + k) then (-1 : ℤ) ^ ((n - k) / 2) * (n - k - 1)‼ * Nat.choose n k else 0 := by
  /-
    n k : Nat
    ⊢ Eq ((Polynomial.hermite n).coeff k) (ite (Even (HAdd.hAdd n k)) (HMul.hMul ( …
  -/
  split_ifs with h
    /-
      case pos
      n k : Nat
      h : Even (HAdd.hAdd n k)
      ⊢ Eq ((Polynomial.hermite n).coeff k) (HMul.hMul (HMul.hMul (HPow.hPow (-1) (H …
    -/
  · exact coeff_hermite_of_even_add h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n k : Nat
      h : Not (Even (HAdd.hAdd n k))
      ⊢ Eq ((Polynomial.hermite n).coeff k) 0
    -/
  · exact coeff_hermite_of_odd_add (Nat.not_even_iff_odd.1 h)
    /-
      🎉 no goals
    -/


