local notation "M" => Matrix n' n' R


noncomputable instance : DivInvMonoid M :=
                     /-
                       n' : Type u_1
                       inst✝² : DecidableEq n'
                       inst✝¹ : Fintype n'
                       R : Type u_2
                       inst✝ : CommRing R
                       ⊢ Monoid (Matrix n' n' R)
                     -/
                     /-
                       🎉 no goals
                     -/
  { show Monoid M by infer_instance, show Inv M by infer_instance with }
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem inv_pow' (A : M) (n : ℕ) : A⁻¹ ^ n = (A ^ n)⁻¹ := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    n : Nat
    ⊢ Eq (HPow.hPow (Inv.inv A) n) (Inv.inv (HPow.hPow A n))
  -/
  induction' n with n ih
    /-
      case zero
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      ⊢ Eq (HPow.hPow (Inv.inv A) 0) (Inv.inv (HPow.hPow A 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n : Nat
      ih : Eq (HPow.hPow (Inv.inv A) n) (Inv.inv (HPow.hPow A n))
      ⊢ Eq (HPow.hPow (Inv.inv A) (HAdd.hAdd n 1)) (Inv.inv (HPow.hPow A (HAdd.hAdd  …
    -/
  · rw [pow_succ A, mul_inv_rev, ← ih, ← pow_succ']
    /-
      🎉 no goals
    -/


theorem pow_sub' (A : M) {m n : ℕ} (ha : IsUnit A.det) (h : n ≤ m) :
    A ^ (m - n) = A ^ m * (A ^ n)⁻¹ := by
  rw [← tsub_add_cancel_of_le h, pow_add, Matrix.mul_assoc, mul_nonsing_inv,
    tsub_add_cancel_of_le h, Matrix.mul_one]
  /-
    case h
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    m n : Nat
    ha : IsUnit A.det
    h : LE.le n m
    ⊢ IsUnit (HPow.hPow A n).det
  -/
  simpa using ha.pow n
  /-
    🎉 no goals
  -/


theorem pow_inv_comm' (A : M) (m n : ℕ) : A⁻¹ ^ m * A ^ n = A ^ n * A⁻¹ ^ m := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    m n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A n)) (HMul.hMul (HPow.hP …
  -/
  induction' n with n IH generalizing m
    /-
      case zero
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      m : Nat
      ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A 0)) (HMul.hMul (HPow.hP …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    n : Nat
    IH : ∀ (m : Nat), Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A n)) (HM …
    m : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A (HAdd.hAdd n 1))) (HMul …
  -/
  cases' m with m m
    /-
      case succ.zero
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n : Nat
      IH : ∀ (m : Nat), Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A n)) (HM …
      ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) 0) (HPow.hPow A (HAdd.hAdd n 1))) (HMul …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    n : Nat
    IH : ∀ (m : Nat), Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A n)) (HM …
    m : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) (HAdd.hAdd m 1)) (HPow.hPow A (HAdd.hAd …
  -/
  rcases nonsing_inv_cancel_or_zero A with (⟨h, h'⟩ | h)
  · calc
       A⁻¹ ^ (m + 1) * A ^ (n + 1) = A⁻¹ ^ m * (A⁻¹ * A) * A ^ n := by
        simp only [pow_succ A⁻¹, pow_succ' A, Matrix.mul_assoc]
      _ = A ^ n * A⁻¹ ^ m := by simp only [h, Matrix.mul_one, Matrix.one_mul, IH m]
      _ = A ^ n * (A * A⁻¹) * A⁻¹ ^ m := by simp only [h', Matrix.mul_one, Matrix.one_mul]
      _ = A ^ (n + 1) * A⁻¹ ^ (m + 1) := by
        simp only [pow_succ A, pow_succ' A⁻¹, Matrix.mul_assoc]
    /-
      case succ.succ.inr
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n : Nat
      IH : ∀ (m : Nat), Eq (HMul.hMul (HPow.hPow (Inv.inv A) m) (HPow.hPow A n)) (HM …
      m : Nat
      h : Eq (Inv.inv A) 0
      ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv A) (HAdd.hAdd m 1)) (HPow.hPow A (HAdd.hAd …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


@[simp]
theorem one_zpow : ∀ n : ℤ, (1 : M) ^ n = 1
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    n : Nat
                    ⊢ Eq (HPow.hPow 1 ↑n) 1
                  -/
  | (n : ℕ) => by rw [zpow_natCast, one_pow]
                  /-
                    🎉 no goals
                  -/
                 /-
                   n' : Type u_1
                   inst✝² : DecidableEq n'
                   inst✝¹ : Fintype n'
                   R : Type u_2
                   inst✝ : CommRing R
                   n : Nat
                   ⊢ Eq (HPow.hPow 1 (Int.negSucc n)) 1
                 -/
  | -[n+1] => by rw [zpow_negSucc, one_pow, inv_one]
                 /-
                   🎉 no goals
                 -/


theorem zero_zpow : ∀ z : ℤ, z ≠ 0 → (0 : M) ^ z = 0
  | (n : ℕ), h => by
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      n : Nat
      h : Ne (↑n) 0
      ⊢ Eq (HPow.hPow 0 ↑n) 0
    -/
    rw [zpow_natCast, zero_pow]
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      n : Nat
      h : Ne (↑n) 0
      ⊢ Ne n 0
    -/
    exact mod_cast h
    /-
      🎉 no goals
    -/
                    /-
                      n' : Type u_1
                      inst✝² : DecidableEq n'
                      inst✝¹ : Fintype n'
                      R : Type u_2
                      inst✝ : CommRing R
                      n : Nat
                      x✝ : Ne (Int.negSucc n) 0
                      ⊢ Eq (HPow.hPow 0 (Int.negSucc n)) 0
                    -/
  | -[n+1], _ => by simp [zero_pow n.succ_ne_zero]
                    /-
                      🎉 no goals
                    -/


theorem zero_zpow_eq (n : ℤ) : (0 : M) ^ n = if n = 0 then 1 else 0 := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    n : Int
    ⊢ Eq (HPow.hPow 0 n) (ite (Eq n 0) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      n : Int
      h : Eq n 0
      ⊢ Eq (HPow.hPow 0 n) 1
    -/
  · rw [h, zpow_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      n : Int
      h : Not (Eq n 0)
      ⊢ Eq (HPow.hPow 0 n) 0
    -/
  · rw [zero_zpow _ h]
    /-
      🎉 no goals
    -/


theorem inv_zpow (A : M) : ∀ n : ℤ, A⁻¹ ^ n = (A ^ n)⁻¹
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    A : Matrix n' n' R
                    n : Nat
                    ⊢ Eq (HPow.hPow (Inv.inv A) ↑n) (Inv.inv (HPow.hPow A ↑n))
                  -/
  | (n : ℕ) => by rw [zpow_natCast, zpow_natCast, inv_pow']
                  /-
                    🎉 no goals
                  -/
                 /-
                   n' : Type u_1
                   inst✝² : DecidableEq n'
                   inst✝¹ : Fintype n'
                   R : Type u_2
                   inst✝ : CommRing R
                   A : Matrix n' n' R
                   n : Nat
                   ⊢ Eq (HPow.hPow (Inv.inv A) (Int.negSucc n)) (Inv.inv (HPow.hPow A (Int.negSuc …
                 -/
  | -[n+1] => by rw [zpow_negSucc, zpow_negSucc, inv_pow']
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem zpow_neg_one (A : M) : A ^ (-1 : ℤ) = A⁻¹ := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    ⊢ Eq (HPow.hPow A (-1)) (Inv.inv A)
  -/
  convert DivInvMonoid.zpow_neg' 0 A
  /-
    case h.e'_3.h.e'_3
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    ⊢ Eq A (DivInvMonoid.zpow (↑(Nat.succ 0)) A)
  -/
  simp only [zpow_one, Int.ofNat_zero, Int.ofNat_succ, zpow_eq_pow, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem zpow_neg_natCast (A : M) (n : ℕ) : A ^ (-n : ℤ) = (A ^ n)⁻¹ := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    n : Nat
    ⊢ Eq (HPow.hPow A (Neg.neg ↑n)) (Inv.inv (HPow.hPow A n))
  -/
  cases n
    /-
      case zero
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      ⊢ Eq (HPow.hPow A (Neg.neg ↑0)) (Inv.inv (HPow.hPow A 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n✝ : Nat
      ⊢ Eq (HPow.hPow A (Neg.neg ↑(HAdd.hAdd n✝ 1))) (Inv.inv (HPow.hPow A (HAdd.hAd …
    -/
  · exact DivInvMonoid.zpow_neg' _ _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-05")] alias zpow_neg_coe_nat := zpow_neg_natCast


theorem _root_.IsUnit.det_zpow {A : M} (h : IsUnit A.det) (n : ℤ) : IsUnit (A ^ n).det := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    h : IsUnit A.det
    n : Int
    ⊢ IsUnit (HPow.hPow A n).det
  -/
  cases' n with n n
    /-
      case ofNat
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      n : Nat
      ⊢ IsUnit (HPow.hPow A (Int.ofNat n)).det
    -/
  · simpa using h.pow n
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      n : Nat
      ⊢ IsUnit (HPow.hPow A (Int.negSucc n)).det
    -/
  · simpa using h.pow n.succ
    /-
      🎉 no goals
    -/


theorem isUnit_det_zpow_iff {A : M} {z : ℤ} : IsUnit (A ^ z).det ↔ IsUnit A.det ∨ z = 0 := by
  induction z using Int.induction_on with
  | hz => simp
  | hp z =>
    rw [← Int.ofNat_succ, zpow_natCast, det_pow, isUnit_pow_succ_iff, ← Int.ofNat_zero,
      Int.ofNat_inj]
    simp
  | hn z =>
    rw [← neg_add', ← Int.ofNat_succ, zpow_neg_natCast, isUnit_nonsing_inv_det_iff, det_pow,
      isUnit_pow_succ_iff, neg_eq_zero, ← Int.ofNat_zero, Int.ofNat_inj]
    simp


theorem zpow_neg {A : M} (h : IsUnit A.det) : ∀ n : ℤ, A ^ (-n) = (A ^ n)⁻¹
  | (n : ℕ) => zpow_neg_natCast _ _
  | -[n+1] => by
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      n : Nat
      ⊢ Eq (HPow.hPow A (Neg.neg (Int.negSucc n))) (Inv.inv (HPow.hPow A (Int.negSuc …
    -/
    rw [zpow_negSucc, neg_negSucc, zpow_natCast, nonsing_inv_nonsing_inv]
    /-
      case h
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      n : Nat
      ⊢ IsUnit (HPow.hPow A (HAdd.hAdd n 1)).det
    -/
    rw [det_pow]
    /-
      case h
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      n : Nat
      ⊢ IsUnit (HPow.hPow A.det (HAdd.hAdd n 1))
    -/
    exact h.pow _
    /-
      🎉 no goals
    -/


theorem inv_zpow' {A : M} (h : IsUnit A.det) (n : ℤ) : A⁻¹ ^ n = A ^ (-n) := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    h : IsUnit A.det
    n : Int
    ⊢ Eq (HPow.hPow (Inv.inv A) n) (HPow.hPow A (Neg.neg n))
  -/
  rw [zpow_neg h, inv_zpow]
  /-
    🎉 no goals
  -/


theorem zpow_add_one {A : M} (h : IsUnit A.det) : ∀ n : ℤ, A ^ (n + 1) = A ^ n * A
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    A : Matrix n' n' R
                    h : IsUnit A.det
                    n : Nat
                    ⊢ Eq (HPow.hPow A (HAdd.hAdd (↑n) 1)) (HMul.hMul (HPow.hPow A ↑n) A)
                  -/
  | (n : ℕ) => by simp only [← Nat.cast_succ, pow_succ, zpow_natCast]
                  /-
                    🎉 no goals
                  -/
  | -[n+1] =>
    calc
      A ^ (-(n + 1) + 1 : ℤ) = (A ^ n)⁻¹ := by
        /-
          n' : Type u_1
          inst✝² : DecidableEq n'
          inst✝¹ : Fintype n'
          R : Type u_2
          inst✝ : CommRing R
          A : Matrix n' n' R
          h : IsUnit A.det
          n : Nat
          ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑n) 1)) 1)) (Inv.inv (HPow.h …
        -/
        rw [neg_add, neg_add_cancel_right, zpow_neg h, zpow_natCast]
        /-
          🎉 no goals
        -/
      _ = (A * A ^ n)⁻¹ * A := by
        /-
          n' : Type u_1
          inst✝² : DecidableEq n'
          inst✝¹ : Fintype n'
          R : Type u_2
          inst✝ : CommRing R
          A : Matrix n' n' R
          h : IsUnit A.det
          n : Nat
          ⊢ Eq (Inv.inv (HPow.hPow A n)) (HMul.hMul (Inv.inv (HMul.hMul A (HPow.hPow A n …
        -/
        rw [mul_inv_rev, Matrix.mul_assoc, nonsing_inv_mul _ h, Matrix.mul_one]
        /-
          🎉 no goals
        -/
      _ = A ^ (-(n + 1 : ℤ)) * A := by
        /-
          n' : Type u_1
          inst✝² : DecidableEq n'
          inst✝¹ : Fintype n'
          R : Type u_2
          inst✝ : CommRing R
          A : Matrix n' n' R
          h : IsUnit A.det
          n : Nat
          ⊢ Eq (HMul.hMul (Inv.inv (HMul.hMul A (HPow.hPow A n))) A) (HMul.hMul (HPow.hP …
        -/
        rw [zpow_neg h, ← Int.ofNat_succ, zpow_natCast, pow_succ']
        /-
          🎉 no goals
        -/


theorem zpow_sub_one {A : M} (h : IsUnit A.det) (n : ℤ) : A ^ (n - 1) = A ^ n * A⁻¹ :=
  calc
    A ^ (n - 1) = A ^ (n - 1) * A * A⁻¹ := by
      /-
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A : Matrix n' n' R
        h : IsUnit A.det
        n : Int
        ⊢ Eq (HPow.hPow A (HSub.hSub n 1)) (HMul.hMul (HMul.hMul (HPow.hPow A (HSub.hS …
      -/
      rw [mul_assoc, mul_nonsing_inv _ h, mul_one]
      /-
        🎉 no goals
      -/
                          /-
                            n' : Type u_1
                            inst✝² : DecidableEq n'
                            inst✝¹ : Fintype n'
                            R : Type u_2
                            inst✝ : CommRing R
                            A : Matrix n' n' R
                            h : IsUnit A.det
                            n : Int
                            ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow A (HSub.hSub n 1)) A) (Inv.inv A)) (HMul …
                          -/
    _ = A ^ n * A⁻¹ := by rw [← zpow_add_one h, sub_add_cancel]
                          /-
                            🎉 no goals
                          -/


theorem zpow_add {A : M} (ha : IsUnit A.det) (m n : ℤ) : A ^ (m + n) = A ^ m * A ^ n := by
  induction n using Int.induction_on with
  | hz => simp
  | hp n ihn => simp only [← add_assoc, zpow_add_one ha, ihn, mul_assoc]
  | hn n ihn => rw [zpow_sub_one ha, ← mul_assoc, ← ihn, ← zpow_sub_one ha, add_sub_assoc]


theorem zpow_add_of_nonpos {A : M} {m n : ℤ} (hm : m ≤ 0) (hn : n ≤ 0) :
    A ^ (m + n) = A ^ m * A ^ n := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    m n : Int
    hm : LE.le m 0
    hn : LE.le n 0
    ⊢ Eq (HPow.hPow A (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow A m) (HPow.hPow A n))
  -/
  rcases nonsing_inv_cancel_or_zero A with (⟨h, _⟩ | h)
    /-
      case inl.intro
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      m n : Int
      hm : LE.le m 0
      hn : LE.le n 0
      h : Eq (HMul.hMul (Inv.inv A) A) 1
      right✝ : Eq (HMul.hMul A (Inv.inv A)) 1
      ⊢ Eq (HPow.hPow A (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow A m) (HPow.hPow A n))
    -/
  · exact zpow_add (isUnit_det_of_left_inverse h) m n
    /-
      🎉 no goals
    -/
    /-
      case inr
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      m n : Int
      hm : LE.le m 0
      hn : LE.le n 0
      h : Eq (Inv.inv A) 0
      ⊢ Eq (HPow.hPow A (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow A m) (HPow.hPow A n))
    -/
  · obtain ⟨k, rfl⟩ := exists_eq_neg_ofNat hm
    /-
      case inr.intro
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n : Int
      hn : LE.le n 0
      h : Eq (Inv.inv A) 0
      k : Nat
      hm : LE.le (Neg.neg ↑k) 0
      ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg ↑k) n)) (HMul.hMul (HPow.hPow A (Neg.neg …
    -/
    obtain ⟨l, rfl⟩ := exists_eq_neg_ofNat hn
    /-
      case inr.intro.intro
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : Eq (Inv.inv A) 0
      k : Nat
      hm : LE.le (Neg.neg ↑k) 0
      l : Nat
      hn : LE.le (Neg.neg ↑l) 0
      ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg ↑k) (Neg.neg ↑l))) (HMul.hMul (HPow.hPow …
    -/
    simp_rw [← neg_add, ← Int.ofNat_add, zpow_neg_natCast, ← inv_pow', h, pow_add]
    /-
      🎉 no goals
    -/


theorem zpow_add_of_nonneg {A : M} {m n : ℤ} (hm : 0 ≤ m) (hn : 0 ≤ n) :
    A ^ (m + n) = A ^ m * A ^ n := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    m n : Int
    hm : LE.le 0 m
    hn : LE.le 0 n
    ⊢ Eq (HPow.hPow A (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow A m) (HPow.hPow A n))
  -/
  obtain ⟨k, rfl⟩ := eq_ofNat_of_zero_le hm
  /-
    case intro
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    n : Int
    hn : LE.le 0 n
    k : Nat
    hm : LE.le 0 ↑k
    ⊢ Eq (HPow.hPow A (HAdd.hAdd (↑k) n)) (HMul.hMul (HPow.hPow A ↑k) (HPow.hPow A …
  -/
  obtain ⟨l, rfl⟩ := eq_ofNat_of_zero_le hn
  /-
    case intro.intro
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    k : Nat
    hm : LE.le 0 ↑k
    l : Nat
    hn : LE.le 0 ↑l
    ⊢ Eq (HPow.hPow A (HAdd.hAdd ↑k ↑l)) (HMul.hMul (HPow.hPow A ↑k) (HPow.hPow A  …
  -/
  rw [← Int.ofNat_add, zpow_natCast, zpow_natCast, zpow_natCast, pow_add]
  /-
    🎉 no goals
  -/


theorem zpow_one_add {A : M} (h : IsUnit A.det) (i : ℤ) : A ^ (1 + i) = A * A ^ i := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    h : IsUnit A.det
    i : Int
    ⊢ Eq (HPow.hPow A (HAdd.hAdd 1 i)) (HMul.hMul A (HPow.hPow A i))
  -/
  rw [zpow_add h, zpow_one]
  /-
    🎉 no goals
  -/


theorem SemiconjBy.zpow_right {A X Y : M} (hx : IsUnit X.det) (hy : IsUnit Y.det)
    (h : SemiconjBy A X Y) : ∀ m : ℤ, SemiconjBy A (X ^ m) (Y ^ m)
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    A X Y : Matrix n' n' R
                    hx : IsUnit X.det
                    hy : IsUnit Y.det
                    h : SemiconjBy A X Y
                    n : Nat
                    ⊢ SemiconjBy A (HPow.hPow X ↑n) (HPow.hPow Y ↑n)
                  -/
  | (n : ℕ) => by simp [h.pow_right n]
                  /-
                    🎉 no goals
                  -/
  | -[n+1] => by
    have hx' : IsUnit (X ^ n.succ).det := by
      rw [det_pow]
      exact hx.pow n.succ
    have hy' : IsUnit (Y ^ n.succ).det := by
      rw [det_pow]
      exact hy.pow n.succ
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A X Y : Matrix n' n' R
      hx : IsUnit X.det
      hy : IsUnit Y.det
      h : SemiconjBy A X Y
      n : Nat
      hx' : IsUnit (HPow.hPow X n.succ).det
      hy' : IsUnit (HPow.hPow Y n.succ).det
      ⊢ SemiconjBy A (HPow.hPow X (Int.negSucc n)) (HPow.hPow Y (Int.negSucc n))
    -/
    rw [zpow_negSucc, zpow_negSucc, nonsing_inv_apply _ hx', nonsing_inv_apply _ hy', SemiconjBy]
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A X Y : Matrix n' n' R
      hx : IsUnit X.det
      hy : IsUnit Y.det
      h : SemiconjBy A X Y
      n : Nat
      hx' : IsUnit (HPow.hPow X n.succ).det
      hy' : IsUnit (HPow.hPow Y n.succ).det
      ⊢ Eq (HMul.hMul A (HSMul.hSMul (↑(Inv.inv hx'.unit)) (HPow.hPow X n.succ).adju …
    -/
    refine (isRegular_of_isLeftRegular_det hy'.isRegular.left).left ?_
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A X Y : Matrix n' n' R
      hx : IsUnit X.det
      hy : IsUnit Y.det
      h : SemiconjBy A X Y
      n : Nat
      hx' : IsUnit (HPow.hPow X n.succ).det
      hy' : IsUnit (HPow.hPow Y n.succ).det
      ⊢ Eq ((fun x => HMul.hMul (HPow.hPow Y n.succ) x) (HMul.hMul A (HSMul.hSMul (↑ …
    -/
    dsimp only
    rw [← mul_assoc, ← (h.pow_right n.succ).eq, mul_assoc, mul_smul,
      mul_adjugate, ← Matrix.mul_assoc,
      mul_smul (Y ^ _) (↑hy'.unit⁻¹ : R), mul_adjugate, smul_smul, smul_smul, hx'.val_inv_mul,
      hy'.val_inv_mul, one_smul, Matrix.mul_one, Matrix.one_mul]


theorem Commute.zpow_right {A B : M} (h : Commute A B) (m : ℤ) : Commute A (B ^ m) := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A B : Matrix n' n' R
    h : Commute A B
    m : Int
    ⊢ Commute A (HPow.hPow B m)
  -/
  rcases nonsing_inv_cancel_or_zero B with (⟨hB, _⟩ | hB)
    /-
      case inl.intro
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A B : Matrix n' n' R
      h : Commute A B
      m : Int
      hB : Eq (HMul.hMul (Inv.inv B) B) 1
      right✝ : Eq (HMul.hMul B (Inv.inv B)) 1
      ⊢ Commute A (HPow.hPow B m)
    -/
                                               /-
                                                 🎉 no goals
                                               -/
  · refine SemiconjBy.zpow_right ?_ ?_ h _ <;> exact isUnit_det_of_left_inverse hB
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      case inr
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A B : Matrix n' n' R
      h : Commute A B
      m : Int
      hB : Eq (Inv.inv B) 0
      ⊢ Commute A (HPow.hPow B m)
    -/
  · cases m
      /-
        case inr.ofNat
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A B : Matrix n' n' R
        h : Commute A B
        hB : Eq (Inv.inv B) 0
        a✝ : Nat
        ⊢ Commute A (HPow.hPow B (Int.ofNat a✝))
      -/
    · simpa using h.pow_right _
      /-
        🎉 no goals
      -/
      /-
        case inr.negSucc
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A B : Matrix n' n' R
        h : Commute A B
        hB : Eq (Inv.inv B) 0
        a✝ : Nat
        ⊢ Commute A (HPow.hPow B (Int.negSucc a✝))
      -/
    · simp [← inv_pow', hB]
      /-
        🎉 no goals
      -/


theorem Commute.zpow_left {A B : M} (h : Commute A B) (m : ℤ) : Commute (A ^ m) B :=
  (Commute.zpow_right h.symm m).symm


theorem Commute.zpow_zpow {A B : M} (h : Commute A B) (m n : ℤ) : Commute (A ^ m) (B ^ n) :=
  Commute.zpow_right (Commute.zpow_left h _) _


theorem Commute.zpow_self (A : M) (n : ℤ) : Commute (A ^ n) A :=
  Commute.zpow_left (Commute.refl A) _


theorem Commute.self_zpow (A : M) (n : ℤ) : Commute A (A ^ n) :=
  Commute.zpow_right (Commute.refl A) _


theorem Commute.zpow_zpow_self (A : M) (m n : ℤ) : Commute (A ^ m) (A ^ n) :=
  Commute.zpow_zpow (Commute.refl A) _ _


theorem zpow_add_one_of_ne_neg_one {A : M} : ∀ n : ℤ, n ≠ -1 → A ^ (n + 1) = A ^ n * A
                     /-
                       n' : Type u_1
                       inst✝² : DecidableEq n'
                       inst✝¹ : Fintype n'
                       R : Type u_2
                       inst✝ : CommRing R
                       A : Matrix n' n' R
                       n : Nat
                       x✝ : Ne (↑n) (-1)
                       ⊢ Eq (HPow.hPow A (HAdd.hAdd (↑n) 1)) (HMul.hMul (HPow.hPow A ↑n) A)
                     -/
  | (n : ℕ), _ => by simp only [pow_succ, ← Nat.cast_succ, zpow_natCast]
                     /-
                       🎉 no goals
                     -/
  | -1, h => absurd rfl h
  | -((n : ℕ) + 2), _ => by
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      n : Nat
      x✝ : Ne (Neg.neg (HAdd.hAdd (↑n) 2)) (-1)
      ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑n) 2)) 1)) (HMul.hMul (HPow …
    -/
    rcases nonsing_inv_cancel_or_zero A with (⟨h, _⟩ | h)
      /-
        case inl.intro
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A : Matrix n' n' R
        n : Nat
        x✝ : Ne (Neg.neg (HAdd.hAdd (↑n) 2)) (-1)
        h : Eq (HMul.hMul (Inv.inv A) A) 1
        right✝ : Eq (HMul.hMul A (Inv.inv A)) 1
        ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑n) 2)) 1)) (HMul.hMul (HPow …
      -/
    · apply zpow_add_one (isUnit_det_of_left_inverse h)
      /-
        🎉 no goals
      -/
      /-
        case inr
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A : Matrix n' n' R
        n : Nat
        x✝ : Ne (Neg.neg (HAdd.hAdd (↑n) 2)) (-1)
        h : Eq (Inv.inv A) 0
        ⊢ Eq (HPow.hPow A (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑n) 2)) 1)) (HMul.hMul (HPow …
      -/
    · show A ^ (-((n + 1 : ℕ) : ℤ)) = A ^ (-((n + 2 : ℕ) : ℤ)) * A
      /-
        case inr
        n' : Type u_1
        inst✝² : DecidableEq n'
        inst✝¹ : Fintype n'
        R : Type u_2
        inst✝ : CommRing R
        A : Matrix n' n' R
        n : Nat
        x✝ : Ne (Neg.neg (HAdd.hAdd (↑n) 2)) (-1)
        h : Eq (Inv.inv A) 0
        ⊢ Eq (HPow.hPow A (Neg.neg ↑(HAdd.hAdd n 1))) (HMul.hMul (HPow.hPow A (Neg.neg …
      -/
      simp_rw [zpow_neg_natCast, ← inv_pow', h, zero_pow <| Nat.succ_ne_zero _, zero_mul]
      /-
        🎉 no goals
      -/


theorem zpow_mul (A : M) (h : IsUnit A.det) : ∀ m n : ℤ, A ^ (m * n) = (A ^ m) ^ n
                           /-
                             n' : Type u_1
                             inst✝² : DecidableEq n'
                             inst✝¹ : Fintype n'
                             R : Type u_2
                             inst✝ : CommRing R
                             A : Matrix n' n' R
                             h : IsUnit A.det
                             m n : Nat
                             ⊢ Eq (HPow.hPow A (HMul.hMul ↑m ↑n)) (HPow.hPow (HPow.hPow A ↑m) ↑n)
                           -/
  | (m : ℕ), (n : ℕ) => by rw [zpow_natCast, zpow_natCast, ← pow_mul, ← zpow_natCast, Int.ofNat_mul]
                           /-
                             🎉 no goals
                           -/
  | (m : ℕ), -[n+1] => by
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      m n : Nat
      ⊢ Eq (HPow.hPow A (HMul.hMul (↑m) (Int.negSucc n))) (HPow.hPow (HPow.hPow A ↑m …
    -/
    rw [zpow_natCast, zpow_negSucc, ← pow_mul, ofNat_mul_negSucc, zpow_neg_natCast]
    /-
      🎉 no goals
    -/
  | -[m+1], (n : ℕ) => by
    rw [zpow_natCast, zpow_negSucc, ← inv_pow', ← pow_mul, negSucc_mul_ofNat, zpow_neg_natCast,
        inv_pow']
  | -[m+1], -[n+1] => by
    rw [zpow_negSucc, zpow_negSucc, negSucc_mul_negSucc, ← Int.ofNat_mul, zpow_natCast, inv_pow', ←
      pow_mul, nonsing_inv_nonsing_inv]
    /-
      case h
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      m n : Nat
      ⊢ IsUnit (HPow.hPow A (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd n 1))).det
    -/
    rw [det_pow]
    /-
      case h
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      A : Matrix n' n' R
      h : IsUnit A.det
      m n : Nat
      ⊢ IsUnit (HPow.hPow A.det (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd n 1)))
    -/
    exact h.pow _
    /-
      🎉 no goals
    -/


theorem zpow_mul' (A : M) (h : IsUnit A.det) (m n : ℤ) : A ^ (m * n) = (A ^ n) ^ m := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    h : IsUnit A.det
    m n : Int
    ⊢ Eq (HPow.hPow A (HMul.hMul m n)) (HPow.hPow (HPow.hPow A n) m)
  -/
  rw [mul_comm, zpow_mul _ h]
  /-
    🎉 no goals
  -/



@[simp, norm_cast]
theorem coe_units_zpow (u : Mˣ) : ∀ n : ℤ, ((u ^ n : Mˣ) : M) = (u : M) ^ n
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    u : Units (Matrix n' n' R)
                    n : Nat
                    ⊢ Eq (↑(HPow.hPow u ↑n)) (HPow.hPow ↑u ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast, zpow_natCast, Units.val_pow_eq_pow_val]
                  /-
                    🎉 no goals
                  -/
  | -[k+1] => by
    /-
      n' : Type u_1
      inst✝² : DecidableEq n'
      inst✝¹ : Fintype n'
      R : Type u_2
      inst✝ : CommRing R
      u : Units (Matrix n' n' R)
      k : Nat
      ⊢ Eq (↑(HPow.hPow u (Int.negSucc k))) (HPow.hPow (↑u) (Int.negSucc k))
    -/
    rw [zpow_negSucc, zpow_negSucc, ← inv_pow, u⁻¹.val_pow_eq_pow_val, ← inv_pow', coe_units_inv]
    /-
      🎉 no goals
    -/


theorem zpow_ne_zero_of_isUnit_det [Nonempty n'] [Nontrivial R] {A : M} (ha : IsUnit A.det)
    (z : ℤ) : A ^ z ≠ 0 := by
  /-
    n' : Type u_1
    inst✝⁴ : DecidableEq n'
    inst✝³ : Fintype n'
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Nonempty n'
    inst✝ : Nontrivial R
    A : Matrix n' n' R
    ha : IsUnit A.det
    z : Int
    ⊢ Ne (HPow.hPow A z) 0
  -/
  have := ha.det_zpow z
  /-
    n' : Type u_1
    inst✝⁴ : DecidableEq n'
    inst✝³ : Fintype n'
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Nonempty n'
    inst✝ : Nontrivial R
    A : Matrix n' n' R
    ha : IsUnit A.det
    z : Int
    this : IsUnit (HPow.hPow A z).det
    ⊢ Ne (HPow.hPow A z) 0
  -/
  contrapose! this
  /-
    n' : Type u_1
    inst✝⁴ : DecidableEq n'
    inst✝³ : Fintype n'
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Nonempty n'
    inst✝ : Nontrivial R
    A : Matrix n' n' R
    ha : IsUnit A.det
    z : Int
    this : Eq (HPow.hPow A z) 0
    ⊢ Not (IsUnit (HPow.hPow A z).det)
  -/
  rw [this, det_zero ‹_›]
  /-
    n' : Type u_1
    inst✝⁴ : DecidableEq n'
    inst✝³ : Fintype n'
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Nonempty n'
    inst✝ : Nontrivial R
    A : Matrix n' n' R
    ha : IsUnit A.det
    z : Int
    this : Eq (HPow.hPow A z) 0
    ⊢ Not (IsUnit 0)
  -/
  exact not_isUnit_zero
  /-
    🎉 no goals
  -/


theorem zpow_sub {A : M} (ha : IsUnit A.det) (z1 z2 : ℤ) : A ^ (z1 - z2) = A ^ z1 / A ^ z2 := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    ha : IsUnit A.det
    z1 z2 : Int
    ⊢ Eq (HPow.hPow A (HSub.hSub z1 z2)) (HDiv.hDiv (HPow.hPow A z1) (HPow.hPow A  …
  -/
  rw [sub_eq_add_neg, zpow_add ha, zpow_neg ha, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem Commute.mul_zpow {A B : M} (h : Commute A B) : ∀ i : ℤ, (A * B) ^ i = A ^ i * B ^ i
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    A B : Matrix n' n' R
                    h : Commute A B
                    n : Nat
                    ⊢ Eq (HPow.hPow (HMul.hMul A B) ↑n) (HMul.hMul (HPow.hPow A ↑n) (HPow.hPow B ↑ …
                  -/
  | (n : ℕ) => by simp [h.mul_pow n]
                  /-
                    🎉 no goals
                  -/
  | -[n+1] => by
    rw [zpow_negSucc, zpow_negSucc, zpow_negSucc, ← mul_inv_rev,
      h.mul_pow n.succ, (h.pow_pow _ _).eq]


theorem zpow_neg_mul_zpow_self (n : ℤ) {A : M} (h : IsUnit A.det) : A ^ (-n) * A ^ n = 1 := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    n : Int
    A : Matrix n' n' R
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (HPow.hPow A (Neg.neg n)) (HPow.hPow A n)) 1
  -/
  rw [zpow_neg h, nonsing_inv_mul _ (h.det_zpow _)]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      n' : Type u_1
                                                                      inst✝² : DecidableEq n'
                                                                      inst✝¹ : Fintype n'
                                                                      R : Type u_2
                                                                      inst✝ : CommRing R
                                                                      A : Matrix n' n' R
                                                                      n : Nat
                                                                      ⊢ Eq (HPow.hPow (HDiv.hDiv 1 A) n) (HDiv.hDiv 1 (HPow.hPow A n))
                                                                    -/
theorem one_div_pow {A : M} (n : ℕ) : (1 / A) ^ n = 1 / A ^ n := by simp only [one_div, inv_pow']
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                     /-
                                                                       n' : Type u_1
                                                                       inst✝² : DecidableEq n'
                                                                       inst✝¹ : Fintype n'
                                                                       R : Type u_2
                                                                       inst✝ : CommRing R
                                                                       A : Matrix n' n' R
                                                                       n : Int
                                                                       ⊢ Eq (HPow.hPow (HDiv.hDiv 1 A) n) (HDiv.hDiv 1 (HPow.hPow A n))
                                                                     -/
theorem one_div_zpow {A : M} (n : ℤ) : (1 / A) ^ n = 1 / A ^ n := by simp only [one_div, inv_zpow]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem transpose_zpow (A : M) : ∀ n : ℤ, (A ^ n)ᵀ = Aᵀ ^ n
                  /-
                    n' : Type u_1
                    inst✝² : DecidableEq n'
                    inst✝¹ : Fintype n'
                    R : Type u_2
                    inst✝ : CommRing R
                    A : Matrix n' n' R
                    n : Nat
                    ⊢ Eq (HPow.hPow A ↑n).transpose (HPow.hPow A.transpose ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast, zpow_natCast, transpose_pow]
                  /-
                    🎉 no goals
                  -/
                 /-
                   n' : Type u_1
                   inst✝² : DecidableEq n'
                   inst✝¹ : Fintype n'
                   R : Type u_2
                   inst✝ : CommRing R
                   A : Matrix n' n' R
                   n : Nat
                   ⊢ Eq (HPow.hPow A (Int.negSucc n)).transpose (HPow.hPow A.transpose (Int.negSu …
                 -/
  | -[n+1] => by rw [zpow_negSucc, zpow_negSucc, transpose_nonsing_inv, transpose_pow]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem conjTranspose_zpow [StarRing R] (A : M) : ∀ n : ℤ, (A ^ n)ᴴ = Aᴴ ^ n
                  /-
                    n' : Type u_1
                    inst✝³ : DecidableEq n'
                    inst✝² : Fintype n'
                    R : Type u_2
                    inst✝¹ : CommRing R
                    inst✝ : StarRing R
                    A : Matrix n' n' R
                    n : Nat
                    ⊢ Eq (HPow.hPow A ↑n).conjTranspose (HPow.hPow A.conjTranspose ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast, zpow_natCast, conjTranspose_pow]
                  /-
                    🎉 no goals
                  -/
                 /-
                   n' : Type u_1
                   inst✝³ : DecidableEq n'
                   inst✝² : Fintype n'
                   R : Type u_2
                   inst✝¹ : CommRing R
                   inst✝ : StarRing R
                   A : Matrix n' n' R
                   n : Nat
                   ⊢ Eq (HPow.hPow A (Int.negSucc n)).conjTranspose (HPow.hPow A.conjTranspose (I …
                 -/
  | -[n+1] => by rw [zpow_negSucc, zpow_negSucc, conjTranspose_nonsing_inv, conjTranspose_pow]
                 /-
                   🎉 no goals
                 -/


theorem IsSymm.zpow {A : M} (h : A.IsSymm) (k : ℤ) :
    (A ^ k).IsSymm := by
  /-
    n' : Type u_1
    inst✝² : DecidableEq n'
    inst✝¹ : Fintype n'
    R : Type u_2
    inst✝ : CommRing R
    A : Matrix n' n' R
    h : A.IsSymm
    k : Int
    ⊢ (HPow.hPow A k).IsSymm
  -/
  rw [IsSymm, transpose_zpow, h]
  /-
    🎉 no goals
  -/


