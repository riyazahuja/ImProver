                                                         /-
                                                           a b : ENNReal
                                                           ⊢ Eq (HDiv.hDiv a b) (HMul.hMul (Inv.inv b) a)
                                                         -/
protected theorem div_eq_inv_mul : a / b = b⁻¹ * a := by rw [div_eq_mul_inv, mul_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp] theorem inv_zero : (0 : ℝ≥0∞)⁻¹ = ∞ :=
                                            /-
                                              ⊢ Eq (InfSet.sInf (setOf fun b => LE.le 1 (HMul.hMul 0 b))) Top.top
                                            -/
  show sInf { b : ℝ≥0∞ | 1 ≤ 0 * b } = ∞ by simp
                                            /-
                                              🎉 no goals
                                            -/


@[simp] theorem inv_top : ∞⁻¹ = 0 :=
                                                                            /-
                                                                              a : ENNReal
                                                                              h : LT.lt 0 a
                                                                              ⊢ Membership.mem (setOf fun b => LE.le 1 (HMul.hMul Top.top b)) a
                                                                            -/
  bot_unique <| le_of_forall_le_of_dense fun a (h : 0 < a) => sInf_le <| by simp [*, h.ne', top_mul]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem coe_inv_le : (↑r⁻¹ : ℝ≥0∞) ≤ (↑r)⁻¹ :=
  le_sInf fun b (hb : 1 ≤ ↑r * b) =>
    coe_le_iff.2 <| by
      /-
        r : NNReal
        b : ENNReal
        hb : LE.le 1 (HMul.hMul (↑r) b)
        ⊢ ∀ (p : NNReal), Eq b ↑p → LE.le (Inv.inv r) p
      -/
      rintro b rfl
      /-
        r b : NNReal
        hb : LE.le 1 (HMul.hMul ↑r ↑b)
        ⊢ LE.le (Inv.inv r) b
      -/
      apply NNReal.inv_le_of_le_mul
      /-
        case h
        r b : NNReal
        hb : LE.le 1 (HMul.hMul ↑r ↑b)
        ⊢ LE.le 1 (HMul.hMul r b)
      -/
      rwa [← coe_mul, ← coe_one, coe_le_coe] at hb
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem coe_inv (hr : r ≠ 0) : (↑r⁻¹ : ℝ≥0∞) = (↑r)⁻¹ :=
                                                      /-
                                                        r : NNReal
                                                        hr : Ne r 0
                                                        ⊢ LE.le 1 (HMul.hMul ↑r ↑(Inv.inv r))
                                                      -/
  coe_inv_le.antisymm <| sInf_le <| mem_setOf.2 <| by rw [← coe_mul, mul_inv_cancel₀ hr, coe_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[norm_cast]
                                                       /-
                                                         ⊢ Eq (↑(Inv.inv 2)) (Inv.inv 2)
                                                       -/
theorem coe_inv_two : ((2⁻¹ : ℝ≥0) : ℝ≥0∞) = 2⁻¹ := by rw [coe_inv _root_.two_ne_zero, coe_two]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp, norm_cast]
theorem coe_div (hr : r ≠ 0) : (↑(p / r) : ℝ≥0∞) = p / r := by
  /-
    r p : NNReal
    hr : Ne r 0
    ⊢ Eq (↑(HDiv.hDiv p r)) (HDiv.hDiv ↑p ↑r)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, coe_mul, coe_inv hr]
  /-
    🎉 no goals
  -/


lemma coe_div_le : ↑(p / r) ≤ (p / r : ℝ≥0∞) := by
  /-
    r p : NNReal
    ⊢ LE.le (↑(HDiv.hDiv p r)) (HDiv.hDiv ↑p ↑r)
  -/
  simpa only [div_eq_mul_inv, coe_mul] using mul_le_mul_left' coe_inv_le _
  /-
    🎉 no goals
  -/


                                               /-
                                                 a : ENNReal
                                                 h : Ne a 0
                                                 ⊢ Eq (HDiv.hDiv a 0) Top.top
                                               -/
theorem div_zero (h : a ≠ 0) : a / 0 = ∞ := by simp [div_eq_mul_inv, h]
                                               /-
                                                 🎉 no goals
                                               -/


instance : DivInvOneMonoid ℝ≥0∞ :=
  { inferInstanceAs (DivInvMonoid ℝ≥0∞) with
                  /-
                    a b c d : ENNReal
                    r p q : NNReal
                    ⊢ Eq (Inv.inv 1) 1
                  -/
    inv_one := by simpa only [coe_inv one_ne_zero, coe_one] using coe_inj.2 inv_one }
                  /-
                    🎉 no goals
                  -/


protected theorem inv_pow : ∀ {a : ℝ≥0∞} {n : ℕ}, (a ^ n)⁻¹ = a⁻¹ ^ n
               /-
                 x✝ : ENNReal
                 ⊢ Eq (Inv.inv (HPow.hPow x✝ 0)) (HPow.hPow (Inv.inv x✝) 0)
               -/
  | _, 0 => by simp only [pow_zero, inv_one]
               /-
                 🎉 no goals
               -/
                   /-
                     n : Nat
                     ⊢ Eq (Inv.inv (HPow.hPow Top.top (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv Top.top …
                   -/
  | ⊤, n + 1 => by simp [top_pow]
                   /-
                     🎉 no goals
                   -/
  | (a : ℝ≥0), n + 1 => by
    /-
      a : NNReal
      n : Nat
      ⊢ Eq (Inv.inv (HPow.hPow (↑a) (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv ↑a) (HAdd. …
    -/
    rcases eq_or_ne a 0 with (rfl | ha)
      /-
        case inl
        n : Nat
        ⊢ Eq (Inv.inv (HPow.hPow (↑0) (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv ↑0) (HAdd. …
      -/
    · simp [top_pow]
      /-
        🎉 no goals
      -/
      /-
        case inr
        a : NNReal
        n : Nat
        ha : Ne a 0
        ⊢ Eq (Inv.inv (HPow.hPow (↑a) (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv ↑a) (HAdd. …
      -/
    · have := pow_ne_zero (n + 1) ha
      /-
        case inr
        a : NNReal
        n : Nat
        ha : Ne a 0
        this : Ne (HPow.hPow a (HAdd.hAdd n 1)) 0
        ⊢ Eq (Inv.inv (HPow.hPow (↑a) (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv ↑a) (HAdd. …
      -/
      norm_cast
      /-
        case inr
        a : NNReal
        n : Nat
        ha : Ne a 0
        this : Ne (HPow.hPow a (HAdd.hAdd n 1)) 0
        ⊢ Eq (Inv.inv (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow (Inv.inv a) (HAdd.hAdd …
      -/
      rw [inv_pow]
      /-
        🎉 no goals
      -/


protected theorem mul_inv_cancel (h0 : a ≠ 0) (ht : a ≠ ∞) : a * a⁻¹ = 1 := by
  /-
    a : ENNReal
    h0 : Ne a 0
    ht : Ne a Top.top
    ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
  -/
  lift a to ℝ≥0 using ht
  /-
    case intro
    a : NNReal
    h0 : Ne (↑a) 0
    ⊢ Eq (HMul.hMul (↑a) (Inv.inv ↑a)) 1
  -/
  norm_cast at h0; norm_cast
  /-
    case intro
    a : NNReal
    h0 : Not (Eq a 0)
    ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
  -/
  exact mul_inv_cancel₀ h0
  /-
    🎉 no goals
  -/


protected theorem inv_mul_cancel (h0 : a ≠ 0) (ht : a ≠ ∞) : a⁻¹ * a = 1 :=
  mul_comm a a⁻¹ ▸ ENNReal.mul_inv_cancel h0 ht


/-- See `ENNReal.inv_mul_cancel_left` for a simpler version assuming `a ≠ 0`, `a ≠ ∞`. -/
protected lemma inv_mul_cancel_left' (ha₀ : a = 0 → b = 0) (ha : a = ∞ → b = 0) :
    a⁻¹ * (a * b) = b := by
  /-
    a b : ENNReal
    ha₀ : Eq a 0 → Eq b 0
    ha : Eq a Top.top → Eq b 0
    ⊢ Eq (HMul.hMul (Inv.inv a) (HMul.hMul a b)) b
  -/
  obtain rfl | ha₀ := eq_or_ne a 0
    /-
      case inl
      b : ENNReal
      ha₀ : Eq 0 0 → Eq b 0
      ha : Eq 0 Top.top → Eq b 0
      ⊢ Eq (HMul.hMul (Inv.inv 0) (HMul.hMul 0 b)) b
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : ENNReal
    ha₀✝ : Eq a 0 → Eq b 0
    ha : Eq a Top.top → Eq b 0
    ha₀ : Ne a 0
    ⊢ Eq (HMul.hMul (Inv.inv a) (HMul.hMul a b)) b
  -/
  obtain rfl | ha := eq_or_ne a ⊤
    /-
      case inr.inl
      b : ENNReal
      ha₀✝ : Eq Top.top 0 → Eq b 0
      ha : Eq Top.top Top.top → Eq b 0
      ha₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul (Inv.inv Top.top) (HMul.hMul Top.top b)) b
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : ENNReal
      ha₀✝ : Eq a 0 → Eq b 0
      ha✝ : Eq a Top.top → Eq b 0
      ha₀ : Ne a 0
      ha : Ne a Top.top
      ⊢ Eq (HMul.hMul (Inv.inv a) (HMul.hMul a b)) b
    -/
  · simp [← mul_assoc, ENNReal.inv_mul_cancel, *]
    /-
      🎉 no goals
    -/


/-- See `ENNReal.inv_mul_cancel_left'` for a stronger version. -/
protected lemma inv_mul_cancel_left (ha₀ : a ≠ 0) (ha : a ≠ ∞) : a⁻¹ * (a * b) = b :=
                                   /-
                                     a b : ENNReal
                                     ha₀ : Ne a 0
                                     ha : Ne a Top.top
                                     ⊢ Eq a 0 → Eq b 0
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ENNReal.inv_mul_cancel_left' (by simp [ha₀]) (by simp [ha])
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- See `ENNReal.mul_inv_cancel_left` for a simpler version assuming `a ≠ 0`, `a ≠ ∞`. -/
protected lemma mul_inv_cancel_left' (ha₀ : a = 0 → b = 0) (ha : a = ∞ → b = 0) :
    a * (a⁻¹ * b) = b := by
  /-
    a b : ENNReal
    ha₀ : Eq a 0 → Eq b 0
    ha : Eq a Top.top → Eq b 0
    ⊢ Eq (HMul.hMul a (HMul.hMul (Inv.inv a) b)) b
  -/
  obtain rfl | ha₀ := eq_or_ne a 0
    /-
      case inl
      b : ENNReal
      ha₀ : Eq 0 0 → Eq b 0
      ha : Eq 0 Top.top → Eq b 0
      ⊢ Eq (HMul.hMul 0 (HMul.hMul (Inv.inv 0) b)) b
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : ENNReal
    ha₀✝ : Eq a 0 → Eq b 0
    ha : Eq a Top.top → Eq b 0
    ha₀ : Ne a 0
    ⊢ Eq (HMul.hMul a (HMul.hMul (Inv.inv a) b)) b
  -/
  obtain rfl | ha := eq_or_ne a ⊤
    /-
      case inr.inl
      b : ENNReal
      ha₀✝ : Eq Top.top 0 → Eq b 0
      ha : Eq Top.top Top.top → Eq b 0
      ha₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul Top.top (HMul.hMul (Inv.inv Top.top) b)) b
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : ENNReal
      ha₀✝ : Eq a 0 → Eq b 0
      ha✝ : Eq a Top.top → Eq b 0
      ha₀ : Ne a 0
      ha : Ne a Top.top
      ⊢ Eq (HMul.hMul a (HMul.hMul (Inv.inv a) b)) b
    -/
  · simp [← mul_assoc, ENNReal.mul_inv_cancel, *]
    /-
      🎉 no goals
    -/


/-- See `ENNReal.mul_inv_cancel_left'` for a stronger version. -/
protected lemma mul_inv_cancel_left (ha₀ : a ≠ 0) (ha : a ≠ ∞) : a * (a⁻¹ * b) = b :=
                                   /-
                                     a b : ENNReal
                                     ha₀ : Ne a 0
                                     ha : Ne a Top.top
                                     ⊢ Eq a 0 → Eq b 0
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ENNReal.mul_inv_cancel_left' (by simp [ha₀]) (by simp [ha])
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- See `ENNReal.mul_inv_cancel_right` for a simpler version assuming `b ≠ 0`, `b ≠ ∞`. -/
protected lemma mul_inv_cancel_right' (hb₀ : b = 0 → a = 0) (hb : b = ∞ → a = 0) :
    a * b * b⁻¹ = a := by
  /-
    a b : ENNReal
    hb₀ : Eq b 0 → Eq a 0
    hb : Eq b Top.top → Eq a 0
    ⊢ Eq (HMul.hMul (HMul.hMul a b) (Inv.inv b)) a
  -/
  obtain rfl | hb₀ := eq_or_ne b 0
    /-
      case inl
      a : ENNReal
      hb₀ : Eq 0 0 → Eq a 0
      hb : Eq 0 Top.top → Eq a 0
      ⊢ Eq (HMul.hMul (HMul.hMul a 0) (Inv.inv 0)) a
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : ENNReal
    hb₀✝ : Eq b 0 → Eq a 0
    hb : Eq b Top.top → Eq a 0
    hb₀ : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul a b) (Inv.inv b)) a
  -/
  obtain rfl | hb := eq_or_ne b ⊤
    /-
      case inr.inl
      a : ENNReal
      hb₀✝ : Eq Top.top 0 → Eq a 0
      hb : Eq Top.top Top.top → Eq a 0
      hb₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul (HMul.hMul a Top.top) (Inv.inv Top.top)) a
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : ENNReal
      hb₀✝ : Eq b 0 → Eq a 0
      hb✝ : Eq b Top.top → Eq a 0
      hb₀ : Ne b 0
      hb : Ne b Top.top
      ⊢ Eq (HMul.hMul (HMul.hMul a b) (Inv.inv b)) a
    -/
  · simp [mul_assoc, ENNReal.mul_inv_cancel, *]
    /-
      🎉 no goals
    -/


/-- See `ENNReal.mul_inv_cancel_right'` for a stronger version. -/
protected lemma mul_inv_cancel_right (hb₀ : b ≠ 0) (hb : b ≠ ∞) : a * b * b⁻¹ = a :=
                                    /-
                                      a b : ENNReal
                                      hb₀ : Ne b 0
                                      hb : Ne b Top.top
                                      ⊢ Eq b 0 → Eq a 0
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  ENNReal.mul_inv_cancel_right' (by simp [hb₀]) (by simp [hb])
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- See `ENNReal.inv_mul_cancel_right` for a simpler version assuming `b ≠ 0`, `b ≠ ∞`. -/
protected lemma inv_mul_cancel_right' (hb₀ : b = 0 → a = 0) (hb : b = ∞ → a = 0) :
    a * b⁻¹ * b = a := by
  /-
    a b : ENNReal
    hb₀ : Eq b 0 → Eq a 0
    hb : Eq b Top.top → Eq a 0
    ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv b)) b) a
  -/
  obtain rfl | hb₀ := eq_or_ne b 0
    /-
      case inl
      a : ENNReal
      hb₀ : Eq 0 0 → Eq a 0
      hb : Eq 0 Top.top → Eq a 0
      ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv 0)) 0) a
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : ENNReal
    hb₀✝ : Eq b 0 → Eq a 0
    hb : Eq b Top.top → Eq a 0
    hb₀ : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv b)) b) a
  -/
  obtain rfl | hb := eq_or_ne b ⊤
    /-
      case inr.inl
      a : ENNReal
      hb₀✝ : Eq Top.top 0 → Eq a 0
      hb : Eq Top.top Top.top → Eq a 0
      hb₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv Top.top)) Top.top) a
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : ENNReal
      hb₀✝ : Eq b 0 → Eq a 0
      hb✝ : Eq b Top.top → Eq a 0
      hb₀ : Ne b 0
      hb : Ne b Top.top
      ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv b)) b) a
    -/
  · simp [mul_assoc, ENNReal.inv_mul_cancel, *]
    /-
      🎉 no goals
    -/


/-- See `ENNReal.inv_mul_cancel_right'` for a stronger version. -/
protected lemma inv_mul_cancel_right (hb₀ : b ≠ 0) (hb : b ≠ ∞) : a * b⁻¹ * b = a :=
                                    /-
                                      a b : ENNReal
                                      hb₀ : Ne b 0
                                      hb : Ne b Top.top
                                      ⊢ Eq b 0 → Eq a 0
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  ENNReal.inv_mul_cancel_right' (by simp [hb₀]) (by simp [hb])
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- See `ENNReal.mul_div_cancel_right` for a simpler version assuming `b ≠ 0`, `b ≠ ∞`. -/
protected lemma mul_div_cancel_right' (hb₀ : b = 0 → a = 0) (hb : b = ∞ → a = 0) :
    a * b / b = a := ENNReal.mul_inv_cancel_right' hb₀ hb


/-- See `ENNReal.mul_div_cancel_right'` for a stronger version. -/
protected lemma mul_div_cancel_right (hb₀ : b ≠ 0) (hb : b ≠ ∞) : a * b / b = a :=
                                    /-
                                      a b : ENNReal
                                      hb₀ : Ne b 0
                                      hb : Ne b Top.top
                                      ⊢ Eq b 0 → Eq a 0
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  ENNReal.mul_div_cancel_right' (by simp [hb₀]) (by simp [hb])
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- See `ENNReal.div_mul_cancel` for a simpler version assuming `a ≠ 0`, `a ≠ ∞`. -/
protected lemma div_mul_cancel' (ha₀ : a = 0 → b = 0) (ha : a = ∞ → b = 0) : b / a * a = b :=
  ENNReal.inv_mul_cancel_right' ha₀ ha


/-- See `ENNReal.div_mul_cancel'` for a stronger version. -/
protected lemma div_mul_cancel (ha₀ : a ≠ 0) (ha : a ≠ ∞) : b / a * a = b :=
                              /-
                                a b : ENNReal
                                ha₀ : Ne a 0
                                ha : Ne a Top.top
                                ⊢ Eq a 0 → Eq b 0
                              -/
                              /-
                                🎉 no goals
                              -/
  ENNReal.div_mul_cancel' (by simp [ha₀]) (by simp [ha])
                                              /-
                                                🎉 no goals
                                              -/


/-- See `ENNReal.mul_div_cancel` for a simpler version assuming `a ≠ 0`, `a ≠ ∞`. -/
protected lemma mul_div_cancel' (ha₀ : a = 0 → b = 0) (ha : a = ∞ → b = 0) : a * (b / a) = b := by
  /-
    a b : ENNReal
    ha₀ : Eq a 0 → Eq b 0
    ha : Eq a Top.top → Eq b 0
    ⊢ Eq (HMul.hMul a (HDiv.hDiv b a)) b
  -/
  rw [mul_comm, ENNReal.div_mul_cancel' ha₀ ha]
  /-
    🎉 no goals
  -/


/-- See `ENNReal.mul_div_cancel'` for a stronger version. -/
protected lemma mul_div_cancel (ha₀ : a ≠ 0) (ha : a ≠ ∞) : a * (b / a) = b :=
                              /-
                                a b : ENNReal
                                ha₀ : Ne a 0
                                ha : Ne a Top.top
                                ⊢ Eq a 0 → Eq b 0
                              -/
                              /-
                                🎉 no goals
                              -/
  ENNReal.mul_div_cancel' (by simp [ha₀]) (by simp [ha])
                                              /-
                                                🎉 no goals
                                              -/

-- Porting note: `simp only [div_eq_mul_inv, mul_comm, mul_assoc]` doesn't work in the following two

protected theorem mul_comm_div : a / b * c = a * (c / b) := by
  /-
    a b c : ENNReal
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HMul.hMul a (HDiv.hDiv c b))
  -/
  simp only [div_eq_mul_inv, mul_right_comm, ← mul_assoc]
  /-
    🎉 no goals
  -/


protected theorem mul_div_right_comm : a * b / c = a / c * b := by
  /-
    a b c : ENNReal
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) c) (HMul.hMul (HDiv.hDiv a c) b)
  -/
  simp only [div_eq_mul_inv, mul_right_comm]
  /-
    🎉 no goals
  -/


instance : InvolutiveInv ℝ≥0∞ where
  inv_inv a := by
    /-
      a✝ b c d : ENNReal
      r p q : NNReal
      a : ENNReal
      ⊢ Eq (Inv.inv (Inv.inv a)) a
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
    by_cases a = 0 <;> cases a <;> simp_all [none_eq_top, some_eq_coe, -coe_inv, (coe_inv _).symm]
                                   /-
                                     🎉 no goals
                                   -/


                                                           /-
                                                             a : ENNReal
                                                             ⊢ Iff (Eq (Inv.inv a) 1) (Eq a 1)
                                                           -/
@[simp] protected lemma inv_eq_one : a⁻¹ = 1 ↔ a = 1 := by rw [← inv_inj, inv_inv, inv_one]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp] theorem inv_eq_top : a⁻¹ = ∞ ↔ a = 0 := inv_zero ▸ inv_inj


                                           /-
                                             a : ENNReal
                                             ⊢ Iff (Ne (Inv.inv a) Top.top) (Ne a 0)
                                           -/
theorem inv_ne_top : a⁻¹ ≠ ∞ ↔ a ≠ 0 := by simp
                                           /-
                                             🎉 no goals
                                           -/


@[aesop (rule_sets := [finiteness]) safe apply]
protected alias ⟨_, Finiteness.inv_ne_top⟩ := ENNReal.inv_ne_top


@[simp]
theorem inv_lt_top {x : ℝ≥0∞} : x⁻¹ < ∞ ↔ 0 < x := by
  /-
    x : ENNReal
    ⊢ Iff (LT.lt (Inv.inv x) Top.top) (LT.lt 0 x)
  -/
  simp only [lt_top_iff_ne_top, inv_ne_top, pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem div_lt_top {x y : ℝ≥0∞} (h1 : x ≠ ∞) (h2 : y ≠ 0) : x / y < ∞ :=
  mul_lt_top h1.lt_top (inv_ne_top.mpr h2).lt_top


@[simp]
protected theorem inv_eq_zero : a⁻¹ = 0 ↔ a = ∞ :=
  inv_top ▸ inv_inj


                                                      /-
                                                        a : ENNReal
                                                        ⊢ Iff (Ne (Inv.inv a) 0) (Ne a Top.top)
                                                      -/
protected theorem inv_ne_zero : a⁻¹ ≠ 0 ↔ a ≠ ∞ := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


protected theorem div_pos (ha : a ≠ 0) (hb : b ≠ ∞) : 0 < a / b :=
  ENNReal.mul_pos ha <| ENNReal.inv_ne_zero.2 hb


protected theorem inv_mul_le_iff {x y z : ℝ≥0∞} (h1 : x ≠ 0) (h2 : x ≠ ∞) :
    x⁻¹ * y ≤ z ↔ y ≤ x * z := by
  /-
    x y z : ENNReal
    h1 : Ne x 0
    h2 : Ne x Top.top
    ⊢ Iff (LE.le (HMul.hMul (Inv.inv x) y) z) (LE.le y (HMul.hMul x z))
  -/
  rw [← mul_le_mul_left h1 h2, ← mul_assoc, ENNReal.mul_inv_cancel h1 h2, one_mul]
  /-
    🎉 no goals
  -/


protected theorem mul_inv_le_iff {x y z : ℝ≥0∞} (h1 : y ≠ 0) (h2 : y ≠ ∞) :
    x * y⁻¹ ≤ z ↔ x ≤ z * y := by
  /-
    x y z : ENNReal
    h1 : Ne y 0
    h2 : Ne y Top.top
    ⊢ Iff (LE.le (HMul.hMul x (Inv.inv y)) z) (LE.le x (HMul.hMul z y))
  -/
  rw [mul_comm, ENNReal.inv_mul_le_iff h1 h2, mul_comm]
  /-
    🎉 no goals
  -/


protected theorem div_le_iff {x y z : ℝ≥0∞} (h1 : y ≠ 0) (h2 : y ≠ ∞) :
    x / y ≤ z ↔ x ≤ z * y := by
  /-
    x y z : ENNReal
    h1 : Ne y 0
    h2 : Ne y Top.top
    ⊢ Iff (LE.le (HDiv.hDiv x y) z) (LE.le x (HMul.hMul z y))
  -/
  rw [div_eq_mul_inv, ENNReal.mul_inv_le_iff h1 h2]
  /-
    🎉 no goals
  -/


protected theorem div_le_iff' {x y z : ℝ≥0∞} (h1 : y ≠ 0) (h2 : y ≠ ∞) :
    x / y ≤ z ↔ x ≤ y * z := by
  /-
    x y z : ENNReal
    h1 : Ne y 0
    h2 : Ne y Top.top
    ⊢ Iff (LE.le (HDiv.hDiv x y) z) (LE.le x (HMul.hMul y z))
  -/
  rw [mul_comm, ENNReal.div_le_iff h1 h2]
  /-
    🎉 no goals
  -/


protected theorem mul_inv {a b : ℝ≥0∞} (ha : a ≠ 0 ∨ b ≠ ∞) (hb : a ≠ ∞ ∨ b ≠ 0) :
    (a * b)⁻¹ = a⁻¹ * b⁻¹ := by
  /-
    a b : ENNReal
    ha : Or (Ne a 0) (Ne b Top.top)
    hb : Or (Ne a Top.top) (Ne b 0)
    ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (Inv.inv b))
  -/
  induction' b with b
    /-
      case top
      a : ENNReal
      ha : Or (Ne a 0) (Ne Top.top Top.top)
      hb : Or (Ne a Top.top) (Ne Top.top 0)
      ⊢ Eq (Inv.inv (HMul.hMul a Top.top)) (HMul.hMul (Inv.inv a) (Inv.inv Top.top))
    -/
  · replace ha : a ≠ 0 := ha.neg_resolve_right rfl
    /-
      case top
      a : ENNReal
      hb : Or (Ne a Top.top) (Ne Top.top 0)
      ha : Ne a 0
      ⊢ Eq (Inv.inv (HMul.hMul a Top.top)) (HMul.hMul (Inv.inv a) (Inv.inv Top.top))
    -/
    simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case coe
    a : ENNReal
    b : NNReal
    ha : Or (Ne a 0) (Ne (↑b) Top.top)
    hb : Or (Ne a Top.top) (Ne (↑b) 0)
    ⊢ Eq (Inv.inv (HMul.hMul a ↑b)) (HMul.hMul (Inv.inv a) (Inv.inv ↑b))
  -/
  induction' a with a
    /-
      case coe.top
      b : NNReal
      ha : Or (Ne Top.top 0) (Ne (↑b) Top.top)
      hb : Or (Ne Top.top Top.top) (Ne (↑b) 0)
      ⊢ Eq (Inv.inv (HMul.hMul Top.top ↑b)) (HMul.hMul (Inv.inv Top.top) (Inv.inv ↑b))
    -/
  · replace hb : b ≠ 0 := coe_ne_zero.1 (hb.neg_resolve_left rfl)
    /-
      case coe.top
      b : NNReal
      ha : Or (Ne Top.top 0) (Ne (↑b) Top.top)
      hb : Ne b 0
      ⊢ Eq (Inv.inv (HMul.hMul Top.top ↑b)) (HMul.hMul (Inv.inv Top.top) (Inv.inv ↑b))
    -/
    simp [hb]
    /-
      🎉 no goals
    -/
  /-
    case coe.coe
    b a : NNReal
    ha : Or (Ne (↑a) 0) (Ne (↑b) Top.top)
    hb : Or (Ne (↑a) Top.top) (Ne (↑b) 0)
    ⊢ Eq (Inv.inv (HMul.hMul ↑a ↑b)) (HMul.hMul (Inv.inv ↑a) (Inv.inv ↑b))
  -/
  by_cases h'a : a = 0
  · simp only [h'a, top_mul, ENNReal.inv_zero, ENNReal.coe_ne_top, zero_mul, Ne,
      not_false_iff, ENNReal.coe_zero, ENNReal.inv_eq_zero]
  /-
    case neg
    b a : NNReal
    ha : Or (Ne (↑a) 0) (Ne (↑b) Top.top)
    hb : Or (Ne (↑a) Top.top) (Ne (↑b) 0)
    h'a : Not (Eq a 0)
    ⊢ Eq (Inv.inv (HMul.hMul ↑a ↑b)) (HMul.hMul (Inv.inv ↑a) (Inv.inv ↑b))
  -/
  by_cases h'b : b = 0
  · simp only [h'b, ENNReal.inv_zero, ENNReal.coe_ne_top, mul_top, Ne, not_false_iff,
      mul_zero, ENNReal.coe_zero, ENNReal.inv_eq_zero]
  rw [← ENNReal.coe_mul, ← ENNReal.coe_inv, ← ENNReal.coe_inv h'a, ← ENNReal.coe_inv h'b, ←
    ENNReal.coe_mul, mul_inv_rev, mul_comm]
  /-
    case neg
    b a : NNReal
    ha : Or (Ne (↑a) 0) (Ne (↑b) Top.top)
    hb : Or (Ne (↑a) Top.top) (Ne (↑b) 0)
    h'a : Not (Eq a 0)
    h'b : Not (Eq b 0)
    ⊢ Ne (HMul.hMul a b) 0
  -/
  simp [h'a, h'b]
  /-
    🎉 no goals
  -/


protected theorem inv_div {a b : ℝ≥0∞} (htop : b ≠ ∞ ∨ a ≠ ∞) (hzero : b ≠ 0 ∨ a ≠ 0) :
    (a / b)⁻¹ = b / a := by
  /-
    a b : ENNReal
    htop : Or (Ne b Top.top) (Ne a Top.top)
    hzero : Or (Ne b 0) (Ne a 0)
    ⊢ Eq (Inv.inv (HDiv.hDiv a b)) (HDiv.hDiv b a)
  -/
  rw [← ENNReal.inv_ne_zero] at htop
  /-
    a b : ENNReal
    htop : Or (Ne (Inv.inv b) 0) (Ne a Top.top)
    hzero : Or (Ne b 0) (Ne a 0)
    ⊢ Eq (Inv.inv (HDiv.hDiv a b)) (HDiv.hDiv b a)
  -/
  rw [← ENNReal.inv_ne_top] at hzero
  /-
    a b : ENNReal
    htop : Or (Ne (Inv.inv b) 0) (Ne a Top.top)
    hzero : Or (Ne (Inv.inv b) Top.top) (Ne a 0)
    ⊢ Eq (Inv.inv (HDiv.hDiv a b)) (HDiv.hDiv b a)
  -/
  rw [ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, ENNReal.mul_inv htop hzero, mul_comm, inv_inv]
  /-
    🎉 no goals
  -/


lemma prod_inv_distrib {ι : Type*} {f : ι → ℝ≥0∞} {s : Finset ι}
    (hf : s.toSet.Pairwise fun i j ↦ f i ≠ 0 ∨ f j ≠ ∞) : (∏ i ∈ s, f i)⁻¹ = ∏ i ∈ s, (f i)⁻¹ := by
  /-
    ι : Type u_1
    f : ι → ENNReal
    s : Finset ι
    hf : (↑s).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top.top)
    ⊢ Eq (Inv.inv (s.prod fun i => f i)) (s.prod fun i => Inv.inv (f i))
  -/
  induction' s using Finset.cons_induction with i s hi ih
    /-
      case empty
      ι : Type u_1
      f : ι → ENNReal
      hf : (↑EmptyCollection.emptyCollection).Pairwise fun i j => Or (Ne (f i) 0) (N …
      ⊢ Eq (Inv.inv (EmptyCollection.emptyCollection.prod fun i => f i)) (EmptyColle …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    ι : Type u_1
    f : ι → ENNReal
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ((↑s).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top.top)) → Eq (Inv.i …
    hf : (↑(Finset.cons i s hi)).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top …
    ⊢ Eq (Inv.inv ((Finset.cons i s hi).prod fun i => f i)) ((Finset.cons i s hi). …
  -/
  simp [← ih (hf.mono <| by simp)]
  refine ENNReal.mul_inv (not_or_of_imp fun hi₀ ↦ prod_ne_top fun j hj ↦ ?_)
    (not_or_of_imp fun hi₀ ↦ Finset.prod_ne_zero_iff.2 fun j hj ↦ ?_)
    /-
      case cons.refine_1
      ι : Type u_1
      f : ι → ENNReal
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ((↑s).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top.top)) → Eq (Inv.i …
      hf : (↑(Finset.cons i s hi)).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top …
      hi₀ : Eq (f i) 0
      j : ι
      hj : Membership.mem s j
      ⊢ Ne (f j) Top.top
    -/
  · exact imp_iff_not_or.2 (hf (by simp) (by simp [hj]) <| .symm <| ne_of_mem_of_not_mem hj hi) hi₀
    /-
      🎉 no goals
    -/
    /-
      case cons.refine_2
      ι : Type u_1
      f : ι → ENNReal
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ((↑s).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top.top)) → Eq (Inv.i …
      hf : (↑(Finset.cons i s hi)).Pairwise fun i j => Or (Ne (f i) 0) (Ne (f j) Top …
      hi₀ : Eq (f i) Top.top
      j : ι
      hj : Membership.mem s j
      ⊢ Ne (f j) 0
    -/
  · exact imp_iff_not_or.2 (hf (by simp [hj]) (by simp) <| ne_of_mem_of_not_mem hj hi).symm hi₀
    /-
      🎉 no goals
    -/


protected theorem mul_div_mul_left (a b : ℝ≥0∞) (hc : c ≠ 0) (hc' : c ≠ ⊤) :
    c * a / (c * b) = a / b := by
  rw [div_eq_mul_inv, div_eq_mul_inv, ENNReal.mul_inv (Or.inl hc) (Or.inl hc'), mul_mul_mul_comm,
    ENNReal.mul_inv_cancel hc hc', one_mul]


protected theorem mul_div_mul_right (a b : ℝ≥0∞) (hc : c ≠ 0) (hc' : c ≠ ⊤) :
    a * c / (b * c) = a / b := by
  rw [div_eq_mul_inv, div_eq_mul_inv, ENNReal.mul_inv (Or.inr hc') (Or.inr hc), mul_mul_mul_comm,
    ENNReal.mul_inv_cancel hc hc', mul_one]


protected theorem sub_div (h : 0 < b → b < a → c ≠ 0) : (a - b) / c = a / c - b / c := by
  /-
    a b c : ENNReal
    h : LT.lt 0 b → LT.lt b a → Ne c 0
    ⊢ Eq (HDiv.hDiv (HSub.hSub a b) c) (HSub.hSub (HDiv.hDiv a c) (HDiv.hDiv b c))
  -/
  simp_rw [div_eq_mul_inv]
  /-
    a b c : ENNReal
    h : LT.lt 0 b → LT.lt b a → Ne c 0
    ⊢ Eq (HMul.hMul (HSub.hSub a b) (Inv.inv c)) (HSub.hSub (HMul.hMul a (Inv.inv  …
  -/
  exact ENNReal.sub_mul (by simpa using h)
  /-
    🎉 no goals
  -/


@[simp]
protected theorem inv_pos : 0 < a⁻¹ ↔ a ≠ ∞ :=
  pos_iff_ne_zero.trans ENNReal.inv_ne_zero


theorem inv_strictAnti : StrictAnti (Inv.inv : ℝ≥0∞ → ℝ≥0∞) := by
  /-
    ⊢ StrictAnti Inv.inv
  -/
  intro a b h
  /-
    a b : ENNReal
    h : LT.lt a b
    ⊢ LT.lt (Inv.inv b) (Inv.inv a)
  -/
  lift a to ℝ≥0 using h.ne_top
  /-
    case intro
    b : ENNReal
    a : NNReal
    h : LT.lt (↑a) b
    ⊢ LT.lt (Inv.inv b) (Inv.inv ↑a)
  -/
  induction b; · simp
                 /-
                   🎉 no goals
                 -/
  /-
    case intro.coe
    a x✝ : NNReal
    h : LT.lt ↑a ↑x✝
    ⊢ LT.lt (Inv.inv ↑x✝) (Inv.inv ↑a)
  -/
  rw [coe_lt_coe] at h
  /-
    case intro.coe
    a x✝ : NNReal
    h : LT.lt a x✝
    ⊢ LT.lt (Inv.inv ↑x✝) (Inv.inv ↑a)
  -/
  rcases eq_or_ne a 0 with (rfl | ha); · simp [h]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case intro.coe.inr
    a x✝ : NNReal
    h : LT.lt a x✝
    ha : Ne a 0
    ⊢ LT.lt (Inv.inv ↑x✝) (Inv.inv ↑a)
  -/
  rw [← coe_inv h.ne_bot, ← coe_inv ha, coe_lt_coe]
  /-
    case intro.coe.inr
    a x✝ : NNReal
    h : LT.lt a x✝
    ha : Ne a 0
    ⊢ LT.lt (Inv.inv x✝) (Inv.inv a)
  -/
  exact NNReal.inv_lt_inv ha h
  /-
    🎉 no goals
  -/


@[simp]
protected theorem inv_lt_inv : a⁻¹ < b⁻¹ ↔ b < a :=
  inv_strictAnti.lt_iff_lt


theorem inv_lt_iff_inv_lt : a⁻¹ < b ↔ b⁻¹ < a := by
  /-
    a b : ENNReal
    ⊢ Iff (LT.lt (Inv.inv a) b) (LT.lt (Inv.inv b) a)
  -/
  simpa only [inv_inv] using @ENNReal.inv_lt_inv a b⁻¹
  /-
    🎉 no goals
  -/


theorem lt_inv_iff_lt_inv : a < b⁻¹ ↔ b < a⁻¹ := by
  /-
    a b : ENNReal
    ⊢ Iff (LT.lt a (Inv.inv b)) (LT.lt b (Inv.inv a))
  -/
  simpa only [inv_inv] using @ENNReal.inv_lt_inv a⁻¹ b
  /-
    🎉 no goals
  -/


@[simp]
protected theorem inv_le_inv : a⁻¹ ≤ b⁻¹ ↔ b ≤ a :=
  inv_strictAnti.le_iff_le


theorem inv_le_iff_inv_le : a⁻¹ ≤ b ↔ b⁻¹ ≤ a := by
  /-
    a b : ENNReal
    ⊢ Iff (LE.le (Inv.inv a) b) (LE.le (Inv.inv b) a)
  -/
  simpa only [inv_inv] using @ENNReal.inv_le_inv a b⁻¹
  /-
    🎉 no goals
  -/


theorem le_inv_iff_le_inv : a ≤ b⁻¹ ↔ b ≤ a⁻¹ := by
  /-
    a b : ENNReal
    ⊢ Iff (LE.le a (Inv.inv b)) (LE.le b (Inv.inv a))
  -/
  simpa only [inv_inv] using @ENNReal.inv_le_inv a⁻¹ b
  /-
    🎉 no goals
  -/


@[gcongr] protected theorem inv_le_inv' (h : a ≤ b) : b⁻¹ ≤ a⁻¹ :=
  ENNReal.inv_strictAnti.antitone h


@[gcongr] protected theorem inv_lt_inv' (h : a < b) : b⁻¹ < a⁻¹ := ENNReal.inv_strictAnti h


@[simp]
                                                     /-
                                                       a : ENNReal
                                                       ⊢ Iff (LE.le (Inv.inv a) 1) (LE.le 1 a)
                                                     -/
protected theorem inv_le_one : a⁻¹ ≤ 1 ↔ 1 ≤ a := by rw [inv_le_iff_inv_le, inv_one]
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                     /-
                                                       a : ENNReal
                                                       ⊢ Iff (LE.le 1 (Inv.inv a)) (LE.le a 1)
                                                     -/
protected theorem one_le_inv : 1 ≤ a⁻¹ ↔ a ≤ 1 := by rw [le_inv_iff_le_inv, inv_one]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                                     /-
                                                       a : ENNReal
                                                       ⊢ Iff (LT.lt (Inv.inv a) 1) (LT.lt 1 a)
                                                     -/
protected theorem inv_lt_one : a⁻¹ < 1 ↔ 1 < a := by rw [inv_lt_iff_inv_lt, inv_one]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                                     /-
                                                       a : ENNReal
                                                       ⊢ Iff (LT.lt 1 (Inv.inv a)) (LT.lt a 1)
                                                     -/
protected theorem one_lt_inv : 1 < a⁻¹ ↔ a < 1 := by rw [lt_inv_iff_lt_inv, inv_one]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The inverse map `fun x ↦ x⁻¹` is an order isomorphism between `ℝ≥0∞` and its `OrderDual` -/
@[simps! apply]
def _root_.OrderIso.invENNReal : ℝ≥0∞ ≃o ℝ≥0∞ᵒᵈ where
  map_rel_iff' := ENNReal.inv_le_inv
  toEquiv := (Equiv.inv ℝ≥0∞).trans OrderDual.toDual


@[simp]
theorem _root_.OrderIso.invENNReal_symm_apply (a : ℝ≥0∞ᵒᵈ) :
    OrderIso.invENNReal.symm a = (OrderDual.ofDual a)⁻¹ :=
  rfl


                                          /-
                                            a : ENNReal
                                            ⊢ Eq (HDiv.hDiv a Top.top) 0
                                          -/
@[simp] theorem div_top : a / ∞ = 0 := by rw [div_eq_mul_inv, inv_top, mul_zero]
                                          /-
                                            🎉 no goals
                                          -/

-- Porting note: reordered 4 lemmas


                                                       /-
                                                         a : ENNReal
                                                         ⊢ Eq (HDiv.hDiv Top.top a) (ite (Eq a Top.top) 0 Top.top)
                                                       -/
theorem top_div : ∞ / a = if a = ∞ then 0 else ∞ := by simp [div_eq_mul_inv, top_mul']
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                        /-
                                                          a : ENNReal
                                                          h : Ne a Top.top
                                                          ⊢ Eq (HDiv.hDiv Top.top a) Top.top
                                                        -/
theorem top_div_of_ne_top (h : a ≠ ∞) : ∞ / a = ∞ := by simp [top_div, h]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] theorem top_div_coe : ∞ / p = ∞ := top_div_of_ne_top coe_ne_top


theorem top_div_of_lt_top (h : a < ∞) : ∞ / a = ∞ := top_div_of_ne_top h.ne


@[simp] protected theorem zero_div : 0 / a = 0 := zero_mul a⁻¹


theorem div_eq_top : a / b = ∞ ↔ a ≠ 0 ∧ b = 0 ∨ a = ∞ ∧ b ≠ ∞ := by
  /-
    a b : ENNReal
    ⊢ Iff (Eq (HDiv.hDiv a b) Top.top) (Or (And (Ne a 0) (Eq b 0)) (And (Eq a Top. …
  -/
  simp [div_eq_mul_inv, ENNReal.mul_eq_top]
  /-
    🎉 no goals
  -/


protected theorem le_div_iff_mul_le (h0 : b ≠ 0 ∨ c ≠ 0) (ht : b ≠ ∞ ∨ c ≠ ∞) :
    a ≤ c / b ↔ a * b ≤ c := by
  /-
    a b c : ENNReal
    h0 : Or (Ne b 0) (Ne c 0)
    ht : Or (Ne b Top.top) (Ne c Top.top)
    ⊢ Iff (LE.le a (HDiv.hDiv c b)) (LE.le (HMul.hMul a b) c)
  -/
  induction' b with b
    /-
      case top
      a b c : ENNReal
      h0 : Or (Ne Top.top 0) (Ne c 0)
      ht : Or (Ne Top.top Top.top) (Ne c Top.top)
      ⊢ Iff (LE.le a (HDiv.hDiv c Top.top)) (LE.le (HMul.hMul a Top.top) c)
    -/
  · lift c to ℝ≥0 using ht.neg_resolve_left rfl
    /-
      case top.intro
      a b : ENNReal
      c : NNReal
      h0 : Or (Ne Top.top 0) (Ne (↑c) 0)
      ht : Or (Ne Top.top Top.top) (Ne (↑c) Top.top)
      ⊢ Iff (LE.le a (HDiv.hDiv (↑c) Top.top)) (LE.le (HMul.hMul a Top.top) ↑c)
    -/
    rw [div_top, nonpos_iff_eq_zero]
    /-
      case top.intro
      a b : ENNReal
      c : NNReal
      h0 : Or (Ne Top.top 0) (Ne (↑c) 0)
      ht : Or (Ne Top.top Top.top) (Ne (↑c) Top.top)
      ⊢ Iff (Eq a 0) (LE.le (HMul.hMul a Top.top) ↑c)
    -/
                                            /-
                                              🎉 no goals
                                            -/
    rcases eq_or_ne a 0 with (rfl | ha) <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/
  /-
    case coe
    a b✝ c : ENNReal
    b : NNReal
    h0 : Or (Ne (↑b) 0) (Ne c 0)
    ht : Or (Ne (↑b) Top.top) (Ne c Top.top)
    ⊢ Iff (LE.le a (HDiv.hDiv c ↑b)) (LE.le (HMul.hMul a ↑b) c)
  -/
  rcases eq_or_ne b 0 with (rfl | hb)
    /-
      case coe.inl
      a b c : ENNReal
      h0 : Or (Ne (↑0) 0) (Ne c 0)
      ht : Or (Ne (↑0) Top.top) (Ne c Top.top)
      ⊢ Iff (LE.le a (HDiv.hDiv c ↑0)) (LE.le (HMul.hMul a ↑0) c)
    -/
  · have hc : c ≠ 0 := h0.neg_resolve_left rfl
    /-
      case coe.inl
      a b c : ENNReal
      h0 : Or (Ne (↑0) 0) (Ne c 0)
      ht : Or (Ne (↑0) Top.top) (Ne c Top.top)
      hc : Ne c 0
      ⊢ Iff (LE.le a (HDiv.hDiv c ↑0)) (LE.le (HMul.hMul a ↑0) c)
    -/
    simp [div_zero hc]
    /-
      🎉 no goals
    -/
    /-
      case coe.inr
      a b✝ c : ENNReal
      b : NNReal
      h0 : Or (Ne (↑b) 0) (Ne c 0)
      ht : Or (Ne (↑b) Top.top) (Ne c Top.top)
      hb : Ne b 0
      ⊢ Iff (LE.le a (HDiv.hDiv c ↑b)) (LE.le (HMul.hMul a ↑b) c)
    -/
  · rw [← coe_ne_zero] at hb
    /-
      case coe.inr
      a b✝ c : ENNReal
      b : NNReal
      h0 : Or (Ne (↑b) 0) (Ne c 0)
      ht : Or (Ne (↑b) Top.top) (Ne c Top.top)
      hb : Ne (↑b) 0
      ⊢ Iff (LE.le a (HDiv.hDiv c ↑b)) (LE.le (HMul.hMul a ↑b) c)
    -/
    rw [← ENNReal.mul_le_mul_right hb coe_ne_top, ENNReal.div_mul_cancel hb coe_ne_top]
    /-
      🎉 no goals
    -/


protected theorem div_le_iff_le_mul (hb0 : b ≠ 0 ∨ c ≠ ∞) (hbt : b ≠ ∞ ∨ c ≠ 0) :
    a / b ≤ c ↔ a ≤ c * b := by
  /-
    a b c : ENNReal
    hb0 : Or (Ne b 0) (Ne c Top.top)
    hbt : Or (Ne b Top.top) (Ne c 0)
    ⊢ Iff (LE.le (HDiv.hDiv a b) c) (LE.le a (HMul.hMul c b))
  -/
  suffices a * b⁻¹ ≤ c ↔ a ≤ c / b⁻¹ by simpa [div_eq_mul_inv]
  /-
    a b c : ENNReal
    hb0 : Or (Ne b 0) (Ne c Top.top)
    hbt : Or (Ne b Top.top) (Ne c 0)
    ⊢ Iff (LE.le (HMul.hMul a (Inv.inv b)) c) (LE.le a (HDiv.hDiv c (Inv.inv b)))
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  refine (ENNReal.le_div_iff_mul_le ?_ ?_).symm <;> simpa
                                                    /-
                                                      🎉 no goals
                                                    -/


protected theorem lt_div_iff_mul_lt (hb0 : b ≠ 0 ∨ c ≠ ∞) (hbt : b ≠ ∞ ∨ c ≠ 0) :
    c < a / b ↔ c * b < a :=
  lt_iff_lt_of_le_iff_le (ENNReal.div_le_iff_le_mul hb0 hbt)


theorem div_le_of_le_mul (h : a ≤ b * c) : a / c ≤ b := by
  /-
    a b c : ENNReal
    h : LE.le a (HMul.hMul b c)
    ⊢ LE.le (HDiv.hDiv a c) b
  -/
  by_cases h0 : c = 0
    /-
      case pos
      a b c : ENNReal
      h : LE.le a (HMul.hMul b c)
      h0 : Eq c 0
      ⊢ LE.le (HDiv.hDiv a c) b
    -/
  · have : a = 0 := by simpa [h0] using h
    /-
      case pos
      a b c : ENNReal
      h : LE.le a (HMul.hMul b c)
      h0 : Eq c 0
      this : Eq a 0
      ⊢ LE.le (HDiv.hDiv a c) b
    -/
    simp [*]
    /-
      🎉 no goals
    -/
  /-
    case neg
    a b c : ENNReal
    h : LE.le a (HMul.hMul b c)
    h0 : Not (Eq c 0)
    ⊢ LE.le (HDiv.hDiv a c) b
  -/
  by_cases hinf : c = ∞; · simp [hinf]
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    a b c : ENNReal
    h : LE.le a (HMul.hMul b c)
    h0 : Not (Eq c 0)
    hinf : Not (Eq c Top.top)
    ⊢ LE.le (HDiv.hDiv a c) b
  -/
  exact (ENNReal.div_le_iff_le_mul (Or.inl h0) (Or.inl hinf)).2 h
  /-
    🎉 no goals
  -/


theorem div_le_of_le_mul' (h : a ≤ b * c) : a / b ≤ c :=
  div_le_of_le_mul <| mul_comm b c ▸ h


                                                                                /-
                                                                                  a : ENNReal
                                                                                  ⊢ LE.le a (HMul.hMul 1 a)
                                                                                -/
@[simp] protected theorem div_self_le_one : a / a ≤ 1 := div_le_of_le_mul <| by rw [one_mul]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp] protected lemma mul_inv_le_one (a : ℝ≥0∞) : a * a⁻¹ ≤ 1 := ENNReal.div_self_le_one

                                                                      /-
                                                                        a : ENNReal
                                                                        ⊢ LE.le (HMul.hMul (Inv.inv a) a) 1
                                                                      -/
@[simp] protected lemma inv_mul_le_one (a : ℝ≥0∞) : a⁻¹ * a ≤ 1 := by simp [mul_comm]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp] lemma mul_inv_ne_top (a : ℝ≥0∞) : a * a⁻¹ ≠ ⊤ :=
  ne_top_of_le_ne_top one_ne_top a.mul_inv_le_one


                                                            /-
                                                              a : ENNReal
                                                              ⊢ Ne (HMul.hMul (Inv.inv a) a) Top.top
                                                            -/
@[simp] lemma inv_mul_ne_top (a : ℝ≥0∞) : a⁻¹ * a ≠ ⊤ := by simp [mul_comm]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem mul_le_of_le_div (h : a ≤ b / c) : a * c ≤ b := by
  /-
    a b c : ENNReal
    h : LE.le a (HDiv.hDiv b c)
    ⊢ LE.le (HMul.hMul a c) b
  -/
  rw [← inv_inv c]
  /-
    a b c : ENNReal
    h : LE.le a (HDiv.hDiv b c)
    ⊢ LE.le (HMul.hMul a (Inv.inv (Inv.inv c))) b
  -/
  exact div_le_of_le_mul h
  /-
    🎉 no goals
  -/


theorem mul_le_of_le_div' (h : a ≤ b / c) : c * a ≤ b :=
  mul_comm a c ▸ mul_le_of_le_div h


protected theorem div_lt_iff (h0 : b ≠ 0 ∨ c ≠ 0) (ht : b ≠ ∞ ∨ c ≠ ∞) : c / b < a ↔ c < a * b :=
  lt_iff_lt_of_le_iff_le <| ENNReal.le_div_iff_mul_le h0 ht


theorem mul_lt_of_lt_div (h : a < b / c) : a * c < b := by
  /-
    a b c : ENNReal
    h : LT.lt a (HDiv.hDiv b c)
    ⊢ LT.lt (HMul.hMul a c) b
  -/
  contrapose! h
  /-
    a b c : ENNReal
    h : LE.le b (HMul.hMul a c)
    ⊢ LE.le (HDiv.hDiv b c) a
  -/
  exact ENNReal.div_le_of_le_mul h
  /-
    🎉 no goals
  -/


theorem mul_lt_of_lt_div' (h : a < b / c) : c * a < b :=
  mul_comm a c ▸ mul_lt_of_lt_div h


theorem div_lt_of_lt_mul (h : a < b * c) : a / c < b :=
                         /-
                           a b c : ENNReal
                           h : LT.lt a (HMul.hMul b c)
                           ⊢ LT.lt a (HDiv.hDiv b (Inv.inv c))
                         -/
  mul_lt_of_lt_div <| by rwa [div_eq_mul_inv, inv_inv]
                         /-
                           🎉 no goals
                         -/


theorem div_lt_of_lt_mul' (h : a < b * c) : a / b < c :=
                         /-
                           a b c : ENNReal
                           h : LT.lt a (HMul.hMul b c)
                           ⊢ LT.lt a (HMul.hMul c b)
                         -/
  div_lt_of_lt_mul <| by rwa [mul_comm]
                         /-
                           🎉 no goals
                         -/


theorem inv_le_iff_le_mul (h₁ : b = ∞ → a ≠ 0) (h₂ : a = ∞ → b ≠ 0) : a⁻¹ ≤ b ↔ 1 ≤ a * b := by
  /-
    a b : ENNReal
    h₁ : Eq b Top.top → Ne a 0
    h₂ : Eq a Top.top → Ne b 0
    ⊢ Iff (LE.le (Inv.inv a) b) (LE.le 1 (HMul.hMul a b))
  -/
  rw [← one_div, ENNReal.div_le_iff_le_mul, mul_comm]
  /-
    case hb0
    a b : ENNReal
    h₁ : Eq b Top.top → Ne a 0
    h₂ : Eq a Top.top → Ne b 0
    ⊢ Or (Ne a 0) (Ne b Top.top)
  -/
  exacts [or_not_of_imp h₁, not_or_of_imp h₂]
  /-
    🎉 no goals
  -/


@[simp 900]
theorem le_inv_iff_mul_le : a ≤ b⁻¹ ↔ a * b ≤ 1 := by
  /-
    a b : ENNReal
    ⊢ Iff (LE.le a (Inv.inv b)) (LE.le (HMul.hMul a b) 1)
  -/
  rw [← one_div, ENNReal.le_div_iff_mul_le] <;>
      /-
        case h0
        a b : ENNReal
        ⊢ Or (Ne b 0) (Ne 1 0)
      -/
      /-
        case h0.h
        a b : ENNReal
        ⊢ Ne 1 0
      -/
      /-
        🎉 no goals
      -/
      /-
        case ht.h
        a b : ENNReal
        ⊢ Ne 1 Top.top
      -/
      simp
      /-
        🎉 no goals
      -/


@[gcongr] protected theorem div_le_div (hab : a ≤ b) (hdc : d ≤ c) : a / c ≤ b / d :=
  div_eq_mul_inv b d ▸ div_eq_mul_inv a c ▸ mul_le_mul' hab (ENNReal.inv_le_inv.mpr hdc)


@[gcongr] protected theorem div_le_div_left (h : a ≤ b) (c : ℝ≥0∞) : c / b ≤ c / a :=
  ENNReal.div_le_div le_rfl h


@[gcongr] protected theorem div_le_div_right (h : a ≤ b) (c : ℝ≥0∞) : a / c ≤ b / c :=
  ENNReal.div_le_div h le_rfl


protected theorem eq_inv_of_mul_eq_one_left (h : a * b = 1) : a = b⁻¹ := by
  rw [← mul_one a, ← ENNReal.mul_inv_cancel (right_ne_zero_of_mul_eq_one h), ← mul_assoc, h,
    one_mul]
  /-
    a b : ENNReal
    h : Eq (HMul.hMul a b) 1
    ⊢ Ne b Top.top
  -/
  rintro rfl
  /-
    a : ENNReal
    h : Eq (HMul.hMul a Top.top) 1
    ⊢ False
  -/
  simp [left_ne_zero_of_mul_eq_one h] at h
  /-
    🎉 no goals
  -/


theorem mul_le_iff_le_inv {a b r : ℝ≥0∞} (hr₀ : r ≠ 0) (hr₁ : r ≠ ∞) : r * a ≤ b ↔ a ≤ r⁻¹ * b := by
  rw [← @ENNReal.mul_le_mul_left _ a _ hr₀ hr₁, ← mul_assoc, ENNReal.mul_inv_cancel hr₀ hr₁,
    one_mul]


instance : PosSMulStrictMono ℝ≥0 ℝ≥0∞ where
  elim _r hr _a _b hab := ENNReal.mul_lt_mul_left' (coe_pos.2 hr).ne' coe_ne_top hab


instance : SMulPosMono ℝ≥0 ℝ≥0∞ where
  elim _r _ _a _b hab := mul_le_mul_right' (coe_le_coe.2 hab) _


theorem le_of_forall_nnreal_lt {x y : ℝ≥0∞} (h : ∀ r : ℝ≥0, ↑r < x → ↑r ≤ y) : x ≤ y := by
  /-
    x y : ENNReal
    h : ∀ (r : NNReal), LT.lt (↑r) x → LE.le (↑r) y
    ⊢ LE.le x y
  -/
  refine le_of_forall_ge_of_dense fun r hr => ?_
  /-
    x y : ENNReal
    h : ∀ (r : NNReal), LT.lt (↑r) x → LE.le (↑r) y
    r : ENNReal
    hr : LT.lt r x
    ⊢ LE.le r y
  -/
  lift r to ℝ≥0 using ne_top_of_lt hr
  /-
    case intro
    x y : ENNReal
    h : ∀ (r : NNReal), LT.lt (↑r) x → LE.le (↑r) y
    r : NNReal
    hr : LT.lt (↑r) x
    ⊢ LE.le (↑r) y
  -/
  exact h r hr
  /-
    🎉 no goals
  -/


theorem le_of_forall_pos_nnreal_lt {x y : ℝ≥0∞} (h : ∀ r : ℝ≥0, 0 < r → ↑r < x → ↑r ≤ y) : x ≤ y :=
  le_of_forall_nnreal_lt fun r hr =>
    (zero_le r).eq_or_lt.elim (fun h => h ▸ zero_le _) fun h0 => h r h0 hr


theorem eq_top_of_forall_nnreal_le {x : ℝ≥0∞} (h : ∀ r : ℝ≥0, ↑r ≤ x) : x = ∞ :=
  top_unique <| le_of_forall_nnreal_lt fun r _ => h r


protected theorem add_div : (a + b) / c = a / c + b / c :=
  right_distrib a b c⁻¹


protected theorem div_add_div_same {a b c : ℝ≥0∞} : a / c + b / c = (a + b) / c :=
  ENNReal.add_div.symm


protected theorem div_self (h0 : a ≠ 0) (hI : a ≠ ∞) : a / a = 1 :=
  ENNReal.mul_inv_cancel h0 hI


theorem mul_div_le : a * (b / a) ≤ b :=
  mul_le_of_le_div' le_rfl


theorem eq_div_iff (ha : a ≠ 0) (ha' : a ≠ ∞) : b = c / a ↔ a * b = c :=
               /-
                 a b c : ENNReal
                 ha : Ne a 0
                 ha' : Ne a Top.top
                 h : Eq b (HDiv.hDiv c a)
                 ⊢ Eq (HMul.hMul a b) c
               -/
  ⟨fun h => by rw [h, ENNReal.mul_div_cancel ha ha'], fun h => by
               /-
                 🎉 no goals
               -/
    /-
      a b c : ENNReal
      ha : Ne a 0
      ha' : Ne a Top.top
      h : Eq (HMul.hMul a b) c
      ⊢ Eq b (HDiv.hDiv c a)
    -/
    rw [← h, mul_div_assoc, ENNReal.mul_div_cancel ha ha']⟩
    /-
      🎉 no goals
    -/


protected theorem div_eq_div_iff (ha : a ≠ 0) (ha' : a ≠ ∞) (hb : b ≠ 0) (hb' : b ≠ ∞) :
    c / b = d / a ↔ a * c = b * d := by
  /-
    a b c d : ENNReal
    ha : Ne a 0
    ha' : Ne a Top.top
    hb : Ne b 0
    hb' : Ne b Top.top
    ⊢ Iff (Eq (HDiv.hDiv c b) (HDiv.hDiv d a)) (Eq (HMul.hMul a c) (HMul.hMul b d))
  -/
  rw [eq_div_iff ha ha']
  /-
    a b c d : ENNReal
    ha : Ne a 0
    ha' : Ne a Top.top
    hb : Ne b 0
    hb' : Ne b Top.top
    ⊢ Iff (Eq (HMul.hMul a (HDiv.hDiv c b)) d) (Eq (HMul.hMul a c) (HMul.hMul b d))
  -/
  conv_rhs => rw [eq_comm]
  /-
    a b c d : ENNReal
    ha : Ne a 0
    ha' : Ne a Top.top
    hb : Ne b 0
    hb' : Ne b Top.top
    ⊢ Iff (Eq (HMul.hMul a (HDiv.hDiv c b)) d) (Eq (HMul.hMul b d) (HMul.hMul a c))
  -/
  rw [← eq_div_iff hb hb', mul_div_assoc, eq_comm]
  /-
    🎉 no goals
  -/


theorem div_eq_one_iff {a b : ℝ≥0∞} (hb₀ : b ≠ 0) (hb₁ : b ≠ ∞) : a / b = 1 ↔ a = b :=
               /-
                 a b : ENNReal
                 hb₀ : Ne b 0
                 hb₁ : Ne b Top.top
                 h : Eq (HDiv.hDiv a b) 1
                 ⊢ Eq a b
               -/
  ⟨fun h => by rw [← (eq_div_iff hb₀ hb₁).mp h.symm, mul_one], fun h =>
               /-
                 🎉 no goals
               -/
    h.symm ▸ ENNReal.div_self hb₀ hb₁⟩


theorem inv_two_add_inv_two : (2 : ℝ≥0∞)⁻¹ + 2⁻¹ = 1 := by
  /-
    ⊢ Eq (HAdd.hAdd (Inv.inv 2) (Inv.inv 2)) 1
  -/
  rw [← two_mul, ← div_eq_mul_inv, ENNReal.div_self two_ne_zero two_ne_top]
  /-
    🎉 no goals
  -/


theorem inv_three_add_inv_three : (3 : ℝ≥0∞)⁻¹ + 3⁻¹ + 3⁻¹ = 1 :=
                                                /-
                                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Inv.inv 3) (Inv.inv 3)) (Inv.inv 3)) (HMul.hMul 3  …
                                                -/
  calc (3 : ℝ≥0∞)⁻¹ + 3⁻¹ + 3⁻¹ = 3 * 3⁻¹ := by ring
                                                /-
                                                  🎉 no goals
                                                -/
                                                            /-
                                                              ⊢ Ne 3 0
                                                            -/
  _ = 1 := ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 <| by decide) coe_ne_top
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
protected theorem add_halves (a : ℝ≥0∞) : a / 2 + a / 2 = a := by
  /-
    a : ENNReal
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv a 2) (HDiv.hDiv a 2)) a
  -/
  rw [div_eq_mul_inv, ← mul_add, inv_two_add_inv_two, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_thirds (a : ℝ≥0∞) : a / 3 + a / 3 + a / 3 = a := by
  /-
    a : ENNReal
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HDiv.hDiv a 3) (HDiv.hDiv a 3)) (HDiv.hDiv a 3)) a
  -/
  rw [div_eq_mul_inv, ← mul_add, ← mul_add, inv_three_add_inv_three, mul_one]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    a b : ENNReal
                                                                    ⊢ Iff (Eq (HDiv.hDiv a b) 0) (Or (Eq a 0) (Eq b Top.top))
                                                                  -/
@[simp] theorem div_eq_zero_iff : a / b = 0 ↔ a = 0 ∨ b = ∞ := by simp [div_eq_mul_inv]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                              /-
                                                                a b : ENNReal
                                                                ⊢ Iff (LT.lt 0 (HDiv.hDiv a b)) (And (Ne a 0) (Ne b Top.top))
                                                              -/
@[simp] theorem div_pos_iff : 0 < a / b ↔ a ≠ 0 ∧ b ≠ ∞ := by simp [pos_iff_ne_zero, not_or]
                                                              /-
                                                                🎉 no goals
                                                              -/


protected lemma div_ne_zero : a / b ≠ 0 ↔ a ≠ 0 ∧ b ≠ ⊤ := by
  /-
    a b : ENNReal
    ⊢ Iff (Ne (HDiv.hDiv a b) 0) (And (Ne a 0) (Ne b Top.top))
  -/
  rw [← pos_iff_ne_zero, div_pos_iff]
  /-
    🎉 no goals
  -/


protected theorem half_pos (h : a ≠ 0) : 0 < a / 2 := by
  /-
    a : ENNReal
    h : Ne a 0
    ⊢ LT.lt 0 (HDiv.hDiv a 2)
  -/
  simp only [div_pos_iff, ne_eq, h, not_false_eq_true, two_ne_top, and_self]
  /-
    🎉 no goals
  -/


protected theorem one_half_lt_one : (2⁻¹ : ℝ≥0∞) < 1 :=
  ENNReal.inv_lt_one.2 <| one_lt_two


protected theorem half_lt_self (hz : a ≠ 0) (ht : a ≠ ∞) : a / 2 < a := by
  /-
    a : ENNReal
    hz : Ne a 0
    ht : Ne a Top.top
    ⊢ LT.lt (HDiv.hDiv a 2) a
  -/
  lift a to ℝ≥0 using ht
  /-
    case intro
    a : NNReal
    hz : Ne (↑a) 0
    ⊢ LT.lt (HDiv.hDiv (↑a) 2) ↑a
  -/
  rw [coe_ne_zero] at hz
  /-
    case intro
    a : NNReal
    hz : Ne a 0
    ⊢ LT.lt (HDiv.hDiv (↑a) 2) ↑a
  -/
  rw [← coe_two, ← coe_div, coe_lt_coe]
  /-
    case intro
    a : NNReal
    hz : Ne a 0
    ⊢ LT.lt (HDiv.hDiv a 2) a
  -/
  exacts [NNReal.half_lt_self hz, two_ne_zero' _]
  /-
    🎉 no goals
  -/


protected theorem half_le_self : a / 2 ≤ a :=
  le_add_self.trans_eq <| ENNReal.add_halves _


theorem sub_half (h : a ≠ ∞) : a - a / 2 = a / 2 := ENNReal.sub_eq_of_eq_add' h a.add_halves.symm


@[simp]
theorem one_sub_inv_two : (1 : ℝ≥0∞) - 2⁻¹ = 2⁻¹ := by
  /-
    ⊢ Eq (HSub.hSub 1 (Inv.inv 2)) (Inv.inv 2)
  -/
  simpa only [div_eq_mul_inv, one_mul] using sub_half one_ne_top
  /-
    🎉 no goals
  -/


private lemma exists_lt_mul_left {a b c : ℝ≥0∞} (hc : c < a * b) : ∃ a' < a, c < a' * b := by
  /-
    a b c : ENNReal
    hc : LT.lt c (HMul.hMul a b)
    ⊢ Exists fun a' => And (LT.lt a' a) (LT.lt c (HMul.hMul a' b))
  -/
  obtain ⟨a', hc, ha'⟩ := exists_between (ENNReal.div_lt_of_lt_mul hc)
  exact ⟨_, ha', (ENNReal.div_lt_iff (.inl <| by rintro rfl; simp at *)
    (.inr <| by rintro rfl; simp at *)).1 hc⟩


private lemma exists_lt_mul_right {a b c : ℝ≥0∞} (hc : c < a * b) : ∃ b' < b, c < a * b' := by
  /-
    a b c : ENNReal
    hc : LT.lt c (HMul.hMul a b)
    ⊢ Exists fun b' => And (LT.lt b' b) (LT.lt c (HMul.hMul a b'))
  -/
  simp_rw [mul_comm a] at hc ⊢; exact exists_lt_mul_left hc
                                /-
                                  🎉 no goals
                                -/


lemma mul_le_of_forall_lt {a b c : ℝ≥0∞} (h : ∀ a' < a, ∀ b' < b, a' * b' ≤ c) : a * b ≤ c := by
  /-
    a b c : ENNReal
    h : ∀ (a' : ENNReal), LT.lt a' a → ∀ (b' : ENNReal), LT.lt b' b → LE.le (HMul. …
    ⊢ LE.le (HMul.hMul a b) c
  -/
  refine le_of_forall_ge_of_dense fun d hd ↦ ?_
  /-
    a b c : ENNReal
    h : ∀ (a' : ENNReal), LT.lt a' a → ∀ (b' : ENNReal), LT.lt b' b → LE.le (HMul. …
    d : ENNReal
    hd : LT.lt d (HMul.hMul a b)
    ⊢ LE.le d c
  -/
  obtain ⟨a', ha', hd⟩ := exists_lt_mul_left hd
  /-
    case intro.intro
    a b c : ENNReal
    h : ∀ (a' : ENNReal), LT.lt a' a → ∀ (b' : ENNReal), LT.lt b' b → LE.le (HMul. …
    d : ENNReal
    hd✝ : LT.lt d (HMul.hMul a b)
    a' : ENNReal
    ha' : LT.lt a' a
    hd : LT.lt d (HMul.hMul a' b)
    ⊢ LE.le d c
  -/
  obtain ⟨b', hb', hd⟩ := exists_lt_mul_right hd
  /-
    case intro.intro.intro.intro
    a b c : ENNReal
    h : ∀ (a' : ENNReal), LT.lt a' a → ∀ (b' : ENNReal), LT.lt b' b → LE.le (HMul. …
    d : ENNReal
    hd✝¹ : LT.lt d (HMul.hMul a b)
    a' : ENNReal
    ha' : LT.lt a' a
    hd✝ : LT.lt d (HMul.hMul a' b)
    b' : ENNReal
    hb' : LT.lt b' b
    hd : LT.lt d (HMul.hMul a' b')
    ⊢ LE.le d c
  -/
  exact le_trans hd.le <| h _ ha' _ hb'
  /-
    🎉 no goals
  -/


lemma le_mul_of_forall_lt {a b c : ℝ≥0∞} (h₁ : a ≠ 0 ∨ b ≠ ∞) (h₂ : a ≠ ∞ ∨ b ≠ 0)
    (h : ∀ a' > a, ∀ b' > b, c ≤ a' * b') : c ≤ a * b := by
  /-
    a b c : ENNReal
    h₁ : Or (Ne a 0) (Ne b Top.top)
    h₂ : Or (Ne a Top.top) (Ne b 0)
    h : ∀ (a' : ENNReal), GT.gt a' a → ∀ (b' : ENNReal), GT.gt b' b → LE.le c (HMu …
    ⊢ LE.le c (HMul.hMul a b)
  -/
  rw [← ENNReal.inv_le_inv, ENNReal.mul_inv h₁ h₂]
  exact mul_le_of_forall_lt fun a' ha' b' hb' ↦ ENNReal.le_inv_iff_le_inv.1 <|
    (h _ (ENNReal.lt_inv_iff_lt_inv.1 ha') _ (ENNReal.lt_inv_iff_lt_inv.1 hb')).trans_eq
    (ENNReal.mul_inv (Or.inr hb'.ne_top) (Or.inl ha'.ne_top)).symm


/-- The birational order isomorphism between `ℝ≥0∞` and the unit interval `Set.Iic (1 : ℝ≥0∞)`. -/
@[simps! apply_coe]
def orderIsoIicOneBirational : ℝ≥0∞ ≃o Iic (1 : ℝ≥0∞) := by
  refine StrictMono.orderIsoOfRightInverse
    (fun x => ⟨(x⁻¹ + 1)⁻¹, ENNReal.inv_le_one.2 <| le_add_self⟩)
    (fun x y hxy => ?_) (fun x => (x.1⁻¹ - 1)⁻¹) fun x => Subtype.ext ?_
    /-
      case refine_1
      a b c d : ENNReal
      r p q : NNReal
      x y : ENNReal
      hxy : LT.lt x y
      ⊢ LT.lt ((fun x => ⟨Inv.inv (HAdd.hAdd (Inv.inv x) 1), ⋯⟩) x) ((fun x => ⟨Inv. …
    -/
  · simpa only [Subtype.mk_lt_mk, ENNReal.inv_lt_inv, ENNReal.add_lt_add_iff_right one_ne_top]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b c d : ENNReal
      r p q : NNReal
      x : ↑(Set.Iic 1)
      ⊢ Eq ↑((fun x => ⟨Inv.inv (HAdd.hAdd (Inv.inv x) 1), ⋯⟩) ((fun x => Inv.inv (H …
    -/
  · have : (1 : ℝ≥0∞) ≤ x.1⁻¹ := ENNReal.one_le_inv.2 x.2
    /-
      case refine_2
      a b c d : ENNReal
      r p q : NNReal
      x : ↑(Set.Iic 1)
      this : LE.le 1 (Inv.inv ↑x)
      ⊢ Eq ↑((fun x => ⟨Inv.inv (HAdd.hAdd (Inv.inv x) 1), ⋯⟩) ((fun x => Inv.inv (H …
    -/
    simp only [inv_inv, Subtype.coe_mk, tsub_add_cancel_of_le this]
    /-
      🎉 no goals
    -/


@[simp]
theorem orderIsoIicOneBirational_symm_apply (x : Iic (1 : ℝ≥0∞)) :
    orderIsoIicOneBirational.symm x = (x.1⁻¹ - 1)⁻¹ :=
  rfl


/-- Order isomorphism between an initial interval in `ℝ≥0∞` and an initial interval in `ℝ≥0`. -/
@[simps! apply_coe]
def orderIsoIicCoe (a : ℝ≥0) : Iic (a : ℝ≥0∞) ≃o Iic a :=
  OrderIso.symm
    { toFun := fun x => ⟨x, coe_le_coe.2 x.2⟩
      invFun := fun x => ⟨ENNReal.toNNReal x, coe_le_coe.1 <| coe_toNNReal_le_self.trans x.2⟩
      left_inv := fun _ => Subtype.ext <| toNNReal_coe _
      right_inv := fun x => Subtype.ext <| coe_toNNReal (ne_top_of_le_ne_top coe_ne_top x.2)
      map_rel_iff' := fun {_ _} => by
        /-
          a✝ b c d : ENNReal
          r p q a : NNReal
          x✝¹ x✝ : ↑(Set.Iic a)
          ⊢ Iff (LE.le ({ toFun := fun x => ⟨↑↑x, ⋯⟩, invFun := fun x => ⟨(↑x).toNNReal, …
        -/
        simp only [Equiv.coe_fn_mk, Subtype.mk_le_mk, coe_le_coe, Subtype.coe_le_coe] }
        /-
          🎉 no goals
        -/


@[simp]
theorem orderIsoIicCoe_symm_apply_coe (a : ℝ≥0) (b : Iic a) :
    ((orderIsoIicCoe a).symm b : ℝ≥0∞) = b :=
  rfl


/-- An order isomorphism between the extended nonnegative real numbers and the unit interval. -/
def orderIsoUnitIntervalBirational : ℝ≥0∞ ≃o Icc (0 : ℝ) 1 :=
  orderIsoIicOneBirational.trans <| (orderIsoIicCoe 1).trans <| (NNReal.orderIsoIccZeroCoe 1).symm


@[simp]
theorem orderIsoUnitIntervalBirational_apply_coe (x : ℝ≥0∞) :
    (orderIsoUnitIntervalBirational x : ℝ) = (x⁻¹ + 1)⁻¹.toReal :=
  rfl


theorem exists_inv_nat_lt {a : ℝ≥0∞} (h : a ≠ 0) : ∃ n : ℕ, (n : ℝ≥0∞)⁻¹ < a :=
                 /-
                   a : ENNReal
                   h : Ne a 0
                   ⊢ Exists fun n => LT.lt (Inv.inv ↑n) (Inv.inv (Inv.inv a))
                 -/
  inv_inv a ▸ by simp only [ENNReal.inv_lt_inv, ENNReal.exists_nat_gt (inv_ne_top.2 h)]
                 /-
                   🎉 no goals
                 -/


theorem exists_nat_pos_mul_gt (ha : a ≠ 0) (hb : b ≠ ∞) : ∃ n > 0, b < (n : ℕ) * a :=
  let ⟨n, hn⟩ := ENNReal.exists_nat_gt (div_lt_top hb ha).ne
  ⟨n, Nat.cast_pos.1 ((zero_le _).trans_lt hn), by
    /-
      a b : ENNReal
      ha : Ne a 0
      hb : Ne b Top.top
      n : Nat
      hn : LT.lt (HDiv.hDiv b a) ↑n
      ⊢ LT.lt b (HMul.hMul (↑n) a)
    -/
    rwa [← ENNReal.div_lt_iff (Or.inl ha) (Or.inr hb)]⟩
    /-
      🎉 no goals
    -/


theorem exists_nat_mul_gt (ha : a ≠ 0) (hb : b ≠ ∞) : ∃ n : ℕ, b < n * a :=
  (exists_nat_pos_mul_gt ha hb).imp fun _ => And.right


theorem exists_nat_pos_inv_mul_lt (ha : a ≠ ∞) (hb : b ≠ 0) :
    ∃ n > 0, ((n : ℕ) : ℝ≥0∞)⁻¹ * a < b := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    ⊢ Exists fun n => And (GT.gt n 0) (LT.lt (HMul.hMul (Inv.inv ↑n) a) b)
  -/
  rcases exists_nat_pos_mul_gt hb ha with ⟨n, npos, hn⟩
  /-
    case intro.intro
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    n : Nat
    npos : GT.gt n 0
    hn : LT.lt a (HMul.hMul (↑n) b)
    ⊢ Exists fun n => And (GT.gt n 0) (LT.lt (HMul.hMul (Inv.inv ↑n) a) b)
  -/
  use n, npos
  /-
    case right
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    n : Nat
    npos : GT.gt n 0
    hn : LT.lt a (HMul.hMul (↑n) b)
    ⊢ LT.lt (HMul.hMul (Inv.inv ↑n) a) b
  -/
  rw [← ENNReal.div_eq_inv_mul]
  /-
    case right
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    n : Nat
    npos : GT.gt n 0
    hn : LT.lt a (HMul.hMul (↑n) b)
    ⊢ LT.lt (HDiv.hDiv a ↑n) b
  -/
  exact div_lt_of_lt_mul' hn
  /-
    🎉 no goals
  -/


theorem exists_nnreal_pos_mul_lt (ha : a ≠ ∞) (hb : b ≠ 0) : ∃ n > 0, ↑(n : ℝ≥0) * a < b := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    ⊢ Exists fun n => And (GT.gt n 0) (LT.lt (HMul.hMul (↑n) a) b)
  -/
  rcases exists_nat_pos_inv_mul_lt ha hb with ⟨n, npos : 0 < n, hn⟩
  /-
    case intro.intro
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    n : Nat
    npos : LT.lt 0 n
    hn : LT.lt (HMul.hMul (Inv.inv ↑n) a) b
    ⊢ Exists fun n => And (GT.gt n 0) (LT.lt (HMul.hMul (↑n) a) b)
  -/
  use (n : ℝ≥0)⁻¹
  /-
    case h
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b 0
    n : Nat
    npos : LT.lt 0 n
    hn : LT.lt (HMul.hMul (Inv.inv ↑n) a) b
    ⊢ And (GT.gt (Inv.inv ↑n) 0) (LT.lt (HMul.hMul (↑(Inv.inv ↑n)) a) b)
  -/
  simp [*, npos.ne', zero_lt_one]
  /-
    🎉 no goals
  -/


theorem exists_inv_two_pow_lt (ha : a ≠ 0) : ∃ n : ℕ, 2⁻¹ ^ n < a := by
  /-
    a : ENNReal
    ha : Ne a 0
    ⊢ Exists fun n => LT.lt (HPow.hPow (Inv.inv 2) n) a
  -/
  rcases exists_inv_nat_lt ha with ⟨n, hn⟩
  /-
    case intro
    a : ENNReal
    ha : Ne a 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) a
    ⊢ Exists fun n => LT.lt (HPow.hPow (Inv.inv 2) n) a
  -/
  refine ⟨n, lt_trans ?_ hn⟩
  /-
    case intro
    a : ENNReal
    ha : Ne a 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) a
    ⊢ LT.lt (HPow.hPow (Inv.inv 2) n) (Inv.inv ↑n)
  -/
  rw [← ENNReal.inv_pow, ENNReal.inv_lt_inv]
  /-
    case intro
    a : ENNReal
    ha : Ne a 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) a
    ⊢ LT.lt (↑n) (HPow.hPow 2 n)
  -/
  norm_cast
  /-
    case intro
    a : ENNReal
    ha : Ne a 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) a
    ⊢ LT.lt n (HPow.hPow 2 n)
  -/
  exact n.lt_two_pow_self
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_zpow (hr : r ≠ 0) (n : ℤ) : (↑(r ^ n) : ℝ≥0∞) = (r : ℝ≥0∞) ^ n := by
  /-
    r : NNReal
    hr : Ne r 0
    n : Int
    ⊢ Eq (↑(HPow.hPow r n)) (HPow.hPow (↑r) n)
  -/
  cases' n with n n
    /-
      case ofNat
      r : NNReal
      hr : Ne r 0
      n : Nat
      ⊢ Eq (↑(HPow.hPow r (Int.ofNat n))) (HPow.hPow (↑r) (Int.ofNat n))
    -/
  · simp only [Int.ofNat_eq_coe, coe_pow, zpow_natCast]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      r : NNReal
      hr : Ne r 0
      n : Nat
      ⊢ Eq (↑(HPow.hPow r (Int.negSucc n))) (HPow.hPow (↑r) (Int.negSucc n))
    -/
  · have : r ^ n.succ ≠ 0 := pow_ne_zero (n + 1) hr
    /-
      case negSucc
      r : NNReal
      hr : Ne r 0
      n : Nat
      this : Ne (HPow.hPow r n.succ) 0
      ⊢ Eq (↑(HPow.hPow r (Int.negSucc n))) (HPow.hPow (↑r) (Int.negSucc n))
    -/
    simp only [zpow_negSucc, coe_inv this, coe_pow]
    /-
      🎉 no goals
    -/


theorem zpow_pos (ha : a ≠ 0) (h'a : a ≠ ∞) (n : ℤ) : 0 < a ^ n := by
  /-
    a : ENNReal
    ha : Ne a 0
    h'a : Ne a Top.top
    n : Int
    ⊢ LT.lt 0 (HPow.hPow a n)
  -/
  cases n
    /-
      case ofNat
      a : ENNReal
      ha : Ne a 0
      h'a : Ne a Top.top
      a✝ : Nat
      ⊢ LT.lt 0 (HPow.hPow a (Int.ofNat a✝))
    -/
  · simpa using ENNReal.pow_pos ha.bot_lt _
    /-
      🎉 no goals
    -/
  · simp only [h'a, pow_eq_top_iff, zpow_negSucc, Ne, not_false, ENNReal.inv_pos, false_and,
      not_false_eq_true]


theorem zpow_lt_top (ha : a ≠ 0) (h'a : a ≠ ∞) (n : ℤ) : a ^ n < ∞ := by
  /-
    a : ENNReal
    ha : Ne a 0
    h'a : Ne a Top.top
    n : Int
    ⊢ LT.lt (HPow.hPow a n) Top.top
  -/
  cases n
    /-
      case ofNat
      a : ENNReal
      ha : Ne a 0
      h'a : Ne a Top.top
      a✝ : Nat
      ⊢ LT.lt (HPow.hPow a (Int.ofNat a✝)) Top.top
    -/
  · simpa using ENNReal.pow_lt_top h'a.lt_top _
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      a : ENNReal
      ha : Ne a 0
      h'a : Ne a Top.top
      a✝ : Nat
      ⊢ LT.lt (HPow.hPow a (Int.negSucc a✝)) Top.top
    -/
  · simp only [ENNReal.pow_pos ha.bot_lt, zpow_negSucc, inv_lt_top]
    /-
      🎉 no goals
    -/


theorem exists_mem_Ico_zpow {x y : ℝ≥0∞} (hx : x ≠ 0) (h'x : x ≠ ∞) (hy : 1 < y) (h'y : y ≠ ⊤) :
    ∃ n : ℤ, x ∈ Ico (y ^ n) (y ^ (n + 1)) := by
  /-
    x y : ENNReal
    hx : Ne x 0
    h'x : Ne x Top.top
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    ⊢ Exists fun n => Membership.mem (Set.Ico (HPow.hPow y n) (HPow.hPow y (HAdd.h …
  -/
  lift x to ℝ≥0 using h'x
  /-
    case intro
    y : ENNReal
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    x : NNReal
    hx : Ne (↑x) 0
    ⊢ Exists fun n => Membership.mem (Set.Ico (HPow.hPow y n) (HPow.hPow y (HAdd.h …
  -/
  lift y to ℝ≥0 using h'y
  /-
    case intro.intro
    x : NNReal
    hx : Ne (↑x) 0
    y : NNReal
    hy : LT.lt 1 ↑y
    ⊢ Exists fun n => Membership.mem (Set.Ico (HPow.hPow (↑y) n) (HPow.hPow (↑y) ( …
  -/
  have A : y ≠ 0 := by simpa only [Ne, coe_eq_zero] using (zero_lt_one.trans hy).ne'
  obtain ⟨n, hn, h'n⟩ : ∃ n : ℤ, y ^ n ≤ x ∧ x < y ^ (n + 1) := by
    refine NNReal.exists_mem_Ico_zpow ?_ (one_lt_coe_iff.1 hy)
    simpa only [Ne, coe_eq_zero] using hx
  /-
    case intro.intro.intro.intro
    x : NNReal
    hx : Ne (↑x) 0
    y : NNReal
    hy : LT.lt 1 ↑y
    A : Ne y 0
    n : Int
    hn : LE.le (HPow.hPow y n) x
    h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
    ⊢ Exists fun n => Membership.mem (Set.Ico (HPow.hPow (↑y) n) (HPow.hPow (↑y) ( …
  -/
  refine ⟨n, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      x : NNReal
      hx : Ne (↑x) 0
      y : NNReal
      hy : LT.lt 1 ↑y
      A : Ne y 0
      n : Int
      hn : LE.le (HPow.hPow y n) x
      h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
      ⊢ LE.le (HPow.hPow (↑y) n) ↑x
    -/
  · rwa [← ENNReal.coe_zpow A, ENNReal.coe_le_coe]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      x : NNReal
      hx : Ne (↑x) 0
      y : NNReal
      hy : LT.lt 1 ↑y
      A : Ne y 0
      n : Int
      hn : LE.le (HPow.hPow y n) x
      h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
      ⊢ LT.lt (↑x) (HPow.hPow (↑y) (HAdd.hAdd n 1))
    -/
  · rwa [← ENNReal.coe_zpow A, ENNReal.coe_lt_coe]
    /-
      🎉 no goals
    -/


theorem exists_mem_Ioc_zpow {x y : ℝ≥0∞} (hx : x ≠ 0) (h'x : x ≠ ∞) (hy : 1 < y) (h'y : y ≠ ⊤) :
    ∃ n : ℤ, x ∈ Ioc (y ^ n) (y ^ (n + 1)) := by
  /-
    x y : ENNReal
    hx : Ne x 0
    h'x : Ne x Top.top
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    ⊢ Exists fun n => Membership.mem (Set.Ioc (HPow.hPow y n) (HPow.hPow y (HAdd.h …
  -/
  lift x to ℝ≥0 using h'x
  /-
    case intro
    y : ENNReal
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    x : NNReal
    hx : Ne (↑x) 0
    ⊢ Exists fun n => Membership.mem (Set.Ioc (HPow.hPow y n) (HPow.hPow y (HAdd.h …
  -/
  lift y to ℝ≥0 using h'y
  /-
    case intro.intro
    x : NNReal
    hx : Ne (↑x) 0
    y : NNReal
    hy : LT.lt 1 ↑y
    ⊢ Exists fun n => Membership.mem (Set.Ioc (HPow.hPow (↑y) n) (HPow.hPow (↑y) ( …
  -/
  have A : y ≠ 0 := by simpa only [Ne, coe_eq_zero] using (zero_lt_one.trans hy).ne'
  obtain ⟨n, hn, h'n⟩ : ∃ n : ℤ, y ^ n < x ∧ x ≤ y ^ (n + 1) := by
    refine NNReal.exists_mem_Ioc_zpow ?_ (one_lt_coe_iff.1 hy)
    simpa only [Ne, coe_eq_zero] using hx
  /-
    case intro.intro.intro.intro
    x : NNReal
    hx : Ne (↑x) 0
    y : NNReal
    hy : LT.lt 1 ↑y
    A : Ne y 0
    n : Int
    hn : LT.lt (HPow.hPow y n) x
    h'n : LE.le x (HPow.hPow y (HAdd.hAdd n 1))
    ⊢ Exists fun n => Membership.mem (Set.Ioc (HPow.hPow (↑y) n) (HPow.hPow (↑y) ( …
  -/
  refine ⟨n, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      x : NNReal
      hx : Ne (↑x) 0
      y : NNReal
      hy : LT.lt 1 ↑y
      A : Ne y 0
      n : Int
      hn : LT.lt (HPow.hPow y n) x
      h'n : LE.le x (HPow.hPow y (HAdd.hAdd n 1))
      ⊢ LT.lt (HPow.hPow (↑y) n) ↑x
    -/
  · rwa [← ENNReal.coe_zpow A, ENNReal.coe_lt_coe]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      x : NNReal
      hx : Ne (↑x) 0
      y : NNReal
      hy : LT.lt 1 ↑y
      A : Ne y 0
      n : Int
      hn : LT.lt (HPow.hPow y n) x
      h'n : LE.le x (HPow.hPow y (HAdd.hAdd n 1))
      ⊢ LE.le (↑x) (HPow.hPow (↑y) (HAdd.hAdd n 1))
    -/
  · rwa [← ENNReal.coe_zpow A, ENNReal.coe_le_coe]
    /-
      🎉 no goals
    -/


theorem Ioo_zero_top_eq_iUnion_Ico_zpow {y : ℝ≥0∞} (hy : 1 < y) (h'y : y ≠ ⊤) :
    Ioo (0 : ℝ≥0∞) (∞ : ℝ≥0∞) = ⋃ n : ℤ, Ico (y ^ n) (y ^ (n + 1)) := by
  /-
    y : ENNReal
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    ⊢ Eq (Set.Ioo 0 Top.top) (Set.iUnion fun n => Set.Ico (HPow.hPow y n) (HPow.hP …
  -/
  ext x
  /-
    case h
    y : ENNReal
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    x : ENNReal
    ⊢ Iff (Membership.mem (Set.Ioo 0 Top.top) x) (Membership.mem (Set.iUnion fun n …
  -/
  simp only [mem_iUnion, mem_Ioo, mem_Ico]
  /-
    case h
    y : ENNReal
    hy : LT.lt 1 y
    h'y : Ne y Top.top
    x : ENNReal
    ⊢ Iff (And (LT.lt 0 x) (LT.lt x Top.top)) (Exists fun i => And (LE.le (HPow.hP …
  -/
  constructor
    /-
      case h.mp
      y : ENNReal
      hy : LT.lt 1 y
      h'y : Ne y Top.top
      x : ENNReal
      ⊢ And (LT.lt 0 x) (LT.lt x Top.top) → Exists fun i => And (LE.le (HPow.hPow y  …
    -/
  · rintro ⟨hx, h'x⟩
    /-
      case h.mp.intro
      y : ENNReal
      hy : LT.lt 1 y
      h'y : Ne y Top.top
      x : ENNReal
      hx : LT.lt 0 x
      h'x : LT.lt x Top.top
      ⊢ Exists fun i => And (LE.le (HPow.hPow y i) x) (LT.lt x (HPow.hPow y (HAdd.hA …
    -/
    exact exists_mem_Ico_zpow hx.ne' h'x.ne hy h'y
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      y : ENNReal
      hy : LT.lt 1 y
      h'y : Ne y Top.top
      x : ENNReal
      ⊢ (Exists fun i => And (LE.le (HPow.hPow y i) x) (LT.lt x (HPow.hPow y (HAdd.h …
    -/
  · rintro ⟨n, hn, h'n⟩
    /-
      case h.mpr.intro.intro
      y : ENNReal
      hy : LT.lt 1 y
      h'y : Ne y Top.top
      x : ENNReal
      n : Int
      hn : LE.le (HPow.hPow y n) x
      h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
      ⊢ And (LT.lt 0 x) (LT.lt x Top.top)
    -/
    constructor
      /-
        case h.mpr.intro.intro.left
        y : ENNReal
        hy : LT.lt 1 y
        h'y : Ne y Top.top
        x : ENNReal
        n : Int
        hn : LE.le (HPow.hPow y n) x
        h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
        ⊢ LT.lt 0 x
      -/
    · apply lt_of_lt_of_le _ hn
      /-
        y : ENNReal
        hy : LT.lt 1 y
        h'y : Ne y Top.top
        x : ENNReal
        n : Int
        hn : LE.le (HPow.hPow y n) x
        h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
        ⊢ LT.lt 0 (HPow.hPow y n)
      -/
      exact ENNReal.zpow_pos (zero_lt_one.trans hy).ne' h'y _
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro.right
        y : ENNReal
        hy : LT.lt 1 y
        h'y : Ne y Top.top
        x : ENNReal
        n : Int
        hn : LE.le (HPow.hPow y n) x
        h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
        ⊢ LT.lt x Top.top
      -/
    · apply lt_trans h'n _
      /-
        y : ENNReal
        hy : LT.lt 1 y
        h'y : Ne y Top.top
        x : ENNReal
        n : Int
        hn : LE.le (HPow.hPow y n) x
        h'n : LT.lt x (HPow.hPow y (HAdd.hAdd n 1))
        ⊢ LT.lt (HPow.hPow y (HAdd.hAdd n 1)) Top.top
      -/
      exact ENNReal.zpow_lt_top (zero_lt_one.trans hy).ne' h'y _
      /-
        🎉 no goals
      -/


@[gcongr]
theorem zpow_le_of_le {x : ℝ≥0∞} (hx : 1 ≤ x) {a b : ℤ} (h : a ≤ b) : x ^ a ≤ x ^ b := by
  /-
    x : ENNReal
    hx : LE.le 1 x
    a b : Int
    h : LE.le a b
    ⊢ LE.le (HPow.hPow x a) (HPow.hPow x b)
  -/
  induction' a with a a <;> induction' b with b b
    /-
      case ofNat.ofNat
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.ofNat a) (Int.ofNat b)
      ⊢ LE.le (HPow.hPow x (Int.ofNat a)) (HPow.hPow x (Int.ofNat b))
    -/
  · simp only [Int.ofNat_eq_coe, zpow_natCast]
    /-
      case ofNat.ofNat
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.ofNat a) (Int.ofNat b)
      ⊢ LE.le (HPow.hPow x a) (HPow.hPow x b)
    -/
    exact pow_right_mono₀ hx (Int.le_of_ofNat_le_ofNat h)
    /-
      🎉 no goals
    -/
    /-
      case ofNat.negSucc
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.ofNat a) (Int.negSucc b)
      ⊢ LE.le (HPow.hPow x (Int.ofNat a)) (HPow.hPow x (Int.negSucc b))
    -/
  · apply absurd h (not_le_of_gt _)
    /-
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.ofNat a) (Int.negSucc b)
      ⊢ GT.gt (Int.ofNat a) (Int.negSucc b)
    -/
    exact lt_of_lt_of_le (Int.negSucc_lt_zero _) (Int.ofNat_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case negSucc.ofNat
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.negSucc a) (Int.ofNat b)
      ⊢ LE.le (HPow.hPow x (Int.negSucc a)) (HPow.hPow x (Int.ofNat b))
    -/
  · simp only [zpow_negSucc, Int.ofNat_eq_coe, zpow_natCast]
    /-
      case negSucc.ofNat
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.negSucc a) (Int.ofNat b)
      ⊢ LE.le (Inv.inv (HPow.hPow x (HAdd.hAdd a 1))) (HPow.hPow x b)
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    refine (ENNReal.inv_le_one.2 ?_).trans ?_ <;> exact one_le_pow_of_one_le' hx _
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case negSucc.negSucc
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.negSucc a) (Int.negSucc b)
      ⊢ LE.le (HPow.hPow x (Int.negSucc a)) (HPow.hPow x (Int.negSucc b))
    -/
  · simp only [zpow_negSucc, ENNReal.inv_le_inv]
    /-
      case negSucc.negSucc
      x : ENNReal
      hx : LE.le 1 x
      a b : Nat
      h : LE.le (Int.negSucc a) (Int.negSucc b)
      ⊢ LE.le (HPow.hPow x (HAdd.hAdd b 1)) (HPow.hPow x (HAdd.hAdd a 1))
    -/
    apply pow_right_mono₀ hx
    simpa only [← Int.ofNat_le, neg_le_neg_iff, Int.ofNat_add, Int.ofNat_one, Int.negSucc_eq] using
      h


theorem monotone_zpow {x : ℝ≥0∞} (hx : 1 ≤ x) : Monotone ((x ^ ·) : ℤ → ℝ≥0∞) := fun _ _ h =>
  zpow_le_of_le hx h


protected theorem zpow_add {x : ℝ≥0∞} (hx : x ≠ 0) (h'x : x ≠ ∞) (m n : ℤ) :
    x ^ (m + n) = x ^ m * x ^ n := by
  /-
    x : ENNReal
    hx : Ne x 0
    h'x : Ne x Top.top
    m n : Int
    ⊢ Eq (HPow.hPow x (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow x m) (HPow.hPow x n))
  -/
  lift x to ℝ≥0 using h'x
  /-
    case intro
    m n : Int
    x : NNReal
    hx : Ne (↑x) 0
    ⊢ Eq (HPow.hPow (↑x) (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow (↑x) m) (HPow.hPow …
  -/
  replace hx : x ≠ 0 := by simpa only [Ne, coe_eq_zero] using hx
  /-
    case intro
    m n : Int
    x : NNReal
    hx : Ne x 0
    ⊢ Eq (HPow.hPow (↑x) (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow (↑x) m) (HPow.hPow …
  -/
  simp only [← coe_zpow hx, zpow_add₀ hx, coe_mul]
  /-
    🎉 no goals
  -/


protected theorem zpow_neg {x : ℝ≥0∞} (x_ne_zero : x ≠ 0) (x_ne_top : x ≠ ⊤) (m : ℤ) :
    x ^ (-m) = (x ^ m)⁻¹ :=
                                        /-
                                          x : ENNReal
                                          x_ne_zero : Ne x 0
                                          x_ne_top : Ne x Top.top
                                          m : Int
                                          ⊢ Eq (HMul.hMul (HPow.hPow x (Neg.neg m)) (HPow.hPow x m)) 1
                                        -/
  ENNReal.eq_inv_of_mul_eq_one_left (by simp [← ENNReal.zpow_add x_ne_zero x_ne_top])
                                        /-
                                          🎉 no goals
                                        -/


protected theorem zpow_sub {x : ℝ≥0∞} (x_ne_zero : x ≠ 0) (x_ne_top : x ≠ ⊤) (m n : ℤ) :
    x ^ (m - n) = (x ^ m) * (x ^ n)⁻¹ := by
  /-
    x : ENNReal
    x_ne_zero : Ne x 0
    x_ne_top : Ne x Top.top
    m n : Int
    ⊢ Eq (HPow.hPow x (HSub.hSub m n)) (HMul.hMul (HPow.hPow x m) (Inv.inv (HPow.h …
  -/
  rw [sub_eq_add_neg, ENNReal.zpow_add x_ne_zero x_ne_top, ENNReal.zpow_neg x_ne_zero x_ne_top n]
  /-
    🎉 no goals
  -/


@[simp] lemma iSup_eq_zero : ⨆ i, f i = 0 ↔ ∀ i, f i = 0 := iSup_eq_bot


                                                        /-
                                                          ι : Sort u_1
                                                          ⊢ Eq (iSup fun x => 0) 0
                                                        -/
@[simp] lemma iSup_zero : ⨆ _ : ι, (0 : ℝ≥0∞) = 0 := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[deprecated (since := "2024-10-22")]
alias iSup_zero_eq_zero := iSup_zero


lemma iSup_natCast : ⨆ n : ℕ, (n : ℝ≥0∞) = ∞ :=
  (iSup_eq_top _).2 fun _b hb => ENNReal.exists_nat_gt (lt_top_iff_ne_top.1 hb)


@[simp] lemma iSup_lt_eq_self (a : ℝ≥0∞) : ⨆ b, ⨆ _ : b < a, b = a := by
  /-
    a : ENNReal
    ⊢ Eq (iSup fun b => iSup fun x => b) a
  -/
  refine le_antisymm (iSup₂_le fun b hb ↦ hb.le) ?_
  /-
    a : ENNReal
    ⊢ LE.le a (iSup fun b => iSup fun x => b)
  -/
  refine le_of_forall_lt fun c hca ↦ ?_
  /-
    a c : ENNReal
    hca : LT.lt c a
    ⊢ LT.lt c (iSup fun b => iSup fun x => b)
  -/
  obtain ⟨d, hcd, hdb⟩ := exists_between hca
  /-
    case intro.intro
    a c : ENNReal
    hca : LT.lt c a
    d : ENNReal
    hcd : LT.lt c d
    hdb : LT.lt d a
    ⊢ LT.lt c (iSup fun b => iSup fun x => b)
  -/
  exact hcd.trans_le <| le_iSup₂_of_le d hdb le_rfl
  /-
    🎉 no goals
  -/


lemma isUnit_iff : IsUnit a ↔ a ≠ 0 ∧ a ≠ ∞ := by
  refine ⟨fun ha ↦ ⟨ha.ne_zero, ?_⟩,
    fun ha ↦ ⟨⟨a, a⁻¹, ENNReal.mul_inv_cancel ha.1 ha.2, ENNReal.inv_mul_cancel ha.1 ha.2⟩, rfl⟩⟩
  /-
    a : ENNReal
    ha : IsUnit a
    ⊢ Ne a Top.top
  -/
  obtain ⟨u, rfl⟩ := ha
  /-
    case intro
    u : Units ENNReal
    ⊢ Ne (↑u) Top.top
  -/
  rintro hu
  /-
    case intro
    u : Units ENNReal
    hu : Eq (↑u) Top.top
    ⊢ False
  -/
  have := congr($hu * u⁻¹)
  /-
    case intro
    u : Units ENNReal
    hu : Eq (↑u) Top.top
    this : Eq (HMul.hMul ↑u ↑(Inv.inv u)) (HMul.hMul Top.top ↑(Inv.inv u))
    ⊢ False
  -/
  norm_cast at this
  /-
    case intro
    u : Units ENNReal
    hu : Eq (↑u) Top.top
    this : Eq (↑(HMul.hMul u (Inv.inv u))) (HMul.hMul Top.top ↑(Inv.inv u))
    ⊢ False
  -/
  simp [mul_inv_cancel] at this
  /-
    🎉 no goals
  -/


/-- Left multiplication by a nonzero finite `a` as an order isomorphism. -/
@[simps! toEquiv apply symm_apply]
def mulLeftOrderIso (a  : ℝ≥0∞) (ha : IsUnit a) : ℝ≥0∞ ≃o ℝ≥0∞ where
  toEquiv := ha.unit.mulLeft
                     /-
                       a✝¹ b c d : ENNReal
                       r p q : NNReal
                       ι : Sort u_1
                       κ : Sort u_2
                       f g : ι → ENNReal
                       s : Set ENNReal
                       a✝ a : ENNReal
                       ha : IsUnit a
                       ⊢ ∀ {a_1 b : ENNReal}, Iff (LE.le (ha.unit.mulLeft a_1) (ha.unit.mulLeft b)) ( …
                     -/
  map_rel_iff' := by simp [ENNReal.mul_le_mul_left, ha.ne_zero, (isUnit_iff.1 ha).2]
                     /-
                       🎉 no goals
                     -/


/-- Right multiplication by a nonzero finite `a` as an order isomorphism. -/
@[simps! toEquiv apply symm_apply]
def mulRightOrderIso (a  : ℝ≥0∞) (ha : IsUnit a) : ℝ≥0∞ ≃o ℝ≥0∞ where
  toEquiv := ha.unit.mulRight
                     /-
                       a✝¹ b c d : ENNReal
                       r p q : NNReal
                       ι : Sort u_1
                       κ : Sort u_2
                       f g : ι → ENNReal
                       s : Set ENNReal
                       a✝ a : ENNReal
                       ha : IsUnit a
                       ⊢ ∀ {a_1 b : ENNReal}, Iff (LE.le (ha.unit.mulRight a_1) (ha.unit.mulRight b)) …
                     -/
  map_rel_iff' := by simp [ENNReal.mul_le_mul_right, ha.ne_zero, (isUnit_iff.1 ha).2]
                     /-
                       🎉 no goals
                     -/


lemma mul_iSup (a : ℝ≥0∞) (f : ι → ℝ≥0∞) : a * ⨆ i, f i = ⨆ i, a * f i := by
  /-
    ι : Sort u_1
    a : ENNReal
    f : ι → ENNReal
    ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
  -/
  by_cases hf : ∀ i, f i = 0
    /-
      case pos
      ι : Sort u_1
      a : ENNReal
      f : ι → ENNReal
      hf : ∀ (i : ι), Eq (f i) 0
      ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
    -/
  · simp [hf]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Sort u_1
    a : ENNReal
    f : ι → ENNReal
    hf : Not (∀ (i : ι), Eq (f i) 0)
    ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
  -/
  obtain rfl | ha₀ := eq_or_ne a 0
    /-
      case neg.inl
      ι : Sort u_1
      f : ι → ENNReal
      hf : Not (∀ (i : ι), Eq (f i) 0)
      ⊢ Eq (HMul.hMul 0 (iSup fun i => f i)) (iSup fun i => HMul.hMul 0 (f i))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case neg.inr
    ι : Sort u_1
    a : ENNReal
    f : ι → ENNReal
    hf : Not (∀ (i : ι), Eq (f i) 0)
    ha₀ : Ne a 0
    ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
  -/
  obtain rfl | ha := eq_or_ne a ∞
    /-
      case neg.inr.inl
      ι : Sort u_1
      f : ι → ENNReal
      hf : Not (∀ (i : ι), Eq (f i) 0)
      ha₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul Top.top (iSup fun i => f i)) (iSup fun i => HMul.hMul Top.top  …
    -/
  · obtain ⟨i, hi⟩ := not_forall.1 hf
    simpa [iSup_eq_zero.not.2 hf, eq_comm (a := ⊤)]
      using le_iSup_of_le (f := fun i => ⊤ * f i) i (top_mul hi).ge
    /-
      case neg.inr.inr
      ι : Sort u_1
      a : ENNReal
      f : ι → ENNReal
      hf : Not (∀ (i : ι), Eq (f i) 0)
      ha₀ : Ne a 0
      ha : Ne a Top.top
      ⊢ Eq (HMul.hMul a (iSup fun i => f i)) (iSup fun i => HMul.hMul a (f i))
    -/
  · exact (mulLeftOrderIso _ <| isUnit_iff.2 ⟨ha₀, ha⟩).map_iSup _
    /-
      🎉 no goals
    -/


lemma iSup_mul (f : ι → ℝ≥0∞) (a : ℝ≥0∞) : (⨆ i, f i) * a = ⨆ i, f i * a := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    ⊢ Eq (HMul.hMul (iSup fun i => f i) a) (iSup fun i => HMul.hMul (f i) a)
  -/
  simp [mul_comm, mul_iSup]
  /-
    🎉 no goals
  -/


lemma mul_sSup {a : ℝ≥0∞} : a * sSup s = ⨆ b ∈ s, a * b := by
  /-
    s : Set ENNReal
    a : ENNReal
    ⊢ Eq (HMul.hMul a (SupSet.sSup s)) (iSup fun b => iSup fun h => HMul.hMul a b)
  -/
  simp only [sSup_eq_iSup, mul_iSup]
  /-
    🎉 no goals
  -/


lemma sSup_mul {a : ℝ≥0∞} : sSup s * a = ⨆ b ∈ s, b * a := by
  /-
    s : Set ENNReal
    a : ENNReal
    ⊢ Eq (HMul.hMul (SupSet.sSup s) a) (iSup fun b => iSup fun h => HMul.hMul b a)
  -/
  simp only [sSup_eq_iSup, iSup_mul]
  /-
    🎉 no goals
  -/


lemma iSup_div (f : ι → ℝ≥0∞) (a : ℝ≥0∞) : iSup f / a = ⨆ i, f i / a := iSup_mul ..

lemma sSup_div (s : Set ℝ≥0∞) (a : ℝ≥0∞) : sSup s / a = ⨆ b ∈ s, b / a := sSup_mul ..


/-- Very general version for distributivity of multiplication over an infimum.

See `ENNReal.mul_iInf_of_ne` for the special case assuming `a ≠ 0` and `a ≠ ∞`, and
`ENNReal.mul_iInf` for the special case assuming `Nonempty ι`. -/
lemma mul_iInf' (hinfty : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) (h₀ : a = 0 → Nonempty ι) :
    a * ⨅ i, f i = ⨅ i, a * f i := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    hinfty : Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
    h₀ : Eq a 0 → Nonempty ι
    ⊢ Eq (HMul.hMul a (iInf fun i => f i)) (iInf fun i => HMul.hMul a (f i))
  -/
  obtain rfl | ha₀ := eq_or_ne a 0
    /-
      case inl
      ι : Sort u_1
      f : ι → ENNReal
      hinfty : Eq 0 Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
      h₀ : Eq 0 0 → Nonempty ι
      ⊢ Eq (HMul.hMul 0 (iInf fun i => f i)) (iInf fun i => HMul.hMul 0 (f i))
    -/
  · simp [h₀ rfl]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    hinfty : Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
    h₀ : Eq a 0 → Nonempty ι
    ha₀ : Ne a 0
    ⊢ Eq (HMul.hMul a (iInf fun i => f i)) (iInf fun i => HMul.hMul a (f i))
  -/
  obtain rfl | ha := eq_or_ne a ∞
    /-
      case inr.inl
      ι : Sort u_1
      f : ι → ENNReal
      hinfty : Eq Top.top Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f …
      h₀ : Eq Top.top 0 → Nonempty ι
      ha₀ : Ne Top.top 0
      ⊢ Eq (HMul.hMul Top.top (iInf fun i => f i)) (iInf fun i => HMul.hMul Top.top  …
    -/
  · obtain ⟨i, hi⟩ | hf := em (∃ i, f i = 0)
      /-
        case inr.inl.inl.intro
        ι : Sort u_1
        f : ι → ENNReal
        hinfty : Eq Top.top Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f …
        h₀ : Eq Top.top 0 → Nonempty ι
        ha₀ : Ne Top.top 0
        i : ι
        hi : Eq (f i) 0
        ⊢ Eq (HMul.hMul Top.top (iInf fun i => f i)) (iInf fun i => HMul.hMul Top.top  …
      -/
    · rw [(iInf_eq_bot _).2, (iInf_eq_bot _).2, bot_eq_zero, mul_zero] <;>
        /-
          case inr.inl.inl.intro
          ι : Sort u_1
          f : ι → ENNReal
          hinfty : Eq Top.top Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f …
          h₀ : Eq Top.top 0 → Nonempty ι
          ha₀ : Ne Top.top 0
          i : ι
          hi : Eq (f i) 0
          ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => LT.lt (HMul.hMul Top.top  …
        -/
        /-
          🎉 no goals
        -/
        exact fun _ _↦ ⟨i, by simpa [hi]⟩
        /-
          🎉 no goals
        -/
      /-
        case inr.inl.inr
        ι : Sort u_1
        f : ι → ENNReal
        hinfty : Eq Top.top Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f …
        h₀ : Eq Top.top 0 → Nonempty ι
        ha₀ : Ne Top.top 0
        hf : Not (Exists fun i => Eq (f i) 0)
        ⊢ Eq (HMul.hMul Top.top (iInf fun i => f i)) (iInf fun i => HMul.hMul Top.top  …
      -/
    · rw [top_mul (mt (hinfty rfl) hf), eq_comm, iInf_eq_top]
      /-
        case inr.inl.inr
        ι : Sort u_1
        f : ι → ENNReal
        hinfty : Eq Top.top Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f …
        h₀ : Eq Top.top 0 → Nonempty ι
        ha₀ : Ne Top.top 0
        hf : Not (Exists fun i => Eq (f i) 0)
        ⊢ ∀ (i : ι), Eq (HMul.hMul Top.top (f i)) Top.top
      -/
      exact fun i ↦ top_mul fun hi ↦ hf ⟨i, hi⟩
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      ι : Sort u_1
      f : ι → ENNReal
      a : ENNReal
      hinfty : Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
      h₀ : Eq a 0 → Nonempty ι
      ha₀ : Ne a 0
      ha : Ne a Top.top
      ⊢ Eq (HMul.hMul a (iInf fun i => f i)) (iInf fun i => HMul.hMul a (f i))
    -/
  · exact (mulLeftOrderIso _ <| isUnit_iff.2 ⟨ha₀, ha⟩).map_iInf _
    /-
      🎉 no goals
    -/


/-- Very general version for distributivity of multiplication over an infimum.

See `ENNReal.iInf_mul_of_ne` for the special case assuming `a ≠ 0` and `a ≠ ∞`, and
`ENNReal.iInf_mul` for the special case assuming `Nonempty ι`. -/
lemma iInf_mul' (hinfty : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) (h₀ : a = 0 → Nonempty ι) :
                                        /-
                                          ι : Sort u_1
                                          f : ι → ENNReal
                                          a : ENNReal
                                          hinfty : Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                                          h₀ : Eq a 0 → Nonempty ι
                                          ⊢ Eq (HMul.hMul (iInf fun i => f i) a) (iInf fun i => HMul.hMul (f i) a)
                                        -/
    (⨅ i, f i) * a = ⨅ i, f i * a := by simpa only [mul_comm a] using mul_iInf' hinfty h₀
                                        /-
                                          🎉 no goals
                                        -/


/-- If `a ≠ 0` and `a ≠ ∞`, then right multiplication by `a` maps infimum to infimum.

See `ENNReal.mul_iInf'` for the general case, and `ENNReal.iInf_mul` for another special case that
assumes `Nonempty ι` but does not require `a ≠ 0`, and `ENNReal`. -/
lemma mul_iInf_of_ne (ha₀ : a ≠ 0) (ha : a ≠ ∞) : a * ⨅ i, f i = ⨅ i, a * f i :=
                /-
                  ι : Sort u_1
                  f : ι → ENNReal
                  a : ENNReal
                  ha₀ : Ne a 0
                  ha : Ne a Top.top
                  ⊢ Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                -/
                /-
                  🎉 no goals
                -/
  mul_iInf' (by simp [ha]) (by simp [ha₀])
                               /-
                                 🎉 no goals
                               -/


/-- If `a ≠ 0` and `a ≠ ∞`, then right multiplication by `a` maps infimum to infimum.

See `ENNReal.iInf_mul'` for the general case, and `ENNReal.iInf_mul` for another special case that
assumes `Nonempty ι` but does not require `a ≠ 0`. -/
lemma iInf_mul_of_ne (ha₀ : a ≠ 0) (ha : a ≠ ∞) : (⨅ i, f i) * a = ⨅ i, f i * a :=
                /-
                  ι : Sort u_1
                  f : ι → ENNReal
                  a : ENNReal
                  ha₀ : Ne a 0
                  ha : Ne a Top.top
                  ⊢ Eq a Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                -/
                /-
                  🎉 no goals
                -/
  iInf_mul' (by simp [ha]) (by simp [ha₀])
                               /-
                                 🎉 no goals
                               -/


/-- See `ENNReal.mul_iInf'` for the general case, and `ENNReal.mul_iInf_of_ne` for another special
case that assumes `a ≠ 0` but does not require `Nonempty ι`. -/
lemma mul_iInf [Nonempty ι] (hinfty : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) :
    a * ⨅ i, f i = ⨅ i, a * f i := mul_iInf' hinfty fun _ ↦ ‹Nonempty ι›


/-- See `ENNReal.iInf_mul'` for the general case, and `ENNReal.iInf_mul_of_ne` for another special
case that assumes `a ≠ 0` but does not require `Nonempty ι`. -/
lemma iInf_mul [Nonempty ι] (hinfty : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) :
    (⨅ i, f i) * a = ⨅ i, f i * a := iInf_mul' hinfty fun _ ↦ ‹Nonempty ι›


/-- Very general version for distributivity of division over an infimum.

See `ENNReal.iInf_div_of_ne` for the special case assuming `a ≠ 0` and `a ≠ ∞`, and
`ENNReal.iInf_div` for the special case assuming `Nonempty ι`. -/
lemma iInf_div' (hinfty : a = 0 → ⨅ i, f i = 0 → ∃ i, f i = 0) (h₀ : a = ∞ → Nonempty ι) :
                                                   /-
                                                     ι : Sort u_1
                                                     f : ι → ENNReal
                                                     a : ENNReal
                                                     hinfty : Eq a 0 → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                                                     h₀ : Eq a Top.top → Nonempty ι
                                                     ⊢ Eq (Inv.inv a) Top.top → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    (⨅ i, f i) / a = ⨅ i, f i / a := iInf_mul' (by simpa) (by simpa)
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If `a ≠ 0` and `a ≠ ∞`, then division by `a` maps infimum to infimum.

See `ENNReal.iInf_div'` for the general case, and `ENNReal.iInf_div` for another special case that
assumes `Nonempty ι` but does not require `a ≠ ∞`. -/
lemma iInf_div_of_ne (ha₀ : a ≠ 0) (ha : a ≠ ∞) : (⨅ i, f i) / a = ⨅ i, f i / a :=
                /-
                  ι : Sort u_1
                  f : ι → ENNReal
                  a : ENNReal
                  ha₀ : Ne a 0
                  ha : Ne a Top.top
                  ⊢ Eq a 0 → Eq (iInf fun i => f i) 0 → Exists fun i => Eq (f i) 0
                -/
                /-
                  🎉 no goals
                -/
  iInf_div' (by simp [ha₀]) (by simp [ha])
                                /-
                                  🎉 no goals
                                -/


/-- See `ENNReal.iInf_div'` for the general case, and `ENNReal.iInf_div_of_ne` for another special
case that assumes `a ≠ ∞` but does not require `Nonempty ι`. -/
lemma iInf_div [Nonempty ι] (hinfty : a = 0 → ⨅ i, f i = 0 → ∃ i, f i = 0) :
    (⨅ i, f i) / a = ⨅ i, f i / a := iInf_div' hinfty fun _ ↦ ‹Nonempty ι›


lemma inv_iInf (f : ι → ℝ≥0∞) : (⨅ i, f i)⁻¹ = ⨆ i, (f i)⁻¹ := OrderIso.invENNReal.map_iInf _

lemma inv_iSup (f : ι → ℝ≥0∞) : (⨆ i, f i)⁻¹ = ⨅ i, (f i)⁻¹ := OrderIso.invENNReal.map_iSup _


                                                                /-
                                                                  s : Set ENNReal
                                                                  ⊢ Eq (Inv.inv (InfSet.sInf s)) (iSup fun a => iSup fun h => Inv.inv a)
                                                                -/
lemma inv_sInf (s : Set ℝ≥0∞) : (sInf s)⁻¹ = ⨆ a ∈ s, a⁻¹ := by simp [sInf_eq_iInf, inv_iInf]
                                                                /-
                                                                  🎉 no goals
                                                                -/

                                                                /-
                                                                  s : Set ENNReal
                                                                  ⊢ Eq (Inv.inv (SupSet.sSup s)) (iInf fun a => iInf fun h => Inv.inv a)
                                                                -/
lemma inv_sSup (s : Set ℝ≥0∞) : (sSup s)⁻¹ = ⨅ a ∈ s, a⁻¹ := by simp [sSup_eq_iSup, inv_iSup]
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma le_iInf_mul {ι : Type*} (u v : ι → ℝ≥0∞) :
    (⨅ i, u i) * ⨅ i, v i ≤ ⨅ i, u i * v i :=
  le_iInf fun i ↦ mul_le_mul' (iInf_le u i) (iInf_le v i)


lemma iSup_mul_le {ι : Type*} {u v : ι → ℝ≥0∞} :
    ⨆ i, u i * v i ≤ (⨆ i, u i) * ⨆ i, v i :=
  iSup_le fun i ↦ mul_le_mul' (le_iSup u i) (le_iSup v i)


lemma add_iSup [Nonempty ι] (f : ι → ℝ≥0∞) : a + ⨆ i, f i = ⨆ i, a + f i := by
  /-
    ι : Sort u_1
    a : ENNReal
    inst✝ : Nonempty ι
    f : ι → ENNReal
    ⊢ Eq (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  obtain rfl | ha := eq_or_ne a ∞
    /-
      case inl
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → ENNReal
      ⊢ Eq (HAdd.hAdd Top.top (iSup fun i => f i)) (iSup fun i => HAdd.hAdd Top.top  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_1
    a : ENNReal
    inst✝ : Nonempty ι
    f : ι → ENNReal
    ha : Ne a Top.top
    ⊢ Eq (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  refine le_antisymm ?_ <| iSup_le fun i ↦ add_le_add_left (le_iSup ..) _
  /-
    case inr
    ι : Sort u_1
    a : ENNReal
    inst✝ : Nonempty ι
    f : ι → ENNReal
    ha : Ne a Top.top
    ⊢ LE.le (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  refine add_le_of_le_tsub_left_of_le (le_iSup_of_le (Classical.arbitrary _) le_self_add) ?_
  /-
    case inr
    ι : Sort u_1
    a : ENNReal
    inst✝ : Nonempty ι
    f : ι → ENNReal
    ha : Ne a Top.top
    ⊢ LE.le (iSup fun i => f i) (HSub.hSub (iSup fun i => HAdd.hAdd a (f i)) a)
  -/
  exact iSup_le fun i ↦ ENNReal.le_sub_of_add_le_left ha <| le_iSup (a + f ·) i
  /-
    🎉 no goals
  -/


lemma iSup_add [Nonempty ι] (f : ι → ℝ≥0∞) : (⨆ i, f i) + a = ⨆ i, f i + a := by
  /-
    ι : Sort u_1
    a : ENNReal
    inst✝ : Nonempty ι
    f : ι → ENNReal
    ⊢ Eq (HAdd.hAdd (iSup fun i => f i) a) (iSup fun i => HAdd.hAdd (f i) a)
  -/
  simp [add_comm, add_iSup]
  /-
    🎉 no goals
  -/


lemma add_biSup' {p : ι → Prop} (h : ∃ i, p i) (f : ι → ℝ≥0∞) :
    a + ⨆ i, ⨆ _ : p i, f i = ⨆ i, ⨆ _ : p i, a + f i := by
  /-
    ι : Sort u_1
    a : ENNReal
    p : ι → Prop
    h : Exists fun i => p i
    f : ι → ENNReal
    ⊢ Eq (HAdd.hAdd a (iSup fun i => iSup fun x => f i)) (iSup fun i => iSup fun x …
  -/
  haveI : Nonempty {i // p i} := nonempty_subtype.2 h
  /-
    ι : Sort u_1
    a : ENNReal
    p : ι → Prop
    h : Exists fun i => p i
    f : ι → ENNReal
    this : Nonempty (Subtype fun i => p i)
    ⊢ Eq (HAdd.hAdd a (iSup fun i => iSup fun x => f i)) (iSup fun i => iSup fun x …
  -/
  simp only [iSup_subtype', add_iSup]
  /-
    🎉 no goals
  -/


lemma biSup_add' {p : ι → Prop} (h : ∃ i, p i) (f : ι → ℝ≥0∞) :
                                                              /-
                                                                ι : Sort u_1
                                                                a : ENNReal
                                                                p : ι → Prop
                                                                h : Exists fun i => p i
                                                                f : ι → ENNReal
                                                                ⊢ Eq (HAdd.hAdd (iSup fun i => iSup fun x => f i) a) (iSup fun i => iSup fun x …
                                                              -/
    (⨆ i, ⨆ _ : p i, f i) + a = ⨆ i, ⨆ _ : p i, f i + a := by simp only [add_comm, add_biSup' h]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma add_biSup {ι : Type*} {s : Set ι} (hs : s.Nonempty) (f : ι → ℝ≥0∞) :
    a + ⨆ i ∈ s, f i = ⨆ i ∈ s, a + f i := add_biSup' hs _


lemma biSup_add {ι : Type*} {s : Set ι} (hs : s.Nonempty) (f : ι → ℝ≥0∞) :
    (⨆ i ∈ s, f i) + a = ⨆ i ∈ s, f i + a := biSup_add' hs _


lemma add_sSup (hs : s.Nonempty) : a + sSup s = ⨆ b ∈ s, a + b := by
  /-
    s : Set ENNReal
    a : ENNReal
    hs : s.Nonempty
    ⊢ Eq (HAdd.hAdd a (SupSet.sSup s)) (iSup fun b => iSup fun h => HAdd.hAdd a b)
  -/
  rw [sSup_eq_iSup, add_biSup hs]
  /-
    🎉 no goals
  -/


lemma sSup_add (hs : s.Nonempty) : sSup s + a = ⨆ b ∈ s, b + a := by
  /-
    s : Set ENNReal
    a : ENNReal
    hs : s.Nonempty
    ⊢ Eq (HAdd.hAdd (SupSet.sSup s) a) (iSup fun b => iSup fun h => HAdd.hAdd b a)
  -/
  rw [sSup_eq_iSup, biSup_add hs]
  /-
    🎉 no goals
  -/


lemma iSup_add_iSup_le [Nonempty ι] [Nonempty κ] {g : κ → ℝ≥0∞} (h : ∀ i j, f i + g j ≤ a) :
                              /-
                                ι : Sort u_1
                                κ : Sort u_2
                                f : ι → ENNReal
                                a : ENNReal
                                inst✝¹ : Nonempty ι
                                inst✝ : Nonempty κ
                                g : κ → ENNReal
                                h : ∀ (i : ι) (j : κ), LE.le (HAdd.hAdd (f i) (g j)) a
                                ⊢ LE.le (HAdd.hAdd (iSup f) (iSup g)) a
                              -/
    iSup f + iSup g ≤ a := by simp_rw [iSup_add, add_iSup]; exact iSup₂_le h
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma biSup_add_biSup_le' {p : ι → Prop} {q : κ → Prop} (hp : ∃ i, p i) (hq : ∃ j, q j)
    {g : κ → ℝ≥0∞} (h : ∀ i, p i → ∀ j, q j → f i + g j ≤ a) :
    (⨆ i, ⨆ _ : p i, f i) + ⨆ j, ⨆ _ : q j, g j ≤ a := by
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    p : ι → Prop
    q : κ → Prop
    hp : Exists fun i => p i
    hq : Exists fun j => q j
    g : κ → ENNReal
    h : ∀ (i : ι), p i → ∀ (j : κ), q j → LE.le (HAdd.hAdd (f i) (g j)) a
    ⊢ LE.le (HAdd.hAdd (iSup fun i => iSup fun x => f i) (iSup fun j => iSup fun x …
  -/
  simp_rw [biSup_add' hp, add_biSup' hq]
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    p : ι → Prop
    q : κ → Prop
    hp : Exists fun i => p i
    hq : Exists fun j => q j
    g : κ → ENNReal
    h : ∀ (i : ι), p i → ∀ (j : κ), q j → LE.le (HAdd.hAdd (f i) (g j)) a
    ⊢ LE.le (iSup fun i => iSup fun x => iSup fun i_1 => iSup fun x => HAdd.hAdd ( …
  -/
  exact iSup₂_le fun i hi => iSup₂_le (h i hi)
  /-
    🎉 no goals
  -/


lemma biSup_add_biSup_le {ι κ : Type*} {s : Set ι} {t : Set κ} (hs : s.Nonempty) (ht : t.Nonempty)
    {f : ι → ℝ≥0∞} {g : κ → ℝ≥0∞} {a : ℝ≥0∞} (h : ∀ i ∈ s, ∀ j ∈ t, f i + g j ≤ a) :
    (⨆ i ∈ s, f i) + ⨆ j ∈ t, g j ≤ a := biSup_add_biSup_le' hs ht h


lemma iSup_add_iSup (h : ∀ i j, ∃ k, f i + g j ≤ f k + g k) : iSup f + iSup g = ⨆ i, f i + g i := by
  /-
    ι : Sort u_1
    f g : ι → ENNReal
    h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
    ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      f g : ι → ENNReal
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : IsEmpty ι
      ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
  · simp only [iSup_of_empty, bot_eq_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      f g : ι → ENNReal
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
  · refine le_antisymm ?_ (iSup_le fun a => add_le_add (le_iSup _ _) (le_iSup _ _))
    /-
      case inr
      ι : Sort u_1
      f g : ι → ENNReal
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      ⊢ LE.le (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    refine iSup_add_iSup_le fun i j => ?_
    /-
      case inr
      ι : Sort u_1
      f g : ι → ENNReal
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      i j : ι
      ⊢ LE.le (HAdd.hAdd (f i) (g j)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    rcases h i j with ⟨k, hk⟩
    /-
      case inr.intro
      ι : Sort u_1
      f g : ι → ENNReal
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      i j k : ι
      hk : LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k) (g k))
      ⊢ LE.le (HAdd.hAdd (f i) (g j)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    exact le_iSup_of_le k hk
    /-
      🎉 no goals
    -/


lemma iSup_add_iSup_of_monotone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {f g : ι → ℝ≥0∞}
    (hf : Monotone f) (hg : Monotone g) : iSup f + iSup g = ⨆ a, f a + g a :=
                                                                      /-
                                                                        ι : Type u_3
                                                                        inst✝¹ : Preorder ι
                                                                        inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                                        f g : ι → ENNReal
                                                                        hf : Monotone f
                                                                        hg : Monotone g
                                                                        i j _k : ι
                                                                        x✝ : And (LE.le i _k) (LE.le j _k)
                                                                        hi : LE.le i _k
                                                                        hj : LE.le j _k
                                                                        ⊢ LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f _k) (g _k))
                                                                      -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  iSup_add_iSup fun i j ↦ (exists_ge_ge i j).imp fun _k ⟨hi, hj⟩ ↦ by gcongr <;> apply_rules
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma finsetSum_iSup {α ι : Type*} {s : Finset α} {f : α → ι → ℝ≥0∞}
    (hf : ∀ i j, ∃ k, ∀ a, f a i ≤ f a k ∧ f a j ≤ f a k) :
    ∑ a ∈ s, ⨆ i, f a i = ⨆ i, ∑ a ∈ s, f a i := by
  /-
    α : Type u_3
    ι : Type u_4
    s : Finset α
    f : α → ι → ENNReal
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    ⊢ Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f a i)
  -/
  induction' s using Finset.cons_induction with a s ha ihs
    /-
      case empty
      α : Type u_3
      ι : Type u_4
      f : α → ι → ENNReal
      hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
      ⊢ Eq (EmptyCollection.emptyCollection.sum fun a => iSup fun i => f a i) (iSup  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_3
    ι : Type u_4
    f : α → ι → ENNReal
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    ⊢ Eq ((Finset.cons a s ha).sum fun a => iSup fun i => f a i) (iSup fun i => (F …
  -/
  simp_rw [Finset.sum_cons, ihs]
  /-
    case cons
    α : Type u_3
    ι : Type u_4
    f : α → ι → ENNReal
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    ⊢ Eq (HAdd.hAdd (iSup fun i => f a i) (iSup fun i => s.sum fun a => f a i)) (i …
  -/
  refine iSup_add_iSup fun i j ↦ (hf i j).imp fun k hk ↦ ?_
  /-
    case cons
    α : Type u_3
    ι : Type u_4
    f : α → ι → ENNReal
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    i j k : ι
    hk : ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.le (f a j) (f a k))
    ⊢ LE.le (HAdd.hAdd (f a i) (s.sum fun a => f a j)) (HAdd.hAdd (f a k) (s.sum f …
  -/
  gcongr
  /-
    case cons.h₁
    α : Type u_3
    ι : Type u_4
    f : α → ι → ENNReal
    hf : ∀ (i j : ι), Exists fun k => ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.l …
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ihs : Eq (s.sum fun a => iSup fun i => f a i) (iSup fun i => s.sum fun a => f  …
    i j k : ι
    hk : ∀ (a : α), And (LE.le (f a i) (f a k)) (LE.le (f a j) (f a k))
    ⊢ LE.le (f a i) (f a k)
  -/
  exacts [(hk a).1, (hk _).2]
  /-
    🎉 no goals
  -/


lemma finsetSum_iSup_of_monotone {α ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {s : Finset α}
    {f : α → ι → ℝ≥0∞} (hf : ∀ a, Monotone (f a)) : (∑ a ∈ s, iSup (f a)) = ⨆ n, ∑ a ∈ s, f a n :=
  finsetSum_iSup fun i j ↦ (exists_ge_ge i j).imp fun _k ⟨hi, hj⟩ a ↦ ⟨hf a hi, hf a hj⟩


@[deprecated (since := "2024-07-14")]
alias finset_sum_iSup_nat := finsetSum_iSup_of_monotone


lemma le_iInf_mul_iInf {g : κ → ℝ≥0∞} (hf : ∃ i, f i ≠ ∞) (hg : ∃ j, g j ≠ ∞)
    (ha : ∀ i j, a ≤ f i * g j) : a ≤ (⨅ i, f i) * ⨅ j, g j := by
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    ⊢ LE.le a (HMul.hMul (iInf fun i => f i) (iInf fun j => g j))
  -/
  rw [← iInf_ne_top_subtype]
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    ⊢ LE.le a (HMul.hMul (iInf fun i => f ↑i) (iInf fun j => g j))
  -/
  have := nonempty_subtype.2 hf
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    this : Nonempty (Subtype fun a => Ne (f a) Top.top)
    ⊢ LE.le a (HMul.hMul (iInf fun i => f ↑i) (iInf fun j => g j))
  -/
  have := hg.nonempty
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    this✝ : Nonempty (Subtype fun a => Ne (f a) Top.top)
    this : Nonempty κ
    ⊢ LE.le a (HMul.hMul (iInf fun i => f ↑i) (iInf fun j => g j))
  -/
  replace hg : ⨅ j, g j ≠ ∞ := by simpa using hg
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    this✝ : Nonempty (Subtype fun a => Ne (f a) Top.top)
    this : Nonempty κ
    hg : Ne (iInf fun j => g j) Top.top
    ⊢ LE.le a (HMul.hMul (iInf fun i => f ↑i) (iInf fun j => g j))
  -/
  rw [iInf_mul fun h ↦ (hg h).elim, le_iInf_iff]
  /-
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    this✝ : Nonempty (Subtype fun a => Ne (f a) Top.top)
    this : Nonempty κ
    hg : Ne (iInf fun j => g j) Top.top
    ⊢ ∀ (i : Subtype fun i => Ne (f i) Top.top), LE.le a (HMul.hMul (f ↑i) (iInf f …
  -/
  rintro ⟨i, hi⟩
  /-
    case mk
    ι : Sort u_1
    κ : Sort u_2
    f : ι → ENNReal
    a : ENNReal
    g : κ → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    ha : ∀ (i : ι) (j : κ), LE.le a (HMul.hMul (f i) (g j))
    this✝ : Nonempty (Subtype fun a => Ne (f a) Top.top)
    this : Nonempty κ
    hg : Ne (iInf fun j => g j) Top.top
    i : ι
    hi : Ne (f i) Top.top
    ⊢ LE.le a (HMul.hMul (f ↑⟨i, hi⟩) (iInf fun j => g j))
  -/
  simpa [mul_iInf fun h ↦ (hi h).elim] using ha i
  /-
    🎉 no goals
  -/


lemma iInf_mul_iInf {f g : ι → ℝ≥0∞} (hf : ∃ i, f i ≠ ∞) (hg : ∃ j, g j ≠ ∞)
    (h : ∀ i j, ∃ k, f k * g k ≤ f i * g j) : (⨅ i, f i) * ⨅ i, g i = ⨅ i, f i * g i := by
  refine le_antisymm (le_iInf fun i ↦ mul_le_mul' (iInf_le ..) (iInf_le ..))
    (le_iInf_mul_iInf hf hg fun i j ↦ ?_)
  /-
    ι : Sort u_1
    f g : ι → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    h : ∀ (i j : ι), Exists fun k => LE.le (HMul.hMul (f k) (g k)) (HMul.hMul (f i …
    i j : ι
    ⊢ LE.le (iInf fun i => HMul.hMul (f i) (g i)) (HMul.hMul (f i) (g j))
  -/
  obtain ⟨k, hk⟩ := h i j
  /-
    case intro
    ι : Sort u_1
    f g : ι → ENNReal
    hf : Exists fun i => Ne (f i) Top.top
    hg : Exists fun j => Ne (g j) Top.top
    h : ∀ (i j : ι), Exists fun k => LE.le (HMul.hMul (f k) (g k)) (HMul.hMul (f i …
    i j k : ι
    hk : LE.le (HMul.hMul (f k) (g k)) (HMul.hMul (f i) (g j))
    ⊢ LE.le (iInf fun i => HMul.hMul (f i) (g i)) (HMul.hMul (f i) (g j))
  -/
  exact iInf_le_of_le k hk
  /-
    🎉 no goals
  -/


lemma smul_iSup {R} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (f : ι → ℝ≥0∞) (c : R) :
    c • ⨆ i, f i = ⨆ i, c • f i := by
  /-
    ι : Sort u_1
    R : Type u_3
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    f : ι → ENNReal
    c : R
    ⊢ Eq (HSMul.hSMul c (iSup fun i => f i)) (iSup fun i => HSMul.hSMul c (f i))
  -/
  simp only [← smul_one_mul c (f _), ← smul_one_mul c (iSup _), ENNReal.mul_iSup]
  /-
    🎉 no goals
  -/


lemma smul_sSup {R} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (s : Set ℝ≥0∞) (c : R) :
    c • sSup s = ⨆ a ∈ s, c • a := by
  /-
    R : Type u_3
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    s : Set ENNReal
    c : R
    ⊢ Eq (HSMul.hSMul c (SupSet.sSup s)) (iSup fun a => iSup fun h => HSMul.hSMul  …
  -/
  simp_rw [← smul_one_mul c (sSup s), ENNReal.mul_sSup, smul_one_mul]
  /-
    🎉 no goals
  -/


lemma sub_iSup [Nonempty ι] (ha : a ≠ ∞) : a - ⨆ i, f i = ⨅ i, a - f i := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
  -/
  obtain ⟨i, hi⟩ | h := em (∃ i, a < f i)
    /-
      case inl.intro
      ι : Sort u_1
      f : ι → ENNReal
      a : ENNReal
      inst✝ : Nonempty ι
      ha : Ne a Top.top
      i : ι
      hi : LT.lt a (f i)
      ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
    -/
  · rw [tsub_eq_zero_iff_le.2 <| le_iSup_of_le _ hi.le, (iInf_eq_bot _).2, bot_eq_zero]
    /-
      case inl.intro
      ι : Sort u_1
      f : ι → ENNReal
      a : ENNReal
      inst✝ : Nonempty ι
      ha : Ne a Top.top
      i : ι
      hi : LT.lt a (f i)
      ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => LT.lt (HSub.hSub a (f i)) b
    -/
    exact fun x hx ↦ ⟨i, by simpa [hi.le, tsub_eq_zero_of_le]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : Not (Exists fun i => LT.lt a (f i))
    ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
  -/
  simp_rw [not_exists, not_lt] at h
  refine le_antisymm (le_iInf fun i ↦ tsub_le_tsub_left (le_iSup ..) _) <|
    ENNReal.le_sub_of_add_le_left (ne_top_of_le_ne_top ha <| iSup_le h) <|
    add_le_of_le_tsub_right_of_le (iInf_le_of_le (Classical.arbitrary _) tsub_le_self) <|
    iSup_le fun i ↦ ?_
  /-
    case inr
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : ∀ (x : ι), LE.le (f x) a
    i : ι
    ⊢ LE.le (f i) (HSub.hSub a (iInf fun i => HSub.hSub a (f i)))
  -/
  rw [← sub_sub_cancel ha (h _)]
  /-
    case inr
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : ∀ (x : ι), LE.le (f x) a
    i : ι
    ⊢ LE.le (HSub.hSub a (HSub.hSub a (f i))) (HSub.hSub a (iInf fun i => HSub.hSu …
  -/
  exact tsub_le_tsub_left (iInf_le (a - f ·) i) _
  /-
    🎉 no goals
  -/

-- TODO: Prove the two one-side versions

lemma exists_lt_add_of_lt_add {x y z : ℝ≥0∞} (h : x < y + z) (hy : y ≠ 0) (hz : z ≠ 0) :
    ∃ y' < y, ∃ z' < z, x < y' + z' := by
  /-
    x y z : ENNReal
    h : LT.lt x (HAdd.hAdd y z)
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Exists fun y' => And (LT.lt y' y) (Exists fun z' => And (LT.lt z' z) (LT.lt  …
  -/
  contrapose! h
  /-
    x y z : ENNReal
    hy : Ne y 0
    hz : Ne z 0
    h : ∀ (y' : ENNReal), LT.lt y' y → ∀ (z' : ENNReal), LT.lt z' z → LE.le (HAdd. …
    ⊢ LE.le (HAdd.hAdd y z) x
  -/
  simpa using biSup_add_biSup_le' (by exact ⟨0, hy.bot_lt⟩) (by exact ⟨0, hz.bot_lt⟩) h
  /-
    🎉 no goals
  -/


