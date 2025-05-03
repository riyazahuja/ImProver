theorem cast_choose {a b : ℕ} (h : a ≤ b) : (b.choose a : K) = b ! / (a ! * (b - a)!) := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    inst✝ : CharZero K
    a b : Nat
    h : LE.le a b
    ⊢ Eq (↑(b.choose a)) (HDiv.hDiv (↑b.factorial) (HMul.hMul ↑a.factorial ↑(HSub. …
  -/
  have : ∀ {n : ℕ}, (n ! : K) ≠ 0 := Nat.cast_ne_zero.2 (factorial_ne_zero _)
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    inst✝ : CharZero K
    a b : Nat
    h : LE.le a b
    this : ∀ {n : Nat}, Ne (↑n.factorial) 0
    ⊢ Eq (↑(b.choose a)) (HDiv.hDiv (↑b.factorial) (HMul.hMul ↑a.factorial ↑(HSub. …
  -/
  rw [eq_div_iff_mul_eq (mul_ne_zero this this)]
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    inst✝ : CharZero K
    a b : Nat
    h : LE.le a b
    this : ∀ {n : Nat}, Ne (↑n.factorial) 0
    ⊢ Eq (HMul.hMul (↑(b.choose a)) (HMul.hMul ↑a.factorial ↑(HSub.hSub b a).facto …
  -/
  rw_mod_cast [← mul_assoc, choose_mul_factorial_mul_factorial h]
  /-
    🎉 no goals
  -/


theorem cast_add_choose {a b : ℕ} : ((a + b).choose a : K) = (a + b)! / (a ! * b !) := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    inst✝ : CharZero K
    a b : Nat
    ⊢ Eq (↑((HAdd.hAdd a b).choose a)) (HDiv.hDiv (↑(HAdd.hAdd a b).factorial) (HM …
  -/
  rw [cast_choose K (_root_.le_add_right le_rfl), add_tsub_cancel_left]
  /-
    🎉 no goals
  -/


theorem cast_choose_eq_ascPochhammer_div (a b : ℕ) :
    (a.choose b : K) = (ascPochhammer K b).eval ↑(a - (b - 1)) / b ! := by
  rw [eq_div_iff_mul_eq (cast_ne_zero.2 b.factorial_ne_zero : (b ! : K) ≠ 0), ← cast_mul,
    mul_comm, ← descFactorial_eq_factorial_mul_choose, ← cast_descFactorial]


theorem cast_choose_two (a : ℕ) : (a.choose 2 : K) = a * (a - 1) / 2 := by
  rw [← cast_descFactorial_two, descFactorial_eq_factorial_mul_choose, factorial_two, mul_comm,
    cast_mul, cast_two, eq_div_iff_mul_eq (two_ne_zero : (2 : K) ≠ 0)]


