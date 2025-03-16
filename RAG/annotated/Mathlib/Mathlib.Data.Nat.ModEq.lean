/-- Modular equality. `n.ModEq a b`, or `a ≡ b [MOD n]`, means that `a - b` is a multiple of `n`. -/
def ModEq (n a b : ℕ) :=
  a % n = b % n


@[inherit_doc]
notation:50 a " ≡ " b " [MOD " n "]" => ModEq n a b


instance : Decidable (ModEq n a b) := inferInstanceAs <| Decidable (a % n = b % n)


@[refl]
protected theorem refl (a : ℕ) : a ≡ a [MOD n] := rfl


protected theorem rfl : a ≡ a [MOD n] :=
  ModEq.refl _


instance : IsRefl _ (ModEq n) :=
  ⟨ModEq.refl⟩


@[symm]
protected theorem symm : a ≡ b [MOD n] → b ≡ a [MOD n] :=
  Eq.symm


@[trans]
protected theorem trans : a ≡ b [MOD n] → b ≡ c [MOD n] → a ≡ c [MOD n] :=
  Eq.trans


instance : Trans (ModEq n) (ModEq n) (ModEq n) where
  trans := Nat.ModEq.trans


protected theorem comm : a ≡ b [MOD n] ↔ b ≡ a [MOD n] :=
  ⟨ModEq.symm, ModEq.symm⟩


                                                         /-
                                                           n a : Nat
                                                           ⊢ Iff (n.ModEq a 0) (Dvd.dvd n a)
                                                         -/
theorem modEq_zero_iff_dvd : a ≡ 0 [MOD n] ↔ n ∣ a := by rw [ModEq, zero_mod, dvd_iff_mod_eq_zero]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem _root_.Dvd.dvd.modEq_zero_nat (h : n ∣ a) : a ≡ 0 [MOD n] :=
  modEq_zero_iff_dvd.2 h


theorem _root_.Dvd.dvd.zero_modEq_nat (h : n ∣ a) : 0 ≡ a [MOD n] :=
  h.modEq_zero_nat.symm


theorem modEq_iff_dvd : a ≡ b [MOD n] ↔ (n : ℤ) ∣ b - a := by
  rw [ModEq, eq_comm, ← Int.natCast_inj, Int.natCast_mod, Int.natCast_mod,
    Int.emod_eq_emod_iff_emod_sub_eq_zero, Int.dvd_iff_emod_eq_zero]


alias ⟨ModEq.dvd, modEq_of_dvd⟩ := modEq_iff_dvd


/-- A variant of `modEq_iff_dvd` with `Nat` divisibility -/
theorem modEq_iff_dvd' (h : a ≤ b) : a ≡ b [MOD n] ↔ n ∣ b - a := by
  /-
    n a b : Nat
    h : LE.le a b
    ⊢ Iff (n.ModEq a b) (Dvd.dvd n (HSub.hSub b a))
  -/
  rw [modEq_iff_dvd, ← Int.natCast_dvd_natCast, Int.ofNat_sub h]
  /-
    🎉 no goals
  -/


theorem mod_modEq (a n) : a % n ≡ a [MOD n] :=
  mod_mod _ _


lemma of_dvd (d : m ∣ n) (h : a ≡ b [MOD n]) : a ≡ b [MOD m] :=
  modEq_of_dvd <| Int.ofNat_dvd.mpr d |>.trans h.dvd


protected theorem mul_left' (c : ℕ) (h : a ≡ b [MOD n]) : c * a ≡ c * b [MOD c * n] := by
  /-
    n a b c : Nat
    h : n.ModEq a b
    ⊢ (HMul.hMul c n).ModEq (HMul.hMul c a) (HMul.hMul c b)
  -/
  unfold ModEq at *; rw [mul_mod_mul_left, mul_mod_mul_left, h]
                     /-
                       🎉 no goals
                     -/


@[gcongr]
protected theorem mul_left (c : ℕ) (h : a ≡ b [MOD n]) : c * a ≡ c * b [MOD n] :=
  (h.mul_left' _).of_dvd (dvd_mul_left _ _)


protected theorem mul_right' (c : ℕ) (h : a ≡ b [MOD n]) : a * c ≡ b * c [MOD n * c] := by
  /-
    n a b c : Nat
    h : n.ModEq a b
    ⊢ (HMul.hMul n c).ModEq (HMul.hMul a c) (HMul.hMul b c)
  -/
  rw [mul_comm a, mul_comm b, mul_comm n]; exact h.mul_left' c
                                           /-
                                             🎉 no goals
                                           -/


@[gcongr]
protected theorem mul_right (c : ℕ) (h : a ≡ b [MOD n]) : a * c ≡ b * c [MOD n] := by
  /-
    n a b c : Nat
    h : n.ModEq a b
    ⊢ n.ModEq (HMul.hMul a c) (HMul.hMul b c)
  -/
  rw [mul_comm a, mul_comm b]; exact h.mul_left c
                               /-
                                 🎉 no goals
                               -/


@[gcongr]
protected theorem mul (h₁ : a ≡ b [MOD n]) (h₂ : c ≡ d [MOD n]) : a * c ≡ b * d [MOD n] :=
  (h₂.mul_left _).trans (h₁.mul_right _)


@[gcongr]
protected theorem pow (m : ℕ) (h : a ≡ b [MOD n]) : a ^ m ≡ b ^ m [MOD n] := by
  induction m with
  | zero => rfl
  | succ d hd =>
    rw [Nat.pow_succ, Nat.pow_succ]
    exact hd.mul h


@[gcongr]
protected theorem add (h₁ : a ≡ b [MOD n]) (h₂ : c ≡ d [MOD n]) : a + c ≡ b + d [MOD n] := by
  /-
    n a b c d : Nat
    h₁ : n.ModEq a b
    h₂ : n.ModEq c d
    ⊢ n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
  -/
  rw [modEq_iff_dvd, Int.ofNat_add, Int.ofNat_add, add_sub_add_comm]
  /-
    n a b c d : Nat
    h₁ : n.ModEq a b
    h₂ : n.ModEq c d
    ⊢ Dvd.dvd (↑n) (HAdd.hAdd (HSub.hSub ↑b ↑a) (HSub.hSub ↑d ↑c))
  -/
  exact Int.dvd_add h₁.dvd h₂.dvd
  /-
    🎉 no goals
  -/


@[gcongr]
protected theorem add_left (c : ℕ) (h : a ≡ b [MOD n]) : c + a ≡ c + b [MOD n] :=
  ModEq.rfl.add h


@[gcongr]
protected theorem add_right (c : ℕ) (h : a ≡ b [MOD n]) : a + c ≡ b + c [MOD n] :=
  h.add ModEq.rfl


protected theorem add_left_cancel (h₁ : a ≡ b [MOD n]) (h₂ : a + c ≡ b + d [MOD n]) :
    c ≡ d [MOD n] := by
  /-
    n a b c d : Nat
    h₁ : n.ModEq a b
    h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
    ⊢ n.ModEq c d
  -/
  simp only [modEq_iff_dvd, Int.ofNat_add] at *
  /-
    n a b c d : Nat
    h₁ : Dvd.dvd (↑n) (HSub.hSub ↑b ↑a)
    h₂ : Dvd.dvd (↑n) (HSub.hSub (HAdd.hAdd ↑b ↑d) (HAdd.hAdd ↑a ↑c))
    ⊢ Dvd.dvd (↑n) (HSub.hSub ↑d ↑c)
  -/
  rw [add_sub_add_comm] at h₂
  /-
    n a b c d : Nat
    h₁ : Dvd.dvd (↑n) (HSub.hSub ↑b ↑a)
    h₂ : Dvd.dvd (↑n) (HAdd.hAdd (HSub.hSub ↑b ↑a) (HSub.hSub ↑d ↑c))
    ⊢ Dvd.dvd (↑n) (HSub.hSub ↑d ↑c)
  -/
  convert Int.dvd_sub h₂ h₁ using 1
  /-
    case h.e'_4
    n a b c d : Nat
    h₁ : Dvd.dvd (↑n) (HSub.hSub ↑b ↑a)
    h₂ : Dvd.dvd (↑n) (HAdd.hAdd (HSub.hSub ↑b ↑a) (HSub.hSub ↑d ↑c))
    ⊢ Eq (HSub.hSub ↑d ↑c) (HSub.hSub (HAdd.hAdd (HSub.hSub ↑b ↑a) (HSub.hSub ↑d ↑ …
  -/
  rw [add_sub_cancel_left]
  /-
    🎉 no goals
  -/


protected theorem add_left_cancel' (c : ℕ) (h : c + a ≡ c + b [MOD n]) : a ≡ b [MOD n] :=
  ModEq.rfl.add_left_cancel h


protected theorem add_right_cancel (h₁ : c ≡ d [MOD n]) (h₂ : a + c ≡ b + d [MOD n]) :
    a ≡ b [MOD n] := by
  /-
    n a b c d : Nat
    h₁ : n.ModEq c d
    h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
    ⊢ n.ModEq a b
  -/
  rw [add_comm a, add_comm b] at h₂
  /-
    n a b c d : Nat
    h₁ : n.ModEq c d
    h₂ : n.ModEq (HAdd.hAdd c a) (HAdd.hAdd d b)
    ⊢ n.ModEq a b
  -/
  exact h₁.add_left_cancel h₂
  /-
    🎉 no goals
  -/


protected theorem add_right_cancel' (c : ℕ) (h : a + c ≡ b + c [MOD n]) : a ≡ b [MOD n] :=
  ModEq.rfl.add_right_cancel h


/-- Cancel left multiplication on both sides of the `≡` and in the modulus.

For cancelling left multiplication in the modulus, see `Nat.ModEq.of_mul_left`. -/
protected theorem mul_left_cancel' {a b c m : ℕ} (hc : c ≠ 0) :
    c * a ≡ c * b [MOD c * m] → a ≡ b [MOD m] := by
  /-
    a b c m : Nat
    hc : Ne c 0
    ⊢ (HMul.hMul c m).ModEq (HMul.hMul c a) (HMul.hMul c b) → m.ModEq a b
  -/
  simp only [modEq_iff_dvd, Int.natCast_mul, ← Int.mul_sub]
  /-
    a b c m : Nat
    hc : Ne c 0
    ⊢ Dvd.dvd (HMul.hMul ↑c ↑m) (HMul.hMul (↑c) (HSub.hSub ↑b ↑a)) → Dvd.dvd (↑m)  …
  -/
  exact fun h => (Int.dvd_of_mul_dvd_mul_left (Int.ofNat_ne_zero.mpr hc) h)
  /-
    🎉 no goals
  -/


protected theorem mul_left_cancel_iff' {a b c m : ℕ} (hc : c ≠ 0) :
    c * a ≡ c * b [MOD c * m] ↔ a ≡ b [MOD m] :=
  ⟨ModEq.mul_left_cancel' hc, ModEq.mul_left' _⟩


/-- Cancel right multiplication on both sides of the `≡` and in the modulus.

For cancelling right multiplication in the modulus, see `Nat.ModEq.of_mul_right`. -/
protected theorem mul_right_cancel' {a b c m : ℕ} (hc : c ≠ 0) :
    a * c ≡ b * c [MOD m * c] → a ≡ b [MOD m] := by
  /-
    a b c m : Nat
    hc : Ne c 0
    ⊢ (HMul.hMul m c).ModEq (HMul.hMul a c) (HMul.hMul b c) → m.ModEq a b
  -/
  simp only [modEq_iff_dvd, Int.natCast_mul, ← Int.sub_mul]
  /-
    a b c m : Nat
    hc : Ne c 0
    ⊢ Dvd.dvd (HMul.hMul ↑m ↑c) (HMul.hMul (HSub.hSub ↑b ↑a) ↑c) → Dvd.dvd (↑m) (H …
  -/
  exact fun h => (Int.dvd_of_mul_dvd_mul_right (Int.ofNat_ne_zero.mpr hc) h)
  /-
    🎉 no goals
  -/


protected theorem mul_right_cancel_iff' {a b c m : ℕ} (hc : c ≠ 0) :
    a * c ≡ b * c [MOD m * c] ↔ a ≡ b [MOD m] :=
  ⟨ModEq.mul_right_cancel' hc, ModEq.mul_right' _⟩


/-- Cancel left multiplication in the modulus.

For cancelling left multiplication on both sides of the `≡`, see `nat.modeq.mul_left_cancel'`. -/
lemma of_mul_left (m : ℕ) (h : a ≡ b [MOD m * n]) : a ≡ b [MOD n] := by
  /-
    n a b m : Nat
    h : (HMul.hMul m n).ModEq a b
    ⊢ n.ModEq a b
  -/
  rw [modEq_iff_dvd] at *
  /-
    n a b m : Nat
    h : Dvd.dvd (↑(HMul.hMul m n)) (HSub.hSub ↑b ↑a)
    ⊢ Dvd.dvd (↑n) (HSub.hSub ↑b ↑a)
  -/
  exact (dvd_mul_left (n : ℤ) (m : ℤ)).trans h
  /-
    🎉 no goals
  -/


/-- Cancel right multiplication in the modulus.

For cancelling right multiplication on both sides of the `≡`, see `nat.modeq.mul_right_cancel'`. -/
lemma of_mul_right (m : ℕ) : a ≡ b [MOD n * m] → a ≡ b [MOD n] := mul_comm m n ▸ of_mul_left _


theorem of_div (h : a / c ≡ b / c [MOD m / c]) (ha : c ∣ a) (ha : c ∣ b) (ha : c ∣ m) :
                        /-
                          m a b c : Nat
                          h : (HDiv.hDiv m c).ModEq (HDiv.hDiv a c) (HDiv.hDiv b c)
                          ha✝¹ : Dvd.dvd c a
                          ha✝ : Dvd.dvd c b
                          ha : Dvd.dvd c m
                          ⊢ m.ModEq a b
                        -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    a ≡ b [MOD m] := by convert h.mul_left' c <;> rwa [Nat.mul_div_cancel']
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                                       /-
                                                                         a b : Nat
                                                                         h : LE.le b a
                                                                         ⊢ Dvd.dvd (↑(HSub.hSub a b)) (HSub.hSub ↑a ↑b)
                                                                       -/
lemma modEq_sub (h : b ≤ a) : a ≡ b [MOD a - b] := (modEq_of_dvd <| by rw [Int.ofNat_sub h]).symm
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma modEq_one : a ≡ b [MOD 1] := modEq_of_dvd <| one_dvd _


                                                           /-
                                                             a b : Nat
                                                             ⊢ Iff (Nat.ModEq 0 a b) (Eq a b)
                                                           -/
@[simp] lemma modEq_zero_iff : a ≡ b [MOD 0] ↔ a = b := by rw [ModEq, mod_zero, mod_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                       /-
                                                         n a : Nat
                                                         ⊢ n.ModEq (HAdd.hAdd n a) a
                                                       -/
@[simp] lemma add_modEq_left : n + a ≡ a [MOD n] := by rw [ModEq, add_mod_left]
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                        /-
                                                          n a : Nat
                                                          ⊢ n.ModEq (HAdd.hAdd a n) a
                                                        -/
@[simp] lemma add_modEq_right : a + n ≡ a [MOD n] := by rw [ModEq, add_mod_right]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem le_of_lt_add (h1 : a ≡ b [MOD m]) (h2 : a < b + m) : a ≤ b :=
  (le_total a b).elim id fun h3 =>
    Nat.le_of_sub_eq_zero
                                                                 /-
                                                                   m a b : Nat
                                                                   h1 : m.ModEq a b
                                                                   h2 : LT.lt a (HAdd.hAdd b m)
                                                                   h3 : LE.le b a
                                                                   ⊢ LT.lt (HSub.hSub a b) m
                                                                 -/
      (eq_zero_of_dvd_of_lt ((modEq_iff_dvd' h3).mp h1.symm) (by omega))
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem add_le_of_lt (h1 : a ≡ b [MOD m]) (h2 : a < b) : a + m ≤ b :=
                                              /-
                                                m a b : Nat
                                                h1 : m.ModEq a b
                                                h2 : LT.lt a b
                                                ⊢ LT.lt (HAdd.hAdd a m) (HAdd.hAdd b m)
                                              -/
  le_of_lt_add (add_modEq_right.trans h1) (by omega)
                                              /-
                                                🎉 no goals
                                              -/


theorem dvd_iff (h : a ≡ b [MOD m]) (hdm : d ∣ m) : d ∣ a ↔ d ∣ b := by
  /-
    m a b d : Nat
    h : m.ModEq a b
    hdm : Dvd.dvd d m
    ⊢ Iff (Dvd.dvd d a) (Dvd.dvd d b)
  -/
  simp only [← modEq_zero_iff_dvd]
  /-
    m a b d : Nat
    h : m.ModEq a b
    hdm : Dvd.dvd d m
    ⊢ Iff (d.ModEq a 0) (d.ModEq b 0)
  -/
  replace h := h.of_dvd hdm
  /-
    m a b d : Nat
    hdm : Dvd.dvd d m
    h : d.ModEq a b
    ⊢ Iff (d.ModEq a 0) (d.ModEq b 0)
  -/
  exact ⟨h.symm.trans, h.trans⟩
  /-
    🎉 no goals
  -/


theorem gcd_eq (h : a ≡ b [MOD m]) : gcd a m = gcd b m := by
  /-
    m a b : Nat
    h : m.ModEq a b
    ⊢ Eq (a.gcd m) (b.gcd m)
  -/
  have h1 := gcd_dvd_right a m
  /-
    m a b : Nat
    h : m.ModEq a b
    h1 : Dvd.dvd (a.gcd m) m
    ⊢ Eq (a.gcd m) (b.gcd m)
  -/
  have h2 := gcd_dvd_right b m
  exact
    dvd_antisymm (dvd_gcd ((h.dvd_iff h1).mp (gcd_dvd_left a m)) h1)
      (dvd_gcd ((h.dvd_iff h2).mpr (gcd_dvd_left b m)) h2)


lemma eq_of_abs_lt (h : a ≡ b [MOD m]) (h2 : |(b : ℤ) - a| < m) : a = b := by
  /-
    m a b : Nat
    h : m.ModEq a b
    h2 : LT.lt (abs (HSub.hSub ↑b ↑a)) ↑m
    ⊢ Eq a b
  -/
  apply Int.ofNat.inj
  /-
    case x
    m a b : Nat
    h : m.ModEq a b
    h2 : LT.lt (abs (HSub.hSub ↑b ↑a)) ↑m
    ⊢ Eq (Int.ofNat a) (Int.ofNat b)
  -/
  rw [eq_comm, ← sub_eq_zero]
  /-
    case x
    m a b : Nat
    h : m.ModEq a b
    h2 : LT.lt (abs (HSub.hSub ↑b ↑a)) ↑m
    ⊢ Eq (HSub.hSub (Int.ofNat b) (Int.ofNat a)) 0
  -/
  exact Int.eq_zero_of_abs_lt_dvd h.dvd h2
  /-
    🎉 no goals
  -/


lemma eq_of_lt_of_lt (h : a ≡ b [MOD m]) (ha : a < m) (hb : b < m) : a = b :=
  h.eq_of_abs_lt <| Int.abs_sub_lt_of_lt_lt ha hb


/-- To cancel a common factor `c` from a `ModEq` we must divide the modulus `m` by `gcd m c` -/
lemma cancel_left_div_gcd (hm : 0 < m) (h : c * a ≡ c * b [MOD m]) :  a ≡ b [MOD m / gcd m c] := by
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    ⊢ (HDiv.hDiv m (m.gcd c)).ModEq a b
  -/
  let d := gcd m c
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    d : Nat := m.gcd c
    ⊢ (HDiv.hDiv m (m.gcd c)).ModEq a b
  -/
  have hmd := gcd_dvd_left m c
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    d : Nat := m.gcd c
    hmd : Dvd.dvd (m.gcd c) m
    ⊢ (HDiv.hDiv m (m.gcd c)).ModEq a b
  -/
  have hcd := gcd_dvd_right m c
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    d : Nat := m.gcd c
    hmd : Dvd.dvd (m.gcd c) m
    hcd : Dvd.dvd (m.gcd c) c
    ⊢ (HDiv.hDiv m (m.gcd c)).ModEq a b
  -/
  rw [modEq_iff_dvd]
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    d : Nat := m.gcd c
    hmd : Dvd.dvd (m.gcd c) m
    hcd : Dvd.dvd (m.gcd c) c
    ⊢ Dvd.dvd (↑(HDiv.hDiv m (m.gcd c))) (HSub.hSub ↑b ↑a)
  -/
  refine @Int.dvd_of_dvd_mul_right_of_gcd_one (m / d) (c / d) (b - a) ?_ ?_
    /-
      case refine_1
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Dvd.dvd (HDiv.hDiv ↑m ↑d) (HMul.hMul (HDiv.hDiv ↑c ↑d) (HSub.hSub ↑b ↑a))
    -/
  · show (m / d : ℤ) ∣ c / d * (b - a)
    /-
      case refine_1
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Dvd.dvd (HDiv.hDiv ↑m ↑d) (HMul.hMul (HDiv.hDiv ↑c ↑d) (HSub.hSub ↑b ↑a))
    -/
    rw [mul_comm, ← Int.mul_ediv_assoc (b - a) (Int.natCast_dvd_natCast.mpr hcd), mul_comm]
    /-
      case refine_1
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Dvd.dvd (HDiv.hDiv ↑m ↑d) (HDiv.hDiv (HMul.hMul (↑c) (HSub.hSub ↑b ↑a)) ↑(m. …
    -/
    apply Int.ediv_dvd_ediv (Int.natCast_dvd_natCast.mpr hmd)
    /-
      case refine_1
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Dvd.dvd (↑m) (HMul.hMul (↑c) (HSub.hSub ↑b ↑a))
    -/
    rw [Int.mul_sub]
    /-
      case refine_1
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Dvd.dvd (↑m) (HSub.hSub (HMul.hMul ↑c ↑b) (HMul.hMul ↑c ↑a))
    -/
    exact modEq_iff_dvd.mp h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      m a b c : Nat
      hm : LT.lt 0 m
      h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
      d : Nat := m.gcd c
      hmd : Dvd.dvd (m.gcd c) m
      hcd : Dvd.dvd (m.gcd c) c
      ⊢ Eq ((HDiv.hDiv ↑m ↑d).gcd (HDiv.hDiv ↑c ↑d)) 1
    -/
  · show Int.gcd (m / d) (c / d) = 1
    simp only [d, ← Int.natCast_div, Int.gcd_natCast_natCast (m / d) (c / d),
      gcd_div hmd hcd, Nat.div_self (gcd_pos_of_pos_left c hm)]


/-- To cancel a common factor `c` from a `ModEq` we must divide the modulus `m` by `gcd m c` -/
lemma cancel_right_div_gcd (hm : 0 < m) (h : a * c ≡ b * c [MOD m]) : a ≡ b [MOD m / gcd m c] := by
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul a c) (HMul.hMul b c)
    ⊢ (HDiv.hDiv m (m.gcd c)).ModEq a b
  -/
  apply cancel_left_div_gcd hm
  /-
    m a b c : Nat
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul a c) (HMul.hMul b c)
    ⊢ m.ModEq (HMul.hMul c a) (HMul.hMul c b)
  -/
  simpa [mul_comm] using h
  /-
    🎉 no goals
  -/


lemma cancel_left_div_gcd' (hm : 0 < m) (hcd : c ≡ d [MOD m]) (h : c * a ≡ d * b [MOD m]) :
    a ≡ b [MOD m / gcd m c] :=
  (h.trans <| hcd.symm.mul_right b).cancel_left_div_gcd hm


lemma cancel_right_div_gcd' (hm : 0 < m) (hcd : c ≡ d [MOD m]) (h : a * c ≡ b * d [MOD m]) :
    a ≡ b [MOD m / gcd m c] :=
  (h.trans <| hcd.symm.mul_left b).cancel_right_div_gcd hm


/-- A common factor that's coprime with the modulus can be cancelled from a `ModEq` -/
lemma cancel_left_of_coprime (hmc : gcd m c = 1) (h : c * a ≡ c * b [MOD m]) : a ≡ b [MOD m] := by
  /-
    m a b c : Nat
    hmc : Eq (m.gcd c) 1
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    ⊢ m.ModEq a b
  -/
  rcases m.eq_zero_or_pos with (rfl | hm)
    /-
      case inl
      a b c : Nat
      hmc : Eq (Nat.gcd 0 c) 1
      h : Nat.ModEq 0 (HMul.hMul c a) (HMul.hMul c b)
      ⊢ Nat.ModEq 0 a b
    -/
  · simp only [gcd_zero_left] at hmc
    /-
      case inl
      a b c : Nat
      h : Nat.ModEq 0 (HMul.hMul c a) (HMul.hMul c b)
      hmc : Eq c 1
      ⊢ Nat.ModEq 0 a b
    -/
    simp only [gcd_zero_left, hmc, one_mul, modEq_zero_iff] at h
    /-
      case inl
      a b c : Nat
      hmc : Eq c 1
      h : Eq a b
      ⊢ Nat.ModEq 0 a b
    -/
    subst h
    /-
      case inl
      a c : Nat
      hmc : Eq c 1
      ⊢ Nat.ModEq 0 a a
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case inr
    m a b c : Nat
    hmc : Eq (m.gcd c) 1
    h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
    hm : GT.gt m 0
    ⊢ m.ModEq a b
  -/
  simpa [hmc] using h.cancel_left_div_gcd hm
  /-
    🎉 no goals
  -/


/-- A common factor that's coprime with the modulus can be cancelled from a `ModEq` -/
lemma cancel_right_of_coprime (hmc : gcd m c = 1) (h : a * c ≡ b * c [MOD m]) : a ≡ b [MOD m] :=
                                   /-
                                     m a b c : Nat
                                     hmc : Eq (m.gcd c) 1
                                     h : m.ModEq (HMul.hMul a c) (HMul.hMul b c)
                                     ⊢ m.ModEq (HMul.hMul c a) (HMul.hMul c b)
                                   -/
  cancel_left_of_coprime hmc <| by simpa [mul_comm] using h
                                   /-
                                     🎉 no goals
                                   -/


/-- The natural number less than `lcm n m` congruent to `a` mod `n` and `b` mod `m` -/
def chineseRemainder' (h : a ≡ b [MOD gcd n m]) : { k // k ≡ a [MOD n] ∧ k ≡ b [MOD m] } :=
  if hn : n = 0 then ⟨a, by
    /-
      m n a b c d : Nat
      h : (n.gcd m).ModEq a b
      hn : Eq n 0
      ⊢ And (n.ModEq a a) (m.ModEq a b)
    -/
    rw [hn, gcd_zero_left] at h; constructor
      /-
        case left
        m n a b c d : Nat
        h : m.ModEq a b
        hn : Eq n 0
        ⊢ n.ModEq a a
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case right
        m n a b c d : Nat
        h : m.ModEq a b
        hn : Eq n 0
        ⊢ m.ModEq a b
      -/
    · exact h⟩
      /-
        🎉 no goals
      -/
  else
    if hm : m = 0 then ⟨b, by
      /-
        m n a b c d : Nat
        h : (n.gcd m).ModEq a b
        hn : Not (Eq n 0)
        hm : Eq m 0
        ⊢ And (n.ModEq b a) (m.ModEq b b)
      -/
      rw [hm, gcd_zero_right] at h; constructor
        /-
          case left
          m n a b c d : Nat
          h : n.ModEq a b
          hn : Not (Eq n 0)
          hm : Eq m 0
          ⊢ n.ModEq b a
        -/
      · exact h.symm
        /-
          🎉 no goals
        -/
        /-
          case right
          m n a b c d : Nat
          h : n.ModEq a b
          hn : Not (Eq n 0)
          hm : Eq m 0
          ⊢ m.ModEq b b
        -/
      · rfl⟩
        /-
          🎉 no goals
        -/
    else
      ⟨let (c, d) := xgcd n m; Int.toNat ((n * c * b + m * d * a) / gcd n m % lcm n m), by
        /-
          m n a b c d : Nat
          h : (n.gcd m).ModEq a b
          hn : Not (Eq n 0)
          hm : Not (Eq m 0)
          ⊢ And (n.ModEq (Nat.chineseRemainder'.match_1 (fun x => Nat) (n.xgcd m) fun c  …
        -/
        rw [xgcd_val]
        /-
          m n a b c d : Nat
          h : (n.gcd m).ModEq a b
          hn : Not (Eq n 0)
          hm : Not (Eq m 0)
          ⊢ And (n.ModEq (Nat.chineseRemainder'.match_1 (fun x => Nat) { fst := n.gcdA m …
        -/
        dsimp
        rw [modEq_iff_dvd, modEq_iff_dvd,
          Int.toNat_of_nonneg (Int.emod_nonneg _ (Int.natCast_ne_zero.2 (lcm_ne_zero hn hm)))]
        have hnonzero : (gcd n m : ℤ) ≠ 0 := by
          norm_cast
          rw [Nat.gcd_eq_zero_iff, not_and]
          exact fun _ => hm
        /-
          m n a b c d : Nat
          h : (n.gcd m).ModEq a b
          hn : Not (Eq n 0)
          hm : Not (Eq m 0)
          hnonzero : Ne (↑(n.gcd m)) 0
          ⊢ And (Dvd.dvd (↑n) (HSub.hSub (↑a) (HMod.hMod (HDiv.hDiv (HAdd.hAdd (HMul.hMu …
        -/
        have hcoedvd : ∀ t, (gcd n m : ℤ) ∣ t * (b - a) := fun t => h.dvd.mul_left _
        /-
          m n a b c d : Nat
          h : (n.gcd m).ModEq a b
          hn : Not (Eq n 0)
          hm : Not (Eq m 0)
          hnonzero : Ne (↑(n.gcd m)) 0
          hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
          ⊢ And (Dvd.dvd (↑n) (HSub.hSub (↑a) (HMod.hMod (HDiv.hDiv (HAdd.hAdd (HMul.hMu …
        -/
        have := gcd_eq_gcd_ab n m
        /-
          m n a b c d : Nat
          h : (n.gcd m).ModEq a b
          hn : Not (Eq n 0)
          hm : Not (Eq m 0)
          hnonzero : Ne (↑(n.gcd m)) 0
          hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
          this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
          ⊢ And (Dvd.dvd (↑n) (HSub.hSub (↑a) (HMod.hMod (HDiv.hDiv (HAdd.hAdd (HMul.hMu …
        -/
        constructor <;> rw [Int.emod_def, ← sub_add] <;>
            /-
              case left
              m n a b c d : Nat
              h : (n.gcd m).ModEq a b
              hn : Not (Eq n 0)
              hm : Not (Eq m 0)
              hnonzero : Ne (↑(n.gcd m)) 0
              hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
              this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
              ⊢ Dvd.dvd (↑n) (HAdd.hAdd (HSub.hSub (↑a) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HM …
            -/
            refine Int.dvd_add ?_ (dvd_mul_of_dvd_left ?_ _) <;>
          /-
            case left.refine_1
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
            ⊢ Dvd.dvd (↑n) (HSub.hSub (↑a) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul (↑n …
          -/
          try norm_cast
          /-
            case left.refine_1
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
            ⊢ Dvd.dvd (↑n) (HSub.hSub (↑a) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul (↑n …
          -/
        · rw [← sub_eq_iff_eq_add'] at this
          rw [← this, Int.sub_mul, ← add_sub_assoc, add_comm, add_sub_assoc, ← Int.mul_sub,
            Int.add_ediv_of_dvd_left, Int.mul_ediv_cancel_left _ hnonzero,
            Int.mul_ediv_assoc _ h.dvd, ← sub_sub, sub_self, zero_sub, Int.dvd_neg, mul_assoc]
            /-
              case left.refine_1
              m n a b c d : Nat
              h : (n.gcd m).ModEq a b
              hn : Not (Eq n 0)
              hm : Not (Eq m 0)
              hnonzero : Ne (↑(n.gcd m)) 0
              hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
              this : Eq (HSub.hSub (↑(n.gcd m)) (HMul.hMul (↑n) (n.gcdA m))) (HMul.hMul (↑m) …
              ⊢ Dvd.dvd (↑n) (HMul.hMul (↑n) (HMul.hMul (n.gcdA m) (HDiv.hDiv (HSub.hSub ↑b  …
            -/
          · exact dvd_mul_right _ _
            /-
              🎉 no goals
            -/
          /-
            case left.refine_1
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (HSub.hSub (↑(n.gcd m)) (HMul.hMul (↑n) (n.gcdA m))) (HMul.hMul (↑m) …
            ⊢ Dvd.dvd (↑(n.gcd m)) (HMul.hMul ↑(n.gcd m) ↑a)
          -/
          norm_cast
          /-
            case left.refine_1
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (HSub.hSub (↑(n.gcd m)) (HMul.hMul (↑n) (n.gcdA m))) (HMul.hMul (↑m) …
            ⊢ Dvd.dvd (n.gcd m) (HMul.hMul (n.gcd m) a)
          -/
          exact dvd_mul_right _ _
          /-
            🎉 no goals
          -/
          /-
            case left.refine_2
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
            ⊢ Dvd.dvd n (n.lcm m)
          -/
        · exact dvd_lcm_left n m
          /-
            🎉 no goals
          -/
          /-
            case right.refine_1
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
            ⊢ Dvd.dvd (↑m) (HSub.hSub (↑b) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul (↑n …
          -/
        · rw [← sub_eq_iff_eq_add] at this
          rw [← this, Int.sub_mul, sub_add, ← Int.mul_sub, Int.sub_ediv_of_dvd,
            Int.mul_ediv_cancel_left _ hnonzero, Int.mul_ediv_assoc _ h.dvd, ← sub_add, sub_self,
            zero_add, mul_assoc]
            /-
              case right.refine_1
              m n a b c d : Nat
              h : (n.gcd m).ModEq a b
              hn : Not (Eq n 0)
              hm : Not (Eq m 0)
              hnonzero : Ne (↑(n.gcd m)) 0
              hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
              this : Eq (HSub.hSub (↑(n.gcd m)) (HMul.hMul (↑m) (n.gcdB m))) (HMul.hMul (↑n) …
              ⊢ Dvd.dvd (↑m) (HMul.hMul (↑m) (HMul.hMul (n.gcdB m) (HDiv.hDiv (HSub.hSub ↑b  …
            -/
          · exact dvd_mul_right _ _
            /-
              🎉 no goals
            -/
            /-
              case right.refine_1.hcb
              m n a b c d : Nat
              h : (n.gcd m).ModEq a b
              hn : Not (Eq n 0)
              hm : Not (Eq m 0)
              hnonzero : Ne (↑(n.gcd m)) 0
              hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
              this : Eq (HSub.hSub (↑(n.gcd m)) (HMul.hMul (↑m) (n.gcdB m))) (HMul.hMul (↑n) …
              ⊢ Dvd.dvd (↑(n.gcd m)) (HMul.hMul (HMul.hMul (↑m) (n.gcdB m)) (HSub.hSub ↑b ↑a))
            -/
          · exact hcoedvd _
            /-
              🎉 no goals
            -/
          /-
            case right.refine_2
            m n a b c d : Nat
            h : (n.gcd m).ModEq a b
            hn : Not (Eq n 0)
            hm : Not (Eq m 0)
            hnonzero : Ne (↑(n.gcd m)) 0
            hcoedvd : ∀ (t : Int), Dvd.dvd (↑(n.gcd m)) (HMul.hMul t (HSub.hSub ↑b ↑a))
            this : Eq (↑(n.gcd m)) (HAdd.hAdd (HMul.hMul (↑n) (n.gcdA m)) (HMul.hMul (↑m)  …
            ⊢ Dvd.dvd m (n.lcm m)
          -/
        · exact dvd_lcm_right n m⟩
          /-
            🎉 no goals
          -/


/-- The natural number less than `n*m` congruent to `a` mod `n` and `b` mod `m` -/
def chineseRemainder (co : n.Coprime m) (a b : ℕ) : { k // k ≡ a [MOD n] ∧ k ≡ b [MOD m] } :=
                        /-
                          m n a✝ b✝ c d : Nat
                          co : n.Coprime m
                          a b : Nat
                          ⊢ (n.gcd m).ModEq a b
                        -/
  chineseRemainder' (by convert @modEq_one a b)
                        /-
                          🎉 no goals
                        -/


theorem chineseRemainder'_lt_lcm (h : a ≡ b [MOD gcd n m]) (hn : n ≠ 0) (hm : m ≠ 0) :
    ↑(chineseRemainder' h) < lcm n m := by
  /-
    m n a b : Nat
    h : (n.gcd m).ModEq a b
    hn : Ne n 0
    hm : Ne m 0
    ⊢ LT.lt (↑(Nat.chineseRemainder' h)) (n.lcm m)
  -/
  dsimp only [chineseRemainder']
  /-
    m n a b : Nat
    h : (n.gcd m).ModEq a b
    hn : Ne n 0
    hm : Ne m 0
    ⊢ LT.lt (↑(dite (Eq n 0) (fun hn => ⟨a, ⋯⟩) fun hn => dite (Eq m 0) (fun hm => …
  -/
  rw [dif_neg hn, dif_neg hm, Subtype.coe_mk, xgcd_val, ← Int.toNat_natCast (lcm n m)]
  /-
    m n a b : Nat
    h : (n.gcd m).ModEq a b
    hn : Ne n 0
    hm : Ne m 0
    ⊢ LT.lt (HMod.hMod (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul ↑n { fst := n.g …
  -/
  have lcm_pos := Int.natCast_pos.mpr (Nat.pos_of_ne_zero (lcm_ne_zero hn hm))
  /-
    m n a b : Nat
    h : (n.gcd m).ModEq a b
    hn : Ne n 0
    hm : Ne m 0
    lcm_pos : LT.lt 0 ↑(n.lcm m)
    ⊢ LT.lt (HMod.hMod (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul ↑n { fst := n.g …
  -/
  exact (Int.toNat_lt_toNat lcm_pos).mpr (Int.emod_lt_of_pos _ lcm_pos)
  /-
    🎉 no goals
  -/


theorem chineseRemainder_lt_mul (co : n.Coprime m) (a b : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) :
    ↑(chineseRemainder co a b) < n * m :=
  lt_of_lt_of_le (chineseRemainder'_lt_lcm _ hn hm) (le_of_eq co.lcm_eq_mul)


theorem mod_lcm (hn : a ≡ b [MOD n]) (hm : a ≡ b [MOD m]) : a ≡ b [MOD lcm n m] :=
  Nat.modEq_iff_dvd.mpr <| Int.lcm_dvd (Nat.modEq_iff_dvd.mp hn) (Nat.modEq_iff_dvd.mp hm)


theorem chineseRemainder_modEq_unique (co : n.Coprime m) {a b z}
    (hzan : z ≡ a [MOD n]) (hzbm : z ≡ b [MOD m]) : z ≡ chineseRemainder co a b [MOD n*m] := by
  simpa [Nat.Coprime.lcm_eq_mul co] using
    mod_lcm (hzan.trans ((chineseRemainder co a b).prop.1).symm)
      (hzbm.trans ((chineseRemainder co a b).prop.2).symm)


theorem modEq_and_modEq_iff_modEq_mul {a b m n : ℕ} (hmn : m.Coprime n) :
    a ≡ b [MOD m] ∧ a ≡ b [MOD n] ↔ a ≡ b [MOD m * n] :=
  ⟨fun h => by
    rw [Nat.modEq_iff_dvd, Nat.modEq_iff_dvd, ← Int.dvd_natAbs, Int.natCast_dvd_natCast,
      ← Int.dvd_natAbs, Int.natCast_dvd_natCast] at h
    /-
      a b m n : Nat
      hmn : m.Coprime n
      h : And (Dvd.dvd m (HSub.hSub ↑b ↑a).natAbs) (Dvd.dvd n (HSub.hSub ↑b ↑a).natA …
      ⊢ (HMul.hMul m n).ModEq a b
    -/
    rw [Nat.modEq_iff_dvd, ← Int.dvd_natAbs, Int.natCast_dvd_natCast]
    /-
      a b m n : Nat
      hmn : m.Coprime n
      h : And (Dvd.dvd m (HSub.hSub ↑b ↑a).natAbs) (Dvd.dvd n (HSub.hSub ↑b ↑a).natA …
      ⊢ Dvd.dvd (HMul.hMul m n) (HSub.hSub ↑b ↑a).natAbs
    -/
    exact hmn.mul_dvd_of_dvd_of_dvd h.1 h.2,
    /-
      🎉 no goals
    -/
   fun h => ⟨h.of_mul_right _, h.of_mul_left _⟩⟩


theorem coprime_of_mul_modEq_one (b : ℕ) {a n : ℕ} (h : a * b ≡ 1 [MOD n]) : a.Coprime n := by
  /-
    b a n : Nat
    h : n.ModEq (HMul.hMul a b) 1
    ⊢ a.Coprime n
  -/
  obtain ⟨g, hh⟩ := Nat.gcd_dvd_right a n
  /-
    case intro
    b a n : Nat
    h : n.ModEq (HMul.hMul a b) 1
    g : Nat
    hh : Eq n (HMul.hMul (a.gcd n) g)
    ⊢ a.Coprime n
  -/
  rw [Nat.coprime_iff_gcd_eq_one, ← Nat.dvd_one, ← Nat.modEq_zero_iff_dvd]
  calc
    1 ≡ a * b [MOD a.gcd n] := (hh ▸ h).symm.of_mul_right g
    _ ≡ 0 * b [MOD a.gcd n] := (Nat.modEq_zero_iff_dvd.mpr (Nat.gcd_dvd_left _ _)).mul_right b
    _ = 0 := by rw [zero_mul]


theorem add_mod_add_ite (a b c : ℕ) :
    ((a + b) % c + if c ≤ a % c + b % c then c else 0) = a % c + b % c :=
  have : (a + b) % c = (a % c + b % c) % c := ((mod_modEq _ _).add <| mod_modEq _ _).symm
                         /-
                           a b c : Nat
                           this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HMod.hMod (HAdd.hAdd (HMod.hMod a c)  …
                           hc0 : Eq c 0
                           ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) (ite (LE.le c (HAdd.hAdd (HMod.h …
                         -/
  if hc0 : c = 0 then by simp [hc0, Nat.mod_zero]
                         /-
                           🎉 no goals
                         -/
  else by
    /-
      a b c : Nat
      this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HMod.hMod (HAdd.hAdd (HMod.hMod a c)  …
      hc0 : Not (Eq c 0)
      ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) (ite (LE.le c (HAdd.hAdd (HMod.h …
    -/
    rw [this]
    /-
      a b c : Nat
      this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HMod.hMod (HAdd.hAdd (HMod.hMod a c)  …
      hc0 : Not (Eq c 0)
      ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c) (ite …
    -/
    split_ifs with h
    · have h2 : (a % c + b % c) / c < 2 :=
        Nat.div_lt_of_lt_mul
          (by
            rw [mul_two]
            exact
              add_lt_add (Nat.mod_lt _ (Nat.pos_of_ne_zero hc0))
                (Nat.mod_lt _ (Nat.pos_of_ne_zero hc0)))
      /-
        case pos
        a b c : Nat
        this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HMod.hMod (HAdd.hAdd (HMod.hMod a c)  …
        hc0 : Not (Eq c 0)
        h : LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c))
        h2 : LT.lt (HDiv.hDiv (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c) 2
        ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c) c) ( …
      -/
      have h0 : 0 < (a % c + b % c) / c := Nat.div_pos h (Nat.pos_of_ne_zero hc0)
      rw [← @add_right_cancel_iff _ _ _ (c * ((a % c + b % c) / c)), add_comm _ c, add_assoc,
        mod_add_div, le_antisymm (le_of_lt_succ h2) h0, mul_one, add_comm]
      /-
        case neg
        a b c : Nat
        this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HMod.hMod (HAdd.hAdd (HMod.hMod a c)  …
        hc0 : Not (Eq c 0)
        h : Not (LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)))
        ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c) 0) ( …
      -/
    · rw [Nat.mod_eq_of_lt (lt_of_not_ge h), add_zero]
      /-
        🎉 no goals
      -/


theorem add_mod_of_add_mod_lt {a b c : ℕ} (hc : a % c + b % c < c) :
                                      /-
                                        a b c : Nat
                                        hc : LT.lt (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c
                                        ⊢ Eq (HMod.hMod (HAdd.hAdd a b) c) (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c))
                                      -/
    (a + b) % c = a % c + b % c := by rw [← add_mod_add_ite, if_neg (not_le_of_lt hc), add_zero]
                                      /-
                                        🎉 no goals
                                      -/


theorem add_mod_add_of_le_add_mod {a b c : ℕ} (hc : c ≤ a % c + b % c) :
                                          /-
                                            a b c : Nat
                                            hc : LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c))
                                            ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) c) (HAdd.hAdd (HMod.hMod a c) (H …
                                          -/
    (a + b) % c + c = a % c + b % c := by rw [← add_mod_add_ite, if_pos hc]
                                          /-
                                            🎉 no goals
                                          -/


theorem add_div {a b c : ℕ} (hc0 : 0 < c) :
    (a + b) / c = a / c + b / c + if c ≤ a % c + b % c then 1 else 0 := by
  /-
    a b c : Nat
    hc0 : LT.lt 0 c
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HAdd.hAdd (HDiv.hDiv a c) (HDiv …
  -/
  rw [← mul_right_inj' hc0.ne', ← @add_left_cancel_iff _ _ _ ((a + b) % c + a % c + b % c)]
  suffices
    (a + b) % c + c * ((a + b) / c) + a % c + b % c =
      (a % c + c * (a / c) + (b % c + c * (b / c)) + c * if c ≤ a % c + b % c then 1 else 0) +
        (a + b) % c
    by simpa only [mul_add, add_comm, add_left_comm, add_assoc]
  /-
    a b c : Nat
    hc0 : LT.lt 0 c
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) (HMul.hMul …
  -/
  rw [mod_add_div, mod_add_div, mod_add_div, mul_ite, add_assoc, add_assoc]
  /-
    a b c : Nat
    hc0 : LT.lt 0 c
    ⊢ Eq (HAdd.hAdd a (HAdd.hAdd b (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)))) ( …
  -/
  conv_lhs => rw [← add_mod_add_ite]
  /-
    a b c : Nat
    hc0 : LT.lt 0 c
    ⊢ Eq (HAdd.hAdd a (HAdd.hAdd b (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) (ite ( …
  -/
  simp only [mul_one, mul_zero]
  /-
    a b c : Nat
    hc0 : LT.lt 0 c
    ⊢ Eq (HAdd.hAdd a (HAdd.hAdd b (HAdd.hAdd (HMod.hMod (HAdd.hAdd a b) c) (ite ( …
  -/
  ac_rfl
  /-
    🎉 no goals
  -/


theorem add_div_eq_of_add_mod_lt {a b c : ℕ} (hc : a % c + b % c < c) :
    (a + b) / c = a / c + b / c :=
                         /-
                           a b c : Nat
                           hc : LT.lt (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c
                           hc0 : Eq c 0
                           ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c))
                         -/
  if hc0 : c = 0 then by simp [hc0]
                         /-
                           🎉 no goals
                         -/
          /-
            a b c : Nat
            hc : LT.lt (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c
            hc0 : Not (Eq c 0)
            ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c))
          -/
  else by rw [add_div (Nat.pos_of_ne_zero hc0), if_neg (not_le_of_lt hc), add_zero]
          /-
            🎉 no goals
          -/


protected theorem add_div_of_dvd_right {a b c : ℕ} (hca : c ∣ a) : (a + b) / c = a / c + b / c :=
                       /-
                         a b c : Nat
                         hca : Dvd.dvd c a
                         h : Eq c 0
                         ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c))
                       -/
  if h : c = 0 then by simp [h]
                       /-
                         🎉 no goals
                       -/
  else
    add_div_eq_of_add_mod_lt
      (by
        /-
          a b c : Nat
          hca : Dvd.dvd c a
          h : Not (Eq c 0)
          ⊢ LT.lt (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)) c
        -/
        rw [Nat.mod_eq_zero_of_dvd hca, zero_add]
        /-
          a b c : Nat
          hca : Dvd.dvd c a
          h : Not (Eq c 0)
          ⊢ LT.lt (HMod.hMod b c) c
        -/
        exact Nat.mod_lt _ (zero_lt_of_ne_zero h))
        /-
          🎉 no goals
        -/


protected theorem add_div_of_dvd_left {a b c : ℕ} (hca : c ∣ b) : (a + b) / c = a / c + b / c := by
  /-
    a b c : Nat
    hca : Dvd.dvd c b
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c))
  -/
  rwa [add_comm, Nat.add_div_of_dvd_right, add_comm]
  /-
    🎉 no goals
  -/


theorem add_div_eq_of_le_mod_add_mod {a b c : ℕ} (hc : c ≤ a % c + b % c) (hc0 : 0 < c) :
                                          /-
                                            a b c : Nat
                                            hc : LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c))
                                            hc0 : LT.lt 0 c
                                            ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HAdd.hAdd (HDiv.hDiv a c) (HDiv …
                                          -/
    (a + b) / c = a / c + b / c + 1 := by rw [add_div hc0, if_pos hc]
                                          /-
                                            🎉 no goals
                                          -/


theorem add_div_le_add_div (a b c : ℕ) : a / c + b / c ≤ (a + b) / c :=
                         /-
                           a b c : Nat
                           hc0 : Eq c 0
                           ⊢ LE.le (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv (HAdd.hAdd a b) …
                         -/
  if hc0 : c = 0 then by simp [hc0]
                         /-
                           🎉 no goals
                         -/
          /-
            a b c : Nat
            hc0 : Not (Eq c 0)
            ⊢ LE.le (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv (HAdd.hAdd a b) …
          -/
  else by rw [Nat.add_div (Nat.pos_of_ne_zero hc0)]; exact Nat.le_add_right _ _
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem le_mod_add_mod_of_dvd_add_of_not_dvd {a b c : ℕ} (h : c ∣ a + b) (ha : ¬c ∣ a) :
    c ≤ a % c + b % c :=
  by_contradiction fun hc => by
    /-
      a b c : Nat
      h : Dvd.dvd c (HAdd.hAdd a b)
      ha : Not (Dvd.dvd c a)
      hc : Not (LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)))
      ⊢ False
    -/
    have : (a + b) % c = a % c + b % c := add_mod_of_add_mod_lt (lt_of_not_ge hc)
    /-
      a b c : Nat
      h : Dvd.dvd c (HAdd.hAdd a b)
      ha : Not (Dvd.dvd c a)
      hc : Not (LE.le c (HAdd.hAdd (HMod.hMod a c) (HMod.hMod b c)))
      this : Eq (HMod.hMod (HAdd.hAdd a b) c) (HAdd.hAdd (HMod.hMod a c) (HMod.hMod  …
      ⊢ False
    -/
    simp_all [dvd_iff_mod_eq_zero]
    /-
      🎉 no goals
    -/


theorem odd_mul_odd {n m : ℕ} : n % 2 = 1 → m % 2 = 1 → n * m % 2 = 1 := by
  /-
    n m : Nat
    ⊢ Eq (HMod.hMod n 2) 1 → Eq (HMod.hMod m 2) 1 → Eq (HMod.hMod (HMul.hMul n m)  …
  -/
  simpa [Nat.ModEq] using @ModEq.mul 2 n 1 m 1
  /-
    🎉 no goals
  -/


theorem odd_mul_odd_div_two {m n : ℕ} (hm1 : m % 2 = 1) (hn1 : n % 2 = 1) :
    m * n / 2 = m * (n / 2) + m / 2 :=
                                                     /-
                                                       m n : Nat
                                                       hm1 : Eq (HMod.hMod m 2) 1
                                                       hn1 : Eq (HMod.hMod n 2) 1
                                                       h : Eq n 0
                                                       ⊢ False
                                                     -/
  have hn0 : 0 < n := Nat.pos_of_ne_zero fun h => by simp_all
                                                     /-
                                                       🎉 no goals
                                                     -/
  mul_right_injective₀ two_ne_zero <| by
    /-
      m n : Nat
      hm1 : Eq (HMod.hMod m 2) 1
      hn1 : Eq (HMod.hMod n 2) 1
      hn0 : LT.lt 0 n
      ⊢ Eq ((fun x => HMul.hMul 2 x) (HDiv.hDiv (HMul.hMul m n) 2)) ((fun x => HMul. …
    -/
    dsimp
    rw [mul_add, two_mul_odd_div_two hm1, mul_left_comm, two_mul_odd_div_two hn1,
      two_mul_odd_div_two (Nat.odd_mul_odd hm1 hn1), Nat.mul_sub, mul_one, ←
      Nat.add_sub_assoc (by omega), Nat.sub_add_cancel (Nat.le_mul_of_pos_right m hn0)]


theorem odd_of_mod_four_eq_one {n : ℕ} : n % 4 = 1 → n % 2 = 1 := by
  /-
    n : Nat
    ⊢ Eq (HMod.hMod n 4) 1 → Eq (HMod.hMod n 2) 1
  -/
  simpa [ModEq] using @ModEq.of_mul_left 2 n 1 2
  /-
    🎉 no goals
  -/


theorem odd_of_mod_four_eq_three {n : ℕ} : n % 4 = 3 → n % 2 = 1 := by
  /-
    n : Nat
    ⊢ Eq (HMod.hMod n 4) 3 → Eq (HMod.hMod n 2) 1
  -/
  simpa [ModEq] using @ModEq.of_mul_left 2 n 3 2
  /-
    🎉 no goals
  -/


/-- A natural number is odd iff it has residue `1` or `3` mod `4`-/
theorem odd_mod_four_iff {n : ℕ} : n % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 3 :=
                                                               /-
                                                                 n : Nat
                                                                 ⊢ ∀ (m : Nat), LT.lt m 4 → Eq (HMod.hMod m 2) 1 → Or (Eq m 1) (Eq m 3)
                                                               -/
  have help : ∀ m : ℕ, m < 4 → m % 2 = 1 → m = 1 ∨ m = 3 := by decide
                                                               /-
                                                                 🎉 no goals
                                                               -/
  ⟨fun hn =>
                               /-
                                 n : Nat
                                 help : ∀ (m : Nat), LT.lt m 4 → Eq (HMod.hMod m 2) 1 → Or (Eq m 1) (Eq m 3)
                                 hn : Eq (HMod.hMod n 2) 1
                                 ⊢ GT.gt 4 0
                               -/
                               /-
                                 🎉 no goals
                               -/
    help (n % 4) (mod_lt n (by omega)) <| (mod_mod_of_dvd n (by decide : 2 ∣ 4)).trans hn,
                                                                /-
                                                                  🎉 no goals
                                                                -/
    fun h => Or.elim h odd_of_mod_four_eq_one odd_of_mod_four_eq_three⟩


lemma mod_eq_of_modEq {a b n} (h : a ≡ b [MOD n]) (hb : b < n) : a % n = b :=
  Eq.trans h (mod_eq_of_lt hb)


