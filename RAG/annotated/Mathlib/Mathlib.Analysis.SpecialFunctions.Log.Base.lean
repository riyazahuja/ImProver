/-- The real logarithm in a given base. As with the natural logarithm, we define `logb b x` to
be `logb b |x|` for `x < 0`, and `0` for `x = 0`. -/
@[pp_nodot]
noncomputable def logb (b x : ℝ) : ℝ :=
  log x / log b


theorem log_div_log : log x / log b = logb b x :=
  rfl


@[simp]
                                       /-
                                         b : Real
                                         ⊢ Eq (Real.logb b 0) 0
                                       -/
theorem logb_zero : logb b 0 = 0 := by simp [logb]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
                                      /-
                                        b : Real
                                        ⊢ Eq (Real.logb b 1) 0
                                      -/
theorem logb_one : logb b 1 = 0 := by simp [logb]
                                      /-
                                        🎉 no goals
                                      -/


                                            /-
                                              x : Real
                                              ⊢ Eq (Real.logb 0 x) 0
                                            -/
theorem logb_zero_left : logb 0 x = 0 := by simp only [← log_div_log, log_zero, div_zero]
                                            /-
                                              🎉 no goals
                                            -/


                                                          /-
                                                            ⊢ Eq (Real.logb 0) 0
                                                          -/
@[simp] theorem logb_zero_left_eq_zero : logb 0 = 0 := by ext; rw [logb_zero_left, Pi.zero_apply]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                           /-
                                             x : Real
                                             ⊢ Eq (Real.logb 1 x) 0
                                           -/
theorem logb_one_left : logb 1 x = 0 := by simp only [← log_div_log, log_one, div_zero]
                                           /-
                                             🎉 no goals
                                           -/


                                                         /-
                                                           ⊢ Eq (Real.logb 1) 0
                                                         -/
@[simp] theorem logb_one_left_eq_zero : logb 1 = 0 := by ext; rw [logb_one_left, Pi.zero_apply]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
lemma logb_self_eq_one (hb : 1 < b) : logb b b = 1 :=
  div_self (log_pos hb).ne'


lemma logb_self_eq_one_iff : logb b b = 1 ↔ b ≠ 0 ∧ b ≠ 1 ∧ b ≠ -1 :=
                            /-
                              b : Real
                              h : Eq (Real.logb b b) 1
                              h' : Eq (Real.log b) 0
                              ⊢ False
                            -/
  Iff.trans ⟨fun h h' => by simp [logb, h'] at h, div_self⟩ log_ne_zero
                            /-
                              🎉 no goals
                            -/


@[simp]
                                                       /-
                                                         b x : Real
                                                         ⊢ Eq (Real.logb b (abs x)) (Real.logb b x)
                                                       -/
theorem logb_abs (x : ℝ) : logb b |x| = logb b x := by rw [logb, logb, log_abs]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem logb_neg_eq_logb (x : ℝ) : logb b (-x) = logb b x := by
  /-
    b x : Real
    ⊢ Eq (Real.logb b (Neg.neg x)) (Real.logb b x)
  -/
  rw [← logb_abs x, ← logb_abs (-x), abs_neg]
  /-
    🎉 no goals
  -/


theorem logb_mul (hx : x ≠ 0) (hy : y ≠ 0) : logb b (x * y) = logb b x + logb b y := by
  /-
    b x y : Real
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (Real.logb b (HMul.hMul x y)) (HAdd.hAdd (Real.logb b x) (Real.logb b y))
  -/
  simp_rw [logb, log_mul hx hy, add_div]
  /-
    🎉 no goals
  -/


theorem logb_div (hx : x ≠ 0) (hy : y ≠ 0) : logb b (x / y) = logb b x - logb b y := by
  /-
    b x y : Real
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (Real.logb b (HDiv.hDiv x y)) (HSub.hSub (Real.logb b x) (Real.logb b y))
  -/
  simp_rw [logb, log_div hx hy, sub_div]
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          b x : Real
                                                          ⊢ Eq (Real.logb b (Inv.inv x)) (Neg.neg (Real.logb b x))
                                                        -/
theorem logb_inv (x : ℝ) : logb b x⁻¹ = -logb b x := by simp [logb, neg_div]
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                           /-
                                                             a b : Real
                                                             ⊢ Eq (Inv.inv (Real.logb a b)) (Real.logb b a)
                                                           -/
theorem inv_logb (a b : ℝ) : (logb a b)⁻¹ = logb b a := by simp_rw [logb, inv_div]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem inv_logb_mul_base {a b : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (c : ℝ) :
    (logb (a * b) c)⁻¹ = (logb a c)⁻¹ + (logb b c)⁻¹ := by
  /-
    a b : Real
    h₁ : Ne a 0
    h₂ : Ne b 0
    c : Real
    ⊢ Eq (Inv.inv (Real.logb (HMul.hMul a b) c)) (HAdd.hAdd (Inv.inv (Real.logb a  …
  -/
  simp_rw [inv_logb]; exact logb_mul h₁ h₂
                      /-
                        🎉 no goals
                      -/


theorem inv_logb_div_base {a b : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (c : ℝ) :
    (logb (a / b) c)⁻¹ = (logb a c)⁻¹ - (logb b c)⁻¹ := by
  /-
    a b : Real
    h₁ : Ne a 0
    h₂ : Ne b 0
    c : Real
    ⊢ Eq (Inv.inv (Real.logb (HDiv.hDiv a b) c)) (HSub.hSub (Inv.inv (Real.logb a  …
  -/
  simp_rw [inv_logb]; exact logb_div h₁ h₂
                      /-
                        🎉 no goals
                      -/


theorem logb_mul_base {a b : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (c : ℝ) :
                                                           /-
                                                             a b : Real
                                                             h₁ : Ne a 0
                                                             h₂ : Ne b 0
                                                             c : Real
                                                             ⊢ Eq (Real.logb (HMul.hMul a b) c) (Inv.inv (HAdd.hAdd (Inv.inv (Real.logb a c …
                                                           -/
    logb (a * b) c = ((logb a c)⁻¹ + (logb b c)⁻¹)⁻¹ := by rw [← inv_logb_mul_base h₁ h₂ c, inv_inv]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem logb_div_base {a b : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (c : ℝ) :
                                                           /-
                                                             a b : Real
                                                             h₁ : Ne a 0
                                                             h₂ : Ne b 0
                                                             c : Real
                                                             ⊢ Eq (Real.logb (HDiv.hDiv a b) c) (Inv.inv (HSub.hSub (Inv.inv (Real.logb a c …
                                                           -/
    logb (a / b) c = ((logb a c)⁻¹ - (logb b c)⁻¹)⁻¹ := by rw [← inv_logb_div_base h₁ h₂ c, inv_inv]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem mul_logb {a b c : ℝ} (h₁ : b ≠ 0) (h₂ : b ≠ 1) (h₃ : b ≠ -1) :
    logb a b * logb b c = logb a c := by
  /-
    a b c : Real
    h₁ : Ne b 0
    h₂ : Ne b 1
    h₃ : Ne b (-1)
    ⊢ Eq (HMul.hMul (Real.logb a b) (Real.logb b c)) (Real.logb a c)
  -/
  unfold logb
  /-
    a b c : Real
    h₁ : Ne b 0
    h₂ : Ne b 1
    h₃ : Ne b (-1)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Real.log b) (Real.log a)) (HDiv.hDiv (Real.log c)  …
  -/
  rw [mul_comm, div_mul_div_cancel₀ (log_ne_zero.mpr ⟨h₁, h₂, h₃⟩)]
  /-
    🎉 no goals
  -/


theorem div_logb {a b c : ℝ} (h₁ : c ≠ 0) (h₂ : c ≠ 1) (h₃ : c ≠ -1) :
    logb a c / logb b c = logb a b :=
  div_div_div_cancel_left' _ _ <| log_ne_zero.mpr ⟨h₁, h₂, h₃⟩


theorem logb_rpow_eq_mul_logb_of_pos (hx : 0 < x) : logb b (x ^ y) = y * logb b x := by
  /-
    b x y : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.logb b (HPow.hPow x y)) (HMul.hMul y (Real.logb b x))
  -/
  rw [logb, log_rpow hx, logb, mul_div_assoc]
  /-
    🎉 no goals
  -/


theorem logb_pow (b x : ℝ) (k : ℕ) : logb b (x ^ k) = k * logb b x := by
  /-
    b x : Real
    k : Nat
    ⊢ Eq (Real.logb b (HPow.hPow x k)) (HMul.hMul (↑k) (Real.logb b x))
  -/
  rw [logb, logb, log_pow, mul_div_assoc]
  /-
    🎉 no goals
  -/


private theorem log_b_ne_zero : log b ≠ 0 := by
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    ⊢ Ne (Real.log b) 0
  -/
  have b_ne_zero : b ≠ 0 := by linarith
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    b_ne_zero : Ne b 0
    ⊢ Ne (Real.log b) 0
  -/
  have b_ne_minus_one : b ≠ -1 := by linarith
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    b_ne_zero : Ne b 0
    b_ne_minus_one : Ne b (-1)
    ⊢ Ne (Real.log b) 0
  -/
  simp [b_ne_one, b_ne_zero, b_ne_minus_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem logb_rpow : logb b (b ^ x) = x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    ⊢ Eq (Real.logb b (HPow.hPow b x)) x
  -/
  rw [logb, div_eq_iff, log_rpow b_pos]
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    ⊢ Ne (Real.log b) 0
  -/
  exact log_b_ne_zero b_pos b_ne_one
  /-
    🎉 no goals
  -/


theorem rpow_logb_eq_abs (hx : x ≠ 0) : b ^ logb b x = |x| := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : Ne x 0
    ⊢ Eq (HPow.hPow b (Real.logb b x)) (abs x)
  -/
  apply log_injOn_pos
    /-
      case a
      b x : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      hx : Ne x 0
      ⊢ Membership.mem (Set.Ioi 0) (HPow.hPow b (Real.logb b x))
    -/
  · simp only [Set.mem_Ioi]
    /-
      case a
      b x : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      hx : Ne x 0
      ⊢ LT.lt 0 (HPow.hPow b (Real.logb b x))
    -/
    apply rpow_pos_of_pos b_pos
    /-
      🎉 no goals
    -/
    /-
      case a
      b x : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      hx : Ne x 0
      ⊢ Membership.mem (Set.Ioi 0) (abs x)
    -/
  · simp only [abs_pos, mem_Ioi, Ne, hx, not_false_iff]
    /-
      🎉 no goals
    -/
  /-
    case a
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : Ne x 0
    ⊢ Eq (Real.log (HPow.hPow b (Real.logb b x))) (Real.log (abs x))
  -/
  rw [log_rpow b_pos, logb, log_abs]
  /-
    case a
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Real.log x) (Real.log b)) (Real.log b)) (Real.log x)
  -/
  field_simp [log_b_ne_zero b_pos b_ne_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem rpow_logb (hx : 0 < x) : b ^ logb b x = x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : LT.lt 0 x
    ⊢ Eq (HPow.hPow b (Real.logb b x)) x
  -/
  rw [rpow_logb_eq_abs b_pos b_ne_one hx.ne']
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : LT.lt 0 x
    ⊢ Eq (abs x) x
  -/
  exact abs_of_pos hx
  /-
    🎉 no goals
  -/


theorem rpow_logb_of_neg (hx : x < 0) : b ^ logb b x = -x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : LT.lt x 0
    ⊢ Eq (HPow.hPow b (Real.logb b x)) (Neg.neg x)
  -/
  rw [rpow_logb_eq_abs b_pos b_ne_one (ne_of_lt hx)]
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hx : LT.lt x 0
    ⊢ Eq (abs x) (Neg.neg x)
  -/
  exact abs_of_neg hx
  /-
    🎉 no goals
  -/


theorem logb_eq_iff_rpow_eq (hy : 0 < y) : logb b y = x ↔ b ^ x = y := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    hy : LT.lt 0 y
    ⊢ Iff (Eq (Real.logb b y) x) (Eq (HPow.hPow b x) y)
  -/
  constructor <;> rintro rfl
    /-
      case mp
      b y : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      hy : LT.lt 0 y
      ⊢ Eq (HPow.hPow b (Real.logb b y)) y
    -/
  · exact rpow_logb b_pos b_ne_one hy
    /-
      🎉 no goals
    -/
    /-
      case mpr
      b x : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      hy : LT.lt 0 (HPow.hPow b x)
      ⊢ Eq (Real.logb b (HPow.hPow b x)) x
    -/
  · exact logb_rpow b_pos b_ne_one
    /-
      🎉 no goals
    -/


theorem surjOn_logb : SurjOn (logb b) (Ioi 0) univ := fun x _ =>
  ⟨b ^ x, rpow_pos_of_pos b_pos x, logb_rpow b_pos b_ne_one⟩


theorem logb_surjective : Surjective (logb b) := fun x => ⟨b ^ x, logb_rpow b_pos b_ne_one⟩


@[simp]
theorem range_logb : range (logb b) = univ :=
  (logb_surjective b_pos b_ne_one).range_eq


theorem surjOn_logb' : SurjOn (logb b) (Iio 0) univ := by
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    ⊢ Set.SurjOn (Real.logb b) (Set.Iio 0) Set.univ
  -/
  intro x _
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    x : Real
    a✝ : Membership.mem Set.univ x
    ⊢ Membership.mem (Set.image (Real.logb b) (Set.Iio 0)) x
  -/
  use -b ^ x
  /-
    case h
    b : Real
    b_pos : LT.lt 0 b
    b_ne_one : Ne b 1
    x : Real
    a✝ : Membership.mem Set.univ x
    ⊢ And (Membership.mem (Set.Iio 0) (Neg.neg (HPow.hPow b x))) (Eq (Real.logb b  …
  -/
  constructor
    /-
      case h.left
      b : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      x : Real
      a✝ : Membership.mem Set.univ x
      ⊢ Membership.mem (Set.Iio 0) (Neg.neg (HPow.hPow b x))
    -/
  · simp only [Right.neg_neg_iff, Set.mem_Iio]
    /-
      case h.left
      b : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      x : Real
      a✝ : Membership.mem Set.univ x
      ⊢ LT.lt 0 (HPow.hPow b x)
    -/
    apply rpow_pos_of_pos b_pos
    /-
      🎉 no goals
    -/
    /-
      case h.right
      b : Real
      b_pos : LT.lt 0 b
      b_ne_one : Ne b 1
      x : Real
      a✝ : Membership.mem Set.univ x
      ⊢ Eq (Real.logb b (Neg.neg (HPow.hPow b x))) x
    -/
  · rw [logb_neg_eq_logb, logb_rpow b_pos b_ne_one]
    /-
      🎉 no goals
    -/


                                    /-
                                      b : Real
                                      hb : LT.lt 1 b
                                      ⊢ LT.lt 0 b
                                    -/
private theorem b_pos : 0 < b := by linarith
                                    /-
                                      🎉 no goals
                                    -/

-- Porting note: prime added to avoid clashing with `b_ne_one` further down the file

                                        /-
                                          b : Real
                                          hb : LT.lt 1 b
                                          ⊢ Ne b 1
                                        -/
private theorem b_ne_one' : b ≠ 1 := by linarith
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem logb_le_logb (h : 0 < x) (h₁ : 0 < y) : logb b x ≤ logb b y ↔ x ≤ y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    h : LT.lt 0 x
    h₁ : LT.lt 0 y
    ⊢ Iff (LE.le (Real.logb b x) (Real.logb b y)) (LE.le x y)
  -/
  rw [logb, logb, div_le_div_iff_of_pos_right (log_pos hb), log_le_log_iff h h₁]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem logb_le_logb_of_le (h : 0 < x) (hxy : x ≤ y) : logb b x ≤ logb b y :=
                         /-
                           b x y : Real
                           hb : LT.lt 1 b
                           h : LT.lt 0 x
                           hxy : LE.le x y
                           ⊢ LT.lt 0 y
                         -/
  (logb_le_logb hb h (by linarith)).mpr hxy
                         /-
                           🎉 no goals
                         -/


@[gcongr]
theorem logb_lt_logb (hx : 0 < x) (hxy : x < y) : logb b x < logb b y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b x) (Real.logb b y)
  -/
  rw [logb, logb, div_lt_div_iff_of_pos_right (log_pos hb)]
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    hxy : LT.lt x y
    ⊢ LT.lt (Real.log x) (Real.log y)
  -/
  exact log_lt_log hx hxy
  /-
    🎉 no goals
  -/


@[simp]
theorem logb_lt_logb_iff (hx : 0 < x) (hy : 0 < y) : logb b x < logb b y ↔ x < y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (Real.logb b x) (Real.logb b y)) (LT.lt x y)
  -/
  rw [logb, logb, div_lt_div_iff_of_pos_right (log_pos hb)]
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (Real.log x) (Real.log y)) (LT.lt x y)
  -/
  exact log_lt_log_iff hx hy
  /-
    🎉 no goals
  -/


theorem logb_le_iff_le_rpow (hx : 0 < x) : logb b x ≤ y ↔ x ≤ b ^ y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.logb b x) y) (LE.le x (HPow.hPow b y))
  -/
  rw [← rpow_le_rpow_left_iff hb, rpow_logb (b_pos hb) (b_ne_one' hb) hx]
  /-
    🎉 no goals
  -/


theorem logb_lt_iff_lt_rpow (hx : 0 < x) : logb b x < y ↔ x < b ^ y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b x) y) (LT.lt x (HPow.hPow b y))
  -/
  rw [← rpow_lt_rpow_left_iff hb, rpow_logb (b_pos hb) (b_ne_one' hb) hx]
  /-
    🎉 no goals
  -/


theorem le_logb_iff_rpow_le (hy : 0 < y) : x ≤ logb b y ↔ b ^ x ≤ y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hy : LT.lt 0 y
    ⊢ Iff (LE.le x (Real.logb b y)) (LE.le (HPow.hPow b x) y)
  -/
  rw [← rpow_le_rpow_left_iff hb, rpow_logb (b_pos hb) (b_ne_one' hb) hy]
  /-
    🎉 no goals
  -/


theorem lt_logb_iff_rpow_lt (hy : 0 < y) : x < logb b y ↔ b ^ x < y := by
  /-
    b x y : Real
    hb : LT.lt 1 b
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt x (Real.logb b y)) (LT.lt (HPow.hPow b x) y)
  -/
  rw [← rpow_lt_rpow_left_iff hb, rpow_logb (b_pos hb) (b_ne_one' hb) hy]
  /-
    🎉 no goals
  -/


theorem logb_pos_iff (hx : 0 < x) : 0 < logb b x ↔ 1 < x := by
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt 0 (Real.logb b x)) (LT.lt 1 x)
  -/
  rw [← @logb_one b]
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b 1) (Real.logb b x)) (LT.lt 1 x)
  -/
  rw [logb_lt_logb_iff hb zero_lt_one hx]
  /-
    🎉 no goals
  -/


theorem logb_pos (hx : 1 < x) : 0 < logb b x := by
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 1 x
    ⊢ LT.lt 0 (Real.logb b x)
  -/
  rw [logb_pos_iff hb (lt_trans zero_lt_one hx)]
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 1 x
    ⊢ LT.lt 1 x
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem logb_neg_iff (h : 0 < x) : logb b x < 0 ↔ x < 1 := by
  /-
    b x : Real
    hb : LT.lt 1 b
    h : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b x) 0) (LT.lt x 1)
  -/
  rw [← logb_one]
  /-
    b x : Real
    hb : LT.lt 1 b
    h : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b x) (Real.logb ?m.19641 1)) (LT.lt x 1)
  -/
  exact logb_lt_logb_iff hb h zero_lt_one
  /-
    🎉 no goals
  -/


theorem logb_neg (h0 : 0 < x) (h1 : x < 1) : logb b x < 0 :=
  (logb_neg_iff hb h0).2 h1


theorem logb_nonneg_iff (hx : 0 < x) : 0 ≤ logb b x ↔ 1 ≤ x := by
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LE.le 0 (Real.logb b x)) (LE.le 1 x)
  -/
  rw [← not_lt, logb_neg_iff hb hx, not_lt]
  /-
    🎉 no goals
  -/


theorem logb_nonneg (hx : 1 ≤ x) : 0 ≤ logb b x :=
  (logb_nonneg_iff hb (zero_lt_one.trans_le hx)).2 hx


theorem logb_nonpos_iff (hx : 0 < x) : logb b x ≤ 0 ↔ x ≤ 1 := by
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.logb b x) 0) (LE.le x 1)
  -/
  rw [← not_lt, logb_pos_iff hb hx, not_lt]
  /-
    🎉 no goals
  -/


theorem logb_nonpos_iff' (hx : 0 ≤ x) : logb b x ≤ 0 ↔ x ≤ 1 := by
  /-
    b x : Real
    hb : LT.lt 1 b
    hx : LE.le 0 x
    ⊢ Iff (LE.le (Real.logb b x) 0) (LE.le x 1)
  -/
  rcases hx.eq_or_lt with (rfl | hx)
    /-
      case inl
      b : Real
      hb : LT.lt 1 b
      hx : LE.le 0 0
      ⊢ Iff (LE.le (Real.logb b 0) 0) (LE.le 0 1)
    -/
  · simp [le_refl, zero_le_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    b x : Real
    hb : LT.lt 1 b
    hx✝ : LE.le 0 x
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.logb b x) 0) (LE.le x 1)
  -/
  exact logb_nonpos_iff hb hx
  /-
    🎉 no goals
  -/


theorem logb_nonpos (hx : 0 ≤ x) (h'x : x ≤ 1) : logb b x ≤ 0 :=
  (logb_nonpos_iff' hb hx).2 h'x


theorem strictMonoOn_logb : StrictMonoOn (logb b) (Set.Ioi 0) := fun _ hx _ _ hxy =>
  logb_lt_logb hb hx hxy


theorem strictAntiOn_logb : StrictAntiOn (logb b) (Set.Iio 0) := by
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ StrictAntiOn (Real.logb b) (Set.Iio 0)
  -/
  rintro x (hx : x < 0) y (hy : y < 0) hxy
  /-
    b : Real
    hb : LT.lt 1 b
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b y) (Real.logb b x)
  -/
  rw [← logb_abs y, ← logb_abs x]
  /-
    b : Real
    hb : LT.lt 1 b
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b (abs y)) (Real.logb b (abs x))
  -/
  refine logb_lt_logb hb (abs_pos.2 hy.ne) ?_
  /-
    b : Real
    hb : LT.lt 1 b
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (abs y) (abs x)
  -/
  rwa [abs_of_neg hy, abs_of_neg hx, neg_lt_neg_iff]
  /-
    🎉 no goals
  -/


theorem logb_injOn_pos : Set.InjOn (logb b) (Set.Ioi 0) :=
  (strictMonoOn_logb hb).injOn


theorem eq_one_of_pos_of_logb_eq_zero (h₁ : 0 < x) (h₂ : logb b x = 0) : x = 1 :=
  logb_injOn_pos hb (Set.mem_Ioi.2 h₁) (Set.mem_Ioi.2 zero_lt_one) (h₂.trans Real.logb_one.symm)


theorem logb_ne_zero_of_pos_of_ne_one (hx_pos : 0 < x) (hx : x ≠ 1) : logb b x ≠ 0 :=
  mt (eq_one_of_pos_of_logb_eq_zero hb hx_pos) hx


theorem tendsto_logb_atTop : Tendsto (logb b) atTop atTop :=
  Tendsto.atTop_div_const (log_pos hb) tendsto_log_atTop


                                       /-
                                         b : Real
                                         b_lt_one : LT.lt b 1
                                         ⊢ Ne b 1
                                       -/
private theorem b_ne_one : b ≠ 1 := by linarith
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem logb_le_logb_of_base_lt_one (h : 0 < x) (h₁ : 0 < y) : logb b x ≤ logb b y ↔ y ≤ x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    h : LT.lt 0 x
    h₁ : LT.lt 0 y
    ⊢ Iff (LE.le (Real.logb b x) (Real.logb b y)) (LE.le y x)
  -/
  rw [logb, logb, div_le_div_right_of_neg (log_neg b_pos b_lt_one), log_le_log_iff h₁ h]
  /-
    🎉 no goals
  -/


theorem logb_lt_logb_of_base_lt_one (hx : 0 < x) (hxy : x < y) : logb b y < logb b x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b y) (Real.logb b x)
  -/
  rw [logb, logb, div_lt_div_right_of_neg (log_neg b_pos b_lt_one)]
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hxy : LT.lt x y
    ⊢ LT.lt (Real.log x) (Real.log y)
  -/
  exact log_lt_log hx hxy
  /-
    🎉 no goals
  -/


@[simp]
theorem logb_lt_logb_iff_of_base_lt_one (hx : 0 < x) (hy : 0 < y) :
    logb b x < logb b y ↔ y < x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (Real.logb b x) (Real.logb b y)) (LT.lt y x)
  -/
  rw [logb, logb, div_lt_div_right_of_neg (log_neg b_pos b_lt_one)]
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (Real.log y) (Real.log x)) (LT.lt y x)
  -/
  exact log_lt_log_iff hy hx
  /-
    🎉 no goals
  -/


theorem logb_le_iff_le_rpow_of_base_lt_one (hx : 0 < x) : logb b x ≤ y ↔ b ^ y ≤ x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.logb b x) y) (LE.le (HPow.hPow b y) x)
  -/
  rw [← rpow_le_rpow_left_iff_of_base_lt_one b_pos b_lt_one, rpow_logb b_pos (b_ne_one b_lt_one) hx]
  /-
    🎉 no goals
  -/


theorem logb_lt_iff_lt_rpow_of_base_lt_one (hx : 0 < x) : logb b x < y ↔ b ^ y < x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b x) y) (LT.lt (HPow.hPow b y) x)
  -/
  rw [← rpow_lt_rpow_left_iff_of_base_lt_one b_pos b_lt_one, rpow_logb b_pos (b_ne_one b_lt_one) hx]
  /-
    🎉 no goals
  -/


theorem le_logb_iff_rpow_le_of_base_lt_one (hy : 0 < y) : x ≤ logb b y ↔ y ≤ b ^ x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hy : LT.lt 0 y
    ⊢ Iff (LE.le x (Real.logb b y)) (LE.le y (HPow.hPow b x))
  -/
  rw [← rpow_le_rpow_left_iff_of_base_lt_one b_pos b_lt_one, rpow_logb b_pos (b_ne_one b_lt_one) hy]
  /-
    🎉 no goals
  -/


theorem lt_logb_iff_rpow_lt_of_base_lt_one (hy : 0 < y) : x < logb b y ↔ y < b ^ x := by
  /-
    b x y : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt x (Real.logb b y)) (LT.lt y (HPow.hPow b x))
  -/
  rw [← rpow_lt_rpow_left_iff_of_base_lt_one b_pos b_lt_one, rpow_logb b_pos (b_ne_one b_lt_one) hy]
  /-
    🎉 no goals
  -/


theorem logb_pos_iff_of_base_lt_one (hx : 0 < x) : 0 < logb b x ↔ x < 1 := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt 0 (Real.logb b x)) (LT.lt x 1)
  -/
  rw [← @logb_one b, logb_lt_logb_iff_of_base_lt_one b_pos b_lt_one zero_lt_one hx]
  /-
    🎉 no goals
  -/


theorem logb_pos_of_base_lt_one (hx : 0 < x) (hx' : x < 1) : 0 < logb b x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hx' : LT.lt x 1
    ⊢ LT.lt 0 (Real.logb b x)
  -/
  rw [logb_pos_iff_of_base_lt_one b_pos b_lt_one hx]
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hx' : LT.lt x 1
    ⊢ LT.lt x 1
  -/
  exact hx'
  /-
    🎉 no goals
  -/


theorem logb_neg_iff_of_base_lt_one (h : 0 < x) : logb b x < 0 ↔ 1 < x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    h : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.logb b x) 0) (LT.lt 1 x)
  -/
  rw [← @logb_one b, logb_lt_logb_iff_of_base_lt_one b_pos b_lt_one h zero_lt_one]
  /-
    🎉 no goals
  -/


theorem logb_neg_of_base_lt_one (h1 : 1 < x) : logb b x < 0 :=
  (logb_neg_iff_of_base_lt_one b_pos b_lt_one (lt_trans zero_lt_one h1)).2 h1


theorem logb_nonneg_iff_of_base_lt_one (hx : 0 < x) : 0 ≤ logb b x ↔ x ≤ 1 := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    ⊢ Iff (LE.le 0 (Real.logb b x)) (LE.le x 1)
  -/
  rw [← not_lt, logb_neg_iff_of_base_lt_one b_pos b_lt_one hx, not_lt]
  /-
    🎉 no goals
  -/


theorem logb_nonneg_of_base_lt_one (hx : 0 < x) (hx' : x ≤ 1) : 0 ≤ logb b x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hx' : LE.le x 1
    ⊢ LE.le 0 (Real.logb b x)
  -/
  rw [logb_nonneg_iff_of_base_lt_one b_pos b_lt_one hx]
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    hx' : LE.le x 1
    ⊢ LE.le x 1
  -/
  exact hx'
  /-
    🎉 no goals
  -/


theorem logb_nonpos_iff_of_base_lt_one (hx : 0 < x) : logb b x ≤ 0 ↔ 1 ≤ x := by
  /-
    b x : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.logb b x) 0) (LE.le 1 x)
  -/
  rw [← not_lt, logb_pos_iff_of_base_lt_one b_pos b_lt_one hx, not_lt]
  /-
    🎉 no goals
  -/


theorem strictAntiOn_logb_of_base_lt_one : StrictAntiOn (logb b) (Set.Ioi 0) := fun _ hx _ _ hxy =>
  logb_lt_logb_of_base_lt_one b_pos b_lt_one hx hxy


theorem strictMonoOn_logb_of_base_lt_one : StrictMonoOn (logb b) (Set.Iio 0) := by
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    ⊢ StrictMonoOn (Real.logb b) (Set.Iio 0)
  -/
  rintro x (hx : x < 0) y (hy : y < 0) hxy
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b x) (Real.logb b y)
  -/
  rw [← logb_abs y, ← logb_abs x]
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.logb b (abs x)) (Real.logb b (abs y))
  -/
  refine logb_lt_logb_of_base_lt_one b_pos b_lt_one (abs_pos.2 hy.ne) ?_
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (abs y) (abs x)
  -/
  rwa [abs_of_neg hy, abs_of_neg hx, neg_lt_neg_iff]
  /-
    🎉 no goals
  -/


theorem logb_injOn_pos_of_base_lt_one : Set.InjOn (logb b) (Set.Ioi 0) :=
  (strictAntiOn_logb_of_base_lt_one b_pos b_lt_one).injOn


theorem eq_one_of_pos_of_logb_eq_zero_of_base_lt_one (h₁ : 0 < x) (h₂ : logb b x = 0) : x = 1 :=
  logb_injOn_pos_of_base_lt_one b_pos b_lt_one (Set.mem_Ioi.2 h₁) (Set.mem_Ioi.2 zero_lt_one)
    (h₂.trans Real.logb_one.symm)


theorem logb_ne_zero_of_pos_of_ne_one_of_base_lt_one (hx_pos : 0 < x) (hx : x ≠ 1) : logb b x ≠ 0 :=
  mt (eq_one_of_pos_of_logb_eq_zero_of_base_lt_one b_pos b_lt_one hx_pos) hx


theorem tendsto_logb_atTop_of_base_lt_one : Tendsto (logb b) atTop atBot := by
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    ⊢ Filter.Tendsto (Real.logb b) Filter.atTop Filter.atBot
  -/
  rw [tendsto_atTop_atBot]
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    ⊢ ∀ (b_1 : Real), Exists fun i => ∀ (a : Real), LE.le i a → LE.le (Real.logb b …
  -/
  intro e
  /-
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    e : Real
    ⊢ Exists fun i => ∀ (a : Real), LE.le i a → LE.le (Real.logb b a) e
  -/
  use 1 ⊔ b ^ e
  /-
    case h
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    e : Real
    ⊢ ∀ (a : Real), LE.le (Max.max 1 (HPow.hPow b e)) a → LE.le (Real.logb b a) e
  -/
  intro a
  /-
    case h
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    e a : Real
    ⊢ LE.le (Max.max 1 (HPow.hPow b e)) a → LE.le (Real.logb b a) e
  -/
  simp only [and_imp, sup_le_iff]
  /-
    case h
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    e a : Real
    ⊢ LE.le 1 a → LE.le (HPow.hPow b e) a → LE.le (Real.logb b a) e
  -/
  intro ha
  /-
    case h
    b : Real
    b_pos : LT.lt 0 b
    b_lt_one : LT.lt b 1
    e a : Real
    ha : LE.le 1 a
    ⊢ LE.le (HPow.hPow b e) a → LE.le (Real.logb b a) e
  -/
  rw [logb_le_iff_le_rpow_of_base_lt_one b_pos b_lt_one]
    /-
      case h
      b : Real
      b_pos : LT.lt 0 b
      b_lt_one : LT.lt b 1
      e a : Real
      ha : LE.le 1 a
      ⊢ LE.le (HPow.hPow b e) a → LE.le (HPow.hPow b e) a
    -/
  · tauto
    /-
      🎉 no goals
    -/
    /-
      case h
      b : Real
      b_pos : LT.lt 0 b
      b_lt_one : LT.lt b 1
      e a : Real
      ha : LE.le 1 a
      ⊢ LT.lt 0 a
    -/
  · exact lt_of_lt_of_le zero_lt_one ha
    /-
      🎉 no goals
    -/


theorem floor_logb_natCast {b : ℕ} {r : ℝ} (hr : 0 ≤ r) :
    ⌊logb b r⌋ = Int.log b r := by
  /-
    b : Nat
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
  -/
  obtain rfl | hr := hr.eq_or_lt
    /-
      case inl
      b : Nat
      hr : LE.le 0 0
      ⊢ Eq (Int.floor (Real.logb (↑b) 0)) (Int.log b 0)
    -/
  · rw [logb_zero, Int.log_zero_right, Int.floor_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    b : Nat
    r : Real
    hr✝ : LE.le 0 r
    hr : LT.lt 0 r
    ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
  -/
  by_cases hb : 1 < b
    /-
      case pos
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : LT.lt 1 b
      ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
    -/
  · have hb1' : 1 < (b : ℝ) := Nat.one_lt_cast.mpr hb
    /-
      case pos
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : LT.lt 1 b
      hb1' : LT.lt 1 ↑b
      ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
    -/
    apply le_antisymm
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (Int.floor (Real.logb (↑b) r)) (Int.log b r)
      -/
    · rw [← Int.zpow_le_iff_le_log hb hr, ← rpow_intCast b]
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (HPow.hPow ↑b ↑(Int.floor (Real.logb (↑b) r))) r
      -/
      refine le_of_le_of_eq ?_ (rpow_logb (zero_lt_one.trans hb1') hb1'.ne' hr)
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (HPow.hPow ↑b ↑(Int.floor (Real.logb (↑b) r))) (HPow.hPow (↑b) (Real.l …
      -/
      exact rpow_le_rpow_of_exponent_le hb1'.le (Int.floor_le _)
      /-
        🎉 no goals
      -/
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (Int.log b r) (Int.floor (Real.logb (↑b) r))
      -/
    · rw [Int.le_floor, le_logb_iff_rpow_le hb1' hr, rpow_intCast]
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (HPow.hPow (↑b) (Int.log b r)) r
      -/
      exact Int.zpow_log_le_self hb hr
      /-
        🎉 no goals
      -/
    /-
      case neg
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : Not (LT.lt 1 b)
      ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
    -/
  · rw [Nat.one_lt_iff_ne_zero_and_ne_one, ← or_iff_not_and_not] at hb
    /-
      case neg
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : Or (Eq b 0) (Eq b 1)
      ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
    -/
    cases hb
      /-
        case neg.inl
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        h✝ : Eq b 0
        ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
      -/
    · simp_all only [CharP.cast_eq_zero, logb_zero_left, Int.floor_zero, Int.log_zero_left]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        h✝ : Eq b 1
        ⊢ Eq (Int.floor (Real.logb (↑b) r)) (Int.log b r)
      -/
    · simp_all only [Nat.cast_one, logb_one_left, Int.floor_zero, Int.log_one_left]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-04-17")]
alias floor_logb_nat_cast := floor_logb_natCast


theorem ceil_logb_natCast {b : ℕ} {r : ℝ} (hr : 0 ≤ r) :
    ⌈logb b r⌉ = Int.clog b r := by
  /-
    b : Nat
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
  -/
  obtain rfl | hr := hr.eq_or_lt
    /-
      case inl
      b : Nat
      hr : LE.le 0 0
      ⊢ Eq (Int.ceil (Real.logb (↑b) 0)) (Int.clog b 0)
    -/
  · rw [logb_zero, Int.clog_zero_right, Int.ceil_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    b : Nat
    r : Real
    hr✝ : LE.le 0 r
    hr : LT.lt 0 r
    ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
  -/
  by_cases hb : 1 < b
    /-
      case pos
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : LT.lt 1 b
      ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
    -/
  · have hb1' : 1 < (b : ℝ) := Nat.one_lt_cast.mpr hb
    /-
      case pos
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : LT.lt 1 b
      hb1' : LT.lt 1 ↑b
      ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
    -/
    apply le_antisymm
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
      -/
    · rw [Int.ceil_le, logb_le_iff_le_rpow hb1' hr, rpow_intCast]
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le r (HPow.hPow (↑b) (Int.clog b r))
      -/
      exact Int.self_le_zpow_clog hb r
      /-
        🎉 no goals
      -/
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (Int.clog b r) (Int.ceil (Real.logb (↑b) r))
      -/
    · rw [← Int.le_zpow_iff_clog_le hb hr, ← rpow_intCast b]
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le r (HPow.hPow ↑b ↑(Int.ceil (Real.logb (↑b) r)))
      -/
      refine (rpow_logb (zero_lt_one.trans hb1') hb1'.ne' hr).symm.trans_le ?_
      /-
        case pos.a
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        hb : LT.lt 1 b
        hb1' : LT.lt 1 ↑b
        ⊢ LE.le (HPow.hPow (↑b) (Real.logb (↑b) r)) (HPow.hPow ↑b ↑(Int.ceil (Real.log …
      -/
      exact rpow_le_rpow_of_exponent_le hb1'.le (Int.le_ceil _)
      /-
        🎉 no goals
      -/
    /-
      case neg
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : Not (LT.lt 1 b)
      ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
    -/
  · rw [Nat.one_lt_iff_ne_zero_and_ne_one, ← or_iff_not_and_not] at hb
    /-
      case neg
      b : Nat
      r : Real
      hr✝ : LE.le 0 r
      hr : LT.lt 0 r
      hb : Or (Eq b 0) (Eq b 1)
      ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
    -/
    cases hb
      /-
        case neg.inl
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        h✝ : Eq b 0
        ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
      -/
    · simp_all only [CharP.cast_eq_zero, logb_zero_left, Int.ceil_zero, Int.clog_zero_left]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        b : Nat
        r : Real
        hr✝ : LE.le 0 r
        hr : LT.lt 0 r
        h✝ : Eq b 1
        ⊢ Eq (Int.ceil (Real.logb (↑b) r)) (Int.clog b r)
      -/
    · simp_all only [Nat.cast_one, logb_one_left, Int.ceil_zero, Int.clog_one_left]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-04-17")]
alias ceil_logb_nat_cast := ceil_logb_natCast


lemma natLog_le_logb (a b : ℕ) : Nat.log b a ≤ Real.logb b a := by
  /-
    a b : Nat
    ⊢ LE.le (↑(Nat.log b a)) (Real.logb ↑b ↑a)
  -/
  apply le_trans _ (Int.floor_le ((b : ℝ).logb a))
  /-
    a b : Nat
    ⊢ LE.le ↑(Nat.log b a) ↑(Int.floor (Real.logb ↑b ↑a))
  -/
  rw [Real.floor_logb_natCast (Nat.cast_nonneg a), Int.log_natCast, Int.cast_natCast]
  /-
    🎉 no goals
  -/


@[simp]
theorem logb_eq_zero : logb b x = 0 ↔ b = 0 ∨ b = 1 ∨ b = -1 ∨ x = 0 ∨ x = 1 ∨ x = -1 := by
  /-
    b x : Real
    ⊢ Iff (Eq (Real.logb b x) 0) (Or (Eq b 0) (Or (Eq b 1) (Or (Eq b (-1)) (Or (Eq …
  -/
  simp_rw [logb, div_eq_zero_iff, log_eq_zero]
  /-
    b x : Real
    ⊢ Iff (Or (Or (Eq x 0) (Or (Eq x 1) (Eq x (-1)))) (Or (Eq b 0) (Or (Eq b 1) (E …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem tendsto_logb_nhdsWithin_zero (hb : 1 < b) :
    Tendsto (logb b) (𝓝[≠] 0) atBot :=
  tendsto_log_nhdsWithin_zero.atBot_div_const (log_pos hb)


theorem tendsto_logb_nhdsWithin_zero_of_base_lt_one (hb₀ : 0 < b) (hb : b < 1) :
    Tendsto (logb b) (𝓝[≠] 0) atTop :=
  tendsto_log_nhdsWithin_zero.atBot_mul_const_of_neg (inv_lt_zero.2 (log_neg hb₀ hb))


lemma tendsto_logb_nhdsWithin_zero_right (hb : 1 < b) : Tendsto (logb b) (𝓝[>] 0) atBot :=
  tendsto_log_nhdsWithin_zero_right.atBot_div_const (log_pos hb)


lemma tendsto_logb_nhdsWithin_zero_right_of_base_lt_one (hb₀ : 0 < b) (hb : b < 1) :
    Tendsto (logb b) (𝓝[>] 0) atTop :=
  tendsto_log_nhdsWithin_zero_right.atBot_mul_const_of_neg (inv_lt_zero.2 (log_neg hb₀ hb))


theorem continuousOn_logb : ContinuousOn (logb b) {0}ᶜ := continuousOn_log.div_const _


/-- The real logarithm base b is continuous as a function from nonzero reals. -/
@[fun_prop]
theorem continuous_logb : Continuous fun x : { x : ℝ // x ≠ 0 } => logb b x :=
  continuous_log.div_const _


/-- The real logarithm base b is continuous as a function from positive reals. -/
@[fun_prop]
theorem continuous_logb' : Continuous fun x : { x : ℝ // 0 < x } => logb b x :=
  continuous_log'.div_const _


theorem continuousAt_logb (hx : x ≠ 0) : ContinuousAt (logb b) x :=
  (continuousAt_log hx).div_const _


@[simp]
theorem continuousAt_logb_iff (hb₀ : 0 < b) (hb : b ≠ 1) : ContinuousAt (logb b) x ↔ x ≠ 0 := by
  /-
    b x : Real
    hb₀ : LT.lt 0 b
    hb : Ne b 1
    ⊢ Iff (ContinuousAt (Real.logb b) x) (Ne x 0)
  -/
  refine ⟨?_, continuousAt_logb⟩
  /-
    b x : Real
    hb₀ : LT.lt 0 b
    hb : Ne b 1
    ⊢ ContinuousAt (Real.logb b) x → Ne x 0
  -/
  rintro h rfl
  cases lt_or_gt_of_ne hb with
  | inl hb₁ =>
      exact not_tendsto_nhds_of_tendsto_atTop (tendsto_logb_nhdsWithin_zero_of_base_lt_one hb₀ hb₁)
        _ (h.tendsto.mono_left inf_le_left)
  | inr hb₁ =>
      exact not_tendsto_nhds_of_tendsto_atBot (tendsto_logb_nhdsWithin_zero hb₁)
        _ (h.tendsto.mono_left inf_le_left)


theorem logb_prod {α : Type*} (s : Finset α) (f : α → ℝ) (hf : ∀ x ∈ s, f x ≠ 0) :
    logb b (∏ i ∈ s, f i) = ∑ i ∈ s, logb b (f i) := by
  classical
    induction' s using Finset.induction_on with a s ha ih
    · simp
    simp only [Finset.mem_insert, forall_eq_or_imp] at hf
    simp [ha, ih hf.2, logb_mul hf.1 (Finset.prod_ne_zero_iff.2 hf.2)]


protected theorem _root_.Finsupp.logb_prod {α β : Type*} [Zero β] (f : α →₀ β) (g : α → β → ℝ)
    (hg : ∀ a, g a (f a) = 0 → f a = 0) : logb b (f.prod g) = f.sum fun a c ↦ logb b (g a c) :=
  logb_prod _ _ fun _x hx h₀ ↦ Finsupp.mem_support_iff.1 hx <| hg _ h₀


theorem logb_nat_eq_sum_factorization (n : ℕ) :
    logb b n = n.factorization.sum fun p t => t * logb b p := by
  /-
    b : Real
    n : Nat
    ⊢ Eq (Real.logb b ↑n) (n.factorization.sum fun p t => HMul.hMul (↑t) (Real.log …
  -/
  simp only [logb, mul_div_assoc', log_nat_eq_sum_factorization n, Finsupp.sum, Finset.sum_div]
  /-
    🎉 no goals
  -/

-- TODO add other limits and continuous API lemmas analogous to those in Log.lean


/-- Induction principle for intervals of real numbers: if a proposition `P` is true
on `[x₀, r * x₀)` and if `P` for `[x₀, r^n * x₀)` implies `P` for `[r^n * x₀, r^(n+1) * x₀)`,
then `P` is true for all `x ≥ x₀`. -/
lemma Real.induction_Ico_mul {P : ℝ → Prop} (x₀ r : ℝ) (hr : 1 < r) (hx₀ : 0 < x₀)
    (base : ∀ x ∈ Set.Ico x₀ (r * x₀), P x)
    (step : ∀ n : ℕ, n ≥ 1 → (∀ z ∈ Set.Ico x₀ (r ^ n * x₀), P z) →
      (∀ z ∈ Set.Ico (r ^ n * x₀) (r ^ (n+1) * x₀), P z)) :
    ∀ x ≥ x₀, P x := by
  suffices ∀ n : ℕ, ∀ x ∈ Set.Ico x₀ (r ^ (n + 1) * x₀), P x by
    intro x hx
    have hx' : 0 < x / x₀ := div_pos (hx₀.trans_le hx) hx₀
    refine this ⌊logb r (x / x₀)⌋₊ x ?_
    rw [mem_Ico, ← div_lt_iff₀ hx₀, ← rpow_natCast, ← logb_lt_iff_lt_rpow hr hx', Nat.cast_add,
      Nat.cast_one]
    exact ⟨hx, Nat.lt_floor_add_one _⟩
  /-
    P : Real → Prop
    x₀ r : Real
    hr : LT.lt 1 r
    hx₀ : LT.lt 0 x₀
    base : ∀ (x : Real), Membership.mem (Set.Ico x₀ (HMul.hMul r x₀)) x → P x
    step : ∀ (n : Nat), GE.ge n 1 → (∀ (z : Real), Membership.mem (Set.Ico x₀ (HMu …
    ⊢ ∀ (n : Nat) (x : Real), Membership.mem (Set.Ico x₀ (HMul.hMul (HPow.hPow r ( …
  -/
  intro n
  induction n with
  | zero => simpa using base
  | succ n ih =>
    exact fun x hx => (Ico_subset_Ico_union_Ico hx).elim (ih x) (step (n + 1) (by simp) ih _)


