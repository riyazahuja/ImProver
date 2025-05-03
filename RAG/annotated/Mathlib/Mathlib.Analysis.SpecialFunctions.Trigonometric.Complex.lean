theorem cos_eq_zero_iff {θ : ℂ} : cos θ = 0 ↔ ∃ k : ℤ, θ = (2 * k + 1) * π / 2 := by
  have h : (exp (θ * I) + exp (-θ * I)) / 2 = 0 ↔ exp (2 * θ * I) = -1 := by
    rw [@div_eq_iff _ _ (exp (θ * I) + exp (-θ * I)) 2 0 two_ne_zero, zero_mul,
      add_eq_zero_iff_eq_neg, neg_eq_neg_one_mul, ← div_eq_iff (exp_ne_zero _), ← exp_sub]
    ring_nf
  /-
    θ : Complex
    h : Iff (Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul θ Complex.I)) (Compl …
    ⊢ Iff (Eq (Complex.cos θ) 0) (Exists fun k => Eq θ (HDiv.hDiv (HMul.hMul (HAdd …
  -/
  rw [cos, h, ← exp_pi_mul_I, exp_eq_exp_iff_exists_int, mul_right_comm]
  /-
    θ : Complex
    h : Iff (Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul θ Complex.I)) (Compl …
    ⊢ Iff (Exists fun n => Eq (HMul.hMul (HMul.hMul 2 Complex.I) θ) (HAdd.hAdd (HM …
  -/
  refine exists_congr fun x => ?_
  /-
    θ : Complex
    h : Iff (Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul θ Complex.I)) (Compl …
    x : Int
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul 2 Complex.I) θ) (HAdd.hAdd (HMul.hMul (↑Real.p …
  -/
  refine (iff_of_eq <| congr_arg _ ?_).trans (mul_right_inj' <| mul_ne_zero two_ne_zero I_ne_zero)
  /-
    θ : Complex
    h : Iff (Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul θ Complex.I)) (Compl …
    x : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑Real.pi) Complex.I) (HMul.hMul (↑x) (HMul.hMul (H …
  -/
  field_simp; ring
              /-
                🎉 no goals
              -/


theorem cos_ne_zero_iff {θ : ℂ} : cos θ ≠ 0 ↔ ∀ k : ℤ, θ ≠ (2 * k + 1) * π / 2 := by
  /-
    θ : Complex
    ⊢ Iff (Ne (Complex.cos θ) 0) (∀ (k : Int), Ne θ (HDiv.hDiv (HMul.hMul (HAdd.hA …
  -/
  rw [← not_exists, not_iff_not, cos_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem sin_eq_zero_iff {θ : ℂ} : sin θ = 0 ↔ ∃ k : ℤ, θ = k * π := by
  /-
    θ : Complex
    ⊢ Iff (Eq (Complex.sin θ) 0) (Exists fun k => Eq θ (HMul.hMul ↑k ↑Real.pi))
  -/
  rw [← Complex.cos_sub_pi_div_two, cos_eq_zero_iff]
  /-
    θ : Complex
    ⊢ Iff (Exists fun k => Eq (HSub.hSub θ (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv (H …
  -/
  constructor
    /-
      case mp
      θ : Complex
      ⊢ (Exists fun k => Eq (HSub.hSub θ (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv (HMul. …
    -/
  · rintro ⟨k, hk⟩
    /-
      case mp.intro
      θ : Complex
      k : Int
      hk : Eq (HSub.hSub θ (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv (HMul.hMul (HAdd.hAd …
      ⊢ Exists fun k => Eq θ (HMul.hMul ↑k ↑Real.pi)
    -/
    use k + 1
    /-
      case h
      θ : Complex
      k : Int
      hk : Eq (HSub.hSub θ (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv (HMul.hMul (HAdd.hAd …
      ⊢ Eq θ (HMul.hMul ↑(HAdd.hAdd k 1) ↑Real.pi)
    -/
    field_simp [eq_add_of_sub_eq hk]
    /-
      case h
      θ : Complex
      k : Int
      hk : Eq (HSub.hSub θ (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv (HMul.hMul (HAdd.hAd …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) ↑Real.pi) ↑Real.pi)  …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case mpr
      θ : Complex
      ⊢ (Exists fun k => Eq θ (HMul.hMul ↑k ↑Real.pi)) → Exists fun k => Eq (HSub.hS …
    -/
  · rintro ⟨k, rfl⟩
    /-
      case mpr.intro
      k : Int
      ⊢ Exists fun k_1 => Eq (HSub.hSub (HMul.hMul ↑k ↑Real.pi) (HDiv.hDiv (↑Real.pi …
    -/
    use k - 1
    /-
      case h
      k : Int
      ⊢ Eq (HSub.hSub (HMul.hMul ↑k ↑Real.pi) (HDiv.hDiv (↑Real.pi) 2)) (HDiv.hDiv ( …
    -/
    field_simp
    /-
      case h
      k : Int
      ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul ↑k ↑Real.pi) 2) ↑Real.pi) (HMul.hMul (HA …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem sin_ne_zero_iff {θ : ℂ} : sin θ ≠ 0 ↔ ∀ k : ℤ, θ ≠ k * π := by
  /-
    θ : Complex
    ⊢ Iff (Ne (Complex.sin θ) 0) (∀ (k : Int), Ne θ (HMul.hMul ↑k ↑Real.pi))
  -/
  rw [← not_exists, not_iff_not, sin_eq_zero_iff]
  /-
    🎉 no goals
  -/


/-- The tangent of a complex number is equal to zero
iff this number is equal to `k * π / 2` for an integer `k`.

Note that this lemma takes into account that we use zero as the junk value for division by zero.
See also `Complex.tan_eq_zero_iff'`. -/
theorem tan_eq_zero_iff {θ : ℂ} : tan θ = 0 ↔ ∃ k : ℤ, k * π / 2 = θ := by
  rw [tan, div_eq_zero_iff, ← mul_eq_zero, ← mul_right_inj' two_ne_zero, mul_zero,
    ← mul_assoc, ← sin_two_mul, sin_eq_zero_iff]
  /-
    θ : Complex
    ⊢ Iff (Exists fun k => Eq (HMul.hMul 2 θ) (HMul.hMul ↑k ↑Real.pi)) (Exists fun …
  -/
  field_simp [mul_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem tan_ne_zero_iff {θ : ℂ} : tan θ ≠ 0 ↔ ∀ k : ℤ, (k * π / 2 : ℂ) ≠ θ := by
  /-
    θ : Complex
    ⊢ Iff (Ne (Complex.tan θ) 0) (∀ (k : Int), Ne (HDiv.hDiv (HMul.hMul ↑k ↑Real.p …
  -/
  rw [← not_exists, not_iff_not, tan_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem tan_int_mul_pi_div_two (n : ℤ) : tan (n * π / 2) = 0 :=
                          /-
                            n : Int
                            ⊢ Exists fun k => Eq (HDiv.hDiv (HMul.hMul ↑k ↑Real.pi) 2) (HDiv.hDiv (HMul.hM …
                          -/
  tan_eq_zero_iff.mpr (by use n)
                          /-
                            🎉 no goals
                          -/


/-- If the tangent of a complex number is well-defined,
then it is equal to zero iff the number is equal to `k * π` for an integer `k`.

See also `Complex.tan_eq_zero_iff` for a version that takes into account junk values of `θ`. -/
theorem tan_eq_zero_iff' {θ : ℂ} (hθ : cos θ ≠ 0) : tan θ = 0 ↔ ∃ k : ℤ, k * π = θ := by
  /-
    θ : Complex
    hθ : Ne (Complex.cos θ) 0
    ⊢ Iff (Eq (Complex.tan θ) 0) (Exists fun k => Eq (HMul.hMul ↑k ↑Real.pi) θ)
  -/
  simp only [tan, hθ, div_eq_zero_iff, sin_eq_zero_iff]; simp [eq_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem cos_eq_cos_iff {x y : ℂ} : cos x = cos y ↔ ∃ k : ℤ, y = 2 * k * π + x ∨ y = 2 * k * π - x :=
  calc
    cos x = cos y ↔ cos x - cos y = 0 := sub_eq_zero.symm
                                                             /-
                                                               x y : Complex
                                                               ⊢ Iff (Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) 0) (Eq (HMul.hMul (HMul. …
                                                             -/
    _ ↔ -2 * sin ((x + y) / 2) * sin ((x - y) / 2) = 0 := by rw [cos_sub_cos]
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                            /-
                                                              x y : Complex
                                                              ⊢ Iff (Eq (HMul.hMul (HMul.hMul (-2) (Complex.sin (HDiv.hDiv (HAdd.hAdd x y) 2 …
                                                            -/
    _ ↔ sin ((x + y) / 2) = 0 ∨ sin ((x - y) / 2) = 0 := by simp [(by norm_num : (2 : ℂ) ≠ 0)]
                                                            /-
                                                              🎉 no goals
                                                            -/
    _ ↔ sin ((x - y) / 2) = 0 ∨ sin ((x + y) / 2) = 0 := or_comm
    _ ↔ (∃ k : ℤ, y = 2 * k * π + x) ∨ ∃ k : ℤ, y = 2 * k * π - x := by
      /-
        x y : Complex
        ⊢ Iff (Or (Eq (Complex.sin (HDiv.hDiv (HSub.hSub x y) 2)) 0) (Eq (Complex.sin  …
      -/
      apply or_congr <;>
        field_simp [sin_eq_zero_iff, (by norm_num : -(2 : ℂ) ≠ 0), eq_sub_iff_add_eq',
          sub_eq_iff_eq_add, mul_comm (2 : ℂ), mul_right_comm _ (2 : ℂ)]
      /-
        case h₁
        x y : Complex
        ⊢ Iff (Exists fun k => Eq x (HAdd.hAdd (HMul.hMul (HMul.hMul ↑k ↑Real.pi) 2) y …
      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
      constructor <;> · rintro ⟨k, rfl⟩; use -k; simp
                                                 /-
                                                   🎉 no goals
                                                 -/
    _ ↔ ∃ k : ℤ, y = 2 * k * π + x ∨ y = 2 * k * π - x := exists_or.symm


theorem sin_eq_sin_iff {x y : ℂ} :
    sin x = sin y ↔ ∃ k : ℤ, y = 2 * k * π + x ∨ y = (2 * k + 1) * π - x := by
  /-
    x y : Complex
    ⊢ Iff (Eq (Complex.sin x) (Complex.sin y)) (Exists fun k => Or (Eq y (HAdd.hAd …
  -/
  simp only [← Complex.cos_sub_pi_div_two, cos_eq_cos_iff, sub_eq_iff_eq_add]
  /-
    x y : Complex
    ⊢ Iff (Exists fun k => Or (Eq y (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul 2  …
  -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
  refine exists_congr fun k => or_congr ?_ ?_ <;> refine Eq.congr rfl ?_ <;> field_simp <;> ring
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem cos_eq_one_iff {x : ℂ} : cos x = 1 ↔ ∃ k : ℤ, k * (2 * π) = x := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.cos x) 1) (Exists fun k => Eq (HMul.hMul (↑k) (HMul.hMul 2  …
  -/
  rw [← cos_zero, eq_comm, cos_eq_cos_iff]
  /-
    x : Complex
    ⊢ Iff (Exists fun k => Or (Eq x (HAdd.hAdd (HMul.hMul (HMul.hMul 2 ↑k) ↑Real.p …
  -/
  simp [mul_assoc, mul_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem cos_eq_neg_one_iff {x : ℂ} : cos x = -1 ↔ ∃ k : ℤ, π + k * (2 * π) = x := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.cos x) (-1)) (Exists fun k => Eq (HAdd.hAdd (↑Real.pi) (HMu …
  -/
  rw [← neg_eq_iff_eq_neg, ← cos_sub_pi, cos_eq_one_iff]
  /-
    x : Complex
    ⊢ Iff (Exists fun k => Eq (HMul.hMul (↑k) (HMul.hMul 2 ↑Real.pi)) (HSub.hSub x …
  -/
  simp only [eq_sub_iff_add_eq']
  /-
    🎉 no goals
  -/


theorem sin_eq_one_iff {x : ℂ} : sin x = 1 ↔ ∃ k : ℤ, π / 2 + k * (2 * π) = x := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.sin x) 1) (Exists fun k => Eq (HAdd.hAdd (HDiv.hDiv (↑Real. …
  -/
  rw [← cos_sub_pi_div_two, cos_eq_one_iff]
  /-
    x : Complex
    ⊢ Iff (Exists fun k => Eq (HMul.hMul (↑k) (HMul.hMul 2 ↑Real.pi)) (HSub.hSub x …
  -/
  simp only [eq_sub_iff_add_eq']
  /-
    🎉 no goals
  -/


theorem sin_eq_neg_one_iff {x : ℂ} : sin x = -1 ↔ ∃ k : ℤ, -(π / 2) + k * (2 * π) = x := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.sin x) (-1)) (Exists fun k => Eq (HAdd.hAdd (Neg.neg (HDiv. …
  -/
  rw [← neg_eq_iff_eq_neg, ← cos_add_pi_div_two, cos_eq_one_iff]
  /-
    x : Complex
    ⊢ Iff (Exists fun k => Eq (HMul.hMul (↑k) (HMul.hMul 2 ↑Real.pi)) (HAdd.hAdd x …
  -/
  simp only [← sub_eq_neg_add, sub_eq_iff_eq_add]
  /-
    🎉 no goals
  -/


theorem tan_add {x y : ℂ}
    (h : ((∀ k : ℤ, x ≠ (2 * k + 1) * π / 2) ∧ ∀ l : ℤ, y ≠ (2 * l + 1) * π / 2) ∨
      (∃ k : ℤ, x = (2 * k + 1) * π / 2) ∧ ∃ l : ℤ, y = (2 * l + 1) * π / 2) :
    tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) := by
  /-
    x y : Complex
    h : Or (And (∀ (k : Int), Ne x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑ …
    ⊢ Eq (Complex.tan (HAdd.hAdd x y)) (HDiv.hDiv (HAdd.hAdd (Complex.tan x) (Comp …
  -/
  rcases h with (⟨h1, h2⟩ | ⟨⟨k, rfl⟩, ⟨l, rfl⟩⟩)
  · rw [tan, sin_add, cos_add, ←
      div_div_div_cancel_right₀ (mul_ne_zero (cos_ne_zero_iff.mpr h1) (cos_ne_zero_iff.mpr h2)),
      add_div, sub_div]
    simp only [← div_mul_div_comm, tan, mul_one, one_mul, div_self (cos_ne_zero_iff.mpr h1),
      div_self (cos_ne_zero_iff.mpr h2)]
    /-
      case inr.intro.intro.intro
      k l : Int
      ⊢ Eq (Complex.tan (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) …
    -/
  · haveI t := tan_int_mul_pi_div_two
    /-
      case inr.intro.intro.intro
      k l : Int
      t : ∀ (n : Int), Eq (Complex.tan (HDiv.hDiv (HMul.hMul ↑n ↑Real.pi) 2)) 0
      ⊢ Eq (Complex.tan (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) …
    -/
    obtain ⟨hx, hy, hxy⟩ := t (2 * k + 1), t (2 * l + 1), t (2 * k + 1 + (2 * l + 1))
    /-
      case inr.intro.intro.intro
      k l : Int
      t : ∀ (n : Int), Eq (Complex.tan (HDiv.hDiv (HMul.hMul ↑n ↑Real.pi) 2)) 0
      hx : Eq (Complex.tan (HDiv.hDiv (HMul.hMul ↑(HAdd.hAdd (HMul.hMul 2 k) 1) ↑Rea …
      hy : Eq (Complex.tan (HDiv.hDiv (HMul.hMul ↑(HAdd.hAdd (HMul.hMul 2 l) 1) ↑Rea …
      hxy : Eq (Complex.tan (HDiv.hDiv (HMul.hMul ↑(HAdd.hAdd (HAdd.hAdd (HMul.hMul  …
      ⊢ Eq (Complex.tan (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) …
    -/
    simp only [Int.cast_add, Int.cast_two, Int.cast_mul, Int.cast_one, hx, hy] at hx hy hxy
    rw [hx, hy, add_zero, zero_div, mul_div_assoc, mul_div_assoc, ←
      add_mul (2 * (k : ℂ) + 1) (2 * l + 1) (π / 2), ← mul_div_assoc, hxy]


theorem tan_add' {x y : ℂ}
    (h : (∀ k : ℤ, x ≠ (2 * k + 1) * π / 2) ∧ ∀ l : ℤ, y ≠ (2 * l + 1) * π / 2) :
    tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) :=
  tan_add (Or.inl h)


theorem tan_two_mul {z : ℂ} : tan (2 * z) = (2 : ℂ) * tan z / ((1 : ℂ) - tan z ^ 2) := by
  /-
    z : Complex
    ⊢ Eq (Complex.tan (HMul.hMul 2 z)) (HDiv.hDiv (HMul.hMul 2 (Complex.tan z)) (H …
  -/
  by_cases h : ∀ k : ℤ, z ≠ (2 * k + 1) * π / 2
    /-
      case pos
      z : Complex
      h : ∀ (k : Int), Ne z (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) ↑Re …
      ⊢ Eq (Complex.tan (HMul.hMul 2 z)) (HDiv.hDiv (HMul.hMul 2 (Complex.tan z)) (H …
    -/
  · rw [two_mul, two_mul, sq, tan_add (Or.inl ⟨h, h⟩)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      z : Complex
      h : Not (∀ (k : Int), Ne z (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1 …
      ⊢ Eq (Complex.tan (HMul.hMul 2 z)) (HDiv.hDiv (HMul.hMul 2 (Complex.tan z)) (H …
    -/
  · rw [not_forall_not] at h
    /-
      case neg
      z : Complex
      h : Exists fun x => Eq z (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑x) 1)  …
      ⊢ Eq (Complex.tan (HMul.hMul 2 z)) (HDiv.hDiv (HMul.hMul 2 (Complex.tan z)) (H …
    -/
    rw [two_mul, two_mul, sq, tan_add (Or.inr ⟨h, h⟩)]
    /-
      🎉 no goals
    -/


theorem tan_add_mul_I {x y : ℂ}
    (h :
      ((∀ k : ℤ, x ≠ (2 * k + 1) * π / 2) ∧ ∀ l : ℤ, y * I ≠ (2 * l + 1) * π / 2) ∨
        (∃ k : ℤ, x = (2 * k + 1) * π / 2) ∧ ∃ l : ℤ, y * I = (2 * l + 1) * π / 2) :
    tan (x + y * I) = (tan x + tanh y * I) / (1 - tan x * tanh y * I) := by
  /-
    x y : Complex
    h : Or (And (∀ (k : Int), Ne x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑ …
    ⊢ Eq (Complex.tan (HAdd.hAdd x (HMul.hMul y Complex.I))) (HDiv.hDiv (HAdd.hAdd …
  -/
  rw [tan_add h, tan_mul_I, mul_assoc]
  /-
    🎉 no goals
  -/


theorem tan_eq {z : ℂ}
    (h :
      ((∀ k : ℤ, (z.re : ℂ) ≠ (2 * k + 1) * π / 2) ∧
          ∀ l : ℤ, (z.im : ℂ) * I ≠ (2 * l + 1) * π / 2) ∨
        (∃ k : ℤ, (z.re : ℂ) = (2 * k + 1) * π / 2) ∧
          ∃ l : ℤ, (z.im : ℂ) * I = (2 * l + 1) * π / 2) :
    tan z = (tan z.re + tanh z.im * I) / (1 - tan z.re * tanh z.im * I) := by
  /-
    z : Complex
    h : Or (And (∀ (k : Int), Ne (↑z.re) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hM …
    ⊢ Eq (Complex.tan z) (HDiv.hDiv (HAdd.hAdd (Complex.tan ↑z.re) (HMul.hMul (Com …
  -/
  convert tan_add_mul_I h; exact (re_add_im z).symm
                           /-
                             🎉 no goals
                           -/


theorem continuousOn_tan : ContinuousOn tan {x | cos x ≠ 0} :=
  continuousOn_sin.div continuousOn_cos fun _x => id


@[continuity]
theorem continuous_tan : Continuous fun x : {x | cos x ≠ 0} => tan x :=
  continuousOn_iff_continuous_restrict.1 continuousOn_tan


theorem cos_eq_iff_quadratic {z w : ℂ} :
    cos z = w ↔ exp (z * I) ^ 2 - 2 * w * exp (z * I) + 1 = 0 := by
  /-
    z w : Complex
    ⊢ Iff (Eq (Complex.cos z) w) (Eq (HAdd.hAdd (HSub.hSub (HPow.hPow (Complex.exp …
  -/
  rw [← sub_eq_zero]
  /-
    z w : Complex
    ⊢ Iff (Eq (HSub.hSub (Complex.cos z) w) 0) (Eq (HAdd.hAdd (HSub.hSub (HPow.hPo …
  -/
  field_simp [cos, exp_neg, exp_ne_zero]
  /-
    z w : Complex
    ⊢ Iff (Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Complex.exp (HMul.hMul z Complex.I …
  -/
  refine Eq.congr ?_ rfl
  /-
    z w : Complex
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Complex.exp (HMul.hMul z Complex.I)) (C …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem cos_surjective : Function.Surjective cos := by
  /-
    ⊢ Function.Surjective Complex.cos
  -/
  intro x
  obtain ⟨w, w₀, hw⟩ : ∃ w ≠ 0, 1 * (w * w) + -2 * x * w + 1 = 0 := by
    rcases exists_quadratic_eq_zero one_ne_zero
        ⟨_, (cpow_nat_inv_pow _ two_ne_zero).symm.trans <| pow_two _⟩ with
      ⟨w, hw⟩
    refine ⟨w, ?_, hw⟩
    rintro rfl
    simp only [zero_add, one_ne_zero, mul_zero] at hw
  /-
    case intro.intro
    x w : Complex
    w₀ : Ne w 0
    hw : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HMul.hMul w w)) (HMul.hMul (HMul.h …
    ⊢ Exists fun a => Eq (Complex.cos a) x
  -/
  refine ⟨log w / I, cos_eq_iff_quadratic.2 ?_⟩
  /-
    case intro.intro
    x w : Complex
    w₀ : Ne w 0
    hw : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HMul.hMul w w)) (HMul.hMul (HMul.h …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HPow.hPow (Complex.exp (HMul.hMul (HDiv.hDiv (Comp …
  -/
  rw [div_mul_cancel₀ _ I_ne_zero, exp_log w₀]
  /-
    case intro.intro
    x w : Complex
    w₀ : Ne w 0
    hw : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HMul.hMul w w)) (HMul.hMul (HMul.h …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HPow.hPow w 2) (HMul.hMul (HMul.hMul 2 x) w)) 1) 0
  -/
  convert hw using 1
  /-
    case h.e'_2
    x w : Complex
    w₀ : Ne w 0
    hw : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HMul.hMul w w)) (HMul.hMul (HMul.h …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HPow.hPow w 2) (HMul.hMul (HMul.hMul 2 x) w)) 1) ( …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem range_cos : Set.range cos = Set.univ :=
  cos_surjective.range_eq


theorem sin_surjective : Function.Surjective sin := by
  /-
    ⊢ Function.Surjective Complex.sin
  -/
  intro x
  /-
    x : Complex
    ⊢ Exists fun a => Eq (Complex.sin a) x
  -/
  rcases cos_surjective x with ⟨z, rfl⟩
  /-
    case intro
    z : Complex
    ⊢ Exists fun a => Eq (Complex.sin a) (Complex.cos z)
  -/
  exact ⟨z + π / 2, sin_add_pi_div_two z⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem range_sin : Set.range sin = Set.univ :=
  sin_surjective.range_eq


theorem cos_eq_zero_iff {θ : ℝ} : cos θ = 0 ↔ ∃ k : ℤ, θ = (2 * k + 1) * π / 2 :=
  mod_cast @Complex.cos_eq_zero_iff θ


theorem cos_ne_zero_iff {θ : ℝ} : cos θ ≠ 0 ↔ ∀ k : ℤ, θ ≠ (2 * k + 1) * π / 2 :=
  mod_cast @Complex.cos_ne_zero_iff θ


theorem cos_eq_cos_iff {x y : ℝ} : cos x = cos y ↔ ∃ k : ℤ, y = 2 * k * π + x ∨ y = 2 * k * π - x :=
  mod_cast @Complex.cos_eq_cos_iff x y


theorem sin_eq_sin_iff {x y : ℝ} :
    sin x = sin y ↔ ∃ k : ℤ, y = 2 * k * π + x ∨ y = (2 * k + 1) * π - x :=
  mod_cast @Complex.sin_eq_sin_iff x y


theorem cos_eq_neg_one_iff {x : ℝ} : cos x = -1 ↔ ∃ k : ℤ, π + k * (2 * π) = x :=
  mod_cast @Complex.cos_eq_neg_one_iff x


theorem sin_eq_one_iff {x : ℝ} : sin x = 1 ↔ ∃ k : ℤ, π / 2 + k * (2 * π) = x :=
  mod_cast @Complex.sin_eq_one_iff x


theorem sin_eq_neg_one_iff {x : ℝ} : sin x = -1 ↔ ∃ k : ℤ, -(π / 2) + k * (2 * π) = x :=
  mod_cast @Complex.sin_eq_neg_one_iff x


theorem tan_eq_zero_iff {θ : ℝ} : tan θ = 0 ↔ ∃ k : ℤ, k * π / 2 = θ :=
  mod_cast @Complex.tan_eq_zero_iff θ


theorem tan_eq_zero_iff' {θ : ℝ} (hθ : cos θ ≠ 0) : tan θ = 0 ↔ ∃ k : ℤ, k * π = θ := by
  /-
    θ : Real
    hθ : Ne (Real.cos θ) 0
    ⊢ Iff (Eq (Real.tan θ) 0) (Exists fun k => Eq (HMul.hMul (↑k) Real.pi) θ)
  -/
  revert hθ
  /-
    θ : Real
    ⊢ Ne (Real.cos θ) 0 → Iff (Eq (Real.tan θ) 0) (Exists fun k => Eq (HMul.hMul ( …
  -/
  exact_mod_cast @Complex.tan_eq_zero_iff' θ
  /-
    🎉 no goals
  -/


theorem tan_ne_zero_iff {θ : ℝ} : tan θ ≠ 0 ↔ ∀ k : ℤ, k * π / 2 ≠ θ :=
  mod_cast @Complex.tan_ne_zero_iff θ


