/-- Wronskian of a pair of polynomials, `W(a, b) = ab' - a'b`. -/
def wronskian (a b : R[X]) : R[X] :=
  a * (derivative b) - (derivative a) * b


variable (R) in
/-- `Polynomial.wronskian` as a bilinear map. -/
def wronskianBilin : R[X] →ₗ[R] R[X] →ₗ[R] R[X] :=
  (LinearMap.mul R R[X]).compl₂ derivative - (LinearMap.mul R R[X]).comp derivative


@[simp]
theorem wronskianBilin_apply (a b : R[X]) : wronskianBilin R a b = wronskian a b := rfl


@[simp]
theorem wronskian_zero_left (a : R[X]) : wronskian 0 a = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a : Polynomial R
    ⊢ Eq (Polynomial.wronskian 0 a) 0
  -/
  rw [← wronskianBilin_apply 0 a, map_zero]; rfl
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem wronskian_zero_right (a : R[X]) : wronskian a 0 = 0 := (wronskianBilin R a).map_zero


theorem wronskian_neg_left (a b : R[X]) : wronskian (-a) b = -wronskian a b :=
  LinearMap.map_neg₂ (wronskianBilin R) a b


theorem wronskian_neg_right (a b : R[X]) : wronskian a (-b) = -wronskian a b :=
  (wronskianBilin R a).map_neg b


theorem wronskian_add_right (a b c : R[X]) : wronskian a (b + c) = wronskian a b + wronskian a c :=
  (wronskianBilin R a).map_add b c


theorem wronskian_add_left (a b c : R[X]) : wronskian (a + b) c = wronskian a c + wronskian b c :=
  (wronskianBilin R).map_add₂ a b c


theorem wronskian_self_eq_zero (a : R[X]) : wronskian a a = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a : Polynomial R
    ⊢ Eq (a.wronskian a) 0
  -/
  rw [wronskian, mul_comm, sub_self]
  /-
    🎉 no goals
  -/


theorem isAlt_wronskianBilin : (wronskianBilin R).IsAlt := wronskian_self_eq_zero


theorem wronskian_neg_eq (a b : R[X]) : -wronskian a b = wronskian b a :=
  LinearMap.IsAlt.neg isAlt_wronskianBilin a b


theorem wronskian_eq_of_sum_zero {a b c : R[X]} (hAdd : a + b + c = 0) :
    wronskian a b = wronskian b c := isAlt_wronskianBilin.eq_of_add_add_eq_zero hAdd


/-- Degree of `W(a,b)` is strictly less than the sum of degrees of `a` and `b` (both nonzero). -/
theorem degree_wronskian_lt_add {a b : R[X]} (ha : a ≠ 0) (hb : b ≠ 0) :
    (wronskian a b).degree < a.degree + b.degree := by
  calc
    (wronskian a b).degree ≤ max (a * derivative b).degree (derivative a * b).degree :=
      Polynomial.degree_sub_le _ _
    _ < a.degree + b.degree := by
      rw [max_lt_iff]
      constructor
      case left =>
        apply lt_of_le_of_lt
        · exact degree_mul_le a (derivative b)
        · rw [← Polynomial.degree_ne_bot] at ha
          rw [WithBot.add_lt_add_iff_left ha]
          exact Polynomial.degree_derivative_lt hb
      case right =>
        apply lt_of_le_of_lt
        · exact degree_mul_le (derivative a) b
        · rw [← Polynomial.degree_ne_bot] at hb
          rw [WithBot.add_lt_add_iff_right hb]
          exact Polynomial.degree_derivative_lt ha


/--
`natDegree` version of the above theorem.
Note this would be false with just `(ha : a ≠ 0) (hb : b ≠ 0),
as when `a = b = 1` we have `(wronskian a b).natDegree = a.natDegree = b.natDegree = 0`.
-/
theorem natDegree_wronskian_lt_add {a b : R[X]} (hw : wronskian a b ≠ 0) :
    (wronskian a b).natDegree < a.natDegree + b.natDegree := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Polynomial R
    hw : Ne (a.wronskian b) 0
    ⊢ LT.lt (a.wronskian b).natDegree (HAdd.hAdd a.natDegree b.natDegree)
  -/
  have ha : a ≠ 0 := by intro h; subst h; rw [wronskian_zero_left] at hw; exact hw rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Polynomial R
    hw : Ne (a.wronskian b) 0
    ha : Ne a 0
    ⊢ LT.lt (a.wronskian b).natDegree (HAdd.hAdd a.natDegree b.natDegree)
  -/
  have hb : b ≠ 0 := by intro h; subst h; rw [wronskian_zero_right] at hw; exact hw rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Polynomial R
    hw : Ne (a.wronskian b) 0
    ha : Ne a 0
    hb : Ne b 0
    ⊢ LT.lt (a.wronskian b).natDegree (HAdd.hAdd a.natDegree b.natDegree)
  -/
  rw [← WithBot.coe_lt_coe, WithBot.coe_add]
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Polynomial R
    hw : Ne (a.wronskian b) 0
    ha : Ne a 0
    hb : Ne b 0
    ⊢ LT.lt (↑(a.wronskian b).natDegree) (HAdd.hAdd ↑a.natDegree ↑b.natDegree)
  -/
  convert ← degree_wronskian_lt_add ha hb
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommRing R
      a b : Polynomial R
      hw : Ne (a.wronskian b) 0
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq (a.wronskian b).degree ↑(a.wronskian b).natDegree
    -/
  · exact Polynomial.degree_eq_natDegree hw
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_5
      R : Type u_1
      inst✝ : CommRing R
      a b : Polynomial R
      hw : Ne (a.wronskian b) 0
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq a.degree ↑a.natDegree
    -/
  · exact Polynomial.degree_eq_natDegree ha
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_6
      R : Type u_1
      inst✝ : CommRing R
      a b : Polynomial R
      hw : Ne (a.wronskian b) 0
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq b.degree ↑b.natDegree
    -/
  · exact Polynomial.degree_eq_natDegree hb
    /-
      🎉 no goals
    -/


/--
For coprime polynomials `a` and `b`, their Wronskian is zero
if and only if their derivatives are zeros.
-/
theorem _root_.IsCoprime.wronskian_eq_zero_iff
    [NoZeroDivisors R] {a b : R[X]} (hc : IsCoprime a b) :
    wronskian a b = 0 ↔ derivative a = 0 ∧ derivative b = 0 where
  mp hw := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      a b : Polynomial R
      hc : IsCoprime a b
      hw : Eq (a.wronskian b) 0
      ⊢ And (Eq (Polynomial.derivative a) 0) (Eq (Polynomial.derivative b) 0)
    -/
    rw [wronskian, sub_eq_iff_eq_add, zero_add] at hw
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      a b : Polynomial R
      hc : IsCoprime a b
      hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
      ⊢ And (Eq (Polynomial.derivative a) 0) (Eq (Polynomial.derivative b) 0)
    -/
    constructor
      /-
        case left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Eq (Polynomial.derivative a) 0
      -/
    · rw [← dvd_derivative_iff]
      /-
        case left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Dvd.dvd a (Polynomial.derivative a)
      -/
      apply hc.dvd_of_dvd_mul_right
      /-
        case left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Dvd.dvd a (HMul.hMul (Polynomial.derivative a) b)
      -/
      rw [← hw]; exact dvd_mul_right _ _
                 /-
                   🎉 no goals
                 -/
      /-
        case right
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Eq (Polynomial.derivative b) 0
      -/
    · rw [← dvd_derivative_iff]
      /-
        case right
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Dvd.dvd b (Polynomial.derivative b)
      -/
      apply hc.symm.dvd_of_dvd_mul_left
      /-
        case right
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        a b : Polynomial R
        hc : IsCoprime a b
        hw : Eq (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial.derivat …
        ⊢ Dvd.dvd b (HMul.hMul a (Polynomial.derivative b))
      -/
      rw [hw]; exact dvd_mul_left _ _
               /-
                 🎉 no goals
               -/
  mpr hdab := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      a b : Polynomial R
      hc : IsCoprime a b
      hdab : And (Eq (Polynomial.derivative a) 0) (Eq (Polynomial.derivative b) 0)
      ⊢ Eq (a.wronskian b) 0
    -/
    cases' hdab with hda hdb
    /-
      case intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      a b : Polynomial R
      hc : IsCoprime a b
      hda : Eq (Polynomial.derivative a) 0
      hdb : Eq (Polynomial.derivative b) 0
      ⊢ Eq (a.wronskian b) 0
    -/
    rw [wronskian]
    /-
      case intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      a b : Polynomial R
      hc : IsCoprime a b
      hda : Eq (Polynomial.derivative a) 0
      hdb : Eq (Polynomial.derivative b) 0
      ⊢ Eq (HSub.hSub (HMul.hMul a (Polynomial.derivative b)) (HMul.hMul (Polynomial …
    -/
    rw [hda, hdb]; simp only [MulZeroClass.mul_zero, MulZeroClass.zero_mul, sub_self]
                   /-
                     🎉 no goals
                   -/


@[deprecated (since := "2024-11-06")]
alias IsCoprime.wronskian_eq_zero_iff := IsCoprime.wronskian_eq_zero_iff


