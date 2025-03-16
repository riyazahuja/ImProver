/-- `intDegree x` is the degree of the rational function `x`, defined as the difference between
the `natDegree` of its numerator and the `natDegree` of its denominator. In particular,
`intDegree 0 = 0`. -/
def intDegree (x : RatFunc K) : ℤ :=
  natDegree x.num - natDegree x.denom


@[simp]
theorem intDegree_zero : intDegree (0 : RatFunc K) = 0 := by
  /-
    K : Type u
    inst✝ : Field K
    ⊢ Eq (RatFunc.intDegree 0) 0
  -/
  rw [intDegree, num_zero, natDegree_zero, denom_zero, natDegree_one, sub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem intDegree_one : intDegree (1 : RatFunc K) = 0 := by
  /-
    K : Type u
    inst✝ : Field K
    ⊢ Eq (RatFunc.intDegree 1) 0
  -/
  rw [intDegree, num_one, denom_one, sub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem intDegree_C (k : K) : intDegree (C k) = 0 := by
  /-
    K : Type u
    inst✝ : Field K
    k : K
    ⊢ Eq (RatFunc.C k).intDegree 0
  -/
  rw [intDegree, num_C, natDegree_C, denom_C, natDegree_one, sub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem intDegree_X : intDegree (X : RatFunc K) = 1 := by
  rw [intDegree, num_X, Polynomial.natDegree_X, denom_X, Polynomial.natDegree_one,
    Int.ofNat_one, Int.ofNat_zero, sub_zero]


@[simp]
theorem intDegree_polynomial {p : K[X]} :
    intDegree (algebraMap K[X] (RatFunc K) p) = natDegree p := by
  rw [intDegree, RatFunc.num_algebraMap, RatFunc.denom_algebraMap, Polynomial.natDegree_one,
    Int.ofNat_zero, sub_zero]


theorem intDegree_mul {x y : RatFunc K} (hx : x ≠ 0) (hy : y ≠ 0) :
    intDegree (x * y) = intDegree x + intDegree y := by
  /-
    K : Type u
    inst✝ : Field K
    x y : RatFunc K
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HMul.hMul x y).intDegree (HAdd.hAdd x.intDegree y.intDegree)
  -/
  simp only [intDegree, add_sub, sub_add, sub_sub_eq_add_sub, sub_sub, sub_eq_sub_iff_add_eq_add]
  /-
    K : Type u
    inst✝ : Field K
    x y : RatFunc K
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (↑(HMul.hMul x y).num.natDegree) (HAdd.hAdd ↑x.denom.natDegree …
  -/
  norm_cast
  rw [← Polynomial.natDegree_mul x.denom_ne_zero y.denom_ne_zero, ←
    Polynomial.natDegree_mul (RatFunc.num_ne_zero (mul_ne_zero hx hy))
      (mul_ne_zero x.denom_ne_zero y.denom_ne_zero),
    ← Polynomial.natDegree_mul (RatFunc.num_ne_zero hx) (RatFunc.num_ne_zero hy), ←
    Polynomial.natDegree_mul (mul_ne_zero (RatFunc.num_ne_zero hx) (RatFunc.num_ne_zero hy))
      (x * y).denom_ne_zero,
    RatFunc.num_denom_mul]


@[simp]
theorem intDegree_neg (x : RatFunc K) : intDegree (-x) = intDegree x := by
  /-
    K : Type u
    inst✝ : Field K
    x : RatFunc K
    ⊢ Eq (Neg.neg x).intDegree x.intDegree
  -/
  by_cases hx : x = 0
    /-
      case pos
      K : Type u
      inst✝ : Field K
      x : RatFunc K
      hx : Eq x 0
      ⊢ Eq (Neg.neg x).intDegree x.intDegree
    -/
  · rw [hx, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u
      inst✝ : Field K
      x : RatFunc K
      hx : Not (Eq x 0)
      ⊢ Eq (Neg.neg x).intDegree x.intDegree
    -/
  · rw [intDegree, intDegree, ← natDegree_neg x.num]
    exact
      natDegree_sub_eq_of_prod_eq (num_ne_zero (neg_ne_zero.mpr hx)) (denom_ne_zero (-x))
        (neg_ne_zero.mpr (num_ne_zero hx)) (denom_ne_zero x) (num_denom_neg x)


theorem intDegree_add {x y : RatFunc K} (hxy : x + y ≠ 0) :
    (x + y).intDegree =
      (x.num * y.denom + x.denom * y.num).natDegree - (x.denom * y.denom).natDegree :=
  natDegree_sub_eq_of_prod_eq (num_ne_zero hxy) (x + y).denom_ne_zero
    (num_mul_denom_add_denom_mul_num_ne_zero hxy) (mul_ne_zero x.denom_ne_zero y.denom_ne_zero)
    (num_denom_add x y)


theorem natDegree_num_mul_right_sub_natDegree_denom_mul_left_eq_intDegree {x : RatFunc K}
    (hx : x ≠ 0) {s : K[X]} (hs : s ≠ 0) :
    ((x.num * s).natDegree : ℤ) - (s * x.denom).natDegree = x.intDegree := by
  apply natDegree_sub_eq_of_prod_eq (mul_ne_zero (num_ne_zero hx) hs)
    (mul_ne_zero hs x.denom_ne_zero) (num_ne_zero hx) x.denom_ne_zero
  /-
    K : Type u
    inst✝ : Field K
    x : RatFunc K
    hx : Ne x 0
    s : Polynomial K
    hs : Ne s 0
    ⊢ Eq (HMul.hMul (HMul.hMul x.num s) x.denom) (HMul.hMul x.num (HMul.hMul s x.d …
  -/
  rw [mul_assoc]
  /-
    🎉 no goals
  -/


theorem intDegree_add_le {x y : RatFunc K} (hy : y ≠ 0) (hxy : x + y ≠ 0) :
    intDegree (x + y) ≤ max (intDegree x) (intDegree y) := by
  /-
    K : Type u
    inst✝ : Field K
    x y : RatFunc K
    hy : Ne y 0
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (HAdd.hAdd x y).intDegree (Max.max x.intDegree y.intDegree)
  -/
  by_cases hx : x = 0
    /-
      case pos
      K : Type u
      inst✝ : Field K
      x y : RatFunc K
      hy : Ne y 0
      hxy : Ne (HAdd.hAdd x y) 0
      hx : Eq x 0
      ⊢ LE.le (HAdd.hAdd x y).intDegree (Max.max x.intDegree y.intDegree)
    -/
  · simp only [hx, zero_add, ne_eq] at hxy
    /-
      case pos
      K : Type u
      inst✝ : Field K
      x y : RatFunc K
      hy : Ne y 0
      hx : Eq x 0
      hxy : Not (Eq y 0)
      ⊢ LE.le (HAdd.hAdd x y).intDegree (Max.max x.intDegree y.intDegree)
    -/
    simp [hx, hxy]
    /-
      🎉 no goals
    -/
  rw [intDegree_add hxy, ←
    natDegree_num_mul_right_sub_natDegree_denom_mul_left_eq_intDegree hx y.denom_ne_zero,
    mul_comm y.denom, ←
    natDegree_num_mul_right_sub_natDegree_denom_mul_left_eq_intDegree hy x.denom_ne_zero,
    le_max_iff, sub_le_sub_iff_right, Int.ofNat_le, sub_le_sub_iff_right, Int.ofNat_le, ←
    le_max_iff, mul_comm y.num]
  /-
    case neg
    K : Type u
    inst✝ : Field K
    x y : RatFunc K
    hy : Ne y 0
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    ⊢ LE.le (HAdd.hAdd (HMul.hMul x.num y.denom) (HMul.hMul x.denom y.num)).natDeg …
  -/
  exact natDegree_add_le _ _
  /-
    🎉 no goals
  -/


