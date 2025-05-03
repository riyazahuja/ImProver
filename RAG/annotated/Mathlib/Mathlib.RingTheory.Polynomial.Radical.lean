theorem degree_radical_le {a : k[X]} (h : a ≠ 0) :
  (radical a).degree ≤ a.degree := degree_le_of_dvd (radical_dvd_self a) h


theorem natDegree_radical_le {a : k[X]} :
    (radical a).natDegree ≤ a.natDegree := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a : Polynomial k
    ⊢ LE.le (UniqueFactorizationMonoid.radical a).natDegree a.natDegree
  -/
  by_cases ha : a = 0
    /-
      case pos
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a : Polynomial k
      ha : Eq a 0
      ⊢ LE.le (UniqueFactorizationMonoid.radical a).natDegree a.natDegree
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a : Polynomial k
      ha : Not (Eq a 0)
      ⊢ LE.le (UniqueFactorizationMonoid.radical a).natDegree a.natDegree
    -/
  · exact natDegree_le_of_dvd (radical_dvd_self a) ha
    /-
      🎉 no goals
    -/


theorem divRadical_dvd_derivative (a : k[X]) : divRadical a ∣ derivative a := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a : Polynomial k
    ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (Polynomial.derivative a)
  -/
  induction a using induction_on_coprime
  · case h0 =>
    rw [derivative_zero]
    apply dvd_zero
  · case h1 a ha =>
    exact (divRadical_isUnit ha).dvd
  · case hpr p i hp =>
    cases i
    · rw [pow_zero, derivative_one]
      apply dvd_zero
    · case succ i =>
      rw [← mul_dvd_mul_iff_left (radical_ne_zero (p ^ i.succ)), radical_mul_divRadical,
        radical_pow_of_prime hp i.succ_pos, derivative_pow_succ, ← mul_assoc]
      apply dvd_mul_of_dvd_left
      rw [mul_comm, mul_assoc]
      apply dvd_mul_of_dvd_right
      rw [pow_succ, mul_dvd_mul_iff_left (pow_ne_zero i hp.ne_zero), dvd_normalize_iff]
  · -- If it holds for coprime pair a and b, then it also holds for a * b.
    case hcp x y hpxy hx hy =>
    have hc : IsCoprime x y :=
      EuclideanDomain.isCoprime_of_dvd
        (fun ⟨hx, hy⟩ => not_isUnit_zero (hpxy (zero_dvd_iff.mpr hx) (zero_dvd_iff.mpr hy)))
        fun p hp _ hpx hpy => hp (hpxy hpx hpy)
    rw [divRadical_mul hc, derivative_mul]
    exact dvd_add (mul_dvd_mul hx (divRadical_dvd_self y)) (mul_dvd_mul (divRadical_dvd_self x) hy)


theorem divRadical_dvd_wronskian_left (a b : k[X]) : divRadical a ∣ wronskian a b := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b : Polynomial k
    ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (a.wronskian b)
  -/
  rw [wronskian]
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b : Polynomial k
    ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (HSub.hSub (HMul.hMul a (Polynomial.d …
  -/
  apply dvd_sub
    /-
      case h₁
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b : Polynomial k
      ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (HMul.hMul a (Polynomial.derivative b))
    -/
  · apply dvd_mul_of_dvd_left
    /-
      case h₁.h
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b : Polynomial k
      ⊢ Dvd.dvd (EuclideanDomain.divRadical a) a
    -/
    exact divRadical_dvd_self a
    /-
      🎉 no goals
    -/
    /-
      case h₂
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b : Polynomial k
      ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (HMul.hMul (Polynomial.derivative a) b)
    -/
  · apply dvd_mul_of_dvd_left
    /-
      case h₂.h
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b : Polynomial k
      ⊢ Dvd.dvd (EuclideanDomain.divRadical a) (Polynomial.derivative a)
    -/
    exact divRadical_dvd_derivative a
    /-
      🎉 no goals
    -/


theorem divRadical_dvd_wronskian_right (a b : k[X]) : divRadical b ∣ wronskian a b := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b : Polynomial k
    ⊢ Dvd.dvd (EuclideanDomain.divRadical b) (a.wronskian b)
  -/
  rw [← wronskian_neg_eq, dvd_neg]
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b : Polynomial k
    ⊢ Dvd.dvd (EuclideanDomain.divRadical b) (b.wronskian a)
  -/
  exact divRadical_dvd_wronskian_left _ _
  /-
    🎉 no goals
  -/

