theorem map_eq_C_mul_X_pow_of_forall_coeff_mem {f : R[X]} {P : Ideal R}
    (hfP : ∀ n : ℕ, ↑n < f.degree → f.coeff n ∈ P) :
    map (mk P) f = C ((mk P) f.leadingCoeff) * X ^ f.natDegree :=
  Polynomial.ext fun n => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      P : Ideal R
      hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
      n : Nat
      ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
    -/
    by_cases hf0 : f = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Eq f 0
        ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
      -/
    · simp [hf0]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      P : Ideal R
      hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
      n : Nat
      hf0 : Not (Eq f 0)
      ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
    -/
    rcases lt_trichotomy (n : WithBot ℕ) (degree f) with (h | h | h)
      /-
        case neg.inl
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Not (Eq f 0)
        h : LT.lt (↑n) f.degree
        ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
      -/
    · rw [coeff_map, eq_zero_iff_mem.2 (hfP n h), coeff_C_mul, coeff_X_pow, if_neg, mul_zero]
      /-
        case neg.inl.hnc
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Not (Eq f 0)
        h : LT.lt (↑n) f.degree
        ⊢ Not (Eq n f.natDegree)
      -/
      rintro rfl
      /-
        case neg.inl.hnc
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        hf0 : Not (Eq f 0)
        h : LT.lt (↑f.natDegree) f.degree
        ⊢ False
      -/
      exact not_lt_of_ge degree_le_natDegree h
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.inl
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Not (Eq f 0)
        h : Eq (↑n) f.degree
        ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
      -/
    · have : natDegree f = n := natDegree_eq_of_degree_eq_some h.symm
      /-
        case neg.inr.inl
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Not (Eq f 0)
        h : Eq (↑n) f.degree
        this : Eq f.natDegree n
        ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
      -/
      rw [coeff_C_mul, coeff_X_pow, if_pos this.symm, mul_one, leadingCoeff, this, coeff_map]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.inr
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        P : Ideal R
        hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
        n : Nat
        hf0 : Not (Eq f 0)
        h : LT.lt f.degree ↑n
        ⊢ Eq ((Polynomial.map (Ideal.Quotient.mk P) f).coeff n) ((HMul.hMul (Polynomia …
      -/
    · rw [coeff_eq_zero_of_degree_lt, coeff_eq_zero_of_degree_lt]
        /-
          case neg.inr.inr
          R : Type u_1
          inst✝ : CommRing R
          f : Polynomial R
          P : Ideal R
          hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
          n : Nat
          hf0 : Not (Eq f 0)
          h : LT.lt f.degree ↑n
          ⊢ LT.lt (HMul.hMul (Polynomial.C ((Ideal.Quotient.mk P) f.leadingCoeff)) (HPow …
        -/
      · refine lt_of_le_of_lt (degree_C_mul_X_pow_le _ _) ?_
        /-
          case neg.inr.inr
          R : Type u_1
          inst✝ : CommRing R
          f : Polynomial R
          P : Ideal R
          hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
          n : Nat
          hf0 : Not (Eq f 0)
          h : LT.lt f.degree ↑n
          ⊢ LT.lt ↑f.natDegree ↑n
        -/
        rwa [← degree_eq_natDegree hf0]
        /-
          🎉 no goals
        -/
        /-
          case neg.inr.inr
          R : Type u_1
          inst✝ : CommRing R
          f : Polynomial R
          P : Ideal R
          hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
          n : Nat
          hf0 : Not (Eq f 0)
          h : LT.lt f.degree ↑n
          ⊢ LT.lt (Polynomial.map (Ideal.Quotient.mk P) f).degree ↑n
        -/
      · exact lt_of_le_of_lt degree_map_le h
        /-
          🎉 no goals
        -/


theorem le_natDegree_of_map_eq_mul_X_pow {n : ℕ} {P : Ideal R} (hP : P.IsPrime) {q : R[X]}
    {c : Polynomial (R ⧸ P)} (hq : map (mk P) q = c * X ^ n) (hc0 : c.degree = 0) :
    n ≤ q.natDegree :=
  Nat.cast_le.1
    (calc
      ↑n = degree (q.map (mk P)) := by
        /-
          R : Type u_1
          inst✝ : CommRing R
          n : Nat
          P : Ideal R
          hP : P.IsPrime
          q : Polynomial R
          c : Polynomial (HasQuotient.Quotient R P)
          hq : Eq (Polynomial.map (Ideal.Quotient.mk P) q) (HMul.hMul c (HPow.hPow Polyn …
          hc0 : Eq c.degree 0
          ⊢ Eq (↑n) (Polynomial.map (Ideal.Quotient.mk P) q).degree
        -/
        rw [hq, degree_mul, hc0, zero_add, degree_pow, degree_X, nsmul_one]
        /-
          🎉 no goals
        -/
      _ ≤ degree q := degree_map_le
      _ ≤ natDegree q := degree_le_natDegree
      )


theorem eval_zero_mem_ideal_of_eq_mul_X_pow {n : ℕ} {P : Ideal R} {q : R[X]}
    {c : Polynomial (R ⧸ P)} (hq : map (mk P) q = c * X ^ n) (hn0 : n ≠ 0) : eval 0 q ∈ P := by
  rw [← coeff_zero_eq_eval_zero, ← eq_zero_iff_mem, ← coeff_map, hq,
    coeff_zero_eq_eval_zero, eval_mul, eval_pow, eval_X, zero_pow hn0, mul_zero]


theorem isUnit_of_natDegree_eq_zero_of_isPrimitive {p q : R[X]}
    -- Porting note: stated using `IsPrimitive` which is defeq to old statement.
    (hu : IsPrimitive (p * q)) (hpm : p.natDegree = 0) : IsUnit p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hu : (HMul.hMul p q).IsPrimitive
    hpm : Eq p.natDegree 0
    ⊢ IsUnit p
  -/
  rw [eq_C_of_degree_le_zero (natDegree_eq_zero_iff_degree_le_zero.1 hpm), isUnit_C]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hu : (HMul.hMul p q).IsPrimitive
    hpm : Eq p.natDegree 0
    ⊢ IsUnit (p.coeff 0)
  -/
  refine hu _ ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hu : (HMul.hMul p q).IsPrimitive
    hpm : Eq p.natDegree 0
    ⊢ Dvd.dvd (Polynomial.C (p.coeff 0)) (HMul.hMul p q)
  -/
  rw [← eq_C_of_degree_le_zero (natDegree_eq_zero_iff_degree_le_zero.1 hpm)]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hu : (HMul.hMul p q).IsPrimitive
    hpm : Eq p.natDegree 0
    ⊢ Dvd.dvd p (HMul.hMul p q)
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


/-- If `f` is a non constant polynomial with coefficients in `R`, and `P` is a prime ideal in `R`,
then if every coefficient in `R` except the leading coefficient is in `P`, and
the trailing coefficient is not in `P^2` and no non units in `R` divide `f`, then `f` is
irreducible. -/
theorem irreducible_of_eisenstein_criterion {f : R[X]} {P : Ideal R} (hP : P.IsPrime)
    (hfl : f.leadingCoeff ∉ P) (hfP : ∀ n : ℕ, ↑n < degree f → f.coeff n ∈ P) (hfd0 : 0 < degree f)
    (h0 : f.coeff 0 ∉ P ^ 2) (hu : f.IsPrimitive) : Irreducible f :=
                                  /-
                                    R : Type u_1
                                    inst✝¹ : CommRing R
                                    inst✝ : IsDomain R
                                    f : Polynomial R
                                    P : Ideal R
                                    hP : P.IsPrime
                                    hfl : Not (Membership.mem P f.leadingCoeff)
                                    hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
                                    hfd0 : LT.lt 0 f.degree
                                    h0 : Not (Membership.mem (HPow.hPow P 2) (f.coeff 0))
                                    hu : f.IsPrimitive
                                    x✝ : Eq f 0
                                    ⊢ False
                                  -/
  have hf0 : f ≠ 0 := fun _ => by simp_all only [not_true, Submodule.zero_mem, coeff_zero]
                                  /-
                                    🎉 no goals
                                  -/
  have hf : f.map (mk P) = C (mk P (leadingCoeff f)) * X ^ natDegree f :=
    map_eq_C_mul_X_pow_of_forall_coeff_mem hfP
  have hfd0 : 0 < f.natDegree := WithBot.coe_lt_coe.1 (lt_of_lt_of_le hfd0 degree_le_natDegree)
                                           /-
                                             R : Type u_1
                                             inst✝¹ : CommRing R
                                             inst✝ : IsDomain R
                                             f : Polynomial R
                                             P : Ideal R
                                             hP : P.IsPrime
                                             hfl : Not (Membership.mem P f.leadingCoeff)
                                             hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
                                             hfd0✝ : LT.lt 0 f.degree
                                             h0 : Not (Membership.mem (HPow.hPow P 2) (f.coeff 0))
                                             hu : f.IsPrimitive
                                             hf0 : Ne f 0
                                             hf : Eq (Polynomial.map (Ideal.Quotient.mk P) f) (HMul.hMul (Polynomial.C ((Id …
                                             hfd0 : LT.lt 0 f.natDegree
                                             h : Eq f.degree 0
                                             ⊢ False
                                           -/
  ⟨mt degree_eq_zero_of_isUnit fun h => by simp_all only [lt_irrefl], by
                                           /-
                                             🎉 no goals
                                           -/
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      f : Polynomial R
      P : Ideal R
      hP : P.IsPrime
      hfl : Not (Membership.mem P f.leadingCoeff)
      hfP : ∀ (n : Nat), LT.lt (↑n) f.degree → Membership.mem P (f.coeff n)
      hfd0✝ : LT.lt 0 f.degree
      h0 : Not (Membership.mem (HPow.hPow P 2) (f.coeff 0))
      hu : f.IsPrimitive
      hf0 : Ne f 0
      hf : Eq (Polynomial.map (Ideal.Quotient.mk P) f) (HMul.hMul (Polynomial.C ((Id …
      hfd0 : LT.lt 0 f.natDegree
      ⊢ ∀ (a b : Polynomial R), Eq f (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
    -/
    rintro p q rfl
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      P : Ideal R
      hP : P.IsPrime
      p q : Polynomial R
      hfl : Not (Membership.mem P (HMul.hMul p q).leadingCoeff)
      hfP : ∀ (n : Nat), LT.lt (↑n) (HMul.hMul p q).degree → Membership.mem P ((HMul …
      hfd0✝ : LT.lt 0 (HMul.hMul p q).degree
      h0 : Not (Membership.mem (HPow.hPow P 2) ((HMul.hMul p q).coeff 0))
      hu : (HMul.hMul p q).IsPrimitive
      hf0 : Ne (HMul.hMul p q) 0
      hf : Eq (Polynomial.map (Ideal.Quotient.mk P) (HMul.hMul p q)) (HMul.hMul (Pol …
      hfd0 : LT.lt 0 (HMul.hMul p q).natDegree
      ⊢ Or (IsUnit p) (IsUnit q)
    -/
    rw [Polynomial.map_mul] at hf
    rcases mul_eq_mul_prime_pow
        (show Prime (X : Polynomial (R ⧸ P)) from monic_X.prime_of_degree_eq_one degree_X) hf with
      ⟨m, n, b, c, hmnd, hbc, hp, hq⟩
    have hmn : 0 < m → 0 < n → False := by
      intro hm0 hn0
      refine h0 ?_
      rw [coeff_zero_eq_eval_zero, eval_mul, sq]
      exact
        Ideal.mul_mem_mul (eval_zero_mem_ideal_of_eq_mul_X_pow hp hm0.ne')
          (eval_zero_mem_ideal_of_eq_mul_X_pow hq hn0.ne')
    /-
      case intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      P : Ideal R
      hP : P.IsPrime
      p q : Polynomial R
      hfl : Not (Membership.mem P (HMul.hMul p q).leadingCoeff)
      hfP : ∀ (n : Nat), LT.lt (↑n) (HMul.hMul p q).degree → Membership.mem P ((HMul …
      hfd0✝ : LT.lt 0 (HMul.hMul p q).degree
      h0 : Not (Membership.mem (HPow.hPow P 2) ((HMul.hMul p q).coeff 0))
      hu : (HMul.hMul p q).IsPrimitive
      hf0 : Ne (HMul.hMul p q) 0
      hf : Eq (HMul.hMul (Polynomial.map (Ideal.Quotient.mk P) p) (Polynomial.map (I …
      hfd0 : LT.lt 0 (HMul.hMul p q).natDegree
      m n : Nat
      b c : Polynomial (HasQuotient.Quotient R P)
      hmnd : Eq (HAdd.hAdd m n) (HMul.hMul p q).natDegree
      hbc : Eq (Polynomial.C ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff)) ( …
      hp : Eq (Polynomial.map (Ideal.Quotient.mk P) p) (HMul.hMul b (HPow.hPow Polyn …
      hq : Eq (Polynomial.map (Ideal.Quotient.mk P) q) (HMul.hMul c (HPow.hPow Polyn …
      hmn : LT.lt 0 m → LT.lt 0 n → False
      ⊢ Or (IsUnit p) (IsUnit q)
    -/
    have hpql0 : (mk P) (p * q).leadingCoeff ≠ 0 := by rwa [Ne, eq_zero_iff_mem]
    have hp0 : p ≠ 0 := fun h => by
      simp_all only [zero_mul, eq_self_iff_true, not_true, Ne]
    have hq0 : q ≠ 0 := fun h => by
      simp_all only [eq_self_iff_true, not_true, Ne, mul_zero]
    have hbc0 : degree b = 0 ∧ degree c = 0 := by
      apply_fun degree at hbc
      rwa [degree_C hpql0, degree_mul, eq_comm, Nat.WithBot.add_eq_zero_iff] at hbc
    /-
      case intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      P : Ideal R
      hP : P.IsPrime
      p q : Polynomial R
      hfl : Not (Membership.mem P (HMul.hMul p q).leadingCoeff)
      hfP : ∀ (n : Nat), LT.lt (↑n) (HMul.hMul p q).degree → Membership.mem P ((HMul …
      hfd0✝ : LT.lt 0 (HMul.hMul p q).degree
      h0 : Not (Membership.mem (HPow.hPow P 2) ((HMul.hMul p q).coeff 0))
      hu : (HMul.hMul p q).IsPrimitive
      hf0 : Ne (HMul.hMul p q) 0
      hf : Eq (HMul.hMul (Polynomial.map (Ideal.Quotient.mk P) p) (Polynomial.map (I …
      hfd0 : LT.lt 0 (HMul.hMul p q).natDegree
      m n : Nat
      b c : Polynomial (HasQuotient.Quotient R P)
      hmnd : Eq (HAdd.hAdd m n) (HMul.hMul p q).natDegree
      hbc : Eq (Polynomial.C ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff)) ( …
      hp : Eq (Polynomial.map (Ideal.Quotient.mk P) p) (HMul.hMul b (HPow.hPow Polyn …
      hq : Eq (Polynomial.map (Ideal.Quotient.mk P) q) (HMul.hMul c (HPow.hPow Polyn …
      hmn : LT.lt 0 m → LT.lt 0 n → False
      hpql0 : Ne ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff) 0
      hp0 : Ne p 0
      hq0 : Ne q 0
      hbc0 : And (Eq b.degree 0) (Eq c.degree 0)
      ⊢ Or (IsUnit p) (IsUnit q)
    -/
    have hmp : m ≤ natDegree p := le_natDegree_of_map_eq_mul_X_pow hP hp hbc0.1
    /-
      case intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      P : Ideal R
      hP : P.IsPrime
      p q : Polynomial R
      hfl : Not (Membership.mem P (HMul.hMul p q).leadingCoeff)
      hfP : ∀ (n : Nat), LT.lt (↑n) (HMul.hMul p q).degree → Membership.mem P ((HMul …
      hfd0✝ : LT.lt 0 (HMul.hMul p q).degree
      h0 : Not (Membership.mem (HPow.hPow P 2) ((HMul.hMul p q).coeff 0))
      hu : (HMul.hMul p q).IsPrimitive
      hf0 : Ne (HMul.hMul p q) 0
      hf : Eq (HMul.hMul (Polynomial.map (Ideal.Quotient.mk P) p) (Polynomial.map (I …
      hfd0 : LT.lt 0 (HMul.hMul p q).natDegree
      m n : Nat
      b c : Polynomial (HasQuotient.Quotient R P)
      hmnd : Eq (HAdd.hAdd m n) (HMul.hMul p q).natDegree
      hbc : Eq (Polynomial.C ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff)) ( …
      hp : Eq (Polynomial.map (Ideal.Quotient.mk P) p) (HMul.hMul b (HPow.hPow Polyn …
      hq : Eq (Polynomial.map (Ideal.Quotient.mk P) q) (HMul.hMul c (HPow.hPow Polyn …
      hmn : LT.lt 0 m → LT.lt 0 n → False
      hpql0 : Ne ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff) 0
      hp0 : Ne p 0
      hq0 : Ne q 0
      hbc0 : And (Eq b.degree 0) (Eq c.degree 0)
      hmp : LE.le m p.natDegree
      ⊢ Or (IsUnit p) (IsUnit q)
    -/
    have hnq : n ≤ natDegree q := le_natDegree_of_map_eq_mul_X_pow hP hq hbc0.2
    have hpmqn : p.natDegree = m ∧ q.natDegree = n := by
      rw [natDegree_mul hp0 hq0] at hmnd
      contrapose hmnd
      apply ne_of_lt
      rw [not_and_or] at hmnd
      cases' hmnd with hmnd hmnd
      · exact add_lt_add_of_lt_of_le (lt_of_le_of_ne hmp (Ne.symm hmnd)) hnq
      · exact add_lt_add_of_le_of_lt hmp (lt_of_le_of_ne hnq (Ne.symm hmnd))
    obtain rfl | rfl : m = 0 ∨ n = 0 := by
      rwa [pos_iff_ne_zero, pos_iff_ne_zero, imp_false, Classical.not_not, ← or_iff_not_imp_left]
        at hmn
      /-
        case intro.intro.intro.intro.intro.intro.intro.inl
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        P : Ideal R
        hP : P.IsPrime
        p q : Polynomial R
        hfl : Not (Membership.mem P (HMul.hMul p q).leadingCoeff)
        hfP : ∀ (n : Nat), LT.lt (↑n) (HMul.hMul p q).degree → Membership.mem P ((HMul …
        hfd0✝ : LT.lt 0 (HMul.hMul p q).degree
        h0 : Not (Membership.mem (HPow.hPow P 2) ((HMul.hMul p q).coeff 0))
        hu : (HMul.hMul p q).IsPrimitive
        hf0 : Ne (HMul.hMul p q) 0
        hf : Eq (HMul.hMul (Polynomial.map (Ideal.Quotient.mk P) p) (Polynomial.map (I …
        hfd0 : LT.lt 0 (HMul.hMul p q).natDegree
        n : Nat
        b c : Polynomial (HasQuotient.Quotient R P)
        hbc : Eq (Polynomial.C ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff)) ( …
        hq : Eq (Polynomial.map (Ideal.Quotient.mk P) q) (HMul.hMul c (HPow.hPow Polyn …
        hpql0 : Ne ((Ideal.Quotient.mk P) (HMul.hMul p q).leadingCoeff) 0
        hp0 : Ne p 0
        hq0 : Ne q 0
        hbc0 : And (Eq b.degree 0) (Eq c.degree 0)
        hnq : LE.le n q.natDegree
        hmnd : Eq (HAdd.hAdd 0 n) (HMul.hMul p q).natDegree
        hp : Eq (Polynomial.map (Ideal.Quotient.mk P) p) (HMul.hMul b (HPow.hPow Polyn …
        hmn : LT.lt 0 0 → LT.lt 0 n → False
        hmp : LE.le 0 p.natDegree
        hpmqn : And (Eq p.natDegree 0) (Eq q.natDegree n)
        ⊢ Or (IsUnit p) (IsUnit q)
      -/
    · exact Or.inl (isUnit_of_natDegree_eq_zero_of_isPrimitive hu hpmqn.1)
      /-
        🎉 no goals
      -/
    · exact Or.inr
          (isUnit_of_natDegree_eq_zero_of_isPrimitive
            (show IsPrimitive (q * p) by simpa [mul_comm] using hu)
            hpmqn.2)⟩


