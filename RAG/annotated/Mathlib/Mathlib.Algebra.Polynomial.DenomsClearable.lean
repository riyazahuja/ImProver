/-- `denomsClearable` formalizes the property that `b ^ N * f (a / b)`
does not have denominators, if the inequality `f.natDegree ≤ N` holds.

The definition asserts the existence of an element `D` of `R` and an
element `bi = 1 / i b` of `K` such that clearing the denominators of
the fraction equals `i D`.
-/
def DenomsClearable (a b : R) (N : ℕ) (f : R[X]) (i : R →+* K) : Prop :=
  ∃ (D : R) (bi : K), bi * i b = 1 ∧ i D = i b ^ N * eval (i a * bi) (f.map i)


theorem denomsClearable_zero (N : ℕ) (a : R) (bu : bi * i b = 1) : DenomsClearable a b N 0 i :=
  ⟨0, bi, bu, by
    /-
      R : Type u_1
      K : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CommSemiring K
      i : RingHom R K
      b : R
      bi : K
      N : Nat
      a : R
      bu : Eq (HMul.hMul bi (i b)) 1
      ⊢ Eq (i 0) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a) bi …
    -/
    simp only [eval_zero, RingHom.map_zero, mul_zero, Polynomial.map_zero]⟩
    /-
      🎉 no goals
    -/


theorem denomsClearable_C_mul_X_pow {N : ℕ} (a : R) (bu : bi * i b = 1) {n : ℕ} (r : R)
    (nN : n ≤ N) : DenomsClearable a b N (C r * X ^ n) i := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ DenomsClearable a b N (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X n) …
  -/
  refine ⟨r * a ^ n * b ^ (N - n), bi, bu, ?_⟩
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ Eq (i (HMul.hMul (HMul.hMul r (HPow.hPow a n)) (HPow.hPow b (HSub.hSub N n)) …
  -/
  rw [C_mul_X_pow_eq_monomial, map_monomial, ← C_mul_X_pow_eq_monomial, eval_mul, eval_pow, eval_C]
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ Eq (i (HMul.hMul (HMul.hMul r (HPow.hPow a n)) (HPow.hPow b (HSub.hSub N n)) …
  -/
  rw [RingHom.map_mul, RingHom.map_mul, RingHom.map_pow, RingHom.map_pow, eval_X, mul_comm]
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ Eq (HMul.hMul (HPow.hPow (i b) (HSub.hSub N n)) (HMul.hMul (i r) (HPow.hPow  …
  -/
  rw [← tsub_add_cancel_of_le nN]
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ Eq (HMul.hMul (HPow.hPow (i b) (HSub.hSub (HAdd.hAdd (HSub.hSub N n) n) n))  …
  -/
  conv_lhs => rw [← mul_one (i a), ← bu]
  /-
    R : Type u_1
    K : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CommSemiring K
    i : RingHom R K
    b : R
    bi : K
    N : Nat
    a : R
    bu : Eq (HMul.hMul bi (i b)) 1
    n : Nat
    r : R
    nN : LE.le n N
    ⊢ Eq (HMul.hMul (HPow.hPow (i b) (HSub.hSub (HAdd.hAdd (HSub.hSub N n) n) n))  …
  -/
  simp [mul_assoc, mul_comm, mul_left_comm, pow_add, mul_pow]
  /-
    🎉 no goals
  -/


theorem DenomsClearable.add {N : ℕ} {f g : R[X]} :
    DenomsClearable a b N f i → DenomsClearable a b N g i → DenomsClearable a b N (f + g) i :=
  fun ⟨Df, bf, bfu, Hf⟩ ⟨Dg, bg, bgu, Hg⟩ =>
  ⟨Df + Dg, bf, bfu, by
    /-
      R : Type u_1
      K : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CommSemiring K
      i : RingHom R K
      a b : R
      N : Nat
      f g : Polynomial R
      x✝¹ : DenomsClearable a b N f i
      x✝ : DenomsClearable a b N g i
      Df : R
      bf : K
      bfu : Eq (HMul.hMul bf (i b)) 1
      Hf : Eq (i Df) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      Dg : R
      bg : K
      bgu : Eq (HMul.hMul bg (i b)) 1
      Hg : Eq (i Dg) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      ⊢ Eq (i (HAdd.hAdd Df Dg)) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HM …
    -/
    rw [RingHom.map_add, Polynomial.map_add, eval_add, mul_add, Hf, Hg]
    /-
      R : Type u_1
      K : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CommSemiring K
      i : RingHom R K
      a b : R
      N : Nat
      f g : Polynomial R
      x✝¹ : DenomsClearable a b N f i
      x✝ : DenomsClearable a b N g i
      Df : R
      bf : K
      bfu : Eq (HMul.hMul bf (i b)) 1
      Hf : Eq (i Df) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      Dg : R
      bg : K
      bgu : Eq (HMul.hMul bg (i b)) 1
      Hg : Eq (i Dg) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i  …
    -/
    congr
    /-
      case e_a.e_a.e_a.e_a
      R : Type u_1
      K : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CommSemiring K
      i : RingHom R K
      a b : R
      N : Nat
      f g : Polynomial R
      x✝¹ : DenomsClearable a b N f i
      x✝ : DenomsClearable a b N g i
      Df : R
      bf : K
      bfu : Eq (HMul.hMul bf (i b)) 1
      Hf : Eq (i Df) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      Dg : R
      bg : K
      bgu : Eq (HMul.hMul bg (i b)) 1
      Hg : Eq (i Dg) (HMul.hMul (HPow.hPow (i b) N) (Polynomial.eval (HMul.hMul (i a …
      ⊢ Eq bg bf
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    refine @inv_unique K _ (i b) bg bf ?_ ?_ <;> rwa [mul_comm]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem denomsClearable_of_natDegree_le (N : ℕ) (a : R) (bu : bi * i b = 1) :
    ∀ f : R[X], f.natDegree ≤ N → DenomsClearable a b N f i :=
  induction_with_natDegree_le _ N (denomsClearable_zero N a bu)
    (fun _ r _ => denomsClearable_C_mul_X_pow a bu r) fun _ _ _ _ df dg => df.add dg


/-- If `i : R → K` is a ring homomorphism, `f` is a polynomial with coefficients in `R`,
`a, b` are elements of `R`, with `i b` invertible, then there is a `D ∈ R` such that
`b ^ f.natDegree * f (a / b)` equals `i D`. -/
theorem denomsClearable_natDegree (i : R →+* K) (f : R[X]) (a : R) (bu : bi * i b = 1) :
    DenomsClearable a b f.natDegree f i :=
  denomsClearable_of_natDegree_le f.natDegree a bu f le_rfl


/-- Evaluating a polynomial with integer coefficients at a rational number and clearing
denominators, yields a number greater than or equal to one.  The target can be any
`LinearOrderedField K`.
The assumption on `K` could be weakened to `LinearOrderedCommRing` assuming that the
image of the denominator is invertible in `K`. -/
theorem one_le_pow_mul_abs_eval_div {K : Type*} [LinearOrderedField K] {f : ℤ[X]} {a b : ℤ}
    (b0 : 0 < b) (fab : eval ((a : K) / b) (f.map (algebraMap ℤ K)) ≠ 0) :
    (1 : K) ≤ (b : K) ^ f.natDegree * |eval ((a : K) / b) (f.map (algebraMap ℤ K))| := by
  obtain ⟨ev, bi, bu, hF⟩ :=
    denomsClearable_natDegree (b := b) (algebraMap ℤ K) f a
      (by
        rw [eq_intCast, one_div_mul_cancel]
        rw [Int.cast_ne_zero]
        exact b0.ne.symm)
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eval (HDiv. …
  -/
  obtain Fa := congr_arg abs hF
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    Fa : Eq (abs ((algebraMap Int K) ev)) (abs (HMul.hMul (HPow.hPow ((algebraMap  …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eval (HDiv. …
  -/
  rw [eq_one_div_of_mul_eq_one_left bu, eq_intCast, eq_intCast, abs_mul] at Fa
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    Fa : Eq (abs ↑ev) (HMul.hMul (abs (HPow.hPow (↑b) f.natDegree)) (abs (Polynomi …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eval (HDiv. …
  -/
  rw [abs_of_pos (pow_pos (Int.cast_pos.mpr b0) _ : 0 < (b : K) ^ _), one_div, eq_intCast] at Fa
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eval (HDiv. …
  -/
  rw [div_eq_mul_inv, ← Fa, ← Int.cast_abs, ← Int.cast_one, Int.cast_le]
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
    ⊢ LE.le 1 (abs ev)
  -/
  refine Int.le_of_lt_add_one ((lt_add_iff_pos_left 1).mpr (abs_pos.mpr fun F0 => fab ?_))
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Eq ((algebraMap Int K) ev) (HMul.hMul (HPow.hPow ((algebraMap Int K) b) f …
    Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
    F0 : Eq ev 0
    ⊢ Eq (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) f)) 0
  -/
  rw [eq_one_div_of_mul_eq_one_left bu, F0, one_div, eq_intCast, Int.cast_zero, zero_eq_mul] at hF
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝ : LinearOrderedField K
    f : Polynomial Int
    a b : Int
    b0 : LT.lt 0 b
    fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
    ev : Int
    bi : K
    bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
    hF : Or (Eq (HPow.hPow ((algebraMap Int K) b) f.natDegree) 0) (Eq (Polynomial. …
    Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
    F0 : Eq ev 0
    ⊢ Eq (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) f)) 0
  -/
  cases' hF with hF hF
    /-
      case intro.intro.intro.inl
      K : Type u_1
      inst✝ : LinearOrderedField K
      f : Polynomial Int
      a b : Int
      b0 : LT.lt 0 b
      fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
      ev : Int
      bi : K
      bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
      Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
      F0 : Eq ev 0
      hF : Eq (HPow.hPow ((algebraMap Int K) b) f.natDegree) 0
      ⊢ Eq (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) f)) 0
    -/
  · exact (not_le.mpr b0 (le_of_eq (Int.cast_eq_zero.mp (pow_eq_zero hF)))).elim
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.inr
      K : Type u_1
      inst✝ : LinearOrderedField K
      f : Polynomial Int
      a b : Int
      b0 : LT.lt 0 b
      fab : Ne (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) …
      ev : Int
      bi : K
      bu : Eq (HMul.hMul bi ((algebraMap Int K) b)) 1
      Fa : Eq (abs ↑ev) (HMul.hMul (HPow.hPow (↑b) f.natDegree) (abs (Polynomial.eva …
      F0 : Eq ev 0
      hF : Eq (Polynomial.eval (HMul.hMul ((algebraMap Int K) a) (Inv.inv ((algebraM …
      ⊢ Eq (Polynomial.eval (HDiv.hDiv ↑a ↑b) (Polynomial.map (algebraMap Int K) f)) 0
    -/
  · rwa [div_eq_mul_inv]
    /-
      🎉 no goals
    -/

