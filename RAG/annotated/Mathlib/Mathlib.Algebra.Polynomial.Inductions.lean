/-- `divX p` returns a polynomial `q` such that `q * X + C (p.coeff 0) = p`.
  It can be used in a semiring where the usual division algorithm is not possible -/
def divX (p : R[X]) : R[X] :=
  ⟨AddMonoidAlgebra.divOf p.toFinsupp 1⟩


@[simp]
theorem coeff_divX : (divX p).coeff n = p.coeff (n + 1) := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (p.divX.coeff n) (p.coeff (HAdd.hAdd n 1))
  -/
  rw [add_comm]; cases p; rfl
                          /-
                            🎉 no goals
                          -/


theorem divX_mul_X_add (p : R[X]) : divX p * X + C (p.coeff 0) = p :=
            /-
              R : Type u
              inst✝ : Semiring R
              p : Polynomial R
              ⊢ ∀ (n : Nat), Eq ((HAdd.hAdd (HMul.hMul p.divX Polynomial.X) (Polynomial.C (p …
            -/
                               /-
                                 🎉 no goals
                               -/
  ext <| by rintro ⟨_ | _⟩ <;> simp [coeff_C, Nat.succ_ne_zero, coeff_mul_X]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem X_mul_divX_add (p : R[X]) : X * divX p + C (p.coeff 0) = p :=
            /-
              R : Type u
              inst✝ : Semiring R
              p : Polynomial R
              ⊢ ∀ (n : Nat), Eq ((HAdd.hAdd (HMul.hMul Polynomial.X p.divX) (Polynomial.C (p …
            -/
                               /-
                                 🎉 no goals
                               -/
  ext <| by rintro ⟨_ | _⟩ <;> simp [coeff_C, Nat.succ_ne_zero, coeff_mul_X]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem divX_C (a : R) : divX (C a) = 0 :=
                  /-
                    R : Type u
                    inst✝ : Semiring R
                    a : R
                    n : Nat
                    ⊢ Eq ((Polynomial.C a).divX.coeff n) (Polynomial.coeff 0 n)
                  -/
  ext fun n => by simp [coeff_divX, coeff_C, Finsupp.single_eq_of_ne _]
                  /-
                    🎉 no goals
                  -/


theorem divX_eq_zero_iff : divX p = 0 ↔ p = C (p.coeff 0) :=
               /-
                 R : Type u
                 inst✝ : Semiring R
                 p : Polynomial R
                 h : Eq p.divX 0
                 ⊢ Eq p (Polynomial.C (p.coeff 0))
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simpa [eq_comm, h] using divX_mul_X_add p, fun h => by rw [h, divX_C]⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem divX_add : divX (p + q) = divX p + divX q :=
            /-
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              ⊢ ∀ (n : Nat), Eq ((HAdd.hAdd p q).divX.coeff n) ((HAdd.hAdd p.divX q.divX).co …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem divX_zero : divX (0 : R[X]) = 0 := leadingCoeff_eq_zero.mp rfl


@[simp]
theorem divX_one : divX (1 : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.divX 1) 0
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    n✝ : Nat
    ⊢ Eq ((Polynomial.divX 1).coeff n✝) (Polynomial.coeff 0 n✝)
  -/
  simpa only [coeff_divX, coeff_zero] using coeff_one
  /-
    🎉 no goals
  -/


@[simp]
theorem divX_C_mul : divX (C a * p) = C a * divX p := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).divX (HMul.hMul (Polynomial.C a) p.divX)
  -/
  ext
  /-
    case a
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    n✝ : Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C a) p).divX.coeff n✝) ((HMul.hMul (Polynomial.C  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem divX_X_pow : divX (X ^ n : R[X]) = if (n = 0) then 0 else X ^ (n - 1) := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    ⊢ Eq (HPow.hPow Polynomial.X n).divX (ite (Eq n 0) 0 (HPow.hPow Polynomial.X ( …
  -/
  cases n
    /-
      case zero
      R : Type u
      inst✝ : Semiring R
      ⊢ Eq (HPow.hPow Polynomial.X 0).divX (ite (Eq 0 0) 0 (HPow.hPow Polynomial.X ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      n✝ : Nat
      ⊢ Eq (HPow.hPow Polynomial.X (HAdd.hAdd n✝ 1)).divX (ite (Eq (HAdd.hAdd n✝ 1)  …
    -/
  · ext n
    /-
      case succ.a
      R : Type u
      inst✝ : Semiring R
      n✝ n : Nat
      ⊢ Eq ((HPow.hPow Polynomial.X (HAdd.hAdd n✝ 1)).divX.coeff n) ((ite (Eq (HAdd. …
    -/
    simp [coeff_X_pow]
    /-
      🎉 no goals
    -/


/-- `divX` as an additive homomorphism. -/
noncomputable
def divX_hom : R[X] →+ R[X] :=
  { toFun := divX
    map_zero' := divX_zero
    map_add' := fun _ _ => divX_add }


@[simp] theorem divX_hom_toFun : divX_hom p = divX p := rfl


theorem natDegree_divX_eq_natDegree_tsub_one : p.divX.natDegree = p.natDegree - 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq p.divX.natDegree (HSub.hSub p.natDegree 1)
  -/
  apply map_natDegree_eq_sub (φ := divX_hom)
    /-
      case φ_k
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (f : Polynomial R), LT.lt f.natDegree 1 → Eq (Polynomial.divX_hom f) 0
    -/
  · intro f
    /-
      case φ_k
      R : Type u
      inst✝ : Semiring R
      p f : Polynomial R
      ⊢ LT.lt f.natDegree 1 → Eq (Polynomial.divX_hom f) 0
    -/
    simpa [divX_hom, divX_eq_zero_iff] using eq_C_of_natDegree_eq_zero
    /-
      🎉 no goals
    -/
    /-
      case φ_mon
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (n : Nat) (c : R), Ne c 0 → Eq (Polynomial.divX_hom ((Polynomial.monomial  …
    -/
  · intros n c c0
    /-
      case φ_mon
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      c : R
      c0 : Ne c 0
      ⊢ Eq (Polynomial.divX_hom ((Polynomial.monomial n) c)).natDegree (HSub.hSub n 1)
    -/
    rw [← C_mul_X_pow_eq_monomial, divX_hom_toFun, divX_C_mul, divX_X_pow]
    /-
      case φ_mon
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      c : R
      c0 : Ne c 0
      ⊢ Eq (HMul.hMul (Polynomial.C c) (ite (Eq n 0) 0 (HPow.hPow Polynomial.X (HSub …
    -/
    split_ifs with n0
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        c : R
        c0 : Ne c 0
        n0 : Eq n 0
        ⊢ Eq (HMul.hMul (Polynomial.C c) 0).natDegree (HSub.hSub n 1)
      -/
    · simp [n0]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        c : R
        c0 : Ne c 0
        n0 : Not (Eq n 0)
        ⊢ Eq (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X (HSub.hSub n 1))).nat …
      -/
    · exact natDegree_C_mul_X_pow (n - 1) c c0
      /-
        🎉 no goals
      -/


theorem natDegree_divX_le : p.divX.natDegree ≤ p.natDegree :=
  natDegree_divX_eq_natDegree_tsub_one.trans_le (Nat.pred_le _)


theorem divX_C_mul_X_pow : divX (C a * X ^ n) = if n = 0 then 0 else C a * X ^ (n - 1) := by
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).divX (ite (Eq n 0 …
  -/
  simp only [divX_C_mul, divX_X_pow, mul_ite, mul_zero]
  /-
    🎉 no goals
  -/


theorem degree_divX_lt (hp0 : p ≠ 0) : (divX p).degree < p.degree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp0 : Ne p 0
    ⊢ LT.lt p.divX.degree p.degree
  -/
  haveI := Nontrivial.of_polynomial_ne hp0
  calc
    degree (divX p) < (divX p * X + C (p.coeff 0)).degree :=
      if h : degree p ≤ 0 then by
        have h' : C (p.coeff 0) ≠ 0 := by rwa [← eq_C_of_degree_le_zero h]
        rw [eq_C_of_degree_le_zero h, divX_C, degree_zero, zero_mul, zero_add]
        exact lt_of_le_of_ne bot_le (Ne.symm (mt degree_eq_bot.1 <| by simpa using h'))
      else by
        have hXp0 : divX p ≠ 0 := by
          simpa [divX_eq_zero_iff, -not_le, degree_le_zero_iff] using h
        have : leadingCoeff (divX p) * leadingCoeff X ≠ 0 := by simpa
        have : degree (C (p.coeff 0)) < degree (divX p * X) :=
          calc
            degree (C (p.coeff 0)) ≤ 0 := degree_C_le
            _ < 1 := by decide
            _ = degree (X : R[X]) := degree_X.symm
            _ ≤ degree (divX p * X) := by
              rw [← zero_add (degree X), degree_mul' this]
              exact add_le_add
                (by rw [zero_le_degree_iff, Ne, divX_eq_zero_iff]
                    exact fun h0 => h (h0.symm ▸ degree_C_le))
                    le_rfl
        rw [degree_add_eq_left_of_degree_lt this]; exact degree_lt_degree_mul_X hXp0
    _ = degree p := congr_arg _ (divX_mul_X_add _)


/-- An induction principle for polynomials, valued in Sort* instead of Prop. -/
@[elab_as_elim]
noncomputable def recOnHorner {M : R[X] → Sort*} (p : R[X]) (M0 : M 0)
    (MC : ∀ p a, coeff p 0 = 0 → a ≠ 0 → M p → M (p + C a))
    (MX : ∀ p, p ≠ 0 → M p → M (p * X)) : M p :=
  letI := Classical.decEq R
  if hp : p = 0 then hp ▸ M0
  else by
    /-
      R : Type u
      S : Type v
      T : Type w
      A : Type z
      a b : R
      n : Nat
      inst✝ : Semiring R
      p✝ q : Polynomial R
      M : Polynomial R → Sort u_1
      p : Polynomial R
      M0 : M 0
      MC : (p : Polynomial R) → (a : R) → Eq (p.coeff 0) 0 → Ne a 0 → M p → M (HAdd. …
      MX : (p : Polynomial R) → Ne p 0 → M p → M (HMul.hMul p Polynomial.X)
      this : DecidableEq R := Classical.decEq R
      hp : Not (Eq p 0)
      ⊢ M p
    -/
    have wf : degree (divX p) < degree p := degree_divX_lt hp
    /-
      R : Type u
      S : Type v
      T : Type w
      A : Type z
      a b : R
      n : Nat
      inst✝ : Semiring R
      p✝ q : Polynomial R
      M : Polynomial R → Sort u_1
      p : Polynomial R
      M0 : M 0
      MC : (p : Polynomial R) → (a : R) → Eq (p.coeff 0) 0 → Ne a 0 → M p → M (HAdd. …
      MX : (p : Polynomial R) → Ne p 0 → M p → M (HMul.hMul p Polynomial.X)
      this : DecidableEq R := Classical.decEq R
      hp : Not (Eq p 0)
      wf : LT.lt p.divX.degree p.degree
      ⊢ M p
    -/
    rw [← divX_mul_X_add p] at *
    exact
      if hcp0 : coeff p 0 = 0 then by
        rw [hcp0, C_0, add_zero]
        exact
          MX _ (fun h : divX p = 0 => by simp [h, hcp0] at hp) (recOnHorner (divX p) M0 MC MX)
      else
        MC _ _ (coeff_mul_X_zero _) hcp0
          (if hpX0 : divX p = 0 then show M (divX p * X) by rw [hpX0, zero_mul]; exact M0
          else MX (divX p) hpX0 (recOnHorner _ M0 MC MX))
termination_by p.degree


/-- A property holds for all polynomials of positive `degree` with coefficients in a semiring `R`
if it holds for
* `a * X`, with `a ∈ R`,
* `p * X`, with `p ∈ R[X]`,
* `p + a`, with `a ∈ R`, `p ∈ R[X]`,
with appropriate restrictions on each term.

See `natDegree_ne_zero_induction_on` for a similar statement involving no explicit multiplication.
 -/
@[elab_as_elim]
theorem degree_pos_induction_on {P : R[X] → Prop} (p : R[X]) (h0 : 0 < degree p)
    (hC : ∀ {a}, a ≠ 0 → P (C a * X)) (hX : ∀ {p}, 0 < degree p → P p → P (p * X))
    (hadd : ∀ {p} {a}, 0 < degree p → P p → P (p + C a)) : P p :=
                             /-
                               R : Type u
                               inst✝ : Semiring R
                               P : Polynomial R → Prop
                               p : Polynomial R
                               h0 : LT.lt 0 p.degree
                               hC : ∀ {a : R}, Ne a 0 → P (HMul.hMul (Polynomial.C a) Polynomial.X)
                               hX : ∀ {p : Polynomial R}, LT.lt 0 p.degree → P p → P (HMul.hMul p Polynomial.X)
                               hadd : ∀ {p : Polynomial R} {a : R}, LT.lt 0 p.degree → P p → P (HAdd.hAdd p ( …
                               h : LT.lt 0 (Polynomial.degree 0)
                               ⊢ P 0
                             -/
  recOnHorner p (fun h => by rw [degree_zero] at h; exact absurd h (by decide))
                                                    /-
                                                      🎉 no goals
                                                    -/
    (fun p a heq0 _ ih h0 =>
      (have : 0 < degree p :=
        (lt_of_not_ge fun h =>
          not_lt_of_ge (degree_C_le (a := a)) <|
               /-
                 R : Type u
                 inst✝ : Semiring R
                 P : Polynomial R → Prop
                 p✝ : Polynomial R
                 h0✝ : LT.lt 0 p✝.degree
                 hC : ∀ {a : R}, Ne a 0 → P (HMul.hMul (Polynomial.C a) Polynomial.X)
                 hX : ∀ {p : Polynomial R}, LT.lt 0 p.degree → P p → P (HMul.hMul p Polynomial.X)
                 hadd : ∀ {p : Polynomial R} {a : R}, LT.lt 0 p.degree → P p → P (HAdd.hAdd p ( …
                 p : Polynomial R
                 a : R
                 heq0 : Eq (p.coeff 0) 0
                 x✝ : Ne a 0
                 ih : LT.lt 0 p.degree → P p
                 h0 : LT.lt 0 (HAdd.hAdd p (Polynomial.C a)).degree
                 h : GE.ge 0 p.degree
                 ⊢ LT.lt 0 (Polynomial.C a).degree
               -/
            by rwa [eq_C_of_degree_le_zero h, ← C_add,heq0,zero_add] at h0)
               /-
                 🎉 no goals
               -/
      hadd this (ih this)))
    (fun p _ ih h0' =>
      if h0 : 0 < degree p then hX h0 (ih h0)
      else by
        /-
          R : Type u
          inst✝ : Semiring R
          P : Polynomial R → Prop
          p✝ : Polynomial R
          h0✝ : LT.lt 0 p✝.degree
          hC : ∀ {a : R}, Ne a 0 → P (HMul.hMul (Polynomial.C a) Polynomial.X)
          hX : ∀ {p : Polynomial R}, LT.lt 0 p.degree → P p → P (HMul.hMul p Polynomial.X)
          hadd : ∀ {p : Polynomial R} {a : R}, LT.lt 0 p.degree → P p → P (HAdd.hAdd p ( …
          p : Polynomial R
          x✝ : Ne p 0
          ih : LT.lt 0 p.degree → P p
          h0' : LT.lt 0 (HMul.hMul p Polynomial.X).degree
          h0 : Not (LT.lt 0 p.degree)
          ⊢ P (HMul.hMul p Polynomial.X)
        -/
        rw [eq_C_of_degree_le_zero (le_of_not_gt h0)] at h0' ⊢
        /-
          R : Type u
          inst✝ : Semiring R
          P : Polynomial R → Prop
          p✝ : Polynomial R
          h0✝ : LT.lt 0 p✝.degree
          hC : ∀ {a : R}, Ne a 0 → P (HMul.hMul (Polynomial.C a) Polynomial.X)
          hX : ∀ {p : Polynomial R}, LT.lt 0 p.degree → P p → P (HMul.hMul p Polynomial.X)
          hadd : ∀ {p : Polynomial R} {a : R}, LT.lt 0 p.degree → P p → P (HAdd.hAdd p ( …
          p : Polynomial R
          x✝ : Ne p 0
          ih : LT.lt 0 p.degree → P p
          h0' : LT.lt 0 (HMul.hMul (Polynomial.C (p.coeff 0)) Polynomial.X).degree
          h0 : Not (LT.lt 0 p.degree)
          ⊢ P (HMul.hMul (Polynomial.C (p.coeff 0)) Polynomial.X)
        -/
        exact hC fun h : coeff p 0 = 0 => by simp [h, Nat.not_lt_zero] at h0')
        /-
          🎉 no goals
        -/
    h0


/-- A property holds for all polynomials of non-zero `natDegree` with coefficients in a
semiring `R` if it holds for
* `p + a`, with `a ∈ R`, `p ∈ R[X]`,
* `p + q`, with `p, q ∈ R[X]`,
* monomials with nonzero coefficient and non-zero exponent,
with appropriate restrictions on each term.
Note that multiplication is "hidden" in the assumption on monomials, so there is no explicit
multiplication in the statement.
See `degree_pos_induction_on` for a similar statement involving more explicit multiplications.
 -/
@[elab_as_elim]
theorem natDegree_ne_zero_induction_on {M : R[X] → Prop} {f : R[X]} (f0 : f.natDegree ≠ 0)
    (h_C_add : ∀ {a p}, M p → M (C a + p)) (h_add : ∀ {p q}, M p → M q → M (p + q))
    (h_monomial : ∀ {n : ℕ} {a : R}, a ≠ 0 → n ≠ 0 → M (monomial n a)) : M f := by
  /-
    R : Type u
    inst✝ : Semiring R
    M : Polynomial R → Prop
    f : Polynomial R
    f0 : Ne f.natDegree 0
    h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
    h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
    ⊢ M f
  -/
  suffices f.natDegree = 0 ∨ M f from Or.recOn this (fun h => (f0 h).elim) id
  /-
    R : Type u
    inst✝ : Semiring R
    M : Polynomial R → Prop
    f : Polynomial R
    f0 : Ne f.natDegree 0
    h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
    h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
    ⊢ Or (Eq f.natDegree 0) (M f)
  -/
  refine Polynomial.induction_on f ?_ ?_ ?_
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      M : Polynomial R → Prop
      f : Polynomial R
      f0 : Ne f.natDegree 0
      h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
      h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
      ⊢ ∀ (a : R), Or (Eq (Polynomial.C a).natDegree 0) (M (Polynomial.C a))
    -/
  · exact fun a => Or.inl (natDegree_C _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : Semiring R
      M : Polynomial R → Prop
      f : Polynomial R
      f0 : Ne f.natDegree 0
      h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
      h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
      ⊢ ∀ (p q : Polynomial R), Or (Eq p.natDegree 0) (M p) → Or (Eq q.natDegree 0)  …
    -/
  · rintro p q (hp | hp) (hq | hq)
      /-
        case refine_2.inl.inl
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : Eq p.natDegree 0
        hq : Eq q.natDegree 0
        ⊢ Or (Eq (HAdd.hAdd p q).natDegree 0) (M (HAdd.hAdd p q))
      -/
    · refine Or.inl ?_
      /-
        case refine_2.inl.inl
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : Eq p.natDegree 0
        hq : Eq q.natDegree 0
        ⊢ Eq (HAdd.hAdd p q).natDegree 0
      -/
      rw [eq_C_of_natDegree_eq_zero hp, eq_C_of_natDegree_eq_zero hq, ← C_add, natDegree_C]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inl.inr
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : Eq p.natDegree 0
        hq : M q
        ⊢ Or (Eq (HAdd.hAdd p q).natDegree 0) (M (HAdd.hAdd p q))
      -/
    · refine Or.inr ?_
      /-
        case refine_2.inl.inr
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : Eq p.natDegree 0
        hq : M q
        ⊢ M (HAdd.hAdd p q)
      -/
      rw [eq_C_of_natDegree_eq_zero hp]
      /-
        case refine_2.inl.inr
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : Eq p.natDegree 0
        hq : M q
        ⊢ M (HAdd.hAdd (Polynomial.C (p.coeff 0)) q)
      -/
      exact h_C_add hq
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inl
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : M p
        hq : Eq q.natDegree 0
        ⊢ Or (Eq (HAdd.hAdd p q).natDegree 0) (M (HAdd.hAdd p q))
      -/
    · refine Or.inr ?_
      /-
        case refine_2.inr.inl
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : M p
        hq : Eq q.natDegree 0
        ⊢ M (HAdd.hAdd p q)
      -/
      rw [eq_C_of_natDegree_eq_zero hq, add_comm]
      /-
        case refine_2.inr.inl
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : M p
        hq : Eq q.natDegree 0
        ⊢ M (HAdd.hAdd (Polynomial.C (q.coeff 0)) p)
      -/
      exact h_C_add hp
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inr
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        p q : Polynomial R
        hp : M p
        hq : M q
        ⊢ Or (Eq (HAdd.hAdd p q).natDegree 0) (M (HAdd.hAdd p q))
      -/
    · exact Or.inr (h_add hp hq)
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      R : Type u
      inst✝ : Semiring R
      M : Polynomial R → Prop
      f : Polynomial R
      f0 : Ne f.natDegree 0
      h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
      h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
      ⊢ ∀ (n : Nat) (a : R), Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomia …
    -/
  · intro n a _
    /-
      case refine_3
      R : Type u
      inst✝ : Semiring R
      M : Polynomial R → Prop
      f : Polynomial R
      f0 : Ne f.natDegree 0
      h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
      h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
      n : Nat
      a : R
      a✝ : Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).natDegree  …
      ⊢ Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1))) …
    -/
    by_cases a0 : a = 0
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        n : Nat
        a : R
        a✝ : Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).natDegree  …
        a0 : Eq a 0
        ⊢ Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1))) …
      -/
    · exact Or.inl (by rw [a0, C_0, zero_mul, natDegree_zero])
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        n : Nat
        a : R
        a✝ : Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).natDegree  …
        a0 : Not (Eq a 0)
        ⊢ Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1))) …
      -/
    · refine Or.inr ?_
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        n : Nat
        a : R
        a✝ : Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).natDegree  …
        a0 : Not (Eq a 0)
        ⊢ M (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1)))
      -/
      rw [C_mul_X_pow_eq_monomial]
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        M : Polynomial R → Prop
        f : Polynomial R
        f0 : Ne f.natDegree 0
        h_C_add : ∀ {a : R} {p : Polynomial R}, M p → M (HAdd.hAdd (Polynomial.C a) p)
        h_add : ∀ {p q : Polynomial R}, M p → M q → M (HAdd.hAdd p q)
        h_monomial : ∀ {n : Nat} {a : R}, Ne a 0 → Ne n 0 → M ((Polynomial.monomial n) …
        n : Nat
        a : R
        a✝ : Or (Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).natDegree  …
        a0 : Not (Eq a 0)
        ⊢ M ((Polynomial.monomial (HAdd.hAdd n 1)) a)
      -/
      exact h_monomial a0 n.succ_ne_zero
      /-
        🎉 no goals
      -/


