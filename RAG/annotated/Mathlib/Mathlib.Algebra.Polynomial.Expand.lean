/-- Expand the polynomial by a factor of p, so `∑ aₙ xⁿ` becomes `∑ aₙ xⁿᵖ`. -/
noncomputable def expand : R[X] →ₐ[R] R[X] :=
  { (eval₂RingHom C (X ^ p) : R[X] →+* R[X]) with commutes' := fun _ => eval₂_C _ _ }


theorem coe_expand : (expand R p : R[X] → R[X]) = eval₂ C (X ^ p) :=
  rfl


theorem expand_eq_comp_X_pow {f : R[X]} : expand R p f = f.comp (X ^ p) := rfl


theorem expand_eq_sum {f : R[X]} : expand R p f = f.sum fun e a => C a * (X ^ p) ^ e := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    ⊢ Eq ((Polynomial.expand R p) f) (f.sum fun e a => HMul.hMul (Polynomial.C a)  …
  -/
  simp [expand, eval₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem expand_C (r : R) : expand R p (C r) = C r :=
  eval₂_C _ _


@[simp]
theorem expand_X : expand R p X = X ^ p :=
  eval₂_X _ _


@[simp]
theorem expand_monomial (r : R) : expand R p (monomial q r) = monomial (q * p) r := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Nat
    r : R
    ⊢ Eq ((Polynomial.expand R p) ((Polynomial.monomial q) r)) ((Polynomial.monomi …
  -/
  simp_rw [← smul_X_eq_monomial, map_smul, map_pow, expand_X, mul_comm, pow_mul]
  /-
    🎉 no goals
  -/


theorem expand_expand (f : R[X]) : expand R p (expand R q f) = expand R (p * q) f :=
                                         /-
                                           R : Type u
                                           inst✝ : CommSemiring R
                                           p q : Nat
                                           f : Polynomial R
                                           r : R
                                           ⊢ Eq ((Polynomial.expand R p) ((Polynomial.expand R q) (Polynomial.C r))) ((Po …
                                         -/
  Polynomial.induction_on f (fun r => by simp_rw [expand_C])
                                         /-
                                           🎉 no goals
                                         -/
                           /-
                             R : Type u
                             inst✝ : CommSemiring R
                             p q : Nat
                             f✝ f g : Polynomial R
                             ihf : Eq ((Polynomial.expand R p) ((Polynomial.expand R q) f)) ((Polynomial.ex …
                             ihg : Eq ((Polynomial.expand R p) ((Polynomial.expand R q) g)) ((Polynomial.ex …
                             ⊢ Eq ((Polynomial.expand R p) ((Polynomial.expand R q) (HAdd.hAdd f g))) ((Pol …
                           -/
    (fun f g ihf ihg => by simp_rw [map_add, ihf, ihg]) fun n r _ => by
                           /-
                             🎉 no goals
                           -/
    /-
      R : Type u
      inst✝ : CommSemiring R
      p q : Nat
      f : Polynomial R
      n : Nat
      r : R
      x✝ : Eq ((Polynomial.expand R p) ((Polynomial.expand R q) (HMul.hMul (Polynomi …
      ⊢ Eq ((Polynomial.expand R p) ((Polynomial.expand R q) (HMul.hMul (Polynomial. …
    -/
    simp_rw [map_mul, expand_C, map_pow, expand_X, map_pow, expand_X, pow_mul]
    /-
      🎉 no goals
    -/


theorem expand_mul (f : R[X]) : expand R (p * q) f = expand R p (expand R q f) :=
  (expand_expand p q f).symm


@[simp]
                                                                   /-
                                                                     R : Type u
                                                                     inst✝ : CommSemiring R
                                                                     f : Polynomial R
                                                                     ⊢ Eq ((Polynomial.expand R 0) f) (Polynomial.C (Polynomial.eval 1 f))
                                                                   -/
theorem expand_zero (f : R[X]) : expand R 0 f = C (eval 1 f) := by simp [expand]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem expand_one (f : R[X]) : expand R 1 f = f :=
                                         /-
                                           R : Type u
                                           inst✝ : CommSemiring R
                                           f : Polynomial R
                                           r : R
                                           ⊢ Eq ((Polynomial.expand R 1) (Polynomial.C r)) (Polynomial.C r)
                                         -/
  Polynomial.induction_on f (fun r => by rw [expand_C])
                                         /-
                                           🎉 no goals
                                         -/
                           /-
                             R : Type u
                             inst✝ : CommSemiring R
                             f✝ f g : Polynomial R
                             ihf : Eq ((Polynomial.expand R 1) f) f
                             ihg : Eq ((Polynomial.expand R 1) g) g
                             ⊢ Eq ((Polynomial.expand R 1) (HAdd.hAdd f g)) (HAdd.hAdd f g)
                           -/
    (fun f g ihf ihg => by rw [map_add, ihf, ihg]) fun n r _ => by
                           /-
                             🎉 no goals
                           -/
    /-
      R : Type u
      inst✝ : CommSemiring R
      f : Polynomial R
      n : Nat
      r : R
      x✝ : Eq ((Polynomial.expand R 1) (HMul.hMul (Polynomial.C r) (HPow.hPow Polyno …
      ⊢ Eq ((Polynomial.expand R 1) (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomia …
    -/
    rw [map_mul, expand_C, map_pow, expand_X, pow_one]
    /-
      🎉 no goals
    -/


theorem expand_pow (f : R[X]) : expand R (p ^ q) f = (expand R p)^[q] f :=
                  /-
                    R : Type u
                    inst✝ : CommSemiring R
                    p q : Nat
                    f : Polynomial R
                    ⊢ Eq ((Polynomial.expand R (HPow.hPow p Nat.zero)) f) (Nat.iterate (⇑(Polynomi …
                  -/
  Nat.recOn q (by rw [pow_zero, expand_one, Function.iterate_zero, id]) fun n ih => by
                  /-
                    🎉 no goals
                  -/
    /-
      R : Type u
      inst✝ : CommSemiring R
      p q : Nat
      f : Polynomial R
      n : Nat
      ih : Eq ((Polynomial.expand R (HPow.hPow p n)) f) (Nat.iterate (⇑(Polynomial.e …
      ⊢ Eq ((Polynomial.expand R (HPow.hPow p n.succ)) f) (Nat.iterate (⇑(Polynomial …
    -/
    rw [Function.iterate_succ_apply', pow_succ', expand_mul, ih]
    /-
      🎉 no goals
    -/


theorem derivative_expand (f : R[X]) : Polynomial.derivative (expand R p f) =
    expand R p (Polynomial.derivative f) * (p * (X ^ (p - 1) : R[X])) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    ⊢ Eq (Polynomial.derivative ((Polynomial.expand R p) f)) (HMul.hMul ((Polynomi …
  -/
  rw [coe_expand, derivative_eval₂_C, derivative_pow, C_eq_natCast, derivative_X, mul_one]
  /-
    🎉 no goals
  -/


theorem coeff_expand {p : ℕ} (hp : 0 < p) (f : R[X]) (n : ℕ) :
    (expand R p f).coeff n = if p ∣ n then f.coeff (n / p) else 0 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    n : Nat
    ⊢ Eq (((Polynomial.expand R p) f).coeff n) (ite (Dvd.dvd p n) (f.coeff (HDiv.h …
  -/
  simp only [expand_eq_sum]
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    n : Nat
    ⊢ Eq ((f.sum fun e a => HMul.hMul (Polynomial.C a) (HPow.hPow (HPow.hPow Polyn …
  -/
  simp_rw [coeff_sum, ← pow_mul, C_mul_X_pow_eq_monomial, coeff_monomial, sum]
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    n : Nat
    ⊢ Eq (f.support.sum fun x => ite (Eq (HMul.hMul p x) n) (f.coeff x) 0) (ite (D …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 0 p
      f : Polynomial R
      n : Nat
      h : Dvd.dvd p n
      ⊢ Eq (f.support.sum fun x => ite (Eq (HMul.hMul p x) n) (f.coeff x) 0) (f.coef …
    -/
  · rw [Finset.sum_eq_single (n / p), Nat.mul_div_cancel' h, if_pos rfl]
      /-
        case pos.h₀
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        ⊢ ∀ (b : Nat), Membership.mem f.support b → Ne b (HDiv.hDiv n p) → Eq (ite (Eq …
      -/
    · intro b _ hb2
      /-
        case pos.h₀
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        b : Nat
        a✝ : Membership.mem f.support b
        hb2 : Ne b (HDiv.hDiv n p)
        ⊢ Eq (ite (Eq (HMul.hMul p b) n) (f.coeff b) 0) 0
      -/
      rw [if_neg]
      /-
        case pos.h₀.hnc
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        b : Nat
        a✝ : Membership.mem f.support b
        hb2 : Ne b (HDiv.hDiv n p)
        ⊢ Not (Eq (HMul.hMul p b) n)
      -/
      intro hb3
      /-
        case pos.h₀.hnc
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        b : Nat
        a✝ : Membership.mem f.support b
        hb2 : Ne b (HDiv.hDiv n p)
        hb3 : Eq (HMul.hMul p b) n
        ⊢ False
      -/
      apply hb2
      /-
        case pos.h₀.hnc
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        b : Nat
        a✝ : Membership.mem f.support b
        hb2 : Ne b (HDiv.hDiv n p)
        hb3 : Eq (HMul.hMul p b) n
        ⊢ Eq b (HDiv.hDiv n p)
      -/
      rw [← hb3, Nat.mul_div_cancel_left b hp]
      /-
        🎉 no goals
      -/
      /-
        case pos.h₁
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        ⊢ Not (Membership.mem f.support (HDiv.hDiv n p)) → Eq (ite (Eq (HMul.hMul p (H …
      -/
    · intro hn
      /-
        case pos.h₁
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        hn : Not (Membership.mem f.support (HDiv.hDiv n p))
        ⊢ Eq (ite (Eq (HMul.hMul p (HDiv.hDiv n p)) n) (f.coeff (HDiv.hDiv n p)) 0) 0
      -/
      rw [not_mem_support_iff.1 hn]
      /-
        case pos.h₁
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        hp : LT.lt 0 p
        f : Polynomial R
        n : Nat
        h : Dvd.dvd p n
        hn : Not (Membership.mem f.support (HDiv.hDiv n p))
        ⊢ Eq (ite (Eq (HMul.hMul p (HDiv.hDiv n p)) n) 0 0) 0
      -/
                    /-
                      🎉 no goals
                    -/
      split_ifs <;> rfl
                    /-
                      🎉 no goals
                    -/
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 0 p
      f : Polynomial R
      n : Nat
      h : Not (Dvd.dvd p n)
      ⊢ Eq (f.support.sum fun x => ite (Eq (HMul.hMul p x) n) (f.coeff x) 0) 0
    -/
  · rw [Finset.sum_eq_zero]
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 0 p
      f : Polynomial R
      n : Nat
      h : Not (Dvd.dvd p n)
      ⊢ ∀ (x : Nat), Membership.mem f.support x → Eq (ite (Eq (HMul.hMul p x) n) (f. …
    -/
    intro k _
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 0 p
      f : Polynomial R
      n : Nat
      h : Not (Dvd.dvd p n)
      k : Nat
      a✝ : Membership.mem f.support k
      ⊢ Eq (ite (Eq (HMul.hMul p k) n) (f.coeff k) 0) 0
    -/
    rw [if_neg]
    /-
      case neg.hnc
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 0 p
      f : Polynomial R
      n : Nat
      h : Not (Dvd.dvd p n)
      k : Nat
      a✝ : Membership.mem f.support k
      ⊢ Not (Eq (HMul.hMul p k) n)
    -/
    exact fun hkn => h ⟨k, hkn.symm⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coeff_expand_mul {p : ℕ} (hp : 0 < p) (f : R[X]) (n : ℕ) :
    (expand R p f).coeff (n * p) = f.coeff n := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    n : Nat
    ⊢ Eq (((Polynomial.expand R p) f).coeff (HMul.hMul n p)) (f.coeff n)
  -/
  rw [coeff_expand hp, if_pos (dvd_mul_left _ _), Nat.mul_div_cancel _ hp]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_expand_mul' {p : ℕ} (hp : 0 < p) (f : R[X]) (n : ℕ) :
                                                   /-
                                                     R : Type u
                                                     inst✝ : CommSemiring R
                                                     p : Nat
                                                     hp : LT.lt 0 p
                                                     f : Polynomial R
                                                     n : Nat
                                                     ⊢ Eq (((Polynomial.expand R p) f).coeff (HMul.hMul p n)) (f.coeff n)
                                                   -/
    (expand R p f).coeff (p * n) = f.coeff n := by rw [mul_comm, coeff_expand_mul hp]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Expansion is injective. -/
theorem expand_injective {n : ℕ} (hn : 0 < n) : Function.Injective (expand R n) := fun g g' H =>
                  /-
                    R : Type u
                    inst✝ : CommSemiring R
                    n : Nat
                    hn : LT.lt 0 n
                    g g' : Polynomial R
                    H : Eq ((Polynomial.expand R n) g) ((Polynomial.expand R n) g')
                    k : Nat
                    ⊢ Eq (g.coeff k) (g'.coeff k)
                  -/
  ext fun k => by rw [← coeff_expand_mul hn, H, coeff_expand_mul hn]
                  /-
                    🎉 no goals
                  -/


theorem expand_inj {p : ℕ} (hp : 0 < p) {f g : R[X]} : expand R p f = expand R p g ↔ f = g :=
  (expand_injective hp).eq_iff


theorem expand_eq_zero {p : ℕ} (hp : 0 < p) {f : R[X]} : expand R p f = 0 ↔ f = 0 :=
  (expand_injective hp).eq_iff' (map_zero _)


theorem expand_ne_zero {p : ℕ} (hp : 0 < p) {f : R[X]} : expand R p f ≠ 0 ↔ f ≠ 0 :=
  (expand_eq_zero hp).not


theorem expand_eq_C {p : ℕ} (hp : 0 < p) {f : R[X]} {r : R} : expand R p f = C r ↔ f = C r := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    r : R
    ⊢ Iff (Eq ((Polynomial.expand R p) f) (Polynomial.C r)) (Eq f (Polynomial.C r))
  -/
  rw [← expand_C, expand_inj hp, expand_C]
  /-
    🎉 no goals
  -/


theorem natDegree_expand (p : ℕ) (f : R[X]) : (expand R p f).natDegree = f.natDegree * p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
  -/
  rcases p.eq_zero_or_pos with hp | hp
    /-
      case inl
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : Eq p 0
      ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
    -/
  · rw [hp, coe_expand, pow_zero, mul_zero, ← C_1, eval₂_hom, natDegree_C]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
  -/
  by_cases hf : f = 0
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Eq f 0
      ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
    -/
  · rw [hf, map_zero, natDegree_zero, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    hf : Not (Eq f 0)
    ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
  -/
  have hf1 : expand R p f ≠ 0 := mt (expand_eq_zero hp).1 hf
  /-
    case neg
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    hf : Not (Eq f 0)
    hf1 : Ne ((Polynomial.expand R p) f) 0
    ⊢ Eq ((Polynomial.expand R p) f).natDegree (HMul.hMul f.natDegree p)
  -/
  rw [← WithBot.coe_eq_coe]
  /-
    case neg
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    hf : Not (Eq f 0)
    hf1 : Ne ((Polynomial.expand R p) f) 0
    ⊢ Eq ↑((Polynomial.expand R p) f).natDegree ↑(HMul.hMul f.natDegree p)
  -/
  convert (degree_eq_natDegree hf1).symm -- Porting note: was `rw [degree_eq_natDegree hf1]`
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    hf : Not (Eq f 0)
    hf1 : Ne ((Polynomial.expand R p) f) 0
    ⊢ Eq (↑(HMul.hMul f.natDegree p)) ((Polynomial.expand R p) f).degree
  -/
  symm
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : GT.gt p 0
    hf : Not (Eq f 0)
    hf1 : Ne ((Polynomial.expand R p) f) 0
    ⊢ Eq ((Polynomial.expand R p) f).degree ↑(HMul.hMul f.natDegree p)
  -/
  refine le_antisymm ((degree_le_iff_coeff_zero _ _).2 fun n hn => ?_) ?_
    /-
      case h.e'_3.refine_1
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Not (Eq f 0)
      hf1 : Ne ((Polynomial.expand R p) f) 0
      n : Nat
      hn : LT.lt ↑(HMul.hMul f.natDegree p) ↑n
      ⊢ Eq (((Polynomial.expand R p) f).coeff n) 0
    -/
  · rw [coeff_expand hp]
    /-
      case h.e'_3.refine_1
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Not (Eq f 0)
      hf1 : Ne ((Polynomial.expand R p) f) 0
      n : Nat
      hn : LT.lt ↑(HMul.hMul f.natDegree p) ↑n
      ⊢ Eq (ite (Dvd.dvd p n) (f.coeff (HDiv.hDiv n p)) 0) 0
    -/
    split_ifs with hpn
      /-
        case pos
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        f : Polynomial R
        hp : GT.gt p 0
        hf : Not (Eq f 0)
        hf1 : Ne ((Polynomial.expand R p) f) 0
        n : Nat
        hn : LT.lt ↑(HMul.hMul f.natDegree p) ↑n
        hpn : Dvd.dvd p n
        ⊢ Eq (f.coeff (HDiv.hDiv n p)) 0
      -/
    · rw [coeff_eq_zero_of_natDegree_lt]
      /-
        case pos
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        f : Polynomial R
        hp : GT.gt p 0
        hf : Not (Eq f 0)
        hf1 : Ne ((Polynomial.expand R p) f) 0
        n : Nat
        hn : LT.lt ↑(HMul.hMul f.natDegree p) ↑n
        hpn : Dvd.dvd p n
        ⊢ LT.lt f.natDegree (HDiv.hDiv n p)
      -/
      contrapose! hn
      /-
        case pos
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        f : Polynomial R
        hp : GT.gt p 0
        hf : Not (Eq f 0)
        hf1 : Ne ((Polynomial.expand R p) f) 0
        n : Nat
        hpn : Dvd.dvd p n
        hn : LE.le (HDiv.hDiv n p) f.natDegree
        ⊢ LE.le ↑n ↑(HMul.hMul f.natDegree p)
      -/
      erw [WithBot.coe_le_coe, ← Nat.div_mul_cancel hpn]
      /-
        case pos
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        f : Polynomial R
        hp : GT.gt p 0
        hf : Not (Eq f 0)
        hf1 : Ne ((Polynomial.expand R p) f) 0
        n : Nat
        hpn : Dvd.dvd p n
        hn : LE.le (HDiv.hDiv n p) f.natDegree
        ⊢ LE.le (↑(HMul.hMul (HDiv.hDiv n p) p)) (HMul.hMul f.natDegree p)
      -/
      exact Nat.mul_le_mul_right p hn
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : CommSemiring R
        p : Nat
        f : Polynomial R
        hp : GT.gt p 0
        hf : Not (Eq f 0)
        hf1 : Ne ((Polynomial.expand R p) f) 0
        n : Nat
        hn : LT.lt ↑(HMul.hMul f.natDegree p) ↑n
        hpn : Not (Dvd.dvd p n)
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3.refine_2
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Not (Eq f 0)
      hf1 : Ne ((Polynomial.expand R p) f) 0
      ⊢ LE.le (↑(HMul.hMul f.natDegree p)) ((Polynomial.expand R p) f).degree
    -/
  · refine le_degree_of_ne_zero ?_
    /-
      case h.e'_3.refine_2
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Not (Eq f 0)
      hf1 : Ne ((Polynomial.expand R p) f) 0
      ⊢ Ne (((Polynomial.expand R p) f).coeff (Mul.mul f.natDegree p)) 0
    -/
    erw [coeff_expand_mul hp, ← leadingCoeff]
    /-
      case h.e'_3.refine_2
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      f : Polynomial R
      hp : GT.gt p 0
      hf : Not (Eq f 0)
      hf1 : Ne ((Polynomial.expand R p) f) 0
      ⊢ Ne f.leadingCoeff 0
    -/
    exact mt leadingCoeff_eq_zero.1 hf
    /-
      🎉 no goals
    -/


theorem leadingCoeff_expand {p : ℕ} {f : R[X]} (hp : 0 < p) :
    (expand R p f).leadingCoeff = f.leadingCoeff := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : LT.lt 0 p
    ⊢ Eq ((Polynomial.expand R p) f).leadingCoeff f.leadingCoeff
  -/
  simp_rw [leadingCoeff, natDegree_expand, coeff_expand_mul hp]
  /-
    🎉 no goals
  -/


theorem monic_expand_iff {p : ℕ} {f : R[X]} (hp : 0 < p) : (expand R p f).Monic ↔ f.Monic := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : LT.lt 0 p
    ⊢ Iff ((Polynomial.expand R p) f).Monic f.Monic
  -/
  simp only [Monic, leadingCoeff_expand hp]
  /-
    🎉 no goals
  -/


alias ⟨_, Monic.expand⟩ := monic_expand_iff


theorem map_expand {p : ℕ} {f : R →+* S} {q : R[X]} :
    map f (expand R p q) = expand S p (map f q) := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    S : Type v
    inst✝ : CommSemiring S
    p : Nat
    f : RingHom R S
    q : Polynomial R
    ⊢ Eq (Polynomial.map f ((Polynomial.expand R p) q)) ((Polynomial.expand S p) ( …
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝¹ : CommSemiring R
      S : Type v
      inst✝ : CommSemiring S
      p : Nat
      f : RingHom R S
      q : Polynomial R
      hp : Eq p 0
      ⊢ Eq (Polynomial.map f ((Polynomial.expand R p) q)) ((Polynomial.expand S p) ( …
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : CommSemiring R
    S : Type v
    inst✝ : CommSemiring S
    p : Nat
    f : RingHom R S
    q : Polynomial R
    hp : Not (Eq p 0)
    ⊢ Eq (Polynomial.map f ((Polynomial.expand R p) q)) ((Polynomial.expand S p) ( …
  -/
  ext
  /-
    case neg.a
    R : Type u
    inst✝¹ : CommSemiring R
    S : Type v
    inst✝ : CommSemiring S
    p : Nat
    f : RingHom R S
    q : Polynomial R
    hp : Not (Eq p 0)
    n✝ : Nat
    ⊢ Eq ((Polynomial.map f ((Polynomial.expand R p) q)).coeff n✝) (((Polynomial.e …
  -/
  rw [coeff_map, coeff_expand (Nat.pos_of_ne_zero hp), coeff_expand (Nat.pos_of_ne_zero hp)]
  /-
    case neg.a
    R : Type u
    inst✝¹ : CommSemiring R
    S : Type v
    inst✝ : CommSemiring S
    p : Nat
    f : RingHom R S
    q : Polynomial R
    hp : Not (Eq p 0)
    n✝ : Nat
    ⊢ Eq (f (ite (Dvd.dvd p n✝) (q.coeff (HDiv.hDiv n✝ p)) 0)) (ite (Dvd.dvd p n✝) …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


@[simp]
theorem expand_eval (p : ℕ) (P : R[X]) (r : R) : eval r (expand R p P) = eval (r ^ p) P := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    P : Polynomial R
    r : R
    ⊢ Eq (Polynomial.eval r ((Polynomial.expand R p) P)) (Polynomial.eval (HPow.hP …
  -/
  refine Polynomial.induction_on P (fun a => by simp) (fun f g hf hg => ?_) fun n a _ => by simp
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    P : Polynomial R
    r : R
    f g : Polynomial R
    hf : Eq (Polynomial.eval r ((Polynomial.expand R p) f)) (Polynomial.eval (HPow …
    hg : Eq (Polynomial.eval r ((Polynomial.expand R p) g)) (Polynomial.eval (HPow …
    ⊢ Eq (Polynomial.eval r ((Polynomial.expand R p) (HAdd.hAdd f g))) (Polynomial …
  -/
  rw [map_add, eval_add, eval_add, hf, hg]
  /-
    🎉 no goals
  -/


@[simp]
theorem expand_aeval {A : Type*} [Semiring A] [Algebra R A] (p : ℕ) (P : R[X]) (r : A) :
    aeval r (expand R p P) = aeval (r ^ p) P := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Nat
    P : Polynomial R
    r : A
    ⊢ Eq ((Polynomial.aeval r) ((Polynomial.expand R p) P)) ((Polynomial.aeval (HP …
  -/
  refine Polynomial.induction_on P (fun a => by simp) (fun f g hf hg => ?_) fun n a _ => by simp
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Nat
    P : Polynomial R
    r : A
    f g : Polynomial R
    hf : Eq ((Polynomial.aeval r) ((Polynomial.expand R p) f)) ((Polynomial.aeval  …
    hg : Eq ((Polynomial.aeval r) ((Polynomial.expand R p) g)) ((Polynomial.aeval  …
    ⊢ Eq ((Polynomial.aeval r) ((Polynomial.expand R p) (HAdd.hAdd f g))) ((Polyno …
  -/
  rw [map_add, aeval_add, aeval_add, hf, hg]
  /-
    🎉 no goals
  -/


/-- The opposite of `expand`: sends `∑ aₙ xⁿᵖ` to `∑ aₙ xⁿ`. -/
noncomputable def contract (p : ℕ) (f : R[X]) : R[X] :=
  ∑ n ∈ range (f.natDegree + 1), monomial n (f.coeff (n * p))


theorem coeff_contract {p : ℕ} (hp : p ≠ 0) (f : R[X]) (n : ℕ) :
    (contract p f).coeff n = f.coeff (n * p) := by
  simp only [contract, coeff_monomial, sum_ite_eq', finset_sum_coeff, mem_range, not_lt,
    ite_eq_left_iff]
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f : Polynomial R
    n : Nat
    ⊢ LE.le (HAdd.hAdd f.natDegree 1) n → Eq 0 (f.coeff (HMul.hMul n p))
  -/
  intro hn
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f : Polynomial R
    n : Nat
    hn : LE.le (HAdd.hAdd f.natDegree 1) n
    ⊢ Eq 0 (f.coeff (HMul.hMul n p))
  -/
  apply (coeff_eq_zero_of_natDegree_lt _).symm
  calc
    f.natDegree < f.natDegree + 1 := Nat.lt_succ_self _
    _ ≤ n * 1 := by simpa only [mul_one] using hn
    _ ≤ n * p := mul_le_mul_of_nonneg_left (show 1 ≤ p from hp.bot_lt) (zero_le n)


theorem map_contract {p : ℕ} (hp : p ≠ 0) {f : R →+* S} {q : R[X]} :
    (q.contract p).map f = (q.map f).contract p := ext fun n ↦ by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    S : Type v
    inst✝ : CommSemiring S
    p : Nat
    hp : Ne p 0
    f : RingHom R S
    q : Polynomial R
    n : Nat
    ⊢ Eq ((Polynomial.map f (Polynomial.contract p q)).coeff n) ((Polynomial.contr …
  -/
  simp only [coeff_map, coeff_contract hp]
  /-
    🎉 no goals
  -/


theorem contract_expand {f : R[X]} (hp : p ≠ 0) : contract p (expand R p f) = f := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : Ne p 0
    ⊢ Eq (Polynomial.contract p ((Polynomial.expand R p) f)) f
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    f : Polynomial R
    hp : Ne p 0
    n✝ : Nat
    ⊢ Eq ((Polynomial.contract p ((Polynomial.expand R p) f)).coeff n✝) (f.coeff n✝)
  -/
  simp [coeff_contract hp, coeff_expand hp.bot_lt, Nat.mul_div_cancel _ hp.bot_lt]
  /-
    🎉 no goals
  -/


theorem contract_one {f : R[X]} : contract 1 f = f :=
                 /-
                   R : Type u
                   inst✝ : CommSemiring R
                   f : Polynomial R
                   n : Nat
                   ⊢ Eq ((Polynomial.contract 1 f).coeff n) (f.coeff n)
                 -/
  ext fun n ↦ by rw [coeff_contract one_ne_zero, mul_one]
                 /-
                   🎉 no goals
                 -/


                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : CommSemiring R
                                                                    p : Nat
                                                                    r : R
                                                                    ⊢ Eq (Polynomial.contract p (Polynomial.C r)) (Polynomial.C r)
                                                                  -/
@[simp] theorem contract_C (r : R) : contract p (C r) = C r := by simp [contract]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem contract_add {p : ℕ} (hp : p ≠ 0) (f g : R[X]) :
    contract p (f + g) = contract p f + contract p g := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f g : Polynomial R
    ⊢ Eq (Polynomial.contract p (HAdd.hAdd f g)) (HAdd.hAdd (Polynomial.contract p …
  -/
  ext; simp_rw [coeff_add, coeff_contract hp, coeff_add]
       /-
         🎉 no goals
       -/


theorem contract_mul_expand {p : ℕ} (hp : p ≠ 0) (f g : R[X]) :
    contract p (f * expand R p g) = contract p f * g := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f g : Polynomial R
    ⊢ Eq (Polynomial.contract p (HMul.hMul f ((Polynomial.expand R p) g))) (HMul.h …
  -/
  ext n
  rw [coeff_contract hp, coeff_mul, coeff_mul, ← sum_subset
    (s₁ := (antidiagonal n).image fun x ↦ (x.1 * p, x.2 * p)), sum_image]
    /-
      case a
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n : Nat
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul.hMul (f.coeff  …
    -/
  · simp_rw [coeff_expand_mul hp.bot_lt, coeff_contract hp]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n : Nat
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
  · intro x hx y hy eq; simpa only [Prod.ext_iff, Nat.mul_right_cancel_iff hp.bot_lt] using eq
                        /-
                          🎉 no goals
                        -/
    /-
      case a.h
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n : Nat
      ⊢ HasSubset.Subset (Finset.image (fun x => { fst := HMul.hMul x.1 p, snd := HM …
    -/
  · simp_rw [subset_iff, mem_image, mem_antidiagonal]; rintro _ ⟨x, rfl, rfl⟩; simp_rw [add_mul]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  /-
    case a.hf
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f g : Polynomial R
    n : Nat
    ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
  -/
  simp_rw [mem_image, mem_antidiagonal]
  /-
    case a.hf
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f g : Polynomial R
    n : Nat
    ⊢ ∀ (x : Prod Nat Nat), Eq (HAdd.hAdd x.1 x.2) (HMul.hMul n p) → Not (Exists f …
  -/
  intro ⟨x, y⟩ eq nex
  /-
    case a.hf
    R : Type u
    inst✝ : CommSemiring R
    p : Nat
    hp : Ne p 0
    f g : Polynomial R
    n x y : Nat
    eq : Eq (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := y }.2) (HMul.hM …
    nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
    ⊢ Eq (HMul.hMul (f.coeff { fst := x, snd := y }.1) (((Polynomial.expand R p) g …
  -/
  by_cases h : p ∣ y
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n x y : Nat
      eq : Eq (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := y }.2) (HMul.hM …
      nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
      h : Dvd.dvd p y
      ⊢ Eq (HMul.hMul (f.coeff { fst := x, snd := y }.1) (((Polynomial.expand R p) g …
    -/
  · obtain ⟨x, rfl⟩ : p ∣ x := (Nat.dvd_add_iff_left h).mpr (eq ▸ dvd_mul_left p n)
    /-
      case pos.intro
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n y : Nat
      h : Dvd.dvd p y
      x : Nat
      eq : Eq (HAdd.hAdd { fst := HMul.hMul p x, snd := y }.1 { fst := HMul.hMul p x …
      nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
      ⊢ Eq (HMul.hMul (f.coeff { fst := HMul.hMul p x, snd := y }.1) (((Polynomial.e …
    -/
    obtain ⟨y, rfl⟩ := h
    /-
      case pos.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n x y : Nat
      eq : Eq (HAdd.hAdd { fst := HMul.hMul p x, snd := HMul.hMul p y }.1 { fst := H …
      nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
      ⊢ Eq (HMul.hMul (f.coeff { fst := HMul.hMul p x, snd := HMul.hMul p y }.1) ((( …
    -/
    refine (nex ⟨⟨x, y⟩, (Nat.mul_right_cancel_iff hp.bot_lt).mp ?_, by simp_rw [mul_comm]⟩).elim
    /-
      case pos.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n x y : Nat
      eq : Eq (HAdd.hAdd { fst := HMul.hMul p x, snd := HMul.hMul p y }.1 { fst := H …
      nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
      ⊢ Eq (HMul.hMul (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := y }.2)  …
    -/
    rw [← eq, mul_comm, mul_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      p : Nat
      hp : Ne p 0
      f g : Polynomial R
      n x y : Nat
      eq : Eq (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := y }.2) (HMul.hM …
      nex : Not (Exists fun a => And (Eq (HAdd.hAdd a.1 a.2) n) (Eq { fst := HMul.hM …
      h : Not (Dvd.dvd p y)
      ⊢ Eq (HMul.hMul (f.coeff { fst := x, snd := y }.1) (((Polynomial.expand R p) g …
    -/
  · rw [coeff_expand hp.bot_lt, if_neg h, mul_zero]
    /-
      🎉 no goals
    -/


@[simp] theorem isCoprime_expand {f g : R[X]} {p : ℕ} (hp : p ≠ 0) :
    IsCoprime (expand R p f) (expand R p g) ↔ IsCoprime f g :=
  ⟨fun ⟨a, b, eq⟩ ↦ ⟨contract p a, contract p b, by
    /-
      R : Type u
      inst✝ : CommSemiring R
      f g : Polynomial R
      p : Nat
      hp : Ne p 0
      x✝ : IsCoprime ((Polynomial.expand R p) f) ((Polynomial.expand R p) g)
      a b : Polynomial R
      eq : Eq (HAdd.hAdd (HMul.hMul a ((Polynomial.expand R p) f)) (HMul.hMul b ((Po …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.contract p a) f) (HMul.hMul (Polynomial …
    -/
    simp_rw [← contract_mul_expand hp, ← contract_add hp, eq, ← C_1, contract_C]⟩, (·.map _)⟩
    /-
      🎉 no goals
    -/


theorem expand_contract [CharP R p] [NoZeroDivisors R] {f : R[X]} (hf : Polynomial.derivative f = 0)
    (hp : p ≠ 0) : expand R p (contract p f) = f := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : NoZeroDivisors R
    f : Polynomial R
    hf : Eq (Polynomial.derivative f) 0
    hp : Ne p 0
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.contract p f)) f
  -/
  ext n
  /-
    case a
    R : Type u
    inst✝² : CommSemiring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : NoZeroDivisors R
    f : Polynomial R
    hf : Eq (Polynomial.derivative f) 0
    hp : Ne p 0
    n : Nat
    ⊢ Eq (((Polynomial.expand R p) (Polynomial.contract p f)).coeff n) (f.coeff n)
  -/
  rw [coeff_expand hp.bot_lt, coeff_contract hp]
  /-
    case a
    R : Type u
    inst✝² : CommSemiring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : NoZeroDivisors R
    f : Polynomial R
    hf : Eq (Polynomial.derivative f) 0
    hp : Ne p 0
    n : Nat
    ⊢ Eq (ite (Dvd.dvd p n) (f.coeff (HMul.hMul (HDiv.hDiv n p) p)) 0) (f.coeff n)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Dvd.dvd p n
      ⊢ Eq (f.coeff (HMul.hMul (HDiv.hDiv n p) p)) (f.coeff n)
    -/
  · rw [Nat.div_mul_cancel h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p n)
      ⊢ Eq 0 (f.coeff n)
    -/
  · cases' n with n
      /-
        case neg.zero
        R : Type u
        inst✝² : CommSemiring R
        p : Nat
        inst✝¹ : CharP R p
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        hf : Eq (Polynomial.derivative f) 0
        hp : Ne p 0
        h : Not (Dvd.dvd p 0)
        ⊢ Eq 0 (f.coeff 0)
      -/
    · exact absurd (dvd_zero p) h
      /-
        🎉 no goals
      -/
    /-
      case neg.succ
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    have := coeff_derivative f n
    /-
      case neg.succ
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      this : Eq ((Polynomial.derivative f).coeff n) (HMul.hMul (f.coeff (HAdd.hAdd n …
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    rw [hf, coeff_zero, zero_eq_mul] at this
    /-
      case neg.succ
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      this : Or (Eq (f.coeff (HAdd.hAdd n 1)) 0) (Eq (HAdd.hAdd (↑n) 1) 0)
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    cases' this with h'
      /-
        case neg.succ.inl
        R : Type u
        inst✝² : CommSemiring R
        p : Nat
        inst✝¹ : CharP R p
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        hf : Eq (Polynomial.derivative f) 0
        hp : Ne p 0
        n : Nat
        h : Not (Dvd.dvd p (HAdd.hAdd n 1))
        h' : Eq (f.coeff (HAdd.hAdd n 1)) 0
        ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
      -/
    · rw [h']
      /-
        🎉 no goals
      -/
    /-
      case neg.succ.inr
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      h✝ : Eq (HAdd.hAdd (↑n) 1) 0
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    rename_i _ _ _ h'
    /-
      case neg.succ.inr
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      h' : Eq (HAdd.hAdd (↑n) 1) 0
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    rw [← Nat.cast_succ, CharP.cast_eq_zero_iff R p] at h'
    /-
      case neg.succ.inr
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hp : Ne p 0
      n : Nat
      h : Not (Dvd.dvd p (HAdd.hAdd n 1))
      h' : Dvd.dvd p n.succ
      ⊢ Eq 0 (f.coeff (HAdd.hAdd n 1))
    -/
    exact absurd h' h
    /-
      🎉 no goals
    -/


theorem expand_contract' [NoZeroDivisors R] {f : R[X]} (hf : Polynomial.derivative f = 0) :
    expand R p (contract p f) = f := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    p : Nat
    inst✝¹ : ExpChar R p
    inst✝ : NoZeroDivisors R
    f : Polynomial R
    hf : Eq (Polynomial.derivative f) 0
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.contract p f)) f
  -/
  obtain _ | @⟨_, hprime, hchar⟩ := ‹ExpChar R p›
    /-
      case zero
      R : Type u
      inst✝³ : CommSemiring R
      inst✝² : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      inst✝¹ : CharZero R
      inst✝ : ExpChar R 1
      ⊢ Eq ((Polynomial.expand R 1) (Polynomial.contract 1 f)) f
    -/
  · rw [expand_one, contract_one]
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u
      inst✝² : CommSemiring R
      p : Nat
      inst✝¹ : ExpChar R p
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      hf : Eq (Polynomial.derivative f) 0
      hprime : Nat.Prime p
      hchar : CharP R p
      ⊢ Eq ((Polynomial.expand R p) (Polynomial.contract p f)) f
    -/
  · haveI := Fact.mk hchar; exact expand_contract p hf hprime.ne_zero
                            /-
                              🎉 no goals
                            -/


theorem expand_char (f : R[X]) : map (frobenius R p) (expand R p f) = f ^ p := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    p : Nat
    inst✝ : ExpChar R p
    f : Polynomial R
    ⊢ Eq (Polynomial.map (frobenius R p) ((Polynomial.expand R p) f)) (HPow.hPow f …
  -/
  refine f.induction_on' (fun a b ha hb => ?_) fun n a => ?_
    /-
      case refine_1
      R : Type u
      inst✝¹ : CommSemiring R
      p : Nat
      inst✝ : ExpChar R p
      f a b : Polynomial R
      ha : Eq (Polynomial.map (frobenius R p) ((Polynomial.expand R p) a)) (HPow.hPo …
      hb : Eq (Polynomial.map (frobenius R p) ((Polynomial.expand R p) b)) (HPow.hPo …
      ⊢ Eq (Polynomial.map (frobenius R p) ((Polynomial.expand R p) (HAdd.hAdd a b)) …
    -/
  · rw [map_add, Polynomial.map_add, ha, hb, add_pow_expChar]
    /-
      🎉 no goals
    -/
  · rw [expand_monomial, map_monomial, ← C_mul_X_pow_eq_monomial, ← C_mul_X_pow_eq_monomial,
      mul_pow, ← C.map_pow, frobenius_def]
    /-
      case refine_2
      R : Type u
      inst✝¹ : CommSemiring R
      p : Nat
      inst✝ : ExpChar R p
      f : Polynomial R
      n : Nat
      a : R
      ⊢ Eq (HMul.hMul (Polynomial.C (HPow.hPow a p)) (HPow.hPow Polynomial.X (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem map_expand_pow_char (f : R[X]) (n : ℕ) :
    map (frobenius R p ^ n) (expand R (p ^ n) f) = f ^ p ^ n := by
  induction n with
  | zero => simp [RingHom.one_def]
  | succ _ n_ih =>
    symm
    rw [pow_succ, pow_mul, ← n_ih, ← expand_char, pow_succ', RingHom.mul_def, ← map_map, mul_comm,
      expand_mul, ← map_expand]


theorem rootMultiplicity_expand_pow :
    (expand R (p ^ n) f).rootMultiplicity r = p ^ n * f.rootMultiplicity (r ^ p ^ n) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    p n : Nat
    inst✝ : ExpChar R p
    f : Polynomial R
    r : R
    ⊢ Eq (Polynomial.rootMultiplicity r ((Polynomial.expand R (HPow.hPow p n)) f)) …
  -/
  obtain rfl | h0 := eq_or_ne f 0; · simp
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case inr
    R : Type u
    inst✝¹ : CommRing R
    p n : Nat
    inst✝ : ExpChar R p
    f : Polynomial R
    r : R
    h0 : Ne f 0
    ⊢ Eq (Polynomial.rootMultiplicity r ((Polynomial.expand R (HPow.hPow p n)) f)) …
  -/
  obtain ⟨g, hg, ndvd⟩ := f.exists_eq_pow_rootMultiplicity_mul_and_not_dvd h0 (r ^ p ^ n)
  /-
    case inr.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    p n : Nat
    inst✝ : ExpChar R p
    f : Polynomial R
    r : R
    h0 : Ne f 0
    g : Polynomial R
    hg : Eq f (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C (HPow.hP …
    ndvd : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C (HPow.hPow r (HPow.h …
    ⊢ Eq (Polynomial.rootMultiplicity r ((Polynomial.expand R (HPow.hPow p n)) f)) …
  -/
  rw [dvd_iff_isRoot, ← eval_X (x := r), ← eval_pow, ← isRoot_comp, ← expand_eq_comp_X_pow] at ndvd
  conv_lhs => rw [hg, map_mul, map_pow, map_sub, expand_X, expand_C, map_pow, ← sub_pow_expChar_pow,
    ← pow_mul, mul_comm, rootMultiplicity_mul_X_sub_C_pow (expand_ne_zero (expChar_pow_pos R p n)
      |>.mpr <| right_ne_zero_of_mul <| hg ▸ h0), rootMultiplicity_eq_zero ndvd, zero_add]


theorem rootMultiplicity_expand :
    (expand R p f).rootMultiplicity r = p * f.rootMultiplicity (r ^ p) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : ExpChar R p
    f : Polynomial R
    r : R
    ⊢ Eq (Polynomial.rootMultiplicity r ((Polynomial.expand R p) f)) (HMul.hMul p  …
  -/
  rw [← pow_one p, rootMultiplicity_expand_pow]
  /-
    🎉 no goals
  -/


theorem isLocalHom_expand {p : ℕ} (hp : 0 < p) : IsLocalHom (expand R p) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Nat
    hp : LT.lt 0 p
    ⊢ IsLocalHom (Polynomial.expand R p)
  -/
  refine ⟨fun f hf1 => ?_⟩
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    hf1 : IsUnit ((Polynomial.expand R p) f)
    ⊢ IsUnit f
  -/
  have hf2 := eq_C_of_degree_eq_zero (degree_eq_zero_of_isUnit hf1)
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    hf1 : IsUnit ((Polynomial.expand R p) f)
    hf2 : Eq ((Polynomial.expand R p) f) (Polynomial.C (((Polynomial.expand R p) f …
    ⊢ IsUnit f
  -/
  rw [coeff_expand hp, if_pos (dvd_zero _), p.zero_div] at hf2
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Nat
    hp : LT.lt 0 p
    f : Polynomial R
    hf1 : IsUnit ((Polynomial.expand R p) f)
    hf2 : Eq ((Polynomial.expand R p) f) (Polynomial.C (f.coeff 0))
    ⊢ IsUnit f
  -/
  rw [hf2, isUnit_C] at hf1; rw [expand_eq_C hp] at hf2; rwa [hf2, isUnit_C]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_expand := isLocalHom_expand


theorem of_irreducible_expand {p : ℕ} (hp : p ≠ 0) {f : R[X]} (hf : Irreducible (expand R p f)) :
    Irreducible f :=
  let _ := isLocalHom_expand R hp.bot_lt
  hf.of_map


theorem of_irreducible_expand_pow {p : ℕ} (hp : p ≠ 0) {f : R[X]} {n : ℕ} :
    Irreducible (expand R (p ^ n) f) → Irreducible f :=
                            /-
                              R : Type u
                              inst✝¹ : CommRing R
                              inst✝ : IsDomain R
                              p : Nat
                              hp : Ne p 0
                              f : Polynomial R
                              n : Nat
                              hf : Irreducible ((Polynomial.expand R (HPow.hPow p Nat.zero)) f)
                              ⊢ Irreducible f
                            -/
  Nat.recOn n (fun hf => by rwa [pow_zero, expand_one] at hf) fun n ih hf =>
                            /-
                              🎉 no goals
                            -/
    ih <| of_irreducible_expand hp <| by
      /-
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        p : Nat
        hp : Ne p 0
        f : Polynomial R
        n✝ n : Nat
        ih : Irreducible ((Polynomial.expand R (HPow.hPow p n)) f) → Irreducible f
        hf : Irreducible ((Polynomial.expand R (HPow.hPow p n.succ)) f)
        ⊢ Irreducible ((Polynomial.expand R p) ((Polynomial.expand R (HPow.hPow p n))  …
      -/
      rw [pow_succ'] at hf
      /-
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        p : Nat
        hp : Ne p 0
        f : Polynomial R
        n✝ n : Nat
        ih : Irreducible ((Polynomial.expand R (HPow.hPow p n)) f) → Irreducible f
        hf : Irreducible ((Polynomial.expand R (HMul.hMul p (HPow.hPow p n))) f)
        ⊢ Irreducible ((Polynomial.expand R p) ((Polynomial.expand R (HPow.hPow p n))  …
      -/
      rwa [expand_expand]
      /-
        🎉 no goals
      -/


