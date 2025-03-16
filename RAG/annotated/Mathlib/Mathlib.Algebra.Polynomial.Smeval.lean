/-- Scalar multiplication together with taking a natural number power. -/
def smul_pow : ℕ → R → S := fun n r => r • x^n


/-- Evaluate a polynomial `p` in the scalar semiring `R` at an element `x` in the target `S` using
scalar multiple `R`-action. -/
irreducible_def smeval : S := p.sum (smul_pow x)


                                                              /-
                                                                R : Type u_1
                                                                inst✝³ : Semiring R
                                                                p : Polynomial R
                                                                S : Type u_2
                                                                inst✝² : AddCommMonoid S
                                                                inst✝¹ : Pow S Nat
                                                                inst✝ : MulActionWithZero R S
                                                                x : S
                                                                ⊢ Eq (p.smeval x) (p.sum (Polynomial.smul_pow x))
                                                              -/
theorem smeval_eq_sum : p.smeval x = p.sum (smul_pow x) := by rw [smeval_def]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem smeval_C : (C r).smeval x = r • x ^ 0 := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    r : R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    ⊢ Eq ((Polynomial.C r).smeval x) (HSMul.hSMul r (HPow.hPow x 0))
  -/
  simp only [smeval_eq_sum, smul_pow, zero_smul, sum_C_index]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_monomial (n : ℕ) :
    (monomial n r).smeval x = r • x ^ n := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    r : R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    n : Nat
    ⊢ Eq (((Polynomial.monomial n) r).smeval x) (HSMul.hSMul r (HPow.hPow x n))
  -/
  simp only [smeval_eq_sum, smul_pow, zero_smul, sum_monomial_index]
  /-
    🎉 no goals
  -/


theorem eval_eq_smeval : p.eval r = p.smeval r := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    p : Polynomial R
    ⊢ Eq (Polynomial.eval r p) (p.smeval r)
  -/
  rw [eval_eq_sum, smeval_eq_sum]
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    p : Polynomial R
    ⊢ Eq (p.sum fun e a => HMul.hMul a (HPow.hPow r e)) (p.sum (Polynomial.smul_po …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem eval₂_smulOneHom_eq_smeval (R : Type*) [Semiring R] {S : Type*} [Semiring S] [Module R S]
    [IsScalarTower R S S] (p : R[X]) (x : S) :
    p.eval₂ RingHom.smulOneHom x = p.smeval x := by
  /-
    R : Type u_3
    inst✝³ : Semiring R
    S : Type u_4
    inst✝² : Semiring S
    inst✝¹ : Module R S
    inst✝ : IsScalarTower R S S
    p : Polynomial R
    x : S
    ⊢ Eq (Polynomial.eval₂ RingHom.smulOneHom x p) (p.smeval x)
  -/
  rw [smeval_eq_sum, eval₂_eq_sum]
  /-
    R : Type u_3
    inst✝³ : Semiring R
    S : Type u_4
    inst✝² : Semiring S
    inst✝¹ : Module R S
    inst✝ : IsScalarTower R S S
    p : Polynomial R
    x : S
    ⊢ Eq (p.sum fun e a => HMul.hMul (RingHom.smulOneHom a) (HPow.hPow x e)) (p.su …
  -/
  congr 1 with e a
  /-
    case e_f.h.h
    R : Type u_3
    inst✝³ : Semiring R
    S : Type u_4
    inst✝² : Semiring S
    inst✝¹ : Module R S
    inst✝ : IsScalarTower R S S
    p : Polynomial R
    x : S
    e : Nat
    a : R
    ⊢ Eq (HMul.hMul (RingHom.smulOneHom a) (HPow.hPow x e)) (Polynomial.smul_pow x …
  -/
  simp only [RingHom.smulOneHom_apply, smul_one_mul, smul_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_zero : (0 : R[X]).smeval x = 0 := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    ⊢ Eq (Polynomial.smeval 0 x) 0
  -/
  simp only [smeval_eq_sum, smul_pow, sum_zero_index]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_one : (1 : R[X]).smeval x = 1 • x ^ 0 := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    ⊢ Eq (Polynomial.smeval 1 x) (HSMul.hSMul 1 (HPow.hPow x 0))
  -/
  rw [← C_1, smeval_C]
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    ⊢ Eq (HSMul.hSMul 1 (HPow.hPow x 0)) (HSMul.hSMul 1 (HPow.hPow x 0))
  -/
  simp only [Nat.cast_one, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_X :
    (X : R[X]).smeval x = x ^ 1 := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    ⊢ Eq (Polynomial.X.smeval x) (HPow.hPow x 1)
  -/
  simp only [smeval_eq_sum, smul_pow, zero_smul, sum_X_index, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_X_pow {n : ℕ} :
    (X ^ n : R[X]).smeval x = x ^ n := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : MulActionWithZero R S
    x : S
    n : Nat
    ⊢ Eq ((HPow.hPow Polynomial.X n).smeval x) (HPow.hPow x n)
  -/
  simp only [smeval_eq_sum, smul_pow, X_pow_eq_monomial, zero_smul, sum_monomial_index, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_add : (p + q).smeval x = p.smeval x + q.smeval x := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    p q : Polynomial R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : Module R S
    x : S
    ⊢ Eq ((HAdd.hAdd p q).smeval x) (HAdd.hAdd (p.smeval x) (q.smeval x))
  -/
  simp only [smeval_eq_sum, smul_pow]
  /-
    R : Type u_1
    inst✝³ : Semiring R
    p q : Polynomial R
    S : Type u_2
    inst✝² : AddCommMonoid S
    inst✝¹ : Pow S Nat
    inst✝ : Module R S
    x : S
    ⊢ Eq ((HAdd.hAdd p q).sum (Polynomial.smul_pow x)) (HAdd.hAdd (p.sum (Polynomi …
  -/
  refine sum_add_index p q (smul_pow x) (fun _ ↦ ?_) (fun _ _ _ ↦ ?_)
    /-
      case refine_1
      R : Type u_1
      inst✝³ : Semiring R
      p q : Polynomial R
      S : Type u_2
      inst✝² : AddCommMonoid S
      inst✝¹ : Pow S Nat
      inst✝ : Module R S
      x : S
      x✝ : Nat
      ⊢ Eq (Polynomial.smul_pow x x✝ 0) 0
    -/
  · rw [smul_pow, zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝³ : Semiring R
      p q : Polynomial R
      S : Type u_2
      inst✝² : AddCommMonoid S
      inst✝¹ : Pow S Nat
      inst✝ : Module R S
      x : S
      x✝² : Nat
      x✝¹ x✝ : R
      ⊢ Eq (Polynomial.smul_pow x x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Polynomial.smu …
    -/
  · rw [smul_pow, smul_pow, smul_pow, add_smul]
    /-
      🎉 no goals
    -/


theorem smeval_natCast (n : ℕ) : (n : R[X]).smeval x = n • x ^ 0 := by
  induction n with
  | zero => simp only [smeval_zero, Nat.cast_zero, zero_smul]
  | succ n ih => rw [n.cast_succ, smeval_add, ih, smeval_one, ← add_nsmul]


@[deprecated (since := "2024-04-17")]
alias smeval_nat_cast := smeval_natCast


@[simp]
theorem smeval_smul (r : R) : (r • p).smeval x = r • p.smeval x := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    rw [smul_add, smeval_add, ph, qh, ← smul_add, smeval_add]
  | h_monomial n a =>
    rw [smul_monomial, smeval_monomial, smeval_monomial, smul_assoc]


/-- `Polynomial.smeval` as a linear map. -/
def smeval.linearMap : R[X] →ₗ[R] S where
  toFun f := f.smeval x
                     /-
                       R : Type u_1
                       inst✝³ : Semiring R
                       p q : Polynomial R
                       S : Type u_2
                       inst✝² : AddCommMonoid S
                       inst✝¹ : Pow S Nat
                       inst✝ : Module R S
                       x : S
                       f g : Polynomial R
                       ⊢ Eq ((fun f => f.smeval x) (HAdd.hAdd f g)) (HAdd.hAdd ((fun f => f.smeval x) …
                     -/
  map_add' f g := by simp only [smeval_add]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u_1
                        inst✝³ : Semiring R
                        p q : Polynomial R
                        S : Type u_2
                        inst✝² : AddCommMonoid S
                        inst✝¹ : Pow S Nat
                        inst✝ : Module R S
                        x : S
                        c : R
                        f : Polynomial R
                        ⊢ Eq ({ toFun := fun f => f.smeval x, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) …
                      -/
  map_smul' c f := by simp only [smeval_smul, smul_eq_mul, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem smeval.linearMap_apply : smeval.linearMap R x p = p.smeval x := rfl


theorem leval_coe_eq_smeval {R : Type*} [Semiring R] (r : R) :
    ⇑(leval r) = fun p => p.smeval r := by
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    ⊢ Eq ⇑(Polynomial.leval r) fun p => p.smeval r
  -/
  rw [funext_iff]
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    ⊢ ∀ (x : Polynomial R), Eq ((Polynomial.leval r) x) (x.smeval r)
  -/
  intro
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    x✝ : Polynomial R
    ⊢ Eq ((Polynomial.leval r) x✝) (x✝.smeval r)
  -/
  rw [leval_apply, smeval_def, eval_eq_sum]
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    x✝ : Polynomial R
    ⊢ Eq (x✝.sum fun e a => HMul.hMul a (HPow.hPow r e)) (x✝.sum (Polynomial.smul_ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem leval_eq_smeval.linearMap {R : Type*} [Semiring R] (r : R) :
    leval r = smeval.linearMap R r := by
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    ⊢ Eq (Polynomial.leval r) (Polynomial.smeval.linearMap R r)
  -/
  refine LinearMap.ext ?_
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    ⊢ ∀ (x : Polynomial R), Eq ((Polynomial.leval r) x) ((Polynomial.smeval.linear …
  -/
  intro
  /-
    R : Type u_3
    inst✝ : Semiring R
    r : R
    x✝ : Polynomial R
    ⊢ Eq ((Polynomial.leval r) x✝) ((Polynomial.smeval.linearMap R r) x✝)
  -/
  rw [leval_apply, smeval.linearMap_apply, eval_eq_smeval]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_neg : (-p).smeval x = - p.smeval x := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Type u_2
    inst✝² : AddCommGroup S
    inst✝¹ : Pow S Nat
    inst✝ : Module R S
    p : Polynomial R
    x : S
    ⊢ Eq ((Neg.neg p).smeval x) (Neg.neg (p.smeval x))
  -/
  rw [← add_eq_zero_iff_eq_neg, ← smeval_add, neg_add_cancel, smeval_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_sub : (p - q).smeval x = p.smeval x - q.smeval x := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Type u_2
    inst✝² : AddCommGroup S
    inst✝¹ : Pow S Nat
    inst✝ : Module R S
    p q : Polynomial R
    x : S
    ⊢ Eq ((HSub.hSub p q).smeval x) (HSub.hSub (p.smeval x) (q.smeval x))
  -/
  rw [sub_eq_add_neg, smeval_add, smeval_neg, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem smeval_neg_nat (S : Type*) [NonAssocRing S] [Pow S ℕ] [NatPowAssoc S] (q : ℕ[X])
    (n : ℕ) : q.smeval (-(n : S)) = q.smeval (-n : ℤ) := by
  /-
    S : Type u_3
    inst✝² : NonAssocRing S
    inst✝¹ : Pow S Nat
    inst✝ : NatPowAssoc S
    q : Polynomial Nat
    n : Nat
    ⊢ Eq (q.smeval (Neg.neg ↑n)) ↑(q.smeval (Neg.neg ↑n))
  -/
  rw [smeval_eq_sum, smeval_eq_sum]
  /-
    S : Type u_3
    inst✝² : NonAssocRing S
    inst✝¹ : Pow S Nat
    inst✝ : NatPowAssoc S
    q : Polynomial Nat
    n : Nat
    ⊢ Eq (q.sum (Polynomial.smul_pow (Neg.neg ↑n))) ↑(q.sum (Polynomial.smul_pow ( …
  -/
  simp only [Polynomial.smul_pow, sum_def, Int.cast_sum, Int.cast_mul, Int.cast_npow]
  /-
    S : Type u_3
    inst✝² : NonAssocRing S
    inst✝¹ : Pow S Nat
    inst✝ : NatPowAssoc S
    q : Polynomial Nat
    n : Nat
    ⊢ Eq (q.support.sum fun x => HSMul.hSMul (q.coeff x) (HPow.hPow (Neg.neg ↑n) x …
  -/
  refine Finset.sum_congr rfl ?_
  /-
    S : Type u_3
    inst✝² : NonAssocRing S
    inst✝¹ : Pow S Nat
    inst✝ : NatPowAssoc S
    q : Polynomial Nat
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem q.support x → Eq (HSMul.hSMul (q.coeff x) (HPow. …
  -/
  intro k _
  rw [show -(n : S) = (-n : ℤ) by simp only [Int.cast_neg, Int.cast_natCast], nsmul_eq_mul,
    ← AddGroupWithOne.intCast_ofNat, ← Int.cast_npow, ← Int.cast_mul, ← nsmul_eq_mul]


theorem smeval_C_mul : (C r * p).smeval x = r • p.smeval x := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [mul_add, smeval_add, ph, qh, smul_add]
  | h_monomial n b =>
    simp only [C_mul_monomial, smeval_monomial, mul_smul]


theorem smeval_at_natCast (q : ℕ[X]) : ∀(n : ℕ), q.smeval (n : S) = q.smeval n := by
  induction q using Polynomial.induction_on' with
  | h_add p q ph qh =>
    intro n
    simp only [add_mul, smeval_add, ph, qh, Nat.cast_add]
  | h_monomial n a =>
    intro n
    rw [smeval_monomial, smeval_monomial, nsmul_eq_mul, smul_eq_mul, Nat.cast_mul, Nat.cast_npow]


@[deprecated (since := "2024-04-17")]
alias smeval_at_nat_cast := smeval_at_natCast


theorem smeval_at_zero : p.smeval (0 : S) = (p.coeff 0) • (1 : S)  := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp_all only [smeval_add, coeff_add, add_smul]
  | h_monomial n a =>
    cases n with
    | zero => simp only [monomial_zero_left, smeval_C, npow_zero, coeff_C_zero]
    | succ n => rw [coeff_monomial_succ, smeval_monomial, npow_add, npow_one, mul_zero, zero_smul,
        smul_zero]


theorem smeval_X_mul : (X * p).smeval x = x * p.smeval x := by
    induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [smeval_add, ph, qh, mul_add]
  | h_monomial n a =>
    rw [← monomial_one_one_eq_X, monomial_mul_monomial, smeval_monomial, one_mul, npow_add,
      npow_one, ← mul_smul_comm, smeval_monomial]


theorem smeval_X_pow_assoc (m n : ℕ) :
    x ^ m * x ^ n * p.smeval x = x ^ m * (x ^ n * p.smeval x) := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [smeval_add, ph, qh, mul_add]
  | h_monomial n a =>
    simp only [smeval_monomial, mul_smul_comm, npow_mul_assoc]


theorem smeval_X_pow_mul : ∀ (n : ℕ), (X^n * p).smeval x = x^n * p.smeval x
  | 0 => by
    /-
      R : Type u_1
      inst✝⁵ : Semiring R
      p : Polynomial R
      S : Type u_2
      inst✝⁴ : NonAssocSemiring S
      inst✝³ : Module R S
      inst✝² : Pow S Nat
      x : S
      inst✝¹ : NatPowAssoc S
      inst✝ : SMulCommClass R S S
      ⊢ Eq ((HMul.hMul (HPow.hPow Polynomial.X 0) p).smeval x) (HMul.hMul (HPow.hPow …
    -/
    simp [npow_zero, one_mul]
    /-
      🎉 no goals
    -/
  | n + 1 => by
    rw [add_comm, npow_add, mul_assoc, npow_one, smeval_X_mul, smeval_X_pow_mul n, npow_add,
      smeval_X_pow_assoc, npow_one]


theorem smeval_monomial_mul (n : ℕ) :
    (monomial n r * p).smeval x = r • (x ^ n * p.smeval x) := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs =>
    simp only [add_comp, hr, hs, smeval_add, add_mul]
    rw [← C_mul_X_pow_eq_monomial, mul_assoc, smeval_C_mul, smeval_X_pow_mul, smeval_add]
  | h_monomial n a =>
    rw [smeval_monomial, monomial_mul_monomial, smeval_monomial, npow_add, mul_smul, mul_smul_comm]


theorem smeval_mul_X : (p * X).smeval x = p.smeval x * x := by
    induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [add_mul, smeval_add, ph, qh]
  | h_monomial n a =>
    simp only [← monomial_one_one_eq_X, monomial_mul_monomial, smeval_monomial, mul_one, pow_succ',
      mul_assoc, npow_add, smul_mul_assoc, npow_one]


theorem smeval_assoc_X_pow (m n : ℕ) :
    p.smeval x * x ^ m * x ^ n = p.smeval x * (x ^ m * x ^ n) := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [smeval_add, ph, qh, add_mul]
  | h_monomial n a =>
    rw [smeval_monomial, smul_mul_assoc, smul_mul_assoc, npow_mul_assoc, ← smul_mul_assoc]


theorem smeval_mul_X_pow : ∀ (n : ℕ), (p * X^n).smeval x = p.smeval x * x^n
  | 0 => by
    /-
      R : Type u_1
      inst✝⁵ : Semiring R
      p : Polynomial R
      S : Type u_2
      inst✝⁴ : NonAssocSemiring S
      inst✝³ : Module R S
      inst✝² : Pow S Nat
      x : S
      inst✝¹ : NatPowAssoc S
      inst✝ : IsScalarTower R S S
      ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X 0)).smeval x) (HMul.hMul (p.smeval  …
    -/
    simp only [npow_zero, mul_one]
    /-
      🎉 no goals
    -/
  | n + 1 => by
    rw [npow_add, ← mul_assoc, npow_one, smeval_mul_X, smeval_mul_X_pow n, npow_add,
      ← smeval_assoc_X_pow, npow_one]


theorem smeval_mul : (p * q).smeval x  = p.smeval x * q.smeval x := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs =>
    simp only [add_comp, hr, hs, smeval_add, add_mul]
  | h_monomial n a =>
    simp only [smeval_monomial, smeval_C_mul, smeval_mul_X_pow, smeval_monomial_mul, smul_mul_assoc]


theorem smeval_pow : ∀ (n : ℕ), (p^n).smeval x = (p.smeval x)^n
  | 0 => by
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      p : Polynomial R
      S : Type u_2
      inst✝⁵ : NonAssocSemiring S
      inst✝⁴ : Module R S
      inst✝³ : Pow S Nat
      x : S
      inst✝² : NatPowAssoc S
      inst✝¹ : IsScalarTower R S S
      inst✝ : SMulCommClass R S S
      ⊢ Eq ((HPow.hPow p 0).smeval x) (HPow.hPow (p.smeval x) 0)
    -/
    simp only [npow_zero, smeval_one, one_smul]
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      p : Polynomial R
      S : Type u_2
      inst✝⁵ : NonAssocSemiring S
      inst✝⁴ : Module R S
      inst✝³ : Pow S Nat
      x : S
      inst✝² : NatPowAssoc S
      inst✝¹ : IsScalarTower R S S
      inst✝ : SMulCommClass R S S
      n : Nat
      ⊢ Eq ((HPow.hPow p (HAdd.hAdd n 1)).smeval x) (HPow.hPow (p.smeval x) (HAdd.hA …
    -/
    rw [npow_add, smeval_mul, smeval_pow n, pow_one, npow_add, npow_one]
    /-
      🎉 no goals
    -/


theorem smeval_comp : (p.comp q).smeval x  = p.smeval (q.smeval x) := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs =>
    simp [add_comp, hr, hs, smeval_add]
  | h_monomial n a =>
    simp [smeval_monomial, smeval_C_mul, smeval_pow]


theorem smeval_commute_left (hc : Commute x y) : Commute (p.smeval x) y := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs => exact (smeval_add R r s x) ▸ Commute.add_left hr hs
  | h_monomial n a =>
    simp only [smeval_monomial]
    refine Commute.smul_left ?_ a
    induction n with
    | zero => simp only [npow_zero, Commute.one_left]
    | succ n ih =>
      refine (commute_iff_eq (x ^ (n + 1)) y).mpr ?_
      rw [commute_iff_eq (x ^ n) y] at ih
      rw [pow_succ, ← mul_assoc, ← ih]
      exact Commute.right_comm hc (x ^ n)


theorem smeval_commute (hc : Commute x y) : Commute (p.smeval x) (q.smeval y) := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs => exact (smeval_add R r s x) ▸ Commute.add_left hr hs
  | h_monomial n a =>
    simp only [smeval_monomial]
    refine Commute.smul_left ?_ a
    induction n with
    | zero => simp only [npow_zero, Commute.one_left]
    | succ n ih =>
      refine (commute_iff_eq (x ^ (n + 1)) (q.smeval y)).mpr ?_
      rw [commute_iff_eq (x ^ n) (q.smeval y)] at ih
      have hxq : x * q.smeval y = q.smeval y * x := by
        refine (commute_iff_eq x (q.smeval y)).mp ?_
        exact Commute.symm (smeval_commute_left R q (Commute.symm hc))
      rw [pow_succ, ← mul_assoc, ← ih, mul_assoc, hxq, mul_assoc]


theorem aeval_eq_smeval {R : Type*} [CommSemiring R] {S : Type*} [Semiring S] [Algebra R S]
    (x : S) (p : R[X]) : aeval x p = p.smeval x := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    x : S
    p : Polynomial R
    ⊢ Eq ((Polynomial.aeval x) p) (p.smeval x)
  -/
  rw [aeval_def, eval₂_def, Algebra.algebraMap_eq_smul_one', smeval_def]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    x : S
    p : Polynomial R
    ⊢ Eq (p.sum fun e a => HMul.hMul ((fun r => HSMul.hSMul r 1) a) (HPow.hPow x e …
  -/
  simp only [Algebra.smul_mul_assoc, one_mul]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    x : S
    p : Polynomial R
    ⊢ Eq (p.sum fun e a => HSMul.hSMul a (HPow.hPow x e)) (p.sum (Polynomial.smul_ …
  -/
  exact rfl
  /-
    🎉 no goals
  -/


theorem aeval_coe_eq_smeval {R : Type*} [CommSemiring R] {S : Type*} [Semiring S] [Algebra R S]
    (x : S) : ⇑(aeval x) = fun (p : R[X]) => p.smeval x := funext fun p => aeval_eq_smeval x p


