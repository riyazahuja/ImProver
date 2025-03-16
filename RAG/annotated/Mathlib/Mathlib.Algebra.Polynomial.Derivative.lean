/-- `derivative p` is the formal derivative of the polynomial `p` -/
def derivative : R[X] →ₗ[R] R[X] where
  toFun p := p.sum fun n a => C (a * n) * X ^ (n - 1)
  map_add' p q := by
    /-
      R : Type u
      S : Type v
      T : Type w
      ι : Type y
      A : Type z
      a b : R
      n : Nat
      inst✝ : Semiring R
      p q : Polynomial R
      ⊢ Eq ((fun p => p.sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a ↑n)) (HP …
    -/
    dsimp only
    /-
      R : Type u
      S : Type v
      T : Type w
      ι : Type y
      A : Type z
      a b : R
      n : Nat
      inst✝ : Semiring R
      p q : Polynomial R
      ⊢ Eq ((HAdd.hAdd p q).sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a ↑n)) …
    -/
    rw [sum_add_index] <;>
      simp only [add_mul, forall_const, RingHom.map_add, eq_self_iff_true, zero_mul,
        RingHom.map_zero]
  map_smul' a p := by
    /-
      R : Type u
      S : Type v
      T : Type w
      ι : Type y
      A : Type z
      a✝ b : R
      n : Nat
      inst✝ : Semiring R
      a : R
      p : Polynomial R
      ⊢ Eq ({ toFun := fun p => p.sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul  …
    -/
    dsimp; rw [sum_smul_index] <;>
      simp only [mul_sum, ← C_mul', mul_assoc, coeff_C_mul, RingHom.map_mul, forall_const, zero_mul,
        RingHom.map_zero, sum]


theorem derivative_apply (p : R[X]) : derivative p = p.sum fun n a => C (a * n) * X ^ (n - 1) :=
  rfl


theorem coeff_derivative (p : R[X]) (n : ℕ) :
    coeff (derivative p) n = coeff p (n + 1) * (n + 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((Polynomial.derivative p).coeff n) (HMul.hMul (p.coeff (HAdd.hAdd n 1))  …
  -/
  rw [derivative_apply]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((p.sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a ↑n)) (HPow.hPow P …
  -/
  simp only [coeff_X_pow, coeff_sum, coeff_C_mul]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (p.sum fun a b => HMul.hMul (HMul.hMul b ↑a) (ite (Eq n (HSub.hSub a 1))  …
  -/
  rw [sum, Finset.sum_eq_single (n + 1)]
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff (HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)) (ite (E …
    -/
  · simp only [Nat.add_succ_sub_one, add_zero, mul_one, if_true, eq_self_iff_true]; norm_cast
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    /-
      case h₀
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ⊢ ∀ (b : Nat), Membership.mem p.support b → Ne b (HAdd.hAdd n 1) → Eq (HMul.hM …
    -/
  · intro b
    /-
      case h₀
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n b : Nat
      ⊢ Membership.mem p.support b → Ne b (HAdd.hAdd n 1) → Eq (HMul.hMul (HMul.hMul …
    -/
    cases b
      /-
        case h₀.zero
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        ⊢ Membership.mem p.support 0 → Ne 0 (HAdd.hAdd n 1) → Eq (HMul.hMul (HMul.hMul …
      -/
    · intros
      /-
        case h₀.zero
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        a✝¹ : Membership.mem p.support 0
        a✝ : Ne 0 (HAdd.hAdd n 1)
        ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff 0) ↑0) (ite (Eq n (HSub.hSub 0 1)) 1 0)) 0
      -/
      rw [Nat.cast_zero, mul_zero, zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case h₀.succ
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n n✝ : Nat
        ⊢ Membership.mem p.support (HAdd.hAdd n✝ 1) → Ne (HAdd.hAdd n✝ 1) (HAdd.hAdd n …
      -/
    · intro _ H
      /-
        case h₀.succ
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n n✝ : Nat
        a✝ : Membership.mem p.support (HAdd.hAdd n✝ 1)
        H : Ne (HAdd.hAdd n✝ 1) (HAdd.hAdd n 1)
        ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff (HAdd.hAdd n✝ 1)) ↑(HAdd.hAdd n✝ 1)) (ite  …
      -/
      rw [Nat.add_one_sub_one, if_neg (mt (congr_arg Nat.succ) H.symm), mul_zero]
      /-
        🎉 no goals
      -/
  · rw [if_pos (add_tsub_cancel_right n 1).symm, mul_one, Nat.cast_add, Nat.cast_one,
      mem_support_iff]
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ⊢ Not (Ne (p.coeff (HAdd.hAdd n 1)) 0) → Eq (HMul.hMul (p.coeff (HAdd.hAdd n 1 …
    -/
    intro h
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : Not (Ne (p.coeff (HAdd.hAdd n 1)) 0)
      ⊢ Eq (HMul.hMul (p.coeff (HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)) 0
    -/
    push_neg at h
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : Eq (p.coeff (HAdd.hAdd n 1)) 0
      ⊢ Eq (HMul.hMul (p.coeff (HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)) 0
    -/
    simp [h]
    /-
      🎉 no goals
    -/


@[simp]
theorem derivative_zero : derivative (0 : R[X]) = 0 :=
  derivative.map_zero


theorem iterate_derivative_zero {k : ℕ} : derivative^[k] (0 : R[X]) = 0 :=
  iterate_map_zero derivative k


@[simp]
theorem derivative_monomial (a : R) (n : ℕ) :
    derivative (monomial n a) = monomial (n - 1) (a * n) := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    n : Nat
    ⊢ Eq (Polynomial.derivative ((Polynomial.monomial n) a)) ((Polynomial.monomial …
  -/
  rw [derivative_apply, sum_monomial_index, C_mul_X_pow_eq_monomial]
  /-
    case hf
    R : Type u
    inst✝ : Semiring R
    a : R
    n : Nat
    ⊢ Eq (HMul.hMul (Polynomial.C (HMul.hMul 0 ↑n)) (HPow.hPow Polynomial.X (HSub. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem derivative_C_mul_X (a : R) : derivative (C a * X) = C a := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ Eq (Polynomial.derivative (HMul.hMul (Polynomial.C a) Polynomial.X)) (Polyno …
  -/
  simp [C_mul_X_eq_monomial, derivative_monomial, Nat.cast_one, mul_one]
  /-
    🎉 no goals
  -/


theorem derivative_C_mul_X_pow (a : R) (n : ℕ) :
    derivative (C a * X ^ n) = C (a * n) * X ^ (n - 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    n : Nat
    ⊢ Eq (Polynomial.derivative (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial. …
  -/
  rw [C_mul_X_pow_eq_monomial, C_mul_X_pow_eq_monomial, derivative_monomial]
  /-
    🎉 no goals
  -/


theorem derivative_C_mul_X_sq (a : R) : derivative (C a * X ^ 2) = C (a * 2) * X := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ Eq (Polynomial.derivative (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial. …
  -/
  rw [derivative_C_mul_X_pow, Nat.cast_two, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem derivative_X_pow (n : ℕ) : derivative (X ^ n : R[X]) = C (n : R) * X ^ (n - 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.derivative (HPow.hPow Polynomial.X n)) (HMul.hMul (Polynomial …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  convert derivative_C_mul_X_pow (1 : R) n <;> simp
                                               /-
                                                 🎉 no goals
                                               -/


theorem derivative_X_sq : derivative (X ^ 2 : R[X]) = C 2 * X := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.derivative (HPow.hPow Polynomial.X 2)) (HMul.hMul (Polynomial …
  -/
  rw [derivative_X_pow, Nat.cast_two, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            R : Type u
                                                            inst✝ : Semiring R
                                                            a : R
                                                            ⊢ Eq (Polynomial.derivative (Polynomial.C a)) 0
                                                          -/
theorem derivative_C {a : R} : derivative (C a) = 0 := by simp [derivative_apply]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem derivative_of_natDegree_zero {p : R[X]} (hp : p.natDegree = 0) : derivative p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Eq p.natDegree 0
    ⊢ Eq (Polynomial.derivative p) 0
  -/
  rw [eq_C_of_natDegree_eq_zero hp, derivative_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem derivative_X : derivative (X : R[X]) = 1 :=
                                        /-
                                          R : Type u
                                          inst✝ : Semiring R
                                          ⊢ Eq ((Polynomial.monomial (HSub.hSub 1 1)) (HMul.hMul 1 ↑1)) 1
                                        -/
  (derivative_monomial _ _).trans <| by simp
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem derivative_one : derivative (1 : R[X]) = 0 :=
  derivative_C


@[simp]
theorem derivative_add {f g : R[X]} : derivative (f + g) = derivative f + derivative g :=
  derivative.map_add f g


theorem derivative_X_add_C (c : R) : derivative (X + C c) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ Eq (Polynomial.derivative (HAdd.hAdd Polynomial.X (Polynomial.C c))) 1
  -/
  rw [derivative_add, derivative_X, derivative_C, add_zero]
  /-
    🎉 no goals
  -/


theorem derivative_sum {s : Finset ι} {f : ι → R[X]} :
    derivative (∑ b ∈ s, f b) = ∑ b ∈ s, derivative (f b) :=
  map_sum ..


theorem iterate_derivative_sum (k : ℕ) (s : Finset ι) (f : ι → R[X]) :
    derivative^[k] (∑ b ∈ s, f b) = ∑ b ∈ s, derivative^[k] (f b) := by
  /-
    R : Type u
    ι : Type y
    inst✝ : Semiring R
    k : Nat
    s : Finset ι
    f : ι → Polynomial R
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (s.sum fun b => f b)) (s.sum fun  …
  -/
  simp_rw [← LinearMap.pow_apply, map_sum]
  /-
    🎉 no goals
  -/


theorem derivative_smul {S : Type*} [Monoid S] [DistribMulAction S R] [IsScalarTower S R R] (s : S)
    (p : R[X]) : derivative (s • p) = s • derivative p :=
  derivative.map_smul_of_tower s p


@[simp]
theorem iterate_derivative_smul {S : Type*} [Monoid S] [DistribMulAction S R] [IsScalarTower S R R]
    (s : S) (p : R[X]) (k : ℕ) : derivative^[k] (s • p) = s • derivative^[k] p := by
  induction k generalizing p with
  | zero => simp
  | succ k ih => simp [ih]


@[simp]
theorem iterate_derivative_C_mul (a : R) (p : R[X]) (k : ℕ) :
    derivative^[k] (C a * p) = C a * derivative^[k] p := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    p : Polynomial R
    k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HMul.hMul (Polynomial.C a) p)) ( …
  -/
  simp_rw [← smul_eq_C_mul, iterate_derivative_smul]
  /-
    🎉 no goals
  -/


theorem derivative_C_mul (a : R) (p : R[X]) :
    derivative (C a * p) = C a * derivative p := iterate_derivative_C_mul _ _ 1


theorem of_mem_support_derivative {p : R[X]} {n : ℕ} (h : n ∈ p.derivative.support) :
    n + 1 ∈ p.support :=
  mem_support_iff.2 fun h1 : p.coeff (n + 1) = 0 =>
                                                            /-
                                                              R : Type u
                                                              inst✝ : Semiring R
                                                              p : Polynomial R
                                                              n : Nat
                                                              h : Membership.mem (Polynomial.derivative p).support n
                                                              h1 : Eq (p.coeff (HAdd.hAdd n 1)) 0
                                                              ⊢ Eq ((Polynomial.derivative p).coeff n) 0
                                                            -/
    mem_support_iff.1 h <| show p.derivative.coeff n = 0 by rw [coeff_derivative, h1, zero_mul]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem degree_derivative_lt {p : R[X]} (hp : p ≠ 0) : p.derivative.degree < p.degree :=
  (Finset.sup_lt_iff <| bot_lt_iff_ne_bot.2 <| mt degree_eq_bot.1 hp).2 fun n hp =>
    lt_of_lt_of_le (WithBot.coe_lt_coe.2 n.lt_succ_self) <|
      Finset.le_sup <| of_mem_support_derivative hp


theorem degree_derivative_le {p : R[X]} : p.derivative.degree ≤ p.degree :=
  letI := Classical.decEq R
                                   /-
                                     R : Type u
                                     inst✝ : Semiring R
                                     p : Polynomial R
                                     this : DecidableEq R := Classical.decEq R
                                     H : Eq p 0
                                     ⊢ Eq (Polynomial.derivative p).degree p.degree
                                   -/
  if H : p = 0 then le_of_eq <| by rw [H, derivative_zero] else (degree_derivative_lt H).le
                                   /-
                                     🎉 no goals
                                   -/


theorem natDegree_derivative_lt {p : R[X]} (hp : p.natDegree ≠ 0) :
    p.derivative.natDegree < p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p.natDegree 0
    ⊢ LT.lt (Polynomial.derivative p).natDegree p.natDegree
  -/
  rcases eq_or_ne (derivative p) 0 with hp' | hp'
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p.natDegree 0
      hp' : Eq (Polynomial.derivative p) 0
      ⊢ LT.lt (Polynomial.derivative p).natDegree p.natDegree
    -/
  · rw [hp', Polynomial.natDegree_zero]
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p.natDegree 0
      hp' : Eq (Polynomial.derivative p) 0
      ⊢ LT.lt 0 p.natDegree
    -/
    exact hp.bot_lt
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p.natDegree 0
      hp' : Ne (Polynomial.derivative p) 0
      ⊢ LT.lt (Polynomial.derivative p).natDegree p.natDegree
    -/
  · rw [natDegree_lt_natDegree_iff hp']
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p.natDegree 0
      hp' : Ne (Polynomial.derivative p) 0
      ⊢ LT.lt (Polynomial.derivative p).degree p.degree
    -/
    exact degree_derivative_lt fun h => hp (h.symm ▸ natDegree_zero)
    /-
      🎉 no goals
    -/


theorem natDegree_derivative_le (p : R[X]) : p.derivative.natDegree ≤ p.natDegree - 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ LE.le (Polynomial.derivative p).natDegree (HSub.hSub p.natDegree 1)
  -/
  by_cases p0 : p.natDegree = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      p0 : Eq p.natDegree 0
      ⊢ LE.le (Polynomial.derivative p).natDegree (HSub.hSub p.natDegree 1)
    -/
  · simp [p0, derivative_of_natDegree_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      p0 : Not (Eq p.natDegree 0)
      ⊢ LE.le (Polynomial.derivative p).natDegree (HSub.hSub p.natDegree 1)
    -/
  · exact Nat.le_sub_one_of_lt (natDegree_derivative_lt p0)
    /-
      🎉 no goals
    -/


theorem natDegree_iterate_derivative (p : R[X]) (k : ℕ) :
    (derivative^[k] p).natDegree ≤ p.natDegree - k := by
  induction k with
  | zero => rw [Function.iterate_zero_apply, Nat.sub_zero]
  | succ d hd =>
      rw [Function.iterate_succ_apply', Nat.sub_succ']
      exact (natDegree_derivative_le _).trans <| Nat.sub_le_sub_right hd 1


@[simp]
theorem derivative_natCast {n : ℕ} : derivative (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.derivative ↑n) 0
  -/
  rw [← map_natCast C n]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.derivative (Polynomial.C ↑n)) 0
  -/
  exact derivative_C
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias derivative_nat_cast := derivative_natCast


@[simp]
theorem derivative_ofNat (n : ℕ) [n.AtLeastTwo] :
    derivative (no_index (OfNat.ofNat n) : R[X]) = 0 :=
  derivative_natCast


theorem iterate_derivative_eq_zero {p : R[X]} {x : ℕ} (hx : p.natDegree < x) :
    Polynomial.derivative^[x] p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : Nat
    hx : LT.lt p.natDegree x
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) x p) 0
  -/
  induction' h : p.natDegree using Nat.strong_induction_on with _ ih generalizing p x
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    n✝ : Nat
    ih : ∀ (m : Nat), LT.lt m n✝ → ∀ {p : Polynomial R} {x : Nat}, LT.lt p.natDegr …
    p : Polynomial R
    x : Nat
    hx : LT.lt p.natDegree x
    h : Eq p.natDegree n✝
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) x p) 0
  -/
  subst h
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : Nat
    hx : LT.lt p.natDegree x
    ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) x p) 0
  -/
  obtain ⟨t, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (pos_of_gt hx).ne'
  /-
    case h.intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
    t : Nat
    hx : LT.lt p.natDegree t.succ
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) t.succ p) 0
  -/
  rw [Function.iterate_succ_apply]
  /-
    case h.intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
    t : Nat
    hx : LT.lt p.natDegree t.succ
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) t (Polynomial.derivative p)) 0
  -/
  by_cases hp : p.natDegree = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
      t : Nat
      hx : LT.lt p.natDegree t.succ
      hp : Eq p.natDegree 0
      ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) t (Polynomial.derivative p)) 0
    -/
  · rw [derivative_of_natDegree_zero hp, iterate_derivative_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
    t : Nat
    hx : LT.lt p.natDegree t.succ
    hp : Not (Eq p.natDegree 0)
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) t (Polynomial.derivative p)) 0
  -/
  have := natDegree_derivative_lt hp
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ih : ∀ (m : Nat), LT.lt m p.natDegree → ∀ {p : Polynomial R} {x : Nat}, LT.lt  …
    t : Nat
    hx : LT.lt p.natDegree t.succ
    hp : Not (Eq p.natDegree 0)
    this : LT.lt (Polynomial.derivative p).natDegree p.natDegree
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) t (Polynomial.derivative p)) 0
  -/
  exact ih _ this (this.trans_le <| Nat.le_of_lt_succ hx) rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem iterate_derivative_C {k} (h : 0 < k) : derivative^[k] (C a : R[X]) = 0 :=
  iterate_derivative_eq_zero <| (natDegree_C _).trans_lt h


@[simp]
theorem iterate_derivative_one {k} (h : 0 < k) : derivative^[k] (1 : R[X]) = 0 :=
  iterate_derivative_C h


@[simp]
theorem iterate_derivative_X {k} (h : 1 < k) : derivative^[k] (X : R[X]) = 0 :=
  iterate_derivative_eq_zero <| natDegree_X_le.trans_lt h


theorem natDegree_eq_zero_of_derivative_eq_zero [NoZeroSMulDivisors ℕ R] {f : R[X]}
    (h : derivative f = 0) : f.natDegree = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    ⊢ Eq f.natDegree 0
  -/
  rcases eq_or_ne f 0 with (rfl | hf)
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      h : Eq (Polynomial.derivative 0) 0
      ⊢ Eq (Polynomial.natDegree 0) 0
    -/
  · exact natDegree_zero
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    ⊢ Eq f.natDegree 0
  -/
  rw [natDegree_eq_zero_iff_degree_le_zero]
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    ⊢ LE.le f.degree 0
  -/
  by_contra! f_nat_degree_pos
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.degree
    ⊢ False
  -/
  rw [← natDegree_pos_iff_degree_pos] at f_nat_degree_pos
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    ⊢ False
  -/
  let m := f.natDegree - 1
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    ⊢ False
  -/
  have hm : m + 1 = f.natDegree := tsub_add_cancel_of_le f_nat_degree_pos
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    ⊢ False
  -/
  have h2 := coeff_derivative f m
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : Eq (Polynomial.derivative f) 0
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    h2 : Eq ((Polynomial.derivative f).coeff m) (HMul.hMul (f.coeff (HAdd.hAdd m 1 …
    ⊢ False
  -/
  rw [Polynomial.ext_iff] at h
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : ∀ (n : Nat), Eq ((Polynomial.derivative f).coeff n) (Polynomial.coeff 0 n)
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    h2 : Eq ((Polynomial.derivative f).coeff m) (HMul.hMul (f.coeff (HAdd.hAdd m 1 …
    ⊢ False
  -/
  rw [h m, coeff_zero, ← Nat.cast_add_one, ← nsmul_eq_mul', eq_comm, smul_eq_zero] at h2
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : ∀ (n : Nat), Eq ((Polynomial.derivative f).coeff n) (Polynomial.coeff 0 n)
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    h2 : Or (Eq (HAdd.hAdd m 1) 0) (Eq (f.coeff (HAdd.hAdd m 1)) 0)
    ⊢ False
  -/
  replace h2 := h2.resolve_left m.succ_ne_zero
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : ∀ (n : Nat), Eq ((Polynomial.derivative f).coeff n) (Polynomial.coeff 0 n)
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    h2 : Eq (f.coeff (HAdd.hAdd m 1)) 0
    ⊢ False
  -/
  rw [hm, ← leadingCoeff, leadingCoeff_eq_zero] at h2
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    f : Polynomial R
    h : ∀ (n : Nat), Eq ((Polynomial.derivative f).coeff n) (Polynomial.coeff 0 n)
    hf : Ne f 0
    f_nat_degree_pos : LT.lt 0 f.natDegree
    m : Nat := HSub.hSub f.natDegree 1
    hm : Eq (HAdd.hAdd m 1) f.natDegree
    h2 : Eq f 0
    ⊢ False
  -/
  exact hf h2
  /-
    🎉 no goals
  -/


theorem eq_C_of_derivative_eq_zero [NoZeroSMulDivisors ℕ R] {f : R[X]} (h : derivative f = 0) :
    f = C (f.coeff 0) :=
  eq_C_of_natDegree_eq_zero <| natDegree_eq_zero_of_derivative_eq_zero h


@[simp]
theorem derivative_mul {f g : R[X]} : derivative (f * g) = derivative f * g + f * derivative g := by
  induction f using Polynomial.induction_on' with
  | h_add => simp only [add_mul, map_add, add_assoc, add_left_comm, *]
  | h_monomial m a =>
  induction g using Polynomial.induction_on' with
  | h_add => simp only [mul_add, map_add, add_assoc, add_left_comm, *]
  | h_monomial n b =>
  simp only [monomial_mul_monomial, derivative_monomial]
  simp only [mul_assoc, (Nat.cast_commute _ _).eq, Nat.cast_add, mul_add, map_add]
  cases m with
  | zero => simp only [zero_add, Nat.cast_zero, mul_zero, map_zero]
  | succ m =>
  cases n with
  | zero => simp only [add_zero, Nat.cast_zero, mul_zero, map_zero]
  | succ n =>
  simp only [Nat.add_succ_sub_one, add_tsub_cancel_right]
  rw [add_assoc, add_comm n 1]


theorem derivative_eval (p : R[X]) (x : R) :
    p.derivative.eval x = p.sum fun n a => a * n * x ^ (n - 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : R
    ⊢ Eq (Polynomial.eval x (Polynomial.derivative p)) (p.sum fun n a => HMul.hMul …
  -/
  simp_rw [derivative_apply, eval_sum, eval_mul_X_pow, eval_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem derivative_map [Semiring S] (p : R[X]) (f : R →+* S) :
    derivative (p.map f) = p.derivative.map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    ⊢ Eq (Polynomial.derivative (Polynomial.map f p)) (Polynomial.map f (Polynomia …
  -/
  let n := max p.natDegree (map f p).natDegree
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max p.natDegree (Polynomial.map f p).natDegree
    ⊢ Eq (Polynomial.derivative (Polynomial.map f p)) (Polynomial.map f (Polynomia …
  -/
  rw [derivative_apply, derivative_apply]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max p.natDegree (Polynomial.map f p).natDegree
    ⊢ Eq ((Polynomial.map f p).sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a …
  -/
  rw [sum_over_range' _ _ (n + 1) ((le_max_left _ _).trans_lt (lt_add_one _))]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max p.natDegree (Polynomial.map f p).natDegree
    ⊢ Eq ((Polynomial.map f p).sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a …
  -/
  on_goal 1 => rw [sum_over_range' _ _ (n + 1) ((le_max_right _ _).trans_lt (lt_add_one _))]
  · simp only [Polynomial.map_sum, Polynomial.map_mul, Polynomial.map_C, map_mul, coeff_map,
      map_natCast, Polynomial.map_natCast, Polynomial.map_pow, map_X]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max p.natDegree (Polynomial.map f p).natDegree
    ⊢ ∀ (n : Nat), Eq (HMul.hMul (Polynomial.C (HMul.hMul 0 ↑n)) (HPow.hPow Polyno …
  -/
  all_goals intro n; rw [zero_mul, C_0, zero_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem iterate_derivative_map [Semiring S] (p : R[X]) (f : R →+* S) (k : ℕ) :
    Polynomial.derivative^[k] (p.map f) = (Polynomial.derivative^[k] p).map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (Polynomial.map f p)) (Polynomial …
  -/
  induction' k with k ih generalizing p
    /-
      case zero
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) 0 (Polynomial.map f p)) (Polynomial …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      k : Nat
      ih : ∀ (p : Polynomial R), Eq (Nat.iterate (⇑Polynomial.derivative) k (Polynom …
      p : Polynomial R
      ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) (HAdd.hAdd k 1) (Polynomial.map f p …
    -/
  · simp only [ih, Function.iterate_succ, Polynomial.derivative_map, Function.comp_apply]
    /-
      🎉 no goals
    -/


theorem derivative_natCast_mul {n : ℕ} {f : R[X]} :
    derivative ((n : R[X]) * f) = n * derivative f := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Eq (Polynomial.derivative (HMul.hMul (↑n) f)) (HMul.hMul (↑n) (Polynomial.de …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias derivative_nat_cast_mul := derivative_natCast_mul


@[simp]
theorem iterate_derivative_natCast_mul {n k : ℕ} {f : R[X]} :
    derivative^[k] ((n : R[X]) * f) = n * derivative^[k] f := by
  /-
    R : Type u
    inst✝ : Semiring R
    n k : Nat
    f : Polynomial R
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HMul.hMul (↑n) f)) (HMul.hMul (↑ …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  induction' k with k ih generalizing f <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-04-17")]
alias iterate_derivative_nat_cast_mul := iterate_derivative_natCast_mul


theorem mem_support_derivative [NoZeroSMulDivisors ℕ R] (p : R[X]) (n : ℕ) :
    n ∈ (derivative p).support ↔ n + 1 ∈ p.support := by
  suffices ¬p.coeff (n + 1) * (n + 1 : ℕ) = 0 ↔ coeff p (n + 1) ≠ 0 by
    simpa only [mem_support_iff, coeff_derivative, Ne, Nat.cast_succ]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    p : Polynomial R
    n : Nat
    ⊢ Iff (Not (Eq (HMul.hMul (p.coeff (HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)) 0)) (Ne  …
  -/
  rw [← nsmul_eq_mul', smul_eq_zero]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    p : Polynomial R
    n : Nat
    ⊢ Iff (Not (Or (Eq (HAdd.hAdd n 1) 0) (Eq (p.coeff (HAdd.hAdd n 1)) 0))) (Ne ( …
  -/
  simp only [Nat.succ_ne_zero, false_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_derivative_eq [NoZeroSMulDivisors ℕ R] (p : R[X]) (hp : 0 < natDegree p) :
    degree (derivative p) = (natDegree p - 1 : ℕ) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    p : Polynomial R
    hp : LT.lt 0 p.natDegree
    ⊢ Eq (Polynomial.derivative p).degree ↑(HSub.hSub p.natDegree 1)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ LE.le (Polynomial.derivative p).degree ↑(HSub.hSub p.natDegree 1)
    -/
  · rw [derivative_apply]
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ LE.le (p.sum fun n a => HMul.hMul (Polynomial.C (HMul.hMul a ↑n)) (HPow.hPow …
    -/
    apply le_trans (degree_sum_le _ _) (Finset.sup_le _)
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ ∀ (b : Nat), Membership.mem p.support b → LE.le ((fun n a => HMul.hMul (Poly …
    -/
    intro n hn
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      n : Nat
      hn : Membership.mem p.support n
      ⊢ LE.le ((fun n a => HMul.hMul (Polynomial.C (HMul.hMul a ↑n)) (HPow.hPow Poly …
    -/
    apply le_trans (degree_C_mul_X_pow_le _ _) (WithBot.coe_le_coe.2 (tsub_le_tsub_right _ _))
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      n : Nat
      hn : Membership.mem p.support n
      ⊢ LE.le n p.natDegree
    -/
    apply le_natDegree_of_mem_supp _ hn
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ LE.le (↑(HSub.hSub p.natDegree 1)) (Polynomial.derivative p).degree
    -/
  · refine le_sup ?_
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ Membership.mem (Polynomial.derivative p).support (HSub.hSub p.natDegree 1)
    -/
    rw [mem_support_derivative, tsub_add_cancel_of_le, mem_support_iff]
      /-
        case a
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroSMulDivisors Nat R
        p : Polynomial R
        hp : LT.lt 0 p.natDegree
        ⊢ Ne (p.coeff p.natDegree) 0
      -/
    · rw [coeff_natDegree, Ne, leadingCoeff_eq_zero]
      /-
        case a
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroSMulDivisors Nat R
        p : Polynomial R
        hp : LT.lt 0 p.natDegree
        ⊢ Not (Eq p 0)
      -/
      intro h
      /-
        case a
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroSMulDivisors Nat R
        p : Polynomial R
        hp : LT.lt 0 p.natDegree
        h : Eq p 0
        ⊢ False
      -/
      rw [h, natDegree_zero] at hp
      /-
        case a
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroSMulDivisors Nat R
        p : Polynomial R
        hp : LT.lt 0 0
        h : Eq p 0
        ⊢ False
      -/
      exact hp.false
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      hp : LT.lt 0 p.natDegree
      ⊢ LE.le 1 p.natDegree
    -/
    exact hp
    /-
      🎉 no goals
    -/


theorem coeff_iterate_derivative {k} (p : R[X]) (m : ℕ) :
    (derivative^[k] p).coeff m = (m + k).descFactorial k • p.coeff (m + k) := by
  induction k generalizing m with
  | zero => simp
  | succ k ih =>
      calc
        (derivative^[k + 1] p).coeff m
        _ = Nat.descFactorial (Nat.succ (m + k)) k • p.coeff (m + k.succ) * (m + 1) := by
          rw [Function.iterate_succ_apply', coeff_derivative, ih m.succ, Nat.succ_add, Nat.add_succ]
        _ = ((m + 1) * Nat.descFactorial (Nat.succ (m + k)) k) • p.coeff (m + k.succ) := by
          rw [← Nat.cast_add_one, ← nsmul_eq_mul', smul_smul]
        _ = Nat.descFactorial (m.succ + k) k.succ • p.coeff (m + k.succ) := by
          rw [← Nat.succ_add, Nat.descFactorial_succ, add_tsub_cancel_right]
        _ = Nat.descFactorial (m + k.succ) k.succ • p.coeff (m + k.succ) := by
          rw [Nat.succ_add_eq_add_succ]


theorem iterate_derivative_eq_sum (p : R[X]) (k : ℕ) :
    derivative^[k] p =
      ∑ x ∈ (derivative^[k] p).support, C ((x + k).descFactorial k • p.coeff (x + k)) * X ^ x := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k p) ((Nat.iterate (⇑Polynomial.der …
  -/
  conv_lhs => rw [(derivative^[k] p).as_sum_support_C_mul_X_pow]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Eq ((Nat.iterate (⇑Polynomial.derivative) k p).support.sum fun i => HMul.hMu …
  -/
  refine sum_congr rfl fun i _ ↦ ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k i : Nat
    x✝ : Membership.mem (Nat.iterate (⇑Polynomial.derivative) k p).support i
    ⊢ Eq (HMul.hMul (Polynomial.C ((Nat.iterate (⇑Polynomial.derivative) k p).coef …
  -/
  rw [coeff_iterate_derivative, Nat.descFactorial_eq_factorial_mul_choose]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_eq_factorial_smul_sum (p : R[X]) (k : ℕ) :
    derivative^[k] p = k ! •
      ∑ x ∈ (derivative^[k] p).support, C ((x + k).choose k • p.coeff (x + k)) * X ^ x := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k p) (HSMul.hSMul k.factorial ((Nat …
  -/
  conv_lhs => rw [iterate_derivative_eq_sum]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Eq ((Nat.iterate (⇑Polynomial.derivative) k p).support.sum fun x => HMul.hMu …
  -/
  rw [smul_sum]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Eq ((Nat.iterate (⇑Polynomial.derivative) k p).support.sum fun x => HMul.hMu …
  -/
  refine sum_congr rfl fun i _ ↦ ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k i : Nat
    x✝ : Membership.mem (Nat.iterate (⇑Polynomial.derivative) k p).support i
    ⊢ Eq (HMul.hMul (Polynomial.C (HSMul.hSMul ((HAdd.hAdd i k).descFactorial k) ( …
  -/
  rw [← smul_mul_assoc, smul_C, smul_smul, Nat.descFactorial_eq_factorial_mul_choose]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_mul {n} (p q : R[X]) :
    derivative^[n] (p * q) =
      ∑ k ∈ range n.succ, (n.choose k • (derivative^[n - k] p * derivative^[k] q)) := by
  induction n with
  | zero =>
    simp [Finset.range]
  | succ n IH =>
    calc
      derivative^[n + 1] (p * q) =
          derivative (∑ k ∈ range n.succ,
              n.choose k • (derivative^[n - k] p * derivative^[k] q)) := by
        rw [Function.iterate_succ_apply', IH]
      _ = (∑ k ∈ range n.succ,
            n.choose k • (derivative^[n - k + 1] p * derivative^[k] q)) +
          ∑ k ∈ range n.succ,
            n.choose k • (derivative^[n - k] p * derivative^[k + 1] q) := by
        simp_rw [derivative_sum, derivative_smul, derivative_mul, Function.iterate_succ_apply',
          smul_add, sum_add_distrib]
      _ = (∑ k ∈ range n.succ,
                n.choose k.succ • (derivative^[n - k] p * derivative^[k + 1] q)) +
              1 • (derivative^[n + 1] p * derivative^[0] q) +
            ∑ k ∈ range n.succ, n.choose k • (derivative^[n - k] p * derivative^[k + 1] q) :=
        ?_
      _ = ((∑ k ∈ range n.succ, n.choose k • (derivative^[n - k] p * derivative^[k + 1] q)) +
              ∑ k ∈ range n.succ,
                n.choose k.succ • (derivative^[n - k] p * derivative^[k + 1] q)) +
            1 • (derivative^[n + 1] p * derivative^[0] q) := by
        rw [add_comm, add_assoc]
      _ = (∑ i ∈ range n.succ,
              (n + 1).choose (i + 1) • (derivative^[n + 1 - (i + 1)] p * derivative^[i + 1] q)) +
            1 • (derivative^[n + 1] p * derivative^[0] q) := by
        simp_rw [Nat.choose_succ_succ, Nat.succ_sub_succ, add_smul, sum_add_distrib]
      _ = ∑ k ∈ range n.succ.succ,
            n.succ.choose k • (derivative^[n.succ - k] p * derivative^[k] q) := by
        rw [sum_range_succ' _ n.succ, Nat.choose_zero_right, tsub_zero]
    congr
    refine (sum_range_succ' _ _).trans (congr_arg₂ (· + ·) ?_ ?_)
    · rw [sum_range_succ, Nat.choose_succ_self, zero_smul, add_zero]
      refine sum_congr rfl fun k hk => ?_
      rw [mem_range] at hk
      congr
      omega
    · rw [Nat.choose_zero_right, tsub_zero]


/--
Iterated derivatives as a finite support function.
-/
@[simps! apply_toFun]
noncomputable def derivativeFinsupp : R[X] →ₗ[R] ℕ →₀ R[X] where
  toFun p := .onFinset (range (p.natDegree + 1)) (derivative^[·] p) fun i ↦ by
    /-
      R : Type u
      S : Type v
      T : Type w
      ι : Type y
      A : Type z
      a b : R
      n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      ⊢ Ne ((fun x => Nat.iterate (⇑Polynomial.derivative) x p) i) 0 → Membership.me …
    -/
    contrapose; simp_all [iterate_derivative_eq_zero, Nat.succ_le]
                /-
                  🎉 no goals
                -/
                     /-
                       R : Type u
                       S : Type v
                       T : Type w
                       ι : Type y
                       A : Type z
                       a b : R
                       n : Nat
                       inst✝ : Semiring R
                       x✝¹ x✝ : Polynomial R
                       ⊢ Eq ((fun p => Finsupp.onFinset (Finset.range (HAdd.hAdd p.natDegree 1)) (fun …
                     -/
  map_add' _ _ := by ext; simp
                          /-
                            🎉 no goals
                          -/
                      /-
                        R : Type u
                        S : Type v
                        T : Type w
                        ι : Type y
                        A : Type z
                        a b : R
                        n : Nat
                        inst✝ : Semiring R
                        x✝¹ : R
                        x✝ : Polynomial R
                        ⊢ Eq ({ toFun := fun p => Finsupp.onFinset (Finset.range (HAdd.hAdd p.natDegre …
                      -/
  map_smul' _ _ := by ext; simp
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem support_derivativeFinsupp_subset_range {p : R[X]} {n : ℕ} (h : p.natDegree < n) :
    (derivativeFinsupp p).support ⊆ range n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ HasSubset.Subset (Polynomial.derivativeFinsupp p).support (Finset.range n)
  -/
  dsimp [derivativeFinsupp]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ HasSubset.Subset (Finsupp.onFinset (Finset.range (HAdd.hAdd p.natDegree 1))  …
  -/
  exact Finsupp.support_onFinset_subset.trans (Finset.range_subset.mpr h)
  /-
    🎉 no goals
  -/


@[simp]
theorem derivativeFinsupp_C (r : R) : derivativeFinsupp (C r : R[X]) = .single 0 (C r) := by
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    ⊢ Eq (Polynomial.derivativeFinsupp (Polynomial.C r)) (Finsupp.single 0 (Polyno …
  -/
  ext i : 1
  match i with
  | 0 => simp
  | i + 1 => simp


@[simp]
theorem derivativeFinsupp_one : derivativeFinsupp (1 : R[X]) = .single 0 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.derivativeFinsupp 1) (Finsupp.single 0 1)
  -/
  simpa using derivativeFinsupp_C (1 : R)
  /-
    🎉 no goals
  -/


@[simp]
theorem derivativeFinsupp_X : derivativeFinsupp (X : R[X]) = .single 0 X + .single 1 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.derivativeFinsupp Polynomial.X) (HAdd.hAdd (Finsupp.single 0  …
  -/
  ext i : 1
  match i with
  | 0 => simp
  | 1 => simp
  | (n + 2) => simp


theorem derivativeFinsupp_map [Semiring S] (p : R[X]) (f : R →+* S) :
                                                                               /-
                                                                                 R : Type u
                                                                                 S : Type v
                                                                                 T : Type w
                                                                                 ι : Type y
                                                                                 A : Type z
                                                                                 a b : R
                                                                                 n : Nat
                                                                                 inst✝¹ : Semiring R
                                                                                 inst✝ : Semiring S
                                                                                 p : Polynomial R
                                                                                 f : RingHom R S
                                                                                 ⊢ Eq ((fun x => Polynomial.map f x) 0) 0
                                                                               -/
    derivativeFinsupp (p.map f) = (derivativeFinsupp p).mapRange (·.map f) (by simp) := by
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    ⊢ Eq (Polynomial.derivativeFinsupp (Polynomial.map f p)) (Finsupp.mapRange (fu …
  -/
  ext i : 1
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    i : Nat
    ⊢ Eq ((Polynomial.derivativeFinsupp (Polynomial.map f p)) i) ((Finsupp.mapRang …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem derivativeFinsupp_derivative (p : R[X]) :
    derivativeFinsupp (derivative p) =
      (derivativeFinsupp p).comapDomain Nat.succ Nat.succ_injective.injOn := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (Polynomial.derivativeFinsupp (Polynomial.derivative p)) (Finsupp.comapDo …
  -/
  ext i : 1
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    ⊢ Eq ((Polynomial.derivativeFinsupp (Polynomial.derivative p)) i) ((Finsupp.co …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem derivative_pow_succ (p : R[X]) (n : ℕ) :
    derivative (p ^ (n + 1)) = C (n + 1 : R) * p ^ n * derivative p :=
                  /-
                    R : Type u
                    inst✝ : CommSemiring R
                    p : Polynomial R
                    n : Nat
                    ⊢ Eq (Polynomial.derivative (HPow.hPow p (HAdd.hAdd Nat.zero 1))) (HMul.hMul ( …
                  -/
  Nat.recOn n (by simp) fun n ih => by
                  /-
                    🎉 no goals
                  -/
    rw [pow_succ, derivative_mul, ih, Nat.add_one, mul_right_comm, C_add,
                                                              /-
                                                                R : Type u
                                                                inst✝ : CommSemiring R
                                                                p : Polynomial R
                                                                n✝ n : Nat
                                                                ih : Eq (Polynomial.derivative (HPow.hPow p (HAdd.hAdd n 1))) (HMul.hMul (HMul …
                                                                ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (HMul.hMul (Polynomial.C ↑n)  …
                                                              -/
      add_mul, add_mul, pow_succ, ← mul_assoc, C_1, one_mul]; simp [add_mul]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem derivative_pow (p : R[X]) (n : ℕ) :
    derivative (p ^ n) = C (n : R) * p ^ (n - 1) * derivative p :=
                    /-
                      R : Type u
                      inst✝ : CommSemiring R
                      p : Polynomial R
                      n : Nat
                      ⊢ Eq (Polynomial.derivative (HPow.hPow p Nat.zero)) (HMul.hMul (HMul.hMul (Pol …
                    -/
  Nat.casesOn n (by rw [pow_zero, derivative_one, Nat.cast_zero, C_0, zero_mul, zero_mul]) fun n =>
                    /-
                      🎉 no goals
                    -/
       /-
         R : Type u
         inst✝ : CommSemiring R
         p : Polynomial R
         n✝ n : Nat
         ⊢ Eq (Polynomial.derivative (HPow.hPow p n.succ)) (HMul.hMul (HMul.hMul (Polyn …
       -/
    by rw [p.derivative_pow_succ n, Nat.add_one_sub_one, n.cast_succ]
       /-
         🎉 no goals
       -/


theorem derivative_sq (p : R[X]) : derivative (p ^ 2) = C 2 * p * derivative p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Polynomial R
    ⊢ Eq (Polynomial.derivative (HPow.hPow p 2)) (HMul.hMul (HMul.hMul (Polynomial …
  -/
  rw [derivative_pow_succ, Nat.cast_one, one_add_one_eq_two, pow_one]
  /-
    🎉 no goals
  -/


theorem pow_sub_one_dvd_derivative_of_pow_dvd {p q : R[X]} {n : ℕ}
    (dvd : q ^ n ∣ p) : q ^ (n - 1) ∣ derivative p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    n : Nat
    dvd : Dvd.dvd (HPow.hPow q n) p
    ⊢ Dvd.dvd (HPow.hPow q (HSub.hSub n 1)) (Polynomial.derivative p)
  -/
  obtain ⟨r, rfl⟩ := dvd
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    q : Polynomial R
    n : Nat
    r : Polynomial R
    ⊢ Dvd.dvd (HPow.hPow q (HSub.hSub n 1)) (Polynomial.derivative (HMul.hMul (HPo …
  -/
  rw [derivative_mul, derivative_pow]
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    q : Polynomial R
    n : Nat
    r : Polynomial R
    ⊢ Dvd.dvd (HPow.hPow q (HSub.hSub n 1)) (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul …
  -/
  exact (((dvd_mul_left _ _).mul_right _).mul_right _).add ((pow_dvd_pow q n.pred_le).mul_right _)
  /-
    🎉 no goals
  -/


theorem pow_sub_dvd_iterate_derivative_of_pow_dvd {p q : R[X]} {n : ℕ} (m : ℕ)
    (dvd : q ^ n ∣ p) : q ^ (n - m) ∣ derivative^[m] p := by
  induction m generalizing p with
  | zero => simpa
  | succ m ih =>
    rw [Nat.sub_succ, Function.iterate_succ']
    exact pow_sub_one_dvd_derivative_of_pow_dvd (ih dvd)


theorem pow_sub_dvd_iterate_derivative_pow (p : R[X]) (n m : ℕ) :
    p ^ (n - m) ∣ derivative^[m] (p ^ n) := pow_sub_dvd_iterate_derivative_of_pow_dvd m dvd_rfl


theorem dvd_iterate_derivative_pow (f : R[X]) (n : ℕ) {m : ℕ} (c : R) (hm : m ≠ 0) :
    (n : R) ∣ eval c (derivative^[m] (f ^ n)) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : Polynomial R
    n m : Nat
    c : R
    hm : Ne m 0
    ⊢ Dvd.dvd (↑n) (Polynomial.eval c (Nat.iterate (⇑Polynomial.derivative) m (HPo …
  -/
  obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hm
  rw [Function.iterate_succ_apply, derivative_pow, mul_assoc, C_eq_natCast,
    iterate_derivative_natCast_mul, eval_mul, eval_natCast]
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    f : Polynomial R
    n : Nat
    c : R
    m : Nat
    hm : Ne m.succ 0
    ⊢ Dvd.dvd (↑n) (HMul.hMul (↑n) (Polynomial.eval c (Nat.iterate (⇑Polynomial.de …
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


theorem iterate_derivative_X_pow_eq_natCast_mul (n k : ℕ) :
    derivative^[k] (X ^ n : R[X]) = ↑(Nat.descFactorial n k : R[X]) * X ^ (n - k) := by
  induction k with
  | zero =>
    rw [Function.iterate_zero_apply, tsub_zero, Nat.descFactorial_zero, Nat.cast_one, one_mul]
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, derivative_natCast_mul, derivative_X_pow, C_eq_natCast,
      Nat.descFactorial_succ, Nat.sub_sub, Nat.cast_mul]
    simp [mul_comm, mul_assoc, mul_left_comm]


@[deprecated (since := "2024-04-17")]
alias iterate_derivative_X_pow_eq_nat_cast_mul := iterate_derivative_X_pow_eq_natCast_mul


theorem iterate_derivative_X_pow_eq_C_mul (n k : ℕ) :
    derivative^[k] (X ^ n : R[X]) = C (Nat.descFactorial n k : R) * X ^ (n - k) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HPow.hPow Polynomial.X n)) (HMul …
  -/
  rw [iterate_derivative_X_pow_eq_natCast_mul n k, C_eq_natCast]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_X_pow_eq_smul (n : ℕ) (k : ℕ) :
    derivative^[k] (X ^ n : R[X]) = (Nat.descFactorial n k : R) • X ^ (n - k) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    n k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HPow.hPow Polynomial.X n)) (HSMu …
  -/
  rw [iterate_derivative_X_pow_eq_C_mul n k, smul_eq_C_mul]
  /-
    🎉 no goals
  -/


theorem derivative_X_add_C_pow (c : R) (m : ℕ) :
    derivative ((X + C c) ^ m) = C (m : R) * (X + C c) ^ (m - 1) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    c : R
    m : Nat
    ⊢ Eq (Polynomial.derivative (HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C c …
  -/
  rw [derivative_pow, derivative_X_add_C, mul_one]
  /-
    🎉 no goals
  -/


theorem derivative_X_add_C_sq (c : R) : derivative ((X + C c) ^ 2) = C 2 * (X + C c) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    c : R
    ⊢ Eq (Polynomial.derivative (HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C c …
  -/
  rw [derivative_sq, derivative_X_add_C, mul_one]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_X_add_pow (n k : ℕ) (c : R) :
    derivative^[k] ((X + C c) ^ n) = Nat.descFactorial n k • (X + C c) ^ (n - k) := by
  induction k with
  | zero => simp
  | succ k IH =>
      rw [Nat.sub_succ', Function.iterate_succ_apply', IH, derivative_smul,
        derivative_X_add_C_pow, map_natCast, Nat.descFactorial_succ, nsmul_eq_mul, nsmul_eq_mul,
        Nat.cast_mul]
      ring


theorem derivative_comp (p q : R[X]) :
    derivative (p.comp q) = derivative q * p.derivative.comp q := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    ⊢ Eq (Polynomial.derivative (p.comp q)) (HMul.hMul (Polynomial.derivative q) ( …
  -/
  induction p using Polynomial.induction_on'
    /-
      case h_add
      R : Type u
      inst✝ : CommSemiring R
      q p✝ q✝ : Polynomial R
      a✝¹ : Eq (Polynomial.derivative (p✝.comp q)) (HMul.hMul (Polynomial.derivative …
      a✝ : Eq (Polynomial.derivative (q✝.comp q)) (HMul.hMul (Polynomial.derivative  …
      ⊢ Eq (Polynomial.derivative ((HAdd.hAdd p✝ q✝).comp q)) (HMul.hMul (Polynomial …
    -/
  · simp [*, mul_add]
    /-
      🎉 no goals
    -/
  · simp only [derivative_pow, derivative_mul, monomial_comp, derivative_monomial, derivative_C,
      zero_mul, C_eq_natCast, zero_add, RingHom.map_mul]
    /-
      case h_monomial
      R : Type u
      inst✝ : CommSemiring R
      q : Polynomial R
      n✝ : Nat
      a✝ : R
      ⊢ Eq (HMul.hMul (Polynomial.C a✝) (HMul.hMul (HMul.hMul (↑n✝) (HPow.hPow q (HS …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Chain rule for formal derivative of polynomials. -/
theorem derivative_eval₂_C (p q : R[X]) :
    derivative (p.eval₂ C q) = p.derivative.eval₂ C q * derivative q :=
                                         /-
                                           R : Type u
                                           inst✝ : CommSemiring R
                                           p q : Polynomial R
                                           r : R
                                           ⊢ Eq (Polynomial.derivative (Polynomial.eval₂ Polynomial.C q (Polynomial.C r)) …
                                         -/
  Polynomial.induction_on p (fun r => by rw [eval₂_C, derivative_C, eval₂_zero, zero_mul])
                                         /-
                                           🎉 no goals
                                         -/
    (fun p₁ p₂ ih₁ ih₂ => by
      /-
        R : Type u
        inst✝ : CommSemiring R
        p q p₁ p₂ : Polynomial R
        ih₁ : Eq (Polynomial.derivative (Polynomial.eval₂ Polynomial.C q p₁)) (HMul.hM …
        ih₂ : Eq (Polynomial.derivative (Polynomial.eval₂ Polynomial.C q p₂)) (HMul.hM …
        ⊢ Eq (Polynomial.derivative (Polynomial.eval₂ Polynomial.C q (HAdd.hAdd p₁ p₂) …
      -/
      rw [eval₂_add, derivative_add, ih₁, ih₂, derivative_add, eval₂_add, add_mul])
      /-
        🎉 no goals
      -/
    fun n r ih => by
    rw [pow_succ, ← mul_assoc, eval₂_mul, eval₂_X, derivative_mul, ih, @derivative_mul _ _ _ X,
      derivative_X, mul_one, eval₂_add, @eval₂_mul _ _ _ _ X, eval₂_X, add_mul, mul_right_comm]


theorem derivative_prod [DecidableEq ι] {s : Multiset ι} {f : ι → R[X]} :
    derivative (Multiset.map f s).prod =
      (Multiset.map (fun i => (Multiset.map f (s.erase i)).prod * derivative (f i)) s).sum := by
  /-
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s : Multiset ι
    f : ι → Polynomial R
    ⊢ Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => H …
  -/
  refine Multiset.induction_on s (by simp) fun i s h => ?_
  rw [Multiset.map_cons, Multiset.prod_cons, derivative_mul, Multiset.map_cons _ i s,
    Multiset.sum_cons, Multiset.erase_cons_head, mul_comm (derivative (f i))]
  /-
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s✝ : Multiset ι
    f : ι → Polynomial R
    i : ι
    s : Multiset ι
    h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Multiset.map f s).prod (Polynomial.derivative (f i …
  -/
  congr
  rw [h, ← AddMonoidHom.coe_mulLeft, (AddMonoidHom.mulLeft (f i)).map_multiset_sum _,
    AddMonoidHom.coe_mulLeft]
  /-
    case e_a
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s✝ : Multiset ι
    f : ι → Polynomial R
    i : ι
    s : Multiset ι
    h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
    ⊢ Eq (Multiset.map (HMul.hMul (f i)) (Multiset.map (fun i => HMul.hMul (Multis …
  -/
  simp only [Function.comp_apply, Multiset.map_map]
  /-
    case e_a
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s✝ : Multiset ι
    f : ι → Polynomial R
    i : ι
    s : Multiset ι
    h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
    ⊢ Eq (Multiset.map (fun x => HMul.hMul (f i) (HMul.hMul (Multiset.map f (s.era …
  -/
  refine congr_arg _ (Multiset.map_congr rfl fun j hj => ?_)
  /-
    case e_a
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s✝ : Multiset ι
    f : ι → Polynomial R
    i : ι
    s : Multiset ι
    h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
    j : ι
    hj : Membership.mem s j
    ⊢ Eq (HMul.hMul (f i) (HMul.hMul (Multiset.map f (s.erase j)).prod (Polynomial …
  -/
  rw [← mul_assoc, ← Multiset.prod_cons, ← Multiset.map_cons]
  /-
    case e_a
    R : Type u
    ι : Type y
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq ι
    s✝ : Multiset ι
    f : ι → Polynomial R
    i : ι
    s : Multiset ι
    h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
    j : ι
    hj : Membership.mem s j
    ⊢ Eq (HMul.hMul (Multiset.map f (Multiset.cons i (s.erase j))).prod (Polynomia …
  -/
  by_cases hij : i = j
    /-
      case pos
      R : Type u
      ι : Type y
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq ι
      s✝ : Multiset ι
      f : ι → Polynomial R
      i : ι
      s : Multiset ι
      h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
      j : ι
      hj : Membership.mem s j
      hij : Eq i j
      ⊢ Eq (HMul.hMul (Multiset.map f (Multiset.cons i (s.erase j))).prod (Polynomia …
    -/
  · simp [hij, ← Multiset.prod_cons, ← Multiset.map_cons, Multiset.cons_erase hj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      ι : Type y
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq ι
      s✝ : Multiset ι
      f : ι → Polynomial R
      i : ι
      s : Multiset ι
      h : Eq (Polynomial.derivative (Multiset.map f s).prod) (Multiset.map (fun i => …
      j : ι
      hj : Membership.mem s j
      hij : Not (Eq i j)
      ⊢ Eq (HMul.hMul (Multiset.map f (Multiset.cons i (s.erase j))).prod (Polynomia …
    -/
  · simp [hij]
    /-
      🎉 no goals
    -/


@[simp]
theorem derivative_neg (f : R[X]) : derivative (-f) = -derivative f :=
  LinearMap.map_neg derivative f


theorem iterate_derivative_neg {f : R[X]} {k : ℕ} : derivative^[k] (-f) = -derivative^[k] f :=
  iterate_map_neg derivative k f


@[simp]
theorem derivative_sub {f g : R[X]} : derivative (f - g) = derivative f - derivative g :=
  LinearMap.map_sub derivative f g


theorem derivative_X_sub_C (c : R) : derivative (X - C c) = 1 := by
  /-
    R : Type u
    inst✝ : Ring R
    c : R
    ⊢ Eq (Polynomial.derivative (HSub.hSub Polynomial.X (Polynomial.C c))) 1
  -/
  rw [derivative_sub, derivative_X, derivative_C, sub_zero]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_sub {k : ℕ} {f g : R[X]} :
    derivative^[k] (f - g) = derivative^[k] f - derivative^[k] g :=
  iterate_map_sub derivative k f g


@[simp]
theorem derivative_intCast {n : ℤ} : derivative (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    ⊢ Eq (Polynomial.derivative ↑n) 0
  -/
  rw [← C_eq_intCast n]
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    ⊢ Eq (Polynomial.derivative (Polynomial.C ↑n)) 0
  -/
  exact derivative_C
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias derivative_int_cast := derivative_intCast


theorem derivative_intCast_mul {n : ℤ} {f : R[X]} : derivative ((n : R[X]) * f) =
    n * derivative f := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    f : Polynomial R
    ⊢ Eq (Polynomial.derivative (HMul.hMul (↑n) f)) (HMul.hMul (↑n) (Polynomial.de …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias derivative_int_cast_mul := derivative_intCast_mul


@[simp]
theorem iterate_derivative_intCast_mul {n : ℤ} {k : ℕ} {f : R[X]} :
    derivative^[k] ((n : R[X]) * f) = n * derivative^[k] f := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    k : Nat
    f : Polynomial R
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HMul.hMul (↑n) f)) (HMul.hMul (↑ …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  induction' k with k ih generalizing f <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-04-17")]
alias iterate_derivative_int_cast_mul := iterate_derivative_intCast_mul


theorem derivative_comp_one_sub_X (p : R[X]) :
                                                                   /-
                                                                     R : Type u
                                                                     inst✝ : CommRing R
                                                                     p : Polynomial R
                                                                     ⊢ Eq (Polynomial.derivative (p.comp (HSub.hSub 1 Polynomial.X))) (Neg.neg ((Po …
                                                                   -/
    derivative (p.comp (1 - X)) = -p.derivative.comp (1 - X) := by simp [derivative_comp]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem iterate_derivative_comp_one_sub_X (p : R[X]) (k : ℕ) :
    derivative^[k] (p.comp (1 - X)) = (-1) ^ k * (derivative^[k] p).comp (1 - X) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    k : Nat
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (p.comp (HSub.hSub 1 Polynomial.X …
  -/
  induction' k with k ih generalizing p
    /-
      case zero
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) 0 (p.comp (HSub.hSub 1 Polynomial.X …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : CommRing R
      k : Nat
      ih : ∀ (p : Polynomial R), Eq (Nat.iterate (⇑Polynomial.derivative) k (p.comp  …
      p : Polynomial R
      ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) (HAdd.hAdd k 1) (p.comp (HSub.hSub  …
    -/
  · simp [ih (derivative p), iterate_derivative_neg, derivative_comp, pow_succ]
    /-
      🎉 no goals
    -/


theorem eval_multiset_prod_X_sub_C_derivative [DecidableEq R]
    {S : Multiset R} {r : R} (hr : r ∈ S) :
    eval r (derivative (Multiset.map (fun a => X - C a) S).prod) =
      (Multiset.map (fun a => r - a) (S.erase r)).prod := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    S : Multiset R
    r : R
    hr : Membership.mem S r
    ⊢ Eq (Polynomial.eval r (Polynomial.derivative (Multiset.map (fun a => HSub.hS …
  -/
  nth_rw 1 [← Multiset.cons_erase hr]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    S : Multiset R
    r : R
    hr : Membership.mem S r
    ⊢ Eq (Polynomial.eval r (Polynomial.derivative (Multiset.map (fun a => HSub.hS …
  -/
  have := (evalRingHom r).map_multiset_prod (Multiset.map (fun a => X - C a) (S.erase r))
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    S : Multiset R
    r : R
    hr : Membership.mem S r
    this : Eq (↑(Polynomial.evalRingHom r) (Multiset.map (fun a => HSub.hSub Polyn …
    ⊢ Eq (Polynomial.eval r (Polynomial.derivative (Multiset.map (fun a => HSub.hS …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem derivative_X_sub_C_pow (c : R) (m : ℕ) :
    derivative ((X - C c) ^ m) = C (m : R) * (X - C c) ^ (m - 1) := by
  /-
    R : Type u
    inst✝ : CommRing R
    c : R
    m : Nat
    ⊢ Eq (Polynomial.derivative (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C c …
  -/
  rw [derivative_pow, derivative_X_sub_C, mul_one]
  /-
    🎉 no goals
  -/


theorem derivative_X_sub_C_sq (c : R) : derivative ((X - C c) ^ 2) = C 2 * (X - C c) := by
  /-
    R : Type u
    inst✝ : CommRing R
    c : R
    ⊢ Eq (Polynomial.derivative (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C c …
  -/
  rw [derivative_sq, derivative_X_sub_C, mul_one]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_X_sub_pow (n k : ℕ) (c : R) :
    derivative^[k] ((X - C c) ^ n) = n.descFactorial k • (X - C c) ^ (n - k) := by
  /-
    R : Type u
    inst✝ : CommRing R
    n k : Nat
    c : R
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) k (HPow.hPow (HSub.hSub Polynomial. …
  -/
  rw [sub_eq_add_neg, ← C_neg, iterate_derivative_X_add_pow]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_X_sub_pow_self (n : ℕ) (c : R) :
    derivative^[n] ((X - C c) ^ n) = n.factorial := by
  /-
    R : Type u
    inst✝ : CommRing R
    n : Nat
    c : R
    ⊢ Eq (Nat.iterate (⇑Polynomial.derivative) n (HPow.hPow (HSub.hSub Polynomial. …
  -/
  rw [iterate_derivative_X_sub_pow, n.sub_self, pow_zero, nsmul_one, n.descFactorial_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem dvd_derivative_iff {P : R[X]} : P ∣ derivative P ↔ derivative P = 0 where
  mp h := by
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      P : Polynomial R
      h : Dvd.dvd P (Polynomial.derivative P)
      ⊢ Eq (Polynomial.derivative P) 0
    -/
    by_cases hP : P = 0
      /-
        case pos
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroDivisors R
        P : Polynomial R
        h : Dvd.dvd P (Polynomial.derivative P)
        hP : Eq P 0
        ⊢ Eq (Polynomial.derivative P) 0
      -/
    · simp only [hP, derivative_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      P : Polynomial R
      h : Dvd.dvd P (Polynomial.derivative P)
      hP : Not (Eq P 0)
      ⊢ Eq (Polynomial.derivative P) 0
    -/
    exact eq_zero_of_dvd_of_degree_lt h (degree_derivative_lt hP)
    /-
      🎉 no goals
    -/
              /-
                R : Type u
                inst✝¹ : Semiring R
                inst✝ : NoZeroDivisors R
                P : Polynomial R
                h : Eq (Polynomial.derivative P) 0
                ⊢ Dvd.dvd P (Polynomial.derivative P)
              -/
  mpr h := by simp [h]
              /-
                🎉 no goals
              -/


theorem derivative_pow_eq_zero {n : ℕ} (chn : (n : R) ≠ 0) {a : R[X]} :
    derivative (a ^ n) = 0 ↔ derivative a = 0 := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    n : Nat
    chn : Ne (↑n) 0
    a : Polynomial R
    ⊢ Iff (Eq (Polynomial.derivative (HPow.hPow a n)) 0) (Eq (Polynomial.derivativ …
  -/
  nontriviality R
  /-
    R : Type u
    inst✝² : CommSemiring R
    inst✝¹ : NoZeroDivisors R
    n : Nat
    chn : Ne (↑n) 0
    a : Polynomial R
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (Polynomial.derivative (HPow.hPow a n)) 0) (Eq (Polynomial.derivativ …
  -/
  rw [← C_ne_zero, C_eq_natCast] at chn
  /-
    R : Type u
    inst✝² : CommSemiring R
    inst✝¹ : NoZeroDivisors R
    n : Nat
    chn : Ne (↑n) 0
    a : Polynomial R
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (Polynomial.derivative (HPow.hPow a n)) 0) (Eq (Polynomial.derivativ …
  -/
  simp +contextual [derivative_pow, or_imp, chn]
  /-
    🎉 no goals
  -/


