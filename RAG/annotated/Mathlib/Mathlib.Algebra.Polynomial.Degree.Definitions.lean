/-- `degree p` is the degree of the polynomial `p`, i.e. the largest `X`-exponent in `p`.
`degree p = some n` when `p ≠ 0` and `n` is the highest power of `X` that appears in `p`, otherwise
`degree 0 = ⊥`. -/
def degree (p : R[X]) : WithBot ℕ :=
  p.support.max


/-- `natDegree p` forces `degree p` to ℕ, by defining `natDegree 0 = 0`. -/
def natDegree (p : R[X]) : ℕ :=
  (degree p).unbot' 0


/-- `leadingCoeff p` gives the coefficient of the highest power of `X` in `p`-/
def leadingCoeff (p : R[X]) : R :=
  coeff p (natDegree p)


/-- a polynomial is `Monic` if its leading coefficient is 1 -/
def Monic (p : R[X]) :=
  leadingCoeff p = (1 : R)


theorem Monic.def : Monic p ↔ leadingCoeff p = 1 :=
  Iff.rfl


                                                                     /-
                                                                       R : Type u
                                                                       S : Type v
                                                                       a b c d : R
                                                                       n m : Nat
                                                                       inst✝¹ : Semiring R
                                                                       p q r : Polynomial R
                                                                       inst✝ : DecidableEq R
                                                                       ⊢ Decidable p.Monic
                                                                     -/
instance Monic.decidable [DecidableEq R] : Decidable (Monic p) := by unfold Monic; infer_instance
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
theorem Monic.leadingCoeff {p : R[X]} (hp : p.Monic) : leadingCoeff p = 1 :=
  hp


theorem Monic.coeff_natDegree {p : R[X]} (hp : p.Monic) : p.coeff p.natDegree = 1 :=
  hp


@[simp]
theorem degree_zero : degree (0 : R[X]) = ⊥ :=
  rfl


@[simp]
theorem natDegree_zero : natDegree (0 : R[X]) = 0 :=
  rfl


@[simp]
theorem coeff_natDegree : coeff p (natDegree p) = leadingCoeff p :=
  rfl


@[simp]
theorem degree_eq_bot : degree p = ⊥ ↔ p = 0 :=
  ⟨fun h => support_eq_empty.1 (Finset.max_eq_bot.1 h), fun h => h.symm ▸ rfl⟩


theorem degree_ne_bot : degree p ≠ ⊥ ↔ p ≠ 0 := degree_eq_bot.not


theorem degree_eq_natDegree (hp : p ≠ 0) : degree p = (natDegree p : WithBot ℕ) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    ⊢ Eq p.degree ↑p.natDegree
  -/
  let ⟨n, hn⟩ := not_forall.1 (mt Option.eq_none_iff_forall_not_mem.2 (mt degree_eq_bot.1 hp))
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    n : Nat
    hn : Not (Not (Membership.mem p.degree n))
    ⊢ Eq p.degree ↑p.natDegree
  -/
  have hn : degree p = some n := Classical.not_not.1 hn
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    n : Nat
    hn✝ : Not (Not (Membership.mem p.degree n))
    hn : Eq p.degree (Option.some n)
    ⊢ Eq p.degree ↑p.natDegree
  -/
  rw [natDegree, hn]; rfl
                      /-
                        🎉 no goals
                      -/


theorem degree_eq_iff_natDegree_eq {p : R[X]} {n : ℕ} (hp : p ≠ 0) :
                                         /-
                                           R : Type u
                                           inst✝ : Semiring R
                                           p : Polynomial R
                                           n : Nat
                                           hp : Ne p 0
                                           ⊢ Iff (Eq p.degree ↑n) (Eq p.natDegree n)
                                         -/
    p.degree = n ↔ p.natDegree = n := by rw [degree_eq_natDegree hp]; exact WithBot.coe_eq_coe
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem degree_eq_iff_natDegree_eq_of_pos {p : R[X]} {n : ℕ} (hn : 0 < n) :
    p.degree = n ↔ p.natDegree = n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (Eq p.degree ↑n) (Eq p.natDegree n)
  -/
  obtain rfl|h := eq_or_ne p 0
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      n : Nat
      hn : LT.lt 0 n
      ⊢ Iff (Eq (Polynomial.degree 0) ↑n) (Eq (Polynomial.natDegree 0) n)
    -/
  · simp [hn.ne]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : LT.lt 0 n
      h : Ne p 0
      ⊢ Iff (Eq p.degree ↑n) (Eq p.natDegree n)
    -/
  · exact degree_eq_iff_natDegree_eq h
    /-
      🎉 no goals
    -/


theorem natDegree_eq_of_degree_eq_some {p : R[X]} {n : ℕ} (h : degree p = n) : natDegree p = n := by
  -- Porting note: `Nat.cast_withBot` is required.
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : Eq p.degree ↑n
    ⊢ Eq p.natDegree n
  -/
  rw [natDegree, h, Nat.cast_withBot, WithBot.unbot'_coe]
  /-
    🎉 no goals
  -/


theorem degree_ne_of_natDegree_ne {n : ℕ} : p.natDegree ≠ n → degree p ≠ n :=
  mt natDegree_eq_of_degree_eq_some


@[simp]
theorem degree_le_natDegree : degree p ≤ natDegree p :=
  WithBot.giUnbot'Bot.gc.le_u_l _


theorem natDegree_eq_of_degree_eq [Semiring S] {q : S[X]} (h : degree p = degree q) :
                                    /-
                                      R : Type u
                                      S : Type v
                                      inst✝¹ : Semiring R
                                      p : Polynomial R
                                      inst✝ : Semiring S
                                      q : Polynomial S
                                      h : Eq p.degree q.degree
                                      ⊢ Eq p.natDegree q.natDegree
                                    -/
    natDegree p = natDegree q := by unfold natDegree; rw [h]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem le_degree_of_ne_zero (h : coeff p n ≠ 0) : (n : WithBot ℕ) ≤ degree p := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (p.coeff n) 0
    ⊢ LE.le (↑n) p.degree
  -/
  rw [Nat.cast_withBot]
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (p.coeff n) 0
    ⊢ LE.le (↑n) p.degree
  -/
  exact Finset.le_sup (mem_support_iff.2 h)
  /-
    🎉 no goals
  -/


theorem degree_mono [Semiring S] {f : R[X]} {g : S[X]} (h : f.support ⊆ g.support) :
    f.degree ≤ g.degree :=
  Finset.sup_mono h


theorem degree_le_degree (h : coeff q (natDegree p) ≠ 0) : degree p ≤ degree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (q.coeff p.natDegree) 0
    ⊢ LE.le p.degree q.degree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (q.coeff p.natDegree) 0
      hp : Eq p 0
      ⊢ LE.le p.degree q.degree
    -/
  · rw [hp, degree_zero]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (q.coeff p.natDegree) 0
      hp : Eq p 0
      ⊢ LE.le Bot.bot q.degree
    -/
    exact bot_le
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (q.coeff p.natDegree) 0
      hp : Not (Eq p 0)
      ⊢ LE.le p.degree q.degree
    -/
  · rw [degree_eq_natDegree hp]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (q.coeff p.natDegree) 0
      hp : Not (Eq p 0)
      ⊢ LE.le (↑p.natDegree) q.degree
    -/
    exact le_degree_of_ne_zero h
    /-
      🎉 no goals
    -/


theorem natDegree_le_iff_degree_le {n : ℕ} : natDegree p ≤ n ↔ degree p ≤ n :=
  WithBot.unbot'_le_iff (fun _ ↦ bot_le)


theorem natDegree_lt_iff_degree_lt (hp : p ≠ 0) : p.natDegree < n ↔ p.degree < ↑n :=
  WithBot.unbot'_lt_iff (absurd · (degree_eq_bot.not.mpr hp))


alias ⟨degree_le_of_natDegree_le, natDegree_le_of_degree_le⟩ := natDegree_le_iff_degree_le


theorem natDegree_le_natDegree [Semiring S] {q : S[X]} (hpq : p.degree ≤ q.degree) :
    p.natDegree ≤ q.natDegree :=
  WithBot.giUnbot'Bot.gc.monotone_l hpq


@[simp]
theorem degree_C (ha : a ≠ 0) : degree (C a) = (0 : WithBot ℕ) := by
  rw [degree, ← monomial_zero_left, support_monomial 0 ha, max_eq_sup_coe, sup_singleton,
    WithBot.coe_zero]


theorem degree_C_le : degree (C a) ≤ 0 := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    ⊢ LE.le (Polynomial.C a).degree 0
  -/
  by_cases h : a = 0
    /-
      case pos
      R : Type u
      a : R
      inst✝ : Semiring R
      h : Eq a 0
      ⊢ LE.le (Polynomial.C a).degree 0
    -/
  · rw [h, C_0]
    /-
      case pos
      R : Type u
      a : R
      inst✝ : Semiring R
      h : Eq a 0
      ⊢ LE.le (Polynomial.degree 0) 0
    -/
    exact bot_le
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      a : R
      inst✝ : Semiring R
      h : Not (Eq a 0)
      ⊢ LE.le (Polynomial.C a).degree 0
    -/
  · rw [degree_C h]
    /-
      🎉 no goals
    -/


theorem degree_C_lt : degree (C a) < 1 :=
  degree_C_le.trans_lt <| WithBot.coe_lt_coe.mpr zero_lt_one


                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : Semiring R
                                                                    ⊢ LE.le (Polynomial.degree 1) 0
                                                                  -/
theorem degree_one_le : degree (1 : R[X]) ≤ (0 : WithBot ℕ) := by rw [← C_1]; exact degree_C_le
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem natDegree_C (a : R) : natDegree (C a) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ Eq (Polynomial.C a).natDegree 0
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      a : R
      ha : Eq a 0
      ⊢ Eq (Polynomial.C a).natDegree 0
    -/
  · have : C a = 0 := by rw [ha, C_0]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      a : R
      ha : Eq a 0
      this : Eq (Polynomial.C a) 0
      ⊢ Eq (Polynomial.C a).natDegree 0
    -/
    rw [natDegree, degree_eq_bot.2 this, WithBot.unbot'_bot]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      a : R
      ha : Not (Eq a 0)
      ⊢ Eq (Polynomial.C a).natDegree 0
    -/
  · rw [natDegree, degree_C ha, WithBot.unbot_zero']
    /-
      🎉 no goals
    -/


@[simp]
theorem natDegree_one : natDegree (1 : R[X]) = 0 :=
  natDegree_C 1


@[simp]
theorem natDegree_natCast (n : ℕ) : natDegree (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (↑n).natDegree 0
  -/
  simp only [← C_eq_natCast, natDegree_C]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias natDegree_nat_cast := natDegree_natCast


@[simp]
theorem natDegree_ofNat (n : ℕ) [Nat.AtLeastTwo n] :
    natDegree (ofNat(n) : R[X]) = 0 :=
  natDegree_natCast _


                                                                                           /-
                                                                                             R : Type u
                                                                                             inst✝ : Semiring R
                                                                                             n : Nat
                                                                                             ⊢ LE.le (↑n).natDegree Zero.zero
                                                                                           -/
theorem degree_natCast_le (n : ℕ) : degree (n : R[X]) ≤ 0 := degree_le_of_natDegree_le (by simp)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[deprecated (since := "2024-04-17")]
alias degree_nat_cast_le := degree_natCast_le


@[simp]
theorem degree_monomial (n : ℕ) (ha : a ≠ 0) : degree (monomial n a) = n := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    n : Nat
    ha : Ne a 0
    ⊢ Eq ((Polynomial.monomial n) a).degree ↑n
  -/
  rw [degree, support_monomial n ha, max_singleton, Nat.cast_withBot]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_C_mul_X_pow (n : ℕ) (ha : a ≠ 0) : degree (C a * X ^ n) = n := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    n : Nat
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).degree ↑n
  -/
  rw [C_mul_X_pow_eq_monomial, degree_monomial n ha]
  /-
    🎉 no goals
  -/


theorem degree_C_mul_X (ha : a ≠ 0) : degree (C a * X) = 1 := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) Polynomial.X).degree 1
  -/
  simpa only [pow_one] using degree_C_mul_X_pow 1 ha
  /-
    🎉 no goals
  -/


theorem degree_monomial_le (n : ℕ) (a : R) : degree (monomial n a) ≤ n :=
  letI := Classical.decEq R
                       /-
                         R : Type u
                         inst✝ : Semiring R
                         n : Nat
                         a : R
                         this : DecidableEq R := Classical.decEq R
                         h : Eq a 0
                         ⊢ LE.le ((Polynomial.monomial n) a).degree ↑n
                       -/
  if h : a = 0 then by rw [h, (monomial n).map_zero, degree_zero]; exact bot_le
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  else le_of_eq (degree_monomial n h)


theorem degree_C_mul_X_pow_le (n : ℕ) (a : R) : degree (C a * X ^ n) ≤ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ LE.le (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).degree ↑n
  -/
  rw [C_mul_X_pow_eq_monomial]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ LE.le ((Polynomial.monomial n) a).degree ↑n
  -/
  apply degree_monomial_le
  /-
    🎉 no goals
  -/


theorem degree_C_mul_X_le (a : R) : degree (C a * X) ≤ 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ LE.le (HMul.hMul (Polynomial.C a) Polynomial.X).degree 1
  -/
  simpa only [pow_one] using degree_C_mul_X_pow_le 1 a
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_C_mul_X_pow (n : ℕ) (a : R) (ha : a ≠ 0) : natDegree (C a * X ^ n) = n :=
  natDegree_eq_of_degree_eq_some (degree_C_mul_X_pow n ha)


@[simp]
theorem natDegree_C_mul_X (a : R) (ha : a ≠ 0) : natDegree (C a * X) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) Polynomial.X).natDegree 1
  -/
  simpa only [pow_one] using natDegree_C_mul_X_pow 1 a ha
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_monomial [DecidableEq R] (i : ℕ) (r : R) :
    natDegree (monomial i r) = if r = 0 then 0 else i := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    i : Nat
    r : R
    ⊢ Eq ((Polynomial.monomial i) r).natDegree (ite (Eq r 0) 0 i)
  -/
  split_ifs with hr
    /-
      case pos
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      i : Nat
      r : R
      hr : Eq r 0
      ⊢ Eq ((Polynomial.monomial i) r).natDegree 0
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      i : Nat
      r : R
      hr : Not (Eq r 0)
      ⊢ Eq ((Polynomial.monomial i) r).natDegree i
    -/
  · rw [← C_mul_X_pow_eq_monomial, natDegree_C_mul_X_pow i r hr]
    /-
      🎉 no goals
    -/


theorem natDegree_monomial_le (a : R) {m : ℕ} : (monomial m a).natDegree ≤ m := by
  classical
  rw [Polynomial.natDegree_monomial]
  split_ifs
  exacts [Nat.zero_le _, le_rfl]


theorem natDegree_monomial_eq (i : ℕ) {r : R} (r0 : r ≠ 0) : (monomial i r).natDegree = i :=
  letI := Classical.decEq R
  Eq.trans (natDegree_monomial _ _) (if_neg r0)


theorem coeff_ne_zero_of_eq_degree (hn : degree p = n) : coeff p n ≠ 0 := fun h =>
  mem_support_iff.mp (mem_of_max hn) h


theorem degree_X_pow_le (n : ℕ) : degree (X ^ n : R[X]) ≤ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ LE.le (HPow.hPow Polynomial.X n).degree ↑n
  -/
  simpa only [C_1, one_mul] using degree_C_mul_X_pow_le n (1 : R)
  /-
    🎉 no goals
  -/


theorem degree_X_le : degree (X : R[X]) ≤ 1 :=
  degree_monomial_le _ _


theorem natDegree_X_le : (X : R[X]).natDegree ≤ 1 :=
  natDegree_le_of_degree_le degree_X_le


@[simp]
theorem degree_one : degree (1 : R[X]) = (0 : WithBot ℕ) :=
  degree_C one_ne_zero


@[simp]
theorem degree_X : degree (X : R[X]) = 1 :=
  degree_monomial _ one_ne_zero


@[simp]
theorem natDegree_X : (X : R[X]).natDegree = 1 :=
  natDegree_eq_of_degree_eq_some degree_X


@[simp]
                                                             /-
                                                               R : Type u
                                                               inst✝ : Ring R
                                                               p : Polynomial R
                                                               ⊢ Eq (Neg.neg p).degree p.degree
                                                             -/
theorem degree_neg (p : R[X]) : degree (-p) = degree p := by unfold degree; rw [support_neg]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem degree_neg_le_of_le {a : WithBot ℕ} {p : R[X]} (hp : degree p ≤ a) : degree (-p) ≤ a :=
  p.degree_neg.le.trans hp


@[simp]
                                                                      /-
                                                                        R : Type u
                                                                        inst✝ : Ring R
                                                                        p : Polynomial R
                                                                        ⊢ Eq (Neg.neg p).natDegree p.natDegree
                                                                      -/
theorem natDegree_neg (p : R[X]) : natDegree (-p) = natDegree p := by simp [natDegree]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem natDegree_neg_le_of_le {p : R[X]} (hp : natDegree p ≤ m) : natDegree (-p) ≤ m :=
  (natDegree_neg p).le.trans hp


@[simp]
theorem natDegree_intCast (n : ℤ) : natDegree (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    ⊢ Eq (↑n).natDegree 0
  -/
  rw [← C_eq_intCast, natDegree_C]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias natDegree_int_cast := natDegree_intCast


                                                                                           /-
                                                                                             R : Type u
                                                                                             inst✝ : Ring R
                                                                                             n : Int
                                                                                             ⊢ LE.le (↑n).natDegree Zero.zero
                                                                                           -/
theorem degree_intCast_le (n : ℤ) : degree (n : R[X]) ≤ 0 := degree_le_of_natDegree_le (by simp)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[deprecated (since := "2024-04-17")]
alias degree_int_cast_le := degree_intCast_le


@[simp]
theorem leadingCoeff_neg (p : R[X]) : (-p).leadingCoeff = -p.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).leadingCoeff (Neg.neg p.leadingCoeff)
  -/
  rw [leadingCoeff, leadingCoeff, natDegree_neg, coeff_neg]
  /-
    🎉 no goals
  -/


/-- The second-highest coefficient, or 0 for constants -/
def nextCoeff (p : R[X]) : R :=
  if p.natDegree = 0 then 0 else p.coeff (p.natDegree - 1)


lemma nextCoeff_eq_zero :
    p.nextCoeff = 0 ↔ p.natDegree = 0 ∨ 0 < p.natDegree ∧ p.coeff (p.natDegree - 1) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.nextCoeff 0) (Or (Eq p.natDegree 0) (And (LT.lt 0 p.natDegree) (Eq …
  -/
  simp [nextCoeff, or_iff_not_imp_left, pos_iff_ne_zero]; aesop
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma nextCoeff_ne_zero : p.nextCoeff ≠ 0 ↔ p.natDegree ≠ 0 ∧ p.coeff (p.natDegree - 1) ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Ne p.nextCoeff 0) (And (Ne p.natDegree 0) (Ne (p.coeff (HSub.hSub p.nat …
  -/
  simp [nextCoeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem nextCoeff_C_eq_zero (c : R) : nextCoeff (C c) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ Eq (Polynomial.C c).nextCoeff 0
  -/
  rw [nextCoeff]
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ Eq (ite (Eq (Polynomial.C c).natDegree 0) 0 ((Polynomial.C c).coeff (HSub.hS …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nextCoeff_of_natDegree_pos (hp : 0 < p.natDegree) :
    nextCoeff p = p.coeff (p.natDegree - 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : LT.lt 0 p.natDegree
    ⊢ Eq p.nextCoeff (p.coeff (HSub.hSub p.natDegree 1))
  -/
  rw [nextCoeff, if_neg]
  /-
    case hnc
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : LT.lt 0 p.natDegree
    ⊢ Not (Eq p.natDegree 0)
  -/
  contrapose! hp
  /-
    case hnc
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Eq p.natDegree 0
    ⊢ LE.le p.natDegree 0
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem degree_add_le (p q : R[X]) : degree (p + q) ≤ max (degree p) (degree q) := by
  simpa only [degree, ← support_toFinsupp, toFinsupp_add]
    using AddMonoidAlgebra.sup_support_add_le _ _ _


theorem degree_add_le_of_degree_le {p q : R[X]} {n : ℕ} (hp : degree p ≤ n) (hq : degree q ≤ n) :
    degree (p + q) ≤ n :=
  (degree_add_le p q).trans <| max_le hp hq


theorem degree_add_le_of_le {a b : WithBot ℕ} (hp : degree p ≤ a) (hq : degree q ≤ b) :
    degree (p + q) ≤ max a b :=
  (p.degree_add_le q).trans <| max_le_max ‹_› ‹_›


theorem natDegree_add_le (p q : R[X]) : natDegree (p + q) ≤ max (natDegree p) (natDegree q) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HAdd.hAdd p q).natDegree (Max.max p.natDegree q.natDegree)
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  cases' le_max_iff.1 (degree_add_le p q) with h h <;> simp [natDegree_le_natDegree h]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem natDegree_add_le_of_degree_le {p q : R[X]} {n : ℕ} (hp : natDegree p ≤ n)
    (hq : natDegree q ≤ n) : natDegree (p + q) ≤ n :=
  (natDegree_add_le p q).trans <| max_le hp hq


theorem natDegree_add_le_of_le (hp : natDegree p ≤ m) (hq : natDegree q ≤ n) :
    natDegree (p + q) ≤ max m n :=
  (p.natDegree_add_le q).trans <| max_le_max ‹_› ‹_›


@[simp]
theorem leadingCoeff_zero : leadingCoeff (0 : R[X]) = 0 :=
  rfl


@[simp]
theorem leadingCoeff_eq_zero : leadingCoeff p = 0 ↔ p = 0 :=
  ⟨fun h =>
    Classical.by_contradiction fun hp =>
      mt mem_support_iff.1 (Classical.not_not.2 h) (mem_of_max (degree_eq_natDegree hp)),
    fun h => h.symm ▸ leadingCoeff_zero⟩


                                                                /-
                                                                  R : Type u
                                                                  inst✝ : Semiring R
                                                                  p : Polynomial R
                                                                  ⊢ Iff (Ne p.leadingCoeff 0) (Ne p 0)
                                                                -/
theorem leadingCoeff_ne_zero : leadingCoeff p ≠ 0 ↔ p ≠ 0 := by rw [Ne, leadingCoeff_eq_zero]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem leadingCoeff_eq_zero_iff_deg_eq_bot : leadingCoeff p = 0 ↔ degree p = ⊥ := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.leadingCoeff 0) (Eq p.degree Bot.bot)
  -/
  rw [leadingCoeff_eq_zero, degree_eq_bot]
  /-
    🎉 no goals
  -/


theorem natDegree_C_mul_X_pow_le (a : R) (n : ℕ) : natDegree (C a * X ^ n) ≤ n :=
  natDegree_le_iff_degree_le.2 <| degree_C_mul_X_pow_le _ _


theorem degree_erase_le (p : R[X]) (n : ℕ) : degree (p.erase n) ≤ degree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ LE.le (Polynomial.erase n p).degree p.degree
  -/
  rcases p with ⟨p⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : AddMonoidAlgebra R Nat
    ⊢ LE.le (Polynomial.erase n { toFinsupp := p }).degree { toFinsupp := p }.degree
  -/
  simp only [erase_def, degree, coeff, support]
  -- Porting note: simpler convert-free proof to be explicit about definition unfolding
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : AddMonoidAlgebra R Nat
    ⊢ LE.le (Finsupp.erase n p).support.max p.support.max
  -/
  apply sup_mono
  /-
    case ofFinsupp.h
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : AddMonoidAlgebra R Nat
    ⊢ HasSubset.Subset (Finsupp.erase n p).support p.support
  -/
  rw [Finsupp.support_erase]
  /-
    case ofFinsupp.h
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : AddMonoidAlgebra R Nat
    ⊢ HasSubset.Subset (p.support.erase n) p.support
  -/
  apply Finset.erase_subset
  /-
    🎉 no goals
  -/


theorem degree_erase_lt (hp : p ≠ 0) : degree (p.erase (natDegree p)) < degree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    ⊢ LT.lt (Polynomial.erase p.natDegree p).degree p.degree
  -/
  apply lt_of_le_of_ne (degree_erase_le _ _)
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    ⊢ Ne (Polynomial.erase p.natDegree p).degree p.degree
  -/
  rw [degree_eq_natDegree hp, degree, support_erase]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    ⊢ Ne (p.support.erase p.natDegree).max ↑p.natDegree
  -/
  exact fun h => not_mem_erase _ _ (mem_of_max h)
  /-
    🎉 no goals
  -/


theorem degree_update_le (p : R[X]) (n : ℕ) (a : R) : degree (p.update n a) ≤ max (degree p) n := by
  classical
  rw [degree, support_update]
  split_ifs
  · exact (Finset.max_mono (erase_subset _ _)).trans (le_max_left _ _)
  · rw [max_insert, max_comm]
    exact le_rfl


theorem degree_sum_le (s : Finset ι) (f : ι → R[X]) :
    degree (∑ i ∈ s, f i) ≤ s.sup fun b => degree (f b) :=
                                 /-
                                   R : Type u
                                   inst✝ : Semiring R
                                   ι : Type u_1
                                   s : Finset ι
                                   f : ι → Polynomial R
                                   ⊢ LE.le (EmptyCollection.emptyCollection.sum fun i => f i).degree (EmptyCollec …
                                 -/
  Finset.cons_induction_on s (by simp only [sum_empty, sup_empty, degree_zero, le_refl])
                                 /-
                                   🎉 no goals
                                 -/
    fun a s has ih =>
    calc
      degree (∑ i ∈ cons a s has, f i) ≤ max (degree (f a)) (degree (∑ i ∈ s, f i)) := by
        /-
          R : Type u
          inst✝ : Semiring R
          ι : Type u_1
          s✝ : Finset ι
          f : ι → Polynomial R
          a : ι
          s : Finset ι
          has : Not (Membership.mem s a)
          ih : LE.le (s.sum fun i => f i).degree (s.sup fun b => (f b).degree)
          ⊢ LE.le ((Finset.cons a s has).sum fun i => f i).degree (Max.max (f a).degree  …
        -/
        rw [Finset.sum_cons]; exact degree_add_le _ _
                              /-
                                🎉 no goals
                              -/
                  /-
                    R : Type u
                    inst✝ : Semiring R
                    ι : Type u_1
                    s✝ : Finset ι
                    f : ι → Polynomial R
                    a : ι
                    s : Finset ι
                    has : Not (Membership.mem s a)
                    ih : LE.le (s.sum fun i => f i).degree (s.sup fun b => (f b).degree)
                    ⊢ LE.le (Max.max (f a).degree (s.sum fun i => f i).degree) ((Finset.cons a s h …
                  -/
      _ ≤ _ := by rw [sup_cons]; exact max_le_max le_rfl ih
                                 /-
                                   🎉 no goals
                                 -/


theorem degree_mul_le (p q : R[X]) : degree (p * q) ≤ degree p + degree q := by
  simpa only [degree, ← support_toFinsupp, toFinsupp_mul]
    using AddMonoidAlgebra.sup_support_mul_le (WithBot.coe_add _ _).le _ _


theorem degree_mul_le_of_le {a b : WithBot ℕ} (hp : degree p ≤ a) (hq : degree q ≤ b) :
    degree (p * q) ≤ a + b :=
  (p.degree_mul_le _).trans <| add_le_add ‹_› ‹_›


theorem degree_pow_le (p : R[X]) : ∀ n : ℕ, degree (p ^ n) ≤ n • degree p
            /-
              R : Type u
              inst✝ : Semiring R
              p : Polynomial R
              ⊢ LE.le (HPow.hPow p 0).degree (HSMul.hSMul 0 p.degree)
            -/
  | 0 => by rw [pow_zero, zero_nsmul]; exact degree_one_le
                                       /-
                                         🎉 no goals
                                       -/
  | n + 1 =>
    calc
      degree (p ^ (n + 1)) ≤ degree (p ^ n) + degree p := by
        /-
          R : Type u
          inst✝ : Semiring R
          p : Polynomial R
          n : Nat
          ⊢ LE.le (HPow.hPow p (HAdd.hAdd n 1)).degree (HAdd.hAdd (HPow.hPow p n).degree …
        -/
        rw [pow_succ]; exact degree_mul_le _ _
                       /-
                         🎉 no goals
                       -/
                  /-
                    R : Type u
                    inst✝ : Semiring R
                    p : Polynomial R
                    n : Nat
                    ⊢ LE.le (HAdd.hAdd (HPow.hPow p n).degree p.degree) (HSMul.hSMul (HAdd.hAdd n  …
                  -/
      _ ≤ _ := by rw [succ_nsmul]; exact add_le_add_right (degree_pow_le _ _) _
                                   /-
                                     🎉 no goals
                                   -/


theorem degree_pow_le_of_le {a : WithBot ℕ} (b : ℕ) (hp : degree p ≤ a) :
    degree (p ^ b) ≤ b * a := by
  induction b with
  | zero => simp [degree_one_le]
  | succ n hn =>
      rw [Nat.cast_succ, add_mul, one_mul, pow_succ]
      exact degree_mul_le_of_le hn hp


@[simp]
theorem leadingCoeff_monomial (a : R) (n : ℕ) : leadingCoeff (monomial n a) = a := by
  classical
  by_cases ha : a = 0
  · simp only [ha, (monomial n).map_zero, leadingCoeff_zero]
  · rw [leadingCoeff, natDegree_monomial, if_neg ha, coeff_monomial]
    simp


theorem leadingCoeff_C_mul_X_pow (a : R) (n : ℕ) : leadingCoeff (C a * X ^ n) = a := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    n : Nat
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).leadingCoeff a
  -/
  rw [C_mul_X_pow_eq_monomial, leadingCoeff_monomial]
  /-
    🎉 no goals
  -/


theorem leadingCoeff_C_mul_X (a : R) : leadingCoeff (C a * X) = a := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ Eq (HMul.hMul (Polynomial.C a) Polynomial.X).leadingCoeff a
  -/
  simpa only [pow_one] using leadingCoeff_C_mul_X_pow a 1
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_C (a : R) : leadingCoeff (C a) = a :=
  leadingCoeff_monomial a 0


theorem leadingCoeff_X_pow (n : ℕ) : leadingCoeff ((X : R[X]) ^ n) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).leadingCoeff 1
  -/
  simpa only [C_1, one_mul] using leadingCoeff_C_mul_X_pow (1 : R) n
  /-
    🎉 no goals
  -/


theorem leadingCoeff_X : leadingCoeff (X : R[X]) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq Polynomial.X.leadingCoeff 1
  -/
  simpa only [pow_one] using @leadingCoeff_X_pow R _ 1
  /-
    🎉 no goals
  -/


@[simp]
theorem monic_X_pow (n : ℕ) : Monic (X ^ n : R[X]) :=
  leadingCoeff_X_pow n


@[simp]
theorem monic_X : Monic (X : R[X]) :=
  leadingCoeff_X


theorem leadingCoeff_one : leadingCoeff (1 : R[X]) = 1 :=
  leadingCoeff_C 1


@[simp]
theorem monic_one : Monic (1 : R[X]) :=
  leadingCoeff_C _


theorem Monic.ne_zero {R : Type*} [Semiring R] [Nontrivial R] {p : R[X]} (hp : p.Monic) :
    p ≠ 0 := by
  /-
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    hp : p.Monic
    ⊢ Ne p 0
  -/
  rintro rfl
  /-
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    hp : Polynomial.Monic 0
    ⊢ False
  -/
  simp [Monic] at hp
  /-
    🎉 no goals
  -/


theorem Monic.ne_zero_of_ne (h : (0 : R) ≠ 1) {p : R[X]} (hp : p.Monic) : p ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    h : Ne 0 1
    p : Polynomial R
    hp : p.Monic
    ⊢ Ne p 0
  -/
  nontriviality R
  /-
    R : Type u
    inst✝¹ : Semiring R
    h : Ne 0 1
    p : Polynomial R
    hp : p.Monic
    inst✝ : Nontrivial R
    ⊢ Ne p 0
  -/
  exact hp.ne_zero
  /-
    🎉 no goals
  -/


theorem Monic.ne_zero_of_polynomial_ne {r} (hp : Monic p) (hne : q ≠ r) : p ≠ 0 :=
  haveI := Nontrivial.of_polynomial_ne hne
  hp.ne_zero


theorem natDegree_mul_le {p q : R[X]} : natDegree (p * q) ≤ natDegree p + natDegree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
  -/
  apply natDegree_le_of_degree_le
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HMul.hMul p q).degree ↑(HAdd.hAdd p.natDegree q.natDegree)
  -/
  apply le_trans (degree_mul_le p q)
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HAdd.hAdd p.degree q.degree) ↑(HAdd.hAdd p.natDegree q.natDegree)
  -/
  rw [Nat.cast_add]
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HAdd.hAdd p.degree q.degree) (HAdd.hAdd ↑p.natDegree ↑q.natDegree)
  -/
                       /-
                         🎉 no goals
                       -/
  apply add_le_add <;> apply degree_le_natDegree
                       /-
                         🎉 no goals
                       -/


theorem natDegree_mul_le_of_le (hp : natDegree p ≤ m) (hg : natDegree q ≤ n) :
    natDegree (p * q) ≤ m + n :=
natDegree_mul_le.trans <| add_le_add ‹_› ‹_›


theorem natDegree_pow_le {p : R[X]} {n : ℕ} : (p ^ n).natDegree ≤ n * p.natDegree := by
  induction n with
  | zero => simp
  | succ i hi =>
    rw [pow_succ, Nat.succ_mul]
    apply le_trans natDegree_mul_le (add_le_add_right hi _)


theorem natDegree_pow_le_of_le (n : ℕ) (hp : natDegree p ≤ m) :
    natDegree (p ^ n) ≤ n * m :=
  natDegree_pow_le.trans (Nat.mul_le_mul le_rfl ‹_›)


theorem natDegree_eq_zero_iff_degree_le_zero : p.natDegree = 0 ↔ p.degree ≤ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.natDegree 0) (LE.le p.degree 0)
  -/
  rw [← nonpos_iff_eq_zero, natDegree_le_iff_degree_le, Nat.cast_zero]
  /-
    🎉 no goals
  -/


theorem degree_zero_le : degree (0 : R[X]) ≤ 0 := natDegree_eq_zero_iff_degree_le_zero.mp rfl


theorem degree_le_iff_coeff_zero (f : R[X]) (n : WithBot ℕ) :
    degree f ≤ n ↔ ∀ m : ℕ, n < m → coeff f m = 0 := by
  -- Porting note: `Nat.cast_withBot` is required.
  simp only [degree, Finset.max, Finset.sup_le_iff, mem_support_iff, Ne, ← not_le,
    not_imp_comm, Nat.cast_withBot]


theorem degree_lt_iff_coeff_zero (f : R[X]) (n : ℕ) :
    degree f < n ↔ ∀ m : ℕ, n ≤ m → coeff f m = 0 := by
  simp only [degree, Finset.sup_lt_iff (WithBot.bot_lt_coe n), mem_support_iff,
    WithBot.coe_lt_coe, ← @not_le ℕ, max_eq_sup_coe, Nat.cast_withBot, Ne, not_imp_not]


theorem natDegree_pos_iff_degree_pos : 0 < natDegree p ↔ 0 < degree p :=
  lt_iff_lt_of_le_iff_le natDegree_le_iff_degree_le


@[simp]
theorem degree_X_pow : degree ((X : R[X]) ^ n) = n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).degree ↑n
  -/
  rw [X_pow_eq_monomial, degree_monomial _ (one_ne_zero' R)]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_X_pow : natDegree ((X : R[X]) ^ n) = n :=
  natDegree_eq_of_degree_eq_some (degree_X_pow n)


theorem degree_sub_le (p q : R[X]) : degree (p - q) ≤ max (degree p) (degree q) := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    ⊢ LE.le (HSub.hSub p q).degree (Max.max p.degree q.degree)
  -/
  simpa only [degree_neg q] using degree_add_le p (-q)
  /-
    🎉 no goals
  -/


theorem degree_sub_le_of_le {a b : WithBot ℕ} (hp : degree p ≤ a) (hq : degree q ≤ b) :
    degree (p - q) ≤ max a b :=
  (p.degree_sub_le q).trans <| max_le_max ‹_› ‹_›


theorem natDegree_sub_le (p q : R[X]) : natDegree (p - q) ≤ max (natDegree p) (natDegree q) := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    ⊢ LE.le (HSub.hSub p q).natDegree (Max.max p.natDegree q.natDegree)
  -/
  simpa only [← natDegree_neg q] using natDegree_add_le p (-q)
  /-
    🎉 no goals
  -/


theorem natDegree_sub_le_of_le (hp : natDegree p ≤ m) (hq : natDegree q ≤ n) :
    natDegree (p - q) ≤ max m n :=
  (p.natDegree_sub_le q).trans <| max_le_max ‹_› ‹_›


theorem degree_sub_lt (hd : degree p = degree q) (hp0 : p ≠ 0)
    (hlc : leadingCoeff p = leadingCoeff q) : degree (p - q) < degree p :=
  have hp : monomial (natDegree p) (leadingCoeff p) + p.erase (natDegree p) = p :=
    monomial_add_erase _ _
  have hq : monomial (natDegree q) (leadingCoeff q) + q.erase (natDegree q) = q :=
    monomial_add_erase _ _
                                             /-
                                               R : Type u
                                               inst✝ : Ring R
                                               p q : Polynomial R
                                               hd : Eq p.degree q.degree
                                               hp0 : Ne p 0
                                               hlc : Eq p.leadingCoeff q.leadingCoeff
                                               hp : Eq (HAdd.hAdd ((Polynomial.monomial p.natDegree) p.leadingCoeff) (Polynom …
                                               hq : Eq (HAdd.hAdd ((Polynomial.monomial q.natDegree) q.leadingCoeff) (Polynom …
                                               ⊢ Eq p.natDegree q.natDegree
                                             -/
  have hd' : natDegree p = natDegree q := by unfold natDegree; rw [hd]
                                                               /-
                                                                 🎉 no goals
                                                               -/
  have hq0 : q ≠ 0 := mt degree_eq_bot.2 (hd ▸ mt degree_eq_bot.1 hp0)
  calc
    degree (p - q) = degree (erase (natDegree q) p + -erase (natDegree q) q) := by
      conv =>
        lhs
        rw [← hp, ← hq, hlc, hd', add_sub_add_left_eq_sub, sub_eq_add_neg]
    _ ≤ max (degree (erase (natDegree q) p)) (degree (erase (natDegree q) q)) :=
      (degree_neg (erase (natDegree q) q) ▸ degree_add_le _ _)
    _ < degree p := max_lt_iff.2 ⟨hd' ▸ degree_erase_lt hp0, hd.symm ▸ degree_erase_lt hq0⟩


theorem degree_X_sub_C_le (r : R) : (X - C r).degree ≤ 1 :=
  (degree_sub_le _ _).trans (max_le degree_X_le (degree_C_le.trans zero_le_one))


theorem natDegree_X_sub_C_le (r : R) : (X - C r).natDegree ≤ 1 :=
  natDegree_le_iff_degree_le.2 <| degree_X_sub_C_le r


