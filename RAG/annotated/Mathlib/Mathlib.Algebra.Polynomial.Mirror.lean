/-- mirror of a polynomial: reverses the coefficients while preserving `Polynomial.natDegree` -/
noncomputable def mirror :=
  p.reverse * X ^ p.natTrailingDegree


@[simp]
                                                  /-
                                                    R : Type u_1
                                                    inst✝ : Semiring R
                                                    ⊢ Eq (Polynomial.mirror 0) 0
                                                  -/
theorem mirror_zero : (0 : R[X]).mirror = 0 := by simp [mirror]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem mirror_monomial (n : ℕ) (a : R) : (monomial n a).mirror = monomial n a := by
  classical
    by_cases ha : a = 0
    · rw [ha, monomial_zero_right, mirror_zero]
    · rw [mirror, reverse, natDegree_monomial n a, if_neg ha, natTrailingDegree_monomial ha, ←
        C_mul_X_pow_eq_monomial, reflect_C_mul_X_pow, revAt_le (le_refl n), tsub_self, pow_zero,
        mul_one]


theorem mirror_C (a : R) : (C a).mirror = C a :=
  mirror_monomial 0 a


theorem mirror_X : X.mirror = (X : R[X]) :=
  mirror_monomial 1 (1 : R)


theorem mirror_natDegree : p.mirror.natDegree = p.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq p.mirror.natDegree p.natDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p 0
      ⊢ Eq p.mirror.natDegree p.natDegree
    -/
  · rw [hp, mirror_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Not (Eq p 0)
    ⊢ Eq p.mirror.natDegree p.natDegree
  -/
  nontriviality R
  rw [mirror, natDegree_mul', reverse_natDegree, natDegree_X_pow,
    tsub_add_cancel_of_le p.natTrailingDegree_le_natDegree]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Not (Eq p 0)
    a✝ : Nontrivial R
    ⊢ Ne (HMul.hMul p.reverse.leadingCoeff (HPow.hPow Polynomial.X p.natTrailingDe …
  -/
  rwa [leadingCoeff_X_pow, mul_one, reverse_leadingCoeff, Ne, trailingCoeff_eq_zero]
  /-
    🎉 no goals
  -/


theorem mirror_natTrailingDegree : p.mirror.natTrailingDegree = p.natTrailingDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq p.mirror.natTrailingDegree p.natTrailingDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p 0
      ⊢ Eq p.mirror.natTrailingDegree p.natTrailingDegree
    -/
  · rw [hp, mirror_zero]
    /-
      🎉 no goals
    -/
  · rw [mirror, natTrailingDegree_mul_X_pow ((mt reverse_eq_zero.mp) hp),
      natTrailingDegree_reverse, zero_add]


theorem coeff_mirror (n : ℕ) :
    p.mirror.coeff n = p.coeff (revAt (p.natDegree + p.natTrailingDegree) n) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (p.mirror.coeff n) (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.n …
  -/
  by_cases h2 : p.natDegree < n
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h2 : LT.lt p.natDegree n
      ⊢ Eq (p.mirror.coeff n) (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.n …
    -/
  · rw [coeff_eq_zero_of_natDegree_lt (by rwa [mirror_natDegree])]
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h2 : LT.lt p.natDegree n
      ⊢ Eq 0 (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDegree) …
    -/
    by_cases h1 : n ≤ p.natDegree + p.natTrailingDegree
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        h2 : LT.lt p.natDegree n
        h1 : LE.le n (HAdd.hAdd p.natDegree p.natTrailingDegree)
        ⊢ Eq 0 (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDegree) …
      -/
    · rw [revAt_le h1, coeff_eq_zero_of_lt_natTrailingDegree]
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        h2 : LT.lt p.natDegree n
        h1 : LE.le n (HAdd.hAdd p.natDegree p.natTrailingDegree)
        ⊢ LT.lt (HSub.hSub (HAdd.hAdd p.natDegree p.natTrailingDegree) n) p.natTrailin …
      -/
      exact (tsub_lt_iff_left h1).mpr (Nat.add_lt_add_right h2 _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        h2 : LT.lt p.natDegree n
        h1 : Not (LE.le n (HAdd.hAdd p.natDegree p.natTrailingDegree))
        ⊢ Eq 0 (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDegree) …
      -/
    · rw [← revAtFun_eq, revAtFun, if_neg h1, coeff_eq_zero_of_natDegree_lt h2]
      /-
        🎉 no goals
      -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : Not (LT.lt p.natDegree n)
    ⊢ Eq (p.mirror.coeff n) (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.n …
  -/
  rw [not_lt] at h2
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : LE.le n p.natDegree
    ⊢ Eq (p.mirror.coeff n) (p.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.n …
  -/
  rw [revAt_le (h2.trans (Nat.le_add_right _ _))]
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : LE.le n p.natDegree
    ⊢ Eq (p.mirror.coeff n) (p.coeff (HSub.hSub (HAdd.hAdd p.natDegree p.natTraili …
  -/
  by_cases h3 : p.natTrailingDegree ≤ n
  · rw [← tsub_add_eq_add_tsub h2, ← tsub_tsub_assoc h2 h3, mirror, coeff_mul_X_pow', if_pos h3,
      coeff_reverse, revAt_le (tsub_le_self.trans h2)]
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : LE.le n p.natDegree
    h3 : Not (LE.le p.natTrailingDegree n)
    ⊢ Eq (p.mirror.coeff n) (p.coeff (HSub.hSub (HAdd.hAdd p.natDegree p.natTraili …
  -/
  rw [not_le] at h3
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : LE.le n p.natDegree
    h3 : LT.lt n p.natTrailingDegree
    ⊢ Eq (p.mirror.coeff n) (p.coeff (HSub.hSub (HAdd.hAdd p.natDegree p.natTraili …
  -/
  rw [coeff_eq_zero_of_natDegree_lt (lt_tsub_iff_right.mpr (Nat.add_lt_add_left h3 _))]
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h2 : LE.le n p.natDegree
    h3 : LT.lt n p.natTrailingDegree
    ⊢ Eq (p.mirror.coeff n) 0
  -/
  exact coeff_eq_zero_of_lt_natTrailingDegree (by rwa [mirror_natTrailingDegree])
  /-
    🎉 no goals
  -/

--TODO: Extract `Finset.sum_range_rev_at` lemma.

theorem mirror_eval_one : p.mirror.eval 1 = p.eval 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (Polynomial.eval 1 p.mirror) (Polynomial.eval 1 p)
  -/
  simp_rw [eval_eq_sum_range, one_pow, mul_one, mirror_natDegree]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun x => p.mirror.coeff x)  …
  -/
  refine Finset.sum_bij_ne_zero ?_ ?_ ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ (a : Nat) → Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) a → Ne ( …
    -/
  · exact fun n _ _ => revAt (p.natDegree + p.natTrailingDegree) n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (a : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) a → Ne  …
    -/
  · intro n hn hp
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
      hp : Ne (p.mirror.coeff n) 0
      ⊢ Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) ((Polynomial.revAt ( …
    -/
    rw [Finset.mem_range_succ_iff] at *
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : LE.le n p.natDegree
      hp : Ne (p.mirror.coeff n) 0
      ⊢ LE.le ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDegree)) n) p.n …
    -/
    rw [revAt_le (hn.trans (Nat.le_add_right _ _))]
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : LE.le n p.natDegree
      hp : Ne (p.mirror.coeff n) 0
      ⊢ LE.le (HSub.hSub (HAdd.hAdd p.natDegree p.natTrailingDegree) n) p.natDegree
    -/
    rw [tsub_le_iff_tsub_le, add_comm, add_tsub_cancel_right, ← mirror_natTrailingDegree]
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : LE.le n p.natDegree
      hp : Ne (p.mirror.coeff n) 0
      ⊢ LE.le p.mirror.natTrailingDegree n
    -/
    exact natTrailingDegree_le_of_ne_zero hp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (a₁ : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) a₁ → N …
    -/
  · exact fun n₁ _ _ _ _ _ h => by rw [← @revAt_invol _ n₁, h, revAt_invol]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) b → Ne  …
    -/
  · intro n hn hp
    /-
      case refine_4
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
      hp : Ne (p.coeff n) 0
      ⊢ Exists fun a => Exists fun h₁ => Exists fun h₂ => Eq ((Polynomial.revAt (HAd …
    -/
    use revAt (p.natDegree + p.natTrailingDegree) n
    /-
      case h
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
      hp : Ne (p.coeff n) 0
      ⊢ Exists fun h₁ => Exists fun h₂ => Eq ((Polynomial.revAt (HAdd.hAdd p.natDegr …
    -/
    refine ⟨?_, ?_, revAt_invol⟩
      /-
        case h.refine_1
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
        hp : Ne (p.coeff n) 0
        ⊢ Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) ((Polynomial.revAt ( …
      -/
    · rw [Finset.mem_range_succ_iff] at *
      /-
        case h.refine_1
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : LE.le n p.natDegree
        hp : Ne (p.coeff n) 0
        ⊢ LE.le ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDegree)) n) p.n …
      -/
      rw [revAt_le (hn.trans (Nat.le_add_right _ _))]
      /-
        case h.refine_1
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : LE.le n p.natDegree
        hp : Ne (p.coeff n) 0
        ⊢ LE.le (HSub.hSub (HAdd.hAdd p.natDegree p.natTrailingDegree) n) p.natDegree
      -/
      rw [tsub_le_iff_tsub_le, add_comm, add_tsub_cancel_right]
      /-
        case h.refine_1
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : LE.le n p.natDegree
        hp : Ne (p.coeff n) 0
        ⊢ LE.le p.natTrailingDegree n
      -/
      exact natTrailingDegree_le_of_ne_zero hp
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
        hp : Ne (p.coeff n) 0
        ⊢ Ne (p.mirror.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDe …
      -/
    · change p.mirror.coeff _ ≠ 0
      /-
        case h.refine_2
        R : Type u_1
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n
        hp : Ne (p.coeff n) 0
        ⊢ Ne (p.mirror.coeff ((Polynomial.revAt (HAdd.hAdd p.natDegree p.natTrailingDe …
      -/
      rwa [coeff_mirror, revAt_invol]
      /-
        🎉 no goals
      -/
    /-
      case refine_5
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ ∀ (a : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) a → Ne  …
    -/
  · exact fun n _ _ => p.coeff_mirror n
    /-
      🎉 no goals
    -/


theorem mirror_mirror : p.mirror.mirror = p :=
  Polynomial.ext fun n => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ⊢ Eq (p.mirror.mirror.coeff n) (p.coeff n)
    -/
    rw [coeff_mirror, coeff_mirror, mirror_natDegree, mirror_natTrailingDegree, revAt_invol]
    /-
      🎉 no goals
    -/


theorem mirror_involutive : Function.Involutive (mirror : R[X] → R[X]) :=
  mirror_mirror


theorem mirror_eq_iff : p.mirror = q ↔ p = q.mirror :=
  mirror_involutive.eq_iff


@[simp]
theorem mirror_inj : p.mirror = q.mirror ↔ p = q :=
  mirror_involutive.injective.eq_iff


@[simp]
theorem mirror_eq_zero : p.mirror = 0 ↔ p = 0 :=
               /-
                 R : Type u_1
                 inst✝ : Semiring R
                 p : Polynomial R
                 h : Eq p.mirror 0
                 ⊢ Eq p 0
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [← p.mirror_mirror, h, mirror_zero], fun h => by rw [h, mirror_zero]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem mirror_trailingCoeff : p.mirror.trailingCoeff = p.leadingCoeff := by
  rw [leadingCoeff, trailingCoeff, mirror_natTrailingDegree, coeff_mirror,
    revAt_le (Nat.le_add_left _ _), add_tsub_cancel_right]


@[simp]
theorem mirror_leadingCoeff : p.mirror.leadingCoeff = p.trailingCoeff := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq p.mirror.leadingCoeff p.trailingCoeff
  -/
  rw [← p.mirror_mirror, mirror_trailingCoeff, p.mirror_mirror]
  /-
    🎉 no goals
  -/


theorem coeff_mul_mirror :
    (p * p.mirror).coeff (p.natDegree + p.natTrailingDegree) = p.sum fun _ => (· ^ 2) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((HMul.hMul p p.mirror).coeff (HAdd.hAdd p.natDegree p.natTrailingDegree) …
  -/
  rw [coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
  refine
    (Finset.sum_congr rfl fun n hn => ?_).trans
      (p.sum_eq_of_subset (fun _ ↦ (· ^ 2)) (fun _ ↦ zero_pow two_ne_zero) fun n hn ↦
          Finset.mem_range_succ_iff.mpr
            ((le_natDegree_of_mem_supp n hn).trans (Nat.le_add_right _ _))).symm
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : Membership.mem (Finset.range (HAdd.hAdd p.natDegree p.natTrailingDegree). …
    ⊢ Eq (HMul.hMul (p.coeff { fst := n, snd := HSub.hSub (HAdd.hAdd p.natDegree p …
  -/
  rw [coeff_mirror, ← revAt_le (Finset.mem_range_succ_iff.mp hn), revAt_invol, ← sq]
  /-
    🎉 no goals
  -/


theorem natDegree_mul_mirror : (p * p.mirror).natDegree = 2 * p.natDegree := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : NoZeroDivisors R
    ⊢ Eq (HMul.hMul p p.mirror).natDegree (HMul.hMul 2 p.natDegree)
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : NoZeroDivisors R
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p p.mirror).natDegree (HMul.hMul 2 p.natDegree)
    -/
  · rw [hp, zero_mul, natDegree_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p p.mirror).natDegree (HMul.hMul 2 p.natDegree)
  -/
  rw [natDegree_mul hp (mt mirror_eq_zero.mp hp), mirror_natDegree, two_mul]
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_mul_mirror :
    (p * p.mirror).natTrailingDegree = 2 * p.natTrailingDegree := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : NoZeroDivisors R
    ⊢ Eq (HMul.hMul p p.mirror).natTrailingDegree (HMul.hMul 2 p.natTrailingDegree)
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : NoZeroDivisors R
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p p.mirror).natTrailingDegree (HMul.hMul 2 p.natTrailingDegree)
    -/
  · rw [hp, zero_mul, natTrailingDegree_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p p.mirror).natTrailingDegree (HMul.hMul 2 p.natTrailingDegree)
  -/
  rw [natTrailingDegree_mul hp (mt mirror_eq_zero.mp hp), mirror_natTrailingDegree, two_mul]
  /-
    🎉 no goals
  -/


theorem mirror_neg : (-p).mirror = -p.mirror := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).mirror (Neg.neg p.mirror)
  -/
  rw [mirror, mirror, reverse_neg, natTrailingDegree_neg, neg_mul_eq_neg_mul]
  /-
    🎉 no goals
  -/


theorem mirror_mul_of_domain : (p * q).mirror = p.mirror * q.mirror := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    ⊢ Eq (HMul.hMul p q).mirror (HMul.hMul p.mirror q.mirror)
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : Ring R
      p q : Polynomial R
      inst✝ : NoZeroDivisors R
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p q).mirror (HMul.hMul p.mirror q.mirror)
    -/
  · rw [hp, zero_mul, mirror_zero, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p q).mirror (HMul.hMul p.mirror q.mirror)
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : Ring R
      p q : Polynomial R
      inst✝ : NoZeroDivisors R
      hp : Not (Eq p 0)
      hq : Eq q 0
      ⊢ Eq (HMul.hMul p q).mirror (HMul.hMul p.mirror q.mirror)
    -/
  · rw [hq, mul_zero, mirror_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    hq : Not (Eq q 0)
    ⊢ Eq (HMul.hMul p q).mirror (HMul.hMul p.mirror q.mirror)
  -/
  rw [mirror, mirror, mirror, reverse_mul_of_domain, natTrailingDegree_mul hp hq, pow_add]
  /-
    case neg
    R : Type u_1
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    hq : Not (Eq q 0)
    ⊢ Eq (HMul.hMul (HMul.hMul p.reverse q.reverse) (HMul.hMul (HPow.hPow Polynomi …
  -/
  rw [mul_assoc, ← mul_assoc q.reverse, ← X_pow_mul (p := reverse q)]
  /-
    case neg
    R : Type u_1
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    hq : Not (Eq q 0)
    ⊢ Eq (HMul.hMul p.reverse (HMul.hMul (HMul.hMul (HPow.hPow Polynomial.X p.natT …
  -/
  repeat' rw [mul_assoc]
  /-
    🎉 no goals
  -/


theorem mirror_smul (a : R) : (a • p).mirror = a • p.mirror := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    p : Polynomial R
    inst✝ : NoZeroDivisors R
    a : R
    ⊢ Eq (HSMul.hSMul a p).mirror (HSMul.hSMul a p.mirror)
  -/
  rw [← C_mul', ← C_mul', mirror_mul_of_domain, mirror_C]
  /-
    🎉 no goals
  -/


theorem irreducible_of_mirror (h1 : ¬IsUnit f)
    (h2 : ∀ k, f * f.mirror = k * k.mirror → k = f ∨ k = -f ∨ k = f.mirror ∨ k = -f.mirror)
    (h3 : IsRelPrime f f.mirror) : Irreducible f := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    f : Polynomial R
    h1 : Not (IsUnit f)
    h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
    h3 : IsRelPrime f f.mirror
    ⊢ Irreducible f
  -/
  constructor
    /-
      case not_unit
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      ⊢ Not (IsUnit f)
    -/
  · exact h1
    /-
      🎉 no goals
    -/
    /-
      case isUnit_or_isUnit'
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      ⊢ ∀ (a b : Polynomial R), Eq f (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
    -/
  · intro g h fgh
    /-
      case isUnit_or_isUnit'
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      g h : Polynomial R
      fgh : Eq f (HMul.hMul g h)
      ⊢ Or (IsUnit g) (IsUnit h)
    -/
    let k := g * h.mirror
    have key : f * f.mirror = k * k.mirror := by
      rw [fgh, mirror_mul_of_domain, mirror_mul_of_domain, mirror_mirror, mul_assoc, mul_comm h,
        mul_comm g.mirror, mul_assoc, ← mul_assoc]
    have g_dvd_f : g ∣ f := by
      rw [fgh]
      exact dvd_mul_right g h
    have h_dvd_f : h ∣ f := by
      rw [fgh]
      exact dvd_mul_left h g
    /-
      case isUnit_or_isUnit'
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      g h : Polynomial R
      fgh : Eq f (HMul.hMul g h)
      k : Polynomial R := HMul.hMul g h.mirror
      key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
      g_dvd_f : Dvd.dvd g f
      h_dvd_f : Dvd.dvd h f
      ⊢ Or (IsUnit g) (IsUnit h)
    -/
    have g_dvd_k : g ∣ k := dvd_mul_right g h.mirror
    have h_dvd_k_rev : h ∣ k.mirror := by
      rw [mirror_mul_of_domain, mirror_mirror]
      exact dvd_mul_left h g.mirror
    /-
      case isUnit_or_isUnit'
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      g h : Polynomial R
      fgh : Eq f (HMul.hMul g h)
      k : Polynomial R := HMul.hMul g h.mirror
      key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
      g_dvd_f : Dvd.dvd g f
      h_dvd_f : Dvd.dvd h f
      g_dvd_k : Dvd.dvd g k
      h_dvd_k_rev : Dvd.dvd h k.mirror
      ⊢ Or (IsUnit g) (IsUnit h)
    -/
    have hk := h2 k key
    /-
      case isUnit_or_isUnit'
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      f : Polynomial R
      h1 : Not (IsUnit f)
      h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
      h3 : IsRelPrime f f.mirror
      g h : Polynomial R
      fgh : Eq f (HMul.hMul g h)
      k : Polynomial R := HMul.hMul g h.mirror
      key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
      g_dvd_f : Dvd.dvd g f
      h_dvd_f : Dvd.dvd h f
      g_dvd_k : Dvd.dvd g k
      h_dvd_k_rev : Dvd.dvd h k.mirror
      hk : Or (Eq k f) (Or (Eq k (Neg.neg f)) (Or (Eq k f.mirror) (Eq k (Neg.neg f.m …
      ⊢ Or (IsUnit g) (IsUnit h)
    -/
    rcases hk with (hk | hk | hk | hk)
      /-
        case isUnit_or_isUnit'.inl
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        h1 : Not (IsUnit f)
        h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
        h3 : IsRelPrime f f.mirror
        g h : Polynomial R
        fgh : Eq f (HMul.hMul g h)
        k : Polynomial R := HMul.hMul g h.mirror
        key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
        g_dvd_f : Dvd.dvd g f
        h_dvd_f : Dvd.dvd h f
        g_dvd_k : Dvd.dvd g k
        h_dvd_k_rev : Dvd.dvd h k.mirror
        hk : Eq k f
        ⊢ Or (IsUnit g) (IsUnit h)
      -/
    · exact Or.inr (h3 h_dvd_f (by rwa [← hk]))
      /-
        🎉 no goals
      -/
      /-
        case isUnit_or_isUnit'.inr.inl
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        h1 : Not (IsUnit f)
        h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
        h3 : IsRelPrime f f.mirror
        g h : Polynomial R
        fgh : Eq f (HMul.hMul g h)
        k : Polynomial R := HMul.hMul g h.mirror
        key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
        g_dvd_f : Dvd.dvd g f
        h_dvd_f : Dvd.dvd h f
        g_dvd_k : Dvd.dvd g k
        h_dvd_k_rev : Dvd.dvd h k.mirror
        hk : Eq k (Neg.neg f)
        ⊢ Or (IsUnit g) (IsUnit h)
      -/
    · exact Or.inr (h3 h_dvd_f (by rwa [← neg_eq_iff_eq_neg.mpr hk, mirror_neg, dvd_neg]))
      /-
        🎉 no goals
      -/
      /-
        case isUnit_or_isUnit'.inr.inr.inl
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        h1 : Not (IsUnit f)
        h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
        h3 : IsRelPrime f f.mirror
        g h : Polynomial R
        fgh : Eq f (HMul.hMul g h)
        k : Polynomial R := HMul.hMul g h.mirror
        key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
        g_dvd_f : Dvd.dvd g f
        h_dvd_f : Dvd.dvd h f
        g_dvd_k : Dvd.dvd g k
        h_dvd_k_rev : Dvd.dvd h k.mirror
        hk : Eq k f.mirror
        ⊢ Or (IsUnit g) (IsUnit h)
      -/
    · exact Or.inl (h3 g_dvd_f (by rwa [← hk]))
      /-
        🎉 no goals
      -/
      /-
        case isUnit_or_isUnit'.inr.inr.inr
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : NoZeroDivisors R
        f : Polynomial R
        h1 : Not (IsUnit f)
        h2 : ∀ (k : Polynomial R), Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror) →  …
        h3 : IsRelPrime f f.mirror
        g h : Polynomial R
        fgh : Eq f (HMul.hMul g h)
        k : Polynomial R := HMul.hMul g h.mirror
        key : Eq (HMul.hMul f f.mirror) (HMul.hMul k k.mirror)
        g_dvd_f : Dvd.dvd g f
        h_dvd_f : Dvd.dvd h f
        g_dvd_k : Dvd.dvd g k
        h_dvd_k_rev : Dvd.dvd h k.mirror
        hk : Eq k (Neg.neg f.mirror)
        ⊢ Or (IsUnit g) (IsUnit h)
      -/
    · exact Or.inl (h3 g_dvd_f (by rwa [← neg_eq_iff_eq_neg.mpr hk, dvd_neg]))
      /-
        🎉 no goals
      -/


