/-- Divide by `monomial 1 s`, discarding terms not divisible by this. -/
noncomputable def divMonomial (p : MvPolynomial σ R) (s : σ →₀ ℕ) : MvPolynomial σ R :=
  AddMonoidAlgebra.divOf p s


local infixl:70 " /ᵐᵒⁿᵒᵐⁱᵃˡ " => divMonomial


@[simp]
theorem coeff_divMonomial (s : σ →₀ ℕ) (x : MvPolynomial σ R) (s' : σ →₀ ℕ) :
    coeff s' (x /ᵐᵒⁿᵒᵐⁱᵃˡ s) = coeff (s + s') x :=
  rfl


@[simp]
theorem support_divMonomial (s : σ →₀ ℕ) (x : MvPolynomial σ R) :
    (x /ᵐᵒⁿᵒᵐⁱᵃˡ s).support = x.support.preimage _ (add_right_injective s).injOn :=
  rfl


@[simp]
theorem zero_divMonomial (s : σ →₀ ℕ) : (0 : MvPolynomial σ R) /ᵐᵒⁿᵒᵐⁱᵃˡ s = 0 :=
  AddMonoidAlgebra.zero_divOf _


theorem divMonomial_zero (x : MvPolynomial σ R) : x /ᵐᵒⁿᵒᵐⁱᵃˡ 0 = x :=
  x.divOf_zero


theorem add_divMonomial (x y : MvPolynomial σ R) (s : σ →₀ ℕ) :
    (x + y) /ᵐᵒⁿᵒᵐⁱᵃˡ s = x /ᵐᵒⁿᵒᵐⁱᵃˡ s + y /ᵐᵒⁿᵒᵐⁱᵃˡ s :=
  map_add (N := _ →₀ _) _ _ _


theorem divMonomial_add (a b : σ →₀ ℕ) (x : MvPolynomial σ R) :
    x /ᵐᵒⁿᵒᵐⁱᵃˡ (a + b) = x /ᵐᵒⁿᵒᵐⁱᵃˡ a /ᵐᵒⁿᵒᵐⁱᵃˡ b :=
  x.divOf_add _ _


@[simp]
theorem divMonomial_monomial_mul (a : σ →₀ ℕ) (x : MvPolynomial σ R) :
    monomial a 1 * x /ᵐᵒⁿᵒᵐⁱᵃˡ a = x :=
  x.of'_mul_divOf _


@[simp]
theorem divMonomial_mul_monomial (a : σ →₀ ℕ) (x : MvPolynomial σ R) :
    x * monomial a 1 /ᵐᵒⁿᵒᵐⁱᵃˡ a = x :=
  x.mul_of'_divOf _


@[simp]
theorem divMonomial_monomial (a : σ →₀ ℕ) : monomial a 1 /ᵐᵒⁿᵒᵐⁱᵃˡ a = (1 : MvPolynomial σ R) :=
  AddMonoidAlgebra.of'_divOf _


/-- The remainder upon division by `monomial 1 s`. -/
noncomputable def modMonomial (x : MvPolynomial σ R) (s : σ →₀ ℕ) : MvPolynomial σ R :=
  x.modOf s


local infixl:70 " %ᵐᵒⁿᵒᵐⁱᵃˡ " => modMonomial


@[simp]
theorem coeff_modMonomial_of_not_le {s' s : σ →₀ ℕ} (x : MvPolynomial σ R) (h : ¬s ≤ s') :
    coeff s' (x %ᵐᵒⁿᵒᵐⁱᵃˡ s) = coeff s' x :=
  x.modOf_apply_of_not_exists_add s s'
    (by
      /-
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        s' s : Finsupp σ Nat
        x : MvPolynomial σ R
        h : Not (LE.le s s')
        ⊢ Not (Exists fun d => Eq s' (HAdd.hAdd s d))
      -/
      rintro ⟨d, rfl⟩
      /-
        case intro
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        s : Finsupp σ Nat
        x : MvPolynomial σ R
        d : Finsupp σ Nat
        h : Not (LE.le s (HAdd.hAdd s d))
        ⊢ False
      -/
      exact h le_self_add)
      /-
        🎉 no goals
      -/


@[simp]
theorem coeff_modMonomial_of_le {s' s : σ →₀ ℕ} (x : MvPolynomial σ R) (h : s ≤ s') :
    coeff s' (x %ᵐᵒⁿᵒᵐⁱᵃˡ s) = 0 :=
  x.modOf_apply_of_exists_add _ _ <| exists_add_of_le h


@[simp]
theorem monomial_mul_modMonomial (s : σ →₀ ℕ) (x : MvPolynomial σ R) :
    monomial s 1 * x %ᵐᵒⁿᵒᵐⁱᵃˡ s = 0 :=
  x.of'_mul_modOf _


@[simp]
theorem mul_monomial_modMonomial (s : σ →₀ ℕ) (x : MvPolynomial σ R) :
    x * monomial s 1 %ᵐᵒⁿᵒᵐⁱᵃˡ s = 0 :=
  x.mul_of'_modOf _


@[simp]
theorem monomial_modMonomial (s : σ →₀ ℕ) : monomial s (1 : R) %ᵐᵒⁿᵒᵐⁱᵃˡ s = 0 :=
  AddMonoidAlgebra.of'_modOf _


theorem divMonomial_add_modMonomial (x : MvPolynomial σ R) (s : σ →₀ ℕ) :
    monomial s 1 * (x /ᵐᵒⁿᵒᵐⁱᵃˡ s) + x %ᵐᵒⁿᵒᵐⁱᵃˡ s = x :=
  AddMonoidAlgebra.divOf_add_modOf x s


theorem modMonomial_add_divMonomial (x : MvPolynomial σ R) (s : σ →₀ ℕ) :
    x %ᵐᵒⁿᵒᵐⁱᵃˡ s + monomial s 1 * (x /ᵐᵒⁿᵒᵐⁱᵃˡ s) = x :=
  AddMonoidAlgebra.modOf_add_divOf x s


theorem monomial_one_dvd_iff_modMonomial_eq_zero {i : σ →₀ ℕ} {x : MvPolynomial σ R} :
    monomial i (1 : R) ∣ x ↔ x %ᵐᵒⁿᵒᵐⁱᵃˡ i = 0 :=
  AddMonoidAlgebra.of'_dvd_iff_modOf_eq_zero


@[simp]
theorem X_mul_divMonomial (i : σ) (x : MvPolynomial σ R) :
    X i * x /ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = x :=
  divMonomial_monomial_mul _ _


@[simp]
theorem X_divMonomial (i : σ) : (X i : MvPolynomial σ R) /ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = 1 :=
  divMonomial_monomial (Finsupp.single i 1)


@[simp]
theorem mul_X_divMonomial (x : MvPolynomial σ R) (i : σ) :
    x * X i /ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = x :=
  divMonomial_mul_monomial _ _


@[simp]
theorem X_mul_modMonomial (i : σ) (x : MvPolynomial σ R) :
    X i * x %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = 0 :=
  monomial_mul_modMonomial _ _


@[simp]
theorem mul_X_modMonomial (x : MvPolynomial σ R) (i : σ) :
    x * X i %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = 0 :=
  mul_monomial_modMonomial _ _


@[simp]
theorem modMonomial_X (i : σ) : (X i : MvPolynomial σ R) %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = 0 :=
  monomial_modMonomial _


theorem divMonomial_add_modMonomial_single (x : MvPolynomial σ R) (i : σ) :
    X i * (x /ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1) + x %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = x :=
  divMonomial_add_modMonomial _ _


theorem modMonomial_add_divMonomial_single (x : MvPolynomial σ R) (i : σ) :
    x %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 + X i * (x /ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1) = x :=
  modMonomial_add_divMonomial _ _


theorem X_dvd_iff_modMonomial_eq_zero {i : σ} {x : MvPolynomial σ R} :
    X i ∣ x ↔ x %ᵐᵒⁿᵒᵐⁱᵃˡ Finsupp.single i 1 = 0 :=
  monomial_one_dvd_iff_modMonomial_eq_zero


theorem monomial_dvd_monomial {r s : R} {i j : σ →₀ ℕ} :
    monomial i r ∣ monomial j s ↔ (s = 0 ∨ i ≤ j) ∧ r ∣ s := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    r s : R
    i j : Finsupp σ Nat
    ⊢ Iff (Dvd.dvd ((MvPolynomial.monomial i) r) ((MvPolynomial.monomial j) s)) (A …
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      r s : R
      i j : Finsupp σ Nat
      ⊢ Dvd.dvd ((MvPolynomial.monomial i) r) ((MvPolynomial.monomial j) s) → And (O …
    -/
  · rintro ⟨x, hx⟩
    /-
      case mp.intro
      σ : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      r s : R
      i j : Finsupp σ Nat
      x : MvPolynomial σ R
      hx : Eq ((MvPolynomial.monomial j) s) (HMul.hMul ((MvPolynomial.monomial i) r) …
      ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s)
    -/
    rw [MvPolynomial.ext_iff] at hx
    /-
      case mp.intro
      σ : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      r s : R
      i j : Finsupp σ Nat
      x : MvPolynomial σ R
      hx : ∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.monomial j …
      ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s)
    -/
    have hj := hx j
    /-
      case mp.intro
      σ : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      r s : R
      i j : Finsupp σ Nat
      x : MvPolynomial σ R
      hx : ∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.monomial j …
      hj : Eq (MvPolynomial.coeff j ((MvPolynomial.monomial j) s)) (MvPolynomial.coe …
      ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s)
    -/
    have hi := hx i
    classical
      simp_rw [coeff_monomial, if_pos] at hj hi
      simp_rw [coeff_monomial_mul'] at hi hj
      split_ifs at hi hj with hi hi
      · exact ⟨Or.inr hi, _, hj⟩
      · exact ⟨Or.inl hj, hj.symm ▸ dvd_zero _⟩
    -- Porting note: two goals remain at this point in Lean 4
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        r s : R
        i j : Finsupp σ Nat
        x : MvPolynomial σ R
        hx : ∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.monomial j …
        hi✝ : Eq (ite (Eq j i) s 0) (ite (LE.le i i) (HMul.hMul r (MvPolynomial.coeff  …
        hi : Not (Eq j i)
        h✝ : LE.le i j
        hj : Eq s (HMul.hMul r (MvPolynomial.coeff (HSub.hSub j i) x))
        ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s)
      -/
    · simp_all only [or_true, dvd_mul_right, and_self]
      /-
        🎉 no goals
      -/
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        r s : R
        i j : Finsupp σ Nat
        x : MvPolynomial σ R
        hx : ∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.monomial j …
        hi✝ : Eq (ite (Eq j i) s 0) (ite (LE.le i i) (HMul.hMul r (MvPolynomial.coeff  …
        hi : Not (Eq j i)
        h✝ : Not (LE.le i j)
        hj : Eq s 0
        ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s)
      -/
    · simp_all only [ite_self, le_refl, ite_true, dvd_mul_right, or_false, and_self]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      r s : R
      i j : Finsupp σ Nat
      ⊢ And (Or (Eq s 0) (LE.le i j)) (Dvd.dvd r s) → Dvd.dvd ((MvPolynomial.monomia …
    -/
  · rintro ⟨h | hij, d, rfl⟩
      /-
        case mpr.intro.inl.intro
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        r : R
        i j : Finsupp σ Nat
        d : R
        h : Eq (HMul.hMul r d) 0
        ⊢ Dvd.dvd ((MvPolynomial.monomial i) r) ((MvPolynomial.monomial j) (HMul.hMul  …
      -/
    · simp_rw [h, monomial_zero, dvd_zero]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.inr.intro
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        r : R
        i j : Finsupp σ Nat
        hij : LE.le i j
        d : R
        ⊢ Dvd.dvd ((MvPolynomial.monomial i) r) ((MvPolynomial.monomial j) (HMul.hMul  …
      -/
    · refine ⟨monomial (j - i) d, ?_⟩
      /-
        case mpr.intro.inr.intro
        σ : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        r : R
        i j : Finsupp σ Nat
        hij : LE.le i j
        d : R
        ⊢ Eq ((MvPolynomial.monomial j) (HMul.hMul r d)) (HMul.hMul ((MvPolynomial.mon …
      -/
      rw [monomial_mul, add_tsub_cancel_of_le hij]
      /-
        🎉 no goals
      -/


@[simp]
theorem monomial_one_dvd_monomial_one [Nontrivial R] {i j : σ →₀ ℕ} :
    monomial i (1 : R) ∣ monomial j 1 ↔ i ≤ j := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i j : Finsupp σ Nat
    ⊢ Iff (Dvd.dvd ((MvPolynomial.monomial i) 1) ((MvPolynomial.monomial j) 1)) (L …
  -/
  rw [monomial_dvd_monomial]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i j : Finsupp σ Nat
    ⊢ Iff (And (Or (Eq 1 0) (LE.le i j)) (Dvd.dvd 1 1)) (LE.le i j)
  -/
  simp_rw [one_ne_zero, false_or, dvd_rfl, and_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem X_dvd_X [Nontrivial R] {i j : σ} :
    (X i : MvPolynomial σ R) ∣ (X j : MvPolynomial σ R) ↔ i = j := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i j : σ
    ⊢ Iff (Dvd.dvd (MvPolynomial.X i) (MvPolynomial.X j)) (Eq i j)
  -/
  refine monomial_one_dvd_monomial_one.trans ?_
  simp_rw [Finsupp.single_le_iff, Nat.one_le_iff_ne_zero, Finsupp.single_apply_ne_zero,
    ne_eq, reduceCtorEq,not_false_eq_true, and_true]


@[simp]
theorem X_dvd_monomial {i : σ} {j : σ →₀ ℕ} {r : R} :
    (X i : MvPolynomial σ R) ∣ monomial j r ↔ r = 0 ∨ j i ≠ 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    i : σ
    j : Finsupp σ Nat
    r : R
    ⊢ Iff (Dvd.dvd (MvPolynomial.X i) ((MvPolynomial.monomial j) r)) (Or (Eq r 0)  …
  -/
  refine monomial_dvd_monomial.trans ?_
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    i : σ
    j : Finsupp σ Nat
    r : R
    ⊢ Iff (And (Or (Eq r 0) (LE.le (Finsupp.single i 1) j)) (Dvd.dvd 1 r)) (Or (Eq …
  -/
  simp_rw [one_dvd, and_true, Finsupp.single_le_iff, Nat.one_le_iff_ne_zero]
  /-
    🎉 no goals
  -/


