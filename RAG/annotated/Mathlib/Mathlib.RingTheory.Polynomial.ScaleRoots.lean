/-- `scaleRoots p s` is a polynomial with root `r * s` for each root `r` of `p`. -/
noncomputable def scaleRoots (p : R[X]) (s : R) : R[X] :=
  ∑ i ∈ p.support, monomial i (p.coeff i * s ^ (p.natDegree - i))


@[simp]
theorem coeff_scaleRoots (p : R[X]) (s : R) (i : ℕ) :
    (scaleRoots p s).coeff i = coeff p i * s ^ (p.natDegree - i) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    i : Nat
    ⊢ Eq ((p.scaleRoots s).coeff i) (HMul.hMul (p.coeff i) (HPow.hPow s (HSub.hSub …
  -/
  simp +contextual [scaleRoots, coeff_monomial]
  /-
    🎉 no goals
  -/


theorem coeff_scaleRoots_natDegree (p : R[X]) (s : R) :
    (scaleRoots p s).coeff p.natDegree = p.leadingCoeff := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    ⊢ Eq ((p.scaleRoots s).coeff p.natDegree) p.leadingCoeff
  -/
  rw [leadingCoeff, coeff_scaleRoots, tsub_self, pow_zero, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_scaleRoots (s : R) : scaleRoots 0 s = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    s : R
    ⊢ Eq (Polynomial.scaleRoots 0 s) 0
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    s : R
    n✝ : Nat
    ⊢ Eq ((Polynomial.scaleRoots 0 s).coeff n✝) (Polynomial.coeff 0 n✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem scaleRoots_ne_zero {p : R[X]} (hp : p ≠ 0) (s : R) : scaleRoots p s ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    s : R
    ⊢ Ne (p.scaleRoots s) 0
  -/
  intro h
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    s : R
    h : Eq (p.scaleRoots s) 0
    ⊢ False
  -/
  have : p.coeff p.natDegree ≠ 0 := mt leadingCoeff_eq_zero.mp hp
  have : (scaleRoots p s).coeff p.natDegree = 0 :=
    congr_fun (congr_arg (coeff : R[X] → ℕ → R) h) p.natDegree
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    s : R
    h : Eq (p.scaleRoots s) 0
    this✝ : Ne (p.coeff p.natDegree) 0
    this : Eq ((p.scaleRoots s).coeff p.natDegree) 0
    ⊢ False
  -/
  rw [coeff_scaleRoots_natDegree] at this
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    s : R
    h : Eq (p.scaleRoots s) 0
    this✝ : Ne (p.coeff p.natDegree) 0
    this : Eq p.leadingCoeff 0
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


theorem support_scaleRoots_le (p : R[X]) (s : R) : (scaleRoots p s).support ≤ p.support := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    ⊢ LE.le (p.scaleRoots s).support p.support
  -/
  intro
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    a✝ : Nat
    ⊢ Membership.mem (p.scaleRoots s).support a✝ → Membership.mem p.support a✝
  -/
  simpa using left_ne_zero_of_mul
  /-
    🎉 no goals
  -/


theorem support_scaleRoots_eq (p : R[X]) {s : R} (hs : s ∈ nonZeroDivisors R) :
    (scaleRoots p s).support = p.support :=
  le_antisymm (support_scaleRoots_le p s)
        /-
          R : Type u_1
          inst✝ : Semiring R
          p : Polynomial R
          s : R
          hs : Membership.mem (nonZeroDivisors R) s
          ⊢ LE.le p.support (p.scaleRoots s).support
        -/
    (by intro i
        /-
          R : Type u_1
          inst✝ : Semiring R
          p : Polynomial R
          s : R
          hs : Membership.mem (nonZeroDivisors R) s
          i : Nat
          ⊢ Membership.mem p.support i → Membership.mem (p.scaleRoots s).support i
        -/
        simp only [coeff_scaleRoots, Polynomial.mem_support_iff]
        /-
          R : Type u_1
          inst✝ : Semiring R
          p : Polynomial R
          s : R
          hs : Membership.mem (nonZeroDivisors R) s
          i : Nat
          ⊢ Ne (p.coeff i) 0 → Ne (HMul.hMul (p.coeff i) (HPow.hPow s (HSub.hSub p.natDe …
        -/
        intro p_ne_zero ps_zero
        /-
          R : Type u_1
          inst✝ : Semiring R
          p : Polynomial R
          s : R
          hs : Membership.mem (nonZeroDivisors R) s
          i : Nat
          p_ne_zero : Ne (p.coeff i) 0
          ps_zero : Eq (HMul.hMul (p.coeff i) (HPow.hPow s (HSub.hSub p.natDegree i))) 0
          ⊢ False
        -/
        have := pow_mem hs (p.natDegree - i) _ ps_zero
        /-
          R : Type u_1
          inst✝ : Semiring R
          p : Polynomial R
          s : R
          hs : Membership.mem (nonZeroDivisors R) s
          i : Nat
          p_ne_zero : Ne (p.coeff i) 0
          ps_zero : Eq (HMul.hMul (p.coeff i) (HPow.hPow s (HSub.hSub p.natDegree i))) 0
          this : Eq (p.coeff i) 0
          ⊢ False
        -/
        contradiction)
        /-
          🎉 no goals
        -/


@[simp]
theorem degree_scaleRoots (p : R[X]) {s : R} : degree (scaleRoots p s) = degree p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    ⊢ Eq (p.scaleRoots s).degree p.degree
  -/
  haveI := Classical.propDecidable
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this : (a : Prop) → Decidable a
    ⊢ Eq (p.scaleRoots s).degree p.degree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      s : R
      this : (a : Prop) → Decidable a
      hp : Eq p 0
      ⊢ Eq (p.scaleRoots s).degree p.degree
    -/
  · rw [hp, zero_scaleRoots]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this : (a : Prop) → Decidable a
    hp : Not (Eq p 0)
    ⊢ Eq (p.scaleRoots s).degree p.degree
  -/
  refine le_antisymm (Finset.sup_mono (support_scaleRoots_le p s)) (degree_le_degree ?_)
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this : (a : Prop) → Decidable a
    hp : Not (Eq p 0)
    ⊢ Ne ((p.scaleRoots s).coeff p.natDegree) 0
  -/
  rw [coeff_scaleRoots_natDegree]
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this : (a : Prop) → Decidable a
    hp : Not (Eq p 0)
    ⊢ Ne p.leadingCoeff 0
  -/
  intro h
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this : (a : Prop) → Decidable a
    hp : Not (Eq p 0)
    h : Eq p.leadingCoeff 0
    ⊢ False
  -/
  have := leadingCoeff_eq_zero.mp h
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    this✝ : (a : Prop) → Decidable a
    hp : Not (Eq p 0)
    h : Eq p.leadingCoeff 0
    this : Eq p 0
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_scaleRoots (p : R[X]) (s : R) : natDegree (scaleRoots p s) = natDegree p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    ⊢ Eq (p.scaleRoots s).natDegree p.natDegree
  -/
  simp only [natDegree, degree_scaleRoots]
  /-
    🎉 no goals
  -/


theorem monic_scaleRoots_iff {p : R[X]} (s : R) : Monic (scaleRoots p s) ↔ Monic p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    s : R
    ⊢ Iff (p.scaleRoots s).Monic p.Monic
  -/
  simp only [Monic, leadingCoeff, natDegree_scaleRoots, coeff_scaleRoots_natDegree]
  /-
    🎉 no goals
  -/


theorem map_scaleRoots (p : R[X]) (x : R) (f : R →+* S) (h : f p.leadingCoeff ≠ 0) :
    (p.scaleRoots x).map f = (p.map f).scaleRoots (f x) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    x : R
    f : RingHom R S
    h : Ne (f p.leadingCoeff) 0
    ⊢ Eq (Polynomial.map f (p.scaleRoots x)) ((Polynomial.map f p).scaleRoots (f x))
  -/
  ext
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    x : R
    f : RingHom R S
    h : Ne (f p.leadingCoeff) 0
    n✝ : Nat
    ⊢ Eq ((Polynomial.map f (p.scaleRoots x)).coeff n✝) (((Polynomial.map f p).sca …
  -/
  simp [Polynomial.natDegree_map_of_leadingCoeff_ne_zero _ h]
  /-
    🎉 no goals
  -/


@[simp]
lemma scaleRoots_C (r c : R) : (C c).scaleRoots r = C c := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r c : R
    ⊢ Eq ((Polynomial.C c).scaleRoots r) (Polynomial.C c)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma scaleRoots_one (p : R[X]) :
                             /-
                               R : Type u_1
                               inst✝ : Semiring R
                               p : Polynomial R
                               ⊢ Eq (p.scaleRoots 1) p
                             -/
    p.scaleRoots 1 = p := by ext; simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma scaleRoots_zero (p : R[X]) :
    p.scaleRoots 0 = p.leadingCoeff • X ^ p.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (p.scaleRoots 0) (HSMul.hSMul p.leadingCoeff (HPow.hPow Polynomial.X p.na …
  -/
  ext n
  simp only [coeff_scaleRoots, ne_eq, tsub_eq_zero_iff_le, not_le, zero_pow_eq, mul_ite,
    mul_one, mul_zero, coeff_smul, coeff_X_pow, smul_eq_mul]
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (ite (LE.le p.natDegree n) (p.coeff n) 0) (ite (Eq n p.natDegree) p.leadi …
  -/
  split_ifs with h₁ h₂ h₂
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h₁ : LE.le p.natDegree n
      h₂ : Eq n p.natDegree
      ⊢ Eq (p.coeff n) p.leadingCoeff
    -/
  · subst h₂; rfl
              /-
                🎉 no goals
              -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h₁ : LE.le p.natDegree n
      h₂ : Not (Eq n p.natDegree)
      ⊢ Eq (p.coeff n) 0
    -/
  · exact coeff_eq_zero_of_natDegree_lt (lt_of_le_of_ne h₁ (Ne.symm h₂))
    /-
      🎉 no goals
    -/
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h₁ : Not (LE.le p.natDegree n)
      h₂ : Eq n p.natDegree
      ⊢ Eq 0 p.leadingCoeff
    -/
  · exact (h₁ h₂.ge).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h₁ : Not (LE.le p.natDegree n)
      h₂ : Not (Eq n p.natDegree)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
lemma one_scaleRoots (r : R) :
                                      /-
                                        R : Type u_1
                                        inst✝ : Semiring R
                                        r : R
                                        ⊢ Eq (Polynomial.scaleRoots 1 r) 1
                                      -/
    (1 : R[X]).scaleRoots r = 1 := by ext; simp
                                           /-
                                             🎉 no goals
                                           -/


theorem scaleRoots_eval₂_mul_of_commute {p : S[X]} (f : S →+* A) (a : A) (s : S)
    (hsa : Commute (f s) a) (hf : ∀ s₁ s₂, Commute (f s₁) (f s₂)) :
    eval₂ f (f s * a) (scaleRoots p s) = f s ^ p.natDegree * eval₂ f a p := by
   calc
    _ = (scaleRoots p s).support.sum fun i =>
          f (coeff p i * s ^ (p.natDegree - i)) * (f s * a) ^ i := by
      simp [eval₂_eq_sum, sum_def]
    _ = p.support.sum fun i => f (coeff p i * s ^ (p.natDegree - i)) * (f s * a) ^ i :=
      (Finset.sum_subset (support_scaleRoots_le p s) fun i _hi hi' => by
        let this : coeff p i * s ^ (p.natDegree - i) = 0 := by simpa using hi'
        simp [this])
    _ = p.support.sum fun i : ℕ => f (p.coeff i) * f s ^ (p.natDegree - i + i) * a ^ i :=
      (Finset.sum_congr rfl fun i _hi => by
        simp_rw [f.map_mul, f.map_pow, pow_add, hsa.mul_pow, mul_assoc])
    _ = p.support.sum fun i : ℕ => f s ^ p.natDegree * (f (p.coeff i) * a ^ i) :=
      Finset.sum_congr rfl fun i hi => by
        rw [mul_assoc, ← map_pow, (hf _ _).left_comm, map_pow, tsub_add_cancel_of_le]
        exact le_natDegree_of_ne_zero (Polynomial.mem_support_iff.mp hi)
    _ = f s ^ p.natDegree * eval₂ f a p := by simp [← Finset.mul_sum, eval₂_eq_sum, sum_def]


theorem scaleRoots_eval₂_mul {p : S[X]} (f : S →+* R) (r : R) (s : S) :
    eval₂ f (f s * r) (scaleRoots p s) = f s ^ p.natDegree * eval₂ f r p :=
  scaleRoots_eval₂_mul_of_commute f r s (mul_comm _ _) fun _ _ ↦ mul_comm _ _


theorem scaleRoots_eval₂_eq_zero {p : S[X]} (f : S →+* R) {r : R} {s : S} (hr : eval₂ f r p = 0) :
                                                 /-
                                                   R : Type u_1
                                                   S : Type u_2
                                                   inst✝¹ : Semiring S
                                                   inst✝ : CommSemiring R
                                                   p : Polynomial S
                                                   f : RingHom S R
                                                   r : R
                                                   s : S
                                                   hr : Eq (Polynomial.eval₂ f r p) 0
                                                   ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f s) r) (p.scaleRoots s)) 0
                                                 -/
    eval₂ f (f s * r) (scaleRoots p s) = 0 := by rw [scaleRoots_eval₂_mul, hr, mul_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem scaleRoots_aeval_eq_zero [Algebra R A] {p : R[X]} {a : A} {r : R} (ha : aeval a p = 0) :
    aeval (algebraMap R A r * a) (scaleRoots p r) = 0 := by
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    a : A
    r : R
    ha : Eq ((Polynomial.aeval a) p) 0
    ⊢ Eq ((Polynomial.aeval (HMul.hMul ((algebraMap R A) r) a)) (p.scaleRoots r)) 0
  -/
  rw [aeval_def, scaleRoots_eval₂_mul_of_commute, ← aeval_def, ha, mul_zero]
    /-
      case hsa
      R : Type u_1
      A : Type u_3
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      p : Polynomial R
      a : A
      r : R
      ha : Eq ((Polynomial.aeval a) p) 0
      ⊢ Commute ((algebraMap R A) r) a
    -/
  · apply Algebra.commutes
    /-
      🎉 no goals
    -/
    /-
      case hf
      R : Type u_1
      A : Type u_3
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      p : Polynomial R
      a : A
      r : R
      ha : Eq ((Polynomial.aeval a) p) 0
      ⊢ ∀ (s₁ s₂ : R), Commute ((algebraMap R A) s₁) ((algebraMap R A) s₂)
    -/
  · intros; rw [Commute, SemiconjBy, ← map_mul, ← map_mul, mul_comm]
            /-
              🎉 no goals
            -/


theorem scaleRoots_eval₂_eq_zero_of_eval₂_div_eq_zero {p : S[X]} {f : S →+* K}
    (hf : Function.Injective f) {r s : S} (hr : eval₂ f (f r / f s) p = 0)
    (hs : s ∈ nonZeroDivisors S) : eval₂ f (f r) (scaleRoots p s) = 0 := by
  -- The proof works without this option, but *much* slower.
  set_option tactic.skipAssignedInstances false in
  nontriviality S using Subsingleton.eq_zero
  convert @scaleRoots_eval₂_eq_zero _ _ _ _ p f _ s hr
  rw [← mul_div_assoc, mul_comm, mul_div_cancel_right₀]
  exact map_ne_zero_of_mem_nonZeroDivisors _ hf hs


theorem scaleRoots_aeval_eq_zero_of_aeval_div_eq_zero [Algebra R K]
    (inj : Function.Injective (algebraMap R K)) {p : R[X]} {r s : R}
    (hr : aeval (algebraMap R K r / algebraMap R K s) p = 0) (hs : s ∈ nonZeroDivisors R) :
    aeval (algebraMap R K r) (scaleRoots p s) = 0 :=
  scaleRoots_eval₂_eq_zero_of_eval₂_div_eq_zero inj hr hs


@[simp]
lemma scaleRoots_mul (p : R[X]) (r s) :
    p.scaleRoots (r * s) = (p.scaleRoots r).scaleRoots s := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p : Polynomial R
    r s : R
    ⊢ Eq (p.scaleRoots (HMul.hMul r s)) ((p.scaleRoots r).scaleRoots s)
  -/
  ext; simp [mul_pow, mul_assoc]
       /-
         🎉 no goals
       -/


/-- Multiplication and `scaleRoots` commute up to a power of `r`. The factor disappears if we
assume that the product of the leading coeffs does not vanish. See `Polynomial.mul_scaleRoots'`. -/
lemma mul_scaleRoots (p q : R[X]) (r : R) :
    r ^ (natDegree p + natDegree q - natDegree (p * q)) • (p * q).scaleRoots r =
      p.scaleRoots r * q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    ⊢ Eq (HSMul.hSMul (HPow.hPow r (HSub.hSub (HAdd.hAdd p.natDegree q.natDegree)  …
  -/
  ext n; simp only [coeff_scaleRoots, coeff_smul, smul_eq_mul]
  trans (∑ x ∈ Finset.antidiagonal n, coeff p x.1 * coeff q x.2) *
    r ^ (natDegree p + natDegree q - n)
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      p q : Polynomial R
      r : R
      n : Nat
      ⊢ Eq (HMul.hMul (HPow.hPow r (HSub.hSub (HAdd.hAdd p.natDegree q.natDegree) (H …
    -/
  · rw [← coeff_mul]
    cases lt_or_le (natDegree (p * q)) n with
    | inl h => simp only [coeff_eq_zero_of_natDegree_lt h, zero_mul, mul_zero]
    | inr h =>
      rw [mul_comm, mul_assoc, ← pow_add, add_comm, tsub_add_tsub_cancel natDegree_mul_le h]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      p q : Polynomial R
      r : R
      n : Nat
      ⊢ Eq (HMul.hMul ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul.hMu …
    -/
  · rw [coeff_mul, Finset.sum_mul]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      p q : Polynomial R
      r : R
      n : Nat
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun i => HMul.hMul (HMul.hMu …
    -/
    apply Finset.sum_congr rfl
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      p q : Polynomial R
      r : R
      n : Nat
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
    simp only [Finset.mem_antidiagonal, coeff_scaleRoots, Prod.forall]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      p q : Polynomial R
      r : R
      n : Nat
      ⊢ ∀ (a b : Nat), Eq (HAdd.hAdd a b) n → Eq (HMul.hMul (HMul.hMul (p.coeff a) ( …
    -/
    intros a b e
    cases lt_or_le (natDegree p) a with
    | inl h => simp only [coeff_eq_zero_of_natDegree_lt h, zero_mul, mul_zero]
    | inr ha =>
      cases lt_or_le (natDegree q) b with
      | inl h => simp only [coeff_eq_zero_of_natDegree_lt h, zero_mul, mul_zero]
      | inr hb =>
        simp only [← e, mul_assoc, mul_comm (r ^ (_ - a)), ← pow_add]
        rw [add_comm (_ - _), tsub_add_tsub_comm ha hb]


lemma mul_scaleRoots' (p q : R[X]) (r : R) (h : leadingCoeff p * leadingCoeff q ≠ 0) :
    (p * q).scaleRoots r = p.scaleRoots r * q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    ⊢ Eq ((HMul.hMul p q).scaleRoots r) (HMul.hMul (p.scaleRoots r) (q.scaleRoots  …
  -/
  rw [← mul_scaleRoots, natDegree_mul' h, tsub_self, pow_zero, one_smul]
  /-
    🎉 no goals
  -/


lemma mul_scaleRoots_of_noZeroDivisors (p q : R[X]) (r : R) [NoZeroDivisors R] :
    (p * q).scaleRoots r = p.scaleRoots r * q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    ⊢ Eq ((HMul.hMul p q).scaleRoots r) (HMul.hMul (p.scaleRoots r) (q.scaleRoots  …
  -/
  by_cases hp : p = 0; · simp [hp]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    ⊢ Eq ((HMul.hMul p q).scaleRoots r) (HMul.hMul (p.scaleRoots r) (q.scaleRoots  …
  -/
  by_cases hq : q = 0; · simp [hq]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    hq : Not (Eq q 0)
    ⊢ Eq ((HMul.hMul p q).scaleRoots r) (HMul.hMul (p.scaleRoots r) (q.scaleRoots  …
  -/
  apply mul_scaleRoots'
  /-
    case neg.h
    R : Type u_1
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    hp : Not (Eq p 0)
    hq : Not (Eq q 0)
    ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
  -/
  simp only [ne_eq, mul_eq_zero, leadingCoeff_eq_zero, hp, hq, or_self, not_false_eq_true]
  /-
    🎉 no goals
  -/


lemma add_scaleRoots_of_natDegree_eq (p q : R[X]) (r : R) (h : natDegree p = natDegree q) :
    r ^ (natDegree p - natDegree (p + q)) • (p + q).scaleRoots r =
      p.scaleRoots r + q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    h : Eq p.natDegree q.natDegree
    ⊢ Eq (HSMul.hSMul (HPow.hPow r (HSub.hSub p.natDegree (HAdd.hAdd p q).natDegre …
  -/
  ext n; simp only [coeff_smul, coeff_scaleRoots, coeff_add, smul_eq_mul,
    mul_comm (r ^ _), ← pow_add, ← h, ← add_mul, add_comm (_ - n)]
  #adaptation_note /-- v4.7.0-rc1
  Previously `mul_assoc` was part of the `simp only` above, and this `rw` was not needed.
  but this now causes a max rec depth error. -/
  /-
    case a
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    h : Eq p.natDegree q.natDegree
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (p.coeff n) (q.coeff n)) (HPow.hPow r (H …
  -/
  rw [mul_assoc, ← pow_add]
  cases lt_or_le (natDegree (p + q)) n with
  | inl hn => simp only [← coeff_add, coeff_eq_zero_of_natDegree_lt hn, zero_mul]
  | inr hn =>
      rw [add_comm (_ - n), tsub_add_tsub_cancel (natDegree_add_le_of_degree_le le_rfl h.ge) hn]


lemma scaleRoots_dvd' (p q : R[X]) {r : R} (hr : IsUnit r)
    (hpq : p ∣ q) : p.scaleRoots r ∣ q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    hpq : Dvd.dvd p q
    ⊢ Dvd.dvd (p.scaleRoots r) (q.scaleRoots r)
  -/
  obtain ⟨a, rfl⟩ := hpq
  rw [← ((hr.pow (natDegree p + natDegree a - natDegree (p * a))).map
    (algebraMap R R[X])).dvd_mul_left, ← Algebra.smul_def, mul_scaleRoots]
  /-
    case intro
    R : Type u_1
    inst✝ : CommSemiring R
    p : Polynomial R
    r : R
    hr : IsUnit r
    a : Polynomial R
    ⊢ Dvd.dvd (p.scaleRoots r) (HMul.hMul (p.scaleRoots r) (a.scaleRoots r))
  -/
  exact dvd_mul_right (scaleRoots p r) (scaleRoots a r)
  /-
    🎉 no goals
  -/


lemma scaleRoots_dvd (p q : R[X]) {r : R} [NoZeroDivisors R] (hpq : p ∣ q) :
    p.scaleRoots r ∣ q.scaleRoots r := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    hpq : Dvd.dvd p q
    ⊢ Dvd.dvd (p.scaleRoots r) (q.scaleRoots r)
  -/
  obtain ⟨a, rfl⟩ := hpq
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    p : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    a : Polynomial R
    ⊢ Dvd.dvd (p.scaleRoots r) ((HMul.hMul p a).scaleRoots r)
  -/
  rw [mul_scaleRoots_of_noZeroDivisors]
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    p : Polynomial R
    r : R
    inst✝ : NoZeroDivisors R
    a : Polynomial R
    ⊢ Dvd.dvd (p.scaleRoots r) (HMul.hMul (p.scaleRoots r) (a.scaleRoots r))
  -/
  exact dvd_mul_right (scaleRoots p r) (scaleRoots a r)
  /-
    🎉 no goals
  -/

alias _root_.Dvd.dvd.scaleRoots := scaleRoots_dvd


lemma scaleRoots_dvd_iff (p q : R[X]) {r : R} (hr : IsUnit r) :
    p.scaleRoots r ∣ q.scaleRoots r ↔ p ∣ q := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    ⊢ Iff (Dvd.dvd (p.scaleRoots r) (q.scaleRoots r)) (Dvd.dvd p q)
  -/
  refine ⟨?_ ∘ scaleRoots_dvd' _ _ (hr.unit⁻¹).isUnit, scaleRoots_dvd' p q hr⟩
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    ⊢ Dvd.dvd ((p.scaleRoots r).scaleRoots ↑(Inv.inv hr.unit)) ((q.scaleRoots r).s …
  -/
  simp [← scaleRoots_mul, scaleRoots_one]
  /-
    🎉 no goals
  -/

alias _root_.IsUnit.scaleRoots_dvd_iff := scaleRoots_dvd_iff


lemma isCoprime_scaleRoots (p q : R[X]) (r : R) (hr : IsUnit r) (h : IsCoprime p q) :
    IsCoprime (p.scaleRoots r) (q.scaleRoots r) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    h : IsCoprime p q
    ⊢ IsCoprime (p.scaleRoots r) (q.scaleRoots r)
  -/
  obtain ⟨a, b, e⟩ := h
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    a b : Polynomial R
    e : Eq (HAdd.hAdd (HMul.hMul a p) (HMul.hMul b q)) 1
    ⊢ IsCoprime (p.scaleRoots r) (q.scaleRoots r)
  -/
  let s : R := ↑hr.unit⁻¹
  have : natDegree (a * p) = natDegree (b * q) := by
    apply natDegree_eq_of_natDegree_add_eq_zero
    rw [e, natDegree_one]
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    a b : Polynomial R
    e : Eq (HAdd.hAdd (HMul.hMul a p) (HMul.hMul b q)) 1
    s : R := ↑(Inv.inv hr.unit)
    this : Eq (HMul.hMul a p).natDegree (HMul.hMul b q).natDegree
    ⊢ IsCoprime (p.scaleRoots r) (q.scaleRoots r)
  -/
  use s ^ natDegree (a * p) • s ^ (natDegree a + natDegree p - natDegree (a * p)) • a.scaleRoots r
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    r : R
    hr : IsUnit r
    a b : Polynomial R
    e : Eq (HAdd.hAdd (HMul.hMul a p) (HMul.hMul b q)) 1
    s : R := ↑(Inv.inv hr.unit)
    this : Eq (HMul.hMul a p).natDegree (HMul.hMul b q).natDegree
    ⊢ Exists fun b => Eq (HAdd.hAdd (HMul.hMul (HSMul.hSMul (HPow.hPow s (HMul.hMu …
  -/
  use s ^ natDegree (a * p) • s ^ (natDegree b + natDegree q - natDegree (b * q)) • b.scaleRoots r
  simp only [s, smul_mul_assoc, ← mul_scaleRoots, smul_smul, Units.smul_def, mul_assoc,
    ← mul_pow, IsUnit.val_inv_mul, one_pow, mul_one, ← smul_add, one_smul, e, natDegree_one,
    one_scaleRoots, ← add_scaleRoots_of_natDegree_eq _ _ _ this, tsub_zero]

alias _root_.IsCoprime.scaleRoots := isCoprime_scaleRoots


