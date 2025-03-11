/-- `pderiv i p` is the partial derivative of `p` with respect to `i` -/
def pderiv (i : σ) : Derivation R (MvPolynomial σ R) (MvPolynomial σ R) :=
  letI := Classical.decEq σ
  mkDerivation R <| Pi.single i 1


theorem pderiv_def [DecidableEq σ] (i : σ) : pderiv i = mkDerivation R (Pi.single i 1) := by
  /-
    R : Type u
    σ : Type v
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    i : σ
    ⊢ Eq (MvPolynomial.pderiv i) (MvPolynomial.mkDerivation R (Pi.single i 1))
  -/
  unfold pderiv; congr!
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem pderiv_monomial {i : σ} :
    pderiv i (monomial s a) = monomial (s - single i 1) (a * s i) := by
  classical
  simp only [pderiv_def, mkDerivation_monomial, Finsupp.smul_sum, smul_eq_mul, ← smul_mul_assoc,
    ← (monomial _).map_smul]
  refine (Finset.sum_eq_single i (fun j _ hne => ?_) fun hi => ?_).trans ?_
  · simp [Pi.single_eq_of_ne hne]
  · rw [Finsupp.not_mem_support_iff] at hi; simp [hi]
  · simp


lemma X_mul_pderiv_monomial {i : σ} {m : σ →₀ ℕ} {r : R} :
    X i * pderiv i (monomial m r) = m i • monomial m r := by
  /-
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i : σ
    m : Finsupp σ Nat
    r : R
    ⊢ Eq (HMul.hMul (MvPolynomial.X i) ((MvPolynomial.pderiv i) ((MvPolynomial.mon …
  -/
  rw [pderiv_monomial, X, monomial_mul, smul_monomial]
  /-
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i : σ
    m : Finsupp σ Nat
    r : R
    ⊢ Eq ((MvPolynomial.monomial (HAdd.hAdd (Finsupp.single i 1) (HSub.hSub m (Fin …
  -/
  by_cases h : m i = 0
    /-
      case pos
      R : Type u
      σ : Type v
      inst✝ : CommSemiring R
      i : σ
      m : Finsupp σ Nat
      r : R
      h : Eq (m i) 0
      ⊢ Eq ((MvPolynomial.monomial (HAdd.hAdd (Finsupp.single i 1) (HSub.hSub m (Fin …
    -/
  · simp_rw [h, Nat.cast_zero, mul_zero, zero_smul, monomial_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i : σ
    m : Finsupp σ Nat
    r : R
    h : Not (Eq (m i) 0)
    ⊢ Eq ((MvPolynomial.monomial (HAdd.hAdd (Finsupp.single i 1) (HSub.hSub m (Fin …
  -/
  rw [one_mul, mul_comm, nsmul_eq_mul, add_comm, sub_add_single_one_cancel h]
  /-
    🎉 no goals
  -/


theorem pderiv_C {i : σ} : pderiv i (C a) = 0 :=
  derivation_C _ _


theorem pderiv_one {i : σ} : pderiv i (1 : MvPolynomial σ R) = 0 := pderiv_C


@[simp]
theorem pderiv_X [DecidableEq σ] (i j : σ) :
    pderiv i (X j : MvPolynomial σ R) = Pi.single (f := fun _ => _) i 1 j := by
  /-
    R : Type u
    σ : Type v
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    i j : σ
    ⊢ Eq ((MvPolynomial.pderiv i) (MvPolynomial.X j)) (Pi.single i 1 j)
  -/
  rw [pderiv_def, mkDerivation_X]
  /-
    🎉 no goals
  -/


@[simp]
                                                                            /-
                                                                              R : Type u
                                                                              σ : Type v
                                                                              inst✝ : CommSemiring R
                                                                              i : σ
                                                                              ⊢ Eq ((MvPolynomial.pderiv i) (MvPolynomial.X i)) 1
                                                                            -/
theorem pderiv_X_self (i : σ) : pderiv i (X i : MvPolynomial σ R) = 1 := by classical simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem pderiv_X_of_ne {i j : σ} (h : j ≠ i) : pderiv i (X j : MvPolynomial σ R) = 0 := by
  /-
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i j : σ
    h : Ne j i
    ⊢ Eq ((MvPolynomial.pderiv i) (MvPolynomial.X j)) 0
  -/
  classical simp [h]
  /-
    🎉 no goals
  -/


theorem pderiv_eq_zero_of_not_mem_vars {i : σ} {f : MvPolynomial σ R} (h : i ∉ f.vars) :
    pderiv i f = 0 :=
  derivation_eq_zero_of_forall_mem_vars fun _ hj => pderiv_X_of_ne <| ne_of_mem_of_not_mem hj h


theorem pderiv_monomial_single {i : σ} {n : ℕ} : pderiv i (monomial (single i n) a) =
                                              /-
                                                R : Type u
                                                σ : Type v
                                                a : R
                                                inst✝ : CommSemiring R
                                                i : σ
                                                n : Nat
                                                ⊢ Eq ((MvPolynomial.pderiv i) ((MvPolynomial.monomial (Finsupp.single i n)) a) …
                                              -/
    monomial (single i (n - 1)) (a * n) := by simp
                                              /-
                                                🎉 no goals
                                              -/


theorem pderiv_mul {i : σ} {f g : MvPolynomial σ R} :
    pderiv i (f * g) = pderiv i f * g + f * pderiv i g := by
  /-
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i : σ
    f g : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.pderiv i) (HMul.hMul f g)) (HAdd.hAdd (HMul.hMul ((MvPolyn …
  -/
  simp only [(pderiv i).leibniz f g, smul_eq_mul, mul_comm, add_comm]
  /-
    🎉 no goals
  -/


theorem pderiv_pow {i : σ} {f : MvPolynomial σ R} {n : ℕ} :
    pderiv i (f ^ n) = n * f ^ (n - 1) * pderiv i f := by
  /-
    R : Type u
    σ : Type v
    inst✝ : CommSemiring R
    i : σ
    f : MvPolynomial σ R
    n : Nat
    ⊢ Eq ((MvPolynomial.pderiv i) (HPow.hPow f n)) (HMul.hMul (HMul.hMul (↑n) (HPo …
  -/
  rw [(pderiv i).leibniz_pow f n, nsmul_eq_mul, smul_eq_mul, mul_assoc]
  /-
    🎉 no goals
  -/


theorem pderiv_C_mul {f : MvPolynomial σ R} {i : σ} : pderiv i (C a * f) = C a * pderiv i f := by
  /-
    R : Type u
    σ : Type v
    a : R
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    i : σ
    ⊢ Eq ((MvPolynomial.pderiv i) (HMul.hMul (MvPolynomial.C a) f)) (HMul.hMul (Mv …
  -/
  rw [C_mul', Derivation.map_smul, C_mul']
  /-
    🎉 no goals
  -/


theorem pderiv_map {S} [CommSemiring S] {φ : R →+* S} {f : MvPolynomial σ R} {i : σ} :
    pderiv i (map φ f) = map φ (pderiv i f) := by
  /-
    R : Type u
    σ : Type v
    inst✝¹ : CommSemiring R
    S : Type u_1
    inst✝ : CommSemiring S
    φ : RingHom R S
    f : MvPolynomial σ R
    i : σ
    ⊢ Eq ((MvPolynomial.pderiv i) ((MvPolynomial.map φ) f)) ((MvPolynomial.map φ)  …
  -/
  apply induction_on f (fun r ↦ by simp) (fun p q hp hq ↦ by simp [hp, hq]) fun p j eq ↦ ?_
  /-
    R : Type u
    σ : Type v
    inst✝¹ : CommSemiring R
    S : Type u_1
    inst✝ : CommSemiring S
    φ : RingHom R S
    f : MvPolynomial σ R
    i : σ
    p : MvPolynomial σ R
    j : σ
    eq : Eq ((MvPolynomial.pderiv i) ((MvPolynomial.map φ) p)) ((MvPolynomial.map  …
    ⊢ Eq ((MvPolynomial.pderiv i) ((MvPolynomial.map φ) (HMul.hMul p (MvPolynomial …
  -/
  obtain rfl | h := eq_or_ne j i
    /-
      case inl
      R : Type u
      σ : Type v
      inst✝¹ : CommSemiring R
      S : Type u_1
      inst✝ : CommSemiring S
      φ : RingHom R S
      f p : MvPolynomial σ R
      j : σ
      eq : Eq ((MvPolynomial.pderiv j) ((MvPolynomial.map φ) p)) ((MvPolynomial.map  …
      ⊢ Eq ((MvPolynomial.pderiv j) ((MvPolynomial.map φ) (HMul.hMul p (MvPolynomial …
    -/
  · simp [eq]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      σ : Type v
      inst✝¹ : CommSemiring R
      S : Type u_1
      inst✝ : CommSemiring S
      φ : RingHom R S
      f : MvPolynomial σ R
      i : σ
      p : MvPolynomial σ R
      j : σ
      eq : Eq ((MvPolynomial.pderiv i) ((MvPolynomial.map φ) p)) ((MvPolynomial.map  …
      h : Ne j i
      ⊢ Eq ((MvPolynomial.pderiv i) ((MvPolynomial.map φ) (HMul.hMul p (MvPolynomial …
    -/
  · simp [eq, h]
    /-
      🎉 no goals
    -/


lemma pderiv_rename {τ : Type*} {f : σ → τ} (hf : Function.Injective f)
    (x : σ) (p : MvPolynomial σ R) :
    pderiv (f x) (rename f p) = rename f (pderiv x p) := by
  classical
  induction' p using MvPolynomial.induction_on with a p q hp hq p a h
  · simp
  · simp [hp, hq]
  · simp only [map_mul, MvPolynomial.rename_X, Derivation.leibniz, MvPolynomial.pderiv_X,
      Pi.single_apply, hf.eq_iff, smul_eq_mul, mul_ite, mul_one, mul_zero, h, map_add, add_left_inj]
    split_ifs <;> simp


lemma aeval_sum_elim_pderiv_inl {S τ : Type*} [CommRing S] [Algebra R S]
    (p : MvPolynomial (σ ⊕ τ) R) (f : τ → S) (j : σ) :
    aeval (Sum.elim X (C ∘ f)) ((pderiv (Sum.inl j)) p) =
      (pderiv j) ((aeval (Sum.elim X (C ∘ f))) p) := by
  classical
  induction' p using MvPolynomial.induction_on with a p q hp hq p q h
  · simp
  · simp [hp, hq]
  · simp only [Derivation.leibniz, pderiv_X, smul_eq_mul, map_add, map_mul, aeval_X, h]
    cases q <;> simp [Pi.single_apply]


