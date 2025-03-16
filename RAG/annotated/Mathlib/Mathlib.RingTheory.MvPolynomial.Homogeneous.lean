/-- A multivariate polynomial `φ` is homogeneous of degree `n`
if all monomials occurring in `φ` have degree `n`. -/
def IsHomogeneous [CommSemiring R] (φ : MvPolynomial σ R) (n : ℕ) :=
  IsWeightedHomogeneous 1 φ n


theorem weightedTotalDegree_one (φ : MvPolynomial σ R) :
    weightedTotalDegree (1 : σ → ℕ) φ = φ.totalDegree := by
  simp only [totalDegree, weightedTotalDegree, weight, LinearMap.toAddMonoidHom_coe,
    linearCombination, Pi.one_apply, Finsupp.coe_lsum, LinearMap.coe_smulRight, LinearMap.id_coe,
    id, Algebra.id.smul_eq_mul, mul_one]


/-- The submodule of homogeneous `MvPolynomial`s of degree `n`. -/
def homogeneousSubmodule (n : ℕ) : Submodule R (MvPolynomial σ R) where
  carrier := { x | x.IsHomogeneous n }
  smul_mem' r a ha c hc := by
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (MvPolynomial.coeff c (HSMul.hSMul r a)) 0
      ⊢ Eq ((Finsupp.weight 1) c) n
    -/
    rw [coeff_smul] at hc
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
      ⊢ Eq ((Finsupp.weight 1) c) n
    -/
    apply ha
    /-
      case a
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
      ⊢ Ne (MvPolynomial.coeff c a) 0
    -/
    intro h
    /-
      case a
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
      h : Eq (MvPolynomial.coeff c a) 0
      ⊢ False
    -/
    apply hc
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      a b : MvPolynomial σ R
      ha : Membership.mem (setOf fun x => x.IsHomogeneous n) a
      hb : Membership.mem (setOf fun x => x.IsHomogeneous n) b
      c : Finsupp σ Nat
      hc : Ne (MvPolynomial.coeff c (HAdd.hAdd a b)) 0
      ⊢ Eq ((Finsupp.weight 1) c) n
    -/
    /-
      case a
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
      h : Eq (MvPolynomial.coeff c a) 0
      ⊢ Eq (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
    -/
    rw [h]
    /-
      case a
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝ : CommSemiring R
      n : Nat
      r : R
      a : MvPolynomial σ R
      ha : Membership.mem { carrier := setOf fun x => x.IsHomogeneous n, add_mem' := …
      c : Finsupp σ Nat
      hc : Ne (HSMul.hSMul r (MvPolynomial.coeff c a)) 0
      h : Eq (MvPolynomial.coeff c a) 0
      ⊢ Eq (HSMul.hSMul r 0) 0
    -/
    exact smul_zero r
      /-
        case inl
        σ : Type u_1
        τ : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝ : CommSemiring R
        n : Nat
        a b : MvPolynomial σ R
        ha : Membership.mem (setOf fun x => x.IsHomogeneous n) a
        hb : Membership.mem (setOf fun x => x.IsHomogeneous n) b
        c : Finsupp σ Nat
        hc : Ne (HAdd.hAdd (MvPolynomial.coeff c a) (MvPolynomial.coeff c b)) 0
        h : Ne (MvPolynomial.coeff c a) 0
        ⊢ Eq ((Finsupp.weight 1) c) n
      -/
    /-
      🎉 no goals
    -/
      /-
        🎉 no goals
      -/
      /-
        case inr
        σ : Type u_1
        τ : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝ : CommSemiring R
        n : Nat
        a b : MvPolynomial σ R
        ha : Membership.mem (setOf fun x => x.IsHomogeneous n) a
        hb : Membership.mem (setOf fun x => x.IsHomogeneous n) b
        c : Finsupp σ Nat
        hc : Ne (HAdd.hAdd (MvPolynomial.coeff c a) (MvPolynomial.coeff c b)) 0
        h : Ne (MvPolynomial.coeff c b) 0
        ⊢ Eq ((Finsupp.weight 1) c) n
      -/
  zero_mem' _ hd := False.elim (hd <| coeff_zero _)
      /-
        🎉 no goals
      -/
  add_mem' {a b} ha hb c hc := by
    rw [coeff_add] at hc
    obtain h | h : coeff c a ≠ 0 ∨ coeff c b ≠ 0 := by
      contrapose! hc
      simp only [hc, add_zero]
    · exact ha h
    · exact hb h


@[simp]
lemma weightedHomogeneousSubmodule_one (n : ℕ) :
    weightedHomogeneousSubmodule R 1 n = homogeneousSubmodule σ R n := rfl


@[simp]
theorem mem_homogeneousSubmodule (n : ℕ) (p : MvPolynomial σ R) :
    p ∈ homogeneousSubmodule σ R n ↔ p.IsHomogeneous n := Iff.rfl


/-- While equal, the former has a convenient definitional reduction. -/
theorem homogeneousSubmodule_eq_finsupp_supported (n : ℕ) :
    homogeneousSubmodule σ R n = Finsupp.supported _ R { d | d.degree = n } := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    ⊢ Eq (MvPolynomial.homogeneousSubmodule σ R n) (Finsupp.supported R R (setOf f …
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    ⊢ Eq (MvPolynomial.homogeneousSubmodule σ R n) (Finsupp.supported R R (setOf f …
  -/
  exact weightedHomogeneousSubmodule_eq_finsupp_supported R 1 n
  /-
    🎉 no goals
  -/


theorem homogeneousSubmodule_mul (m n : ℕ) :
    homogeneousSubmodule σ R m * homogeneousSubmodule σ R n ≤ homogeneousSubmodule σ R (m + n) :=
  weightedHomogeneousSubmodule_mul 1 m n


theorem isHomogeneous_monomial {d : σ →₀ ℕ} (r : R) {n : ℕ} (hn : d.degree = n) :
    IsHomogeneous (monomial d r) n := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    d : Finsupp σ Nat
    r : R
    n : Nat
    hn : Eq d.degree n
    ⊢ ((MvPolynomial.monomial d) r).IsHomogeneous n
  -/
  rw [degree_eq_weight_one] at hn
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    d : Finsupp σ Nat
    r : R
    n : Nat
    hn : Eq ((Finsupp.weight 1) d) n
    ⊢ ((MvPolynomial.monomial d) r).IsHomogeneous n
  -/
  exact isWeightedHomogeneous_monomial 1 d r hn
  /-
    🎉 no goals
  -/


theorem totalDegree_eq_zero_iff (p : MvPolynomial σ R) :
    p.totalDegree = 0 ↔ ∀ (m : σ →₀ ℕ) (_ : m ∈ p.support) (x : σ), m x = 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Eq p.totalDegree 0) (∀ (m : Finsupp σ Nat), Membership.mem p.support m  …
  -/
  rw [← weightedTotalDegree_one, weightedTotalDegree_eq_zero_iff _ p]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ MvPolynomial.NonTorsionWeight 1
  -/
  exact nonTorsionWeight_of (Function.const σ one_ne_zero)
  /-
    🎉 no goals
  -/


theorem totalDegree_zero_iff_isHomogeneous {p : MvPolynomial σ R} :
    p.totalDegree = 0 ↔ IsHomogeneous p 0 := by
  rw [← weightedTotalDegree_one,
    ← isWeightedHomogeneous_zero_iff_weightedTotalDegree_eq_zero, IsHomogeneous]


alias ⟨isHomogeneous_of_totalDegree_zero, _⟩ := totalDegree_zero_iff_isHomogeneous


theorem isHomogeneous_C (r : R) : IsHomogeneous (C r : MvPolynomial σ R) 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    r : R
    ⊢ (MvPolynomial.C r).IsHomogeneous 0
  -/
  apply isHomogeneous_monomial
  /-
    case hn
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq (Finsupp.degree 0) 0
  -/
  simp only [Finsupp.degree, Finsupp.zero_apply, Finset.sum_const_zero]
  /-
    🎉 no goals
  -/


theorem isHomogeneous_zero (n : ℕ) : IsHomogeneous (0 : MvPolynomial σ R) n :=
  (homogeneousSubmodule σ R n).zero_mem


theorem isHomogeneous_one : IsHomogeneous (1 : MvPolynomial σ R) 0 :=
  isHomogeneous_C _ _


theorem isHomogeneous_X (i : σ) : IsHomogeneous (X i : MvPolynomial σ R) 1 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    i : σ
    ⊢ (MvPolynomial.X i).IsHomogeneous 1
  -/
  apply isHomogeneous_monomial
  /-
    case hn
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    i : σ
    ⊢ Eq (Finsupp.single i 1).degree 1
  -/
  rw [Finsupp.degree, Finsupp.support_single_ne_zero _ one_ne_zero, Finset.sum_singleton]
  /-
    case hn
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    i : σ
    ⊢ Eq ((Finsupp.single i 1) i) 1
  -/
  exact Finsupp.single_eq_same
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero (hφ : IsHomogeneous φ n) {d : σ →₀ ℕ} (hd : d.degree ≠ n) :
    coeff d φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Ne d.degree n
    ⊢ Eq (MvPolynomial.coeff d φ) 0
  -/
  rw [degree_eq_weight_one] at hd
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Ne ((Finsupp.weight 1) d) n
    ⊢ Eq (MvPolynomial.coeff d φ) 0
  -/
  exact IsWeightedHomogeneous.coeff_eq_zero hφ d hd
  /-
    🎉 no goals
  -/


theorem inj_right (hm : IsHomogeneous φ m) (hn : IsHomogeneous φ n) (hφ : φ ≠ 0) : m = n := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m n : Nat
    hm : φ.IsHomogeneous m
    hn : φ.IsHomogeneous n
    hφ : Ne φ 0
    ⊢ Eq m n
  -/
  obtain ⟨d, hd⟩ : ∃ d, coeff d φ ≠ 0 := exists_coeff_ne_zero hφ
  /-
    case intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m n : Nat
    hm : φ.IsHomogeneous m
    hn : φ.IsHomogeneous n
    hφ : Ne φ 0
    d : Finsupp σ Nat
    hd : Ne (MvPolynomial.coeff d φ) 0
    ⊢ Eq m n
  -/
  rw [← hm hd, ← hn hd]
  /-
    🎉 no goals
  -/


theorem add (hφ : IsHomogeneous φ n) (hψ : IsHomogeneous ψ n) : IsHomogeneous (φ + ψ) n :=
  (homogeneousSubmodule σ R n).add_mem hφ hψ


theorem sum {ι : Type*} (s : Finset ι) (φ : ι → MvPolynomial σ R) (n : ℕ)
    (h : ∀ i ∈ s, IsHomogeneous (φ i) n) : IsHomogeneous (∑ i ∈ s, φ i) n :=
  (homogeneousSubmodule σ R n).sum_mem h


theorem mul (hφ : IsHomogeneous φ m) (hψ : IsHomogeneous ψ n) : IsHomogeneous (φ * ψ) (m + n) :=
  homogeneousSubmodule_mul m n <| Submodule.mul_mem_mul hφ hψ


theorem prod {ι : Type*} (s : Finset ι) (φ : ι → MvPolynomial σ R) (n : ι → ℕ)
    (h : ∀ i ∈ s, IsHomogeneous (φ i) (n i)) : IsHomogeneous (∏ i ∈ s, φ i) (∑ i ∈ s, n i) := by
  classical
  revert h
  refine Finset.induction_on s ?_ ?_
  · intro
    simp only [isHomogeneous_one, Finset.sum_empty, Finset.prod_empty]
  · intro i s his IH h
    simp only [his, Finset.prod_insert, Finset.sum_insert, not_false_iff]
    apply (h i (Finset.mem_insert_self _ _)).mul (IH _)
    intro j hjs
    exact h j (Finset.mem_insert_of_mem hjs)


lemma C_mul (hφ : φ.IsHomogeneous m) (r : R) :
    (C r * φ).IsHomogeneous m := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m : Nat
    hφ : φ.IsHomogeneous m
    r : R
    ⊢ (HMul.hMul (MvPolynomial.C r) φ).IsHomogeneous m
  -/
  simpa only [zero_add] using (isHomogeneous_C _ _).mul hφ
  /-
    🎉 no goals
  -/


lemma _root_.MvPolynomial.isHomogeneous_C_mul_X (r : R) (i : σ) :
    (C r * X i).IsHomogeneous 1 :=
  (isHomogeneous_X _ _).C_mul _


@[deprecated (since := "2024-03-21")]
alias _root_.MvPolynomial.C_mul_X := _root_.MvPolynomial.isHomogeneous_C_mul_X


lemma pow (hφ : φ.IsHomogeneous m) (n : ℕ) : (φ ^ n).IsHomogeneous (m * n) := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m : Nat
    hφ : φ.IsHomogeneous m
    n : Nat
    ⊢ (HPow.hPow φ n).IsHomogeneous (HMul.hMul m n)
  -/
  rw [show φ ^ n = ∏ _i ∈ Finset.range n, φ by simp]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m : Nat
    hφ : φ.IsHomogeneous m
    n : Nat
    ⊢ ((Finset.range n).prod fun _i => φ).IsHomogeneous (HMul.hMul m n)
  -/
  rw [show m * n = ∑ _i ∈ Finset.range n, m by simp [mul_comm]]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    m : Nat
    hφ : φ.IsHomogeneous m
    n : Nat
    ⊢ ((Finset.range n).prod fun _i => φ).IsHomogeneous ((Finset.range n).sum fun  …
  -/
  apply IsHomogeneous.prod _ _ _ (fun _ _ ↦ hφ)
  /-
    🎉 no goals
  -/


lemma _root_.MvPolynomial.isHomogeneous_X_pow (i : σ) (n : ℕ) :
    (X (R := R) i ^ n).IsHomogeneous n := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    i : σ
    n : Nat
    ⊢ (HPow.hPow (MvPolynomial.X i) n).IsHomogeneous n
  -/
  simpa only [one_mul] using (isHomogeneous_X _ _).pow n
  /-
    🎉 no goals
  -/


lemma _root_.MvPolynomial.isHomogeneous_C_mul_X_pow (r : R) (i : σ) (n : ℕ) :
    (C r * X i ^ n).IsHomogeneous n :=
  (isHomogeneous_X_pow _ _).C_mul _


lemma eval₂ (hφ : φ.IsHomogeneous m) (f : R →+* MvPolynomial τ S) (g : σ → MvPolynomial τ S)
    (hf : ∀ r, (f r).IsHomogeneous 0) (hg : ∀ i, (g i).IsHomogeneous n) :
    (eval₂ f g φ).IsHomogeneous (n * m) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    m n : Nat
    hφ : φ.IsHomogeneous m
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    hf : ∀ (r : R), (f r).IsHomogeneous 0
    hg : ∀ (i : σ), (g i).IsHomogeneous n
    ⊢ (MvPolynomial.eval₂ f g φ).IsHomogeneous (HMul.hMul n m)
  -/
  apply IsHomogeneous.sum
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    m n : Nat
    hφ : φ.IsHomogeneous m
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    hf : ∀ (r : R), (f r).IsHomogeneous 0
    hg : ∀ (i : σ), (g i).IsHomogeneous n
    ⊢ ∀ (i : Finsupp σ Nat), Membership.mem φ.support i → ((fun s a => HMul.hMul ( …
  -/
  intro i hi
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    m n : Nat
    hφ : φ.IsHomogeneous m
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    hf : ∀ (r : R), (f r).IsHomogeneous 0
    hg : ∀ (i : σ), (g i).IsHomogeneous n
    i : Finsupp σ Nat
    hi : Membership.mem φ.support i
    ⊢ ((fun s a => HMul.hMul (f a) (s.prod fun n e => HPow.hPow (g n) e)) i (φ i)) …
  -/
  rw [← zero_add (n * m)]
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    m n : Nat
    hφ : φ.IsHomogeneous m
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    hf : ∀ (r : R), (f r).IsHomogeneous 0
    hg : ∀ (i : σ), (g i).IsHomogeneous n
    i : Finsupp σ Nat
    hi : Membership.mem φ.support i
    ⊢ ((fun s a => HMul.hMul (f a) (s.prod fun n e => HPow.hPow (g n) e)) i (φ i)) …
  -/
  apply IsHomogeneous.mul (hf _) _
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    m n : Nat
    hφ : φ.IsHomogeneous m
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    hf : ∀ (r : R), (f r).IsHomogeneous 0
    hg : ∀ (i : σ), (g i).IsHomogeneous n
    i : Finsupp σ Nat
    hi : Membership.mem φ.support i
    ⊢ (i.prod fun n e => HPow.hPow (g n) e).IsHomogeneous (HMul.hMul n m)
  -/
  convert IsHomogeneous.prod _ _ (fun k ↦ n * i k) _
    /-
      case h.e'_1
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      φ : MvPolynomial σ R
      m n : Nat
      hφ : φ.IsHomogeneous m
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      hf : ∀ (r : R), (f r).IsHomogeneous 0
      hg : ∀ (i : σ), (g i).IsHomogeneous n
      i : Finsupp σ Nat
      hi : Membership.mem φ.support i
      ⊢ Eq (HMul.hMul n m) (i.support.sum fun i_1 => HMul.hMul n (i i_1))
    -/
  · rw [Finsupp.mem_support_iff] at hi
    /-
      case h.e'_1
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      φ : MvPolynomial σ R
      m n : Nat
      hφ : φ.IsHomogeneous m
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      hf : ∀ (r : R), (f r).IsHomogeneous 0
      hg : ∀ (i : σ), (g i).IsHomogeneous n
      i : Finsupp σ Nat
      hi : Ne (φ i) 0
      ⊢ Eq (HMul.hMul n m) (i.support.sum fun i_1 => HMul.hMul n (i i_1))
    -/
    rw [← Finset.mul_sum, ← hφ hi, weight_apply]
    /-
      case h.e'_1
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      φ : MvPolynomial σ R
      m n : Nat
      hφ : φ.IsHomogeneous m
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      hf : ∀ (r : R), (f r).IsHomogeneous 0
      hg : ∀ (i : σ), (g i).IsHomogeneous n
      i : Finsupp σ Nat
      hi : Ne (φ i) 0
      ⊢ Eq (HMul.hMul n (i.sum fun i c => HSMul.hSMul c (1 i))) (HMul.hMul n (i.supp …
    -/
    simp_rw [smul_eq_mul, Finsupp.sum, Pi.one_apply, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case convert_6
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      φ : MvPolynomial σ R
      m n : Nat
      hφ : φ.IsHomogeneous m
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      hf : ∀ (r : R), (f r).IsHomogeneous 0
      hg : ∀ (i : σ), (g i).IsHomogeneous n
      i : Finsupp σ Nat
      hi : Membership.mem φ.support i
      ⊢ ∀ (i_1 : σ), Membership.mem i.support i_1 → ((fun n e => HPow.hPow (g n) e)  …
    -/
  · rintro k -
    /-
      case convert_6
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      φ : MvPolynomial σ R
      m n : Nat
      hφ : φ.IsHomogeneous m
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      hf : ∀ (r : R), (f r).IsHomogeneous 0
      hg : ∀ (i : σ), (g i).IsHomogeneous n
      i : Finsupp σ Nat
      hi : Membership.mem φ.support i
      k : σ
      ⊢ ((fun n e => HPow.hPow (g n) e) k (i k)).IsHomogeneous ((fun k => HMul.hMul  …
    -/
    apply (hg k).pow
    /-
      🎉 no goals
    -/


lemma map (hφ : φ.IsHomogeneous n) (f : R →+* S) : (map f φ).IsHomogeneous n := by
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    f : RingHom R S
    ⊢ ((MvPolynomial.map f) φ).IsHomogeneous n
  -/
  simpa only [one_mul] using hφ.eval₂ _ _ (fun r ↦ isHomogeneous_C _ (f r)) (isHomogeneous_X _)
  /-
    🎉 no goals
  -/


lemma aeval [Algebra R S] (hφ : φ.IsHomogeneous m)
    (g : σ → MvPolynomial τ S) (hg : ∀ i, (g i).IsHomogeneous n) :
    (aeval g φ).IsHomogeneous (n * m) :=
  hφ.eval₂ _ _ (fun _ ↦ isHomogeneous_C _ _) hg


theorem neg (hφ : IsHomogeneous φ n) : IsHomogeneous (-φ) n :=
  (homogeneousSubmodule σ R n).neg_mem hφ


theorem sub (hφ : IsHomogeneous φ n) (hψ : IsHomogeneous ψ n) : IsHomogeneous (φ - ψ) n :=
  (homogeneousSubmodule σ R n).sub_mem hφ hψ


/-- The homogeneous degree bounds the total degree.

See also `MvPolynomial.IsHomogeneous.totalDegree` when `φ` is non-zero. -/
lemma totalDegree_le (hφ : IsHomogeneous φ n) : φ.totalDegree ≤ n := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    ⊢ LE.le φ.totalDegree n
  -/
  apply Finset.sup_le
  /-
    case a
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    ⊢ ∀ (b : Finsupp σ Nat), Membership.mem φ.support b → LE.le (b.sum fun x e =>  …
  -/
  intro d hd
  /-
    case a
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ LE.le (d.sum fun x e => e) n
  -/
  rw [mem_support_iff] at hd
  simp_rw [Finsupp.sum, ← hφ hd, weight_apply, Pi.one_apply, smul_eq_mul, mul_one, Finsupp.sum,
    le_rfl]


theorem totalDegree (hφ : IsHomogeneous φ n) (h : φ ≠ 0) : totalDegree φ = n := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    ⊢ Eq φ.totalDegree n
  -/
  apply le_antisymm hφ.totalDegree_le
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    ⊢ LE.le n φ.totalDegree
  -/
  obtain ⟨d, hd⟩ : ∃ d, coeff d φ ≠ 0 := exists_coeff_ne_zero h
  /-
    case intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    d : Finsupp σ Nat
    hd : Ne (MvPolynomial.coeff d φ) 0
    ⊢ LE.le n φ.totalDegree
  -/
  simp only [← hφ hd, MvPolynomial.totalDegree, Finsupp.sum]
  /-
    case intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    d : Finsupp σ Nat
    hd : Ne (MvPolynomial.coeff d φ) 0
    ⊢ LE.le ((Finsupp.weight 1) d) (φ.support.sup fun s => s.support.sum fun x =>  …
  -/
  replace hd := Finsupp.mem_support_iff.mpr hd
  /-
    case intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ LE.le ((Finsupp.weight 1) d) (φ.support.sup fun s => s.support.sum fun x =>  …
  -/
  simp only [weight_apply, Pi.one_apply, smul_eq_mul, mul_one]
  -- Porting note: Original proof did not define `f`
  /-
    case intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    hφ : φ.IsHomogeneous n
    h : Ne φ 0
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ LE.le (d.sum fun i c => c) (φ.support.sup fun s => s.support.sum fun x => s x)
  -/
  exact Finset.le_sup (f := fun s ↦ ∑ x ∈ s.support, s x) hd
  /-
    🎉 no goals
  -/


theorem rename_isHomogeneous {f : σ → τ} (h : φ.IsHomogeneous n) :
    (rename f φ).IsHomogeneous n := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    ⊢ ((MvPolynomial.rename f) φ).IsHomogeneous n
  -/
  rw [← φ.support_sum_monomial_coeff, map_sum]; simp_rw [rename_monomial]
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    ⊢ (φ.support.sum fun x => (MvPolynomial.monomial (Finsupp.mapDomain f x)) (MvP …
  -/
  apply IsHomogeneous.sum _ _ _ fun d hd ↦ isHomogeneous_monomial _ _
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    ⊢ ∀ (d : Finsupp σ Nat), Membership.mem φ.support d → Eq (Finsupp.mapDomain f  …
  -/
  intro d hd
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ Eq (Finsupp.mapDomain f d).degree n
  -/
  apply (Finsupp.sum_mapDomain_index_addMonoidHom fun _ ↦ .id ℕ).trans
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ Eq (d.sum fun a m => (AddMonoidHom.id Nat) m) n
  -/
  convert h (mem_support_iff.mp hd)
  /-
    case h.e'_2
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    h : φ.IsHomogeneous n
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ Eq (d.sum fun a m => (AddMonoidHom.id Nat) m) ((Finsupp.weight 1) d)
  -/
  simp only [weight_apply, AddMonoidHom.id_apply, Pi.one_apply, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem rename_isHomogeneous_iff {f : σ → τ} (hf : f.Injective) :
    (rename f φ).IsHomogeneous n ↔ φ.IsHomogeneous n := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    hf : Function.Injective f
    ⊢ Iff (((MvPolynomial.rename f) φ).IsHomogeneous n) (φ.IsHomogeneous n)
  -/
  refine ⟨fun h d hd ↦ ?_, rename_isHomogeneous⟩
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    f : σ → τ
    hf : Function.Injective f
    h : ((MvPolynomial.rename f) φ).IsHomogeneous n
    d : Finsupp σ Nat
    hd : Ne (MvPolynomial.coeff d φ) 0
    ⊢ Eq ((Finsupp.weight 1) d) n
  -/
  convert ← @h (d.mapDomain f) _
    /-
      case h.e'_2
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      n : Nat
      f : σ → τ
      hf : Function.Injective f
      h : ((MvPolynomial.rename f) φ).IsHomogeneous n
      d : Finsupp σ Nat
      hd : Ne (MvPolynomial.coeff d φ) 0
      ⊢ Eq ((Finsupp.weight 1) (Finsupp.mapDomain f d)) ((Finsupp.weight 1) d)
    -/
  · simp only [weight_apply, Pi.one_apply, smul_eq_mul, mul_one]
    /-
      case h.e'_2
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      n : Nat
      f : σ → τ
      hf : Function.Injective f
      h : ((MvPolynomial.rename f) φ).IsHomogeneous n
      d : Finsupp σ Nat
      hd : Ne (MvPolynomial.coeff d φ) 0
      ⊢ Eq ((Finsupp.mapDomain f d).sum fun i c => c) (d.sum fun i c => c)
    -/
    exact Finsupp.sum_mapDomain_index_inj (h := fun _ ↦ id) hf
    /-
      🎉 no goals
    -/
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      n : Nat
      f : σ → τ
      hf : Function.Injective f
      h : ((MvPolynomial.rename f) φ).IsHomogeneous n
      d : Finsupp σ Nat
      hd : Ne (MvPolynomial.coeff d φ) 0
      ⊢ Ne (MvPolynomial.coeff (Finsupp.mapDomain f d) ((MvPolynomial.rename f) φ)) 0
    -/
  · rwa [coeff_rename_mapDomain f hf]
    /-
      🎉 no goals
    -/


lemma finSuccEquiv_coeff_isHomogeneous {N : ℕ} {φ : MvPolynomial (Fin (N+1)) R} {n : ℕ}
    (hφ : φ.IsHomogeneous n) (i j : ℕ) (h : i + j = n) :
    ((finSuccEquiv _ _ φ).coeff i).IsHomogeneous j := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    φ : MvPolynomial (Fin (HAdd.hAdd N 1)) R
    n : Nat
    hφ : φ.IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    ⊢ (((MvPolynomial.finSuccEquiv R N) φ).coeff i).IsHomogeneous j
  -/
  intro d hd
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    φ : MvPolynomial (Fin (HAdd.hAdd N 1)) R
    n : Nat
    hφ : φ.IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    d : Finsupp (Fin N) Nat
    hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) φ).coeff i)) 0
    ⊢ Eq ((Finsupp.weight 1) d) j
  -/
  rw [finSuccEquiv_coeff_coeff] at hd
  have h' : (weight 1) (Finsupp.cons i d) = i + j := by
    simpa [Finset.sum_subset_zero_on_sdiff (g := d.cons i)
     (d.cons_support (y := i)) (by simp) (fun _ _ ↦ rfl), ← h] using hφ hd
  simp only [weight_apply, Pi.one_apply, smul_eq_mul, mul_one, Finsupp.sum_cons,
    add_right_inj] at h' ⊢
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    φ : MvPolynomial (Fin (HAdd.hAdd N 1)) R
    n : Nat
    hφ : φ.IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    d : Finsupp (Fin N) Nat
    hd : Ne (MvPolynomial.coeff (Finsupp.cons i d) φ) 0
    h' : Eq (d.sum fun x e => e) j
    ⊢ Eq (d.sum fun i c => c) j
  -/
  exact h'
  /-
    🎉 no goals
  -/

-- TODO: develop API for `optionEquivLeft` and get rid of the `[Fintype σ]` assumption

lemma coeff_isHomogeneous_of_optionEquivLeft_symm
    [hσ : Finite σ] {p : Polynomial (MvPolynomial σ R)}
    (hp : ((optionEquivLeft R σ).symm p).IsHomogeneous n) (i j : ℕ) (h : i + j = n) :
    (p.coeff i).IsHomogeneous j := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  obtain ⟨k, ⟨e⟩⟩ := Finite.exists_equiv_fin σ
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  let e' := e.optionCongr.trans (_root_.finSuccEquiv _).symm
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  let F := renameEquiv R e
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  let F' := renameEquiv R e'
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  let φ := F' ((optionEquivLeft R σ).symm p)
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    φ : MvPolynomial (Fin (HAdd.hAdd k 1)) R := F' ((MvPolynomial.optionEquivLeft  …
    ⊢ (p.coeff i).IsHomogeneous j
  -/
  have hφ : φ.IsHomogeneous n := hp.rename_isHomogeneous
  suffices IsHomogeneous (F (p.coeff i)) j by
    rwa [← (IsHomogeneous.rename_isHomogeneous_iff e.injective)]
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    φ : MvPolynomial (Fin (HAdd.hAdd k 1)) R := F' ((MvPolynomial.optionEquivLeft  …
    hφ : φ.IsHomogeneous n
    ⊢ (F (p.coeff i)).IsHomogeneous j
  -/
  convert hφ.finSuccEquiv_coeff_isHomogeneous i j h using 1
  /-
    case h.e'_4
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    φ : MvPolynomial (Fin (HAdd.hAdd k 1)) R := F' ((MvPolynomial.optionEquivLeft  …
    hφ : φ.IsHomogeneous n
    ⊢ Eq (F (p.coeff i)) (((MvPolynomial.finSuccEquiv R k) φ).coeff i)
  -/
  dsimp only [φ, F', F, renameEquiv_apply]
  /-
    case h.e'_4
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    φ : MvPolynomial (Fin (HAdd.hAdd k 1)) R := F' ((MvPolynomial.optionEquivLeft  …
    hφ : φ.IsHomogeneous n
    ⊢ Eq ((MvPolynomial.rename ⇑e) (p.coeff i)) (((MvPolynomial.finSuccEquiv R k)  …
  -/
  rw [finSuccEquiv_rename_finSuccEquiv, AlgEquiv.apply_symm_apply]
  /-
    case h.e'_4
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    hσ : Finite σ
    p : Polynomial (MvPolynomial σ R)
    hp : ((MvPolynomial.optionEquivLeft R σ).symm p).IsHomogeneous n
    i j : Nat
    h : Eq (HAdd.hAdd i j) n
    k : Nat
    e : Equiv σ (Fin k)
    e' : Equiv (Option σ) (Fin (HAdd.hAdd k 1)) := e.optionCongr.trans (_root_.fin …
    F : AlgEquiv R (MvPolynomial σ R) (MvPolynomial (Fin k) R) := MvPolynomial.ren …
    F' : AlgEquiv R (MvPolynomial (Option σ) R) (MvPolynomial (Fin (HAdd.hAdd k 1) …
    φ : MvPolynomial (Fin (HAdd.hAdd k 1)) R := F' ((MvPolynomial.optionEquivLeft  …
    hφ : φ.IsHomogeneous n
    ⊢ Eq ((MvPolynomial.rename ⇑e) (p.coeff i)) ((Polynomial.map (MvPolynomial.ren …
  -/
  simp
  /-
    🎉 no goals
  -/


open Polynomial in
private
lemma exists_eval_ne_zero_of_coeff_finSuccEquiv_ne_zero_aux
    {N : ℕ} {F : MvPolynomial (Fin (Nat.succ N)) R} {n : ℕ} (hF : IsHomogeneous F n)
    (hFn : ((finSuccEquiv R N) F).coeff n ≠ 0) :
    ∃ r, eval r F ≠ 0 := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hFn : Ne (((MvPolynomial.finSuccEquiv R N) F).coeff n) 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) F) 0
  -/
  have hF₀ : F ≠ 0 := by contrapose! hFn; simp [hFn]
  have hdeg : natDegree (finSuccEquiv R N F) < n + 1 := by
    linarith [natDegree_finSuccEquiv F, degreeOf_le_totalDegree F 0, hF.totalDegree hF₀]
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hFn : Ne (((MvPolynomial.finSuccEquiv R N) F).coeff n) 0
    hF₀ : Ne F 0
    hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) F) 0
  -/
  use Fin.cons 1 0
  have aux : ∀ i ∈ Finset.range n, constantCoeff ((finSuccEquiv R N F).coeff i) = 0 := by
    intro i hi
    rw [Finset.mem_range] at hi
    apply (hF.finSuccEquiv_coeff_isHomogeneous i (n-i) (by omega)).coeff_eq_zero
    simp only [Finsupp.degree_zero]
    rw [← Nat.sub_ne_zero_iff_lt] at hi
    exact hi.symm
  simp_rw [eval_eq_eval_mv_eval', eval_one_map, Polynomial.eval_eq_sum_range' hdeg,
    eval_zero, one_pow, mul_one, map_sum, Finset.sum_range_succ, Finset.sum_eq_zero aux, zero_add]
  /-
    case h
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hFn : Ne (((MvPolynomial.finSuccEquiv R N) F).coeff n) 0
    hF₀ : Ne F 0
    hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
    aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
    ⊢ Ne (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coeff n) …
  -/
  contrapose! hFn
  /-
    case h
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hF₀ : Ne F 0
    hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
    aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
    hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
    ⊢ Eq (((MvPolynomial.finSuccEquiv R N) F).coeff n) 0
  -/
  ext d
  /-
    case h.a
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hF₀ : Ne F 0
    hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
    aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
    hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
    d : Finsupp (Fin N) Nat
    ⊢ Eq (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) (MvP …
  -/
  rw [coeff_zero]
  /-
    case h.a
    R : Type u_3
    inst✝ : CommSemiring R
    N : Nat
    F : MvPolynomial (Fin N.succ) R
    n : Nat
    hF : F.IsHomogeneous n
    hF₀ : Ne F 0
    hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
    aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
    hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
    d : Finsupp (Fin N) Nat
    ⊢ Eq (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
  -/
  obtain rfl | hd := eq_or_ne d 0
    /-
      case h.a.inl
      R : Type u_3
      inst✝ : CommSemiring R
      N : Nat
      F : MvPolynomial (Fin N.succ) R
      n : Nat
      hF : F.IsHomogeneous n
      hF₀ : Ne F 0
      hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
      aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
      hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
      ⊢ Eq (MvPolynomial.coeff 0 (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
    -/
  · apply hFn
    /-
      🎉 no goals
    -/
    /-
      case h.a.inr
      R : Type u_3
      inst✝ : CommSemiring R
      N : Nat
      F : MvPolynomial (Fin N.succ) R
      n : Nat
      hF : F.IsHomogeneous n
      hF₀ : Ne F 0
      hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
      aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
      hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
      d : Finsupp (Fin N) Nat
      hd : Ne d 0
      ⊢ Eq (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
    -/
  · contrapose! hd
    /-
      case h.a.inr
      R : Type u_3
      inst✝ : CommSemiring R
      N : Nat
      F : MvPolynomial (Fin N.succ) R
      n : Nat
      hF : F.IsHomogeneous n
      hF₀ : Ne F 0
      hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
      aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
      hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
      d : Finsupp (Fin N) Nat
      hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
      ⊢ Eq d 0
    -/
    ext i
    /-
      case h.a.inr.h
      R : Type u_3
      inst✝ : CommSemiring R
      N : Nat
      F : MvPolynomial (Fin N.succ) R
      n : Nat
      hF : F.IsHomogeneous n
      hF₀ : Ne F 0
      hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
      aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
      hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
      d : Finsupp (Fin N) Nat
      hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
      i : Fin N
      ⊢ Eq (d i) (0 i)
    -/
    rw [Finsupp.coe_zero, Pi.zero_apply]
    /-
      case h.a.inr.h
      R : Type u_3
      inst✝ : CommSemiring R
      N : Nat
      F : MvPolynomial (Fin N.succ) R
      n : Nat
      hF : F.IsHomogeneous n
      hF₀ : Ne F 0
      hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
      aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
      hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
      d : Finsupp (Fin N) Nat
      hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
      i : Fin N
      ⊢ Eq (d i) 0
    -/
    by_cases hi : i ∈ d.support
      /-
        case pos
        R : Type u_3
        inst✝ : CommSemiring R
        N : Nat
        F : MvPolynomial (Fin N.succ) R
        n : Nat
        hF : F.IsHomogeneous n
        hF₀ : Ne F 0
        hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
        aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
        hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
        d : Finsupp (Fin N) Nat
        hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
        i : Fin N
        hi : Membership.mem d.support i
        ⊢ Eq (d i) 0
      -/
    · have := hF.finSuccEquiv_coeff_isHomogeneous n 0 (add_zero _) hd
      /-
        case pos
        R : Type u_3
        inst✝ : CommSemiring R
        N : Nat
        F : MvPolynomial (Fin N.succ) R
        n : Nat
        hF : F.IsHomogeneous n
        hF₀ : Ne F 0
        hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
        aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
        hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
        d : Finsupp (Fin N) Nat
        hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
        i : Fin N
        hi : Membership.mem d.support i
        this : Eq ((Finsupp.weight 1) d) 0
        ⊢ Eq (d i) 0
      -/
      simp only [weight_apply, Pi.one_apply, smul_eq_mul, mul_one, Finsupp.sum] at this
      /-
        case pos
        R : Type u_3
        inst✝ : CommSemiring R
        N : Nat
        F : MvPolynomial (Fin N.succ) R
        n : Nat
        hF : F.IsHomogeneous n
        hF₀ : Ne F 0
        hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
        aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
        hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
        d : Finsupp (Fin N) Nat
        hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
        i : Fin N
        hi : Membership.mem d.support i
        this : Eq (d.support.sum fun x => d x) 0
        ⊢ Eq (d i) 0
      -/
      rw [Finset.sum_eq_zero_iff_of_nonneg (fun _ _ ↦ zero_le')] at this
      /-
        case pos
        R : Type u_3
        inst✝ : CommSemiring R
        N : Nat
        F : MvPolynomial (Fin N.succ) R
        n : Nat
        hF : F.IsHomogeneous n
        hF₀ : Ne F 0
        hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
        aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
        hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
        d : Finsupp (Fin N) Nat
        hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
        i : Fin N
        hi : Membership.mem d.support i
        this : ∀ (i : Fin N), Membership.mem d.support i → Eq (d i) 0
        ⊢ Eq (d i) 0
      -/
      exact this i hi
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_3
        inst✝ : CommSemiring R
        N : Nat
        F : MvPolynomial (Fin N.succ) R
        n : Nat
        hF : F.IsHomogeneous n
        hF₀ : Ne F 0
        hdeg : LT.lt ((MvPolynomial.finSuccEquiv R N) F).natDegree (HAdd.hAdd n 1)
        aux : ∀ (i : Nat), Membership.mem (Finset.range n) i → Eq (MvPolynomial.consta …
        hFn : Eq (MvPolynomial.constantCoeff (((MvPolynomial.finSuccEquiv R N) F).coef …
        d : Finsupp (Fin N) Nat
        hd : Ne (MvPolynomial.coeff d (((MvPolynomial.finSuccEquiv R N) F).coeff n)) 0
        i : Fin N
        hi : Not (Membership.mem d.support i)
        ⊢ Eq (d i) 0
      -/
    · simpa using hi
      /-
        🎉 no goals
      -/


private
lemma exists_eval_ne_zero_of_totalDegree_le_card_aux {N : ℕ} {F : MvPolynomial (Fin N) R} {n : ℕ}
    (hF : F.IsHomogeneous n) (hF₀ : F ≠ 0) (hnR : n ≤ #R) :
    ∃ r, eval r F ≠ 0 := by
  induction N generalizing n with
  | zero =>
    use 0
    contrapose! hF₀
    ext d
    simpa only [Subsingleton.elim d 0, eval_zero, coeff_zero] using hF₀
  | succ N IH =>
    have hdeg : natDegree (finSuccEquiv R N F) < n + 1 := by
      linarith [natDegree_finSuccEquiv F, degreeOf_le_totalDegree F 0, hF.totalDegree hF₀]
    obtain ⟨i, hi⟩ : ∃ i : ℕ, (finSuccEquiv R N F).coeff i ≠ 0 := by
      contrapose! hF₀
      exact (finSuccEquiv _ _).injective <| Polynomial.ext <| by simpa using hF₀
    have hin : i ≤ n := by
      contrapose! hi
      exact coeff_eq_zero_of_natDegree_lt <| (Nat.le_of_lt_succ hdeg).trans_lt hi
    obtain hFn | hFn := ne_or_eq ((finSuccEquiv R N F).coeff n) 0
    · exact hF.exists_eval_ne_zero_of_coeff_finSuccEquiv_ne_zero_aux hFn
    have hin : i < n := hin.lt_or_eq.elim id <| by aesop
    obtain ⟨j, hj⟩ : ∃ j, i + (j + 1) = n := (Nat.exists_eq_add_of_lt hin).imp <| by omega
    obtain ⟨r, hr⟩ : ∃ r, (eval r) (Polynomial.coeff ((finSuccEquiv R N) F) i) ≠ 0 :=
      IH (hF.finSuccEquiv_coeff_isHomogeneous _ _ hj) hi (.trans (by norm_cast; omega) hnR)
    set φ : R[X] := Polynomial.map (eval r) (finSuccEquiv _ _ F) with hφ
    have hφ₀ : φ ≠ 0 := fun hφ₀ ↦ hr <| by
      rw [← coeff_eval_eq_eval_coeff, ← hφ, hφ₀, Polynomial.coeff_zero]
    have hφR : φ.natDegree < #R := by
      refine lt_of_lt_of_le ?_ hnR
      norm_cast
      refine lt_of_le_of_lt natDegree_map_le ?_
      suffices (finSuccEquiv _ _ F).natDegree ≠ n by omega
      rintro rfl
      refine leadingCoeff_ne_zero.mpr ?_ hFn
      simpa using (finSuccEquiv R N).injective.ne hF₀
    obtain ⟨r₀, hr₀⟩ : ∃ r₀, Polynomial.eval r₀ φ ≠ 0 :=
      φ.exists_eval_ne_zero_of_natDegree_lt_card hφ₀ hφR
    use Fin.cons r₀ r
    rwa [eval_eq_eval_mv_eval']


/-- See `MvPolynomial.IsHomogeneous.eq_zero_of_forall_eval_eq_zero`
for a version that assumes `Infinite R`. -/
lemma eq_zero_of_forall_eval_eq_zero_of_le_card
    (hF : F.IsHomogeneous n) (h : ∀ r : σ → R, eval r F = 0) (hnR : n ≤ #R) :
    F = 0 := by
  /-
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    F : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) 0
    hnR : LE.le (↑n) (Cardinal.mk R)
    ⊢ Eq F 0
  -/
  contrapose! h
  -- reduce to the case where σ is finite
  /-
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    F : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hnR : LE.le (↑n) (Cardinal.mk R)
    h : Ne F 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) F) 0
  -/
  obtain ⟨k, f, hf, F, rfl⟩ := exists_fin_rename F
  /-
    case intro.intro.intro.intro
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  have hF₀ : F ≠ 0 := by rintro rfl; simp at h
  /-
    case intro.intro.intro.intro
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    hF₀ : Ne F 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  have hF : F.IsHomogeneous n := by rwa [rename_isHomogeneous_iff hf] at hF
  /-
    case intro.intro.intro.intro
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF✝ : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    hF₀ : Ne F 0
    hF : F.IsHomogeneous n
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  obtain ⟨r, hr⟩ := exists_eval_ne_zero_of_totalDegree_le_card_aux hF hF₀ hnR
  /-
    case intro.intro.intro.intro.intro
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF✝ : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    hF₀ : Ne F 0
    hF : F.IsHomogeneous n
    r : Fin k → R
    hr : Ne ((MvPolynomial.eval r) F) 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  obtain ⟨r, rfl⟩ := (Function.factorsThrough_iff _).mp <| (hf.factorsThrough r)
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF✝ : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    hF₀ : Ne F 0
    hF : F.IsHomogeneous n
    r : σ → R
    hr : Ne ((MvPolynomial.eval (Function.comp r f)) F) 0
    ⊢ Exists fun r => Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  use r
  /-
    case h
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hnR : LE.le (↑n) (Cardinal.mk R)
    k : Nat
    f : Fin k → σ
    hf : Function.Injective f
    F : MvPolynomial (Fin k) R
    hF✝ : ((MvPolynomial.rename f) F).IsHomogeneous n
    h : Ne ((MvPolynomial.rename f) F) 0
    hF₀ : Ne F 0
    hF : F.IsHomogeneous n
    r : σ → R
    hr : Ne ((MvPolynomial.eval (Function.comp r f)) F) 0
    ⊢ Ne ((MvPolynomial.eval r) ((MvPolynomial.rename f) F)) 0
  -/
  rwa [eval_rename]
  /-
    🎉 no goals
  -/


/-- See `MvPolynomial.IsHomogeneous.funext`
for a version that assumes `Infinite R`. -/
lemma funext_of_le_card (hF : F.IsHomogeneous n) (hG : G.IsHomogeneous n)
    (h : ∀ r : σ → R, eval r F = eval r G) (hnR : n ≤ #R) :
    F = G := by
  /-
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    F G : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hG : G.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) ((MvPolynomial.eval r) G)
    hnR : LE.le (↑n) (Cardinal.mk R)
    ⊢ Eq F G
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    F G : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hG : G.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) ((MvPolynomial.eval r) G)
    hnR : LE.le (↑n) (Cardinal.mk R)
    ⊢ Eq (HSub.hSub F G) 0
  -/
  apply eq_zero_of_forall_eval_eq_zero_of_le_card (hF.sub hG) _ hnR
  /-
    R : Type u_5
    σ : Type u_6
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    F G : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hG : G.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) ((MvPolynomial.eval r) G)
    hnR : LE.le (↑n) (Cardinal.mk R)
    ⊢ ∀ (r : σ → R), Eq ((MvPolynomial.eval r) (HSub.hSub F G)) 0
  -/
  simpa [sub_eq_zero] using h
  /-
    🎉 no goals
  -/


/-- See `MvPolynomial.IsHomogeneous.eq_zero_of_forall_eval_eq_zero_of_le_card`
for a version that assumes `n ≤ #R`. -/
lemma eq_zero_of_forall_eval_eq_zero [Infinite R] {F : MvPolynomial σ R} {n : ℕ}
    (hF : F.IsHomogeneous n) (h : ∀ r : σ → R, eval r F = 0) : F = 0 := by
  /-
    R : Type u_5
    σ : Type u_6
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    F : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) 0
    ⊢ Eq F 0
  -/
  apply eq_zero_of_forall_eval_eq_zero_of_le_card hF h
  /-
    R : Type u_5
    σ : Type u_6
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    F : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) 0
    ⊢ LE.le (↑n) (Cardinal.mk R)
  -/
  exact (Cardinal.nat_lt_aleph0 _).le.trans <| Cardinal.infinite_iff.mp ‹Infinite R›
  /-
    🎉 no goals
  -/


/-- See `MvPolynomial.IsHomogeneous.funext_of_le_card`
for a version that assumes `n ≤ #R`. -/
lemma funext [Infinite R] {F G : MvPolynomial σ R} {n : ℕ}
    (hF : F.IsHomogeneous n) (hG : G.IsHomogeneous n)
    (h : ∀ r : σ → R, eval r F = eval r G) : F = G := by
  /-
    R : Type u_5
    σ : Type u_6
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    F G : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hG : G.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) ((MvPolynomial.eval r) G)
    ⊢ Eq F G
  -/
  apply funext_of_le_card hF hG h
  /-
    R : Type u_5
    σ : Type u_6
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    F G : MvPolynomial σ R
    n : Nat
    hF : F.IsHomogeneous n
    hG : G.IsHomogeneous n
    h : ∀ (r : σ → R), Eq ((MvPolynomial.eval r) F) ((MvPolynomial.eval r) G)
    ⊢ LE.le (↑n) (Cardinal.mk R)
  -/
  exact (Cardinal.nat_lt_aleph0 _).le.trans <| Cardinal.infinite_iff.mp ‹Infinite R›
  /-
    🎉 no goals
  -/


/-- The homogeneous submodules form a graded ring. This instance is used by `DirectSum.commSemiring`
and `DirectSum.algebra`. -/
instance HomogeneousSubmodule.gcommSemiring : SetLike.GradedMonoid (homogeneousSubmodule σ R) where
  one_mem := isHomogeneous_one σ R
  mul_mem _ _ _ _ := IsHomogeneous.mul


/-- `homogeneousComponent n φ` is the part of `φ` that is homogeneous of degree `n`.
See `sum_homogeneousComponent` for the statement that `φ` is equal to the sum
of all its homogeneous components. -/
def homogeneousComponent [CommSemiring R] (n : ℕ) : MvPolynomial σ R →ₗ[R] MvPolynomial σ R :=
  weightedHomogeneousComponent 1 n


theorem homogeneousComponent_mem  :
    homogeneousComponent n φ ∈ homogeneousSubmodule σ R n :=
  weightedHomogeneousComponent_mem _ φ n


theorem coeff_homogeneousComponent (d : σ →₀ ℕ) :
    coeff d (homogeneousComponent n φ) = if d.degree = n then coeff d φ else 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d ((MvPolynomial.homogeneousComponent n) φ)) (ite (Eq …
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d ((MvPolynomial.homogeneousComponent n) φ)) (ite (Eq …
  -/
  convert coeff_weightedHomogeneousComponent n φ d
  /-
    🎉 no goals
  -/


theorem homogeneousComponent_apply :
    homogeneousComponent n φ = ∑ d ∈ φ.support with d.degree = n, monomial d (coeff d φ) := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.homogeneousComponent n) φ) ((Finset.filter (fun d => Eq d. …
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.homogeneousComponent n) φ) ((Finset.filter (fun d => Eq (( …
  -/
  convert weightedHomogeneousComponent_apply n φ
  /-
    🎉 no goals
  -/


theorem homogeneousComponent_isHomogeneous : (homogeneousComponent n φ).IsHomogeneous n :=
  weightedHomogeneousComponent_isWeightedHomogeneous n φ


@[simp]
theorem homogeneousComponent_zero : homogeneousComponent 0 φ = C (coeff 0 φ) :=
  weightedHomogeneousComponent_zero φ (fun _ => Nat.succ_ne_zero Nat.zero)


@[simp]
theorem homogeneousComponent_C_mul (n : ℕ) (r : R) :
    homogeneousComponent n (C r * φ) = C r * homogeneousComponent n φ :=
  weightedHomogeneousComponent_C_mul φ n r


theorem homogeneousComponent_eq_zero'
    (h : ∀ d : σ →₀ ℕ, d ∈ φ.support → d.degree ≠ n) :
    homogeneousComponent n φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Membership.mem φ.support d → Ne d.degree n
    ⊢ Eq ((MvPolynomial.homogeneousComponent n) φ) 0
  -/
  simp_rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Membership.mem φ.support d → Ne ((Finsupp.weight 1) …
    ⊢ Eq ((MvPolynomial.homogeneousComponent n) φ) 0
  -/
  exact weightedHomogeneousComponent_eq_zero' n φ h
  /-
    🎉 no goals
  -/


theorem homogeneousComponent_eq_zero (h : φ.totalDegree < n) : homogeneousComponent n φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    h : LT.lt φ.totalDegree n
    ⊢ Eq ((MvPolynomial.homogeneousComponent n) φ) 0
  -/
  apply homogeneousComponent_eq_zero'
  /-
    case h
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    h : LT.lt φ.totalDegree n
    ⊢ ∀ (d : Finsupp σ Nat), Membership.mem φ.support d → Ne d.degree n
  -/
  rw [totalDegree, Finset.sup_lt_iff (lt_of_le_of_lt (Nat.zero_le _) h)] at h
  /-
    case h
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    n : Nat
    φ : MvPolynomial σ R
    h : ∀ (b : Finsupp σ Nat), Membership.mem φ.support b → LT.lt (b.sum fun x e = …
    ⊢ ∀ (d : Finsupp σ Nat), Membership.mem φ.support d → Ne d.degree n
  -/
  intro d hd; exact ne_of_lt (h d hd)
              /-
                🎉 no goals
              -/


theorem sum_homogeneousComponent :
    (∑ i ∈ range (φ.totalDegree + 1), homogeneousComponent i φ) = φ := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    ⊢ Eq ((Finset.range (HAdd.hAdd φ.totalDegree 1)).sum fun i => (MvPolynomial.ho …
  -/
  ext1 d
  suffices φ.totalDegree < d.support.sum d → 0 = coeff d φ by
    simpa [coeff_sum, coeff_homogeneousComponent]
  /-
    case a
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ LT.lt φ.totalDegree (d.support.sum ⇑d) → Eq 0 (MvPolynomial.coeff d φ)
  -/
  exact fun h => (coeff_eq_zero_of_totalDegree_lt h).symm
  /-
    🎉 no goals
  -/


theorem homogeneousComponent_of_mem {m n : ℕ} {p : MvPolynomial σ R}
    (h : p ∈ homogeneousSubmodule σ R n) :
    homogeneousComponent m p = if m = n then p else 0 :=
  weightedHomogeneousComponent_of_mem h


/-- The homogeneous submodules form a graded ring.
This instance is used by `DirectSum.commSemiring` and `DirectSum.algebra`. -/
lemma HomogeneousSubmodule.gradedMonoid :
    SetLike.GradedMonoid (homogeneousSubmodule σ R) :=
  WeightedHomogeneousSubmodule.gradedMonoid


/-- The decomposition of `MvPolynomial σ R` into homogeneous submodules. -/
abbrev decomposition :
    DirectSum.Decomposition (homogeneousSubmodule σ R) :=
  weightedDecomposition R (1 : σ → ℕ)


/-- `MvPolynomial σ R` as a graded algebra, graded by the degree.
We do not make this a global instance because one may want to consider a different
graded algebra structure on `MvPolynomial σ R`, induced by another weight function.
To make it a local instance, you may use
`attribute [local instance] MvPolynomial.gradedAlgebra`.
-/
abbrev gradedAlgebra : GradedAlgebra (homogeneousSubmodule σ R) :=
  weightedGradedAlgebra R (1 : σ → ℕ)


theorem decomposition.decompose'_apply (φ : MvPolynomial σ R) (i : ℕ) :
    (decomposition.decompose' φ i : MvPolynomial σ R) = homogeneousComponent i φ :=
  weightedDecomposition.decompose'_apply R _ φ i


theorem decomposition.decompose'_eq :
    decomposition.decompose' = fun φ : MvPolynomial σ R =>
      DirectSum.mk (fun i : ℕ => ↥(homogeneousSubmodule σ R i)) (φ.support.image Finsupp.degree)
        fun m => ⟨homogeneousComponent m φ, homogeneousComponent_mem m φ⟩ := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    ⊢ Eq DirectSum.Decomposition.decompose' fun φ => (DirectSum.mk (fun i => Subty …
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    ⊢ Eq DirectSum.Decomposition.decompose' fun φ => (DirectSum.mk (fun i => Subty …
  -/
  rfl
  /-
    🎉 no goals
  -/


