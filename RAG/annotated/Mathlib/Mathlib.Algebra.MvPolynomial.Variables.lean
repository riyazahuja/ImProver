/-- `vars p` is the set of variables appearing in the polynomial `p` -/
def vars (p : MvPolynomial σ R) : Finset σ :=
  letI := Classical.decEq σ
  p.degrees.toFinset


theorem vars_def [DecidableEq σ] (p : MvPolynomial σ R) : p.vars = p.degrees.toFinset := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p : MvPolynomial σ R
    ⊢ Eq p.vars p.degrees.toFinset
  -/
  rw [vars]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p : MvPolynomial σ R
    ⊢ Eq p.degrees.toFinset p.degrees.toFinset
  -/
  convert rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem vars_0 : (0 : MvPolynomial σ R).vars = ∅ := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.vars 0) EmptyCollection.emptyCollection
  -/
  classical rw [vars_def, degrees_zero, Multiset.toFinset_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem vars_monomial (h : r ≠ 0) : (monomial s r).vars = s.support := by
  /-
    R : Type u
    σ : Type u_1
    r : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    h : Ne r 0
    ⊢ Eq ((MvPolynomial.monomial s) r).vars s.support
  -/
  classical rw [vars_def, degrees_monomial_eq _ _ h, Finsupp.toFinset_toMultiset]
  /-
    🎉 no goals
  -/


@[simp]
theorem vars_C : (C r : MvPolynomial σ R).vars = ∅ := by
  /-
    R : Type u
    σ : Type u_1
    r : R
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.C r).vars EmptyCollection.emptyCollection
  -/
  classical rw [vars_def, degrees_C, Multiset.toFinset_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem vars_X [Nontrivial R] : (X n : MvPolynomial σ R).vars = {n} := by
  /-
    R : Type u
    σ : Type u_1
    n : σ
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Eq (MvPolynomial.X n).vars (Singleton.singleton n)
  -/
  rw [X, vars_monomial (one_ne_zero' R), Finsupp.support_single_ne_zero _ (one_ne_zero' ℕ)]
  /-
    🎉 no goals
  -/


theorem mem_vars (i : σ) : i ∈ p.vars ↔ ∃ d ∈ p.support, i ∈ d.support := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    ⊢ Iff (Membership.mem p.vars i) (Exists fun d => And (Membership.mem p.support …
  -/
  classical simp only [vars_def, Multiset.mem_toFinset, mem_degrees, mem_support_iff, exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_support_not_mem_vars_zero {f : MvPolynomial σ R} {x : σ →₀ ℕ} (H : x ∈ f.support)
    {v : σ} (h : v ∉ vars f) : x v = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    x : Finsupp σ Nat
    H : Membership.mem f.support x
    v : σ
    h : Not (Membership.mem f.vars v)
    ⊢ Eq (x v) 0
  -/
  contrapose! h
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    x : Finsupp σ Nat
    H : Membership.mem f.support x
    v : σ
    h : Ne (x v) 0
    ⊢ Membership.mem f.vars v
  -/
  exact (mem_vars v).mpr ⟨x, H, Finsupp.mem_support_iff.mpr h⟩
  /-
    🎉 no goals
  -/


theorem vars_add_subset [DecidableEq σ] (p q : MvPolynomial σ R) :
    (p + q).vars ⊆ p.vars ∪ q.vars := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ HasSubset.Subset (HAdd.hAdd p q).vars (Union.union p.vars q.vars)
  -/
  intro x hx
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    x : σ
    hx : Membership.mem (HAdd.hAdd p q).vars x
    ⊢ Membership.mem (Union.union p.vars q.vars) x
  -/
  simp only [vars_def, Finset.mem_union, Multiset.mem_toFinset] at hx ⊢
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    x : σ
    hx : Membership.mem (HAdd.hAdd p q).degrees x
    ⊢ Or (Membership.mem p.degrees x) (Membership.mem q.degrees x)
  -/
  simpa using Multiset.mem_of_le (degrees_add _ _) hx
  /-
    🎉 no goals
  -/


theorem vars_add_of_disjoint [DecidableEq σ] (h : Disjoint p.vars q.vars) :
    (p + q).vars = p.vars ∪ q.vars := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    h : Disjoint p.vars q.vars
    ⊢ Eq (HAdd.hAdd p q).vars (Union.union p.vars q.vars)
  -/
  refine (vars_add_subset p q).antisymm fun x hx => ?_
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    h : Disjoint p.vars q.vars
    x : σ
    hx : Membership.mem (Union.union p.vars q.vars) x
    ⊢ Membership.mem (HAdd.hAdd p q).vars x
  -/
  simp only [vars_def, Multiset.disjoint_toFinset] at h hx ⊢
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    x : σ
    h : Disjoint p.degrees q.degrees
    hx : Membership.mem (Union.union p.degrees.toFinset q.degrees.toFinset) x
    ⊢ Membership.mem (HAdd.hAdd p q).degrees.toFinset x
  -/
  rwa [degrees_add_of_disjoint h, Multiset.toFinset_union]
  /-
    🎉 no goals
  -/


theorem vars_mul [DecidableEq σ] (φ ψ : MvPolynomial σ R) : (φ * ψ).vars ⊆ φ.vars ∪ ψ.vars := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    φ ψ : MvPolynomial σ R
    ⊢ HasSubset.Subset (HMul.hMul φ ψ).vars (Union.union φ.vars ψ.vars)
  -/
  simp_rw [vars_def, ← Multiset.toFinset_add, Multiset.toFinset_subset]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    φ ψ : MvPolynomial σ R
    ⊢ HasSubset.Subset (HMul.hMul φ ψ).degrees (HAdd.hAdd φ.degrees ψ.degrees)
  -/
  exact Multiset.subset_of_le (degrees_mul φ ψ)
  /-
    🎉 no goals
  -/


@[simp]
theorem vars_one : (1 : MvPolynomial σ R).vars = ∅ :=
  vars_C


theorem vars_pow (φ : MvPolynomial σ R) (n : ℕ) : (φ ^ n).vars ⊆ φ.vars := by
  classical
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ']
    apply Finset.Subset.trans (vars_mul _ _)
    exact Finset.union_subset (Finset.Subset.refl _) ih


/-- The variables of the product of a family of polynomials
are a subset of the union of the sets of variables of each polynomial.
-/
theorem vars_prod {ι : Type*} [DecidableEq σ] {s : Finset ι} (f : ι → MvPolynomial σ R) :
    (∏ i ∈ s, f i).vars ⊆ s.biUnion fun i => (f i).vars := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert hs hsub =>
    simp only [hs, Finset.biUnion_insert, Finset.prod_insert, not_false_iff]
    apply Finset.Subset.trans (vars_mul _ _)
    exact Finset.union_subset_union (Finset.Subset.refl _) hsub


theorem vars_C_mul (a : A) (ha : a ≠ 0) (φ : MvPolynomial σ A) :
    (C a * φ : MvPolynomial σ A).vars = φ.vars := by
  /-
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    ⊢ Eq (HMul.hMul (MvPolynomial.C a) φ).vars φ.vars
  -/
  ext1 i
  /-
    case h
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    i : σ
    ⊢ Iff (Membership.mem (HMul.hMul (MvPolynomial.C a) φ).vars i) (Membership.mem …
  -/
  simp only [mem_vars, exists_prop, mem_support_iff]
  /-
    case h
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    i : σ
    ⊢ Iff (Exists fun d => And (Ne (MvPolynomial.coeff d (HMul.hMul (MvPolynomial. …
  -/
  apply exists_congr
  /-
    case h.h
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    i : σ
    ⊢ ∀ (a_1 : Finsupp σ Nat), Iff (And (Ne (MvPolynomial.coeff a_1 (HMul.hMul (Mv …
  -/
  intro d
  /-
    case h.h
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    i : σ
    d : Finsupp σ Nat
    ⊢ Iff (And (Ne (MvPolynomial.coeff d (HMul.hMul (MvPolynomial.C a) φ)) 0) (Mem …
  -/
  apply and_congr _ Iff.rfl
  /-
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : NoZeroDivisors A
    a : A
    ha : Ne a 0
    φ : MvPolynomial σ A
    i : σ
    d : Finsupp σ Nat
    ⊢ Iff (Ne (MvPolynomial.coeff d (HMul.hMul (MvPolynomial.C a) φ)) 0) (Ne (MvPo …
  -/
  rw [coeff_C_mul, mul_ne_zero_iff, eq_true ha, true_and]
  /-
    🎉 no goals
  -/


theorem vars_sum_subset [DecidableEq σ] :
    (∑ i ∈ t, φ i).vars ⊆ Finset.biUnion t fun i => (φ i).vars := by
  classical
  induction t using Finset.induction_on with
  | empty => simp
  | insert has hsum =>
    rw [Finset.biUnion_insert, Finset.sum_insert has]
    refine Finset.Subset.trans
      (vars_add_subset _ _) (Finset.union_subset_union (Finset.Subset.refl _) ?_)
    assumption


theorem vars_sum_of_disjoint [DecidableEq σ] (h : Pairwise <| (Disjoint on fun i => (φ i).vars)) :
    (∑ i ∈ t, φ i).vars = Finset.biUnion t fun i => (φ i).vars := by
  classical
  induction t using Finset.induction_on with
  | empty => simp
  | insert has hsum =>
    rw [Finset.biUnion_insert, Finset.sum_insert has, vars_add_of_disjoint, hsum]
    unfold Pairwise onFun at h
    rw [hsum]
    simp only [Finset.disjoint_iff_ne] at h ⊢
    intro v hv v2 hv2
    rw [Finset.mem_biUnion] at hv2
    rcases hv2 with ⟨i, his, hi⟩
    refine h ?_ _ hv _ hi
    rintro rfl
    contradiction


                                                 /-
                                                   R : Type u
                                                   S : Type v
                                                   σ : Type u_1
                                                   inst✝¹ : CommSemiring R
                                                   p : MvPolynomial σ R
                                                   inst✝ : CommSemiring S
                                                   f : RingHom R S
                                                   ⊢ HasSubset.Subset ((MvPolynomial.map f) p).vars p.vars
                                                 -/
theorem vars_map : (map f p).vars ⊆ p.vars := by classical simp [vars_def, degrees_map]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem vars_map_of_injective (hf : Injective f) : (map f p).vars = p.vars := by
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    ⊢ Eq ((MvPolynomial.map f) p).vars p.vars
  -/
  simp [vars, degrees_map_of_injective _ hf]
  /-
    🎉 no goals
  -/


theorem vars_monomial_single (i : σ) {e : ℕ} {r : R} (he : e ≠ 0) (hr : r ≠ 0) :
    (monomial (Finsupp.single i e) r).vars = {i} := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    e : Nat
    r : R
    he : Ne e 0
    hr : Ne r 0
    ⊢ Eq ((MvPolynomial.monomial (Finsupp.single i e)) r).vars (Singleton.singleto …
  -/
  rw [vars_monomial hr, Finsupp.support_single_ne_zero _ he]
  /-
    🎉 no goals
  -/


theorem vars_eq_support_biUnion_support [DecidableEq σ] :
    p.vars = p.support.biUnion Finsupp.support := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : DecidableEq σ
    ⊢ Eq p.vars (p.support.biUnion Finsupp.support)
  -/
  ext i
  /-
    case h
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : DecidableEq σ
    i : σ
    ⊢ Iff (Membership.mem p.vars i) (Membership.mem (p.support.biUnion Finsupp.sup …
  -/
  rw [mem_vars, Finset.mem_biUnion]
  /-
    🎉 no goals
  -/


theorem eval₂Hom_eq_constantCoeff_of_vars (f : R →+* S) {g : σ → S} {p : MvPolynomial σ R}
    (hp : ∀ i ∈ p.vars, g i = 0) : eval₂Hom f g p = f (constantCoeff p) := by
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → S
    p : MvPolynomial σ R
    hp : ∀ (i : σ), Membership.mem p.vars i → Eq (g i) 0
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) p) (f (MvPolynomial.constantCoeff p))
  -/
  conv_lhs => rw [p.as_sum]
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → S
    p : MvPolynomial σ R
    hp : ∀ (i : σ), Membership.mem p.vars i → Eq (g i) 0
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) (p.support.sum fun v => (MvPolynomial.monomi …
  -/
  simp only [map_sum, eval₂Hom_monomial]
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → S
    p : MvPolynomial σ R
    hp : ∀ (i : σ), Membership.mem p.vars i → Eq (g i) 0
    ⊢ Eq (p.support.sum fun x => HMul.hMul (f (MvPolynomial.coeff x p)) (x.prod fu …
  -/
  by_cases h0 : constantCoeff p = 0
  on_goal 1 =>
    rw [h0, f.map_zero, Finset.sum_eq_zero]
    intro d hd
  on_goal 2 =>
    rw [Finset.sum_eq_single (0 : σ →₀ ℕ)]
    · rw [Finsupp.prod_zero_index, mul_one]
      rfl
    on_goal 1 => intro d hd hd0
  on_goal 3 =>
    rw [constantCoeff_eq, coeff, ← Ne, ← Finsupp.mem_support_iff] at h0
    intro
    contradiction
  repeat'
    obtain ⟨i, hi⟩ : Finset.Nonempty (Finsupp.support d) := by
      rw [constantCoeff_eq, coeff, ← Finsupp.not_mem_support_iff] at h0
      rw [Finset.nonempty_iff_ne_empty, Ne, Finsupp.support_eq_empty]
      rintro rfl
      contradiction
    rw [Finsupp.prod, Finset.prod_eq_zero hi, mul_zero]
    rw [hp, zero_pow (Finsupp.mem_support_iff.1 hi)]
    rw [mem_vars]
    exact ⟨d, hd, hi⟩


theorem aeval_eq_constantCoeff_of_vars [Algebra R S] {g : σ → S} {p : MvPolynomial σ R}
    (hp : ∀ i ∈ p.vars, g i = 0) : aeval g p = algebraMap _ _ (constantCoeff p) :=
  eval₂Hom_eq_constantCoeff_of_vars _ hp


theorem eval₂Hom_congr' {f₁ f₂ : R →+* S} {g₁ g₂ : σ → S} {p₁ p₂ : MvPolynomial σ R} :
    f₁ = f₂ →
      (∀ i, i ∈ p₁.vars → i ∈ p₂.vars → g₁ i = g₂ i) →
        p₁ = p₂ → eval₂Hom f₁ g₁ p₁ = eval₂Hom f₂ g₂ p₂ := by
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ f₂ : RingHom R S
    g₁ g₂ : σ → S
    p₁ p₂ : MvPolynomial σ R
    ⊢ Eq f₁ f₂ → (∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₂.vars i → …
  -/
  rintro rfl h rfl
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    ⊢ Eq ((MvPolynomial.eval₂Hom f₁ g₁) p₁) ((MvPolynomial.eval₂Hom f₁ g₂) p₁)
  -/
  rw [p₁.as_sum]
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    ⊢ Eq ((MvPolynomial.eval₂Hom f₁ g₁) (p₁.support.sum fun v => (MvPolynomial.mon …
  -/
  simp only [map_sum, eval₂Hom_monomial]
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    ⊢ Eq (p₁.support.sum fun x => HMul.hMul (f₁ (MvPolynomial.coeff x p₁)) (x.prod …
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    ⊢ ∀ (x : Finsupp σ Nat), Membership.mem p₁.support x → Eq (HMul.hMul (f₁ (MvPo …
  -/
  intro d hd
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    d : Finsupp σ Nat
    hd : Membership.mem p₁.support d
    ⊢ Eq (HMul.hMul (f₁ (MvPolynomial.coeff d p₁)) (d.prod fun i k => HPow.hPow (g …
  -/
  congr 1
  /-
    case e_a
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    d : Finsupp σ Nat
    hd : Membership.mem p₁.support d
    ⊢ Eq (d.prod fun i k => HPow.hPow (g₁ i) k) (d.prod fun i k => HPow.hPow (g₂ i …
  -/
  simp only [Finsupp.prod]
  /-
    case e_a
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    d : Finsupp σ Nat
    hd : Membership.mem p₁.support d
    ⊢ Eq (d.support.prod fun x => HPow.hPow (g₁ x) (d x)) (d.support.prod fun x => …
  -/
  apply Finset.prod_congr rfl
  /-
    case e_a
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    d : Finsupp σ Nat
    hd : Membership.mem p₁.support d
    ⊢ ∀ (x : σ), Membership.mem d.support x → Eq (HPow.hPow (g₁ x) (d x)) (HPow.hP …
  -/
  intro i hi
  have : i ∈ p₁.vars := by
    rw [mem_vars]
    exact ⟨d, hd, hi⟩
  /-
    case e_a
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f₁ : RingHom R S
    g₁ g₂ : σ → S
    p₁ : MvPolynomial σ R
    h : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₁.vars i → Eq (g₁ i) …
    d : Finsupp σ Nat
    hd : Membership.mem p₁.support d
    i : σ
    hi : Membership.mem d.support i
    this : Membership.mem p₁.vars i
    ⊢ Eq (HPow.hPow (g₁ i) (d i)) (HPow.hPow (g₂ i) (d i))
  -/
  rw [h i this this]
  /-
    🎉 no goals
  -/


/-- If `f₁` and `f₂` are ring homs out of the polynomial ring and `p₁` and `p₂` are polynomials,
  then `f₁ p₁ = f₂ p₂` if `p₁ = p₂` and `f₁` and `f₂` are equal on `R` and on the variables
  of `p₁`. -/
theorem hom_congr_vars {f₁ f₂ : MvPolynomial σ R →+* S} {p₁ p₂ : MvPolynomial σ R}
    (hC : f₁.comp C = f₂.comp C) (hv : ∀ i, i ∈ p₁.vars → i ∈ p₂.vars → f₁ (X i) = f₂ (X i))
    (hp : p₁ = p₂) : f₁ p₁ = f₂ p₂ :=
  calc
                                                                      /-
                                                                        R : Type u
                                                                        S : Type v
                                                                        σ : Type u_1
                                                                        inst✝¹ : CommSemiring R
                                                                        inst✝ : CommSemiring S
                                                                        f₁ f₂ : RingHom (MvPolynomial σ R) S
                                                                        p₁ p₂ : MvPolynomial σ R
                                                                        hC : Eq (f₁.comp MvPolynomial.C) (f₂.comp MvPolynomial.C)
                                                                        hv : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₂.vars i → Eq (f₁ ( …
                                                                        hp : Eq p₁ p₂
                                                                        ⊢ Eq f₁ (MvPolynomial.eval₂Hom (f₁.comp MvPolynomial.C) (Function.comp (⇑f₁) M …
                                                                      -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    f₁ p₁ = eval₂Hom (f₁.comp C) (f₁ ∘ X) p₁ := RingHom.congr_fun (by ext <;> simp) _
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    _ = eval₂Hom (f₂.comp C) (f₂ ∘ X) p₂ := eval₂Hom_congr' hC hv hp
                                       /-
                                         R : Type u
                                         S : Type v
                                         σ : Type u_1
                                         inst✝¹ : CommSemiring R
                                         inst✝ : CommSemiring S
                                         f₁ f₂ : RingHom (MvPolynomial σ R) S
                                         p₁ p₂ : MvPolynomial σ R
                                         hC : Eq (f₁.comp MvPolynomial.C) (f₂.comp MvPolynomial.C)
                                         hv : ∀ (i : σ), Membership.mem p₁.vars i → Membership.mem p₂.vars i → Eq (f₁ ( …
                                         hp : Eq p₁ p₂
                                         ⊢ Eq (MvPolynomial.eval₂Hom (f₂.comp MvPolynomial.C) (Function.comp (⇑f₂) MvPo …
                                       -/
                                               /-
                                                 🎉 no goals
                                               -/
    _ = f₂ p₂ := RingHom.congr_fun (by ext <;> simp) _
                                               /-
                                                 🎉 no goals
                                               -/


theorem exists_rename_eq_of_vars_subset_range (p : MvPolynomial σ R) (f : τ → σ) (hfi : Injective f)
    (hf : ↑p.vars ⊆ Set.range f) : ∃ q : MvPolynomial τ R, rename f q = p :=
  ⟨aeval (fun i : σ => Option.elim' 0 X <| partialInv f i) p,
    by
      /-
        R : Type u
        σ : Type u_1
        τ : Type u_2
        inst✝ : CommSemiring R
        p : MvPolynomial σ R
        f : τ → σ
        hfi : Function.Injective f
        hf : HasSubset.Subset (↑p.vars) (Set.range f)
        ⊢ Eq ((MvPolynomial.rename f) ((MvPolynomial.aeval fun i => Option.elim' 0 MvP …
      -/
      show (rename f).toRingHom.comp _ p = RingHom.id _ p
      /-
        R : Type u
        σ : Type u_1
        τ : Type u_2
        inst✝ : CommSemiring R
        p : MvPolynomial σ R
        f : τ → σ
        hfi : Function.Injective f
        hf : HasSubset.Subset (↑p.vars) (Set.range f)
        ⊢ Eq (((MvPolynomial.rename f).comp (MvPolynomial.aeval fun i => Option.elim'  …
      -/
      refine hom_congr_vars ?_ ?_ ?_
        /-
          case refine_1
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          ⊢ Eq (((MvPolynomial.rename f).comp (MvPolynomial.aeval fun i => Option.elim'  …
        -/
      · ext1
        /-
          case refine_1.a
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          x✝ : R
          ⊢ Eq ((((MvPolynomial.rename f).comp (MvPolynomial.aeval fun i => Option.elim' …
        -/
        simp [algebraMap_eq]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          ⊢ ∀ (i : σ), Membership.mem p.vars i → Membership.mem p.vars i → Eq (((MvPolyn …
        -/
      · intro i hip _
        /-
          case refine_2
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          i : σ
          hip a✝ : Membership.mem p.vars i
          ⊢ Eq (((MvPolynomial.rename f).comp (MvPolynomial.aeval fun i => Option.elim'  …
        -/
        rcases hf hip with ⟨i, rfl⟩
        /-
          case refine_2.intro
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          i : τ
          hip a✝ : Membership.mem p.vars (f i)
          ⊢ Eq (((MvPolynomial.rename f).comp (MvPolynomial.aeval fun i => Option.elim'  …
        -/
        simp [partialInv_left hfi]
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          R : Type u
          σ : Type u_1
          τ : Type u_2
          inst✝ : CommSemiring R
          p : MvPolynomial σ R
          f : τ → σ
          hfi : Function.Injective f
          hf : HasSubset.Subset (↑p.vars) (Set.range f)
          ⊢ Eq p p
        -/
      · rfl⟩
        /-
          🎉 no goals
        -/


theorem vars_rename [DecidableEq τ] (f : σ → τ) (φ : MvPolynomial σ R) :
    (rename f φ).vars ⊆ φ.vars.image f := by
  classical
  intro i hi
  simp only [vars_def, exists_prop, Multiset.mem_toFinset, Finset.mem_image] at hi ⊢
  simpa only [Multiset.mem_map] using degrees_rename _ _ hi


theorem mem_vars_rename (f : σ → τ) (φ : MvPolynomial σ R) {j : τ} (h : j ∈ (rename f φ).vars) :
    ∃ i : σ, i ∈ φ.vars ∧ f i = j := by
  classical
  simpa only [exists_prop, Finset.mem_image] using vars_rename f φ h


lemma aeval_ite_mem_eq_self (q : MvPolynomial σ R) {s : Set σ} (hs : q.vars.toSet ⊆ s)
    [∀ i, Decidable (i ∈ s)] :
    MvPolynomial.aeval (fun i ↦ if i ∈ s then .X i else 0) q = q := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    q : MvPolynomial σ R
    s : Set σ
    hs : HasSubset.Subset (↑q.vars) s
    inst✝ : (i : σ) → Decidable (Membership.mem s i)
    ⊢ Eq ((MvPolynomial.aeval fun i => ite (Membership.mem s i) (MvPolynomial.X i) …
  -/
  rw [MvPolynomial.as_sum q, MvPolynomial.aeval_sum]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    q : MvPolynomial σ R
    s : Set σ
    hs : HasSubset.Subset (↑q.vars) s
    inst✝ : (i : σ) → Decidable (Membership.mem s i)
    ⊢ Eq (q.support.sum fun i => (MvPolynomial.aeval fun i => ite (Membership.mem  …
  -/
  refine Finset.sum_congr rfl fun u hu ↦ ?_
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    q : MvPolynomial σ R
    s : Set σ
    hs : HasSubset.Subset (↑q.vars) s
    inst✝ : (i : σ) → Decidable (Membership.mem s i)
    u : Finsupp σ Nat
    hu : Membership.mem q.support u
    ⊢ Eq ((MvPolynomial.aeval fun i => ite (Membership.mem s i) (MvPolynomial.X i) …
  -/
  rw [MvPolynomial.aeval_monomial, MvPolynomial.monomial_eq]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    q : MvPolynomial σ R
    s : Set σ
    hs : HasSubset.Subset (↑q.vars) s
    inst✝ : (i : σ) → Decidable (Membership.mem s i)
    u : Finsupp σ Nat
    hu : Membership.mem q.support u
    ⊢ Eq (HMul.hMul ((algebraMap R (MvPolynomial σ R)) (MvPolynomial.coeff u q)) ( …
  -/
  congr 1
  /-
    case e_a
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    q : MvPolynomial σ R
    s : Set σ
    hs : HasSubset.Subset (↑q.vars) s
    inst✝ : (i : σ) → Decidable (Membership.mem s i)
    u : Finsupp σ Nat
    hu : Membership.mem q.support u
    ⊢ Eq (u.prod fun i k => HPow.hPow (ite (Membership.mem s i) (MvPolynomial.X i) …
  -/
  exact Finsupp.prod_congr (fun i hi ↦ by simp [hs ((MvPolynomial.mem_vars _).mpr ⟨u, hu, hi⟩)])
  /-
    🎉 no goals
  -/


lemma leadingCoeff_toLex : p.leadingCoeff toLex = p.coeff (ofLex <| p.supDegree toLex) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : LinearOrder σ
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff (⇑toLex) p) (MvPolynomial.coeff (ofLex (Ad …
  -/
  rw [leadingCoeff]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : LinearOrder σ
    ⊢ Eq (p (Function.invFun (⇑toLex) (AddMonoidAlgebra.supDegree (⇑toLex) p))) (M …
  -/
  apply congr_arg p.coeff
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : LinearOrder σ
    ⊢ Eq (Function.invFun (⇑toLex) (AddMonoidAlgebra.supDegree (⇑toLex) p)) (ofLex …
  -/
  apply toLex.injective
  /-
    case a
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    inst✝ : LinearOrder σ
    ⊢ Eq (toLex (Function.invFun (⇑toLex) (AddMonoidAlgebra.supDegree (⇑toLex) p)) …
  -/
  rw [Function.rightInverse_invFun toLex.surjective, toLex_ofLex]
  /-
    🎉 no goals
  -/


lemma supDegree_toLex_C (r : R) : supDegree toLex (C (σ := σ) r) = 0 := by
  classical
    exact (supDegree_single _ r).trans (ite_eq_iff'.mpr ⟨fun _ => rfl, fun _ => rfl⟩)


lemma leadingCoeff_toLex_C (r : R) : leadingCoeff toLex (C (σ := σ) r) = r :=
  leadingCoeff_single toLex.injective _ r


