/-- The maximal degrees of each variable in a multi-variable polynomial, expressed as a multiset.

(For example, `degrees (x^2 * y + y^3)` would be `{x, x, y, y, y}`.)
-/
def degrees (p : MvPolynomial σ R) : Multiset σ :=
  letI := Classical.decEq σ
  p.support.sup fun s : σ →₀ ℕ => toMultiset s


theorem degrees_def [DecidableEq σ] (p : MvPolynomial σ R) :
                                                                           /-
                                                                             R : Type u
                                                                             σ : Type u_1
                                                                             inst✝¹ : CommSemiring R
                                                                             inst✝ : DecidableEq σ
                                                                             p : MvPolynomial σ R
                                                                             ⊢ Eq p.degrees (p.support.sup fun s => Finsupp.toMultiset s)
                                                                           -/
    p.degrees = p.support.sup fun s : σ →₀ ℕ => Finsupp.toMultiset s := by rw [degrees]; convert rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem degrees_monomial (s : σ →₀ ℕ) (a : R) : degrees (monomial s a) ≤ toMultiset s := by
  classical
    refine (supDegree_single s a).trans_le ?_
    split_ifs
    exacts [bot_le, le_rfl]


theorem degrees_monomial_eq (s : σ →₀ ℕ) (a : R) (ha : a ≠ 0) :
    degrees (monomial s a) = toMultiset s := by
  classical
    exact (supDegree_single s a).trans (if_neg ha)


theorem degrees_C (a : R) : degrees (C a : MvPolynomial σ R) = 0 :=
  Multiset.le_zero.1 <| degrees_monomial _ _


theorem degrees_X' (n : σ) : degrees (X n : MvPolynomial σ R) ≤ {n} :=
  le_trans (degrees_monomial _ _) <| le_of_eq <| toMultiset_single _ _


@[simp]
theorem degrees_X [Nontrivial R] (n : σ) : degrees (X n : MvPolynomial σ R) = {n} :=
  (degrees_monomial_eq _ (1 : R) one_ne_zero).trans (toMultiset_single _ _)


@[simp]
theorem degrees_zero : degrees (0 : MvPolynomial σ R) = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.degrees 0) 0
  -/
  rw [← C_0]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.C 0).degrees 0
  -/
  exact degrees_C 0
  /-
    🎉 no goals
  -/


@[simp]
theorem degrees_one : degrees (1 : MvPolynomial σ R) = 0 :=
  degrees_C 1


theorem degrees_add [DecidableEq σ] (p q : MvPolynomial σ R) :
    (p + q).degrees ≤ p.degrees ⊔ q.degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ LE.le (HAdd.hAdd p q).degrees (Max.max p.degrees q.degrees)
  -/
  simp_rw [degrees_def]; exact supDegree_add_le
                         /-
                           🎉 no goals
                         -/


theorem degrees_sum {ι : Type*} [DecidableEq σ] (s : Finset ι) (f : ι → MvPolynomial σ R) :
    (∑ i ∈ s, f i).degrees ≤ s.sup fun i => (f i).degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq σ
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (s.sum fun i => f i).degrees (s.sup fun i => (f i).degrees)
  -/
  simp_rw [degrees_def]; exact supDegree_sum_le
                         /-
                           🎉 no goals
                         -/


theorem degrees_mul (p q : MvPolynomial σ R) : (p * q).degrees ≤ p.degrees + q.degrees := by
  classical
  simp_rw [degrees_def]
  exact supDegree_mul_le (map_add _)


theorem degrees_prod {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial σ R) :
    (∏ i ∈ s, f i).degrees ≤ ∑ i ∈ s, (f i).degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (s.prod fun i => f i).degrees (s.sum fun i => (f i).degrees)
  -/
  classical exact supDegree_prod_le (map_zero _) (map_add _)
  /-
    🎉 no goals
  -/


theorem degrees_pow (p : MvPolynomial σ R) (n : ℕ) : (p ^ n).degrees ≤ n • p.degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    n : Nat
    ⊢ LE.le (HPow.hPow p n).degrees (HSMul.hSMul n p.degrees)
  -/
  simpa using degrees_prod (Finset.range n) fun _ ↦ p
  /-
    🎉 no goals
  -/


theorem mem_degrees {p : MvPolynomial σ R} {i : σ} :
    i ∈ p.degrees ↔ ∃ d, p.coeff d ≠ 0 ∧ i ∈ d.support := by
  classical
  simp only [degrees_def, Multiset.mem_sup, ← mem_support_iff, Finsupp.mem_toMultiset, exists_prop]


theorem le_degrees_add {p q : MvPolynomial σ R} (h : Disjoint p.degrees q.degrees) :
    p.degrees ≤ (p + q).degrees := by
  classical
  apply Finset.sup_le
  intro d hd
  rw [Multiset.disjoint_iff_ne] at h
  obtain rfl | h0 := eq_or_ne d 0
  · rw [toMultiset_zero]; apply Multiset.zero_le
  · refine Finset.le_sup_of_le (b := d) ?_ le_rfl
    rw [mem_support_iff, coeff_add]
    suffices q.coeff d = 0 by rwa [this, add_zero, coeff, ← Finsupp.mem_support_iff]
    rw [Ne, ← Finsupp.support_eq_empty, ← Ne, ← Finset.nonempty_iff_ne_empty] at h0
    obtain ⟨j, hj⟩ := h0
    contrapose! h
    rw [mem_support_iff] at hd
    refine ⟨j, ?_, j, ?_, rfl⟩
    all_goals rw [mem_degrees]; refine ⟨d, ?_, hj⟩; assumption


theorem degrees_add_of_disjoint [DecidableEq σ] {p q : MvPolynomial σ R}
    (h : Disjoint p.degrees q.degrees) : (p + q).degrees = p.degrees ∪ q.degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    h : Disjoint p.degrees q.degrees
    ⊢ Eq (HAdd.hAdd p q).degrees (Union.union p.degrees q.degrees)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq σ
      p q : MvPolynomial σ R
      h : Disjoint p.degrees q.degrees
      ⊢ LE.le (HAdd.hAdd p q).degrees (Union.union p.degrees q.degrees)
    -/
  · apply degrees_add
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq σ
      p q : MvPolynomial σ R
      h : Disjoint p.degrees q.degrees
      ⊢ LE.le (Union.union p.degrees q.degrees) (HAdd.hAdd p q).degrees
    -/
  · apply Multiset.union_le
      /-
        case a.h₁
        R : Type u
        σ : Type u_1
        inst✝¹ : CommSemiring R
        inst✝ : DecidableEq σ
        p q : MvPolynomial σ R
        h : Disjoint p.degrees q.degrees
        ⊢ LE.le p.degrees (HAdd.hAdd p q).degrees
      -/
    · apply le_degrees_add h
      /-
        🎉 no goals
      -/
      /-
        case a.h₂
        R : Type u
        σ : Type u_1
        inst✝¹ : CommSemiring R
        inst✝ : DecidableEq σ
        p q : MvPolynomial σ R
        h : Disjoint p.degrees q.degrees
        ⊢ LE.le q.degrees (HAdd.hAdd p q).degrees
      -/
    · rw [add_comm]
      /-
        case a.h₂
        R : Type u
        σ : Type u_1
        inst✝¹ : CommSemiring R
        inst✝ : DecidableEq σ
        p q : MvPolynomial σ R
        h : Disjoint p.degrees q.degrees
        ⊢ LE.le q.degrees (HAdd.hAdd q p).degrees
      -/
      apply le_degrees_add h.symm
      /-
        🎉 no goals
      -/


theorem degrees_map [CommSemiring S] (p : MvPolynomial σ R) (f : R →+* S) :
    (map f p).degrees ⊆ p.degrees := by
  classical
  dsimp only [degrees]
  apply Multiset.subset_of_le
  apply Finset.sup_mono
  apply MvPolynomial.support_map_subset


theorem degrees_rename (f : σ → τ) (φ : MvPolynomial σ R) :
    (rename f φ).degrees ⊆ φ.degrees.map f := by
  classical
  intro i
  rw [mem_degrees, Multiset.mem_map]
  rintro ⟨d, hd, hi⟩
  obtain ⟨x, rfl, hx⟩ := coeff_rename_ne_zero _ _ _ hd
  simp only [Finsupp.mapDomain, Finsupp.mem_support_iff] at hi
  rw [sum_apply, Finsupp.sum] at hi
  contrapose! hi
  rw [Finset.sum_eq_zero]
  intro j hj
  simp only [exists_prop, mem_degrees] at hi
  specialize hi j ⟨x, hx, hj⟩
  rw [Finsupp.single_apply, if_neg hi]


theorem degrees_map_of_injective [CommSemiring S] (p : MvPolynomial σ R) {f : R →+* S}
    (hf : Injective f) : (map f p).degrees = p.degrees := by
  /-
    R : Type u
    S : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    p : MvPolynomial σ R
    f : RingHom R S
    hf : Function.Injective ⇑f
    ⊢ Eq ((MvPolynomial.map f) p).degrees p.degrees
  -/
  simp only [degrees, MvPolynomial.support_map_of_injective _ hf]
  /-
    🎉 no goals
  -/


theorem degrees_rename_of_injective {p : MvPolynomial σ R} {f : σ → τ} (h : Function.Injective f) :
    degrees (rename f p) = (degrees p).map f := by
  classical
  simp only [degrees, Multiset.map_finset_sup p.support Finsupp.toMultiset f h,
    support_rename_of_injective h, Finset.sup_image]
  refine Finset.sup_congr rfl fun x _ => ?_
  exact (Finsupp.toMultiset_map _ _).symm


/-- `degreeOf n p` gives the highest power of X_n that appears in `p` -/
def degreeOf (n : σ) (p : MvPolynomial σ R) : ℕ :=
  letI := Classical.decEq σ
  p.degrees.count n


theorem degreeOf_def [DecidableEq σ] (n : σ) (p : MvPolynomial σ R) :
                                           /-
                                             R : Type u
                                             σ : Type u_1
                                             inst✝¹ : CommSemiring R
                                             inst✝ : DecidableEq σ
                                             n : σ
                                             p : MvPolynomial σ R
                                             ⊢ Eq (MvPolynomial.degreeOf n p) (Multiset.count n p.degrees)
                                           -/
    p.degreeOf n = p.degrees.count n := by rw [degreeOf]; convert rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem degreeOf_eq_sup (n : σ) (f : MvPolynomial σ R) :
    degreeOf n f = f.support.sup fun m => m n := by
  classical
  rw [degreeOf_def, degrees, Multiset.count_finset_sup]
  congr
  ext
  simp only [count_toMultiset]


theorem degreeOf_lt_iff {n : σ} {f : MvPolynomial σ R} {d : ℕ} (h : 0 < d) :
    degreeOf n f < d ↔ ∀ m : σ →₀ ℕ, m ∈ f.support → m n < d := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : σ
    f : MvPolynomial σ R
    d : Nat
    h : LT.lt 0 d
    ⊢ Iff (LT.lt (MvPolynomial.degreeOf n f) d) (∀ (m : Finsupp σ Nat), Membership …
  -/
  rwa [degreeOf_eq_sup, Finset.sup_lt_iff]
  /-
    🎉 no goals
  -/


lemma degreeOf_le_iff {n : σ} {f : MvPolynomial σ R} {d : ℕ} :
    degreeOf n f ≤ d ↔ ∀ m ∈ support f, m n ≤ d := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : σ
    f : MvPolynomial σ R
    d : Nat
    ⊢ Iff (LE.le (MvPolynomial.degreeOf n f) d) (∀ (m : Finsupp σ Nat), Membership …
  -/
  rw [degreeOf_eq_sup, Finset.sup_le_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem degreeOf_zero (n : σ) : degreeOf n (0 : MvPolynomial σ R) = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : σ
    ⊢ Eq (MvPolynomial.degreeOf n 0) 0
  -/
  classical simp only [degreeOf_def, degrees_zero, Multiset.count_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem degreeOf_C (a : R) (x : σ) : degreeOf x (C a : MvPolynomial σ R) = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    a : R
    x : σ
    ⊢ Eq (MvPolynomial.degreeOf x (MvPolynomial.C a)) 0
  -/
  classical simp [degreeOf_def, degrees_C]
  /-
    🎉 no goals
  -/


theorem degreeOf_X [DecidableEq σ] (i j : σ) [Nontrivial R] :
    degreeOf i (X j : MvPolynomial σ R) = if i = j then 1 else 0 := by
  classical
  by_cases c : i = j
  · simp only [c, if_true, eq_self_iff_true, degreeOf_def, degrees_X, Multiset.count_singleton]
  simp [c, if_false, degreeOf_def, degrees_X]


theorem degreeOf_add_le (n : σ) (f g : MvPolynomial σ R) :
    degreeOf n (f + g) ≤ max (degreeOf n f) (degreeOf n g) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : σ
    f g : MvPolynomial σ R
    ⊢ LE.le (MvPolynomial.degreeOf n (HAdd.hAdd f g)) (Max.max (MvPolynomial.degre …
  -/
  simp_rw [degreeOf_eq_sup]; exact supDegree_add_le
                             /-
                               🎉 no goals
                             -/


theorem monomial_le_degreeOf (i : σ) {f : MvPolynomial σ R} {m : σ →₀ ℕ} (h_m : m ∈ f.support) :
    m i ≤ degreeOf i f := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    f : MvPolynomial σ R
    m : Finsupp σ Nat
    h_m : Membership.mem f.support m
    ⊢ LE.le (m i) (MvPolynomial.degreeOf i f)
  -/
  rw [degreeOf_eq_sup i]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    f : MvPolynomial σ R
    m : Finsupp σ Nat
    h_m : Membership.mem f.support m
    ⊢ LE.le (m i) (f.support.sup fun m => m i)
  -/
  apply Finset.le_sup h_m
  /-
    🎉 no goals
  -/


lemma degreeOf_monomial_eq (s : σ →₀ ℕ) (i : σ) {a : R} (ha : a ≠ 0) :
    (monomial s a).degreeOf i = s i := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : Finsupp σ Nat
    i : σ
    a : R
    ha : Ne a 0
    ⊢ Eq (MvPolynomial.degreeOf i ((MvPolynomial.monomial s) a)) (s i)
  -/
  classical rw [degreeOf_def, degrees_monomial_eq _ _ ha, Finsupp.count_toMultiset]
  /-
    🎉 no goals
  -/

-- TODO we can prove equality with `NoZeroDivisors R`

theorem degreeOf_mul_le (i : σ) (f g : MvPolynomial σ R) :
    degreeOf i (f * g) ≤ degreeOf i f + degreeOf i g := by
  classical
  simp only [degreeOf]
  convert Multiset.count_le_of_le i (degrees_mul f g)
  rw [Multiset.count_add]


theorem degreeOf_sum_le {ι : Type*} (i : σ) (s : Finset ι) (f : ι → MvPolynomial σ R) :
    degreeOf i (∑ j ∈ s, f j) ≤ s.sup fun j => degreeOf i (f j) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    i : σ
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (MvPolynomial.degreeOf i (s.sum fun j => f j)) (s.sup fun j => MvPolyn …
  -/
  simp_rw [degreeOf_eq_sup]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    i : σ
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le ((s.sum fun j => f j).support.sup fun m => m i) (s.sup fun j => (f j). …
  -/
  exact supDegree_sum_le
  /-
    🎉 no goals
  -/

-- TODO we can prove equality with `NoZeroDivisors R`

theorem degreeOf_prod_le {ι : Type*} (i : σ) (s : Finset ι) (f : ι → MvPolynomial σ R) :
    degreeOf i (∏ j ∈ s, f j) ≤ ∑ j ∈ s, (f j).degreeOf i := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    i : σ
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (MvPolynomial.degreeOf i (s.prod fun j => f j)) (s.sum fun j => MvPoly …
  -/
  simp_rw [degreeOf_eq_sup]
  exact supDegree_prod_le (by simp only [coe_zero, Pi.zero_apply])
    (fun _ _ => by simp only [coe_add, Pi.add_apply])

-- TODO we can prove equality with `NoZeroDivisors R`

theorem degreeOf_pow_le (i : σ) (p : MvPolynomial σ R) (n : ℕ) :
    degreeOf i (p ^ n) ≤ n * degreeOf i p := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    p : MvPolynomial σ R
    n : Nat
    ⊢ LE.le (MvPolynomial.degreeOf i (HPow.hPow p n)) (HMul.hMul n (MvPolynomial.d …
  -/
  simpa using degreeOf_prod_le i (Finset.range n) (fun _ => p)
  /-
    🎉 no goals
  -/


theorem degreeOf_mul_X_of_ne {i j : σ} (f : MvPolynomial σ R) (h : i ≠ j) :
    degreeOf i (f * X j) = degreeOf i f := by
  classical
  simp only [degreeOf_eq_sup i, support_mul_X, Finset.sup_map]
  congr
  ext
  simp only [Finsupp.single, add_right_eq_self, addRightEmbedding_apply, coe_mk,
    Pi.add_apply, comp_apply, ite_eq_right_iff, Finsupp.coe_add, Pi.single_eq_of_ne h]


@[deprecated (since := "2024-12-01")] alias degreeOf_mul_X_ne := degreeOf_mul_X_of_ne


theorem degreeOf_mul_X_self (j : σ) (f : MvPolynomial σ R) :
    degreeOf j (f * X j) ≤ degreeOf j f + 1 := by
  classical
  simp only [degreeOf]
  apply (Multiset.count_le_of_le j (degrees_mul f (X j))).trans
  simp only [Multiset.count_add, add_le_add_iff_left]
  convert Multiset.count_le_of_le j <| degrees_X' j
  rw [Multiset.count_singleton_self]


@[deprecated (since := "2024-12-01")] alias degreeOf_mul_X_eq := degreeOf_mul_X_self


theorem degreeOf_mul_X_eq_degreeOf_add_one_iff (j : σ) (f : MvPolynomial σ R) :
    degreeOf j (f * X j) = degreeOf j f + 1 ↔ f ≠ 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    ⊢ Iff (Eq (MvPolynomial.degreeOf j (HMul.hMul f (MvPolynomial.X j))) (HAdd.hAd …
  -/
  refine ⟨fun h => by by_contra ha; simp [ha] at h, fun h => ?_⟩
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    ⊢ Eq (MvPolynomial.degreeOf j (HMul.hMul f (MvPolynomial.X j))) (HAdd.hAdd (Mv …
  -/
  apply Nat.le_antisymm (degreeOf_mul_X_self j f)
  have : (f.support.sup fun m ↦ m j) + 1 = (f.support.sup fun m ↦ (m j + 1)) :=
    Finset.comp_sup_eq_sup_comp_of_nonempty @Nat.succ_le_succ (support_nonempty.mpr h)
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    ⊢ LE.le (HAdd.hAdd (MvPolynomial.degreeOf j f) 1) (MvPolynomial.degreeOf j (HM …
  -/
  simp only [degreeOf_eq_sup, support_mul_X, this]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    ⊢ LE.le (f.support.sup fun m => HAdd.hAdd (m j) 1) ((Finset.map (addRightEmbed …
  -/
  apply Finset.sup_le
  /-
    case a
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    ⊢ ∀ (b : Finsupp σ Nat), Membership.mem f.support b → LE.le (HAdd.hAdd (b j) 1 …
  -/
  intro x hx
  /-
    case a
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    x : Finsupp σ Nat
    hx : Membership.mem f.support x
    ⊢ LE.le (HAdd.hAdd (x j) 1) ((Finset.map (addRightEmbedding (Finsupp.single j  …
  -/
  simp only [Finset.sup_map, bot_eq_zero', add_pos_iff, zero_lt_one, or_true, Finset.le_sup_iff]
  /-
    case a
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    x : Finsupp σ Nat
    hx : Membership.mem f.support x
    ⊢ Exists fun b => And (Membership.mem f.support b) (LE.le (HAdd.hAdd (x j) 1)  …
  -/
  use x
  /-
    case h
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    j : σ
    f : MvPolynomial σ R
    h : Ne f 0
    this : Eq (HAdd.hAdd (f.support.sup fun m => m j) 1) (f.support.sup fun m => H …
    x : Finsupp σ Nat
    hx : Membership.mem f.support x
    ⊢ And (Membership.mem f.support x) (LE.le (HAdd.hAdd (x j) 1) (Function.comp ( …
  -/
  simpa using mem_support_iff.mp hx
  /-
    🎉 no goals
  -/


theorem degreeOf_C_mul_le (p : MvPolynomial σ R) (i : σ) (c : R) :
    (C c * p).degreeOf i ≤ p.degreeOf i := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ LE.le (MvPolynomial.degreeOf i (HMul.hMul (MvPolynomial.C c) p)) (MvPolynomi …
  -/
  unfold degreeOf
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ LE.le (Multiset.count i (HMul.hMul (MvPolynomial.C c) p).degrees) (Multiset. …
  -/
  convert Multiset.count_le_of_le i <| degrees_mul (C c) p
  /-
    case h.e'_4.h.e'_4
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ Eq p.degrees (HAdd.hAdd (MvPolynomial.C c).degrees p.degrees)
  -/
  simp only [degrees_C, zero_add]
  /-
    🎉 no goals
  -/


theorem degreeOf_mul_C_le (p : MvPolynomial σ R) (i : σ) (c : R) :
    (p * C c).degreeOf i ≤ p.degreeOf i := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ LE.le (MvPolynomial.degreeOf i (HMul.hMul p (MvPolynomial.C c))) (MvPolynomi …
  -/
  unfold degreeOf
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ LE.le (Multiset.count i (HMul.hMul p (MvPolynomial.C c)).degrees) (Multiset. …
  -/
  convert Multiset.count_le_of_le i <| degrees_mul p (C c)
  /-
    case h.e'_4.h.e'_4
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    i : σ
    c : R
    ⊢ Eq p.degrees (HAdd.hAdd p.degrees (MvPolynomial.C c).degrees)
  -/
  simp only [degrees_C, add_zero]
  /-
    🎉 no goals
  -/


theorem degreeOf_rename_of_injective {p : MvPolynomial σ R} {f : σ → τ} (h : Function.Injective f)
    (i : σ) : degreeOf (f i) (rename f p) = degreeOf i p := by
  classical
  simp only [degreeOf, degrees_rename_of_injective h, Multiset.count_map_eq_count' f p.degrees h]


/-- `totalDegree p` gives the maximum |s| over the monomials X^s in `p` -/
def totalDegree (p : MvPolynomial σ R) : ℕ :=
  p.support.sup fun s => s.sum fun _ e => e


theorem totalDegree_eq (p : MvPolynomial σ R) :
    p.totalDegree = p.support.sup fun m => Multiset.card (toMultiset m) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Eq p.totalDegree (p.support.sup fun m => (Finsupp.toMultiset m).card)
  -/
  rw [totalDegree]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Eq (p.support.sup fun s => s.sum fun x e => e) (p.support.sup fun m => (Fins …
  -/
  congr; funext m
  /-
    case e_f.h
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    m : Finsupp σ Nat
    ⊢ Eq (m.sum fun x e => e) (Finsupp.toMultiset m).card
  -/
  exact (Finsupp.card_toMultiset _).symm
  /-
    🎉 no goals
  -/


theorem le_totalDegree {p : MvPolynomial σ R} {s : σ →₀ ℕ} (h : s ∈ p.support) :
    (s.sum fun _ e => e) ≤ totalDegree p :=
  Finset.le_sup (α := ℕ) (f := fun s => sum s fun _ e => e) h


theorem totalDegree_le_degrees_card (p : MvPolynomial σ R) :
    p.totalDegree ≤ Multiset.card p.degrees := by
  classical
  rw [totalDegree_eq]
  exact Finset.sup_le fun s hs => Multiset.card_le_card <| Finset.le_sup hs


theorem totalDegree_le_of_support_subset (h : p.support ⊆ q.support) :
    totalDegree p ≤ totalDegree q :=
  Finset.sup_mono h


@[simp]
theorem totalDegree_C (a : R) : (C a : MvPolynomial σ R).totalDegree = 0 :=
                                     /-
                                       R : Type u
                                       σ : Type u_1
                                       inst✝ : CommSemiring R
                                       a : R
                                       ⊢ Eq (ite (Eq a 0) Bot.bot (Finsupp.sum 0 fun x e => e)) 0
                                     -/
  (supDegree_single 0 a).trans <| by rw [sum_zero_index, bot_eq_zero', ite_self]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem totalDegree_zero : (0 : MvPolynomial σ R).totalDegree = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.totalDegree 0) 0
  -/
  rw [← C_0]; exact totalDegree_C (0 : R)
              /-
                🎉 no goals
              -/


@[simp]
theorem totalDegree_one : (1 : MvPolynomial σ R).totalDegree = 0 :=
  totalDegree_C (1 : R)


@[simp]
theorem totalDegree_X {R} [CommSemiring R] [Nontrivial R] (s : σ) :
    (X s : MvPolynomial σ R).totalDegree = 1 := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    s : σ
    ⊢ Eq (MvPolynomial.X s).totalDegree 1
  -/
  rw [totalDegree, support_X]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    s : σ
    ⊢ Eq ((Singleton.singleton (Finsupp.single s 1)).sup fun s => s.sum fun x e => …
  -/
  simp only [Finset.sup, Finsupp.sum_single_index, Finset.fold_singleton, sup_bot_eq]
  /-
    🎉 no goals
  -/


theorem totalDegree_add (a b : MvPolynomial σ R) :
    (a + b).totalDegree ≤ max a.totalDegree b.totalDegree :=
  sup_support_add_le _ _ _


theorem totalDegree_add_eq_left_of_totalDegree_lt {p q : MvPolynomial σ R}
    (h : q.totalDegree < p.totalDegree) : (p + q).totalDegree = p.totalDegree := by
  classical
    apply le_antisymm
    · rw [← max_eq_left_of_lt h]
      exact totalDegree_add p q
    by_cases hp : p = 0
    · simp [hp]
    obtain ⟨b, hb₁, hb₂⟩ :=
      p.support.exists_mem_eq_sup (Finsupp.support_nonempty_iff.mpr hp) fun m : σ →₀ ℕ =>
        Multiset.card (toMultiset m)
    have hb : ¬b ∈ q.support := by
      contrapose! h
      rw [totalDegree_eq p, hb₂, totalDegree_eq]
      apply Finset.le_sup h
    have hbb : b ∈ (p + q).support := by
      apply support_sdiff_support_subset_support_add
      rw [Finset.mem_sdiff]
      exact ⟨hb₁, hb⟩
    rw [totalDegree_eq, hb₂, totalDegree_eq]
    exact Finset.le_sup (f := fun m => Multiset.card (Finsupp.toMultiset m)) hbb


theorem totalDegree_add_eq_right_of_totalDegree_lt {p q : MvPolynomial σ R}
    (h : q.totalDegree < p.totalDegree) : (q + p).totalDegree = p.totalDegree := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p q : MvPolynomial σ R
    h : LT.lt q.totalDegree p.totalDegree
    ⊢ Eq (HAdd.hAdd q p).totalDegree p.totalDegree
  -/
  rw [add_comm, totalDegree_add_eq_left_of_totalDegree_lt h]
  /-
    🎉 no goals
  -/


theorem totalDegree_mul (a b : MvPolynomial σ R) :
    (a * b).totalDegree ≤ a.totalDegree + b.totalDegree :=
                         /-
                           R : Type u
                           σ : Type u_1
                           inst✝ : CommSemiring R
                           a b : MvPolynomial σ R
                           ⊢ ∀ {a b : Finsupp σ Nat}, LE.le ((HAdd.hAdd a b).sum fun x e => e) (HAdd.hAdd …
                         -/
  sup_support_mul_le (by exact (Finsupp.sum_add_index' (fun _ => rfl) (fun _ _ _ => rfl)).le) _ _
                         /-
                           🎉 no goals
                         -/


theorem totalDegree_smul_le [CommSemiring S] [DistribMulAction R S] (a : R) (f : MvPolynomial σ S) :
    (a • f).totalDegree ≤ f.totalDegree :=
  Finset.sup_mono support_smul


theorem totalDegree_pow (a : MvPolynomial σ R) (n : ℕ) :
    (a ^ n).totalDegree ≤ n * a.totalDegree := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    a : MvPolynomial σ R
    n : Nat
    ⊢ LE.le (HPow.hPow a n).totalDegree (HMul.hMul n a.totalDegree)
  -/
  rw [Finset.pow_eq_prod_const, ← Nat.nsmul_eq_mul, Finset.nsmul_eq_sum_const]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    a : MvPolynomial σ R
    n : Nat
    ⊢ LE.le ((Finset.range n).prod fun _k => a).totalDegree ((Finset.range n).sum  …
  -/
  refine supDegree_prod_le rfl (fun _ _ => ?_)
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    a : MvPolynomial σ R
    n : Nat
    x✝¹ x✝ : Finsupp σ Nat
    ⊢ Eq ((HAdd.hAdd x✝¹ x✝).sum fun x e => e) (HAdd.hAdd (x✝¹.sum fun x e => e) ( …
  -/
  exact Finsupp.sum_add_index' (fun _ => rfl) (fun _ _ _ => rfl)
  /-
    🎉 no goals
  -/


@[simp]
theorem totalDegree_monomial (s : σ →₀ ℕ) {c : R} (hc : c ≠ 0) :
    (monomial s c : MvPolynomial σ R).totalDegree = s.sum fun _ e => e := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : Finsupp σ Nat
    c : R
    hc : Ne c 0
    ⊢ Eq ((MvPolynomial.monomial s) c).totalDegree (s.sum fun x e => e)
  -/
  classical simp [totalDegree, support_monomial, if_neg hc]
  /-
    🎉 no goals
  -/


theorem totalDegree_monomial_le (s : σ →₀ ℕ) (c : R) :
    (monomial s c).totalDegree ≤ s.sum fun _ ↦ id := by
  if hc : c = 0 then
    simp only [hc, map_zero, totalDegree_zero, zero_le]
  else
    rw [totalDegree_monomial _ hc, Function.id_def]


@[simp]
theorem totalDegree_X_pow [Nontrivial R] (s : σ) (n : ℕ) :
                                                       /-
                                                         R : Type u
                                                         σ : Type u_1
                                                         inst✝¹ : CommSemiring R
                                                         inst✝ : Nontrivial R
                                                         s : σ
                                                         n : Nat
                                                         ⊢ Eq (HPow.hPow (MvPolynomial.X s) n).totalDegree n
                                                       -/
    (X s ^ n : MvPolynomial σ R).totalDegree = n := by simp [X_pow_eq_monomial, one_ne_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem totalDegree_list_prod :
    ∀ s : List (MvPolynomial σ R), s.prod.totalDegree ≤ (s.map MvPolynomial.totalDegree).sum
             /-
               R : Type u
               σ : Type u_1
               inst✝ : CommSemiring R
               ⊢ LE.le List.nil.prod.totalDegree (List.map MvPolynomial.totalDegree List.nil) …
             -/
  | [] => by rw [List.prod_nil, totalDegree_one, List.map_nil, List.sum_nil]
             /-
               🎉 no goals
             -/
  | p::ps => by
    /-
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      ps : List (MvPolynomial σ R)
      ⊢ LE.le (List.cons p ps).prod.totalDegree (List.map MvPolynomial.totalDegree ( …
    -/
    rw [List.prod_cons, List.map, List.sum_cons]
    /-
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      ps : List (MvPolynomial σ R)
      ⊢ LE.le (HMul.hMul p ps.prod).totalDegree (HAdd.hAdd p.totalDegree (List.map M …
    -/
    exact le_trans (totalDegree_mul _ _) (add_le_add_left (totalDegree_list_prod ps) _)
    /-
      🎉 no goals
    -/


theorem totalDegree_multiset_prod (s : Multiset (MvPolynomial σ R)) :
    s.prod.totalDegree ≤ (s.map MvPolynomial.totalDegree).sum := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : Multiset (MvPolynomial σ R)
    ⊢ LE.le s.prod.totalDegree (Multiset.map MvPolynomial.totalDegree s).sum
  -/
  refine Quotient.inductionOn s fun l => ?_
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : Multiset (MvPolynomial σ R)
    l : List (MvPolynomial σ R)
    ⊢ LE.le (Multiset.prod (Quotient.mk (List.isSetoid (MvPolynomial σ R)) l)).tot …
  -/
  rw [Multiset.quot_mk_to_coe, Multiset.prod_coe, Multiset.map_coe, Multiset.sum_coe]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : Multiset (MvPolynomial σ R)
    l : List (MvPolynomial σ R)
    ⊢ LE.le l.prod.totalDegree (List.map MvPolynomial.totalDegree l).sum
  -/
  exact totalDegree_list_prod l
  /-
    🎉 no goals
  -/


theorem totalDegree_finset_prod {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial σ R) :
    (s.prod f).totalDegree ≤ ∑ i ∈ s, (f i).totalDegree := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (s.prod f).totalDegree (s.sum fun i => (f i).totalDegree)
  -/
  refine le_trans (totalDegree_multiset_prod _) ?_
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (Multiset.map MvPolynomial.totalDegree (Multiset.map f s.val)).sum (s. …
  -/
  simp only [Multiset.map_map, comp_apply, Finset.sum_map_val, le_refl]
  /-
    🎉 no goals
  -/


theorem totalDegree_finset_sum {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial σ R) :
    (s.sum f).totalDegree ≤ Finset.sup s fun i => (f i).totalDegree := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    s : Finset ι
    f : ι → MvPolynomial σ R
    ⊢ LE.le (s.sum f).totalDegree (s.sup fun i => (f i).totalDegree)
  -/
  induction' s using Finset.cons_induction with a s has hind
    /-
      case empty
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      ι : Type u_3
      f : ι → MvPolynomial σ R
      ⊢ LE.le (EmptyCollection.emptyCollection.sum f).totalDegree (EmptyCollection.e …
    -/
  · exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      ι : Type u_3
      f : ι → MvPolynomial σ R
      a : ι
      s : Finset ι
      has : Not (Membership.mem s a)
      hind : LE.le (s.sum f).totalDegree (s.sup fun i => (f i).totalDegree)
      ⊢ LE.le ((Finset.cons a s has).sum f).totalDegree ((Finset.cons a s has).sup f …
    -/
  · rw [Finset.sum_cons, Finset.sup_cons]
    /-
      case cons
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      ι : Type u_3
      f : ι → MvPolynomial σ R
      a : ι
      s : Finset ι
      has : Not (Membership.mem s a)
      hind : LE.le (s.sum f).totalDegree (s.sup fun i => (f i).totalDegree)
      ⊢ LE.le (HAdd.hAdd (f a) (s.sum fun x => f x)).totalDegree (Max.max (f a).tota …
    -/
    exact (MvPolynomial.totalDegree_add _ _).trans (max_le_max le_rfl hind)
    /-
      🎉 no goals
    -/


lemma totalDegree_finsetSum_le {ι : Type*} {s : Finset ι} {f : ι → MvPolynomial σ R} {d : ℕ}
    (hf : ∀ i ∈ s, (f i).totalDegree ≤ d) : (s.sum f).totalDegree ≤ d :=
  (totalDegree_finset_sum ..).trans <| Finset.sup_le hf


lemma degreeOf_le_totalDegree (f : MvPolynomial σ R) (i : σ) : f.degreeOf i ≤ f.totalDegree :=
  degreeOf_le_iff.mpr fun d hd ↦ (eq_or_ne (d i) 0).elim (·.trans_le zero_le') fun h ↦
    (Finset.single_le_sum (fun _ _ ↦ zero_le') <| Finsupp.mem_support_iff.mpr h).trans
    (le_totalDegree hd)


theorem exists_degree_lt [Fintype σ] (f : MvPolynomial σ R) (n : ℕ)
    (h : f.totalDegree < n * Fintype.card σ) {d : σ →₀ ℕ} (hd : d ∈ f.support) : ∃ i, d i < n := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    f : MvPolynomial σ R
    n : Nat
    h : LT.lt f.totalDegree (HMul.hMul n (Fintype.card σ))
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    ⊢ Exists fun i => LT.lt (d i) n
  -/
  contrapose! h
  calc
    n * Fintype.card σ = ∑ _s : σ, n := by
      rw [Finset.sum_const, Nat.nsmul_eq_mul, mul_comm, Finset.card_univ]
    _ ≤ ∑ s, d s := Finset.sum_le_sum fun s _ => h s
    _ ≤ d.sum fun _ e => e := by
      rw [Finsupp.sum_fintype]
      intros
      rfl
    _ ≤ f.totalDegree := le_totalDegree hd


theorem coeff_eq_zero_of_totalDegree_lt {f : MvPolynomial σ R} {d : σ →₀ ℕ}
    (h : f.totalDegree < ∑ i ∈ d.support, d i) : coeff d f = 0 := by
  classical
    rw [totalDegree, Finset.sup_lt_iff] at h
    · specialize h d
      rw [mem_support_iff] at h
      refine not_not.mp (mt h ?_)
      exact lt_irrefl _
    · exact lt_of_le_of_lt (Nat.zero_le _) h


theorem totalDegree_eq_zero_iff_eq_C {p : MvPolynomial σ R} :
    p.totalDegree = 0 ↔ p = C (p.coeff 0) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Eq p.totalDegree 0) (Eq p (MvPolynomial.C (MvPolynomial.coeff 0 p)))
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      h : Eq p.totalDegree 0
      ⊢ Eq p (MvPolynomial.C (MvPolynomial.coeff 0 p))
    -/
  · ext m; classical rw [coeff_C]; split_ifs with hm; · rw [← hm]
    /-
      case neg
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      h : Eq p.totalDegree 0
      m : Finsupp σ Nat
      hm : Not (Eq 0 m)
      ⊢ Eq (MvPolynomial.coeff m p) 0
    -/
    apply coeff_eq_zero_of_totalDegree_lt; rw [h]
    exact Finset.sum_pos (fun i hi ↦ Nat.pos_of_ne_zero <| Finsupp.mem_support_iff.mp hi)
      (Finsupp.support_nonempty_iff.mpr <| Ne.symm hm)
    /-
      case mpr
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      h : Eq p (MvPolynomial.C (MvPolynomial.coeff 0 p))
      ⊢ Eq p.totalDegree 0
    -/
  · rw [h, totalDegree_C]
    /-
      🎉 no goals
    -/


theorem totalDegree_rename_le (f : σ → τ) (p : MvPolynomial σ R) :
    (rename f p).totalDegree ≤ p.totalDegree :=
  Finset.sup_le fun b => by
    classical
    intro h
    rw [rename_eq] at h
    have h' := Finsupp.mapDomain_support h
    rw [Finset.mem_image] at h'
    rcases h' with ⟨s, hs, rfl⟩
    exact (sum_mapDomain_index (fun _ => rfl) (fun _ _ _ => rfl)).trans_le (le_totalDegree hs)


