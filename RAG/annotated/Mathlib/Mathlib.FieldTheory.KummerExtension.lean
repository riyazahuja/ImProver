lemma root_X_pow_sub_C_pow (n : ℕ) (a : K) :
    (AdjoinRoot.root (X ^ n - C a)) ^ n = AdjoinRoot.of _ a := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    ⊢ Eq (HPow.hPow (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polyno …
  -/
  rw [← sub_eq_zero, ← AdjoinRoot.eval₂_root, eval₂_sub, eval₂_C, eval₂_pow, eval₂_X]
  /-
    🎉 no goals
  -/


lemma root_X_pow_sub_C_ne_zero {n : ℕ} (hn : 1 < n) (a : K) :
    (AdjoinRoot.root (X ^ n - C a)) ≠ 0 :=
  mk_ne_zero_of_natDegree_lt (monic_X_pow_sub_C _ (Nat.not_eq_zero_of_lt hn))
                    /-
                      K : Type u
                      inst✝ : Field K
                      n : Nat
                      hn : LT.lt 1 n
                      a : K
                      ⊢ LT.lt Polynomial.X.natDegree (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomi …
                    -/
    X_ne_zero <| by rwa [natDegree_X_pow_sub_C, natDegree_X]
                    /-
                      🎉 no goals
                    -/


lemma root_X_pow_sub_C_ne_zero' {n : ℕ} {a : K} (hn : 0 < n) (ha : a ≠ 0) :
    (AdjoinRoot.root (X ^ n - C a)) ≠ 0 := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    hn : LT.lt 0 n
    ha : Ne a 0
    ⊢ Ne (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))) 0
  -/
  obtain (rfl|hn) := (Nat.succ_le_iff.mpr hn).eq_or_lt
    /-
      case inl
      K : Type u
      inst✝ : Field K
      a : K
      ha : Ne a 0
      hn : LT.lt 0 (Nat.succ 0)
      ⊢ Ne (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X (Nat.succ 0)) (Polyno …
    -/
  · rw [pow_one]
    /-
      case inl
      K : Type u
      inst✝ : Field K
      a : K
      ha : Ne a 0
      hn : LT.lt 0 (Nat.succ 0)
      ⊢ Ne (AdjoinRoot.root (HSub.hSub Polynomial.X (Polynomial.C a))) 0
    -/
    intro e
    /-
      case inl
      K : Type u
      inst✝ : Field K
      a : K
      ha : Ne a 0
      hn : LT.lt 0 (Nat.succ 0)
      e : Eq (AdjoinRoot.root (HSub.hSub Polynomial.X (Polynomial.C a))) 0
      ⊢ False
    -/
    refine mk_ne_zero_of_natDegree_lt (monic_X_sub_C a) (C_ne_zero.mpr ha) (by simp) ?_
    /-
      case inl
      K : Type u
      inst✝ : Field K
      a : K
      ha : Ne a 0
      hn : LT.lt 0 (Nat.succ 0)
      e : Eq (AdjoinRoot.root (HSub.hSub Polynomial.X (Polynomial.C a))) 0
      ⊢ Eq ((AdjoinRoot.mk (HSub.hSub Polynomial.X (Polynomial.C a))) (Polynomial.C  …
    -/
    trans AdjoinRoot.mk (X - C a) (X - (X - C a))
      /-
        K : Type u
        inst✝ : Field K
        a : K
        ha : Ne a 0
        hn : LT.lt 0 (Nat.succ 0)
        e : Eq (AdjoinRoot.root (HSub.hSub Polynomial.X (Polynomial.C a))) 0
        ⊢ Eq ((AdjoinRoot.mk (HSub.hSub Polynomial.X (Polynomial.C a))) (Polynomial.C  …
      -/
    · rw [sub_sub_cancel]
      /-
        🎉 no goals
      -/
      /-
        K : Type u
        inst✝ : Field K
        a : K
        ha : Ne a 0
        hn : LT.lt 0 (Nat.succ 0)
        e : Eq (AdjoinRoot.root (HSub.hSub Polynomial.X (Polynomial.C a))) 0
        ⊢ Eq ((AdjoinRoot.mk (HSub.hSub Polynomial.X (Polynomial.C a))) (HSub.hSub Pol …
      -/
    · rw [map_sub, mk_self, sub_zero, mk_X, e]
      /-
        🎉 no goals
      -/
    /-
      case inr
      K : Type u
      inst✝ : Field K
      n : Nat
      a : K
      hn✝ : LT.lt 0 n
      ha : Ne a 0
      hn : LT.lt (Nat.succ 0) n
      ⊢ Ne (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))) 0
    -/
  · exact root_X_pow_sub_C_ne_zero hn a
    /-
      🎉 no goals
    -/


theorem X_pow_sub_C_splits_of_isPrimitiveRoot
    {n : ℕ} {ζ : K} (hζ : IsPrimitiveRoot ζ n) {α a : K} (e : α ^ n = a) :
    (X ^ n - C a).Splits (RingHom.id _) := by
  cases n.eq_zero_or_pos with
  | inl hn =>
    rw [hn, pow_zero, ← C.map_one, ← map_sub]
    exact splits_C _ _
  | inr hn =>
    rw [splits_iff_card_roots, ← nthRoots, hζ.card_nthRoots, natDegree_X_pow_sub_C, if_pos ⟨α, e⟩]

-- make this private, as we only use it to prove a strictly more general version

private
theorem X_pow_sub_C_eq_prod'
    {n : ℕ} {ζ : K} (hζ : IsPrimitiveRoot ζ n) {α a : K} (hn : 0 < n) (e : α ^ n = a) :
    (X ^ n - C a) = ∏ i ∈ Finset.range n, (X - C (ζ ^ i * α)) := by
  rw [eq_prod_roots_of_monic_of_splits_id (monic_X_pow_sub_C _ (Nat.pos_iff_ne_zero.mp hn))
    (X_pow_sub_C_splits_of_isPrimitiveRoot hζ e), ← nthRoots, hζ.nthRoots_eq e, Multiset.map_map]
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    α a : K
    hn : LT.lt 0 n
    e : Eq (HPow.hPow α n) a
    ⊢ Eq (Multiset.map (Function.comp (fun a => HSub.hSub Polynomial.X (Polynomial …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma X_pow_sub_C_eq_prod {R : Type*} [CommRing R] [IsDomain R]
    {n : ℕ} {ζ : R} (hζ : IsPrimitiveRoot ζ n) {α a : R} (hn : 0 < n) (e : α ^ n = a) :
    (X ^ n - C a) = ∏ i ∈ Finset.range n, (X - C (ζ ^ i * α)) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    hn : LT.lt 0 n
    e : Eq (HPow.hPow α n) a
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) ((Finset.range n) …
  -/
  let K := FractionRing R
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    hn : LT.lt 0 n
    e : Eq (HPow.hPow α n) a
    K : Type u_1 := FractionRing R
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) ((Finset.range n) …
  -/
  let i := algebraMap R K
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    hn : LT.lt 0 n
    e : Eq (HPow.hPow α n) a
    K : Type u_1 := FractionRing R
    i : RingHom R K := algebraMap R K
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) ((Finset.range n) …
  -/
  have h := NoZeroSMulDivisors.algebraMap_injective R K
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    hn : LT.lt 0 n
    e : Eq (HPow.hPow α n) a
    K : Type u_1 := FractionRing R
    i : RingHom R K := algebraMap R K
    h : Function.Injective ⇑(algebraMap R K)
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) ((Finset.range n) …
  -/
  apply_fun Polynomial.map i using map_injective i h
  simpa only [Polynomial.map_sub, Polynomial.map_pow, map_X, map_C, map_mul, map_pow,
    Polynomial.map_prod, Polynomial.map_mul]
    using X_pow_sub_C_eq_prod' (hζ.map_of_injective h) hn <| map_pow i α n ▸ congrArg i e


lemma ne_zero_of_irreducible_X_pow_sub_C {n : ℕ} {a : K} (H : Irreducible (X ^ n - C a)) :
    n ≠ 0 := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ Ne n 0
  -/
  rintro rfl
  /-
    K : Type u
    inst✝ : Field K
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X 0) (Polynomial.C a))
    ⊢ False
  -/
  rw [pow_zero, ← C.map_one, ← map_sub] at H
  /-
    K : Type u
    inst✝ : Field K
    a : K
    H : Irreducible (Polynomial.C (HSub.hSub 1 a))
    ⊢ False
  -/
  exact not_irreducible_C _ H
  /-
    🎉 no goals
  -/


lemma ne_zero_of_irreducible_X_pow_sub_C' {n : ℕ} (hn : n ≠ 1) {a : K}
    (H : Irreducible (X ^ n - C a)) : a ≠ 0 := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hn : Ne n 1
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ Ne a 0
  -/
  rintro rfl
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hn : Ne n 1
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C 0))
    ⊢ False
  -/
  rw [map_zero, sub_zero] at H
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hn : Ne n 1
    H : Irreducible (HPow.hPow Polynomial.X n)
    ⊢ False
  -/
  exact not_irreducible_pow hn H
  /-
    🎉 no goals
  -/


lemma root_X_pow_sub_C_eq_zero_iff {n : ℕ} {a : K} (H : Irreducible (X ^ n - C a)) :
    (AdjoinRoot.root (X ^ n - C a)) = 0 ↔ a = 0 := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ Iff (Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
  -/
  have hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    hn : LT.lt 0 n
    ⊢ Iff (Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
  -/
  refine ⟨not_imp_not.mp (root_X_pow_sub_C_ne_zero' hn), ?_⟩
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    hn : LT.lt 0 n
    ⊢ Eq a 0 → Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomi …
  -/
  rintro rfl
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hn : LT.lt 0 n
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C 0))
    ⊢ Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C 0))) 0
  -/
  have := not_imp_not.mp (fun hn ↦ ne_zero_of_irreducible_X_pow_sub_C' hn H) rfl
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hn : LT.lt 0 n
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C 0))
    this : Eq n 1
    ⊢ Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C 0))) 0
  -/
  rw [this, pow_one, map_zero, sub_zero, ← mk_X, mk_self]
  /-
    🎉 no goals
  -/


lemma root_X_pow_sub_C_ne_zero_iff {n : ℕ} {a : K} (H : Irreducible (X ^ n - C a)) :
    (AdjoinRoot.root (X ^ n - C a)) ≠ 0 ↔ a ≠ 0 :=
  (root_X_pow_sub_C_eq_zero_iff H).not


theorem pow_ne_of_irreducible_X_pow_sub_C {n : ℕ} {a : K}
    (H : Irreducible (X ^ n - C a)) {m : ℕ} (hm : m ∣ n) (hm' : m ≠ 1) (b : K) : b ^ m ≠ a := by
  have hn : n ≠ 0 := fun e ↦ not_irreducible_C
    (1 - a) (by simpa only [e, pow_zero, ← C.map_one, ← map_sub] using H)
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    m : Nat
    hm : Dvd.dvd m n
    hm' : Ne m 1
    b : K
    hn : Ne n 0
    ⊢ Ne (HPow.hPow b m) a
  -/
  obtain ⟨k, rfl⟩ := hm
  /-
    case intro
    K : Type u
    inst✝ : Field K
    a : K
    m : Nat
    hm' : Ne m 1
    b : K
    k : Nat
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X (HMul.hMul m k)) (Polynomia …
    hn : Ne (HMul.hMul m k) 0
    ⊢ Ne (HPow.hPow b m) a
  -/
  rintro rfl
  /-
    case intro
    K : Type u
    inst✝ : Field K
    m : Nat
    hm' : Ne m 1
    b : K
    k : Nat
    hn : Ne (HMul.hMul m k) 0
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X (HMul.hMul m k)) (Polynomia …
    ⊢ False
  -/
  obtain ⟨q, hq⟩ := sub_dvd_pow_sub_pow (X ^ k) (C b) m
  /-
    case intro.intro
    K : Type u
    inst✝ : Field K
    m : Nat
    hm' : Ne m 1
    b : K
    k : Nat
    hn : Ne (HMul.hMul m k) 0
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X (HMul.hMul m k)) (Polynomia …
    q : Polynomial K
    hq : Eq (HSub.hSub (HPow.hPow (HPow.hPow Polynomial.X k) m) (HPow.hPow (Polyno …
    ⊢ False
  -/
  rw [mul_comm, pow_mul, map_pow, hq] at H
  have : degree q = 0 := by
    simpa [isUnit_iff_degree_eq_zero, degree_X_pow_sub_C,
      Nat.pos_iff_ne_zero, (mul_ne_zero_iff.mp hn).2] using H.2 _ q rfl
  /-
    case intro.intro
    K : Type u
    inst✝ : Field K
    m : Nat
    hm' : Ne m 1
    b : K
    k : Nat
    hn : Ne (HMul.hMul m k) 0
    q : Polynomial K
    H : Irreducible (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X k) (Polynomial.C …
    hq : Eq (HSub.hSub (HPow.hPow (HPow.hPow Polynomial.X k) m) (HPow.hPow (Polyno …
    this : Eq q.degree 0
    ⊢ False
  -/
  apply_fun degree at hq
  simp only [this, ← pow_mul, mul_comm k m, degree_X_pow_sub_C, Nat.pos_iff_ne_zero.mpr hn,
    Nat.pos_iff_ne_zero.mpr (mul_ne_zero_iff.mp hn).2, degree_mul, ← map_pow, add_zero,
    Nat.cast_injective.eq_iff] at hq
  /-
    case intro.intro
    K : Type u
    inst✝ : Field K
    m : Nat
    hm' : Ne m 1
    b : K
    k : Nat
    hn : Ne (HMul.hMul m k) 0
    q : Polynomial K
    H : Irreducible (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X k) (Polynomial.C …
    this : Eq q.degree 0
    hq : Eq (HMul.hMul m k) k
    ⊢ False
  -/
  exact hm' ((mul_eq_right₀ (mul_ne_zero_iff.mp hn).2).mp hq)
  /-
    🎉 no goals
  -/


/--Let `p` be a prime number. Let `K` be a field.
Let `t ∈ K` be an element which does not have a `p`th root in `K`.
Then the polynomial `x ^ p - t` is irreducible over `K`.-/
@[stacks 09HF "We proved the result without the condition that `K` is char p in 09HF."]
theorem X_pow_sub_C_irreducible_of_prime {p : ℕ} (hp : p.Prime) {a : K} (ha : ∀ b : K, b ^ p ≠ a) :
    Irreducible (X ^ p - C a) := by
  -- First of all, We may find an irreducible factor `g` of `X ^ p - C a`.
  have : ¬ IsUnit (X ^ p - C a) := by
    rw [Polynomial.isUnit_iff_degree_eq_zero, degree_X_pow_sub_C hp.pos, Nat.cast_eq_zero]
    exact hp.ne_zero
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    this : Not (IsUnit (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a)))
    ⊢ Irreducible (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a))
  -/
  have ⟨g, hg, hg'⟩ := WfDvdMonoid.exists_irreducible_factor this (X_pow_sub_C_ne_zero hp.pos a)
  -- It suffices to show that `deg g = p`.
  suffices natDegree g = p from (associated_of_dvd_of_natDegree_le hg'
    (X_pow_sub_C_ne_zero hp.pos a) (this.trans natDegree_X_pow_sub_C.symm).ge).irreducible hg
  -- Suppose `deg g ≠ p`.
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    this : Not (IsUnit (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a)))
    g : Polynomial K
    hg : Irreducible g
    hg' : Dvd.dvd g (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a))
    ⊢ Eq g.natDegree p
  -/
  by_contra h
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    this : Not (IsUnit (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a)))
    g : Polynomial K
    hg : Irreducible g
    hg' : Dvd.dvd g (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a))
    h : Not (Eq g.natDegree p)
    ⊢ False
  -/
  have : Fact (Irreducible g) := ⟨hg⟩
  -- Let `r` be a root of `g`, then `N_K(r) ^ p = N_K(r ^ p) = N_K(a) = a ^ (deg g)`.
  have key : (Algebra.norm K (AdjoinRoot.root g)) ^ p = a ^ g.natDegree := by
    have := eval₂_eq_zero_of_dvd_of_eval₂_eq_zero _ _ hg' (AdjoinRoot.eval₂_root g)
    rw [eval₂_sub, eval₂_pow, eval₂_C, eval₂_X, sub_eq_zero] at this
    rw [← map_pow, this, ← AdjoinRoot.algebraMap_eq, Algebra.norm_algebraMap,
      ← finrank_top', ← IntermediateField.adjoin_root_eq_top g,
      IntermediateField.adjoin.finrank,
      AdjoinRoot.minpoly_root hg.ne_zero, natDegree_mul_C]
    · simpa using hg.ne_zero
    · exact AdjoinRoot.isIntegral_root hg.ne_zero
  -- Since `a ^ (deg g)` is a `p`-power, and `p` is coprime to `deg g`, we conclude that `a` is
  -- also a `p`-power, contradicting the hypothesis
  have : p.Coprime (natDegree g) := hp.coprime_iff_not_dvd.mpr (fun e ↦ h (((natDegree_le_of_dvd hg'
    (X_pow_sub_C_ne_zero hp.pos a)).trans_eq natDegree_X_pow_sub_C).antisymm (Nat.le_of_dvd
      (natDegree_pos_iff_degree_pos.mpr <| Polynomial.degree_pos_of_irreducible hg) e)))
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    this✝¹ : Not (IsUnit (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a)))
    g : Polynomial K
    hg : Irreducible g
    hg' : Dvd.dvd g (HSub.hSub (HPow.hPow Polynomial.X p) (Polynomial.C a))
    h : Not (Eq g.natDegree p)
    this✝ : Fact (Irreducible g)
    key : Eq (HPow.hPow ((Algebra.norm K) (AdjoinRoot.root g)) p) (HPow.hPow a g.n …
    this : p.Coprime g.natDegree
    ⊢ False
  -/
  exact ha _ ((pow_mem_range_pow_of_coprime this.symm a).mp ⟨_, key⟩).choose_spec
  /-
    🎉 no goals
  -/


theorem X_pow_sub_C_irreducible_iff_of_prime {p : ℕ} (hp : p.Prime) {a : K} :
    Irreducible (X ^ p - C a) ↔ ∀ b, b ^ p ≠ a :=
  ⟨(pow_ne_of_irreducible_X_pow_sub_C · dvd_rfl hp.ne_one), X_pow_sub_C_irreducible_of_prime hp⟩


theorem X_pow_mul_sub_C_irreducible
    {n m : ℕ} {a : K} (hm : Irreducible (X ^ m - C a))
    (hn : ∀ (E : Type u) [Field E] [Algebra K E] (x : E) (_ : minpoly K x = X ^ m - C a),
      Irreducible (X ^ n - C (AdjoinSimple.gen K x))) :
    Irreducible (X ^ (n * m) - C a) := by
  have hm' : m ≠ 0 := by
    rintro rfl
    rw [pow_zero, ← C.map_one, ← map_sub] at hm
    exact not_irreducible_C _ hm
  simpa [pow_mul] using irreducible_comp (monic_X_pow_sub_C a hm') (monic_X_pow n) hm
    (by simpa only [Polynomial.map_pow, map_X] using hn)

-- TODO: generalize to even `n`

theorem X_pow_sub_C_irreducible_of_odd
    {n : ℕ} (hn : Odd n) {a : K} (ha : ∀ p : ℕ, p.Prime → p ∣ n → ∀ b : K, b ^ p ≠ a) :
    Irreducible (X ^ n - C a) := by
  induction n using induction_on_primes generalizing K a with
  | h₀ => simp [← Nat.not_even_iff_odd] at hn
  | h₁ => simpa using irreducible_X_sub_C a
  | h p n hp IH =>
    rw [mul_comm]
    apply X_pow_mul_sub_C_irreducible
      (X_pow_sub_C_irreducible_of_prime hp (ha p hp (dvd_mul_right _ _)))
    intro E _ _ x hx
    have : IsIntegral K x := not_not.mp fun h ↦ by
      simpa only [degree_zero, degree_X_pow_sub_C hp.pos,
        WithBot.natCast_ne_bot] using congr_arg degree (hx.symm.trans (dif_neg h))
    apply IH (Nat.odd_mul.mp hn).2
    intros q hq hqn b hb
    apply ha q hq (dvd_mul_of_dvd_right hqn p) (Algebra.norm _ b)
    rw [← map_pow, hb, ← adjoin.powerBasis_gen this,
      Algebra.PowerBasis.norm_gen_eq_coeff_zero_minpoly]
    simp [minpoly_gen, hx, hp.ne_zero.symm, (Nat.odd_mul.mp hn).1.neg_pow]


theorem X_pow_sub_C_irreducible_iff_forall_prime_of_odd {n : ℕ} (hn : Odd n) {a : K} :
    Irreducible (X ^ n - C a) ↔ (∀ p : ℕ, p.Prime → p ∣ n → ∀ b : K, b ^ p ≠ a) :=
  ⟨fun e _ hp hpn ↦ pow_ne_of_irreducible_X_pow_sub_C e hpn hp.ne_one,
    X_pow_sub_C_irreducible_of_odd hn⟩


theorem X_pow_sub_C_irreducible_iff_of_odd {n : ℕ} (hn : Odd n) {a : K} :
    Irreducible (X ^ n - C a) ↔ (∀ d, d ∣ n → d ≠ 1 → ∀ b : K, b ^ d ≠ a) :=
  ⟨fun e _ ↦ pow_ne_of_irreducible_X_pow_sub_C e,
    fun H ↦ X_pow_sub_C_irreducible_of_odd hn fun p hp hpn ↦ (H p hpn hp.ne_one)⟩

-- TODO: generalize to `p = 2`

theorem X_pow_sub_C_irreducible_of_prime_pow
    {p : ℕ} (hp : p.Prime) (hp' : p ≠ 2) (n : ℕ) {a : K} (ha : ∀ b : K, b ^ p ≠ a) :
    Irreducible (X ^ (p ^ n) - C a) := by
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    hp' : Ne p 2
    n : Nat
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    ⊢ Irreducible (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p n)) (Polynomial. …
  -/
  apply X_pow_sub_C_irreducible_of_odd (hp.odd_of_ne_two hp').pow
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    hp' : Ne p 2
    n : Nat
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    ⊢ ∀ (p_1 : Nat), Nat.Prime p_1 → Dvd.dvd p_1 (HPow.hPow p n) → ∀ (b : K), Ne ( …
  -/
  intros q hq hq'
  /-
    K : Type u
    inst✝ : Field K
    p : Nat
    hp : Nat.Prime p
    hp' : Ne p 2
    n : Nat
    a : K
    ha : ∀ (b : K), Ne (HPow.hPow b p) a
    q : Nat
    hq : Nat.Prime q
    hq' : Dvd.dvd q (HPow.hPow p n)
    ⊢ ∀ (b : K), Ne (HPow.hPow b q) a
  -/
  simpa [(Nat.prime_dvd_prime_iff_eq hq hp).mp (hq.dvd_of_dvd_pow hq')] using ha
  /-
    🎉 no goals
  -/


theorem X_pow_sub_C_irreducible_iff_of_prime_pow
    {p : ℕ} (hp : p.Prime) (hp' : p ≠ 2) {n} (hn : n ≠ 0) {a : K} :
    Irreducible (X ^ p ^ n - C a) ↔ ∀ b, b ^ p ≠ a :=
  ⟨(pow_ne_of_irreducible_X_pow_sub_C · (dvd_pow dvd_rfl hn) hp.ne_one),
    X_pow_sub_C_irreducible_of_prime_pow hp hp' n⟩


set_option quotPrecheck false in
scoped[KummerExtension] notation3 "K[" n "√" a "]" => AdjoinRoot (Polynomial.X ^ n - Polynomial.C a)


include hζ H in
/-- Also see `Polynomial.separable_X_pow_sub_C_unit` -/
theorem Polynomial.separable_X_pow_sub_C_of_irreducible : (X ^ n - C a).Separable := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  letI := Fact.mk H
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  letI : Algebra K K[n√a] := inferInstance
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  have hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    hn : LT.lt 0 n
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  by_cases hn' : n = 1
    /-
      case pos
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      hn : LT.lt 0 n
      hn' : Eq n 1
      ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
    -/
  · rw [hn', pow_one]; exact separable_X_sub_C
                       /-
                         🎉 no goals
                       -/
  /-
    case neg
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    hn : LT.lt 0 n
    hn' : Not (Eq n 1)
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  have ⟨ζ, hζ⟩ := hζ
  /-
    case neg
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ✝ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    hn : LT.lt 0 n
    hn' : Not (Eq n 1)
    ζ : K
    hζ : Membership.mem (primitiveRoots n K) ζ
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Separable
  -/
  rw [mem_primitiveRoots (Nat.pos_of_ne_zero <| ne_zero_of_irreducible_X_pow_sub_C H)] at hζ
  rw [← separable_map (algebraMap K K[n√a]), Polynomial.map_sub, Polynomial.map_pow, map_C, map_X,
    AdjoinRoot.algebraMap_eq,
    X_pow_sub_C_eq_prod (hζ.map_of_injective (algebraMap K _).injective) hn
    (root_X_pow_sub_C_pow n a), separable_prod_X_sub_C_iff']
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/5376 we need to provide this helper instance.
  -/
  /-
    case neg
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ✝ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    hn : LT.lt 0 n
    hn' : Not (Eq n 1)
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → ∀ (y : Nat), Membership.mem …
  -/
  have : MonoidHomClass (K →+* K[n√a]) K K[n√a] := inferInstance
  exact (hζ.map_of_injective (algebraMap K K[n√a]).injective).injOn_pow_mul
    (root_X_pow_sub_C_ne_zero (lt_of_le_of_ne (show 1 ≤ n from hn) (Ne.symm hn')) _)


/-- The natural embedding of the roots of unity of `K` into `Gal(K[ⁿ√a]/K)`, by sending
`η ↦ (ⁿ√a ↦ η • ⁿ√a)`. Also see `autAdjoinRootXPowSubC` for the `AlgEquiv` version. -/
noncomputable
def autAdjoinRootXPowSubCHom :
    rootsOfUnity n K →* (K[n√a] →ₐ[K] K[n√a]) where
  toFun := fun η ↦ liftHom (X ^ n - C a) (((η : Kˣ) : K) • (root _) : K[n√a]) <| by
    /-
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      ⊢ Eq ((Polynomial.aeval (HSMul.hSMul (↑↑η) (AdjoinRoot.root (HSub.hSub (HPow.h …
    -/
    have := (mem_rootsOfUnity' _ _).mp η.prop
    rw [map_sub, map_pow, aeval_C, aeval_X, Algebra.smul_def, mul_pow, root_X_pow_sub_C_pow,
      AdjoinRoot.algebraMap_eq, ← map_pow, this, map_one, one_mul, sub_self]
                               /-
                                 K : Type u
                                 inst✝ : Field K
                                 n : Nat
                                 hζ : (primitiveRoots n K).Nonempty
                                 a : K
                                 H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
                                 ⊢ Eq (((fun η => AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
                               -/
  map_one' := algHom_ext <| by simp
                               /-
                                 🎉 no goals
                               -/
                                         /-
                                           K : Type u
                                           inst✝ : Field K
                                           n : Nat
                                           hζ : (primitiveRoots n K).Nonempty
                                           a : K
                                           H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
                                           ε η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
                                           ⊢ Eq (({ toFun := fun η => AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial …
                                         -/
  map_mul' := fun ε η ↦ algHom_ext <| by simp [mul_smul, smul_comm ((ε : Kˣ) : K)]
                                         /-
                                           🎉 no goals
                                         -/


/-- The natural embedding of the roots of unity of `K` into `Gal(K[ⁿ√a]/K)`, by sending
`η ↦ (ⁿ√a ↦ η • ⁿ√a)`. This is an isomorphism when `K` contains a primitive root of unity.
See `autAdjoinRootXPowSubCEquiv`. -/
noncomputable
def autAdjoinRootXPowSubC :
    rootsOfUnity n K →* (K[n√a] ≃ₐ[K] K[n√a]) :=
  (AlgEquiv.algHomUnitsEquiv _ _).toMonoidHom.comp (autAdjoinRootXPowSubCHom n a).toHomUnits


lemma autAdjoinRootXPowSubC_root (η) :
    autAdjoinRootXPowSubC n a η (root _) = ((η : Kˣ) : K) • root _ := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
    ⊢ Eq (((autAdjoinRootXPowSubC n a) η) (AdjoinRoot.root (HSub.hSub (HPow.hPow P …
  -/
  dsimp [autAdjoinRootXPowSubC, autAdjoinRootXPowSubCHom, AlgEquiv.algHomUnitsEquiv]
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    a : K
    η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
    ⊢ Eq ((AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
  -/
  apply liftHom_root
  /-
    🎉 no goals
  -/


/-- The inverse function of `autAdjoinRootXPowSubC` if `K` has all roots of unity.
See `autAdjoinRootXPowSubCEquiv`. -/
noncomputable
def AdjoinRootXPowSubCEquivToRootsOfUnity [NeZero n] (σ : K[n√a] ≃ₐ[K] K[n√a]) :
    rootsOfUnity n K :=
  letI := Fact.mk H
  letI : IsDomain K[n√a] := inferInstance
  letI := Classical.decEq K
  (rootsOfUnityEquivOfPrimitiveRoots (n := n) (algebraMap K K[n√a]).injective hζ).symm
    (rootsOfUnity.mkOfPowEq (if a = 0 then 1 else σ (root _) / root _) (by
    -- The if is needed in case `n = 1` and `a = 0` and `K[n√a] = K`.
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : DecidableEq K := Classical.decEq K
      ⊢ Eq (HPow.hPow (ite (Eq a 0) 1 (HDiv.hDiv (σ (AdjoinRoot.root (HSub.hSub (HPo …
    -/
    split
      /-
        case isTrue
        K : Type u
        inst✝¹ : Field K
        n : Nat
        hζ : (primitiveRoots n K).Nonempty
        a : K
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
        inst✝ : NeZero n
        σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        this : DecidableEq K := Classical.decEq K
        h✝ : Eq a 0
        ⊢ Eq (HPow.hPow 1 n) 1
      -/
    · exact one_pow _
      /-
        🎉 no goals
      -/
    /-
      case isFalse
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : DecidableEq K := Classical.decEq K
      h✝ : Not (Eq a 0)
      ⊢ Eq (HPow.hPow (HDiv.hDiv (σ (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomia …
    -/
    rw [div_pow, ← map_pow]
    /-
      case isFalse
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : DecidableEq K := Classical.decEq K
      h✝ : Not (Eq a 0)
      ⊢ Eq (HDiv.hDiv (σ (HPow.hPow (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomia …
    -/
    simp only [root_X_pow_sub_C_pow, ← AdjoinRoot.algebraMap_eq, AlgEquiv.commutes]
    /-
      case isFalse
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : DecidableEq K := Classical.decEq K
      h✝ : Not (Eq a 0)
      ⊢ Eq (HDiv.hDiv ((algebraMap K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X  …
    -/
    rw [div_self]
    /-
      case isFalse
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : DecidableEq K := Classical.decEq K
      h✝ : Not (Eq a 0)
      ⊢ Ne ((algebraMap K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynom …
    -/
    rwa [Ne, map_eq_zero_iff _ (algebraMap K _).injective]))
    /-
      🎉 no goals
    -/


/-- The equivalence between the roots of unity of `K` and `Gal(K[ⁿ√a]/K)`. -/
noncomputable
def autAdjoinRootXPowSubCEquiv [NeZero n] :
    rootsOfUnity n K ≃* (K[n√a] ≃ₐ[K] K[n√a]) where
  __ := autAdjoinRootXPowSubC n a
  invFun := AdjoinRootXPowSubCEquivToRootsOfUnity hζ H
  left_inv := by
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      ⊢ Function.LeftInverse (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H) (↑__spread …
    -/
    intro η
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      ⊢ Eq (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H ((↑__spread✝⁻⁰).toFun η)) η
    -/
    have := Fact.mk H
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
      ⊢ Eq (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H ((↑__spread✝⁻⁰).toFun η)) η
    -/
    have : IsDomain K[n√a] := inferInstance
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial. …
      ⊢ Eq (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H ((↑__spread✝⁻⁰).toFun η)) η
    -/
    letI : Algebra K K[n√a] := inferInstance
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H ((↑__spread✝⁻⁰).toFun η)) η
    -/
    apply (rootsOfUnityEquivOfPrimitiveRoots (algebraMap K K[n√a]).injective hζ).injective
    /-
      case a
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq ((rootsOfUnityEquivOfPrimitiveRoots ⋯ hζ) (AdjoinRootXPowSubCEquivToRoots …
    -/
    ext
    simp only [AdjoinRoot.algebraMap_eq, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe,
      autAdjoinRootXPowSubC_root, Algebra.smul_def, ne_eq, MulEquiv.apply_symm_apply,
      rootsOfUnity.val_mkOfPowEq_coe, val_rootsOfUnityEquivOfPrimitiveRoots_apply_coe,
      AdjoinRootXPowSubCEquivToRootsOfUnity]
    /-
      case a.a.a
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
      this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (ite (Eq a 0) 1 (HDiv.hDiv (HMul.hMul ((AdjoinRoot.of (HSub.hSub (HPow.hP …
    -/
    split_ifs with h
      /-
        case pos
        K : Type u
        inst✝¹ : Field K
        n : Nat
        hζ : (primitiveRoots n K).Nonempty
        a : K
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
        inst✝ : NeZero n
        η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
        this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        h : Eq a 0
        ⊢ Eq 1 ((AdjoinRoot.of (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) …
      -/
    · obtain rfl := not_imp_not.mp (fun hn ↦ ne_zero_of_irreducible_X_pow_sub_C' hn H) h
      /-
        case pos
        K : Type u
        inst✝¹ : Field K
        a : K
        h : Eq a 0
        hζ : (primitiveRoots 1 K).Nonempty
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C a))
        inst✝ : NeZero 1
        __spread✝⁻⁰ : MonoidHom (Subtype fun x => Membership.mem (rootsOfUnity 1 K) x) …
        η : Subtype fun x => Membership.mem (rootsOfUnity 1 K) x
        this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C …
        this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial …
        this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial …
        ⊢ Eq 1 ((AdjoinRoot.of (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C a)) …
      -/
      have : (η : Kˣ) = 1 := (pow_one _).symm.trans η.prop
      /-
        case pos
        K : Type u
        inst✝¹ : Field K
        a : K
        h : Eq a 0
        hζ : (primitiveRoots 1 K).Nonempty
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C a))
        inst✝ : NeZero 1
        __spread✝⁻⁰ : MonoidHom (Subtype fun x => Membership.mem (rootsOfUnity 1 K) x) …
        η : Subtype fun x => Membership.mem (rootsOfUnity 1 K) x
        this✝² : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C …
        this✝¹ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomia …
        this✝ : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomia …
        this : Eq (↑η) 1
        ⊢ Eq 1 ((AdjoinRoot.of (HSub.hSub (HPow.hPow Polynomial.X 1) (Polynomial.C a)) …
      -/
      simp only [this, Units.val_one, map_one]
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u
        inst✝¹ : Field K
        n : Nat
        hζ : (primitiveRoots n K).Nonempty
        a : K
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
        inst✝ : NeZero n
        η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
        this✝¹ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝ : IsDomain (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        h : Not (Eq a 0)
        ⊢ Eq (HDiv.hDiv (HMul.hMul ((AdjoinRoot.of (HSub.hSub (HPow.hPow Polynomial.X  …
      -/
    · exact mul_div_cancel_right₀ _ (root_X_pow_sub_C_ne_zero' (NeZero.pos n) h)
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      ⊢ Function.RightInverse (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H) (↑__sprea …
    -/
    intro e
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      ⊢ Eq ((↑__spread✝⁻⁰).toFun (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H e)) e
    -/
    have := Fact.mk H
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
      ⊢ Eq ((↑__spread✝⁻⁰).toFun (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H e)) e
    -/
    letI : Algebra K K[n√a] := inferInstance
    /-
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq ((↑__spread✝⁻⁰).toFun (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H e)) e
    -/
    apply AlgEquiv.coe_algHom_injective
    /-
      case a
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq ↑((↑__spread✝⁻⁰).toFun (AdjoinRootXPowSubCEquivToRootsOfUnity hζ H e)) ↑e
    -/
    apply AdjoinRoot.algHom_ext
    simp only [AdjoinRootXPowSubCEquivToRootsOfUnity, AdjoinRoot.algebraMap_eq, OneHom.toFun_eq_coe,
      MonoidHom.toOneHom_coe, AlgHom.coe_coe, autAdjoinRootXPowSubC_root, Algebra.smul_def]
    /-
      case a.h
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (HMul.hMul ((AdjoinRoot.of (HSub.hSub (HPow.hPow Polynomial.X n) (Polynom …
    -/
    rw [rootsOfUnityEquivOfPrimitiveRoots_symm_apply, rootsOfUnity.val_mkOfPowEq_coe]
    /-
      case a.h
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (HMul.hMul (ite (Eq a 0) 1 (HDiv.hDiv (e (AdjoinRoot.root (HSub.hSub (HPo …
    -/
    split_ifs with h
      /-
        case pos
        K : Type u
        inst✝¹ : Field K
        n : Nat
        hζ : (primitiveRoots n K).Nonempty
        a : K
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
        inst✝ : NeZero n
        e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
        this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        h : Eq a 0
        ⊢ Eq (HMul.hMul 1 (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Poly …
      -/
    · obtain rfl := not_imp_not.mp (fun hn ↦ ne_zero_of_irreducible_X_pow_sub_C' hn H) h
      rw [(pow_one _).symm.trans (root_X_pow_sub_C_pow 1 a), one_mul,
        ← AdjoinRoot.algebraMap_eq, AlgEquiv.commutes]
      /-
        case neg
        K : Type u
        inst✝¹ : Field K
        n : Nat
        hζ : (primitiveRoots n K).Nonempty
        a : K
        H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
        inst✝ : NeZero n
        e : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
        this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
        this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
        h : Not (Eq a 0)
        ⊢ Eq (HMul.hMul (HDiv.hDiv (e (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomia …
      -/
    · refine div_mul_cancel₀ _ (root_X_pow_sub_C_ne_zero' (NeZero.pos n) h)
      /-
        🎉 no goals
      -/


lemma autAdjoinRootXPowSubCEquiv_root [NeZero n] (η) :
    autAdjoinRootXPowSubCEquiv hζ H η (root _) = ((η : Kˣ) : K) • root _ :=
  autAdjoinRootXPowSubC_root a η


lemma autAdjoinRootXPowSubCEquiv_symm_smul [NeZero n] (σ) :
    ((autAdjoinRootXPowSubCEquiv hζ H).symm σ : Kˣ) • (root _ : K[n√a]) = σ (root _) := by
  /-
    K : Type u
    inst✝¹ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    inst✝ : NeZero n
    σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
    ⊢ Eq (HSMul.hSMul (↑((autAdjoinRootXPowSubCEquiv hζ H).symm σ)) (AdjoinRoot.ro …
  -/
  have := Fact.mk H
  simp only [autAdjoinRootXPowSubCEquiv, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe,
    MulEquiv.symm_mk, MulEquiv.coe_mk, Equiv.coe_fn_symm_mk, AdjoinRootXPowSubCEquivToRootsOfUnity,
    AdjoinRoot.algebraMap_eq, rootsOfUnity.mkOfPowEq, Units.smul_def, Algebra.smul_def,
    rootsOfUnityEquivOfPrimitiveRoots_symm_apply, Units.val_ofPowEqOne, ite_mul, one_mul]
  /-
    K : Type u
    inst✝¹ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    inst✝ : NeZero n
    σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
    this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
    ⊢ Eq (ite (Eq a 0) (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Pol …
  -/
  simp_rw [← root_X_pow_sub_C_eq_zero_iff H]
  /-
    K : Type u
    inst✝¹ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    inst✝ : NeZero n
    σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
    this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
    ⊢ Eq (ite (Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomi …
  -/
  split_ifs with h
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
      h : Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
      ⊢ Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))) …
    -/
  · rw [h, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u
      inst✝¹ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      inst✝ : NeZero n
      σ : AlgEquiv K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
      this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
      h : Not (Eq (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (HMul.hMul (HDiv.hDiv (σ (AdjoinRoot.root (HSub.hSub (HPow.hPow Polynomia …
    -/
  · rw [div_mul_cancel₀ _ h]
    /-
      🎉 no goals
    -/


include hζ in
lemma isSplittingField_AdjoinRoot_X_pow_sub_C :
    haveI := Fact.mk H
    letI : Algebra K K[n√a] := inferInstance
    IsSplittingField K K[n√a] (X ^ n - C a) := by
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    ⊢ Polynomial.IsSplittingField K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  have := Fact.mk H
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
    ⊢ Polynomial.IsSplittingField K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  letI : Algebra K K[n√a] := inferInstance
  /-
    K : Type u
    inst✝ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
    this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
    ⊢ Polynomial.IsSplittingField K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  constructor
  · rw [← splits_id_iff_splits, Polynomial.map_sub, Polynomial.map_pow, Polynomial.map_C,
      Polynomial.map_X]
    /-
      case splits'
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Polynomial.Splits (RingHom.id (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X …
    -/
    have ⟨_, hζ⟩ := hζ
    /-
      case splits'
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ✝ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      w✝ : K
      hζ : Membership.mem (primitiveRoots n K) w✝
      ⊢ Polynomial.Splits (RingHom.id (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X …
    -/
    rw [mem_primitiveRoots (Nat.pos_of_ne_zero <| ne_zero_of_irreducible_X_pow_sub_C H)] at hζ
    exact X_pow_sub_C_splits_of_isPrimitiveRoot (hζ.map_of_injective (algebraMap K _).injective)
      (root_X_pow_sub_C_pow n a)
    /-
      case adjoin_rootSet'
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ Eq (Algebra.adjoin K ((HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a) …
    -/
  · rw [eq_top_iff, ← AdjoinRoot.adjoinRoot_eq_top]
    /-
      case adjoin_rootSet'
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ LE.le (Algebra.adjoin K (Singleton.singleton (AdjoinRoot.root (HSub.hSub (HP …
    -/
    apply Algebra.adjoin_mono
    /-
      case adjoin_rootSet'.H
      K : Type u
      inst✝ : Field K
      n : Nat
      hζ : (primitiveRoots n K).Nonempty
      a : K
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Algebra K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial …
      ⊢ HasSubset.Subset (Singleton.singleton (AdjoinRoot.root (HSub.hSub (HPow.hPow …
    -/
    have := ne_zero_of_irreducible_X_pow_sub_C H
    rw [Set.singleton_subset_iff, mem_rootSet_of_ne (X_pow_sub_C_ne_zero
      (Nat.pos_of_ne_zero this) a), aeval_def, AdjoinRoot.algebraMap_eq, AdjoinRoot.eval₂_root]


/-- Suppose `L/K` is the splitting field of `Xⁿ - a`, then a choice of `ⁿ√a` gives an equivalence of
`L` with `K[n√a]`. -/
noncomputable
def adjoinRootXPowSubCEquiv (hζ : (primitiveRoots n K).Nonempty) (H : Irreducible (X ^ n - C a))
    (hα : α ^ n = algebraMap K L a) : K[n√a] ≃ₐ[K] L :=
                                                               /-
                                                                 K : Type u
                                                                 inst✝³ : Field K
                                                                 n : Nat
                                                                 hζ✝ : (primitiveRoots n K).Nonempty
                                                                 a : K
                                                                 H✝ : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
                                                                 L : Type u_1
                                                                 inst✝² : Field L
                                                                 inst✝¹ : Algebra K L
                                                                 inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
                                                                 α : L
                                                                 hα✝ : Eq (HPow.hPow α n) ((algebraMap K L) a)
                                                                 hζ : (primitiveRoots n K).Nonempty
                                                                 H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
                                                                 hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
                                                                 ⊢ Eq ((Polynomial.aeval α) (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C …
                                                               -/
  AlgEquiv.ofBijective (AdjoinRoot.liftHom (X ^ n - C a) α (by simp [hα])) <| by
                                                               /-
                                                                 🎉 no goals
                                                               -/
    /-
      K : Type u
      inst✝³ : Field K
      n : Nat
      hζ✝ : (primitiveRoots n K).Nonempty
      a : K
      H✝ : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
      α : L
      hα✝ : Eq (HPow.hPow α n) ((algebraMap K L) a)
      hζ : (primitiveRoots n K).Nonempty
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
      ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial.X n …
    -/
    haveI := Fact.mk H
    /-
      K : Type u
      inst✝³ : Field K
      n : Nat
      hζ✝ : (primitiveRoots n K).Nonempty
      a : K
      H✝ : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
      α : L
      hα✝ : Eq (HPow.hPow α n) ((algebraMap K L) a)
      hζ : (primitiveRoots n K).Nonempty
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
      this : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a …
      ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial.X n …
    -/
    letI := isSplittingField_AdjoinRoot_X_pow_sub_C hζ H
    /-
      K : Type u
      inst✝³ : Field K
      n : Nat
      hζ✝ : (primitiveRoots n K).Nonempty
      a : K
      H✝ : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
      α : L
      hα✝ : Eq (HPow.hPow α n) ((algebraMap K L) a)
      hζ : (primitiveRoots n K).Nonempty
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Polynomial.IsSplittingField K (AdjoinRoot (HSub.hSub (HPow.hPow Polynom …
      ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (HSub.hSub (HPow.hPow Polynomial.X n …
    -/
    refine ⟨(liftHom (X ^ n - C a) α _).injective, ?_⟩
    rw [← AlgHom.range_eq_top, ← IsSplittingField.adjoin_rootSet _ (X ^ n - C a),
      eq_comm, adjoin_rootSet_eq_range, IsSplittingField.adjoin_rootSet]
    /-
      case h
      K : Type u
      inst✝³ : Field K
      n : Nat
      hζ✝ : (primitiveRoots n K).Nonempty
      a : K
      H✝ : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
      α : L
      hα✝ : Eq (HPow.hPow α n) ((algebraMap K L) a)
      hζ : (primitiveRoots n K).Nonempty
      H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
      hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
      this✝ : Fact (Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C  …
      this : Polynomial.IsSplittingField K (AdjoinRoot (HSub.hSub (HPow.hPow Polynom …
      ⊢ Polynomial.Splits (algebraMap K (AdjoinRoot (HSub.hSub (HPow.hPow Polynomial …
    -/
    exact IsSplittingField.splits _ _
    /-
      🎉 no goals
    -/


lemma adjoinRootXPowSubCEquiv_root :
    adjoinRootXPowSubCEquiv hζ H hα (root _) = α := by
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    ⊢ Eq ((adjoinRootXPowSubCEquiv hζ H hα) (AdjoinRoot.root (HSub.hSub (HPow.hPow …
  -/
  rw [adjoinRootXPowSubCEquiv, AlgEquiv.coe_ofBijective, liftHom_root]
  /-
    🎉 no goals
  -/


lemma adjoinRootXPowSubCEquiv_symm_eq_root :
    (adjoinRootXPowSubCEquiv hζ H hα).symm α = root _ := by
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    ⊢ Eq ((adjoinRootXPowSubCEquiv hζ H hα).symm α) (AdjoinRoot.root (HSub.hSub (H …
  -/
  apply (adjoinRootXPowSubCEquiv hζ H hα).injective
  /-
    case a
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    ⊢ Eq ((adjoinRootXPowSubCEquiv hζ H hα) ((adjoinRootXPowSubCEquiv hζ H hα).sym …
  -/
  rw [(adjoinRootXPowSubCEquiv hζ H hα).apply_symm_apply, adjoinRootXPowSubCEquiv_root]
  /-
    🎉 no goals
  -/


include hζ H hα in
lemma Algebra.adjoin_root_eq_top_of_isSplittingField :
    Algebra.adjoin K {α} = ⊤ := by
  apply Subalgebra.map_injective (B := K[n√a]) (f := (adjoinRootXPowSubCEquiv hζ H hα).symm)
    (adjoinRootXPowSubCEquiv hζ H hα).symm.injective
  rw [Algebra.map_top, (AlgHom.range_eq_top _).mpr
    (adjoinRootXPowSubCEquiv hζ H hα).symm.surjective, AlgHom.map_adjoin,
    Set.image_singleton, AlgHom.coe_coe, adjoinRootXPowSubCEquiv_symm_eq_root, adjoinRoot_eq_top]


include hζ H hα in
lemma IntermediateField.adjoin_root_eq_top_of_isSplittingField :
    K⟮α⟯ = ⊤ := by
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    ⊢ Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
  -/
  refine (IntermediateField.eq_adjoin_of_eq_algebra_adjoin _ _ _ ?_).symm
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    ⊢ Eq Top.top.toSubalgebra (Algebra.adjoin K (Singleton.singleton α))
  -/
  exact (Algebra.adjoin_root_eq_top_of_isSplittingField hζ H hα).symm
  /-
    🎉 no goals
  -/


/-- An arbitrary choice of `ⁿ√a` in the splitting field of `Xⁿ - a`. -/
noncomputable
abbrev rootOfSplitsXPowSubC (hn : 0 < n) (a : K)
    (L) [Field L] [Algebra K L] [IsSplittingField K L (X ^ n - C a)] : L :=
  (rootOfSplits _ (IsSplittingField.splits L (X ^ n - C a))
          /-
            K : Type u
            inst✝⁶ : Field K
            n : Nat
            hζ : (primitiveRoots n K).Nonempty
            a✝ : K
            H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a✝))
            L✝ : Type u_1
            inst✝⁵ : Field L✝
            inst✝⁴ : Algebra K L✝
            inst✝³ : Polynomial.IsSplittingField K L✝ (HSub.hSub (HPow.hPow Polynomial.X n …
            α : L✝
            hα : Eq (HPow.hPow α n) ((algebraMap K L✝) a✝)
            hn : LT.lt 0 n
            a : K
            L : Type ?u.502984
            inst✝² : Field L
            inst✝¹ : Algebra K L
            inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
            ⊢ Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree 0
          -/
      (by simpa [degree_X_pow_sub_C hn] using Nat.pos_iff_ne_zero.mp hn))
          /-
            🎉 no goals
          -/


lemma rootOfSplitsXPowSubC_pow [NeZero n] :
    (rootOfSplitsXPowSubC (NeZero.pos n) a L) ^ n = algebraMap K L a := by
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    ⊢ Eq (HPow.hPow (rootOfSplitsXPowSubC ⋯ a L) n) ((algebraMap K L) a)
  -/
  have := map_rootOfSplits _ (IsSplittingField.splits L (X ^ n - C a))
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    this : ∀ (hfd : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).deg …
    ⊢ Eq (HPow.hPow (rootOfSplitsXPowSubC ⋯ a L) n) ((algebraMap K L) a)
  -/
  simp only [eval₂_sub, eval₂_X_pow, eval₂_C, sub_eq_zero] at this
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    this : ∀ (hfd : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).deg …
    ⊢ Eq (HPow.hPow (rootOfSplitsXPowSubC ⋯ a L) n) ((algebraMap K L) a)
  -/
  exact this _
  /-
    🎉 no goals
  -/


/-- Suppose `L/K` is the splitting field of `Xⁿ - a`, then `Gal(L/K)` is isomorphic to the
roots of unity in `K` if `K` contains all of them.
Note that this does not depend on a choice of `ⁿ√a`. -/
noncomputable
def autEquivRootsOfUnity [NeZero n] :
    (L ≃ₐ[K] L) ≃* (rootsOfUnity n K) :=
  (AlgEquiv.autCongr (adjoinRootXPowSubCEquiv hζ H (rootOfSplitsXPowSubC_pow a L)).symm).trans
    (autAdjoinRootXPowSubCEquiv hζ H).symm


lemma autEquivRootsOfUnity_apply_rootOfSplit [NeZero n] (σ : L ≃ₐ[K] L) :
    σ (rootOfSplitsXPowSubC (NeZero.pos n) a L) =
      autEquivRootsOfUnity hζ H L σ • (rootOfSplitsXPowSubC (NeZero.pos n) a L) := by
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ⊢ Eq (σ (rootOfSplitsXPowSubC ⋯ a L)) (HSMul.hSMul ((autEquivRootsOfUnity hζ H …
  -/
  obtain ⟨η, rfl⟩ := (autEquivRootsOfUnity hζ H L).symm.surjective σ
  /-
    case intro
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
    ⊢ Eq (((autEquivRootsOfUnity hζ H L).symm η) (rootOfSplitsXPowSubC ⋯ a L)) (HS …
  -/
  rw [MulEquiv.apply_symm_apply, autEquivRootsOfUnity]
  simp only [MulEquiv.symm_trans_apply, AlgEquiv.autCongr_symm, AlgEquiv.symm_symm,
    MulEquiv.symm_symm, AlgEquiv.autCongr_apply, AlgEquiv.trans_apply,
    adjoinRootXPowSubCEquiv_symm_eq_root, autAdjoinRootXPowSubCEquiv_root, map_smul,
    adjoinRootXPowSubCEquiv_root]
  /-
    case intro
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    η : Subtype fun x => Membership.mem (rootsOfUnity n K) x
    ⊢ Eq (HSMul.hSMul (↑↑η) (rootOfSplitsXPowSubC ⋯ a L)) (HSMul.hSMul η (rootOfSp …
  -/
  rfl
  /-
    🎉 no goals
  -/


include hα in
lemma autEquivRootsOfUnity_smul [NeZero n] (σ : L ≃ₐ[K] L) :
    autEquivRootsOfUnity hζ H L σ • α = σ α := by
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity hζ H L) σ) α) (σ α)
  -/
  have ⟨ζ, hζ'⟩ := hζ
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ζ : K
    hζ' : Membership.mem (primitiveRoots n K) ζ
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) σ) α) (σ α)
  -/
  have hn := NeZero.pos n
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ζ : K
    hζ' : Membership.mem (primitiveRoots n K) ζ
    hn : LT.lt 0 n
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) σ) α) (σ α)
  -/
  rw [mem_primitiveRoots hn] at hζ'
  rw [← mem_nthRoots hn, (hζ'.map_of_injective (algebraMap K L).injective).nthRoots_eq
    (rootOfSplitsXPowSubC_pow a L)] at hα
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ζ : K
    hα : Membership.mem (Multiset.map (fun x => HMul.hMul (HPow.hPow ((algebraMap  …
    hζ'✝ : Membership.mem (primitiveRoots n K) ζ
    hζ' : IsPrimitiveRoot ζ n
    hn : LT.lt 0 n
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) σ) α) (σ α)
  -/
  simp only [Finset.range_val, Multiset.mem_map, Multiset.mem_range] at hα
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ζ : K
    hζ'✝ : Membership.mem (primitiveRoots n K) ζ
    hζ' : IsPrimitiveRoot ζ n
    hn : LT.lt 0 n
    hα : Exists fun a_1 => And (LT.lt a_1 n) (Eq (HMul.hMul (HPow.hPow ((algebraMa …
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) σ) α) (σ α)
  -/
  obtain ⟨i, _, rfl⟩ := hα
  simp only [map_mul, ← map_pow, ← Algebra.smul_def, map_smul,
    autEquivRootsOfUnity_apply_rootOfSplit hζ H L]
  /-
    case intro.intro
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    inst✝ : NeZero n
    σ : AlgEquiv K L L
    ζ : K
    hζ'✝ : Membership.mem (primitiveRoots n K) ζ
    hζ' : IsPrimitiveRoot ζ n
    hn : LT.lt 0 n
    i : Nat
    left✝ : LT.lt i n
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) σ) (HSMul.hSMul (HPow.hPow ζ i …
  -/
  exact smul_comm _ _ _
  /-
    🎉 no goals
  -/


/-- Suppose `L/K` is the splitting field of `Xⁿ - a`, and `ζ` is a `n`-th primitive root of unity
in `K`, then `Gal(L/K)` is isomorphic to `ZMod n`. -/
noncomputable
def autEquivZmod [NeZero n] {ζ : K} (hζ : IsPrimitiveRoot ζ n) :
    (L ≃ₐ[K] L) ≃* Multiplicative (ZMod n) :=
  haveI hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  (autEquivRootsOfUnity ⟨ζ, (mem_primitiveRoots hn).mpr hζ⟩ H L).trans
    ((MulEquiv.subgroupCongr (IsPrimitiveRoot.zpowers_eq
      (hζ.isUnit_unit' hn)).symm).trans (AddEquiv.toMultiplicative'
        (hζ.isUnit_unit' hn).zmodEquivZPowers.symm))


include hα in
lemma autEquivZmod_symm_apply_intCast [NeZero n] {ζ : K} (hζ : IsPrimitiveRoot ζ n) (m : ℤ) :
    (autEquivZmod H L hζ).symm (Multiplicative.ofAdd (m : ZMod n)) α = ζ ^ m • α := by
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    m : Int
    ⊢ Eq (((autEquivZmod H L hζ).symm (Multiplicative.ofAdd ↑m)) α) (HSMul.hSMul ( …
  -/
  have hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    m : Int
    hn : LT.lt 0 n
    ⊢ Eq (((autEquivZmod H L hζ).symm (Multiplicative.ofAdd ↑m)) α) (HSMul.hSMul ( …
  -/
  rw [← autEquivRootsOfUnity_smul ⟨ζ, (mem_primitiveRoots hn).mpr hζ⟩ H L hα]
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    m : Int
    hn : LT.lt 0 n
    ⊢ Eq (HSMul.hSMul ((autEquivRootsOfUnity ⋯ H L) ((autEquivZmod H L hζ).symm (M …
  -/
  simp [MulEquiv.subgroupCongr_symm_apply, Subgroup.smul_def, Units.smul_def, autEquivZmod]
  /-
    🎉 no goals
  -/


include hα in
lemma autEquivZmod_symm_apply_natCast [NeZero n] {ζ : K} (hζ : IsPrimitiveRoot ζ n) (m : ℕ) :
    (autEquivZmod H L hζ).symm (Multiplicative.ofAdd (m : ZMod n)) α = ζ ^ m • α := by
  /-
    K : Type u
    inst✝⁴ : Field K
    n : Nat
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n) …
    α : L
    hα : Eq (HPow.hPow α n) ((algebraMap K L) a)
    inst✝ : NeZero n
    ζ : K
    hζ : IsPrimitiveRoot ζ n
    m : Nat
    ⊢ Eq (((autEquivZmod H L hζ).symm (Multiplicative.ofAdd ↑m)) α) (HSMul.hSMul ( …
  -/
  simpa only [Int.cast_natCast, zpow_natCast] using autEquivZmod_symm_apply_intCast H L hα hζ m
  /-
    🎉 no goals
  -/


include hζ H in
lemma isCyclic_of_isSplittingField_X_pow_sub_C [NeZero n] : IsCyclic (L ≃ₐ[K] L) :=
  have hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  isCyclic_of_surjective _
    (autEquivZmod H _ <| (mem_primitiveRoots hn).mp hζ.choose_spec).symm.surjective


include hζ H in
lemma isGalois_of_isSplittingField_X_pow_sub_C : IsGalois K L :=
  IsGalois.of_separable_splitting_field (separable_X_pow_sub_C_of_irreducible hζ a H)


include hζ H in
lemma finrank_of_isSplittingField_X_pow_sub_C : Module.finrank K L = n := by
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    ⊢ Eq (Module.finrank K L) n
  -/
  have := Polynomial.IsSplittingField.finiteDimensional L (X ^ n - C a)
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    this : FiniteDimensional K L
    ⊢ Eq (Module.finrank K L) n
  -/
  have := isGalois_of_isSplittingField_X_pow_sub_C hζ H L
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    ⊢ Eq (Module.finrank K L) n
  -/
  have hn := Nat.pos_iff_ne_zero.mpr (ne_zero_of_irreducible_X_pow_sub_C H)
  /-
    K : Type u
    inst✝³ : Field K
    n : Nat
    hζ : (primitiveRoots n K).Nonempty
    a : K
    H : Irreducible (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a))
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X n)  …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    hn : LT.lt 0 n
    ⊢ Eq (Module.finrank K L) n
  -/
  have : NeZero n := ⟨ne_zero_of_irreducible_X_pow_sub_C H⟩
  rw [← IsGalois.card_aut_eq_finrank, Fintype.card_congr ((autEquivZmod H L <|
    (mem_primitiveRoots hn).mp hζ.choose_spec).toEquiv.trans Multiplicative.toAdd), ZMod.card]


include hK in
/-- If `L/K` is a cyclic extension of degree `n`, and `K` contains all `n`-th roots of unity,
then `L = K[α]` for some `α ^ n ∈ K`. -/
lemma exists_root_adjoin_eq_top_of_isCyclic [IsGalois K L] [IsCyclic (L ≃ₐ[K] L)] :
    ∃ (α : L), α ^ (finrank K L) ∈ Set.range (algebraMap K L) ∧ K⟮α⟯ = ⊤ := by
  -- Let `ζ` be an `n`-th root of unity, and `σ` be a generator of `L ≃ₐ[K] L`.
  /-
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  have ⟨ζ, hζ⟩ := hK
  /-
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : Membership.mem (primitiveRoots (Module.finrank K L) K) ζ
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  rw [mem_primitiveRoots finrank_pos] at hζ
  /-
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  obtain ⟨σ, hσ⟩ := ‹IsCyclic (L ≃ₐ[K] L)›
  /-
    case mk.intro
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    σ : AlgEquiv K L L
    hσ : Function.Surjective fun x => HPow.hPow σ x
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  have hσ' := orderOf_eq_card_of_forall_mem_zpowers hσ
  -- Since the minimal polynomial of `σ` over `K` is `Xⁿ - 1`,
  -- `σ` has an eigenvector `v` with eigenvalue `ζ`.
  have : IsRoot (minpoly K σ.toLinearMap) ζ := by
    simpa [minpoly_algEquiv_toLinearMap σ (isOfFinOrder_of_finite σ), hσ',
      sub_eq_zero, IsGalois.card_aut_eq_finrank] using hζ.pow_eq_one
  /-
    case mk.intro
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    σ : AlgEquiv K L L
    hσ : Function.Surjective fun x => HPow.hPow σ x
    hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
    this : (minpoly K σ.toLinearMap).IsRoot ζ
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  obtain ⟨v, hv⟩ := (Module.End.hasEigenvalue_of_isRoot this).exists_hasEigenvector
  /-
    case mk.intro.intro
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    σ : AlgEquiv K L L
    hσ : Function.Surjective fun x => HPow.hPow σ x
    hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
    this : (minpoly K σ.toLinearMap).IsRoot ζ
    v : L
    hv : Module.End.HasEigenvector σ.toLinearMap ζ v
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  have hv' := hv.pow_apply
  /-
    case mk.intro.intro
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    σ : AlgEquiv K L L
    hσ : Function.Surjective fun x => HPow.hPow σ x
    hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
    this : (minpoly K σ.toLinearMap).IsRoot ζ
    v : L
    hv : Module.End.HasEigenvector σ.toLinearMap ζ v
    hv' : ∀ (n : Nat), Eq ((HPow.hPow σ.toLinearMap n) v) (HSMul.hSMul (HPow.hPow  …
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  simp_rw [← AlgEquiv.pow_toLinearMap, AlgEquiv.toLinearMap_apply] at hv'
  -- We claim that `v` is the desired root.
  /-
    case mk.intro.intro
    K : Type u
    inst✝⁵ : Field K
    L : Type u_1
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    inst✝¹ : IsGalois K L
    inst✝ : IsCyclic (AlgEquiv K L L)
    ζ : K
    hζ : IsPrimitiveRoot ζ (Module.finrank K L)
    σ : AlgEquiv K L L
    hσ : Function.Surjective fun x => HPow.hPow σ x
    hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
    this : (minpoly K σ.toLinearMap).IsRoot ζ
    v : L
    hv : Module.End.HasEigenvector σ.toLinearMap ζ v
    hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
    ⊢ Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L)) (HPow.hPow …
  -/
  refine ⟨v, ?_, ?_⟩
  · -- Since `v ^ n` is fixed by `σ` (`σ (v ^ n) = ζ ^ n • v ^ n = v ^ n`), it is in `K`.
    rw [← IntermediateField.mem_bot,
      ← OrderIso.map_bot IsGalois.intermediateFieldEquivSubgroup.symm]
    /-
      case mk.intro.intro.refine_1
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      ⊢ Membership.mem (IsGalois.intermediateFieldEquivSubgroup.symm Bot.bot) (HPow. …
    -/
    intro ⟨σ', hσ'⟩
    /-
      case mk.intro.intro.refine_1
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      σ' : AlgEquiv K L L
      hσ' : Membership.mem Bot.bot σ'
      ⊢ Eq (HSMul.hSMul ⟨σ', hσ'⟩ (HPow.hPow v (Module.finrank K L))) (HPow.hPow v ( …
    -/
    obtain ⟨n, rfl : σ ^ n = σ'⟩ := mem_powers_iff_mem_zpowers.mpr (hσ σ')
    rw [smul_pow', Submonoid.smul_def, AlgEquiv.smul_def, hv', smul_pow, ← pow_mul,
      mul_comm, pow_mul, hζ.pow_eq_one, one_pow, one_smul]
  · -- Since `σ` does not fix `K⟮α⟯`, `K⟮α⟯` is `L`.
    /-
      case mk.intro.intro.refine_2
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      ⊢ Eq (IntermediateField.adjoin K (Singleton.singleton v)) Top.top
    -/
    apply IsGalois.intermediateFieldEquivSubgroup.injective
    /-
      case mk.intro.intro.refine_2.a
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      ⊢ Eq (IsGalois.intermediateFieldEquivSubgroup (IntermediateField.adjoin K (Sin …
    -/
    rw [map_top, eq_top_iff]
    /-
      case mk.intro.intro.refine_2.a
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ' : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      ⊢ LE.le Top.top (IsGalois.intermediateFieldEquivSubgroup (IntermediateField.ad …
    -/
    intros σ' hσ'
    /-
      case mk.intro.intro.refine_2.a
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      σ' : AlgEquiv K L L
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      ⊢ Membership.mem Top.top σ'
    -/
    obtain ⟨n, rfl : σ ^ n = σ'⟩ := mem_powers_iff_mem_zpowers.mpr (hσ σ')
    /-
      case mk.intro.intro.refine_2.a.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      n : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      ⊢ Membership.mem Top.top (HPow.hPow σ n)
    -/
    have := hσ' ⟨v, IntermediateField.mem_adjoin_simple_self K v⟩
    /-
      case mk.intro.intro.refine_2.a.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this✝ : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      n : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      this : Eq (HSMul.hSMul (HPow.hPow σ n) ↑⟨v, ⋯⟩) ↑⟨v, ⋯⟩
      ⊢ Membership.mem Top.top (HPow.hPow σ n)
    -/
    simp only [AlgEquiv.smul_def, hv'] at this
    /-
      case mk.intro.intro.refine_2.a.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this✝ : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      n : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      this : Eq (HSMul.hSMul (HPow.hPow ζ n) v) v
      ⊢ Membership.mem Top.top (HPow.hPow σ n)
    -/
    conv_rhs at this => rw [← one_smul K v]
    /-
      case mk.intro.intro.refine_2.a.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this✝ : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      n : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      this : Eq (HSMul.hSMul (HPow.hPow ζ n) v) (HSMul.hSMul 1 v)
      ⊢ Membership.mem Top.top (HPow.hPow σ n)
    -/
    obtain ⟨k, rfl⟩ := hζ.dvd_of_pow_eq_one n (smul_left_injective K hv.2 this)
    /-
      case mk.intro.intro.refine_2.a.intro.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this✝ : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      k : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      this : Eq (HSMul.hSMul (HPow.hPow ζ (HMul.hMul (Module.finrank K L) k)) v) (HS …
      ⊢ Membership.mem Top.top (HPow.hPow σ (HMul.hMul (Module.finrank K L) k))
    -/
    rw [pow_mul, ← IsGalois.card_aut_eq_finrank, pow_card_eq_one, one_pow]
    /-
      case mk.intro.intro.refine_2.a.intro.intro
      K : Type u
      inst✝⁵ : Field K
      L : Type u_1
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      inst✝¹ : IsGalois K L
      inst✝ : IsCyclic (AlgEquiv K L L)
      ζ : K
      hζ : IsPrimitiveRoot ζ (Module.finrank K L)
      σ : AlgEquiv K L L
      hσ : Function.Surjective fun x => HPow.hPow σ x
      hσ'✝ : Eq (orderOf σ) (Nat.card (AlgEquiv K L L))
      this✝ : (minpoly K σ.toLinearMap).IsRoot ζ
      v : L
      hv : Module.End.HasEigenvector σ.toLinearMap ζ v
      hv' : ∀ (n : Nat), Eq ((HPow.hPow σ n) v) (HSMul.hSMul (HPow.hPow ζ n) v)
      k : Nat
      hσ' : Membership.mem (IsGalois.intermediateFieldEquivSubgroup (IntermediateFie …
      this : Eq (HSMul.hSMul (HPow.hPow ζ (HMul.hMul (Module.finrank K L) k)) v) (HS …
      ⊢ Membership.mem Top.top 1
    -/
    exact one_mem _
    /-
      🎉 no goals
    -/


lemma irreducible_X_pow_sub_C_of_root_adjoin_eq_top
    {a : K} {α : L} (ha : α ^ (finrank K L) = algebraMap K L a) (hα : K⟮α⟯ = ⊤) :
    Irreducible (X ^ (finrank K L) - C a) := by
  have : X ^ (finrank K L) - C a = minpoly K α := by
    refine minpoly.unique _ _ (monic_X_pow_sub_C _ finrank_pos.ne.symm) ?_ ?_
    · simp only [aeval_def, eval₂_sub, eval₂_X_pow, ha, eval₂_C, sub_self]
    · intros q hq hq'
      refine le_trans ?_ (degree_le_of_dvd (minpoly.dvd _ _ hq') hq.ne_zero)
      rw [degree_X_pow_sub_C finrank_pos,
        degree_eq_natDegree (minpoly.ne_zero (IsIntegral.of_finite K α)),
        ← IntermediateField.adjoin.finrank (IsIntegral.of_finite K α), hα, Nat.cast_le]
      exact (finrank_top K L).ge
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    a : K
    α : L
    ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
    hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
    this : Eq (HSub.hSub (HPow.hPow Polynomial.X (Module.finrank K L)) (Polynomial …
    ⊢ Irreducible (HSub.hSub (HPow.hPow Polynomial.X (Module.finrank K L)) (Polyno …
  -/
  exact this ▸ minpoly.irreducible (IsIntegral.of_finite K α)
  /-
    🎉 no goals
  -/


include hK in
lemma isSplittingField_X_pow_sub_C_of_root_adjoin_eq_top
    {a : K} {α : L} (ha : α ^ (finrank K L) = algebraMap K L a) (hα : K⟮α⟯ = ⊤) :
    IsSplittingField K L (X ^ (finrank K L) - C a) := by
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u_1
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    a : K
    α : L
    ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
    hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
    ⊢ Polynomial.IsSplittingField K L (HSub.hSub (HPow.hPow Polynomial.X (Module.f …
  -/
  constructor
  · rw [← splits_id_iff_splits, Polynomial.map_sub, Polynomial.map_pow, Polynomial.map_C,
      Polynomial.map_X]
    /-
      case splits'
      K : Type u
      inst✝³ : Field K
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      a : K
      α : L
      ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
      hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
      ⊢ Polynomial.Splits (RingHom.id L) (HSub.hSub (HPow.hPow Polynomial.X (Module. …
    -/
    have ⟨_, hζ⟩ := hK
    /-
      case splits'
      K : Type u
      inst✝³ : Field K
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      a : K
      α : L
      ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
      hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
      w✝ : K
      hζ : Membership.mem (primitiveRoots (Module.finrank K L) K) w✝
      ⊢ Polynomial.Splits (RingHom.id L) (HSub.hSub (HPow.hPow Polynomial.X (Module. …
    -/
    rw [mem_primitiveRoots finrank_pos] at hζ
    /-
      case splits'
      K : Type u
      inst✝³ : Field K
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      a : K
      α : L
      ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
      hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
      w✝ : K
      hζ : IsPrimitiveRoot w✝ (Module.finrank K L)
      ⊢ Polynomial.Splits (RingHom.id L) (HSub.hSub (HPow.hPow Polynomial.X (Module. …
    -/
    exact X_pow_sub_C_splits_of_isPrimitiveRoot (hζ.map_of_injective (algebraMap K _).injective) ha
    /-
      🎉 no goals
    -/
  · rw [eq_top_iff, ← IntermediateField.top_toSubalgebra, ← hα,
      IntermediateField.adjoin_simple_toSubalgebra_of_integral (IsIntegral.of_finite K α)]
    /-
      case adjoin_rootSet'
      K : Type u
      inst✝³ : Field K
      L : Type u_1
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      hK : (primitiveRoots (Module.finrank K L) K).Nonempty
      a : K
      α : L
      ha : Eq (HPow.hPow α (Module.finrank K L)) ((algebraMap K L) a)
      hα : Eq (IntermediateField.adjoin K (Singleton.singleton α)) Top.top
      ⊢ LE.le (Algebra.adjoin K (Singleton.singleton α)) (Algebra.adjoin K ((HSub.hS …
    -/
    apply Algebra.adjoin_mono
    rw [Set.singleton_subset_iff, mem_rootSet_of_ne (X_pow_sub_C_ne_zero finrank_pos a),
      aeval_def, eval₂_sub, eval₂_X_pow, eval₂_C, ha, sub_self]


open Module in
/--
Suppose `L/K` is a finite extension of dimension `n`, and `K` contains all `n`-th roots of unity.
Then `L/K` is cyclic iff
`L` is a splitting field of some irreducible polynomial of the form `Xⁿ - a : K[X]` iff
`L = K[α]` for some `αⁿ ∈ K`.
-/
lemma isCyclic_tfae (K L) [Field K] [Field L] [Algebra K L] [FiniteDimensional K L]
    (hK : (primitiveRoots (Module.finrank K L) K).Nonempty) :
    List.TFAE [
      IsGalois K L ∧ IsCyclic (L ≃ₐ[K] L),
      ∃ a : K, Irreducible (X ^ (finrank K L) - C a) ∧
        IsSplittingField K L (X ^ (finrank K L) - C a),
      ∃ (α : L), α ^ (finrank K L) ∈ Set.range (algebraMap K L) ∧ K⟮α⟯ = ⊤] := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    ⊢ (List.cons (And (IsGalois K L) (IsCyclic (AlgEquiv K L L))) (List.cons (Exis …
  -/
  have : NeZero (Module.finrank K L) := NeZero.of_pos finrank_pos
  tfae_have 1 → 3
  | ⟨inst₁, inst₂⟩ => exists_root_adjoin_eq_top_of_isCyclic K L hK
  tfae_have 3 → 2
  | ⟨α, ⟨a, ha⟩, hα⟩ => ⟨a, irreducible_X_pow_sub_C_of_root_adjoin_eq_top ha.symm hα,
      isSplittingField_X_pow_sub_C_of_root_adjoin_eq_top hK ha.symm hα⟩
  tfae_have 2 → 1
  | ⟨a, H, inst⟩ => ⟨isGalois_of_isSplittingField_X_pow_sub_C hK H L,
      isCyclic_of_isSplittingField_X_pow_sub_C hK H L⟩
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    hK : (primitiveRoots (Module.finrank K L) K).Nonempty
    this : NeZero (Module.finrank K L)
    tfae_1_to_3 : And (IsGalois K L) (IsCyclic (AlgEquiv K L L)) → Exists fun α => …
    tfae_3_to_2 : (Exists fun α => And (Membership.mem (Set.range ⇑(algebraMap K L …
    tfae_2_to_1 : (Exists fun a => And (Irreducible (HSub.hSub (HPow.hPow Polynomi …
    ⊢ (List.cons (And (IsGalois K L) (IsCyclic (AlgEquiv K L L))) (List.cons (Exis …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/

