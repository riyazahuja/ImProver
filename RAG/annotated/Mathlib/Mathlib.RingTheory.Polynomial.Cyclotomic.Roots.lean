theorem isRoot_of_unity_of_root_cyclotomic {ζ : R} {i : ℕ} (hi : i ∈ n.divisors)
    (h : (cyclotomic i R).IsRoot ζ) : ζ ^ n = 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    ⊢ Eq (HPow.hPow ζ n) 1
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      R : Type u_1
      inst✝ : CommRing R
      ζ : R
      i : Nat
      h : (Polynomial.cyclotomic i R).IsRoot ζ
      hi : Membership.mem (Nat.divisors 0) i
      ⊢ Eq (HPow.hPow ζ 0) 1
    -/
  · exact pow_zero _
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    ⊢ Eq (HPow.hPow ζ n) 1
  -/
  have := congr_arg (eval ζ) (prod_cyclotomic_eq_X_pow_sub_one hn R).symm
  /-
    case inr
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    this : Eq (Polynomial.eval ζ (HSub.hSub (HPow.hPow Polynomial.X n) 1)) (Polyno …
    ⊢ Eq (HPow.hPow ζ n) 1
  -/
  rw [eval_sub, eval_pow, eval_X, eval_one] at this
  /-
    case inr
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    this : Eq (HSub.hSub (HPow.hPow ζ n) 1) (Polynomial.eval ζ (n.divisors.prod fu …
    ⊢ Eq (HPow.hPow ζ n) 1
  -/
  convert eq_add_of_sub_eq' this
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    this : Eq (HSub.hSub (HPow.hPow ζ n) 1) (Polynomial.eval ζ (n.divisors.prod fu …
    ⊢ Eq 1 (HAdd.hAdd 1 (Polynomial.eval ζ (n.divisors.prod fun i => Polynomial.cy …
  -/
  convert (add_zero (M := R) _).symm
  /-
    case h.e'_3.h.e'_6
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    this : Eq (HSub.hSub (HPow.hPow ζ n) 1) (Polynomial.eval ζ (n.divisors.prod fu …
    ⊢ Eq (Polynomial.eval ζ (n.divisors.prod fun i => Polynomial.cyclotomic i R)) 0
  -/
  apply eval_eq_zero_of_dvd_of_eval_eq_zero _ h
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ζ : R
    i : Nat
    hi : Membership.mem n.divisors i
    h : (Polynomial.cyclotomic i R).IsRoot ζ
    hn : GT.gt n 0
    this : Eq (HSub.hSub (HPow.hPow ζ n) 1) (Polynomial.eval ζ (n.divisors.prod fu …
    ⊢ Dvd.dvd (Polynomial.cyclotomic i R) (n.divisors.prod fun i => Polynomial.cyc …
  -/
  exact Finset.dvd_prod_of_mem _ hi
  /-
    🎉 no goals
  -/


theorem _root_.isRoot_of_unity_iff (h : 0 < n) (R : Type*) [CommRing R] [IsDomain R] {ζ : R} :
    ζ ^ n = 1 ↔ ∃ i ∈ n.divisors, (cyclotomic i R).IsRoot ζ := by
  rw [← mem_nthRoots h, nthRoots, mem_roots <| X_pow_sub_C_ne_zero h _, C_1, ←
      prod_cyclotomic_eq_X_pow_sub_one h, isRoot_prod]


/-- Any `n`-th primitive root of unity is a root of `cyclotomic n R`. -/
theorem _root_.IsPrimitiveRoot.isRoot_cyclotomic (hpos : 0 < n) {μ : R} (h : IsPrimitiveRoot μ n) :
    IsRoot (cyclotomic n R) μ := by
  rw [← mem_roots (cyclotomic_ne_zero n R), cyclotomic_eq_prod_X_sub_primitiveRoots h,
    roots_prod_X_sub_C, ← Finset.mem_def]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    n : Nat
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    μ : R
    h : IsPrimitiveRoot μ n
    ⊢ Membership.mem (primitiveRoots n R) μ
  -/
  rwa [← mem_primitiveRoots hpos] at h
  /-
    🎉 no goals
  -/


private theorem isRoot_cyclotomic_iff' {n : ℕ} {K : Type*} [Field K] {μ : K} [NeZero (n : K)] :
    IsRoot (cyclotomic n K) μ ↔ IsPrimitiveRoot μ n := by
  -- in this proof, `o` stands for `orderOf μ`
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    ⊢ Iff ((Polynomial.cyclotomic n K).IsRoot μ) (IsPrimitiveRoot μ n)
  -/
  have hnpos : 0 < n := (NeZero.of_neZero_natCast K).out.bot_lt
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    ⊢ Iff ((Polynomial.cyclotomic n K).IsRoot μ) (IsPrimitiveRoot μ n)
  -/
  refine ⟨fun hμ => ?_, IsPrimitiveRoot.isRoot_cyclotomic hnpos⟩
  have hμn : μ ^ n = 1 := by
    rw [isRoot_of_unity_iff hnpos _]
    exact ⟨n, n.mem_divisors_self hnpos.ne', hμ⟩
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    ⊢ IsPrimitiveRoot μ n
  -/
  by_contra hnμ
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ⊢ False
  -/
  have ho : 0 < orderOf μ := (isOfFinOrder_iff_pow_eq_one.2 <| ⟨n, hnpos, hμn⟩).orderOf_pos
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ho : LT.lt 0 (orderOf μ)
    ⊢ False
  -/
  have := pow_orderOf_eq_one μ
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ho : LT.lt 0 (orderOf μ)
    this : Eq (HPow.hPow μ (orderOf μ)) 1
    ⊢ False
  -/
  rw [isRoot_of_unity_iff ho] at this
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ho : LT.lt 0 (orderOf μ)
    this : Exists fun i => And (Membership.mem (orderOf μ).divisors i) ((Polynomia …
    ⊢ False
  -/
  obtain ⟨i, hio, hiμ⟩ := this
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Membership.mem (orderOf μ).divisors i
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    ⊢ False
  -/
  replace hio := Nat.dvd_of_mem_divisors hio
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Not (IsPrimitiveRoot μ n)
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    hio : Dvd.dvd i (orderOf μ)
    ⊢ False
  -/
  rw [IsPrimitiveRoot.not_iff] at hnμ
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Eq (HPow.hPow μ n) 1
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    hio : Dvd.dvd i (orderOf μ)
    ⊢ False
  -/
  rw [← orderOf_dvd_iff_pow_eq_one] at hμn
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    hio : Dvd.dvd i (orderOf μ)
    ⊢ False
  -/
  have key : i < n := (Nat.le_of_dvd ho hio).trans_lt ((Nat.le_of_dvd hnpos hμn).lt_of_ne hnμ)
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    ⊢ False
  -/
  have key' : i ∣ n := hio.trans hμn
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : (Polynomial.cyclotomic n K).IsRoot μ
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : (Polynomial.cyclotomic i K).IsRoot μ
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    ⊢ False
  -/
  rw [← Polynomial.dvd_iff_isRoot] at hμ hiμ
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C μ)) (Polynomial.cyclotomic  …
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C μ)) (Polynomial.cyclotomic …
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    ⊢ False
  -/
  have hni : {i, n} ⊆ n.divisors := by simpa [Finset.insert_subset_iff, key'] using hnpos.ne'
  /-
    case intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C μ)) (Polynomial.cyclotomic  …
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hiμ : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C μ)) (Polynomial.cyclotomic …
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    ⊢ False
  -/
  obtain ⟨k, hk⟩ := hiμ
  /-
    case intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμ : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C μ)) (Polynomial.cyclotomic  …
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    ⊢ False
  -/
  obtain ⟨j, hj⟩ := hμ
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    ⊢ False
  -/
  have := prod_cyclotomic_eq_X_pow_sub_one hnpos K
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    this : Eq (n.divisors.prod fun i => Polynomial.cyclotomic i K) (HSub.hSub (HPo …
    ⊢ False
  -/
  rw [← Finset.prod_sdiff hni, Finset.prod_pair key.ne, hk, hj] at this
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    this : Eq (HMul.hMul ((SDiff.sdiff n.divisors (Insert.insert i (Singleton.sing …
    ⊢ False
  -/
  have hn := (X_pow_sub_one_separable_iff.mpr <| NeZero.natCast_ne n K).squarefree
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    this : Eq (HMul.hMul ((SDiff.sdiff n.divisors (Insert.insert i (Singleton.sing …
    hn : Squarefree (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ⊢ False
  -/
  rw [← this, Squarefree] at hn
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    this : Eq (HMul.hMul ((SDiff.sdiff n.divisors (Insert.insert i (Singleton.sing …
    hn : ∀ (x : Polynomial K), Dvd.dvd (HMul.hMul x x) (HMul.hMul ((SDiff.sdiff n. …
    ⊢ False
  -/
  specialize hn (X - C μ) ⟨(∏ x ∈ n.divisors \ {i, n}, cyclotomic x K) * k * j, by ring⟩
  /-
    case intro.intro.intro.intro
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    inst✝ : NeZero ↑n
    hnpos : LT.lt 0 n
    hμn : Dvd.dvd (orderOf μ) n
    hnμ : Ne (orderOf μ) n
    ho : LT.lt 0 (orderOf μ)
    i : Nat
    hio : Dvd.dvd i (orderOf μ)
    key : LT.lt i n
    key' : Dvd.dvd i n
    hni : HasSubset.Subset (Insert.insert i (Singleton.singleton n)) n.divisors
    k : Polynomial K
    hk : Eq (Polynomial.cyclotomic i K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    j : Polynomial K
    hj : Eq (Polynomial.cyclotomic n K) (HMul.hMul (HSub.hSub Polynomial.X (Polyno …
    this : Eq (HMul.hMul ((SDiff.sdiff n.divisors (Insert.insert i (Singleton.sing …
    hn : IsUnit (HSub.hSub Polynomial.X (Polynomial.C μ))
    ⊢ False
  -/
  simp [Polynomial.isUnit_iff_degree_eq_zero] at hn
  /-
    🎉 no goals
  -/


theorem isRoot_cyclotomic_iff [NeZero (n : R)] {μ : R} :
    IsRoot (cyclotomic n R) μ ↔ IsPrimitiveRoot μ n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    μ : R
    ⊢ Iff ((Polynomial.cyclotomic n R).IsRoot μ) (IsPrimitiveRoot μ n)
  -/
  have hf : Function.Injective _ := IsFractionRing.injective R (FractionRing R)
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    μ : R
    hf : Function.Injective ⇑(algebraMap R (FractionRing R))
    ⊢ Iff ((Polynomial.cyclotomic n R).IsRoot μ) (IsPrimitiveRoot μ n)
  -/
  haveI : NeZero (n : FractionRing R) := NeZero.nat_of_injective hf
  rw [← isRoot_map_iff hf, ← IsPrimitiveRoot.map_iff_of_injective hf, map_cyclotomic, ←
    isRoot_cyclotomic_iff']


theorem roots_cyclotomic_nodup [NeZero (n : R)] : (cyclotomic n R).roots.Nodup := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    ⊢ (Polynomial.cyclotomic n R).roots.Nodup
  -/
  obtain h | ⟨ζ, hζ⟩ := (cyclotomic n R).roots.empty_or_exists_mem
    /-
      case inl
      R : Type u_1
      inst✝² : CommRing R
      n : Nat
      inst✝¹ : IsDomain R
      inst✝ : NeZero ↑n
      h : Eq (Polynomial.cyclotomic n R).roots 0
      ⊢ (Polynomial.cyclotomic n R).roots.Nodup
    -/
  · exact h.symm ▸ Multiset.nodup_zero
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    ζ : R
    hζ : Membership.mem (Polynomial.cyclotomic n R).roots ζ
    ⊢ (Polynomial.cyclotomic n R).roots.Nodup
  -/
  rw [mem_roots <| cyclotomic_ne_zero n R, isRoot_cyclotomic_iff] at hζ
  refine Multiset.nodup_of_le
    (roots.le_of_dvd (X_pow_sub_C_ne_zero (NeZero.pos_of_neZero_natCast R) 1) <|
      cyclotomic.dvd_X_pow_sub_one n R) hζ.nthRoots_one_nodup


theorem cyclotomic.roots_to_finset_eq_primitiveRoots [NeZero (n : R)] :
    (⟨(cyclotomic n R).roots, roots_cyclotomic_nodup⟩ : Finset _) = primitiveRoots n R := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    ⊢ Eq { val := (Polynomial.cyclotomic n R).roots, nodup := ⋯ } (primitiveRoots  …
  -/
  ext a
  -- Porting note: was
  -- `simp [cyclotomic_ne_zero n R, isRoot_cyclotomic_iff, mem_primitiveRoots,`
  -- `  NeZero.pos_of_neZero_natCast R]`
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    a : R
    ⊢ Iff (Membership.mem { val := (Polynomial.cyclotomic n R).roots, nodup := ⋯ } …
  -/
  simp only [mem_primitiveRoots, NeZero.pos_of_neZero_natCast R]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    a : R
    ⊢ Iff (Membership.mem { val := (Polynomial.cyclotomic n R).roots, nodup := ⋯ } …
  -/
  convert isRoot_cyclotomic_iff (n := n) (μ := a) using 0
  /-
    case a
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    a : R
    ⊢ Iff (Iff (Membership.mem { val := (Polynomial.cyclotomic n R).roots, nodup : …
  -/
  simp [cyclotomic_ne_zero n R]
  /-
    🎉 no goals
  -/


theorem cyclotomic.roots_eq_primitiveRoots_val [NeZero (n : R)] :
    (cyclotomic n R).roots = (primitiveRoots n R).val := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : IsDomain R
    inst✝ : NeZero ↑n
    ⊢ Eq (Polynomial.cyclotomic n R).roots (primitiveRoots n R).val
  -/
  rw [← cyclotomic.roots_to_finset_eq_primitiveRoots]
  /-
    🎉 no goals
  -/


/-- If `R` is of characteristic zero, then `ζ` is a root of `cyclotomic n R` if and only if it is a
primitive `n`-th root of unity. -/
theorem isRoot_cyclotomic_iff_charZero {n : ℕ} {R : Type*} [CommRing R] [IsDomain R] [CharZero R]
    {μ : R} (hn : 0 < n) : (Polynomial.cyclotomic n R).IsRoot μ ↔ IsPrimitiveRoot μ n :=
  letI := NeZero.of_gt hn
  isRoot_cyclotomic_iff


/-- Over a ring `R` of characteristic zero, `fun n => cyclotomic n R` is injective. -/
theorem cyclotomic_injective [CharZero R] : Function.Injective fun n => cyclotomic n R := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    ⊢ Function.Injective fun n => Polynomial.cyclotomic n R
  -/
  intro n m hnm
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n m : Nat
    hnm : Eq ((fun n => Polynomial.cyclotomic n R) n) ((fun n => Polynomial.cyclot …
    ⊢ Eq n m
  -/
  simp only at hnm
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n m : Nat
    hnm : Eq (Polynomial.cyclotomic n R) (Polynomial.cyclotomic m R)
    ⊢ Eq n m
  -/
  rcases eq_or_ne n 0 with (rfl | hzero)
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      m : Nat
      hnm : Eq (Polynomial.cyclotomic 0 R) (Polynomial.cyclotomic m R)
      ⊢ Eq 0 m
    -/
  · rw [cyclotomic_zero] at hnm
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      m : Nat
      hnm : Eq 1 (Polynomial.cyclotomic m R)
      ⊢ Eq 0 m
    -/
    replace hnm := congr_arg natDegree hnm
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      m : Nat
      hnm : Eq (Polynomial.natDegree 1) (Polynomial.cyclotomic m R).natDegree
      ⊢ Eq 0 m
    -/
    rwa [natDegree_one, natDegree_cyclotomic, eq_comm, Nat.totient_eq_zero, eq_comm] at hnm
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hnm : Eq (Polynomial.cyclotomic n R) (Polynomial.cyclotomic m R)
      hzero : Ne n 0
      ⊢ Eq n m
    -/
  · haveI := NeZero.mk hzero
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hnm : Eq (Polynomial.cyclotomic n R) (Polynomial.cyclotomic m R)
      hzero : Ne n 0
      this : NeZero n
      ⊢ Eq n m
    -/
    rw [← map_cyclotomic_int _ R, ← map_cyclotomic_int _ R] at hnm
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hnm : Eq (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)) (P …
      hzero : Ne n 0
      this : NeZero n
      ⊢ Eq n m
    -/
    replace hnm := map_injective (Int.castRingHom R) Int.cast_injective hnm
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Int) (Polynomial.cyclotomic m Int)
      ⊢ Eq n m
    -/
    replace hnm := congr_arg (map (Int.castRingHom ℂ)) hnm
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.map (Int.castRingHom Complex) (Polynomial.cyclotomic n In …
      ⊢ Eq n m
    -/
    rw [map_cyclotomic_int, map_cyclotomic_int] at hnm
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      ⊢ Eq n m
    -/
    have hprim := Complex.isPrimitiveRoot_exp _ hzero
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      ⊢ Eq n m
    -/
    have hroot := isRoot_cyclotomic_iff (R := ℂ).2 hprim
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hroot : (Polynomial.cyclotomic n Complex).IsRoot (Complex.exp (HDiv.hDiv (HMul …
      ⊢ Eq n m
    -/
    rw [hnm] at hroot
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hroot : (Polynomial.cyclotomic m Complex).IsRoot (Complex.exp (HDiv.hDiv (HMul …
      ⊢ Eq n m
    -/
    haveI hmzero : NeZero m := ⟨fun h => by simp [h] at hroot⟩
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hroot : (Polynomial.cyclotomic m Complex).IsRoot (Complex.exp (HDiv.hDiv (HMul …
      hmzero : NeZero m
      ⊢ Eq n m
    -/
    rw [isRoot_cyclotomic_iff (R := ℂ)] at hroot
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hroot : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hmzero : NeZero m
      ⊢ Eq n m
    -/
    replace hprim := hprim.eq_orderOf
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n m : Nat
      hzero : Ne n 0
      this : NeZero n
      hnm : Eq (Polynomial.cyclotomic n Complex) (Polynomial.cyclotomic m Complex)
      hroot : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      hmzero : NeZero m
      hprim : Eq n (orderOf (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi …
      ⊢ Eq n m
    -/
    rwa [← IsPrimitiveRoot.eq_orderOf hroot] at hprim
    /-
      🎉 no goals
    -/


/-- The minimal polynomial of a primitive `n`-th root of unity `μ` divides `cyclotomic n ℤ`. -/
theorem _root_.IsPrimitiveRoot.minpoly_dvd_cyclotomic {n : ℕ} {K : Type*} [Field K] {μ : K}
    (h : IsPrimitiveRoot μ n) (hpos : 0 < n) [CharZero K] : minpoly ℤ μ ∣ cyclotomic n ℤ := by
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    inst✝ : CharZero K
    ⊢ Dvd.dvd (minpoly Int μ) (Polynomial.cyclotomic n Int)
  -/
  apply minpoly.isIntegrallyClosed_dvd (h.isIntegral hpos)
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    inst✝ : CharZero K
    ⊢ Eq ((Polynomial.aeval μ) (Polynomial.cyclotomic n Int)) 0
  -/
  simpa [aeval_def, eval₂_eq_eval_map, IsRoot.def] using h.isRoot_cyclotomic hpos
  /-
    🎉 no goals
  -/


theorem _root_.IsPrimitiveRoot.minpoly_eq_cyclotomic_of_irreducible {K : Type*} [Field K]
    {R : Type*} [CommRing R] [IsDomain R] {μ : R} {n : ℕ} [Algebra K R] (hμ : IsPrimitiveRoot μ n)
    (h : Irreducible <| cyclotomic n K) [NeZero (n : K)] : cyclotomic n K = minpoly K μ := by
  /-
    K : Type u_2
    inst✝⁴ : Field K
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    μ : R
    n : Nat
    inst✝¹ : Algebra K R
    hμ : IsPrimitiveRoot μ n
    h : Irreducible (Polynomial.cyclotomic n K)
    inst✝ : NeZero ↑n
    ⊢ Eq (Polynomial.cyclotomic n K) (minpoly K μ)
  -/
  haveI := NeZero.of_noZeroSMulDivisors K R n
  /-
    K : Type u_2
    inst✝⁴ : Field K
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    μ : R
    n : Nat
    inst✝¹ : Algebra K R
    hμ : IsPrimitiveRoot μ n
    h : Irreducible (Polynomial.cyclotomic n K)
    inst✝ : NeZero ↑n
    this : NeZero ↑n
    ⊢ Eq (Polynomial.cyclotomic n K) (minpoly K μ)
  -/
  refine minpoly.eq_of_irreducible_of_monic h ?_ (cyclotomic.monic n K)
  /-
    K : Type u_2
    inst✝⁴ : Field K
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    μ : R
    n : Nat
    inst✝¹ : Algebra K R
    hμ : IsPrimitiveRoot μ n
    h : Irreducible (Polynomial.cyclotomic n K)
    inst✝ : NeZero ↑n
    this : NeZero ↑n
    ⊢ Eq ((Polynomial.aeval μ) (Polynomial.cyclotomic n K)) 0
  -/
  rwa [aeval_def, eval₂_eq_eval_map, map_cyclotomic, ← IsRoot.def, isRoot_cyclotomic_iff]
  /-
    🎉 no goals
  -/


/-- `cyclotomic n ℤ` is the minimal polynomial of a primitive `n`-th root of unity `μ`. -/
theorem cyclotomic_eq_minpoly {n : ℕ} {K : Type*} [Field K] {μ : K} (h : IsPrimitiveRoot μ n)
    (hpos : 0 < n) [CharZero K] : cyclotomic n ℤ = minpoly ℤ μ := by
  refine eq_of_monic_of_dvd_of_natDegree_le (minpoly.monic (IsPrimitiveRoot.isIntegral h hpos))
    (cyclotomic.monic n ℤ) (h.minpoly_dvd_cyclotomic hpos) ?_
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    inst✝ : CharZero K
    ⊢ LE.le (Polynomial.cyclotomic n Int).natDegree (minpoly Int μ).natDegree
  -/
  simpa [natDegree_cyclotomic n ℤ] using totient_le_degree_minpoly h
  /-
    🎉 no goals
  -/


/-- `cyclotomic n ℚ` is the minimal polynomial of a primitive `n`-th root of unity `μ`. -/
theorem cyclotomic_eq_minpoly_rat {n : ℕ} {K : Type*} [Field K] {μ : K} (h : IsPrimitiveRoot μ n)
    (hpos : 0 < n) [CharZero K] : cyclotomic n ℚ = minpoly ℚ μ := by
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    inst✝ : CharZero K
    ⊢ Eq (Polynomial.cyclotomic n Rat) (minpoly Rat μ)
  -/
  rw [← map_cyclotomic_int, cyclotomic_eq_minpoly h hpos]
  /-
    n : Nat
    K : Type u_2
    inst✝¹ : Field K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    inst✝ : CharZero K
    ⊢ Eq (Polynomial.map (Int.castRingHom Rat) (minpoly Int μ)) (minpoly Rat μ)
  -/
  exact (minpoly.isIntegrallyClosed_eq_field_fractions' _ (IsPrimitiveRoot.isIntegral h hpos)).symm
  /-
    🎉 no goals
  -/


/-- `cyclotomic n ℤ` is irreducible. -/
theorem cyclotomic.irreducible {n : ℕ} (hpos : 0 < n) : Irreducible (cyclotomic n ℤ) := by
  /-
    n : Nat
    hpos : LT.lt 0 n
    ⊢ Irreducible (Polynomial.cyclotomic n Int)
  -/
  rw [cyclotomic_eq_minpoly (isPrimitiveRoot_exp n hpos.ne') hpos]
  /-
    n : Nat
    hpos : LT.lt 0 n
    ⊢ Irreducible (minpoly Int (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Re …
  -/
  apply minpoly.irreducible
  /-
    case hx
    n : Nat
    hpos : LT.lt 0 n
    ⊢ IsIntegral Int (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Com …
  -/
  exact (isPrimitiveRoot_exp n hpos.ne').isIntegral hpos
  /-
    🎉 no goals
  -/


/-- `cyclotomic n ℚ` is irreducible. -/
theorem cyclotomic.irreducible_rat {n : ℕ} (hpos : 0 < n) : Irreducible (cyclotomic n ℚ) := by
  /-
    n : Nat
    hpos : LT.lt 0 n
    ⊢ Irreducible (Polynomial.cyclotomic n Rat)
  -/
  rw [← map_cyclotomic_int]
  exact (IsPrimitive.irreducible_iff_irreducible_map_fraction_map (cyclotomic.isPrimitive n ℤ)).1
    (cyclotomic.irreducible hpos)


/-- If `n ≠ m`, then `(cyclotomic n ℚ)` and `(cyclotomic m ℚ)` are coprime. -/
theorem cyclotomic.isCoprime_rat {n m : ℕ} (h : n ≠ m) :
    IsCoprime (cyclotomic n ℚ) (cyclotomic m ℚ) := by
  /-
    n m : Nat
    h : Ne n m
    ⊢ IsCoprime (Polynomial.cyclotomic n Rat) (Polynomial.cyclotomic m Rat)
  -/
  rcases n.eq_zero_or_pos with (rfl | hnzero)
    /-
      case inl
      m : Nat
      h : Ne 0 m
      ⊢ IsCoprime (Polynomial.cyclotomic 0 Rat) (Polynomial.cyclotomic m Rat)
    -/
  · exact isCoprime_one_left
    /-
      🎉 no goals
    -/
  /-
    case inr
    n m : Nat
    h : Ne n m
    hnzero : GT.gt n 0
    ⊢ IsCoprime (Polynomial.cyclotomic n Rat) (Polynomial.cyclotomic m Rat)
  -/
  rcases m.eq_zero_or_pos with (rfl | hmzero)
    /-
      case inr.inl
      n : Nat
      hnzero : GT.gt n 0
      h : Ne n 0
      ⊢ IsCoprime (Polynomial.cyclotomic n Rat) (Polynomial.cyclotomic 0 Rat)
    -/
  · exact isCoprime_one_right
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    n m : Nat
    h : Ne n m
    hnzero : GT.gt n 0
    hmzero : GT.gt m 0
    ⊢ IsCoprime (Polynomial.cyclotomic n Rat) (Polynomial.cyclotomic m Rat)
  -/
  rw [Irreducible.coprime_iff_not_dvd <| cyclotomic.irreducible_rat <| hnzero]
  exact fun hdiv => h <| cyclotomic_injective <|
    eq_of_monic_of_associated (cyclotomic.monic n ℚ) (cyclotomic.monic m ℚ) <|
      Irreducible.associated_of_dvd (cyclotomic.irreducible_rat hnzero)
        (cyclotomic.irreducible_rat hmzero) hdiv


