/-- The modified `n`-th cyclotomic polynomial with coefficients in `R`, it is the usual cyclotomic
polynomial if there is a primitive `n`-th root of unity in `R`. -/
def cyclotomic' (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] : R[X] :=
  ∏ μ ∈ primitiveRoots n R, (X - C μ)


/-- The zeroth modified cyclotomic polyomial is `1`. -/
@[simp]
theorem cyclotomic'_zero (R : Type*) [CommRing R] [IsDomain R] : cyclotomic' 0 R = 1 := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (Polynomial.cyclotomic' 0 R) 1
  -/
  simp only [cyclotomic', Finset.prod_empty, primitiveRoots_zero]
  /-
    🎉 no goals
  -/


/-- The first modified cyclotomic polyomial is `X - 1`. -/
@[simp]
theorem cyclotomic'_one (R : Type*) [CommRing R] [IsDomain R] : cyclotomic' 1 R = X - 1 := by
  simp only [cyclotomic', Finset.prod_singleton, RingHom.map_one,
    IsPrimitiveRoot.primitiveRoots_one]


/-- The second modified cyclotomic polyomial is `X + 1` if the characteristic of `R` is not `2`. -/
@[simp]
theorem cyclotomic'_two (R : Type*) [CommRing R] [IsDomain R] (p : ℕ) [CharP R p] (hp : p ≠ 2) :
    cyclotomic' 2 R = X + 1 := by
  /-
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Nat
    inst✝ : CharP R p
    hp : Ne p 2
    ⊢ Eq (Polynomial.cyclotomic' 2 R) (HAdd.hAdd Polynomial.X 1)
  -/
  rw [cyclotomic']
  have prim_root_two : primitiveRoots 2 R = {(-1 : R)} := by
    simp only [Finset.eq_singleton_iff_unique_mem, mem_primitiveRoots two_pos]
    exact ⟨IsPrimitiveRoot.neg_one p hp, fun x => IsPrimitiveRoot.eq_neg_one_of_two_right⟩
  /-
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Nat
    inst✝ : CharP R p
    hp : Ne p 2
    prim_root_two : Eq (primitiveRoots 2 R) (Singleton.singleton (-1))
    ⊢ Eq ((primitiveRoots 2 R).prod fun μ => HSub.hSub Polynomial.X (Polynomial.C  …
  -/
  simp only [prim_root_two, Finset.prod_singleton, RingHom.map_neg, RingHom.map_one, sub_neg_eq_add]
  /-
    🎉 no goals
  -/


/-- `cyclotomic' n R` is monic. -/
theorem cyclotomic'.monic (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] :
    (cyclotomic' n R).Monic :=
  monic_prod_of_monic _ _ fun _ _ => monic_X_sub_C _


/-- `cyclotomic' n R` is different from `0`. -/
theorem cyclotomic'_ne_zero (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] : cyclotomic' n R ≠ 0 :=
  (cyclotomic'.monic n R).ne_zero


/-- The natural degree of `cyclotomic' n R` is `totient n` if there is a primitive root of
unity in `R`. -/
theorem natDegree_cyclotomic' {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    (cyclotomic' n R).natDegree = Nat.totient n := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.cyclotomic' n R).natDegree n.totient
  -/
  rw [cyclotomic']
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Eq ((primitiveRoots n R).prod fun μ => HSub.hSub Polynomial.X (Polynomial.C  …
  -/
  rw [natDegree_prod (primitiveRoots n R) fun z : R => X - C z]
  · simp only [IsPrimitiveRoot.card_primitiveRoots h, mul_one, natDegree_X_sub_C, Nat.cast_id,
      Finset.sum_const, nsmul_eq_mul]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ ∀ (i : R), Membership.mem (primitiveRoots n R) i → Ne (HSub.hSub Polynomial. …
  -/
  intro z _
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    z : R
    a✝ : Membership.mem (primitiveRoots n R) z
    ⊢ Ne (HSub.hSub Polynomial.X (Polynomial.C z)) 0
  -/
  exact X_sub_C_ne_zero z
  /-
    🎉 no goals
  -/


/-- The degree of `cyclotomic' n R` is `totient n` if there is a primitive root of unity in `R`. -/
theorem degree_cyclotomic' {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    (cyclotomic' n R).degree = Nat.totient n := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.cyclotomic' n R).degree ↑n.totient
  -/
  simp only [degree_eq_natDegree (cyclotomic'_ne_zero n R), natDegree_cyclotomic' h]
  /-
    🎉 no goals
  -/


/-- The roots of `cyclotomic' n R` are the primitive `n`-th roots of unity. -/
theorem roots_of_cyclotomic (n : ℕ) (R : Type*) [CommRing R] [IsDomain R] :
    (cyclotomic' n R).roots = (primitiveRoots n R).val := by
  /-
    n : Nat
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (Polynomial.cyclotomic' n R).roots (primitiveRoots n R).val
  -/
  rw [cyclotomic']; exact roots_prod_X_sub_C (primitiveRoots n R)
                    /-
                      🎉 no goals
                    -/


/-- If there is a primitive `n`th root of unity in `K`, then `X ^ n - 1 = ∏ (X - μ)`, where `μ`
varies over the `n`-th roots of unity. -/
theorem X_pow_sub_one_eq_prod {ζ : R} {n : ℕ} (hpos : 0 < n) (h : IsPrimitiveRoot ζ n) :
    X ^ n - 1 = ∏ ζ ∈ nthRootsFinset n R, (X - C ζ) := by
  classical
  rw [nthRootsFinset, ← Multiset.toFinset_eq (IsPrimitiveRoot.nthRoots_one_nodup h)]
  simp only [Finset.prod_mk, RingHom.map_one]
  rw [nthRoots]
  have hmonic : (X ^ n - C (1 : R)).Monic := monic_X_pow_sub_C (1 : R) (ne_of_lt hpos).symm
  symm
  apply prod_multiset_X_sub_C_of_monic_of_roots_card_eq hmonic
  rw [@natDegree_X_pow_sub_C R _ _ n 1, ← nthRoots]
  exact IsPrimitiveRoot.card_nthRoots_one h


/-- `cyclotomic' n K` splits. -/
theorem cyclotomic'_splits (n : ℕ) : Splits (RingHom.id K) (cyclotomic' n K) := by
  /-
    K : Type u_1
    inst✝ : Field K
    n : Nat
    ⊢ Polynomial.Splits (RingHom.id K) (Polynomial.cyclotomic' n K)
  -/
  apply splits_prod (RingHom.id K)
  /-
    K : Type u_1
    inst✝ : Field K
    n : Nat
    ⊢ ∀ (j : K), Membership.mem (primitiveRoots n K) j → Polynomial.Splits (RingHo …
  -/
  intro z _
  /-
    K : Type u_1
    inst✝ : Field K
    n : Nat
    z : K
    a✝ : Membership.mem (primitiveRoots n K) z
    ⊢ Polynomial.Splits (RingHom.id K) (HSub.hSub Polynomial.X (Polynomial.C z))
  -/
  simp only [splits_X_sub_C (RingHom.id K)]
  /-
    🎉 no goals
  -/


/-- If there is a primitive `n`-th root of unity in `K`, then `X ^ n - 1` splits. -/
theorem X_pow_sub_one_splits {ζ : K} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    Splits (RingHom.id K) (X ^ n - C (1 : K)) := by
  /-
    K : Type u_1
    inst✝ : Field K
    ζ : K
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Polynomial.Splits (RingHom.id K) (HSub.hSub (HPow.hPow Polynomial.X n) (Poly …
  -/
  rw [splits_iff_card_roots, ← nthRoots, IsPrimitiveRoot.card_nthRoots_one h, natDegree_X_pow_sub_C]
  /-
    🎉 no goals
  -/


/-- If there is a primitive `n`-th root of unity in `K`, then
`∏ i ∈ Nat.divisors n, cyclotomic' i K = X ^ n - 1`. -/
theorem prod_cyclotomic'_eq_X_pow_sub_one {K : Type*} [CommRing K] [IsDomain K] {ζ : K} {n : ℕ}
    (hpos : 0 < n) (h : IsPrimitiveRoot ζ n) :
    ∏ i ∈ Nat.divisors n, cyclotomic' i K = X ^ n - 1 := by
  classical
  have hd : (n.divisors : Set ℕ).PairwiseDisjoint fun k => primitiveRoots k K :=
    fun x _ y _ hne => IsPrimitiveRoot.disjoint hne
  simp only [X_pow_sub_one_eq_prod hpos h, cyclotomic', ← Finset.prod_biUnion hd,
    h.nthRoots_one_eq_biUnion_primitiveRoots]


/-- If there is a primitive `n`-th root of unity in `K`, then
`cyclotomic' n K = (X ^ k - 1) /ₘ (∏ i ∈ Nat.properDivisors k, cyclotomic' i K)`. -/
theorem cyclotomic'_eq_X_pow_sub_one_div {K : Type*} [CommRing K] [IsDomain K] {ζ : K} {n : ℕ}
    (hpos : 0 < n) (h : IsPrimitiveRoot ζ n) :
    cyclotomic' n K = (X ^ n - 1) /ₘ ∏ i ∈ Nat.properDivisors n, cyclotomic' i K := by
  rw [← prod_cyclotomic'_eq_X_pow_sub_one hpos h, ← Nat.cons_self_properDivisors hpos.ne',
    Finset.prod_cons]
  have prod_monic : (∏ i ∈ Nat.properDivisors n, cyclotomic' i K).Monic := by
    apply monic_prod_of_monic
    intro i _
    exact cyclotomic'.monic i K
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    ⊢ Eq (Polynomial.cyclotomic' n K) ((HMul.hMul (Polynomial.cyclotomic' n K) (n. …
  -/
  rw [(div_modByMonic_unique (cyclotomic' n K) 0 prod_monic _).1]
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    ⊢ And (Eq (HAdd.hAdd 0 (HMul.hMul (n.properDivisors.prod fun i => Polynomial.c …
  -/
  simp only [degree_zero, zero_add]
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    ⊢ And (Eq (HMul.hMul (n.properDivisors.prod fun i => Polynomial.cyclotomic' i  …
  -/
  refine ⟨by rw [mul_comm], ?_⟩
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    ⊢ LT.lt Bot.bot (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).de …
  -/
  rw [bot_lt_iff_ne_bot]
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    ⊢ Ne (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).degree Bot.bot
  -/
  intro h
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hpos : LT.lt 0 n
    h✝ : IsPrimitiveRoot ζ n
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).Monic
    h : Eq (n.properDivisors.prod fun i => Polynomial.cyclotomic' i K).degree Bot. …
    ⊢ False
  -/
  exact Monic.ne_zero prod_monic (degree_eq_bot.1 h)
  /-
    🎉 no goals
  -/


/-- If there is a primitive `n`-th root of unity in `K`, then `cyclotomic' n K` comes from a
monic polynomial with integer coefficients. -/
theorem int_coeff_of_cyclotomic' {K : Type*} [CommRing K] [IsDomain K] {ζ : K} {n : ℕ}
    (h : IsPrimitiveRoot ζ n) : ∃ P : ℤ[X], map (Int.castRingHom K) P =
      cyclotomic' n K ∧ P.degree = (cyclotomic' n K).degree ∧ P.Monic := by
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Exists fun P => And (Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.c …
  -/
  refine lifts_and_degree_eq_and_monic ?_ (cyclotomic'.monic n K)
  /-
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  induction' n using Nat.strong_induction_on with k ihk generalizing ζ
  /-
    case h
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  rcases k.eq_zero_or_pos with (rfl | hpos)
    /-
      case h.inl
      K : Type u_2
      inst✝¹ : CommRing K
      inst✝ : IsDomain K
      ζ : K
      ihk : ∀ (m : Nat), LT.lt m 0 → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
      h : IsPrimitiveRoot ζ 0
      ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
    -/
  · use 1
    /-
      case h
      K : Type u_2
      inst✝¹ : CommRing K
      inst✝ : IsDomain K
      ζ : K
      ihk : ∀ (m : Nat), LT.lt m 0 → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
      h : IsPrimitiveRoot ζ 0
      ⊢ Eq ((Polynomial.mapRingHom (Int.castRingHom K)) 1) (Polynomial.cyclotomic' 0 …
    -/
    simp only [cyclotomic'_zero, coe_mapRingHom, Polynomial.map_one]
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  let B : K[X] := ∏ i ∈ Nat.properDivisors k, cyclotomic' i K
  have Bmo : B.Monic := by
    apply monic_prod_of_monic
    intro i _
    exact cyclotomic'.monic i K
  have Bint : B ∈ lifts (Int.castRingHom K) := by
    refine Subsemiring.prod_mem (lifts (Int.castRingHom K)) ?_
    intro x hx
    have xsmall := (Nat.mem_properDivisors.1 hx).2
    obtain ⟨d, hd⟩ := (Nat.mem_properDivisors.1 hx).1
    rw [mul_comm] at hd
    exact ihk x xsmall (h.pow hpos hd)
  /-
    case h.inr
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    Bint : Membership.mem (Polynomial.lifts (Int.castRingHom K)) B
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  replace Bint := lifts_and_degree_eq_and_monic Bint Bmo
  /-
    case h.inr
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    Bint : Exists fun q => And (Eq (Polynomial.map (Int.castRingHom K) q) B) (And  …
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  obtain ⟨B₁, hB₁, _, hB₁mo⟩ := Bint
  /-
    case h.inr.intro.intro.intro
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  let Q₁ : ℤ[X] := (X ^ k - 1) /ₘ B₁
  have huniq : 0 + B * cyclotomic' k K = X ^ k - 1 ∧ (0 : K[X]).degree < B.degree := by
    constructor
    · rw [zero_add, mul_comm, ← prod_cyclotomic'_eq_X_pow_sub_one hpos h, ←
        Nat.cons_self_properDivisors hpos.ne', Finset.prod_cons]
    · simpa only [degree_zero, bot_lt_iff_ne_bot, Ne, degree_eq_bot] using Bmo.ne_zero
  /-
    case h.inr.intro.intro.intro
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    Q₁ : Polynomial Int := (HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B₁
    huniq : And (Eq (HAdd.hAdd 0 (HMul.hMul B (Polynomial.cyclotomic' k K))) (HSub …
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  replace huniq := div_modByMonic_unique (cyclotomic' k K) (0 : K[X]) Bmo huniq
  /-
    case h.inr.intro.intro.intro
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    Q₁ : Polynomial Int := (HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B₁
    huniq : And (Eq ((HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B) (Polyn …
    ⊢ Membership.mem (Polynomial.lifts (Int.castRingHom K)) (Polynomial.cyclotomic …
  -/
  simp only [lifts, RingHom.mem_rangeS]
  /-
    case h.inr.intro.intro.intro
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    Q₁ : Polynomial Int := (HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B₁
    huniq : And (Eq ((HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B) (Polyn …
    ⊢ Exists fun x => Eq ((Polynomial.mapRingHom (Int.castRingHom K)) x) (Polynomi …
  -/
  use Q₁
  /-
    case h
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    Q₁ : Polynomial Int := (HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B₁
    huniq : And (Eq ((HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B) (Polyn …
    ⊢ Eq ((Polynomial.mapRingHom (Int.castRingHom K)) Q₁) (Polynomial.cyclotomic'  …
  -/
  rw [coe_mapRingHom, map_divByMonic (Int.castRingHom K) hB₁mo, hB₁, ← huniq.1]
  /-
    case h
    K : Type u_2
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    ihk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Membership.mem …
    ζ : K
    h : IsPrimitiveRoot ζ k
    hpos : GT.gt k 0
    B : Polynomial K := k.properDivisors.prod fun i => Polynomial.cyclotomic' i K
    Bmo : B.Monic
    B₁ : Polynomial Int
    hB₁ : Eq (Polynomial.map (Int.castRingHom K) B₁) B
    left✝ : Eq B₁.degree B.degree
    hB₁mo : B₁.Monic
    Q₁ : Polynomial Int := (HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B₁
    huniq : And (Eq ((HSub.hSub (HPow.hPow Polynomial.X k) 1).divByMonic B) (Polyn …
    ⊢ Eq ((Polynomial.map (Int.castRingHom K) (HSub.hSub (HPow.hPow Polynomial.X k …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `K` is of characteristic `0` and there is a primitive `n`-th root of unity in `K`,
then `cyclotomic n K` comes from a unique polynomial with integer coefficients. -/
theorem unique_int_coeff_of_cycl {K : Type*} [CommRing K] [IsDomain K] [CharZero K] {ζ : K}
    {n : ℕ+} (h : IsPrimitiveRoot ζ n) :
    ∃! P : ℤ[X], map (Int.castRingHom K) P = cyclotomic' n K := by
  /-
    K : Type u_2
    inst✝² : CommRing K
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    ζ : K
    n : PNat
    h : IsPrimitiveRoot ζ ↑n
    ⊢ ExistsUnique fun P => Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial. …
  -/
  obtain ⟨P, hP⟩ := int_coeff_of_cyclotomic' h
  /-
    case intro
    K : Type u_2
    inst✝² : CommRing K
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    ζ : K
    n : PNat
    h : IsPrimitiveRoot ζ ↑n
    P : Polynomial Int
    hP : And (Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.cyclotomic' (↑ …
    ⊢ ExistsUnique fun P => Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial. …
  -/
  refine ⟨P, hP.1, fun Q hQ => ?_⟩
  /-
    case intro
    K : Type u_2
    inst✝² : CommRing K
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    ζ : K
    n : PNat
    h : IsPrimitiveRoot ζ ↑n
    P : Polynomial Int
    hP : And (Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.cyclotomic' (↑ …
    Q : Polynomial Int
    hQ : (fun P => Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.cyclotomi …
    ⊢ Eq Q P
  -/
  apply map_injective (Int.castRingHom K) Int.cast_injective
  /-
    case intro.a
    K : Type u_2
    inst✝² : CommRing K
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    ζ : K
    n : PNat
    h : IsPrimitiveRoot ζ ↑n
    P : Polynomial Int
    hP : And (Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.cyclotomic' (↑ …
    Q : Polynomial Int
    hQ : (fun P => Eq (Polynomial.map (Int.castRingHom K) P) (Polynomial.cyclotomi …
    ⊢ Eq (Polynomial.map (Int.castRingHom K) Q) (Polynomial.map (Int.castRingHom K …
  -/
  rw [hP.1, hQ]
  /-
    🎉 no goals
  -/


/-- The `n`-th cyclotomic polynomial with coefficients in `R`. -/
def cyclotomic (n : ℕ) (R : Type*) [Ring R] : R[X] :=
  if h : n = 0 then 1
  else map (Int.castRingHom R) (int_coeff_of_cyclotomic' (Complex.isPrimitiveRoot_exp n h)).choose


theorem int_cyclotomic_rw {n : ℕ} (h : n ≠ 0) :
    cyclotomic n ℤ = (int_coeff_of_cyclotomic' (Complex.isPrimitiveRoot_exp n h)).choose := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ Eq (Polynomial.cyclotomic n Int) ⋯.choose
  -/
  simp only [cyclotomic, h, dif_neg, not_false_iff]
  /-
    n : Nat
    h : Ne n 0
    ⊢ Eq (Polynomial.map (Int.castRingHom Int) ⋯.choose) ⋯.choose
  -/
  ext i
  /-
    case a
    n : Nat
    h : Ne n 0
    i : Nat
    ⊢ Eq ((Polynomial.map (Int.castRingHom Int) ⋯.choose).coeff i) (⋯.choose.coeff …
  -/
  simp only [coeff_map, Int.cast_id, eq_intCast]
  /-
    🎉 no goals
  -/


/-- `cyclotomic n R` comes from `cyclotomic n ℤ`. -/
theorem map_cyclotomic_int (n : ℕ) (R : Type*) [Ring R] :
    map (Int.castRingHom R) (cyclotomic n ℤ) = cyclotomic n R := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)) (Polyn …
  -/
  by_cases hzero : n = 0
    /-
      case pos
      n : Nat
      R : Type u_1
      inst✝ : Ring R
      hzero : Eq n 0
      ⊢ Eq (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)) (Polyn …
    -/
  · simp only [hzero, cyclotomic, dif_pos, Polynomial.map_one]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    hzero : Not (Eq n 0)
    ⊢ Eq (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)) (Polyn …
  -/
  simp [cyclotomic, hzero]
  /-
    🎉 no goals
  -/


theorem int_cyclotomic_spec (n : ℕ) :
    map (Int.castRingHom ℂ) (cyclotomic n ℤ) = cyclotomic' n ℂ ∧
      (cyclotomic n ℤ).degree = (cyclotomic' n ℂ).degree ∧ (cyclotomic n ℤ).Monic := by
  /-
    n : Nat
    ⊢ And (Eq (Polynomial.map (Int.castRingHom Complex) (Polynomial.cyclotomic n I …
  -/
  by_cases hzero : n = 0
  · simp only [hzero, cyclotomic, degree_one, monic_one, cyclotomic'_zero, dif_pos,
      eq_self_iff_true, Polynomial.map_one, and_self_iff]
  /-
    case neg
    n : Nat
    hzero : Not (Eq n 0)
    ⊢ And (Eq (Polynomial.map (Int.castRingHom Complex) (Polynomial.cyclotomic n I …
  -/
  rw [int_cyclotomic_rw hzero]
  /-
    case neg
    n : Nat
    hzero : Not (Eq n 0)
    ⊢ And (Eq (Polynomial.map (Int.castRingHom Complex) ⋯.choose) (Polynomial.cycl …
  -/
  exact (int_coeff_of_cyclotomic' (Complex.isPrimitiveRoot_exp n hzero)).choose_spec
  /-
    🎉 no goals
  -/


theorem int_cyclotomic_unique {n : ℕ} {P : ℤ[X]} (h : map (Int.castRingHom ℂ) P = cyclotomic' n ℂ) :
    P = cyclotomic n ℤ := by
  /-
    n : Nat
    P : Polynomial Int
    h : Eq (Polynomial.map (Int.castRingHom Complex) P) (Polynomial.cyclotomic' n  …
    ⊢ Eq P (Polynomial.cyclotomic n Int)
  -/
  apply map_injective (Int.castRingHom ℂ) Int.cast_injective
  /-
    case a
    n : Nat
    P : Polynomial Int
    h : Eq (Polynomial.map (Int.castRingHom Complex) P) (Polynomial.cyclotomic' n  …
    ⊢ Eq (Polynomial.map (Int.castRingHom Complex) P) (Polynomial.map (Int.castRin …
  -/
  rw [h, (int_cyclotomic_spec n).1]
  /-
    🎉 no goals
  -/


/-- The definition of `cyclotomic n R` commutes with any ring homomorphism. -/
@[simp]
theorem map_cyclotomic (n : ℕ) {R S : Type*} [Ring R] [Ring S] (f : R →+* S) :
    map f (cyclotomic n R) = cyclotomic n S := by
  /-
    n : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    ⊢ Eq (Polynomial.map f (Polynomial.cyclotomic n R)) (Polynomial.cyclotomic n S)
  -/
  rw [← map_cyclotomic_int n R, ← map_cyclotomic_int n S, map_map]
  /-
    n : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    ⊢ Eq (Polynomial.map (f.comp (Int.castRingHom R)) (Polynomial.cyclotomic n Int …
  -/
  have : Subsingleton (ℤ →+* S) := inferInstance
  /-
    n : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    this : Subsingleton (RingHom Int S)
    ⊢ Eq (Polynomial.map (f.comp (Int.castRingHom R)) (Polynomial.cyclotomic n Int …
  -/
  congr!
  /-
    🎉 no goals
  -/


theorem cyclotomic.eval_apply {R S : Type*} (q : R) (n : ℕ) [Ring R] [Ring S] (f : R →+* S) :
    eval (f q) (cyclotomic n S) = f (eval q (cyclotomic n R)) := by
  /-
    R : Type u_1
    S : Type u_2
    q : R
    n : Nat
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    ⊢ Eq (Polynomial.eval (f q) (Polynomial.cyclotomic n S)) (f (Polynomial.eval q …
  -/
  rw [← map_cyclotomic n f, eval_map, eval₂_at_apply]
  /-
    🎉 no goals
  -/


/-- The zeroth cyclotomic polyomial is `1`. -/
@[simp]
theorem cyclotomic_zero (R : Type*) [Ring R] : cyclotomic 0 R = 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (Polynomial.cyclotomic 0 R) 1
  -/
  simp only [cyclotomic, dif_pos]
  /-
    🎉 no goals
  -/


/-- The first cyclotomic polyomial is `X - 1`. -/
@[simp]
theorem cyclotomic_one (R : Type*) [Ring R] : cyclotomic 1 R = X - 1 := by
  have hspec : map (Int.castRingHom ℂ) (X - 1) = cyclotomic' 1 ℂ := by
    simp only [cyclotomic'_one, PNat.one_coe, map_X, Polynomial.map_one, Polynomial.map_sub]
  /-
    R : Type u_1
    inst✝ : Ring R
    hspec : Eq (Polynomial.map (Int.castRingHom Complex) (HSub.hSub Polynomial.X 1 …
    ⊢ Eq (Polynomial.cyclotomic 1 R) (HSub.hSub Polynomial.X 1)
  -/
  symm
  /-
    R : Type u_1
    inst✝ : Ring R
    hspec : Eq (Polynomial.map (Int.castRingHom Complex) (HSub.hSub Polynomial.X 1 …
    ⊢ Eq (HSub.hSub Polynomial.X 1) (Polynomial.cyclotomic 1 R)
  -/
  rw [← map_cyclotomic_int, ← int_cyclotomic_unique hspec]
  /-
    R : Type u_1
    inst✝ : Ring R
    hspec : Eq (Polynomial.map (Int.castRingHom Complex) (HSub.hSub Polynomial.X 1 …
    ⊢ Eq (HSub.hSub Polynomial.X 1) (Polynomial.map (Int.castRingHom R) (HSub.hSub …
  -/
  simp only [map_X, Polynomial.map_one, Polynomial.map_sub]
  /-
    🎉 no goals
  -/


/-- `cyclotomic n` is monic. -/
theorem cyclotomic.monic (n : ℕ) (R : Type*) [Ring R] : (cyclotomic n R).Monic := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    ⊢ (Polynomial.cyclotomic n R).Monic
  -/
  rw [← map_cyclotomic_int]
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    ⊢ (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)).Monic
  -/
  exact (int_cyclotomic_spec n).2.2.map _
  /-
    🎉 no goals
  -/


/-- `cyclotomic n` is primitive. -/
theorem cyclotomic.isPrimitive (n : ℕ) (R : Type*) [CommRing R] : (cyclotomic n R).IsPrimitive :=
  (cyclotomic.monic n R).isPrimitive


/-- `cyclotomic n R` is different from `0`. -/
theorem cyclotomic_ne_zero (n : ℕ) (R : Type*) [Ring R] [Nontrivial R] : cyclotomic n R ≠ 0 :=
  (cyclotomic.monic n R).ne_zero


/-- The degree of `cyclotomic n` is `totient n`. -/
theorem degree_cyclotomic (n : ℕ) (R : Type*) [Ring R] [Nontrivial R] :
    (cyclotomic n R).degree = Nat.totient n := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ Eq (Polynomial.cyclotomic n R).degree ↑n.totient
  -/
  rw [← map_cyclotomic_int]
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ Eq (Polynomial.map (Int.castRingHom R) (Polynomial.cyclotomic n Int)).degree …
  -/
  rw [degree_map_eq_of_leadingCoeff_ne_zero (Int.castRingHom R) _]
    /-
      n : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : Nontrivial R
      ⊢ Eq (Polynomial.cyclotomic n Int).degree ↑n.totient
    -/
  · cases' n with k
      /-
        case zero
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        ⊢ Eq (Polynomial.cyclotomic 0 Int).degree ↑(Nat.totient 0)
      -/
    · simp only [cyclotomic, degree_one, dif_pos, Nat.totient_zero, CharP.cast_eq_zero]
      /-
        🎉 no goals
      -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : Nontrivial R
      k : Nat
      ⊢ Eq (Polynomial.cyclotomic (HAdd.hAdd k 1) Int).degree ↑(HAdd.hAdd k 1).totient
    -/
    rw [← degree_cyclotomic' (Complex.isPrimitiveRoot_exp k.succ (Nat.succ_ne_zero k))]
    /-
      case succ
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : Nontrivial R
      k : Nat
      ⊢ Eq (Polynomial.cyclotomic (HAdd.hAdd k 1) Int).degree (Polynomial.cyclotomic …
    -/
    exact (int_cyclotomic_spec k.succ).2.1
    /-
      🎉 no goals
    -/
  simp only [(int_cyclotomic_spec n).right.right, eq_intCast, Monic.leadingCoeff, Int.cast_one,
    Ne, not_false_iff, one_ne_zero]


/-- The natural degree of `cyclotomic n` is `totient n`. -/
theorem natDegree_cyclotomic (n : ℕ) (R : Type*) [Ring R] [Nontrivial R] :
    (cyclotomic n R).natDegree = Nat.totient n := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ Eq (Polynomial.cyclotomic n R).natDegree n.totient
  -/
  rw [natDegree, degree_cyclotomic]; norm_cast
                                     /-
                                       🎉 no goals
                                     -/


/-- The degree of `cyclotomic n R` is positive. -/
theorem degree_cyclotomic_pos (n : ℕ) (R : Type*) (hpos : 0 < n) [Ring R] [Nontrivial R] :
    0 < (cyclotomic n R).degree := by
  /-
    n : Nat
    R : Type u_1
    hpos : LT.lt 0 n
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ LT.lt 0 (Polynomial.cyclotomic n R).degree
  -/
  rwa [degree_cyclotomic n R, Nat.cast_pos, Nat.totient_pos]
  /-
    🎉 no goals
  -/


/-- `∏ i ∈ Nat.divisors n, cyclotomic i R = X ^ n - 1`. -/
theorem prod_cyclotomic_eq_X_pow_sub_one {n : ℕ} (hpos : 0 < n) (R : Type*) [CommRing R] :
    ∏ i ∈ Nat.divisors n, cyclotomic i R = X ^ n - 1 := by
  have integer : ∏ i ∈ Nat.divisors n, cyclotomic i ℤ = X ^ n - 1 := by
    apply map_injective (Int.castRingHom ℂ) Int.cast_injective
    simp only [Polynomial.map_prod, int_cyclotomic_spec, Polynomial.map_pow, map_X,
      Polynomial.map_one, Polynomial.map_sub]
    exact prod_cyclotomic'_eq_X_pow_sub_one hpos (Complex.isPrimitiveRoot_exp n hpos.ne')
  simpa only [Polynomial.map_prod, map_cyclotomic_int, Polynomial.map_sub, Polynomial.map_one,
    Polynomial.map_pow, Polynomial.map_X] using congr_arg (map (Int.castRingHom R)) integer


theorem cyclotomic.dvd_X_pow_sub_one (n : ℕ) (R : Type*) [Ring R] :
    cyclotomic n R ∣ X ^ n - 1 := by
  suffices cyclotomic n ℤ ∣ X ^ n - 1 by
    simpa only [map_cyclotomic_int, Polynomial.map_sub, Polynomial.map_one, Polynomial.map_pow,
      Polynomial.map_X] using map_dvd (Int.castRingHom R) this
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    ⊢ Dvd.dvd (Polynomial.cyclotomic n Int) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      ⊢ Dvd.dvd (Polynomial.cyclotomic 0 Int) (HSub.hSub (HPow.hPow Polynomial.X 0) 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    hn : GT.gt n 0
    ⊢ Dvd.dvd (Polynomial.cyclotomic n Int) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  rw [← prod_cyclotomic_eq_X_pow_sub_one hn]
  /-
    case inr
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    hn : GT.gt n 0
    ⊢ Dvd.dvd (Polynomial.cyclotomic n Int) (n.divisors.prod fun i => Polynomial.c …
  -/
  exact Finset.dvd_prod_of_mem _ (n.mem_divisors_self hn.ne')
  /-
    🎉 no goals
  -/


theorem prod_cyclotomic_eq_geom_sum {n : ℕ} (h : 0 < n) (R) [CommRing R] :
    ∏ i ∈ n.divisors.erase 1, cyclotomic i R = ∑ i ∈ Finset.range n, X ^ i := by
  suffices (∏ i ∈ n.divisors.erase 1, cyclotomic i ℤ) = ∑ i ∈ Finset.range n, X ^ i by
    simpa only [Polynomial.map_prod, map_cyclotomic_int, Polynomial.map_sum, Polynomial.map_pow,
      Polynomial.map_X] using congr_arg (map (Int.castRingHom R)) this
  rw [← mul_left_inj' (cyclotomic_ne_zero 1 ℤ), prod_erase_mul _ _ (Nat.one_mem_divisors.2 h.ne'),
    cyclotomic_one, geom_sum_mul, prod_cyclotomic_eq_X_pow_sub_one h]


/-- If `p` is prime, then `cyclotomic p R = ∑ i ∈ range p, X ^ i`. -/
theorem cyclotomic_prime (R : Type*) [Ring R] (p : ℕ) [hp : Fact p.Prime] :
    cyclotomic p R = ∑ i ∈ Finset.range p, X ^ i := by
  suffices cyclotomic p ℤ = ∑ i ∈ range p, X ^ i by
    simpa only [map_cyclotomic_int, Polynomial.map_sum, Polynomial.map_pow, Polynomial.map_X] using
      congr_arg (map (Int.castRingHom R)) this
  rw [← prod_cyclotomic_eq_geom_sum hp.out.pos, hp.out.divisors,
    erase_insert (mem_singleton.not.2 hp.out.ne_one.symm), prod_singleton]


theorem cyclotomic_prime_mul_X_sub_one (R : Type*) [Ring R] (p : ℕ) [hn : Fact (Nat.Prime p)] :
                                               /-
                                                 R : Type u_1
                                                 inst✝ : Ring R
                                                 p : Nat
                                                 hn : Fact (Nat.Prime p)
                                                 ⊢ Eq (HMul.hMul (Polynomial.cyclotomic p R) (HSub.hSub Polynomial.X 1)) (HSub. …
                                               -/
    cyclotomic p R * (X - 1) = X ^ p - 1 := by rw [cyclotomic_prime, geom_sum_mul]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                                           /-
                                                                             R : Type u_1
                                                                             inst✝ : Ring R
                                                                             ⊢ Eq (Polynomial.cyclotomic 2 R) (HAdd.hAdd Polynomial.X 1)
                                                                           -/
theorem cyclotomic_two (R : Type*) [Ring R] : cyclotomic 2 R = X + 1 := by simp [cyclotomic_prime]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem cyclotomic_three (R : Type*) [Ring R] : cyclotomic 3 R = X ^ 2 + X + 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (Polynomial.cyclotomic 3 R) (HAdd.hAdd (HAdd.hAdd (HPow.hPow Polynomial.X …
  -/
  simp [cyclotomic_prime, sum_range_succ']
  /-
    🎉 no goals
  -/


theorem cyclotomic_dvd_geom_sum_of_dvd (R) [Ring R] {d n : ℕ} (hdn : d ∣ n) (hd : d ≠ 1) :
    cyclotomic d R ∣ ∑ i ∈ Finset.range n, X ^ i := by
  suffices cyclotomic d ℤ ∣ ∑ i ∈ Finset.range n, X ^ i by
    simpa only [map_cyclotomic_int, Polynomial.map_sum, Polynomial.map_pow, Polynomial.map_X] using
      map_dvd (Int.castRingHom R) this
  /-
    R : Type u_1
    inst✝ : Ring R
    d n : Nat
    hdn : Dvd.dvd d n
    hd : Ne d 1
    ⊢ Dvd.dvd (Polynomial.cyclotomic d Int) ((Finset.range n).sum fun i => HPow.hP …
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      d : Nat
      hd : Ne d 1
      hdn : Dvd.dvd d 0
      ⊢ Dvd.dvd (Polynomial.cyclotomic d Int) ((Finset.range 0).sum fun i => HPow.hP …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝ : Ring R
    d n : Nat
    hdn : Dvd.dvd d n
    hd : Ne d 1
    hn : GT.gt n 0
    ⊢ Dvd.dvd (Polynomial.cyclotomic d Int) ((Finset.range n).sum fun i => HPow.hP …
  -/
  rw [← prod_cyclotomic_eq_geom_sum hn]
  /-
    case inr
    R : Type u_1
    inst✝ : Ring R
    d n : Nat
    hdn : Dvd.dvd d n
    hd : Ne d 1
    hn : GT.gt n 0
    ⊢ Dvd.dvd (Polynomial.cyclotomic d Int) ((n.divisors.erase 1).prod fun i => Po …
  -/
  apply Finset.dvd_prod_of_mem
  /-
    case inr.ha
    R : Type u_1
    inst✝ : Ring R
    d n : Nat
    hdn : Dvd.dvd d n
    hd : Ne d 1
    hn : GT.gt n 0
    ⊢ Membership.mem (n.divisors.erase 1) d
  -/
  simp [hd, hdn, hn.ne']
  /-
    🎉 no goals
  -/


theorem X_pow_sub_one_mul_prod_cyclotomic_eq_X_pow_sub_one_of_dvd (R) [CommRing R] {d n : ℕ}
    (h : d ∈ n.properDivisors) :
    ((X ^ d - 1) * ∏ x ∈ n.divisors \ d.divisors, cyclotomic x R) = X ^ n - 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) ((SDiff.sdiff n.divis …
  -/
  obtain ⟨hd, hdn⟩ := Nat.mem_properDivisors.mp h
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hd : Dvd.dvd d n
    hdn : LT.lt d n
    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) ((SDiff.sdiff n.divis …
  -/
  have h0n : 0 < n := pos_of_gt hdn
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hd : Dvd.dvd d n
    hdn : LT.lt d n
    h0n : LT.lt 0 n
    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) ((SDiff.sdiff n.divis …
  -/
  have h0d : 0 < d := Nat.pos_of_dvd_of_pos hd h0n
  rw [← prod_cyclotomic_eq_X_pow_sub_one h0d, ← prod_cyclotomic_eq_X_pow_sub_one h0n, mul_comm,
    Finset.prod_sdiff (Nat.divisors_subset_of_dvd h0n.ne' hd)]


theorem X_pow_sub_one_mul_cyclotomic_dvd_X_pow_sub_one_of_dvd (R) [CommRing R] {d n : ℕ}
    (h : d ∈ n.properDivisors) : (X ^ d - 1) * cyclotomic n R ∣ X ^ n - 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    ⊢ Dvd.dvd (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) (Polynomial.cycl …
  -/
  have hdn := (Nat.mem_properDivisors.mp h).2
  /-
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Dvd.dvd (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) (Polynomial.cycl …
  -/
  use ∏ x ∈ n.properDivisors \ d.divisors, cyclotomic x R
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) 1) (HMul.hMul (HMul.hMul (HSub.hSub …
  -/
  symm
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) (Polynomia …
  -/
  convert X_pow_sub_one_mul_prod_cyclotomic_eq_X_pow_sub_one_of_dvd R h using 1
  /-
    case h.e'_2
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) (Polynomia …
  -/
  rw [mul_assoc]
  /-
    case h.e'_2
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow Polynomial.X d) 1) (HMul.hMul (Polynomia …
  -/
  congr 1
  /-
    case h.e'_2.e_a
    R : Type u_1
    inst✝ : CommRing R
    d n : Nat
    h : Membership.mem n.properDivisors d
    hdn : LT.lt d n
    ⊢ Eq (HMul.hMul (Polynomial.cyclotomic n R) ((SDiff.sdiff n.properDivisors d.d …
  -/
  rw [← Nat.insert_self_properDivisors hdn.ne_bot, insert_sdiff_of_not_mem, prod_insert]
    /-
      case h.e'_2.e_a
      R : Type u_1
      inst✝ : CommRing R
      d n : Nat
      h : Membership.mem n.properDivisors d
      hdn : LT.lt d n
      ⊢ Not (Membership.mem (SDiff.sdiff n.properDivisors d.divisors) n)
    -/
  · exact Finset.not_mem_sdiff_of_not_mem_left Nat.properDivisors.not_self_mem
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.e_a.h
      R : Type u_1
      inst✝ : CommRing R
      d n : Nat
      h : Membership.mem n.properDivisors d
      hdn : LT.lt d n
      ⊢ Not (Membership.mem d.divisors n)
    -/
  · exact fun hk => hdn.not_le <| Nat.divisor_le hk
    /-
      🎉 no goals
    -/


/-- `cyclotomic n R` can be expressed as a product in a fraction field of `R[X]`
  using Möbius inversion. -/
theorem cyclotomic_eq_prod_X_pow_sub_one_pow_moebius {n : ℕ} (R : Type*) [CommRing R]
    [IsDomain R] : algebraMap _ (RatFunc R) (cyclotomic n R) =
      ∏ i ∈ n.divisorsAntidiagonal, algebraMap R[X] _ (X ^ i.snd - 1) ^ μ i.fst := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq ((algebraMap (Polynomial R) (RatFunc R)) (Polynomial.cyclotomic n R)) (n. …
  -/
  rcases n.eq_zero_or_pos with (rfl | hpos)
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ Eq ((algebraMap (Polynomial R) (RatFunc R)) (Polynomial.cyclotomic 0 R)) ((N …
    -/
  · simp
    /-
      🎉 no goals
    -/
  have h : ∀ n : ℕ, 0 < n → (∏ i ∈ Nat.divisors n, algebraMap _ (RatFunc R) (cyclotomic i R)) =
      algebraMap _ _ (X ^ n - 1 : R[X]) := by
    intro n hn
    rw [← prod_cyclotomic_eq_X_pow_sub_one hn R, map_prod]
  /-
    case inr
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    hpos : GT.gt n 0
    h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
    ⊢ Eq ((algebraMap (Polynomial R) (RatFunc R)) (Polynomial.cyclotomic n R)) (n. …
  -/
  rw [(prod_eq_iff_prod_pow_moebius_eq_of_nonzero (fun n hn => _) fun n hn => _).1 h n hpos] <;>
    /-
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hpos : GT.gt n 0
      h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
      ⊢ ∀ (n : Nat), LT.lt 0 n → Ne ((algebraMap (Polynomial R) (RatFunc R)) (Polyno …
    -/
    simp_rw [Ne, IsFractionRing.to_map_eq_zero_iff]
    /-
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hpos : GT.gt n 0
      h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
      ⊢ ∀ (n : Nat), LT.lt 0 n → Not (Eq (Polynomial.cyclotomic n R) 0)
    -/
  · simp [cyclotomic_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hpos : GT.gt n 0
      h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
      ⊢ ∀ (n : Nat), LT.lt 0 n → Not (Eq (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0)
    -/
  · intro n hn
    /-
      n✝ : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hpos : GT.gt n✝ 0
      h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
      n : Nat
      hn : LT.lt 0 n
      ⊢ Not (Eq (HSub.hSub (HPow.hPow Polynomial.X n) 1) 0)
    -/
    apply Monic.ne_zero
    /-
      case hp
      n✝ : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hpos : GT.gt n✝ 0
      h : ∀ (n : Nat), LT.lt 0 n → Eq (n.divisors.prod fun i => (algebraMap (Polynom …
      n : Nat
      hn : LT.lt 0 n
      ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) 1).Monic
    -/
    apply monic_X_pow_sub_C _ (ne_of_gt hn)
    /-
      🎉 no goals
    -/


/-- We have
`cyclotomic n R = (X ^ k - 1) /ₘ (∏ i ∈ Nat.properDivisors k, cyclotomic i K)`. -/
theorem cyclotomic_eq_X_pow_sub_one_div {R : Type*} [CommRing R] {n : ℕ} (hpos : 0 < n) :
    cyclotomic n R = (X ^ n - 1) /ₘ ∏ i ∈ Nat.properDivisors n, cyclotomic i R := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    ⊢ Eq (Polynomial.cyclotomic n R) ((HSub.hSub (HPow.hPow Polynomial.X n) 1).div …
  -/
  nontriviality R
  rw [← prod_cyclotomic_eq_X_pow_sub_one hpos, ← Nat.cons_self_properDivisors hpos.ne',
    Finset.prod_cons]
  have prod_monic : (∏ i ∈ Nat.properDivisors n, cyclotomic i R).Monic := by
    apply monic_prod_of_monic
    intro i _
    exact cyclotomic.monic i R
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    ⊢ Eq (Polynomial.cyclotomic n R) ((HMul.hMul (Polynomial.cyclotomic n R) (n.pr …
  -/
  rw [(div_modByMonic_unique (cyclotomic n R) 0 prod_monic _).1]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    ⊢ And (Eq (HAdd.hAdd 0 (HMul.hMul (n.properDivisors.prod fun i => Polynomial.c …
  -/
  simp only [degree_zero, zero_add]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    ⊢ And (Eq (HMul.hMul (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      a✝ : Nontrivial R
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      ⊢ Eq (HMul.hMul (n.properDivisors.prod fun i => Polynomial.cyclotomic i R) (Po …
    -/
  · rw [mul_comm]
    /-
      🎉 no goals
    -/
  /-
    case right
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    ⊢ LT.lt Bot.bot (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).deg …
  -/
  rw [bot_lt_iff_ne_bot]
  /-
    case right
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    ⊢ Ne (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).degree Bot.bot
  -/
  intro h
  /-
    case right
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    a✝ : Nontrivial R
    prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
    h : Eq (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).degree Bot.bot
    ⊢ False
  -/
  exact Monic.ne_zero prod_monic (degree_eq_bot.1 h)
  /-
    🎉 no goals
  -/


/-- If `m` is a proper divisor of `n`, then `X ^ m - 1` divides
`∏ i ∈ Nat.properDivisors n, cyclotomic i R`. -/
theorem X_pow_sub_one_dvd_prod_cyclotomic (R : Type*) [CommRing R] {n m : ℕ} (hpos : 0 < n)
    (hm : m ∣ n) (hdiff : m ≠ n) : X ^ m - 1 ∣ ∏ i ∈ Nat.properDivisors n, cyclotomic i R := by
  replace hm := Nat.mem_properDivisors.2
    ⟨hm, lt_of_le_of_ne (Nat.divisor_le (Nat.mem_divisors.2 ⟨hm, hpos.ne'⟩)) hdiff⟩
  rw [← Finset.sdiff_union_of_subset (Nat.divisors_subset_properDivisors (ne_of_lt hpos).symm
    (Nat.mem_properDivisors.1 hm).1 (ne_of_lt (Nat.mem_properDivisors.1 hm).2)),
    Finset.prod_union Finset.sdiff_disjoint,
    prod_cyclotomic_eq_X_pow_sub_one (Nat.pos_of_mem_properDivisors hm)]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n m : Nat
    hpos : LT.lt 0 n
    hdiff : Ne m n
    hm : Membership.mem n.properDivisors m
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow Polynomial.X m) 1) (HMul.hMul ((SDiff.sdiff n. …
  -/
  exact ⟨∏ x ∈ n.properDivisors \ m.divisors, cyclotomic x R, by rw [mul_comm]⟩
  /-
    🎉 no goals
  -/


/-- If there is a primitive `n`-th root of unity in `K`, then
`cyclotomic n K = ∏ μ ∈ primitiveRoots n K, (X - C μ)`. ∈ particular,
`cyclotomic n K = cyclotomic' n K` -/
theorem cyclotomic_eq_prod_X_sub_primitiveRoots {K : Type*} [CommRing K] [IsDomain K] {ζ : K}
    {n : ℕ} (hz : IsPrimitiveRoot ζ n) : cyclotomic n K = ∏ μ ∈ primitiveRoots n K, (X - C μ) := by
  /-
    K : Type u_1
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hz : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.cyclotomic n K) ((primitiveRoots n K).prod fun μ => HSub.hSub …
  -/
  rw [← cyclotomic']
  /-
    K : Type u_1
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    ζ : K
    n : Nat
    hz : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.cyclotomic n K) (Polynomial.cyclotomic' n K)
  -/
  induction' n using Nat.strong_induction_on with k hk generalizing ζ
  /-
    case h
    K : Type u_1
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    k : Nat
    hk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Eq (Polynomial. …
    ζ : K
    hz : IsPrimitiveRoot ζ k
    ⊢ Eq (Polynomial.cyclotomic k K) (Polynomial.cyclotomic' k K)
  -/
  obtain hzero | hpos := k.eq_zero_or_pos
    /-
      case h.inl
      K : Type u_1
      inst✝¹ : CommRing K
      inst✝ : IsDomain K
      k : Nat
      hk : ∀ (m : Nat), LT.lt m k → ∀ {ζ : K}, IsPrimitiveRoot ζ m → Eq (Polynomial. …
      ζ : K
      hz : IsPrimitiveRoot ζ k
      hzero : Eq k 0
      ⊢ Eq (Polynomial.cyclotomic k K) (Polynomial.cyclotomic' k K)
    -/
  · simp only [hzero, cyclotomic'_zero, cyclotomic_zero]
    /-
      🎉 no goals
    -/
  have h : ∀ i ∈ k.properDivisors, cyclotomic i K = cyclotomic' i K := by
    intro i hi
    obtain ⟨d, hd⟩ := (Nat.mem_properDivisors.1 hi).1
    rw [mul_comm] at hd
    exact hk i (Nat.mem_properDivisors.1 hi).2 (IsPrimitiveRoot.pow hpos hz hd)
  rw [@cyclotomic_eq_X_pow_sub_one_div _ _ _ hpos, cyclotomic'_eq_X_pow_sub_one_div hpos hz,
    Finset.prod_congr (refl k.properDivisors) h]


theorem eq_cyclotomic_iff {R : Type*} [CommRing R] {n : ℕ} (hpos : 0 < n) (P : R[X]) :
    P = cyclotomic n R ↔
    (P * ∏ i ∈ Nat.properDivisors n, Polynomial.cyclotomic i R) = X ^ n - 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    P : Polynomial R
    ⊢ Iff (Eq P (Polynomial.cyclotomic n R)) (Eq (HMul.hMul P (n.properDivisors.pr …
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hpos : LT.lt 0 n
    P : Polynomial R
    a✝ : Nontrivial R
    ⊢ Iff (Eq P (Polynomial.cyclotomic n R)) (Eq (HMul.hMul P (n.properDivisors.pr …
  -/
  refine ⟨fun hcycl => ?_, fun hP => ?_⟩
  · rw [hcycl, ← prod_cyclotomic_eq_X_pow_sub_one hpos R, ← Nat.cons_self_properDivisors hpos.ne',
      Finset.prod_cons]
  · have prod_monic : (∏ i ∈ Nat.properDivisors n, cyclotomic i R).Monic := by
      apply monic_prod_of_monic
      intro i _
      exact cyclotomic.monic i R
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      P : Polynomial R
      a✝ : Nontrivial R
      hP : Eq (HMul.hMul P (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      ⊢ Eq P (Polynomial.cyclotomic n R)
    -/
    rw [@cyclotomic_eq_X_pow_sub_one_div R _ _ hpos, (div_modByMonic_unique P 0 prod_monic _).1]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      P : Polynomial R
      a✝ : Nontrivial R
      hP : Eq (HMul.hMul P (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      ⊢ And (Eq (HAdd.hAdd 0 (HMul.hMul (n.properDivisors.prod fun i => Polynomial.c …
    -/
    refine ⟨by rwa [zero_add, mul_comm], ?_⟩
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      P : Polynomial R
      a✝ : Nontrivial R
      hP : Eq (HMul.hMul P (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      ⊢ LT.lt (Polynomial.degree 0) (n.properDivisors.prod fun i => Polynomial.cyclo …
    -/
    rw [degree_zero, bot_lt_iff_ne_bot]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      P : Polynomial R
      a✝ : Nontrivial R
      hP : Eq (HMul.hMul P (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      ⊢ Ne (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).degree Bot.bot
    -/
    intro h
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hpos : LT.lt 0 n
      P : Polynomial R
      a✝ : Nontrivial R
      hP : Eq (HMul.hMul P (n.properDivisors.prod fun i => Polynomial.cyclotomic i R …
      prod_monic : (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).Monic
      h : Eq (n.properDivisors.prod fun i => Polynomial.cyclotomic i R).degree Bot.bot
      ⊢ False
    -/
    exact Monic.ne_zero prod_monic (degree_eq_bot.1 h)
    /-
      🎉 no goals
    -/


/-- If `p ^ k` is a prime power, then
`cyclotomic (p ^ (n + 1)) R = ∑ i ∈ range p, (X ^ (p ^ n)) ^ i`. -/
theorem cyclotomic_prime_pow_eq_geom_sum {R : Type*} [CommRing R] {p n : ℕ} (hp : p.Prime) :
    cyclotomic (p ^ (n + 1)) R = ∑ i ∈ Finset.range p, (X ^ p ^ n) ^ i := by
  have : ∀ m, (cyclotomic (p ^ (m + 1)) R = ∑ i ∈ Finset.range p, (X ^ p ^ m) ^ i) ↔
      ((∑ i ∈ Finset.range p, (X ^ p ^ m) ^ i) *
        ∏ x ∈ Finset.range (m + 1), cyclotomic (p ^ x) R) = X ^ p ^ (m + 1) - 1 := by
    intro m
    have := eq_cyclotomic_iff (R := R) (P := ∑ i ∈ range p, (X ^ p ^ m) ^ i)
      (pow_pos hp.pos (m + 1))
    rw [eq_comm] at this
    rw [this, Nat.prod_properDivisors_prime_pow hp]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p n : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    ⊢ Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) R) ((Finset.range p) …
  -/
  induction' n with n_n n_ih
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
      ⊢ Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd 0 1)) R) ((Finset.range p) …
    -/
  · haveI := Fact.mk hp; simp [cyclotomic_prime]
                         /-
                           🎉 no goals
                         -/
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    n_n : Nat
    n_ih : Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n_n 1)) R) ((Finset.r …
    ⊢ Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n_n 1) 1)) R) ( …
  -/
  rw [((eq_cyclotomic_iff (pow_pos hp.pos (n_n + 1 + 1)) _).mpr _).symm]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    n_n : Nat
    n_ih : Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n_n 1)) R) ((Finset.r …
    ⊢ Eq (HMul.hMul ((Finset.range p).sum fun i => HPow.hPow (HPow.hPow Polynomial …
  -/
  rw [Nat.prod_properDivisors_prime_pow hp, Finset.prod_range_succ, n_ih]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    n_n : Nat
    n_ih : Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n_n 1)) R) ((Finset.r …
    ⊢ Eq (HMul.hMul ((Finset.range p).sum fun i => HPow.hPow (HPow.hPow Polynomial …
  -/
  rw [this] at n_ih
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    n_n : Nat
    n_ih : Eq (HMul.hMul ((Finset.range p).sum fun i => HPow.hPow (HPow.hPow Polyn …
    ⊢ Eq (HMul.hMul ((Finset.range p).sum fun i => HPow.hPow (HPow.hPow Polynomial …
  -/
  rw [mul_comm _ (∑ i ∈ _, _), n_ih, geom_sum_mul, sub_left_inj, ← pow_mul]
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    this : ∀ (m : Nat), Iff (Eq (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 1 …
    n_n : Nat
    n_ih : Eq (HMul.hMul ((Finset.range p).sum fun i => HPow.hPow (HPow.hPow Polyn …
    ⊢ Eq (HPow.hPow Polynomial.X (HMul.hMul (HPow.hPow p (HAdd.hAdd n_n 1)) p)) (H …
  -/
  simp only [pow_add, pow_one]
  /-
    🎉 no goals
  -/


theorem cyclotomic_prime_pow_mul_X_pow_sub_one (R : Type*) [CommRing R] (p k : ℕ)
    [hn : Fact (Nat.Prime p)] :
    cyclotomic (p ^ (k + 1)) R * (X ^ p ^ k - 1) = X ^ p ^ (k + 1) - 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p k : Nat
    hn : Fact (Nat.Prime p)
    ⊢ Eq (HMul.hMul (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd k 1)) R) (HSub. …
  -/
  rw [cyclotomic_prime_pow_eq_geom_sum hn.out, geom_sum_mul, ← pow_mul, pow_succ, mul_comm]
  /-
    🎉 no goals
  -/


/-- The constant term of `cyclotomic n R` is `1` if `2 ≤ n`. -/
theorem cyclotomic_coeff_zero (R : Type*) [CommRing R] {n : ℕ} (hn : 1 < n) :
    (cyclotomic n R).coeff 0 = 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hn : LT.lt 1 n
    ⊢ Eq ((Polynomial.cyclotomic n R).coeff 0) 1
  -/
  induction' n using Nat.strong_induction_on with n hi
  have hprod : (∏ i ∈ Nat.properDivisors n, (Polynomial.cyclotomic i R).coeff 0) = -1 := by
    rw [← Finset.insert_erase (Nat.one_mem_properDivisors_iff_one_lt.2
      (lt_of_lt_of_le one_lt_two hn)), Finset.prod_insert (Finset.not_mem_erase 1 _),
      cyclotomic_one R]
    have hleq : ∀ j ∈ n.properDivisors.erase 1, 2 ≤ j := by
      intro j hj
      apply Nat.succ_le_of_lt
      exact (Ne.le_iff_lt (Finset.mem_erase.1 hj).1.symm).mp
        (Nat.succ_le_of_lt (Nat.pos_of_mem_properDivisors (Finset.mem_erase.1 hj).2))
    have hcongr : ∀ j ∈ n.properDivisors.erase 1, (cyclotomic j R).coeff 0 = 1 := by
      intro j hj
      exact hi j (Nat.mem_properDivisors.1 (Finset.mem_erase.1 hj).2).2 (hleq j hj)
    have hrw : (∏ x ∈ n.properDivisors.erase 1, (cyclotomic x R).coeff 0) = 1 := by
      rw [Finset.prod_congr (refl (n.properDivisors.erase 1)) hcongr]
      simp only [Finset.prod_const_one]
    simp only [hrw, mul_one, zero_sub, coeff_one_zero, coeff_X_zero, coeff_sub]
  have heq : (X ^ n - 1 : R[X]).coeff 0 = -(cyclotomic n R).coeff 0 := by
    rw [← prod_cyclotomic_eq_X_pow_sub_one (zero_le_one.trans_lt hn), ←
      Nat.cons_self_properDivisors hn.ne_bot, Finset.prod_cons, mul_coeff_zero, coeff_zero_prod,
      hprod, mul_neg, mul_one]
  have hzero : (X ^ n - 1 : R[X]).coeff 0 = (-1 : R) := by
    rw [coeff_zero_eq_eval_zero _]
    simp only [zero_pow (by positivity : n ≠ 0), eval_X, eval_one, zero_sub, eval_pow, eval_sub]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hi : ∀ (m : Nat), LT.lt m n → LT.lt 1 m → Eq ((Polynomial.cyclotomic m R).coef …
    hn : LT.lt 1 n
    hprod : Eq (n.properDivisors.prod fun i => (Polynomial.cyclotomic i R).coeff 0 …
    heq : Eq ((HSub.hSub (HPow.hPow Polynomial.X n) 1).coeff 0) (Neg.neg ((Polynom …
    hzero : Eq ((HSub.hSub (HPow.hPow Polynomial.X n) 1).coeff 0) (-1)
    ⊢ Eq ((Polynomial.cyclotomic n R).coeff 0) 1
  -/
  rw [hzero] at heq
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    hi : ∀ (m : Nat), LT.lt m n → LT.lt 1 m → Eq ((Polynomial.cyclotomic m R).coef …
    hn : LT.lt 1 n
    hprod : Eq (n.properDivisors.prod fun i => (Polynomial.cyclotomic i R).coeff 0 …
    heq : Eq (-1) (Neg.neg ((Polynomial.cyclotomic n R).coeff 0))
    hzero : Eq ((HSub.hSub (HPow.hPow Polynomial.X n) 1).coeff 0) (-1)
    ⊢ Eq ((Polynomial.cyclotomic n R).coeff 0) 1
  -/
  exact neg_inj.mp (Eq.symm heq)
  /-
    🎉 no goals
  -/


/-- If `(a : ℕ)` is a root of `cyclotomic n (ZMod p)`, where `p` is a prime, then `a` and `p` are
coprime. -/
theorem coprime_of_root_cyclotomic {n : ℕ} (hpos : 0 < n) {p : ℕ} [hprime : Fact p.Prime] {a : ℕ}
    (hroot : IsRoot (cyclotomic n (ZMod p)) (Nat.castRingHom (ZMod p) a)) : a.Coprime p := by
  /-
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    ⊢ a.Coprime p
  -/
  apply Nat.Coprime.symm
  /-
    case a
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    ⊢ p.Coprime a
  -/
  rw [hprime.1.coprime_iff_not_dvd]
  /-
    case a
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    ⊢ Not (Dvd.dvd p a)
  -/
  intro h
  /-
    case a
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    h : Dvd.dvd p a
    ⊢ False
  -/
  replace h := (ZMod.natCast_zmod_eq_zero_iff_dvd a p).2 h
  /-
    case a
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    h : Eq (↑a) 0
    ⊢ False
  -/
  rw [IsRoot.def, eq_natCast, h, ← coeff_zero_eq_eval_zero] at hroot
  /-
    case a
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : Eq ((Polynomial.cyclotomic n (ZMod p)).coeff 0) 0
    h : Eq (↑a) 0
    ⊢ False
  -/
  by_cases hone : n = 1
  · simp only [hone, cyclotomic_one, zero_sub, coeff_one_zero, coeff_X_zero, neg_eq_zero,
      one_ne_zero, coeff_sub] at hroot
  rw [cyclotomic_coeff_zero (ZMod p) (Nat.succ_le_of_lt
    (lt_of_le_of_ne (Nat.succ_le_of_lt hpos) (Ne.symm hone)))] at hroot
  /-
    case neg
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    hprime : Fact (Nat.Prime p)
    a : Nat
    hroot : Eq 1 0
    h : Eq (↑a) 0
    hone : Not (Eq n 1)
    ⊢ False
  -/
  exact one_ne_zero hroot
  /-
    🎉 no goals
  -/


/-- If `(a : ℕ)` is a root of `cyclotomic n (ZMod p)`, then the multiplicative order of `a` modulo
`p` divides `n`. -/
theorem orderOf_root_cyclotomic_dvd {n : ℕ} (hpos : 0 < n) {p : ℕ} [Fact p.Prime] {a : ℕ}
    (hroot : IsRoot (cyclotomic n (ZMod p)) (Nat.castRingHom (ZMod p) a)) :
    orderOf (ZMod.unitOfCoprime a (coprime_of_root_cyclotomic hpos hroot)) ∣ n := by
  /-
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    ⊢ Dvd.dvd (orderOf (ZMod.unitOfCoprime a ⋯)) n
  -/
  apply orderOf_dvd_of_pow_eq_one
  suffices hpow : eval (Nat.castRingHom (ZMod p) a) (X ^ n - 1 : (ZMod p)[X]) = 0 by
    simp only [eval_X, eval_one, eval_pow, eval_sub, eq_natCast] at hpow
    apply Units.val_eq_one.1
    simp only [sub_eq_zero.mp hpow, ZMod.coe_unitOfCoprime, Units.val_pow_eq_pow_val]
  /-
    case h
    n : Nat
    hpos : LT.lt 0 n
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    hroot : (Polynomial.cyclotomic n (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) a)
    ⊢ Eq (Polynomial.eval ((Nat.castRingHom (ZMod p)) a) (HSub.hSub (HPow.hPow Pol …
  -/
  rw [IsRoot.def] at hroot
  rw [← prod_cyclotomic_eq_X_pow_sub_one hpos (ZMod p), ← Nat.cons_self_properDivisors hpos.ne',
    Finset.prod_cons, eval_mul, hroot, zero_mul]


lemma dvd_C_mul_X_sub_one_pow_add_one {p : ℕ} (hpri : p.Prime)
    (hp : p ≠ 2) (a r : R) (h₁ : r ∣ a ^ p) (h₂ : r ∣ p * a) : C r ∣ (C a * X - 1) ^ p + 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hpri : Nat.Prime p
    hp : Ne p 2
    a r : R
    h₁ : Dvd.dvd r (HPow.hPow a p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) a)
    ⊢ Dvd.dvd (Polynomial.C r) (HAdd.hAdd (HPow.hPow (HSub.hSub (HMul.hMul (Polyno …
  -/
  have := hpri.dvd_add_pow_sub_pow_of_dvd (C a * X) (-1) (r := C r) ?_ ?_
    /-
      case refine_3
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hpri : Nat.Prime p
      hp : Ne p 2
      a r : R
      h₁ : Dvd.dvd r (HPow.hPow a p)
      h₂ : Dvd.dvd r (HMul.hMul (↑p) a)
      this : Dvd.dvd (Polynomial.C r) (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul (P …
      ⊢ Dvd.dvd (Polynomial.C r) (HAdd.hAdd (HPow.hPow (HSub.hSub (HMul.hMul (Polyno …
    -/
  · rwa [← sub_eq_add_neg, (hpri.odd_of_ne_two hp).neg_pow, one_pow, sub_neg_eq_add] at this
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hpri : Nat.Prime p
      hp : Ne p 2
      a r : R
      h₁ : Dvd.dvd r (HPow.hPow a p)
      h₂ : Dvd.dvd r (HMul.hMul (↑p) a)
      ⊢ Dvd.dvd (Polynomial.C r) (HPow.hPow (HMul.hMul (Polynomial.C a) Polynomial.X …
    -/
  · simp only [mul_pow, ← map_pow, dvd_mul_right, (_root_.map_dvd C h₁).trans]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hpri : Nat.Prime p
    hp : Ne p 2
    a r : R
    h₁ : Dvd.dvd r (HPow.hPow a p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) a)
    ⊢ Dvd.dvd (Polynomial.C r) (HMul.hMul (↑p) (HMul.hMul (Polynomial.C a) Polynom …
  -/
  simp only [map_mul, map_natCast, ← mul_assoc, dvd_mul_right, (_root_.map_dvd C h₂).trans]
  /-
    🎉 no goals
  -/


private theorem _root_.IsPrimitiveRoot.pow_sub_pow_eq_prod_sub_mul_field {K : Type*}
    [Field K] {ζ : K} (x y : K) (hpos : 0 < n) (h : IsPrimitiveRoot ζ n) :
    x ^ n - y ^ n = ∏ ζ ∈ nthRootsFinset n K, (x - ζ * y) := by
  /-
    n : Nat
    K : Type u_2
    inst✝ : Field K
    ζ x y : K
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)) ((Polynomial.nthRootsFinset n …
  -/
  by_cases hy : y = 0
    /-
      case pos
      n : Nat
      K : Type u_2
      inst✝ : Field K
      ζ x y : K
      hpos : LT.lt 0 n
      h : IsPrimitiveRoot ζ n
      hy : Eq y 0
      ⊢ Eq (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)) ((Polynomial.nthRootsFinset n …
    -/
  · simp only [hy, zero_pow (Nat.not_eq_zero_of_lt hpos), sub_zero, mul_zero, prod_const]
    /-
      case pos
      n : Nat
      K : Type u_2
      inst✝ : Field K
      ζ x y : K
      hpos : LT.lt 0 n
      h : IsPrimitiveRoot ζ n
      hy : Eq y 0
      ⊢ Eq (HPow.hPow x n) (HPow.hPow x (Polynomial.nthRootsFinset n K).card)
    -/
    congr
    /-
      case pos.e_a
      n : Nat
      K : Type u_2
      inst✝ : Field K
      ζ x y : K
      hpos : LT.lt 0 n
      h : IsPrimitiveRoot ζ n
      hy : Eq y 0
      ⊢ Eq n (Polynomial.nthRootsFinset n K).card
    -/
    rw [h.card_nthRootsFinset]
    /-
      🎉 no goals
    -/
  convert congr_arg (eval (x/y) · * y ^ card (nthRootsFinset n K)) <| X_pow_sub_one_eq_prod hpos h
    using 1
    /-
      case h.e'_2
      n : Nat
      K : Type u_2
      inst✝ : Field K
      ζ x y : K
      hpos : LT.lt 0 n
      h : IsPrimitiveRoot ζ n
      hy : Not (Eq y 0)
      ⊢ Eq (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)) (HMul.hMul (Polynomial.eval ( …
    -/
  · simp [sub_mul, div_pow, hy, h.card_nthRootsFinset]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      n : Nat
      K : Type u_2
      inst✝ : Field K
      ζ x y : K
      hpos : LT.lt 0 n
      h : IsPrimitiveRoot ζ n
      hy : Not (Eq y 0)
      ⊢ Eq ((Polynomial.nthRootsFinset n K).prod fun ζ => HSub.hSub x (HMul.hMul ζ y …
    -/
  · simp [eval_prod, prod_mul_pow_card, sub_mul, hy]
    /-
      🎉 no goals
    -/


/-- If there is a primitive `n`th root of unity in `R`, then `X ^ n - Y ^ n = ∏ (X - μ Y)`,
where `μ` varies over the `n`-th roots of unity. -/
theorem _root_.IsPrimitiveRoot.pow_sub_pow_eq_prod_sub_mul (hpos : 0 < n)
    (h : IsPrimitiveRoot ζ n) : x ^ n - y ^ n = ∏ ζ ∈ nthRootsFinset n R, (x - ζ * y) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)) ((Polynomial.nthRootsFinset n …
  -/
  let K := FractionRing R
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    ⊢ Eq (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)) ((Polynomial.nthRootsFinset n …
  -/
  apply NoZeroSMulDivisors.algebraMap_injective R K
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    ⊢ Eq ((algebraMap R K) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) ((algebraM …
  -/
  rw [map_sub, map_pow, map_pow, map_prod]
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    ⊢ Eq (HSub.hSub (HPow.hPow ((algebraMap R K) x) n) (HPow.hPow ((algebraMap R K …
  -/
  simp_rw [map_sub, map_mul]
  have h' : IsPrimitiveRoot (algebraMap R K ζ) n :=
    h.map_of_injective <| NoZeroSMulDivisors.algebraMap_injective R K
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    h' : IsPrimitiveRoot ((algebraMap R K) ζ) n
    ⊢ Eq (HSub.hSub (HPow.hPow ((algebraMap R K) x) n) (HPow.hPow ((algebraMap R K …
  -/
  rw [h'.pow_sub_pow_eq_prod_sub_mul_field _ _ hpos]
  refine (prod_nbij (algebraMap R K) (fun a ha ↦ map_mem_nthRootsFinset ha _) (fun a _ b _ H ↦
    NoZeroSMulDivisors.algebraMap_injective R K H) (fun a ha ↦ ?_) (fun _ _ ↦ rfl)).symm
  have := Set.surj_on_of_inj_on_of_ncard_le (s := nthRootsFinset n R)
    (t := nthRootsFinset n K) _ (fun _ hr ↦ map_mem_nthRootsFinset hr _)
    (fun a _ b _ H ↦ NoZeroSMulDivisors.algebraMap_injective R K H)
    (by simp [h.card_nthRootsFinset, h'.card_nthRootsFinset])
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    h' : IsPrimitiveRoot ((algebraMap R K) ζ) n
    a : K
    ha : Membership.mem (↑(Polynomial.nthRootsFinset n K)) a
    this : ∀ (b : K), Membership.mem (↑(Polynomial.nthRootsFinset n K)) b → Exists …
    ⊢ Membership.mem (Set.image ⇑(algebraMap R K) ↑(Polynomial.nthRootsFinset n R) …
  -/
  obtain ⟨x, hx, hx1⟩ := this _ ha
  /-
    case a.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x✝ y : R
    inst✝ : IsDomain R
    hpos : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    K : Type u_1 := FractionRing R
    h' : IsPrimitiveRoot ((algebraMap R K) ζ) n
    a : K
    ha : Membership.mem (↑(Polynomial.nthRootsFinset n K)) a
    this : ∀ (b : K), Membership.mem (↑(Polynomial.nthRootsFinset n K)) b → Exists …
    x : R
    hx : Membership.mem (↑(Polynomial.nthRootsFinset n R)) x
    hx1 : Eq a ((algebraMap R K) x)
    ⊢ Membership.mem (Set.image ⇑(algebraMap R K) ↑(Polynomial.nthRootsFinset n R) …
  -/
  exact ⟨x, hx, hx1.symm⟩
  /-
    🎉 no goals
  -/


/-- If there is a primitive `n`th root of unity in `R` and `n` is odd, then
`X ^ n + Y ^ n = ∏ (X + μ Y)`, where `μ` varies over the `n`-th roots of unity. -/
theorem _root_.IsPrimitiveRoot.pow_add_pow_eq_prod_add_mul (hodd : Odd n)
    (h : IsPrimitiveRoot ζ n) : x ^ n + y ^ n = ∏ ζ ∈ nthRootsFinset n R, (x + ζ * y) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    ζ : R
    n : Nat
    x y : R
    inst✝ : IsDomain R
    hodd : Odd n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n)) ((Polynomial.nthRootsFinset n …
  -/
  simpa [hodd.neg_pow] using h.pow_sub_pow_eq_prod_sub_mul x (-y) hodd.pos
  /-
    🎉 no goals
  -/


