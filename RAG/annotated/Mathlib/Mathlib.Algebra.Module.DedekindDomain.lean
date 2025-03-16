/-- Over a Dedekind domain, an `I`-torsion module is the internal direct sum of its `p i ^ e i`-
torsion submodules, where `I = ∏ i, p i ^ e i` is its unique decomposition in prime ideals. -/
theorem isInternal_prime_power_torsion_of_is_torsion_by_ideal [DecidableEq (Ideal R)]
    {I : Ideal R} (hI : I ≠ ⊥) (hM : Module.IsTorsionBySet R M I) :
    DirectSum.IsInternal fun p : (factors I).toFinset =>
      torsionBySet R M (p ^ (factors I).count ↑p : Ideal R) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    M : Type v
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsDedekindDomain R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    hI : Ne I Bot.bot
    hM : Module.IsTorsionBySet R M ↑I
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBySet R M ↑(HPow.hPow (↑p) (M …
  -/
  let P := factors I
  have prime_of_mem := fun p (hp : p ∈ P.toFinset) =>
    prime_of_factor p (Multiset.mem_toFinset.mp hp)
  /-
    R : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    M : Type v
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsDedekindDomain R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    hI : Ne I Bot.bot
    hM : Module.IsTorsionBySet R M ↑I
    P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
    prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBySet R M ↑(HPow.hPow (↑p) (M …
  -/
  apply torsionBySet_isInternal (p := fun p => p ^ P.count p) _
    /-
      R : Type u
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsDedekindDomain R
      inst✝ : DecidableEq (Ideal R)
      I : Ideal R
      hI : Ne I Bot.bot
      hM : Module.IsTorsionBySet R M ↑I
      P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
      prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
      ⊢ Module.IsTorsionBySet R M ↑(iInf fun i => iInf fun h => HPow.hPow i (Multise …
    -/
  · convert hM
    rw [← Finset.inf_eq_iInf, IsDedekindDomain.inf_prime_pow_eq_prod, ← Finset.prod_multiset_count,
      ← associated_iff_eq]
      /-
        case h.e'_6.h.e'_4
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        ⊢ Associated (UniqueFactorizationMonoid.factors I).prod I
      -/
    · exact factors_prod hI
      /-
        🎉 no goals
      -/
      /-
        case h.e'_6.h.e'_4.prime
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        ⊢ ∀ (i : Ideal R), Membership.mem (UniqueFactorizationMonoid.factors I).toFins …
      -/
    · exact prime_of_mem
      /-
        🎉 no goals
      -/
      /-
        case h.e'_6.h.e'_4.coprime
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        ⊢ ∀ (i : Ideal R), Membership.mem (UniqueFactorizationMonoid.factors I).toFins …
      -/
    · exact fun _ _ _ _ ij => ij
      /-
        🎉 no goals
      -/
    /-
      R : Type u
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsDedekindDomain R
      inst✝ : DecidableEq (Ideal R)
      I : Ideal R
      hI : Ne I Bot.bot
      hM : Module.IsTorsionBySet R M ↑I
      P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
      prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
      ⊢ (↑(UniqueFactorizationMonoid.factors I).toFinset).Pairwise fun i j => Eq (Ma …
    -/
  · intro p hp q hq pq; dsimp
    /-
      R : Type u
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsDedekindDomain R
      inst✝ : DecidableEq (Ideal R)
      I : Ideal R
      hI : Ne I Bot.bot
      hM : Module.IsTorsionBySet R M ↑I
      P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
      prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
      p : Ideal R
      hp : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) p
      q : Ideal R
      hq : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) q
      pq : Ne p q
      ⊢ Eq (Max.max (HPow.hPow p (Multiset.count p P)) (HPow.hPow q (Multiset.count  …
    -/
    rw [irreducible_pow_sup]
      /-
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        p : Ideal R
        hp : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) p
        q : Ideal R
        hq : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) q
        pq : Ne p q
        ⊢ Eq (HPow.hPow p (Min.min (Multiset.count p (UniqueFactorizationMonoid.normal …
      -/
    · suffices (normalizedFactors _).count p = 0 by rw [this, zero_min, pow_zero, Ideal.one_eq_top]
      rw [Multiset.count_eq_zero,
        normalizedFactors_of_irreducible_pow (prime_of_mem q hq).irreducible,
        Multiset.mem_replicate]
      /-
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        p : Ideal R
        hp : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) p
        q : Ideal R
        hq : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) q
        pq : Ne p q
        ⊢ Not (And (Ne (Multiset.count q P) 0) (Eq p (normalize q)))
      -/
      exact fun H => pq <| H.2.trans <| normalize_eq q
      /-
        🎉 no goals
      -/
      /-
        case hI
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        p : Ideal R
        hp : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) p
        q : Ideal R
        hq : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) q
        pq : Ne p q
        ⊢ Ne (HPow.hPow q (Multiset.count q P)) Bot.bot
      -/
    · rw [← Ideal.zero_eq_bot]; apply pow_ne_zero; exact (prime_of_mem q hq).ne_zero
                                                   /-
                                                     🎉 no goals
                                                   -/
      /-
        case hJ
        R : Type u
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        M : Type v
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : IsDedekindDomain R
        inst✝ : DecidableEq (Ideal R)
        I : Ideal R
        hI : Ne I Bot.bot
        hM : Module.IsTorsionBySet R M ↑I
        P : Multiset (Ideal R) := UniqueFactorizationMonoid.factors I
        prime_of_mem : ∀ (p : Ideal R), Membership.mem P.toFinset p → Prime p
        p : Ideal R
        hp : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) p
        q : Ideal R
        hq : Membership.mem (↑(UniqueFactorizationMonoid.factors I).toFinset) q
        pq : Ne p q
        ⊢ Irreducible p
      -/
    · exact (prime_of_mem p hp).irreducible
      /-
        🎉 no goals
      -/


/-- A finitely generated torsion module over a Dedekind domain is an internal direct sum of its
`p i ^ e i`-torsion submodules where `p i` are factors of `(⊤ : Submodule R M).annihilator` and
`e i` are their multiplicities. -/
theorem isInternal_prime_power_torsion [DecidableEq (Ideal R)] [Module.Finite R M]
    (hM : Module.IsTorsion R M) :
    DirectSum.IsInternal fun p : (factors (⊤ : Submodule R M).annihilator).toFinset =>
      torsionBySet R M (p ^ (factors (⊤ : Submodule R M).annihilator).count ↑p : Ideal R) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDedekindDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBySet R M ↑(HPow.hPow (↑p) (M …
  -/
  have hM' := Module.isTorsionBySet_annihilator_top R M
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDedekindDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    hM' : Module.IsTorsionBySet R M ↑Top.top.annihilator
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBySet R M ↑(HPow.hPow (↑p) (M …
  -/
  have hI := Submodule.annihilator_top_inter_nonZeroDivisors hM
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDedekindDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    hM' : Module.IsTorsionBySet R M ↑Top.top.annihilator
    hI : Ne (Inter.inter ↑Top.top.annihilator ↑(nonZeroDivisors R)) EmptyCollectio …
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBySet R M ↑(HPow.hPow (↑p) (M …
  -/
  refine isInternal_prime_power_torsion_of_is_torsion_by_ideal ?_ hM'
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDedekindDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    hM' : Module.IsTorsionBySet R M ↑Top.top.annihilator
    hI : Ne (Inter.inter ↑Top.top.annihilator ↑(nonZeroDivisors R)) EmptyCollectio …
    ⊢ Ne Top.top.annihilator Bot.bot
  -/
  rw [← Set.nonempty_iff_ne_empty] at hI; rw [Submodule.ne_bot_iff]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDedekindDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    hM' : Module.IsTorsionBySet R M ↑Top.top.annihilator
    hI : (Inter.inter ↑Top.top.annihilator ↑(nonZeroDivisors R)).Nonempty
    ⊢ Exists fun x => And (Membership.mem Top.top.annihilator x) (Ne x 0)
  -/
  obtain ⟨x, H, hx⟩ := hI; exact ⟨x, H, nonZeroDivisors.ne_zero hx⟩
                           /-
                             🎉 no goals
                           -/


/-- A finitely generated torsion module over a Dedekind domain is an internal direct sum of its
`p i ^ e i`-torsion submodules for some prime ideals `p i` and numbers `e i`. -/
theorem exists_isInternal_prime_power_torsion [Module.Finite R M] (hM : Module.IsTorsion R M) :
    ∃ (P : Finset <| Ideal R) (_ : DecidableEq P) (_ : ∀ p ∈ P, Prime p) (e : P → ℕ),
      DirectSum.IsInternal fun p : P => torsionBySet R M (p ^ e p : Ideal R) := by
  classical
  exact ⟨_, _, fun p hp => prime_of_factor p (Multiset.mem_toFinset.mp hp), _,
    isInternal_prime_power_torsion hM⟩


