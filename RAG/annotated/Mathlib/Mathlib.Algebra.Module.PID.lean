theorem Submodule.isSemisimple_torsionBy_of_irreducible {a : R} (h : Irreducible a) :
    IsSemisimpleModule R (torsionBy R M a) :=
  haveI := PrincipalIdealRing.isMaximal_of_irreducible h
  letI := Ideal.Quotient.field (R ∙ a)
  (submodule_torsionBy_orderIso a).complementedLattice


/-- A finitely generated torsion module over a PID is an internal direct sum of its
`p i ^ e i`-torsion submodules for some primes `p i` and numbers `e i`. -/
theorem Submodule.isInternal_prime_power_torsion_of_pid [DecidableEq (Ideal R)] [Module.Finite R M]
    (hM : Module.IsTorsion R M) :
    DirectSum.IsInternal fun p : (factors (⊤ : Submodule R M).annihilator).toFinset =>
      torsionBy R M
        (IsPrincipal.generator (p : Ideal R) ^
          (factors (⊤ : Submodule R M).annihilator).count ↑p) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsPrincipalIdealRing R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    ⊢ DirectSum.IsInternal fun p => Submodule.torsionBy R M (HPow.hPow (Submodule. …
  -/
  convert isInternal_prime_power_torsion hM
  /-
    case h.e'_8.h
    R : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : IsPrincipalIdealRing R
    M : Type v
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq (Ideal R)
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    x✝ : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors Top.to …
    ⊢ Eq (Submodule.torsionBy R M (HPow.hPow (Submodule.IsPrincipal.generator ↑x✝) …
  -/
  ext p : 1
  rw [← torsionBySet_span_singleton_eq, Ideal.submodule_span_eq, ← Ideal.span_singleton_pow,
    Ideal.span_singleton_generator]


/-- A finitely generated torsion module over a PID is an internal direct sum of its
`p i ^ e i`-torsion submodules for some primes `p i` and numbers `e i`. -/
theorem Submodule.exists_isInternal_prime_power_torsion_of_pid [Module.Finite R M]
    (hM : Module.IsTorsion R M) :
    ∃ (ι : Type u) (_ : Fintype ι) (_ : DecidableEq ι) (p : ι → R) (_ : ∀ i, Irreducible <| p i)
        (e : ι → ℕ), DirectSum.IsInternal fun i => torsionBy R M <| p i ^ e i := by
  classical
  refine ⟨_, ?_, _, _, ?_, _, Submodule.isInternal_prime_power_torsion_of_pid hM⟩
  · exact Finset.fintypeCoeSort _
  · rintro ⟨p, hp⟩
    have hP := prime_of_factor p (Multiset.mem_toFinset.mp hp)
    haveI := Ideal.isPrime_of_prime hP
    exact (IsPrincipal.prime_generator_of_isPrime p hP.ne_zero).irreducible


theorem _root_.Ideal.torsionOf_eq_span_pow_pOrder (x : M) :
    torsionOf R M x = span {p ^ pOrder hM x} := by
  classical
  dsimp only [pOrder]
  rw [← (torsionOf R M x).span_singleton_generator, Ideal.span_singleton_eq_span_singleton, ←
    Associates.mk_eq_mk_iff_associated, Associates.mk_pow]
  have prop :
    (fun n : ℕ => p ^ n • x = 0) = fun n : ℕ =>
      (Associates.mk <| generator <| torsionOf R M x) ∣ Associates.mk p ^ n := by
    ext n; rw [← Associates.mk_pow, Associates.mk_dvd_mk, ← mem_iff_generator_dvd]; rfl
  have := (isTorsion'_powers_iff p).mp hM x; rw [prop] at this
  convert Associates.eq_pow_find_of_dvd_irreducible_pow (Associates.irreducible_mk.mpr hp)
    this.choose_spec


theorem p_pow_smul_lift {x y : M} {k : ℕ} (hM' : Module.IsTorsionBy R M (p ^ pOrder hM y))
    (h : p ^ k • x ∈ R ∙ y) : ∃ a : R, p ^ k • x = p ^ k • a • y := by
  -- Porting note: needed to make `smul_smul` work below.
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    M : Type v
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    dec : (x : M) → Decidable (Eq x 0)
    x y : M
    k : Nat
    hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
    h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
    ⊢ Exists fun a => Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p …
  -/
  letI : MulAction R M := MulActionWithZero.toMulAction
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    M : Type v
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    dec : (x : M) → Decidable (Eq x 0)
    x y : M
    k : Nat
    hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
    h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
    this : MulAction R M := MulActionWithZero.toMulAction
    ⊢ Exists fun a => Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p …
  -/
  by_cases hk : k ≤ pOrder hM y
  · let f :=
      ((R ∙ p ^ (pOrder hM y - k) * p ^ k).quotEquivOfEq _ ?_).trans
        (quotTorsionOfEquivSpanSingleton R M y)
    · have : f.symm ⟨p ^ k • x, h⟩ ∈
          R ∙ Ideal.Quotient.mk (R ∙ p ^ (pOrder hM y - k) * p ^ k) (p ^ k) := by
        rw [← Quotient.torsionBy_eq_span_singleton, mem_torsionBy_iff, ← f.symm.map_smul]
        · convert f.symm.map_zero; ext
          rw [coe_smul_of_tower, coe_mk, coe_zero, smul_smul, ← pow_add, Nat.sub_add_cancel hk,
            @hM' x]
        · exact mem_nonZeroDivisors_of_ne_zero (pow_ne_zero _ hp.ne_zero)
      /-
        case pos.refine_2
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        M : Type v
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsDomain R
        p : R
        hp : Irreducible p
        hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
        dec : (x : M) → Decidable (Eq x 0)
        x y : M
        k : Nat
        hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
        h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
        this✝ : MulAction R M := MulActionWithZero.toMulAction
        hk : LE.le k (Submodule.pOrder hM y)
        f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Sing …
        this : Membership.mem (Submodule.span R (Singleton.singleton ((Ideal.Quotient. …
        ⊢ Exists fun a => Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p …
      -/
      rw [Submodule.mem_span_singleton] at this; obtain ⟨a, ha⟩ := this; use a
      /-
        case h
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        M : Type v
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsDomain R
        p : R
        hp : Irreducible p
        hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
        dec : (x : M) → Decidable (Eq x 0)
        x y : M
        k : Nat
        hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
        h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
        this : MulAction R M := MulActionWithZero.toMulAction
        hk : LE.le k (Submodule.pOrder hM y)
        f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Sing …
        a : R
        ha : Eq (HSMul.hSMul a ((Ideal.Quotient.mk (Submodule.span R (Singleton.single …
        ⊢ Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p k) (HSMul.hSMul …
      -/
      rw [f.eq_symm_apply, ← Ideal.Quotient.mk_eq_mk, ← Quotient.mk_smul] at ha
      dsimp only [smul_eq_mul, LinearEquiv.trans_apply, Submodule.quotEquivOfEq_mk,
        quotTorsionOfEquivSpanSingleton_apply_mk] at ha
      /-
        case h
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        M : Type v
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsDomain R
        p : R
        hp : Irreducible p
        hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
        dec : (x : M) → Decidable (Eq x 0)
        x y : M
        k : Nat
        hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
        h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
        this : MulAction R M := MulActionWithZero.toMulAction
        hk : LE.le k (Submodule.pOrder hM y)
        f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Sing …
        a : R
        ha : Eq (f (Submodule.Quotient.mk (HMul.hMul a (HPow.hPow p k)))) ⟨HSMul.hSMul …
        ⊢ Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p k) (HSMul.hSMul …
      -/
      rw [smul_smul, mul_comm]; exact congr_arg ((↑) : _ → M) ha.symm
                                /-
                                  🎉 no goals
                                -/
      /-
        case pos.refine_1
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        M : Type v
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsDomain R
        p : R
        hp : Irreducible p
        hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
        dec : (x : M) → Decidable (Eq x 0)
        x y : M
        k : Nat
        hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
        h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
        this : MulAction R M := MulActionWithZero.toMulAction
        hk : LE.le k (Submodule.pOrder hM y)
        ⊢ Eq (Submodule.span R (Singleton.singleton (HMul.hMul (HPow.hPow p (HSub.hSub …
      -/
    · symm; convert Ideal.torsionOf_eq_span_pow_pOrder hp hM y
      /-
        case h.e'_3.h.e'_1.h.e'_4
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        M : Type v
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsDomain R
        p : R
        hp : Irreducible p
        hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
        dec : (x : M) → Decidable (Eq x 0)
        x y : M
        k : Nat
        hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
        h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
        this : MulAction R M := MulActionWithZero.toMulAction
        hk : LE.le k (Submodule.pOrder hM y)
        ⊢ Eq (HMul.hMul (HPow.hPow p (HSub.hSub (Submodule.pOrder hM y) k)) (HPow.hPow …
      -/
      rw [← pow_add, Nat.sub_add_cancel hk]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      M : Type v
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsDomain R
      p : R
      hp : Irreducible p
      hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
      dec : (x : M) → Decidable (Eq x 0)
      x y : M
      k : Nat
      hM' : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM y))
      h : Membership.mem (Submodule.span R (Singleton.singleton y)) (HSMul.hSMul (HP …
      this : MulAction R M := MulActionWithZero.toMulAction
      hk : Not (LE.le k (Submodule.pOrder hM y))
      ⊢ Exists fun a => Eq (HSMul.hSMul (HPow.hPow p k) x) (HSMul.hSMul (HPow.hPow p …
    -/
  · use 0
    rw [zero_smul, smul_zero, ← Nat.sub_add_cancel (le_of_not_le hk), pow_add, mul_smul, hM',
      smul_zero]


theorem exists_smul_eq_zero_and_mk_eq {z : M} (hz : Module.IsTorsionBy R M (p ^ pOrder hM z))
    {k : ℕ} (f : (R ⧸ R ∙ p ^ k) →ₗ[R] M ⧸ R ∙ z) :
    ∃ x : M, p ^ k • x = 0 ∧ Submodule.Quotient.mk (p := span R {z}) x = f 1 := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    M : Type v
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    dec : (x : M) → Decidable (Eq x 0)
    z : M
    hz : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM z))
    k : Nat
    f : LinearMap (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Single …
    ⊢ Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p k) x) 0) (Eq (Submodule.Qu …
  -/
  have f1 := mk_surjective (R ∙ z) (f 1)
  have : p ^ k • f1.choose ∈ R ∙ z := by
    rw [← Quotient.mk_eq_zero, mk_smul, f1.choose_spec, ← f.map_smul]
    convert f.map_zero; change _ • Submodule.Quotient.mk _ = _
    rw [← mk_smul, Quotient.mk_eq_zero, Algebra.id.smul_eq_mul, mul_one]
    exact Submodule.mem_span_singleton_self _
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    M : Type v
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    dec : (x : M) → Decidable (Eq x 0)
    z : M
    hz : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM z))
    k : Nat
    f : LinearMap (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Single …
    f1 : Exists fun a => Eq (Submodule.Quotient.mk a) (f 1)
    this : Membership.mem (Submodule.span R (Singleton.singleton z)) (HSMul.hSMul  …
    ⊢ Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p k) x) 0) (Eq (Submodule.Qu …
  -/
  obtain ⟨a, ha⟩ := p_pow_smul_lift hp hM hz this
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    M : Type v
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    dec : (x : M) → Decidable (Eq x 0)
    z : M
    hz : Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM z))
    k : Nat
    f : LinearMap (RingHom.id R) (HasQuotient.Quotient R (Submodule.span R (Single …
    f1 : Exists fun a => Eq (Submodule.Quotient.mk a) (f 1)
    this : Membership.mem (Submodule.span R (Singleton.singleton z)) (HSMul.hSMul  …
    a : R
    ha : Eq (HSMul.hSMul (HPow.hPow p k) f1.choose) (HSMul.hSMul (HPow.hPow p k) ( …
    ⊢ Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p k) x) 0) (Eq (Submodule.Qu …
  -/
  refine ⟨f1.choose - a • z, by rw [smul_sub, sub_eq_zero, ha], ?_⟩
  rw [mk_sub, mk_smul, (Quotient.mk_eq_zero _).mpr <| Submodule.mem_span_singleton_self _,
    smul_zero, sub_zero, f1.choose_spec]


/-- A finitely generated `p ^ ∞`-torsion module over a PID is isomorphic to a direct sum of some
  `R ⧸ R ∙ (p ^ e i)` for some `e i`. -/
theorem torsion_by_prime_power_decomposition (hN : Module.IsTorsion' N (Submonoid.powers p))
    [h' : Module.Finite R N] :
    ∃ (d : ℕ) (k : Fin d → ℕ), Nonempty <| N ≃ₗ[R] ⨁ i : Fin d, R ⧸ R ∙ p ^ (k i : ℕ) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
    h' : Module.Finite R N
    ⊢ Exists fun d => Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (Dire …
  -/
  obtain ⟨d, s, hs⟩ := @Module.Finite.exists_fin _ _ _ _ _ h'; use d; clear h'
  /-
    case h
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
    d : Nat
    s : Fin d → N
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin d) fu …
  -/
  induction' d with d IH generalizing N
  · -- Porting note: was `use fun i => finZeroElim i`
    /-
      case h.zero
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin 0 → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin 0) fu …
    -/
    use finZeroElim
    /-
      case h
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin 0 → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      ⊢ Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin 0) fun i => HasQuotie …
    -/
    rw [Set.range_eq_empty, Submodule.span_empty] at hs
    haveI : Unique N :=
      ⟨⟨0⟩, fun x => by dsimp; rw [← Submodule.mem_bot R, hs]; exact Submodule.mem_top⟩
    /-
      case h
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin 0 → N
      hs : Eq Bot.bot Top.top
      this : Unique N
      ⊢ Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin 0) fun i => HasQuotie …
    -/
    haveI : IsEmpty (Fin Nat.zero) := inferInstanceAs (IsEmpty (Fin 0))
    /-
      case h
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin 0 → N
      hs : Eq Bot.bot Top.top
      this✝ : Unique N
      this : IsEmpty (Fin Nat.zero)
      ⊢ Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin 0) fun i => HasQuotie …
    -/
    exact ⟨0⟩
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      d : Nat
      IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin (HAdd.hAdd d 1) → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
    -/
  · have : ∀ x : N, Decidable (x = 0) := fun _ => by classical infer_instance
    /-
      case h.succ
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      d : Nat
      IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin (HAdd.hAdd d 1) → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      this : (x : N) → Decidable (Eq x 0)
      ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
    -/
    obtain ⟨j, hj⟩ := exists_isTorsionBy hN d.succ d.succ_ne_zero s hs
    /-
      case h.succ.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      d : Nat
      IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin (HAdd.hAdd d 1) → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      this : (x : N) → Decidable (Eq x 0)
      j : Fin d.succ
      hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
      ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
    -/
    let s' : Fin d → N ⧸ R ∙ s j := Submodule.Quotient.mk ∘ s ∘ j.succAbove
    -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5732):
    -- `obtain` doesn't work with placeholders.
    /-
      case h.succ.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : IsDomain R
      p : R
      hp : Irreducible p
      d : Nat
      IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
      N : Type (max u v)
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
      s : Fin (HAdd.hAdd d 1) → N
      hs : Eq (Submodule.span R (Set.range s)) Top.top
      this : (x : N) → Decidable (Eq x 0)
      j : Fin d.succ
      hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
      s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
      ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
    -/
    have := IH ?_ s' ?_
      /-
        case h.succ.intro.refine_3
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this✝ : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        this : Exists fun k => Nonempty (LinearEquiv (RingHom.id R) (HasQuotient.Quoti …
        ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
      -/
    · obtain ⟨k, ⟨f⟩⟩ := this
      /-
        case h.succ.intro.refine_3.intro.intro
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        k : Fin d → Nat
        f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
        ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
      -/
      clear IH
      have : ∀ i : Fin d,
          ∃ x : N, p ^ k i • x = 0 ∧ f (Submodule.Quotient.mk x) = DirectSum.lof R _ _ i 1 := by
        intro i
        let fi := f.symm.toLinearMap.comp (DirectSum.lof _ _ _ i)
        obtain ⟨x, h0, h1⟩ := exists_smul_eq_zero_and_mk_eq hp hN hj fi; refine ⟨x, h0, ?_⟩; rw [h1]
        simp only [fi, LinearMap.coe_comp, f.symm.coe_toLinearMap, f.apply_symm_apply,
          Function.comp_apply]
      /-
        case h.succ.intro.refine_3.intro.intro
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this✝ : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        k : Fin d → Nat
        f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
        this : ∀ (i : Fin d), Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p (k i)) …
        ⊢ Exists fun k => Nonempty (LinearEquiv (RingHom.id R) N (DirectSum (Fin (HAdd …
      -/
      refine ⟨?_, ⟨?_⟩⟩
        /-
          case h.succ.intro.refine_3.intro.intro.refine_1
          R : Type u
          inst✝⁴ : CommRing R
          inst✝³ : IsPrincipalIdealRing R
          inst✝² : IsDomain R
          p : R
          hp : Irreducible p
          d : Nat
          N : Type (max u v)
          inst✝¹ : AddCommGroup N
          inst✝ : Module R N
          hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
          s : Fin (HAdd.hAdd d 1) → N
          hs : Eq (Submodule.span R (Set.range s)) Top.top
          this✝ : (x : N) → Decidable (Eq x 0)
          j : Fin d.succ
          hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
          s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
          k : Fin d → Nat
          f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
          this : ∀ (i : Fin d), Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p (k i)) …
          ⊢ Fin (HAdd.hAdd d 1) → Nat
        -/
      · exact fun a => (fun i => (Option.rec (pOrder hN (s j)) k i : ℕ)) (finSuccEquiv d a)
        /-
          🎉 no goals
        -/
      · refine (((lequivProdOfRightSplitExact
          (g := (f.trans ULift.moduleEquiv.{u, u, v}.symm).toLinearMap.comp <| mkQ _)
          (f := (DirectSum.toModule _ _ _ fun i => (liftQSpanSingleton (p ^ k i)
              (LinearMap.toSpanSingleton _ _ _) (this i).choose_spec.left : R ⧸ _ →ₗ[R] _)).comp
            ULift.moduleEquiv.toLinearMap) (R ∙ s j).injective_subtype ?_ ?_).symm.trans
          (((quotTorsionOfEquivSpanSingleton R N (s j)).symm.trans
          (quotEquivOfEq (torsionOf R N (s j)) _
          (Ideal.torsionOf_eq_span_pow_pOrder hp hN (s j)))).prod
          (ULift.moduleEquiv))).trans
          (@DirectSum.lequivProdDirectSum R _ _
          (fun i => R ⧸ R ∙ p ^ @Option.rec _ (fun _ => ℕ) (pOrder hN <| s j) k i) _ _).symm).trans
          (DirectSum.lequivCongrLeft R (finSuccEquiv d).symm)
          /-
            case h.succ.intro.refine_3.intro.intro.refine_2.refine_1
            R : Type u
            inst✝⁴ : CommRing R
            inst✝³ : IsPrincipalIdealRing R
            inst✝² : IsDomain R
            p : R
            hp : Irreducible p
            d : Nat
            N : Type (max u v)
            inst✝¹ : AddCommGroup N
            inst✝ : Module R N
            hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
            s : Fin (HAdd.hAdd d 1) → N
            hs : Eq (Submodule.span R (Set.range s)) Top.top
            this✝ : (x : N) → Decidable (Eq x 0)
            j : Fin d.succ
            hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
            s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
            k : Fin d → Nat
            f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
            this : ∀ (i : Fin d), Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p (k i)) …
            ⊢ Eq (LinearMap.range (Submodule.span R (Singleton.singleton (s j))).subtype)  …
          -/
        · rw [range_subtype, LinearEquiv.ker_comp, ker_mkQ]
          /-
            🎉 no goals
          -/
        · rw [← f.comp_coe, LinearMap.comp_assoc, LinearMap.comp_assoc,
            LinearEquiv.toLinearMap_symm_comp_eq, LinearMap.comp_id, ← LinearMap.comp_assoc,
            ← LinearMap.comp_assoc]
          suffices (f.toLinearMap.comp (R ∙ s j).mkQ).comp _ = LinearMap.id by
            rw [this, LinearMap.id_comp]
          /-
            case h.succ.intro.refine_3.intro.intro.refine_2.refine_2
            R : Type u
            inst✝⁴ : CommRing R
            inst✝³ : IsPrincipalIdealRing R
            inst✝² : IsDomain R
            p : R
            hp : Irreducible p
            d : Nat
            N : Type (max u v)
            inst✝¹ : AddCommGroup N
            inst✝ : Module R N
            hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
            s : Fin (HAdd.hAdd d 1) → N
            hs : Eq (Submodule.span R (Set.range s)) Top.top
            this✝ : (x : N) → Decidable (Eq x 0)
            j : Fin d.succ
            hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
            s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
            k : Fin d → Nat
            f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
            this : ∀ (i : Fin d), Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p (k i)) …
            ⊢ Eq (((↑f).comp (Submodule.span R (Singleton.singleton (s j))).mkQ).comp (Dir …
          -/
          ext i : 3
          /-
            case h.succ.intro.refine_3.intro.intro.refine_2.refine_2.H.h.h
            R : Type u
            inst✝⁴ : CommRing R
            inst✝³ : IsPrincipalIdealRing R
            inst✝² : IsDomain R
            p : R
            hp : Irreducible p
            d : Nat
            N : Type (max u v)
            inst✝¹ : AddCommGroup N
            inst✝ : Module R N
            hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
            s : Fin (HAdd.hAdd d 1) → N
            hs : Eq (Submodule.span R (Set.range s)) Top.top
            this✝ : (x : N) → Decidable (Eq x 0)
            j : Fin d.succ
            hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
            s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
            k : Fin d → Nat
            f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.span R (Sing …
            this : ∀ (i : Fin d), Exists fun x => And (Eq (HSMul.hSMul (HPow.hPow p (k i)) …
            i : Fin d
            ⊢ Eq ((((((↑f).comp (Submodule.span R (Singleton.singleton (s j))).mkQ).comp ( …
          -/
          simp only [LinearMap.coe_comp, Function.comp_apply, mkQ_apply]
          rw [LinearEquiv.coe_toLinearMap, LinearMap.id_apply, DirectSum.toModule_lof,
            liftQSpanSingleton_apply, LinearMap.toSpanSingleton_one, Ideal.Quotient.mk_eq_mk,
            map_one (Ideal.Quotient.mk _), (this i).choose_spec.right]
    · exact (mk_surjective _).forall.mpr fun x =>
        ⟨(@hN x).choose, by rw [← Quotient.mk_smul, (@hN x).choose_spec, Quotient.mk_zero]⟩
      /-
        case h.succ.intro.refine_2
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        ⊢ Eq (Submodule.span R (Set.range s')) Top.top
      -/
    · have hs' := congr_arg (Submodule.map <| mkQ <| R ∙ s j) hs
      /-
        case h.succ.intro.refine_2
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        hs' : Eq (Submodule.map (Submodule.span R (Singleton.singleton (s j))).mkQ (Su …
        ⊢ Eq (Submodule.span R (Set.range s')) Top.top
      -/
      rw [Submodule.map_span, Submodule.map_top, range_mkQ] at hs'; simp only [mkQ_apply] at hs'
      /-
        case h.succ.intro.refine_2
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        hs' : Eq (Submodule.span R (Set.image (fun a => Submodule.Quotient.mk a) (Set. …
        ⊢ Eq (Submodule.span R (Set.range s')) Top.top
      -/
      simp only [s']; rw [← Function.comp_assoc, Set.range_comp (_ ∘ s), Fin.range_succAbove]
      rw [← Set.range_comp, ← Set.insert_image_compl_eq_range _ j, Function.comp_apply,
        (Quotient.mk_eq_zero _).mpr (Submodule.mem_span_singleton_self _), span_insert_zero] at hs'
      /-
        case h.succ.intro.refine_2
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : IsDomain R
        p : R
        hp : Irreducible p
        d : Nat
        IH : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N], Mod …
        N : Type (max u v)
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        hN : Module.IsTorsion' N (Subtype fun x => Membership.mem (Submonoid.powers p) …
        s : Fin (HAdd.hAdd d 1) → N
        hs : Eq (Submodule.span R (Set.range s)) Top.top
        this : (x : N) → Decidable (Eq x 0)
        j : Fin d.succ
        hj : Module.IsTorsionBy R N (HPow.hPow p (Submodule.pOrder hN (s j)))
        s' : Fin d → HasQuotient.Quotient N (Submodule.span R (Singleton.singleton (s  …
        hs' : Eq (Submodule.span R (Set.image (Function.comp (fun a => Submodule.Quoti …
        ⊢ Eq (Submodule.span R (Set.image (Function.comp Submodule.Quotient.mk s) (Has …
      -/
      exact hs'
      /-
        🎉 no goals
      -/


/-- A finitely generated torsion module over a PID is isomorphic to a direct sum of some
  `R ⧸ R ∙ (p i ^ e i)` where the `p i ^ e i` are prime powers. -/
theorem equiv_directSum_of_isTorsion [h' : Module.Finite R N] (hN : Module.IsTorsion R N) :
    ∃ (ι : Type u) (_ : Fintype ι) (p : ι → R) (_ : ∀ i, Irreducible <| p i) (e : ι → ℕ),
      Nonempty <| N ≃ₗ[R] ⨁ i : ι, R ⧸ R ∙ p i ^ e i := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    hN : Module.IsTorsion R N
    ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
  -/
  obtain ⟨I, fI, _, p, hp, e, h⟩ := Submodule.exists_isInternal_prime_power_torsion_of_pid hN
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    hN : Module.IsTorsion R N
    I : Type u
    fI : Fintype I
    w✝ : DecidableEq I
    p : I → R
    hp : ∀ (i : I), Irreducible (p i)
    e : I → Nat
    h : DirectSum.IsInternal fun i => Submodule.torsionBy R N (HPow.hPow (p i) (e  …
    ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
  -/
  haveI := fI
  have :
    ∀ i,
      ∃ (d : ℕ) (k : Fin d → ℕ),
        Nonempty <| torsionBy R N (p i ^ e i) ≃ₗ[R] ⨁ j, R ⧸ R ∙ p i ^ k j := by
    haveI := fun i => isNoetherian_submodule' (torsionBy R N <| p i ^ e i)
    exact fun i =>
      torsion_by_prime_power_decomposition.{u, v} (hp i)
        ((isTorsion'_powers_iff <| p i).mpr fun x => ⟨e i, smul_torsionBy _ _⟩)
  classical
  refine
    ⟨Σ i, Fin (this i).choose, inferInstance, fun ⟨i, _⟩ => p i, fun ⟨i, _⟩ => hp i, fun ⟨i, j⟩ =>
      (this i).choose_spec.choose j,
      ⟨(LinearEquiv.ofBijective (DirectSum.coeLinearMap _) h).symm.trans <|
          (DFinsupp.mapRange.linearEquiv fun i => (this i).choose_spec.choose_spec.some).trans <|
            (DirectSum.sigmaLcurryEquiv R).symm.trans
              (DFinsupp.mapRange.linearEquiv fun i => quotEquivOfEq _ _ ?_)⟩⟩
  cases' i with i j
  simp only


/-- **Structure theorem of finitely generated modules over a PID** : A finitely generated
  module over a PID is isomorphic to the product of a free module and a direct sum of some
  `R ⧸ R ∙ (p i ^ e i)` where the `p i ^ e i` are prime powers. -/
theorem equiv_free_prod_directSum [h' : Module.Finite R N] :
    ∃ (n : ℕ) (ι : Type u) (_ : Fintype ι) (p : ι → R) (_ : ∀ i, Irreducible <| p i) (e : ι → ℕ),
      Nonempty <| N ≃ₗ[R] (Fin n →₀ R) × ⨁ i : ι, R ⧸ R ∙ p i ^ e i := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  haveI := isNoetherian_submodule' (torsion R N)
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    this : IsNoetherian R (Subtype fun x => Membership.mem (Submodule.torsion R N) …
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  haveI := Module.Finite.of_surjective _ (torsion R N).mkQ_surjective
  obtain ⟨I, fI, p, hp, e, ⟨h⟩⟩ :=
    equiv_directSum_of_isTorsion.{u, v} (@torsion_isTorsion R N _ _ _)
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    this✝ : IsNoetherian R (Subtype fun x => Membership.mem (Submodule.torsion R N …
    this : Module.Finite R (HasQuotient.Quotient N (Submodule.torsion R N))
    I : Type u
    fI : Fintype I
    p : I → R
    hp : ∀ (i : I), Irreducible (p i)
    e : I → Nat
    h : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.tor …
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  obtain ⟨n, ⟨g⟩⟩ := @Module.basisOfFiniteTypeTorsionFree' R _ (N ⧸ torsion R N) _ _ _ _ _ _
  /-
    case intro.intro.intro.intro.intro.intro.mk.ofRepr
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    this✝ : IsNoetherian R (Subtype fun x => Membership.mem (Submodule.torsion R N …
    this : Module.Finite R (HasQuotient.Quotient N (Submodule.torsion R N))
    I : Type u
    fI : Fintype I
    p : I → R
    hp : ∀ (i : I), Irreducible (p i)
    e : I → Nat
    h : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.tor …
    n : Nat
    g : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.torsion R N) …
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  haveI : Module.Projective R (N ⧸ torsion R N) := Module.Projective.of_basis ⟨g⟩
  /-
    case intro.intro.intro.intro.intro.intro.mk.ofRepr
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    this✝¹ : IsNoetherian R (Subtype fun x => Membership.mem (Submodule.torsion R  …
    this✝ : Module.Finite R (HasQuotient.Quotient N (Submodule.torsion R N))
    I : Type u
    fI : Fintype I
    p : I → R
    hp : ∀ (i : I), Irreducible (p i)
    e : I → Nat
    h : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.tor …
    n : Nat
    g : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.torsion R N) …
    this : Module.Projective R (HasQuotient.Quotient N (Submodule.torsion R N))
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  obtain ⟨f, hf⟩ := Module.projective_lifting_property _ LinearMap.id (torsion R N).mkQ_surjective
  refine
    ⟨n, I, fI, p, hp, e,
      ⟨(lequivProdOfRightSplitExact (torsion R N).injective_subtype ?_ hf).symm.trans <|
          (h.prod g).trans <| LinearEquiv.prodComm.{u, u} R _ (Fin n →₀ R) ⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.mk.ofRepr.intro
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsPrincipalIdealRing R
    N : Type (max u v)
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsDomain R
    h' : Module.Finite R N
    this✝¹ : IsNoetherian R (Subtype fun x => Membership.mem (Submodule.torsion R  …
    this✝ : Module.Finite R (HasQuotient.Quotient N (Submodule.torsion R N))
    I : Type u
    fI : Fintype I
    p : I → R
    hp : ∀ (i : I), Irreducible (p i)
    e : I → Nat
    h : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.tor …
    n : Nat
    g : LinearEquiv (RingHom.id R) (HasQuotient.Quotient N (Submodule.torsion R N) …
    this : Module.Projective R (HasQuotient.Quotient N (Submodule.torsion R N))
    f : LinearMap (RingHom.id R) (HasQuotient.Quotient N (Submodule.torsion R N)) N
    hf : Eq ((Submodule.torsion R N).mkQ.comp f) LinearMap.id
    ⊢ Eq (LinearMap.range (Submodule.torsion R N).subtype) (LinearMap.ker (Submodu …
  -/
  rw [range_subtype, ker_mkQ]
  /-
    🎉 no goals
  -/


