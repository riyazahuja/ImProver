/-- An integral domain is a *discrete valuation ring* (DVR) if it's a local PID which
  is not a field. -/
class IsDiscreteValuationRing (R : Type u) [CommRing R] [IsDomain R]
    extends IsPrincipalIdealRing R, IsLocalRing R : Prop where
  not_a_field' : maximalIdeal R ≠ ⊥


theorem not_a_field : maximalIdeal R ≠ ⊥ :=
  not_a_field'


/-- A discrete valuation ring `R` is not a field. -/
theorem not_isField : ¬IsField R :=
  IsLocalRing.isField_iff_maximalIdeal_eq.not.mpr (not_a_field R)


theorem irreducible_of_span_eq_maximalIdeal {R : Type*} [CommRing R] [IsLocalRing R] [IsDomain R]
    (ϖ : R) (hϖ : ϖ ≠ 0) (h : maximalIdeal R = Ideal.span {ϖ}) : Irreducible ϖ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    ⊢ Irreducible ϖ
  -/
  have h2 : ¬IsUnit ϖ := show ϖ ∈ maximalIdeal R from h.symm ▸ Submodule.mem_span_singleton_self ϖ
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    ⊢ Irreducible ϖ
  -/
  refine ⟨h2, ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    ⊢ ∀ (a b : R), Eq ϖ (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
  -/
  intro a b hab
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  by_contra! h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h✝ : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul a b)
    h : And (Not (IsUnit a)) (Not (IsUnit b))
    ⊢ False
  -/
  obtain ⟨ha : a ∈ maximalIdeal R, hb : b ∈ maximalIdeal R⟩ := h
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul a b)
    ha : Membership.mem (IsLocalRing.maximalIdeal R) a
    hb : Membership.mem (IsLocalRing.maximalIdeal R) b
    ⊢ False
  -/
  rw [h, mem_span_singleton'] at ha hb
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul a b)
    ha : Exists fun a_1 => Eq (HMul.hMul a_1 ϖ) a
    hb : Exists fun a => Eq (HMul.hMul a ϖ) b
    ⊢ False
  -/
  rcases ha with ⟨a, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    b : R
    hb : Exists fun a => Eq (HMul.hMul a ϖ) b
    a : R
    hab : Eq ϖ (HMul.hMul (HMul.hMul a ϖ) b)
    ⊢ False
  -/
  rcases hb with ⟨b, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul (HMul.hMul a ϖ) (HMul.hMul b ϖ))
    ⊢ False
  -/
  rw [show a * ϖ * (b * ϖ) = ϖ * (ϖ * (a * b)) by ring] at hab
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul ϖ (HMul.hMul ϖ (HMul.hMul a b)))
    ⊢ False
  -/
  apply hϖ
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul ϖ (HMul.hMul ϖ (HMul.hMul a b)))
    ⊢ Eq ϖ 0
  -/
  apply eq_zero_of_mul_eq_self_right _ hab.symm
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsDomain R
    ϖ : R
    hϖ : Ne ϖ 0
    h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
    h2 : Not (IsUnit ϖ)
    a b : R
    hab : Eq ϖ (HMul.hMul ϖ (HMul.hMul ϖ (HMul.hMul a b)))
    ⊢ Ne (HMul.hMul ϖ (HMul.hMul a b)) 1
  -/
  exact fun hh => h2 (isUnit_of_dvd_one ⟨_, hh.symm⟩)
  /-
    🎉 no goals
  -/


/-- An element of a DVR is irreducible iff it is a uniformizer, that is, generates the
  maximal ideal of `R`. -/
theorem irreducible_iff_uniformizer (ϖ : R) : Irreducible ϖ ↔ maximalIdeal R = Ideal.span {ϖ} :=
  ⟨fun hϖ => (eq_maximalIdeal (isMaximal_of_irreducible hϖ)).symm,
    fun h => irreducible_of_span_eq_maximalIdeal ϖ
                                    /-
                                      R : Type u
                                      inst✝² : CommRing R
                                      inst✝¹ : IsDomain R
                                      inst✝ : IsDiscreteValuationRing R
                                      ϖ : R
                                      h : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton ϖ))
                                      e : Eq ϖ 0
                                      ⊢ Eq (IsLocalRing.maximalIdeal R) Bot.bot
                                    -/
      (fun e => not_a_field R <| by rwa [h, span_singleton_eq_bot]) h⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem _root_.Irreducible.maximalIdeal_eq {ϖ : R} (h : Irreducible ϖ) :
    maximalIdeal R = Ideal.span {ϖ} :=
  (irreducible_iff_uniformizer _).mp h


/-- Uniformizers exist in a DVR. -/
theorem exists_irreducible : ∃ ϖ : R, Irreducible ϖ := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ⊢ Exists fun ϖ => Irreducible ϖ
  -/
  simp_rw [irreducible_iff_uniformizer]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ⊢ Exists fun ϖ => Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singl …
  -/
  exact (IsPrincipalIdealRing.principal <| maximalIdeal R).principal
  /-
    🎉 no goals
  -/


/-- Uniformizers exist in a DVR. -/
theorem exists_prime : ∃ ϖ : R, Prime ϖ :=
  (exists_irreducible R).imp fun _ => irreducible_iff_prime.1


/-- An integral domain is a DVR iff it's a PID with a unique non-zero prime ideal. -/
theorem iff_pid_with_one_nonzero_prime (R : Type u) [CommRing R] [IsDomain R] :
    IsDiscreteValuationRing R ↔ IsPrincipalIdealRing R ∧ ∃! P : Ideal R, P ≠ ⊥ ∧ IsPrime P := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Iff (IsDiscreteValuationRing R) (And (IsPrincipalIdealRing R) (ExistsUnique  …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ IsDiscreteValuationRing R → And (IsPrincipalIdealRing R) (ExistsUnique fun P …
    -/
  · intro RDVR
    /-
      case mp
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RDVR : IsDiscreteValuationRing R
      ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
    -/
    rcases id RDVR with ⟨Rlocal⟩
    /-
      case mp.mk
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RDVR : IsDiscreteValuationRing R
      toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
      toIsLocalRing✝ : IsLocalRing R
      Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
      ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
    -/
    constructor
      /-
        case mp.mk.left
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        ⊢ IsPrincipalIdealRing R
      -/
    · assumption
      /-
        🎉 no goals
      -/
    /-
      case mp.mk.right
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RDVR : IsDiscreteValuationRing R
      toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
      toIsLocalRing✝ : IsLocalRing R
      Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
      ⊢ ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
    -/
    use IsLocalRing.maximalIdeal R
    /-
      case h
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RDVR : IsDiscreteValuationRing R
      toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
      toIsLocalRing✝ : IsLocalRing R
      Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
      ⊢ And ((fun P => And (Ne P Bot.bot) P.IsPrime) (IsLocalRing.maximalIdeal R)) ( …
    -/
    constructor
      /-
        case h.left
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        ⊢ (fun P => And (Ne P Bot.bot) P.IsPrime) (IsLocalRing.maximalIdeal R)
      -/
    · exact ⟨Rlocal, inferInstance⟩
      /-
        🎉 no goals
      -/
      /-
        case h.right
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        ⊢ ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y (IsLocalRi …
      -/
    · rintro Q ⟨hQ1, hQ2⟩
      /-
        case h.right.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        Q : Ideal R
        hQ1 : Ne Q Bot.bot
        hQ2 : Q.IsPrime
        ⊢ Eq Q (IsLocalRing.maximalIdeal R)
      -/
      obtain ⟨q, rfl⟩ := (IsPrincipalIdealRing.principal Q).1
      have hq : q ≠ 0 := by
        rintro rfl
        apply hQ1
        simp
      /-
        case h.right.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        q : R
        hQ1 : Ne (Submodule.span R (Singleton.singleton q)) Bot.bot
        hQ2 : Ideal.IsPrime (Submodule.span R (Singleton.singleton q))
        hq : Ne q 0
        ⊢ Eq (Submodule.span R (Singleton.singleton q)) (IsLocalRing.maximalIdeal R)
      -/
      erw [span_singleton_prime hq] at hQ2
      /-
        case h.right.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        q : R
        hQ1 : Ne (Submodule.span R (Singleton.singleton q)) Bot.bot
        hQ2 : Prime q
        hq : Ne q 0
        ⊢ Eq (Submodule.span R (Singleton.singleton q)) (IsLocalRing.maximalIdeal R)
      -/
      replace hQ2 := hQ2.irreducible
      /-
        case h.right.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        q : R
        hQ1 : Ne (Submodule.span R (Singleton.singleton q)) Bot.bot
        hq : Ne q 0
        hQ2 : Irreducible q
        ⊢ Eq (Submodule.span R (Singleton.singleton q)) (IsLocalRing.maximalIdeal R)
      -/
      rw [irreducible_iff_uniformizer] at hQ2
      /-
        case h.right.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        RDVR : IsDiscreteValuationRing R
        toIsPrincipalIdealRing✝ : IsPrincipalIdealRing R
        toIsLocalRing✝ : IsLocalRing R
        Rlocal : Ne (IsLocalRing.maximalIdeal R) Bot.bot
        q : R
        hQ1 : Ne (Submodule.span R (Singleton.singleton q)) Bot.bot
        hq : Ne q 0
        hQ2 : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton q))
        ⊢ Eq (Submodule.span R (Singleton.singleton q)) (IsLocalRing.maximalIdeal R)
      -/
      exact hQ2.symm
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
    -/
  · rintro ⟨RPID, Punique⟩
    /-
      case mpr.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      Punique : ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
      ⊢ IsDiscreteValuationRing R
    -/
    haveI : IsLocalRing R := IsLocalRing.of_unique_nonzero_prime Punique
    /-
      case mpr.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      Punique : ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
      this : IsLocalRing R
      ⊢ IsDiscreteValuationRing R
    -/
    refine { not_a_field' := ?_ }
    /-
      case mpr.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      Punique : ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
      this : IsLocalRing R
      ⊢ Ne (IsLocalRing.maximalIdeal R) Bot.bot
    -/
    rcases Punique with ⟨P, ⟨hP1, hP2⟩, _⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      this : IsLocalRing R
      P : Ideal R
      right✝ : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
      hP1 : Ne P Bot.bot
      hP2 : P.IsPrime
      ⊢ Ne (IsLocalRing.maximalIdeal R) Bot.bot
    -/
    have hPM : P ≤ maximalIdeal R := le_maximalIdeal hP2.1
    /-
      case mpr.intro.intro.intro.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      this : IsLocalRing R
      P : Ideal R
      right✝ : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
      hP1 : Ne P Bot.bot
      hP2 : P.IsPrime
      hPM : LE.le P (IsLocalRing.maximalIdeal R)
      ⊢ Ne (IsLocalRing.maximalIdeal R) Bot.bot
    -/
    intro h
    /-
      case mpr.intro.intro.intro.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      this : IsLocalRing R
      P : Ideal R
      right✝ : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
      hP1 : Ne P Bot.bot
      hP2 : P.IsPrime
      hPM : LE.le P (IsLocalRing.maximalIdeal R)
      h : Eq (IsLocalRing.maximalIdeal R) Bot.bot
      ⊢ False
    -/
    rw [h, le_bot_iff] at hPM
    /-
      case mpr.intro.intro.intro.intro
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      RPID : IsPrincipalIdealRing R
      this : IsLocalRing R
      P : Ideal R
      right✝ : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
      hP1 : Ne P Bot.bot
      hP2 : P.IsPrime
      hPM : Eq P Bot.bot
      h : Eq (IsLocalRing.maximalIdeal R) Bot.bot
      ⊢ False
    -/
    exact hP1 hPM
    /-
      🎉 no goals
    -/


theorem associated_of_irreducible {a b : R} (ha : Irreducible a) (hb : Irreducible b) :
    Associated a b := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    a b : R
    ha : Irreducible a
    hb : Irreducible b
    ⊢ Associated a b
  -/
  rw [irreducible_iff_uniformizer] at ha hb
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    a b : R
    ha : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton a))
    hb : Eq (IsLocalRing.maximalIdeal R) (Ideal.span (Singleton.singleton b))
    ⊢ Associated a b
  -/
  rw [← span_singleton_eq_span_singleton, ← ha, hb]
  /-
    🎉 no goals
  -/


/-- Alternative characterisation of discrete valuation rings. -/
def HasUnitMulPowIrreducibleFactorization [CommRing R] : Prop :=
  ∃ p : R, Irreducible p ∧ ∀ {x : R}, x ≠ 0 → ∃ n : ℕ, Associated (p ^ n) x


theorem unique_irreducible (hR : HasUnitMulPowIrreducibleFactorization R)
    ⦃p q : R⦄ (hp : Irreducible p) (hq : Irreducible q) :
    Associated p q := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    ⊢ Associated p q
  -/
  rcases hR with ⟨ϖ, hϖ, hR⟩
  suffices ∀ {p : R} (_ : Irreducible p), Associated p ϖ by
    apply Associated.trans (this hp) (this hq).symm
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    ϖ : R
    hϖ : Irreducible ϖ
    hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
    ⊢ ∀ {p : R}, Irreducible p → Associated p ϖ
  -/
  clear hp hq p q
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    ϖ : R
    hϖ : Irreducible ϖ
    hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
    ⊢ ∀ {p : R}, Irreducible p → Associated p ϖ
  -/
  intro p hp
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    ϖ : R
    hϖ : Irreducible ϖ
    hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
    p : R
    hp : Irreducible p
    ⊢ Associated p ϖ
  -/
  obtain ⟨n, hn⟩ := hR hp.ne_zero
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    ϖ : R
    hϖ : Irreducible ϖ
    hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
    p : R
    hp : Irreducible p
    n : Nat
    hn : Associated (HPow.hPow ϖ n) p
    ⊢ Associated p ϖ
  -/
  have : Irreducible (ϖ ^ n) := hn.symm.irreducible hp
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    ϖ : R
    hϖ : Irreducible ϖ
    hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
    p : R
    hp : Irreducible p
    n : Nat
    hn : Associated (HPow.hPow ϖ n) p
    this : Irreducible (HPow.hPow ϖ n)
    ⊢ Associated p ϖ
  -/
  rcases lt_trichotomy n 1 with (H | rfl | H)
  · obtain rfl : n = 0 := by
      clear hn this
      revert H n
      decide
    /-
      case intro.intro.intro.inl
      R : Type u_1
      inst✝ : CommRing R
      ϖ : R
      hϖ : Irreducible ϖ
      hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
      p : R
      hp : Irreducible p
      hn : Associated (HPow.hPow ϖ 0) p
      this : Irreducible (HPow.hPow ϖ 0)
      H : LT.lt 0 1
      ⊢ Associated p ϖ
    -/
    simp [not_irreducible_one, pow_zero] at this
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.inr.inl
      R : Type u_1
      inst✝ : CommRing R
      ϖ : R
      hϖ : Irreducible ϖ
      hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
      p : R
      hp : Irreducible p
      hn : Associated (HPow.hPow ϖ 1) p
      this : Irreducible (HPow.hPow ϖ 1)
      ⊢ Associated p ϖ
    -/
  · simpa only [pow_one] using hn.symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.inr.inr
      R : Type u_1
      inst✝ : CommRing R
      ϖ : R
      hϖ : Irreducible ϖ
      hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
      p : R
      hp : Irreducible p
      n : Nat
      hn : Associated (HPow.hPow ϖ n) p
      this : Irreducible (HPow.hPow ϖ n)
      H : LT.lt 1 n
      ⊢ Associated p ϖ
    -/
  · obtain ⟨n, rfl⟩ : ∃ k, n = 1 + k + 1 := Nat.exists_eq_add_of_lt H
    /-
      case intro.intro.intro.inr.inr.intro
      R : Type u_1
      inst✝ : CommRing R
      ϖ : R
      hϖ : Irreducible ϖ
      hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
      p : R
      hp : Irreducible p
      n : Nat
      hn : Associated (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1)) p
      this : Irreducible (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1))
      H : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 n) 1)
      ⊢ Associated p ϖ
    -/
    rw [pow_succ'] at this
    /-
      case intro.intro.intro.inr.inr.intro
      R : Type u_1
      inst✝ : CommRing R
      ϖ : R
      hϖ : Irreducible ϖ
      hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
      p : R
      hp : Irreducible p
      n : Nat
      hn : Associated (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1)) p
      this : Irreducible (HMul.hMul ϖ (HPow.hPow ϖ (HAdd.hAdd 1 n)))
      H : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 n) 1)
      ⊢ Associated p ϖ
    -/
    rcases this.isUnit_or_isUnit rfl with (H0 | H0)
      /-
        case intro.intro.intro.inr.inr.intro.inl
        R : Type u_1
        inst✝ : CommRing R
        ϖ : R
        hϖ : Irreducible ϖ
        hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
        p : R
        hp : Irreducible p
        n : Nat
        hn : Associated (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1)) p
        this : Irreducible (HMul.hMul ϖ (HPow.hPow ϖ (HAdd.hAdd 1 n)))
        H : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 n) 1)
        H0 : IsUnit ϖ
        ⊢ Associated p ϖ
      -/
    · exact (hϖ.not_unit H0).elim
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inr.inr.intro.inr
        R : Type u_1
        inst✝ : CommRing R
        ϖ : R
        hϖ : Irreducible ϖ
        hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
        p : R
        hp : Irreducible p
        n : Nat
        hn : Associated (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1)) p
        this : Irreducible (HMul.hMul ϖ (HPow.hPow ϖ (HAdd.hAdd 1 n)))
        H : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 n) 1)
        H0 : IsUnit (HPow.hPow ϖ (HAdd.hAdd 1 n))
        ⊢ Associated p ϖ
      -/
    · rw [add_comm, pow_succ'] at H0
      /-
        case intro.intro.intro.inr.inr.intro.inr
        R : Type u_1
        inst✝ : CommRing R
        ϖ : R
        hϖ : Irreducible ϖ
        hR : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow ϖ n) x
        p : R
        hp : Irreducible p
        n : Nat
        hn : Associated (HPow.hPow ϖ (HAdd.hAdd (HAdd.hAdd 1 n) 1)) p
        this : Irreducible (HMul.hMul ϖ (HPow.hPow ϖ (HAdd.hAdd 1 n)))
        H : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 n) 1)
        H0 : IsUnit (HMul.hMul ϖ (HPow.hPow ϖ n))
        ⊢ Associated p ϖ
      -/
      exact (hϖ.not_unit (isUnit_of_mul_isUnit_left H0)).elim
      /-
        🎉 no goals
      -/


/-- An integral domain in which there is an irreducible element `p`
such that every nonzero element is associated to a power of `p` is a unique factorization domain.
See `IsDiscreteValuationRing.ofHasUnitMulPowIrreducibleFactorization`. -/
theorem toUniqueFactorizationMonoid (hR : HasUnitMulPowIrreducibleFactorization R) :
    UniqueFactorizationMonoid R :=
  let p := Classical.choose hR
  let spec := Classical.choose_spec hR
  UniqueFactorizationMonoid.of_exists_prime_factors fun x hx => by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
      p : R := Classical.choose hR
      spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
      x : R
      hx : Ne x 0
      ⊢ Exists fun f => And (∀ (b : R), Membership.mem f b → Prime b) (Associated f. …
    -/
    use Multiset.replicate (Classical.choose (spec.2 hx)) p
    /-
      case h
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
      p : R := Classical.choose hR
      spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
      x : R
      hx : Ne x 0
      ⊢ And (∀ (b : R), Membership.mem (Multiset.replicate (Classical.choose ⋯) p) b …
    -/
    constructor
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        ⊢ ∀ (b : R), Membership.mem (Multiset.replicate (Classical.choose ⋯) p) b → Pr …
      -/
    · intro q hq
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        ⊢ Prime q
      -/
      have hpq := Multiset.eq_of_mem_replicate hq
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        ⊢ Prime q
      -/
      rw [hpq]
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        ⊢ Prime p
      -/
      refine ⟨spec.1.ne_zero, spec.1.not_unit, ?_⟩
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        ⊢ ∀ (a b : R), Dvd.dvd p (HMul.hMul a b) → Or (Dvd.dvd p a) (Dvd.dvd p b)
      -/
      intro a b h
      /-
        case h.left
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        a b : R
        h : Dvd.dvd p (HMul.hMul a b)
        ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
      -/
      by_cases ha : a = 0
        /-
          case pos
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
          p : R := Classical.choose hR
          spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
          x : R
          hx : Ne x 0
          q : R
          hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
          hpq : Eq q p
          a b : R
          h : Dvd.dvd p (HMul.hMul a b)
          ha : Eq a 0
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
      · rw [ha]
        /-
          case pos
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
          p : R := Classical.choose hR
          spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
          x : R
          hx : Ne x 0
          q : R
          hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
          hpq : Eq q p
          a b : R
          h : Dvd.dvd p (HMul.hMul a b)
          ha : Eq a 0
          ⊢ Or (Dvd.dvd p 0) (Dvd.dvd p b)
        -/
        simp only [true_or, dvd_zero]
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        a b : R
        h : Dvd.dvd p (HMul.hMul a b)
        ha : Not (Eq a 0)
        ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
      -/
      obtain ⟨m, u, rfl⟩ := spec.2 ha
      /-
        case neg.intro.intro
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        m : Nat
        u : Units R
        h : Dvd.dvd p (HMul.hMul (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
        ⊢ Or (Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u)) (Dvd.dvd p …
      -/
      rw [mul_assoc, mul_left_comm, Units.dvd_mul_left] at h
      /-
        case neg.intro.intro
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        m : Nat
        u : Units R
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
        ⊢ Or (Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u)) (Dvd.dvd p …
      -/
      rw [Units.dvd_mul_right]
      /-
        case neg.intro.intro
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        m : Nat
        u : Units R
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
        ⊢ Or (Dvd.dvd p (HPow.hPow (Classical.choose hR) m)) (Dvd.dvd p b)
      -/
      by_cases hm : m = 0
        /-
          case pos
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
          p : R := Classical.choose hR
          spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
          x : R
          hx : Ne x 0
          q : R
          hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
          hpq : Eq q p
          b : R
          m : Nat
          u : Units R
          h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) b)
          ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
          hm : Eq m 0
          ⊢ Or (Dvd.dvd p (HPow.hPow (Classical.choose hR) m)) (Dvd.dvd p b)
        -/
      · simp only [hm, one_mul, pow_zero] at h ⊢
        /-
          case pos
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
          p : R := Classical.choose hR
          spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
          x : R
          hx : Ne x 0
          q : R
          hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
          hpq : Eq q p
          b : R
          m : Nat
          u : Units R
          ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
          hm : Eq m 0
          h : Dvd.dvd p b
          ⊢ Or (Dvd.dvd p 1) (Dvd.dvd p b)
        -/
        right
        /-
          case pos.h
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
          p : R := Classical.choose hR
          spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
          x : R
          hx : Ne x 0
          q : R
          hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
          hpq : Eq q p
          b : R
          m : Nat
          u : Units R
          ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
          hm : Eq m 0
          h : Dvd.dvd p b
          ⊢ Dvd.dvd p b
        -/
        exact h
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        m : Nat
        u : Units R
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
        hm : Not (Eq m 0)
        ⊢ Or (Dvd.dvd p (HPow.hPow (Classical.choose hR) m)) (Dvd.dvd p b)
      -/
      left
      /-
        case neg.h
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        m : Nat
        u : Units R
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m) ↑u) 0)
        hm : Not (Eq m 0)
        ⊢ Dvd.dvd p (HPow.hPow (Classical.choose hR) m)
      -/
      obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hm
      /-
        case neg.h.intro
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        u : Units R
        m : Nat
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m.succ) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m.succ) ↑u) 0)
        hm : Not (Eq m.succ 0)
        ⊢ Dvd.dvd p (HPow.hPow (Classical.choose hR) m.succ)
      -/
      rw [pow_succ']
      /-
        case neg.h.intro
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        q : R
        hq : Membership.mem (Multiset.replicate (Classical.choose ⋯) p) q
        hpq : Eq q p
        b : R
        u : Units R
        m : Nat
        h : Dvd.dvd p (HMul.hMul (HPow.hPow (Classical.choose hR) m.succ) b)
        ha : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose hR) m.succ) ↑u) 0)
        hm : Not (Eq m.succ 0)
        ⊢ Dvd.dvd p (HMul.hMul (Classical.choose hR) (HPow.hPow (Classical.choose hR)  …
      -/
      apply dvd_mul_of_dvd_left dvd_rfl _
      /-
        🎉 no goals
      -/
      /-
        case h.right
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        ⊢ Associated (Multiset.replicate (Classical.choose ⋯) p).prod x
      -/
    · rw [Multiset.prod_replicate]
      /-
        case h.right
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
        p : R := Classical.choose hR
        spec : And (Irreducible (Classical.choose hR)) (∀ {x : R}, Ne x 0 → Exists fun …
        x : R
        hx : Ne x 0
        ⊢ Associated (HPow.hPow p (Classical.choose ⋯)) x
      -/
      exact Classical.choose_spec (spec.2 hx)
      /-
        🎉 no goals
      -/


theorem of_ufd_of_unique_irreducible [UniqueFactorizationMonoid R] (h₁ : ∃ p : R, Irreducible p)
    (h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q) :
    HasUnitMulPowIrreducibleFactorization R := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₁ : Exists fun p => Irreducible p
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    ⊢ IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
  -/
  obtain ⟨p, hp⟩ := h₁
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    ⊢ IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
  -/
  refine ⟨p, hp, ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    ⊢ ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow p n) x
  -/
  intro x hx
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    ⊢ Exists fun n => Associated (HPow.hPow p n) x
  -/
  cases' WfDvdMonoid.exists_factors x hx with fx hfx
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    ⊢ Exists fun n => Associated (HPow.hPow p n) x
  -/
  refine ⟨Multiset.card fx, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    ⊢ Associated (HPow.hPow p fx.card) x
  -/
  have H := hfx.2
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Associated fx.prod x
    ⊢ Associated (HPow.hPow p fx.card) x
  -/
  rw [← Associates.mk_eq_mk_iff_associated] at H ⊢
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Associates.mk (HPow.hPow p fx.card)) (Associates.mk x)
  -/
  rw [← H, ← Associates.prod_mk, Associates.mk_pow, ← Multiset.prod_replicate]
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Multiset.replicate fx.card (Associates.mk p)).prod (Multiset.map Associa …
  -/
  congr 1
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Multiset.replicate fx.card (Associates.mk p)) (Multiset.map Associates.m …
  -/
  symm
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Multiset.map Associates.mk fx) (Multiset.replicate fx.card (Associates.m …
  -/
  rw [Multiset.eq_replicate]
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ And (Eq (Multiset.map Associates.mk fx).card fx.card) (∀ (b : Associates R), …
  -/
  simp only [true_and, and_imp, Multiset.card_map, eq_self_iff_true, Multiset.mem_map, exists_imp]
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ ∀ (b : Associates R) (x : R), Membership.mem fx x → Eq (Associates.mk x) b → …
  -/
  rintro _ q hq rfl
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    q : R
    hq : Membership.mem fx q
    ⊢ Eq (Associates.mk q) (Associates.mk p)
  -/
  rw [Associates.mk_eq_mk_iff_associated]
  /-
    case intro.intro.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    p : R
    hp : Irreducible p
    x : R
    hx : Ne x 0
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    q : R
    hq : Membership.mem fx q
    ⊢ Associated q p
  -/
  apply h₂ (hfx.1 _ hq) hp
  /-
    🎉 no goals
  -/


theorem aux_pid_of_ufd_of_unique_irreducible (R : Type u) [CommRing R] [IsDomain R]
    [UniqueFactorizationMonoid R] (h₁ : ∃ p : R, Irreducible p)
    (h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q) :
    IsPrincipalIdealRing R := by
  classical
  constructor
  intro I
  by_cases I0 : I = ⊥
  · rw [I0]
    use 0
    simp only [Set.singleton_zero, Submodule.span_zero]
  obtain ⟨x, hxI, hx0⟩ : ∃ x ∈ I, x ≠ (0 : R) := I.ne_bot_iff.mp I0
  obtain ⟨p, _, H⟩ := HasUnitMulPowIrreducibleFactorization.of_ufd_of_unique_irreducible h₁ h₂
  have ex : ∃ n : ℕ, p ^ n ∈ I := by
    obtain ⟨n, u, rfl⟩ := H hx0
    refine ⟨n, ?_⟩
    simpa only [Units.mul_inv_cancel_right] using I.mul_mem_right (↑u⁻¹) hxI
  constructor
  use p ^ Nat.find ex
  show I = Ideal.span _
  apply le_antisymm
  · intro r hr
    by_cases hr0 : r = 0
    · simp only [hr0, Submodule.zero_mem]
    obtain ⟨n, u, rfl⟩ := H hr0
    simp only [mem_span_singleton, Units.isUnit, IsUnit.dvd_mul_right]
    apply pow_dvd_pow
    apply Nat.find_min'
    simpa only [Units.mul_inv_cancel_right] using I.mul_mem_right (↑u⁻¹) hr
  · erw [Submodule.span_singleton_le_iff_mem]
    exact Nat.find_spec ex


/-- A unique factorization domain with at least one irreducible element
in which all irreducible elements are associated
is a discrete valuation ring.
-/
theorem of_ufd_of_unique_irreducible {R : Type u} [CommRing R] [IsDomain R]
    [UniqueFactorizationMonoid R] (h₁ : ∃ p : R, Irreducible p)
    (h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q) :
    IsDiscreteValuationRing R := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₁ : Exists fun p => Irreducible p
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    ⊢ IsDiscreteValuationRing R
  -/
  rw [iff_pid_with_one_nonzero_prime]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₁ : Exists fun p => Irreducible p
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
  -/
  haveI PID : IsPrincipalIdealRing R := aux_pid_of_ufd_of_unique_irreducible R h₁ h₂
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₁ : Exists fun p => Irreducible p
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    PID : IsPrincipalIdealRing R
    ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
  -/
  obtain ⟨p, hp⟩ := h₁
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : UniqueFactorizationMonoid R
    h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
    PID : IsPrincipalIdealRing R
    p : R
    hp : Irreducible p
    ⊢ And (IsPrincipalIdealRing R) (ExistsUnique fun P => And (Ne P Bot.bot) P.IsP …
  -/
  refine ⟨PID, ⟨Ideal.span {p}, ⟨?_, ?_⟩, ?_⟩⟩
    /-
      case intro.refine_1
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      ⊢ Ne (Ideal.span (Singleton.singleton p)) Bot.bot
    -/
  · rw [Submodule.ne_bot_iff]
    /-
      case intro.refine_1
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      ⊢ Exists fun x => And (Membership.mem (Ideal.span (Singleton.singleton p)) x)  …
    -/
    exact ⟨p, Ideal.mem_span_singleton.mpr (dvd_refl p), hp.ne_zero⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      ⊢ (Ideal.span (Singleton.singleton p)).IsPrime
    -/
  · rwa [Ideal.span_singleton_prime hp.ne_zero, ← UniqueFactorizationMonoid.irreducible_iff_prime]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      ⊢ ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y (Ideal.spa …
    -/
  · intro I
    /-
      case intro.refine_3
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      ⊢ (fun P => And (Ne P Bot.bot) P.IsPrime) I → Eq I (Ideal.span (Singleton.sing …
    -/
    rw [← Submodule.IsPrincipal.span_singleton_generator I]
    /-
      case intro.refine_3
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      ⊢ (fun P => And (Ne P Bot.bot) P.IsPrime) (Submodule.span R (Singleton.singlet …
    -/
    rintro ⟨I0, hI⟩
    /-
      case intro.refine_3.intro
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      I0 : Ne (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generato …
      hI : Ideal.IsPrime (Submodule.span R (Singleton.singleton (Submodule.IsPrincip …
      ⊢ Eq (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generator I …
    -/
    apply span_singleton_eq_span_singleton.mpr
    /-
      case intro.refine_3.intro
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      I0 : Ne (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generato …
      hI : Ideal.IsPrime (Submodule.span R (Singleton.singleton (Submodule.IsPrincip …
      ⊢ Associated (Submodule.IsPrincipal.generator I) p
    -/
    apply h₂ _ hp
    /-
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      I0 : Ne (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generato …
      hI : Ideal.IsPrime (Submodule.span R (Singleton.singleton (Submodule.IsPrincip …
      ⊢ Irreducible (Submodule.IsPrincipal.generator I)
    -/
    erw [Ne, span_singleton_eq_bot] at I0
    /-
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : UniqueFactorizationMonoid R
      h₂ : ∀ ⦃p q : R⦄, Irreducible p → Irreducible q → Associated p q
      PID : IsPrincipalIdealRing R
      p : R
      hp : Irreducible p
      I : Ideal R
      I0 : Not (Eq (Submodule.IsPrincipal.generator I) 0)
      hI : Ideal.IsPrime (Submodule.span R (Singleton.singleton (Submodule.IsPrincip …
      ⊢ Irreducible (Submodule.IsPrincipal.generator I)
    -/
    rwa [UniqueFactorizationMonoid.irreducible_iff_prime, ← Ideal.span_singleton_prime I0]
    /-
      🎉 no goals
    -/


/-- An integral domain in which there is an irreducible element `p`
such that every nonzero element is associated to a power of `p`
is a discrete valuation ring.
-/
theorem ofHasUnitMulPowIrreducibleFactorization {R : Type u} [CommRing R] [IsDomain R]
    (hR : HasUnitMulPowIrreducibleFactorization R) : IsDiscreteValuationRing R := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
    ⊢ IsDiscreteValuationRing R
  -/
  letI : UniqueFactorizationMonoid R := hR.toUniqueFactorizationMonoid
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
    this : UniqueFactorizationMonoid R := IsDiscreteValuationRing.HasUnitMulPowIrr …
    ⊢ IsDiscreteValuationRing R
  -/
  apply of_ufd_of_unique_irreducible _ hR.unique_irreducible
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    hR : IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization R
    this : UniqueFactorizationMonoid R := IsDiscreteValuationRing.HasUnitMulPowIrr …
    ⊢ Exists fun p => Irreducible p
  -/
  obtain ⟨p, hp, H⟩ := hR
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : R
    hp : Irreducible p
    H : ∀ {x : R}, Ne x 0 → Exists fun n => Associated (HPow.hPow p n) x
    this : UniqueFactorizationMonoid R := IsDiscreteValuationRing.HasUnitMulPowIrr …
    ⊢ Exists fun p => Irreducible p
  -/
  exact ⟨p, hp⟩
  /-
    🎉 no goals
  -/

/- If a ring is equivalent to a DVR, it is itself a DVR. -/

theorem RingEquivClass.isDiscreteValuationRing {A B E : Type*} [CommRing A] [IsDomain A]
    [CommRing B] [IsDomain B] [IsDiscreteValuationRing A] [EquivLike E A B] [RingEquivClass E A B]
    (e : E) : IsDiscreteValuationRing B where
  principal := (isPrincipalIdealRing_iff _).1 <|
    IsPrincipalIdealRing.of_surjective _ (e : A ≃+* B).surjective
  __ : IsLocalRing B := (e : A ≃+* B).isLocalRing
  not_a_field' := by
    obtain ⟨a, ha⟩ := Submodule.nonzero_mem_of_bot_lt (bot_lt_iff_ne_bot.mpr
      <| IsDiscreteValuationRing.not_a_field A)
    /-
      case intro
      A : Type u_2
      B : Type u_3
      E : Type u_4
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : CommRing B
      inst✝³ : IsDomain B
      inst✝² : IsDiscreteValuationRing A
      inst✝¹ : EquivLike E A B
      inst✝ : RingEquivClass E A B
      e : E
      a : Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal A) x
      ha : Ne a 0
      ⊢ Ne (IsLocalRing.maximalIdeal B) Bot.bot
    -/
    rw [Submodule.ne_bot_iff]
    refine ⟨e a, ⟨?_, by simp only [ne_eq, EmbeddingLike.map_eq_zero_iff, ZeroMemClass.coe_eq_zero,
      ha, not_false_eq_true]⟩⟩
    /-
      case intro
      A : Type u_2
      B : Type u_3
      E : Type u_4
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : CommRing B
      inst✝³ : IsDomain B
      inst✝² : IsDiscreteValuationRing A
      inst✝¹ : EquivLike E A B
      inst✝ : RingEquivClass E A B
      e : E
      a : Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal A) x
      ha : Ne a 0
      ⊢ Membership.mem (IsLocalRing.maximalIdeal B) (e ↑a)
    -/
    rw [IsLocalRing.mem_maximalIdeal, map_mem_nonunits_iff e, ← IsLocalRing.mem_maximalIdeal]
    /-
      case intro
      A : Type u_2
      B : Type u_3
      E : Type u_4
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : CommRing B
      inst✝³ : IsDomain B
      inst✝² : IsDiscreteValuationRing A
      inst✝¹ : EquivLike E A B
      inst✝ : RingEquivClass E A B
      e : E
      a : Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal A) x
      ha : Ne a 0
      ⊢ Membership.mem (IsLocalRing.maximalIdeal A) ↑a
    -/
    exact a.2
    /-
      🎉 no goals
    -/


theorem associated_pow_irreducible {x : R} (hx : x ≠ 0) {ϖ : R} (hirr : Irreducible ϖ) :
    ∃ n : ℕ, Associated x (ϖ ^ n) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    ⊢ Exists fun n => Associated x (HPow.hPow ϖ n)
  -/
  have : WfDvdMonoid R := IsNoetherianRing.wfDvdMonoid
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    ⊢ Exists fun n => Associated x (HPow.hPow ϖ n)
  -/
  cases' WfDvdMonoid.exists_factors x hx with fx hfx
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    ⊢ Exists fun n => Associated x (HPow.hPow ϖ n)
  -/
  use Multiset.card fx
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    ⊢ Associated x (HPow.hPow ϖ fx.card)
  -/
  have H := hfx.2
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Associated fx.prod x
    ⊢ Associated x (HPow.hPow ϖ fx.card)
  -/
  rw [← Associates.mk_eq_mk_iff_associated] at H ⊢
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Associates.mk x) (Associates.mk (HPow.hPow ϖ fx.card))
  -/
  rw [← H, ← Associates.prod_mk, Associates.mk_pow, ← Multiset.prod_replicate]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Multiset.map Associates.mk fx).prod (Multiset.replicate fx.card (Associa …
  -/
  congr 1
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ Eq (Multiset.map Associates.mk fx) (Multiset.replicate fx.card (Associates.m …
  -/
  rw [Multiset.eq_replicate]
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ And (Eq (Multiset.map Associates.mk fx).card fx.card) (∀ (b : Associates R), …
  -/
  simp only [true_and, and_imp, Multiset.card_map, eq_self_iff_true, Multiset.mem_map, exists_imp]
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    ⊢ ∀ (b : Associates R) (x : R), Membership.mem fx x → Eq (Associates.mk x) b → …
  -/
  rintro _ _ _ rfl
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    x✝ : R
    a✝ : Membership.mem fx x✝
    ⊢ Eq (Associates.mk x✝) (Associates.mk ϖ)
  -/
  rw [Associates.mk_eq_mk_iff_associated]
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    x✝ : R
    a✝ : Membership.mem fx x✝
    ⊢ Associated x✝ ϖ
  -/
  refine associated_of_irreducible _ ?_ hirr
  /-
    case h.e_a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    x✝ : R
    a✝ : Membership.mem fx x✝
    ⊢ Irreducible x✝
  -/
  apply hfx.1
  /-
    case h.e_a.a
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    this : WfDvdMonoid R
    fx : Multiset R
    hfx : And (∀ (b : R), Membership.mem fx b → Irreducible b) (Associated fx.prod …
    H : Eq (Associates.mk fx.prod) (Associates.mk x)
    x✝ : R
    a✝ : Membership.mem fx x✝
    ⊢ Membership.mem fx x✝
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem eq_unit_mul_pow_irreducible {x : R} (hx : x ≠ 0) {ϖ : R} (hirr : Irreducible ϖ) :
    ∃ (n : ℕ) (u : Rˣ), x = u * ϖ ^ n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    ⊢ Exists fun n => Exists fun u => Eq x (HMul.hMul (↑u) (HPow.hPow ϖ n))
  -/
  obtain ⟨n, hn⟩ := associated_pow_irreducible hx hirr
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    x : R
    hx : Ne x 0
    ϖ : R
    hirr : Irreducible ϖ
    n : Nat
    hn : Associated x (HPow.hPow ϖ n)
    ⊢ Exists fun n => Exists fun u => Eq x (HMul.hMul (↑u) (HPow.hPow ϖ n))
  -/
  obtain ⟨u, rfl⟩ := hn.symm
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    n : Nat
    u : Units R
    hx : Ne (HMul.hMul (HPow.hPow ϖ n) ↑u) 0
    hn : Associated (HMul.hMul (HPow.hPow ϖ n) ↑u) (HPow.hPow ϖ n)
    ⊢ Exists fun n_1 => Exists fun u_1 => Eq (HMul.hMul (HPow.hPow ϖ n) ↑u) (HMul. …
  -/
  use n, u
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    n : Nat
    u : Units R
    hx : Ne (HMul.hMul (HPow.hPow ϖ n) ↑u) 0
    hn : Associated (HMul.hMul (HPow.hPow ϖ n) ↑u) (HPow.hPow ϖ n)
    ⊢ Eq (HMul.hMul (HPow.hPow ϖ n) ↑u) (HMul.hMul (↑u) (HPow.hPow ϖ n))
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


theorem ideal_eq_span_pow_irreducible {s : Ideal R} (hs : s ≠ ⊥) {ϖ : R} (hirr : Irreducible ϖ) :
    ∃ n : ℕ, s = Ideal.span {ϖ ^ n} := by
  have gen_ne_zero : generator s ≠ 0 := by
    rw [Ne, ← eq_bot_iff_generator_eq_zero]
    assumption
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    s : Ideal R
    hs : Ne s Bot.bot
    ϖ : R
    hirr : Irreducible ϖ
    gen_ne_zero : Ne (Submodule.IsPrincipal.generator s) 0
    ⊢ Exists fun n => Eq s (Ideal.span (Singleton.singleton (HPow.hPow ϖ n)))
  -/
  rcases associated_pow_irreducible gen_ne_zero hirr with ⟨n, u, hnu⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    s : Ideal R
    hs : Ne s Bot.bot
    ϖ : R
    hirr : Irreducible ϖ
    gen_ne_zero : Ne (Submodule.IsPrincipal.generator s) 0
    n : Nat
    u : Units R
    hnu : Eq (HMul.hMul (Submodule.IsPrincipal.generator s) ↑u) (HPow.hPow ϖ n)
    ⊢ Exists fun n => Eq s (Ideal.span (Singleton.singleton (HPow.hPow ϖ n)))
  -/
  use n
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    s : Ideal R
    hs : Ne s Bot.bot
    ϖ : R
    hirr : Irreducible ϖ
    gen_ne_zero : Ne (Submodule.IsPrincipal.generator s) 0
    n : Nat
    u : Units R
    hnu : Eq (HMul.hMul (Submodule.IsPrincipal.generator s) ↑u) (HPow.hPow ϖ n)
    ⊢ Eq s (Ideal.span (Singleton.singleton (HPow.hPow ϖ n)))
  -/
  have : span _ = _ := Ideal.span_singleton_generator s
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    s : Ideal R
    hs : Ne s Bot.bot
    ϖ : R
    hirr : Irreducible ϖ
    gen_ne_zero : Ne (Submodule.IsPrincipal.generator s) 0
    n : Nat
    u : Units R
    hnu : Eq (HMul.hMul (Submodule.IsPrincipal.generator s) ↑u) (HPow.hPow ϖ n)
    this : Eq (Ideal.span (Singleton.singleton (Submodule.IsPrincipal.generator s) …
    ⊢ Eq s (Ideal.span (Singleton.singleton (HPow.hPow ϖ n)))
  -/
  rw [← this, ← hnu, span_singleton_eq_span_singleton]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    s : Ideal R
    hs : Ne s Bot.bot
    ϖ : R
    hirr : Irreducible ϖ
    gen_ne_zero : Ne (Submodule.IsPrincipal.generator s) 0
    n : Nat
    u : Units R
    hnu : Eq (HMul.hMul (Submodule.IsPrincipal.generator s) ↑u) (HPow.hPow ϖ n)
    this : Eq (Ideal.span (Singleton.singleton (Submodule.IsPrincipal.generator s) …
    ⊢ Associated (Submodule.IsPrincipal.generator s) (HMul.hMul (Submodule.IsPrinc …
  -/
  use u
  /-
    🎉 no goals
  -/


theorem unit_mul_pow_congr_pow {p q : R} (hp : Irreducible p) (hq : Irreducible q) (u v : Rˣ)
    (m n : ℕ) (h : ↑u * p ^ m = v * q ^ n) : m = n := by
  have key : Associated (Multiset.replicate m p).prod (Multiset.replicate n q).prod := by
    rw [Multiset.prod_replicate, Multiset.prod_replicate, Associated]
    refine ⟨u * v⁻¹, ?_⟩
    simp only [Units.val_mul]
    rw [mul_left_comm, ← mul_assoc, h, mul_right_comm, Units.mul_inv, one_mul]
  have := by
    refine Multiset.card_eq_card_of_rel (UniqueFactorizationMonoid.factors_unique ?_ ?_ key)
    all_goals
      intro x hx
      obtain rfl := Multiset.eq_of_mem_replicate hx
      assumption
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    u v : Units R
    m n : Nat
    h : Eq (HMul.hMul (↑u) (HPow.hPow p m)) (HMul.hMul (↑v) (HPow.hPow q n))
    key : Associated (Multiset.replicate m p).prod (Multiset.replicate n q).prod
    this : Eq (Multiset.replicate m p).card (Multiset.replicate n q).card
    ⊢ Eq m n
  -/
  simpa only [Multiset.card_replicate]
  /-
    🎉 no goals
  -/


theorem unit_mul_pow_congr_unit {ϖ : R} (hirr : Irreducible ϖ) (u v : Rˣ) (m n : ℕ)
    (h : ↑u * ϖ ^ m = v * ϖ ^ n) : u = v := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    u v : Units R
    m n : Nat
    h : Eq (HMul.hMul (↑u) (HPow.hPow ϖ m)) (HMul.hMul (↑v) (HPow.hPow ϖ n))
    ⊢ Eq u v
  -/
  obtain rfl : m = n := unit_mul_pow_congr_pow hirr hirr u v m n h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    u v : Units R
    m : Nat
    h : Eq (HMul.hMul (↑u) (HPow.hPow ϖ m)) (HMul.hMul (↑v) (HPow.hPow ϖ m))
    ⊢ Eq u v
  -/
  rw [← sub_eq_zero] at h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    u v : Units R
    m : Nat
    h : Eq (HSub.hSub (HMul.hMul (↑u) (HPow.hPow ϖ m)) (HMul.hMul (↑v) (HPow.hPow  …
    ⊢ Eq u v
  -/
  rw [← sub_mul, mul_eq_zero] at h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    hirr : Irreducible ϖ
    u v : Units R
    m : Nat
    h : Or (Eq (HSub.hSub ↑u ↑v) 0) (Eq (HPow.hPow ϖ m) 0)
    ⊢ Eq u v
  -/
  cases' h with h h
    /-
      case inl
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      ϖ : R
      hirr : Irreducible ϖ
      u v : Units R
      m : Nat
      h : Eq (HSub.hSub ↑u ↑v) 0
      ⊢ Eq u v
    -/
  · rw [sub_eq_zero] at h
    /-
      case inl
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      ϖ : R
      hirr : Irreducible ϖ
      u v : Units R
      m : Nat
      h : Eq ↑u ↑v
      ⊢ Eq u v
    -/
    exact mod_cast h
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      ϖ : R
      hirr : Irreducible ϖ
      u v : Units R
      m : Nat
      h : Eq (HPow.hPow ϖ m) 0
      ⊢ Eq u v
    -/
  · apply (hirr.ne_zero (pow_eq_zero h)).elim
    /-
      🎉 no goals
    -/


open Classical in
/-- The `ℕ∞`-valued additive valuation on a DVR. -/
noncomputable def addVal (R : Type u) [CommRing R] [IsDomain R] [IsDiscreteValuationRing R] :
    AddValuation R ℕ∞ :=
  multiplicity_addValuation (Classical.choose_spec (exists_prime R))


theorem addVal_def (r : R) (u : Rˣ) {ϖ : R} (hϖ : Irreducible ϖ) (n : ℕ) (hr : r = u * ϖ ^ n) :
    addVal R r = n := by
  classical
  rw [addVal, multiplicity_addValuation_apply, hr, emultiplicity_eq_of_associated_left
      (associated_of_irreducible R hϖ (Classical.choose_spec (exists_prime R)).irreducible),
    emultiplicity_eq_of_associated_right (Associated.symm ⟨u, mul_comm _ _⟩),
    emultiplicity_pow_self_of_prime (irreducible_iff_prime.1 hϖ)]


/-- An alternative definition of the additive valuation, taking units into account.-/
theorem addVal_def' (u : Rˣ) {ϖ : R} (hϖ : Irreducible ϖ) (n : ℕ) :
    addVal R ((u : R) * ϖ ^ n) = n :=
  addVal_def _ u hϖ n rfl


theorem addVal_zero : addVal R 0 = ⊤ :=
  (addVal R).map_zero


theorem addVal_one : addVal R 1 = 0 :=
  (addVal R).map_one


@[simp]
theorem addVal_uniformizer {ϖ : R} (hϖ : Irreducible ϖ) : addVal R ϖ = 1 := by
  simpa only [one_mul, eq_self_iff_true, Units.val_one, pow_one, forall_true_left, Nat.cast_one]
    using addVal_def ϖ 1 hϖ 1


theorem addVal_mul {a b : R} :
    addVal R (a * b) = addVal R a + addVal R b :=
  (addVal R).map_mul _ _


theorem addVal_pow (a : R) (n : ℕ) : addVal R (a ^ n) = n • addVal R a :=
  (addVal R).map_pow _ _


nonrec theorem _root_.Irreducible.addVal_pow {ϖ : R} (h : Irreducible ϖ) (n : ℕ) :
    addVal R (ϖ ^ n) = n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    ϖ : R
    h : Irreducible ϖ
    n : Nat
    ⊢ Eq ((IsDiscreteValuationRing.addVal R) (HPow.hPow ϖ n)) ↑n
  -/
  rw [addVal_pow, addVal_uniformizer h, nsmul_one]
  /-
    🎉 no goals
  -/


theorem addVal_eq_top_iff {a : R} : addVal R a = ⊤ ↔ a = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    a : R
    ⊢ Iff (Eq ((IsDiscreteValuationRing.addVal R) a) Top.top) (Eq a 0)
  -/
  have hi := (Classical.choose_spec (exists_prime R)).irreducible
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsDiscreteValuationRing R
    a : R
    hi : Irreducible (Classical.choose ⋯)
    ⊢ Iff (Eq ((IsDiscreteValuationRing.addVal R) a) Top.top) (Eq a 0)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      a : R
      hi : Irreducible (Classical.choose ⋯)
      ⊢ Eq ((IsDiscreteValuationRing.addVal R) a) Top.top → Eq a 0
    -/
  · contrapose
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      a : R
      hi : Irreducible (Classical.choose ⋯)
      ⊢ Not (Eq a 0) → Not (Eq ((IsDiscreteValuationRing.addVal R) a) Top.top)
    -/
    intro h
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      a : R
      hi : Irreducible (Classical.choose ⋯)
      h : Not (Eq a 0)
      ⊢ Not (Eq ((IsDiscreteValuationRing.addVal R) a) Top.top)
    -/
    obtain ⟨n, ha⟩ := associated_pow_irreducible h hi
    /-
      case mp.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      a : R
      hi : Irreducible (Classical.choose ⋯)
      h : Not (Eq a 0)
      n : Nat
      ha : Associated a (HPow.hPow (Classical.choose ⋯) n)
      ⊢ Not (Eq ((IsDiscreteValuationRing.addVal R) a) Top.top)
    -/
    obtain ⟨u, rfl⟩ := ha.symm
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      hi : Irreducible (Classical.choose ⋯)
      n : Nat
      u : Units R
      h : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose ⋯) n) ↑u) 0)
      ha : Associated (HMul.hMul (HPow.hPow (Classical.choose ⋯) n) ↑u) (HPow.hPow ( …
      ⊢ Not (Eq ((IsDiscreteValuationRing.addVal R) (HMul.hMul (HPow.hPow (Classical …
    -/
    rw [mul_comm, addVal_def' u hi n]
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      hi : Irreducible (Classical.choose ⋯)
      n : Nat
      u : Units R
      h : Not (Eq (HMul.hMul (HPow.hPow (Classical.choose ⋯) n) ↑u) 0)
      ha : Associated (HMul.hMul (HPow.hPow (Classical.choose ⋯) n) ↑u) (HPow.hPow ( …
      ⊢ Not (Eq (↑n) Top.top)
    -/
    nofun
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      a : R
      hi : Irreducible (Classical.choose ⋯)
      ⊢ Eq a 0 → Eq ((IsDiscreteValuationRing.addVal R) a) Top.top
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      hi : Irreducible (Classical.choose ⋯)
      ⊢ Eq ((IsDiscreteValuationRing.addVal R) 0) Top.top
    -/
    exact addVal_zero
    /-
      🎉 no goals
    -/


theorem addVal_le_iff_dvd {a b : R} : addVal R a ≤ addVal R b ↔ a ∣ b := by
  classical
  have hp := Classical.choose_spec (exists_prime R)
  constructor <;> intro h
  · by_cases ha0 : a = 0
    · rw [ha0, addVal_zero, top_le_iff, addVal_eq_top_iff] at h
      rw [h]
      apply dvd_zero
    obtain ⟨n, ha⟩ := associated_pow_irreducible ha0 hp.irreducible
    rw [addVal, multiplicity_addValuation_apply, multiplicity_addValuation_apply,
      emultiplicity_le_emultiplicity_iff] at h
    exact ha.dvd.trans (h n ha.symm.dvd)
  · rw [addVal, multiplicity_addValuation_apply, multiplicity_addValuation_apply]
    exact emultiplicity_le_emultiplicity_of_dvd_right h


theorem addVal_add {a b : R} : min (addVal R a) (addVal R b) ≤ addVal R (a + b) :=
  (addVal R).map_add _ _


instance (R : Type*) [CommRing R] [IsDomain R] [IsDiscreteValuationRing R] :
    IsHausdorff (maximalIdeal R) R where
  haus' x hx := by
    /-
      R✝¹ : Type u
      inst✝⁵ : CommRing R✝¹
      inst✝⁴ : IsDomain R✝¹
      inst✝³ : IsDiscreteValuationRing R✝¹
      R✝ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      x : R
      hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow (IsLocalRing.maximalIdeal R)  …
      ⊢ Eq x 0
    -/
    obtain ⟨ϖ, hϖ⟩ := exists_irreducible R
    simp only [← Ideal.one_eq_top, smul_eq_mul, mul_one, SModEq.zero, hϖ.maximalIdeal_eq,
      Ideal.span_singleton_pow, Ideal.mem_span_singleton, ← addVal_le_iff_dvd, hϖ.addVal_pow] at hx
    /-
      case intro
      R✝¹ : Type u
      inst✝⁵ : CommRing R✝¹
      inst✝⁴ : IsDomain R✝¹
      inst✝³ : IsDiscreteValuationRing R✝¹
      R✝ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsDiscreteValuationRing R
      x ϖ : R
      hϖ : Irreducible ϖ
      hx : ∀ (n : Nat), LE.le (↑n) ((IsDiscreteValuationRing.addVal R) x)
      ⊢ Eq x 0
    -/
    rwa [← addVal_eq_top_iff, ← WithTop.forall_ge_iff_eq_top]
    /-
      🎉 no goals
    -/


/-- A DVR is a valuation ring. -/
instance (priority := 100) of_isDiscreteValuationRing : ValuationRing A := inferInstance


