theorem Module.free_of_isLocalizedModule {Rₛ Mₛ} [AddCommGroup Mₛ] [Module R Mₛ]
    [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ Mₛ] [IsScalarTower R Rₛ Mₛ]
    (S) (f : M →ₗ[R] Mₛ) [IsLocalization S Rₛ] [IsLocalizedModule S f] [Module.Free R M] :
    Module.Free Rₛ Mₛ :=
    Free.of_equiv (IsLocalizedModule.isBaseChange S Rₛ f).equiv


universe uR' uM' in
/--
Also see `IsLocalizedModule.lift_rank_eq` for a version for non-free modules,
but requires `S` to not contain any zero-divisors.
-/
theorem Module.lift_rank_of_isLocalizedModule_of_free
    (Rₛ : Type uR') {Mₛ : Type uM'} [AddCommGroup Mₛ] [Module R Mₛ]
    [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ Mₛ] [IsScalarTower R Rₛ Mₛ] (S : Submonoid R)
    (f : M →ₗ[R] Mₛ) [IsLocalization S Rₛ] [IsLocalizedModule S f] [Module.Free R M]
    [Nontrivial Rₛ] :
    Cardinal.lift.{uM} (Module.rank Rₛ Mₛ) = Cardinal.lift.{uM'} (Module.rank R M) := by
  /-
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type uR'
    Mₛ : Type uM'
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    ⊢ Eq (Cardinal.lift.{uM, uM'} (Module.rank Rₛ Mₛ)) (Cardinal.lift.{uM', uM} (M …
  -/
  apply Cardinal.lift_injective.{max uM' uR'}
  /-
    case a
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type uR'
    Mₛ : Type uM'
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    ⊢ Eq (Cardinal.lift.{max uM' uR', max uM uM'} (Cardinal.lift.{uM, uM'} (Module …
  -/
  have := (algebraMap R Rₛ).domain_nontrivial
  /-
    case a
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type uR'
    Mₛ : Type uM'
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    this : Nontrivial R
    ⊢ Eq (Cardinal.lift.{max uM' uR', max uM uM'} (Cardinal.lift.{uM, uM'} (Module …
  -/
  have := (IsLocalizedModule.isBaseChange S Rₛ f).equiv.lift_rank_eq.symm
  simp only [rank_tensorProduct, rank_self,
    Cardinal.lift_one, one_mul, Cardinal.lift_lift] at this ⊢
  /-
    case a
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type uR'
    Mₛ : Type uM'
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{max uM uR', uM'} (Module.rank Rₛ Mₛ)) (Cardinal.lift …
    ⊢ Eq (Cardinal.lift.{max uM uM' uR', uM'} (Module.rank Rₛ Mₛ)) (Cardinal.lift. …
  -/
  convert this
  /-
    case h.e'_2.h.e
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type uR'
    Mₛ : Type uM'
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{max uM uR', uM'} (Module.rank Rₛ Mₛ)) (Cardinal.lift …
    ⊢ Eq Cardinal.lift.{max uM uM' uR', uM'} Cardinal.lift.{max uM uR', uM'}
  -/
  exact Cardinal.lift_umax
  /-
    🎉 no goals
  -/


theorem Module.finrank_of_isLocalizedModule_of_free
    (Rₛ : Type*) {Mₛ : Type*} [AddCommGroup Mₛ] [Module R Mₛ]
    [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ Mₛ] [IsScalarTower R Rₛ Mₛ] (S : Submonoid R)
    (f : M →ₗ[R] Mₛ) [IsLocalization S Rₛ] [IsLocalizedModule S f] [Module.Free R M]
    [Nontrivial Rₛ] :
    Module.finrank Rₛ Mₛ = Module.finrank R M := by
  /-
    R : Type u_1
    M : Type uM
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    Rₛ : Type u_4
    Mₛ : Type u_5
    inst✝⁹ : AddCommGroup Mₛ
    inst✝⁸ : Module R Mₛ
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ Mₛ
    inst✝⁴ : IsScalarTower R Rₛ Mₛ
    S : Submonoid R
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝³ : IsLocalization S Rₛ
    inst✝² : IsLocalizedModule S f
    inst✝¹ : Module.Free R M
    inst✝ : Nontrivial Rₛ
    ⊢ Eq (Module.finrank Rₛ Mₛ) (Module.finrank R M)
  -/
  simpa using congr(Cardinal.toNat $(Module.lift_rank_of_isLocalizedModule_of_free Rₛ S f))
  /-
    🎉 no goals
  -/


theorem Module.projective_of_isLocalizedModule {Rₛ Mₛ} [AddCommGroup Mₛ] [Module R Mₛ]
    [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ Mₛ] [IsScalarTower R Rₛ Mₛ]
    (S) (f : M →ₗ[R] Mₛ) [IsLocalization S Rₛ] [IsLocalizedModule S f] [Module.Projective R M] :
    Module.Projective Rₛ Mₛ :=
  Projective.of_equiv (IsLocalizedModule.isBaseChange S Rₛ f).equiv


theorem LinearMap.split_surjective_of_localization_maximal
    (f : M →ₗ[R] N) [Module.FinitePresentation R N]
    (H : ∀ (I : Ideal R) (_ : I.IsMaximal),
    ∃ (g : _ →ₗ[Localization.AtPrime I] _),
      (LocalizedModule.map I.primeCompl f).comp g = LinearMap.id) :
    ∃ (g : N →ₗ[R] M), f.comp g = LinearMap.id := by
  /-
    R : Type u_1
    N : Type u_2
    M : Type uM
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝ : Module.FinitePresentation R N
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
    ⊢ Exists fun g => Eq (f.comp g) LinearMap.id
  -/
  show LinearMap.id ∈ LinearMap.range (LinearMap.llcomp R N M N f)
  refine Submodule.mem_of_localization_maximal _ (fun P _ ↦ LocalizedModule.map P.primeCompl) _ _
    fun I hI ↦ ?_
  /-
    R : Type u_1
    N : Type u_2
    M : Type uM
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝ : Module.FinitePresentation R N
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
    I : Ideal R
    hI : I.IsMaximal
    ⊢ Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => LocalizedModu …
  -/
  rw [LocalizedModule.map_id]
  have : LinearMap.id ∈ LinearMap.range (LinearMap.llcomp _
    (LocalizedModule I.primeCompl N) _ _ (LocalizedModule.map I.primeCompl f)) := H I hI
  /-
    R : Type u_1
    N : Type u_2
    M : Type uM
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝ : Module.FinitePresentation R N
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
    I : Ideal R
    hI : I.IsMaximal
    this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
    ⊢ Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => LocalizedModu …
  -/
  convert this
    /-
      case h.e
      R : Type u_1
      N : Type u_2
      M : Type uM
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      f : LinearMap (RingHom.id R) M N
      inst✝ : Module.FinitePresentation R N
      H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
      I : Ideal R
      hI : I.IsMaximal
      this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
      ⊢ Eq (Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => Localized …
    -/
  · ext f
    /-
      case h.e.h.a
      R : Type u_1
      N : Type u_2
      M : Type uM
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      f✝ : LinearMap (RingHom.id R) M N
      inst✝ : Module.FinitePresentation R N
      H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
      I : Ideal R
      hI : I.IsMaximal
      this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
      f : LinearMap (RingHom.id (Localization I.primeCompl)) (LocalizedModule I.prim …
      ⊢ Iff (Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => Localize …
    -/
    constructor
      /-
        case h.e.h.a.mp
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f✝ : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        f : LinearMap (RingHom.id (Localization I.primeCompl)) (LocalizedModule I.prim …
        ⊢ Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => LocalizedModu …
      -/
    · intro hf
      /-
        case h.e.h.a.mp
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f✝ : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        f : LinearMap (RingHom.id (Localization I.primeCompl)) (LocalizedModule I.prim …
        hf : Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => LocalizedM …
        ⊢ Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.primeComp …
      -/
      obtain ⟨a, ha, c, rfl⟩ := hf
      /-
        case h.e.h.a.mp.intro.intro.intro
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        a : LinearMap (RingHom.id R) N N
        ha : Membership.mem (LinearMap.range ((LinearMap.llcomp R N M N) f)) a
        c : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.primeComp …
      -/
      obtain ⟨g, rfl⟩ := ha
      /-
        case h.e.h.a.mp.intro.intro.intro.intro
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.primeComp …
      -/
      use IsLocalizedModule.mk' (LocalizedModule.map I.primeCompl) g c
      apply ((Module.End_isUnit_iff _).mp <| IsLocalizedModule.map_units
        (LocalizedModule.map I.primeCompl) c).injective
      /-
        case h.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (((algebraMap R (Module.End R (LinearMap (RingHom.id (Localization I.prim …
      -/
      dsimp
      /-
        case h.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (HSMul.hSMul (↑c) (((LinearMap.llcomp (Localization I.primeCompl) (Locali …
      -/
      conv_rhs => rw [← Submonoid.smul_def]
      /-
        case h.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (HSMul.hSMul (↑c) (((LinearMap.llcomp (Localization I.primeCompl) (Locali …
      -/
      conv_lhs => rw [← LinearMap.map_smul_of_tower]
      /-
        case h.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (((LinearMap.llcomp (Localization I.primeCompl) (LocalizedModule I.primeC …
      -/
      rw [← Submonoid.smul_def, IsLocalizedModule.mk'_cancel', IsLocalizedModule.mk'_cancel']
      /-
        case h.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (((LinearMap.llcomp (Localization I.primeCompl) (LocalizedModule I.primeC …
      -/
      apply LinearMap.restrictScalars_injective R
      /-
        case h.a.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq (↑R (((LinearMap.llcomp (Localization I.primeCompl) (LocalizedModule I.pr …
      -/
      apply IsLocalizedModule.ext I.primeCompl (LocalizedModule.mkLinearMap I.primeCompl N)
        /-
          case h.a.a.map_unit
          R : Type u_1
          N : Type u_2
          M : Type uM
          inst✝⁵ : CommRing R
          inst✝⁴ : AddCommGroup M
          inst✝³ : Module R M
          inst✝² : AddCommGroup N
          inst✝¹ : Module R N
          f : LinearMap (RingHom.id R) M N
          inst✝ : Module.FinitePresentation R N
          H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
          I : Ideal R
          hI : I.IsMaximal
          this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
          c : Subtype fun x => Membership.mem I.primeCompl x
          g : LinearMap (RingHom.id R) N M
          ⊢ ∀ (x : Subtype fun x => Membership.mem I.primeCompl x), IsUnit ((algebraMap  …
        -/
      · exact IsLocalizedModule.map_units (LocalizedModule.mkLinearMap I.primeCompl N)
        /-
          🎉 no goals
        -/
      /-
        case h.a.a.h
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        c : Subtype fun x => Membership.mem I.primeCompl x
        g : LinearMap (RingHom.id R) N M
        ⊢ Eq ((↑R (((LinearMap.llcomp (Localization I.primeCompl) (LocalizedModule I.p …
      -/
      ext
      simp only [LocalizedModule.map_mk, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
        Function.comp_apply, LocalizedModule.mkLinearMap_apply, LinearMap.llcomp_apply,
        LocalizedModule.map_mk]
      /-
        case h.e.h.a.mpr
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f✝ : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        f : LinearMap (RingHom.id (Localization I.primeCompl)) (LocalizedModule I.prim …
        ⊢ Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.primeComp …
      -/
    · rintro ⟨g, rfl⟩
      obtain ⟨⟨g, s⟩, rfl⟩ :=
        IsLocalizedModule.mk'_surjective I.primeCompl (LocalizedModule.map I.primeCompl) g
      /-
        case h.e.h.a.mpr.intro.intro.mk
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        g : LinearMap (RingHom.id R) N M
        s : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Membership.mem (Submodule.localized₀ I.primeCompl ((fun P x => LocalizedModu …
      -/
      simp only [Function.uncurry_apply_pair, Submodule.restrictScalars_mem]
      /-
        case h.e.h.a.mpr.intro.intro.mk
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        g : LinearMap (RingHom.id R) N M
        s : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Membership.mem (Submodule.localized₀ I.primeCompl (LocalizedModule.map I.pri …
      -/
      refine ⟨f.comp g, ⟨g, rfl⟩, s, ?_⟩
      apply ((Module.End_isUnit_iff _).mp <| IsLocalizedModule.map_units
         (LocalizedModule.map I.primeCompl) s).injective
      simp only [Module.algebraMap_end_apply, ← Submonoid.smul_def, IsLocalizedModule.mk'_cancel',
        ← LinearMap.map_smul_of_tower]
      /-
        case h.e.h.a.mpr.intro.intro.mk.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        g : LinearMap (RingHom.id R) N M
        s : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Eq ((LocalizedModule.map I.primeCompl) (f.comp g)) (((LinearMap.llcomp (Loca …
      -/
      apply LinearMap.restrictScalars_injective R
      /-
        case h.e.h.a.mpr.intro.intro.mk.a.a
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        g : LinearMap (RingHom.id R) N M
        s : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Eq (↑R ((LocalizedModule.map I.primeCompl) (f.comp g))) (↑R (((LinearMap.llc …
      -/
      apply IsLocalizedModule.ext I.primeCompl (LocalizedModule.mkLinearMap I.primeCompl N)
        /-
          case h.e.h.a.mpr.intro.intro.mk.a.a.map_unit
          R : Type u_1
          N : Type u_2
          M : Type uM
          inst✝⁵ : CommRing R
          inst✝⁴ : AddCommGroup M
          inst✝³ : Module R M
          inst✝² : AddCommGroup N
          inst✝¹ : Module R N
          f : LinearMap (RingHom.id R) M N
          inst✝ : Module.FinitePresentation R N
          H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
          I : Ideal R
          hI : I.IsMaximal
          this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
          g : LinearMap (RingHom.id R) N M
          s : Subtype fun x => Membership.mem I.primeCompl x
          ⊢ ∀ (x : Subtype fun x => Membership.mem I.primeCompl x), IsUnit ((algebraMap  …
        -/
      · exact IsLocalizedModule.map_units (LocalizedModule.mkLinearMap I.primeCompl N)
        /-
          🎉 no goals
        -/
      /-
        case h.e.h.a.mpr.intro.intro.mk.a.a.h
        R : Type u_1
        N : Type u_2
        M : Type uM
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        f : LinearMap (RingHom.id R) M N
        inst✝ : Module.FinitePresentation R N
        H : ∀ (I : Ideal R) (x : I.IsMaximal), Exists fun g => Eq (((LocalizedModule.m …
        I : Ideal R
        hI : I.IsMaximal
        this : Membership.mem (LinearMap.range ((LinearMap.llcomp (Localization I.prim …
        g : LinearMap (RingHom.id R) N M
        s : Subtype fun x => Membership.mem I.primeCompl x
        ⊢ Eq ((↑R ((LocalizedModule.map I.primeCompl) (f.comp g))).comp (LocalizedModu …
      -/
      ext
      simp only [coe_comp, coe_restrictScalars, Function.comp_apply,
        LocalizedModule.mkLinearMap_apply, LocalizedModule.map_mk, llcomp_apply]


theorem Module.projective_of_localization_maximal (H : ∀ (I : Ideal R) (_ : I.IsMaximal),
    Module.Projective (Localization.AtPrime I) (LocalizedModule I.primeCompl M))
    [Module.FinitePresentation R M] : Module.Projective R M := by
  /-
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    ⊢ Module.Projective R M
  -/
  have : Module.Finite R M := by infer_instance
  /-
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this : Module.Finite R M
    ⊢ Module.Projective R M
  -/
  have : (⊤ : Submodule R M).FG := this.out
  /-
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝ : Module.Finite R M
    this : Top.top.FG
    ⊢ Module.Projective R M
  -/
  have : ∃ (s : Finset M), _ := this
  /-
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝¹ : Module.Finite R M
    this✝ : Top.top.FG
    this : Exists fun s => Eq (Submodule.span R ↑s) Top.top
    ⊢ Module.Projective R M
  -/
  obtain ⟨s, hs⟩ := this
  /-
    case intro
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝ : Module.Finite R M
    this : Top.top.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) Top.top
    ⊢ Module.Projective R M
  -/
  let N := s →₀ R
  /-
    case intro
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝ : Module.Finite R M
    this : Top.top.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) Top.top
    N : Type (max uM u_1) := Finsupp (Subtype fun x => Membership.mem s x) R
    ⊢ Module.Projective R M
  -/
  let f : N →ₗ[R] M := Finsupp.linearCombination R (Subtype.val : s → M)
  have hf : Function.Surjective f := by
    rw [← LinearMap.range_eq_top, Finsupp.range_linearCombination, Subtype.range_val]
    convert hs
  have (I : Ideal R) (hI : I.IsMaximal) :=
    letI := H I hI
    Module.projective_lifting_property (LocalizedModule.map I.primeCompl f) LinearMap.id
    (LocalizedModule.map_surjective _ _ hf)
  /-
    case intro
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝¹ : Module.Finite R M
    this✝ : Top.top.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) Top.top
    N : Type (max uM u_1) := Finsupp (Subtype fun x => Membership.mem s x) R
    f : LinearMap (RingHom.id R) N M := Finsupp.linearCombination R Subtype.val
    hf : Function.Surjective ⇑f
    this : ∀ (I : Ideal R) (hI : I.IsMaximal), Exists fun h => Eq (((LocalizedModu …
    ⊢ Module.Projective R M
  -/
  obtain ⟨g, hg⟩ := LinearMap.split_surjective_of_localization_maximal _ this
  /-
    case intro.intro
    R : Type u_1
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime …
    inst✝ : Module.FinitePresentation R M
    this✝¹ : Module.Finite R M
    this✝ : Top.top.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) Top.top
    N : Type (max uM u_1) := Finsupp (Subtype fun x => Membership.mem s x) R
    f : LinearMap (RingHom.id R) N M := Finsupp.linearCombination R Subtype.val
    hf : Function.Surjective ⇑f
    this : ∀ (I : Ideal R) (hI : I.IsMaximal), Exists fun h => Eq (((LocalizedModu …
    g : LinearMap (RingHom.id R) M N
    hg : Eq (f.comp g) LinearMap.id
    ⊢ Module.Projective R M
  -/
  exact Module.Projective.of_split _ _ hg
  /-
    🎉 no goals
  -/


attribute [local instance] RingHomInvPair.of_ringEquiv in
include f in
/--
A variant of `Module.projective_of_localization_maximal` that accepts `IsLocalizedModule`.
-/
theorem Module.projective_of_localization_maximal'
    (H : ∀ (I : Ideal R) (_ : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I))
    [Module.FinitePresentation R M] : Module.Projective R M := by
  /-
    R : Type u_1
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommRing (Rₚ P)
    inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
    inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
    inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
    inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl (f …
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I)
    inst✝ : Module.FinitePresentation R M
    ⊢ Module.Projective R M
  -/
  apply Module.projective_of_localization_maximal
  /-
    case H
    R : Type u_1
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommRing (Rₚ P)
    inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
    inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
    inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
    inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl (f …
    H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I)
    inst✝ : Module.FinitePresentation R M
    ⊢ ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Localization.AtPrime I …
  -/
  intros P hP
  refine Module.Projective.of_ringEquiv (M := Mₚ P)
    (IsLocalization.algEquiv P.primeCompl (Rₚ P) (Localization.AtPrime P)).toRingEquiv
    { __ := IsLocalizedModule.linearEquiv P.primeCompl (f P)
        (LocalizedModule.mkLinearMap P.primeCompl M)
      map_smul' := ?_ }
    /-
      case H
      R : Type u_1
      M : Type uM
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
      inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommRing (Rₚ P)
      inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
      inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
      Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
      inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
      inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
      inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
      inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
      f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
      inst : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl (f …
      H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I)
      inst✝ : Module.FinitePresentation R M
      P : Ideal R
      hP : P.IsMaximal
      ⊢ ∀ (m : Rₚ P) (x : Mₚ P), Eq ((↑__spread✝⁻⁰).toFun (HSMul.hSMul m x)) (HSMul. …
    -/
  · intros r m
    /-
      case H
      R : Type u_1
      M : Type uM
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
      inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommRing (Rₚ P)
      inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
      inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
      Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
      inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
      inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
      inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
      inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
      f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
      inst : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl (f …
      H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I)
      inst✝ : Module.FinitePresentation R M
      P : Ideal R
      hP : P.IsMaximal
      r : Rₚ P
      m : Mₚ P
      ⊢ Eq ((↑__spread✝⁻⁰).toFun (HSMul.hSMul r m)) (HSMul.hSMul (↑(IsLocalization.a …
    -/
    obtain ⟨r, s, rfl⟩ := IsLocalization.mk'_surjective P.primeCompl r
    apply ((Module.End_isUnit_iff _).mp
      (IsLocalizedModule.map_units (LocalizedModule.mkLinearMap P.primeCompl M) s)).1
    /-
      case H.intro.intro.a
      R : Type u_1
      M : Type uM
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
      inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommRing (Rₚ P)
      inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
      inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
      Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
      inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
      inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
      inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
      inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
      f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
      inst : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl (f …
      H : ∀ (I : Ideal R) (x : I.IsMaximal), Module.Projective (Rₚ I) (Mₚ I)
      inst✝ : Module.FinitePresentation R M
      P : Ideal R
      hP : P.IsMaximal
      m : Mₚ P
      r : R
      s : Subtype fun x => Membership.mem P.primeCompl x
      ⊢ Eq (((algebraMap R (Module.End R (LocalizedModule P.primeCompl M))) ↑s) ((↑_ …
    -/
    dsimp
    simp only [← map_smul, ← smul_assoc, IsLocalization.smul_mk'_self, algebraMap_smul,
      IsLocalization.map_id_mk']

