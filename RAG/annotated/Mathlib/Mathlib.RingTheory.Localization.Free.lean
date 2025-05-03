include f in
/--
If `M` is a finitely presented `R`-module,
then any `Rₛ`-basis of `Mₛ` for some `S : Submonoid R` can be lifted to
a `Rᵣ`-basis of `Mᵣ` for some `r ∈ S`.
-/
lemma Module.FinitePresentation.exists_basis_localizedModule_powers
    (Rₛ) [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ M'] [IsScalarTower R Rₛ M']
    [IsLocalization S Rₛ] [Module.FinitePresentation R M]
    {I} [Finite I] (b : Basis I Rₛ M') :
    ∃ (r : R) (hr : r ∈ S)
      (b' : Basis I (Localization (.powers r)) (LocalizedModule (.powers r) M)),
      ∀ i, (LocalizedModule.lift (.powers r) f fun s ↦ IsLocalizedModule.map_units f
        ⟨s.1, SetLike.le_def.mp (Submonoid.powers_le.mpr hr) s.2⟩) (b' i) = b i := by
  /-
    R : Type u_4
    M : Type u_5
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝⁹ : AddCommGroup M'
    inst✝⁸ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁷ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁶ : CommRing Rₛ
    inst✝⁵ : Algebra R Rₛ
    inst✝⁴ : Module Rₛ M'
    inst✝³ : IsScalarTower R Rₛ M'
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    I : Type u_6
    inst✝ : Finite I
    b : Basis I Rₛ M'
    ⊢ Exists fun r => Exists fun hr => Exists fun b' => ∀ (i : I), Eq ((LocalizedM …
  -/
  have : Module.FinitePresentation R (I →₀ R) := Module.finitePresentation_of_projective _ _
  obtain ⟨r, hr, e, he⟩ := Module.FinitePresentation.exists_lift_equiv_of_isLocalizedModule S f
    (Finsupp.mapRange.linearMap (Algebra.linearMap R Rₛ)) (b.repr.restrictScalars R)
  let e' := IsLocalizedModule.iso (.powers r) (Finsupp.mapRange.linearMap (α := I)
    (Algebra.linearMap R (Localization (.powers r))))
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝⁹ : AddCommGroup M'
    inst✝⁸ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁷ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁶ : CommRing Rₛ
    inst✝⁵ : Algebra R Rₛ
    inst✝⁴ : Module Rₛ M'
    inst✝³ : IsScalarTower R Rₛ M'
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    I : Type u_6
    inst✝ : Finite I
    b : Basis I Rₛ M'
    this : Module.FinitePresentation R (Finsupp I R)
    r : R
    hr : Membership.mem S r
    e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
    he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
    e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
    ⊢ Exists fun r => Exists fun hr => Exists fun b' => ∀ (i : I), Eq ((LocalizedM …
  -/
  refine ⟨r, hr, .ofRepr (e ≪≫ₗ ?_), ?_⟩
  · exact
    { __ := e',
      toLinearMap := e'.extendScalarsOfIsLocalization (.powers r) (Localization (.powers r)) }
    /-
      case intro.intro.intro.refine_2
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      ⊢ ∀ (i : I),
          Eq
            ((LocalizedModule.lift (Submonoid.powers r) f ⋯)
              ({
                  repr :=
                    e.trans
                      (let __spread.0 := e';
                      { toLinearMap := LinearMap.extendScalarsOfIsLocalization (Subm …
                i))
            (b i)
    -/
  · intro i
    have : e'.symm _ = _ := LinearMap.congr_fun (IsLocalizedModule.iso_symm_comp (.powers r)
      (Finsupp.mapRange.linearMap (Algebra.linearMap R (Localization (.powers r)))))
      (Finsupp.single i 1)
    simp only [Finsupp.mapRange.linearMap_apply, Finsupp.mapRange_single, Algebra.linearMap_apply,
      map_one, LocalizedModule.mkLinearMap_apply] at this
    /-
      case intro.intro.intro.refine_2
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this✝ : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      i : I
      this : Eq (e'.symm (Finsupp.single i 1)) (LocalizedModule.mk (Finsupp.single i …
      ⊢ Eq
          ((LocalizedModule.lift (Submonoid.powers r) f ⋯)
            ({
                repr :=
                  e.trans
                    (let __spread.0 := e';
                    { toLinearMap := LinearMap.extendScalarsOfIsLocalization (Submon …
              i))
          (b i)
    -/
    show LocalizedModule.lift _ _ _ (e.symm (e'.symm _)) = _
    /-
      case intro.intro.intro.refine_2
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this✝ : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      i : I
      this : Eq (e'.symm (Finsupp.single i 1)) (LocalizedModule.mk (Finsupp.single i …
      ⊢ Eq ((LocalizedModule.lift (Submonoid.powers r) f ⋯) (e.symm (e'.symm (Finsup …
    -/
    replace he := LinearMap.congr_fun he (e.symm (e'.symm (Finsupp.single i 1)))
    simp only [LinearMap.coe_comp, LinearMap.coe_restrictScalars, LinearEquiv.coe_coe,
      Function.comp_apply, LinearEquiv.apply_symm_apply, LinearEquiv.restrictScalars_apply] at he
    /-
      case intro.intro.intro.refine_2
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this✝ : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      i : I
      this : Eq (e'.symm (Finsupp.single i 1)) (LocalizedModule.mk (Finsupp.single i …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      ⊢ Eq ((LocalizedModule.lift (Submonoid.powers r) f ⋯) (e.symm (e'.symm (Finsup …
    -/
    apply b.repr.injective
    /-
      case intro.intro.intro.refine_2.a
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this✝ : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      i : I
      this : Eq (e'.symm (Finsupp.single i 1)) (LocalizedModule.mk (Finsupp.single i …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      ⊢ Eq (b.repr ((LocalizedModule.lift (Submonoid.powers r) f ⋯) (e.symm (e'.symm …
    -/
    rw [← he, Basis.repr_self, this, LocalizedModule.lift_mk]
    /-
      case intro.intro.intro.refine_2.a
      R : Type u_4
      M : Type u_5
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      S : Submonoid R
      M' : Type u_1
      inst✝⁹ : AddCommGroup M'
      inst✝⁸ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝⁷ : IsLocalizedModule S f
      Rₛ : Type u_3
      inst✝⁶ : CommRing Rₛ
      inst✝⁵ : Algebra R Rₛ
      inst✝⁴ : Module Rₛ M'
      inst✝³ : IsScalarTower R Rₛ M'
      inst✝² : IsLocalization S Rₛ
      inst✝¹ : Module.FinitePresentation R M
      I : Type u_6
      inst✝ : Finite I
      b : Basis I Rₛ M'
      this✝ : Module.FinitePresentation R (Finsupp I R)
      r : R
      hr : Membership.mem S r
      e : LinearEquiv (RingHom.id (Localization (Submonoid.powers r))) (LocalizedMod …
      e' : LinearEquiv (RingHom.id R) (LocalizedModule (Submonoid.powers r) (Finsupp …
      i : I
      this : Eq (e'.symm (Finsupp.single i 1)) (LocalizedModule.mk (Finsupp.single i …
      he : Eq ((LocalizedModule.lift (Submonoid.powers r) (Finsupp.mapRange.linearMa …
      ⊢ Eq (↑(Inv.inv ⋯.unit) ((Finsupp.mapRange.linearMap (Algebra.linearMap R Rₛ)) …
    -/
    simp
    /-
      🎉 no goals
    -/


include f in
/--
If `M` is a finitely presented `R`-module
such that `Mₛ` is free over `Rₛ` for some `S : Submonoid R`,
then `Mᵣ` is already free over `Rᵣ` for some `r ∈ S`.
-/
lemma Module.FinitePresentation.exists_free_localizedModule_powers
    (Rₛ) [CommRing Rₛ] [Algebra R Rₛ] [Module Rₛ M'] [IsScalarTower R Rₛ M'] [Nontrivial Rₛ]
    [IsLocalization S Rₛ] [Module.FinitePresentation R M] [Module.Free Rₛ M'] :
    ∃ r, r ∈ S ∧
      Module.Free (Localization (.powers r)) (LocalizedModule (.powers r) M) ∧
      Module.finrank (Localization (.powers r)) (LocalizedModule (.powers r) M) =
        Module.finrank Rₛ M' := by
  /-
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    ⊢ Exists fun r => And (Membership.mem S r) (And (Module.Free (Localization (Su …
  -/
  let I := Module.Free.ChooseBasisIndex Rₛ M'
  /-
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    I : Type u_1 := Module.Free.ChooseBasisIndex Rₛ M'
    ⊢ Exists fun r => And (Membership.mem S r) (And (Module.Free (Localization (Su …
  -/
  let b : Basis I Rₛ M' := Module.Free.chooseBasis Rₛ M'
  /-
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    I : Type u_1 := Module.Free.ChooseBasisIndex Rₛ M'
    b : Basis I Rₛ M' := Module.Free.chooseBasis Rₛ M'
    ⊢ Exists fun r => And (Membership.mem S r) (And (Module.Free (Localization (Su …
  -/
  have : Module.Finite Rₛ M' := Module.Finite.of_isLocalizedModule S (Rₚ := Rₛ) f
  /-
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    I : Type u_1 := Module.Free.ChooseBasisIndex Rₛ M'
    b : Basis I Rₛ M' := Module.Free.chooseBasis Rₛ M'
    this : Module.Finite Rₛ M'
    ⊢ Exists fun r => And (Membership.mem S r) (And (Module.Free (Localization (Su …
  -/
  obtain ⟨r, hr, b', _⟩ := Module.FinitePresentation.exists_basis_localizedModule_powers S f Rₛ b
  have := (show Localization (.powers r) →+* Rₛ from IsLocalization.map (M := .powers r) (T := S) _
    (RingHom.id _) (Submonoid.powers_le.mpr hr)).domain_nontrivial
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    I : Type u_1 := Module.Free.ChooseBasisIndex Rₛ M'
    b : Basis I Rₛ M' := Module.Free.chooseBasis Rₛ M'
    this✝ : Module.Finite Rₛ M'
    r : R
    hr : Membership.mem S r
    b' : Basis I (Localization (Submonoid.powers r)) (LocalizedModule (Submonoid.p …
    h✝ : ∀ (i : I), Eq ((LocalizedModule.lift (Submonoid.powers r) f ⋯) (b' i)) (b …
    this : Nontrivial (Localization (Submonoid.powers r))
    ⊢ Exists fun r => And (Membership.mem S r) (And (Module.Free (Localization (Su …
  -/
  refine ⟨r, hr, .of_basis b', ?_⟩
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : AddCommGroup M
    inst✝¹¹ : Module R M
    S : Submonoid R
    M' : Type u_1
    inst✝¹⁰ : AddCommGroup M'
    inst✝⁹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁸ : IsLocalizedModule S f
    Rₛ : Type u_3
    inst✝⁷ : CommRing Rₛ
    inst✝⁶ : Algebra R Rₛ
    inst✝⁵ : Module Rₛ M'
    inst✝⁴ : IsScalarTower R Rₛ M'
    inst✝³ : Nontrivial Rₛ
    inst✝² : IsLocalization S Rₛ
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Free Rₛ M'
    I : Type u_1 := Module.Free.ChooseBasisIndex Rₛ M'
    b : Basis I Rₛ M' := Module.Free.chooseBasis Rₛ M'
    this✝ : Module.Finite Rₛ M'
    r : R
    hr : Membership.mem S r
    b' : Basis I (Localization (Submonoid.powers r)) (LocalizedModule (Submonoid.p …
    h✝ : ∀ (i : I), Eq ((LocalizedModule.lift (Submonoid.powers r) f ⋯) (b' i)) (b …
    this : Nontrivial (Localization (Submonoid.powers r))
    ⊢ Eq (Module.finrank (Localization (Submonoid.powers r)) (LocalizedModule (Sub …
  -/
  rw [Module.finrank_eq_nat_card_basis b, Module.finrank_eq_nat_card_basis b']
  /-
    🎉 no goals
  -/

