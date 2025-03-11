instance localization_unit_isIso (R : CommRingCat) :
    IsIso (CommRingCat.ofHom <| algebraMap R (Localization.Away (1 : R))) :=
  Iso.isIso_hom (IsLocalization.atOne R (Localization.Away (1 : R))).toRingEquiv.toCommRingCatIso


instance localization_unit_isIso' (R : CommRingCat) :
    @IsIso CommRingCat _ R _ (CommRingCat.ofHom <| algebraMap R (Localization.Away (1 : R))) := by
  /-
    R : CommRingCat
    ⊢ CategoryTheory.IsIso (CommRingCat.ofHom (algebraMap (↑R) (Localization.Away  …
  -/
  cases R
  /-
    case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.mk
    carrier✝ : Type u_1
    commRing✝ : CommRing carrier✝
    ⊢ CategoryTheory.IsIso (CommRingCat.ofHom (algebraMap (↑(CommRingCat.mk✝ carri …
  -/
  exact localization_unit_isIso _
  /-
    🎉 no goals
  -/


theorem IsLocalization.epi {R : Type*} [CommRing R] (M : Submonoid R) (S : Type _) [CommRing S]
    [Algebra R S] [IsLocalization M S] : Epi (CommRingCat.ofHom <| algebraMap R S) :=
  ⟨fun {T} _ _ h => CommRingCat.hom_ext <|
    @IsLocalization.ringHom_ext R _ M S _ _ T _ _ _ _ (congrArg CommRingCat.Hom.hom h)⟩


instance Localization.epi {R : Type*} [CommRing R] (M : Submonoid R) :
    Epi (CommRingCat.ofHom <| algebraMap R <| Localization M) :=
  IsLocalization.epi M _


instance Localization.epi' {R : CommRingCat} (M : Submonoid R) :
    @Epi CommRingCat _ R _ (CommRingCat.ofHom <| algebraMap R <| Localization M : _) := by
  /-
    R : CommRingCat
    M : Submonoid ↑R
    ⊢ CategoryTheory.Epi (CommRingCat.ofHom (algebraMap (↑R) (Localization M)))
  -/
  rcases R with ⟨α, str⟩
  /-
    case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.mk
    α : Type u_1
    commRing✝ : CommRing α
    M : Submonoid ↑(CommRingCat.mk✝ α)
    ⊢ CategoryTheory.Epi (CommRingCat.ofHom (algebraMap (↑(CommRingCat.mk✝ α)) (Lo …
  -/
  exact IsLocalization.epi M _
  /-
    🎉 no goals
  -/


@[instance]
theorem CommRingCat.isLocalHom_comp {R S T : CommRingCat} (f : R ⟶ S) (g : S ⟶ T)
    [IsLocalHom g.hom] [IsLocalHom f.hom] : IsLocalHom (f ≫ g).hom :=
  RingHom.isLocalHom_comp _ _


@[deprecated (since := "2024-10-10")]
alias CommRingCat.isLocalRingHom_comp := CommRingCat.isLocalHom_comp


theorem isLocalHom_of_iso {R S : CommRingCat} (f : R ≅ S) : IsLocalHom f.hom.hom :=
  { map_nonunit := fun a ha => by
      /-
        R S : CommRingCat
        f : CategoryTheory.Iso R S
        a : ↑R
        ha : IsUnit (f.hom.hom a)
        ⊢ IsUnit a
      -/
      convert f.inv.hom.isUnit_map ha
      /-
        case h.e'_3
        R S : CommRingCat
        f : CategoryTheory.Iso R S
        a : ↑R
        ha : IsUnit (f.hom.hom a)
        ⊢ Eq a (f.inv.hom (f.hom.hom a))
      -/
      simp }
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_of_iso := isLocalHom_of_iso

-- see Note [lower instance priority]

@[instance 100]
theorem isLocalHom_of_isIso {R S : CommRingCat} (f : R ⟶ S) [IsIso f] :
    IsLocalHom f.hom :=
  isLocalHom_of_iso (asIso f)


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_of_isIso := isLocalHom_of_isIso

