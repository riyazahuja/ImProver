/--
The residue field at a prime ideal, defined to be the residue field of the local ring
`Localization.Prime I`.
We also provide an `IsFractionRing (R ⧸ I) I.ResidueField` instance.
-/
abbrev Ideal.ResidueField : Type _ :=
  IsLocalRing.ResidueField (Localization.AtPrime I)


/-- If `I = f⁻¹(J)`, then there is an canonical embedding `κ(I) ↪ κ(J)`. -/
noncomputable
abbrev Ideal.ResidueField.map (I : Ideal R) [I.IsPrime] (J : Ideal A) [J.IsPrime]
    (f : R →+* A) (hf : I = J.comap f) : I.ResidueField →+* J.ResidueField :=
  IsLocalRing.ResidueField.map (Localization.localRingHom I J f hf)


/-- If `I = f⁻¹(J)`, then there is an canonical embedding `κ(I) ↪ κ(J)`. -/
noncomputable
def Ideal.ResidueField.mapₐ (I : Ideal R) [I.IsPrime] (J : Ideal A) [J.IsPrime]
    (hf : I = J.comap (algebraMap R A)) : I.ResidueField →ₐ[R] J.ResidueField where
  __ := Ideal.ResidueField.map I J (algebraMap R A) hf
  commutes' r := by
    rw [IsScalarTower.algebraMap_apply R (Localization.AtPrime I),
      IsLocalRing.ResidueField.algebraMap_eq]
    simp only [RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe,
      MonoidHom.coe_coe, IsLocalRing.ResidueField.map_residue, Localization.localRingHom_to_map]
    /-
      R : Type ?u.12425
      A : Type ?u.12428
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      I✝ : Ideal R
      inst✝² : I✝.IsPrime
      I : Ideal R
      inst✝¹ : I.IsPrime
      J : Ideal A
      inst✝ : J.IsPrime
      hf : Eq I (Ideal.comap (algebraMap R A) J)
      r : R
      ⊢ Eq ((IsLocalRing.residue (Localization.AtPrime J)) ((algebraMap A (Localizat …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp] lemma Ideal.ResidueField.mapₐ_apply (I : Ideal R) [I.IsPrime] (J : Ideal A) [J.IsPrime]
    (hf : I = J.comap (algebraMap R A)) (x) :
    Ideal.ResidueField.mapₐ I J hf x = Ideal.ResidueField.map I J _ hf x := rfl


variable {I} in
@[simp]
lemma Ideal.algebraMap_residueField_eq_zero {x} :
    algebraMap R I.ResidueField x = 0 ↔ x ∈ I := by
  rw [IsScalarTower.algebraMap_apply R (Localization.AtPrime I),
    IsLocalRing.ResidueField.algebraMap_eq, IsLocalRing.residue_eq_zero_iff]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    x : R
    ⊢ Iff (Membership.mem (IsLocalRing.maximalIdeal (Localization.AtPrime I)) ((al …
  -/
  exact IsLocalization.AtPrime.to_map_mem_maximal_iff _ _ _
  /-
    🎉 no goals
  -/


@[simp]
lemma Ideal.ker_algebraMap_residueField :
    RingHom.ker (algebraMap R I.ResidueField) = I :=
  Ideal.ext fun _ ↦ Ideal.algebraMap_residueField_eq_zero


attribute [-instance] IsLocalRing.ResidueField.field in
instance : Algebra (R ⧸ I) I.ResidueField :=
  (Ideal.Quotient.liftₐ I (Algebra.ofId _ _)
    fun _ ↦ Ideal.algebraMap_residueField_eq_zero.mpr).toRingHom.toAlgebra


instance : IsScalarTower R (R ⧸ I) I.ResidueField :=
  IsScalarTower.of_algebraMap_eq fun _ ↦ rfl


@[simp]
lemma algebraMap_mk (x) :
    algebraMap (R ⧸ I) I.ResidueField (Ideal.Quotient.mk _ x) =
    algebraMap R I.ResidueField x := rfl


lemma Ideal.injective_algebraMap_quotient_residueField :
    Function.Injective (algebraMap (R ⧸ I) I.ResidueField) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Function.Injective ⇑(algebraMap (HasQuotient.Quotient R I) I.ResidueField)
  -/
  rw [RingHom.injective_iff_ker_eq_bot]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Eq (RingHom.ker (algebraMap (HasQuotient.Quotient R I) I.ResidueField)) Bot. …
  -/
  refine (Ideal.ker_quotient_lift _ _).trans ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Eq (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker ↑(Algebra.ofId R I.ResidueF …
  -/
  show map (Quotient.mk I) (RingHom.ker (algebraMap R I.ResidueField)) = ⊥
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Eq (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker (algebraMap R I.ResidueFiel …
  -/
  rw [Ideal.ker_algebraMap_residueField, map_quotient_self]
  /-
    🎉 no goals
  -/


instance : IsFractionRing (R ⧸ I) I.ResidueField where
  map_units' y := isUnit_iff_ne_zero.mpr
    (map_ne_zero_of_mem_nonZeroDivisors _ I.injective_algebraMap_quotient_residueField y.2)
  surj' x := by
    /-
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x : I.ResidueField
      ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (HasQuotient.Quotient R I) I. …
    -/
    obtain ⟨x, rfl⟩ := IsLocalRing.residue_surjective x
    /-
      case intro
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x : Localization.AtPrime I
      ⊢ Exists fun x_1 => Eq (HMul.hMul ((IsLocalRing.residue (Localization.AtPrime  …
    -/
    obtain ⟨x, ⟨s, hs⟩, rfl⟩ := IsLocalization.mk'_surjective I.primeCompl x
    /-
      case intro.intro.intro.mk
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x s : R
      hs : Membership.mem I.primeCompl s
      ⊢ Exists fun x_1 => Eq (HMul.hMul ((IsLocalRing.residue (Localization.AtPrime  …
    -/
    refine ⟨⟨Ideal.Quotient.mk _ x, ⟨Ideal.Quotient.mk _ s, ?_⟩⟩, ?_⟩
      /-
        case intro.intro.intro.mk.refine_1
        R : Type u_1
        A : Type ?u.61279
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        I : Ideal R
        inst✝ : I.IsPrime
        x s : R
        hs : Membership.mem I.primeCompl s
        ⊢ Membership.mem (nonZeroDivisors (HasQuotient.Quotient R I)) ((Ideal.Quotient …
      -/
    · rwa [mem_nonZeroDivisors_iff_ne_zero, ne_eq, Ideal.Quotient.eq_zero_iff_mem]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.mk.refine_2
        R : Type u_1
        A : Type ?u.61279
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        I : Ideal R
        inst✝ : I.IsPrime
        x s : R
        hs : Membership.mem I.primeCompl s
        ⊢ Eq (HMul.hMul ((IsLocalRing.residue (Localization.AtPrime I)) (IsLocalizatio …
      -/
    · simp [IsScalarTower.algebraMap_eq R (Localization.AtPrime I) I.ResidueField, ← map_mul]
      /-
        🎉 no goals
      -/
  exists_of_eq {x y} e := by
    /-
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x y : HasQuotient.Quotient R I
      e : Eq ((algebraMap (HasQuotient.Quotient R I) I.ResidueField) x) ((algebraMap …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    /-
      case intro
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      y : HasQuotient.Quotient R I
      x : R
      e : Eq ((algebraMap (HasQuotient.Quotient R I) I.ResidueField) ((Ideal.Quotien …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) ((Ideal.Quotient.mk I) x)) (HMul.hMul (↑c …
    -/
    obtain ⟨y, rfl⟩ := Ideal.Quotient.mk_surjective y
    /-
      case intro.intro
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x y : R
      e : Eq ((algebraMap (HasQuotient.Quotient R I) I.ResidueField) ((Ideal.Quotien …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) ((Ideal.Quotient.mk I) x)) (HMul.hMul (↑c …
    -/
    rw [← sub_eq_zero, ← map_sub, ← map_sub] at e
    simp only [IsLocalRing.ResidueField.algebraMap_eq, IsLocalRing.residue_eq_zero_iff,
      IsScalarTower.algebraMap_apply R (Localization.AtPrime I) I.ResidueField, algebraMap_mk,
      IsLocalization.AtPrime.to_map_mem_maximal_iff _ I, ← Ideal.Quotient.mk_eq_mk_iff_sub_mem] at e
    /-
      case intro.intro
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x y : R
      e : Eq ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) ((Ideal.Quotient.mk I) x)) (HMul.hMul (↑c …
    -/
    use 1
    /-
      case h
      R : Type u_1
      A : Type ?u.61279
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal R
      inst✝ : I.IsPrime
      x y : R
      e : Eq ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)
      ⊢ Eq (HMul.hMul (↑1) ((Ideal.Quotient.mk I) x)) (HMul.hMul (↑1) ((Ideal.Quotie …
    -/
    simp [e]
    /-
      🎉 no goals
    -/


lemma Ideal.bijective_algebraMap_quotient_residueField (I : Ideal R) [I.IsMaximal] :
    Function.Bijective (algebraMap (R ⧸ I) I.ResidueField) :=
  ⟨I.injective_algebraMap_quotient_residueField, IsFractionRing.surjective_iff_isField.mpr
    ((Quotient.maximal_ideal_iff_isField_quotient I).mp inferInstance)⟩

