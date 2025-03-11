lemma ker_residue : RingHom.ker (residue R) = maximalIdeal R :=
  Ideal.mk_ker


@[simp]
lemma residue_eq_zero_iff (x : R) : residue R x = 0 ↔ x ∈ maximalIdeal R := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    x : R
    ⊢ Iff (Eq ((IsLocalRing.residue R) x) 0) (Membership.mem (IsLocalRing.maximalI …
  -/
  rw [← RingHom.mem_ker, ker_residue]
  /-
    🎉 no goals
  -/


lemma residue_ne_zero_iff_isUnit (x : R) : residue R x ≠ 0 ↔ IsUnit x := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    x : R
    ⊢ Iff (Ne ((IsLocalRing.residue R) x) 0) (IsUnit x)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma residue_surjective :
    Function.Surjective (IsLocalRing.residue R) :=
  Ideal.Quotient.mk_surjective


instance ResidueField.algebra {R₀} [CommRing R₀] [Algebra R₀ R] :
    Algebra R₀ (ResidueField R) :=
  Ideal.Quotient.algebra _


instance {R₁ R₂} [CommRing R₁] [CommRing R₂]
    [Algebra R₁ R₂] [Algebra R₁ R] [Algebra R₂ R] [IsScalarTower R₁ R₂ R] :
    IsScalarTower R₁ R₂ (IsLocalRing.ResidueField R) := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : IsLocalRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : IsLocalRing S
    inst✝⁷ : CommRing T
    inst✝⁶ : IsLocalRing T
    R₁ : Type u_4
    R₂ : Type u_5
    inst✝⁵ : CommRing R₁
    inst✝⁴ : CommRing R₂
    inst✝³ : Algebra R₁ R₂
    inst✝² : Algebra R₁ R
    inst✝¹ : Algebra R₂ R
    inst✝ : IsScalarTower R₁ R₂ R
    ⊢ IsScalarTower R₁ R₂ (IsLocalRing.ResidueField R)
  -/
  delta IsLocalRing.ResidueField; infer_instance
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem ResidueField.algebraMap_eq : algebraMap R (ResidueField R) = residue R :=
  rfl


instance : IsLocalHom (IsLocalRing.residue R) :=
  ⟨fun _ ha =>
    Classical.not_not.mp (Ideal.Quotient.eq_zero_iff_mem.not.mp (isUnit_iff_ne_zero.mp ha))⟩


/-- A local ring homomorphism into a field can be descended onto the residue field. -/
def lift {R S : Type*} [CommRing R] [IsLocalRing R] [Field S] (f : R →+* S) [IsLocalHom f] :
    IsLocalRing.ResidueField R →+* S :=
  Ideal.Quotient.lift _ f fun a ha =>
    by_contradiction fun h => ha (isUnit_of_map_unit f a (isUnit_iff_ne_zero.mpr h))


theorem lift_comp_residue {R S : Type*} [CommRing R] [IsLocalRing R] [Field S] (f : R →+* S)
    [IsLocalHom f] : (lift f).comp (residue R) = f :=
  RingHom.ext fun _ => rfl


@[simp]
theorem lift_residue_apply {R S : Type*} [CommRing R] [IsLocalRing R] [Field S] (f : R →+* S)
    [IsLocalHom f] (x) : lift f (residue R x) = f x :=
  rfl


/-- The map on residue fields induced by a local homomorphism between local rings -/
def map (f : R →+* S) [IsLocalHom f] : ResidueField R →+* ResidueField S :=
  Ideal.Quotient.lift (maximalIdeal R) ((Ideal.Quotient.mk _).comp f) fun a ha => by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : IsLocalRing R
      inst✝⁴ : CommRing S
      inst✝³ : IsLocalRing S
      inst✝² : CommRing T
      inst✝¹ : IsLocalRing T
      f : RingHom R S
      inst✝ : IsLocalHom f
      a : R
      ha : Membership.mem (IsLocalRing.maximalIdeal R) a
      ⊢ Eq (((Ideal.Quotient.mk (IsLocalRing.maximalIdeal S)).comp f) a) 0
    -/
    erw [Ideal.Quotient.eq_zero_iff_mem]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : IsLocalRing R
      inst✝⁴ : CommRing S
      inst✝³ : IsLocalRing S
      inst✝² : CommRing T
      inst✝¹ : IsLocalRing T
      f : RingHom R S
      inst✝ : IsLocalHom f
      a : R
      ha : Membership.mem (IsLocalRing.maximalIdeal R) a
      ⊢ Membership.mem (IsLocalRing.maximalIdeal S) (f a)
    -/
    exact map_nonunit f a ha
    /-
      🎉 no goals
    -/


/-- Applying `IsLocalRing.ResidueField.map` to the identity ring homomorphism gives the identity
ring homomorphism. -/
@[simp]
theorem map_id :
    IsLocalRing.ResidueField.map (RingHom.id R) = RingHom.id (IsLocalRing.ResidueField R) :=
  Ideal.Quotient.ringHom_ext <| RingHom.ext fun _ => rfl


/-- The composite of two `IsLocalRing.ResidueField.map`s is the `IsLocalRing.ResidueField.map` of
the composite. -/
theorem map_comp (f : T →+* R) (g : R →+* S) [IsLocalHom f] [IsLocalHom g] :
    IsLocalRing.ResidueField.map (g.comp f) =
      (IsLocalRing.ResidueField.map g).comp (IsLocalRing.ResidueField.map f) :=
  Ideal.Quotient.ringHom_ext <| RingHom.ext fun _ => rfl


theorem map_comp_residue (f : R →+* S) [IsLocalHom f] :
    (ResidueField.map f).comp (residue R) = (residue S).comp f :=
  rfl


theorem map_residue (f : R →+* S) [IsLocalHom f] (r : R) :
    ResidueField.map f (residue R r) = residue S (f r) :=
  rfl


theorem map_id_apply (x : ResidueField R) : map (RingHom.id R) x = x :=
  DFunLike.congr_fun map_id x


@[simp]
theorem map_map (f : R →+* S) (g : S →+* T) (x : ResidueField R) [IsLocalHom f]
    [IsLocalHom g] : map g (map f x) = map (g.comp f) x :=
  DFunLike.congr_fun (map_comp f g).symm x


/-- A ring isomorphism defines an isomorphism of residue fields. -/
@[simps apply]
def mapEquiv (f : R ≃+* S) : IsLocalRing.ResidueField R ≃+* IsLocalRing.ResidueField S where
  toFun := map (f : R →+* S)
  invFun := map (f.symm : S →+* R)
                   /-
                     R : Type u_1
                     S : Type u_2
                     T : Type u_3
                     inst✝⁵ : CommRing R
                     inst✝⁴ : IsLocalRing R
                     inst✝³ : CommRing S
                     inst✝² : IsLocalRing S
                     inst✝¹ : CommRing T
                     inst✝ : IsLocalRing T
                     f : RingEquiv R S
                     x : IsLocalRing.ResidueField R
                     ⊢ Eq ((IsLocalRing.ResidueField.map ↑f.symm) ((IsLocalRing.ResidueField.map ↑f …
                   -/
  left_inv x := by simp only [map_map, RingEquiv.symm_comp, map_id, RingHom.id_apply]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      T : Type u_3
                      inst✝⁵ : CommRing R
                      inst✝⁴ : IsLocalRing R
                      inst✝³ : CommRing S
                      inst✝² : IsLocalRing S
                      inst✝¹ : CommRing T
                      inst✝ : IsLocalRing T
                      f : RingEquiv R S
                      x : IsLocalRing.ResidueField S
                      ⊢ Eq ((IsLocalRing.ResidueField.map ↑f) ((IsLocalRing.ResidueField.map ↑f.symm …
                    -/
  right_inv x := by simp only [map_map, RingEquiv.comp_symm, map_id, RingHom.id_apply]
                    /-
                      🎉 no goals
                    -/
  map_mul' := RingHom.map_mul _
  map_add' := RingHom.map_add _


@[simp]
theorem mapEquiv.symm (f : R ≃+* S) : (mapEquiv f).symm = mapEquiv f.symm :=
  rfl


@[simp]
theorem mapEquiv_trans (e₁ : R ≃+* S) (e₂ : S ≃+* T) :
    mapEquiv (e₁.trans e₂) = (mapEquiv e₁).trans (mapEquiv e₂) :=
  RingEquiv.toRingHom_injective <| map_comp (e₁ : R →+* S) (e₂ : S →+* T)


@[simp]
theorem mapEquiv_refl : mapEquiv (RingEquiv.refl R) = RingEquiv.refl _ :=
  RingEquiv.toRingHom_injective map_id


/-- The group homomorphism from `RingAut R` to `RingAut k` where `k`
is the residue field of `R`. -/
@[simps]
def mapAut : RingAut R →* RingAut (IsLocalRing.ResidueField R) where
  toFun := mapEquiv
  map_mul' e₁ e₂ := mapEquiv_trans e₂ e₁
  map_one' := mapEquiv_refl


/-- If `G` acts on `R` as a `MulSemiringAction`, then it also acts on `IsLocalRing.ResidueField R`.
-/
instance : MulSemiringAction G (IsLocalRing.ResidueField R) :=
  MulSemiringAction.compHom _ <| mapAut.comp (MulSemiringAction.toRingAut G R)


@[simp]
theorem residue_smul (g : G) (r : R) : residue R (g • r) = g • residue R r :=
  rfl


noncomputable instance : Algebra (ResidueField R) (ResidueField S) :=
  (ResidueField.map (algebraMap R S)).toAlgebra


instance : IsScalarTower R (ResidueField R) (ResidueField S) :=
  IsScalarTower.of_algebraMap_eq (congrFun rfl)


instance finiteDimensional_of_noetherian [IsNoetherian R S] :
    FiniteDimensional (ResidueField R) (ResidueField S) := by
  apply IsNoetherian.iff_fg.mp <|
    isNoetherian_of_tower R (S := ResidueField R) (M := ResidueField S) _
  convert isNoetherian_of_surjective S (Ideal.Quotient.mkₐ R (maximalIdeal S)).toLinearMap
    (LinearMap.range_eq_top.mpr Ideal.Quotient.mk_surjective)

-- We want to be able to refer to `hfin`

set_option linter.unusedVariables false in
lemma finite_of_finite [IsNoetherian R S] (hfin : Finite (ResidueField R)) :
    Finite (ResidueField S) := Module.finite_of_finite (ResidueField R)


theorem isLocalHom_residue : IsLocalHom (IsLocalRing.residue R) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    ⊢ IsLocalHom (IsLocalRing.residue R)
  -/
  constructor
  /-
    case map_nonunit
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    ⊢ ∀ (a : R), IsUnit ((IsLocalRing.residue R) a) → IsUnit a
  -/
  intro a ha
  /-
    case map_nonunit
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    a : R
    ha : IsUnit ((IsLocalRing.residue R) a)
    ⊢ IsUnit a
  -/
  by_contra h
  /-
    case map_nonunit
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    a : R
    ha : IsUnit ((IsLocalRing.residue R) a)
    h : Not (IsUnit a)
    ⊢ False
  -/
  erw [Ideal.Quotient.eq_zero_iff_mem.mpr ((IsLocalRing.mem_maximalIdeal _).mpr h)] at ha
  /-
    case map_nonunit
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    a : R
    ha : IsUnit 0
    h : Not (IsUnit a)
    ⊢ False
  -/
  exact ha.ne_zero rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_residue := isLocalHom_residue


@[deprecated (since := "2024-11-11")]
alias LocalRing.ker_residue := IsLocalRing.ker_residue


@[deprecated (since := "2024-11-11")]
alias LocalRing.residue_eq_zero_iff := IsLocalRing.residue_eq_zero_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.residue_ne_zero_iff_isUnit := IsLocalRing.residue_ne_zero_iff_isUnit


@[deprecated (since := "2024-11-11")]
alias LocalRing.residue_surjective := IsLocalRing.residue_surjective


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.algebraMap_eq := IsLocalRing.ResidueField.algebraMap_eq


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.lift := IsLocalRing.ResidueField.lift


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.lift_comp_residue := IsLocalRing.ResidueField.lift_comp_residue


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.lift_residue_apply := IsLocalRing.ResidueField.lift_residue_apply


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map := IsLocalRing.ResidueField.map


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_id := IsLocalRing.ResidueField.map_id


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_comp := IsLocalRing.ResidueField.map_comp


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_comp_residue := IsLocalRing.ResidueField.map_comp_residue


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_residue := IsLocalRing.ResidueField.map_residue


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_id_apply := IsLocalRing.ResidueField.map_id_apply


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.map_map := IsLocalRing.ResidueField.map_map


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.mapEquiv := IsLocalRing.ResidueField.mapEquiv


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.mapEquiv.symm := IsLocalRing.ResidueField.mapEquiv.symm


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.mapEquiv_trans := IsLocalRing.ResidueField.mapEquiv_trans


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.mapEquiv_refl := IsLocalRing.ResidueField.mapEquiv_refl


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.mapAut := IsLocalRing.ResidueField.mapAut


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.residue_smul := IsLocalRing.ResidueField.residue_smul


@[deprecated (since := "2024-11-11")]
alias LocalRing.ResidueField.finite_of_finite := IsLocalRing.ResidueField.finite_of_finite

