/-- The function field of an irreducible scheme is the local ring at its generic point.
Despite the name, this is a field only when the scheme is integral. -/
noncomputable abbrev Scheme.functionField [IrreducibleSpace X] : CommRingCat :=
  X.presheaf.stalk (genericPoint X)


/-- The restriction map from a component to the function field. -/
noncomputable abbrev Scheme.germToFunctionField [IrreducibleSpace X] (U : X.Opens)
    [h : Nonempty U] : Γ(X, U) ⟶ X.functionField :=
  X.presheaf.germ U
    (genericPoint X)
                                                                 /-
                                                                   X : AlgebraicGeometry.Scheme
                                                                   inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
                                                                   U : X.Opens
                                                                   h : Nonempty ↑↑(↑U).toPresheafedSpace
                                                                   ⊢ (Inter.inter Set.univ ↑U).Nonempty
                                                                 -/
      (((genericPoint_spec X).mem_open_set_iff U.isOpen).mpr (by simpa using h))
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


noncomputable instance [IrreducibleSpace X] (U : X.Opens) [Nonempty U] :
    Algebra Γ(X, U) X.functionField :=
  (X.germToFunctionField U).hom.toAlgebra


noncomputable instance [IsIntegral X] : Field X.functionField := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    ⊢ Field ↑X.functionField
  -/
  refine .ofIsUnitOrEqZero fun a ↦ ?_
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    a : ↑X.functionField
    ⊢ Or (IsUnit a) (Eq a 0)
  -/
  obtain ⟨U, m, s, rfl⟩ := TopCat.Presheaf.germ_exist (C := CommRingCat) _ _ a
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    m : Membership.mem U (genericPoint ↑↑X.toPresheafedSpace)
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    ⊢ Or (IsUnit ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s)) ( …
  -/
  rw [or_iff_not_imp_right, ← (X.presheaf.germ _ _ m).hom.map_zero]
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    m : Membership.mem U (genericPoint ↑↑X.toPresheafedSpace)
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    ⊢ Not (Eq ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s) ((X.p …
  -/
  intro ha
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    m : Membership.mem U (genericPoint ↑↑X.toPresheafedSpace)
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    ha : Not (Eq ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s) (( …
    ⊢ IsUnit ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s)
  -/
  replace ha := ne_of_apply_ne _ ha
  have hs : genericPoint X ∈ RingedSpace.basicOpen _ s := by
    rw [← SetLike.mem_coe, (genericPoint_spec X).mem_open_set_iff,
      Set.univ_inter, Set.nonempty_iff_ne_empty, Ne, ← Opens.coe_bot, ← SetLike.ext'_iff]
    · erw [basicOpen_eq_bot_iff]
      exact ha
    · exact (RingedSpace.basicOpen _ _).isOpen
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    m : Membership.mem U (genericPoint ↑↑X.toPresheafedSpace)
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    ha : Ne s 0
    hs : Membership.mem (AlgebraicGeometry.RingedSpace.basicOpen X.toSheafedSpace  …
    ⊢ IsUnit ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s)
  -/
  have := (X.presheaf.germ _ _ hs).hom.isUnit_map (RingedSpace.isUnit_res_basicOpen _ s)
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    m : Membership.mem U (genericPoint ↑↑X.toPresheafedSpace)
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    ha : Ne s 0
    hs : Membership.mem (AlgebraicGeometry.RingedSpace.basicOpen X.toSheafedSpace  …
    this : IsUnit ((X.presheaf.germ (AlgebraicGeometry.RingedSpace.basicOpen X.toS …
    ⊢ IsUnit ((X.presheaf.germ U (genericPoint ↑↑X.toPresheafedSpace) m) s)
  -/
  rwa [CommRingCat.germ_res_apply] at this
  /-
    🎉 no goals
  -/


theorem germ_injective_of_isIntegral [IsIntegral X] {U : X.Opens} (x : X) (hx : x ∈ U) :
    Function.Injective (X.presheaf.germ U x hx) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Function.Injective ⇑(X.presheaf.germ U x hx).hom
  -/
  rw [injective_iff_map_eq_zero]
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ ∀ (a : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).hom a) …
  -/
  intro y hy
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : ↑(X.presheaf.obj { unop := U })
    hy : Eq ((X.presheaf.germ U x hx).hom y) 0
    ⊢ Eq y 0
  -/
  rw [← (X.presheaf.germ U x hx).hom.map_zero] at hy
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : ↑(X.presheaf.obj { unop := U })
    hy : Eq ((X.presheaf.germ U x hx).hom y) ((X.presheaf.germ U x hx).hom 0)
    ⊢ Eq y 0
  -/
  obtain ⟨W, hW, iU, iV, e⟩ := X.presheaf.germ_eq _ hx hx _ _ hy
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : ↑(X.presheaf.obj { unop := U })
    hy : Eq ((X.presheaf.germ U x hx).hom y) ((X.presheaf.germ U x hx).hom 0)
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hW : Membership.mem W x
    iU iV : Quiver.Hom W U
    e : Eq ((X.presheaf.map iU.op) y) ((X.presheaf.map iV.op) 0)
    ⊢ Eq y 0
  -/
  cases Subsingleton.elim iU iV
  /-
    case intro.intro.intro.intro.refl
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : ↑(X.presheaf.obj { unop := U })
    hy : Eq ((X.presheaf.germ U x hx).hom y) ((X.presheaf.germ U x hx).hom 0)
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hW : Membership.mem W x
    iU : Quiver.Hom W U
    e : Eq ((X.presheaf.map iU.op) y) ((X.presheaf.map iU.op) 0)
    ⊢ Eq y 0
  -/
  haveI : Nonempty W := ⟨⟨_, hW⟩⟩
  /-
    case intro.intro.intro.intro.refl
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : ↑(X.presheaf.obj { unop := U })
    hy : Eq ((X.presheaf.germ U x hx).hom y) ((X.presheaf.germ U x hx).hom 0)
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hW : Membership.mem W x
    iU : Quiver.Hom W U
    e : Eq ((X.presheaf.map iU.op) y) ((X.presheaf.map iU.op) 0)
    this : Nonempty ↑↑(↑W).toPresheafedSpace
    ⊢ Eq y 0
  -/
  exact map_injective_of_isIntegral X iU e
  /-
    🎉 no goals
  -/


theorem Scheme.germToFunctionField_injective [IsIntegral X] (U : X.Opens) [Nonempty U] :
    Function.Injective (X.germToFunctionField U) :=
  germ_injective_of_isIntegral _ _ _


theorem genericPoint_eq_of_isOpenImmersion {X Y : Scheme} (f : X ⟶ Y) [H : IsOpenImmersion f]
    [hX : IrreducibleSpace X] [IrreducibleSpace Y] :
    f.base (genericPoint X) = genericPoint Y := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    hX : IrreducibleSpace ↑↑X.toPresheafedSpace
    inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
    ⊢ Eq (f.base (genericPoint ↑↑X.toPresheafedSpace)) (genericPoint ↑↑Y.toPreshea …
  -/
  apply ((genericPoint_spec Y).eq _).symm
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    hX : IrreducibleSpace ↑↑X.toPresheafedSpace
    inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
    ⊢ IsGenericPoint (f.base (genericPoint ↑↑X.toPresheafedSpace)) Set.univ
  -/
  convert (genericPoint_spec X).image (show Continuous f.base by fun_prop)
  /-
    case h.e'_4
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    hX : IrreducibleSpace ↑↑X.toPresheafedSpace
    inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
    ⊢ Eq Set.univ (closure (Set.image (⇑f.base) Set.univ))
  -/
  symm
  /-
    case h.e'_4
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    hX : IrreducibleSpace ↑↑X.toPresheafedSpace
    inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
    ⊢ Eq (closure (Set.image (⇑f.base) Set.univ)) Set.univ
  -/
  rw [← Set.univ_subset_iff]
  /-
    case h.e'_4
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    hX : IrreducibleSpace ↑↑X.toPresheafedSpace
    inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
    ⊢ HasSubset.Subset Set.univ (closure (Set.image (⇑f.base) Set.univ))
  -/
  convert subset_closure_inter_of_isPreirreducible_of_isOpen _ H.base_open.isOpen_range _
    /-
      case h.e'_4.h.e'_3
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      hX : IrreducibleSpace ↑↑X.toPresheafedSpace
      inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
      ⊢ Eq (Set.image (⇑f.base) Set.univ) (Inter.inter Set.univ (Set.range ⇑(Algebra …
    -/
  · rw [Set.univ_inter, Set.image_univ]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.convert_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      hX : IrreducibleSpace ↑↑X.toPresheafedSpace
      inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
      ⊢ IsPreirreducible Set.univ
    -/
  · apply PreirreducibleSpace.isPreirreducible_univ (X := Y)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.convert_3
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      hX : IrreducibleSpace ↑↑X.toPresheafedSpace
      inst✝ : IrreducibleSpace ↑↑Y.toPresheafedSpace
      ⊢ (Inter.inter Set.univ (Set.range ⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom f). …
    -/
  · exact ⟨_, trivial, Set.mem_range_self hX.2.some⟩
    /-
      🎉 no goals
    -/


noncomputable instance stalkFunctionFieldAlgebra [IrreducibleSpace X] (x : X) :
    Algebra (X.presheaf.stalk x) X.functionField := by
  -- TODO: can we write this normally after the refactor finishes?
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    ⊢ Algebra ↑(X.presheaf.stalk x) ↑X.functionField
  -/
  apply RingHom.toAlgebra
  /-
    case i
    X : AlgebraicGeometry.Scheme
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    ⊢ RingHom ↑(X.presheaf.stalk x) ↑X.functionField
  -/
  exact (X.presheaf.stalkSpecializes ((genericPoint_spec X).specializes trivial)).hom
  /-
    🎉 no goals
  -/


instance functionField_isScalarTower [IrreducibleSpace X] (U : X.Opens) (x : U)
    [Nonempty U] : IsScalarTower Γ(X, U) (X.presheaf.stalk x) X.functionField := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    x : Subtype fun x => Membership.mem U x
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ IsScalarTower ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf.stalk ↑x) ↑X.func …
  -/
  apply IsScalarTower.of_algebraMap_eq'
  /-
    case h
    X : AlgebraicGeometry.Scheme
    inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    x : Subtype fun x => Membership.mem U x
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ Eq (algebraMap ↑(X.presheaf.obj { unop := U }) ↑X.functionField) ((algebraMa …
  -/
  simp_rw [RingHom.algebraMap_toAlgebra]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    x : Subtype fun x => Membership.mem U x
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ Eq (X.germToFunctionField U).hom ((X.presheaf.stalkSpecializes ⋯).hom.comp ( …
  -/
  change _ = (X.presheaf.germ U x x.2 ≫ _).hom
  /-
    case h
    X : AlgebraicGeometry.Scheme
    inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    x : Subtype fun x => Membership.mem U x
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ Eq (X.germToFunctionField U).hom (CategoryTheory.CategoryStruct.comp (X.pres …
  -/
  rw [X.presheaf.germ_stalkSpecializes]
  /-
    🎉 no goals
  -/


noncomputable instance (R : CommRingCat.{u}) [IsDomain R] :
    Algebra R (Spec R).functionField :=
  -- TODO: can we write this normally after the refactor finishes?
                          /-
                            X : AlgebraicGeometry.Scheme
                            R : CommRingCat
                            inst✝ : IsDomain ↑R
                            ⊢ RingHom ↑R ↑(AlgebraicGeometry.Spec R).functionField
                          -/
  RingHom.toAlgebra <| by apply CommRingCat.Hom.hom; apply StructureSheaf.toStalk
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem genericPoint_eq_bot_of_affine (R : CommRingCat) [IsDomain R] :
    genericPoint (Spec R) = (⊥ : PrimeSpectrum R) := by
  /-
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (genericPoint ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace) Bot.bot
  -/
  apply (genericPoint_spec (Spec R)).eq
  /-
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ IsGenericPoint Bot.bot Set.univ
  -/
  rw [isGenericPoint_def]
  /-
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (closure (Singleton.singleton Bot.bot)) Set.univ
  -/
  rw [← PrimeSpectrum.zeroLocus_vanishingIdeal_eq_closure, PrimeSpectrum.vanishingIdeal_singleton]
  /-
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (PrimeSpectrum.zeroLocus ↑Bot.bot.asIdeal) Set.univ
  -/
  rw [← PrimeSpectrum.zeroLocus_singleton_zero]
  /-
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (PrimeSpectrum.zeroLocus ↑Bot.bot.asIdeal) (PrimeSpectrum.zeroLocus (Sing …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance functionField_isFractionRing_of_affine (R : CommRingCat.{u}) [IsDomain R] :
    IsFractionRing R (Spec R).functionField := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ IsFractionRing ↑R ↑(AlgebraicGeometry.Spec R).functionField
  -/
  convert StructureSheaf.IsLocalization.to_stalk R (genericPoint (Spec R))
  /-
    case a
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Iff (IsFractionRing ↑R ↑(AlgebraicGeometry.Spec R).functionField) (IsLocaliz …
  -/
  delta IsFractionRing IsLocalization.AtPrime
  -- Porting note: `congr` does not work for `Iff`
  /-
    case a
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Iff (IsLocalization (nonZeroDivisors ↑R) ↑(AlgebraicGeometry.Spec R).functio …
  -/
  apply Eq.to_iff
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (IsLocalization (nonZeroDivisors ↑R) ↑(AlgebraicGeometry.Spec R).function …
  -/
  congr 1
  /-
    case a.a.e_M
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (nonZeroDivisors ↑R) (genericPoint ↑↑(AlgebraicGeometry.Spec R).toPreshea …
  -/
  rw [genericPoint_eq_bot_of_affine]
  /-
    case a.a.e_M
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ Eq (nonZeroDivisors ↑R) Bot.bot.asIdeal.primeCompl
  -/
  ext
  /-
    case a.a.e_M.h
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    x✝ : ↑R
    ⊢ Iff (Membership.mem (nonZeroDivisors ↑R) x✝) (Membership.mem Bot.bot.asIdeal …
  -/
  exact mem_nonZeroDivisors_iff_ne_zero
  /-
    🎉 no goals
  -/


instance {X : Scheme} [IsIntegral X] {U : X.Opens} [Nonempty U] :
    IsIntegral U :=
  isIntegral_of_isOpenImmersion U.ι


theorem IsAffineOpen.primeIdealOf_genericPoint {X : Scheme} [IsIntegral X] {U : X.Opens}
    (hU : IsAffineOpen U) [h : Nonempty U] :
    hU.primeIdealOf
        ⟨genericPoint X,
                                                                    /-
                                                                      X✝ : AlgebraicGeometry.Scheme
                                                                      X : AlgebraicGeometry.Scheme
                                                                      inst✝ : AlgebraicGeometry.IsIntegral X
                                                                      U : X.Opens
                                                                      hU : AlgebraicGeometry.IsAffineOpen U
                                                                      h : Nonempty ↑↑(↑U).toPresheafedSpace
                                                                      ⊢ (Inter.inter Set.univ ↑U).Nonempty
                                                                    -/
          ((genericPoint_spec X).mem_open_set_iff U.isOpen).mpr (by simpa using h)⟩ =
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      genericPoint (Spec Γ(X, U)) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ Eq (hU.primeIdealOf ⟨genericPoint ↑↑X.toPresheafedSpace, ⋯⟩) (genericPoint ↑ …
  -/
  haveI : IsAffine _ := hU
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : Nonempty ↑↑(↑U).toPresheafedSpace
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq (hU.primeIdealOf ⟨genericPoint ↑↑X.toPresheafedSpace, ⋯⟩) (genericPoint ↑ …
  -/
  delta IsAffineOpen.primeIdealOf
  convert
    genericPoint_eq_of_isOpenImmersion
      (U.toScheme.isoSpec.hom ≫ Spec.map (X.presheaf.map (eqToHom U.isOpenEmbedding_obj_top).op))
  -- Porting note: this was `ext1`
  /-
    case h.e'_2.h.h.e'_6
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : Nonempty ↑↑(↑U).toPresheafedSpace
    this : AlgebraicGeometry.IsAffine ↑U
    e_1✝ : Eq (PrimeSpectrum ↑(X.presheaf.obj { unop := U })) ↑↑(AlgebraicGeometry …
    ⊢ Eq ⟨genericPoint ↑↑X.toPresheafedSpace, ⋯⟩ (genericPoint ↑↑(↑U).toPresheafed …
  -/
  apply Subtype.ext
  /-
    case h.e'_2.h.h.e'_6.a
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : Nonempty ↑↑(↑U).toPresheafedSpace
    this : AlgebraicGeometry.IsAffine ↑U
    e_1✝ : Eq (PrimeSpectrum ↑(X.presheaf.obj { unop := U })) ↑↑(AlgebraicGeometry …
    ⊢ Eq ↑⟨genericPoint ↑↑X.toPresheafedSpace, ⋯⟩ ↑(genericPoint ↑↑(↑U).toPresheaf …
  -/
  exact (genericPoint_eq_of_isOpenImmersion U.ι).symm
  /-
    🎉 no goals
  -/


theorem functionField_isFractionRing_of_isAffineOpen [IsIntegral X] (U : X.Opens)
    (hU : IsAffineOpen U) [Nonempty U] :
    IsFractionRing Γ(X, U) X.functionField := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ IsFractionRing ↑(X.presheaf.obj { unop := U }) ↑X.functionField
  -/
  haveI : IsAffine _ := hU
  haveI : IsIntegral U :=
    @isIntegral_of_isAffine_of_isDomain _ _ _
      (by rw [Scheme.Opens.toScheme_presheaf_obj, Opens.isOpenEmbedding_obj_top]; infer_instance)
  /-
    X : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    this✝ : AlgebraicGeometry.IsAffine ↑U
    this : AlgebraicGeometry.IsIntegral ↑U
    ⊢ IsFractionRing ↑(X.presheaf.obj { unop := U }) ↑X.functionField
  -/
  delta IsFractionRing Scheme.functionField
  convert hU.isLocalization_stalk ⟨genericPoint X,
    (((genericPoint_spec X).mem_open_set_iff U.isOpen).mpr (by simpa using ‹Nonempty U›))⟩ using 1
  /-
    case h.e'_3.h
    X : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    this✝ : AlgebraicGeometry.IsAffine ↑U
    this : AlgebraicGeometry.IsIntegral ↑U
    ⊢ Eq (nonZeroDivisors ↑(X.presheaf.obj { unop := U })) (hU.primeIdealOf ⟨gener …
  -/
  rw [hU.primeIdealOf_genericPoint, genericPoint_eq_bot_of_affine]
  /-
    case h.e'_3.h
    X : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    inst✝ : Nonempty ↑↑(↑U).toPresheafedSpace
    this✝ : AlgebraicGeometry.IsAffine ↑U
    this : AlgebraicGeometry.IsIntegral ↑U
    ⊢ Eq (nonZeroDivisors ↑(X.presheaf.obj { unop := U })) Bot.bot.asIdeal.primeCo …
  -/
  ext; exact mem_nonZeroDivisors_iff_ne_zero
       /-
         🎉 no goals
       -/


instance (x : X) : IsAffine (X.affineCover.obj x) :=
  AlgebraicGeometry.isAffine_Spec _


instance [IsIntegral X] (x : X) :
    IsFractionRing (X.presheaf.stalk x) X.functionField :=
  let U : X.Opens := (X.affineCover.map x).opensRange
  have hU : IsAffineOpen U := isAffineOpen_opensRange (X.affineCover.map x)
  let x : U := ⟨x, X.affineCover.covers x⟩
  have : Nonempty U := ⟨x⟩
  let M := (hU.primeIdealOf x).asIdeal.primeCompl
  have := hU.isLocalization_stalk x
  have := functionField_isFractionRing_of_isAffineOpen X U hU
  -- Porting note: the following two lines were not needed.
  let _hA := Presheaf.algebra_section_stalk X.presheaf x
  have := functionField_isScalarTower X U x
  .isFractionRing_of_isDomain_of_isLocalization M ↑(Presheaf.stalk X.presheaf x)
    (Scheme.functionField X)


