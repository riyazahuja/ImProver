/--
The functor from the category of `R`-modules to presheaves on `CompHausLike P` given by locally
constant maps.
-/
@[simps]
def functorToPresheaves : ModuleCat.{max u w} R ⥤ ((CompHausLike.{u} P)ᵒᵖ ⥤ ModuleCat R) where
  obj X := {
    obj := fun ⟨S⟩ ↦ ModuleCat.of R (LocallyConstant S X)
    map := fun f ↦ ModuleCat.ofHom (comapₗ R f.unop) }
  map f := { app := fun S ↦ ModuleCat.ofHom (mapₗ R f.hom) }


/-- `CompHausLike.LocallyConstantModule.functorToPresheaves` lands in sheaves. -/
@[simps]
def functor : haveI := CompHausLike.preregular hs
    ModuleCat R ⥤ Sheaf (coherentTopology (CompHausLike.{u} P)) (ModuleCat R) where
  obj X := {
    val := (functorToPresheaves.{w, u} R).obj X
    cond := by
      /-
        P : TopCat → Prop
        R : Type (max u w)
        inst✝² : Ring R
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacks P
        hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
        X : ModuleCat R
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
      -/
      have := CompHausLike.preregular hs
      apply Presheaf.isSheaf_coherent_of_hasPullbacks_of_comp
        (s := CategoryTheory.forget (ModuleCat R))
      /-
        case hF
        P : TopCat → Prop
        R : Type (max u w)
        inst✝² : Ring R
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacks P
        hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
        X : ModuleCat R
        this : CategoryTheory.Preregular (CompHausLike P)
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
      -/
      exact ((CompHausLike.LocallyConstant.functor P hs).obj _).cond }
      /-
        🎉 no goals
      -/
  map f := ⟨(functorToPresheaves.{w, u} R).map f⟩


/-- `functorToPresheaves` in the case of `CompHaus`. -/
abbrev functorToPresheaves : ModuleCat.{u+1} R ⥤ (CompHaus.{u}ᵒᵖ ⥤ ModuleCat R) :=
  CompHausLike.LocallyConstantModule.functorToPresheaves.{u+1, u} R


/-- `functorToPresheaves` as a functor to condensed modules. -/
abbrev functor : ModuleCat R ⥤ CondensedMod.{u} R :=
  CompHausLike.LocallyConstantModule.functor.{u+1, u} R
                  /-
                    P : TopCat → Prop
                    R : Type (u + 1)
                    inst✝ : Ring R
                    x✝² x✝¹ : CompHausLike fun x => True
                    x✝ : Quiver.Hom x✝² x✝¹
                    ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.27181 x✝² x✝¹ x✝)) (List.con …
                  -/
                  /-
                    🎉 no goals
                  -/
    (fun _ _ _ ↦ ((CompHaus.effectiveEpi_tfae _).out 0 2).mp)
                  /-
                    🎉 no goals
                  -/


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteAux₁ (M : ModuleCat.{u+1} R) :
    M ≅ (ModuleCat.of R (LocallyConstant (CompHaus.of PUnit.{u+1}) M)) where
  hom := ModuleCat.ofHom (constₗ R)
  inv := ModuleCat.ofHom (evalₗ R PUnit.unit)


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteAux₂ (M : ModuleCat R) :
    (discrete _).obj M ≅ (discrete _).obj
      (ModuleCat.of R (LocallyConstant (CompHaus.of PUnit.{u+1}) M)) :=
  (discrete _).mapIso (functorIsoDiscreteAux₁ R M)


instance (M : ModuleCat R) : IsIso ((forget R).map
    ((discreteUnderlyingAdj (ModuleCat R)).counit.app ((functor R).obj M))) := by
  /-
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso ((Condensed.forget R).map ((Condensed.discreteUnderlyin …
  -/
  dsimp [Condensed.forget, discreteUnderlyingAdj]
  /-
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso ((CategoryTheory.sheafCompose (CategoryTheory.coherentT …
  -/
  rw [← constantSheafAdj_counit_w]
  /-
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.co …
  -/
  refine IsIso.comp_isIso' inferInstance ?_
  have : (constantSheaf (coherentTopology CompHaus) (Type (u + 1))).Faithful :=
    inferInstanceAs (discrete _).Faithful
  have : (constantSheaf (coherentTopology CompHaus) (Type (u + 1))).Full :=
    inferInstanceAs (discrete _).Full
  /-
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHau …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHaus …
    ⊢ CategoryTheory.IsIso ((CategoryTheory.constantSheafAdj (CategoryTheory.coher …
  -/
  rw [← Sheaf.isConstant_iff_isIso_counit_app]
  /-
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHau …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHaus …
    ⊢ CategoryTheory.Sheaf.IsConstant (CategoryTheory.coherentTopology CompHaus) ( …
  -/
  constructor
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHau …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHaus …
    ⊢ Membership.mem (CategoryTheory.constantSheaf (CategoryTheory.coherentTopolog …
  -/
  change _ ∈ (discrete _).essImage
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHau …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHaus …
    ⊢ Membership.mem (Condensed.discrete (Type (u + 1))).essImage ((CategoryTheory …
  -/
  rw [essImage_eq_of_natIso CondensedSet.LocallyConstant.iso.symm]
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type (u + 1)
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHau …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology CompHaus …
    ⊢ Membership.mem CondensedSet.LocallyConstant.functor.essImage ((CategoryTheor …
  -/
  exact obj_mem_essImage CondensedSet.LocallyConstant.functor M
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteComponents (M : ModuleCat R) :
    (discrete _).obj M ≅ (functor R).obj M :=
  have : (Condensed.forget R).ReflectsIsomorphisms :=
    inferInstanceAs (sheafCompose _ _).ReflectsIsomorphisms
  have : IsIso ((discreteUnderlyingAdj (ModuleCat R)).counit.app ((functor R).obj M)) :=
    isIso_of_reflects_iso _ (Condensed.forget R)
  functorIsoDiscreteAux₂ R M ≪≫ asIso ((discreteUnderlyingAdj _).counit.app ((functor R).obj M))


/--
`CondensedMod.LocallyConstant.functor` is naturally isomorphic to the constant sheaf functor from
`R`-modules to condensed `R`-modules.
 -/
noncomputable def functorIsoDiscrete : functor R ≅ discrete _ :=
  NatIso.ofComponents (fun M ↦ (functorIsoDiscreteComponents R M).symm) fun f ↦ by
    /-
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CondensedMod.LocallyConstant.functo …
    -/
    dsimp
    /-
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CondensedMod.LocallyConstant.functo …
    -/
    rw [Iso.eq_inv_comp, ← Category.assoc, Iso.comp_inv_eq]
    /-
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CondensedMod.LocallyConstant.functor …
    -/
    dsimp [functorIsoDiscreteComponents]
    rw [assoc, ← Iso.eq_inv_comp,
      ← (discreteUnderlyingAdj (ModuleCat R)).counit_naturality]
    /-
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Condensed.discrete (ModuleCat R)).m …
    -/
    simp only [← assoc]
    /-
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Condensed.discrete (ModuleCat R)).m …
    -/
    congr 1
    /-
      case e_a
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq ((Condensed.discrete (ModuleCat R)).map ((Condensed.underlying (ModuleCat …
    -/
    rw [← Iso.comp_inv_eq]
    /-
      case e_a
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Condensed.discrete (ModuleCat R)).m …
    -/
    apply Sheaf.hom_ext
    /-
      case e_a.h
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Condensed.discrete (ModuleCat R)).m …
    -/
    simp [functorIsoDiscreteAux₂, ← Functor.map_comp]
    /-
      case e_a.h
      P : TopCat → Prop
      R : Type (u + 1)
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq ((CategoryTheory.presheafToSheaf (CategoryTheory.coherentTopology CompHau …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
`CondensedMod.LocallyConstant.functor` is left adjoint to the forgetful functor from condensed
`R`-modules to `R`-modules.
-/
noncomputable def adjunction : functor R ⊣ underlying (ModuleCat R) :=
  Adjunction.ofNatIsoLeft (discreteUnderlyingAdj _) (functorIsoDiscrete R).symm


/--
`CondensedMod.LocallyConstant.functor` is fully faithful.
-/
noncomputable def fullyFaithfulFunctor : (functor R).FullyFaithful :=
  (adjunction R).fullyFaithfulLOfCompIsoId
     /-
       P : TopCat → Prop
       R : Type (u + 1)
       inst✝ : Ring R
       ⊢ ∀ {X Y : ModuleCat R} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruc …
     -/
    (NatIso.ofComponents fun M ↦ (functorIsoDiscreteAux₁ R _).symm)
     /-
       🎉 no goals
     -/


instance : (functor R).Faithful := (fullyFaithfulFunctor R).faithful


instance : (functor R).Full := (fullyFaithfulFunctor R).full


instance : (discrete (ModuleCat R)).Faithful :=
  Functor.Faithful.of_iso (functorIsoDiscrete R)


instance : (constantSheaf (coherentTopology CompHaus) (ModuleCat R)).Faithful :=
  inferInstanceAs (discrete (ModuleCat R)).Faithful


instance : (discrete (ModuleCat R)).Full :=
  Functor.Full.of_iso (functorIsoDiscrete R)


instance : (constantSheaf (coherentTopology CompHaus) (ModuleCat R)).Full :=
  inferInstanceAs (discrete (ModuleCat R)).Full


instance : (constantSheaf (coherentTopology CompHaus) (Type (u + 1))).Faithful :=
  inferInstanceAs (discrete (Type (u + 1))).Faithful


instance : (constantSheaf (coherentTopology CompHaus) (Type (u + 1))).Full :=
  inferInstanceAs (discrete (Type (u + 1))).Full


/-- `functorToPresheaves` in the case of `LightProfinite`. -/
abbrev functorToPresheaves : ModuleCat.{u} R ⥤ (LightProfinite.{u}ᵒᵖ ⥤ ModuleCat R) :=
  CompHausLike.LocallyConstantModule.functorToPresheaves.{u, u} R


/-- `functorToPresheaves` as a functor to light condensed modules. -/
abbrev functor : ModuleCat R ⥤ LightCondMod.{u} R :=
  CompHausLike.LocallyConstantModule.functor.{u, u} R
    (fun _ _ _ ↦ (LightProfinite.effectiveEpi_iff_surjective _).mp)


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteAux₁ (M : ModuleCat.{u} R) :
    M ≅ (ModuleCat.of R (LocallyConstant (LightProfinite.of PUnit.{u+1}) M)) where
  hom := ModuleCat.ofHom (constₗ R)
  inv := ModuleCat.ofHom (evalₗ R PUnit.unit)


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteAux₂ (M : ModuleCat.{u} R) :
    (discrete _).obj M ≅ (discrete _).obj
      (ModuleCat.of R (LocallyConstant (LightProfinite.of PUnit.{u+1}) M)) :=
  (discrete _).mapIso (functorIsoDiscreteAux₁ R M)

-- Not stating this explicitly causes timeouts below.

instance : HasSheafify (coherentTopology LightProfinite.{u}) (ModuleCat.{u} R) :=
  inferInstance


instance (M : ModuleCat R) :
    IsIso ((LightCondensed.forget R).map
    ((discreteUnderlyingAdj (ModuleCat R)).counit.app
      ((functor R).obj M))) := by
  /-
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso ((LightCondensed.forget R).map ((LightCondensed.discret …
  -/
  dsimp [LightCondensed.forget, discreteUnderlyingAdj]
  /-
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso ((CategoryTheory.sheafCompose (CategoryTheory.coherentT …
  -/
  rw [← constantSheafAdj_counit_w]
  /-
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.co …
  -/
  refine IsIso.comp_isIso' inferInstance ?_
  have : (constantSheaf (coherentTopology LightProfinite) (Type u)).Faithful :=
    inferInstanceAs (discrete _).Faithful
  have : (constantSheaf (coherentTopology LightProfinite) (Type u)).Full :=
    inferInstanceAs (discrete _).Full
  /-
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPr …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPro …
    ⊢ CategoryTheory.IsIso ((CategoryTheory.constantSheafAdj (CategoryTheory.coher …
  -/
  rw [← Sheaf.isConstant_iff_isIso_counit_app]
  /-
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPr …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPro …
    ⊢ CategoryTheory.Sheaf.IsConstant (CategoryTheory.coherentTopology LightProfin …
  -/
  constructor
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPr …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPro …
    ⊢ Membership.mem (CategoryTheory.constantSheaf (CategoryTheory.coherentTopolog …
  -/
  change _ ∈ (discrete _).essImage
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPr …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPro …
    ⊢ Membership.mem (LightCondensed.discrete (Type u)).essImage ((CategoryTheory. …
  -/
  rw [essImage_eq_of_natIso LightCondSet.LocallyConstant.iso.symm]
  /-
    case mem_essImage
    P : TopCat → Prop
    R : Type u
    inst✝ : Ring R
    M : ModuleCat R
    this✝ : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPr …
    this : (CategoryTheory.constantSheaf (CategoryTheory.coherentTopology LightPro …
    ⊢ Membership.mem LightCondSet.LocallyConstant.functor.essImage ((CategoryTheor …
  -/
  exact obj_mem_essImage LightCondSet.LocallyConstant.functor M
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `functorIsoDiscrete`. -/
noncomputable def functorIsoDiscreteComponents (M : ModuleCat R) :
    (discrete _).obj M ≅ (functor R).obj M :=
  have : (LightCondensed.forget R).ReflectsIsomorphisms :=
    inferInstanceAs (sheafCompose _ _).ReflectsIsomorphisms
  have : IsIso ((discreteUnderlyingAdj (ModuleCat R)).counit.app ((functor R).obj M)) :=
    isIso_of_reflects_iso _ (LightCondensed.forget R)
  functorIsoDiscreteAux₂ R M ≪≫ asIso ((discreteUnderlyingAdj _).counit.app ((functor R).obj M))


/--
`LightCondMod.LocallyConstant.functor` is naturally isomorphic to the constant sheaf functor from
`R`-modules to light condensed `R`-modules.
 -/
noncomputable def functorIsoDiscrete : functor R ≅ discrete _ :=
  NatIso.ofComponents (fun M ↦ (functorIsoDiscreteComponents R M).symm) fun f ↦ by
    /-
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondMod.LocallyConstant.functo …
    -/
    dsimp
    /-
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondMod.LocallyConstant.functo …
    -/
    rw [Iso.eq_inv_comp, ← Category.assoc, Iso.comp_inv_eq]
    /-
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (LightCondMod.LocallyConstant.functor …
    -/
    dsimp [functorIsoDiscreteComponents]
    rw [Category.assoc, ← Iso.eq_inv_comp,
      ← (discreteUnderlyingAdj (ModuleCat R)).counit_naturality]
    /-
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondensed.discrete (ModuleCat  …
    -/
    simp only [← assoc]
    /-
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondensed.discrete (ModuleCat  …
    -/
    congr 1
    /-
      case e_a
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq ((LightCondensed.discrete (ModuleCat R)).map ((LightCondensed.underlying  …
    -/
    rw [← Iso.comp_inv_eq]
    /-
      case e_a
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondensed.discrete (ModuleCat  …
    -/
    apply Sheaf.hom_ext
    /-
      case e_a.h
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((LightCondensed.discrete (ModuleCat  …
    -/
    simp [functorIsoDiscreteAux₂, ← Functor.map_comp]
    /-
      case e_a.h
      P : TopCat → Prop
      R : Type u
      inst✝ : Ring R
      X✝ Y✝ : ModuleCat R
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq ((CategoryTheory.presheafToSheaf (CategoryTheory.coherentTopology LightPr …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
`LightCondMod.LocallyConstant.functor` is left adjoint to the forgetful functor from light condensed
`R`-modules to `R`-modules.
 -/
noncomputable def adjunction : functor R ⊣ underlying (ModuleCat R) :=
  Adjunction.ofNatIsoLeft (discreteUnderlyingAdj _) (functorIsoDiscrete R).symm


/--
`LightCondMod.LocallyConstant.functor` is fully faithful.
-/
noncomputable def fullyFaithfulFunctor : (functor R).FullyFaithful :=
  (adjunction R).fullyFaithfulLOfCompIsoId
     /-
       P : TopCat → Prop
       R : Type u
       inst✝ : Ring R
       ⊢ ∀ {X Y : ModuleCat R} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruc …
     -/
    (NatIso.ofComponents fun M ↦ (functorIsoDiscreteAux₁ R _).symm)
     /-
       🎉 no goals
     -/


instance : (discrete.{u} (ModuleCat R)).Faithful := Functor.Faithful.of_iso (functorIsoDiscrete R)


instance : (constantSheaf (coherentTopology LightProfinite.{u}) (ModuleCat.{u} R)).Faithful :=
  inferInstanceAs (discrete.{u} (ModuleCat R)).Faithful


instance : (discrete (ModuleCat.{u} R)).Full :=
  Functor.Full.of_iso (functorIsoDiscrete R)


instance : (constantSheaf (coherentTopology LightProfinite.{u}) (ModuleCat.{u} R)).Full :=
  inferInstanceAs (discrete.{u} (ModuleCat.{u} R)).Full


instance : (constantSheaf (coherentTopology LightProfinite) (Type u)).Faithful :=
  inferInstanceAs (discrete (Type u)).Faithful


instance : (constantSheaf (coherentTopology LightProfinite) (Type u)).Full :=
  inferInstanceAs (discrete (Type u)).Full


