/--
A category `C` is said to have exact colimits of shape `J` provided that colimits of shape `J`
exist and are exact (in the sense that they preserve finite limits).
-/
class HasExactColimitsOfShape (J : Type u') [Category.{v'} J] (C : Type u) [Category.{v} C]
    [HasColimitsOfShape J C]  where
  /-- Exactness of `J`-shaped colimits stated as `colim : (J ⥤ C) ⥤ C` preserving finite limits. -/
  preservesFiniteLimits : PreservesFiniteLimits (colim (J := J) (C := C))


/--
A category `C` is said to have exact limits of shape `J` provided that limits of shape `J`
exist and are exact (in the sense that they preserve finite colimits).
-/
class HasExactLimitsOfShape (J : Type u') [Category.{v'} J] (C : Type u) [Category.{v} C]
    [HasLimitsOfShape J C] where
  /-- Exactness of `J`-shaped limits stated as `lim : (J ⥤ C) ⥤ C` preserving finite colimits. -/
  preservesFiniteColimits : PreservesFiniteColimits (lim (J := J) (C := C))


variable {C} in
/--
Pull back a `HasExactColimitsOfShape J` along a functor which preserves and reflects finite limits
and preserves colimits of shape `J`
-/
lemma HasExactColimitsOfShape.domain_of_functor {D : Type*} (J : Type*) [Category J] [Category D]
    [HasColimitsOfShape J C] [HasColimitsOfShape J D] [HasExactColimitsOfShape J D]
    (F : C ⥤ D) [PreservesFiniteLimits F] [ReflectsFiniteLimits F] [HasFiniteLimits C]
    [PreservesColimitsOfShape J F] : HasExactColimitsOfShape J C where
  preservesFiniteLimits := { preservesFiniteLimits I := { preservesLimit {G} := {
    preserves {c} hc := by
      /-
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} J
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} D
        inst✝⁸ : CategoryTheory.Limits.HasColimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactColimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteLimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteLimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cone G
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.colim.mapCone …
      -/
      constructor
      /-
        case val
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} J
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} D
        inst✝⁸ : CategoryTheory.Limits.HasColimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactColimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteLimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteLimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cone G
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.colim.mapCone c)
      -/
      apply isLimitOfReflects F
      refine (IsLimit.equivOfNatIsoOfIso (isoWhiskerLeft G (preservesColimitNatIso F).symm)
        ((_ ⋙ colim).mapCone c) _ ?_) (isLimitOfPreserves _ hc)
      exact Cones.ext ((preservesColimitNatIso F).symm.app _)
        fun i ↦ (preservesColimitNatIso F).inv.naturality _ } } }


variable {C} in
/--
Pull back a `HasExactLimitsOfShape J` along a functor which preserves and reflects finite colimits
and preserves limits of shape `J`
-/
lemma HasExactLimitsOfShape.domain_of_functor {D : Type*} (J : Type*) [Category D] [Category J]
    [HasLimitsOfShape J C] [HasLimitsOfShape J D] [HasExactLimitsOfShape J D]
    (F : C ⥤ D) [PreservesFiniteColimits F] [ReflectsFiniteColimits F] [HasFiniteColimits C]
    [PreservesLimitsOfShape J F] : HasExactLimitsOfShape J C where
  preservesFiniteColimits := { preservesFiniteColimits I := { preservesColimit {G} := {
    preserves {c} hc := by
      /-
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} D
        inst✝⁹ : CategoryTheory.Category.{u_4, u_2} J
        inst✝⁸ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactLimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteColimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteColimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cocone G
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.lim.mapCoco …
      -/
      constructor
      /-
        case val
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} D
        inst✝⁹ : CategoryTheory.Category.{u_4, u_2} J
        inst✝⁸ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactLimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteColimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteColimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cocone G
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.lim.mapCocone c)
      -/
      apply isColimitOfReflects F
      refine (IsColimit.equivOfNatIsoOfIso (isoWhiskerLeft G (preservesLimitNatIso F).symm)
        ((_ ⋙ lim).mapCocone c) _ ?_) (isColimitOfPreserves _ hc)
      /-
        case val.t
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} D
        inst✝⁹ : CategoryTheory.Category.{u_4, u_2} J
        inst✝⁸ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactLimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteColimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteColimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cocone G
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
      -/
      refine Cocones.ext ((preservesLimitNatIso F).symm.app _) fun i ↦ ?_
      simp only [Functor.comp_obj, lim_obj, Functor.mapCocone_pt, isoWhiskerLeft_inv, Iso.symm_inv,
        Cocones.precompose_obj_pt, whiskeringRight_obj_obj, Functor.const_obj_obj,
        Cocones.precompose_obj_ι, NatTrans.comp_app, whiskerLeft_app, preservesLimitNatIso_hom_app,
        Functor.mapCocone_ι_app, Functor.comp_map, whiskeringRight_obj_map, lim_map, Iso.app_hom,
        Iso.symm_hom, preservesLimitNatIso_inv_app, Category.assoc]
      /-
        case val.t
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} D
        inst✝⁹ : CategoryTheory.Category.{u_4, u_2} J
        inst✝⁸ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactLimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteColimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteColimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cocone G
        hc : CategoryTheory.Limits.IsColimit c
        i : I
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIso F ( …
      -/
      rw [← Iso.eq_inv_comp]
      /-
        case val.t
        C : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        J : Type u_2
        inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} D
        inst✝⁹ : CategoryTheory.Category.{u_4, u_2} J
        inst✝⁸ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝⁷ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝⁶ : CategoryTheory.HasExactLimitsOfShape J D
        F : CategoryTheory.Functor C D
        inst✝⁵ : CategoryTheory.Limits.PreservesFiniteColimits F
        inst✝⁴ : CategoryTheory.Limits.ReflectsFiniteColimits F
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J F
        I : Type
        inst✝¹ : CategoryTheory.SmallCategory I
        inst✝ : CategoryTheory.FinCategory I
        G : CategoryTheory.Functor I (CategoryTheory.Functor J C)
        c : CategoryTheory.Limits.Cocone G
        hc : CategoryTheory.Limits.IsColimit c
        i : I
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
      -/
      exact (preservesLimitNatIso F).inv.naturality _ } } }
      /-
        🎉 no goals
      -/


/--
Transport a `HasExactColimitsOfShape` along an equivalence of the shape.

Note: When `C` has finite limits, this lemma holds with the equivalence replaced by a final
functor, see `hasExactColimitsOfShape_of_final` below.
-/
lemma HasExactColimitsOfShape.of_domain_equivalence {J J' : Type*} [Category J] [Category J']
    (e : J ≌ J') [HasColimitsOfShape J C] [HasExactColimitsOfShape J C] :
    haveI : HasColimitsOfShape J' C := hasColimitsOfShape_of_equivalence e
    HasExactColimitsOfShape J' C :=
  haveI : HasColimitsOfShape J' C := hasColimitsOfShape_of_equivalence e
  ⟨preservesFiniteLimits_of_natIso (Functor.Final.colimIso e.functor)⟩


variable {C} in
lemma HasExactColimitsOfShape.of_codomain_equivalence (J : Type*) [Category J] {D : Type*}
    [Category D] (e : C ≌ D) [HasColimitsOfShape J C] [HasExactColimitsOfShape J C] :
    haveI : HasColimitsOfShape J D := Adjunction.hasColimitsOfShape_of_equivalence e.inverse
    HasExactColimitsOfShape J D := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.HasExactColimitsOfShape J C
    ⊢ CategoryTheory.HasExactColimitsOfShape J D
  -/
  haveI : HasColimitsOfShape J D := Adjunction.hasColimitsOfShape_of_equivalence e.inverse
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.HasExactColimitsOfShape J C
    this : CategoryTheory.Limits.HasColimitsOfShape J D
    ⊢ CategoryTheory.HasExactColimitsOfShape J D
  -/
  refine ⟨⟨fun _ _ _ => ⟨@fun K => ?_⟩⟩⟩
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.HasExactColimitsOfShape J C
    this : CategoryTheory.Limits.HasColimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Limits.PreservesLimit K CategoryTheory.Limits.colim
  -/
  refine preservesLimit_of_natIso K (?_ : e.congrRight.inverse ⋙ colim ⋙ e.functor ≅ colim)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.HasExactColimitsOfShape J C
    this : CategoryTheory.Limits.HasColimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Iso (e.congrRight.inverse.comp (CategoryTheory.Limits.colim.c …
  -/
  apply e.symm.congrRight.fullyFaithfulFunctor.preimageIso
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.HasExactColimitsOfShape J C
    this : CategoryTheory.Limits.HasColimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Iso (e.symm.congrRight.functor.obj (e.congrRight.inverse.comp …
  -/
  exact isoWhiskerLeft (_ ⋙ colim) e.unitIso.symm ≪≫ (preservesColimitNatIso e.inverse).symm
  /-
    🎉 no goals
  -/


/--
Transport a `HasExactLimitsOfShape` along an equivalence of the shape.

Note: When `C` has finite colimits, this lemma holds with the equivalence replaced by a initial
functor, see `hasExactLimitsOfShape_of_initial` below.
-/
lemma HasExactLimitsOfShape.of_domain_equivalence {J J' : Type*} [Category J] [Category J']
    (e : J ≌ J') [HasLimitsOfShape J C] [HasExactLimitsOfShape J C] :
    haveI : HasLimitsOfShape J' C := hasLimitsOfShape_of_equivalence e
    HasExactLimitsOfShape J' C :=
  haveI : HasLimitsOfShape J' C := hasLimitsOfShape_of_equivalence e
  ⟨preservesFiniteColimits_of_natIso (Functor.Initial.limIso e.functor)⟩


variable {C} in
lemma HasExactLimitsOfShape.of_codomain_equivalence (J : Type*) [Category J] {D : Type*}
    [Category D] (e : C ≌ D) [HasLimitsOfShape J C] [HasExactLimitsOfShape J C] :
    haveI : HasLimitsOfShape J D := Adjunction.hasLimitsOfShape_of_equivalence e.inverse
    HasExactLimitsOfShape J D := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.HasExactLimitsOfShape J C
    ⊢ CategoryTheory.HasExactLimitsOfShape J D
  -/
  haveI : HasLimitsOfShape J D := Adjunction.hasLimitsOfShape_of_equivalence e.inverse
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.HasExactLimitsOfShape J C
    this : CategoryTheory.Limits.HasLimitsOfShape J D
    ⊢ CategoryTheory.HasExactLimitsOfShape J D
  -/
  refine ⟨⟨fun _ _ _ => ⟨@fun K => ?_⟩⟩⟩
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.HasExactLimitsOfShape J C
    this : CategoryTheory.Limits.HasLimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Limits.PreservesColimit K CategoryTheory.Limits.lim
  -/
  refine preservesColimit_of_natIso K (?_ : e.congrRight.inverse ⋙ lim ⋙ e.functor ≅ lim)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.HasExactLimitsOfShape J C
    this : CategoryTheory.Limits.HasLimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Iso (e.congrRight.inverse.comp (CategoryTheory.Limits.lim.com …
  -/
  apply e.symm.congrRight.fullyFaithfulFunctor.preimageIso
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} J
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.HasExactLimitsOfShape J C
    this : CategoryTheory.Limits.HasLimitsOfShape J D
    x✝² : Type
    x✝¹ : CategoryTheory.SmallCategory x✝²
    x✝ : CategoryTheory.FinCategory x✝²
    K : CategoryTheory.Functor x✝² (CategoryTheory.Functor J D)
    ⊢ CategoryTheory.Iso (e.symm.congrRight.functor.obj (e.congrRight.inverse.comp …
  -/
  exact isoWhiskerLeft (_ ⋙ lim) e.unitIso.symm ≪≫ (preservesLimitNatIso e.inverse).symm
  /-
    🎉 no goals
  -/


/-- Let `adj : F ⊣ G` be an adjunction, with `G : D ⥤ C` reflective.
Assume that `D` has finite limits and `F` commutes to them.
If `C` has exact colimits of shape `J`, then `D` also has exact colimits of shape `J`. -/
lemma hasExactColimitsOfShape (adj : F ⊣ G) [G.Full] [G.Faithful]
    (J : Type u') [Category.{v'} J] [HasColimitsOfShape J C] [HasColimitsOfShape J D]
    [HasExactColimitsOfShape J C] [HasFiniteLimits D] [PreservesFiniteLimits F] :
    HasExactColimitsOfShape J D where
  preservesFiniteLimits := ⟨fun K _ _ ↦ ⟨fun {H} ↦ by
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : G.Full
      inst✝⁶ : G.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J D
      inst✝² : CategoryTheory.HasExactColimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits D
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J D)
      ⊢ CategoryTheory.Limits.PreservesLimit H CategoryTheory.Limits.colim
    -/
    have : PreservesLimitsOfSize.{0, 0} G := adj.rightAdjoint_preservesLimits
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : G.Full
      inst✝⁶ : G.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J D
      inst✝² : CategoryTheory.HasExactColimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits D
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J D)
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v'', v, u'', u} G
      ⊢ CategoryTheory.Limits.PreservesLimit H CategoryTheory.Limits.colim
    -/
    have : PreservesColimitsOfSize.{v', u'} F := adj.leftAdjoint_preservesColimits
    let e : (whiskeringRight J D C).obj G ⋙ colim ⋙ F ≅ colim :=
      isoWhiskerLeft _ (preservesColimitNatIso F) ≪≫ (Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (whiskeringRightObjCompIso G F) _ ≪≫
        isoWhiskerRight ((whiskeringRight J D D).mapIso (asIso adj.counit)) _ ≪≫
        isoWhiskerRight wiskeringRightObjIdIso _ ≪≫ colim.leftUnitor
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : G.Full
      inst✝⁶ : G.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J D
      inst✝² : CategoryTheory.HasExactColimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits D
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J D)
      this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v'', v, u'', u} G
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{v', u', v, v'', u, u''} F
      e : CategoryTheory.Iso (((CategoryTheory.whiskeringRight J D C).obj G).comp (C …
      ⊢ CategoryTheory.Limits.PreservesLimit H CategoryTheory.Limits.colim
    -/
    exact preservesLimit_of_natIso _ e⟩⟩
    /-
      🎉 no goals
    -/


/-- Let `adj : F ⊣ G` be an adjunction, with `F : C ⥤ D` coreflective.
Assume that `C` has finite colimits and `G` commutes to them.
If `D` has exact limits of shape `J`, then `C` also has exact limits of shape `J`. -/
lemma hasExactLimitsOfShape (adj : F ⊣ G) [F.Full] [F.Faithful]
    (J : Type u') [Category.{v'} J] [HasLimitsOfShape J C] [HasLimitsOfShape J D]
    [HasExactLimitsOfShape J D] [HasFiniteColimits C] [PreservesFiniteColimits G] :
    HasExactLimitsOfShape J C where
  preservesFiniteColimits:= ⟨fun K _ _ ↦ ⟨fun {H} ↦ by
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : F.Full
      inst✝⁶ : F.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J D
      inst✝² : CategoryTheory.HasExactLimitsOfShape J D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits G
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J C)
      ⊢ CategoryTheory.Limits.PreservesColimit H CategoryTheory.Limits.lim
    -/
    have : PreservesLimitsOfSize.{v', u'} G := adj.rightAdjoint_preservesLimits
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : F.Full
      inst✝⁶ : F.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J D
      inst✝² : CategoryTheory.HasExactLimitsOfShape J D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits G
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J C)
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{v', u', v'', v, u'', u} G
      ⊢ CategoryTheory.Limits.PreservesColimit H CategoryTheory.Limits.lim
    -/
    have : PreservesColimitsOfSize.{0, 0} F := adj.leftAdjoint_preservesColimits
    let e : (whiskeringRight J _ _).obj F ⋙ lim ⋙ G ≅ lim :=
      isoWhiskerLeft _ (preservesLimitNatIso G) ≪≫
        (Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (whiskeringRightObjCompIso F G) _ ≪≫
        isoWhiskerRight ((whiskeringRight J C C).mapIso (asIso adj.unit).symm) _ ≪≫
        isoWhiskerRight wiskeringRightObjIdIso _ ≪≫ lim.leftUnitor
    /-
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁷ : F.Full
      inst✝⁶ : F.Faithful
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J D
      inst✝² : CategoryTheory.HasExactLimitsOfShape J D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits G
      K : Type
      x✝¹ : CategoryTheory.SmallCategory K
      x✝ : CategoryTheory.FinCategory K
      H : CategoryTheory.Functor K (CategoryTheory.Functor J C)
      this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{v', u', v'', v, u'', u} G
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v'', u, u''} F
      e : CategoryTheory.Iso (((CategoryTheory.whiskeringRight J C D).obj F).comp (C …
      ⊢ CategoryTheory.Limits.PreservesColimit H CategoryTheory.Limits.lim
    -/
    exact preservesColimit_of_natIso _ e⟩⟩
    /-
      🎉 no goals
    -/


/--
A category `C` which has coproducts is said to have `AB4` of size `w` provided that
coproducts of size `w` are exact.
-/
@[pp_with_univ]
class AB4OfSize [HasCoproducts.{w} C] where
  ofShape (α : Type w) : HasExactColimitsOfShape (Discrete α) C


/--
A category `C` which has coproducts is said to have `AB4` provided that
coproducts are exact.
-/
@[stacks 079B]
abbrev AB4 [HasCoproducts C] := AB4OfSize.{v} C


lemma AB4OfSize_shrink [HasCoproducts.{max w w'} C] [AB4OfSize.{max w w'} C] :
    haveI : HasCoproducts.{w} C := hasCoproducts_shrink.{w, w'}
    AB4OfSize.{w} C :=
  haveI := hasCoproducts_shrink.{w, w'} (C := C)
  ⟨fun J ↦ HasExactColimitsOfShape.of_domain_equivalence C
    (Discrete.equivalence Equiv.ulift : Discrete (ULift.{w'} J) ≌ _)⟩


instance (priority := 100) [HasCoproducts.{w} C] [AB4OfSize.{w} C] :
    haveI : HasCoproducts.{0} C := hasCoproducts_shrink
    AB4OfSize.{0} C := AB4OfSize_shrink C


/-- A category `C` which has products is said to have `AB4Star` (in literature `AB4*`)
provided that products are exact. -/
@[pp_with_univ, stacks 079B]
class AB4StarOfSize [HasProducts.{w} C] where
  ofShape (α : Type w) : HasExactLimitsOfShape (Discrete α) C


/-- A category `C` which has products is said to have `AB4Star` (in literature `AB4*`)
provided that products are exact. -/
abbrev AB4Star [HasProducts C] := AB4StarOfSize.{v} C


lemma AB4StarOfSize_shrink [HasProducts.{max w w'} C] [AB4StarOfSize.{max w w'} C] :
    haveI : HasProducts.{w} C := hasProducts_shrink.{w, w'}
    AB4StarOfSize.{w} C :=
  haveI := hasProducts_shrink.{w, w'} (C := C)
  ⟨fun J ↦ HasExactLimitsOfShape.of_domain_equivalence C
    (Discrete.equivalence Equiv.ulift : Discrete (ULift.{w'} J) ≌ _)⟩


instance (priority := 100) [HasProducts.{w} C] [AB4StarOfSize.{w} C] :
    haveI : HasProducts.{0} C := hasProducts_shrink
    AB4StarOfSize.{0} C := AB4StarOfSize_shrink C


/--
A category `C` which has countable coproducts is said to have countable `AB4` provided that
countable coproducts are exact.
-/
class CountableAB4 [HasCountableCoproducts C] where
  ofShape (α : Type) [Countable α] : HasExactColimitsOfShape (Discrete α) C


instance (priority := 100) [HasCoproducts.{0} C] [AB4OfSize.{0} C] : CountableAB4 C :=
  ⟨inferInstance⟩


/--
A category `C` which has countable coproducts is said to have countable `AB4Star` provided that
countable products are exact.
-/
class CountableAB4Star [HasCountableProducts C] where
  ofShape (α : Type) [Countable α] : HasExactLimitsOfShape (Discrete α) C


instance (priority := 100) [HasProducts.{0} C] [AB4StarOfSize.{0} C] : CountableAB4Star C :=
  ⟨inferInstance⟩


/--
A category `C` which has filtered colimits of a given size is said to have `AB5` of that size
provided that these filtered colimits are exact.

`AB5OfSize.{w, w'} C` means that `C` has exact colimits of shape `J : Type w'` with
`Category.{w} J` such that `J` is filtered.
-/
@[pp_with_univ]
class AB5OfSize [HasFilteredColimitsOfSize.{w, w'} C] where
  ofShape (J : Type w') [Category.{w} J] [IsFiltered J] : HasExactColimitsOfShape J C


/--
A category `C` which has filtered colimits is said to have `AB5` provided that
filtered colimits are exact.
-/
@[stacks 079B]
abbrev AB5 [HasFilteredColimits C] := AB5OfSize.{v, v} C


lemma AB5OfSize_of_univLE [HasFilteredColimitsOfSize.{w₂, w₂'} C] [UnivLE.{w, w₂}]
    [UnivLE.{w', w₂'}] [AB5OfSize.{w₂, w₂'} C] :
    haveI : HasFilteredColimitsOfSize.{w, w'} C := hasFilteredColimitsOfSize_of_univLE.{w}
    AB5OfSize.{w, w'} C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5OfSize.{w₂, w₂', v, u} C
    ⊢ CategoryTheory.AB5OfSize.{w, w', v, u} C
  -/
  haveI : HasFilteredColimitsOfSize.{w, w'} C := hasFilteredColimitsOfSize_of_univLE.{w}
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5OfSize.{w₂, w₂', v, u} C
    this : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w', v, u} C
    ⊢ CategoryTheory.AB5OfSize.{w, w', v, u} C
  -/
  constructor
  /-
    case ofShape
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5OfSize.{w₂, w₂', v, u} C
    this : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w', v, u} C
    ⊢ ∀ (J : Type w') [inst : CategoryTheory.Category.{w, w'} J] [inst_1 : Categor …
  -/
  intro J _ _
  haveI := IsFiltered.of_equivalence ((ShrinkHoms.equivalence.{w₂} J).trans <|
    Shrink.equivalence.{w₂'} (ShrinkHoms.{w'} J))
  exact HasExactColimitsOfShape.of_domain_equivalence _ ((ShrinkHoms.equivalence.{w₂} J).trans <|
    Shrink.equivalence.{w₂'} (ShrinkHoms.{w'} J)).symm


lemma AB5OfSize_shrink [HasFilteredColimitsOfSize.{max w w₂, max w' w₂'} C]
    [AB5OfSize.{max w w₂, max w' w₂'} C] :
    haveI : HasFilteredColimitsOfSize.{w, w'} C := hasFilteredColimitsOfSize_shrink
    AB5OfSize.{w, w'} C :=
  AB5OfSize_of_univLE C


/--
A category `C` which has cofiltered limits is said to have `AB5Star` (in literature `AB5*`)
provided that cofiltered limits are exact.
-/
@[pp_with_univ, stacks 079B]
class AB5StarOfSize [HasCofilteredLimitsOfSize.{w, w'} C] where
  ofShape (J : Type w') [Category.{w} J] [IsCofiltered J] : HasExactLimitsOfShape J C


/--
A category `C` which has cofiltered limits is said to have `AB5Star` (in literature `AB5*`)
provided that cofiltered limits are exact.
-/
abbrev AB5Star [HasCofilteredLimits C] := AB5StarOfSize.{v, v} C


lemma AB5StarOfSize_of_univLE [HasCofilteredLimitsOfSize.{w₂, w₂'} C] [UnivLE.{w, w₂}]
    [UnivLE.{w', w₂'}] [AB5StarOfSize.{w₂, w₂'} C] :
    haveI : HasCofilteredLimitsOfSize.{w, w'} C := hasCofilteredLimitsOfSize_of_univLE.{w}
    AB5StarOfSize.{w, w'} C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasCofilteredLimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5StarOfSize.{w₂, w₂', v, u} C
    ⊢ CategoryTheory.AB5StarOfSize.{w, w', v, u} C
  -/
  haveI : HasCofilteredLimitsOfSize.{w, w'} C := hasCofilteredLimitsOfSize_of_univLE.{w}
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasCofilteredLimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5StarOfSize.{w₂, w₂', v, u} C
    this : CategoryTheory.Limits.HasCofilteredLimitsOfSize.{w, w', v, u} C
    ⊢ CategoryTheory.AB5StarOfSize.{w, w', v, u} C
  -/
  constructor
  /-
    case ofShape
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasCofilteredLimitsOfSize.{w₂, w₂', v, u} C
    inst✝² : UnivLE.{w, w₂}
    inst✝¹ : UnivLE.{w', w₂'}
    inst✝ : CategoryTheory.AB5StarOfSize.{w₂, w₂', v, u} C
    this : CategoryTheory.Limits.HasCofilteredLimitsOfSize.{w, w', v, u} C
    ⊢ ∀ (J : Type w') [inst : CategoryTheory.Category.{w, w'} J] [inst_1 : Categor …
  -/
  intro J _ _
  haveI := IsCofiltered.of_equivalence ((ShrinkHoms.equivalence.{w₂} J).trans <|
    Shrink.equivalence.{w₂'} (ShrinkHoms.{w'} J))
  exact HasExactLimitsOfShape.of_domain_equivalence _ ((ShrinkHoms.equivalence.{w₂} J).trans <|
    Shrink.equivalence.{w₂'} (ShrinkHoms.{w'} J)).symm


lemma AB5StarOfSize_shrink [HasCofilteredLimitsOfSize.{max w w₂, max w' w₂'} C]
    [AB5StarOfSize.{max w w₂, max w' w₂'} C] :
    haveI : HasCofilteredLimitsOfSize.{w, w'} C := hasCofilteredLimitsOfSize_shrink
    AB5StarOfSize.{w, w'} C :=
  AB5StarOfSize_of_univLE C


/-- `HasExactColimitsOfShape` can be "pushed forward" along final functors -/
lemma hasExactColimitsOfShape_of_final [HasFiniteLimits C] {J J' : Type*} [Category J] [Category J']
    (F : J ⥤ J') [F.Final] [HasColimitsOfShape J' C] [HasColimitsOfShape J C]
    [HasExactColimitsOfShape J C] : HasExactColimitsOfShape J' C where
  preservesFiniteLimits :=
    letI : PreservesFiniteLimits ((whiskeringLeft J J' C).obj F) := ⟨fun _ ↦ inferInstance⟩
    letI := comp_preservesFiniteLimits ((whiskeringLeft J J' C).obj F) colim
    preservesFiniteLimits_of_natIso (Functor.Final.colimIso F)


/-- `HasExactLimitsOfShape` can be "pushed forward" along initial functors -/
lemma hasExactLimitsOfShape_of_initial [HasFiniteColimits C] {J J' : Type*} [Category J]
    [Category J'] (F : J ⥤ J') [F.Initial]  [HasLimitsOfShape J' C] [HasLimitsOfShape J C]
    [HasExactLimitsOfShape J C] : HasExactLimitsOfShape J' C where
  preservesFiniteColimits :=
    letI : PreservesFiniteColimits ((whiskeringLeft J J' C).obj F) := ⟨fun _ ↦ inferInstance⟩
    letI := comp_preservesFiniteColimits ((whiskeringLeft J J' C).obj F) lim
    preservesFiniteColimits_of_natIso (Functor.Initial.limIso F)


instance preservesFiniteLimits_liftToFinset : PreservesFiniteLimits (liftToFinset C α) :=
  preservesFiniteLimits_of_evaluation _ fun I =>
    letI : PreservesFiniteLimits (colim (J := Discrete I) (C := C)) :=
      preservesFiniteLimits_of_natIso HasBiproductsOfShape.colimIsoLim.symm
    letI : PreservesFiniteLimits ((whiskeringLeft (Discrete I) (Discrete α) C).obj
        (Discrete.functor fun x ↦ ↑x)) :=
      ⟨fun J _ _ => whiskeringLeft_preservesLimitsOfShape J _⟩
    letI : PreservesFiniteLimits ((whiskeringLeft (Discrete I) (Discrete α) C).obj
        (Discrete.functor (·.val)) ⋙ colim) :=
      comp_preservesFiniteLimits _ _
    preservesFiniteLimits_of_natIso (liftToFinsetEvaluationIso I).symm


/--
`HasExactColimitsOfShape (Finset (Discrete J)) C` implies `HasExactColimitsOfShape (Discrete J) C`
-/
lemma hasExactColimitsOfShape_discrete_of_hasExactColimitsOfShape_finset_discrete
    [HasColimitsOfShape (Discrete J) C] [HasColimitsOfShape (Finset (Discrete J)) C]
    [HasExactColimitsOfShape (Finset (Discrete J)) C] : HasExactColimitsOfShape (Discrete J) C where
  preservesFiniteLimits :=
    letI : PreservesFiniteLimits (liftToFinset C J ⋙ colim) :=
      comp_preservesFiniteLimits _ _
    preservesFiniteLimits_of_natIso (liftToFinsetColimIso)


attribute [local instance] hasCoproducts_of_finite_and_filtered in
/-- A category with finite biproducts and finite limits is AB4 if it is AB5. -/
lemma AB4.of_AB5 [HasFilteredColimitsOfSize.{w, w} C]
    [AB5OfSize.{w, w} C] : AB4OfSize.{w} C where
  ofShape _ := hasExactColimitsOfShape_discrete_of_hasExactColimitsOfShape_finset_discrete _ _


/--
A category with finite biproducts and finite limits has countable AB4 if sequential colimits are
exact.
-/
lemma CountableAB4.of_countableAB5 [HasColimitsOfShape ℕ C] [HasExactColimitsOfShape ℕ C]
    [HasCountableCoproducts C] : CountableAB4 C where
  ofShape J :=
    have : HasColimitsOfShape (Finset (Discrete J)) C :=
      Functor.Final.hasColimitsOfShape_of_final
        (IsFiltered.sequentialFunctor (Finset (Discrete J)))
    have := hasExactColimitsOfShape_of_final C (IsFiltered.sequentialFunctor (Finset (Discrete J)))
    hasExactColimitsOfShape_discrete_of_hasExactColimitsOfShape_finset_discrete _ _


instance preservesFiniteColimits_liftToFinset : PreservesFiniteColimits (liftToFinset C α) :=
  preservesFiniteColimits_of_evaluation _ fun ⟨I⟩ =>
    letI : PreservesFiniteColimits (lim (J := Discrete I) (C := C)) :=
      preservesFiniteColimits_of_natIso HasBiproductsOfShape.colimIsoLim
    letI : PreservesFiniteColimits ((whiskeringLeft (Discrete I) (Discrete α) C).obj
        (Discrete.functor fun x ↦ ↑x)) := ⟨fun _ _ _ => inferInstance⟩
    letI : PreservesFiniteColimits ((whiskeringLeft (Discrete I) (Discrete α) C).obj
        (Discrete.functor (·.val)) ⋙ lim) :=
      comp_preservesFiniteColimits _ _
    preservesFiniteColimits_of_natIso (liftToFinsetEvaluationIso _ _ I).symm


/--
`HasExactLimitsOfShape (Finset (Discrete J))ᵒᵖ C` implies  `HasExactLimitsOfShape (Discrete J) C`
-/
lemma hasExactLimitsOfShape_discrete_of_hasExactLimitsOfShape_finset_discrete_op
    [HasLimitsOfShape (Discrete J) C] [HasLimitsOfShape (Finset (Discrete J))ᵒᵖ C]
    [HasExactLimitsOfShape (Finset (Discrete J))ᵒᵖ C] :
    HasExactLimitsOfShape (Discrete J) C where
  preservesFiniteColimits :=
    letI : PreservesFiniteColimits (ProductsFromFiniteCofiltered.liftToFinset C J ⋙ lim) :=
      comp_preservesFiniteColimits _ _
    preservesFiniteColimits_of_natIso (ProductsFromFiniteCofiltered.liftToFinsetLimIso _ _)


attribute [local instance] hasProducts_of_finite_and_cofiltered in
/-- A category with finite biproducts and finite limits is AB4 if it is AB5. -/
lemma AB4Star.of_AB5Star [HasCofilteredLimitsOfSize.{w, w} C] [AB5StarOfSize.{w, w} C] :
    AB4StarOfSize.{w} C where
  ofShape _ := hasExactLimitsOfShape_discrete_of_hasExactLimitsOfShape_finset_discrete_op _ _


/--
A category with finite biproducts and finite limits has countable AB4* if sequential limits are
exact.
-/
lemma CountableAB4Star.of_countableAB5Star [HasLimitsOfShape ℕᵒᵖ C] [HasExactLimitsOfShape ℕᵒᵖ C]
    [HasCountableProducts C] : CountableAB4Star C where
  ofShape J :=
    have : HasLimitsOfShape (Finset (Discrete J))ᵒᵖ C :=
      Functor.Initial.hasLimitsOfShape_of_initial
        (IsFiltered.sequentialFunctor (Finset (Discrete J))).op
    have := hasExactLimitsOfShape_of_initial C
      (IsFiltered.sequentialFunctor (Finset (Discrete J))).op
    hasExactLimitsOfShape_discrete_of_hasExactLimitsOfShape_finset_discrete_op _ _


/--
Checking exactness of colimits of shape `Discrete ℕ` and `Discrete J` for finite `J` is enough for
countable AB4.
-/
lemma CountableAB4.of_hasExactColimitsOfShape_nat_and_finite [HasCountableCoproducts C]
    [HasFiniteLimits C] [∀ (J : Type) [Finite J], HasExactColimitsOfShape (Discrete J) C]
    [HasExactColimitsOfShape (Discrete ℕ) C] :
    CountableAB4 C where
  ofShape J := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasCountableCoproducts C
      inst✝² : CategoryTheory.Limits.HasFiniteLimits C
      inst✝¹ : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
      inst✝ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
      J : Type
      ⊢ ∀ [inst : Countable J], CategoryTheory.HasExactColimitsOfShape (CategoryTheo …
    -/
    by_cases h : Finite J
      /-
        case pos
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableCoproducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
        inst✝¹ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Finite J
        ⊢ CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableCoproducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
        inst✝¹ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        ⊢ CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · have : Infinite J := ⟨h⟩
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableCoproducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
        inst✝¹ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        ⊢ CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete J) C
      -/
      let _ := Encodable.ofCountable J
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableCoproducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
        inst✝¹ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        x✝ : Encodable J := Encodable.ofCountable J
        ⊢ CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete J) C
      -/
      let _ := Denumerable.ofEncodableOfInfinite J
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableCoproducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShap …
        inst✝¹ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        x✝¹ : Encodable J := Encodable.ofCountable J
        x✝ : Denumerable J := Denumerable.ofEncodableOfInfinite J
        ⊢ CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete J) C
      -/
      exact hasExactColimitsOfShape_of_final C (Discrete.equivalence (Denumerable.eqv J)).inverse
      /-
        🎉 no goals
      -/


/--
Checking exactness of limits of shape `Discrete ℕ` and `Discrete J` for finite `J` is enough for
countable AB4*.
-/
lemma CountableAB4Star.of_hasExactLimitsOfShape_nat_and_finite [HasCountableProducts C]
    [HasFiniteColimits C] [∀ (J : Type) [Finite J], HasExactLimitsOfShape (Discrete J) C]
    [HasExactLimitsOfShape (Discrete ℕ) C] :
    CountableAB4Star C where
  ofShape J := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasCountableProducts C
      inst✝² : CategoryTheory.Limits.HasFiniteColimits C
      inst✝¹ : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
      inst✝ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
      J : Type
      ⊢ ∀ [inst : Countable J], CategoryTheory.HasExactLimitsOfShape (CategoryTheory …
    -/
    by_cases h : Finite J
      /-
        case pos
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableProducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
        inst✝¹ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Finite J
        ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableProducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
        inst✝¹ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · have : Infinite J := ⟨h⟩
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableProducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
        inst✝¹ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) C
      -/
      let _ := Encodable.ofCountable J
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableProducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
        inst✝¹ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        x✝ : Encodable J := Encodable.ofCountable J
        ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) C
      -/
      let _ := Denumerable.ofEncodableOfInfinite J
      /-
        case neg
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        inst✝⁴ : CategoryTheory.Limits.HasCountableProducts C
        inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝² : ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape  …
        inst✝¹ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
        J : Type
        inst✝ : Countable J
        h : Not (Finite J)
        this : Infinite J
        x✝¹ : Encodable J := Encodable.ofCountable J
        x✝ : Denumerable J := Denumerable.ofEncodableOfInfinite J
        ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) C
      -/
      exact hasExactLimitsOfShape_of_initial C (Discrete.equivalence (Denumerable.eqv J)).inverse
      /-
        🎉 no goals
      -/


noncomputable instance hasExactColimitsOfShape_discrete_finite (J : Type*) [Finite J] :
    HasExactColimitsOfShape (Discrete J) C where
  preservesFiniteLimits := preservesFiniteLimits_of_natIso HasBiproductsOfShape.colimIsoLim.symm


noncomputable instance hasExactLimitsOfShape_discrete_finite {J : Type*} [Finite J] :
    HasExactLimitsOfShape (Discrete J) C where
  preservesFiniteColimits := preservesFiniteColimits_of_natIso HasBiproductsOfShape.colimIsoLim


/--
Checking AB of shape `Discrete ℕ` is enough for countable AB4, provided that the category has
finite biproducts and finite limits.
-/
lemma CountableAB4.of_hasExactColimitsOfShape_nat [HasFiniteLimits C] [HasCountableCoproducts C]
    [HasExactColimitsOfShape (Discrete ℕ) C] : CountableAB4 C := by
  apply (config := { allowSynthFailures := true })
      CountableAB4.of_hasExactColimitsOfShape_nat_and_finite
  /-
    case inst
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝² : CategoryTheory.Limits.HasFiniteLimits C
    inst✝¹ : CategoryTheory.Limits.HasCountableCoproducts C
    inst✝ : CategoryTheory.HasExactColimitsOfShape (CategoryTheory.Discrete Nat) C
    ⊢ ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactColimitsOfShape (Cate …
  -/
  exact fun _ ↦ inferInstance
  /-
    🎉 no goals
  -/


/--
Checking AB* of shape `Discrete ℕ` is enough for countable AB4*, provided that the category has
finite biproducts and finite colimits.
-/
lemma CountableAB4Star.of_hasExactLimitsOfShape_nat [HasFiniteColimits C]
    [HasCountableProducts C] [HasExactLimitsOfShape (Discrete ℕ) C] : CountableAB4Star C := by
  apply (config := { allowSynthFailures := true })
      CountableAB4Star.of_hasExactLimitsOfShape_nat_and_finite
  /-
    case inst
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝² : CategoryTheory.Limits.HasFiniteColimits C
    inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
    inst✝ : CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete Nat) C
    ⊢ ∀ (J : Type) [inst : Finite J], CategoryTheory.HasExactLimitsOfShape (Catego …
  -/
  exact fun _ ↦ inferInstance
  /-
    🎉 no goals
  -/


/--
If `colim` of shape `J` into an abelian category `C` preserves monomorphisms, then `C` has AB of
shape `J`.
-/
lemma hasExactColimitsOfShape_of_preservesMono [HasColimitsOfShape J C]
    [PreservesMonomorphisms (colim (J := J) (C := C))] : HasExactColimitsOfShape J C where
  preservesFiniteLimits := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Abelian C
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
      inst✝ : CategoryTheory.Limits.colim.PreservesMonomorphisms
      ⊢ CategoryTheory.Limits.PreservesFiniteLimits CategoryTheory.Limits.colim
    -/
    apply (config := { allowSynthFailures := true }) preservesFiniteLimits_of_preservesHomology
      /-
        case inst
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Abelian C
        J : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} J
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
        inst✝ : CategoryTheory.Limits.colim.PreservesMonomorphisms
        ⊢ CategoryTheory.Limits.colim.PreservesHomology
      -/
    · exact preservesHomology_of_preservesMonos_and_cokernels _
      /-
        🎉 no goals
      -/
      /-
        case inst
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Abelian C
        J : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} J
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
        inst✝ : CategoryTheory.Limits.colim.PreservesMonomorphisms
        ⊢ CategoryTheory.Limits.colim.Additive
      -/
    · exact additive_of_preservesBinaryBiproducts _
      /-
        🎉 no goals
      -/


/--
If `lim` of shape `J` into an abelian category `C` preserves epimorphisms, then `C` has AB* of
shape `J`.
-/
lemma hasExactLimitsOfShape_of_preservesEpi [HasLimitsOfShape J C]
    [PreservesEpimorphisms (lim (J := J) (C := C))] : HasExactLimitsOfShape J C where
  preservesFiniteColimits := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Abelian C
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝ : CategoryTheory.Limits.lim.PreservesEpimorphisms
      ⊢ CategoryTheory.Limits.PreservesFiniteColimits CategoryTheory.Limits.lim
    -/
    apply (config := { allowSynthFailures := true }) preservesFiniteColimits_of_preservesHomology
      /-
        case inst
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Abelian C
        J : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} J
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.Limits.lim.PreservesEpimorphisms
        ⊢ CategoryTheory.Limits.lim.PreservesHomology
      -/
    · exact preservesHomology_of_preservesEpis_and_kernels _
      /-
        🎉 no goals
      -/
      /-
        case inst
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Abelian C
        J : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} J
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.Limits.lim.PreservesEpimorphisms
        ⊢ CategoryTheory.Limits.lim.Additive
      -/
    · exact additive_of_preservesBinaryBiproducts _
      /-
        🎉 no goals
      -/


