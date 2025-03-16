/-- The constant presheaf functor is left adjoint to evaluation at a terminal object. -/
@[simps! unit_app counit_app_app]
noncomputable def constantPresheafAdj {T : C} (hT : IsTerminal T) :
    Functor.const Cᵒᵖ ⊣ (evaluation Cᵒᵖ D).obj (op T) where
  unit := (Functor.constCompEvaluationObj D (op T)).hom
  counit := {
    app := fun F => {
      app := fun ⟨X⟩ => F.map (IsTerminal.from hT X).op
      naturality := fun _ _ _ => by
        simp only [Functor.comp_obj, Functor.const_obj_obj, Functor.id_obj, Functor.const_obj_map,
          Category.id_comp, ← Functor.map_comp]
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.59, u_1} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type u_2
          inst✝ : CategoryTheory.Category.{?u.82, u_2} D
          T : C
          hT : CategoryTheory.Limits.IsTerminal T
          F : CategoryTheory.Functor (Opposite C) D
          x✝² x✝¹ : Opposite C
          x✝ : Quiver.Hom x✝² x✝¹
          ⊢ Eq (F.map (hT.from (Opposite.unop x✝¹)).op) (F.map (CategoryTheory.CategoryS …
        -/
        congr
        /-
          case e_a.e_f
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.59, u_1} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type u_2
          inst✝ : CategoryTheory.Category.{?u.82, u_2} D
          T : C
          hT : CategoryTheory.Limits.IsTerminal T
          F : CategoryTheory.Functor (Opposite C) D
          x✝² x✝¹ : Opposite C
          x✝ : Quiver.Hom x✝² x✝¹
          ⊢ Eq (hT.from (Opposite.unop x✝¹)) (CategoryTheory.CategoryStruct.comp x✝.unop …
        -/
        simp }
        /-
          🎉 no goals
        -/
                     /-
                       C : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.59, u_1} C
                       J : CategoryTheory.GrothendieckTopology C
                       D : Type u_2
                       inst✝ : CategoryTheory.Category.{?u.82, u_2} D
                       T : C
                       hT : CategoryTheory.Limits.IsTerminal T
                       ⊢ ∀ ⦃X Y : CategoryTheory.Functor (Opposite C) D⦄ (f : Quiver.Hom X Y), Eq (Ca …
                     -/
    naturality := by intros; ext; simp /- Note: `aesop` works but is kind of slow -/ }
                                  /-
                                    🎉 no goals
                                  -/


/--
The functor which maps an object of `D` to the constant sheaf at that object, i.e. the
sheafification of the constant presheaf.
-/
noncomputable def constantSheaf : D ⥤ Sheaf J D := Functor.const Cᵒᵖ ⋙ (presheafToSheaf J D)


/-- The constant sheaf functor is left adjoint to evaluation at a terminal object. -/
@[simps! counit_app]
noncomputable def constantSheafAdj {T : C} (hT : IsTerminal T) :
    constantSheaf J D ⊣ (sheafSections J D).obj (op T) :=
  (constantPresheafAdj D hT).comp (sheafificationAdjunction J D)


/--
A sheaf is constant if it is in the essential image of the constant sheaf functor.
-/
class IsConstant (F : Sheaf J D) : Prop where
  mem_essImage : F ∈ (constantSheaf J D).essImage


lemma mem_essImage_of_isConstant (F : Sheaf J D) [IsConstant J F] :
    F ∈ (constantSheaf J D).essImage :=
  IsConstant.mem_essImage


lemma isConstant_congr {F G : Sheaf J D} (i : F ≅ G) [IsConstant J F] : IsConstant J G where
  mem_essImage := essImage.ofIso i F.mem_essImage_of_isConstant


lemma isConstant_of_iso {F : Sheaf J D} {X : D} (i : F ≅ (constantSheaf J D).obj X) :
    IsConstant J F := ⟨_, ⟨i.symm⟩⟩


lemma isConstant_iff_mem_essImage {L : D ⥤ Sheaf J D} {T : C} (hT : IsTerminal T)
    (adj : L ⊣ (sheafSections J D).obj ⟨T⟩)
    (F : Sheaf J D) : IsConstant J F ↔ F ∈ L.essImage := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.HasWeakSheafify J D
    L : CategoryTheory.Functor D (CategoryTheory.Sheaf J D)
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    adj : CategoryTheory.Adjunction L ((CategoryTheory.sheafSections J D).obj { un …
    F : CategoryTheory.Sheaf J D
    ⊢ Iff (CategoryTheory.Sheaf.IsConstant J F) (Membership.mem L.essImage F)
  -/
  rw [essImage_eq_of_natIso (adj.leftAdjointUniq (constantSheafAdj J D hT))]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.HasWeakSheafify J D
    L : CategoryTheory.Functor D (CategoryTheory.Sheaf J D)
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    adj : CategoryTheory.Adjunction L ((CategoryTheory.sheafSections J D).obj { un …
    F : CategoryTheory.Sheaf J D
    ⊢ Iff (CategoryTheory.Sheaf.IsConstant J F) (Membership.mem (CategoryTheory.co …
  -/
  exact ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  /-
    🎉 no goals
  -/


lemma isConstant_of_isIso_counit_app (F : Sheaf J D) [HasTerminal C]
    [IsIso <| (constantSheafAdj J D terminalIsTerminal).counit.app F] : IsConstant J F where
  mem_essImage := ⟨_, ⟨asIso <| (constantSheafAdj J D terminalIsTerminal).counit.app F⟩⟩


instance [(constantSheaf J D).Faithful] [(constantSheaf J D).Full] (F : Sheaf J D)
    [IsConstant J F] {T : C} (hT : IsTerminal T) :
    IsIso ((constantSheafAdj J D hT).counit.app F) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    inst✝³ : CategoryTheory.HasWeakSheafify J D
    inst✝² : (CategoryTheory.constantSheaf J D).Faithful
    inst✝¹ : (CategoryTheory.constantSheaf J D).Full
    F : CategoryTheory.Sheaf J D
    inst✝ : CategoryTheory.Sheaf.IsConstant J F
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ CategoryTheory.IsIso ((CategoryTheory.constantSheafAdj J D hT).counit.app F)
  -/
  rw [isIso_counit_app_iff_mem_essImage]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    inst✝³ : CategoryTheory.HasWeakSheafify J D
    inst✝² : (CategoryTheory.constantSheaf J D).Faithful
    inst✝¹ : (CategoryTheory.constantSheaf J D).Full
    F : CategoryTheory.Sheaf J D
    inst✝ : CategoryTheory.Sheaf.IsConstant J F
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ Membership.mem (CategoryTheory.constantSheaf J D).essImage F
  -/
  exact F.mem_essImage_of_isConstant
  /-
    🎉 no goals
  -/


/--
If the constant sheaf functor is fully faithful, then a sheaf is constant if and only if the
counit of the constant sheaf adjunction applied to it is an isomorphism.
-/
lemma isConstant_iff_isIso_counit_app [(constantSheaf J D).Faithful] [(constantSheaf J D).Full]
    (F : Sheaf J D) {T : C} (hT : IsTerminal T) :
      IsConstant J F ↔ (IsIso <| (constantSheafAdj J D hT).counit.app F) :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ ⟨_, ⟨asIso <| (constantSheafAdj J D hT).counit.app F⟩⟩⟩


/--
A variant of `isConstant_iff_isIso_counit_app` for a general left adjoint to evaluation at a
terminal object.
-/
lemma isConstant_iff_isIso_counit_app'  {L : D ⥤ Sheaf J D} {T : C} (hT : IsTerminal T)
    (adj : L ⊣ (sheafSections J D).obj ⟨T⟩)
    [L.Faithful] [L.Full] (F : Sheaf J D) : IsConstant J F ↔ IsIso (adj.counit.app F) :=
  (isConstant_iff_mem_essImage J hT adj F).trans (isIso_counit_app_iff_mem_essImage adj).symm


variable (D) in
/--
The constant sheaf functor commutes up to isomorphism the equivalence of sheaf categories induced
by a dense subsite.
-/
noncomputable def equivCommuteConstant :
    constantSheaf J D ⋙ (sheafEquiv G J K D).functor ≅ constantSheaf K D :=
  ((constantSheafAdj J D hT).comp (sheafEquiv G J K D).toAdjunction).leftAdjointUniq
    (constantSheafAdj K D hT')


variable (D) in
/--
The constant sheaf functor commutes up to isomorphism the inverse equivalence of sheaf categories
induced by a dense subsite.
-/
noncomputable def equivCommuteConstant' :
    constantSheaf J D ≅ constantSheaf K D ⋙ (sheafEquiv G J K D).inverse :=
  isoWhiskerLeft (constantSheaf J D) (sheafEquiv G J K D).unitIso ≪≫
    isoWhiskerRight (equivCommuteConstant J D K G hT hT') (sheafEquiv G J K D).inverse

/- TODO: find suitable assumptions for proving generalizations of `equivCommuteConstant` and
`equivCommuteConstant'` above, to commute `constantSheaf` with pullback/pushforward of sheaves. -/


include hT hT' in
/--
The property of a sheaf of being constant is invariant under equivalence of sheaf
categories.
-/
lemma Sheaf.isConstant_iff_of_equivalence (F : Sheaf K D) :
    ((sheafEquiv G J K D).inverse.obj F).IsConstant J ↔ IsConstant K F := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    C' : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C'
    K : CategoryTheory.GrothendieckTopology C'
    inst✝² : CategoryTheory.HasWeakSheafify K D
    G : CategoryTheory.Functor C C'
    inst✝¹ : ∀ (X : Opposite C'), CategoryTheory.Limits.HasLimitsOfShape (Category …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    hT' : CategoryTheory.Limits.IsTerminal (G.obj T)
    F : CategoryTheory.Sheaf K D
    ⊢ Iff (CategoryTheory.Sheaf.IsConstant J ((CategoryTheory.Functor.IsDenseSubsi …
  -/
  constructor
  · exact fun ⟨Y, ⟨i⟩⟩ ↦ ⟨_, ⟨(equivCommuteConstant J D K G hT hT').symm.app _ ≪≫
      (sheafEquiv G J K D).functor.mapIso i ≪≫ (sheafEquiv G J K D).counitIso.app _⟩⟩
  · exact fun ⟨Y, ⟨i⟩⟩ ↦ ⟨_, ⟨(equivCommuteConstant' J D K G hT hT').app _ ≪≫
      (sheafEquiv G J K D).inverse.mapIso i⟩⟩


/--
The constant sheaf functor commutes with `sheafCompose J U` up to isomorphism, provided that `U`
preserves sheafification.
-/
noncomputable def constantCommuteCompose :
    constantSheaf J D ⋙ sheafCompose J U ≅ U ⋙ constantSheaf J B :=
  (isoWhiskerLeft (const Cᵒᵖ)
    (sheafComposeNatIso J U (sheafificationAdjunction J D) (sheafificationAdjunction J B)).symm) ≪≫
      isoWhiskerRight (compConstIso _ _).symm _


lemma constantCommuteCompose_hom_app_val (X : D) : ((constantCommuteCompose J U).hom.app X).val =
    (sheafifyComposeIso J U ((const Cᵒᵖ).obj X)).inv ≫ sheafifyMap J (constComp Cᵒᵖ X U).hom := rfl


/-- The counit of `constantSheafAdj` factors through the isomorphism `constantCommuteCompose`. -/
lemma constantSheafAdj_counit_w {T : C} (hT : IsTerminal T) :
    ((constantCommuteCompose J U).hom.app (F.val.obj ⟨T⟩)) ≫
      ((constantSheafAdj J B hT).counit.app ((sheafCompose J U).obj F)) =
        ((sheafCompose J U).map ((constantSheafAdj J D hT).counit.app F)) := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.constantCommuteCompo …
  -/
  apply Sheaf.hom_ext
  /-
    case h
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.constantCommuteCompo …
  -/
  rw [instCategorySheaf_comp_val, constantCommuteCompose_hom_app_val, assoc, Iso.inv_comp_eq]
  /-
    case h
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.sheafifyMap J (Catego …
  -/
  apply sheafify_hom_ext _ _ _ ((sheafCompose J U).obj F).cond
  /-
    case h
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (((Categ …
  -/
  ext
  simp? says simp only [comp_obj, const_obj_obj, sheafCompose_obj_val, id_obj,
      constantSheafAdj_counit_app, instCategorySheaf_comp_val,
      sheafificationAdjunction_counit_app_val, sheafifyMap_sheafifyLift, comp_id,
      toSheafify_sheafifyLift, NatTrans.comp_app, constComp_hom_app,
      constantPresheafAdj_counit_app_app, Functor.comp_map, id_comp, flip_obj_obj,
      sheafToPresheaf_obj, map_comp, sheafCompose_map_val, sheafComposeIso_hom_fac_assoc,
      whiskerRight_app]
  /-
    case h.w.h
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    x✝ : Opposite C
    ⊢ Eq (U.map (F.val.map (hT.from (Opposite.unop x✝)).op)) (CategoryTheory.Categ …
  -/
  simp [← map_comp, ← NatTrans.comp_app]
  /-
    🎉 no goals
  -/


lemma Sheaf.isConstant_of_forget [constantSheaf J D |>.Faithful] [constantSheaf J D |>.Full]
    [constantSheaf J B |>.Faithful] [constantSheaf J B |>.Full]
    [(sheafCompose J U).ReflectsIsomorphisms] [((sheafCompose J U).obj F).IsConstant J]
    {T : C} (hT : IsTerminal T) : F.IsConstant J := by
  have : IsIso ((sheafCompose J U).map ((constantSheafAdj J D hT).counit.app F)) := by
    rw [← constantSheafAdj_counit_w]
    infer_instance
  /-
    C : Type u_1
    inst✝¹² : CategoryTheory.Category.{u_5, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹⁰ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} B
    U : CategoryTheory.Functor D B
    inst✝⁸ : CategoryTheory.HasWeakSheafify J B
    inst✝⁷ : J.PreservesSheafification U
    inst✝⁶ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    inst✝⁵ : (CategoryTheory.constantSheaf J D).Faithful
    inst✝⁴ : (CategoryTheory.constantSheaf J D).Full
    inst✝³ : (CategoryTheory.constantSheaf J B).Faithful
    inst✝² : (CategoryTheory.constantSheaf J B).Full
    inst✝¹ : (CategoryTheory.sheafCompose J U).ReflectsIsomorphisms
    inst✝ : CategoryTheory.Sheaf.IsConstant J ((CategoryTheory.sheafCompose J U).o …
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    this : CategoryTheory.IsIso ((CategoryTheory.sheafCompose J U).map ((CategoryT …
    ⊢ CategoryTheory.Sheaf.IsConstant J F
  -/
  rw [F.isConstant_iff_isIso_counit_app (hT := hT)]
  /-
    C : Type u_1
    inst✝¹² : CategoryTheory.Category.{u_5, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹⁰ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} B
    U : CategoryTheory.Functor D B
    inst✝⁸ : CategoryTheory.HasWeakSheafify J B
    inst✝⁷ : J.PreservesSheafification U
    inst✝⁶ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    inst✝⁵ : (CategoryTheory.constantSheaf J D).Faithful
    inst✝⁴ : (CategoryTheory.constantSheaf J D).Full
    inst✝³ : (CategoryTheory.constantSheaf J B).Faithful
    inst✝² : (CategoryTheory.constantSheaf J B).Full
    inst✝¹ : (CategoryTheory.sheafCompose J U).ReflectsIsomorphisms
    inst✝ : CategoryTheory.Sheaf.IsConstant J ((CategoryTheory.sheafCompose J U).o …
    T : C
    hT : CategoryTheory.Limits.IsTerminal T
    this : CategoryTheory.IsIso ((CategoryTheory.sheafCompose J U).map ((CategoryT …
    ⊢ CategoryTheory.IsIso ((CategoryTheory.constantSheafAdj J D hT).counit.app F)
  -/
  exact isIso_of_reflects_iso _ (sheafCompose J U)
  /-
    🎉 no goals
  -/


instance [h : F.IsConstant J] : ((sheafCompose J U).obj F).IsConstant J := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : CategoryTheory.HasWeakSheafify J D
    B : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_3} B
    U : CategoryTheory.Functor D B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.PreservesSheafification U
    inst✝ : J.HasSheafCompose U
    F : CategoryTheory.Sheaf J D
    h : CategoryTheory.Sheaf.IsConstant J F
    ⊢ CategoryTheory.Sheaf.IsConstant J ((CategoryTheory.sheafCompose J U).obj F)
  -/
  obtain ⟨Y, ⟨i⟩⟩ := h
  exact ⟨U.obj Y, ⟨(fullyFaithfulSheafToPresheaf _ _).preimageIso
    (((sheafifyComposeIso J U ((const Cᵒᵖ).obj Y)).symm ≪≫
      (presheafToSheaf J B ⋙ sheafToPresheaf J B).mapIso (constComp Cᵒᵖ Y U)).symm ≪≫
        (sheafToPresheaf _ _).mapIso ((sheafCompose J U).mapIso i))⟩⟩


lemma Sheaf.isConstant_iff_forget [constantSheaf J D |>.Faithful] [constantSheaf J D |>.Full]
    [constantSheaf J B |>.Faithful] [constantSheaf J B |>.Full]
      [(sheafCompose J U).ReflectsIsomorphisms] {T : C} (hT : IsTerminal T) :
        F.IsConstant J ↔ ((sheafCompose J U).obj F).IsConstant J :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ Sheaf.isConstant_of_forget _ U F hT⟩


