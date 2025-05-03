/-- The forgetful functor from `Sheaf J D` to sheaves of types, for a concrete category `D`
whose forgetful functor preserves the correct limits. -/
abbrev sheafForget [ConcreteCategory D] [HasSheafCompose J (forget D)] :
    Sheaf J D ⥤ Sheaf J (Type _) :=
  sheafCompose J (forget D)


/-- An adjunction `adj : G ⊣ F` with `F : D ⥤ E` and `G : E ⥤ D` induces an adjunction
between `Sheaf J D` and `Sheaf J E`, in contexts where one can sheafify `D`-valued presheaves,
and postcomposing with `F` preserves the property of being a sheaf. -/
def adjunction [HasWeakSheafify J D] [HasSheafCompose J F] (adj : G ⊣ F) :
    composeAndSheafify J G ⊣ sheafCompose J F :=
  Adjunction.restrictFullyFaithful ((adj.whiskerRight Cᵒᵖ).comp (sheafificationAdjunction J D))
    (fullyFaithfulSheafToPresheaf J E) (Functor.FullyFaithful.id _) (Iso.refl _) (Iso.refl _)


@[simp]
lemma adjunction_unit_app_val [HasWeakSheafify J D] [HasSheafCompose J F] (adj : G ⊣ F)
    (X : Sheaf J E) : ((adjunction J adj).unit.app X).val =
      (adj.whiskerRight Cᵒᵖ).unit.app _ ≫ whiskerRight (toSheafify J (X.val ⋙ G)) F  := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor D E
    G : CategoryTheory.Functor E D
    inst✝¹ : CategoryTheory.HasWeakSheafify J D
    inst✝ : J.HasSheafCompose F
    adj : CategoryTheory.Adjunction G F
    X : CategoryTheory.Sheaf J E
    ⊢ Eq ((CategoryTheory.Sheaf.adjunction J adj).unit.app X).val (CategoryTheory. …
  -/
  change (sheafToPresheaf _ _).map ((adjunction J adj).unit.app X) = _
  simp only [Functor.id_obj, Functor.comp_obj, whiskeringRight_obj_obj, adjunction,
    Adjunction.map_restrictFullyFaithful_unit_app, Adjunction.comp_unit_app,
    sheafificationAdjunction_unit_app, whiskeringRight_obj_map, Iso.refl_hom, NatTrans.id_app,
    Functor.comp_map, Functor.map_id, whiskerRight_id', Category.comp_id]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor D E
    G : CategoryTheory.Functor E D
    inst✝¹ : CategoryTheory.HasWeakSheafify J D
    inst✝ : J.HasSheafCompose F
    adj : CategoryTheory.Adjunction G F
    X : CategoryTheory.Sheaf J E
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Adjunction.whiskerRi …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma adjunction_counit_app_val [HasWeakSheafify J D] [HasSheafCompose J F] (adj : G ⊣ F)
    (Y : Sheaf J D) : ((adjunction J adj).counit.app Y).val =
      sheafifyLift J (((adj.whiskerRight Cᵒᵖ).counit.app Y.val)) Y.cond := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor D E
    G : CategoryTheory.Functor E D
    inst✝¹ : CategoryTheory.HasWeakSheafify J D
    inst✝ : J.HasSheafCompose F
    adj : CategoryTheory.Adjunction G F
    Y : CategoryTheory.Sheaf J D
    ⊢ Eq ((CategoryTheory.Sheaf.adjunction J adj).counit.app Y).val (CategoryTheor …
  -/
  change ((𝟭 (Sheaf _ _)).map ((adjunction J adj).counit.app Y)).val = _
  simp only [Functor.comp_obj, sheafToPresheaf_obj, sheafCompose_obj_val, whiskeringRight_obj_obj,
    adjunction, Adjunction.map_restrictFullyFaithful_counit_app, Iso.refl_inv, NatTrans.id_app,
    Functor.comp_map, whiskeringRight_obj_map, Adjunction.comp_counit_app,
    instCategorySheaf_comp_val, instCategorySheaf_id_val, sheafificationAdjunction_counit_app_val,
    sheafifyMap_sheafifyLift, Functor.id_obj, whiskerRight_id', Category.comp_id, Category.id_comp]



instance [HasWeakSheafify J D] [F.IsRightAdjoint] : (sheafCompose J F).IsRightAdjoint :=
  (adjunction J (Adjunction.ofIsRightAdjoint F)).isRightAdjoint


instance [HasWeakSheafify J D] [G.IsLeftAdjoint] : (composeAndSheafify J G).IsLeftAdjoint :=
  (adjunction J (Adjunction.ofIsLeftAdjoint G)).isLeftAdjoint


lemma preservesSheafification_of_adjunction (adj : G ⊣ F) :
    J.PreservesSheafification G where
  le P Q f hf := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) E D).obj G) f
    -/
    have := adj.isRightAdjoint
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) E D).obj G) f
    -/
    rw [MorphismProperty.inverseImage_iff]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      ⊢ J.W (((CategoryTheory.whiskeringRight (Opposite C) E D).obj G).map f)
    -/
    dsimp
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      ⊢ J.W (CategoryTheory.whiskerRight f G)
    -/
    intro R hR
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp (CategoryTheo …
    -/
    rw [← ((adj.whiskerRight Cᵒᵖ).homEquiv P R).comp_bijective]
    convert (((adj.whiskerRight Cᵒᵖ).homEquiv Q R).trans
      (hf.homEquiv (R ⋙ F) ((sheafCompose J F).obj ⟨R, hR⟩).cond)).bijective
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      e_1✝ : Eq (Quiver.Hom (Q.comp G) R) (Quiver.Hom (((CategoryTheory.whiskeringRi …
      e_2✝ : Eq (Quiver.Hom P (((CategoryTheory.whiskeringRight (Opposite C) D E).ob …
      ⊢ Eq (Function.comp ⇑((CategoryTheory.Adjunction.whiskerRight (Opposite C) adj …
    -/
    ext g X
    -- The rest of this proof was
    -- `dsimp [Adjunction.whiskerRight, Adjunction.mkOfUnitCounit]; simp` before https://github.com/leanprover-community/mathlib4/pull/16317.
    /-
      case h.e'_3.h.h.w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      e_1✝ : Eq (Quiver.Hom (Q.comp G) R) (Quiver.Hom (((CategoryTheory.whiskeringRi …
      e_2✝ : Eq (Quiver.Hom P (((CategoryTheory.whiskeringRight (Opposite C) D E).ob …
      g : Quiver.Hom (Q.comp G) R
      X : Opposite C
      ⊢ Eq ((Function.comp (⇑((CategoryTheory.Adjunction.whiskerRight (Opposite C) a …
    -/
    dsimp
    /-
      case h.e'_3.h.h.w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      e_1✝ : Eq (Quiver.Hom (Q.comp G) R) (Quiver.Hom (((CategoryTheory.whiskeringRi …
      e_2✝ : Eq (Quiver.Hom P (((CategoryTheory.whiskeringRight (Opposite C) D E).ob …
      g : Quiver.Hom (Q.comp G) R
      X : Opposite C
      ⊢ Eq ((((CategoryTheory.Adjunction.whiskerRight (Opposite C) adj).homEquiv P R …
    -/
    rw [← NatTrans.comp_app]
    /-
      case h.e'_3.h.h.w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      e_1✝ : Eq (Quiver.Hom (Q.comp G) R) (Quiver.Hom (((CategoryTheory.whiskeringRi …
      e_2✝ : Eq (Quiver.Hom P (((CategoryTheory.whiskeringRight (Opposite C) D E).ob …
      g : Quiver.Hom (Q.comp G) R
      X : Opposite C
      ⊢ Eq ((((CategoryTheory.Adjunction.whiskerRight (Opposite C) adj).homEquiv P R …
    -/
    congr
    /-
      case h.e'_3.h.h.w.h.e_self
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} E
      F : CategoryTheory.Functor D E
      G : CategoryTheory.Functor E D
      adj : CategoryTheory.Adjunction G F
      P Q : CategoryTheory.Functor (Opposite C) E
      f : Quiver.Hom P Q
      hf : J.W f
      this : F.IsRightAdjoint
      R : CategoryTheory.Functor (Opposite C) D
      hR : CategoryTheory.Presheaf.IsSheaf J R
      e_1✝ : Eq (Quiver.Hom (Q.comp G) R) (Quiver.Hom (((CategoryTheory.whiskeringRi …
      e_2✝ : Eq (Quiver.Hom P (((CategoryTheory.whiskeringRight (Opposite C) D E).ob …
      g : Quiver.Hom (Q.comp G) R
      X : Opposite C
      ⊢ Eq (((CategoryTheory.Adjunction.whiskerRight (Opposite C) adj).homEquiv P R) …
    -/
    exact Adjunction.homEquiv_naturality_left _ _ _
    /-
      🎉 no goals
    -/


instance [G.IsLeftAdjoint] : J.PreservesSheafification G :=
  preservesSheafification_of_adjunction J (Adjunction.ofIsLeftAdjoint G)


@[deprecated (since := "2024-11-26")] alias composeAndSheafifyFromTypes := composeAndSheafify


/-- The adjunction `composeAndSheafify J G ⊣ sheafForget J`. -/
@[deprecated Sheaf.adjunction (since := "2024-11-26")] abbrev adjunctionToTypes
    {G : Type max v₁ u₁ ⥤ D} (adj : G ⊣ forget D) :
    composeAndSheafify J G ⊣ sheafForget J :=
  adjunction _ adj


