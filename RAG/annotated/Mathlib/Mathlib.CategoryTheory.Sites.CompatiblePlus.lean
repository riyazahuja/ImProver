/-- The diagram used to define `P⁺`, composed with `F`, is isomorphic
to the diagram used to define `P ⋙ F`. -/
def diagramCompIso (X : C) : J.diagram P X ⋙ F ≅ J.diagram (P ⋙ F) X :=
  NatIso.ofComponents
    (fun W => by
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝³ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
        P : CategoryTheory.Functor (Opposite C) D
        X : C
        W : Opposite (J.Cover X)
        ⊢ CategoryTheory.Iso (((J.diagram P X).comp F).obj W) ((J.diagram (P.comp F) X …
      -/
      refine ?_ ≪≫ HasLimit.isoOfNatIso (W.unop.multicospanComp _ _).symm
      refine
        (isLimitOfPreserves F (limit.isLimit _)).conePointUniqueUpToIso (limit.isLimit _))
    (by
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝³ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
        P : CategoryTheory.Functor (Opposite C) D
        X : C
        ⊢ ∀ {X_1 Y : Opposite (J.Cover X)} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory. …
      -/
      intro A B f
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝³ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
        P : CategoryTheory.Functor (Opposite C) D
        X : C
        A B : Opposite (J.Cover X)
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((J.diagram P X).comp F).map f) ((fu …
      -/
      dsimp
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝³ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
        P : CategoryTheory.Functor (Opposite C) D
        X : C
        A B : Opposite (J.Cover X)
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
      -/
      ext g
      /-
        case h
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝³ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
        P : CategoryTheory.Functor (Opposite C) D
        X : C
        A B : Opposite (J.Cover X)
        f : Quiver.Hom A B
        g : ((Opposite.unop B).index (P.comp F)).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp [← F.map_comp])
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem diagramCompIso_hom_ι (X : C) (W : (J.Cover X)ᵒᵖ) (i : W.unop.Arrow) :
    (J.diagramCompIso F P X).hom.app W ≫ Multiequalizer.ι ((unop W).index (P ⋙ F)) i =
  F.map (Multiequalizer.ι _ _) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝³ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    P : CategoryTheory.Functor (Opposite C) D
    X : C
    W : Opposite (J.Cover X)
    i : (Opposite.unop W).Arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramCompIso F P X).hom.app W)  …
  -/
  delta diagramCompIso
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝³ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    P : CategoryTheory.Functor (Opposite C) D
    X : C
    W : Opposite (J.Cover X)
    i : (Opposite.unop W).Arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.NatIso.ofComponents  …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁴ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝³ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    P : CategoryTheory.Functor (Opposite C) D
    X : C
    W : Opposite (J.Cover X)
    i : (Opposite.unop W).Arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The isomorphism between `P⁺ ⋙ F` and `(P ⋙ F)⁺`. -/
def plusCompIso : J.plusObj P ⋙ F ≅ J.plusObj (P ⋙ F) :=
  NatIso.ofComponents
    (fun X => by
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X : Opposite C
        ⊢ CategoryTheory.Iso (((J.plusObj P).comp F).obj X) ((J.plusObj (P.comp F)).ob …
      -/
      refine ?_ ≪≫ HasColimit.isoOfNatIso (J.diagramCompIso F P X.unop)
      refine
        (isColimitOfPreserves F
              (colimit.isColimit (J.diagram P (unop X)))).coconePointUniqueUpToIso
          (colimit.isColimit _))
    (by
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
      -/
      intro X Y f
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((J.plusObj P).comp F).map f) ((fun  …
      -/
      apply (isColimitOfPreserves F (colimit.isColimit (J.diagram P X.unop))).hom_ext
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        ⊢ ∀ (j : Opposite (J.Cover (Opposite.unop X))), Eq (CategoryTheory.CategoryStr …
      -/
      intro W
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapCocone (CategoryTheory.Limits. …
      -/
      dsimp [plusObj, plusMap]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.colimit …
      -/
      simp only [Functor.map_comp, Category.assoc]
      slice_rhs 1 2 =>
        erw [(isColimitOfPreserves F (colimit.isColimit (J.diagram P X.unop))).fac]
      slice_lhs 1 3 =>
        simp only [← F.map_comp]
        dsimp [colimMap, IsColimit.map, colimit.pre]
        simp only [colimit.ι_desc_assoc, colimit.ι_desc]
        dsimp [Cocones.precompose]
        simp only [Category.assoc, colimit.ι_desc]
        dsimp [Cocone.whisker]
        rw [F.map_comp]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.assoc]
      slice_lhs 2 3 =>
        erw [(isColimitOfPreserves F (colimit.isColimit (J.diagram P Y.unop))).fac]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
      -/
      dsimp
      simp only [HasColimit.isoOfNatIso_ι_hom_assoc, GrothendieckTopology.diagramPullback_app,
        colimit.ι_pre, HasColimit.isoOfNatIso_ι_hom, ι_colimMap_assoc]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
      -/
      simp only [← Category.assoc]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      congr 1
      /-
        case e_a
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
      -/
      ext
      /-
        case e_a.h
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        a✝ : (((Opposite.unop W).pullback f.unop).index (P.comp F)).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      /-
        case e_a.h
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w₁
        inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
        E : Type w₂
        inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
        F : CategoryTheory.Functor D E
        inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
        inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
        P : CategoryTheory.Functor (Opposite C) D
        inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
        X Y : Opposite C
        f : Quiver.Hom X Y
        W : Opposite (J.Cover (Opposite.unop X))
        a✝ : (((Opposite.unop W).pullback f.unop).index (P.comp F)).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.assoc]
      rw [Multiequalizer.lift_ι, diagramCompIso_hom_ι, diagramCompIso_hom_ι, ← F.map_comp,
        Multiequalizer.lift_ι])


@[reassoc (attr := simp)]
theorem ι_plusCompIso_hom (X) (W) :
    F.map (colimit.ι _ W) ≫ (J.plusCompIso F P).hom.app X =
      (J.diagramCompIso F P X.unop).hom.app W ≫ colimit.ι _ W := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.colimit …
  -/
  delta diagramCompIso plusCompIso
  simp only [IsColimit.descCoconeMorphism_hom, IsColimit.uniqueUpToIso_hom,
    Cocones.forget_map, Iso.trans_hom, NatIso.ofComponents_hom_app, Functor.mapIso_hom, ←
    Category.assoc]
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [(isColimitOfPreserves F (colimit.isColimit (J.diagram P (unop X)))).fac]
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem plusCompIso_whiskerLeft {F G : D ⥤ E} (η : F ⟶ G) (P : Cᵒᵖ ⥤ D)
    [∀ X : C, PreservesColimitsOfShape (J.Cover X)ᵒᵖ F]
    [∀ (X : C) (W : J.Cover X) (P : Cᵒᵖ ⥤ D), PreservesLimit (W.index P).multicospan F]
    [∀ X : C, PreservesColimitsOfShape (J.Cover X)ᵒᵖ G]
    [∀ (X : C) (W : J.Cover X) (P : Cᵒᵖ ⥤ D), PreservesLimit (W.index P).multicospan G] :
    whiskerLeft _ η ≫ (J.plusCompIso G P).hom =
      (J.plusCompIso F P).hom ≫ J.plusMap (whiskerLeft _ η) := by
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft (J.plusOb …
  -/
  ext X
  /-
    case w.h
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft (J.plusO …
  -/
  apply (isColimitOfPreserves F (colimit.isColimit (J.diagram P X.unop))).hom_ext
  /-
    case w.h
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    ⊢ ∀ (j : Opposite (J.Cover (Opposite.unop X))), Eq (CategoryTheory.CategoryStr …
  -/
  intro W
  /-
    case w.h
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapCocone (CategoryTheory.Limits. …
  -/
  dsimp [plusObj, plusMap]
  simp only [ι_plusCompIso_hom, ι_colimMap, whiskerLeft_app, ι_plusCompIso_hom_assoc,
    NatTrans.naturality_assoc, GrothendieckTopology.diagramNatTrans_app]
  /-
    case w.h
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app (CategoryTheory.Limits.multieq …
  -/
  simp only [← Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case w.h.e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁹ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁸ : CategoryTheory.Category.{max v u, w₂} E
    inst✝⁷ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁶ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    F G : CategoryTheory.Functor D E
    η : Quiver.Hom F G
    P : CategoryTheory.Functor (Opposite C) D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D), …
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app (CategoryTheory.Limits.multieq …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The isomorphism between `P⁺ ⋙ F` and `(P ⋙ F)⁺`, functorially in `F`. -/
@[simps! hom_app inv_app]
def plusFunctorWhiskerLeftIso (P : Cᵒᵖ ⥤ D)
    [∀ (F : D ⥤ E) (X : C), PreservesColimitsOfShape (J.Cover X)ᵒᵖ F]
    [∀ (F : D ⥤ E) (X : C) (W : J.Cover X) (P : Cᵒᵖ ⥤ D),
        PreservesLimit (W.index P).multicospan F] :
    (whiskeringLeft _ _ E).obj (J.plusObj P) ≅ (whiskeringLeft _ _ _).obj P ⋙ J.plusFunctor E :=
  NatIso.ofComponents (fun _ => plusCompIso _ _ _) @fun _ _ _ => plusCompIso_whiskerLeft _ _ _


@[reassoc (attr := simp)]
theorem plusCompIso_whiskerRight {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) :
    whiskerRight (J.plusMap η) F ≫ (J.plusCompIso F Q).hom =
      (J.plusCompIso F P).hom ≫ J.plusMap (whiskerRight η F) := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (J.plusM …
  -/
  ext X
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (J.plus …
  -/
  apply (isColimitOfPreserves F (colimit.isColimit (J.diagram P X.unop))).hom_ext
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    ⊢ ∀ (j : Opposite (J.Cover (Opposite.unop X))), Eq (CategoryTheory.CategoryStr …
  -/
  intro W
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapCocone (CategoryTheory.Limits. …
  -/
  dsimp [plusObj, plusMap]
  simp only [ι_colimMap, whiskerRight_app, ι_plusCompIso_hom_assoc,
    GrothendieckTopology.diagramNatTrans_app]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.colimit …
  -/
  simp only [← Category.assoc, ← F.map_comp]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  dsimp [colimMap, IsColimit.map]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp only [colimit.ι_desc]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (((CategoryTheory.Limits.Cocon …
  -/
  dsimp [Cocones.precompose]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp only [Functor.map_comp, Category.assoc, ι_plusCompIso_hom]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
  -/
  simp only [← Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): this used to work with `ext`
  /-
    case w.h.e_a
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
  -/
  apply Multiequalizer.hom_ext
  /-
    case w.h.e_a.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    ⊢ ∀ (a : ((Opposite.unop W).index (Q.comp F)).L), Eq (CategoryTheory.CategoryS …
  -/
  intro a
  /-
    case w.h.e_a.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    a : ((Opposite.unop W).index (Q.comp F)).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  simp only [diagramCompIso_hom_ι_assoc, Multiequalizer.lift_ι, diagramCompIso_hom_ι,
    Category.assoc]
  /-
    case w.h.e_a.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    X : Opposite C
    W : Opposite (J.Cover (Opposite.unop X))
    a : ((Opposite.unop W).index (Q.comp F)).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.Multieq …
  -/
  simp only [← F.map_comp, Multiequalizer.lift_ι]
  /-
    🎉 no goals
  -/


/-- The isomorphism between `P⁺ ⋙ F` and `(P ⋙ F)⁺`, functorially in `P`. -/
@[simps! hom_app inv_app]
def plusFunctorWhiskerRightIso :
    J.plusFunctor D ⋙ (whiskeringRight _ _ _).obj F ≅
      (whiskeringRight _ _ _).obj F ⋙ J.plusFunctor E :=
  NatIso.ofComponents (fun _ => J.plusCompIso _ _) @fun _ _ _ => plusCompIso_whiskerRight _ _ _


@[reassoc (attr := simp)]
theorem whiskerRight_toPlus_comp_plusCompIso_hom :
    whiskerRight (J.toPlus _) _ ≫ (J.plusCompIso F P).hom = J.toPlus _ := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (J.toPlu …
  -/
  ext
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (J.toPl …
  -/
  dsimp [toPlus]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp only [ι_plusCompIso_hom, Functor.map_comp, Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (Top.top.toMultiequalizer P))  …
  -/
  simp only [← Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): was ext
  /-
    case w.h.e_a
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (Top.top.toMultiequalizer P))  …
  -/
  apply Multiequalizer.hom_ext; intro a
  /-
    case w.h.e_a.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    a : ((Opposite.unop { unop := Top.top }).index (P.comp F)).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, diagramCompIso_hom_ι, ← F.map_comp]
  /-
    case w.h.e_a.h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    x✝ : Opposite C
    a : ((Opposite.unop { unop := Top.top }).index (P.comp F)).L
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) ( …
  -/
  simp only [unop_op, limit.lift_π, Multifork.ofι_π_app, Functor.comp_obj, Functor.comp_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem toPlus_comp_plusCompIso_inv :
                                                                             /-
                                                                               C : Type u
                                                                               inst✝⁸ : CategoryTheory.Category.{v, u} C
                                                                               J : CategoryTheory.GrothendieckTopology C
                                                                               D : Type w₁
                                                                               inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
                                                                               E : Type w₂
                                                                               inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
                                                                               F : CategoryTheory.Functor D E
                                                                               inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
                                                                               inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
                                                                               inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
                                                                               P : CategoryTheory.Functor (Opposite C) D
                                                                               inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
                                                                               inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
                                                                               inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus (P.comp F)) (J.plusCompIso  …
                                                                             -/
    J.toPlus _ ≫ (J.plusCompIso F P).inv = whiskerRight (J.toPlus _) _ := by simp [Iso.comp_inv_eq]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem plusCompIso_inv_eq_plusLift (hP : Presheaf.IsSheaf J (J.plusObj P ⋙ F)) :
    (J.plusCompIso F P).inv = J.plusLift (whiskerRight (J.toPlus _) _) hP := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    hP : CategoryTheory.Presheaf.IsSheaf J ((J.plusObj P).comp F)
    ⊢ Eq (J.plusCompIso F P).inv (J.plusLift (CategoryTheory.whiskerRight (J.toPlu …
  -/
  apply J.plusLift_unique
  /-
    case hγ
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w₁
    inst✝⁷ : CategoryTheory.Category.{max v u, w₁} D
    E : Type w₂
    inst✝⁶ : CategoryTheory.Category.{max v u, w₂} E
    F : CategoryTheory.Functor D E
    inst✝⁵ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝⁴ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Has …
    inst✝³ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    hP : CategoryTheory.Presheaf.IsSheaf J ((J.plusObj P).comp F)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus (P.comp F)) (J.plusCompIso  …
  -/
  simp [Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


