/-- An auxiliary definition to be used in the proof of the fact that
`J.diagramFunctor D X` preserves limits. -/
@[simps]
def coneCompEvaluationOfConeCompDiagramFunctorCompEvaluation {X : C} {K : Type max v u}
    [SmallCategory K] {F : K ⥤ Cᵒᵖ ⥤ D} {W : J.Cover X} (i : W.Arrow)
    (E : Cone (F ⋙ J.diagramFunctor D X ⋙ (evaluation (J.Cover X)ᵒᵖ D).obj (op W))) :
    Cone (F ⋙ (evaluation _ _).obj (op i.Y)) where
  pt := E.pt
  π :=
    { app := fun k => E.π.app k ≫ Multiequalizer.ι (W.index (F.obj k)) i
      naturality := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{max v u, w} D
          inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          X : C
          K : Type (max v u)
          inst✝ : CategoryTheory.SmallCategory K
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          W : J.Cover X
          i : W.Arrow
          E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
          ⊢ ∀ ⦃X_1 Y : K⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
        -/
        intro a b f
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{max v u, w} D
          inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          X : C
          K : Type (max v u)
          inst✝ : CategoryTheory.SmallCategory K
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          W : J.Cover X
          i : W.Arrow
          E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
          a b : K
          f : Quiver.Hom a b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const K).ob …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{max v u, w} D
          inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          X : C
          K : Type (max v u)
          inst✝ : CategoryTheory.SmallCategory K
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          W : J.Cover X
          i : W.Arrow
          E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
          a b : K
          f : Quiver.Hom a b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id E.p …
        -/
        rw [Category.id_comp, Category.assoc, ← E.w f]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{max v u, w} D
          inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          X : C
          K : Type (max v u)
          inst✝ : CategoryTheory.SmallCategory K
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          W : J.Cover X
          i : W.Arrow
          E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
          a b : K
          f : Quiver.Hom a b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        dsimp [diagramNatTrans]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{max v u, w} D
          inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          X : C
          K : Type (max v u)
          inst✝ : CategoryTheory.SmallCategory K
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          W : J.Cover X
          i : W.Arrow
          E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
          a b : K
          f : Quiver.Hom a b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Multiequalizer.lift_ι, Category.assoc] }
        /-
          🎉 no goals
        -/


/-- Auxiliary definition for `liftToDiagramLimitObj`. -/
def liftToDiagramLimitObjAux {X : C} {K : Type max v u} [SmallCategory K] [HasLimitsOfShape K D]
    {W : (J.Cover X)ᵒᵖ} (F : K ⥤ Cᵒᵖ ⥤ D)
    (E : Cone (F ⋙ J.diagramFunctor D X ⋙ (evaluation (J.Cover X)ᵒᵖ D).obj W))
    (i : (unop W).Arrow) :
    E.pt ⟶ (limit F).obj (op i.Y) :=
  (isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op i.Y)) (limit.isLimit F)).lift
        (coneCompEvaluationOfConeCompDiagramFunctorCompEvaluation.{w, v, u} i E)


@[reassoc (attr := simp)]
lemma liftToDiagramLimitObjAux_fac {X : C} {K : Type max v u} [SmallCategory K]
    [HasLimitsOfShape K D] {W : (J.Cover X)ᵒᵖ} (F : K ⥤ Cᵒᵖ ⥤ D)
    (E : Cone (F ⋙ J.diagramFunctor D X ⋙ (evaluation (J.Cover X)ᵒᵖ D).obj W))
    (i : (unop W).Arrow) (k : K) :
    liftToDiagramLimitObjAux F E i ≫ (limit.π F k).app (op i.Y) = E.π.app k ≫
      Multiequalizer.ι ((unop W).index (F.obj k)) i :=
  IsLimit.fac _ _ _


/-- An auxiliary definition to be used in the proof of the fact that
`J.diagramFunctor D X` preserves limits. -/
abbrev liftToDiagramLimitObj {X : C} {K : Type max v u} [SmallCategory K] [HasLimitsOfShape K D]
    {W : (J.Cover X)ᵒᵖ} (F : K ⥤ Cᵒᵖ ⥤ D)
    (E : Cone (F ⋙ J.diagramFunctor D X ⋙ (evaluation (J.Cover X)ᵒᵖ D).obj W)) :
    E.pt ⟶ (J.diagram (limit F) X).obj W :=
  Multiequalizer.lift ((unop W).index (limit F)) E.pt (liftToDiagramLimitObjAux F E)
    (by
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        ⊢ ∀ (b : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R), Eq (Cat …
      -/
      intro i
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
      -/
      dsimp
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
      -/
      ext k
      /-
        case w
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      /-
        case w
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.assoc, NatTrans.naturality, liftToDiagramLimitObjAux_fac_assoc]
      /-
        case w
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.π.app k) (CategoryTheory.CategoryS …
      -/
      erw [Multiequalizer.condition]
      /-
        case w
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        W : Opposite (J.Cover X)
        F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
        E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
        i : ((Opposite.unop W).index (CategoryTheory.Limits.limit F)).R
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.π.app k) (CategoryTheory.CategoryS …
      -/
      rfl)
      /-
        🎉 no goals
      -/


instance preservesLimit_diagramFunctor
    (X : C) (K : Type max v u) [SmallCategory K] [HasLimitsOfShape K D] (F : K ⥤ Cᵒᵖ ⥤ D) :
    PreservesLimit F (J.diagramFunctor D X) :=
  preservesLimit_of_evaluation _ _ fun W =>
    preservesLimit_of_preserves_limit_cone (limit.isLimit _)
      { lift := fun E => liftToDiagramLimitObj.{w, v, u} F E
        fac := by
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            ⊢ ∀ (s : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Cat …
          -/
          intro E k
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            k : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun E => CategoryTheory.Grothendiec …
          -/
          dsimp [diagramNatTrans]
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            k : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
          -/
          refine Multiequalizer.hom_ext _ _ _ (fun a => ?_)
          simp only [Multiequalizer.lift_ι, Multiequalizer.lift_ι_assoc, Category.assoc,
            liftToDiagramLimitObjAux_fac]
        uniq := by
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            ⊢ ∀ (s : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Cat …
          -/
          intro E m hm
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            m : Quiver.Hom E.pt (((J.diagramFunctor D X).comp ((CategoryTheory.evaluation  …
            hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.diagramFunctor  …
            ⊢ Eq m ((fun E => CategoryTheory.GrothendieckTopology.liftToDiagramLimitObj F  …
          -/
          refine Multiequalizer.hom_ext _ _ _ (fun a => limit_obj_ext (fun j => ?_))
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            m : Quiver.Hom E.pt (((J.diagramFunctor D X).comp ((CategoryTheory.evaluation  …
            hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.diagramFunctor  …
            a : ((Opposite.unop W).index (CategoryTheory.Limits.limit.cone F).pt).L
            j : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
          -/
          dsimp [liftToDiagramLimitObj]
          rw [Multiequalizer.lift_ι, Category.assoc, liftToDiagramLimitObjAux_fac, ← hm,
            Category.assoc]
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            m : Quiver.Hom E.pt (((J.diagramFunctor D X).comp ((CategoryTheory.evaluation  …
            hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.diagramFunctor  …
            a : ((Opposite.unop W).index (CategoryTheory.Limits.limit.cone F).pt).L
            j : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
          -/
          dsimp
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            m : Quiver.Hom E.pt (((J.diagramFunctor D X).comp ((CategoryTheory.evaluation  …
            hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.diagramFunctor  …
            a : ((Opposite.unop W).index (CategoryTheory.Limits.limit.cone F).pt).L
            j : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
          -/
          rw [limit.lift_π]
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            D : Type w
            inst✝³ : CategoryTheory.Category.{max v u, w} D
            inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
            X : C
            K : Type (max v u)
            inst✝¹ : CategoryTheory.SmallCategory K
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
            F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
            W : Opposite (J.Cover X)
            E : CategoryTheory.Limits.Cone (F.comp ((J.diagramFunctor D X).comp ((Category …
            m : Quiver.Hom E.pt (((J.diagramFunctor D X).comp ((CategoryTheory.evaluation  …
            hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.diagramFunctor  …
            a : ((Opposite.unop W).index (CategoryTheory.Limits.limit.cone F).pt).L
            j : K
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
          -/
          dsimp }
          /-
            🎉 no goals
          -/


instance preservesLimitsOfShape_diagramFunctor
    (X : C) (K : Type max v u) [SmallCategory K] [HasLimitsOfShape K D] :
    PreservesLimitsOfShape K (J.diagramFunctor D X) :=
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝³ : CategoryTheory.Category.{max v u, w} D
        inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
        X : C
        K : Type (max v u)
        inst✝¹ : CategoryTheory.SmallCategory K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        ⊢ ∀ {K_1 : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)},  …
      -/
  ⟨by apply preservesLimit_diagramFunctor.{w, v, u}⟩
      /-
        🎉 no goals
      -/


instance preservesLimits_diagramFunctor (X : C) [HasLimits D] :
    PreservesLimits (J.diagramFunctor D X) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    X : C
    inst✝ : CategoryTheory.Limits.HasLimits D
    ⊢ CategoryTheory.Limits.PreservesLimits (J.diagramFunctor D X)
  -/
  constructor
  /-
    case preservesLimitsOfShape
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    X : C
    inst✝ : CategoryTheory.Limits.HasLimits D
    ⊢ autoParam (∀ {J_1 : Type (max u v)} [inst : CategoryTheory.Category.{max u v …
  -/
  intro _ _
  /-
    case preservesLimitsOfShape
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    X : C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    J✝ : Type (max u v)
    inst✝ : CategoryTheory.Category.{max u v, max u v} J✝
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J✝ (J.diagramFunctor D X)
  -/
  apply preservesLimitsOfShape_diagramFunctor.{w, v, u}
  /-
    🎉 no goals
  -/


/-- An auxiliary definition to be used in the proof that `J.plusFunctor D` commutes
with finite limits. -/
def liftToPlusObjLimitObj {K : Type max v u} [SmallCategory K] [FinCategory K]
    [HasLimitsOfShape K D] [PreservesLimitsOfShape K (forget D)]
    [ReflectsLimitsOfShape K (forget D)] (F : K ⥤ Cᵒᵖ ⥤ D) (X : C)
    (S : Cone (F ⋙ J.plusFunctor D ⋙ (evaluation Cᵒᵖ D).obj (op X))) :
    S.pt ⟶ (J.plusObj (limit F)).obj (op X) :=
  let e := colimitLimitIso (F ⋙ J.diagramFunctor D X)
  let t : J.diagram (limit F) X ≅ limit (F ⋙ J.diagramFunctor D X) :=
    (isLimitOfPreserves (J.diagramFunctor D X) (limit.isLimit F)).conePointUniqueUpToIso
      (limit.isLimit _)
  let p : (J.plusObj (limit F)).obj (op X) ≅ colimit (limit (F ⋙ J.diagramFunctor D X)) :=
    HasColimit.isoOfNatIso t
  let s :
    colimit (F ⋙ J.diagramFunctor D X).flip ≅ F ⋙ J.plusFunctor D ⋙ (evaluation Cᵒᵖ D).obj (op X) :=
    NatIso.ofComponents (fun k => colimitObjIsoColimitCompEvaluation _ k)
      (by
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          ⊢ ∀ {X_1 Y : K} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
        -/
        intro i j f
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          i j : K
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit (F.co …
        -/
        rw [← Iso.eq_comp_inv, Category.assoc, ← Iso.inv_comp_eq]
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          i j : K
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun k => CategoryTheory.Limits.coli …
        -/
        refine colimit.hom_ext (fun w => ?_)
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          i j : K
          f : Quiver.Hom i j
          w : Opposite (J.Cover (Opposite.unop { unop := X }))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
        -/
        dsimp [plusMap]
        erw [colimit.ι_map_assoc,
          colimitObjIsoColimitCompEvaluation_ι_inv (F ⋙ J.diagramFunctor D X).flip w j,
          colimitObjIsoColimitCompEvaluation_ι_inv_assoc (F ⋙ J.diagramFunctor D X).flip w i]
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          i j : K
          f : Quiver.Hom i j
          w : Opposite (J.Cover (Opposite.unop { unop := X }))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.ι (F. …
        -/
        rw [← (colimit.ι (F ⋙ J.diagramFunctor D X).flip w).naturality]
        /-
          C : Type u
          inst✝¹⁰ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁹ : CategoryTheory.Category.{max v u, w} D
          inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝⁶ : CategoryTheory.ConcreteCategory D
          inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          K : Type (max v u)
          inst✝⁴ : CategoryTheory.SmallCategory K
          inst✝³ : CategoryTheory.FinCategory K
          inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
          inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
          F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
          X : C
          S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
          e : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.l …
          t : CategoryTheory.Iso (J.diagram (CategoryTheory.Limits.limit F) X) (Category …
          p : CategoryTheory.Iso ((J.plusObj (CategoryTheory.Limits.limit F)).obj { unop …
          i j : K
          f : Quiver.Hom i j
          w : Opposite (J.Cover (Opposite.unop { unop := X }))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.comp (J.diagramFunctor D X)).fli …
        -/
        rfl)
        /-
          🎉 no goals
        -/
  limit.lift _ S ≫ (HasLimit.isoOfNatIso s.symm).hom ≫ e.inv ≫ p.inv

-- This lemma should not be used directly. Instead, one should use the fact that
-- `J.plusFunctor D` preserves finite limits, along with the fact that
-- evaluation preserves limits.

theorem liftToPlusObjLimitObj_fac {K : Type max v u} [SmallCategory K] [FinCategory K]
    [HasLimitsOfShape K D] [PreservesLimitsOfShape K (forget D)]
    [ReflectsLimitsOfShape K (forget D)] (F : K ⥤ Cᵒᵖ ⥤ D) (X : C)
    (S : Cone (F ⋙ J.plusFunctor D ⋙ (evaluation Cᵒᵖ D).obj (op X))) (k) :
    liftToPlusObjLimitObj.{w, v, u} F X S ≫ (J.plusMap (limit.π F k)).app (op X) = S.π.app k := by
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
  -/
  dsimp only [liftToPlusObjLimitObj]
  rw [← (limit.isLimit (F ⋙ J.plusFunctor D ⋙ (evaluation Cᵒᵖ D).obj (op X))).fac S k,
    Category.assoc]
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift (F. …
  -/
  congr 1
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, Category.assoc, ← Iso.eq_inv_comp, Iso.inv_comp_eq, Iso.inv_comp_eq]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    ⊢ Eq ((J.plusMap (CategoryTheory.Limits.limit.π F k)).app { unop := X }) (Cate …
  -/
  refine colimit.hom_ext (fun j => ?_)
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  dsimp [plusMap]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  simp only [HasColimit.isoOfNatIso_ι_hom_assoc, ι_colimMap]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans (CategoryTheory.L …
  -/
  dsimp [IsLimit.conePointUniqueUpToIso, HasLimit.isoOfNatIso, IsLimit.map]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  rw [limit.lift_π]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  dsimp
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  rw [ι_colimitLimitIso_limit_π_assoc]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  simp_rw [← Category.assoc, ← NatTrans.comp_app]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  rw [limit.lift_π, Category.assoc]
  /-
    case e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  congr 1
  /-
    case e_a.e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.Limits.colimit.ι (J.diagram (F.obj k) X) j) (CategoryTheo …
  -/
  rw [← Iso.comp_inv_eq]
  /-
    case e_a.e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  erw [colimit.ι_desc]
  /-
    case e_a.e_a
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : C
    S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
    k : K
    j : Opposite (J.Cover (Opposite.unop { unop := X }))
    ⊢ Eq ((((CategoryTheory.evaluation K D).obj k).mapCocone (CategoryTheory.Limit …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance preservesLimitsOfShape_plusFunctor
    (K : Type max v u) [SmallCategory K] [FinCategory K] [HasLimitsOfShape K D]
    [PreservesLimitsOfShape K (forget D)] [ReflectsLimitsOfShape K (forget D)] :
    PreservesLimitsOfShape K (J.plusFunctor D) := by
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape K (J.plusFunctor D)
  -/
  constructor; intro F; apply preservesLimit_of_evaluation; intro X
  /-
    case preservesLimit.H
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : Opposite C
    ⊢ CategoryTheory.Limits.PreservesLimit F ((J.plusFunctor D).comp ((CategoryThe …
  -/
  apply preservesLimit_of_preserves_limit_cone (limit.isLimit F)
  /-
    case preservesLimit.H
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    K : Type (max v u)
    inst✝⁴ : CategoryTheory.SmallCategory K
    inst✝³ : CategoryTheory.FinCategory K
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
    X : Opposite C
    ⊢ CategoryTheory.Limits.IsLimit (((J.plusFunctor D).comp ((CategoryTheory.eval …
  -/
  refine ⟨fun S => liftToPlusObjLimitObj.{w, v, u} F X.unop S, ?_, ?_⟩
    /-
      case preservesLimit.H.refine_1
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((Category …
    -/
  · intro S k
    /-
      case preservesLimit.H.refine_1
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun S => CategoryTheory.Grothendiec …
    -/
    apply liftToPlusObjLimitObj_fac
    /-
      🎉 no goals
    -/
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((Category …
    -/
  · intro S m hm
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      ⊢ Eq m ((fun S => CategoryTheory.GrothendieckTopology.liftToPlusObjLimitObj F  …
    -/
    dsimp [liftToPlusObjLimitObj]
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift ( …
    -/
    simp_rw [← Category.assoc, Iso.eq_comp_inv, ← Iso.comp_inv_eq]
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    refine limit.hom_ext (fun k => ?_)
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [limit.lift_π, Category.assoc, ← hm]
    /-
      case preservesLimit.H.refine_2
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
    -/
    congr 1
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasColimit.iso …
    -/
    refine colimit.hom_ext (fun k => ?_)
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    dsimp [plusMap, plusObj]
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    erw [colimit.ι_map, colimit.ι_desc_assoc, limit.lift_π]
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
    -/
    conv_lhs => dsimp
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc]
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.isLimitOfPre …
    -/
    rw [ι_colimitLimitIso_limit_π_assoc]
    simp only [NatIso.ofComponents_inv_app, colimitObjIsoColimitCompEvaluation_ι_app_hom,
      Iso.symm_inv]
    conv_lhs =>
      dsimp [IsLimit.conePointUniqueUpToIso]
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.limit.lift (F …
    -/
    rw [← Category.assoc, ← NatTrans.comp_app, limit.lift_π]
    /-
      case preservesLimit.H.refine_2.e_a
      C : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁹ : CategoryTheory.Category.{max v u, w} D
      inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      K : Type (max v u)
      inst✝⁴ : CategoryTheory.SmallCategory K
      inst✝³ : CategoryTheory.FinCategory K
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.forget …
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
      F : CategoryTheory.Functor K (CategoryTheory.Functor (Opposite C) D)
      X : Opposite C
      S : CategoryTheory.Limits.Cone (F.comp ((J.plusFunctor D).comp ((CategoryTheor …
      m : Quiver.Hom S.pt (((J.plusFunctor D).comp ((CategoryTheory.evaluation (Oppo …
      hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((J.plusFunctor D). …
      k✝ : K
      k : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((J.diagramFunctor D (Opposite.unop …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance preserveFiniteLimits_plusFunctor
    [HasFiniteLimits D] [PreservesFiniteLimits (forget D)] [(forget D).ReflectsIsomorphisms] :
    PreservesFiniteLimits (J.plusFunctor D) := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁷ : CategoryTheory.Category.{max v u, w} D
    inst✝⁶ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : CategoryTheory.Limits.HasFiniteLimits D
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (J.plusFunctor D)
  -/
  apply preservesFiniteLimits_of_preservesFiniteLimitsOfSize.{max v u}
  /-
    case h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁷ : CategoryTheory.Category.{max v u, w} D
    inst✝⁶ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : CategoryTheory.Limits.HasFiniteLimits D
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    ⊢ ∀ (J_1 : Type (max v u)) {𝒥 : CategoryTheory.SmallCategory J_1}, CategoryThe …
  -/
  intro K _ _
  /-
    case h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁷ : CategoryTheory.Category.{max v u, w} D
    inst✝⁶ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : CategoryTheory.Limits.HasFiniteLimits D
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type (max v u)
    𝒥✝ : CategoryTheory.SmallCategory K
    x✝ : CategoryTheory.FinCategory K
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape K (J.plusFunctor D)
  -/
  have : ReflectsLimitsOfShape K (forget D) := reflectsLimitsOfShape_of_reflectsIsomorphisms
  /-
    case h
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁷ : CategoryTheory.Category.{max v u, w} D
    inst✝⁶ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝² : CategoryTheory.Limits.HasFiniteLimits D
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type (max v u)
    𝒥✝ : CategoryTheory.SmallCategory K
    x✝ : CategoryTheory.FinCategory K
    this : CategoryTheory.Limits.ReflectsLimitsOfShape K (CategoryTheory.forget D)
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape K (J.plusFunctor D)
  -/
  apply preservesLimitsOfShape_plusFunctor.{w, v, u}
  /-
    🎉 no goals
  -/


instance preservesLimitsOfShape_sheafification
    (K : Type max v u) [SmallCategory K] [FinCategory K] [HasLimitsOfShape K D]
    [PreservesLimitsOfShape K (forget D)] [ReflectsLimitsOfShape K (forget D)] :
    PreservesLimitsOfShape K (J.sheafification D) :=
  Limits.comp_preservesLimitsOfShape _ _


instance preservesFiniteLimits_sheafification
    [HasFiniteLimits D] [PreservesFiniteLimits (forget D)] [(forget D).ReflectsIsomorphisms] :
    PreservesFiniteLimits (J.sheafification D) :=
  Limits.comp_preservesFiniteLimits _ _


instance preservesLimitsOfShape_presheafToSheaf :
    PreservesLimitsOfShape K (plusPlusSheaf J D) := by
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝² : CategoryTheory.SmallCategory K
    inst✝¹ : CategoryTheory.FinCategory K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.plusPlusSheaf …
  -/
  let e := (FinCategory.equivAsType K).symm.trans (AsSmall.equiv.{0, 0, max v u})
  haveI : HasLimitsOfShape (AsSmall.{max v u} (FinCategory.AsType K)) D :=
    Limits.hasLimitsOfShape_of_equivalence e
  haveI : FinCategory (AsSmall.{max v u} (FinCategory.AsType K)) := by
    constructor
    · show Fintype (ULift _)
      infer_instance
    · intro j j'
      show Fintype (ULift _)
      infer_instance
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝² : CategoryTheory.SmallCategory K
    inst✝¹ : CategoryTheory.FinCategory K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    e : CategoryTheory.Equivalence K (CategoryTheory.AsSmall (CategoryTheory.FinCa …
    this✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.AsSmall (Catego …
    this : CategoryTheory.FinCategory (CategoryTheory.AsSmall (CategoryTheory.FinC …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape K (CategoryTheory.plusPlusSheaf …
  -/
  refine @preservesLimitsOfShape_of_equiv _ _ _ _ _ _ _ _ e.symm _ (show _ from ?_)
  /-
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝² : CategoryTheory.SmallCategory K
    inst✝¹ : CategoryTheory.FinCategory K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    e : CategoryTheory.Equivalence K (CategoryTheory.AsSmall (CategoryTheory.FinCa …
    this✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.AsSmall (Catego …
    this : CategoryTheory.FinCategory (CategoryTheory.AsSmall (CategoryTheory.FinC …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.AsSmall (Catego …
  -/
  constructor; intro F; constructor; intro S hS; constructor
  /-
    case preservesLimit.preserves.val
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝² : CategoryTheory.SmallCategory K
    inst✝¹ : CategoryTheory.FinCategory K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    e : CategoryTheory.Equivalence K (CategoryTheory.AsSmall (CategoryTheory.FinCa …
    this✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.AsSmall (Catego …
    this : CategoryTheory.FinCategory (CategoryTheory.AsSmall (CategoryTheory.FinC …
    F : CategoryTheory.Functor (CategoryTheory.AsSmall (CategoryTheory.FinCategory …
    S : CategoryTheory.Limits.Cone F
    hS : CategoryTheory.Limits.IsLimit S
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.plusPlusSheaf J D).mapCone S)
  -/
  apply isLimitOfReflects (sheafToPresheaf J D)
  have : ReflectsLimitsOfShape (AsSmall.{max v u} (FinCategory.AsType K)) (forget D) :=
    reflectsLimitsOfShape_of_reflectsIsomorphisms
  /-
    case preservesLimit.preserves.val.t
    C : Type u
    inst✝¹⁰ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁹ : CategoryTheory.Category.{max v u, w} D
    inst✝⁸ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁷ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁶ : CategoryTheory.ConcreteCategory D
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝² : CategoryTheory.SmallCategory K
    inst✝¹ : CategoryTheory.FinCategory K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    e : CategoryTheory.Equivalence K (CategoryTheory.AsSmall (CategoryTheory.FinCa …
    this✝¹ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.AsSmall (Categ …
    this✝ : CategoryTheory.FinCategory (CategoryTheory.AsSmall (CategoryTheory.Fin …
    F : CategoryTheory.Functor (CategoryTheory.AsSmall (CategoryTheory.FinCategory …
    S : CategoryTheory.Limits.Cone F
    hS : CategoryTheory.Limits.IsLimit S
    this : CategoryTheory.Limits.ReflectsLimitsOfShape (CategoryTheory.AsSmall (Ca …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.sheafToPresheaf J D).mapCone  …
  -/
  apply isLimitOfPreserves (J.sheafification D) hS
  /-
    🎉 no goals
  -/


instance preservesfiniteLimits_presheafToSheaf [HasFiniteLimits D] :
    PreservesFiniteLimits (plusPlusSheaf J D) := by
  /-
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹⁰ : CategoryTheory.Category.{max v u, w} D
    inst✝⁹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁸ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝⁴ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝³ : CategoryTheory.SmallCategory K
    inst✝² : CategoryTheory.FinCategory K
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝ : CategoryTheory.Limits.HasFiniteLimits D
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.plusPlusSheaf J D)
  -/
  apply preservesFiniteLimits_of_preservesFiniteLimitsOfSize.{max v u}
  /-
    case h
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹⁰ : CategoryTheory.Category.{max v u, w} D
    inst✝⁹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁸ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝⁴ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝³ : CategoryTheory.SmallCategory K
    inst✝² : CategoryTheory.FinCategory K
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝ : CategoryTheory.Limits.HasFiniteLimits D
    ⊢ ∀ (J_1 : Type (max v u)) {𝒥 : CategoryTheory.SmallCategory J_1}, CategoryThe …
  -/
  intros
  /-
    case h
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹⁰ : CategoryTheory.Category.{max v u, w} D
    inst✝⁹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁸ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝⁴ : (CategoryTheory.forget D).ReflectsIsomorphisms
    K : Type w'
    inst✝³ : CategoryTheory.SmallCategory K
    inst✝² : CategoryTheory.FinCategory K
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape K D
    inst✝ : CategoryTheory.Limits.HasFiniteLimits D
    J✝ : Type (max v u)
    𝒥✝ : CategoryTheory.SmallCategory J✝
    x✝ : CategoryTheory.FinCategory J✝
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J✝ (CategoryTheory.plusPlusShea …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- `plusPlusSheaf` is isomorphic to an arbitrary choice of left adjoint. -/
def plusPlusSheafIsoPresheafToSheaf : plusPlusSheaf J D ≅ presheafToSheaf J D :=
  (plusPlusAdjunction J D).leftAdjointUniq (sheafificationAdjunction J D)


/-- `plusPlusFunctor` is isomorphic to `sheafification`. -/
def plusPlusFunctorIsoSheafification : J.sheafification D ≅ sheafification J D :=
  isoWhiskerRight (plusPlusSheafIsoPresheafToSheaf J D) (sheafToPresheaf J D)


/-- `plusPlus` is isomorphic to `sheafify`. -/
def plusPlusIsoSheafify (P : Cᵒᵖ ⥤ D) : J.sheafify P ≅ sheafify J P :=
  (sheafToPresheaf J D).mapIso ((plusPlusSheafIsoPresheafToSheaf J D).app P)


@[reassoc (attr := simp)]
lemma toSheafify_plusPlusIsoSheafify_hom (P : Cᵒᵖ ⥤ D) :
    J.toSheafify P ≫ (plusPlusIsoSheafify J D P).hom = toSheafify J P := by
  convert Adjunction.unit_leftAdjointUniq_hom_app
    (plusPlusAdjunction J D) (sheafificationAdjunction J D) P
  /-
    case h.e'_2.h.h.e'_6.h.h.e
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    e_1✝ : Eq (Quiver.Hom P (CategoryTheory.sheafify J P)) (Quiver.Hom ((CategoryT …
    e_3✝ : Eq P ((CategoryTheory.Functor.id (CategoryTheory.Functor (Opposite C) D …
    e_4✝ : Eq (J.sheafify P) (((CategoryTheory.plusPlusSheaf J D).comp (CategoryTh …
    ⊢ Eq J.toSheafify (CategoryTheory.plusPlusAdjunction J D).unit.app
  -/
  ext1 P
  /-
    case h.e'_2.h.h.e'_6.h.h.e.h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P✝ : CategoryTheory.Functor (Opposite C) D
    e_1✝ : Eq (Quiver.Hom P✝ (CategoryTheory.sheafify J P✝)) (Quiver.Hom ((Categor …
    e_3✝ : Eq P✝ ((CategoryTheory.Functor.id (CategoryTheory.Functor (Opposite C)  …
    e_4✝ : Eq (J.sheafify P✝) (((CategoryTheory.plusPlusSheaf J D).comp (CategoryT …
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.toSheafify P) ((CategoryTheory.plusPlusAdjunction J D).unit.app P)
  -/
  dsimp [GrothendieckTopology.toSheafify, plusPlusAdjunction]
  /-
    case h.e'_2.h.h.e'_6.h.h.e.h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P✝ : CategoryTheory.Functor (Opposite C) D
    e_1✝ : Eq (Quiver.Hom P✝ (CategoryTheory.sheafify J P✝)) (Quiver.Hom ((Categor …
    e_3✝ : Eq P✝ ((CategoryTheory.Functor.id (CategoryTheory.Functor (Opposite C)  …
    e_4✝ : Eq (J.sheafify P✝) (((CategoryTheory.plusPlusSheaf J D).comp (CategoryT …
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plusMap (J.toPlus P)) …
  -/
  rw [Category.comp_id]
  /-
    🎉 no goals
  -/


instance [HasFiniteLimits D] : HasSheafify J D := HasSheafify.mk' J D (plusPlusAdjunction J D)


instance [FinitaryExtensive D] [HasPullbacks D] [HasSheafify J D] :
    FinitaryExtensive (Sheaf J D) :=
  finitaryExtensive_of_reflective (sheafificationAdjunction _ _)


instance [Adhesive D] [HasPullbacks D] [HasPushouts D] [HasSheafify J D] :
    Adhesive (Sheaf J D) :=
  adhesive_of_reflective (sheafificationAdjunction _ _)


instance SheafOfTypes.finitary_extensive [HasSheafify J (Type w)] :
    FinitaryExtensive (Sheaf J (Type w)) :=
  inferInstance


instance SheafOfTypes.adhesive [HasSheafify J (Type w)] :
    Adhesive (Sheaf J (Type w)) :=
  inferInstance


instance SheafOfTypes.balanced [HasSheafify J (Type w)] :
    Balanced (Sheaf J (Type w)) :=
  inferInstance


