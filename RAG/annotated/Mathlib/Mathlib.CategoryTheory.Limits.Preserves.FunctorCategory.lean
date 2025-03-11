/-- If `X × -` preserves colimits in `D` for any `X : D`, then the product functor `F ⨯ -` for
`F : C ⥤ D` also preserves colimits.

Note this is (mathematically) a special case of the statement that
"if limits commute with colimits in `D`, then they do as well in `C ⥤ D`"
but the story in Lean is a bit more complex, and this statement isn't directly a special case.
That is, even with a formalised proof of the general statement, there would still need to be some
work to convert to this version: namely, the natural isomorphism
`(evaluation C D).obj k ⋙ prod.functor.obj (F.obj k) ≅
  prod.functor.obj F ⋙ (evaluation C D).obj k`
-/
lemma FunctorCategory.prod_preservesColimits [HasBinaryProducts D] [HasColimits D]
    [∀ X : D, PreservesColimits (prod.functor.obj X)] (F : C ⥤ D) :
    PreservesColimits (prod.functor.obj F) where
  preservesColimitsOfShape {J : Type u} [Category.{u, u} J] :=
    {
      preservesColimit := fun {K : J ⥤ C ⥤ D} => ({
          preserves := fun {c : Cocone K} (t : IsColimit c) => ⟨by
            /-
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v₁, u} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{u, u₂} D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
              inst✝² : CategoryTheory.Limits.HasColimits D
              inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
              F : CategoryTheory.Functor C D
              J : Type u
              inst✝ : CategoryTheory.Category.{u, u} J
              K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
              c : CategoryTheory.Limits.Cocone K
              t : CategoryTheory.Limits.IsColimit c
              ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.prod.functor.obj F). …
            -/
            apply evaluationJointlyReflectsColimits _ fun {k} => ?_
            /-
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v₁, u} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{u, u₂} D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
              inst✝² : CategoryTheory.Limits.HasColimits D
              inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
              F : CategoryTheory.Functor C D
              J : Type u
              inst✝ : CategoryTheory.Category.{u, u} J
              K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
              c : CategoryTheory.Limits.Cocone K
              t : CategoryTheory.Limits.IsColimit c
              k : C
              ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation C D).obj k).map …
            -/
            change IsColimit ((prod.functor.obj F ⋙ (evaluation _ _).obj k).mapCocone c)
            let this :=
              isColimitOfPreserves ((evaluation C D).obj k ⋙ prod.functor.obj (F.obj k)) t
            /-
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v₁, u} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{u, u₂} D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
              inst✝² : CategoryTheory.Limits.HasColimits D
              inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
              F : CategoryTheory.Functor C D
              J : Type u
              inst✝ : CategoryTheory.Category.{u, u} J
              K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
              c : CategoryTheory.Limits.Cocone K
              t : CategoryTheory.Limits.IsColimit c
              k : C
              this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
              ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.Limits.prod.functor.obj F) …
            -/
            apply IsColimit.mapCoconeEquiv _ this
            /-
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v₁, u} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{u, u₂} D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
              inst✝² : CategoryTheory.Limits.HasColimits D
              inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
              F : CategoryTheory.Functor C D
              J : Type u
              inst✝ : CategoryTheory.Category.{u, u} J
              K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
              c : CategoryTheory.Limits.Cocone K
              t : CategoryTheory.Limits.IsColimit c
              k : C
              this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
              ⊢ CategoryTheory.Iso (((CategoryTheory.evaluation C D).obj k).comp (CategoryTh …
            -/
            apply (NatIso.ofComponents _ _).symm
              /-
                C : Type u
                inst✝⁵ : CategoryTheory.Category.{v₁, u} C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{u, u₂} D
                inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
                inst✝² : CategoryTheory.Limits.HasColimits D
                inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
                F : CategoryTheory.Functor C D
                J : Type u
                inst✝ : CategoryTheory.Category.{u, u} J
                K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
                c : CategoryTheory.Limits.Cocone K
                t : CategoryTheory.Limits.IsColimit c
                k : C
                this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
                ⊢ (X : CategoryTheory.Functor C D) → CategoryTheory.Iso (((CategoryTheory.Limi …
              -/
            · intro G
              /-
                C : Type u
                inst✝⁵ : CategoryTheory.Category.{v₁, u} C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{u, u₂} D
                inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
                inst✝² : CategoryTheory.Limits.HasColimits D
                inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
                F : CategoryTheory.Functor C D
                J : Type u
                inst✝ : CategoryTheory.Category.{u, u} J
                K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
                c : CategoryTheory.Limits.Cocone K
                t : CategoryTheory.Limits.IsColimit c
                k : C
                this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
                G : CategoryTheory.Functor C D
                ⊢ CategoryTheory.Iso (((CategoryTheory.Limits.prod.functor.obj F).comp ((Categ …
              -/
              apply asIso (prodComparison ((evaluation C D).obj k) F G)
              /-
                🎉 no goals
              -/
              /-
                C : Type u
                inst✝⁵ : CategoryTheory.Category.{v₁, u} C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{u, u₂} D
                inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
                inst✝² : CategoryTheory.Limits.HasColimits D
                inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
                F : CategoryTheory.Functor C D
                J : Type u
                inst✝ : CategoryTheory.Category.{u, u} J
                K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
                c : CategoryTheory.Limits.Cocone K
                t : CategoryTheory.Limits.IsColimit c
                k : C
                this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
                ⊢ ∀ {X Y : CategoryTheory.Functor C D} (f : Quiver.Hom X Y), Eq (CategoryTheor …
              -/
            · intro G G'
              /-
                C : Type u
                inst✝⁵ : CategoryTheory.Category.{v₁, u} C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{u, u₂} D
                inst✝³ : CategoryTheory.Limits.HasBinaryProducts D
                inst✝² : CategoryTheory.Limits.HasColimits D
                inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimits (CategoryTheory.Li …
                F : CategoryTheory.Functor C D
                J : Type u
                inst✝ : CategoryTheory.Category.{u, u} J
                K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
                c : CategoryTheory.Limits.Cocone K
                t : CategoryTheory.Limits.IsColimit c
                k : C
                this : CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj  …
                G G' : CategoryTheory.Functor C D
                ⊢ ∀ (f : Quiver.Hom G G'), Eq (CategoryTheory.CategoryStruct.comp (((CategoryT …
              -/
              apply prodComparison_natural ((evaluation C D).obj k) (𝟙 F)⟩ } ) }
              /-
                🎉 no goals
              -/


instance whiskeringLeft_preservesLimitsOfShape (J : Type u) [Category.{v} J]
    [HasLimitsOfShape J D] (F : C ⥤ E) :
    PreservesLimitsOfShape J ((whiskeringLeft C E D).obj F) :=
  ⟨fun {K} =>
    ⟨fun c {hc} => ⟨by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.whiskeringLeft C E D).obj F) …
      -/
      apply evaluationJointlyReflectsLimits
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ (k : C) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation C D).ob …
      -/
      intro Y
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        Y : C
        ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation C D).obj Y).mapCo …
      -/
      change IsLimit (((evaluation E D).obj (F.obj Y)).mapCone c)
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        Y : C
        ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation E D).obj (F.obj Y …
      -/
      exact isLimitOfPreserves _ hc⟩⟩⟩
      /-
        🎉 no goals
      -/


instance whiskeringLeft_preservesColimitsOfShape (J : Type u) [Category.{v} J]
    [HasColimitsOfShape J D] (F : C ⥤ E) :
    PreservesColimitsOfShape J ((whiskeringLeft C E D).obj F) :=
  ⟨fun {K} =>
    ⟨fun c {hc} => ⟨by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.whiskeringLeft C E D).obj  …
      -/
      apply evaluationJointlyReflectsColimits
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ (k : C) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation C D). …
      -/
      intro Y
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        Y : C
        ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation C D).obj Y).map …
      -/
      change IsColimit (((evaluation E D).obj (F.obj Y)).mapCocone c)
      /-
        case t
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} E
        J : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} J
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor C E
        K : CategoryTheory.Functor J (CategoryTheory.Functor E D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        Y : C
        ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation E D).obj (F.obj …
      -/
      exact isColimitOfPreserves _ hc⟩⟩⟩
      /-
        🎉 no goals
      -/


instance whiskeringLeft_preservesLimits [HasLimitsOfSize.{w, w'} D] (F : C ⥤ E) :
    PreservesLimitsOfSize.{w, w'} ((whiskeringLeft C E D).obj F) :=
  ⟨fun {J} _ => whiskeringLeft_preservesLimitsOfShape J F⟩


instance whiskeringLeft_preservesColimit [HasColimitsOfSize.{w, w'} D] (F : C ⥤ E) :
    PreservesColimitsOfSize.{w, w'} ((whiskeringLeft C E D).obj F) :=
  ⟨fun {J} _ => whiskeringLeft_preservesColimitsOfShape J F⟩


instance whiskeringRight_preservesLimitsOfShape {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasLimitsOfShape J D] (F : D ⥤ E) [PreservesLimitsOfShape J F] :
    PreservesLimitsOfShape J ((whiskeringRight C D E).obj F) :=
  ⟨fun {K} =>
    ⟨fun c {hc} => ⟨by
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.whiskeringRight C D E).obj F …
      -/
      apply evaluationJointlyReflectsLimits _ (fun k => ?_)
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        k : C
        ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation C E).obj k).mapCo …
      -/
      change IsLimit (((evaluation _ _).obj k ⋙ F).mapCone c)
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cone K
        hc : CategoryTheory.Limits.IsLimit c
        k : C
        ⊢ CategoryTheory.Limits.IsLimit ((((CategoryTheory.evaluation C D).obj k).comp …
      -/
      exact isLimitOfPreserves _ hc⟩⟩⟩
      /-
        🎉 no goals
      -/


/-- Whiskering right and then taking a limit is the same as taking the limit and applying the
    functor. -/
def limitCompWhiskeringRightIsoLimitComp {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasLimitsOfShape J D] (F : D ⥤ E) [PreservesLimitsOfShape J F] (G : J ⥤ C ⥤ D) :
    limit (G ⋙ (whiskeringRight _ _ _).obj F) ≅ limit G ⋙ F :=
  (preservesLimitIso _ _).symm


@[reassoc (attr := simp)]
theorem limitCompWhiskeringRightIsoLimitComp_inv_π {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasLimitsOfShape J D] (F : D ⥤ E) [PreservesLimitsOfShape J F] (G : J ⥤ C ⥤ D) (j : J) :
    (limitCompWhiskeringRightIsoLimitComp F G).inv ≫
      limit.π (G ⋙ (whiskeringRight _ _ _).obj F) j = whiskerRight (limit.π G j) F := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_7, u_3} E
    J : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_4} J
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
    F : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F
    G : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.limitCompWhiskeringRi …
  -/
  simp [limitCompWhiskeringRightIsoLimitComp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limitCompWhiskeringRightIsoLimitComp_hom_whiskerRight_π {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasLimitsOfShape J D] (F : D ⥤ E) [PreservesLimitsOfShape J F] (G : J ⥤ C ⥤ D) (j : J) :
    (limitCompWhiskeringRightIsoLimitComp F G).hom ≫ whiskerRight (limit.π G j) F =
      limit.π (G ⋙ (whiskeringRight _ _ _).obj F) j := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_7, u_3} E
    J : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_4} J
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
    F : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F
    G : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.limitCompWhiskeringRi …
  -/
  simp [← Iso.eq_inv_comp]
  /-
    🎉 no goals
  -/


instance whiskeringRight_preservesColimitsOfShape {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasColimitsOfShape J D] (F : D ⥤ E) [PreservesColimitsOfShape J F] :
    PreservesColimitsOfShape J ((whiskeringRight C D E).obj F) :=
  ⟨fun {K} =>
    ⟨fun c {hc} => ⟨by
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.whiskeringRight C D E).obj …
      -/
      apply evaluationJointlyReflectsColimits _ (fun k => ?_)
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        k : C
        ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation C E).obj k).map …
      -/
      change IsColimit (((evaluation _ _).obj k ⋙ F).mapCocone c)
      /-
        C✝ : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C✝
        D✝ : Type u₂
        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D✝
        E✝ : Type u₃
        inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E✝
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{u_7, u_3} E
        J : Type u_4
        inst✝² : CategoryTheory.Category.{u_8, u_4} J
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        F : CategoryTheory.Functor D E
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J F
        K : CategoryTheory.Functor J (CategoryTheory.Functor C D)
        c : CategoryTheory.Limits.Cocone K
        hc : CategoryTheory.Limits.IsColimit c
        k : C
        ⊢ CategoryTheory.Limits.IsColimit ((((CategoryTheory.evaluation C D).obj k).co …
      -/
      exact isColimitOfPreserves _ hc⟩⟩⟩
      /-
        🎉 no goals
      -/


/-- Whiskering right and then taking a colimit is the same as taking the colimit and applying the
    functor. -/
def colimitCompWhiskeringRightIsoColimitComp {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasColimitsOfShape J D] (F : D ⥤ E) [PreservesColimitsOfShape J F] (G : J ⥤ C ⥤ D) :
    colimit (G ⋙ (whiskeringRight _ _ _).obj F) ≅ colimit G ⋙ F :=
  (preservesColimitIso _ _).symm


@[reassoc (attr := simp)]
theorem ι_colimitCompWhiskeringRightIsoColimitComp_hom {C : Type*} [Category C] {D : Type*}
    [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasColimitsOfShape J D] (F : D ⥤ E) [PreservesColimitsOfShape J F] (G : J ⥤ C ⥤ D) (j : J) :
    colimit.ι (G ⋙ (whiskeringRight _ _ _).obj F) j ≫
      (colimitCompWhiskeringRightIsoColimitComp F G).hom = whiskerRight (colimit.ι G j) F := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_7, u_3} E
    J : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_4} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
    F : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J F
    G : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (G.c …
  -/
  simp [colimitCompWhiskeringRightIsoColimitComp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_ι_colimitCompWhiskeringRightIsoColimitComp_inv {C : Type*} [Category C]
    {D : Type*} [Category D] {E : Type*} [Category E] {J : Type*} [Category J]
    [HasColimitsOfShape J D] (F : D ⥤ E) [PreservesColimitsOfShape J F] (G : J ⥤ C ⥤ D) (j : J) :
    whiskerRight (colimit.ι G j) F ≫ (colimitCompWhiskeringRightIsoColimitComp F G).inv =
      colimit.ι (G ⋙ (whiskeringRight _ _ _).obj F) j := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_7, u_3} E
    J : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_4} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
    F : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J F
    G : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Categor …
  -/
  simp [Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


instance whiskeringRightPreservesLimits {C : Type*} [Category C] {D : Type*} [Category D]
    {E : Type*} [Category E] (F : D ⥤ E) [HasLimitsOfSize.{w, w'} D]
    [PreservesLimitsOfSize.{w, w'} F] :
    PreservesLimitsOfSize.{w, w'} ((whiskeringRight C D E).obj F) :=
  ⟨inferInstance⟩


instance whiskeringRightPreservesColimits {C : Type*} [Category C] {D : Type*} [Category D]
    {E : Type*} [Category E] (F : D ⥤ E) [HasColimitsOfSize.{w, w'} D]
    [PreservesColimitsOfSize.{w, w'} F] :
    PreservesColimitsOfSize.{w, w'} ((whiskeringRight C D E).obj F) :=
  ⟨inferInstance⟩

-- Porting note: fixed spelling mistake in def

/-- If `Lan F.op : (Cᵒᵖ ⥤ Type*) ⥤ (Dᵒᵖ ⥤ Type*)` preserves limits of shape `J`, so will `F`. -/
lemma preservesLimit_of_lan_preservesLimit {C D : Type u} [SmallCategory C]
    [SmallCategory D] (F : C ⥤ D) (J : Type u) [SmallCategory J]
    [PreservesLimitsOfShape J (F.op.lan : _ ⥤ Dᵒᵖ ⥤ Type u)] : PreservesLimitsOfShape J F := by
  /-
    C D : Type u
    inst✝³ : CategoryTheory.SmallCategory C
    inst✝² : CategoryTheory.SmallCategory D
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F.op.lan
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
  -/
  apply @preservesLimitsOfShape_of_reflects_of_preserves _ _ _ _ _ _ _ _ F yoneda ?_
  /-
    C D : Type u
    inst✝³ : CategoryTheory.SmallCategory C
    inst✝² : CategoryTheory.SmallCategory D
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J F.op.lan
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp CategoryTheory.yoneda)
  -/
  exact preservesLimitsOfShape_of_natIso (Presheaf.compYonedaIsoYonedaCompLan F).symm
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ D ⥤ E` preserves finite limits if it does for each `d : D`. -/
lemma preservesFiniteLimits_of_evaluation {D : Type*} [Category D] {E : Type*} [Category E]
    (F : C ⥤ D ⥤ E) (h : ∀ d : D, PreservesFiniteLimits (F ⋙ (evaluation D E).obj d)) :
    PreservesFiniteLimits F :=
  ⟨fun J _ _ => preservesLimitsOfShape_of_evaluation F J fun k => (h k).preservesFiniteLimits _⟩


/-- `F : C ⥤ D ⥤ E` preserves finite limits if it does for each `d : D`. -/
lemma preservesFiniteColimits_of_evaluation {D : Type*} [Category D] {E : Type*} [Category E]
    (F : C ⥤ D ⥤ E) (h : ∀ d : D, PreservesFiniteColimits (F ⋙ (evaluation D E).obj d)) :
    PreservesFiniteColimits F :=
  ⟨fun J _ _ => preservesColimitsOfShape_of_evaluation F J fun k => (h k).preservesFiniteColimits _⟩


noncomputable instance : PreservesLimitsOfShape J (colim : (K ⥤ D ⥤ C) ⥤ _) :=
  preservesLimitsOfShape_of_evaluation _ _ (fun d =>
    let i : (colim : (K ⥤ D ⥤ C) ⥤ _) ⋙ (evaluation D C).obj d ≅
        colimit ((whiskeringRight K (D ⥤ C) C).obj ((evaluation D C).obj d)).flip :=
      NatIso.ofComponents (fun X => (colimitObjIsoColimitCompEvaluation _ _) ≪≫
              /-
                C : Type u
                inst✝⁶ : CategoryTheory.Category.{v, u} C
                J : Type u₁
                inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                D : Type u₃
                inst✝³ : CategoryTheory.Category.{v₃, u₃} D
                inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
                inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
                inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J CategoryTheory.Limits.c …
                d : D
                X : CategoryTheory.Functor K (CategoryTheory.Functor D C)
                ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (X.comp ((CategoryTheory.e …
              -/
          (by exact HasColimit.isoOfNatIso (Iso.refl _)) ≪≫
              /-
                🎉 no goals
              -/
          (colimitObjIsoColimitCompEvaluation _ _).symm)
                                                     /-
                                                       C : Type u
                                                       inst✝⁶ : CategoryTheory.Category.{v, u} C
                                                       J : Type u₁
                                                       inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                                       K : Type u₂
                                                       inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                                       D : Type u₃
                                                       inst✝³ : CategoryTheory.Category.{v₃, u₃} D
                                                       inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
                                                       inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
                                                       inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J CategoryTheory.Limits.c …
                                                       d : D
                                                       F G : CategoryTheory.Functor K (CategoryTheory.Functor D C)
                                                       η : Quiver.Hom F G
                                                       j : K
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.ι F j …
                                                     -/
        (fun {F G} η => colimit_obj_ext (fun j => by simp [← NatTrans.comp_app_assoc]))
                                                     /-
                                                       🎉 no goals
                                                     -/
    preservesLimitsOfShape_of_natIso (i ≪≫ colimitFlipIsoCompColim _).symm)


noncomputable instance : PreservesColimitsOfShape J (lim : (K ⥤ D ⥤ C) ⥤ _) :=
  preservesColimitsOfShape_of_evaluation _ _ (fun d =>
    let i : (lim : (K ⥤ D ⥤ C) ⥤ _) ⋙ (evaluation D C).obj d ≅
        limit ((whiskeringRight K (D ⥤ C) C).obj ((evaluation D C).obj d)).flip :=
      NatIso.ofComponents (fun X => (limitObjIsoLimitCompEvaluation _ _) ≪≫
              /-
                C : Type u
                inst✝⁶ : CategoryTheory.Category.{v, u} C
                J : Type u₁
                inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                D : Type u₃
                inst✝³ : CategoryTheory.Category.{v₃, u₃} D
                inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
                inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape K C
                inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J CategoryTheory.Limits …
                d : D
                X : CategoryTheory.Functor K (CategoryTheory.Functor D C)
                ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (X.comp ((CategoryTheory.eva …
              -/
          (by exact HasLimit.isoOfNatIso (Iso.refl _)) ≪≫
              /-
                🎉 no goals
              -/
          (limitObjIsoLimitCompEvaluation _ _).symm)
                                                   /-
                                                     C : Type u
                                                     inst✝⁶ : CategoryTheory.Category.{v, u} C
                                                     J : Type u₁
                                                     inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                                     K : Type u₂
                                                     inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                                     D : Type u₃
                                                     inst✝³ : CategoryTheory.Category.{v₃, u₃} D
                                                     inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
                                                     inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape K C
                                                     inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J CategoryTheory.Limits …
                                                     d : D
                                                     F G : CategoryTheory.Functor K (CategoryTheory.Functor D C)
                                                     η : Quiver.Hom F G
                                                     j : K
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                   -/
        (fun {F G} η => limit_obj_ext (fun j => by simp [← NatTrans.comp_app]))
                                                   /-
                                                     🎉 no goals
                                                   -/
    preservesColimitsOfShape_of_natIso (i ≪≫ limitFlipIsoCompLim _).symm)


