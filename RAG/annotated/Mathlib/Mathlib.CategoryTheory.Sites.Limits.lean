/-- An auxiliary definition to be used below.

Whenever `E` is a cone of shape `K` of sheaves, and `S` is the multifork associated to a
covering `W` of an object `X`, with respect to the cone point `E.X`, this provides a cone of
shape `K` of objects in `D`, with cone point `S.X`.

See `isLimitMultiforkOfIsLimit` for more on how this definition is used.
-/
def multiforkEvaluationCone (F : K ⥤ Sheaf J D) (E : Cone (F ⋙ sheafToPresheaf J D)) (X : C)
    (W : J.Cover X) (S : Multifork (W.index E.pt)) :
    Cone (F ⋙ sheafToPresheaf J D ⋙ (evaluation Cᵒᵖ D).obj (op X)) where
  pt := S.pt
  π :=
    { app := fun k => (Presheaf.isLimitOfIsSheaf J (F.obj k).1 W (F.obj k).2).lift <|
        Multifork.ofι _ S.pt (fun i => S.ι i ≫ (E.π.app k).app (op i.Y))
          (by
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              ⊢ ∀ (b : (W.index (F.obj k).val).R), Eq (CategoryTheory.CategoryStruct.comp (( …
            -/
            intro i
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => CategoryTheory.CategoryStr …
            -/
            simp only [Category.assoc]
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.ι ((W.index (F.obj k).val).fstTo i …
            -/
            erw [← (E.π.app k).naturality, ← (E.π.app k).naturality]
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.ι ((W.index (F.obj k).val).fstTo i …
            -/
            dsimp
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.ι i.fst) (CategoryTheory.CategoryS …
            -/
            simp only [← Category.assoc]
            /-
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
            congr 1
            /-
              case e_a
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝¹ : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝ : CategoryTheory.Category.{z', z} K
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              X : C
              W : J.Cover X
              S : CategoryTheory.Limits.Multifork (W.index E.pt)
              k : K
              i : (W.index (F.obj k).val).R
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.ι i.fst) (E.pt.map i.r.g₁.op)) (Ca …
            -/
            apply S.condition)
            /-
              🎉 no goals
            -/
      naturality := by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          ⊢ ∀ ⦃X_1 Y : K⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
        -/
        intro i j f
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const K).ob …
        -/
        dsimp [Presheaf.isLimitOfIsSheaf]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S.p …
        -/
        rw [Category.id_comp]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ⊢ Eq (⋯.amalgamate W (fun x => (CategoryTheory.Limits.Multifork.ofι (W.index ( …
        -/
        apply Presheaf.IsSheaf.hom_ext (F.obj j).2 W
        /-
          case h
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ⊢ ∀ (I : W.Arrow), Eq (CategoryTheory.CategoryStruct.comp (⋯.amalgamate W (fun …
        -/
        intro ii
        rw [Presheaf.IsSheaf.amalgamate_map, Category.assoc, ← (F.map f).val.naturality, ←
          Category.assoc, Presheaf.IsSheaf.amalgamate_map]
        /-
          case h
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ii : W.Arrow
          ⊢ Eq ((CategoryTheory.Limits.Multifork.ofι (W.index (F.obj j).val) S.pt (fun i …
        -/
        dsimp [Multifork.ofι]
        /-
          case h
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ii : W.Arrow
          ⊢ Eq (CategoryTheory.Limits.Multifork.ι { pt := S.pt, π := { app := fun x => C …
        -/
        erw [Category.assoc, ← E.w f]
        /-
          case h
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝ : CategoryTheory.Category.{z', z} K
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          X : C
          W : J.Cover X
          S : CategoryTheory.Limits.Multifork (W.index E.pt)
          i j : K
          f : Quiver.Hom i j
          ii : W.Arrow
          ⊢ Eq (CategoryTheory.Limits.Multifork.ι { pt := S.pt, π := { app := fun x => C …
        -/
        aesop_cat }
        /-
          🎉 no goals
        -/


/-- If `E` is a cone of shape `K` of sheaves, which is a limit on the level of presheaves,
this definition shows that the limit presheaf satisfies the multifork variant of the sheaf
condition, at a given covering `W`.

This is used below in `isSheaf_of_isLimit` to show that the limit presheaf is indeed a sheaf.
-/
def isLimitMultiforkOfIsLimit (F : K ⥤ Sheaf J D) (E : Cone (F ⋙ sheafToPresheaf J D))
    (hE : IsLimit E) (X : C) (W : J.Cover X) : IsLimit (W.multifork E.pt) :=
  Multifork.IsLimit.mk _
    (fun S => (isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op X)) hE).lift <|
      multiforkEvaluationCone F E X W S)
    (by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        ⊢ ∀ (E_1 : CategoryTheory.Limits.Multifork (W.index E.pt)) (i : (W.index E.pt) …
      -/
      intro S i
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun S => (CategoryTheory.Limits.isL …
      -/
      apply (isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op i.Y)) hE).hom_ext
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
      -/
      intro k
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp [Multifork.ofι]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      erw [Category.assoc, (E.π.app k).naturality]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.isLimitOfPres …
      -/
      dsimp
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.isLimitOfPres …
      -/
      rw [← Category.assoc]
      erw [(isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op X)) hE).fac
        (multiforkEvaluationCone F E X W S)]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Sheaf.multiforkEvalu …
      -/
      dsimp [multiforkEvaluationCone, Presheaf.isLimitOfIsSheaf]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (⋯.amalgamate W (fun x => (CategoryTh …
      -/
      rw [Presheaf.IsSheaf.amalgamate_map]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        i : (W.index E.pt).L
        k : K
        ⊢ Eq ((CategoryTheory.Limits.Multifork.ofι (W.index (F.obj k).val) S.pt (fun i …
      -/
      rfl)
      /-
        🎉 no goals
      -/
    (by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        ⊢ ∀ (E_1 : CategoryTheory.Limits.Multifork (W.index E.pt)) (m : Quiver.Hom E_1 …
      -/
      intro S m hm
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        ⊢ Eq m ((fun S => (CategoryTheory.Limits.isLimitOfPreserves ((CategoryTheory.e …
      -/
      apply (isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op X)) hE).hom_ext
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ((((CategoryTheory.evalu …
      -/
      intro k
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m ((((CategoryTheory.evaluation (Oppo …
      -/
      dsimp
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m ((E.π.app k).app { unop := X })) (C …
      -/
      erw [(isLimitOfPreserves ((evaluation Cᵒᵖ D).obj (op X)) hE).fac]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m ((E.π.app k).app { unop := X })) (( …
      -/
      apply Presheaf.IsSheaf.hom_ext (F.obj k).2 W
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        ⊢ ∀ (I : W.Arrow), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cate …
      -/
      intro i
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      dsimp only [multiforkEvaluationCone, Presheaf.isLimitOfIsSheaf]
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      rw [(F.obj k).cond.amalgamate_map]
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      dsimp [Multifork.ofι]
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      change _ = S.ι i ≫ _
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      erw [← hm, Category.assoc, ← (E.π.app k).naturality, Category.assoc]
      /-
        case h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝² : CategoryTheory.Category.{w', w} D
        K : Type z
        inst✝¹ : CategoryTheory.Category.{z', z} K
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
        F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
        E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
        hE : CategoryTheory.Limits.IsLimit E
        X : C
        W : J.Cover X
        S : CategoryTheory.Limits.Multifork (W.index E.pt)
        m : Quiver.Hom S.pt (W.multifork E.pt).pt
        hm : ∀ (i : (W.index E.pt).L), Eq (CategoryTheory.CategoryStruct.comp m ((W.mu …
        k : K
        i : W.Arrow
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      rfl)
      /-
        🎉 no goals
      -/


/-- If `E` is a cone which is a limit on the level of presheaves,
then the limit presheaf is again a sheaf.

This is used to show that the forgetful functor from sheaves to presheaves creates limits.
-/
theorem isSheaf_of_isLimit (F : K ⥤ Sheaf J D) (E : Cone (F ⋙ sheafToPresheaf J D))
    (hE : IsLimit E) : Presheaf.IsSheaf J E.pt := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{w', w} D
    K : Type z
    inst✝¹ : CategoryTheory.Category.{z', z} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
    hE : CategoryTheory.Limits.IsLimit E
    ⊢ CategoryTheory.Presheaf.IsSheaf J E.pt
  -/
  rw [Presheaf.isSheaf_iff_multifork]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{w', w} D
    K : Type z
    inst✝¹ : CategoryTheory.Category.{z', z} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
    hE : CategoryTheory.Limits.IsLimit E
    ⊢ ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.multif …
  -/
  intro X S
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{w', w} D
    K : Type z
    inst✝¹ : CategoryTheory.Category.{z', z} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
    F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
    hE : CategoryTheory.Limits.IsLimit E
    X : C
    S : J.Cover X
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork E.pt))
  -/
  exact ⟨isLimitMultiforkOfIsLimit _ _ hE _ _⟩
  /-
    🎉 no goals
  -/


instance (F : K ⥤ Sheaf J D) : CreatesLimit F (sheafToPresheaf J D) :=
  createsLimitOfReflectsIso fun E hE =>
    { liftedCone := ⟨⟨E.pt, isSheaf_of_isLimit _ _ hE⟩,
        ⟨fun _ => ⟨E.π.app _⟩, fun _ _ _ => Sheaf.Hom.ext <| E.π.naturality _⟩⟩
      validLift := Cones.ext (eqToIso rfl) fun j => by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝¹ : CategoryTheory.Category.{z', z} K
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          hE : CategoryTheory.Limits.IsLimit E
          j : K
          ⊢ Eq (((CategoryTheory.sheafToPresheaf J D).mapCone { pt := { val := E.pt, con …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝² : CategoryTheory.Category.{w', w} D
          K : Type z
          inst✝¹ : CategoryTheory.Category.{z', z} K
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
          F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
          E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
          hE : CategoryTheory.Limits.IsLimit E
          j : K
          ⊢ Eq (E.π.app j) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        -/
        simp
        /-
          🎉 no goals
        -/
      makesLimit :=
        { lift := fun S => ⟨hE.lift ((sheafToPresheaf J D).mapCone S)⟩
          fac := fun S j => by
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝² : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝¹ : CategoryTheory.Category.{z', z} K
              inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              hE : CategoryTheory.Limits.IsLimit E
              S : CategoryTheory.Limits.Cone F
              j : K
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun S => { val := hE.lift ((Categor …
            -/
            ext1
            /-
              case h
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝² : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝¹ : CategoryTheory.Category.{z', z} K
              inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              hE : CategoryTheory.Limits.IsLimit E
              S : CategoryTheory.Limits.Cone F
              j : K
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun S => { val := hE.lift ((Categor …
            -/
            apply hE.fac ((sheafToPresheaf J D).mapCone S) j
            /-
              🎉 no goals
            -/
          uniq := fun S m hm => by
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝² : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝¹ : CategoryTheory.Category.{z', z} K
              inst✝ : CategoryTheory.Limits.HasLimitsOfShape K D
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.sheafToPresheaf J D))
              hE : CategoryTheory.Limits.IsLimit E
              S : CategoryTheory.Limits.Cone F
              m : Quiver.Hom S.pt { liftedCone := { pt := { val := E.pt, cond := ⋯ }, π := { …
              hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp m ({ liftedCone := { pt …
              ⊢ Eq m ((fun S => { val := hE.lift ((CategoryTheory.sheafToPresheaf J D).mapCo …
            -/
            ext1
            exact hE.uniq ((sheafToPresheaf J D).mapCone S) m.val fun j =>
              congr_arg Hom.val (hm j) } }


instance createsLimitsOfShape : CreatesLimitsOfShape K (sheafToPresheaf J D) where


instance : HasLimitsOfShape K (Sheaf J D) :=
  hasLimitsOfShape_of_hasLimitsOfShape_createsLimitsOfShape (sheafToPresheaf J D)


instance [HasFiniteProducts D] : HasFiniteProducts (Sheaf J D) :=
  ⟨inferInstance⟩


instance [HasFiniteLimits D] : HasFiniteLimits (Sheaf J D) :=
  ⟨fun _ ↦ inferInstance⟩


instance createsLimits [HasLimitsOfSize.{u₁, u₂} D] :
    CreatesLimitsOfSize.{u₁, u₂} (sheafToPresheaf J D) :=
  ⟨createsLimitsOfShape⟩


instance hasLimitsOfSize [HasLimitsOfSize.{u₁, u₂} D] : HasLimitsOfSize.{u₁, u₂} (Sheaf J D) :=
  hasLimits_of_hasLimits_createsLimits (sheafToPresheaf J D)


/-- Construct a cocone by sheafifying a cocone point of a cocone `E` of presheaves
over a functor which factors through sheaves.
In `isColimitSheafifyCocone`, we show that this is a colimit cocone when `E` is a colimit. -/
noncomputable def sheafifyCocone {F : K ⥤ Sheaf J D}
    (E : Cocone (F ⋙ sheafToPresheaf J D)) : Cocone F :=
  (Cocones.precompose
    (isoWhiskerLeft F (asIso (sheafificationAdjunction J D).counit).symm).hom).obj
    ((presheafToSheaf J D).mapCocone E)


/-- If `E` is a colimit cocone of presheaves, over a diagram factoring through sheaves,
then `sheafifyCocone E` is a colimit cocone. -/
noncomputable def isColimitSheafifyCocone {F : K ⥤ Sheaf J D}
    (E : Cocone (F ⋙ sheafToPresheaf J D)) (hE : IsColimit E) : IsColimit (sheafifyCocone E) :=
  (IsColimit.precomposeHomEquiv _ ((presheafToSheaf J D).mapCocone E)).symm
    (isColimitOfPreserves _ hE)


instance [HasColimitsOfShape K D] : HasColimitsOfShape K (Sheaf J D) :=
  ⟨fun _ => HasColimit.mk
    ⟨sheafifyCocone (colimit.cocone _), isColimitSheafifyCocone _ (colimit.isColimit _)⟩⟩


instance [HasFiniteCoproducts D] : HasFiniteCoproducts (Sheaf J D) :=
  ⟨inferInstance⟩


instance [HasFiniteColimits D] : HasFiniteColimits (Sheaf J D) :=
  ⟨fun _ ↦ inferInstance⟩


instance [HasColimitsOfSize.{u₁, u₂} D] : HasColimitsOfSize.{u₁, u₂} (Sheaf J D) :=
  ⟨inferInstance⟩


/--
If every cocone on a diagram of sheaves which is a colimit on the level of presheaves satisfies
the condition that the cocone point is a sheaf, then the functor from sheaves to preseheaves
creates colimits of the diagram.
Note: this almost never holds in sheaf categories in general, but it does for the extensive
topology (see `Mathlib.CategoryTheory.Sites.Coherent.ExtensiveColimits`).
-/
def createsColimitOfIsSheaf (F : K ⥤ Sheaf J D)
    (h : ∀ (c : Cocone (F ⋙ sheafToPresheaf J D)) (_ : IsColimit c), Presheaf.IsSheaf J c.pt) :
    CreatesColimit F (sheafToPresheaf J D) :=
  createsColimitOfReflectsIso fun E hE =>
    { liftedCocone := ⟨⟨E.pt, h _ hE⟩,
        ⟨fun _ => ⟨E.ι.app _⟩, fun _ _ _ => Sheaf.Hom.ext <| E.ι.naturality _⟩⟩
                                                         /-
                                                           C : Type u
                                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                                           J : CategoryTheory.GrothendieckTopology C
                                                           D : Type w
                                                           inst✝² : CategoryTheory.Category.{w', w} D
                                                           K : Type z
                                                           inst✝¹ : CategoryTheory.Category.{z', z} K
                                                           inst✝ : CategoryTheory.HasWeakSheafify J D
                                                           F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
                                                           h : ∀ (c : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPreshea …
                                                           E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPresheaf J D))
                                                           hE : CategoryTheory.Limits.IsColimit E
                                                           j : K
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.sheafToPresheaf J D …
                                                         -/
      validLift := Cocones.ext (eqToIso rfl) fun j => by simp
                                                         /-
                                                           🎉 no goals
                                                         -/
      makesColimit :=
        { desc := fun S => ⟨hE.desc ((sheafToPresheaf J D).mapCocone S)⟩
                               /-
                                 C : Type u
                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                 J : CategoryTheory.GrothendieckTopology C
                                 D : Type w
                                 inst✝² : CategoryTheory.Category.{w', w} D
                                 K : Type z
                                 inst✝¹ : CategoryTheory.Category.{z', z} K
                                 inst✝ : CategoryTheory.HasWeakSheafify J D
                                 F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
                                 h : ∀ (c : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPreshea …
                                 E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPresheaf J D))
                                 hE : CategoryTheory.Limits.IsColimit E
                                 S : CategoryTheory.Limits.Cocone F
                                 j : K
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ liftedCocone := { pt := { val := E …
                               -/
          fac := fun S j => by ext1; dsimp; rw [hE.fac]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/
          uniq := fun S m hm => by
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              J : CategoryTheory.GrothendieckTopology C
              D : Type w
              inst✝² : CategoryTheory.Category.{w', w} D
              K : Type z
              inst✝¹ : CategoryTheory.Category.{z', z} K
              inst✝ : CategoryTheory.HasWeakSheafify J D
              F : CategoryTheory.Functor K (CategoryTheory.Sheaf J D)
              h : ∀ (c : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPreshea …
              E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.sheafToPresheaf J D))
              hE : CategoryTheory.Limits.IsColimit E
              S : CategoryTheory.Limits.Cocone F
              m : Quiver.Hom { liftedCocone := { pt := { val := E.pt, cond := ⋯ }, ι := { ap …
              hm : ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp ({ liftedCocone := { pt …
              ⊢ Eq m ((fun S => { val := hE.desc ((CategoryTheory.sheafToPresheaf J D).mapCo …
            -/
            ext1
            exact hE.uniq ((sheafToPresheaf J D).mapCocone S) m.val fun j =>
              congr_arg Hom.val (hm j) } }


