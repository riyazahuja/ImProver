/-- Auxiliary definition for `mapTrifunctor`. -/
@[simps]
def mapTrifunctorObj {I₁ : Type*} (X₁ : GradedObject I₁ C₁) (I₂ I₃ : Type*) :
    GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤ GradedObject (I₁ × I₂ × I₃) C₄ where
  obj X₂ :=
    { obj := fun X₃ x => ((F.obj (X₁ x.1)).obj (X₂ x.2.1)).obj (X₃ x.2.2)
      map := fun {_ _} φ x => ((F.obj (X₁ x.1)).obj (X₂ x.2.1)).map (φ x.2.2) }
  map {X₂ Y₂} φ :=
    { app := fun X₃ x => ((F.obj (X₁ x.1)).map (φ x.2.1)).app (X₃ x.2.2) }


/-- Given a trifunctor `F : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` and types `I₁`, `I₂`, `I₃`,
this is the obvious functor
`GradedObject I₁ C₁ ⥤ GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤ GradedObject (I₁ × I₂ × I₃) C₄`.
-/
@[simps]
def mapTrifunctor (I₁ I₂ I₃ : Type*) :
    GradedObject I₁ C₁ ⥤ GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤
      GradedObject (I₁ × I₂ × I₃) C₄ where
  obj X₁ := mapTrifunctorObj F X₁ I₂ I₃
  map {X₁ Y₁} φ :=
    { app := fun X₂ =>
        { app := fun X₃ x => ((F.map (φ x.1)).app (X₂ x.2.1)).app (X₃ x.2.2) }
      naturality := fun {X₂ Y₂} ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.17454, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.17458, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.17462, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.17466, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.17470, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.17474, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.GradedObj …
        -/
        ext X₃ x
        /-
          case w.h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.17454, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.17458, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.17462, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.17466, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.17470, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.17474, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          x : Prod I₁ (Prod I₂ I₃)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.GradedOb …
        -/
        dsimp
        /-
          case w.h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.17454, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.17458, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.17462, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.17466, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.17470, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.17474, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          x : Prod I₁ (Prod I₂ I₃)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.obj (X₁ x.1)).map (ψ x.2.1)).app …
        -/
        simp only [← NatTrans.comp_app]
        /-
          case w.h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.17454, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.17458, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.17462, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.17466, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.17470, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.17474, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          x : Prod I₁ (Prod I₂ I₃)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((F.obj (X₁ x.1)).map (ψ x.2.1)) ((F …
        -/
        congr 1
        /-
          case w.h.h.e_self
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.17454, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.17458, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.17462, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.17466, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.17470, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.17474, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          x : Prod I₁ (Prod I₂ I₃)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (X₁ x.1)).map (ψ x.2.1)) ((F. …
        -/
        rw [NatTrans.naturality] }
        /-
          🎉 no goals
        -/

/-- The natural transformation `mapTrifunctor F I₁ I₂ I₃ ⟶ mapTrifunctor F' I₁ I₂ I₃`
induced by a natural transformation `F ⟶ F` of trifunctors. -/
@[simps]
def mapTrifunctorMapNatTrans (α : F ⟶ F') (I₁ I₂ I₃ : Type*) :
    mapTrifunctor F I₁ I₂ I₃ ⟶ mapTrifunctor F' I₁ I₂ I₃ where
  app X₁ :=
    { app := fun X₂ =>
        { app := fun _ _ => ((α.app _).app _).app _ }
      naturality := fun {X₂ Y₂} φ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          α : Quiver.Hom F F'
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ : CategoryTheory.GradedObject I₁ C₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          φ : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.GradedObject.mapTri …
        -/
        ext X₃ ⟨i₁, i₂, i₃⟩
        /-
          case w.h.h.mk.mk
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          α : Quiver.Hom F F'
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ : CategoryTheory.GradedObject I₁ C₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          φ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          i₁ : I₁
          i₂ : I₂
          i₃ : I₃
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.GradedObject.mapTr …
        -/
        dsimp
        /-
          case w.h.h.mk.mk
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
          α : Quiver.Hom F F'
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          X₁ : CategoryTheory.GradedObject I₁ C₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          φ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          i₁ : I₁
          i₂ : I₂
          i₃ : I₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.obj (X₁ i₁)).map (φ i₂)).app (X₃ …
        -/
        simp only [← NatTrans.comp_app, NatTrans.naturality] }
        /-
          🎉 no goals
        -/
  naturality := fun {X₁ Y₁} φ => by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      α : Quiver.Hom F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
      φ : Quiver.Hom X₁ Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GradedObject.mapTrif …
    -/
    ext X₂ X₃ ⟨i₁, i₂, i₃⟩
    /-
      case w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      α : Quiver.Hom F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
      φ : Quiver.Hom X₁ Y₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp ((CategoryTheory.GradedObject.mapTr …
    -/
    dsimp
    /-
      case w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.35660, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.35664, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.35668, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.35672, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.35676, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.35680, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      α : Quiver.Hom F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
      φ : Quiver.Hom X₁ Y₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.map (φ i₁)).app (X₂ i₂)).app (X₃ …
    -/
    simp only [← NatTrans.comp_app, NatTrans.naturality]
    /-
      🎉 no goals
    -/


/-- The natural isomorphism `mapTrifunctor F I₁ I₂ I₃ ≅ mapTrifunctor F' I₁ I₂ I₃`
induced by a natural isomorphism `F ≅ F` of trifunctors. -/
@[simps]
def mapTrifunctorMapIso (e : F ≅ F') (I₁ I₂ I₃ : Type*) :
    mapTrifunctor F I₁ I₂ I₃ ≅ mapTrifunctor F' I₁ I₂ I₃ where
  hom := mapTrifunctorMapNatTrans e.hom I₁ I₂ I₃
  inv := mapTrifunctorMapNatTrans e.inv I₁ I₂ I₃
  hom_inv_id := by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapTrifu …
    -/
    ext X₁ X₂ X₃ ⟨i₁, i₂, i₃⟩
    /-
      case w.h.w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ : CategoryTheory.GradedObject I₁ C₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq ((((CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapTr …
    -/
    dsimp
    /-
      case w.h.w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ : CategoryTheory.GradedObject I₁ C₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((e.hom.app (X₁ i₁)).app (X₂ i₂)).ap …
    -/
    simp only [← NatTrans.comp_app, e.hom_inv_id, NatTrans.id_app]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapTrifu …
    -/
    ext X₁ X₂ X₃ ⟨i₁, i₂, i₃⟩
    /-
      case w.h.w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ : CategoryTheory.GradedObject I₁ C₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq ((((CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapTr …
    -/
    dsimp
    /-
      case w.h.w.h.w.h.h.mk.mk
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.46844, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.46848, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.46852, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.46856, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.46860, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.46864, u_6} C₂₃
      F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Fu …
      e : CategoryTheory.Iso F F'
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      X₁ : CategoryTheory.GradedObject I₁ C₁
      X₂ : CategoryTheory.GradedObject I₂ C₂
      X₃ : CategoryTheory.GradedObject I₃ C₃
      i₁ : I₁
      i₂ : I₂
      i₃ : I₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((e.inv.app (X₁ i₁)).app (X₂ i₂)).ap …
    -/
    simp only [← NatTrans.comp_app, e.inv_hom_id, NatTrans.id_app]
    /-
      🎉 no goals
    -/


/-- Given a trifunctor `F : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₃`, graded objects `X₁ : GradedObject I₁ C₁`,
`X₂ : GradedObject I₂ C₂`, `X₃ : GradedObject I₃ C₃`, and a map `p : I₁ × I₂ × I₃ → J`,
this is the `J`-graded object sending `j` to the coproduct of
`((F.obj (X₁ i₁)).obj (X₂ i₂)).obj (X₃ i₃)` for `p ⟨i₁, i₂, i₃⟩ = k`. -/
noncomputable def mapTrifunctorMapObj (X₁ : GradedObject I₁ C₁) (X₂ : GradedObject I₂ C₂)
    (X₃ : GradedObject I₃ C₃)
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p] :
    GradedObject J C₄ :=
  ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃).mapObj p


/-- The obvious inclusion
`((F.obj (X₁ i₁)).obj (X₂ i₂)).obj (X₃ i₃) ⟶ mapTrifunctorMapObj F p X₁ X₂ X₃ j` when
`p ⟨i₁, i₂, i₃⟩ = j`. -/
noncomputable def ιMapTrifunctorMapObj (X₁ : GradedObject I₁ C₁) (X₂ : GradedObject I₂ C₂)
    (X₃ : GradedObject I₃ C₃) (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J) (h : p ⟨i₁, i₂, i₃⟩ = j)
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p] :
    ((F.obj (X₁ i₁)).obj (X₂ i₂)).obj (X₃ i₃) ⟶ mapTrifunctorMapObj F p X₁ X₂ X₃ j :=
  ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃).ιMapObj p ⟨i₁, i₂, i₃⟩ j h


/-- The maps `mapTrifunctorMapObj F p X₁ X₂ X₃ ⟶ mapTrifunctorMapObj F p Y₁ Y₂ Y₃` which
express the functoriality of `mapTrifunctorMapObj`, see `mapTrifunctorMap` -/
noncomputable def mapTrifunctorMapMap {X₁ Y₁ : GradedObject I₁ C₁} (f₁ : X₁ ⟶ Y₁)
    {X₂ Y₂ : GradedObject I₂ C₂} (f₂ : X₂ ⟶ Y₂)
    {X₃ Y₃ : GradedObject I₃ C₃} (f₃ : X₃ ⟶ Y₃)
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p]
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj Y₂).obj Y₃) p] :
    mapTrifunctorMapObj F p X₁ X₂ X₃ ⟶ mapTrifunctorMapObj F p Y₁ Y₂ Y₃ :=
  GradedObject.mapMap ((((mapTrifunctor F I₁ I₂ I₃).map f₁).app X₂).app X₃ ≫
    (((mapTrifunctor F I₁ I₂ I₃).obj Y₁).map f₂).app X₃ ≫
    (((mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj Y₂).map f₃) p


@[reassoc (attr := simp)]
lemma ι_mapTrifunctorMapMap {X₁ Y₁ : GradedObject I₁ C₁} (f₁ : X₁ ⟶ Y₁)
    {X₂ Y₂ : GradedObject I₂ C₂} (f₂ : X₂ ⟶ Y₂)
    {X₃ Y₃ : GradedObject I₃ C₃} (f₃ : X₃ ⟶ Y₃)
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p]
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj Y₂).obj Y₃) p]
    (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J) (h : p ⟨i₁, i₂, i₃⟩ = j) :
  ιMapTrifunctorMapObj F p X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ mapTrifunctorMapMap F p f₁ f₂ f₃ j =
    ((F.map (f₁ i₁)).app (X₂ i₂)).app (X₃ i₃) ≫
      ((F.obj (Y₁ i₁)).map (f₂ i₂)).app (X₃ i₃) ≫
      ((F.obj (Y₁ i₁)).obj (Y₂ i₂)).map (f₃ i₃) ≫
      ιMapTrifunctorMapObj F p Y₁ Y₂ Y₃ i₁ i₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_12, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_14, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
    f₁ : Quiver.Hom X₁ Y₁
    X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
    f₂ : Quiver.Hom X₂ Y₂
    X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
    f₃ : Quiver.Hom X₃ Y₃
    inst✝¹ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj …
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
  -/
  dsimp only [ιMapTrifunctorMapObj, mapTrifunctorMapMap]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_12, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_14, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
    f₁ : Quiver.Hom X₁ Y₁
    X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
    f₂ : Quiver.Hom X₂ Y₂
    X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
    f₃ : Quiver.Hom X₃ Y₃
    inst✝¹ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj …
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((((CategoryTheory.GradedObject.mapT …
  -/
  rw [ι_mapMap]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_12, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_14, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
    f₁ : Quiver.Hom X₁ Y₁
    X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
    f₂ : Quiver.Hom X₂ Y₂
    X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
    f₃ : Quiver.Hom X₃ Y₃
    inst✝¹ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj …
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_12, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_14, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
    f₁ : Quiver.Hom X₁ Y₁
    X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
    f₂ : Quiver.Hom X₂ Y₂
    X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
    f₃ : Quiver.Hom X₃ Y₃
    inst✝¹ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj …
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj Y₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, assoc]
  /-
    🎉 no goals
  -/


@[ext]
lemma mapTrifunctorMapObj_ext {X₁ : GradedObject I₁ C₁} {X₂ : GradedObject I₂ C₂}
    {X₃ : GradedObject I₃ C₃} {Y : C₄} (j : J)
    [HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p]
    {φ φ' : mapTrifunctorMapObj F p X₁ X₂ X₃ j ⟶ Y}
    (h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : p ⟨i₁, i₂, i₃⟩ = j),
      ιMapTrifunctorMapObj F p X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ φ =
        ιMapTrifunctorMapObj F p X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ φ') : φ = φ' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝¹ : CategoryTheory.Category.{u_11, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    Y : C₄
    j : J
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj  …
    φ φ' : Quiver.Hom (CategoryTheory.GradedObject.mapTrifunctorMapObj F p X₁ X₂ X …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (p { fst := i₁, snd := { fst := i₂ …
    ⊢ Eq φ φ'
  -/
  apply mapObj_ext
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝¹ : CategoryTheory.Category.{u_11, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    Y : C₄
    j : J
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj  …
    φ φ' : Quiver.Hom (CategoryTheory.GradedObject.mapTrifunctorMapObj F p X₁ X₂ X …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (p { fst := i₁, snd := { fst := i₂ …
    ⊢ ∀ (i : Prod I₁ (Prod I₂ I₃)) (hij : Eq (p i) j), Eq (CategoryTheory.Category …
  -/
  rintro ⟨i₁, i₂, i₃⟩ hi
  /-
    case hfg.mk.mk
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝¹ : CategoryTheory.Category.{u_11, u_4} C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    p : Prod I₁ (Prod I₂ I₃) → J
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    Y : C₄
    j : J
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor F I₁ I₂ I₃).obj X₁).obj  …
    φ φ' : Quiver.Hom (CategoryTheory.GradedObject.mapTrifunctorMapObj F p X₁ X₂ X …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (p { fst := i₁, snd := { fst := i₂ …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    hi : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((((CategoryTheory.GradedObject.mapT …
  -/
  apply h
  /-
    🎉 no goals
  -/


instance (X₁ : GradedObject I₁ C₁) (X₂ : GradedObject I₂ C₂) (X₃ : GradedObject I₃ C₃)
  [h : HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p] :
      HasMap (((mapTrifunctorObj F X₁ I₂ I₃).obj X₂).obj X₃) p := h


/-- Given a trifunctor `F : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄`, a map `p : I₁ × I₂ × I₃ → J`, and
graded objects `X₁ : GradedObject I₁ C₁`, `X₂ : GradedObject I₂ C₂` and `X₃ : GradedObject I₃ C₃`,
this is the `J`-graded object sending `j` to the coproduct of
`((F.obj (X₁ i₁)).obj (X₂ i₂)).obj (X₃ i₃)` for `p ⟨i₁, i₂, i₃⟩ = j`. -/
@[simps]
noncomputable def mapTrifunctorMapFunctorObj (X₁ : GradedObject I₁ C₁)
    [∀ X₂ X₃, HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p] :
    GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤ GradedObject J C₄ where
  obj X₂ :=
    { obj := fun X₃ => mapTrifunctorMapObj F p X₁ X₂ X₃
      map := fun {_ _} φ => mapTrifunctorMapMap F p (𝟙 X₁) (𝟙 X₂) φ
      map_id := fun X₃ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ : CategoryTheory.GradedObject I₂ C₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          ⊢ Eq ({ obj := fun X₃ => CategoryTheory.GradedObject.mapTrifunctorMapObj F p X …
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ : CategoryTheory.GradedObject I₂ C₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          ⊢ Eq (CategoryTheory.GradedObject.mapTrifunctorMapMap F p (CategoryTheory.Cate …
        -/
        ext j i₁ i₂ i₃ h
        simp only [ι_mapTrifunctorMapMap, categoryOfGradedObjects_id, Functor.map_id,
          NatTrans.id_app, id_comp, comp_id]
      map_comp := fun {X₃ Y₃ Z₃} φ ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ : CategoryTheory.GradedObject I₂ C₂
          X₃ Y₃ Z₃ : CategoryTheory.GradedObject I₃ C₃
          φ : Quiver.Hom X₃ Y₃
          ψ : Quiver.Hom Y₃ Z₃
          ⊢ Eq ({ obj := fun X₃ => CategoryTheory.GradedObject.mapTrifunctorMapObj F p X …
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ : CategoryTheory.GradedObject I₂ C₂
          X₃ Y₃ Z₃ : CategoryTheory.GradedObject I₃ C₃
          φ : Quiver.Hom X₃ Y₃
          ψ : Quiver.Hom Y₃ Z₃
          ⊢ Eq (CategoryTheory.GradedObject.mapTrifunctorMapMap F p (CategoryTheory.Cate …
        -/
        ext j i₁ i₂ i₃ h
        simp only [ι_mapTrifunctorMapMap, categoryOfGradedObjects_id, Functor.map_id,
          NatTrans.id_app, categoryOfGradedObjects_comp, Functor.map_comp, assoc, id_comp,
          ι_mapTrifunctorMapMap_assoc] }
  map {X₂ Y₂} φ :=
    { app := fun X₃ => mapTrifunctorMapMap F p (𝟙 X₁) φ (𝟙 X₃)
      naturality := fun {X₃ Y₃} ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          φ : Quiver.Hom X₂ Y₂
          X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
          ψ : Quiver.Hom X₃ Y₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X₂ => { obj := fun X₃ => Categ …
        -/
        ext j i₁ i₂ i₃ h
        /-
          case h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          X₁ : CategoryTheory.GradedObject I₁ C₁
          inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          φ : Quiver.Hom X₂ Y₂
          X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
          ψ : Quiver.Hom X₃ Y₃
          j : J
          i₁ : I₁
          i₂ : I₂
          i₃ : I₃
          h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
        -/
        dsimp
        simp only [ι_mapTrifunctorMapMap_assoc, categoryOfGradedObjects_id, Functor.map_id,
          NatTrans.id_app, ι_mapTrifunctorMapMap, id_comp, NatTrans.naturality_assoc] }
  map_id X₂ := by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
      inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
      inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
      inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
      inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
      inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      J : Type u_10
      p : Prod I₁ (Prod I₂ I₃) → J
      X₁ : CategoryTheory.GradedObject I₁ C₁
      inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
      X₂ : CategoryTheory.GradedObject I₂ C₂
      ⊢ Eq ({ obj := fun X₂ => { obj := fun X₃ => CategoryTheory.GradedObject.mapTri …
    -/
    dsimp
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
      inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
      inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
      inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
      inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
      inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      J : Type u_10
      p : Prod I₁ (Prod I₂ I₃) → J
      X₁ : CategoryTheory.GradedObject I₁ C₁
      inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
      X₂ : CategoryTheory.GradedObject I₂ C₂
      ⊢ Eq { app := fun X₃ => CategoryTheory.GradedObject.mapTrifunctorMapMap F p (C …
    -/
    ext X₃ j i₁ i₂ i₃ h
    simp only [ι_mapTrifunctorMapMap, categoryOfGradedObjects_id, Functor.map_id,
      NatTrans.id_app, id_comp, comp_id]
  map_comp {X₂ Y₂ Z₂} φ ψ := by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
      inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
      inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
      inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
      inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
      inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      J : Type u_10
      p : Prod I₁ (Prod I₂ I₃) → J
      X₁ : CategoryTheory.GradedObject I₁ C₁
      inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
      X₂ Y₂ Z₂ : CategoryTheory.GradedObject I₂ C₂
      φ : Quiver.Hom X₂ Y₂
      ψ : Quiver.Hom Y₂ Z₂
      ⊢ Eq ({ obj := fun X₂ => { obj := fun X₃ => CategoryTheory.GradedObject.mapTri …
    -/
    dsimp
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁶ : CategoryTheory.Category.{?u.68337, u_1} C₁
      inst✝⁵ : CategoryTheory.Category.{?u.68341, u_2} C₂
      inst✝⁴ : CategoryTheory.Category.{?u.68345, u_3} C₃
      inst✝³ : CategoryTheory.Category.{?u.68349, u_4} C₄
      inst✝² : CategoryTheory.Category.{?u.68353, u_5} C₁₂
      inst✝¹ : CategoryTheory.Category.{?u.68357, u_6} C₂₃
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
      I₁ : Type u_7
      I₂ : Type u_8
      I₃ : Type u_9
      J : Type u_10
      p : Prod I₁ (Prod I₂ I₃) → J
      X₁ : CategoryTheory.GradedObject I₁ C₁
      inst✝ : ∀ (X₂ : CategoryTheory.GradedObject I₂ C₂) (X₃ : CategoryTheory.Graded …
      X₂ Y₂ Z₂ : CategoryTheory.GradedObject I₂ C₂
      φ : Quiver.Hom X₂ Y₂
      ψ : Quiver.Hom Y₂ Z₂
      ⊢ Eq { app := fun X₃ => CategoryTheory.GradedObject.mapTrifunctorMapMap F p (C …
    -/
    ext X₃ j i₁ i₂ i₃
    simp only [ι_mapTrifunctorMapMap, categoryOfGradedObjects_id, Functor.map_id,
      NatTrans.id_app, categoryOfGradedObjects_comp, Functor.map_comp, NatTrans.comp_app,
      id_comp, assoc, ι_mapTrifunctorMapMap_assoc]


/-- Given a trifunctor `F : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` and a map `p : I₁ × I₂ × I₃ → J`,
this is the functor
`GradedObject I₁ C₁ ⥤ GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤ GradedObject J C₄`
sending `X₁ : GradedObject I₁ C₁`, `X₂ : GradedObject I₂ C₂` and `X₃ : GradedObject I₃ C₃`
to the `J`-graded object sending `j` to the coproduct of
`((F.obj (X₁ i₁)).obj (X₂ i₂)).obj (X₃ i₃)` for `p ⟨i₁, i₂, i₃⟩ = j`. -/
noncomputable def mapTrifunctorMap
    [∀ X₁ X₂ X₃, HasMap ((((mapTrifunctor F I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) p] :
    GradedObject I₁ C₁ ⥤ GradedObject I₂ C₂ ⥤ GradedObject I₃ C₃ ⥤ GradedObject J C₄ where
  obj X₁ := mapTrifunctorMapFunctorObj F p X₁
  map := fun {X₁ Y₁} φ =>
    { app := fun X₂ =>
        { app := fun X₃ => mapTrifunctorMapMap F p φ (𝟙 X₂) (𝟙 X₃)
          naturality := fun {X₃ Y₃} φ => by
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
              inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
              inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
              inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
              inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
              inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
              I₁ : Type u_7
              I₂ : Type u_8
              I₃ : Type u_9
              J : Type u_10
              p : Prod I₁ (Prod I₂ I₃) → J
              inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
              X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
              φ✝ : Quiver.Hom X₁ Y₁
              X₂ : CategoryTheory.GradedObject I₂ C₂
              X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
              φ : Quiver.Hom X₃ Y₃
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun X₁ => CategoryTheory.GradedOb …
            -/
            dsimp
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
              inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
              inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
              inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
              inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
              inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
              I₁ : Type u_7
              I₂ : Type u_8
              I₃ : Type u_9
              J : Type u_10
              p : Prod I₁ (Prod I₂ I₃) → J
              inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
              X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
              φ✝ : Quiver.Hom X₁ Y₁
              X₂ : CategoryTheory.GradedObject I₂ C₂
              X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
              φ : Quiver.Hom X₃ Y₃
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.GradedObject.mapTri …
            -/
            ext j i₁ i₂ i₃ h
            /-
              case h.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
              inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
              inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
              inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
              inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
              inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
              I₁ : Type u_7
              I₂ : Type u_8
              I₃ : Type u_9
              J : Type u_10
              p : Prod I₁ (Prod I₂ I₃) → J
              inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
              X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
              φ✝ : Quiver.Hom X₁ Y₁
              X₂ : CategoryTheory.GradedObject I₂ C₂
              X₃ Y₃ : CategoryTheory.GradedObject I₃ C₃
              φ : Quiver.Hom X₃ Y₃
              j : J
              i₁ : I₁
              i₂ : I₂
              i₃ : I₃
              h : Eq (p { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
            -/
            dsimp
            simp only [ι_mapTrifunctorMapMap_assoc, categoryOfGradedObjects_id, Functor.map_id,
              NatTrans.id_app, ι_mapTrifunctorMapMap, id_comp, NatTrans.naturality_assoc] }
      naturality := fun {X₂ Y₂} ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.GradedObj …
        -/
        ext X₃ j
        /-
          case w.h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          j : J
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.GradedOb …
        -/
        dsimp
        /-
          case w.h.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁶ : CategoryTheory.Category.{?u.83771, u_1} C₁
          inst✝⁵ : CategoryTheory.Category.{?u.83775, u_2} C₂
          inst✝⁴ : CategoryTheory.Category.{?u.83779, u_3} C₃
          inst✝³ : CategoryTheory.Category.{?u.83783, u_4} C₄
          inst✝² : CategoryTheory.Category.{?u.83787, u_5} C₁₂
          inst✝¹ : CategoryTheory.Category.{?u.83791, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
          I₁ : Type u_7
          I₂ : Type u_8
          I₃ : Type u_9
          J : Type u_10
          p : Prod I₁ (Prod I₂ I₃) → J
          inst✝ : ∀ (X₁ : CategoryTheory.GradedObject I₁ C₁) (X₂ : CategoryTheory.Graded …
          X₁ Y₁ : CategoryTheory.GradedObject I₁ C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : CategoryTheory.GradedObject I₂ C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : CategoryTheory.GradedObject I₃ C₃
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapTrifu …
        -/
        ext i₁ i₂ i₃ h
        simp only [ι_mapTrifunctorMapMap_assoc, categoryOfGradedObjects_id, Functor.map_id,
          NatTrans.id_app, ι_mapTrifunctorMapMap, id_comp,
          NatTrans.naturality_app_assoc] }


attribute [simps] mapTrifunctorMap


/-- Given a map `r : I₁ × I₂ × I₃ → J`, a `BifunctorComp₁₂IndexData r` consists of the data
of a type `I₁₂`, maps `p : I₁ × I₂ → I₁₂` and `q : I₁₂ × I₃ → J`, such that `r` is obtained
by composition of `p` and `q`. -/
structure BifunctorComp₁₂IndexData where
  /-- an auxiliary type -/
  I₁₂ : Type*
  /-- a map `I₁ × I₂ → I₁₂` -/
  p : I₁ × I₂ → I₁₂
  /-- a map `I₁₂ × I₃ → J` -/
  q : I₁₂ × I₃ → J
  hpq (i : I₁ × I₂ × I₃) : q ⟨p ⟨i.1, i.2.1⟩, i.2.2⟩ = r i


/-- Given bifunctors `F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂`, `G : C₁₂ ⥤ C₃ ⥤ C₄`, graded objects
`X₁ : GradedObject I₁ C₁`, `X₂ : GradedObject I₂ C₂`, `X₃ : GradedObject I₃ C₃` and
`ρ₁₂ : BifunctorComp₁₂IndexData r`, this asserts that for all `i₁₂ : ρ₁₂.I₁₂` and `i₃ : I₃`,
the functor `G(-, X₃ i₃)` commutes with the coproducts of the `F₁₂(X₁ i₁, X₂ i₂)`
such that `ρ₁₂.p ⟨i₁, i₂⟩ = i₁₂`. -/
abbrev HasGoodTrifunctor₁₂Obj :=
  ∀ (i₁₂ : ρ₁₂.I₁₂) (i₃ : I₃), PreservesColimit
    (Discrete.functor (mapObjFun (((mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂) ρ₁₂.p i₁₂))
      ((Functor.flip G).obj (X₃ i₃))


/-- The inclusion of `(G.obj ((F₁₂.obj (X₁ i₁)).obj (X₂ i₂))).obj (X₃ i₃)` in
`mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j`
when `r (i₁, i₂, i₃) = j`. -/
noncomputable def ιMapBifunctor₁₂BifunctorMapObj (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    (G.obj ((F₁₂.obj (X₁ i₁)).obj (X₂ i₂))).obj (X₃ i₃) ⟶
      mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j :=
  (G.map (ιMapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂ i₁ i₂ _ rfl)).app (X₃ i₃) ≫
    ιMapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ (ρ₁₂.p ⟨i₁, i₂⟩) i₃ j
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            C₃ : Type u_3
            C₄ : Type u_4
            C₁₂ : Type u_5
            C₂₃ : Type u_6
            inst✝⁷ : CategoryTheory.Category.{?u.127015, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.127019, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.127023, u_3} C₃
            inst✝⁴ : CategoryTheory.Category.{?u.127027, u_4} C₄
            inst✝³ : CategoryTheory.Category.{?u.127031, u_5} C₁₂
            inst✝² : CategoryTheory.Category.{?u.127035, u_6} C₂₃
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            I₁ : Type u_7
            I₂ : Type u_8
            I₃ : Type u_9
            J : Type u_10
            r : Prod I₁ (Prod I₂ I₃) → J
            ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
            X₁ : CategoryTheory.GradedObject I₁ C₁
            X₂ : CategoryTheory.GradedObject I₂ C₂
            X₃ : CategoryTheory.GradedObject I₃ C₃
            inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
            inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
            i₁ : I₁
            i₂ : I₂
            i₃ : I₃
            j : J
            h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            ⊢ Eq (ρ₁₂.q { fst := ρ₁₂.p { fst := i₁, snd := i₂ }, snd := i₃ }) j
          -/
      (by rw [← h, ← ρ₁₂.hpq])
          /-
            🎉 no goals
          -/


@[reassoc]
lemma ιMapBifunctor₁₂BifunctorMapObj_eq (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) (i₁₂ : ρ₁₂.I₁₂) (h₁₂ : ρ₁₂.p ⟨i₁, i₂⟩ = i₁₂) :
    ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h =
      (G.map (ιMapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂ i₁ i₂ i₁₂ h₁₂)).app (X₃ i₃) ≫
    ιMapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ i₁₂ i₃ j
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            C₃ : Type u_3
            C₄ : Type u_4
            C₁₂ : Type u_5
            C₂₃ : Type u_6
            inst✝⁷ : CategoryTheory.Category.{?u.130144, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.130148, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.130152, u_3} C₃
            inst✝⁴ : CategoryTheory.Category.{?u.130156, u_4} C₄
            inst✝³ : CategoryTheory.Category.{?u.130160, u_5} C₁₂
            inst✝² : CategoryTheory.Category.{?u.130164, u_6} C₂₃
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            I₁ : Type u_7
            I₂ : Type u_8
            I₃ : Type u_9
            J : Type u_10
            r : Prod I₁ (Prod I₂ I₃) → J
            ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
            X₁ : CategoryTheory.GradedObject I₁ C₁
            X₂ : CategoryTheory.GradedObject I₂ C₂
            X₃ : CategoryTheory.GradedObject I₃ C₃
            inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
            inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
            i₁ : I₁
            i₂ : I₂
            i₃ : I₃
            j : J
            h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₁₂ : ρ₁₂.I₁₂
            h₁₂ : Eq (ρ₁₂.p { fst := i₁, snd := i₂ }) i₁₂
            ⊢ Eq (ρ₁₂.q { fst := i₁₂, snd := i₃ }) j
          -/
      (by rw [← h₁₂, ← h, ← ρ₁₂.hpq]) := by
          /-
            🎉 no goals
          -/
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_15, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_12, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_14, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    i₁₂ : ρ₁₂.I₁₂
    h₁₂ : Eq (ρ₁₂.p { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.GradedObject.ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁  …
  -/
  subst h₁₂
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_15, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_13, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_12, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_14, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.GradedObject.ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The cofan consisting of the inclusions given by `ιMapBifunctor₁₂BifunctorMapObj`. -/
noncomputable def cofan₃MapBifunctor₁₂BifunctorMapObj (j : J) :
    ((((mapTrifunctor (bifunctorComp₁₂ F₁₂ G) I₁ I₂ I₃).obj X₁).obj X₂).obj
      X₃).CofanMapObjFun r j :=
  Cofan.mk (mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j)
    (fun ⟨⟨i₁, i₂, i₃⟩, (hi : r ⟨i₁, i₂, i₃⟩ = j)⟩ =>
      ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j hi)


/-- The cofan `cofan₃MapBifunctor₁₂BifunctorMapObj` is a colimit, see the induced isomorphism
`mapBifunctorComp₁₂MapObjIso`. -/
noncomputable def isColimitCofan₃MapBifunctor₁₂BifunctorMapObj (j : J) :
    IsColimit (cofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let c₁₂ := fun i₁₂ => (((mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂).cofanMapObj ρ₁₂.p i₁₂
  have h₁₂ : ∀ i₁₂, IsColimit (c₁₂ i₁₂) := fun i₁₂ =>
    (((mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂).isColimitCofanMapObj ρ₁₂.p i₁₂
  let c := (((mapBifunctor G ρ₁₂.I₁₂ I₃).obj
    (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂)).obj X₃).cofanMapObj ρ₁₂.q j
  have hc : IsColimit c := (((mapBifunctor G ρ₁₂.I₁₂ I₃).obj
    (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂)).obj X₃).isColimitCofanMapObj ρ₁₂.q j
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let c₁₂' := fun (i : ρ₁₂.q ⁻¹' {j}) => (G.flip.obj (X₃ i.1.2)).mapCocone (c₁₂ i.1.1)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  have hc₁₂' : ∀ i, IsColimit (c₁₂' i) := fun i => isColimitOfPreserves _ (h₁₂ i.1.1)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let Z := (((mapTrifunctor (bifunctorComp₁₂ F₁₂ G) I₁ I₂ I₃).obj X₁).obj X₂).obj X₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let p' : I₁ × I₂ × I₃ → ρ₁₂.I₁₂ × I₃ := fun ⟨i₁, i₂, i₃⟩ => ⟨ρ₁₂.p ⟨i₁, i₂⟩, i₃⟩
  let e : ∀ (i₁₂ : ρ₁₂.I₁₂) (i₃ : I₃), p' ⁻¹' {(i₁₂, i₃)} ≃ ρ₁₂.p ⁻¹' {i₁₂} := fun i₁₂ i₃ =>
    { toFun := fun ⟨⟨i₁, i₂, i₃'⟩, hi⟩ => ⟨⟨i₁, i₂⟩, by aesop_cat⟩
      invFun := fun ⟨⟨i₁, i₂⟩, hi⟩ => ⟨⟨i₁, i₂, i₃⟩, by aesop_cat⟩
      left_inv := fun ⟨⟨i₁, i₂, i₃'⟩, hi⟩ => by
        obtain rfl : i₃ = i₃' := by aesop_cat
        rfl
      right_inv := fun _ => rfl }
  let c₁₂'' : ∀ (i : ρ₁₂.q ⁻¹' {j}), CofanMapObjFun Z p' (i.1.1, i.1.2) :=
    fun ⟨⟨i₁₂, i₃⟩, hi⟩ => by
      refine (Cocones.precompose (Iso.hom ?_)).obj ((Cocones.whiskeringEquivalence
        (Discrete.equivalence (e i₁₂ i₃))).functor.obj (c₁₂' ⟨⟨i₁₂, i₃⟩, hi⟩))
      refine (Discrete.natIso (fun ⟨⟨i₁, i₂, i₃'⟩, hi⟩ =>
        (G.obj ((F₁₂.obj (X₁ i₁)).obj (X₂ i₂))).mapIso (eqToIso ?_)))
      obtain rfl : i₃' = i₃ := congr_arg _root_.Prod.snd hi
      rfl
  have h₁₂'' : ∀ i, IsColimit (c₁₂'' i) := fun _ =>
    (IsColimit.precomposeHomEquiv _ _).symm (IsColimit.whiskerEquivalenceEquiv _ (hc₁₂' _))
  refine IsColimit.ofIsoColimit (isColimitCofanMapObjComp Z p' ρ₁₂.q r ρ₁₂.hpq j
    (fun ⟨i₁₂, i₃⟩ h => c₁₂'' ⟨⟨i₁₂, i₃⟩, h⟩) (fun ⟨i₁₂, i₃⟩ h => h₁₂'' ⟨⟨i₁₂, i₃⟩, h⟩) c hc)
    (Cocones.ext (Iso.refl _) (fun ⟨⟨i₁, i₂, i₃⟩, h⟩ => ?_))
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod ρ₁₂.I₁₂ I₃ := fun x => CategoryTheory.GradedO …
    e : (i₁₂ : ρ₁₂.I₁₂) → (i₃ : I₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Z.cofanMapObjComp p' ρ₁₂.q r ⋯ j (f …
  -/
  dsimp [Cofan.inj, c₁₂'', Z, p']
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod ρ₁₂.I₁₂ I₃ := fun x => CategoryTheory.GradedO …
    e : (i₁₂ : ρ₁₂.I₁₂) → (i₃ : I₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [comp_id, Functor.map_id, id_comp]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.139632, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.139636, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.139640, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.139644, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.139648, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.139652, u_6} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    c₁₂ : (i₁₂ : ρ₁₂.I₁₂) → (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂) …
    h₁₂ : (i₁₂ : ρ₁₂.I₁₂) → CategoryTheory.Limits.IsColimit (c₁₂ i₁₂)
    c : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (CategoryThe …
    hc : CategoryTheory.Limits.IsColimit c
    c₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₁₂' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod ρ₁₂.I₁₂ I₃ := fun x => CategoryTheory.GradedO …
    e : (i₁₂ : ρ₁₂.I₁₂) → (i₃ : I₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₁₂'' : (i : ↑(Set.preimage ρ₁₂.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c₁₂' ⟨{ fst := ρ₁₂.p { fst := i₁, s …
  -/
  rfl
  /-
    🎉 no goals
  -/


include ρ₁₂ in
lemma HasGoodTrifunctor₁₂Obj.hasMap :
    HasMap ((((mapTrifunctor (bifunctorComp₁₂ F₁₂ G) I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) r :=
  fun j => ⟨_, isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j⟩


/-- The action on graded objects of a trifunctor obtained by composition of two
bifunctors can be computed as a composition of the actions of these two bifunctors. -/
noncomputable def mapBifunctorComp₁₂MapObjIso :
    mapTrifunctorMapObj (bifunctorComp₁₂ F₁₂ G) r X₁ X₂ X₃ ≅
    mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ :=
  isoMk _ _ (fun j => (CofanMapObjFun.iso
    (isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j)).symm)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorComp₁₂MapObjIso_hom (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    ιMapTrifunctorMapObj (bifunctorComp₁₂ F₁₂ G) r X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorComp₁₂MapObjIso F₁₂ G ρ₁₂ X₁ X₂ X₃).hom j =
      ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝³ : CategoryTheory.Category.{u_15, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
  -/
  dsimp [mapBifunctorComp₁₂MapObjIso]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝³ : CategoryTheory.Category.{u_15, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
  -/
  apply CofanMapObjFun.ιMapObj_iso_inv
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_mapBifunctorComp₁₂MapObjIso_inv (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorComp₁₂MapObjIso F₁₂ G ρ₁₂ X₁ X₂ X₃).inv j =
      ιMapTrifunctorMapObj (bifunctorComp₁₂ F₁₂ G) r X₁ X₂ X₃ i₁ i₂ i₃ j h :=
  CofanMapObjFun.inj_iso_hom
    (isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j) _ h


@[ext]
lemma mapBifunctor₁₂BifunctorMapObj_ext {A : C₄}
    {f g : mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j ⟶ A}
    (h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : r ⟨i₁, i₂, i₃⟩ = j),
      ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ f =
        ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ g) : f = g := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_14, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_13, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj G ρ₁₂.q (Cate …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    ⊢ Eq f g
  -/
  apply Cofan.IsColimit.hom_ext (isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j)
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_14, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_13, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj G ρ₁₂.q (Cate …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    ⊢ ∀ (i : ↑(Set.preimage r (Singleton.singleton j))), Eq (CategoryTheory.Catego …
  -/
  rintro ⟨i, hi⟩
  /-
    case h.mk
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_14, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_13, u_5} C₁₂
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Categor …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj G ρ₁₂.q (Cate …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    i : Prod I₁ (Prod I₂ I₃)
    hi : Membership.mem (Set.preimage r (Singleton.singleton j)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
  -/
  exact h _ _ _ hi
  /-
    🎉 no goals
  -/


/-- Constructor for morphisms from
`mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j`. -/
noncomputable def mapBifunctor₁₂BifunctorDesc :
    mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ j ⟶ A :=
  Cofan.IsColimit.desc (isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j)
    (fun i ↦ f i.1.1 i.1.2.1 i.1.2.2 i.2)


@[reassoc (attr := simp)]
lemma ι_mapBifunctor₁₂BifunctorDesc
    (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : r ⟨i₁, i₂, i₃⟩ = j) :
    ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      mapBifunctor₁₂BifunctorDesc f = f i₁ i₂ i₃ h :=
  Cofan.IsColimit.fac
    (isColimitCofan₃MapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ j) _ ⟨_, h⟩


/-- Given a map `r : I₁ × I₂ × I₃ → J`, a `BifunctorComp₂₃IndexData r` consists of the data
of a type `I₂₃`, maps `p : I₂ × I₃ → I₂₃` and `q : I₁ × I₂₃ → J`, such that `r` is obtained
by composition of `p` and `q`. -/
structure BifunctorComp₂₃IndexData where
  /-- an auxiliary type -/
  I₂₃ : Type*
  /-- a map `I₂ × I₃ → I₂₃` -/
  p : I₂ × I₃ → I₂₃
  /-- a map `I₁ × I₂₃ → J` -/
  q : I₁ × I₂₃ → J
  hpq (i : I₁ × I₂ × I₃) : q ⟨i.1, p i.2⟩ = r i


/-- Given bifunctors `F : C₁ ⥤ C₂₃ ⥤ C₄`, `G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃`, graded objects
`X₁ : GradedObject I₁ C₁`, `X₂ : GradedObject I₂ C₂`, `X₃ : GradedObject I₃ C₃` and
`ρ₂₃ : BifunctorComp₂₃IndexData r`, this asserts that for all `i₁ : I₁` and `i₂₃ : ρ₂₃.I₂₃`,
the functor `F(X₁ i₁, _)` commutes with the coproducts of the `G₂₃(X₂ i₂, X₃ i₃)`
such that `ρ₂₃.p ⟨i₂, i₃⟩ = i₂₃`. -/
abbrev HasGoodTrifunctor₂₃Obj :=
  ∀ (i₁ : I₁) (i₂₃ : ρ₂₃.I₂₃), PreservesColimit (Discrete.functor
    (mapObjFun (((mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃) ρ₂₃.p i₂₃)) (F.obj (X₁ i₁))


/-- The inclusion of `(F.obj (X₁ i₁)).obj ((G₂₃.obj (X₂ i₂)).obj (X₃ i₃))` in
`mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j`
when `r (i₁, i₂, i₃) = j`. -/
noncomputable def ιMapBifunctorBifunctor₂₃MapObj (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    (F.obj (X₁ i₁)).obj ((G₂₃.obj (X₂ i₂)).obj (X₃ i₃)) ⟶
      mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j :=
  (F.obj (X₁ i₁)).map (ιMapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃ i₂ i₃ _ rfl) ≫
    ιMapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) i₁ (ρ₂₃.p ⟨i₂, i₃⟩) j
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            C₃ : Type u_3
            C₄ : Type u_4
            C₁₂ : Type u_5
            C₂₃ : Type u_6
            inst✝⁷ : CategoryTheory.Category.{?u.227898, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.227902, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.227906, u_3} C₃
            inst✝⁴ : CategoryTheory.Category.{?u.227910, u_4} C₄
            inst✝³ : CategoryTheory.Category.{?u.227914, u_5} C₁₂
            inst✝² : CategoryTheory.Category.{?u.227918, u_6} C₂₃
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
            G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
            I₁ : Type u_7
            I₂ : Type u_8
            I₃ : Type u_9
            J : Type u_10
            r : Prod I₁ (Prod I₂ I₃) → J
            ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
            X₁ : CategoryTheory.GradedObject I₁ C₁
            X₂ : CategoryTheory.GradedObject I₂ C₂
            X₃ : CategoryTheory.GradedObject I₃ C₃
            inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
            inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
            i₁ : I₁
            i₂ : I₂
            i₃ : I₃
            j : J
            h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            ⊢ Eq (ρ₂₃.q { fst := i₁, snd := ρ₂₃.p { fst := i₂, snd := i₃ } }) j
          -/
      (by rw [← h, ← ρ₂₃.hpq])
          /-
            🎉 no goals
          -/


@[reassoc]
lemma ιMapBifunctorBifunctor₂₃MapObj_eq (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) (i₂₃ : ρ₂₃.I₂₃) (h₂₃ : ρ₂₃.p ⟨i₂, i₃⟩ = i₂₃) :
    ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h =
  (F.obj (X₁ i₁)).map (ιMapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃ i₂ i₃ i₂₃ h₂₃) ≫
    ιMapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) i₁ i₂₃ j
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            C₃ : Type u_3
            C₄ : Type u_4
            C₁₂ : Type u_5
            C₂₃ : Type u_6
            inst✝⁷ : CategoryTheory.Category.{?u.230809, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.230813, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.230817, u_3} C₃
            inst✝⁴ : CategoryTheory.Category.{?u.230821, u_4} C₄
            inst✝³ : CategoryTheory.Category.{?u.230825, u_5} C₁₂
            inst✝² : CategoryTheory.Category.{?u.230829, u_6} C₂₃
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
            G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
            I₁ : Type u_7
            I₂ : Type u_8
            I₃ : Type u_9
            J : Type u_10
            r : Prod I₁ (Prod I₂ I₃) → J
            ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
            X₁ : CategoryTheory.GradedObject I₁ C₁
            X₂ : CategoryTheory.GradedObject I₂ C₂
            X₃ : CategoryTheory.GradedObject I₃ C₃
            inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
            inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
            i₁ : I₁
            i₂ : I₂
            i₃ : I₃
            j : J
            h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₂₃ : ρ₂₃.I₂₃
            h₂₃ : Eq (ρ₂₃.p { fst := i₂, snd := i₃ }) i₂₃
            ⊢ Eq (ρ₂₃.q { fst := i₁, snd := i₂₃ }) j
          -/
      (by rw [← h, ← h₂₃, ← ρ₂₃.hpq]) := by
          /-
            🎉 no goals
          -/
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁶ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_15, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_12, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_13, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    i₂₃ : ρ₂₃.I₂₃
    h₂₃ : Eq (ρ₂₃.p { fst := i₂, snd := i₃ }) i₂₃
    ⊢ Eq (CategoryTheory.GradedObject.ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁  …
  -/
  subst h₂₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁶ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_15, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_12, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_13, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.GradedObject.ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The cofan consisting of the inclusions given by `ιMapBifunctorBifunctor₂₃MapObj`. -/
noncomputable def cofan₃MapBifunctorBifunctor₂₃MapObj (j : J) :
    ((((mapTrifunctor (bifunctorComp₂₃ F G₂₃) I₁ I₂ I₃).obj X₁).obj X₂).obj
      X₃).CofanMapObjFun r j :=
  Cofan.mk (mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j)
    (fun ⟨⟨i₁, i₂, i₃⟩, (hi : r ⟨i₁, i₂, i₃⟩ = j)⟩ =>
      ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j hi)


/-- The cofan `cofan₃MapBifunctorBifunctor₂₃MapObj` is a colimit, see the induced isomorphism
`mapBifunctorComp₁₂MapObjIso`. -/
noncomputable def isColimitCofan₃MapBifunctorBifunctor₂₃MapObj (j : J) :
    IsColimit (cofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let c₂₃ := fun i₂₃ => (((mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃).cofanMapObj ρ₂₃.p i₂₃
  have h₂₃ : ∀ i₂₃, IsColimit (c₂₃ i₂₃) := fun i₂₃ =>
    (((mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃).isColimitCofanMapObj ρ₂₃.p i₂₃
  let c := (((mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj
    (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃)).cofanMapObj ρ₂₃.q j
  have hc : IsColimit c := (((mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj
    (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃)).isColimitCofanMapObj ρ₂₃.q j
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let c₂₃' := fun (i : ρ₂₃.q ⁻¹' {j}) => (F.obj (X₁ i.1.1)).mapCocone (c₂₃ i.1.2)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  have hc₂₃' : ∀ i, IsColimit (c₂₃' i) := fun i => isColimitOfPreserves _ (h₂₃ i.1.2)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let Z := (((mapTrifunctor (bifunctorComp₂₃ F G₂₃) I₁ I₂ I₃).obj X₁).obj X₂).obj X₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.GradedObject.cofan₃MapBifunc …
  -/
  let p' : I₁ × I₂ × I₃ → I₁ × ρ₂₃.I₂₃ := fun ⟨i₁, i₂, i₃⟩ => ⟨i₁, ρ₂₃.p ⟨i₂, i₃⟩⟩
  let e : ∀ (i₁ : I₁) (i₂₃ : ρ₂₃.I₂₃) , p' ⁻¹' {(i₁, i₂₃)} ≃ ρ₂₃.p ⁻¹' {i₂₃} := fun i₁ i₂₃ =>
    { toFun := fun ⟨⟨i₁', i₂, i₃⟩, hi⟩ => ⟨⟨i₂, i₃⟩, by aesop_cat⟩
      invFun := fun ⟨⟨i₂, i₃⟩, hi⟩  => ⟨⟨i₁, i₂, i₃⟩, by aesop_cat⟩
      left_inv := fun ⟨⟨i₁', i₂, i₃⟩, hi⟩ => by
        obtain rfl : i₁ = i₁' := by aesop_cat
        rfl
      right_inv := fun _ => rfl }
  let c₂₃'' : ∀ (i : ρ₂₃.q ⁻¹' {j}), CofanMapObjFun Z p' (i.1.1, i.1.2) :=
    fun ⟨⟨i₁, i₂₃⟩, hi⟩ => by
      refine (Cocones.precompose (Iso.hom ?_)).obj ((Cocones.whiskeringEquivalence
        (Discrete.equivalence (e i₁ i₂₃))).functor.obj (c₂₃' ⟨⟨i₁, i₂₃⟩, hi⟩))
      refine Discrete.natIso (fun ⟨⟨i₁', i₂, i₃⟩, hi⟩ => eqToIso ?_)
      obtain rfl : i₁' = i₁ := congr_arg _root_.Prod.fst hi
      rfl
  have h₂₃'' : ∀ i, IsColimit (c₂₃'' i) := fun _ =>
    (IsColimit.precomposeHomEquiv _ _).symm (IsColimit.whiskerEquivalenceEquiv _ (hc₂₃' _))
  refine IsColimit.ofIsoColimit (isColimitCofanMapObjComp Z p' ρ₂₃.q r ρ₂₃.hpq j
    (fun ⟨i₁, i₂₃⟩ h => c₂₃'' ⟨⟨i₁, i₂₃⟩, h⟩) (fun ⟨i₁, i₂₃⟩ h => h₂₃'' ⟨⟨i₁, i₂₃⟩, h⟩) c hc)
    (Cocones.ext (Iso.refl _) (fun ⟨⟨i₁, i₂, i₃⟩, h⟩ => ?_))
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod I₁ ρ₂₃.I₂₃ := fun x => CategoryTheory.GradedO …
    e : (i₁ : I₁) → (i₂₃ : ρ₂₃.I₂₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Z.cofanMapObjComp p' ρ₂₃.q r ⋯ j (f …
  -/
  dsimp [Cofan.inj, c₂₃'', Z, p', e]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod I₁ ρ₂₃.I₂₃ := fun x => CategoryTheory.GradedO …
    e : (i₁ : I₁) → (i₂₃ : ρ₂₃.I₂₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [comp_id, id_comp]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₁₂ : Type u_5
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{?u.239976, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.239980, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.239984, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{?u.239988, u_4} C₄
    inst✝³ : CategoryTheory.Category.{?u.239992, u_5} C₁₂
    inst✝² : CategoryTheory.Category.{?u.239996, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    c₂₃ : (i₂₃ : ρ₂₃.I₂₃) → (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃) …
    h₂₃ : (i₂₃ : ρ₂₃.I₂₃) → CategoryTheory.Limits.IsColimit (c₂₃ i₂₃)
    c : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj (Cat …
    hc : CategoryTheory.Limits.IsColimit c
    c₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.Li …
    hc₂₃' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    Z : CategoryTheory.GradedObject (Prod I₁ (Prod I₂ I₃)) C₄ := (((CategoryTheory …
    p' : Prod I₁ (Prod I₂ I₃) → Prod I₁ ρ₂₃.I₂₃ := fun x => CategoryTheory.GradedO …
    e : (i₁ : I₁) → (i₂₃ : ρ₂₃.I₂₃) → Equiv ↑(Set.preimage p' (Singleton.singleton …
    c₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → Z.CofanMapObjFun …
    h₂₃'' : (i : ↑(Set.preimage ρ₂₃.q (Singleton.singleton j))) → CategoryTheory.L …
    x✝ : CategoryTheory.Discrete ↑(Set.preimage r (Singleton.singleton j))
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    h : Membership.mem (Set.preimage r (Singleton.singleton j)) { fst := i₁, snd : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c₂₃' ⟨{ fst := i₁, snd := ρ₂₃.p { f …
  -/
  rfl
  /-
    🎉 no goals
  -/


include ρ₂₃ in
lemma HasGoodTrifunctor₂₃Obj.hasMap :
    HasMap ((((mapTrifunctor (bifunctorComp₂₃ F G₂₃) I₁ I₂ I₃).obj X₁).obj X₂).obj X₃) r :=
  fun j => ⟨_, isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j⟩


/-- The action on graded objects of a trifunctor obtained by composition of two
bifunctors can be computed as a composition of the actions of these two bifunctors. -/
noncomputable def mapBifunctorComp₂₃MapObjIso :
    mapTrifunctorMapObj (bifunctorComp₂₃ F G₂₃) r X₁ X₂ X₃ ≅
    mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) :=
  isoMk _ _ (fun j => (CofanMapObjFun.iso
    (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j)).symm)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorComp₂₃MapObjIso_hom (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    ιMapTrifunctorMapObj (bifunctorComp₂₃ F G₂₃) r X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorComp₂₃MapObjIso F G₂₃ ρ₂₃ X₁ X₂ X₃).hom j =
      ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝³ : CategoryTheory.Category.{u_15, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
  -/
  dsimp [mapBifunctorComp₂₃MapObjIso]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_12, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝³ : CategoryTheory.Category.{u_15, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    inst✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapTrif …
  -/
  apply CofanMapObjFun.ιMapObj_iso_inv
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_mapBifunctorComp₂₃MapObjIso_inv (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J)
    (h : r (i₁, i₂, i₃) = j) :
    ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorComp₂₃MapObjIso F G₂₃ ρ₂₃ X₁ X₂ X₃).inv j =
      ιMapTrifunctorMapObj (bifunctorComp₂₃ F G₂₃) r X₁ X₂ X₃ i₁ i₂ i₃ j h :=
  CofanMapObjFun.inj_iso_hom
    (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j) _ h


@[ext]
lemma mapBifunctorBifunctor₂₃MapObj_ext
    {f g : mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j ⟶ A}
    (h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : r ⟨i₁, i₂, i₃⟩ = j),
      ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ f =
        ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ g) : f = g := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁶ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_15, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_16, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_14, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F ρ₂₃.q X₁ (C …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    ⊢ Eq f g
  -/
  apply Cofan.IsColimit.hom_ext (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j)
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁶ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_15, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_16, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_14, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F ρ₂₃.q X₁ (C …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    ⊢ ∀ (i : ↑(Set.preimage r (Singleton.singleton j))), Eq (CategoryTheory.Catego …
  -/
  rintro ⟨i, hi⟩
  /-
    case h.mk
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    C₂₃ : Type u_6
    inst✝⁶ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_15, u_2} C₂
    inst✝⁴ : CategoryTheory.Category.{u_16, u_3} C₃
    inst✝³ : CategoryTheory.Category.{u_11, u_4} C₄
    inst✝² : CategoryTheory.Category.{u_14, u_6} C₂₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    j : J
    A : C₄
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F ρ₂₃.q X₁ (C …
    h : ∀ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : Eq (r { fst := i₁, snd := { fst := i₂ …
    i : Prod I₁ (Prod I₂ I₃)
    hi : Membership.mem (Set.preimage r (Singleton.singleton j)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
  -/
  exact h _ _ _ hi
  /-
    🎉 no goals
  -/


/-- Constructor for morphisms from
`mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j`. -/
noncomputable def mapBifunctorBifunctor₂₃Desc :
    mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) j ⟶ A :=
  Cofan.IsColimit.desc (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j)
    (fun i ↦ f i.1.1 i.1.2.1 i.1.2.2 i.2)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorBifunctor₂₃Desc
    (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (h : r ⟨i₁, i₂, i₃⟩ = j) :
    ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      mapBifunctorBifunctor₂₃Desc f = f i₁ i₂ i₃ h :=
  Cofan.IsColimit.fac
    (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ j) _ ⟨_, h⟩


