/-- Auxiliary definition for `bifunctorComp₁₂`. -/
@[simps]
def bifunctorComp₁₂Obj (F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂) (G : C₁₂ ⥤ C₃ ⥤ C₄) (X₁ : C₁) :
    C₂ ⥤ C₃ ⥤ C₄ where
  obj X₂ :=
    { obj := fun X₃ => (G.obj ((F₁₂.obj X₁).obj X₂)).obj X₃
      map := fun {_ _} φ => (G.obj ((F₁₂.obj X₁).obj X₂)).map φ }
  map {X₂ Y₂} φ :=
    { app := fun X₃ => (G.map ((F₁₂.obj X₁).map φ)).app X₃ }


/-- Given two bifunctors `F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂` and `G : C₁₂ ⥤ C₃ ⥤ C₄`, this is
the trifunctor `C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` obtained by composition. -/
@[simps]
def bifunctorComp₁₂ (F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂) (G : C₁₂ ⥤ C₃ ⥤ C₄) :
    C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj X₁ := bifunctorComp₁₂Obj F₁₂ G X₁
  map {X₁ Y₁} φ :=
    { app := fun X₂ =>
        { app := fun X₃ => (G.map ((F₁₂.map φ).app X₂)).app X₃ }
      naturality := fun {X₂ Y₂} ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.4768, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.4772, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.4776, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.4780, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.4784, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.4788, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : C₂
          ψ : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.bifunctor …
        -/
        ext X₃
        /-
          case w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.4768, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.4772, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.4776, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.4780, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.4784, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.4788, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : C₃
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun X₁ => CategoryTheory.bifuncto …
        -/
        dsimp
        /-
          case w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.4768, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.4772, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.4776, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.4780, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.4784, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.4788, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          φ : Quiver.Hom X₁ Y₁
          X₂ Y₂ : C₂
          ψ : Quiver.Hom X₂ Y₂
          X₃ : C₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁₂.obj X₁).map ψ)).app X₃) …
        -/
        simp only [← NatTrans.comp_app, ← G.map_comp, NatTrans.naturality] }
        /-
          🎉 no goals
        -/


/-- Auxiliary definition for `bifunctorComp₁₂Functor`. -/
@[simps]
def bifunctorComp₁₂FunctorObj (F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂) :
    (C₁₂ ⥤ C₃ ⥤ C₄) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj G := bifunctorComp₁₂ F₁₂ G
  map {G G'} φ :=
    { app X₁ :=
        { app X₂ :=
            { app X₃ := (φ.app ((F₁₂.obj X₁).obj X₂)).app X₃ }
          naturality := fun X₂ Y₂ f ↦ by
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              φ : Quiver.Hom G G'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun G => CategoryTheory.bifunctor …
            -/
            ext X₃
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              φ : Quiver.Hom G G'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((((fun G => CategoryTheory.bifuncto …
            -/
            dsimp
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              φ : Quiver.Hom G G'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁₂.obj X₁).map f)).app X₃) …
            -/
            simp only [← NatTrans.comp_app, NatTrans.naturality] }
            /-
              🎉 no goals
            -/
      naturality X₁ Y₁ f := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          φ : Quiver.Hom G G'
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun G => CategoryTheory.bifunctorC …
        -/
        ext X₂ X₃
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          φ : Quiver.Hom G G'
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (((CategoryTheory.CategoryStruct.comp (((fun G => CategoryTheory.bifuncto …
        -/
        dsimp
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.15747, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.15751, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.15755, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.15759, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.15763, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.15767, u_6} C₂₃
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          φ : Quiver.Hom G G'
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁₂.map f).app X₂)).app X₃) …
        -/
        simp only [← NatTrans.comp_app, NatTrans.naturality] }
        /-
          🎉 no goals
        -/


/-- Auxiliary definition for `bifunctorComp₁₂Functor`. -/
@[simps]
def bifunctorComp₁₂FunctorMap {F₁₂ F₁₂' : C₁ ⥤ C₂ ⥤ C₁₂} (φ : F₁₂ ⟶ F₁₂') :
    bifunctorComp₁₂FunctorObj (C₃ := C₃) (C₄ := C₄) F₁₂ ⟶ bifunctorComp₁₂FunctorObj F₁₂' where
  app G :=
    { app X₁ :=
        { app X₂ := { app X₃ := (G.map ((φ.app X₁).app X₂)).app X₃ }
          naturality := fun X₂ Y₂ f ↦ by
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
              F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              φ : Quiver.Hom F₁₂ F₁₂'
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.bifunctorComp₁₂Fun …
            -/
            ext X₃
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
              F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              φ : Quiver.Hom F₁₂ F₁₂'
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((((CategoryTheory.bifunctorComp₁₂Fu …
            -/
            dsimp
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
              F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              φ : Quiver.Hom F₁₂ F₁₂'
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁₂.obj X₁).map f)).app X₃) …
            -/
            simp only [← NatTrans.comp_app, NatTrans.naturality, ← G.map_comp] }
            /-
              🎉 no goals
            -/
      naturality X₁ Y₁ f := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
          F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          φ : Quiver.Hom F₁₂ F₁₂'
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.bifunctorComp₁₂Func …
        -/
        ext X₂ X₃
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
          F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          φ : Quiver.Hom F₁₂ F₁₂'
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (((CategoryTheory.CategoryStruct.comp (((CategoryTheory.bifunctorComp₁₂Fu …
        -/
        dsimp
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
          F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          φ : Quiver.Hom F₁₂ F₁₂'
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁₂.map f).app X₂)).app X₃) …
        -/
        simp only [← NatTrans.comp_app, NatTrans.naturality, ← G.map_comp] }
        /-
          🎉 no goals
        -/
  naturality G G' f := by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
      F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      φ : Quiver.Hom F₁₂ F₁₂'
      G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      f : Quiver.Hom G G'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.bifunctorComp₁₂Funct …
    -/
    ext X₁ X₂ X₃
    /-
      case w.h.w.h.w.h
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
      F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      φ : Quiver.Hom F₁₂ F₁₂'
      G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      f : Quiver.Hom G G'
      X₁ : C₁
      X₂ : C₂
      X₃ : C₃
      ⊢ Eq ((((CategoryTheory.CategoryStruct.comp ((CategoryTheory.bifunctorComp₁₂Fu …
    -/
    dsimp
    /-
      case w.h.w.h.w.h
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      C₄ : Type u_4
      C₁₂ : Type u_5
      C₂₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{?u.39101, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{?u.39105, u_2} C₂
      inst✝³ : CategoryTheory.Category.{?u.39109, u_3} C₃
      inst✝² : CategoryTheory.Category.{?u.39113, u_4} C₄
      inst✝¹ : CategoryTheory.Category.{?u.39117, u_5} C₁₂
      inst✝ : CategoryTheory.Category.{?u.39121, u_6} C₂₃
      F₁₂ F₁₂' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      φ : Quiver.Hom F₁₂ F₁₂'
      G G' : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      f : Quiver.Hom G G'
      X₁ : C₁
      X₂ : C₂
      X₃ : C₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((f.app ((F₁₂.obj X₁).obj X₂)).app X₃ …
    -/
    simp only [← NatTrans.comp_app, NatTrans.naturality]
    /-
      🎉 no goals
    -/


/-- The functor `(C₁ ⥤ C₂ ⥤ C₁₂) ⥤ (C₁₂ ⥤ C₃ ⥤ C₄) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` which
sends `F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂` and `G : C₁₂ ⥤ C₃ ⥤ C₄` to the functor
`bifunctorComp₁₂ F₁₂ G : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄`. -/
@[simps]
def bifunctorComp₁₂Functor : (C₁ ⥤ C₂ ⥤ C₁₂) ⥤ (C₁₂ ⥤ C₃ ⥤ C₄) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj := bifunctorComp₁₂FunctorObj
  map := bifunctorComp₁₂FunctorMap


/-- Auxiliary definition for `bifunctorComp₂₃`. -/
@[simps]
def bifunctorComp₂₃Obj (F : C₁ ⥤ C₂₃ ⥤ C₄) (G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃) (X₁ : C₁) :
    C₂ ⥤ C₃ ⥤ C₄ where
  obj X₂ :=
    { obj X₃ := (F.obj X₁).obj ((G₂₃.obj X₂).obj X₃)
      map φ := (F.obj X₁).map ((G₂₃.obj X₂).map φ) }
  map {X₂ Y₂} φ :=
    { app X₃ := (F.obj X₁).map ((G₂₃.map φ).app X₃)
      naturality X₃ Y₃ φ := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.59404, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.59408, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.59412, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.59416, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.59420, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.59424, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          X₁ : C₁
          X₂ Y₂ : C₂
          φ✝ : Quiver.Hom X₂ Y₂
          X₃ Y₃ : C₃
          φ : Quiver.Hom X₃ Y₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X₂ => { obj := fun X₃ => (F.ob …
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.59404, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.59408, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.59412, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.59416, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.59420, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.59424, u_6} C₂₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          X₁ : C₁
          X₂ Y₂ : C₂
          φ✝ : Quiver.Hom X₂ Y₂
          X₃ Y₃ : C₃
          φ : Quiver.Hom X₃ Y₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj X₁).map ((G₂₃.obj X₂).map φ)) …
        -/
        simp only [← Functor.map_comp, NatTrans.naturality] }
        /-
          🎉 no goals
        -/


/-- Given two bifunctors `F : C₁ ⥤ C₂₃ ⥤ C₄` and `G₂₃ : C₂ ⥤ C₃ ⥤ C₄`, this is
the trifunctor `C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` obtained by composition. -/
@[simps]
def bifunctorComp₂₃ (F : C₁ ⥤ C₂₃ ⥤ C₄) (G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃) :
    C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj X₁ := bifunctorComp₂₃Obj F G₂₃ X₁
  map {X₁ Y₁} φ :=
    { app := fun X₂ =>
        { app := fun X₃ => (F.map φ).app ((G₂₃.obj X₂).obj X₃) } }


/-- Auxiliary definition for `bifunctorComp₂₃Functor`. -/
@[simps]
def bifunctorComp₂₃FunctorObj (F : C₁ ⥤ C₂₃ ⥤ C₄) :
    (C₂ ⥤ C₃ ⥤ C₂₃) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj G₂₃ := bifunctorComp₂₃ F G₂₃
  map {G₂₃ G₂₃'} φ :=
    { app X₁ :=
        { app X₂ :=
            { app X₃ := (F.obj X₁).map ((φ.app X₂).app X₃)
              naturality X₃ Y₃ f := by
                /-
                  C₁ : Type u_1
                  C₂ : Type u_2
                  C₃ : Type u_3
                  C₄ : Type u_4
                  C₁₂ : Type u_5
                  C₂₃ : Type u_6
                  inst✝⁵ : CategoryTheory.Category.{?u.71046, u_1} C₁
                  inst✝⁴ : CategoryTheory.Category.{?u.71050, u_2} C₂
                  inst✝³ : CategoryTheory.Category.{?u.71054, u_3} C₃
                  inst✝² : CategoryTheory.Category.{?u.71058, u_4} C₄
                  inst✝¹ : CategoryTheory.Category.{?u.71062, u_5} C₁₂
                  inst✝ : CategoryTheory.Category.{?u.71066, u_6} C₂₃
                  F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
                  G₂₃ G₂₃' : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
                  φ : Quiver.Hom G₂₃ G₂₃'
                  X₁ : C₁
                  X₂ : C₂
                  X₃ Y₃ : C₃
                  f : Quiver.Hom X₃ Y₃
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((((fun G₂₃ => CategoryTheory.bifunc …
                -/
                dsimp
                /-
                  C₁ : Type u_1
                  C₂ : Type u_2
                  C₃ : Type u_3
                  C₄ : Type u_4
                  C₁₂ : Type u_5
                  C₂₃ : Type u_6
                  inst✝⁵ : CategoryTheory.Category.{?u.71046, u_1} C₁
                  inst✝⁴ : CategoryTheory.Category.{?u.71050, u_2} C₂
                  inst✝³ : CategoryTheory.Category.{?u.71054, u_3} C₃
                  inst✝² : CategoryTheory.Category.{?u.71058, u_4} C₄
                  inst✝¹ : CategoryTheory.Category.{?u.71062, u_5} C₁₂
                  inst✝ : CategoryTheory.Category.{?u.71066, u_6} C₂₃
                  F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
                  G₂₃ G₂₃' : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
                  φ : Quiver.Hom G₂₃ G₂₃'
                  X₁ : C₁
                  X₂ : C₂
                  X₃ Y₃ : C₃
                  f : Quiver.Hom X₃ Y₃
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj X₁).map ((G₂₃.obj X₂).map f)) …
                -/
                simp only [← Functor.map_comp, NatTrans.naturality] }
                /-
                  🎉 no goals
                -/
          naturality X₂ Y₂ f := by
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.71046, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.71050, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.71054, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.71058, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.71062, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.71066, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ G₂₃' : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              φ : Quiver.Hom G₂₃ G₂₃'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun G₂₃ => CategoryTheory.bifunct …
            -/
            ext X₃
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.71046, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.71050, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.71054, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.71058, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.71062, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.71066, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ G₂₃' : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              φ : Quiver.Hom G₂₃ G₂₃'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((((fun G₂₃ => CategoryTheory.bifunc …
            -/
            dsimp
            /-
              case w.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₃ : Type u_3
              C₄ : Type u_4
              C₁₂ : Type u_5
              C₂₃ : Type u_6
              inst✝⁵ : CategoryTheory.Category.{?u.71046, u_1} C₁
              inst✝⁴ : CategoryTheory.Category.{?u.71050, u_2} C₂
              inst✝³ : CategoryTheory.Category.{?u.71054, u_3} C₃
              inst✝² : CategoryTheory.Category.{?u.71058, u_4} C₄
              inst✝¹ : CategoryTheory.Category.{?u.71062, u_5} C₁₂
              inst✝ : CategoryTheory.Category.{?u.71066, u_6} C₂₃
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ G₂₃' : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              φ : Quiver.Hom G₂₃ G₂₃'
              X₁ : C₁
              X₂ Y₂ : C₂
              f : Quiver.Hom X₂ Y₂
              X₃ : C₃
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj X₁).map ((G₂₃.map f).app X₃)) …
            -/
            simp only [← NatTrans.comp_app, ← Functor.map_comp, NatTrans.naturality] } }
            /-
              🎉 no goals
            -/


/-- Auxiliary definition for `bifunctorComp₂₃Functor`. -/
@[simps]
def bifunctorComp₂₃FunctorMap {F F' : C₁ ⥤ C₂₃ ⥤ C₄} (φ : F ⟶ F') :
    bifunctorComp₂₃FunctorObj F (C₂ := C₂) (C₃ := C₃) ⟶ bifunctorComp₂₃FunctorObj F' where
  app G₂₃ :=
    { app X₁ := { app X₂ := { app X₃ := (φ.app X₁).app ((G₂₃.obj X₂).obj X₃) } }
      naturality X₁ Y₁ f := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.115740, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.115744, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.115748, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.115752, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.115756, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.115760, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          φ : Quiver.Hom F F'
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.bifunctorComp₂₃Func …
        -/
        ext X₂ X₃
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.115740, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.115744, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.115748, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.115752, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.115756, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.115760, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          φ : Quiver.Hom F F'
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (((CategoryTheory.CategoryStruct.comp (((CategoryTheory.bifunctorComp₂₃Fu …
        -/
        dsimp
        /-
          case w.h.w.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          C₄ : Type u_4
          C₁₂ : Type u_5
          C₂₃ : Type u_6
          inst✝⁵ : CategoryTheory.Category.{?u.115740, u_1} C₁
          inst✝⁴ : CategoryTheory.Category.{?u.115744, u_2} C₂
          inst✝³ : CategoryTheory.Category.{?u.115748, u_3} C₃
          inst✝² : CategoryTheory.Category.{?u.115752, u_4} C₄
          inst✝¹ : CategoryTheory.Category.{?u.115756, u_5} C₁₂
          inst✝ : CategoryTheory.Category.{?u.115760, u_6} C₂₃
          F F' : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          φ : Quiver.Hom F F'
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          X₁ Y₁ : C₁
          f : Quiver.Hom X₁ Y₁
          X₂ : C₂
          X₃ : C₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app ((G₂₃.obj X₂).obj X₃)) …
        -/
        simp only [← NatTrans.comp_app, NatTrans.naturality] }
        /-
          🎉 no goals
        -/


/-- The functor `(C₁ ⥤ C₂₃ ⥤ C₄) ⥤ (C₂ ⥤ C₃ ⥤ C₂₃) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄` which
sends `F : C₁ ⥤ C₂₃ ⥤ C₄` and `G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃` to the
functor `bifunctorComp₂₃ F G₂₃ : C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄`. -/
@[simps]
def bifunctorComp₂₃Functor :
    (C₁ ⥤ C₂₃ ⥤ C₄) ⥤ (C₂ ⥤ C₃ ⥤ C₂₃) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ C₄ where
  obj := bifunctorComp₂₃FunctorObj
  map := bifunctorComp₂₃FunctorMap


