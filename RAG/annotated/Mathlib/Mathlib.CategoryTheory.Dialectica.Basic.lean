variable (C) in
/-- The Dialectica category. An object of the category is a triple `⟨U, X, α ⊆ U × X⟩`,
and a morphism from `⟨U, X, α⟩` to `⟨V, Y, β⟩` is a pair `(f : U ⟶ V, F : U ⨯ Y ⟶ X)` such that
`{(u,y) | α(u, F(u, y))} ⊆ {(u,y) | β(f(u), y)}`. The subset `α` is actually encoded as an element
of `Subobject (U × X)`, and the above inequality is expressed using pullbacks. -/
structure Dial where
  /-- The source object -/
  src : C
  /-- The target object -/
  tgt : C
  /-- A subobject of `src ⨯ tgt`, interpreted as a relation -/
  rel : Subobject (src ⨯ tgt)


local notation "π₁" => prod.fst

local notation "π₂" => prod.snd

local notation "π(" a ", " b ")" => prod.lift a b


/-- A morphism in the `Dial C` category from `⟨U, X, α⟩` to `⟨V, Y, β⟩` is a pair
`(f : U ⟶ V, F : U ⨯ Y ⟶ X)` such that `{(u,y) | α(u, F(u, y))} ≤ {(u,y) | β(f(u), y)}`. -/
@[ext] structure Hom (X Y : Dial C) where
  /-- Maps the sources -/
  f : X.src ⟶ Y.src
  /-- Maps the targets (contravariantly) -/
  F : X.src ⨯ Y.tgt ⟶ X.tgt
  /-- This says `{(u, y) | α(u, F(u, y))} ⊆ {(u, y) | β(f(u), y)}` using subobject pullbacks -/
  le :
    (Subobject.pullback π(π₁, F)).obj X.rel ≤
    (Subobject.pullback (prod.map f (𝟙 _))).obj Y.rel


theorem comp_le_lemma {X Y Z : Dial C} (F : Dial.Hom X Y) (G : Dial.Hom Y Z) :
    (Subobject.pullback π(π₁, π(π₁, prod.map F.f (𝟙 _) ≫ G.F) ≫ F.F)).obj X.rel ≤
    (Subobject.pullback (prod.map (F.f ≫ G.f) (𝟙 Z.tgt))).obj Z.rel := by
  refine
    le_trans ?_ <| ((Subobject.pullback (π(π₁, prod.map F.f (𝟙 _) ≫ G.F))).monotone F.le).trans <|
    le_trans ?_ <| ((Subobject.pullback (prod.map F.f (𝟙 Z.tgt))).monotone G.le).trans ?_
        /-
          case refine_1
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y Z : CategoryTheory.Dial C
          F : X.Hom Y
          G : Y.Hom Z
          ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.lift C …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> simp [← Subobject.pullback_comp]
        /-
          🎉 no goals
        -/


@[simps]
instance : Category (Dial C) where
  Hom := Dial.Hom
  id X := {
    f := 𝟙 _
    F := π₂
             /-
               C : Type u
               inst✝² : CategoryTheory.Category.{v, u} C
               inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
               inst✝ : CategoryTheory.Limits.HasPullbacks C
               X : CategoryTheory.Dial C
               ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.lift C …
             -/
    le := by simp
             /-
               🎉 no goals
             -/
  }
  comp {_ _ _} (F G : Dial.Hom ..) := {
    f := F.f ≫ G.f
    F := π(π₁, prod.map F.f (𝟙 _) ≫ G.F) ≫ F.F
    le := comp_le_lemma F G
  }
                  /-
                    C : Type u
                    inst✝² : CategoryTheory.Category.{v, u} C
                    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                    inst✝ : CategoryTheory.Limits.HasPullbacks C
                    X✝ Y✝ : CategoryTheory.Dial C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
  id_comp f := by simp; rfl
                        /-
                          🎉 no goals
                        -/
                  /-
                    C : Type u
                    inst✝² : CategoryTheory.Category.{v, u} C
                    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                    inst✝ : CategoryTheory.Limits.HasPullbacks C
                    X✝ Y✝ : CategoryTheory.Dial C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                  -/
  comp_id f := by simp; rfl
                        /-
                          🎉 no goals
                        -/
  assoc f g h := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Dial C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [Category.assoc, Hom.mk.injEq, true_and]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Dial C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift Cate …
    -/
    rw [← Category.assoc, ← Category.assoc]; congr 1
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Dial C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift Cate …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp
            /-
              🎉 no goals
            -/


@[ext] theorem hom_ext {X Y : Dial C} {x y : X ⟶ Y} (hf : x.f = y.f) (hF : x.F = y.F) : x = y :=
   Hom.ext hf hF


/--
An isomorphism in `Dial C` can be induced by isomorphisms on the source and target,
which respect the respective relations on `X` and `Y`.
-/
@[simps] def isoMk {X Y : Dial C} (e₁ : X.src ≅ Y.src) (e₂ : X.tgt ≅ Y.tgt)
    (eq : X.rel = (Subobject.pullback (prod.map e₁.hom e₂.hom)).obj Y.rel) : X ≅ Y where
  hom := {
    f := e₁.hom
    F := π₂ ≫ e₂.inv
             /-
               C : Type u
               inst✝² : CategoryTheory.Category.{v, u} C
               inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
               inst✝ : CategoryTheory.Limits.HasPullbacks C
               X Y : CategoryTheory.Dial C
               e₁ : CategoryTheory.Iso X.src Y.src
               e₂ : CategoryTheory.Iso X.tgt Y.tgt
               eq : Eq X.rel ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod. …
               ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.lift C …
             -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    le := by rw [eq, ← Subobject.pullback_comp]; apply le_of_eq; congr; ext <;> simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  }
  inv := {
    f := e₁.inv
    F := π₂ ≫ e₂.hom
             /-
               C : Type u
               inst✝² : CategoryTheory.Category.{v, u} C
               inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
               inst✝ : CategoryTheory.Limits.HasPullbacks C
               X Y : CategoryTheory.Dial C
               e₁ : CategoryTheory.Iso X.src Y.src
               e₂ : CategoryTheory.Iso X.tgt Y.tgt
               eq : Eq X.rel ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod. …
               ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.lift C …
             -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    le := by rw [eq, ← Subobject.pullback_comp]; apply le_of_eq; congr; ext <;> simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  }


