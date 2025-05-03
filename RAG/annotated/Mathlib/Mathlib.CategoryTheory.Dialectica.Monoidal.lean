local notation "π₁" => prod.fst

local notation "π₂" => prod.snd

local notation "π(" a ", " b ")" => prod.lift a b


/-- The object `X ⊗ Y` in the `Dial C` category just tuples the left and right components. -/
@[simps] def tensorObj (X Y : Dial C) : Dial C where
  src := X.src ⨯ Y.src
  tgt := X.tgt ⨯ Y.tgt
  rel :=
    (Subobject.pullback (prod.map π₁ π₁)).obj X.rel ⊓
    (Subobject.pullback (prod.map π₂ π₂)).obj Y.rel


/-- The functorial action of `X ⊗ Y` in `Dial C`. -/
@[simps] def tensorHom {X₁ X₂ Y₁ Y₂ : Dial C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂) :
    tensorObj X₁ Y₁ ⟶ tensorObj X₂ Y₂ where
  f := prod.map f.f g.f
  F := π(prod.map π₁ π₁ ≫ f.F, prod.map π₂ π₂ ≫ g.F)
  le := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
      f : Quiver.Hom X₁ X₂
      g : Quiver.Hom Y₁ Y₂
      ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.lift C …
    -/
    simp only [tensorObj, Subobject.inf_pullback]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
      f : Quiver.Hom X₁ X₂
      g : Quiver.Hom Y₁ Y₂
      ⊢ LE.le (Min.min ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.pr …
    -/
    apply inf_le_inf <;> rw [← Subobject.pullback_comp, ← Subobject.pullback_comp]
    · have := (Subobject.pullback (prod.map π₁ π₁ :
        (X₁.src ⨯ Y₁.src) ⨯ X₂.tgt ⨯ Y₂.tgt ⟶ _)).monotone (Hom.le f)
      /-
        case h₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom Y₁ Y₂
        this : LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.m …
        ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.com …
      -/
      rw [← Subobject.pullback_comp, ← Subobject.pullback_comp] at this
      /-
        case h₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom Y₁ Y₂
        this : LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruc …
        ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.com …
      -/
                               /-
                                 🎉 no goals
                               -/
      convert this using 3 <;> simp
                               /-
                                 🎉 no goals
                               -/
    · have := (Subobject.pullback (prod.map π₂ π₂ :
        (X₁.src ⨯ Y₁.src) ⨯ X₂.tgt ⨯ Y₂.tgt ⟶ _)).monotone (Hom.le g)
      /-
        case h₂
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom Y₁ Y₂
        this : LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.Limits.prod.m …
        ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.com …
      -/
      rw [← Subobject.pullback_comp, ← Subobject.pullback_comp] at this
      /-
        case h₂
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X₁ X₂ Y₁ Y₂ : CategoryTheory.Dial C
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom Y₁ Y₂
        this : LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruc …
        ⊢ LE.le ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.com …
      -/
                               /-
                                 🎉 no goals
                               -/
      convert this using 3 <;> simp
                               /-
                                 🎉 no goals
                               -/


/-- The unit for the tensor `X ⊗ Y` in `Dial C`. -/
@[simps] def tensorUnit : Dial C := { src := ⊤_ _, tgt := ⊤_ _, rel := ⊤ }


/-- Left unit cancellation `1 ⊗ X ≅ X` in `Dial C`. -/
@[simps!] def leftUnitor (X : Dial C) : tensorObj tensorUnit X ≅ X :=
                                                                    /-
                                                                      C : Type u
                                                                      inst✝² : CategoryTheory.Category.{v, u} C
                                                                      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                      inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                      X : CategoryTheory.Dial C
                                                                      ⊢ Eq (CategoryTheory.Dial.tensorUnit.tensorObj X).rel ((CategoryTheory.Subobje …
                                                                    -/
  isoMk (Limits.prod.leftUnitor _) (Limits.prod.leftUnitor _) <| by simp [Subobject.pullback_top]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Right unit cancellation `X ⊗ 1 ≅ X` in `Dial C`. -/
@[simps!] def rightUnitor (X : Dial C) : tensorObj X tensorUnit ≅ X :=
                                                                      /-
                                                                        C : Type u
                                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                                        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                        inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                        X : CategoryTheory.Dial C
                                                                        ⊢ Eq (X.tensorObj CategoryTheory.Dial.tensorUnit).rel ((CategoryTheory.Subobje …
                                                                      -/
  isoMk (Limits.prod.rightUnitor _) (Limits.prod.rightUnitor _) <| by simp [Subobject.pullback_top]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The associator for tensor, `(X ⊗ Y) ⊗ Z ≅ X ⊗ (Y ⊗ Z)` in `Dial C`. -/
@[simps!]
def associator (X Y Z : Dial C) : tensorObj (tensorObj X Y) Z ≅ tensorObj X (tensorObj Y Z) :=
  isoMk (prod.associator ..) (prod.associator ..) <| by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y Z : CategoryTheory.Dial C
      ⊢ Eq ((X.tensorObj Y).tensorObj Z).rel ((CategoryTheory.Subobject.pullback (Ca …
    -/
    simp [Subobject.inf_pullback, ← Subobject.pullback_comp, inf_assoc]
    /-
      🎉 no goals
    -/


@[simps!]
instance : MonoidalCategoryStruct (Dial C) where
  tensorUnit := tensorUnit
  tensorObj := tensorObj
  whiskerLeft X _ _ f := tensorHom (𝟙 X) f
  whiskerRight f Y := tensorHom f (𝟙 Y)
  tensorHom := tensorHom
  leftUnitor := leftUnitor
  rightUnitor := rightUnitor
  associator := associator


                                                                                        /-
                                                                                          C : Type u
                                                                                          inst✝² : CategoryTheory.Category.{v, u} C
                                                                                          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                                          inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                                          X₁ X₂ : CategoryTheory.Dial C
                                                                                          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
                                                                                        -/
theorem tensor_id (X₁ X₂ : Dial C) : (𝟙 X₁ ⊗ 𝟙 X₂ : _ ⟶ _) = 𝟙 (X₁ ⊗ X₂ : Dial C) := by aesop_cat
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem tensor_comp {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : Dial C}
    (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : Y₁ ⟶ Z₁) (g₂ : Y₂ ⟶ Z₂) :
    tensorHom (f₁ ≫ g₁) (f₂ ≫ g₂) = tensorHom f₁ f₂ ≫ tensorHom g₁ g₂ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X₁ Y₁ Z₁ X₂ Y₂ Z₂ : CategoryTheory.Dial C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (CategoryTheory.Dial.tensorHom (CategoryTheory.CategoryStruct.comp f₁ g₁) …
  -/
          /-
            🎉 no goals
          -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  ext <;> simp; ext <;> simp <;> (rw [← Category.assoc]; congr 1; simp)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem associator_naturality {X₁ X₂ X₃ Y₁ Y₂ Y₃ : Dial C}
    (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃) :
    tensorHom (tensorHom f₁ f₂) f₃ ≫ (associator Y₁ Y₂ Y₃).hom =
                                                                     /-
                                                                       C : Type u
                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                       inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                       inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                       X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.Dial C
                                                                       f₁ : Quiver.Hom X₁ Y₁
                                                                       f₂ : Quiver.Hom X₂ Y₂
                                                                       f₃ : Quiver.Hom X₃ Y₃
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Dial.tensorHom (Categ …
                                                                     -/
    (associator X₁ X₂ X₃).hom ≫ tensorHom f₁ (tensorHom f₂ f₃) := by aesop_cat
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem leftUnitor_naturality {X Y : Dial C} (f : X ⟶ Y) :
    (𝟙 (𝟙_ (Dial C)) ⊗ f) ≫ (λ_ Y).hom = (λ_ X).hom ≫ f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : CategoryTheory.Dial C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
          /-
            🎉 no goals
          -/
                                            /-
                                              🎉 no goals
                                            -/
  ext <;> simp; ext; simp; congr 1; ext <;> simp
                                            /-
                                              🎉 no goals
                                            -/


theorem rightUnitor_naturality {X Y : Dial C} (f : X ⟶ Y) :
    (f ⊗ 𝟙 (𝟙_ (Dial C))) ≫ (ρ_ Y).hom = (ρ_ X).hom ≫ f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : CategoryTheory.Dial C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
          /-
            🎉 no goals
          -/
                                            /-
                                              🎉 no goals
                                            -/
  ext <;> simp; ext; simp; congr 1; ext <;> simp
                                            /-
                                              🎉 no goals
                                            -/


theorem pentagon (W X Y Z : Dial C) :
    (tensorHom (associator W X Y).hom (𝟙 Z)) ≫ (associator W (tensorObj X Y) Z).hom ≫
      (tensorHom (𝟙 W) (associator X Y Z).hom) =
    (associator (tensorObj W X) Y Z).hom ≫ (associator W X (tensorObj Y Z)).hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    W X Y Z : CategoryTheory.Dial C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Dial.tensorHom (W.ass …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem triangle (X Y : Dial C) :
    (associator X (𝟙_ (Dial C)) Y).hom ≫ tensorHom (𝟙 X) (leftUnitor Y).hom =
                                              /-
                                                C : Type u
                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                X Y : CategoryTheory.Dial C
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.associator CategoryTheory.Monoidal …
                                              -/
    tensorHom (rightUnitor X).hom (𝟙 Y) := by aesop_cat
                                              /-
                                                🎉 no goals
                                              -/


instance : MonoidalCategory (Dial C) :=
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ⊢ ∀ (X : CategoryTheory.Dial C) {Y₁ Y₂ : CategoryTheory.Dial C} (f : Quiver.Ho …
  -/
  /-
    🎉 no goals
  -/
  .ofTensorHom
  /-
    🎉 no goals
  -/
    (tensor_id := tensor_id)
    (tensor_comp := tensor_comp)
    (associator_naturality := associator_naturality)
    (leftUnitor_naturality := leftUnitor_naturality)
    (rightUnitor_naturality := rightUnitor_naturality)
    (pentagon := pentagon)
    (triangle := triangle)


/-- The braiding isomorphism `X ⊗ Y ≅ Y ⊗ X` in `Dial C`. -/
@[simps!] def braiding (X Y : Dial C) : tensorObj X Y ≅ tensorObj Y X :=
  isoMk (prod.braiding ..) (prod.braiding ..) <| by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : CategoryTheory.Dial C
      ⊢ Eq (X.tensorObj Y).rel ((CategoryTheory.Subobject.pullback (CategoryTheory.L …
    -/
    simp [Subobject.inf_pullback, ← Subobject.pullback_comp, inf_comm]
    /-
      🎉 no goals
    -/


theorem symmetry (X Y : Dial C) :
                                                                      /-
                                                                        C : Type u
                                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                                        inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                        inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                        X Y : CategoryTheory.Dial C
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.braiding Y).hom (Y.braiding X).hom …
                                                                      -/
    (braiding X Y).hom ≫ (braiding Y X).hom = 𝟙 (tensorObj X Y) := by aesop_cat
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem braiding_naturality_right (X : Dial C) {Y Z : Dial C} (f : Y ⟶ Z) :
                                                                                          /-
                                                                                            C : Type u
                                                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                                                            inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                                            inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                                            X Y Z : CategoryTheory.Dial C
                                                                                            f : Quiver.Hom Y Z
                                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Dial.tensorHom (Categ …
                                                                                          -/
    tensorHom (𝟙 X) f ≫ (braiding X Z).hom = (braiding X Y).hom ≫ tensorHom f (𝟙 X) := by aesop_cat
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem braiding_naturality_left {X Y : Dial C} (f : X ⟶ Y) (Z : Dial C) :
                                                                                          /-
                                                                                            C : Type u
                                                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                                                            inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                                                            inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                                            X Y : CategoryTheory.Dial C
                                                                                            f : Quiver.Hom X Y
                                                                                            Z : CategoryTheory.Dial C
                                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Dial.tensorHom f (Cat …
                                                                                          -/
    tensorHom f (𝟙 Z) ≫ (braiding Y Z).hom = (braiding X Z).hom ≫ tensorHom (𝟙 Z) f := by aesop_cat
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem hexagon_forward (X Y Z : Dial C) :
    (associator X Y Z).hom ≫ (braiding X (Y ⊗ Z)).hom ≫ (associator Y Z X).hom =
      tensorHom (braiding X Y).hom (𝟙 Z) ≫ (associator Y X Z).hom ≫
                                               /-
                                                 C : Type u
                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                 inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                 inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                 X Y Z : CategoryTheory.Dial C
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.associator Y Z).hom (CategoryTheor …
                                               -/
      tensorHom (𝟙 Y) (braiding X Z).hom := by aesop_cat
                                               /-
                                                 🎉 no goals
                                               -/


theorem hexagon_reverse (X Y Z : Dial C) :
    (associator X Y Z).inv ≫ (braiding (X ⊗ Y) Z).hom ≫ (associator Z X Y).inv =
      tensorHom (𝟙 X) (braiding Y Z).hom ≫ (associator X Z Y).inv ≫
                                               /-
                                                 C : Type u
                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                 inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                 inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                 X Y Z : CategoryTheory.Dial C
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.associator Y Z).inv (CategoryTheor …
                                               -/
      tensorHom (braiding X Z).hom (𝟙 Y) := by aesop_cat
                                               /-
                                                 🎉 no goals
                                               -/


instance : SymmetricCategory (Dial C) where
  braiding := braiding
  braiding_naturality_right := braiding_naturality_right
  braiding_naturality_left := braiding_naturality_left
  hexagon_forward := hexagon_forward
  hexagon_reverse := hexagon_reverse
  symmetry := symmetry


