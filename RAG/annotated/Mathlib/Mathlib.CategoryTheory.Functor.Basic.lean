/-- `Functor C D` represents a functor between categories `C` and `D`.

To apply a functor `F` to an object use `F.obj X`, and to a morphism use `F.map f`.

The axiom `map_id` expresses preservation of identities, and
`map_comp` expresses functoriality.

See <https://stacks.math.columbia.edu/tag/001B>.
-/
structure Functor (C : Type u₁) [Category.{v₁} C] (D : Type u₂) [Category.{v₂} D]
    extends Prefunctor C D : Type max v₁ v₂ u₁ u₂ where
  /-- A functor preserves identity morphisms. -/
  map_id : ∀ X : C, map (𝟙 X) = 𝟙 (obj X) := by aesop_cat
  /-- A functor preserves composition. -/
  map_comp : ∀ {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z), map (f ≫ g) = map f ≫ map g := by aesop_cat


/-- Notation for a functor between categories. -/
-- A functor is basically a function, so give ⥤ a similar precedence to → (25).
-- For example, `C × D ⥤ E` should parse as `(C × D) ⥤ E` not `C × (D ⥤ E)`.
scoped [CategoryTheory] infixr:26 " ⥤ " => Functor -- type as \func


lemma Functor.map_comp_assoc {C : Type u₁} [Category C] {D : Type u₂} [Category D] (F : C ⥤ D)
    {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) {W : D} (h : F.obj Z ⟶ W) :
    (F.map (f ≫ g)) ≫ h = F.map f ≫ F.map g ≫ h := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u_1, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{u_2, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    W : D
    h : Quiver.Hom (F.obj Z) W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  rw [F.map_comp, Category.assoc]
  /-
    🎉 no goals
  -/


/-- `𝟭 C` is the identity functor on a category `C`. -/
protected def id : C ⥤ C where
  obj X := X
  map f := f


/-- Notation for the identity functor on a category. -/
scoped [CategoryTheory] notation "𝟭" => Functor.id -- Type this as `\sb1`


instance : Inhabited (C ⥤ C) :=
  ⟨Functor.id C⟩


@[simp]
theorem id_obj (X : C) : (𝟭 C).obj X = X := rfl


@[simp]
theorem id_map {X Y : C} (f : X ⟶ Y) : (𝟭 C).map f = f := rfl


/-- `F ⋙ G` is the composition of a functor `F` and a functor `G` (`F` first, then `G`).
-/
@[simps obj]
def comp (F : C ⥤ D) (G : D ⥤ E) : C ⥤ E where
  obj X := G.obj (F.obj X)
  map f := G.map (F.map f)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.Functor C D
                   G : CategoryTheory.Functor D E
                   ⊢ ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := fun X  …
                 -/
  map_comp := by intros; dsimp; rw [F.map_comp, G.map_comp]
                                /-
                                  🎉 no goals
                                -/


/-- Notation for composition of functors. -/
scoped [CategoryTheory] infixr:80 " ⋙ " => Functor.comp


@[simp]
theorem comp_map (F : C ⥤ D) (G : D ⥤ E) {X Y : C} (f : X ⟶ Y) :
    (F ⋙ G).map f = G.map (F.map f) := rfl

-- These are not simp lemmas because rewriting along equalities between functors
-- is not necessarily a good idea.
-- Natural isomorphisms are also provided in `Whiskering.lean`.

                                                          /-
                                                            C : Type u₁
                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                            D : Type u₂
                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                            F : CategoryTheory.Functor C D
                                                            ⊢ Eq (F.comp (CategoryTheory.Functor.id D)) F
                                                          -/
protected theorem comp_id (F : C ⥤ D) : F ⋙ 𝟭 D = F := by cases F; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


                                                          /-
                                                            C : Type u₁
                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                            D : Type u₂
                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                            F : CategoryTheory.Functor C D
                                                            ⊢ Eq ((CategoryTheory.Functor.id C).comp F) F
                                                          -/
protected theorem id_comp (F : C ⥤ D) : 𝟭 C ⋙ F = F := by cases F; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem map_dite (F : C ⥤ D) {X Y : C} {P : Prop} [Decidable P]
    (f : P → (X ⟶ Y)) (g : ¬P → (X ⟶ Y)) :
    F.map (if h : P then f h else g h) = if h : P then F.map (f h) else F.map (g h) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    P : Prop
    inst✝ : Decidable P
    f : P → Quiver.Hom X Y
    g : Not P → Quiver.Hom X Y
    ⊢ Eq (F.map (dite P (fun h => f h) fun h => g h)) (dite P (fun h => F.map (f h …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem toPrefunctor_comp (F : C ⥤ D) (G : D ⥤ E) :
    F.toPrefunctor.comp G.toPrefunctor = (F ⋙ G).toPrefunctor := rfl


