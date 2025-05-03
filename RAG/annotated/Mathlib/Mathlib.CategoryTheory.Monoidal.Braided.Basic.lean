/-- A braided monoidal category is a monoidal category equipped with a braiding isomorphism
`β_ X Y : X ⊗ Y ≅ Y ⊗ X`
which is natural in both arguments,
and also satisfies the two hexagon identities.
-/
class BraidedCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] where
  /-- The braiding natural isomorphism. -/
  braiding : ∀ X Y : C, X ⊗ Y ≅ Y ⊗ X
  braiding_naturality_right :
    ∀ (X : C) {Y Z : C} (f : Y ⟶ Z),
      X ◁ f ≫ (braiding X Z).hom = (braiding X Y).hom ≫ f ▷ X := by
    aesop_cat
  braiding_naturality_left :
    ∀ {X Y : C} (f : X ⟶ Y) (Z : C),
      f ▷ Z ≫ (braiding Y Z).hom = (braiding X Z).hom ≫ Z ◁ f := by
    aesop_cat
  /-- The first hexagon identity. -/
  hexagon_forward :
    ∀ X Y Z : C,
      (α_ X Y Z).hom ≫ (braiding X (Y ⊗ Z)).hom ≫ (α_ Y Z X).hom =
        ((braiding X Y).hom ▷ Z) ≫ (α_ Y X Z).hom ≫ (Y ◁ (braiding X Z).hom) := by
    aesop_cat
  /-- The second hexagon identity. -/
  hexagon_reverse :
    ∀ X Y Z : C,
      (α_ X Y Z).inv ≫ (braiding (X ⊗ Y) Z).hom ≫ (α_ Z X Y).inv =
        (X ◁ (braiding Y Z).hom) ≫ (α_ X Z Y).inv ≫ ((braiding X Z).hom ▷ Y) := by
    aesop_cat


attribute [reassoc (attr := simp)]
  BraidedCategory.braiding_naturality_left
  BraidedCategory.braiding_naturality_right

attribute [reassoc] BraidedCategory.hexagon_forward BraidedCategory.hexagon_reverse


@[inherit_doc]
notation "β_" => BraidedCategory.braiding


@[simp, reassoc]
theorem braiding_tensor_left (X Y Z : C) :
    (β_ (X ⊗ Y) Z).hom  =
      (α_ X Y Z).hom ≫ X ◁ (β_ Y Z).hom ≫ (α_ X Z Y).inv ≫
        (β_ X Z).hom ▷ Y ≫ (α_ Z X Y).hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding (CategoryTheory.MonoidalCategory …
  -/
  apply (cancel_epi (α_ X Y Z).inv).1
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply (cancel_mono (α_ Z X Y).inv).1
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [hexagon_reverse]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem braiding_tensor_right (X Y Z : C) :
    (β_ X (Y ⊗ Z)).hom  =
      (α_ X Y Z).inv ≫ (β_ X Y).hom ▷ Z ≫ (α_ Y X Z).hom ≫
        Y ◁ (β_ X Z).hom ≫ (α_ Y Z X).inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding X (CategoryTheory.MonoidalCatego …
  -/
  apply (cancel_epi (α_ X Y Z).hom).1
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply (cancel_mono (α_ Y Z X).hom).1
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [hexagon_forward]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem braiding_inv_tensor_left (X Y Z : C) :
    (β_ (X ⊗ Y) Z).inv  =
      (α_ Z X Y).inv ≫ (β_ X Z).inv ▷ Y ≫ (α_ X Z Y).hom ≫
        X ◁ (β_ Y Z).inv ≫ (α_ X Y Z).inv :=
                       /-
                         C : Type u
                         inst✝² : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.MonoidalCategory C
                         inst✝ : CategoryTheory.BraidedCategory C
                         X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.BraidedCategory.braiding (CategoryThe …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[simp, reassoc]
theorem braiding_inv_tensor_right (X Y Z : C) :
    (β_ X (Y ⊗ Z)).inv  =
      (α_ Y Z X).hom ≫ Y ◁ (β_ X Z).inv ≫ (α_ Y X Z).inv ≫
        (β_ X Y).inv ▷ Z ≫ (α_ X Y Z).hom :=
                       /-
                         C : Type u
                         inst✝² : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.MonoidalCategory C
                         inst✝ : CategoryTheory.BraidedCategory C
                         X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.BraidedCategory.braiding X (CategoryT …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem braiding_naturality {X X' Y Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
    (f ⊗ g) ≫ (braiding Y Y').hom = (braiding X X').hom ≫ (g ⊗ f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X X' Y Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [tensorHom_def' f g, tensorHom_def g f]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X X' Y Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp_rw [Category.assoc, braiding_naturality_left, braiding_naturality_right_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem braiding_inv_naturality_right (X : C) {Y Z : C} (f : Y ⟶ Z) :
    X ◁ f ≫ (β_ Z X).inv = (β_ Y X).inv ≫ f ▷ X :=
  CommSq.w <| .vert_inv <| .mk <| braiding_naturality_left f X


@[reassoc (attr := simp)]
theorem braiding_inv_naturality_left {X Y : C} (f : X ⟶ Y) (Z : C) :
    f ▷ Z ≫ (β_ Z Y).inv = (β_ Z X).inv ≫ Z ◁ f :=
  CommSq.w <| .vert_inv <| .mk <| braiding_naturality_right Z f


@[reassoc (attr := simp)]
theorem braiding_inv_naturality {X X' Y Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
    (f ⊗ g) ≫ (β_ Y' Y).inv = (β_ X' X).inv ≫ (g ⊗ f) :=
  CommSq.w <| .vert_inv <| .mk <| braiding_naturality g f


@[reassoc]
theorem yang_baxter (X Y Z : C) :
    (α_ X Y Z).inv ≫ (β_ X Y).hom ▷ Z ≫ (α_ Y X Z).hom ≫
    Y ◁ (β_ X Z).hom ≫ (α_ Y Z X).inv ≫ (β_ Y Z).hom ▷ X ≫ (α_ Z Y X).hom =
      X ◁ (β_ Y Z).hom ≫ (α_ X Z Y).inv ≫ (β_ X Z).hom ▷ Y ≫
      (α_ Z X Y).hom ≫ Z ◁ (β_ X Y).hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← braiding_tensor_right_assoc X Y Z, ← cancel_mono (α_ Z Y X).inv]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  repeat rw [assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  rw [Iso.hom_inv_id, comp_id, ← braiding_naturality_right, braiding_tensor_right]
  /-
    🎉 no goals
  -/


theorem yang_baxter' (X Y Z : C) :
    (β_ X Y).hom ▷ Z ⊗≫ Y ◁ (β_ X Z).hom ⊗≫ (β_ Y Z).hom ▷ X =
      𝟙 _ ⊗≫ (X ◁ (β_ Y Z).hom ⊗≫ (β_ X Z).hom ▷ Y ⊗≫ Z ◁ (β_ X Y).hom) ⊗≫ 𝟙 _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
  -/
  rw [← cancel_epi (α_ X Y Z).inv, ← cancel_mono (α_ Z Y X).hom]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  convert yang_baxter X Y Z using 1
  /-
    case h.e'_2
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  all_goals monoidal
  /-
    🎉 no goals
  -/


theorem yang_baxter_iso (X Y Z : C) :
    (α_ X Y Z).symm ≪≫ whiskerRightIso (β_ X Y) Z ≪≫ α_ Y X Z ≪≫
    whiskerLeftIso Y (β_ X Z) ≪≫ (α_ Y Z X).symm ≪≫
    whiskerRightIso (β_ Y Z) X ≪≫ (α_ Z Y X) =
      whiskerLeftIso X (β_ Y Z) ≪≫ (α_ X Z Y).symm ≪≫
      whiskerRightIso (β_ X Z) Y ≪≫ α_ Z X Y ≪≫
      whiskerLeftIso Z (β_ X Y) := Iso.ext (yang_baxter X Y Z)


theorem hexagon_forward_iso (X Y Z : C) :
    α_ X Y Z ≪≫ β_ X (Y ⊗ Z) ≪≫ α_ Y Z X =
      whiskerRightIso (β_ X Y) Z ≪≫ α_ Y X Z ≪≫ whiskerLeftIso Y (β_ X Z) :=
  Iso.ext (hexagon_forward X Y Z)


theorem hexagon_reverse_iso (X Y Z : C) :
    (α_ X Y Z).symm ≪≫ β_ (X ⊗ Y) Z ≪≫ (α_ Z X Y).symm =
      whiskerLeftIso X (β_ Y Z) ≪≫ (α_ X Z Y).symm ≪≫ whiskerRightIso (β_ X Z) Y :=
  Iso.ext (hexagon_reverse X Y Z)


@[reassoc]
theorem hexagon_forward_inv (X Y Z : C) :
    (α_ Y Z X).inv ≫ (β_ X (Y ⊗ Z)).inv ≫ (α_ X Y Z).inv =
      Y ◁ (β_ X Z).inv ≫ (α_ Y X Z).inv ≫ (β_ X Y).inv ▷ Z := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem hexagon_reverse_inv (X Y Z : C) :
    (α_ Z X Y).hom ≫ (β_ (X ⊗ Y) Z).inv ≫ (α_ X Y Z).hom =
      (β_ X Z).inv ▷ Y ≫ (α_ X Z Y).hom ≫ X ◁ (β_ Y Z).inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
Verifying the axioms for a braiding by checking that the candidate braiding is sent to a braiding
by a faithful monoidal functor.
-/
def braidedCategoryOfFaithful {C D : Type*} [Category C] [Category D] [MonoidalCategory C]
    [MonoidalCategory D] (F : C ⥤ D) [F.Monoidal] [F.Faithful] [BraidedCategory D]
    (β : ∀ X Y : C, X ⊗ Y ≅ Y ⊗ X)
    (w : ∀ X Y, μ F _ _ ≫ F.map (β X Y).hom = (β_ _ _).hom ≫ μ F _ _) : BraidedCategory C where
  braiding := β
  braiding_naturality_left := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.CategoryStruct. …
    -/
    intros
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      Z✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    apply F.map_injective
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      Z✝ : C
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    refine (cancel_epi (μ F ?_ ?_)).1 ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      ⊢ ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct. …
    -/
    rw [Functor.map_comp, ← μ_natural_left_assoc, w, Functor.map_comp,
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
      reassoc_of% w, braiding_naturality_left_assoc, μ_natural_right]
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
  braiding_naturality_right := by
    intros
    apply F.map_injective
    refine (cancel_epi (μ F ?_ ?_)).1 ?_
    rw [Functor.map_comp, ← μ_natural_right_assoc, w, Functor.map_comp,
      reassoc_of% w, braiding_naturality_right_assoc, μ_natural_left]
  hexagon_forward := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      ⊢ ∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoid …
    -/
    intros
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    apply F.map_injective
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    refine (cancel_epi (μ F _ _)).1 ?_
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    refine (cancel_epi (μ F _ _ ▷ _)).1 ?_
    rw [Functor.map_comp, Functor.map_comp, Functor.map_comp, Functor.map_comp, ←
      μ_natural_left_assoc, ← comp_whiskerRight_assoc, w,
      comp_whiskerRight_assoc, Functor.LaxMonoidal.associativity_assoc,
      Functor.LaxMonoidal.associativity_assoc, ← μ_natural_right, ←
      MonoidalCategory.whiskerLeft_comp_assoc, w, MonoidalCategory.whiskerLeft_comp_assoc,
      reassoc_of% w, braiding_naturality_right_assoc,
      Functor.LaxMonoidal.associativity, hexagon_forward_assoc]
  hexagon_reverse := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      ⊢ ∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoid …
    -/
    intros
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    apply F.map_injective
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    refine (cancel_epi (μ F _ _)).1 ?_
    /-
      case a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.61498, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.61502, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      inst✝¹ : F.Faithful
      inst✝ : CategoryTheory.BraidedCategory D
      β : (X Y : C) → CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tens …
      w : ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functo …
      X✝ Y✝ Z✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    refine (cancel_epi (_ ◁ μ F _ _)).1 ?_
    rw [Functor.map_comp, Functor.map_comp, Functor.map_comp, Functor.map_comp, ←
      μ_natural_right_assoc, ← MonoidalCategory.whiskerLeft_comp_assoc, w,
      MonoidalCategory.whiskerLeft_comp_assoc, Functor.LaxMonoidal.associativity_inv_assoc,
      Functor.LaxMonoidal.associativity_inv_assoc, ← μ_natural_left,
      ← comp_whiskerRight_assoc, w, comp_whiskerRight_assoc, reassoc_of% w,
      braiding_naturality_left_assoc, Functor.LaxMonoidal.associativity_inv, hexagon_reverse_assoc]


/-- Pull back a braiding along a fully faithful monoidal functor. -/
noncomputable def braidedCategoryOfFullyFaithful {C D : Type*} [Category C] [Category D]
    [MonoidalCategory C] [MonoidalCategory D] (F : C ⥤ D) [F.Monoidal] [F.Full]
    [F.Faithful] [BraidedCategory D] : BraidedCategory C :=
  braidedCategoryOfFaithful F
    (fun X Y => F.preimageIso
      ((μIso F _ _).symm ≪≫ β_ (F.obj X) (F.obj Y) ≪≫ (μIso F _ _)))
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁷ : CategoryTheory.Category.{?u.88118, u_1} C
          inst✝⁶ : CategoryTheory.Category.{?u.88122, u_2} D
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          inst✝⁴ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝³ : F.Monoidal
          inst✝² : F.Full
          inst✝¹ : F.Faithful
          inst✝ : CategoryTheory.BraidedCategory D
          ⊢ ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor. …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


theorem braiding_leftUnitor_aux₁ (X : C) :
    (α_ (𝟙_ C) (𝟙_ C) X).hom ≫
        (𝟙_ C ◁ (β_ X (𝟙_ C)).inv) ≫ (α_ _ X _).inv ≫ ((λ_ X).hom ▷ _) =
      ((λ_ _).hom ▷ X) ≫ (β_ X (𝟙_ C)).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem braiding_leftUnitor_aux₂ (X : C) :
    ((β_ X (𝟙_ C)).hom ▷ 𝟙_ C) ≫ ((λ_ X).hom ▷ 𝟙_ C) = (ρ_ X).hom ▷ 𝟙_ C :=
  calc
    ((β_ X (𝟙_ C)).hom ▷ 𝟙_ C) ≫ ((λ_ X).hom ▷ 𝟙_ C) =
      ((β_ X (𝟙_ C)).hom ▷ 𝟙_ C) ≫ (α_ _ _ _).hom ≫ (α_ _ _ _).inv ≫ ((λ_ X).hom ▷ 𝟙_ C) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      monoidal
      /-
        🎉 no goals
      -/
    _ = ((β_ X (𝟙_ C)).hom ▷ 𝟙_ C) ≫ (α_ _ _ _).hom ≫ (_ ◁ (β_ X _).hom) ≫
          (_ ◁ (β_ X _).inv) ≫ (α_ _ _ _).inv ≫ ((λ_ X).hom ▷ 𝟙_ C) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp
      /-
        🎉 no goals
      -/
    _ = (α_ _ _ _).hom ≫ (β_ _ _).hom ≫ (α_ _ _ _).hom ≫ (_ ◁ (β_ X _).inv) ≫ (α_ _ _ _).inv ≫
          ((λ_ X).hom ▷ 𝟙_ C) := by
       /-
         C : Type u₁
         inst✝² : CategoryTheory.Category.{v₁, u₁} C
         inst✝¹ : CategoryTheory.MonoidalCategory C
         inst✝ : CategoryTheory.BraidedCategory C
         X : C
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
       -/
      (slice_lhs 1 3 => rw [← hexagon_forward]); simp only [assoc]
                                                 /-
                                                   🎉 no goals
                                                 -/
    _ = (α_ _ _ _).hom ≫ (β_ _ _).hom ≫ ((λ_ _).hom ▷ X) ≫ (β_ X _).inv := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rw [braiding_leftUnitor_aux₁]
      /-
        🎉 no goals
      -/
    _ = (α_ _ _ _).hom ≫ (_ ◁ (λ_ _).hom) ≫ (β_ _ _).hom ≫ (β_ X _).inv := by
       /-
         C : Type u₁
         inst✝² : CategoryTheory.Category.{v₁, u₁} C
         inst✝¹ : CategoryTheory.MonoidalCategory C
         inst✝ : CategoryTheory.BraidedCategory C
         X : C
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
       -/
      (slice_lhs 2 3 => rw [← braiding_naturality_right]); simp only [assoc]
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                /-
                                                  C : Type u₁
                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                  inst✝¹ : CategoryTheory.MonoidalCategory C
                                                  inst✝ : CategoryTheory.BraidedCategory C
                                                  X : C
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                -/
    _ = (α_ _ _ _).hom ≫ (_ ◁ (λ_ _).hom) := by rw [Iso.hom_inv_id, comp_id]
                                                /-
                                                  🎉 no goals
                                                -/
                                /-
                                  C : Type u₁
                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                  inst✝¹ : CategoryTheory.MonoidalCategory C
                                  inst✝ : CategoryTheory.BraidedCategory C
                                  X : C
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                -/
    _ = (ρ_ X).hom ▷ 𝟙_ C := by rw [triangle]
                                /-
                                  🎉 no goals
                                -/


@[reassoc]
theorem braiding_leftUnitor (X : C) : (β_ X (𝟙_ C)).hom ≫ (λ_ X).hom = (ρ_ X).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  rw [← whiskerRight_iff, comp_whiskerRight, braiding_leftUnitor_aux₂]
  /-
    🎉 no goals
  -/


theorem braiding_rightUnitor_aux₁ (X : C) :
    (α_ X (𝟙_ C) (𝟙_ C)).inv ≫
        ((β_ (𝟙_ C) X).inv ▷ 𝟙_ C) ≫ (α_ _ X _).hom ≫ (_ ◁ (ρ_ X).hom) =
      (X ◁ (ρ_ _).hom) ≫ (β_ (𝟙_ C) X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem braiding_rightUnitor_aux₂ (X : C) :
    (𝟙_ C ◁ (β_ (𝟙_ C) X).hom) ≫ (𝟙_ C ◁ (ρ_ X).hom) = 𝟙_ C ◁ (λ_ X).hom :=
  calc
    (𝟙_ C ◁ (β_ (𝟙_ C) X).hom) ≫ (𝟙_ C ◁ (ρ_ X).hom) =
      (𝟙_ C ◁ (β_ (𝟙_ C) X).hom) ≫ (α_ _ _ _).inv ≫ (α_ _ _ _).hom ≫ (𝟙_ C ◁ (ρ_ X).hom) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      monoidal
      /-
        🎉 no goals
      -/
    _ = (𝟙_ C ◁ (β_ (𝟙_ C) X).hom) ≫ (α_ _ _ _).inv ≫ ((β_ _ X).hom ▷ _) ≫
          ((β_ _ X).inv ▷ _) ≫ (α_ _ _ _).hom ≫ (𝟙_ C ◁ (ρ_ X).hom) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp
      /-
        🎉 no goals
      -/
    _ = (α_ _ _ _).inv ≫ (β_ _ _).hom ≫ (α_ _ _ _).inv ≫ ((β_ _ X).inv ▷ _) ≫ (α_ _ _ _).hom ≫
          (𝟙_ C ◁ (ρ_ X).hom) := by
       /-
         C : Type u₁
         inst✝² : CategoryTheory.Category.{v₁, u₁} C
         inst✝¹ : CategoryTheory.MonoidalCategory C
         inst✝ : CategoryTheory.BraidedCategory C
         X : C
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
       -/
      (slice_lhs 1 3 => rw [← hexagon_reverse]); simp only [assoc]
                                                 /-
                                                   🎉 no goals
                                                 -/
    _ = (α_ _ _ _).inv ≫ (β_ _ _).hom ≫ (X ◁ (ρ_ _).hom) ≫ (β_ _ X).inv := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.BraidedCategory C
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rw [braiding_rightUnitor_aux₁]
      /-
        🎉 no goals
      -/
    _ = (α_ _ _ _).inv ≫ ((ρ_ _).hom ▷ _) ≫ (β_ _ X).hom ≫ (β_ _ _).inv := by
       /-
         C : Type u₁
         inst✝² : CategoryTheory.Category.{v₁, u₁} C
         inst✝¹ : CategoryTheory.MonoidalCategory C
         inst✝ : CategoryTheory.BraidedCategory C
         X : C
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
       -/
      (slice_lhs 2 3 => rw [← braiding_naturality_left]); simp only [assoc]
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                /-
                                                  C : Type u₁
                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                  inst✝¹ : CategoryTheory.MonoidalCategory C
                                                  inst✝ : CategoryTheory.BraidedCategory C
                                                  X : C
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                -/
    _ = (α_ _ _ _).inv ≫ ((ρ_ _).hom ▷ _) := by rw [Iso.hom_inv_id, comp_id]
                                                /-
                                                  🎉 no goals
                                                -/
                                /-
                                  C : Type u₁
                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                  inst✝¹ : CategoryTheory.MonoidalCategory C
                                  inst✝ : CategoryTheory.BraidedCategory C
                                  X : C
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                -/
    _ = 𝟙_ C ◁ (λ_ X).hom := by rw [triangle_assoc_comp_right]
                                /-
                                  🎉 no goals
                                -/


@[reassoc]
theorem braiding_rightUnitor (X : C) : (β_ (𝟙_ C) X).hom ≫ (ρ_ X).hom = (λ_ X).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  rw [← whiskerLeft_iff, MonoidalCategory.whiskerLeft_comp, braiding_rightUnitor_aux₂]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem braiding_tensorUnit_left (X : C) : (β_ (𝟙_ C) X).hom = (λ_ X).hom ≫ (ρ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding CategoryTheory.MonoidalCategoryS …
  -/
  simp [← braiding_rightUnitor]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem braiding_inv_tensorUnit_left (X : C) : (β_ (𝟙_ C) X).inv = (ρ_ X).hom ≫ (λ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding CategoryTheory.MonoidalCategoryS …
  -/
  rw [Iso.inv_ext]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  rw [braiding_tensorUnit_left]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  monoidal
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftUnitor_inv_braiding (X : C) : (λ_ X).inv ≫ (β_ (𝟙_ C) X).hom = (ρ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem rightUnitor_inv_braiding (X : C) : (ρ_ X).inv ≫ (β_ X (𝟙_ C)).hom = (λ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply (cancel_mono (λ_ X).hom).1
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, braiding_leftUnitor, Iso.inv_hom_id]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem braiding_tensorUnit_right (X : C) : (β_ X (𝟙_ C)).hom = (ρ_ X).hom ≫ (λ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding X CategoryTheory.MonoidalCategor …
  -/
  simp [← rightUnitor_inv_braiding]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem braiding_inv_tensorUnit_right (X : C) : (β_ X (𝟙_ C)).inv = (λ_ X).hom ≫ (ρ_ X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding X CategoryTheory.MonoidalCategor …
  -/
  rw [Iso.inv_ext]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  rw [braiding_tensorUnit_right]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  monoidal
  /-
    🎉 no goals
  -/


/--
A symmetric monoidal category is a braided monoidal category for which the braiding is symmetric.

See <https://stacks.math.columbia.edu/tag/0FFW>.
-/
class SymmetricCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] extends
    BraidedCategory.{v} C where
  -- braiding symmetric:
  symmetry : ∀ X Y : C, (β_ X Y).hom ≫ (β_ Y X).hom = 𝟙 (X ⊗ Y) := by aesop_cat


attribute [reassoc (attr := simp)] SymmetricCategory.symmetry


lemma SymmetricCategory.braiding_swap_eq_inv_braiding {C : Type u₁}
    [Category.{v₁} C] [MonoidalCategory C] [SymmetricCategory C] (X Y : C) :
    (β_ Y X).hom = (β_ X Y).inv := Iso.inv_ext' (symmetry X Y)


/-- A lax braided functor between braided monoidal categories is a lax monoidal functor
which preserves the braiding.
-/
class Functor.LaxBraided (F : C ⥤ D) extends F.LaxMonoidal where
  braided : ∀ X Y : C, μ F X Y ≫ F.map (β_ X Y).hom =
    (β_ (F.obj X) (F.obj Y)).hom ≫ μ F Y X := by aesop_cat


attribute [reassoc] braided


instance id : (𝟭 C).LaxBraided where


instance (F : C ⥤ D) (G : D ⥤ E) [F.LaxBraided] [G.LaxBraided] :
    (F ⋙ G).LaxBraided where
  braided X Y := by
    /-
      C : Type u₁
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.MonoidalCategory C
      inst✝⁸ : CategoryTheory.BraidedCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      inst✝⁵ : CategoryTheory.BraidedCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      inst✝² : CategoryTheory.BraidedCategory E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxBraided
      inst✝ : G.LaxBraided
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    dsimp
    slice_lhs 2 3 =>
      rw [← CategoryTheory.Functor.map_comp, braided, CategoryTheory.Functor.map_comp]
    /-
      C : Type u₁
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.MonoidalCategory C
      inst✝⁸ : CategoryTheory.BraidedCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      inst✝⁵ : CategoryTheory.BraidedCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      inst✝² : CategoryTheory.BraidedCategory E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxBraided
      inst✝ : G.LaxBraided
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    slice_lhs 1 2 => rw [braided]
    /-
      C : Type u₁
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.MonoidalCategory C
      inst✝⁸ : CategoryTheory.BraidedCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      inst✝⁵ : CategoryTheory.BraidedCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      inst✝² : CategoryTheory.BraidedCategory E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxBraided
      inst✝ : G.LaxBraided
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc]
    /-
      🎉 no goals
    -/


/-- Bundled version of lax braided functors. -/
structure LaxBraidedFunctor extends C ⥤ D where
  laxBraided : toFunctor.LaxBraided := by infer_instance


/-- Constructor for `LaxBraidedFunctor C D`. -/
@[simps toFunctor]
def of (F : C ⥤ D) [F.LaxBraided] : LaxBraidedFunctor C D where
  toFunctor := F


/-- The lax monoidal functor induced by a lax braided functor. -/
@[simps toFunctor]
def toLaxMonoidalFunctor (F : LaxBraidedFunctor C D) : LaxMonoidalFunctor C D where
  toFunctor := F.toFunctor


instance : Category (LaxBraidedFunctor C D) :=
  InducedCategory.category (toLaxMonoidalFunctor)


@[simp]
lemma id_hom (F : LaxBraidedFunctor C D) : LaxMonoidalFunctor.Hom.hom (𝟙 F) = 𝟙 _ := rfl


@[reassoc, simp]
lemma comp_hom {F G H : LaxBraidedFunctor C D} (α : F ⟶ G) (β : G ⟶ H) :
    (α ≫ β).hom = α.hom ≫ β.hom := rfl


@[ext]
lemma hom_ext {F G : LaxBraidedFunctor C D} {α β : F ⟶ G} (h : α.hom = β.hom) : α = β :=
  LaxMonoidalFunctor.hom_ext h


/-- Constructor for morphisms in the category `LaxBraiededFunctor C D`. -/
@[simps]
def homMk {F G : LaxBraidedFunctor C D} (f : F.toFunctor ⟶ G.toFunctor) [NatTrans.IsMonoidal f] :
    F ⟶ G := ⟨f, inferInstance⟩


/-- Constructor for isomorphisms in the category `LaxBraidedFunctor C D`. -/
@[simps]
def isoMk {F G : LaxBraidedFunctor C D} (e : F.toFunctor ≅ G.toFunctor)
    [NatTrans.IsMonoidal e.hom] :
    F ≅ G where
  hom := homMk e.hom
  inv := homMk e.inv


/-- The forgetful functor from lax braided functors to lax monoidal functors. -/
@[simps! obj map]
def forget : LaxBraidedFunctor C D ⥤ LaxMonoidalFunctor C D :=
  inducedFunctor _


/-- The forgetful functor from lax braided functors to lax monoidal functors
is fully faithful. -/
def fullyFaithfulForget : (forget (C := C) (D := D)).FullyFaithful :=
  fullyFaithfulInducedFunctor _


variable {F G : LaxBraidedFunctor C D} (e : ∀ X, F.obj X ≅ G.obj X)
    (naturality : ∀ {X Y : C} (f : X ⟶ Y), F.map f ≫ (e Y).hom = (e X).hom ≫ G.map f := by
      aesop_cat)
    (unit : ε F.toFunctor ≫ (e (𝟙_ C)).hom = ε G.toFunctor := by aesop_cat)
    (tensor : ∀ X Y, μ F.toFunctor X Y ≫ (e (X ⊗ Y)).hom =
      ((e X).hom ⊗ (e Y).hom) ≫ μ G.toFunctor X Y := by aesop_cat)


/-- Constructor for isomorphisms between lax braided functors. -/
def isoOfComponents :
    F ≅ G :=
  fullyFaithfulForget.preimageIso
    (LaxMonoidalFunctor.isoOfComponents e naturality unit tensor)


@[simp]
lemma isoOfComponents_hom_hom_app (X : C) :
    (isoOfComponents e naturality unit tensor).hom.hom.app X = (e X).hom := rfl


@[simp]
lemma isoOfComponents_inv_hom_app (X : C) :
    (isoOfComponents e naturality unit tensor).inv.hom.app X = (e X).inv := rfl


/-- A braided functor between braided monoidal categories is a monoidal functor
which preserves the braiding.
-/
class Functor.Braided (F : C ⥤ D) extends F.Monoidal, F.LaxBraided where


@[simp, reassoc]
lemma Functor.map_braiding (F : C ⥤ D) (X Y : C) [F.Braided] :
    F.map (β_ X Y).hom =
    δ F X Y ≫ (β_ (F.obj X) (F.obj Y)).hom ≫ μ F Y X := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    inst✝⁴ : CategoryTheory.BraidedCategory C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.MonoidalCategory D
    inst✝¹ : CategoryTheory.BraidedCategory D
    F : CategoryTheory.Functor C D
    X Y : C
    inst✝ : F.Braided
    ⊢ Eq (F.map (CategoryTheory.BraidedCategory.braiding X Y).hom) (CategoryTheory …
  -/
  rw [← Functor.Braided.braided, δ_μ_assoc]
  /-
    🎉 no goals
  -/


/--
A braided category with a faithful braided functor to a symmetric category is itself symmetric.
-/
def symmetricCategoryOfFaithful {C D : Type*} [Category C] [Category D] [MonoidalCategory C]
    [MonoidalCategory D] [BraidedCategory C] [SymmetricCategory D] (F : C ⥤ D) [F.Braided]
    [F.Faithful] : SymmetricCategory C where
                                      /-
                                        C✝ : Type u₁
                                        inst✝¹⁶ : CategoryTheory.Category.{v₁, u₁} C✝
                                        inst✝¹⁵ : CategoryTheory.MonoidalCategory C✝
                                        inst✝¹⁴ : CategoryTheory.BraidedCategory C✝
                                        D✝ : Type u₂
                                        inst✝¹³ : CategoryTheory.Category.{v₂, u₂} D✝
                                        inst✝¹² : CategoryTheory.MonoidalCategory D✝
                                        inst✝¹¹ : CategoryTheory.BraidedCategory D✝
                                        E : Type u₃
                                        inst✝¹⁰ : CategoryTheory.Category.{v₃, u₃} E
                                        inst✝⁹ : CategoryTheory.MonoidalCategory E
                                        inst✝⁸ : CategoryTheory.BraidedCategory E
                                        C : Type u_1
                                        D : Type u_2
                                        inst✝⁷ : CategoryTheory.Category.{?u.179199, u_1} C
                                        inst✝⁶ : CategoryTheory.Category.{?u.179203, u_2} D
                                        inst✝⁵ : CategoryTheory.MonoidalCategory C
                                        inst✝⁴ : CategoryTheory.MonoidalCategory D
                                        inst✝³ : CategoryTheory.BraidedCategory C
                                        inst✝² : CategoryTheory.SymmetricCategory D
                                        F : CategoryTheory.Functor C D
                                        inst✝¹ : F.Braided
                                        inst✝ : F.Faithful
                                        X Y : C
                                        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategor …
                                      -/
  symmetry X Y := F.map_injective (by simp)
                                      /-
                                        🎉 no goals
                                      -/


instance : (𝟭 C).Braided where


instance (F : C ⥤ D) (G : D ⥤ E) [F.Braided] [G.Braided] : (F ⋙ G).Braided where


instance : BraidedCategory (Discrete M) where
  braiding X Y := Discrete.eqToIso (mul_comm X.as Y.as)


/-- A multiplicative morphism between commutative monoids gives a braided functor between
the corresponding discrete braided monoidal categories.
-/
instance Discrete.monoidalFunctorBraided (F : M →* N) :
    (Discrete.monoidalFunctor F).Braided where


/-- Swap the second and third objects in `(X₁ ⊗ X₂) ⊗ (Y₁ ⊗ Y₂)`. This is used to strength the
tensor product functor from `C × C` to `C` as a monoidal functor. -/
def tensorμ (X₁ X₂ Y₁ Y₂ : C) : (X₁ ⊗ X₂) ⊗ Y₁ ⊗ Y₂ ⟶ (X₁ ⊗ Y₁) ⊗ X₂ ⊗ Y₂ :=
  (α_ X₁ X₂ (Y₁ ⊗ Y₂)).hom ≫
    (X₁ ◁ (α_ X₂ Y₁ Y₂).inv) ≫
      (X₁ ◁ (β_ X₂ Y₁).hom ▷ Y₂) ≫
        (X₁ ◁ (α_ Y₁ X₂ Y₂).hom) ≫ (α_ X₁ Y₁ (X₂ ⊗ Y₂)).inv


/-- The inverse of `tensorμ`. -/
def tensorδ (X₁ X₂ Y₁ Y₂ : C) : (X₁ ⊗ Y₁) ⊗ X₂ ⊗ Y₂ ⟶ (X₁ ⊗ X₂) ⊗ Y₁ ⊗ Y₂ :=
  (α_ X₁ Y₁ (X₂ ⊗ Y₂)).hom ≫
    (X₁ ◁ (α_ Y₁ X₂ Y₂).inv) ≫
      (X₁ ◁ (β_ X₂ Y₁).inv ▷ Y₂) ≫
        (X₁ ◁ (α_ X₂ Y₁ Y₂).hom) ≫
          (α_ X₁ X₂ (Y₁ ⊗ Y₂)).inv


@[reassoc (attr := simp)]
lemma tensorμ_tensorδ (X₁ X₂ Y₁ Y₂ : C) :
    tensorμ X₁ X₂ Y₁ Y₂ ≫ tensorδ X₁ X₂ Y₁ Y₂ = 𝟙 _ := by
  simp only [tensorμ, tensorδ, assoc, Iso.inv_hom_id_assoc,
    ← MonoidalCategory.whiskerLeft_comp_assoc, Iso.hom_inv_id_assoc,
    hom_inv_whiskerRight_assoc, Iso.hom_inv_id, Iso.inv_hom_id,
    MonoidalCategory.whiskerLeft_id, id_comp]


@[reassoc (attr := simp)]
lemma tensorδ_tensorμ (X₁ X₂ Y₁ Y₂ : C) :
    tensorδ X₁ X₂ Y₁ Y₂ ≫ tensorμ X₁ X₂ Y₁ Y₂ = 𝟙 _ := by
  simp only [tensorμ, tensorδ, assoc, Iso.inv_hom_id_assoc,
    ← MonoidalCategory.whiskerLeft_comp_assoc, Iso.hom_inv_id_assoc,
    inv_hom_whiskerRight_assoc, Iso.inv_hom_id, Iso.hom_inv_id,
    MonoidalCategory.whiskerLeft_id, id_comp]


@[reassoc]
theorem tensorμ_natural {X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : U₁ ⟶ V₁)
    (g₂ : U₂ ⟶ V₂) :
    ((f₁ ⊗ f₂) ⊗ g₁ ⊗ g₂) ≫ tensorμ Y₁ Y₂ V₁ V₂ =
      tensorμ X₁ X₂ U₁ U₂ ≫ ((f₁ ⊗ g₁) ⊗ f₂ ⊗ g₂) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp only [tensorμ]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp_rw [← id_tensorHom, ← tensorHom_id]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_naturality]
  slice_lhs 2 3 =>
    rw [← tensor_comp, comp_id f₁, ← id_comp f₁, associator_inv_naturality, tensor_comp]
  slice_lhs 3 4 =>
    rw [← tensor_comp, ← tensor_comp, comp_id f₁, ← id_comp f₁, comp_id g₂, ← id_comp g₂,
      braiding_naturality, tensor_comp, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 4 5 => rw [← tensor_comp, comp_id f₁, ← id_comp f₁, associator_naturality, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 5 6 => rw [associator_inv_naturality]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ U₁ U₂ V₁ V₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom U₁ V₁
    g₂ : Quiver.Hom U₂ V₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem tensorμ_natural_left {X₁ X₂ Y₁ Y₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (Z₁ Z₂ : C) :
    (f₁ ⊗ f₂) ▷ (Z₁ ⊗ Z₂) ≫ tensorμ Y₁ Y₂ Z₁ Z₂ =
      tensorμ X₁ X₂ Z₁ Z₂ ≫ (f₁ ▷ Z₁ ⊗ f₂ ▷ Z₂) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    Z₁ Z₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  convert tensorμ_natural f₁ f₂ (𝟙 Z₁) (𝟙 Z₂) using 1 <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc]
theorem tensorμ_natural_right (Z₁ Z₂ : C) {X₁ X₂ Y₁ Y₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) :
    (Z₁ ⊗ Z₂) ◁ (f₁ ⊗ f₂) ≫ tensorμ Z₁ Z₂ Y₁ Y₂ =
      tensorμ Z₁ Z₂ X₁ X₂ ≫ (Z₁ ◁ f₁ ⊗ Z₂ ◁ f₂) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    Z₁ Z₂ X₁ X₂ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  convert tensorμ_natural (𝟙 Z₁) (𝟙 Z₂) f₁ f₂ using 1 <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc]
theorem tensor_left_unitality (X₁ X₂ : C) :
    (λ_ (X₁ ⊗ X₂)).hom =
      ((λ_ (𝟙_ C)).inv ▷ (X₁ ⊗ X₂)) ≫
        tensorμ (𝟙_ C) (𝟙_ C) X₁ X₂ ≫ ((λ_ X₁).hom ⊗ (λ_ X₂).hom) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  dsimp only [tensorμ]
  have :
    ((λ_ (𝟙_ C)).inv ▷ (X₁ ⊗ X₂)) ≫
        (α_ (𝟙_ C) (𝟙_ C) (X₁ ⊗ X₂)).hom ≫ (𝟙_ C ◁ (α_ (𝟙_ C) X₁ X₂).inv) =
      𝟙_ C ◁ (λ_ X₁).inv ▷ X₂ := by
    monoidal
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  slice_rhs 1 3 => rw [this]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  clear this
  slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, ← comp_whiskerRight,
    leftUnitor_inv_braiding]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  simp [tensorHom_id, id_tensorHom, tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem tensor_right_unitality (X₁ X₂ : C) :
    (ρ_ (X₁ ⊗ X₂)).hom =
      ((X₁ ⊗ X₂) ◁ (λ_ (𝟙_ C)).inv) ≫
        tensorμ X₁ X₂ (𝟙_ C) (𝟙_ C) ≫ ((ρ_ X₁).hom ⊗ (ρ_ X₂).hom) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
  -/
  dsimp only [tensorμ]
  have :
    ((X₁ ⊗ X₂) ◁ (λ_ (𝟙_ C)).inv) ≫
        (α_ X₁ X₂ (𝟙_ C ⊗ 𝟙_ C)).hom ≫ (X₁ ◁ (α_ X₂ (𝟙_ C) (𝟙_ C)).inv) =
      (α_ X₁ X₂ (𝟙_ C)).hom ≫ (X₁ ◁ (ρ_ X₂).inv ▷ 𝟙_ C) := by
    monoidal
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
  -/
  slice_rhs 1 3 => rw [this]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
  -/
  clear this
  slice_rhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, ← comp_whiskerRight,
    rightUnitor_inv_braiding]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
  -/
  simp [tensorHom_id, id_tensorHom, tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem tensor_associativity (X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C) :
    (tensorμ X₁ X₂ Y₁ Y₂ ▷ (Z₁ ⊗ Z₂)) ≫
        tensorμ (X₁ ⊗ Y₁) (X₂ ⊗ Y₂) Z₁ Z₂ ≫ ((α_ X₁ Y₁ Z₁).hom ⊗ (α_ X₂ Y₂ Z₂).hom) =
      (α_ (X₁ ⊗ X₂) (Y₁ ⊗ Y₂) (Z₁ ⊗ Z₂)).hom ≫
        ((X₁ ⊗ X₂) ◁ tensorμ Y₁ Y₂ Z₁ Z₂) ≫ tensorμ X₁ X₂ (Y₁ ⊗ Z₁) (Y₂ ⊗ Z₂) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp only [tensor_obj, prodMonoidal_tensorObj, tensorμ]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [braiding_tensor_left, braiding_tensor_right]
  calc
    _ = 𝟙 _ ⊗≫
      X₁ ◁ ((β_ X₂ Y₁).hom ▷ (Y₂ ⊗ Z₁) ≫ (Y₁ ⊗ X₂) ◁ (β_ Y₂ Z₁).hom) ▷ Z₂ ⊗≫
        X₁ ◁ Y₁ ◁ (β_ X₂ Z₁).hom ▷ Y₂ ▷ Z₂ ⊗≫ 𝟙 _ := by monoidal
    _ = _ := by rw [← whisker_exchange]; monoidal


instance tensorMonoidal : (tensor C).Monoidal :=
    Functor.CoreMonoidal.toMonoidal
      { εIso := (λ_ (𝟙_ C)).symm
        μIso := fun X Y ↦
          { hom := tensorμ X.1 X.2 Y.1 Y.2
            inv := tensorδ X.1 X.2 Y.1 Y.2 }
        μIso_hom_natural_left := fun f Z ↦ tensorμ_natural_left f.1 f.2 Z.1 Z.2
        μIso_hom_natural_right := fun Z f ↦ tensorμ_natural_right Z.1 Z.2 f.1 f.2
        associativity := fun X Y Z ↦ tensor_associativity X.1 X.2 Y.1 Y.2 Z.1 Z.2
        left_unitality := fun ⟨X₁, X₂⟩ ↦ tensor_left_unitality X₁ X₂
        right_unitality := fun ⟨X₁, X₂⟩ ↦ tensor_right_unitality X₁ X₂ }


@[simp] lemma tensor_ε : ε (tensor C) = (λ_ (𝟙_ C)).inv := rfl

@[simp] lemma tensor_η : η (tensor C) = (λ_ (𝟙_ C)).hom := rfl

@[simp] lemma tensor_μ (X Y : C × C) : μ (tensor C) X Y = tensorμ X.1 X.2 Y.1 Y.2 := rfl

@[simp] lemma tensor_δ (X Y : C × C) : δ (tensor C) X Y = tensorδ X.1 X.2 Y.1 Y.2 := rfl


@[reassoc]
theorem leftUnitor_monoidal (X₁ X₂ : C) :
    (λ_ X₁).hom ⊗ (λ_ X₂).hom =
      tensorμ (𝟙_ C) X₁ (𝟙_ C) X₂ ≫ ((λ_ (𝟙_ C)).hom ▷ (X₁ ⊗ X₂)) ≫ (λ_ (X₁ ⊗ X₂)).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  dsimp only [tensorμ]
  have :
    (λ_ X₁).hom ⊗ (λ_ X₂).hom =
      (α_ (𝟙_ C) X₁ (𝟙_ C ⊗ X₂)).hom ≫
        (𝟙_ C ◁ (α_ X₁ (𝟙_ C) X₂).inv) ≫ (λ_ ((X₁ ⊗ 𝟙_ C) ⊗ X₂)).hom ≫ ((ρ_ X₁).hom ▷ X₂) := by
    monoidal
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Mon …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  rw [this]; clear this
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← braiding_leftUnitor]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


@[reassoc]
theorem rightUnitor_monoidal (X₁ X₂ : C) :
    (ρ_ X₁).hom ⊗ (ρ_ X₂).hom =
      tensorμ X₁ (𝟙_ C) X₂ (𝟙_ C) ≫ ((X₁ ⊗ X₂) ◁ (λ_ (𝟙_ C)).hom) ≫ (ρ_ (X₁ ⊗ X₂)).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  dsimp only [tensorμ]
  have :
    (ρ_ X₁).hom ⊗ (ρ_ X₂).hom =
      (α_ X₁ (𝟙_ C) (X₂ ⊗ 𝟙_ C)).hom ≫
        (X₁ ◁ (α_ (𝟙_ C) X₂ (𝟙_ C)).inv) ≫ (X₁ ◁ (ρ_ (𝟙_ C ⊗ X₂)).hom) ≫ (X₁ ◁ (λ_ X₂).hom) := by
    monoidal
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    this : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Mon …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  rw [this]; clear this
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← braiding_rightUnitor]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


@[reassoc]
theorem associator_monoidal (X₁ X₂ X₃ Y₁ Y₂ Y₃ : C) :
    tensorμ (X₁ ⊗ X₂) X₃ (Y₁ ⊗ Y₂) Y₃ ≫
        (tensorμ X₁ X₂ Y₁ Y₂ ▷ (X₃ ⊗ Y₃)) ≫ (α_ (X₁ ⊗ Y₁) (X₂ ⊗ Y₂) (X₃ ⊗ Y₃)).hom =
      ((α_ X₁ X₂ X₃).hom ⊗ (α_ Y₁ Y₂ Y₃).hom) ≫
        tensorμ X₁ (X₂ ⊗ X₃) Y₁ (Y₂ ⊗ Y₃) ≫ ((X₁ ⊗ Y₁) ◁ tensorμ X₂ X₃ Y₂ Y₃) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  dsimp only [tensorμ]
  calc
    _ = 𝟙 _ ⊗≫ X₁ ◁ X₂ ◁ (β_ X₃ Y₁).hom ▷ Y₂ ▷ Y₃ ⊗≫
      X₁ ◁ ((X₂ ⊗ Y₁) ◁ (β_ X₃ Y₂).hom ≫
        (β_ X₂ Y₁).hom ▷ (Y₂ ⊗ X₃)) ▷ Y₃ ⊗≫ 𝟙 _ := by
          rw [braiding_tensor_right]; monoidal
    _ = _ := by rw [whisker_exchange, braiding_tensor_left]; monoidal


instance : BraidedCategory Cᵒᵖ where
  braiding X Y := (β_ Y.unop X.unop).op
                                                                   /-
                                                                     C : Type u₁
                                                                     inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                                                     inst✝⁷ : CategoryTheory.MonoidalCategory C
                                                                     inst✝⁶ : CategoryTheory.BraidedCategory C
                                                                     D : Type u₂
                                                                     inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                                                     inst✝⁴ : CategoryTheory.MonoidalCategory D
                                                                     inst✝³ : CategoryTheory.BraidedCategory D
                                                                     E : Type u₃
                                                                     inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                                                     inst✝¹ : CategoryTheory.MonoidalCategory E
                                                                     inst✝ : CategoryTheory.BraidedCategory E
                                                                     X x✝¹ x✝ : Opposite C
                                                                     f : Quiver.Hom x✝¹ x✝
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                   -/
  braiding_naturality_right X {_ _} f := Quiver.Hom.unop_inj <| by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                  /-
                                                                    C : Type u₁
                                                                    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                                                    inst✝⁷ : CategoryTheory.MonoidalCategory C
                                                                    inst✝⁶ : CategoryTheory.BraidedCategory C
                                                                    D : Type u₂
                                                                    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                                                    inst✝⁴ : CategoryTheory.MonoidalCategory D
                                                                    inst✝³ : CategoryTheory.BraidedCategory D
                                                                    E : Type u₃
                                                                    inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                                                    inst✝¹ : CategoryTheory.MonoidalCategory E
                                                                    inst✝ : CategoryTheory.BraidedCategory E
                                                                    x✝¹ x✝ : Opposite C
                                                                    f : Quiver.Hom x✝¹ x✝
                                                                    Z : Opposite C
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                  -/
  braiding_naturality_left {_ _} f Z := Quiver.Hom.unop_inj <| by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp] lemma op_braiding (X Y : C) : (β_ X Y).op = β_ (op Y) (op X) := rfl

@[simp] lemma unop_braiding (X Y : Cᵒᵖ) : (β_ X Y).unop = β_ (unop Y) (unop X) := rfl


@[simp] lemma op_hom_braiding (X Y : C) : (β_ X Y).hom.op = (β_ (op Y) (op X)).hom := rfl

@[simp] lemma unop_hom_braiding (X Y : Cᵒᵖ) : (β_ X Y).hom.unop = (β_ (unop Y) (unop X)).hom := rfl


@[simp] lemma op_inv_braiding (X Y : C) : (β_ X Y).inv.op = (β_ (op Y) (op X)).inv := rfl

@[simp] lemma unop_inv_braiding (X Y : Cᵒᵖ) : (β_ X Y).inv.unop = (β_ (unop Y) (unop X)).inv := rfl


instance instBraiding : BraidedCategory Cᴹᵒᵖ where
  braiding X Y := (β_ Y.unmop X.unmop).mop
                                                                    /-
                                                                      C : Type u₁
                                                                      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                                                      inst✝⁷ : CategoryTheory.MonoidalCategory C
                                                                      inst✝⁶ : CategoryTheory.BraidedCategory C
                                                                      D : Type u₂
                                                                      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                                                      inst✝⁴ : CategoryTheory.MonoidalCategory D
                                                                      inst✝³ : CategoryTheory.BraidedCategory D
                                                                      E : Type u₃
                                                                      inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                                                      inst✝¹ : CategoryTheory.MonoidalCategory E
                                                                      inst✝ : CategoryTheory.BraidedCategory E
                                                                      X x✝¹ x✝ : CategoryTheory.MonoidalOpposite C
                                                                      f : Quiver.Hom x✝¹ x✝
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                    -/
  braiding_naturality_right X {_ _} f := Quiver.Hom.unmop_inj <| by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                   /-
                                                                     C : Type u₁
                                                                     inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                                                     inst✝⁷ : CategoryTheory.MonoidalCategory C
                                                                     inst✝⁶ : CategoryTheory.BraidedCategory C
                                                                     D : Type u₂
                                                                     inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                                                     inst✝⁴ : CategoryTheory.MonoidalCategory D
                                                                     inst✝³ : CategoryTheory.BraidedCategory D
                                                                     E : Type u₃
                                                                     inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                                                     inst✝¹ : CategoryTheory.MonoidalCategory E
                                                                     inst✝ : CategoryTheory.BraidedCategory E
                                                                     x✝¹ x✝ : CategoryTheory.MonoidalOpposite C
                                                                     f : Quiver.Hom x✝¹ x✝
                                                                     Z : CategoryTheory.MonoidalOpposite C
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                   -/
  braiding_naturality_left {_ _} f Z := Quiver.Hom.unmop_inj <| by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma mop_braiding (X Y : C) : (β_ X Y).mop = β_ (mop Y) (mop X) := rfl

@[simp] lemma unmop_braiding (X Y : Cᴹᵒᵖ) : (β_ X Y).unmop = β_ (unmop Y) (unmop X) := rfl


@[simp] lemma mop_hom_braiding (X Y : C) : (β_ X Y).hom.mop = (β_ (mop Y) (mop X)).hom := rfl

@[simp]
lemma unmop_hom_braiding (X Y : Cᴹᵒᵖ) : (β_ X Y).hom.unmop = (β_ (unmop Y) (unmop X)).hom := rfl


@[simp] lemma mop_inv_braiding (X Y : C) : (β_ X Y).inv.mop = (β_ (mop Y) (mop X)).inv := rfl

@[simp]
lemma unmop_inv_braiding (X Y : Cᴹᵒᵖ) : (β_ X Y).inv.unmop = (β_ (unmop Y) (unmop X)).inv := rfl


instance : (mopFunctor C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun X Y ↦ β_ (mop X) (mop Y)
                                      /-
                                        C : Type u₁
                                        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                        inst✝⁷ : CategoryTheory.MonoidalCategory C
                                        inst✝⁶ : CategoryTheory.BraidedCategory C
                                        D : Type u₂
                                        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                        inst✝⁴ : CategoryTheory.MonoidalCategory D
                                        inst✝³ : CategoryTheory.BraidedCategory D
                                        E : Type u₃
                                        inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                        inst✝¹ : CategoryTheory.MonoidalCategory E
                                        inst✝ : CategoryTheory.BraidedCategory E
                                        X Y Z : C
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                      -/
      associativity := fun X Y Z ↦ by simp [← yang_baxter_assoc] }
                                      /-
                                        🎉 no goals
                                      -/


@[simp] lemma mopFunctor_ε : ε (mopFunctor C) = 𝟙 _ := rfl

@[simp] lemma mopFunctor_η : η (mopFunctor C) = 𝟙 _ := rfl

@[simp] lemma mopFunctor_μ (X Y : C) : μ (mopFunctor C) X Y = (β_ (mop X) (mop Y)).hom := rfl

@[simp] lemma mopFunctor_δ (X Y : C) : δ (mopFunctor C) X Y = (β_ (mop X) (mop Y)).inv := rfl


instance : (unmopFunctor C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun X Y ↦ β_ (unmop X) (unmop Y)
                                      /-
                                        C : Type u₁
                                        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                        inst✝⁷ : CategoryTheory.MonoidalCategory C
                                        inst✝⁶ : CategoryTheory.BraidedCategory C
                                        D : Type u₂
                                        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                        inst✝⁴ : CategoryTheory.MonoidalCategory D
                                        inst✝³ : CategoryTheory.BraidedCategory D
                                        E : Type u₃
                                        inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                        inst✝¹ : CategoryTheory.MonoidalCategory E
                                        inst✝ : CategoryTheory.BraidedCategory E
                                        X Y Z : CategoryTheory.MonoidalOpposite C
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                      -/
      associativity := fun X Y Z ↦ by simp [← yang_baxter_assoc] }
                                      /-
                                        🎉 no goals
                                      -/


@[simp] lemma unmopFunctor_ε : ε (unmopFunctor C) = 𝟙 _ := rfl

@[simp] lemma unmopFunctor_η : η (unmopFunctor C) = 𝟙 _ := rfl

@[simp] lemma unmopFunctor_μ (X Y : Cᴹᵒᵖ) :
    μ (unmopFunctor C) X Y = (β_ (unmop X) (unmop Y)).hom := rfl

@[simp] lemma unmopFunctor_δ (X Y : Cᴹᵒᵖ) :
    δ (unmopFunctor C) X Y = (β_ (unmop X) (unmop Y)).inv := rfl


/-- The identity functor on `C`, viewed as a functor from `C` to its
monoidal opposite, upgraded to a braided functor. -/
instance : (mopFunctor C).Braided where


/-- The identity functor on `C`, viewed as a functor from the
monoidal opposite of `C` to `C`, upgraded to a braided functor. -/
instance : (unmopFunctor C).Braided where


/-- The braided monoidal category obtained from `C` by replacing its braiding
`β_ X Y : X ⊗ Y ≅ Y ⊗ X` with the inverse `(β_ Y X)⁻¹ : X ⊗ Y ≅ Y ⊗ X`.
This corresponds to the automorphism of the braid group swapping
over-crossings and under-crossings. -/
abbrev reverseBraiding : BraidedCategory C where
  braiding X Y := (β_ Y X).symm
                                            /-
                                              C : Type u₁
                                              inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                              inst✝⁷ : CategoryTheory.MonoidalCategory C
                                              inst✝⁶ : CategoryTheory.BraidedCategory C
                                              D : Type u₂
                                              inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                              inst✝⁴ : CategoryTheory.MonoidalCategory D
                                              inst✝³ : CategoryTheory.BraidedCategory D
                                              E : Type u₃
                                              inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                              inst✝¹ : CategoryTheory.MonoidalCategory E
                                              inst✝ : CategoryTheory.BraidedCategory E
                                              X x✝¹ x✝ : C
                                              f : Quiver.Hom x✝¹ x✝
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                            -/
  braiding_naturality_right X {_ _} f := by simp
                                            /-
                                              🎉 no goals
                                            -/
                                           /-
                                             C : Type u₁
                                             inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                                             inst✝⁷ : CategoryTheory.MonoidalCategory C
                                             inst✝⁶ : CategoryTheory.BraidedCategory C
                                             D : Type u₂
                                             inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                             inst✝⁴ : CategoryTheory.MonoidalCategory D
                                             inst✝³ : CategoryTheory.BraidedCategory D
                                             E : Type u₃
                                             inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                             inst✝¹ : CategoryTheory.MonoidalCategory E
                                             inst✝ : CategoryTheory.BraidedCategory E
                                             x✝¹ x✝ : C
                                             f : Quiver.Hom x✝¹ x✝
                                             Z : C
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                           -/
  braiding_naturality_left {_ _} f Z := by simp
                                           /-
                                             🎉 no goals
                                           -/


lemma SymmetricCategory.reverseBraiding_eq (C : Type u₁) [Category.{v₁} C]
    [MonoidalCategory C] [i : SymmetricCategory C] :
    reverseBraiding C = i.toBraidedCategory := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    i : CategoryTheory.SymmetricCategory C
    ⊢ Eq (CategoryTheory.reverseBraiding C) CategoryTheory.SymmetricCategory.toBra …
  -/
  dsimp only [reverseBraiding]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    i : CategoryTheory.SymmetricCategory C
    ⊢ Eq { braiding := fun X Y => (CategoryTheory.BraidedCategory.braiding Y X).sy …
  -/
  congr
  /-
    case e_braiding
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    i : CategoryTheory.SymmetricCategory C
    ⊢ Eq (fun X Y => (CategoryTheory.BraidedCategory.braiding Y X).symm) CategoryT …
  -/
  funext X Y
  /-
    case e_braiding.h.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    i : CategoryTheory.SymmetricCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.BraidedCategory.braiding Y X).symm (CategoryTheory.Braide …
  -/
  exact Iso.ext (braiding_swap_eq_inv_braiding Y X).symm
  /-
    🎉 no goals
  -/


/-- The identity functor from `C` to `C`, where the codomain is given the
reversed braiding, upgraded to a braided functor. -/
def SymmetricCategory.equivReverseBraiding (C : Type u₁) [Category.{v₁} C]
    [MonoidalCategory C] [SymmetricCategory C] :=
  @Functor.Braided.mk C _ _ _ C _ _ (reverseBraiding C) (𝟭 C) _ <| by
    /-
      C✝ : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C✝
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C✝
      inst✝⁹ : CategoryTheory.BraidedCategory C✝
      D : Type u₂
      inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁷ : CategoryTheory.MonoidalCategory D
      inst✝⁶ : CategoryTheory.BraidedCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      inst✝³ : CategoryTheory.BraidedCategory E
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.MonoidalCategory C
      inst✝ : CategoryTheory.SymmetricCategory C
      ⊢ ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor. …
    -/
    intros; simp [reverseBraiding, braiding_swap_eq_inv_braiding]
            /-
              🎉 no goals
            -/


