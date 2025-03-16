lemma parallelPair_initial_mk' {X Y : C} (f g : X ⟶ Y)
    (h₁ : ∀ Z, Nonempty (X ⟶ Z))
    (h₂ : ∀ ⦃Z : C⦄ (i j : X ⟶ Z),
      Zigzag (J := CostructuredArrow (parallelPair f g) Z)
        (mk (Y := zero) i) (mk (Y := zero) j)) :
    (parallelPair f g).Initial where
  out Z := by
    have : Nonempty (CostructuredArrow (parallelPair f g) Z) :=
      ⟨mk (Y := zero) (h₁ Z).some⟩
    have : ∀ (x : CostructuredArrow (parallelPair f g) Z), Zigzag x
      (mk (Y := zero) (h₁ Z).some) := by
        rintro ⟨(_|_), ⟨⟩, φ⟩
        · apply h₂
        · refine Zigzag.trans ?_ (h₂ (f ≫ φ) _)
          exact Zigzag.of_inv (homMk left)
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f g : Quiver.Hom X Y
      h₁ : ∀ (Z : C), Nonempty (Quiver.Hom X Z)
      h₂ : ∀ ⦃Z : C⦄ (i j : Quiver.Hom X Z), CategoryTheory.Zigzag (CategoryTheory.C …
      Z : C
      this✝ : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.para …
      this : ∀ (x : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.parallel …
      ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (CategoryTheory …
    -/
    exact zigzag_isConnected (fun x y => (this x).trans (this y).symm)
    /-
      🎉 no goals
    -/


lemma parallelPair_initial_mk {X Y : C} (f g : X ⟶ Y)
    (h₁ : ∀ Z, Nonempty (X ⟶ Z))
    (h₂ : ∀ ⦃Z : C⦄ (i j : X ⟶ Z), ∃ (a : Y ⟶ Z), i = f ≫ a ∧ j = g ≫ a) :
    (parallelPair f g).Initial :=
  parallelPair_initial_mk' f g h₁ (fun Z i j => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f g : Quiver.Hom X Y
      h₁ : ∀ (Z : C), Nonempty (Quiver.Hom X Z)
      h₂ : ∀ ⦃Z : C⦄ (i j : Quiver.Hom X Z), Exists fun a => And (Eq i (CategoryTheo …
      Z : C
      i j : Quiver.Hom X Z
      ⊢ CategoryTheory.Zigzag (CategoryTheory.CostructuredArrow.mk i) (CategoryTheor …
    -/
    obtain ⟨a, rfl, rfl⟩ := h₂ i j
    let f₁ : (mk (Y := zero) (f ≫ a) : CostructuredArrow (parallelPair f g) Z) ⟶ mk (Y := one) a :=
      homMk left
    let f₂ : (mk (Y := zero) (g ≫ a) : CostructuredArrow (parallelPair f g) Z) ⟶ mk (Y := one) a :=
      homMk right
    /-
      case intro.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f g : Quiver.Hom X Y
      h₁ : ∀ (Z : C), Nonempty (Quiver.Hom X Z)
      h₂ : ∀ ⦃Z : C⦄ (i j : Quiver.Hom X Z), Exists fun a => And (Eq i (CategoryTheo …
      Z : C
      a : Quiver.Hom Y Z
      f₁ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryS …
      f₂ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryS …
      ⊢ CategoryTheory.Zigzag (CategoryTheory.CostructuredArrow.mk (CategoryTheory.C …
    -/
    exact Zigzag.of_hom_inv f₁ f₂)
    /-
      🎉 no goals
    -/


