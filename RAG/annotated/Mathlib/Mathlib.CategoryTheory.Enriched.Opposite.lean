/-- For a `V`-category `C`, construct the opposite `V`-category structure on the type `Cᵒᵖ`
using the braiding in `V`. -/
instance EnrichedCategory.opposite : EnrichedCategory V Cᵒᵖ where
  Hom y x := EnrichedCategory.Hom x.unop y.unop
  id x := EnrichedCategory.id x.unop
  comp z y x := (β_ _ _).hom ≫ EnrichedCategory.comp (x.unop) (y.unop) (z.unop)
  id_comp _ _ := by
    simp only [braiding_naturality_left_assoc, braiding_tensorUnit_left,
      Category.assoc, Iso.inv_hom_id_assoc]
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x✝¹ x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    exact EnrichedCategory.comp_id _ _
    /-
      🎉 no goals
    -/
  comp_id _ _ := by
    simp only [braiding_naturality_right_assoc, braiding_tensorUnit_right,
      Category.assoc, Iso.inv_hom_id_assoc]
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x✝¹ x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    exact EnrichedCategory.id_comp _ _
    /-
      🎉 no goals
    -/
  assoc _ _ _ _ := by
    simp only [braiding_naturality_left_assoc,
      MonoidalCategory.whiskerLeft_comp, Category.assoc]
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x✝³ x✝² x✝¹ x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [← EnrichedCategory.assoc]
    simp only [braiding_tensor_left, Category.assoc, Iso.inv_hom_id_assoc,
      braiding_naturality_right_assoc, braiding_tensor_right]


/-- Unfold the definition of composition in the enriched opposite category. -/
@[reassoc]
lemma eComp_op_eq {C : Type u} [EnrichedCategory V C] (x y z : Cᵒᵖ) :
    eComp V z y x = (β_ _ _).hom ≫ eComp V x.unop y.unop z.unop :=
  rfl


/-- When composing a tensor product of morphisms with the `V`-composition morphism in `Cᵒᵖ`,
this re-writes the `V`-composition to be in `C` and moves the braiding to the left. -/
@[reassoc]
lemma tensorHom_eComp_op_eq {C : Type u} [EnrichedCategory V C] {x y z : Cᵒᵖ} {v w : V}
    (f : v ⟶ EnrichedCategory.Hom z y) (g : w ⟶ EnrichedCategory.Hom y x) :
    (f ⊗ g) ≫ eComp V z y x = (β_ v w).hom ≫ (g ⊗ f) ≫ eComp V x.unop y.unop z.unop := by
  /-
    V : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} V
    inst✝² : CategoryTheory.MonoidalCategory V
    inst✝¹ : CategoryTheory.BraidedCategory V
    C : Type u
    inst✝ : CategoryTheory.EnrichedCategory V C
    x y z : Opposite C
    v w : V
    f : Quiver.Hom v (CategoryTheory.EnrichedCategory.Hom z y)
    g : Quiver.Hom w (CategoryTheory.EnrichedCategory.Hom y x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [eComp_op_eq]
  /-
    V : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} V
    inst✝² : CategoryTheory.MonoidalCategory V
    inst✝¹ : CategoryTheory.BraidedCategory V
    C : Type u
    inst✝ : CategoryTheory.EnrichedCategory V C
    x y z : Opposite C
    v w : V
    f : Quiver.Hom v (CategoryTheory.EnrichedCategory.Hom z y)
    g : Quiver.Hom w (CategoryTheory.EnrichedCategory.Hom y x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  exact braiding_naturality_assoc f g _
  /-
    🎉 no goals
  -/

-- This section establishes the equivalence on underlying categories

/-- The functor going from the underlying category of the enriched category `Cᵒᵖ`
to the opposite of the underlying category of the enriched category `C`. -/
def forgetEnrichmentOppositeEquivalence.functor :
    ForgetEnrichment V Cᵒᵖ ⥤ (ForgetEnrichment V C)ᵒᵖ where
  obj x := x
  map {x y} f := f.op
  map_comp {x y z} f g := by
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : CategoryTheory.ForgetEnrichment V (Opposite C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      ⊢ Eq ({ obj := fun x => x, map := fun {x y} f => f.op }.map (CategoryTheory.Ca …
    -/
    have : (f ≫ g) = homTo V (f ≫ g) := rfl
    rw [this, forgetEnrichment_comp, Category.assoc, tensorHom_eComp_op_eq,
      leftUnitor_inv_braiding_assoc, ← unitors_inv_equal, ← Category.assoc]
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : CategoryTheory.ForgetEnrichment V (Opposite C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      this : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.ForgetEnric …
      ⊢ Eq ({ obj := fun x => x, map := fun {x y} f => f.op }.map (CategoryTheory.Ca …
    -/
    congr 1
    /-
      🎉 no goals
    -/


/-- The functor going from the opposite of the underlying category of the enriched category `C`
to the underlying category of the enriched category `Cᵒᵖ`. -/
def forgetEnrichmentOppositeEquivalence.inverse :
    (ForgetEnrichment V C)ᵒᵖ ⥤ ForgetEnrichment V Cᵒᵖ where
  obj x := x
  map {x y} f := f.unop
  map_comp {x y z} f g := by
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : Opposite (CategoryTheory.ForgetEnrichment V C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      ⊢ Eq ({ obj := fun x => x, map := fun {x y} f => f.unop }.map (CategoryTheory. …
    -/
    have : g.unop ≫ f.unop = homTo V (g.unop ≫ f.unop) := rfl
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : Opposite (CategoryTheory.ForgetEnrichment V C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      this : Eq (CategoryTheory.CategoryStruct.comp g.unop f.unop) (CategoryTheory.F …
      ⊢ Eq ({ obj := fun x => x, map := fun {x y} f => f.unop }.map (CategoryTheory. …
    -/
    dsimp
    rw [this, forgetEnrichment_comp, Category.assoc, unitors_inv_equal,
      ← leftUnitor_inv_braiding_assoc]
    have : (β_ _ _).hom ≫ (homTo V g.unop ⊗ homTo V f.unop) ≫
      eComp V («to» V z.unop) («to» V y.unop) («to» V x.unop) =
      ((homTo V f.unop) ⊗ (homTo V g.unop)) ≫ eComp V x y z := (tensorHom_eComp_op_eq V _ _).symm
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : Opposite (CategoryTheory.ForgetEnrichment V C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      this✝ : Eq (CategoryTheory.CategoryStruct.comp g.unop f.unop) (CategoryTheory. …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [this, ← Category.assoc]
    /-
      V : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} V
      inst✝² : CategoryTheory.MonoidalCategory V
      inst✝¹ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝ : CategoryTheory.EnrichedCategory V C
      x y z : Opposite (CategoryTheory.ForgetEnrichment V C)
      f : Quiver.Hom x y
      g : Quiver.Hom y z
      this✝ : Eq (CategoryTheory.CategoryStruct.comp g.unop f.unop) (CategoryTheory. …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      🎉 no goals
    -/


/-- The equivalence between the underlying category of the enriched category `Cᵒᵖ` and
the opposite of the underlying category of the enriched category `C`. -/
@[simps]
def forgetEnrichmentOppositeEquivalence : ForgetEnrichment V Cᵒᵖ ≌ (ForgetEnrichment V C)ᵒᵖ where
  functor := forgetEnrichmentOppositeEquivalence.functor V C
  inverse := forgetEnrichmentOppositeEquivalence.inverse V C
             /-
               V : Type u₁
               inst✝³ : CategoryTheory.Category.{v₁, u₁} V
               inst✝² : CategoryTheory.MonoidalCategory V
               inst✝¹ : CategoryTheory.BraidedCategory V
               C : Type u
               inst✝ : CategoryTheory.EnrichedCategory V C
               ⊢ ∀ {X Y : CategoryTheory.ForgetEnrichment V (Opposite C)} (f : Quiver.Hom X Y …
             -/
  unitIso := NatIso.ofComponents (fun _ ↦ Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 V : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} V
                 inst✝² : CategoryTheory.MonoidalCategory V
                 inst✝¹ : CategoryTheory.BraidedCategory V
                 C : Type u
                 inst✝ : CategoryTheory.EnrichedCategory V C
                 ⊢ ∀ {X Y : Opposite (CategoryTheory.ForgetEnrichment V C)} (f : Quiver.Hom X Y …
               -/
  counitIso := NatIso.ofComponents (fun _ ↦ Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- If `D` is an enriched ordinary category then `Dᵒᵖ` is an enriched ordinary category. -/
instance EnrichedOrdinaryCategory.opposite {D : Type u} [Category.{v} D]
    [EnrichedOrdinaryCategory V D] : EnrichedOrdinaryCategory V Dᵒᵖ where
  homEquiv := Quiver.Hom.opEquiv.symm.trans homEquiv
  homEquiv_id x := homEquiv_id (x.unop)
  homEquiv_comp f g := by
    /-
      V : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      inst✝³ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝² : CategoryTheory.EnrichedCategory V C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V D
      X✝ Y✝ Z✝ : Opposite D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ((fun {X Y} => Quiver.Hom.opEquiv.symm.trans CategoryTheory.EnrichedOrdin …
    -/
    simp only [unop_comp, tensorHom_eComp_op_eq, leftUnitor_inv_braiding_assoc, ← unitors_inv_equal]
    /-
      V : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      inst✝³ : CategoryTheory.BraidedCategory V
      C : Type u
      inst✝² : CategoryTheory.EnrichedCategory V C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V D
      X✝ Y✝ Z✝ : Opposite D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ((Quiver.Hom.opEquiv.symm.trans CategoryTheory.EnrichedOrdinaryCategory.h …
    -/
    exact homEquiv_comp g.unop f.unop
    /-
      🎉 no goals
    -/


