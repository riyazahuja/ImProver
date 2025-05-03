/-- A wrapper for promoting any type to a category,
with a unique morphisms between any two objects of the category.
-/
@[ext, aesop safe cases (rule_sets := [CategoryTheory])]
structure Codiscrete (α : Type u) where
  /-- A wrapper for promoting any type to a category,
  with a unique morphisms between any two objects of the category. -/
  as : α


@[simp]
theorem Codiscrete.mk_as {α : Type u} (X : Codiscrete α) : Codiscrete.mk X.as = X := rfl


/-- `Codiscrete α` is equivalent to the original type `α`. -/
@[simps]
def codiscreteEquiv {α : Type u} : Codiscrete α ≃ α where
  toFun := Codiscrete.as
  invFun := Codiscrete.mk
                 /-
                   α : Type u
                   ⊢ Function.LeftInverse CategoryTheory.Codiscrete.mk CategoryTheory.Codiscrete.as
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u
                    ⊢ Function.RightInverse CategoryTheory.Codiscrete.mk CategoryTheory.Codiscrete …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


instance {α : Type u} [DecidableEq α] : DecidableEq (Codiscrete α) :=
  codiscreteEquiv.decidableEq


instance (A : Type*) : Category (Codiscrete A) where
  Hom _ _ := Unit
  id _ := ⟨⟩
  comp _ _ := ⟨⟩


/-- Any function `C → A` lifts to a functor `C ⥤ Codiscrete A`.-/
def functor (F : C → A) : C ⥤ Codiscrete A where
  obj := Codiscrete.mk ∘ F
  map _ := ⟨⟩


/-- The underlying function `C → A` of a functor `C ⥤ Codiscrete A`. -/
def invFunctor (F : C ⥤ Codiscrete A) : C → A := Codiscrete.as ∘ F.obj


/-- Given two functors to a codiscrete category, there is a trivial natural transformation.-/
def natTrans {F G : C ⥤ Codiscrete A} : F ⟶ G where
  app _ := ⟨⟩


/-- Given two functors into a codiscrete category, the trivial natural transformation is an
natural isomorphism.-/
def natIso {F G : C ⥤ Codiscrete A} : F ≅ G where
  hom := natTrans
  inv := natTrans


/-- Every functor `F` to a codiscrete category is naturally isomorphic {(actually, equal)} to
  `Codiscrete.as ∘ F.obj`. -/
@[simps!]
def natIsoFunctor {F : C ⥤ Codiscrete A} : F ≅ functor (Codiscrete.as ∘ F.obj) := Iso.refl _


/-- A function induces a functor between codiscrete categories.-/
def functorOfFun {A B : Type*} (f : A → B) : Codiscrete A ⥤ Codiscrete B :=
  functor (f ∘ Codiscrete.as)


/-- A codiscrete category is equivalent to its opposite category. -/
def oppositeEquivalence (A : Type*) : (Codiscrete A)ᵒᵖ ≌ Codiscrete A where
  functor := functor (fun x ↦ Codiscrete.as x.unop)
  inverse := (functor (fun x ↦ Codiscrete.as x.unop)).rightOp
                                              /-
                                                A : Type u_1
                                                x✝ : Opposite (CategoryTheory.Codiscrete A)
                                                ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (Opposite (CategoryTheory.Cod …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  unitIso := NatIso.ofComponents (fun _ => by exact Iso.refl _)
             /-
               🎉 no goals
             -/
  counitIso := natIso


/-- `Codiscrete.functorToCat` turns a type into a codiscrete category-/
def functorToCat : Type u ⥤ Cat.{0,u} where
  obj A := Cat.of (Codiscrete A)
  map := functorOfFun


/-- For a category `C` and type `A`, there is an equivalence between functions `objects.obj C ⟶ A`
and functors `C ⥤ Codiscrete A`.-/
def equivFunctorToCodiscrete {C : Type u} [Category.{v} C] {A : Type w} :
    (C → A) ≃ (C ⥤ Codiscrete A) where
  toFun := functor
  invFun := invFunctor
  left_inv _ := rfl
  right_inv _ := rfl


/-- The functor that turns a type into a codiscrete category is right adjoint to the objects
functor.-/
def adj : objects ⊣ functorToCat := mkOfHomEquiv {
  homEquiv := fun _ _ => equivFunctorToCodiscrete
  homEquiv_naturality_left_symm := fun _ _ => rfl
  homEquiv_naturality_right := fun _ _ => rfl }


/-- Components of the unit of the adjunction `Cat.objects ⊣ Codiscrete.functorToCat`. -/
def unitApp (C : Type u) [Category.{v} C] : C ⥤ Codiscrete C := functor id


/-- Components of the counit of the adjunction `Cat.objects ⊣ Codiscrete.functorToCat` -/
def counitApp (A : Type u) : Codiscrete A → A := Codiscrete.as


lemma adj_unit_app (X : Cat.{0, u}) :
    adj.unit.app X = unitApp X := rfl


lemma adj_counit_app (A : Type u) :
    adj.counit.app A = counitApp A := rfl


/-- Left triangle equality of the adjunction `Cat.objects ⊣ Codiscrete.functorToCat`,
as a universe polymorphic statement. -/
lemma left_triangle_components (C : Type u) [Category.{v} C] :
    (counitApp C).comp (unitApp C).obj = id :=
  rfl


/-- Right triangle equality of the adjunction `Cat.objects ⊣ Codiscrete.functorToCat`,
stated using a composition of functors. -/
lemma right_triangle_components (X : Type u) :
    unitApp (Codiscrete X) ⋙ functorOfFun (counitApp X) = 𝟭 (Codiscrete X) :=
  rfl


