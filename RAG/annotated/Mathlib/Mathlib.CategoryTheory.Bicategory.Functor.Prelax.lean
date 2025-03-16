/-- A `PrelaxFunctorStruct` between bicategories consists of functions between objects,
1-morphisms, and 2-morphisms. This structure will be extended to define `PrelaxFunctor`.
-/
structure PrelaxFunctorStruct extends Prefunctor B C where
  /-- The action of a lax prefunctor on 2-morphisms. -/
  map₂ {a b : B} {f g : a ⟶ b} : (f ⟶ g) → (map f ⟶ map g)


/-- Construct a lax prefunctor from a map on objects, and prefunctors between the corresponding
hom types. -/
@[simps]
def mkOfHomPrefunctors (F : B → C) (F' : (a : B) → (b : B) → Prefunctor (a ⟶ b) (F a ⟶ F b)) :
    PrelaxFunctorStruct B C where
  obj := F
  map {a b} := (F' a b).obj
  map₂ {a b} := (F' a b).map


/-- The identity lax prefunctor. -/
@[simps]
def id (B : Type u₁) [Quiver.{v₁ + 1} B] [∀ a b : B, Quiver.{w₁ + 1} (a ⟶ b)] :
    PrelaxFunctorStruct B B :=
  { Prefunctor.id B with map₂ := fun η => η }


instance : Inhabited (PrelaxFunctorStruct B B) :=
  ⟨PrelaxFunctorStruct.id B⟩


/-- Composition of lax prefunctors. -/
@[simps]
def comp (F : PrelaxFunctorStruct B C) (G : PrelaxFunctorStruct C D) : PrelaxFunctorStruct B D where
  toPrefunctor := F.toPrefunctor.comp G.toPrefunctor
  map₂ := fun η => G.map₂ (F.map₂ η)


/-- A prelax functor between bicategories is a lax prefunctor such that `map₂` is a functor.
This structure will be extended to define `LaxFunctor` and `OplaxFunctor`.
-/
structure PrelaxFunctor (B : Type u₁) [Bicategory.{w₁, v₁} B] (C : Type u₂) [Bicategory.{w₂, v₂} C]
    extends PrelaxFunctorStruct B C where
  /-- Prelax functors preserves identity 2-morphisms. -/
  map₂_id : ∀ {a b : B} (f : a ⟶ b), map₂ (𝟙 f) = 𝟙 (map f) := by aesop -- TODO: why not aesop_cat?
  /-- Prelax functors preserves compositions of 2-morphisms. -/
  map₂_comp : ∀ {a b : B} {f g h : a ⟶ b} (η : f ⟶ g) (θ : g ⟶ h),
      map₂ (η ≫ θ) = map₂ η ≫ map₂ θ := by aesop_cat


attribute [reassoc] map₂_comp

/-- Construct a prelax functor from a map on objects, and functors between the corresponding
hom types. -/
@[simps]
def mkOfHomFunctors (F : B → C) (F' : (a : B) → (b : B) → (a ⟶ b) ⥤ (F a ⟶ F b)) :
    PrelaxFunctor B C where
  toPrelaxFunctorStruct := PrelaxFunctorStruct.mkOfHomPrefunctors F fun a b => (F' a b).toPrefunctor
  map₂_id {a b} := (F' a b).map_id
  map₂_comp {a b} := (F' a b).map_comp


/-- The identity prelax functor. -/
@[simps]
def id (B : Type u₁) [Bicategory.{w₁, v₁} B] : PrelaxFunctor B B where
  toPrelaxFunctorStruct := PrelaxFunctorStruct.id B


/-- Composition of prelax functors. -/
@[simps]
def comp (G : PrelaxFunctor C D) : PrelaxFunctor B D where
  toPrelaxFunctorStruct := PrelaxFunctorStruct.comp F.toPrelaxFunctorStruct G.toPrelaxFunctorStruct


/-- Function between 1-morphisms as a functor. -/
@[simps]
def mapFunctor (a b : B) : (a ⟶ b) ⥤ (F.obj a ⟶ F.obj b) where
  obj f := F.map f
  map η := F.map₂ η


@[simp]
lemma mkOfHomFunctors_mapFunctor (F : B → C) (F' : (a : B) → (b : B) → (a ⟶ b) ⥤ (F a ⟶ F b))
    (a b : B) : (mkOfHomFunctors F F').mapFunctor a b = F' a b :=
  rfl


/-- A prelaxfunctor `F` sends 2-isomorphisms `η : f ≅ f` to 2-isomorphisms `F.map f ≅ F.map g`. -/
@[simps!]
abbrev map₂Iso {f g : a ⟶ b} (η : f ≅ g) : F.map f ≅ F.map g :=
  (F.mapFunctor a b).mapIso η


instance map₂_isIso {f g : a ⟶ b} (η : f ⟶ g) [IsIso η] : IsIso (F.map₂ η) :=
  (F.map₂Iso (asIso η)).isIso_hom


@[simp]
lemma map₂_inv {f g : a ⟶ b} (η : f ⟶ g) [IsIso η] : F.map₂ (inv η) = inv (F.map₂ η) := by
  /-
    B : Type u₁
    inst✝² : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝¹ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (F.map₂ (CategoryTheory.inv η)) (CategoryTheory.inv (F.map₂ η))
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    B : Type u₁
    inst✝² : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝¹ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map₂ η) (F.map₂ (CategoryTheory.in …
  -/
  simp [← F.map₂_comp η (inv η)]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
lemma map₂_hom_inv {f g : a ⟶ b} (η : f ≅ g) :
    F.map₂ η.hom ≫ F.map₂ η.inv = 𝟙 (F.map f) := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map₂ η.hom) (F.map₂ η.inv)) (Categ …
  -/
  rw [← F.map₂_comp, Iso.hom_inv_id, F.map₂_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map₂_hom_inv_isIso {f g : a ⟶ b} (η : f ⟶ g) [IsIso η] :
    F.map₂ η ≫ F.map₂ (inv η) = 𝟙 (F.map f) := by
  /-
    B : Type u₁
    inst✝² : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝¹ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map₂ η) (F.map₂ (CategoryTheory.in …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc, simp]
lemma map₂_inv_hom {f g : a ⟶ b} (η : f ≅ g) :
    F.map₂ η.inv ≫ F.map₂ η.hom = 𝟙 (F.map g) := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map₂ η.inv) (F.map₂ η.hom)) (Categ …
  -/
  rw [← F.map₂_comp, Iso.inv_hom_id, F.map₂_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map₂_inv_hom_isIso {f g : a ⟶ b} (η : f ⟶ g) [IsIso η] :
    F.map₂ (inv η) ≫ F.map₂ η = 𝟙 (F.map g) := by
  /-
    B : Type u₁
    inst✝² : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝¹ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map₂ (CategoryTheory.inv η)) (F.ma …
  -/
  simp
  /-
    🎉 no goals
  -/


