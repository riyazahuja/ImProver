/-- An isomorphism (a.k.a. an invertible morphism) between two objects of a category.
The inverse morphism is bundled.

See also `CategoryTheory.Core` for the category with the same objects and isomorphisms playing
the role of morphisms.

See <https://stacks.math.columbia.edu/tag/0017>.
-/
structure Iso {C : Type u} [Category.{v} C] (X Y : C) where
  /-- The forward direction of an isomorphism. -/
  hom : X ⟶ Y
  /-- The backwards direction of an isomorphism. -/
  inv : Y ⟶ X
  /-- Composition of the two directions of an isomorphism is the identity on the source. -/
  hom_inv_id : hom ≫ inv = 𝟙 X := by aesop_cat
  /-- Composition of the two directions of an isomorphism in reverse order
  is the identity on the target. -/
  inv_hom_id : inv ≫ hom = 𝟙 Y := by aesop_cat


attribute [reassoc (attr := simp)] Iso.hom_inv_id Iso.inv_hom_id


/-- Notation for an isomorphism in a category. -/
infixr:10 " ≅ " => Iso -- type as \cong or \iso


@[ext]
theorem ext ⦃α β : X ≅ Y⦄ (w : α.hom = β.hom) : α = β :=
  suffices α.inv = β.inv by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      α β : CategoryTheory.Iso X Y
      w : Eq α.hom β.hom
      this : Eq α.inv β.inv
      ⊢ Eq α β
    -/
    cases α
    /-
      case mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      β : CategoryTheory.Iso X Y
      hom✝ : Quiver.Hom X Y
      inv✝ : Quiver.Hom Y X
      hom_inv_id✝ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝) (CategoryTheor …
      inv_hom_id✝ : Eq (CategoryTheory.CategoryStruct.comp inv✝ hom✝) (CategoryTheor …
      w : Eq { hom := hom✝, inv := inv✝, hom_inv_id := hom_inv_id✝, inv_hom_id := in …
      this : Eq { hom := hom✝, inv := inv✝, hom_inv_id := hom_inv_id✝, inv_hom_id := …
      ⊢ Eq { hom := hom✝, inv := inv✝, hom_inv_id := hom_inv_id✝, inv_hom_id := inv_ …
    -/
    cases β
    /-
      case mk.mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hom✝¹ : Quiver.Hom X Y
      inv✝¹ : Quiver.Hom Y X
      hom_inv_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp hom✝¹ inv✝¹) (CategoryTh …
      inv_hom_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp inv✝¹ hom✝¹) (CategoryTh …
      hom✝ : Quiver.Hom X Y
      inv✝ : Quiver.Hom Y X
      hom_inv_id✝ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝) (CategoryTheor …
      inv_hom_id✝ : Eq (CategoryTheory.CategoryStruct.comp inv✝ hom✝) (CategoryTheor …
      w : Eq { hom := hom✝¹, inv := inv✝¹, hom_inv_id := hom_inv_id✝¹, inv_hom_id := …
      this : Eq { hom := hom✝¹, inv := inv✝¹, hom_inv_id := hom_inv_id✝¹, inv_hom_id …
      ⊢ Eq { hom := hom✝¹, inv := inv✝¹, hom_inv_id := hom_inv_id✝¹, inv_hom_id := i …
    -/
    cases w
                                          /-
                                            C : Type u
                                            inst✝ : CategoryTheory.Category.{v, u} C
                                            X Y : C
                                            α β : CategoryTheory.Iso X Y
                                            w : Eq α.hom β.hom
                                            ⊢ Eq α.inv (CategoryTheory.CategoryStruct.comp α.inv (CategoryTheory.CategoryS …
                                          -/
    /-
      case mk.mk.refl
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hom✝ : Quiver.Hom X Y
      inv✝¹ : Quiver.Hom Y X
      hom_inv_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝¹) (CategoryThe …
      inv_hom_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp inv✝¹ hom✝) (CategoryThe …
      inv✝ : Quiver.Hom Y X
      hom_inv_id✝ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝) (CategoryTheor …
      inv_hom_id✝ : Eq (CategoryTheory.CategoryStruct.comp inv✝ hom✝) (CategoryTheor …
      this : Eq { hom := hom✝, inv := inv✝¹, hom_inv_id := hom_inv_id✝¹, inv_hom_id  …
      ⊢ Eq { hom := hom✝, inv := inv✝¹, hom_inv_id := hom_inv_id✝¹, inv_hom_id := in …
    -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            C : Type u
                                            inst✝ : CategoryTheory.Category.{v, u} C
                                            X Y : C
                                            α β : CategoryTheory.Iso X Y
                                            w : Eq α.hom β.hom
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp α.inv (CategoryTheory.CategoryStruct. …
                                          -/
    cases this
                                          /-
                                            🎉 no goals
                                          -/
                                           /-
                                             C : Type u
                                             inst✝ : CategoryTheory.Category.{v, u} C
                                             X Y : C
                                             α β : CategoryTheory.Iso X Y
                                             w : Eq α.hom β.hom
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp α …
                                           -/
    /-
      case mk.mk.refl.refl
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hom✝ : Quiver.Hom X Y
      inv✝ : Quiver.Hom Y X
      hom_inv_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝) (CategoryTheo …
      inv_hom_id✝¹ : Eq (CategoryTheory.CategoryStruct.comp inv✝ hom✝) (CategoryTheo …
      hom_inv_id✝ : Eq (CategoryTheory.CategoryStruct.comp hom✝ inv✝) (CategoryTheor …
      inv_hom_id✝ : Eq (CategoryTheory.CategoryStruct.comp inv✝ hom✝) (CategoryTheor …
      ⊢ Eq { hom := hom✝, inv := inv✝, hom_inv_id := hom_inv_id✝¹, inv_hom_id := inv …
    -/
                                           /-
                                             🎉 no goals
                                           -/
    rfl
    /-
      🎉 no goals
    -/
  calc
    α.inv = α.inv ≫ β.hom ≫ β.inv   := by rw [Iso.hom_inv_id, Category.comp_id]
    _     = (α.inv ≫ α.hom) ≫ β.inv := by rw [Category.assoc, ← w]
    _     = β.inv                    := by rw [Iso.inv_hom_id, Category.id_comp]


/-- Inverse isomorphism. -/
@[symm]
def symm (I : X ≅ Y) : Y ≅ X where
  hom := I.inv
  inv := I.hom


@[simp]
theorem symm_hom (α : X ≅ Y) : α.symm.hom = α.inv :=
  rfl


@[simp]
theorem symm_inv (α : X ≅ Y) : α.symm.inv = α.hom :=
  rfl


@[simp]
theorem symm_mk {X Y : C} (hom : X ⟶ Y) (inv : Y ⟶ X) (hom_inv_id) (inv_hom_id) :
    Iso.symm { hom, inv, hom_inv_id := hom_inv_id, inv_hom_id := inv_hom_id } =
      { hom := inv, inv := hom, hom_inv_id := inv_hom_id, inv_hom_id := hom_inv_id } :=
  rfl


@[simp]
theorem symm_symm_eq {X Y : C} (α : X ≅ Y) : α.symm.symm = α := rfl


@[simp]
theorem symm_eq_iff {X Y : C} {α β : X ≅ Y} : α.symm = β.symm ↔ α = β :=
  ⟨fun h => symm_symm_eq α ▸ symm_symm_eq β ▸ congr_arg symm h, congr_arg symm⟩


theorem nonempty_iso_symm (X Y : C) : Nonempty (X ≅ Y) ↔ Nonempty (Y ≅ X) :=
  ⟨fun h => ⟨h.some.symm⟩, fun h => ⟨h.some.symm⟩⟩


/-- Identity isomorphism. -/
@[refl, simps]
def refl (X : C) : X ≅ X where
  hom := 𝟙 X
  inv := 𝟙 X


instance : Inhabited (X ≅ X) := ⟨Iso.refl X⟩


theorem nonempty_iso_refl (X : C) : Nonempty (X ≅ X) := ⟨default⟩


@[simp]
theorem refl_symm (X : C) : (Iso.refl X).symm = Iso.refl X := rfl

-- Porting note: It seems that the trans `trans` attribute isn't working properly
-- in this case, so we have to manually add a `Trans` instance (with a `simps` tag).

/-- Composition of two isomorphisms -/
@[trans, simps]
def trans (α : X ≅ Y) (β : Y ≅ Z) : X ≅ Z where
  hom := α.hom ≫ β.hom
  inv := β.inv ≫ α.inv


@[simps]
instance instTransIso : Trans (α := C) (· ≅ ·) (· ≅ ·) (· ≅ ·) where
  trans := trans


/-- Notation for composition of isomorphisms. -/
infixr:80 " ≪≫ " => Iso.trans -- type as `\ll \gg`.


@[simp]
theorem trans_mk {X Y Z : C} (hom : X ⟶ Y) (inv : Y ⟶ X) (hom_inv_id) (inv_hom_id)
    (hom' : Y ⟶ Z) (inv' : Z ⟶ Y) (hom_inv_id') (inv_hom_id') (hom_inv_id'') (inv_hom_id'') :
    Iso.trans ⟨hom, inv, hom_inv_id, inv_hom_id⟩ ⟨hom', inv', hom_inv_id', inv_hom_id'⟩ =
     ⟨hom ≫ hom', inv' ≫ inv, hom_inv_id'', inv_hom_id''⟩ :=
  rfl


@[simp]
theorem trans_symm (α : X ≅ Y) (β : Y ≅ Z) : (α ≪≫ β).symm = β.symm ≪≫ α.symm :=
  rfl


@[simp]
theorem trans_assoc {Z' : C} (α : X ≅ Y) (β : Y ≅ Z) (γ : Z ≅ Z') :
    (α ≪≫ β) ≪≫ γ = α ≪≫ β ≪≫ γ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z Z' : C
    α : CategoryTheory.Iso X Y
    β : CategoryTheory.Iso Y Z
    γ : CategoryTheory.Iso Z Z'
    ⊢ Eq ((α.trans β).trans γ) (α.trans (β.trans γ))
  -/
  ext; simp only [trans_hom, Category.assoc]
       /-
         🎉 no goals
       -/


@[simp]
                                                           /-
                                                             C : Type u
                                                             inst✝ : CategoryTheory.Category.{v, u} C
                                                             X Y : C
                                                             α : CategoryTheory.Iso X Y
                                                             ⊢ Eq ((CategoryTheory.Iso.refl X).trans α) α
                                                           -/
theorem refl_trans (α : X ≅ Y) : Iso.refl X ≪≫ α = α := by ext; apply Category.id_comp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
                                                           /-
                                                             C : Type u
                                                             inst✝ : CategoryTheory.Category.{v, u} C
                                                             X Y : C
                                                             α : CategoryTheory.Iso X Y
                                                             ⊢ Eq (α.trans (CategoryTheory.Iso.refl Y)) α
                                                           -/
theorem trans_refl (α : X ≅ Y) : α ≪≫ Iso.refl Y = α := by ext; apply Category.comp_id
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem symm_self_id (α : X ≅ Y) : α.symm ≪≫ α = Iso.refl Y :=
  ext α.inv_hom_id


@[simp]
theorem self_symm_id (α : X ≅ Y) : α ≪≫ α.symm = Iso.refl X :=
  ext α.hom_inv_id


@[simp]
theorem symm_self_id_assoc (α : X ≅ Y) (β : Y ≅ Z) : α.symm ≪≫ α ≪≫ β = β := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    α : CategoryTheory.Iso X Y
    β : CategoryTheory.Iso Y Z
    ⊢ Eq (α.symm.trans (α.trans β)) β
  -/
  rw [← trans_assoc, symm_self_id, refl_trans]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_symm_id_assoc (α : X ≅ Y) (β : X ≅ Z) : α ≪≫ α.symm ≪≫ β = β := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    α : CategoryTheory.Iso X Y
    β : CategoryTheory.Iso X Z
    ⊢ Eq (α.trans (α.symm.trans β)) β
  -/
  rw [← trans_assoc, self_symm_id, refl_trans]
  /-
    🎉 no goals
  -/


theorem inv_comp_eq (α : X ≅ Y) {f : X ⟶ Z} {g : Y ⟶ Z} : α.inv ≫ f = g ↔ f = α.hom ≫ g :=
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y Z : C
                 α : CategoryTheory.Iso X Y
                 f : Quiver.Hom X Z
                 g : Quiver.Hom Y Z
                 H : Eq (CategoryTheory.CategoryStruct.comp α.inv f) g
                 ⊢ Eq f (CategoryTheory.CategoryStruct.comp α.hom g)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun H => by simp [H.symm], fun H => by simp [H]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem eq_inv_comp (α : X ≅ Y) {f : X ⟶ Z} {g : Y ⟶ Z} : g = α.inv ≫ f ↔ α.hom ≫ g = f :=
  (inv_comp_eq α.symm).symm


theorem comp_inv_eq (α : X ≅ Y) {f : Z ⟶ Y} {g : Z ⟶ X} : f ≫ α.inv = g ↔ f = g ≫ α.hom :=
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y Z : C
                 α : CategoryTheory.Iso X Y
                 f : Quiver.Hom Z Y
                 g : Quiver.Hom Z X
                 H : Eq (CategoryTheory.CategoryStruct.comp f α.inv) g
                 ⊢ Eq f (CategoryTheory.CategoryStruct.comp g α.hom)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun H => by simp [H.symm], fun H => by simp [H]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem eq_comp_inv (α : X ≅ Y) {f : Z ⟶ Y} {g : Z ⟶ X} : g = f ≫ α.inv ↔ g ≫ α.hom = f :=
  (comp_inv_eq α.symm).symm


theorem inv_eq_inv (f g : X ≅ Y) : f.inv = g.inv ↔ f.hom = g.hom :=
                                                                                     /-
                                                                                       C : Type u
                                                                                       inst✝ : CategoryTheory.Category.{v, u} C
                                                                                       X Y : C
                                                                                       f✝ g✝ : CategoryTheory.Iso X Y
                                                                                       X✝ Y✝ : C
                                                                                       f g : CategoryTheory.Iso X✝ Y✝
                                                                                       h : Eq f.hom g.hom
                                                                                       ⊢ Eq f.inv g.inv
                                                                                     -/
  have : ∀ {X Y : C} (f g : X ≅ Y), f.hom = g.hom → f.inv = g.inv := fun f g h => by rw [ext h]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  ⟨this f.symm g.symm, this f g⟩


theorem hom_comp_eq_id (α : X ≅ Y) {f : Y ⟶ X} : α.hom ≫ f = 𝟙 X ↔ f = α.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    f : Quiver.Hom Y X
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp α.hom f) (CategoryTheory.Categor …
  -/
  rw [← eq_inv_comp, comp_id]
  /-
    🎉 no goals
  -/


theorem comp_hom_eq_id (α : X ≅ Y) {f : Y ⟶ X} : f ≫ α.hom = 𝟙 Y ↔ f = α.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    f : Quiver.Hom Y X
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f α.hom) (CategoryTheory.Categor …
  -/
  rw [← eq_comp_inv, id_comp]
  /-
    🎉 no goals
  -/


theorem inv_comp_eq_id (α : X ≅ Y) {f : X ⟶ Y} : α.inv ≫ f = 𝟙 Y ↔ f = α.hom :=
  hom_comp_eq_id α.symm


theorem comp_inv_eq_id (α : X ≅ Y) {f : X ⟶ Y} : f ≫ α.inv = 𝟙 X ↔ f = α.hom :=
  comp_hom_eq_id α.symm


theorem hom_eq_inv (α : X ≅ Y) (β : Y ≅ X) : α.hom = β.inv ↔ β.hom = α.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    β : CategoryTheory.Iso Y X
    ⊢ Iff (Eq α.hom β.inv) (Eq β.hom α.inv)
  -/
  erw [inv_eq_inv α.symm β, eq_comm]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    β : CategoryTheory.Iso Y X
    ⊢ Iff (Eq β.hom α.symm.hom) (Eq β.hom α.inv)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The bijection `(Z ⟶ X) ≃ (Z ⟶ Y)` induced by `α : X ≅ Y`. -/
@[simps]
def homToEquiv (α : X ≅ Y) {Z : C} : (Z ⟶ X) ≃ (Z ⟶ Y) where
  toFun f := f ≫ α.hom
  invFun g := g ≫ α.inv
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y Z✝ : C
                   α : CategoryTheory.Iso X Y
                   Z : C
                   ⊢ Function.LeftInverse (fun g => CategoryTheory.CategoryStruct.comp g α.inv) f …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    X Y Z✝ : C
                    α : CategoryTheory.Iso X Y
                    Z : C
                    ⊢ Function.RightInverse (fun g => CategoryTheory.CategoryStruct.comp g α.inv)  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- The bijection `(X ⟶ Z) ≃ (Y ⟶ Z)` induced by `α : X ≅ Y`. -/
@[simps]
def homFromEquiv (α : X ≅ Y) {Z : C} : (X ⟶ Z) ≃ (Y ⟶ Z) where
  toFun f := α.inv ≫ f
  invFun g := α.hom ≫ g
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y Z✝ : C
                   α : CategoryTheory.Iso X Y
                   Z : C
                   ⊢ Function.LeftInverse (fun g => CategoryTheory.CategoryStruct.comp α.hom g) f …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    X Y Z✝ : C
                    α : CategoryTheory.Iso X Y
                    Z : C
                    ⊢ Function.RightInverse (fun g => CategoryTheory.CategoryStruct.comp α.hom g)  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- `IsIso` typeclass expressing that a morphism is invertible. -/
class IsIso (f : X ⟶ Y) : Prop where
  /-- The existence of an inverse morphism. -/
  out : ∃ inv : Y ⟶ X, f ≫ inv = 𝟙 X ∧ inv ≫ f = 𝟙 Y


/-- The inverse of a morphism `f` when we have `[IsIso f]`.
-/
noncomputable def inv (f : X ⟶ Y) [I : IsIso f] : Y ⟶ X :=
  Classical.choose I.1


@[simp]
theorem hom_inv_id (f : X ⟶ Y) [I : IsIso f] : f ≫ inv f = 𝟙 X :=
  (Classical.choose_spec I.1).left


@[simp]
theorem inv_hom_id (f : X ⟶ Y) [I : IsIso f] : inv f ≫ f = 𝟙 Y :=
  (Classical.choose_spec I.1).right

-- FIXME putting @[reassoc] on the `hom_inv_id` above somehow unfolds `inv`
-- This happens even if we make `inv` irreducible!
-- I don't understand how this is happening: it is likely a bug.

-- attribute [reassoc] hom_inv_id inv_hom_id
-- #print hom_inv_id_assoc
--   theorem CategoryTheory.IsIso.hom_inv_id_assoc {X Y : C} (f : X ⟶ Y) [I : IsIso f]
--     {Z : C} (h : X ⟶ Z),
--     f ≫ Classical.choose (_ : Exists fun inv ↦ f ≫ inv = 𝟙 X ∧ inv ≫ f = 𝟙 Y) ≫ h = h := ...


@[simp]
theorem hom_inv_id_assoc (f : X ⟶ Y) [I : IsIso f] {Z} (g : X ⟶ Z) : f ≫ inv f ≫ g = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.IsIso f
    Z : C
    g : Quiver.Hom X Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
  -/
  simp [← Category.assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_hom_id_assoc (f : X ⟶ Y) [I : IsIso f] {Z} (g : Y ⟶ Z) : inv f ≫ f ≫ g = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.IsIso f
    Z : C
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) (CategoryTheor …
  -/
  simp [← Category.assoc]
  /-
    🎉 no goals
  -/


lemma Iso.isIso_hom (e : X ≅ Y) : IsIso e.hom :=
             /-
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X Y : C
               e : CategoryTheory.Iso X Y
               ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom e.inv) (CategoryTheory.Category …
             -/
             /-
               🎉 no goals
             -/
  ⟨e.inv, by simp, by simp⟩
                      /-
                        🎉 no goals
                      -/


lemma Iso.isIso_inv (e : X ≅ Y) : IsIso e.inv := e.symm.isIso_hom


/-- Reinterpret a morphism `f` with an `IsIso f` instance as an `Iso`. -/
noncomputable def asIso (f : X ⟶ Y) [IsIso f] : X ≅ Y :=
  ⟨f, inv f, hom_inv_id f, inv_hom_id f⟩

-- Porting note: the `IsIso f` argument had been instance implicit,
-- but we've changed it to implicit as a `rw` in `Mathlib.CategoryTheory.Closed.Functor`
-- was failing to generate it by typeclass search.

@[simp]
theorem asIso_hom (f : X ⟶ Y) {_ : IsIso f} : (asIso f).hom = f :=
  rfl

-- Porting note: the `IsIso f` argument had been instance implicit,
-- but we've changed it to implicit as a `rw` in `Mathlib.CategoryTheory.Closed.Functor`
-- was failing to generate it by typeclass search.

@[simp]
theorem asIso_inv (f : X ⟶ Y) {_ : IsIso f} : (asIso f).inv = inv f :=
  rfl


instance (priority := 100) epi_of_iso (f : X ⟶ Y) [IsIso f] : Epi f where
  left_cancellation g h w := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      Z✝ : C
      g h : Quiver.Hom Y Z✝
      w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
      ⊢ Eq g h
    -/
    rw [← IsIso.inv_hom_id_assoc f g, w, IsIso.inv_hom_id_assoc f h]
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

instance (priority := 100) mono_of_iso (f : X ⟶ Y) [IsIso f] : Mono f where
  right_cancellation g h w := by
    rw [← Category.comp_id g, ← Category.comp_id h, ← IsIso.hom_inv_id f,
      ← Category.assoc, w, ← Category.assoc]


@[aesop apply safe (rule_sets := [CategoryTheory])]
theorem inv_eq_of_hom_inv_id {f : X ⟶ Y} [IsIso f] {g : Y ⟶ X} (hom_inv_id : f ≫ g = 𝟙 X) :
    inv f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom Y X
    hom_inv_id : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.inv f) g
  -/
  apply (cancel_epi f).mp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom Y X
    hom_inv_id : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.inv f)) (CategoryTh …
  -/
  simp [hom_inv_id]
  /-
    🎉 no goals
  -/


theorem inv_eq_of_inv_hom_id {f : X ⟶ Y} [IsIso f] {g : Y ⟶ X} (inv_hom_id : g ≫ f = 𝟙 Y) :
    inv f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom Y X
    inv_hom_id : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.inv f) g
  -/
  apply (cancel_mono f).mp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom Y X
    inv_hom_id : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) f) (CategoryTh …
  -/
  simp [inv_hom_id]
  /-
    🎉 no goals
  -/


@[aesop apply safe (rule_sets := [CategoryTheory])]
theorem eq_inv_of_hom_inv_id {f : X ⟶ Y} [IsIso f] {g : Y ⟶ X} (hom_inv_id : f ≫ g = 𝟙 X) :
    g = inv f :=
  (inv_eq_of_hom_inv_id hom_inv_id).symm


theorem eq_inv_of_inv_hom_id {f : X ⟶ Y} [IsIso f] {g : Y ⟶ X} (inv_hom_id : g ≫ f = 𝟙 Y) :
    g = inv f :=
  (inv_eq_of_inv_hom_id inv_hom_id).symm


                                               /-
                                                 C : Type u
                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                 X✝ Y Z X : C
                                                 ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.i …
                                               -/
instance id (X : C) : IsIso (𝟙 X) := ⟨⟨𝟙 X, by simp⟩⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[deprecated (since := "2024-05-15")] alias of_iso := CategoryTheory.Iso.isIso_hom

@[deprecated (since := "2024-05-15")] alias of_iso_inv := CategoryTheory.Iso.isIso_inv


instance inv_isIso [IsIso f] : IsIso (inv f) :=
  (asIso f).isIso_inv

/- The following instance has lower priority for the following reason:
Suppose we are given `f : X ≅ Y` with `X Y : Type u`.
Without the lower priority, typeclass inference cannot deduce `IsIso f.hom`
because `f.hom` is defeq to `(fun x ↦ x) ≫ f.hom`, triggering a loop. -/

instance (priority := 900) comp_isIso [IsIso f] [IsIso h] : IsIso (f ≫ h) :=
  (asIso f ≪≫ asIso h).isIso_hom


/--
The composition of isomorphisms is an isomorphism. Here the arguments of type `IsIso` are
explicit, to make this easier to use with the `refine` tactic, for instance.
-/
lemma comp_isIso' (_ : IsIso f) (_ : IsIso h) : IsIso (f ≫ h) := inferInstance


@[simp]
theorem inv_id : inv (𝟙 X) = 𝟙 X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.id X)) (CategoryTheory …
  -/
  apply inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem inv_comp [IsIso f] [IsIso h] : inv (f ≫ h) = inv h ≫ inv f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso h
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp f h)) (CategoryTh …
  -/
  apply inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso h
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_inv [IsIso f] : inv (inv f) = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.inv f)) f
  -/
  apply inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) f) (CategoryTh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem Iso.inv_inv (f : X ≅ Y) : inv f.inv = f.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.inv f.inv) f.hom
  -/
  apply inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv f.hom) (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem Iso.inv_hom (f : X ≅ Y) : inv f.hom = f.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.inv f.hom) f.inv
  -/
  apply inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom f.inv) (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_comp_eq (α : X ⟶ Y) [IsIso α] {f : X ⟶ Z} {g : Y ⟶ Z} : inv α ≫ f = g ↔ f = α ≫ g :=
  (asIso α).inv_comp_eq


@[simp]
theorem eq_inv_comp (α : X ⟶ Y) [IsIso α] {f : X ⟶ Z} {g : Y ⟶ Z} : g = inv α ≫ f ↔ α ≫ g = f :=
  (asIso α).eq_inv_comp


@[simp]
theorem comp_inv_eq (α : X ⟶ Y) [IsIso α] {f : Z ⟶ Y} {g : Z ⟶ X} : f ≫ inv α = g ↔ f = g ≫ α :=
  (asIso α).comp_inv_eq


@[simp]
theorem eq_comp_inv (α : X ⟶ Y) [IsIso α] {f : Z ⟶ Y} {g : Z ⟶ X} : g = f ≫ inv α ↔ g ≫ α = f :=
  (asIso α).eq_comp_inv


theorem of_isIso_comp_left {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] [IsIso (f ≫ g)] :
    IsIso g := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso g
  -/
  rw [← id_comp g, ← inv_hom_id f, assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem of_isIso_comp_right {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso g] [IsIso (f ≫ g)] :
    IsIso f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso g
    inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso f
  -/
  rw [← comp_id f, ← hom_inv_id g, ← assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso g
    inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem of_isIso_fac_left {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} {h : X ⟶ Z} [IsIso f]
    [hh : IsIso h] (w : f ≫ g = h) : IsIso g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    hh : CategoryTheory.IsIso h
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.IsIso g
  -/
  rw [← w] at hh
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    hh : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.IsIso g
  -/
  haveI := hh
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    hh : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso g
  -/
  exact of_isIso_comp_left f g
  /-
    🎉 no goals
  -/


theorem of_isIso_fac_right {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} {h : X ⟶ Z} [IsIso g]
    [hh : IsIso h] (w : f ≫ g = h) : IsIso f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    hh : CategoryTheory.IsIso h
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.IsIso f
  -/
  rw [← w] at hh
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    hh : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.IsIso f
  -/
  haveI := hh
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    hh : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.IsIso f
  -/
  exact of_isIso_comp_right f g
  /-
    🎉 no goals
  -/


theorem eq_of_inv_eq_inv {f g : X ⟶ Y} [IsIso f] [IsIso g] (p : inv f = inv g) : f = g := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso g
    p : Eq (CategoryTheory.inv f) (CategoryTheory.inv g)
    ⊢ Eq f g
  -/
  apply (cancel_epi (inv f)).1
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : CategoryTheory.IsIso g
    p : Eq (CategoryTheory.inv f) (CategoryTheory.inv g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) f) (CategoryTh …
  -/
  rw [inv_hom_id, p, inv_hom_id]
  /-
    🎉 no goals
  -/


theorem IsIso.inv_eq_inv {f g : X ⟶ Y} [IsIso f] [IsIso g] : inv f = inv g ↔ f = g :=
  Iso.inv_eq_inv (asIso f) (asIso g)


theorem hom_comp_eq_id (g : X ⟶ Y) [IsIso g] {f : Y ⟶ X} : g ≫ f = 𝟙 X ↔ f = inv g :=
  (asIso g).hom_comp_eq_id


theorem comp_hom_eq_id (g : X ⟶ Y) [IsIso g] {f : Y ⟶ X} : f ≫ g = 𝟙 Y ↔ f = inv g :=
  (asIso g).comp_hom_eq_id


theorem inv_comp_eq_id (g : X ⟶ Y) [IsIso g] {f : X ⟶ Y} : inv g ≫ f = 𝟙 Y ↔ f = g :=
  (asIso g).inv_comp_eq_id


theorem comp_inv_eq_id (g : X ⟶ Y) [IsIso g] {f : X ⟶ Y} : f ≫ inv g = 𝟙 X ↔ f = g :=
  (asIso g).comp_inv_eq_id


theorem isIso_of_hom_comp_eq_id (g : X ⟶ Y) [IsIso g] {f : Y ⟶ X} (h : g ≫ f = 𝟙 X) : IsIso f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso g
    f : Quiver.Hom Y X
    h : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso f
  -/
  rw [(hom_comp_eq_id _).mp h]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso g
    f : Quiver.Hom Y X
    h : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso (CategoryTheory.inv g)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem isIso_of_comp_hom_eq_id (g : X ⟶ Y) [IsIso g] {f : Y ⟶ X} (h : f ≫ g = 𝟙 Y) : IsIso f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso g
    f : Quiver.Hom Y X
    h : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso f
  -/
  rw [(comp_hom_eq_id _).mp h]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso g
    f : Quiver.Hom Y X
    h : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso (CategoryTheory.inv g)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[aesop apply safe (rule_sets := [CategoryTheory])]
theorem inv_ext {f : X ≅ Y} {g : Y ⟶ X} (hom_inv_id : f.hom ≫ g = 𝟙 X) : f.inv = g :=
  ((hom_comp_eq_id f).1 hom_inv_id).symm


@[aesop apply safe (rule_sets := [CategoryTheory])]
theorem inv_ext' {f : X ≅ Y} {g : Y ⟶ X} (hom_inv_id : f.hom ≫ g = 𝟙 X) : g = f.inv :=
  (hom_comp_eq_id f).1 hom_inv_id


@[simp]
theorem cancel_iso_hom_left {X Y Z : C} (f : X ≅ Y) (g g' : Y ⟶ Z) :
    f.hom ≫ g = f.hom ≫ g' ↔ g = g' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : CategoryTheory.Iso X Y
    g g' : Quiver.Hom Y Z
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f.hom g) (CategoryTheory.Categor …
  -/
  simp only [cancel_epi]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_iso_inv_left {X Y Z : C} (f : Y ≅ X) (g g' : Y ⟶ Z) :
    f.inv ≫ g = f.inv ≫ g' ↔ g = g' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : CategoryTheory.Iso Y X
    g g' : Quiver.Hom Y Z
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f.inv g) (CategoryTheory.Categor …
  -/
  simp only [cancel_epi]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_iso_hom_right {X Y Z : C} (f f' : X ⟶ Y) (g : Y ≅ Z) :
    f ≫ g.hom = f' ≫ g.hom ↔ f = f' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f f' : Quiver.Hom X Y
    g : CategoryTheory.Iso Y Z
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f g.hom) (CategoryTheory.Categor …
  -/
  simp only [cancel_mono]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_iso_inv_right {X Y Z : C} (f f' : X ⟶ Y) (g : Z ≅ Y) :
    f ≫ g.inv = f' ≫ g.inv ↔ f = f' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f f' : Quiver.Hom X Y
    g : CategoryTheory.Iso Z Y
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f g.inv) (CategoryTheory.Categor …
  -/
  simp only [cancel_mono]
  /-
    🎉 no goals
  -/

/-
Unfortunately cancelling an isomorphism from the right of a chain of compositions is awkward.
We would need separate lemmas for each chain length (worse: for each pair of chain lengths).

We provide two more lemmas, for case of three morphisms, because this actually comes up in practice,
but then stop.
-/

@[simp]
theorem cancel_iso_hom_right_assoc {W X X' Y Z : C} (f : W ⟶ X) (g : X ⟶ Y) (f' : W ⟶ X')
    (g' : X' ⟶ Y) (h : Y ≅ Z) : f ≫ g ≫ h.hom = f' ≫ g' ≫ h.hom ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X X' Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' Y
    h : CategoryTheory.Iso Y Z
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_iso_inv_right_assoc {W X X' Y Z : C} (f : W ⟶ X) (g : X ⟶ Y) (f' : W ⟶ X')
    (g' : X' ⟶ Y) (h : Z ≅ Y) : f ≫ g ≫ h.inv = f' ≫ g' ≫ h.inv ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X X' Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' Y
    h : CategoryTheory.Iso Z Y
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_hom_inv_id (F : C ⥤ D) :
    F.map e.hom ≫ F.map e.inv = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    X Y : C
    e : CategoryTheory.Iso X Y
    F : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map e.hom) (F.map e.inv)) (Categor …
  -/
  rw [← F.map_comp, e.hom_inv_id, F.map_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_inv_hom_id (F : C ⥤ D) :
    F.map e.inv ≫ F.map e.hom = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    X Y : C
    e : CategoryTheory.Iso X Y
    F : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map e.inv) (F.map e.hom)) (Categor …
  -/
  rw [← F.map_comp, e.inv_hom_id, F.map_id]
  /-
    🎉 no goals
  -/


/-- A functor `F : C ⥤ D` sends isomorphisms `i : X ≅ Y` to isomorphisms `F.obj X ≅ F.obj Y` -/
@[simps]
def mapIso (F : C ⥤ D) {X Y : C} (i : X ≅ Y) : F.obj X ≅ F.obj Y where
  hom := F.map i.hom
  inv := F.map i.inv


@[simp]
theorem mapIso_symm (F : C ⥤ D) {X Y : C} (i : X ≅ Y) : F.mapIso i.symm = (F.mapIso i).symm :=
  rfl


@[simp]
theorem mapIso_trans (F : C ⥤ D) {X Y Z : C} (i : X ≅ Y) (j : Y ≅ Z) :
    F.mapIso (i ≪≫ j) = F.mapIso i ≪≫ F.mapIso j := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    i : CategoryTheory.Iso X Y
    j : CategoryTheory.Iso Y Z
    ⊢ Eq (F.mapIso (i.trans j)) ((F.mapIso i).trans (F.mapIso j))
  -/
  ext; apply Functor.map_comp
       /-
         🎉 no goals
       -/


@[simp]
theorem mapIso_refl (F : C ⥤ D) (X : C) : F.mapIso (Iso.refl X) = Iso.refl (F.obj X) :=
  Iso.ext <| F.map_id X


instance map_isIso (F : C ⥤ D) (f : X ⟶ Y) [IsIso f] : IsIso (F.map f) :=
  (F.mapIso (asIso f)).isIso_hom


@[simp]
theorem map_inv (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) [IsIso f] : F.map (inv f) = inv (F.map f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (F.map (CategoryTheory.inv f)) (CategoryTheory.inv (F.map f))
  -/
  apply eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map (CategoryTheory.inv  …
  -/
  simp [← F.map_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem map_hom_inv (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) [IsIso f] :
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  D : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                  F : CategoryTheory.Functor C D
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  inst✝ : CategoryTheory.IsIso f
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map (CategoryTheory.inv  …
                                                -/
    F.map f ≫ F.map (inv f) = 𝟙 (F.obj X) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[reassoc]
theorem map_inv_hom (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) [IsIso f] :
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  D : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                  F : CategoryTheory.Functor C D
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  inst✝ : CategoryTheory.IsIso f
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.inv f)) (F.map …
                                                -/
    F.map (inv f) ≫ F.map f = 𝟙 (F.obj Y) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


