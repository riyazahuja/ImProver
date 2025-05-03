/-- A differential object in a category with zero morphisms and a shift is
an object `obj` equipped with
a morphism `d : obj ⟶ obj⟦1⟧`, such that `d^2 = 0`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`
structure DifferentialObject where
  /-- The underlying object of a differential object. -/
  obj : C
  /-- The differential of a differential object. -/
  d : obj ⟶ obj⟦(1 : S)⟧
  /-- The differential `d` satisfies that `d² = 0`. -/
  d_squared : d ≫ d⟦(1 : S)⟧' = 0 := by aesop_cat


attribute [reassoc (attr := simp)] DifferentialObject.d_squared


/-- A morphism of differential objects is a morphism commuting with the differentials. -/
@[ext] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `nolint has_nonempty_instance`
structure Hom (X Y : DifferentialObject S C) where
  /-- The morphism between underlying objects of the two differentiable objects. -/
  f : X.obj ⟶ Y.obj
  comm : X.d ≫ f⟦1⟧' = f ≫ Y.d := by aesop_cat


attribute [reassoc (attr := simp)] Hom.comm


/-- The identity morphism of a differential object. -/
@[simps]
def id (X : DifferentialObject S C) : Hom X X where
  f := 𝟙 X.obj


/-- The composition of morphisms of differential objects. -/
@[simps]
def comp {X Y Z : DifferentialObject S C} (f : Hom X Y) (g : Hom Y Z) : Hom X Z where
  f := f.f ≫ g.f


instance categoryOfDifferentialObjects : Category (DifferentialObject S C) where
  Hom := Hom
  id := Hom.id
  comp f g := Hom.comp f g

-- Porting note: added

@[ext]
theorem ext {A B : DifferentialObject S C} {f g : A ⟶ B} (w : f.f = g.f := by aesop_cat) : f = g :=
  Hom.ext w


@[simp]
theorem id_f (X : DifferentialObject S C) : (𝟙 X : X ⟶ X).f = 𝟙 X.obj := rfl


@[simp]
theorem comp_f {X Y Z : DifferentialObject S C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).f = f.f ≫ g.f :=
  rfl


@[simp]
theorem eqToHom_f {X Y : DifferentialObject S C} (h : X = Y) :
    Hom.f (eqToHom h) = eqToHom (congr_arg _ h) := by
  /-
    S : Type u_1
    inst✝³ : AddMonoidWithOne S
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.HasShift C S
    X Y : CategoryTheory.DifferentialObject S C
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).f (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    S : Type u_1
    inst✝³ : AddMonoidWithOne S
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.HasShift C S
    X : CategoryTheory.DifferentialObject S C
    ⊢ Eq (CategoryTheory.eqToHom ⋯).f (CategoryTheory.eqToHom ⋯)
  -/
  rw [eqToHom_refl, eqToHom_refl]
  /-
    S : Type u_1
    inst✝³ : AddMonoidWithOne S
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.HasShift C S
    X : CategoryTheory.DifferentialObject S C
    ⊢ Eq (CategoryTheory.CategoryStruct.id X).f (CategoryTheory.CategoryStruct.id  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The forgetful functor taking a differential object to its underlying object. -/
def forget : DifferentialObject S C ⥤ C where
  obj X := X.obj
  map f := f.f


instance forget_faithful : (forget S C).Faithful where


instance {X Y : DifferentialObject S C} : Zero (X ⟶ Y) := ⟨{f := 0}⟩


@[simp]
theorem zero_f (P Q : DifferentialObject S C) : (0 : P ⟶ Q).f = 0 := rfl


instance hasZeroMorphisms : HasZeroMorphisms (DifferentialObject S C) where


/-- An isomorphism of differential objects gives an isomorphism of the underlying objects. -/
@[simps]
def isoApp {X Y : DifferentialObject S C} (f : X ≅ Y) : X.obj ≅ Y.obj where
  hom := f.hom.f
  inv := f.inv.f
                   /-
                     S : Type u_1
                     inst✝³ : AddMonoidWithOne S
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝ : CategoryTheory.HasShift C S
                     X Y : CategoryTheory.DifferentialObject S C
                     f : CategoryTheory.Iso X Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.f f.inv.f) (CategoryTheory.Cate …
                   -/
  hom_inv_id := by rw [← comp_f, Iso.hom_inv_id, id_f]
                   /-
                     🎉 no goals
                   -/
                   /-
                     S : Type u_1
                     inst✝³ : AddMonoidWithOne S
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝ : CategoryTheory.HasShift C S
                     X Y : CategoryTheory.DifferentialObject S C
                     f : CategoryTheory.Iso X Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv.f f.hom.f) (CategoryTheory.Cate …
                   -/
  inv_hom_id := by rw [← comp_f, Iso.inv_hom_id, id_f]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem isoApp_refl (X : DifferentialObject S C) : isoApp (Iso.refl X) = Iso.refl X.obj := rfl


@[simp]
theorem isoApp_symm {X Y : DifferentialObject S C} (f : X ≅ Y) : isoApp f.symm = (isoApp f).symm :=
  rfl


@[simp]
theorem isoApp_trans {X Y Z : DifferentialObject S C} (f : X ≅ Y) (g : Y ≅ Z) :
    isoApp (f ≪≫ g) = isoApp f ≪≫ isoApp g := rfl


/-- An isomorphism of differential objects can be constructed
from an isomorphism of the underlying objects that commutes with the differentials. -/
@[simps]
def mkIso {X Y : DifferentialObject S C} (f : X.obj ≅ Y.obj) (hf : X.d ≫ f.hom⟦1⟧' = f.hom ≫ Y.d) :
    X ≅ Y where
  hom := ⟨f.hom, hf⟩
  inv := ⟨f.inv, by
    rw [← Functor.mapIso_inv, Iso.comp_inv_eq, Category.assoc, Iso.eq_inv_comp, Functor.mapIso_hom,
      hf]⟩
                   /-
                     S : Type u_1
                     inst✝³ : AddMonoidWithOne S
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝ : CategoryTheory.HasShift C S
                     X Y : CategoryTheory.DifferentialObject S C
                     f : CategoryTheory.Iso X.obj Y.obj
                     hf : Eq (CategoryTheory.CategoryStruct.comp X.d ((CategoryTheory.shiftFunctor  …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := f.hom, comm := hf } { f := f.i …
                   -/
  hom_inv_id := by ext1; dsimp; exact f.hom_inv_id
                                /-
                                  🎉 no goals
                                -/
                   /-
                     S : Type u_1
                     inst✝³ : AddMonoidWithOne S
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝ : CategoryTheory.HasShift C S
                     X Y : CategoryTheory.DifferentialObject S C
                     f : CategoryTheory.Iso X.obj Y.obj
                     hf : Eq (CategoryTheory.CategoryStruct.comp X.d ((CategoryTheory.shiftFunctor  …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := f.inv, comm := ⋯ } { f := f.ho …
                   -/
  inv_hom_id := by ext1; dsimp; exact f.inv_hom_id
                                /-
                                  🎉 no goals
                                -/


/-- A functor `F : C ⥤ D` which commutes with shift functors on `C` and `D` and preserves zero
morphisms can be lifted to a functor `DifferentialObject S C ⥤ DifferentialObject S D`. -/
@[simps]
def mapDifferentialObject (F : C ⥤ D)
    (η : (shiftFunctor C (1 : S)).comp F ⟶ F.comp (shiftFunctor D (1 : S)))
    (hF : ∀ c c', F.map (0 : c ⟶ c') = 0) : DifferentialObject S C ⥤ DifferentialObject S D where
  obj X :=
    { obj := F.obj X.obj
      d := F.map X.d ≫ η.app X.obj
      d_squared := by
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Functor.map_comp, ← Functor.comp_map F (shiftFunctor D (1 : S))]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_lhs 2 3 => rw [← η.naturality X.d]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map X.d) (CategoryTheory.CategoryS …
        -/
        rw [Functor.comp_map]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map X.d) (CategoryTheory.CategoryS …
        -/
        slice_lhs 1 2 => rw [← F.map_comp, X.d_squared, hF]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp 0 …
        -/
        rw [zero_comp, zero_comp] }
        /-
          🎉 no goals
        -/
  map f :=
    { f := F.map f.f
      comm := by
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { obj := F.obj X.obj, d := …
        -/
        dsimp
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_lhs 2 3 => rw [← Functor.comp_map F (shiftFunctor D (1 : S)), ← η.naturality f.f]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map X✝.d) (CategoryTheory.Category …
        -/
        slice_lhs 1 2 => rw [Functor.comp_map, ← F.map_comp, f.comm, F.map_comp]
        /-
          S : Type u_1
          inst✝⁶ : AddMonoidWithOne S
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝³ : CategoryTheory.HasShift C S
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          inst✝ : CategoryTheory.HasShift D S
          F : CategoryTheory.Functor C D
          η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
          hF : ∀ (c c' : C), Eq (F.map 0) 0
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Category.assoc] }
        /-
          🎉 no goals
        -/
               /-
                 S : Type u_1
                 inst✝⁶ : AddMonoidWithOne S
                 C : Type u
                 inst✝⁵ : CategoryTheory.Category.{v, u} C
                 inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                 inst✝³ : CategoryTheory.HasShift C S
                 D : Type u'
                 inst✝² : CategoryTheory.Category.{v', u'} D
                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                 inst✝ : CategoryTheory.HasShift D S
                 F : CategoryTheory.Functor C D
                 η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
                 hF : ∀ (c c' : C), Eq (F.map 0) 0
                 ⊢ ∀ (X : CategoryTheory.DifferentialObject S C), Eq ({ obj := fun X => { obj : …
               -/
  map_id := by intros; ext; simp
                            /-
                              🎉 no goals
                            -/
                 /-
                   S : Type u_1
                   inst✝⁶ : AddMonoidWithOne S
                   C : Type u
                   inst✝⁵ : CategoryTheory.Category.{v, u} C
                   inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                   inst✝³ : CategoryTheory.HasShift C S
                   D : Type u'
                   inst✝² : CategoryTheory.Category.{v', u'} D
                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                   inst✝ : CategoryTheory.HasShift D S
                   F : CategoryTheory.Functor C D
                   η : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).comp F) (F.comp (CategoryThe …
                   hF : ∀ (c c' : C), Eq (F.map 0) 0
                   ⊢ ∀ {X Y Z : CategoryTheory.DifferentialObject S C} (f : Quiver.Hom X Y) (g :  …
                 -/
  map_comp := by intros; ext; simp
                              /-
                                🎉 no goals
                              -/


instance hasZeroObject : HasZeroObject (DifferentialObject S C) where
  zero := ⟨{ obj := 0, d := 0 },
                                                        /-
                                                          S : Type u_1
                                                          inst✝⁵ : AddMonoidWithOne S
                                                          C : Type u
                                                          inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                          inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                          inst✝¹ : CategoryTheory.HasShift C S
                                                          inst✝ : (CategoryTheory.shiftFunctor C 1).PreservesZeroMorphisms
                                                          X : CategoryTheory.DifferentialObject S C
                                                          f : Quiver.Hom { obj := 0, d := 0, d_squared := ⋯ } X
                                                          ⊢ Eq f Inhabited.default
                                                        -/
    { unique_to := fun X => ⟨⟨⟨{ f := 0 }⟩, fun f => by ext⟩⟩,
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                          /-
                                                            S : Type u_1
                                                            inst✝⁵ : AddMonoidWithOne S
                                                            C : Type u
                                                            inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                            inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                            inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                            inst✝¹ : CategoryTheory.HasShift C S
                                                            inst✝ : (CategoryTheory.shiftFunctor C 1).PreservesZeroMorphisms
                                                            X : CategoryTheory.DifferentialObject S C
                                                            f : Quiver.Hom X { obj := 0, d := 0, d_squared := ⋯ }
                                                            ⊢ Eq f Inhabited.default
                                                          -/
      unique_from := fun X => ⟨⟨⟨{ f := 0 }⟩, fun f => by ext⟩⟩ }⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


instance concreteCategoryOfDifferentialObjects : ConcreteCategory (DifferentialObject S C) where
  forget := forget S C ⋙ CategoryTheory.forget C


instance : HasForget₂ (DifferentialObject S C) C where
  forget₂ := forget S C


/-- The shift functor on `DifferentialObject S C`. -/
@[simps]
def shiftFunctor (n : S) : DifferentialObject S C ⥤ DifferentialObject S C where
  obj X :=
    { obj := X.obj⟦n⟧
      d := X.d⟦n⟧' ≫ (shiftComm _ _ _).hom
      d_squared := by
        rw [Functor.map_comp, Category.assoc, shiftComm_hom_comp_assoc, ← Functor.map_comp_assoc,
          X.d_squared, Functor.map_zero, zero_comp] }
  map f :=
    { f := f.f⟦n⟧'
      comm := by
        /-
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { obj := (CategoryTheory.s …
        -/
        dsimp
        erw [Category.assoc, shiftComm_hom_comp, ← Functor.map_comp_assoc, f.comm,
          Functor.map_comp_assoc]
        /-
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X✝ Y✝ : CategoryTheory.DifferentialObject S C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C n).ma …
        -/
        rfl }
        /-
          🎉 no goals
        -/
                 /-
                   S : Type u_1
                   inst✝³ : AddCommGroupWithOne S
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                   inst✝ : CategoryTheory.HasShift C S
                   n : S
                   X : CategoryTheory.DifferentialObject S C
                   ⊢ Eq ({ obj := fun X => { obj := (CategoryTheory.shiftFunctor C n).obj X.obj,  …
                 -/
  map_id X := by ext1; dsimp; rw [Functor.map_id]
                              /-
                                🎉 no goals
                              -/
                     /-
                       S : Type u_1
                       inst✝³ : AddCommGroupWithOne S
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                       inst✝ : CategoryTheory.HasShift C S
                       n : S
                       X✝ Y✝ Z✝ : CategoryTheory.DifferentialObject S C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => { obj := (CategoryTheory.shiftFunctor C n).obj X.obj,  …
                     -/
  map_comp f g := by ext1; dsimp; rw [Functor.map_comp]
                                  /-
                                    🎉 no goals
                                  -/


/-- The shift functor on `DifferentialObject S C` is additive. -/
@[simps!]
nonrec def shiftFunctorAdd (m n : S) :
    shiftFunctor C (m + n) ≅ shiftFunctor C m ⋙ shiftFunctor C n := by
  /-
    S : Type u_1
    inst✝³ : AddCommGroupWithOne S
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.HasShift C S
    m n : S
    ⊢ CategoryTheory.Iso (CategoryTheory.DifferentialObject.shiftFunctor C (HAdd.h …
  -/
  refine NatIso.ofComponents (fun X => mkIso (shiftAdd X.obj _ _) ?_) (fun f => ?_)
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
    -/
  · dsimp
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← cancel_epi ((shiftFunctorAdd C m n).inv.app X.obj)]
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C m  …
    -/
    simp only [Category.assoc, Iso.inv_hom_id_app_assoc]
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C m  …
    -/
    rw [← NatTrans.naturality_assoc]
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.shiftFunctor C m).c …
    -/
    dsimp
    simp only [Functor.map_comp, Category.assoc,
      shiftFunctorComm_hom_app_comp_shift_shiftFunctorAdd_hom_app 1 m n X.obj,
      Iso.inv_hom_id_app_assoc]
    /-
      case refine_2
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      m n : S
      X✝ Y✝ : CategoryTheory.DifferentialObject S C
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
    -/
  · ext; dsimp; exact NatTrans.naturality _ _
                /-
                  🎉 no goals
                -/


/-- The shift by zero is naturally isomorphic to the identity. -/
@[simps!]
def shiftZero : shiftFunctor C (0 : S) ≅ 𝟭 (DifferentialObject S C) := by
  /-
    S : Type u_1
    inst✝³ : AddCommGroupWithOne S
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.HasShift C S
    ⊢ CategoryTheory.Iso (CategoryTheory.DifferentialObject.shiftFunctor C 0) (Cat …
  -/
  refine NatIso.ofComponents (fun X => mkIso ((shiftFunctorZero C S).app X.obj) ?_) (fun f => ?_)
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
    -/
  · erw [← NatTrans.naturality]
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
    -/
    dsimp
    /-
      case refine_1
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      X : CategoryTheory.DifferentialObject S C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [shiftFunctorZero_hom_app_shift, Category.assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      S : Type u_1
      inst✝³ : AddCommGroupWithOne S
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.HasShift C S
      X✝ Y✝ : CategoryTheory.DifferentialObject S C
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/


instance : HasShift (DifferentialObject S C) S :=
  hasShiftMk _ _
    { F := shiftFunctor C
      zero := shiftZero C
      add := shiftFunctorAdd C
      assoc_hom_app := fun m₁ m₂ m₃ X => by
        /-
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          m₁ m₂ m₃ : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.DifferentialObject.s …
        -/
        ext1
        /-
          case w
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          m₁ m₂ m₃ : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Different …
        -/
        convert shiftFunctorAdd_assoc_hom_app m₁ m₂ m₃ X.obj
        /-
          case h.e'_3.h
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          m₁ m₂ m₃ : S
          X : CategoryTheory.DifferentialObject S C
          e_1✝ : Eq (Quiver.Hom ((CategoryTheory.DifferentialObject.shiftFunctor C (HAdd …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        dsimp [shiftFunctorAdd']
        /-
          case h.e'_3.h
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          m₁ m₂ m₃ : S
          X : CategoryTheory.DifferentialObject S C
          e_1✝ : Eq (Quiver.Hom ((CategoryTheory.DifferentialObject.shiftFunctor C (HAdd …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯).f (Categor …
        -/
        simp
        /-
          🎉 no goals
        -/
      zero_add_hom_app := fun n X => by
        /-
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq ((CategoryTheory.DifferentialObject.shiftFunctorAdd C 0 n).hom.app X) (Ca …
        -/
        ext1
        /-
          case w
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ autoParam (Eq ((CategoryTheory.DifferentialObject.shiftFunctorAdd C 0 n).hom …
        -/
        convert shiftFunctorAdd_zero_add_hom_app n X.obj
        /-
          case h.e'_3.h
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          e_1✝ : Eq (Quiver.Hom ((CategoryTheory.DifferentialObject.shiftFunctor C (HAdd …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((Category …
        -/
        simp
        /-
          🎉 no goals
        -/
      add_zero_hom_app := fun n X => by
        /-
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ Eq ((CategoryTheory.DifferentialObject.shiftFunctorAdd C n 0).hom.app X) (Ca …
        -/
        ext1
        /-
          case w
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          ⊢ autoParam (Eq ((CategoryTheory.DifferentialObject.shiftFunctorAdd C n 0).hom …
        -/
        convert shiftFunctorAdd_add_zero_hom_app n X.obj
        /-
          case h.e'_3.h
          S : Type u_1
          inst✝³ : AddCommGroupWithOne S
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.HasShift C S
          n : S
          X : CategoryTheory.DifferentialObject S C
          e_1✝ : Eq (Quiver.Hom ((CategoryTheory.DifferentialObject.shiftFunctor C (HAdd …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((Category …
        -/
        simp }
        /-
          🎉 no goals
        -/


