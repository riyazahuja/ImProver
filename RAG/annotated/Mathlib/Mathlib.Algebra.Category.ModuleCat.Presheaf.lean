variable (R) in
/-- A presheaf of modules over `R : Cᵒᵖ ⥤ RingCat` consists of family of
objects `obj X : ModuleCat (R.obj X)` for all `X : Cᵒᵖ` together with
functorial maps `obj X ⟶ (ModuleCat.restrictScalars (R.map f)).obj (obj Y)`
for all `f : X ⟶ Y` in `Cᵒᵖ`. -/
structure PresheafOfModules where
  /-- a family of modules over `R.obj X` for all `X` -/
  obj (X : Cᵒᵖ) : ModuleCat.{v} (R.obj X)
  /-- the restriction maps of a presheaf of modules -/
  map {X Y : Cᵒᵖ} (f : X ⟶ Y) : obj X ⟶ (ModuleCat.restrictScalars (R.map f).hom).obj (obj Y)
  map_id (X : Cᵒᵖ) :
    map (𝟙 X) =
      (ModuleCat.restrictScalarsId' _ (congrArg RingCat.Hom.hom (R.map_id X))).inv.app _ := by
        aesop_cat
  map_comp {X Y Z : Cᵒᵖ} (f : X ⟶ Y) (g : Y ⟶ Z) :
    map (f ≫ g) = map f ≫ (ModuleCat.restrictScalars _).map (map g) ≫
      (ModuleCat.restrictScalarsComp' _ _ _
        (congrArg RingCat.Hom.hom <| R.map_comp f g)).inv.app _ := by aesop_cat


attribute [reassoc] map_comp


lemma map_smul {X Y : Cᵒᵖ} (f : X ⟶ Y) (r : R.obj X) (m : M.obj X) :
                                                  /-
                                                    C : Type u₁
                                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                    R : CategoryTheory.Functor (Opposite C) RingCat
                                                    M : PresheafOfModules R
                                                    X Y : Opposite C
                                                    f : Quiver.Hom X Y
                                                    r : ↑(R.obj X)
                                                    m : ↑(M.obj X)
                                                    ⊢ Eq ((M.map f).hom (HSMul.hSMul r m)) (HSMul.hSMul ((R.map f).hom r) ((M.map  …
                                                  -/
    M.map f (r • m) = R.map f r • M.map f m := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma congr_map_apply {X Y : Cᵒᵖ} {f g : X ⟶ Y} (h : f = g) (m : M.obj X) :
                                /-
                                  C : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                  R : CategoryTheory.Functor (Opposite C) RingCat
                                  M : PresheafOfModules R
                                  X Y : Opposite C
                                  f g : Quiver.Hom X Y
                                  h : Eq f g
                                  m : ↑(M.obj X)
                                  ⊢ Eq ((M.map f).hom m) ((M.map g).hom m)
                                -/
    M.map f m = M.map g m := by rw [h]
                                /-
                                  🎉 no goals
                                -/


/-- A morphism of presheaves of modules consists of a family of linear maps which
satisfy the naturality condition. -/
@[ext]
structure Hom where
  /-- a family of linear maps `M₁.obj X ⟶ M₂.obj X` for all `X`. -/
  app (X : Cᵒᵖ) : M₁.obj X ⟶ M₂.obj X
  naturality {X Y : Cᵒᵖ} (f : X ⟶ Y) :
      M₁.map f ≫ (ModuleCat.restrictScalars (R.map f).hom).map (app Y) =
        app X ≫ M₂.map f := by aesop_cat


attribute [reassoc (attr := simp)] Hom.naturality


instance : Category (PresheafOfModules.{v} R) where
  Hom := Hom
  id _ := { app := fun _ ↦ 𝟙 _ }
  comp f g := { app := fun _ ↦ f.app _ ≫ g.app _ }


@[ext]
lemma hom_ext {f g : M₁ ⟶ M₂} (h : ∀ (X : Cᵒᵖ), f.app X = g.app X) :
                         /-
                           C : Type u₁
                           inst✝ : CategoryTheory.Category.{v₁, u₁} C
                           R : CategoryTheory.Functor (Opposite C) RingCat
                           M₁ M₂ : PresheafOfModules R
                           f g : Quiver.Hom M₁ M₂
                           h : ∀ (X : Opposite C), Eq (f.app X) (g.app X)
                           ⊢ Eq f.app g.app
                         -/
    f = g := Hom.ext (by ext1; apply h)
                               /-
                                 🎉 no goals
                               -/


@[simp]
lemma id_app (M : PresheafOfModules R) (X : Cᵒᵖ) : Hom.app (𝟙 M) X = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    X : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.id M).app X) (CategoryTheory.CategoryStru …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_app {M₁ M₂ M₃ : PresheafOfModules R} (f : M₁ ⟶ M₂) (g : M₂ ⟶ M₃) (X : Cᵒᵖ) :
    (f ≫ g).app X = f.app X ≫ g.app X := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ M₃ : PresheafOfModules R
    f : Quiver.Hom M₁ M₂
    g : Quiver.Hom M₂ M₃
    X : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).app X) (CategoryTheory.Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma naturality_apply (f : M₁ ⟶ M₂) {X Y : Cᵒᵖ} (g : X ⟶ Y) (x : M₁.obj X) :
    Hom.app f Y (M₁.map g x) = M₂.map g (Hom.app f X x) :=
  congr_fun ((forget _).congr_map (Hom.naturality f g)) x


/-- Constructor for isomorphisms in the category of presheaves of modules. -/
@[simps!]
def isoMk (app : ∀ (X : Cᵒᵖ), M₁.obj X ≅ M₂.obj X)
    (naturality : ∀ ⦃X Y : Cᵒᵖ⦄ (f : X ⟶ Y),
      M₁.map f ≫ (ModuleCat.restrictScalars (R.map f).hom).map (app Y).hom =
        (app X).hom ≫ M₂.map f := by aesop_cat) : M₁ ≅ M₂ where
  hom := { app := fun X ↦ (app X).hom }
  inv :=
    { app := fun X ↦ (app X).inv
      naturality := fun {X Y} f ↦ by
        rw [← cancel_epi (app X).hom, ← reassoc_of% (naturality f), Iso.map_hom_inv_id,
          Category.comp_id, Iso.hom_inv_id_assoc]}


/-- The underlying presheaf of abelian groups of a presheaf of modules. -/
def presheaf : Cᵒᵖ ⥤ Ab where
  obj X := (forget₂ _ _).obj (M.obj X)
                                          /-
                                            C : Type u₁
                                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                            R : CategoryTheory.Functor (Opposite C) RingCat
                                            M M₁ M₂ : PresheafOfModules R
                                            X✝ Y✝ : Opposite C
                                            f : Quiver.Hom X✝ Y✝
                                            ⊢ ∀ (a b : ↑((fun X => (CategoryTheory.forget₂ (ModuleCat ↑(R.obj X)) Ab).obj  …
                                          -/
  map f := AddMonoidHom.mk' (M.map f) (by simp)
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
lemma presheaf_obj_coe (X : Cᵒᵖ) :
    (M.presheaf.obj X : Type _) = M.obj X := rfl


@[simp]
lemma presheaf_map_apply_coe {X Y : Cᵒᵖ} (f : X ⟶ Y) (x : M.obj X) :
    DFunLike.coe (α := M.obj X) (β := fun _ ↦ M.obj Y) (M.presheaf.map f) x = M.map f x := rfl


instance (M : PresheafOfModules R) (X : Cᵒᵖ) :
    Module (R.obj X) (M.presheaf.obj X) :=
  inferInstanceAs (Module (R.obj X) (M.obj X))


variable (R) in
/-- The forgetful functor `PresheafOfModules R ⥤ Cᵒᵖ ⥤ Ab`. -/
def toPresheaf : PresheafOfModules.{v} R ⥤ Cᵒᵖ ⥤ Ab where
  obj M := M.presheaf
  map f :=
                                                        /-
                                                          C : Type u₁
                                                          inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                          R : CategoryTheory.Functor (Opposite C) RingCat
                                                          M M₁ M₂ X✝ Y✝ : PresheafOfModules R
                                                          f : Quiver.Hom X✝ Y✝
                                                          X : Opposite C
                                                          ⊢ ∀ (a b : ↑(((fun M => M.presheaf) X✝).obj X)), Eq ((f.app X).hom (HAdd.hAdd  …
                                                        -/
    { app := fun X ↦ AddMonoidHom.mk' (Hom.app f X) (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/
                                   /-
                                     C : Type u₁
                                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                     R : CategoryTheory.Functor (Opposite C) RingCat
                                     M M₁ M₂ X✝ Y✝ : PresheafOfModules R
                                     f : Quiver.Hom X✝ Y✝
                                     X Y : Opposite C
                                     g : Quiver.Hom X Y
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun M => M.presheaf) X✝).map g) (( …
                                   -/
      naturality := fun X Y g ↦ by ext x; exact naturality_apply f g x }
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
lemma toPresheaf_obj_coe (X : Cᵒᵖ) :
    (((toPresheaf R).obj M).obj X : Type _) = M.obj X := rfl


@[simp]
lemma toPresheaf_map_app_apply (f : M₁ ⟶ M₂) (X : Cᵒᵖ) (x : M₁.obj X) :
    DFunLike.coe (α := M₁.obj X) (β := fun _ ↦ M₂.obj X)
      (((toPresheaf R).map f).app X) x = f.app X x := rfl


instance : (toPresheaf R).Faithful where
  map_injective {_ _ f g} h := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M M₁ M₂ : PresheafOfModules R
      x✝¹ x✝ : PresheafOfModules R
      f g : Quiver.Hom x✝¹ x✝
      h : Eq ((PresheafOfModules.toPresheaf R).map f) ((PresheafOfModules.toPresheaf …
      ⊢ Eq f g
    -/
    ext X x
    /-
      case h.hf.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M M₁ M₂ : PresheafOfModules R
      x✝¹ x✝ : PresheafOfModules R
      f g : Quiver.Hom x✝¹ x✝
      h : Eq ((PresheafOfModules.toPresheaf R).map f) ((PresheafOfModules.toPresheaf …
      X : Opposite C
      x : ↑(x✝¹.obj X)
      ⊢ Eq ((f.app X).hom x) ((g.app X).hom x)
    -/
    exact congr_fun (((evaluation _ _).obj X ⋙ forget _).congr_map h) x
    /-
      🎉 no goals
    -/


/-- The object in `PresheafOfModules R` that is obtained from `M : Cᵒᵖ ⥤ Ab.{v}` such
that for all `X : Cᵒᵖ`, `M.obj X` is a `R.obj X` module, in such a way that the
restriction maps are semilinear. (This constructor should be used only in cases
when the preferred constructor `PresheafOfModules.mk` is not as convenient as this one.) -/
@[simps]
def ofPresheaf : PresheafOfModules.{v} R where
  obj X := ModuleCat.of _ (M.obj X)
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  map {X Y} f := ModuleCat.ofHom
      (Y := (ModuleCat.restrictScalars (R.map f).hom).obj (ModuleCat.of _ (M.obj Y)))
    { toFun := fun x ↦ M.map f x
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                       R : CategoryTheory.Functor (Opposite C) RingCat
                       M✝ M₁ M₂ : PresheafOfModules R
                       M : CategoryTheory.Functor (Opposite C) Ab
                       inst✝ : (X : Opposite C) → Module ↑(R.obj X) ↑(M.obj X)
                       map_smul : ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y) (r : ↑(R.obj X)) (m : ↑(M …
                       X Y : Opposite C
                       f : Quiver.Hom X Y
                       ⊢ ∀ (x y : ↑(M.obj X)), Eq ((fun x => (M.map f) x) (HAdd.hAdd x y)) (HAdd.hAdd …
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
      map_smul' := fun r m ↦ map_smul f r m }


@[simp]
lemma ofPresheaf_presheaf : (ofPresheaf M map_smul).presheaf = M := rfl


/-- The morphism of presheaves of modules `M₁ ⟶ M₂` given by a morphism
of abelian presheaves `M₁.presheaf ⟶ M₂.presheaf`
which satisfy a suitable linearity condition. -/
@[simps]
def homMk (φ : M₁.presheaf ⟶ M₂.presheaf)
    (hφ : ∀ (X : Cᵒᵖ) (r : R.obj X) (m : M₁.obj X), φ.app X (r • m) = r • φ.app X m) :
    M₁ ⟶ M₂ where
  app X := ModuleCat.ofHom
    { toFun := φ.app X
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       R : CategoryTheory.Functor (Opposite C) RingCat
                       M M₁ M₂ : PresheafOfModules R
                       φ : Quiver.Hom M₁.presheaf M₂.presheaf
                       hφ : ∀ (X : Opposite C) (r : ↑(R.obj X)) (m : ↑(M₁.obj X)), Eq ((φ.app X) (HSM …
                       X : Opposite C
                       ⊢ ∀ (x y : ↑(M₁.1 X)), Eq ((φ.app X) (HAdd.hAdd x y)) (HAdd.hAdd ((φ.app X) x) …
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
      map_smul' := hφ X }
  naturality := fun f ↦ by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M M₁ M₂ : PresheafOfModules R
      φ : Quiver.Hom M₁.presheaf M₂.presheaf
      hφ : ∀ (X : Opposite C) (r : ↑(R.obj X)) (m : ↑(M₁.obj X)), Eq ((φ.app X) (HSM …
      X✝ Y✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.map f) ((ModuleCat.restrictScalar …
    -/
    ext x
    /-
      case hf.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M M₁ M₂ : PresheafOfModules R
      φ : Quiver.Hom M₁.presheaf M₂.presheaf
      hφ : ∀ (X : Opposite C) (r : ↑(R.obj X)) (m : ↑(M₁.obj X)), Eq ((φ.app X) (HSM …
      X✝ Y✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      x : ↑(M₁.obj X✝)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M₁.map f) ((ModuleCat.restrictScala …
    -/
    exact congr_fun ((forget _).congr_map (φ.naturality f)) x
    /-
      🎉 no goals
    -/


instance : Zero (M₁ ⟶ M₂) where
  zero := { app := fun _ ↦ 0 }


variable (M₁ M₂) in
@[simp] lemma zero_app (X : Cᵒᵖ) : (0 : M₁ ⟶ M₂).app X = 0 := rfl


instance : Neg (M₁ ⟶ M₂) where
  neg f :=
    { app := fun X ↦ -f.app X
      naturality := fun {X Y} h ↦ by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScalar …
        -/
        ext x
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          x : ↑(M₁.obj X)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScala …
        -/
        simp [← naturality_apply] }
        /-
          🎉 no goals
        -/


instance : Add (M₁ ⟶ M₂) where
  add f g :=
    { app := fun X ↦ f.app X + g.app X
      naturality := fun {X Y} h ↦ by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f g : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScalar …
        -/
        ext x
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f g : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          x : ↑(M₁.obj X)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScala …
        -/
        simp [← naturality_apply] }
        /-
          🎉 no goals
        -/


instance : Sub (M₁ ⟶ M₂) where
  sub f g :=
    { app := fun X ↦ f.app X - g.app X
      naturality := fun {X Y} h ↦ by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f g : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScalar …
        -/
        ext x
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M M₁ M₂ : PresheafOfModules R
          f g : Quiver.Hom M₁ M₂
          X Y : Opposite C
          h : Quiver.Hom X Y
          x : ↑(M₁.obj X)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M₁.map h) ((ModuleCat.restrictScala …
        -/
        simp [← naturality_apply] }
        /-
          🎉 no goals
        -/


@[simp] lemma neg_app (f : M₁ ⟶ M₂) (X : Cᵒᵖ) : (-f).app X = -f.app X := rfl

@[simp] lemma add_app (f g : M₁ ⟶ M₂) (X : Cᵒᵖ) : (f + g).app X = f.app X + g.app X := rfl

@[simp] lemma sub_app (f g : M₁ ⟶ M₂) (X : Cᵒᵖ) : (f - g).app X = f.app X - g.app X := rfl


instance : AddCommGroup (M₁ ⟶ M₂) where
                  /-
                    C : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                    R : CategoryTheory.Functor (Opposite C) RingCat
                    M M₁ M₂ : PresheafOfModules R
                    ⊢ ∀ (a b c : Quiver.Hom M₁ M₂), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a  …
                  -/
  add_assoc := by intros; ext1; simp only [add_app, add_assoc]
                                /-
                                  🎉 no goals
                                -/
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   R : CategoryTheory.Functor (Opposite C) RingCat
                   M M₁ M₂ : PresheafOfModules R
                   ⊢ ∀ (a : Quiver.Hom M₁ M₂), Eq (HAdd.hAdd 0 a) a
                 -/
  zero_add := by intros; ext1; simp only [add_app, zero_app, zero_add]
                               /-
                                 🎉 no goals
                               -/
                       /-
                         C : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                         R : CategoryTheory.Functor (Opposite C) RingCat
                         M M₁ M₂ : PresheafOfModules R
                         ⊢ ∀ (a : Quiver.Hom M₁ M₂), Eq (HAdd.hAdd (Neg.neg a) a) 0
                       -/
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   R : CategoryTheory.Functor (Opposite C) RingCat
                   M M₁ M₂ : PresheafOfModules R
                   ⊢ ∀ (a : Quiver.Hom M₁ M₂), Eq (HAdd.hAdd a 0) a
                 -/
  neg_add_cancel := by intros; ext1; simp only [add_app, neg_app, neg_add_cancel, zero_app]
                               /-
                                 🎉 no goals
                               -/
                                     /-
                                       🎉 no goals
                                     -/
                       /-
                         C : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                         R : CategoryTheory.Functor (Opposite C) RingCat
                         M M₁ M₂ : PresheafOfModules R
                         ⊢ ∀ (a b : Quiver.Hom M₁ M₂), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
                       -/
  add_zero := by intros; ext1; simp only [add_app, zero_app, add_zero]
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   R : CategoryTheory.Functor (Opposite C) RingCat
                   M M₁ M₂ : PresheafOfModules R
                   ⊢ ∀ (a b : Quiver.Hom M₁ M₂), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                 -/
  add_comm := by intros; ext1; simp only [add_app]; apply add_comm
                                                    /-
                                                      🎉 no goals
                                                    -/
  sub_eq_add_neg := by intros; ext1; simp only [add_app, sub_app, neg_app, sub_eq_add_neg]
  nsmul := nsmulRec
  zsmul := zsmulRec


instance : Preadditive (PresheafOfModules R) where


instance : (toPresheaf R).Additive where


lemma zsmul_app (n : ℤ) (f : M₁ ⟶ M₂) (X : Cᵒᵖ) : (n • f).app X = n • f.app X := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    n : Int
    f : Quiver.Hom M₁ M₂
    X : Opposite C
    ⊢ Eq ((HSMul.hSMul n f).app X) (HSMul.hSMul n (f.app X))
  -/
  ext x
  /-
    case hf.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    n : Int
    f : Quiver.Hom M₁ M₂
    X : Opposite C
    x : ↑(M₁.obj X)
    ⊢ Eq (((HSMul.hSMul n f).app X).hom x) ((HSMul.hSMul n (f.app X)).hom x)
  -/
  change (toPresheaf R ⋙ (evaluation _ _).obj X).map (n • f) x = _
  /-
    case hf.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    n : Int
    f : Quiver.Hom M₁ M₂
    X : Opposite C
    x : ↑(M₁.obj X)
    ⊢ Eq ((((PresheafOfModules.toPresheaf R).comp ((CategoryTheory.evaluation (Opp …
  -/
  rw [Functor.map_zsmul]
  /-
    case hf.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    n : Int
    f : Quiver.Hom M₁ M₂
    X : Opposite C
    x : ↑(M₁.obj X)
    ⊢ Eq ((HSMul.hSMul n (((PresheafOfModules.toPresheaf R).comp ((CategoryTheory. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Evaluation on an object `X` gives a functor
`PresheafOfModules R ⥤ ModuleCat (R.obj X)`. -/
@[simps]
def evaluation (X : Cᵒᵖ) : PresheafOfModules.{v} R ⥤ ModuleCat (R.obj X) where
  obj M := M.obj X
  map f := f.app X


instance (X : Cᵒᵖ) : (evaluation.{v} R X).Additive where


/-- The restriction natural transformation on presheaves of modules, considered as linear maps
to restriction of scalars. -/
@[simps]
noncomputable def restriction {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    evaluation R X ⟶ evaluation R Y ⋙ ModuleCat.restrictScalars (R.map f).hom where
  app M := M.map f


/-- The obvious free presheaf of modules of rank `1`. -/
def unit : PresheafOfModules R where
  obj X := ModuleCat.of _ (R.obj X)
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  map {X Y} f := ModuleCat.ofHom
      (Y := (ModuleCat.restrictScalars (R.map f).hom).obj (ModuleCat.of (R.obj Y) (R.obj Y)))
    { toFun := fun x ↦ R.map f x
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       R : CategoryTheory.Functor (Opposite C) RingCat
                       M M₁ M₂ : PresheafOfModules R
                       X Y : Opposite C
                       f : Quiver.Hom X Y
                       ⊢ ∀ (x y : ↑(R.obj X)), Eq ((fun x => (R.map f).hom x) (HAdd.hAdd x y)) (HAdd. …
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        C : Type u₁
                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                        R : CategoryTheory.Functor (Opposite C) RingCat
                        M M₁ M₂ : PresheafOfModules R
                        X Y : Opposite C
                        f : Quiver.Hom X Y
                        ⊢ ∀ (m x : ↑(R.obj X)), Eq ({ toFun := fun x => (R.map f).hom x, map_add' := ⋯ …
                      -/
      map_smul' := by aesop_cat }
                      /-
                        🎉 no goals
                      -/


lemma unit_map_one {X Y : Cᵒᵖ} (f : X ⟶ Y) : (unit R).map f (1 : R.obj X) = (1 : R.obj Y) :=
  (R.map f).hom.map_one


/-- The type of sections of a presheaf of modules. -/
def sections (M : PresheafOfModules.{v} R) : Type _ := (M.presheaf ⋙ forget _).sections


/-- Given a presheaf of modules `M`, `s : M.sections` and `X : Cᵒᵖ`, this is the induced
element in `M.obj X`. -/
abbrev sections.eval {M : PresheafOfModules.{v} R} (s : M.sections) (X : Cᵒᵖ) : M.obj X := s.1 X


@[simp]
lemma sections_property {M : PresheafOfModules.{v} R} (s : M.sections)
    {X Y : Cᵒᵖ} (f : X ⟶ Y) : M.map f (s.1 X) = s.1 Y := s.2 f


/-- Constructor for sections of a presheaf of modules. -/
@[simps]
def sectionsMk {M : PresheafOfModules.{v} R} (s : ∀ X, M.obj X)
    (hs : ∀ ⦃X Y : Cᵒᵖ⦄ (f : X ⟶ Y), M.map f (s X) = s Y) : M.sections where
  val := s
  property f := hs f


@[ext]
lemma sections_ext {M : PresheafOfModules.{v} R} (s t : M.sections)
    (h : ∀ (X : Cᵒᵖ), s.val X = t.val X) : s = t :=
                  /-
                    C : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                    R : CategoryTheory.Functor (Opposite C) RingCat
                    M : PresheafOfModules R
                    s t : M.sections
                    h : ∀ (X : Opposite C), Eq (↑s X) (↑t X)
                    ⊢ Eq ↑s ↑t
                  -/
  Subtype.ext (by ext; apply h)
                       /-
                         🎉 no goals
                       -/


/-- The map `M.sections → N.sections` induced by a morphisms `M ⟶ N` of presheaves of modules. -/
@[simps!]
def sectionsMap {M N : PresheafOfModules.{v} R} (f : M ⟶ N) (s : M.sections) : N.sections :=
  N.sectionsMk (fun X ↦ f.app X (s.1 _))
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      R : CategoryTheory.Functor (Opposite C) RingCat
                      M✝ M₁ M₂ M N : PresheafOfModules R
                      f : Quiver.Hom M N
                      s : M.sections
                      X Y : Opposite C
                      g : Quiver.Hom X Y
                      ⊢ Eq ((N.map g).hom ((fun X => (f.app X).hom (↑s X)) X)) ((fun X => (f.app X). …
                    -/
    (fun X Y g ↦ by rw [← naturality_apply, sections_property])
                    /-
                      🎉 no goals
                    -/


@[simp]
lemma sectionsMap_comp {M N P : PresheafOfModules.{v} R} (f : M ⟶ N) (g : N ⟶ P) (s : M.sections) :
    sectionsMap (f ≫ g) s = sectionsMap g (sectionsMap f s) := rfl


@[simp]
lemma sectionsMap_id {M : PresheafOfModules.{v} R} (s : M.sections) :
    sectionsMap (𝟙 M) s = s := rfl


/-- The bijection `(unit R ⟶ M) ≃ M.sections` for `M : PresheafOfModules R`. -/
@[simps! apply_coe]
def unitHomEquiv (M : PresheafOfModules R) :
    (unit R ⟶ M) ≃ M.sections where
  toFun f := sectionsMk (fun X ↦ Hom.app f X (1 : R.obj X))
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M✝ M₁ M₂ : PresheafOfModules R
          M : PresheafOfModules R
          f : Quiver.Hom (PresheafOfModules.unit R) M
          ⊢ ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y), Eq ((M.map f_1).hom ((fun X =>  …
        -/
    (by intros; rw [← naturality_apply, unit_map_one])
                /-
                  🎉 no goals
                -/
  invFun s :=
    { app := fun X ↦ ModuleCat.ofHom
        ((LinearMap.ringLmapEquivSelf (R.obj X) ℤ (M.obj X)).symm (s.val X))
      naturality := fun {X Y} f ↦ by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M✝ M₁ M₂ : PresheafOfModules R
          M : PresheafOfModules R
          s : M.sections
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.unit R).map f) (( …
        -/
        ext
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M✝ M₁ M₂ : PresheafOfModules R
          M : PresheafOfModules R
          s : M.sections
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((PresheafOfModules.unit R).map f) ( …
        -/
        dsimp
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M✝ M₁ M₂ : PresheafOfModules R
          M : PresheafOfModules R
          s : M.sections
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (HSMul.hSMul (1 (((PresheafOfModules.unit R).map f).hom 1)) (↑s Y)) ((M.m …
        -/
        change R.map f 1 • s.eval Y = M.map f (1 • s.eval X)
        /-
          case hf.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          M✝ M₁ M₂ : PresheafOfModules R
          M : PresheafOfModules R
          s : M.sections
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (HSMul.hSMul ((R.map f).hom 1) (s.eval Y)) ((M.map f).hom (HSMul.hSMul 1  …
        -/
        simp }
        /-
          🎉 no goals
        -/
  left_inv f := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      M : PresheafOfModules R
      f : Quiver.Hom (PresheafOfModules.unit R) M
      ⊢ Eq ((fun s => { app := fun X => ModuleCat.ofHom ((LinearMap.ringLmapEquivSel …
    -/
    ext X : 2
    /-
      case h.hf
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      M : PresheafOfModules R
      f : Quiver.Hom (PresheafOfModules.unit R) M
      X : Opposite C
      ⊢ Eq (((fun s => { app := fun X => ModuleCat.ofHom ((LinearMap.ringLmapEquivSe …
    -/
    exact (LinearMap.ringLmapEquivSelf (R.obj X) ℤ (M.obj X)).symm_apply_apply (f.app X).hom
    /-
      🎉 no goals
    -/
  right_inv s := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      M : PresheafOfModules R
      s : M.sections
      ⊢ Eq ((fun f => PresheafOfModules.sectionsMk (fun X => (f.app X).hom 1) ⋯) ((f …
    -/
    ext X
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      M : PresheafOfModules R
      s : M.sections
      X : Opposite C
      ⊢ Eq (↑((fun f => PresheafOfModules.sectionsMk (fun X => (f.app X).hom 1) ⋯) ( …
    -/
    exact (LinearMap.ringLmapEquivSelf (R.obj X) ℤ (M.obj X)).apply_symm_apply (s.val X)
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `forgetToPresheafModuleCatObj`. -/
noncomputable abbrev forgetToPresheafModuleCatObjObj (Y : Cᵒᵖ) : ModuleCat (R.obj X) :=
  (ModuleCat.restrictScalars (R.map (hX.to Y)).hom).obj (M.obj Y)

-- This should not be a `simp` lemma because `M.obj Y` is missing the `Module (R.obj X)` instance,
-- so `simp`ing breaks downstream proofs.

lemma forgetToPresheafModuleCatObjObj_coe (Y : Cᵒᵖ) :
    (forgetToPresheafModuleCatObjObj X hX M Y : Type _) = M.obj Y := rfl


/-- Auxiliary definition for `forgetToPresheafModuleCatObj`. -/
def forgetToPresheafModuleCatObjMap {Y Z : Cᵒᵖ} (f : Y ⟶ Z) :
    forgetToPresheafModuleCatObjObj X hX M Y ⟶
      forgetToPresheafModuleCatObjObj X hX M Z :=
  ModuleCat.ofHom
    (X := forgetToPresheafModuleCatObjObj X hX M Y) (Y := forgetToPresheafModuleCatObjObj X hX M Z)
  { toFun := fun x => M.map f x
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     R : CategoryTheory.Functor (Opposite C) RingCat
                     M✝ M₁ M₂ : PresheafOfModules R
                     X : Opposite C
                     hX : CategoryTheory.Limits.IsInitial X
                     M : PresheafOfModules R
                     Y Z : Opposite C
                     f : Quiver.Hom Y Z
                     ⊢ ∀ (x y : ↑(PresheafOfModules.forgetToPresheafModuleCatObjObj X hX M Y)), Eq  …
                   -/
    map_add' := by simp
                   /-
                     🎉 no goals
                   -/
    map_smul' := fun r x => by
      simp only [ModuleCat.restrictScalars.smul_def, AddHom.toFun_eq_coe, AddHom.coe_mk,
        RingHom.id_apply, M.map_smul]
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        R : CategoryTheory.Functor (Opposite C) RingCat
        M✝ M₁ M₂ : PresheafOfModules R
        X : Opposite C
        hX : CategoryTheory.Limits.IsInitial X
        M : PresheafOfModules R
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        r : ↑(R.obj X)
        x : ↑(PresheafOfModules.forgetToPresheafModuleCatObjObj X hX M Y)
        ⊢ Eq (HSMul.hSMul ((R.map f).hom ((R.map (hX.to Y)).hom r)) ((M.map f).hom x)) …
      -/
      rw [← RingCat.comp_apply, ← R.map_comp]
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        R : CategoryTheory.Functor (Opposite C) RingCat
        M✝ M₁ M₂ : PresheafOfModules R
        X : Opposite C
        hX : CategoryTheory.Limits.IsInitial X
        M : PresheafOfModules R
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        r : ↑(R.obj X)
        x : ↑(PresheafOfModules.forgetToPresheafModuleCatObjObj X hX M Y)
        ⊢ Eq (HSMul.hSMul ((R.map (CategoryTheory.CategoryStruct.comp (hX.to Y) f)).ho …
      -/
      congr
      /-
        case e_a.e_a.e_self.e_a
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        R : CategoryTheory.Functor (Opposite C) RingCat
        M✝ M₁ M₂ : PresheafOfModules R
        X : Opposite C
        hX : CategoryTheory.Limits.IsInitial X
        M : PresheafOfModules R
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        r : ↑(R.obj X)
        x : ↑(PresheafOfModules.forgetToPresheafModuleCatObjObj X hX M Y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (hX.to Y) f) (hX.to Z)
      -/
      apply hX.hom_ext }
      /-
        🎉 no goals
      -/


@[simp]
lemma forgetToPresheafModuleCatObjMap_apply {Y Z : Cᵒᵖ} (f : Y ⟶ Z) (m : M.obj Y) :
  (forgetToPresheafModuleCatObjMap X hX M f).hom m = M.map f m := rfl


/--
Implementation of the functor `PresheafOfModules R ⥤ Cᵒᵖ ⥤ ModuleCat (R.obj X)`
when `X` is initial.

The functor is implemented as, on object level `M ↦ (c ↦ M(c))` where the `R(X)`-module structure
on `M(c)` is given by restriction of scalars along the unique morphism `R(c) ⟶ R(X)`; and on
morphism level `(f : M ⟶ N) ↦ (c ↦ f(c))`.
-/
@[simps]
noncomputable def forgetToPresheafModuleCatObj
    (X : Cᵒᵖ) (hX : Limits.IsInitial X) (M : PresheafOfModules.{v} R) :
    Cᵒᵖ ⥤ ModuleCat (R.obj X) where
  obj Y := forgetToPresheafModuleCatObjObj X hX M Y
  map f := forgetToPresheafModuleCatObjMap X hX M f


/--
Implementation of the functor `PresheafOfModules R ⥤ Cᵒᵖ ⥤ ModuleCat (R.obj X)`
when `X` is initial.

The functor is implemented as, on object level `M ↦ (c ↦ M(c))` where the `R(X)`-module structure
on `M(c)` is given by restriction of scalars along the unique morphism `R(c) ⟶ R(X)`; and on
morphism level `(f : M ⟶ N) ↦ (c ↦ f(c))`.
-/
noncomputable def forgetToPresheafModuleCatMap
    (X : Cᵒᵖ) (hX : Limits.IsInitial X) {M N : PresheafOfModules.{v} R} (f : M ⟶ N) :
    forgetToPresheafModuleCatObj X hX M ⟶ forgetToPresheafModuleCatObj X hX N where
  app Y := ModuleCat.ofHom
      (X := (forgetToPresheafModuleCatObj X hX M).obj Y)
      (Y := (forgetToPresheafModuleCatObj X hX N).obj Y)
    { toFun := f.app Y
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       R : CategoryTheory.Functor (Opposite C) RingCat
                       M✝ M₁ M₂ : PresheafOfModules R
                       X✝ : Opposite C
                       hX✝ : CategoryTheory.Limits.IsInitial X✝
                       X : Opposite C
                       hX : CategoryTheory.Limits.IsInitial X
                       M N : PresheafOfModules R
                       f : Quiver.Hom M N
                       Y : Opposite C
                       ⊢ ∀ (x y : ↑((PresheafOfModules.forgetToPresheafModuleCatObj X hX M).obj Y)),  …
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
      map_smul' := fun r ↦ (f.app Y).hom.map_smul (R.1.map (hX.to Y) _) }
  naturality Y Z g := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      X✝ : Opposite C
      hX✝ : CategoryTheory.Limits.IsInitial X✝
      X : Opposite C
      hX : CategoryTheory.Limits.IsInitial X
      M N : PresheafOfModules R
      f : Quiver.Hom M N
      Y Z : Opposite C
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.forgetToPresheafM …
    -/
    ext x
    /-
      case hf.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M✝ M₁ M₂ : PresheafOfModules R
      X✝ : Opposite C
      hX✝ : CategoryTheory.Limits.IsInitial X✝
      X : Opposite C
      hX : CategoryTheory.Limits.IsInitial X
      M N : PresheafOfModules R
      f : Quiver.Hom M N
      Y Z : Opposite C
      g : Quiver.Hom Y Z
      x : ↑((PresheafOfModules.forgetToPresheafModuleCatObj X hX M).obj Y)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((PresheafOfModules.forgetToPresheaf …
    -/
    exact naturality_apply f g x
    /-
      🎉 no goals
    -/


/--
The forgetful functor from presheaves of modules over a presheaf of rings `R` to presheaves of
`R(X)`-modules where `X` is an initial object.

The functor is implemented as, on object level `M ↦ (c ↦ M(c))` where the `R(X)`-module structure
on `M(c)` is given by restriction of scalars along the unique morphism `R(c) ⟶ R(X)`; and on
morphism level `(f : M ⟶ N) ↦ (c ↦ f(c))`.
-/
@[simps]
noncomputable def forgetToPresheafModuleCat (X : Cᵒᵖ) (hX : Limits.IsInitial X) :
    PresheafOfModules.{v} R ⥤ Cᵒᵖ ⥤ ModuleCat (R.obj X) where
  obj M := forgetToPresheafModuleCatObj X hX M
  map f := forgetToPresheafModuleCatMap X hX f


