/-- The functor sending `X : C` to the constant functor `J ⥤ C` sending everything to `X`.
-/
@[simps]
def const : C ⥤ J ⥤ C where
  obj X :=
    { obj := fun _ => X
      map := fun _ => 𝟙 X }
  map f := { app := fun _ => f }


/-- The constant functor `Jᵒᵖ ⥤ Cᵒᵖ` sending everything to `op X`
is (naturally isomorphic to) the opposite of the constant functor `J ⥤ C` sending everything to `X`.
-/
@[simps]
def opObjOp (X : C) : (const Jᵒᵖ).obj (op X) ≅ ((const J).obj X).op where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


/-- The constant functor `Jᵒᵖ ⥤ C` sending everything to `unop X`
is (naturally isomorphic to) the opposite of
the constant functor `J ⥤ Cᵒᵖ` sending everything to `X`.
-/
def opObjUnop (X : Cᵒᵖ) : (const Jᵒᵖ).obj (unop X) ≅ ((const J).obj X).leftOp where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }

-- Lean needs some help with universes here.

@[simp]
theorem opObjUnop_hom_app (X : Cᵒᵖ) (j : Jᵒᵖ) : (opObjUnop.{v₁, v₂} X).hom.app j = 𝟙 _ :=
  rfl


@[simp]
theorem opObjUnop_inv_app (X : Cᵒᵖ) (j : Jᵒᵖ) : (opObjUnop.{v₁, v₂} X).inv.app j = 𝟙 _ :=
  rfl


@[simp]
theorem unop_functor_op_obj_map (X : Cᵒᵖ) {j₁ j₂ : J} (f : j₁ ⟶ j₂) :
    (unop ((Functor.op (const J)).obj X)).map f = 𝟙 (unop X) :=
  rfl


/-- These are actually equal, of course, but not definitionally equal
  (the equality requires F.map (𝟙 _) = 𝟙 _). A natural isomorphism is
  more convenient than an equality between functors (compare id_to_iso). -/
@[simps]
def constComp (X : C) (F : C ⥤ D) : (const J).obj X ⋙ F ≅ (const J).obj (F.obj X) where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


/-- If `J` is nonempty, then the constant functor over `J` is faithful. -/
instance [Nonempty J] : Faithful (const J : C ⥤ J ⥤ C) where
  map_injective e := NatTrans.congr_app e (Classical.arbitrary J)


/-- The canonical isomorphism
`F ⋙ Functor.const J ≅ Functor.const F ⋙ (whiskeringRight J _ _).obj L`. -/
@[simps!]
def compConstIso (F : C ⥤ D) :
    F ⋙ Functor.const J ≅ Functor.const J ⋙ (whiskeringRight J C D).obj F :=
  NatIso.ofComponents
                                                            /-
                                                              J : Type u₁
                                                              inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                              C : Type u₂
                                                              inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
                                                              D : Type u₃
                                                              inst✝ : CategoryTheory.Category.{v₃, u₃} D
                                                              F : CategoryTheory.Functor C D
                                                              X : C
                                                              ⊢ ∀ {X_1 Y : J} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
                                                            -/
    (fun X => NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat))
                                                            /-
                                                              🎉 no goals
                                                            -/
        /-
          J : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} J
          C : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝ : CategoryTheory.Category.{v₃, u₃} D
          F : CategoryTheory.Functor C D
          ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((F …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- The canonical isomorphism
`const D ⋙ (whiskeringLeft J _ _).obj F ≅ const J`.-/
@[simps!]
def constCompWhiskeringLeftIso (F : J ⥤ D) :
    const D ⋙ (whiskeringLeft J D C).obj F ≅ const J :=
                               /-
                                 J : Type u₁
                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                 C : Type u₂
                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
                                 D : Type u₃
                                 inst✝ : CategoryTheory.Category.{v₃, u₃} D
                                 F : CategoryTheory.Functor J D
                                 X : C
                                 ⊢ ∀ {X_1 Y : J} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => NatIso.ofComponents fun Y => Iso.refl _
  /-
    🎉 no goals
  -/


