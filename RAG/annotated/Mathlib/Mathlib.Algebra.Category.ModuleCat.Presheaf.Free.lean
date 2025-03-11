variable {R} in
/-- Given a presheaf of types `F : Cᵒᵖ ⥤ Type u`, this is the presheaf
of modules over `R` which sends `X : Cᵒᵖ` to the free `R.obj X`-module on `F.obj X`. -/
@[simps]
noncomputable def freeObj (F : Cᵒᵖ ⥤ Type u) : PresheafOfModules.{u} R where
  obj X := (ModuleCat.free (R.obj X)).obj (F.obj X)
  map {X Y} f := ModuleCat.freeDesc (fun x ↦ ModuleCat.freeMk (F.map f x))
               /-
                 C : Type u₁
                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                 R : CategoryTheory.Functor (Opposite C) RingCat
                 F : CategoryTheory.Functor (Opposite C) (Type u)
                 ⊢ ∀ (X : Opposite C), Eq ((fun {X Y} f => ModuleCat.freeDesc fun x => ModuleCa …
               -/
  map_id := by aesop
               /-
                 🎉 no goals
               -/


/-- The free presheaf of modules functor `(Cᵒᵖ ⥤ Type u) ⥤ PresheafOfModules.{u} R`. -/
@[simps]
noncomputable def free : (Cᵒᵖ ⥤ Type u) ⥤ PresheafOfModules.{u} R where
  obj := freeObj
  map {F G} φ :=
    { app := fun X ↦ (ModuleCat.free (R.obj X)).map (φ.app X)
      naturality := fun {X Y} f ↦ by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          F G : CategoryTheory.Functor (Opposite C) (Type u)
          φ : Quiver.Hom F G
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.freeObj F).map f) …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          F G : CategoryTheory.Functor (Opposite C) (Type u)
          φ : Quiver.Hom F G
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.freeDesc fun x => ModuleCa …
        -/
        ext x
        /-
          case h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          F G : CategoryTheory.Functor (Opposite C) (Type u)
          φ : Quiver.Hom F G
          X Y : Opposite C
          f : Quiver.Hom X Y
          x : F.obj X
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.freeDesc fun x => ModuleC …
        -/
        simp [FunctorToTypes.naturality] }
        /-
          🎉 no goals
        -/


/-- The morphism of presheaves of modules `freeObj F ⟶ G` corresponding to
a morphism `F ⟶ G.presheaf ⋙ forget _` of presheaves of types. -/
@[simps]
noncomputable def freeObjDesc (φ : F ⟶ G.presheaf ⋙ forget _) : freeObj F ⟶ G where
  app X := ModuleCat.freeDesc (φ.app X)
  naturality {X Y} f := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      F : CategoryTheory.Functor (Opposite C) (Type u)
      G : PresheafOfModules R
      φ : Quiver.Hom F (G.presheaf.comp (CategoryTheory.forget Ab))
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.freeObj F).map f) …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      F : CategoryTheory.Functor (Opposite C) (Type u)
      G : PresheafOfModules R
      φ : Quiver.Hom F (G.presheaf.comp (CategoryTheory.forget Ab))
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.freeDesc fun x => ModuleCa …
    -/
    ext x
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      F : CategoryTheory.Functor (Opposite C) (Type u)
      G : PresheafOfModules R
      φ : Quiver.Hom F (G.presheaf.comp (CategoryTheory.forget Ab))
      X Y : Opposite C
      f : Quiver.Hom X Y
      x : F.obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.freeDesc fun x => ModuleC …
    -/
    simpa using NatTrans.naturality_apply φ f x
    /-
      🎉 no goals
    -/


variable (F R) in
/-- The unit of `PresheafOfModules.freeAdjunction`. -/
@[simps]
noncomputable def freeAdjunctionUnit : F ⟶ (freeObj (R := R) F).presheaf ⋙ forget _ where
  app X x := ModuleCat.freeMk x
                         /-
                           C : Type u₁
                           inst✝ : CategoryTheory.Category.{v₁, u₁} C
                           R : CategoryTheory.Functor (Opposite C) RingCat
                           F : CategoryTheory.Functor (Opposite C) (Type u)
                           G : PresheafOfModules R
                           X Y : Opposite C
                           f : Quiver.Hom X Y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X x => ModuleCat.free …
                         -/
  naturality X Y f := by ext; simp [presheaf]
                              /-
                                🎉 no goals
                              -/


/-- The bijection `(freeObj F ⟶ G) ≃ (F ⟶ G.presheaf ⋙ forget _)` when
`F` is a presheaf of types and `G` a presheaf of modules. -/
noncomputable def freeHomEquiv : (freeObj F ⟶ G) ≃ (F ⟶ G.presheaf ⋙ forget _) where
  toFun ψ := freeAdjunctionUnit R F ≫ whiskerRight ((toPresheaf _).map ψ) _
  invFun φ := freeObjDesc φ
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     R : CategoryTheory.Functor (Opposite C) RingCat
                     F : CategoryTheory.Functor (Opposite C) (Type u)
                     G : PresheafOfModules R
                     ψ : Quiver.Hom (PresheafOfModules.freeObj F) G
                     ⊢ Eq ((fun φ => PresheafOfModules.freeObjDesc φ) ((fun ψ => CategoryTheory.Cat …
                   -/
  left_inv ψ := by ext1 X; dsimp; ext x; simp [toPresheaf]
                                         /-
                                           🎉 no goals
                                         -/
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      R : CategoryTheory.Functor (Opposite C) RingCat
                      F : CategoryTheory.Functor (Opposite C) (Type u)
                      G : PresheafOfModules R
                      φ : Quiver.Hom F (G.presheaf.comp (CategoryTheory.forget Ab))
                      ⊢ Eq ((fun ψ => CategoryTheory.CategoryStruct.comp (PresheafOfModules.freeAdju …
                    -/
  right_inv φ := by ext; simp [toPresheaf]
                         /-
                           🎉 no goals
                         -/


lemma free_hom_ext {ψ ψ' : freeObj F ⟶ G}
    (h : freeAdjunctionUnit R F ≫ whiskerRight ((toPresheaf _).map ψ) _ =
      freeAdjunctionUnit R F ≫ whiskerRight ((toPresheaf _).map ψ') _ ) : ψ = ψ' :=
  freeHomEquiv.injective h


variable (R) in
/-- The free presheaf of modules functor is left adjoint to the forget functor
`PresheafOfModules.{u} R ⥤ Cᵒᵖ ⥤ Type u`. -/
noncomputable def freeAdjunction :
    free.{u} R ⊣ (toPresheaf R ⋙ (whiskeringRight _ _ _).obj (forget Ab)) :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ ↦ freeHomEquiv
      homEquiv_naturality_left_symm := fun {F₁ F₂ G} f g ↦
                         /-
                           C : Type u₁
                           inst✝ : CategoryTheory.Category.{v₁, u₁} C
                           R : CategoryTheory.Functor (Opposite C) RingCat
                           F : CategoryTheory.Functor (Opposite C) (Type u)
                           G✝ : PresheafOfModules R
                           F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type u)
                           G : PresheafOfModules R
                           f : Quiver.Hom F₁ F₂
                           g : Quiver.Hom F₂ (((PresheafOfModules.toPresheaf R).comp ((CategoryTheory.whi …
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (PresheafOfModules.freeAdjunctionUnit …
                         -/
        free_hom_ext (by ext; simp [freeHomEquiv, toPresheaf])
                              /-
                                🎉 no goals
                              -/
      homEquiv_naturality_right := fun {F G₁ G₂} f g ↦ rfl }


variable (F G) in
@[simp]
lemma freeAdjunction_homEquiv : (freeAdjunction R).homEquiv F G = freeHomEquiv := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    F : CategoryTheory.Functor (Opposite C) (Type u)
    G : PresheafOfModules R
    ⊢ Eq ((PresheafOfModules.freeAdjunction R).homEquiv F G) PresheafOfModules.fre …
  -/
  simp [freeAdjunction, Adjunction.mkOfHomEquiv_homEquiv]
  /-
    🎉 no goals
  -/


variable (R F) in
@[simp]
lemma freeAdjunction_unit_app :
    (freeAdjunction R).unit.app F = freeAdjunctionUnit R F := rfl


