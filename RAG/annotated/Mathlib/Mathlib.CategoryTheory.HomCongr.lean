/-- If `X` is isomorphic to `X₁` and `Y` is isomorphic to `Y₁`, then
there is a natural bijection between `X ⟶ Y` and `X₁ ⟶ Y₁`. See also `Equiv.arrowCongr`. -/
@[simps]
def homCongr {X Y X₁ Y₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) : (X ⟶ Y) ≃ (X₁ ⟶ Y₁) where
  toFun f := α.inv ≫ f ≫ β.hom
  invFun f := α.hom ≫ f ≫ β.inv
  left_inv f :=
    show α.hom ≫ (α.inv ≫ f ≫ β.hom) ≫ β.inv = f by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y X₁ Y₁ : C
        α : CategoryTheory.Iso X X₁
        β : CategoryTheory.Iso Y Y₁
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp α.hom (CategoryTheory.CategoryStruct. …
      -/
      rw [Category.assoc, Category.assoc, β.hom_inv_id, α.hom_inv_id_assoc, Category.comp_id]
      /-
        🎉 no goals
      -/
  right_inv f :=
    show α.inv ≫ (α.hom ≫ f ≫ β.inv) ≫ β.hom = f by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y X₁ Y₁ : C
        α : CategoryTheory.Iso X X₁
        β : CategoryTheory.Iso Y Y₁
        f : Quiver.Hom X₁ Y₁
        ⊢ Eq (CategoryTheory.CategoryStruct.comp α.inv (CategoryTheory.CategoryStruct. …
      -/
      rw [Category.assoc, Category.assoc, β.inv_hom_id, α.inv_hom_id_assoc, Category.comp_id]
      /-
        🎉 no goals
      -/


theorem homCongr_comp {X Y Z X₁ Y₁ Z₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) (γ : Z ≅ Z₁) (f : X ⟶ Y)
                                                                               /-
                                                                                 C : Type u
                                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                                 X Y Z X₁ Y₁ Z₁ : C
                                                                                 α : CategoryTheory.Iso X X₁
                                                                                 β : CategoryTheory.Iso Y Y₁
                                                                                 γ : CategoryTheory.Iso Z Z₁
                                                                                 f : Quiver.Hom X Y
                                                                                 g : Quiver.Hom Y Z
                                                                                 ⊢ Eq ((α.homCongr γ) (CategoryTheory.CategoryStruct.comp f g)) (CategoryTheory …
                                                                               -/
    (g : Y ⟶ Z) : α.homCongr γ (f ≫ g) = α.homCongr β f ≫ β.homCongr γ g := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


                                                                                             /-
                                                                                               C : Type u
                                                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                                                               X Y : C
                                                                                               f : Quiver.Hom X Y
                                                                                               ⊢ Eq (((CategoryTheory.Iso.refl X).homCongr (CategoryTheory.Iso.refl Y)) f) f
                                                                                             -/
theorem homCongr_refl {X Y : C} (f : X ⟶ Y) : (Iso.refl X).homCongr (Iso.refl Y) f = f := by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


theorem homCongr_trans {X₁ Y₁ X₂ Y₂ X₃ Y₃ : C} (α₁ : X₁ ≅ X₂) (β₁ : Y₁ ≅ Y₂) (α₂ : X₂ ≅ X₃)
    (β₂ : Y₂ ≅ Y₃) (f : X₁ ⟶ Y₁) :
                                                                                       /-
                                                                                         C : Type u
                                                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                                                         X₁ Y₁ X₂ Y₂ X₃ Y₃ : C
                                                                                         α₁ : CategoryTheory.Iso X₁ X₂
                                                                                         β₁ : CategoryTheory.Iso Y₁ Y₂
                                                                                         α₂ : CategoryTheory.Iso X₂ X₃
                                                                                         β₂ : CategoryTheory.Iso Y₂ Y₃
                                                                                         f : Quiver.Hom X₁ Y₁
                                                                                         ⊢ Eq (((α₁.trans α₂).homCongr (β₁.trans β₂)) f) (((α₁.homCongr β₁).trans (α₂.h …
                                                                                       -/
    (α₁ ≪≫ α₂).homCongr (β₁ ≪≫ β₂) f = (α₁.homCongr β₁).trans (α₂.homCongr β₂) f := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem homCongr_symm {X₁ Y₁ X₂ Y₂ : C} (α : X₁ ≅ X₂) (β : Y₁ ≅ Y₂) :
    (α.homCongr β).symm = α.symm.homCongr β.symm :=
  rfl


/-- If `X` is isomorphic to `X₁` and `Y` is isomorphic to `Y₁`, then
there is a bijection between `X ≅ Y` and `X₁ ≅ Y₁`. -/
@[simps]
def isoCongr {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ≅ X₂) (g : Y₁ ≅ Y₂) : (X₁ ≅ Y₁) ≃ (X₂ ≅ Y₂) where
  toFun h := f.symm.trans <| h.trans <| g
  invFun h := f.trans <| h.trans <| g.symm
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X₁ Y₁ X₂ Y₂ : C
                   f : CategoryTheory.Iso X₁ X₂
                   g : CategoryTheory.Iso Y₁ Y₂
                   ⊢ Function.LeftInverse (fun h => f.trans (h.trans g.symm)) fun h => f.symm.tra …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    X₁ Y₁ X₂ Y₂ : C
                    f : CategoryTheory.Iso X₁ X₂
                    g : CategoryTheory.Iso Y₁ Y₂
                    ⊢ Function.RightInverse (fun h => f.trans (h.trans g.symm)) fun h => f.symm.tr …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- If `X₁` is isomorphic to `X₂`, then there is a bijection between `X₁ ≅ Y` and `X₂ ≅ Y`. -/
def isoCongrLeft {X₁ X₂ Y : C} (f : X₁ ≅ X₂) : (X₁ ≅ Y) ≃ (X₂ ≅ Y) :=
  isoCongr f (Iso.refl _)


/-- If `Y₁` is isomorphic to `Y₂`, then there is a bijection between `X ≅ Y₁` and `X ≅ Y₂`. -/
def isoCongrRight {X Y₁ Y₂ : C} (g : Y₁ ≅ Y₂) : (X ≅ Y₁) ≃ (X ≅ Y₂) :=
  isoCongr (Iso.refl _) g


theorem map_homCongr {X Y X₁ Y₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) (f : X ⟶ Y) :
                                                                                        /-
                                                                                          C : Type u
                                                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                          D : Type u₁
                                                                                          inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                                                                          F : CategoryTheory.Functor C D
                                                                                          X Y X₁ Y₁ : C
                                                                                          α : CategoryTheory.Iso X X₁
                                                                                          β : CategoryTheory.Iso Y Y₁
                                                                                          f : Quiver.Hom X Y
                                                                                          ⊢ Eq (F.map ((α.homCongr β) f)) (((F.mapIso α).homCongr (F.mapIso β)) (F.map f))
                                                                                        -/
    F.map (Iso.homCongr α β f) = Iso.homCongr (F.mapIso α) (F.mapIso β) (F.map f) := by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem map_isoCongr {X Y X₁ Y₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) (f : X ≅ Y) :
    F.mapIso (Iso.isoCongr α β f) = Iso.isoCongr (F.mapIso α) (F.mapIso β) (F.mapIso f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    F : CategoryTheory.Functor C D
    X Y X₁ Y₁ : C
    α : CategoryTheory.Iso X X₁
    β : CategoryTheory.Iso Y Y₁
    f : CategoryTheory.Iso X Y
    ⊢ Eq (F.mapIso ((α.isoCongr β) f)) (((F.mapIso α).isoCongr (F.mapIso β)) (F.ma …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    F : CategoryTheory.Functor C D
    X Y X₁ Y₁ : C
    α : CategoryTheory.Iso X X₁
    β : CategoryTheory.Iso Y Y₁
    f : CategoryTheory.Iso X Y
    ⊢ Eq (F.mapIso ((α.isoCongr β) f)).hom (((F.mapIso α).isoCongr (F.mapIso β)) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


