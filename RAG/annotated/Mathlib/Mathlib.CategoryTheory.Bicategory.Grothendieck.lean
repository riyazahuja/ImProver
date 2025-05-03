/-- The type of objects in the fibered category associated to a presheaf valued in types. -/
@[ext]
structure Pseudofunctor.Grothendieck (F : Pseudofunctor (LocallyDiscrete 𝒮ᵒᵖ) Cat.{v₂, u₂}) where
  /-- The underlying object in the base category. -/
  base : 𝒮
  /-- The object in the fiber of the base object. -/
  fiber : F.obj ⟨op base⟩


/-- Notation for the Grothendieck category associated to a pseudofunctor `F`. -/
scoped prefix:75 "∫ " => Pseudofunctor.Grothendieck


/-- A morphism in the Grothendieck category `F : C ⥤ Cat` consists of
`base : X.base ⟶ Y.base` and `f.fiber : (F.map base).obj X.fiber ⟶ Y.fiber`.
-/
structure Hom (X Y : ∫ F) where
  /-- The morphism between base objects. -/
  base : X.base ⟶ Y.base
  /-- The morphism in the fiber over the domain. -/
  fiber : X.fiber ⟶ (F.map base.op.toLoc).obj Y.fiber


@[simps!]
instance categoryStruct : CategoryStruct (∫ F) where
  Hom X Y := Hom X Y
  id X := {
    base := 𝟙 X.base
    fiber := (F.mapId ⟨op X.base⟩).inv.app X.fiber }
  comp {_ _ Z} f g := {
    base := f.base ≫ g.base
    fiber := f.fiber ≫ (F.map f.base.op.toLoc).map g.fiber ≫
      (F.mapComp g.base.op.toLoc f.base.op.toLoc).inv.app Z.fiber }


@[ext (iff := false)]
lemma Hom.ext (f g : a ⟶ b) (hfg₁ : f.base = g.base)
    (hfg₂ : f.fiber = g.fiber ≫ eqToHom (hfg₁ ▸ rfl)) : f = g := by
  /-
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    f g : Quiver.Hom a b
    hfg₁ : Eq f.base g.base
    hfg₂ : Eq f.fiber (CategoryTheory.CategoryStruct.comp g.fiber (CategoryTheory. …
    ⊢ Eq f g
  -/
  cases f; cases g
  /-
    case mk.mk
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    base✝¹ : Quiver.Hom a.base b.base
    fiber✝¹ : Quiver.Hom a.fiber ((F.map base✝¹.op.toLoc).obj b.fiber)
    base✝ : Quiver.Hom a.base b.base
    fiber✝ : Quiver.Hom a.fiber ((F.map base✝.op.toLoc).obj b.fiber)
    hfg₁ : Eq { base := base✝¹, fiber := fiber✝¹ }.base { base := base✝, fiber :=  …
    hfg₂ : Eq { base := base✝¹, fiber := fiber✝¹ }.fiber (CategoryTheory.CategoryS …
    ⊢ Eq { base := base✝¹, fiber := fiber✝¹ } { base := base✝, fiber := fiber✝ }
  -/
  congr
  /-
    case mk.mk.h.e_7
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    base✝¹ : Quiver.Hom a.base b.base
    fiber✝¹ : Quiver.Hom a.fiber ((F.map base✝¹.op.toLoc).obj b.fiber)
    base✝ : Quiver.Hom a.base b.base
    fiber✝ : Quiver.Hom a.fiber ((F.map base✝.op.toLoc).obj b.fiber)
    hfg₁ : Eq { base := base✝¹, fiber := fiber✝¹ }.base { base := base✝, fiber :=  …
    hfg₂ : Eq { base := base✝¹, fiber := fiber✝¹ }.fiber (CategoryTheory.CategoryS …
    ⊢ HEq fiber✝¹ fiber✝
  -/
  dsimp at hfg₁
  /-
    case mk.mk.h.e_7
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    base✝¹ : Quiver.Hom a.base b.base
    fiber✝¹ : Quiver.Hom a.fiber ((F.map base✝¹.op.toLoc).obj b.fiber)
    base✝ : Quiver.Hom a.base b.base
    fiber✝ : Quiver.Hom a.fiber ((F.map base✝.op.toLoc).obj b.fiber)
    hfg₁ : Eq base✝¹ base✝
    hfg₂ : Eq { base := base✝¹, fiber := fiber✝¹ }.fiber (CategoryTheory.CategoryS …
    ⊢ HEq fiber✝¹ fiber✝
  -/
  rw [← conj_eqToHom_iff_heq _ _ rfl (hfg₁ ▸ rfl)]
  /-
    case mk.mk.h.e_7
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    base✝¹ : Quiver.Hom a.base b.base
    fiber✝¹ : Quiver.Hom a.fiber ((F.map base✝¹.op.toLoc).obj b.fiber)
    base✝ : Quiver.Hom a.base b.base
    fiber✝ : Quiver.Hom a.fiber ((F.map base✝.op.toLoc).obj b.fiber)
    hfg₁ : Eq base✝¹ base✝
    hfg₂ : Eq { base := base✝¹, fiber := fiber✝¹ }.fiber (CategoryTheory.CategoryS …
    ⊢ Eq fiber✝¹ (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (C …
  -/
  simpa only [eqToHom_refl, id_comp] using hfg₂
  /-
    🎉 no goals
  -/


lemma Hom.ext_iff (f g : a ⟶ b) :
    f = g ↔ ∃ (hfg : f.base = g.base), f.fiber = g.fiber ≫ eqToHom (hfg ▸ rfl) where
                /-
                  𝒮 : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
                  F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
                  a b : F.Grothendieck
                  f g : Quiver.Hom a b
                  hfg : Eq f g
                  ⊢ Eq f.base g.base
                -/
                /-
                  🎉 no goals
                -/
  mp hfg := ⟨by rw [hfg], by simp [hfg]⟩
                             /-
                               🎉 no goals
                             -/
  mpr := fun ⟨hfg₁, hfg₂⟩ => Hom.ext f g hfg₁ hfg₂


lemma Hom.congr {a b : ∫ F} {f g : a ⟶ b} (h : f = g) :
    f.fiber = g.fiber ≫ eqToHom (h ▸ rfl) := by
  /-
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
    a b : F.Grothendieck
    f g : Quiver.Hom a b
    h : Eq f g
    ⊢ Eq f.fiber (CategoryTheory.CategoryStruct.comp g.fiber (CategoryTheory.eqToH …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


/-- The category structure on `∫ F`. -/
instance category : Category (∫ F) where
  toCategoryStruct := Pseudofunctor.Grothendieck.categoryStruct
  id_comp {a b} f := by
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
      a b : F.Grothendieck
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a)  …
    -/
    ext
      /-
        case hfg₁
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        a b : F.Grothendieck
        f : Quiver.Hom a b
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a)  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case hfg₂
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        a b : F.Grothendieck
        f : Quiver.Hom a b
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a)  …
      -/
    · simp [F.mapComp_id_right_inv_app, Strict.rightUnitor_eqToIso, ← NatTrans.naturality_assoc]
      /-
        🎉 no goals
      -/
  comp_id {a b} f := by
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
      a b : F.Grothendieck
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id b …
    -/
    ext
      /-
        case hfg₁
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        a b : F.Grothendieck
        f : Quiver.Hom a b
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id b …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case hfg₂
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        a b : F.Grothendieck
        f : Quiver.Hom a b
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id b …
      -/
    · simp [F.mapComp_id_left_inv_app, ← Functor.map_comp_assoc, Strict.leftUnitor_eqToIso]
      /-
        🎉 no goals
      -/
  assoc f g h := by
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
      W✝ X✝ Y✝ Z✝ : F.Grothendieck
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    ext
      /-
        case hfg₁
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        W✝ X✝ Y✝ Z✝ : F.Grothendieck
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case hfg₂
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        F : CategoryTheory.Pseudofunctor (CategoryTheory.LocallyDiscrete (Opposite 𝒮)) …
        W✝ X✝ Y✝ Z✝ : F.Grothendieck
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · simp [← NatTrans.naturality_assoc, F.mapComp_assoc_right_inv_app, Strict.associator_eqToIso]
      /-
        🎉 no goals
      -/


/-- The projection `∫ F ⥤ 𝒮` given by projecting both objects and homs to the first
factor. -/
@[simps]
def forget : ∫ F ⥤ 𝒮 where
  obj X := X.base
  map f := f.base


