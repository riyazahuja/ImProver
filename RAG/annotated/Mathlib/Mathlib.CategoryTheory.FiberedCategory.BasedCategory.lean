/-- A based category over `𝒮` is a category `𝒳` together with a functor `p : 𝒳 ⥤ 𝒮`. -/
@[nolint checkUnivs]
structure BasedCategory (𝒮 : Type u₁) [Category.{v₁} 𝒮] where
  /-- The type of objects in a `BasedCategory`-/
  obj : Type u₂
  /-- The underlying category of a `BasedCategory`. -/
  category : Category.{v₂} obj := by infer_instance
  /-- The functor to the base. -/
  p : obj ⥤ 𝒮


instance (𝒳 : BasedCategory.{v₂, u₂} 𝒮) : Category 𝒳.obj := 𝒳.category


/-- The based category associated to a functor `p : 𝒳 ⥤ 𝒮`. -/
def BasedCategory.ofFunctor {𝒳 : Type u₂} [Category.{v₂} 𝒳] (p : 𝒳 ⥤ 𝒮) : BasedCategory 𝒮 where
  obj := 𝒳
  p := p


/-- A functor between based categories is a functor between the underlying categories that commutes
with the projections. -/
structure BasedFunctor (𝒳 : BasedCategory.{v₂, u₂} 𝒮) (𝒴 : BasedCategory.{v₃, u₃} 𝒮) extends
    𝒳.obj ⥤ 𝒴.obj where
  w : toFunctor ⋙ 𝒴.p = 𝒳.p := by aesop_cat


/-- Notation for `BasedFunctor`. -/
scoped infixr:26 " ⥤ᵇ " => BasedFunctor


/-- The identity based functor. -/
@[simps]
def id (𝒳 : BasedCategory.{v₂, u₂} 𝒮) : 𝒳 ⥤ᵇ 𝒳 where
  toFunctor := 𝟭 𝒳.obj


/-- Notation for the identity functor on a based category. -/
scoped notation "𝟭" => BasedFunctor.id


/-- The composition of two based functors. -/
@[simps]
def comp {𝒵 : BasedCategory.{v₄, u₄} 𝒮} (F : 𝒳 ⥤ᵇ 𝒴) (G : 𝒴 ⥤ᵇ 𝒵) : 𝒳 ⥤ᵇ 𝒵 where
  toFunctor := F.toFunctor ⋙ G.toFunctor
          /-
            𝒮 : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
            𝒳 : CategoryTheory.BasedCategory 𝒮
            𝒴 : CategoryTheory.BasedCategory 𝒮
            𝒵 : CategoryTheory.BasedCategory 𝒮
            F : CategoryTheory.BasedFunctor 𝒳 𝒴
            G : CategoryTheory.BasedFunctor 𝒴 𝒵
            ⊢ Eq ((F.comp G.toFunctor).comp 𝒵.p) 𝒳.p
          -/
  w := by rw [Functor.assoc, G.w, F.w]
          /-
            🎉 no goals
          -/


/-- Notation for composition of based functors. -/
scoped infixr:80 " ⋙ " => BasedFunctor.comp


@[simp]
lemma comp_id (F : 𝒳 ⥤ᵇ 𝒴) :  F ⋙ 𝟭 𝒴 = F :=
  rfl


@[simp]
lemma id_comp (F : 𝒳 ⥤ᵇ 𝒴) : 𝟭 𝒳 ⋙ F = F :=
  rfl


@[simp]
lemma comp_assoc {𝒵 : BasedCategory.{v₄, u₄} 𝒮} {𝒜 : BasedCategory.{v₅, u₅} 𝒮} (F : 𝒳 ⥤ᵇ 𝒴)
    (G : 𝒴 ⥤ᵇ 𝒵) (H : 𝒵 ⥤ᵇ 𝒜) : (F ⋙ G) ⋙ H = F ⋙ (G ⋙ H) :=
  rfl


@[simp]
lemma w_obj (F : 𝒳 ⥤ᵇ 𝒴) (a : 𝒳.obj) : 𝒴.p.obj (F.obj a) = 𝒳.p.obj a := by
  /-
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F : CategoryTheory.BasedFunctor 𝒳 𝒴
    a : 𝒳.obj
    ⊢ Eq (𝒴.p.obj (F.obj a)) (𝒳.p.obj a)
  -/
  rw [← Functor.comp_obj, F.w]
  /-
    🎉 no goals
  -/


instance (F : 𝒳 ⥤ᵇ 𝒴) (a : 𝒳.obj) : IsHomLift 𝒴.p (𝟙 (𝒳.p.obj a)) (𝟙 (F.obj a)) :=
  IsHomLift.id (w_obj F a)


/-- For a based functor `F : 𝒳 ⟶ 𝒴`, then whenever an arrow `φ` in `𝒳` lifts some `f` in `𝒮`,
then `F(φ)` also lifts `f`. -/
instance preserves_isHomLift [IsHomLift 𝒳.p f φ] : IsHomLift 𝒴.p f (F.map φ) := by
  apply of_fac 𝒴.p f (F.map φ) (Eq.trans (F.w_obj a) (domain_eq 𝒳.p f φ))
    (Eq.trans (F.w_obj b) (codomain_eq 𝒳.p f φ))
  /-
    𝒮 : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F : CategoryTheory.BasedFunctor 𝒳 𝒴
    R S : 𝒮
    a b : 𝒳.obj
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : 𝒳.p.IsHomLift f φ
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
  -/
  rw [← Functor.comp_map, congr_hom F.w]
  /-
    𝒮 : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F : CategoryTheory.BasedFunctor 𝒳 𝒴
    R S : 𝒮
    a b : 𝒳.obj
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : 𝒳.p.IsHomLift f φ
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
  -/
  simpa using (fac 𝒳.p f φ)
  /-
    🎉 no goals
  -/


/-- For a based functor `F : 𝒳 ⟶ 𝒴`, and an arrow `φ` in `𝒳`, then `φ` lifts an arrow `f` in `𝒮`
if `F(φ)` does. -/
lemma isHomLift_map [IsHomLift 𝒴.p f (F.map φ)] : IsHomLift 𝒳.p f φ := by
  apply of_fac 𝒳.p f φ  (F.w_obj a ▸ domain_eq 𝒴.p f (F.map φ))
    (F.w_obj b ▸ codomain_eq 𝒴.p f (F.map φ))
  /-
    𝒮 : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F : CategoryTheory.BasedFunctor 𝒳 𝒴
    R S : 𝒮
    a b : 𝒳.obj
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : 𝒴.p.IsHomLift f (F.map φ)
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
  -/
  simp [congr_hom F.w.symm, fac 𝒴.p f (F.map φ)]
  /-
    🎉 no goals
  -/


lemma isHomLift_iff : IsHomLift 𝒴.p f (F.map φ) ↔ IsHomLift 𝒳.p f φ :=
  ⟨fun _ ↦ isHomLift_map F f φ, fun _ ↦ preserves_isHomLift F f φ⟩


/-- A `BasedNatTrans` between two `BasedFunctor`s is a natural transformation `α` between the
underlying functors, such that for all `a : 𝒳`, `α.app a` lifts `𝟙 S` whenever `𝒳.p.obj a = S`. -/
structure BasedNatTrans {𝒳 : BasedCategory.{v₂, u₂} 𝒮} {𝒴 : BasedCategory.{v₃, u₃} 𝒮}
    (F G : 𝒳 ⥤ᵇ 𝒴) extends CategoryTheory.NatTrans F.toFunctor G.toFunctor where
  isHomLift' : ∀ (a : 𝒳.obj), IsHomLift 𝒴.p (𝟙 (𝒳.p.obj a)) (toNatTrans.app a) := by aesop_cat


@[ext]
lemma ext (β : BasedNatTrans F G) (h : α.toNatTrans = β.toNatTrans) : α = β := by
  /-
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F G : CategoryTheory.BasedFunctor 𝒳 𝒴
    α β : CategoryTheory.BasedNatTrans F G
    h : Eq α.toNatTrans β.toNatTrans
    ⊢ Eq α β
  -/
  cases α; subst h; rfl
                    /-
                      🎉 no goals
                    -/


instance app_isHomLift (a : 𝒳.obj) : IsHomLift 𝒴.p (𝟙 (𝒳.p.obj a)) (α.toNatTrans.app a) :=
  α.isHomLift' a


lemma isHomLift {a : 𝒳.obj} {S : 𝒮} (ha : 𝒳.p.obj a = S) :
    IsHomLift 𝒴.p (𝟙 S) (α.toNatTrans.app a) := by
  /-
    𝒮 : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F G : CategoryTheory.BasedFunctor 𝒳 𝒴
    α : CategoryTheory.BasedNatTrans F G
    a : 𝒳.obj
    S : 𝒮
    ha : Eq (𝒳.p.obj a) S
    ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id S) (α.app a)
  -/
  subst ha; infer_instance
            /-
              🎉 no goals
            -/


/-- The identity natural transformation is a `BasedNatTrans`. -/
@[simps]
def id (F : 𝒳 ⥤ᵇ 𝒴) : BasedNatTrans F F where
  toNatTrans := CategoryTheory.NatTrans.id F.toFunctor
                                                                   /-
                                                                     𝒮 : Type u₁
                                                                     inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
                                                                     𝒳 : CategoryTheory.BasedCategory 𝒮
                                                                     𝒴 : CategoryTheory.BasedCategory 𝒮
                                                                     F : CategoryTheory.BasedFunctor 𝒳 𝒴
                                                                     a : 𝒳.obj
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (CategoryTheory.CategorySt …
                                                                   -/
  isHomLift' := fun a ↦ of_fac 𝒴.p _ _ (w_obj F a) (w_obj F a) (by simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Composition of `BasedNatTrans`, given by composition of the underlying natural
transformations. -/
@[simps]
def comp {F G H : 𝒳 ⥤ᵇ 𝒴} (α : BasedNatTrans F G) (β : BasedNatTrans G H) : BasedNatTrans F H where
  toNatTrans := CategoryTheory.NatTrans.vcomp α.toNatTrans β.toNatTrans
  isHomLift' := by
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      𝒳 : CategoryTheory.BasedCategory 𝒮
      𝒴 : CategoryTheory.BasedCategory 𝒮
      F G H : CategoryTheory.BasedFunctor 𝒳 𝒴
      α : CategoryTheory.BasedNatTrans F G
      β : CategoryTheory.BasedNatTrans G H
      ⊢ ∀ (a : 𝒳.obj), 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a))  …
    -/
    intro a
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      𝒳 : CategoryTheory.BasedCategory 𝒮
      𝒴 : CategoryTheory.BasedCategory 𝒮
      F G H : CategoryTheory.BasedFunctor 𝒳 𝒴
      α : CategoryTheory.BasedNatTrans F G
      β : CategoryTheory.BasedNatTrans G H
      a : 𝒳.obj
      ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) ((α.vcomp β.toN …
    -/
    rw [CategoryTheory.NatTrans.vcomp_app]
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      𝒳 : CategoryTheory.BasedCategory 𝒮
      𝒴 : CategoryTheory.BasedCategory 𝒮
      F G H : CategoryTheory.BasedFunctor 𝒳 𝒴
      α : CategoryTheory.BasedNatTrans F G
      β : CategoryTheory.BasedNatTrans G H
      a : 𝒳.obj
      ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (CategoryTheory …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[simps]
instance homCategory (𝒳 : BasedCategory.{v₂, u₂} 𝒮) (𝒴 : BasedCategory.{v₃, u₃} 𝒮) :
    Category (𝒳 ⥤ᵇ 𝒴) where
  Hom := BasedNatTrans
  id := BasedNatTrans.id
  comp := BasedNatTrans.comp


@[ext]
lemma homCategory.ext {F G : 𝒳 ⥤ᵇ 𝒴} (α β : F ⟶ G) (h : α.toNatTrans = β.toNatTrans) : α = β :=
  BasedNatTrans.ext α β h


/-- The forgetful functor from the category of based functors `𝒳 ⥤ᵇ 𝒴` to the category of
functors of underlying categories, `𝒳.obj ⥤ 𝒴.obj`. -/
@[simps]
def forgetful (𝒳 : BasedCategory.{v₂, u₂} 𝒮) (𝒴 : BasedCategory.{v₃, u₃} 𝒮) :
    (𝒳 ⥤ᵇ 𝒴) ⥤ (𝒳.obj ⥤ 𝒴.obj) where
  obj := fun F ↦ F.toFunctor
  map := fun α ↦ α.toNatTrans


instance : (forgetful 𝒳 𝒴).ReflectsIsomorphisms where
  reflects {F G} α _ := by
    /-
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      𝒳 : CategoryTheory.BasedCategory 𝒮
      𝒴 : CategoryTheory.BasedCategory 𝒮
      F G : CategoryTheory.BasedFunctor 𝒳 𝒴
      α : Quiver.Hom F G
      x✝ : CategoryTheory.IsIso ((CategoryTheory.BasedNatTrans.forgetful 𝒳 𝒴).map α)
      ⊢ CategoryTheory.IsIso α
    -/
    constructor
    use {
      toNatTrans := inv ((forgetful 𝒳 𝒴).map α)
      isHomLift' := fun a ↦ by simp [lift_id_inv_isIso] }
    /-
      case h
      𝒮 : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
      𝒳 : CategoryTheory.BasedCategory 𝒮
      𝒴 : CategoryTheory.BasedCategory 𝒮
      F G : CategoryTheory.BasedFunctor 𝒳 𝒴
      α : Quiver.Hom F G
      x✝ : CategoryTheory.IsIso ((CategoryTheory.BasedNatTrans.forgetful 𝒳 𝒴).map α)
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp α { toNatTrans := CategoryTheory …
    -/
    aesop
    /-
      🎉 no goals
    -/


instance {F G : 𝒳 ⥤ᵇ 𝒴} (α : F ⟶ G) [IsIso α] : IsIso (X := F.toFunctor) α.toNatTrans := by
  /-
    𝒮 : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
    𝒳 : CategoryTheory.BasedCategory 𝒮
    𝒴 : CategoryTheory.BasedCategory 𝒮
    F G : CategoryTheory.BasedFunctor 𝒳 𝒴
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    ⊢ CategoryTheory.IsIso α.toNatTrans
  -/
  rw [← forgetful_map]; infer_instance
                        /-
                          🎉 no goals
                        -/


/-- The identity natural transformation is a based natural isomorphism. -/
@[simps]
def id (F : 𝒳 ⥤ᵇ 𝒴) : F ≅ F where
  hom := 𝟙 F
  inv := 𝟙 F


/-- The inverse of a based natural transformation whose underlying natural transformation is an
isomorphism. -/
def mkNatIso (α : F.toFunctor ≅ G.toFunctor)
    (isHomLift' : ∀ a : 𝒳.obj, IsHomLift 𝒴.p (𝟙 (𝒳.p.obj a)) (α.hom.app a)) : F ≅ G where
  hom := { toNatTrans := α.hom }
  inv := {
    toNatTrans := α.inv
    isHomLift' := fun a ↦ by
      /-
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        𝒳 : CategoryTheory.BasedCategory 𝒮
        𝒴 : CategoryTheory.BasedCategory 𝒮
        F G : CategoryTheory.BasedFunctor 𝒳 𝒴
        α : CategoryTheory.Iso F.toFunctor G.toFunctor
        isHomLift' : ∀ (a : 𝒳.obj), 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳 …
        a : 𝒳.obj
        ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (α.inv.app a)
      -/
      have : 𝒴.p.IsHomLift (𝟙 (𝒳.p.obj a)) (α.app a).hom := (NatIso.app_hom α a) ▸ isHomLift' a
      /-
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        𝒳 : CategoryTheory.BasedCategory 𝒮
        𝒴 : CategoryTheory.BasedCategory 𝒮
        F G : CategoryTheory.BasedFunctor 𝒳 𝒴
        α : CategoryTheory.Iso F.toFunctor G.toFunctor
        isHomLift' : ∀ (a : 𝒳.obj), 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳 …
        a : 𝒳.obj
        this : 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (α.app a). …
        ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (α.inv.app a)
      -/
      rw [← NatIso.app_inv]
      /-
        𝒮 : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} 𝒮
        𝒳 : CategoryTheory.BasedCategory 𝒮
        𝒴 : CategoryTheory.BasedCategory 𝒮
        F G : CategoryTheory.BasedFunctor 𝒳 𝒴
        α : CategoryTheory.Iso F.toFunctor G.toFunctor
        isHomLift' : ∀ (a : 𝒳.obj), 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳 …
        a : 𝒳.obj
        this : 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (α.app a). …
        ⊢ 𝒴.p.IsHomLift (CategoryTheory.CategoryStruct.id (𝒳.p.obj a)) (α.app a).inv
      -/
      apply IsHomLift.lift_id_inv }
      /-
        🎉 no goals
      -/


lemma isIso_of_toNatTrans_isIso (α : F ⟶ G) [IsIso (X := F.toFunctor) α.toNatTrans] : IsIso α :=
                                             /-
                                               𝒮 : Type u₁
                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
                                               𝒳 : CategoryTheory.BasedCategory 𝒮
                                               𝒴 : CategoryTheory.BasedCategory 𝒮
                                               F G : CategoryTheory.BasedFunctor 𝒳 𝒴
                                               α : Quiver.Hom F G
                                               inst✝ : CategoryTheory.IsIso α.toNatTrans
                                               ⊢ CategoryTheory.IsIso ((CategoryTheory.BasedNatTrans.forgetful 𝒳 𝒴).map α)
                                             -/
  have : IsIso ((forgetful 𝒳 𝒴).map α) := by simp_all
                                             /-
                                               🎉 no goals
                                             -/
  Functor.ReflectsIsomorphisms.reflects (forgetful 𝒳 𝒴) α


/-- Left-whiskering in the bicategory `BasedCategory` is given by whiskering the underlying functors
and natural transformations. -/
@[simps]
def whiskerLeft {𝒵 : BasedCategory.{v₄, u₄} 𝒮} (F : 𝒳 ⥤ᵇ 𝒴) {G H : 𝒴 ⥤ᵇ 𝒵} (α : G ⟶ H) :
    F ⋙ G ⟶ F ⋙ H where
  toNatTrans := CategoryTheory.whiskerLeft F.toFunctor α.toNatTrans
  isHomLift' := fun a ↦ α.isHomLift (F.w_obj a)


/-- Right-whiskering in the bicategory `BasedCategory` is given by whiskering the underlying
functors and natural transformations. -/
@[simps]
def whiskerRight {𝒵 : BasedCategory.{v₄, u₄} 𝒮} {F G : 𝒳 ⥤ᵇ 𝒴} (α : F ⟶ G) (H : 𝒴 ⥤ᵇ 𝒵) :
    F ⋙ H ⟶ G ⋙ H where
  toNatTrans := CategoryTheory.whiskerRight α.toNatTrans H.toFunctor
  isHomLift' := fun _ ↦ BasedFunctor.preserves_isHomLift _ _ _


/-- The category of based categories. -/
@[simps]
instance : Category (BasedCategory.{v₂, u₂} 𝒮) where
  Hom := BasedFunctor
  id := id
  comp := comp


/-- The bicategory of based categories. -/
instance bicategory : Bicategory (BasedCategory.{v₂, u₂} 𝒮) where
  Hom 𝒳 𝒴 :=  𝒳 ⥤ᵇ 𝒴
  id 𝒳 := 𝟭 𝒳
  comp F G := F ⋙ G
  homCategory 𝒳 𝒴 := homCategory 𝒳 𝒴
  whiskerLeft {_ _ _} F {_ _} α := whiskerLeft F α
  whiskerRight {_ _ _} _ _ α H := whiskerRight α H
  associator _ _ _ := BasedNatIso.id _
  leftUnitor {_ _} F := BasedNatIso.id F
  rightUnitor {_ _} F := BasedNatIso.id F


/-- The bicategory structure on `BasedCategory.{v₂, u₂} 𝒮` is strict. -/
instance : Bicategory.Strict (BasedCategory.{v₂, u₂} 𝒮) where


