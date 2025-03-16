/-- `prod F G` is the explicit binary product of type-valued functors `F` and `G`. -/
def prod : C ⥤ Type w where
  obj a := F.obj a × G.obj a
  map f a := (F.map f a.1, G.map f a.2)


/-- The first projection of `prod F G`, onto `F`. -/
@[simps]
def prod.fst : prod F G ⟶ F where
  app _ a := a.1


/-- The second projection of `prod F G`, onto `G`. -/
@[simps]
def prod.snd : prod F G ⟶ G where
  app _ a := a.2


/-- Given natural transformations `F ⟶ F₁` and `F ⟶ F₂`, construct
a natural transformation `F ⟶ prod F₁ F₂`. -/
@[simps]
def prod.lift {F₁ F₂ : C ⥤ Type w} (τ₁ : F ⟶ F₁) (τ₂ : F ⟶ F₂) :
    F ⟶ prod F₁ F₂ where
  app x y := ⟨τ₁.app x y, τ₂.app x y⟩
  naturality _ _ _ := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G F₁ F₂ : CategoryTheory.Functor C (Type w)
      τ₁ : Quiver.Hom F F₁
      τ₂ : Quiver.Hom F F₂
      x✝² x✝¹ : C
      x✝ : Quiver.Hom x✝² x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map x✝) ((fun x y => { fst := τ₁.a …
    -/
    ext a
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G F₁ F₂ : CategoryTheory.Functor C (Type w)
      τ₁ : Quiver.Hom F F₁
      τ₂ : Quiver.Hom F F₂
      x✝² x✝¹ : C
      x✝ : Quiver.Hom x✝² x✝¹
      a : F.obj x✝²
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map x✝) ((fun x y => { fst := τ₁.a …
    -/
    simp only [types_comp_apply, FunctorToTypes.naturality]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G F₁ F₂ : CategoryTheory.Functor C (Type w)
      τ₁ : Quiver.Hom F F₁
      τ₂ : Quiver.Hom F F₂
      x✝² x✝¹ : C
      x✝ : Quiver.Hom x✝² x✝¹
      a : F.obj x✝²
      ⊢ Eq { fst := F₁.map x✝ (τ₁.app x✝² a), snd := F₂.map x✝ (τ₂.app x✝² a) } ((Ca …
    -/
    aesop
    /-
      🎉 no goals
    -/


@[simp]
lemma prod.lift_fst {F₁ F₂ : C ⥤ Type w} (τ₁ : F ⟶ F₁) (τ₂ : F ⟶ F₂) :
    prod.lift τ₁ τ₂ ≫ prod.fst = τ₁ := rfl


@[simp]
lemma prod.lift_snd {F₁ F₂ : C ⥤ Type w} (τ₁ : F ⟶ F₁) (τ₂ : F ⟶ F₂) :
    prod.lift τ₁ τ₂ ≫ prod.snd = τ₂ := rfl


/-- The binary fan whose point is `prod F G`. -/
@[simps!]
def binaryProductCone : BinaryFan F G :=
  BinaryFan.mk prod.fst prod.snd


/-- `prod F G` is a limit cone. -/
@[simps]
def binaryProductLimit : IsLimit (binaryProductCone F G) where
  lift (s : BinaryFan F G) := prod.lift s.fst s.snd
  fac _ := fun ⟨j⟩ ↦ WalkingPair.casesOn j rfl rfl
  uniq _ _ h := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C (Type w)
      x✝¹ : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair F G)
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.FunctorToTypes.binaryProductCone F G).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq x✝ ((fun s => CategoryTheory.FunctorToTypes.prod.lift s.fst s.snd) x✝¹)
    -/
    simp only [← h ⟨WalkingPair.right⟩, ← h ⟨WalkingPair.left⟩]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C (Type w)
      x✝¹ : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair F G)
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.FunctorToTypes.binaryProductCone F G).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq x✝ (CategoryTheory.FunctorToTypes.prod.lift (CategoryTheory.CategoryStruc …
    -/
    congr
    /-
      🎉 no goals
    -/


/-- `prod F G` is a binary product for `F` and `G`. -/
def binaryProductLimitCone : Limits.LimitCone (pair F G) :=
  ⟨_, binaryProductLimit F G⟩


/-- The categorical binary product of type-valued functors is `prod F G`. -/
noncomputable def binaryProductIso : F ⨯ G ≅ prod F G :=
  limit.isoLimitCone (binaryProductLimitCone F G)


@[simp]
lemma binaryProductIso_hom_comp_fst :
    (binaryProductIso F G).hom ≫ prod.fst = Limits.prod.fst := rfl


@[simp]
lemma binaryProductIso_hom_comp_snd :
    (binaryProductIso F G).hom ≫ prod.snd = Limits.prod.snd := rfl


@[simp]
lemma binaryProductIso_inv_comp_fst :
    (binaryProductIso F G).inv ≫ Limits.prod.fst = prod.fst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.FunctorToTypes.binary …
  -/
  simp [binaryProductIso, binaryProductLimitCone]
  /-
    🎉 no goals
  -/


@[simp]
lemma binaryProductIso_inv_comp_fst_apply (a : C) (z : (prod F G).obj a) :
    (Limits.prod.fst (X := F)).app a ((binaryProductIso F G).inv.app a z) = z.1 :=
  congr_fun (congr_app (binaryProductIso_inv_comp_fst F G) a) z


@[simp]
lemma binaryProductIso_inv_comp_snd :
    (binaryProductIso F G).inv ≫ Limits.prod.snd = prod.snd := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.FunctorToTypes.binary …
  -/
  simp [binaryProductIso, binaryProductLimitCone]
  /-
    🎉 no goals
  -/


@[simp]
lemma binaryProductIso_inv_comp_snd_apply (a : C) (z : (prod F G).obj a) :
    (Limits.prod.snd (X := F)).app a ((binaryProductIso F G).inv.app a z) = z.2 :=
  congr_fun (congr_app (binaryProductIso_inv_comp_snd F G) a) z


/-- Construct an element of `(F ⨯ G).obj a` from an element of `F.obj a` and
an element of `G.obj a`. -/
noncomputable
def prodMk {a : C} (x : F.obj a) (y : G.obj a) : (F ⨯ G).obj a :=
  ((binaryProductIso F G).inv).app a ⟨x, y⟩


@[simp]
lemma prodMk_fst {a : C} (x : F.obj a) (y : G.obj a) :
    (Limits.prod.fst (X := F)).app a (prodMk x y) = x := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    a : C
    x : F.obj a
    y : G.obj a
    ⊢ Eq (CategoryTheory.Limits.prod.fst.app a (CategoryTheory.FunctorToTypes.prod …
  -/
  simp only [prodMk, binaryProductIso_inv_comp_fst_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma prodMk_snd {a : C} (x : F.obj a) (y : G.obj a) :
    (Limits.prod.snd (X := F)).app a (prodMk x y) = y := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    a : C
    x : F.obj a
    y : G.obj a
    ⊢ Eq (CategoryTheory.Limits.prod.snd.app a (CategoryTheory.FunctorToTypes.prod …
  -/
  simp only [prodMk, binaryProductIso_inv_comp_snd_apply]
  /-
    🎉 no goals
  -/


@[ext]
lemma prod_ext {a : C} (z w : (prod F G).obj a) (h1 : z.1 = w.1) (h2 : z.2 = w.2) :
    z = w := Prod.ext h1 h2


/-- `(F ⨯ G).obj a` is in bijection with the product of `F.obj a` and `G.obj a`. -/
@[simps]
noncomputable
def binaryProductEquiv (a : C) : (F ⨯ G).obj a ≃ (F.obj a) × (G.obj a) where
  toFun z := ⟨((binaryProductIso F G).hom.app a z).1, ((binaryProductIso F G).hom.app a z).2⟩
  invFun z := prodMk z.1 z.2
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     F G : CategoryTheory.Functor C (Type w)
                     a : C
                     x✝ : (CategoryTheory.Limits.prod F G).obj a
                     ⊢ Eq ((fun z => CategoryTheory.FunctorToTypes.prodMk z.1 z.2) ((fun z => { fst …
                   -/
  left_inv _ := by simp [prodMk]
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      F G : CategoryTheory.Functor C (Type w)
                      a : C
                      x✝ : Prod (F.obj a) (G.obj a)
                      ⊢ Eq ((fun z => { fst := ((CategoryTheory.FunctorToTypes.binaryProductIso F G) …
                    -/
  right_inv _ := by simp [prodMk]
                    /-
                      🎉 no goals
                    -/


@[ext]
lemma prod_ext' (a : C) (z w : (F ⨯ G).obj a)
    (h1 : (Limits.prod.fst (X := F)).app a z = (Limits.prod.fst (X := F)).app a w)
    (h2 : (Limits.prod.snd (X := F)).app a z = (Limits.prod.snd (X := F)).app a w) :
    z = w := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    a : C
    z w : (CategoryTheory.Limits.prod F G).obj a
    h1 : Eq (CategoryTheory.Limits.prod.fst.app a z) (CategoryTheory.Limits.prod.f …
    h2 : Eq (CategoryTheory.Limits.prod.snd.app a z) (CategoryTheory.Limits.prod.s …
    ⊢ Eq z w
  -/
  apply Equiv.injective (binaryProductEquiv F G a)
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    a : C
    z w : (CategoryTheory.Limits.prod F G).obj a
    h1 : Eq (CategoryTheory.Limits.prod.fst.app a z) (CategoryTheory.Limits.prod.f …
    h2 : Eq (CategoryTheory.Limits.prod.snd.app a z) (CategoryTheory.Limits.prod.s …
    ⊢ Eq ((CategoryTheory.FunctorToTypes.binaryProductEquiv F G a) z) ((CategoryTh …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- `coprod F G` is the explicit binary coproduct of type-valued functors `F` and `G`. -/
def coprod : C ⥤ Type w where
  obj a := F.obj a ⊕ G.obj a
  map f x := by
    cases x with
    | inl x => exact .inl (F.map f x)
    | inr x => exact .inr (G.map f x)


/-- The left inclusion of `F` into `coprod F G`. -/
@[simps]
def coprod.inl : F ⟶ coprod F G where
  app _ x := .inl x


/-- The right inclusion of `G` into `coprod F G`. -/
@[simps]
def coprod.inr : G ⟶ coprod F G where
  app _ x := .inr x


/-- Given natural transformations `F₁ ⟶ F` and `F₂ ⟶ F`, construct
a natural transformation `coprod F₁ F₂ ⟶ F`. -/
@[simps]
def coprod.desc {F₁ F₂ : C ⥤ Type w} (τ₁ : F₁ ⟶ F) (τ₂ : F₂ ⟶ F) :
    coprod F₁ F₂ ⟶ F where
  app a x := by
     cases x with
     | inl x => exact τ₁.app a x
     | inr x => exact τ₂.app a x
  naturality _ _ _ := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G F₁ F₂ : CategoryTheory.Functor C (Type w)
      τ₁ : Quiver.Hom F₁ F
      τ₂ : Quiver.Hom F₂ F
      x✝² x✝¹ : C
      x✝ : Quiver.Hom x✝² x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FunctorToTypes.copro …
    -/
    ext x
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G F₁ F₂ : CategoryTheory.Functor C (Type w)
      τ₁ : Quiver.Hom F₁ F
      τ₂ : Quiver.Hom F₂ F
      x✝² x✝¹ : C
      x✝ : Quiver.Hom x✝² x✝¹
      x : (CategoryTheory.FunctorToTypes.coprod F₁ F₂).obj x✝²
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FunctorToTypes.copro …
    -/
    /-
      🎉 no goals
    -/
    cases x with | _ => simp only [coprod, types_comp_apply, FunctorToTypes.naturality]


@[simp]
lemma coprod.desc_inl {F₁ F₂ : C ⥤ Type w} (τ₁ : F₁ ⟶ F) (τ₂ : F₂ ⟶ F) :
    coprod.inl ≫ coprod.desc τ₁ τ₂ = τ₁ := rfl


@[simp]
lemma coprod.desc_inr {F₁ F₂ : C ⥤ Type w} (τ₁ : F₁ ⟶ F) (τ₂ : F₂ ⟶ F) :
    coprod.inr ≫ coprod.desc τ₁ τ₂ = τ₂ := rfl


/-- The binary cofan whose point is `coprod F G`. -/
@[simps!]
def binaryCoproductCocone : BinaryCofan F G :=
  BinaryCofan.mk coprod.inl coprod.inr


/-- `coprod F G` is a colimit cocone. -/
@[simps]
def binaryCoproductColimit : IsColimit (binaryCoproductCocone F G) where
  desc (s : BinaryCofan F G) := coprod.desc s.inl s.inr
  fac _ := fun ⟨j⟩ ↦ WalkingPair.casesOn j rfl rfl
  uniq _ _ h := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C (Type w)
      x✝¹ : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair F G)
      x✝ : Quiver.Hom (CategoryTheory.FunctorToTypes.binaryCoproductCocone F G).pt x …
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq x✝ ((fun s => CategoryTheory.FunctorToTypes.coprod.desc s.inl s.inr) x✝¹)
    -/
    ext _ x
    /-
      case w.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C (Type w)
      x✝² : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair F G)
      x✝¹ : Quiver.Hom (CategoryTheory.FunctorToTypes.binaryCoproductCocone F G).pt  …
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      x✝ : C
      x : (CategoryTheory.FunctorToTypes.binaryCoproductCocone F G).pt.obj x✝
      ⊢ Eq (x✝¹.app x✝ x) (((fun s => CategoryTheory.FunctorToTypes.coprod.desc s.in …
    -/
    /-
      🎉 no goals
    -/
    cases x with | _ => simp [← h ⟨WalkingPair.right⟩, ← h ⟨WalkingPair.left⟩]


/-- `coprod F G` is a binary coproduct for `F` and `G`. -/
def binaryCoproductColimitCocone : Limits.ColimitCocone (pair F G) :=
  ⟨_, binaryCoproductColimit F G⟩


/-- The categorical binary coproduct of type-valued functors is `coprod F G`. -/
noncomputable def binaryCoproductIso : F ⨿ G ≅ coprod F G :=
  colimit.isoColimitCocone (binaryCoproductColimitCocone F G)


@[simp]
lemma inl_comp_binaryCoproductIso_hom :
    Limits.coprod.inl ≫ (binaryCoproductIso F G).hom = coprod.inl := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
  -/
  simp only [binaryCoproductIso]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
  -/
  aesop
  /-
    🎉 no goals
  -/


@[simp]
lemma inl_comp_binaryCoproductIso_hom_apply (a : C) (x : F.obj a) :
    (binaryCoproductIso F G).hom.app a ((Limits.coprod.inl (X := F)).app a x) = .inl x :=
  congr_fun (congr_app (inl_comp_binaryCoproductIso_hom F G) a) x


@[simp]
lemma inr_comp_binaryCoproductIso_hom :
    Limits.coprod.inr ≫ (binaryCoproductIso F G).hom = coprod.inr := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
  -/
  simp [binaryCoproductIso]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C (Type w)
    ⊢ Eq (CategoryTheory.Limits.BinaryCofan.inr (CategoryTheory.FunctorToTypes.bin …
  -/
  aesop
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_comp_binaryCoproductIso_hom_apply (a : C) (x : G.obj a) :
    (binaryCoproductIso F G).hom.app a ((Limits.coprod.inr (X := F)).app a x) = .inr x :=
  congr_fun (congr_app (inr_comp_binaryCoproductIso_hom F G) a) x


@[simp]
lemma inl_comp_binaryCoproductIso_inv :
    coprod.inl ≫ (binaryCoproductIso F G).inv = (Limits.coprod.inl (X := F)) := rfl


@[simp]
lemma inl_comp_binaryCoproductIso_inv_apply (a : C) (x : F.obj a) :
    (binaryCoproductIso F G).inv.app a (.inl x) = (Limits.coprod.inl (X := F)).app a x := rfl


@[simp]
lemma inr_comp_binaryCoproductIso_inv :
    coprod.inr ≫ (binaryCoproductIso F G).inv = (Limits.coprod.inr (X := F)) := rfl


@[simp]
lemma inr_comp_binaryCoproductIso_inv_apply (a : C) (x : G.obj a) :
    (binaryCoproductIso F G).inv.app a (.inr x) = (Limits.coprod.inr (X := F)).app a x := rfl


/-- Construct an element of `(F ⨿ G).obj a` from an element of `F.obj a` -/
noncomputable
abbrev coprodInl {a : C} (x : F.obj a) : (F ⨿ G).obj a :=
  (binaryCoproductIso F G).inv.app a (.inl x)


/-- Construct an element of `(F ⨿ G).obj a` from an element of `G.obj a` -/
noncomputable
abbrev coprodInr {a : C} (x : G.obj a) : (F ⨿ G).obj a :=
  (binaryCoproductIso F G).inv.app a (.inr x)


/-- `(F ⨿ G).obj a` is in bijection with disjoint union of `F.obj a` and `G.obj a`. -/
@[simps]
noncomputable
def binaryCoproductEquiv (a : C) :
    (F ⨿ G).obj a ≃ (F.obj a) ⊕ (G.obj a) where
  toFun z := (binaryCoproductIso F G).hom.app a z
  invFun z := (binaryCoproductIso F G).inv.app a z
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     F G : CategoryTheory.Functor C (Type w)
                     a : C
                     x✝ : (CategoryTheory.Limits.coprod F G).obj a
                     ⊢ Eq ((fun z => (CategoryTheory.FunctorToTypes.binaryCoproductIso F G).inv.app …
                   -/
  left_inv _ := by simp only [hom_inv_id_app_apply]
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      F G : CategoryTheory.Functor C (Type w)
                      a : C
                      x✝ : Sum (F.obj a) (G.obj a)
                      ⊢ Eq ((fun z => (CategoryTheory.FunctorToTypes.binaryCoproductIso F G).hom.app …
                    -/
  right_inv _ := by simp only [inv_hom_id_app_apply]
                    /-
                      🎉 no goals
                    -/


