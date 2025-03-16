/-- A functor `F` is additive provided `F.map` is an additive homomorphism. -/
class Functor.Additive {C D : Type*} [Category C] [Category D] [Preadditive C] [Preadditive D]
  (F : C ⥤ D) : Prop where
  /-- the addition of two morphisms is mapped to the sum of their images -/
  map_add : ∀ {X Y : C} {f g : X ⟶ Y}, F.map (f + g) = F.map f + F.map g := by aesop_cat


@[simp]
theorem map_add {X Y : C} {f g : X ⟶ Y} : F.map (f + g) = F.map f + F.map g :=
  Functor.Additive.map_add

-- Porting note: it was originally @[simps (config := .asFn)]

/-- `F.mapAddHom` is an additive homomorphism whose underlying function is `F.map`. -/
@[simps!]
def mapAddHom {X Y : C} : (X ⟶ Y) →+ (F.obj X ⟶ F.obj Y) :=
  AddMonoidHom.mk' (fun f => F.map f) fun _ _ => F.map_add


theorem coe_mapAddHom {X Y : C} : ⇑(F.mapAddHom : (X ⟶ Y) →+ _) = F.map :=
  rfl


instance (priority := 100) preservesZeroMorphisms_of_additive : PreservesZeroMorphisms F where
  map_zero _ _ := F.mapAddHom.map_zero


instance : Additive (𝟭 C) where


instance {E : Type*} [Category E] [Preadditive E] (G : D ⥤ E) [Functor.Additive G] :
    Additive (F ⋙ G) where


instance {J : Type*} [Category J] (j : J) : ((evaluation J C).obj j).Additive where


@[simp]
theorem map_neg {X Y : C} {f : X ⟶ Y} : F.map (-f) = -F.map f :=
  (F.mapAddHom : (X ⟶ Y) →+ (F.obj X ⟶ F.obj Y)).map_neg _


@[simp]
theorem map_sub {X Y : C} {f g : X ⟶ Y} : F.map (f - g) = F.map f - F.map g :=
  (F.mapAddHom : (X ⟶ Y) →+ (F.obj X ⟶ F.obj Y)).map_sub _ _


theorem map_nsmul {X Y : C} {f : X ⟶ Y} {n : ℕ} : F.map (n • f) = n • F.map f :=
  (F.mapAddHom : (X ⟶ Y) →+ (F.obj X ⟶ F.obj Y)).map_nsmul _ _

-- You can alternatively just use `Functor.map_smul` here, with an explicit `(r : ℤ)` argument.

theorem map_zsmul {X Y : C} {f : X ⟶ Y} {r : ℤ} : F.map (r • f) = r • F.map f :=
  (F.mapAddHom : (X ⟶ Y) →+ (F.obj X ⟶ F.obj Y)).map_zsmul _ _


@[simp]
nonrec theorem map_sum {X Y : C} {α : Type*} (f : α → (X ⟶ Y)) (s : Finset α) :
    F.map (∑ a ∈ s, f a) = ∑ a ∈ s, F.map (f a) :=
  map_sum F.mapAddHom f s


lemma additive_of_iso {G : C ⥤ D} (e : F ≅ G) : G.Additive := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    ⊢ G.Additive
  -/
  constructor
  /-
    case map_add
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    ⊢ autoParam (∀ {X Y : C} {f g : Quiver.Hom X Y}, Eq (G.map (HAdd.hAdd f g)) (H …
  -/
  intro X Y f g
  simp only [← NatIso.naturality_1 e (f + g), map_add, Preadditive.add_comp,
    NatTrans.naturality, Preadditive.comp_add, Iso.inv_hom_id_app_assoc]


lemma additive_of_full_essSurj_comp [Full F] [EssSurj F] (G : D ⥤ E)
    [(F ⋙ G).Additive] : G.Additive where
  map_add {X Y f g} := by
    obtain ⟨f', hf'⟩ := F.map_surjective ((F.objObjPreimageIso X).hom ≫ f ≫
      (F.objObjPreimageIso Y).inv)
    obtain ⟨g', hg'⟩ := F.map_surjective ((F.objObjPreimageIso X).hom ≫ g ≫
      (F.objObjPreimageIso Y).inv)
    simp only [← cancel_mono (G.map (F.objObjPreimageIso Y).inv),
      ← cancel_epi (G.map (F.objObjPreimageIso X).hom),
      Preadditive.add_comp, Preadditive.comp_add, ← Functor.map_comp]
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁷ : CategoryTheory.Category.{u_6, u_3} E
      inst✝⁶ : CategoryTheory.Preadditive C
      inst✝⁵ : CategoryTheory.Preadditive D
      inst✝⁴ : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝³ : F.Additive
      inst✝² : F.Full
      inst✝¹ : F.EssSurj
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).Additive
      X Y : D
      f g : Quiver.Hom X Y
      f' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hf' : Eq (F.map f') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      g' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hg' : Eq (F.map g') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      ⊢ Eq (G.map (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIs …
    -/
    erw [← hf', ← hg', ← (F ⋙ G).map_add]
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁷ : CategoryTheory.Category.{u_6, u_3} E
      inst✝⁶ : CategoryTheory.Preadditive C
      inst✝⁵ : CategoryTheory.Preadditive D
      inst✝⁴ : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝³ : F.Additive
      inst✝² : F.Full
      inst✝¹ : F.EssSurj
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).Additive
      X Y : D
      f g : Quiver.Hom X Y
      f' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hf' : Eq (F.map f') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      g' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hg' : Eq (F.map g') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      ⊢ Eq (G.map (HAdd.hAdd (F.map f') (F.map g'))) ((F.comp G).map (HAdd.hAdd f' g …
    -/
    dsimp
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁷ : CategoryTheory.Category.{u_6, u_3} E
      inst✝⁶ : CategoryTheory.Preadditive C
      inst✝⁵ : CategoryTheory.Preadditive D
      inst✝⁴ : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝³ : F.Additive
      inst✝² : F.Full
      inst✝¹ : F.EssSurj
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).Additive
      X Y : D
      f g : Quiver.Hom X Y
      f' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hf' : Eq (F.map f') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      g' : Quiver.Hom (F.objPreimage X) (F.objPreimage Y)
      hg' : Eq (F.map g') (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso X …
      ⊢ Eq (G.map (HAdd.hAdd (F.map f') (F.map g'))) (G.map (F.map (HAdd.hAdd f' g')))
    -/
    rw [F.map_add]
    /-
      🎉 no goals
    -/


lemma additive_of_comp_faithful
    (F : C ⥤ D) (G : D ⥤ E) [G.Additive] [(F ⋙ G).Additive] [Faithful G] :
    F.Additive where
  map_add {_ _ f₁ f₂} := G.map_injective (by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Category.{u_6, u_3} E
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝² : G.Additive
      inst✝¹ : (F.comp G).Additive
      inst✝ : G.Faithful
      x✝¹ x✝ : C
      f₁ f₂ : Quiver.Hom x✝¹ x✝
      ⊢ Eq (G.map (F.map (HAdd.hAdd f₁ f₂))) (G.map (HAdd.hAdd (F.map f₁) (F.map f₂)))
    -/
    rw [← Functor.comp_map, G.map_add, (F ⋙ G).map_add, Functor.comp_map, Functor.comp_map])
    /-
      🎉 no goals
    -/


open ZeroObject Limits in
include F in
lemma hasZeroObject_of_additive [HasZeroObject C] :
    HasZeroObject D where
                       /-
                         C : Type u_1
                         D : Type u_2
                         inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
                         inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
                         inst✝³ : CategoryTheory.Preadditive C
                         inst✝² : CategoryTheory.Preadditive D
                         F : CategoryTheory.Functor C D
                         inst✝¹ : F.Additive
                         inst✝ : CategoryTheory.Limits.HasZeroObject C
                         ⊢ CategoryTheory.Limits.IsZero (F.obj 0)
                       -/
  zero := ⟨F.obj 0, by rw [IsZero.iff_id_eq_zero, ← F.map_id, id_zero, F.map_zero]⟩
                       /-
                         🎉 no goals
                       -/


instance inducedFunctor_additive : Functor.Additive (inducedFunctor F) where


instance fullSubcategoryInclusion_additive {C : Type*} [Category C] [Preadditive C]
    (Z : C → Prop) : (fullSubcategoryInclusion Z).Additive where


instance (priority := 100) preservesFiniteBiproductsOfAdditive [Additive F] :
    PreservesFiniteBiproducts F where
  preserves :=
    { preserves :=
      { preserves := fun hb =>
          ⟨isBilimitOfTotal _ (by
            /-
              C : Type u₁
              D : Type u₂
              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.Preadditive D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.Additive
              J✝ : Type
              inst✝ : Fintype J✝
              f✝ : J✝ → C
              b✝ : CategoryTheory.Limits.Bicone f✝
              hb : b✝.IsBilimit
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp ((F.mapBicon …
            -/
            simp_rw [F.mapBicone_π, F.mapBicone_ι, ← F.map_comp]
            /-
              C : Type u₁
              D : Type u₂
              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.Preadditive D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.Additive
              J✝ : Type
              inst✝ : Fintype J✝
              f✝ : J✝ → C
              b✝ : CategoryTheory.Limits.Bicone f✝
              hb : b✝.IsBilimit
              ⊢ Eq (Finset.univ.sum fun x => F.map (CategoryTheory.CategoryStruct.comp (b✝.π …
            -/
            erw [← F.map_sum, ← F.map_id, IsBilimit.total hb])⟩ } }
            /-
              🎉 no goals
            -/


theorem additive_of_preservesBinaryBiproducts [HasBinaryBiproducts C] [PreservesZeroMorphisms F]
    [PreservesBinaryBiproducts F] : Additive F where
  map_add {X Y f g} := by
    rw [biprod.add_eq_lift_id_desc, F.map_comp, ← biprod.lift_mapBiprod,
      ← biprod.mapBiprod_hom_desc, Category.assoc, Iso.inv_hom_id_assoc, F.map_id,
      biprod.add_eq_lift_id_desc]


lemma additive_of_preserves_binary_products
    [HasBinaryProducts C] [PreservesLimitsOfShape (Discrete WalkingPair) F]
    [F.PreservesZeroMorphisms] : F.Additive := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasBinaryProducts C
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : F.PreservesZeroMorphisms
    ⊢ F.Additive
  -/
  have : HasBinaryBiproducts C := HasBinaryBiproducts.of_hasBinaryProducts
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasBinaryProducts C
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : F.PreservesZeroMorphisms
    this : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ F.Additive
  -/
  have := preservesBinaryBiproducts_of_preservesBinaryProducts F
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasBinaryProducts C
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : F.PreservesZeroMorphisms
    this✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    this : CategoryTheory.Limits.PreservesBinaryBiproducts F
    ⊢ F.Additive
  -/
  exact Functor.additive_of_preservesBinaryBiproducts F
  /-
    🎉 no goals
  -/


instance inverse_additive (e : C ≌ D) [e.functor.Additive] : e.inverse.Additive where
                                               /-
                                                 C : Type u_1
                                                 D : Type u_2
                                                 inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
                                                 inst✝³ : CategoryTheory.Category.{u_4, u_2} D
                                                 inst✝² : CategoryTheory.Preadditive C
                                                 inst✝¹ : CategoryTheory.Preadditive D
                                                 e : CategoryTheory.Equivalence C D
                                                 inst✝ : e.functor.Additive
                                                 f g : D
                                                 f✝ g✝ : Quiver.Hom f g
                                                 ⊢ Eq (e.functor.map (e.inverse.map (HAdd.hAdd f✝ g✝))) (e.functor.map (HAdd.hA …
                                               -/
  map_add {f g} := e.functor.map_injective (by simp)
                                               /-
                                                 🎉 no goals
                                               -/


/-- Bundled additive functors. -/
def AdditiveFunctor :=
  FullSubcategory fun F : C ⥤ D => F.Additive


instance : Category (AdditiveFunctor C D) :=
  FullSubcategory.category _


/-- the category of additive functors is denoted `C ⥤+ D` -/
infixr:26 " ⥤+ " => AdditiveFunctor


instance : Preadditive (C ⥤+ D) :=
  Preadditive.inducedCategory _


/-- An additive functor is in particular a functor. -/
def AdditiveFunctor.forget : (C ⥤+ D) ⥤ C ⥤ D :=
  fullSubcategoryInclusion _


instance : (AdditiveFunctor.forget C D).Full :=
  FullSubcategory.full _


/-- Turn an additive functor into an object of the category `AdditiveFunctor C D`. -/
def AdditiveFunctor.of (F : C ⥤ D) [F.Additive] : C ⥤+ D :=
  ⟨F, inferInstance⟩


@[simp]
theorem AdditiveFunctor.of_fst (F : C ⥤ D) [F.Additive] : (AdditiveFunctor.of F).1 = F :=
  rfl


@[simp]
theorem AdditiveFunctor.forget_obj (F : C ⥤+ D) : (AdditiveFunctor.forget C D).obj F = F.1 :=
  rfl


theorem AdditiveFunctor.forget_obj_of (F : C ⥤ D) [F.Additive] :
    (AdditiveFunctor.forget C D).obj (AdditiveFunctor.of F) = F :=
  rfl


@[simp]
theorem AdditiveFunctor.forget_map (F G : C ⥤+ D) (α : F ⟶ G) :
    (AdditiveFunctor.forget C D).map α = α :=
  rfl


instance : Functor.Additive (AdditiveFunctor.forget C D) where map_add := rfl


instance (F : C ⥤+ D) : Functor.Additive F.1 :=
  F.2


/-- Turn a left exact functor into an additive functor. -/
def AdditiveFunctor.ofLeftExact : (C ⥤ₗ D) ⥤ C ⥤+ D :=
  FullSubcategory.map fun F ⟨_⟩ =>
    Functor.additive_of_preservesBinaryBiproducts F


instance : (AdditiveFunctor.ofLeftExact C D).Full := FullSubcategory.full_map _

instance : (AdditiveFunctor.ofLeftExact C D).Faithful := FullSubcategory.faithful_map _


/-- Turn a right exact functor into an additive functor. -/
def AdditiveFunctor.ofRightExact : (C ⥤ᵣ D) ⥤ C ⥤+ D :=
  FullSubcategory.map fun F ⟨_⟩ =>
    Functor.additive_of_preservesBinaryBiproducts F


instance : (AdditiveFunctor.ofRightExact C D).Full := FullSubcategory.full_map _

instance : (AdditiveFunctor.ofRightExact C D).Faithful := FullSubcategory.faithful_map _


/-- Turn an exact functor into an additive functor. -/
def AdditiveFunctor.ofExact : (C ⥤ₑ D) ⥤ C ⥤+ D :=
  FullSubcategory.map fun F ⟨⟨_⟩, _⟩ =>
    Functor.additive_of_preservesBinaryBiproducts F


instance : (AdditiveFunctor.ofExact C D).Full := FullSubcategory.full_map _

instance : (AdditiveFunctor.ofExact C D).Faithful := FullSubcategory.faithful_map _


@[simp]
theorem AdditiveFunctor.ofLeftExact_obj_fst (F : C ⥤ₗ D) :
    ((AdditiveFunctor.ofLeftExact C D).obj F).obj = F.obj :=
  rfl


@[simp]
theorem AdditiveFunctor.ofRightExact_obj_fst (F : C ⥤ᵣ D) :
    ((AdditiveFunctor.ofRightExact C D).obj F).obj = F.obj :=
  rfl


@[simp]
theorem AdditiveFunctor.ofExact_obj_fst (F : C ⥤ₑ D) :
    ((AdditiveFunctor.ofExact C D).obj F).obj = F.obj :=
  rfl


@[simp]
theorem AdditiveFunctor.ofLeftExact_map {F G : C ⥤ₗ D} (α : F ⟶ G) :
    (AdditiveFunctor.ofLeftExact C D).map α = α :=
  rfl


@[simp]
theorem AdditiveFunctor.ofRightExact_map {F G : C ⥤ᵣ D} (α : F ⟶ G) :
    (AdditiveFunctor.ofRightExact C D).map α = α :=
  rfl


@[simp]
theorem AdditiveFunctor.ofExact_map {F G : C ⥤ₑ D} (α : F ⟶ G) :
    (AdditiveFunctor.ofExact C D).map α = α :=
  rfl


