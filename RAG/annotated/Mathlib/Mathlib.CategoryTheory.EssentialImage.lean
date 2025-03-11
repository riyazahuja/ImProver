/-- The essential image of a functor `F` consists of those objects in the target category which are
isomorphic to an object in the image of the function `F.obj`. In other words, this is the closure
under isomorphism of the function `F.obj`.
This is the "non-evil" way of describing the image of a functor.
-/
def essImage (F : C ⥤ D) : Set D := fun Y => ∃ X : C, Nonempty (F.obj X ≅ Y)


/-- Get the witnessing object that `Y` is in the subcategory given by `F`. -/
def essImage.witness {Y : D} (h : Y ∈ F.essImage) : C :=
  h.choose


/-- Extract the isomorphism between `F.obj h.witness` and `Y` itself. -/
-- Porting note: in the next, the dot notation `h.witness` no longer works
def essImage.getIso {Y : D} (h : Y ∈ F.essImage) : F.obj (essImage.witness h) ≅ Y :=
  Classical.choice h.choose_spec


/-- Being in the essential image is a "hygienic" property: it is preserved under isomorphism. -/
theorem essImage.ofIso {Y Y' : D} (h : Y ≅ Y') (hY : Y ∈ essImage F) : Y' ∈ essImage F :=
  hY.imp fun _ => Nonempty.map (· ≪≫ h)


/-- If `Y` is in the essential image of `F` then it is in the essential image of `F'` as long as
`F ≅ F'`.
-/
theorem essImage.ofNatIso {F' : C ⥤ D} (h : F ≅ F') {Y : D} (hY : Y ∈ essImage F) :
    Y ∈ essImage F' :=
  hY.imp fun X => Nonempty.map fun t => h.symm.app X ≪≫ t


/-- Isomorphic functors have equal essential images. -/
theorem essImage_eq_of_natIso {F' : C ⥤ D} (h : F ≅ F') : essImage F = essImage F' :=
  funext fun _ => propext ⟨essImage.ofNatIso h, essImage.ofNatIso h.symm⟩


/-- An object in the image is in the essential image. -/
theorem obj_mem_essImage (F : D ⥤ C) (Y : D) : F.obj Y ∈ essImage F :=
  ⟨Y, ⟨Iso.refl _⟩⟩


/-- The essential image of a functor, interpreted as a full subcategory of the target category. -/
-- Porting note: no hasNonEmptyInstance linter yet
def EssImageSubcategory (F : C ⥤ D) :=
  FullSubcategory F.essImage

-- Porting note: `deriving Category` is not able to derive this instance

instance : Category (EssImageSubcategory F) :=
  (inferInstance : Category.{v₂} (FullSubcategory _))


/-- The essential image as a subcategory has a fully faithful inclusion into the target category. -/
@[simps!]
def essImageInclusion (F : C ⥤ D) : F.EssImageSubcategory ⥤ D :=
  fullSubcategoryInclusion _

-- Porting note: `deriving Full` is not able to derive this instance

instance : Full (essImageInclusion F) :=
  (inferInstance : Full (fullSubcategoryInclusion _))

-- Porting note: `deriving Faithful` is not able to derive this instance

instance : Faithful (essImageInclusion F) :=
  (inferInstance : Faithful (fullSubcategoryInclusion _))


lemma essImage_ext (F : C ⥤ D) {X Y : F.EssImageSubcategory} (f g : X ⟶ Y)
    (h : F.essImageInclusion.map f = F.essImageInclusion.map g) : f = g := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : F.EssImageSubcategory
    f g : Quiver.Hom X Y
    h : Eq (F.essImageInclusion.map f) (F.essImageInclusion.map g)
    ⊢ Eq f g
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/--
Given a functor `F : C ⥤ D`, we have an (essentially surjective) functor from `C` to the essential
image of `F`.
-/
@[simps!]
def toEssImage (F : C ⥤ D) : C ⥤ F.EssImageSubcategory :=
  FullSubcategory.lift _ F (obj_mem_essImage _)


/-- The functor `F` factorises through its essential image, where the first functor is essentially
surjective and the second is fully faithful.
-/
@[simps!]
def toEssImageCompEssentialImageInclusion (F : C ⥤ D) : F.toEssImage ⋙ F.essImageInclusion ≅ F :=
  FullSubcategory.lift_comp_inclusion _ _ _


/-- A functor `F : C ⥤ D` is essentially surjective if every object of `D` is in the essential
image of `F`. In other words, for every `Y : D`, there is some `X : C` with `F.obj X ≅ Y`.

See <https://stacks.math.columbia.edu/tag/001C>.
-/
class EssSurj (F : C ⥤ D) : Prop where
  /-- All the objects of the target category are in the essential image. -/
  mem_essImage (Y : D) : Y ∈ F.essImage


instance EssSurj.toEssImage : EssSurj F.toEssImage where
  mem_essImage := fun ⟨_, hY⟩ =>
    ⟨_, ⟨⟨_, _, hY.getIso.hom_inv_id, hY.getIso.inv_hom_id⟩⟩⟩


theorem essSurj_of_surj (h : Function.Surjective F.obj) : EssSurj F where
  mem_essImage Y := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      h : Function.Surjective F.obj
      Y : D
      ⊢ Membership.mem F.essImage Y
    -/
    obtain ⟨X, rfl⟩ := h Y
    /-
      case intro
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      h : Function.Surjective F.obj
      X : C
      ⊢ Membership.mem F.essImage (F.obj X)
    -/
    apply obj_mem_essImage
    /-
      🎉 no goals
    -/


/-- Given an essentially surjective functor, we can find a preimage for every object `Y` in the
    codomain. Applying the functor to this preimage will yield an object isomorphic to `Y`, see
    `obj_obj_preimage_iso`. -/
def objPreimage (Y : D) : C :=
  essImage.witness (@EssSurj.mem_essImage _ _ _ _ F _ Y)


/-- Applying an essentially surjective functor to a preimage of `Y` yields an object that is
    isomorphic to `Y`. -/
def objObjPreimageIso (Y : D) : F.obj (F.objPreimage Y) ≅ Y :=
  Functor.essImage.getIso _


/-- The induced functor of a faithful functor is faithful. -/
instance Faithful.toEssImage (F : C ⥤ D) [Faithful F] : Faithful F.toEssImage :=
  Faithful.of_comp_iso F.toEssImageCompEssentialImageInclusion


/-- The induced functor of a full functor is full. -/
instance Full.toEssImage (F : C ⥤ D) [Full F] : Full F.toEssImage :=
  Full.of_comp_faithful_iso F.toEssImageCompEssentialImageInclusion


instance instEssSurjId : EssSurj (𝟭 C) where
  mem_essImage Y := ⟨Y, ⟨Iso.refl _⟩⟩


lemma essSurj_of_iso {F G : C ⥤ D} [EssSurj F] (α : F ≅ G) : EssSurj G where
  mem_essImage Y := Functor.essImage.ofNatIso α (EssSurj.mem_essImage Y)


instance essSurj_comp (F : C ⥤ D) (G : D ⥤ E) [F.EssSurj] [G.EssSurj] :
    (F ⋙ G).EssSurj where
  mem_essImage Z := ⟨_, ⟨G.mapIso (F.objObjPreimageIso _) ≪≫ G.objObjPreimageIso Z⟩⟩


lemma essSurj_of_comp_fully_faithful (F : C ⥤ D) (G : D ⥤ E) [(F ⋙ G).EssSurj]
    [G.Faithful] [G.Full] : F.EssSurj where
  mem_essImage X := ⟨_, ⟨G.preimageIso ((F ⋙ G).objObjPreimageIso (G.obj X))⟩⟩


@[deprecated (since := "2024-04-06")] alias EssSurj := Functor.EssSurj

@[deprecated (since := "2024-04-06")] alias Iso.map_essSurj := Functor.essSurj_of_iso


