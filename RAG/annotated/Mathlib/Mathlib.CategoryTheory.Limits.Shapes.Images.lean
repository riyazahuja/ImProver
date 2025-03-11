/-- A factorisation of a morphism `f = e ≫ m`, with `m` monic. -/
structure MonoFactorisation (f : X ⟶ Y) where
  I : C -- Porting note: violates naming conventions but can't think a better replacement
  m : I ⟶ Y
  [m_mono : Mono m]
  e : X ⟶ I
  fac : e ≫ m = f := by aesop_cat


attribute [reassoc (attr := simp)] MonoFactorisation.fac


/-- The obvious factorisation of a monomorphism through itself. -/
def self [Mono f] : MonoFactorisation f where
  I := X
  m := f
  e := 𝟙 X

-- I'm not sure we really need this, but the linter says that an inhabited instance
-- ought to exist...

instance [Mono f] : Inhabited (MonoFactorisation f) := ⟨self f⟩


/-- The morphism `m` in a factorisation `f = e ≫ m` through a monomorphism is uniquely
determined. -/
@[ext (iff := false)]
theorem ext {F F' : MonoFactorisation f} (hI : F.I = F'.I)
    (hm : F.m = eqToHom hI ≫ F'.m) : F = F' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    F F' : CategoryTheory.Limits.MonoFactorisation f
    hI : Eq F.I F'.I
    hm : Eq F.m (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom hI) F' …
    ⊢ Eq F F'
  -/
  cases' F with _ Fm _ _ Ffac; cases' F' with _ Fm' _ _ Ffac'
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I✝¹ : C
    Fm : Quiver.Hom I✝¹ Y
    m_mono✝¹ : CategoryTheory.Mono Fm
    e✝¹ : Quiver.Hom X I✝¹
    Ffac : Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) f
    I✝ : C
    Fm' : Quiver.Hom I✝ Y
    m_mono✝ : CategoryTheory.Mono Fm'
    e✝ : Quiver.Hom X I✝
    Ffac' : Eq (CategoryTheory.CategoryStruct.comp e✝ Fm') f
    hI : Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝¹ Fm e✝¹ Ffac).I (Catego …
    hm : Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝¹ Fm e✝¹ Ffac).m (Catego …
    ⊢ Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝¹ Fm e✝¹ Ffac) (CategoryThe …
  -/
  cases' hI
  /-
    case mk.mk.refl
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I✝ : C
    Fm : Quiver.Hom I✝ Y
    m_mono✝¹ : CategoryTheory.Mono Fm
    e✝¹ : Quiver.Hom X I✝
    Ffac : Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) f
    Fm' : Quiver.Hom I✝ Y
    m_mono✝ : CategoryTheory.Mono Fm'
    e✝ : Quiver.Hom X I✝
    Ffac' : Eq (CategoryTheory.CategoryStruct.comp e✝ Fm') f
    hm : Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝ Fm e✝¹ Ffac).m (Categor …
    ⊢ Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝ Fm e✝¹ Ffac) (CategoryTheo …
  -/
  simp? at hm says simp only [eqToHom_refl, Category.id_comp] at hm
  /-
    case mk.mk.refl
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I✝ : C
    Fm : Quiver.Hom I✝ Y
    m_mono✝¹ : CategoryTheory.Mono Fm
    e✝¹ : Quiver.Hom X I✝
    Ffac : Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) f
    Fm' : Quiver.Hom I✝ Y
    m_mono✝ : CategoryTheory.Mono Fm'
    e✝ : Quiver.Hom X I✝
    Ffac' : Eq (CategoryTheory.CategoryStruct.comp e✝ Fm') f
    hm : Eq Fm Fm'
    ⊢ Eq (CategoryTheory.Limits.MonoFactorisation.mk I✝ Fm e✝¹ Ffac) (CategoryTheo …
  -/
  congr
  /-
    case mk.mk.refl.e_e
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I✝ : C
    Fm : Quiver.Hom I✝ Y
    m_mono✝¹ : CategoryTheory.Mono Fm
    e✝¹ : Quiver.Hom X I✝
    Ffac : Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) f
    Fm' : Quiver.Hom I✝ Y
    m_mono✝ : CategoryTheory.Mono Fm'
    e✝ : Quiver.Hom X I✝
    Ffac' : Eq (CategoryTheory.CategoryStruct.comp e✝ Fm') f
    hm : Eq Fm Fm'
    ⊢ Eq e✝¹ e✝
  -/
  apply (cancel_mono Fm).1
  /-
    case mk.mk.refl.e_e
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    I✝ : C
    Fm : Quiver.Hom I✝ Y
    m_mono✝¹ : CategoryTheory.Mono Fm
    e✝¹ : Quiver.Hom X I✝
    Ffac : Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) f
    Fm' : Quiver.Hom I✝ Y
    m_mono✝ : CategoryTheory.Mono Fm'
    e✝ : Quiver.Hom X I✝
    Ffac' : Eq (CategoryTheory.CategoryStruct.comp e✝ Fm') f
    hm : Eq Fm Fm'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e✝¹ Fm) (CategoryTheory.CategoryStruc …
  -/
  rw [Ffac, hm, Ffac']
  /-
    🎉 no goals
  -/


/-- Any mono factorisation of `f` gives a mono factorisation of `f ≫ g` when `g` is a mono. -/
@[simps]
def compMono (F : MonoFactorisation f) {Y' : C} (g : Y ⟶ Y') [Mono g] :
    MonoFactorisation (f ≫ g) where
  I := F.I
  m := F.m ≫ g
  m_mono := mono_comp _ _
  e := F.e


/-- A mono factorisation of `f ≫ g`, where `g` is an isomorphism,
gives a mono factorisation of `f`. -/
@[simps]
def ofCompIso {Y' : C} {g : Y ⟶ Y'} [IsIso g] (F : MonoFactorisation (f ≫ g)) :
    MonoFactorisation f where
  I := F.I
  m := F.m ≫ inv g
  m_mono := mono_comp _ _
  e := F.e


/-- Any mono factorisation of `f` gives a mono factorisation of `g ≫ f`. -/
@[simps]
def isoComp (F : MonoFactorisation f) {X' : C} (g : X' ⟶ X) : MonoFactorisation (g ≫ f) where
  I := F.I
  m := F.m
  e := g ≫ F.e


/-- A mono factorisation of `g ≫ f`, where `g` is an isomorphism,
gives a mono factorisation of `f`. -/
@[simps]
def ofIsoComp {X' : C} (g : X' ⟶ X) [IsIso g] (F : MonoFactorisation (g ≫ f)) :
    MonoFactorisation f where
  I := F.I
  m := F.m
  e := inv g ≫ F.e


/-- If `f` and `g` are isomorphic arrows, then a mono factorisation of `f`
gives a mono factorisation of `g` -/
@[simps]
def ofArrowIso {f g : Arrow C} (F : MonoFactorisation f.hom) (sq : f ⟶ g) [IsIso sq] :
    MonoFactorisation g.hom where
  I := F.I
  m := F.m ≫ sq.right
  e := inv sq.left ≫ F.e
  m_mono := mono_comp _ _
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X Y : C
              f✝ : Quiver.Hom X Y
              f g : CategoryTheory.Arrow C
              F : CategoryTheory.Limits.MonoFactorisation f.hom
              sq : Quiver.Hom f g
              inst✝ : CategoryTheory.IsIso sq
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
  fac := by simp only [fac_assoc, Arrow.w, IsIso.inv_comp_eq, Category.assoc]
            /-
              🎉 no goals
            -/


/-- Data exhibiting that a given factorisation through a mono is initial. -/
structure IsImage (F : MonoFactorisation f) where
  lift : ∀ F' : MonoFactorisation f, F.I ⟶ F'.I
  lift_fac : ∀ F' : MonoFactorisation f, lift F' ≫ F'.m = F.m := by aesop_cat


attribute [reassoc (attr := simp)] IsImage.lift_fac


@[reassoc (attr := simp)]
theorem fac_lift {F : MonoFactorisation f} (hF : IsImage F) (F' : MonoFactorisation f) :
    F.e ≫ hF.lift F' = F'.e :=
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               f : Quiver.Hom X Y
                               F : CategoryTheory.Limits.MonoFactorisation f
                               hF : CategoryTheory.Limits.IsImage F
                               F' : CategoryTheory.Limits.MonoFactorisation f
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp F …
                             -/
  (cancel_mono F'.m).1 <| by simp
                             /-
                               🎉 no goals
                             -/


/-- The trivial factorisation of a monomorphism satisfies the universal property. -/
@[simps]
def self [Mono f] : IsImage (MonoFactorisation.self f) where lift F' := F'.e


instance [Mono f] : Inhabited (IsImage (MonoFactorisation.self f)) :=
  ⟨self f⟩


/-- Two factorisations through monomorphisms satisfying the universal property
must factor through isomorphic objects. -/
@[simps]
def isoExt {F F' : MonoFactorisation f} (hF : IsImage F) (hF' : IsImage F') :
    F.I ≅ F'.I where
  hom := hF.lift F'
  inv := hF'.lift F
                                        /-
                                          C : Type u
                                          inst✝ : CategoryTheory.Category.{v, u} C
                                          X Y : C
                                          f : Quiver.Hom X Y
                                          F F' : CategoryTheory.Limits.MonoFactorisation f
                                          hF : CategoryTheory.Limits.IsImage F
                                          hF' : CategoryTheory.Limits.IsImage F'
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                        -/
  hom_inv_id := (cancel_mono F.m).1 (by simp)
                                        /-
                                          🎉 no goals
                                        -/
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X Y : C
                                           f : Quiver.Hom X Y
                                           F F' : CategoryTheory.Limits.MonoFactorisation f
                                           hF : CategoryTheory.Limits.IsImage F
                                           hF' : CategoryTheory.Limits.IsImage F'
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                         -/
  inv_hom_id := (cancel_mono F'.m).1 (by simp)
                                         /-
                                           🎉 no goals
                                         -/


                                                              /-
                                                                C : Type u
                                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                F F' : CategoryTheory.Limits.MonoFactorisation f
                                                                hF : CategoryTheory.Limits.IsImage F
                                                                hF' : CategoryTheory.Limits.IsImage F'
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (hF.isoExt hF').hom F'.m) F.m
                                                              -/
theorem isoExt_hom_m : (isoExt hF hF').hom ≫ F'.m = F.m := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                C : Type u
                                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                F F' : CategoryTheory.Limits.MonoFactorisation f
                                                                hF : CategoryTheory.Limits.IsImage F
                                                                hF' : CategoryTheory.Limits.IsImage F'
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (hF.isoExt hF').inv F.m) F'.m
                                                              -/
theorem isoExt_inv_m : (isoExt hF hF').inv ≫ F.m = F'.m := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                C : Type u
                                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                F F' : CategoryTheory.Limits.MonoFactorisation f
                                                                hF : CategoryTheory.Limits.IsImage F
                                                                hF' : CategoryTheory.Limits.IsImage F'
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp F.e (hF.isoExt hF').hom) F'.e
                                                              -/
theorem e_isoExt_hom : F.e ≫ (isoExt hF hF').hom = F'.e := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                C : Type u
                                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                F F' : CategoryTheory.Limits.MonoFactorisation f
                                                                hF : CategoryTheory.Limits.IsImage F
                                                                hF' : CategoryTheory.Limits.IsImage F'
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp F'.e (hF.isoExt hF').inv) F.e
                                                              -/
theorem e_isoExt_inv : F'.e ≫ (isoExt hF hF').inv = F.e := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If `f` and `g` are isomorphic arrows, then a mono factorisation of `f` that is an image
gives a mono factorisation of `g` that is an image -/
@[simps]
def ofArrowIso {f g : Arrow C} {F : MonoFactorisation f.hom} (hF : IsImage F) (sq : f ⟶ g)
    [IsIso sq] : IsImage (F.ofArrowIso sq) where
  lift F' := hF.lift (F'.ofArrowIso (inv sq))
  lift_fac F' := by
    simpa only [MonoFactorisation.ofArrowIso_m, Arrow.inv_right, ← Category.assoc,
      IsIso.comp_inv_eq] using hF.lift_fac (F'.ofArrowIso (inv sq))


/-- Data exhibiting that a morphism `f` has an image. -/
structure ImageFactorisation (f : X ⟶ Y) where
  F : MonoFactorisation f -- Porting note: another violation of the naming convention
  isImage : IsImage F


instance [Mono f] : Inhabited (ImageFactorisation f) :=
  ⟨⟨_, IsImage.self f⟩⟩


/-- If `f` and `g` are isomorphic arrows, then an image factorisation of `f`
gives an image factorisation of `g` -/
@[simps]
def ofArrowIso {f g : Arrow C} (F : ImageFactorisation f.hom) (sq : f ⟶ g) [IsIso sq] :
    ImageFactorisation g.hom where
  F := F.F.ofArrowIso sq
  isImage := F.isImage.ofArrowIso sq


/-- `has_image f` means that there exists an image factorisation of `f`. -/
class HasImage (f : X ⟶ Y) : Prop where mk' ::
  exists_image : Nonempty (ImageFactorisation f)


theorem HasImage.mk {f : X ⟶ Y} (F : ImageFactorisation f) : HasImage f :=
  ⟨Nonempty.intro F⟩


theorem HasImage.of_arrow_iso {f g : Arrow C} [h : HasImage f.hom] (sq : f ⟶ g) [IsIso sq] :
    HasImage g.hom :=
  ⟨⟨h.exists_image.some.ofArrowIso sq⟩⟩


instance (priority := 100) mono_hasImage (f : X ⟶ Y) [Mono f] : HasImage f :=
  HasImage.mk ⟨_, IsImage.self f⟩


/-- Some factorisation of `f` through a monomorphism (selected with choice). -/
def Image.monoFactorisation : MonoFactorisation f :=
  (Classical.choice HasImage.exists_image).F


/-- The witness of the universal property for the chosen factorisation of `f` through
a monomorphism. -/
def Image.isImage : IsImage (Image.monoFactorisation f) :=
  (Classical.choice HasImage.exists_image).isImage


/-- The categorical image of a morphism. -/
def image : C :=
  (Image.monoFactorisation f).I


/-- The inclusion of the image of a morphism into the target. -/
def image.ι : image f ⟶ Y :=
  (Image.monoFactorisation f).m


@[simp]
theorem image.as_ι : (Image.monoFactorisation f).m = image.ι f := rfl


instance : Mono (image.ι f) :=
  (Image.monoFactorisation f).m_mono


/-- The map from the source to the image of a morphism. -/
def factorThruImage : X ⟶ image f :=
  (Image.monoFactorisation f).e


/-- Rewrite in terms of the `factorThruImage` interface. -/
@[simp]
theorem as_factorThruImage : (Image.monoFactorisation f).e = factorThruImage f :=
  rfl


@[reassoc (attr := simp)]
theorem image.fac : factorThruImage f ≫ image.ι f = f :=
  (Image.monoFactorisation f).fac


/-- Any other factorisation of the morphism `f` through a monomorphism receives a map from the
image. -/
def image.lift (F' : MonoFactorisation f) : image f ⟶ F'.I :=
  (Image.isImage f).lift F'


@[reassoc (attr := simp)]
theorem image.lift_fac (F' : MonoFactorisation f) : image.lift F' ≫ F'.m = image.ι f :=
  (Image.isImage f).lift_fac F'


@[reassoc (attr := simp)]
theorem image.fac_lift (F' : MonoFactorisation f) : factorThruImage f ≫ image.lift F' = F'.e :=
  (Image.isImage f).fac_lift F'


@[simp]
theorem image.isImage_lift (F : MonoFactorisation f) : (Image.isImage f).lift F = image.lift F :=
  rfl


@[reassoc (attr := simp)]
theorem IsImage.lift_ι {F : MonoFactorisation f} (hF : IsImage F) :
    hF.lift (Image.monoFactorisation f) ≫ image.ι f = F.m :=
  hF.lift_fac _

-- TODO we could put a category structure on `MonoFactorisation f`,
-- with the morphisms being `g : I ⟶ I'` commuting with the `m`s
-- (they then automatically commute with the `e`s)
-- and show that an `imageOf f` gives an initial object there
-- (uniqueness of the lift comes for free).

instance image.lift_mono (F' : MonoFactorisation f) : Mono (image.lift F') := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    F' : CategoryTheory.Limits.MonoFactorisation f
    ⊢ CategoryTheory.Mono (CategoryTheory.Limits.image.lift F')
  -/
  refine @mono_of_mono _ _ _ _ _ _ F'.m ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    F' : CategoryTheory.Limits.MonoFactorisation f
    ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limi …
  -/
  simpa using MonoFactorisation.m_mono _
  /-
    🎉 no goals
  -/


theorem HasImage.uniq (F' : MonoFactorisation f) (l : image f ⟶ F'.I) (w : l ≫ F'.m = image.ι f) :
    l = image.lift F' :=
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             X Y : C
                             f : Quiver.Hom X Y
                             inst✝ : CategoryTheory.Limits.HasImage f
                             F' : CategoryTheory.Limits.MonoFactorisation f
                             l : Quiver.Hom (CategoryTheory.Limits.image f) F'.I
                             w : Eq (CategoryTheory.CategoryStruct.comp l F'.m) (CategoryTheory.Limits.imag …
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp l F'.m) (CategoryTheory.CategoryStruc …
                           -/
  (cancel_mono F'.m).1 (by simp [w])
                           /-
                             🎉 no goals
                           -/


/-- If `has_image g`, then `has_image (f ≫ g)` when `f` is an isomorphism. -/
instance {X Y Z : C} (f : X ⟶ Y) [IsIso f] (g : Y ⟶ Z) [HasImage g] : HasImage (f ≫ g) where
  exists_image :=
    ⟨{  F :=
          { I := image g
            m := image.ι g
            e := f ≫ factorThruImage g }
        isImage :=
          { lift := fun F' => image.lift
                { I := F'.I
                  m := F'.m
                  e := inv f ≫ F'.e } } }⟩


/-- `HasImages` asserts that every morphism has an image. -/
class HasImages : Prop where
  has_image : ∀ {X Y : C} (f : X ⟶ Y), HasImage f


/-- The image of a monomorphism is isomorphic to the source. -/
def imageMonoIsoSource [Mono f] : image f ≅ X :=
  IsImage.isoExt (Image.isImage f) (IsImage.self f)


@[reassoc (attr := simp)]
theorem imageMonoIsoSource_inv_ι [Mono f] : (imageMonoIsoSource f).inv ≫ image.ι f = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageMonoIsoSo …
  -/
  simp [imageMonoIsoSource]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem imageMonoIsoSource_hom_self [Mono f] : (imageMonoIsoSource f).hom ≫ f = image.ι f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageMonoIsoSo …
  -/
  simp only [← imageMonoIsoSource_inv_ι f]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageMonoIsoSo …
  -/
  rw [← Category.assoc, Iso.hom_inv_id, Category.id_comp]
  /-
    🎉 no goals
  -/

-- This is the proof that `factorThruImage f` is an epimorphism
-- from https://en.wikipedia.org/wiki/Image_%28category_theory%29, which is in turn taken from:
-- Mitchell, Barry (1965), Theory of categories, MR 0202787, p.12, Proposition 10.1

@[ext (iff := false)]
theorem image.ext [HasImage f] {W : C} {g h : image f ⟶ W} [HasLimit (parallelPair g h)]
    (w : factorThruImage f ≫ g = factorThruImage f ≫ h) : g = h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    W : C
    g h : Quiver.Hom (CategoryTheory.Limits.image f) W
    inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruIm …
    ⊢ Eq g h
  -/
  let q := equalizer.ι g h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    W : C
    g h : Quiver.Hom (CategoryTheory.Limits.image f) W
    inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruIm …
    q : Quiver.Hom (CategoryTheory.Limits.equalizer g h) (CategoryTheory.Limits.im …
    ⊢ Eq g h
  -/
  let e' := equalizer.lift _ w
  let F' : MonoFactorisation f :=
    { I := equalizer g h
      m := q ≫ image.ι f
      m_mono := mono_comp _ _
      e := e' }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    W : C
    g h : Quiver.Hom (CategoryTheory.Limits.image f) W
    inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruIm …
    q : Quiver.Hom (CategoryTheory.Limits.equalizer g h) (CategoryTheory.Limits.im …
    e' : Quiver.Hom X (CategoryTheory.Limits.equalizer g h) := CategoryTheory.Limi …
    F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
    ⊢ Eq g h
  -/
  let v := image.lift F'
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    W : C
    g h : Quiver.Hom (CategoryTheory.Limits.image f) W
    inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruIm …
    q : Quiver.Hom (CategoryTheory.Limits.equalizer g h) (CategoryTheory.Limits.im …
    e' : Quiver.Hom X (CategoryTheory.Limits.equalizer g h) := CategoryTheory.Limi …
    F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
    v : Quiver.Hom (CategoryTheory.Limits.image f) F'.I := CategoryTheory.Limits.i …
    ⊢ Eq g h
  -/
  have t₀ : v ≫ q ≫ image.ι f = image.ι f := image.lift_fac F'
  have t : v ≫ q = 𝟙 (image f) :=
    (cancel_mono_id (image.ι f)).1
      (by
        convert t₀ using 1
        rw [Category.assoc])
  -- The proof from wikipedia next proves `q ≫ v = 𝟙 _`,
  -- and concludes that `equalizer g h ≅ image f`,
  -- but this isn't necessary.
  calc
    g = 𝟙 (image f) ≫ g := by rw [Category.id_comp]
    _ = v ≫ q ≫ g := by rw [← t, Category.assoc]
    _ = v ≫ q ≫ h := by rw [equalizer.condition g h]
    _ = 𝟙 (image f) ≫ h := by rw [← Category.assoc, t]
    _ = h := by rw [Category.id_comp]


instance [HasImage f] [∀ {Z : C} (g h : image f ⟶ Z), HasLimit (parallelPair g h)] :
    Epi (factorThruImage f) :=
  ⟨fun _ _ w => image.ext f w⟩


theorem epi_image_of_epi {X Y : C} (f : X ⟶ Y) [HasImage f] [E : Epi f] : Epi (image.ι f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    E : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
  -/
  rw [← image.fac f] at E
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    E : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
  -/
  exact epi_of_epi (factorThruImage f) (image.ι f)
  /-
    🎉 no goals
  -/


theorem epi_of_epi_image {X Y : C} (f : X ⟶ Y) [HasImage f] [Epi (image.ι f)]
    [Epi (factorThruImage f)] : Epi f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasImage f
    inst✝¹ : CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
    inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
    ⊢ CategoryTheory.Epi f
  -/
  rw [← image.fac f]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasImage f
    inst✝¹ : CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
    inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  apply epi_comp
  /-
    🎉 no goals
  -/


/-- An equation between morphisms gives a comparison map between the images
(which momentarily we prove is an iso).
-/
def image.eqToHom (h : f = f') : image f ⟶ image f' :=
  image.lift
    { I := image f'
      m := image.ι f'
      e := factorThruImage f'
                /-
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  X Y : C
                  f f' : Quiver.Hom X Y
                  inst✝¹ : CategoryTheory.Limits.HasImage f
                  inst✝ : CategoryTheory.Limits.HasImage f'
                  h : Eq f f'
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                -/
      fac := by rw [h]; simp only [image.fac]}
                        /-
                          🎉 no goals
                        -/


instance (h : f = f') : IsIso (image.eqToHom h) :=
  ⟨⟨image.eqToHom h.symm,
      ⟨(cancel_mono (image.ι f)).1 (by
          -- Porting note: added let's for used to be a simp [image.eqToHom]
          let F : MonoFactorisation f' :=
            ⟨image f, image.ι f, factorThruImage f, (by aesop_cat)⟩
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F : CategoryTheory.Limits.MonoFactorisation f' := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp [image.eqToHom]
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F : CategoryTheory.Limits.MonoFactorisation f' := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [Category.id_comp,Category.assoc,image.lift_fac F]
          let F' : MonoFactorisation f :=
            ⟨image f', image.ι f', factorThruImage f', (by aesop_cat)⟩
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F : CategoryTheory.Limits.MonoFactorisation f' := CategoryTheory.Limits.MonoFa …
            F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
          -/
          rw [image.lift_fac F'] ),
          /-
            🎉 no goals
          -/
        (cancel_mono (image.ι f')).1 (by
          -- Porting note: added let's for used to be a simp [image.eqToHom]
          let F' : MonoFactorisation f :=
            ⟨image f', image.ι f', factorThruImage f', (by aesop_cat)⟩
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp [image.eqToHom]
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [Category.id_comp,Category.assoc,image.lift_fac F']
          let F : MonoFactorisation f' :=
            ⟨image f, image.ι f, factorThruImage f, (by aesop_cat)⟩
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            X Y : C
            f f' : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.Limits.HasImage f'
            h : Eq f f'
            F' : CategoryTheory.Limits.MonoFactorisation f := CategoryTheory.Limits.MonoFa …
            F : CategoryTheory.Limits.MonoFactorisation f' := CategoryTheory.Limits.MonoFa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
          -/
          rw [image.lift_fac F])⟩⟩⟩
          /-
            🎉 no goals
          -/


/-- An equation between morphisms gives an isomorphism between the images. -/
def image.eqToIso (h : f = f') : image f ≅ image f' :=
  asIso (image.eqToHom h)


/-- As long as the category has equalizers,
the image inclusion maps commute with `image.eqToIso`.
-/
theorem image.eq_fac [HasEqualizers C] (h : f = f') :
    image.ι f = (image.eqToIso h).hom ≫ image.ι f' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f f' : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasImage f
    inst✝¹ : CategoryTheory.Limits.HasImage f'
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    h : Eq f f'
    ⊢ Eq (CategoryTheory.Limits.image.ι f) (CategoryTheory.CategoryStruct.comp (Ca …
  -/
  apply image.ext
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f f' : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasImage f
    inst✝¹ : CategoryTheory.Limits.HasImage f'
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    h : Eq f f'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  dsimp [asIso,image.eqToIso, image.eqToHom]
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f f' : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasImage f
    inst✝¹ : CategoryTheory.Limits.HasImage f'
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    h : Eq f f'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  rw [image.lift_fac] -- Porting note: simp did not fire with this it seems
  /-
    🎉 no goals
  -/


/-- The comparison map `image (f ≫ g) ⟶ image g`. -/
def image.preComp [HasImage g] [HasImage (f ≫ g)] : image (f ≫ g) ⟶ image g :=
  image.lift
    { I := image g
      m := image.ι g
      e := f ≫ factorThruImage g }


@[reassoc (attr := simp)]
theorem image.preComp_ι [HasImage g] [HasImage (f ≫ g)] :
    image.preComp f g ≫ image.ι g = image.ι (f ≫ g) := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Y Z
        inst✝¹ : CategoryTheory.Limits.HasImage g
        inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.preComp  …
      -/
      dsimp [image.preComp]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Y Z
        inst✝¹ : CategoryTheory.Limits.HasImage g
        inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
      -/
      rw [image.lift_fac] -- Porting note: also here, see image.eq_fac
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem image.factorThruImage_preComp [HasImage g] [HasImage (f ≫ g)] :
                                                                              /-
                                                                                C : Type u
                                                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                                                X Y : C
                                                                                f : Quiver.Hom X Y
                                                                                Z : C
                                                                                g : Quiver.Hom Y Z
                                                                                inst✝¹ : CategoryTheory.Limits.HasImage g
                                                                                inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
                                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                                                                              -/
    factorThruImage (f ≫ g) ≫ image.preComp f g = f ≫ factorThruImage g := by simp [image.preComp]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- `image.preComp f g` is a monomorphism.
-/
instance image.preComp_mono [HasImage g] [HasImage (f ≫ g)] : Mono (image.preComp f g) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.Mono (CategoryTheory.Limits.image.preComp f g)
  -/
  refine @mono_of_mono _ _ _ _ _ _ (image.ι g) ?_
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limi …
  -/
  simp only [image.preComp_ι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
    ⊢ CategoryTheory.Mono (CategoryTheory.Limits.image.ι (CategoryTheory.CategoryS …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The two step comparison map
  `image (f ≫ (g ≫ h)) ⟶ image (g ≫ h) ⟶ image h`
agrees with the one step comparison map
  `image (f ≫ (g ≫ h)) ≅ image ((f ≫ g) ≫ h) ⟶ image h`.
 -/
theorem image.preComp_comp {W : C} (h : Z ⟶ W) [HasImage (g ≫ h)] [HasImage (f ≫ g ≫ h)]
    [HasImage h] [HasImage ((f ≫ g) ≫ h)] :
    image.preComp f (g ≫ h) ≫ image.preComp g h =
      image.eqToHom (Category.assoc f g h).symm ≫ image.preComp (f ≫ g) h := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    W : C
    h : Quiver.Hom Z W
    inst✝³ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp g h)
    inst✝² : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f  …
    inst✝¹ : CategoryTheory.Limits.HasImage h
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp (Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.preComp  …
  -/
  apply (cancel_mono (image.ι h)).1
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    W : C
    h : Quiver.Hom Z W
    inst✝³ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp g h)
    inst✝² : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f  …
    inst✝¹ : CategoryTheory.Limits.HasImage h
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp (Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [image.preComp, image.eqToHom]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    W : C
    h : Quiver.Hom Z W
    inst✝³ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp g h)
    inst✝² : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f  …
    inst✝¹ : CategoryTheory.Limits.HasImage h
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp (Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  repeat (rw [Category.assoc,image.lift_fac])
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    W : C
    h : Quiver.Hom Z W
    inst✝³ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp g h)
    inst✝² : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f  …
    inst✝¹ : CategoryTheory.Limits.HasImage h
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp (Ca …
    ⊢ Eq (CategoryTheory.Limits.image.ι (CategoryTheory.CategoryStruct.comp f (Cat …
  -/
  rw [image.lift_fac,image.lift_fac]
  /-
    🎉 no goals
  -/


/-- `image.preComp f g` is an epimorphism when `f` is an epimorphism
(we need `C` to have equalizers to prove this).
-/
instance image.preComp_epi_of_epi [HasImage g] [HasImage (f ≫ g)] [Epi f] :
    Epi (image.preComp f g) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasImage g
    inst✝¹ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.image.preComp f g)
  -/
  apply @epi_of_epi_fac _ _ _ _ _ _ _ _ ?_ (image.factorThruImage_preComp _ _)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasImage g
    inst✝¹ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp f g)
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Lim …
  -/
  exact epi_comp _ _
  /-
    🎉 no goals
  -/


instance hasImage_iso_comp [IsIso f] [HasImage g] : HasImage (f ≫ g) :=
  HasImage.mk
    { F := (Image.monoFactorisation g).isoComp f
      isImage := { lift := fun F' => image.lift (F'.ofIsoComp f)
                   lift_fac := fun F' => by
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      X Y : C
                      f : Quiver.Hom X Y
                      Z : C
                      g : Quiver.Hom Y Z
                      inst✝² : CategoryTheory.Limits.HasEqualizers C
                      inst✝¹ : CategoryTheory.IsIso f
                      inst✝ : CategoryTheory.Limits.HasImage g
                      F' : CategoryTheory.Limits.MonoFactorisation (CategoryTheory.CategoryStruct.co …
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun F' => CategoryTheory.Limits.ima …
                    -/
                    dsimp
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      X Y : C
                      f : Quiver.Hom X Y
                      Z : C
                      g : Quiver.Hom Y Z
                      inst✝² : CategoryTheory.Limits.HasEqualizers C
                      inst✝¹ : CategoryTheory.IsIso f
                      inst✝ : CategoryTheory.Limits.HasImage g
                      F' : CategoryTheory.Limits.MonoFactorisation (CategoryTheory.CategoryStruct.co …
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
                    -/
                    have : (MonoFactorisation.ofIsoComp f F').m = F'.m := rfl
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      X Y : C
                      f : Quiver.Hom X Y
                      Z : C
                      g : Quiver.Hom Y Z
                      inst✝² : CategoryTheory.Limits.HasEqualizers C
                      inst✝¹ : CategoryTheory.IsIso f
                      inst✝ : CategoryTheory.Limits.HasImage g
                      F' : CategoryTheory.Limits.MonoFactorisation (CategoryTheory.CategoryStruct.co …
                      this : Eq (CategoryTheory.Limits.MonoFactorisation.ofIsoComp f F').m F'.m
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
                    -/
                    rw [← this,image.lift_fac (MonoFactorisation.ofIsoComp f F')] } }
                    /-
                      🎉 no goals
                    -/


/-- `image.preComp f g` is an isomorphism when `f` is an isomorphism
(we need `C` to have equalizers to prove this).
-/
instance image.isIso_precomp_iso (f : X ⟶ Y) [IsIso f] [HasImage g] : IsIso (image.preComp f g) :=
  ⟨⟨image.lift
        { I := image (f ≫ g)
          m := image.ι (f ≫ g)
          e := inv f ≫ factorThruImage (f ≫ g) },
      ⟨by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : Quiver.Hom X Y
          Z : C
          g : Quiver.Hom Y Z
          inst✝² : CategoryTheory.Limits.HasEqualizers C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.IsIso f
          inst✝ : CategoryTheory.Limits.HasImage g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.preComp  …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : Quiver.Hom X Y
          Z : C
          g : Quiver.Hom Y Z
          inst✝² : CategoryTheory.Limits.HasEqualizers C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.IsIso f
          inst✝ : CategoryTheory.Limits.HasImage g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
        -/
        simp [image.preComp], by
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : Quiver.Hom X Y
          Z : C
          g : Quiver.Hom Y Z
          inst✝² : CategoryTheory.Limits.HasEqualizers C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.IsIso f
          inst✝ : CategoryTheory.Limits.HasImage g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : Quiver.Hom X Y
          Z : C
          g : Quiver.Hom Y Z
          inst✝² : CategoryTheory.Limits.HasEqualizers C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.IsIso f
          inst✝ : CategoryTheory.Limits.HasImage g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
        -/
        simp [image.preComp]⟩⟩⟩
        /-
          🎉 no goals
        -/

-- Note that in general we don't have the other comparison map you might expect
-- `image f ⟶ image (f ≫ g)`.

instance hasImage_comp_iso [HasImage f] [IsIso g] : HasImage (f ≫ g) :=
  HasImage.mk
    { F := (Image.monoFactorisation f).compMono g
      isImage :=
      { lift := fun F' => image.lift F'.ofCompIso
        lift_fac := fun F' => by
          rw [← Category.comp_id (image.lift (MonoFactorisation.ofCompIso F') ≫ F'.m),
            ← IsIso.inv_hom_id g,← Category.assoc]
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : Quiver.Hom X Y
            Z : C
            g : Quiver.Hom Y Z
            inst✝² : CategoryTheory.Limits.HasEqualizers C
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.IsIso g
            F' : CategoryTheory.Limits.MonoFactorisation (CategoryTheory.CategoryStruct.co …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          refine congrArg (· ≫ g) ?_
          have : (image.lift (MonoFactorisation.ofCompIso F') ≫ F'.m) ≫ inv g =
            image.lift (MonoFactorisation.ofCompIso F') ≫
            ((MonoFactorisation.ofCompIso F').m) := by
              simp only [MonoFactorisation.ofCompIso_I, Category.assoc,
                MonoFactorisation.ofCompIso_m]
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : Quiver.Hom X Y
            Z : C
            g : Quiver.Hom Y Z
            inst✝² : CategoryTheory.Limits.HasEqualizers C
            inst✝¹ : CategoryTheory.Limits.HasImage f
            inst✝ : CategoryTheory.IsIso g
            F' : CategoryTheory.Limits.MonoFactorisation (CategoryTheory.CategoryStruct.co …
            this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [this, image.lift_fac (MonoFactorisation.ofCompIso F'),image.as_ι] }}
          /-
            🎉 no goals
          -/


/-- Postcomposing by an isomorphism induces an isomorphism on the image. -/
def image.compIso [HasImage f] [IsIso g] : image f ≅ image (f ≫ g) where
  hom := image.lift (Image.monoFactorisation (f ≫ g)).ofCompIso
  inv := image.lift ((Image.monoFactorisation f).compMono g)


@[reassoc (attr := simp)]
theorem image.compIso_hom_comp_image_ι [HasImage f] [IsIso g] :
    (image.compIso f g).hom ≫ image.ι (f ≫ g) = image.ι f ≫ g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.compIso  …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  simp [image.compIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem image.compIso_inv_comp_image_ι [HasImage f] [IsIso g] :
    (image.compIso f g).inv ≫ image.ι f = image.ι (f ≫ g) ≫ inv g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.compIso  …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  simp [image.compIso]
  /-
    🎉 no goals
  -/


instance {X Y : C} (f : X ⟶ Y) [HasImage f] : HasImage (Arrow.mk f).hom :=
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       X Y : C
                       f : Quiver.Hom X Y
                       inst✝ : CategoryTheory.Limits.HasImage f
                       ⊢ CategoryTheory.Limits.HasImage f
                     -/
  show HasImage f by infer_instance
                     /-
                       🎉 no goals
                     -/


/-- An image map is a morphism `image f → image g` fitting into a commutative square and satisfying
    the obvious commutativity conditions. -/
structure ImageMap {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g) where
  map : image f.hom ⟶ image g.hom
  map_ι : map ≫ image.ι g.hom = image.ι f.hom ≫ sq.right := by aesop


instance inhabitedImageMap {f : Arrow C} [HasImage f.hom] : Inhabited (ImageMap (𝟙 f)) :=
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              f : CategoryTheory.Arrow C
              inst✝ : CategoryTheory.Limits.HasImage f.hom
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
            -/
  ⟨⟨𝟙 _, by aesop⟩⟩
            /-
              🎉 no goals
            -/


attribute [reassoc (attr := simp)] ImageMap.map_ι


@[reassoc (attr := simp)]
theorem ImageMap.factor_map {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g)
    (m : ImageMap sq) : factorThruImage f.hom ≫ m.map = sq.left ≫ factorThruImage g.hom :=
                                        /-
                                          C : Type u
                                          inst✝² : CategoryTheory.Category.{v, u} C
                                          f g : CategoryTheory.Arrow C
                                          inst✝¹ : CategoryTheory.Limits.HasImage f.hom
                                          inst✝ : CategoryTheory.Limits.HasImage g.hom
                                          sq : Quiver.Hom f g
                                          m : CategoryTheory.Limits.ImageMap sq
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                        -/
  (cancel_mono (image.ι g.hom)).1 <| by simp
                                        /-
                                          🎉 no goals
                                        -/


/-- To give an image map for a commutative square with `f` at the top and `g` at the bottom, it
    suffices to give a map between any mono factorisation of `f` and any image factorisation of
    `g`. -/
def ImageMap.transport {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g)
    (F : MonoFactorisation f.hom) {F' : MonoFactorisation g.hom} (hF' : IsImage F')
    {map : F.I ⟶ F'.I} (map_ι : map ≫ F'.m = F.m ≫ sq.right) : ImageMap sq where
  map := image.lift F ≫ map ≫ hF'.lift (Image.monoFactorisation g.hom)
              /-
                C : Type u
                inst✝² : CategoryTheory.Category.{v, u} C
                f g : CategoryTheory.Arrow C
                inst✝¹ : CategoryTheory.Limits.HasImage f.hom
                inst✝ : CategoryTheory.Limits.HasImage g.hom
                sq : Quiver.Hom f g
                F : CategoryTheory.Limits.MonoFactorisation f.hom
                F' : CategoryTheory.Limits.MonoFactorisation g.hom
                hF' : CategoryTheory.Limits.IsImage F'
                map : Quiver.Hom F.I F'.I
                map_ι : Eq (CategoryTheory.CategoryStruct.comp map F'.m) (CategoryTheory.Categ …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
  map_ι := by simp [map_ι]
              /-
                🎉 no goals
              -/


/-- `HasImageMap sq` means that there is an `ImageMap` for the square `sq`. -/
class HasImageMap {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g) : Prop where
mk' ::
  has_image_map : Nonempty (ImageMap sq)


theorem HasImageMap.mk {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] {sq : f ⟶ g}
    (m : ImageMap sq) : HasImageMap sq :=
  ⟨Nonempty.intro m⟩


theorem HasImageMap.transport {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g)
    (F : MonoFactorisation f.hom) {F' : MonoFactorisation g.hom} (hF' : IsImage F')
    (map : F.I ⟶ F'.I) (map_ι : map ≫ F'.m = F.m ≫ sq.right) : HasImageMap sq :=
  HasImageMap.mk <| ImageMap.transport sq F hF' map_ι


/-- Obtain an `ImageMap` from a `HasImageMap` instance. -/
def HasImageMap.imageMap {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] (sq : f ⟶ g)
    [HasImageMap sq] : ImageMap sq :=
  Classical.choice <| @HasImageMap.has_image_map _ _ _ _ _ _ sq _

-- see Note [lower instance priority]

instance (priority := 100) hasImageMapOfIsIso {f g : Arrow C} [HasImage f.hom] [HasImage g.hom]
    (sq : f ⟶ g) [IsIso sq] : HasImageMap sq :=
  HasImageMap.mk
    { map := image.lift ((Image.monoFactorisation g.hom).ofArrowIso (inv sq))
      map_ι := by
        erw [← cancel_mono (inv sq).right, Category.assoc, ← MonoFactorisation.ofArrowIso_m,
          image.lift_fac, Category.assoc, ← Comma.comp_right, IsIso.hom_inv_id, Comma.id_right,
          Category.comp_id] }


instance HasImageMap.comp {f g h : Arrow C} [HasImage f.hom] [HasImage g.hom] [HasImage h.hom]
    (sq1 : f ⟶ g) (sq2 : g ⟶ h) [HasImageMap sq1] [HasImageMap sq2] : HasImageMap (sq1 ≫ sq2) :=
  HasImageMap.mk
    { map := (HasImageMap.imageMap sq1).map ≫ (HasImageMap.imageMap sq2).map
      map_ι := by
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          f g h : CategoryTheory.Arrow C
          inst✝⁴ : CategoryTheory.Limits.HasImage f.hom
          inst✝³ : CategoryTheory.Limits.HasImage g.hom
          inst✝² : CategoryTheory.Limits.HasImage h.hom
          sq1 : Quiver.Hom f g
          sq2 : Quiver.Hom g h
          inst✝¹ : CategoryTheory.Limits.HasImageMap sq1
          inst✝ : CategoryTheory.Limits.HasImageMap sq2
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Category.assoc,ImageMap.map_ι, ImageMap.map_ι_assoc, Comma.comp_right] }
        /-
          🎉 no goals
        -/


attribute [local ext] ImageMap

/- Porting note: ImageMap.mk.injEq has LHS simplify to True due to the next instance
We make a replacement -/

theorem ImageMap.map_uniq_aux {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] {sq : f ⟶ g}
    (map : image f.hom ⟶ image g.hom)
    (map_ι : map ≫ image.ι g.hom = image.ι f.hom ≫ sq.right := by aesop_cat)
    (map' : image f.hom ⟶ image g.hom)
    (map_ι' : map' ≫ image.ι g.hom = image.ι f.hom ≫ sq.right) : (map = map') := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    f g : CategoryTheory.Arrow C
    inst✝¹ : CategoryTheory.Limits.HasImage f.hom
    inst✝ : CategoryTheory.Limits.HasImage g.hom
    sq : Quiver.Hom f g
    map : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.im …
    map_ι : autoParam (Eq (CategoryTheory.CategoryStruct.comp map (CategoryTheory. …
    map' : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.i …
    map_ι' : Eq (CategoryTheory.CategoryStruct.comp map' (CategoryTheory.Limits.im …
    ⊢ Eq map map'
  -/
  have : map ≫ image.ι g.hom = map' ≫ image.ι g.hom := by rw [map_ι,map_ι']
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    f g : CategoryTheory.Arrow C
    inst✝¹ : CategoryTheory.Limits.HasImage f.hom
    inst✝ : CategoryTheory.Limits.HasImage g.hom
    sq : Quiver.Hom f g
    map : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.im …
    map_ι : autoParam (Eq (CategoryTheory.CategoryStruct.comp map (CategoryTheory. …
    map' : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.i …
    map_ι' : Eq (CategoryTheory.CategoryStruct.comp map' (CategoryTheory.Limits.im …
    this : Eq (CategoryTheory.CategoryStruct.comp map (CategoryTheory.Limits.image …
    ⊢ Eq map map'
  -/
  apply (cancel_mono (image.ι g.hom)).1 this
  /-
    🎉 no goals
  -/

-- Porting note: added to get variant on ImageMap.mk.injEq below

theorem ImageMap.map_uniq {f g : Arrow C} [HasImage f.hom] [HasImage g.hom]
    {sq : f ⟶ g} (F G : ImageMap sq) : F.map = G.map := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    f g : CategoryTheory.Arrow C
    inst✝¹ : CategoryTheory.Limits.HasImage f.hom
    inst✝ : CategoryTheory.Limits.HasImage g.hom
    sq : Quiver.Hom f g
    F G : CategoryTheory.Limits.ImageMap sq
    ⊢ Eq F.map G.map
  -/
  apply ImageMap.map_uniq_aux _ F.map_ι _ G.map_ι
  /-
    🎉 no goals
  -/


@[simp]
theorem ImageMap.mk.injEq' {f g : Arrow C} [HasImage f.hom] [HasImage g.hom] {sq : f ⟶ g}
    (map : image f.hom ⟶ image g.hom)
    (map_ι : map ≫ image.ι g.hom = image.ι f.hom ≫ sq.right := by aesop_cat)
    (map' : image f.hom ⟶ image g.hom)
    (map_ι' : map' ≫ image.ι g.hom = image.ι f.hom ≫ sq.right) : (map = map') = True := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    f g : CategoryTheory.Arrow C
    inst✝¹ : CategoryTheory.Limits.HasImage f.hom
    inst✝ : CategoryTheory.Limits.HasImage g.hom
    sq : Quiver.Hom f g
    map : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.im …
    map_ι : autoParam (Eq (CategoryTheory.CategoryStruct.comp map (CategoryTheory. …
    map' : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.i …
    map_ι' : Eq (CategoryTheory.CategoryStruct.comp map' (CategoryTheory.Limits.im …
    ⊢ Eq (Eq map map') True
  -/
  simp only [Functor.id_obj, eq_iff_iff, iff_true]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    f g : CategoryTheory.Arrow C
    inst✝¹ : CategoryTheory.Limits.HasImage f.hom
    inst✝ : CategoryTheory.Limits.HasImage g.hom
    sq : Quiver.Hom f g
    map : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.im …
    map_ι : autoParam (Eq (CategoryTheory.CategoryStruct.comp map (CategoryTheory. …
    map' : Quiver.Hom (CategoryTheory.Limits.image f.hom) (CategoryTheory.Limits.i …
    map_ι' : Eq (CategoryTheory.CategoryStruct.comp map' (CategoryTheory.Limits.im …
    ⊢ Eq map map'
  -/
  apply ImageMap.map_uniq_aux _ map_ι _ map_ι'
  /-
    🎉 no goals
  -/


instance : Subsingleton (ImageMap sq) :=
  Subsingleton.intro fun a b =>
    ImageMap.ext <| ImageMap.map_uniq a b


/-- The map on images induced by a commutative square. -/
abbrev image.map : image f.hom ⟶ image g.hom :=
  (HasImageMap.imageMap sq).map


theorem image.factor_map :
                                                                                 /-
                                                                                   C : Type u
                                                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                   f g : CategoryTheory.Arrow C
                                                                                   inst✝² : CategoryTheory.Limits.HasImage f.hom
                                                                                   inst✝¹ : CategoryTheory.Limits.HasImage g.hom
                                                                                   sq : Quiver.Hom f g
                                                                                   inst✝ : CategoryTheory.Limits.HasImageMap sq
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                                                                                 -/
    factorThruImage f.hom ≫ image.map sq = sq.left ≫ factorThruImage g.hom := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


                                                                                    /-
                                                                                      C : Type u
                                                                                      inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                      f g : CategoryTheory.Arrow C
                                                                                      inst✝² : CategoryTheory.Limits.HasImage f.hom
                                                                                      inst✝¹ : CategoryTheory.Limits.HasImage g.hom
                                                                                      sq : Quiver.Hom f g
                                                                                      inst✝ : CategoryTheory.Limits.HasImageMap sq
                                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.map sq)  …
                                                                                    -/
theorem image.map_ι : image.map sq ≫ image.ι g.hom = image.ι f.hom ≫ sq.right := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem image.map_homMk'_ι {X Y P Q : C} {k : X ⟶ Y} [HasImage k] {l : P ⟶ Q} [HasImage l]
    {m : X ⟶ P} {n : Y ⟶ Q} (w : m ≫ l = k ≫ n) [HasImageMap (Arrow.homMk' w)] :
    image.map (Arrow.homMk' w) ≫ image.ι l = image.ι k ≫ n :=
  image.map_ι _


/-- Image maps for composable commutative squares induce an image map in the composite square. -/
def imageMapComp : ImageMap (sq ≫ sq') where map := image.map sq ≫ image.map sq'


@[simp]
theorem image.map_comp [HasImageMap (sq ≫ sq')] :
    image.map (sq ≫ sq') = image.map sq ≫ image.map sq' :=
  show (HasImageMap.imageMap (sq ≫ sq')).map = (imageMapComp sq sq').map by
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      f g : CategoryTheory.Arrow C
      inst✝⁵ : CategoryTheory.Limits.HasImage f.hom
      inst✝⁴ : CategoryTheory.Limits.HasImage g.hom
      sq : Quiver.Hom f g
      inst✝³ : CategoryTheory.Limits.HasImageMap sq
      h : CategoryTheory.Arrow C
      inst✝² : CategoryTheory.Limits.HasImage h.hom
      sq' : Quiver.Hom g h
      inst✝¹ : CategoryTheory.Limits.HasImageMap sq'
      inst✝ : CategoryTheory.Limits.HasImageMap (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq (CategoryTheory.Limits.HasImageMap.imageMap (CategoryTheory.CategoryStruc …
    -/
    congr; simp only [eq_iff_true_of_subsingleton]
           /-
             🎉 no goals
           -/


/-- The identity `image f ⟶ image f` fits into the commutative square represented by the identity
    morphism `𝟙 f` in the arrow category. -/
def imageMapId : ImageMap (𝟙 f) where map := 𝟙 (image f.hom)


@[simp]
theorem image.map_id [HasImageMap (𝟙 f)] : image.map (𝟙 f) = 𝟙 (image f.hom) :=
  show (HasImageMap.imageMap (𝟙 f)).map = (imageMapId f).map by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      f : CategoryTheory.Arrow C
      inst✝¹ : CategoryTheory.Limits.HasImage f.hom
      inst✝ : CategoryTheory.Limits.HasImageMap (CategoryTheory.CategoryStruct.id f)
      ⊢ Eq (CategoryTheory.Limits.HasImageMap.imageMap (CategoryTheory.CategoryStruc …
    -/
    congr; simp only [eq_iff_true_of_subsingleton]
           /-
             🎉 no goals
           -/


/-- If a category `has_image_maps`, then all commutative squares induce morphisms on images. -/
class HasImageMaps : Prop where
  has_image_map : ∀ {f g : Arrow C} (st : f ⟶ g), HasImageMap st


/-- The functor from the arrow category of `C` to `C` itself that maps a morphism to its image
    and a commutative square to the induced morphism on images. -/
@[simps]
def im : Arrow C ⥤ C where
  obj f := image f.hom
  map st := image.map st


/-- A strong epi-mono factorisation is a decomposition `f = e ≫ m` with `e` a strong epimorphism
    and `m` a monomorphism. -/
structure StrongEpiMonoFactorisation {X Y : C} (f : X ⟶ Y) extends MonoFactorisation f where
  [e_strong_epi : StrongEpi e]


/-- Satisfying the inhabited linter -/
instance strongEpiMonoFactorisationInhabited {X Y : C} (f : X ⟶ Y) [StrongEpi f] :
    Inhabited (StrongEpiMonoFactorisation f) :=
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y : C
                     f : Quiver.Hom X Y
                     inst✝ : CategoryTheory.StrongEpi f
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                   -/
  ⟨⟨⟨Y, 𝟙 Y, f, by simp⟩⟩⟩
                   /-
                     🎉 no goals
                   -/


/-- A mono factorisation coming from a strong epi-mono factorisation always has the universal
    property of the image. -/
def StrongEpiMonoFactorisation.toMonoIsImage {X Y : C} {f : X ⟶ Y}
    (F : StrongEpiMonoFactorisation f) : IsImage F.toMonoFactorisation where
  lift G :=
                                              /-
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                F : CategoryTheory.Limits.StrongEpiMonoFactorisation f
                                                G : CategoryTheory.Limits.MonoFactorisation f
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp G.e G.m) (CategoryTheory.CategoryStru …
                                              -/
    (CommSq.mk (show G.e ≫ G.m = F.e ≫ F.m by rw [F.toMonoFactorisation.fac, G.fac])).lift
                                              /-
                                                🎉 no goals
                                              -/


/-- A category has strong epi-mono factorisations if every morphism admits a strong epi-mono
    factorisation. -/
class HasStrongEpiMonoFactorisations : Prop where mk' ::
  has_fac : ∀ {X Y : C} (f : X ⟶ Y), Nonempty (StrongEpiMonoFactorisation f)


theorem HasStrongEpiMonoFactorisations.mk
    (d : ∀ {X Y : C} (f : X ⟶ Y), StrongEpiMonoFactorisation f) :
    HasStrongEpiMonoFactorisations C :=
  ⟨fun f => Nonempty.intro <| d f⟩


instance (priority := 100) hasImages_of_hasStrongEpiMonoFactorisations
    [HasStrongEpiMonoFactorisations C] : HasImages C where
  has_image f :=
    let F' := Classical.choice (HasStrongEpiMonoFactorisations.has_fac f)
    HasImage.mk
      { F := F'.toMonoFactorisation
        isImage := F'.toMonoIsImage }


/-- A category has strong epi images if it has all images and `factorThruImage f` is a strong
    epimorphism for all `f`. -/
class HasStrongEpiImages : Prop where
  strong_factorThruImage : ∀ {X Y : C} (f : X ⟶ Y), StrongEpi (factorThruImage f)


/-- If there is a single strong epi-mono factorisation of `f`, then every image factorisation is a
    strong epi-mono factorisation. -/
theorem strongEpi_of_strongEpiMonoFactorisation {X Y : C} {f : X ⟶ Y}
    (F : StrongEpiMonoFactorisation f) {F' : MonoFactorisation f} (hF' : IsImage F') :
    StrongEpi F'.e := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    F : CategoryTheory.Limits.StrongEpiMonoFactorisation f
    F' : CategoryTheory.Limits.MonoFactorisation f
    hF' : CategoryTheory.Limits.IsImage F'
    ⊢ CategoryTheory.StrongEpi F'.e
  -/
  rw [← IsImage.e_isoExt_hom F.toMonoIsImage hF']
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    F : CategoryTheory.Limits.StrongEpiMonoFactorisation f
    F' : CategoryTheory.Limits.MonoFactorisation f
    hF' : CategoryTheory.Limits.IsImage F'
    ⊢ CategoryTheory.StrongEpi (CategoryTheory.CategoryStruct.comp F.e (F.toMonoIs …
  -/
  apply strongEpi_comp
  /-
    🎉 no goals
  -/


theorem strongEpi_factorThruImage_of_strongEpiMonoFactorisation {X Y : C} {f : X ⟶ Y} [HasImage f]
    (F : StrongEpiMonoFactorisation f) : StrongEpi (factorThruImage f) :=
  strongEpi_of_strongEpiMonoFactorisation F <| Image.isImage f


/-- If we constructed our images from strong epi-mono factorisations, then these images are
    strong epi images. -/
instance (priority := 100) hasStrongEpiImages_of_hasStrongEpiMonoFactorisations
    [HasStrongEpiMonoFactorisations C] : HasStrongEpiImages C where
  strong_factorThruImage f :=
    strongEpi_factorThruImage_of_strongEpiMonoFactorisation <|
      Classical.choice <| HasStrongEpiMonoFactorisations.has_fac f


/-- A category with strong epi images has image maps. -/
instance (priority := 100) hasImageMapsOfHasStrongEpiImages [HasStrongEpiImages C] :
    HasImageMaps C where
  has_image_map {f} {g} st :=
    HasImageMap.mk
      { map :=
          (CommSq.mk
              (show
                (st.left ≫ factorThruImage g.hom) ≫ image.ι g.hom =
                  factorThruImage f.hom ≫ image.ι f.hom ≫ st.right
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Limits.HasImages C
                     inst✝ : CategoryTheory.Limits.HasStrongEpiImages C
                     f g : CategoryTheory.Arrow C
                     st : Quiver.Hom f g
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
                   -/
                by simp)).lift }
                   /-
                     🎉 no goals
                   -/


/-- If a category has images, equalizers and pullbacks, then images are automatically strong epi
    images. -/
instance (priority := 100) hasStrongEpiImages_of_hasPullbacks_of_hasEqualizers [HasPullbacks C]
    [HasEqualizers C] : HasStrongEpiImages C where
  strong_factorThruImage f :=
    StrongEpi.mk' fun {A} {B} h h_mono x y sq =>
      CommSq.HasLift.mk'
        { l :=
            image.lift
                { I := pullback h y
                  m := pullback.snd h y ≫ image.ι f
                  m_mono := mono_comp _ _
                  e := pullback.lift _ _ sq.w } ≫
              pullback.fst h y
                         /-
                           C : Type u
                           inst✝³ : CategoryTheory.Category.{v, u} C
                           inst✝² : CategoryTheory.Limits.HasImages C
                           inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                           inst✝ : CategoryTheory.Limits.HasEqualizers C
                           X✝ Y✝ : C
                           f : Quiver.Hom X✝ Y✝
                           A B : C
                           h : Quiver.Hom A B
                           h_mono : CategoryTheory.Mono h
                           x : Quiver.Hom X✝ A
                           y : Quiver.Hom (CategoryTheory.Limits.image f) B
                           sq : CategoryTheory.CommSq x (CategoryTheory.Limits.factorThruImage f) h y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                         -/
          fac_left := by simp only [image.fac_lift_assoc, pullback.lift_fst]
                         /-
                           🎉 no goals
                         -/
          fac_right := by
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              inst✝² : CategoryTheory.Limits.HasImages C
              inst✝¹ : CategoryTheory.Limits.HasPullbacks C
              inst✝ : CategoryTheory.Limits.HasEqualizers C
              X✝ Y✝ : C
              f : Quiver.Hom X✝ Y✝
              A B : C
              h : Quiver.Hom A B
              h_mono : CategoryTheory.Mono h
              x : Quiver.Hom X✝ A
              y : Quiver.Hom (CategoryTheory.Limits.image f) B
              sq : CategoryTheory.CommSq x (CategoryTheory.Limits.factorThruImage f) h y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
            apply image.ext
            /-
              case w
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              inst✝² : CategoryTheory.Limits.HasImages C
              inst✝¹ : CategoryTheory.Limits.HasPullbacks C
              inst✝ : CategoryTheory.Limits.HasEqualizers C
              X✝ Y✝ : C
              f : Quiver.Hom X✝ Y✝
              A B : C
              h : Quiver.Hom A B
              h_mono : CategoryTheory.Mono h
              x : Quiver.Hom X✝ A
              y : Quiver.Hom (CategoryTheory.Limits.image f) B
              sq : CategoryTheory.CommSq x (CategoryTheory.Limits.factorThruImage f) h y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
            -/
            simp only [sq.w, Category.assoc, image.fac_lift_assoc, pullback.lift_fst_assoc] }
            /-
              🎉 no goals
            -/


/--
If `C` has strong epi mono factorisations, then the image is unique up to isomorphism, in that if
`f` factors as a strong epi followed by a mono, this factorisation is essentially the image
factorisation.
-/
def image.isoStrongEpiMono {I' : C} (e : X ⟶ I') (m : I' ⟶ Y) (comm : e ≫ m = f) [StrongEpi e]
    [Mono m] : I' ≅ image f :=
  let F : StrongEpiMonoFactorisation f := { I := I', m := m, e := e}
  IsImage.isoExt F.toMonoIsImage <| Image.isImage f


@[simp]
theorem image.isoStrongEpiMono_hom_comp_ι {I' : C} (e : X ⟶ I') (m : I' ⟶ Y) (comm : e ≫ m = f)
    [StrongEpi e] [Mono m] : (image.isoStrongEpiMono e m comm).hom ≫ image.ι f = m := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
    X Y : C
    f : Quiver.Hom X Y
    I' : C
    e : Quiver.Hom X I'
    m : Quiver.Hom I' Y
    comm : Eq (CategoryTheory.CategoryStruct.comp e m) f
    inst✝¹ : CategoryTheory.StrongEpi e
    inst✝ : CategoryTheory.Mono m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.isoStron …
  -/
  dsimp [isoStrongEpiMono]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
    X Y : C
    f : Quiver.Hom X Y
    I' : C
    e : Quiver.Hom X I'
    m : Quiver.Hom I' Y
    comm : Eq (CategoryTheory.CategoryStruct.comp e m) f
    inst✝¹ : CategoryTheory.StrongEpi e
    inst✝ : CategoryTheory.Mono m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.StrongEpiMono …
  -/
  apply IsImage.lift_fac
  /-
    🎉 no goals
  -/


@[simp]
theorem image.isoStrongEpiMono_inv_comp_mono {I' : C} (e : X ⟶ I') (m : I' ⟶ Y) (comm : e ≫ m = f)
    [StrongEpi e] [Mono m] : (image.isoStrongEpiMono e m comm).inv ≫ m = image.ι f :=
  image.lift_fac _


/-- A category with strong epi mono factorisations admits functorial epi/mono factorizations. -/
noncomputable def functorialEpiMonoFactorizationData :
    FunctorialFactorizationData (epimorphisms C) (monomorphisms C) where
  Z := im
  i := { app := fun f => factorThruImage f.hom }
  p := { app := fun f => image.ι f.hom }
  hi _ := epimorphisms.infer_property _
  hp _ := monomorphisms.infer_property _


theorem hasStrongEpiMonoFactorisations_imp_of_isEquivalence (F : C ⥤ D) [IsEquivalence F]
    [h : HasStrongEpiMonoFactorisations C] : HasStrongEpiMonoFactorisations D :=
  ⟨fun {X} {Y} f => by
    let em : StrongEpiMonoFactorisation (F.inv.map f) :=
      (HasStrongEpiMonoFactorisations.has_fac (F.inv.map f)).some
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      h : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
      X Y : D
      f : Quiver.Hom X Y
      em : CategoryTheory.Limits.StrongEpiMonoFactorisation (F.inv.map f) := ⋯.some
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    haveI : Mono (F.map em.m ≫ F.asEquivalence.counitIso.hom.app Y) := mono_comp _ _
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      h : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
      X Y : D
      f : Quiver.Hom X Y
      em : CategoryTheory.Limits.StrongEpiMonoFactorisation (F.inv.map f) := ⋯.some
      this : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (F.map em.m) (F …
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    haveI : StrongEpi (F.asEquivalence.counitIso.inv.app X ≫ F.map em.e) := strongEpi_comp _ _
    exact
      Nonempty.intro
        { I := F.obj em.I
          e := F.asEquivalence.counitIso.inv.app X ≫ F.map em.e
          m := F.map em.m ≫ F.asEquivalence.counitIso.hom.app Y
          fac := by
            simp only [asEquivalence_functor, Category.assoc, ← F.map_comp_assoc,
              MonoFactorisation.fac, fun_inv_map, id_obj, Iso.inv_hom_id_app, Category.comp_id,
              Iso.inv_hom_id_app_assoc] }⟩


