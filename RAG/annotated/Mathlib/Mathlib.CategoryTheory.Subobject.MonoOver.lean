/-- The category of monomorphisms into `X` as a full subcategory of the over category.
This isn't skeletal, so it's not a partial order.

Later we define `Subobject X` as the quotient of this by isomorphisms.
-/
def MonoOver (X : C) :=
  FullSubcategory fun f : Over X => Mono f.hom


instance (X : C) : Category (MonoOver X) :=
  FullSubcategory.category _


/-- Construct a `MonoOver X`. -/
@[simps]
def mk' {X A : C} (f : A ⟶ X) [hf : Mono f] : MonoOver X where
  obj := Over.mk f
  property := hf


/-- The inclusion from monomorphisms over X to morphisms over X. -/
def forget (X : C) : MonoOver X ⥤ Over X :=
  fullSubcategoryInclusion _


instance : CoeOut (MonoOver X) C where coe Y := Y.obj.left


@[simp]
theorem forget_obj_left {f} : ((forget X).obj f).left = (f : C) :=
  rfl


@[simp]
theorem mk'_coe' {X A : C} (f : A ⟶ X) [Mono f] : (mk' f : C) = A :=
  rfl


/-- Convenience notation for the underlying arrow of a monomorphism over X. -/
abbrev arrow (f : MonoOver X) : (f : C) ⟶ X :=
  ((forget X).obj f).hom


@[simp]
theorem mk'_arrow {X A : C} (f : A ⟶ X) [Mono f] : (mk' f).arrow = f :=
  rfl


@[simp]
theorem forget_obj_hom {f} : ((forget X).obj f).hom = f.arrow :=
  rfl


/-- The forget functor `MonoOver X ⥤ Over X` is fully faithful. -/
def fullyFaithfulForget (X : C) : (forget X).FullyFaithful :=
  fullyFaithfulFullSubcategoryInclusion _


instance : (forget X).Full :=
  FullSubcategory.full _


instance : (forget X).Faithful :=
  FullSubcategory.faithful _


instance mono (f : MonoOver X) : Mono f.arrow :=
  f.property


/-- The category of monomorphisms over X is a thin category,
which makes defining its skeleton easy. -/
instance isThin {X : C} : Quiver.IsThin (MonoOver X) := fun f g =>
  ⟨by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      f g : CategoryTheory.MonoOver X
      ⊢ ∀ (a b : Quiver.Hom f g), Eq a b
    -/
    intro h₁ h₂
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      f g : CategoryTheory.MonoOver X
      h₁ h₂ : Quiver.Hom f g
      ⊢ Eq h₁ h₂
    -/
    apply Over.OverMorphism.ext
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      f g : CategoryTheory.MonoOver X
      h₁ h₂ : Quiver.Hom f g
      ⊢ Eq h₁.left h₂.left
    -/
    erw [← cancel_mono g.arrow, Over.w h₁, Over.w h₂]⟩
    /-
      🎉 no goals
    -/


@[reassoc]
theorem w {f g : MonoOver X} (k : f ⟶ g) : k.left ≫ g.arrow = f.arrow :=
  Over.w _


/-- Convenience constructor for a morphism in monomorphisms over `X`. -/
abbrev homMk {f g : MonoOver X} (h : f.obj.left ⟶ g.obj.left)
    (w : h ≫ g.arrow = f.arrow := by aesop_cat) : f ⟶ g :=
  Over.homMk h w


/-- Convenience constructor for an isomorphism in monomorphisms over `X`. -/
@[simps]
def isoMk {f g : MonoOver X} (h : f.obj.left ≅ g.obj.left)
    (w : h.hom ≫ g.arrow = f.arrow := by aesop_cat) : f ≅ g where
  hom := homMk h.hom w
                         /-
                           C : Type u₁
                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                           X Y Z : C
                           D : Type u₂
                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                           f g : CategoryTheory.MonoOver X
                           h : CategoryTheory.Iso f.obj.left g.obj.left
                           w : autoParam (Eq (CategoryTheory.CategoryStruct.comp h.hom g.arrow) f.arrow)  …
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp h.inv f.arrow) g.arrow
                         -/
  inv := homMk h.inv (by rw [h.inv_comp_eq, w])
                         /-
                           🎉 no goals
                         -/


/-- If `f : MonoOver X`, then `mk' f.arrow` is of course just `f`, but not definitionally, so we
    package it as an isomorphism. -/
@[simp]
def mk'ArrowIso {X : C} (f : MonoOver X) : mk' f.arrow ≅ f :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X : C
    f : CategoryTheory.MonoOver X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
  -/
  isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Lift a functor between over categories to a functor between `MonoOver` categories,
given suitable evidence that morphisms are taken to monomorphisms.
-/
@[simps]
def lift {Y : D} (F : Over Y ⥤ Over X)
    (h : ∀ f : MonoOver Y, Mono (F.obj ((MonoOver.forget Y).obj f)).hom) :
    MonoOver Y ⥤ MonoOver X where
  obj f := ⟨_, h f⟩
  map k := (MonoOver.forget Y ⋙ F).map k


/-- Isomorphic functors `Over Y ⥤ Over X` lift to isomorphic functors `MonoOver Y ⥤ MonoOver X`.
-/
def liftIso {Y : D} {F₁ F₂ : Over Y ⥤ Over X} (h₁ h₂) (i : F₁ ≅ F₂) : lift F₁ h₁ ≅ lift F₂ h₂ :=
  Functor.fullyFaithfulCancelRight (MonoOver.forget X) (isoWhiskerLeft (MonoOver.forget Y) i)


/-- `MonoOver.lift` commutes with composition of functors. -/
def liftComp {X Z : C} {Y : D} (F : Over X ⥤ Over Y) (G : Over Y ⥤ Over Z) (h₁ h₂) :
    lift F h₁ ⋙ lift G h₂ ≅ lift (F ⋙ G) fun f => h₂ ⟨_, h₁ f⟩ :=
  Functor.fullyFaithfulCancelRight (MonoOver.forget _) (Iso.refl _)


/-- `MonoOver.lift` preserves the identity functor. -/
def liftId : (lift (𝟭 (Over X)) fun f => f.2) ≅ 𝟭 _ :=
  Functor.fullyFaithfulCancelRight (MonoOver.forget _) (Iso.refl _)


@[simp]
theorem lift_comm (F : Over Y ⥤ Over X)
    (h : ∀ f : MonoOver Y, Mono (F.obj ((MonoOver.forget Y).obj f)).hom) :
    lift F h ⋙ MonoOver.forget X = MonoOver.forget Y ⋙ F :=
  rfl


@[simp]
theorem lift_obj_arrow {Y : D} (F : Over Y ⥤ Over X)
    (h : ∀ f : MonoOver Y, Mono (F.obj ((MonoOver.forget Y).obj f)).hom) (f : MonoOver Y) :
    ((lift F h).obj f).arrow = (F.obj ((forget Y).obj f)).hom :=
  rfl


/-- Monomorphisms over an object `f : Over A` in an over category
are equivalent to monomorphisms over the source of `f`.
-/
def slice {A : C} {f : Over A}
    (h₁ : ∀ (g : MonoOver f),
      Mono ((Over.iteratedSliceEquiv f).functor.obj ((forget f).obj g)).hom)
    (h₂ : ∀ (g : MonoOver f.left),
      Mono ((Over.iteratedSliceEquiv f).inverse.obj ((forget f.left).obj g)).hom) :
    MonoOver f ≌ MonoOver f.left where
  functor := MonoOver.lift f.iteratedSliceEquiv.functor h₁
  inverse := MonoOver.lift f.iteratedSliceEquiv.inverse h₂
  unitIso :=
    MonoOver.liftId.symm ≪≫
      MonoOver.liftIso _ _ f.iteratedSliceEquiv.unitIso ≪≫ (MonoOver.liftComp _ _ _ _).symm
  counitIso :=
    MonoOver.liftComp _ _ _ _ ≪≫
      MonoOver.liftIso _ _ f.iteratedSliceEquiv.counitIso ≪≫ MonoOver.liftId


/-- When `C` has pullbacks, a morphism `f : X ⟶ Y` induces a functor `MonoOver Y ⥤ MonoOver X`,
by pulling back a monomorphism along `f`. -/
def pullback (f : X ⟶ Y) : MonoOver Y ⥤ MonoOver X :=
  MonoOver.lift (Over.pullback f) (fun g => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      f : Quiver.Hom X Y
      g : CategoryTheory.MonoOver Y
      ⊢ CategoryTheory.Mono ((CategoryTheory.Over.pullback f).obj ((CategoryTheory.M …
    -/
    haveI : Mono ((forget Y).obj g).hom := (inferInstance : Mono g.arrow)
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      f : Quiver.Hom X Y
      g : CategoryTheory.MonoOver Y
      this : CategoryTheory.Mono ((CategoryTheory.MonoOver.forget Y).obj g).hom
      ⊢ CategoryTheory.Mono ((CategoryTheory.Over.pullback f).obj ((CategoryTheory.M …
    -/
    apply pullback.snd_of_mono)
    /-
      🎉 no goals
    -/


/-- pullback commutes with composition (up to a natural isomorphism) -/
def pullbackComp (f : X ⟶ Y) (g : Y ⟶ Z) : pullback (f ≫ g) ≅ pullback g ⋙ pullback f :=
  liftIso _ _ (Over.pullbackComp _ _) ≪≫ (liftComp _ _ _ _).symm


/-- pullback preserves the identity (up to a natural isomorphism) -/
def pullbackId : pullback (𝟙 X) ≅ 𝟭 _ :=
  liftIso _ _ Over.pullbackId ≪≫ liftId


@[simp]
theorem pullback_obj_left (f : X ⟶ Y) (g : MonoOver Y) :
    ((pullback f).obj g : C) = Limits.pullback g.arrow f :=
  rfl


@[simp]
theorem pullback_obj_arrow (f : X ⟶ Y) (g : MonoOver Y) :
    ((pullback f).obj g).arrow = pullback.snd _ _ :=
  rfl


/-- We can map monomorphisms over `X` to monomorphisms over `Y`
by post-composition with a monomorphism `f : X ⟶ Y`.
-/
def map (f : X ⟶ Y) [Mono f] : MonoOver X ⥤ MonoOver Y :=
  lift (Over.map f) fun g => mono_comp g.arrow f


/-- `MonoOver.map` commutes with composition (up to a natural isomorphism). -/
def mapComp (f : X ⟶ Y) (g : Y ⟶ Z) [Mono f] [Mono g] : map (f ≫ g) ≅ map f ⋙ map g :=
  liftIso _ _ (Over.mapComp _ _) ≪≫ (liftComp _ _ _ _).symm


/-- `MonoOver.map` preserves the identity (up to a natural isomorphism). -/
def mapId : map (𝟙 X) ≅ 𝟭 _ :=
  liftIso _ _ (Over.mapId X) ≪≫ liftId


@[simp]
theorem map_obj_left (f : X ⟶ Y) [Mono f] (g : MonoOver X) : ((map f).obj g : C) = g.obj.left :=
  rfl


@[simp]
theorem map_obj_arrow (f : X ⟶ Y) [Mono f] (g : MonoOver X) : ((map f).obj g).arrow = g.arrow ≫ f :=
  rfl


instance full_map (f : X ⟶ Y) [Mono f] : Functor.Full (map f) where
  map_surjective {g h} e := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      g h : CategoryTheory.MonoOver X
      e : Quiver.Hom ((CategoryTheory.MonoOver.map f).obj g) ((CategoryTheory.MonoOv …
      ⊢ Exists fun a => Eq ((CategoryTheory.MonoOver.map f).map a) e
    -/
    refine ⟨homMk e.left ?_, rfl⟩
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.Mono f
        g h : CategoryTheory.MonoOver X
        e : Quiver.Hom ((CategoryTheory.MonoOver.map f).obj g) ((CategoryTheory.MonoOv …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp e.left h.arrow) g.arrow
      -/
    · rw [← cancel_mono f, assoc]
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.Mono f
        g h : CategoryTheory.MonoOver X
        e : Quiver.Hom ((CategoryTheory.MonoOver.map f).obj g) ((CategoryTheory.MonoOv …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp e.left (CategoryTheory.CategoryStruct …
      -/
      apply w e
      /-
        🎉 no goals
      -/


instance faithful_map (f : X ⟶ Y) [Mono f] : Functor.Faithful (map f) where


/-- Isomorphic objects have equivalent `MonoOver` categories.
-/
@[simps]
def mapIso {A B : C} (e : A ≅ B) : MonoOver A ≌ MonoOver B where
  functor := map e.hom
  inverse := map e.inv
                                                /-
                                                  C : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                  X Y Z : C
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  A B : C
                                                  e : CategoryTheory.Iso A B
                                                  ⊢ Eq (CategoryTheory.MonoOver.map (CategoryTheory.CategoryStruct.comp e.hom e. …
                                                -/
  unitIso := ((mapComp _ _).symm ≪≫ eqToIso (by simp) ≪≫ (mapId _)).symm
                                                /-
                                                  🎉 no goals
                                                -/
                                                 /-
                                                   C : Type u₁
                                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                   X Y Z : C
                                                   D : Type u₂
                                                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                   A B : C
                                                   e : CategoryTheory.Iso A B
                                                   ⊢ Eq (CategoryTheory.MonoOver.map (CategoryTheory.CategoryStruct.comp e.inv e. …
                                                 -/
  counitIso := (mapComp _ _).symm ≪≫ eqToIso (by simp) ≪≫ (mapId _)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- An equivalence of categories `e` between `C` and `D` induces an equivalence between
    `MonoOver X` and `MonoOver (e.functor.obj X)` whenever `X` is an object of `C`. -/
@[simps]
def congr (e : C ≌ D) : MonoOver X ≌ MonoOver (e.functor.obj X) where
  functor :=
    lift (Over.post e.functor) fun f => by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Equivalence C D
        f : CategoryTheory.MonoOver X
        ⊢ CategoryTheory.Mono ((CategoryTheory.Over.post e.functor).obj ((CategoryTheo …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Equivalence C D
        f : CategoryTheory.MonoOver X
        ⊢ CategoryTheory.Mono (e.functor.map f.arrow)
      -/
      infer_instance
      /-
        🎉 no goals
      -/
  inverse :=
    (lift (Over.post e.inverse) fun f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Equivalence C D
          f : CategoryTheory.MonoOver (e.functor.obj X)
          ⊢ CategoryTheory.Mono ((CategoryTheory.Over.post e.inverse).obj ((CategoryTheo …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Equivalence C D
          f : CategoryTheory.MonoOver (e.functor.obj X)
          ⊢ CategoryTheory.Mono (e.inverse.map f.arrow)
        -/
        infer_instance) ⋙
        /-
          🎉 no goals
        -/
      (mapIso (e.unitIso.symm.app X)).functor
                                          /-
                                            C : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                            X Y✝ Z : C
                                            D : Type u₂
                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                            e : CategoryTheory.Equivalence C D
                                            Y : CategoryTheory.MonoOver X
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unitIso.app Y.obj.left).hom (((Cat …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun Y => isoMk (e.unitIso.app Y)
             /-
               🎉 no goals
             -/
                                            /-
                                              C : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                              X Y✝ Z : C
                                              D : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                              e : CategoryTheory.Equivalence C D
                                              Y : CategoryTheory.MonoOver (e.functor.obj X)
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.counitIso.app Y.obj.left).hom ((Ca …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun Y => isoMk (e.counitIso.app Y)
               /-
                 🎉 no goals
               -/


/-- `map f` is left adjoint to `pullback f` when `f` is a monomorphism -/
def mapPullbackAdj (f : X ⟶ Y) [Mono f] : map f ⊣ pullback f :=
  (Over.mapPullbackAdj f).restrictFullyFaithful (fullyFaithfulForget X) (fullyFaithfulForget Y)
    (Iso.refl _) (Iso.refl _)


/-- `MonoOver.map f` followed by `MonoOver.pullback f` is the identity. -/
def pullbackMapSelf (f : X ⟶ Y) [Mono f] : map f ⋙ pullback f ≅ 𝟭 _ :=
  (asIso (MonoOver.mapPullbackAdj f).unit).symm


/-- The `MonoOver Y` for the image inclusion for a morphism `f : X ⟶ Y`.
-/
def imageMonoOver (f : X ⟶ Y) [HasImage f] : MonoOver Y :=
  MonoOver.mk' (image.ι f)


@[simp]
theorem imageMonoOver_arrow (f : X ⟶ Y) [HasImage f] : (imageMonoOver f).arrow = image.ι f :=
  rfl


/-- Taking the image of a morphism gives a functor `Over X ⥤ MonoOver X`.
-/
@[simps]
def image : Over X ⥤ MonoOver X where
  obj f := imageMonoOver f.hom
  map {f g} k := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasImages C
      f g : CategoryTheory.Over X
      k : Quiver.Hom f g
      ⊢ Quiver.Hom ((fun f => CategoryTheory.MonoOver.imageMonoOver f.hom) f) ((fun  …
    -/
    apply (forget X).preimage _
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasImages C
      f g : CategoryTheory.Over X
      k : Quiver.Hom f g
      ⊢ Quiver.Hom ((CategoryTheory.MonoOver.forget X).obj ((fun f => CategoryTheory …
    -/
    apply Over.homMk _ _
    · exact
        image.lift
          { I := Limits.image _
            m := image.ι g.hom
            e := k.left ≫ factorThruImage g.hom }
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        inst✝ : CategoryTheory.Limits.HasImages C
        f g : CategoryTheory.Over X
        k : Quiver.Hom f g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
      -/
    · apply image.lift_fac
      /-
        🎉 no goals
      -/


/-- `MonoOver.image : Over X ⥤ MonoOver X` is left adjoint to
`MonoOver.forget : MonoOver X ⥤ Over X`
-/
def imageForgetAdj : image ⊣ forget X :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun f g =>
        { toFun := fun k => by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom (CategoryTheory.MonoOver.image.obj f) g
              ⊢ Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
            -/
            apply Over.homMk (factorThruImage f.hom ≫ k.left) _
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom (CategoryTheory.MonoOver.image.obj f) g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
            change (factorThruImage f.hom ≫ k.left) ≫ _ = f.hom
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom (CategoryTheory.MonoOver.image.obj f) g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
            rw [assoc, Over.w k]
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom (CategoryTheory.MonoOver.image.obj f) g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
            -/
            apply image.fac
            /-
              🎉 no goals
            -/
          invFun := fun k => by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
              ⊢ Quiver.Hom (CategoryTheory.MonoOver.image.obj f) g
            -/
            refine Over.homMk ?_ ?_
            · exact
                image.lift
                  { I := g.obj.left
                    m := g.arrow
                    e := k.left
                    fac := Over.w k }
              /-
                case refine_2
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                X Y Z : C
                D : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                inst✝ : CategoryTheory.Limits.HasImages C
                f : CategoryTheory.Over X
                g : CategoryTheory.MonoOver X
                k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
              -/
            · apply image.lift_fac
              /-
                🎉 no goals
              -/
          left_inv := fun _ => Subsingleton.elim _ _
          right_inv := fun k => by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
              ⊢ Eq ((fun k => CategoryTheory.Over.homMk (CategoryTheory.CategoryStruct.comp  …
            -/
            ext1
            /-
              case h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
              ⊢ Eq ((fun k => CategoryTheory.Over.homMk (CategoryTheory.CategoryStruct.comp  …
            -/
            change factorThruImage _ ≫ image.lift _ = _
            /-
              case h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
            -/
            rw [← cancel_mono g.arrow, assoc, image.lift_fac, image.fac f.hom]
            /-
              case h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : CategoryTheory.Limits.HasImages C
              f : CategoryTheory.Over X
              g : CategoryTheory.MonoOver X
              k : Quiver.Hom f ((CategoryTheory.MonoOver.forget X).obj g)
              ⊢ Eq f.hom (CategoryTheory.CategoryStruct.comp k.left g.arrow)
            -/
            exact (Over.w k).symm } }
            /-
              🎉 no goals
            -/


instance : (forget X).IsRightAdjoint :=
  ⟨_, ⟨imageForgetAdj⟩⟩


instance reflective : Reflective (forget X) where
  adj := imageForgetAdj


/-- Forgetting that a monomorphism over `X` is a monomorphism, then taking its image,
is the identity functor.
-/
def forgetImage : forget X ⋙ image ≅ 𝟭 (MonoOver X) :=
  asIso (Adjunction.counit imageForgetAdj)


/-- In the case where `f` is not a monomorphism but `C` has images,
we can still take the "forward map" under it, which agrees with `MonoOver.map f`.
-/
def «exists» (f : X ⟶ Y) : MonoOver X ⥤ MonoOver Y :=
  forget _ ⋙ Over.map f ⋙ image


instance faithful_exists (f : X ⟶ Y) : Functor.Faithful («exists» f) where


/-- When `f : X ⟶ Y` is a monomorphism, `exists f` agrees with `map f`.
-/
def existsIsoMap (f : X ⟶ Y) [Mono f] : «exists» f ≅ map f :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ ∀ {X_1 Y_1 : CategoryTheory.MonoOver X} (f_1 : Quiver.Hom X_1 Y_1), Eq (Cate …
  -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasImages C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      ⊢ (X_1 : CategoryTheory.MonoOver X) → CategoryTheory.Iso ((CategoryTheory.Mono …
    -/
  NatIso.ofComponents (by
  /-
    🎉 no goals
  -/
    intro Z
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z✝ : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasImages C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      Z : CategoryTheory.MonoOver X
      ⊢ CategoryTheory.Iso ((CategoryTheory.MonoOver.forget Y).obj ((CategoryTheory. …
    -/
    suffices (forget _).obj ((«exists» f).obj Z) ≅ (forget _).obj ((map f).obj Z) by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z✝ : C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasImages C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.Mono f
        Z : CategoryTheory.MonoOver X
        ⊢ CategoryTheory.Iso ((CategoryTheory.MonoOver.forget Y).obj ((CategoryTheory. …
      -/
      apply (forget _).preimageIso this
      /-
        🎉 no goals
      -/
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z✝ : C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasImages C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.Mono f
        Z : CategoryTheory.MonoOver X
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageMonoIsoSo …
      -/
    apply Over.isoMk _ _
      /-
        🎉 no goals
      -/
    · apply imageMonoIsoSource (Z.arrow ≫ f)
    · apply imageMonoIsoSource_hom_self)


/-- `exists` is adjoint to `pullback` when images exist -/
def existsPullbackAdj (f : X ⟶ Y) [HasPullbacks C] : «exists» f ⊣ pullback f :=
  ((Over.mapPullbackAdj f).comp imageForgetAdj).restrictFullyFaithful
    (fullyFaithfulForget X) (Functor.FullyFaithful.id _) (Iso.refl _) (Iso.refl _)


