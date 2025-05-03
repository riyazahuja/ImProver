/-- A `SheafedSpace C` is a topological space equipped with a sheaf of `C`s. -/
structure SheafedSpace extends PresheafedSpace C where
  /-- A sheafed space is presheafed space which happens to be sheaf. -/
  IsSheaf : presheaf.IsSheaf


instance coeCarrier : CoeOut (SheafedSpace C) TopCat where coe X := X.carrier


instance coeSort : CoeSort (SheafedSpace C) Type* where
  coe X := X.1


/-- Extract the `sheaf C (X : Top)` from a `SheafedSpace C`. -/
def sheaf (X : SheafedSpace C) : Sheaf C (X : TopCat) :=
  ⟨X.presheaf, X.IsSheaf⟩

-- Porting note: this is a syntactic tautology, so removed
-- @[simp]
-- theorem as_coe (X : SheafedSpace C) : X.carrier = (X : TopCat) :=
--   rfl

-- Porting note: this gives a `simpVarHead` error (`LEFT-HAND SIDE HAS VARIABLE AS HEAD SYMBOL.`).
-- so removed @[simp]

theorem mk_coe (carrier) (presheaf) (h) :
    (({ carrier
        presheaf
        IsSheaf := h } : SheafedSpace C) : TopCat) = carrier :=
  rfl


instance (X : SheafedSpace C) : TopologicalSpace X :=
  X.carrier.str


/-- The trivial `unit` valued sheaf on any topological space. -/
def unit (X : TopCat) : SheafedSpace (Discrete Unit) :=
  { @PresheafedSpace.const (Discrete Unit) _ X ⟨⟨⟩⟩ with IsSheaf := Presheaf.isSheaf_unit _ }


instance : Inhabited (SheafedSpace (Discrete Unit)) :=
  ⟨unit (TopCat.of PEmpty)⟩


instance : Category (SheafedSpace C) :=
  show Category (InducedCategory (PresheafedSpace C) SheafedSpace.toPresheafedSpace) by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ CategoryTheory.Category.{?u.2543, max (max (?u.2560 + 1) v) u} (CategoryTheo …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[ext (iff := false)]
theorem ext {X Y : SheafedSpace C} (α β : X ⟶ Y) (w : α.base = β.base)
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X Y : AlgebraicGeometry.SheafedSpace C
                                           α β : Quiver.Hom X Y
                                           w : Eq α.base β.base
                                           ⊢ Eq (TopologicalSpace.Opens.map α.base).op (TopologicalSpace.Opens.map β.base …
                                         -/
    (h : α.c ≫ whiskerRight (eqToHom (by rw [w])) _ = β.c) : α = β :=
                                         /-
                                           🎉 no goals
                                         -/
  PresheafedSpace.ext α β w h


/-- Constructor for isomorphisms in the category `SheafedSpace C`. -/
@[simps]
def isoMk {X Y : SheafedSpace C} (e : X.toPresheafedSpace ≅ Y.toPresheafedSpace) : X ≅ Y where
  hom := e.hom
  inv := e.inv
  hom_inv_id := e.hom_inv_id
  inv_hom_id := e.inv_hom_id


/-- Forgetting the sheaf condition is a functor from `SheafedSpace C` to `PresheafedSpace C`. -/
@[simps! obj map]
def forgetToPresheafedSpace : SheafedSpace C ⥤ PresheafedSpace C :=
  inducedFunctor _

-- Porting note: can't derive `Full` functor automatically

instance forgetToPresheafedSpace_full : (forgetToPresheafedSpace (C := C)).Full where
  map_surjective f := ⟨f, rfl⟩

-- Porting note: can't derive `Faithful` functor automatically

instance forgetToPresheafedSpace_faithful : (forgetToPresheafedSpace (C := C)).Faithful where


instance is_presheafedSpace_iso {X Y : SheafedSpace C} (f : X ⟶ Y) [IsIso f] :
    @IsIso (PresheafedSpace C) _ _ _ f :=
  SheafedSpace.forgetToPresheafedSpace.map_isIso f


attribute [local simp] id comp


@[simp]
theorem id_base (X : SheafedSpace C) : (𝟙 X : X ⟶ X).base = 𝟙 (X : TopCat) :=
  rfl


theorem id_c (X : SheafedSpace C) :
    (𝟙 X : X ⟶ X).c = eqToHom (Presheaf.Pushforward.id_eq X.presheaf).symm :=
  rfl


@[simp]
theorem id_c_app (X : SheafedSpace C) (U) :
    (𝟙 X : X ⟶ X).c.app U = 𝟙 _ := rfl


@[simp]
theorem comp_base {X Y Z : SheafedSpace C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl


@[simp]
theorem comp_c_app {X Y Z : SheafedSpace C} (α : X ⟶ Y) (β : Y ⟶ Z) (U) :
    (α ≫ β).c.app U = β.c.app U ≫ α.c.app (op ((Opens.map β.base).obj (unop U))) :=
  rfl


theorem comp_c_app' {X Y Z : SheafedSpace C} (α : X ⟶ Y) (β : Y ⟶ Z) (U) :
    (α ≫ β).c.app (op U) = β.c.app (op U) ≫ α.c.app (op ((Opens.map β.base).obj U)) :=
  rfl


theorem congr_app {X Y : SheafedSpace C} {α β : X ⟶ Y} (h : α = β) (U) :
                                                        /-
                                                          C : Type u
                                                          inst✝ : CategoryTheory.Category.{v, u} C
                                                          X Y : AlgebraicGeometry.SheafedSpace C
                                                          α β : Quiver.Hom X Y
                                                          h : Eq α β
                                                          U : Opposite (TopologicalSpace.Opens ↑↑Y.toPresheafedSpace)
                                                          ⊢ Eq ((TopologicalSpace.Opens.map β.base).op.obj U) ((TopologicalSpace.Opens.m …
                                                        -/
    α.c.app U = β.c.app U ≫ X.presheaf.map (eqToHom (by subst h; rfl)) :=
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  PresheafedSpace.congr_app h U


/-- The forgetful functor from `SheafedSpace` to `Top`. -/
def forget : SheafedSpace C ⥤ TopCat where
  obj X := (X : TopCat)
  map {_ _} f := f.base


/-- The restriction of a sheafed space along an open embedding into the space.
-/
def restrict {U : TopCat} (X : SheafedSpace C) {f : U ⟶ (X : TopCat)} (h : IsOpenEmbedding f) :
    SheafedSpace C :=
  { X.toPresheafedSpace.restrict h with IsSheaf := isSheaf_of_isOpenEmbedding h X.IsSheaf }


/-- The map from the restriction of a presheafed space.
-/
@[simps!]
def ofRestrict {U : TopCat} (X : SheafedSpace C) {f : U ⟶ (X : TopCat)}
    (h : IsOpenEmbedding f) : X.restrict h ⟶ X := X.toPresheafedSpace.ofRestrict h


/-- The restriction of a sheafed space `X` to the top subspace is isomorphic to `X` itself.
-/
@[simps! hom inv]
def restrictTopIso (X : SheafedSpace C) : X.restrict (Opens.isOpenEmbedding ⊤) ≅ X :=
  isoMk (X.toPresheafedSpace.restrictTopIso)


/-- The global sections, notated Gamma.
-/
def Γ : (SheafedSpace C)ᵒᵖ ⥤ C :=
  forgetToPresheafedSpace.op ⋙ PresheafedSpace.Γ


theorem Γ_def : (Γ : _ ⥤ C) = forgetToPresheafedSpace.op ⋙ PresheafedSpace.Γ :=
  rfl


@[simp]
theorem Γ_obj (X : (SheafedSpace C)ᵒᵖ) : Γ.obj X = (unop X).presheaf.obj (op ⊤) :=
  rfl


theorem Γ_obj_op (X : SheafedSpace C) : Γ.obj (op X) = X.presheaf.obj (op ⊤) :=
  rfl


@[simp]
theorem Γ_map {X Y : (SheafedSpace C)ᵒᵖ} (f : X ⟶ Y) : Γ.map f = f.unop.c.app (op ⊤) :=
  rfl


theorem Γ_map_op {X Y : SheafedSpace C} (f : X ⟶ Y) : Γ.map f.op = f.c.app (op ⊤) :=
  rfl


noncomputable instance [HasLimits C] :
    CreatesColimits (forgetToPresheafedSpace : SheafedSpace C ⥤ _) :=
  ⟨fun {_ _} =>
    ⟨fun {K} =>
      createsColimitOfFullyFaithfulOfIso
        ⟨(PresheafedSpace.colimitCocone (K ⋙ forgetToPresheafedSpace)).pt,
          limit_isSheaf _ fun j => Sheaf.pushforward_sheaf_of_sheaf _ (K.obj (unop j)).2⟩
        (colimit.isoColimitCocone ⟨_, PresheafedSpace.colimitCoconeIsColimit _⟩).symm⟩⟩


instance [HasLimits C] : HasColimits.{v} (SheafedSpace C) :=
  hasColimits_of_hasColimits_createsColimits forgetToPresheafedSpace


noncomputable instance [HasLimits C] : PreservesColimits (forget.{_, _, v} C) :=
  Limits.comp_preservesColimits forgetToPresheafedSpace (PresheafedSpace.forget C)


attribute [local instance] ConcreteCategory.instFunLike in
lemma hom_stalk_ext {X Y : SheafedSpace C} (f g : X ⟶ Y) (h : f.base = g.base)
    (h' : ∀ x, f.stalkMap x = (Y.presheaf.stalkCongr (h ▸ rfl)).hom ≫ g.stalkMap x) :
    f = g := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f g : Quiver.Hom X Y
    h : Eq f.base g.base
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq (AlgebraicGeometry.PresheafedSpace.Hom. …
    ⊢ Eq f g
  -/
  obtain ⟨f, fc⟩ := f
  /-
    case mk
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    g : Quiver.Hom X Y
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    h : Eq { base := f, c := fc }.base g.base
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    ⊢ Eq { base := f, c := fc } g
  -/
  obtain ⟨g, gc⟩ := g
  /-
    case mk.mk
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    g : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    gc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C g).obj X.presheaf)
    h : Eq { base := f, c := fc }.base { base := g, c := gc }.base
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    ⊢ Eq { base := f, c := fc } { base := g, c := gc }
  -/
  obtain rfl : f = g := h
  /-
    case mk.mk
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc gc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    ⊢ Eq { base := f, c := fc } { base := f, c := gc }
  -/
  congr
  /-
    case mk.mk.e_c
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc gc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    ⊢ Eq fc gc
  -/
  ext U s
  refine section_ext X.sheaf _ _ _ fun x hx ↦
    show X.presheaf.germ _ x _ _ = X.presheaf.germ _ x _ _ from ?_
  /-
    case mk.mk.e_c.w.w
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc gc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
    s : (CategoryTheory.forget C).obj (Y.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem ((TopologicalSpace.Opens.map f).obj (Opposite.unop { unop  …
    ⊢ Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f).obj (Opposite.unop { un …
  -/
  erw [← PresheafedSpace.stalkMap_germ_apply ⟨f, fc⟩, ← PresheafedSpace.stalkMap_germ_apply ⟨f, gc⟩]
  /-
    case mk.mk.e_c.w.w
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom ↑X.toPresheafedSpace ↑Y.toPresheafedSpace
    fc gc : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C f).obj X.presheaf)
    h' : ∀ (x : ↑↑X.toPresheafedSpace), Eq ({ base := f, c := fc }.stalkMap x) (Ca …
    U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
    s : (CategoryTheory.forget C).obj (Y.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem ((TopologicalSpace.Opens.map f).obj (Opposite.unop { unop  …
    ⊢ Eq (({ base := f, c := fc }.stalkMap x) ((Y.presheaf.germ (Opposite.unop { u …
  -/
  simp [h']
  /-
    🎉 no goals
  -/


lemma mono_of_base_injective_of_stalk_epi {X Y : SheafedSpace C} (f : X ⟶ Y)
    (h₁ : Function.Injective f.base)
    (h₂ : ∀ x, Epi (f.stalkMap x)) : Mono f := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    h₁ : Function.Injective ⇑f.base
    h₂ : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.Epi (AlgebraicGeometry.Pres …
    ⊢ CategoryTheory.Mono f
  -/
  constructor
  /-
    case right_cancellation
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    h₁ : Function.Injective ⇑f.base
    h₂ : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.Epi (AlgebraicGeometry.Pres …
    ⊢ ∀ {Z : AlgebraicGeometry.SheafedSpace C} (g h : Quiver.Hom Z X), Eq (Categor …
  -/
  intro Z ⟨g, gc⟩ ⟨h, hc⟩ e
  /-
    case right_cancellation
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    h₁ : Function.Injective ⇑f.base
    h₂ : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.Epi (AlgebraicGeometry.Pres …
    Z : AlgebraicGeometry.SheafedSpace C
    g : Quiver.Hom ↑Z.toPresheafedSpace ↑X.toPresheafedSpace
    gc : Quiver.Hom X.presheaf ((TopCat.Presheaf.pushforward C g).obj Z.presheaf)
    h : Quiver.Hom ↑Z.toPresheafedSpace ↑X.toPresheafedSpace
    hc : Quiver.Hom X.presheaf ((TopCat.Presheaf.pushforward C h).obj Z.presheaf)
    e : Eq (CategoryTheory.CategoryStruct.comp { base := g, c := gc } f) (Category …
    ⊢ Eq { base := g, c := gc } { base := h, c := hc }
  -/
  obtain rfl : g = h := ConcreteCategory.hom_ext _ _ fun x ↦ h₁ congr(($e).base x)
  /-
    case right_cancellation
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    h₁ : Function.Injective ⇑f.base
    h₂ : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.Epi (AlgebraicGeometry.Pres …
    Z : AlgebraicGeometry.SheafedSpace C
    g : Quiver.Hom ↑Z.toPresheafedSpace ↑X.toPresheafedSpace
    gc hc : Quiver.Hom X.presheaf ((TopCat.Presheaf.pushforward C g).obj Z.presheaf)
    e : Eq (CategoryTheory.CategoryStruct.comp { base := g, c := gc } f) (Category …
    ⊢ Eq { base := g, c := gc } { base := g, c := hc }
  -/
  refine SheafedSpace.hom_stalk_ext ⟨g, gc⟩ ⟨g, hc⟩ rfl fun x ↦ ?_
  rw [← cancel_epi (f.stalkMap (g x)), stalkCongr_hom, stalkSpecializes_refl, Category.id_comp,
    ← PresheafedSpace.stalkMap.comp ⟨g, gc⟩ f, ← PresheafedSpace.stalkMap.comp ⟨g, hc⟩ f]
  /-
    case right_cancellation
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.HasColimits C
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    h₁ : Function.Injective ⇑f.base
    h₂ : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.Epi (AlgebraicGeometry.Pres …
    Z : AlgebraicGeometry.SheafedSpace C
    g : Quiver.Hom ↑Z.toPresheafedSpace ↑X.toPresheafedSpace
    gc hc : Quiver.Hom X.presheaf ((TopCat.Presheaf.pushforward C g).obj Z.presheaf)
    e : Eq (CategoryTheory.CategoryStruct.comp { base := g, c := gc } f) (Category …
    x : ↑↑Z.toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.CategoryS …
  -/
  congr 1
  /-
    🎉 no goals
  -/


