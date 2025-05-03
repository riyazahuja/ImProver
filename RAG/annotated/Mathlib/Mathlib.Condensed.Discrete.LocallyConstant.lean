/--
The functor from the category of sets to presheaves on `CompHausLike P` given by locally constant
maps.
-/
@[simps]
def functorToPresheaves : Type (max u w) ⥤ ((CompHausLike.{u} P)ᵒᵖ ⥤ Type max u w) where
  obj X := {
    obj := fun ⟨S⟩ ↦ LocallyConstant S X
    map := fun f g ↦ g.comap f.unop }
  map f := { app := fun _ t ↦ t.map f }


/--
Locally constant maps are the same as continuous maps when the target is equipped with the discrete
topology
-/
@[simps]
def locallyConstantIsoContinuousMap (Y X : Type*) [TopologicalSpace Y] :
    LocallyConstant Y X ≅ C(Y, TopCat.discrete.obj X) :=
  letI : TopologicalSpace X := ⊥
  haveI : DiscreteTopology X := ⟨rfl⟩
  { hom := fun f ↦ (f : C(Y, X))
    inv := fun f ↦ ⟨f, (IsLocallyConstant.iff_continuous f).mpr f.2⟩ }


/-- A fiber of a locally constant map as a `CompHausLike P`. -/
def fiber : CompHausLike.{u} P := CompHausLike.of P a.val


instance : HasProp P (fiber r a) := inferInstanceAs (HasProp P (Subtype _))


/-- The inclusion map from a component of the coproduct induced by `f` into `S`. -/
def sigmaIncl : fiber r a ⟶ Q := TopologicalSpace.Fiber.sigmaIncl _ a


/-- The canonical map from the coproduct induced by `f` to `S` as an isomorphism in
`CompHausLike P`. -/
noncomputable def sigmaIso [HasExplicitFiniteCoproducts.{u} P] : (finiteCoproduct (fiber r)) ≅ Q :=
  isoOfBijective (sigmaIsoHom r) ⟨sigmaIsoHom_inj r, sigmaIsoHom_surj r⟩


lemma sigmaComparison_comp_sigmaIso [HasExplicitFiniteCoproducts.{u} P]
    (X : (CompHausLike.{u} P)ᵒᵖ ⥤ Type max u w) :
    (X.mapIso (sigmaIso r).op).hom ≫ sigmaComparison X (fun a ↦ (fiber r a).1) ≫
      (fun g ↦ g a) = X.map (sigmaIncl r a).op := by
  /-
    P : TopCat → Prop
    inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    Q : CompHausLike P
    Z : Type (max u w)
    r : LocallyConstant (↑Q.toTop) Z
    a : Function.Fiber ⇑r
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.mapIso (CompHausLike.LocallyConsta …
  -/
  ext
  simp only [Functor.mapIso_hom, Iso.op_hom, types_comp_apply, sigmaComparison, coe_of,
    ← FunctorToTypes.map_comp_apply]
  /-
    case h
    P : TopCat → Prop
    inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    Q : CompHausLike P
    Z : Type (max u w)
    r : LocallyConstant (↑Q.toTop) Z
    a : Function.Fiber ⇑r
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    a✝ : X.obj { unop := Q }
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (CompHausLike.LocallyConstant. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The projection of the counit. -/
noncomputable def counitAppAppImage : (a : Fiber f) → Y.obj ⟨fiber f a⟩ :=
  fun a ↦ Y.map (CompHausLike.isTerminalPUnit.from _).op a.image


/--
The counit is defined as follows: given a locally constant map `f : S → Y(*)`, let
`S = S₁ ⊔ ⋯ ⊔ Sₙ` be the corresponding decomposition of `S` into the fibers. We need to provide an
element of `Y(S)`. It suffices to provide an element of `Y(Sᵢ)` for all `i`. Let `yᵢ ∈ Y(*)` denote
the value of `f` on `Sᵢ`. Our desired element is the image of `yᵢ` under the canonical map
`Y(*) → Y(Sᵢ)`.
-/
noncomputable def counitAppApp (S : CompHausLike.{u} P) (Y : (CompHausLike.{u} P)ᵒᵖ ⥤ Type max u w)
    [PreservesFiniteProducts Y] [HasExplicitFiniteCoproducts.{u} P] :
    LocallyConstant S (Y.obj (op (CompHausLike.of P PUnit.{u+1}))) ⟶ Y.obj ⟨S⟩ :=
  fun r ↦ ((inv (sigmaComparison Y (fun a ↦ (fiber r a).1))) ≫
    (Y.mapIso (sigmaIso r).op).inv) (counitAppAppImage r)

-- This is the key lemma to prove naturality of the counit:

/--
To check equality of two elements of `X(S)`, it suffices to check equality after composing with
each `X(S) → X(Sᵢ)`.
-/
lemma presheaf_ext (X : (CompHausLike.{u} P)ᵒᵖ ⥤ Type max u w)
    [PreservesFiniteProducts X] (x y : X.obj ⟨S⟩)
    [HasExplicitFiniteCoproducts.{u} P]
    (h : ∀ (a : Fiber f), X.map (sigmaIncl f a).op x = X.map (sigmaIncl f a).op y) : x = y := by
  /-
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    h : ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIn …
    ⊢ Eq x y
  -/
  apply injective_of_mono (X.mapIso (sigmaIso f).op).hom
  /-
    case a
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    h : ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIn …
    ⊢ Eq ((X.mapIso (CompHausLike.LocallyConstant.sigmaIso f).op).hom x) ((X.mapIs …
  -/
  apply injective_of_mono (sigmaComparison X (fun a ↦ (fiber f a).1))
  /-
    case a.a
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    h : ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIn …
    ⊢ Eq (CompHausLike.sigmaComparison X (fun a => ↑(CompHausLike.LocallyConstant. …
  -/
  ext a
  /-
    case a.a.h
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    h : ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIn …
    a : Function.Fiber ⇑f
    ⊢ Eq (CompHausLike.sigmaComparison X (fun a => ↑(CompHausLike.LocallyConstant. …
  -/
  specialize h a
  /-
    case a.a.h
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    a : Function.Fiber ⇑f
    h : Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f a).op x) (X.map (CompH …
    ⊢ Eq (CompHausLike.sigmaComparison X (fun a => ↑(CompHausLike.LocallyConstant. …
  -/
  rw [← sigmaComparison_comp_sigmaIso] at h
  /-
    case a.a.h
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts X
    x y : X.obj { unop := S }
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    a : Function.Fiber ⇑f
    h : Eq (CategoryTheory.CategoryStruct.comp (X.mapIso (CompHausLike.LocallyCons …
    ⊢ Eq (CompHausLike.sigmaComparison X (fun a => ↑(CompHausLike.LocallyConstant. …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma incl_of_counitAppApp [PreservesFiniteProducts Y] [HasExplicitFiniteCoproducts.{u} P]
    (a : Fiber f) : Y.map (sigmaIncl f a).op (counitAppApp S Y f) = counitAppAppImage f a := by
  /-
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts Y
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    a : Function.Fiber ⇑f
    ⊢ Eq (Y.map (CompHausLike.LocallyConstant.sigmaIncl f a).op (CompHausLike.Loca …
  -/
  rw [← sigmaComparison_comp_sigmaIso, Functor.mapIso_hom, Iso.op_hom, types_comp_apply]
  simp only [counitAppApp, Functor.mapIso_inv, ← Iso.op_hom, types_comp_apply,
    ← FunctorToTypes.map_comp_apply, Iso.inv_hom_id, FunctorToTypes.map_id_apply]
  exact congrFun (inv_hom_id_apply (asIso (sigmaComparison Y (fun a ↦ (fiber f a).1)))
    (counitAppAppImage f)) _


/--
This is an auxiliary definition, the details do not matter. What's important is that this map exists
so that the lemma `incl_comap` works.
-/
def componentHom (a : Fiber (f.comap g)) :
    fiber _ a ⟶ fiber _ (Fiber.mk f (g a.preimage)) where
  toFun x := ⟨g x.val, by
    /-
      P : TopCat → Prop
      inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      a : Function.Fiber ⇑(LocallyConstant.comap g f)
      x : ↑(CompHausLike.LocallyConstant.fiber (LocallyConstant.comap g f) a).toTop
      ⊢ Membership.mem (↑(Function.Fiber.mk (⇑f) (g (Function.Fiber.preimage (⇑(Loca …
    -/
    simp only [Fiber.mk, Set.mem_preimage, Set.mem_singleton_iff]
    /-
      P : TopCat → Prop
      inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      a : Function.Fiber ⇑(LocallyConstant.comap g f)
      x : ↑(CompHausLike.LocallyConstant.fiber (LocallyConstant.comap g f) a).toTop
      ⊢ Eq (f (g ↑x)) (f (g (Function.Fiber.preimage (⇑(LocallyConstant.comap g f))  …
    -/
    convert map_eq_image _ _ x
    /-
      case h.e'_3
      P : TopCat → Prop
      inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      a : Function.Fiber ⇑(LocallyConstant.comap g f)
      x : ↑(CompHausLike.LocallyConstant.fiber (LocallyConstant.comap g f) a).toTop
      ⊢ Eq (f (g (Function.Fiber.preimage (⇑(LocallyConstant.comap g f)) a))) (Funct …
    -/
    exact map_preimage_eq_image_map _ _ a⟩
    /-
      🎉 no goals
    -/
                         /-
                           P : TopCat → Prop
                           inst✝¹ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
                           S : CompHausLike P
                           Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
                           inst✝ : CompHausLike.HasProp P PUnit.{u + 1}
                           f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
                           T : CompHausLike P
                           g : Quiver.Hom T S
                           a : Function.Fiber ⇑(LocallyConstant.comap g f)
                           ⊢ Continuous fun x => ⟨g ↑x, ⋯⟩
                         -/
  continuous_toFun := by exact Continuous.subtype_mk (g.continuous.comp continuous_subtype_val) _
                         /-
                           🎉 no goals
                         -/
    -- term mode gives "unknown free variable" error.


lemma incl_comap {S T : (CompHausLike P)ᵒᵖ}
    (f : LocallyConstant S.unop (Y.obj (op (CompHausLike.of P PUnit.{u+1}))))
      (g : S ⟶ T) (a : Fiber (f.comap g.unop)) :
        g ≫ (sigmaIncl (f.comap g.unop) a).op =
          (sigmaIncl f _).op ≫ (componentHom f g.unop a).op :=
  rfl


/-- The counit is natural in `S : CompHausLike P` -/
@[simps!]
noncomputable def counitApp [HasExplicitFiniteCoproducts.{u} P]
    (Y : (CompHausLike.{u} P)ᵒᵖ ⥤ Type max u w) [PreservesFiniteProducts Y] :
    (functorToPresheaves.obj (Y.obj (op (CompHausLike.of P PUnit.{u+1})))) ⟶ Y where
  app := fun ⟨S⟩ ↦ counitAppApp S Y
  naturality := by
    /-
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u +  …
      T : CompHausLike P
      g : Quiver.Hom T S
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      ⊢ ∀ ⦃X Y_1 : Opposite (CompHausLike P)⦄ (f : Quiver.Hom X Y_1), Eq (CategoryTh …
    -/
    intro S T g
    /-
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u + …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.functo …
    -/
    ext f
    /-
      case h
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      f : (CompHausLike.LocallyConstant.functorToPresheaves.obj (Y.obj { unop := Com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.functo …
    -/
    apply presheaf_ext (f.comap g.unop)
    /-
      case h.h
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      f : (CompHausLike.LocallyConstant.functorToPresheaves.obj (Y.obj { unop := Com …
      ⊢ ∀ (a : Function.Fiber ⇑(LocallyConstant.comap g.unop f)), Eq (Y.map (CompHau …
    -/
    intro a
    simp only [op_unop, functorToPresheaves_obj_obj, types_comp_apply, functorToPresheaves_obj_map,
      incl_of_counitAppApp, ← FunctorToTypes.map_comp_apply, incl_comap]
    /-
      case h.h
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      f : (CompHausLike.LocallyConstant.functorToPresheaves.obj (Y.obj { unop := Com …
      a : Function.Fiber ⇑(LocallyConstant.comap g.unop f)
      ⊢ Eq (CompHausLike.LocallyConstant.counitAppAppImage (LocallyConstant.comap g. …
    -/
    simp only [FunctorToTypes.map_comp_apply, incl_of_counitAppApp]
    simp only [counitAppAppImage, ← FunctorToTypes.map_comp_apply, ← op_comp,
      terminal.comp_from]
    /-
      case h.h
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      f : (CompHausLike.LocallyConstant.functorToPresheaves.obj (Y.obj { unop := Com …
      a : Function.Fiber ⇑(LocallyConstant.comap g.unop f)
      ⊢ Eq (Y.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConstant.f …
    -/
    apply congrArg
    /-
      case h.h.h
      P : TopCat → Prop
      inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T✝ : CompHausLike P
      g✝ : Quiver.Hom T✝ S✝
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts Y
      S T : Opposite (CompHausLike P)
      g : Quiver.Hom S T
      f : (CompHausLike.LocallyConstant.functorToPresheaves.obj (Y.obj { unop := Com …
      a : Function.Fiber ⇑(LocallyConstant.comap g.unop f)
      ⊢ Eq (Function.Fiber.image (⇑(LocallyConstant.comap g.unop f)) a) (Function.Fi …
    -/
    exact image_eq_image_mk (g := g.unop) (a := a)
    /-
      🎉 no goals
    -/


/-- `locallyConstantIsoContinuousMap` is a natural isomorphism. -/
noncomputable def functorToPresheavesIso (X : Type (max u w)) :
    functorToPresheaves.{u, w}.obj X ≅ ((TopCat.discrete.obj X).toSheafCompHausLike P hs).val :=
  /-
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    T : CompHausLike P
    g : Quiver.Hom T S
    X✝ : TopCat
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    X : Type (max u w)
    ⊢ ∀ {X_1 Y : Opposite (CompHausLike P)} (f : Quiver.Hom X_1 Y), Eq (CategoryTh …
  -/
  NatIso.ofComponents (fun S ↦ locallyConstantIsoContinuousMap _ _)
  /-
    🎉 no goals
  -/


/-- `CompHausLike.LocallyConstant.functorToPresheaves` lands in sheaves. -/
@[simps]
def functor :
    haveI := CompHausLike.preregular hs
    Type (max u w) ⥤ Sheaf (coherentTopology (CompHausLike.{u} P)) (Type (max u w)) where
  obj X := {
    val := functorToPresheaves.{u, w}.obj X
    cond := by
      /-
        P : TopCat → Prop
        inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
        S : CompHausLike P
        Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
        inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
        f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
        T : CompHausLike P
        g : Quiver.Hom T S
        X✝ : TopCat
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacks P
        hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
        X : Type (max u w)
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
      -/
      rw [Presheaf.isSheaf_of_iso_iff (functorToPresheavesIso P hs X)]
      /-
        P : TopCat → Prop
        inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
        S : CompHausLike P
        Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
        inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
        f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
        T : CompHausLike P
        g : Quiver.Hom T S
        X✝ : TopCat
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacks P
        hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
        X : Type (max u w)
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
      -/
      exact ((TopCat.discrete.obj X).toSheafCompHausLike P hs).cond }
      /-
        🎉 no goals
      -/
  map f := ⟨functorToPresheaves.{u, w}.map f⟩


/--
`CompHausLike.LocallyConstant.functor` is naturally isomorphic to the restriction of
`topCatToSheafCompHausLike` to discrete topological spaces.
-/
noncomputable def functorIso :
    functor.{u, w} P hs ≅ TopCat.discrete.{max w u} ⋙ topCatToSheafCompHausLike P hs :=
  /-
    P : TopCat → Prop
    inst✝³ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    S : CompHausLike P
    Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝² : CompHausLike.HasProp P PUnit.{u + 1}
    f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
    T : CompHausLike P
    g : Quiver.Hom T S
    X : TopCat
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    ⊢ ∀ {X Y : Type (max u w)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategorySt …
  -/
  NatIso.ofComponents (fun X ↦ (fullyFaithfulSheafToPresheaf _ _).preimageIso
  /-
    🎉 no goals
  -/
    (functorToPresheavesIso P hs X))


/-- The counit is natural in both `S : CompHausLike P` and
`Y : Sheaf (coherentTopology (CompHausLike P)) (Type (max u w))` -/
@[simps]
noncomputable def counit [HasExplicitFiniteCoproducts.{u} P] : haveI := CompHausLike.preregular hs
    (sheafSections _ _).obj ⟨CompHausLike.of P PUnit.{u+1}⟩ ⋙ functor.{u, w} P hs ⟶
        𝟭 (Sheaf (coherentTopology (CompHausLike.{u} P)) (Type (max u w))) where
  app X := haveI := CompHausLike.preregular hs
    ⟨counitApp X.val⟩
  naturality X Y g := by
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u +  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.sheafSections (Cat …
    -/
    have := CompHausLike.preregular hs
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u +  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.sheafSections (Cat …
    -/
    apply Sheaf.hom_ext
    simp only [functor, id_eq, eq_mpr_eq_cast, Functor.comp_obj, Functor.flip_obj_obj,
      sheafToPresheaf_obj, Functor.id_obj, Functor.comp_map, Functor.flip_obj_map,
      sheafToPresheaf_map, Sheaf.instCategorySheaf_comp_val, Functor.id_map]
    /-
      case h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u +  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.LocallyConstant.functor …
    -/
    ext S (f : LocallyConstant _ _)
    /-
      case h.w.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CompHausLike.LocallyConstant.functo …
    -/
    simp only [FunctorToTypes.comp, counitApp_app]
    /-
      case h.w.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      ⊢ Eq (CompHausLike.LocallyConstant.counitAppApp (Opposite.unop S) Y.val ((Comp …
    -/
    apply presheaf_ext (f.map (g.val.app (op (CompHausLike.of P PUnit.{u+1}))))
    /-
      case h.w.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      ⊢ ∀ (a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLik …
    -/
    intro a
    /-
      case h.w.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      ⊢ Eq (Y.val.map (CompHausLike.LocallyConstant.sigmaIncl (LocallyConstant.map ( …
    -/
    simp only [op_unop, functorToPresheaves_map_app, incl_of_counitAppApp]
    /-
      case h.w.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      ⊢ Eq (CompHausLike.LocallyConstant.counitAppAppImage (LocallyConstant.map (g.v …
    -/
    apply presheaf_ext (f.comap (sigmaIncl _ _))
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      ⊢ ∀ (a_1 : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstan …
    -/
    intro b
    simp only [counitAppAppImage, ← FunctorToTypes.map_comp_apply, ← op_comp, CompHausLike.coe_of,
      map_apply, IsTerminal.comp_from, ← map_preimage_eq_image_map]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (Y.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    change (_ ≫ Y.val.map _) _ = (_ ≫ Y.val.map _) _
    simp only [← g.val.naturality,
      show sigmaIncl (f.comap (sigmaIncl (f.map _) a)) b ≫ sigmaIncl (f.map _) a =
        (sigmaInclIncl f _ a b) ≫ sigmaIncl f (Fiber.mk f _) from rfl]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.val.map (CompHausLike.isTerminalPU …
    -/
    simp only [op_comp, Functor.map_comp, types_comp_apply, incl_of_counitAppApp]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (g.val.app { unop := CompHausLike.LocallyConstant.fiber (LocallyConstant. …
    -/
    simp only [counitAppAppImage, ← FunctorToTypes.map_comp_apply, ← op_comp, terminal.comp_from]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (g.val.app { unop := CompHausLike.LocallyConstant.fiber (LocallyConstant. …
    -/
    rw [mk_image]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (g.val.app { unop := CompHausLike.LocallyConstant.fiber (LocallyConstant. …
    -/
    change (X.val.map _ ≫ _) _ = (X.val.map _ ≫ _) _
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.val.map (CompHausLike.isTerminalPU …
    -/
    simp only [g.val.naturality]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.val.app { unop := CompHausLike.of  …
    -/
    simp only [types_comp_apply]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      ⊢ Eq (Y.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    have := map_preimage_eq_image (f := g.val.app _ ∘ f) (a := a)
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (Function.comp (g.val.app { unop := CompHausLike.of P PUnit.{u + 1}  …
      ⊢ Eq (Y.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    simp only [Function.comp_apply] at this
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f (Function. …
      ⊢ Eq (Y.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    rw [this]
    /-
      case h.w.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f (Function. …
      ⊢ Eq (Y.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    apply congrArg
    /-
      case h.w.h.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f (Function. …
      ⊢ Eq (Function.Fiber.image (Function.comp (g.val.app { unop := CompHausLike.of …
    -/
    symm
    /-
      case h.w.h.h.h.h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f (Function. …
      ⊢ Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f ↑(Function.Fibe …
    -/
    convert (b.preimage).prop
    /-
      case a
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S✝ : CompHausLike P
      Y✝ : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f✝ : LocallyConstant (↑S✝.toTop) (Y✝.obj { unop := CompHausLike.of P PUnit.{u  …
      T : CompHausLike P
      g✝ : Quiver.Hom T S✝
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P))  …
      g : Quiver.Hom X Y
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      S : Opposite (CompHausLike P)
      f : LocallyConstant (↑(Opposite.unop S).toTop) (X.val.obj { unop := CompHausLi …
      a : Function.Fiber ⇑(LocallyConstant.map (g.val.app { unop := CompHausLike.of  …
      b : Function.Fiber ⇑(LocallyConstant.comap (CompHausLike.LocallyConstant.sigma …
      this : Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f (Function. …
      ⊢ Iff (Eq (g.val.app { unop := CompHausLike.of P PUnit.{u + 1} } (f ↑(Function …
    -/
    exact (mem_iff_eq_image (g.val.app _ ∘ f) _ _).symm
    /-
      🎉 no goals
    -/


/--
The unit of the adjunciton is given by mapping each element to the corresponding constant map.
-/
@[simps]
def unit : 𝟭 _ ⟶ functor P hs ⋙ (sheafSections _ _).obj ⟨CompHausLike.of P PUnit.{u+1}⟩ where
  app _ x := LocallyConstant.const _ x


/-- The unit of the adjunction is an iso. -/
noncomputable def unitIso : 𝟭 (Type max u w) ≅ functor.{u, w} P hs ⋙
    (sheafSections _ _).obj ⟨CompHausLike.of P PUnit.{u+1}⟩ where
  hom := unit P hs
  inv := { app := fun _ f ↦ f.toFun PUnit.unit }


lemma adjunction_left_triangle [HasExplicitFiniteCoproducts.{u} P]
    (X : Type max u w) : functorToPresheaves.{u, w}.map ((unit P hs).app X) ≫
      ((counit P hs).app ((functor P hs).obj X)).val = 𝟙 (functorToPresheaves.obj X) := by
  /-
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.LocallyConstant.functor …
  -/
  ext ⟨S⟩ (f : LocallyConstant _ X)
  simp only [Functor.id_obj, Functor.comp_obj, FunctorToTypes.comp, NatTrans.id_app,
    functorToPresheaves_obj_obj, types_id_apply]
  /-
    case w.h.op.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    ⊢ Eq (((CompHausLike.LocallyConstant.counit P hs).app ((CompHausLike.LocallyCo …
  -/
  simp only [counit, counitApp_app]
  /-
    case w.h.op.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    ⊢ Eq (CompHausLike.LocallyConstant.counitAppApp S ((CompHausLike.LocallyConsta …
  -/
  have := CompHausLike.preregular hs
  apply presheaf_ext
    (X := ((functor P hs).obj X).val) (Y := ((functor.{u, w} P hs).obj X).val)
      (f.map ((unit P hs).app X))
  /-
    case w.h.op.h.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    this : CategoryTheory.Preregular (CompHausLike P)
    ⊢ ∀ (a : Function.Fiber ⇑(LocallyConstant.map ((CompHausLike.LocallyConstant.u …
  -/
  intro a
  /-
    case w.h.op.h.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    this : CategoryTheory.Preregular (CompHausLike P)
    a : Function.Fiber ⇑(LocallyConstant.map ((CompHausLike.LocallyConstant.unit P …
    ⊢ Eq (((CompHausLike.LocallyConstant.functor P hs).obj X).val.map (CompHausLik …
  -/
  erw [incl_of_counitAppApp]
  simp only [functor_obj_val, functorToPresheaves_obj_obj, coe_of, Functor.id_obj,
    counitAppAppImage, LocallyConstant.map_apply, functorToPresheaves_obj_map, Quiver.Hom.unop_op]
  /-
    case w.h.op.h.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    this : CategoryTheory.Preregular (CompHausLike P)
    a : Function.Fiber ⇑(LocallyConstant.map ((CompHausLike.LocallyConstant.unit P …
    ⊢ Eq (LocallyConstant.comap (CompHausLike.isTerminalPUnit.from (CompHausLike.L …
  -/
  ext x
  /-
    case w.h.op.h.h.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    this : CategoryTheory.Preregular (CompHausLike P)
    a : Function.Fiber ⇑(LocallyConstant.map ((CompHausLike.LocallyConstant.unit P …
    x : ↑(CompHausLike.LocallyConstant.fiber (LocallyConstant.map ((CompHausLike.L …
    ⊢ Eq ((LocallyConstant.comap (CompHausLike.isTerminalPUnit.from (CompHausLike. …
  -/
  erw [← map_eq_image _ a x]
  /-
    case w.h.op.h.h.h
    P : TopCat → Prop
    inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
    inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
    inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    X : Type (max u w)
    S : CompHausLike P
    f : LocallyConstant (↑S.toTop) X
    this : CategoryTheory.Preregular (CompHausLike P)
    a : Function.Fiber ⇑(LocallyConstant.map ((CompHausLike.LocallyConstant.unit P …
    x : ↑(CompHausLike.LocallyConstant.fiber (LocallyConstant.map ((CompHausLike.L …
    ⊢ Eq ((LocallyConstant.comap (CompHausLike.isTerminalPUnit.from (CompHausLike. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
`CompHausLike.LocallyConstant.functor` is left adjoint to the forgetful functor.
-/
@[simps]
noncomputable def adjunction [HasExplicitFiniteCoproducts.{u} P] :
    functor.{u, w} P hs ⊣ (sheafSections _ _).obj ⟨CompHausLike.of P PUnit.{u+1}⟩ where
  unit := unit P hs
  counit := counit P hs
  left_triangle_components := by
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      ⊢ ∀ (X : Type (max u w)), Eq (CategoryTheory.CategoryStruct.comp ((CompHausLik …
    -/
    intro X
    simp only [Functor.comp_obj, Functor.id_obj, NatTrans.comp_app, Functor.flip_obj_obj,
      sheafToPresheaf_obj, functor_obj_val, functorToPresheaves_obj_obj, coe_of, whiskerRight_app,
      Functor.associator_hom_app, whiskerLeft_app, Category.id_comp, NatTrans.id_app']
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : Type (max u w)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.functo …
    -/
    apply Sheaf.hom_ext
    /-
      case h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : Type (max u w)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.functo …
    -/
    rw [Sheaf.instCategorySheaf_comp_val, Sheaf.instCategorySheaf_id_val]
    /-
      case h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : Type (max u w)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.functo …
    -/
    exact adjunction_left_triangle P hs X
    /-
      🎉 no goals
    -/
  right_triangle_components := by
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      ⊢ ∀ (Y : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P …
    -/
    intro X
    /-
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.unit P …
    -/
    ext (x : X.val.obj _)
    simp only [Functor.comp_obj, Functor.id_obj, Functor.flip_obj_obj, sheafToPresheaf_obj,
      FunctorToTypes.comp, whiskerLeft_app, unit_app, coe_of, Functor.associator_inv_app,
      functor_obj_val, functorToPresheaves_obj_obj, types_id_apply, whiskerRight_app,
      Functor.flip_obj_map, sheafToPresheaf_map, counit_app_val, counitApp_app, NatTrans.id_app']
    /-
      case h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.unit P …
    -/
    have := CompHausLike.preregular hs
    letI : PreservesFiniteProducts ((sheafToPresheaf (coherentTopology _) _).obj X) :=
      inferInstanceAs (PreservesFiniteProducts (Sheaf.val _))
    /-
      case h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CompHausLike.LocallyConstant.unit P …
    -/
    apply presheaf_ext ((unit P hs).app _ x)
    /-
      case h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      ⊢ ∀ (a : Function.Fiber ⇑((CompHausLike.LocallyConstant.unit P hs).app (X.val. …
    -/
    intro a
    /-
      case h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      a : Function.Fiber ⇑((CompHausLike.LocallyConstant.unit P hs).app (X.val.obj { …
      ⊢ Eq (X.val.map (CompHausLike.LocallyConstant.sigmaIncl ((CompHausLike.Locally …
    -/
    erw [incl_of_counitAppApp]
    /-
      case h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      a : Function.Fiber ⇑((CompHausLike.LocallyConstant.unit P hs).app (X.val.obj { …
      ⊢ Eq (CompHausLike.LocallyConstant.counitAppAppImage ((CompHausLike.LocallyCon …
    -/
    simp only [sheafToPresheaf_obj, unit_app, coe_of, counitAppAppImage, coe_const]
    /-
      case h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      a : Function.Fiber ⇑((CompHausLike.LocallyConstant.unit P hs).app (X.val.obj { …
      ⊢ Eq (X.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    erw [← map_eq_image _ a ⟨PUnit.unit, by simp [mem_iff_eq_image, ← map_preimage_eq_image]⟩]
    /-
      case h.h
      P : TopCat → Prop
      inst✝⁴ : ∀ (S : CompHausLike P) (p : ↑S.toTop → Prop), CompHausLike.HasProp P  …
      S : CompHausLike P
      Y : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
      inst✝³ : CompHausLike.HasProp P PUnit.{u + 1}
      f : LocallyConstant (↑S.toTop) (Y.obj { unop := CompHausLike.of P PUnit.{u + 1 …
      T : CompHausLike P
      g : Quiver.Hom T S
      X✝ : TopCat
      inst✝² : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝¹ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      X : CategoryTheory.Sheaf (CategoryTheory.coherentTopology (CompHausLike P)) (T …
      x : X.val.obj { unop := CompHausLike.of P PUnit.{u + 1} }
      this✝ : CategoryTheory.Preregular (CompHausLike P)
      this : CategoryTheory.Limits.PreservesFiniteProducts ((CategoryTheory.sheafToP …
      a : Function.Fiber ⇑((CompHausLike.LocallyConstant.unit P hs).app (X.val.obj { …
      ⊢ Eq (X.val.map (CompHausLike.isTerminalPUnit.from (CompHausLike.LocallyConsta …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance [HasExplicitFiniteCoproducts.{u} P] : IsIso (adjunction P hs).unit :=
  inferInstanceAs (IsIso (unitIso P hs).hom)


/-- The functor from sets to condensed sets given by locally constant maps into the set. -/
abbrev functor : Type (u+1) ⥤ CondensedSet.{u} :=
  CompHausLike.LocallyConstant.functor.{u, u+1} (P := fun _ ↦ True)
                        /-
                          P : TopCat → Prop
                          x✝² x✝¹ : CompHausLike fun x => True
                          x✝ : Quiver.Hom x✝² x✝¹
                          ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.198295 x✝² x✝¹ x✝)) (List.co …
                        -/
                        /-
                          🎉 no goals
                        -/
    (hs := fun _ _ _ ↦ ((CompHaus.effectiveEpi_tfae _).out 0 2).mp)
                        /-
                          🎉 no goals
                        -/


/--
`CondensedSet.LocallyConstant.functor` is isomorphic to `Condensed.discrete`
(by uniqueness of adjoints).
-/
noncomputable def iso : functor ≅ discrete (Type (u+1)) :=
  (LocallyConstant.adjunction _ _).leftAdjointUniq (discreteUnderlyingAdj _)


/-- `CondensedSet.LocallyConstant.functor` is fully faithful. -/
noncomputable def functorFullyFaithful : functor.FullyFaithful :=
  (LocallyConstant.adjunction.{u, u+1} _ _).fullyFaithfulLOfIsIsoUnit


noncomputable instance : functor.Faithful := functorFullyFaithful.faithful


noncomputable instance : functor.Full := functorFullyFaithful.full


instance : (discrete (Type _)).Faithful := Functor.Faithful.of_iso iso


noncomputable instance : (discrete (Type _)).Full := Functor.Full.of_iso iso


/-- The functor from sets to light condensed sets given by locally constant maps into the set. -/
abbrev functor : Type u ⥤ LightCondSet.{u} :=
  CompHausLike.LocallyConstant.functor.{u, u}
    (P := fun X ↦ TotallyDisconnectedSpace X ∧ SecondCountableTopology X)
    (hs := fun _ _ _ ↦ (LightProfinite.effectiveEpi_iff_surjective _).mp)


instance (S : LightProfinite.{u}) (p : S → Prop) :
    HasProp (fun X ↦ TotallyDisconnectedSpace X ∧ SecondCountableTopology X) (Subtype p) :=
  ⟨⟨(inferInstance : TotallyDisconnectedSpace (Subtype p)),
    (inferInstance : SecondCountableTopology {s | p s})⟩⟩


/--
`LightCondSet.LocallyConstant.functor` is isomorphic to `LightCondensed.discrete`
(by uniqueness of adjoints).
-/
noncomputable def iso : functor ≅ LightCondensed.discrete (Type u) :=
  (LocallyConstant.adjunction _ _).leftAdjointUniq (LightCondensed.discreteUnderlyingAdj _)


/-- `LightCondSet.LocallyConstant.functor` is fully faithful. -/
noncomputable def functorFullyFaithful : functor.{u}.FullyFaithful :=
  (LocallyConstant.adjunction _ _).fullyFaithfulLOfIsIsoUnit


instance : functor.{u}.Faithful := functorFullyFaithful.faithful


instance : LightCondSet.LocallyConstant.functor.Full := functorFullyFaithful.full


instance : (LightCondensed.discrete (Type u)).Faithful := Functor.Faithful.of_iso iso.{u}


instance : (LightCondensed.discrete (Type u)).Full := Functor.Full.of_iso iso.{u}


