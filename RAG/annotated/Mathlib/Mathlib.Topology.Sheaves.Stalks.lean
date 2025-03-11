/-- Stalks are functorial with respect to morphisms of presheaves over a fixed `X`. -/
def stalkFunctor (x : X) : X.Presheaf C ⥤ C :=
  (whiskeringLeft _ _ C).obj (OpenNhds.inclusion x).op ⋙ colim


/-- The stalk of a presheaf `F` at a point `x` is calculated as the colimit of the functor
nbhds x ⥤ opens F.X ⥤ C
-/
def stalk (ℱ : X.Presheaf C) (x : X) : C :=
  (stalkFunctor C x).obj ℱ

-- -- colimit ((open_nhds.inclusion x).op ⋙ ℱ)

@[simp]
theorem stalkFunctor_obj (ℱ : X.Presheaf C) (x : X) : (stalkFunctor C x).obj ℱ = ℱ.stalk x :=
  rfl


/-- The germ of a section of a presheaf over an open at a point of that open.
-/
def germ (F : X.Presheaf C) (U : Opens X) (x : X) (hx : x ∈ U) : F.obj (op U) ⟶ stalk F x :=
  colimit.ι ((OpenNhds.inclusion x).op ⋙ F) (op ⟨U, hx⟩)


/-- The germ of a global section of a presheaf at a point. -/
def Γgerm (F : X.Presheaf C) (x : X) : F.obj (op ⊤) ⟶ stalk F x :=
  F.germ ⊤ x True.intro


@[reassoc]
theorem germ_res (F : X.Presheaf C) {U V : Opens X} (i : U ⟶ V) (x : X) (hx : x ∈ U) :
    F.map i.op ≫ F.germ U x hx = F.germ V x (i.le hx) :=
  let i' : (⟨U, hx⟩ : OpenNhds x) ⟶ ⟨V, i.le hx⟩ := i
  colimit.w ((OpenNhds.inclusion x).op ⋙ F) i'.op


/-- A variant of `germ_res` with `op V ⟶ op U`
so that the LHS is more general and simp fires more easier. -/
@[reassoc (attr := simp)]
theorem germ_res' (F : X.Presheaf C) {U V : Opens X} (i : op V ⟶ op U) (x : X) (hx : x ∈ U) :
    F.map i ≫ F.germ U x hx = F.germ V x (i.unop.le hx) :=
  let i' : (⟨U, hx⟩ : OpenNhds x) ⟶ ⟨V, i.unop.le hx⟩ := i.unop
  colimit.w ((OpenNhds.inclusion x).op ⋙ F) i'.op


@[reassoc]
lemma map_germ_eq_Γgerm (F : X.Presheaf C) {U : Opens X} {i : U ⟶ ⊤} (x : X) (hx : x ∈ U) :
    F.map i.op ≫ F.germ U x hx = F.Γgerm x :=
  germ_res F i x hx


attribute [local instance] ConcreteCategory.instFunLike in
theorem germ_res_apply (F : X.Presheaf C)
    {U V : Opens X} (i : U ⟶ V) (x : X) (hx : x ∈ U) [ConcreteCategory C] (s) :
                                                              /-
                                                                C : Type u
                                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                                inst✝¹ : CategoryTheory.Limits.HasColimits C
                                                                X : TopCat
                                                                F : TopCat.Presheaf C X
                                                                U V : TopologicalSpace.Opens ↑X
                                                                i : Quiver.Hom U V
                                                                x : ↑X
                                                                hx : Membership.mem U x
                                                                inst✝ : CategoryTheory.ConcreteCategory C
                                                                s : (CategoryTheory.forget C).obj (F.obj { unop := V })
                                                                ⊢ Eq ((F.germ U x hx) ((F.map i.op) s)) ((F.germ V x ⋯) s)
                                                              -/
  F.germ U x hx (F.map i.op s) = F.germ V x (i.le hx) s := by rw [← comp_apply, germ_res]
                                                              /-
                                                                🎉 no goals
                                                              -/


attribute [local instance] ConcreteCategory.instFunLike in
theorem germ_res_apply' (F : X.Presheaf C)
    {U V : Opens X} (i : op V ⟶ op U) (x : X) (hx : x ∈ U) [ConcreteCategory C] (s) :
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  inst✝¹ : CategoryTheory.Limits.HasColimits C
                                                                  X : TopCat
                                                                  F : TopCat.Presheaf C X
                                                                  U V : TopologicalSpace.Opens ↑X
                                                                  i : Quiver.Hom { unop := V } { unop := U }
                                                                  x : ↑X
                                                                  hx : Membership.mem U x
                                                                  inst✝ : CategoryTheory.ConcreteCategory C
                                                                  s : (CategoryTheory.forget C).obj (F.obj { unop := V })
                                                                  ⊢ Eq ((F.germ U x hx) ((F.map i) s)) ((F.germ V x ⋯) s)
                                                                -/
  F.germ U x hx (F.map i s) = F.germ V x (i.unop.le hx) s := by rw [← comp_apply, germ_res']
                                                                /-
                                                                  🎉 no goals
                                                                -/


attribute [local instance] ConcreteCategory.instFunLike in
lemma Γgerm_res_apply (F : X.Presheaf C)
    {U : Opens X} {i : U ⟶ ⊤} (x : X) (hx : x ∈ U) [ConcreteCategory C] (s) :
  F.germ U x hx (F.map i.op s) = F.Γgerm x s := F.germ_res_apply i x hx s


/-- A morphism from the stalk of `F` at `x` to some object `Y` is completely determined by its
composition with the `germ` morphisms.
-/
@[ext]
theorem stalk_hom_ext (F : X.Presheaf C) {x} {Y : C} {f₁ f₂ : F.stalk x ⟶ Y}
    (ih : ∀ (U : Opens X) (hxU : x ∈ U), F.germ U x hxU ≫ f₁ = F.germ U x hxU ≫ f₂) : f₁ = f₂ :=
  colimit.hom_ext fun U => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      F : TopCat.Presheaf C X
      x : ↑X
      Y : C
      f₁ f₂ : Quiver.Hom (F.stalk x) Y
      ih : ∀ (U : TopologicalSpace.Opens ↑X) (hxU : Membership.mem U x), Eq (Categor …
      U : Opposite (TopologicalSpace.OpenNhds x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (((C …
    -/
    induction' U using Opposite.rec with U; cases' U with U hxU; exact ih U hxU
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc (attr := simp)]
theorem stalkFunctor_map_germ {F G : X.Presheaf C} (U : Opens X) (x : X) (hx : x ∈ U) (f : F ⟶ G) :
    F.germ U x hx ≫ (stalkFunctor C x).map f = f.app (op U) ≫ G.germ U x hx :=
  colimit.ι_map (whiskerLeft (OpenNhds.inclusion x).op f) (op ⟨U, hx⟩)


attribute [local instance] ConcreteCategory.instFunLike in
theorem stalkFunctor_map_germ_apply [ConcreteCategory C]
    {F G : X.Presheaf C} (U : Opens X) (x : X) (hx : x ∈ U) (f : F ⟶ G) (s) :
    (stalkFunctor C x).map f (F.germ U x hx s) = G.germ U x hx (f.app (op U) s) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝ : CategoryTheory.ConcreteCategory C
    F G : TopCat.Presheaf C X
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem U x
    f : Quiver.Hom F G
    s : (CategoryTheory.forget C).obj (F.obj { unop := U })
    ⊢ Eq (((TopCat.Presheaf.stalkFunctor C x).map f) ((F.germ U x hx) s)) ((G.germ …
  -/
  rw [← comp_apply, ← stalkFunctor_map_germ]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝ : CategoryTheory.ConcreteCategory C
    F G : TopCat.Presheaf C X
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem U x
    f : Quiver.Hom F G
    s : (CategoryTheory.forget C).obj (F.obj { unop := U })
    ⊢ Eq (((TopCat.Presheaf.stalkFunctor C x).map f) ((F.germ U x hx) s)) ((Catego …
  -/
  exact (comp_apply _ _ _).symm
  /-
    🎉 no goals
  -/

-- a variant of `stalkFunctor_map_germ_apply` that makes simpNF happy.

attribute [local instance] ConcreteCategory.instFunLike in
@[simp]
theorem stalkFunctor_map_germ_apply' [ConcreteCategory C]
    {F G : X.Presheaf C} (U : Opens X) (x : X) (hx : x ∈ U) (f : F ⟶ G) (s) :
    DFunLike.coe (F := F.stalk x ⟶ G.stalk x) ((stalkFunctor C x).map f) (F.germ U x hx s) =
      G.germ U x hx (f.app (op U) s) :=
  stalkFunctor_map_germ_apply U x hx f s


/-- For a presheaf `F` on a space `X`, a continuous map `f : X ⟶ Y` induces a morphisms between the
stalk of `f _ * F` at `f x` and the stalk of `F` at `x`.
-/
def stalkPushforward (f : X ⟶ Y) (F : X.Presheaf C) (x : X) : (f _* F).stalk (f x) ⟶ F.stalk x := by
  -- This is a hack; Lean doesn't like to elaborate the term written directly.
  -- Porting note: The original proof was `trans; swap`, but `trans` does nothing.
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y Z : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    x : ↑X
    ⊢ Quiver.Hom (((TopCat.Presheaf.pushforward C f).obj F).stalk (f x)) (F.stalk x)
  -/
  refine ?_ ≫ colimit.pre _ (OpenNhds.map f x).op
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y Z : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    x : ↑X
    ⊢ Quiver.Hom (((TopCat.Presheaf.pushforward C f).obj F).stalk (f x)) (Category …
  -/
  exact colim.map (whiskerRight (NatTrans.op (OpenNhds.inclusionMapIso f x).inv) F)
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem stalkPushforward_germ (f : X ⟶ Y) (F : X.Presheaf C) (U : Opens Y)
    (x : X) (hx : f x ∈ U) :
      (f _* F).germ U (f x) hx ≫ F.stalkPushforward C f x = F.germ ((Opens.map f).obj U) x hx := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    U : TopologicalSpace.Opens ↑Y
    x : ↑X
    hx : Membership.mem U (f x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C f).o …
  -/
  simp [germ, stalkPushforward]
  /-
    🎉 no goals
  -/

-- Here are two other potential solutions, suggested by @fpvandoorn at
-- <https://github.com/leanprover-community/mathlib/pull/1018#discussion_r283978240>
-- However, I can't get the subsequent two proofs to work with either one.
-- def stalkPushforward'' (f : X ⟶ Y) (ℱ : X.Presheaf C) (x : X) :
--   (f _* ℱ).stalk (f x) ⟶ ℱ.stalk x :=
-- colim.map ((Functor.associator _ _ _).inv ≫
--   whiskerRight (NatTrans.op (OpenNhds.inclusionMapIso f x).inv) ℱ) ≫
-- colimit.pre ((OpenNhds.inclusion x).op ⋙ ℱ) (OpenNhds.map f x).op
-- def stalkPushforward''' (f : X ⟶ Y) (ℱ : X.Presheaf C) (x : X) :
--   (f _* ℱ).stalk (f x) ⟶ ℱ.stalk x :=
-- (colim.map (whiskerRight (NatTrans.op (OpenNhds.inclusionMapIso f x).inv) ℱ) :
--   colim.obj ((OpenNhds.inclusion (f x) ⋙ Opens.map f).op ⋙ ℱ) ⟶ _) ≫
-- colimit.pre ((OpenNhds.inclusion x).op ⋙ ℱ) (OpenNhds.map f x).op


@[simp]
theorem id (ℱ : X.Presheaf C) (x : X) :
    ℱ.stalkPushforward C (𝟙 X) x = (stalkFunctor C x).map (Pushforward.id ℱ).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    ℱ : TopCat.Presheaf C X
    x : ↑X
    ⊢ Eq (TopCat.Presheaf.stalkPushforward C (CategoryTheory.CategoryStruct.id X)  …
  -/
  ext
  /-
    case ih
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    ℱ : TopCat.Presheaf C X
    x : ↑X
    U✝ : TopologicalSpace.Opens ↑X
    hxU✝ : Membership.mem U✝ ((CategoryTheory.CategoryStruct.id X) x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C (Cat …
  -/
  simp only [stalkPushforward, germ, colim_map, ι_colimMap_assoc, whiskerRight_app]
  /-
    case ih
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    ℱ : TopCat.Presheaf C X
    x : ↑X
    U✝ : TopologicalSpace.Opens ↑X
    hxU✝ : Membership.mem U✝ ((CategoryTheory.CategoryStruct.id X) x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ℱ.map ((CategoryTheory.NatTrans.op ( …
  -/
  erw [CategoryTheory.Functor.map_id]
  /-
    case ih
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    ℱ : TopCat.Presheaf C X
    x : ↑X
    U✝ : TopologicalSpace.Opens ↑X
    hxU✝ : Membership.mem U✝ ((CategoryTheory.CategoryStruct.id X) x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (ℱ. …
  -/
  simp [stalkFunctor]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp (ℱ : X.Presheaf C) (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :
    ℱ.stalkPushforward C (f ≫ g) x =
      (f _* ℱ).stalkPushforward C g (f x) ≫ ℱ.stalkPushforward C f x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y Z : TopCat
    ℱ : TopCat.Presheaf C X
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    x : ↑X
    ⊢ Eq (TopCat.Presheaf.stalkPushforward C (CategoryTheory.CategoryStruct.comp f …
  -/
  ext
  /-
    case ih
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y Z : TopCat
    ℱ : TopCat.Presheaf C X
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    x : ↑X
    U✝ : TopologicalSpace.Opens ↑Z
    hxU✝ : Membership.mem U✝ ((CategoryTheory.CategoryStruct.comp f g) x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C (Cat …
  -/
  simp [germ, stalkPushforward]
  /-
    🎉 no goals
  -/


theorem stalkPushforward_iso_of_isInducing {f : X ⟶ Y} (hf : IsInducing f)
    (F : X.Presheaf C) (x : X) : IsIso (F.stalkPushforward _ f x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    F : TopCat.Presheaf C X
    x : ↑X
    ⊢ CategoryTheory.IsIso (TopCat.Presheaf.stalkPushforward C f F x)
  -/
  haveI := Functor.initial_of_adjunction (hf.adjunctionNhds x)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    F : TopCat.Presheaf C X
    x : ↑X
    this : (TopologicalSpace.OpenNhds.map f x).Initial
    ⊢ CategoryTheory.IsIso (TopCat.Presheaf.stalkPushforward C f F x)
  -/
  convert (Functor.Final.colimitIso (OpenNhds.map f x).op ((OpenNhds.inclusion x).op ⋙ F)).isIso_hom
  /-
    case h.e'_5.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    F : TopCat.Presheaf C X
    x : ↑X
    this : (TopologicalSpace.OpenNhds.map f x).Initial
    e_3✝ : Eq (((TopCat.Presheaf.pushforward C f).obj F).stalk (f x)) (CategoryThe …
    e_4✝ : Eq (F.stalk x) (CategoryTheory.Limits.colimit ((TopologicalSpace.OpenNh …
    ⊢ Eq (TopCat.Presheaf.stalkPushforward C f F x) (CategoryTheory.Functor.Final. …
  -/
  refine stalk_hom_ext _ fun U hU ↦ (stalkPushforward_germ _ f F _ x hU).trans ?_
  /-
    case h.e'_5.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    F : TopCat.Presheaf C X
    x : ↑X
    this : (TopologicalSpace.OpenNhds.map f x).Initial
    e_3✝ : Eq (((TopCat.Presheaf.pushforward C f).obj F).stalk (f x)) (CategoryThe …
    e_4✝ : Eq (F.stalk x) (CategoryTheory.Limits.colimit ((TopologicalSpace.OpenNh …
    U : TopologicalSpace.Opens ↑Y
    hU : Membership.mem U (f x)
    ⊢ Eq (F.germ ((TopologicalSpace.Opens.map f).obj U) x hU) (CategoryTheory.Cate …
  -/
  symm
  /-
    case h.e'_5.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    F : TopCat.Presheaf C X
    x : ↑X
    this : (TopologicalSpace.OpenNhds.map f x).Initial
    e_3✝ : Eq (((TopCat.Presheaf.pushforward C f).obj F).stalk (f x)) (CategoryThe …
    e_4✝ : Eq (F.stalk x) (CategoryTheory.Limits.colimit ((TopologicalSpace.OpenNh …
    U : TopologicalSpace.Opens ↑Y
    hU : Membership.mem U (f x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C f).o …
  -/
  exact colimit.ι_pre ((OpenNhds.inclusion x).op ⋙ F) (OpenNhds.map f x).op _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-27")]
alias stalkPushforward_iso_of_isOpenEmbedding := stalkPushforward_iso_of_isInducing

@[deprecated (since := "2024-10-18")]
alias stalkPushforward_iso_of_openEmbedding := stalkPushforward_iso_of_isInducing


/-- The morphism `ℱ_{f x} ⟶ (f⁻¹ℱ)ₓ` that factors through `(f_*f⁻¹ℱ)_{f x}`. -/
def stalkPullbackHom (f : X ⟶ Y) (F : Y.Presheaf C) (x : X) :
    F.stalk (f x) ⟶ ((pullback C f).obj F).stalk x :=
  (stalkFunctor _ (f x)).map ((pushforwardPullbackAdjunction C f).unit.app F) ≫
    stalkPushforward _ _ _ x


@[reassoc (attr := simp)]
lemma germ_stalkPullbackHom
    (f : X ⟶ Y) (F : Y.Presheaf C) (x : X) (U : Opens Y) (hU : f x ∈ U) :
    F.germ U (f x) hU ≫ stalkPullbackHom C f F x =
      ((pushforwardPullbackAdjunction C f).unit.app F).app _ ≫
        ((pullback C f).obj F).germ ((Opens.map f).obj U) x hU := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    x : ↑X
    U : TopologicalSpace.Opens ↑Y
    hU : Membership.mem U (f x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.germ U (f x) hU) (TopCat.Presheaf. …
  -/
  simp [stalkPullbackHom, germ, stalkFunctor, stalkPushforward]
  /-
    🎉 no goals
  -/


/-- The morphism `(f⁻¹ℱ)(U) ⟶ ℱ_{f(x)}` for some `U ∋ x`. -/
def germToPullbackStalk (f : X ⟶ Y) (F : Y.Presheaf C) (U : Opens X) (x : X) (hx : x ∈ U) :
    ((pullback C f).obj F).obj (op U) ⟶ F.stalk (f x) :=
  ((Opens.map f).op.isPointwiseLeftKanExtensionLeftKanExtensionUnit F (op U)).desc
    { pt := F.stalk ((f : X → Y) (x : X))
      ι :=
        { app := fun V => F.germ _ (f x) (V.hom.unop.le hx)
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.Limits.HasColimits C
                                          X Y Z : TopCat
                                          f : Quiver.Hom X Y
                                          F : TopCat.Presheaf C Y
                                          U : TopologicalSpace.Opens ↑X
                                          x : ↑X
                                          hx : Membership.mem U x
                                          x✝¹ x✝ : CategoryTheory.CostructuredArrow (TopologicalSpace.Opens.map f).op {  …
                                          i : Quiver.Hom x✝¹ x✝
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
                                        -/
          naturality := fun _ _ i => by simp } }
                                        /-
                                          🎉 no goals
                                        -/


variable {C} in
@[ext]
lemma pullback_obj_obj_ext {Z : C} {f : X ⟶ Y} {F : Y.Presheaf C} (U : (Opens X)ᵒᵖ)
    {φ ψ : ((pullback C f).obj F).obj U ⟶ Z}
    (h : ∀ (V : Opens Y) (hV : U.unop ≤ (Opens.map f).obj V),
      ((pushforwardPullbackAdjunction C f).unit.app F).app (op V) ≫
        ((pullback C f).obj F).map (homOfLE hV).op ≫ φ =
      ((pushforwardPullbackAdjunction C f).unit.app F).app (op V) ≫
        ((pullback C f).obj F).map (homOfLE hV).op ≫ ψ) : φ = ψ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    Z : C
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : Opposite (TopologicalSpace.Opens ↑X)
    φ ψ : Quiver.Hom (((TopCat.Presheaf.pullback C f).obj F).obj U) Z
    h : ∀ (V : TopologicalSpace.Opens ↑Y) (hV : LE.le (Opposite.unop U) ((Topologi …
    ⊢ Eq φ ψ
  -/
  obtain ⟨U⟩ := U
  /-
    case op
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    Z : C
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : TopologicalSpace.Opens ↑X
    φ ψ : Quiver.Hom (((TopCat.Presheaf.pullback C f).obj F).obj { unop := U }) Z
    h : ∀ (V : TopologicalSpace.Opens ↑Y) (hV : LE.le (Opposite.unop { unop := U } …
    ⊢ Eq φ ψ
  -/
  apply ((Opens.map f).op.isPointwiseLeftKanExtensionLeftKanExtensionUnit F _).hom_ext
  /-
    case op
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    Z : C
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : TopologicalSpace.Opens ↑X
    φ ψ : Quiver.Hom (((TopCat.Presheaf.pullback C f).obj F).obj { unop := U }) Z
    h : ∀ (V : TopologicalSpace.Opens ↑Y) (hV : LE.le (Opposite.unop { unop := U } …
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow (TopologicalSpace.Opens.map f).op {  …
  -/
  rintro ⟨⟨V⟩, ⟨⟩, ⟨b⟩⟩
  simpa [pushforwardPullbackAdjunction, Functor.lanAdjunction_unit]
    using h V (leOfHom b)


@[reassoc (attr := simp)]
lemma pushforwardPullbackAdjunction_unit_pullback_map_germToPullbackStalk
    (f : X ⟶ Y) (F : Y.Presheaf C) (U : Opens X) (x : X) (hx : x ∈ U) (V : Opens Y)
    (hV : U ≤ (Opens.map f).obj V) :
    ((pushforwardPullbackAdjunction C f).unit.app F).app (op V) ≫
      ((pullback C f).obj F).map (homOfLE hV).op ≫ germToPullbackStalk C f F U x hx  =
        F.germ _ (f x) (hV hx) := by
  simpa [pushforwardPullbackAdjunction] using
    ((Opens.map f).op.isPointwiseLeftKanExtensionLeftKanExtensionUnit F (op U)).fac _
      (CostructuredArrow.mk (homOfLE hV).op)


@[reassoc (attr := simp)]
lemma germToPullbackStalk_stalkPullbackHom
    (f : X ⟶ Y) (F : Y.Presheaf C) (U : Opens X) (x : X) (hx : x ∈ U) :
    germToPullbackStalk C f F U x hx ≫ stalkPullbackHom C f F x =
      ((pullback C f).obj F).germ _ x hx := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.germToPullbackStalk  …
  -/
  ext V hV
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem U x
    V : TopologicalSpace.Opens ↑Y
    hV : LE.le (Opposite.unop { unop := U }) ((TopologicalSpace.Opens.map f).obj V)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforwardPullbac …
  -/
  dsimp
  simp only [pushforwardPullbackAdjunction_unit_pullback_map_germToPullbackStalk_assoc,
    germ_stalkPullbackHom, germ_res]


@[reassoc (attr := simp)]
lemma pushforwardPullbackAdjunction_unit_app_app_germToPullbackStalk
    (f : X ⟶ Y) (F : Y.Presheaf C) (V : (Opens Y)ᵒᵖ) (x : X) (hx : f x ∈ V.unop) :
    ((pushforwardPullbackAdjunction C f).unit.app F).app V ≫ germToPullbackStalk C f F _ x hx =
      F.germ _ (f x) hx := by
  simpa using pushforwardPullbackAdjunction_unit_pullback_map_germToPullbackStalk
    C f F ((Opens.map f).obj V.unop) x hx V.unop (by rfl)


/-- The morphism `(f⁻¹ℱ)ₓ ⟶ ℱ_{f(x)}`. -/
def stalkPullbackInv (f : X ⟶ Y) (F : Y.Presheaf C) (x : X) :
    ((pullback C f).obj F).stalk x ⟶ F.stalk (f x) :=
  colimit.desc ((OpenNhds.inclusion x).op ⋙ (Presheaf.pullback C f).obj F)
    { pt := F.stalk (f x)
      ι :=
        { app := fun U => F.germToPullbackStalk _ f (unop U).1 x (unop U).2
          naturality := fun U V i => by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasColimits C
              X Y Z : TopCat
              f : Quiver.Hom X Y
              F : TopCat.Presheaf C Y
              x : ↑X
              U V : Opposite (TopologicalSpace.OpenNhds x)
              i : Quiver.Hom U V
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopologicalSpace.OpenNhds.inclusio …
            -/
            dsimp
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasColimits C
              X Y Z : TopCat
              f : Quiver.Hom X Y
              F : TopCat.Presheaf C Y
              x : ↑X
              U V : Opposite (TopologicalSpace.OpenNhds x)
              i : Quiver.Hom U V
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pullback C f).obj  …
            -/
            ext W hW
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasColimits C
              X Y Z : TopCat
              f : Quiver.Hom X Y
              F : TopCat.Presheaf C Y
              x : ↑X
              U V : Opposite (TopologicalSpace.OpenNhds x)
              i : Quiver.Hom U V
              W : TopologicalSpace.Opens ↑Y
              hW : LE.le (Opposite.unop { unop := (TopologicalSpace.OpenNhds.inclusion x).ob …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforwardPullbac …
            -/
            dsimp [OpenNhds.inclusion]
            rw [Category.comp_id, ← Functor.map_comp_assoc,
              pushforwardPullbackAdjunction_unit_pullback_map_germToPullbackStalk]
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasColimits C
              X Y Z : TopCat
              f : Quiver.Hom X Y
              F : TopCat.Presheaf C Y
              x : ↑X
              U V : Opposite (TopologicalSpace.OpenNhds x)
              i : Quiver.Hom U V
              W : TopologicalSpace.Opens ↑Y
              hW : LE.le (Opposite.unop { unop := (TopologicalSpace.OpenNhds.inclusion x).ob …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforwardPullbac …
            -/
            erw [pushforwardPullbackAdjunction_unit_pullback_map_germToPullbackStalk] } }
            /-
              🎉 no goals
            -/


@[reassoc (attr := simp)]
lemma germ_stalkPullbackInv (f : X ⟶ Y) (F : Y.Presheaf C) (x : X) (V : Opens X) (hV : x ∈ V) :
    ((pullback C f).obj F).germ _ x hV ≫ stalkPullbackInv C f F x =
    F.germToPullbackStalk _ f V x hV := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    x : ↑X
    V : TopologicalSpace.Opens ↑X
    hV : Membership.mem V x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pullback C f).obj  …
  -/
  apply colimit.ι_desc
  /-
    🎉 no goals
  -/


/-- The isomorphism `ℱ_{f(x)} ≅ (f⁻¹ℱ)ₓ`. -/
def stalkPullbackIso (f : X ⟶ Y) (F : Y.Presheaf C) (x : X) :
    F.stalk (f x) ≅ ((pullback C f).obj F).stalk x where
  hom := stalkPullbackHom _ _ _ _
  inv := stalkPullbackInv _ _ _ _
  hom_inv_id := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      x : ↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.stalkPullbackHom C f …
    -/
    ext U hU
    /-
      case ih
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      x : ↑X
      U : TopologicalSpace.Opens ↑Y
      hU : Membership.mem U (f x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.germ U (f x) hU) (CategoryTheory.C …
    -/
    dsimp
    rw [germ_stalkPullbackHom_assoc, germ_stalkPullbackInv, Category.comp_id,
      pushforwardPullbackAdjunction_unit_app_app_germToPullbackStalk]
  inv_hom_id := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      x : ↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.stalkPullbackInv C f …
    -/
    ext V hV
    /-
      case ih.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      x : ↑X
      V : TopologicalSpace.Opens ↑X
      hV : Membership.mem V x
      V✝ : TopologicalSpace.Opens ↑Y
      hV✝ : LE.le (Opposite.unop { unop := V }) ((TopologicalSpace.Opens.map f).obj  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforwardPullbac …
    -/
    dsimp
    /-
      case ih.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      x : ↑X
      V : TopologicalSpace.Opens ↑X
      hV : Membership.mem V x
      V✝ : TopologicalSpace.Opens ↑Y
      hV✝ : LE.le (Opposite.unop { unop := V }) ((TopologicalSpace.Opens.map f).obj  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforwardPullbac …
    -/
    rw [germ_stalkPullbackInv_assoc, Category.comp_id, germToPullbackStalk_stalkPullbackHom]
    /-
      🎉 no goals
    -/


/-- If `x` specializes to `y`, then there is a natural map `F.stalk y ⟶ F.stalk x`. -/
noncomputable def stalkSpecializes (F : X.Presheaf C) {x y : X} (h : x ⤳ y) :
    F.stalk y ⟶ F.stalk x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y Z : TopCat
    F : TopCat.Presheaf C X
    x y : ↑X
    h : Specializes x y
    ⊢ Quiver.Hom (F.stalk y) (F.stalk x)
  -/
  refine colimit.desc _ ⟨_, fun U => ?_, ?_⟩
  · exact
      colimit.ι ((OpenNhds.inclusion x).op ⋙ F)
        (op ⟨(unop U).1, (specializes_iff_forall_open.mp h _ (unop U).1.2 (unop U).2 : _)⟩)
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      ⊢ ∀ ⦃X_1 Y : Opposite (TopologicalSpace.OpenNhds y)⦄ (f : Quiver.Hom X_1 Y), E …
    -/
  · intro U V i
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      U V : Opposite (TopologicalSpace.OpenNhds y)
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.whiskeringLeft (Op …
    -/
    dsimp
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      U V : Opposite (TopologicalSpace.OpenNhds y)
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((TopologicalSpace.OpenNhds.in …
    -/
    rw [Category.comp_id]
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      U V : Opposite (TopologicalSpace.OpenNhds y)
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((TopologicalSpace.OpenNhds.in …
    -/
    let U' : OpenNhds x := ⟨_, (specializes_iff_forall_open.mp h _ (unop U).1.2 (unop U).2 : _)⟩
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      U V : Opposite (TopologicalSpace.OpenNhds y)
      i : Quiver.Hom U V
      U' : TopologicalSpace.OpenNhds x := { obj := (Opposite.unop U).obj, property : …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((TopologicalSpace.OpenNhds.in …
    -/
    let V' : OpenNhds x := ⟨_, (specializes_iff_forall_open.mp h _ (unop V).1.2 (unop V).2 : _)⟩
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X Y Z : TopCat
      F : TopCat.Presheaf C X
      x y : ↑X
      h : Specializes x y
      U V : Opposite (TopologicalSpace.OpenNhds y)
      i : Quiver.Hom U V
      U' : TopologicalSpace.OpenNhds x := { obj := (Opposite.unop U).obj, property : …
      V' : TopologicalSpace.OpenNhds x := { obj := (Opposite.unop V).obj, property : …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((TopologicalSpace.OpenNhds.in …
    -/
    exact colimit.w ((OpenNhds.inclusion x).op ⋙ F) (show V' ⟶ U' from i.unop).op
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp), elementwise nosimp]
theorem germ_stalkSpecializes (F : X.Presheaf C)
    {U : Opens X} {y : X} (hy : y ∈ U) {x : X} (h : x ⤳ y) :
    F.germ U y hy ≫ F.stalkSpecializes h = F.germ U x (h.mem_open U.isOpen hy) :=
  colimit.ι_desc _ _


@[deprecated (since := "2024-07-30")] alias germ_stalkSpecializes' := germ_stalkSpecializes


@[simp]
theorem stalkSpecializes_refl {C : Type*} [Category C] [Limits.HasColimits C] {X : TopCat}
    (F : X.Presheaf C) (x : X) : F.stalkSpecializes (specializes_refl x) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F : TopCat.Presheaf C X
    x : ↑X
    ⊢ Eq (F.stalkSpecializes ⋯) (CategoryTheory.CategoryStruct.id (F.stalk x))
  -/
  ext
  /-
    case ih
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F : TopCat.Presheaf C X
    x : ↑X
    U✝ : TopologicalSpace.Opens ↑X
    hxU✝ : Membership.mem U✝ x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.germ U✝ x hxU✝) (F.stalkSpecialize …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem stalkSpecializes_comp {C : Type*} [Category C] [Limits.HasColimits C] {X : TopCat}
    (F : X.Presheaf C) {x y z : X} (h : x ⤳ y) (h' : y ⤳ z) :
    F.stalkSpecializes h' ≫ F.stalkSpecializes h = F.stalkSpecializes (h.trans h') := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F : TopCat.Presheaf C X
    x y z : ↑X
    h : Specializes x y
    h' : Specializes y z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.stalkSpecializes h') (F.stalkSpeci …
  -/
  ext
  /-
    case ih
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F : TopCat.Presheaf C X
    x y z : ↑X
    h : Specializes x y
    h' : Specializes y z
    U✝ : TopologicalSpace.Opens ↑X
    hxU✝ : Membership.mem U✝ z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.germ U✝ z hxU✝) (CategoryTheory.Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem stalkSpecializes_stalkFunctor_map {F G : X.Presheaf C} (f : F ⟶ G) {x y : X} (h : x ⤳ y) :
    F.stalkSpecializes h ≫ (stalkFunctor C x).map f =
      (stalkFunctor C y).map f ≫ G.stalkSpecializes h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    x y : ↑X
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.stalkSpecializes h) ((TopCat.Presh …
  -/
  change (_ : colimit _ ⟶ _) = (_ : colimit _ ⟶ _)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    x y : ↑X
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.stalkSpecializes h) ((TopCat.Presh …
  -/
  ext; delta stalkFunctor; simpa [stalkSpecializes] using by rfl
                           /-
                             🎉 no goals
                           -/

-- See https://github.com/leanprover-community/batteries/issues/365 for the simpNF issue.

@[reassoc, elementwise, simp, nolint simpNF]
theorem stalkSpecializes_stalkPushforward (f : X ⟶ Y) (F : X.Presheaf C) {x y : X} (h : x ⤳ y) :
    (f _* F).stalkSpecializes (f.map_specializes h) ≫ F.stalkPushforward _ f x =
      F.stalkPushforward _ f y ≫ F.stalkSpecializes h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    x y : ↑X
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C f).o …
  -/
  change (_ : colimit _ ⟶ _) = (_ : colimit _ ⟶ _)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    x y : ↑X
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.pushforward C f).o …
  -/
  ext; delta stalkPushforward
  simp only [stalkSpecializes, colimit.ι_desc_assoc, colimit.ι_map_assoc, colimit.ι_pre,
    Category.assoc, colimit.pre_desc, colimit.ι_desc]
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimits C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C X
    x y : ↑X
    h : Specializes x y
    j✝ : Opposite (TopologicalSpace.OpenNhds (f y))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerRight (Catego …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The stalks are isomorphic on inseparable points -/
@[simps]
def stalkCongr {X : TopCat} {C : Type*} [Category C] [HasColimits C] (F : X.Presheaf C) {x y : X}
    (e : Inseparable x y) : F.stalk x ≅ F.stalk y :=
                                                        /-
                                                          C✝ : Type u
                                                          inst✝³ : CategoryTheory.Category.{v, u} C✝
                                                          inst✝² : CategoryTheory.Limits.HasColimits C✝
                                                          X✝ Y Z : TopCat
                                                          X : TopCat
                                                          C : Type u_1
                                                          inst✝¹ : CategoryTheory.Category.{?u.407612, u_1} C
                                                          inst✝ : CategoryTheory.Limits.HasColimits C
                                                          F : TopCat.Presheaf C X
                                                          x y : ↑X
                                                          e : Inseparable x y
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.stalkSpecializes ⋯) (F.stalkSpecia …
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  ⟨F.stalkSpecializes e.ge, F.stalkSpecializes e.le, by simp, by simp⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem germ_ext (F : X.Presheaf C) {U V : Opens X} {x : X} {hxU : x ∈ U} {hxV : x ∈ V}
    (W : Opens X) (hxW : x ∈ W) (iWU : W ⟶ U) (iWV : W ⟶ V) {sU : F.obj (op U)} {sV : F.obj (op V)}
    (ih : F.map iWU.op sU = F.map iWV.op sV) :
      F.germ _ x hxU sU = F.germ _ x hxV sV := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝ : CategoryTheory.ConcreteCategory C
    F : TopCat.Presheaf C X
    U V : TopologicalSpace.Opens ↑X
    x : ↑X
    hxU : Membership.mem U x
    hxV : Membership.mem V x
    W : TopologicalSpace.Opens ↑X
    hxW : Membership.mem W x
    iWU : Quiver.Hom W U
    iWV : Quiver.Hom W V
    sU : (CategoryTheory.forget C).obj (F.obj { unop := U })
    sV : (CategoryTheory.forget C).obj (F.obj { unop := V })
    ih : Eq ((F.map iWU.op) sU) ((F.map iWV.op) sV)
    ⊢ Eq ((F.germ U x hxU) sU) ((F.germ V x hxV) sV)
  -/
  rw [← F.germ_res iWU x hxW, ← F.germ_res iWV x hxW, comp_apply, comp_apply, ih]
  /-
    🎉 no goals
  -/


/--
For presheaves valued in a concrete category whose forgetful functor preserves filtered colimits,
every element of the stalk is the germ of a section.
-/
theorem germ_exist (F : X.Presheaf C) (x : X) (t : (stalk.{v, u} F x : Type v)) :
    ∃ (U : Opens X) (m : x ∈ U) (s : F.obj (op U)), F.germ _ x m s = t := by
  obtain ⟨U, s, e⟩ :=
    Types.jointly_surjective.{v, v} _ (isColimitOfPreserves (forget C) (colimit.isColimit _)) t
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    x : ↑X
    t : (CategoryTheory.forget C).obj (F.stalk x)
    U : Opposite (TopologicalSpace.OpenNhds x)
    s : ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhds x)) …
    e : Eq (((CategoryTheory.forget C).mapCocone (CategoryTheory.Limits.colimit.co …
    ⊢ Exists fun U => Exists fun m => Exists fun s => Eq ((F.germ U x m) s) t
  -/
  revert s e
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    x : ↑X
    t : (CategoryTheory.forget C).obj (F.stalk x)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ ∀ (s : ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhd …
  -/
  induction U with | h U => ?_
  /-
    case intro.intro.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    x : ↑X
    t : (CategoryTheory.forget C).obj (F.stalk x)
    U : TopologicalSpace.OpenNhds x
    ⊢ ∀ (s : ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhd …
  -/
  cases' U with V m
  /-
    case intro.intro.h.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    x : ↑X
    t : (CategoryTheory.forget C).obj (F.stalk x)
    V : TopologicalSpace.Opens ↑X
    m : Membership.mem V x
    ⊢ ∀ (s : ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhd …
  -/
  intro s e
  /-
    case intro.intro.h.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    x : ↑X
    t : (CategoryTheory.forget C).obj (F.stalk x)
    V : TopologicalSpace.Opens ↑X
    m : Membership.mem V x
    s : ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhds x)) …
    e : Eq (((CategoryTheory.forget C).mapCocone (CategoryTheory.Limits.colimit.co …
    ⊢ Exists fun U => Exists fun m => Exists fun s => Eq ((F.germ U x m) s) t
  -/
  exact ⟨V, m, s, e⟩
  /-
    🎉 no goals
  -/


theorem germ_eq (F : X.Presheaf C) {U V : Opens X} (x : X) (mU : x ∈ U) (mV : x ∈ V)
    (s : F.obj (op U)) (t : F.obj (op V)) (h : F.germ U x mU s = F.germ V x mV t) :
    ∃ (W : Opens X) (_m : x ∈ W) (iU : W ⟶ U) (iV : W ⟶ V), F.map iU.op s = F.map iV.op t := by
  obtain ⟨W, iU, iV, e⟩ :=
    (Types.FilteredColimit.isColimit_eq_iff.{v, v} _
          (isColimitOfPreserves _ (colimit.isColimit ((OpenNhds.inclusion x).op ⋙ F)))).mp h
  /-
    case intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F : TopCat.Presheaf C X
    U V : TopologicalSpace.Opens ↑X
    x : ↑X
    mU : Membership.mem U x
    mV : Membership.mem V x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U })
    t : (CategoryTheory.forget C).obj (F.obj { unop := V })
    h : Eq ((F.germ U x mU) s) ((F.germ V x mV) t)
    W : Opposite (TopologicalSpace.OpenNhds x)
    iU : Quiver.Hom { unop := { obj := U, property := mU } } W
    iV : Quiver.Hom { unop := { obj := V, property := mV } } W
    e : Eq ((((TopologicalSpace.OpenNhds.inclusion x).op.comp F).comp (CategoryThe …
    ⊢ Exists fun W => Exists fun _m => Exists fun iU => Exists fun iV => Eq ((F.ma …
  -/
  exact ⟨(unop W).1, (unop W).2, iU.unop, iV.unop, e⟩
  /-
    🎉 no goals
  -/


theorem stalkFunctor_map_injective_of_app_injective {F G : Presheaf C X} (f : F ⟶ G)
    (h : ∀ U : Opens X, Function.Injective (f.app (op U))) (x : X) :
    Function.Injective ((stalkFunctor C x).map f) := fun s t hst => by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    s t : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj F)
    hst : Eq (((TopCat.Presheaf.stalkFunctor C x).map f) s) (((TopCat.Presheaf.sta …
    ⊢ Eq s t
  -/
  rcases germ_exist F x s with ⟨U₁, hxU₁, s, rfl⟩
  /-
    case intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    t : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj F)
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    hst : Eq (((TopCat.Presheaf.stalkFunctor C x).map f) ((F.germ U₁ x hxU₁) s)) ( …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) t
  -/
  rcases germ_exist F x t with ⟨U₂, hxU₂, t, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq (((TopCat.Presheaf.stalkFunctor C x).map f) ((F.germ U₁ x hxU₁) s)) ( …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ U₂ x hxU₂) t)
  -/
  rw [stalkFunctor_map_germ_apply, stalkFunctor_map_germ_apply] at hst
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq ((G.germ U₁ x hxU₁) ((f.app { unop := U₁ }) s)) ((G.germ U₂ x hxU₂) ( …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ U₂ x hxU₂) t)
  -/
  obtain ⟨W, hxW, iWU₁, iWU₂, heq⟩ := G.germ_eq x hxU₁ hxU₂ _ _ hst
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq ((G.germ U₁ x hxU₁) ((f.app { unop := U₁ }) s)) ((G.germ U₂ x hxU₂) ( …
    W : TopologicalSpace.Opens ↑X
    hxW : Membership.mem W x
    iWU₁ : Quiver.Hom W U₁
    iWU₂ : Quiver.Hom W U₂
    heq : Eq ((G.map iWU₁.op) ((f.app { unop := U₁ }) s)) ((G.map iWU₂.op) ((f.app …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ U₂ x hxU₂) t)
  -/
  rw [← comp_apply, ← comp_apply, ← f.naturality, ← f.naturality, comp_apply, comp_apply] at heq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq ((G.germ U₁ x hxU₁) ((f.app { unop := U₁ }) s)) ((G.germ U₂ x hxU₂) ( …
    W : TopologicalSpace.Opens ↑X
    hxW : Membership.mem W x
    iWU₁ : Quiver.Hom W U₁
    iWU₂ : Quiver.Hom W U₂
    heq : Eq ((f.app { unop := W }) ((F.map iWU₁.op) s)) ((f.app { unop := W }) (( …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ U₂ x hxU₂) t)
  -/
  replace heq := h W heq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq ((G.germ U₁ x hxU₁) ((f.app { unop := U₁ }) s)) ((G.germ U₂ x hxU₂) ( …
    W : TopologicalSpace.Opens ↑X
    hxW : Membership.mem W x
    iWU₁ : Quiver.Hom W U₁
    iWU₂ : Quiver.Hom W U₂
    heq : Eq ((F.map iWU₁.op) s) ((F.map iWU₂.op) t)
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ U₂ x hxU₂) t)
  -/
  convert congr_arg (F.germ _ x hxW) heq using 1
  /-
    case h.e'_2.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    F G : TopCat.Presheaf C X
    f : Quiver.Hom F G
    h : ∀ (U : TopologicalSpace.Opens ↑X), Function.Injective ⇑(f.app { unop := U })
    x : ↑X
    U₁ : TopologicalSpace.Opens ↑X
    hxU₁ : Membership.mem U₁ x
    s : (CategoryTheory.forget C).obj (F.obj { unop := U₁ })
    U₂ : TopologicalSpace.Opens ↑X
    hxU₂ : Membership.mem U₂ x
    t : (CategoryTheory.forget C).obj (F.obj { unop := U₂ })
    hst : Eq ((G.germ U₁ x hxU₁) ((f.app { unop := U₁ }) s)) ((G.germ U₂ x hxU₂) ( …
    W : TopologicalSpace.Opens ↑X
    hxW : Membership.mem W x
    iWU₁ : Quiver.Hom W U₁
    iWU₂ : Quiver.Hom W U₂
    heq : Eq ((F.map iWU₁.op) s) ((F.map iWU₂.op) t)
    e_1✝ : Eq ((CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).o …
    ⊢ Eq ((F.germ U₁ x hxU₁) s) ((F.germ W x hxW) ((F.map iWU₁.op) s))
  -/
  exacts [(F.germ_res_apply iWU₁ x hxW s).symm, (F.germ_res_apply iWU₂ x hxW t).symm]
  /-
    🎉 no goals
  -/


/-- Let `F` be a sheaf valued in a concrete category, whose forgetful functor reflects isomorphisms,
preserves limits and filtered colimits. Then two sections who agree on every stalk must be equal.
-/
theorem section_ext (F : Sheaf C X) (U : Opens X) (s t : F.1.obj (op U))
    (h : ∀ (x : X) (hx : x ∈ U), F.presheaf.germ U x hx s = F.presheaf.germ U x hx t) : s = t := by
  -- We use `germ_eq` and the axiom of choice, to pick for every point `x` a neighbourhood
  -- `V x`, such that the restrictions of `s` and `t` to `V x` coincide.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F : TopCat.Sheaf C X
    U : TopologicalSpace.Opens ↑X
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
    h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
    ⊢ Eq s t
  -/
  choose V m i₁ i₂ heq using fun x : U => F.presheaf.germ_eq x.1 x.2 x.2 s t (h x.1 x.2)
  -- Since `F` is a sheaf, we can prove the equality locally, if we can show that these
  -- neighborhoods form a cover of `U`.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F : TopCat.Sheaf C X
    U : TopologicalSpace.Opens ↑X
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
    h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
    ⊢ Eq s t
  -/
  apply F.eq_of_locally_eq' V U i₁
    /-
      case hcover
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
      inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
      F : TopCat.Sheaf C X
      U : TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
      h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
      V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
      m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
      i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
      heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
      ⊢ LE.le U (iSup V)
    -/
  · intro x hxU
    /-
      case hcover
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
      inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
      F : TopCat.Sheaf C X
      U : TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
      h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
      V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
      m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
      i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
      heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
      x : ↑X
      hxU : Membership.mem (↑U) x
      ⊢ Membership.mem (↑(iSup V)) x
    -/
    simp only [Opens.coe_iSup, Set.mem_iUnion, SetLike.mem_coe]
    /-
      case hcover
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
      inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
      F : TopCat.Sheaf C X
      U : TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
      h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
      V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
      m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
      i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
      heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
      x : ↑X
      hxU : Membership.mem (↑U) x
      ⊢ Exists fun i => Membership.mem (V i) x
    -/
    exact ⟨⟨x, hxU⟩, m ⟨x, hxU⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
      inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
      F : TopCat.Sheaf C X
      U : TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
      h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
      V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
      m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
      i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
      heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
      ⊢ ∀ (i : Subtype fun x => Membership.mem U x), Eq ((F.val.map (i₁ i).op) s) (( …
    -/
  · intro x
    /-
      case h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.Limits.HasColimits C
      X : TopCat
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
      inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
      F : TopCat.Sheaf C X
      U : TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
      h : ∀ (x : ↑X) (hx : Membership.mem U x), Eq ((F.presheaf.germ U x hx) s) ((F. …
      V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑X
      m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
      i₁ i₂ : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
      heq : ∀ (x : Subtype fun x => Membership.mem U x), Eq ((F.presheaf.map (i₁ x). …
      x : Subtype fun x => Membership.mem U x
      ⊢ Eq ((F.val.map (i₁ x).op) s) ((F.val.map (i₁ x).op) t)
    -/
    rw [heq, Subsingleton.elim (i₁ x) (i₂ x)]
    /-
      🎉 no goals
    -/

/-
Note that the analogous statement for surjectivity is false: Surjectivity on stalks does not
imply surjectivity of the components of a sheaf morphism. However it does imply that the morphism
is an epi, but this fact is not yet formalized.
-/

theorem app_injective_of_stalkFunctor_map_injective {F : Sheaf C X} {G : Presheaf C X} (f : F.1 ⟶ G)
    (U : Opens X) (h : ∀ x ∈ U, Function.Injective ((stalkFunctor C x).map f)) :
    Function.Injective (f.app (op U)) := fun s t hst =>
  section_ext F _ _ _ fun x hx =>
                 /-
                   C : Type u
                   inst✝⁶ : CategoryTheory.Category.{v, u} C
                   inst✝⁵ : CategoryTheory.Limits.HasColimits C
                   X : TopCat
                   inst✝⁴ : CategoryTheory.ConcreteCategory C
                   inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
                   inst✝² : CategoryTheory.Limits.HasLimits C
                   inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
                   inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
                   F : TopCat.Sheaf C X
                   G : TopCat.Presheaf C X
                   f : Quiver.Hom F.val G
                   U : TopologicalSpace.Opens ↑X
                   h : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf.sta …
                   s t : (CategoryTheory.forget C).obj (F.val.obj { unop := U })
                   hst : Eq ((f.app { unop := U }) s) ((f.app { unop := U }) t)
                   x : ↑X
                   hx : Membership.mem U x
                   ⊢ Eq (((TopCat.Presheaf.stalkFunctor C x).map f) ((F.presheaf.germ U x hx) s)) …
                 -/
    h x hx <| by rw [stalkFunctor_map_germ_apply, stalkFunctor_map_germ_apply, hst]
                 /-
                   🎉 no goals
                 -/


theorem app_injective_iff_stalkFunctor_map_injective {F : Sheaf C X} {G : Presheaf C X}
    (f : F.1 ⟶ G) :
    (∀ x : X, Function.Injective ((stalkFunctor C x).map f)) ↔
      ∀ U : Opens X, Function.Injective (f.app (op U)) :=
  ⟨fun h U => app_injective_of_stalkFunctor_map_injective f U fun x _ => h x,
    stalkFunctor_map_injective_of_app_injective f⟩


instance stalkFunctor_preserves_mono (x : X) :
    Functor.PreservesMonomorphisms (Sheaf.forget C X ⋙ stalkFunctor C x) :=
  ⟨@fun _𝓐 _𝓑 f _ =>
    ConcreteCategory.mono_of_injective _ <|
      (app_injective_iff_stalkFunctor_map_injective f.1).mpr
        (fun c =>
          (ConcreteCategory.mono_iff_injective_of_preservesPullback (f.1.app (op c))).mp
            ((NatTrans.mono_iff_mono_app f.1).mp
                (CategoryTheory.presheaf_mono_of_mono ..) <|
              op c))
        x⟩


theorem stalk_mono_of_mono {F G : Sheaf C X} (f : F ⟶ G) [Mono f] :
    ∀ x, Mono <| (stalkFunctor C x).map f.1 :=
  fun x => Functor.map_mono (Sheaf.forget.{v} C X ⋙ stalkFunctor C x) f


theorem mono_of_stalk_mono {F G : Sheaf C X} (f : F ⟶ G) [∀ x, Mono <| (stalkFunctor C x).map f.1] :
    Mono f :=
  (Sheaf.Hom.mono_iff_presheaf_mono _ _ _).mpr <|
    (NatTrans.mono_iff_mono_app _).mpr fun U =>
      (ConcreteCategory.mono_iff_injective_of_preservesPullback _).mpr <|
        app_injective_of_stalkFunctor_map_injective f.1 U.unop fun _x _hx =>
          (ConcreteCategory.mono_iff_injective_of_preservesPullback _).mp <| inferInstance


theorem mono_iff_stalk_mono {F G : Sheaf C X} (f : F ⟶ G) :
    Mono f ↔ ∀ x, Mono ((stalkFunctor C x).map f.1) :=
  ⟨fun _ => stalk_mono_of_mono _, fun _ => mono_of_stalk_mono _⟩


/-- For surjectivity, we are given an arbitrary section `t` and need to find a preimage for it.
We claim that it suffices to find preimages *locally*. That is, for each `x : U` we construct
a neighborhood `V ≤ U` and a section `s : F.obj (op V))` such that `f.app (op V) s` and `t`
agree on `V`. -/
theorem app_surjective_of_injective_of_locally_surjective {F G : Sheaf C X} (f : F ⟶ G)
    (U : Opens X) (hinj : ∀ x ∈ U, Function.Injective ((stalkFunctor C x).map f.1))
    (hsurj : ∀ (t x) (_ : x ∈ U), ∃ (V : Opens X) (_ : x ∈ V) (iVU : V ⟶ U) (s : F.1.obj (op V)),
          f.1.app (op V) s = G.1.map iVU.op t) :
    Function.Surjective (f.1.app (op U)) := by
  conv at hsurj =>
    enter [t]
    rw [Subtype.forall' (p := (· ∈ U))]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    ⊢ Function.Surjective ⇑(f.val.app { unop := U })
  -/
  intro t
  -- We use the axiom of choice to pick around each point `x` an open neighborhood `V` and a
  -- preimage under `f` on `V`.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    ⊢ Exists fun a => Eq ((f.val.app { unop := U }) a) t
  -/
  choose V mV iVU sf heq using hsurj t
  -- These neighborhoods clearly cover all of `U`.
  have V_cover : U ≤ iSup V := by
    intro x hxU
    simp only [Opens.coe_iSup, Set.mem_iUnion, SetLike.mem_coe]
    exact ⟨⟨x, hxU⟩, mV ⟨x, hxU⟩⟩
  suffices IsCompatible F.val V sf by
    -- Since `F` is a sheaf, we can glue all the local preimages together to get a global preimage.
    obtain ⟨s, s_spec, -⟩ := F.existsUnique_gluing' V U iVU V_cover sf this
    · use s
      apply G.eq_of_locally_eq' V U iVU V_cover
      intro x
      rw [← comp_apply, ← f.1.naturality, comp_apply, s_spec, heq]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    ⊢ TopCat.Presheaf.IsCompatible F.val V sf
  -/
  intro x y
  -- What's left to show here is that the sections `sf` are compatible, i.e. they agree on
  -- the intersections `V x ⊓ V y`. We prove this by showing that all germs are equal.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    ⊢ Eq ((F.val.map ((V x).infLELeft (V y)).op) (sf x)) ((F.val.map ((V x).infLER …
  -/
  apply section_ext
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    ⊢ ∀ (x_1 : ↑X) (hx : Membership.mem (Min.min (V x) (V y)) x_1), Eq ((F.preshea …
  -/
  intro z hz
  -- Here, we need to use injectivity of the stalk maps.
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    z : ↑X
    hz : Membership.mem (Min.min (V x) (V y)) z
    ⊢ Eq ((F.presheaf.germ (Min.min (V x) (V y)) z hz) ((F.val.map ((V x).infLELef …
  -/
  apply hinj z ((iVU x).le ((inf_le_left : V x ⊓ V y ≤ V x) hz))
  /-
    case h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    z : ↑X
    hz : Membership.mem (Min.min (V x) (V y)) z
    ⊢ Eq (((TopCat.Presheaf.stalkFunctor C z).map f.val) ((F.presheaf.germ (Min.mi …
  -/
  dsimp only
  /-
    case h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    z : ↑X
    hz : Membership.mem (Min.min (V x) (V y)) z
    ⊢ Eq (((TopCat.Presheaf.stalkFunctor C z).map f.val) ((F.presheaf.germ (Min.mi …
  -/
  rw [stalkFunctor_map_germ_apply, stalkFunctor_map_germ_apply]
  /-
    case h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    z : ↑X
    hz : Membership.mem (Min.min (V x) (V y)) z
    ⊢ Eq ((TopCat.Presheaf.germ G.val (Min.min (V x) (V y)) z hz) ((f.val.app { un …
  -/
  simp_rw [← comp_apply, f.1.naturality, comp_apply, heq, ← comp_apply, ← G.1.map_comp]
  /-
    case h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    hinj : ∀ (x : ↑X), Membership.mem U x → Function.Injective ⇑((TopCat.Presheaf. …
    hsurj : ∀ (t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })) (x : S …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    V : (Subtype fun a => Membership.mem U a) → TopologicalSpace.Opens ↑X
    mV : ∀ (x : Subtype fun a => Membership.mem U a), Membership.mem (V x) ↑x
    iVU : (x : Subtype fun a => Membership.mem U a) → Quiver.Hom (V x) U
    sf : (x : Subtype fun a => Membership.mem U a) → (CategoryTheory.forget C).obj …
    heq : ∀ (x : Subtype fun a => Membership.mem U a), Eq ((f.val.app { unop := V  …
    V_cover : LE.le U (iSup V)
    x y : Subtype fun a => Membership.mem U a
    z : ↑X
    hz : Membership.mem (Min.min (V x) (V y)) z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (G.val.map (CategoryTheory.CategoryS …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem app_surjective_of_stalkFunctor_map_bijective {F G : Sheaf C X} (f : F ⟶ G) (U : Opens X)
    (h : ∀ x ∈ U, Function.Bijective ((stalkFunctor C x).map f.1)) :
    Function.Surjective (f.1.app (op U)) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    ⊢ Function.Surjective ⇑(f.val.app { unop := U })
  -/
  refine app_surjective_of_injective_of_locally_surjective f U (And.left <| h · ·) fun t x hx => ?_
  -- Now we need to prove our initial claim: That we can find preimages of `t` locally.
  -- Since `f` is surjective on stalks, we can find a preimage `s₀` of the germ of `t` at `x`
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  obtain ⟨s₀, hs₀⟩ := (h x hx).2 (G.presheaf.germ U x hx t)
  -- ... and this preimage must come from some section `s₁` defined on some open neighborhood `V₁`
  /-
    case intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    s₀ : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj F.v …
    hs₀ : Eq (((TopCat.Presheaf.stalkFunctor C x).map f.val) s₀) ((G.presheaf.germ …
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  obtain ⟨V₁, hxV₁, s₁, hs₁⟩ := F.presheaf.germ_exist x s₀
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    s₀ : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj F.v …
    hs₀ : Eq (((TopCat.Presheaf.stalkFunctor C x).map f.val) s₀) ((G.presheaf.germ …
    V₁ : TopologicalSpace.Opens ↑X
    hxV₁ : Membership.mem V₁ x
    s₁ : (CategoryTheory.forget C).obj (F.presheaf.obj { unop := V₁ })
    hs₁ : Eq ((F.presheaf.germ V₁ x hxV₁) s₁) s₀
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  subst hs₁; rename' hs₀ => hs₁
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    V₁ : TopologicalSpace.Opens ↑X
    hxV₁ : Membership.mem V₁ x
    s₁ : (CategoryTheory.forget C).obj (F.presheaf.obj { unop := V₁ })
    hs₁ : Eq (((TopCat.Presheaf.stalkFunctor C x).map f.val) ((F.presheaf.germ V₁  …
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  rw [stalkFunctor_map_germ_apply V₁ x hxV₁ f.1 s₁] at hs₁
  -- Now, the germ of `f.app (op V₁) s₁` equals the germ of `t`, hence they must coincide on
  -- some open neighborhood `V₂`.
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    V₁ : TopologicalSpace.Opens ↑X
    hxV₁ : Membership.mem V₁ x
    s₁ : (CategoryTheory.forget C).obj (F.presheaf.obj { unop := V₁ })
    hs₁ : Eq ((TopCat.Presheaf.germ G.val V₁ x hxV₁) ((f.val.app { unop := V₁ }) s …
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  obtain ⟨V₂, hxV₂, iV₂V₁, iV₂U, heq⟩ := G.presheaf.germ_eq x hxV₁ hx _ _ hs₁
  -- The restriction of `s₁` to that neighborhood is our desired local preimage.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    V₁ : TopologicalSpace.Opens ↑X
    hxV₁ : Membership.mem V₁ x
    s₁ : (CategoryTheory.forget C).obj (F.presheaf.obj { unop := V₁ })
    hs₁ : Eq ((TopCat.Presheaf.germ G.val V₁ x hxV₁) ((f.val.app { unop := V₁ }) s …
    V₂ : TopologicalSpace.Opens ↑X
    hxV₂ : Membership.mem V₂ x
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂U : Quiver.Hom V₂ U
    heq : Eq ((G.presheaf.map iV₂V₁.op) ((f.val.app { unop := V₁ }) s₁)) ((G.presh …
    ⊢ Exists fun V => Exists fun x => Exists fun iVU => Exists fun s => Eq ((f.val …
  -/
  use V₂, hxV₂, iV₂U, F.1.map iV₂V₁.op s₁
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    h : ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.sta …
    t : (CategoryTheory.forget C).obj (G.val.obj { unop := U })
    x : ↑X
    hx : Membership.mem U x
    V₁ : TopologicalSpace.Opens ↑X
    hxV₁ : Membership.mem V₁ x
    s₁ : (CategoryTheory.forget C).obj (F.presheaf.obj { unop := V₁ })
    hs₁ : Eq ((TopCat.Presheaf.germ G.val V₁ x hxV₁) ((f.val.app { unop := V₁ }) s …
    V₂ : TopologicalSpace.Opens ↑X
    hxV₂ : Membership.mem V₂ x
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂U : Quiver.Hom V₂ U
    heq : Eq ((G.presheaf.map iV₂V₁.op) ((f.val.app { unop := V₁ }) s₁)) ((G.presh …
    ⊢ Eq ((f.val.app { unop := V₂ }) ((F.val.map iV₂V₁.op) s₁)) ((G.val.map iV₂U.o …
  -/
  rw [← comp_apply, f.1.naturality, comp_apply, heq]
  /-
    🎉 no goals
  -/


theorem app_bijective_of_stalkFunctor_map_bijective {F G : Sheaf C X} (f : F ⟶ G) (U : Opens X)
    (h : ∀ x ∈ U, Function.Bijective ((stalkFunctor C x).map f.1)) :
    Function.Bijective (f.1.app (op U)) :=
  ⟨app_injective_of_stalkFunctor_map_injective f.1 U fun x hx => (h x hx).1,
    app_surjective_of_stalkFunctor_map_bijective f U h⟩


theorem app_isIso_of_stalkFunctor_map_iso {F G : Sheaf C X} (f : F ⟶ G) (U : Opens X)
    [∀ x : U, IsIso ((stalkFunctor C x.val).map f.1)] : IsIso (f.1.app (op U)) := by
  -- Since the forgetful functor of `C` reflects isomorphisms, it suffices to see that the
  -- underlying map between types is an isomorphism, i.e. bijective.
  suffices IsIso ((forget C).map (f.1.app (op U))) by
    exact isIso_of_reflects_iso (f.1.app (op U)) (forget C)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    inst✝ : ∀ (x : Subtype fun x => Membership.mem U x), CategoryTheory.IsIso ((To …
    ⊢ CategoryTheory.IsIso ((CategoryTheory.forget C).map (f.val.app { unop := U }))
  -/
  rw [isIso_iff_bijective]
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    inst✝ : ∀ (x : Subtype fun x => Membership.mem U x), CategoryTheory.IsIso ((To …
    ⊢ Function.Bijective ((CategoryTheory.forget C).map (f.val.app { unop := U }))
  -/
  apply app_bijective_of_stalkFunctor_map_bijective
  /-
    case h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    inst✝ : ∀ (x : Subtype fun x => Membership.mem U x), CategoryTheory.IsIso ((To …
    ⊢ ∀ (x : ↑X), Membership.mem U x → Function.Bijective ⇑((TopCat.Presheaf.stalk …
  -/
  intro x hx
  /-
    case h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    inst✝ : ∀ (x : Subtype fun x => Membership.mem U x), CategoryTheory.IsIso ((To …
    x : ↑X
    hx : Membership.mem U x
    ⊢ Function.Bijective ⇑((TopCat.Presheaf.stalkFunctor C x).map f.val)
  -/
  apply (isIso_iff_bijective _).mp
  /-
    case h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    U : TopologicalSpace.Opens ↑X
    inst✝ : ∀ (x : Subtype fun x => Membership.mem U x), CategoryTheory.IsIso ((To …
    x : ↑X
    hx : Membership.mem U x
    ⊢ CategoryTheory.IsIso ⇑((TopCat.Presheaf.stalkFunctor C x).map f.val)
  -/
  exact Functor.map_isIso (forget C) ((stalkFunctor C (⟨x, hx⟩ : U).1).map f.1)
  /-
    🎉 no goals
  -/

-- Making this an instance would cause a loop in typeclass resolution with `Functor.map_isIso`

/-- Let `F` and `G` be sheaves valued in a concrete category, whose forgetful functor reflects
isomorphisms, preserves limits and filtered colimits. Then if the stalk maps of a morphism
`f : F ⟶ G` are all isomorphisms, `f` must be an isomorphism.
-/
theorem isIso_of_stalkFunctor_map_iso {F G : Sheaf C X} (f : F ⟶ G)
    [∀ x : X, IsIso ((stalkFunctor C x).map f.1)] : IsIso f := by
  -- Since the inclusion functor from sheaves to presheaves is fully faithful, it suffices to
  -- show that `f`, as a morphism between _presheaves_, is an isomorphism.
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    inst✝ : ∀ (x : ↑X), CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C x).m …
    ⊢ CategoryTheory.IsIso f
  -/
  suffices IsIso ((Sheaf.forget C X).map f) by exact isIso_of_fully_faithful (Sheaf.forget C X) f
  -- We show that all components of `f` are isomorphisms.
  suffices ∀ U : (Opens X)ᵒᵖ, IsIso (f.1.app U) by
    exact @NatIso.isIso_of_isIso_app _ _ _ _ F.1 G.1 f.1 this
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    inst✝ : ∀ (x : ↑X), CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C x).m …
    ⊢ ∀ (U : Opposite (TopologicalSpace.Opens ↑X)), CategoryTheory.IsIso (f.val.ap …
  -/
  intro U; induction U
  /-
    case h
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Limits.HasColimits C
    X : TopCat
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝³ : CategoryTheory.Limits.HasLimits C
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
    inst✝¹ : (CategoryTheory.forget C).ReflectsIsomorphisms
    F G : TopCat.Sheaf C X
    f : Quiver.Hom F G
    inst✝ : ∀ (x : ↑X), CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C x).m …
    X✝ : TopologicalSpace.Opens ↑X
    ⊢ CategoryTheory.IsIso (f.val.app { unop := X✝ })
  -/
  apply app_isIso_of_stalkFunctor_map_iso
  /-
    🎉 no goals
  -/


/-- Let `F` and `G` be sheaves valued in a concrete category, whose forgetful functor reflects
isomorphisms, preserves limits and filtered colimits. Then a morphism `f : F ⟶ G` is an
isomorphism if and only if all of its stalk maps are isomorphisms.
-/
theorem isIso_iff_stalkFunctor_map_iso {F G : Sheaf C X} (f : F ⟶ G) :
    IsIso f ↔ ∀ x : X, IsIso ((stalkFunctor C x).map f.1) :=
  ⟨fun _ x =>
    @Functor.map_isIso _ _ _ _ _ _ (stalkFunctor C x) f.1 ((Sheaf.forget C X).map_isIso f),
   fun _ => isIso_of_stalkFunctor_map_iso f⟩


