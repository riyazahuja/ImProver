/--
A continuous map from a profinite set to a finite set factors through one of the components of
the profinite set when written as a cofiltered limit of finite sets.
-/
lemma exists_hom (hc : IsLimit c) {X : FintypeCat} (f : c.pt ⟶ toProfinite.obj X) :
    ∃ (i : I) (g : F.obj i ⟶ X), f = c.π.app i ≫ toProfinite.map g := by
  /-
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    ⊢ Exists fun i => Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp (c. …
  -/
  let _ : TopologicalSpace X := ⊥
  /-
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    x✝ : TopologicalSpace ↑X := Bot.bot
    ⊢ Exists fun i => Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp (c. …
  -/
  have : DiscreteTopology (toProfinite.obj X) := ⟨rfl⟩
  let f' : LocallyConstant c.pt (toProfinite.obj X) :=
    ⟨f, (IsLocallyConstant.iff_continuous _).mpr f.continuous⟩
  /-
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    x✝ : TopologicalSpace ↑X := Bot.bot
    this : DiscreteTopology ↑(FintypeCat.toProfinite.obj X).toTop
    f' : LocallyConstant ↑c.pt.toTop ↑(FintypeCat.toProfinite.obj X).toTop := { to …
    ⊢ Exists fun i => Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp (c. …
  -/
  obtain ⟨i, g, h⟩ := exists_locallyConstant.{_, u} c hc f'
  /-
    case intro.intro
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    x✝ : TopologicalSpace ↑X := Bot.bot
    this : DiscreteTopology ↑(FintypeCat.toProfinite.obj X).toTop
    f' : LocallyConstant ↑c.pt.toTop ↑(FintypeCat.toProfinite.obj X).toTop := { to …
    i : I
    g : LocallyConstant ↑((F.comp FintypeCat.toProfinite).obj i).toTop ↑(FintypeCa …
    h : Eq f' (LocallyConstant.comap (c.π.app i) g)
    ⊢ Exists fun i => Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp (c. …
  -/
  refine ⟨i, (g : _ → _), ?_⟩
  /-
    case intro.intro
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    x✝ : TopologicalSpace ↑X := Bot.bot
    this : DiscreteTopology ↑(FintypeCat.toProfinite.obj X).toTop
    f' : LocallyConstant ↑c.pt.toTop ↑(FintypeCat.toProfinite.obj X).toTop := { to …
    i : I
    g : LocallyConstant ↑((F.comp FintypeCat.toProfinite).obj i).toTop ↑(FintypeCa …
    h : Eq f' (LocallyConstant.comap (c.π.app i) g)
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfinite …
  -/
  ext x
  /-
    case intro.intro.w
    I : Type u
    inst✝¹ : CategoryTheory.SmallCategory I
    inst✝ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    X : FintypeCat
    f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
    x✝ : TopologicalSpace ↑X := Bot.bot
    this : DiscreteTopology ↑(FintypeCat.toProfinite.obj X).toTop
    f' : LocallyConstant ↑c.pt.toTop ↑(FintypeCat.toProfinite.obj X).toTop := { to …
    i : I
    g : LocallyConstant ↑((F.comp FintypeCat.toProfinite).obj i).toTop ↑(FintypeCa …
    h : Eq f' (LocallyConstant.comap (c.π.app i) g)
    x : (CategoryTheory.forget Profinite).obj c.pt
    ⊢ Eq (f x) ((CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProf …
  -/
  exact LocallyConstant.congr_fun h x
  /-
    🎉 no goals
  -/


/--
Given a cone in `Profinite`, consisting of finite sets and indexed by a cofiltered category,
we obtain a functor from the indexing category to `StructuredArrow c.pt toProfinite`.
-/
@[simps]
def functor : I ⥤ StructuredArrow c.pt toProfinite where
  obj i := StructuredArrow.mk (c.π.app i)
  map f := StructuredArrow.homMk (F.map f) (c.w f)

-- We check that the original diagram factors through `Profinite.Extend.functor`.

/--
Given a cone in `Profinite`, consisting of finite sets and indexed by a cofiltered category,
we obtain a functor from the opposite of the indexing category to
`CostructuredArrow toProfinite.op ⟨c.pt⟩`.
-/
@[simps! obj map]
def functorOp : Iᵒᵖ ⥤ CostructuredArrow toProfinite.op ⟨c.pt⟩ :=
  (functor c).op ⋙ StructuredArrow.toCostructuredArrow _ _

-- We check that the opposite of the original diagram factors through `Profinite.Extend.functorOp`.

/--
If the projection maps in the cone are epimorphic and the cone is limiting, then
`Profinite.Extend.functor` is initial.

TODO: investigate how to weaken the assumption `∀ i, Epi (c.π.app i)` to
`∀ i, ∃ j (_ : j ⟶ i), Epi (c.π.app j)`.
-/
lemma functor_initial (hc : IsLimit c) [∀ i, Epi (c.π.app i)] : Initial (functor c) := by
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    ⊢ (Profinite.Extend.functor c).Initial
  -/
  let e : I ≌ ULiftHom.{w} (ULift.{w} I) := ULiftHomULiftCategory.equiv _
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
    ⊢ (Profinite.Extend.functor c).Initial
  -/
  suffices (e.inverse ⋙ functor c).Initial from initial_of_equivalence_comp e.inverse (functor c)
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
    ⊢ (e.inverse.comp (Profinite.Extend.functor c)).Initial
  -/
  rw [initial_iff_of_isCofiltered (F := e.inverse ⋙ functor c)]
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
    ⊢ And (∀ (d : CategoryTheory.StructuredArrow c.pt FintypeCat.toProfinite), Exi …
  -/
  constructor
    /-
      case left
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      ⊢ ∀ (d : CategoryTheory.StructuredArrow c.pt FintypeCat.toProfinite), Exists f …
    -/
  · intro ⟨_, X, (f : c.pt ⟶ _)⟩
    /-
      case left
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝ : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      ⊢ Exists fun c_1 => Nonempty (Quiver.Hom ((e.inverse.comp (Profinite.Extend.fu …
    -/
    obtain ⟨i, g, h⟩ := exists_hom c hc f
    /-
      case left.intro.intro
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝ : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      i : I
      g : Quiver.Hom (F.obj i) X
      h : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfini …
      ⊢ Exists fun c_1 => Nonempty (Quiver.Hom ((e.inverse.comp (Profinite.Extend.fu …
    -/
    refine ⟨⟨i⟩, ⟨StructuredArrow.homMk g h.symm⟩⟩
    /-
      🎉 no goals
    -/
  · intro ⟨_, X, (f : c.pt ⟶ _)⟩ ⟨i⟩ ⟨_, (s : F.obj i ⟶ X), (w : f = c.π.app i ≫ _)⟩
      ⟨_, (s' : F.obj i ⟶ X), (w' : f = c.π.app i ≫ _)⟩
    simp only [functor_obj, functor_map, StructuredArrow.hom_eq_iff, StructuredArrow.mk_right,
      StructuredArrow.comp_right, StructuredArrow.homMk_right]
    /-
      case right
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝² : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      i : I
      left✝¹ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down  …
      s : Quiver.Hom (F.obj i) X
      w : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfini …
      left✝ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down : …
      s' : Quiver.Hom (F.obj i) X
      w' : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfin …
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp ((e. …
    -/
    refine ⟨⟨i⟩, 𝟙 _, ?_⟩
    /-
      case right
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝² : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      i : I
      left✝¹ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down  …
      s : Quiver.Hom (F.obj i) X
      w : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfini …
      left✝ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down : …
      s' : Quiver.Hom (F.obj i) X
      w' : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfin …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.inverse.comp (Profinite.Extend.fu …
    -/
    simp only [CategoryTheory.Functor.map_id, Category.id_comp]
    /-
      case right
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝² : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      i : I
      left✝¹ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down  …
      s : Quiver.Hom (F.obj i) X
      w : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfini …
      left✝ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down : …
      s' : Quiver.Hom (F.obj i) X
      w' : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfin …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((e …
    -/
    rw [w] at w'
    /-
      case right
      I : Type u
      inst✝² : CategoryTheory.SmallCategory I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      e : CategoryTheory.Equivalence I (CategoryTheory.ULiftHom (ULift.{w, u} I)) := …
      left✝² : CategoryTheory.Discrete PUnit.{1}
      X : FintypeCat
      f : Quiver.Hom c.pt (FintypeCat.toProfinite.obj X)
      i : I
      left✝¹ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down  …
      s : Quiver.Hom (F.obj i) X
      w : Eq f (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfini …
      left✝ : Quiver.Hom ((e.inverse.comp (Profinite.Extend.functor c)).obj { down : …
      s' : Quiver.Hom (F.obj i) X
      w' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app i) (FintypeCat.toProfinit …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((e …
    -/
    exact toProfinite.map_injective <| Epi.left_cancellation _ _ w'
    /-
      🎉 no goals
    -/


/--
If the projection maps in the cone are epimorphic and the cone is limiting, then
`Profinite.Extend.functorOp` is final.
-/
lemma functorOp_final (hc : IsLimit c) [∀ i, Epi (c.π.app i)] : Final (functorOp c) := by
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    ⊢ (Profinite.Extend.functorOp c).Final
  -/
  have := functor_initial c hc
  have : ((StructuredArrow.toCostructuredArrow toProfinite c.pt)).IsEquivalence  :=
    (inferInstance : (structuredArrowOpEquivalence _ _).functor.IsEquivalence )
  /-
    I : Type u
    inst✝² : CategoryTheory.SmallCategory I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    this✝ : (Profinite.Extend.functor c).Initial
    this : (CategoryTheory.StructuredArrow.toCostructuredArrow FintypeCat.toProfin …
    ⊢ (Profinite.Extend.functorOp c).Final
  -/
  exact Functor.final_comp (functor c).op _
  /-
    🎉 no goals
  -/


/--
Given a functor `G` from `Profinite` and `S : Profinite`, we obtain a cone on
`(StructuredArrow.proj S toProfinite ⋙ toProfinite ⋙ G)` with cone point `G.obj S`.

Whiskering this cone with `Profinite.Extend.functor c` gives `G.mapCone c` as we check in the
example below.
-/
@[simps]
def cone (S : Profinite) :
    Cone (StructuredArrow.proj S toProfinite ⋙ toProfinite ⋙ G) where
  pt := G.obj S
  π := {
    app := fun i ↦ G.map i.hom
                                  /-
                                    I : Type u
                                    inst✝² : CategoryTheory.SmallCategory I
                                    inst✝¹ : CategoryTheory.IsCofiltered I
                                    F : CategoryTheory.Functor I FintypeCat
                                    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
                                    C : Type u_1
                                    inst✝ : CategoryTheory.Category.{?u.30861, u_1} C
                                    G : CategoryTheory.Functor Profinite C
                                    S : Profinite
                                    x✝¹ x✝ : CategoryTheory.StructuredArrow S FintypeCat.toProfinite
                                    f : Quiver.Hom x✝¹ x✝
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
                                  -/
    naturality := fun _ _ f ↦ (by simp [← map_comp]) }
                                  /-
                                    🎉 no goals
                                  -/


/--
If `c` and `G.mapCone c` are limit cones and the projection maps in `c` are epimorphic,
then `cone G c.pt` is a limit cone.
-/
noncomputable
def isLimitCone (hc : IsLimit c) [∀ i, Epi (c.π.app i)] (hc' : IsLimit <| G.mapCone c) :
    IsLimit (cone G c.pt) := (functor_initial c hc).isLimitWhiskerEquiv _ _ hc'


/--
Given a functor `G` from `Profiniteᵒᵖ` and `S : Profinite`, we obtain a cocone on
`(CostructuredArrow.proj toProfinite.op ⟨S⟩ ⋙ toProfinite.op ⋙ G)` with cocone point `G.obj ⟨S⟩`.

Whiskering this cocone with `Profinite.Extend.functorOp c` gives `G.mapCocone c.op` as we check in
the example below.
-/
@[simps]
def cocone (S : Profinite) :
    Cocone (CostructuredArrow.proj toProfinite.op ⟨S⟩ ⋙ toProfinite.op ⋙ G) where
  pt := G.obj ⟨S⟩
  ι := {
    app := fun i ↦ G.map i.hom
    naturality := fun _ _ f ↦ (by
      /-
        I : Type u
        inst✝² : CategoryTheory.SmallCategory I
        inst✝¹ : CategoryTheory.IsCofiltered I
        F : CategoryTheory.Functor I FintypeCat
        c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.40805, u_1} C
        G : CategoryTheory.Functor (Opposite Profinite) C
        S : Profinite
        x✝¹ x✝ : CategoryTheory.CostructuredArrow FintypeCat.toProfinite.op { unop :=  …
        f : Quiver.Hom x✝¹ x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
      -/
      have := f.w
      simp only [op_obj, const_obj_obj, op_map, CostructuredArrow.right_eq_id, const_obj_map,
        Category.comp_id] at this
      /-
        I : Type u
        inst✝² : CategoryTheory.SmallCategory I
        inst✝¹ : CategoryTheory.IsCofiltered I
        F : CategoryTheory.Functor I FintypeCat
        c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.40805, u_1} C
        G : CategoryTheory.Functor (Opposite Profinite) C
        S : Profinite
        x✝¹ x✝ : CategoryTheory.CostructuredArrow FintypeCat.toProfinite.op { unop :=  …
        f : Quiver.Hom x✝¹ x✝
        this : Eq (CategoryTheory.CategoryStruct.comp (FintypeCat.toProfinite.map f.le …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
      -/
      simp [← map_comp, this]) }
      /-
        🎉 no goals
      -/


/--
If `c` is a limit cone, `G.mapCocone c.op` is a colimit cone and the projection maps in `c`
are epimorphic, then `cocone G c.pt` is a colimit cone.
-/
noncomputable
def isColimitCocone (hc : IsLimit c) [∀ i, Epi (c.π.app i)] (hc' : IsColimit <| G.mapCocone c.op) :
    IsColimit (cocone G c.pt) := (functorOp_final c hc).isColimitWhiskerEquiv _ _ hc'


/--
A functor `StructuredArrow S toProfinite ⥤ FintypeCat` whose limit in `Profinite` is isomorphic
to `S`.
-/
abbrev fintypeDiagram' : StructuredArrow S toProfinite ⥤ FintypeCat :=
  StructuredArrow.proj S toProfinite


/-- An abbreviation for `S.fintypeDiagram' ⋙ toProfinite`. -/
abbrev diagram' : StructuredArrow S toProfinite ⥤ Profinite :=
  S.fintypeDiagram' ⋙ toProfinite


/-- A cone over `S.diagram'` whose cone point is `S`. -/
abbrev asLimitCone' : Cone (S.diagram') := cone (𝟭 _) S


instance (i : DiscreteQuotient S) : Epi (S.asLimitCone.π.app i) :=
  (epi_iff_surjective _).mpr i.proj_surjective


/-- `S.asLimitCone'` is a limit cone. -/
noncomputable def asLimit' : IsLimit S.asLimitCone' := isLimitCone _ (𝟭 _) S.asLimit S.asLimit


/-- A bundled version of `S.asLimitCone'` and `S.asLimit'`. -/
noncomputable def lim' : LimitCone S.diagram' := ⟨S.asLimitCone', S.asLimit'⟩


