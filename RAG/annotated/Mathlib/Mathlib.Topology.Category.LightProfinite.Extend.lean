/--
Given a sequential cone in `LightProfinite` consisting of finite sets,
we obtain a functor from the indexing category to `StructuredArrow c.pt toLightProfinite`.
-/
@[simps]
def functor : ℕᵒᵖ ⥤ StructuredArrow c.pt toLightProfinite where
  obj i := StructuredArrow.mk (c.π.app i)
  map f := StructuredArrow.homMk (F.map f) (c.w f)

-- We check that the original diagram factors through `LightProfinite.Extend.functor`.

/--
Given a sequential cone in `LightProfinite` consisting of finite sets,
we obtain a functor from the opposite of the indexing category to
`CostructuredArrow toProfinite.op ⟨c.pt⟩`.
-/
@[simps! obj map]
def functorOp : ℕ ⥤ CostructuredArrow toLightProfinite.op ⟨c.pt⟩ :=
  (functor c).rightOp ⋙ StructuredArrow.toCostructuredArrow _ _

-- We check that the opposite of the original diagram factors through `Profinite.Extend.functorOp`.

/--
If the projection maps in the cone are epimorphic and the cone is limiting, then
`LightProfinite.Extend.functor` is initial.
-/
theorem functor_initial (hc : IsLimit c) [∀ i, Epi (c.π.app i)] : Initial (functor c) := by
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    ⊢ (LightProfinite.Extend.functor c).Initial
  -/
  rw [initial_iff_comp_equivalence _ (StructuredArrow.post _ _ lightToProfinite)]
  have : ∀ i, Epi ((lightToProfinite.mapCone c).π.app i) :=
    fun i ↦ inferInstanceAs (Epi (lightToProfinite.map (c.π.app i)))
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    this : ∀ (i : Opposite Nat), CategoryTheory.Epi ((lightToProfinite.mapCone c). …
    ⊢ ((LightProfinite.Extend.functor c).comp (CategoryTheory.StructuredArrow.post …
  -/
  exact Profinite.Extend.functor_initial _ (isLimitOfPreserves lightToProfinite hc)
  /-
    🎉 no goals
  -/


/--
If the projection maps in the cone are epimorphic and the cone is limiting, then
`LightProfinite.Extend.functorOp` is final.
-/
theorem functorOp_final (hc : IsLimit c) [∀ i, Epi (c.π.app i)] : Final (functorOp c) := by
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    ⊢ (LightProfinite.Extend.functorOp c).Final
  -/
  have := functor_initial c hc
  have : ((StructuredArrow.toCostructuredArrow toLightProfinite c.pt)).IsEquivalence  :=
    (inferInstance : (structuredArrowOpEquivalence _ _).functor.IsEquivalence )
  have : (functor c).rightOp.Final :=
    inferInstanceAs ((opOpEquivalence ℕ).inverse ⋙ (functor c).op).Final
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    this✝¹ : (LightProfinite.Extend.functor c).Initial
    this✝ : (CategoryTheory.StructuredArrow.toCostructuredArrow FintypeCat.toLight …
    this : (LightProfinite.Extend.functor c).rightOp.Final
    ⊢ (LightProfinite.Extend.functorOp c).Final
  -/
  exact Functor.final_comp (functor c).rightOp _
  /-
    🎉 no goals
  -/


/--
Given a functor `G` from `LightProfinite` and `S : LightProfinite`, we obtain a cone on
`(StructuredArrow.proj S toLightProfinite ⋙ toLightProfinite ⋙ G)` with cone point `G.obj S`.

Whiskering this cone with `LightProfinite.Extend.functor c` gives `G.mapCone c` as we check in the
example below.
-/
def cone (S : LightProfinite) :
    Cone (StructuredArrow.proj S toLightProfinite ⋙ toLightProfinite ⋙ G) where
  pt := G.obj S
  π := {
    app := fun i ↦ G.map i.hom
    naturality := fun _ _ f ↦ (by
      /-
        F : CategoryTheory.Functor (Opposite Nat) FintypeCat
        c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.48182, u_1} C
        G : CategoryTheory.Functor LightProfinite C
        S : LightProfinite
        x✝¹ x✝ : CategoryTheory.StructuredArrow S FintypeCat.toLightProfinite
        f : Quiver.Hom x✝¹ x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
      -/
      have := f.w
      simp only [const_obj_obj, StructuredArrow.left_eq_id, const_obj_map, Category.id_comp,
        StructuredArrow.w] at this
      simp only [const_obj_obj, comp_obj, StructuredArrow.proj_obj, const_obj_map, Category.id_comp,
        Functor.comp_map, StructuredArrow.proj_map, ← map_comp, StructuredArrow.w]) }


/--
If `c` and `G.mapCone c` are limit cones and the projection maps in `c` are epimorphic,
then `cone G c.pt` is a limit cone.
-/
noncomputable
def isLimitCone (hc : IsLimit c) [∀ i, Epi (c.π.app i)] (hc' : IsLimit <| G.mapCone c) :
    IsLimit (cone G c.pt) := (functor_initial c hc).isLimitWhiskerEquiv _ _ hc'


/--
Given a functor `G` from `LightProfiniteᵒᵖ` and `S : LightProfinite`, we obtain a cocone on
`(CostructuredArrow.proj toLightProfinite.op ⟨S⟩ ⋙ toLightProfinite.op ⋙ G)` with cocone point
`G.obj ⟨S⟩`.

Whiskering this cocone with `LightProfinite.Extend.functorOp c` gives `G.mapCocone c.op` as we
check in the example below.
-/
@[simps]
def cocone (S : LightProfinite) :
    Cocone (CostructuredArrow.proj toLightProfinite.op ⟨S⟩ ⋙ toLightProfinite.op ⋙ G) where
  pt := G.obj ⟨S⟩
  ι := {
    app := fun i ↦ G.map i.hom
    naturality := fun _ _ f ↦ (by
      /-
        F : CategoryTheory.Functor (Opposite Nat) FintypeCat
        c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.58034, u_1} C
        G : CategoryTheory.Functor (Opposite LightProfinite) C
        S : LightProfinite
        x✝¹ x✝ : CategoryTheory.CostructuredArrow FintypeCat.toLightProfinite.op { uno …
        f : Quiver.Hom x✝¹ x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
      -/
      have := f.w
      simp only [op_obj, const_obj_obj, op_map, CostructuredArrow.right_eq_id, const_obj_map,
        Category.comp_id] at this
      simp only [comp_obj, CostructuredArrow.proj_obj, op_obj, const_obj_obj, Functor.comp_map,
        CostructuredArrow.proj_map, op_map, ← map_comp, this, const_obj_map, Category.comp_id]) }


/--
If `c` is a limit cone, `G.mapCocone c.op` is a colimit cone and the projection maps in `c`
are epimorphic, then `cocone G c.pt` is a colimit cone.
-/
noncomputable
def isColimitCocone (hc : IsLimit c) [∀ i, Epi (c.π.app i)] (hc' : IsColimit <| G.mapCocone c.op) :
    IsColimit (cocone G c.pt) :=
  haveI := functorOp_final c hc
  (Functor.final_comp (opOpEquivalence ℕ).functor (functorOp c)).isColimitWhiskerEquiv _ _ hc'


/--
A functor `StructuredArrow S toLightProfinite ⥤ FintypeCat` whose limit in `LightProfinite` is
isomorphic to `S`.
-/
abbrev fintypeDiagram' : StructuredArrow S toLightProfinite ⥤ FintypeCat :=
  StructuredArrow.proj S toLightProfinite


/-- An abbreviation for `S.fintypeDiagram' ⋙ toLightProfinite`. -/
abbrev diagram' : StructuredArrow S toLightProfinite ⥤ LightProfinite :=
  S.fintypeDiagram' ⋙ toLightProfinite


/-- A cone over `S.diagram'` whose cone point is `S`. -/
def asLimitCone' : Cone (S.diagram') := cone (𝟭 _) S


instance (i : ℕᵒᵖ) : Epi (S.asLimitCone.π.app i) :=
  (epi_iff_surjective _).mpr (S.proj_surjective _)


/-- `S.asLimitCone'` is a limit cone. -/
noncomputable def asLimit' : IsLimit S.asLimitCone' := isLimitCone _ (𝟭 _) S.asLimit S.asLimit


/-- A bundled version of `S.asLimitCone'` and `S.asLimit'`. -/
noncomputable def lim' : LimitCone S.diagram' := ⟨S.asLimitCone', S.asLimit'⟩


