/-- The presheaf of dependently typed functions on `X`, with fibres given by a type family `T`.
There is no requirement that the functions are continuous, here.
-/
def presheafToTypes (T : X → Type v) : X.Presheaf (Type v) where
  obj U := ∀ x : U.unop, T x
  map {_ V} i g := fun x : V.unop => g (i.unop x)
  map_id U := by
    /-
      X : TopCat
      T : ↑X → Type v
      U : Opposite (TopologicalSpace.Opens ↑X)
      ⊢ Eq ({ obj := fun U => (x : Subtype fun x => Membership.mem (Opposite.unop U) …
    -/
    ext g
    /-
      case h.h
      X : TopCat
      T : ↑X → Type v
      U : Opposite (TopologicalSpace.Opens ↑X)
      g : { obj := fun U => (x : Subtype fun x => Membership.mem (Opposite.unop U) x …
      x✝ : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Eq ({ obj := fun U => (x : Subtype fun x => Membership.mem (Opposite.unop U) …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp {_ _ _} _ _ := rfl


@[simp]
theorem presheafToTypes_obj {T : X → Type v} {U : (Opens X)ᵒᵖ} :
    (presheafToTypes X T).obj U = ∀ x : U.unop, T x :=
  rfl


@[simp]
theorem presheafToTypes_map {T : X → Type v} {U V : (Opens X)ᵒᵖ} {i : U ⟶ V} {f} :
    (presheafToTypes X T).map i f = fun x => f (i.unop x) :=
  rfl

-- We don't just define this in terms of `presheafToTypes`,
-- as it's helpful later to see (at a syntactic level) that `(presheafToType X T).obj U`
-- is a non-dependent function.
-- We don't use `@[simps]` to generate the projection lemmas here,
-- as it turns out to be useful to have `presheafToType_map`
-- written as an equality of functions (rather than being applied to some argument).

/-- The presheaf of functions on `X` with values in a type `T`.
There is no requirement that the functions are continuous, here.
-/
def presheafToType (T : Type v) : X.Presheaf (Type v) where
  obj U := U.unop → T
  map {_ _} i g := g ∘ i.unop
  map_id U := by
    /-
      X : TopCat
      T : Type v
      U : Opposite (TopologicalSpace.Opens ↑X)
      ⊢ Eq ({ obj := fun U => (Subtype fun x => Membership.mem (Opposite.unop U) x)  …
    -/
    ext g
    /-
      case h.h
      X : TopCat
      T : Type v
      U : Opposite (TopologicalSpace.Opens ↑X)
      g : { obj := fun U => (Subtype fun x => Membership.mem (Opposite.unop U) x) →  …
      x✝ : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Eq ({ obj := fun U => (Subtype fun x => Membership.mem (Opposite.unop U) x)  …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp {_ _ _} _ _ := rfl


@[simp]
theorem presheafToType_obj {T : Type v} {U : (Opens X)ᵒᵖ} :
    (presheafToType X T).obj U = (U.unop → T) :=
  rfl


@[simp]
theorem presheafToType_map {T : Type v} {U V : (Opens X)ᵒᵖ} {i : U ⟶ V} {f} :
    (presheafToType X T).map i f = f ∘ i.unop :=
  rfl


/-- The presheaf of continuous functions on `X` with values in fixed target topological space
`T`. -/
def presheafToTop (T : TopCat.{v}) : X.Presheaf (Type v) :=
  (Opens.toTopCat X).op ⋙ yoneda.obj T


@[simp]
theorem presheafToTop_obj (T : TopCat.{v}) (U : (Opens X)ᵒᵖ) :
    (presheafToTop X T).obj U = ((Opens.toTopCat X).obj (unop U) ⟶ T) :=
  rfl


