/-- The fundamental group is the automorphism group (vertex group) of the basepoint
in the fundamental groupoid. -/
def FundamentalGroup (X : Type u) [TopologicalSpace X] (x : X) :=
  @Aut (FundamentalGroupoid X) _ ⟨x⟩


instance (X : Type u) [TopologicalSpace X] (x : X) : Group (FundamentalGroup X x) := by
  /-
    X✝ : Type u
    inst✝¹ : TopologicalSpace X✝
    x₀ x₁ : X✝
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Group (FundamentalGroup X x)
  -/
  dsimp only [FundamentalGroup]
  /-
    X✝ : Type u
    inst✝¹ : TopologicalSpace X✝
    x₀ x₁ : X✝
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Group (CategoryTheory.Aut { as := x })
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (X : Type u) [TopologicalSpace X] (x : X) : Inhabited (FundamentalGroup X x) := by
  /-
    X✝ : Type u
    inst✝¹ : TopologicalSpace X✝
    x₀ x₁ : X✝
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Inhabited (FundamentalGroup X x)
  -/
  dsimp only [FundamentalGroup]
  /-
    X✝ : Type u
    inst✝¹ : TopologicalSpace X✝
    x₀ x₁ : X✝
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Inhabited (CategoryTheory.Aut { as := x })
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Get an isomorphism between the fundamental groups at two points given a path -/
def fundamentalGroupMulEquivOfPath (p : Path x₀ x₁) :
    FundamentalGroup X x₀ ≃* FundamentalGroup X x₁ :=
  Aut.autMulEquivOfIso (asIso ⟦p⟧)


/-- The fundamental group of a path connected space is independent of the choice of basepoint. -/
def fundamentalGroupMulEquivOfPathConnected [PathConnectedSpace X] :
    FundamentalGroup X x₀ ≃* FundamentalGroup X x₁ :=
  fundamentalGroupMulEquivOfPath (PathConnectedSpace.somePath x₀ x₁)


/-- An element of the fundamental group as an arrow in the fundamental groupoid. -/
abbrev toArrow {X : TopCat} {x : X} (p : FundamentalGroup X x) :
    FundamentalGroupoid.mk x ⟶ FundamentalGroupoid.mk x :=
  p.hom


/-- An element of the fundamental group as a quotient of homotopic paths. -/
abbrev toPath {X : TopCat} {x : X} (p : FundamentalGroup X x) : Path.Homotopic.Quotient x x :=
  toArrow p


/-- An element of the fundamental group, constructed from an arrow in the fundamental groupoid. -/
abbrev fromArrow {X : TopCat} {x : X}
    (p : FundamentalGroupoid.mk x ⟶ FundamentalGroupoid.mk x) :
    FundamentalGroup X x where
  hom := p
  inv := CategoryTheory.Groupoid.inv p


/-- An element of the fundamental group, constructed from a quotient of homotopic paths. -/
abbrev fromPath {X : TopCat} {x : X} (p : Path.Homotopic.Quotient x x) : FundamentalGroup X x :=
  fromArrow p


