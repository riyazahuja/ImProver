/-- The object part of the compactification functor from types to Stonean spaces. -/
def stoneCechObj (X : Type u) : Stonean :=
  letI : TopologicalSpace X := ⊥
  haveI : DiscreteTopology X := ⟨rfl⟩
  haveI : ExtremallyDisconnected (StoneCech X) :=
    CompactT2.Projective.extremallyDisconnected StoneCech.projective
  of (StoneCech X)


/-- The equivalence of homsets to establish the adjunction between the Stone-Cech compactification
functor and the forgetful functor. -/
noncomputable def stoneCechEquivalence (X : Type u) (Y : Stonean.{u}) :
    (stoneCechObj X ⟶ Y) ≃ (X ⟶ (forget Stonean).obj Y) := by
  /-
    X : Type u
    Y : Stonean
    ⊢ Equiv (Quiver.Hom (Stonean.stoneCechObj X) Y) (Quiver.Hom X ((CategoryTheory …
  -/
  letI : TopologicalSpace X := ⊥
  /-
    X : Type u
    Y : Stonean
    this : TopologicalSpace X := Bot.bot
    ⊢ Equiv (Quiver.Hom (Stonean.stoneCechObj X) Y) (Quiver.Hom X ((CategoryTheory …
  -/
  haveI : DiscreteTopology X := ⟨rfl⟩
  /-
    X : Type u
    Y : Stonean
    this✝ : TopologicalSpace X := Bot.bot
    this : DiscreteTopology X
    ⊢ Equiv (Quiver.Hom (Stonean.stoneCechObj X) Y) (Quiver.Hom X ((CategoryTheory …
  -/
  refine fullyFaithfulToCompHaus.homEquiv.trans ?_
  exact (_root_.stoneCechEquivalence (TopCat.of X) (toCompHaus.obj Y)).trans
    (TopCat.adj₁.homEquiv _ _)


/-- The Stone-Cech compactification functor from types to Stonean spaces. -/
noncomputable def typeToStonean : Type u ⥤ Stonean.{u} :=
  leftAdjointOfEquiv Stonean.stoneCechEquivalence fun _ _ _ _ _ => rfl


/-- The Stone-Cech compactification functor is left adjoint to the forgetful functor. -/
noncomputable def stoneCechAdjunction : typeToStonean ⊣ (forget Stonean) :=
  adjunctionOfEquivLeft stoneCechEquivalence fun _ _ _ _ _ => rfl


/-- The forgetful functor from Stonean spaces, being a right adjoint, preserves limits. -/
noncomputable instance forget.preservesLimits : Limits.PreservesLimits (forget Stonean) :=
  rightAdjoint_preservesLimits stoneCechAdjunction


