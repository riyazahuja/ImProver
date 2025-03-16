/--
`X.Over S` is the typeclass containing the data of a structure morphism `X ↘ S : X ⟶ S`.
-/
protected abbrev Over (X S : Scheme.{u}) := OverClass X S


/--
`X.CanonicallyOver S` is the typeclass containing the data of a structure morphism `X ↘ S : X ⟶ S`,
and that `S` is (uniquely) inferable from the structure of `X`.
-/
abbrev CanonicallyOver := CanonicallyOverClass X S


/-- Given `X.Over S` and `Y.Over S` and `f : X ⟶ Y`,
`f.IsOver S` is the typeclass asserting `f` commutes with the structure morphisms. -/
abbrev Hom.IsOver (f : X.Hom Y) (S : Scheme.{u}) [X.Over S] [Y.Over S] := HomIsOver f S


@[simp]
lemma Hom.isOver_iff [X.Over S] [Y.Over S] {f : X ⟶ Y} : f.IsOver S ↔ f ≫ Y ↘ S = X ↘ S :=
  ⟨fun H ↦ H.1, fun h ↦ ⟨h⟩⟩


/-- Given `X.Over S`, this is the bundled object of `Over S`. -/
abbrev asOver (X S : Scheme.{u}) [X.Over S] := OverClass.asOver X S


/-- Given a morphism `X ⟶ Y` with `f.IsOver S`, this is the bundled morphism in `Over S`. -/
abbrev Hom.asOver (f : X.Hom Y) (S : Scheme.{u}) [X.Over S] [Y.Over S] [f.IsOver S] :=
  OverClass.asOverHom S f


