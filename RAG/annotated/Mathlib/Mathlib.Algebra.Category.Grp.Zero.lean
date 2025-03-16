@[to_additive]
theorem isZero_of_subsingleton (G : Grp) [Subsingleton G] : IsZero G := by
  /-
    G : Grp
    inst✝ : Subsingleton ↑G
    ⊢ CategoryTheory.Limits.IsZero G
  -/
  refine ⟨fun X => ⟨⟨⟨1⟩, fun f => ?_⟩⟩, fun X => ⟨⟨⟨1⟩, fun f => ?_⟩⟩⟩
    /-
      case refine_1
      G : Grp
      inst✝ : Subsingleton ↑G
      X : Grp
      f : Quiver.Hom G X
      ⊢ Eq f Inhabited.default
    -/
  · ext x
    /-
      case refine_1.w
      G : Grp
      inst✝ : Subsingleton ↑G
      X : Grp
      f : Quiver.Hom G X
      x : ↑G
      ⊢ Eq (f x) (Inhabited.default x)
    -/
    have : x = 1 := Subsingleton.elim _ _
    /-
      case refine_1.w
      G : Grp
      inst✝ : Subsingleton ↑G
      X : Grp
      f : Quiver.Hom G X
      x : ↑G
      this : Eq x 1
      ⊢ Eq (f x) (Inhabited.default x)
    -/
    rw [this, map_one, map_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Grp
      inst✝ : Subsingleton ↑G
      X : Grp
      f : Quiver.Hom X G
      ⊢ Eq f Inhabited.default
    -/
  · ext
    /-
      case refine_2.w
      G : Grp
      inst✝ : Subsingleton ↑G
      X : Grp
      f : Quiver.Hom X G
      x✝ : ↑X
      ⊢ Eq (f x✝) (Inhabited.default x✝)
    -/
    subsingleton
    /-
      🎉 no goals
    -/


@[to_additive AddGrp.hasZeroObject]
instance : HasZeroObject Grp :=
  ⟨⟨of PUnit, isZero_of_subsingleton _⟩⟩


@[to_additive]
theorem isZero_of_subsingleton (G : CommGrp) [Subsingleton G] : IsZero G := by
  /-
    G : CommGrp
    inst✝ : Subsingleton ↑G
    ⊢ CategoryTheory.Limits.IsZero G
  -/
  refine ⟨fun X => ⟨⟨⟨1⟩, fun f => ?_⟩⟩, fun X => ⟨⟨⟨1⟩, fun f => ?_⟩⟩⟩
    /-
      case refine_1
      G : CommGrp
      inst✝ : Subsingleton ↑G
      X : CommGrp
      f : Quiver.Hom G X
      ⊢ Eq f Inhabited.default
    -/
  · ext x
    /-
      case refine_1.w
      G : CommGrp
      inst✝ : Subsingleton ↑G
      X : CommGrp
      f : Quiver.Hom G X
      x : ↑G
      ⊢ Eq (f x) (Inhabited.default x)
    -/
    have : x = 1 := Subsingleton.elim _ _
    /-
      case refine_1.w
      G : CommGrp
      inst✝ : Subsingleton ↑G
      X : CommGrp
      f : Quiver.Hom G X
      x : ↑G
      this : Eq x 1
      ⊢ Eq (f x) (Inhabited.default x)
    -/
    rw [this, map_one, map_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : CommGrp
      inst✝ : Subsingleton ↑G
      X : CommGrp
      f : Quiver.Hom X G
      ⊢ Eq f Inhabited.default
    -/
  · ext
    /-
      case refine_2.w
      G : CommGrp
      inst✝ : Subsingleton ↑G
      X : CommGrp
      f : Quiver.Hom X G
      x✝ : ↑X
      ⊢ Eq (f x✝) (Inhabited.default x✝)
    -/
    subsingleton
    /-
      🎉 no goals
    -/


@[to_additive AddCommGrp.hasZeroObject]
instance : HasZeroObject CommGrp :=
  ⟨⟨of PUnit, isZero_of_subsingleton _⟩⟩


