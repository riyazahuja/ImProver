/-- A reformulation of `CategoryTheory.Paths.induction` in terms of `MorphismProperty`. -/
lemma morphismProperty_eq_top
    (P : MorphismProperty (Paths V))
    (id : ∀ {v : V}, P (𝟙 (of.obj v)))
    (comp : ∀ {u v w : V} (p : of.obj u ⟶ of.obj v) (q : v ⟶ w), P p → P (p ≫ of.map q)) :
    P = ⊤ := by
  /-
    V : Type u₁
    inst✝ : Quiver V
    P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
    id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
    comp : ∀ {u v w : V} (p : Quiver.Hom (CategoryTheory.Paths.of.obj u) (Category …
    ⊢ Eq P Top.top
  -/
  ext; constructor
    /-
      case h.mp
      V : Type u₁
      inst✝ : Quiver V
      P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
      id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
      comp : ∀ {u v w : V} (p : Quiver.Hom (CategoryTheory.Paths.of.obj u) (Category …
      X✝ Y✝ : CategoryTheory.Paths V
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ P f✝ → Top.top f✝
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      V : Type u₁
      inst✝ : Quiver V
      P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
      id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
      comp : ∀ {u v w : V} (p : Quiver.Hom (CategoryTheory.Paths.of.obj u) (Category …
      X✝ Y✝ : CategoryTheory.Paths V
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Top.top f✝ → P f✝
    -/
  · exact fun _ ↦ induction (fun f ↦ P f) id comp _
    /-
      🎉 no goals
    -/


/-- A reformulation of `CategoryTheory.Paths.induction'` in terms of `MorphismProperty`. -/
lemma morphismProperty_eq_top'
    (P : MorphismProperty (Paths V))
    (id : ∀ {v : V}, P (𝟙 (of.obj v)))
    (comp : ∀ {u v w : V} (p : u ⟶ v) (q : of.obj v ⟶ of.obj w), P q → P (of.map p ≫ q)) :
    P = ⊤ := by
  /-
    V : Type u₁
    inst✝ : Quiver V
    P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
    id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
    comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
    ⊢ Eq P Top.top
  -/
  ext; constructor
    /-
      case h.mp
      V : Type u₁
      inst✝ : Quiver V
      P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
      id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
      comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
      X✝ Y✝ : CategoryTheory.Paths V
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ P f✝ → Top.top f✝
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      V : Type u₁
      inst✝ : Quiver V
      P : CategoryTheory.MorphismProperty (CategoryTheory.Paths V)
      id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
      comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
      X✝ Y✝ : CategoryTheory.Paths V
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Top.top f✝ → P f✝
    -/
  · exact fun _ ↦ induction' (fun f ↦ P f) id comp _
    /-
      🎉 no goals
    -/


