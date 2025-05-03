/-- This structure contains the data of two left fractions for
`W : MorphismProperty C` that have the same "denominator". -/
structure LeftFraction₂ (X Y : C) where
  /-- the auxiliary object of left fractions -/
  {Y' : C}
  /-- the numerator of the first left fraction -/
  f : X ⟶ Y'
  /-- the numerator of the second left fraction -/
  f' : X ⟶ Y'
  /-- the denominator of the left fractions -/
  s : Y ⟶ Y'
  /-- the condition that the denominator belongs to the given morphism property -/
  hs : W s


/-- This structure contains the data of three left fractions for
`W : MorphismProperty C` that have the same "denominator". -/
structure LeftFraction₃ (X Y : C) where
  /-- the auxiliary object of left fractions -/
  {Y' : C}
  /-- the numerator of the first left fraction -/
  f : X ⟶ Y'
  /-- the numerator of the second left fraction -/
  f' : X ⟶ Y'
  /-- the numerator of the third left fraction -/
  f'' : X ⟶ Y'
  /-- the denominator of the left fractions -/
  s : Y ⟶ Y'
  /-- the condition that the denominator belongs to the given morphism property -/
  hs : W s


/-- This structure contains the data of two right fractions for
`W : MorphismProperty C` that have the same "denominator". -/
structure RightFraction₂ (X Y : C) where
  /-- the auxiliary object of right fractions -/
  {X' : C}
  /-- the denominator of the right fractions -/
  s : X' ⟶ X
  /-- the condition that the denominator belongs to the given morphism property -/
  hs : W s
  /-- the numerator of the first right fraction -/
  f : X' ⟶ Y
  /-- the numerator of the second right fraction -/
  f' : X' ⟶ Y


/-- The equivalence relation on tuples of left fractions with the same denominator
for a morphism property `W`. The fact it is an equivalence relation is not
formalized, but it would follow easily from `LeftFraction₂.map_eq_iff`. -/
def LeftFraction₂Rel {X Y : C} (z₁ z₂ : W.LeftFraction₂ X Y) : Prop :=
  ∃ (Z : C) (t₁ : z₁.Y' ⟶ Z) (t₂ : z₂.Y' ⟶ Z) (_ : z₁.s ≫ t₁ = z₂.s ≫ t₂)
    (_ : z₁.f ≫ t₁ = z₂.f ≫ t₂) (_ : z₁.f' ≫ t₁ = z₂.f' ≫ t₂), W (z₁.s ≫ t₁)


/-- The first left fraction. -/
abbrev fst : W.LeftFraction X Y where
  Y' := φ.Y'
  f := φ.f
  s := φ.s
  hs := φ.hs


/-- The second left fraction. -/
abbrev snd : W.LeftFraction X Y where
  Y' := φ.Y'
  f := φ.f'
  s := φ.s
  hs := φ.hs


/-- The exchange of the two fractions. -/
abbrev symm : W.LeftFraction₂ X Y where
  Y' := φ.Y'
  f := φ.f'
  f' := φ.f
  s := φ.s
  hs := φ.hs


/-- The third left fraction. -/
abbrev thd : W.LeftFraction X Y where
  Y' := φ.Y'
  f := φ.f''
  s := φ.s
  hs := φ.hs


/-- Forgets the first fraction. -/
abbrev forgetFst : W.LeftFraction₂ X Y where
  Y' := φ.Y'
  f := φ.f'
  f' := φ.f''
  s := φ.s
  hs := φ.hs


/-- Forgets the second fraction. -/
abbrev forgetSnd : W.LeftFraction₂ X Y where
  Y' := φ.Y'
  f := φ.f
  f' := φ.f''
  s := φ.s
  hs := φ.hs


/-- Forgets the third fraction. -/
abbrev forgetThd : W.LeftFraction₂ X Y where
  Y' := φ.Y'
  f := φ.f
  f' := φ.f'
  s := φ.s
  hs := φ.hs


lemma fst (h : LeftFraction₂Rel z₁ z₂) : LeftFractionRel z₁.fst z₂.fst := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction₂ X Y
    h : CategoryTheory.MorphismProperty.LeftFraction₂Rel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.fst z₂.fst
  -/
  obtain ⟨Z, t₁, t₂, hst, hft, _, ht⟩ := h
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction₂ X Y
    Z : C
    t₁ : Quiver.Hom z₁.Y' Z
    t₂ : Quiver.Hom z₂.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    w✝ : Eq (CategoryTheory.CategoryStruct.comp z₁.f' t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.fst z₂.fst
  -/
  exact ⟨Z, t₁, t₂, hst, hft, ht⟩
  /-
    🎉 no goals
  -/


lemma snd (h : LeftFraction₂Rel z₁ z₂) : LeftFractionRel z₁.snd z₂.snd := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction₂ X Y
    h : CategoryTheory.MorphismProperty.LeftFraction₂Rel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.snd z₂.snd
  -/
  obtain ⟨Z, t₁, t₂, hst, _, hft', ht⟩ := h
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction₂ X Y
    Z : C
    t₁ : Quiver.Hom z₁.Y' Z
    t₂ : Quiver.Hom z₂.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    w✝ : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.CategoryS …
    hft' : Eq (CategoryTheory.CategoryStruct.comp z₁.f' t₁) (CategoryTheory.Catego …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.snd z₂.snd
  -/
  exact ⟨Z, t₁, t₂, hst, hft', ht⟩
  /-
    🎉 no goals
  -/


lemma map_eq_iff {X Y : C} (φ ψ : W.LeftFraction₂ X Y) :
    (φ.fst.map L (Localization.inverts _ _) = ψ.fst.map L (Localization.inverts _ _) ∧
    φ.snd.map L (Localization.inverts _ _) = ψ.snd.map L (Localization.inverts _ _)) ↔
      LeftFraction₂Rel φ ψ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ ψ : W.LeftFraction₂ X Y
    ⊢ Iff (And (Eq (φ.fst.map L ⋯) (ψ.fst.map L ⋯)) (Eq (φ.snd.map L ⋯) (ψ.snd.map …
  -/
  simp only [LeftFraction.map_eq_iff L W]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ ψ : W.LeftFraction₂ X Y
    ⊢ Iff (And (CategoryTheory.MorphismProperty.LeftFractionRel φ.fst ψ.fst) (Cate …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      ⊢ And (CategoryTheory.MorphismProperty.LeftFractionRel φ.fst ψ.fst) (CategoryT …
    -/
  · intro ⟨h, h'⟩
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel φ.fst ψ.fst
      h' : CategoryTheory.MorphismProperty.LeftFractionRel φ.snd ψ.snd
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    obtain ⟨Z, t₁, t₂, hst, hft, ht⟩ := h
    /-
      case mp.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      h' : CategoryTheory.MorphismProperty.LeftFractionRel φ.snd ψ.snd
      Z : C
      t₁ : Quiver.Hom φ.fst.Y' Z
      t₂ : Quiver.Hom ψ.fst.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.fst.s t₁) (CategoryTheory.Categ …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.fst.f t₁) (CategoryTheory.Categ …
      ht : W (CategoryTheory.CategoryStruct.comp φ.fst.s t₁)
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    obtain ⟨Z', t₁', t₂', hst', hft', ht'⟩ := h'
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.fst.Y' Z
      t₂ : Quiver.Hom ψ.fst.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.fst.s t₁) (CategoryTheory.Categ …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.fst.f t₁) (CategoryTheory.Categ …
      ht : W (CategoryTheory.CategoryStruct.comp φ.fst.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.snd.Y' Z'
      t₂' : Quiver.Hom ψ.snd.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.snd.s t₁') (CategoryTheory.Cat …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.snd.f t₁') (CategoryTheory.Cat …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.snd.s t₁')
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    dsimp at t₁ t₂ t₁' t₂' hst hft hst' hft' ht ht'
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    have ⟨α, hα⟩ := (RightFraction.mk _ ht (φ.s ≫ t₁')).exists_leftFraction
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      α : W.LeftFraction Z Z'
      hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    simp only [Category.assoc] at hα
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      α : W.LeftFraction Z Z'
      hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    obtain ⟨Z'', u, hu, fac⟩ := HasLeftCalculusOfFractions.ext _ _ _ φ.hs hα
    have hα' : ψ.s ≫ t₂ ≫ α.f ≫ u = ψ.s ≫ t₂' ≫ α.s ≫ u := by
      rw [← reassoc_of% hst, ← reassoc_of% hα, ← reassoc_of% hst']
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      α : W.LeftFraction Z Z'
      hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
      Z'' : C
      u : Quiver.Hom α.Y' Z''
      hu : W u
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    obtain ⟨Z''', u', hu', fac'⟩ := HasLeftCalculusOfFractions.ext _ _ _ ψ.hs hα'
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      α : W.LeftFraction Z Z'
      hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
      Z'' : C
      u : Quiver.Hom α.Y' Z''
      hu : W u
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
      Z''' : C
      u' : Quiver.Hom Z'' Z'''
      hu' : W u'
      fac' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    simp only [Category.assoc] at fac fac'
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      Z : C
      t₁ : Quiver.Hom φ.Y' Z
      t₂ : Quiver.Hom ψ.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
      hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
      ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
      Z' : C
      t₁' : Quiver.Hom φ.Y' Z'
      t₂' : Quiver.Hom ψ.Y' Z'
      hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
      hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
      ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
      α : W.LeftFraction Z Z'
      hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
      Z'' : C
      u : Quiver.Hom α.Y' Z''
      hu : W u
      hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
      Z''' : C
      u' : Quiver.Hom Z'' Z'''
      hu' : W u'
      fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
      fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
    -/
    refine ⟨Z''', t₁' ≫ α.s ≫ u ≫ u', t₂' ≫ α.s ≫ u ≫ u', ?_, ?_, ?_, ?_⟩
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Category.{u_4, u_2} D
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        X Y : C
        φ ψ : W.LeftFraction₂ X Y
        Z : C
        t₁ : Quiver.Hom φ.Y' Z
        t₂ : Quiver.Hom ψ.Y' Z
        hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
        hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
        ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
        Z' : C
        t₁' : Quiver.Hom φ.Y' Z'
        t₂' : Quiver.Hom ψ.Y' Z'
        hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
        hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
        ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
        α : W.LeftFraction Z Z'
        hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
        Z'' : C
        u : Quiver.Hom α.Y' Z''
        hu : W u
        hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
        Z''' : C
        u' : Quiver.Hom Z'' Z'''
        hu' : W u'
        fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
        fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct.co …
      -/
    · rw [reassoc_of% hst']
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Category.{u_4, u_2} D
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        X Y : C
        φ ψ : W.LeftFraction₂ X Y
        Z : C
        t₁ : Quiver.Hom φ.Y' Z
        t₂ : Quiver.Hom ψ.Y' Z
        hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
        hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
        ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
        Z' : C
        t₁' : Quiver.Hom φ.Y' Z'
        t₂' : Quiver.Hom ψ.Y' Z'
        hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
        hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
        ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
        α : W.LeftFraction Z Z'
        hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
        Z'' : C
        u : Quiver.Hom α.Y' Z''
        hu : W u
        hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
        Z''' : C
        u' : Quiver.Hom Z'' Z'''
        hu' : W u'
        fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
        fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.f (CategoryTheory.CategoryStruct.co …
      -/
    · rw [reassoc_of% fac, reassoc_of% hft, fac']
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Category.{u_4, u_2} D
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        X Y : C
        φ ψ : W.LeftFraction₂ X Y
        Z : C
        t₁ : Quiver.Hom φ.Y' Z
        t₂ : Quiver.Hom ψ.Y' Z
        hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
        hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
        ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
        Z' : C
        t₁' : Quiver.Hom φ.Y' Z'
        t₂' : Quiver.Hom ψ.Y' Z'
        hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
        hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
        ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
        α : W.LeftFraction Z Z'
        hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
        Z'' : C
        u : Quiver.Hom α.Y' Z''
        hu : W u
        hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
        Z''' : C
        u' : Quiver.Hom Z'' Z'''
        hu' : W u'
        fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
        fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.f' (CategoryTheory.CategoryStruct.c …
      -/
    · rw [reassoc_of% hft']
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Category.{u_4, u_2} D
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        X Y : C
        φ ψ : W.LeftFraction₂ X Y
        Z : C
        t₁ : Quiver.Hom φ.Y' Z
        t₂ : Quiver.Hom ψ.Y' Z
        hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
        hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
        ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
        Z' : C
        t₁' : Quiver.Hom φ.Y' Z'
        t₂' : Quiver.Hom ψ.Y' Z'
        hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
        hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
        ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
        α : W.LeftFraction Z Z'
        hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
        Z'' : C
        u : Quiver.Hom α.Y' Z''
        hu : W u
        hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
        Z''' : C
        u' : Quiver.Hom Z'' Z'''
        hu' : W u'
        fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
        fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
        ⊢ W (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct.com …
      -/
    · rw [← Category.assoc]
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Category.{u_4, u_2} D
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        X Y : C
        φ ψ : W.LeftFraction₂ X Y
        Z : C
        t₁ : Quiver.Hom φ.Y' Z
        t₂ : Quiver.Hom ψ.Y' Z
        hst : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁) (CategoryTheory.CategoryS …
        hft : Eq (CategoryTheory.CategoryStruct.comp φ.f t₁) (CategoryTheory.CategoryS …
        ht : W (CategoryTheory.CategoryStruct.comp φ.s t₁)
        Z' : C
        t₁' : Quiver.Hom φ.Y' Z'
        t₂' : Quiver.Hom ψ.Y' Z'
        hst' : Eq (CategoryTheory.CategoryStruct.comp φ.s t₁') (CategoryTheory.Categor …
        hft' : Eq (CategoryTheory.CategoryStruct.comp φ.f' t₁') (CategoryTheory.Catego …
        ht' : W (CategoryTheory.CategoryStruct.comp φ.s t₁')
        α : W.LeftFraction Z Z'
        hα : Eq (CategoryTheory.CategoryStruct.comp φ.s (CategoryTheory.CategoryStruct …
        Z'' : C
        u : Quiver.Hom α.Y' Z''
        hu : W u
        hα' : Eq (CategoryTheory.CategoryStruct.comp ψ.s (CategoryTheory.CategoryStruc …
        Z''' : C
        u' : Quiver.Hom Z'' Z'''
        hu' : W u'
        fac : Eq (CategoryTheory.CategoryStruct.comp t₁' (CategoryTheory.CategoryStruc …
        fac' : Eq (CategoryTheory.CategoryStruct.comp t₂ (CategoryTheory.CategoryStruc …
        ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ. …
      -/
      exact W.comp_mem _ _ ht' (W.comp_mem _ _ α.hs (W.comp_mem _ _ hu hu'))
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      ⊢ CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ → And (CategoryTheory.M …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction₂ X Y
      h : CategoryTheory.MorphismProperty.LeftFraction₂Rel φ ψ
      ⊢ And (CategoryTheory.MorphismProperty.LeftFractionRel φ.fst ψ.fst) (CategoryT …
    -/
    exact ⟨h.fst, h.snd⟩
    /-
      🎉 no goals
    -/


/-- The first right fraction. -/
abbrev fst : W.RightFraction X Y where
  X' := φ.X'
  f := φ.f
  s := φ.s
  hs := φ.hs


/-- The second right fraction. -/
abbrev snd : W.RightFraction X Y where
  X' := φ.X'
  f := φ.f'
  s := φ.s
  hs := φ.hs


lemma exists_leftFraction₂ [W.HasLeftCalculusOfFractions] :
    ∃ (ψ : W.LeftFraction₂ X Y), φ.f ≫ ψ.s = φ.s ≫ ψ.f ∧
      φ.f' ≫ ψ.s = φ.s ≫ ψ.f' := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction₂ X Y
    inst✝ : W.HasLeftCalculusOfFractions
    ⊢ Exists fun ψ => And (Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (Catego …
  -/
  obtain ⟨ψ₁, hψ₁⟩ := φ.fst.exists_leftFraction
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction₂ X Y
    inst✝ : W.HasLeftCalculusOfFractions
    ψ₁ : W.LeftFraction X Y
    hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.fst.f ψ₁.s) (CategoryTheory.Cat …
    ⊢ Exists fun ψ => And (Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (Catego …
  -/
  obtain ⟨ψ₂, hψ₂⟩ := φ.snd.exists_leftFraction
  /-
    case intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction₂ X Y
    inst✝ : W.HasLeftCalculusOfFractions
    ψ₁ : W.LeftFraction X Y
    hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.fst.f ψ₁.s) (CategoryTheory.Cat …
    ψ₂ : W.LeftFraction X Y
    hψ₂ : Eq (CategoryTheory.CategoryStruct.comp φ.snd.f ψ₂.s) (CategoryTheory.Cat …
    ⊢ Exists fun ψ => And (Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (Catego …
  -/
  obtain ⟨α, hα⟩ := (RightFraction.mk _ ψ₁.hs ψ₂.s).exists_leftFraction
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction₂ X Y
    inst✝ : W.HasLeftCalculusOfFractions
    ψ₁ : W.LeftFraction X Y
    hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.fst.f ψ₁.s) (CategoryTheory.Cat …
    ψ₂ : W.LeftFraction X Y
    hψ₂ : Eq (CategoryTheory.CategoryStruct.comp φ.snd.f ψ₂.s) (CategoryTheory.Cat …
    α : W.LeftFraction ψ₁.Y' ψ₂.Y'
    hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ⊢ Exists fun ψ => And (Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (Catego …
  -/
  dsimp at hψ₁ hψ₂ hα
  refine ⟨LeftFraction₂.mk (ψ₁.f ≫ α.f) (ψ₂.f ≫ α.s) (ψ₂.s ≫ α.s)
      (W.comp_mem _ _ ψ₂.hs α.hs), ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      φ : W.RightFraction₂ X Y
      inst✝ : W.HasLeftCalculusOfFractions
      ψ₁ : W.LeftFraction X Y
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.f ψ₁.s) (CategoryTheory.Categor …
      ψ₂ : W.LeftFraction X Y
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp φ.f' ψ₂.s) (CategoryTheory.Catego …
      α : W.LeftFraction ψ₁.Y' ψ₂.Y'
      hα : Eq (CategoryTheory.CategoryStruct.comp ψ₂.s α.s) (CategoryTheory.Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.f (CategoryTheory.MorphismProperty. …
    -/
  · dsimp
    /-
      case intro.intro.intro.refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      φ : W.RightFraction₂ X Y
      inst✝ : W.HasLeftCalculusOfFractions
      ψ₁ : W.LeftFraction X Y
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.f ψ₁.s) (CategoryTheory.Categor …
      ψ₂ : W.LeftFraction X Y
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp φ.f' ψ₂.s) (CategoryTheory.Catego …
      α : W.LeftFraction ψ₁.Y' ψ₂.Y'
      hα : Eq (CategoryTheory.CategoryStruct.comp ψ₂.s α.s) (CategoryTheory.Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.f (CategoryTheory.CategoryStruct.co …
    -/
    rw [hα, reassoc_of% hψ₁]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      φ : W.RightFraction₂ X Y
      inst✝ : W.HasLeftCalculusOfFractions
      ψ₁ : W.LeftFraction X Y
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp φ.f ψ₁.s) (CategoryTheory.Categor …
      ψ₂ : W.LeftFraction X Y
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp φ.f' ψ₂.s) (CategoryTheory.Catego …
      α : W.LeftFraction ψ₁.Y' ψ₂.Y'
      hα : Eq (CategoryTheory.CategoryStruct.comp ψ₂.s α.s) (CategoryTheory.Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.f' (CategoryTheory.MorphismProperty …
    -/
  · rw [reassoc_of% hψ₂]
    /-
      🎉 no goals
    -/


lemma exists_leftFraction₂ {X Y : C} (f f' : L.obj X ⟶ L.obj Y) :
    ∃ (φ : W.LeftFraction₂ X Y), f = φ.fst.map L (inverts L W) ∧
      f' = φ.snd.map L (inverts L W) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  have ⟨φ, hφ⟩ := exists_leftFraction L W f
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.LeftFraction X Y
    hφ : Eq f (φ.map L ⋯)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  have ⟨φ', hφ'⟩ := exists_leftFraction L W f'
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.LeftFraction X Y
    hφ : Eq f (φ.map L ⋯)
    φ' : W.LeftFraction X Y
    hφ' : Eq f' (φ'.map L ⋯)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  obtain ⟨α, hα⟩ := (RightFraction.mk _ φ.hs φ'.s).exists_leftFraction
  let ψ : W.LeftFraction₂ X Y :=
    { Y' := α.Y'
      f := φ.f ≫ α.f
      f' := φ'.f ≫ α.s
      s := φ'.s ≫ α.s
      hs := W.comp_mem _ _ φ'.hs α.hs }
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.LeftFraction X Y
    hφ : Eq f (φ.map L ⋯)
    φ' : W.LeftFraction X Y
    hφ' : Eq f' (φ'.map L ⋯)
    α : W.LeftFraction φ.Y' φ'.Y'
    hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ψ : W.LeftFraction₂ X Y := CategoryTheory.MorphismProperty.LeftFraction₂.mk (C …
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  have := inverts L W _ φ'.hs
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.LeftFraction X Y
    hφ : Eq f (φ.map L ⋯)
    φ' : W.LeftFraction X Y
    hφ' : Eq f' (φ'.map L ⋯)
    α : W.LeftFraction φ.Y' φ'.Y'
    hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ψ : W.LeftFraction₂ X Y := CategoryTheory.MorphismProperty.LeftFraction₂.mk (C …
    this : CategoryTheory.IsIso (L.map φ'.s)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  have := inverts L W _ α.hs
  have : IsIso (L.map (φ'.s ≫ α.s)) := by
    rw [L.map_comp]
    infer_instance
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.LeftFraction X Y
    hφ : Eq f (φ.map L ⋯)
    φ' : W.LeftFraction X Y
    hφ' : Eq f' (φ'.map L ⋯)
    α : W.LeftFraction φ.Y' φ'.Y'
    hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ψ : W.LeftFraction₂ X Y := CategoryTheory.MorphismProperty.LeftFraction₂.mk (C …
    this✝¹ : CategoryTheory.IsIso (L.map φ'.s)
    this✝ : CategoryTheory.IsIso (L.map α.s)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ'.s α. …
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (Eq f' (φ.snd.map L ⋯))
  -/
  refine ⟨ψ, ?_, ?_⟩
  · rw [← cancel_mono (L.map (φ'.s ≫ α.s)), LeftFraction.map_comp_map_s,
      hα, L.map_comp, hφ, LeftFraction.map_comp_map_s_assoc,
      L.map_comp]
    /-
      case intro.refine_2
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f f' : Quiver.Hom (L.obj X) (L.obj Y)
      φ : W.LeftFraction X Y
      hφ : Eq f (φ.map L ⋯)
      φ' : W.LeftFraction X Y
      hφ' : Eq f' (φ'.map L ⋯)
      α : W.LeftFraction φ.Y' φ'.Y'
      hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
      ψ : W.LeftFraction₂ X Y := CategoryTheory.MorphismProperty.LeftFraction₂.mk (C …
      this✝¹ : CategoryTheory.IsIso (L.map φ'.s)
      this✝ : CategoryTheory.IsIso (L.map α.s)
      this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ'.s α. …
      ⊢ Eq f' (ψ.snd.map L ⋯)
    -/
  · rw [← cancel_mono (L.map (φ'.s ≫ α.s)), hφ']
    /-
      case intro.refine_2
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f f' : Quiver.Hom (L.obj X) (L.obj Y)
      φ : W.LeftFraction X Y
      hφ : Eq f (φ.map L ⋯)
      φ' : W.LeftFraction X Y
      hφ' : Eq f' (φ'.map L ⋯)
      α : W.LeftFraction φ.Y' φ'.Y'
      hα : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
      ψ : W.LeftFraction₂ X Y := CategoryTheory.MorphismProperty.LeftFraction₂.mk (C …
      this✝¹ : CategoryTheory.IsIso (L.map φ'.s)
      this✝ : CategoryTheory.IsIso (L.map α.s)
      this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ'.s α. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ'.map L ⋯) (L.map (CategoryTheory.C …
    -/
    nth_rw 1 [L.map_comp]
    rw [LeftFraction.map_comp_map_s_assoc, LeftFraction.map_comp_map_s,
      L.map_comp]


lemma exists_leftFraction₃ {X Y : C} (f f' f'' : L.obj X ⟶ L.obj Y) :
    ∃ (φ : W.LeftFraction₃ X Y), f = φ.fst.map L (inverts L W) ∧
      f' = φ.snd.map L (inverts L W) ∧
      f'' = φ.thd.map L (inverts L W) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  obtain ⟨α, hα, hα'⟩ := exists_leftFraction₂ L W f f'
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  have ⟨β, hβ⟩ := exists_leftFraction L W f''
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    β : W.LeftFraction X Y
    hβ : Eq f'' (β.map L ⋯)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  obtain ⟨γ, hγ⟩ := (RightFraction.mk _ α.hs β.s).exists_leftFraction
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    β : W.LeftFraction X Y
    hβ : Eq f'' (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  dsimp at hγ
  let ψ : W.LeftFraction₃ X Y :=
    { Y' := γ.Y'
      f := α.f ≫ γ.f
      f' := α.f' ≫ γ.f
      f'' := β.f ≫ γ.s
      s := β.s ≫ γ.s
      hs := W.comp_mem _ _ β.hs γ.hs }
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    β : W.LeftFraction X Y
    hβ : Eq f'' (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
    ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  have := inverts L W _ β.hs
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    β : W.LeftFraction X Y
    hβ : Eq f'' (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
    ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
    this : CategoryTheory.IsIso (L.map β.s)
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  have := inverts L W _ γ.hs
  have : IsIso (L.map (β.s ≫ γ.s)) := by
    rw [L.map_comp]
    infer_instance
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    hα : Eq f (α.fst.map L ⋯)
    hα' : Eq f' (α.snd.map L ⋯)
    β : W.LeftFraction X Y
    hβ : Eq f'' (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
    ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
    this✝¹ : CategoryTheory.IsIso (L.map β.s)
    this✝ : CategoryTheory.IsIso (L.map γ.s)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp β.s γ.s))
    ⊢ Exists fun φ => And (Eq f (φ.fst.map L ⋯)) (And (Eq f' (φ.snd.map L ⋯)) (Eq  …
  -/
  refine ⟨ψ, ?_, ?_, ?_⟩
  · rw [← cancel_mono (L.map (β.s ≫ γ.s)), LeftFraction.map_comp_map_s, hα, hγ,
      L.map_comp, LeftFraction.map_comp_map_s_assoc, L.map_comp]
  · rw [← cancel_mono (L.map (β.s ≫ γ.s)), LeftFraction.map_comp_map_s, hα', hγ,
      L.map_comp, LeftFraction.map_comp_map_s_assoc, L.map_comp]
    /-
      case intro.intro.intro.refine_3
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
      α : W.LeftFraction₂ X Y
      hα : Eq f (α.fst.map L ⋯)
      hα' : Eq f' (α.snd.map L ⋯)
      β : W.LeftFraction X Y
      hβ : Eq f'' (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
      ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp β.s γ.s))
      ⊢ Eq f'' (ψ.thd.map L ⋯)
    -/
  · rw [← cancel_mono (L.map (β.s ≫ γ.s)), hβ]
    /-
      case intro.intro.intro.refine_3
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
      α : W.LeftFraction₂ X Y
      hα : Eq f (α.fst.map L ⋯)
      hα' : Eq f' (α.snd.map L ⋯)
      β : W.LeftFraction X Y
      hβ : Eq f'' (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
      ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp β.s γ.s))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (β.map L ⋯) (L.map (CategoryTheory.Ca …
    -/
    nth_rw 1 [L.map_comp]
    /-
      case intro.intro.intro.refine_3
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f f' f'' : Quiver.Hom (L.obj X) (L.obj Y)
      α : W.LeftFraction₂ X Y
      hα : Eq f (α.fst.map L ⋯)
      hα' : Eq f' (α.snd.map L ⋯)
      β : W.LeftFraction X Y
      hβ : Eq f'' (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.s γ.s) (CategoryTheory.CategoryS …
      ψ : W.LeftFraction₃ X Y := CategoryTheory.MorphismProperty.LeftFraction₃.mk (C …
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp β.s γ.s))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (β.map L ⋯) (CategoryTheory.CategoryS …
    -/
    rw [LeftFraction.map_comp_map_s_assoc, LeftFraction.map_comp_map_s, L.map_comp]
    /-
      🎉 no goals
    -/


