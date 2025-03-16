/-- A left fraction from `X : C` to `Y : C` for `W : MorphismProperty C` consists of the
datum of an object `Y' : C` and maps `f : X ⟶ Y'` and `s : Y ⟶ Y'` such that `W s`. -/
structure LeftFraction (W : MorphismProperty C) (X Y : C) where
  /-- the auxiliary object of a left fraction -/
  {Y' : C}
  /-- the numerator of a left fraction -/
  f : X ⟶ Y'
  /-- the denominator of a left fraction -/
  s : Y ⟶ Y'
  /-- the condition that the denominator belongs to the given morphism property -/
  hs : W s


/-- The left fraction from `X` to `Y` given by a morphism `f : X ⟶ Y`. -/
@[simps]
def ofHom (f : X ⟶ Y) [W.ContainsIdentities] :
    W.LeftFraction X Y := mk f (𝟙 Y) (W.id_mem Y)


/-- The left fraction from `X` to `Y` given by a morphism `s : Y ⟶ X` such that `W s`. -/
@[simps]
def ofInv (s : Y ⟶ X) (hs : W s) :
    W.LeftFraction X Y := mk (𝟙 X) s hs


/-- If `φ : W.LeftFraction X Y` and `L` is a functor which inverts `W`, this is the
induced morphism `L.obj X ⟶ L.obj Y`  -/
noncomputable def map (φ : W.LeftFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    L.obj X ⟶ L.obj Y :=
  have := hL _ φ.hs
  L.map φ.f ≫ inv (L.map φ.s)


@[reassoc (attr := simp)]
lemma map_comp_map_s (φ : W.LeftFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    φ.map L hL ≫ L.map φ.s = L.map φ.f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.map L hL) (L.map φ.s)) (L.map φ.f)
  -/
  letI := hL _ φ.hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    this : CategoryTheory.IsIso (L.map φ.s) := hL φ.s φ.hs
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.map L hL) (L.map φ.s)) (L.map φ.f)
  -/
  simp [map]
  /-
    🎉 no goals
  -/


lemma map_ofHom (f : X ⟶ Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) [W.ContainsIdentities] :
    (ofHom W f).map L hL = L.map f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    inst✝ : W.ContainsIdentities
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.ofHom W f).map L hL) (L.ma …
  -/
  simp [map]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_ofInv_hom_id (s : Y ⟶ X) (hs : W s) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    (ofInv s hs).map L hL ≫ L.map s = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Lef …
  -/
  letI := hL _ hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    this : CategoryTheory.IsIso (L.map s) := hL s hs
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Lef …
  -/
  simp [map]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_hom_ofInv_id (s : Y ⟶ X) (hs : W s) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    L.map s ≫ (ofInv s hs).map L hL = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map s) ((CategoryTheory.MorphismPr …
  -/
  letI := hL _ hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    this : CategoryTheory.IsIso (L.map s) := hL s hs
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map s) ((CategoryTheory.MorphismPr …
  -/
  simp [map]
  /-
    🎉 no goals
  -/


lemma cases (α : W.LeftFraction X Y) :
    ∃ (Y' : C) (f : X ⟶ Y') (s : Y ⟶ Y') (hs : W s), α = LeftFraction.mk f s hs :=
  ⟨_, _, _, _, rfl⟩


/-- A right fraction from `X : C` to `Y : C` for `W : MorphismProperty C` consists of the
datum of an object `X' : C` and maps `s : X' ⟶ X` and `f : X' ⟶ Y` such that `W s`. -/
structure RightFraction (W : MorphismProperty C) (X Y : C) where
  /-- the auxiliary object of a right fraction -/
  {X' : C}
  /-- the denominator of a right fraction -/
  s : X' ⟶ X
  /-- the condition that the denominator belongs to the given morphism property -/
  hs : W s
  /-- the numerator of a right fraction -/
  f : X' ⟶ Y


/-- The right fraction from `X` to `Y` given by a morphism `f : X ⟶ Y`. -/
@[simps]
def ofHom (f : X ⟶ Y) [W.ContainsIdentities] :
    W.RightFraction X Y := mk (𝟙 X) (W.id_mem X) f


/-- The right fraction from `X` to `Y` given by a morphism `s : Y ⟶ X` such that `W s`. -/
@[simps]
def ofInv (s : Y ⟶ X) (hs : W s) :
    W.RightFraction X Y := mk s hs (𝟙 Y)


/-- If `φ : W.RightFraction X Y` and `L` is a functor which inverts `W`, this is the
induced morphism `L.obj X ⟶ L.obj Y`  -/
noncomputable def map (φ : W.RightFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    L.obj X ⟶ L.obj Y :=
  have := hL _ φ.hs
  inv (L.map φ.s) ≫ L.map φ.f


@[reassoc (attr := simp)]
lemma map_s_comp_map (φ : W.RightFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    L.map φ.s ≫ φ.map L hL = L.map φ.f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.s) (φ.map L hL)) (L.map φ.f)
  -/
  letI := hL _ φ.hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    this : CategoryTheory.IsIso (L.map φ.s) := hL φ.s φ.hs
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.s) (φ.map L hL)) (L.map φ.f)
  -/
  simp [map]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_ofHom (f : X ⟶ Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) [W.ContainsIdentities] :
    (ofHom W f).map L hL = L.map f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    inst✝ : W.ContainsIdentities
    ⊢ Eq ((CategoryTheory.MorphismProperty.RightFraction.ofHom W f).map L hL) (L.m …
  -/
  simp [map]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_ofInv_hom_id (s : Y ⟶ X) (hs : W s) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    (ofInv s hs).map L hL ≫ L.map s = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Rig …
  -/
  letI := hL _ hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    s : Quiver.Hom Y X
    hs : W s
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    this : CategoryTheory.IsIso (L.map s) := hL s hs
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Rig …
  -/
  simp [map]
  /-
    🎉 no goals
  -/


lemma cases (α : W.RightFraction X Y) :
    ∃ (X' : C) (s : X' ⟶ X) (hs : W s) (f : X' ⟶ Y) , α = RightFraction.mk s hs f :=
  ⟨_, _, _, _, rfl⟩


/-- A multiplicative morphism property `W` has left calculus of fractions if
any right fraction can be turned into a left fraction and that two morphisms
that can be equalized by precomposition with a morphism in `W` can also
be equalized by postcomposition with a morphism in `W`. -/
class HasLeftCalculusOfFractions extends W.IsMultiplicative : Prop where
  exists_leftFraction ⦃X Y : C⦄ (φ : W.RightFraction X Y) :
    ∃ (ψ : W.LeftFraction X Y), φ.f ≫ ψ.s = φ.s ≫ ψ.f
  ext : ∀ ⦃X' X Y : C⦄ (f₁ f₂ : X ⟶ Y) (s : X' ⟶ X) (_ : W s)
    (_ : s ≫ f₁ = s ≫ f₂), ∃ (Y' : C) (t : Y ⟶ Y') (_ : W t), f₁ ≫ t = f₂ ≫ t


/-- A multiplicative morphism property `W` has right calculus of fractions if
any left fraction can be turned into a right fraction and that two morphisms
that can be equalized by postcomposition with a morphism in `W` can also
be equalized by precomposition with a morphism in `W`. -/
class HasRightCalculusOfFractions extends W.IsMultiplicative : Prop where
  exists_rightFraction ⦃X Y : C⦄ (φ : W.LeftFraction X Y) :
    ∃ (ψ : W.RightFraction X Y), ψ.s ≫ φ.f = ψ.f ≫ φ.s
  ext : ∀ ⦃X Y Y' : C⦄ (f₁ f₂ : X ⟶ Y) (s : Y ⟶ Y') (_ : W s)
    (_ : f₁ ≫ s = f₂ ≫ s), ∃ (X' : C) (t : X' ⟶ X) (_ : W t), t ≫ f₁ = t ≫ f₂


lemma RightFraction.exists_leftFraction [W.HasLeftCalculusOfFractions] {X Y : C}
    (φ : W.RightFraction X Y) : ∃ (ψ : W.LeftFraction X Y), φ.f ≫ ψ.s = φ.s ≫ ψ.f :=
  HasLeftCalculusOfFractions.exists_leftFraction φ


/-- A choice of a left fraction deduced from a right fraction for a morphism property `W`
when `W` has left calculus of fractions. -/
noncomputable def RightFraction.leftFraction [W.HasLeftCalculusOfFractions] {X Y : C}
    (φ : W.RightFraction X Y) : W.LeftFraction X Y :=
  φ.exists_leftFraction.choose


@[reassoc]
lemma RightFraction.leftFraction_fac [W.HasLeftCalculusOfFractions] {X Y : C}
    (φ : W.RightFraction X Y) : φ.f ≫ φ.leftFraction.s = φ.s ≫ φ.leftFraction.f :=
  φ.exists_leftFraction.choose_spec


lemma LeftFraction.exists_rightFraction [W.HasRightCalculusOfFractions] {X Y : C}
    (φ : W.LeftFraction X Y) : ∃ (ψ : W.RightFraction X Y), ψ.s ≫ φ.f = ψ.f ≫ φ.s :=
  HasRightCalculusOfFractions.exists_rightFraction φ


/-- A choice of a right fraction deduced from a left fraction for a morphism property `W`
when `W` has right calculus of fractions. -/
noncomputable def LeftFraction.rightFraction [W.HasRightCalculusOfFractions] {X Y : C}
    (φ : W.LeftFraction X Y) : W.RightFraction X Y :=
  φ.exists_rightFraction.choose


@[reassoc]
lemma LeftFraction.rightFraction_fac [W.HasRightCalculusOfFractions] {X Y : C}
    (φ : W.LeftFraction X Y) : φ.rightFraction.s ≫ φ.f = φ.rightFraction.f ≫ φ.s :=
  φ.exists_rightFraction.choose_spec


/-- The equivalence relation on left fractions for a morphism property `W`. -/
def LeftFractionRel {X Y : C} (z₁ z₂ : W.LeftFraction X Y) : Prop :=
  ∃ (Z : C) (t₁ : z₁.Y' ⟶ Z) (t₂ : z₂.Y' ⟶ Z) (_ : z₁.s ≫ t₁ = z₂.s ≫ t₂)
    (_ : z₁.f ≫ t₁ = z₂.f ≫ t₂), W (z₁.s ≫ t₁)


lemma refl {X Y : C} (z : W.LeftFraction X Y) : LeftFractionRel z z :=
                                /-
                                  C : Type u_1
                                  inst✝ : CategoryTheory.Category.{u_3, u_1} C
                                  W : CategoryTheory.MorphismProperty C
                                  X Y : C
                                  z : W.LeftFraction X Y
                                  ⊢ W (CategoryTheory.CategoryStruct.comp z.s (CategoryTheory.CategoryStruct.id  …
                                -/
  ⟨z.Y', 𝟙 _, 𝟙 _, rfl, rfl, by simpa only [Category.comp_id] using z.hs⟩
                                /-
                                  🎉 no goals
                                -/


lemma symm {X Y : C} {z₁ z₂ : W.LeftFraction X Y} (h : LeftFractionRel z₁ z₂) :
    LeftFractionRel z₂ z₁ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction X Y
    h : CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₂ z₁
  -/
  obtain ⟨Z, t₁, t₂, hst, hft, ht⟩ := h
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction X Y
    Z : C
    t₁ : Quiver.Hom z₁.Y' Z
    t₂ : Quiver.Hom z₂.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₂ z₁
  -/
  exact ⟨Z, t₂, t₁, hst.symm, hft.symm, by simpa only [← hst] using ht⟩
  /-
    🎉 no goals
  -/


lemma trans {X Y : C} {z₁ z₂ z₃ : W.LeftFraction X Y}
    [HasLeftCalculusOfFractions W]
    (h₁₂ : LeftFractionRel z₁ z₂) (h₂₃ : LeftFractionRel z₂ z₃) :
    LeftFractionRel z₁ z₃ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    h₁₂ : CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₂
    h₂₃ : CategoryTheory.MorphismProperty.LeftFractionRel z₂ z₃
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  obtain ⟨Z₄, t₁, t₂, hst, hft, ht⟩ := h₁₂
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    h₂₃ : CategoryTheory.MorphismProperty.LeftFractionRel z₂ z₃
    Z₄ : C
    t₁ : Quiver.Hom z₁.Y' Z₄
    t₂ : Quiver.Hom z₂.Y' Z₄
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  obtain ⟨Z₅, u₂, u₃, hsu, hfu, hu⟩ := h₂₃
  obtain ⟨⟨v₄, v₅, hv₅⟩, fac⟩ := HasLeftCalculusOfFractions.exists_leftFraction
    (RightFraction.mk (z₁.s ≫ t₁) ht (z₃.s ≫ u₃))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    Z₄ : C
    t₁ : Quiver.Hom z₁.Y' Z₄
    t₂ : Quiver.Hom z₂.Y' Z₄
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    Z₅ : C
    u₂ : Quiver.Hom z₂.Y' Z₅
    u₃ : Quiver.Hom z₃.Y' Z₅
    hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
    hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
    hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
    Y'✝ : C
    v₄ : Quiver.Hom Z₄ Y'✝
    v₅ : Quiver.Hom Z₅ Y'✝
    hv₅ : W v₅
    fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  simp only [Category.assoc] at fac
  have eq : z₂.s ≫ u₂ ≫ v₅  = z₂.s ≫ t₂ ≫ v₄ := by
    simpa only [← reassoc_of% hsu, reassoc_of% hst] using fac
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    Z₄ : C
    t₁ : Quiver.Hom z₁.Y' Z₄
    t₂ : Quiver.Hom z₂.Y' Z₄
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    Z₅ : C
    u₂ : Quiver.Hom z₂.Y' Z₅
    u₃ : Quiver.Hom z₃.Y' Z₅
    hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
    hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
    hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
    Y'✝ : C
    v₄ : Quiver.Hom Z₄ Y'✝
    v₅ : Quiver.Hom Z₅ Y'✝
    hv₅ : W v₅
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  obtain ⟨Z₇, w, hw, fac'⟩ := HasLeftCalculusOfFractions.ext _ _ _ z₂.hs eq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    Z₄ : C
    t₁ : Quiver.Hom z₁.Y' Z₄
    t₂ : Quiver.Hom z₂.Y' Z₄
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    Z₅ : C
    u₂ : Quiver.Hom z₂.Y' Z₅
    u₃ : Quiver.Hom z₃.Y' Z₅
    hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
    hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
    hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
    Y'✝ : C
    v₄ : Quiver.Hom Z₄ Y'✝
    v₅ : Quiver.Hom Z₅ Y'✝
    hv₅ : W v₅
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
    Z₇ : C
    w : Quiver.Hom Y'✝ Z₇
    hw : W w
    fac' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  simp only [Category.assoc] at fac'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ z₃ : W.LeftFraction X Y
    inst✝ : W.HasLeftCalculusOfFractions
    Z₄ : C
    t₁ : Quiver.Hom z₁.Y' Z₄
    t₂ : Quiver.Hom z₂.Y' Z₄
    hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
    Z₅ : C
    u₂ : Quiver.Hom z₂.Y' Z₅
    u₃ : Quiver.Hom z₃.Y' Z₅
    hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
    hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
    hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
    Y'✝ : C
    v₄ : Quiver.Hom Z₄ Y'✝
    v₅ : Quiver.Hom Z₅ Y'✝
    hv₅ : W v₅
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
    Z₇ : C
    w : Quiver.Hom Y'✝ Z₇
    hw : W w
    fac' : Eq (CategoryTheory.CategoryStruct.comp u₂ (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₃
  -/
  refine ⟨Z₇, t₁ ≫ v₄ ≫ w, u₃ ≫ v₅ ≫ w, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      z₁ z₂ z₃ : W.LeftFraction X Y
      inst✝ : W.HasLeftCalculusOfFractions
      Z₄ : C
      t₁ : Quiver.Hom z₁.Y' Z₄
      t₂ : Quiver.Hom z₂.Y' Z₄
      hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
      Z₅ : C
      u₂ : Quiver.Hom z₂.Y' Z₅
      u₃ : Quiver.Hom z₃.Y' Z₅
      hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
      hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
      hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
      Y'✝ : C
      v₄ : Quiver.Hom Z₄ Y'✝
      v₅ : Quiver.Hom Z₅ Y'✝
      hv₅ : W v₅
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
      Z₇ : C
      w : Quiver.Hom Y'✝ Z₇
      hw : W w
      fac' : Eq (CategoryTheory.CategoryStruct.comp u₂ (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruct.c …
    -/
  · rw [reassoc_of% fac]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      z₁ z₂ z₃ : W.LeftFraction X Y
      inst✝ : W.HasLeftCalculusOfFractions
      Z₄ : C
      t₁ : Quiver.Hom z₁.Y' Z₄
      t₂ : Quiver.Hom z₂.Y' Z₄
      hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
      Z₅ : C
      u₂ : Quiver.Hom z₂.Y' Z₅
      u₃ : Quiver.Hom z₃.Y' Z₅
      hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
      hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
      hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
      Y'✝ : C
      v₄ : Quiver.Hom Z₄ Y'✝
      v₅ : Quiver.Hom Z₅ Y'✝
      hv₅ : W v₅
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
      Z₇ : C
      w : Quiver.Hom Y'✝ Z₇
      hw : W w
      fac' : Eq (CategoryTheory.CategoryStruct.comp u₂ (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStruct.c …
    -/
  · rw [reassoc_of% hft, ← fac', reassoc_of% hfu]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      z₁ z₂ z₃ : W.LeftFraction X Y
      inst✝ : W.HasLeftCalculusOfFractions
      Z₄ : C
      t₁ : Quiver.Hom z₁.Y' Z₄
      t₂ : Quiver.Hom z₂.Y' Z₄
      hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
      Z₅ : C
      u₂ : Quiver.Hom z₂.Y' Z₅
      u₃ : Quiver.Hom z₃.Y' Z₅
      hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
      hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
      hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
      Y'✝ : C
      v₄ : Quiver.Hom Z₄ Y'✝
      v₅ : Quiver.Hom Z₅ Y'✝
      hv₅ : W v₅
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
      Z₇ : C
      w : Quiver.Hom Y'✝ Z₇
      hw : W w
      fac' : Eq (CategoryTheory.CategoryStruct.comp u₂ (CategoryTheory.CategoryStruc …
      ⊢ W (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruct.co …
    -/
  · rw [← reassoc_of% fac, ← reassoc_of% hsu, ← Category.assoc]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intr …
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      X Y : C
      z₁ z₂ z₃ : W.LeftFraction X Y
      inst✝ : W.HasLeftCalculusOfFractions
      Z₄ : C
      t₁ : Quiver.Hom z₁.Y' Z₄
      t₂ : Quiver.Hom z₂.Y' Z₄
      hst : Eq (CategoryTheory.CategoryStruct.comp z₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp z₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp z₁.s t₁)
      Z₅ : C
      u₂ : Quiver.Hom z₂.Y' Z₅
      u₃ : Quiver.Hom z₃.Y' Z₅
      hsu : Eq (CategoryTheory.CategoryStruct.comp z₂.s u₂) (CategoryTheory.Category …
      hfu : Eq (CategoryTheory.CategoryStruct.comp z₂.f u₂) (CategoryTheory.Category …
      hu : W (CategoryTheory.CategoryStruct.comp z₂.s u₂)
      Y'✝ : C
      v₄ : Quiver.Hom Z₄ Y'✝
      v₅ : Quiver.Hom Z₅ Y'✝
      hv₅ : W v₅
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃.s (CategoryTheory.CategoryStru …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruc …
      Z₇ : C
      w : Quiver.Hom Y'✝ Z₇
      hw : W w
      fac' : Eq (CategoryTheory.CategoryStruct.comp u₂ (CategoryTheory.CategoryStruc …
      ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp z₂ …
    -/
    exact W.comp_mem _ _ hu (W.comp_mem _ _ hv₅ hw)
    /-
      🎉 no goals
    -/


lemma equivalenceLeftFractionRel [W.HasLeftCalculusOfFractions] (X Y : C) :
    @_root_.Equivalence (W.LeftFraction X Y) LeftFractionRel where
  refl := LeftFractionRel.refl
  symm := LeftFractionRel.symm
  trans := LeftFractionRel.trans


/-- Auxiliary definition for the composition of left fractions. -/
@[simp]
def comp₀ [W.HasLeftCalculusOfFractions] {X Y Z : C}
    (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z) (z₃ : W.LeftFraction z₁.Y' z₂.Y') :
    W.LeftFraction X Z :=
  mk (z₁.f ≫ z₃.f) (z₂.s ≫ z₃.s) (W.comp_mem _ _ z₂.hs z₃.hs)


/-- The equivalence class of `z₁.comp₀ z₂ z₃` does not depend on the choice of `z₃` provided
they satisfy the compatibility `z₂.f ≫ z₃.s = z₁.s ≫ z₃.f`. -/
lemma comp₀_rel [W.HasLeftCalculusOfFractions]
    {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z)
    (z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y') (h₃ : z₂.f ≫ z₃.s = z₁.s ≫ z₃.f)
    (h₃' : z₂.f ≫ z₃'.s = z₁.s ≫ z₃'.f) :
    LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z₂ z₃') := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z …
  -/
  obtain ⟨z₄, fac⟩ := exists_leftFraction (RightFraction.mk z₃.s z₃.hs z₃'.s)
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
    z₄ : W.LeftFraction z₃.Y' z₃'.Y'
    fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z …
  -/
  dsimp at fac
  have eq : z₁.s ≫ z₃.f ≫ z₄.f = z₁.s ≫ z₃'.f ≫ z₄.s := by
    rw [← reassoc_of% h₃, ← reassoc_of% h₃', fac]
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
    z₄ : W.LeftFraction z₃.Y' z₃'.Y'
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z …
  -/
  obtain ⟨Y, t, ht, fac'⟩ := HasLeftCalculusOfFractions.ext _ _ _ z₁.hs eq
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y✝ Z : C
    z₁ : W.LeftFraction X Y✝
    z₂ : W.LeftFraction Y✝ Z
    z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
    z₄ : W.LeftFraction z₃.Y' z₃'.Y'
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
    Y : C
    t : Quiver.Hom z₄.Y' Y
    ht : W t
    fac' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z …
  -/
  simp only [assoc] at fac'
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y✝ Z : C
    z₁ : W.LeftFraction X Y✝
    z₂ : W.LeftFraction Y✝ Z
    z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
    z₄ : W.LeftFraction z₃.Y' z₃'.Y'
    fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
    eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
    Y : C
    t : Quiver.Hom z₄.Y' Y
    ht : W t
    fac' : Eq (CategoryTheory.CategoryStruct.comp z₃.f (CategoryTheory.CategoryStr …
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (z₁.comp₀ z₂ z₃) (z₁.comp₀ z …
  -/
  refine ⟨Y, z₄.f ≫ t, z₄.s ≫ t, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y✝ Z : C
      z₁ : W.LeftFraction X Y✝
      z₂ : W.LeftFraction Y✝ Z
      z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
      h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
      h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
      z₄ : W.LeftFraction z₃.Y' z₃'.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
      Y : C
      t : Quiver.Hom z₄.Y' Y
      ht : W t
      fac' : Eq (CategoryTheory.CategoryStruct.comp z₃.f (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (z₁.comp₀ z₂ z₃).s (CategoryTheory.Ca …
    -/
  · simp only [comp₀, assoc, reassoc_of% fac]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y✝ Z : C
      z₁ : W.LeftFraction X Y✝
      z₂ : W.LeftFraction Y✝ Z
      z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
      h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
      h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
      z₄ : W.LeftFraction z₃.Y' z₃'.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
      Y : C
      t : Quiver.Hom z₄.Y' Y
      ht : W t
      fac' : Eq (CategoryTheory.CategoryStruct.comp z₃.f (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (z₁.comp₀ z₂ z₃).f (CategoryTheory.Ca …
    -/
  · simp only [comp₀, assoc, fac']
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_3
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y✝ Z : C
      z₁ : W.LeftFraction X Y✝
      z₂ : W.LeftFraction Y✝ Z
      z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
      h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
      h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
      z₄ : W.LeftFraction z₃.Y' z₃'.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
      Y : C
      t : Quiver.Hom z₄.Y' Y
      ht : W t
      fac' : Eq (CategoryTheory.CategoryStruct.comp z₃.f (CategoryTheory.CategoryStr …
      ⊢ W (CategoryTheory.CategoryStruct.comp (z₁.comp₀ z₂ z₃).s (CategoryTheory.Cat …
    -/
  · simp only [comp₀, assoc, ← reassoc_of% fac]
    /-
      case intro.intro.intro.intro.refine_3
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y✝ Z : C
      z₁ : W.LeftFraction X Y✝
      z₂ : W.LeftFraction Y✝ Z
      z₃ z₃' : W.LeftFraction z₁.Y' z₂.Y'
      h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
      h₃' : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃'.s) (CategoryTheory.Categ …
      z₄ : W.LeftFraction z₃.Y' z₃'.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp z₃'.s z₄.s) (CategoryTheory.Categ …
      eq : Eq (CategoryTheory.CategoryStruct.comp z₁.s (CategoryTheory.CategoryStruc …
      Y : C
      t : Quiver.Hom z₄.Y' Y
      ht : W t
      fac' : Eq (CategoryTheory.CategoryStruct.comp z₃.f (CategoryTheory.CategoryStr …
      ⊢ W (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStruct.co …
    -/
    exact W.comp_mem _ _ z₂.hs (W.comp_mem _ _ z₃'.hs (W.comp_mem _ _ z₄.hs ht))
    /-
      🎉 no goals
    -/


/-- The morphisms in the constructed localized category for a morphism property `W`
that has left calculus of fractions are equivalence classes of left fractions. -/
def Localization.Hom (X Y : C) :=
  Quot (LeftFractionRel : W.LeftFraction X Y → W.LeftFraction X Y → Prop)


/-- The morphism in the constructed localized category that is induced by a left fraction. -/
def Localization.Hom.mk {X Y : C} (z : W.LeftFraction X Y) : Localization.Hom W X Y :=
  Quot.mk _ z


lemma Localization.Hom.mk_surjective {X Y : C} (f : Localization.Hom W X Y) :
    ∃ (z : W.LeftFraction X Y), f = mk z := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
    ⊢ Exists fun z => Eq f (CategoryTheory.MorphismProperty.LeftFraction.Localizat …
  -/
  obtain ⟨z⟩ := f
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
    z : W.LeftFraction X Y
    ⊢ Exists fun z_1 => Eq (Quot.mk CategoryTheory.MorphismProperty.LeftFractionRe …
  -/
  exact ⟨z, rfl⟩
  /-
    🎉 no goals
  -/


/-- Auxiliary definition towards the definition of the composition of morphisms
in the constructed localized category for a morphism property that has
left calculus of fractions. -/
noncomputable def comp
    {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z) :
    Localization.Hom W X Z :=
  Localization.Hom.mk (z₁.comp₀ z₂ (RightFraction.mk z₁.s z₁.hs z₂.f).leftFraction)


lemma comp_eq {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z)
    (z₃ : W.LeftFraction z₁.Y' z₂.Y') (h₃ : z₂.f ≫ z₃.s = z₁.s ≫ z₃.f) :
    z₁.comp z₂ = Localization.Hom.mk (z₁.comp₀ z₂ z₃) :=
  Quot.sound (LeftFraction.comp₀_rel _ _ _ _
    (RightFraction.leftFraction_fac (RightFraction.mk z₁.s z₁.hs z₂.f)) h₃)


/-- Composition of morphisms in the constructed localized category
for a morphism property that has left calculus of fractions. -/
noncomputable def Hom.comp {X Y Z : C} (z₁ : Hom W X Y) (z₂ : Hom W Y Z) : Hom W X Z := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
    z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
    ⊢ CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Z
  -/
  refine Quot.lift₂ (fun a b => a.comp b) ?_ ?_ z₁ z₂
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      ⊢ ∀ (a : W.LeftFraction X Y) (b₁ b₂ : W.LeftFraction Y Z), CategoryTheory.Morp …
    -/
  · rintro a b₁ b₂ ⟨U, t₁, t₂, hst, hft, ht⟩
    /-
      case refine_1.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    obtain ⟨z₁, fac₁⟩ := exists_leftFraction (RightFraction.mk a.s a.hs b₁.f)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    obtain ⟨z₂, fac₂⟩ := exists_leftFraction (RightFraction.mk a.s a.hs b₂.f)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    obtain ⟨w₁, fac₁'⟩ := exists_leftFraction (RightFraction.mk z₁.s z₁.hs t₁)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    obtain ⟨w₂, fac₂'⟩ := exists_leftFraction (RightFraction.mk z₂.s z₂.hs t₂)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    obtain ⟨u, fac₃⟩ := exists_leftFraction (RightFraction.mk w₁.s w₁.hs w₂.s)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      ⊢ Eq ((fun a b => a.comp b) a b₁) ((fun a b => a.comp b) a b₂)
    -/
    dsimp at fac₁ fac₂ fac₁' fac₂' fac₃ ⊢
    have eq : a.s ≫ z₁.f ≫ w₁.f ≫ u.f = a.s ≫ z₂.f ≫ w₂.f ≫ u.s := by
      rw [← reassoc_of% fac₁, ← reassoc_of% fac₂, ← reassoc_of% fac₁', ← reassoc_of% fac₂',
        reassoc_of% hft, fac₃]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
      eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
      ⊢ Eq (a.comp b₁) (a.comp b₂)
    -/
    obtain ⟨Z, p, hp, fac₄⟩ := HasLeftCalculusOfFractions.ext _ _ _ a.hs eq
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z✝ : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z✝
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
      eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
      Z : C
      p : Quiver.Hom u.Y' Z
      hp : W p
      fac₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ Eq (a.comp b₁) (a.comp b₂)
    -/
    simp only [assoc] at fac₄
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z✝ : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z✝
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
      eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
      Z : C
      p : Quiver.Hom u.Y' Z
      hp : W p
      fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
      ⊢ Eq (a.comp b₁) (a.comp b₂)
    -/
    rw [comp_eq _ _ z₁ fac₁, comp_eq _ _ z₂ fac₂]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z✝ : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z✝
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
      eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
      Z : C
      p : Quiver.Hom u.Y' Z
      hp : W p
      fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (a.comp …
    -/
    apply Quot.sound
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z✝ : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
      a : W.LeftFraction X Y
      b₁ b₂ : W.LeftFraction Y Z✝
      U : C
      t₁ : Quiver.Hom b₁.Y' U
      t₂ : Quiver.Hom b₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
      z₁ : W.LeftFraction a.Y' b₁.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
      z₂ : W.LeftFraction a.Y' b₂.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
      w₁ : W.LeftFraction z₁.Y' U
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
      w₂ : W.LeftFraction z₂.Y' U
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
      u : W.LeftFraction w₁.Y' w₂.Y'
      fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
      eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
      Z : C
      p : Quiver.Hom u.Y' Z
      hp : W p
      fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a.comp₀ b₁ z₁) (a.comp₀ b₂  …
    -/
    refine ⟨Z, w₁.f ≫ u.f ≫ p, w₂.f ≫ u.s ≫ p, ?_, ?_, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (a.comp₀ b₁ z₁).s (CategoryTheory.Cat …
      -/
    · dsimp
      simp only [assoc, ← reassoc_of% fac₁', ← reassoc_of% fac₂',
        reassoc_of% hst, reassoc_of% fac₃]
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (a.comp₀ b₁ z₁).f (CategoryTheory.Cat …
      -/
    · dsimp
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
      -/
      simp only [assoc, fac₄]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ W (CategoryTheory.CategoryStruct.comp (a.comp₀ b₁ z₁).s (CategoryTheory.Cate …
      -/
    · dsimp
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b₁ …
      -/
      simp only [assoc]
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ W (CategoryTheory.CategoryStruct.comp b₁.s (CategoryTheory.CategoryStruct.co …
      -/
      rw [← reassoc_of% fac₁', ← reassoc_of% fac₃, ← assoc]
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a : W.LeftFraction X Y
        b₁ b₂ : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom b₁.Y' U
        t₂ : Quiver.Hom b₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp b₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp b₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp b₁.s t₁)
        z₁ : W.LeftFraction a.Y' b₁.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b₁.f z₁.s) (CategoryTheory.Categ …
        z₂ : W.LeftFraction a.Y' b₂.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b₂.f z₂.s) (CategoryTheory.Categ …
        w₁ : W.LeftFraction z₁.Y' U
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp t₁ w₁.s) (CategoryTheory.Catego …
        w₂ : W.LeftFraction z₂.Y' U
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp t₂ w₂.s) (CategoryTheory.Catego …
        u : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp w₂.s u.s) (CategoryTheory.Catego …
        eq : Eq (CategoryTheory.CategoryStruct.comp a.s (CategoryTheory.CategoryStruct …
        Z : C
        p : Quiver.Hom u.Y' Z
        hp : W p
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b₁ …
      -/
      exact W.comp_mem _ _ ht (W.comp_mem _ _ w₂.hs (W.comp_mem _ _ u.hs hp))
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      ⊢ ∀ (a₁ a₂ : W.LeftFraction X Y) (b : W.LeftFraction Y Z), CategoryTheory.Morp …
    -/
  · rintro a₁ a₂ b ⟨U, t₁, t₂, hst, hft, ht⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      ⊢ Eq ((fun a b => a.comp b) a₁ b) ((fun a b => a.comp b) a₂ b)
    -/
    obtain ⟨z₁, fac₁⟩ := exists_leftFraction (RightFraction.mk a₁.s a₁.hs b.f)
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      ⊢ Eq ((fun a b => a.comp b) a₁ b) ((fun a b => a.comp b) a₂ b)
    -/
    obtain ⟨z₂, fac₂⟩ := exists_leftFraction (RightFraction.mk a₂.s a₂.hs b.f)
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      ⊢ Eq ((fun a b => a.comp b) a₁ b) ((fun a b => a.comp b) a₂ b)
    -/
    obtain ⟨w₁, fac₁'⟩ := exists_leftFraction (RightFraction.mk (a₁.s ≫ t₁) ht (b.f ≫ z₁.s))
    obtain ⟨w₂, fac₂'⟩ := exists_leftFraction (RightFraction.mk (a₂.s ≫ t₂)
      (show W _ by rw [← hst]; exact ht) (b.f ≫ z₂.s))
    let p₁ : W.LeftFraction X Z := LeftFraction.mk (a₁.f ≫ t₁ ≫ w₁.f) (b.s ≫ z₁.s ≫ w₁.s)
      (W.comp_mem _ _ b.hs (W.comp_mem _ _ z₁.hs w₁.hs))
    let p₂ : W.LeftFraction X Z := LeftFraction.mk (a₂.f ≫ t₂ ≫ w₂.f) (b.s ≫ z₂.s ≫ w₂.s)
      (W.comp_mem _ _ b.hs (W.comp_mem _ _ z₂.hs w₂.hs))
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
      w₁ : W.LeftFraction U z₁.Y'
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      w₂ : W.LeftFraction U z₂.Y'
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      ⊢ Eq ((fun a b => a.comp b) a₁ b) ((fun a b => a.comp b) a₂ b)
    -/
    dsimp at fac₁ fac₂ fac₁' fac₂' ⊢
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
      w₁ : W.LeftFraction U z₁.Y'
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
      w₂ : W.LeftFraction U z₂.Y'
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
      p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      ⊢ Eq (a₁.comp b) (a₂.comp b)
    -/
    simp only [assoc] at fac₁' fac₂'
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
      w₁ : W.LeftFraction U z₁.Y'
      w₂ : W.LeftFraction U z₂.Y'
      p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      ⊢ Eq (a₁.comp b) (a₂.comp b)
    -/
    rw [comp_eq _ _ z₁ fac₁, comp_eq _ _ z₂ fac₂]
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
      w₁ : W.LeftFraction U z₁.Y'
      w₂ : W.LeftFraction U z₂.Y'
      p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (a₁.com …
    -/
    apply Quot.sound
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
      a₁ a₂ : W.LeftFraction X Y
      b : W.LeftFraction Y Z
      U : C
      t₁ : Quiver.Hom a₁.Y' U
      t₂ : Quiver.Hom a₂.Y' U
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      z₁ : W.LeftFraction a₁.Y' b.Y'
      fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
      z₂ : W.LeftFraction a₂.Y' b.Y'
      fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
      w₁ : W.LeftFraction U z₁.Y'
      w₂ : W.LeftFraction U z₂.Y'
      p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
      fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a₁.comp₀ b z₁) (a₂.comp₀ b  …
    -/
    refine LeftFractionRel.trans ?_ ((?_ : LeftFractionRel p₁ p₂).trans ?_)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a₁.comp₀ b z₁) p₁
      -/
    · have eq : a₁.s ≫ z₁.f ≫ w₁.s = a₁.s ≫ t₁ ≫ w₁.f := by rw [← fac₁', reassoc_of% fac₁]
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a₁.comp₀ b z₁) p₁
      -/
      obtain ⟨Z, u, hu, fac₃⟩ := HasLeftCalculusOfFractions.ext _ _ _ a₁.hs eq
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom w₁.Y' Z
        hu : W u
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a₁.comp₀ b z₁) p₁
      -/
      simp only [assoc] at fac₃
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom w₁.Y' Z
        hu : W u
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (a₁.comp₀ b z₁) p₁
      -/
      refine ⟨Z, w₁.s ≫ u, u, ?_, ?_, ?_⟩
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (a₁.comp₀ b z₁).s (CategoryTheory.Cat …
        -/
      · dsimp [p₁]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b …
        -/
        simp only [assoc]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (a₁.comp₀ b z₁).f (CategoryTheory.Cat …
        -/
      · dsimp [p₁]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
        -/
        simp only [assoc, fac₃]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp (a₁.comp₀ b z₁).s (CategoryTheory.Cate …
        -/
      · dsimp
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b. …
        -/
        simp only [assoc]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₁.Y' Z
          hu : W u
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₁.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp b.s (CategoryTheory.CategoryStruct.com …
        -/
        exact W.comp_mem _ _ b.hs (W.comp_mem _ _ z₁.hs (W.comp_mem _ _ w₁.hs hu))
        /-
          🎉 no goals
        -/
    · obtain ⟨q, fac₃⟩ := exists_leftFraction (RightFraction.mk (z₁.s ≫ w₁.s)
        (W.comp_mem _ _ z₁.hs w₁.hs) (z₂.s ≫ w₂.s))
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        q : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₁ p₂
      -/
      dsimp at fac₃
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        q : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₁ p₂
      -/
      simp only [assoc] at fac₃
      have eq : a₁.s ≫ t₁ ≫ w₁.f ≫ q.f = a₁.s ≫ t₁ ≫ w₂.f ≫ q.s := by
        rw [← reassoc_of% fac₁', ← fac₃, reassoc_of% hst, reassoc_of% fac₂']
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        q : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₁ p₂
      -/
      obtain ⟨Z, u, hu, fac₄⟩ := HasLeftCalculusOfFractions.ext _ _ _ a₁.hs eq
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        q : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom q.Y' Z
        hu : W u
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₁ p₂
      -/
      simp only [assoc] at fac₄
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        q : W.LeftFraction w₁.Y' w₂.Y'
        fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom q.Y' Z
        hu : W u
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruc …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₁ p₂
      -/
      refine ⟨Z, q.f ≫ u, q.s ≫ u, ?_, ?_, ?_⟩
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          q : W.LeftFraction w₁.Y' w₂.Y'
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom q.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruc …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp p₁.s (CategoryTheory.CategoryStruct.c …
        -/
      · simp only [p₁, p₂, assoc, reassoc_of% fac₃]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          q : W.LeftFraction w₁.Y' w₂.Y'
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom q.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruc …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp p₁.f (CategoryTheory.CategoryStruct.c …
        -/
      · rw [assoc, assoc, assoc, assoc, fac₄, reassoc_of% hft]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_2 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          q : W.LeftFraction w₁.Y' w₂.Y'
          fac₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.s (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₁.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom q.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruc …
          ⊢ W (CategoryTheory.CategoryStruct.comp p₁.s (CategoryTheory.CategoryStruct.co …
        -/
      · simp only [p₁, p₂, assoc, ← reassoc_of% fac₃]
        exact W.comp_mem _ _ b.hs (W.comp_mem _ _ z₂.hs
          (W.comp_mem _ _ w₂.hs (W.comp_mem _ _ q.hs hu)))
    · have eq : a₂.s ≫ z₂.f ≫ w₂.s = a₂.s ≫ t₂ ≫ w₂.f := by
        rw [← fac₂', reassoc_of% fac₂]
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        p₂ : W.LeftFraction X Z := CategoryTheory.MorphismProperty.LeftFraction.mk (Ca …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₂ (a₂.comp₀ b z₂)
      -/
      obtain ⟨Z, u, hu, fac₄⟩ := HasLeftCalculusOfFractions.ext _ _ _ a₂.hs eq
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom w₂.Y' Z
        hu : W u
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₂ (a₂.comp₀ b z₂)
      -/
      simp only [assoc] at fac₄
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
        C : Type u_1
        D : Type u_2
        inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
        W : CategoryTheory.MorphismProperty C
        inst✝ : W.HasLeftCalculusOfFractions
        X Y Z✝ : C
        z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
        z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
        a₁ a₂ : W.LeftFraction X Y
        b : W.LeftFraction Y Z✝
        U : C
        t₁ : Quiver.Hom a₁.Y' U
        t₂ : Quiver.Hom a₂.Y' U
        hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
        hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
        ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
        z₁ : W.LeftFraction a₁.Y' b.Y'
        fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
        z₂ : W.LeftFraction a₂.Y' b.Y'
        fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
        w₁ : W.LeftFraction U z₁.Y'
        w₂ : W.LeftFraction U z₂.Y'
        p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
        fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
        eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
        Z : C
        u : Quiver.Hom w₂.Y' Z
        hu : W u
        fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
        ⊢ CategoryTheory.MorphismProperty.LeftFractionRel p₂ (a₂.comp₀ b z₂)
      -/
      refine ⟨Z, u, w₂.s ≫ u, ?_, ?_, ?_⟩
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp p₂.s u) (CategoryTheory.CategoryStruc …
        -/
      · dsimp [p₁, p₂]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b …
        -/
        simp only [assoc]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp p₂.f u) (CategoryTheory.CategoryStruc …
        -/
      · dsimp [p₁, p₂]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
        -/
        simp only [assoc, fac₄]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp p₂.s u)
        -/
      · dsimp [p₁, p₂]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp b. …
        -/
        simp only [assoc]
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_3 …
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.34330, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.34334, u_2} D
          W : CategoryTheory.MorphismProperty C
          inst✝ : W.HasLeftCalculusOfFractions
          X Y Z✝ : C
          z₁✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
          z₂✝ : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W Y Z✝
          a₁ a₂ : W.LeftFraction X Y
          b : W.LeftFraction Y Z✝
          U : C
          t₁ : Quiver.Hom a₁.Y' U
          t₂ : Quiver.Hom a₂.Y' U
          hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
          hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
          ht : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
          z₁ : W.LeftFraction a₁.Y' b.Y'
          fac₁ : Eq (CategoryTheory.CategoryStruct.comp b.f z₁.s) (CategoryTheory.Catego …
          z₂ : W.LeftFraction a₂.Y' b.Y'
          fac₂ : Eq (CategoryTheory.CategoryStruct.comp b.f z₂.s) (CategoryTheory.Catego …
          w₁ : W.LeftFraction U z₁.Y'
          w₂ : W.LeftFraction U z₂.Y'
          p₁ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          p₂ : W.LeftFraction X Z✝ := CategoryTheory.MorphismProperty.LeftFraction.mk (C …
          fac₁' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          fac₂' : Eq (CategoryTheory.CategoryStruct.comp b.f (CategoryTheory.CategoryStr …
          eq : Eq (CategoryTheory.CategoryStruct.comp a₂.s (CategoryTheory.CategoryStruc …
          Z : C
          u : Quiver.Hom w₂.Y' Z
          hu : W u
          fac₄ : Eq (CategoryTheory.CategoryStruct.comp z₂.f (CategoryTheory.CategoryStr …
          ⊢ W (CategoryTheory.CategoryStruct.comp b.s (CategoryTheory.CategoryStruct.com …
        -/
        exact W.comp_mem _ _ b.hs (W.comp_mem _ _ z₂.hs (W.comp_mem _ _ w₂.hs hu))
        /-
          🎉 no goals
        -/


lemma Hom.comp_eq {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z) :
    Hom.comp (mk z₁) (mk z₂) = z₁.comp z₂ := rfl


/-- The constructed localized category for a morphism property
that has left calculus of fractions. -/
@[nolint unusedArguments]
def Localization (_ : MorphismProperty C) := C


noncomputable instance : Category (Localization W) where
  Hom X Y := Localization.Hom W X Y
  id _ := Localization.Hom.mk (ofHom W (𝟙 _))
  comp f g := f.comp g
  comp_id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      ⊢ ∀ {X Y : CategoryTheory.MorphismProperty.LeftFraction.Localization W} (f : Q …
    -/
    rintro (X Y : C) f
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
    -/
    obtain ⟨z, rfl⟩ := Hom.mk_surjective f
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Left …
    -/
    change (Hom.mk z).comp (Hom.mk (ofHom W (𝟙 Y))) = Hom.mk z
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk z).com …
    -/
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      ⊢ ∀ {X Y : CategoryTheory.MorphismProperty.LeftFraction.Localization W} (f : Q …
    -/
    rw [Hom.comp_eq, comp_eq z (ofHom W (𝟙 Y)) (ofInv z.s z.hs) (by simp)]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (z.comp …
    -/
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    dsimp [comp₀]
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (Categ …
    -/
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (Catego …
    -/
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk ((Categ …
    -/
    simp only [comp_id, id_comp]
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      z : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (Catego …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  id_comp := by
    rintro (X Y : C) f
    obtain ⟨z, rfl⟩ := Hom.mk_surjective f
    change (Hom.mk (ofHom W (𝟙 X))).comp (Hom.mk z) = Hom.mk z
    rw [Hom.comp_eq, comp_eq (ofHom W (𝟙 X)) z (ofHom W z.f) (by simp)]
    dsimp
    simp only [comp₀, id_comp, comp_id]
  assoc := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      ⊢ ∀ {W_1 X Y Z : CategoryTheory.MorphismProperty.LeftFraction.Localization W}  …
    -/
    rintro (X₁ X₂ X₃ X₄ : C) f₁ f₂ f₃
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      f₃ : Quiver.Hom X₃ X₄
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    obtain ⟨z₁, rfl⟩ := Hom.mk_surjective f₁
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      f₂ : Quiver.Hom X₂ X₃
      f₃ : Quiver.Hom X₃ X₄
      z₁ : W.LeftFraction X₁ X₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    obtain ⟨z₂, rfl⟩ := Hom.mk_surjective f₂
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      f₃ : Quiver.Hom X₃ X₄
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    obtain ⟨z₃, rfl⟩ := Hom.mk_surjective f₃
    change ((Hom.mk z₁).comp (Hom.mk z₂)).comp (Hom.mk z₃) =
      (Hom.mk z₁).comp ((Hom.mk z₂).comp (Hom.mk z₃))
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      ⊢ Eq (((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk z₁).c …
    -/
    rw [Hom.comp_eq z₁ z₂, Hom.comp_eq z₂ z₃]
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      ⊢ Eq ((z₁.comp z₂).comp (CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    -/
    obtain ⟨z₁₂, fac₁₂⟩ := exists_leftFraction (RightFraction.mk z₁.s z₁.hs z₂.f)
    /-
      case intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      z₁₂ : W.LeftFraction z₁.Y' z₂.Y'
      fac₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      ⊢ Eq ((z₁.comp z₂).comp (CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    -/
    obtain ⟨z₂₃, fac₂₃⟩ := exists_leftFraction (RightFraction.mk z₂.s z₂.hs z₃.f)
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      z₁₂ : W.LeftFraction z₁.Y' z₂.Y'
      fac₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      z₂₃ : W.LeftFraction z₂.Y' z₃.Y'
      fac₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      ⊢ Eq ((z₁.comp z₂).comp (CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    -/
    obtain ⟨z', fac⟩ := exists_leftFraction (RightFraction.mk z₁₂.s z₁₂.hs z₂₃.f)
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      z₁₂ : W.LeftFraction z₁.Y' z₂.Y'
      fac₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      z₂₃ : W.LeftFraction z₂.Y' z₃.Y'
      fac₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismPropert …
      z' : W.LeftFraction z₁₂.Y' z₂₃.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
      ⊢ Eq ((z₁.comp z₂).comp (CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    -/
    dsimp at fac₁₂ fac₂₃ fac
    rw [comp_eq z₁ z₂ z₁₂ fac₁₂, comp_eq z₂ z₃ z₂₃ fac₂₃, comp₀, comp₀,
      Hom.comp_eq, Hom.comp_eq,
      comp_eq _ z₃ (mk z'.f (z₂₃.s ≫ z'.s) (W.comp_mem _ _ z₂₃.hs z'.hs))
        (by dsimp; rw [assoc, reassoc_of% fac₂₃, fac]),
      comp_eq z₁ _ (mk (z₁₂.f ≫ z'.f) z'.s z'.hs)
        (by dsimp; rw [assoc, ← reassoc_of% fac₁₂, fac])]
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.54931, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.54935, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X₁ X₂ X₃ X₄ : C
      z₁ : W.LeftFraction X₁ X₂
      z₂ : W.LeftFraction X₂ X₃
      z₃ : W.LeftFraction X₃ X₄
      z₁₂ : W.LeftFraction z₁.Y' z₂.Y'
      fac₁₂ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₁₂.s) (CategoryTheory.Cat …
      z₂₃ : W.LeftFraction z₂.Y' z₃.Y'
      fac₂₃ : Eq (CategoryTheory.CategoryStruct.comp z₃.f z₂₃.s) (CategoryTheory.Cat …
      z' : W.LeftFraction z₁₂.Y' z₂₃.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp z₂₃.f z'.s) (CategoryTheory.Categ …
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk ((Categ …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The localization functor to the constructed localized category for a morphism property
that has left calculus of fractions. -/
@[simps obj]
def Q : C ⥤ Localization W where
  obj X := X
  map f := Hom.mk (ofHom W f)
  map_id _ := rfl
  map_comp {X Y Z} f g := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.62270, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.62274, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => CategoryTheory.MorphismProper …
    -/
    change _ = Hom.comp _ _
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.62270, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.62274, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => CategoryTheory.MorphismProper …
    -/
    rw [Hom.comp_eq, comp_eq (ofHom W f) (ofHom W g) (ofHom W g) (by simp)]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.62270, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.62274, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => CategoryTheory.MorphismProper …
    -/
    simp only [ofHom, comp₀, comp_id]
    /-
      🎉 no goals
    -/


/-- The morphism on `Localization W` that is induced by a left fraction. -/
abbrev homMk {X Y : C} (f : W.LeftFraction X Y) : (Q W).obj X ⟶ (Q W).obj Y := Hom.mk f


lemma homMk_eq_hom_mk {X Y : C} (f : W.LeftFraction X Y) : homMk f = Hom.mk f := rfl


lemma Q_map {X Y : C} (f : X ⟶ Y) : (Q W).map f = homMk (ofHom W f) := rfl


lemma homMk_comp_homMk {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z)
    (z₃ : W.LeftFraction z₁.Y' z₂.Y') (h₃ : z₂.f ≫ z₃.s = z₁.s ≫ z₃.f) :
    homMk z₁ ≫ homMk z₂ = homMk (z₁.comp₀ z₂ z₃) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Left …
  -/
  change Hom.comp _ _ = _
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.comp (Cate …
  -/
  rw [Hom.comp_eq, comp_eq z₁ z₂ z₃ h₃]
  /-
    🎉 no goals
  -/


lemma homMk_eq_of_leftFractionRel {X Y : C} (z₁ z₂ : W.LeftFraction X Y)
    (h : LeftFractionRel z₁ z₂) :
    homMk z₁ = homMk z₂ :=
  Quot.sound h


lemma homMk_eq_iff_leftFractionRel {X Y : C} (z₁ z₂ : W.LeftFraction X Y) :
    homMk z₁ = homMk z₂ ↔ LeftFractionRel z₁ z₂ :=
  @Equivalence.quot_mk_eq_iff _ _ (equivalenceLeftFractionRel W X Y) _ _


/-- The morphism in `Localization W` that is the formal inverse of a morphism
which belongs to `W`. -/
def Qinv {X Y : C} (s : X ⟶ Y) (hs : W s) : (Q W).obj Y ⟶ (Q W).obj X := homMk (ofInv s hs)


lemma Q_map_comp_Qinv {X Y Y' : C} (f : X ⟶ Y') (s : Y ⟶ Y') (hs : W s) :
    (Q W).map f ≫ Qinv s hs = homMk (mk f s hs) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Y' : C
    f : Quiver.Hom X Y'
    s : Quiver.Hom Y Y'
    hs : W s
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Lef …
  -/
  dsimp only [Q_map, Qinv]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Y' : C
    f : Quiver.Hom X Y'
    s : Quiver.Hom Y Y'
    hs : W s
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Left …
  -/
  rw [homMk_comp_homMk (ofHom W f) (ofInv s hs) (ofHom W (𝟙 _)) (by simp)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Y' : C
    f : Quiver.Hom X Y'
    s : Quiver.Hom Y Y'
    hs : W s
    ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.homMk ((Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The isomorphism in `Localization W` that is induced by a morphism in `W`. -/
@[simps]
def Qiso {X Y : C} (s : X ⟶ Y) (hs : W s) : (Q W).obj X ≅ (Q W).obj Y where
  hom := (Q W).map s
  inv := Qinv s hs
  hom_inv_id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Lef …
    -/
    rw [Q_map_comp_Qinv]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.homMk (Categor …
    -/
    apply homMk_eq_of_leftFractionRel
    /-
      case h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel (CategoryTheory.MorphismProp …
    -/
    exact ⟨_, 𝟙 Y, s, by simp, by simp, by simpa using hs⟩
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Left …
    -/
    dsimp only [Qinv, Q_map]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Left …
    -/
    rw [homMk_comp_homMk (ofInv s hs) (ofHom W s) (ofHom W (𝟙 Y)) (by simp)]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.homMk ((Catego …
    -/
    apply homMk_eq_of_leftFractionRel
    /-
      case h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.70891, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.70895, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      s : Quiver.Hom X Y
      hs : W s
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel ((CategoryTheory.MorphismPro …
    -/
    exact ⟨_, 𝟙 Y, 𝟙 Y, by simp, by simp, by simpa using W.id_mem Y⟩
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma Qiso_hom_inv_id {X Y : C} (s : X ⟶ Y) (hs : W s) :
    (Q W).map s ≫ Qinv s hs = 𝟙 _ := (Qiso s hs).hom_inv_id


@[reassoc (attr := simp)]
lemma Qiso_inv_hom_id {X Y : C} (s : X ⟶ Y) (hs : W s) :
    Qinv s hs  ≫ (Q W).map s = 𝟙 _ := (Qiso s hs).inv_hom_id


instance {X Y : C} (s : X ⟶ Y) (hs : W s) : IsIso (Qinv s hs) :=
  (inferInstance : IsIso (Qiso s hs).inv)


/-- The image by a functor which inverts `W` of an equivalence class of left fractions. -/
noncomputable def Hom.map {X Y : C} (f : Hom W X Y) (F : C ⥤ E) (hF : W.IsInvertedBy F) :
    F.obj X ⟶ F.obj Y :=
  Quot.lift (fun f => f.map F hF) (by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.78597, u_1} C
      inst✝² : CategoryTheory.Category.{?u.78601, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.78643, u_3} E
      X Y : C
      f : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      ⊢ ∀ (a b : W.LeftFraction X Y), CategoryTheory.MorphismProperty.LeftFractionRe …
    -/
    intro a₁ a₂ ⟨Z, t₁, t₂, hst, hft, h⟩
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.78597, u_1} C
      inst✝² : CategoryTheory.Category.{?u.78601, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.78643, u_3} E
      X Y : C
      f : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      a₁ a₂ : W.LeftFraction X Y
      Z : C
      t₁ : Quiver.Hom a₁.Y' Z
      t₂ : Quiver.Hom a₂.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      h : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      ⊢ Eq ((fun f => f.map F hF) a₁) ((fun f => f.map F hF) a₂)
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.78597, u_1} C
      inst✝² : CategoryTheory.Category.{?u.78601, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.78643, u_3} E
      X Y : C
      f : CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom W X Y
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      a₁ a₂ : W.LeftFraction X Y
      Z : C
      t₁ : Quiver.Hom a₁.Y' Z
      t₂ : Quiver.Hom a₂.Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp a₁.s t₁) (CategoryTheory.Category …
      hft : Eq (CategoryTheory.CategoryStruct.comp a₁.f t₁) (CategoryTheory.Category …
      h : W (CategoryTheory.CategoryStruct.comp a₁.s t₁)
      ⊢ Eq (a₁.map F hF) (a₂.map F hF)
    -/
    have := hF _ h
    rw [← cancel_mono (F.map (a₁.s ≫ t₁)), F.map_comp, map_comp_map_s_assoc,
      ← F.map_comp, ← F.map_comp, hst, hft, F.map_comp,
      F.map_comp, map_comp_map_s_assoc]) f


@[simp]
lemma Hom.map_mk {W} {X Y : C} (f : LeftFraction W X Y)
    (F : C ⥤ E) (hF : W.IsInvertedBy F) :
  Hom.map (Hom.mk f) F hF = f.map F hF := rfl


lemma inverts : W.IsInvertedBy (Q W) := fun _ _ s hs =>
  (inferInstance : IsIso (Qiso s hs).hom)


/-- The functor `Localization W ⥤ E` that is induced by a functor `C ⥤ E` which inverts `W`,
when `W` has a left calculus of fractions. -/
noncomputable def lift (F : C ⥤ E) (hF : W.IsInvertedBy F) :
    Localization W ⥤ E where
  obj X := F.obj X
  map {_ _ : C} f := f.map F hF
  map_id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      ⊢ ∀ (X : CategoryTheory.MorphismProperty.LeftFraction.Localization W), Eq ({ o …
    -/
    intro (X : C)
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X : C
      ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => CategoryTheory.Morphi …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X : C
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.map (Categ …
    -/
    change (Hom.mk (ofHom W (𝟙 X))).map F hF = _
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X : C
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk (Categ …
    -/
    rw [Hom.map_mk, map_ofHom, F.map_id]
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      ⊢ ∀ {X Y Z : CategoryTheory.MorphismProperty.LeftFraction.Localization W} (f : …
    -/
    rintro (X Y Z : C) f g
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => CategoryTheory.Morphi …
    -/
    obtain ⟨f, rfl⟩ := Hom.mk_surjective f
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      g : Quiver.Hom Y Z
      f : W.LeftFraction X Y
      ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => CategoryTheory.Morphi …
    -/
    obtain ⟨g, rfl⟩ := Hom.mk_surjective g
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => CategoryTheory.Morphi …
    -/
    dsimp
    obtain ⟨z, fac⟩ := HasLeftCalculusOfFractions.exists_leftFraction
      (RightFraction.mk f.s f.hs g.f)
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.map (Categ …
    -/
    rw [homMk_comp_homMk f g z fac, Hom.map_mk]
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
      ⊢ Eq ((f.comp₀ g z).map F hF) (CategoryTheory.CategoryStruct.comp (f.map F hF) …
    -/
    dsimp at fac ⊢
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp g.f z.s) (CategoryTheory.Category …
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (CategoryTheory.Categor …
    -/
    have := hF _ g.hs
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp g.f z.s) (CategoryTheory.Category …
      this : CategoryTheory.IsIso (F.map g.s)
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (CategoryTheory.Categor …
    -/
    have := hF _ z.hs
    rw [← cancel_mono (F.map g.s), assoc, map_comp_map_s,
      ← cancel_mono (F.map z.s), assoc, assoc, ← F.map_comp,
      ← F.map_comp, map_comp_map_s, fac]
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp g.f z.s) (CategoryTheory.Category …
      this✝ : CategoryTheory.IsIso (F.map g.s)
      this : CategoryTheory.IsIso (F.map z.s)
      ⊢ Eq (F.map (CategoryTheory.MorphismProperty.LeftFraction.mk (CategoryTheory.C …
    -/
    dsimp
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.82278, u_1} C
      inst✝² : CategoryTheory.Category.{?u.82282, u_2} D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.82324, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y Z : C
      f : W.LeftFraction X Y
      g : W.LeftFraction Y Z
      z : W.LeftFraction f.Y' g.Y'
      fac : Eq (CategoryTheory.CategoryStruct.comp g.f z.s) (CategoryTheory.Category …
      this✝ : CategoryTheory.IsIso (F.map g.s)
      this : CategoryTheory.IsIso (F.map z.s)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f.f z.f)) (CategoryTheory.Cate …
    -/
    rw [F.map_comp, F.map_comp, map_comp_map_s_assoc]
    /-
      🎉 no goals
    -/


lemma fac (F : C ⥤ E) (hF : W.IsInvertedBy F) : Q W ⋙ lift F hF = F :=
  Functor.ext (fun _ => rfl) (fun X Y f => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp (C …
    -/
    dsimp [lift]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.map ((Cate …
    -/
    rw [Q_map, Hom.map_mk, id_comp, comp_id, map_ofHom])
    /-
      🎉 no goals
    -/


lemma uniq (F₁ F₂ : Localization W ⥤ E) (h : Q W ⋙ F₁ = Q W ⋙ F₂) : F₁ = F₂ :=
  Functor.ext (fun X => Functor.congr_obj h X) (by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      ⊢ ∀ (X Y : CategoryTheory.MorphismProperty.LeftFraction.Localization W) (f : Q …
    -/
    rintro (X Y : C) f
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (F₁.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) …
    -/
    obtain ⟨f, rfl⟩ := Hom.mk_surjective f
    rw [show Hom.mk f = homMk (mk f.f f.s f.hs) by rfl,
      ← Q_map_comp_Qinv f.f f.s f.hs, F₁.map_comp, F₂.map_comp, assoc]
    /-
      case intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map ((CategoryTheory.MorphismProp …
    -/
    erw [Functor.congr_hom h f.f]
    /-
      case intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, assoc]
    /-
      case intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    congr 2
    /-
      case intro.e_a.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (F₁.map (C …
    -/
    have := inverts W _ f.hs
    rw [← cancel_epi (F₂.map ((Q W).map f.s)), ← F₂.map_comp_assoc,
      Qiso_hom_inv_id, Functor.map_id, id_comp]
    /-
      case intro.e_a.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      this : CategoryTheory.IsIso ((CategoryTheory.MorphismProperty.LeftFraction.Loc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₂.map ((CategoryTheory.MorphismProp …
    -/
    erw [Functor.congr_hom h.symm f.s]
    /-
      case intro.e_a.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      this : CategoryTheory.IsIso ((CategoryTheory.MorphismProperty.LeftFraction.Loc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp
    rw [assoc, assoc, eqToHom_trans_assoc, eqToHom_refl, id_comp, ← F₁.map_comp,
      Qiso_hom_inv_id]
    /-
      case intro.e_a.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      this : CategoryTheory.IsIso ((CategoryTheory.MorphismProperty.LeftFraction.Loc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (F₁.map (C …
    -/
    dsimp
    /-
      case intro.e_a.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝ : CategoryTheory.Category.{u_5, u_3} E
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.MorphismProperty.LeftFraction.L …
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W).comp F …
      X Y : C
      f : W.LeftFraction X Y
      this : CategoryTheory.IsIso ((CategoryTheory.MorphismProperty.LeftFraction.Loc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (F₁.map (C …
    -/
    rw [F₁.map_id, comp_id])
    /-
      🎉 no goals
    -/


open StrictUniversalPropertyFixedTarget in
/-- The universal property of the localization for the constructed localized category
when there is a left calculus of fractions. -/
noncomputable def strictUniversalPropertyFixedTarget (E : Type*) [Category E] :
    Localization.StrictUniversalPropertyFixedTarget (Q W) W E where
  inverts := inverts W
  lift := lift
  fac := fac
  uniq := uniq


instance : (Q W).IsLocalization W :=
  Functor.IsLocalization.mk' _ _
    (strictUniversalPropertyFixedTarget W _)
    (strictUniversalPropertyFixedTarget W _)


lemma homMk_eq {X Y : C} (f : LeftFraction W X Y) :
    homMk f = f.map (Q W) (Localization.inverts _ W) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : W.LeftFraction X Y
    ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.homMk f) (f.ma …
  -/
  have := Localization.inverts (Q W) W f.s f.hs
  rw [← Q_map_comp_Qinv f.f f.s f.hs, ← cancel_mono ((Q W).map f.s),
    assoc, Qiso_inv_hom_id, comp_id, map_comp_map_s]


lemma map_eq_iff {X Y : C} (f g : LeftFraction W X Y) :
    f.map (LeftFraction.Localization.Q W) (Localization.inverts _ _) =
        g.map (LeftFraction.Localization.Q W) (Localization.inverts _ _) ↔
      LeftFractionRel f g := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f g : W.LeftFraction X Y
    ⊢ Iff (Eq (f.map (CategoryTheory.MorphismProperty.LeftFraction.Localization.Q  …
  -/
  simp only [← Hom.map_mk _ (Q W)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f g : W.LeftFraction X Y
    ⊢ Iff (Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f).map …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f).m …
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel f g
    -/
    rw [← homMk_eq_iff_leftFractionRel, homMk_eq, homMk_eq]
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      h : Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f).m …
      ⊢ Eq (f.map (CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W) ⋯) …
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel f g → Eq ((CategoryTheory.Mo …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel f g
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f).map …
    -/
    congr 1
    /-
      case mpr.e_f
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      W : CategoryTheory.MorphismProperty C
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f g : W.LeftFraction X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel f g
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk f) (Cat …
    -/
    exact Quot.sound h
    /-
      🎉 no goals
    -/


lemma map_eq {W} {X Y : C} (φ : W.LeftFraction X Y) (L : C ⥤ D) [L.IsLocalization W] :
    φ.map L (Localization.inverts L W) =
      L.map φ.f ≫ (Localization.isoOfHom L W φ.s φ.hs).inv := rfl


lemma map_compatibility {W} {X Y : C}
    (φ : W.LeftFraction X Y) {E : Type*} [Category E]
    (L₁ : C ⥤ D) (L₂ : C ⥤ E) [L₁.IsLocalization W] [L₂.IsLocalization W] :
    (Localization.uniq L₁ L₂ W).functor.map (φ.map L₁ (Localization.inverts L₁ W)) =
      (Localization.compUniqFunctor L₁ L₂ W).hom.app X ≫
        φ.map L₂ (Localization.inverts L₂ W) ≫
        (Localization.compUniqFunctor L₁ L₂ W).inv.app Y := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    E : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} E
    L₁ : CategoryTheory.Functor C D
    L₂ : CategoryTheory.Functor C E
    inst✝¹ : L₁.IsLocalization W
    inst✝ : L₂.IsLocalization W
    ⊢ Eq ((CategoryTheory.Localization.uniq L₁ L₂ W).functor.map (φ.map L₁ ⋯)) (Ca …
  -/
  let e := Localization.compUniqFunctor L₁ L₂ W
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    E : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} E
    L₁ : CategoryTheory.Functor C D
    L₂ : CategoryTheory.Functor C E
    inst✝¹ : L₁.IsLocalization W
    inst✝ : L₂.IsLocalization W
    e : CategoryTheory.Iso (L₁.comp (CategoryTheory.Localization.uniq L₁ L₂ W).fun …
    ⊢ Eq ((CategoryTheory.Localization.uniq L₁ L₂ W).functor.map (φ.map L₁ ⋯)) (Ca …
  -/
  have := Localization.inverts L₂ W φ.s φ.hs
  rw [← cancel_mono (e.hom.app Y), assoc, assoc, e.inv_hom_id_app, comp_id,
    ← cancel_mono (L₂.map φ.s), assoc, assoc, map_comp_map_s, ← e.hom.naturality]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    E : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} E
    L₁ : CategoryTheory.Functor C D
    L₂ : CategoryTheory.Functor C E
    inst✝¹ : L₁.IsLocalization W
    inst✝ : L₂.IsLocalization W
    e : CategoryTheory.Iso (L₁.comp (CategoryTheory.Localization.uniq L₁ L₂ W).fun …
    this : CategoryTheory.IsIso (L₂.map φ.s)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.uniq L₁ …
  -/
  simpa [← Functor.map_comp_assoc, map_comp_map_s] using e.hom.naturality φ.f
  /-
    🎉 no goals
  -/


lemma map_eq_of_map_eq {W} {X Y : C}
    (φ₁ φ₂ : W.LeftFraction X Y) {E : Type*} [Category E]
    (L₁ : C ⥤ D) (L₂ : C ⥤ E) [L₁.IsLocalization W] [L₂.IsLocalization W]
    (h : φ₁.map L₁ (Localization.inverts L₁ W) = φ₂.map L₁ (Localization.inverts L₁ W)) :
    φ₁.map L₂ (Localization.inverts L₂ W) = φ₂.map L₂ (Localization.inverts L₂ W) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ₁ φ₂ : W.LeftFraction X Y
    E : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} E
    L₁ : CategoryTheory.Functor C D
    L₂ : CategoryTheory.Functor C E
    inst✝¹ : L₁.IsLocalization W
    inst✝ : L₂.IsLocalization W
    h : Eq (φ₁.map L₁ ⋯) (φ₂.map L₁ ⋯)
    ⊢ Eq (φ₁.map L₂ ⋯) (φ₂.map L₂ ⋯)
  -/
  apply (Localization.uniq L₂ L₁ W).functor.map_injective
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ₁ φ₂ : W.LeftFraction X Y
    E : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} E
    L₁ : CategoryTheory.Functor C D
    L₂ : CategoryTheory.Functor C E
    inst✝¹ : L₁.IsLocalization W
    inst✝ : L₂.IsLocalization W
    h : Eq (φ₁.map L₁ ⋯) (φ₂.map L₁ ⋯)
    ⊢ Eq ((CategoryTheory.Localization.uniq L₂ L₁ W).functor.map (φ₁.map L₂ ⋯)) (( …
  -/
  rw [map_compatibility φ₁ L₂ L₁, map_compatibility φ₂ L₂ L₁, h]
  /-
    🎉 no goals
  -/


lemma map_comp_map_eq_map {X Y Z : C} (z₁ : W.LeftFraction X Y) (z₂ : W.LeftFraction Y Z)
    (z₃ : W.LeftFraction z₁.Y' z₂.Y') (h₃ : z₂.f ≫ z₃.s = z₁.s ≫ z₃.f)
    (L : C ⥤ D) [L.IsLocalization W] :
    z₁.map L (Localization.inverts L W) ≫ z₂.map L (Localization.inverts L W) =
      (z₁.comp₀ z₂ z₃).map L (Localization.inverts L W) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z₁.map L ⋯) (z₂.map L ⋯)) ((z₁.comp₀ …
  -/
  have := Localization.inverts L W _ z₂.hs
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    this : CategoryTheory.IsIso (L.map z₂.s)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z₁.map L ⋯) (z₂.map L ⋯)) ((z₁.comp₀ …
  -/
  have := Localization.inverts L W _ z₃.hs
  have : IsIso (L.map (z₂.s ≫ z₃.s)) := by
    rw [L.map_comp]
    infer_instance
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.HasLeftCalculusOfFractions
    X Y Z : C
    z₁ : W.LeftFraction X Y
    z₂ : W.LeftFraction Y Z
    z₃ : W.LeftFraction z₁.Y' z₂.Y'
    h₃ : Eq (CategoryTheory.CategoryStruct.comp z₂.f z₃.s) (CategoryTheory.Categor …
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    this✝¹ : CategoryTheory.IsIso (L.map z₂.s)
    this✝ : CategoryTheory.IsIso (L.map z₃.s)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp z₂.s z₃ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z₁.map L ⋯) (z₂.map L ⋯)) ((z₁.comp₀ …
  -/
  dsimp [LeftFraction.comp₀]
  rw [← cancel_mono (L.map (z₂.s ≫ z₃.s)), map_comp_map_s,
    L.map_comp, assoc, map_comp_map_s_assoc, ← L.map_comp, h₃,
    L.map_comp, map_comp_map_s_assoc, L.map_comp]


lemma Localization.exists_leftFraction {X Y : C} (f : L.obj X ⟶ L.obj Y) :
    ∃ (φ : W.LeftFraction X Y), f = φ.map L (Localization.inverts L W) := by
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
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Exists fun φ => Eq f (φ.map L ⋯)
  -/
  let E := Localization.uniq (MorphismProperty.LeftFraction.Localization.Q W) L W
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
    f : Quiver.Hom (L.obj X) (L.obj Y)
    E : CategoryTheory.Equivalence (CategoryTheory.MorphismProperty.LeftFraction.L …
    ⊢ Exists fun φ => Eq f (φ.map L ⋯)
  -/
  let e : _ ⋙ E.functor ≅ L := Localization.compUniqFunctor _ _ _
  obtain ⟨f', rfl⟩ : ∃ (f' : E.functor.obj X ⟶ E.functor.obj Y),
      f = e.inv.app _ ≫ f' ≫ e.hom.app _ := ⟨e.hom.app _ ≫ f ≫ e.inv.app _, by simp⟩
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
    E : CategoryTheory.Equivalence (CategoryTheory.MorphismProperty.LeftFraction.L …
    e : CategoryTheory.Iso ((CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    f' : Quiver.Hom (E.functor.obj X) (E.functor.obj Y)
    ⊢ Exists fun φ => Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (Catego …
  -/
  obtain ⟨g, rfl⟩ := E.functor.map_surjective f'
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
    E : CategoryTheory.Equivalence (CategoryTheory.MorphismProperty.LeftFraction.L …
    e : CategoryTheory.Iso ((CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    g : Quiver.Hom X Y
    ⊢ Exists fun φ => Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (Catego …
  -/
  obtain ⟨g, rfl⟩ := MorphismProperty.LeftFraction.Localization.Hom.mk_surjective g
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
    E : CategoryTheory.Equivalence (CategoryTheory.MorphismProperty.LeftFraction.L …
    e : CategoryTheory.Iso ((CategoryTheory.MorphismProperty.LeftFraction.Localiza …
    g : W.LeftFraction X Y
    ⊢ Exists fun φ => Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (Catego …
  -/
  refine ⟨g, ?_⟩
  rw [← MorphismProperty.LeftFraction.Localization.homMk_eq_hom_mk,
    MorphismProperty.LeftFraction.Localization.homMk_eq g,
    g.map_compatibility (MorphismProperty.LeftFraction.Localization.Q W) L,
    assoc, assoc, Iso.inv_hom_id_app, comp_id, Iso.inv_hom_id_app_assoc]


lemma MorphismProperty.LeftFraction.map_eq_iff
    {X Y : C} (φ ψ : W.LeftFraction X Y) :
    φ.map L (Localization.inverts _ _) = ψ.map L (Localization.inverts _ _) ↔
      LeftFractionRel φ ψ := by
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
    φ ψ : W.LeftFraction X Y
    ⊢ Iff (Eq (φ.map L ⋯) (ψ.map L ⋯)) (CategoryTheory.MorphismProperty.LeftFracti …
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
      φ ψ : W.LeftFraction X Y
      ⊢ Eq (φ.map L ⋯) (ψ.map L ⋯) → CategoryTheory.MorphismProperty.LeftFractionRel …
    -/
  · intro h
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
      φ ψ : W.LeftFraction X Y
      h : Eq (φ.map L ⋯) (ψ.map L ⋯)
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel φ ψ
    -/
    rw [← MorphismProperty.LeftFraction.Localization.map_eq_iff]
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
      φ ψ : W.LeftFraction X Y
      h : Eq (φ.map L ⋯) (ψ.map L ⋯)
      ⊢ Eq (φ.map (CategoryTheory.MorphismProperty.LeftFraction.Localization.Q W) ⋯) …
    -/
    apply map_eq_of_map_eq _ _ _ _ h
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
      φ ψ : W.LeftFraction X Y
      ⊢ CategoryTheory.MorphismProperty.LeftFractionRel φ ψ → Eq (φ.map L ⋯) (ψ.map  …
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
      φ ψ : W.LeftFraction X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel φ ψ
      ⊢ Eq (φ.map L ⋯) (ψ.map L ⋯)
    -/
    simp only [← Localization.Hom.map_mk _ L (Localization.inverts _ _)]
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
      φ ψ : W.LeftFraction X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel φ ψ
      ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk φ).map …
    -/
    congr 1
    /-
      case mpr.e_f
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      φ ψ : W.LeftFraction X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel φ ψ
      ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.Localization.Hom.mk φ) (Cat …
    -/
    exact Quot.sound h
    /-
      🎉 no goals
    -/


lemma MorphismProperty.map_eq_iff_postcomp {X Y : C} (f₁ f₂ : X ⟶ Y) :
    L.map f₁ = L.map f₂ ↔ ∃ (Z : C) (s : Y ⟶ Z) (_ : W s), f₁ ≫ s = f₂ ≫ s := by
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
    f₁ f₂ : Quiver.Hom X Y
    ⊢ Iff (Eq (L.map f₁) (L.map f₂)) (Exists fun Z => Exists fun s => Exists fun x …
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
      f₁ f₂ : Quiver.Hom X Y
      ⊢ Eq (L.map f₁) (L.map f₂) → Exists fun Z => Exists fun s => Exists fun x => E …
    -/
  · intro h
    rw [← LeftFraction.map_ofHom W _ L (Localization.inverts _ _),
      ← LeftFraction.map_ofHom W _ L (Localization.inverts _ _),
      LeftFraction.map_eq_iff] at h
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
      f₁ f₂ : Quiver.Hom X Y
      h : CategoryTheory.MorphismProperty.LeftFractionRel (CategoryTheory.MorphismPr …
      ⊢ Exists fun Z => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
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
      f₁ f₂ : Quiver.Hom X Y
      Z : C
      t₁ : Quiver.Hom (CategoryTheory.MorphismProperty.LeftFraction.ofHom W f₁).Y' Z
      t₂ : Quiver.Hom (CategoryTheory.MorphismProperty.LeftFraction.ofHom W f₂).Y' Z
      hst : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
      hft : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
      ht : W (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Le …
      ⊢ Exists fun Z => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    dsimp at t₁ t₂ hst hft ht
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
      f₁ f₂ : Quiver.Hom X Y
      Z : C
      t₁ t₂ : Quiver.Hom Y Z
      hst : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id …
      hft : Eq (CategoryTheory.CategoryStruct.comp f₁ t₁) (CategoryTheory.CategorySt …
      ht : W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y …
      ⊢ Exists fun Z => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    simp only [id_comp] at hst
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
      f₁ f₂ : Quiver.Hom X Y
      Z : C
      t₁ t₂ : Quiver.Hom Y Z
      hft : Eq (CategoryTheory.CategoryStruct.comp f₁ t₁) (CategoryTheory.CategorySt …
      ht : W (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y …
      hst : Eq t₁ t₂
      ⊢ Exists fun Z => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    exact ⟨Z, t₁, by simpa using ht, by rw [hft, hst]⟩
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
      f₁ f₂ : Quiver.Hom X Y
      ⊢ (Exists fun Z => Exists fun s => Exists fun x => Eq (CategoryTheory.Category …
    -/
  · rintro ⟨Z, s, hs, fac⟩
    simp only [← cancel_mono (Localization.isoOfHom L W s hs).hom,
      Localization.isoOfHom_hom, ← L.map_comp, fac]


include W in
lemma Localization.essSurj_mapArrow :
    L.mapArrow.EssSurj where
  mem_essImage f := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      f : CategoryTheory.Arrow D
      ⊢ Membership.mem L.mapArrow.essImage f
    -/
    have := Localization.essSurj L W
    obtain ⟨X, ⟨eX⟩⟩ : ∃ (X : C), Nonempty (L.obj X ≅ f.left) :=
      ⟨_, ⟨L.objObjPreimageIso f.left⟩⟩
    obtain ⟨Y, ⟨eY⟩⟩ : ∃ (Y : C), Nonempty (L.obj Y ≅ f.right) :=
      ⟨_, ⟨L.objObjPreimageIso f.right⟩⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      f : CategoryTheory.Arrow D
      this : L.EssSurj
      X : C
      eX : CategoryTheory.Iso (L.obj X) f.left
      Y : C
      eY : CategoryTheory.Iso (L.obj Y) f.right
      ⊢ Membership.mem L.mapArrow.essImage f
    -/
    obtain ⟨φ, hφ⟩ := Localization.exists_leftFraction L W (eX.hom ≫ f.hom ≫ eY.inv)
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      f : CategoryTheory.Arrow D
      this : L.EssSurj
      X : C
      eX : CategoryTheory.Iso (L.obj X) f.left
      Y : C
      eY : CategoryTheory.Iso (L.obj Y) f.right
      φ : W.LeftFraction X Y
      hφ : Eq (CategoryTheory.CategoryStruct.comp eX.hom (CategoryTheory.CategoryStr …
      ⊢ Membership.mem L.mapArrow.essImage f
    -/
    refine ⟨Arrow.mk φ.f, ⟨Iso.symm ?_⟩⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      f : CategoryTheory.Arrow D
      this : L.EssSurj
      X : C
      eX : CategoryTheory.Iso (L.obj X) f.left
      Y : C
      eY : CategoryTheory.Iso (L.obj Y) f.right
      φ : W.LeftFraction X Y
      hφ : Eq (CategoryTheory.CategoryStruct.comp eX.hom (CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.Iso f (L.mapArrow.obj (CategoryTheory.Arrow.mk φ.f))
    -/
    refine Arrow.isoMk eX.symm (eY.symm ≪≫ Localization.isoOfHom L W φ.s φ.hs) ?_
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      f : CategoryTheory.Arrow D
      this : L.EssSurj
      X : C
      eX : CategoryTheory.Iso (L.obj X) f.left
      Y : C
      eY : CategoryTheory.Iso (L.obj Y) f.right
      φ : W.LeftFraction X Y
      hφ : Eq (CategoryTheory.CategoryStruct.comp eX.hom (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp eX.symm.hom (L.mapArrow.obj (Category …
    -/
    dsimp
    simp only [← cancel_epi eX.hom, Iso.hom_inv_id_assoc, reassoc_of% hφ,
      MorphismProperty.LeftFraction.map_comp_map_s]


/-- The right fraction in the opposite category corresponding to a left fraction. -/
@[simps]
def LeftFraction.op {X Y : C} (φ : W.LeftFraction X Y) :
    W.op.RightFraction (Opposite.op Y) (Opposite.op X) where
  X' := Opposite.op φ.Y'
  s := φ.s.op
  hs := φ.hs
  f := φ.f.op


/-- The left fraction in the opposite category corresponding to a right fraction. -/
@[simps]
def RightFraction.op {X Y : C} (φ : W.RightFraction X Y) :
    W.op.LeftFraction (Opposite.op Y) (Opposite.op X) where
  Y' := Opposite.op φ.X'
  s := φ.s.op
  hs := φ.hs
  f := φ.f.op


/-- The right fraction corresponding to a left fraction in the opposite category. -/
@[simps]
def LeftFraction.unop {W : MorphismProperty Cᵒᵖ}
    {X Y : Cᵒᵖ} (φ : W.LeftFraction X Y) :
    W.unop.RightFraction (Opposite.unop Y) (Opposite.unop X) where
  X' := Opposite.unop φ.Y'
  s := φ.s.unop
  hs := φ.hs
  f := φ.f.unop


/-- The left fraction corresponding to a right fraction in the opposite category. -/
@[simps]
def RightFraction.unop {W : MorphismProperty Cᵒᵖ}
    {X Y : Cᵒᵖ} (φ : W.RightFraction X Y) :
    W.unop.LeftFraction (Opposite.unop Y) (Opposite.unop X) where
  Y' := Opposite.unop φ.X'
  s := φ.s.unop
  hs := φ.hs
  f := φ.f.unop


lemma RightFraction.op_map
    {X Y : C} (φ : W.RightFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    (φ.map L hL).op = φ.op.map L.op hL.op := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (φ.map L hL).op (φ.op.map L.op ⋯)
  -/
  dsimp [map, LeftFraction.map]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.RightFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.f).op (CategoryTheory.inv (L …
  -/
  rw [op_inv]
  /-
    🎉 no goals
  -/


lemma LeftFraction.op_map
    {X Y : C} (φ : W.LeftFraction X Y) (L : C ⥤ D) (hL : W.IsInvertedBy L) :
    (φ.map L hL).op = φ.op.map L.op hL.op := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (φ.map L hL).op (φ.op.map L.op ⋯)
  -/
  dsimp [map, RightFraction.map]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction X Y
    L : CategoryTheory.Functor C D
    hL : W.IsInvertedBy L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (L.map φ.s)).op ( …
  -/
  rw [op_inv]
  /-
    🎉 no goals
  -/


instance [h : W.HasLeftCalculusOfFractions] : W.op.HasRightCalculusOfFractions where
  exists_rightFraction X Y φ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.135537, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasLeftCalculusOfFractions
      X Y : Opposite C
      φ : W.op.LeftFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    obtain ⟨ψ, eq⟩ := h.exists_leftFraction φ.unop
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.135537, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasLeftCalculusOfFractions
      X Y : Opposite C
      φ : W.op.LeftFraction X Y
      ψ : W.LeftFraction (Opposite.unop Y) (Opposite.unop X)
      eq : Eq (CategoryTheory.CategoryStruct.comp φ.unop.f ψ.s) (CategoryTheory.Cate …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    exact ⟨ψ.op, Quiver.Hom.unop_inj eq⟩
    /-
      🎉 no goals
    -/
  ext X Y Y' f₁ f₂ s hs eq := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.135537, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasLeftCalculusOfFractions
      X Y Y' : Opposite C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom Y Y'
      hs : W.op s
      eq : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStru …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨X', t, ht, fac⟩ := h.ext f₁.unop f₂.unop s.unop hs (Quiver.Hom.op_inj eq)
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.135537, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasLeftCalculusOfFractions
      X Y Y' : Opposite C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom Y Y'
      hs : W.op s
      eq : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStru …
      X' : C
      t : Quiver.Hom (Opposite.unop X) X'
      ht : W t
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁.unop t) (CategoryTheory.Catego …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    exact ⟨Opposite.op X', t.op, ht, Quiver.Hom.unop_inj fac⟩
    /-
      🎉 no goals
    -/


instance [h : W.HasRightCalculusOfFractions] : W.op.HasLeftCalculusOfFractions where
  exists_leftFraction X Y φ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.136343, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasRightCalculusOfFractions
      X Y : Opposite C
      φ : W.op.RightFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    obtain ⟨ψ, eq⟩ := h.exists_rightFraction φ.unop
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.136343, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasRightCalculusOfFractions
      X Y : Opposite C
      φ : W.op.RightFraction X Y
      ψ : W.RightFraction (Opposite.unop Y) (Opposite.unop X)
      eq : Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.unop.f) (CategoryTheory.Cate …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    exact ⟨ψ.op, Quiver.Hom.unop_inj eq⟩
    /-
      🎉 no goals
    -/
  ext X' X Y f₁ f₂ s hs eq := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.136343, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasRightCalculusOfFractions
      X' X Y : Opposite C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      hs : W.op s
      eq : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStru …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨Y', t, ht, fac⟩ := h.ext f₁.unop f₂.unop s.unop hs (Quiver.Hom.op_inj eq)
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.136343, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      h : W.HasRightCalculusOfFractions
      X' X Y : Opposite C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      hs : W.op s
      eq : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStru …
      Y' : C
      t : Quiver.Hom Y' (Opposite.unop Y)
      ht : W t
      fac : Eq (CategoryTheory.CategoryStruct.comp t f₁.unop) (CategoryTheory.Catego …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    exact ⟨Opposite.op Y', t.op, ht, Quiver.Hom.unop_inj fac⟩
    /-
      🎉 no goals
    -/


instance (W : MorphismProperty Cᵒᵖ) [h : W.HasLeftCalculusOfFractions] :
    W.unop.HasRightCalculusOfFractions where
  exists_rightFraction X Y φ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137144, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasLeftCalculusOfFractions
      X Y : C
      φ : W.unop.LeftFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    obtain ⟨ψ, eq⟩ := h.exists_leftFraction φ.op
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137144, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasLeftCalculusOfFractions
      X Y : C
      φ : W.unop.LeftFraction X Y
      ψ : W.LeftFraction { unop := Y } { unop := X }
      eq : Eq (CategoryTheory.CategoryStruct.comp φ.op.f ψ.s) (CategoryTheory.Catego …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    exact ⟨ψ.unop, Quiver.Hom.op_inj eq⟩
    /-
      🎉 no goals
    -/
  ext X Y Y' f₁ f₂ s hs eq := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137144, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasLeftCalculusOfFractions
      X Y Y' : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom Y Y'
      hs : W.unop s
      eq : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStru …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨X', t, ht, fac⟩ := h.ext f₁.op f₂.op s.op hs (Quiver.Hom.unop_inj eq)
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137144, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasLeftCalculusOfFractions
      X Y Y' : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom Y Y'
      hs : W.unop s
      eq : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStru …
      X' : Opposite C
      t : Quiver.Hom { unop := X } X'
      ht : W t
      fac : Eq (CategoryTheory.CategoryStruct.comp f₁.op t) (CategoryTheory.Category …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    exact ⟨Opposite.unop X', t.unop, ht, Quiver.Hom.op_inj fac⟩
    /-
      🎉 no goals
    -/


instance (W : MorphismProperty Cᵒᵖ) [h : W.HasRightCalculusOfFractions] :
    W.unop.HasLeftCalculusOfFractions where
  exists_leftFraction X Y φ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137954, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasRightCalculusOfFractions
      X Y : C
      φ : W.unop.RightFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    obtain ⟨ψ, eq⟩ := h.exists_rightFraction φ.op
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137954, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasRightCalculusOfFractions
      X Y : C
      φ : W.unop.RightFraction X Y
      ψ : W.RightFraction { unop := Y } { unop := X }
      eq : Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.op.f) (CategoryTheory.Catego …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    exact ⟨ψ.unop, Quiver.Hom.op_inj eq⟩
    /-
      🎉 no goals
    -/
  ext X' X Y f₁ f₂ s hs eq := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137954, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasRightCalculusOfFractions
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      hs : W.unop s
      eq : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStru …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨Y', t, ht, fac⟩ := h.ext f₁.op f₂.op s.op hs (Quiver.Hom.unop_inj eq)
    /-
      case intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.137954, u_2} D
      L : CategoryTheory.Functor C D
      W✝ : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W✝
      W : CategoryTheory.MorphismProperty (Opposite C)
      h : W.HasRightCalculusOfFractions
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      hs : W.unop s
      eq : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStru …
      Y' : Opposite C
      t : Quiver.Hom Y' { unop := Y }
      ht : W t
      fac : Eq (CategoryTheory.CategoryStruct.comp t f₁.op) (CategoryTheory.Category …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    exact ⟨Opposite.unop Y', t.unop, ht, Quiver.Hom.op_inj fac⟩
    /-
      🎉 no goals
    -/


/-- The equivalence relation on right fractions for a morphism property `W`. -/
def RightFractionRel {X Y : C} (z₁ z₂ : W.RightFraction X Y) : Prop :=
  ∃ (Z : C) (t₁ : Z ⟶ z₁.X') (t₂ : Z ⟶ z₂.X') (_ : t₁ ≫ z₁.s = t₂ ≫ z₂.s)
    (_ : t₁ ≫ z₁.f = t₂ ≫ z₂.f), W (t₁ ≫ z₁.s)


lemma RightFractionRel.op {X Y : C} {z₁ z₂ : W.RightFraction X Y}
    (h : RightFractionRel z₁ z₂) : LeftFractionRel z₁.op z₂.op := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.RightFraction X Y
    h : CategoryTheory.MorphismProperty.RightFractionRel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.op z₂.op
  -/
  obtain ⟨Z, t₁, t₂, hs, hf, ht⟩ := h
  exact ⟨Opposite.op Z, t₁.op, t₂.op, Quiver.Hom.unop_inj hs,
    Quiver.Hom.unop_inj hf, ht⟩


lemma RightFractionRel.unop {W : MorphismProperty Cᵒᵖ} {X Y : Cᵒᵖ}
    {z₁ z₂ : W.RightFraction X Y}
    (h : RightFractionRel z₁ z₂) : LeftFractionRel z₁.unop z₂.unop := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty (Opposite C)
    X Y : Opposite C
    z₁ z₂ : W.RightFraction X Y
    h : CategoryTheory.MorphismProperty.RightFractionRel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.LeftFractionRel z₁.unop z₂.unop
  -/
  obtain ⟨Z, t₁, t₂, hs, hf, ht⟩ := h
  exact ⟨Opposite.unop Z, t₁.unop, t₂.unop, Quiver.Hom.op_inj hs,
    Quiver.Hom.op_inj hf, ht⟩


lemma LeftFractionRel.op {X Y : C} {z₁ z₂ : W.LeftFraction X Y}
    (h : LeftFractionRel z₁ z₂) : RightFractionRel z₁.op z₂.op := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    z₁ z₂ : W.LeftFraction X Y
    h : CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.RightFractionRel z₁.op z₂.op
  -/
  obtain ⟨Z, t₁, t₂, hs, hf, ht⟩ := h
  exact ⟨Opposite.op Z, t₁.op, t₂.op, Quiver.Hom.unop_inj hs,
    Quiver.Hom.unop_inj hf, ht⟩


lemma LeftFractionRel.unop {W : MorphismProperty Cᵒᵖ} {X Y : Cᵒᵖ}
    {z₁ z₂ : W.LeftFraction X Y}
    (h : LeftFractionRel z₁ z₂) : RightFractionRel z₁.unop z₂.unop := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    W : CategoryTheory.MorphismProperty (Opposite C)
    X Y : Opposite C
    z₁ z₂ : W.LeftFraction X Y
    h : CategoryTheory.MorphismProperty.LeftFractionRel z₁ z₂
    ⊢ CategoryTheory.MorphismProperty.RightFractionRel z₁.unop z₂.unop
  -/
  obtain ⟨Z, t₁, t₂, hs, hf, ht⟩ := h
  exact ⟨Opposite.unop Z, t₁.unop, t₂.unop, Quiver.Hom.op_inj hs,
    Quiver.Hom.op_inj hf, ht⟩


lemma leftFractionRel_op_iff
    {X Y : C} (z₁ z₂ : W.RightFraction X Y) :
    LeftFractionRel z₁.op z₂.op ↔ RightFractionRel z₁ z₂ :=
  ⟨fun h => h.unop, fun h => h.op⟩


lemma rightFractionRel_op_iff
    {X Y : C} (z₁ z₂ : W.LeftFraction X Y) :
    RightFractionRel z₁.op z₂.op ↔ LeftFractionRel z₁ z₂ :=
  ⟨fun h => h.unop, fun h => h.op⟩


lemma refl {X Y : C} (z : W.RightFraction X Y) : RightFractionRel z z :=
  (LeftFractionRel.refl z.op).unop


lemma symm {X Y : C} {z₁ z₂ : W.RightFraction X Y} (h : RightFractionRel z₁ z₂) :
    RightFractionRel z₂ z₁ :=
  h.op.symm.unop


lemma trans {X Y : C} {z₁ z₂ z₃ : W.RightFraction X Y}
    [HasRightCalculusOfFractions W]
    (h₁₂ : RightFractionRel z₁ z₂) (h₂₃ : RightFractionRel z₂ z₃) :
    RightFractionRel z₁ z₃ :=
  (h₁₂.op.trans h₂₃.op).unop


lemma equivalenceRightFractionRel (X Y : C) [HasRightCalculusOfFractions W] :
    @_root_.Equivalence (W.RightFraction X Y) RightFractionRel where
  refl := RightFractionRel.refl
  symm := RightFractionRel.symm
  trans := RightFractionRel.trans


lemma Localization.exists_rightFraction {X Y : C} (f : L.obj X ⟶ L.obj Y) :
    ∃ (φ : W.RightFraction X Y), f = φ.map L (Localization.inverts L W) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Exists fun φ => Eq f (φ.map L ⋯)
  -/
  obtain ⟨φ, eq⟩ := Localization.exists_leftFraction L.op W.op f.op
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.op.LeftFraction { unop := Y } { unop := X }
    eq : Eq f.op (φ.map L.op ⋯)
    ⊢ Exists fun φ => Eq f (φ.map L ⋯)
  -/
  refine ⟨φ.unop, Quiver.Hom.op_inj ?_⟩
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.op.LeftFraction { unop := Y } { unop := X }
    eq : Eq f.op (φ.map L.op ⋯)
    ⊢ Eq f.op (φ.unop.map L ⋯).op
  -/
  rw [eq, MorphismProperty.RightFraction.op_map]
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    φ : W.op.LeftFraction { unop := Y } { unop := X }
    eq : Eq f.op (φ.map L.op ⋯)
    ⊢ Eq (φ.map L.op ⋯) (φ.unop.op.map L.op ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma MorphismProperty.RightFraction.map_eq_iff
    {X Y : C} (φ ψ : W.RightFraction X Y) :
    φ.map L (Localization.inverts _ _) = ψ.map L (Localization.inverts _ _) ↔
      RightFractionRel φ ψ := by
  rw [← leftFractionRel_op_iff, ← LeftFraction.map_eq_iff L.op W.op φ.op ψ.op,
    ← φ.op_map L (Localization.inverts _ _), ← ψ.op_map L (Localization.inverts _ _)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    X Y : C
    φ ψ : W.RightFraction X Y
    ⊢ Iff (Eq (φ.map L ⋯) (ψ.map L ⋯)) (Eq (φ.map L ⋯).op (ψ.map L ⋯).op)
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
      inst✝ : W.HasRightCalculusOfFractions
      X Y : C
      φ ψ : W.RightFraction X Y
      ⊢ Eq (φ.map L ⋯) (ψ.map L ⋯) → Eq (φ.map L ⋯).op (ψ.map L ⋯).op
    -/
  · apply Quiver.Hom.unop_inj
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
      inst✝ : W.HasRightCalculusOfFractions
      X Y : C
      φ ψ : W.RightFraction X Y
      ⊢ Eq (φ.map L ⋯).op (ψ.map L ⋯).op → Eq (φ.map L ⋯) (ψ.map L ⋯)
    -/
  · apply Quiver.Hom.op_inj
    /-
      🎉 no goals
    -/


lemma MorphismProperty.map_eq_iff_precomp {Y Z : C} (f₁ f₂ : Y ⟶ Z) :
    L.map f₁ = L.map f₂ ↔ ∃ (X : C) (s : X ⟶ Y) (_ : W s), s ≫ f₁ = s ≫ f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasRightCalculusOfFractions
    Y Z : C
    f₁ f₂ : Quiver.Hom Y Z
    ⊢ Iff (Eq (L.map f₁) (L.map f₂)) (Exists fun X => Exists fun s => Exists fun x …
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
      inst✝ : W.HasRightCalculusOfFractions
      Y Z : C
      f₁ f₂ : Quiver.Hom Y Z
      ⊢ Eq (L.map f₁) (L.map f₂) → Exists fun X => Exists fun s => Exists fun x => E …
    -/
  · intro h
    rw [← RightFraction.map_ofHom W _ L (Localization.inverts _ _),
      ← RightFraction.map_ofHom W _ L (Localization.inverts _ _),
      RightFraction.map_eq_iff] at h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasRightCalculusOfFractions
      Y Z : C
      f₁ f₂ : Quiver.Hom Y Z
      h : CategoryTheory.MorphismProperty.RightFractionRel (CategoryTheory.MorphismP …
      ⊢ Exists fun X => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
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
      inst✝ : W.HasRightCalculusOfFractions
      Y Z✝ : C
      f₁ f₂ : Quiver.Hom Y Z✝
      Z : C
      t₁ : Quiver.Hom Z (CategoryTheory.MorphismProperty.RightFraction.ofHom W f₁).X'
      t₂ : Quiver.Hom Z (CategoryTheory.MorphismProperty.RightFraction.ofHom W f₂).X'
      hst : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.MorphismProper …
      hft : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.MorphismProper …
      ht : W (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.MorphismProperty …
      ⊢ Exists fun X => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    dsimp at t₁ t₂ hst hft ht
    /-
      case mp.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasRightCalculusOfFractions
      Y Z✝ : C
      f₁ f₂ : Quiver.Hom Y Z✝
      Z : C
      t₁ t₂ : Quiver.Hom Z Y
      hst : Eq (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruct …
      hft : Eq (CategoryTheory.CategoryStruct.comp t₁ f₁) (CategoryTheory.CategorySt …
      ht : W (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruct.i …
      ⊢ Exists fun X => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    simp only [comp_id] at hst
    /-
      case mp.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasRightCalculusOfFractions
      Y Z✝ : C
      f₁ f₂ : Quiver.Hom Y Z✝
      Z : C
      t₁ t₂ : Quiver.Hom Z Y
      hft : Eq (CategoryTheory.CategoryStruct.comp t₁ f₁) (CategoryTheory.CategorySt …
      ht : W (CategoryTheory.CategoryStruct.comp t₁ (CategoryTheory.CategoryStruct.i …
      hst : Eq t₁ t₂
      ⊢ Exists fun X => Exists fun s => Exists fun x => Eq (CategoryTheory.CategoryS …
    -/
    exact ⟨Z, t₁, by simpa using ht, by rw [hft, hst]⟩
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
      inst✝ : W.HasRightCalculusOfFractions
      Y Z : C
      f₁ f₂ : Quiver.Hom Y Z
      ⊢ (Exists fun X => Exists fun s => Exists fun x => Eq (CategoryTheory.Category …
    -/
  · rintro ⟨Z, s, hs, fac⟩
    simp only [← cancel_epi (Localization.isoOfHom L W s hs).hom,
      Localization.isoOfHom_hom, ← L.map_comp, fac]


include W in
lemma Localization.essSurj_mapArrow_of_hasRightCalculusOfFractions :
    L.mapArrow.EssSurj where
  mem_essImage f := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasRightCalculusOfFractions
      f : CategoryTheory.Arrow D
      ⊢ Membership.mem L.mapArrow.essImage f
    -/
    have := Localization.essSurj_mapArrow L.op W.op
    obtain ⟨g, ⟨e⟩⟩ : ∃ (g : _), Nonempty (L.op.mapArrow.obj g ≅ Arrow.mk f.hom.op) :=
      ⟨_, ⟨Functor.objObjPreimageIso _ _⟩⟩
    exact ⟨Arrow.mk g.hom.unop, ⟨Arrow.isoMk (Arrow.rightFunc.mapIso e.symm).unop
      (Arrow.leftFunc.mapIso e.symm).unop (Quiver.Hom.op_inj e.inv.w.symm)⟩⟩


