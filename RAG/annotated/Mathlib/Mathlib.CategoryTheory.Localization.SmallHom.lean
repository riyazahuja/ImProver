/-- This property holds if the type of morphisms between `X` and `Y`
in the localized category with respect to `W : MorphismProperty C`
is small. -/
class HasSmallLocalizedHom : Prop where
  small : Small.{w} (W.Q.obj X ⟶ W.Q.obj Y)


lemma hasSmallLocalizedHom_iff :
    HasSmallLocalizedHom.{w} W X Y ↔ Small.{w} (L.obj X ⟶ L.obj Y) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    X Y : C
    ⊢ Iff (CategoryTheory.Localization.HasSmallLocalizedHom W X Y) (Small.{w, v₂}  …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      W : CategoryTheory.MorphismProperty C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      inst✝ : L.IsLocalization W
      X Y : C
      ⊢ CategoryTheory.Localization.HasSmallLocalizedHom W X Y → Small.{w, v₂} (Quiv …
    -/
  · intro h
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      W : CategoryTheory.MorphismProperty C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      inst✝ : L.IsLocalization W
      X Y : C
      h : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
      ⊢ Small.{w, v₂} (Quiver.Hom (L.obj X) (L.obj Y))
    -/
    have := h.small
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      W : CategoryTheory.MorphismProperty C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      inst✝ : L.IsLocalization W
      X Y : C
      h : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
      this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y))
      ⊢ Small.{w, v₂} (Quiver.Hom (L.obj X) (L.obj Y))
    -/
    exact small_map (homEquiv W W.Q L).symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      W : CategoryTheory.MorphismProperty C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      inst✝ : L.IsLocalization W
      X Y : C
      ⊢ Small.{w, v₂} (Quiver.Hom (L.obj X) (L.obj Y)) → CategoryTheory.Localization …
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      W : CategoryTheory.MorphismProperty C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      inst✝ : L.IsLocalization W
      X Y : C
      h : Small.{w, v₂} (Quiver.Hom (L.obj X) (L.obj Y))
      ⊢ CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    -/
    exact ⟨small_map (homEquiv W W.Q L)⟩
    /-
      🎉 no goals
    -/


include L in
lemma hasSmallLocalizedHom_of_isLocalization :
    HasSmallLocalizedHom.{v₂} W X Y := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    X Y : C
    ⊢ CategoryTheory.Localization.HasSmallLocalizedHom W X Y
  -/
  rw [hasSmallLocalizedHom_iff W L]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝ : L.IsLocalization W
    X Y : C
    ⊢ Small.{v₂, v₂} (Quiver.Hom (L.obj X) (L.obj Y))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


variable (X Y) in
lemma small_of_hasSmallLocalizedHom [HasSmallLocalizedHom.{w} W X Y] :
    Small.{w} (L.obj X ⟶ L.obj Y) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝¹ : L.IsLocalization W
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    ⊢ Small.{w, v₂} (Quiver.Hom (L.obj X) (L.obj Y))
  -/
  rwa [← hasSmallLocalizedHom_iff W]
  /-
    🎉 no goals
  -/


lemma hasSmallLocalizedHom_iff_of_isos {X' Y' : C} (e : X ≅ X') (e' : Y ≅ Y') :
    HasSmallLocalizedHom.{w} W X Y ↔ HasSmallLocalizedHom.{w} W X' Y' := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y X' Y' : C
    e : CategoryTheory.Iso X X'
    e' : CategoryTheory.Iso Y Y'
    ⊢ Iff (CategoryTheory.Localization.HasSmallLocalizedHom W X Y) (CategoryTheory …
  -/
  simp only [hasSmallLocalizedHom_iff W W.Q]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y X' Y' : C
    e : CategoryTheory.Iso X X'
    e' : CategoryTheory.Iso Y Y'
    ⊢ Iff (Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y))) (Small.{w, m …
  -/
  exact small_congr (Iso.homCongr (W.Q.mapIso e) (W.Q.mapIso e'))
  /-
    🎉 no goals
  -/


variable (X) in
lemma hasSmallLocalizedHom_iff_target {Y Y' : C} (f : Y ⟶  Y') (hf : W f):
    HasSmallLocalizedHom.{w} W X Y ↔ HasSmallLocalizedHom.{w} W X Y' := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y Y' : C
    f : Quiver.Hom Y Y'
    hf : W f
    ⊢ Iff (CategoryTheory.Localization.HasSmallLocalizedHom W X Y) (CategoryTheory …
  -/
  simp only [hasSmallLocalizedHom_iff W W.Q]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y Y' : C
    f : Quiver.Hom Y Y'
    hf : W f
    ⊢ Iff (Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y))) (Small.{w, m …
  -/
  exact small_congr (Iso.homCongr (Iso.refl _) (Localization.isoOfHom W.Q W f hf))
  /-
    🎉 no goals
  -/


lemma hasSmallLocalizedHom_iff_source {X' : C} (f : X ⟶  X') (hf : W f) (Y : C) :
    HasSmallLocalizedHom.{w} W X Y ↔ HasSmallLocalizedHom.{w} W X' Y := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X X' : C
    f : Quiver.Hom X X'
    hf : W f
    Y : C
    ⊢ Iff (CategoryTheory.Localization.HasSmallLocalizedHom W X Y) (CategoryTheory …
  -/
  simp only [hasSmallLocalizedHom_iff W W.Q]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X X' : C
    f : Quiver.Hom X X'
    hf : W f
    Y : C
    ⊢ Iff (Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y))) (Small.{w, m …
  -/
  exact small_congr (Iso.homCongr (Localization.isoOfHom W.Q W f hf) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- The type of morphisms from `X` to `Y` in the localized category
with respect to `W : MorphismProperty C` that is shrunk to `Type w`
when `HasSmallLocalizedHom.{w} W X Y` holds. -/
def SmallHom (X Y : C) [HasSmallLocalizedHom.{w} W X Y] : Type w :=
  Shrink.{w} (W.Q.obj X ⟶ W.Q.obj Y)


/-- The canonical bijection `SmallHom.{w} W X Y ≃ (L.obj X ⟶ L.obj Y)`
when `L` is a localization functor for `W : MorphismProperty C` and
that `HasSmallLocalizedHom.{w} W X Y` holds. -/
noncomputable def equiv (L : C ⥤ D) [L.IsLocalization W] {X Y : C}
    [HasSmallLocalizedHom.{w} W X Y] :
    SmallHom.{w} W X Y ≃ (L.obj X ⟶ L.obj Y) :=
  letI := small_of_hasSmallLocalizedHom.{w} W W.Q X Y
  (equivShrink _).symm.trans (homEquiv W W.Q L)


lemma equiv_equiv_symm (L : C ⥤ D) [L.IsLocalization W]
    (L' : C ⥤ D') [L'.IsLocalization W] (G : D ⥤ D')
    (e : L ⋙ G ≅ L') {X Y : C} [HasSmallLocalizedHom.{w} W X Y]
    (f : L.obj X ⟶ L.obj Y) :
    equiv W L' ((equiv W L).symm f) =
      e.inv.app X ≫ G.map f ≫ e.hom.app Y := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    D' : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} D'
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    L' : CategoryTheory.Functor C D'
    inst✝¹ : L'.IsLocalization W
    G : CategoryTheory.Functor D D'
    e : CategoryTheory.Iso (L.comp G) L'
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L') ((CategoryTheory.Local …
  -/
  dsimp [equiv]
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    D' : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} D'
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    L' : CategoryTheory.Functor C D'
    inst✝¹ : L'.IsLocalization W
    G : CategoryTheory.Functor D D'
    e : CategoryTheory.Iso (L.comp G) L'
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W W.Q L') ((equivShrink (Quiver.Ho …
  -/
  rw [Equiv.symm_apply_apply, homEquiv_trans]
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    D' : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} D'
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    L' : CategoryTheory.Functor C D'
    inst✝¹ : L'.IsLocalization W
    G : CategoryTheory.Functor D D'
    e : CategoryTheory.Iso (L.comp G) L'
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L L') f) (CategoryTheory.Categor …
  -/
  apply homEquiv_eq
  /-
    🎉 no goals
  -/


/-- The element in `SmallHom W X Y` induced by `f : X ⟶ Y`. -/
noncomputable def mk {X Y : C} [HasSmallLocalizedHom.{w} W X Y] (f : X ⟶ Y) :
    SmallHom.{w} W X Y :=
  (equiv.{w} W W.Q).symm (W.Q.map f)


@[simp]
lemma equiv_mk (L : C ⥤ D) [L.IsLocalization W] {X Y : C}
    [HasSmallLocalizedHom.{w} W X Y] (f : X ⟶ Y) :
    equiv.{w} W L (mk W f) = L.map f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝¹ : L.IsLocalization W
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  simp [equiv, mk]
  /-
    🎉 no goals
  -/


/-- The formal inverse in `SmallHom W X Y` of a morphism `f : Y ⟶ X` such that `W f`. -/
noncomputable def mkInv {X Y : C} (f : Y ⟶ X) (hf : W f) [HasSmallLocalizedHom.{w} W X Y] :
    SmallHom.{w} W X Y :=
  (equiv.{w} W W.Q).symm (Localization.isoOfHom W.Q W f hf).inv


@[simp]
lemma equiv_mkInv (L : C ⥤ D) [L.IsLocalization W] {X Y : C} (f : Y ⟶ X) (hf : W f)
    [HasSmallLocalizedHom.{w} W X Y] :
    equiv.{w} W L (mkInv f hf) = (Localization.isoOfHom L W f hf).inv := by
  simp only [equiv, mkInv, Equiv.symm_trans_apply, Equiv.symm_symm, homEquiv_symm_apply,
    Equiv.trans_apply, Equiv.symm_apply_apply, homEquiv_isoOfHom_inv]


/-- The composition on `SmallHom W`. -/
noncomputable def comp {X Y Z : C} [HasSmallLocalizedHom.{w} W X Y]
    [HasSmallLocalizedHom.{w} W Y Z] [HasSmallLocalizedHom.{w} W X Z]
    (α : SmallHom.{w} W X Y) (β : SmallHom.{w} W Y Z) :
    SmallHom.{w} W X Z :=
  (equiv W W.Q).symm (equiv W W.Q α ≫ equiv W W.Q β)


lemma equiv_comp (L : C ⥤ D) [L.IsLocalization W] {X Y Z : C} [HasSmallLocalizedHom.{w} W X Y]
    [HasSmallLocalizedHom.{w} W Y Z] [HasSmallLocalizedHom.{w} W X Z]
    (α : SmallHom.{w} W X Y) (β : SmallHom.{w} W Y Z) :
    equiv W L (α.comp β) = equiv W L α ≫ equiv W L β := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    α : CategoryTheory.Localization.SmallHom W X Y
    β : CategoryTheory.Localization.SmallHom W Y Z
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (α.comp β)) (CategoryTh …
  -/
  letI := small_of_hasSmallLocalizedHom.{w} W W.Q X Y
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    α : CategoryTheory.Localization.SmallHom W X Y
    β : CategoryTheory.Localization.SmallHom W Y Z
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryTh …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (α.comp β)) (CategoryTh …
  -/
  letI := small_of_hasSmallLocalizedHom.{w} W W.Q Y Z
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    α : CategoryTheory.Localization.SmallHom W X Y
    β : CategoryTheory.Localization.SmallHom W Y Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (α.comp β)) (CategoryTh …
  -/
  obtain ⟨α, rfl⟩ := (equivShrink _).surjective α
  /-
    case intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    β : CategoryTheory.Localization.SmallHom W Y Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    α : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  obtain ⟨β, rfl⟩ := (equivShrink _).surjective β
  /-
    case intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    α : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    β : Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  dsimp [equiv, comp]
  /-
    case intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    α : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    β : Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W W.Q L) ((equivShrink (Quiver.Hom …
  -/
  rw [Equiv.symm_apply_apply]
  /-
    case intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    α : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    β : Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W W.Q L) ((CategoryTheory.Localiza …
  -/
  erw [(equivShrink _).symm_apply_apply, (equivShrink _).symm_apply_apply]
  /-
    case intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    X Y Z : C
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    this✝ : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj X) (W.Q.obj Y)) := CategoryT …
    this : Small.{w, max u₁ v₁} (Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)) := CategoryTh …
    α : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    β : Quiver.Hom (W.Q.obj Y) (W.Q.obj Z)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W W.Q L) ((CategoryTheory.Localiza …
  -/
  simp only [homEquiv_refl, homEquiv_comp]
  /-
    🎉 no goals
  -/


lemma mk_comp_mk [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w} W Y Z]
    [HasSmallLocalizedHom.{w} W X Z] (f : X ⟶ Y) (g : Y ⟶ Z) :
    (mk W f).comp (mk W g) = mk W (f ≫ g) :=
                              /-
                                C : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                W : CategoryTheory.MorphismProperty C
                                X Y Z : C
                                inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
                                inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
                                inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
                                f : Quiver.Hom X Y
                                g : Quiver.Hom Y Z
                                ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) ((CategoryTheory.Loca …
                              -/
  (equiv W W.Q).injective (by simp [equiv_comp])
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma comp_mk_id [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w} W Y Y]
    (α : SmallHom.{w} W X Y)  :
    α.comp (mk W (𝟙 Y)) = α :=
                              /-
                                C : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                W : CategoryTheory.MorphismProperty C
                                X Y : C
                                inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
                                inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Y
                                α : CategoryTheory.Localization.SmallHom W X Y
                                ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) (α.comp (CategoryTheo …
                              -/
  (equiv W W.Q).injective (by simp [equiv_comp])
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma mk_id_comp [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w} W X X]
    (α : SmallHom.{w} W X Y) :
    (mk W (𝟙 X)).comp α = α :=
                              /-
                                C : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                W : CategoryTheory.MorphismProperty C
                                X Y : C
                                inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
                                inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X X
                                α : CategoryTheory.Localization.SmallHom W X Y
                                ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) ((CategoryTheory.Loca …
                              -/
  (equiv W W.Q).injective (by simp [equiv_comp])
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma comp_assoc [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w} W X Z]
    [HasSmallLocalizedHom.{w} W X T] [HasSmallLocalizedHom.{w} W Y Z]
    [HasSmallLocalizedHom.{w} W Y T] [HasSmallLocalizedHom.{w} W Z T]
    (α : SmallHom.{w} W X Y) (β : SmallHom.{w} W Y Z) (γ : SmallHom.{w} W Z T) :
    (α.comp β).comp γ = α.comp (β.comp γ) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y Z T : C
    inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedHom W X T
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y T
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W Z T
    α : CategoryTheory.Localization.SmallHom W X Y
    β : CategoryTheory.Localization.SmallHom W Y Z
    γ : CategoryTheory.Localization.SmallHom W Z T
    ⊢ Eq ((α.comp β).comp γ) (α.comp (β.comp γ))
  -/
  apply (equiv W W.Q).injective
  /-
    case a
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    X Y Z T : C
    inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedHom W X Z
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedHom W X T
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W Y Z
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y T
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W Z T
    α : CategoryTheory.Localization.SmallHom W X Y
    β : CategoryTheory.Localization.SmallHom W Y Z
    γ : CategoryTheory.Localization.SmallHom W Z T
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) ((α.comp β).comp γ))  …
  -/
  simp only [equiv_comp, assoc]
  /-
    🎉 no goals
  -/


@[simp]
lemma mk_comp_mkInv [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w} W Y X]
    [HasSmallLocalizedHom.{w} W Y Y] (f : Y ⟶ X) (hf : W f) :
    (mk W f).comp (mkInv f hf) = mk W (𝟙 Y) :=
                              /-
                                C : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                W : CategoryTheory.MorphismProperty C
                                X Y : C
                                inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
                                inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W Y X
                                inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W Y Y
                                f : Quiver.Hom Y X
                                hf : W f
                                ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) ((CategoryTheory.Loca …
                              -/
  (equiv W W.Q).injective (by simp [equiv_comp])
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma mkInv_comp_mk [HasSmallLocalizedHom.{w} W X X] [HasSmallLocalizedHom.{w} W X Y]
    [HasSmallLocalizedHom.{w} W Y X] (f : Y ⟶ X) (hf : W f) :
    (mkInv f hf).comp (mk W f) = mk W (𝟙 X) :=
                              /-
                                C : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                W : CategoryTheory.MorphismProperty C
                                X Y : C
                                inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W X X
                                inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
                                inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W Y X
                                f : Quiver.Hom Y X
                                hf : W f
                                ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W W.Q) ((CategoryTheory.Loca …
                              -/
  (equiv W W.Q).injective (by simp [equiv_comp])
                              /-
                                🎉 no goals
                              -/


/-- Up to an equivalence, the type `SmallHom.{w} W X Y n` does not depend on the universe `w`. -/
noncomputable def chgUniv {X Y : C}
    [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w''} W X Y] :
    SmallHom.{w} W X Y ≃ SmallHom.{w''} W X Y :=
  (equiv.{w} W W.Q).trans (equiv.{w''} W W.Q).symm


lemma equiv_chgUniv (L : C ⥤ D) [L.IsLocalization W] {X Y : C}
    [HasSmallLocalizedHom.{w} W X Y] [HasSmallLocalizedHom.{w''} W X Y]
    (e : SmallHom.{w} W X Y) :
    equiv W L (chgUniv.{w''} e) = equiv W L e := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    X Y : C
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    e : CategoryTheory.Localization.SmallHom W X Y
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  obtain ⟨f, rfl⟩ := (equiv W W.Q).symm.surjective e
  /-
    case intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    X Y : C
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W X Y
    f : Quiver.Hom (W.Q.obj X) (W.Q.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  dsimp [chgUniv]
  simp only [Equiv.apply_symm_apply,
    equiv_equiv_symm W _ _ _ (Localization.compUniqFunctor W.Q L W)]


/-- The action of a localizer morphism on `SmallHom`. -/
noncomputable def smallHomMap (f : SmallHom.{w} W₁ X Y) :
    SmallHom.{w'} W₂ (Φ.functor.obj X) (Φ.functor.obj Y) :=
  (SmallHom.equiv W₂ W₂.Q).symm
    (Iso.homCongr ((CatCommSq.iso Φ.functor W₁.Q W₂.Q _).symm.app _)
      ((CatCommSq.iso Φ.functor W₁.Q W₂.Q _).symm.app _)
      ((Φ.localizedFunctor W₁.Q W₂.Q).map ((SmallHom.equiv W₁ W₁.Q) f)))


lemma equiv_smallHomMap (G : D₁ ⥤ D₂) (e : Φ.functor ⋙ L₂ ≅ L₁ ⋙ G)
    (f : SmallHom.{w} W₁ X Y) :
    (SmallHom.equiv W₂ L₂) (Φ.smallHomMap f) =
      e.hom.app X ≫ G.map (SmallHom.equiv W₁ L₁ f) ≫ e.inv.app Y := by
  /-
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : CategoryTheory.Localization.SmallHom W₁ X Y
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) (Φ.smallHomMap f)) (C …
  -/
  obtain ⟨g, rfl⟩ := (SmallHom.equiv W₁ W₁.Q).symm.surjective f
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) (Φ.smallHomMap ((Cate …
  -/
  simp only [smallHomMap, Equiv.apply_symm_apply]
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let G' := Φ.localizedFunctor W₁.Q W₂.Q
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let β := CatCommSq.iso Φ.functor W₁.Q W₂.Q G'
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let E₁ := (uniq W₁.Q L₁ W₁).functor
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let α₁ : W₁.Q ⋙ E₁ ≅ L₁ := compUniqFunctor W₁.Q L₁ W₁
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    α₁ : CategoryTheory.Iso (W₁.Q.comp E₁) L₁ := CategoryTheory.Localization.compU …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let E₂ := (uniq W₂.Q L₂ W₂).functor
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    α₁ : CategoryTheory.Iso (W₁.Q.comp E₁) L₁ := CategoryTheory.Localization.compU …
    E₂ : CategoryTheory.Functor W₂.Localization D₂ := (CategoryTheory.Localization …
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W₂ L₂) ((CategoryTheory.Loca …
  -/
  let α₂ : W₂.Q ⋙ E₂ ≅ L₂ := compUniqFunctor W₂.Q L₂ W₂
  rw [SmallHom.equiv_equiv_symm W₁ W₁.Q L₁ E₁ α₁,
    SmallHom.equiv_equiv_symm W₂ W₂.Q L₂ E₂ α₂]
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    α₁ : CategoryTheory.Iso (W₁.Q.comp E₁) L₁ := CategoryTheory.Localization.compU …
    E₂ : CategoryTheory.Functor W₂.Localization D₂ := (CategoryTheory.Localization …
    α₂ : CategoryTheory.Iso (W₂.Q.comp E₂) L₂ := CategoryTheory.Localization.compU …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α₂.inv.app (Φ.functor.obj X)) (Categ …
  -/
  change α₂.inv.app _ ≫ E₂.map (β.hom.app X ≫ G'.map g ≫ β.inv.app Y) ≫ _ = _
  let γ : G' ⋙ E₂ ≅ E₁ ⋙ G := liftNatIso W₁.Q W₁ (W₁.Q ⋙ G' ⋙ E₂) (W₁.Q ⋙ E₁ ⋙ G) _ _
    ((Functor.associator _ _ _).symm ≪≫ isoWhiskerRight β.symm E₂ ≪≫
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ α₂ ≪≫ e ≪≫
      isoWhiskerRight α₁.symm G ≪≫ Functor.associator _ _ _)
  have hγ : ∀ (X : C₁), γ.hom.app (W₁.Q.obj X) =
      E₂.map (β.inv.app X) ≫ α₂.hom.app (Φ.functor.obj X) ≫
        e.hom.app X ≫ G.map (α₁.inv.app X) := fun X ↦ by
    simp [γ, id_comp, comp_id]
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    α₁ : CategoryTheory.Iso (W₁.Q.comp E₁) L₁ := CategoryTheory.Localization.compU …
    E₂ : CategoryTheory.Functor W₂.Localization D₂ := (CategoryTheory.Localization …
    α₂ : CategoryTheory.Iso (W₂.Q.comp E₂) L₂ := CategoryTheory.Localization.compU …
    γ : CategoryTheory.Iso (G'.comp E₂) (E₁.comp G) := CategoryTheory.Localization …
    hγ : ∀ (X : C₁), Eq (γ.hom.app (W₁.Q.obj X)) (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α₂.inv.app (Φ.functor.obj X)) (Categ …
  -/
  simp only [Functor.map_comp, assoc]
  /-
    case intro
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    D₁ : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    D₂ : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₂.IsLocalization W₂
    X Y : C₁
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    g : Quiver.Hom (W₁.Q.obj X) (W₁.Q.obj Y)
    G' : CategoryTheory.Functor W₁.Localization W₂.Localization := Φ.localizedFunc …
    β : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp G') := CategoryTheory. …
    E₁ : CategoryTheory.Functor W₁.Localization D₁ := (CategoryTheory.Localization …
    α₁ : CategoryTheory.Iso (W₁.Q.comp E₁) L₁ := CategoryTheory.Localization.compU …
    E₂ : CategoryTheory.Functor W₂.Localization D₂ := (CategoryTheory.Localization …
    α₂ : CategoryTheory.Iso (W₂.Q.comp E₂) L₂ := CategoryTheory.Localization.compU …
    γ : CategoryTheory.Iso (G'.comp E₂) (E₁.comp G) := CategoryTheory.Localization …
    hγ : ∀ (X : C₁), Eq (γ.hom.app (W₁.Q.obj X)) (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α₂.inv.app (Φ.functor.obj X)) (Categ …
  -/
  erw [← NatIso.naturality_1 γ]
  simp only [Functor.comp_map, ← cancel_epi (e.inv.app X), ← cancel_epi (G.map (α₁.hom.app X)),
    ← cancel_epi (γ.hom.app (W₁.Q.obj X)), assoc, Iso.inv_hom_id_app_assoc,
    ← Functor.map_comp_assoc, Iso.hom_inv_id_app, Functor.map_id, id_comp,
    Iso.hom_inv_id_app_assoc]
  simp only [hγ, assoc, ← Functor.map_comp_assoc, Iso.inv_hom_id_app,
    Functor.map_id, id_comp, Iso.hom_inv_id_app_assoc,
    Iso.inv_hom_id_app_assoc, Iso.hom_inv_id_app, Functor.comp_obj, comp_id]


lemma smallHomMap_comp (f : SmallHom.{w} W₁ X Y) (g : SmallHom.{w} W₁ Y Z) :
    Φ.smallHomMap (f.comp g) = (Φ.smallHomMap f).comp (Φ.smallHomMap g) := by
  /-
    C₁ : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    W₁ : CategoryTheory.MorphismProperty C₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    X Y Z : C₁
    inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Y
    inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ Y Z
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedHom W₁ X Z
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X) …
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj Y) …
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedHom W₂ (Φ.functor.obj X)  …
    f : CategoryTheory.Localization.SmallHom W₁ X Y
    g : CategoryTheory.Localization.SmallHom W₁ Y Z
    ⊢ Eq (Φ.smallHomMap (f.comp g)) ((Φ.smallHomMap f).comp (Φ.smallHomMap g))
  -/
  apply (SmallHom.equiv W₂ W₂.Q).injective
  simp [Φ.equiv_smallHomMap W₁.Q W₂.Q (Φ.localizedFunctor W₁.Q W₂.Q) (CatCommSq.iso _ _ _ _),
    SmallHom.equiv_comp]


