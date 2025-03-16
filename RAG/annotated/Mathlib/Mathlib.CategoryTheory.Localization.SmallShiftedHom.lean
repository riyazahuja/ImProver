/-- Given objects `X` and `Y` in a category `C`, this is the property that
all the types of morphisms from `X⟦a⟧` to `Y⟦b⟧` are `w`-small
in the localized category with respect to a class of morphisms `W`. -/
abbrev HasSmallLocalizedShiftedHom : Prop :=
  ∀ (a b : M), HasSmallLocalizedHom.{w} W (X⟦a⟧) (Y⟦b⟧)


lemma hasSmallLocalizedShiftedHom_iff
    (L : C ⥤ D) [L.IsLocalization W] [L.CommShift M] (X Y : C) :
    HasSmallLocalizedShiftedHom.{w} W M X Y ↔
      ∀ (a b : M), Small.{w} ((L.obj X)⟦a⟧ ⟶ (L.obj Y)⟦b⟧) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁴ : AddMonoid M
    inst✝³ : CategoryTheory.HasShift C M
    inst✝² : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝¹ : L.IsLocalization W
    inst✝ : L.CommShift M
    X Y : C
    ⊢ Iff (CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y) (∀ (a  …
  -/
  dsimp [HasSmallLocalizedShiftedHom]
  have eq := fun (a b : M) ↦ small_congr.{w}
    (Iso.homCongr ((L.commShiftIso a).app X) ((L.commShiftIso b).app Y))
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁴ : AddMonoid M
    inst✝³ : CategoryTheory.HasShift C M
    inst✝² : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝¹ : L.IsLocalization W
    inst✝ : L.CommShift M
    X Y : C
    eq : ∀ (a b : M), Iff (Small.{w, v₂} (Quiver.Hom (((CategoryTheory.shiftFuncto …
    ⊢ Iff (∀ (a b : M), CategoryTheory.Localization.HasSmallLocalizedHom W ((Categ …
  -/
  dsimp at eq
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁴ : AddMonoid M
    inst✝³ : CategoryTheory.HasShift C M
    inst✝² : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝¹ : L.IsLocalization W
    inst✝ : L.CommShift M
    X Y : C
    eq : ∀ (a b : M), Iff (Small.{w, v₂} (Quiver.Hom (L.obj ((CategoryTheory.shift …
    ⊢ Iff (∀ (a b : M), CategoryTheory.Localization.HasSmallLocalizedHom W ((Categ …
  -/
  simp only [hasSmallLocalizedHom_iff _ L, eq]
  /-
    🎉 no goals
  -/


variable {Y} in
lemma hasSmallLocalizedShiftedHom_iff_target [W.IsCompatibleWithShift M]
    {Y' : C} (f : Y ⟶  Y') (hf : W f) :
    HasSmallLocalizedShiftedHom.{w} W M X Y ↔ HasSmallLocalizedShiftedHom.{w} W M X Y' :=
  forall_congr' (fun a ↦ forall_congr' (fun b ↦
    hasSmallLocalizedHom_iff_target W (X⟦a⟧) (f⟦b⟧') (W.shift hf b)))


variable {X} in
lemma hasSmallLocalizedShiftedHom_iff_source [W.IsCompatibleWithShift M]
    {X' : C} (f : X ⟶  X') (hf : W f) (Y : C) :
    HasSmallLocalizedShiftedHom.{w} W M X Y ↔ HasSmallLocalizedShiftedHom.{w} W M X' Y :=
  forall_congr' (fun a ↦ forall_congr' (fun b ↦
    hasSmallLocalizedHom_iff_source W (f⟦a⟧') (W.shift hf a) (Y⟦b⟧)))


include M in
lemma hasSmallLocalizedHom_of_hasSmallLocalizedShiftedHom₀ :
    HasSmallLocalizedHom.{w} W X Y :=
  (hasSmallLocalizedHom_iff_of_isos W
    ((shiftFunctorZero C M).app X) ((shiftFunctorZero C M).app Y)).1 inferInstance


instance (m : M) : HasSmallLocalizedHom.{w} W X (Y⟦m⟧) :=
  (hasSmallLocalizedHom_iff_of_isos W
    ((shiftFunctorZero C M).app X) (Iso.refl (Y⟦m⟧))).1 inferInstance


instance (m : M) : HasSmallLocalizedHom.{w} W (X⟦m⟧) Y :=
  (hasSmallLocalizedHom_iff_of_isos W
    (Iso.refl (X⟦m⟧)) ((shiftFunctorZero C M).app Y)).1 inferInstance


instance (m m' n : M) : HasSmallLocalizedHom.{w} W (X⟦m⟧⟦m'⟧) (Y⟦n⟧) :=
  (hasSmallLocalizedHom_iff_of_isos W
    ((shiftFunctorAdd C m m').app X) (Iso.refl (Y⟦n⟧))).1 inferInstance


instance (m n n' : M) : HasSmallLocalizedHom.{w} W (X⟦m⟧) (Y⟦n⟧⟦n'⟧) :=
  (hasSmallLocalizedHom_iff_of_isos W
    (Iso.refl (X⟦m⟧)) ((shiftFunctorAdd C n n').app Y)).1 inferInstance


/-- Given `f : SmallHom W X Y` and `a : M`, this is the element
in `SmallHom W (X⟦a⟧) (Y⟦a⟧)` obtained by shifting by `a`. -/
noncomputable def shift : SmallHom.{w} W (X⟦a⟧) (Y⟦a⟧) :=
  (W.shiftLocalizerMorphism a).smallHomMap f


lemma equiv_shift : equiv W L (f.shift a) =
    (L.commShiftIso a).hom.app X ≫ (equiv W L f)⟦a⟧' ≫ (L.commShiftIso a).inv.app Y :=
  (W.shiftLocalizerMorphism a).equiv_smallHomMap _ _ _ (L.commShiftIso a) f


/-- The type of morphisms from `X` to `Y⟦m⟧` in the localized category
with respect to `W : MorphismProperty C` that is shrunk to `Type w`
when `HasSmallLocalizedShiftedHom.{w} W X Y` holds. -/
def SmallShiftedHom (X Y : C) [HasSmallLocalizedShiftedHom.{w} W M X Y] (m : M) : Type w :=
  SmallHom W X (Y⟦m⟧)


/-- Given `f : SmallShiftedHom.{w} W X Y a`, this is the element in
`SmallHom.{w} W (X⟦n⟧) (Y⟦a'⟧)` that is obtained by shifting by `n`
when `a + n = a'`. -/
noncomputable def shift {a : M} [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w} W M Y Y]
  (f : SmallShiftedHom.{w} W X Y a) (n a' : M) (h : a + n = a') :
    SmallHom.{w} W (X⟦n⟧) (Y⟦a'⟧) :=
  (SmallHom.shift f n).comp (SmallHom.mk W ((shiftFunctorAdd' C a n a' h).inv.app Y))


/-- The composition on `SmallShiftedHom W`. -/
noncomputable def comp {a b c : M} [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w} W M Y Z] [HasSmallLocalizedShiftedHom.{w} W M X Z]
    [HasSmallLocalizedShiftedHom.{w} W M Z Z]
    (f : SmallShiftedHom.{w} W X Y a) (g : SmallShiftedHom.{w} W Y Z b) (h : b + a = c) :
    SmallShiftedHom.{w} W X Z c :=
  SmallHom.comp f (g.shift a c h)


variable (W) in
/-- The canonical map `(X ⟶ Y) → SmallShiftedHom.{w} W X Y m₀` when `m₀ = 0` and
`[HasSmallLocalizedShiftedHom.{w} W M X Y]` holds. -/
noncomputable def mk₀ [HasSmallLocalizedShiftedHom.{w} W M X Y]
    (m₀ : M) (hm₀ : m₀ = 0) (f : X ⟶ Y) :
    SmallShiftedHom.{w} W X Y m₀ :=
  SmallHom.mk W (f ≫ (shiftFunctorZero' C m₀ hm₀).inv.app Y)


/-- The bijection `SmallShiftedHom.{w} W X Y m ≃ ShiftedHom (L.obj X) (L.obj Y) m`
for all `m : M`, and `X` and `Y` in `C` when `L : C ⥤ D` is a localization functor for
`W : MorphismProperty C` such that the category `D` is equipped with a shift by `M`
and `L` commutes with the shifts. -/
noncomputable def equiv [HasSmallLocalizedShiftedHom.{w} W M X Y] {m : M} :
    SmallShiftedHom.{w} W X Y m ≃ ShiftedHom (L.obj X) (L.obj Y) m :=
  (SmallHom.equiv W L).trans ((L.commShiftIso m).app Y).homToEquiv


lemma equiv_shift' {a : M} [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w} W M Y Y]
    (f : SmallShiftedHom.{w} W X Y a) (n a' : M) (h : a + n = a') :
    SmallHom.equiv W L (f.shift n a' h) = (L.commShiftIso n).hom.app X ≫
      (SmallHom.equiv W L f)⟦n⟧' ≫ ((L.commShiftIso a).hom.app Y)⟦n⟧' ≫
        (shiftFunctorAdd' D a n a' h).inv.app (L.obj Y) ≫ (L.commShiftIso a').inv.app Y := by
  simp only [shift, SmallHom.equiv_comp, SmallHom.equiv_shift, SmallHom.equiv_mk, assoc,
    L.commShiftIso_add' h, Functor.CommShift.isoAdd'_inv_app, Iso.inv_hom_id_app_assoc,
    ← Functor.map_comp_assoc, Iso.hom_inv_id_app, Functor.comp_obj, comp_id]


lemma equiv_shift {a : M} [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w} W M Y Y]
    (f : SmallShiftedHom.{w} W X Y a) (n a' : M) (h : a + n = a') :
    equiv W L (f.shift n a' h) = (L.commShiftIso n).hom.app X ≫ (equiv W L f)⟦n⟧' ≫
      (shiftFunctorAdd' D a n a' h).inv.app (L.obj Y) := by
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁷ : AddMonoid M
    inst✝⁶ : CategoryTheory.HasShift C M
    inst✝⁵ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁴ : L.IsLocalization W
    inst✝³ : L.CommShift M
    X Y : C
    inst✝² : W.IsCompatibleWithShift M
    a : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Y
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    n a' : M
    h : Eq (HAdd.hAdd a n) a'
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W L) (f.shift n a' h) …
  -/
  dsimp [equiv]
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁷ : AddMonoid M
    inst✝⁶ : CategoryTheory.HasShift C M
    inst✝⁵ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁴ : L.IsLocalization W
    inst✝³ : L.CommShift M
    X Y : C
    inst✝² : W.IsCompatibleWithShift M
    a : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Y
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    n a' : M
    h : Eq (HAdd.hAdd a n) a'
    ⊢ Eq (((L.commShiftIso a').app Y).homToEquiv ((CategoryTheory.Localization.Sma …
  -/
  erw [Iso.homToEquiv_apply, Iso.homToEquiv_apply, equiv_shift']
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁷ : AddMonoid M
    inst✝⁶ : CategoryTheory.HasShift C M
    inst✝⁵ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁴ : L.IsLocalization W
    inst✝³ : L.CommShift M
    X Y : C
    inst✝² : W.IsCompatibleWithShift M
    a : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Y
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    n a' : M
    h : Eq (HAdd.hAdd a n) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Functor.comp_obj, Iso.app_hom, assoc, Iso.inv_hom_id_app, comp_id, Functor.map_comp]
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁷ : AddMonoid M
    inst✝⁶ : CategoryTheory.HasShift C M
    inst✝⁵ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁴ : L.IsLocalization W
    inst✝³ : L.CommShift M
    X Y : C
    inst✝² : W.IsCompatibleWithShift M
    a : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Y
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    n a' : M
    h : Eq (HAdd.hAdd a n) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.commShiftIso n).hom.app X) (Categ …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma equiv_comp [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w} W M Y Z] [HasSmallLocalizedShiftedHom.{w} W M X Z]
    [HasSmallLocalizedShiftedHom.{w} W M Z Z] {a b c : M}
    (f : SmallShiftedHom.{w} W X Y a) (g : SmallShiftedHom.{w} W Y Z b) (h : b + a = c) :
    equiv W L (f.comp g h) = (equiv W L f).comp (equiv W L g) h := by
  /-
    C : Type u₁
    inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁹ : AddMonoid M
    inst✝⁸ : CategoryTheory.HasShift C M
    inst✝⁷ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁶ : L.IsLocalization W
    inst✝⁵ : L.CommShift M
    X Y Z : C
    inst✝⁴ : W.IsCompatibleWithShift M
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
    a b c : M
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    g : CategoryTheory.Localization.SmallShiftedHom W Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W L) (f.comp g h)) (( …
  -/
  dsimp [comp, equiv, ShiftedHom.comp]
  /-
    C : Type u₁
    inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁹ : AddMonoid M
    inst✝⁸ : CategoryTheory.HasShift C M
    inst✝⁷ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁶ : L.IsLocalization W
    inst✝⁵ : L.CommShift M
    X Y Z : C
    inst✝⁴ : W.IsCompatibleWithShift M
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
    a b c : M
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    g : CategoryTheory.Localization.SmallShiftedHom W Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (((L.commShiftIso c).app Z).homToEquiv ((CategoryTheory.Localization.Smal …
  -/
  erw [Iso.homToEquiv_apply, Iso.homToEquiv_apply, Iso.homToEquiv_apply, SmallHom.equiv_comp]
  simp only [equiv_shift', Functor.comp_obj, Iso.app_hom, assoc, Iso.inv_hom_id_app,
    comp_id, Functor.map_comp]
  /-
    C : Type u₁
    inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁹ : AddMonoid M
    inst✝⁸ : CategoryTheory.HasShift C M
    inst✝⁷ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝⁶ : L.IsLocalization W
    inst✝⁵ : L.CommShift M
    X Y Z : C
    inst✝⁴ : W.IsCompatibleWithShift M
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
    a b c : M
    f : CategoryTheory.Localization.SmallShiftedHom W X Y a
    g : CategoryTheory.Localization.SmallShiftedHom W Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.SmallHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma equiv_mk₀ [HasSmallLocalizedShiftedHom.{w} W M X Y]
    (m₀ : M) (hm₀ : m₀ = 0) (f : X ⟶ Y) :
    equiv W L (SmallShiftedHom.mk₀ W m₀ hm₀ f) =
      ShiftedHom.mk₀ m₀ hm₀ (L.map f) := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁵ : AddMonoid M
    inst✝⁴ : CategoryTheory.HasShift C M
    inst✝³ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    inst✝¹ : L.CommShift M
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    m₀ : M
    hm₀ : Eq m₀ 0
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W L) (CategoryTheory. …
  -/
  subst hm₀
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁵ : AddMonoid M
    inst✝⁴ : CategoryTheory.HasShift C M
    inst✝³ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    inst✝¹ : L.CommShift M
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W L) (CategoryTheory. …
  -/
  dsimp [equiv, mk₀]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁵ : AddMonoid M
    inst✝⁴ : CategoryTheory.HasShift C M
    inst✝³ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    inst✝¹ : L.CommShift M
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    f : Quiver.Hom X Y
    ⊢ Eq (((L.commShiftIso 0).app Y).homToEquiv ((CategoryTheory.Localization.Smal …
  -/
  erw [SmallHom.equiv_mk, Iso.homToEquiv_apply, Functor.map_comp]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁵ : AddMonoid M
    inst✝⁴ : CategoryTheory.HasShift C M
    inst✝³ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝² : L.IsLocalization W
    inst✝¹ : L.CommShift M
    X Y : C
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [equiv, mk₀, ShiftedHom.mk₀, shiftFunctorZero']
  simp only [comp_id, L.commShiftIso_zero, Functor.CommShift.isoZero_hom_app, assoc,
    ← Functor.map_comp_assoc, Iso.inv_hom_id_app, Functor.id_obj, Functor.map_id, id_comp]


lemma comp_assoc {X Y Z T : C} {a₁ a₂ a₃ a₁₂ a₂₃ a : M}
    [HasSmallLocalizedShiftedHom.{w} W M X Y] [HasSmallLocalizedShiftedHom.{w} W M X Z]
    [HasSmallLocalizedShiftedHom.{w} W M X T] [HasSmallLocalizedShiftedHom.{w} W M Y Z]
    [HasSmallLocalizedShiftedHom.{w} W M Y T] [HasSmallLocalizedShiftedHom.{w} W M Z T]
    [HasSmallLocalizedShiftedHom.{w} W M Z Z] [HasSmallLocalizedShiftedHom.{w} W M T T]
    (α : SmallShiftedHom.{w} W X Y a₁) (β : SmallShiftedHom.{w} W Y Z a₂)
    (γ : SmallShiftedHom.{w} W Z T a₃)
    (h₁₂ : a₂ + a₁ = a₁₂) (h₂₃ : a₃ + a₂ = a₂₃) (h : a₃ + a₂ + a₁ = a) :
                                                /-
                                                  C : Type u₁
                                                  inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
                                                  D : Type u₂
                                                  inst✝¹² : CategoryTheory.Category.{v₂, u₂} D
                                                  W : CategoryTheory.MorphismProperty C
                                                  M : Type w'
                                                  inst✝¹¹ : AddMonoid M
                                                  inst✝¹⁰ : CategoryTheory.HasShift C M
                                                  inst✝⁹ : CategoryTheory.HasShift D M
                                                  inst✝⁸ : W.IsCompatibleWithShift M
                                                  X Y Z T : C
                                                  a₁ a₂ a₃ a₁₂ a₂₃ a : M
                                                  inst✝⁷ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
                                                  inst✝⁶ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
                                                  inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X T
                                                  inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
                                                  inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y T
                                                  inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z T
                                                  inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
                                                  inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M T T
                                                  α : CategoryTheory.Localization.SmallShiftedHom W X Y a₁
                                                  β : CategoryTheory.Localization.SmallShiftedHom W Y Z a₂
                                                  γ : CategoryTheory.Localization.SmallShiftedHom W Z T a₃
                                                  h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
                                                  h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
                                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
                                                  ⊢ Eq (HAdd.hAdd a₃ a₁₂) a
                                                -/
    (α.comp β h₁₂).comp γ (show a₃ + a₁₂ = a by rw [← h₁₂, ← add_assoc, h]) =
                                                /-
                                                  🎉 no goals
                                                -/
                                /-
                                  C : Type u₁
                                  inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₂
                                  inst✝¹² : CategoryTheory.Category.{v₂, u₂} D
                                  W : CategoryTheory.MorphismProperty C
                                  M : Type w'
                                  inst✝¹¹ : AddMonoid M
                                  inst✝¹⁰ : CategoryTheory.HasShift C M
                                  inst✝⁹ : CategoryTheory.HasShift D M
                                  inst✝⁸ : W.IsCompatibleWithShift M
                                  X Y Z T : C
                                  a₁ a₂ a₃ a₁₂ a₂₃ a : M
                                  inst✝⁷ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
                                  inst✝⁶ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
                                  inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X T
                                  inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
                                  inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y T
                                  inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z T
                                  inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
                                  inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M T T
                                  α : CategoryTheory.Localization.SmallShiftedHom W X Y a₁
                                  β : CategoryTheory.Localization.SmallShiftedHom W Y Z a₂
                                  γ : CategoryTheory.Localization.SmallShiftedHom W Z T a₃
                                  h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
                                  h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
                                  ⊢ Eq (HAdd.hAdd a₂₃ a₁) a
                                -/
      α.comp (β.comp γ h₂₃) (by rw [← h₂₃, h]) := by
                                /-
                                  🎉 no goals
                                -/
  /-
    C : Type u₁
    inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝¹⁰ : AddMonoid M
    inst✝⁹ : CategoryTheory.HasShift C M
    inst✝⁸ : W.IsCompatibleWithShift M
    X Y Z T : C
    a₁ a₂ a₃ a₁₂ a₂₃ a : M
    inst✝⁷ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝⁶ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
    inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X T
    inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y T
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z T
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M T T
    α : CategoryTheory.Localization.SmallShiftedHom W X Y a₁
    β : CategoryTheory.Localization.SmallShiftedHom W Y Z a₂
    γ : CategoryTheory.Localization.SmallShiftedHom W Z T a₃
    h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
    h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
    h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
    ⊢ Eq ((α.comp β h₁₂).comp γ ⋯) (α.comp (β.comp γ h₂₃) ⋯)
  -/
  apply (equiv W W.Q).injective
  /-
    case a
    C : Type u₁
    inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝¹⁰ : AddMonoid M
    inst✝⁹ : CategoryTheory.HasShift C M
    inst✝⁸ : W.IsCompatibleWithShift M
    X Y Z T : C
    a₁ a₂ a₃ a₁₂ a₂₃ a : M
    inst✝⁷ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝⁶ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Z
    inst✝⁵ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X T
    inst✝⁴ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y Z
    inst✝³ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Y T
    inst✝² : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z T
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M Z Z
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M T T
    α : CategoryTheory.Localization.SmallShiftedHom W X Y a₁
    β : CategoryTheory.Localization.SmallShiftedHom W Y Z a₂
    γ : CategoryTheory.Localization.SmallShiftedHom W Z T a₃
    h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
    h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
    h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W W.Q) ((α.comp β h₁₂ …
  -/
  simp only [equiv_comp, ShiftedHom.comp_assoc _ _ _ h₁₂ h₂₃ h]
  /-
    🎉 no goals
  -/


/-- Up to an equivalence, the type `SmallShiftedHom.{w} W X Y m` does
not depend on the universe `w`. -/
noncomputable def chgUniv {X Y : C} {m : M}
    [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w''} W M X Y] :
    SmallShiftedHom.{w} W X Y m ≃ SmallShiftedHom.{w''} W X Y m :=
  SmallHom.chgUniv


lemma equiv_chgUniv (L : C ⥤ D) [L.IsLocalization W] [L.CommShift M] {X Y : C} {m : M}
    [HasSmallLocalizedShiftedHom.{w} W M X Y]
    [HasSmallLocalizedShiftedHom.{w''} W M X Y]
    (e : SmallShiftedHom.{w} W X Y m) :
    equiv W L (chgUniv.{w''} e) = equiv W L e := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    inst✝² : L.CommShift M
    X Y : C
    m : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    e : CategoryTheory.Localization.SmallShiftedHom W X Y m
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv W L) (CategoryTheory. …
  -/
  dsimp [equiv]
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    inst✝² : L.CommShift M
    X Y : C
    m : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    e : CategoryTheory.Localization.SmallShiftedHom W X Y m
    ⊢ Eq (((L.commShiftIso m).app Y).homToEquiv ((CategoryTheory.Localization.Smal …
  -/
  congr
  /-
    case h.e_6.h
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
    W : CategoryTheory.MorphismProperty C
    M : Type w'
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : CategoryTheory.HasShift D M
    L : CategoryTheory.Functor C D
    inst✝³ : L.IsLocalization W
    inst✝² : L.CommShift M
    X Y : C
    m : M
    inst✝¹ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    inst✝ : CategoryTheory.Localization.HasSmallLocalizedShiftedHom W M X Y
    e : CategoryTheory.Localization.SmallShiftedHom W X Y m
    ⊢ Eq ((CategoryTheory.Localization.SmallHom.equiv W L) (CategoryTheory.Localiz …
  -/
  apply SmallHom.equiv_chgUniv
  /-
    🎉 no goals
  -/


