/-- The opposite of a left fraction. -/
abbrev LeftFraction.neg {X Y : C} (φ : W.LeftFraction X Y) :
    W.LeftFraction X Y where
  Y' := φ.Y'
  f := -φ.f
  s := φ.s
  hs := φ.hs


/-- The sum of two left fractions with the same denominator. -/
abbrev add : W.LeftFraction X Y where
  Y' := φ.Y'
  f := φ.f + φ.f'
  s := φ.s
  hs := φ.hs


@[simp]
lemma symm_add : φ.symm.add = φ.add := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction₂ X Y
    ⊢ Eq φ.symm.add φ.add
  -/
  dsimp [add, symm]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction₂ X Y
    ⊢ Eq (CategoryTheory.MorphismProperty.LeftFraction.mk (HAdd.hAdd φ.f' φ.f) φ.s …
  -/
  congr 1
  /-
    case e_f
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction₂ X Y
    ⊢ Eq (HAdd.hAdd φ.f' φ.f) (HAdd.hAdd φ.f φ.f')
  -/
  apply add_comm
  /-
    🎉 no goals
  -/


@[simp]
lemma map_add (F : C ⥤ D) (hF : W.IsInvertedBy F) [Preadditive D] [F.Additive] :
    φ.add.map F hF = φ.fst.map F hF + φ.snd.map F hF := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    W : CategoryTheory.MorphismProperty C
    X Y : C
    φ : W.LeftFraction₂ X Y
    F : CategoryTheory.Functor C D
    hF : W.IsInvertedBy F
    inst✝¹ : CategoryTheory.Preadditive D
    inst✝ : F.Additive
    ⊢ Eq (φ.add.map F hF) (HAdd.hAdd (φ.fst.map F hF) (φ.snd.map F hF))
  -/
  have := hF φ.s φ.hs
  rw [← cancel_mono (F.map φ.s), add_comp, LeftFraction.map_comp_map_s,
    LeftFraction.map_comp_map_s, LeftFraction.map_comp_map_s, F.map_add]


/-- The opposite of a map `L.obj X ⟶ L.obj Y` when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def neg' (f : L.obj X ⟶ L.obj Y) : L.obj X ⟶ L.obj Y :=
  (exists_leftFraction L W f).choose.neg.map L (inverts L W)


lemma neg'_eq (f : L.obj X ⟶ L.obj Y) (φ : W.LeftFraction X Y)
    (hφ : f = φ.map L (inverts L W)) :
    neg' W f = φ.neg.map L (inverts L W) := by
  obtain ⟨φ₀, rfl, hφ₀⟩ : ∃ (φ₀ : W.LeftFraction X Y)
    (_ : f = φ₀.map L (inverts L W)),
      neg' W f = φ₀.neg.map L (inverts L W) :=
    ⟨_, (exists_leftFraction L W f).choose_spec, rfl⟩
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction X Y
    hφ : Eq (φ₀.map L ⋯) (φ.map L ⋯)
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ₀.neg …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ.neg.map  …
  -/
  rw [MorphismProperty.LeftFraction.map_eq_iff] at hφ
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction X Y
    hφ : CategoryTheory.MorphismProperty.LeftFractionRel φ₀ φ
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ₀.neg …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ.neg.map  …
  -/
  obtain ⟨Y', t₁, t₂, hst, hft, ht⟩ := hφ
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction X Y
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ₀.neg …
    Y' : C
    t₁ : Quiver.Hom φ₀.Y' Y'
    t₂ : Quiver.Hom φ.Y' Y'
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ.neg.map  …
  -/
  have := inverts L W _ ht
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction X Y
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ₀.neg …
    Y' : C
    t₁ : Quiver.Hom φ₀.Y' Y'
    t₂ : Quiver.Hom φ.Y' Y'
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ₀.s t₁))
    ⊢ Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ.neg.map  …
  -/
  rw [← cancel_mono (L.map (φ₀.s ≫ t₁))]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction X Y
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.neg' W (φ₀.map L ⋯)) (φ₀.neg …
    Y' : C
    t₁ : Quiver.Hom φ₀.Y' Y'
    t₂ : Quiver.Hom φ.Y' Y'
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ₀.s t₁))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  nth_rw 1 [L.map_comp]
  rw [hφ₀, hst, LeftFraction.map_comp_map_s_assoc, L.map_comp,
    LeftFraction.map_comp_map_s_assoc, ← L.map_comp, ← L.map_comp,
    neg_comp, neg_comp, hft]


/-- The addition of two maps `L.obj X ⟶ L.obj Y` when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def add' (f₁ f₂ : L.obj X ⟶ L.obj Y) : L.obj X ⟶ L.obj Y :=
  (exists_leftFraction₂ L W f₁ f₂).choose.add.map L (inverts L W)


lemma add'_eq (f₁ f₂ : L.obj X ⟶ L.obj Y) (φ : W.LeftFraction₂ X Y)
    (hφ₁ : f₁ = φ.fst.map L (inverts L W))
    (hφ₂ : f₂ = φ.snd.map L (inverts L W)) :
    add' W f₁ f₂ = φ.add.map L (inverts L W) := by
  obtain ⟨φ₀, rfl, rfl, hφ₀⟩ : ∃ (φ₀ : W.LeftFraction₂ X Y)
    (_ : f₁ = φ₀.fst.map L (inverts L W))
    (_ : f₂ = φ₀.snd.map L (inverts L W)),
    add' W f₁ f₂ = φ₀.add.map L (inverts L W) :=
    ⟨(exists_leftFraction₂ L W f₁ f₂).choose,
      (exists_leftFraction₂ L W f₁ f₂).choose_spec.1,
      (exists_leftFraction₂ L W f₁ f₂).choose_spec.2, rfl⟩
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction₂ X Y
    hφ₁ : Eq (φ₀.fst.map L ⋯) (φ.fst.map L ⋯)
    hφ₂ : Eq (φ₀.snd.map L ⋯) (φ.snd.map L ⋯)
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀. …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀.snd. …
  -/
  obtain ⟨Z, t₁, t₂, hst, hft, hft', ht⟩ := (LeftFraction₂.map_eq_iff L W φ₀ φ).1 ⟨hφ₁, hφ₂⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction₂ X Y
    hφ₁ : Eq (φ₀.fst.map L ⋯) (φ.fst.map L ⋯)
    hφ₂ : Eq (φ₀.snd.map L ⋯) (φ.snd.map L ⋯)
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀. …
    Z : C
    t₁ : Quiver.Hom φ₀.Y' Z
    t₂ : Quiver.Hom φ.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    hft' : Eq (CategoryTheory.CategoryStruct.comp φ₀.f' t₁) (CategoryTheory.Catego …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀.snd. …
  -/
  have := inverts L W _ ht
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction₂ X Y
    hφ₁ : Eq (φ₀.fst.map L ⋯) (φ.fst.map L ⋯)
    hφ₂ : Eq (φ₀.snd.map L ⋯) (φ.snd.map L ⋯)
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀. …
    Z : C
    t₁ : Quiver.Hom φ₀.Y' Z
    t₂ : Quiver.Hom φ.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    hft' : Eq (CategoryTheory.CategoryStruct.comp φ₀.f' t₁) (CategoryTheory.Catego …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ₀.s t₁))
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀.snd. …
  -/
  rw [hφ₀, ← cancel_mono (L.map (φ₀.s ≫ t₁))]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    φ φ₀ : W.LeftFraction₂ X Y
    hφ₁ : Eq (φ₀.fst.map L ⋯) (φ.fst.map L ⋯)
    hφ₂ : Eq (φ₀.snd.map L ⋯) (φ.snd.map L ⋯)
    hφ₀ : Eq (CategoryTheory.Localization.Preadditive.add' W (φ₀.fst.map L ⋯) (φ₀. …
    Z : C
    t₁ : Quiver.Hom φ₀.Y' Z
    t₂ : Quiver.Hom φ.Y' Z
    hst : Eq (CategoryTheory.CategoryStruct.comp φ₀.s t₁) (CategoryTheory.Category …
    hft : Eq (CategoryTheory.CategoryStruct.comp φ₀.f t₁) (CategoryTheory.Category …
    hft' : Eq (CategoryTheory.CategoryStruct.comp φ₀.f' t₁) (CategoryTheory.Catego …
    ht : W (CategoryTheory.CategoryStruct.comp φ₀.s t₁)
    this : CategoryTheory.IsIso (L.map (CategoryTheory.CategoryStruct.comp φ₀.s t₁))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ₀.add.map L ⋯) (L.map (CategoryTheo …
  -/
  nth_rw 2 [hst]
  rw [L.map_comp, L.map_comp, LeftFraction.map_comp_map_s_assoc,
    LeftFraction.map_comp_map_s_assoc, ← L.map_comp, ← L.map_comp,
    add_comp, add_comp, hft, hft']


lemma add'_comm (f₁ f₂ : L.obj X ⟶ L.obj Y) :
    add' W f₁ f₂ = add' W f₂ f₁ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W f₁ f₂) (CategoryTheory.Lo …
  -/
  obtain ⟨α, h₁, h₂⟩ := exists_leftFraction₂ L W f₁ f₂
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W f₁ f₂) (CategoryTheory.Lo …
  -/
  rw [add'_eq W f₁ f₂ α h₁ h₂, add'_eq W f₂ f₁ α.symm h₂ h₁, α.symm_add]
  /-
    🎉 no goals
  -/


lemma add'_zero (f : L.obj X ⟶ L.obj Y) :
    add' W f (L.map 0) = f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W f (L.map 0)) f
  -/
  obtain ⟨α, hα⟩ := exists_leftFraction L W f
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W f (L.map 0)) f
  -/
  rw [add'_eq W f (L.map 0) (LeftFraction₂.mk α.f 0 α.s α.hs) hα, hα]; swap
    /-
      case intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      α : W.LeftFraction X Y
      hα : Eq f (α.map L ⋯)
      ⊢ Eq (L.map 0) ((CategoryTheory.MorphismProperty.LeftFraction₂.mk α.f 0 α.s ⋯) …
    -/
  · have := inverts L W _ α.hs
    rw [← cancel_mono (L.map α.s), ← L.map_comp, Limits.zero_comp,
      LeftFraction.map_comp_map_s]
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction₂.mk α.f 0 α.s ⋯).add.map L …
  -/
  dsimp [LeftFraction₂.add]
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (HAdd.hAdd α.f 0) α.s ⋯ …
  -/
  rw [add_zero]
  /-
    🎉 no goals
  -/


lemma zero_add' (f : L.obj X ⟶ L.obj Y) :
    add' W (L.map 0) f = f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (L.map 0) f) f
  -/
  rw [add'_comm, add'_zero]
  /-
    🎉 no goals
  -/


lemma neg'_add'_self (f : L.obj X ⟶ L.obj Y) :
    add' W (neg' W f) f = L.map 0 := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (CategoryTheory.Localizat …
  -/
  obtain ⟨α, rfl⟩ := exists_leftFraction L W f
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    α : W.LeftFraction X Y
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (CategoryTheory.Localizat …
  -/
  have := inverts L W _ α.hs
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    α : W.LeftFraction X Y
    this : CategoryTheory.IsIso (L.map α.s)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (CategoryTheory.Localizat …
  -/
  rw [add'_eq W _ _ (LeftFraction₂.mk (-α.f) α.f α.s α.hs) (neg'_eq W _ _ rfl) rfl]
  simp only [← cancel_mono (L.map α.s), LeftFraction.map_comp_map_s, ← L.map_comp,
    Limits.zero_comp, neg_add_cancel]


lemma add'_assoc (f₁ f₂ f₃ : L.obj X ⟶ L.obj Y) :
    add' W (add' W f₁ f₂) f₃ = add' W f₁ (add' W f₂ f₃) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ f₃ : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add' W (CategoryTheory.Localizat …
  -/
  obtain ⟨α, h₁, h₂, h₃⟩ := exists_leftFraction₃ L W f₁ f₂ f₃
  rw [add'_eq W f₁ f₂ α.forgetThd h₁ h₂, add'_eq W f₂ f₃ α.forgetFst h₂ h₃,
    add'_eq W _ _ (LeftFraction₂.mk (α.f + α.f') α.f'' α.s α.hs) rfl h₃,
    add'_eq W _ _ (LeftFraction₂.mk α.f (α.f' + α.f'') α.s α.hs) h₁ rfl]
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ f₃ : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₃ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    h₃ : Eq f₃ (α.thd.map L ⋯)
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction₂.mk (HAdd.hAdd α.f α.f') α …
  -/
  dsimp [LeftFraction₂.add]
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ f₃ : Quiver.Hom (L.obj X) (L.obj Y)
    α : W.LeftFraction₃ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    h₃ : Eq f₃ (α.thd.map L ⋯)
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (HAdd.hAdd (HAdd.hAdd α …
  -/
  rw [add_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma add'_comp (f₁ f₂ : L.obj X ⟶ L.obj Y) (g : L.obj Y ⟶ L.obj Z) :
    add' W f₁ f₂ ≫ g = add' W (f₁ ≫ g) (f₂ ≫ g) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨α, h₁, h₂⟩ := exists_leftFraction₂ L W f₁ f₂
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨β, hβ⟩ := exists_leftFraction L W g
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    β : W.LeftFraction Y Z
    hβ : Eq g (β.map L ⋯)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨γ, hγ⟩ := (RightFraction.mk _ α.hs β.f).exists_leftFraction
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    β : W.LeftFraction Y Z
    hβ : Eq g (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  dsimp at hγ
  rw [add'_eq W f₁ f₂ α h₁ h₂, add'_eq W (f₁ ≫ g) (f₂ ≫ g)
    (LeftFraction₂.mk (α.f ≫ γ.f) (α.f' ≫ γ.f) (β.s ≫ γ.s)
                                  /-
                                    case intro.intro.intro.intro
                                    C : Type u_1
                                    D : Type u_2
                                    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
                                    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
                                    inst✝² : CategoryTheory.Preadditive C
                                    L : CategoryTheory.Functor C D
                                    W : CategoryTheory.MorphismProperty C
                                    inst✝¹ : L.IsLocalization W
                                    inst✝ : W.HasLeftCalculusOfFractions
                                    X Y Z : C
                                    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
                                    g : Quiver.Hom (L.obj Y) (L.obj Z)
                                    α : W.LeftFraction₂ X Y
                                    h₁ : Eq f₁ (α.fst.map L ⋯)
                                    h₂ : Eq f₂ (α.snd.map L ⋯)
                                    β : W.LeftFraction Y Z
                                    hβ : Eq g (β.map L ⋯)
                                    γ : W.LeftFraction α.Y' β.Y'
                                    hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.add.map L ⋯) g) ((CategoryTheory.M …
                                  -/
    (W.comp_mem _ _ β.hs γ.hs))]; rotate_left
    /-
      case intro.intro.intro.intro.hφ₁
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
      g : Quiver.Hom (L.obj Y) (L.obj Z)
      α : W.LeftFraction₂ X Y
      h₁ : Eq f₁ (α.fst.map L ⋯)
      h₂ : Eq f₂ (α.snd.map L ⋯)
      β : W.LeftFraction Y Z
      hβ : Eq g (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ g) ((CategoryTheory.MorphismProper …
    -/
  · rw [h₁, hβ]
    /-
      case intro.intro.intro.intro.hφ₁
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
      g : Quiver.Hom (L.obj Y) (L.obj Z)
      α : W.LeftFraction₂ X Y
      h₁ : Eq f₁ (α.fst.map L ⋯)
      h₂ : Eq f₂ (α.snd.map L ⋯)
      β : W.LeftFraction Y Z
      hβ : Eq g (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.fst.map L ⋯) (β.map L ⋯)) ((Catego …
    -/
    exact LeftFraction.map_comp_map_eq_map _ _ _ hγ _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.hφ₂
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
      g : Quiver.Hom (L.obj Y) (L.obj Z)
      α : W.LeftFraction₂ X Y
      h₁ : Eq f₁ (α.fst.map L ⋯)
      h₂ : Eq f₂ (α.snd.map L ⋯)
      β : W.LeftFraction Y Z
      hβ : Eq g (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₂ g) ((CategoryTheory.MorphismProper …
    -/
  · rw [h₂, hβ]
    /-
      case intro.intro.intro.intro.hφ₂
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasLeftCalculusOfFractions
      X Y Z : C
      f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
      g : Quiver.Hom (L.obj Y) (L.obj Z)
      α : W.LeftFraction₂ X Y
      h₁ : Eq f₁ (α.fst.map L ⋯)
      h₂ : Eq f₂ (α.snd.map L ⋯)
      β : W.LeftFraction Y Z
      hβ : Eq g (β.map L ⋯)
      γ : W.LeftFraction α.Y' β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.snd.map L ⋯) (β.map L ⋯)) ((Catego …
    -/
    exact LeftFraction.map_comp_map_eq_map _ _ _ hγ _
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    β : W.LeftFraction Y Z
    hβ : Eq g (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.add.map L ⋯) g) ((CategoryTheory.M …
  -/
  rw [hβ, LeftFraction.map_comp_map_eq_map _ _ γ hγ]
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    β : W.LeftFraction Y Z
    hβ : Eq g (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
    ⊢ Eq ((α.add.comp₀ β γ).map L ⋯) ((CategoryTheory.MorphismProperty.LeftFractio …
  -/
  dsimp [LeftFraction₂.add]
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction₂ X Y
    h₁ : Eq f₁ (α.fst.map L ⋯)
    h₂ : Eq f₂ (α.snd.map L ⋯)
    β : W.LeftFraction Y Z
    hβ : Eq g (β.map L ⋯)
    γ : W.LeftFraction α.Y' β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.CategoryS …
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (CategoryTheory.Categor …
  -/
  rw [add_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma comp_add' (f : L.obj X ⟶ L.obj Y) (g₁ g₂ : L.obj Y ⟶ L.obj Z) :
    f ≫ add' W g₁ g₂ = add' W (f ≫ g₁) (f ≫ g₂) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Localization.Preadd …
  -/
  obtain ⟨α, hα⟩ := exists_leftFraction L W f
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Localization.Preadd …
  -/
  obtain ⟨β, hβ₁, hβ₂⟩ := exists_leftFraction₂ L W g₁ g₂
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    β : W.LeftFraction₂ Y Z
    hβ₁ : Eq g₁ (β.fst.map L ⋯)
    hβ₂ : Eq g₂ (β.snd.map L ⋯)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Localization.Preadd …
  -/
  obtain ⟨γ, hγ₁, hγ₂⟩ := (RightFraction₂.mk _ α.hs β.f β.f').exists_leftFraction₂
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    β : W.LeftFraction₂ Y Z
    hβ₁ : Eq g₁ (β.fst.map L ⋯)
    hβ₂ : Eq g₂ (β.snd.map L ⋯)
    γ : W.LeftFraction₂ α.Y' β.Y'
    hγ₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
    hγ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Localization.Preadd …
  -/
  dsimp at hγ₁ hγ₂
  rw [add'_eq W g₁ g₂ β hβ₁ hβ₂, add'_eq W (f ≫ g₁) (f ≫ g₂)
    (LeftFraction₂.mk (α.f ≫ γ.f) (α.f ≫ γ.f') (β.s ≫ γ.s) (W.comp_mem _ _ β.hs γ.hs))
    (by simpa only [hα, hβ₁] using LeftFraction.map_comp_map_eq_map α β.fst γ.fst hγ₁ L)
    (by simpa only [hα, hβ₂] using LeftFraction.map_comp_map_eq_map α β.snd γ.snd hγ₂ L),
    hα, LeftFraction.map_comp_map_eq_map α β.add γ.add
      (by simp only [add_comp, hγ₁, hγ₂, comp_add])]
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    β : W.LeftFraction₂ Y Z
    hβ₁ : Eq g₁ (β.fst.map L ⋯)
    hβ₂ : Eq g₂ (β.snd.map L ⋯)
    γ : W.LeftFraction₂ α.Y' β.Y'
    hγ₁ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.Category …
    hγ₂ : Eq (CategoryTheory.CategoryStruct.comp β.f' γ.s) (CategoryTheory.Categor …
    ⊢ Eq ((α.comp₀ β.add γ.add).map L ⋯) ((CategoryTheory.MorphismProperty.LeftFra …
  -/
  dsimp [LeftFraction₂.add]
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    α : W.LeftFraction X Y
    hα : Eq f (α.map L ⋯)
    β : W.LeftFraction₂ Y Z
    hβ₁ : Eq g₁ (β.fst.map L ⋯)
    hβ₂ : Eq g₂ (β.snd.map L ⋯)
    γ : W.LeftFraction₂ α.Y' β.Y'
    hγ₁ : Eq (CategoryTheory.CategoryStruct.comp β.f γ.s) (CategoryTheory.Category …
    hγ₂ : Eq (CategoryTheory.CategoryStruct.comp β.f' γ.s) (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.MorphismProperty.LeftFraction.mk (CategoryTheory.Categor …
  -/
  rw [comp_add]
  /-
    🎉 no goals
  -/


@[simp]
lemma add'_map (f₁ f₂ : X ⟶ Y) :
    add' W (L.map f₁) (L.map f₂) = L.map (f₁ + f₂) :=
  (add'_eq W (L.map f₁) (L.map f₂) (LeftFraction₂.mk f₁ f₂ (𝟙 _) (W.id_mem _))
    (LeftFraction.map_ofHom _ _ _ _).symm (LeftFraction.map_ofHom _ _ _ _).symm).trans
    (LeftFraction.map_ofHom _ _ _ _)


/-- The abelian group structure on `L.obj X ⟶ L.obj Y` when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def addCommGroup' : AddCommGroup (L.obj X ⟶ L.obj Y) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.49351, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.49355, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    ⊢ AddCommGroup (Quiver.Hom (L.obj X) (L.obj Y))
  -/
  letI : Zero (L.obj X ⟶ L.obj Y) := ⟨L.map 0⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.49351, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.49355, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    this : Zero (Quiver.Hom (L.obj X) (L.obj Y)) := { zero := L.map 0 }
    ⊢ AddCommGroup (Quiver.Hom (L.obj X) (L.obj Y))
  -/
  letI : Add (L.obj X ⟶ L.obj Y) := ⟨add' W⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.49351, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.49355, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    this✝ : Zero (Quiver.Hom (L.obj X) (L.obj Y)) := { zero := L.map 0 }
    this : Add (Quiver.Hom (L.obj X) (L.obj Y)) := { add := CategoryTheory.Localiz …
    ⊢ AddCommGroup (Quiver.Hom (L.obj X) (L.obj Y))
  -/
  letI : Neg (L.obj X ⟶ L.obj Y) := ⟨neg' W⟩
  exact
    { add_assoc := add'_assoc _
      add_zero := add'_zero _
      add_comm := add'_comm _
      zero_add := zero_add' _
      neg_add_cancel := neg'_add'_self _
      nsmul := nsmulRec
      zsmul := zsmulRec }


/-- The bijection `(X' ⟶ Y') ≃ (L.obj X ⟶ L.obj Y)` induced by isomorphisms
`eX : L.obj X ≅ X'` and `eY : L.obj Y ≅ Y'`. -/
@[simps]
def homEquiv : (X' ⟶ Y') ≃ (L.obj X ⟶ L.obj Y) where
  toFun f := eX.hom ≫ f ≫ eY.inv
  invFun g := eX.inv ≫ g ≫ eY.hom
                   /-
                     C : Type u_1
                     D : Type u_2
                     inst✝⁴ : CategoryTheory.Category.{?u.54324, u_1} C
                     inst✝³ : CategoryTheory.Category.{?u.54328, u_2} D
                     inst✝² : CategoryTheory.Preadditive C
                     L : CategoryTheory.Functor C D
                     W : CategoryTheory.MorphismProperty C
                     inst✝¹ : L.IsLocalization W
                     inst✝ : W.HasLeftCalculusOfFractions
                     X Y Z : C
                     X' Y' Z' : D
                     eX : CategoryTheory.Iso (L.obj X) X'
                     eY : CategoryTheory.Iso (L.obj Y) Y'
                     eZ : CategoryTheory.Iso (L.obj Z) Z'
                     x✝ : Quiver.Hom X' Y'
                     ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp eX.inv (CategoryTheory.Cate …
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u_1
                      D : Type u_2
                      inst✝⁴ : CategoryTheory.Category.{?u.54324, u_1} C
                      inst✝³ : CategoryTheory.Category.{?u.54328, u_2} D
                      inst✝² : CategoryTheory.Preadditive C
                      L : CategoryTheory.Functor C D
                      W : CategoryTheory.MorphismProperty C
                      inst✝¹ : L.IsLocalization W
                      inst✝ : W.HasLeftCalculusOfFractions
                      X Y Z : C
                      X' Y' Z' : D
                      eX : CategoryTheory.Iso (L.obj X) X'
                      eY : CategoryTheory.Iso (L.obj Y) Y'
                      eZ : CategoryTheory.Iso (L.obj Z) Z'
                      x✝ : Quiver.Hom (L.obj X) (L.obj Y)
                      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp eX.hom (CategoryTheory.Cate …
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/


/-- The addition of morphisms in `D`, when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def add (f₁ f₂ : X' ⟶ Y') : X' ⟶ Y' :=
  (homEquiv eX eY).symm (add' W (homEquiv eX eY f₁) (homEquiv eX eY f₂))


@[reassoc]
lemma add_comp (f₁ f₂ : X' ⟶ Y') (g : Y' ⟶ Z') :
    add W eX eY f₁ f₂ ≫ g = add W eX eZ (f₁ ≫ g) (f₂ ≫ g) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    f₁ f₂ : Quiver.Hom X' Y'
    g : Quiver.Hom Y' Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨f₁, rfl⟩ := (homEquiv eX eY).symm.surjective f₁
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    f₂ : Quiver.Hom X' Y'
    g : Quiver.Hom Y' Z'
    f₁ : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨f₂, rfl⟩ := (homEquiv eX eY).symm.surjective f₂
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    g : Quiver.Hom Y' Z'
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  obtain ⟨g, rfl⟩ := (homEquiv eY eZ).symm.surjective g
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    f₁ f₂ : Quiver.Hom (L.obj X) (L.obj Y)
    g : Quiver.Hom (L.obj Y) (L.obj Z)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Preaddit …
  -/
  simp [add]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma comp_add (f : X' ⟶ Y') (g₁ g₂ : Y' ⟶ Z') :
    f ≫ add W eY eZ g₁ g₂ = add W eX eZ (f ≫ g₁) (f ≫ g₂) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    f : Quiver.Hom X' Y'
    g₁ g₂ : Quiver.Hom Y' Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Localization.Preadd …
  -/
  obtain ⟨f, rfl⟩ := (homEquiv eX eY).symm.surjective f
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    g₁ g₂ : Quiver.Hom Y' Z'
    f : Quiver.Hom (L.obj X) (L.obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Preaddi …
  -/
  obtain ⟨g₁, rfl⟩ := (homEquiv eY eZ).symm.surjective g₁
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    g₂ : Quiver.Hom Y' Z'
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ : Quiver.Hom (L.obj Y) (L.obj Z)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Preaddi …
  -/
  obtain ⟨g₂, rfl⟩ := (homEquiv eY eZ).symm.surjective g₂
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    f : Quiver.Hom (L.obj X) (L.obj Y)
    g₁ g₂ : Quiver.Hom (L.obj Y) (L.obj Z)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Preaddi …
  -/
  simp [add]
  /-
    🎉 no goals
  -/


lemma add_eq_add {X'' Y'' : C} (eX' : L.obj X'' ≅ X') (eY' : L.obj Y'' ≅ Y')
    (f₁ f₂ : X' ⟶ Y') :
    add W eX eY f₁ f₂ = add W eX' eY' f₁ f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    X'' Y'' : C
    eX' : CategoryTheory.Iso (L.obj X'') X'
    eY' : CategoryTheory.Iso (L.obj Y'') Y'
    f₁ f₂ : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryTheo …
  -/
  have h₁ := comp_add W eX' eX eY (𝟙 _) f₁ f₂
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    X'' Y'' : C
    eX' : CategoryTheory.Iso (L.obj X'') X'
    eY' : CategoryTheory.Iso (L.obj Y'') Y'
    f₁ f₂ : Quiver.Hom X' Y'
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryTheo …
  -/
  have h₂ := add_comp W eX' eY eY' f₁ f₂ (𝟙 _)
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    X'' Y'' : C
    eX' : CategoryTheory.Iso (L.obj X'') X'
    eY' : CategoryTheory.Iso (L.obj Y'') Y'
    f₁ f₂ : Quiver.Hom X' Y'
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Pread …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryTheo …
  -/
  simp only [id_comp] at h₁
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    X'' Y'' : C
    eX' : CategoryTheory.Iso (L.obj X'') X'
    eY' : CategoryTheory.Iso (L.obj Y'') Y'
    f₁ f₂ : Quiver.Hom X' Y'
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.Pread …
    h₁ : Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryT …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryTheo …
  -/
  simp only [comp_id] at h₂
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    X'' Y'' : C
    eX' : CategoryTheory.Iso (L.obj X'') X'
    eY' : CategoryTheory.Iso (L.obj Y'') Y'
    f₁ f₂ : Quiver.Hom X' Y'
    h₁ : Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryT …
    h₂ : Eq (CategoryTheory.Localization.Preadditive.add W eX' eY f₁ f₂) (Category …
    ⊢ Eq (CategoryTheory.Localization.Preadditive.add W eX eY f₁ f₂) (CategoryTheo …
  -/
  rw [h₁, h₂]
  /-
    🎉 no goals
  -/


variable (L X' Y') in
/-- The abelian group structure on morphisms in `D`, when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def addCommGroup : AddCommGroup (X' ⟶ Y') := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.72781, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.72785, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    ⊢ AddCommGroup (Quiver.Hom X' Y')
  -/
  have := Localization.essSurj L W
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.72781, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.72785, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    this : L.EssSurj
    ⊢ AddCommGroup (Quiver.Hom X' Y')
  -/
  letI := addCommGroup' L W (L.objPreimage X') (L.objPreimage Y')
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.72781, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.72785, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y Z : C
    X' Y' Z' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    eZ : CategoryTheory.Iso (L.obj Z) Z'
    this✝ : L.EssSurj
    this : AddCommGroup (Quiver.Hom (L.obj (L.objPreimage X')) (L.obj (L.objPreima …
    ⊢ AddCommGroup (Quiver.Hom X' Y')
  -/
  exact Equiv.addCommGroup (homEquiv (L.objObjPreimageIso X') (L.objObjPreimageIso Y'))
  /-
    🎉 no goals
  -/


lemma add_eq (f₁ f₂ : X' ⟶ Y') :
    letI := addCommGroup L W X' Y'
    f₁ + f₂ = add W eX eY f₁ f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    X' Y' : D
    eX : CategoryTheory.Iso (L.obj X) X'
    eY : CategoryTheory.Iso (L.obj Y) Y'
    f₁ f₂ : Quiver.Hom X' Y'
    ⊢ Eq (HAdd.hAdd f₁ f₂) (CategoryTheory.Localization.Preadditive.add W eX eY f₁ …
  -/
  apply add_eq_add
  /-
    🎉 no goals
  -/


lemma map_add (f₁ f₂ : X ⟶ Y) :
    letI := addCommGroup L W (L.obj X) (L.obj Y)
    L.map (f₁ + f₂) = L.map f₁ + L.map f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ : Quiver.Hom X Y
    ⊢ Eq (L.map (HAdd.hAdd f₁ f₂)) (HAdd.hAdd (L.map f₁) (L.map f₂))
  -/
  rw [add_eq W (Iso.refl _) (Iso.refl _) (L.map f₁) (L.map f₂)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : C
    f₁ f₂ : Quiver.Hom X Y
    ⊢ Eq (L.map (HAdd.hAdd f₁ f₂)) (CategoryTheory.Localization.Preadditive.add W  …
  -/
  simp [add]
  /-
    🎉 no goals
  -/


/-- The preadditive structure on `D`, when `L : C ⥤ D` is a localization
functor, `C` is preadditive and there is a left calculus of fractions. -/
noncomputable def preadditive : Preadditive D where
  homGroup := Preadditive.addCommGroup L W
                             /-
                               C : Type u_1
                               D : Type u_2
                               inst✝⁴ : CategoryTheory.Category.{?u.80969, u_1} C
                               inst✝³ : CategoryTheory.Category.{?u.80973, u_2} D
                               inst✝² : CategoryTheory.Preadditive C
                               L : CategoryTheory.Functor C D
                               W : CategoryTheory.MorphismProperty C
                               inst✝¹ : L.IsLocalization W
                               inst✝ : W.HasLeftCalculusOfFractions
                               x✝⁵ x✝⁴ x✝³ : D
                               x✝² x✝¹ : Quiver.Hom x✝⁵ x✝⁴
                               x✝ : Quiver.Hom x✝⁴ x✝³
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (C …
                             -/
  add_comp _ _ _ _ _ _ := by apply Preadditive.add_comp
                             /-
                               🎉 no goals
                             -/
                             /-
                               C : Type u_1
                               D : Type u_2
                               inst✝⁴ : CategoryTheory.Category.{?u.80969, u_1} C
                               inst✝³ : CategoryTheory.Category.{?u.80973, u_2} D
                               inst✝² : CategoryTheory.Preadditive C
                               L : CategoryTheory.Functor C D
                               W : CategoryTheory.MorphismProperty C
                               inst✝¹ : L.IsLocalization W
                               inst✝ : W.HasLeftCalculusOfFractions
                               x✝⁵ x✝⁴ x✝³ : D
                               x✝² : Quiver.Hom x✝⁵ x✝⁴
                               x✝¹ x✝ : Quiver.Hom x✝⁴ x✝³
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (C …
                             -/
  comp_add _ _ _ _ _ _ := by apply Preadditive.comp_add
                             /-
                               🎉 no goals
                             -/


lemma functor_additive :
    letI := preadditive L W
    L.Additive :=
  letI := preadditive L W
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_2} D
        inst✝² : CategoryTheory.Preadditive C
        L : CategoryTheory.Functor C D
        W : CategoryTheory.MorphismProperty C
        inst✝¹ : L.IsLocalization W
        inst✝ : W.HasLeftCalculusOfFractions
        this : CategoryTheory.Preadditive D := CategoryTheory.Localization.preadditive …
        ⊢ ∀ {X Y : C} {f g : Quiver.Hom X Y}, Eq (L.map (HAdd.hAdd f g)) (HAdd.hAdd (L …
      -/
  ⟨by apply Preadditive.map_add⟩
      /-
        🎉 no goals
      -/


include W in
lemma functor_additive_iff {E : Type*} [Category E] [Preadditive E] [Preadditive D] [L.Additive]
    (G : D ⥤ E) :
    G.Additive ↔ (L ⋙ G).Additive := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁵ : L.IsLocalization W
    inst✝⁴ : W.HasLeftCalculusOfFractions
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} E
    inst✝² : CategoryTheory.Preadditive E
    inst✝¹ : CategoryTheory.Preadditive D
    inst✝ : L.Additive
    G : CategoryTheory.Functor D E
    ⊢ Iff G.Additive (L.comp G).Additive
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      ⊢ G.Additive → (L.comp G).Additive
    -/
  · intro
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      a✝ : G.Additive
      ⊢ (L.comp G).Additive
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      ⊢ (L.comp G).Additive → G.Additive
    -/
  · intro h
    suffices ∀ ⦃X Y : C⦄ (f g : L.obj X ⟶ L.obj Y), G.map (f + g) = G.map f + G.map g by
      refine ⟨fun {X Y f g} => ?_⟩
      have hL := essSurj L W
      have eq := this ((L.objObjPreimageIso X).hom ≫ f ≫ (L.objObjPreimageIso Y).inv)
        ((L.objObjPreimageIso X).hom ≫ g ≫ (L.objObjPreimageIso Y).inv)
      rw [Functor.map_comp, Functor.map_comp, Functor.map_comp, Functor.map_comp,
        ← comp_add, ← comp_add, ← add_comp, ← add_comp, Functor.map_comp, Functor.map_comp] at eq
      rw [← cancel_mono (G.map (L.objObjPreimageIso Y).inv),
        ← cancel_epi (G.map (L.objObjPreimageIso X).hom), eq]
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      h : (L.comp G).Additive
      ⊢ ∀ ⦃X Y : C⦄ (f g : Quiver.Hom (L.obj X) (L.obj Y)), Eq (G.map (HAdd.hAdd f g …
    -/
    intros X Y f g
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      h : (L.comp G).Additive
      X Y : C
      f g : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Eq (G.map (HAdd.hAdd f g)) (HAdd.hAdd (G.map f) (G.map g))
    -/
    obtain ⟨φ, rfl, rfl⟩ := exists_leftFraction₂ L W f g
    /-
      case mpr.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Preadditive C
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁵ : L.IsLocalization W
      inst✝⁴ : W.HasLeftCalculusOfFractions
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} E
      inst✝² : CategoryTheory.Preadditive E
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : L.Additive
      G : CategoryTheory.Functor D E
      h : (L.comp G).Additive
      X Y : C
      φ : W.LeftFraction₂ X Y
      ⊢ Eq (G.map (HAdd.hAdd (φ.fst.map L ⋯) (φ.snd.map L ⋯))) (HAdd.hAdd (G.map (φ. …
    -/
    have := Localization.inverts L W φ.s φ.hs
    rw [← φ.map_add L (inverts L W), ← cancel_mono (G.map (L.map φ.s)), ← G.map_comp,
      add_comp, ← G.map_comp, ← G.map_comp, LeftFraction.map_comp_map_s,
      LeftFraction.map_comp_map_s, LeftFraction.map_comp_map_s, ← Functor.comp_map,
      Functor.map_add, Functor.comp_map, Functor.comp_map]


noncomputable instance : Preadditive W.Localization := preadditive W.Q W

instance : W.Q.Additive := functor_additive W.Q W

instance [HasZeroObject C] : HasZeroObject W.Localization := W.Q.hasZeroObject_of_additive


noncomputable instance : Preadditive W.Localization' := preadditive W.Q' W

instance : W.Q'.Additive := functor_additive W.Q' W

instance [HasZeroObject C] : HasZeroObject W.Localization' := W.Q'.hasZeroObject_of_additive


