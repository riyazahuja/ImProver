/-- Given `Φ : LocalizerMorphism W₁ W₂`, `L : C₂ ⥤ D` a localization functor for `W₂` and
a morphism `y : L.obj X₂ ⟶ X₃`, this is the functor which sends `R : Φ.RightResolution d` to
`(isoOfHom L W₂ R.w R.hw).inv ≫ y` in the category `w.CostructuredArrowDownwards y`
where `w` is `TwoSquare.mk Φ.functor (Φ.functor ⋙ L) L (𝟭 _) (Functor.rightUnitor _).inv`. -/
@[simps]
noncomputable def fromRightResolution :
    Φ.RightResolution X₂ ⥤ (TwoSquare.mk Φ.functor (Φ.functor ⋙ L) L (𝟭 _)
      (Functor.rightUnitor _).inv).CostructuredArrowDownwards y where
  obj R := CostructuredArrow.mk (Y := StructuredArrow.mk R.w)
     /-
       C₁ : Type u_1
       C₂ : Type u_2
       inst✝⁷ : CategoryTheory.Category.{?u.1532, u_1} C₁
       inst✝⁶ : CategoryTheory.Category.{?u.1536, u_2} C₂
       W₁ : CategoryTheory.MorphismProperty C₁
       W₂ : CategoryTheory.MorphismProperty C₂
       Φ : CategoryTheory.LocalizerMorphism W₁ W₂
       inst✝⁵ : W₁.IsMultiplicative
       inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
       inst✝³ : Φ.arrow.HasRightResolutions
       inst✝² : W₂.ContainsIdentities
       D : Type u_3
       inst✝¹ : CategoryTheory.Category.{?u.2160, u_3} D
       L : CategoryTheory.Functor C₂ D
       inst✝ : L.IsLocalization W₂
       X₂ : C₂
       X₃ : D
       y : Quiver.Hom (L.obj X₂) X₃
       R : Φ.RightResolution X₂
       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.TwoSquare.mk Φ.func …
     -/
    (StructuredArrow.homMk ((isoOfHom L W₂ R.w R.hw).inv ≫ y))
     /-
       🎉 no goals
     -/
                                           /-
                                             C₁ : Type u_1
                                             C₂ : Type u_2
                                             inst✝⁷ : CategoryTheory.Category.{?u.1532, u_1} C₁
                                             inst✝⁶ : CategoryTheory.Category.{?u.1536, u_2} C₂
                                             W₁ : CategoryTheory.MorphismProperty C₁
                                             W₂ : CategoryTheory.MorphismProperty C₂
                                             Φ : CategoryTheory.LocalizerMorphism W₁ W₂
                                             inst✝⁵ : W₁.IsMultiplicative
                                             inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
                                             inst✝³ : Φ.arrow.HasRightResolutions
                                             inst✝² : W₂.ContainsIdentities
                                             D : Type u_3
                                             inst✝¹ : CategoryTheory.Category.{?u.2160, u_3} D
                                             L : CategoryTheory.Functor C₂ D
                                             inst✝ : L.IsLocalization W₂
                                             X₂ : C₂
                                             X₃ : D
                                             y : Quiver.Hom (L.obj X₂) X₃
                                             R R' : Φ.RightResolution X₂
                                             φ : Quiver.Hom R R'
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun R => CategoryTheory.Costructure …
                                           -/
  map {R R'} φ := CostructuredArrow.homMk (StructuredArrow.homMk φ.f) (by
                                           /-
                                             🎉 no goals
                                           -/
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{?u.1532, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{?u.1536, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{?u.2160, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      R R' : Φ.RightResolution X₂
      φ : Quiver.Hom R R'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.TwoSquare.mk Φ.func …
    -/
    ext
    /-
      case h
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{?u.1532, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{?u.1536, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{?u.2160, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      R R' : Φ.RightResolution X₂
      φ : Quiver.Hom R R'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.TwoSquare.mk Φ.func …
    -/
    dsimp
    rw [← assoc, ← cancel_epi (isoOfHom L W₂ R.w R.hw).hom,
      isoOfHom_hom, isoOfHom_hom_inv_id_assoc, assoc, ← L.map_comp_assoc,
      φ.comm, isoOfHom_hom_inv_id_assoc])


lemma isConnected :
    IsConnected ((TwoSquare.mk Φ.functor (Φ.functor ⋙ L) L (𝟭 _)
      (Functor.rightUnitor _).inv).CostructuredArrowDownwards y) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    ⊢ CategoryTheory.IsConnected ((CategoryTheory.TwoSquare.mk Φ.functor (Φ.functo …
  -/
  let w := (TwoSquare.mk Φ.functor (Φ.functor ⋙ L) L (𝟭 _) (Functor.rightUnitor _).inv)
  have : Nonempty (w.CostructuredArrowDownwards y) :=
    ⟨(fromRightResolution Φ L y).obj (Classical.arbitrary _)⟩
  suffices ∀ (X : w.CostructuredArrowDownwards y),
      ∃ Y, Zigzag X ((fromRightResolution Φ L y).obj Y) by
    refine zigzag_isConnected (fun X X' => ?_)
    obtain ⟨Y, hX⟩ := this X
    obtain ⟨Y', hX'⟩ := this X'
    exact hX.trans ((zigzag_obj_of_zigzag _ (isPreconnected_zigzag Y Y')).trans hX'.symm)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    ⊢ ∀ (X : w.CostructuredArrowDownwards y), Exists fun Y => CategoryTheory.Zigza …
  -/
  intro X
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    X : w.CostructuredArrowDownwards y
    ⊢ Exists fun Y => CategoryTheory.Zigzag X ((CategoryTheory.LocalizerMorphism.I …
  -/
  obtain ⟨c, g, x, fac, rfl⟩ := TwoSquare.CostructuredArrowDownwards.mk_surjective X
  /-
    case intro.intro.intro.intro
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    c : C₁
    g : Quiver.Hom X₂ (Φ.functor.obj c)
    x : Quiver.Hom ((Φ.functor.comp L).obj c) X₃
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Categor …
    ⊢ Exists fun Y => CategoryTheory.Zigzag (CategoryTheory.TwoSquare.Costructured …
  -/
  dsimp [w] at x fac
  /-
    case intro.intro.intro.intro
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    c : C₁
    g : Quiver.Hom X₂ (Φ.functor.obj c)
    x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Categor …
    ⊢ Exists fun Y => CategoryTheory.Zigzag (CategoryTheory.TwoSquare.Costructured …
  -/
  rw [id_comp] at fac
  /-
    case intro.intro.intro.intro
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    c : C₁
    g : Quiver.Hom X₂ (Φ.functor.obj c)
    x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
    ⊢ Exists fun Y => CategoryTheory.Zigzag (CategoryTheory.TwoSquare.Costructured …
  -/
  let ρ : Φ.arrow.RightResolution (Arrow.mk g) := Classical.arbitrary _
  /-
    case intro.intro.intro.intro
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this : Nonempty (w.CostructuredArrowDownwards y)
    c : C₁
    g : Quiver.Hom X₂ (Φ.functor.obj c)
    x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
    ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
    ⊢ Exists fun Y => CategoryTheory.Zigzag (CategoryTheory.TwoSquare.Costructured …
  -/
  refine ⟨RightResolution.mk ρ.w.left ρ.hw.1, ?_⟩
  have := zigzag_obj_of_zigzag
    (fromRightResolution Φ L x ⋙ w.costructuredArrowDownwardsPrecomp x y g fac)
      (isPreconnected_zigzag (RightResolution.mk (𝟙 _) (W₂.id_mem _))
        (RightResolution.mk ρ.w.right ρ.hw.2))
  /-
    case intro.intro.intro.intro
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁵ : W₁.IsMultiplicative
    inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝³ : Φ.arrow.HasRightResolutions
    inst✝² : W₂.ContainsIdentities
    D : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
    L : CategoryTheory.Functor C₂ D
    inst✝ : L.IsLocalization W₂
    X₂ : C₂
    X₃ : D
    y : Quiver.Hom (L.obj X₂) X₃
    w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
    this✝ : Nonempty (w.CostructuredArrowDownwards y)
    c : C₁
    g : Quiver.Hom X₂ (Φ.functor.obj c)
    x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
    ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
    this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
    ⊢ CategoryTheory.Zigzag (CategoryTheory.TwoSquare.CostructuredArrowDownwards.m …
  -/
  refine Zigzag.trans ?_ (Zigzag.trans this ?_)
    /-
      case intro.intro.intro.intro.refine_1
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
      this✝ : Nonempty (w.CostructuredArrowDownwards y)
      c : C₁
      g : Quiver.Hom X₂ (Φ.functor.obj c)
      x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
      fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
      ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
      this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
      ⊢ CategoryTheory.Zigzag (CategoryTheory.TwoSquare.CostructuredArrowDownwards.m …
    -/
  · exact Zigzag.of_hom (eqToHom (by aesop))
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
      this✝ : Nonempty (w.CostructuredArrowDownwards y)
      c : C₁
      g : Quiver.Hom X₂ (Φ.functor.obj c)
      x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
      fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
      ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
      this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
      ⊢ CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDerivabilit …
    -/
  · apply Zigzag.of_inv
    /-
      case intro.intro.intro.intro.refine_2.f
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
      this✝ : Nonempty (w.CostructuredArrowDownwards y)
      c : C₁
      g : Quiver.Hom X₂ (Φ.functor.obj c)
      x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
      fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
      ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
      this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
      ⊢ Quiver.Hom ((CategoryTheory.LocalizerMorphism.IsRightDerivabilityStructure.C …
    -/
    refine CostructuredArrow.homMk (StructuredArrow.homMk ρ.X₁.hom (by simp)) ?_
    /-
      case intro.intro.intro.intro.refine_2.f
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
      this✝ : Nonempty (w.CostructuredArrowDownwards y)
      c : C₁
      g : Quiver.Hom X₂ (Φ.functor.obj c)
      x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
      fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
      ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
      this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.TwoSquare.mk Φ.func …
    -/
    ext
    /-
      case intro.intro.intro.intro.refine_2.f.h
      C₁ : Type u_1
      C₂ : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝⁶ : CategoryTheory.Category.{u_5, u_2} C₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      inst✝⁵ : W₁.IsMultiplicative
      inst✝⁴ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
      inst✝³ : Φ.arrow.HasRightResolutions
      inst✝² : W₂.ContainsIdentities
      D : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} D
      L : CategoryTheory.Functor C₂ D
      inst✝ : L.IsLocalization W₂
      X₂ : C₂
      X₃ : D
      y : Quiver.Hom (L.obj X₂) X₃
      w : CategoryTheory.TwoSquare Φ.functor (Φ.functor.comp L) L (CategoryTheory.Fu …
      this✝ : Nonempty (w.CostructuredArrowDownwards y)
      c : C₁
      g : Quiver.Hom X₂ (Φ.functor.obj c)
      x : Quiver.Hom (L.obj (Φ.functor.obj c)) X₃
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map g) (CategoryTheory.Catego …
      fac : Eq (CategoryTheory.CategoryStruct.comp (L.map g) x) y
      ρ : Φ.arrow.RightResolution (CategoryTheory.Arrow.mk g) := Classical.arbitrary …
      this : CategoryTheory.Zigzag (((CategoryTheory.LocalizerMorphism.IsRightDeriva …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.TwoSquare.mk Φ.func …
    -/
    dsimp
    rw [← cancel_epi (isoOfHom L W₂ ρ.w.left ρ.hw.1).hom, isoOfHom_hom,
      isoOfHom_hom_inv_id_assoc, ← L.map_comp_assoc, Arrow.w_mk_right, Arrow.mk_hom,
      L.map_comp, assoc, isoOfHom_hom_inv_id_assoc, fac]


/-- If a localizer morphism `Φ` is a localized equivalence, then it is a right
derivability structure if the categories of right resolutions are connected and the
categories of right resolutions of arrows are nonempty. -/
lemma mk' [Φ.IsLocalizedEquivalence] : Φ.IsRightDerivabilityStructure := by
  rw [Φ.isRightDerivabilityStructure_iff (Φ.functor ⋙ W₂.Q) W₂.Q (𝟭 _)
    (Functor.rightUnitor _).symm, TwoSquare.guitartExact_iff_isConnected_downwards]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁴ : W₁.IsMultiplicative
    inst✝³ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝² : Φ.arrow.HasRightResolutions
    inst✝¹ : W₂.ContainsIdentities
    inst✝ : Φ.IsLocalizedEquivalence
    ⊢ ∀ {X₂ : C₂} {X₃ : W₂.Localization} (g : Quiver.Hom (W₂.Q.obj X₂) ((CategoryT …
  -/
  intro X₂ X₃ g
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C₁
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝⁴ : W₁.IsMultiplicative
    inst✝³ : ∀ (X₂ : C₂), CategoryTheory.IsConnected (Φ.RightResolution X₂)
    inst✝² : Φ.arrow.HasRightResolutions
    inst✝¹ : W₂.ContainsIdentities
    inst✝ : Φ.IsLocalizedEquivalence
    X₂ : C₂
    X₃ : W₂.Localization
    g : Quiver.Hom (W₂.Q.obj X₂) ((CategoryTheory.Functor.id W₂.Localization).obj  …
    ⊢ CategoryTheory.IsConnected (CategoryTheory.TwoSquare.CostructuredArrowDownwa …
  -/
  apply Constructor.isConnected
  /-
    🎉 no goals
  -/


