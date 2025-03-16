/-- If a span is the pullback span over the terminal object, then it is a binary product. -/
def isBinaryProductOfIsTerminalIsPullback (F : Discrete WalkingPair ⥤ C) (c : Cone F) {X : C}
    (hX : IsTerminal X) (f : F.obj ⟨WalkingPair.left⟩ ⟶ X) (g : F.obj ⟨WalkingPair.right⟩ ⟶ X)
    (hc : IsLimit
      (PullbackCone.mk (c.π.app ⟨WalkingPair.left⟩) (c.π.app ⟨WalkingPair.right⟩ : _) <|
        hX.hom_ext (_ ≫ f) (_ ≫ g))) : IsLimit c where
  lift s :=
    hc.lift
      (PullbackCone.mk (s.π.app ⟨WalkingPair.left⟩) (s.π.app ⟨WalkingPair.right⟩) (hX.hom_ext _ _))
  fac _ j :=
    Discrete.casesOn j fun j =>
      WalkingPair.casesOn j (hc.fac _ WalkingCospan.left) (hc.fac _ WalkingCospan.right)
  uniq s m J := by
    let c' :=
      PullbackCone.mk (m ≫ c.π.app ⟨WalkingPair.left⟩) (m ≫ c.π.app ⟨WalkingPair.right⟩ : _)
        (hX.hom_ext (_ ≫ f) (_ ≫ g))
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cone F
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      f : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.left }) X
      g : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.right }) X
      hc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (c.π …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PullbackCone f g := CategoryTheory.Limits.PullbackC …
      ⊢ Eq m ((fun s => hc.lift (CategoryTheory.Limits.PullbackCone.mk (s.π.app { as …
    -/
    dsimp; rw [← J, ← J]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cone F
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      f : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.left }) X
      g : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.right }) X
      hc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (c.π …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PullbackCone f g := CategoryTheory.Limits.PullbackC …
      ⊢ Eq m (hc.lift (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Categor …
    -/
    apply hc.hom_ext
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cone F
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      f : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.left }) X
      g : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.right }) X
      hc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (c.π …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PullbackCone f g := CategoryTheory.Limits.PullbackC …
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryStru …
    -/
    rintro (_ | (_ | _)) <;> simp only [PullbackCone.mk_π_app_one, PullbackCone.mk_π_app]
    exacts [(Category.assoc _ _ _).symm.trans (hc.fac_assoc c' WalkingCospan.left f).symm,
      (hc.fac c' WalkingCospan.left).symm, (hc.fac c' WalkingCospan.right).symm]


/-- The pullback over the terminal object is the product -/
def isProductOfIsTerminalIsPullback {W X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) (h : W ⟶ X) (k : W ⟶ Y)
    (H₁ : IsTerminal Z)
    (H₂ : IsLimit (PullbackCone.mk _ _ (show h ≫ f = k ≫ g from H₁.hom_ext _ _))) :
    IsLimit (BinaryFan.mk h k) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk h k ⋯)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
  -/
  apply isBinaryProductOfIsTerminalIsPullback _ _ H₁
  /-
    case hc
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk h k ⋯)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk ((Categ …
  -/
  exact H₂
  /-
    🎉 no goals
  -/


/-- The product is the pullback over the terminal object. -/
def isPullbackOfIsTerminalIsProduct {W X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) (h : W ⟶ X) (k : W ⟶ Y)
    (H₁ : IsTerminal Z) (H₂ : IsLimit (BinaryFan.mk h k)) :
    IsLimit (PullbackCone.mk _ _ (show h ≫ f = k ≫ g from H₁.hom_ext _ _)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk h k ⋯)
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    ⊢ (s : CategoryTheory.Limits.PullbackCone f g) → Subtype fun l => And (Eq (Cat …
  -/
  intro s
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
  -/
  use H₂.lift (BinaryFan.mk s.fst s.snd)
  /-
    case property
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (H₂.lift (CategoryTheory.Limits. …
  -/
  use H₂.fac (BinaryFan.mk s.fst s.snd) ⟨WalkingPair.left⟩
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (H₂.lift (CategoryTheory.Limits. …
  -/
  use H₂.fac (BinaryFan.mk s.fst s.snd) ⟨WalkingPair.right⟩
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ {m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk h k ⋯).pt}, Eq …
  -/
  intro m h₁ h₂
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk h k ⋯).pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    ⊢ Eq m (H₂.lift (CategoryTheory.Limits.BinaryFan.mk s.fst s.snd))
  -/
  apply H₂.hom_ext
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsTerminal Z
    H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
    s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk h k ⋯).pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
  rintro ⟨⟨⟩⟩
    /-
      case right.mk.left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      h : Quiver.Hom W X
      k : Quiver.Hom W Y
      H₁ : CategoryTheory.Limits.IsTerminal Z
      H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk h k ⋯).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limits.BinaryFan.m …
    -/
  · exact h₁.trans (H₂.fac (BinaryFan.mk s.fst s.snd) ⟨WalkingPair.left⟩).symm
    /-
      🎉 no goals
    -/
    /-
      case right.mk.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      h : Quiver.Hom W X
      k : Quiver.Hom W Y
      H₁ : CategoryTheory.Limits.IsTerminal Z
      H₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk h k)
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk h k ⋯).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limits.BinaryFan.m …
    -/
  · exact h₂.trans (H₂.fac (BinaryFan.mk s.fst s.snd) ⟨WalkingPair.right⟩).symm
    /-
      🎉 no goals
    -/


/-- Any category with pullbacks and a terminal object has a limit cone for each walking pair. -/
noncomputable def limitConeOfTerminalAndPullbacks [HasTerminal C] [HasPullbacks C]
    (F : Discrete WalkingPair ⥤ C) : LimitCone F where
  cone :=
    { pt :=
        pullback (terminal.from (F.obj ⟨WalkingPair.left⟩))
          (terminal.from (F.obj ⟨WalkingPair.right⟩))
      π :=
        Discrete.natTrans fun x =>
          Discrete.casesOn x fun x => WalkingPair.casesOn x (pullback.fst _ _) (pullback.snd _ _) }
  isLimit :=
    isBinaryProductOfIsTerminalIsPullback F _ terminalIsTerminal _ _ (pullbackIsPullback _ _)


/-- Any category with pullbacks and terminal object has binary products. -/
theorem hasBinaryProducts_of_hasTerminal_and_pullbacks [HasTerminal C] [HasPullbacks C] :
    HasBinaryProducts C :=
  { has_limit := fun F => HasLimit.mk (limitConeOfTerminalAndPullbacks F) }


/-- A functor that preserves terminal objects and pullbacks preserves binary products. -/
lemma preservesBinaryProducts_of_preservesTerminal_and_pullbacks [HasTerminal C]
    [HasPullbacks C] [PreservesLimitsOfShape (Discrete.{0} PEmpty) F]
    [PreservesLimitsOfShape WalkingCospan F] : PreservesLimitsOfShape (Discrete WalkingPair) F :=
  ⟨fun {K} =>
    preservesLimit_of_preserves_limit_cone (limitConeOfTerminalAndPullbacks K).2
      (by
        apply
          isBinaryProductOfIsTerminalIsPullback _ _ (isLimitOfHasTerminalOfPreservesLimit F)
        /-
          case hc
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} D
          F : CategoryTheory.Functor C D
          inst✝³ : CategoryTheory.Limits.HasTerminal C
          inst✝² : CategoryTheory.Limits.HasPullbacks C
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
          inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
          K : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk ((F.map …
        -/
        apply isLimitOfHasPullbackOfPreservesLimit)⟩
        /-
          🎉 no goals
        -/


/-- In a category with a terminal object and pullbacks,
a product of objects `X` and `Y` is isomorphic to a pullback. -/
noncomputable def prodIsoPullback [HasTerminal C] [HasPullbacks C] (X Y : C)
    [HasBinaryProduct X Y] : X ⨯ Y ≅ pullback (terminal.from X) (terminal.from Y) :=
  limit.isoLimitCone (limitConeOfTerminalAndPullbacks _)


@[reassoc (attr := simp)]
lemma prodIsoPullback_hom_fst [HasTerminal C] [HasPullbacks C] (X Y : C)
    [HasBinaryProduct X Y] : (prodIsoPullback X Y).hom ≫ pullback.fst _ _ = prod.fst :=
  limit.isoLimitCone_hom_π (limitConeOfTerminalAndPullbacks _) ⟨.left⟩


@[reassoc (attr := simp)]
lemma prodIsoPullback_hom_snd [HasTerminal C] [HasPullbacks C] (X Y : C)
    [HasBinaryProduct X Y] : (prodIsoPullback X Y).hom ≫ pullback.snd _ _ = prod.snd :=
  limit.isoLimitCone_hom_π (limitConeOfTerminalAndPullbacks _) ⟨.right⟩


@[reassoc (attr := simp)]
lemma prodIsoPullback_inv_fst [HasTerminal C] [HasPullbacks C] (X Y : C)
    [HasBinaryProduct X Y] : (prodIsoPullback X Y).inv ≫ prod.fst = pullback.fst _ _ :=
  limit.isoLimitCone_inv_π (limitConeOfTerminalAndPullbacks _) ⟨.left⟩


@[reassoc (attr := simp)]
lemma prodIsoPullback_inv_snd [HasTerminal C] [HasPullbacks C] (X Y : C)
    [HasBinaryProduct X Y] : (prodIsoPullback X Y).inv ≫ prod.snd = pullback.snd _ _ :=
  limit.isoLimitCone_inv_π (limitConeOfTerminalAndPullbacks _) ⟨.right⟩


/-- If a cospan is the pushout cospan under the initial object, then it is a binary coproduct. -/
def isBinaryCoproductOfIsInitialIsPushout (F : Discrete WalkingPair ⥤ C) (c : Cocone F) {X : C}
    (hX : IsInitial X) (f : X ⟶ F.obj ⟨WalkingPair.left⟩) (g : X ⟶ F.obj ⟨WalkingPair.right⟩)
    (hc :
      IsColimit
        (PushoutCocone.mk (c.ι.app ⟨WalkingPair.left⟩) (c.ι.app ⟨WalkingPair.right⟩ : _) <|
          hX.hom_ext (f ≫ _) (g ≫ _))) :
    IsColimit c where
  desc s :=
    hc.desc
      (PushoutCocone.mk (s.ι.app ⟨WalkingPair.left⟩) (s.ι.app ⟨WalkingPair.right⟩) (hX.hom_ext _ _))
  fac _ j :=
    Discrete.casesOn j fun j =>
      WalkingPair.casesOn j (hc.fac _ WalkingSpan.left) (hc.fac _ WalkingSpan.right)
  uniq s m J := by
    let c' :=
      PushoutCocone.mk (c.ι.app ⟨WalkingPair.left⟩ ≫ m) (c.ι.app ⟨WalkingPair.right⟩ ≫ m)
        (hX.hom_ext (f ≫ _) (g ≫ _))
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cocone F
      X : C
      hX : CategoryTheory.Limits.IsInitial X
      f : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.left })
      g : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.right })
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ( …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PushoutCocone f g := CategoryTheory.Limits.PushoutC …
      ⊢ Eq m ((fun s => hc.desc (CategoryTheory.Limits.PushoutCocone.mk (s.ι.app { a …
    -/
    dsimp; rw [← J, ← J]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cocone F
      X : C
      hX : CategoryTheory.Limits.IsInitial X
      f : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.left })
      g : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.right })
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ( …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PushoutCocone f g := CategoryTheory.Limits.PushoutC …
      ⊢ Eq m (hc.desc (CategoryTheory.Limits.PushoutCocone.mk (CategoryTheory.Catego …
    -/
    apply hc.hom_ext
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cocone F
      X : C
      hX : CategoryTheory.Limits.IsInitial X
      f : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.left })
      g : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.right })
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ( …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PushoutCocone f g := CategoryTheory.Limits.PushoutC …
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
    -/
    rintro (_ | (_ | _)) <;>
      /-
        case none
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        F✝ : CategoryTheory.Functor C D
        F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
        c : CategoryTheory.Limits.Cocone F
        X : C
        hX : CategoryTheory.Limits.IsInitial X
        f : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.left })
        g : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.right })
        hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ( …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom c.pt s.pt
        J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        c' : CategoryTheory.Limits.PushoutCocone f g := CategoryTheory.Limits.PushoutC …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.PushoutCocone …
      -/
      simp only [PushoutCocone.mk_ι_app_zero, PushoutCocone.mk_ι_app, Category.assoc]
    /-
      case none
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ : CategoryTheory.Functor C D
      F : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
      c : CategoryTheory.Limits.Cocone F
      X : C
      hX : CategoryTheory.Limits.IsInitial X
      f : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.left })
      g : Quiver.Hom X (F.obj { as := CategoryTheory.Limits.WalkingPair.right })
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ( …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      J : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      c' : CategoryTheory.Limits.PushoutCocone f g := CategoryTheory.Limits.PushoutC …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    on_goal 1 => congr 1
    exacts [(hc.fac c' WalkingSpan.left).symm, (hc.fac c' WalkingSpan.left).symm,
      (hc.fac c' WalkingSpan.right).symm]


/-- The pushout under the initial object is the coproduct -/
def isCoproductOfIsInitialIsPushout {W X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) (h : W ⟶ X) (k : W ⟶ Y)
    (H₁ : IsInitial W)
    (H₂ : IsColimit (PushoutCocone.mk _ _ (show h ≫ f = k ≫ g from H₁.hom_ext _ _))) :
    IsColimit (BinaryCofan.mk f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk f …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
  -/
  apply isBinaryCoproductOfIsInitialIsPushout _ _ H₁
  /-
    case hc
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk f …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ((Ca …
  -/
  exact H₂
  /-
    🎉 no goals
  -/


/-- The coproduct is the pushout under the initial object. -/
def isPushoutOfIsInitialIsCoproduct {W X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) (h : W ⟶ X) (k : W ⟶ Y)
    (H₁ : IsInitial W) (H₂ : IsColimit (BinaryCofan.mk f g)) :
    IsColimit (PushoutCocone.mk _ _ (show h ≫ f = k ≫ g from H₁.hom_ext _ _)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk f g ⋯)
  -/
  apply PushoutCocone.isColimitAux'
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    ⊢ (s : CategoryTheory.Limits.PushoutCocone h k) → Subtype fun l => And (Eq (Ca …
  -/
  intro s
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
  -/
  use H₂.desc (BinaryCofan.mk s.inl s.inr)
  /-
    case property
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCo …
  -/
  use H₂.fac (BinaryCofan.mk s.inl s.inr) ⟨WalkingPair.left⟩
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCo …
  -/
  use H₂.fac (BinaryCofan.mk s.inl s.inr) ⟨WalkingPair.right⟩
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    ⊢ ∀ {m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk f g ⋯).pt s.pt}, E …
  -/
  intro m h₁ h₂
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk f g ⋯).pt s.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
    ⊢ Eq m (H₂.desc (CategoryTheory.Limits.BinaryCofan.mk s.inl s.inr))
  -/
  apply H₂.hom_ext
  /-
    case right
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    H₁ : CategoryTheory.Limits.IsInitial W
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
    s : CategoryTheory.Limits.PushoutCocone h k
    m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk f g ⋯).pt s.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
  rintro ⟨⟨⟩⟩
    /-
      case right.mk.left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      h : Quiver.Hom W X
      k : Quiver.Hom W Y
      H₁ : CategoryTheory.Limits.IsInitial W
      H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
      s : CategoryTheory.Limits.PushoutCocone h k
      m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk f g ⋯).pt s.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.BinaryCofan.m …
    -/
  · exact h₁.trans (H₂.fac (BinaryCofan.mk s.inl s.inr) ⟨WalkingPair.left⟩).symm
    /-
      🎉 no goals
    -/
    /-
      case right.mk.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      h : Quiver.Hom W X
      k : Quiver.Hom W Y
      H₁ : CategoryTheory.Limits.IsInitial W
      H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f g)
      s : CategoryTheory.Limits.PushoutCocone h k
      m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk f g ⋯).pt s.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.BinaryCofan.m …
    -/
  · exact h₂.trans (H₂.fac (BinaryCofan.mk s.inl s.inr) ⟨WalkingPair.right⟩).symm
    /-
      🎉 no goals
    -/


/-- Any category with pushouts and an initial object has a colimit cocone for each walking pair. -/
noncomputable def colimitCoconeOfInitialAndPushouts [HasInitial C] [HasPushouts C]
    (F : Discrete WalkingPair ⥤ C) : ColimitCocone F where
  cocone :=
    { pt := pushout (initial.to (F.obj ⟨WalkingPair.left⟩)) (initial.to (F.obj ⟨WalkingPair.right⟩))
      ι :=
        Discrete.natTrans fun x =>
          Discrete.casesOn x fun x => WalkingPair.casesOn x (pushout.inl _ _) (pushout.inr _ _) }
  isColimit := isBinaryCoproductOfIsInitialIsPushout F _ initialIsInitial _ _ (pushoutIsPushout _ _)


/-- Any category with pushouts and initial object has binary coproducts. -/
theorem hasBinaryCoproducts_of_hasInitial_and_pushouts [HasInitial C] [HasPushouts C] :
    HasBinaryCoproducts C :=
  { has_colimit := fun F => HasColimit.mk (colimitCoconeOfInitialAndPushouts F) }


/-- A functor that preserves initial objects and pushouts preserves binary coproducts. -/
lemma preservesBinaryCoproducts_of_preservesInitial_and_pushouts [HasInitial C]
    [HasPushouts C] [PreservesColimitsOfShape (Discrete.{0} PEmpty) F]
    [PreservesColimitsOfShape WalkingSpan F] : PreservesColimitsOfShape (Discrete WalkingPair) F :=
  ⟨fun {K} =>
    preservesColimit_of_preserves_colimit_cocone (colimitCoconeOfInitialAndPushouts K).2 (by
      apply
        isBinaryCoproductOfIsInitialIsPushout _ _
          (isColimitOfHasInitialOfPreservesColimit F)
      /-
        case hc
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasInitial C
        inst✝² : CategoryTheory.Limits.HasPushouts C
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
        K : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ((F. …
      -/
      apply isColimitOfHasPushoutOfPreservesColimit)⟩
      /-
        🎉 no goals
      -/


/-- In a category with an initial object and pushouts,
a coproduct of objects `X` and `Y` is isomorphic to a pushout. -/
noncomputable def coprodIsoPushout [HasInitial C] [HasPushouts C] (X Y : C)
    [HasBinaryCoproduct X Y] : X ⨿ Y ≅ pushout (initial.to X) (initial.to Y) :=
  colimit.isoColimitCocone (colimitCoconeOfInitialAndPushouts _)


@[reassoc (attr := simp)]
lemma inl_coprodIsoPushout_hom [HasInitial C] [HasPushouts C] (X Y : C)
    [HasBinaryCoproduct X Y] : coprod.inl ≫ (coprodIsoPushout X Y).hom = pushout.inl _ _ :=
  colimit.isoColimitCocone_ι_hom (colimitCoconeOfInitialAndPushouts _) _


@[reassoc (attr := simp)]
lemma inr_coprodIsoPushout_hom [HasInitial C] [HasPushouts C] (X Y : C)
    [HasBinaryCoproduct X Y] : coprod.inr ≫ (coprodIsoPushout X Y).hom = pushout.inr _ _ :=
  colimit.isoColimitCocone_ι_hom (colimitCoconeOfInitialAndPushouts _) _


@[reassoc (attr := simp)]
lemma inl_coprodIsoPushout_inv [HasInitial C] [HasPushouts C] (X Y : C)
    [HasBinaryCoproduct X Y] : pushout.inl _ _ ≫ (coprodIsoPushout X Y).inv = coprod.inl :=
  colimit.isoColimitCocone_ι_inv (colimitCoconeOfInitialAndPushouts (pair X Y)) ⟨.left⟩


@[reassoc (attr := simp)]
lemma inr_coprodIsoPushout_inv [HasInitial C] [HasPushouts C] (X Y : C)
    [HasBinaryCoproduct X Y] : pushout.inr _ _ ≫ (coprodIsoPushout X Y).inv = coprod.inr :=
  colimit.isoColimitCocone_ι_inv (colimitCoconeOfInitialAndPushouts (pair X Y)) ⟨.right⟩

