/-- If `P` is closed under limits of shape `J` in `Comma L R`, then when `D` has
a limit in `Comma L R`, the forgetful functor creates this limit. -/
noncomputable def forgetCreatesLimitOfClosed
    (h : ClosedUnderLimitsOfShape J (fun f : Comma L R ↦ P f.hom))
    [HasLimit (D ⋙ forget L R P ⊤ ⊤)] :
    CreatesLimit D (forget L R P ⊤ ⊤) :=
  createsLimitOfFullyFaithfulOfIso
    (⟨limit (D ⋙ forget L R P ⊤ ⊤), h.limit fun j ↦ (D.obj j).prop⟩)
    (Iso.refl _)


/-- If `Comma L R` has limits of shape `J` and `Comma L R` is closed under limits of shape
`J`, then `forget L R P ⊤ ⊤` creates limits of shape `J`. -/
noncomputable def forgetCreatesLimitsOfShapeOfClosed [HasLimitsOfShape J (Comma L R)]
    (h : ClosedUnderLimitsOfShape J (fun f : Comma L R ↦ P f.hom)) :
    CreatesLimitsOfShape J (forget L R P ⊤ ⊤) where
  CreatesLimit := forgetCreatesLimitOfClosed _ _ h


lemma hasLimit_of_closedUnderLimitsOfShape
    (h : ClosedUnderLimitsOfShape J (fun f : Comma L R ↦ P f.hom))
    [HasLimit (D ⋙ forget L R P ⊤ ⊤)] :
    HasLimit D :=
  haveI : CreatesLimit D (forget L R P ⊤ ⊤) := forgetCreatesLimitOfClosed _ D h
  hasLimit_of_created D (forget L R P ⊤ ⊤)


lemma hasLimitsOfShape_of_closedUnderLimitsOfShape [HasLimitsOfShape J (Comma L R)]
    (h : ClosedUnderLimitsOfShape J (fun f : Comma L R ↦ P f.hom)) :
    HasLimitsOfShape J (P.Comma L R ⊤ ⊤) where
  has_limit _ := hasLimit_of_closedUnderLimitsOfShape _ _ h


lemma CostructuredArrow.closedUnderLimitsOfShape_discrete_empty [L.Faithful] [L.Full] {Y : A}
    [P.ContainsIdentities] [P.RespectsIso] :
    ClosedUnderLimitsOfShape (Discrete PEmpty.{1})
      (fun f : CostructuredArrow L (L.obj Y) ↦ P f.hom) := by
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} T
    P : CategoryTheory.MorphismProperty T
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} A
    L : CategoryTheory.Functor A T
    inst✝³ : L.Faithful
    inst✝² : L.Full
    Y : A
    inst✝¹ : P.ContainsIdentities
    inst✝ : P.RespectsIso
    ⊢ CategoryTheory.Limits.ClosedUnderLimitsOfShape (CategoryTheory.Discrete PEmp …
  -/
  rintro D c hc -
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} T
    P : CategoryTheory.MorphismProperty T
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} A
    L : CategoryTheory.Functor A T
    inst✝³ : L.Faithful
    inst✝² : L.Full
    Y : A
    inst✝¹ : P.ContainsIdentities
    inst✝ : P.RespectsIso
    D : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
    c : CategoryTheory.Limits.Cone D
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ P c.pt.hom
  -/
  have : D = Functor.empty _ := Functor.empty_ext' _ _
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} T
    P : CategoryTheory.MorphismProperty T
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} A
    L : CategoryTheory.Functor A T
    inst✝³ : L.Faithful
    inst✝² : L.Full
    Y : A
    inst✝¹ : P.ContainsIdentities
    inst✝ : P.RespectsIso
    D : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
    c : CategoryTheory.Limits.Cone D
    hc : CategoryTheory.Limits.IsLimit c
    this : Eq D (CategoryTheory.Functor.empty (CategoryTheory.CostructuredArrow L  …
    ⊢ P c.pt.hom
  -/
  subst this
  let e : c.pt ≅ CostructuredArrow.mk (𝟙 (L.obj Y)) :=
    hc.conePointUniqueUpToIso CostructuredArrow.mkIdTerminal
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} T
    P : CategoryTheory.MorphismProperty T
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} A
    L : CategoryTheory.Functor A T
    inst✝³ : L.Faithful
    inst✝² : L.Full
    Y : A
    inst✝¹ : P.ContainsIdentities
    inst✝ : P.RespectsIso
    c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
    hc : CategoryTheory.Limits.IsLimit c
    e : CategoryTheory.Iso c.pt (CategoryTheory.CostructuredArrow.mk (CategoryTheo …
    ⊢ P c.pt.hom
  -/
  rw [P.costructuredArrow_iso_iff e]
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} T
    P : CategoryTheory.MorphismProperty T
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} A
    L : CategoryTheory.Functor A T
    inst✝³ : L.Faithful
    inst✝² : L.Full
    Y : A
    inst✝¹ : P.ContainsIdentities
    inst✝ : P.RespectsIso
    c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
    hc : CategoryTheory.Limits.IsLimit c
    e : CategoryTheory.Iso c.pt (CategoryTheory.CostructuredArrow.mk (CategoryTheo …
    ⊢ P (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.id (L. …
  -/
  simpa using P.id_mem (L.obj Y)
  /-
    🎉 no goals
  -/


lemma Over.closedUnderLimitsOfShape_discrete_empty [P.ContainsIdentities] [P.RespectsIso] :
    ClosedUnderLimitsOfShape (Discrete PEmpty.{1}) (fun f : Over X ↦ P f.hom) :=
  CostructuredArrow.closedUnderLimitsOfShape_discrete_empty P


/-- Let `P` be stable under composition and base change. If `P` satisfies cancellation on the right,
the subcategory of `Over X` defined by `P` is closed under pullbacks.

Without the cancellation property, this does not in general. Consider for example
`P = Function.Surjective` on `Type`. -/
lemma Over.closedUnderLimitsOfShape_pullback [HasPullbacks T]
    [P.IsStableUnderComposition] [P.IsStableUnderBaseChange] [P.HasOfPostcompProperty P] :
    ClosedUnderLimitsOfShape WalkingCospan (fun f : Over X ↦ P f.hom) := by
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P : CategoryTheory.MorphismProperty T
    X : T
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderComposition
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.HasOfPostcompProperty P
    ⊢ CategoryTheory.Limits.ClosedUnderLimitsOfShape CategoryTheory.Limits.Walking …
  -/
  intro D c hc hf
  have h : IsPullback (c.π.app .left).left (c.π.app .right).left (D.map WalkingCospan.Hom.inl).left
        (D.map WalkingCospan.Hom.inr).left := IsPullback.of_isLimit_cone <|
    Limits.isLimitOfPreserves (CategoryTheory.Over.forget X) hc
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P : CategoryTheory.MorphismProperty T
    X : T
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderComposition
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.HasOfPostcompProperty P
    D : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan (CategoryTheory …
    c : CategoryTheory.Limits.Cone D
    hc : CategoryTheory.Limits.IsLimit c
    hf : ∀ (j : CategoryTheory.Limits.WalkingCospan), (fun f => P f.hom) (D.obj j)
    h : CategoryTheory.IsPullback (c.π.app CategoryTheory.Limits.WalkingCospan.lef …
    ⊢ P c.pt.hom
  -/
  rw [show c.pt.hom = (c.π.app .left).left ≫ (D.obj .left).hom by simp]
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P : CategoryTheory.MorphismProperty T
    X : T
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderComposition
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.HasOfPostcompProperty P
    D : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan (CategoryTheory …
    c : CategoryTheory.Limits.Cone D
    hc : CategoryTheory.Limits.IsLimit c
    hf : ∀ (j : CategoryTheory.Limits.WalkingCospan), (fun f => P f.hom) (D.obj j)
    h : CategoryTheory.IsPullback (c.π.app CategoryTheory.Limits.WalkingCospan.lef …
    ⊢ P (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walking …
  -/
  apply P.comp_mem _ _ (P.of_isPullback h.flip ?_) (hf _)
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P : CategoryTheory.MorphismProperty T
    X : T
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderComposition
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.HasOfPostcompProperty P
    D : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan (CategoryTheory …
    c : CategoryTheory.Limits.Cone D
    hc : CategoryTheory.Limits.IsLimit c
    hf : ∀ (j : CategoryTheory.Limits.WalkingCospan), (fun f => P f.hom) (D.obj j)
    h : CategoryTheory.IsPullback (c.π.app CategoryTheory.Limits.WalkingCospan.lef …
    ⊢ P (D.map CategoryTheory.Limits.WalkingCospan.Hom.inr).left
  -/
  exact P.of_postcomp _ (D.obj WalkingCospan.one).hom (hf .one) (by simpa using hf .right)
  /-
    🎉 no goals
  -/


noncomputable instance [P.ContainsIdentities] [P.RespectsIso] :
    CreatesLimitsOfShape (Discrete PEmpty.{1}) (Over.forget P ⊤ X) :=
  haveI : HasLimitsOfShape (Discrete PEmpty.{1}) (Comma (𝟭 T) (Functor.fromPUnit X)) := by
    /-
      T : Type u_1
      inst✝² : CategoryTheory.Category.{?u.28640, u_1} T
      P : CategoryTheory.MorphismProperty T
      X : T
      inst✝¹ : P.ContainsIdentities
      inst✝ : P.RespectsIso
      ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete PEmpty.{1})  …
    -/
    show HasLimitsOfShape _ (Over X)
    /-
      T : Type u_1
      inst✝² : CategoryTheory.Category.{?u.28640, u_1} T
      P : CategoryTheory.MorphismProperty T
      X : T
      inst✝¹ : P.ContainsIdentities
      inst✝ : P.RespectsIso
      ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete PEmpty.{1})  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  forgetCreatesLimitsOfShapeOfClosed P
    (Over.closedUnderLimitsOfShape_discrete_empty _)


variable {X} in
instance [P.ContainsIdentities] (Y : P.Over ⊤ X) :
    Unique (Y ⟶ Over.mk ⊤ (𝟙 X) (P.id_mem X)) where
             /-
               T : Type u_1
               inst✝¹ : CategoryTheory.Category.{?u.31457, u_1} T
               P : CategoryTheory.MorphismProperty T
               X : T
               inst✝ : P.ContainsIdentities
               Y : P.Over Top.top X
               ⊢ Eq (CategoryTheory.CategoryStruct.comp Y.hom (CategoryTheory.MorphismPropert …
             -/
             /-
               🎉 no goals
             -/
  default := Over.homMk Y.hom
             /-
               🎉 no goals
             -/
  uniq a := by
    /-
      T : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.31457, u_1} T
      P : CategoryTheory.MorphismProperty T
      X : T
      inst✝ : P.ContainsIdentities
      Y : P.Over Top.top X
      a : Quiver.Hom Y (CategoryTheory.MorphismProperty.Over.mk Top.top (CategoryThe …
      ⊢ Eq a Inhabited.default
    -/
    ext
      /-
        case h
        T : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.31457, u_1} T
        P : CategoryTheory.MorphismProperty T
        X : T
        inst✝ : P.ContainsIdentities
        Y : P.Over Top.top X
        a : Quiver.Hom Y (CategoryTheory.MorphismProperty.Over.mk Top.top (CategoryThe …
        ⊢ Eq a.left Inhabited.default.left
      -/
    · simp only [mk_left, Hom.hom_left, homMk_hom, Over.homMk_left]
      /-
        case h
        T : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.31457, u_1} T
        P : CategoryTheory.MorphismProperty T
        X : T
        inst✝ : P.ContainsIdentities
        Y : P.Over Top.top X
        a : Quiver.Hom Y (CategoryTheory.MorphismProperty.Over.mk Top.top (CategoryThe …
        ⊢ Eq a.left Y.hom
      -/
      rw [← Over.w a]
      /-
        case h
        T : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.31457, u_1} T
        P : CategoryTheory.MorphismProperty T
        X : T
        inst✝ : P.ContainsIdentities
        Y : P.Over Top.top X
        a : Quiver.Hom Y (CategoryTheory.MorphismProperty.Over.mk Top.top (CategoryThe …
        ⊢ Eq a.left (CategoryTheory.CategoryStruct.comp a.left (CategoryTheory.Morphis …
      -/
      simp only [mk_left, Functor.const_obj_obj, Hom.hom_left, mk_hom, Category.comp_id]
      /-
        🎉 no goals
      -/


/-- `X ⟶ X` is the terminal object of `P.Over ⊤ X`. -/
def mkIdTerminal [P.ContainsIdentities] :
    IsTerminal (Over.mk ⊤ (𝟙 X) (P.id_mem X)) :=
  IsTerminal.ofUnique _


instance [P.ContainsIdentities] : HasTerminal (P.Over ⊤ X) :=
  let h : IsTerminal (Over.mk ⊤ (𝟙 X) (P.id_mem X)) := Over.mkIdTerminal P X
  h.hasTerminal


/-- If `P` is stable under composition, base change and satisfies post-cancellation,
`Over.forget P ⊤ X` creates pullbacks. -/
noncomputable instance createsLimitsOfShape_walkingCospan [HasPullbacks T]
    [P.IsStableUnderComposition] [P.IsStableUnderBaseChange] [P.HasOfPostcompProperty P] :
    CreatesLimitsOfShape WalkingCospan (Over.forget P ⊤ X) :=
  haveI : HasLimitsOfShape WalkingCospan (Comma (𝟭 T) (Functor.fromPUnit X)) :=
    inferInstanceAs <| HasLimitsOfShape WalkingCospan (Over X)
  forgetCreatesLimitsOfShapeOfClosed P
    (Over.closedUnderLimitsOfShape_pullback P)


/-- If `P` is stable under composition, base change and satisfies post-cancellation,
`P.Over ⊤ X` has pullbacks -/
instance (priority := 900) hasPullbacks [HasPullbacks T] [P.IsStableUnderComposition]
    [P.IsStableUnderBaseChange] [P.HasOfPostcompProperty P] : HasPullbacks (P.Over ⊤ X) :=
  haveI : HasLimitsOfShape WalkingCospan (Comma (𝟭 T) (Functor.fromPUnit X)) :=
    inferInstanceAs <| HasLimitsOfShape WalkingCospan (Over X)
  hasLimitsOfShape_of_closedUnderLimitsOfShape P
    (Over.closedUnderLimitsOfShape_pullback P)


