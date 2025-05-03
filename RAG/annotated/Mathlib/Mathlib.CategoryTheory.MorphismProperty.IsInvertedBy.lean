/-- If `P : MorphismProperty C` and `F : C ⥤ D`, then
`P.IsInvertedBy F` means that all morphisms in `P` are mapped by `F`
to isomorphisms in `D`. -/
def IsInvertedBy (P : MorphismProperty C) (F : C ⥤ D) : Prop :=
  ∀ ⦃X Y : C⦄ (f : X ⟶ Y) (_ : P f), IsIso (F.map f)


lemma of_le (P Q : MorphismProperty C) (F : C ⥤ D) (hQ : Q.IsInvertedBy F) (h : P ≤ Q) :
    P.IsInvertedBy F :=
  fun _ _ _ hf => hQ _ (h _ hf)


theorem of_comp {C₁ C₂ C₃ : Type*} [Category C₁] [Category C₂] [Category C₃]
    (W : MorphismProperty C₁) (F : C₁ ⥤ C₂) (hF : W.IsInvertedBy F) (G : C₂ ⥤ C₃) :
    W.IsInvertedBy (F ⋙ G) := fun X Y f hf => by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} C₂
    inst✝ : CategoryTheory.Category.{u_6, u_3} C₃
    W : CategoryTheory.MorphismProperty C₁
    F : CategoryTheory.Functor C₁ C₂
    hF : W.IsInvertedBy F
    G : CategoryTheory.Functor C₂ C₃
    X Y : C₁
    f : Quiver.Hom X Y
    hf : W f
    ⊢ CategoryTheory.IsIso ((F.comp G).map f)
  -/
  haveI := hF f hf
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} C₂
    inst✝ : CategoryTheory.Category.{u_6, u_3} C₃
    W : CategoryTheory.MorphismProperty C₁
    F : CategoryTheory.Functor C₁ C₂
    hF : W.IsInvertedBy F
    G : CategoryTheory.Functor C₂ C₃
    X Y : C₁
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (F.map f)
    ⊢ CategoryTheory.IsIso ((F.comp G).map f)
  -/
  dsimp
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} C₂
    inst✝ : CategoryTheory.Category.{u_6, u_3} C₃
    W : CategoryTheory.MorphismProperty C₁
    F : CategoryTheory.Functor C₁ C₂
    hF : W.IsInvertedBy F
    G : CategoryTheory.Functor C₂ C₃
    X Y : C₁
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (F.map f)
    ⊢ CategoryTheory.IsIso (G.map (F.map f))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem op {W : MorphismProperty C} {L : C ⥤ D} (h : W.IsInvertedBy L) : W.op.IsInvertedBy L.op :=
  fun X Y f hf => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C D
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    ⊢ CategoryTheory.IsIso (L.op.map f)
  -/
  haveI := h f.unop hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C D
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    this : CategoryTheory.IsIso (L.map f.unop)
    ⊢ CategoryTheory.IsIso (L.op.map f)
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C D
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    this : CategoryTheory.IsIso (L.map f.unop)
    ⊢ CategoryTheory.IsIso (L.map f.unop).op
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem rightOp {W : MorphismProperty C} {L : Cᵒᵖ ⥤ D} (h : W.op.IsInvertedBy L) :
    W.IsInvertedBy L.rightOp := fun X Y f hf => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) D
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    ⊢ CategoryTheory.IsIso (L.rightOp.map f)
  -/
  haveI := h f.op hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) D
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (L.map f.op)
    ⊢ CategoryTheory.IsIso (L.rightOp.map f)
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) D
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (L.map f.op)
    ⊢ CategoryTheory.IsIso (L.map f.op).op
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem leftOp {W : MorphismProperty C} {L : C ⥤ Dᵒᵖ} (h : W.IsInvertedBy L) :
    W.op.IsInvertedBy L.leftOp := fun X Y f hf => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C (Opposite D)
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    ⊢ CategoryTheory.IsIso (L.leftOp.map f)
  -/
  haveI := h f.unop hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C (Opposite D)
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    this : CategoryTheory.IsIso (L.map f.unop)
    ⊢ CategoryTheory.IsIso (L.leftOp.map f)
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor C (Opposite D)
    h : W.IsInvertedBy L
    X Y : Opposite C
    f : Quiver.Hom X Y
    hf : W.op f
    this : CategoryTheory.IsIso (L.map f.unop)
    ⊢ CategoryTheory.IsIso (L.map f.unop).unop
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem unop {W : MorphismProperty C} {L : Cᵒᵖ ⥤ Dᵒᵖ} (h : W.op.IsInvertedBy L) :
    W.IsInvertedBy L.unop := fun X Y f hf => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) (Opposite D)
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    ⊢ CategoryTheory.IsIso (L.unop.map f)
  -/
  haveI := h f.op hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) (Opposite D)
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (L.map f.op)
    ⊢ CategoryTheory.IsIso (L.unop.map f)
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    L : CategoryTheory.Functor (Opposite C) (Opposite D)
    h : W.op.IsInvertedBy L
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    this : CategoryTheory.IsIso (L.map f.op)
    ⊢ CategoryTheory.IsIso (L.map f.op).unop
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma prod {C₁ C₂ : Type*} [Category C₁] [Category C₂]
    {W₁ : MorphismProperty C₁} {W₂ : MorphismProperty C₂}
    {E₁ E₂ : Type*} [Category E₁] [Category E₂] {F₁ : C₁ ⥤ E₁} {F₂ : C₂ ⥤ E₂}
    (h₁ : W₁.IsInvertedBy F₁) (h₂ : W₂.IsInvertedBy F₂) :
    (W₁.prod W₂).IsInvertedBy (F₁.prod F₂) := fun _ _ f hf => by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_6, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E₁ : Type u_3
    E₂ : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} E₁
    inst✝ : CategoryTheory.Category.{u_8, u_4} E₂
    F₁ : CategoryTheory.Functor C₁ E₁
    F₂ : CategoryTheory.Functor C₂ E₂
    h₁ : W₁.IsInvertedBy F₁
    h₂ : W₂.IsInvertedBy F₂
    x✝¹ x✝ : Prod C₁ C₂
    f : Quiver.Hom x✝¹ x✝
    hf : W₁.prod W₂ f
    ⊢ CategoryTheory.IsIso ((F₁.prod F₂).map f)
  -/
  rw [isIso_prod_iff]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_6, u_2} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E₁ : Type u_3
    E₂ : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} E₁
    inst✝ : CategoryTheory.Category.{u_8, u_4} E₂
    F₁ : CategoryTheory.Functor C₁ E₁
    F₂ : CategoryTheory.Functor C₂ E₂
    h₁ : W₁.IsInvertedBy F₁
    h₂ : W₂.IsInvertedBy F₂
    x✝¹ x✝ : Prod C₁ C₂
    f : Quiver.Hom x✝¹ x✝
    hf : W₁.prod W₂ f
    ⊢ And (CategoryTheory.IsIso ((F₁.prod F₂).map f).1) (CategoryTheory.IsIso ((F₁ …
  -/
  exact ⟨h₁ _ hf.1, h₂ _ hf.2⟩
  /-
    🎉 no goals
  -/


lemma pi {J : Type w} {C : J → Type u} {D : J → Type u'}
    [∀ j, Category.{v} (C j)] [∀ j, Category.{v'} (D j)]
    (W : ∀ j, MorphismProperty (C j)) (F : ∀ j, C j ⥤ D j)
    (hF : ∀ j, (W j).IsInvertedBy (F j)) :
    (MorphismProperty.pi W).IsInvertedBy (Functor.pi F) := by
  /-
    J : Type w
    C : J → Type u
    D : J → Type u'
    inst✝¹ : (j : J) → CategoryTheory.Category.{v, u} (C j)
    inst✝ : (j : J) → CategoryTheory.Category.{v', u'} (D j)
    W : (j : J) → CategoryTheory.MorphismProperty (C j)
    F : (j : J) → CategoryTheory.Functor (C j) (D j)
    hF : ∀ (j : J), (W j).IsInvertedBy (F j)
    ⊢ (CategoryTheory.MorphismProperty.pi W).IsInvertedBy (CategoryTheory.Functor. …
  -/
  intro _ _ f hf
  /-
    J : Type w
    C : J → Type u
    D : J → Type u'
    inst✝¹ : (j : J) → CategoryTheory.Category.{v, u} (C j)
    inst✝ : (j : J) → CategoryTheory.Category.{v', u'} (D j)
    W : (j : J) → CategoryTheory.MorphismProperty (C j)
    F : (j : J) → CategoryTheory.Functor (C j) (D j)
    hF : ∀ (j : J), (W j).IsInvertedBy (F j)
    X✝ Y✝ : (j : J) → C j
    f : Quiver.Hom X✝ Y✝
    hf : CategoryTheory.MorphismProperty.pi W f
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Functor.pi F).map f)
  -/
  rw [isIso_pi_iff]
  /-
    J : Type w
    C : J → Type u
    D : J → Type u'
    inst✝¹ : (j : J) → CategoryTheory.Category.{v, u} (C j)
    inst✝ : (j : J) → CategoryTheory.Category.{v', u'} (D j)
    W : (j : J) → CategoryTheory.MorphismProperty (C j)
    F : (j : J) → CategoryTheory.Functor (C j) (D j)
    hF : ∀ (j : J), (W j).IsInvertedBy (F j)
    X✝ Y✝ : (j : J) → C j
    f : Quiver.Hom X✝ Y✝
    hf : CategoryTheory.MorphismProperty.pi W f
    ⊢ ∀ (i : J), CategoryTheory.IsIso ((CategoryTheory.Functor.pi F).map f i)
  -/
  intro j
  /-
    J : Type w
    C : J → Type u
    D : J → Type u'
    inst✝¹ : (j : J) → CategoryTheory.Category.{v, u} (C j)
    inst✝ : (j : J) → CategoryTheory.Category.{v', u'} (D j)
    W : (j : J) → CategoryTheory.MorphismProperty (C j)
    F : (j : J) → CategoryTheory.Functor (C j) (D j)
    hF : ∀ (j : J), (W j).IsInvertedBy (F j)
    X✝ Y✝ : (j : J) → C j
    f : Quiver.Hom X✝ Y✝
    hf : CategoryTheory.MorphismProperty.pi W f
    j : J
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Functor.pi F).map f j)
  -/
  exact hF j _ (hf j)
  /-
    🎉 no goals
  -/


/-- The full subcategory of `C ⥤ D` consisting of functors inverting morphisms in `W` -/
def FunctorsInverting (W : MorphismProperty C) (D : Type*) [Category D] :=
  FullSubcategory fun F : C ⥤ D => W.IsInvertedBy F


@[ext]
lemma FunctorsInverting.ext {W : MorphismProperty C} {F₁ F₂ : FunctorsInverting W D}
    (h : F₁.obj = F₂.obj) : F₁ = F₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F₁ F₂ : W.FunctorsInverting D
    h : Eq F₁.obj F₂.obj
    ⊢ Eq F₁ F₂
  -/
  cases F₁
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F₂ : W.FunctorsInverting D
    obj✝ : CategoryTheory.Functor C D
    property✝ : W.IsInvertedBy obj✝
    h : Eq { obj := obj✝, property := property✝ }.obj F₂.obj
    ⊢ Eq { obj := obj✝, property := property✝ } F₂
  -/
  cases F₂
  /-
    case mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    obj✝¹ : CategoryTheory.Functor C D
    property✝¹ : W.IsInvertedBy obj✝¹
    obj✝ : CategoryTheory.Functor C D
    property✝ : W.IsInvertedBy obj✝
    h : Eq { obj := obj✝¹, property := property✝¹ }.obj { obj := obj✝, property := …
    ⊢ Eq { obj := obj✝¹, property := property✝¹ } { obj := obj✝, property := prope …
  -/
  subst h
  /-
    case mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    obj✝ : CategoryTheory.Functor C D
    property✝¹ : W.IsInvertedBy obj✝
    property✝ : W.IsInvertedBy { obj := obj✝, property := property✝¹ }.obj
    ⊢ Eq { obj := obj✝, property := property✝¹ } { obj := { obj := obj✝, property  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (W : MorphismProperty C) (D : Type*) [Category D] : Category (FunctorsInverting W D) :=
  FullSubcategory.category _


@[ext]
lemma FunctorsInverting.hom_ext {W : MorphismProperty C} {F₁ F₂ : FunctorsInverting W D}
    {α β : F₁ ⟶ F₂} (h : α.app = β.app) : α = β :=
  NatTrans.ext h


/-- A constructor for `W.FunctorsInverting D` -/
def FunctorsInverting.mk {W : MorphismProperty C} {D : Type*} [Category D] (F : C ⥤ D)
    (hF : W.IsInvertedBy F) : W.FunctorsInverting D :=
  ⟨F, hF⟩


theorem IsInvertedBy.iff_of_iso (W : MorphismProperty C) {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂) :
    W.IsInvertedBy F₁ ↔ W.IsInvertedBy F₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F₁ F₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F₁ F₂
    ⊢ Iff (W.IsInvertedBy F₁) (W.IsInvertedBy F₂)
  -/
  dsimp [IsInvertedBy]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F₁ F₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F₁ F₂
    ⊢ Iff (∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), W f → CategoryTheory.IsIso (F₁.map f) …
  -/
  simp only [NatIso.isIso_map_iff e]
  /-
    🎉 no goals
  -/


@[simp]
lemma IsInvertedBy.isoClosure_iff (W : MorphismProperty C) (F : C ⥤ D) :
    W.isoClosure.IsInvertedBy F ↔ W.IsInvertedBy F := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    ⊢ Iff (W.isoClosure.IsInvertedBy F) (W.IsInvertedBy F)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      ⊢ W.isoClosure.IsInvertedBy F → W.IsInvertedBy F
    -/
  · intro h X Y f hf
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      h : W.isoClosure.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      hf : W f
      ⊢ CategoryTheory.IsIso (F.map f)
    -/
    exact h _ (W.le_isoClosure _ hf)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      ⊢ W.IsInvertedBy F → W.isoClosure.IsInvertedBy F
    -/
  · intro h X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      h : W.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : W f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
      ⊢ CategoryTheory.IsIso (F.map f)
    -/
    simp only [Arrow.iso_w' e, F.map_comp]
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      h : W.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : W f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (F.map e.inv.left)  …
    -/
    have := h _ hf'
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      W : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      h : W.IsInvertedBy F
      X Y : C
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : W f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
      this : CategoryTheory.IsIso (F.map f')
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (F.map e.inv.left)  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[simp]
lemma IsInvertedBy.iff_comp {C₁ C₂ C₃ : Type*} [Category C₁] [Category C₂] [Category C₃]
    (W : MorphismProperty C₁) (F : C₁ ⥤ C₂) (G : C₂ ⥤ C₃) [G.ReflectsIsomorphisms] :
    W.IsInvertedBy (F ⋙ G) ↔ W.IsInvertedBy F := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
    W : CategoryTheory.MorphismProperty C₁
    F : CategoryTheory.Functor C₁ C₂
    G : CategoryTheory.Functor C₂ C₃
    inst✝ : G.ReflectsIsomorphisms
    ⊢ Iff (W.IsInvertedBy (F.comp G)) (W.IsInvertedBy F)
  -/
  constructor
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
      W : CategoryTheory.MorphismProperty C₁
      F : CategoryTheory.Functor C₁ C₂
      G : CategoryTheory.Functor C₂ C₃
      inst✝ : G.ReflectsIsomorphisms
      ⊢ W.IsInvertedBy (F.comp G) → W.IsInvertedBy F
    -/
  · intro h X Y f hf
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
      W : CategoryTheory.MorphismProperty C₁
      F : CategoryTheory.Functor C₁ C₂
      G : CategoryTheory.Functor C₂ C₃
      inst✝ : G.ReflectsIsomorphisms
      h : W.IsInvertedBy (F.comp G)
      X Y : C₁
      f : Quiver.Hom X Y
      hf : W f
      ⊢ CategoryTheory.IsIso (F.map f)
    -/
    have : IsIso (G.map (F.map f)) := h _ hf
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
      W : CategoryTheory.MorphismProperty C₁
      F : CategoryTheory.Functor C₁ C₂
      G : CategoryTheory.Functor C₂ C₃
      inst✝ : G.ReflectsIsomorphisms
      h : W.IsInvertedBy (F.comp G)
      X Y : C₁
      f : Quiver.Hom X Y
      hf : W f
      this : CategoryTheory.IsIso (G.map (F.map f))
      ⊢ CategoryTheory.IsIso (F.map f)
    -/
    exact isIso_of_reflects_iso (F.map f) G
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
      W : CategoryTheory.MorphismProperty C₁
      F : CategoryTheory.Functor C₁ C₂
      G : CategoryTheory.Functor C₂ C₃
      inst✝ : G.ReflectsIsomorphisms
      ⊢ W.IsInvertedBy F → W.IsInvertedBy (F.comp G)
    -/
  · intro hF
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_6, u_3} C₃
      W : CategoryTheory.MorphismProperty C₁
      F : CategoryTheory.Functor C₁ C₂
      G : CategoryTheory.Functor C₂ C₃
      inst✝ : G.ReflectsIsomorphisms
      hF : W.IsInvertedBy F
      ⊢ W.IsInvertedBy (F.comp G)
    -/
    exact IsInvertedBy.of_comp W F hF G
    /-
      🎉 no goals
    -/


lemma IsInvertedBy.iff_le_inverseImage_isomorphisms (W : MorphismProperty C) (F : C ⥤ D) :
    W.IsInvertedBy F ↔ W ≤ (isomorphisms D).inverseImage F := Iff.rfl


lemma IsInvertedBy.iff_map_le_isomorphisms (W : MorphismProperty C) (F : C ⥤ D) :
    W.IsInvertedBy F ↔ W.map F ≤ isomorphisms D := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    W : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    ⊢ Iff (W.IsInvertedBy F) (LE.le (W.map F) (CategoryTheory.MorphismProperty.iso …
  -/
  rw [iff_le_inverseImage_isomorphisms, map_le_iff]
  /-
    🎉 no goals
  -/


lemma IsInvertedBy.map_iff {C₁ C₂ C₃ : Type*} [Category C₁] [Category C₂] [Category C₃]
    (W : MorphismProperty C₁) (F : C₁ ⥤ C₂) (G : C₂ ⥤ C₃) :
    (W.map F).IsInvertedBy G ↔ W.IsInvertedBy (F ⋙ G) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C₁
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} C₂
    inst✝ : CategoryTheory.Category.{u_6, u_3} C₃
    W : CategoryTheory.MorphismProperty C₁
    F : CategoryTheory.Functor C₁ C₂
    G : CategoryTheory.Functor C₂ C₃
    ⊢ Iff ((W.map F).IsInvertedBy G) (W.IsInvertedBy (F.comp G))
  -/
  simp only [IsInvertedBy.iff_map_le_isomorphisms, map_map]
  /-
    🎉 no goals
  -/


