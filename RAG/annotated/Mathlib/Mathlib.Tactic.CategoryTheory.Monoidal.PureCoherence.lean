local infixr:81 " ◁ " => MonoidalCategory.whiskerLeftIso

local infixl:81 " ▷ " => MonoidalCategory.whiskerRightIso


/-- The composition of the normalizing isomorphisms `η_f : p ⊗ f ≅ pf` and `η_g : pf ⊗ g ≅ pfg`. -/
abbrev normalizeIsoComp {p f g pf pfg : C} (η_f : p ⊗ f ≅ pf) (η_g : pf ⊗ g ≅ pfg) :=
  (α_ _ _ _).symm ≪≫ whiskerRightIso η_f g ≪≫ η_g


theorem naturality_associator {p f g h pf pfg pfgh : C}
    (η_f : p ⊗ f ≅ pf) (η_g : pf ⊗ g ≅ pfg) (η_h : pfg ⊗ h ≅ pfgh) :
    p ◁ (α_ f g h) ≪≫ normalizeIsoComp η_f (normalizeIsoComp η_g η_h) =
    normalizeIsoComp (normalizeIsoComp η_f η_g) η_h :=
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.MonoidalCategory C
                p f g h pf pfg pfgh : C
                η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
                η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf g …
                η_h : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pfg  …
                ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_leftUnitor {p f pf : C} (η_f : p ⊗ f ≅ pf) :
    p ◁ (λ_ f) ≪≫ η_f = normalizeIsoComp (ρ_ p) η_f :=
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.MonoidalCategory C
                p f pf : C
                η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
                ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_rightUnitor {p f pf : C} (η_f : p ⊗ f ≅ pf) :
    p ◁ (ρ_ f) ≪≫ η_f = normalizeIsoComp η_f (ρ_ pf) :=
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.MonoidalCategory C
                p f pf : C
                η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
                ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_id {p f pf : C} (η_f : p ⊗ f ≅ pf) :
    p ◁ Iso.refl f ≪≫ η_f = η_f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f pf : C
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Iso.re …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem naturality_comp {p f g h pf : C} {η : f ≅ g} {θ : g ≅ h}
    (η_f : p ⊗ f ≅ pf) (η_g : p ⊗ g ≅ pf) (η_h : p ⊗ h ≅ pf)
    (ih_η : p ◁ η ≪≫ η_g = η_f) (ih_θ : p ◁ θ ≪≫ η_h = η_g) :
    p ◁ (η ≪≫ θ) ≪≫ η_h = η_f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf : C
    η : CategoryTheory.Iso f g
    θ : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    η_h : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p h) …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ih_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p θ).trans η_h) η_g
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (η.trans θ)).trans η_h …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem naturality_whiskerLeft {p f g h pf pfg : C} {η : g ≅ h}
    (η_f : p ⊗ f ≅ pf) (η_fg : pf ⊗ g ≅ pfg) (η_fh : (pf ⊗ h) ≅ pfg)
    (ih_η : pf ◁ η ≪≫ η_fh = η_fg) :
    p ◁ (f ◁ η) ≪≫ normalizeIsoComp η_f η_fh = normalizeIsoComp η_f η_fg := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfg : C
    η : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_fg : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf η).trans η_fh) η …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  rw [← ih_η]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfg : C
    η : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_fg : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf η).trans η_fh) η …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  apply Iso.ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfg : C
    η : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_fg : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf η).trans η_fh) η …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  simp [← whisker_exchange_assoc]
  /-
    🎉 no goals
  -/


theorem naturality_whiskerRight {p f g h pf pfh : C} {η : f ≅ g}
    (η_f : p ⊗ f ≅ pf) (η_g : p ⊗ g ≅ pf) (η_fh : (pf ⊗ h) ≅ pfh)
    (ih_η : p ◁ η ≪≫ η_g = η_f) :
    p ◁ (η ▷ h) ≪≫ normalizeIsoComp η_g η_fh = normalizeIsoComp η_f η_fh := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfh : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  rw [← ih_η]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfh : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  apply Iso.ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g h pf pfh : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    η_fh : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf  …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem naturality_tensorHom {p f₁ g₁ f₂ g₂ pf₁ pf₁f₂ : C} {η : f₁ ≅ g₁} {θ : f₂ ≅ g₂}
    (η_f₁ : p ⊗ f₁ ≅ pf₁) (η_g₁ : p ⊗ g₁ ≅ pf₁) (η_f₂ : pf₁ ⊗ f₂ ≅ pf₁f₂) (η_g₂ : pf₁ ⊗ g₂ ≅ pf₁f₂)
    (ih_η : p ◁ η ≪≫ η_g₁ = η_f₁)
    (ih_θ : pf₁ ◁ θ ≪≫ η_g₂ = η_f₂) :
    p ◁ (η ⊗ θ) ≪≫ normalizeIsoComp η_g₁ η_g₂ = normalizeIsoComp η_f₁ η_f₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f₁ g₁ f₂ g₂ pf₁ pf₁f₂ : C
    η : CategoryTheory.Iso f₁ g₁
    θ : CategoryTheory.Iso f₂ g₂
    η_f₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f …
    η_g₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g …
    η_f₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
    η_g₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g₁) η_f₁
    ih_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf₁ θ).trans η_g₂)  …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
  -/
  rw [tensorIso_def]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f₁ g₁ f₂ g₂ pf₁ pf₁f₂ : C
    η : CategoryTheory.Iso f₁ g₁
    θ : CategoryTheory.Iso f₂ g₂
    η_f₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f …
    η_g₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g …
    η_f₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
    η_g₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
    ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g₁) η_f₁
    ih_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf₁ θ).trans η_g₂)  …
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p ((CategoryTheory.Monoi …
  -/
  apply naturality_comp
    /-
      case ih_η
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      p f₁ g₁ f₂ g₂ pf₁ pf₁f₂ : C
      η : CategoryTheory.Iso f₁ g₁
      θ : CategoryTheory.Iso f₂ g₂
      η_f₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f …
      η_g₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g …
      η_f₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
      η_g₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
      ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g₁) η_f₁
      ih_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf₁ θ).trans η_g₂)  …
      ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
    -/
  · apply naturality_whiskerRight _ _ _ ih_η
    /-
      🎉 no goals
    -/
    /-
      case ih_θ
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      p f₁ g₁ f₂ g₂ pf₁ pf₁f₂ : C
      η : CategoryTheory.Iso f₁ g₁
      θ : CategoryTheory.Iso f₂ g₂
      η_f₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f …
      η_g₁ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g …
      η_f₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
      η_g₂ : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj pf₁ …
      ih_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g₁) η_f₁
      ih_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso pf₁ θ).trans η_g₂)  …
      ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p (CategoryTheory.Monoid …
    -/
  · apply naturality_whiskerLeft _ _ _ ih_θ
    /-
      🎉 no goals
    -/


theorem naturality_inv {p f g pf : C} {η : f ≅ g}
    (η_f : p ⊗ f ≅ pf) (η_g : p ⊗ g ≅ pf) (ih : p ◁ η ≪≫ η_g = η_f) :
    p ◁ η.symm ≪≫ η_f = η_g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g pf : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    ih : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η.symm).trans η_f) η_g
  -/
  rw [← ih]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g pf : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    ih : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η.symm).trans ((Catego …
  -/
  apply Iso.ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    p f g pf : C
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p f) …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj p g) …
    ih : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso p η.symm).trans ((Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : MonadNormalizeNaturality MonoidalM where
  mkNaturalityAssociator p pf pfg pfgh f g h η_f η_g η_h := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have pf : Q($ctx.C) := pf.e.e
    have pfg : Q($ctx.C) := pfg.e.e
    have pfgh : Q($ctx.C) := pfgh.e.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    have η_g : Q($pf ⊗ $g ≅ $pfg) := η_g.e
    have η_h : Q($pfg ⊗ $h ≅ $pfgh) := η_h.e
    return q(naturality_associator $η_f $η_g $η_h)
  mkNaturalityLeftUnitor p pf f η_f := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have pf : Q($ctx.C) := pf.e.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    return q(naturality_leftUnitor $η_f)
  mkNaturalityRightUnitor p pf f η_f := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have pf : Q($ctx.C) := pf.e.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    return q(naturality_rightUnitor $η_f)
  mkNaturalityId p pf f η_f := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have pf : Q($ctx.C) := pf.e.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    return q(naturality_id $η_f)
  mkNaturalityComp p pf f g h η θ η_f η_g η_h ih_η ih_θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have pf : Q($ctx.C) := pf.e.e
    have η : Q($f ≅ $g) := η.e
    have θ : Q($g ≅ $h) := θ.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    have η_g : Q($p ⊗ $g ≅ $pf) := η_g.e
    have η_h : Q($p ⊗ $h ≅ $pf) := η_h.e
    have ih_η : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih_η
    have ih_θ : Q($p ◁ $θ ≪≫ $η_h = $η_g) := ih_θ
    return q(naturality_comp $η_f $η_g $η_h $ih_η $ih_θ)
  mkNaturalityWhiskerLeft p pf pfg f g h η η_f η_fg η_fh ih_η := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have pf : Q($ctx.C) := pf.e.e
    have pfg : Q($ctx.C) := pfg.e.e
    have η : Q($g ≅ $h) := η.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    have η_fg : Q($pf ⊗ $g ≅ $pfg) := η_fg.e
    have η_fh : Q($pf ⊗ $h ≅ $pfg) := η_fh.e
    have ih_η : Q($pf ◁ $η ≪≫ $η_fh = $η_fg) := ih_η
    return q(naturality_whiskerLeft $η_f $η_fg $η_fh $ih_η)
  mkNaturalityWhiskerRight p pf pfh f g h η η_f η_g η_fh ih_η := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have pf : Q($ctx.C) := pf.e.e
    have pfh : Q($ctx.C) := pfh.e.e
    have η : Q($f ≅ $g) := η.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    have η_g : Q($p ⊗ $g ≅ $pf) := η_g.e
    have η_fh : Q($pf ⊗ $h ≅ $pfh) := η_fh.e
    have ih_η : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih_η
    return q(naturality_whiskerRight $η_f $η_g $η_fh $ih_η)
  mkNaturalityHorizontalComp p pf₁ pf₁f₂ f₁ g₁ f₂ g₂ η θ η_f₁ η_g₁ η_f₂ η_g₂ ih_η ih_θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f₁ : Q($ctx.C) := f₁.e
    have g₁ : Q($ctx.C) := g₁.e
    have f₂ : Q($ctx.C) := f₂.e
    have g₂ : Q($ctx.C) := g₂.e
    have pf₁ : Q($ctx.C) := pf₁.e.e
    have pf₁f₂ : Q($ctx.C) := pf₁f₂.e.e
    have η : Q($f₁ ≅ $g₁) := η.e
    have θ : Q($f₂ ≅ $g₂) := θ.e
    have η_f₁ : Q($p ⊗ $f₁ ≅ $pf₁) := η_f₁.e
    have η_g₁ : Q($p ⊗ $g₁ ≅ $pf₁) := η_g₁.e
    have η_f₂ : Q($pf₁ ⊗ $f₂ ≅ $pf₁f₂) := η_f₂.e
    have η_g₂ : Q($pf₁ ⊗ $g₂ ≅ $pf₁f₂) := η_g₂.e
    have ih_η : Q($p ◁ $η ≪≫ $η_g₁ = $η_f₁) := ih_η
    have ih_θ : Q($pf₁ ◁ $θ ≪≫ $η_g₂ = $η_f₂) := ih_θ
    return q(naturality_tensorHom $η_f₁ $η_g₁ $η_f₂ $η_g₂ $ih_η $ih_θ)
  mkNaturalityInv p pf f g η η_f η_g ih := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    have p : Q($ctx.C) := p.e.e
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have pf : Q($ctx.C) := pf.e.e
    have η : Q($f ≅ $g) := η.e
    have η_f : Q($p ⊗ $f ≅ $pf) := η_f.e
    have η_g : Q($p ⊗ $g ≅ $pf) := η_g.e
    have ih : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih
    return q(naturality_inv $η_f $η_g $ih)


theorem of_normalize_eq {f g f' : C} {η θ : f ≅ g} (η_f : 𝟙_ C ⊗ f ≅ f') (η_g : 𝟙_ C ⊗ g ≅ f')
    (h_η : 𝟙_ C ◁ η ≪≫ η_g = η_f)
    (h_θ : 𝟙_ C ◁ θ ≪≫ η_g = η_f) : η = θ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g f' : C
    η θ : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
    η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
    h_η : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoi …
    h_θ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoi …
    ⊢ Eq η θ
  -/
  apply Iso.ext
  calc
    η.hom = (λ_ f).inv ≫ η_f.hom ≫ η_g.inv ≫ (λ_ g).hom := by
      simp [← reassoc_of% (congrArg Iso.hom h_η)]
    _ = θ.hom := by
      simp [← reassoc_of% (congrArg Iso.hom h_θ)]


theorem mk_eq_of_naturality {f g f' : C} {η θ : f ⟶ g} {η' θ' : f ≅ g}
    (η_f : 𝟙_ C ⊗ f ≅ f') (η_g : 𝟙_ C ⊗ g ≅ f')
    (η_hom : η'.hom = η) (Θ_hom : θ'.hom = θ)
    (Hη : whiskerLeftIso (𝟙_ C) η' ≪≫ η_g = η_f)
    (Hθ : whiskerLeftIso (𝟙_ C) θ' ≪≫ η_g = η_f) : η = θ :=
  calc
    η = η'.hom := η_hom.symm
    _ = (λ_ f).inv ≫ η_f.hom ≫ η_g.inv ≫ (λ_ g).hom := by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        f g f' : C
        η θ : Quiver.Hom f g
        η' θ' : CategoryTheory.Iso f g
        η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
        η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
        η_hom : Eq η'.hom η
        Θ_hom : Eq θ'.hom θ
        Hη : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoid …
        Hθ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoid …
        ⊢ Eq η'.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
      -/
      simp [← reassoc_of% (congrArg Iso.hom Hη)]
      /-
        🎉 no goals
      -/
    _ = θ'.hom := by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        f g f' : C
        η θ : Quiver.Hom f g
        η' θ' : CategoryTheory.Iso f g
        η_f : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
        η_g : CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj Cate …
        η_hom : Eq η'.hom η
        Θ_hom : Eq θ'.hom θ
        Hη : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoid …
        Hθ : Eq ((CategoryTheory.MonoidalCategory.whiskerLeftIso CategoryTheory.Monoid …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp [← reassoc_of% (congrArg Iso.hom Hθ)]
      /-
        🎉 no goals
      -/
    _ = θ := Θ_hom


instance : MkEqOfNaturality MonoidalM where
  mkEqOfNaturality η θ ηIso θIso η_f η_g Hη Hθ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let η' := ηIso.e
    let θ' := θIso.e
    let f ← η'.srcM
    let g ← η'.tgtM
    let f' ← η_f.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have f' : Q($ctx.C) := f'.e
    have η : Q($f ⟶ $g) := η
    have θ : Q($f ⟶ $g) := θ
    have η'_e : Q($f ≅ $g) := η'.e
    have θ'_e : Q($f ≅ $g) := θ'.e
    have η_f : Q(tensorUnit ⊗ $f ≅ $f') := η_f.e
    have η_g : Q(tensorUnit ⊗ $g ≅ $f') := η_g.e
    have η_hom : Q(Iso.hom $η'_e = $η) := ηIso.eq
    have Θ_hom : Q(Iso.hom $θ'_e = $θ) := θIso.eq
    have Hη : Q(whiskerLeftIso tensorUnit $η'_e ≪≫ $η_g = $η_f) := Hη
    have Hθ : Q(whiskerLeftIso tensorUnit $θ'_e ≪≫ $η_g = $η_f) := Hθ
    return q(mk_eq_of_naturality $η_f $η_g $η_hom $Θ_hom $Hη $Hθ)


/-- Close the goal of the form `η = θ`, where `η` and `θ` are 2-isomorphisms made up only of
associators, unitors, and identities.
```lean
example {C : Type} [Category C] [MonoidalCategory C] :
  (λ_ (𝟙_ C)).hom = (ρ_ (𝟙_ C)).hom := by
  monoidal_coherence
```
-/
def pureCoherence (mvarId : MVarId) : MetaM (List MVarId) :=
  BicategoryLike.pureCoherence Monoidal.Context `monoidal mvarId


@[inherit_doc pureCoherence]
elab "monoidal_coherence" : tactic => withMainContext do
  replaceMainGoal <| ← Monoidal.pureCoherence <| ← getMainGoal


