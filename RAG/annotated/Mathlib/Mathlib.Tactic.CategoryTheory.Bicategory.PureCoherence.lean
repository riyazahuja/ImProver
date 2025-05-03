local infixr:81 " ◁ " => Bicategory.whiskerLeftIso

local infixl:81 " ▷ " => Bicategory.whiskerRightIso


/-- The composition of the normalizing isomorphisms `η_f : p ≫ f ≅ pf` and `η_g : pf ≫ g ≅ pfg`. -/
abbrev normalizeIsoComp {p : a ⟶ b} {f : b ⟶ c} {g : c ⟶ d} {pf : a ⟶ c} {pfg : a ⟶ d}
    (η_f : p ≫ f ≅ pf) (η_g : pf ≫ g ≅ pfg) :=
  (α_ _ _ _).symm ≪≫ whiskerRightIso η_f g ≪≫ η_g


theorem naturality_associator
    {p : a ⟶ b} {f : b ⟶ c} {g : c ⟶ d} {h : d ⟶ e} {pf : a ⟶ c} {pfg : a ⟶ d} {pfgh : a ⟶ e}
    (η_f : p ≫ f ≅ pf) (η_g : pf ≫ g ≅ pfg) (η_h : pfg ≫ h ≅ pfgh) :
    p ◁ (α_ f g h) ≪≫ (normalizeIsoComp η_f (normalizeIsoComp η_g η_h)) =
    (normalizeIsoComp (normalizeIsoComp η_f η_g) η_h) :=
              /-
                B : Type u
                inst✝ : CategoryTheory.Bicategory B
                a b c d e : B
                p : Quiver.Hom a b
                f : Quiver.Hom b c
                g : Quiver.Hom c d
                h : Quiver.Hom d e
                pf : Quiver.Hom a c
                pfg : Quiver.Hom a d
                pfgh : Quiver.Hom a e
                η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
                η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf g) pfg
                η_h : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pfg h) pfgh
                ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.a …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_leftUnitor {p : a ⟶ b} {f : b ⟶ c} {pf : a ⟶ c} (η_f : p ≫ f ≅ pf) :
    p ◁ (λ_ f) ≪≫ η_f = normalizeIsoComp (ρ_ p) η_f :=
              /-
                B : Type u
                inst✝ : CategoryTheory.Bicategory B
                a b c : B
                p : Quiver.Hom a b
                f : Quiver.Hom b c
                pf : Quiver.Hom a c
                η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
                ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.l …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_rightUnitor {p : a ⟶ b} {f : b ⟶ c} {pf : a ⟶ c} (η_f : p ≫ f ≅ pf) :
    p ◁ (ρ_ f) ≪≫ η_f = normalizeIsoComp η_f (ρ_ pf) :=
              /-
                B : Type u
                inst✝ : CategoryTheory.Bicategory B
                a b c : B
                p : Quiver.Hom a b
                f : Quiver.Hom b c
                pf : Quiver.Hom a c
                η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
                ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.r …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_id {p : a ⟶ b} {f : b ⟶ c} {pf : a ⟶ c} (η_f : p ≫ f ≅ pf) :
    p ◁ Iso.refl f ≪≫ η_f = η_f :=
              /-
                B : Type u
                inst✝ : CategoryTheory.Bicategory B
                a b c : B
                p : Quiver.Hom a b
                f : Quiver.Hom b c
                pf : Quiver.Hom a c
                η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
                ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Iso.refl f)) …
              -/
  Iso.ext (by simp)
              /-
                🎉 no goals
              -/


theorem naturality_comp {p : a ⟶ b} {f g h : b ⟶ c} {pf : a ⟶ c} {η : f ≅ g} {θ : g ≅ h}
    (η_f : p ≫ f ≅ pf) (η_g : p ≫ g ≅ pf) (η_h : p ≫ h ≅ pf)
    (ih_η : p ◁ η ≪≫ η_g = η_f) (ih_θ : p ◁ θ ≪≫ η_h = η_g) :
    p ◁ (η ≪≫ θ) ≪≫ η_h = η_f := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    p : Quiver.Hom a b
    f g h : Quiver.Hom b c
    pf : Quiver.Hom a c
    η : CategoryTheory.Iso f g
    θ : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    η_h : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p h) pf
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ih_θ : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p θ).trans η_h) η_g
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (η.trans θ)).trans η_h) η_f
  -/
  rw [← ih_η, ← ih_θ]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    p : Quiver.Hom a b
    f g h : Quiver.Hom b c
    pf : Quiver.Hom a c
    η : CategoryTheory.Iso f g
    θ : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    η_h : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p h) pf
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ih_θ : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p θ).trans η_h) η_g
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (η.trans θ)).trans η_h) ((Ca …
  -/
  apply Iso.ext (by simp)
  /-
    🎉 no goals
  -/


theorem naturality_whiskerLeft {p : a ⟶ b} {f : b ⟶ c} {g h : c ⟶ d} {pf : a ⟶ c} {pfg : a ⟶ d}
    {η : g ≅ h} (η_f : p ≫ f ≅ pf) (η_fg : pf ≫ g ≅ pfg) (η_fh : pf ≫ h ≅ pfg)
    (ih_η : pf ◁ η ≪≫ η_fh = η_fg) :
    p ◁ (f ◁ η) ≪≫ normalizeIsoComp η_f η_fh = normalizeIsoComp η_f η_fg := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d : B
    p : Quiver.Hom a b
    f : Quiver.Hom b c
    g h : Quiver.Hom c d
    pf : Quiver.Hom a c
    pfg : Quiver.Hom a d
    η : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_fg : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf g) pfg
    η_fh : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf h) pfg
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso pf η).trans η_fh) η_fg
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.w …
  -/
  rw [← ih_η]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d : B
    p : Quiver.Hom a b
    f : Quiver.Hom b c
    g h : Quiver.Hom c d
    pf : Quiver.Hom a c
    pfg : Quiver.Hom a d
    η : CategoryTheory.Iso g h
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_fg : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf g) pfg
    η_fh : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf h) pfg
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso pf η).trans η_fh) η_fg
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.w …
  -/
  apply Iso.ext (by simp [← whisker_exchange_assoc])
  /-
    🎉 no goals
  -/


theorem naturality_whiskerRight {p : a ⟶ b} {f g : b ⟶ c} {h : c ⟶ d} {pf : a ⟶ c} {pfh : a ⟶ d}
    {η : f ≅ g} (η_f : p ≫ f ≅ pf) (η_g : p ≫ g ≅ pf) (η_fh : pf ≫ h ≅ pfh)
    (ih_η : p ◁ η ≪≫ η_g = η_f) :
    p ◁ (η ▷ h) ≪≫ normalizeIsoComp η_g η_fh = normalizeIsoComp η_f η_fh := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d : B
    p : Quiver.Hom a b
    f g : Quiver.Hom b c
    h : Quiver.Hom c d
    pf : Quiver.Hom a c
    pfh : Quiver.Hom a d
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    η_fh : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf h) pfh
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.w …
  -/
  rw [← ih_η]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d : B
    p : Quiver.Hom a b
    f g : Quiver.Hom b c
    h : Quiver.Hom c d
    pf : Quiver.Hom a c
    pfh : Quiver.Hom a d
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    η_fh : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp pf h) pfh
    ih_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p (CategoryTheory.Bicategory.w …
  -/
  apply Iso.ext (by simp)
  /-
    🎉 no goals
  -/


theorem naturality_inv {p : a ⟶ b} {f g : b ⟶ c} {pf : a ⟶ c}
    {η : f ≅ g} (η_f : p ≫ f ≅ pf) (η_g : p ≫ g ≅ pf) (ih : p ◁ η ≪≫ η_g = η_f) :
    p ◁ η.symm ≪≫ η_f = η_g := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    p : Quiver.Hom a b
    f g : Quiver.Hom b c
    pf : Quiver.Hom a c
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    ih : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η.symm).trans η_f) η_g
  -/
  rw [← ih]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    p : Quiver.Hom a b
    f g : Quiver.Hom b c
    pf : Quiver.Hom a c
    η : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p f) pf
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp p g) pf
    ih : Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η).trans η_g) η_f
    ⊢ Eq ((CategoryTheory.Bicategory.whiskerLeftIso p η.symm).trans ((CategoryTheo …
  -/
  apply Iso.ext (by simp)
  /-
    🎉 no goals
  -/


instance : MonadNormalizeNaturality BicategoryM where
  mkNaturalityAssociator p pf pfg pfgh f g h η_f η_g η_h := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have d : Q($ctx.B) := g.tgt.e
    have e : Q($ctx.B) := h.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have g : Q($c ⟶ $d) := g.e
    have h : Q($d ⟶ $e) := h.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have pfg : Q($a ⟶ $d) := pfg.e.e
    have pfgh : Q($a ⟶ $e) := pfgh.e.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    have η_g : Q($pf ≫ $g ≅ $pfg) := η_g.e
    have η_h : Q($pfg ≫ $h ≅ $pfgh) := η_h.e
    return q(naturality_associator $η_f $η_g $η_h)
  mkNaturalityLeftUnitor p pf f η_f := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    return q(naturality_leftUnitor $η_f)
  mkNaturalityRightUnitor p pf f η_f := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    return q(naturality_rightUnitor $η_f)
  mkNaturalityId p pf f η_f := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    return q(naturality_id $η_f)
  mkNaturalityComp p pf f g h η θ η_f η_g η_h ih_η ih_θ := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have g : Q($b ⟶ $c) := g.e
    have h : Q($b ⟶ $c) := h.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have η : Q($f ≅ $g) := η.e
    have θ : Q($g ≅ $h) := θ.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    have η_g : Q($p ≫ $g ≅ $pf) := η_g.e
    have η_h : Q($p ≫ $h ≅ $pf) := η_h.e
    have ih_η : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih_η
    have ih_θ : Q($p ◁ $θ ≪≫ $η_h = $η_g) := ih_θ
    return q(naturality_comp $η_f $η_g $η_h $ih_η $ih_θ)
  mkNaturalityWhiskerLeft p pf pfg f g h η η_f η_fg η_fh ih_η := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have d : Q($ctx.B) := g.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have g : Q($c ⟶ $d) := g.e
    have h : Q($c ⟶ $d) := h.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have pfg : Q($a ⟶ $d) := pfg.e.e
    have η : Q($g ≅ $h) := η.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    have η_fg : Q($pf ≫ $g ≅ $pfg) := η_fg.e
    have η_fh : Q($pf ≫ $h ≅ $pfg) := η_fh.e
    have ih_η : Q($pf ◁ $η ≪≫ $η_fh = $η_fg) := ih_η
    return q(naturality_whiskerLeft $η_f $η_fg $η_fh $ih_η)
  mkNaturalityWhiskerRight p pf pfh f g h η η_f η_g η_fh ih_η := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have d : Q($ctx.B) := h.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have g : Q($b ⟶ $c) := g.e
    have h : Q($c ⟶ $d) := h.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have pfh : Q($a ⟶ $d) := pfh.e.e
    have η : Q($f ≅ $g) := η.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    have η_g : Q($p ≫ $g ≅ $pf) := η_g.e
    have η_fh : Q($pf ≫ $h ≅ $pfh) := η_fh.e
    have ih_η : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih_η
    return q(naturality_whiskerRight $η_f $η_g $η_fh $ih_η)
  mkNaturalityHorizontalComp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ := do
    throwError "horizontal composition is not implemented"
  mkNaturalityInv p pf f g η η_f η_g ih := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    have a : Q($ctx.B) := p.src.e
    have b : Q($ctx.B) := p.tgt.e
    have c : Q($ctx.B) := f.tgt.e
    have p : Q($a ⟶ $b) := p.e.e
    have f : Q($b ⟶ $c) := f.e
    have g : Q($b ⟶ $c) := g.e
    have pf : Q($a ⟶ $c) := pf.e.e
    have η : Q($f ≅ $g) := η.e
    have η_f : Q($p ≫ $f ≅ $pf) := η_f.e
    have η_g : Q($p ≫ $g ≅ $pf) := η_g.e
    have ih : Q($p ◁ $η ≪≫ $η_g = $η_f) := ih
    return q(naturality_inv $η_f $η_g $ih)


theorem of_normalize_eq {f g f' : a ⟶ b} {η θ : f ≅ g} (η_f : 𝟙 a ≫ f ≅ f') (η_g : 𝟙 a ≫ g ≅ f')
    (h_η : 𝟙 a ◁ η ≪≫ η_g = η_f)
    (h_θ : 𝟙 a ◁ θ ≪≫ η_g = η_f) : η = θ := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    f g f' : Quiver.Hom a b
    η θ : CategoryTheory.Iso f g
    η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
    η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
    h_η : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
    h_θ : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
    ⊢ Eq η θ
  -/
  apply Iso.ext
  calc
    η.hom = (λ_ f).inv ≫ η_f.hom ≫ η_g.inv ≫ (λ_ g).hom := by
      simp [← reassoc_of% (congrArg Iso.hom h_η)]
    _ = θ.hom := by
      simp [← reassoc_of% (congrArg Iso.hom h_θ)]


theorem mk_eq_of_naturality {f g f' : a ⟶ b} {η θ : f ⟶ g} {η' θ' : f ≅ g}
    (η_f : 𝟙 a ≫ f ≅ f') (η_g : 𝟙 a ≫ g ≅ f')
    (Hη : η'.hom = η) (Hθ : θ'.hom = θ)
    (Hη' : whiskerLeftIso (𝟙 a) η' ≪≫ η_g = η_f)
    (Hθ' : whiskerLeftIso (𝟙 a) θ' ≪≫ η_g = η_f) : η = θ :=
  calc
    η = η'.hom := Hη.symm
    _ = (λ_ f).inv ≫ η_f.hom ≫ η_g.inv ≫ (λ_ g).hom := by
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b : B
        f g f' : Quiver.Hom a b
        η θ : Quiver.Hom f g
        η' θ' : CategoryTheory.Iso f g
        η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
        η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
        Hη : Eq η'.hom η
        Hθ : Eq θ'.hom θ
        Hη' : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
        Hθ' : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
        ⊢ Eq η'.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.lef …
      -/
      simp [← reassoc_of% (congrArg Iso.hom Hη')]
      /-
        🎉 no goals
      -/
    _ = θ'.hom := by
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b : B
        f g f' : Quiver.Hom a b
        η θ : Quiver.Hom f g
        η' θ' : CategoryTheory.Iso f g
        η_f : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
        η_g : CategoryTheory.Iso (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
        Hη : Eq η'.hom η
        Hθ : Eq θ'.hom θ
        Hη' : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
        Hθ' : Eq ((CategoryTheory.Bicategory.whiskerLeftIso (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.leftUnitor …
      -/
      simp [← reassoc_of% (congrArg Iso.hom Hθ')]
      /-
        🎉 no goals
      -/
    _ = θ := Hθ


instance : MkEqOfNaturality BicategoryM where
  mkEqOfNaturality η θ ηIso θIso η_f η_g Hη Hθ := do
    let ctx ← read
    let _bicat := ctx.instBicategory
    let η' := ηIso.e
    let θ' := θIso.e
    let f ← η'.srcM
    let g ← η'.tgtM
    let f' ← η_f.tgtM
    have a : Q($ctx.B) := f.src.e
    have b : Q($ctx.B) := f.tgt.e
    have f : Q($a ⟶ $b) := f.e
    have g : Q($a ⟶ $b) := g.e
    have f' : Q($a ⟶ $b) := f'.e
    have η : Q($f ⟶ $g) := η
    have θ : Q($f ⟶ $g) := θ
    have η'_e : Q($f ≅ $g) := η'.e
    have θ'_e : Q($f ≅ $g) := θ'.e
    have η_f : Q(𝟙 $a ≫ $f ≅ $f') := η_f.e
    have η_g : Q(𝟙 $a ≫ $g ≅ $f') := η_g.e
    have η_hom : Q(Iso.hom $η'_e = $η) := ηIso.eq
    have Θ_hom : Q(Iso.hom $θ'_e = $θ) := θIso.eq
    have Hη : Q(whiskerLeftIso (𝟙 $a) $η'_e ≪≫ $η_g = $η_f) := Hη
    have Hθ : Q(whiskerLeftIso (𝟙 $a) $θ'_e ≪≫ $η_g = $η_f) := Hθ
    return q(mk_eq_of_naturality $η_f $η_g $η_hom $Θ_hom $Hη $Hθ)


/-- Close the goal of the form `η = θ`, where `η` and `θ` are 2-isomorphisms made up only of
associators, unitors, and identities.
```lean
example {B : Type} [Bicategory B] {a : B} :
  (λ_ (𝟙 a)).hom = (ρ_ (𝟙 a)).hom := by
  bicategory_coherence
```
-/
def pureCoherence (mvarId : MVarId) : MetaM (List MVarId) :=
  BicategoryLike.pureCoherence Bicategory.Context `bicategory mvarId


@[inherit_doc pureCoherence]
elab "bicategory_coherence" : tactic => withMainContext do
  replaceMainGoal <| ← Bicategory.pureCoherence <| ← getMainGoal


