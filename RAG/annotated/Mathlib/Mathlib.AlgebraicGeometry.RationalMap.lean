/--
A partial map from `X` to `Y` (`X.PartialMap Y`) is a morphism into `Y`
defined on a dense open subscheme of `X`.
-/
structure PartialMap (X Y : Scheme.{u}) where
  /-- The domain of definition of a partial map. -/
  domain : X.Opens
  dense_domain : Dense (domain : Set X)
  /-- The underlying morphism of a partial map. -/
  hom : ↑domain ⟶ Y


variable (S) in
/-- A partial map is a `S`-map if the underlying morphism is. -/
abbrev PartialMap.IsOver [X.Over S] [Y.Over S] (f : X.PartialMap Y) :=
  f.hom.IsOver S


lemma ext_iff (f g : X.PartialMap Y) :
    f = g ↔ ∃ e : f.domain = g.domain, f.hom = (X.isoOfEq e).hom ≫ g.hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : X.PartialMap Y
    ⊢ Iff (Eq f g) (Exists fun e => Eq f.hom (CategoryTheory.CategoryStruct.comp ( …
  -/
  constructor
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      f g : X.PartialMap Y
      ⊢ Eq f g → Exists fun e => Eq f.hom (CategoryTheory.CategoryStruct.comp (X.iso …
    -/
  · rintro rfl
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      f : X.PartialMap Y
      ⊢ Exists fun e => Eq f.hom (CategoryTheory.CategoryStruct.comp (X.isoOfEq e).h …
    -/
    simp only [exists_true_left, Scheme.isoOfEq_rfl, Iso.refl_hom, Category.id_comp]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      f g : X.PartialMap Y
      ⊢ (Exists fun e => Eq f.hom (CategoryTheory.CategoryStruct.comp (X.isoOfEq e). …
    -/
  · obtain ⟨U, hU, f⟩ := f
    /-
      case mpr.mk
      X Y : AlgebraicGeometry.Scheme
      g : X.PartialMap Y
      U : X.Opens
      hU : Dense ↑U
      f : Quiver.Hom (↑U) Y
      ⊢ (Exists fun e => Eq { domain := U, dense_domain := hU, hom := f }.hom (Categ …
    -/
    obtain ⟨V, hV, g⟩ := g
    /-
      case mpr.mk.mk
      X Y : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : Dense ↑U
      f : Quiver.Hom (↑U) Y
      V : X.Opens
      hV : Dense ↑V
      g : Quiver.Hom (↑V) Y
      ⊢ (Exists fun e => Eq { domain := U, dense_domain := hU, hom := f }.hom (Categ …
    -/
    rintro ⟨rfl : U = V, e⟩
    /-
      case mpr.mk.mk.intro
      X Y : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : Dense ↑U
      f : Quiver.Hom (↑U) Y
      hV : Dense ↑U
      g : Quiver.Hom (↑U) Y
      e : Eq { domain := U, dense_domain := hU, hom := f }.hom (CategoryTheory.Categ …
      ⊢ Eq { domain := U, dense_domain := hU, hom := f } { domain := U, dense_domain …
    -/
    congr 1
    /-
      case mpr.mk.mk.intro.e_hom
      X Y : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : Dense ↑U
      f : Quiver.Hom (↑U) Y
      hV : Dense ↑U
      g : Quiver.Hom (↑U) Y
      e : Eq { domain := U, dense_domain := hU, hom := f }.hom (CategoryTheory.Categ …
      ⊢ Eq f g
    -/
    simpa using e
    /-
      🎉 no goals
    -/


@[ext]
lemma ext (f g : X.PartialMap Y) (e : f.domain = g.domain)
    (H : f.hom = (X.isoOfEq e).hom ≫ g.hom) : f = g := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : X.PartialMap Y
    e : Eq f.domain g.domain
    H : Eq f.hom (CategoryTheory.CategoryStruct.comp (X.isoOfEq e).hom g.hom)
    ⊢ Eq f g
  -/
  rw [ext_iff]
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : X.PartialMap Y
    e : Eq f.domain g.domain
    H : Eq f.hom (CategoryTheory.CategoryStruct.comp (X.isoOfEq e).hom g.hom)
    ⊢ Exists fun e => Eq f.hom (CategoryTheory.CategoryStruct.comp (X.isoOfEq e).h …
  -/
  exact ⟨e, H⟩
  /-
    🎉 no goals
  -/


/-- The restriction of a partial map to a smaller domain. -/
@[simps hom domain]
noncomputable
def restrict (f : X.PartialMap Y) (U : X.Opens)
    (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) : X.PartialMap Y where
  domain := U
  dense_domain := hU
  hom := X.homOfLE hU' ≫ f.hom


@[simp]
lemma restrict_id (f : X.PartialMap Y) : f.restrict f.domain f.dense_domain le_rfl = f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    ⊢ Eq (f.restrict f.domain ⋯ ⋯) f
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp [restrict_domain]
           /-
             🎉 no goals
           -/


lemma restrict_id_hom (f : X.PartialMap Y) :
    (f.restrict f.domain f.dense_domain le_rfl).hom = f.hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    ⊢ Eq (f.restrict f.domain ⋯ ⋯).hom f.hom
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma restrict_restrict (f : X.PartialMap Y)
    (U : X.Opens) (hU : Dense (U : Set X)) (hU' : U ≤ f.domain)
    (V : X.Opens) (hV : Dense (V : Set X)) (hV' : V ≤ U) :
    (f.restrict U hU hU').restrict V hV hV' = f.restrict V hV (hV'.trans hU') := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    V : X.Opens
    hV : Dense ↑V
    hV' : LE.le V U
    ⊢ Eq ((f.restrict U hU hU').restrict V hV hV') (f.restrict V hV ⋯)
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp [restrict_domain]
           /-
             🎉 no goals
           -/


lemma restrict_restrict_hom (f : X.PartialMap Y)
    (U : X.Opens) (hU : Dense (U : Set X)) (hU' : U ≤ f.domain)
    (V : X.Opens) (hV : Dense (V : Set X)) (hV' : V ≤ U) :
    ((f.restrict U hU hU').restrict V hV hV').hom = (f.restrict V hV (hV'.trans hU')).hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    V : X.Opens
    hV : Dense ↑V
    hV' : LE.le V U
    ⊢ Eq ((f.restrict U hU hU').restrict V hV hV').hom (f.restrict V hV ⋯).hom
  -/
  simp
  /-
    🎉 no goals
  -/


instance [X.Over S] [Y.Over S] (f : X.PartialMap Y) [f.IsOver S]
    (U : X.Opens) (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) :
    (f.restrict U hU hU').IsOver S where


/-- The composition of a partial map and a morphism on the right. -/
@[simps]
def compHom (f : X.PartialMap Y) (g : Y ⟶ Z) : X.PartialMap Z where
  domain := f.domain
  dense_domain := f.dense_domain
  hom := f.hom ≫ g


instance [X.Over S] [Y.Over S] [Z.Over S] (f : X.PartialMap Y) (g : Y ⟶ Z)
    [f.IsOver S] [g.IsOver S] : (f.compHom g).IsOver S where


/-- A scheme morphism as a partial map. -/
@[simps]
def _root_.AlgebraicGeometry.Scheme.Hom.toPartialMap (f : X.Hom Y) :
    X.PartialMap Y := ⟨⊤, dense_univ, X.topIso.hom ≫ f⟩


instance [X.Over S] [Y.Over S] (f : X ⟶ Y) [f.IsOver S] : f.toPartialMap.IsOver S where


lemma isOver_iff [X.Over S] [Y.Over S] {f : X.PartialMap Y} :
    f.IsOver S ↔ (f.compHom (Y ↘ S)).hom = f.domain.ι ≫ X ↘ S := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : X.Over S
    inst✝ : Y.Over S
    f : X.PartialMap Y
    ⊢ Iff (AlgebraicGeometry.Scheme.PartialMap.IsOver S f) (Eq (f.compHom (Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma isOver_iff_eq_restrict [X.Over S] [Y.Over S] {f : X.PartialMap Y} :
                                                                                        /-
                                                                                          X Y Z S : AlgebraicGeometry.Scheme
                                                                                          sX : Quiver.Hom X S
                                                                                          sY : Quiver.Hom Y S
                                                                                          inst✝¹ : X.Over S
                                                                                          inst✝ : Y.Over S
                                                                                          f : X.PartialMap Y
                                                                                          ⊢ LE.le f.domain (AlgebraicGeometry.Scheme.Hom.toPartialMap (CategoryTheory.ov …
                                                                                        -/
    f.IsOver S ↔ f.compHom (Y ↘ S) = (X ↘ S).toPartialMap.restrict _ f.dense_domain (by simp) := by
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : X.Over S
    inst✝ : Y.Over S
    f : X.PartialMap Y
    ⊢ Iff (AlgebraicGeometry.Scheme.PartialMap.IsOver S f) (Eq (f.compHom (Categor …
  -/
  simp [isOver_iff, PartialMap.ext_iff]
  /-
    🎉 no goals
  -/


/-- If `x` is in the domain of a partial map `f`, then `f` restricts to a map from `Spec 𝒪_x`. -/
noncomputable
def fromSpecStalkOfMem (f : X.PartialMap Y) {x} (hx : x ∈ f.domain) :
    Spec (X.presheaf.stalk x) ⟶ Y :=
  f.domain.fromSpecStalkOfMem x hx ≫ f.hom


/-- A partial map restricts to a map from `Spec K(X)`. -/
noncomputable
abbrev fromFunctionField [IrreducibleSpace X] (f : X.PartialMap Y) :
    Spec X.functionField ⟶ Y :=
  f.fromSpecStalkOfMem
    ((genericPoint_specializes _).mem_open f.domain.2 f.dense_domain.nonempty.choose_spec)


lemma fromSpecStalkOfMem_restrict (f : X.PartialMap Y)
    {U : X.Opens} (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) {x} (hx : x ∈ U) :
    (f.restrict U hU hU').fromSpecStalkOfMem hx = f.fromSpecStalkOfMem (hU' hx) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((f.restrict U hU hU').fromSpecStalkOfMem hx) (f.fromSpecStalkOfMem ⋯)
  -/
  dsimp only [fromSpecStalkOfMem, restrict, Scheme.Opens.fromSpecStalkOfMem]
  have e : ⟨x, hU' hx⟩ = (X.homOfLE hU').base ⟨x, hx⟩ := by
    rw [Scheme.homOfLE_base]
    rfl
  rw [Category.assoc, ← Spec_map_stalkMap_fromSpecStalk_assoc,
    ← Spec_map_stalkSpecializes_fromSpecStalk (Inseparable.of_eq e).specializes,
    ← TopCat.Presheaf.stalkCongr_inv _ (Inseparable.of_eq e)]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    e : Eq ⟨x, ⋯⟩ ((X.homOfLE hU').base ⟨x, hx⟩)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Category …
  -/
  simp only [← Category.assoc, ← Spec.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    e : Eq ⟨x, ⋯⟩ ((X.homOfLE hU').base ⟨x, hx⟩)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 3
  rw [Iso.eq_inv_comp, ← Category.assoc, IsIso.comp_inv_eq, IsIso.eq_inv_comp,
    stalkMap_congr_hom _ _ (X.homOfLE_ι hU').symm]
  /-
    case e_a.e_a.e_f
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    e : Eq ⟨x, ⋯⟩ ((X.homOfLE hU').base ⟨x, hx⟩)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalkMa …
  -/
  simp only [restrictFunctor_obj_left, homOfLE_leOfHom, TopCat.Presheaf.stalkCongr_hom]
  /-
    case e_a.e_a.e_f
    X Y : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    e : Eq ⟨x, ⋯⟩ ((X.homOfLE hU').base ⟨x, hx⟩)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalkMa …
  -/
  rw [← stalkSpecializes_stalkMap_assoc, stalkMap_comp]
  /-
    🎉 no goals
  -/


lemma fromFunctionField_restrict (f : X.PartialMap Y) [IrreducibleSpace X]
    {U : X.Opens} (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) :
    (f.restrict U hU hU').fromFunctionField = f.fromFunctionField :=
  fromSpecStalkOfMem_restrict f _ _ _


/--
Given `S`-schemes `X` and `Y` such that `Y` is locally of finite type and
`X` is irreducible germ-injective at `x` (e.g. when `X` is integral),
any `S`-morphism `Spec 𝒪ₓ ⟶ Y` spreads out to a partial map from `X` to `Y`.
-/
noncomputable
def ofFromSpecStalk [IrreducibleSpace X] [LocallyOfFiniteType sY] {x : X} [X.IsGermInjectiveAt x]
    (φ : Spec (X.presheaf.stalk x) ⟶ Y) (h : φ ≫ sY = X.fromSpecStalk x ≫ sX) : X.PartialMap Y where
  hom := (spread_out_of_isGermInjective' sX sY φ h).choose_spec.choose_spec.choose
  dense_domain := (spread_out_of_isGermInjective' sX sY φ h).choose.2.dense
    ⟨_, (spread_out_of_isGermInjective' sX sY φ h).choose_spec.choose⟩


lemma ofFromSpecStalk_comp [IrreducibleSpace X] [LocallyOfFiniteType sY]
    {x : X} [X.IsGermInjectiveAt x] (φ : Spec (X.presheaf.stalk x) ⟶ Y)
    (h : φ ≫ sY = X.fromSpecStalk x ≫ sX) :
    (ofFromSpecStalk sX sY φ h).hom ≫ sY = (ofFromSpecStalk sX sY φ h).domain.ι ≫ sX :=
  (spread_out_of_isGermInjective' sX sY φ h).choose_spec.choose_spec.choose_spec.2


lemma mem_domain_ofFromSpecStalk [IrreducibleSpace X] [LocallyOfFiniteType sY]
    {x : X} [X.IsGermInjectiveAt x] (φ : Spec (X.presheaf.stalk x) ⟶ Y)
    (h : φ ≫ sY = X.fromSpecStalk x ≫ sX) : x ∈ (ofFromSpecStalk sX sY φ h).domain :=
  (spread_out_of_isGermInjective' sX sY φ h).choose_spec.choose


lemma fromSpecStalkOfMem_ofFromSpecStalk [IrreducibleSpace X] [LocallyOfFiniteType sY]
    {x : X} [X.IsGermInjectiveAt x] (φ : Spec (X.presheaf.stalk x) ⟶ Y)
    (h : φ ≫ sY = X.fromSpecStalk x ≫ sX) :
    (ofFromSpecStalk sX sY φ h).fromSpecStalkOfMem (mem_domain_ofFromSpecStalk sX sY φ h) = φ :=
  (spread_out_of_isGermInjective' sX sY φ h).choose_spec.choose_spec.choose_spec.1.symm


@[simp]
lemma fromSpecStalkOfMem_compHom (f : X.PartialMap Y) (g : Y ⟶ Z) (x) (hx) :
    (f.compHom g).fromSpecStalkOfMem (x := x) hx = f.fromSpecStalkOfMem hx ≫ g := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : X.PartialMap Y
    g : Quiver.Hom Y Z
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (f.compHom g).domain x
    ⊢ Eq ((f.compHom g).fromSpecStalkOfMem hx) (CategoryTheory.CategoryStruct.comp …
  -/
  simp [fromSpecStalkOfMem]
  /-
    🎉 no goals
  -/


@[simp]
lemma fromSpecStalkOfMem_toPartialMap (f : X ⟶ Y) (x) :
    f.toPartialMap.fromSpecStalkOfMem (x := x) trivial = X.fromSpecStalk x ≫ f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.toPartialMap f).fromSpecStalkOfMem trivial …
  -/
  simp [fromSpecStalkOfMem]
  /-
    🎉 no goals
  -/


/-- Two partial maps are equivalent if they are equal on a dense open subscheme. -/
protected noncomputable
def equiv (f g : X.PartialMap Y) : Prop :=
  ∃ (W : X.Opens) (hW : Dense (W : Set X)) (hWl : W ≤ f.domain) (hWr : W ≤ g.domain),
    (f.restrict W hW hWl).hom = (g.restrict W hW hWr).hom


lemma equivalence_rel : Equivalence (@Scheme.PartialMap.equiv X Y) where
                                          /-
                                            X Y : AlgebraicGeometry.Scheme
                                            f : X.PartialMap Y
                                            ⊢ Exists fun hWl => Exists fun hWr => Eq (f.restrict f.domain ⋯ hWl).hom (f.re …
                                          -/
  refl f := ⟨f.domain, f.dense_domain, by simp⟩
                                          /-
                                            🎉 no goals
                                          -/
  symm {f g} := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f g : X.PartialMap Y
      ⊢ f.equiv g → g.equiv f
    -/
    intro ⟨W, hW, hWl, hWr, e⟩
    /-
      X Y : AlgebraicGeometry.Scheme
      f g : X.PartialMap Y
      W : X.Opens
      hW : Dense ↑W
      hWl : LE.le W f.domain
      hWr : LE.le W g.domain
      e : Eq (f.restrict W hW hWl).hom (g.restrict W hW hWr).hom
      ⊢ g.equiv f
    -/
    exact ⟨W, hW, hWr, hWl, e.symm⟩
    /-
      🎉 no goals
    -/
  trans {f g h} := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f g h : X.PartialMap Y
      ⊢ f.equiv g → g.equiv h → f.equiv h
    -/
    intro ⟨W₁, hW₁, hW₁l, hW₁r, e₁⟩ ⟨W₂, hW₂, hW₂l, hW₂r, e₂⟩
    refine ⟨W₁ ⊓ W₂, hW₁.inter_of_isOpen_left hW₂ W₁.2, inf_le_left.trans hW₁l,
      inf_le_right.trans hW₂r, ?_⟩
    /-
      X Y : AlgebraicGeometry.Scheme
      f g h : X.PartialMap Y
      W₁ : X.Opens
      hW₁ : Dense ↑W₁
      hW₁l : LE.le W₁ f.domain
      hW₁r : LE.le W₁ g.domain
      e₁ : Eq (f.restrict W₁ hW₁ hW₁l).hom (g.restrict W₁ hW₁ hW₁r).hom
      W₂ : X.Opens
      hW₂ : Dense ↑W₂
      hW₂l : LE.le W₂ g.domain
      hW₂r : LE.le W₂ h.domain
      e₂ : Eq (g.restrict W₂ hW₂ hW₂l).hom (h.restrict W₂ hW₂ hW₂r).hom
      ⊢ Eq (f.restrict (Min.min W₁ W₂) ⋯ ⋯).hom (h.restrict (Min.min W₁ W₂) ⋯ ⋯).hom
    -/
    dsimp at e₁ e₂
    simp only [restrict_domain, restrict_hom, restrictFunctor_obj_left, homOfLE_leOfHom,
      ← X.homOfLE_homOfLE (U := W₁ ⊓ W₂) inf_le_left hW₁l, Functor.map_comp, Over.comp_left,
      Category.assoc, e₁, ← X.homOfLE_homOfLE (U := W₁ ⊓ W₂) inf_le_right hW₂r, ← e₂]
    /-
      X Y : AlgebraicGeometry.Scheme
      f g h : X.PartialMap Y
      W₁ : X.Opens
      hW₁ : Dense ↑W₁
      hW₁l : LE.le W₁ f.domain
      hW₁r : LE.le W₁ g.domain
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE hW₁l) f.hom) (CategoryT …
      W₂ : X.Opens
      hW₂ : Dense ↑W₂
      hW₂l : LE.le W₂ g.domain
      hW₂r : LE.le W₂ h.domain
      e₂ : Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE hW₂l) g.hom) (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) (CategoryTheory.Categor …
    -/
    simp only [homOfLE_homOfLE_assoc]
    /-
      🎉 no goals
    -/


instance : Setoid (X.PartialMap Y) := ⟨@PartialMap.equiv X Y, equivalence_rel⟩


lemma restrict_equiv (f : X.PartialMap Y) (U : X.Opens)
    (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) : (f.restrict U hU hU').equiv f :=
                          /-
                            X Y : AlgebraicGeometry.Scheme
                            f : X.PartialMap Y
                            U : X.Opens
                            hU : Dense ↑U
                            hU' : LE.le U f.domain
                            ⊢ Eq ((f.restrict U hU hU').restrict U hU ⋯).hom (f.restrict U hU hU').hom
                          -/
  ⟨U, hU, le_rfl, hU', by simp⟩
                          /-
                            🎉 no goals
                          -/


lemma equiv_of_fromSpecStalkOfMem_eq [IrreducibleSpace X]
    {x : X} [X.IsGermInjectiveAt x] (f g : X.PartialMap Y)
    (hxf : x ∈ f.domain) (hxg : x ∈ g.domain)
    (H : f.fromSpecStalkOfMem hxf = g.fromSpecStalkOfMem hxg) : f.equiv g := by
  have hdense : Dense ((f.domain ⊓ g.domain) : Set X) :=
    f.dense_domain.inter_of_isOpen_left g.dense_domain f.domain.2
  have := (isGermInjectiveAt_iff_of_isOpenImmersion (f := (f.domain ⊓ g.domain).ι)
    (x := ⟨x, hxf, hxg⟩)).mp ‹_›
  have := spread_out_unique_of_isGermInjective' (X := (f.domain ⊓ g.domain).toScheme)
    (X.homOfLE inf_le_left ≫ f.hom) (X.homOfLE inf_le_right ≫ g.hom) (x := ⟨x, hxf, hxg⟩) ?_
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : X.PartialMap Y
      hxf : Membership.mem f.domain x
      hxg : Membership.mem g.domain x
      H : Eq (f.fromSpecStalkOfMem hxf) (g.fromSpecStalkOfMem hxg)
      hdense : Dense (Min.min ↑f.domain ↑g.domain)
      this✝ : (↑(Min.min f.domain g.domain)).IsGermInjectiveAt ⟨x, ⋯⟩
      this : Exists fun U => And (Membership.mem U ⟨x, ⋯⟩) (Eq (CategoryTheory.Categ …
      ⊢ f.equiv g
    -/
  · obtain ⟨U, hxU, e⟩ := this
    refine ⟨(f.domain ⊓ g.domain).ι ''ᵁ U, ((f.domain ⊓ g.domain).ι ''ᵁ U).2.dense
      ⟨_, ⟨_, hxU, rfl⟩⟩,
      ((Set.image_subset_range _ _).trans_eq (Subtype.range_val)).trans inf_le_left,
      ((Set.image_subset_range _ _).trans_eq (Subtype.range_val)).trans inf_le_right, ?_⟩
    /-
      case refine_2.intro.intro
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : X.PartialMap Y
      hxf : Membership.mem f.domain x
      hxg : Membership.mem g.domain x
      H : Eq (f.fromSpecStalkOfMem hxf) (g.fromSpecStalkOfMem hxg)
      hdense : Dense (Min.min ↑f.domain ↑g.domain)
      this : (↑(Min.min f.domain g.domain)).IsGermInjectiveAt ⟨x, ⋯⟩
      U : (↑(Min.min f.domain g.domain)).Opens
      hxU : Membership.mem U ⟨x, ⋯⟩
      e : Eq (CategoryTheory.CategoryStruct.comp U.ι (CategoryTheory.CategoryStruct. …
      ⊢ Eq (f.restrict ((AlgebraicGeometry.Scheme.Hom.opensFunctor (Min.min f.domain …
    -/
    rw [← cancel_epi (Scheme.Hom.isoImage _ _).hom]
    simp only [TopologicalSpace.Opens.carrier_eq_coe, IsOpenMap.coe_functor_obj,
      TopologicalSpace.Opens.coe_inf, restrict_hom, ← Category.assoc] at e ⊢
    /-
      case refine_2.intro.intro
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : IrreducibleSpace ↑↑X.toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : X.PartialMap Y
      hxf : Membership.mem f.domain x
      hxg : Membership.mem g.domain x
      H : Eq (f.fromSpecStalkOfMem hxf) (g.fromSpecStalkOfMem hxg)
      hdense : Dense (Min.min ↑f.domain ↑g.domain)
      this : (↑(Min.min f.domain g.domain)).IsGermInjectiveAt ⟨x, ⋯⟩
      U : (↑(Min.min f.domain g.domain)).Opens
      hxU : Membership.mem U ⟨x, ⋯⟩
      e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    convert e using 2 <;> rw [← cancel_mono (Scheme.Opens.ι _)] <;> simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  · rw [← f.fromSpecStalkOfMem_restrict hdense inf_le_left ⟨hxf, hxg⟩,
      ← g.fromSpecStalkOfMem_restrict hdense inf_le_right ⟨hxf, hxg⟩] at H
    simpa only [fromSpecStalkOfMem, restrict_domain, Opens.fromSpecStalkOfMem, Spec.map_inv,
      restrict_hom, Category.assoc, IsIso.eq_inv_comp, IsIso.hom_inv_id_assoc] using H


instance (U : X.Opens) [IsReduced X] : IsReduced U := isReduced_of_isOpenImmersion U.ι


lemma Opens.isDominant_ι {U : X.Opens} (hU : Dense (X := X) U) : IsDominant U.ι :=
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : Dense ↑U
        ⊢ DenseRange ⇑U.ι.base
      -/
  ⟨by simpa [DenseRange] using hU⟩
      /-
        🎉 no goals
      -/


lemma Opens.isDominant_homOfLE {U V : X.Opens} (hU : Dense (X := X) U) (hU' : U ≤ V) :
    IsDominant (X.homOfLE hU') :=
                                                      /-
                                                        X : AlgebraicGeometry.Scheme
                                                        U V : X.Opens
                                                        hU : Dense ↑U
                                                        hU' : LE.le U V
                                                        ⊢ AlgebraicGeometry.IsDominant (CategoryTheory.CategoryStruct.comp (X.homOfLE  …
                                                      -/
  have : IsDominant (X.homOfLE hU' ≫ Opens.ι _) := by simpa using Opens.isDominant_ι hU
                                                      /-
                                                        🎉 no goals
                                                      -/
  IsDominant.of_comp_of_isOpenImmersion (g := Opens.ι _) _


/-- Two partial maps from reduced schemes to separated schemes are equivalent if and only if
they are equal on **any** open dense subset. -/
lemma equiv_iff_of_isSeparated_of_le [X.Over S] [Y.Over S] [IsReduced X]
    [IsSeparated (Y ↘ S)] {f g : X.PartialMap Y} [f.IsOver S] [g.IsOver S]
    {W : X.Opens} (hW : Dense (X := X) W) (hWl : W ≤ f.domain) (hWr : W ≤ g.domain) : f.equiv g ↔
      (f.restrict W hW hWl).hom = (g.restrict W hW hWr).hom := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f g : X.PartialMap Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f.domain
    hWr : LE.le W g.domain
    ⊢ Iff (f.equiv g) (Eq (f.restrict W hW hWl).hom (g.restrict W hW hWr).hom)
  -/
  refine ⟨fun ⟨V, hV, hVl, hVr, e⟩ ↦ ?_, fun e ↦ ⟨_, _, _, _, e⟩⟩
  have : IsDominant (X.homOfLE (inf_le_left : W ⊓ V ≤ W)) :=
    Opens.isDominant_homOfLE (hW.inter_of_isOpen_left hV W.2) _
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f g : X.PartialMap Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f.domain
    hWr : LE.le W g.domain
    x✝ : f.equiv g
    V : X.Opens
    hV : Dense ↑V
    hVl : LE.le V f.domain
    hVr : LE.le V g.domain
    e : Eq (f.restrict V hV hVl).hom (g.restrict V hV hVr).hom
    this : AlgebraicGeometry.IsDominant (X.homOfLE ⋯)
    ⊢ Eq (f.restrict W hW hWl).hom (g.restrict W hW hWr).hom
  -/
  apply ext_of_isDominant_of_isSeparated' S (X.homOfLE (inf_le_left : W ⊓ V ≤ W))
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f g : X.PartialMap Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f.domain
    hWr : LE.le W g.domain
    x✝ : f.equiv g
    V : X.Opens
    hV : Dense ↑V
    hVl : LE.le V f.domain
    hVr : LE.le V g.domain
    e : Eq (f.restrict V hV hVl).hom (g.restrict V hV hVr).hom
    this : AlgebraicGeometry.IsDominant (X.homOfLE ⋯)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) (f.restrict W hW hWl).h …
  -/
  simpa using congr(X.homOfLE (inf_le_right : W ⊓ V ≤ V) ≫ $e)
  /-
    🎉 no goals
  -/


/-- Two partial maps from reduced schemes to separated schemes are equivalent if and only if
they are equal on the intersection of the domains. -/
lemma equiv_iff_of_isSeparated [X.Over S] [Y.Over S] [IsReduced X]
    [IsSeparated (Y ↘ S)] {f g : X.PartialMap Y}
    [f.IsOver S] [g.IsOver S] : f.equiv g ↔
      (f.restrict _ (f.2.inter_of_isOpen_left g.2 f.domain.2) inf_le_left).hom =
      (g.restrict _ (f.2.inter_of_isOpen_left g.2 f.domain.2) inf_le_right).hom :=
  equiv_iff_of_isSeparated_of_le (S := S) _ _ _


/-- Two partial maps from reduced schemes to separated schemes with the same domain are equivalent
if and only if they are equal. -/
lemma equiv_iff_of_domain_eq_of_isSeparated [X.Over S] [Y.Over S] [IsReduced X]
    [IsSeparated (Y ↘ S)] {f g : X.PartialMap Y} (hfg : f.domain = g.domain)
    [f.IsOver S] [g.IsOver S] : f.equiv g ↔ f = g := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f g : X.PartialMap Y
    hfg : Eq f.domain g.domain
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    ⊢ Iff (f.equiv g) (Eq f g)
  -/
  rw [equiv_iff_of_isSeparated_of_le (S := S) f.dense_domain le_rfl hfg.le]
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f g : X.PartialMap Y
    hfg : Eq f.domain g.domain
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    ⊢ Iff (Eq (f.restrict f.domain ⋯ ⋯).hom (g.restrict f.domain ⋯ ⋯).hom) (Eq f g)
  -/
  obtain ⟨Uf, _, f⟩ := f
  /-
    case mk
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    g : X.PartialMap Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S g
    Uf : X.Opens
    dense_domain✝ : Dense ↑Uf
    f : Quiver.Hom (↑Uf) Y
    hfg : Eq { domain := Uf, dense_domain := dense_domain✝, hom := f }.domain g.do …
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S { domain := Uf, dense_dom …
    ⊢ Iff (Eq ({ domain := Uf, dense_domain := dense_domain✝, hom := f }.restrict  …
  -/
  obtain ⟨Ug, _, g⟩ := g
  /-
    case mk.mk
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    Uf : X.Opens
    dense_domain✝¹ : Dense ↑Uf
    f : Quiver.Hom (↑Uf) Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S { domain := Uf, dense_do …
    Ug : X.Opens
    dense_domain✝ : Dense ↑Ug
    g : Quiver.Hom (↑Ug) Y
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S { domain := Ug, dense_dom …
    hfg : Eq { domain := Uf, dense_domain := dense_domain✝¹, hom := f }.domain { d …
    ⊢ Iff (Eq ({ domain := Uf, dense_domain := dense_domain✝¹, hom := f }.restrict …
  -/
  obtain rfl : Uf = Ug := hfg
  /-
    case mk.mk
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    Uf : X.Opens
    dense_domain✝¹ : Dense ↑Uf
    f : Quiver.Hom (↑Uf) Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S { domain := Uf, dense_do …
    dense_domain✝ : Dense ↑Uf
    g : Quiver.Hom (↑Uf) Y
    inst✝ : AlgebraicGeometry.Scheme.PartialMap.IsOver S { domain := Uf, dense_dom …
    ⊢ Iff (Eq ({ domain := Uf, dense_domain := dense_domain✝¹, hom := f }.restrict …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A partial map from a reduced scheme to a separated scheme is equivalent to a morphism
if and only if it is equal to the restriction of the morphism. -/
lemma equiv_toPartialMap_iff_of_isSeparated [X.Over S] [Y.Over S] [IsReduced X]
    [IsSeparated (Y ↘ S)] {f : X.PartialMap Y} {g : X ⟶ Y}
    [f.IsOver S] [g.IsOver S] : f.equiv g.toPartialMap ↔
      f.hom = f.domain.ι ≫ g := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f : X.PartialMap Y
    g : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver g S
    ⊢ Iff (f.equiv (AlgebraicGeometry.Scheme.Hom.toPartialMap g)) (Eq f.hom (Categ …
  -/
  rw [equiv_iff_of_isSeparated (S := S), ← cancel_epi (X.isoOfEq (inf_top_eq f.domain)).hom]
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f : X.PartialMap Y
    g : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver g S
    ⊢ Iff (Eq (f.restrict (Min.min f.domain (AlgebraicGeometry.Scheme.Hom.toPartia …
  -/
  simp
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝⁵ : X.Over S
    inst✝⁴ : Y.Over S
    inst✝³ : AlgebraicGeometry.IsReduced X
    inst✝² : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
    f : X.PartialMap Y
    g : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
    inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver g S
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) f.hom) (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A rational map from `X` to `Y` (`X ⤏ Y`) is an equivalence class of partial maps,
where two partial maps are equivalent if they are equal on a dense open subscheme.  -/
def RationalMap (X Y : Scheme.{u}) : Type u :=
  @Quotient (X.PartialMap Y) inferInstance


/-- The notation for rational maps. -/
scoped[AlgebraicGeometry] infix:10 " ⤏ " => Scheme.RationalMap


/-- A partial map as a rational map. -/
def PartialMap.toRationalMap (f : X.PartialMap Y) : X ⤏ Y := Quotient.mk _ f


/-- A scheme morphism as a rational map. -/
abbrev Hom.toRationalMap (f : X.Hom Y) : X ⤏ Y := f.toPartialMap.toRationalMap


variable (S) in
/-- A rational map is a `S`-map if some partial map in the equivalence class is a `S`-map. -/
class RationalMap.IsOver [X.Over S] [Y.Over S] (f : X ⤏ Y) : Prop where
  exists_partialMap_over : ∃ g : X.PartialMap Y, g.IsOver S ∧ g.toRationalMap = f


lemma PartialMap.toRationalMap_surjective : Function.Surjective (@toRationalMap X Y) :=
  Quotient.exists_rep


lemma RationalMap.exists_rep (f : X ⤏ Y) : ∃ g : X.PartialMap Y, g.toRationalMap = f :=
  Quotient.exists_rep f


lemma PartialMap.toRationalMap_eq_iff {f g : X.PartialMap Y} :
    f.toRationalMap = g.toRationalMap ↔ f.equiv g :=
  Quotient.eq


@[simp]
lemma PartialMap.restrict_toRationalMap (f : X.PartialMap Y) (U : X.Opens)
    (hU : Dense (U : Set X)) (hU' : U ≤ f.domain) :
    (f.restrict U hU hU').toRationalMap = f.toRationalMap :=
  toRationalMap_eq_iff.mpr (f.restrict_equiv U hU hU')


instance [X.Over S] [Y.Over S] (f : X.PartialMap Y) [f.IsOver S] : f.toRationalMap.IsOver S :=
  ⟨f, ‹_›, rfl⟩


variable (S) in
lemma RationalMap.exists_partialMap_over [X.Over S] [Y.Over S] (f : X ⤏ Y) [f.IsOver S] :
    ∃ g : X.PartialMap Y, g.IsOver S ∧ g.toRationalMap = f :=
  IsOver.exists_partialMap_over


/-- The composition of a rational map and a morphism on the right. -/
def RationalMap.compHom (f : X ⤏ Y) (g : Y ⟶ Z) : X ⤏ Z := by
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    f : X.RationalMap Y
    g : Quiver.Hom Y Z
    ⊢ X.RationalMap Z
  -/
  refine Quotient.map (PartialMap.compHom · g) ?_ f
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    f : X.RationalMap Y
    g : Quiver.Hom Y Z
    ⊢ ∀ ⦃a b : X.PartialMap Y⦄, HasEquiv.Equiv a b → HasEquiv.Equiv ((fun x => x.c …
  -/
  intro f₁ f₂ ⟨W, hW, hWl, hWr, e⟩
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    f : X.RationalMap Y
    g : Quiver.Hom Y Z
    f₁ f₂ : X.PartialMap Y
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f₁.domain
    hWr : LE.le W f₂.domain
    e : Eq (f₁.restrict W hW hWl).hom (f₂.restrict W hW hWr).hom
    ⊢ HasEquiv.Equiv ((fun x => x.compHom g) f₁) ((fun x => x.compHom g) f₂)
  -/
  refine ⟨W, hW, hWl, hWr, ?_⟩
  simp only [PartialMap.restrict_domain, PartialMap.restrict_hom, PartialMap.compHom_domain,
    PartialMap.compHom_hom] at e ⊢
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    f : X.RationalMap Y
    g : Quiver.Hom Y Z
    f₁ f₂ : X.PartialMap Y
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f₁.domain
    hWr : LE.le W f₂.domain
    e : Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE hWl) f₁.hom) (CategoryTh …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE hWl) (CategoryTheory.Categ …
  -/
  rw [reassoc_of% e]
  /-
    🎉 no goals
  -/


@[simp]
lemma RationalMap.compHom_toRationalMap (f : X.PartialMap Y) (g : Y ⟶ Z) :
    (f.compHom g).toRationalMap = f.toRationalMap.compHom g := rfl


instance [X.Over S] [Y.Over S] [Z.Over S] (f : X ⤏ Y) (g : Y ⟶ Z)
    [f.IsOver S] [g.IsOver S] : (f.compHom g).IsOver S where
  exists_partialMap_over := by
    /-
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝⁴ : X.Over S
      inst✝³ : Y.Over S
      inst✝² : Z.Over S
      f : X.RationalMap Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver g S
      ⊢ Exists fun g_1 => And (AlgebraicGeometry.Scheme.PartialMap.IsOver S g_1) (Eq …
    -/
    obtain ⟨f, hf, rfl⟩ := f.exists_partialMap_over S
    /-
      case intro.intro
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝⁴ : X.Over S
      inst✝³ : Y.Over S
      inst✝² : Z.Over S
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.Scheme.Hom.IsOver g S
      f : X.PartialMap Y
      hf : AlgebraicGeometry.Scheme.PartialMap.IsOver S f
      inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
      ⊢ Exists fun g_1 => And (AlgebraicGeometry.Scheme.PartialMap.IsOver S g_1) (Eq …
    -/
    exact ⟨f.compHom g, inferInstance, rfl⟩
    /-
      🎉 no goals
    -/


variable (S) in
lemma PartialMap.exists_restrict_isOver [X.Over S] [Y.Over S] (f : X.PartialMap Y)
    [f.toRationalMap.IsOver S] : ∃ U hU hU', (f.restrict U hU hU').IsOver S := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝² : X.Over S
    inst✝¹ : Y.Over S
    f : X.PartialMap Y
    inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    ⊢ Exists fun U => Exists fun hU => Exists fun hU' => AlgebraicGeometry.Scheme. …
  -/
  obtain ⟨f', hf₁, hf₂⟩ := RationalMap.IsOver.exists_partialMap_over (S := S) (f := f.toRationalMap)
  /-
    case intro.intro
    X Y S : AlgebraicGeometry.Scheme
    inst✝² : X.Over S
    inst✝¹ : Y.Over S
    f : X.PartialMap Y
    inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    f' : X.PartialMap Y
    hf₁ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f'
    hf₂ : Eq f'.toRationalMap f.toRationalMap
    ⊢ Exists fun U => Exists fun hU => Exists fun hU' => AlgebraicGeometry.Scheme. …
  -/
  obtain ⟨U, hU, hUl, hUr, e⟩ := PartialMap.toRationalMap_eq_iff.mp hf₂
  /-
    case intro.intro.intro.intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    inst✝² : X.Over S
    inst✝¹ : Y.Over S
    f : X.PartialMap Y
    inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    f' : X.PartialMap Y
    hf₁ : AlgebraicGeometry.Scheme.PartialMap.IsOver S f'
    hf₂ : Eq f'.toRationalMap f.toRationalMap
    U : X.Opens
    hU : Dense ↑U
    hUl : LE.le U f'.domain
    hUr : LE.le U f.domain
    e : Eq (f'.restrict U hU hUl).hom (f.restrict U hU hUr).hom
    ⊢ Exists fun U => Exists fun hU => Exists fun hU' => AlgebraicGeometry.Scheme. …
  -/
  exact ⟨U, hU, hUr, by rw [IsOver, ← e]; infer_instance⟩
  /-
    🎉 no goals
  -/


lemma RationalMap.isOver_iff [X.Over S] [Y.Over S] {f : X ⤏ Y} :
    f.IsOver S ↔ f.compHom (Y ↘ S) = (X ↘ S).toRationalMap := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : X.Over S
    inst✝ : Y.Over S
    f : X.RationalMap Y
    ⊢ Iff (AlgebraicGeometry.Scheme.RationalMap.IsOver S f) (Eq (f.compHom (Catego …
  -/
  constructor
    /-
      case mp
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.RationalMap Y
      ⊢ AlgebraicGeometry.Scheme.RationalMap.IsOver S f → Eq (f.compHom (CategoryThe …
    -/
  · intro h
    /-
      case mp
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.RationalMap Y
      h : AlgebraicGeometry.Scheme.RationalMap.IsOver S f
      ⊢ Eq (f.compHom (CategoryTheory.over Y S inferInstance)) (AlgebraicGeometry.Sc …
    -/
    obtain ⟨g, hg, e⟩ := f.exists_partialMap_over S
    rw [← e, Hom.toRationalMap, ← compHom_toRationalMap, PartialMap.isOver_iff_eq_restrict.mp hg,
      PartialMap.restrict_toRationalMap]
    /-
      case mpr
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.RationalMap Y
      ⊢ Eq (f.compHom (CategoryTheory.over Y S inferInstance)) (AlgebraicGeometry.Sc …
    -/
  · intro e
    /-
      case mpr
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.RationalMap Y
      e : Eq (f.compHom (CategoryTheory.over Y S inferInstance)) (AlgebraicGeometry. …
      ⊢ AlgebraicGeometry.Scheme.RationalMap.IsOver S f
    -/
    obtain ⟨f, rfl⟩ := PartialMap.toRationalMap_surjective f
    /-
      case mpr.intro
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.PartialMap Y
      e : Eq (f.toRationalMap.compHom (CategoryTheory.over Y S inferInstance)) (Alge …
      ⊢ AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    -/
    obtain ⟨U, hU, hUl, hUr, e⟩ := PartialMap.toRationalMap_eq_iff.mp e
    /-
      case mpr.intro.intro.intro.intro.intro
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : X.Over S
      inst✝ : Y.Over S
      f : X.PartialMap Y
      e✝ : Eq (f.toRationalMap.compHom (CategoryTheory.over Y S inferInstance)) (Alg …
      U : X.Opens
      hU : Dense ↑U
      hUl : LE.le U ((fun x => x.compHom (CategoryTheory.over Y S inferInstance)) f) …
      hUr : LE.le U (AlgebraicGeometry.Scheme.Hom.toPartialMap (CategoryTheory.over  …
      e : Eq (((fun x => x.compHom (CategoryTheory.over Y S inferInstance)) f).restr …
      ⊢ AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    -/
    exact ⟨⟨f.restrict U hU hUl, by simpa using e, by simp⟩⟩
    /-
      🎉 no goals
    -/


lemma PartialMap.isOver_toRationalMap_iff_of_isSeparated [X.Over S] [Y.Over S] [IsReduced X]
    [S.IsSeparated] {f : X.PartialMap Y} :
    f.toRationalMap.IsOver S ↔ f.IsOver S := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝³ : X.Over S
    inst✝² : Y.Over S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : S.IsSeparated
    f : X.PartialMap Y
    ⊢ Iff (AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap) (Algebra …
  -/
  refine ⟨fun _ ↦ ?_, fun _ ↦ inferInstance⟩
  /-
    X Y S : AlgebraicGeometry.Scheme
    inst✝³ : X.Over S
    inst✝² : Y.Over S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : S.IsSeparated
    f : X.PartialMap Y
    x✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    ⊢ AlgebraicGeometry.Scheme.PartialMap.IsOver S f
  -/
  obtain ⟨U, hU, hU', H⟩ := f.exists_restrict_isOver (S := S)
  /-
    case intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    inst✝³ : X.Over S
    inst✝² : Y.Over S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : S.IsSeparated
    f : X.PartialMap Y
    x✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    H : AlgebraicGeometry.Scheme.PartialMap.IsOver S (f.restrict U hU hU')
    ⊢ AlgebraicGeometry.Scheme.PartialMap.IsOver S f
  -/
  rw [isOver_iff]
  /-
    case intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    inst✝³ : X.Over S
    inst✝² : Y.Over S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : S.IsSeparated
    f : X.PartialMap Y
    x✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    H : AlgebraicGeometry.Scheme.PartialMap.IsOver S (f.restrict U hU hU')
    ⊢ Eq (f.compHom (CategoryTheory.over Y S inferInstance)).hom (CategoryTheory.C …
  -/
  have : IsDominant (X.homOfLE hU') := Opens.isDominant_homOfLE hU _
  /-
    case intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    inst✝³ : X.Over S
    inst✝² : Y.Over S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : S.IsSeparated
    f : X.PartialMap Y
    x✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f.toRationalMap
    U : X.Opens
    hU : Dense ↑U
    hU' : LE.le U f.domain
    H : AlgebraicGeometry.Scheme.PartialMap.IsOver S (f.restrict U hU hU')
    this : AlgebraicGeometry.IsDominant (X.homOfLE hU')
    ⊢ Eq (f.compHom (CategoryTheory.over Y S inferInstance)).hom (CategoryTheory.C …
  -/
  exact ext_of_isDominant (ι := X.homOfLE hU') (by simpa using H.1)
  /-
    🎉 no goals
  -/


/-- A rational map restricts to a map from `Spec K(X)`. -/
noncomputable
def RationalMap.fromFunctionField [IrreducibleSpace X] (f : X ⤏ Y) :
    Spec X.functionField ⟶ Y := by
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    f : X.RationalMap Y
    ⊢ Quiver.Hom (AlgebraicGeometry.Spec X.functionField) Y
  -/
  refine Quotient.lift PartialMap.fromFunctionField ?_ f
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    f : X.RationalMap Y
    ⊢ ∀ (a b : X.PartialMap Y), HasEquiv.Equiv a b → Eq a.fromFunctionField b.from …
  -/
  intro f g ⟨W, hW, hWl, hWr, e⟩
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    f✝ : X.RationalMap Y
    f g : X.PartialMap Y
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f.domain
    hWr : LE.le W g.domain
    e : Eq (f.restrict W hW hWl).hom (g.restrict W hW hWr).hom
    ⊢ Eq f.fromFunctionField g.fromFunctionField
  -/
  have : f.restrict W hW hWl = g.restrict W hW hWr := by ext1; rfl; rw [e]; simp
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝ : IrreducibleSpace ↑↑X.toPresheafedSpace
    f✝ : X.RationalMap Y
    f g : X.PartialMap Y
    W : X.Opens
    hW : Dense ↑W
    hWl : LE.le W f.domain
    hWr : LE.le W g.domain
    e : Eq (f.restrict W hW hWl).hom (g.restrict W hW hWr).hom
    this : Eq (f.restrict W hW hWl) (g.restrict W hW hWr)
    ⊢ Eq f.fromFunctionField g.fromFunctionField
  -/
  rw [← f.fromFunctionField_restrict hW hWl, this, g.fromFunctionField_restrict]
  /-
    🎉 no goals
  -/


@[simp]
lemma RationalMap.fromFunctionField_toRationalMap [IrreducibleSpace X] (f : X.PartialMap Y) :
    f.toRationalMap.fromFunctionField = f.fromFunctionField := rfl


/--
Given `S`-schemes `X` and `Y` such that `Y` is locally of finite type and `X` is integral,
any `S`-morphism `Spec K(X) ⟶ Y` spreads out to a rational map from `X` to `Y`.
-/
noncomputable
def RationalMap.ofFunctionField [IsIntegral X] [LocallyOfFiniteType sY]
    (f : Spec X.functionField ⟶ Y) (h : f ≫ sY = X.fromSpecStalk _ ≫ sX) : X ⤏ Y :=
  (PartialMap.ofFromSpecStalk sX sY f h).toRationalMap


lemma RationalMap.fromFunctionField_ofFunctionField [IsIntegral X] [LocallyOfFiniteType sY]
    (f : Spec X.functionField ⟶ Y) (h : f ≫ sY = X.fromSpecStalk _ ≫ sX) :
    (ofFunctionField sX sY f h).fromFunctionField = f :=
  PartialMap.fromSpecStalkOfMem_ofFromSpecStalk sX sY _ _


lemma RationalMap.eq_of_fromFunctionField_eq [IsIntegral X] (f g : X.RationalMap Y)
    (H : f.fromFunctionField = g.fromFunctionField) : f = g := by
    /-
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      f g : X.RationalMap Y
      H : Eq f.fromFunctionField g.fromFunctionField
      ⊢ Eq f g
    -/
    obtain ⟨f, rfl⟩ := f.exists_rep
    /-
      case intro
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      g : X.RationalMap Y
      f : X.PartialMap Y
      H : Eq f.toRationalMap.fromFunctionField g.fromFunctionField
      ⊢ Eq f.toRationalMap g
    -/
    obtain ⟨g, rfl⟩ := g.exists_rep
    /-
      case intro.intro
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      f g : X.PartialMap Y
      H : Eq f.toRationalMap.fromFunctionField g.toRationalMap.fromFunctionField
      ⊢ Eq f.toRationalMap g.toRationalMap
    -/
    refine PartialMap.toRationalMap_eq_iff.mpr ?_
    /-
      case intro.intro
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      f g : X.PartialMap Y
      H : Eq f.toRationalMap.fromFunctionField g.toRationalMap.fromFunctionField
      ⊢ f.equiv g
    -/
    exact PartialMap.equiv_of_fromSpecStalkOfMem_eq _ _ _ _ H
    /-
      🎉 no goals
    -/


/--
Given `S`-schemes `X` and `Y` such that `Y` is locally of finite type and `X` is integral,
`S`-morphisms `Spec K(X) ⟶ Y` correspond bijectively to `S`-rational maps from `X` to `Y`.
-/
noncomputable
def RationalMap.equivFunctionField [IsIntegral X] [LocallyOfFiniteType sY] :
    { f : Spec X.functionField ⟶ Y // f ≫ sY = X.fromSpecStalk _ ≫ sX } ≃
      { f : X ⤏ Y // f.compHom sY = sX.toRationalMap } where
  toFun f := ⟨.ofFunctionField sX sY f f.2, PartialMap.toRationalMap_eq_iff.mpr
                                                        /-
                                                          X Y Z S : AlgebraicGeometry.Scheme
                                                          sX : Quiver.Hom X S
                                                          sY : Quiver.Hom Y S
                                                          inst✝¹ : AlgebraicGeometry.IsIntegral X
                                                          inst✝ : AlgebraicGeometry.LocallyOfFiniteType sY
                                                          f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp f sY) (CategoryThe …
                                                          ⊢ Eq (((fun x => x.compHom sY) (AlgebraicGeometry.Scheme.PartialMap.ofFromSpec …
                                                        -/
      ⟨_, PartialMap.dense_domain _, le_rfl, le_top, by simp [PartialMap.ofFromSpecStalk_comp]⟩⟩
                                                        /-
                                                          🎉 no goals
                                                        -/
  invFun f := ⟨f.1.fromFunctionField, by
    /-
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsIntegral X
      inst✝ : AlgebraicGeometry.LocallyOfFiniteType sY
      f : Subtype fun f => Eq (f.compHom sY) (AlgebraicGeometry.Scheme.Hom.toRationa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑f).fromFunctionField sY) (CategoryT …
    -/
    obtain ⟨f, hf⟩ := f
    /-
      case mk
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsIntegral X
      inst✝ : AlgebraicGeometry.LocallyOfFiniteType sY
      f : X.RationalMap Y
      hf : Eq (f.compHom sY) (AlgebraicGeometry.Scheme.Hom.toRationalMap sX)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑⟨f, hf⟩).fromFunctionField sY) (Cat …
    -/
    obtain ⟨f, rfl⟩ := f.exists_rep
    /-
      case mk.intro
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsIntegral X
      inst✝ : AlgebraicGeometry.LocallyOfFiniteType sY
      f : X.PartialMap Y
      hf : Eq (f.toRationalMap.compHom sY) (AlgebraicGeometry.Scheme.Hom.toRationalM …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑⟨f.toRationalMap, hf⟩).fromFunction …
    -/
    simpa [fromFunctionField_toRationalMap] using congr(RationalMap.fromFunctionField $hf)⟩
    /-
      🎉 no goals
    -/
  left_inv f := Subtype.ext (RationalMap.fromFunctionField_ofFunctionField _ _ _ _)
  right_inv f := Subtype.ext (RationalMap.eq_of_fromFunctionField_eq
      (ofFunctionField sX sY f.1.fromFunctionField _) f
      (RationalMap.fromFunctionField_ofFunctionField _ _ _ _))


/--
Given `S`-schemes `X` and `Y` such that `Y` is locally of finite type and `X` is integral,
`S`-morphisms `Spec K(X) ⟶ Y` correspond bijectively to `S`-rational maps from `X` to `Y`.
-/
noncomputable
def RationalMap.equivFunctionFieldOver [X.Over S] [Y.Over S] [IsIntegral X]
    [LocallyOfFiniteType (Y ↘ S)] :
    { f : Spec X.functionField ⟶ Y // f.IsOver S } ≃ { f : X ⤏ Y // f.IsOver S } :=
                               /-
                                 X Y Z S : AlgebraicGeometry.Scheme
                                 sX : Quiver.Hom X S
                                 sY : Quiver.Hom Y S
                                 inst✝³ : X.Over S
                                 inst✝² : Y.Over S
                                 inst✝¹ : AlgebraicGeometry.IsIntegral X
                                 inst✝ : AlgebraicGeometry.LocallyOfFiniteType (CategoryTheory.over Y S inferIn …
                                 ⊢ Eq (fun f => AlgebraicGeometry.Scheme.Hom.IsOver f S) fun f => Eq (CategoryT …
                               -/
  ((Equiv.subtypeEquivProp (by simp only [Hom.isOver_iff]; rfl)).trans
                                                           /-
                                                             🎉 no goals
                                                           -/
    (RationalMap.equivFunctionField (X ↘ S) (Y ↘ S))).trans
                                  /-
                                    X Y Z S : AlgebraicGeometry.Scheme
                                    sX : Quiver.Hom X S
                                    sY : Quiver.Hom Y S
                                    inst✝³ : X.Over S
                                    inst✝² : Y.Over S
                                    inst✝¹ : AlgebraicGeometry.IsIntegral X
                                    inst✝ : AlgebraicGeometry.LocallyOfFiniteType (CategoryTheory.over Y S inferIn …
                                    ⊢ Eq (fun f => Eq (f.compHom (CategoryTheory.over Y S inferInstance)) (Algebra …
                                  -/
      (Equiv.subtypeEquivProp (by ext f; rw [RationalMap.isOver_iff]))
                                         /-
                                           🎉 no goals
                                         -/


/-- The domain of definition of a rational map. -/
def RationalMap.domain (f : X ⤏ Y) : X.Opens :=
  sSup { PartialMap.domain g | (g) (_ : g.toRationalMap = f) }


lemma PartialMap.le_domain_toRationalMap (f : X.PartialMap Y) :
    f.domain ≤ f.toRationalMap.domain :=
  le_sSup ⟨f, rfl, rfl⟩


lemma RationalMap.mem_domain {f : X ⤏ Y} {x} :
    x ∈ f.domain ↔ ∃ g : X.PartialMap Y, x ∈ g.domain ∧ g.toRationalMap = f :=
                                            /-
                                              X Y : AlgebraicGeometry.Scheme
                                              f : X.RationalMap Y
                                              x : ↑↑X.toPresheafedSpace
                                              ⊢ Iff (Exists fun u => And (Membership.mem (setOf fun x => Exists fun g => Exi …
                                            -/
  TopologicalSpace.Opens.mem_sSup.trans (by simp [@and_comm (x ∈ _)])
                                            /-
                                              🎉 no goals
                                            -/


lemma RationalMap.dense_domain (f : X ⤏ Y) : Dense (X := X) f.domain :=
  f.inductionOn (fun g ↦ g.dense_domain.mono g.le_domain_toRationalMap)


/-- The open cover of the domain of `f : X ⤏ Y`,
consisting of all the domains of the partial maps in the equivalence class. -/
noncomputable
def RationalMap.openCoverDomain (f : X ⤏ Y) : f.domain.toScheme.OpenCover where
  J := { PartialMap.domain g | (g) (_ : g.toRationalMap = f) }
  obj U := U.1.toScheme
  map U := X.homOfLE (le_sSup U.2)
  f x := ⟨_, (TopologicalSpace.Opens.mem_sSup.mp x.2).choose_spec.1⟩
                                                                                              /-
                                                                                                X Y Z S : AlgebraicGeometry.Scheme
                                                                                                sX : Quiver.Hom X S
                                                                                                sY : Quiver.Hom Y S
                                                                                                f : X.RationalMap Y
                                                                                                x : ↑↑(↑f.domain).toPresheafedSpace
                                                                                                ⊢ Eq ↑(((fun U => X.homOfLE ⋯) ((fun x => ⟨(Classical.indefiniteDescription (f …
                                                                                              -/
  covers x := ⟨⟨x.1, (TopologicalSpace.Opens.mem_sSup.mp x.2).choose_spec.2⟩, Subtype.ext (by simp)⟩
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


/-- If `f : X ⤏ Y` is a rational map from a reduced scheme to a separated scheme,
then `f` can be represented as a partial map on its domain of definition. -/
noncomputable
def RationalMap.toPartialMap [IsReduced X] [Y.IsSeparated] (f : X ⤏ Y) : X.PartialMap Y := by
  refine ⟨f.domain, f.dense_domain, f.openCoverDomain.glueMorphisms
    (fun x ↦ (X.isoOfEq x.2.choose_spec.2).inv ≫ x.2.choose.hom) ?_⟩
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    ⊢ ∀ (x y : f.openCoverDomain.J), Eq (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  intros x y
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  let g (x : f.openCoverDomain.J) := x.2.choose
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  have hg₁ (x) : (g x).toRationalMap = f := x.2.choose_spec.1
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  have hg₂ (x) : (g x).domain = x.1 := x.2.choose_spec.2
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
    hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  refine (cancel_epi (isPullback_opens_inf_le (le_sSup x.2) (le_sSup y.2)).isoPullback.hom).mp ?_
  simp only [openCoverDomain, IsPullback.isoPullback_hom_fst_assoc,
    IsPullback.isoPullback_hom_snd_assoc]
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
    hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) (CategoryTheory.Categor …
  -/
  show _ ≫ _ ≫ (g x).hom = _ ≫ _ ≫ (g y).hom
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
    hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) (CategoryTheory.Categor …
  -/
  simp_rw [← cancel_epi (X.isoOfEq congr($(hg₂ x) ⊓ $(hg₂ y))).hom, ← Category.assoc]
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    x y : f.openCoverDomain.J
    g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
    hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
    hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  convert (PartialMap.equiv_iff_of_isSeparated (S := ⊤_ _) (f := g x) (g := g y)).mp ?_ using 1
    /-
      case h.e'_2.h
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.RationalMap Y
      x y : f.openCoverDomain.J
      g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
      hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
      hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
      e_1✝ : Eq (Quiver.Hom (↑(Min.min (Mathlib.Tactic.TermCongr.cHole (g x).domain  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp; congr 1; simp [g, ← cancel_mono (Opens.ι _)]
                    /-
                      🎉 no goals
                    -/
    /-
      case h.e'_3.h
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.RationalMap Y
      x y : f.openCoverDomain.J
      g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
      hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
      hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
      e_1✝ : Eq (Quiver.Hom (↑(Min.min (Mathlib.Tactic.TermCongr.cHole (g x).domain  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp; congr 1; simp [g, ← cancel_mono (Opens.ι _)]
                    /-
                      🎉 no goals
                    -/
    /-
      X Y Z S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.RationalMap Y
      x y : f.openCoverDomain.J
      g : f.openCoverDomain.J → X.PartialMap Y := fun x => Exists.choose ⋯
      hg₁ : ∀ (x : f.openCoverDomain.J), Eq (g x).toRationalMap f
      hg₂ : ∀ (x : f.openCoverDomain.J), Eq (g x).domain ↑x
      ⊢ (g x).equiv (g y)
    -/
  · rw [← PartialMap.toRationalMap_eq_iff, hg₁, hg₁]
    /-
      🎉 no goals
    -/


lemma PartialMap.toPartialMap_toRationalMap_restrict [IsReduced X] [Y.IsSeparated]
    (f : X.PartialMap Y) : (f.toRationalMap.toPartialMap.restrict _ f.dense_domain
      f.le_domain_toRationalMap).hom = f.hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.PartialMap Y
    ⊢ Eq (f.toRationalMap.toPartialMap.restrict f.domain ⋯ ⋯).hom f.hom
  -/
  dsimp [RationalMap.toPartialMap]
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.PartialMap Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) (AlgebraicGeometry.Sche …
  -/
  refine (f.toRationalMap.openCoverDomain.ι_glueMorphisms _ _ ⟨_, f, rfl, rfl⟩).trans ?_
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.PartialMap Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.isoOfEq ⋯).inv (Exists.choose ⋯).h …
  -/
  generalize_proofs _ _ H _
  have : H.choose = f := (equiv_iff_of_domain_eq_of_isSeparated (S := ⊤_ _) H.choose_spec.2).mp
    (toRationalMap_eq_iff.mp H.choose_spec.1)
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.PartialMap Y
    pf✝² : Membership.mem (setOf fun x => Exists fun g => Exists fun x_1 => Eq g.d …
    pf✝¹ : Exists fun g => Exists fun x => Eq g.domain f.domain
    H : Exists fun g => Exists fun x => Eq g.domain ↑⟨f.domain, pf✝¹⟩
    pf✝ : Eq H.choose.domain ↑⟨f.domain, pf✝²⟩
    this : Eq H.choose f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.isoOfEq pf✝).inv H.choose.hom) f.hom
  -/
  exact ((ext_iff _ _).mp this.symm).choose_spec.symm
  /-
    🎉 no goals
  -/


@[simp]
lemma RationalMap.toRationalMap_toPartialMap [IsReduced X] [Y.IsSeparated]
    (f : X ⤏ Y) : f.toPartialMap.toRationalMap = f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsReduced X
    inst✝ : Y.IsSeparated
    f : X.RationalMap Y
    ⊢ Eq f.toPartialMap.toRationalMap f
  -/
  obtain ⟨f, rfl⟩ := PartialMap.toRationalMap_surjective f
  trans (f.toRationalMap.toPartialMap.restrict _
    f.dense_domain f.le_domain_toRationalMap).toRationalMap
    /-
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.PartialMap Y
      ⊢ Eq f.toRationalMap.toPartialMap.toRationalMap (f.toRationalMap.toPartialMap. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.PartialMap Y
      ⊢ Eq (f.toRationalMap.toPartialMap.restrict f.domain ⋯ ⋯).toRationalMap f.toRa …
    -/
  · congr 1
    /-
      case e_f
      X Y : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsReduced X
      inst✝ : Y.IsSeparated
      f : X.PartialMap Y
      ⊢ Eq (f.toRationalMap.toPartialMap.restrict f.domain ⋯ ⋯) f
    -/
    exact PartialMap.ext _ f rfl (by simpa using f.toPartialMap_toRationalMap_restrict)
    /-
      🎉 no goals
    -/


instance [IsReduced X] [Y.IsSeparated] [S.IsSeparated] [X.Over S] [Y.Over S]
    (f : X ⤏ Y) [f.IsOver S] : f.toPartialMap.IsOver S := by
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝⁵ : AlgebraicGeometry.IsReduced X
    inst✝⁴ : Y.IsSeparated
    inst✝³ : S.IsSeparated
    inst✝² : X.Over S
    inst✝¹ : Y.Over S
    f : X.RationalMap Y
    inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f
    ⊢ AlgebraicGeometry.Scheme.PartialMap.IsOver S f.toPartialMap
  -/
  rw [← PartialMap.isOver_toRationalMap_iff_of_isSeparated, f.toRationalMap_toPartialMap]
  /-
    X Y Z S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝⁵ : AlgebraicGeometry.IsReduced X
    inst✝⁴ : Y.IsSeparated
    inst✝³ : S.IsSeparated
    inst✝² : X.Over S
    inst✝¹ : Y.Over S
    f : X.RationalMap Y
    inst✝ : AlgebraicGeometry.Scheme.RationalMap.IsOver S f
    ⊢ AlgebraicGeometry.Scheme.RationalMap.IsOver S f
  -/
  infer_instance
  /-
    🎉 no goals
  -/


