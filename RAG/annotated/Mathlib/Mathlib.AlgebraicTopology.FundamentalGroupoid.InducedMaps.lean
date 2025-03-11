/-- The path 0 ⟶ 1 in `I` -/
def path01 : Path (0 : I) 1 where
  toFun := id
  source' := rfl
  target' := rfl


/-- The path 0 ⟶ 1 in `ULift I` -/
def upath01 : Path (ULift.up 0 : ULift.{u} I) (ULift.up 1) where
  toFun := ULift.up
  source' := rfl
  target' := rfl


/-- The homotopy path class of 0 → 1 in `ULift I` -/
def uhpath01 : @fromTop (TopCat.of <| ULift.{u} I) (ULift.up (0 : I)) ⟶ fromTop (ULift.up 1) :=
  ⟦upath01⟧


/-- Abbreviation for `eqToHom` that accepts points in a topological space -/
abbrev hcast {X : TopCat} {x₀ x₁ : X} (hx : x₀ = x₁) : fromTop x₀ ⟶ fromTop x₁ :=
  eqToHom <| FundamentalGroupoid.ext hx


@[simp]
theorem hcast_def {X : TopCat} {x₀ x₁ : X} (hx₀ : x₀ = x₁) :
    hcast hx₀ = eqToHom (FundamentalGroupoid.ext hx₀) :=
  rfl


/-- If `f(p(t) = g(q(t))` for two paths `p` and `q`, then the induced path homotopy classes
`f(p)` and `g(p)` are the same as well, despite having a priori different types -/
theorem heq_path_of_eq_image : HEq ((πₘ f).map ⟦p⟧) ((πₘ g).map ⟦q⟧) := by
  /-
    X₁ X₂ Y : TopCat
    f : ContinuousMap ↑X₁ ↑Y
    g : ContinuousMap ↑X₂ ↑Y
    x₀ x₁ : ↑X₁
    x₂ x₃ : ↑X₂
    p : Path x₀ x₁
    q : Path x₂ x₃
    hfg : ∀ (t : ↑unitInterval), Eq (f (p t)) (g (q t))
    ⊢ HEq ((FundamentalGroupoid.fundamentalGroupoidFunctor.map f).map (Quotient.mk …
  -/
  simp only [map_eq, ← Path.Homotopic.map_lift]; apply Path.Homotopic.hpath_hext; exact hfg
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


                                               /-
                                                 X₁ X₂ Y : TopCat
                                                 f : ContinuousMap ↑X₁ ↑Y
                                                 g : ContinuousMap ↑X₂ ↑Y
                                                 x₀ x₁ : ↑X₁
                                                 x₂ x₃ : ↑X₂
                                                 p : Path x₀ x₁
                                                 q : Path x₂ x₃
                                                 hfg : ∀ (t : ↑unitInterval), Eq (f (p t)) (g (q t))
                                                 ⊢ Eq (f x₀) (g x₂)
                                               -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
private theorem start_path : f x₀ = g x₂ := by convert hfg 0 <;> simp only [Path.source]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                             /-
                                               X₁ X₂ Y : TopCat
                                               f : ContinuousMap ↑X₁ ↑Y
                                               g : ContinuousMap ↑X₂ ↑Y
                                               x₀ x₁ : ↑X₁
                                               x₂ x₃ : ↑X₂
                                               p : Path x₀ x₁
                                               q : Path x₂ x₃
                                               hfg : ∀ (t : ↑unitInterval), Eq (f (p t)) (g (q t))
                                               ⊢ Eq (f x₁) (g x₃)
                                             -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
private theorem end_path : f x₁ = g x₃ := by convert hfg 1 <;> simp only [Path.target]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem eq_path_of_eq_image :
    (πₘ f).map ⟦p⟧ = hcast (start_path hfg) ≫ (πₘ g).map ⟦q⟧ ≫ hcast (end_path hfg).symm := by
  rw [conj_eqToHom_iff_heq
    ((πₘ f).map ⟦p⟧) ((πₘ g).map ⟦q⟧)
    (FundamentalGroupoid.ext <| start_path hfg)
    (FundamentalGroupoid.ext <| end_path hfg)]
  /-
    X₁ X₂ Y : TopCat
    f : ContinuousMap ↑X₁ ↑Y
    g : ContinuousMap ↑X₂ ↑Y
    x₀ x₁ : ↑X₁
    x₂ x₃ : ↑X₂
    p : Path x₀ x₁
    q : Path x₂ x₃
    hfg : ∀ (t : ↑unitInterval), Eq (f (p t)) (g (q t))
    ⊢ HEq ((FundamentalGroupoid.fundamentalGroupoidFunctor.map f).map (Quotient.mk …
  -/
  exact heq_path_of_eq_image hfg
  /-
    🎉 no goals
  -/


/-- Interpret a homotopy `H : C(I × X, Y)` as a map `C(ULift I × X, Y)` -/
def uliftMap : C(TopCat.of (ULift.{u} I × X), Y) :=
  ⟨fun x => H (x.1.down, x.2),
    H.continuous.comp ((continuous_uLift_down.comp continuous_fst).prod_mk continuous_snd)⟩

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem ulift_apply (i : ULift.{u} I) (x : X) : H.uliftMap (i, x) = H (i.down, x) :=
  rfl


/-- An abbreviation for `prodToProdTop`, with some types already in place to help the
 typechecker. In particular, the first path should be on the ulifted unit interval. -/
abbrev prodToProdTopI {a₁ a₂ : TopCat.of (ULift I)} {b₁ b₂ : X} (p₁ : fromTop a₁ ⟶ fromTop a₂)
    (p₂ : fromTop b₁ ⟶ fromTop b₂) :=
  (prodToProdTop (TopCat.of <| ULift I) X).map (X := (⟨a₁⟩, ⟨b₁⟩)) (Y := (⟨a₂⟩, ⟨b₂⟩)) (p₁, p₂)


/-- The diagonal path `d` of a homotopy `H` on a path `p` -/
def diagonalPath : fromTop (H (0, x₀)) ⟶ fromTop (H (1, x₁)) :=
  (πₘ H.uliftMap).map (prodToProdTopI uhpath01 p)


/-- The diagonal path, but starting from `f x₀` and going to `g x₁` -/
def diagonalPath' : fromTop (f x₀) ⟶ fromTop (g x₁) :=
  hcast (H.apply_zero x₀).symm ≫ H.diagonalPath p ≫ hcast (H.apply_one x₁)


/-- Proof that `f(p) = H(0 ⟶ 0, p)`, with the appropriate casts -/
theorem apply_zero_path : (πₘ f).map p = hcast (H.apply_zero x₀).symm ≫
    (πₘ H.uliftMap).map (prodToProdTopI (𝟙 (@fromTop (TopCat.of _) (ULift.up 0))) p) ≫
    hcast (H.apply_zero x₁) :=
  Quotient.inductionOn p fun p' => by
    /-
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      p' : Path (FundamentalGroupoid.fromTop x₀).as (FundamentalGroupoid.fromTop x₁) …
      ⊢ Eq ((FundamentalGroupoid.fundamentalGroupoidFunctor.map f).map (Quotient.mk  …
    -/
    apply @eq_path_of_eq_image _ _ _ _ H.uliftMap _ _ _ _ _ ((Path.refl (ULift.up _)).prod p')
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      p' : Path (FundamentalGroupoid.fromTop x₀).as (FundamentalGroupoid.fromTop x₁) …
      ⊢ ∀ (t : ↑unitInterval), Eq (f (p' t)) (H.uliftMap (((Path.refl { down := 0 }) …
    -/
    erw [Path.prod_coe]; simp_rw [ulift_apply]; simp
                                                /-
                                                  🎉 no goals
                                                -/


/-- Proof that `g(p) = H(1 ⟶ 1, p)`, with the appropriate casts -/
theorem apply_one_path : (πₘ g).map p = hcast (H.apply_one x₀).symm ≫
    (πₘ H.uliftMap).map (prodToProdTopI (𝟙 (@fromTop (TopCat.of _) (ULift.up 1))) p) ≫
    hcast (H.apply_one x₁) :=
  Quotient.inductionOn p fun p' => by
    /-
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      p' : Path (FundamentalGroupoid.fromTop x₀).as (FundamentalGroupoid.fromTop x₁) …
      ⊢ Eq ((FundamentalGroupoid.fundamentalGroupoidFunctor.map g).map (Quotient.mk  …
    -/
    apply @eq_path_of_eq_image _ _ _ _ H.uliftMap _ _ _ _ _ ((Path.refl (ULift.up _)).prod p')
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      p' : Path (FundamentalGroupoid.fromTop x₀).as (FundamentalGroupoid.fromTop x₁) …
      ⊢ ∀ (t : ↑unitInterval), Eq (g (p' t)) (H.uliftMap (((Path.refl { down := 1 }) …
    -/
    erw [Path.prod_coe]; simp_rw [ulift_apply]; simp
                                                /-
                                                  🎉 no goals
                                                -/


/-- Proof that `H.evalAt x = H(0 ⟶ 1, x ⟶ x)`, with the appropriate casts -/
theorem evalAt_eq (x : X) : ⟦H.evalAt x⟧ = hcast (H.apply_zero x).symm ≫
    (πₘ H.uliftMap).map (prodToProdTopI uhpath01 (𝟙 (fromTop x))) ≫
      hcast (H.apply_one x).symm.symm := by
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x : ↑X
    ⊢ Eq (Quotient.mk (Path.Homotopic.setoid (FundamentalGroupoid.fromTop (f x)).a …
  -/
  dsimp only [prodToProdTopI, uhpath01, hcast]
  refine (@conj_eqToHom_iff_heq (πₓ Y) _ _ _ _ _ _ _ _
    (FundamentalGroupoid.ext <| H.apply_one x).symm).mpr ?_
  simp only [id_eq_path_refl, prodToProdTop_map, Path.Homotopic.prod_lift, map_eq, ←
    Path.Homotopic.map_lift]
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x : ↑X
    ⊢ HEq (Quotient.mk (Path.Homotopic.setoid (f x) (g x)) (H.evalAt x)) (Path.Hom …
  -/
  apply Path.Homotopic.hpath_hext; intro; rfl
                                          /-
                                            🎉 no goals
                                          -/

-- Finally, we show `d = f(p) ≫ H₁ = H₀ ≫ g(p)`

theorem eq_diag_path : (πₘ f).map p ≫ ⟦H.evalAt x₁⟧ = H.diagonalPath' p ∧
    (⟦H.evalAt x₀⟧ ≫ (πₘ g).map p : fromTop (f x₀) ⟶ fromTop (g x₁)) = H.diagonalPath' p := by
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x₀ x₁ : ↑X
    p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((FundamentalGroupoid.fundamenta …
  -/
  rw [H.apply_zero_path, H.apply_one_path, H.evalAt_eq]
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x₀ x₁ : ↑X
    p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
  -/
  erw [H.evalAt_eq] -- Porting note: `rw` didn't work, so using `erw`
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x₀ x₁ : ↑X
    p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
  -/
  dsimp only [prodToProdTopI]
  /-
    X Y : TopCat
    f g : ContinuousMap ↑X ↑Y
    H : f.Homotopy g
    x₀ x₁ : ↑X
    p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
  -/
  constructor
    /-
      case left
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · slice_lhs 2 4 => rw [eqToHom_trans, eqToHom_refl] -- Porting note: this ↓ `simp` didn't do this
    /-
      case left
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ContinuousMap.Homotopy.hcast ⋯) (Cat …
    -/
    slice_lhs 2 4 => simp [← CategoryTheory.Functor.map_comp]
    /-
      case left
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ContinuousMap.Homotopy.hcast ⋯) (Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case right
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · slice_lhs 2 4 => rw [eqToHom_trans, eqToHom_refl] -- Porting note: this ↓ `simp` didn't do this
    /-
      case right
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ContinuousMap.Homotopy.hcast ⋯) (Cat …
    -/
    slice_lhs 2 4 => simp [← CategoryTheory.Functor.map_comp]
    /-
      case right
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      x₀ x₁ : ↑X
      p : Quiver.Hom (FundamentalGroupoid.fromTop x₀) (FundamentalGroupoid.fromTop x₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ContinuousMap.Homotopy.hcast ⋯) (Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Given a homotopy H : f ∼ g, we have an associated natural isomorphism between the induced
functors `f` and `g` -/
-- Porting note: couldn't use category arrow `\hom` in statement, needed to expand
def homotopicMapsNatIso : @Quiver.Hom _ Functor.category.toQuiver (πₘ f) (πₘ g) where
  app x := ⟦H.evalAt x.as⟧
  -- Porting note: Turned `rw` into `erw` in the line below
                         /-
                           X Y : TopCat
                           f g : ContinuousMap ↑X ↑Y
                           H : f.Homotopy g
                           x y : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj X)
                           p : Quiver.Hom x y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((FundamentalGroupoid.fundamentalGrou …
                         -/
  naturality x y p := by erw [(H.eq_diag_path p).1, (H.eq_diag_path p).2]
                         /-
                           🎉 no goals
                         -/


                                               /-
                                                 X Y : TopCat
                                                 f g : ContinuousMap ↑X ↑Y
                                                 H : f.Homotopy g
                                                 ⊢ CategoryTheory.IsIso (FundamentalGroupoidFunctor.homotopicMapsNatIso H)
                                               -/
instance : IsIso (homotopicMapsNatIso H) := by apply NatIso.isIso_of_isIso_app
                                               /-
                                                 🎉 no goals
                                               -/


/-- Homotopy equivalent topological spaces have equivalent fundamental groupoids. -/
def equivOfHomotopyEquiv (hequiv : X ≃ₕ Y) : πₓ X ≌ πₓ Y := by
  apply CategoryTheory.Equivalence.mk (πₘ hequiv.toFun : πₓ X ⥤ πₓ Y)
    (πₘ hequiv.invFun : πₓ Y ⥤ πₓ X) <;>
    /-
      case η
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      hequiv : ContinuousMap.HomotopyEquiv ↑X ↑Y
      ⊢ CategoryTheory.Iso (CategoryTheory.Functor.id ↑(FundamentalGroupoid.fundamen …
    -/
    simp only [Grpd.hom_to_functor, Grpd.id_to_functor]
    /-
      case η
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      hequiv : ContinuousMap.HomotopyEquiv ↑X ↑Y
      ⊢ CategoryTheory.Iso (CategoryTheory.CategoryStruct.id (FundamentalGroupoid.fu …
    -/
  · convert (asIso (homotopicMapsNatIso hequiv.left_inv.some)).symm
    /-
      case h.e'_3
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      hequiv : ContinuousMap.HomotopyEquiv ↑X ↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.id (FundamentalGroupoid.fundamentalGroupoi …
    -/
    exacts [((π).map_id X).symm, ((π).map_comp _ _).symm]
    /-
      🎉 no goals
    -/
    /-
      case ε
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      hequiv : ContinuousMap.HomotopyEquiv ↑X ↑Y
      ⊢ CategoryTheory.Iso (CategoryTheory.Functor.comp (FundamentalGroupoid.fundame …
    -/
  · convert asIso (homotopicMapsNatIso hequiv.right_inv.some)
    /-
      case h.e'_3
      X Y : TopCat
      f g : ContinuousMap ↑X ↑Y
      H : f.Homotopy g
      hequiv : ContinuousMap.HomotopyEquiv ↑X ↑Y
      ⊢ Eq (CategoryTheory.Functor.comp (FundamentalGroupoid.fundamentalGroupoidFunc …
    -/
    exacts [((π).map_comp _ _).symm, ((π).map_id Y).symm]
    /-
      🎉 no goals
    -/


