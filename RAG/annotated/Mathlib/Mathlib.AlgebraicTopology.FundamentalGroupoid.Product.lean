/-- The projection map Π i, X i → X i induces a map π(Π i, X i) ⟶ π(X i).
-/
def proj (i : I) : πₓ (TopCat.of (∀ i, X i)) ⥤ πₓ (X i) :=
  πₘ ⟨_, continuous_apply i⟩


/-- The projection map is precisely `Path.Homotopic.proj` interpreted as a functor -/
@[simp]
theorem proj_map (i : I) (x₀ x₁ : πₓ (TopCat.of (∀ i, X i))) (p : x₀ ⟶ x₁) :
    (proj X i).map p = @Path.Homotopic.proj _ _ _ _ _ i p :=
  rfl


/-- The map taking the pi product of a family of fundamental groupoids to the fundamental
groupoid of the pi product. This is actually an isomorphism (see `piIso`)
-/
@[simps]
def piToPiTop : (∀ i, πₓ (X i)) ⥤ πₓ (TopCat.of (∀ i, X i)) where
  obj g := ⟨fun i => (g i).as⟩
  map p := Path.Homotopic.pi p
  map_id x := by
    /-
      I : Type u
      X : I → TopCat
      x : (i : I) → ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (X i))
      ⊢ Eq ({ obj := fun g => { as := fun i => (g i).as }, map := fun {X_1 Y} p => P …
    -/
    change (Path.Homotopic.pi fun i => ⟦_⟧) = _
    /-
      I : Type u
      X : I → TopCat
      x : (i : I) → ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (X i))
      ⊢ Eq (Path.Homotopic.pi fun i => Quotient.mk (Path.Homotopic.setoid (((fun g = …
    -/
    simp only [FundamentalGroupoid.id_eq_path_refl, Path.Homotopic.pi_lift]
    /-
      I : Type u
      X : I → TopCat
      x : (i : I) → ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (X i))
      ⊢ Eq (Quotient.mk (Path.Homotopic.setoid (fun i => (x i).as) fun i => (x i).as …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := (Path.Homotopic.comp_pi_eq_pi_comp f g).symm


/-- Shows `piToPiTop` is an isomorphism, whose inverse is precisely the pi product
of the induced projections. This shows that `fundamentalGroupoidFunctor` preserves products.
-/
@[simps]
def piIso : CategoryTheory.Grpd.of (∀ i : I, πₓ (X i)) ≅ πₓ (TopCat.of (∀ i, X i)) where
  hom := piToPiTop X
  inv := CategoryTheory.Functor.pi' (proj X)
  hom_inv_id := by
    /-
      I : Type u
      X : I → TopCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (FundamentalGroupoidFunctor.piToPiTop …
    -/
    change piToPiTop X ⋙ CategoryTheory.Functor.pi' (proj X) = 𝟭 _
    /-
      I : Type u
      X : I → TopCat
      ⊢ Eq ((FundamentalGroupoidFunctor.piToPiTop X).comp (CategoryTheory.Functor.pi …
    -/
    apply CategoryTheory.Functor.ext ?_ ?_
      /-
        I : Type u
        X : I → TopCat
        ⊢ ∀ (X_1 : (i : I) → ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (X i …
      -/
    · intros; rfl
              /-
                🎉 no goals
              -/
      /-
        I : Type u
        X : I → TopCat
        ⊢ ∀ (X_1 Y : (i : I) → ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (X …
      -/
    · intros; ext; simp
                   /-
                     🎉 no goals
                   -/
  inv_hom_id := by
    /-
      I : Type u
      X : I → TopCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.pi' (Fundamen …
    -/
    change CategoryTheory.Functor.pi' (proj X) ⋙ piToPiTop X = 𝟭 _
    /-
      I : Type u
      X : I → TopCat
      ⊢ Eq ((CategoryTheory.Functor.pi' (FundamentalGroupoidFunctor.proj X)).comp (F …
    -/
    apply CategoryTheory.Functor.ext
      /-
        case h_map
        I : Type u
        X : I → TopCat
        ⊢ autoParam (∀ (X_1 Y : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj ( …
      -/
    · intro _ _ f
      /-
        case h_map
        I : Type u
        X : I → TopCat
        X✝ Y✝ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of ((i :  …
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (((CategoryTheory.Functor.pi' (FundamentalGroupoidFunctor.proj X)).comp ( …
      -/
      suffices Path.Homotopic.pi ((CategoryTheory.Functor.pi' (proj X)).map f) = f by simpa
      /-
        case h_map
        I : Type u
        X : I → TopCat
        X✝ Y✝ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of ((i :  …
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (Path.Homotopic.pi ((CategoryTheory.Functor.pi' (FundamentalGroupoidFunct …
      -/
      change Path.Homotopic.pi (fun i => (CategoryTheory.Functor.pi' (proj X)).map f i) = _
      /-
        case h_map
        I : Type u
        X : I → TopCat
        X✝ Y✝ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of ((i :  …
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (Path.Homotopic.pi fun i => (CategoryTheory.Functor.pi' (FundamentalGroup …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h_obj
        I : Type u
        X : I → TopCat
        ⊢ ∀ (X_1 : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of ((i …
      -/
    · intros; rfl
              /-
                🎉 no goals
              -/


/-- Equivalence between the categories of cones over the objects `π Xᵢ` written in two ways -/
def coneDiscreteComp :
    Limits.Cone (Discrete.functor X ⋙ π) ≌ Limits.Cone (Discrete.functor fun i => πₓ (X i)) :=
  Limits.Cones.postcomposeEquivalence (Discrete.compNatIsoDiscrete X π)


theorem coneDiscreteComp_obj_mapCone :
    -- Porting note: check universe parameters here
    (coneDiscreteComp X).functor.obj (Functor.mapCone π (TopCat.piFan.{u,u} X)) =
      Limits.Fan.mk (πₓ (TopCat.of (∀ i, X i))) (proj X) :=
  rfl


/-- This is `piIso.inv` as a cone morphism (in fact, isomorphism) -/
def piTopToPiCone :
    Limits.Fan.mk (πₓ (TopCat.of (∀ i, X i))) (proj X) ⟶ Grpd.piLimitFan fun i : I => πₓ (X i) where
  hom := CategoryTheory.Functor.pi' (proj X)


instance : IsIso (piTopToPiCone X) :=
  haveI : IsIso (piTopToPiCone X).hom := (inferInstance : IsIso (piIso X).inv)
  Limits.Cones.cone_iso_of_hom_iso (piTopToPiCone X)


/-- The fundamental groupoid functor preserves products -/
lemma preservesProduct : Limits.PreservesLimit (Discrete.functor X) π := by
  -- Porting note: check universe parameters here
  /-
    I : Type u
    X : I → TopCat
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor X) Fun …
  -/
  apply Limits.preservesLimit_of_preserves_limit_cone (TopCat.piFanIsLimit.{u,u} X)
  /-
    I : Type u
    X : I → TopCat
    ⊢ CategoryTheory.Limits.IsLimit (FundamentalGroupoid.fundamentalGroupoidFuncto …
  -/
  apply (Limits.IsLimit.ofConeEquiv (coneDiscreteComp X)).toFun
  /-
    I : Type u
    X : I → TopCat
    ⊢ CategoryTheory.Limits.IsLimit ((FundamentalGroupoidFunctor.coneDiscreteComp  …
  -/
  simp only [coneDiscreteComp_obj_mapCone]
  /-
    I : Type u
    X : I → TopCat
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fan.mk (FundamentalGrou …
  -/
  apply Limits.IsLimit.ofIsoLimit _ (asIso (piTopToPiCone X)).symm
  /-
    I : Type u
    X : I → TopCat
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Grpd.piLimitFan fun i => Funda …
  -/
  exact Grpd.piLimitFanIsLimit _
  /-
    🎉 no goals
  -/


/-- The induced map of the left projection map X × Y → X -/
def projLeft : πₓ (TopCat.of (A × B)) ⥤ πₓ A :=
  πₘ ⟨_, continuous_fst⟩


/-- The induced map of the right projection map X × Y → Y -/
def projRight : πₓ (TopCat.of (A × B)) ⥤ πₓ B :=
  πₘ ⟨_, continuous_snd⟩


@[simp]
theorem projLeft_map (x₀ x₁ : πₓ (TopCat.of (A × B))) (p : x₀ ⟶ x₁) :
    (projLeft A B).map p = Path.Homotopic.projLeft p :=
  rfl


@[simp]
theorem projRight_map (x₀ x₁ : πₓ (TopCat.of (A × B))) (p : x₀ ⟶ x₁) :
    (projRight A B).map p = Path.Homotopic.projRight p :=
  rfl


/--
The map taking the product of two fundamental groupoids to the fundamental groupoid of the product
of the two topological spaces. This is in fact an isomorphism (see `prodIso`).
-/
@[simps obj]
def prodToProdTop : πₓ A × πₓ B ⥤ πₓ (TopCat.of (A × B)) where
  obj g := ⟨g.fst.as, g.snd.as⟩
  map {x y} p :=
    match x, y, p with
    | (_, _), (_, _), (p₀, p₁) => @Path.Homotopic.prod _ _ (_) (_) _ _ _ _ p₀ p₁
  map_id := by
    /-
      A B : TopCat
      ⊢ ∀ (X : Prod ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A) ↑(Fundam …
    -/
    rintro ⟨x₀, x₁⟩
    /-
      case mk
      A B : TopCat
      x₀ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A)
      x₁ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj B)
      ⊢ Eq ({ obj := fun g => { as := { fst := g.1.as, snd := g.2.as } }, map := fun …
    -/
    simp only [CategoryTheory.prod_id, FundamentalGroupoid.id_eq_path_refl]
    /-
      case mk
      A B : TopCat
      x₀ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A)
      x₁ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj B)
      ⊢ Eq (Path.Homotopic.prod (CategoryTheory.CategoryStruct.id x₀) (CategoryTheor …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp {x y z} f g :=
    match x, y, z, f, g with
    | (_, _), (_, _), (_, _), (f₀, f₁), (g₀, g₁) =>
      (Path.Homotopic.comp_prod_eq_prod_comp f₀ f₁ g₀ g₁).symm


theorem prodToProdTop_map {x₀ x₁ : πₓ A} {y₀ y₁ : πₓ B} (p₀ : x₀ ⟶ x₁) (p₁ : y₀ ⟶ y₁) :
    (prodToProdTop A B).map (X := (x₀, y₀)) (Y := (x₁, y₁)) (p₀, p₁) =
      Path.Homotopic.prod p₀ p₁ :=
  rfl


/-- Shows `prodToProdTop` is an isomorphism, whose inverse is precisely the product
of the induced left and right projections.
-/
@[simps]
def prodIso : CategoryTheory.Grpd.of (πₓ A × πₓ B) ≅ πₓ (TopCat.of (A × B)) where
  hom := prodToProdTop A B
  inv := (projLeft A B).prod' (projRight A B)
  hom_inv_id := by
    /-
      A B : TopCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (FundamentalGroupoidFunctor.prodToPro …
    -/
    change prodToProdTop A B ⋙ (projLeft A B).prod' (projRight A B) = 𝟭 _
    /-
      A B : TopCat
      ⊢ Eq ((FundamentalGroupoidFunctor.prodToProdTop A B).comp ((FundamentalGroupoi …
    -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    apply CategoryTheory.Functor.hext; · intros; ext <;> simp <;> rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    /-
      case h_map
      A B : TopCat
      ⊢ ∀ (X Y : Prod ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A) ↑(Fund …
    -/
    rintro ⟨x₀, x₁⟩ ⟨y₀, y₁⟩ ⟨f₀, f₁⟩
    have : Path.Homotopic.projLeft ((prodToProdTop A B).map (f₀, f₁)) = f₀ ∧
      Path.Homotopic.projRight ((prodToProdTop A B).map (f₀, f₁)) = f₁ :=
        And.intro (Path.Homotopic.projLeft_prod f₀ f₁) (Path.Homotopic.projRight_prod f₀ f₁)
    /-
      case h_map.mk.mk.mk
      A B : TopCat
      x₀ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A)
      x₁ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj B)
      y₀ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj A)
      y₁ : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj B)
      f₀ : Quiver.Hom { fst := x₀, snd := x₁ }.1 { fst := y₀, snd := y₁ }.1
      f₁ : Quiver.Hom { fst := x₀, snd := x₁ }.2 { fst := y₀, snd := y₁ }.2
      this : And (Eq (Path.Homotopic.projLeft ((FundamentalGroupoidFunctor.prodToPro …
      ⊢ HEq (((FundamentalGroupoidFunctor.prodToProdTop A B).comp ((FundamentalGroup …
    -/
    simpa
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      A B : TopCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((FundamentalGroupoidFunctor.projLeft …
    -/
    change (projLeft A B).prod' (projRight A B) ⋙ prodToProdTop A B = 𝟭 _
    /-
      A B : TopCat
      ⊢ Eq (((FundamentalGroupoidFunctor.projLeft A B).prod' (FundamentalGroupoidFun …
    -/
    apply CategoryTheory.Functor.hext
      /-
        case h_obj
        A B : TopCat
        ⊢ ∀ (X : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of (Prod …
      -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    · intros; apply FundamentalGroupoid.ext; apply Prod.ext <;> simp <;> rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    /-
      case h_map
      A B : TopCat
      ⊢ ∀ (X Y : ↑(FundamentalGroupoid.fundamentalGroupoidFunctor.obj (TopCat.of (Pr …
    -/
    rintro ⟨x₀, x₁⟩ ⟨y₀, y₁⟩ f
    /-
      case h_map.mk.mk.mk.mk
      A B : TopCat
      x₀ : ↑A
      x₁ : ↑B
      y₀ : ↑A
      y₁ : ↑B
      f : Quiver.Hom { as := { fst := x₀, snd := x₁ } } { as := { fst := y₀, snd :=  …
      ⊢ HEq ((((FundamentalGroupoidFunctor.projLeft A B).prod' (FundamentalGroupoidF …
    -/
    have := Path.Homotopic.prod_projLeft_projRight f
    -- Porting note: was simpa but TopSpace instances might be getting in the way
    simp only [CategoryTheory.Functor.comp_obj, CategoryTheory.Functor.prod'_obj, prodToProdTop_obj,
      CategoryTheory.Functor.comp_map, CategoryTheory.Functor.prod'_map, projLeft_map,
      projRight_map, CategoryTheory.Functor.id_obj, CategoryTheory.Functor.id_map, heq_eq_eq]
    /-
      case h_map.mk.mk.mk.mk
      A B : TopCat
      x₀ : ↑A
      x₁ : ↑B
      y₀ : ↑A
      y₁ : ↑B
      f : Quiver.Hom { as := { fst := x₀, snd := x₁ } } { as := { fst := y₀, snd :=  …
      this : Eq (Path.Homotopic.prod (Path.Homotopic.projLeft f) (Path.Homotopic.pro …
      ⊢ Eq ((FundamentalGroupoidFunctor.prodToProdTop A B).map { fst := Path.Homotop …
    -/
    apply this
    /-
      🎉 no goals
    -/


