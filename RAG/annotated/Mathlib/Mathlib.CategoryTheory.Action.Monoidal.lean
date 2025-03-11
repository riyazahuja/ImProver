@[simps! tensorUnit_V tensorObj_V tensorHom_hom whiskerLeft_hom whiskerRight_hom
  associator_hom_hom associator_inv_hom leftUnitor_hom_hom leftUnitor_inv_hom
  rightUnitor_hom_hom rightUnitor_inv_hom]
instance instMonoidalCategory : MonoidalCategory (Action V G) :=
  Monoidal.transport (Action.functorCategoryEquivalence _ _).symm

/- Adding this solves `simpNF` linter report at `tensorUnit_ρ` -/

@[simp]
theorem tensorUnit_ρ' {g : G} :
    @DFunLike.coe (G →* MonCat.of (End (𝟙_ V))) _ _ _ (𝟙_ (Action V G)).ρ g = 𝟙 (𝟙_ V) := by
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.MonoidalCategory V
    g : ↑G
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorUnit.ρ g) (CategoryTheory.Ca …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorUnit_ρ {g : G} : (𝟙_ (Action V G)).ρ g = 𝟙 (𝟙_ V) :=
  rfl

/- Adding this solves `simpNF` linter report at `tensor_ρ` -/

@[simp]
theorem tensor_ρ' {X Y : Action V G} {g : G} :
    @DFunLike.coe (G →* MonCat.of (End (X.V ⊗ Y.V))) _ _ _ (X ⊗ Y).ρ g = X.ρ g ⊗ Y.ρ g :=
  rfl


@[simp]
theorem tensor_ρ {X Y : Action V G} {g : G} : (X ⊗ Y).ρ g = X.ρ g ⊗ Y.ρ g :=
  rfl


/-- Given an object `X` isomorphic to the tensor unit of `V`, `X` equipped with the trivial action
is isomorphic to the tensor unit of `Action V G`. -/
def tensorUnitIso {X : V} (f : 𝟙_ V ≅ X) : 𝟙_ (Action V G) ≅ Action.mk X 1 :=
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.MonoidalCategory V
    X : V
    f : CategoryTheory.Iso CategoryTheory.MonoidalCategoryStruct.tensorUnit X
    ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalC …
  -/
  Action.mkIso f
  /-
    🎉 no goals
  -/


instance : (Action.forget V G).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _ }


@[simp] lemma forget_ε : ε (Action.forget V G) = 𝟙 _ := rfl

@[simp] lemma forget_η : ε (Action.forget V G) = 𝟙 _ := rfl


@[simp] lemma forget_μ (X Y : Action V G) : μ (Action.forget V G) X Y = 𝟙 _ := rfl

@[simp] lemma forget_δ (X Y : Action V G) : δ (Action.forget V G) X Y = 𝟙 _ := rfl


instance : BraidedCategory (Action V G) :=
  braidedCategoryOfFaithful (Action.forget V G) (fun X Y => mkIso (β_ _ _)
                 /-
                   V : Type (u + 1)
                   inst✝² : CategoryTheory.LargeCategory V
                   G : MonCat
                   inst✝¹ : CategoryTheory.MonoidalCategory V
                   inst✝ : CategoryTheory.BraidedCategory V
                   X Y : Action V G
                   g : ↑G
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
                 -/
                 /-
                   🎉 no goals
                 -/
    (fun g => by simp [FunctorCategoryEquivalence.inverse])) (by aesop_cat)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- When `V` is braided the forgetful functor `Action V G` to `V` is braided. -/
instance : (Action.forget V G).Braided where


instance [SymmetricCategory V] : SymmetricCategory (Action V G) :=
  symmetricCategoryOfFaithful (Action.forget V G)


instance : MonoidalPreadditive (Action V G) where


instance : MonoidalLinear R (Action V G) where


/-- Upgrading the functor `Action V G ⥤ (SingleObj G ⥤ V)` to a monoidal functor. -/
instance : (FunctorCategoryEquivalence.functor (V := V) (G := G)).Monoidal :=
  inferInstanceAs (Monoidal.equivalenceTransported
    (Action.functorCategoryEquivalence V G).symm).inverse.Monoidal


instance : (functorCategoryEquivalence V G).functor.Monoidal := by
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.MonoidalCategory V
    ⊢ (Action.functorCategoryEquivalence V G).functor.Monoidal
  -/
  dsimp only [functorCategoryEquivalence_functor]; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Upgrading the functor `(SingleObj G ⥤ V) ⥤ Action V G` to a monoidal functor. -/
instance : (FunctorCategoryEquivalence.inverse (V := V) (G := G)).Monoidal :=
  inferInstanceAs (Monoidal.equivalenceTransported
    (Action.functorCategoryEquivalence V G).symm).functor.Monoidal


instance : (functorCategoryEquivalence V G).inverse.Monoidal := by
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.MonoidalCategory V
    ⊢ (Action.functorCategoryEquivalence V G).inverse.Monoidal
  -/
  dsimp only [functorCategoryEquivalence_inverse]; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
lemma FunctorCategoryEquivalence.functor_ε :
    ε (FunctorCategoryEquivalence.functor (V := V) (G := G)) = 𝟙 _ := rfl


@[simp]
lemma FunctorCategoryEquivalence.functor_η :
    η (FunctorCategoryEquivalence.functor (V := V) (G := G)) = 𝟙 _ := rfl


@[simp]
lemma FunctorCategoryEquivalence.functor_μ (A B : Action V G) :
    μ FunctorCategoryEquivalence.functor A B = 𝟙 _ := rfl


@[simp]
lemma FunctorCategoryEquivalence.functor_δ (A B : Action V G) :
    δ FunctorCategoryEquivalence.functor A B = 𝟙 _ := rfl



instance [RightRigidCategory V] : RightRigidCategory (SingleObj (H : MonCat.{u}) ⥤ V) := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝¹ : CategoryTheory.MonoidalCategory V
    H : Grp
    inst✝ : CategoryTheory.RightRigidCategory V
    ⊢ CategoryTheory.RightRigidCategory (CategoryTheory.Functor (CategoryTheory.Si …
  -/
  change RightRigidCategory (SingleObj H ⥤ V); infer_instance
                                               /-
                                                 🎉 no goals
                                               -/


/-- If `V` is right rigid, so is `Action V G`. -/
instance [RightRigidCategory V] : RightRigidCategory (Action V H) :=
  rightRigidCategoryOfEquivalence
    (functorCategoryEquivalence V H).toAdjunction


instance [LeftRigidCategory V] : LeftRigidCategory (SingleObj (H : MonCat.{u}) ⥤ V) := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝¹ : CategoryTheory.MonoidalCategory V
    H : Grp
    inst✝ : CategoryTheory.LeftRigidCategory V
    ⊢ CategoryTheory.LeftRigidCategory (CategoryTheory.Functor (CategoryTheory.Sin …
  -/
  change LeftRigidCategory (SingleObj H ⥤ V); infer_instance
                                              /-
                                                🎉 no goals
                                              -/


/-- If `V` is left rigid, so is `Action V G`. -/
instance [LeftRigidCategory V] : LeftRigidCategory (Action V H) :=
  leftRigidCategoryOfEquivalence (functorCategoryEquivalence V H).toAdjunction


instance [RigidCategory V] : RigidCategory (SingleObj (H : MonCat.{u}) ⥤ V) := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝¹ : CategoryTheory.MonoidalCategory V
    H : Grp
    inst✝ : CategoryTheory.RigidCategory V
    ⊢ CategoryTheory.RigidCategory (CategoryTheory.Functor (CategoryTheory.SingleO …
  -/
  change RigidCategory (SingleObj H ⥤ V); infer_instance
                                          /-
                                            🎉 no goals
                                          -/


/-- If `V` is rigid, so is `Action V G`. -/
instance [RigidCategory V] : RigidCategory (Action V H) :=
  rigidCategoryOfEquivalence (functorCategoryEquivalence V H).toAdjunction


@[simp]
theorem rightDual_v [RightRigidCategory V] : Xᘁ.V = X.Vᘁ :=
  rfl


@[simp]
theorem leftDual_v [LeftRigidCategory V] : (ᘁX).V = ᘁX.V :=
  rfl

-- This lemma was always bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644

@[simp, nolint simpNF]
theorem rightDual_ρ [RightRigidCategory V] (h : H) : Xᘁ.ρ h = (X.ρ (h⁻¹ : H))ᘁ := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    inst✝¹ : CategoryTheory.MonoidalCategory V
    H : Grp
    X : Action V ((CategoryTheory.forget₂ Grp MonCat).obj H)
    inst✝ : CategoryTheory.RightRigidCategory V
    h : ↑H
    ⊢ Eq ((CategoryTheory.HasRightDual.rightDual X).ρ h) (CategoryTheory.rightAdjo …
  -/
  rw [← SingleObj.inv_as_inv]; rfl
                               /-
                                 🎉 no goals
                               -/

-- This lemma was always bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644

@[simp, nolint simpNF]
theorem leftDual_ρ [LeftRigidCategory V] (h : H) : (ᘁX).ρ h = ᘁX.ρ (h⁻¹ : H) := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    inst✝¹ : CategoryTheory.MonoidalCategory V
    H : Grp
    X : Action V ((CategoryTheory.forget₂ Grp MonCat).obj H)
    inst✝ : CategoryTheory.LeftRigidCategory V
    h : ↑H
    ⊢ Eq ((CategoryTheory.HasLeftDual.leftDual X).ρ h) (CategoryTheory.leftAdjoint …
  -/
  rw [← SingleObj.inv_as_inv]; rfl
                               /-
                                 🎉 no goals
                               -/


/-- Given `X : Action (Type u) (MonCat.of G)` for `G` a group, then `G × X` (with `G` acting as left
multiplication on the first factor and by `X.ρ` on the second) is isomorphic as a `G`-set to
`G × X` (with `G` acting as left multiplication on the first factor and trivially on the second).
The isomorphism is given by `(g, x) ↦ (g, g⁻¹ • x)`. -/
@[simps]
noncomputable def leftRegularTensorIso (G : Type u) [Group G] (X : Action (Type u) (MonCat.of G)) :
    leftRegular G ⊗ X ≅ leftRegular G ⊗ Action.mk X.V 1 where
  hom :=
    { hom := fun g => ⟨g.1, (X.ρ (g.1⁻¹ : G) g.2 : X.V)⟩
      comm := fun (g : G) => by
        /-
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g : G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        funext ⟨(x₁ : G), (x₂ : X.V)⟩
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        refine Prod.ext rfl ?_
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        change (X.ρ ((g * x₁)⁻¹ : G) * X.ρ g) x₂ = X.ρ _ _
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (HMul.hMul (X.ρ (Inv.inv (HMul.hMul g x₁))) (X.ρ g) x₂) (X.ρ (Inv.inv { f …
        -/
        rw [mul_inv_rev, ← X.ρ.map_mul, inv_mul_cancel_right] }
        /-
          🎉 no goals
        -/
  inv :=
    { hom := fun g => ⟨g.1, X.ρ g.1 g.2⟩
      comm := fun (g : G) => by
        /-
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g : G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        funext ⟨(x₁ : G), (x₂ : X.V)⟩
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        refine Prod.ext rfl ?_
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
        -/
        rw [tensor_ρ, tensor_ρ]
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        dsimp
        -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (X.ρ ((Action.leftRegular G).ρ g x₁) x₂) (X.ρ g (X.ρ x₁ x₂))
        -/
        erw [leftRegular_ρ_apply]
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (X.ρ (HMul.hMul g x₁) x₂) (X.ρ g (X.ρ x₁ x₂))
        -/
        rw [map_mul]
        /-
          case h
          V : Type (u + 1)
          inst✝¹ : CategoryTheory.LargeCategory V
          G✝ : MonCat
          G : Type u
          inst✝ : Group G
          X : Action (Type u) (MonCat.of G)
          g x₁ : G
          x₂ : X.V
          ⊢ Eq (HMul.hMul (X.ρ g) (X.ρ x₁) x₂) (X.ρ g (X.ρ x₁ x₂))
        -/
        rfl }
        /-
          🎉 no goals
        -/
  hom_inv_id := by
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd : …
    -/
    apply Hom.ext
    /-
      case hom
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd : …
    -/
    funext x
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) X).V
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd  …
    -/
    refine Prod.ext rfl ?_
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) X).V
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd  …
    -/
    change (X.ρ x.1 * X.ρ (x.1⁻¹ : G)) x.2 = x.2
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) X).V
      ⊢ Eq (HMul.hMul (X.ρ x.1) (X.ρ (Inv.inv x.1)) x.2) x.2
    -/
    rw [← X.ρ.map_mul, mul_inv_cancel, X.ρ.map_one, MonCat.one_of, End.one_def, types_id_apply]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd : …
    -/
    apply Hom.ext
    /-
      case hom
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd : …
    -/
    funext x
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) {  …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd  …
    -/
    refine Prod.ext rfl ?_
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) {  …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { hom := fun g => { fst := g.1, snd  …
    -/
    change (X.ρ (x.1⁻¹ : G) * X.ρ x.1) x.2 = x.2
    /-
      case hom.h
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G✝ : MonCat
      G : Type u
      inst✝ : Group G
      X : Action (Type u) (MonCat.of G)
      x : (CategoryTheory.MonoidalCategoryStruct.tensorObj (Action.leftRegular G) {  …
      ⊢ Eq (HMul.hMul (X.ρ (Inv.inv x.1)) (X.ρ x.1) x.2) x.2
    -/
    rw [← X.ρ.map_mul, inv_mul_cancel, X.ρ.map_one, MonCat.one_of, End.one_def, types_id_apply]
    /-
      🎉 no goals
    -/


/-- The natural isomorphism of `G`-sets `Gⁿ⁺¹ ≅ G × Gⁿ`, where `G` acts by left multiplication on
each factor. -/
@[simps!]
noncomputable def diagonalSucc (G : Type u) [Monoid G] (n : ℕ) :
    diagonal G (n + 1) ≅ leftRegular G ⊗ diagonal G n :=
  mkIso (Fin.consEquiv _).symm.toIso fun _ => rfl


set_option maxHeartbeats 400000 in
/-- A lax monoidal functor induces a lax monoidal functor between
the categories of `G`-actions within those categories. -/
instance [F.LaxMonoidal] : (F.mapAction G).LaxMonoidal where
  ε' :=
    { hom := ε F
      comm := fun g => by
        /-
          V : Type (u + 1)
          inst✝⁴ : CategoryTheory.LargeCategory V
          G : MonCat
          W : Type (u + 1)
          inst✝³ : CategoryTheory.LargeCategory W
          inst✝² : CategoryTheory.MonoidalCategory V
          inst✝¹ : CategoryTheory.MonoidalCategory W
          F : CategoryTheory.Functor V W
          inst✝ : F.LaxMonoidal
          g : ↑G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        dsimp [FunctorCategoryEquivalence.inverse, Functor.mapAction]
        /-
          V : Type (u + 1)
          inst✝⁴ : CategoryTheory.LargeCategory V
          G : MonCat
          W : Type (u + 1)
          inst✝³ : CategoryTheory.LargeCategory W
          inst✝² : CategoryTheory.MonoidalCategory V
          inst✝¹ : CategoryTheory.MonoidalCategory W
          F : CategoryTheory.Functor V W
          inst✝ : F.LaxMonoidal
          g : ↑G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Cat …
        -/
        rw [Category.id_comp, F.map_id, Category.comp_id] }
        /-
          🎉 no goals
        -/
  μ' X Y :=
    { hom := μ F X.V Y.V
      comm := fun g => μ_natural F (X.ρ g) (Y.ρ g) }
                            /-
                              V : Type (u + 1)
                              inst✝⁴ : CategoryTheory.LargeCategory V
                              G : MonCat
                              W : Type (u + 1)
                              inst✝³ : CategoryTheory.LargeCategory W
                              inst✝² : CategoryTheory.MonoidalCategory V
                              inst✝¹ : CategoryTheory.MonoidalCategory W
                              F : CategoryTheory.Functor V W
                              inst✝ : F.LaxMonoidal
                              X✝ Y✝ : Action V G
                              x✝¹ : Quiver.Hom X✝ Y✝
                              x✝ : Action V G
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                            -/
  μ'_natural_left _ _ := by ext; simp
                                 /-
                                   🎉 no goals
                                 -/
                             /-
                               V : Type (u + 1)
                               inst✝⁴ : CategoryTheory.LargeCategory V
                               G : MonCat
                               W : Type (u + 1)
                               inst✝³ : CategoryTheory.LargeCategory W
                               inst✝² : CategoryTheory.MonoidalCategory V
                               inst✝¹ : CategoryTheory.MonoidalCategory W
                               F : CategoryTheory.Functor V W
                               inst✝ : F.LaxMonoidal
                               X✝ Y✝ x✝¹ : Action V G
                               x✝ : Quiver.Hom X✝ Y✝
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                             -/
  μ'_natural_right _ _ := by ext; simp
                                  /-
                                    🎉 no goals
                                  -/
                             /-
                               V : Type (u + 1)
                               inst✝⁴ : CategoryTheory.LargeCategory V
                               G : MonCat
                               W : Type (u + 1)
                               inst✝³ : CategoryTheory.LargeCategory W
                               inst✝² : CategoryTheory.MonoidalCategory V
                               inst✝¹ : CategoryTheory.MonoidalCategory W
                               F : CategoryTheory.Functor V W
                               inst✝ : F.LaxMonoidal
                               x✝² x✝¹ x✝ : Action V G
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                             -/
  associativity' _ _ _ := by ext; simp
                                  /-
                                    🎉 no goals
                                  -/
                          /-
                            V : Type (u + 1)
                            inst✝⁴ : CategoryTheory.LargeCategory V
                            G : MonCat
                            W : Type (u + 1)
                            inst✝³ : CategoryTheory.LargeCategory W
                            inst✝² : CategoryTheory.MonoidalCategory V
                            inst✝¹ : CategoryTheory.MonoidalCategory W
                            F : CategoryTheory.Functor V W
                            inst✝ : F.LaxMonoidal
                            x✝ : Action V G
                            ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((F.mapAction G).obj x✝ …
                          -/
  left_unitality' _ := by ext; simp
                               /-
                                 🎉 no goals
                               -/
                           /-
                             V : Type (u + 1)
                             inst✝⁴ : CategoryTheory.LargeCategory V
                             G : MonCat
                             W : Type (u + 1)
                             inst✝³ : CategoryTheory.LargeCategory W
                             inst✝² : CategoryTheory.MonoidalCategory V
                             inst✝¹ : CategoryTheory.MonoidalCategory W
                             F : CategoryTheory.Functor V W
                             inst✝ : F.LaxMonoidal
                             x✝ : Action V G
                             ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((F.mapAction G).obj x …
                           -/
  right_unitality' _ := by ext; simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
lemma mapAction_ε_hom [F.LaxMonoidal] : (ε (F.mapAction G)).hom = ε F := rfl


@[simp]
lemma mapAction_μ_hom [F.LaxMonoidal] (X Y : Action V G) :
    (μ (F.mapAction G) X Y).hom = μ F X.V Y.V := rfl


/-- An oplax monoidal functor induces an oplax monoidal functor between
the categories of `G`-actions within those categories. -/
instance [F.OplaxMonoidal] : (F.mapAction G).OplaxMonoidal where
  η' :=
    { hom := η F
      comm := fun g => by
        /-
          V : Type (u + 1)
          inst✝⁴ : CategoryTheory.LargeCategory V
          G : MonCat
          W : Type (u + 1)
          inst✝³ : CategoryTheory.LargeCategory W
          inst✝² : CategoryTheory.MonoidalCategory V
          inst✝¹ : CategoryTheory.MonoidalCategory W
          F : CategoryTheory.Functor V W
          inst✝ : F.OplaxMonoidal
          g : ↑G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.mapAction G).obj CategoryTheory. …
        -/
        dsimp [FunctorCategoryEquivalence.inverse, Functor.mapAction]
        /-
          V : Type (u + 1)
          inst✝⁴ : CategoryTheory.LargeCategory V
          G : MonCat
          W : Type (u + 1)
          inst✝³ : CategoryTheory.LargeCategory W
          inst✝² : CategoryTheory.MonoidalCategory V
          inst✝¹ : CategoryTheory.MonoidalCategory W
          F : CategoryTheory.Functor V W
          inst✝ : F.OplaxMonoidal
          g : ↑G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
        -/
        rw [map_id, Category.id_comp, Category.comp_id] }
        /-
          🎉 no goals
        -/
  δ' X Y :=
    { hom := δ F X.V Y.V
      comm := fun g => (δ_natural F (X.ρ g) (Y.ρ g)).symm }
                            /-
                              V : Type (u + 1)
                              inst✝⁴ : CategoryTheory.LargeCategory V
                              G : MonCat
                              W : Type (u + 1)
                              inst✝³ : CategoryTheory.LargeCategory W
                              inst✝² : CategoryTheory.MonoidalCategory V
                              inst✝¹ : CategoryTheory.MonoidalCategory W
                              F : CategoryTheory.Functor V W
                              inst✝ : F.OplaxMonoidal
                              X✝ Y✝ : Action V G
                              x✝¹ : Quiver.Hom X✝ Y✝
                              x✝ : Action V G
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => { hom := CategoryTheory. …
                            -/
  δ'_natural_left _ _ := by ext; simp
                                 /-
                                   🎉 no goals
                                 -/
                             /-
                               V : Type (u + 1)
                               inst✝⁴ : CategoryTheory.LargeCategory V
                               G : MonCat
                               W : Type (u + 1)
                               inst✝³ : CategoryTheory.LargeCategory W
                               inst✝² : CategoryTheory.MonoidalCategory V
                               inst✝¹ : CategoryTheory.MonoidalCategory W
                               F : CategoryTheory.Functor V W
                               inst✝ : F.OplaxMonoidal
                               X✝ Y✝ x✝¹ : Action V G
                               x✝ : Quiver.Hom X✝ Y✝
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => { hom := CategoryTheory. …
                             -/
  δ'_natural_right _ _ := by ext; simp
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     V : Type (u + 1)
                                     inst✝⁴ : CategoryTheory.LargeCategory V
                                     G : MonCat
                                     W : Type (u + 1)
                                     inst✝³ : CategoryTheory.LargeCategory W
                                     inst✝² : CategoryTheory.MonoidalCategory V
                                     inst✝¹ : CategoryTheory.MonoidalCategory W
                                     F : CategoryTheory.Functor V W
                                     inst✝ : F.OplaxMonoidal
                                     x✝² x✝¹ x✝ : Action V G
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => { hom := CategoryTheory. …
                                   -/
  oplax_associativity' _ _ _ := by ext; simp
                                        /-
                                          🎉 no goals
                                        -/
                                /-
                                  V : Type (u + 1)
                                  inst✝⁴ : CategoryTheory.LargeCategory V
                                  G : MonCat
                                  W : Type (u + 1)
                                  inst✝³ : CategoryTheory.LargeCategory W
                                  inst✝² : CategoryTheory.MonoidalCategory V
                                  inst✝¹ : CategoryTheory.MonoidalCategory W
                                  F : CategoryTheory.Functor V W
                                  inst✝ : F.OplaxMonoidal
                                  x✝ : Action V G
                                  ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((F.mapAction G).obj x✝ …
                                -/
  oplax_left_unitality' _ := by ext; simp
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   V : Type (u + 1)
                                   inst✝⁴ : CategoryTheory.LargeCategory V
                                   G : MonCat
                                   W : Type (u + 1)
                                   inst✝³ : CategoryTheory.LargeCategory W
                                   inst✝² : CategoryTheory.MonoidalCategory V
                                   inst✝¹ : CategoryTheory.MonoidalCategory W
                                   F : CategoryTheory.Functor V W
                                   inst✝ : F.OplaxMonoidal
                                   x✝ : Action V G
                                   ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((F.mapAction G).obj x …
                                 -/
  oplax_right_unitality' _ := by ext; simp
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
lemma mapAction_η_hom [F.OplaxMonoidal] : (η (F.mapAction G)).hom = η F := rfl


@[simp]
lemma mapAction_δ_hom [F.OplaxMonoidal] (X Y : Action V G) :
    (δ (F.mapAction G) X Y).hom = δ F X.V Y.V := rfl


/-- A monoidal functor induces a monoidal functor between
the categories of `G`-actions within those categories. -/
instance [F.Monoidal] : (F.mapAction G).Monoidal where
            /-
              V : Type (u + 1)
              inst✝⁴ : CategoryTheory.LargeCategory V
              G : MonCat
              W : Type (u + 1)
              inst✝³ : CategoryTheory.LargeCategory W
              inst✝² : CategoryTheory.MonoidalCategory V
              inst✝¹ : CategoryTheory.MonoidalCategory W
              F : CategoryTheory.Functor V W
              inst✝ : F.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
            -/
            /-
              V : Type (u + 1)
              inst✝⁴ : CategoryTheory.LargeCategory V
              G : MonCat
              W : Type (u + 1)
              inst✝³ : CategoryTheory.LargeCategory W
              inst✝² : CategoryTheory.MonoidalCategory V
              inst✝¹ : CategoryTheory.MonoidalCategory W
              F : CategoryTheory.Functor V W
              inst✝ : F.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
            -/
  η_ε := by ext; dsimp; rw [η_ε]
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
  ε_η := by ext; dsimp; rw [ε_η]
                /-
                  V : Type (u + 1)
                  inst✝⁴ : CategoryTheory.LargeCategory V
                  G : MonCat
                  W : Type (u + 1)
                  inst✝³ : CategoryTheory.LargeCategory W
                  inst✝² : CategoryTheory.MonoidalCategory V
                  inst✝¹ : CategoryTheory.MonoidalCategory W
                  F : CategoryTheory.Functor V W
                  inst✝ : F.Monoidal
                  x✝¹ x✝ : Action V G
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
                -/
  μ_δ _ _ := by ext; dsimp; rw [μ_δ]
                            /-
                              🎉 no goals
                            -/
                /-
                  V : Type (u + 1)
                  inst✝⁴ : CategoryTheory.LargeCategory V
                  G : MonCat
                  W : Type (u + 1)
                  inst✝³ : CategoryTheory.LargeCategory W
                  inst✝² : CategoryTheory.MonoidalCategory V
                  inst✝¹ : CategoryTheory.MonoidalCategory W
                  F : CategoryTheory.Functor V W
                  inst✝ : F.Monoidal
                  x✝¹ x✝ : Action V G
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
                -/
  δ_μ _ _ := by ext; dsimp; rw [δ_μ]
                            /-
                              🎉 no goals
                            -/


