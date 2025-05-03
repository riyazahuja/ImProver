/-- Define `FGModuleCat` as the subtype of `ModuleCat.{u} R` of finitely generated modules. -/
def FGModuleCat :=
  FullSubcategory fun V : ModuleCat.{u} R => Module.Finite R V
-- Porting note: still no derive handler via `dsimp`.
-- see https://github.com/leanprover-community/mathlib4/issues/5020
-- deriving LargeCategory, ConcreteCategory,Preadditive


/-- A synonym for `M.obj.carrier`, which we can mark with `@[coe]`. -/
def FGModuleCat.carrier (M : FGModuleCat R) : Type u := M.obj.carrier


instance : CoeSort (FGModuleCat R) (Type u) :=
  ⟨FGModuleCat.carrier⟩


@[simp] lemma obj_carrier (M : FGModuleCat R) : M.obj.carrier = M.carrier := rfl


instance (M : FGModuleCat R) : AddCommGroup M := by
  /-
    R : Type u
    inst✝ : Ring R
    M : FGModuleCat R
    ⊢ AddCommGroup ↑M
  -/
  change AddCommGroup M.obj
  /-
    R : Type u
    inst✝ : Ring R
    M : FGModuleCat R
    ⊢ AddCommGroup ↑M.obj
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (M : FGModuleCat R) : Module R M := by
  /-
    R : Type u
    inst✝ : Ring R
    M : FGModuleCat R
    ⊢ Module R ↑M
  -/
  change Module R M.obj
  /-
    R : Type u
    inst✝ : Ring R
    M : FGModuleCat R
    ⊢ Module R ↑M.obj
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : LargeCategory (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.LargeCategory (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.LargeCategory (CategoryTheory.FullSubcategory fun V => Module …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : ConcreteCategory (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.ConcreteCategory (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.ConcreteCategory (CategoryTheory.FullSubcategory fun V => Mod …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Preadditive (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.Preadditive (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.Preadditive (CategoryTheory.FullSubcategory fun V => Module.F …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance finite (V : FGModuleCat R) : Module.Finite R V :=
  V.property


instance : Inhabited (FGModuleCat R) :=
  ⟨⟨ModuleCat.of R R, Module.Finite.self R⟩⟩


/-- Lift an unbundled finitely generated module to `FGModuleCat R`. -/
abbrev of (V : Type u) [AddCommGroup V] [Module R V] [Module.Finite R V] : FGModuleCat R :=
                        /-
                          R : Type u
                          inst✝³ : Ring R
                          V : Type u
                          inst✝² : AddCommGroup V
                          inst✝¹ : Module R V
                          inst✝ : Module.Finite R V
                          ⊢ Module.Finite R ↑(ModuleCat.of R V)
                        -/
  ⟨ModuleCat.of R V, by change Module.Finite R V; infer_instance⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


variable {R} in
/-- Lift a linear map between finitely generated modules to `FGModuleCat R`. -/
abbrev ofHom {V W : Type u} [AddCommGroup V] [Module R V] [Module.Finite R V]
    [AddCommGroup W] [Module R W] [Module.Finite R W]
    (f : V →ₗ[R] W) : of R V ⟶ of R W :=
  ModuleCat.ofHom f


variable {R} in
@[ext] lemma hom_ext {V W : FGModuleCat R} {f g : V ⟶ W} (h : f.hom = g.hom) : f = g :=
  ModuleCat.hom_ext h


instance (V : FGModuleCat R) : Module.Finite R V :=
  V.property


instance : HasForget₂ (FGModuleCat.{u} R) (ModuleCat.{u} R) := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.HasForget₂ (FGModuleCat R) (ModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ CategoryTheory.HasForget₂ (CategoryTheory.FullSubcategory fun V => Module.Fi …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : (forget₂ (FGModuleCat R) (ModuleCat.{u} R)).Full where
  map_surjective f := ⟨f, rfl⟩


/-- Converts and isomorphism in the category `FGModuleCat R` to
a `LinearEquiv` between the underlying modules. -/
def isoToLinearEquiv {V W : FGModuleCat R} (i : V ≅ W) : V ≃ₗ[R] W :=
  ((forget₂ (FGModuleCat.{u} R) (ModuleCat.{u} R)).mapIso i).toLinearEquiv


/-- Converts a `LinearEquiv` to an isomorphism in the category `FGModuleCat R`. -/
@[simps]
def _root_.LinearEquiv.toFGModuleCatIso
    {V W : Type u} [AddCommGroup V] [Module R V] [Module.Finite R V]
    [AddCommGroup W] [Module R W] [Module.Finite R W] (e : V ≃ₗ[R] W) :
    FGModuleCat.of R V ≅ FGModuleCat.of R W where
  hom := ModuleCat.ofHom e.toLinearMap
  inv := ModuleCat.ofHom e.symm.toLinearMap
                   /-
                     R : Type u
                     inst✝⁶ : Ring R
                     V W : Type u
                     inst✝⁵ : AddCommGroup V
                     inst✝⁴ : Module R V
                     inst✝³ : Module.Finite R V
                     inst✝² : AddCommGroup W
                     inst✝¹ : Module R W
                     inst✝ : Module.Finite R W
                     e : LinearEquiv (RingHom.id R) V W
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom ↑e) (ModuleCat.ofHom …
                   -/
  hom_inv_id := by ext x; exact e.left_inv x
                          /-
                            🎉 no goals
                          -/
                   /-
                     R : Type u
                     inst✝⁶ : Ring R
                     V W : Type u
                     inst✝⁵ : AddCommGroup V
                     inst✝⁴ : Module R V
                     inst✝³ : Module.Finite R V
                     inst✝² : AddCommGroup W
                     inst✝¹ : Module R W
                     inst✝ : Module.Finite R W
                     e : LinearEquiv (RingHom.id R) V W
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom ↑e.symm) (ModuleCat. …
                   -/
  inv_hom_id := by ext x; exact e.right_inv x
                          /-
                            🎉 no goals
                          -/


instance : Linear R (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.Linear R (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.Linear R (CategoryTheory.FullSubcategory fun V => Module.Fini …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance monoidalPredicate_module_finite :
    MonoidalCategory.MonoidalPredicate fun V : ModuleCat.{u} R => Module.Finite R V where
  prop_id := Module.Finite.self R
  prop_tensor := @fun X Y _ _ => Module.Finite.tensorProduct R X Y


instance instMonoidalCategory : MonoidalCategory (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalCategory (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalCategory (CategoryTheory.FullSubcategory fun V => Mod …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp] lemma tensorUnit_obj : (𝟙_ (FGModuleCat R)).obj = 𝟙_ (ModuleCat R) := rfl

@[simp] lemma tensorObj_obj (M N : FGModuleCat.{u} R) : (M ⊗ N).obj = (M.obj ⊗ N.obj) := rfl


instance : SymmetricCategory (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.SymmetricCategory (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.SymmetricCategory (CategoryTheory.FullSubcategory fun V => Mo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : MonoidalPreadditive (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalPreadditive (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalPreadditive (CategoryTheory.FullSubcategory fun V =>  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : MonoidalLinear R (FGModuleCat R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalLinear R (FGModuleCat R)
  -/
  dsimp [FGModuleCat]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalLinear R (CategoryTheory.FullSubcategory fun V => Mod …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The forgetful functor `FGModuleCat R ⥤ Module R` is a monoidal functor. -/
instance : (forget₂ (FGModuleCat.{u} R) (ModuleCat.{u} R)).Monoidal :=
  fullSubcategoryInclusionMonoidal _


instance : (forget₂ (FGModuleCat.{u} R) (ModuleCat.{u} R)).Additive where

instance : (forget₂ (FGModuleCat.{u} R) (ModuleCat.{u} R)).Linear R where


theorem Iso.conj_eq_conj {V W : FGModuleCat R} (i : V ≅ W) (f : End V) :
    Iso.conj i f = FGModuleCat.ofHom (LinearEquiv.conj (isoToLinearEquiv i) f.hom) :=
  rfl


theorem Iso.conj_hom_eq_conj {V W : FGModuleCat R} (i : V ≅ W) (f : End V) :
    (Iso.conj i f).hom = (LinearEquiv.conj (isoToLinearEquiv i) f.hom) :=
  rfl


instance (V W : FGModuleCat K) : Module.Finite K (V ⟶ W) :=
  (inferInstanceAs <| Module.Finite K (V →ₗ[K] W)).equiv ModuleCat.homLinearEquiv.symm


instance closedPredicateModuleFinite :
    MonoidalCategory.ClosedPredicate fun V : ModuleCat.{u} K ↦ Module.Finite K V where
  prop_ihom {X Y} _ _ :=
    (inferInstanceAs <| Module.Finite K (X →ₗ[K] Y)).equiv ModuleCat.homLinearEquiv.symm


instance : MonoidalClosed (FGModuleCat K) := by
  /-
    K : Type u
    inst✝ : Field K
    ⊢ CategoryTheory.MonoidalClosed (FGModuleCat K)
  -/
  dsimp [FGModuleCat]
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11187): was `infer_instance`
  exact MonoidalCategory.fullMonoidalClosedSubcategory
    (fun V : ModuleCat.{u} K => Module.Finite K V)


@[simp]
theorem ihom_obj : (ihom V).obj W = FGModuleCat.of K (V ⟶ W) :=
  rfl


/-- The dual module is the dual in the rigid monoidal category `FGModuleCat K`. -/
def FGModuleCatDual : FGModuleCat K :=
  ⟨ModuleCat.of K (Module.Dual K V), Subspace.instModuleDualFiniteDimensional⟩


@[simp] lemma FGModuleCatDual_obj : (FGModuleCatDual K V).obj = ModuleCat.of K (Module.Dual K V) :=
  rfl

@[simp] lemma FGModuleCatDual_coe : (FGModuleCatDual K V : Type u) = Module.Dual K V := rfl


/-- The coevaluation map is defined in `LinearAlgebra.coevaluation`. -/
def FGModuleCatCoevaluation : 𝟙_ (FGModuleCat K) ⟶ V ⊗ FGModuleCatDual K V :=
  ModuleCat.ofHom <| coevaluation K V


theorem FGModuleCatCoevaluation_apply_one :
    (FGModuleCatCoevaluation K V).hom (1 : K) =
      ∑ i : Basis.ofVectorSpaceIndex K V,
        (Basis.ofVectorSpace K V) i ⊗ₜ[K] (Basis.ofVectorSpace K V).coord i :=
  coevaluation_apply_one K V


/-- The evaluation morphism is given by the contraction map. -/
def FGModuleCatEvaluation : FGModuleCatDual K V ⊗ V ⟶ 𝟙_ (FGModuleCat K) :=
  ModuleCat.ofHom <| contractLeft K V


theorem FGModuleCatEvaluation_apply (f : FGModuleCatDual K V) (x : V) :
    (FGModuleCatEvaluation K V).hom (f ⊗ₜ x) = f.toFun x :=
  contractLeft_apply f x


/-- `@[simp]`-normal form of `FGModuleCatEvaluation_apply`, where the carriers have been unfolded.
-/
@[simp]
theorem FGModuleCatEvaluation_apply' (f : FGModuleCatDual K V) (x : V) :
    DFunLike.coe
      (F := ((ModuleCat.of K (Module.Dual K V) ⊗ V.obj).carrier →ₗ[K] (𝟙_ (ModuleCat K))))
      (FGModuleCatEvaluation K V).hom (f ⊗ₜ x) = f.toFun x :=
  contractLeft_apply f x


private theorem coevaluation_evaluation :
    letI V' : FGModuleCat K := FGModuleCatDual K V
    V' ◁ FGModuleCatCoevaluation K V ≫ (α_ V' V V').inv ≫ FGModuleCatEvaluation K V ▷ V' =
      (ρ_ V').hom ≫ (λ_ V').inv := by
  /-
    K : Type u
    inst✝ : Field K
    V : FGModuleCat K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext : 1
  /-
    case h
    K : Type u
    inst✝ : Field K
    V : FGModuleCat K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply contractLeft_assoc_coevaluation K V
  /-
    🎉 no goals
  -/


private theorem evaluation_coevaluation :
    FGModuleCatCoevaluation K V ▷ V ≫
        (α_ V (FGModuleCatDual K V) V).hom ≫ V ◁ FGModuleCatEvaluation K V =
      (λ_ V).hom ≫ (ρ_ V).inv := by
  /-
    K : Type u
    inst✝ : Field K
    V : FGModuleCat K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext : 1
  /-
    case h
    K : Type u
    inst✝ : Field K
    V : FGModuleCat K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply contractLeft_assoc_coevaluation' K V
  /-
    🎉 no goals
  -/


instance exactPairing : ExactPairing V (FGModuleCatDual K V) where
  coevaluation' := FGModuleCatCoevaluation K V
  evaluation' := FGModuleCatEvaluation K V
  coevaluation_evaluation' := coevaluation_evaluation K V
  evaluation_coevaluation' := evaluation_coevaluation K V


instance rightDual : HasRightDual V :=
  ⟨FGModuleCatDual K V⟩


instance rightRigidCategory : RightRigidCategory (FGModuleCat K) where


@[simp] theorem LinearMap.comp_id_fgModuleCat
    {R} [Ring R] {G : FGModuleCat.{u} R} {H : Type u} [AddCommGroup H] [Module R H]
    (f : G →ₗ[R] H) : f.comp (ModuleCat.Hom.hom (𝟙 G)) = f :=
  ModuleCat.hom_ext_iff.mp <| Category.id_comp (ModuleCat.ofHom f)


@[simp] theorem LinearMap.id_fgModuleCat_comp
    {R} [Ring R] {G : Type u} [AddCommGroup G] [Module R G] {H : FGModuleCat.{u} R}
    (f : G →ₗ[R] H) : LinearMap.comp (ModuleCat.Hom.hom (𝟙 H)) f = f :=
  ModuleCat.hom_ext_iff.mp <| Category.comp_id (ModuleCat.ofHom f)

