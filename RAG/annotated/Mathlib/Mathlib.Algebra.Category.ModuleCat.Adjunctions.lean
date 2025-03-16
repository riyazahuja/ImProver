/-- The free functor `Type u ⥤ ModuleCat R` sending a type `X` to the
free `R`-module with generators `x : X`, implemented as the type `X →₀ R`.
-/
def free : Type u ⥤ ModuleCat R where
  obj X := ModuleCat.of R (X →₀ R)
  map {_ _} f := ofHom <| Finsupp.lmapDomain _ _ f
               /-
                 R : Type u
                 inst✝ : Ring R
                 ⊢ ∀ (X : Type u), Eq ({ obj := fun X => ModuleCat.of R (Finsupp X R), map := f …
               -/
  map_id := by intros; ext : 1; exact Finsupp.lmapDomain_id _ _
                                /-
                                  🎉 no goals
                                -/
                 /-
                   R : Type u
                   inst✝ : Ring R
                   ⊢ ∀ {X Y Z : Type u} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := f …
                 -/
  map_comp := by intros; ext : 1; exact Finsupp.lmapDomain_comp _ _ _ _
                                  /-
                                    🎉 no goals
                                  -/


/-- Constructor for elements in the module `(free R).obj X`. -/
noncomputable def freeMk {X : Type u} (x : X) : (free R).obj X := Finsupp.single x 1


@[ext 1200]
lemma free_hom_ext {X : Type u} {M : ModuleCat.{u} R} {f g : (free R).obj X ⟶ M}
    (h : ∀ (x : X), f (freeMk x) = g (freeMk x)) :
    f = g :=
  ModuleCat.hom_ext (Finsupp.lhom_ext' (fun x ↦ LinearMap.ext_ring (h x)))


/-- The morphism of modules `(free R).obj X ⟶ M` corresponding
to a map `f : X ⟶ M`. -/
noncomputable def freeDesc {X : Type u} {M : ModuleCat.{u} R} (f : X ⟶ M) :
    (free R).obj X ⟶ M :=
  ofHom <| Finsupp.lift M R X f


@[simp]
lemma freeDesc_apply {X : Type u} {M : ModuleCat.{u} R} (f : X ⟶ M) (x : X) :
    freeDesc f (freeMk x) = f x := by
  /-
    R : Type u
    inst✝ : Ring R
    X : Type u
    M : ModuleCat R
    f : Quiver.Hom X ↑M
    x : X
    ⊢ Eq ((ModuleCat.freeDesc f).hom (ModuleCat.freeMk x)) (f x)
  -/
  dsimp [freeDesc]
  /-
    R : Type u
    inst✝ : Ring R
    X : Type u
    M : ModuleCat R
    f : Quiver.Hom X ↑M
    x : X
    ⊢ Eq (((Finsupp.lift (↑M) R X) f) (ModuleCat.freeMk x)) (f x)
  -/
  erw [Finsupp.lift_apply, Finsupp.sum_single_index]
  /-
    R : Type u
    inst✝ : Ring R
    X : Type u
    M : ModuleCat R
    f : Quiver.Hom X ↑M
    x : X
    ⊢ Eq (HSMul.hSMul 1 (f x)) (f x)
  -/
  all_goals simp
  /-
    🎉 no goals
  -/


@[simp]
lemma free_map_apply {X Y : Type u} (f : X → Y) (x : X) :
    (free R).map f (freeMk x) = freeMk (f x) := by
  /-
    R : Type u
    inst✝ : Ring R
    X Y : Type u
    f : X → Y
    x : X
    ⊢ Eq (((ModuleCat.free R).map f).hom (ModuleCat.freeMk x)) (ModuleCat.freeMk ( …
  -/
  apply Finsupp.mapDomain_single
  /-
    🎉 no goals
  -/


/-- The bijection `((free R).obj X ⟶ M) ≃ (X → M)` when `X` is a type and `M` a module. -/
@[simps]
def freeHomEquiv {X : Type u} {M : ModuleCat.{u} R} :
    ((free R).obj X ⟶ M) ≃ (X → M) where
  toFun φ x := φ (freeMk x)
  invFun ψ := freeDesc ψ
                   /-
                     R : Type u
                     inst✝ : Ring R
                     X : Type u
                     M : ModuleCat R
                     x✝ : Quiver.Hom ((ModuleCat.free R).obj X) M
                     ⊢ Eq ((fun ψ => ModuleCat.freeDesc ψ) ((fun φ x => φ.hom (ModuleCat.freeMk x)) …
                   -/
  left_inv _ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u
                      inst✝ : Ring R
                      X : Type u
                      M : ModuleCat R
                      x✝ : X → ↑M
                      ⊢ Eq ((fun φ x => φ.hom (ModuleCat.freeMk x)) ((fun ψ => ModuleCat.freeDesc ψ) …
                    -/
  right_inv _ := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- The free-forgetful adjunction for R-modules.
-/
def adj : free R ⊣ forget (ModuleCat.{u} R) :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => freeHomEquiv
                                                            /-
                                                              R : Type u
                                                              inst✝ : Ring R
                                                              X Y : Type u
                                                              M : ModuleCat R
                                                              f : Quiver.Hom X Y
                                                              g : Quiver.Hom Y ((CategoryTheory.forget (ModuleCat R)).obj M)
                                                              ⊢ Eq (((fun x x_1 => ModuleCat.freeHomEquiv) X M).symm (CategoryTheory.Categor …
                                                            -/
      homEquiv_naturality_left_symm := fun {X Y M} f g ↦ by ext; simp [freeHomEquiv] }
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
lemma adj_homEquiv (X : Type u) (M : ModuleCat.{u} R) :
    (adj R).homEquiv X M = freeHomEquiv := by
  /-
    R : Type u
    inst✝ : Ring R
    X : Type u
    M : ModuleCat R
    ⊢ Eq ((ModuleCat.adj R).homEquiv X M) ModuleCat.freeHomEquiv
  -/
  simp only [adj, Adjunction.mkOfHomEquiv_homEquiv]
  /-
    🎉 no goals
  -/


instance : (forget (ModuleCat.{u} R)).IsRightAdjoint  :=
  (adj R).isRightAdjoint


/-- The canonical isomorphism `𝟙_ (ModuleCat R) ≅ (free R).obj (𝟙_ (Type u))`.
(This should not be used directly: it is part of the implementation of the
monoidal structure on the functor `free R`.) -/
def εIso : 𝟙_ (ModuleCat R) ≅ (free R).obj (𝟙_ (Type u)) where
  hom := ofHom <| Finsupp.lsingle PUnit.unit
  inv := ofHom <| Finsupp.lapply PUnit.unit
  hom_inv_id := by
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (Finsupp.lsingle PUn …
    -/
    ext
    /-
      case hf.h
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (Finsupp.lsingle PU …
    -/
    simp [free]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (Finsupp.lapply PUni …
    -/
    ext ⟨⟩
    /-
      case h.unit
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (Finsupp.lapply PUn …
    -/
    dsimp [freeMk]
    /-
      case h.unit
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq ((Finsupp.lsingle PUnit.unit) ((Finsupp.lapply PUnit.unit) (Finsupp.singl …
    -/
    erw [Finsupp.lapply_apply, Finsupp.lsingle_apply]
    /-
      case h.unit
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq (Finsupp.single PUnit.unit ((Finsupp.single PUnit.unit 1) PUnit.unit)) (F …
    -/
    rw [Finsupp.single_eq_same]
    /-
      🎉 no goals
    -/


@[simp]
lemma εIso_hom_one : (εIso R).hom 1 = freeMk PUnit.unit := rfl


@[simp]
lemma εIso_inv_freeMk (x : PUnit) : (εIso R).inv (freeMk x) = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : PUnit.{u + 1}
    ⊢ Eq ((ModuleCat.FreeMonoidal.εIso R).inv.hom (ModuleCat.freeMk x)) 1
  -/
  dsimp [εIso, freeMk]
  /-
    R : Type u
    inst✝ : CommRing R
    x : PUnit.{u + 1}
    ⊢ Eq ((Finsupp.lapply PUnit.unit) (Finsupp.single x 1)) 1
  -/
  erw [Finsupp.lapply_apply]
  /-
    R : Type u
    inst✝ : CommRing R
    x : PUnit.{u + 1}
    ⊢ Eq ((Finsupp.single x 1) PUnit.unit) 1
  -/
  rw [Finsupp.single_eq_same]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `(free R).obj X ⊗ (free R).obj Y ≅ (free R).obj (X ⊗ Y)`
for two types `X` and `Y`.
(This should not be used directly: it is is part of the implementation of the
monoidal structure on the functor `free R`.) -/
def μIso (X Y : Type u) :
    (free R).obj X ⊗ (free R).obj Y ≅ (free R).obj (X ⊗ Y) :=
  (finsuppTensorFinsupp' R _ _).toModuleIso


@[simp]
lemma μIso_hom_freeMk_tmul_freeMk {X Y : Type u} (x : X) (y : Y) :
    (μIso R X Y).hom (freeMk x ⊗ₜ freeMk y) = freeMk ⟨x, y⟩ := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    x : X
    y : Y
    ⊢ Eq ((ModuleCat.FreeMonoidal.μIso R X Y).hom.hom (TensorProduct.tmul R (Modul …
  -/
  dsimp [μIso, freeMk]
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    x : X
    y : Y
    ⊢ Eq (↑(finsuppTensorFinsupp' R X Y) (TensorProduct.tmul R (Finsupp.single x 1 …
  -/
  erw [finsuppTensorFinsupp'_single_tmul_single]
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    x : X
    y : Y
    ⊢ Eq (Finsupp.single { fst := x, snd := y } (HMul.hMul 1 1)) (Finsupp.single { …
  -/
  rw [mul_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma μIso_inv_freeMk {X Y : Type u} (z : X ⊗ Y) :
    (μIso R X Y).inv (freeMk z) = freeMk z.1 ⊗ₜ freeMk z.2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    z : CategoryTheory.MonoidalCategoryStruct.tensorObj X Y
    ⊢ Eq ((ModuleCat.FreeMonoidal.μIso R X Y).inv.hom (ModuleCat.freeMk z)) (Tenso …
  -/
  dsimp [μIso, freeMk]
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    z : CategoryTheory.MonoidalCategoryStruct.tensorObj X Y
    ⊢ Eq (↑(finsuppTensorFinsupp' R X Y).symm (Finsupp.single z 1)) (TensorProduct …
  -/
  erw [finsuppTensorFinsupp'_symm_single_eq_single_one_tmul]
  /-
    🎉 no goals
  -/


open FreeMonoidal in
/-- The free functor `Type u ⥤ ModuleCat R` is a monoidal functor. -/
instance : (free R).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := εIso R
      μIso := μIso R
      μIso_hom_natural_left := fun {X Y} f X' ↦ by
        /-
          R : Type u
          inst✝ : CommRing R
          X Y : Type u
          f : Quiver.Hom X Y
          X' : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        rw [← cancel_epi (μIso R X X').inv]
        /-
          R : Type u
          inst✝ : CommRing R
          X Y : Type u
          f : Quiver.Hom X Y
          X' : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.FreeMonoidal.μIso R X X'). …
        -/
        aesop
        /-
          🎉 no goals
        -/
      μIso_hom_natural_right := fun {X Y} X' f ↦ by
        /-
          R : Type u
          inst✝ : CommRing R
          X Y X' : Type u
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        rw [← cancel_epi (μIso R X' X).inv]
        /-
          R : Type u
          inst✝ : CommRing R
          X Y X' : Type u
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.FreeMonoidal.μIso R X' X). …
        -/
        aesop
        /-
          🎉 no goals
        -/
      associativity := fun X Y Z ↦ by
        /-
          R : Type u
          inst✝ : CommRing R
          X Y Z : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        rw [← cancel_epi ((μIso R X Y).inv ▷ _), ← cancel_epi (μIso R _ _).inv]
        /-
          R : Type u
          inst✝ : CommRing R
          X Y Z : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.FreeMonoidal.μIso R (Categ …
        -/
        ext ⟨⟨x, y⟩, z⟩
        /-
          case h.mk.mk
          R : Type u
          inst✝ : CommRing R
          X Y Z : Type u
          z : Z
          x : X
          y : Y
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.FreeMonoidal.μIso R (Cate …
        -/
        dsimp
        rw [μIso_inv_freeMk, MonoidalCategory.whiskerRight_apply, μIso_inv_freeMk,
          MonoidalCategory.whiskerRight_apply, μIso_hom_freeMk_tmul_freeMk,
          μIso_hom_freeMk_tmul_freeMk, free_map_apply,
          CategoryTheory.associator_hom_apply, MonoidalCategory.associator_hom_apply,
          MonoidalCategory.whiskerLeft_apply, μIso_hom_freeMk_tmul_freeMk,
          μIso_hom_freeMk_tmul_freeMk]
      left_unitality := fun X ↦ by
        /-
          R : Type u
          inst✝ : CommRing R
          X : Type u
          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((ModuleCat.free R).obj …
        -/
        rw [← cancel_epi (λ_ _).inv, Iso.inv_hom_id]
        /-
          R : Type u
          inst✝ : CommRing R
          X : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((ModuleCat.free R).obj X)) (CategoryTh …
        -/
        aesop
        /-
          🎉 no goals
        -/
      right_unitality := fun X ↦ by
        /-
          R : Type u
          inst✝ : CommRing R
          X : Type u
          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((ModuleCat.free R).ob …
        -/
        rw [← cancel_epi (ρ_ _).inv, Iso.inv_hom_id]
        /-
          R : Type u
          inst✝ : CommRing R
          X : Type u
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((ModuleCat.free R).obj X)) (CategoryTh …
        -/
        aesop }
        /-
          🎉 no goals
        -/


@[simp]
lemma free_ε_one : ε (free R) 1 = freeMk PUnit.unit := rfl


@[simp]
lemma free_η_freeMk (x : PUnit) : η (free R) (freeMk x) = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : PUnit.{u + 1}
    ⊢ Eq ((CategoryTheory.Functor.OplaxMonoidal.η (ModuleCat.free R)).hom (ModuleC …
  -/
  apply FreeMonoidal.εIso_inv_freeMk
  /-
    🎉 no goals
  -/


@[simp]
lemma free_μ_freeMk_tmul_freeMk {X Y : Type u} (x : X) (y : Y) :
    μ (free R) _ _ (freeMk x ⊗ₜ freeMk y) = freeMk ⟨x, y⟩ := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    x : X
    y : Y
    ⊢ Eq ((CategoryTheory.Functor.LaxMonoidal.μ (ModuleCat.free R) X Y).hom (Tenso …
  -/
  apply FreeMonoidal.μIso_hom_freeMk_tmul_freeMk
  /-
    🎉 no goals
  -/


@[simp]
lemma free_δ_freeMk {X Y : Type u} (z : X ⊗ Y) :
    δ (free R) _ _ (freeMk z) = freeMk z.1 ⊗ₜ freeMk z.2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : Type u
    z : CategoryTheory.MonoidalCategoryStruct.tensorObj X Y
    ⊢ Eq ((CategoryTheory.Functor.OplaxMonoidal.δ (ModuleCat.free R) X Y).hom (Mod …
  -/
  apply FreeMonoidal.μIso_inv_freeMk
  /-
    🎉 no goals
  -/


/-- `Free R C` is a type synonym for `C`, which, given `[CommRing R]` and `[Category C]`,
we will equip with a category structure where the morphisms are formal `R`-linear combinations
of the morphisms in `C`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/pull/5171): Removed has_nonempty_instance nolint; linter not ported yet
@[nolint unusedArguments]
def Free (_ : Type*) (C : Type u) :=
  C


/-- Consider an object of `C` as an object of the `R`-linear completion.

It may be preferable to use `(Free.embedding R C).obj X` instead;
this functor can also be used to lift morphisms.
-/
def Free.of (R : Type*) {C : Type u} (X : C) : Free R C :=
  X


instance categoryFree : Category (Free R C) where
  Hom := fun X Y : C => (X ⟶ Y) →₀ R
  id := fun X : C => Finsupp.single (𝟙 X) 1
  comp {X _ Z : C} f g :=
    (f.sum (fun f' s => g.sum (fun g' t => Finsupp.single (f' ≫ g') (s * t))) : (X ⟶ Z) →₀ R)
  assoc {W X Y Z} f g h := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom W X
      g : Quiver.Hom X Y
      h : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    dsimp
    -- This imitates the proof of associativity for `MonoidAlgebra`.
    simp only [sum_sum_index, sum_single_index, single_zero, single_add, eq_self_iff_true,
      forall_true_iff, forall₃_true_iff, add_mul, mul_add, Category.assoc, mul_assoc,
      zero_mul, mul_zero, sum_zero, sum_add]


instance : Preadditive (Free R C) where
  homGroup _ _ := Finsupp.instAddCommGroup
  add_comp X Y Z f f' g := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f f' : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f f') g) (HAdd.hAdd (Categ …
    -/
    dsimp [CategoryTheory.categoryFree]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f f' : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ((HAdd.hAdd f f').sum fun f' s => Finsupp.sum g fun g' t => Finsupp.singl …
    -/
                                      /-
                                        🎉 no goals
                                      -/
    rw [Finsupp.sum_add_index'] <;> · simp [add_mul]
                                      /-
                                        🎉 no goals
                                      -/
  comp_add X Y Z f g g' := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd g g')) (HAdd.hAdd (Categ …
    -/
    dsimp [CategoryTheory.categoryFree]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      ⊢ Eq (Finsupp.sum f fun f' s => (HAdd.hAdd g g').sum fun g' t => Finsupp.singl …
    -/
    rw [← Finsupp.sum_add]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      ⊢ Eq (Finsupp.sum f fun f' s => (HAdd.hAdd g g').sum fun g' t => Finsupp.singl …
    -/
    congr; ext r h
    /-
      case e_g.h.h.h
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      r : Quiver.Hom X Y
      h : R
      a✝ : Quiver.Hom X Z
      ⊢ Eq (((HAdd.hAdd g g').sum fun g' t => Finsupp.single (CategoryTheory.Categor …
    -/
                                      /-
                                        🎉 no goals
                                      -/
    rw [Finsupp.sum_add_index'] <;> · simp [mul_add]
                                      /-
                                        🎉 no goals
                                      -/


instance : Linear R (Free R C) where
  homModule _ _ := Finsupp.module _ R
  smul_comp X Y Z r f g := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      r : R
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r f) g) (HSMul.hSMul r ( …
    -/
    dsimp [CategoryTheory.categoryFree]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      r : R
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ((HSMul.hSMul r f).sum fun f' s => Finsupp.sum g fun g' t => Finsupp.sing …
    -/
                                    /-
                                      🎉 no goals
                                    -/
    rw [Finsupp.sum_smul_index] <;> simp [Finsupp.smul_sum, mul_assoc]
                                    /-
                                      🎉 no goals
                                    -/
  comp_smul X Y Z f r g := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HSMul.hSMul r g)) (HSMul.hSMul r ( …
    -/
    dsimp [CategoryTheory.categoryFree]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      g : Quiver.Hom Y Z
      ⊢ Eq (Finsupp.sum f fun f' s => (HSMul.hSMul r g).sum fun g' t => Finsupp.sing …
    -/
    simp_rw [Finsupp.smul_sum]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      g : Quiver.Hom Y Z
      ⊢ Eq (Finsupp.sum f fun f' s => (HSMul.hSMul r g).sum fun g' t => Finsupp.sing …
    -/
    congr; ext h s
    /-
      case e_g.h.h.h
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      g : Quiver.Hom Y Z
      h : Quiver.Hom X Y
      s : R
      a✝ : Quiver.Hom X Z
      ⊢ Eq (((HSMul.hSMul r g).sum fun g' t => Finsupp.single (CategoryTheory.Catego …
    -/
                                    /-
                                      🎉 no goals
                                    -/
    rw [Finsupp.sum_smul_index] <;> simp [Finsupp.smul_sum, mul_left_comm]
                                    /-
                                      🎉 no goals
                                    -/


theorem single_comp_single {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (r s : R) :
    (single f r ≫ single g s : Free.of R X ⟶ Free.of R Z) = single (f ≫ g) (r * s) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    r s : R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Finsupp.single f r) (Finsupp.single  …
  -/
  dsimp [CategoryTheory.categoryFree]; simp
                                       /-
                                         🎉 no goals
                                       -/


/-- A category embeds into its `R`-linear completion.
-/
@[simps]
def embedding : C ⥤ Free R C where
  obj X := X
  map {_ _} f := Finsupp.single f 1
  map_id _ := rfl
  map_comp {X Y Z} f g := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10959): simp used to be able to close this goal
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {x x_1} f => Finsupp.single f 1 }.map (C …
    -/
    dsimp only []
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (Finsupp.single (CategoryTheory.CategoryStruct.comp f g) 1) (CategoryTheo …
    -/
    rw [single_comp_single, one_mul]
    /-
      🎉 no goals
    -/


/-- A functor to an `R`-linear category lifts to a functor from its `R`-linear completion.
-/
@[simps]
def lift (F : C ⥤ D) : Free R C ⥤ D where
  obj X := F.obj X
  map {_ _} f := f.sum fun f' r => r • F.map f'
               /-
                 R : Type u_1
                 inst✝⁴ : CommRing R
                 C : Type u
                 inst✝³ : CategoryTheory.Category.{v, u} C
                 D : Type u
                 inst✝² : CategoryTheory.Category.{v, u} D
                 inst✝¹ : CategoryTheory.Preadditive D
                 inst✝ : CategoryTheory.Linear R D
                 F : CategoryTheory.Functor C D
                 ⊢ ∀ (X : CategoryTheory.Free R C), Eq ({ obj := fun X => F.obj X, map := fun { …
               -/
  map_id := by dsimp [CategoryTheory.categoryFree]; simp
                                                    /-
                                                      🎉 no goals
                                                    -/
  map_comp {X Y Z} f g := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝² : CategoryTheory.Category.{v, u} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      F : CategoryTheory.Functor C D
      X Y Z : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
    -/
    apply Finsupp.induction_linear f
      /-
        case h0
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case hadd
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ ∀ (f g_1 : Finsupp (Quiver.Hom X Y) R), Eq ({ obj := fun X => F.obj X, map : …
      -/
    · intro f₁ f₂ w₁ w₂
      /-
        case hadd
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        f₁ f₂ : Finsupp (Quiver.Hom X Y) R
        w₁ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
        w₂ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
        ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
      -/
      rw [add_comp]
      /-
        case hadd
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        f₁ f₂ : Finsupp (Quiver.Hom X Y) R
        w₁ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
        w₂ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
        ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
      -/
      dsimp at *
      /-
        case hadd
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        f₁ f₂ : Finsupp (Quiver.Hom X Y) R
        w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
        w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
        ⊢ Eq (Finsupp.sum (HAdd.hAdd (CategoryTheory.CategoryStruct.comp f₁ g) (Catego …
      -/
      rw [Finsupp.sum_add_index', Finsupp.sum_add_index']
        /-
          case hadd
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
          ⊢ Eq (HAdd.hAdd (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f'  …
        -/
      · simp only [w₁, w₂, add_comp]
        /-
          🎉 no goals
        -/
        /-
          case hadd.h_zero
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
          ⊢ ∀ (a : Quiver.Hom X Y), Eq (HSMul.hSMul 0 (F.map a)) 0
        -/
      · intros; rw [zero_smul]
                /-
                  🎉 no goals
                -/
        /-
          case hadd.h_add
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
          ⊢ ∀ (a : Quiver.Hom X Y) (b₁ b₂ : R), Eq (HSMul.hSMul (HAdd.hAdd b₁ b₂) (F.map …
        -/
      · intros; simp only [add_smul]
                /-
                  🎉 no goals
                -/
        /-
          case hadd.h_zero
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
          ⊢ ∀ (a : Quiver.Hom X Z), Eq (HSMul.hSMul 0 (F.map a)) 0
        -/
      · intros; rw [zero_smul]
                /-
                  🎉 no goals
                -/
        /-
          case hadd.h_add
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₁ g) fun f' r => HSM …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp f₂ g) fun f' r => HSM …
          ⊢ ∀ (a : Quiver.Hom X Z) (b₁ b₂ : R), Eq (HSMul.hSMul (HAdd.hAdd b₁ b₂) (F.map …
        -/
      · intros; simp only [add_smul]
                /-
                  🎉 no goals
                -/
      /-
        case hsingle
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ ∀ (a : Quiver.Hom X Y) (b : R), Eq ({ obj := fun X => F.obj X, map := fun {x …
      -/
    · intro f' r
      /-
        case hsingle
        R : Type u_1
        inst✝⁴ : CommRing R
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝² : CategoryTheory.Category.{v, u} D
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : CategoryTheory.Linear R D
        F : CategoryTheory.Functor C D
        X Y Z : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        f' : Quiver.Hom X Y
        r : R
        ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
      -/
      apply Finsupp.induction_linear g
        /-
          case hsingle.h0
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case hsingle.hadd
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          ⊢ ∀ (f g : Finsupp (Quiver.Hom Y Z) R), Eq ({ obj := fun X => F.obj X, map :=  …
        -/
      · intro f₁ f₂ w₁ w₂
        /-
          case hsingle.hadd
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
          w₁ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
          w₂ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
          ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
        -/
        rw [comp_add]
        /-
          case hsingle.hadd
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
          w₁ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
          w₂ : Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun  …
          ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
        -/
        dsimp at *
        /-
          case hsingle.hadd
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
          w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
          w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
          ⊢ Eq (Finsupp.sum (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Finsupp.sing …
        -/
        rw [Finsupp.sum_add_index', Finsupp.sum_add_index']
          /-
            case hsingle.hadd
            R : Type u_1
            inst✝⁴ : CommRing R
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u
            inst✝² : CategoryTheory.Category.{v, u} D
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : CategoryTheory.Linear R D
            F : CategoryTheory.Functor C D
            X Y Z : CategoryTheory.Free R C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            f' : Quiver.Hom X Y
            r : R
            f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
            w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            ⊢ Eq (HAdd.hAdd (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.sing …
          -/
        · simp only [w₁, w₂, comp_add]
          /-
            🎉 no goals
          -/
          /-
            case hsingle.hadd.h_zero
            R : Type u_1
            inst✝⁴ : CommRing R
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u
            inst✝² : CategoryTheory.Category.{v, u} D
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : CategoryTheory.Linear R D
            F : CategoryTheory.Functor C D
            X Y Z : CategoryTheory.Free R C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            f' : Quiver.Hom X Y
            r : R
            f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
            w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            ⊢ ∀ (a : Quiver.Hom Y Z), Eq (HSMul.hSMul 0 (F.map a)) 0
          -/
        · intros; rw [zero_smul]
                  /-
                    🎉 no goals
                  -/
          /-
            case hsingle.hadd.h_add
            R : Type u_1
            inst✝⁴ : CommRing R
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u
            inst✝² : CategoryTheory.Category.{v, u} D
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : CategoryTheory.Linear R D
            F : CategoryTheory.Functor C D
            X Y Z : CategoryTheory.Free R C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            f' : Quiver.Hom X Y
            r : R
            f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
            w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            ⊢ ∀ (a : Quiver.Hom Y Z) (b₁ b₂ : R), Eq (HSMul.hSMul (HAdd.hAdd b₁ b₂) (F.map …
          -/
        · intros; simp only [add_smul]
                  /-
                    🎉 no goals
                  -/
          /-
            case hsingle.hadd.h_zero
            R : Type u_1
            inst✝⁴ : CommRing R
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u
            inst✝² : CategoryTheory.Category.{v, u} D
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : CategoryTheory.Linear R D
            F : CategoryTheory.Functor C D
            X Y Z : CategoryTheory.Free R C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            f' : Quiver.Hom X Y
            r : R
            f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
            w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            ⊢ ∀ (a : Quiver.Hom X Z), Eq (HSMul.hSMul 0 (F.map a)) 0
          -/
        · intros; rw [zero_smul]
                  /-
                    🎉 no goals
                  -/
          /-
            case hsingle.hadd.h_add
            R : Type u_1
            inst✝⁴ : CommRing R
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u
            inst✝² : CategoryTheory.Category.{v, u} D
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : CategoryTheory.Linear R D
            F : CategoryTheory.Functor C D
            X Y Z : CategoryTheory.Free R C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            f' : Quiver.Hom X Y
            r : R
            f₁ f₂ : Finsupp (Quiver.Hom Y Z) R
            w₁ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            w₂ : Eq (Finsupp.sum (CategoryTheory.CategoryStruct.comp (Finsupp.single f' r) …
            ⊢ ∀ (a : Quiver.Hom X Z) (b₁ b₂ : R), Eq (HSMul.hSMul (HAdd.hAdd b₁ b₂) (F.map …
          -/
        · intros; simp only [add_smul]
                  /-
                    🎉 no goals
                  -/
        /-
          case hsingle.hsingle
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          ⊢ ∀ (a : Quiver.Hom Y Z) (b : R), Eq ({ obj := fun X => F.obj X, map := fun {x …
        -/
      · intro g' s
        /-
          case hsingle.hsingle
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          g' : Quiver.Hom Y Z
          s : R
          ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
        -/
        rw [single_comp_single _ _ f' g' r s]
        /-
          case hsingle.hsingle
          R : Type u_1
          inst✝⁴ : CommRing R
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝² : CategoryTheory.Category.{v, u} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F : CategoryTheory.Functor C D
          X Y Z : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X Y
          r : R
          g' : Quiver.Hom Y Z
          s : R
          ⊢ Eq ({ obj := fun X => F.obj X, map := fun {x x_1} f => Finsupp.sum f fun f'  …
        -/
        simp [mul_comm r s, mul_smul]
        /-
          🎉 no goals
        -/


theorem lift_map_single (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) (r : R) :
                                                    /-
                                                      R : Type u_1
                                                      inst✝⁴ : CommRing R
                                                      C : Type u
                                                      inst✝³ : CategoryTheory.Category.{v, u} C
                                                      D : Type u
                                                      inst✝² : CategoryTheory.Category.{v, u} D
                                                      inst✝¹ : CategoryTheory.Preadditive D
                                                      inst✝ : CategoryTheory.Linear R D
                                                      F : CategoryTheory.Functor C D
                                                      X Y : C
                                                      f : Quiver.Hom X Y
                                                      r : R
                                                      ⊢ Eq ((CategoryTheory.Free.lift R F).map (Finsupp.single f r)) (HSMul.hSMul r  …
                                                    -/
    (lift R F).map (single f r) = r • F.map f := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


instance lift_additive (F : C ⥤ D) : (lift R F).Additive where
  map_add {X Y} f g := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝² : CategoryTheory.Category.{v, u} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      F : CategoryTheory.Functor C D
      X Y : CategoryTheory.Free R C
      f g : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.Free.lift R F).map (HAdd.hAdd f g)) (HAdd.hAdd ((Categor …
    -/
    dsimp
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝² : CategoryTheory.Category.{v, u} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      F : CategoryTheory.Functor C D
      X Y : CategoryTheory.Free R C
      f g : Quiver.Hom X Y
      ⊢ Eq (Finsupp.sum (HAdd.hAdd f g) fun f' r => HSMul.hSMul r (F.map f')) (HAdd. …
    -/
                                    /-
                                      🎉 no goals
                                    -/
    rw [Finsupp.sum_add_index'] <;> simp [add_smul]
                                    /-
                                      🎉 no goals
                                    -/


instance lift_linear (F : C ⥤ D) : (lift R F).Linear R where
  map_smul {X Y} f r := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝² : CategoryTheory.Category.{v, u} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      F : CategoryTheory.Functor C D
      X Y : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      ⊢ Eq ((CategoryTheory.Free.lift R F).map (HSMul.hSMul r f)) (HSMul.hSMul r ((C …
    -/
    dsimp
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝² : CategoryTheory.Category.{v, u} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      F : CategoryTheory.Functor C D
      X Y : CategoryTheory.Free R C
      f : Quiver.Hom X Y
      r : R
      ⊢ Eq (Finsupp.sum (HSMul.hSMul r f) fun f' r => HSMul.hSMul r (F.map f')) (HSM …
    -/
                                    /-
                                      🎉 no goals
                                    -/
    rw [Finsupp.sum_smul_index] <;> simp [Finsupp.smul_sum, mul_smul]
                                    /-
                                      🎉 no goals
                                    -/


/-- The embedding into the `R`-linear completion, followed by the lift,
is isomorphic to the original functor.
-/
def embeddingLiftIso (F : C ⥤ D) : embedding R C ⋙ lift R F ≅ F :=
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    inst✝¹ : CategoryTheory.Preadditive D
    inst✝ : CategoryTheory.Linear R D
    F : CategoryTheory.Functor C D
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- Two `R`-linear functors out of the `R`-linear completion are isomorphic iff their
compositions with the embedding functor are isomorphic.
-/
def ext {F G : Free R C ⥤ D} [F.Additive] [F.Linear R] [G.Additive] [G.Linear R]
    (α : embedding R C ⋙ F ≅ embedding R C ⋙ G) : F ≅ G :=
  NatIso.ofComponents (fun X => α.app X)
    (by
      /-
        R : Type u_1
        inst✝⁸ : CommRing R
        C : Type u
        inst✝⁷ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} D
        inst✝⁵ : CategoryTheory.Preadditive D
        inst✝⁴ : CategoryTheory.Linear R D
        F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
        inst✝³ : F.Additive
        inst✝² : CategoryTheory.Functor.Linear R F
        inst✝¹ : G.Additive
        inst✝ : CategoryTheory.Functor.Linear R G
        α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
        ⊢ ∀ {X Y : CategoryTheory.Free R C} (f : Quiver.Hom X Y), Eq (CategoryTheory.C …
      -/
      intro X Y f
      /-
        R : Type u_1
        inst✝⁸ : CommRing R
        C : Type u
        inst✝⁷ : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} D
        inst✝⁵ : CategoryTheory.Preadditive D
        inst✝⁴ : CategoryTheory.Linear R D
        F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
        inst✝³ : F.Additive
        inst✝² : CategoryTheory.Functor.Linear R F
        inst✝¹ : G.Additive
        inst✝ : CategoryTheory.Functor.Linear R G
        α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
        X Y : CategoryTheory.Free R C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X => α.app X) Y).hom) …
      -/
      apply Finsupp.induction_linear f
        /-
          case h0
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) ((fun X => α.app X) Y).hom) …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case hadd
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          ⊢ ∀ (f g : Finsupp (Quiver.Hom X Y) R), Eq (CategoryTheory.CategoryStruct.comp …
        -/
      · intro f₁ f₂ w₁ w₂
        -- Porting note: Using rw instead of simp
        /-
          case hadd
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          f₁ f₂ : Finsupp (Quiver.Hom X Y) R
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map f₁) ((fun X => α.app X) Y). …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map f₂) ((fun X => α.app X) Y). …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (HAdd.hAdd f₁ f₂)) ((fun X =>  …
        -/
        rw [Functor.map_add, add_comp, w₁, w₂, Functor.map_add, comp_add]
        /-
          🎉 no goals
        -/
        /-
          case hsingle
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          ⊢ ∀ (a : Quiver.Hom X Y) (b : R), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        -/
      · intro f' r
        rw [Iso.app_hom, Iso.app_hom, ← smul_single_one, F.map_smul, G.map_smul, smul_comp,
          comp_smul]
        /-
          case hsingle
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          f' : Quiver.Hom X Y
          r : R
          ⊢ Eq (HSMul.hSMul r (CategoryTheory.CategoryStruct.comp (F.map (Finsupp.single …
        -/
        change r • (embedding R C ⋙ F).map f' ≫ _ = r • _ ≫ (embedding R C ⋙ G).map f'
        /-
          case hsingle
          R : Type u_1
          inst✝⁸ : CommRing R
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          D : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} D
          inst✝⁵ : CategoryTheory.Preadditive D
          inst✝⁴ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor (CategoryTheory.Free R C) D
          inst✝³ : F.Additive
          inst✝² : CategoryTheory.Functor.Linear R F
          inst✝¹ : G.Additive
          inst✝ : CategoryTheory.Functor.Linear R G
          α : CategoryTheory.Iso ((CategoryTheory.Free.embedding R C).comp F) ((Category …
          X Y : CategoryTheory.Free R C
          f : Quiver.Hom X Y
          f' : Quiver.Hom X Y
          r : R
          ⊢ Eq (HSMul.hSMul r (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Free …
        -/
        rw [α.hom.naturality f'])
        /-
          🎉 no goals
        -/


/-- `Free.lift` is unique amongst `R`-linear functors `Free R C ⥤ D`
which compose with `embedding ℤ C` to give the original functor.
-/
def liftUnique (F : C ⥤ D) (L : Free R C ⥤ D) [L.Additive] [L.Linear R]
    (α : embedding R C ⋙ L ≅ F) : L ≅ lift R F :=
  ext R (α.trans (embeddingLiftIso R F).symm)


