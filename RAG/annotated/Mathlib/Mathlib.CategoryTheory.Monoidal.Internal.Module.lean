instance Ring_of_Mon_ (A : Mon_ (ModuleCat.{u} R)) : Ring A.X :=
  { (inferInstance : AddCommGroup A.X) with
    one := A.one (1 : R)
    mul := fun x y => A.mul (x ⊗ₜ y)
    one_mul := fun x => by
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq (HMul.hMul 1 x) x
      -/
      convert LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.one_mul) ((1 : R) ⊗ₜ x)
      /-
        case h.e'_3
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq x ((CategoryTheory.MonoidalCategoryStruct.leftUnitor A.X).hom.hom (Tensor …
      -/
      rw [MonoidalCategory.leftUnitor_hom_apply, one_smul]
      /-
        🎉 no goals
      -/
    mul_one := fun x => by
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
      -/
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq (HMul.hMul x 1) x
      -/
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (HAdd.hAdd (HMul.hMul x y) (HMul.hMul x z))
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (A.mul.hom (HAdd.hAdd (TensorProduct.tmul R …
      -/
      convert LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.mul_one) (x ⊗ₜ (1 : R))
      /-
        case h.e'_2
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (A.mul.hom (TensorProduct.tmul R x (HAdd.hA …
      -/
      /-
        case h.e'_3
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq x ((CategoryTheory.MonoidalCategoryStruct.rightUnitor A.X).hom.hom (Tenso …
      -/
      /-
        🎉 no goals
      -/
      rw [MonoidalCategory.rightUnitor_hom_apply, one_smul]
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (A.mul.hom (HAdd.hAdd (TensorProduct.tmul R …
      -/
    mul_assoc := fun x y z => by
      /-
        case h.e'_2
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y z : ↑A.X
        ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (A.mul.hom (TensorProduct.tmul R (HAdd.hAdd …
      -/
      convert LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.mul_assoc) (x ⊗ₜ y ⊗ₜ z)
      /-
        🎉 no goals
      -/
    left_distrib := fun x y z => by
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq (A.mul.hom (TensorProduct.tmul R 0 x)) 0
      -/
      convert A.mul.hom.map_add (x ⊗ₜ y) (x ⊗ₜ z)
      /-
        🎉 no goals
      -/
      rw [← TensorProduct.tmul_add]
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x : ↑A.X
        ⊢ Eq (A.mul.hom (TensorProduct.tmul R x 0)) 0
      -/
      rfl
      /-
        🎉 no goals
      -/
    right_distrib := fun x y z => by
      convert A.mul.hom.map_add (x ⊗ₜ z) (y ⊗ₜ z)
      rw [← TensorProduct.add_tmul]
      rfl
    zero_mul := fun x => show A.mul _ = 0 by
      rw [TensorProduct.zero_tmul, map_zero]
    mul_zero := fun x => show A.mul _ = 0 by
      rw [TensorProduct.tmul_zero, map_zero] }


instance Algebra_of_Mon_ (A : Mon_ (ModuleCat.{u} R)) : Algebra R A.X :=
  { A.one.hom with
    map_zero' := A.one.hom.map_zero
    map_one' := rfl
    map_mul' := fun x y => by
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y : R
        ⊢ Eq ({ toFun := __src✝.toFun, map_one' := ⋯ }.toFun (HMul.hMul x y)) (HMul.hM …
      -/
      have h := LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.one_mul.symm) (x ⊗ₜ A.one y)
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        x y : R
        h : Eq ((CategoryTheory.MonoidalCategoryStruct.leftUnitor A.X).hom.hom (Tensor …
        ⊢ Eq ({ toFun := __src✝.toFun, map_one' := ⋯ }.toFun (HMul.hMul x y)) (HMul.hM …
      -/
      rwa [MonoidalCategory.leftUnitor_hom_apply, ← A.one.hom.map_smul] at h
      /-
        🎉 no goals
      -/
    commutes' := fun r a => by
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        r : R
        a : ↑A.X
        ⊢ Eq (HMul.hMul ({ toFun := __src✝.toFun, map_one' := ⋯, map_mul' := ⋯, map_ze …
      -/
      dsimp
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        r : R
        a : ↑A.X
        ⊢ Eq (HMul.hMul (A.one.hom r) a) (HMul.hMul a (A.one.hom r))
      -/
      have h₁ := LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.one_mul) (r ⊗ₜ a)
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        r : R
        a : ↑A.X
        h₁ : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryS …
        ⊢ Eq (HMul.hMul (A.one.hom r) a) (HMul.hMul a (A.one.hom r))
      -/
      have h₂ := LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.mul_one) (a ⊗ₜ r)
      /-
        R : Type u
        inst✝ : CommRing R
        A : Mon_ (ModuleCat R)
        r : R
        a : ↑A.X
        h₁ : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryS …
        h₂ : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryS …
        ⊢ Eq (HMul.hMul (A.one.hom r) a) (HMul.hMul a (A.one.hom r))
      -/
      exact h₁.trans h₂.symm
      /-
        🎉 no goals
      -/
    smul_def' := fun r a =>
      (LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp A.one_mul) (r ⊗ₜ a)).symm }


@[simp]
theorem algebraMap (A : Mon_ (ModuleCat.{u} R)) (r : R) : algebraMap R A.X r = A.one r :=
  rfl


/-- Converting a monoid object in `ModuleCat R` to a bundled algebra.
-/
@[simps!]
def functor : Mon_ (ModuleCat.{u} R) ⥤ AlgebraCat R where
  obj A := AlgebraCat.of R A.X
  map {_ _} f := AlgebraCat.ofHom
    { f.hom.hom.toAddMonoidHom with
      toFun := f.hom
      map_one' := LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp f.one_hom) (1 : R)
      map_mul' := fun x y => LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp f.mul_hom) (x ⊗ₜ y)
      commutes' := fun r => LinearMap.congr_fun (ModuleCat.hom_ext_iff.mp f.one_hom) r }


/-- Converting a bundled algebra to a monoid object in `ModuleCat R`.
-/
@[simps]
def inverseObj (A : AlgebraCat.{u} R) : Mon_ (ModuleCat.{u} R) where
  X := ModuleCat.of R A
  one := ofHom <| Algebra.linearMap R A
  mul := ofHom <| LinearMap.mul' R A
  one_mul := by
    /-
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext : 1
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `TensorProduct.ext`
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    refine TensorProduct.ext <| LinearMap.ext_ring <| LinearMap.ext fun x => ?_
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq ((((TensorProduct.mk R ↑CategoryTheory.MonoidalCategoryStruct.tensorUnit  …
    -/
    rw [compr₂_apply, compr₂_apply, hom_comp, LinearMap.comp_apply]
    -- Porting note: this `dsimp` does nothing
    -- dsimp [AlgebraCat.id_apply, TensorProduct.mk_apply, Algebra.linearMap_apply,
    --    LinearMap.compr₂_apply, Function.comp_apply, RingHom.map_one,
    --    ModuleCat.MonoidalCategory.tensorHom_tmul, AlgebraCat.hom_comp,
    --    ModuleCat.MonoidalCategory.leftUnitor_hom_apply]
    -- Porting note: because `dsimp` is not effective, `rw` needs to be changed to `erw`
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq ((ModuleCat.ofHom (LinearMap.mul' R ↑A)).hom ((CategoryTheory.MonoidalCat …
    -/
    dsimp
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq ((LinearMap.mul' R ↑A) ((CategoryTheory.MonoidalCategoryStruct.whiskerRig …
    -/
    erw [LinearMap.mul'_apply, MonoidalCategory.leftUnitor_hom_apply, ← Algebra.smul_def]
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq (HSMul.hSMul { fst := 1, snd := x }.1 (LinearMap.id { fst := 1, snd := x  …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  mul_one := by
    /-
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext : 1
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `TensorProduct.ext`
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    refine TensorProduct.ext <| LinearMap.ext fun x => LinearMap.ext_ring ?_
    -- Porting note: this `dsimp` does nothing
    -- dsimp only [AlgebraCat.id_apply, TensorProduct.mk_apply, Algebra.linearMap_apply,
    --   LinearMap.compr₂_apply, Function.comp_apply, ModuleCat.MonoidalCategory.hom_apply,
    --   AlgebraCat.coe_comp]
    -- Porting note: because `dsimp` is not effective, `rw` needs to be changed to `erw`
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq ((((TensorProduct.mk R ↑(ModuleCat.of R ↑A) ↑CategoryTheory.MonoidalCateg …
    -/
    erw [compr₂_apply, compr₂_apply]
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
    -/
    rw [ModuleCat.hom_comp, LinearMap.comp_apply]
    erw [LinearMap.mul'_apply, ModuleCat.MonoidalCategory.rightUnitor_hom_apply, ← Algebra.commutes,
      ← Algebra.smul_def]
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      x : ↑(ModuleCat.of R ↑A)
      ⊢ Eq (HSMul.hSMul { fst := x, snd := 1 }.2 (LinearMap.id { fst := x, snd := 1  …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  mul_assoc := by
    /-
      R : Type u
      inst✝ : CommRing R
      A : AlgebraCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext : 1
    set_option tactic.skipAssignedInstances false in
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `TensorProduct.ext`
    refine TensorProduct.ext <| TensorProduct.ext <| LinearMap.ext fun x => LinearMap.ext fun y =>
      LinearMap.ext fun z => ?_
    dsimp only [compr₂_apply, TensorProduct.mk_apply]
    rw [compr₂_apply, compr₂_apply]
    rw [hom_comp, LinearMap.comp_apply, hom_comp, LinearMap.comp_apply, hom_comp,
        LinearMap.comp_apply]
    erw [LinearMap.mul'_apply, LinearMap.mul'_apply]
    dsimp only [id_coe, id_eq]
    erw [TensorProduct.mk_apply, TensorProduct.mk_apply, mul'_apply, LinearMap.id_apply, mul'_apply]
    simp only [LinearMap.mul'_apply, mul_assoc]


/-- Converting a bundled algebra to a monoid object in `ModuleCat R`.
-/
@[simps]
def inverse : AlgebraCat.{u} R ⥤ Mon_ (ModuleCat.{u} R) where
  obj := inverseObj
  map f :=
    { hom := ofHom <| f.hom.toLinearMap
      one_hom := hom_ext <| LinearMap.ext f.hom.commutes
      mul_hom := hom_ext <| TensorProduct.ext <| LinearMap.ext₂ <| map_mul f.hom }


set_option maxHeartbeats 400000 in
/-- The category of internal monoid objects in `ModuleCat R`
is equivalent to the category of "native" bundled `R`-algebras.
-/
def monModuleEquivalenceAlgebra : Mon_ (ModuleCat.{u} R) ≌ AlgebraCat R where
  functor := functor
  inverse := inverse
  unitIso :=
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y : Mon_ (ModuleCat R)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
    -/
    NatIso.ofComponents
    /-
      🎉 no goals
    -/
      (fun A =>
        { hom :=
            { hom := ofHom
                { toFun := _root_.id
                  map_add' := fun _ _ => rfl
                /-
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Mon_ (Mo …
                -/
                  map_smul' := fun _ _ => rfl }
              mul_hom := by
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Mon_ (Mo …
                -/
                ext : 1
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq ((TensorProduct.mk R ↑((CategoryTheory.Functor.id (Mon_ (ModuleCat R))).o …
                -/
                -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `TensorProduct.ext`
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq ((TensorProduct.mk R ↑A.X ↑A.X).compr₂ ({ toFun := id, map_add' := ⋯, map …
                -/
                refine TensorProduct.ext ?_
                /-
                  🎉 no goals
                -/
                dsimp at *
                rfl }
          inv :=
            { hom := ofHom
                { toFun := _root_.id
                  map_add' := fun _ _ => rfl
                /-
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.MonModuleEquivalenceAlgeb …
                -/
                  map_smul' := fun _ _ => rfl }
              mul_hom := by
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.MonModuleEquivalenceAlgeb …
                -/
                ext : 1
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq ((TensorProduct.mk R ↑((ModuleCat.MonModuleEquivalenceAlgebra.functor.com …
                -/
                -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `TensorProduct.ext`
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  A : Mon_ (ModuleCat R)
                  ⊢ Eq ((TensorProduct.mk R ↑A.X ↑A.X).compr₂ ({ toFun := id, map_add' := ⋯, map …
                -/
                refine TensorProduct.ext ?_
                /-
                  🎉 no goals
                -/
                dsimp at *
                rfl } })
  counitIso :=
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y : AlgebraCat R} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
    -/
    NatIso.ofComponents
    /-
      🎉 no goals
    -/
      (fun A =>
        { hom := AlgebraCat.ofHom
            { toFun := _root_.id
              map_zero' := rfl
              map_add' := fun _ _ => rfl
              map_one' := (algebraMap R A).map_one
              map_mul' := fun x y => @LinearMap.mul'_apply R _ _ _ _ _ _ x y
              commutes' := fun _ => rfl }
          inv := AlgebraCat.ofHom
            { toFun := _root_.id
              map_zero' := rfl
              map_add' := fun _ _ => rfl
              map_one' := (algebraMap R A).map_one.symm
              map_mul' := fun x y => (@LinearMap.mul'_apply R _ _ _ _ _ _ x y).symm
              commutes' := fun _ => rfl } })


/-- The equivalence `Mon_ (ModuleCat R) ≌ AlgebraCat R`
is naturally compatible with the forgetful functors to `ModuleCat R`.
-/
def monModuleEquivalenceAlgebraForget :
    MonModuleEquivalenceAlgebra.functor ⋙ forget₂ (AlgebraCat.{u} R) (ModuleCat.{u} R) ≅
      Mon_.forget (ModuleCat.{u} R) :=
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ ∀ {X Y : Mon_ (ModuleCat R)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
  -/
  NatIso.ofComponents
  /-
    🎉 no goals
  -/
    (fun A =>
      { hom := ofHom
          { toFun := _root_.id
            map_add' := fun _ _ => rfl
            map_smul' := fun _ _ => rfl }
        inv := ofHom
          { toFun := _root_.id
            map_add' := fun _ _ => rfl
            map_smul' := fun _ _ => rfl } })


