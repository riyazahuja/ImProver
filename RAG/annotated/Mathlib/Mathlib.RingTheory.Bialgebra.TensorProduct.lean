lemma counit_eq_algHom_toLinearMap :
    Coalgebra.counit (R := R) (A := A ⊗[R] B) =
      ((Algebra.TensorProduct.lmul' R).comp (Algebra.TensorProduct.map
      (Bialgebra.counitAlgHom R A) (Bialgebra.counitAlgHom R B))).toLinearMap := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    ⊢ Eq CoalgebraStruct.counit ((Algebra.TensorProduct.lmul' R).comp (Algebra.Ten …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma comul_eq_algHom_toLinearMap :
    Coalgebra.comul (R := R) (A := A ⊗[R] B) =
      ((Algebra.TensorProduct.tensorTensorTensorComm R A A B B).toAlgHom.comp
      (Algebra.TensorProduct.map (Bialgebra.comulAlgHom R A)
      (Bialgebra.comulAlgHom R B))).toLinearMap := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    ⊢ Eq CoalgebraStruct.comul ((↑(Algebra.TensorProduct.tensorTensorTensorComm R  …
  -/
  rfl
  /-
    🎉 no goals
  -/


noncomputable instance TensorProduct.instBialgebra
    {R A B : Type u} [CommRing R] [Ring A] [Ring B]
    [Bialgebra R A] [Bialgebra R B] : Bialgebra R (A ⊗[R] B) := by
  /-
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    ⊢ Bialgebra R (TensorProduct R A B)
  -/
  have hcounit := congr(DFunLike.coe $(Bialgebra.TensorProduct.counit_eq_algHom_toLinearMap R A B))
  /-
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    hcounit : Eq ⇑CoalgebraStruct.counit ⇑((Algebra.TensorProduct.lmul' R).comp (A …
    ⊢ Bialgebra R (TensorProduct R A B)
  -/
  have hcomul := congr(DFunLike.coe $(Bialgebra.TensorProduct.comul_eq_algHom_toLinearMap R A B))
  /-
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    hcounit : Eq ⇑CoalgebraStruct.counit ⇑((Algebra.TensorProduct.lmul' R).comp (A …
    hcomul : Eq ⇑CoalgebraStruct.comul ⇑((↑(Algebra.TensorProduct.tensorTensorTens …
    ⊢ Bialgebra R (TensorProduct R A B)
  -/
  refine Bialgebra.mk' R (A ⊗[R] B) ?_ (fun {x y} => ?_) ?_ (fun {x y} => ?_) <;>
  /-
    case refine_1
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    hcounit : Eq ⇑CoalgebraStruct.counit ⇑((Algebra.TensorProduct.lmul' R).comp (A …
    hcomul : Eq ⇑CoalgebraStruct.comul ⇑((↑(Algebra.TensorProduct.tensorTensorTens …
    ⊢ Eq (CoalgebraStruct.counit 1) 1
  -/
  simp_all only [AlgHom.toLinearMap_apply] <;>
  /-
    case refine_1
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Bialgebra R A
    inst✝ : Bialgebra R B
    hcounit : Eq ⇑CoalgebraStruct.counit ⇑((Algebra.TensorProduct.lmul' R).comp (A …
    hcomul : Eq ⇑CoalgebraStruct.comul ⇑((↑(Algebra.TensorProduct.tensorTensorTens …
    ⊢ Eq (((Algebra.TensorProduct.lmul' R).comp (Algebra.TensorProduct.map (Bialge …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp only [_root_.map_one, _root_.map_mul]
  /-
    🎉 no goals
  -/


/-- The tensor product of two bialgebra morphisms as a bialgebra morphism. -/
noncomputable def map (f : A →ₐc[R] B) (g : C →ₐc[R] D) :
    A ⊗[R] C →ₐc[R] B ⊗[R] D :=
  { Coalgebra.TensorProduct.map (f : A →ₗc[R] B) (g : C →ₗc[R] D),
    Algebra.TensorProduct.map (f : A →ₐ[R] B) (g : C →ₐ[R] D) with }


@[simp]
theorem map_tmul (f : A →ₐc[R] B) (g : C →ₐc[R] D) (x : A) (y : C) :
    map f g (x ⊗ₜ y) = f x ⊗ₜ g y :=
  rfl


@[simp]
theorem map_toCoalgHom (f : A →ₐc[R] B) (g : C →ₐc[R] D) :
    map f g = Coalgebra.TensorProduct.map (f : A →ₗc[R] B) (g : C →ₗc[R] D) := rfl


@[simp]
theorem map_toAlgHom (f : A →ₐc[R] B) (g : C →ₐc[R] D) :
    (map f g : A ⊗[R] C →ₐ[R] B ⊗[R] D) =
      Algebra.TensorProduct.map (f : A →ₐ[R] B) (g : C →ₐ[R] D) :=
  rfl


variable (R A B C) in
/-- The associator for tensor products of R-bialgebras, as a bialgebra equivalence. -/
protected noncomputable def assoc :
    (A ⊗[R] B) ⊗[R] C ≃ₐc[R] A ⊗[R] (B ⊗[R] C) :=
  { Coalgebra.TensorProduct.assoc R A B C, Algebra.TensorProduct.assoc R A B C with }


@[simp]
theorem assoc_tmul (x : A) (y : B) (z : C) :
    Bialgebra.TensorProduct.assoc R A B C ((x ⊗ₜ y) ⊗ₜ z) = x ⊗ₜ (y ⊗ₜ z) :=
  rfl


@[simp]
theorem assoc_symm_tmul (x : A) (y : B) (z : C) :
    (Bialgebra.TensorProduct.assoc R A B C).symm (x ⊗ₜ (y ⊗ₜ z)) = (x ⊗ₜ y) ⊗ₜ z :=
  rfl


@[simp]
theorem assoc_toCoalgEquiv :
    (Bialgebra.TensorProduct.assoc R A B C : _ ≃ₗc[R] _) =
    Coalgebra.TensorProduct.assoc R A B C := rfl


@[simp]
theorem assoc_toAlgEquiv :
    (Bialgebra.TensorProduct.assoc R A B C : _ ≃ₐ[R] _) =
    Algebra.TensorProduct.assoc R A B C := rfl


variable (R A) in
/-- The base ring is a left identity for the tensor product of bialgebras, up to
bialgebra equivalence. -/
protected noncomputable def lid : R ⊗[R] A ≃ₐc[R] A :=
  { Coalgebra.TensorProduct.lid R A, Algebra.TensorProduct.lid R A with }


@[simp]
theorem lid_toCoalgEquiv :
    (Bialgebra.TensorProduct.lid R A : R ⊗[R] A ≃ₗc[R] A) = Coalgebra.TensorProduct.lid R A := rfl


@[simp]
theorem lid_toAlgEquiv :
    (Bialgebra.TensorProduct.lid R A : R ⊗[R] A ≃ₐ[R] A) = Algebra.TensorProduct.lid R A := rfl


@[simp]
theorem lid_tmul (r : R) (a : A) : Bialgebra.TensorProduct.lid R A (r ⊗ₜ a) = r • a := rfl


@[simp]
theorem lid_symm_apply (a : A) : (Bialgebra.TensorProduct.lid R A).symm a = 1 ⊗ₜ a := rfl

/- TODO: make this defeq, which would involve adding a heterobasic version of
`Coalgebra.TensorProduct.rid`. -/

theorem coalgebra_rid_eq_algebra_rid_apply (x : A ⊗[R] R) :
    Coalgebra.TensorProduct.rid R A x = Algebra.TensorProduct.rid R R A x :=
  congr($((TensorProduct.AlgebraTensorModule.rid_eq_rid R A).symm) x)


variable (R A) in
/-- The base ring is a right identity for the tensor product of bialgebras, up to
bialgebra equivalence. -/
protected noncomputable def rid : A ⊗[R] R ≃ₐc[R] A where
  toCoalgEquiv := Coalgebra.TensorProduct.rid R A
  map_mul' x y := by
    simp only [CoalgEquiv.toCoalgHom_eq_coe, CoalgHom.toLinearMap_eq_coe, AddHom.toFun_eq_coe,
      LinearMap.coe_toAddHom, CoalgHom.coe_toLinearMap, CoalgHom.coe_coe,
      coalgebra_rid_eq_algebra_rid_apply, map_mul]


@[simp]
theorem rid_toCoalgEquiv :
    (Bialgebra.TensorProduct.rid R A : A ⊗[R] R ≃ₗc[R] A) = Coalgebra.TensorProduct.rid R A := rfl


@[simp]
theorem rid_toAlgEquiv :
    (Bialgebra.TensorProduct.rid R A : A ⊗[R] R ≃ₐ[R] A) = Algebra.TensorProduct.rid R R A := by
  /-
    R A : Type u
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Bialgebra R A
    ⊢ Eq (↑(Bialgebra.TensorProduct.rid R A)) (Algebra.TensorProduct.rid R R A)
  -/
  ext x
  /-
    case h
    R A : Type u
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Bialgebra R A
    x : TensorProduct R A R
    ⊢ Eq (↑(Bialgebra.TensorProduct.rid R A) x) ((Algebra.TensorProduct.rid R R A) …
  -/
  exact coalgebra_rid_eq_algebra_rid_apply x
  /-
    🎉 no goals
  -/


@[simp]
theorem rid_tmul (r : R) (a : A) : Bialgebra.TensorProduct.rid R A (a ⊗ₜ r) = r • a := rfl


@[simp]
theorem rid_symm_apply (a : A) : (Bialgebra.TensorProduct.rid R A).symm a = a ⊗ₜ 1 := rfl


/-- `lTensor A f : A ⊗ B →ₐc A ⊗ C` is the natural bialgebra morphism induced by `f : B →ₐc C`. -/
noncomputable abbrev lTensor (f : B →ₐc[R] C) : A ⊗[R] B →ₐc[R] A ⊗[R] C :=
  Bialgebra.TensorProduct.map (BialgHom.id R A) f


/-- `rTensor A f : B ⊗ A →ₐc C ⊗ A` is the natural bialgebra morphism induced by `f : B →ₐc C`. -/
noncomputable abbrev rTensor (f : B →ₐc[R] C) : B ⊗[R] A →ₐc[R] C ⊗[R] A :=
  Bialgebra.TensorProduct.map f (BialgHom.id R A)


