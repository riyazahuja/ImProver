open MonoidalCategory in
noncomputable instance TensorProduct.instCoalgebra : Coalgebra R (M ⊗[R] N) :=
  let I := Monoidal.transport ((CoalgebraCat.comonEquivalence R).symm)
  CoalgEquiv.toCoalgebra
    (A := (CoalgebraCat.of R M ⊗ CoalgebraCat.of R N : CoalgebraCat R))
    { LinearEquiv.refl R _ with
      counit_comp := rfl
      map_comp_comul := by
        /-
          R M N P Q : Type u
          inst✝⁶ : CommRing R
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : AddCommGroup N
          inst✝³ : Module R M
          inst✝² : Module R N
          inst✝¹ : Coalgebra R M
          inst✝ : Coalgebra R N
          I : CategoryTheory.MonoidalCategory (CoalgebraCat R) := CategoryTheory.Monoida …
          ⊢ Eq ((TensorProduct.map ↑__src✝ ↑__src✝).comp CoalgebraStruct.comul) (Coalgeb …
        -/
        rw [CoalgebraCat.ofComonObjCoalgebraStruct_comul]
        simp [-Mon_.monMonoidalStruct_tensorObj_X,
          ModuleCat.MonoidalCategory.instMonoidalCategoryStruct_tensorHom_hom,
          ModuleCat.hom_comp, ModuleCat.of, ModuleCat.ofHom,
          ModuleCat.MonoidalCategory.tensorμ_eq_tensorTensorTensorComm] }


/-- The tensor product of two coalgebra morphisms as a coalgebra morphism. -/
noncomputable def map (f : M →ₗc[R] N) (g : P →ₗc[R] Q) :
    M ⊗[R] P →ₗc[R] N ⊗[R] Q where
  toLinearMap := _root_.TensorProduct.map f.toLinearMap g.toLinearMap
  counit_comp := by
    /-
      R M N P Q : Type u
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : AddCommGroup N
      inst✝⁹ : AddCommGroup P
      inst✝⁸ : AddCommGroup Q
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R P
      inst✝⁴ : Module R Q
      inst✝³ : Coalgebra R M
      inst✝² : Coalgebra R N
      inst✝¹ : Coalgebra R P
      inst✝ : Coalgebra R Q
      f : CoalgHom R M N
      g : CoalgHom R P Q
      ⊢ Eq (CoalgebraStruct.counit.comp (_root_.TensorProduct.map f.toLinearMap g.to …
    -/
    simp_rw [← tensorHom_toLinearMap]
    /-
      R M N P Q : Type u
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : AddCommGroup N
      inst✝⁹ : AddCommGroup P
      inst✝⁸ : AddCommGroup Q
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R P
      inst✝⁴ : Module R Q
      inst✝³ : Coalgebra R M
      inst✝² : Coalgebra R N
      inst✝¹ : Coalgebra R P
      inst✝ : Coalgebra R Q
      f : CoalgHom R M N
      g : CoalgHom R P Q
      ⊢ Eq (CoalgebraStruct.counit.comp (CategoryTheory.MonoidalCategoryStruct.tenso …
    -/
    apply (CoalgebraCat.ofHom f ⊗ CoalgebraCat.ofHom g).1.counit_comp
    /-
      🎉 no goals
    -/
  map_comp_comul := by
    /-
      R M N P Q : Type u
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : AddCommGroup N
      inst✝⁹ : AddCommGroup P
      inst✝⁸ : AddCommGroup Q
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R P
      inst✝⁴ : Module R Q
      inst✝³ : Coalgebra R M
      inst✝² : Coalgebra R N
      inst✝¹ : Coalgebra R P
      inst✝ : Coalgebra R Q
      f : CoalgHom R M N
      g : CoalgHom R P Q
      ⊢ Eq ((_root_.TensorProduct.map (_root_.TensorProduct.map f.toLinearMap g.toLi …
    -/
    simp_rw [← tensorHom_toLinearMap, ← comul_tensorObj]
    /-
      R M N P Q : Type u
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : AddCommGroup N
      inst✝⁹ : AddCommGroup P
      inst✝⁸ : AddCommGroup Q
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module R P
      inst✝⁴ : Module R Q
      inst✝³ : Coalgebra R M
      inst✝² : Coalgebra R N
      inst✝¹ : Coalgebra R P
      inst✝ : Coalgebra R Q
      f : CoalgHom R M N
      g : CoalgHom R P Q
      ⊢ Eq ((_root_.TensorProduct.map (CategoryTheory.MonoidalCategoryStruct.tensorH …
    -/
    apply (CoalgebraCat.ofHom f ⊗ CoalgebraCat.ofHom g).1.map_comp_comul
    /-
      🎉 no goals
    -/


@[simp]
theorem map_tmul (f : M →ₗc[R] N) (g : P →ₗc[R] Q) (x : M) (y : P) :
    map f g (x ⊗ₜ y) = f x ⊗ₜ g y :=
  rfl


@[simp]
theorem map_toLinearMap (f : M →ₗc[R] N) (g : P →ₗc[R] Q) :
    map f g = _root_.TensorProduct.map (f : M →ₗ[R] N) (g : P →ₗ[R] Q) := rfl


/-- The associator for tensor products of R-coalgebras, as a coalgebra equivalence. -/
protected noncomputable def assoc :
    (M ⊗[R] N) ⊗[R] P ≃ₗc[R] M ⊗[R] (N ⊗[R] P) :=
  { _root_.TensorProduct.assoc R M N P with
    counit_comp := by
      simp_rw [← associator_hom_toLinearMap, ← counit_tensorObj_tensorObj_right,
        ← counit_tensorObj_tensorObj_left]
      apply CoalgHom.counit_comp (α_ (CoalgebraCat.of R M) (CoalgebraCat.of R N)
        (CoalgebraCat.of R P)).hom.1
    map_comp_comul := by
      simp_rw [← associator_hom_toLinearMap, ← comul_tensorObj_tensorObj_left,
        ← comul_tensorObj_tensorObj_right]
      exact CoalgHom.map_comp_comul (α_ (CoalgebraCat.of R M)
        (CoalgebraCat.of R N) (CoalgebraCat.of R P)).hom.1 }


@[simp]
theorem assoc_tmul (x : M) (y : N) (z : P) :
    Coalgebra.TensorProduct.assoc R M N P ((x ⊗ₜ y) ⊗ₜ z) = x ⊗ₜ (y ⊗ₜ z) :=
  rfl


@[simp]
theorem assoc_symm_tmul (x : M) (y : N) (z : P) :
    (Coalgebra.TensorProduct.assoc R M N P).symm (x ⊗ₜ (y ⊗ₜ z)) = (x ⊗ₜ y) ⊗ₜ z :=
  rfl


@[simp]
theorem assoc_toLinearEquiv :
    Coalgebra.TensorProduct.assoc R M N P = _root_.TensorProduct.assoc R M N P := rfl


/-- The base ring is a left identity for the tensor product of coalgebras, up to
coalgebra equivalence. -/
protected noncomputable def lid : R ⊗[R] M ≃ₗc[R] M :=
  { _root_.TensorProduct.lid R M with
    counit_comp := by
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq (CoalgebraStruct.counit.comp ↑__src✝) CoalgebraStruct.counit
      -/
      simp only [← leftUnitor_hom_toLinearMap]
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq (CoalgebraStruct.counit.comp (CategoryTheory.MonoidalCategoryStruct.leftU …
      -/
      apply CoalgHom.counit_comp (λ_ (CoalgebraCat.of R M)).hom.1
      /-
        🎉 no goals
      -/
    map_comp_comul := by
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq ((_root_.TensorProduct.map ↑__src✝ ↑__src✝).comp CoalgebraStruct.comul) ( …
      -/
      simp_rw [← leftUnitor_hom_toLinearMap, ← comul_tensorObj]
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq ((_root_.TensorProduct.map (CategoryTheory.MonoidalCategoryStruct.leftUni …
      -/
      apply CoalgHom.map_comp_comul (λ_ (CoalgebraCat.of R M)).hom.1 }
      /-
        🎉 no goals
      -/


@[simp]
theorem lid_toLinearEquiv :
    (Coalgebra.TensorProduct.lid R M) = _root_.TensorProduct.lid R M := rfl


@[simp]
theorem lid_tmul (r : R) (a : M) : Coalgebra.TensorProduct.lid R M (r ⊗ₜ a) = r • a := rfl


@[simp]
theorem lid_symm_apply (a : M) : (Coalgebra.TensorProduct.lid R M).symm a = 1 ⊗ₜ a := rfl


/-- The base ring is a right identity for the tensor product of coalgebras, up to
coalgebra equivalence. -/
protected noncomputable def rid : M ⊗[R] R ≃ₗc[R] M :=
  { _root_.TensorProduct.rid R M with
    counit_comp := by
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq (CoalgebraStruct.counit.comp ↑__src✝) CoalgebraStruct.counit
      -/
      simp only [← rightUnitor_hom_toLinearMap]
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq (CoalgebraStruct.counit.comp (CategoryTheory.MonoidalCategoryStruct.right …
      -/
      apply CoalgHom.counit_comp (ρ_ (CoalgebraCat.of R M)).hom.1
      /-
        🎉 no goals
      -/
    map_comp_comul := by
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq ((_root_.TensorProduct.map ↑__src✝ ↑__src✝).comp CoalgebraStruct.comul) ( …
      -/
      simp_rw [← rightUnitor_hom_toLinearMap, ← comul_tensorObj]
      /-
        R M N P Q : Type u
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : AddCommGroup N
        inst✝⁹ : AddCommGroup P
        inst✝⁸ : AddCommGroup Q
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module R P
        inst✝⁴ : Module R Q
        inst✝³ : Coalgebra R M
        inst✝² : Coalgebra R N
        inst✝¹ : Coalgebra R P
        inst✝ : Coalgebra R Q
        ⊢ Eq ((_root_.TensorProduct.map (CategoryTheory.MonoidalCategoryStruct.rightUn …
      -/
      apply CoalgHom.map_comp_comul (ρ_ (CoalgebraCat.of R M)).hom.1 }
      /-
        🎉 no goals
      -/


@[simp]
theorem rid_toLinearEquiv :
    (Coalgebra.TensorProduct.rid R M) = _root_.TensorProduct.rid R M := rfl


@[simp]
theorem rid_tmul (r : R) (a : M) : Coalgebra.TensorProduct.rid R M (a ⊗ₜ r) = r • a := rfl


@[simp]
theorem rid_symm_apply (a : M) : (Coalgebra.TensorProduct.rid R M).symm a = a ⊗ₜ 1 := rfl


/-- `lTensor M f : M ⊗ N →ₗc M ⊗ P` is the natural coalgebra morphism induced by `f : N →ₗc P`. -/
noncomputable abbrev lTensor (f : N →ₗc[R] P) : M ⊗[R] N →ₗc[R] M ⊗[R] P :=
  Coalgebra.TensorProduct.map (CoalgHom.id R M) f


/-- `rTensor M f : N ⊗ M →ₗc P ⊗ M` is the natural coalgebra morphism induced by `f : N →ₗc P`. -/
noncomputable abbrev rTensor (f : N →ₗc[R] P) : N ⊗[R] M →ₗc[R] P ⊗[R] M :=
  Coalgebra.TensorProduct.map f (CoalgHom.id R M)


