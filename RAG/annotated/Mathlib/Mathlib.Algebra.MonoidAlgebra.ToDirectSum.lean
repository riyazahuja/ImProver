/-- Interpret an `AddMonoidAlgebra` as a homogeneous `DirectSum`. -/
def AddMonoidAlgebra.toDirectSum [Semiring M] (f : AddMonoidAlgebra M ι) : ⨁ _ : ι, M :=
  Finsupp.toDFinsupp f


@[simp]
theorem AddMonoidAlgebra.toDirectSum_single (i : ι) (m : M) :
    AddMonoidAlgebra.toDirectSum (Finsupp.single i m) = DirectSum.of _ i m :=
  Finsupp.toDFinsupp_single i m


/-- Interpret a homogeneous `DirectSum` as an `AddMonoidAlgebra`. -/
def DirectSum.toAddMonoidAlgebra (f : ⨁ _ : ι, M) : AddMonoidAlgebra M ι :=
  DFinsupp.toFinsupp f


@[simp]
theorem DirectSum.toAddMonoidAlgebra_of (i : ι) (m : M) :
    (DirectSum.of _ i m : ⨁ _ : ι, M).toAddMonoidAlgebra = Finsupp.single i m :=
  DFinsupp.toFinsupp_single i m


@[simp]
theorem AddMonoidAlgebra.toDirectSum_toAddMonoidAlgebra (f : AddMonoidAlgebra M ι) :
    f.toDirectSum.toAddMonoidAlgebra = f :=
  Finsupp.toDFinsupp_toFinsupp f


@[simp]
theorem DirectSum.toAddMonoidAlgebra_toDirectSum (f : ⨁ _ : ι, M) :
    f.toAddMonoidAlgebra.toDirectSum = f :=
  (DFinsupp.toFinsupp_toDFinsupp (show Π₀ _ : ι, M from f) : _)


@[simp]
theorem toDirectSum_zero [Semiring M] : (0 : AddMonoidAlgebra M ι).toDirectSum = 0 :=
  Finsupp.toDFinsupp_zero


@[simp]
theorem toDirectSum_add [Semiring M] (f g : AddMonoidAlgebra M ι) :
    (f + g).toDirectSum = f.toDirectSum + g.toDirectSum :=
  Finsupp.toDFinsupp_add _ _


@[simp]
theorem toDirectSum_mul [DecidableEq ι] [AddMonoid ι] [Semiring M] (f g : AddMonoidAlgebra M ι) :
    (f * g).toDirectSum = f.toDirectSum * g.toDirectSum := by
  let to_hom : AddMonoidAlgebra M ι →+ ⨁ _ : ι, M :=
  { toFun := toDirectSum
    map_zero' := toDirectSum_zero
    map_add' := toDirectSum_add }
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    f g : AddMonoidAlgebra M ι
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    ⊢ Eq (HMul.hMul f g).toDirectSum (HMul.hMul f.toDirectSum g.toDirectSum)
  -/
  show to_hom (f * g) = to_hom f * to_hom g
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    f g : AddMonoidAlgebra M ι
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    ⊢ Eq (to_hom (HMul.hMul f g)) (HMul.hMul (to_hom f) (to_hom g))
  -/
  let _ : NonUnitalNonAssocSemiring (ι →₀ M) := AddMonoidAlgebra.nonUnitalNonAssocSemiring
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    f g : AddMonoidAlgebra M ι
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    ⊢ Eq (to_hom (HMul.hMul f g)) (HMul.hMul (to_hom f) (to_hom g))
  -/
  revert f g
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    ⊢ ∀ (f g : AddMonoidAlgebra M ι), Eq (to_hom (HMul.hMul f g)) (HMul.hMul (to_h …
  -/
  rw [AddMonoidHom.map_mul_iff]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): does not find `addHom_ext'`, was `ext (xi xv yi yv) : 4`
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    ⊢ Eq (AddMonoidHom.mul.compr₂ to_hom) ((AddMonoidHom.mul.comp to_hom).compl₂ t …
  -/
  refine Finsupp.addHom_ext' fun xi => AddMonoidHom.ext fun xv => ?_
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    ⊢ Eq (((AddMonoidHom.mul.compr₂ to_hom).comp (Finsupp.singleAddHom xi)) xv) (( …
  -/
  refine Finsupp.addHom_ext' fun yi => AddMonoidHom.ext fun yv => ?_
  dsimp only [AddMonoidHom.comp_apply, AddMonoidHom.compl₂_apply, AddMonoidHom.compr₂_apply,
    AddMonoidHom.mul_apply, Finsupp.singleAddHom_apply]
  -- This was not needed before https://github.com/leanprover/lean4/pull/2644
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq (((AddMonoidHom.mul.compr₂ to_hom) (Finsupp.single xi xv)) (Finsupp.singl …
  -/
  erw [AddMonoidHom.compl₂_apply]
  -- If we remove the next `rw`, the `erw` after it will complain (when we get an `erw` linter)
  -- that it could be a `rw`. But the `erw` and `rw` will rewrite different occurrences.
  -- So first get rid of the `rw`-able occurrences to force `erw` to do the expensive rewrite only.
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq (((AddMonoidHom.mul.compr₂ to_hom) (Finsupp.single xi xv)) (Finsupp.singl …
  -/
  rw [AddMonoidHom.coe_mk, AddMonoidHom.coe_mk]
  -- This was not needed before https://github.com/leanprover/lean4/pull/2644
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq (((AddMonoidHom.mul.compr₂ to_hom) (Finsupp.single xi xv)) (Finsupp.singl …
  -/
  erw [AddMonoidHom.coe_mk]
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq ({ toFun := Function.comp ⇑to_hom ⇑(AddMonoidHom.mul (Finsupp.single xi x …
  -/
  simp only [AddMonoidHom.coe_mk, ZeroHom.coe_mk, toDirectSum_single]
  -- This was not needed before https://github.com/leanprover/lean4/pull/2644
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq (Function.comp (⇑to_hom) (⇑(AddMonoidHom.mul (Finsupp.single xi xv))) (Fi …
  -/
  dsimp
  rw [AddMonoidAlgebra.single_mul_single, AddMonoidHom.coe_mk, AddMonoidHom.coe_mk, ZeroHom.coe_mk,
    AddMonoidAlgebra.toDirectSum_single]
  simp only [AddMonoidHom.coe_comp, AddMonoidHom.coe_mul, AddMonoidHom.coe_mk, ZeroHom.coe_mk,
    Function.comp_apply, toDirectSum_single, AddMonoidHom.id_apply, Finsupp.singleAddHom_apply,
    AddMonoidHom.coe_mulLeft]
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : Semiring M
    to_hom : AddMonoidHom (AddMonoidAlgebra M ι) (DirectSum ι fun x => M) := { toF …
    x✝ : NonUnitalNonAssocSemiring (Finsupp ι M) := AddMonoidAlgebra.nonUnitalNonA …
    xi : ι
    xv : M
    yi : ι
    yv : M
    ⊢ Eq ((DirectSum.of (fun i => M) (HAdd.hAdd xi yi)) (HMul.hMul xv yv)) (HMul.h …
  -/
  rw [DirectSum.of_mul_of, Mul.gMul_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toAddMonoidAlgebra_zero [Semiring M] [∀ m : M, Decidable (m ≠ 0)] :
    toAddMonoidAlgebra 0 = (0 : AddMonoidAlgebra M ι) :=
  DFinsupp.toFinsupp_zero


@[simp]
theorem toAddMonoidAlgebra_add [Semiring M] [∀ m : M, Decidable (m ≠ 0)] (f g : ⨁ _ : ι, M) :
    (f + g).toAddMonoidAlgebra = toAddMonoidAlgebra f + toAddMonoidAlgebra g :=
  DFinsupp.toFinsupp_add _ _


@[simp]
theorem toAddMonoidAlgebra_mul [AddMonoid ι] [Semiring M]
    [∀ m : M, Decidable (m ≠ 0)] (f g : ⨁ _ : ι, M) :
    (f * g).toAddMonoidAlgebra = toAddMonoidAlgebra f * toAddMonoidAlgebra g := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : DecidableEq ι
    inst✝² : AddMonoid ι
    inst✝¹ : Semiring M
    inst✝ : (m : M) → Decidable (Ne m 0)
    f g : DirectSum ι fun x => M
    ⊢ Eq (HMul.hMul f g).toAddMonoidAlgebra (HMul.hMul f.toAddMonoidAlgebra g.toAd …
  -/
  apply_fun AddMonoidAlgebra.toDirectSum
    /-
      ι : Type u_1
      M : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddMonoid ι
      inst✝¹ : Semiring M
      inst✝ : (m : M) → Decidable (Ne m 0)
      f g : DirectSum ι fun x => M
      ⊢ Eq (HMul.hMul f g).toAddMonoidAlgebra.toDirectSum (HMul.hMul f.toAddMonoidAl …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inj
      ι : Type u_1
      M : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddMonoid ι
      inst✝¹ : Semiring M
      inst✝ : (m : M) → Decidable (Ne m 0)
      f g : DirectSum ι fun x => M
      ⊢ Function.Injective AddMonoidAlgebra.toDirectSum
    -/
  · apply Function.LeftInverse.injective
    /-
      case inj.a
      ι : Type u_1
      M : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddMonoid ι
      inst✝¹ : Semiring M
      inst✝ : (m : M) → Decidable (Ne m 0)
      f g : DirectSum ι fun x => M
      ⊢ Function.LeftInverse ?inj.g AddMonoidAlgebra.toDirectSum
    -/
    apply AddMonoidAlgebra.toDirectSum_toAddMonoidAlgebra
    /-
      🎉 no goals
    -/


/-- `AddMonoidAlgebra.toDirectSum` and `DirectSum.toAddMonoidAlgebra` together form an
equiv. -/
@[simps (config := .asFn)]
def addMonoidAlgebraEquivDirectSum [DecidableEq ι] [Semiring M] [∀ m : M, Decidable (m ≠ 0)] :
    AddMonoidAlgebra M ι ≃ ⨁ _ : ι, M :=
  { finsuppEquivDFinsupp with
    toFun := AddMonoidAlgebra.toDirectSum
    invFun := DirectSum.toAddMonoidAlgebra }


/-- The additive version of `AddMonoidAlgebra.addMonoidAlgebraEquivDirectSum`. -/
@[simps (config := .asFn)]
def addMonoidAlgebraAddEquivDirectSum [DecidableEq ι] [Semiring M] [∀ m : M, Decidable (m ≠ 0)] :
    AddMonoidAlgebra M ι ≃+ ⨁ _ : ι, M :=
  { addMonoidAlgebraEquivDirectSum with
    toFun := AddMonoidAlgebra.toDirectSum
    invFun := DirectSum.toAddMonoidAlgebra
    map_add' := AddMonoidAlgebra.toDirectSum_add }


/-- The ring version of `AddMonoidAlgebra.addMonoidAlgebraEquivDirectSum`. -/
@[simps (config := .asFn)]
def addMonoidAlgebraRingEquivDirectSum [DecidableEq ι] [AddMonoid ι] [Semiring M]
    [∀ m : M, Decidable (m ≠ 0)] : AddMonoidAlgebra M ι ≃+* ⨁ _ : ι, M :=
  { (addMonoidAlgebraAddEquivDirectSum : AddMonoidAlgebra M ι ≃+ ⨁ _ : ι, M) with
    toFun := AddMonoidAlgebra.toDirectSum
    invFun := DirectSum.toAddMonoidAlgebra
    map_mul' := AddMonoidAlgebra.toDirectSum_mul }


/-- The algebra version of `AddMonoidAlgebra.addMonoidAlgebraEquivDirectSum`. -/
@[simps (config := .asFn)]
def addMonoidAlgebraAlgEquivDirectSum [DecidableEq ι] [AddMonoid ι] [CommSemiring R] [Semiring A]
    [Algebra R A] [∀ m : A, Decidable (m ≠ 0)] : AddMonoidAlgebra A ι ≃ₐ[R] ⨁ _ : ι, A :=
  { (addMonoidAlgebraRingEquivDirectSum : AddMonoidAlgebra A ι ≃+* ⨁ _ : ι, A) with
    toFun := AddMonoidAlgebra.toDirectSum
    invFun := DirectSum.toAddMonoidAlgebra
    commutes' := fun _r => AddMonoidAlgebra.toDirectSum_single _ _ }


