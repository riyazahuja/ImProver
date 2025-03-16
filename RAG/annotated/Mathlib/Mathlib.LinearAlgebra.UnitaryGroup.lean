/-- `Matrix.unitaryGroup n` is the group of `n` by `n` matrices where the star-transpose is the
inverse.
-/
abbrev unitaryGroup :=
  unitary (Matrix n n α)


theorem mem_unitaryGroup_iff : A ∈ Matrix.unitaryGroup n α ↔ A * star A = 1 := by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    ⊢ Iff (Membership.mem (Matrix.unitaryGroup n α) A) (Eq (HMul.hMul A (Star.star …
  -/
  refine ⟨And.right, fun hA => ⟨?_, hA⟩⟩
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    hA : Eq (HMul.hMul A (Star.star A)) 1
    ⊢ Eq (HMul.hMul (Star.star A) A) 1
  -/
  simpa only [mul_eq_one_comm] using hA
  /-
    🎉 no goals
  -/


theorem mem_unitaryGroup_iff' : A ∈ Matrix.unitaryGroup n α ↔ star A * A = 1 := by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    ⊢ Iff (Membership.mem (Matrix.unitaryGroup n α) A) (Eq (HMul.hMul (Star.star A …
  -/
  refine ⟨And.left, fun hA => ⟨hA, ?_⟩⟩
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    hA : Eq (HMul.hMul (Star.star A) A) 1
    ⊢ Eq (HMul.hMul A (Star.star A)) 1
  -/
  rwa [mul_eq_one_comm] at hA
  /-
    🎉 no goals
  -/


theorem det_of_mem_unitary {A : Matrix n n α} (hA : A ∈ Matrix.unitaryGroup n α) :
    A.det ∈ unitary α := by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    hA : Membership.mem (Matrix.unitaryGroup n α) A
    ⊢ Membership.mem (unitary α) A.det
  -/
  constructor
    /-
      case left
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      α : Type v
      inst✝¹ : CommRing α
      inst✝ : StarRing α
      A : Matrix n n α
      hA : Membership.mem (Matrix.unitaryGroup n α) A
      ⊢ Eq (HMul.hMul (Star.star A.det) A.det) 1
    -/
  · simpa [star, det_transpose] using congr_arg det hA.1
    /-
      🎉 no goals
    -/
    /-
      case right
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      α : Type v
      inst✝¹ : CommRing α
      inst✝ : StarRing α
      A : Matrix n n α
      hA : Membership.mem (Matrix.unitaryGroup n α) A
      ⊢ Eq (HMul.hMul A.det (Star.star A.det)) 1
    -/
  · simpa [star, det_transpose] using congr_arg det hA.2
    /-
      🎉 no goals
    -/


instance coeMatrix : Coe (unitaryGroup n α) (Matrix n n α) :=
  ⟨Subtype.val⟩


instance coeFun : CoeFun (unitaryGroup n α) fun _ => n → n → α where coe A := A.val


/-- `Matrix.UnitaryGroup.toLin' A` is matrix multiplication of vectors by `A`, as a linear map.

After the group structure on `Matrix.unitaryGroup n` is defined, we show in
`Matrix.UnitaryGroup.toLinearEquiv` that this gives a linear equivalence.
-/
def toLin' (A : unitaryGroup n α) :=
  Matrix.toLin' A.1


theorem ext_iff (A B : unitaryGroup n α) : A = B ↔ ∀ i j, A i j = B i j :=
  Subtype.ext_iff_val.trans ⟨fun h i j => congr_fun (congr_fun h i) j, Matrix.ext⟩


@[ext]
theorem ext (A B : unitaryGroup n α) : (∀ i j, A i j = B i j) → A = B :=
  (UnitaryGroup.ext_iff A B).mpr


theorem star_mul_self (A : unitaryGroup n α) : star A.1 * A.1 = 1 :=
  A.2.1


@[simp]
theorem det_isUnit (A : unitaryGroup n α) : IsUnit (A : Matrix n n α).det :=
  isUnit_iff_isUnit_det _ |>.mp <| (unitary.toUnits A).isUnit


@[simp] theorem inv_val : ↑A⁻¹ = (star A : Matrix n n α) := rfl


@[simp] theorem inv_apply : ⇑A⁻¹ = (star A : Matrix n n α) := rfl


@[simp] theorem mul_val : ↑(A * B) = A.1 * B.1 := rfl


@[simp] theorem mul_apply : ⇑(A * B) = A.1 * B.1 := rfl


@[simp] theorem one_val : ↑(1 : unitaryGroup n α) = (1 : Matrix n n α) := rfl


@[simp] theorem one_apply : ⇑(1 : unitaryGroup n α) = (1 : Matrix n n α) := rfl


@[simp]
theorem toLin'_mul : toLin' (A * B) = (toLin' A).comp (toLin' B) :=
  Matrix.toLin'_mul A.1 B.1


@[simp]
theorem toLin'_one : toLin' (1 : unitaryGroup n α) = LinearMap.id :=
  Matrix.toLin'_one


/-- `Matrix.unitaryGroup.toLinearEquiv A` is matrix multiplication of vectors by `A`, as a linear
equivalence. -/
def toLinearEquiv (A : unitaryGroup n α) : (n → α) ≃ₗ[α] n → α :=
  { Matrix.toLin' A.1 with
    invFun := toLin' A⁻¹
    left_inv := fun x =>
      calc
                                                                    /-
                                                                      n : Type u
                                                                      inst✝³ : DecidableEq n
                                                                      inst✝² : Fintype n
                                                                      α : Type v
                                                                      inst✝¹ : CommRing α
                                                                      inst✝ : StarRing α
                                                                      A✝ : Matrix n n α
                                                                      A : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
                                                                      x : n → α
                                                                      ⊢ Eq (((Matrix.UnitaryGroup.toLin' (Inv.inv A)).comp (Matrix.UnitaryGroup.toLi …
                                                                    -/
        (toLin' A⁻¹).comp (toLin' A) x = (toLin' (A⁻¹ * A)) x := by rw [← toLin'_mul]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                    /-
                      n : Type u
                      inst✝³ : DecidableEq n
                      inst✝² : Fintype n
                      α : Type v
                      inst✝¹ : CommRing α
                      inst✝ : StarRing α
                      A✝ : Matrix n n α
                      A : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
                      x : n → α
                      ⊢ Eq ((Matrix.UnitaryGroup.toLin' (HMul.hMul (Inv.inv A) A)) x) x
                    -/
        _ = x := by rw [inv_mul_cancel, toLin'_one, id_apply]
                    /-
                      🎉 no goals
                    -/
    right_inv := fun x =>
      calc
                                                                  /-
                                                                    n : Type u
                                                                    inst✝³ : DecidableEq n
                                                                    inst✝² : Fintype n
                                                                    α : Type v
                                                                    inst✝¹ : CommRing α
                                                                    inst✝ : StarRing α
                                                                    A✝ : Matrix n n α
                                                                    A : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
                                                                    x : n → α
                                                                    ⊢ Eq (((Matrix.UnitaryGroup.toLin' A).comp (Matrix.UnitaryGroup.toLin' (Inv.in …
                                                                  -/
        (toLin' A).comp (toLin' A⁻¹) x = toLin' (A * A⁻¹) x := by rw [← toLin'_mul]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                    /-
                      n : Type u
                      inst✝³ : DecidableEq n
                      inst✝² : Fintype n
                      α : Type v
                      inst✝¹ : CommRing α
                      inst✝ : StarRing α
                      A✝ : Matrix n n α
                      A : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
                      x : n → α
                      ⊢ Eq ((Matrix.UnitaryGroup.toLin' (HMul.hMul A (Inv.inv A))) x) x
                    -/
        _ = x := by rw [mul_inv_cancel, toLin'_one, id_apply] }
                    /-
                      🎉 no goals
                    -/


/-- `Matrix.unitaryGroup.toGL` is the map from the unitary group to the general linear group -/
def toGL (A : unitaryGroup n α) : GeneralLinearGroup α (n → α) :=
  GeneralLinearGroup.ofLinearEquiv (toLinearEquiv A)


theorem coe_toGL (A : unitaryGroup n α) : (toGL A).1 = toLin' A := rfl


@[simp]
theorem toGL_one : toGL (1 : unitaryGroup n α) = 1 := Units.ext <| by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    ⊢ Eq ↑(Matrix.UnitaryGroup.toGL 1) ↑1
  -/
  simp only [coe_toGL, toLin'_one]
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    ⊢ Eq LinearMap.id ↑1
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toGL_mul (A B : unitaryGroup n α) : toGL (A * B) = toGL A * toGL B := Units.ext <| by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A B : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
    ⊢ Eq ↑(Matrix.UnitaryGroup.toGL (HMul.hMul A B)) ↑(HMul.hMul (Matrix.UnitaryGr …
  -/
  simp only [coe_toGL, toLin'_mul]
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    α : Type v
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A B : Subtype fun x => Membership.mem (Matrix.unitaryGroup n α) x
    ⊢ Eq ((Matrix.UnitaryGroup.toLin' A).comp (Matrix.UnitaryGroup.toLin' B)) ↑(HM …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Matrix.unitaryGroup.embeddingGL` is the embedding from `Matrix.unitaryGroup n α` to
`LinearMap.GeneralLinearGroup n α`. -/
def embeddingGL : unitaryGroup n α →* GeneralLinearGroup α (n → α) :=
  ⟨⟨fun A => toGL A, toGL_one⟩, toGL_mul⟩


/-- `Matrix.specialUnitaryGroup` is the group of unitary `n` by `n` matrices where the determinant
is 1. (This definition is only correct if 2 is invertible.)-/
abbrev specialUnitaryGroup := unitaryGroup n α ⊓ MonoidHom.mker detMonoidHom


theorem mem_specialUnitaryGroup_iff :
    A ∈ specialUnitaryGroup n α ↔ A ∈ unitaryGroup n α ∧ A.det = 1 :=
  Iff.rfl


/-- `Matrix.orthogonalGroup n` is the group of `n` by `n` matrices where the transpose is the
inverse. -/
abbrev orthogonalGroup := unitaryGroup n β


theorem mem_orthogonalGroup_iff {A : Matrix n n β} :
    A ∈ Matrix.orthogonalGroup n β ↔ A * star A = 1 := by
  /-
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    β : Type v
    inst✝ : CommRing β
    A : Matrix n n β
    ⊢ Iff (Membership.mem (Matrix.orthogonalGroup n β) A) (Eq (HMul.hMul A (Star.s …
  -/
  refine ⟨And.right, fun hA => ⟨?_, hA⟩⟩
  /-
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    β : Type v
    inst✝ : CommRing β
    A : Matrix n n β
    hA : Eq (HMul.hMul A (Star.star A)) 1
    ⊢ Eq (HMul.hMul (Star.star A) A) 1
  -/
  simpa only [mul_eq_one_comm] using hA
  /-
    🎉 no goals
  -/


theorem mem_orthogonalGroup_iff' {A : Matrix n n β} :
    A ∈ Matrix.orthogonalGroup n β ↔ star A * A = 1 := by
  /-
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    β : Type v
    inst✝ : CommRing β
    A : Matrix n n β
    ⊢ Iff (Membership.mem (Matrix.orthogonalGroup n β) A) (Eq (HMul.hMul (Star.sta …
  -/
  refine ⟨And.left, fun hA => ⟨hA, ?_⟩⟩
  /-
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    β : Type v
    inst✝ : CommRing β
    A : Matrix n n β
    hA : Eq (HMul.hMul (Star.star A) A) 1
    ⊢ Eq (HMul.hMul A (Star.star A)) 1
  -/
  rwa [mul_eq_one_comm] at hA
  /-
    🎉 no goals
  -/


/-- `Matrix.specialOrthogonalGroup n` is the group of orthogonal `n` by `n` where the determinant
is one. (This definition is only correct if 2 is invertible.)-/
abbrev specialOrthogonalGroup := specialUnitaryGroup n β


theorem mem_specialOrthogonalGroup_iff :
    A ∈ specialOrthogonalGroup n β ↔ A ∈ orthogonalGroup n β ∧ A.det = 1 :=
  Iff.rfl


